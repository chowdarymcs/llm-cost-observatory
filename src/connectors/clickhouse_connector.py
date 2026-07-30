"""
ClickHouse connector for self-hosted Langfuse instances.

Handles both Langfuse v2 schema (separate prompt_tokens / completion_tokens columns)
and Langfuse v3 schema (usage_details / cost_details Map columns).
Schema version is auto-detected on first query.
"""

import logging
from datetime import datetime
from functools import lru_cache

import clickhouse_connect
import pandas as pd

from config import MODEL_PRICING, DEFAULT_PRICING, get_settings
from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class ClickHouseConnector(BaseConnector):
    def __init__(self):
        settings = get_settings()
        self.client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_database,
        )
        self.project_id = settings.langfuse_project_id
        self._schema_version = self._detect_schema()

    def _detect_schema(self) -> int:
        """Returns 3 if usage_details Map column exists, else 2."""
        try:
            cols = self.client.query(
                "SELECT name FROM system.columns "
                "WHERE table = 'observations' AND name = 'usage_details'"
            ).result_rows
            return 3 if cols else 2
        except Exception:
            return 2

    def _project_filter(self, alias: str = "o") -> str:
        if self.project_id:
            return f"AND {alias}.project_id = '{self.project_id}'"
        return ""

    def fetch_observations(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: str | None = None,
    ) -> pd.DataFrame:
        proj_filter = f"AND o.project_id = '{project_id}'" if project_id else self._project_filter()

        if self._schema_version == 3:
            token_cols = """
                toInt64OrZero(usage_details['input'])      AS input_tokens,
                toInt64OrZero(usage_details['output'])     AS output_tokens,
                toInt64OrZero(usage_details['cache_read_input_tokens']) AS cache_read_tokens,
                toInt64OrZero(usage_details['total'])      AS total_tokens,
                toFloat64OrZero(cost_details['input'])     AS input_cost,
                toFloat64OrZero(cost_details['output'])    AS output_cost,
                toFloat64OrZero(cost_details['total'])     AS total_cost
            """
        else:
            token_cols = """
                coalesce(prompt_tokens, 0)                 AS input_tokens,
                coalesce(completion_tokens, 0)             AS output_tokens,
                0                                          AS cache_read_tokens,
                coalesce(total_tokens, 0)                  AS total_tokens,
                coalesce(calculated_input_cost, 0)         AS input_cost,
                coalesce(calculated_output_cost, 0)        AS output_cost,
                coalesce(calculated_total_cost, 0)         AS total_cost
            """

        query = f"""
            SELECT
                o.id,
                o.trace_id,
                t.name           AS workflow,
                t.session_id,
                o.model,
                o.start_time     AS timestamp,
                {token_cols}
            FROM observations o
            LEFT JOIN traces t ON o.trace_id = t.id
            WHERE o.type = 'GENERATION'
              AND o.start_time >= '{start_date.strftime("%Y-%m-%d %H:%M:%S")}'
              AND o.start_time <  '{end_date.strftime("%Y-%m-%d %H:%M:%S")}'
              {proj_filter}
            ORDER BY o.start_time ASC
        """

        df = self.client.query_df(query)
        if df.empty:
            return self._empty_observations()

        # Recalculate cache_read_cost if schema v2 (not stored separately)
        if self._schema_version == 2:
            df["cache_read_cost"] = df.apply(
                lambda r: _cache_cost(r["model"], r["cache_read_tokens"]), axis=1
            )
        else:
            df["cache_read_cost"] = df["total_cost"] - df["input_cost"] - df["output_cost"]
            df["cache_read_cost"] = df["cache_read_cost"].clip(lower=0)

        return df

    def fetch_traces(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: str | None = None,
    ) -> pd.DataFrame:
        proj_filter = f"AND project_id = '{project_id}'" if project_id else self._project_filter("t")
        query = f"""
            SELECT
                id          AS trace_id,
                name        AS trace_name,
                session_id,
                user_id,
                timestamp,
                tags
            FROM traces t
            WHERE timestamp >= '{start_date.strftime("%Y-%m-%d %H:%M:%S")}'
              AND timestamp <  '{end_date.strftime("%Y-%m-%d %H:%M:%S")}'
              {proj_filter}
            ORDER BY timestamp ASC
        """
        return self.client.query_df(query)

    @staticmethod
    def _empty_observations() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "id", "trace_id", "workflow", "session_id", "model", "timestamp",
            "input_tokens", "output_tokens", "cache_read_tokens", "total_tokens",
            "input_cost", "output_cost", "cache_read_cost", "total_cost",
        ])


def _cache_cost(model: str, cache_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model or "", DEFAULT_PRICING)
    return (cache_tokens / 1_000_000) * pricing["cache_read"]
