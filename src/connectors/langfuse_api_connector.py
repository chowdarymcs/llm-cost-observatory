"""
Langfuse Cloud API connector.

Uses the Langfuse REST API with basic-auth pagination to fetch traces
and observations. Works with both Langfuse Cloud and self-hosted instances
that expose the REST API.
"""

import logging
from datetime import datetime
from typing import Generator

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import MODEL_PRICING, DEFAULT_PRICING, get_settings
from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

PAGE_SIZE = 100


class LangfuseAPIConnector(BaseConnector):
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.langfuse_host.rstrip("/")
        self.auth = (settings.langfuse_public_key, settings.langfuse_secret_key)
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        return session

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        resp = self.session.get(url, auth=self.auth, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, endpoint: str, params: dict) -> Generator[dict, None, None]:
        page = 1
        while True:
            params["page"] = page
            params["limit"] = PAGE_SIZE
            data = self._get(endpoint, params)
            items = data.get("data", [])
            if not items:
                break
            yield from items
            meta = data.get("meta", {})
            if page >= meta.get("totalPages", 1):
                break
            page += 1

    def fetch_observations(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: str | None = None,
    ) -> pd.DataFrame:
        params = {
            "type": "GENERATION",
            "fromStartTime": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "toStartTime": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        rows = []
        for obs in self._paginate("/api/public/observations", params):
            usage = obs.get("usage", {}) or {}
            input_tokens = usage.get("input", 0) or 0
            output_tokens = usage.get("output", 0) or 0
            cache_tokens = usage.get("cacheReadInputTokens", 0) or 0
            total_tokens = usage.get("total", input_tokens + output_tokens) or 0
            model = obs.get("model") or ""
            pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

            input_cost = obs.get("calculatedInputCost") or (input_tokens / 1_000_000 * pricing["input"])
            output_cost = obs.get("calculatedOutputCost") or (output_tokens / 1_000_000 * pricing["output"])
            cache_cost = cache_tokens / 1_000_000 * pricing["cache_read"]
            total_cost = obs.get("calculatedTotalCost") or (input_cost + output_cost + cache_cost)

            rows.append({
                "id":               obs.get("id", ""),
                "trace_id":         obs.get("traceId", ""),
                "workflow":         obs.get("name", ""),
                "session_id":       None,          # joined from traces below
                "model":            model,
                "timestamp":        pd.to_datetime(obs.get("startTime")),
                "input_tokens":     int(input_tokens),
                "output_tokens":    int(output_tokens),
                "cache_read_tokens": int(cache_tokens),
                "total_tokens":     int(total_tokens),
                "input_cost":       float(input_cost),
                "output_cost":      float(output_cost),
                "cache_read_cost":  float(cache_cost),
                "total_cost":       float(total_cost),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Enrich with session_id and trace name (workflow) from traces
        traces_df = self.fetch_traces(start_date, end_date)
        if not traces_df.empty:
            trace_map = traces_df.set_index("trace_id")[["session_id", "trace_name"]]
            df = df.join(trace_map, on="trace_id", rsuffix="_trace")
            df["session_id"] = df.get("session_id_trace", df.get("session_id"))
            df["workflow"] = df["trace_name"].fillna(df["workflow"])
            df.drop(columns=["session_id_trace", "trace_name"], errors="ignore", inplace=True)

        return df

    def fetch_traces(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: str | None = None,
    ) -> pd.DataFrame:
        params = {
            "fromTimestamp": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "toTimestamp": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        rows = []
        for t in self._paginate("/api/public/traces", params):
            rows.append({
                "trace_id":   t.get("id", ""),
                "trace_name": t.get("name", ""),
                "session_id": t.get("sessionId"),
                "user_id":    t.get("userId"),
                "timestamp":  pd.to_datetime(t.get("timestamp")),
                "tags":       t.get("tags", []),
            })
        return pd.DataFrame(rows)
