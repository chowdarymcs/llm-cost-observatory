from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd


class BaseConnector(ABC):
    """
    Common interface for all data source connectors.
    Each connector must return normalized DataFrames so downstream
    analysis modules work identically regardless of data source.
    """

    @abstractmethod
    def fetch_observations(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of LLM generation observations with columns:
            id, trace_id, workflow, session_id, model, timestamp,
            input_tokens, output_tokens, cache_read_tokens, total_tokens,
            input_cost, output_cost, cache_read_cost, total_cost
        """
        ...

    @abstractmethod
    def fetch_traces(
        self,
        start_date: datetime,
        end_date: datetime,
        project_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of traces with columns:
            trace_id, trace_name, session_id, user_id, timestamp, tags
        """
        ...

    def health_check(self) -> bool:
        """Lightweight connectivity check. Override if needed."""
        try:
            from datetime import timedelta
            end = datetime.utcnow()
            start = end - timedelta(hours=1)
            df = self.fetch_traces(start, end)
            return isinstance(df, __import__("pandas").DataFrame)
        except Exception:
            return False
