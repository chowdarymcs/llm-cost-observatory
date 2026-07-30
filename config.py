from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # Connector
    connector_mode: Literal["clickhouse", "api"] = Field("api", env="CONNECTOR_MODE")

    # Langfuse Cloud API
    langfuse_public_key: str = Field("", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", env="LANGFUSE_HOST")

    # ClickHouse
    clickhouse_host: str = Field("localhost", env="CLICKHOUSE_HOST")
    clickhouse_port: int = Field(8123, env="CLICKHOUSE_PORT")
    clickhouse_user: str = Field("default", env="CLICKHOUSE_USER")
    clickhouse_password: str = Field("", env="CLICKHOUSE_PASSWORD")
    clickhouse_database: str = Field("default", env="CLICKHOUSE_DATABASE")
    langfuse_project_id: str = Field("", env="LANGFUSE_PROJECT_ID")

    # Dashboard
    default_lookback_days: int = Field(30, env="DEFAULT_LOOKBACK_DAYS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Model pricing table (per 1M tokens — input / output / cache_read)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":        {"input": 3.00,   "output": 15.00,  "cache_read": 0.30},
    "claude-opus-4-6":          {"input": 15.00,  "output": 75.00,  "cache_read": 1.50},
    "claude-haiku-4-5":         {"input": 0.80,   "output": 4.00,   "cache_read": 0.08},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00,  "cache_read": 0.30},
    "claude-3-5-haiku-20241022":  {"input": 0.80, "output": 4.00,   "cache_read": 0.08},
    "gpt-4o":                   {"input": 2.50,   "output": 10.00,  "cache_read": 1.25},
    "gpt-4o-mini":              {"input": 0.15,   "output": 0.60,   "cache_read": 0.075},
    "gpt-4-turbo":              {"input": 10.00,  "output": 30.00,  "cache_read": 5.00},
    "gpt-3.5-turbo":            {"input": 0.50,   "output": 1.50,   "cache_read": 0.25},
    "gemini-1.5-pro":           {"input": 1.25,   "output": 5.00,   "cache_read": 0.3125},
    "gemini-1.5-flash":         {"input": 0.075,  "output": 0.30,   "cache_read": 0.01875},
}

# Fallback if model not in pricing table
DEFAULT_PRICING = {"input": 2.00, "output": 8.00, "cache_read": 0.50}


def get_settings() -> Settings:
    return Settings()
