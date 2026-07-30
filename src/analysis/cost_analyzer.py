"""
Cost analysis: per-workflow, per-model, per-session, and daily trend breakdowns.
All functions accept a normalized observations DataFrame and return summary DataFrames
ready for Plotly / Streamlit rendering.
"""

import pandas as pd
import numpy as np


def cost_kpis(df: pd.DataFrame) -> dict:
    """Top-level KPIs for the overview page."""
    if df.empty:
        return {"total_cost": 0.0, "total_traces": 0, "avg_cost_per_trace": 0.0,
                "total_tokens": 0, "p95_cost_per_trace": 0.0}

    per_trace = df.groupby("trace_id")["total_cost"].sum()
    return {
        "total_cost":          df["total_cost"].sum(),
        "total_traces":        df["trace_id"].nunique(),
        "avg_cost_per_trace":  per_trace.mean(),
        "p95_cost_per_trace":  per_trace.quantile(0.95),
        "total_tokens":        int(df["total_tokens"].sum()),
    }


def daily_cost_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregated cost, token usage, and generation count."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return (
        df.groupby("date")
        .agg(
            total_cost=("total_cost", "sum"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            cache_read_tokens=("cache_read_tokens", "sum"),
            generation_count=("id", "count"),
        )
        .reset_index()
        .sort_values("date")
    )


def cost_by_workflow(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Total cost and generation count per workflow (trace name)."""
    if df.empty:
        return pd.DataFrame()
    result = (
        df.groupby("workflow")
        .agg(
            total_cost=("total_cost", "sum"),
            avg_cost=("total_cost", "mean"),
            generation_count=("id", "count"),
            trace_count=("trace_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
        .head(top_n)
    )
    return result


def cost_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Cost breakdown per model — useful for model routing decisions."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("model")
        .agg(
            total_cost=("total_cost", "sum"),
            input_cost=("input_cost", "sum"),
            output_cost=("output_cost", "sum"),
            cache_read_cost=("cache_read_cost", "sum"),
            generation_count=("id", "count"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
    )


def cost_per_trace_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trace cost for histogram / percentile analysis."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("trace_id")
        .agg(
            total_cost=("total_cost", "sum"),
            workflow=("workflow", "first"),
            session_id=("session_id", "first"),
            timestamp=("timestamp", "min"),
        )
        .reset_index()
        .sort_values("total_cost", ascending=False)
    )


def routing_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies workflows where the bulk of spend is on expensive models
    but average output is small — good candidates for model downgrading.
    """
    if df.empty:
        return pd.DataFrame()

    EXPENSIVE_MODELS = {"claude-opus-4-6", "gpt-4-turbo", "gpt-4o"}
    wf = df.groupby(["workflow", "model"]).agg(
        total_cost=("total_cost", "sum"),
        avg_output_tokens=("output_tokens", "mean"),
        count=("id", "count"),
    ).reset_index()

    expensive = wf[wf["model"].isin(EXPENSIVE_MODELS)].copy()
    expensive["flag"] = expensive["avg_output_tokens"] < 200
    return expensive.sort_values("total_cost", ascending=False)
