"""
Context bloat detector.

Core insight: in a healthy agentic session each turn's input_tokens should
grow only by the prior output + any new tool results. Unbounded growth
(compounding re-injection of full history / verbose tool outputs) is waste.

Bloat score = actual_input_tokens / expected_input_tokens
where expected_input_tokens = cumulative_output_tokens + system_prompt_estimate

A score > 2.0 signals significant bloat; > 5.0 is severe.
"""

import pandas as pd
import numpy as np

SYSTEM_PROMPT_ESTIMATE = 500       # conservative baseline tokens per call
BLOAT_THRESHOLD_MODERATE = 2.0
BLOAT_THRESHOLD_SEVERE = 5.0


def assign_turn_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add turn_index (0-based position within a session) to observations.
    Falls back to trace-level ordering when session_id is null.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    group_col = "session_id" if df["session_id"].notna().any() else "trace_id"
    df["turn_index"] = df.groupby(group_col).cumcount()
    df["_group"] = df[group_col]
    return df


def compute_bloat_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-session bloat metrics:
        session_id / trace_id, turns, peak_input_tokens,
        cumulative_output_tokens, bloat_score, waste_tokens, waste_usd
    """
    if df.empty:
        return pd.DataFrame()

    df = assign_turn_index(df)
    group_col = "_group"

    sessions = []
    for gid, grp in df.groupby(group_col):
        grp = grp.sort_values("turn_index")
        turns = len(grp)
        if turns < 2:
            continue

        cumulative_output = grp["output_tokens"].cumsum().shift(1).fillna(0)
        # Expected input = all prior outputs + system prompt per turn
        expected = cumulative_output + SYSTEM_PROMPT_ESTIMATE

        # Actual final-turn input (worst case, where bloat is highest)
        last = grp.iloc[-1]
        last_expected = expected.iloc[-1]
        bloat_score = last["input_tokens"] / max(last_expected, 1)

        # Waste = tokens paid for above expected across all turns
        waste_tokens = max(0, int((grp["input_tokens"] - expected).clip(lower=0).sum()))

        # Estimate waste cost using a blended rate from model pricing
        from config import MODEL_PRICING, DEFAULT_PRICING
        pricing = MODEL_PRICING.get(grp["model"].mode()[0] if not grp["model"].empty else "", DEFAULT_PRICING)
        waste_usd = (waste_tokens / 1_000_000) * pricing["input"]

        sessions.append({
            "session_id":              gid,
            "workflow":                grp["workflow"].mode()[0] if not grp["workflow"].empty else "",
            "turns":                   turns,
            "peak_input_tokens":       int(grp["input_tokens"].max()),
            "cumulative_output_tokens":int(grp["output_tokens"].sum()),
            "bloat_score":             round(bloat_score, 2),
            "waste_tokens":            waste_tokens,
            "waste_usd":               round(waste_usd, 4),
            "severity":                _severity(bloat_score),
        })

    result = pd.DataFrame(sessions).sort_values("bloat_score", ascending=False)
    return result


def session_token_growth(df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """
    Token growth chart data for a single session —
    returns turn-by-turn input_tokens, output_tokens, and expected input.
    """
    df = assign_turn_index(df)
    grp = df[df["_group"] == session_id].sort_values("turn_index")
    if grp.empty:
        return pd.DataFrame()

    cumulative_output = grp["output_tokens"].cumsum().shift(1).fillna(0)
    grp = grp.copy()
    grp["expected_input_tokens"] = (cumulative_output + SYSTEM_PROMPT_ESTIMATE).astype(int)
    grp["excess_tokens"] = (grp["input_tokens"] - grp["expected_input_tokens"]).clip(lower=0).astype(int)
    return grp[["turn_index", "input_tokens", "output_tokens",
                "expected_input_tokens", "excess_tokens", "model", "workflow"]].reset_index(drop=True)


def bloat_summary_stats(bloat_df: pd.DataFrame) -> dict:
    """Aggregate KPIs for the bloat detection overview card."""
    if bloat_df.empty:
        return {"total_waste_usd": 0.0, "severe_sessions": 0,
                "moderate_sessions": 0, "avg_bloat_score": 0.0}
    return {
        "total_waste_usd":   round(bloat_df["waste_usd"].sum(), 2),
        "severe_sessions":   int((bloat_df["severity"] == "Severe").sum()),
        "moderate_sessions": int((bloat_df["severity"] == "Moderate").sum()),
        "avg_bloat_score":   round(bloat_df["bloat_score"].mean(), 2),
    }


def _severity(score: float) -> str:
    if score >= BLOAT_THRESHOLD_SEVERE:
        return "Severe"
    if score >= BLOAT_THRESHOLD_MODERATE:
        return "Moderate"
    return "Healthy"
