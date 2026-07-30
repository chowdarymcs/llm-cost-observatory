"""
Cache hit rate and prompt-caching savings analyzer.

Quantifies: how much you saved via prompt caching, which workflows
benefit most, and what the before/after cost comparison looks like
if cache were disabled.
"""

import pandas as pd


def cache_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"hit_rate": 0.0, "total_savings_usd": 0.0,
                "cached_tokens": 0, "uncached_equiv_cost": 0.0}

    total_input = df["input_tokens"].sum() + df["cache_read_tokens"].sum()
    cached = df["cache_read_tokens"].sum()
    hit_rate = cached / total_input if total_input > 0 else 0.0

    # Savings = difference between paying full input price vs cache_read price
    savings = (df["cache_read_cost"] * _full_vs_cache_ratio(df)).sum()

    # What would cost have been if no caching existed
    from config import MODEL_PRICING, DEFAULT_PRICING
    uncached_extra = df.apply(
        lambda r: (r["cache_read_tokens"] / 1_000_000)
                  * MODEL_PRICING.get(r["model"] or "", DEFAULT_PRICING)["input"],
        axis=1,
    ).sum()

    return {
        "hit_rate":           round(hit_rate * 100, 1),   # percentage
        "total_savings_usd":  round(uncached_extra - df["cache_read_cost"].sum(), 4),
        "cached_tokens":      int(cached),
        "uncached_equiv_cost": round(df["total_cost"].sum() + uncached_extra - df["cache_read_cost"].sum(), 4),
    }


def cache_by_workflow(df: pd.DataFrame) -> pd.DataFrame:
    """Per-workflow cache hit rate and savings."""
    if df.empty:
        return pd.DataFrame()

    def _stats(g):
        total_inp = g["input_tokens"].sum() + g["cache_read_tokens"].sum()
        cached = g["cache_read_tokens"].sum()
        return pd.Series({
            "cache_hit_rate_pct": round(100 * cached / total_inp, 1) if total_inp else 0,
            "cached_tokens":      int(cached),
            "cache_read_cost":    g["cache_read_cost"].sum(),
            "total_cost":         g["total_cost"].sum(),
            "generation_count":   len(g),
        })

    return (
        df.groupby("workflow")
        .apply(_stats)
        .reset_index()
        .sort_values("cache_hit_rate_pct", ascending=False)
    )


def daily_cache_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Day-by-day cache hit rate and cumulative savings."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date").agg(
        total_input_tokens=("input_tokens", "sum"),
        cache_read_tokens=("cache_read_tokens", "sum"),
        cache_read_cost=("cache_read_cost", "sum"),
    ).reset_index()

    daily["cache_hit_rate_pct"] = (
        100 * daily["cache_read_tokens"]
        / (daily["total_input_tokens"] + daily["cache_read_tokens"]).replace(0, 1)
    ).round(1)

    return daily.sort_values("date")


def before_after_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side cost table: actual cost vs. hypothetical cost without caching.
    Returns one row per model.
    """
    if df.empty:
        return pd.DataFrame()

    from config import MODEL_PRICING, DEFAULT_PRICING
    rows = []
    for model, g in df.groupby("model"):
        pricing = MODEL_PRICING.get(model or "", DEFAULT_PRICING)
        cached_tokens = g["cache_read_tokens"].sum()
        actual_cost = g["total_cost"].sum()
        without_cache_cost = actual_cost - g["cache_read_cost"].sum() + \
                             (cached_tokens / 1_000_000) * pricing["input"]
        rows.append({
            "model":             model,
            "actual_cost_usd":   round(actual_cost, 4),
            "without_cache_usd": round(without_cache_cost, 4),
            "savings_usd":       round(without_cache_cost - actual_cost, 4),
            "savings_pct":       round(100 * (without_cache_cost - actual_cost)
                                       / without_cache_cost, 1) if without_cache_cost else 0,
        })
    return pd.DataFrame(rows).sort_values("savings_usd", ascending=False)


def _full_vs_cache_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of full input price to cache_read price per row (for savings calc)."""
    from config import MODEL_PRICING, DEFAULT_PRICING
    return df["model"].apply(
        lambda m: MODEL_PRICING.get(m or "", DEFAULT_PRICING)["input"]
                  / MODEL_PRICING.get(m or "", DEFAULT_PRICING)["cache_read"]
    )
