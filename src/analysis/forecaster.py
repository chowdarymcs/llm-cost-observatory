"""
Savings Forecaster.

Projects current 30-day spend and the optimised scenario (implementing
all recommendations). Returns data for the waterfall chart and KPI cards.
"""

import pandas as pd
import numpy as np


def build_forecast(
    obs_df: pd.DataFrame,
    recommendations: list,
    date_range_days: int,
) -> dict:
    """
    Returns a dict with:
        daily_cost              — average daily spend in selected range
        monthly_projection      — extrapolated 30-day cost at current pace
        optimized_monthly       — projected cost after all recommendations
        total_monthly_savings   — sum of all recommendation savings
        savings_pct             — percentage reduction
        waterfall_data          — list of dicts for waterfall chart
        by_effort               — savings grouped by effort level
    """
    days = max(date_range_days, 1)

    if obs_df.empty:
        return {
            "daily_cost": 0, "monthly_projection": 0,
            "optimized_monthly": 0, "total_monthly_savings": 0,
            "savings_pct": 0, "waterfall_data": [], "by_effort": {},
        }

    total_cost         = obs_df["total_cost"].sum()
    daily_cost         = total_cost / days
    monthly_projection = daily_cost * 30

    # Deduplicate recs by pattern (take highest savings per pattern)
    seen_patterns: dict[str, float] = {}
    for r in recommendations:
        if r.pattern not in seen_patterns or r.monthly_savings_usd > seen_patterns[r.pattern]:
            seen_patterns[r.pattern] = r.monthly_savings_usd

    # Build waterfall items — sorted by savings descending
    waterfall: list[dict] = [{"label": "Current (30-day)", "value": monthly_projection, "type": "total"}]
    running = monthly_projection

    sorted_recs = sorted(recommendations, key=lambda r: r.monthly_savings_usd, reverse=True)
    added_patterns: set[str] = set()
    for r in sorted_recs:
        if r.pattern in added_patterns:
            continue
        added_patterns.add(r.pattern)
        saving = min(r.monthly_savings_usd, running * 0.50)  # cap at 50% of remaining to stay realistic
        running -= saving
        waterfall.append({
            "label": r.title,
            "value": -saving,
            "type": "saving",
            "effort": r.effort,
            "pattern": r.pattern,
        })

    waterfall.append({"label": "Optimised (30-day)", "value": running, "type": "total"})

    total_savings = monthly_projection - running
    savings_pct   = (total_savings / monthly_projection * 100) if monthly_projection > 0 else 0

    # Group savings by effort level
    by_effort: dict[str, float] = {}
    for item in waterfall[1:-1]:
        effort = item.get("effort", "Low")
        by_effort[effort] = by_effort.get(effort, 0) + abs(item["value"])

    return {
        "daily_cost":           round(daily_cost, 4),
        "monthly_projection":   round(monthly_projection, 4),
        "optimized_monthly":    round(running, 4),
        "total_monthly_savings":round(total_savings, 4),
        "savings_pct":          round(savings_pct, 1),
        "waterfall_data":       waterfall,
        "by_effort":            {k: round(v, 4) for k, v in by_effort.items()},
    }
