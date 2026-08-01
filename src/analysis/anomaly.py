"""
Anomaly Detector — cost spike, bloat regression, cache degradation alerts.

Splits the observation window into two equal halves and compares them.
All alerts are returned as Alert dataclass instances ready for the dashboard.
"""

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class Alert:
    alert_id: str
    severity: str          # "Critical" | "Warning" | "Info"
    alert_type: str        # COST_SPIKE | BLOAT_REGRESSION | CACHE_DEGRADATION | NEW_EXPENSIVE_WF
    workflow: str
    title: str
    description: str
    current_value: float
    baseline_value: float
    change_pct: float


SEVERITY_EMOJI = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}


def detect_alerts(
    obs_df: pd.DataFrame,
    bloat_df: pd.DataFrame,
    daily_cache: pd.DataFrame,
) -> list[Alert]:
    alerts: list[Alert] = []

    if obs_df.empty:
        return alerts

    obs_df = obs_df.copy()
    obs_df["timestamp"] = pd.to_datetime(obs_df["timestamp"])
    ts_min, ts_max = obs_df["timestamp"].min(), obs_df["timestamp"].max()
    midpoint = ts_min + (ts_max - ts_min) / 2

    early = obs_df[obs_df["timestamp"] <= midpoint]
    late  = obs_df[obs_df["timestamp"] >  midpoint]

    # ── 1. Cost spike per workflow ────────────────────────────────────
    if not early.empty and not late.empty:
        early_wf = early.groupby("workflow")["total_cost"].sum()
        late_wf  = late.groupby("workflow")["total_cost"].sum()
        all_wf   = early_wf.index.union(late_wf.index)

        for wf in all_wf:
            e_cost = float(early_wf.get(wf, 0))
            l_cost = float(late_wf.get(wf, 0))
            if e_cost < 0.001:
                continue
            ratio = l_cost / e_cost
            if ratio >= 3.0:
                sev = "Critical"
            elif ratio >= 2.0:
                sev = "Warning"
            else:
                continue

            alerts.append(Alert(
                alert_id=f"spike_{wf[:20]}",
                severity=sev,
                alert_type="COST_SPIKE",
                workflow=wf,
                title=f"Cost spike detected — {wf}",
                description=(
                    f"Cost increased {ratio:.1f}× in the second half of the period "
                    f"(${e_cost:.4f} → ${l_cost:.4f}). "
                    "Check for increased traffic, new expensive operations, or disabled caching."
                ),
                current_value=round(l_cost, 4),
                baseline_value=round(e_cost, 4),
                change_pct=round((ratio - 1) * 100, 1),
            ))

    # ── 2. Bloat regression ───────────────────────────────────────────
    if not bloat_df.empty and "session_id" in obs_df.columns:
        obs_df["_half"] = obs_df["timestamp"].apply(
            lambda t: "early" if t <= midpoint else "late"
        )
        session_half = (
            obs_df.dropna(subset=["session_id"])
            .groupby(["session_id"])["_half"]
            .first()
            .reset_index()
        )
        bloat_with_half = bloat_df.merge(
            session_half, left_on="session_id", right_on="session_id", how="left"
        )
        if "_half" in bloat_with_half.columns:
            half_avg = (
                bloat_with_half.groupby(["workflow", "_half"])["bloat_score"]
                .mean()
                .unstack(fill_value=0)
            )
            for wf in half_avg.index:
                e_score = float(half_avg.loc[wf].get("early", 0))
                l_score = float(half_avg.loc[wf].get("late", 0))
                if e_score < 1.0 or l_score == 0:
                    continue
                ratio = l_score / e_score
                if ratio >= 1.5:
                    alerts.append(Alert(
                        alert_id=f"bloat_reg_{wf[:20]}",
                        severity="Warning",
                        alert_type="BLOAT_REGRESSION",
                        workflow=wf,
                        title=f"Bloat regression — {wf}",
                        description=(
                            f"Context bloat score grew {ratio:.1f}× over the period "
                            f"({e_score:.2f}× → {l_score:.2f}×). "
                            "A recent code change may have re-enabled full history injection "
                            "or disabled a summarization step."
                        ),
                        current_value=round(l_score, 2),
                        baseline_value=round(e_score, 2),
                        change_pct=round((ratio - 1) * 100, 1),
                    ))

    # ── 3. Cache hit rate degradation ─────────────────────────────────
    if not daily_cache.empty and len(daily_cache) >= 4:
        daily_cache = daily_cache.sort_values("date")
        half = len(daily_cache) // 2
        early_rate = float(daily_cache["cache_hit_rate_pct"].iloc[:half].mean())
        late_rate  = float(daily_cache["cache_hit_rate_pct"].iloc[half:].mean())

        if early_rate > 5 and late_rate < early_rate * 0.8:
            drop = early_rate - late_rate
            alerts.append(Alert(
                alert_id="cache_degradation",
                severity="Warning",
                alert_type="CACHE_DEGRADATION",
                workflow="(all workflows)",
                title="Cache hit rate dropping",
                description=(
                    f"Cache hit rate fell from {early_rate:.1f}% to {late_rate:.1f}% "
                    f"(−{drop:.1f}pp). This may indicate a prompt structure change "
                    "that broke cache key consistency, or a new workflow bypassing caching."
                ),
                current_value=round(late_rate, 1),
                baseline_value=round(early_rate, 1),
                change_pct=round(((late_rate - early_rate) / early_rate) * 100, 1),
            ))

    # ── 4. New expensive workflow appeared ────────────────────────────
    if not early.empty and not late.empty:
        avg_wf_cost = float(obs_df.groupby("workflow")["total_cost"].sum().mean())
        early_wf_set = set(early["workflow"].dropna().unique())
        late_only = late[~late["workflow"].isin(early_wf_set)]
        if not late_only.empty:
            new_wf_costs = late_only.groupby("workflow")["total_cost"].sum()
            for wf, cost in new_wf_costs.items():
                if cost > avg_wf_cost * 2:
                    alerts.append(Alert(
                        alert_id=f"new_wf_{wf[:20]}",
                        severity="Info",
                        alert_type="NEW_EXPENSIVE_WF",
                        workflow=wf,
                        title=f"New high-cost workflow detected — {wf}",
                        description=(
                            f"Workflow '{wf}' appeared in the second half of the period "
                            f"with cost ${cost:.4f} — {cost/avg_wf_cost:.1f}× above average. "
                            "Review this workflow's token usage before it scales up."
                        ),
                        current_value=round(cost, 4),
                        baseline_value=round(avg_wf_cost, 4),
                        change_pct=round((cost / avg_wf_cost - 1) * 100, 1),
                    ))

    # Sort: Critical first, then Warning, then Info
    order = {"Critical": 0, "Warning": 1, "Info": 2}
    return sorted(alerts, key=lambda a: order.get(a.severity, 3))
