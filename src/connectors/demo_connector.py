"""
Demo Connector — generates realistic synthetic Langfuse-style trace data.

Two scenarios:
    "clean"   — a well-optimised system: good cache hit rates, controlled context
                growth, appropriate model routing. Baseline for comparison.
    "anomaly" — a system with injected problems: runaway history accumulation,
                uncompressed tool outputs, RAG over-fetch, near-zero cache hits,
                an expensive model doing trivial work, plus a mid-period cost spike.

Produces the same normalized DataFrame shape as the ClickHouse and API connectors,
so every analysis module works unchanged.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import MODEL_PRICING, DEFAULT_PRICING
from src.connectors.base import BaseConnector


# ── Workflow definitions ────────────────────────────────────────────────
# (name, model, avg_turns, base_input, base_output, sessions_per_day, cache_ratio)
CLEAN_WORKFLOWS = [
    ("document-qa/retrieve",      "claude-haiku-4-5",  3,  1200,  180,  40, 0.72),
    ("document-qa/generate",      "claude-sonnet-4-6", 4,  2200,  650,  35, 0.68),
    ("support-agent/classify",    "claude-haiku-4-5",  1,   600,   90,  90, 0.80),
    ("support-agent/respond",     "claude-sonnet-4-6", 5,  1800,  520,  55, 0.65),
    ("data-extraction/parse",     "claude-haiku-4-5",  2,   900,  240,  70, 0.75),
    ("report-writer/draft",       "claude-sonnet-4-6", 6,  2800,  1400, 12, 0.60),
]

ANOMALY_WORKFLOWS = [
    # (name, model, avg_turns, base_input, base_output, sessions/day, cache_ratio, bloat_factor)
    ("document-qa/retrieve",      "claude-haiku-4-5",  3,  1200,  180,  40, 0.70, 1.0),
    ("document-qa/generate",      "claude-sonnet-4-6", 4,  2200,  650,  35, 0.62, 1.0),
    # PROBLEM 1: runaway history accumulation — multi-turn agent re-sending everything
    ("research-agent/investigate","claude-sonnet-4-6", 14, 3000,  700,  18, 0.05, 6.5),
    # PROBLEM 2: uncompressed tool outputs — high variance input spikes
    ("data-pipeline/execute",     "claude-sonnet-4-6", 8,  4500,  400,  25, 0.10, 2.8),
    # PROBLEM 3: RAG over-fetch — uniformly enormous input from turn 1
    ("knowledge-base/search",     "claude-sonnet-4-6", 2,  9500,  350,  30, 0.08, 1.0),
    # PROBLEM 4: expensive model doing trivial classification work
    ("triage/categorize",         "claude-opus-4-6",   1,   800,   85, 120, 0.12, 1.0),
    ("support-agent/classify",    "claude-haiku-4-5",  1,   600,   90,  90, 0.55, 1.0),
]


def _pricing(model: str) -> dict:
    return MODEL_PRICING.get(model, DEFAULT_PRICING)


def _build_observation(
    rng, obs_id, trace_id, session_id, workflow, model,
    turn_idx, input_tokens, output_tokens, cache_tokens, timestamp,
) -> dict:
    p = _pricing(model)
    input_cost      = (input_tokens  / 1_000_000) * p["input"]
    output_cost     = (output_tokens / 1_000_000) * p["output"]
    cache_read_cost = (cache_tokens  / 1_000_000) * p["cache_read"]
    return {
        "id":                obs_id,
        "trace_id":          trace_id,
        "workflow":          workflow,
        "session_id":        session_id,
        "model":             model,
        "timestamp":         timestamp,
        "input_tokens":      int(input_tokens),
        "output_tokens":     int(output_tokens),
        "cache_read_tokens": int(cache_tokens),
        "total_tokens":      int(input_tokens + output_tokens + cache_tokens),
        "input_cost":        round(input_cost, 8),
        "output_cost":       round(output_cost, 8),
        "cache_read_cost":   round(cache_read_cost, 8),
        "total_cost":        round(input_cost + output_cost + cache_read_cost, 8),
    }


def generate_traces(
    scenario: str = "anomaly",
    days: int = 30,
    seed: int = 42,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic observations DataFrame.

    scenario: "clean" | "anomaly"
    """
    rng = np.random.default_rng(seed)
    end_date = end_date or datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    workflows = CLEAN_WORKFLOWS if scenario == "clean" else ANOMALY_WORKFLOWS
    rows: list[dict] = []
    obs_counter = 0
    midpoint = start_date + timedelta(days=days / 2)

    for wf_def in workflows:
        if scenario == "clean":
            name, model, avg_turns, base_in, base_out, spd, cache_ratio = wf_def
            bloat_factor = 1.0
        else:
            name, model, avg_turns, base_in, base_out, spd, cache_ratio, bloat_factor = wf_def

        total_sessions = int(spd * days)

        for s in range(total_sessions):
            session_id = f"sess_{name.split('/')[0][:6]}_{s:05d}"
            # Distribute sessions across the period
            offset_hours = rng.uniform(0, days * 24)
            session_start = start_date + timedelta(hours=offset_hours)
            is_late_half = session_start > midpoint

            # ANOMALY: inject a cost spike in the second half for one workflow
            spike_mult = 1.0
            if scenario == "anomaly" and name == "data-pipeline/execute" and is_late_half:
                spike_mult = 3.2  # traffic + payload size increase

            n_turns = max(1, int(rng.normal(avg_turns, avg_turns * 0.3)))
            trace_id = f"trace_{obs_counter:07d}"
            cumulative_output = 0

            for turn in range(n_turns):
                obs_counter += 1
                ts = session_start + timedelta(seconds=turn * rng.uniform(8, 45))

                # ── Input token model ──────────────────────────────────
                if bloat_factor > 1.0:
                    # Bloated: input grows with FULL history re-injection
                    growth = cumulative_output * bloat_factor
                    noise = rng.normal(1.0, 0.15)
                    input_tokens = (base_in + growth) * noise * spike_mult
                    # Tool output injection spikes on random turns
                    if name == "data-pipeline/execute" and rng.random() < 0.35:
                        input_tokens *= rng.uniform(1.8, 3.5)
                else:
                    # Healthy: input grows only by prior output + small overhead
                    input_tokens = (base_in + cumulative_output * 0.85) * rng.normal(1.0, 0.10)

                input_tokens = max(100, input_tokens)
                output_tokens = max(20, base_out * rng.normal(1.0, 0.25))
                cumulative_output += output_tokens

                # ── Cache model ────────────────────────────────────────
                effective_cache = cache_ratio
                # ANOMALY: cache hit rate degrades over the period globally
                if scenario == "anomaly" and is_late_half:
                    effective_cache *= 0.45

                cache_tokens = input_tokens * effective_cache
                billable_input = input_tokens * (1 - effective_cache)

                rows.append(_build_observation(
                    rng, f"obs_{obs_counter:07d}", trace_id, session_id,
                    name, model, turn, billable_input, output_tokens,
                    cache_tokens, ts,
                ))

    # ANOMALY: a brand-new expensive workflow appears in the second half only
    if scenario == "anomaly":
        for s in range(180):
            session_id = f"sess_newagt_{s:05d}"
            offset_hours = rng.uniform(days * 12, days * 24)  # second half only
            session_start = start_date + timedelta(hours=offset_hours)
            trace_id = f"trace_new_{s:05d}"
            cumulative_output = 0
            for turn in range(int(rng.normal(6, 2))):
                obs_counter += 1
                ts = session_start + timedelta(seconds=turn * rng.uniform(10, 50))
                input_tokens = (5200 + cumulative_output * 4.0) * rng.normal(1.0, 0.2)
                output_tokens = max(50, 800 * rng.normal(1.0, 0.3))
                cumulative_output += output_tokens
                cache_tokens = input_tokens * 0.03
                rows.append(_build_observation(
                    rng, f"obs_{obs_counter:07d}", trace_id, session_id,
                    "multi-agent/orchestrate", "claude-opus-4-6", turn,
                    input_tokens * 0.97, output_tokens, cache_tokens, ts,
                ))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


class DemoConnector(BaseConnector):
    """Drop-in connector that serves synthetic data — no credentials required."""

    def __init__(self, scenario: str = "anomaly", seed: int = 42):
        self.scenario = scenario
        self.seed = seed

    def fetch_observations(
        self, start_date: datetime, end_date: datetime, project_id: str | None = None
    ) -> pd.DataFrame:
        days = max((end_date - start_date).days, 1)
        return generate_traces(
            scenario=self.scenario, days=days, seed=self.seed, end_date=end_date
        )

    def fetch_traces(
        self, start_date: datetime, end_date: datetime, project_id: str | None = None
    ) -> pd.DataFrame:
        obs = self.fetch_observations(start_date, end_date)
        return (
            obs.groupby("trace_id")
            .agg(
                trace_name=("workflow", "first"),
                session_id=("session_id", "first"),
                timestamp=("timestamp", "min"),
            )
            .reset_index()
            .assign(user_id=None, tags=lambda d: [[] for _ in range(len(d))])
        )

    def health_check(self) -> bool:
        return True


if __name__ == "__main__":
    for sc in ("clean", "anomaly"):
        df = generate_traces(scenario=sc, days=30)
        print(f"\n=== {sc.upper()} ===")
        print(f"Observations : {len(df):,}")
        print(f"Sessions     : {df['session_id'].nunique():,}")
        print(f"Workflows    : {df['workflow'].nunique()}")
        print(f"Total cost   : ${df['total_cost'].sum():.2f}")
        print(f"Avg input    : {df['input_tokens'].mean():,.0f} tokens")
        cache_pct = df["cache_read_tokens"].sum() / (
            df["input_tokens"].sum() + df["cache_read_tokens"].sum()
        )
        print(f"Cache hit    : {cache_pct:.1%}")
