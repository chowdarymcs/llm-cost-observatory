"""
Recommendations Engine.

Detects anti-patterns in LLM usage, quantifies their cost impact,
and generates prioritized, code-level fixes with monthly savings estimates.

Patterns detected:
    HISTORY_ACCUMULATION   — unbounded turn history re-injection
    TOOL_OUTPUT_INJECTION  — raw tool/function results dumped verbatim into context
    RAG_OVERFETCH          — retrieving far more content than the query needs
    CACHE_MISS             — static tokens (system prompt, schemas) re-billed every call
    EXPENSIVE_MODEL_OVERUSE — high-cost model doing low-complexity work
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from config import MODEL_PRICING, DEFAULT_PRICING

# ── Cheaper model fallback map ─────────────────────────────────────────
CHEAPER_MODEL = {
    "claude-opus-4-6":           "claude-sonnet-4-6",
    "claude-opus-4-7":           "claude-sonnet-4-6",
    "claude-opus-4-8":           "claude-sonnet-4-6",
    "gpt-4-turbo":               "gpt-4o",
    "gpt-4o":                    "gpt-4o-mini",
    "gemini-1.5-pro":            "gemini-1.5-flash",
    "claude-3-5-sonnet-20241022":"claude-haiku-4-5",
}
EXPENSIVE_MODELS = set(CHEAPER_MODEL.keys())
EFFORT_SCORE = {"Low": 1, "Medium": 2, "High": 3}


# ── Code snippet library ───────────────────────────────────────────────
SNIPPETS = {
    "HISTORY_ACCUMULATION": '''\
# Rolling summarization — collapse old turns before re-injecting
from langfuse.decorators import observe

SUMMARIZE_AFTER_TURNS = 8   # tune based on your avg session length

@observe()
async def compress_history(history: list[dict], llm) -> list[dict]:
    """Collapse turns older than N into a single summary, preserving recency."""
    if len(history) <= SUMMARIZE_AFTER_TURNS:
        return history

    to_summarize = history[:-SUMMARIZE_AFTER_TURNS]
    recent       = history[-SUMMARIZE_AFTER_TURNS:]

    summary = await llm.complete(
        f"Summarize this conversation in 150 words:\\n{to_summarize}"
    )
    return [{"role": "assistant", "content": f"[Summary]: {summary}"}] + recent
''',

    "TOOL_OUTPUT_INJECTION": '''\
# Tool output compression middleware
import json

MAX_TOOL_TOKENS = 400   # safe ceiling before injection becomes noise

def compress_tool_output(raw_output: str | dict, max_chars: int = MAX_TOOL_TOKENS * 4) -> str:
    """Summarize oversized tool results before injecting into context."""
    if isinstance(raw_output, dict):
        raw_output = json.dumps(raw_output, indent=2)

    if len(raw_output) <= max_chars:
        return raw_output

    lines  = raw_output.strip().split("\\n")
    head   = "\\n".join(lines[:15])
    tail   = "\\n".join(lines[-5:])
    elided = len(lines) - 20
    return f"{head}\\n... [{elided} lines compressed — attach full output if needed] ...\\n{tail}"
''',

    "RAG_OVERFETCH": '''\
# Two-phase retrieval — anchor search first, full fetch only on demand
async def two_phase_retrieve(
    query: str,
    vector_store,
    k_anchor: int = 5,
    k_full: int   = 1,
) -> tuple[list[dict], str]:
    """
    Phase 1: retrieve lightweight anchors (id + summary).
    Phase 2: fetch full content only for the top-ranked anchor.
    Token cost: anchor tokens only unless full content is actually needed.
    """
    # Phase 1 — summaries / signatures, not full docs
    anchors = await vector_store.search(
        query, k=k_anchor, fields=["id", "summary", "score"]
    )

    # Phase 2 — full content only for best match
    best_id      = anchors[0]["id"]
    full_content = await vector_store.fetch(best_id)

    return anchors, full_content
''',

    "CACHE_MISS": '''\
# Prompt caching for static system prompt + tool schemas
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """..."""   # your static system prompt / guard rail instructions

def call_with_caching(conversation_history: list[dict]) -> str:
    """Cache the static system prompt — billed once per 5-min TTL instead of every call."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # ← key line
            }
        ],
        messages=conversation_history,
    )
    return response.content[0].text
''',

    "EXPENSIVE_MODEL_OVERUSE": '''\
# Model router — cheapest model capable of the task
def route_model(task_type: str, estimated_output_tokens: int) -> str:
    """
    Route to the cheapest model that handles the task.
    Classification / extraction / short answers → Haiku.
    Reasoning / code / longer outputs              → Sonnet.
    Complex multi-step or very long generation     → Opus.
    """
    if task_type in ("classification", "extraction", "summarization"):
        return "claude-haiku-4-5"
    if estimated_output_tokens < 500:
        return "claude-haiku-4-5"
    if estimated_output_tokens < 2000:
        return "claude-sonnet-4-6"
    return "claude-opus-4-6"
''',
}


@dataclass
class Recommendation:
    id: str
    workflow: str
    pattern: str
    title: str
    problem: str
    fix: str
    code_snippet: str
    waste_tokens_per_session: int
    waste_usd_per_session: float
    sessions_per_day: float
    monthly_savings_usd: float
    effort: str                    # "Low" | "Medium" | "High"
    effort_score: int              # 1-3
    priority_score: float          # monthly_savings / effort_score
    impacted_sessions: int = 0


def _rec_id(workflow: str, pattern: str) -> str:
    return hashlib.md5(f"{workflow}:{pattern}".encode()).hexdigest()[:8]


def _pricing_for(model: str) -> dict:
    return MODEL_PRICING.get(model or "", DEFAULT_PRICING)


def generate_recommendations(
    obs_df: pd.DataFrame,
    bloat_df: pd.DataFrame,
    cache_kpis: dict,
    date_range_days: int = 30,
) -> list[Recommendation]:
    """
    Analyse observations and bloat data to produce a prioritised
    list of Recommendation objects with quantified monthly savings.
    """
    recs: list[Recommendation] = []
    if obs_df.empty:
        return recs

    days = max(date_range_days, 1)

    # ── Per-workflow aggregates ────────────────────────────────────────
    wf_stats = (
        obs_df.groupby("workflow")
        .agg(
            total_cost      =("total_cost",       "sum"),
            input_tokens    =("input_tokens",      "mean"),
            output_tokens   =("output_tokens",     "mean"),
            cache_read_tokens=("cache_read_tokens","sum"),
            session_count   =("session_id",        "nunique"),
            generation_count=("id",                "count"),
        )
        .reset_index()
    )
    wf_stats["sessions_per_day"] = wf_stats["session_count"] / days

    # ── 1. HISTORY_ACCUMULATION ────────────────────────────────────────
    if not bloat_df.empty:
        bloated = bloat_df[bloat_df["bloat_score"] >= 2.0].copy()
        if not bloated.empty:
            for _, row in bloated.iterrows():
                wf = row["workflow"]
                waste_tokens = int(row["waste_tokens"])
                spd = wf_stats.loc[wf_stats["workflow"] == wf, "sessions_per_day"]
                spd_val = float(spd.values[0]) if not spd.empty else 0.1

                model = obs_df[obs_df["workflow"] == wf]["model"].mode()
                model_val = model.values[0] if not model.empty else ""
                price = _pricing_for(model_val)["input"]
                waste_usd = (waste_tokens / 1_000_000) * price
                monthly = waste_usd * spd_val * 30 * 0.70  # 70% recoverable via summarization

                recs.append(Recommendation(
                    id=_rec_id(wf, "HISTORY_ACCUMULATION"),
                    workflow=wf,
                    pattern="HISTORY_ACCUMULATION",
                    title="Implement rolling summarization",
                    problem=(
                        f"Workflow '{wf}' has a bloat score of {row['bloat_score']:.1f}× — "
                        f"input tokens are growing {row['bloat_score']:.1f}× faster than expected "
                        f"from output alone. Estimated ~{waste_tokens:,} wasted tokens/session."
                    ),
                    fix="Collapse turns older than N into a compact summary before re-injecting. "
                        "Rolling summarization typically recovers 60–80% of compounding token waste.",
                    code_snippet=SNIPPETS["HISTORY_ACCUMULATION"],
                    waste_tokens_per_session=waste_tokens,
                    waste_usd_per_session=round(waste_usd, 5),
                    sessions_per_day=round(spd_val, 2),
                    monthly_savings_usd=round(monthly, 4),
                    effort="Medium",
                    effort_score=2,
                    priority_score=round(monthly / 2, 4),
                    impacted_sessions=int(row.get("turns", 1)),
                ))

    # ── 2. TOOL_OUTPUT_INJECTION (high input variance within sessions) ─
    if "session_id" in obs_df.columns:
        sess_var = (
            obs_df.dropna(subset=["session_id"])
            .groupby(["workflow", "session_id"])["input_tokens"]
            .std()
            .reset_index()
            .rename(columns={"input_tokens": "input_std"})
        )
        sess_mean = (
            obs_df.groupby("workflow")["input_tokens"].mean()
            .reset_index()
            .rename(columns={"input_tokens": "input_mean"})
        )
        sess_var = sess_var.merge(sess_mean, on="workflow")
        sess_var["cv"] = sess_var["input_std"] / sess_var["input_mean"].replace(0, 1)
        high_var_wf = sess_var.groupby("workflow")["cv"].mean()
        high_var_wf = high_var_wf[high_var_wf > 0.5].index.tolist()

        for wf in high_var_wf:
            row = wf_stats[wf_stats["workflow"] == wf]
            if row.empty:
                continue
            avg_input = float(row["input_tokens"].values[0])
            spd_val = float(row["sessions_per_day"].values[0])
            model = obs_df[obs_df["workflow"] == wf]["model"].mode()
            model_val = model.values[0] if not model.empty else ""
            price = _pricing_for(model_val)["input"]
            est_waste = avg_input * 0.25  # conservative: 25% of input is tool payload
            waste_usd = (est_waste / 1_000_000) * price
            monthly = waste_usd * spd_val * 30 * 0.50

            recs.append(Recommendation(
                id=_rec_id(wf, "TOOL_OUTPUT_INJECTION"),
                workflow=wf,
                pattern="TOOL_OUTPUT_INJECTION",
                title="Add tool output compression middleware",
                problem=(
                    f"Workflow '{wf}' shows high input token variance (CV > 0.5), "
                    "indicating raw tool/function results are being injected verbatim. "
                    f"Avg input: {avg_input:,.0f} tokens — spikes suggest uncompressed payloads."
                ),
                fix="Intercept tool outputs before context injection. Truncate or summarize "
                    "results above a token threshold. Aim for < 400 tokens per tool result.",
                code_snippet=SNIPPETS["TOOL_OUTPUT_INJECTION"],
                waste_tokens_per_session=int(est_waste),
                waste_usd_per_session=round(waste_usd, 5),
                sessions_per_day=round(spd_val, 2),
                monthly_savings_usd=round(monthly, 4),
                effort="Low",
                effort_score=1,
                priority_score=round(monthly / 1, 4),
                impacted_sessions=int(row["session_count"].values[0]),
            ))

    # ── 3. RAG_OVERFETCH (uniformly high input from turn 1) ───────────
    high_input_wf = wf_stats[wf_stats["input_tokens"] > 2000]
    for _, row in high_input_wf.iterrows():
        wf = row["workflow"]
        if any(r.workflow == wf and r.pattern in
               ("HISTORY_ACCUMULATION", "TOOL_OUTPUT_INJECTION") for r in recs):
            continue  # already covered by another pattern
        avg_input = float(row["input_tokens"])
        spd_val = float(row["sessions_per_day"])
        model = obs_df[obs_df["workflow"] == wf]["model"].mode()
        model_val = model.values[0] if not model.empty else ""
        price = _pricing_for(model_val)["input"]
        est_waste = avg_input * 0.40
        waste_usd = (est_waste / 1_000_000) * price
        monthly = waste_usd * spd_val * 30

        recs.append(Recommendation(
            id=_rec_id(wf, "RAG_OVERFETCH"),
            workflow=wf,
            pattern="RAG_OVERFETCH",
            title="Switch to two-phase retrieval (anchor → fetch)",
            problem=(
                f"Workflow '{wf}' averages {avg_input:,.0f} input tokens — "
                "suggesting full document chunks are retrieved regardless of query specificity. "
                "Two-phase retrieval fetches summaries first, full content only when needed."
            ),
            fix="Phase 1: retrieve lightweight anchors (id + 1-sentence summary). "
                "Phase 2: fetch full content only for the top-ranked result. "
                "Typically cuts RAG input tokens by 35–55%.",
            code_snippet=SNIPPETS["RAG_OVERFETCH"],
            waste_tokens_per_session=int(est_waste),
            waste_usd_per_session=round(waste_usd, 5),
            sessions_per_day=round(spd_val, 2),
            monthly_savings_usd=round(monthly, 4),
            effort="Medium",
            effort_score=2,
            priority_score=round(monthly / 2, 4),
            impacted_sessions=int(row["session_count"]),
        ))

    # ── 4. CACHE_MISS ─────────────────────────────────────────────────
    hit_rate = cache_kpis.get("hit_rate", 0)
    if hit_rate < 20:
        total_input = obs_df["input_tokens"].sum()
        total_sessions = obs_df["session_id"].nunique() if "session_id" in obs_df.columns else 1
        spd_val = total_sessions / days
        est_system_prompt = 500  # conservative estimate
        model = obs_df["model"].mode()
        model_val = model.values[0] if not model.empty else ""
        price_full  = _pricing_for(model_val)["input"]
        price_cache = _pricing_for(model_val)["cache_read"]
        savings_per_call = (est_system_prompt / 1_000_000) * (price_full - price_cache)
        monthly = savings_per_call * spd_val * 30

        recs.append(Recommendation(
            id=_rec_id("global", "CACHE_MISS"),
            workflow="(all workflows)",
            pattern="CACHE_MISS",
            title="Enable prompt caching for static context",
            problem=(
                f"Cache hit rate is only {hit_rate:.1f}%. "
                "Static tokens (system prompt, guard rail instructions, tool schemas) "
                "are being re-billed at full input price on every call. "
                f"Cache reads cost ~10× less than regular input tokens."
            ),
            fix="Add cache_control: ephemeral to all static content blocks. "
                "System prompts, tool schemas, and RAG knowledge base context "
                "are the highest-value targets.",
            code_snippet=SNIPPETS["CACHE_MISS"],
            waste_tokens_per_session=est_system_prompt,
            waste_usd_per_session=round(savings_per_call, 6),
            sessions_per_day=round(spd_val, 2),
            monthly_savings_usd=round(monthly, 4),
            effort="Low",
            effort_score=1,
            priority_score=round(monthly / 1, 4),
            impacted_sessions=int(total_sessions),
        ))

    # ── 5. EXPENSIVE_MODEL_OVERUSE ────────────────────────────────────
    model_wf = (
        obs_df.groupby(["workflow", "model"])
        .agg(total_cost=("total_cost", "sum"),
             avg_output=("output_tokens", "mean"),
             count=("id", "count"))
        .reset_index()
    )
    for _, row in model_wf.iterrows():
        wf = row["workflow"]
        model = row["model"]
        if model not in EXPENSIVE_MODELS:
            continue
        if row["avg_output"] > 400:
            continue  # model is justified for long outputs
        cheaper = CHEAPER_MODEL[model]
        p_curr  = _pricing_for(model)
        p_cheap = _pricing_for(cheaper)
        wf_row  = wf_stats[wf_stats["workflow"] == wf]
        spd_val = float(wf_row["sessions_per_day"].values[0]) if not wf_row.empty else 0.1
        avg_in  = float(wf_row["input_tokens"].values[0]) if not wf_row.empty else 500
        avg_out = float(row["avg_output"])
        curr_cost_per  = (avg_in / 1e6) * p_curr["input"]  + (avg_out / 1e6) * p_curr["output"]
        cheap_cost_per = (avg_in / 1e6) * p_cheap["input"] + (avg_out / 1e6) * p_cheap["output"]
        monthly = (curr_cost_per - cheap_cost_per) * spd_val * 30

        recs.append(Recommendation(
            id=_rec_id(wf, "EXPENSIVE_MODEL_OVERUSE"),
            workflow=wf,
            pattern="EXPENSIVE_MODEL_OVERUSE",
            title=f"Route '{wf}' from {model} → {cheaper}",
            problem=(
                f"Workflow '{wf}' uses {model} with avg output of only "
                f"{row['avg_output']:.0f} tokens — well below the threshold where "
                f"the premium model adds value over {cheaper}."
            ),
            fix=f"Route this workflow to {cheaper}. For tasks producing < 400 tokens, "
                "the quality difference is negligible. Add a task_type classifier to "
                "route dynamically based on query complexity.",
            code_snippet=SNIPPETS["EXPENSIVE_MODEL_OVERUSE"],
            waste_tokens_per_session=0,
            waste_usd_per_session=round(curr_cost_per - cheap_cost_per, 6),
            sessions_per_day=round(spd_val, 2),
            monthly_savings_usd=round(monthly, 4),
            effort="Low",
            effort_score=1,
            priority_score=round(monthly / 1, 4),
            impacted_sessions=int(wf_row["session_count"].values[0]) if not wf_row.empty else 0,
        ))

    # Deduplicate by id, sort by priority_score descending
    seen = set()
    unique = []
    for r in sorted(recs, key=lambda x: x.priority_score, reverse=True):
        if r.id not in seen:
            seen.add(r.id)
            unique.append(r)

    return unique


def priority_matrix(recs: list[Recommendation]) -> pd.DataFrame:
    """Returns a DataFrame formatted for the priority matrix table."""
    if not recs:
        return pd.DataFrame()
    rows = [{
        "Priority":         i + 1,
        "Workflow":         r.workflow,
        "Pattern":          r.pattern,
        "Fix":              r.title,
        "Monthly Savings":  r.monthly_savings_usd,
        "Effort":           r.effort,
        "Priority Score":   round(r.priority_score, 4),
        "Sessions/Day":     r.sessions_per_day,
    } for i, r in enumerate(recs)]
    return pd.DataFrame(rows)
