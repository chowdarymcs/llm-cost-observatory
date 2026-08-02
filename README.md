# 🔭 LLM Cost Observatory

> **Observability-driven LLM cost optimization.**  
> Ingest Langfuse traces from ClickHouse or the Langfuse Cloud API — detect context bloat,
> identify cost spikes and regressions, quantify savings from each fix with copy-paste code snippets,
> and forecast your optimised monthly spend — all in a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Langfuse](https://img.shields.io/badge/Langfuse-Cloud%20%7C%20Self--hosted-purple)
![License](https://img.shields.io/badge/License-MIT-green)
[![Kaggle](https://img.shields.io/badge/Kaggle-Read%20the%20analysis-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/chandrasekhargenai/llm-cost-observatory-finding-where-your-llm-spend)
[![Live Report](https://img.shields.io/badge/Live%20Report-View%20Dashboard-2ECC71?logo=githubpages&logoColor=white)](https://chowdarymcs.github.io/llm-cost-observatory/)

🔗 **[View the live report →](https://chowdarymcs.github.io/llm-cost-observatory/)** — the full analysis output (alerts, savings forecast, priority matrix, code fixes) with zero setup required.

📓 **[Read the full methodology on Kaggle →](https://www.kaggle.com/code/chandrasekhargenai/llm-cost-observatory-finding-where-your-llm-spend)** — a runnable walkthrough comparing an optimised system against an unoptimised one costing **7× more** on the same workload.

---

## Why this exists

Most LLM cost discussions are about pricing tiers. The real cost driver is **architecture** —
specifically, what you're putting in the context window and why.

In a production agentic system (multi-step agent loops, RAG retrieval, MCP tool calls),
token cost distribution is almost never uniform:
- One or two runaway sessions can represent 40–60% of spend
- Context that compounds across turns (full history re-injection, raw tool output,
  over-fetched RAG chunks) inflates input tokens without adding value
- Prompt caching saves 80–90% on static tokens — but only if it's actually firing

This tool turns those abstract risks into a concrete, queryable dashboard so you can
fix the right thing instead of guessing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                      │
│  Overview │ Cost Breakdown │ Context Bloat │ Cache Analysis  │
└───────────────────────┬─────────────────────────────────────┘
                        │ pandas DataFrame (normalized)
          ┌─────────────┴──────────────┐
          │       Analysis Layer        │
          │  cost_analyzer.py           │
          │  bloat_detector.py          │
          │  cache_analyzer.py          │
          └─────────────┬──────────────┘
                        │
          ┌─────────────┴──────────────┐
          │      Connector Layer        │
          ├────────────────────────────┤
          │ ClickHouseConnector         │  ← self-hosted Langfuse
          │ (auto-detects v2/v3 schema) │     queries observations
          ├────────────────────────────┤     and traces tables
          │ LangfuseAPIConnector        │  ← Langfuse Cloud or
          │ (paginated REST API)        │     self-hosted REST API
          └────────────────────────────┘
```

**Key design decisions:**
- Both connectors return identical normalized DataFrames → analysis modules are source-agnostic
- Schema auto-detection handles Langfuse v2 (`prompt_tokens` columns) and v3 (`usage_details` Map)
- `@st.cache_data(ttl=300)` prevents redundant API calls while keeping data fresh
- Bloat scoring is model-agnostic — uses cumulative output as expected-input baseline

---

## Features

### 📊 Overview
- Total spend, traces, avg/P95 cost per trace
- Daily spend trend (stacked input / output / cache)
- Top workflows by cost, cost distribution by model

### 💰 Cost Breakdown
- Cost by workflow (top 20, color-coded by avg cost)
- Per-trace cost histogram + sortable table
- Model comparison (stacked input / output / cache costs)
- Routing opportunity detector — flags expensive models running low-output workloads

### 🧠 Context Bloat Detection
- Bloat score per session: `actual_input_tokens / expected_input_tokens`
- Severity tiers: Healthy (<2×), Moderate (2-5×), Severe (>5×)
- Estimated waste in tokens and USD per session
- Turn-by-turn token growth chart with excess overlay
- Contextual fix recommendations per severity level

### ⚡ Cache Analysis
- Overall cache hit rate and total savings
- Daily cache hit rate trend
- Hit rate by workflow — identifies which workflows benefit most (or least)
- Before/After comparison table: actual cost vs. hypothetical cost without caching
- Hit-rate-based recommendations

### 🎯 Recommendations (action centre)
- **Active Alerts** — cost spikes (2×+ vs baseline), bloat regressions, cache hit rate drops, new expensive workflows detected automatically
- **Savings Forecaster** — waterfall chart showing current 30-day projection → optimised projection after implementing each fix
- **Optimization Priority Matrix** — all opportunities ranked by `monthly_savings ÷ effort`, colour-coded by implementation complexity
- **Detailed Recommendations** — five anti-pattern detectors, each with:
  - Quantified monthly savings estimate
  - Plain-English problem description
  - Specific fix with implementation notes
  - **Copy-paste Python code snippet** for rolling summarization, tool output compression, two-phase RAG retrieval, prompt caching, and model routing

---

## Quick Start

### Try it in 30 seconds — no credentials needed

```bash
git clone https://github.com/chowdarymcs/llm-cost-observatory.git
cd llm-cost-observatory
pip install -r requirements.txt
streamlit run app.py
```

Select **🎭 Demo Mode** in the sidebar. Two scenarios ship with the repo:

| Scenario | What it simulates |
|---|---|
| ⚠️ **Unoptimised** | Runaway history accumulation, uncompressed tool outputs, RAG over-fetch, ~6% cache hit rate, a premium model doing trivial classification, and a mid-period cost spike |
| ✅ **Well-optimised** | Healthy baseline — controlled context growth, ~66% cache hit rate, task-appropriate model routing |

Running both back to back shows the same workload costing **7× more** purely from architecture.

### Generate a shareable HTML report

```bash
python generate_report.py --demo anomaly
python generate_report.py --demo clean

# Against real Langfuse data
python generate_report.py --source api --days 30
python generate_report.py --source clickhouse --days 30
```

Produces a single self-contained HTML file — all charts embedded, opens in any browser, no server required.

### Interactive notebook

[`notebooks/llm_cost_observatory_analysis.ipynb`](notebooks/llm_cost_observatory_analysis.ipynb) walks through the full methodology — bloat score derivation, root cause fingerprinting, savings quantification, and regression detection — with matplotlib visualisations at each step. Runs standalone with no dependency on the rest of the repo.

---

## Connecting to real data

### Prerequisites
- Python 3.11+
- A Langfuse account (Cloud) or self-hosted Langfuse instance with ClickHouse

### Configure
```bash
cp .env.example .env
```

Edit `.env` — for **Langfuse Cloud**:
```env
CONNECTOR_MODE=api
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

For **self-hosted** (ClickHouse direct):
```env
CONNECTOR_MODE=clickhouse
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_DATABASE=default
```

Then run `streamlit run app.py`, switch the sidebar connector from Demo Mode to your source, select a date range, and click **Load Data**.

---

## Model Pricing

`config.py` ships with pricing for Claude (Sonnet, Opus, Haiku), GPT-4o, GPT-4-Turbo,
GPT-3.5-Turbo, and Gemini 1.5 Pro/Flash. To add a model:

```python
# config.py → MODEL_PRICING
"your-model-name": {"input": 1.00, "output": 4.00, "cache_read": 0.10},
```

Units are **USD per 1M tokens**.

---

## Instrumentation (how to feed it data)

This dashboard reads from Langfuse. To populate it, instrument your LLM calls with Langfuse:

```python
from langfuse.decorators import langfuse_context, observe

@observe()
def my_agent_step(user_input: str) -> str:
    langfuse_context.update_current_trace(
        session_id="session-abc-123",   # critical for bloat analysis
        name="document-qa/generate",    # becomes the "workflow" label in the dashboard
    )
    # ... your LLM call here
```

Pass `session_id` consistently across turns within the same conversation —
this is what the bloat detector uses to compute turn-by-turn token growth.

---

## Project Structure

```
llm-cost-observatory/
├── app.py                          # Streamlit entry point
├── generate_report.py              # Standalone HTML report CLI
├── config.py                       # Settings (Pydantic) + model pricing table
├── requirements.txt
├── .env.example
├── notebooks/
│   └── llm_cost_observatory_analysis.ipynb   # Full methodology walkthrough
├── src/
│   ├── connectors/
│   │   ├── base.py                 # Abstract connector interface
│   │   ├── demo_connector.py       # Synthetic data — no credentials needed
│   │   ├── clickhouse_connector.py # Self-hosted Langfuse via ClickHouse
│   │   └── langfuse_api_connector.py # Cloud / REST API
│   ├── analysis/
│   │   ├── cost_analyzer.py        # Cost aggregations
│   │   ├── bloat_detector.py       # Context bloat scoring
│   │   ├── cache_analyzer.py       # Cache hit rate and savings
│   │   ├── recommendations.py      # Anti-pattern detection + code fixes
│   │   ├── anomaly.py              # Cost spike / regression alerts
│   │   └── forecaster.py           # Savings projection
│   ├── models/
│   │   └── trace_models.py         # Normalized data models
│   └── pages/
│       ├── overview.py
│       ├── cost_breakdown.py
│       ├── bloat_detection.py
│       ├── cache_analysis.py
│       └── recommendations.py      # Alerts, priority matrix, forecast
└── outputs/                        # Generated reports land here
```

---

## Methodology — how the bloat score works

In a healthy multi-turn session, the input tokens for turn *N* should be approximately:

```
expected_input(N) = system_prompt + Σ output(1..N-1)
```

You re-send what was said before, plus your instructions. Nothing more.

```
bloat_score = actual_input_tokens / expected_input_tokens
```

| Score | Interpretation |
|---|---|
| < 2× | Healthy |
| 2–5× | Moderate — worth investigating |
| > 5× | Severe — something is being re-injected repeatedly |

The score needs no code instrumentation beyond the token counts already present in any Langfuse trace. Root causes are then separated by their statistical signature:

| Pattern | Signature | Fix |
|---|---|---|
| History accumulation | Input grows steadily with turn index | Rolling summarization |
| Tool output injection | High input variance *within* a session | Compress tool results |
| RAG over-fetch | Uniformly high input from turn 1 | Two-phase retrieval |
| Cache miss | Low cache-read ratio across the board | Enable prompt caching |
| Model over-spend | Premium model, consistently small outputs | Route to cheaper model |

---

## License

MIT
