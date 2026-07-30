# 🔭 LLM Cost Observatory

> **Observability-driven LLM cost optimization.**  
> Ingest Langfuse traces from ClickHouse or the Langfuse Cloud API — detect context bloat,
> measure cache efficiency, and identify model routing opportunities — all in a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Langfuse](https://img.shields.io/badge/Langfuse-Cloud%20%7C%20Self--hosted-purple)
![License](https://img.shields.io/badge/License-MIT-green)

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
- **Routing opportunity detector** — flags expensive models running low-output workloads

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
- **Before/After comparison table**: actual cost vs. hypothetical cost without caching
- Hit-rate-based recommendations (when to enable caching, when to extend it to RAG context)

---

## Quick Start

### Prerequisites
- Python 3.11+
- A Langfuse account (Cloud) or self-hosted Langfuse instance with ClickHouse

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/llm-cost-observatory.git
cd llm-cost-observatory
pip install -r requirements.txt
```

### 2. Configure
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

### 3. Run
```bash
streamlit run app.py
```

Open `http://localhost:8501`, select your date range, click **Load Data**.

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
        name="coding-assistant/act",    # becomes the "workflow" label
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
├── config.py                       # Settings (Pydantic) + model pricing table
├── requirements.txt
├── .env.example
├── src/
│   ├── connectors/
│   │   ├── base.py                 # Abstract connector interface
│   │   ├── clickhouse_connector.py # Self-hosted Langfuse via ClickHouse
│   │   └── langfuse_api_connector.py # Cloud / REST API
│   ├── analysis/
│   │   ├── cost_analyzer.py        # Cost aggregations
│   │   ├── bloat_detector.py       # Context bloat scoring
│   │   └── cache_analyzer.py       # Cache hit rate and savings
│   ├── models/
│   │   └── trace_models.py         # Normalized data models
│   └── pages/
│       ├── overview.py
│       ├── cost_breakdown.py
│       ├── bloat_detection.py
│       └── cache_analysis.py
└── docs/
```

---



---

## License

MIT
