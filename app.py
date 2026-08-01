"""
LLM Cost Observatory
====================
Streamlit dashboard for Langfuse trace analysis.
Run: streamlit run app.py
"""

import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="LLM Cost Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 LLM Cost Observatory")
    st.caption("Langfuse trace cost analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Cost Breakdown", "Context Bloat", "Cache Analysis", "🎯 Recommendations"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Data Source")
    source = st.selectbox(
        "Connector",
        ["🎭 Demo Mode (no credentials)", "Langfuse Cloud API", "ClickHouse (Self-hosted)"],
        label_visibility="collapsed",
    )

    demo_scenario = "anomaly"
    if source.startswith("🎭"):
        demo_scenario = st.radio(
            "Scenario",
            ["anomaly", "clean"],
            format_func=lambda s: (
                "⚠️ Unoptimised system" if s == "anomaly" else "✅ Well-optimised system"
            ),
            help=(
                "Unoptimised: injected context bloat, cache misses, model over-spend, "
                "and a mid-period cost spike. Well-optimised: healthy baseline for comparison."
            ),
        )

    st.markdown("### Date Range")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("From", value=start_date.date())
    with col2:
        end = st.date_input("To", value=end_date.date())

    load_btn = st.button("🔄 Load Data", use_container_width=True, type="primary")

# ── Data loading (cached per connector + date range) ───────────────────
@st.cache_data(ttl=300, show_spinner="Loading traces…")
def load_data(connector_type: str, start_str: str, end_str: str, scenario: str = "anomaly"):
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_str)
    end_dt = datetime.fromisoformat(end_str)

    if connector_type.startswith("🎭"):
        from src.connectors.demo_connector import DemoConnector
        conn = DemoConnector(scenario=scenario)
    elif connector_type == "ClickHouse (Self-hosted)":
        from src.connectors.clickhouse_connector import ClickHouseConnector
        conn = ClickHouseConnector()
    else:
        from src.connectors.langfuse_api_connector import LangfuseAPIConnector
        conn = LangfuseAPIConnector()

    return conn.fetch_observations(start_dt, end_dt)


# ── Load or use session state ──────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None

if load_btn or st.session_state.df is None:
    try:
        st.session_state.df = load_data(
            source,
            datetime.combine(start, datetime.min.time()).isoformat(),
            datetime.combine(end, datetime.max.time()).isoformat(),
            demo_scenario,
        )
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.info(
            "Check your `.env` file — ensure `LANGFUSE_PUBLIC_KEY`, "
            "`LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` are set for API mode, "
            "or ClickHouse credentials for self-hosted mode."
        )
        st.stop()

df = st.session_state.df
if df is not None and not df.empty:
    # Workflow filter
    workflows = ["All"] + sorted(df["workflow"].dropna().unique().tolist())
    with st.sidebar:
        st.markdown("### Filter")
        selected_wf = st.multiselect("Workflow", workflows[1:], default=[])
        if selected_wf:
            df = df[df["workflow"].isin(selected_wf)]

# ── Route to page ──────────────────────────────────────────────────────
if df is None:
    st.info("👈 Configure your data source in the sidebar and click **Load Data** to begin.")
    st.stop()

if source.startswith("🎭"):
    scenario_label = (
        "⚠️ **Unoptimised system** — injected context bloat, cache misses, model over-spend, and a cost spike"
        if demo_scenario == "anomaly"
        else "✅ **Well-optimised system** — healthy baseline with good cache rates and controlled context growth"
    )
    st.info(f"🎭 **Demo Mode** — synthetic data, no Langfuse connection required. {scenario_label}")

if page == "Overview":
    from src.pages.overview import render
    render(df)
elif page == "Cost Breakdown":
    from src.pages.cost_breakdown import render
    render(df)
elif page == "Context Bloat":
    from src.pages.bloat_detection import render
    render(df)
elif page == "Cache Analysis":
    from src.pages.cache_analysis import render
    render(df)
elif page == "🎯 Recommendations":
    from src.pages.recommendations import render
    render(df)
