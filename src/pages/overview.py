import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.analysis.cost_analyzer import cost_kpis, daily_cost_trend, cost_by_workflow, cost_by_model


def render(df):
    st.header("📊 Overview")

    if df.empty:
        st.warning("No data returned for the selected date range.")
        return

    kpis = cost_kpis(df)

    # ── KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total LLM Spend", f"${kpis['total_cost']:.4f}")
    c2.metric("Total Traces", f"{kpis['total_traces']:,}")
    c3.metric("Avg Cost / Trace", f"${kpis['avg_cost_per_trace']:.5f}")
    c4.metric("P95 Cost / Trace", f"${kpis['p95_cost_per_trace']:.5f}")

    st.markdown("---")

    # ── Daily cost trend ───────────────────────────────────────────────
    daily = daily_cost_trend(df)
    if not daily.empty:
        st.subheader("Daily Spend Trend")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["input_cost"] if "input_cost" in daily.columns
                             else daily["total_cost"],
                             name="Input Cost", marker_color="#4F8EF7"))
        fig.add_trace(go.Bar(x=daily["date"], y=daily.get("output_cost", daily["total_cost"] * 0),
                             name="Output Cost", marker_color="#F4845F"))
        fig.add_trace(go.Bar(x=daily["date"], y=daily.get("cache_read_tokens", daily["total_cost"] * 0),
                             name="Cache Read Cost", marker_color="#2ECC71"))
        fig.update_layout(barmode="stack", xaxis_title="Date",
                          yaxis_title="Cost (USD)", height=320,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Top workflows by cost ──────────────────────────────────────────
    with col1:
        st.subheader("Top Workflows by Spend")
        wf = cost_by_workflow(df, top_n=10)
        if not wf.empty:
            fig = px.bar(wf, x="total_cost", y="workflow", orientation="h",
                         color="total_cost", color_continuous_scale="Blues",
                         labels={"total_cost": "Total Cost (USD)", "workflow": "Workflow"})
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                              coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Cost by model ──────────────────────────────────────────────────
    with col2:
        st.subheader("Cost Distribution by Model")
        mdl = cost_by_model(df)
        if not mdl.empty:
            fig = px.pie(mdl, values="total_cost", names="model",
                         hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
