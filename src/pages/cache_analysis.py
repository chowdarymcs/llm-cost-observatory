import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.analysis.cache_analyzer import (
    cache_kpis, cache_by_workflow, daily_cache_trend, before_after_comparison
)


def render(df):
    st.header("⚡ Cache Analysis")
    st.caption(
        "Quantifies savings from prompt caching and identifies which workflows "
        "benefit most — and which are leaving savings on the table."
    )

    if df.empty:
        st.warning("No data returned for the selected date range.")
        return

    kpis = cache_kpis(df)

    # ── KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cache Hit Rate", f"{kpis['hit_rate']}%")
    c2.metric("Total Savings", f"${kpis['total_savings_usd']:.4f}")
    c3.metric("Cached Tokens", f"{kpis['cached_tokens']:,}")
    c4.metric("Cost Without Cache", f"${kpis['uncached_equiv_cost']:.4f}")

    st.markdown("---")

    # ── Daily cache hit rate trend ─────────────────────────────────────
    daily = daily_cache_trend(df)
    if not daily.empty:
        st.subheader("Daily Cache Hit Rate")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cache_hit_rate_pct"],
            mode="lines+markers", name="Cache Hit Rate %",
            line=dict(color="#2ECC71", width=2), fill="tozeroy",
            fillcolor="rgba(46,204,113,0.15)"
        ))
        fig.update_layout(
            yaxis_title="Cache Hit Rate (%)", xaxis_title="Date",
            height=280, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Cache by workflow ──────────────────────────────────────────────
    with col1:
        st.subheader("Cache Hit Rate by Workflow")
        wf = cache_by_workflow(df)
        if not wf.empty:
            fig = px.bar(
                wf.head(12), x="cache_hit_rate_pct", y="workflow",
                orientation="h", color="cache_hit_rate_pct",
                color_continuous_scale="Greens",
                labels={"cache_hit_rate_pct": "Hit Rate (%)", "workflow": "Workflow"}
            )
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                              coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Before / After table ───────────────────────────────────────────
    with col2:
        st.subheader("Before vs. After Caching (by Model)")
        ba = before_after_comparison(df)
        if not ba.empty:
            st.dataframe(
                ba.style.format({
                    "actual_cost_usd":   "${:.5f}",
                    "without_cache_usd": "${:.5f}",
                    "savings_usd":       "${:.5f}",
                    "savings_pct":       "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ── Recommendations ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Optimization Recommendations")
    hit_rate = kpis["hit_rate"]
    if hit_rate < 20:
        st.error(
            "🔴 Cache hit rate is very low. Ensure prompt caching is enabled for system prompts, "
            "guard rail instructions, and tool schemas — these are static across calls and should "
            "always be cached. Verify the `cache_control` parameter is set on your static blocks."
        )
    elif hit_rate < 50:
        st.warning(
            "🟡 Moderate cache efficiency. Beyond static system prompts, look at caching "
            "frequently-repeated RAG context chunks. Consider a two-phase retrieval pattern: "
            "cache the full knowledge base summary, fetch detail only when needed."
        )
    else:
        st.success(
            "🟢 Good cache utilization. Focus next on model routing — shift short-output "
            "classification and extraction tasks to cheaper models to compound savings."
        )
