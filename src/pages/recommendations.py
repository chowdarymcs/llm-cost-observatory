"""
Recommendations Page — the action centre of the observatory.

Layout:
  1. Active Alerts (cost spikes, regressions, cache drops)
  2. Savings Forecast (waterfall chart — current → optimised)
  3. Optimization Priority Matrix (ranked table)
  4. Detailed Recommendations (expandable code snippets)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.analysis.recommendations import generate_recommendations, priority_matrix, Recommendation
from src.analysis.anomaly import detect_alerts, SEVERITY_EMOJI
from src.analysis.forecaster import build_forecast
from src.analysis.bloat_detector import compute_bloat_scores
from src.analysis.cache_analyzer import cache_kpis, daily_cache_trend
from src.analysis.cost_analyzer import daily_cost_trend

EFFORT_COLOR = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}
PATTERN_LABEL = {
    "HISTORY_ACCUMULATION":  "📈 History Accumulation",
    "TOOL_OUTPUT_INJECTION": "🔧 Tool Output Injection",
    "RAG_OVERFETCH":         "📚 RAG Over-fetch",
    "CACHE_MISS":            "⚡ Cache Miss",
    "EXPENSIVE_MODEL_OVERUSE":"💸 Model Over-spend",
}


def render(df: pd.DataFrame):
    st.header("🎯 Recommendations & Optimization")

    if df.empty:
        st.warning("No data returned for the selected date range.")
        return

    # ── Compute all dependencies ───────────────────────────────────────
    with st.spinner("Analysing patterns..."):
        bloat_df      = compute_bloat_scores(df)
        c_kpis        = cache_kpis(df)
        daily_cache   = daily_cache_trend(df)
        date_range_days = max(
            (pd.to_datetime(df["timestamp"]).max() -
             pd.to_datetime(df["timestamp"]).min()).days, 1
        )
        recs   = generate_recommendations(df, bloat_df, c_kpis, date_range_days)
        alerts = detect_alerts(df, bloat_df, daily_cache)
        fc     = build_forecast(df, recs, date_range_days)

    # ══════════════════════════════════════════════════════════════════
    # 1. ACTIVE ALERTS
    # ══════════════════════════════════════════════════════════════════
    st.subheader(f"🚨 Active Alerts ({len(alerts)})")

    if not alerts:
        st.success("✅ No anomalies detected in the selected date range.")
    else:
        for alert in alerts:
            emoji = SEVERITY_EMOJI.get(alert.severity, "🔵")
            color = {"Critical": "#fde8e8", "Warning": "#fff8e1", "Info": "#e8f4fd"}[alert.severity]
            border = {"Critical": "#E74C3C", "Warning": "#F39C12", "Info": "#4F8EF7"}[alert.severity]
            with st.container():
                st.markdown(
                    f"""<div style="background:{color};border-left:4px solid {border};
                    padding:12px 16px;border-radius:6px;margin-bottom:8px;">
                    <strong>{emoji} {alert.severity} — {alert.title}</strong><br/>
                    <span style="font-size:0.9em">{alert.description}</span><br/>
                    <span style="font-size:0.8em;color:#666">
                    Current: <strong>{alert.current_value}</strong> &nbsp;|&nbsp;
                    Baseline: {alert.baseline_value} &nbsp;|&nbsp;
                    Change: {alert.change_pct:+.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 2. SAVINGS FORECAST
    # ══════════════════════════════════════════════════════════════════
    st.subheader("📉 Savings Forecast — Current vs Optimised (30-day)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current 30-day Projection", f"${fc['monthly_projection']:.2f}")
    col2.metric("Optimised Projection",       f"${fc['optimized_monthly']:.2f}")
    col3.metric("Total Monthly Savings",      f"${fc['total_monthly_savings']:.2f}")
    col4.metric("Savings %",                  f"{fc['savings_pct']:.1f}%",
                delta=f"{fc['savings_pct']:.1f}%", delta_color="normal")

    # Waterfall chart
    if fc["waterfall_data"]:
        labels   = [item["label"] for item in fc["waterfall_data"]]
        measures = ["absolute" if item["type"] == "total" else "relative"
                    for item in fc["waterfall_data"]]
        values   = [item["value"] for item in fc["waterfall_data"]]
        colors   = []
        for item in fc["waterfall_data"]:
            if item["type"] == "total":
                colors.append("#4F8EF7")
            else:
                effort = item.get("effort", "Low")
                colors.append(EFFORT_COLOR.get(effort, "#2ECC71"))

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#2ECC71"}},
            increasing={"marker": {"color": "#E74C3C"}},
            totals={"marker": {"color": "#4F8EF7"}},
            textposition="outside",
            text=[f"${abs(v):.3f}" for v in values],
        ))
        fig.update_layout(
            title="",
            yaxis_title="Monthly Cost (USD)",
            height=380,
            margin=dict(l=0, r=0, t=10, b=80),
            xaxis_tickangle=-20,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "🟢 Green bars = savings | Blue bars = cost totals. "
            "Savings are conservative estimates; actual results depend on implementation."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 3. OPTIMIZATION PRIORITY MATRIX
    # ══════════════════════════════════════════════════════════════════
    st.subheader("📊 Optimization Priority Matrix")
    st.caption("Ranked by monthly savings ÷ implementation effort. Fix the top rows first.")

    if not recs:
        st.info("No optimization opportunities detected in the current data window.")
    else:
        pm = priority_matrix(recs)

        # Colour effort column
        def colour_effort(val):
            c = {"Low": "#d4edda", "Medium": "#fff3cd", "High": "#f8d7da"}.get(val, "white")
            return f"background-color: {c}"

        styled = pm.style.format({
            "Monthly Savings": "${:.4f}",
            "Priority Score":  "{:.4f}",
            "Sessions/Day":    "{:.1f}",
        }).applymap(colour_effort, subset=["Effort"])

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Bubble chart — savings vs effort
        pm["Effort Num"] = pm["Effort"].map({"Low": 1, "Medium": 2, "High": 3})
        fig2 = px.scatter(
            pm,
            x="Effort Num", y="Monthly Savings",
            size="Monthly Savings", color="Effort",
            text="Fix",
            color_discrete_map={"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"},
            labels={"Effort Num": "Implementation Effort", "Monthly Savings": "Monthly Savings (USD)"},
        )
        fig2.update_xaxes(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"])
        fig2.update_traces(textposition="top center", marker=dict(sizemin=10))
        fig2.update_layout(
            height=340, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 4. DETAILED RECOMMENDATIONS WITH CODE SNIPPETS
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🔧 Detailed Recommendations")

    if not recs:
        st.info("No recommendations to display.")
        return

    for i, rec in enumerate(recs):
        pattern_label = PATTERN_LABEL.get(rec.pattern, rec.pattern)
        effort_color  = EFFORT_COLOR.get(rec.effort, "#888")

        with st.expander(
            f"#{i+1}  {rec.title} — "
            f"${rec.monthly_savings_usd:.4f}/mo savings · {rec.effort} effort",
            expanded=(i == 0),
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Monthly Savings",   f"${rec.monthly_savings_usd:.4f}")
            c2.metric("Sessions/Day",      f"{rec.sessions_per_day:.1f}")
            c3.metric("Effort",            rec.effort)

            st.markdown(f"**Pattern:** {pattern_label}")
            st.markdown(f"**Workflow:** `{rec.workflow}`")

            st.markdown("**Problem**")
            st.warning(rec.problem)

            st.markdown("**Fix**")
            st.info(rec.fix)

            st.markdown("**Implementation**")
            st.code(rec.code_snippet, language="python")

            if rec.waste_tokens_per_session > 0:
                st.caption(
                    f"Estimated waste: {rec.waste_tokens_per_session:,} tokens/session × "
                    f"{rec.sessions_per_day:.1f} sessions/day × 30 days = "
                    f"${rec.monthly_savings_usd:.4f}/month recoverable"
                )
