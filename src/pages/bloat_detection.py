import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.analysis.bloat_detector import (
    compute_bloat_scores, session_token_growth, bloat_summary_stats
)

SEVERITY_COLORS = {"Severe": "#E74C3C", "Moderate": "#F39C12", "Healthy": "#2ECC71"}


def render(df):
    st.header("🧠 Context Bloat Detection")
    st.caption(
        "Measures how much your input tokens grow beyond what's necessary. "
        "Bloat score > 2.0 = moderate waste. > 5.0 = severe. "
        "Root causes: verbatim history accumulation, uncompressed tool outputs, over-fetching in RAG."
    )

    if df.empty:
        st.warning("No data returned for the selected date range.")
        return

    bloat_df = compute_bloat_scores(df)

    if bloat_df.empty:
        st.info("Not enough multi-turn session data to compute bloat scores. "
                "Ensure session_id is propagated in your Langfuse traces.")
        return

    stats = bloat_summary_stats(bloat_df)

    # ── KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated Waste", f"${stats['total_waste_usd']:.4f}")
    c2.metric("Severe Sessions", stats["severe_sessions"],
              delta=f"{stats['severe_sessions']} to fix", delta_color="inverse")
    c3.metric("Moderate Sessions", stats["moderate_sessions"])
    c4.metric("Avg Bloat Score", f"{stats['avg_bloat_score']}×")

    st.markdown("---")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Sessions Ranked by Bloat Score")
        display = bloat_df.copy()
        display["severity"] = display["severity"].apply(
            lambda s: f"🔴 {s}" if s == "Severe" else (f"🟡 {s}" if s == "Moderate" else f"🟢 {s}")
        )
        st.dataframe(
            display[["session_id", "workflow", "turns", "bloat_score",
                      "waste_tokens", "waste_usd", "severity"]],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader("Bloat Score Distribution")
        fig = px.histogram(bloat_df, x="bloat_score", nbins=20,
                           color="severity",
                           color_discrete_map=SEVERITY_COLORS,
                           labels={"bloat_score": "Bloat Score"})
        fig.add_vline(x=2.0, line_dash="dash", line_color="#F39C12",
                      annotation_text="Moderate", annotation_position="top right")
        fig.add_vline(x=5.0, line_dash="dash", line_color="#E74C3C",
                      annotation_text="Severe", annotation_position="top right")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-session deep dive ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Session Deep Dive — Token Growth Chart")
    session_ids = bloat_df["session_id"].tolist()
    selected = st.selectbox(
        "Select a session to inspect",
        session_ids,
        format_func=lambda s: f"{s[:24]}… | score={bloat_df.loc[bloat_df['session_id']==s, 'bloat_score'].values[0]}×"
    )

    if selected:
        growth = session_token_growth(df, selected)
        if not growth.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=growth["turn_index"], y=growth["input_tokens"],
                                     name="Actual Input Tokens", mode="lines+markers",
                                     line=dict(color="#E74C3C", width=2)))
            fig.add_trace(go.Scatter(x=growth["turn_index"], y=growth["expected_input_tokens"],
                                     name="Expected (no bloat)", mode="lines+markers",
                                     line=dict(color="#2ECC71", width=2, dash="dash")))
            fig.add_trace(go.Bar(x=growth["turn_index"], y=growth["excess_tokens"],
                                 name="Excess Tokens (waste)", marker_color="rgba(231,76,60,0.3)",
                                 yaxis="y"))
            fig.update_layout(
                xaxis_title="Turn Index",
                yaxis_title="Tokens",
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Recommendations for this session:**")
            score = bloat_df.loc[bloat_df["session_id"] == selected, "bloat_score"].values[0]
            _recommendations(score)


def _recommendations(score: float):
    recs = []
    if score >= 5.0:
        recs += [
            "🔴 **Rolling summarization**: collapse turns older than N into a summary before re-injecting.",
            "🔴 **Tool output compression**: summarize tool results before returning to the model — never inject raw payloads.",
        ]
    if score >= 2.0:
        recs += [
            "🟡 **Externalize state to storage**: pass IDs/references through the agent loop; rehydrate only at point of use.",
            "🟡 **Scope tool catalogs per agent mode**: don't expose all routes on every call.",
        ]
    recs.append("🟢 **Enable prompt caching** for static system prompt + guard rail instructions to offset base overhead.")
    for r in recs:
        st.markdown(r)
