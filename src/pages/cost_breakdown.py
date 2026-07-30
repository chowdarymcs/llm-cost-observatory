import streamlit as st
import plotly.express as px
from src.analysis.cost_analyzer import (
    cost_by_workflow, cost_by_model, cost_per_trace_distribution, routing_opportunity
)


def render(df):
    st.header("💰 Cost Breakdown")

    if df.empty:
        st.warning("No data returned for the selected date range.")
        return

    tab1, tab2, tab3 = st.tabs(["By Workflow", "By Model", "Routing Opportunities"])

    # ── Tab 1: By Workflow ─────────────────────────────────────────────
    with tab1:
        wf = cost_by_workflow(df, top_n=20)
        if not wf.empty:
            fig = px.bar(wf, x="workflow", y="total_cost",
                         color="avg_cost",
                         color_continuous_scale="RdYlGn_r",
                         labels={"total_cost": "Total Cost (USD)",
                                 "avg_cost": "Avg Cost / Gen"},
                         hover_data=["generation_count", "trace_count"])
            fig.update_layout(height=380, xaxis_tickangle=-30,
                              margin=dict(l=0, r=0, t=10, b=80))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Per-Trace Cost Distribution")
            dist = cost_per_trace_distribution(df)
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.histogram(dist, x="total_cost", nbins=40,
                                    color_discrete_sequence=["#4F8EF7"],
                                    labels={"total_cost": "Cost per Trace (USD)"})
                fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.dataframe(
                    dist.head(20).style.format({"total_cost": "${:.5f}"}),
                    use_container_width=True, hide_index=True,
                )

    # ── Tab 2: By Model ────────────────────────────────────────────────
    with tab2:
        mdl = cost_by_model(df)
        if not mdl.empty:
            fig = px.bar(
                mdl, x="model",
                y=["input_cost", "output_cost", "cache_read_cost"],
                barmode="stack",
                color_discrete_map={
                    "input_cost": "#4F8EF7",
                    "output_cost": "#F4845F",
                    "cache_read_cost": "#2ECC71"
                },
                labels={"value": "Cost (USD)", "variable": "Cost Type"}
            )
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                mdl.style.format({
                    "total_cost":        "${:.5f}",
                    "input_cost":        "${:.5f}",
                    "output_cost":       "${:.5f}",
                    "cache_read_cost":   "${:.5f}",
                    "avg_input_tokens":  "{:.0f}",
                    "avg_output_tokens": "{:.0f}",
                }),
                use_container_width=True, hide_index=True,
            )

    # ── Tab 3: Routing Opportunities ───────────────────────────────────
    with tab3:
        st.caption(
            "Workflows using expensive models (Opus, GPT-4o) but producing "
            "short outputs (<200 tokens) are candidates for model downgrade. "
            "A cheaper model costs 5-20× less per token."
        )
        opp = routing_opportunity(df)
        if not opp.empty:
            flagged = opp[opp["flag"]].copy()
            if not flagged.empty:
                st.warning(f"⚠️ {len(flagged)} workflow/model combinations flagged for potential routing optimization")
                st.dataframe(
                    flagged.style.format({
                        "total_cost":        "${:.5f}",
                        "avg_output_tokens": "{:.0f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("✅ No obvious model routing inefficiencies detected.")
        else:
            st.info("Insufficient data for routing analysis.")
