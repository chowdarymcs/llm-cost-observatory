"""
Static HTML Report Generator
=============================
Runs the full analysis pipeline and exports a single self-contained HTML file
with all charts embedded — no server, no dependencies to view.

Usage:
    # Demo data (no credentials needed)
    python generate_report.py --demo anomaly
    python generate_report.py --demo clean

    # Real Langfuse data
    python generate_report.py --source api    --days 30
    python generate_report.py --source clickhouse --days 30

    # Custom output path
    python generate_report.py --demo anomaly --output reports/my_report.html
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_html

from src.analysis.bloat_detector import compute_bloat_scores, bloat_summary_stats
from src.analysis.cache_analyzer import cache_kpis, daily_cache_trend, cache_by_workflow
from src.analysis.cost_analyzer import cost_kpis, daily_cost_trend, cost_by_workflow, cost_by_model
from src.analysis.recommendations import generate_recommendations, priority_matrix
from src.analysis.anomaly import detect_alerts, SEVERITY_EMOJI
from src.analysis.forecaster import build_forecast

PLOT_CFG = {"displayModeBar": False, "responsive": True}
EFFORT_COLOR = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}


# ══════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════
def _fig_html(fig, height=340) -> str:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=12),
    )
    return to_html(fig, include_plotlyjs=False, full_html=False, config=PLOT_CFG)


def chart_daily_cost(df) -> str:
    daily = daily_cost_trend(df)
    if daily.empty:
        return "<p>No data.</p>"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["total_cost"],
                         marker_color="#4F8EF7", name="Daily Cost"))
    fig.update_layout(title="Daily Spend", yaxis_title="Cost (USD)", xaxis_title="")
    return _fig_html(fig)


def chart_workflow_cost(df) -> str:
    wf = cost_by_workflow(df, top_n=12)
    if wf.empty:
        return "<p>No data.</p>"
    fig = px.bar(wf, x="total_cost", y="workflow", orientation="h",
                 color="total_cost", color_continuous_scale="Blues")
    fig.update_layout(title="Cost by Workflow", yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False, xaxis_title="Cost (USD)", yaxis_title="")
    return _fig_html(fig, height=380)


def chart_model_split(df) -> str:
    mdl = cost_by_model(df)
    if mdl.empty:
        return "<p>No data.</p>"
    fig = px.pie(mdl, values="total_cost", names="model", hole=0.45,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(title="Cost by Model", showlegend=False)
    return _fig_html(fig)


def chart_bloat_dist(bloat_df) -> str:
    if bloat_df.empty:
        return "<p>Insufficient multi-turn session data.</p>"
    fig = px.histogram(bloat_df, x="bloat_score", nbins=30, color="severity",
                       color_discrete_map={"Severe": "#E74C3C", "Moderate": "#F39C12",
                                            "Healthy": "#2ECC71"})
    fig.add_vline(x=2.0, line_dash="dash", line_color="#F39C12")
    fig.add_vline(x=5.0, line_dash="dash", line_color="#E74C3C")
    fig.update_layout(title="Context Bloat Score Distribution",
                      xaxis_title="Bloat Score (×)", yaxis_title="Sessions")
    return _fig_html(fig)


def chart_cache_trend(df) -> str:
    daily = daily_cache_trend(df)
    if daily.empty:
        return "<p>No cache data.</p>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["cache_hit_rate_pct"],
                             mode="lines+markers", line=dict(color="#2ECC71", width=2),
                             fill="tozeroy", fillcolor="rgba(46,204,113,0.15)"))
    fig.update_layout(title="Cache Hit Rate Over Time",
                      yaxis_title="Hit Rate (%)", xaxis_title="")
    return _fig_html(fig)


def chart_waterfall(fc) -> str:
    data = fc["waterfall_data"]
    if not data:
        return "<p>No optimization opportunities found.</p>"
    labels   = [d["label"][:38] for d in data]
    measures = ["absolute" if d["type"] == "total" else "relative" for d in data]
    values   = [d["value"] for d in data]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels, y=values,
        connector={"line": {"color": "#999"}},
        decreasing={"marker": {"color": "#2ECC71"}},
        increasing={"marker": {"color": "#E74C3C"}},
        totals={"marker": {"color": "#4F8EF7"}},
        textposition="outside", text=[f"${abs(v):.2f}" for v in values],
    ))
    fig.update_layout(title="30-Day Savings Forecast",
                      yaxis_title="Monthly Cost (USD)", xaxis_tickangle=-25,
                      showlegend=False)
    return _fig_html(fig, height=420)


def chart_priority_bubble(pm) -> str:
    if pm.empty:
        return "<p>No recommendations.</p>"
    pm = pm.copy()
    pm["Effort Num"] = pm["Effort"].map({"Low": 1, "Medium": 2, "High": 3})
    fig = px.scatter(pm, x="Effort Num", y="Monthly Savings", size="Monthly Savings",
                     color="Effort", text="Workflow",
                     color_discrete_map=EFFORT_COLOR)
    fig.update_xaxes(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"])
    fig.update_traces(textposition="top center", marker=dict(sizemin=8))
    fig.update_layout(title="Priority Matrix — Savings vs Effort",
                      xaxis_title="Implementation Effort",
                      yaxis_title="Monthly Savings (USD)", showlegend=False)
    return _fig_html(fig, height=380)


# ══════════════════════════════════════════════════════════════════════
# HTML assembly
# ══════════════════════════════════════════════════════════════════════
CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 32px 24px 80px; color: #1a1a2e;
       background: #fafbfc; line-height: 1.55; }
h1 { font-size: 2em; margin-bottom: 4px; }
h2 { margin-top: 48px; padding-bottom: 8px; border-bottom: 2px solid #e4e8ee;
     font-size: 1.4em; }
h3 { margin-top: 28px; font-size: 1.1em; }
.subtitle { color: #6b7280; margin-bottom: 28px; font-size: .95em; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px; margin: 22px 0; }
.kpi { background: #fff; border: 1px solid #e4e8ee; border-radius: 10px;
       padding: 18px; text-align: center; }
.kpi .v { font-size: 1.75em; font-weight: 700; color: #4F8EF7; }
.kpi .l { font-size: .78em; color: #6b7280; margin-top: 5px;
          text-transform: uppercase; letter-spacing: .4px; }
.kpi.danger .v { color: #E74C3C; }
.kpi.success .v { color: #2ECC71; }
.card { background: #fff; border: 1px solid #e4e8ee; border-radius: 10px;
        padding: 18px; margin: 16px 0; }
.alert { border-left: 4px solid; padding: 14px 18px; border-radius: 8px;
         margin-bottom: 12px; }
.alert.Critical { background: #fdeaea; border-color: #E74C3C; }
.alert.Warning  { background: #fff8e6; border-color: #F39C12; }
.alert.Info     { background: #eaf3fd; border-color: #4F8EF7; }
.alert .t { font-weight: 600; margin-bottom: 5px; }
.alert .d { font-size: .9em; color: #444; }
.alert .m { font-size: .8em; color: #777; margin-top: 7px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: .9em;
        background: #fff; border-radius: 8px; overflow: hidden; }
th { background: #4F8EF7; color: #fff; padding: 10px 12px; text-align: left;
     font-weight: 600; font-size: .85em; }
td { padding: 9px 12px; border-bottom: 1px solid #eef1f5; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) { background: #fafbfc; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-size: .78em; font-weight: 600; }
.badge.Low { background: #d4f4dd; color: #14663a; }
.badge.Medium { background: #fdf0d0; color: #7a5310; }
.badge.High { background: #fbdcdc; color: #8b1a1a; }
.rec { background: #fff; border: 1px solid #e4e8ee; border-radius: 10px;
       padding: 20px; margin: 18px 0; }
.rec-head { display: flex; justify-content: space-between; align-items: flex-start;
            gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.rec-title { font-size: 1.08em; font-weight: 650; }
.rec-save { font-size: 1.2em; font-weight: 700; color: #2ECC71; white-space: nowrap; }
.rec-meta { font-size: .82em; color: #6b7280; margin-bottom: 12px; }
.problem { background: #fff8e6; border-left: 3px solid #F39C12;
           padding: 11px 14px; border-radius: 6px; margin: 10px 0; font-size: .9em; }
.fix { background: #eaf3fd; border-left: 3px solid #4F8EF7;
       padding: 11px 14px; border-radius: 6px; margin: 10px 0; font-size: .9em; }
pre { background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px;
      overflow-x: auto; font-size: .82em; line-height: 1.5;
      font-family: 'SF Mono', Monaco, Consolas, monospace; }
.footer { margin-top: 60px; padding-top: 18px; border-top: 1px solid #e4e8ee;
          font-size: .82em; color: #9ca3af; text-align: center; }
.demo-banner { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
               color: #fff; padding: 13px 18px; border-radius: 8px;
               margin-bottom: 24px; font-size: .9em; }
"""


def build_html(df, scenario_note="") -> str:
    days = max((pd.to_datetime(df["timestamp"]).max()
                - pd.to_datetime(df["timestamp"]).min()).days, 1)

    bloat_df  = compute_bloat_scores(df)
    b_stats   = bloat_summary_stats(bloat_df)
    c_kpis    = cache_kpis(df)
    k         = cost_kpis(df)
    recs      = generate_recommendations(df, bloat_df, c_kpis, days)
    alerts    = detect_alerts(df, bloat_df, daily_cache_trend(df))
    fc        = build_forecast(df, recs, days)
    pm        = priority_matrix(recs)

    # ── Alerts HTML ────────────────────────────────────────────────────
    alerts_html = ""
    if alerts:
        for a in alerts:
            alerts_html += f"""
            <div class="alert {a.severity}">
              <div class="t">{SEVERITY_EMOJI.get(a.severity,'')} {a.severity} — {a.title}</div>
              <div class="d">{a.description}</div>
              <div class="m">Current: <strong>{a.current_value}</strong> ·
                   Baseline: {a.baseline_value} · Change: {a.change_pct:+.1f}%</div>
            </div>"""
    else:
        alerts_html = ('<div class="alert Info"><div class="t">✅ No anomalies detected</div>'
                       '<div class="d">Cost, bloat scores, and cache hit rates are stable '
                       'across the analysed period.</div></div>')

    # ── Priority table ─────────────────────────────────────────────────
    if not pm.empty:
        rows = ""
        for _, r in pm.iterrows():
            rows += (f"<tr><td>{r['Priority']}</td><td><code>{r['Workflow']}</code></td>"
                     f"<td>{r['Fix']}</td><td><strong>${r['Monthly Savings']:.2f}</strong></td>"
                     f"<td><span class='badge {r['Effort']}'>{r['Effort']}</span></td>"
                     f"<td>{r['Sessions/Day']:.1f}</td></tr>")
        priority_table = (
            "<table><tr><th>#</th><th>Workflow</th><th>Recommended Fix</th>"
            "<th>Monthly Savings</th><th>Effort</th><th>Sessions/Day</th></tr>"
            f"{rows}</table>"
        )
    else:
        priority_table = "<p>No optimization opportunities detected.</p>"

    # ── Detailed recommendations ───────────────────────────────────────
    recs_html = ""
    for i, r in enumerate(recs, 1):
        recs_html += f"""
        <div class="rec">
          <div class="rec-head">
            <div>
              <div class="rec-title">#{i} · {r.title}</div>
              <div class="rec-meta">Pattern: <code>{r.pattern}</code> ·
                   Workflow: <code>{r.workflow}</code> ·
                   <span class="badge {r.effort}">{r.effort} effort</span></div>
            </div>
            <div class="rec-save">${r.monthly_savings_usd:.2f}<span
                 style="font-size:.6em;color:#888">/mo</span></div>
          </div>
          <div class="problem"><strong>Problem:</strong> {r.problem}</div>
          <div class="fix"><strong>Fix:</strong> {r.fix}</div>
          <pre><code>{r.code_snippet.replace('<','&lt;').replace('>','&gt;')}</code></pre>
        </div>"""
    if not recs_html:
        recs_html = "<p>No recommendations — the system appears well optimised.</p>"

    banner = f'<div class="demo-banner">{scenario_note}</div>' if scenario_note else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LLM Cost Observatory — Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-3.7.0.min.js"></script>
<style>{CSS}</style></head><body>

<h1>🔭 LLM Cost Observatory</h1>
<div class="subtitle">Analysis report · {days}-day window ·
  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
{banner}

<h2>Executive Summary</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="v">${k['total_cost']:.2f}</div><div class="l">Total Spend</div></div>
  <div class="kpi"><div class="v">{k['total_traces']:,}</div><div class="l">Traces</div></div>
  <div class="kpi {'danger' if c_kpis['hit_rate']<30 else 'success'}">
      <div class="v">{c_kpis['hit_rate']:.1f}%</div><div class="l">Cache Hit Rate</div></div>
  <div class="kpi {'danger' if b_stats['avg_bloat_score']>2 else 'success'}">
      <div class="v">{b_stats['avg_bloat_score']:.1f}×</div><div class="l">Avg Bloat Score</div></div>
  <div class="kpi success"><div class="v">${fc['total_monthly_savings']:.2f}</div>
      <div class="l">Monthly Savings Available</div></div>
  <div class="kpi success"><div class="v">{fc['savings_pct']:.0f}%</div>
      <div class="l">Potential Reduction</div></div>
</div>

<h2>🚨 Active Alerts ({len(alerts)})</h2>
{alerts_html}

<h2>📉 Savings Forecast</h2>
<div class="card">{chart_waterfall(fc)}</div>
<p style="font-size:.88em;color:#6b7280">
  Current 30-day projection: <strong>${fc['monthly_projection']:.2f}</strong> →
  Optimised: <strong>${fc['optimized_monthly']:.2f}</strong>
  (saving <strong>${fc['total_monthly_savings']:.2f}/month</strong>).
  Estimates are conservative and capped per fix to avoid double-counting.</p>

<h2>📊 Optimization Priority Matrix</h2>
{priority_table}
<div class="card">{chart_priority_bubble(pm)}</div>

<h2>💰 Cost Analysis</h2>
<div class="card">{chart_daily_cost(df)}</div>
<div class="card">{chart_workflow_cost(df)}</div>
<div class="card">{chart_model_split(df)}</div>

<h2>🧠 Context Bloat</h2>
<div class="kpi-grid">
  <div class="kpi danger"><div class="v">{b_stats['severe_sessions']:,}</div>
       <div class="l">Severe Sessions (&gt;5×)</div></div>
  <div class="kpi"><div class="v">{b_stats['moderate_sessions']:,}</div>
       <div class="l">Moderate (2–5×)</div></div>
  <div class="kpi danger"><div class="v">${b_stats['total_waste_usd']:.2f}</div>
       <div class="l">Estimated Waste</div></div>
</div>
<div class="card">{chart_bloat_dist(bloat_df)}</div>

<h2>⚡ Cache Efficiency</h2>
<div class="card">{chart_cache_trend(df)}</div>

<h2>🔧 Detailed Recommendations</h2>
{recs_html}

<div class="footer">Generated by
  <a href="https://github.com/chowdarymcs/llm-cost-observatory">llm-cost-observatory</a>
</div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Generate a static LLM cost analysis report")
    ap.add_argument("--demo", choices=["clean", "anomaly"],
                    help="Use synthetic demo data instead of a live connection")
    ap.add_argument("--source", choices=["api", "clickhouse"], default="api",
                    help="Live data source (ignored when --demo is set)")
    ap.add_argument("--days", type=int, default=30, help="Lookback window in days")
    ap.add_argument("--output", default="outputs/llm_cost_report.html")
    args = ap.parse_args()

    end   = datetime.utcnow()
    start = end - timedelta(days=args.days)
    note  = ""

    if args.demo:
        from src.connectors.demo_connector import DemoConnector
        conn = DemoConnector(scenario=args.demo)
        note = ("🎭 <strong>Demo data</strong> — "
                + ("simulating an unoptimised system with injected context bloat, "
                   "cache misses, model over-spend, and a mid-period cost spike."
                   if args.demo == "anomaly" else
                   "simulating a well-optimised baseline system."))
        print(f"Generating report from demo data (scenario: {args.demo})…")
    elif args.source == "clickhouse":
        from src.connectors.clickhouse_connector import ClickHouseConnector
        conn = ClickHouseConnector()
        print("Fetching from ClickHouse…")
    else:
        from src.connectors.langfuse_api_connector import LangfuseAPIConnector
        conn = LangfuseAPIConnector()
        print("Fetching from Langfuse API…")

    df = conn.fetch_observations(start, end)
    if df.empty:
        print("No data returned for the selected window.", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {len(df):,} observations across {df['workflow'].nunique()} workflows…")
    html = build_html(df, note)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"\n✓ Report written → {out}  ({size_kb:.0f} KB)")
    print("  Open it in any browser — fully self-contained.")


if __name__ == "__main__":
    main()
