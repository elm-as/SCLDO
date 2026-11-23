"""Standalone HTML5 + Plotly light editorial dashboard for SCLDO."""

import json
from pathlib import Path

from src.core.contracts import ScldoReportPayload
from src.visualization.results_exporter import serialize_scldo_payload

_DASHBOARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #F8FAFC; color: #0F172A; line-height: 1.45; padding: 20px;
}
.container { max-width: 1380px; margin: 0 auto; }
.header {
  background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #0F2942;
  border-radius: 6px; padding: 18px 22px; margin-bottom: 16px;
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
}
.header h1 { font-size: 1.28rem; font-weight: 700; color: #0F2942; }
.header p { font-size: 0.87rem; color: #475569; margin-top: 3px; }
.meta-strip { display: flex; gap: 8px; flex-wrap: wrap; }
.meta-tag {
  font-size: 0.78rem; padding: 4px 10px; border-radius: 4px;
  border: 1px solid #CBD5E1; background: #F8FAFC; color: #0F2942; font-weight: 600;
}
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
  gap: 12px; margin-bottom: 16px;
}
.kpi-card {
  background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px 16px;
}
.kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: #475569; font-weight: 600; }
.kpi-val {
  font-size: 1.48rem; font-weight: 700; color: #0F2942; margin: 4px 0;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
  font-family: "JetBrains Mono", "Consolas", ui-monospace, monospace;
}
.kpi-sub { font-size: 0.78rem; color: #475569; }
.accent-rust { color: #C2410C; }
.accent-teal { color: #0D9488; }
.accent-ochre { color: #B45309; }
.chart-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
  gap: 14px; margin-bottom: 16px;
}
.panel {
  background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px 16px;
  margin-bottom: 14px;
}
.panel-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px; gap: 10px; flex-wrap: wrap;
}
.panel-title { font-size: 0.95rem; font-weight: 700; color: #0F2942; }
select {
  border: 1px solid #CBD5E1; border-radius: 4px; padding: 4px 8px;
  font-size: 0.82rem; background: #FFFFFF; color: #0F172A;
}
.table-wrap { overflow-x: auto; margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
th, td { padding: 7px 10px; border-bottom: 1px solid #E2E8F0; text-align: left; }
th { background: #F8FAFC; color: #334155; font-weight: 600; border-top: 1px solid #CBD5E1; }
.num, td.num, th.num {
  text-align: right; font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
  font-family: "JetBrains Mono", "Consolas", ui-monospace, monospace;
}
"""

_DASHBOARD_JS = """
const DATA = __SCLDO_JSON__;
const layoutBase = {
  template: "plotly_white", paper_bgcolor: "#FFFFFF", plot_bgcolor: "#FFFFFF",
  margin: { l: 58, r: 22, t: 24, b: 48 }, font: { family: "Inter, sans-serif", color: "#0F172A", size: 11 }
};

function renderBenchmark() {
  const metric = document.getElementById("bench-metric").value;
  const iso = DATA.temporal_benchmark.records.filter(r => r.calibration_method === "isotonic");
  const labels = iso.map(r => (r.regime === "post_carrier_leaky" ? "Leaky: " : "Clean: ") + r.model_name);
  const colors = iso.map(r => r.regime === "post_carrier_leaky" ? "#C2410C" : (r.model_name === DATA.temporal_benchmark.best_clean_model_name ? "#0D9488" : "#0F2942"));
  const traces = [
    { x: labels, y: iso.map(r => r.cv_metrics[metric]), name: "5-Fold Expanding CV", type: "bar", marker: { color: "#94A3B8" } },
    { x: labels, y: iso.map(r => r.oot_metrics[metric]), name: "Strict OOT Holdout", type: "bar", marker: { color: colors } }
  ];
  Plotly.newPlot("plot-benchmark", traces, {
    ...layoutBase, barmode: "group", yaxis: { title: metric.toUpperCase(), gridcolor: "#E2E8F0" },
    legend: { orientation: "h", y: 1.13 }
  }, { responsive: true });
}

function renderCausal() {
  const buckets = DATA.causal_review_impact.buckets;
  const labels = buckets.map(b => b.bucket_label);
  const traces = [
    { x: labels, y: buckets.map(b => b.unadjusted_one_star_rate * 100), name: "Unadjusted 1-Star %", type: "bar", marker: { color: "#94A3B8" } },
    {
      x: labels, y: buckets.map(b => b.adjusted_one_star_rate * 100), name: "Adjusted Standardized Risk %", type: "bar", marker: { color: "#0F2942" },
      error_y: {
        type: "data", symmetric: false,
        array: buckets.map(b => (b.adjusted_rate_ci_high - b.adjusted_one_star_rate) * 100),
        arrayminus: buckets.map(b => (b.adjusted_one_star_rate - b.adjusted_rate_ci_low) * 100)
      }
    },
    { x: labels, y: buckets.map(b => b.ipw_one_star_rate * 100), name: "Stabilized IPW %", mode: "markers", marker: { color: "#C2410C", size: 9, symbol: "diamond" } }
  ];
  Plotly.newPlot("plot-causal", traces, {
    ...layoutBase, barmode: "group", yaxis: { title: "1-Star Review Rate (%)", range: [0, 82], gridcolor: "#E2E8F0" },
    legend: { orientation: "h", y: 1.13 }
  }, { responsive: true });
}

function renderShap() {
  const drivers = [...DATA.shap_drivers.drivers].reverse();
  Plotly.newPlot("plot-shap", [{
    y: drivers.map(d => d.feature_name),
    x: drivers.map(d => d.mean_abs_shap),
    type: "bar", orientation: "h",
    marker: { color: drivers.map(d => d.direction_corr >= 0 ? "#C2410C" : "#0D9488") }
  }], {
    ...layoutBase, margin: { l: 195, r: 22, t: 18, b: 44 },
    xaxis: { title: "Mean |TreeSHAP Value| (OOT Holdout)", gridcolor: "#E2E8F0" }
  }, { responsive: true });
}

function renderSla() {
  const sweep = DATA.sla_policy.sweep;
  const opt = DATA.sla_policy.empirical_optimal_policy;
  const traces = [
    { x: sweep.map(p => p.threshold), y: sweep.map(p => p.expected_cost_per_order), name: "Expected SLA Cost/Order", mode: "lines", line: { color: "#0F2942", width: 2.5 } },
    { x: sweep.map(p => p.threshold), y: sweep.map(p => p.recall * 10), name: "Recall x 10 (Scale)", mode: "lines", line: { color: "#0D9488", width: 1.8, dash: "dot" } }
  ];
  Plotly.newPlot("plot-sla", traces, {
    ...layoutBase, xaxis: { title: "Intervention Threshold (tau)", gridcolor: "#E2E8F0" },
    yaxis: { title: "Expected Cost / Order (8:1 units)", gridcolor: "#E2E8F0" },
    shapes: [{ type: "line", x0: opt.threshold, x1: opt.threshold, y0: 0, y1: DATA.sla_policy.no_action_cost_per_order * 1.15, line: { color: "#C2410C", dash: "dash", width: 1.6 } }],
    legend: { orientation: "h", y: 1.13 }
  }, { responsive: true });
}

function populateTables() {
  const benchBody = document.getElementById("bench-table-body");
  benchBody.innerHTML = DATA.temporal_benchmark.records.map(r => `<tr>
    <td>${r.regime}</td><td>${r.model_name}</td><td>${r.calibration_method}</td>
    <td class="num">${r.cv_metrics.roc_auc.toFixed(4)}</td><td class="num">${r.oot_metrics.roc_auc.toFixed(4)}</td>
    <td class="num">${r.oot_metrics.pr_auc.toFixed(4)}</td><td class="num">${r.oot_metrics.brier_score.toFixed(4)}</td>
    <td class="num">${r.oot_metrics.ece_10.toFixed(4)}</td>
  </tr>`).join("");

  const causalBody = document.getElementById("causal-table-body");
  causalBody.innerHTML = DATA.causal_review_impact.buckets.map(b => `<tr>
    <td>${b.bucket_label}</td><td class="num">${b.sample_count.toLocaleString()}</td>
    <td class="num">${b.mean_review_score.toFixed(2)}</td>
    <td class="num">${(b.unadjusted_one_star_rate * 100).toFixed(2)}%</td>
    <td class="num">${(b.adjusted_one_star_rate * 100).toFixed(2)}%</td>
    <td class="num">[${(b.adjusted_rate_ci_low * 100).toFixed(2)}%, ${(b.adjusted_rate_ci_high * 100).toFixed(2)}%]</td>
    <td class="num">+${(b.risk_diff_vs_ontime * 100).toFixed(2)} pp</td>
    <td class="num">${(b.ipw_one_star_rate * 100).toFixed(2)}%</td>
  </tr>`).join("");
}

document.getElementById("bench-metric").addEventListener("change", renderBenchmark);
renderBenchmark(); renderCausal(); renderShap(); renderSla(); populateTables();
"""


def _build_kpi_cards_html(payload: ScldoReportPayload) -> str:
    audit = payload.audit
    best_rec = next(
        rec
        for rec in payload.benchmark.records
        if rec.regime == "pre_dispatch_clean"
        and rec.model_name == payload.benchmark.best_clean_model_name
        and rec.calibration_method == payload.benchmark.best_clean_calibration
    )
    leaky_rec = next(
        rec
        for rec in payload.benchmark.records
        if rec.regime == "post_carrier_leaky" and rec.calibration_method == "isotonic"
    )
    opt = payload.sla_policy.empirical_optimal_policy
    severe = payload.causal_impact.buckets[-1]
    cards = [
        ("Delivered Orders", f"{audit.valid_delivered_orders:,}", f"Late rate {audit.overall_late_rate*100:.2f}% ({audit.overall_late_count:,})", ""),
        ("OOT Clean ROC-AUC", f"{best_rec.oot_metrics.roc_auc:.4f}", f"{best_rec.model_name} (ECE={best_rec.oot_metrics.ece_10:.4f})", "accent-teal"),
        ("Carrier Leakage Illusion", f"{leaky_rec.oot_metrics.roc_auc:.4f}", f"+{(leaky_rec.oot_metrics.roc_auc - best_rec.oot_metrics.roc_auc):.4f} AUC from post-dispatch leak", "accent-rust"),
        ("Causal Lift (>7d Late)", f"+{severe.risk_diff_vs_ontime*100:.1f} pp", f"Adjusted 1-star risk {severe.adjusted_one_star_rate*100:.1f}%", "accent-rust"),
        ("Optimal SLA Threshold", f"{opt.threshold:.3f}", f"Recall {opt.recall*100:.1f}% | Flagged {opt.flagged_rate*100:.1f}%", "accent-ochre"),
        ("SLA Savings / 10k Orders", f"{opt.savings_vs_no_action_per_10k:,.0f}", f"Cost/order {opt.expected_cost_per_order:.2f} vs {payload.sla_policy.no_action_cost_per_order:.2f} no-action", "accent-teal"),
    ]
    return "\n".join(
        f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-val {cls}">{val}</div><div class="kpi-sub">{sub}</div></div>'
        for lbl, val, sub, cls in cards
    )


def build_scldo_dashboard_html(
    payload: ScldoReportPayload, output_path: Path
) -> Path:
    """Build and write the standalone HTML5 + Plotly SCLDO editorial dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(serialize_scldo_payload(payload), ensure_ascii=False)
    kpi_html = _build_kpi_cards_html(payload)
    script = _DASHBOARD_JS.replace("__SCLDO_JSON__", serialized)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SCLDO - Supply Chain Late-Delivery &amp; SLA Risk Intelligence</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>{_DASHBOARD_CSS}</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>SCLDO - Supply Chain Late-Delivery &amp; SLA Risk Intelligence</h1>
        <p>Olist Brazilian E-Commerce (9 Relational Tables) · Strict Temporal Validation · Causal Review Econometrics · 8:1 SLA Policy</p>
      </div>
      <div class="meta-strip">
        <span class="meta-tag">OOT Cutoff: {payload.audit.oot_cutoff_iso[:10]}</span>
        <span class="meta-tag">5-Fold Expanding CV</span>
        <span class="meta-tag">ELMAS.md Compliant</span>
      </div>
    </div>
    <div class="kpi-grid">{kpi_html}</div>
    <div class="chart-grid">
      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">Temporal Benchmark &amp; Carrier Leakage Ablation</span>
          <select id="bench-metric">
            <option value="roc_auc">ROC-AUC</option>
            <option value="pr_auc">PR-AUC</option>
            <option value="brier_score">Brier Score</option>
            <option value="ece_10">ECE-10</option>
          </select>
        </div>
        <div id="plot-benchmark" style="height:360px;"></div>
      </div>
      <div class="panel">
        <div class="panel-head"><span class="panel-title">Causal Delay Impact on 1-Star Reviews (95% Bootstrap CI)</span></div>
        <div id="plot-causal" style="height:360px;"></div>
      </div>
      <div class="panel">
        <div class="panel-head"><span class="panel-title">Top Pre-Dispatch Delay Risk Drivers (TreeSHAP)</span></div>
        <div id="plot-shap" style="height:390px;"></div>
      </div>
      <div class="panel">
        <div class="panel-head"><span class="panel-title">Asymmetric 8:1 Proactive SLA Cost Curve</span></div>
        <div id="plot-sla" style="height:390px;"></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Model Benchmark Table (5-Fold Expanding CV vs Strict OOT Holdout)</span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Regime</th><th>Model</th><th>Calibration</th><th class="num">CV ROC-AUC</th><th class="num">OOT ROC-AUC</th><th class="num">OOT PR-AUC</th><th class="num">OOT Brier</th><th class="num">OOT ECE-10</th></tr></thead>
          <tbody id="bench-table-body"></tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Causal Econometric Summary: Delay Bucket vs 1-Star Review Risk</span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Delay Bucket</th><th class="num">Orders</th><th class="num">Mean Score</th><th class="num">Unadjusted 1-Star</th><th class="num">Adjusted Risk</th><th class="num">95% Bootstrap CI</th><th class="num">Causal Lift</th><th class="num">IPW Check</th></tr></thead>
          <tbody id="causal-table-body"></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>{script}</script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
