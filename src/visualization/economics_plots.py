"""300 DPI causal review impact and 8:1 SLA policy economics figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.contracts import ScldoReportPayload

_PALETTE = {
    "navy": "#0F2942",
    "rust": "#C2410C",
    "teal": "#0D9488",
    "ochre": "#B45309",
    "muted": "#64748B",
}


def render_causal_reviews_figure(
    payload: ScldoReportPayload, output_path: Path
) -> Path:
    """Render 300 DPI adjusted 1-star review risk and causal lift figure."""
    buckets = payload.causal_impact.buckets
    labels = [b.bucket_label for b in buckets]
    x_pos = np.arange(len(labels))

    unadj = np.array([b.unadjusted_one_star_rate * 100.0 for b in buckets])
    adj = np.array([b.adjusted_one_star_rate * 100.0 for b in buckets])
    adj_low = np.maximum(adj - np.array([b.adjusted_rate_ci_low * 100.0 for b in buckets]), 0.0)
    adj_high = np.maximum(np.array([b.adjusted_rate_ci_high * 100.0 for b in buckets]) - adj, 0.0)

    diffs = np.array([b.risk_diff_vs_ontime * 100.0 for b in buckets[1:]])
    diff_low = np.maximum(diffs - np.array([b.risk_diff_ci_low * 100.0 for b in buckets[1:]]), 0.0)
    diff_high = np.maximum(np.array([b.risk_diff_ci_high * 100.0 for b in buckets[1:]]) - diffs, 0.0)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    width = 0.35
    ax_left.bar(x_pos - width / 2, unadj, width, color="#94A3B8", label="Unadjusted 1-Star Rate")
    ax_left.bar(
        x_pos + width / 2,
        adj,
        width,
        color=_PALETTE["navy"],
        yerr=[adj_low, adj_high],
        capsize=4,
        label="Adjusted Standardized Risk (95% CI)",
    )
    for idx, val in enumerate(adj):
        ax_left.text(
            idx + width / 2,
            val + 2.2,
            f"{val:.1f}%",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=_PALETTE["navy"],
        )
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylim(0.0, 82.0)
    ax_left.set_ylabel("1-Star Review Probability (%)")
    ax_left.set_title(
        f"Adjusted 1-Star Review Risk by Delay Severity (n = {payload.causal_impact.sample_size:,})",
        fontweight="bold",
        color=_PALETTE["navy"],
    )
    ax_left.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    x_late = np.arange(len(buckets) - 1)
    late_labels = [f"{b.bucket_label}\n(n={b.sample_count:,})" for b in buckets[1:]]
    ax_right.bar(
        x_late,
        diffs,
        width=0.52,
        color=[_PALETTE["ochre"], _PALETTE["rust"], "#7F1D1D"],
        yerr=[diff_low, diff_high],
        capsize=5,
    )
    for idx, val in enumerate(diffs):
        ax_right.text(
            idx,
            val + 2.2,
            f"+{val:.1f} pp",
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color=_PALETTE["rust"],
        )
    ax_right.set_xticks(x_late)
    ax_right.set_xticklabels(late_labels)
    ax_right.set_ylim(0.0, 75.0)
    ax_right.set_ylabel("Causal Risk Difference vs On-Time (pp)")
    ax_right.set_title(
        "Confounder-Adjusted Causal Lift in 1-Star Reviews (95% Bootstrap CI)",
        fontweight="bold",
        color=_PALETTE["navy"],
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_sla_policy_figure(
    payload: ScldoReportPayload, output_path: Path
) -> Path:
    """Render 300 DPI asymmetric 8:1 SLA intervention cost curve and operational tradeoffs."""
    sla = payload.sla_policy
    taus = [pt.threshold for pt in sla.sweep]
    costs = [pt.expected_cost_per_order for pt in sla.sweep]
    recalls = [pt.recall * 100.0 for pt in sla.sweep]
    precisions = [pt.precision * 100.0 for pt in sla.sweep]
    flagged = [pt.flagged_rate * 100.0 for pt in sla.sweep]

    fig, (ax_cost, ax_ops) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax_cost.plot(taus, costs, color=_PALETTE["navy"], lw=2.4, label="Expected SLA Cost / Order")
    ax_cost.axhline(
        sla.no_action_cost_per_order,
        linestyle="--",
        color=_PALETTE["muted"],
        lw=1.4,
        label=f"No Action Baseline ({sla.no_action_cost_per_order:.2f})",
    )
    opt = sla.empirical_optimal_policy
    default = sla.default_policy
    ax_cost.scatter(
        [opt.threshold],
        [opt.expected_cost_per_order],
        color=_PALETTE["teal"],
        s=75,
        zorder=5,
        label=f"Optimal tau={opt.threshold:.3f} (Cost={opt.expected_cost_per_order:.2f})",
    )
    ax_cost.scatter(
        [default.threshold],
        [default.expected_cost_per_order],
        color=_PALETTE["rust"],
        s=65,
        zorder=5,
        label=f"Default tau=0.50 (Cost={default.expected_cost_per_order:.2f})",
    )
    ax_cost.set_xlabel("Proactive Intervention Probability Threshold (tau)")
    ax_cost.set_ylabel("Expected SLA Cost per Order (R$ units, 8:1 ratio)")
    ax_cost.set_title(
        "Asymmetric 8:1 SLA Intervention Cost Curve (OOT Holdout)",
        fontweight="bold",
        color=_PALETTE["navy"],
    )
    ax_cost.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.3)

    ax_ops.plot(taus, recalls, color=_PALETTE["rust"], lw=2.2, label="Late-Delivery Recall (%)")
    ax_ops.plot(taus, precisions, color=_PALETTE["teal"], lw=2.2, label="Intervention Precision (%)")
    ax_ops.plot(taus, flagged, color=_PALETTE["ochre"], lw=1.9, linestyle=":", label="Orders Flagged (%)")
    ax_ops.axvline(
        opt.threshold,
        color=_PALETTE["navy"],
        linestyle="--",
        lw=1.4,
        label=f"Optimal tau* = {opt.threshold:.3f}",
    )
    ax_ops.set_xlabel("Proactive Intervention Probability Threshold (tau)")
    ax_ops.set_ylabel("Operational Rate (%)")
    ax_ops.set_ylim(0.0, 102.0)
    ax_ops.set_title(
        f"Operational Tradeoff (Saves {opt.savings_vs_no_action_per_10k:,.0f} R$/10k Orders)",
        fontweight="bold",
        color=_PALETTE["navy"],
    )
    ax_ops.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
