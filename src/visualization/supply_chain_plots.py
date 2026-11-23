"""Publication-grade 300 DPI benchmark, SHAP, causal, and SLA figure orchestrator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.contracts import ModelEvaluationRecord, ScldoReportPayload
from src.visualization.economics_plots import (
    render_causal_reviews_figure,
    render_sla_policy_figure,
)

_PALETTE = {
    "navy": "#0F2942",
    "rust": "#C2410C",
    "teal": "#0D9488",
    "slate": "#334155",
    "grid": "#E2E8F0",
}


def _configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": _PALETTE["grid"],
            "grid.linewidth": 0.65,
            "font.size": 9.5,
        }
    )


def _find_record(
    records: tuple[ModelEvaluationRecord, ...], key: tuple[str, str, str]
) -> ModelEvaluationRecord:
    regime, model_name, cal_method = key
    return next(
        rec
        for rec in records
        if rec.regime == regime
        and rec.model_name == model_name
        and rec.calibration_method == cal_method
    )


def _render_temporal_benchmark_figure(
    payload: ScldoReportPayload, output_path: Path
) -> Path:
    fig, (ax_bar, ax_rel) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    recs = payload.benchmark.records
    specs = [
        ("pre_dispatch_clean", "logistic_regression", "isotonic", "Logistic\n(Clean)"),
        ("pre_dispatch_clean", "hist_gradient_boosting", "isotonic", "HistGB\n(Clean)"),
        ("pre_dispatch_clean", "lightgbm", "isotonic", "LightGBM\n(Clean)"),
        ("post_carrier_leaky", "lightgbm", "isotonic", "LightGBM\n(Leaky Carrier)"),
    ]
    labels = [s[3] for s in specs]
    aucs = [_find_record(recs, s[:3]).oot_metrics.roc_auc for s in specs]
    prs = [_find_record(recs, s[:3]).oot_metrics.pr_auc for s in specs]

    x_pos = np.arange(len(labels))
    width = 0.36
    bars_auc = ax_bar.bar(
        x_pos - width / 2,
        aucs,
        width,
        color=[_PALETTE["navy"], _PALETTE["navy"], _PALETTE["teal"], _PALETTE["rust"]],
        label="OOT ROC-AUC",
    )
    bars_pr = ax_bar.bar(
        x_pos + width / 2,
        prs,
        width,
        color=["#64748B", "#64748B", "#5EEAD4", "#FDBA74"],
        edgecolor=_PALETTE["slate"],
        linewidth=0.6,
        label="OOT PR-AUC",
    )
    for bar in list(bars_auc) + list(bars_pr):
        val = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_ylabel("Out-of-Time Holdout Score")
    ax_bar.set_title(
        "OOT Temporal Benchmark: Clean Pre-Dispatch vs Carrier Leakage",
        fontweight="bold",
        color=_PALETTE["navy"],
    )
    ax_bar.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    best_name = payload.benchmark.best_clean_model_name
    raw_rec = _find_record(recs, ("pre_dispatch_clean", best_name, "raw"))
    iso_rec = _find_record(recs, ("pre_dispatch_clean", best_name, "isotonic"))
    ax_rel.plot(
        [0, 1], [0, 1], linestyle="--", color=_PALETTE["navy"], lw=1.3, label="Ideal y = x"
    )
    for rec, color, marker, name in (
        (raw_rec, _PALETTE["rust"], "o", f"Raw (ECE={raw_rec.oot_metrics.ece_10:.4f})"),
        (
            iso_rec,
            _PALETTE["teal"],
            "^",
            f"OOF Isotonic (ECE={iso_rec.oot_metrics.ece_10:.4f})",
        ),
    ):
        xs = [b.mean_predicted for b in rec.oot_reliability_bins if b.sample_count > 0]
        ys = [b.empirical_rate for b in rec.oot_reliability_bins if b.sample_count > 0]
        ax_rel.plot(xs, ys, marker=marker, color=color, lw=2.0, markersize=5.5, label=name)
    ax_rel.set_xlim(0.0, 0.85)
    ax_rel.set_ylim(0.0, 0.85)
    ax_rel.set_xlabel("Mean Predicted Late-Delivery Probability")
    ax_rel.set_ylabel("Observed Late-Delivery Rate (OOT)")
    ax_rel.set_title(
        f"OOT Reliability Diagram ({best_name.replace('_', ' ').title()}, 10 Bins)",
        fontweight="bold",
        color=_PALETTE["navy"],
    )
    ax_rel.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_shap_drivers_figure(
    payload: ScldoReportPayload, output_path: Path
) -> Path:
    drivers = list(payload.shap_summary.drivers)[::-1]
    names = [d.feature_name for d in drivers]
    vals = [d.mean_abs_shap for d in drivers]
    colors = [
        _PALETTE["rust"] if d.direction_corr >= 0 else _PALETTE["teal"]
        for d in drivers
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, height=0.68, edgecolor="#FFFFFF")
    for bar, drv in zip(bars, drivers):
        sign = "+" if drv.direction_corr >= 0 else "-"
        ax.text(
            bar.get_width() + max(vals) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{drv.mean_abs_shap:.3f} ({sign})",
            va="center",
            fontsize=8.2,
            color=_PALETTE["slate"],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.8)
    ax.set_xlim(0.0, max(vals) * 1.22)
    ax.set_xlabel("Mean |TreeSHAP Value| (Log-Odds Impact on Late Delivery)")
    ax.set_title(
        f"Top Pre-Dispatch Late-Delivery Drivers (OOT n = {payload.shap_summary.sample_size:,})",
        fontweight="bold",
        color=_PALETTE["navy"],
    )
    ax.plot([], [], color=_PALETTE["rust"], lw=6, label="Higher value increases late risk (+)")
    ax.plot([], [], color=_PALETTE["teal"], lw=6, label="Higher value reduces late risk (-)")
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_all_scldo_figures(
    payload: ScldoReportPayload, images_dir: Path
) -> dict[str, Path]:
    """Render all four 300 DPI SCLDO figures into images_dir and return their paths."""
    _configure_matplotlib_style()
    images_dir.mkdir(parents=True, exist_ok=True)
    return {
        "temporal_benchmark": _render_temporal_benchmark_figure(
            payload, images_dir / "scldo_temporal_model_benchmark.png"
        ),
        "shap_drivers": _render_shap_drivers_figure(
            payload, images_dir / "scldo_shap_delay_drivers.png"
        ),
        "causal_reviews": render_causal_reviews_figure(
            payload, images_dir / "scldo_causal_delay_reviews.png"
        ),
        "sla_economics": render_sla_policy_figure(
            payload, images_dir / "scldo_sla_policy_economics.png"
        ),
    }
