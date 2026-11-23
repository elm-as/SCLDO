"""CLI entrypoint orchestrating the end-to-end SCLDO supply-chain pipeline."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

from src.core.config import ModelHyperparamConfig, PipelineConfig
from src.core.contracts import ScldoReportPayload
from src.evaluation.causal_review_impact import estimate_causal_delay_review_impact
from src.evaluation.shap_delay_drivers import (
    ShapComputationInput,
    compute_shap_delay_drivers,
)
from src.evaluation.sla_policy_optimizer import optimize_sla_intervention_policy
from src.ingestion.olist_loader import load_olist_raw_tables
from src.ingestion.order_dataset_builder import build_olist_order_dataset
from src.modeling.benchmark_runner import run_temporal_benchmark_and_ablation
from src.visualization.results_exporter import export_results_json
from src.visualization.supply_chain_dashboard import build_scldo_dashboard_html
from src.visualization.supply_chain_plots import render_all_scldo_figures


def _find_best_clean_probs(payload_benchmark: object) -> object:
    return next(
        rec.oot_probabilities
        for rec in payload_benchmark.records
        if rec.regime == "pre_dispatch_clean"
        and rec.model_name == payload_benchmark.best_clean_model_name
        and rec.calibration_method == payload_benchmark.best_clean_calibration
    )


def run_scldo_study(config: PipelineConfig) -> ScldoReportPayload:
    """Execute ingestion, temporal CV/OOT modeling, causal econometrics, SHAP, and visuals."""
    start_time = time.perf_counter()
    print("[1/6] Loading and joining all 9 Olist relational CSV tables...", flush=True)
    raw_tables = load_olist_raw_tables(config.paths.raw_data_dir)
    orders_df, audit_stats = build_olist_order_dataset(raw_tables, config.temporal)
    print(
        f"      Valid delivered orders: {audit_stats.valid_delivered_orders:,} "
        f"(late rate: {audit_stats.overall_late_rate * 100:.2f}%)",
        flush=True,
    )

    print("[2/6] Running 5-fold expanding temporal CV, OOF Isotonic & leakage ablation...", flush=True)
    benchmark = run_temporal_benchmark_and_ablation(orders_df, config)

    print("[3/6] Computing TreeSHAP pre-dispatch risk drivers on OOT holdout...", flush=True)
    shap_input = ShapComputationInput(
        model=benchmark.clean_lgbm_model,
        feature_matrix=benchmark.oot_transformed_matrix,
        feature_names=benchmark.clean_feature_names,
        sample_size=config.hyperparams.shap_sample_size,
        random_state=config.hyperparams.random_state,
    )
    shap_summary = compute_shap_delay_drivers(shap_input, top_k=15)

    print("[4/6] Estimating causal delay -> 1-star review impact with bootstrap CIs...", flush=True)
    causal_impact = estimate_causal_delay_review_impact(
        orders_df,
        bootstrap_reps=config.hyperparams.causal_bootstrap_reps,
        random_state=config.hyperparams.random_state,
    )

    print("[5/6] Optimizing proactive SLA intervention policy (8:1 asymmetric cost)...", flush=True)
    best_oot_probs = _find_best_clean_probs(benchmark)
    sla_policy = optimize_sla_intervention_policy(
        benchmark.oot_y_true, best_oot_probs, config.economics
    )

    payload = ScldoReportPayload(
        audit=audit_stats,
        benchmark=benchmark,
        shap_summary=shap_summary,
        causal_impact=causal_impact,
        sla_policy=sla_policy,
    )

    print("[6/6] Rendering 300 DPI figures, Plotly dashboard, and results.json...", flush=True)
    render_all_scldo_figures(payload, config.paths.images_dir)
    build_scldo_dashboard_html(payload, config.paths.artifacts_dir / "dashboard.html")
    export_results_json(payload, config.paths.artifacts_dir / "results.json")

    elapsed = time.perf_counter() - start_time
    print(f"SCLDO pipeline completed cleanly in {elapsed:.1f}s.", flush=True)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCLDO: Supply Chain Late-Delivery & SLA Risk Intelligence"
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Run reduced tree and bootstrap counts for rapid smoke verification.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint constructing PipelineConfig and running the SCLDO study."""
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    config = PipelineConfig.create_default(project_root)
    if args.fast_smoke:
        config = replace(config, hyperparams=ModelHyperparamConfig.fast_smoke())
    run_scldo_study(config)


if __name__ == "__main__":
    main()
