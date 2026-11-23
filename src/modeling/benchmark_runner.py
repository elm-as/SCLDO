"""Benchmark orchestrator comparing clean pre-dispatch models vs post-carrier ablation."""

import pandas as pd

from src.core.config import PipelineConfig
from src.core.contracts import ModelEvaluationRecord, TemporalBenchmarkResult
from src.features.supply_chain_preprocessor import CLEAN_REGIME, LEAKY_REGIME
from src.features.temporal_splitter import split_chronological_oot
from src.modeling.temporal_calibrator import (
    TemporalRunSpec,
    evaluate_temporal_model,
)

_CLEAN_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "hist_gradient_boosting",
    "lightgbm",
)


def run_temporal_benchmark_and_ablation(
    orders_df: pd.DataFrame, config: PipelineConfig
) -> TemporalBenchmarkResult:
    """Run clean pre-dispatch temporal benchmark and post-carrier leakage ablation.

    Evaluates LogisticRegression, HistGradientBoosting, and LightGBM under the strict
    pre-dispatch regime, then runs a LightGBM post-carrier ablation to quantify the
    metric inflation caused by leaking order_delivered_carrier_date.
    """
    split = split_chronological_oot(orders_df, config.temporal)
    records: list[ModelEvaluationRecord] = []
    clean_lgbm_outcome = None

    for model_name in _CLEAN_MODELS:
        spec = TemporalRunSpec(
            model_name=model_name, regime=CLEAN_REGIME, config=config
        )
        outcome = evaluate_temporal_model(split, spec)
        records.extend([outcome.raw_record, outcome.isotonic_record])
        if model_name == "lightgbm":
            clean_lgbm_outcome = outcome

    ablation_spec = TemporalRunSpec(
        model_name="lightgbm", regime=LEAKY_REGIME, config=config
    )
    ablation_outcome = evaluate_temporal_model(split, ablation_spec)
    records.extend([ablation_outcome.raw_record, ablation_outcome.isotonic_record])

    clean_iso_records = [
        rec
        for rec in records
        if rec.regime == CLEAN_REGIME and rec.calibration_method == "isotonic"
    ]
    best_clean = max(clean_iso_records, key=lambda rec: rec.oot_metrics.roc_auc)
    if clean_lgbm_outcome is None:
        raise RuntimeError("Clean LightGBM outcome was not produced.")

    return TemporalBenchmarkResult(
        records=tuple(records),
        best_clean_model_name=best_clean.model_name,
        best_clean_calibration=best_clean.calibration_method,
        cv_fold_summaries=clean_lgbm_outcome.fold_summaries,
        clean_lgbm_model=clean_lgbm_outcome.fitted_oot_model,
        clean_feature_names=clean_lgbm_outcome.feature_names,
        oot_transformed_matrix=clean_lgbm_outcome.oot_matrix,
        oot_y_true=split.oot_test_df[config.schema.target_col].to_numpy(dtype=int),
    )
