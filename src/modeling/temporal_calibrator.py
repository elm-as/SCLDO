"""Expanding temporal CV with nested OOF Isotonic calibration and OOT evaluation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.core.config import PipelineConfig
from src.core.contracts import ModelEvaluationRecord
from src.evaluation.calibration_metrics import compute_calibration_metrics
from src.features.supply_chain_preprocessor import (
    build_supply_chain_preprocessor,
    extract_clean_feature_frame,
)
from src.features.temporal_splitter import (
    ChronologicalSplit,
    build_expanding_temporal_folds,
)
from src.modeling.estimator_factory import create_supply_chain_estimator


@dataclass(frozen=True)
class TemporalRunSpec:
    """Specification for evaluating one (model_name, regime) pair."""

    model_name: str
    regime: str
    config: PipelineConfig


@dataclass(frozen=True)
class TemporalModelOutcome:
    """Raw + calibrated evaluation records plus fitted OOT model artifacts."""

    raw_record: ModelEvaluationRecord
    isotonic_record: ModelEvaluationRecord
    fold_summaries: tuple[dict[str, Any], ...]
    fitted_oot_model: Any
    feature_names: tuple[str, ...]
    oot_matrix: np.ndarray


def _fit_and_predict_single_split(
    x_train_df: pd.DataFrame, x_val_df: pd.DataFrame, spec: TemporalRunSpec
) -> tuple[Any, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Preprocess strictly on train split, returning unfitted-model inputs."""
    scale_num = spec.model_name == "logistic_regression"
    preprocessor = build_supply_chain_preprocessor(
        spec.config.schema, spec.regime, scale_numeric=scale_num
    )
    x_tr = preprocessor.fit_transform(x_train_df)
    x_va = preprocessor.transform(x_val_df)
    names = tuple(str(n) for n in preprocessor.get_feature_names_out())
    return preprocessor, np.asarray(x_tr, dtype=float), np.asarray(x_va, dtype=float), names


def _fit_inner_isotonic_calibrator(
    x_sub_df: pd.DataFrame, y_sub: np.ndarray, spec: TemporalRunSpec
) -> IsotonicRegression:
    """Fit an Isotonic calibrator strictly on inner expanding temporal OOF scores."""
    inner_ts = pd.Series(np.arange(len(y_sub)))
    inner_folds = build_expanding_temporal_folds(
        inner_ts, spec.config.temporal.inner_cal_splits
    )
    oof_scores: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []

    for tr_idx, va_idx in inner_folds:
        _, x_tr, x_va, _ = _fit_and_predict_single_split(
            x_sub_df.iloc[tr_idx], x_sub_df.iloc[va_idx], spec
        )
        est = create_supply_chain_estimator(spec.model_name, spec.config.hyperparams)
        est.fit(x_tr, y_sub[tr_idx])
        oof_scores.append(est.predict_proba(x_va)[:, 1])
        oof_targets.append(y_sub[va_idx])

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.concatenate(oof_scores), np.concatenate(oof_targets))
    return calibrator


def _run_outer_temporal_cv(
    train_df: pd.DataFrame, x_train_all: pd.DataFrame, spec: TemporalRunSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    """Run 5-fold expanding temporal CV producing raw and nested-calibrated OOF scores."""
    y_train = train_df[spec.config.schema.target_col].to_numpy(dtype=int)
    folds = build_expanding_temporal_folds(
        train_df[spec.config.schema.timestamp_col],
        spec.config.temporal.outer_cv_splits,
    )
    raw_list: list[np.ndarray] = []
    iso_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []

    for fold_num, (tr_idx, va_idx) in enumerate(folds, start=1):
        _, x_tr, x_va, _ = _fit_and_predict_single_split(
            x_train_all.iloc[tr_idx], x_train_all.iloc[va_idx], spec
        )
        estimator = create_supply_chain_estimator(
            spec.model_name, spec.config.hyperparams
        )
        estimator.fit(x_tr, y_train[tr_idx])
        val_raw = estimator.predict_proba(x_va)[:, 1]

        calibrator = _fit_inner_isotonic_calibrator(
            x_train_all.iloc[tr_idx], y_train[tr_idx], spec
        )
        val_iso = calibrator.predict(val_raw)
        val_metrics, _ = compute_calibration_metrics(
            y_train[va_idx], val_iso, n_bins=spec.config.economics.calibration_bins
        )
        summaries.append(
            {
                "fold": fold_num,
                "train_size": int(len(tr_idx)),
                "val_size": int(len(va_idx)),
                "val_start": str(train_df[spec.config.schema.timestamp_col].iloc[va_idx[0]]),
                "val_end": str(train_df[spec.config.schema.timestamp_col].iloc[va_idx[-1]]),
                "val_late_rate": float(y_train[va_idx].mean()),
                "roc_auc": val_metrics.roc_auc,
                "pr_auc": val_metrics.pr_auc,
                "brier_score": val_metrics.brier_score,
                "ece_10": val_metrics.ece_10,
            }
        )
        raw_list.append(val_raw)
        iso_list.append(val_iso)
        target_list.append(y_train[va_idx])

    return (
        np.concatenate(target_list),
        np.concatenate(raw_list),
        np.concatenate(iso_list),
        tuple(summaries),
    )


def evaluate_temporal_model(
    split: ChronologicalSplit, spec: TemporalRunSpec
) -> TemporalModelOutcome:
    """Evaluate a model with expanding temporal CV, nested Isotonic calibration, and OOT test.

    Guarantees zero leakage from OOT holdout into both preprocessing and calibration:
    the OOT Isotonic calibrator is trained strictly on the outer temporal CV predictions.
    """
    x_train_all = extract_clean_feature_frame(
        split.train_df, spec.config.schema, spec.regime
    )
    x_oot_all = extract_clean_feature_frame(
        split.oot_test_df, spec.config.schema, spec.regime
    )
    y_train = split.train_df[spec.config.schema.target_col].to_numpy(dtype=int)
    y_oot = split.oot_test_df[spec.config.schema.target_col].to_numpy(dtype=int)

    cv_y, cv_raw, cv_iso, fold_summaries = _run_outer_temporal_cv(
        split.train_df, x_train_all, spec
    )
    _, x_tr_full, x_oot_mat, feature_names = _fit_and_predict_single_split(
        x_train_all, x_oot_all, spec
    )
    oot_estimator = create_supply_chain_estimator(
        spec.model_name, spec.config.hyperparams
    )
    oot_estimator.fit(x_tr_full, y_train)
    oot_raw = oot_estimator.predict_proba(x_oot_mat)[:, 1]

    # Calibrate OOT predictions strictly using historical outer-CV out-of-fold scores.
    oot_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    oot_calibrator.fit(cv_raw, cv_y)
    oot_iso = oot_calibrator.predict(oot_raw)

    n_bins = spec.config.economics.calibration_bins
    cv_raw_metrics, _ = compute_calibration_metrics(cv_y, cv_raw, n_bins=n_bins)
    cv_iso_metrics, _ = compute_calibration_metrics(cv_y, cv_iso, n_bins=n_bins)
    oot_raw_metrics, oot_raw_bins = compute_calibration_metrics(
        y_oot, oot_raw, n_bins=n_bins
    )
    oot_iso_metrics, oot_iso_bins = compute_calibration_metrics(
        y_oot, oot_iso, n_bins=n_bins
    )

    raw_record = ModelEvaluationRecord(
        regime=spec.regime,
        model_name=spec.model_name,
        calibration_method="raw",
        cv_metrics=cv_raw_metrics,
        oot_metrics=oot_raw_metrics,
        oot_reliability_bins=oot_raw_bins,
        oot_probabilities=oot_raw,
    )
    iso_record = ModelEvaluationRecord(
        regime=spec.regime,
        model_name=spec.model_name,
        calibration_method="isotonic",
        cv_metrics=cv_iso_metrics,
        oot_metrics=oot_iso_metrics,
        oot_reliability_bins=oot_iso_bins,
        oot_probabilities=oot_iso,
    )
    return TemporalModelOutcome(
        raw_record=raw_record,
        isotonic_record=iso_record,
        fold_summaries=fold_summaries,
        fitted_oot_model=oot_estimator,
        feature_names=feature_names,
        oot_matrix=x_oot_mat,
    )
