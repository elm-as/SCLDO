"""Typed immutable data contracts exchanged across SCLDO domain layers."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OlistRawTables:
    """Container holding all nine raw Olist relational CSV tables."""

    orders: pd.DataFrame
    order_items: pd.DataFrame
    order_payments: pd.DataFrame
    order_reviews: pd.DataFrame
    products: pd.DataFrame
    sellers: pd.DataFrame
    customers: pd.DataFrame
    geolocation: pd.DataFrame
    category_translation: pd.DataFrame


@dataclass(frozen=True)
class IngestionAuditStats:
    """Data integrity summary computed during relational ingestion."""

    total_raw_orders: int
    delivered_orders: int
    valid_delivered_orders: int
    orders_with_reviews: int
    overall_late_count: int
    overall_late_rate: float
    train_orders: int
    train_late_rate: float
    oot_test_orders: int
    oot_test_late_rate: float
    oot_cutoff_iso: str
    min_purchase_iso: str
    max_purchase_iso: str
    unique_sellers: int
    unique_customer_states: int
    valid_geo_zip_prefixes: int
    distance_coverage_rate: float


@dataclass(frozen=True)
class CalibrationMetrics:
    """Discrimination and calibration quality metrics for a binary risk model."""

    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    ece_10: float


@dataclass(frozen=True)
class ReliabilityBin:
    """Single bin of a uniform reliability diagram."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_predicted: float
    empirical_rate: float


@dataclass(frozen=True)
class ModelEvaluationRecord:
    """Evaluated predictions and metrics for one model, regime, and calibration."""

    regime: str
    model_name: str
    calibration_method: str
    cv_metrics: CalibrationMetrics
    oot_metrics: CalibrationMetrics
    oot_reliability_bins: tuple[ReliabilityBin, ...]
    oot_probabilities: np.ndarray


@dataclass(frozen=True)
class TemporalBenchmarkResult:
    """Full comparison across clean pre-dispatch models and post-carrier ablation."""

    records: tuple[ModelEvaluationRecord, ...]
    best_clean_model_name: str
    best_clean_calibration: str
    cv_fold_summaries: tuple[dict[str, Any], ...]
    clean_lgbm_model: Any
    clean_feature_names: tuple[str, ...]
    oot_transformed_matrix: np.ndarray
    oot_y_true: np.ndarray


@dataclass(frozen=True)
class ShapDriverRecord:
    """Global TreeSHAP importance and directional sign for one pre-dispatch feature."""

    feature_name: str
    mean_abs_shap: float
    direction_corr: float


@dataclass(frozen=True)
class ShapSummary:
    """Ranked TreeSHAP attribution report on the OOT holdout sample."""

    sample_size: int
    drivers: tuple[ShapDriverRecord, ...]


@dataclass(frozen=True)
class CausalBucketEstimate:
    """Adjusted and unadjusted 1-star review risk for a delivery delay bucket."""

    bucket_label: str
    sample_count: int
    mean_review_score: float
    unadjusted_one_star_rate: float
    adjusted_one_star_rate: float
    adjusted_rate_ci_low: float
    adjusted_rate_ci_high: float
    risk_diff_vs_ontime: float
    risk_diff_ci_low: float
    risk_diff_ci_high: float
    ipw_one_star_rate: float


@dataclass(frozen=True)
class CausalReviewImpactResult:
    """Complete econometric report quantifying delay impact on 1-star reviews."""

    sample_size: int
    bootstrap_replicates: int
    confounders_used: tuple[str, ...]
    buckets: tuple[CausalBucketEstimate, ...]


@dataclass(frozen=True)
class SlaThresholdPoint:
    """Operational and economic metrics at one SLA intervention threshold."""

    threshold: float
    recall: float
    precision: float
    specificity: float
    flagged_rate: float
    expected_cost_per_order: float
    cost_per_10k_orders: float
    savings_vs_no_action_per_10k: float


@dataclass(frozen=True)
class SlaPolicySummary:
    """Comparison of SLA intervention policies under 8:1 asymmetric costs."""

    cost_ratio_fn_to_fp: float
    proactive_cost: float
    unmitigated_loss: float
    theoretical_bayes_threshold: float
    no_action_cost_per_order: float
    default_policy: SlaThresholdPoint
    bayes_policy: SlaThresholdPoint
    empirical_optimal_policy: SlaThresholdPoint
    sweep: tuple[SlaThresholdPoint, ...]


@dataclass(frozen=True)
class ScldoReportPayload:
    """Master payload consumed by figure renderers, dashboard, and JSON exporter."""

    audit: IngestionAuditStats
    benchmark: TemporalBenchmarkResult
    shap_summary: ShapSummary
    causal_impact: CausalReviewImpactResult
    sla_policy: SlaPolicySummary
