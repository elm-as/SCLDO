"""Immutable configuration dataclasses for the SCLDO supply-chain pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout for raw Olist tables, artifacts, and figures."""

    project_root: Path
    raw_data_dir: Path
    artifacts_dir: Path
    images_dir: Path


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Chronological split and expanding cross-validation specification."""

    oot_cutoff_iso: str = "2018-04-30 23:59:59"
    outer_cv_splits: int = 5
    inner_cal_splits: int = 3
    seller_prior_weight: float = 10.0
    seller_prior_late_rate: float = 0.08


@dataclass(frozen=True)
class ModelHyperparamConfig:
    """Model hyperparameters supporting full runs and fast smoke tests."""

    random_state: int = 42
    lgbm_n_estimators: int = 320
    lgbm_learning_rate: float = 0.04
    lgbm_num_leaves: int = 48
    lgbm_min_child_samples: int = 40
    hgb_max_iter: int = 240
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 48
    logreg_max_iter: int = 500
    shap_sample_size: int = 4000
    causal_bootstrap_reps: int = 120

    @classmethod
    def fast_smoke(cls) -> "ModelHyperparamConfig":
        """Build lightweight hyperparameters for rapid smoke verification."""
        return cls(
            lgbm_n_estimators=40,
            hgb_max_iter=40,
            logreg_max_iter=150,
            shap_sample_size=600,
            causal_bootstrap_reps=25,
        )


@dataclass(frozen=True)
class SlaEconomicsConfig:
    """Asymmetric SLA intervention cost parameters (8:1 FN:FP ratio)."""

    proactive_intervention_cost: float = 10.0
    unmitigated_late_loss: float = 80.0
    threshold_grid_points: int = 79
    calibration_bins: int = 10


@dataclass(frozen=True)
class FeatureSchemaConfig:
    """Explicit separation between clean pre-dispatch and leaky post-carrier columns."""

    target_col: str = "is_late"
    timestamp_col: str = "order_purchase_timestamp"
    delay_days_col: str = "delay_days"
    review_score_col: str = "review_score"
    pre_dispatch_numeric: tuple[str, ...] = (
        "promised_lead_days",
        "shipping_limit_lead_days",
        "approval_lag_hours",
        "distance_km",
        "is_same_state",
        "item_count",
        "unique_product_count",
        "unique_seller_count",
        "total_price",
        "total_freight",
        "freight_ratio",
        "payment_value_sum",
        "max_installments",
        "payment_count",
        "product_weight_g",
        "product_volume_cm3",
        "product_photos_qty",
        "product_description_length",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "is_weekend_purchase",
        "is_q4_peak",
        "seller_prior_order_count",
        "seller_prior_resolved_count",
        "seller_past_late_rate",
        "seller_past_avg_delay_days",
    )
    pre_dispatch_categorical: tuple[str, ...] = (
        "customer_state",
        "seller_state",
        "product_category_english",
        "primary_payment_type",
    )
    post_carrier_leaky_numeric: tuple[str, ...] = (
        "carrier_handoff_days",
        "carrier_delay_vs_limit_days",
        "remaining_promised_days_after_carrier",
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level immutable configuration for the entire SCLDO study."""

    paths: PathConfig
    temporal: TemporalSplitConfig
    hyperparams: ModelHyperparamConfig
    economics: SlaEconomicsConfig
    schema: FeatureSchemaConfig

    @classmethod
    def create_default(cls, project_root: Path) -> "PipelineConfig":
        """Create standard SCLDO configuration rooted at project_root."""
        paths = PathConfig(
            project_root=project_root,
            raw_data_dir=project_root / "data" / "raw",
            artifacts_dir=project_root / "artifacts",
            images_dir=project_root / "images",
        )
        return cls(
            paths=paths,
            temporal=TemporalSplitConfig(),
            hyperparams=ModelHyperparamConfig(),
            economics=SlaEconomicsConfig(),
            schema=FeatureSchemaConfig(),
        )
