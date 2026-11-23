"""Tests verifying zero temporal leakage, geospatial distance, and clean regimes."""

import numpy as np
import pandas as pd

from src.core.config import FeatureSchemaConfig, TemporalSplitConfig
from src.features.supply_chain_preprocessor import (
    CLEAN_REGIME,
    LEAKY_REGIME,
    get_feature_columns_for_regime,
)
from src.features.temporal_splitter import (
    build_expanding_temporal_folds,
    split_chronological_oot,
)
from src.ingestion.geospatial_builder import CoordinatePairs, compute_haversine_km
from src.ingestion.seller_history_builder import build_strict_past_seller_features


def _make_synthetic_seller_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "primary_seller_id": ["s1", "s1", "s1", "s1"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-05", "2018-01-20", "2018-02-10"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-10", "2018-01-18", "2018-02-01", "2018-02-15"]
            ),
            "is_late": [1, 0, 1, 0],
            "delay_days": [5.0, -2.0, 4.0, -1.0],
        }
    )


def test_seller_history_has_zero_future_or_inflight_leakage() -> None:
    """Past-only seller rate must ignore in-flight orders and be invariant to future edits."""
    cfg = TemporalSplitConfig(seller_prior_weight=10.0, seller_prior_late_rate=0.08)
    base_df = _make_synthetic_seller_orders()
    enriched_base = build_strict_past_seller_features(base_df, cfg)

    # Order o2 (purchased Jan 5) happens BEFORE o1 is delivered (Jan 10): 0 resolved!
    assert enriched_base.loc[0, "seller_prior_resolved_count"] == 0.0
    assert enriched_base.loc[1, "seller_prior_resolved_count"] == 0.0
    # Order o3 (purchased Jan 20) sees o1 (deliv Jan 10, late=1) and o2 (deliv Jan 18, late=0).
    assert enriched_base.loc[2, "seller_prior_resolved_count"] == 2.0
    expected_rate_o3 = (1.0 + 10.0 * 0.08) / (2.0 + 10.0)
    assert np.isclose(enriched_base.loc[2, "seller_past_late_rate"], expected_rate_o3)

    # Mutating the final future row (o4) must NEVER change rows 0, 1, or 2.
    mutated = base_df.copy()
    mutated.loc[3, "is_late"] = 1
    mutated.loc[3, "delay_days"] = 99.0
    enriched_mut = build_strict_past_seller_features(mutated, cfg)
    assert np.allclose(
        enriched_base["seller_past_late_rate"].iloc[:3],
        enriched_mut["seller_past_late_rate"].iloc[:3],
    )


def test_clean_regime_excludes_carrier_and_review_columns() -> None:
    """Pre-dispatch clean regime must strictly omit post-carrier and review fields."""
    schema = FeatureSchemaConfig()
    clean_num, clean_cat = get_feature_columns_for_regime(schema, CLEAN_REGIME)
    leaky_num, _ = get_feature_columns_for_regime(schema, LEAKY_REGIME)
    clean_all = set(clean_num) | set(clean_cat)

    for forbidden in schema.post_carrier_leaky_numeric:
        assert forbidden not in clean_all
        assert forbidden in leaky_num
    assert "review_score" not in clean_all
    assert "delay_days" not in clean_all


def test_chronological_oot_and_expanding_folds_preserve_time_order() -> None:
    """Every train timestamp must precede OOT test and every CV validation fold."""
    dates = pd.date_range("2018-01-01", periods=120, freq="2D")
    frame = pd.DataFrame(
        {
            "order_id": [f"id_{i}" for i in range(120)],
            "order_purchase_timestamp": dates,
            "is_late": np.tile([0, 0, 0, 1], 30),
        }
    )
    cfg = TemporalSplitConfig(oot_cutoff_iso="2018-06-01 23:59:59", outer_cv_splits=5)
    split = split_chronological_oot(frame, cfg)
    assert split.train_df["order_purchase_timestamp"].max() < (
        split.oot_test_df["order_purchase_timestamp"].min()
    )

    folds = build_expanding_temporal_folds(
        split.train_df["order_purchase_timestamp"], n_splits=5
    )
    assert len(folds) == 5
    for tr_idx, va_idx in folds:
        assert tr_idx.max() < va_idx.min()


def test_haversine_distance_sao_paulo_to_rio() -> None:
    """Verify Haversine distance between Sao Paulo and Rio de Janeiro (~357 km)."""
    coords = CoordinatePairs(
        origin_lat=np.array([-23.5505]),
        origin_lng=np.array([-46.6333]),
        dest_lat=np.array([-22.9068]),
        dest_lng=np.array([-43.1729]),
    )
    dist_km = float(compute_haversine_km(coords)[0])
    assert 350.0 <= dist_km <= 365.0
