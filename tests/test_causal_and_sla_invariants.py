"""Tests for causal delay-to-review invariants, calibration metrics, and SLA policy."""

import numpy as np
import pandas as pd

from src.core.config import SlaEconomicsConfig
from src.evaluation.calibration_metrics import compute_calibration_metrics
from src.evaluation.causal_review_impact import estimate_causal_delay_review_impact
from src.evaluation.sla_policy_optimizer import optimize_sla_intervention_policy


def test_calibration_metrics_perfect_vs_miscalibrated() -> None:
    """Well-calibrated probabilities must yield lower Brier and ECE than constant shift."""
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.05, 0.95, size=2000)
    y_true = (rng.uniform(0.0, 1.0, size=2000) < probs).astype(int)

    good_metrics, bins = compute_calibration_metrics(y_true, probs, n_bins=10)
    bad_metrics, _ = compute_calibration_metrics(
        y_true, np.clip(probs + 0.30, 0.01, 0.99), n_bins=10
    )
    assert len(bins) == 10
    assert good_metrics.ece_10 < bad_metrics.ece_10
    assert good_metrics.brier_score < bad_metrics.brier_score
    assert 0.5 < good_metrics.roc_auc <= 1.0


def test_causal_review_impact_recovers_monotonic_delay_effect() -> None:
    """Synthetic orders with increasing delay penalty must yield monotonic adjusted lift."""
    rng = np.random.default_rng(123)
    n_rows = 2400
    delay_days = rng.choice([-4.0, 2.0, 5.5, 12.0], size=n_rows, p=[0.6, 0.15, 0.12, 0.13])
    price = rng.lognormal(mean=4.5, sigma=0.6, size=n_rows)
    distance = rng.uniform(50.0, 1500.0, size=n_rows)

    logits = (
        -2.6
        + 0.08 * np.log1p(price)
        + 0.7 * (delay_days > 0)
        + 1.6 * (delay_days > 3)
        + 1.0 * (delay_days > 7)
    )
    p_one_star = 1.0 / (1.0 + np.exp(-logits))
    is_one_star = (rng.uniform(size=n_rows) < p_one_star).astype(int)
    review_score = np.where(is_one_star == 1, 1.0, 5.0)

    synthetic = pd.DataFrame(
        {
            "delay_days": delay_days,
            "review_score": review_score,
            "total_price": price,
            "freight_ratio": rng.uniform(0.05, 0.4, size=n_rows),
            "distance_km": distance,
            "promised_lead_days": rng.uniform(10.0, 30.0, size=n_rows),
            "item_count": np.ones(n_rows),
            "is_same_state": rng.integers(0, 2, size=n_rows).astype(float),
            "customer_state": rng.choice(["SP", "RJ", "MG"], size=n_rows),
            "product_category_english": rng.choice(["health_beauty", "watches_gifts"], size=n_rows),
            "order_purchase_timestamp": pd.date_range("2018-01-01", periods=n_rows, freq="h"),
        }
    )

    result = estimate_causal_delay_review_impact(
        synthetic, bootstrap_reps=25, random_state=42
    )
    rates = [b.adjusted_one_star_rate for b in result.buckets]
    assert rates[0] < rates[1] < rates[2] < rates[3]
    for bucket in result.buckets:
        assert bucket.adjusted_rate_ci_low <= bucket.adjusted_rate_ci_high
    assert result.buckets[3].risk_diff_ci_low > 0.30


def test_sla_policy_optimizer_beats_default_and_no_action_under_8_to_1() -> None:
    """Under 8:1 asymmetric costs, optimal threshold beats default 0.50 and no-action."""
    rng = np.random.default_rng(99)
    probs = rng.beta(1.2, 12.0, size=5000)
    y_true = (rng.uniform(size=5000) < probs).astype(int)
    cfg = SlaEconomicsConfig(
        proactive_intervention_cost=10.0, unmitigated_late_loss=80.0
    )

    summary = optimize_sla_intervention_policy(y_true, probs, cfg)
    assert np.isclose(summary.theoretical_bayes_threshold, 10.0 / 80.0)
    assert summary.empirical_optimal_policy.expected_cost_per_order <= (
        summary.default_policy.expected_cost_per_order
    )
    assert summary.empirical_optimal_policy.expected_cost_per_order < (
        summary.no_action_cost_per_order
    )
    assert summary.empirical_optimal_policy.savings_vs_no_action_per_10k > 0.0
