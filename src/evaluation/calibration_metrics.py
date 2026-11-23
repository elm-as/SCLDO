"""Discrimination and calibration evaluation metrics (ROC-AUC, PR-AUC, Brier, ECE-10)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.core.contracts import CalibrationMetrics, ReliabilityBin


def compute_reliability_bins(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Partition predicted probabilities into n_bins uniform reliability intervals."""
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []

    for idx in range(n_bins):
        low, high = float(edges[idx]), float(edges[idx + 1])
        mask = (
            (p_arr >= low) & (p_arr <= high)
            if idx == n_bins - 1
            else (p_arr >= low) & (p_arr < high)
        )
        count = int(mask.sum())
        mean_pred = float(p_arr[mask].mean()) if count > 0 else 0.0
        emp_rate = float(y_arr[mask].mean()) if count > 0 else 0.0
        bins.append(
            ReliabilityBin(
                bin_index=idx,
                lower_bound=low,
                upper_bound=high,
                sample_count=count,
                mean_predicted=mean_pred,
                empirical_rate=emp_rate,
            )
        )
    return tuple(bins)


def compute_ece_from_bins(bins: tuple[ReliabilityBin, ...]) -> float:
    """Compute sample-weighted Expected Calibration Error (ECE) from reliability bins."""
    total = sum(b.sample_count for b in bins)
    if total == 0:
        return 0.0
    weighted_gap = sum(
        b.sample_count * abs(b.mean_predicted - b.empirical_rate) for b in bins
    )
    return float(weighted_gap / total)


def compute_calibration_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> tuple[CalibrationMetrics, tuple[ReliabilityBin, ...]]:
    """Compute ROC-AUC, PR-AUC, Brier score, log-loss, ECE-10, and reliability bins."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    bins = compute_reliability_bins(y_arr, p_arr, n_bins=n_bins)
    metrics = CalibrationMetrics(
        roc_auc=float(roc_auc_score(y_arr, p_arr)),
        pr_auc=float(average_precision_score(y_arr, p_arr)),
        brier_score=float(brier_score_loss(y_arr, p_arr)),
        log_loss=float(log_loss(y_arr, p_arr)),
        ece_10=compute_ece_from_bins(bins),
    )
    return metrics, bins
