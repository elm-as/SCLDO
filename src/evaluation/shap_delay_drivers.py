"""TreeSHAP attribution of pre-dispatch late-delivery risk drivers on OOT holdout."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import shap

from src.core.contracts import ShapDriverRecord, ShapSummary


@dataclass(frozen=True)
class ShapComputationInput:
    """Fitted tree model, OOT transformed matrix, and feature names for TreeSHAP."""

    model: Any
    feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    sample_size: int
    random_state: int = 42


def _extract_positive_class_shap(raw_values: Any) -> np.ndarray:
    """Normalize TreeSHAP output across LightGBM / SHAP version formats."""
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1], dtype=float)
    arr = np.asarray(raw_values, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _safe_correlation(feature_col: np.ndarray, shap_col: np.ndarray) -> float:
    """Compute Pearson sign correlation guarding against zero-variance slices."""
    if np.std(feature_col) < 1e-12 or np.std(shap_col) < 1e-12:
        return 0.0
    return float(np.corrcoef(feature_col, shap_col)[0, 1])


def compute_shap_delay_drivers(
    shap_input: ShapComputationInput, top_k: int = 15
) -> ShapSummary:
    """Compute ranked global mean |SHAP| values and directional effect signs.

    Samples up to shap_input.sample_size rows deterministically from the OOT matrix
    and returns the top_k pre-dispatch risk drivers.
    """
    matrix = shap_input.feature_matrix
    n_rows = matrix.shape[0]
    n_sample = min(n_rows, shap_input.sample_size)
    rng = np.random.default_rng(shap_input.random_state)
    indices = rng.choice(n_rows, size=n_sample, replace=False)
    sample_matrix = matrix[indices]

    explainer = shap.TreeExplainer(shap_input.model)
    raw_shap = explainer.shap_values(sample_matrix)
    shap_matrix = _extract_positive_class_shap(raw_shap)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    ranked_indices = np.argsort(mean_abs)[::-1][:top_k]

    drivers = tuple(
        ShapDriverRecord(
            feature_name=shap_input.feature_names[int(col_idx)],
            mean_abs_shap=float(mean_abs[int(col_idx)]),
            direction_corr=_safe_correlation(
                sample_matrix[:, int(col_idx)], shap_matrix[:, int(col_idx)]
            ),
        )
        for col_idx in ranked_indices
    )
    return ShapSummary(sample_size=n_sample, drivers=drivers)
