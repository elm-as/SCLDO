"""Causal econometric estimation of delivery delay impact on 1-star customer reviews."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.core.contracts import CausalBucketEstimate, CausalReviewImpactResult

BUCKET_LABELS: tuple[str, ...] = (
    "0d (On-Time)",
    "1-3d Late",
    "4-7d Late",
    ">7d Late",
)

_BUCKET_EDGES = (-1e9, 0.0, 3.0, 7.0, 1e9)
_CONFOUNDER_NAMES: tuple[str, ...] = (
    "log_total_price",
    "freight_ratio",
    "distance_km",
    "promised_lead_days",
    "item_count",
    "is_same_state",
    "customer_state_top8",
    "product_category_top10",
    "purchase_quarter",
)


@dataclass(frozen=True)
class _CausalDesignData:
    """Compact numerical design arrays prepared for fast G-computation and IPW."""

    confounders: np.ndarray
    bucket_codes: np.ndarray
    y_one_star: np.ndarray
    review_scores: np.ndarray


def _fit_ridge_logistic_irls(
    design: np.ndarray, y: np.ndarray, ridge: float = 1e-2
) -> np.ndarray:
    """Fit L2-regularized binary logistic regression via fast Newton-Raphson IRLS."""
    n_cols = design.shape[1]
    beta = np.zeros(n_cols, dtype=float)
    eye = np.eye(n_cols, dtype=float)
    eye[0, 0] = 0.0

    for _ in range(10):
        logits = np.clip(design @ beta, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        weights = np.clip(probs * (1.0 - probs), 1e-6, None)
        grad = design.T @ (probs - y) + ridge * (eye @ beta)
        hess = (design.T * weights) @ design + ridge * eye
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-5:
            break
    return beta


def _build_full_design(confounders: np.ndarray, bucket_codes: np.ndarray) -> np.ndarray:
    """Concatenate intercept, three treatment bucket indicators, and confounders."""
    n_rows = len(bucket_codes)
    intercept = np.ones((n_rows, 1), dtype=float)
    t_dummies = np.column_stack(
        [(bucket_codes == code).astype(float) for code in (1, 2, 3)]
    )
    return np.hstack([intercept, t_dummies, confounders])


def _compute_standardized_risks(
    confounders: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Compute marginal G-computation adjusted 1-star probabilities for all 4 buckets."""
    n_rows = confounders.shape[0]
    base_linear = beta[0] + confounders @ beta[4:]
    risks = np.zeros(4, dtype=float)
    risks[0] = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(base_linear, -30.0, 30.0)))))
    for code in (1, 2, 3):
        shifted = np.clip(base_linear + beta[code], -30.0, 30.0)
        risks[code] = float(np.mean(1.0 / (1.0 + np.exp(-shifted))))
    return risks


def _compute_stabilized_ipw_rates(data: _CausalDesignData) -> np.ndarray:
    """Estimate stabilized multinomial IPW 1-star rates as an econometric cross-check."""
    prop_model = LogisticRegression(max_iter=200, C=1.0, random_state=42)
    prop_model.fit(data.confounders, data.bucket_codes)
    propensities = np.clip(prop_model.predict_proba(data.confounders), 0.01, 0.99)

    ipw_rates = np.zeros(4, dtype=float)
    for code in range(4):
        mask = data.bucket_codes == code
        marginal_p = float(mask.mean())
        weights = np.clip(marginal_p / propensities[mask, code], 0.1, 10.0)
        ipw_rates[code] = float(np.average(data.y_one_star[mask], weights=weights))
    return ipw_rates


def _prepare_causal_design(orders_df: pd.DataFrame) -> _CausalDesignData:
    """Build standardized confounder matrix and delay bucket codes on reviewed orders."""
    reviewed = orders_df.dropna(subset=["review_score", "delay_days"]).copy()
    bucket_cat = pd.cut(
        reviewed["delay_days"], bins=_BUCKET_EDGES, labels=BUCKET_LABELS
    )
    bucket_codes = bucket_cat.cat.codes.to_numpy(dtype=int)

    num_frame = pd.DataFrame(
        {
            "log_total_price": np.log1p(reviewed["total_price"].clip(lower=0.0)),
            "freight_ratio": reviewed["freight_ratio"].clip(0.0, 5.0),
            "distance_km": reviewed["distance_km"].fillna(
                reviewed["distance_km"].median()
            ),
            "promised_lead_days": reviewed["promised_lead_days"].clip(0.0, 120.0),
            "item_count": reviewed["item_count"].clip(1.0, 20.0),
            "is_same_state": reviewed["is_same_state"].fillna(0.0),
        }
    )
    standardized_num = (num_frame - num_frame.mean()) / (num_frame.std() + 1e-8)

    top_states = reviewed["customer_state"].value_counts().head(8).index
    state_clean = reviewed["customer_state"].where(
        reviewed["customer_state"].isin(top_states), "OTHER"
    )
    top_cats = reviewed["product_category_english"].value_counts().head(10).index
    cat_clean = reviewed["product_category_english"].where(
        reviewed["product_category_english"].isin(top_cats), "OTHER"
    )
    quarter = reviewed["order_purchase_timestamp"].dt.to_period("Q").astype(str)

    dummies = pd.get_dummies(
        pd.DataFrame(
            {"state": state_clean, "cat": cat_clean, "quarter": quarter},
            index=reviewed.index,
        ),
        drop_first=True,
        dtype=float,
    )
    confounders = np.hstack([standardized_num.to_numpy(dtype=float), dummies.to_numpy()])
    return _CausalDesignData(
        confounders=confounders,
        bucket_codes=bucket_codes,
        y_one_star=(reviewed["review_score"] == 1).to_numpy(dtype=float),
        review_scores=reviewed["review_score"].to_numpy(dtype=float),
    )


def _run_bootstrap_replicates(
    data: _CausalDesignData, n_replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap distributions for adjusted risks and risk differences."""
    rng = np.random.default_rng(seed)
    n_rows = len(data.y_one_star)
    boot_risks = np.zeros((n_replicates, 4), dtype=float)
    boot_diffs = np.zeros((n_replicates, 4), dtype=float)

    for rep in range(n_replicates):
        idx = rng.integers(0, n_rows, size=n_rows)
        design_b = _build_full_design(data.confounders[idx], data.bucket_codes[idx])
        beta_b = _fit_ridge_logistic_irls(design_b, data.y_one_star[idx])
        risks_b = _compute_standardized_risks(data.confounders[idx], beta_b)
        boot_risks[rep] = risks_b
        boot_diffs[rep] = risks_b - risks_b[0]
    return boot_risks, boot_diffs


def estimate_causal_delay_review_impact(
    orders_df: pd.DataFrame, bootstrap_reps: int = 120, random_state: int = 42
) -> CausalReviewImpactResult:
    """Estimate adjusted causal effect of delay buckets on 1-star review probability.

    Uses G-computation standardized logistic regression controlling for price, freight
    ratio, Haversine distance, promised lead time, state, category, and quarter, with
    95% percentile bootstrap confidence intervals and a stabilized IPW cross-check.
    """
    data = _prepare_causal_design(orders_df)
    full_design = _build_full_design(data.confounders, data.bucket_codes)
    beta_hat = _fit_ridge_logistic_irls(full_design, data.y_one_star)
    adj_risks = _compute_standardized_risks(data.confounders, beta_hat)
    ipw_risks = _compute_stabilized_ipw_rates(data)

    boot_risks, boot_diffs = _run_bootstrap_replicates(
        data, bootstrap_reps, random_state
    )
    risk_low = np.percentile(boot_risks, 2.5, axis=0)
    risk_high = np.percentile(boot_risks, 97.5, axis=0)
    diff_low = np.percentile(boot_diffs, 2.5, axis=0)
    diff_high = np.percentile(boot_diffs, 97.5, axis=0)

    estimates: list[CausalBucketEstimate] = []
    for code, label in enumerate(BUCKET_LABELS):
        mask = data.bucket_codes == code
        estimates.append(
            CausalBucketEstimate(
                bucket_label=label,
                sample_count=int(mask.sum()),
                mean_review_score=float(data.review_scores[mask].mean()),
                unadjusted_one_star_rate=float(data.y_one_star[mask].mean()),
                adjusted_one_star_rate=float(adj_risks[code]),
                adjusted_rate_ci_low=float(risk_low[code]),
                adjusted_rate_ci_high=float(risk_high[code]),
                risk_diff_vs_ontime=float(adj_risks[code] - adj_risks[0]),
                risk_diff_ci_low=float(diff_low[code]),
                risk_diff_ci_high=float(diff_high[code]),
                ipw_one_star_rate=float(ipw_risks[code]),
            )
        )

    return CausalReviewImpactResult(
        sample_size=len(data.y_one_star),
        bootstrap_replicates=bootstrap_reps,
        confounders_used=_CONFOUNDER_NAMES,
        buckets=tuple(estimates),
    )
