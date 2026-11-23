"""Strictly past-only expanding seller performance features with zero future leakage."""

import numpy as np
import pandas as pd

from src.core.config import TemporalSplitConfig


def _compute_single_seller_history(
    group: pd.DataFrame, config: TemporalSplitConfig
) -> pd.DataFrame:
    """Compute resolved-before-purchase seller history for one seller chronologically."""
    purch_ns = group["order_purchase_timestamp"].astype("int64").to_numpy()
    deliv_ns = group["order_delivered_customer_date"].astype("int64").to_numpy()
    effective_resolved_ns = np.maximum(deliv_ns, purch_ns)

    late_vals = group["is_late"].to_numpy(dtype=float)
    delay_vals = group["delay_days"].to_numpy(dtype=float)

    resolved_order = np.argsort(effective_resolved_ns, kind="mergesort")
    sorted_resolved_ns = effective_resolved_ns[resolved_order]
    cum_late = np.concatenate(([0.0], np.cumsum(late_vals[resolved_order])))
    cum_delay = np.concatenate(([0.0], np.cumsum(delay_vals[resolved_order])))

    resolved_counts = np.searchsorted(sorted_resolved_ns, purch_ns, side="left")
    prior_late_sums = cum_late[resolved_counts]
    prior_delay_sums = cum_delay[resolved_counts]

    prior_weight = config.seller_prior_weight
    prior_rate = config.seller_prior_late_rate
    smoothed_late_rate = (prior_late_sums + prior_weight * prior_rate) / (
        resolved_counts + prior_weight
    )
    avg_delay = np.where(
        resolved_counts > 0,
        prior_delay_sums / np.maximum(resolved_counts, 1),
        0.0,
    )

    prior_orders = np.searchsorted(purch_ns, purch_ns, side="left")

    return pd.DataFrame(
        {
            "seller_prior_order_count": prior_orders.astype(float),
            "seller_prior_resolved_count": resolved_counts.astype(float),
            "seller_past_late_rate": smoothed_late_rate.astype(float),
            "seller_past_avg_delay_days": avg_delay.astype(float),
        },
        index=group.index,
    )


def build_strict_past_seller_features(
    orders_df: pd.DataFrame, config: TemporalSplitConfig
) -> pd.DataFrame:
    """Attach strictly causal past-only seller late rate and order volume to orders_df.

    Expects orders_df to contain primary_seller_id, order_purchase_timestamp,
    order_delivered_customer_date, is_late, and delay_days. Returns a copy sorted
    chronologically without mutating the input DataFrame.
    """
    required = {
        "primary_seller_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "is_late",
        "delay_days",
    }
    missing = required - set(orders_df.columns)
    if missing:
        raise ValueError(f"Missing columns for seller history: {sorted(missing)}")

    ordered = orders_df.sort_values(
        ["order_purchase_timestamp", "order_id"], ascending=[True, True]
    ).reset_index(drop=True)

    history_blocks = [
        _compute_single_seller_history(group, config)
        for _, group in ordered.groupby("primary_seller_id", sort=False)
    ]
    history_df = pd.concat(history_blocks, axis=0).reindex(ordered.index)
    return pd.concat([ordered, history_df], axis=1)
