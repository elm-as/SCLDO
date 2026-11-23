"""Strict chronological out-of-time splitting and expanding TimeSeriesSplit."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.core.config import TemporalSplitConfig


@dataclass(frozen=True)
class ChronologicalSplit:
    """Out-of-time train and holdout test DataFrames sorted strictly by purchase time."""

    train_df: pd.DataFrame
    oot_test_df: pd.DataFrame
    cutoff_timestamp: pd.Timestamp


def split_chronological_oot(
    orders_df: pd.DataFrame, config: TemporalSplitConfig
) -> ChronologicalSplit:
    """Split orders chronologically into train (<= cutoff) and OOT holdout (> cutoff)."""
    ordered = orders_df.sort_values(
        ["order_purchase_timestamp", "order_id"], ascending=[True, True]
    ).reset_index(drop=True)
    cutoff = pd.Timestamp(config.oot_cutoff_iso)
    train_mask = ordered["order_purchase_timestamp"] <= cutoff
    train_df = ordered.loc[train_mask].reset_index(drop=True)
    oot_test_df = ordered.loc[~train_mask].reset_index(drop=True)
    if train_df.empty or oot_test_df.empty:
        raise ValueError(
            f"OOT cutoff {config.oot_cutoff_iso} produced an empty train or test split."
        )
    return ChronologicalSplit(
        train_df=train_df, oot_test_df=oot_test_df, cutoff_timestamp=cutoff
    )


def build_expanding_temporal_folds(
    timestamps: pd.Series, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build strictly expanding TimeSeriesSplit folds verified for chronological order."""
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted monotonically before temporal CV.")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    ts_values = timestamps.to_numpy()
    for train_idx, val_idx in splitter.split(ts_values):
        if ts_values[train_idx[-1]] > ts_values[val_idx[0]]:
            raise ValueError("Temporal fold violation: train timestamp exceeds val.")
        folds.append((train_idx, val_idx))
    return folds
