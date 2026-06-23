"""Temporal data splitting — no random shuffling on time-ordered forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def sort_by_date(df: pd.DataFrame, date_col: str = "as_of_date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(date_col)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    date_col: str = "as_of_date",
) -> TemporalSplit:
    """Chronological 70/15/15 split (remaining 15% is test).

    Random k-fold is forbidden here: future meetings must never appear in
    the training set when predicting past holdout meetings.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    ordered = sort_by_date(df, date_col=date_col)
    n = len(ordered)
    if n < 3:
        raise ValueError(f"Need at least 3 rows for temporal split, got {n}")

    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    val_end = min(val_end, n - 1)

    train = ordered.iloc[:train_end]
    val = ordered.iloc[train_end:val_end]
    test = ordered.iloc[val_end:]
    return TemporalSplit(train=train, val=val, test=test)


def expanding_window_cv(
    n_samples: int,
    *,
    n_splits: int = 5,
    min_train_size: int | None = None,
) -> TimeSeriesSplit:
    """Sklearn ``TimeSeriesSplit`` with project defaults."""
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for TimeSeriesSplit")
    gap = 0
    max_splits = max(2, n_samples - 1)
    splits = min(n_splits, max_splits)
    return TimeSeriesSplit(n_splits=splits, gap=gap, test_size=None)


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    return x, y
