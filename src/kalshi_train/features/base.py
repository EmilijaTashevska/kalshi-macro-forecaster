"""Low-level PIT feature primitives.

Every function here routes through ``pit_value`` / ``pit_history`` so
features cannot accidentally peek at future vintages or unreleased data.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from kalshi_train.db.point_in_time import pit_history, pit_value

DateLike = date | str


def pit_level(
    series_id: str,
    as_of_date: DateLike,
    *,
    db_path: Path | None = None,
) -> float | None:
    """Latest level of ``series_id`` knowable on ``as_of_date``."""
    val = pit_value(series_id, as_of_date, db_path=db_path)
    return None if val is None or pd.isna(val) else float(val)


def pit_change_bdays(
    series_id: str,
    as_of_date: DateLike,
    bdays: int,
    *,
    db_path: Path | None = None,
) -> float | None:
    """Level change over ``bdays`` NYSE business days ending at ``as_of_date``."""
    as_of = pd.Timestamp(as_of_date).normalize()
    start = as_of - pd.tseries.offsets.BDay(bdays)
    hist = pit_history(
        series_id,
        start.date(),
        as_of.date(),
        freq="B",
        db_path=db_path,
    )
    if hist.empty:
        return None
    values = hist["value"].dropna()
    if len(values) < 2:
        return None
    return float(values.iloc[-1] - values.iloc[0])


def pit_yoy_index_change(
    series_id: str,
    as_of_date: DateLike,
    *,
    months: int = 12,
    db_path: Path | None = None,
) -> float | None:
    """Percent change vs ``months`` calendar months ago (for index levels like CPI)."""
    as_of = pd.Timestamp(as_of_date).normalize()
    lookback = as_of - pd.DateOffset(months=months)
    current = pit_level(series_id, as_of.date(), db_path=db_path)
    past = pit_level(series_id, lookback.date(), db_path=db_path)
    if current is None or past is None or past == 0:
        return None
    return float((current / past - 1.0) * 100.0)
