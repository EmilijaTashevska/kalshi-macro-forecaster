"""Point-in-time (PIT) query interface — the ONLY legal way to read
time-series data for ML purposes.

This module is intentionally a stub in Phase 0. The real implementation
arrives in Phase 1.1, alongside its property-based tests.

The contract (which the tests will enforce):

    pit_value(series_id, as_of_date)
        Returns the value of `series_id` for the most recently RELEASED
        observation whose `release_date <= as_of_date`, using the
        `vintage_date` that was current at `as_of_date`.

        Equivalent to: "what would I have read on FRED on `as_of_date`?"

    pit_frame(series_ids, as_of_date)
        Returns a single-row DataFrame of all requested series, each
        resolved point-in-time.

    pit_history(series_id, start_date, end_date)
        Returns the full time series as it would have looked, day by
        day, between start_date and end_date — i.e. each row uses the
        vintage that was current on its own date. Useful for building
        training tables.

Why be so paranoid? The single most common way ML projects on
financial data silently lie to themselves is to train using
TODAY's revised values for HISTORICAL features. The model looks like
genius in backtest and dies in production. We will not.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a PIT query is asked for data that wasn't yet released."""


def pit_value(series_id: str, as_of_date: date | str) -> float | None:
    """Phase 1.1 — will return the PIT-resolved value of one series."""
    del series_id, as_of_date
    raise NotImplementedError(
        "pit_value is implemented in Phase 1.1; this is a Phase 0 stub.",
    )


def pit_frame(
    series_ids: list[str],
    as_of_date: date | str,
) -> pd.DataFrame:
    """Phase 1.1 — will return a single-row DataFrame of multiple series."""
    del series_ids, as_of_date
    raise NotImplementedError(
        "pit_frame is implemented in Phase 1.1; this is a Phase 0 stub.",
    )


def pit_history(
    series_id: str,
    start_date: date | str,
    end_date: date | str,
) -> pd.DataFrame:
    """Phase 1.1 — will return a daily series, each row using its own as-of vintage."""
    del series_id, start_date, end_date
    raise NotImplementedError(
        "pit_history is implemented in Phase 1.1; this is a Phase 0 stub.",
    )
