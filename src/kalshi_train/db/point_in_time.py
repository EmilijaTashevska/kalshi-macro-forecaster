"""Point-in-time (PIT) query interface — the ONLY legal way to read
time-series data for ML features.

═══════════════════════════════════════════════════════════════════════
WHY THIS MODULE EXISTS
═══════════════════════════════════════════════════════════════════════

Historical economic data has two failure modes that silently corrupt
ML backtests:

  (A) Release-timing leak. Using a feature whose value was not yet
      published at the prediction's `as_of_date`. E.g. predicting the
      Oct-10 CPI release while using the same release as a feature.

  (B) Revision leak. Using today's revised value of a past datapoint as
      a feature, when at the historical `as_of_date` the original
      (unrevised) value was what the market saw. Models trained on
      revised data look brilliant in backtest and lose money in
      production, because production only has first vintages.

This module enforces:

      For every value returned by a PIT query,
          observation.release_date <= as_of_date
      AND   observation.vintage_date <= as_of_date.

The vintage we pick is the LATEST one satisfying both inequalities —
i.e. "what would FRED/ALFRED have shown me on `as_of_date`?".

═══════════════════════════════════════════════════════════════════════
VINTAGE POLICIES
═══════════════════════════════════════════════════════════════════════

For series that DO revise (CPI, NFP, GDP, etc.), the default policy is
``VintagePolicy.FIRST_KNOWN_AT`` — the vintage available on as_of_date.
This is what a contemporary market participant actually saw.

For series that DO NOT revise (Treasury yields, S&P 500 closes, etc.),
all vintages of the same observation are identical, and the policy is
moot. ``series_definitions.revises`` documents this per series.

For special analytical needs (e.g. "use the latest revised value
regardless of as_of_date") the ``LATEST_REVISION`` policy is provided,
but it is DELIBERATELY not the default and emits a warning when used
with `as_of_date` set, to make accidental leakage hard.

═══════════════════════════════════════════════════════════════════════
PUBLIC API
═══════════════════════════════════════════════════════════════════════

  pit_value(series_id, as_of_date)
      Latest value of one series, point-in-time.

  pit_frame(series_ids, as_of_date)
      Single-row DataFrame containing one value per series. Useful for
      building a feature vector for a specific prediction.

  pit_history(series_id, start_date, end_date)
      Daily timeline of "what was the latest-known value on each date"
      between start_date and end_date. Each row uses the vintage that
      was current on its OWN date.
"""

from __future__ import annotations

import sqlite3
import warnings
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from kalshi_train.db.connection import connect

DateLike = date | datetime | str


class VintagePolicy(StrEnum):
    """How to choose a vintage when multiple are present.

    FIRST_KNOWN_AT: the most recent vintage with vintage_date <= as_of_date.
        This is what was actually visible to a contemporary observer.
        It is the project-wide default.

    LATEST_REVISION: ignore vintage_date entirely, return the most-
        recently-reported vintage that exists in the DB. Use only for
        understanding-the-world questions (e.g. structural studies),
        not for predicting market behavior.
    """

    FIRST_KNOWN_AT = "first_known_at"
    LATEST_REVISION = "latest_revision"


class LeakageError(RuntimeError):
    """Raised when a PIT query receives obviously contradictory arguments."""


# ── Internal date helpers ─────────────────────────────────────────────


def _to_iso_date(value: DateLike) -> str:
    """Normalize a date-ish argument into 'YYYY-MM-DD'."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Could not parse date from {value!r}: {e}") from e


def _end_of_day_utc(iso_date: str) -> str:
    """Convert 'YYYY-MM-DD' into the end-of-day UTC timestamp string.

    We store release timestamps with full intraday precision (CPI is
    8:30am ET, NFP is 8:30am ET, etc.), but as_of_date is a date.
    Treating the as_of as "end-of-day UTC" makes the natural query
    `release_date <= as_of_end_of_day` work for all releases that
    happened anywhere on that calendar day. This is conservative — we
    include all intraday releases.
    """
    return f"{iso_date}T23:59:59.999999+00:00"


# ── pit_value ─────────────────────────────────────────────────────────


def pit_value(
    series_id: str,
    as_of_date: DateLike,
    *,
    policy: VintagePolicy = VintagePolicy.FIRST_KNOWN_AT,
    db_path: Path | None = None,
) -> float | None:
    """Return the value of ``series_id`` as known on ``as_of_date``.

    Returns ``None`` if no observation has been released by ``as_of_date``
    under the chosen vintage policy.

    Parameters
    ----------
    series_id:
        FRED-style series identifier.
    as_of_date:
        The historical date as which to view the data. Anything dated
        later than this MUST NOT influence the returned value.
    policy:
        See ``VintagePolicy``. Default is ``FIRST_KNOWN_AT``, the only
        policy safe for predictive features.
    db_path:
        Override the configured DB path; tests use this.
    """
    iso_as_of = _to_iso_date(as_of_date)
    if policy is VintagePolicy.LATEST_REVISION:
        warnings.warn(
            "pit_value called with LATEST_REVISION policy: this ignores "
            "vintage history and may introduce look-ahead bias. Confirm "
            "this is intended (e.g. structural-study feature) before "
            "using in production predictions.",
            stacklevel=2,
        )

    with connect(db_path, read_only=True) as conn:
        row = _select_latest_observation(conn, series_id, iso_as_of, policy)
    if row is None:
        return None
    val = row["value"]
    return float(val) if val is not None else None


def _select_latest_observation(
    conn: sqlite3.Connection,
    series_id: str,
    iso_as_of: str,
    policy: VintagePolicy,
) -> sqlite3.Row | None:
    """Run the SQL that picks the appropriate observation row.

    Two filtering rules apply:

      release_date <= end_of_day(iso_as_of)
          (the row was published before/on as_of_date)

      vintage_date <= iso_as_of           (FIRST_KNOWN_AT only)
          (we are using a vintage knowable on as_of_date)

    Within those filters, we choose:

      latest observation_date first   (most recent period)
      then latest vintage_date        (most recent vintage of that period)
    """
    as_of_eod = _end_of_day_utc(iso_as_of)
    if policy is VintagePolicy.FIRST_KNOWN_AT:
        sql = """
        SELECT observation_date, vintage_date, release_date, value, value_text
        FROM series_observations
        WHERE series_id = :series_id
          AND release_date <= :as_of_eod
          AND vintage_date <= :iso_as_of
        ORDER BY observation_date DESC, vintage_date DESC
        LIMIT 1
        """
    else:
        sql = """
        SELECT observation_date, vintage_date, release_date, value, value_text
        FROM series_observations
        WHERE series_id = :series_id
          AND release_date <= :as_of_eod
        ORDER BY observation_date DESC, vintage_date DESC
        LIMIT 1
        """
    row: sqlite3.Row | None = conn.execute(
        sql,
        {"series_id": series_id, "iso_as_of": iso_as_of, "as_of_eod": as_of_eod},
    ).fetchone()
    return row


# ── pit_frame ─────────────────────────────────────────────────────────


def pit_frame(
    series_ids: list[str],
    as_of_date: DateLike,
    *,
    policy: VintagePolicy = VintagePolicy.FIRST_KNOWN_AT,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return a 1-row DataFrame of the requested series, PIT-resolved.

    The single row is indexed by ``as_of_date``. Columns are the
    requested series ids. Missing series (not released by as_of_date)
    appear as ``NaN``.

    Useful for building a single feature vector for a prediction.
    """
    iso_as_of = _to_iso_date(as_of_date)
    if not series_ids:
        return pd.DataFrame(
            index=pd.DatetimeIndex([pd.Timestamp(iso_as_of)], name="as_of_date")
        )

    data: dict[str, list[float]] = {}
    with connect(db_path, read_only=True) as conn:
        for sid in series_ids:
            row = _select_latest_observation(conn, sid, iso_as_of, policy)
            val = (
                float(row["value"])
                if row is not None and row["value"] is not None
                else float("nan")
            )
            data[sid] = [val]

    return pd.DataFrame(
        data,
        index=pd.DatetimeIndex([pd.Timestamp(iso_as_of)], name="as_of_date"),
        dtype="float64",
    )


# ── pit_history ───────────────────────────────────────────────────────


def pit_history(
    series_id: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    freq: str = "D",
    policy: VintagePolicy = VintagePolicy.FIRST_KNOWN_AT,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return a daily history of "value as known" for ``series_id``.

    Output schema:
        index            ``as_of_date`` DatetimeIndex (one row per ``freq`` step)
        ``value``         the value as known on that ``as_of_date``
        ``observation_date`` the period the value describes
        ``vintage_date``  the vintage that was current then
        ``release_date``  when that observation was originally published

    This is the right "history" object to use when constructing
    training datasets — each row reflects what was knowable on its own
    ``as_of_date``, so concatenating rows produces a contemporaneous
    view rather than a today's-revised view.

    Parameters
    ----------
    freq:
        Pandas date-range frequency. Defaults to daily ('D'). Use 'B'
        for business days, 'W-FRI' for Fridays, etc.
    """
    start = _to_iso_date(start_date)
    end = _to_iso_date(end_date)
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")

    idx = pd.date_range(start=start, end=end, freq=freq)
    rows: list[dict[str, Any]] = []

    with connect(db_path, read_only=True) as conn:
        for ts in idx:
            iso = ts.date().isoformat()
            row = _select_latest_observation(conn, series_id, iso, policy)
            if row is None:
                rows.append(
                    {
                        "value": None,
                        "observation_date": None,
                        "vintage_date": None,
                        "release_date": None,
                    }
                )
            else:
                rows.append(
                    {
                        "value": float(row["value"]) if row["value"] is not None else None,
                        "observation_date": row["observation_date"],
                        "vintage_date": row["vintage_date"],
                        "release_date": row["release_date"],
                    }
                )

    df = pd.DataFrame(rows, index=idx)
    df.index.name = "as_of_date"
    return df
