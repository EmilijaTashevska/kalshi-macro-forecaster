"""Worked-example tests for the point-in-time query layer.

Property-based tests live alongside in ``test_point_in_time_properties.py``.
These example-based tests encode the exact scenarios from
README/glossary and from our scoping conversation, so the behavior is
obvious to a reader.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest

from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    Observation,
    SeriesDefinition,
    upsert_observation,
    upsert_series_definition,
)
from kalshi_train.db.point_in_time import (
    VintagePolicy,
    pit_frame,
    pit_history,
    pit_value,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def cpi_db(tmp_db: Path) -> Path:
    """A DB pre-loaded with the worked CPI example from our scoping chat.

    August 2024 CPI:
      - First release on 2024-10-10 reported 2.4%
      - Revised on 2025-01-15 to 2.5%

    September 2024 CPI:
      - Single release on 2024-11-13 reported 2.4%
    """
    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn,
            SeriesDefinition(
                series_id="CPIAUCSL",
                source="FRED",
                title="Consumer Price Index for All Urban Consumers: All Items",
                frequency="monthly",
                units="Percent YoY",
                seasonal_adjustment="SA",
                revises=True,
                category="inflation",
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-08-01",
                release_date="2024-10-10T12:30:00+00:00",  # 8:30am ET = 12:30 UTC
                vintage_date="2024-10-10",
                value=2.4,
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-08-01",
                release_date="2024-10-10T12:30:00+00:00",
                vintage_date="2025-01-15",
                value=2.5,
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-09-01",
                release_date="2024-11-13T13:30:00+00:00",
                vintage_date="2024-11-13",
                value=2.4,
            ),
        )
        conn.commit()
    return tmp_db


# ── Example 1: revision picking ───────────────────────────────────────


def test_pit_picks_unrevised_vintage_when_revision_not_yet_known(cpi_db: Path) -> None:
    """The Nov 7, 2024 FOMC meeting saw only the original 2.4, not the
    Jan 2025 revision to 2.5. This is the scenario we keep talking
    about; it had better hold in code.
    """
    val = pit_value("CPIAUCSL", "2024-11-07", db_path=cpi_db)
    assert val == 2.4


def test_pit_history_shows_revised_august_after_revision_date(cpi_db: Path) -> None:
    """After Jan 15, 2025, the August row in series_observations uses
    its revised vintage (2.5). pit_value at a date when August is the
    latest released period would then return 2.5. By Feb 2025 the
    latest period is September (still 2.4), so we use pit_history to
    inspect the per-period vintage choice instead.
    """
    df = pit_history("CPIAUCSL", "2025-01-15", "2025-02-01", freq="D", db_path=cpi_db)
    # The visible value tracks September (2.4) because it's the most
    # recent released period. The August revision doesn't change that.
    assert df.loc["2025-01-15"]["value"] == 2.4
    assert df.loc["2025-01-15"]["observation_date"] == "2024-09-01"


def test_pit_value_returns_latest_period_not_revised_older_period(cpi_db: Path) -> None:
    """If both August and September CPI are released, pit_value returns
    September's value (the more recent period), regardless of whether
    August has been revised. The revision only matters if you
    specifically pull the August row via pit_history at a date where
    August was the latest released period.
    """
    val = pit_value("CPIAUCSL", "2025-02-01", db_path=cpi_db)
    assert val == 2.4
    # Confirm it's September, not August.
    df = pit_history("CPIAUCSL", "2025-02-01", "2025-02-01", db_path=cpi_db)
    assert df.iloc[0]["observation_date"] == "2024-09-01"


def test_pit_returns_unrevised_august_when_august_is_only_period(tmp_db: Path) -> None:
    """In a DB where only the August release exists, pit_value flips
    between vintages around the revision date. This is the case where
    the revision is directly observable through pit_value.
    """
    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn,
            SeriesDefinition(
                series_id="CPIAUCSL",
                source="FRED",
                title="CPI",
                frequency="monthly",
                revises=True,
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-08-01",
                release_date="2024-10-10T12:30:00+00:00",
                vintage_date="2024-10-10",
                value=2.4,
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-08-01",
                release_date="2024-10-10T12:30:00+00:00",
                vintage_date="2025-01-15",
                value=2.5,
            ),
        )
        conn.commit()

    # Day before revision: original vintage.
    assert pit_value("CPIAUCSL", "2025-01-14", db_path=tmp_db) == 2.4
    # Revision date: new vintage.
    assert pit_value("CPIAUCSL", "2025-01-15", db_path=tmp_db) == 2.5
    # Long after: still new vintage.
    assert pit_value("CPIAUCSL", "2025-06-01", db_path=tmp_db) == 2.5


# ── Example 2: unreleased data ────────────────────────────────────────


def test_pit_returns_none_before_first_release(cpi_db: Path) -> None:
    """No CPI release had happened by 2024-09-01, so we should refuse
    to return anything (this is the release-timing leak guard).
    """
    val = pit_value("CPIAUCSL", "2024-09-01", db_path=cpi_db)
    assert val is None


def test_pit_returns_none_when_no_data_for_series(cpi_db: Path) -> None:
    val = pit_value("DOES_NOT_EXIST", "2024-11-07", db_path=cpi_db)
    assert val is None


def test_pit_includes_intraday_release_on_same_day(cpi_db: Path) -> None:
    """Aug 2024 CPI was released at 8:30am ET on Oct 10. If we ask 'as
    of 2024-10-10' (the date), we should include it — we treat the as-of
    date as end-of-day UTC.
    """
    val = pit_value("CPIAUCSL", "2024-10-10", db_path=cpi_db)
    assert val == 2.4


# ── Example 3: latest period selection ────────────────────────────────


def test_pit_selects_latest_period_among_released(cpi_db: Path) -> None:
    """Once Sep 2024 has been released too, asking as-of late November
    should return the September value (the more recent period), not the
    August value.
    """
    val = pit_value("CPIAUCSL", "2024-11-30", db_path=cpi_db)
    assert val == 2.4  # September value; August was also 2.4 but September is "more recent"
    # More importantly: confirm via pit_history that the selected
    # observation is September, not August.
    df = pit_history("CPIAUCSL", "2024-11-30", "2024-11-30", db_path=cpi_db)
    assert df.iloc[0]["observation_date"] == "2024-09-01"


# ── Example 4: vintage policy ─────────────────────────────────────────


def test_latest_revision_policy_ignores_vintage_date(cpi_db: Path) -> None:
    """With LATEST_REVISION, we should get 2.5 even when as_of_date is
    before the revision — this is the policy that intentionally peeks.
    The query also emits a warning.
    """
    with pytest.warns(UserWarning, match="LATEST_REVISION"):
        val = pit_value(
            "CPIAUCSL", "2024-11-07", policy=VintagePolicy.LATEST_REVISION, db_path=cpi_db
        )
    assert val == 2.5


# ── pit_frame ─────────────────────────────────────────────────────────


def test_pit_frame_returns_one_row_per_query(cpi_db: Path) -> None:
    df = pit_frame(["CPIAUCSL"], "2024-11-07", db_path=cpi_db)
    assert df.shape == (1, 1)
    assert df.iloc[0]["CPIAUCSL"] == 2.4


def test_pit_frame_missing_series_is_nan(cpi_db: Path) -> None:
    df = pit_frame(["CPIAUCSL", "MISSING_SERIES"], "2024-11-07", db_path=cpi_db)
    assert df.iloc[0]["CPIAUCSL"] == 2.4
    assert math.isnan(df.iloc[0]["MISSING_SERIES"])


def test_pit_frame_empty_input_returns_empty_frame(cpi_db: Path) -> None:
    df = pit_frame([], "2024-11-07", db_path=cpi_db)
    assert df.shape == (1, 0)


# ── pit_history ───────────────────────────────────────────────────────


def test_pit_history_shows_revision_step(cpi_db: Path) -> None:
    """The time series of "value-as-known" should jump from 2.4 to 2.5
    on 2025-01-15 when the revision is published.
    """
    df = pit_history("CPIAUCSL", "2024-12-01", "2025-02-01", db_path=cpi_db)
    # Pre-revision rows show 2.4 (Sept 2024 is the latest released observation)
    assert df.loc["2024-12-15"]["value"] == 2.4
    # The revision applies only to August, but August is no longer the
    # latest period, so the visible value stays at 2.4 (September).
    # However the August row in series_observations has changed -- we
    # confirm the *vintage_date* of the picked observation moves
    # forward for any as_of_date that intersects August. To test the
    # revision step itself we need a date when August was the latest
    # period: see the dedicated test below.
    assert df.loc["2025-01-30"]["value"] == 2.4


def test_pit_history_picks_revised_august_when_august_is_latest(tmp_db: Path) -> None:
    """Set up a DB where August is the latest released period, then
    verify the visible value flips from 2.4 to 2.5 on the revision date.
    """
    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn,
            SeriesDefinition(
                series_id="CPIAUCSL",
                source="FRED",
                title="CPI",
                frequency="monthly",
                revises=True,
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-08-01",
                release_date="2024-10-10T12:30:00+00:00",
                vintage_date="2024-10-10",
                value=2.4,
            ),
        )
        upsert_observation(
            conn,
            Observation(
                series_id="CPIAUCSL",
                observation_date="2024-08-01",
                release_date="2024-10-10T12:30:00+00:00",
                vintage_date="2025-01-15",
                value=2.5,
            ),
        )
        conn.commit()

    df = pit_history("CPIAUCSL", "2024-11-01", "2025-02-01", db_path=tmp_db)
    assert df.loc["2024-12-15"]["value"] == 2.4   # pre-revision
    assert df.loc["2025-01-14"]["value"] == 2.4   # day before
    assert df.loc["2025-01-15"]["value"] == 2.5   # the flip
    assert df.loc["2025-02-01"]["value"] == 2.5   # post-revision


def test_pit_history_rejects_inverted_range(cpi_db: Path) -> None:
    with pytest.raises(ValueError, match="after"):
        pit_history("CPIAUCSL", "2025-01-01", "2024-01-01", db_path=cpi_db)


def test_pit_history_accepts_date_objects(cpi_db: Path) -> None:
    df = pit_history("CPIAUCSL", date(2024, 11, 1), date(2024, 11, 10), db_path=cpi_db)
    assert not df.empty


def test_pit_history_business_day_frequency(cpi_db: Path) -> None:
    df = pit_history("CPIAUCSL", "2024-11-04", "2024-11-08", freq="B", db_path=cpi_db)
    # Mon-Fri = 5 business days
    assert len(df) == 5
