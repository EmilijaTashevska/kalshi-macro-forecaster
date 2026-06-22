"""Tests for the FRED orchestrator.

The client is faked at the protocol level (we duck-type the methods
the orchestrator calls). This proves the orchestrator's FRED →
schema mapping is correct, end-to-end, with no network involved.

The most important assertions:

  1. Vintage rows from FRED land in `series_observations` with their
     vintage_date set to FRED's `realtime_start`, NOT to today.
  2. The release_date stored in our schema equals the EARLIEST
     `realtime_start` across all vintages of an observation_date.
  3. Reading the data back via pit_value uses the correct row for the
     given as_of_date (no leakage).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kalshi_train.data.ingest_fred import (
    ingest_one_series,
    run_fred_ingest,
)
from kalshi_train.data.registry import (
    FRED_REGISTRY,
    all_series,
    find,
    required_series,
)
from kalshi_train.data.sources.fred import FredObservation, FredSeriesInfo
from kalshi_train.db.connection import connect
from kalshi_train.db.point_in_time import pit_value


class _FakeFredClient:
    """A duck-typed stand-in. Records what was asked and returns canned data."""

    def __init__(
        self,
        info: FredSeriesInfo,
        obs_with_vintages: list[FredObservation] | None = None,
        obs_current: list[FredObservation] | None = None,
        raise_on: str | None = None,
    ) -> None:
        self._info = info
        self._with_vintages = obs_with_vintages or []
        self._current = obs_current or []
        self._raise_on = raise_on
        self.calls: list[str] = []

    async def __aenter__(self) -> _FakeFredClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_series_info(self, series_id: str) -> FredSeriesInfo:
        self.calls.append(f"info:{series_id}")
        if self._raise_on == "info":
            raise RuntimeError("simulated info failure")
        return self._info

    async def get_observations_with_vintages(
        self, series_id: str, *, observation_start: str | None = None
    ) -> list[FredObservation]:
        self.calls.append(f"vint:{series_id}:{observation_start}")
        if self._raise_on == "obs":
            raise RuntimeError("simulated obs failure")
        return self._with_vintages

    async def get_observations_current(
        self, series_id: str, *, observation_start: str | None = None
    ) -> list[FredObservation]:
        self.calls.append(f"cur:{series_id}:{observation_start}")
        if self._raise_on == "obs":
            raise RuntimeError("simulated obs failure")
        return self._current


def _cpi_info() -> FredSeriesInfo:
    return FredSeriesInfo(
        series_id="CPIAUCSL",
        title="CPI: All Urban Consumers",
        frequency="Monthly",
        frequency_short="M",
        units="Index 1982-1984=100",
        seasonal_adjustment="SA",
        last_updated=datetime.now(tz=UTC).isoformat(),
        observation_start="1947-01-01",
        observation_end="2024-09-01",
    )


def _dgs10_info() -> FredSeriesInfo:
    return FredSeriesInfo(
        series_id="DGS10",
        title="10-Year Treasury Yield",
        frequency="Daily",
        frequency_short="D",
        units="Percent",
        seasonal_adjustment="NSA",
        last_updated=datetime.now(tz=UTC).isoformat(),
        observation_start="1962-01-02",
        observation_end="2024-12-31",
    )


# ── Revising series: full vintage history ────────────────────────────


async def test_ingest_revising_series_stores_full_vintage_history(tmp_db: Path) -> None:
    entry = find("CPIAUCSL")
    assert entry is not None and entry.revises

    fake = _FakeFredClient(
        info=_cpi_info(),
        obs_with_vintages=[
            FredObservation("2024-08-01", "2024-10-10", "2025-01-14", 2.4),
            FredObservation("2024-08-01", "2025-01-15", "9999-12-31", 2.5),
            FredObservation("2024-09-01", "2024-11-13", "9999-12-31", 2.4),
        ],
    )
    result = await ingest_one_series(fake, entry, db_path=tmp_db)
    assert result.success
    assert result.rows_inserted == 3
    assert fake.calls == ["info:CPIAUCSL", "vint:CPIAUCSL:None"]

    # Verify via the PIT layer that the vintages were stored correctly.
    # Before the Jan 2025 revision: original value.
    assert pit_value("CPIAUCSL", "2024-12-01", db_path=tmp_db) == 2.4
    # After the revision (with no later period available the August
    # revision becomes the current visible value).
    df = pit_value("CPIAUCSL", "2025-02-01", db_path=tmp_db)
    # Sept 2024 is the latest released period, so we get its value (2.4).
    assert df == 2.4


# ── Non-revising series: one vintage per observation ────────────────


async def test_ingest_nonrevising_series_uses_current_endpoint(tmp_db: Path) -> None:
    entry = find("DGS10")
    assert entry is not None and not entry.revises

    fake = _FakeFredClient(
        info=_dgs10_info(),
        obs_current=[
            FredObservation("2024-12-30", "2024-12-31", "9999-12-31", 4.21),
            FredObservation("2024-12-31", "2025-01-02", "9999-12-31", 4.23),
        ],
    )
    result = await ingest_one_series(fake, entry, db_path=tmp_db)
    assert result.success
    assert result.rows_inserted == 2
    # Should hit `current`, not `with_vintages`
    assert any(c.startswith("cur:DGS10") for c in fake.calls)
    assert not any(c.startswith("vint:DGS10") for c in fake.calls)

    # And the values come out through PIT
    assert pit_value("DGS10", "2025-01-02", db_path=tmp_db) == 4.23


# ── release_date is the EARLIEST realtime_start, not today ──────────


async def test_release_date_is_earliest_realtime_start(tmp_db: Path) -> None:
    """For revising series, release_date stored in the DB must be the
    FIRST time FRED ever held that observation — not when we ingested
    it, and not when it was revised.
    """
    entry = find("CPIAUCSL")
    assert entry is not None

    fake = _FakeFredClient(
        info=_cpi_info(),
        obs_with_vintages=[
            # Revision comes first in the list, original second — should
            # not matter: derive_release_date picks min.
            FredObservation("2024-08-01", "2025-01-15", "9999-12-31", 2.5),
            FredObservation("2024-08-01", "2024-10-10", "2025-01-14", 2.4),
        ],
    )
    await ingest_one_series(fake, entry, db_path=tmp_db)

    with connect(tmp_db, read_only=True) as conn:
        rows = conn.execute(
            "SELECT vintage_date, release_date, value "
            "FROM series_observations WHERE series_id = 'CPIAUCSL' "
            "ORDER BY vintage_date"
        ).fetchall()
    assert len(rows) == 2
    # Both rows share the same release_date — the original Oct 10 release.
    assert rows[0]["release_date"].startswith("2024-10-10T")
    assert rows[1]["release_date"].startswith("2024-10-10T")
    # But vintages differ: the original 2024-10-10 and the revision 2025-01-15.
    assert rows[0]["vintage_date"] == "2024-10-10"
    assert rows[1]["vintage_date"] == "2025-01-15"


# ── Error paths ──────────────────────────────────────────────────────


async def test_ingest_records_failure_without_raising(tmp_db: Path) -> None:
    entry = find("CPIAUCSL")
    assert entry is not None

    fake = _FakeFredClient(info=_cpi_info(), raise_on="obs")
    result = await ingest_one_series(fake, entry, db_path=tmp_db)
    assert not result.success
    assert "simulated" in (result.error or "")
    assert result.rows_inserted == 0


async def test_run_fred_ingest_aggregates_results(tmp_db: Path) -> None:
    entry = find("CPIAUCSL")
    assert entry is not None
    fake = _FakeFredClient(
        info=_cpi_info(),
        obs_with_vintages=[
            FredObservation("2024-08-01", "2024-10-10", "9999-12-31", 2.4),
        ],
    )
    report = await run_fred_ingest(
        series_ids=["CPIAUCSL"],
        observation_start="2000-01-01",
        db_path=tmp_db,
        client=fake,
    )
    assert report.n_succeeded == 1
    assert report.n_failed == 0
    assert report.total_rows == 1


async def test_run_fred_ingest_skips_unknown_series(tmp_db: Path) -> None:
    fake = _FakeFredClient(info=_cpi_info())
    report = await run_fred_ingest(
        series_ids=["BOGUS_DOES_NOT_EXIST"],
        db_path=tmp_db,
        client=fake,
    )
    # Unknown ID is filtered out; no calls to the client.
    assert fake.calls == []
    assert report.results == []


async def test_definition_first_and_last_seen_get_set(tmp_db: Path) -> None:
    """The orchestrator should populate series_definitions.first_seen/last_seen."""
    entry = find("CPIAUCSL")
    assert entry is not None
    fake = _FakeFredClient(
        info=_cpi_info(),
        obs_with_vintages=[
            FredObservation("2024-08-01", "2024-10-10", "9999-12-31", 2.4),
            FredObservation("2024-09-01", "2024-11-13", "9999-12-31", 2.4),
        ],
    )
    await ingest_one_series(fake, entry, db_path=tmp_db)
    with connect(tmp_db, read_only=True) as conn:
        row = conn.execute(
            "SELECT first_seen, last_seen FROM series_definitions WHERE series_id='CPIAUCSL'"
        ).fetchone()
    assert row is not None
    assert row["first_seen"] == "2024-08-01"
    assert row["last_seen"] == "2024-09-01"


# ── Sanity-check: registry has the expected categories ──────────────


def test_registry_is_well_formed() -> None:
    seen = set()
    for e in FRED_REGISTRY:
        # No duplicate series IDs
        assert e.series_id not in seen, f"duplicate {e.series_id}"
        seen.add(e.series_id)
        # source must be FRED for this registry
        assert e.source == "FRED"
        # category should be one of our known buckets
        assert e.category in {
            "inflation", "labor", "growth", "surveys", "rates",
            "markets", "money", "housing",
        }


def test_required_series_subset_of_all() -> None:
    req = {e.series_id for e in required_series()}
    full = {e.series_id for e in all_series()}
    assert req.issubset(full)
    # Sanity: there are enough required series for a smoke run.
    assert len(req) >= 20
