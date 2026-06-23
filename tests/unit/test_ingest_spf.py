"""Unit tests for the SPF ingestion orchestrator.

We test the orchestrator with a fake client that returns a canned
``SPFWorkbook``. End-to-end verification covers:

  - YEAR/QUARTER cells become the correct observation_date / release_date
  - NaN cells are skipped (not stored as NaN floats)
  - The PIT query returns sensible values
  - first_seen / last_seen get populated on the derived series_definitions
"""

from __future__ import annotations

from datetime import UTC, date
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_train.data.ingest_spf import (
    _quarter_start,
    _release_date_for,
    ingest_workbook,
    run_spf_ingest,
)
from kalshi_train.data.sources.spf import SPFWorkbook
from kalshi_train.data.spf_registry import SPF_VARIABLES, all_derived_series_ids
from kalshi_train.db.connection import connect
from kalshi_train.db.point_in_time import pit_value


class _FakeSPFClient:
    """Duck-typed SPF client returning a canned workbook."""

    def __init__(self, wb: SPFWorkbook) -> None:
        self._wb = wb
        self.calls: list[str] = []

    async def __aenter__(self) -> _FakeSPFClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_median_level(self) -> SPFWorkbook:
        self.calls.append("median")
        return self._wb


def _make_cpi_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "YEAR": [2024, 2024, 2024, 2024],
            "QUARTER": [1, 2, 3, 4],
            "CPI1": [3.0, 3.5, 3.2, 2.9],
            "CPI2": [3.1, 3.3, 3.0, 2.8],  # nowcast — what we want
            "CPI3": [2.9, 3.1, 2.8, 2.7],
            "CPI4": [2.7, 2.9, 2.6, 2.5],
            "CPI5": [2.6, 2.7, 2.5, 2.4],
            "CPI6": [2.5, 2.6, 2.4, 2.3],
            "CPIA": [3.0, 3.0, 2.9, 2.8],
            "CPIB": [2.7, 2.6, 2.6, 2.5],
            "CPIC": [2.5, 2.5, 2.5, 2.5],
        }
    )


def _make_minimal_workbook(extra_sheets: dict[str, pd.DataFrame] | None = None) -> SPFWorkbook:
    sheets = {"CPI": _make_cpi_sheet()}
    if extra_sheets:
        sheets.update(extra_sheets)
    return SPFWorkbook(source_url="https://example.invalid/fake.xlsx", sheets=sheets)


# ── Helpers ───────────────────────────────────────────────────────────


def test_release_date_for_uses_mid_quarter_second_month() -> None:
    assert _release_date_for(2024, 1) == date(2024, 2, 15)
    assert _release_date_for(2024, 2) == date(2024, 5, 15)
    assert _release_date_for(2024, 3) == date(2024, 8, 15)
    assert _release_date_for(2024, 4) == date(2024, 11, 15)


def test_quarter_start_returns_first_day_of_first_month() -> None:
    assert _quarter_start(2024, 1) == date(2024, 1, 1)
    assert _quarter_start(2024, 4) == date(2024, 10, 1)


# ── ingest_workbook ───────────────────────────────────────────────────


def test_ingest_workbook_writes_expected_derived_series(tmp_db: Path) -> None:
    wb = _make_minimal_workbook()
    # Restrict to the CPI variable so the test is hermetic.
    cpi_var = next(v for v in SPF_VARIABLES if v.spf_sheet == "CPI")

    results = ingest_workbook(wb, variables=[cpi_var], db_path=tmp_db)
    by_sid = {r.series_id: r for r in results}

    expected_sids = set(cpi_var.horizon_to_series_id.values())
    assert set(by_sid.keys()) == expected_sids
    # Each derived series should have 4 rows (Q1..Q4 2024).
    for sid in expected_sids:
        assert by_sid[sid].success is True
        assert by_sid[sid].rows_inserted == 4


def test_ingest_workbook_pit_value_matches_canned_cpi_nowcast(tmp_db: Path) -> None:
    """The CPI2 (nowcast) column of the 2024:Q3 survey was 3.0 in our
    fixture, released on 2024-08-15. A PIT query as of 2024-09-01
    should return 3.0.
    """
    wb = _make_minimal_workbook()
    cpi_var = next(v for v in SPF_VARIABLES if v.spf_sheet == "CPI")
    ingest_workbook(wb, variables=[cpi_var], db_path=tmp_db)

    val = pit_value("SPF_CPI_MEDIAN_NOWCAST", "2024-09-01", db_path=tmp_db)
    assert val == 3.0

    # Before the Q3 release date, only Q2 data is knowable.
    val_pre = pit_value("SPF_CPI_MEDIAN_NOWCAST", "2024-08-14", db_path=tmp_db)
    assert val_pre == 3.3


def test_ingest_workbook_skips_nan_cells(tmp_db: Path) -> None:
    """Older SPF rows had NaN for some horizons. We must not store NaN values."""
    df = _make_cpi_sheet()
    df.loc[0, "CPI2"] = np.nan  # remove the 2024:Q1 nowcast
    wb = SPFWorkbook(source_url="x", sheets={"CPI": df})
    cpi_var = next(v for v in SPF_VARIABLES if v.spf_sheet == "CPI")
    ingest_workbook(wb, variables=[cpi_var], db_path=tmp_db)
    with connect(tmp_db, read_only=True) as conn:
        rows = conn.execute(
            "SELECT observation_date, value FROM series_observations "
            "WHERE series_id = 'SPF_CPI_MEDIAN_NOWCAST' ORDER BY observation_date"
        ).fetchall()
    obs_dates = {r["observation_date"] for r in rows}
    assert "2024-01-01" not in obs_dates  # NaN row not stored
    assert "2024-04-01" in obs_dates


def test_ingest_workbook_handles_missing_sheet_gracefully(tmp_db: Path) -> None:
    wb = SPFWorkbook(source_url="x", sheets={})  # no sheets at all
    cpi_var = next(v for v in SPF_VARIABLES if v.spf_sheet == "CPI")
    results = ingest_workbook(wb, variables=[cpi_var], db_path=tmp_db)
    assert all(not r.success for r in results)
    assert all("not in workbook" in (r.error or "") for r in results)


def test_ingest_workbook_populates_first_and_last_seen(tmp_db: Path) -> None:
    wb = _make_minimal_workbook()
    cpi_var = next(v for v in SPF_VARIABLES if v.spf_sheet == "CPI")
    ingest_workbook(wb, variables=[cpi_var], db_path=tmp_db)
    with connect(tmp_db, read_only=True) as conn:
        row = conn.execute(
            "SELECT first_seen, last_seen FROM series_definitions "
            "WHERE series_id = 'SPF_CPI_MEDIAN_NOWCAST'"
        ).fetchone()
    assert row is not None
    assert row["first_seen"] == "2024-01-01"
    assert row["last_seen"] == "2024-10-01"


# ── run_spf_ingest (with fake client) ─────────────────────────────────


async def test_run_spf_ingest_uses_injected_client(tmp_db: Path) -> None:
    wb = _make_minimal_workbook()
    client = _FakeSPFClient(wb)
    report = await run_spf_ingest(db_path=tmp_db, client=client)
    assert client.calls == ["median"]
    # Some series will be "missing sheet" errors because we only put CPI
    # in the workbook. CPI's derived series should all succeed.
    cpi_var = next(v for v in SPF_VARIABLES if v.spf_sheet == "CPI")
    cpi_sids = set(cpi_var.horizon_to_series_id.values())
    successes = {r.series_id for r in report.results if r.success}
    assert cpi_sids.issubset(successes)


# ── Registry sanity ──────────────────────────────────────────────────


def test_registry_has_no_duplicate_series_ids() -> None:
    sids = all_derived_series_ids()
    assert len(sids) == len(set(sids)), "duplicate derived series IDs in SPF registry"


def test_registry_covers_expected_categories() -> None:
    seen_categories = {v.category for v in SPF_VARIABLES}
    assert seen_categories >= {"inflation", "growth", "labor", "rates"}


# Pin timezone helpers to ensure they're imported correctly even if we
# don't directly use them — guards against accidental tree-shake later.
_ = UTC
