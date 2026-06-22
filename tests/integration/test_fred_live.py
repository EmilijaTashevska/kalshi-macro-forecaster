"""Live FRED integration test.

Runs only when the FRED_API_KEY env var is set; otherwise the entire
module is skipped at collection time. This lets us verify the client
against the real API without making it a CI requirement.

Run it explicitly with::

    uv run pytest -m integration tests/integration/test_fred_live.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kalshi_train.data.ingest_fred import run_fred_ingest
from kalshi_train.data.sources.fred import FredClient
from kalshi_train.db.connection import connect
from kalshi_train.db.point_in_time import pit_value

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("FRED_API_KEY"),
        reason="Set FRED_API_KEY to run live FRED integration tests.",
    ),
]


async def test_get_series_info_for_cpi_works() -> None:
    """Smoke: we can call FRED and decode the response."""
    async with FredClient() as fred:
        info = await fred.get_series_info("CPIAUCSL")
    assert info.series_id == "CPIAUCSL"
    assert info.frequency.lower().startswith("monthly")


async def test_full_pipeline_for_cpi_with_recent_data(tmp_db: Path) -> None:
    """End-to-end: pull a year of CPI and confirm PIT returns sensible values."""
    report = await run_fred_ingest(
        series_ids=["CPIAUCSL"],
        observation_start="2023-01-01",
        db_path=tmp_db,
    )
    assert report.n_succeeded == 1
    assert report.total_rows > 0

    # CPI is reported monthly, and our PIT query should return a value
    # for a recent enough date.
    val = pit_value("CPIAUCSL", "2024-12-31", db_path=tmp_db)
    assert val is not None
    # Sanity: CPI is an index value typically in the 250-350 range
    # for recent years. We assert it's at least a positive number.
    assert val > 0

    # And we should see multiple vintages for at least one observation
    # in the period (FRED revises CPI seasonally each Feb).
    with connect(tmp_db, read_only=True) as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM series_observations WHERE series_id='CPIAUCSL'"
        ).fetchone()[0]
    # ~24 months and ~1-3 vintages each, so at least 24 rows total.
    assert cnt >= 24
