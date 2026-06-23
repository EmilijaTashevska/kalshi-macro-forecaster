"""Live integration test against the real Philly Fed SPF endpoint.

Runs only when the env var ``KALSHI_TRAIN_LIVE_SPF`` is set, because
SPF doesn't require auth (the URL is public) but we don't want to hit
the Philly Fed servers on every CI run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kalshi_train.data.ingest_spf import run_spf_ingest
from kalshi_train.data.sources.spf import SPFClient
from kalshi_train.db.point_in_time import pit_value

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("KALSHI_TRAIN_LIVE_SPF"),
        reason="Set KALSHI_TRAIN_LIVE_SPF=1 to run live SPF integration tests.",
    ),
]


async def test_get_median_level_smoke() -> None:
    async with SPFClient() as spf:
        wb = await spf.get_median_level()
    # All the variables we register on must be present in the workbook.
    expected_sheets = {"CPI", "CORECPI", "PCE", "COREPCE", "RGDP", "UNEMP", "TBILL", "TBOND"}
    assert expected_sheets.issubset(set(wb.sheets.keys()))


async def test_full_spf_pipeline_writes_recent_values(tmp_db: Path) -> None:
    report = await run_spf_ingest(db_path=tmp_db)
    assert report.n_succeeded > 0
    assert report.total_rows > 0

    # A 2024:Q4 SPF release happened around Nov 15, 2024. A query for
    # the CPI nowcast as of Dec 2024 must return a sensible number.
    val = pit_value("SPF_CPI_MEDIAN_NOWCAST", "2024-12-01", db_path=tmp_db)
    assert val is not None
    assert 0.0 < val < 20.0   # plausible CPI inflation rate
