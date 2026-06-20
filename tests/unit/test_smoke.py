"""Smoke tests — Phase 0.

The point of these tests is not to test logic (we have no logic yet)
but to prove that the project is installable, importable, and that the
schema applies cleanly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import kalshi_train
from kalshi_train.data.sources.kalshi_models import Candlestick
from kalshi_train.db.connection import connect
from kalshi_train.db.point_in_time import pit_value


def test_package_imports_and_has_version() -> None:
    assert kalshi_train.__version__
    parts = kalshi_train.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_schema_creates_expected_tables(tmp_db: Path) -> None:
    expected = {
        "metadata",
        "question_templates",
        "series_definitions",
        "series_observations",
        "text_documents",
        "kalshi_markets",
        "kalshi_price_history",
        "polymarket_markets",
        "polymarket_price_history",
        "event_calendar",
        "resolutions",
        "ingest_runs",
    }
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    actual = {r[0] for r in rows}
    missing = expected - actual
    assert not missing, f"missing tables: {missing}"


def test_question_templates_seeded(tmp_db: Path) -> None:
    expected_ids = {
        "fed_decision",
        "cpi_yoy",
        "nfp",
        "unemployment",
        "gdp",
        "yield_10y",
        "recession_12m",
    }
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT template_id FROM question_templates").fetchall()
    actual_ids = {r[0] for r in rows}
    assert actual_ids == expected_ids


def test_pragmas_applied(tmp_db: Path) -> None:
    with connect(tmp_db) as conn:
        jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert jm.lower() == "wal", f"expected WAL, got {jm}"
    assert fk == 1, "foreign_keys not enabled"


def test_pit_stubs_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        pit_value("CPIAUCSL", "2024-01-01")


def test_kalshi_models_round_trip() -> None:
    payload = {
        "end_period_ts": 1700000000,
        "price": {"close": 23, "mean": 22},
        "yes_bid": {"close": 22},
        "yes_ask": {"close": 24},
        "volume": 1000,
    }
    c = Candlestick.model_validate(payload)
    assert c.end_period_ts == 1700000000
    # cent int 23 -> "0.2300" dollar string
    assert c.price.get_close() == "0.2300"
