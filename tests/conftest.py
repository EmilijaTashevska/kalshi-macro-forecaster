"""Shared pytest fixtures.

We make the test database a temp file so tests never touch the real
project DB. Each test gets a fresh schema-initialized DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_train import config as cfg
from kalshi_train.db.connection import init_schema


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh, schema-initialized SQLite DB scoped to this test."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("KALSHI_TRAIN_DB_PATH", str(db_path))
    # Force reload of the settings singleton so it picks up the new env.
    monkeypatch.setattr(cfg, "settings", cfg.Settings())
    init_schema(db_path)
    return db_path
