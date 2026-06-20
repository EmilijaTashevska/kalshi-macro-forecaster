"""SQLite connection helpers — both sync and async flavors.

We enable a small set of pragmas on every connection:

- `journal_mode=WAL`: write-ahead logging, allows concurrent readers
  while a writer is active. Necessary for letting Datasette browse the
  DB while an ingest job is running.
- `foreign_keys=ON`: SQLite has FKs declared in the schema but does not
  enforce them unless explicitly enabled, *per connection*. We do.
- `synchronous=NORMAL`: a reasonable durability/speed trade-off for
  development data. We can tighten to FULL later if needed.
- `temp_store=MEMORY`: faster temp tables, no disk thrash.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import aiosqlite

from kalshi_train.config import settings

PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA busy_timeout = 5000",
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_pragmas_sync(conn: sqlite3.Connection) -> None:
    for stmt in PRAGMAS:
        conn.execute(stmt)


async def _apply_pragmas_async(conn: aiosqlite.Connection) -> None:
    for stmt in PRAGMAS:
        await conn.execute(stmt)


@contextmanager
def connect(
    db_path: Path | None = None,
    *,
    read_only: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Synchronous connection. Use for short operations and CLI scripts."""
    path = db_path or settings.kalshi_train_db_path
    _ensure_parent(path)
    if read_only:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _apply_pragmas_sync(conn)
    try:
        yield conn
    finally:
        conn.close()


async def aconnect(db_path: Path | None = None) -> aiosqlite.Connection:
    """Async connection. Caller is responsible for closing it.

    Prefer using `async with aconnect_ctx(...)` to guarantee cleanup.
    """
    path = db_path or settings.kalshi_train_db_path
    _ensure_parent(path)
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await _apply_pragmas_async(conn)
    return conn


def init_schema(db_path: Path | None = None) -> None:
    """Apply the canonical schema.sql to an empty (or existing) DB.

    Safe to call repeatedly: the schema uses `CREATE TABLE IF NOT EXISTS`.
    """
    sql = settings.schema_path.read_text()
    with connect(db_path) as conn:
        conn.executescript(sql)
        conn.commit()
