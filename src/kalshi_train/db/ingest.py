"""Low-level DB ingestion helpers.

These wrap the SQL inserts so callers don't write raw SQL. They are
deliberately small and dumb — actual data-source clients live in
``kalshi_train.data.sources`` and call these functions to persist what
they fetch.

Two flavors exist for ``series_observations``:

- ``upsert_observation`` — single row, the path used by interactive
  scripts and tests.
- ``bulk_insert_observations`` — many rows at once with proper
  conflict handling, used by real ingestors that pull thousands of rows
  per call.

We keep timestamps in ISO-8601 strings (UTC). SQLite has no native
datetime type and ISO strings sort lexicographically, which is the
cleanest portable representation.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

DateLike = date | datetime | str


def _to_iso_date(value: DateLike) -> str:
    """Normalize a date-ish argument into an ISO date string ('YYYY-MM-DD').

    Accepts ``date``, ``datetime``, or pre-formatted strings. Rejects
    anything that doesn't parse, to fail fast instead of silently
    accepting bad input that would later corrupt our PIT queries.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    # Accept either a full ISO timestamp or a bare date string.
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError as e:
        raise ValueError(f"Could not parse date from {value!r}: {e}") from e


def _to_iso_datetime(value: DateLike) -> str:
    """Normalize a date-ish argument into an ISO-8601 datetime string."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC).isoformat()
    # String input — accept either date or full timestamp form.
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Could not parse datetime from {value!r}: {e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


# ── series_definitions ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SeriesDefinition:
    """Metadata for a single numeric series.

    `revises=True` means the series may be restated after release (CPI,
    GDP, NFP, etc.) and we should track vintages. `revises=False` means
    the value is fixed at release (e.g. Treasury yields — the closing
    yield on a specific day never changes).
    """

    series_id: str
    source: str
    title: str
    frequency: str
    units: str = ""
    seasonal_adjustment: str = ""
    revises: bool = False
    category: str = ""
    notes: str = ""


def upsert_series_definition(conn: sqlite3.Connection, defn: SeriesDefinition) -> None:
    """Insert or update a series definition. Idempotent."""
    conn.execute(
        """
        INSERT INTO series_definitions (
            series_id, source, title, units, frequency, seasonal_adjustment,
            revises, category, notes, last_ingested_at
        ) VALUES (
            :series_id, :source, :title, :units, :frequency, :seasonal_adjustment,
            :revises, :category, :notes, :now
        )
        ON CONFLICT(series_id) DO UPDATE SET
            source              = excluded.source,
            title               = excluded.title,
            units               = excluded.units,
            frequency           = excluded.frequency,
            seasonal_adjustment = excluded.seasonal_adjustment,
            revises             = excluded.revises,
            category            = excluded.category,
            notes               = excluded.notes,
            last_ingested_at    = excluded.last_ingested_at
        """,
        {
            "series_id": defn.series_id,
            "source": defn.source,
            "title": defn.title,
            "units": defn.units,
            "frequency": defn.frequency,
            "seasonal_adjustment": defn.seasonal_adjustment,
            "revises": int(defn.revises),
            "category": defn.category,
            "notes": defn.notes,
            "now": _now_iso(),
        },
    )


# ── series_observations ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Observation:
    """A single (series, period, vintage) triple with a value.

    Vintages of the same observation_date coexist; the composite primary
    key is (series_id, observation_date, vintage_date).
    """

    series_id: str
    observation_date: DateLike
    release_date: DateLike
    vintage_date: DateLike
    value: float | None
    value_text: str | None = None


def upsert_observation(conn: sqlite3.Connection, obs: Observation) -> None:
    """Upsert one observation. Two rows differing only in vintage_date coexist."""
    conn.execute(
        """
        INSERT INTO series_observations (
            series_id, observation_date, vintage_date, release_date,
            value, value_text, ingested_at
        ) VALUES (
            :series_id, :observation_date, :vintage_date, :release_date,
            :value, :value_text, :ingested_at
        )
        ON CONFLICT(series_id, observation_date, vintage_date) DO UPDATE SET
            release_date = excluded.release_date,
            value        = excluded.value,
            value_text   = excluded.value_text,
            ingested_at  = excluded.ingested_at
        """,
        {
            "series_id": obs.series_id,
            "observation_date": _to_iso_date(obs.observation_date),
            "vintage_date": _to_iso_date(obs.vintage_date),
            "release_date": _to_iso_datetime(obs.release_date),
            "value": obs.value,
            "value_text": obs.value_text if obs.value_text is not None
                          else (None if obs.value is None else f"{obs.value}"),
            "ingested_at": _now_iso(),
        },
    )


def bulk_insert_observations(
    conn: sqlite3.Connection, observations: Iterable[Observation]
) -> int:
    """Insert many observations at once. Returns count of rows attempted."""
    rows = [
        {
            "series_id": o.series_id,
            "observation_date": _to_iso_date(o.observation_date),
            "vintage_date": _to_iso_date(o.vintage_date),
            "release_date": _to_iso_datetime(o.release_date),
            "value": o.value,
            "value_text": o.value_text if o.value_text is not None
                          else (None if o.value is None else f"{o.value}"),
            "ingested_at": _now_iso(),
        }
        for o in observations
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO series_observations (
            series_id, observation_date, vintage_date, release_date,
            value, value_text, ingested_at
        ) VALUES (
            :series_id, :observation_date, :vintage_date, :release_date,
            :value, :value_text, :ingested_at
        )
        ON CONFLICT(series_id, observation_date, vintage_date) DO UPDATE SET
            release_date = excluded.release_date,
            value        = excluded.value,
            value_text   = excluded.value_text,
            ingested_at  = excluded.ingested_at
        """,
        rows,
    )
    return len(rows)


# ── ingest_runs (audit) ────────────────────────────────────────────────


@dataclass(slots=True)
class IngestRun:
    """One ingest invocation's audit record."""

    source: str
    target: str
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    status: str = "running"
    rows_added: int = 0
    rows_updated: int = 0
    error_message: str = ""


def record_ingest_run(conn: sqlite3.Connection, run: IngestRun) -> int:
    """Insert an ingest-run row; returns the auto-generated run_id."""
    cur = conn.execute(
        """
        INSERT INTO ingest_runs (
            source, target, started_at, finished_at, status,
            rows_added, rows_updated, error_message
        ) VALUES (
            :source, :target, :started_at, :finished_at, :status,
            :rows_added, :rows_updated, :error_message
        )
        """,
        {
            "source": run.source,
            "target": run.target,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "status": run.status,
            "rows_added": run.rows_added,
            "rows_updated": run.rows_updated,
            "error_message": run.error_message,
        },
    )
    return int(cur.lastrowid or 0)


# ── metadata ───────────────────────────────────────────────────────────


def set_metadata(conn: sqlite3.Connection, key: str, value: str | Mapping[str, Any]) -> None:
    """Upsert a key in the generic metadata table. Dicts are JSON-encoded."""
    encoded = json.dumps(value) if isinstance(value, Mapping) else value
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, encoded),
    )


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return str(row[0])
