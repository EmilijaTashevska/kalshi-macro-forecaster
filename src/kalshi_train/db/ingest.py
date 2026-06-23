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


# ── kalshi_markets ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class KalshiMarketRow:
    """One Kalshi market plus its classification against our templates.

    ``template_id`` / ``strike_value`` / ``strike_direction`` are filled
    by the Phase 1.5 classifier; they may be ``None``/"" for markets we
    ingest for their price history but couldn't structure.
    """

    ticker: str
    event_ticker: str
    series_ticker: str = ""
    market_type: str = ""
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""
    open_time: str | None = None
    close_time: str | None = None
    created_time: str | None = None
    settlement_time: str | None = None
    status: str = ""
    result: str = ""
    settlement_value_dollars: str | None = None
    template_id: str | None = None
    strike_value: float | None = None
    strike_direction: str = ""
    last_price_dollars: str = "0.0000"
    volume_fp: str = "0.00"
    open_interest_fp: str = "0.00"


def upsert_kalshi_market(conn: sqlite3.Connection, market: KalshiMarketRow) -> None:
    """Insert or refresh a Kalshi market row. Idempotent on ``ticker``.

    ``ingested_at`` is preserved on conflict (first-seen), while
    ``last_refreshed_at`` and the mutable price/status fields are updated.
    """
    conn.execute(
        """
        INSERT INTO kalshi_markets (
            ticker, event_ticker, series_ticker, market_type, title, subtitle,
            yes_sub_title, no_sub_title, rules_primary, rules_secondary,
            open_time, close_time, created_time, settlement_time,
            status, result, settlement_value_dollars,
            template_id, strike_value, strike_direction,
            last_price_dollars, volume_fp, open_interest_fp,
            ingested_at, last_refreshed_at
        ) VALUES (
            :ticker, :event_ticker, :series_ticker, :market_type, :title, :subtitle,
            :yes_sub_title, :no_sub_title, :rules_primary, :rules_secondary,
            :open_time, :close_time, :created_time, :settlement_time,
            :status, :result, :settlement_value_dollars,
            :template_id, :strike_value, :strike_direction,
            :last_price_dollars, :volume_fp, :open_interest_fp,
            :now, :now
        )
        ON CONFLICT(ticker) DO UPDATE SET
            event_ticker             = excluded.event_ticker,
            series_ticker            = excluded.series_ticker,
            market_type              = excluded.market_type,
            title                    = excluded.title,
            subtitle                 = excluded.subtitle,
            yes_sub_title            = excluded.yes_sub_title,
            no_sub_title             = excluded.no_sub_title,
            rules_primary            = excluded.rules_primary,
            rules_secondary          = excluded.rules_secondary,
            open_time                = excluded.open_time,
            close_time               = excluded.close_time,
            created_time             = excluded.created_time,
            settlement_time          = excluded.settlement_time,
            status                   = excluded.status,
            result                   = excluded.result,
            settlement_value_dollars = excluded.settlement_value_dollars,
            template_id              = excluded.template_id,
            strike_value             = excluded.strike_value,
            strike_direction         = excluded.strike_direction,
            last_price_dollars       = excluded.last_price_dollars,
            volume_fp                = excluded.volume_fp,
            open_interest_fp         = excluded.open_interest_fp,
            last_refreshed_at        = excluded.last_refreshed_at
        """,
        {
            "ticker": market.ticker,
            "event_ticker": market.event_ticker,
            "series_ticker": market.series_ticker,
            "market_type": market.market_type,
            "title": market.title,
            "subtitle": market.subtitle,
            "yes_sub_title": market.yes_sub_title,
            "no_sub_title": market.no_sub_title,
            "rules_primary": market.rules_primary,
            "rules_secondary": market.rules_secondary,
            "open_time": market.open_time,
            "close_time": market.close_time,
            "created_time": market.created_time,
            "settlement_time": market.settlement_time,
            "status": market.status,
            "result": market.result,
            "settlement_value_dollars": market.settlement_value_dollars,
            "template_id": market.template_id,
            "strike_value": market.strike_value,
            "strike_direction": market.strike_direction,
            "last_price_dollars": market.last_price_dollars,
            "volume_fp": market.volume_fp,
            "open_interest_fp": market.open_interest_fp,
            "now": _now_iso(),
        },
    )


# ── kalshi_price_history ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PriceHistoryRow:
    """One daily candlestick of a Kalshi market's price history."""

    ticker: str
    period_end_ts: int
    period_end_date: str
    open_dollars: str | None = None
    high_dollars: str | None = None
    low_dollars: str | None = None
    close_dollars: str | None = None
    mean_dollars: str | None = None
    yes_bid_close: str | None = None
    yes_ask_close: str | None = None
    volume_fp: str | None = None
    open_interest_fp: str | None = None


def bulk_insert_price_history(
    conn: sqlite3.Connection, rows: Iterable[PriceHistoryRow]
) -> int:
    """Insert many candlesticks at once. Idempotent on (ticker, period_end_ts)."""
    payload = [
        {
            "ticker": r.ticker,
            "period_end_ts": r.period_end_ts,
            "period_end_date": r.period_end_date,
            "open_dollars": r.open_dollars,
            "high_dollars": r.high_dollars,
            "low_dollars": r.low_dollars,
            "close_dollars": r.close_dollars,
            "mean_dollars": r.mean_dollars,
            "yes_bid_close": r.yes_bid_close,
            "yes_ask_close": r.yes_ask_close,
            "volume_fp": r.volume_fp,
            "open_interest_fp": r.open_interest_fp,
        }
        for r in rows
    ]
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT INTO kalshi_price_history (
            ticker, period_end_ts, period_end_date,
            open_dollars, high_dollars, low_dollars, close_dollars, mean_dollars,
            yes_bid_close, yes_ask_close, volume_fp, open_interest_fp
        ) VALUES (
            :ticker, :period_end_ts, :period_end_date,
            :open_dollars, :high_dollars, :low_dollars, :close_dollars, :mean_dollars,
            :yes_bid_close, :yes_ask_close, :volume_fp, :open_interest_fp
        )
        ON CONFLICT(ticker, period_end_ts) DO UPDATE SET
            period_end_date  = excluded.period_end_date,
            open_dollars     = excluded.open_dollars,
            high_dollars     = excluded.high_dollars,
            low_dollars      = excluded.low_dollars,
            close_dollars    = excluded.close_dollars,
            mean_dollars     = excluded.mean_dollars,
            yes_bid_close    = excluded.yes_bid_close,
            yes_ask_close    = excluded.yes_ask_close,
            volume_fp        = excluded.volume_fp,
            open_interest_fp = excluded.open_interest_fp
        """,
        payload,
    )
    return len(payload)


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
