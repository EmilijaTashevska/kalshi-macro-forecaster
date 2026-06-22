"""FRED ingestion orchestrator.

Walks the registry, fetches each series via the FRED/ALFRED client,
and writes through ``db.ingest`` helpers using a vintage-honest mapping
from FRED's response into our schema.

Mapping from FRED to ``series_observations``:

  FRED row                       │ Our column
  ───────────────────────────────┼───────────────────────────────────
  observation row['date']        │ observation_date
  realtime_start (first ever)    │ release_date  (computed by
                                 │ derive_release_date over all
                                 │ vintages of the same observation)
  realtime_start (per row)       │ vintage_date
  observation row['value']       │ value          ("." → NULL)

Notes on the release_date computation: FRED's realtime_start is a
date, but our schema's release_date is a datetime. Real economic
releases land at fixed intraday times (CPI is 8:30am ET, FOMC at
2pm ET, etc.) but FRED only records the calendar date.  We store the
end-of-day UTC timestamp for the recorded date — this is the most
conservative point-in-time semantic (treat the release as "available
from this date forward") and matches our PIT query's end-of-day
treatment of as_of_date.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kalshi_train.data.registry import (
    FRED_REGISTRY,
    FredSeriesEntry,
    all_series,
    find,
    required_series,
)
from kalshi_train.data.sources.fred import (
    FredClient,
    FredObservation,
    derive_release_date,
)
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    IngestRun,
    Observation,
    SeriesDefinition,
    bulk_insert_observations,
    record_ingest_run,
    upsert_series_definition,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SeriesIngestResult:
    """Summary of one series' ingest. Returned for reporting."""

    series_id: str
    success: bool
    rows_inserted: int = 0
    error: str | None = None


@dataclass(slots=True)
class FredIngestReport:
    """Aggregate result of an orchestrator run."""

    started_at: datetime
    finished_at: datetime | None
    results: list[SeriesIngestResult]

    @property
    def total_rows(self) -> int:
        return sum(r.rows_inserted for r in self.results)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


def _to_release_datetime(iso_date: str) -> str:
    """Treat 'YYYY-MM-DD' as end-of-day UTC, our conservative release timestamp."""
    return f"{iso_date}T23:59:59.999999+00:00"


def _entry_to_definition(entry: FredSeriesEntry, frequency_long: str, units: str,
                         seasonal_adjustment: str) -> SeriesDefinition:
    return SeriesDefinition(
        series_id=entry.series_id,
        source=entry.source,
        title=entry.title,
        frequency=_normalize_frequency(frequency_long),
        units=units,
        seasonal_adjustment=seasonal_adjustment,
        revises=entry.revises,
        category=entry.category,
        notes=entry.notes,
    )


def _normalize_frequency(fred_frequency: str) -> str:
    """FRED's frequency strings are like "Daily", "Monthly". We
    normalize to lowercase verbs used throughout the codebase."""
    f = fred_frequency.lower().strip()
    if "daily" in f:
        return "daily"
    if "week" in f:
        return "weekly"
    if "month" in f:
        return "monthly"
    if "quarter" in f:
        return "quarterly"
    if "annual" in f or "year" in f:
        return "annual"
    return f or "unknown"


def _to_observation_rows(
    series_id: str,
    fred_obs: list[FredObservation],
) -> list[Observation]:
    """Translate FRED vintage rows into our schema's ``Observation`` rows."""
    release_dates = derive_release_date(fred_obs)
    out: list[Observation] = []
    for o in fred_obs:
        # Skip rows where FRED has NULL value AND realtime semantics
        # of "not yet existed" — represented in FRED as value="." with
        # both realtime endpoints set. We keep the row anyway so the
        # vintage timeline is preserved; null-handling is the PIT
        # query's responsibility.
        out.append(
            Observation(
                series_id=series_id,
                observation_date=o.observation_date,
                release_date=_to_release_datetime(
                    release_dates.get(o.observation_date, o.realtime_start)
                ),
                vintage_date=o.realtime_start,
                value=o.value,
            )
        )
    return out


async def ingest_one_series(
    client: FredClient,
    entry: FredSeriesEntry,
    *,
    db_path: Path | None = None,
    observation_start: str | None = None,
) -> SeriesIngestResult:
    """Ingest a single series. Returns a result object even on failure
    so the orchestrator can report partial success cleanly."""
    logger.info("Ingesting %s (%s)", entry.series_id, entry.title)

    try:
        info = await client.get_series_info(entry.series_id)
        if entry.revises:
            obs = await client.get_observations_with_vintages(
                entry.series_id,
                observation_start=observation_start,
            )
        else:
            obs = await client.get_observations_current(
                entry.series_id,
                observation_start=observation_start,
            )
    except Exception as e:
        logger.exception("Failed to fetch %s", entry.series_id)
        return SeriesIngestResult(
            series_id=entry.series_id, success=False, error=str(e)
        )

    definition = _entry_to_definition(
        entry,
        frequency_long=info.frequency,
        units=info.units,
        seasonal_adjustment=info.seasonal_adjustment,
    )
    rows = _to_observation_rows(entry.series_id, obs)

    try:
        with connect(db_path) as conn:
            upsert_series_definition(conn, definition)
            inserted = bulk_insert_observations(conn, rows)
            # Update last_seen / first_seen via a single statement so
            # callers can find them in series_definitions afterwards.
            conn.execute(
                """
                UPDATE series_definitions
                   SET first_seen = (
                           SELECT MIN(observation_date)
                           FROM series_observations
                           WHERE series_id = ?
                       ),
                       last_seen  = (
                           SELECT MAX(observation_date)
                           FROM series_observations
                           WHERE series_id = ?
                       )
                 WHERE series_id = ?
                """,
                (entry.series_id, entry.series_id, entry.series_id),
            )
            conn.commit()
    except Exception as e:
        logger.exception("Failed to persist %s", entry.series_id)
        return SeriesIngestResult(
            series_id=entry.series_id, success=False, error=str(e)
        )

    logger.info("  → %s: stored %d rows", entry.series_id, inserted)
    return SeriesIngestResult(
        series_id=entry.series_id, success=True, rows_inserted=inserted
    )


async def run_fred_ingest(
    *,
    series_ids: Iterable[str] | None = None,
    include_optional: bool = True,
    observation_start: str | None = None,
    limit: int | None = None,
    db_path: Path | None = None,
    client: FredClient | None = None,
) -> FredIngestReport:
    """Run the full FRED ingestion.

    Parameters
    ----------
    series_ids:
        Restrict to these series. ``None`` means "use the registry".
    include_optional:
        If False, only ``required_series()`` are processed. Useful
        for fast smoke runs.
    observation_start:
        Earliest observation_date to fetch (ISO YYYY-MM-DD). FRED
        defaults to the start of the series if omitted.
    limit:
        Process only the first N entries after filtering. For dev.
    db_path / client:
        Test injection points.
    """
    entries: list[FredSeriesEntry]
    if series_ids is not None:
        entries = []
        for sid in series_ids:
            entry = find(sid)
            if entry is None:
                logger.warning("Skipping %s: not in registry", sid)
                continue
            entries.append(entry)
    else:
        entries = list(all_series() if include_optional else required_series())

    if limit is not None:
        entries = entries[:limit]

    started_at = datetime.now(tz=UTC)
    report = FredIngestReport(started_at=started_at, finished_at=None, results=[])

    # Open audit run
    audit_id: int | None = None
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="fred",
                target=f"{len(entries)} series",
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    owns_client = client is None
    client_ctx = FredClient() if client is None else client

    try:
        if owns_client:
            await client_ctx.__aenter__()
        for entry in entries:
            result = await ingest_one_series(
                client_ctx, entry,
                db_path=db_path, observation_start=observation_start,
            )
            report.results.append(result)
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    report.finished_at = datetime.now(tz=UTC)

    # Close audit run
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at = ?,
                   status      = ?,
                   rows_added  = ?,
                   rows_updated= 0,
                   error_message = ?
             WHERE run_id = ?
            """,
            (
                report.finished_at.isoformat(),
                "ok" if report.n_failed == 0 else "partial",
                report.total_rows,
                "; ".join(
                    f"{r.series_id}: {r.error}"
                    for r in report.results
                    if not r.success
                )[:2000],
                audit_id,
            ),
        )
        conn.commit()

    logger.info(
        "FRED ingest complete: %d succeeded, %d failed, %d total rows",
        report.n_succeeded, report.n_failed, report.total_rows,
    )
    return report


__all__ = [
    "FRED_REGISTRY",
    "FredIngestReport",
    "SeriesIngestResult",
    "ingest_one_series",
    "run_fred_ingest",
]
