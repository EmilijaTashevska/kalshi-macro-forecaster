"""SPF ingestion orchestrator.

Downloads the medianLevel.xlsx workbook from the Philadelphia Fed,
extracts the columns for the variables in
``kalshi_train.data.spf_registry``, and writes one row per
(variable, horizon, survey_quarter) tuple into ``series_observations``
as derived series.

Vintage semantics:
    SPF forecasts are PUBLISHED quarterly and do not get revised after
    publication. Each row is therefore a single-vintage observation.
    The mapping into our schema is:

        observation_date = first day of the SURVEY quarter
                           (e.g. 2024-01-01 for the 2024:Q1 survey)
        release_date     = approximate publication date of that survey
                           (we use the 15th of the second month of the
                           quarter, which is conservative: real releases
                           land between the 8th-12th of that month)
        vintage_date     = same calendar date as release_date
        value            = the forecast value (float)

We treat each SPF release as a single-vintage point-in-time fact:
``pit_value("SPF_CPI_MEDIAN_Q1", as_of="2024-11-07")`` returns the
median 1-quarter-ahead CPI forecast from the most recent SPF release
known on November 7, 2024 (the 2024:Q4 survey, released ~Nov 8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from kalshi_train.data.sources.spf import SPFClient, SPFWorkbook
from kalshi_train.data.spf_registry import SPF_VARIABLES, SPFVariable
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    IngestRun,
    Observation,
    SeriesDefinition,
    bulk_insert_observations,
    record_ingest_run,
    upsert_series_definition,
)

logger = logging.getLogger(__name__)


# Month-of-quarter when SPF is published (the SECOND month). The actual
# release lands around the 8th-12th of this month historically; we use
# the 15th to be conservative (treats data as not yet known a few days
# longer than reality).
_RELEASE_MONTH_BY_QUARTER = {1: 2, 2: 5, 3: 8, 4: 11}
_RELEASE_DAY = 15


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass(slots=True)
class SPFSeriesResult:
    series_id: str
    rows_inserted: int = 0
    success: bool = True
    error: str | None = None


@dataclass(slots=True)
class SPFIngestReport:
    started_at: datetime
    finished_at: datetime | None
    results: list[SPFSeriesResult]

    @property
    def total_rows(self) -> int:
        return sum(r.rows_inserted for r in self.results)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


# ── Helpers ───────────────────────────────────────────────────────────


def _release_date_for(year: int, quarter: int) -> date:
    """Approximate SPF release date for a given survey quarter."""
    month = _RELEASE_MONTH_BY_QUARTER[quarter]
    return date(year, month, _RELEASE_DAY)


def _quarter_start(year: int, quarter: int) -> date:
    return date(year, (quarter - 1) * 3 + 1, 1)


def _to_release_datetime(d: date) -> str:
    return f"{d.isoformat()}T23:59:59.999999+00:00"


def _to_observation_rows(
    variable: SPFVariable,
    sheet_df: pd.DataFrame,
) -> dict[str, list[Observation]]:
    """Translate one SPF worksheet into rows keyed by derived series_id.

    The sheet has columns YEAR, QUARTER, and one column per horizon
    (e.g. CPI2, CPI3, CPIA, CPIB). For each (row, horizon) combination
    we emit one ``Observation`` per derived series_id we care about.

    NaN cells are skipped — they represent surveys where the forecaster
    panel did not produce a value for that horizon (e.g. early
    1968-1980 surveys had narrower horizon coverage).
    """
    by_series: dict[str, list[Observation]] = {
        sid: [] for sid in variable.horizon_to_series_id.values()
    }

    if "YEAR" not in sheet_df.columns or "QUARTER" not in sheet_df.columns:
        raise ValueError(
            f"SPF sheet {variable.spf_sheet} missing YEAR/QUARTER columns; "
            f"found columns: {list(sheet_df.columns)}"
        )

    for _, row in sheet_df.iterrows():
        try:
            year = int(row["YEAR"])
            quarter = int(row["QUARTER"])
        except (ValueError, TypeError):
            # Some sheets have trailing empty rows or footnote text.
            continue
        if quarter not in (1, 2, 3, 4):
            continue

        obs_date = _quarter_start(year, quarter)
        rel_date = _release_date_for(year, quarter)
        rel_iso = _to_release_datetime(rel_date)
        vint_iso = rel_date.isoformat()

        for horizon_key, series_id in variable.horizon_to_series_id.items():
            col_name = f"{variable.spf_sheet}{horizon_key}"
            if col_name not in sheet_df.columns:
                continue
            raw = row[col_name]
            if pd.isna(raw):
                continue
            try:
                value = float(raw)
            except (ValueError, TypeError):
                continue
            by_series[series_id].append(
                Observation(
                    series_id=series_id,
                    observation_date=obs_date.isoformat(),
                    release_date=rel_iso,
                    vintage_date=vint_iso,
                    value=value,
                )
            )

    return by_series


def _make_definition(variable: SPFVariable, horizon_key: str, series_id: str) -> SeriesDefinition:
    label = variable.horizon_to_label.get(horizon_key, horizon_key)
    title = (
        f"SPF median forecast: {variable.description} ({label})"
    )
    return SeriesDefinition(
        series_id=series_id,
        source="SPF",
        title=title[:200],
        frequency="quarterly",
        units="",
        seasonal_adjustment="",
        revises=False,
        category=variable.category,
        notes=(
            f"Derived from SPF sheet {variable.spf_sheet}, "
            f"horizon column {variable.spf_sheet}{horizon_key}."
        ),
    )


# ── Per-variable ingest ───────────────────────────────────────────────


def ingest_workbook(
    workbook: SPFWorkbook,
    variables: list[SPFVariable] | None = None,
    *,
    db_path: Path | None = None,
) -> list[SPFSeriesResult]:
    """Ingest derived series from a parsed SPF workbook.

    This is sync because the heavy lifting is pandas / SQLite, both
    sync. The async surface is at the network layer (the client).
    """
    variables = list(variables if variables is not None else SPF_VARIABLES)
    results: list[SPFSeriesResult] = []

    with connect(db_path) as conn:
        for var in variables:
            sheet = workbook.sheets.get(var.spf_sheet)
            if sheet is None:
                logger.warning(
                    "SPF workbook is missing expected sheet %s; skipping",
                    var.spf_sheet,
                )
                for sid in var.horizon_to_series_id.values():
                    results.append(
                        SPFSeriesResult(
                            series_id=sid,
                            rows_inserted=0,
                            success=False,
                            error=f"sheet {var.spf_sheet} not in workbook",
                        )
                    )
                continue

            try:
                grouped = _to_observation_rows(var, sheet)
            except Exception as e:
                logger.exception("Parsing %s failed", var.spf_sheet)
                for sid in var.horizon_to_series_id.values():
                    results.append(
                        SPFSeriesResult(
                            series_id=sid, success=False, error=str(e)
                        )
                    )
                continue

            for horizon_key, series_id in var.horizon_to_series_id.items():
                rows = grouped.get(series_id, [])
                try:
                    upsert_series_definition(
                        conn, _make_definition(var, horizon_key, series_id)
                    )
                    inserted = bulk_insert_observations(conn, rows)
                    conn.execute(
                        """
                        UPDATE series_definitions
                           SET first_seen = (
                                   SELECT MIN(observation_date)
                                   FROM series_observations
                                   WHERE series_id = ?
                               ),
                               last_seen = (
                                   SELECT MAX(observation_date)
                                   FROM series_observations
                                   WHERE series_id = ?
                               )
                         WHERE series_id = ?
                        """,
                        (series_id, series_id, series_id),
                    )
                except Exception as e:
                    logger.exception("Persisting %s failed", series_id)
                    results.append(
                        SPFSeriesResult(
                            series_id=series_id, success=False, error=str(e)
                        )
                    )
                    continue

                results.append(
                    SPFSeriesResult(series_id=series_id, rows_inserted=inserted)
                )
                logger.info("  → %s: stored %d rows", series_id, inserted)
        conn.commit()

    return results


# ── Top-level orchestrator ────────────────────────────────────────────


async def run_spf_ingest(
    *,
    db_path: Path | None = None,
    client: SPFClient | None = None,
) -> SPFIngestReport:
    """Download + ingest the SPF medianLevel workbook end-to-end."""
    started_at = datetime.now(tz=UTC)

    audit_id: int = 0
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="spf",
                target="medianLevel.xlsx",
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    owns_client = client is None
    client_ctx: SPFClient = SPFClient() if client is None else client
    try:
        if owns_client:
            await client_ctx.__aenter__()
        workbook = await client_ctx.get_median_level()
        results = ingest_workbook(workbook, db_path=db_path)
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    finished_at = datetime.now(tz=UTC)
    report = SPFIngestReport(
        started_at=started_at, finished_at=finished_at, results=results
    )

    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at = ?,
                   status      = ?,
                   rows_added  = ?,
                   error_message = ?
             WHERE run_id = ?
            """,
            (
                finished_at.isoformat(),
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
        "SPF ingest complete: %d series ok, %d failed, %d total rows",
        report.n_succeeded, report.n_failed, report.total_rows,
    )
    return report


__all__ = [
    "SPFIngestReport",
    "SPFSeriesResult",
    "ingest_workbook",
    "run_spf_ingest",
]
