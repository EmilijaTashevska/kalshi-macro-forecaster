"""Print a human-readable summary of the local database.

Invoked via `make db-summary`. Designed to be re-run after every ingest
phase so you can spot-check that data landed where you expect.

Phase 0: just shows row counts per table + seed data preview.
Later phases extend with per-series coverage and per-source freshness.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kalshi_train.config import settings

console = Console()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _row_count(conn: sqlite3.Connection, name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])


def _print_header() -> None:
    db_path = settings.kalshi_train_db_path
    exists = Path(db_path).exists()
    size_kb = Path(db_path).stat().st_size / 1024 if exists else 0
    console.print(
        Panel.fit(
            f"[bold]Kalshi Model Train — Database Summary[/bold]\n"
            f"Path:    [cyan]{db_path}[/cyan]\n"
            f"Exists:  [green]yes[/green] ({size_kb:,.1f} KB)" if exists else
            f"Path:    [cyan]{db_path}[/cyan]\n"
            f"Exists:  [red]no[/red] — run `make db-init` first",
            border_style="blue",
        )
    )


def _print_row_counts(conn: sqlite3.Connection) -> None:
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' "
            "AND name NOT LIKE '%_segments' AND name NOT LIKE '%_docsize' "
            "AND name NOT LIKE '%_idx' AND name NOT LIKE '%_data' "
            "AND name NOT LIKE '%_config' "
            "ORDER BY name"
        ).fetchall()
    ]

    table = Table(title="Tables")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Phase", style="dim")

    phase_map = {
        "metadata": "0",
        "question_templates": "0 (seed)",
        "series_definitions": "1.2",
        "series_observations": "1.2",
        "text_documents": "1.4",
        "kalshi_markets": "1.5",
        "kalshi_price_history": "1.5",
        "polymarket_markets": "1.5",
        "polymarket_price_history": "1.5",
        "event_calendar": "1.6",
        "resolutions": "1.5/4",
        "ingest_runs": "1.x",
    }

    for name in tables:
        count = _row_count(conn, name)
        table.add_row(name, f"{count:,}", phase_map.get(name, "—"))

    console.print(table)


def _print_question_templates(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "question_templates"):
        return
    rows = conn.execute(
        "SELECT template_id, title, frequency, outcome_type FROM question_templates "
        "ORDER BY template_id"
    ).fetchall()
    if not rows:
        return
    table = Table(title="Question templates (seed data)")
    table.add_column("template_id", style="cyan")
    table.add_column("title")
    table.add_column("frequency", style="dim")
    table.add_column("outcome", style="dim")
    for r in rows:
        table.add_row(r[0], r[1], r[2], r[3])
    console.print(table)


def _print_metadata(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "metadata"):
        return
    rows = conn.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
    if not rows:
        return
    table = Table(title="metadata key-value")
    table.add_column("key", style="cyan")
    table.add_column("value")
    for r in rows:
        table.add_row(r[0], r[1])
    console.print(table)


def _print_series_coverage(conn: sqlite3.Connection) -> None:
    """Per-series coverage table — only printed once data exists."""
    if not _table_exists(conn, "series_definitions"):
        return
    if _row_count(conn, "series_definitions") == 0:
        return

    rows = conn.execute(
        """
        SELECT d.series_id,
               d.source,
               d.category,
               d.frequency,
               d.revises,
               d.first_seen,
               d.last_seen,
               COUNT(o.series_id)             AS n_rows,
               COUNT(DISTINCT o.observation_date) AS n_obs
        FROM series_definitions d
        LEFT JOIN series_observations o ON o.series_id = d.series_id
        GROUP BY d.series_id
        ORDER BY d.category, d.series_id
        """
    ).fetchall()

    table = Table(title="Series coverage")
    table.add_column("series_id", style="cyan")
    table.add_column("source", style="dim")
    table.add_column("category", style="dim")
    table.add_column("freq", style="dim")
    table.add_column("revises", justify="center")
    table.add_column("obs", justify="right")
    table.add_column("rows", justify="right", style="green")
    table.add_column("first", style="dim")
    table.add_column("last", style="dim")

    for r in rows:
        table.add_row(
            r["series_id"],
            r["source"],
            r["category"],
            r["frequency"],
            "✓" if r["revises"] else "·",
            f"{r['n_obs']:,}",
            f"{r['n_rows']:,}",
            r["first_seen"] or "—",
            r["last_seen"] or "—",
        )
    console.print(table)


def _print_recent_ingest_runs(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "ingest_runs"):
        return
    if _row_count(conn, "ingest_runs") == 0:
        return
    rows = conn.execute(
        "SELECT source, target, started_at, finished_at, status, "
        "rows_added, error_message FROM ingest_runs "
        "ORDER BY run_id DESC LIMIT 10"
    ).fetchall()
    table = Table(title="Last 10 ingest runs")
    table.add_column("source", style="cyan")
    table.add_column("target")
    table.add_column("started", style="dim")
    table.add_column("status", style="dim")
    table.add_column("rows", justify="right", style="green")
    table.add_column("error", style="red")
    for r in rows:
        table.add_row(
            r["source"],
            (r["target"] or "")[:40],
            (r["started_at"] or "")[:19],
            r["status"],
            f"{r['rows_added']:,}",
            (r["error_message"] or "")[:60],
        )
    console.print(table)


def main() -> None:
    _print_header()
    db_path = settings.kalshi_train_db_path
    if not Path(db_path).exists():
        return

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _print_row_counts(conn)
        console.print()
        _print_question_templates(conn)
        console.print()
        _print_series_coverage(conn)
        console.print()
        _print_recent_ingest_runs(conn)
        console.print()
        _print_metadata(conn)
        console.print(f"\n[dim]Generated at {datetime.now().isoformat(timespec='seconds')}[/dim]")


if __name__ == "__main__":
    main()
