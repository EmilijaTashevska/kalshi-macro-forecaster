"""Top-level command-line interface.

Available subcommands:
    kalshi-train --version
    kalshi-train init-db                                (idempotent schema apply)
    kalshi-train db-info                                (rows-per-table summary)
    kalshi-train pit SERIES_ID --as-of YYYY-MM-DD       (point-in-time spot check)
    kalshi-train pit-history SERIES_ID --start --end    (PIT timeline)
    kalshi-train ingest fred [OPTIONS]                  (Phase 1.2 FRED ingest)
    kalshi-train ingest spf                             (Phase 1.3 SPF ingest)
    kalshi-train train fed-cut [OPTIONS]                (Phase 2 XGBoost baseline)

More subcommands arrive as we hit each phase.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from kalshi_train import __version__
from kalshi_train.config import settings
from kalshi_train.data.ingest_fred import run_fred_ingest
from kalshi_train.data.ingest_spf import run_spf_ingest
from kalshi_train.db.connection import connect, init_schema
from kalshi_train.db.point_in_time import (
    VintagePolicy,
    pit_history,
    pit_value,
)
from kalshi_train.training.phase2_fed_cut import run_phase2_fed_cut

app = typer.Typer(add_completion=False, help="Kalshi Model Train CLI.")
ingest_app = typer.Typer(add_completion=False, help="Data ingestion commands.")
train_app = typer.Typer(add_completion=False, help="Model training commands.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(train_app, name="train")
console = Console()


def _configure_logging(level: str = "INFO") -> None:
    """Pretty Rich-backed logging for CLI commands."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"kalshi-train {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """Kalshi Model Train CLI."""
    _ = version  # consumed by callback


@app.command("init-db")
def init_db_cmd() -> None:
    """Create / refresh the database schema at the configured path."""
    init_schema()
    console.print(f"[green]✓[/green] Schema applied to {settings.kalshi_train_db_path}")


@app.command("db-info")
def db_info_cmd() -> None:
    """Print a quick summary of the database contents."""
    with connect(read_only=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' "
            "ORDER BY name"
        ).fetchall()

        table = Table(title=f"Database: {settings.kalshi_train_db_path}")
        table.add_column("Table", style="cyan")
        table.add_column("Rows", justify="right", style="green")

        for r in rows:
            name = r["name"]
            count_row = conn.execute(f"SELECT COUNT(*) AS c FROM {name}").fetchone()
            count = count_row["c"] if count_row else 0
            table.add_row(name, f"{count:,}")
        console.print(table)


@app.command("pit")
def pit_cmd(
    series_id: str = typer.Argument(..., help="Series ID, e.g. CPIAUCSL."),
    as_of: str = typer.Option(..., "--as-of", help="As-of date, YYYY-MM-DD."),
    policy: str = typer.Option(
        "first_known_at",
        "--policy",
        help="Vintage policy: first_known_at | latest_revision.",
    ),
) -> None:
    """Resolve a series' value as it was known on a given date.

    Useful for spot-checking the leakage guard. Example::

        uv run kalshi-train pit CPIAUCSL --as-of 2024-11-07
    """
    vp = VintagePolicy(policy)
    val = pit_value(series_id, as_of, policy=vp)
    if val is None:
        console.print(
            f"[yellow]No observation of [bold]{series_id}[/bold] was knowable "
            f"on [bold]{as_of}[/bold] under policy {vp.value}.[/yellow]"
        )
    else:
        console.print(
            f"[green]{series_id}[/green] as of [bold]{as_of}[/bold] = "
            f"[bold cyan]{val}[/bold cyan]  ([dim]policy={vp.value}[/dim])"
        )


@app.command("pit-history")
def pit_history_cmd(
    series_id: str = typer.Argument(..., help="Series ID."),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD."),
    freq: str = typer.Option("D", "--freq", help="Pandas date-range frequency (D, B, W-FRI...)."),
    head: int = typer.Option(20, "--head", help="Rows to print."),
) -> None:
    """Print the first ``head`` rows of a point-in-time history."""
    df = pit_history(series_id, start, end, freq=freq)
    if df.empty:
        console.print("[yellow]Empty result.[/yellow]")
        return
    table = Table(title=f"PIT history for {series_id}  ({start} → {end}, freq={freq})")
    table.add_column("as_of_date", style="cyan")
    table.add_column("value", justify="right")
    table.add_column("observation_date", style="dim")
    table.add_column("vintage_date", style="dim")
    for ts, row in df.head(head).iterrows():
        table.add_row(
            ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts),
            "—" if row["value"] is None else f"{row['value']:.4f}",
            str(row["observation_date"] or "—"),
            str(row["vintage_date"] or "—"),
        )
    console.print(table)


@ingest_app.command("fred")
def ingest_fred_cmd(
    series: list[str] = typer.Option(  # noqa: B008
        [],
        "--series",
        "-s",
        help="Restrict to these series IDs. Repeatable. Default: all in registry.",
    ),
    skip_optional: bool = typer.Option(
        False,
        "--skip-optional",
        help="Skip series marked optional in the registry — fast smoke run.",
    ),
    observation_start: str | None = typer.Option(
        "2000-01-01",
        "--observation-start",
        help="Earliest observation date to fetch (YYYY-MM-DD).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help="Process only the first N entries after filtering.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG/INFO/WARNING/ERROR."),
) -> None:
    """Ingest FRED / ALFRED series into the local SQLite DB.

    Examples::

        # Smoke test: required series only, last ~25 years
        kalshi-train ingest fred --skip-optional --observation-start 2000-01-01

        # Single series, full vintage history
        kalshi-train ingest fred -s CPIAUCSL

        # First 5 entries (dev iteration)
        kalshi-train ingest fred --limit 5
    """
    _configure_logging(log_level)
    if settings.fred_api_key is None:
        console.print(
            "[red]FRED_API_KEY is not set.[/red] Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "and put it in .env."
        )
        raise typer.Exit(code=2)

    report = asyncio.run(
        run_fred_ingest(
            series_ids=series or None,
            include_optional=not skip_optional,
            observation_start=observation_start,
            limit=limit,
        )
    )

    table = Table(title="FRED ingest summary")
    table.add_column("series_id", style="cyan")
    table.add_column("rows", justify="right", style="green")
    table.add_column("status", style="dim")
    table.add_column("error", style="red")
    for r in report.results:
        table.add_row(
            r.series_id,
            f"{r.rows_inserted:,}",
            "ok" if r.success else "fail",
            (r.error or "")[:80],
        )
    console.print(table)
    console.print(
        f"[bold]Total:[/bold] {report.n_succeeded} ok, "
        f"{report.n_failed} failed, [green]{report.total_rows:,}[/green] rows."
    )


@ingest_app.command("spf")
def ingest_spf_cmd(
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG/INFO/WARNING/ERROR."),
) -> None:
    """Ingest the Philly Fed Survey of Professional Forecasters median series.

    Downloads the medianLevel.xlsx workbook from the Philadelphia Fed,
    extracts the variables we care about (CPI, core CPI, PCE, core
    PCE, real GDP, unemployment, T-bill, T-bond), and writes derived
    series (SPF_CPI_MEDIAN_NOWCAST, etc.) into the local SQLite DB.

    No API key required. Example::

        kalshi-train ingest spf
    """
    _configure_logging(log_level)
    report = asyncio.run(run_spf_ingest())

    table = Table(title="SPF ingest summary")
    table.add_column("series_id", style="cyan")
    table.add_column("rows", justify="right", style="green")
    table.add_column("status", style="dim")
    table.add_column("error", style="red")
    for r in report.results:
        table.add_row(
            r.series_id,
            f"{r.rows_inserted:,}",
            "ok" if r.success else "fail",
            (r.error or "")[:80],
        )
    console.print(table)
    console.print(
        f"[bold]Total:[/bold] {report.n_succeeded} ok, "
        f"{report.n_failed} failed, [green]{report.total_rows:,}[/green] rows."
    )


DEFAULT_REPORT_PATH = Path("reports/phase2_xgboost.md")


@train_app.command("fed-cut")
def train_fed_cut_cmd(
    start: str = typer.Option("2000-01-01", "--start", help="First meeting year (ISO date)."),
    end: str | None = typer.Option(None, "--end", help="Last meeting (default: today)."),
    report: Path | None = typer.Option(  # noqa: B008
        None,
        "--report",
        help="Markdown report output path.",
    ),
    no_report: bool = typer.Option(False, "--no-report", help="Skip writing the report file."),
) -> None:
    """Train the Phase 2 Fed-cut XGBoost baseline with temporal CV.

    Requires FRED series in the DB (``kalshi-train ingest fred``).
    """
    report_path = report or DEFAULT_REPORT_PATH
    try:
        result = run_phase2_fed_cut(
            start=start,
            end=end,
            report_path=report_path,
            write_report=not no_report,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    table = Table(title="Phase 2 — held-out test metrics")
    table.add_column("Model", style="cyan")
    table.add_column("Brier", justify="right")
    table.add_column("Log loss", justify="right")
    for name, metrics in result.test_metrics.items():
        table.add_row(name, f"{metrics.brier:.4f}", f"{metrics.log_loss:.4f}")
    console.print(table)
    console.print(
        f"[bold]{result.n_examples}[/bold] meetings, "
        f"[bold]{result.n_features}[/bold] features, "
        f"split {result.train_size}/{result.val_size}/{result.test_size}."
    )
    if not no_report:
        console.print(f"[green]Report:[/green] {result.report_path}")
    if result.reliability_plot:
        console.print(f"[green]Plot:[/green] {result.reliability_plot}")


if __name__ == "__main__":
    app()
