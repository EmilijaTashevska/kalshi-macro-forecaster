"""Top-level command-line interface.

Phase 0/1.1 subcommands:
    kalshi-train --version
    kalshi-train init-db                                (idempotent schema apply)
    kalshi-train db-info                                (rows-per-table summary)
    kalshi-train pit SERIES_ID --as-of YYYY-MM-DD       (point-in-time spot check)
    kalshi-train pit-history SERIES_ID --start --end    (PIT timeline)

More subcommands arrive as we hit each phase.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from kalshi_train import __version__
from kalshi_train.config import settings
from kalshi_train.db.connection import connect, init_schema
from kalshi_train.db.point_in_time import (
    VintagePolicy,
    pit_history,
    pit_value,
)

app = typer.Typer(add_completion=False, help="Kalshi Model Train CLI.")
console = Console()


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


if __name__ == "__main__":
    app()
