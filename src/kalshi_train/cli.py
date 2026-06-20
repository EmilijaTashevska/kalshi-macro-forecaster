"""Top-level command-line interface.

Phase 0 supports:
    kalshi-train --version
    kalshi-train init-db     (idempotent schema apply)
    kalshi-train db-info     (rows-per-table summary)

More subcommands arrive as we hit each phase.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from kalshi_train import __version__
from kalshi_train.config import settings
from kalshi_train.db.connection import connect, init_schema

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


if __name__ == "__main__":
    app()
