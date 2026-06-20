# Kalshi Model Train — convenience commands.
# All commands run via `uv` so they use the project-managed Python.

.PHONY: help install dev-install sync lock test test-fast lint format typecheck check \
        db-init db-shell db-summary db-browser clean clean-cache \
        pre-commit-install pre-commit-run

help:
	@echo "Kalshi Model Train — available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install            Install runtime dependencies"
	@echo "    dev-install        Install with dev + data-sources extras"
	@echo "    sync               Sync the lockfile (after editing pyproject.toml)"
	@echo "    pre-commit-install Install git pre-commit hooks"
	@echo ""
	@echo "  Quality:"
	@echo "    lint               Run ruff lint"
	@echo "    format             Run ruff format (writes changes)"
	@echo "    typecheck          Run mypy --strict on src/"
	@echo "    test               Run all tests"
	@echo "    test-fast          Run tests excluding slow + integration"
	@echo "    check              Run lint + typecheck + test-fast"
	@echo ""
	@echo "  Database (read-only inspection):"
	@echo "    db-init            Initialize an empty database with the schema"
	@echo "    db-shell           Open an interactive SQLite shell"
	@echo "    db-summary         Print a one-shot DB summary report"
	@echo "    db-browser         Launch Datasette read-only web UI on :8001"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean              Remove build / cache artifacts"
	@echo "    clean-cache        Remove ruff/mypy/pytest caches only"

# ── Setup ──────────────────────────────────────────────────────────────

install:
	uv sync

dev-install:
	uv sync --extra dev --extra data-sources

sync:
	uv lock

pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

# ── Quality ────────────────────────────────────────────────────────────

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow and not integration"

check: lint typecheck test-fast

# ── Database ───────────────────────────────────────────────────────────

DB_PATH ?= data/kalshi_train.db

db-init:
	uv run python -m kalshi_train.scripts.init_db

db-shell:
	@echo "Opening SQLite shell on $(DB_PATH). Type .help for commands, .quit to exit."
	@sqlite3 -column -header $(DB_PATH)

db-summary:
	uv run python scripts/inspect_db.py

db-browser:
	@echo "Launching Datasette on http://localhost:8001 (read-only)."
	uv run datasette serve $(DB_PATH) --port 8001 --immutable $(DB_PATH)

# ── Cleanup ────────────────────────────────────────────────────────────

clean: clean-cache
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-cache:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .hypothesis .coverage htmlcov
