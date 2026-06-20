"""Standalone DB init for `make db-init`. Equivalent to `kalshi-train init-db`."""

from __future__ import annotations

from kalshi_train.config import settings
from kalshi_train.db.connection import init_schema


def main() -> None:
    init_schema()
    print(f"✓ Schema applied to {settings.kalshi_train_db_path}")


if __name__ == "__main__":
    main()
