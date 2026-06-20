"""Database access layer.

The schema lives in `schema.sql` (single source of truth, read by both
sync and async code). Connection helpers live in `connection.py`. The
ONLY legal way to read time-series data for ML purposes is via the
`point_in_time` module — using raw SQL bypasses our leakage guards.
"""

from __future__ import annotations
