"""External data sources.

Each source is a self-contained module exposing an async client. The
ingestion orchestrator (Phase 1.x) consumes them via a common interface.
"""

from __future__ import annotations
