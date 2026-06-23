"""Kalshi API client.

Adapted from the author's earlier project
``black-swan-event-intelligence/backend/kalshi.py``
(https://github.com/emilija-tashevska/black-swan-event-intelligence).

Changes from the original:

- Lives inside the ``kalshi_train`` package; uses our internal pydantic
  models from ``kalshi_train.data.sources.kalshi_models``.
- All public methods are async; pagination yields batches to allow
  streaming inserts.
- Tightened types (strict mypy compatible).
- Dropped the Black Swan-specific filters; we want every macro market.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import httpx

from kalshi_train.data.sources.kalshi_models import Candlestick

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_LIMIT = 1000
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
RATE_LIMIT_DELAY = 0.15  # seconds between requests; conservative


class KalshiClient:
    """Minimal async client for Kalshi's public read endpoints.

    Use as an async context manager::

        async with KalshiClient() as client:
            cutoff = await client.get_cutoff()
            async for batch in client.paginate_batches(
                "/markets", "markets", {"status": "settled"}
            ):
                ...
    """

    def __init__(self, base_url: str = BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Internal HTTP helpers ──────────────────────────────────────

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("KalshiClient must be used as an async context manager.")
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(RATE_LIMIT_DELAY)
                resp = await self._client.get(path, params=params)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF * (attempt + 1)
                    logger.warning("Rate limited, waiting %.1fs", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            except httpx.HTTPStatusError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning("HTTP %s on %s, retrying", e.response.status_code, path)
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            except httpx.RequestError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning("Request error on %s: %s, retrying", path, e)
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                return data
        return {}

    async def paginate_batches(
        self,
        path: str,
        key: str,
        params: dict[str, Any] | None = None,
        batch_size: int = DEFAULT_LIMIT,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield successive pages from a paginated endpoint."""
        params = dict(params or {})
        params.setdefault("limit", batch_size)
        total = 0

        while True:
            data = await self._get(path, params)
            items: list[dict[str, Any]] = data.get(key, [])
            if items:
                total += len(items)
                yield items
            cursor = data.get("cursor", "")
            if not cursor or len(items) < int(params["limit"]):
                break
            params["cursor"] = cursor
            logger.info("Fetched %d %s so far", total, key)

    # ── Public endpoints ───────────────────────────────────────────

    async def get_cutoff(self) -> dict[str, Any]:
        """Return the historical/live cutoff timestamp."""
        return await self._get("/historical/cutoff")

    async def get_series_by_category(self, category: str) -> list[dict[str, Any]]:
        """List every series in a category (e.g. "Economics", "Financials").

        The ``/series`` endpoint returns the full category in one shot
        (no pagination), so we return the list directly.
        """
        data = await self._get("/series", {"category": category})
        series: list[dict[str, Any]] = data.get("series", [])
        return series

    async def iter_events(
        self,
        series_ticker: str,
        status: str | None = None,
        batch_size: int = 200,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield pages of events for a series.

        ``status`` filters server-side ("settled", "active", ...);
        ``None`` returns every event regardless of status.
        """
        params: dict[str, Any] = {"series_ticker": series_ticker}
        if status:
            params["status"] = status
        async for batch in self.paginate_batches("/events", "events", params, batch_size):
            yield batch

    async def get_event(self, event_ticker: str) -> dict[str, Any]:
        data = await self._get(f"/events/{event_ticker}")
        event = data.get("event", data)
        if not isinstance(event, dict):
            return {}
        return event

    async def get_event_markets(self, event_ticker: str) -> list[dict[str, Any]]:
        """Return the markets nested inside an event.

        We request ``with_nested_markets=true`` so the markets come back
        in the same call as the event detail — one round-trip per event.
        """
        data = await self._get(
            f"/events/{event_ticker}", {"with_nested_markets": "true"}
        )
        event = data.get("event", data)
        if not isinstance(event, dict):
            return []
        markets: list[dict[str, Any]] = event.get("markets", [])
        return markets

    async def get_historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> list[Candlestick]:
        data = await self._get(
            f"/historical/markets/{ticker}/candlesticks",
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        raw = data.get("candlesticks", [])
        return [Candlestick.model_validate(c) for c in raw]

    async def get_live_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> list[Candlestick]:
        data = await self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        raw = data.get("candlesticks", [])
        return [Candlestick.model_validate(c) for c in raw]
