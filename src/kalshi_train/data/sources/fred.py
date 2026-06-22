"""Async FRED + ALFRED client.

We talk directly to https://api.stlouisfed.org/fred/* via httpx because:

  1. The official ``fredapi`` is sync-only and would awkwardly compose
     with our async ingestion pipeline.
  2. FRED's vintage history (the ALFRED feature) is exposed via two
     query parameters — ``realtime_start`` and ``realtime_end`` — on
     the regular observations endpoint. We need fine-grained control
     over those, which ``fredapi`` does not give us.
  3. The REST API is small and well-documented; rolling our own client
     is ~150 lines and gives us the rate-limit policy we want.

API key is loaded from ``settings.fred_api_key``. Missing key raises
at the first request, not at import.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import TracebackType
from typing import Any, Self

import httpx

from kalshi_train import config as _config

logger = logging.getLogger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_TIMEOUT_SECONDS = 30.0
RATE_LIMIT_DELAY = 0.6  # 100 requests/minute, well under FRED's ~120/min
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 1.5

# When asking for full vintage history we use this as realtime_start.
# FRED's first archived revision is 1959-01-01 for many series.
EARLIEST_REALTIME = "1900-01-01"
LATEST_REALTIME = "9999-12-31"


class FredAPIError(RuntimeError):
    """Raised when FRED returns an error or we cannot decode its response."""


class FredAuthError(FredAPIError):
    """No API key configured — set FRED_API_KEY in .env."""


# ── Lightweight value objects ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FredSeriesInfo:
    """A subset of the metadata FRED returns for ``/series``.

    We keep only what we'll persist into ``series_definitions``.
    """

    series_id: str
    title: str
    frequency: str           # "Daily", "Weekly", "Monthly", "Quarterly", ...
    frequency_short: str     # "D", "W", "M", "Q"
    units: str
    seasonal_adjustment: str
    last_updated: str        # ISO timestamp
    observation_start: str   # earliest observation_date FRED has
    observation_end: str     # latest observation_date FRED has


@dataclass(frozen=True, slots=True)
class FredObservation:
    """A single vintage of a single observation.

    For a non-revising series, exactly one row exists per
    ``observation_date``. For revising series, FRED returns multiple
    rows per observation_date — one per (value, realtime_window).
    """

    observation_date: str     # ISO date, the period this value describes
    realtime_start: str       # ISO date, when this vintage became current
    realtime_end: str         # ISO date, when this vintage stopped being current
    value: float | None       # FRED reports "." for missing data


# ── The client ────────────────────────────────────────────────────────


class FredClient:
    """Minimal async client over the FRED REST API.

    Usage::

        async with FredClient() as fred:
            info = await fred.get_series_info("CPIAUCSL")
            obs  = await fred.get_observations_with_vintages("CPIAUCSL")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        *,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ) -> None:
        if api_key is None:
            secret = _config.settings.fred_api_key
            if secret is None:
                raise FredAuthError(
                    "FRED_API_KEY is not set. Get a free key at "
                    "https://fred.stlouisfed.org/docs/api/api_key.html "
                    "and put it in .env."
                )
            api_key = secret.get_secret_value()
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._rate_limit_delay = rate_limit_delay
        self._client: httpx.AsyncClient | None = None
        # We serialize requests through a lock to honor the rate limit
        # even when a caller fan-outs many tasks concurrently.
        self._lock = asyncio.Lock()
        self._last_request_at: float = 0.0

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=DEFAULT_TIMEOUT_SECONDS,
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

    # ── Internal HTTP ──

    async def _wait_for_rate_limit(self) -> None:
        """Sleep just enough to honor the rate limit window."""
        loop = asyncio.get_event_loop()
        now = loop.time()
        elapsed = now - self._last_request_at
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_at = loop.time()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("FredClient must be used as an async context manager.")
        merged = {**params, "api_key": self._api_key, "file_type": "json"}

        for attempt in range(MAX_RETRIES):
            async with self._lock:
                await self._wait_for_rate_limit()
            try:
                resp = await self._client.get(path, params=merged)
            except httpx.RequestError as e:
                if attempt == MAX_RETRIES - 1:
                    raise FredAPIError(f"Network error on {path}: {e}") from e
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("FRED request error on %s: %s (sleep %.1fs, retry)", path, e, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 429:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning("FRED 429 on %s, sleeping %.1fs", path, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code >= 400:
                # FRED returns JSON {"error_code": ..., "error_message": ...}
                try:
                    body = resp.json()
                    msg = body.get("error_message", resp.text)
                except Exception:
                    msg = resp.text
                raise FredAPIError(
                    f"FRED returned HTTP {resp.status_code} for {path}: {msg}"
                )
            try:
                data: dict[str, Any] = resp.json()
            except Exception as e:
                raise FredAPIError(f"FRED returned non-JSON for {path}: {e}") from e
            return data
        raise FredAPIError(f"Exhausted retries for {path}")

    # ── Public methods ──

    async def get_series_info(self, series_id: str) -> FredSeriesInfo:
        """Fetch series metadata. Raises FredAPIError if unknown series."""
        data = await self._get("/series", {"series_id": series_id})
        seriess = data.get("seriess") or []
        if not seriess:
            raise FredAPIError(f"FRED returned no metadata for series {series_id!r}")
        s = seriess[0]
        return FredSeriesInfo(
            series_id=str(s["id"]),
            title=str(s.get("title", "")),
            frequency=str(s.get("frequency", "")),
            frequency_short=str(s.get("frequency_short", "")),
            units=str(s.get("units", "")),
            seasonal_adjustment=str(s.get("seasonal_adjustment_short", "")),
            last_updated=str(s.get("last_updated", "")),
            observation_start=str(s.get("observation_start", "")),
            observation_end=str(s.get("observation_end", "")),
        )

    async def get_observations_current(
        self,
        series_id: str,
        *,
        observation_start: str | date | None = None,
        observation_end: str | date | None = None,
    ) -> list[FredObservation]:
        """Return the current value of each observation, one row each.

        FRED's default behavior (no realtime params) tags every row
        with realtime_start = today, which is useless for point-in-time
        because it makes every historical observation look like it was
        first knowable today. THIS METHOD'S CALLER MUST NOT trust the
        returned ``realtime_start``; instead, the orchestrator
        overrides it with ``observation_date`` for non-revising series
        (whose values are by definition known at observation time).

        We use this rather than a full-history vintage query because
        FRED caps the number of vintages it will return per request,
        and high-frequency daily series (yields, futures) blow through
        that cap. Since non-revising series have no vintage history
        worth preserving anyway, this is the right trade-off.
        """
        return await self._fetch_observations(
            series_id=series_id,
            realtime_start=None,
            realtime_end=None,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    async def get_observations_with_vintages(
        self,
        series_id: str,
        *,
        observation_start: str | date | None = None,
        observation_end: str | date | None = None,
        realtime_start: str | date = EARLIEST_REALTIME,
        realtime_end: str | date = LATEST_REALTIME,
    ) -> list[FredObservation]:
        """Return the FULL vintage history for each observation.

        Default ``realtime_start`` / ``realtime_end`` cover all of FRED's
        archive. Every (observation_date, realtime_window) pair becomes
        its own row, which is exactly what we want for ALFRED-style
        backtesting.
        """
        return await self._fetch_observations(
            series_id=series_id,
            realtime_start=_iso(realtime_start),
            realtime_end=_iso(realtime_end),
            observation_start=observation_start,
            observation_end=observation_end,
        )

    async def _fetch_observations(
        self,
        *,
        series_id: str,
        realtime_start: str | None,
        realtime_end: str | None,
        observation_start: str | date | None,
        observation_end: str | date | None,
    ) -> list[FredObservation]:
        params: dict[str, Any] = {"series_id": series_id}
        if realtime_start is not None:
            params["realtime_start"] = realtime_start
        if realtime_end is not None:
            params["realtime_end"] = realtime_end
        if observation_start is not None:
            params["observation_start"] = _iso(observation_start)
        if observation_end is not None:
            params["observation_end"] = _iso(observation_end)
        # FRED capping is 100_000; well above any series we touch.
        params["limit"] = 100_000

        data = await self._get("/series/observations", params)
        raw = data.get("observations") or []
        out: list[FredObservation] = []
        for r in raw:
            value_str = r.get("value", "")
            value: float | None
            if value_str in {"", ".", None}:
                value = None
            else:
                try:
                    value = float(value_str)
                except (TypeError, ValueError):
                    value = None
            out.append(
                FredObservation(
                    observation_date=str(r["date"]),
                    realtime_start=str(r["realtime_start"]),
                    realtime_end=str(r["realtime_end"]),
                    value=value,
                )
            )
        return out


def _iso(value: str | date | datetime) -> str:
    """Normalize input to an ISO date string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def derive_release_date(observations: Sequence[FredObservation]) -> dict[str, str]:
    """Given all vintages for a series, return ``{observation_date: first_realtime_start}``.

    The "release date" of an observation is the earliest ``realtime_start``
    at which that observation existed in FRED — i.e. the day the value
    was first published. We persist this so the PIT query can enforce
    "was this observation published by as_of_date?".
    """
    earliest: dict[str, str] = {}
    for obs in observations:
        prev = earliest.get(obs.observation_date)
        if prev is None or obs.realtime_start < prev:
            earliest[obs.observation_date] = obs.realtime_start
    return earliest
