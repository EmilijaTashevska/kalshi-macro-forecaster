"""Unit tests for the FRED client.

We use httpx's ``MockTransport`` to intercept calls; no real network
involved. This validates URL/param construction, response decoding,
retry behavior, and rate-limit pacing.
"""

from __future__ import annotations

import json

import httpx
import pytest

import kalshi_train.data.sources.fred as fred_mod
from kalshi_train import config as cfg
from kalshi_train.data.sources.fred import (
    FredAPIError,
    FredAuthError,
    FredClient,
    FredObservation,
    derive_release_date,
)


def _make_client(transport: httpx.MockTransport) -> FredClient:
    """Build a FredClient that talks to a mock transport, no real network."""
    client = FredClient(api_key="TEST_KEY", rate_limit_delay=0.0)
    # We override the AsyncClient that's created in __aenter__. Setting
    # the protected attr is the simplest reliable way; the alternative
    # is wiring transport plumbing through the constructor and we
    # prefer to keep the constructor simple for production callers.
    client._client = httpx.AsyncClient(
        base_url="https://api.stlouisfed.org/fred",
        transport=transport,
    )
    return client


async def test_get_series_info_decodes_fred_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/fred/series"
        # The api_key + file_type should be appended
        assert request.url.params["api_key"] == "TEST_KEY"
        assert request.url.params["file_type"] == "json"
        return httpx.Response(
            200,
            json={
                "seriess": [
                    {
                        "id": "CPIAUCSL",
                        "title": "CPI: All Urban Consumers",
                        "frequency": "Monthly",
                        "frequency_short": "M",
                        "units": "Index 1982-1984=100",
                        "seasonal_adjustment_short": "SA",
                        "last_updated": "2025-09-11 08:30:00-05",
                        "observation_start": "1947-01-01",
                        "observation_end": "2025-08-01",
                    }
                ]
            },
        )

    client = _make_client(httpx.MockTransport(handler))
    info = await client.get_series_info("CPIAUCSL")
    assert info.series_id == "CPIAUCSL"
    assert info.frequency == "Monthly"
    assert info.seasonal_adjustment == "SA"
    assert info.observation_start == "1947-01-01"
    await client._client.aclose()


async def test_get_observations_with_vintages_parses_dot_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # The vintage call sets realtime_start / realtime_end
        assert request.url.path == "/fred/series/observations"
        assert "realtime_start" in request.url.params
        assert "realtime_end" in request.url.params
        return httpx.Response(
            200,
            json={
                "observations": [
                    {
                        "date": "2024-08-01",
                        "realtime_start": "2024-10-10",
                        "realtime_end": "2025-01-14",
                        "value": "2.4",
                    },
                    {
                        "date": "2024-08-01",
                        "realtime_start": "2025-01-15",
                        "realtime_end": "9999-12-31",
                        "value": "2.5",
                    },
                    # Missing value — FRED uses literal "."
                    {
                        "date": "2024-09-01",
                        "realtime_start": "2024-11-13",
                        "realtime_end": "9999-12-31",
                        "value": ".",
                    },
                ]
            },
        )

    client = _make_client(httpx.MockTransport(handler))
    obs = await client.get_observations_with_vintages("CPIAUCSL")
    assert len(obs) == 3
    assert obs[0].value == 2.4
    assert obs[1].value == 2.5
    assert obs[2].value is None
    await client._client.aclose()


async def test_get_observations_current_does_not_set_realtime() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # No realtime params on the current call — returns only the
        # latest vintage of each observation.
        assert "realtime_start" not in request.url.params
        assert "realtime_end" not in request.url.params
        return httpx.Response(
            200,
            json={
                "observations": [
                    {
                        "date": "2024-12-31",
                        "realtime_start": "2024-12-31",
                        "realtime_end": "9999-12-31",
                        "value": "4.25",
                    }
                ]
            },
        )

    client = _make_client(httpx.MockTransport(handler))
    obs = await client.get_observations_current("DGS10")
    assert len(obs) == 1
    assert obs[0].value == 4.25
    await client._client.aclose()


async def test_429_triggers_retry_then_success() -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] == 1:
            return httpx.Response(429, text="Too many requests")
        return httpx.Response(200, json={"seriess": [{
            "id": "X", "title": "X", "frequency": "Daily",
            "frequency_short": "D", "units": "", "seasonal_adjustment_short": "",
            "last_updated": "", "observation_start": "", "observation_end": "",
        }]})

    client = _make_client(httpx.MockTransport(handler))
    # Patch the retry backoff base low so the test is fast.
    orig = fred_mod.RETRY_BACKOFF_BASE
    fred_mod.RETRY_BACKOFF_BASE = 1.0
    try:
        info = await client.get_series_info("X")
    finally:
        fred_mod.RETRY_BACKOFF_BASE = orig
    assert info.series_id == "X"
    assert counter["n"] == 2  # one 429, one success
    await client._client.aclose()


async def test_4xx_raises_fred_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            text=json.dumps({"error_code": 400, "error_message": "bad request"}),
        )

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(FredAPIError, match="bad request"):
        await client.get_series_info("X")
    await client._client.aclose()


def test_missing_api_key_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the env has no key, instantiation must fail loudly."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "settings", cfg.Settings())
    with pytest.raises(FredAuthError):
        FredClient()


def test_derive_release_date_picks_earliest_realtime_start() -> None:
    obs = [
        FredObservation("2024-08-01", "2025-01-15", "9999-12-31", 2.5),
        FredObservation("2024-08-01", "2024-10-10", "2025-01-14", 2.4),
        FredObservation("2024-09-01", "2024-11-13", "9999-12-31", 2.4),
    ]
    out = derive_release_date(obs)
    # August: earliest is 2024-10-10 (the original release, not the revision)
    assert out["2024-08-01"] == "2024-10-10"
    # September only has one vintage
    assert out["2024-09-01"] == "2024-11-13"
