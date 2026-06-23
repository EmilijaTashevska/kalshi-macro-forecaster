"""Unit tests for the Kalshi ingestion orchestrator (Phase 1.5b).

We drive the orchestrator with a fake client that returns canned
series / events / markets / candlesticks, so nothing hits the network.
Coverage:

  - allowlist filtering: non-macro series are dropped
  - market_to_row carries the series_ticker and classifies the strike
  - candles_to_rows derives period_end_date and pulls dollar fields
  - run_kalshi_ingest writes kalshi_markets + kalshi_price_history and
    records an ingest_runs audit row
  - --no-prices style runs skip candlestick fetches entirely
  - series-ticker restriction respects the allowlist
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kalshi_train.data.ingest_kalshi import (
    _parse_iso_ts,
    _ts_to_iso,
    candles_to_rows,
    market_to_row,
    run_kalshi_ingest,
)
from kalshi_train.data.sources.kalshi_models import Candlestick, Market
from kalshi_train.db.connection import connect

# ── canned API payloads ───────────────────────────────────────────────


def _fed_market(strike: str, result: str = "yes") -> dict[str, Any]:
    return {
        "ticker": f"KXFED-26JUN-T{strike}",
        "event_ticker": "KXFED-26JUN",
        "market_type": "binary",
        "title": f"Will the upper bound be above {strike}% after the meeting?",
        "subtitle": f"{strike}%",
        "yes_sub_title": f"Above {strike}%",
        "no_sub_title": f"Below {strike}%",
        "rules_primary": "Resolves yes if ...",
        "open_time": "2025-08-06T14:30:00Z",
        "close_time": "2026-06-17T17:55:00Z",
        "created_time": "2025-08-01T00:00:00Z",
        "settlement_ts": 1781700000,
        "status": "finalized",
        "result": result,
        "last_price_dollars": "0.9900",
        "volume_fp": "1367.23",
        "open_interest_fp": "52618.86",
    }


def _candle(ts: int, close: str) -> dict[str, Any]:
    return {
        "end_period_ts": ts,
        "price": {
            "open_dollars": close,
            "high_dollars": close,
            "low_dollars": close,
            "close_dollars": close,
            "mean_dollars": close,
        },
        "yes_bid": {"close_dollars": close},
        "yes_ask": {"close_dollars": "1.0000"},
        "volume_fp": "10.00",
        "open_interest_fp": "100.00",
    }


class _FakeKalshiClient:
    """Duck-typed Kalshi client serving canned data."""

    def __init__(self) -> None:
        self.candle_calls: list[tuple[str, str]] = []

    async def __aenter__(self) -> _FakeKalshiClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get_series_by_category(self, category: str) -> list[dict[str, Any]]:
        if category == "Economics":
            return [
                {"ticker": "KXFED", "title": "Fed decision"},
                {"ticker": "KXWHEAT", "title": "Wheat price"},  # not on allowlist
            ]
        return []  # Financials: nothing in this fixture

    async def iter_events(
        self, series_ticker: str, status: str | None = None
    ) -> AsyncIterator[list[dict[str, Any]]]:
        if series_ticker == "KXFED":
            yield [{"event_ticker": "KXFED-26JUN", "series_ticker": "KXFED"}]

    async def get_event_markets(self, event_ticker: str) -> list[dict[str, Any]]:
        if event_ticker == "KXFED-26JUN":
            return [_fed_market("2.75"), _fed_market("3.00", result="no")]
        return []

    async def get_live_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> list[Candlestick]:
        self.candle_calls.append((series_ticker, ticker))
        return [
            Candlestick.model_validate(_candle(1781568000, "0.5000")),
            Candlestick.model_validate(_candle(1781654400, "0.9900")),
        ]


# ── pure helpers ──────────────────────────────────────────────────────


def test_parse_iso_ts_roundtrips() -> None:
    ts = _parse_iso_ts("2025-08-06T14:30:00Z")
    assert ts == int(datetime(2025, 8, 6, 14, 30, tzinfo=UTC).timestamp())
    assert _parse_iso_ts(None) is None
    assert _parse_iso_ts("not-a-date") is None


def test_ts_to_iso_is_utc() -> None:
    assert _ts_to_iso(None) is None
    iso = _ts_to_iso(1781568000)
    assert iso is not None
    assert iso.endswith("+00:00")


def test_market_to_row_classifies_strike_and_keeps_series() -> None:
    market = Market.model_validate(_fed_market("2.75"))
    row = market_to_row(market, "KXFED")
    assert row.series_ticker == "KXFED"
    assert row.template_id == "fed_decision"
    assert row.strike_value == 2.75
    assert row.strike_direction == "above"
    assert row.result == "yes"
    # settlement_ts converted to an ISO timestamp
    assert row.settlement_time is not None


def test_candles_to_rows_derives_period_date() -> None:
    candles = [Candlestick.model_validate(_candle(1781568000, "0.5000"))]
    rows = candles_to_rows("KXFED-26JUN-T2.75", candles)
    assert len(rows) == 1
    r = rows[0]
    assert r.ticker == "KXFED-26JUN-T2.75"
    assert r.period_end_ts == 1781568000
    assert r.close_dollars == "0.5000"
    assert r.yes_ask_close == "1.0000"
    # period_end_date is the UTC calendar date of the timestamp
    expected = datetime.fromtimestamp(1781568000, tz=UTC).date().isoformat()
    assert r.period_end_date == expected


# ── full orchestrator runs ────────────────────────────────────────────


async def test_run_kalshi_ingest_writes_markets_and_prices(tmp_db: Path) -> None:
    client = _FakeKalshiClient()
    report = await run_kalshi_ingest(db_path=tmp_db, client=client)

    # Only KXFED is on the allowlist; KXWHEAT is dropped.
    assert report.n_succeeded == 1
    assert report.n_failed == 0
    assert report.total_markets == 2
    assert report.total_candles == 4  # 2 markets * 2 candles

    with connect(tmp_db, read_only=True) as conn:
        markets = conn.execute(
            "SELECT ticker, template_id, strike_value, series_ticker "
            "FROM kalshi_markets ORDER BY ticker"
        ).fetchall()
        assert {m["ticker"] for m in markets} == {
            "KXFED-26JUN-T2.75",
            "KXFED-26JUN-T3.00",
        }
        assert all(m["template_id"] == "fed_decision" for m in markets)
        assert all(m["series_ticker"] == "KXFED" for m in markets)

        prices = conn.execute(
            "SELECT COUNT(*) AS n FROM kalshi_price_history"
        ).fetchone()
        assert prices["n"] == 4

        audit = conn.execute(
            "SELECT source, status, rows_added, rows_updated FROM ingest_runs "
            "WHERE source = 'kalshi' ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        assert audit["status"] == "ok"
        assert audit["rows_added"] == 2  # markets
        assert audit["rows_updated"] == 4  # candles


async def test_run_kalshi_ingest_no_prices_skips_candles(tmp_db: Path) -> None:
    client = _FakeKalshiClient()
    report = await run_kalshi_ingest(db_path=tmp_db, client=client, fetch_prices=False)

    assert report.total_markets == 2
    assert report.total_candles == 0
    assert client.candle_calls == []  # never fetched candlesticks

    with connect(tmp_db, read_only=True) as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM kalshi_price_history").fetchone()
        assert n["n"] == 0


async def test_run_kalshi_ingest_is_idempotent(tmp_db: Path) -> None:
    client = _FakeKalshiClient()
    await run_kalshi_ingest(db_path=tmp_db, client=client)
    await run_kalshi_ingest(db_path=tmp_db, client=_FakeKalshiClient())

    with connect(tmp_db, read_only=True) as conn:
        markets = conn.execute("SELECT COUNT(*) AS n FROM kalshi_markets").fetchone()
        prices = conn.execute(
            "SELECT COUNT(*) AS n FROM kalshi_price_history"
        ).fetchone()
    # Re-running must not duplicate rows (upsert on primary keys).
    assert markets["n"] == 2
    assert prices["n"] == 4


async def test_run_kalshi_ingest_series_restriction_respects_allowlist(
    tmp_db: Path,
) -> None:
    client = _FakeKalshiClient()
    # KXWHEAT is not on the allowlist → resolves to zero targets.
    report = await run_kalshi_ingest(
        db_path=tmp_db, client=client, series_tickers=["KXWHEAT"]
    )
    assert report.results == []
    assert report.total_markets == 0
