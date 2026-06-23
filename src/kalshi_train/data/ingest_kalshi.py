"""Kalshi ingestion orchestrator (Phase 1.5b).

Walks Kalshi's macro markets and persists the ones that map onto our 7
question templates, together with their daily price history.

Traversal (see ``docs/phase_1_5_kalshi_research.md``)::

    /series?category=Economics|Financials   → candidate series
        ↓  (keep only allowlisted series)
    /events?series_ticker=...                → events under each series
        ↓
    /events/{event_ticker}?with_nested_markets=true
                                             → markets inside the event
        ↓
    /series/{series}/markets/{ticker}/candlesticks
                                             → daily price history

Two real-world quirks this module bakes in, learned during recon:

1. The series_ticker is a property of the *event*, not the market, so we
   carry it down from the series loop and hand it to the classifier and
   the candlestick fetch.
2. The ``/historical/markets/.../candlesticks`` endpoint 404s for these
   markets; the working endpoint is the series-scoped
   ``/series/{series}/markets/{ticker}/candlesticks`` one, i.e.
   :meth:`KalshiClient.get_live_candlesticks`.

Classification is intentionally permissive: a market in an allowlisted
series is always stored (its price history is useful) even when the
strike can't be parsed — those rows just get a NULL ``strike_value``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kalshi_train.data.kalshi_classifier import classify_market, classify_series
from kalshi_train.data.sources.kalshi import KalshiClient
from kalshi_train.data.sources.kalshi_models import Candlestick, Market
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    IngestRun,
    KalshiMarketRow,
    PriceHistoryRow,
    bulk_insert_price_history,
    record_ingest_run,
    upsert_kalshi_market,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("Economics", "Financials")
DAILY_INTERVAL = 1440  # minutes per candlestick period


@dataclass(slots=True)
class SeriesIngestResult:
    """Summary of one series' ingest. Returned for reporting."""

    series_ticker: str
    template_id: str
    markets_ingested: int = 0
    candles_ingested: int = 0
    success: bool = True
    error: str | None = None


@dataclass(slots=True)
class KalshiIngestReport:
    """Aggregate result of an orchestrator run."""

    started_at: datetime
    finished_at: datetime | None = None
    results: list[SeriesIngestResult] = field(default_factory=list)

    @property
    def total_markets(self) -> int:
        return sum(r.markets_ingested for r in self.results)

    @property
    def total_candles(self) -> int:
        return sum(r.candles_ingested for r in self.results)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


# ── conversion helpers ─────────────────────────────────────────────────


def _parse_iso_ts(value: str | None) -> int | None:
    """Parse an ISO timestamp string into unix seconds (UTC)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def _ts_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def market_to_row(market: Market, series_ticker: str) -> KalshiMarketRow:
    """Build a persistable row from a parsed market + its series ticker."""
    result = classify_market(
        series_ticker=series_ticker,
        yes_sub_title=market.yes_sub_title,
        title=market.title,
    )
    return KalshiMarketRow(
        ticker=market.ticker,
        event_ticker=market.event_ticker,
        series_ticker=series_ticker,
        market_type=market.market_type,
        title=market.title,
        subtitle=market.subtitle,
        yes_sub_title=market.yes_sub_title,
        no_sub_title=market.no_sub_title,
        rules_primary=market.rules_primary,
        rules_secondary=market.rules_secondary,
        open_time=market.open_time,
        close_time=market.close_time,
        created_time=market.created_time,
        settlement_time=_ts_to_iso(market.settlement_ts),
        status=market.status,
        result=market.result,
        settlement_value_dollars=market.settlement_value_dollars,
        template_id=result.template_id,
        strike_value=result.strike_value,
        strike_direction=result.strike_direction,
        last_price_dollars=market.last_price_dollars or "0.0000",
        volume_fp=market.volume_fp or "0.00",
        open_interest_fp=market.open_interest_fp or "0.00",
    )


def candles_to_rows(ticker: str, candles: list[Candlestick]) -> list[PriceHistoryRow]:
    """Translate API candlesticks into price-history rows."""
    rows: list[PriceHistoryRow] = []
    for c in candles:
        period_date = datetime.fromtimestamp(c.end_period_ts, tz=UTC).date().isoformat()
        rows.append(
            PriceHistoryRow(
                ticker=ticker,
                period_end_ts=c.end_period_ts,
                period_end_date=period_date,
                open_dollars=c.price.get_open(),
                high_dollars=c.price.get_high(),
                low_dollars=c.price.get_low(),
                close_dollars=c.price.get_close(),
                mean_dollars=c.price.get_mean(),
                yes_bid_close=c.yes_bid.get_close(),
                yes_ask_close=c.yes_ask.get_close(),
                volume_fp=c.volume_fp,
                open_interest_fp=c.open_interest_fp,
            )
        )
    return rows


# ── per-series ingest ──────────────────────────────────────────────────


async def _fetch_candles_for_market(
    client: KalshiClient,
    series_ticker: str,
    market: Market,
) -> list[Candlestick]:
    """Fetch daily candlesticks for one market, bounded by its lifetime.

    Returns an empty list (rather than raising) when bounds are missing
    or the API has no history for the market.
    """
    start_ts = _parse_iso_ts(market.open_time)
    end_ts = _parse_iso_ts(market.close_time)
    if start_ts is None:
        return []
    now_ts = int(datetime.now(tz=UTC).timestamp())
    if end_ts is None or end_ts > now_ts:
        end_ts = now_ts
    if end_ts <= start_ts:
        return []
    try:
        return await client.get_live_candlesticks(
            series_ticker, market.ticker, start_ts, end_ts, DAILY_INTERVAL
        )
    except Exception:
        logger.warning("Candlestick fetch failed for %s", market.ticker, exc_info=True)
        return []


async def ingest_one_series(
    client: KalshiClient,
    series_ticker: str,
    template_id: str,
    *,
    db_path: Path | None = None,
    event_status: str | None = None,
    fetch_prices: bool = True,
    event_limit: int | None = None,
    market_limit: int | None = None,
) -> SeriesIngestResult:
    """Ingest every event/market under one allowlisted series."""
    logger.info("Ingesting Kalshi series %s → %s", series_ticker, template_id)
    result = SeriesIngestResult(series_ticker=series_ticker, template_id=template_id)

    try:
        events_seen = 0
        async for event_batch in client.iter_events(series_ticker, status=event_status):
            for event in event_batch:
                if event_limit is not None and events_seen >= event_limit:
                    break
                events_seen += 1
                event_ticker = event.get("event_ticker", "")
                if not event_ticker:
                    continue
                raw_markets = await client.get_event_markets(event_ticker)
                markets = [Market.model_validate(m) for m in raw_markets]
                if market_limit is not None:
                    markets = markets[:market_limit]

                rows = [market_to_row(m, series_ticker) for m in markets]
                with connect(db_path) as conn:
                    for row in rows:
                        upsert_kalshi_market(conn, row)
                    conn.commit()
                result.markets_ingested += len(rows)

                if fetch_prices:
                    for market in markets:
                        candles = await _fetch_candles_for_market(
                            client, series_ticker, market
                        )
                        price_rows = candles_to_rows(market.ticker, candles)
                        if price_rows:
                            with connect(db_path) as conn:
                                result.candles_ingested += bulk_insert_price_history(
                                    conn, price_rows
                                )
                                conn.commit()
            if event_limit is not None and events_seen >= event_limit:
                break
    except Exception as e:
        logger.exception("Failed ingesting series %s", series_ticker)
        result.success = False
        result.error = str(e)

    logger.info(
        "  → %s: %d markets, %d candles",
        series_ticker, result.markets_ingested, result.candles_ingested,
    )
    return result


# ── top-level orchestrator ─────────────────────────────────────────────


async def run_kalshi_ingest(
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    series_tickers: list[str] | None = None,
    event_status: str | None = None,
    fetch_prices: bool = True,
    series_limit: int | None = None,
    event_limit: int | None = None,
    market_limit: int | None = None,
    db_path: Path | None = None,
    client: KalshiClient | None = None,
) -> KalshiIngestReport:
    """Run the full Kalshi ingest.

    Parameters
    ----------
    categories:
        Kalshi categories to scan for candidate series.
    series_tickers:
        Restrict to these series tickers (must still be on the
        classifier allowlist). ``None`` means "discover from categories".
    event_status:
        Server-side event filter ("settled", "active", ...). ``None``
        fetches every status.
    fetch_prices:
        When False, skip candlestick downloads (markets metadata only) —
        much faster for a structural smoke run.
    series_limit / event_limit / market_limit:
        Dev caps to bound a smoke run.
    db_path / client:
        Test/injection points.
    """
    started_at = datetime.now(tz=UTC)
    report = KalshiIngestReport(started_at=started_at)

    owns_client = client is None
    client_ctx = KalshiClient() if client is None else client

    audit_id: int | None = None
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="kalshi",
                target=",".join(series_tickers or categories),
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    try:
        if owns_client:
            await client_ctx.__aenter__()

        targets = await _resolve_targets(
            client_ctx, categories, series_tickers, series_limit
        )
        for series_ticker, template_id in targets:
            result = await ingest_one_series(
                client_ctx,
                series_ticker,
                template_id,
                db_path=db_path,
                event_status=event_status,
                fetch_prices=fetch_prices,
                event_limit=event_limit,
                market_limit=market_limit,
            )
            report.results.append(result)
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    report.finished_at = datetime.now(tz=UTC)

    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at  = ?,
                   status       = ?,
                   rows_added   = ?,
                   rows_updated = ?,
                   error_message = ?
             WHERE run_id = ?
            """,
            (
                report.finished_at.isoformat(),
                "ok" if report.n_failed == 0 else "partial",
                report.total_markets,
                report.total_candles,
                "; ".join(
                    f"{r.series_ticker}: {r.error}"
                    for r in report.results
                    if not r.success
                )[:2000],
                audit_id,
            ),
        )
        conn.commit()

    logger.info(
        "Kalshi ingest complete: %d series ok, %d failed, %d markets, %d candles",
        report.n_succeeded, report.n_failed, report.total_markets, report.total_candles,
    )
    return report


async def _resolve_targets(
    client: KalshiClient,
    categories: tuple[str, ...],
    series_tickers: list[str] | None,
    series_limit: int | None,
) -> list[tuple[str, str]]:
    """Resolve the list of (series_ticker, template_id) pairs to ingest.

    When ``series_tickers`` is given we classify them directly; otherwise
    we discover series from each category and keep only allowlisted ones.
    """
    targets: list[tuple[str, str]] = []
    seen: set[str] = set()

    if series_tickers is not None:
        for ticker in series_tickers:
            template_id = classify_series(ticker)
            if template_id is None:
                logger.warning("Skipping %s: not on the classifier allowlist", ticker)
                continue
            if ticker not in seen:
                seen.add(ticker)
                targets.append((ticker, template_id))
    else:
        for category in categories:
            series_list: list[dict[str, Any]] = await client.get_series_by_category(
                category
            )
            for s in series_list:
                ticker = s.get("ticker", "")
                template_id = classify_series(ticker)
                if template_id is None or ticker in seen:
                    continue
                seen.add(ticker)
                targets.append((ticker, template_id))

    if series_limit is not None:
        targets = targets[:series_limit]
    logger.info("Resolved %d Kalshi series to ingest", len(targets))
    return targets


__all__ = [
    "KalshiIngestReport",
    "SeriesIngestResult",
    "candles_to_rows",
    "ingest_one_series",
    "market_to_row",
    "run_kalshi_ingest",
]
