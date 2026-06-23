"""Pydantic models for the subset of Kalshi API responses we care about.

Adapted from the author's earlier ``black-swan-event-intelligence``
project. Trimmed to the candlestick + market fields that matter for
historical training data; black-swan-specific models removed.

Kalshi prices come in two forms:
  - integer "cents" (0-100), e.g. 23
  - dollar string with 4 decimals, e.g. "0.2300"

The ``_to_dollar_str`` helper normalizes both into the canonical
4-decimal string we store in SQLite.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def _to_dollar_str(v: str | int | float | None) -> str | None:
    """Normalize cent ints or dollar strings into 4-decimal dollar strings."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return f"{v / 100:.4f}"
    return str(v)


class CandlestickPrice(BaseModel):
    """Price section of a Kalshi candlestick. Lots of optional fields
    because the API has changed shape over time; we tolerate either form.
    """

    model_config = ConfigDict(extra="ignore")

    open: str | int | float | None = None
    high: str | int | float | None = None
    low: str | int | float | None = None
    close: str | int | float | None = None
    mean: str | int | float | None = None
    previous: str | int | float | None = None

    open_dollars: str | None = None
    high_dollars: str | None = None
    low_dollars: str | None = None
    close_dollars: str | None = None
    mean_dollars: str | None = None
    previous_dollars: str | None = None

    def get_close(self) -> str | None:
        return self.close_dollars or _to_dollar_str(self.close)

    def get_mean(self) -> str | None:
        return self.mean_dollars or _to_dollar_str(self.mean)

    def get_open(self) -> str | None:
        return self.open_dollars or _to_dollar_str(self.open)

    def get_high(self) -> str | None:
        return self.high_dollars or _to_dollar_str(self.high)

    def get_low(self) -> str | None:
        return self.low_dollars or _to_dollar_str(self.low)


class CandlestickBidAsk(BaseModel):
    """Bid or ask section of a candlestick."""

    model_config = ConfigDict(extra="ignore")

    open: str | int | float | None = None
    high: str | int | float | None = None
    low: str | int | float | None = None
    close: str | int | float | None = None

    open_dollars: str | None = None
    high_dollars: str | None = None
    low_dollars: str | None = None
    close_dollars: str | None = None

    def get_close(self) -> str | None:
        return self.close_dollars or _to_dollar_str(self.close)


class Candlestick(BaseModel):
    """One period's candlestick for a Kalshi market."""

    model_config = ConfigDict(extra="ignore")

    end_period_ts: int
    price: CandlestickPrice
    yes_bid: CandlestickBidAsk
    yes_ask: CandlestickBidAsk
    volume: str | int | float | None = None
    volume_fp: str | None = None
    open_interest: str | int | float | None = None
    open_interest_fp: str | None = None


class Market(BaseModel):
    """A single Kalshi market (one binary contract within an event).

    Only the fields we persist to ``kalshi_markets`` are modeled; the
    live API returns many more (order-book sizes, fee structure, etc.)
    which we ignore. ``series_ticker`` is *not* part of the market
    payload — it lives on the parent event, so the orchestrator passes
    it in separately.

    Price/volume fields already arrive as dollar/float strings on the
    modern API (``last_price_dollars``, ``volume_fp``); we store them
    verbatim, consistent with the candlestick handling above.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    event_ticker: str = ""
    market_type: str = ""
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""

    open_time: str | None = None
    close_time: str | None = None
    created_time: str | None = None
    settlement_ts: int | None = None

    status: str = ""
    result: str = ""
    settlement_value_dollars: str | None = None

    last_price_dollars: str | None = None
    volume_fp: str | None = None
    open_interest_fp: str | None = None
