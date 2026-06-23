"""Tests for the Kalshi market classifier.

Two layers:

1. Series-to-template mapping is exercised on the curated allowlist.
2. Strike parsing is exercised on hand-built market titles drawn from
   real-world Kalshi phrasings (recorded during Phase 1.5 recon).
"""

from __future__ import annotations

import pytest

from kalshi_train.data.kalshi_classifier import (
    SERIES_TO_TEMPLATE,
    classify_market,
    classify_series,
    parse_strike,
)

# ── classify_series ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("KXFED", "fed_decision"),
        ("kxfed", "fed_decision"),  # case-insensitive
        ("FEDDECISION", "fed_decision"),
        ("KXACPI", "cpi_yoy"),
        ("KXCPICORE", "cpi_yoy"),
        ("KXPAYROLLS", "nfp"),
        ("KXECONSTATU3", "unemployment"),
        ("GDP", "gdp"),
        ("KXTNOTE", "yield_10y"),
        ("KXRECSSNBER", "recession_12m"),
    ],
)
def test_classify_series_known_tickers(ticker: str, expected: str) -> None:
    assert classify_series(ticker) == expected


@pytest.mark.parametrize(
    "ticker",
    [
        "KXMVECROSSCATEGORY",      # combo market — not macro
        "KXFOMCDISSENTCOUNT",       # FOMC dissents — not the rate decision
        "KXBOE",                    # foreign CB, out of scope
        "KXCBDECISIONKOREA",        # foreign CB
        "KXCPIDELAY",               # release-timing meta-question, not the value
        "KXJUSTATEST",              # not a real macro series
        "RANDOM_GARBAGE",
        "",
    ],
)
def test_classify_series_rejects_non_macro(ticker: str) -> None:
    assert classify_series(ticker) is None


def test_classify_series_allowlist_has_no_template_typos() -> None:
    """Every value in the allowlist must be a known question_template_id."""
    known_templates = {
        "fed_decision", "cpi_yoy", "nfp", "unemployment",
        "gdp", "yield_10y", "recession_12m",
    }
    seen = set(SERIES_TO_TEMPLATE.values())
    extras = seen - known_templates
    assert not extras, f"unknown template_ids in allowlist: {extras}"


def test_classify_series_allowlist_covers_all_seven_templates() -> None:
    """Sanity check: every template should have at least one mapped series."""
    expected = {
        "fed_decision", "cpi_yoy", "nfp", "unemployment",
        "gdp", "yield_10y", "recession_12m",
    }
    seen = set(SERIES_TO_TEMPLATE.values())
    assert expected.issubset(seen)


# ── parse_strike ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Above patterns
        ("Will CPI YoY be above 3.0%?", (3.0, "above")),
        ("yes 4.35 or higher", (4.35, "above")),
        ("Higher than 200K", (200.0, "above")),
        ("CPI exceeds 4.0%", (4.0, "above")),
        # Below patterns
        ("Will the 10Y close below 4.0%?", (4.0, "below")),
        ("less than 150K", (150.0, "below")),
        ("UNRATE under 4.5%", (4.5, "below")),
        # Between
        ("between 3.5% and 4.0%", (3.75, "between")),
        # Implicit between ("X to Y" form, no "between" keyword)
        ("6.0% to 6.9%", (6.45, "between")),
        ("150K to 200K", (175.0, "between")),
        ("4.5-4.75%", (4.625, "between")),
        # Bare percentage (interpreted as "equals" / specific bucket)
        ("4.50% target rate", (4.5, "equals")),
        # Negative numbers (e.g. negative GDP)
        ("less than -0.5%", (-0.5, "below")),
    ],
)
def test_parse_strike_extracts_value_and_direction(
    text: str, expected: tuple[float, str]
) -> None:
    assert parse_strike(text) == expected


def test_parse_strike_returns_none_when_no_number() -> None:
    assert parse_strike("Will the Fed cut rates at the next meeting?") == (None, "")
    assert parse_strike("", "") == (None, "")


def test_parse_strike_prefers_yes_sub_title_over_title() -> None:
    """If both contain numbers, the yes_sub_title's wins (it's usually more specific)."""
    yes_sub = "above 3.0%"
    title = "Will CPI be greater than 5.0?"
    val, direction = parse_strike(yes_sub_title=yes_sub, title=title)
    assert val == 3.0
    assert direction == "above"


def test_parse_strike_falls_back_to_title_when_yes_sub_empty() -> None:
    val, direction = parse_strike(yes_sub_title="", title="below 200K")
    assert val == 200.0
    assert direction == "below"


# ── classify_market end-to-end ────────────────────────────────────────


def test_classify_market_fed_binary_has_template_but_no_strike() -> None:
    """Binary Fed-decision markets have no numeric threshold."""
    result = classify_market(
        series_ticker="KXFED",
        yes_sub_title="Yes, cut rates",
        title="Will the Fed cut rates at the December 2024 meeting?",
    )
    assert result.template_id == "fed_decision"
    assert result.strike_value is None
    assert result.strike_direction == ""


def test_classify_market_cpi_with_strike_above() -> None:
    result = classify_market(
        series_ticker="KXACPI",
        yes_sub_title="3.0% or higher",
        title="Will October CPI YoY be 3.0% or higher?",
    )
    assert result.template_id == "cpi_yoy"
    assert result.strike_value == 3.0
    assert result.strike_direction == "above"


def test_classify_market_10y_close_above() -> None:
    result = classify_market(
        series_ticker="KXNOTE10M",
        yes_sub_title="Above 4.35%",
        title="Where will the 10Y US Treasury close this month?",
    )
    assert result.template_id == "yield_10y"
    assert result.strike_value == 4.35
    assert result.strike_direction == "above"


def test_classify_market_drops_non_macro_series() -> None:
    """A market on a non-macro series gets template_id=None and no strike."""
    result = classify_market(
        series_ticker="KXMVECROSSCATEGORY",
        yes_sub_title="42.5% chance",
        title="Random combo market",
    )
    assert result.template_id is None
    assert result.strike_value is None
    assert result.strike_direction == ""


def test_classify_market_recession_binary_is_template_no_strike() -> None:
    result = classify_market(
        series_ticker="KXRECSSNBER",
        yes_sub_title="Yes, recession",
        title="Will NBER declare a recession in 2025?",
    )
    assert result.template_id == "recession_12m"
    assert result.strike_value is None


def test_classify_market_handles_case_variation_in_ticker() -> None:
    """Real-world tickers sometimes come back in mixed case."""
    result = classify_market(
        series_ticker="kxfeddecision",
        yes_sub_title="hold rates",
        title="Will the Fed hold rates at the next meeting?",
    )
    assert result.template_id == "fed_decision"


def test_classify_market_unparseable_strike_still_returns_template() -> None:
    """Don't drop a market just because we can't parse its strike."""
    result = classify_market(
        series_ticker="KXACPI",
        yes_sub_title="some prose without numbers",
        title="Will CPI be high next year?",
    )
    assert result.template_id == "cpi_yoy"
    assert result.strike_value is None
