"""Classify Kalshi markets against our 7 question_templates.

The Phase 1.5 pipeline filters tens of thousands of Kalshi macro
markets down to the subset that maps cleanly onto our 7 prediction
targets. This module is the central source of truth for that mapping.

The classification has two parts:

1. Series-ticker allowlist
       Each Kalshi market belongs to a SERIES (e.g. KXFED). We curate
       a small allowlist of series tickers that correspond 1:1 to our
       question templates. A market whose series is not on the
       allowlist is silently dropped.

2. Strike parser
       For markets in an allowlisted series, we extract the strike
       value and direction from the yes_sub_title (e.g. "Will CPI YoY
       exceed 3.0%?" → strike=3.0, direction="above"). Unparseable
       markets are still ingested with NULL strike, since their price
       history is still useful.

Curation principles for the allowlist:

  - Only include series whose markets are unambiguous instances of
    one of our 7 templates.
  - When Kalshi runs both an old-style (no KX prefix) and a new-style
    (KX prefix) series for the same concept, include both.
  - Reject series that mix concepts (e.g. KXFOMCDISSENTCOUNT is about
    dissents, not the Fed rate decision itself — skip).
  - Foreign-central-bank rate decisions are out of scope for Phase
    1.5; we focus on US targets.

See docs/phase_1_5_kalshi_research.md for the discovery notes that
informed this list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """The classifier's verdict for a single Kalshi market.

    ``template_id`` is ``None`` when the market doesn't match any of
    our 7 templates. ``strike_value`` is ``None`` when we couldn't
    parse the strike (this happens for binary markets like "Will the
    Fed cut?" which have no numeric threshold).
    """

    template_id: str | None
    strike_value: float | None
    strike_direction: str  # "above" / "below" / "between" / "equals" / ""


# ── Series-ticker allowlist ───────────────────────────────────────────


SERIES_TO_TEMPLATE: Final[dict[str, str]] = {
    # Fed decisions
    "KXFED": "fed_decision",
    "KXFEDDECISION": "fed_decision",
    "FEDDECISION": "fed_decision",
    "KXRATECUTE": "fed_decision",
    "KXTERMINALRATE": "fed_decision",
    "TERMINALRATE": "fed_decision",
    "KXFEDRATEMIN": "fed_decision",
    "LOWESTRATE": "fed_decision",
    "KXFEDCHGCOUNT": "fed_decision",
    "KXRATEHIKE": "fed_decision",
    "RATEHIKE": "fed_decision",
    "KXLARGECUT": "fed_decision",
    "KXEMERCUTS": "fed_decision",
    # CPI inflation
    "KXACPI": "cpi_yoy",
    "ACPICORE": "cpi_yoy",
    "ACPICORE-": "cpi_yoy",
    "KXACPICORE": "cpi_yoy",
    "KXCPICORE": "cpi_yoy",
    "KXECONSTATCPIYOY": "cpi_yoy",
    "KXECONSTATCORECPIYOY": "cpi_yoy",
    "KXECONSTATCPI": "cpi_yoy",
    "KXECONSTATCPICORE": "cpi_yoy",
    "KXCPICOREA": "cpi_yoy",
    "LCPIYOY": "cpi_yoy",
    "CPICOREYOY": "cpi_yoy",
    "KXCOREUND": "cpi_yoy",
    # NFP / employment change
    "KXPAYROLLS": "nfp",
    "KXPROLLS": "nfp",
    "KXJOBLESSCLAIMS": "nfp",
    "KXADP": "nfp",
    # Unemployment rate
    "KXECONSTATU3": "unemployment",
    "KXUE": "unemployment",
    "U3MIN": "unemployment",
    "U3MAX": "unemployment",
    "KXU3MAX": "unemployment",
    # GDP
    "GDP": "gdp",
    "KXGDPNOM": "gdp",
    "KXNGDPQ": "gdp",
    "NGDPQ": "gdp",
    "NGDP": "gdp",
    "KXGDPUSMIN": "gdp",
    "KXGDPUSMAX": "gdp",
    # 10-year Treasury yield
    "KXTNOTE": "yield_10y",
    "TNOTE": "yield_10y",
    "KXTNOTEW": "yield_10y",
    "TNOTEW": "yield_10y",
    "TNOTED": "yield_10y",
    "KXTNOTED": "yield_10y",
    "KXNOTE10": "yield_10y",
    "KXNOTE10M": "yield_10y",
    "KXNOTE10W": "yield_10y",
    "KXNOTE10Y": "yield_10y",
    "KX10YUSTSRY": "yield_10y",
    # Recession (12-month horizon-ish; we treat NBER and Sahm flags equivalently)
    "KXRECSSNBER": "recession_12m",
    "RECSSNBER": "recession_12m",
    "KXNBERRECESSQ": "recession_12m",
    "KXSAHM": "recession_12m",
}


def classify_series(series_ticker: str) -> str | None:
    """Map a series_ticker to a template_id, or None if not in allowlist.

    Case-insensitive on the input.
    """
    return SERIES_TO_TEMPLATE.get(series_ticker.upper())


# ── Strike parsing ────────────────────────────────────────────────────


# Pre-compiled regex matchers, ordered: try the more specific patterns first.
# ``yes_sub_title`` is preferred input because Kalshi authors usually
# write the cleanest threshold expression there.

# Above/below numeric thresholds.
# Prefix form:  "above 3.0%", "below 4.5%", "less than 200K"
# Postfix form: "3.0% or higher", "4.5% or above", "200K+"

_ABOVE_PREFIX_RE = re.compile(
    r"\b(?:above|over|exceeds?|greater than|higher than|>|>=|at least)\s*"
    r"\$?(?P<value>-?\d+(?:\.\d+)?)\s*(?:%|bps|K)?",
    flags=re.IGNORECASE,
)
_ABOVE_POSTFIX_RE = re.compile(
    r"\$?(?P<value>-?\d+(?:\.\d+)?)\s*(?:%|bps|K)?\s*"
    r"(?:or\s+(?:higher|above|greater|more|over)|\+|or more)",
    flags=re.IGNORECASE,
)
_BELOW_PREFIX_RE = re.compile(
    r"\b(?:below|under|less than|fewer than|<|<=|at most)\s*"
    r"\$?(?P<value>-?\d+(?:\.\d+)?)\s*(?:%|bps|K)?",
    flags=re.IGNORECASE,
)
_BELOW_POSTFIX_RE = re.compile(
    r"\$?(?P<value>-?\d+(?:\.\d+)?)\s*(?:%|bps|K)?\s*"
    r"(?:or\s+(?:lower|below|less|fewer|under)|-|or less)",
    flags=re.IGNORECASE,
)
# Range pattern: "between X and Y", "X to Y", "X-Y" (when both look like the
# same unit). Kalshi range markets commonly say things like "6.0% to 6.9%".
_BETWEEN_RE = re.compile(
    r"\b(?:between\s*)?\$?(?P<low>-?\d+(?:\.\d+)?)\s*(?:%|bps|K)?\s*"
    r"(?:to|and|-|\u2013|\u2014)\s*"  # supports hyphen, en dash, em dash
    r"\$?(?P<high>-?\d+(?:\.\d+)?)\s*(?:%|bps|K)?",
    flags=re.IGNORECASE,
)
# Bare percentages used in titles like "Fed funds rate will be 4.50% range"
_BARE_PCT_RE = re.compile(r"\b(?P<value>-?\d+(?:\.\d+)?)\s*%")


def parse_strike(yes_sub_title: str, title: str = "") -> tuple[float | None, str]:
    """Extract ``(strike_value, strike_direction)`` from market text.

    We try ``yes_sub_title`` first, then fall back to ``title``.
    Returns ``(None, "")`` if no numeric threshold can be parsed —
    the market may still be ingested (binary markets like "Will the
    Fed cut?" don't have a strike, and that's fine).
    """
    for text in (yes_sub_title, title):
        if not text:
            continue
        match = _BETWEEN_RE.search(text)
        if match:
            try:
                low = float(match["low"])
                high = float(match["high"])
            except (TypeError, ValueError):
                continue
            return ((low + high) / 2.0, "between")

        # Try prefix forms FIRST (above X / below X) before postfix
        # forms (X or higher / X+) because they're less ambiguous.
        for pattern, direction in (
            (_ABOVE_PREFIX_RE, "above"),
            (_BELOW_PREFIX_RE, "below"),
            (_ABOVE_POSTFIX_RE, "above"),
            (_BELOW_POSTFIX_RE, "below"),
        ):
            match = pattern.search(text)
            if match:
                try:
                    return (float(match["value"]), direction)
                except (TypeError, ValueError):
                    continue

        match = _BARE_PCT_RE.search(text)
        if match:
            # Without an above/below modifier we treat a bare pct as
            # "equals" — Kalshi sometimes encodes specific outcome
            # buckets that way.
            try:
                return (float(match["value"]), "equals")
            except (TypeError, ValueError):
                continue

    return (None, "")


# ── Top-level convenience ─────────────────────────────────────────────


def classify_market(
    *,
    series_ticker: str,
    yes_sub_title: str = "",
    title: str = "",
) -> ClassificationResult:
    """Return a :class:`ClassificationResult` for a single Kalshi market.

    The orchestrator calls this once per market when deciding whether
    to persist it and how to populate ``template_id`` / ``strike_value``
    / ``strike_direction`` on ``kalshi_markets``.
    """
    template_id = classify_series(series_ticker)
    if template_id is None:
        return ClassificationResult(
            template_id=None, strike_value=None, strike_direction=""
        )
    strike_value, strike_direction = parse_strike(yes_sub_title, title)
    return ClassificationResult(
        template_id=template_id,
        strike_value=strike_value,
        strike_direction=strike_direction,
    )


__all__ = [
    "SERIES_TO_TEMPLATE",
    "ClassificationResult",
    "classify_market",
    "classify_series",
    "parse_strike",
]
