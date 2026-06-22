"""The canonical list of numeric series we ingest from FRED in Phase 1.2.

This file is the single source of truth that the orchestrator consumes.
It mirrors ``docs/data_spec.md`` — if you change one, change the other.

The categories below match the README:

    inflation, labor, growth, surveys, rates, markets, money_banking,
    housing, international

``revises`` semantics:
    True  → ingest via the ALFRED endpoint, store full vintage history
    False → ingest current values only (daily market data, etc.)

Series flagged ``optional`` are skipped by default on partial runs but
included on full runs. We use this for slow-changing nice-to-haves so
that "smoke test the FRED pipeline" stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class FredSeriesEntry:
    """One row in the registry.

    ``source`` is always "FRED" here; the field exists so that when we
    add BLS/BEA/etc. ingestors they can share the same registry shape.
    """

    series_id: str
    title: str
    category: str
    revises: bool
    source: str = "FRED"
    optional: bool = False
    notes: str = ""


FRED_REGISTRY: Final[tuple[FredSeriesEntry, ...]] = (
    # ── Inflation ──────────────────────────────────────────────────────
    FredSeriesEntry("CPIAUCSL",                "CPI: All Urban Consumers, SA",                "inflation", revises=True),
    FredSeriesEntry("CPILFESL",                "Core CPI (excl food & energy), SA",           "inflation", revises=True),
    FredSeriesEntry("PCEPI",                   "PCE Price Index",                              "inflation", revises=True),
    FredSeriesEntry("PCEPILFE",                "Core PCE",                                     "inflation", revises=True),
    FredSeriesEntry("PPIACO",                  "Producer Price Index, All Commodities",       "inflation", revises=True),
    FredSeriesEntry("PCETRIM12M656SFRBDAL",    "Trimmed Mean PCE (Dallas Fed)",               "inflation", revises=True, optional=True),
    FredSeriesEntry("MEDCPIM158SFRBCLE",       "Median CPI (Cleveland Fed)",                  "inflation", revises=True, optional=True),
    FredSeriesEntry("STICKCPIM157SFRBATL",     "Sticky Price CPI (Atlanta Fed)",              "inflation", revises=True, optional=True),
    FredSeriesEntry("T5YIFR",                  "5Y5Y Forward Inflation Expectation",          "inflation", revises=False),
    FredSeriesEntry("MICH",                    "Michigan 1Y Inflation Expectation",           "inflation", revises=True, optional=True),
    # ── Labor market ───────────────────────────────────────────────────
    FredSeriesEntry("PAYEMS",                  "Non-Farm Payrolls",                            "labor",     revises=True),
    FredSeriesEntry("UNRATE",                  "Unemployment Rate (U-3)",                      "labor",     revises=True),
    FredSeriesEntry("U6RATE",                  "U-6 Broad Underemployment",                    "labor",     revises=True, optional=True),
    FredSeriesEntry("CIVPART",                 "Labor Force Participation Rate",               "labor",     revises=True),
    FredSeriesEntry("CES0500000003",           "Average Hourly Earnings YoY",                  "labor",     revises=True, optional=True),
    FredSeriesEntry("JTSJOL",                  "Job Openings (JOLTS)",                         "labor",     revises=True),
    FredSeriesEntry("JTSQUR",                  "Quits Rate (JOLTS)",                           "labor",     revises=True, optional=True),
    FredSeriesEntry("ICSA",                    "Initial Jobless Claims, SA",                   "labor",     revises=True),
    FredSeriesEntry("CCSA",                    "Continuing Jobless Claims, SA",                "labor",     revises=True, optional=True),
    # ── Growth & activity ──────────────────────────────────────────────
    FredSeriesEntry("GDPC1",                   "Real Gross Domestic Product",                  "growth",    revises=True),
    FredSeriesEntry("INDPRO",                  "Industrial Production Index",                  "growth",    revises=True),
    FredSeriesEntry("TCU",                     "Capacity Utilization",                         "growth",    revises=True, optional=True),
    FredSeriesEntry("RSXFS",                   "Retail Sales (advance, excl food services)",  "growth",    revises=True),
    FredSeriesEntry("PCE",                     "Personal Consumption Expenditures",            "growth",    revises=True, optional=True),
    FredSeriesEntry("PI",                      "Personal Income",                              "growth",    revises=True, optional=True),
    FredSeriesEntry("DGORDER",                 "Durable Goods Orders",                         "growth",    revises=True, optional=True),
    FredSeriesEntry("CFNAI",                   "Chicago Fed National Activity Index",          "growth",    revises=True, optional=True),
    FredSeriesEntry("USSLIND",                 "Conference Board LEI (state diffusion)",      "growth",    revises=True, optional=True),
    # ── Surveys ────────────────────────────────────────────────────────
    FredSeriesEntry("GACDISA066MSFRBNY",       "Empire State Manufacturing Survey",            "surveys",   revises=True, optional=True),
    FredSeriesEntry("GACDFSA066MSFRBPHI",      "Philly Fed Manufacturing Survey",              "surveys",   revises=True, optional=True),
    # ── Interest rates ─────────────────────────────────────────────────
    FredSeriesEntry("DFF",                     "Effective Federal Funds Rate",                 "rates",     revises=False),
    FredSeriesEntry("DFEDTARU",                "Fed Funds Target Upper",                       "rates",     revises=False),
    FredSeriesEntry("DFEDTARL",                "Fed Funds Target Lower",                       "rates",     revises=False),
    FredSeriesEntry("SOFR",                    "Secured Overnight Financing Rate",             "rates",     revises=False),
    FredSeriesEntry("DGS3MO",                  "3-Month Treasury Bill Yield",                  "rates",     revises=False),
    FredSeriesEntry("DGS2",                    "2-Year Treasury Yield",                        "rates",     revises=False),
    FredSeriesEntry("DGS5",                    "5-Year Treasury Yield",                        "rates",     revises=False),
    FredSeriesEntry("DGS10",                   "10-Year Treasury Yield",                       "rates",     revises=False),
    FredSeriesEntry("DGS30",                   "30-Year Treasury Yield",                       "rates",     revises=False, optional=True),
    FredSeriesEntry("T10Y2Y",                  "10Y minus 2Y Spread",                          "rates",     revises=False),
    FredSeriesEntry("T10Y3M",                  "10Y minus 3M Spread",                          "rates",     revises=False),
    FredSeriesEntry("DFII5",                   "5-Year TIPS Yield",                            "rates",     revises=False, optional=True),
    FredSeriesEntry("DFII10",                  "10-Year TIPS Yield",                           "rates",     revises=False, optional=True),
    FredSeriesEntry("T5YIE",                   "5-Year Breakeven Inflation",                   "rates",     revises=False),
    FredSeriesEntry("T10YIE",                  "10-Year Breakeven Inflation",                  "rates",     revises=False),
    # ── Financial markets ──────────────────────────────────────────────
    FredSeriesEntry("SP500",                   "S&P 500 (FRED limited window)",                "markets",   revises=False, notes="FRED only holds the last 10 years; longer history via Yahoo later"),
    FredSeriesEntry("VIXCLS",                  "VIX",                                          "markets",   revises=False),
    FredSeriesEntry("NASDAQCOM",               "NASDAQ Composite",                             "markets",   revises=False, optional=True),
    FredSeriesEntry("DTWEXBGS",                "Trade-Weighted Dollar Index",                  "markets",   revises=False),
    FredSeriesEntry("DCOILWTICO",              "WTI Crude",                                    "markets",   revises=False),
    FredSeriesEntry("GOLDAMGBD228NLBM",        "Gold (LBMA AM Fix)",                           "markets",   revises=False, optional=True),
    FredSeriesEntry("BAA10YM",                 "BAA Corporate - 10Y Spread",                   "markets",   revises=False),
    FredSeriesEntry("BAMLH0A0HYM2",            "ICE BofA High Yield Option-Adjusted Spread",   "markets",   revises=False),
    FredSeriesEntry("NFCI",                    "Chicago Fed National Financial Conditions",    "markets",   revises=True, optional=True),
    # ── Money & banking ────────────────────────────────────────────────
    FredSeriesEntry("M2SL",                    "M2 Money Supply",                              "money",     revises=True, optional=True),
    FredSeriesEntry("WRESBAL",                 "Reserves of Depository Institutions",          "money",     revises=True, optional=True),
    FredSeriesEntry("WALCL",                   "Total Assets of Federal Reserve",              "money",     revises=False),
    FredSeriesEntry("DRTSCILM",                "SLOOS: C&I Lending Standards (Large)",         "money",     revises=True, optional=True),
    FredSeriesEntry("MORTGAGE30US",            "30-Year Mortgage Rate",                        "money",     revises=False),
    # ── Housing ────────────────────────────────────────────────────────
    FredSeriesEntry("HOUST",                   "Housing Starts",                               "housing",   revises=True, optional=True),
    FredSeriesEntry("PERMIT",                  "Building Permits",                             "housing",   revises=True, optional=True),
    FredSeriesEntry("EXHOSLUSM495S",           "Existing Home Sales",                          "housing",   revises=True, optional=True),
    FredSeriesEntry("HSN1F",                   "New Home Sales",                               "housing",   revises=True, optional=True),
    FredSeriesEntry("CSUSHPINSA",              "Case-Shiller U.S. National HPI",               "housing",   revises=True, optional=True),
)


def required_series() -> tuple[FredSeriesEntry, ...]:
    """The minimal smoke-test set: non-optional entries only."""
    return tuple(e for e in FRED_REGISTRY if not e.optional)


def all_series() -> tuple[FredSeriesEntry, ...]:
    """Everything in the registry, optional and required."""
    return FRED_REGISTRY


def find(series_id: str) -> FredSeriesEntry | None:
    for e in FRED_REGISTRY:
        if e.series_id == series_id:
            return e
    return None


def by_category(category: str) -> tuple[FredSeriesEntry, ...]:
    return tuple(e for e in FRED_REGISTRY if e.category == category)
