"""Registry of scheduled economic releases that populate ``event_calendar``.

Phase 1.6 builds the event calendar *from data we already hold* rather than
from a paid calendar API. Every macro indicator we ingest from FRED/ALFRED
carries a ``release_date`` per observation, so the set of historical release
*events* is already implicit in ``series_observations``. This registry names
the subset of series that correspond to genuine, scheduled macro releases —
the monthly/quarterly/weekly prints a forecaster would mark on a calendar —
as opposed to daily market quotes (Treasury yields, the S&P 500, …) which are
not "releases" in the event sense.

For each entry we record:

    series_id            the FRED series whose first-vintage prints are events
    event_name           human-readable name ("CPI release")
    template_id          link to one of our 7 question templates, when relevant
    consensus_series_id  an SPF median series to use as a point-in-time
                         consensus, when one exists with COMPARABLE units
    consensus_comparable when True, ``surprise = actual - consensus`` is
                         meaningful (same units / cadence). When False we still
                         record the consensus for reference but leave surprise
                         NULL so we never publish an apples-to-oranges number.

Consensus coverage is deliberately conservative. The Survey of Professional
Forecasters (our only point-in-time consensus source so far) is quarterly and
quoted in particular units, so it only lines up cleanly with a few series
(e.g. the unemployment rate, both in percent). Monthly consensus for CPI / NFP
needs a dedicated survey feed (Trading Economics / DBnomics) which is slated as
a later enrichment; until then those events carry ``actual`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CalendarSeries:
    """One scheduled-release series that generates ``event_calendar`` rows."""

    series_id: str
    event_name: str
    template_id: str | None = None
    consensus_series_id: str | None = None
    consensus_comparable: bool = True
    notes: str = ""


# The headline scheduled releases. Daily market series (yields, equities, FX,
# oil) are intentionally excluded — they print continuously and are not
# "release events". Frequencies are documented for readers; the orchestrator
# does not depend on them (it reads actual cadence from the observations).
CALENDAR_REGISTRY: Final[tuple[CalendarSeries, ...]] = (
    # ── Inflation (monthly) ─────────────────────────────────────────────
    CalendarSeries("CPIAUCSL", "CPI release", template_id="cpi_yoy"),
    CalendarSeries("CPILFESL", "Core CPI release"),
    CalendarSeries("PCEPI", "PCE price index release"),
    CalendarSeries("PCEPILFE", "Core PCE release"),
    CalendarSeries("PPIACO", "PPI release"),
    # ── Labor (monthly, claims weekly) ──────────────────────────────────
    CalendarSeries("PAYEMS", "Nonfarm Payrolls release", template_id="nfp"),
    CalendarSeries(
        "UNRATE",
        "Unemployment Rate release",
        template_id="unemployment",
        consensus_series_id="SPF_UNEMP_MEDIAN_NOWCAST",
        consensus_comparable=True,
        notes=(
            "Consensus is the SPF median nowcast of the quarterly-average U-3 "
            "(percent). The monthly print is compared to that quarterly "
            "expectation — an approximate but same-unit surprise."
        ),
    ),
    CalendarSeries("CIVPART", "Labor Force Participation release"),
    CalendarSeries("JTSJOL", "JOLTS Job Openings release"),
    CalendarSeries("ICSA", "Initial Jobless Claims release"),
    CalendarSeries("CCSA", "Continuing Jobless Claims release"),
    # ── Growth / activity ───────────────────────────────────────────────
    CalendarSeries("GDPC1", "Real GDP release", template_id="gdp"),
    CalendarSeries("INDPRO", "Industrial Production release"),
    CalendarSeries("RSXFS", "Retail Sales release"),
    CalendarSeries("PCE", "Personal Consumption Expenditures release"),
    CalendarSeries("PI", "Personal Income release"),
    CalendarSeries("DGORDER", "Durable Goods Orders release"),
    # ── Housing ─────────────────────────────────────────────────────────
    CalendarSeries("HOUST", "Housing Starts release"),
    CalendarSeries("PERMIT", "Building Permits release"),
    CalendarSeries("HSN1F", "New Home Sales release"),
    CalendarSeries("EXHOSLUSM495S", "Existing Home Sales release"),
    CalendarSeries("CSUSHPINSA", "Case-Shiller HPI release"),
    # ── Money & banking ─────────────────────────────────────────────────
    CalendarSeries("M2SL", "M2 Money Supply release"),
)


def registry_series_ids() -> tuple[str, ...]:
    """Every FRED series that should generate calendar release events."""
    return tuple(e.series_id for e in CALENDAR_REGISTRY)


def find(series_id: str) -> CalendarSeries | None:
    for e in CALENDAR_REGISTRY:
        if e.series_id == series_id:
            return e
    return None
