"""Registry of SPF derived series we ingest into series_observations.

The SPF Excel files contain many more sheets than we need. This file
documents the subset we care about for our 7 prediction targets, and
maps each (SPF variable, horizon column) pair to a derived
``series_id`` we store in our database.

Horizon naming convention (from SPF docs):

    X1  real-time historical value (previous quarter — we IGNORE this,
        since it's just the lagged actual, not a forecast)
    X2  nowcast for the survey quarter itself
    X3  forecast for Q+1 (one quarter after the survey)
    X4  forecast for Q+2
    X5  forecast for Q+3
    X6  forecast for Q+4
    XA  annual-average forecast for the survey year
    XB  annual-average forecast for the following year
    XC  annual-average forecast for two years out (inflation, U-rate, rates)
    XD  annual-average forecast for three years out (some series)

We pull the most useful horizons (nowcast through Q+2, and the two
annual horizons) for a focused subset of variables that align with our
7 question templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class SPFVariable:
    """One SPF variable we ingest.

    ``horizon_to_series_id`` maps an SPF horizon column suffix
    (e.g. "2", "3", "A", "B") to the derived series_id we write into
    ``series_observations``.

    ``forecast_horizon_quarters`` documents the *meaning* of each
    horizon — 0 for nowcast, 1 for Q+1, etc., or "annual_year_0",
    "annual_year_1", "annual_year_2", "annual_year_3" for annual rows.
    We store this so the orchestrator can build sensible series titles.
    """

    spf_sheet: str
    description: str
    category: str
    horizon_to_series_id: dict[str, str]
    horizon_to_label: dict[str, str]


SPF_VARIABLES: Final[tuple[SPFVariable, ...]] = (
    SPFVariable(
        spf_sheet="CPI",
        description="CPI inflation, annualized quarterly rate",
        category="inflation",
        horizon_to_series_id={
            "2": "SPF_CPI_MEDIAN_NOWCAST",
            "3": "SPF_CPI_MEDIAN_Q1",
            "4": "SPF_CPI_MEDIAN_Q2",
            "A": "SPF_CPI_MEDIAN_ANNUAL_Y0",
            "B": "SPF_CPI_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "4": "Q+2",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="CORECPI",
        description="Core CPI inflation (ex food and energy), annualized quarterly rate",
        category="inflation",
        horizon_to_series_id={
            "2": "SPF_CORECPI_MEDIAN_NOWCAST",
            "3": "SPF_CORECPI_MEDIAN_Q1",
            "4": "SPF_CORECPI_MEDIAN_Q2",
            "A": "SPF_CORECPI_MEDIAN_ANNUAL_Y0",
            "B": "SPF_CORECPI_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "4": "Q+2",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="PCE",
        description="Headline PCE inflation",
        category="inflation",
        horizon_to_series_id={
            "2": "SPF_PCE_MEDIAN_NOWCAST",
            "3": "SPF_PCE_MEDIAN_Q1",
            "A": "SPF_PCE_MEDIAN_ANNUAL_Y0",
            "B": "SPF_PCE_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="COREPCE",
        description="Core PCE inflation (Fed's preferred inflation gauge)",
        category="inflation",
        horizon_to_series_id={
            "2": "SPF_COREPCE_MEDIAN_NOWCAST",
            "3": "SPF_COREPCE_MEDIAN_Q1",
            "A": "SPF_COREPCE_MEDIAN_ANNUAL_Y0",
            "B": "SPF_COREPCE_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="RGDP",
        description="Real GDP level (we'll often convert to growth downstream)",
        category="growth",
        horizon_to_series_id={
            "2": "SPF_RGDP_MEDIAN_NOWCAST",
            "3": "SPF_RGDP_MEDIAN_Q1",
            "4": "SPF_RGDP_MEDIAN_Q2",
            "A": "SPF_RGDP_MEDIAN_ANNUAL_Y0",
            "B": "SPF_RGDP_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "4": "Q+2",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="UNEMP",
        description="Unemployment rate (U-3), quarterly average",
        category="labor",
        horizon_to_series_id={
            "2": "SPF_UNEMP_MEDIAN_NOWCAST",
            "3": "SPF_UNEMP_MEDIAN_Q1",
            "4": "SPF_UNEMP_MEDIAN_Q2",
            "A": "SPF_UNEMP_MEDIAN_ANNUAL_Y0",
            "B": "SPF_UNEMP_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "4": "Q+2",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="TBILL",
        description="3-month Treasury bill rate",
        category="rates",
        horizon_to_series_id={
            "2": "SPF_TBILL_MEDIAN_NOWCAST",
            "3": "SPF_TBILL_MEDIAN_Q1",
            "4": "SPF_TBILL_MEDIAN_Q2",
            "A": "SPF_TBILL_MEDIAN_ANNUAL_Y0",
            "B": "SPF_TBILL_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "4": "Q+2",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
    SPFVariable(
        spf_sheet="TBOND",
        description="10-year Treasury bond yield",
        category="rates",
        horizon_to_series_id={
            "2": "SPF_TBOND_MEDIAN_NOWCAST",
            "3": "SPF_TBOND_MEDIAN_Q1",
            "4": "SPF_TBOND_MEDIAN_Q2",
            "A": "SPF_TBOND_MEDIAN_ANNUAL_Y0",
            "B": "SPF_TBOND_MEDIAN_ANNUAL_Y1",
        },
        horizon_to_label={
            "2": "current quarter",
            "3": "Q+1",
            "4": "Q+2",
            "A": "current year annual",
            "B": "next year annual",
        },
    ),
)


def all_derived_series_ids() -> list[str]:
    """Return every derived series_id this registry produces.

    Useful for tests and DB inspection — answers "what should we expect
    in series_definitions after an SPF ingest?"
    """
    return [
        sid
        for var in SPF_VARIABLES
        for sid in var.horizon_to_series_id.values()
    ]
