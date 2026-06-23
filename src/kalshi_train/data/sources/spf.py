"""Async client for the Philadelphia Fed's Survey of Professional Forecasters.

The Philly Fed publishes a small number of Excel workbooks, each
containing one worksheet per macroeconomic variable. The workbooks we
care about (Phase 1.3):

  - MedianLevel.xlsx    median forecasts in levels
  - MeanLevel.xlsx      mean forecasts in levels
  - Dispersion_1.xlsx   cross-sectional dispersion in levels

Each worksheet has the columns ``YEAR``, ``QUARTER``, then one column
per forecast horizon. The horizon-column naming convention is documented
by the SPF docs: ``X1`` is the real-time historical value (previous
quarter), ``X2`` is the nowcast (forecast for the survey quarter),
``X3``-``X6`` are quarterly forecasts at horizons 1-4, and ``XA``,
``XB``, ``XC``, ``XD`` are annual forecasts for the survey year and
the following years.

Known wart we have to work around: the xlsx files Philly Fed publishes
contain malformed ``dcterms:modified`` and ``dcterms:created`` core
properties that openpyxl refuses to parse. We strip these from
``docProps/core.xml`` before handing the buffer to pandas. The bug has
been reported upstream but is unlikely to be fixed because the files
are technically out of spec.
"""

from __future__ import annotations

import io
import logging
import re
import warnings
import zipfile
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# ── Canonical download URLs ───────────────────────────────────────────

BASE_URL = "https://www.philadelphiafed.org"
MEDIAN_LEVEL_URL = (
    f"{BASE_URL}/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/medianLevel.xlsx"
)
MEAN_LEVEL_URL = (
    f"{BASE_URL}/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/meanLevel.xlsx"
)
DISPERSION_LEVEL_URL = (
    f"{BASE_URL}/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/Dispersion_1.xlsx"
)


class SPFAPIError(RuntimeError):
    """Network or parsing failure while pulling SPF data."""


@dataclass(frozen=True, slots=True)
class SPFWorkbook:
    """Parsed contents of an SPF Excel workbook.

    Keys of ``sheets`` are SPF variable names (CPI, RGDP, UNEMP, ...).
    Values are DataFrames with columns YEAR, QUARTER, and the horizon
    columns. Missing values are ``NaN`` (the source file uses '#N/A').
    """

    source_url: str
    sheets: dict[str, pd.DataFrame]


# ── xlsx repair: strip malformed metadata ─────────────────────────────


_BAD_DCTERMS = re.compile(
    r"<dcterms:(modified|created)[^>]*>[^<]*</dcterms:(modified|created)>"
)


def _patch_xlsx_metadata(raw: bytes) -> bytes:
    """Strip dcterms:modified / dcterms:created from docProps/core.xml.

    These fields are sometimes serialized in a format openpyxl can't
    parse (e.g. a bare ``YYYY-MM-DD`` instead of ISO datetime). Pandoc
    and Excel itself tolerate this; openpyxl raises. The cleanest
    workaround is to drop the offending tags before parsing — they're
    metadata, not actual data, so we lose nothing of substance.
    """
    src = io.BytesIO(raw)
    dst = io.BytesIO()
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                text = data.decode("utf-8", errors="ignore")
                data = _BAD_DCTERMS.sub("", text).encode("utf-8")
            zout.writestr(item, data)
    dst.seek(0)
    return dst.getvalue()


def _parse_workbook(raw: bytes, source_url: str) -> SPFWorkbook:
    """Parse an SPF xlsx into a SPFWorkbook.

    openpyxl can also emit a stream of header/footer warnings that
    we don't care about (the Philly Fed file has decorative print
    headers we'll never use). We suppress them locally so logs stay
    clean.
    """
    patched = _patch_xlsx_metadata(raw)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        wb: dict[str, pd.DataFrame] = pd.read_excel(
            io.BytesIO(patched), sheet_name=None, engine="openpyxl"
        )
    return SPFWorkbook(source_url=source_url, sheets=wb)


# ── The client ────────────────────────────────────────────────────────


class SPFClient:
    """Minimal async client for downloading SPF data files.

    Usage::

        async with SPFClient() as spf:
            wb = await spf.get_median_level()
            cpi = wb.sheets["CPI"]   # DataFrame
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        median_url: str = MEDIAN_LEVEL_URL,
        mean_url: str = MEAN_LEVEL_URL,
        dispersion_url: str = DISPERSION_LEVEL_URL,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._median_url = median_url
        self._mean_url = mean_url
        self._dispersion_url = dispersion_url
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            timeout=self._timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "kalshi-train/0.0.1 (research; non-commercial)"},
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

    # ── Public methods ──

    async def get_median_level(self) -> SPFWorkbook:
        """Download and parse the medianLevel.xlsx workbook."""
        return await self._fetch_workbook(self._median_url)

    async def get_mean_level(self) -> SPFWorkbook:
        """Download and parse the meanLevel.xlsx workbook."""
        return await self._fetch_workbook(self._mean_url)

    async def get_dispersion_level(self) -> SPFWorkbook:
        """Download and parse the Dispersion_1.xlsx workbook."""
        return await self._fetch_workbook(self._dispersion_url)

    # ── Internal ──

    async def _fetch_workbook(self, url: str) -> SPFWorkbook:
        if self._client is None:
            raise RuntimeError("SPFClient must be used as an async context manager.")
        logger.info("Fetching SPF workbook from %s", url)
        resp = await self._client.get(url)
        if resp.status_code >= 400:
            raise SPFAPIError(
                f"Philly Fed returned HTTP {resp.status_code} for {url}: "
                f"{resp.text[:200]}"
            )
        # SPF responses use a redirect to a hash-versioned URL but follow_redirects=True
        # handles that. Sometimes the 200 still hands back an HTML 404 page — guard.
        ctype = resp.headers.get("content-type", "")
        if "html" in ctype.lower():
            raise SPFAPIError(
                f"Expected xlsx from {url} but got HTML (content-type={ctype}). "
                "The URL may have been deprecated."
            )
        return _parse_workbook(resp.content, source_url=url)
