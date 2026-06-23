"""Unit tests for the SPF client.

Uses ``httpx.MockTransport`` so no network is involved. We construct
a minimal in-memory xlsx workbook that triggers the same metadata bug
the real Philly Fed files have, verifying that the workaround is in
place and the parsed sheets are what we expect.
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pandas as pd
import pytest
from openpyxl import Workbook

from kalshi_train.data.sources.spf import (
    SPFAPIError,
    SPFClient,
    _parse_workbook,
    _patch_xlsx_metadata,
)


def _build_test_xlsx() -> bytes:
    """Construct a small, valid xlsx that mimics the SPF CPI sheet."""
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "CPI"
    ws.append(["YEAR", "QUARTER", "CPI2", "CPI3", "CPIA", "CPIB"])
    ws.append([2024, 1, 2.5, 2.3, 2.4, 2.2])
    ws.append([2024, 2, 3.0, 2.6, 2.7, 2.3])
    # A second sheet for a different variable.
    rgdp = wb.create_sheet("RGDP")
    rgdp.append(["YEAR", "QUARTER", "RGDP2", "RGDP3"])
    rgdp.append([2024, 1, 21500.0, 21600.0])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _inject_malformed_metadata(raw: bytes) -> bytes:
    """Replace docProps/core.xml with one that has bad dcterms:modified.

    The real Philly Fed files have a ``modified`` value that openpyxl
    rejects; we reproduce that here to make sure our workaround handles
    it. Without ``_patch_xlsx_metadata`` the file would fail to parse.
    """
    src = io.BytesIO(raw)
    dst = io.BytesIO()
    bad_core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/'
        'package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2024-05-15</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2024-05-15</dcterms:modified>'
        "</cp:coreProperties>"
    )
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        wrote_core = False
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = bad_core.encode("utf-8")
                wrote_core = True
            zout.writestr(item, data)
        if not wrote_core:
            zout.writestr("docProps/core.xml", bad_core)
    return dst.getvalue()


def _make_client(transport: httpx.MockTransport, urls: dict[str, str]) -> SPFClient:
    client = SPFClient(
        median_url=urls["median"],
        mean_url=urls.get("mean", "https://example.invalid/mean.xlsx"),
        dispersion_url=urls.get("dispersion", "https://example.invalid/dispersion.xlsx"),
    )
    client._client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    return client


# ── Direct unit tests of the parser ──────────────────────────────────


def test_parse_workbook_handles_clean_xlsx() -> None:
    raw = _build_test_xlsx()
    wb = _parse_workbook(raw, source_url="https://example.invalid/test.xlsx")
    assert set(wb.sheets.keys()) == {"CPI", "RGDP"}
    assert list(wb.sheets["CPI"].columns) == ["YEAR", "QUARTER", "CPI2", "CPI3", "CPIA", "CPIB"]
    assert wb.sheets["CPI"].shape == (2, 6)


def test_patch_xlsx_metadata_strips_bad_dcterms() -> None:
    raw = _build_test_xlsx()
    malformed = _inject_malformed_metadata(raw)

    # Without patching, openpyxl chokes on the bare YYYY-MM-DD dcterms.
    with pytest.raises(TypeError):
        pd.read_excel(io.BytesIO(malformed), engine="openpyxl")

    patched = _patch_xlsx_metadata(malformed)
    # After patching it loads cleanly.
    df = pd.read_excel(io.BytesIO(patched), sheet_name="CPI", engine="openpyxl")
    assert len(df) == 2
    assert df.iloc[0]["CPI2"] == 2.5


def test_parse_workbook_uses_patch_so_malformed_files_load() -> None:
    raw = _build_test_xlsx()
    malformed = _inject_malformed_metadata(raw)
    wb = _parse_workbook(malformed, source_url="https://example.invalid/bad.xlsx")
    assert "CPI" in wb.sheets
    assert wb.sheets["CPI"].iloc[1]["CPI3"] == 2.6


# ── HTTP path ─────────────────────────────────────────────────────────


async def test_get_median_level_returns_parsed_workbook() -> None:
    raw = _build_test_xlsx()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.invalid/median.xlsx"
        return httpx.Response(
            200,
            content=raw,
            headers={
                "content-type": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            },
        )

    # We bypass __aenter__ here so the mock transport we pre-inject
    # via _client isn't overwritten by a freshly-built AsyncClient.
    client = _make_client(
        httpx.MockTransport(handler),
        urls={"median": "https://example.invalid/median.xlsx"},
    )
    wb = await client.get_median_level()
    assert "CPI" in wb.sheets
    assert wb.sheets["CPI"].iloc[0]["CPI2"] == 2.5
    await client._client.aclose()


async def test_get_median_level_raises_when_html_returned() -> None:
    """If Philly Fed renames the URL we get HTML 404 with a 200 status.
    The client must detect this from content-type and raise."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<!DOCTYPE html><html>Not found</html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    client = _make_client(
        httpx.MockTransport(handler),
        urls={"median": "https://example.invalid/median.xlsx"},
    )
    with pytest.raises(SPFAPIError, match="HTML"):
        await client.get_median_level()
    await client._client.aclose()


async def test_get_median_level_raises_on_4xx() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found", headers={"content-type": "text/plain"})

    client = _make_client(
        httpx.MockTransport(handler),
        urls={"median": "https://example.invalid/median.xlsx"},
    )
    with pytest.raises(SPFAPIError, match="404"):
        await client.get_median_level()
    await client._client.aclose()
