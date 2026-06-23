# Phase 1.5 — Kalshi research notes

Recorded findings from live recon against the Kalshi public API on
2026-06-23. Captured here so a future reader (you, or another agent)
doesn't have to re-discover the API shape.

## Top-line: macro markets live in Economics + Financials categories

Kalshi's API exposes ~11,000 series. Walking ``/historical/markets``
without a category filter is *not* useful — the first 6,000 settled
markets we sampled were dominated by sports parlays and crypto combos
with hash-like tickers (e.g. ``KXMVECROSSCATEGORY-...``).

The right entry point is ``/series?category=Economics`` (591 series)
and ``/series?category=Financials`` (670 series). From there we
follow each series to its events and markets.

## API request flow we'll use

```text
/series?category=Economics              → list candidate series
   ↓ (filter by our allowlist)
/events?series_ticker=KXFED&status=settled
                                        → list events under each series
   ↓
/events/{event_ticker}                  → market list inside the event
   ↓
/historical/markets/{ticker}/candlesticks?...
                                        → daily price history per market
```

## Series → template_id mapping

Curated allowlist (see ``data/kalshi_classifier.py`` for the
authoritative version):

| Template | Series tickers |
|---|---|
| ``fed_decision`` | ``KXFED``, ``KXFEDDECISION``, ``FEDDECISION``, ``KXRATECUTE``, ``KXTERMINALRATE``, ``TERMINALRATE``, ``KXFEDRATEMIN``, ``LOWESTRATE``, ``KXFEDCHGCOUNT`` |
| ``cpi_yoy`` | ``KXACPI``, ``KXCPICORE``, ``ACPICORE``, ``ACPICORE-``, ``KXECONSTATCPIYOY``, ``KXECONSTATCORECPIYOY``, ``LCPIYOY``, ``CPICOREYOY``, ``KXCOREUND`` |
| ``nfp`` | ``KXPAYROLLS``, ``KXPROLLS``, ``KXJOBLESSCLAIMS``, ``KXADP`` |
| ``unemployment`` | ``KXECONSTATU3``, ``KXUE``, ``U3MIN``, ``U3MAX``, ``KXU3MAX`` |
| ``gdp`` | ``GDP``, ``KXGDPNOM``, ``KXNGDPQ``, ``NGDPQ``, ``NGDP`` |
| ``yield_10y`` | ``KXTNOTE``, ``TNOTE``, ``KXNOTE10M``, ``KXNOTE10W``, ``KXTNOTEW``, ``TNOTEW``, ``TNOTED`` |
| ``recession_12m`` | ``KXRECSSNBER``, ``RECSSNBER``, ``KXNBERRECESSQ``, ``KXSAHM`` |

## Known wart: changing tickers

Tickers without the ``KX`` prefix (``GDP``, ``TNOTE``, ``FEDDECISION``)
are the *older* ticker convention; new markets often duplicate them
under ``KX<NAME>`` prefixes. Both can coexist for the same conceptual
question. Our allowlist deliberately includes both so we can ingest
the full history.

## What we are NOT ingesting in this phase

- **Polymarket.** Subgraph queries are doable but yield much thinner
  macro coverage (Fed decisions yes, but not consistent CPI/NFP).
  Cost/benefit doesn't justify the engineering yet.
- **Daily SOFR / Eurodollar / dollar / euro markets.** They're under
  Financials but aren't direct targets for our 7 templates.
- **Foreign macro series** (``KXCBDECISIONKOREA``, ``KXBOJDECISION``,
  ``KXBOE``, ``KXEZDEPRATE`` etc.). Useful future expansion; out of
  scope for Phase 1.5.

## Strike parsing approach (per template)

Kalshi market titles encode the strike in heterogeneous formats:

- ``"Will the Fed cut rates at the next meeting?"`` — fed_decision, binary, no strike
- ``"Will CPI YoY be 3.0% or higher in October?"`` — cpi_yoy, strike=3.0, direction=above
- ``"Will the 10Y close above 4.35% this month?"`` — yield_10y, strike=4.35, direction=above

We'll handle this with one regex per template_id, applied to the
``yes_sub_title`` (which usually has the cleanest expression of the
threshold). Unparseable markets get ``strike_value = NULL`` and
``strike_direction = ""`` — they still get stored, just without
structured strike info.
