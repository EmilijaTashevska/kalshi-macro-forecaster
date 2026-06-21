# Phase 1 Data Specification

This document is the locked, canonical list of data we ingest in Phase 1. It captures:

- The **identifier** to pull each series under (FRED-style where applicable)
- The **vintage policy** for that series — i.e. whether we need ALFRED for vintage history or can use the latest value
- The **category** for organizational purposes
- The **target phase** the series unlocks

Anything not in this document is intentionally out of scope for Phase 1.

---

## Vintage policy taxonomy

We label each series with one of three policies, which corresponds to how it behaves in our `point_in_time` query layer:

- **`revises: true`** — the series is revised after release (CPI, GDP, NFP, etc.). We ingest from ALFRED to get the full vintage history, and the `FIRST_KNOWN_AT` policy returns the value as known at any historical `as_of_date`.
- **`revises: false`** — the series does not get revised once a daily value is published (Treasury yields, S&P 500 closes, exchange rates). `vintage_date == release_date == observation_date` for every row. We can safely use the latest FRED value.
- **`revises: nowcast`** — special case: nowcasting series (GDPNow, NY Fed Nowcast) are *only ever* point-in-time. Each daily update is its own observation. We treat each release as a separate (`observation_date`, `vintage_date`) pair.

The boolean column in `series_definitions.revises` follows the first two cases. Nowcasts use the same boolean (set to `true`) but each daily release gets its own row.

---

## 1. Inflation (10 series)

| Series ID | Title | Frequency | Revises | Source | Phase notes |
|---|---|---|---|---|---|
| `CPIAUCSL` | CPI for All Urban Consumers, SA | Monthly | Yes | FRED/ALFRED | Headline CPI |
| `CPILFESL` | Core CPI (excl food and energy), SA | Monthly | Yes | FRED/ALFRED | What the Fed watches |
| `PCEPI` | PCE Price Index | Monthly | Yes | FRED/ALFRED | Fed's preferred gauge |
| `PCEPILFE` | Core PCE | Monthly | Yes | FRED/ALFRED | Underlying inflation |
| `PPIACO` | Producer Price Index, All Commodities | Monthly | Yes | FRED/ALFRED | Upstream signal |
| `PCETRIM12M656SFRBDAL` | Trimmed Mean PCE (Dallas Fed) | Monthly | Yes | FRED | Robust inflation measure |
| `MEDCPIM158SFRBCLE` | Median CPI (Cleveland Fed) | Monthly | Yes | FRED | Robust inflation measure |
| `STICKCPIM157SFRBATL` | Sticky Price CPI (Atlanta Fed) | Monthly | Yes | FRED | Persistent inflation |
| `T5YIFR` | 5Y5Y Forward Inflation Expectation | Daily | No | FRED | Market-implied long-run |
| `MICH` | Michigan 1Y Inflation Expectation | Monthly | Sometimes (light) | FRED | Survey-based |

## 2. Labor market (9 series)

| Series ID | Title | Frequency | Revises | Source | Phase notes |
|---|---|---|---|---|---|
| `PAYEMS` | Non-Farm Payrolls | Monthly | Yes | FRED/ALFRED | The big number |
| `UNRATE` | Unemployment Rate (U-3) | Monthly | Yes | FRED/ALFRED | Headline |
| `U6RATE` | U-6 Broad Underemployment | Monthly | Yes | FRED | Wider definition |
| `CIVPART` | Labor Force Participation Rate | Monthly | Yes | FRED/ALFRED | Context |
| `CES0500000003` | Average Hourly Earnings YoY | Monthly | Yes | FRED | Wage inflation |
| `JTSJOL` | Job Openings (JOLTS) | Monthly | Yes | FRED | Labor demand |
| `JTSQUR` | Quits Rate (JOLTS) | Monthly | Yes | FRED | Tightness signal |
| `ICSA` | Initial Jobless Claims (SA) | Weekly | Yes (light) | FRED | Highest-frequency labor signal |
| `CCSA` | Continuing Jobless Claims (SA) | Weekly | Yes (light) | FRED | Lagging but informative |

## 3. Growth and activity (10 series)

| Series ID | Title | Frequency | Revises | Source | Phase notes |
|---|---|---|---|---|---|
| `GDPC1` | Real GDP | Quarterly | Yes (heavy) | FRED/ALFRED | All three release vintages tracked |
| `INDPRO` | Industrial Production | Monthly | Yes | FRED/ALFRED | Production-side |
| `TCU` | Capacity Utilization | Monthly | Yes | FRED | Slack |
| `RSXFS` | Retail Sales (advance, excl food svc) | Monthly | Yes | FRED/ALFRED | Consumer pulse |
| `PCE` | Personal Consumption Expenditures | Monthly | Yes | FRED | Broader consumer spending |
| `PI` | Personal Income | Monthly | Yes | FRED | Income side |
| `DGORDER` | Durable Goods Orders | Monthly | Yes | FRED | Business investment proxy |
| `CFNAI` | Chicago Fed National Activity Index | Monthly | Yes (light) | FRED | Composite |
| `USSLIND` | Conference Board LEI | Monthly | Yes (light) | FRED | Recession composite |
| `GDPNOW` | Atlanta Fed GDPNow | Updated several times weekly | Nowcast | Atlanta Fed (scrape) | Special handling: each daily release is a row |

## 4. ISM / Regional Fed surveys (4 series)

| Series ID | Title | Frequency | Revises | Source | Phase notes |
|---|---|---|---|---|---|
| `NAPM` | ISM Manufacturing PMI | Monthly | No | FRED | May be incomplete; fall back to S&P Global PMI if blocked |
| `NMFCI` | ISM Services PMI | Monthly | No | FRED | Same caveat |
| `GACDISA066MSFRBNY` | Empire State Manufacturing Survey | Monthly | Yes | FRED | NY Fed |
| `GACDFSA066MSFRBPHI` | Philly Fed Survey | Monthly | Yes | FRED | |

## 5. Interest rates and yields (14 series)

All daily, none revise — Treasury yields are fixed at close.

| Series ID | Title | Frequency | Revises | Source |
|---|---|---|---|---|
| `DFF` | Effective Fed Funds Rate | Daily | No | FRED |
| `DFEDTARU` | Fed Funds Target Upper | Daily | No | FRED |
| `DFEDTARL` | Fed Funds Target Lower | Daily | No | FRED |
| `SOFR` | Secured Overnight Financing Rate | Daily | No | FRED |
| `DGS3MO` | 3-Month Treasury Bill | Daily | No | FRED |
| `DGS2` | 2-Year Treasury Yield | Daily | No | FRED |
| `DGS5` | 5-Year Treasury Yield | Daily | No | FRED |
| `DGS10` | 10-Year Treasury Yield | Daily | No | FRED |
| `DGS30` | 30-Year Treasury Yield | Daily | No | FRED |
| `T10Y2Y` | 10Y minus 2Y Spread | Daily | No | FRED |
| `T10Y3M` | 10Y minus 3M Spread | Daily | No | FRED |
| `DFII5` | 5-Year TIPS Yield | Daily | No | FRED |
| `DFII10` | 10-Year TIPS Yield | Daily | No | FRED |
| `T5YIE` | 5-Year Breakeven Inflation | Daily | No | FRED |
| `T10YIE` | 10-Year Breakeven Inflation | Daily | No | FRED |

**Plus** Fed funds futures front 6 contracts from CME, daily settlements. This is a separate ingestor; the data lands in `series_observations` under composite series IDs like `FF_FUT_001`, `FF_FUT_002`, etc.

## 6. Financial markets (9 series)

| Series ID | Title | Frequency | Revises | Source |
|---|---|---|---|---|
| `SP500` | S&P 500 Close | Daily | No | FRED |
| `VIXCLS` | VIX | Daily | No | FRED |
| `NASDAQCOM` | NASDAQ Composite | Daily | No | FRED |
| `DTWEXBGS` | Trade-Weighted Dollar Index | Daily | No | FRED |
| `DCOILWTICO` | WTI Crude | Daily | No | FRED |
| `GOLDAMGBD228NLBM` | Gold (LBMA AM Fix) | Daily | No | FRED |
| `BAA10YM` | BAA Corporate – 10Y Spread | Daily | No | FRED |
| `BAMLH0A0HYM2` | ICE BofA High Yield Spread | Daily | No | FRED |
| `NFCI` | Chicago Fed National Financial Conditions Index | Weekly | Yes (light) | FRED |

## 7. Money and banking (5 series)

| Series ID | Title | Frequency | Revises | Source |
|---|---|---|---|---|
| `M2SL` | M2 Money Supply | Monthly | Yes | FRED |
| `WRESBAL` | Bank Reserves | Weekly | Yes (light) | FRED |
| `WALCL` | Fed Balance Sheet Total Assets | Weekly | Yes (light) | FRED |
| `DRTSCILM` | SLOOS C&I Lending Standards | Quarterly | Yes | FRED |
| `MORTGAGE30US` | 30-Year Mortgage Rate | Weekly | No | FRED |

## 8. Housing (5 series)

| Series ID | Title | Frequency | Revises | Source |
|---|---|---|---|---|
| `HOUST` | Housing Starts | Monthly | Yes | FRED/ALFRED |
| `PERMIT` | Building Permits | Monthly | Yes | FRED/ALFRED |
| `EXHOSLUSM495S` | Existing Home Sales | Monthly | Yes | FRED |
| `HSN1F` | New Home Sales | Monthly | Yes | FRED |
| `CSUSHPINSA` | Case-Shiller HPI | Monthly | Yes (light) | FRED |

## 9. International (4 series)

| Series ID | Title | Frequency | Revises | Source |
|---|---|---|---|---|
| `CP0000EZ19M086NEST` | Eurozone HICP (or scrape Eurostat) | Monthly | Yes | FRED / Eurostat |
| `ECB_DFR` | ECB Deposit Facility Rate | Per meeting | No | ECB scrape |
| `BOE_BANK_RATE` | BoE Bank Rate | Per meeting | No | BoE scrape |
| `DTWEXBGS` | Trade-Weighted Dollar (dup of section 6) | Daily | No | FRED |

## 10. Survey of Professional Forecasters

Phila Fed publishes the SPF as quarterly CSVs. We ingest each forecaster's median/mean for the main targets. This becomes its own table in Phase 1.3 since the shape (per-respondent forecasts per quarter for multiple horizons) doesn't fit `series_observations` cleanly.

---

## Total

- **~71 numeric series** in `series_observations`
- **~5,000 text documents** in `text_documents`
- Free APIs throughout. Required keys: FRED (free), BLS (free), BEA (free).

---

## Vintage policy quick reference

| revises | Behaviour |
|---|---|
| `true` | Multiple rows per `observation_date` with different `vintage_date`. PIT query picks the latest vintage with `vintage_date <= as_of_date`. |
| `false` | One row per `observation_date`. PIT query trivially returns it. |
| nowcast (revises true, daily updates) | Each daily update is a fresh `vintage_date`. PIT history shows the path of the nowcast over time. |

---

## Sub-phase mapping

- **Phase 1.2** ingests all sections 1-3 + 5-9 from FRED (~67 series). The FRED/ALFRED client lands here.
- **Phase 1.3** ingests section 4 (ISM) + section 9 (international, ECB/BoE) + section 10 (SPF) + Fed funds futures from CME.
- **Phase 1.4** ingests the text corpus.
- **Phase 1.5** ingests Kalshi + Polymarket markets.
- **Phase 1.6** ingests the economic calendar (consensus + actual + surprise).
- **Phase 1.7** produces the data quality dashboard.

The intent is that after Phase 1.7, we can run `make db-summary` and see every row in this document accounted for.
