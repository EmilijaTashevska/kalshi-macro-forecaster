"""Hand-engineered features for the Fed-cut XGBoost baseline (~25 features).

Feature names are stable column identifiers used in reports and tests.
Each feature is computed exclusively through the PIT layer in ``base.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from kalshi_train.features.base import pit_change_bdays, pit_level, pit_yoy_index_change
from kalshi_train.targets.fed_cut import FedCutExample

FeatureFn = Callable[[FedCutExample, Path | None], float | None]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    compute: FeatureFn
    group: str
    description: str = ""


def _level(series_id: str) -> FeatureFn:
    def _fn(ex: FedCutExample, db_path: Path | None) -> float | None:
        return pit_level(series_id, ex.as_of_date, db_path=db_path)

    return _fn


def _change(series_id: str, bdays: int) -> FeatureFn:
    def _fn(ex: FedCutExample, db_path: Path | None) -> float | None:
        return pit_change_bdays(series_id, ex.as_of_date, bdays, db_path=db_path)

    return _fn


def _yoy(series_id: str, months: int = 12) -> FeatureFn:
    def _fn(ex: FedCutExample, db_path: Path | None) -> float | None:
        return pit_yoy_index_change(series_id, ex.as_of_date, months=months, db_path=db_path)

    return _fn


def _days_to_meeting(ex: FedCutExample, _db_path: Path | None) -> float:
    return float((ex.meeting_date - ex.as_of_date).days)


def _meetings_since_last_cut(
    ex: FedCutExample,
    db_path: Path | None,
    *,
    history: Sequence[FedCutExample] | None = None,
) -> float | None:
    """Count consecutive prior meetings without a cut (including holds/hikes)."""
    if history is None:
        return None
    streak = 0
    for prior in reversed(history):
        if prior.meeting_date >= ex.meeting_date:
            continue
        if prior.label == 1:
            break
        streak += 1
    return float(streak)


def _prior_meeting_was_cut(
    ex: FedCutExample,
    _db_path: Path | None,
    *,
    history: Sequence[FedCutExample] | None = None,
) -> float | None:
    if history is None:
        return None
    for prior in reversed(history):
        if prior.meeting_date >= ex.meeting_date:
            continue
        return float(prior.label)
    return None


FED_CUT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # ── Policy rates ─────────────────────────────────────────────────
    FeatureSpec("ff_target_upper", _level("DFEDTARU"), "rates", "Fed funds target upper bound"),
    FeatureSpec("ff_target_lower", _level("DFEDTARL"), "rates", "Fed funds target lower bound"),
    FeatureSpec("ff_effective", _level("DFF"), "rates", "Effective fed funds rate"),
    FeatureSpec("sofr", _level("SOFR"), "rates", "SOFR overnight rate"),
    FeatureSpec("dgs2", _level("DGS2"), "rates", "2Y Treasury yield"),
    FeatureSpec("dgs10", _level("DGS10"), "rates", "10Y Treasury yield"),
    FeatureSpec("t10y2y", _level("T10Y2Y"), "rates", "10Y-2Y yield spread"),
    FeatureSpec("t10y3m", _level("T10Y3M"), "rates", "10Y-3M yield spread"),
    FeatureSpec("dff_chg_21bd", _change("DFF", 21), "rates", "21bd change in effective FF"),
    FeatureSpec("dgs2_chg_21bd", _change("DGS2", 21), "rates", "21bd change in 2Y yield"),
    FeatureSpec("dgs10_chg_63bd", _change("DGS10", 63), "rates", "63bd change in 10Y yield"),
    # ── Inflation ────────────────────────────────────────────────────
    FeatureSpec("cpi_yoy", _yoy("CPIAUCSL"), "inflation", "Headline CPI YoY % (index ratio)"),
    FeatureSpec("core_cpi_yoy", _yoy("CPILFESL"), "inflation", "Core CPI YoY %"),
    FeatureSpec("pce_yoy", _yoy("PCEPI"), "inflation", "Headline PCE YoY %"),
    FeatureSpec("t5yie", _level("T5YIE"), "inflation", "5Y breakeven inflation"),
    FeatureSpec("t10yie", _level("T10YIE"), "inflation", "10Y breakeven inflation"),
    # ── Labor ────────────────────────────────────────────────────────
    FeatureSpec("unrate", _level("UNRATE"), "labor", "U-3 unemployment rate"),
    FeatureSpec("payems_chg_21bd", _change("PAYEMS", 21), "labor", "21bd change in payrolls"),
    FeatureSpec("icsa", _level("ICSA"), "labor", "Initial jobless claims"),
    FeatureSpec("civpart", _level("CIVPART"), "labor", "Labor force participation"),
    # ── Growth / activity ────────────────────────────────────────────
    FeatureSpec(
        "indpro_chg_63bd",
        _change("INDPRO", 63),
        "growth",
        "63bd change in industrial production",
    ),
    FeatureSpec("rsxfs_chg_21bd", _change("RSXFS", 21), "growth", "21bd change in retail sales"),
    # ── Financial conditions ───────────────────────────────────────────
    FeatureSpec("vix", _level("VIXCLS"), "markets", "VIX close"),
    FeatureSpec("sp500_chg_21bd", _change("SP500", 21), "markets", "21bd change in S&P 500"),
    FeatureSpec("baa10ym", _level("BAA10YM"), "markets", "BAA corporate spread vs 10Y"),
    FeatureSpec("hy_spread", _level("BAMLH0A0HYM2"), "markets", "High-yield OAS"),
    # ── Fed balance sheet ────────────────────────────────────────────
    FeatureSpec("walcl_chg_63bd", _change("WALCL", 63), "money", "63bd change in Fed assets"),
    # ── Meeting context ────────────────────────────────────────────────
    FeatureSpec("days_to_meeting", _days_to_meeting, "context", "Calendar days until FOMC"),
)

CONTEXT_FEATURE_NAMES = ("meetings_since_last_cut", "prior_meeting_was_cut")


def feature_names(*, include_context: bool = True) -> list[str]:
    names = [spec.name for spec in FED_CUT_FEATURE_SPECS]
    if include_context:
        names.extend(CONTEXT_FEATURE_NAMES)
    return names


def build_feature_matrix(
    examples: Sequence[FedCutExample],
    *,
    db_path: Path | None = None,
    include_context: bool = True,
) -> pd.DataFrame:
    """Materialize one feature row per ``FedCutExample``.

    Rows are indexed by ``resolution_id``. Context features that depend on
    label history are computed in chronological order (safe: they only look
    at *past* meeting outcomes, never the current row's label).
    """
    rows: list[dict[str, Any]] = []
    history: list[FedCutExample] = []
    sorted_examples = sorted(examples, key=lambda e: e.meeting_date)

    for ex in sorted_examples:
        row: dict[str, Any] = {
            "meeting_date": ex.meeting_date,
            "as_of_date": ex.as_of_date,
            "label": ex.label,
        }
        for spec in FED_CUT_FEATURE_SPECS:
            row[spec.name] = spec.compute(ex, db_path)
        if include_context:
            row["meetings_since_last_cut"] = _meetings_since_last_cut(ex, db_path, history=history)
            row["prior_meeting_was_cut"] = _prior_meeting_was_cut(ex, db_path, history=history)
        rows.append(row)
        history.append(ex)

    df = pd.DataFrame(rows)
    df.index = [f"fed_cut_{d.isoformat()}" for d in df["meeting_date"]]
    df.index.name = "resolution_id"
    return df
