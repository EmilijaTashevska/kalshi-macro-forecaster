"""Phase 2 unit tests — labels, features, metrics, and pipeline on synthetic data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalshi_train.data.fomc_calendar import fomc_meeting_dates, previous_business_day
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    Observation,
    SeriesDefinition,
    upsert_observation,
    upsert_series_definition,
)
from kalshi_train.eval.metrics import baseline_always_half, compute_metrics, reliability_bins
from kalshi_train.eval.splits import temporal_train_val_test_split
from kalshi_train.features.fed_cut import build_feature_matrix, feature_names
from kalshi_train.models.xgboost_baseline import train_xgboost_temporal_cv
from kalshi_train.targets.fed_cut import build_fed_cut_examples


@pytest.fixture
def fed_cut_db(tmp_db: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Minimal DB with DFEDTARU + a tiny synthetic FOMC calendar."""
    calendar = tmp_path / "fomc.txt"
    calendar.write_text(
        "\n".join(
            [
                "2020-03-15",
                "2020-03-19",
                "2020-04-28",
                "2020-06-10",
                "2020-07-29",
                "2020-09-16",
                "2020-11-05",
                "2020-12-16",
                "2021-01-27",
                "2021-03-17",
                "2021-04-28",
                "2021-06-16",
                "2021-07-28",
                "2021-09-22",
                "2021-11-03",
                "2021-12-15",
                "2022-01-26",
                "2022-03-16",
                "2022-05-04",
                "2022-06-15",
            ]
        )
        + "\n"
    )

    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn,
            SeriesDefinition(
                series_id="DFEDTARU",
                source="FRED",
                title="Fed Funds Target Upper",
                frequency="daily",
                revises=False,
                category="rates",
            ),
        )
        schedule = [
            ("2020-03-15", 1.75),
            ("2020-03-19", 0.25),
            ("2020-04-28", 0.25),
            ("2020-06-10", 0.25),
            ("2020-07-29", 0.25),
            ("2020-09-16", 0.25),
            ("2020-11-05", 0.25),
            ("2020-12-16", 0.25),
            ("2021-01-27", 0.25),
            ("2021-03-17", 0.25),
            ("2021-04-28", 0.25),
            ("2021-06-16", 0.25),
            ("2021-07-28", 0.25),
            ("2021-09-22", 0.25),
            ("2021-11-03", 0.25),
            ("2021-12-15", 0.25),
            ("2022-01-26", 0.25),
            ("2022-03-16", 0.50),
            ("2022-05-04", 1.00),
            ("2022-06-15", 1.75),
        ]
        for meeting, rate in schedule:
            upsert_observation(
                conn,
                Observation(
                    series_id="DFEDTARU",
                    observation_date=meeting,
                    vintage_date=meeting,
                    release_date=f"{meeting}T18:00:00+00:00",
                    value=rate,
                ),
            )
        conn.commit()

    return tmp_db, calendar


def test_fomc_calendar_loads_static_subset(tmp_path: Path) -> None:
    cal = tmp_path / "meetings.txt"
    cal.write_text("2020-01-29\n2020-03-15\n")
    dates = fomc_meeting_dates(
        "2020-01-01",
        "2020-12-31",
        calendar_path=cal,
        prefer_db=False,
    )
    assert dates == (date(2020, 1, 29), date(2020, 3, 15))


def test_previous_business_day_skips_weekends() -> None:
    assert previous_business_day("2020-03-16") == date(2020, 3, 13)


def test_build_fed_cut_examples_with_custom_calendar(
    fed_cut_db: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, calendar = fed_cut_db

    def _meetings(start, end, *, db_path=None, calendar_path=None, prefer_db=True):
        return fomc_meeting_dates(
            start, end, db_path=db_path, calendar_path=calendar, prefer_db=False
        )

    monkeypatch.setattr("kalshi_train.targets.fed_cut.fomc_meeting_dates", _meetings)
    examples = build_fed_cut_examples(start="2020-03-15", end="2022-06-15", db_path=db_path)
    assert len(examples) == 19
    cut = next(e for e in examples if e.meeting_date == date(2020, 3, 19))
    assert cut.label == 1
    assert cut.rate_before == 1.75
    assert cut.rate_after == 0.25


def test_feature_matrix_shape(
    fed_cut_db: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, calendar = fed_cut_db
    monkeypatch.setattr(
        "kalshi_train.targets.fed_cut.fomc_meeting_dates",
        lambda start, end, **kw: fomc_meeting_dates(
            start, end, calendar_path=calendar, prefer_db=False, db_path=db_path
        ),
    )
    examples = build_fed_cut_examples(start="2020-03-15", end="2022-06-15", db_path=db_path)
    matrix = build_feature_matrix(examples, db_path=db_path, include_context=True)
    assert len(matrix) == len(examples)
    assert set(feature_names()) <= set(matrix.columns)
    assert matrix["ff_target_upper"].notna().all()


def test_temporal_split_is_chronological() -> None:
    df = pd.DataFrame(
        {
            "as_of_date": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "label": range(10),
            "x": range(10),
        }
    )
    split = temporal_train_val_test_split(df)
    assert split.train["as_of_date"].max() < split.val["as_of_date"].min()
    assert split.val["as_of_date"].max() < split.test["as_of_date"].min()


def test_metrics_and_reliability_bins() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.3, 0.7, 0.8])
    report = compute_metrics(y, p)
    assert report.brier < 0.2
    assert report.log_loss < 1.0
    bins = reliability_bins(y, p, n_bins=2)
    assert len(bins) == 2
    assert baseline_always_half(3).tolist() == [0.5, 0.5, 0.5]


@pytest.mark.slow
def test_xgboost_temporal_cv_runs(
    fed_cut_db: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, calendar = fed_cut_db
    monkeypatch.setattr(
        "kalshi_train.targets.fed_cut.fomc_meeting_dates",
        lambda start, end, **kw: fomc_meeting_dates(
            start, end, calendar_path=calendar, prefer_db=False, db_path=db_path
        ),
    )
    examples = build_fed_cut_examples(start="2020-03-15", end="2022-06-15", db_path=db_path)
    matrix = build_feature_matrix(examples, db_path=db_path)
    cols = [c for c in feature_names() if c in matrix.columns]
    result = train_xgboost_temporal_cv(matrix, cols, n_splits=3)
    assert result.mean_metrics.n_samples > 0
    assert not result.feature_importance.empty
