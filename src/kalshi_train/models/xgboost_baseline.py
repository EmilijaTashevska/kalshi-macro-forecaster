"""Classical ML baselines — XGBoost with temporal cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

from kalshi_train.eval.metrics import MetricReport, compute_metrics
from kalshi_train.eval.splits import expanding_window_cv, split_xy


@dataclass(frozen=True, slots=True)
class XGBoostCVResult:
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    feature_importance: pd.Series
    fold_metrics: list[MetricReport]
    mean_metrics: MetricReport


def _default_xgb_params() -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def train_xgboost_temporal_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    n_splits: int = 5,
    xgb_params: dict[str, Any] | None = None,
) -> XGBoostCVResult:
    """Expanding-window CV with out-of-fold predictions for diagnostics."""
    ordered = df.sort_values("as_of_date")
    x, y = split_xy(ordered, feature_cols, label_col=label_col)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_imputed = imputer.fit_transform(x)

    tscv = expanding_window_cv(len(ordered), n_splits=n_splits)
    oof = np.full(len(ordered), np.nan)
    fold_metrics: list[MetricReport] = []
    importances = np.zeros(len(feature_cols), dtype=float)
    params = _default_xgb_params()
    if xgb_params:
        params.update(xgb_params)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(x_imputed)):
        model = xgb.XGBClassifier(**params)
        model.fit(x_imputed[train_idx], y[train_idx])
        probs = model.predict_proba(x_imputed[test_idx])[:, 1]
        oof[test_idx] = probs
        fold_metrics.append(compute_metrics(y[test_idx], probs))
        importances += model.feature_importances_
        _ = fold_idx

    valid = ~np.isnan(oof)
    mean_metrics = compute_metrics(y[valid], oof[valid])
    importance = pd.Series(importances / max(len(fold_metrics), 1), index=feature_cols)
    importance = importance.sort_values(ascending=False)

    return XGBoostCVResult(
        oof_predictions=oof,
        oof_indices=np.arange(len(ordered)),
        feature_importance=importance,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
    )


def train_xgboost_final(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    xgb_params: dict[str, Any] | None = None,
) -> tuple[xgb.XGBClassifier, SimpleImputer]:
    """Fit on the full training split for held-out test evaluation."""
    x, y = split_xy(train_df, feature_cols, label_col=label_col)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_imputed = imputer.fit_transform(x)
    params = _default_xgb_params()
    if xgb_params:
        params.update(xgb_params)
    model = xgb.XGBClassifier(**params)
    model.fit(x_imputed, y)
    return model, imputer


def predict_proba(
    model: xgb.XGBClassifier,
    imputer: SimpleImputer,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    x = imputer.transform(df[feature_cols].to_numpy(dtype=float))
    return model.predict_proba(x)[:, 1]
