"""Forecast evaluation metrics and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

DEFAULT_RELIABILITY_BINS = 10


@dataclass(frozen=True, slots=True)
class MetricReport:
    brier: float
    log_loss: float
    n_samples: int
    positive_rate: float
    mean_prediction: float


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    fraction_positive: float
    count: int


def clip_probabilities(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return cast(np.ndarray, np.clip(probs, eps, 1.0 - eps))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricReport:
    """Brier score and log loss for binary outcomes."""
    y = np.asarray(y_true, dtype=float)
    p = clip_probabilities(np.asarray(y_prob, dtype=float))
    return MetricReport(
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        n_samples=len(y),
        positive_rate=float(y.mean()) if len(y) else 0.0,
        mean_prediction=float(p.mean()) if len(p) else 0.0,
    )


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> list[ReliabilityBin]:
    """Bin predictions for a reliability / calibration diagram."""
    y = np.asarray(y_true, dtype=float)
    p = clip_probabilities(np.asarray(y_prob, dtype=float))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    mean_predicted=float("nan"),
                    fraction_positive=float("nan"),
                    count=0,
                )
            )
            continue
        bins.append(
            ReliabilityBin(
                bin_lower=lo,
                bin_upper=hi,
                mean_predicted=float(p[mask].mean()),
                fraction_positive=float(y[mask].mean()),
                count=count,
            )
        )
    return bins


def baseline_always_half(n: int) -> np.ndarray:
    return np.full(n, 0.5)


def baseline_prior_rate(y_train: np.ndarray, n_test: int) -> np.ndarray:
    rate = float(np.asarray(y_train, dtype=float).mean()) if len(y_train) else 0.5
    return np.full(n_test, rate)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    title: str = "Reliability diagram",
    output_path: Path | None = None,
    n_bins: int = DEFAULT_RELIABILITY_BINS,
) -> Path | None:
    """Save a reliability diagram PNG. Returns the path if saved."""
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    xs = [b.mean_predicted for b in bins if b.count > 0 and not np.isnan(b.mean_predicted)]
    ys = [b.fraction_positive for b in bins if b.count > 0 and not np.isnan(b.fraction_positive)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    if xs:
        ax.plot(xs, ys, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def metrics_table(rows: list[tuple[str, MetricReport]]) -> pd.DataFrame:
    """Pretty comparison table for reports."""
    return pd.DataFrame(
        [
            {
                "model": name,
                "brier": r.brier,
                "log_loss": r.log_loss,
                "n": r.n_samples,
                "pos_rate": r.positive_rate,
                "mean_pred": r.mean_prediction,
            }
            for name, r in rows
        ]
    ).set_index("model")
