"""Classification metrics shared by model comparison and the per-team predictability scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def binary_entropy(rate: float | np.ndarray) -> float | np.ndarray:
    r = np.clip(rate, EPS, 1 - EPS)
    return -(r * np.log(r) + (1 - r) * np.log(1 - r))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.clip(np.asarray(p), EPS, 1 - EPS)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float(((p >= 0.5) == y).mean()),
        "ece": expected_calibration_error(y, p),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return df.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), pass_rate=("y", "mean")).reset_index()
