"""Model diagnostics: plots + tabular reports for the selected model's final holdout
predictions. All plotting uses matplotlib's non-interactive Agg backend so this module runs
headless in scripts/run_pipeline.py. Ported near-verbatim from the sibling cfb_win_total_model
project's modeling/diagnostics.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)


def plot_actual_vs_predicted(df: pd.DataFrame, path: Path, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[y_true_col], df[y_pred_col], alpha=0.4, s=12)
    lims = [min(df[y_true_col].min(), df[y_pred_col].min()), max(df[y_true_col].max(), df[y_pred_col].max())]
    ax.plot(lims, lims, "r--", label="y = x")
    ax.set_xlabel("Actual rushing yards")
    ax.set_ylabel("Predicted rushing yards")
    ax.set_title("Actual vs. Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(df: pd.DataFrame, by: str | None, path: Path, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> None:
    df = df.copy()
    df["residual"] = df[y_pred_col] - df[y_true_col]
    fig, ax = plt.subplots(figsize=(8, 5))
    if by is None:
        ax.hist(df["residual"], bins=30)
        ax.set_xlabel("Residual (predicted - actual)")
        ax.set_title("Residual distribution")
    else:
        df.boxplot(column="residual", by=by, ax=ax, rot=90)
        ax.set_title(f"Residuals by {by}")
        plt.suptitle("")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(calibration_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(calibration_df["mean_predicted"], calibration_df["mean_actual"], marker="o", label="Observed")
    lims = [calibration_df["mean_predicted"].min(), calibration_df["mean_predicted"].max()]
    ax.plot(lims, lims, "r--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted rushing yards (bucket)")
    ax.set_ylabel("Mean actual rushing yards (bucket)")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def largest_misses(df: pd.DataFrame, n: int = 15, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    df = df.copy()
    df["abs_error"] = (df[y_pred_col] - df[y_true_col]).abs()
    df["signed_error"] = df[y_pred_col] - df[y_true_col]
    return df.sort_values("abs_error", ascending=False).head(n)


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
    else:
        logger.warning(f"Model {type(model).__name__} has neither feature_importances_ nor coef_")
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False, key=abs)


def permutation_importance_report(model, X, y, feature_names: list[str], n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, scoring="neg_mean_absolute_error")
    return pd.DataFrame(
        {"feature": feature_names, "importance_mean": result.importances_mean, "importance_std": result.importances_std}
    ).sort_values("importance_mean", ascending=False)
