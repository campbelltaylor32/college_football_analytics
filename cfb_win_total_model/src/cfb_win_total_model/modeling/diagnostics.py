"""Model diagnostics: plots + tabular reports for the selected model's final holdout
predictions. All plotting uses matplotlib's non-interactive Agg backend so this module runs
headless in scripts/run_pipeline.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)


def plot_actual_vs_predicted(df: pd.DataFrame, path: Path, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[y_true_col], df[y_pred_col], alpha=0.5)
    lims = [min(df[y_true_col].min(), df[y_pred_col].min()), max(df[y_true_col].max(), df[y_pred_col].max())]
    ax.plot(lims, lims, "r--", label="y = x")
    ax.set_xlabel("Actual wins")
    ax.set_ylabel("Predicted wins")
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
        ax.hist(df["residual"], bins=20)
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
    ax.set_xlabel("Mean predicted wins (bucket)")
    ax.set_ylabel("Mean actual wins (bucket)")
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


def shap_report(model, X: pd.DataFrame):
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP diagnostics (pip install -e '.[shap]')")
        return None
    try:
        explainer = shap.Explainer(model, X)
        return explainer(X)
    except Exception as e:  # SHAP's explainer selection can fail for some model/estimator combos
        logger.warning(f"SHAP explanation failed: {e}")
        return None


def partial_dependence_plots(model, X: pd.DataFrame, features: list[str], path: Path) -> None:
    valid_features = [f for f in features if f in X.columns]
    if not valid_features:
        return
    fig, ax = plt.subplots(figsize=(4 * len(valid_features), 4))
    PartialDependenceDisplay.from_estimator(model, X, valid_features, ax=ax if len(valid_features) > 1 else None)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF implemented inline via numpy.linalg (1/(1-R^2) from an OLS of each column on the
    rest) rather than pulling in statsmodels as a new dependency."""
    X = X.dropna()
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    vifs = []
    for i in range(1, X_with_const.shape[1]):
        y_col = X_with_const[:, i]
        others = np.delete(X_with_const, i, axis=1)
        coefs, *_ = np.linalg.lstsq(others, y_col, rcond=None)
        pred = others @ coefs
        ss_res = np.sum((y_col - pred) ** 2)
        ss_tot = np.sum((y_col - y_col.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vifs.append(vif)
    return pd.DataFrame({"feature": X.columns, "vif": vifs}).sort_values("vif", ascending=False)


def coefficient_stability_across_folds(fold_values: dict[int, np.ndarray], feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(fold_values, index=feature_names)
    df["mean"] = df.mean(axis=1)
    df["std"] = df.drop(columns=["mean"]).std(axis=1)
    return df.reset_index().rename(columns={"index": "feature"})


def learning_curve_by_window(mae_by_window_size: dict[int, float]) -> pd.DataFrame:
    """Takes pre-computed {n_training_seasons: mae} pairs (produced by the caller re-running
    the final holdout fold with training truncated to the most recent N seasons) and returns
    them as a tidy DataFrame for plotting/reporting."""
    return pd.DataFrame(sorted(mae_by_window_size.items()), columns=["n_training_seasons", "mae"])
