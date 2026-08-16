"""Evaluation metrics. MAE is the primary metric everywhere (the target is expressed
directly in wins, so MAE is directly interpretable) -- every other metric is secondary."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    return {
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
        "mean_bias": float(np.mean(y_pred - y_true)),
        "pct_within_1": float(np.mean(abs_err <= 1)),
        "pct_within_2": float(np.mean(abs_err <= 2)),
    }


def calibration_by_predicted_bucket(y_true: np.ndarray, y_pred: np.ndarray, bucket_width: int = 2) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    max_bucket = int(np.ceil(df["y_pred"].max() / bucket_width) * bucket_width) + bucket_width
    bins = list(range(0, max_bucket, bucket_width))
    df["bucket"] = pd.cut(df["y_pred"], bins=bins, right=False)
    out = df.groupby("bucket", observed=True).agg(
        n=("y_true", "count"), mean_predicted=("y_pred", "mean"), mean_actual=("y_true", "mean")
    ).reset_index()
    return out


def calibration_by_actual_bucket(y_true: np.ndarray, y_pred: np.ndarray, bucket_width: int = 2) -> pd.DataFrame:
    """Mirror of calibration_by_predicted_bucket, but bucketed on y_true instead of y_pred.

    Bucketing on y_pred (as calibration_by_predicted_bucket does) can look well-calibrated
    even when predictions are severely compressed, because a model that never predicts
    outside a narrow band simply produces buckets that are all near the mean -- the tails
    of the *actual* distribution never get their own bucket. Bucketing on y_true instead
    is what actually surfaces compression: it asks "for teams that truly won 0-2 games (or
    10-12), what did the model predict on average," which a compressed model will answer
    with a mean_predicted far from mean_actual in the extreme buckets.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    max_bucket = int(np.ceil(df["y_true"].max() / bucket_width) * bucket_width) + bucket_width
    bins = list(range(0, max_bucket, bucket_width))
    df["bucket"] = pd.cut(df["y_true"], bins=bins, right=False)
    out = df.groupby("bucket", observed=True).agg(
        n=("y_true", "count"), mean_predicted=("y_pred", "mean"), mean_actual=("y_true", "mean")
    ).reset_index()
    return out


def regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope/intercept of y ~ x via np.polyfit (no statsmodels dependency).

    Direction matters for diagnosing compression and the two are NOT interchangeable:
    - slope of (actual ~ predicted) close to 1 says "as predicted rises, actual rises
      proportionally" -- this can look fine even when predictions are compressed, because
      it only measures the *actual* side's response to whatever narrow range predictions
      occupy.
    - slope of (predicted ~ actual) substantially BELOW 1 is the real signature of
      compression: it says the model's own output moves less than 1-for-1 as the true
      target varies, i.e., predictions are being pulled toward the mean.
    Callers should always compute and label both directions explicitly rather than picking
    one, since either direction alone can be misread.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def std_range_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """n, std/range for actual and predicted, and the std ratio (pred/actual) -- the
    single clearest number for "how compressed are predictions relative to reality."
    A ratio near 1 means predictions spread out as much as actual outcomes do; a ratio
    well below 1 (this model's holdout ratio is ~0.49) is compression, quantified."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    std_actual = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")
    std_pred = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else float("nan")
    return {
        "n": len(y_true),
        "std_actual": std_actual,
        "std_pred": std_pred,
        "std_ratio_pred_over_actual": std_pred / std_actual if std_actual else float("nan"),
        "min_actual": float(np.min(y_true)),
        "max_actual": float(np.max(y_true)),
        "min_pred": float(np.min(y_pred)),
        "max_pred": float(np.max(y_pred)),
        "range_actual": float(np.max(y_true) - np.min(y_true)),
        "range_pred": float(np.max(y_pred) - np.min(y_pred)),
    }


def variance_aware_score(
    y_true: np.ndarray, y_pred: np.ndarray, min_std_ratio: float = 0.85, penalty_weight: float = 1.0
) -> float:
    """MAE, penalized when predictions are more compressed than min_std_ratio allows.

    outputs/diagnostics_compression/tables/regularization_grid_boundary_check.csv showed every
    tuned candidate model pinned at the shrinkage-favoring extreme of its hyperparameter grid
    under plain neg_mean_absolute_error scoring, in every walk-forward fold -- MAE alone has no
    way to penalize a predictor for being uselessly narrow. This adds exactly that penalty:
    zero if std(y_pred)/std(y_true) already clears min_std_ratio, otherwise
    penalty_weight * shortfall * std(y_true) (scaled into wins, the same units as MAE) added to
    the MAE being minimized. Intended for use as a GridSearchCV `scoring` callable via
    make_variance_aware_scorer(), NOT wired into the default candidate-model tuning used by
    scripts/train_models.py/evaluate_models.py -- see scripts/diagnostics/variance_aware_retune.py.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    std_true = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    std_pred = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    ratio = std_pred / std_true if std_true > 0 else 1.0
    shortfall = max(0.0, min_std_ratio - ratio)
    penalty = penalty_weight * shortfall * std_true
    return -(mae + penalty)


def make_variance_aware_scorer(min_std_ratio: float = 0.85, penalty_weight: float = 1.0):
    """sklearn scorer wrapping variance_aware_score, for GridSearchCV's `scoring` param."""
    return make_scorer(
        variance_aware_score, greater_is_better=True, min_std_ratio=min_std_ratio, penalty_weight=penalty_weight
    )


def walk_forward_results(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model_name, fold_validation_season) metrics from a long OOF-predictions frame
    with columns model_name, fold_validation_season, y_true, y_pred. Shared by
    scripts/evaluate_models.py (production model selection) and the diagnostics scripts
    so both compute walk-forward metrics identically."""
    rows = []
    for (model_name, val_season), group in oof_df.groupby(["model_name", "fold_validation_season"]):
        metrics = evaluate_predictions(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        metrics["model_name"] = model_name
        metrics["fold_validation_season"] = val_season
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_by_breakdown(df_with_preds: pd.DataFrame, breakdown_col: str, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    rows = []
    for value, group in df_with_preds.groupby(breakdown_col, observed=True):
        metrics = evaluate_predictions(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        metrics[breakdown_col] = value
        rows.append(metrics)
    return pd.DataFrame(rows)


def out_of_fold_residuals(oof_df: pd.DataFrame, y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> np.ndarray:
    return (oof_df[y_true_col] - oof_df[y_pred_col]).to_numpy()


def prediction_interval_from_residuals(point_prediction: np.ndarray, residuals: np.ndarray, levels: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold residual quantile method: for a new prediction y_hat, the interval is
    [y_hat + quantile(resid, lo), y_hat + quantile(resid, hi)] using the pooled OOF residual
    distribution of the selected model. See docs/modeling_methodology.md for why this method
    was chosen over bootstrapping/conformal prediction."""
    lo_q, hi_q = np.quantile(residuals, levels[0]), np.quantile(residuals, levels[1])
    return point_prediction + lo_q, point_prediction + hi_q
