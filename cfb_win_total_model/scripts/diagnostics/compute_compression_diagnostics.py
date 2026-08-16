#!/usr/bin/env python
"""Diagnoses win-total prediction compression: predictions cluster ~4-8 wins regardless of
whether a team actually won 0 or 12 games. Reads only already-existing artifacts
(outputs/model_comparison/oof_predictions.csv, holdout_2025_predictions.csv, the saved
per-fold outputs/models/fold_<year>_<model>.joblib pipelines, data/processed/
modeling_dataset.parquet) plus two small read-only DB queries -- no retraining happens here.

Writes everything under outputs/diagnostics_compression/ -- never touches
outputs/model_comparison/, outputs/models/, or outputs/predictions/.

Usage:
    python scripts/diagnostics/compute_compression_diagnostics.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.dataset import NON_FEATURE_COLS
from cfb_win_total_model.database import get_engine, run_query
from cfb_win_total_model.modeling import diagnostics as diag_plots
from cfb_win_total_model.modeling.evaluation import (
    calibration_by_actual_bucket,
    calibration_by_predicted_bucket,
    evaluate_by_breakdown,
    evaluate_predictions,
    regression_slope_intercept,
    std_range_summary,
)
from cfb_win_total_model.modeling.splits import generate_walk_forward_folds
from cfb_win_total_model.modeling.train import TARGET_COL, get_feature_columns, predict_with_pipeline
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_LOGS,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_MODELS,
    ensure_dirs,
)

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The shipped 2025 predictions are a refit of gradient_boosting -- give this a distinct
# model_name so it's never confused with the per-fold OOF "gradient_boosting" rows, which
# come from 5 separately-tuned pipelines, one per walk-forward fold.
HOLDOUT_MODEL_NAME = "gradient_boosting_final_refit_2025"

# Selected model + the two next-most-competitive families (one tree-based one linear) as a
# contrast set for calibration/conference/season breakdowns, rather than all 14 models.
CALIBRATION_CONTRAST_MODELS = ["gradient_boosting", "ridge", "elasticnet"]

_HYPERPARAM_ATTRS = {
    "ridge": ["alpha"],
    "lasso": ["alpha"],
    "elasticnet": ["alpha", "l1_ratio"],
    "random_forest": ["n_estimators", "max_depth"],
    "gradient_boosting": ["n_estimators", "max_depth", "learning_rate"],
    "hist_gradient_boosting": ["max_iter", "max_depth", "learning_rate"],
    "xgboost": ["n_estimators", "max_depth", "learning_rate"],
    "lightgbm": ["n_estimators", "max_depth", "learning_rate"],
}


def _bucket(series: pd.Series, width: int = 2) -> pd.Series:
    max_bucket = int(np.ceil(series.max() / width) * width) + width
    bins = list(range(0, max_bucket, width))
    return pd.cut(series, bins=bins, right=False)


# ---------------------------------------------------------------------------
# 1. std/range comparison
# ---------------------------------------------------------------------------


def build_std_range_table(oof_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in oof_df.groupby("model_name"):
        summary = std_range_summary(group["y_true"], group["y_pred"])
        summary.update(model_name=model_name, split="oof_pooled")
        rows.append(summary)
        for val_season, fold_group in group.groupby("fold_validation_season"):
            fold_summary = std_range_summary(fold_group["y_true"], fold_group["y_pred"])
            fold_summary.update(model_name=model_name, split=f"oof_fold_{val_season}")
            rows.append(fold_summary)
    holdout_summary = std_range_summary(holdout_df["y_true"], holdout_df["y_pred"])
    holdout_summary.update(model_name=HOLDOUT_MODEL_NAME, split="holdout_2025")
    rows.append(holdout_summary)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. slope/intercept, BOTH regression directions -- see the docstring on
#    regression_slope_intercept() for why both are needed, not just one.
# ---------------------------------------------------------------------------


def build_slope_intercept_table(oof_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _row(y_true, y_pred, model_name, split) -> dict:
        slope_a_on_p, intercept_a_on_p = regression_slope_intercept(y_pred, y_true)
        slope_p_on_a, intercept_p_on_a = regression_slope_intercept(y_true, y_pred)
        return {
            "model_name": model_name,
            "split": split,
            "n": len(y_true),
            "slope_actual_on_pred": slope_a_on_p,
            "intercept_actual_on_pred": intercept_a_on_p,
            "slope_pred_on_actual": slope_p_on_a,
            "intercept_pred_on_actual": intercept_p_on_a,
        }

    for model_name, group in oof_df.groupby("model_name"):
        rows.append(_row(group["y_true"], group["y_pred"], model_name, "oof_pooled"))
        for val_season, fold_group in group.groupby("fold_validation_season"):
            rows.append(_row(fold_group["y_true"], fold_group["y_pred"], model_name, f"oof_fold_{val_season}"))

    rows.append(_row(holdout_df["y_true"], holdout_df["y_pred"], HOLDOUT_MODEL_NAME, "holdout_2025"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. calibration, both bucket directions
# ---------------------------------------------------------------------------


def build_calibration_tables(oof_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_rows, pred_rows = [], []
    for model_name in CALIBRATION_CONTRAST_MODELS:
        group = oof_df[oof_df["model_name"] == model_name]
        if group.empty:
            continue
        a = calibration_by_actual_bucket(group["y_true"], group["y_pred"])
        a["model_name"], a["split"] = model_name, "oof_pooled"
        actual_rows.append(a)
        p = calibration_by_predicted_bucket(group["y_true"], group["y_pred"])
        p["model_name"], p["split"] = model_name, "oof_pooled"
        pred_rows.append(p)

    a_holdout = calibration_by_actual_bucket(holdout_df["y_true"], holdout_df["y_pred"])
    a_holdout["model_name"], a_holdout["split"] = HOLDOUT_MODEL_NAME, "holdout_2025"
    actual_rows.append(a_holdout)

    return pd.concat(actual_rows, ignore_index=True), pd.concat(pred_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. conference join (game_team_stats.conference is ~100% populated; modeling_dataset.parquet
#    itself has no conference column) + breakdown
# ---------------------------------------------------------------------------


def fetch_conference_by_school_season(engine) -> pd.DataFrame:
    df = run_query(
        "SELECT school, season, conference FROM game_team_stats WHERE conference IS NOT NULL",
        engine=engine,
    )
    mode_conf = (
        df.groupby(["school", "season"])["conference"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
    )
    return mode_conf


def build_conference_breakdown(oof_df: pd.DataFrame, conference_by_school_season: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in CALIBRATION_CONTRAST_MODELS:
        group = oof_df[oof_df["model_name"] == model_name].merge(
            conference_by_school_season, on=["school", "season"], how="left"
        )
        group = group.dropna(subset=["conference"])
        breakdown = evaluate_by_breakdown(group, "conference")
        breakdown["model_name"] = model_name
        rows.append(breakdown)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. season breakdown, OOF -- spans 2019-2024, unlike the degenerate single-season
#    holdout_2025_breakdown_season.csv that already exists.
# ---------------------------------------------------------------------------


def build_season_breakdown(oof_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in CALIBRATION_CONTRAST_MODELS:
        group = oof_df[oof_df["model_name"] == model_name]
        breakdown = evaluate_by_breakdown(group, "season")
        breakdown["model_name"] = model_name
        rows.append(breakdown)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 6. residual plots by actual-bucket, predicted-bucket, season, conference
# ---------------------------------------------------------------------------


def build_residual_plots(oof_df: pd.DataFrame, holdout_df: pd.DataFrame, conference_by_school_season: pd.DataFrame) -> None:
    gb_oof = oof_df[oof_df["model_name"] == "gradient_boosting"].copy()

    gb_oof["actual_bucket"] = _bucket(gb_oof["y_true"])
    diag_plots.plot_residuals(
        gb_oof, by="actual_bucket", path=OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "residuals_by_actual_bucket_oof.png"
    )

    gb_oof["predicted_bucket"] = _bucket(gb_oof["y_pred"])
    diag_plots.plot_residuals(
        gb_oof, by="predicted_bucket", path=OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "residuals_by_predicted_bucket_oof.png"
    )

    diag_plots.plot_residuals(
        gb_oof, by="fold_validation_season", path=OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "residuals_by_season_oof.png"
    )

    merged = gb_oof.merge(conference_by_school_season, on=["school", "season"], how="left")
    if merged["conference"].notna().any():
        diag_plots.plot_residuals(
            merged, by="conference", path=OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "residuals_by_conference_oof.png"
        )
    else:
        logger.warning("No conference matches found for OOF rows; skipping residuals_by_conference_oof.png")

    holdout_copy = holdout_df.copy()
    holdout_copy["actual_bucket"] = _bucket(holdout_copy["y_true"])
    diag_plots.plot_residuals(
        holdout_copy, by="actual_bucket", path=OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "residuals_by_actual_bucket_holdout.png"
    )


# ---------------------------------------------------------------------------
# Summary bar charts (std comparison, slope comparison, OOF-vs-holdout calibration overlay).
# Reuses the same plain-matplotlib style as modeling/diagnostics.py (no custom palette --
# this is an internal diagnostics artifact, not a user-facing dashboard).
# ---------------------------------------------------------------------------


def build_summary_bar_plots(std_range_df: pd.DataFrame, slope_df: pd.DataFrame, calib_actual_df: pd.DataFrame) -> None:
    groups = [
        ("gradient_boosting", "oof_pooled", "gradient_boosting\n(OOF)"),
        ("ridge", "oof_pooled", "ridge\n(OOF)"),
        ("elasticnet", "oof_pooled", "elasticnet\n(OOF)"),
        (HOLDOUT_MODEL_NAME, "holdout_2025", "gradient_boosting\n(2025 holdout refit)"),
    ]

    # std_actual vs std_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    width = 0.35
    std_actual_vals, std_pred_vals = [], []
    for model_name, split, _ in groups:
        row = std_range_df[(std_range_df.model_name == model_name) & (std_range_df.split == split)].iloc[0]
        std_actual_vals.append(row["std_actual"])
        std_pred_vals.append(row["std_pred"])
    ax.bar(x - width / 2, std_actual_vals, width, label="std(actual)")
    ax.bar(x + width / 2, std_pred_vals, width, label="std(predicted)")
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in groups])
    ax.set_ylabel("Std. dev. of wins")
    ax.set_title("Prediction variance vs. actual variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "std_comparison_bar.png")
    plt.close(fig)

    # slope_pred_on_actual (the compression-diagnostic direction) with a reference line at 1.0
    fig, ax = plt.subplots(figsize=(8, 5))
    slope_vals = []
    for model_name, split, _ in groups:
        row = slope_df[(slope_df.model_name == model_name) & (slope_df.split == split)].iloc[0]
        slope_vals.append(row["slope_pred_on_actual"])
    ax.bar(x, slope_vals)
    ax.axhline(1.0, color="r", linestyle="--", label="slope = 1 (no compression)")
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in groups])
    ax.set_ylabel("Slope of (predicted ~ actual)")
    ax.set_title("Compression: how much does predicted wins move per 1 actual win?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "slope_comparison_bar.png")
    plt.close(fig)

    # calibration-by-actual-bucket overlay: OOF gradient_boosting vs the 2025 holdout refit
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, split, label in [
        ("gradient_boosting", "oof_pooled", "gradient_boosting (OOF)"),
        (HOLDOUT_MODEL_NAME, "holdout_2025", "gradient_boosting (2025 holdout refit)"),
    ]:
        subset = calib_actual_df[(calib_actual_df.model_name == model_name) & (calib_actual_df.split == split)]
        ax.plot(subset["mean_actual"], subset["mean_predicted"], marker="o", label=label)
    lims = [0, max(calib_actual_df["mean_actual"].max(), calib_actual_df["mean_predicted"].max()) + 1]
    ax.plot(lims, lims, "k--", label="perfect calibration (y=x)")
    ax.set_xlabel("Mean actual wins (bucket, bucketed on actual)")
    ax.set_ylabel("Mean predicted wins (bucket)")
    ax.set_title("Calibration by actual-win bucket: OOF vs. 2025 holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS / "calibration_oof_vs_holdout.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. train-vs-validation metrics per fold -- the underfitting/model-bias check. No train-set
#    predictions are computed anywhere in the production pipeline; this reconstructs them by
#    scoring the already-saved fold pipelines against their own training rows.
# ---------------------------------------------------------------------------


def build_train_vs_val_table(modeling_cfg, dataset_df: pd.DataFrame, oof_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(dataset_df, NON_FEATURE_COLS)
    folds = generate_walk_forward_folds(modeling_cfg)
    rows = []
    for fold in folds:
        train_df = dataset_df[dataset_df["season"].isin(fold.train_seasons)]
        for model_name in modeling_cfg.candidate_models:
            joblib_path = OUTPUTS_MODELS / f"fold_{fold.validation_season}_{model_name}.joblib"
            if not joblib_path.exists():
                continue
            pipeline = joblib.load(joblib_path)
            train_preds = predict_with_pipeline(pipeline, train_df, feature_cols)
            train_summary = std_range_summary(train_df[TARGET_COL], train_preds)
            slope_a_on_p, _ = regression_slope_intercept(train_preds, train_df[TARGET_COL])
            slope_p_on_a, _ = regression_slope_intercept(train_df[TARGET_COL], train_preds)
            train_metrics = evaluate_predictions(train_df[TARGET_COL], train_preds)
            rows.append(
                {
                    "model_name": model_name,
                    "fold_validation_season": fold.validation_season,
                    "split": "train",
                    "n": train_summary["n"],
                    "mae": train_metrics["mae"],
                    "std_actual": train_summary["std_actual"],
                    "std_pred": train_summary["std_pred"],
                    "std_ratio_pred_over_actual": train_summary["std_ratio_pred_over_actual"],
                    "slope_actual_on_pred": slope_a_on_p,
                    "slope_pred_on_actual": slope_p_on_a,
                }
            )

            val_group = oof_df[
                (oof_df["model_name"] == model_name) & (oof_df["fold_validation_season"] == fold.validation_season)
            ]
            if val_group.empty:
                continue
            val_summary = std_range_summary(val_group["y_true"], val_group["y_pred"])
            val_slope_a_on_p, _ = regression_slope_intercept(val_group["y_pred"], val_group["y_true"])
            val_slope_p_on_a, _ = regression_slope_intercept(val_group["y_true"], val_group["y_pred"])
            val_metrics = evaluate_predictions(val_group["y_true"], val_group["y_pred"])
            rows.append(
                {
                    "model_name": model_name,
                    "fold_validation_season": fold.validation_season,
                    "split": "val",
                    "n": val_summary["n"],
                    "mae": val_metrics["mae"],
                    "std_actual": val_summary["std_actual"],
                    "std_pred": val_summary["std_pred"],
                    "std_ratio_pred_over_actual": val_summary["std_ratio_pred_over_actual"],
                    "slope_actual_on_pred": val_slope_a_on_p,
                    "slope_pred_on_actual": val_slope_p_on_a,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. regularization / hyperparameter grid boundary check
# ---------------------------------------------------------------------------


def _extract_hyperparam_rows(pipeline, model_name: str, fold_label, grid: dict) -> list[dict]:
    estimator = pipeline.named_steps["model"]
    rows = []
    for attr in _HYPERPARAM_ATTRS.get(model_name, []):
        if not hasattr(estimator, attr):
            continue
        value = getattr(estimator, attr)
        grid_values = grid.get(attr, [])
        numeric_grid = [v for v in grid_values if isinstance(v, (int, float))]
        # None (sklearn "no max depth") and -1 (lightgbm "no max depth") are unbounded
        # sentinels, not extremes of a numeric scale -- flag them separately rather than
        # letting them silently sort as the numeric min, which would mislabel the LEAST
        # regularized option as "pinned at grid min".
        is_sentinel = value is None or value == -1
        position = "n/a"
        if is_sentinel:
            position = "unbounded_sentinel (None/-1 = no max depth, i.e. LEAST regularized)"
        elif isinstance(value, (int, float)) and numeric_grid:
            if value == min(numeric_grid):
                position = "pinned_at_grid_min"
            elif value == max(numeric_grid):
                position = "pinned_at_grid_max"
            else:
                position = "interior"
        rows.append(
            {
                "model_name": model_name,
                "fold_validation_season": fold_label,
                "hyperparam": attr,
                "selected_value": value,
                "grid_values": str(grid_values),
                "position_in_grid": position,
            }
        )
    return rows


def build_regularization_check_table(modeling_cfg) -> pd.DataFrame:
    folds = generate_walk_forward_folds(modeling_cfg)
    rows = []
    for fold in folds:
        for model_name in _HYPERPARAM_ATTRS:
            joblib_path = OUTPUTS_MODELS / f"fold_{fold.validation_season}_{model_name}.joblib"
            if not joblib_path.exists():
                continue
            pipeline = joblib.load(joblib_path)
            grid = modeling_cfg.hyperparam_grids.get(model_name, {})
            rows.extend(_extract_hyperparam_rows(pipeline, model_name, fold.validation_season, grid))

    final_path = OUTPUTS_MODELS / "final_model.joblib"
    selected_path = OUTPUTS_MODEL_COMPARISON / "selected_model.json"
    if final_path.exists() and selected_path.exists():
        model_name = json.loads(selected_path.read_text())["model_name"]
        if model_name in _HYPERPARAM_ATTRS:
            pipeline = joblib.load(final_path)
            grid = modeling_cfg.hyperparam_grids.get(model_name, {})
            rows.extend(_extract_hyperparam_rows(pipeline, model_name, "final_refit_2025", grid))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. n_games endogeneity check -- n_games is the single most important feature (33.7%
#    permutation importance); this checks whether it's contaminated by the very outcome
#    it's supposedly predicting (conference-championship/bowl eligibility requires already
#    being good, so "games played" for season t is not fully preseason-knowable).
# ---------------------------------------------------------------------------


def build_n_games_endogeneity_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if "n_games" not in dataset_df.columns:
        logger.warning("n_games not in modeling dataset; skipping endogeneity check")
        return pd.DataFrame()
    rows = []
    for season, group in dataset_df.groupby("season"):
        for n_games, subgroup in group.groupby("n_games"):
            rows.append(
                {
                    "season": season,
                    "n_games": n_games,
                    "n_teams": len(subgroup),
                    "mean_wins": subgroup[TARGET_COL].mean(),
                }
            )
    table = pd.DataFrame(rows)
    overall_corr = dataset_df[["n_games", TARGET_COL]].corr().iloc[0, 1]
    logger.info(f"Correlation(n_games, {TARGET_COL}) across all seasons: {overall_corr:.3f}")
    return table


# ---------------------------------------------------------------------------
# Sanity checks -- reproduce the numbers already hand-verified against the live CSVs/joblib
# files before trusting anything this script computes. A mismatch means a bug in the new
# evaluation functions, not new information.
# ---------------------------------------------------------------------------


def _sanity_checks(std_range_df: pd.DataFrame, slope_df: pd.DataFrame, reg_df: pd.DataFrame) -> None:
    def _get(df: pd.DataFrame, **filters) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for k, v in filters.items():
            mask &= df[k] == v
        return df[mask]

    holdout_std = _get(std_range_df, model_name=HOLDOUT_MODEL_NAME, split="holdout_2025")
    if not holdout_std.empty:
        row = holdout_std.iloc[0]
        assert abs(row["std_actual"] - 2.73) < 0.05, f"holdout std_actual mismatch: {row['std_actual']}"
        assert abs(row["std_pred"] - 1.34) < 0.05, f"holdout std_pred mismatch: {row['std_pred']}"
        logger.info("Sanity check PASSED: holdout std_actual/std_pred match hand-verified values (2.73 / 1.34)")

    oof_gb_pooled = _get(std_range_df, model_name="gradient_boosting", split="oof_pooled")
    if not oof_gb_pooled.empty:
        row = oof_gb_pooled.iloc[0]
        assert abs(row["std_actual"] - 2.77) < 0.05, f"OOF std_actual mismatch: {row['std_actual']}"
        assert abs(row["std_pred"] - 2.05) < 0.05, f"OOF std_pred mismatch: {row['std_pred']}"
        assert row["n"] == 658, f"OOF gradient_boosting row count mismatch: {row['n']}"
        logger.info("Sanity check PASSED: OOF pooled gradient_boosting std/n match hand-verified values (2.77 / 2.05 / 658)")

    oof_gb_slope = _get(slope_df, model_name="gradient_boosting", split="oof_pooled")
    if not oof_gb_slope.empty:
        slope = oof_gb_slope.iloc[0]["slope_actual_on_pred"]
        assert abs(slope - 0.977) < 0.02, f"OOF slope_actual_on_pred mismatch: {slope}"
        logger.info("Sanity check PASSED: OOF pooled gradient_boosting slope_actual_on_pred matches hand-verified value (0.977)")

    holdout_slope = _get(slope_df, model_name=HOLDOUT_MODEL_NAME, split="holdout_2025")
    if not holdout_slope.empty:
        slope = holdout_slope.iloc[0]["slope_pred_on_actual"]
        assert abs(slope - 0.23) < 0.03, f"holdout slope_pred_on_actual mismatch: {slope}"
        logger.info("Sanity check PASSED: holdout slope_pred_on_actual matches hand-verified value (~0.23)")

    if not reg_df.empty:
        ridge_rows = reg_df[
            (reg_df["model_name"] == "ridge")
            & (reg_df["hyperparam"] == "alpha")
            & (reg_df["fold_validation_season"] != "final_refit_2025")
        ]
        if not ridge_rows.empty:
            assert (ridge_rows["position_in_grid"] == "pinned_at_grid_max").all(), (
                "ridge alpha not pinned at grid max in all folds as hand-verified"
            )
            logger.info("Sanity check PASSED: ridge alpha pinned at grid max (100.0) in all 5 walk-forward folds")

        gb_depth_rows = reg_df[
            (reg_df["model_name"] == "gradient_boosting")
            & (reg_df["hyperparam"] == "max_depth")
            & (reg_df["fold_validation_season"] != "final_refit_2025")
        ]
        if not gb_depth_rows.empty:
            assert (gb_depth_rows["position_in_grid"] == "pinned_at_grid_min").all(), (
                "gradient_boosting max_depth not pinned at grid min in all folds as hand-verified"
            )
            logger.info("Sanity check PASSED: gradient_boosting max_depth pinned at grid min (2) in all 5 walk-forward folds")


def main() -> int:
    ensure_dirs()
    logger.info("Loading OOF and holdout predictions, modeling dataset, and config...")
    oof_df = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv")
    holdout_df = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "holdout_2025_predictions.csv")
    dataset_df = pd.read_parquet(DATASET_PATH)
    modeling_cfg = load_modeling_config()
    engine = get_engine()

    logger.info("[1/9] std/range comparison")
    std_range_df = build_std_range_table(oof_df, holdout_df)
    std_range_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "std_range_comparison.csv", index=False)

    logger.info("[2/9] slope/intercept, both regression directions")
    slope_df = build_slope_intercept_table(oof_df, holdout_df)
    slope_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "slope_intercept_by_model_fold.csv", index=False)

    logger.info("[3/9] calibration, both bucket directions")
    calib_actual_df, calib_pred_df = build_calibration_tables(oof_df, holdout_df)
    calib_actual_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "calibration_by_actual_bucket.csv", index=False)
    calib_pred_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "calibration_by_predicted_bucket_oof.csv", index=False)

    logger.info("[4/9] conference join + breakdown")
    conference_by_school_season = fetch_conference_by_school_season(engine)
    conference_breakdown = build_conference_breakdown(oof_df, conference_by_school_season)
    conference_breakdown.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "conference_breakdown_oof.csv", index=False)

    logger.info("[5/9] season breakdown (OOF, multi-season)")
    season_breakdown = build_season_breakdown(oof_df)
    season_breakdown.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "season_breakdown_oof.csv", index=False)

    logger.info("[6/9] residual plots + summary bar charts")
    build_residual_plots(oof_df, holdout_df, conference_by_school_season)
    build_summary_bar_plots(std_range_df, slope_df, calib_actual_df)

    logger.info("[7/9] train-vs-validation metrics by fold (underfitting check)")
    train_vs_val_df = build_train_vs_val_table(modeling_cfg, dataset_df, oof_df)
    train_vs_val_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "train_vs_val_metrics_by_fold.csv", index=False)

    logger.info("[8/9] regularization grid boundary check")
    reg_df = build_regularization_check_table(modeling_cfg)
    reg_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "regularization_grid_boundary_check.csv", index=False)

    logger.info("[9/9] n_games endogeneity check")
    n_games_df = build_n_games_endogeneity_table(dataset_df)
    n_games_df.to_csv(OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES / "n_games_endogeneity_check.csv", index=False)

    logger.info("Running sanity checks against hand-verified numbers...")
    _sanity_checks(std_range_df, slope_df, reg_df)

    git_sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    manifest = {
        "script": "compute_compression_diagnostics.py",
        "git_sha": git_sha_result.stdout.strip() or "unknown",
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "inputs": {
            str(p): p.stat().st_mtime
            for p in [
                OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv",
                OUTPUTS_MODEL_COMPARISON / "holdout_2025_predictions.csv",
                DATASET_PATH,
            ]
        },
    }
    (OUTPUTS_DIAGNOSTICS_COMPRESSION_LOGS / "compute_compression_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    logger.info("Done. Outputs written to outputs/diagnostics_compression/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
