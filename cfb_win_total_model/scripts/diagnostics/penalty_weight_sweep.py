#!/usr/bin/env python
"""Follow-up to variance_aware_retune.py: that script tested exactly one penalty_weight (1.0),
which moved gradient_boosting's std_ratio from 0.738 to 0.845 (not quite the 0.85 target used
in the penalty) at an 8.9% MAE cost. This sweeps penalty_weight across a range spanning both
cheaper (weaker penalty) and stronger options to trace out the actual MAE-vs-compression
tradeoff curve, for the same 3-model contrast set and the same widened grids
(reused from variance_aware_retune.py, not redefined here).

EVALUATE-ONLY: nothing here touches outputs/model_comparison/selected_model.json,
outputs/models/final_model.joblib, or outputs/predictions/predicted_win_totals_2025.csv. All
new artifacts go under outputs/diagnostics_compression/experiments/.

Usage:
    python scripts/diagnostics/penalty_weight_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.dataset import NON_FEATURE_COLS
from cfb_win_total_model.modeling.evaluation import (
    make_variance_aware_scorer,
    regression_slope_intercept,
    std_range_summary,
    walk_forward_results,
)
from cfb_win_total_model.modeling.models import get_candidate_models
from cfb_win_total_model.modeling.splits import generate_walk_forward_folds
from cfb_win_total_model.modeling.train import fit_candidate_on_fold, get_feature_columns, predict_with_pipeline
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS,
    OUTPUTS_MODEL_COMPARISON,
    ensure_dirs,
)
from variance_aware_retune import TARGET_MODELS, WIDENED_GRIDS, _oof_frame

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
MIN_STD_RATIO = 0.85
# Spans both cheaper (weaker penalty, closer to plain MAE) and stronger (beyond the
# already-tested 1.0) options to trace the MAE-vs-compression tradeoff curve.
PENALTY_WEIGHTS = [0.1, 0.25, 0.5, 1.0, 2.0]


def run_sweep(df: pd.DataFrame, modeling_cfg) -> pd.DataFrame:
    feature_cols = get_feature_columns(df, NON_FEATURE_COLS)
    folds = generate_walk_forward_folds(modeling_cfg)
    oof_rows = []
    for fold in folds:
        train_df = df[df["season"].isin(fold.train_seasons)]
        val_df = df[df["season"] == fold.validation_season]
        for model_name in TARGET_MODELS:
            for weight in PENALTY_WEIGHTS:
                scorer = make_variance_aware_scorer(min_std_ratio=MIN_STD_RATIO, penalty_weight=weight)
                estimator = get_candidate_models([model_name], modeling_cfg.random_seed)[model_name]
                pipeline = fit_candidate_on_fold(estimator, WIDENED_GRIDS[model_name], train_df, feature_cols, scoring=scorer)
                preds = predict_with_pipeline(pipeline, val_df, feature_cols)
                full_model_name = f"{model_name}__pw_{weight}"
                oof_rows.append(_oof_frame(val_df, preds, full_model_name, fold.validation_season))
                logger.info(f"Fold {fold.validation_season}, {model_name}, penalty_weight={weight}: done")
    return pd.concat(oof_rows, ignore_index=True)


def build_sweep_table(baseline_oof: pd.DataFrame, sweep_oof: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([baseline_oof, sweep_oof], ignore_index=True)
    wf = walk_forward_results(combined)
    rows = []
    for model_name in TARGET_MODELS:
        variant_names = [model_name] + [f"{model_name}__pw_{w}" for w in PENALTY_WEIGHTS]
        for name in variant_names:
            model_oof = combined[combined["model_name"] == name]
            if model_oof.empty:
                continue
            model_wf = wf[wf["model_name"] == name]
            summary = std_range_summary(model_oof["y_true"], model_oof["y_pred"])
            slope_a_on_p, _ = regression_slope_intercept(model_oof["y_pred"], model_oof["y_true"])
            slope_p_on_a, _ = regression_slope_intercept(model_oof["y_true"], model_oof["y_pred"])
            is_baseline = name == model_name
            rows.append(
                {
                    "base_model": model_name,
                    "model_name": name,
                    "penalty_weight": None if is_baseline else float(name.rsplit("_", 1)[1]),
                    "variant": "baseline (production, MAE-only)" if is_baseline else f"variance_aware pw={name.rsplit('_', 1)[1]}",
                    "mean_fold_mae": model_wf["mae"].mean(),
                    "std_fold_mae": model_wf["mae"].std(),
                    "n_oof": summary["n"],
                    "std_actual": summary["std_actual"],
                    "std_pred": summary["std_pred"],
                    "std_ratio_pred_over_actual": summary["std_ratio_pred_over_actual"],
                    "slope_actual_on_pred": slope_a_on_p,
                    "slope_pred_on_actual": slope_p_on_a,
                }
            )
    out = pd.DataFrame(rows)
    out["mae_pct_change_vs_baseline"] = out.groupby("base_model")["mean_fold_mae"].transform(lambda s: (s / s.iloc[0] - 1) * 100)
    return out


def main() -> int:
    ensure_dirs()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"{DATASET_PATH} not found -- run scripts/build_modeling_dataset.py first")

    modeling_cfg = load_modeling_config()
    df = pd.read_parquet(DATASET_PATH)

    logger.info(f"Sweeping penalty_weight={PENALTY_WEIGHTS} x {TARGET_MODELS} (evaluate-only, no promotion)...")
    sweep_oof = run_sweep(df, modeling_cfg)

    sweep_oof_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "penalty_weight_sweep_oof_predictions.csv"
    sweep_oof.to_csv(sweep_oof_path, index=False)
    logger.info(f"Wrote {len(sweep_oof)} sweep OOF predictions -> {sweep_oof_path}")

    baseline_oof = pd.read_csv(OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv")
    baseline_oof = baseline_oof[baseline_oof["model_name"].isin(TARGET_MODELS)]

    for model_name in TARGET_MODELS:
        n_baseline = len(baseline_oof[baseline_oof["model_name"] == model_name])
        for weight in PENALTY_WEIGHTS:
            n_variant = len(sweep_oof[sweep_oof["model_name"] == f"{model_name}__pw_{weight}"])
            assert n_baseline == n_variant, (
                f"{model_name}__pw_{weight}: row count ({n_variant}) != baseline ({n_baseline})"
            )
    logger.info("Sanity check PASSED: every sweep variant's OOF row count matches the production baseline")

    table = build_sweep_table(baseline_oof, sweep_oof)
    table_path = OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS / "penalty_weight_sweep_results.csv"
    table.to_csv(table_path, index=False)
    logger.info(f"Penalty weight sweep results:\n{table.to_string(index=False)}")
    logger.info(f"Wrote sweep results -> {table_path}")
    logger.info(
        "EVALUATE-ONLY: no production artifact (selected_model.json, final_model.joblib, "
        "predicted_win_totals_2025.csv) was modified by this script."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
