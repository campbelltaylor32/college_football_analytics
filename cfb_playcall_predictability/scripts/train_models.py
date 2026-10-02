"""Walk-forward model comparison, 2025 holdout, and the final production fit.

Writes:
  outputs/model_comparison/fold_metrics.csv        every model x fold
  outputs/model_comparison/summary.csv             mean walk-forward + holdout metrics per model
  outputs/model_comparison/calibration_holdout.csv best model, 2025 holdout
  outputs/model_comparison/oos_predictions.parquet out-of-sample P(pass) for every validation /
                                                   holdout play (all models) -- feeds the
                                                   year-over-year stability check
  outputs/models/best_model_<date>.joblib + selected_features_<date>.json + model_metadata_<date>.json

Usage: python scripts/train_models.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from cfb_playcall_predictability.config import load_modeling_config
from cfb_playcall_predictability.dataset import TARGET, feature_columns
from cfb_playcall_predictability.modeling.evaluation import calibration_table, classification_metrics
from cfb_playcall_predictability.modeling.models import build_model
from cfb_playcall_predictability.modeling.splits import final_holdout_fold, generate_walk_forward_folds
from cfb_playcall_predictability.utils.logging import get_logger
from cfb_playcall_predictability.utils.paths import (
    DATA_PROCESSED_DIR, OUTPUTS_MODEL_COMPARISON, OUTPUTS_MODELS, ensure_output_dirs,
)

logger = get_logger("train_models")
PRED_ID_COLUMNS = ["season", "week", "game_id", "play_seq", "pos_team", TARGET]


def main() -> None:
    cfg = load_modeling_config()
    ensure_output_dirs()
    df = pd.read_parquet(DATA_PROCESSED_DIR / "modeling_dataset.parquet")
    df = df[df["season"] >= cfg.full_feature_start_season].reset_index(drop=True)
    features = feature_columns(df)
    model_names = cfg.baseline_models + cfg.candidate_models
    logger.info(f"{len(df):,} calls, {len(features)} features, models: {model_names}")

    folds = [(f, "walk_forward") for f in generate_walk_forward_folds(cfg)]
    folds.append((final_holdout_fold(cfg), "holdout"))

    metric_rows, pred_frames = [], []
    for fold, kind in folds:
        train = df[df["season"].isin(fold.train_seasons)]
        val = df[df["season"] == fold.validation_season]
        preds = val[PRED_ID_COLUMNS].copy()
        for name in model_names:
            t0 = time.time()
            model = build_model(name, features, cfg.hyperparams, cfg.random_seed)
            model.fit(train, train[TARGET].to_numpy())
            p = model.predict_proba(val)[:, 1]
            preds[f"p_{name}"] = p
            m = classification_metrics(val[TARGET].to_numpy(), p)
            metric_rows.append({"model": name, "kind": kind, "validation_season": fold.validation_season, **m})
            logger.info(f"{kind} {fold.validation_season} {name:18s} log_loss={m['log_loss']:.4f} "
                        f"auc={m['auc']:.4f} acc={m['accuracy']:.4f} ece={m['ece']:.4f} ({time.time() - t0:.0f}s)")
        pred_frames.append(preds)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUTS_MODEL_COMPARISON / "fold_metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(
        OUTPUTS_MODEL_COMPARISON / "oos_predictions.parquet", index=False
    )

    cols = ["log_loss", "brier", "auc", "accuracy", "ece"]
    summary = (
        metrics.groupby(["model", "kind"])[cols].mean().unstack("kind")
        .sort_values(("log_loss", "walk_forward"))
    )
    summary.columns = [f"{kind}_{metric}" for metric, kind in summary.columns]
    summary.to_csv(OUTPUTS_MODEL_COMPARISON / "summary.csv")
    logger.info(f"Model comparison:\n{summary.round(4).to_string()}")

    best = summary.loc[summary.index.isin(cfg.candidate_models), "walk_forward_log_loss"].idxmin()
    logger.info(f"Best candidate by mean walk-forward log loss: {best}")

    holdout = pd.concat(pred_frames)
    holdout = holdout[holdout["season"] == cfg.final_holdout_season]
    calibration_table(holdout[TARGET].to_numpy(), holdout[f"p_{best}"].to_numpy()).to_csv(
        OUTPUTS_MODEL_COMPARISON / "calibration_holdout.csv", index=False
    )

    # Production fit: every season through the holdout season.
    final_train = df[df["season"] <= cfg.final_holdout_season]
    model = build_model(best, features, cfg.hyperparams, cfg.random_seed)
    model.fit(final_train, final_train[TARGET].to_numpy())

    stamp = date.today().isoformat()
    joblib.dump(model, OUTPUTS_MODELS / f"best_model_{stamp}.joblib")
    with open(OUTPUTS_MODELS / f"selected_features_{stamp}.json", "w") as f:
        json.dump(features, f, indent=2)
    with open(OUTPUTS_MODELS / f"model_metadata_{stamp}.json", "w") as f:
        json.dump({
            "model": best,
            "trained_on_seasons": sorted(final_train["season"].unique().tolist()),
            "n_train_calls": int(len(final_train)),
            "walk_forward_log_loss": float(summary.loc[best, "walk_forward_log_loss"]),
            "holdout_log_loss": float(summary.loc[best, "holdout_log_loss"]),
            "holdout_auc": float(summary.loc[best, "holdout_auc"]),
            "hyperparams": cfg.hyperparams.get(best, {}),
        }, f, indent=2)
    logger.info(f"Saved production {best} model to {OUTPUTS_MODELS}")


if __name__ == "__main__":
    main()
