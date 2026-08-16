#!/usr/bin/env python
"""Step 10 of the pipeline: produces the final predicted-win-totals CSV for
modeling.yaml's target_season. In this demo build, target_season == final_holdout_season
(2025), so this script formats scripts/evaluate_models.py's already-computed holdout
predictions per the required output schema, applies clipping, and attaches prediction
intervals via the out-of-fold residual quantile method (see docs/modeling_methodology.md).

Usage:
    python scripts/generate_predictions.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_win_total_model.config import load_modeling_config
from cfb_win_total_model.modeling.evaluation import out_of_fold_residuals, prediction_interval_from_residuals
from cfb_win_total_model.utils.logging import get_logger
from cfb_win_total_model.utils.paths import DATA_PROCESSED_DIR, OUTPUTS_MODEL_COMPARISON, OUTPUTS_PREDICTIONS, ensure_dirs

logger = get_logger(__name__)

DATASET_PATH = DATA_PROCESSED_DIR / "modeling_dataset.parquet"
CONTEXT_COLS = {
    "prior_season_wins": "previous_season_wins",
    "talent": "team_talent",
    "returning_percent_ppa": "returning_production",
    "avg_opponent_prior_win_pct": "schedule_strength",
    "coaching_change_indicator": "coaching_change",
    "net_transfer_talent": "transfer_net_talent",
}


def main() -> int:
    ensure_dirs()
    modeling_cfg = load_modeling_config()

    holdout_preds_path = OUTPUTS_MODEL_COMPARISON / "holdout_2025_predictions.csv"
    selection_path = OUTPUTS_MODEL_COMPARISON / "selected_model.json"
    oof_path = OUTPUTS_MODEL_COMPARISON / "oof_predictions.csv"
    if not (holdout_preds_path.exists() and selection_path.exists() and oof_path.exists()):
        raise FileNotFoundError("Run scripts/train_models.py and scripts/evaluate_models.py first")

    selection = json.loads(selection_path.read_text())
    model_name = selection["model_name"]

    preds = pd.read_csv(holdout_preds_path)[["school", "season", "y_pred"]].rename(
        columns={"y_pred": "predicted_wins"}
    )

    df = pd.read_parquet(DATASET_PATH)
    target_rows = df[df["season"] == modeling_cfg.target_season]
    merged = preds.merge(target_rows, on=["school", "season"], how="left")

    merged["predicted_wins"] = merged["predicted_wins"].clip(lower=modeling_cfg.clip_min_wins, upper=merged["scheduled_games"])
    merged["predicted_win_percentage"] = merged["predicted_wins"] / merged["scheduled_games"]

    oof_df = pd.read_csv(oof_path)
    model_oof = oof_df[oof_df["model_name"] == model_name]
    if len(model_oof) >= 2:
        residuals = out_of_fold_residuals(model_oof)
        lo, hi = prediction_interval_from_residuals(merged["predicted_wins"].to_numpy(), residuals, modeling_cfg.prediction_interval_levels)
        merged["prediction_interval_low"] = lo.clip(min=modeling_cfg.clip_min_wins)
        merged["prediction_interval_high"] = hi.clip(max=merged["scheduled_games"])
    else:
        logger.warning(f"Fewer than 2 OOF predictions for model '{model_name}'; skipping prediction intervals")
        merged["prediction_interval_low"] = None
        merged["prediction_interval_high"] = None

    merged["model_name"] = model_name
    merged["model_version"] = f"{model_name}_{date.today().isoformat()}"
    merged["data_cutoff_date"] = date.today().isoformat()
    merged = merged.rename(columns={"school": "team", "season": "season"})

    for src_col, out_col in CONTEXT_COLS.items():
        if src_col not in merged.columns:
            merged[out_col] = None
        elif out_col != src_col:
            merged[out_col] = merged[src_col]

    output_cols = [
        "team", "season", "predicted_wins", "scheduled_games", "predicted_win_percentage",
        "prediction_interval_low", "prediction_interval_high", "model_name", "model_version", "data_cutoff_date",
        "previous_season_wins", "team_talent", "returning_production", "schedule_strength", "coaching_change", "transfer_net_talent",
    ]
    out = merged[output_cols].sort_values("predicted_wins", ascending=False)

    out_path = OUTPUTS_PREDICTIONS / f"predicted_win_totals_{modeling_cfg.target_season}.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out)} predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
