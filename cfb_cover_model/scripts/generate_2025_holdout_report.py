#!/usr/bin/env python
"""Refits the production model (walk-forward-selected, per
outputs/threshold_selection/chosen_threshold_per_model.csv) on the full train_pool and
scores every game in the true 2025 holdout individually, with week-level detail - the
per-game granularity evaluate_models.py's aggregate holdout_model_comparison.csv doesn't
retain. Not part of the core pipeline chain (run_pipeline.py) since it's a reporting view
on top of already-computed model choices, not a stage that feeds anything downstream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.modeling.evaluation import precision_at_threshold
from cfb_cover_model.modeling.splits import get_holdout_split
from evaluate_models import build_full_variant
from train_models import get_reduced_features, make_track_a_specs, make_track_b_specs

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
THRESHOLD_TABLE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "predictions"


def main() -> None:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())
    threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH)

    model_name = threshold_table.sort_values("walk_forward_precision", ascending=False).iloc[0]["model_name"]
    threshold = float(threshold_table.loc[threshold_table["model_name"] == model_name, "threshold"].iloc[0])
    print(f"Production model: {model_name!r}, threshold={threshold}")

    full_variant, variant_cols = build_full_variant(frame, feature_columns, winning_cfg)
    train_pool, holdout = get_holdout_split(frame, data_cfg)
    train_idx, holdout_idx = train_pool.index, holdout.index

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}
    spec = spec_lookup[model_name]

    y_tr = full_variant.loc[train_idx, "home_covered"]
    season_tr = full_variant.loc[train_idx, "season"]
    X_tr_full = full_variant.loc[train_idx, variant_cols]
    X_ho_full = full_variant.loc[holdout_idx, variant_cols]

    X_tr, X_ho, _report = get_reduced_features(
        {}, X_tr_full, X_ho_full, y_tr, season_tr, spec["feature_set_mode"], features_cfg, random_state
    )
    model = spec["builder"]()
    if spec["kind"] == "classifier":
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, full_variant.loc[train_idx, "cover_margin"])
    proba = model.predict_proba(X_ho)[:, 1]

    game_report = frame.loc[holdout_idx, ["game_id", "home_team", "away_team", "season", "week", "home_covered"]].copy()
    game_report["cover_probability"] = proba
    game_report["bet_flag"] = game_report["cover_probability"] >= threshold
    game_report["correct"] = (
        (game_report["bet_flag"] & (game_report["home_covered"] == 1))
        | (~game_report["bet_flag"] & (game_report["home_covered"] == 0))
    )
    game_report = game_report.sort_values(["week", "game_id"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    game_path = OUT_DIR / "2025_holdout_predictions.csv"
    game_report.to_csv(game_path, index=False)
    print(f"Wrote {game_path} ({len(game_report)} games)")

    rows = []
    for week, g in game_report.groupby("week"):
        flagged = g[g["bet_flag"]]
        precision, coverage = precision_at_threshold(g["home_covered"].to_numpy(), g["cover_probability"].to_numpy(), threshold)
        accuracy = (g["bet_flag"] == (g["home_covered"] == 1)).mean()  # predict "cover" if flagged, "no cover" otherwise
        rows.append(
            {
                "week": int(week),
                "n_games": len(g),
                "n_flagged": len(flagged),
                "flagged_precision": precision,
                "flagged_coverage": coverage,
                "full_slate_accuracy": accuracy,
                "actual_home_covered_rate": g["home_covered"].mean(),
            }
        )
    weekly = pd.DataFrame(rows)
    weekly_path = OUT_DIR / "2025_holdout_weekly_summary.csv"
    weekly.to_csv(weekly_path, index=False)

    overall_precision, overall_coverage = precision_at_threshold(
        game_report["home_covered"].to_numpy(), game_report["cover_probability"].to_numpy(), threshold
    )
    overall_accuracy = (game_report["bet_flag"] == (game_report["home_covered"] == 1)).mean()

    print(f"\nWrote {weekly_path}")
    pd.set_option("display.width", 140)
    print(weekly.to_string(index=False))
    print(f"\nOverall: {len(game_report)} games, {int(game_report['bet_flag'].sum())} flagged, "
          f"flagged precision={overall_precision:.3f}, flagged coverage={overall_coverage:.3f}, "
          f"full-slate accuracy (bet-flag vs actual)={overall_accuracy:.3f}")


if __name__ == "__main__":
    main()
