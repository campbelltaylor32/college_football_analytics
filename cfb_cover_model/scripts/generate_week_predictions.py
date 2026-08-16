#!/usr/bin/env python
"""Replacement for ../Python Scripts/Week_Predictions.ipynb: scores a single week's
CFB_Pred_Week_<N>.csv with the production model chosen in evaluate_models.py, refit one
final time on the *entire* train_pool (all non-holdout, non-excluded-season history,
including the 2025 holdout season - by the time a real future week is being scored, 2025
is no longer "held out," it's simply more training history).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_cover_model.cleaning import prepare_week_frame
from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config, resolve_path
from cfb_cover_model.data import load_week_predictors_df
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.selection import apply_feature_set, fit_feature_set
from cfb_cover_model.modeling.classifiers import build_classifier
from cfb_cover_model.modeling.splits import get_eligible_frame

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
FINAL_SUMMARY_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison" / "final_summary.json"
THRESHOLD_TABLE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
PRED_DIR = Path(__file__).resolve().parents[1] / "outputs" / "predictions"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, help="Week number, matching ../Data/CFB_Pred_Week_<N>.csv")
    parser.add_argument("--model-name", type=str, default=None, help="Override the production model choice from final_summary.json")
    args = parser.parse_args()

    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    final_summary = json.loads(FINAL_SUMMARY_PATH.read_text())
    model_name = args.model_name or final_summary["best_holdout_model"]
    threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH)
    threshold_row = threshold_table.loc[threshold_table["model_name"] == model_name]
    if threshold_row.empty:
        raise ValueError(f"No threshold recorded for model {model_name!r} in {THRESHOLD_TABLE_PATH}")
    threshold = float(threshold_row.iloc[0]["threshold"])

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())

    all_history = get_eligible_frame(frame, data_cfg)  # excludes 2020 only, keeps 2025

    week_csv_path = resolve_path(data_cfg["paths"]["predictors_csv"]).parent / f"CFB_Pred_Week_{args.week}.csv"
    week_df_raw = load_week_predictors_df(week_csv_path)
    week_df, week_feature_columns = prepare_week_frame(week_df_raw, data_cfg)
    assert set(week_feature_columns) == set(feature_columns), (
        "Week file's engineered feature set doesn't match the historical one recorded in "
        "feature_columns.json - re-run scripts/load_and_validate_dataset.py."
    )

    train_variant, train_cols = build_transform_variant(all_history, feature_columns, winning_cfg["transforms"])
    train_variant, train_cols = apply_home_away_representation(train_variant, train_cols, winning_cfg["representation"])
    train_variant = train_variant.reset_index(drop=True)
    train_variant.index = all_history.index
    y_train = all_history["home_covered"]
    season_train = all_history["season"]

    week_variant, week_cols = build_transform_variant(week_df, feature_columns, winning_cfg["transforms"])
    week_variant, week_cols = apply_home_away_representation(week_variant, week_cols, winning_cfg["representation"])
    assert set(week_cols) == set(train_cols), "Week file and training history produced a different feature set."

    from train_models import make_track_a_specs, make_track_b_specs  # noqa: E402

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}

    if model_name not in spec_lookup:
        raise ValueError(
            f"generate_week_predictions.py only supports a single Track A/B model as the "
            f"production model (got {model_name!r}); stacking_ensemble/baselines are not "
            f"supported as a deployable pick here."
        )
    spec = spec_lookup[model_name]

    artifact, _report = fit_feature_set(
        train_variant[train_cols], y_train, season_train, spec["feature_set_mode"], features_cfg, random_state
    )
    X_train = apply_feature_set(train_variant[train_cols], spec["feature_set_mode"], artifact)
    X_week = apply_feature_set(week_variant[week_cols], spec["feature_set_mode"], artifact)

    model = spec["builder"]()
    if spec["kind"] == "classifier":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, all_history["cover_margin"])
    proba = model.predict_proba(X_week)[:, 1]

    out = week_df[["game_id", "home_team", "away_team", "season", "week", "spread"]].copy()
    out["model_name"] = model_name
    out["cover_probability"] = proba
    out["threshold"] = threshold
    out["bet_flag"] = out["cover_probability"] >= threshold
    out = out.sort_values("cover_probability", ascending=False)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"week_{args.week}_predictions.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\n{int(out['bet_flag'].sum())} of {len(out)} games flagged at threshold {threshold}.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
