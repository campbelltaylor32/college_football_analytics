#!/usr/bin/env python
"""Production weekly prediction script: scores a week's games with BOTH
logistic_regression (Track A classifier) and xgboost_regressor (Track B regression-to-
probability), and flags a bet only when they agree - both cross their own
walk-forward-tuned threshold. This is the recommended approach documented in
docs/final_writeup_2026.md: neither model alone is trusted enough to act on solo, but the
combination of two structurally different models (linear classifier vs. tree-ensemble
regressor) agreeing is the most defensible signal this project produced - see
outputs/model_comparison/model_agreement_combinations.csv for why this pair specifically.

Both models are refit one final time on *all* eligible history (everything except the
excluded 2020 season - by the time a real future week is being scored, every prior season,
including whichever ones served as "holdout" during model development, is just training
history now).

Usage:
    python scripts/generate_weekly_predictions.py --week 3
        (reads ../Data/CFB_Pred_Week_3.csv - the original, R-pipeline-dependent path)
    python scripts/generate_weekly_predictions.py --file ../Data/CFB_Pred_2026_Week_3.csv
        (explicit path override, for whatever the 2026 weekly-update naming convention turns
        out to be - this script does not assume the file name, only its column schema, which
        must match ../Data/CFB_Gambling_Predictors_Final_PBP.csv's.)
    python scripts/generate_weekly_predictions.py --live --season 2026 --week 3
        (no R dependency at all: pulls schedule/spread/team form directly from the CFBD API
        via src/cfb_cover_model/ingest/pipeline.py::build_current_week_rows - see
        docs/api_ingestion.md for what's been validated about this path and what's a known
        approximation, in particular EPA-derived features and long-tenured-coach records)
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
from cfb_cover_model.modeling.splits import get_eligible_frame

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
EXTENDED_HISTORY_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "extended_history.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
THRESHOLD_TABLE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
PRED_DIR = Path(__file__).resolve().parents[1] / "outputs" / "predictions"

CLASSIFIER_MODEL = "logistic_regression"
REGRESSOR_MODEL = "xgboost_regressor"


def resolve_week_file(args, data_cfg: dict) -> Path:
    if args.file:
        return Path(args.file).resolve()
    if args.week is None:
        raise ValueError("Pass either --week <N> or --file <path>.")
    return resolve_path(data_cfg["paths"]["predictors_csv"]).parent / f"CFB_Pred_Week_{args.week}.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None, help="Week number, matching ../Data/CFB_Pred_Week_<N>.csv")
    parser.add_argument("--file", type=str, default=None, help="Explicit path to the week's prediction CSV, overriding --week's naming assumption")
    parser.add_argument("--live", action="store_true", help="Pull the week's features directly from the CFBD API instead of reading a CSV - see ingest/pipeline.py::build_current_week_rows")
    parser.add_argument("--season", type=int, default=None, help="Season year, required with --live")
    args = parser.parse_args()
    if args.live and args.season is None:
        parser.error("--live requires --season")
    if args.live and args.week is None:
        parser.error("--live requires --week")

    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH).set_index("model_name")["threshold"].to_dict()
    for name in (CLASSIFIER_MODEL, REGRESSOR_MODEL):
        if name not in threshold_table:
            raise ValueError(f"No walk-forward-tuned threshold recorded for {name!r} in {THRESHOLD_TABLE_PATH}")

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())

    all_history = get_eligible_frame(frame, data_cfg)  # excludes 2020 only - everything else is training history
    if EXTENDED_HISTORY_PATH.exists():
        # Newly-completed games appended by scripts/ingest_and_update_history.py - additive
        # to the original R-sourced modeling_dataset.parquet, not a replacement for it.
        extended = pd.read_parquet(EXTENDED_HISTORY_PATH)
        extended = get_eligible_frame(extended, data_cfg)
        missing = [c for c in feature_columns if c not in extended.columns]
        if missing:
            raise ValueError(
                f"extended_history.parquet is missing {len(missing)} feature column(s) that "
                f"modeling_dataset.parquet has ({missing[:5]}...) - the live ingest pipeline's "
                "column schema has drifted from the trained feature set; re-run "
                "scripts/validate_against_r_pipeline.py before trusting this history for training."
            )
        keep_cols = sorted(
            set(feature_columns) | {"game_id", "season", "week", "home_team", "away_team", "home_covered", "cover_margin", "home_favored"}
        )
        all_history = pd.concat([all_history[keep_cols], extended[keep_cols]], ignore_index=True)

    if args.live:
        from cfb_cover_model.ingest import cfbd_client, pipeline

        print(f"Pulling season={args.season} week={args.week} live from the CFBD API")
        client = cfbd_client.get_client()
        week_df_raw = pipeline.build_current_week_rows(client, args.season, args.week)
        week_path = f"<live season={args.season} week={args.week}>"
    else:
        week_path = resolve_week_file(args, data_cfg)
        print(f"Scoring {week_path}")
        week_df_raw = load_week_predictors_df(week_path)
    week_df, week_feature_columns = prepare_week_frame(week_df_raw, data_cfg)
    assert set(week_feature_columns) == set(feature_columns), (
        "Week file's engineered feature set doesn't match the historical one recorded in "
        "feature_columns.json - engineered_features.py may have changed since the last "
        "load_and_validate_dataset.py run; re-run that stage before scoring a new week."
    )

    train_variant, train_cols = build_transform_variant(all_history, feature_columns, winning_cfg["transforms"])
    train_variant, train_cols = apply_home_away_representation(train_variant, train_cols, winning_cfg["representation"])
    train_variant = train_variant.reset_index(drop=True)
    train_variant.index = all_history.index
    y_train = all_history["home_covered"]
    cover_margin_train = all_history["cover_margin"]
    season_train = all_history["season"]

    week_variant, week_cols = build_transform_variant(week_df, feature_columns, winning_cfg["transforms"])
    week_variant, week_cols = apply_home_away_representation(week_variant, week_cols, winning_cfg["representation"])
    assert set(week_cols) == set(train_cols), (
        "Week file and training history produced a different feature set - the week CSV's "
        "column schema no longer matches CFB_Gambling_Predictors_Final_PBP.csv's."
    )

    from train_models import make_track_a_specs, make_track_b_specs  # noqa: E402

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}
    classifier_spec = spec_lookup[CLASSIFIER_MODEL]
    regressor_spec = spec_lookup[REGRESSOR_MODEL]

    # Both models use feature_set_mode="reduced" (config/modeling.yaml) - fit the reducer
    # once and reuse for both, exactly as they were compared in
    # scripts/analyze_model_agreement.py, so this isn't a different feature set than what
    # was actually evaluated.
    assert classifier_spec["feature_set_mode"] == regressor_spec["feature_set_mode"] == "reduced"
    artifact, _report = fit_feature_set(
        train_variant[train_cols], y_train, season_train, "reduced", features_cfg, random_state
    )
    X_train = apply_feature_set(train_variant[train_cols], "reduced", artifact)
    X_week = apply_feature_set(week_variant[week_cols], "reduced", artifact)

    classifier = classifier_spec["builder"]()
    classifier.fit(X_train, y_train)
    classifier_proba = classifier.predict_proba(X_week)[:, 1]

    regressor = regressor_spec["builder"]()
    regressor.fit(X_train, cover_margin_train)
    regressor_proba = regressor.predict_proba(X_week)[:, 1]

    classifier_threshold = threshold_table[CLASSIFIER_MODEL]
    regressor_threshold = threshold_table[REGRESSOR_MODEL]

    out = week_df[["game_id", "home_team", "away_team", "season", "week", "spread"]].copy()
    out[f"{CLASSIFIER_MODEL}_probability"] = classifier_proba
    out[f"{CLASSIFIER_MODEL}_flag"] = classifier_proba >= classifier_threshold
    out[f"{REGRESSOR_MODEL}_probability"] = regressor_proba
    out[f"{REGRESSOR_MODEL}_flag"] = regressor_proba >= regressor_threshold
    out["agreement_bet"] = out[f"{CLASSIFIER_MODEL}_flag"] & out[f"{REGRESSOR_MODEL}_flag"]
    out["avg_probability"] = (classifier_proba + regressor_proba) / 2
    out = out.sort_values(["agreement_bet", "avg_probability"], ascending=[False, False])

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if args.live:
        label = f"live_{args.season}_week_{args.week}"
    elif args.week is not None:
        label = f"week_{args.week}"
    else:
        label = Path(week_path).stem
    out_path = PRED_DIR / f"{label}_dual_model_predictions.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    n_agree = int(out["agreement_bet"].sum())
    n_classifier = int(out[f"{CLASSIFIER_MODEL}_flag"].sum())
    n_regressor = int(out[f"{REGRESSOR_MODEL}_flag"].sum())
    print(f"\n{CLASSIFIER_MODEL} flagged {n_classifier}/{len(out)} at threshold {classifier_threshold}")
    print(f"{REGRESSOR_MODEL} flagged {n_regressor}/{len(out)} at threshold {regressor_threshold}")
    print(f"BOTH agree (recommended bets): {n_agree}/{len(out)}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
