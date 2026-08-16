#!/usr/bin/env python
"""Stage 7: threshold selection (on pooled walk-forward OOF only), calibration testing,
and - critically - scoring of every candidate model (not just the walk-forward-selected
winner) on the true 2025 holdout, refit once on the entire train_pool. The holdout is
touched exactly once per model, at a threshold chosen before ever looking at it.
"""
from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.selection import apply_feature_set, fit_feature_set
from cfb_cover_model.modeling.calibration import CALIBRATORS
from cfb_cover_model.modeling.classifiers import (
    AlwaysFavoriteBaseline,
    MajorityClassBaseline,
    build_classifier,
)
from cfb_cover_model.modeling.evaluation import (
    best_precision_at_coverage_floor,
    calibration_report,
    precision_at_threshold,
)
from cfb_cover_model.modeling.regressor import (
    QuantileProbabilityRegressor,
    ResidualProbabilityRegressor,
    build_point_regressor,
)
from cfb_cover_model.modeling.splits import get_holdout_split
from cfb_cover_model.modeling.stacking import fit_stacking_ensemble, safe_inner_min_seasons
from train_models import get_reduced_features, make_track_a_specs, make_track_b_specs  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
OOF_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison" / "oof_predictions.csv"
THRESHOLD_DIR = Path(__file__).resolve().parents[1] / "outputs" / "threshold_selection"
CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "outputs" / "calibration"
MODEL_COMPARISON_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"


def build_full_variant(frame: pd.DataFrame, feature_columns: list[str], winning_cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    variant, variant_cols = build_transform_variant(frame, feature_columns, winning_cfg["transforms"])
    variant, variant_cols = apply_home_away_representation(variant, variant_cols, winning_cfg["representation"])
    variant = variant.reset_index(drop=True)
    variant.index = frame.index
    for col in ("season", "home_covered", "cover_margin", "home_favored", "game_id"):
        variant[col] = frame[col]
    return variant, variant_cols


def select_thresholds(oof_df: pd.DataFrame, modeling_cfg: dict) -> pd.DataFrame:
    rows = []
    grid = modeling_cfg["evaluation"]["threshold_grid"]
    floor = modeling_cfg["evaluation"]["min_coverage_floor"]
    for model_name, g in oof_df.groupby("model_name"):
        y_true, y_proba = g["y_true"].to_numpy(), g["y_proba"].to_numpy()
        chosen = best_precision_at_coverage_floor(y_true, y_proba, grid, floor)
        cal = calibration_report(y_true, y_proba)
        rows.append(
            {
                "model_name": model_name,
                "n_walk_forward_predictions": len(g),
                "n_walk_forward_folds": g["val_season"].nunique(),
                "threshold": chosen["threshold"],
                "walk_forward_precision": chosen["precision"],
                "walk_forward_coverage": chosen["coverage"],
                "met_coverage_floor": chosen["met_floor"],
                "walk_forward_roc_auc": cal["roc_auc"],
                "walk_forward_brier": cal["brier_score"],
                "walk_forward_rank_monotonicity": cal["rank_monotonicity"],
            }
        )
        grid_rows = [
            {"model_name": model_name, "threshold": t, **dict(zip(("precision", "coverage"), precision_at_threshold(y_true, y_proba, t)))}
            for t in grid
        ]
        pd.DataFrame(grid_rows).to_csv(THRESHOLD_DIR / f"grid_{model_name}.csv", index=False)
    return pd.DataFrame(rows).sort_values("walk_forward_precision", ascending=False)


def refit_and_score_on_holdout(
    model_name: str,
    spec: dict | None,
    frame_variant: pd.DataFrame,
    variant_cols: list[str],
    train_idx: pd.Index,
    holdout_idx: pd.Index,
    features_cfg: dict,
    modeling_cfg: dict,
    threshold: float,
    stacking_specs=None,
    stacking_lookup=None,
    random_state: int = 42,
) -> dict:
    y_tr = frame_variant.loc[train_idx, "home_covered"]
    y_ho = frame_variant.loc[holdout_idx, "home_covered"]
    cover_margin_tr = frame_variant.loc[train_idx, "cover_margin"]
    season_tr = frame_variant.loc[train_idx, "season"]
    favored_tr = frame_variant.loc[train_idx, "home_favored"]
    favored_ho = frame_variant.loc[holdout_idx, "home_favored"]
    X_tr_full = frame_variant.loc[train_idx, variant_cols]
    X_ho_full = frame_variant.loc[holdout_idx, variant_cols]

    if model_name == "majority_class":
        model = MajorityClassBaseline().fit(X_tr_full, y_tr)
        proba = model.predict_proba(X_ho_full)[:, 1]
    elif model_name == "always_favorite":
        model = AlwaysFavoriteBaseline().fit(X_tr_full, y_tr, favored_tr)
        proba = model.predict_proba(X_ho_full, favored_ho)[:, 1]
    elif model_name == "stacking_ensemble":
        X_tr_reduced, X_ho_reduced, _ = get_reduced_features(
            {}, X_tr_full, X_ho_full, y_tr, season_tr, "reduced", features_cfg, random_state
        )
        inner_min_seasons = safe_inner_min_seasons(season_tr, modeling_cfg["validation"]["min_train_seasons"])
        proba, _report = fit_stacking_ensemble(
            X_tr_reduced, y_tr, cover_margin_tr, season_tr, X_ho_reduced,
            stacking_specs, stacking_lookup, inner_min_seasons,
        )
    else:
        X_tr, X_ho, _ = get_reduced_features(
            {}, X_tr_full, X_ho_full, y_tr, season_tr, spec["feature_set_mode"], features_cfg, random_state
        )
        model = spec["builder"]()
        if spec["kind"] == "classifier":
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, cover_margin_tr)
        proba = model.predict_proba(X_ho)[:, 1]

    precision, coverage = precision_at_threshold(y_ho.to_numpy(), proba, threshold)
    cal = calibration_report(y_ho.to_numpy(), proba)
    return {
        "model_name": model_name,
        "threshold": threshold,
        "n_holdout": len(y_ho),
        "n_flagged": int((proba >= threshold).sum()),
        "holdout_precision": precision,
        "holdout_coverage": coverage,
        "holdout_roc_auc": cal["roc_auc"],
        "holdout_brier": cal["brier_score"],
        "holdout_rank_monotonicity": cal["rank_monotonicity"],
    }, proba


def main() -> None:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    for d in (THRESHOLD_DIR, CALIBRATION_DIR, MODEL_COMPARISON_DIR):
        d.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())
    oof_df = pd.read_csv(OOF_PATH)

    print("Selecting thresholds from pooled walk-forward OOF...")
    threshold_table = select_thresholds(oof_df, modeling_cfg)
    threshold_table.to_csv(THRESHOLD_DIR / "chosen_threshold_per_model.csv", index=False)
    print(threshold_table.to_string(index=False))

    full_variant, variant_cols = build_full_variant(frame, feature_columns, winning_cfg)
    train_pool, holdout = get_holdout_split(frame, data_cfg)
    train_idx, holdout_idx = train_pool.index, holdout.index

    track_a_specs = make_track_a_specs(modeling_cfg, random_state)
    track_b_specs = make_track_b_specs(modeling_cfg, random_state)
    spec_lookup = {s["name"]: s for s in track_a_specs + track_b_specs}
    stacking_cfg = modeling_cfg["track_c_stacking"]
    stacking_specs = [spec_lookup[n] for n in stacking_cfg["base_models"]]
    stacking_meta_builder = partial(
        build_classifier, stacking_cfg["meta_learner"]["kind"], stacking_cfg["meta_learner"]["params"], random_state
    )

    print("\nRefitting every candidate on the full train_pool and scoring on the true 2025 holdout...")
    holdout_rows = []
    holdout_proba_by_model = {}
    for _, row in threshold_table.iterrows():
        model_name, threshold = row["model_name"], row["threshold"]
        spec = spec_lookup.get(model_name)
        result, proba = refit_and_score_on_holdout(
            model_name, spec, full_variant, variant_cols, train_idx, holdout_idx,
            features_cfg, modeling_cfg, threshold,
            stacking_specs=stacking_specs, stacking_lookup=stacking_meta_builder, random_state=random_state,
        )
        holdout_rows.append(result)
        holdout_proba_by_model[model_name] = proba
        print(f"  {model_name}: holdout precision={result['holdout_precision']}, "
              f"coverage={result['holdout_coverage']:.2f}, n_flagged={result['n_flagged']}")

    holdout_table = pd.DataFrame(holdout_rows).merge(
        threshold_table[["model_name", "walk_forward_precision", "walk_forward_coverage"]], on="model_name"
    )
    holdout_table["walk_forward_rank"] = holdout_table["walk_forward_precision"].rank(ascending=False, method="min")
    holdout_table["holdout_rank"] = holdout_table["holdout_precision"].rank(ascending=False, method="min")
    holdout_table = holdout_table.sort_values("walk_forward_precision", ascending=False)
    holdout_table.to_csv(MODEL_COMPARISON_DIR / "holdout_model_comparison.csv", index=False)
    print("\n" + holdout_table.to_string(index=False))

    walk_forward_winner = threshold_table.iloc[0]["model_name"]
    y_ho_true = full_variant.loc[holdout_idx, "home_covered"].to_numpy()
    y_wf_true = oof_df.loc[oof_df["model_name"] == walk_forward_winner, "y_true"].to_numpy()
    y_wf_proba = oof_df.loc[oof_df["model_name"] == walk_forward_winner, "y_proba"].to_numpy()
    y_ho_proba = holdout_proba_by_model[walk_forward_winner]

    print(f"\nCalibration testing on production model ({walk_forward_winner})...")
    wf_cal = calibration_report(y_wf_true, y_wf_proba)
    ho_cal = calibration_report(y_ho_true, y_ho_proba)
    (CALIBRATION_DIR / "walk_forward_buckets.csv").write_text(pd.DataFrame(wf_cal["buckets"]).to_csv(index=False))
    (CALIBRATION_DIR / "holdout_buckets.csv").write_text(pd.DataFrame(ho_cal["buckets"]).to_csv(index=False))

    calibration_summary = {"none": {"walk_forward": wf_cal, "holdout": ho_cal}}
    for cal_name in modeling_cfg["evaluation"]["calibration_methods_to_test"]:
        if cal_name == "none":
            continue
        fit_fn, apply_fn = CALIBRATORS[cal_name]
        calibrator = fit_fn(y_wf_true, y_wf_proba)
        wf_calibrated = apply_fn(calibrator, y_wf_proba)
        ho_calibrated = apply_fn(calibrator, y_ho_proba)
        calibration_summary[cal_name] = {
            "walk_forward": calibration_report(y_wf_true, wf_calibrated),
            "holdout": calibration_report(y_ho_true, ho_calibrated),
        }
    (CALIBRATION_DIR / "walk_forward_summary.json").write_text(
        json.dumps({k: v["walk_forward"] for k, v in calibration_summary.items()}, indent=2, default=str)
    )
    (CALIBRATION_DIR / "holdout_summary.json").write_text(
        json.dumps({k: v["holdout"] for k, v in calibration_summary.items()}, indent=2, default=str)
    )

    winner_holdout_row = holdout_table.loc[holdout_table["model_name"] == walk_forward_winner].iloc[0]
    final_summary = {
        "winning_feature_config": winning_cfg,
        "walk_forward_selected_model": walk_forward_winner,
        "walk_forward_precision": winner_holdout_row["walk_forward_precision"],
        "walk_forward_coverage": winner_holdout_row["walk_forward_coverage"],
        "holdout_precision": winner_holdout_row["holdout_precision"],
        "holdout_coverage": winner_holdout_row["holdout_coverage"],
        "holdout_rank_of_walk_forward_winner": int(winner_holdout_row["holdout_rank"]),
        "n_candidates_compared": len(holdout_table),
        "target_precision": 0.53,
        "meets_target_on_holdout": bool((winner_holdout_row["holdout_precision"] or 0) >= 0.53),
        "best_holdout_model": holdout_table.sort_values("holdout_precision", ascending=False).iloc[0]["model_name"],
        "best_holdout_precision": float(holdout_table["holdout_precision"].max()),
    }
    (MODEL_COMPARISON_DIR / "final_summary.json").write_text(json.dumps(final_summary, indent=2, default=str))
    print("\n" + json.dumps(final_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
