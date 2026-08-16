#!/usr/bin/env python
"""Stage 6: walk-forward out-of-fold predictions for every candidate model - Track A
(direct classification), Track B (regression-to-probability), Track C (stacked ensemble),
and two zero-parameter baselines. Every stage-3/4/5 feature reduction is refit fresh inside
each fold's training data only (see feature_selection/selection.py); nothing here reuses a
reduction fit on data outside the fold it's applied to.

Writes outputs/model_comparison/oof_predictions.csv (one row per game x candidate model) -
evaluate_models.py consumes this for threshold selection, calibration, and the final
holdout comparison. No threshold selection or holdout scoring happens in this script.
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.selection import apply_feature_set, fit_feature_set
from cfb_cover_model.modeling.classifiers import (
    AlwaysFavoriteBaseline,
    MajorityClassBaseline,
    build_classifier,
)
from cfb_cover_model.modeling.regressor import (
    QuantileProbabilityRegressor,
    ResidualProbabilityRegressor,
    build_point_regressor,
)
from cfb_cover_model.modeling.splits import get_holdout_split, walk_forward_folds
from cfb_cover_model.modeling.stacking import fit_stacking_ensemble, safe_inner_min_seasons

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"


def make_track_a_specs(modeling_cfg: dict, random_state: int) -> list[dict]:
    specs = []
    for c in modeling_cfg["track_a_classifiers"]:
        specs.append(
            {
                "name": c["name"],
                "track": "A",
                "kind": "classifier",
                "feature_set_mode": c["feature_set"],
                "builder": partial(build_classifier, c["kind"], c["params"], random_state),
            }
        )
    return specs


def make_track_b_specs(modeling_cfg: dict, random_state: int) -> list[dict]:
    specs = []
    for r in modeling_cfg["track_b_regressors"]:
        if r["kind"] == "quantile_regression":
            builder = partial(QuantileProbabilityRegressor, r["quantiles"], r["params"], random_state)
        else:
            builder = (
                lambda kind=r["kind"], params=r["params"]: ResidualProbabilityRegressor(
                    build_point_regressor(kind, params, random_state)
                )
            )
        specs.append(
            {
                "name": r["name"],
                "track": "B",
                "kind": "regressor",
                "feature_set_mode": r["feature_set"],
                "builder": builder,
            }
        )
    return specs


def get_reduced_features(cache, X_tr, X_val, y_tr, season_tr, mode, features_cfg, random_state):
    if mode not in cache:
        artifact, report = fit_feature_set(X_tr, y_tr, season_tr, mode, features_cfg, random_state)
        cache[mode] = (artifact, report)
    artifact, report = cache[mode]
    return apply_feature_set(X_tr, mode, artifact), apply_feature_set(X_val, mode, artifact), report


def main() -> None:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())

    train_pool, _holdout = get_holdout_split(frame, data_cfg)

    variant, variant_cols = build_transform_variant(train_pool, feature_columns, winning_cfg["transforms"])
    variant, variant_cols = apply_home_away_representation(variant, variant_cols, winning_cfg["representation"])
    variant = variant.reset_index(drop=True)
    variant.index = train_pool.index
    variant["season"] = train_pool["season"]
    variant["home_covered"] = train_pool["home_covered"]
    variant["cover_margin"] = train_pool["cover_margin"]
    variant["home_favored"] = train_pool["home_favored"]

    folds = walk_forward_folds(train_pool, modeling_cfg)
    track_a_specs = make_track_a_specs(modeling_cfg, random_state)
    track_b_specs = make_track_b_specs(modeling_cfg, random_state)
    base_lookup = {s["name"]: s for s in track_a_specs + track_b_specs}
    stacking_cfg = modeling_cfg["track_c_stacking"]

    rows = []
    for fold in folds:
        t0 = time.time()
        X_tr_full = variant.loc[fold.train_idx, variant_cols]
        X_val_full = variant.loc[fold.val_idx, variant_cols]
        y_tr = variant.loc[fold.train_idx, "home_covered"]
        y_val = variant.loc[fold.val_idx, "home_covered"]
        cover_margin_tr = variant.loc[fold.train_idx, "cover_margin"]
        season_tr = variant.loc[fold.train_idx, "season"]
        favored_tr = variant.loc[fold.train_idx, "home_favored"]
        favored_val = variant.loc[fold.val_idx, "home_favored"]
        game_id_val = train_pool.loc[fold.val_idx, "game_id"]

        feature_cache: dict = {}

        for spec in track_a_specs + track_b_specs:
            X_tr, X_val, _report = get_reduced_features(
                feature_cache, X_tr_full, X_val_full, y_tr, season_tr,
                spec["feature_set_mode"], features_cfg, random_state,
            )
            model = spec["builder"]()
            if spec["kind"] == "classifier":
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, cover_margin_tr)
            proba = model.predict_proba(X_val)[:, 1]
            for gid, yt, yp in zip(game_id_val, y_val, proba):
                rows.append(
                    {"model_name": spec["name"], "track": spec["track"], "val_season": int(fold.val_season),
                     "game_id": int(gid), "y_true": int(yt), "y_proba": float(yp)}
                )

        # Track C: stacking, on the shared "reduced" feature set (see stacking.py docstring)
        X_tr_reduced, X_val_reduced, _report = get_reduced_features(
            feature_cache, X_tr_full, X_val_full, y_tr, season_tr, "reduced", features_cfg, random_state
        )
        base_specs_for_stack = [base_lookup[name] for name in stacking_cfg["base_models"]]
        meta_kind = stacking_cfg["meta_learner"]["kind"]
        meta_params = stacking_cfg["meta_learner"]["params"]
        meta_builder = partial(build_classifier, meta_kind, meta_params, random_state)
        inner_min_seasons = safe_inner_min_seasons(season_tr, modeling_cfg["validation"]["min_train_seasons"])
        stacked_proba, _stack_report = fit_stacking_ensemble(
            X_tr_reduced, y_tr, cover_margin_tr, season_tr, X_val_reduced,
            base_specs_for_stack, meta_builder, inner_min_seasons,
        )
        for gid, yt, yp in zip(game_id_val, y_val, stacked_proba):
            rows.append(
                {"model_name": "stacking_ensemble", "track": "C", "val_season": int(fold.val_season),
                 "game_id": int(gid), "y_true": int(yt), "y_proba": float(yp)}
            )

        # Baselines
        maj = MajorityClassBaseline().fit(X_tr_full, y_tr)
        maj_proba = maj.predict_proba(X_val_full)[:, 1]
        for gid, yt, yp in zip(game_id_val, y_val, maj_proba):
            rows.append(
                {"model_name": "majority_class", "track": "baseline", "val_season": int(fold.val_season),
                 "game_id": int(gid), "y_true": int(yt), "y_proba": float(yp)}
            )

        fav = AlwaysFavoriteBaseline().fit(X_tr_full, y_tr, favored_tr)
        fav_proba = fav.predict_proba(X_val_full, favored_val)[:, 1]
        for gid, yt, yp in zip(game_id_val, y_val, fav_proba):
            rows.append(
                {"model_name": "always_favorite", "track": "baseline", "val_season": int(fold.val_season),
                 "game_id": int(gid), "y_true": int(yt), "y_proba": float(yp)}
            )

        print(f"fold val_season={fold.val_season} done in {time.time()-t0:.1f}s "
              f"(n_train={len(fold.train_idx)}, n_val={len(fold.val_idx)})")

    oof_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    summary = (
        oof_df.groupby("model_name")
        .apply(lambda g: pd.Series({"n_predictions": len(g), "n_folds": g["val_season"].nunique()}))
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {len(oof_df)} predictions to {OUT_DIR / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
