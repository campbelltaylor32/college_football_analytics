#!/usr/bin/env python
"""Feature deep-dive, stage 2: single-shot permutation importance on the true holdout.

Refits logistic_regression (the walk-forward-selected model) once on the entire
train_pool, then measures permutation importance on the never-touched 2025 holdout,
scored by the project's own precision-at-coverage-floor metric at that model's
walk-forward-chosen threshold - not accuracy/AUC, consistent with how every other
evaluation in this project is scored. Mirrors ../cfb_spread_model/scripts/explain_model.py,
but rolls up by domain category (src/cfb_cover_model/feature_categories.py) instead of by
temporal transform, since the winning config here is ~100% prev_week already.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.feature_categories import parse_column
from cfb_cover_model.modeling.evaluation import precision_at_threshold
from cfb_cover_model.modeling.splits import get_holdout_split
from evaluate_models import build_full_variant
from train_models import get_reduced_features, make_track_a_specs, make_track_b_specs

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
THRESHOLD_TABLE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"

MODEL_NAME = "logistic_regression"
N_REPEATS = 50


def make_precision_scorer(threshold: float):
    def scorer(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        precision, _coverage = precision_at_threshold(np.asarray(y), proba, threshold)
        return precision if precision is not None else 0.0

    return scorer


def main() -> None:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())
    threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH)

    threshold_row = threshold_table.loc[threshold_table["model_name"] == MODEL_NAME]
    if threshold_row.empty:
        raise ValueError(f"No threshold recorded for {MODEL_NAME!r} in {THRESHOLD_TABLE_PATH}")
    threshold = float(threshold_row.iloc[0]["threshold"])
    print(f"Scoring permutation importance for {MODEL_NAME!r} at threshold={threshold}")

    full_variant, variant_cols = build_full_variant(frame, feature_columns, winning_cfg)
    train_pool, holdout = get_holdout_split(frame, data_cfg)
    train_idx, holdout_idx = train_pool.index, holdout.index

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}
    spec = spec_lookup[MODEL_NAME]

    y_tr = full_variant.loc[train_idx, "home_covered"]
    y_ho = full_variant.loc[holdout_idx, "home_covered"]
    season_tr = full_variant.loc[train_idx, "season"]
    X_tr_full = full_variant.loc[train_idx, variant_cols]
    X_ho_full = full_variant.loc[holdout_idx, variant_cols]

    X_tr, X_ho, _report = get_reduced_features(
        {}, X_tr_full, X_ho_full, y_tr, season_tr, spec["feature_set_mode"], features_cfg, random_state
    )
    print(f"Refitting on {len(X_tr)} train_pool rows, {X_tr.shape[1]} reduced features...")
    model = spec["builder"]()
    model.fit(X_tr, y_tr)

    scorer = make_precision_scorer(threshold)
    baseline_precision = scorer(model, X_ho, y_ho)
    print(f"Baseline holdout precision at threshold {threshold}: {baseline_precision}")

    print(f"Running permutation importance ({N_REPEATS} repeats) on {len(X_ho)} holdout rows...")
    result = permutation_importance(
        model, X_ho, y_ho, scoring=scorer, n_repeats=N_REPEATS, random_state=random_state
    )

    rows = []
    for i, col in enumerate(X_ho.columns):
        parsed = parse_column(col)
        rows.append(
            {
                "feature": col,
                "base_stat": parsed["base_stat"],
                "side": parsed["side"],
                "domain": parsed["domain"],
                "offense_defense": parsed["offense_defense"],
                "permutation_importance_mean": result.importances_mean[i],
                "permutation_importance_std": result.importances_std[i],
            }
        )
    df = pd.DataFrame(rows).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    df["permutation_importance_rank"] = df.index + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "feature_importance_holdout.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTop 15 features by holdout permutation importance:\n{df.head(15).to_string(index=False)}")
    print(f"\nWrote {out_path}")

    category_rollup = (
        df.groupby("domain")
        .agg(
            n_features=("feature", "size"),
            total_permutation_importance=("permutation_importance_mean", "sum"),
            mean_permutation_importance=("permutation_importance_mean", "mean"),
        )
        .reset_index()
        .sort_values("total_permutation_importance", ascending=False)
    )
    total_mass = category_rollup["total_permutation_importance"].sum()
    category_rollup["pct_of_importance_mass"] = (
        category_rollup["total_permutation_importance"] / total_mass if total_mass else 0.0
    )
    cat_path = OUT_DIR / "feature_importance_holdout_by_category.csv"
    category_rollup.to_csv(cat_path, index=False)
    print(f"\nCategory rollup:\n{category_rollup.to_string(index=False)}")
    print(f"\nWrote {cat_path}")


if __name__ == "__main__":
    main()
