#!/usr/bin/env python
"""Feature deep-dive, stage 1: fold-stability analysis.

For two configs - the winning prev_week_only/raw_dual, and the best avg_all_only
combination - refits stage 3 (correlation pruning) + stage 4 (embedded selection) on each
of the 6 walk-forward folds' training rows only, and records which features were selected
and their standardized coefficient magnitude. A feature that's selected in most/all folds,
with a consistently large coefficient, is a much stronger candidate "real predictor" than
one that only shows up in a single fold or a single config - see
docs/feature_importance_findings.md for the synthesized write-up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.feature_categories import parse_column
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.correlation_pruning import prune_correlated_features
from cfb_cover_model.feature_selection.embedded_selection import select_features_embedded
from cfb_cover_model.modeling.splits import get_holdout_split, walk_forward_folds

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
ABLATION_RESULTS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "transform_ablation_results.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis"


def pick_best_avg_all_only_config(ablation_results: list[dict]) -> dict:
    candidates = [r for r in ablation_results if r["transform_name"] == "avg_all_only"]
    candidates.sort(key=lambda r: (r["pooled"]["precision"] if r["pooled"]["precision"] is not None else -1.0), reverse=True)
    best = candidates[0]
    return {"transform_name": best["transform_name"], "transforms": best["transforms"], "representation": best["representation"]}


def analyze_config(config_name: str, config: dict, train_pool: pd.DataFrame, features_cfg: dict, modeling_cfg: dict, random_state: int) -> pd.DataFrame:
    print(f"\n=== {config_name}: {config['transform_name']} / {config['representation']} ===")
    variant, variant_cols = build_transform_variant(train_pool, json.loads(FEATURE_COLUMNS_PATH.read_text()), config["transforms"])
    variant, variant_cols = apply_home_away_representation(variant, variant_cols, config["representation"])
    variant = variant.reset_index(drop=True)
    variant.index = train_pool.index
    variant["season"] = train_pool["season"]
    variant["home_covered"] = train_pool["home_covered"]

    folds = walk_forward_folds(train_pool, modeling_cfg)
    print(f"  {len(variant_cols)} candidate columns, {len(folds)} folds")

    selected_by_fold: dict[str, list[bool]] = {c: [] for c in variant_cols}
    coef_by_fold: dict[str, list[float]] = {c: [] for c in variant_cols}

    for fold in folds:
        X_tr = variant.loc[fold.train_idx, variant_cols]
        y_tr = variant.loc[fold.train_idx, "home_covered"]
        season_tr = variant.loc[fold.train_idx, "season"]

        pruned_cols, _corr_report = prune_correlated_features(
            X_tr, y_tr, features_cfg["correlation_pruning"]["correlation_threshold"]
        )
        selected, embed_report = select_features_embedded(
            X_tr[pruned_cols], y_tr, season_tr,
            features_cfg["embedded_selection"]["l1_ratio_grid"],
            features_cfg["embedded_selection"]["C_grid"],
            features_cfg["embedded_selection"]["inner_cv_folds"],
            features_cfg["embedded_selection"]["max_features"],
            random_state=random_state,
        )
        selected_set = set(selected)
        coef_lookup = dict(embed_report["ranked_coefficients"])

        for col in variant_cols:
            selected_by_fold[col].append(col in selected_set)
            if col in coef_lookup:
                coef_by_fold[col].append(coef_lookup[col])

        print(f"  fold val_season={fold.val_season}: {len(pruned_cols)} after corr-pruning -> {len(selected)} selected")

    # Note: coef_by_fold[col] only has entries for folds where the feature had a nonzero
    # elastic-net coefficient at all (embedded_selection's ranked_coefficients), which is a
    # superset of folds where it made the final top-max_features cutoff (selected_by_fold).
    # mean_abs_coef_when_nonzero is intentionally the broader of the two - a feature with a
    # real but modest coefficient can be consistently nonzero without always cracking the cap.
    rows = []
    for col in variant_cols:
        parsed = parse_column(col)
        sel = selected_by_fold[col]
        coefs = coef_by_fold[col]
        mean_abs_coef = float(np.mean(coefs)) if coefs else 0.0
        rows.append(
            {
                "column": col,
                "base_stat": parsed["base_stat"],
                "side": parsed["side"],
                "domain": parsed["domain"],
                "offense_defense": parsed["offense_defense"],
                "selection_frequency": int(sum(sel)),
                "n_folds": len(sel),
                "n_folds_nonzero_coef": len(coefs),
                "mean_abs_coef_when_nonzero": mean_abs_coef,
                "consistency_score": int(sum(sel)) * mean_abs_coef,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["selection_frequency", "mean_abs_coef_when_nonzero"], ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / f"feature_stability_{config_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    category_rollup = (
        df.groupby("domain")
        .agg(
            n_features=("column", "size"),
            n_ever_selected=("selection_frequency", lambda s: int((s > 0).sum())),
            total_selection_count=("selection_frequency", "sum"),
            mean_selection_frequency=("selection_frequency", "mean"),
            total_consistency_score=("consistency_score", "sum"),
        )
        .reset_index()
        .sort_values("total_consistency_score", ascending=False)
    )
    category_rollup["pct_of_total_consistency_score"] = (
        category_rollup["total_consistency_score"] / category_rollup["total_consistency_score"].sum()
    )
    cat_path = OUT_DIR / f"feature_stability_by_category_{config_name}.csv"
    category_rollup.to_csv(cat_path, index=False)
    print(f"  wrote {cat_path}")

    return df


def main():
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    frame = pd.read_parquet(DATASET_PATH)
    train_pool, _holdout = get_holdout_split(frame, data_cfg)

    winning_config = json.loads(WINNING_CONFIG_PATH.read_text())
    ablation_results = json.loads(ABLATION_RESULTS_PATH.read_text())
    avg_all_config = pick_best_avg_all_only_config(ablation_results)
    print(f"avg_all_only comparison config: {avg_all_config}")

    winning_config_name = f"winning_{winning_config['transform_name']}"
    df_winning = analyze_config(winning_config_name, winning_config, train_pool, features_cfg, modeling_cfg, random_state)
    df_avg_all = analyze_config("avg_all_only", avg_all_config, train_pool, features_cfg, modeling_cfg, random_state)

    # Cross-config comparison at the base_stat level (side-agnostic, since one config is
    # raw_dual [home_X/away_X separate] and the other is differential [diff_X]).
    winning_by_stat = (
        df_winning.groupby("base_stat")
        .agg(winning_selection_frequency=("selection_frequency", "max"), winning_domain=("domain", "first"))
        .reset_index()
    )
    avg_all_by_stat = (
        df_avg_all.groupby("base_stat")
        .agg(avg_all_selection_frequency=("selection_frequency", "max"), avg_all_domain=("domain", "first"))
        .reset_index()
    )
    cross = winning_by_stat.merge(avg_all_by_stat, on="base_stat", how="outer").fillna(0)
    cross["domain"] = cross["winning_domain"].where(cross["winning_domain"] != 0, cross["avg_all_domain"])
    cross["important_in_both"] = (cross["winning_selection_frequency"] > 0) & (cross["avg_all_selection_frequency"] > 0)
    cross = cross.sort_values(
        ["important_in_both", "winning_selection_frequency", "avg_all_selection_frequency"], ascending=False
    )
    cross_path = OUT_DIR / "feature_stability_cross_transform_comparison.csv"
    cross.to_csv(cross_path, index=False)
    print(f"\nwrote {cross_path}")
    print(f"base stats important (selected in >=1 fold) under BOTH configs: {int(cross['important_in_both'].sum())} of {len(cross)}")


if __name__ == "__main__":
    main()
