#!/usr/bin/env python
"""Diagnostic: in-sample training accuracy vs. pooled walk-forward OOF vs. true holdout,
for the walk-forward-selected model and the best true-holdout model. Distinguishes
overfitting (fits training data well, doesn't transfer) from insufficient signal (doesn't
even fit its own training data well) - see docs/project_story.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfb_cover_model.config import load_data_config, load_features_config, load_modeling_config
from cfb_cover_model.modeling.splits import get_holdout_split
from evaluate_models import build_full_variant
from train_models import get_reduced_features, make_track_a_specs, make_track_b_specs

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset.parquet"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_inventory" / "feature_columns.json"
WINNING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis" / "winning_feature_config.json"
OOF_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison" / "oof_predictions.csv"
THRESHOLD_TABLE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "threshold_selection" / "chosen_threshold_per_model.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison" / "train_vs_holdout_accuracy.csv"


def split_report(name, y_true, y_proba):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "split": name,
        "n": len(y_true),
        "accuracy@0.5": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6)),
        "base_rate": float(y_true.mean()),
    }


def main():
    data_cfg, features_cfg, modeling_cfg = load_data_config(), load_features_config(), load_modeling_config()
    random_state = modeling_cfg["random_state"]

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())
    oof_df = pd.read_csv(OOF_PATH)
    threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH)

    full_variant, variant_cols = build_full_variant(frame, feature_columns, winning_cfg)
    train_pool, holdout = get_holdout_split(frame, data_cfg)
    train_idx, holdout_idx = train_pool.index, holdout.index

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}

    models_to_check = [
        threshold_table.sort_values("walk_forward_precision", ascending=False).iloc[0]["model_name"],
        "gradient_boosting",
        "logistic_no_selection",
        "xgboost",
        "lightgbm",
        "xgboost_regressor",
        "elastic_net_regression",
    ]

    rows = []
    for model_name in dict.fromkeys(models_to_check):  # dedupe, keep order
        spec = spec_lookup[model_name]
        y_tr = full_variant.loc[train_idx, "home_covered"]
        y_ho = full_variant.loc[holdout_idx, "home_covered"]
        cover_margin_tr = full_variant.loc[train_idx, "cover_margin"]
        season_tr = full_variant.loc[train_idx, "season"]
        X_tr_full = full_variant.loc[train_idx, variant_cols]
        X_ho_full = full_variant.loc[holdout_idx, variant_cols]

        X_tr, X_ho, _ = get_reduced_features({}, X_tr_full, X_ho_full, y_tr, season_tr, spec["feature_set_mode"], features_cfg, random_state)
        model = spec["builder"]()
        if spec["kind"] == "classifier":
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, cover_margin_tr)

        train_proba = model.predict_proba(X_tr)[:, 1]
        holdout_proba = model.predict_proba(X_ho)[:, 1]
        wf_sub = oof_df[oof_df["model_name"] == model_name]

        for rep in [
            split_report("train (in-sample)", y_tr, train_proba),
            split_report("walk_forward_oof (pooled)", wf_sub["y_true"], wf_sub["y_proba"]),
            split_report("holdout (true 2025)", y_ho, holdout_proba),
        ]:
            rows.append({"model_name": model_name, **rep})

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    pd.set_option("display.width", 140)
    print(out_df.to_string(index=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
