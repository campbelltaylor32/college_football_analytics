#!/usr/bin/env python
"""Tests whether requiring agreement between a Track A classifier and a Track B
regressor - both flagging the same game as a bet, at each model's own walk-forward-
tuned threshold - produces a more reliable subset than either model alone. All
candidates are refit once on the full train_pool and scored on the never-touched 2025
holdout, exactly as evaluate_models.py does; this script only adds the pairwise
agreement view on top of predictions each model already produces individually.

Not part of the core run_pipeline.py chain - a reporting/exploration script over
already-selected model choices, like generate_2025_holdout_report.py.
"""
from __future__ import annotations

import itertools
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
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model_comparison"

# Curated shortlist - every Track A classifier and every Track B regressor, excluding
# the two diagnostic-only baselines (majority_class, always_favorite) and the PCA/no-
# selection variants (already known to be either weaker or a diagnostic anchor, not a
# real agreement-ensemble candidate).
TRACK_A_CANDIDATES = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"]
TRACK_B_CANDIDATES = ["elastic_net_regression", "xgboost_regressor", "quantile_regression"]
MIN_COVERAGE_FLOOR = 0.10  # looser than the project's usual 0.20 - agreement subsets are inherently smaller


def main() -> None:
    data_cfg = load_data_config()
    features_cfg = load_features_config()
    modeling_cfg = load_modeling_config()
    random_state = modeling_cfg["random_state"]

    frame = pd.read_parquet(DATASET_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    winning_cfg = json.loads(WINNING_CONFIG_PATH.read_text())
    threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH).set_index("model_name")["threshold"].to_dict()

    full_variant, variant_cols = build_full_variant(frame, feature_columns, winning_cfg)
    train_pool, holdout = get_holdout_split(frame, data_cfg)
    train_idx, holdout_idx = train_pool.index, holdout.index

    spec_lookup = {s["name"]: s for s in make_track_a_specs(modeling_cfg, random_state) + make_track_b_specs(modeling_cfg, random_state)}

    y_tr = full_variant.loc[train_idx, "home_covered"]
    y_ho = full_variant.loc[holdout_idx, "home_covered"].to_numpy()
    cover_margin_tr = full_variant.loc[train_idx, "cover_margin"]
    season_tr = full_variant.loc[train_idx, "season"]
    X_tr_full = full_variant.loc[train_idx, variant_cols]
    X_ho_full = full_variant.loc[holdout_idx, variant_cols]

    predictions: dict[str, dict] = {}
    feature_cache: dict = {}
    for model_name in TRACK_A_CANDIDATES + TRACK_B_CANDIDATES:
        spec = spec_lookup[model_name]
        threshold = threshold_table[model_name]
        X_tr, X_ho, _report = get_reduced_features(
            feature_cache, X_tr_full, X_ho_full, y_tr, season_tr, spec["feature_set_mode"], features_cfg, random_state
        )
        model = spec["builder"]()
        if spec["kind"] == "classifier":
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, cover_margin_tr)
        proba = model.predict_proba(X_ho)[:, 1]
        bet_flag = proba >= threshold
        predictions[model_name] = {"proba": proba, "bet_flag": bet_flag, "threshold": threshold}
        precision, coverage = precision_at_threshold(y_ho, proba, threshold)
        print(f"  {model_name}: threshold={threshold}, n_flagged={int(bet_flag.sum())}, "
              f"precision={precision}, coverage={coverage:.3f}")

    rows = []
    for classifier, regressor in itertools.product(TRACK_A_CANDIDATES, TRACK_B_CANDIDATES):
        agree = predictions[classifier]["bet_flag"] & predictions[regressor]["bet_flag"]
        n_agree = int(agree.sum())
        if n_agree == 0:
            precision, coverage = None, 0.0
        else:
            precision = float((y_ho[agree] == 1).mean())
            coverage = n_agree / len(y_ho)
        rows.append(
            {
                "classifier": classifier,
                "regressor": regressor,
                "n_agree": n_agree,
                "agreement_precision": precision,
                "agreement_coverage": coverage,
                "classifier_alone_precision": precision_at_threshold(y_ho, predictions[classifier]["proba"], predictions[classifier]["threshold"])[0],
                "regressor_alone_precision": precision_at_threshold(y_ho, predictions[regressor]["proba"], predictions[regressor]["threshold"])[0],
            }
        )

    df = pd.DataFrame(rows)
    df["meets_floor"] = df["agreement_coverage"] >= MIN_COVERAGE_FLOOR
    df["lift_vs_classifier"] = df["agreement_precision"] - df["classifier_alone_precision"]
    df["lift_vs_regressor"] = df["agreement_precision"] - df["regressor_alone_precision"]
    df = df.sort_values("agreement_precision", ascending=False)

    out_path = OUT_DIR / "model_agreement_combinations.csv"
    df.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
