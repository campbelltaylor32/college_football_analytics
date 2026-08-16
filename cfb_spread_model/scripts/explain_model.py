#!/usr/bin/env python
"""Feature importance for the current production model (scripts/evaluate_models.py's saved
artifact), computed on the true 2025 holdout rows -- no retraining, the saved pipeline is
already fit. Two measures (gain-based + precision-scored permutation importance), joined with
each feature's temporal-transform category, answer both "what matters" and "how much do
single-game (prev_week_*) predictors matter" for whatever model is currently in production.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from cfb_spread_model.artifacts import load_latest_production_artifact
from cfb_spread_model.config import load_data_config, load_modeling_config
from cfb_spread_model.data import build_feature_matrix
from cfb_spread_model.feature_selection.precision_scoring import precision_at_coverage_floor_scorer
from cfb_spread_model.modeling import importance
from cfb_spread_model.modeling.holdout import load_holdout_frame
from cfb_spread_model.utils.logging import get_logger
from cfb_spread_model.utils.paths import OUTPUTS_MODEL_COMPARISON, ensure_dirs

logger = get_logger(__name__)


def main() -> int:
    ensure_dirs()
    data_cfg = load_data_config()
    modeling_cfg = load_modeling_config()

    model, features, metadata = load_latest_production_artifact()

    val_df = load_holdout_frame(modeling_cfg)
    X_val_full, y_val = build_feature_matrix(val_df, data_cfg)
    X_val = X_val_full[features]

    scorer = precision_at_coverage_floor_scorer(modeling_cfg.precision_objective.min_coverage_floor)
    perm_df = importance.compute_permutation_importance_for_model(
        model, X_val, y_val, scorer, n_repeats=50, random_seed=modeling_cfg.random_seed
    )

    inner_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
    try:
        gain_importance = importance.compute_gain_importance(inner_model, features)
    except AttributeError as e:
        logger.warning(f"No gain-based importance available for {metadata['model_name']}: {e}")
        gain_importance = pd.Series(dtype=float, name="gain_importance")

    report = importance.build_feature_importance_report(gain_importance, perm_df)
    category_summary = importance.summarize_by_temporal_transform(report)

    today = date.today().strftime("%Y%m%d")
    report_path = OUTPUTS_MODEL_COMPARISON / f"feature_importance_{today}.csv"
    summary_path = OUTPUTS_MODEL_COMPARISON / f"feature_importance_by_category_{today}.csv"
    report.to_csv(report_path, index=False)
    category_summary.to_csv(summary_path, index=False)

    top25 = report.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top25["feature"], top25["permutation_importance_mean"])
    ax.set_xlabel("Permutation importance (precision @ coverage floor)")
    ax.set_title(f"Top 25 features -- {metadata['model_name']} ({today})")
    fig.tight_layout()
    chart_path = OUTPUTS_MODEL_COMPARISON / "feature_importance_top25.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    prev_week_row = category_summary[category_summary["temporal_transform"] == "prev_week"]
    n_prev_week = int(prev_week_row["n_features"].sum())
    pct_perm_mass = float(prev_week_row["pct_of_permutation_importance"].sum())
    logger.info(
        f"{metadata['model_name']} ({len(features)} features): {n_prev_week} are prev_week_* "
        f"({n_prev_week / len(features):.0%} of features), accounting for {pct_perm_mass:.0%} "
        f"of total permutation-importance mass"
    )
    logger.info(f"Category breakdown:\n{category_summary.to_string(index=False)}")
    logger.info(f"Wrote {report_path.name}, {summary_path.name}, {chart_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
