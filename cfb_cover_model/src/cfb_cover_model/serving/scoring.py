"""Given an already-fitted ProductionArtifact (see artifact.py), score a week's raw feature
rows - whether they came from a CFB_Pred_Week_<N>.csv file or the live CFBD-API ingest path
(ingest/pipeline.py::build_current_week_rows) - into the same 12-column dual-model output
scripts/generate_weekly_predictions.py has always produced, plus one extra `artifact_version`
column so downstream monitoring (scripts/run_weekly_monitoring.py) can tell exactly which
artifact version produced a given week's predictions even after a later retrain.
"""
from __future__ import annotations

import pandas as pd

from cfb_cover_model.cleaning import prepare_week_frame
from cfb_cover_model.feature_engineering import apply_home_away_representation, build_transform_variant
from cfb_cover_model.feature_selection.selection import apply_feature_set
from cfb_cover_model.serving.artifact import CLASSIFIER_MODEL, REGRESSOR_MODEL, ProductionArtifact

OUTPUT_COLUMNS = [
    "game_id", "home_team", "away_team", "season", "week", "spread",
    f"{CLASSIFIER_MODEL}_probability", f"{CLASSIFIER_MODEL}_flag",
    f"{REGRESSOR_MODEL}_probability", f"{REGRESSOR_MODEL}_flag",
    "agreement_bet", "avg_probability", "artifact_version",
]


def build_feature_frame(
    df: pd.DataFrame, feature_columns: list[str], transforms: list[str], representation: str, data_cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Raw week rows -> engineered + transformed + home/away-represented feature frame, with
    the same schema-drift guard generate_weekly_predictions.py has always run inline. Shared
    by score_week() below and the data-drift monitoring module, which needs this exact same
    frame (pre-feature-selection) to compare against the artifact's training-time reference."""
    week_df, week_feature_columns = prepare_week_frame(df, data_cfg)
    assert set(week_feature_columns) == set(feature_columns), (
        "Week file's engineered feature set doesn't match the training artifact's recorded "
        "feature_columns - re-run scripts/load_and_validate_dataset.py and retrain if "
        "engineered_features.py has changed since this artifact was trained."
    )

    week_variant, week_cols = build_transform_variant(week_df, feature_columns, transforms)
    week_variant, week_cols = apply_home_away_representation(week_variant, week_cols, representation)
    return week_df, week_variant, week_cols


def score_week(artifact: ProductionArtifact, week_df_raw: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    week_df, week_variant, week_cols = build_feature_frame(
        week_df_raw, artifact.feature_columns, artifact.transforms, artifact.representation, data_cfg
    )
    assert set(week_cols) >= set(artifact.selected_columns), (
        "Week file is missing one or more of the artifact's selected feature columns - the "
        "week CSV's column schema no longer matches what this artifact was trained on."
    )

    X_week = apply_feature_set(week_variant[week_cols], "reduced", artifact.selected_columns)

    classifier_proba = artifact.classifier.predict_proba(X_week)[:, 1]
    regressor_proba = artifact.regressor.predict_proba(X_week)[:, 1]

    out = week_df[["game_id", "home_team", "away_team", "season", "week", "spread"]].copy()
    out[f"{CLASSIFIER_MODEL}_probability"] = classifier_proba
    out[f"{CLASSIFIER_MODEL}_flag"] = classifier_proba >= artifact.classifier_threshold
    out[f"{REGRESSOR_MODEL}_probability"] = regressor_proba
    out[f"{REGRESSOR_MODEL}_flag"] = regressor_proba >= artifact.regressor_threshold
    out["agreement_bet"] = out[f"{CLASSIFIER_MODEL}_flag"] & out[f"{REGRESSOR_MODEL}_flag"]
    out["avg_probability"] = (classifier_proba + regressor_proba) / 2
    out["artifact_version"] = artifact.version
    out = out.sort_values(["agreement_bet", "avg_probability"], ascending=[False, False]).reset_index(drop=True)

    return out[OUTPUT_COLUMNS]
