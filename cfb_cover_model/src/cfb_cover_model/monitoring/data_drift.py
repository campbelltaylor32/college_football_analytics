"""Data-drift monitoring via Evidently - the only module in this project that imports
`evidently`, deliberately, so a future forced version bump only ever touches this one file.
Callers (the API's /monitoring/status route, scripts/run_weekly_monitoring.py, and the tests)
consume run_data_drift_report()'s trimmed summary dict, never Evidently's raw nested JSON, so
they're insulated from Evidently's own API churn across versions.

Compares the artifact's training-time reference feature snapshot (outputs/models/production/
<version>/reference_training_features.parquet) against an upcoming week's post-feature-
engineering, pre-selection frame - i.e. the same columns build_transform_variant +
apply_home_away_representation produce, before fit_feature_set narrows them down to the
"reduced" set the models actually consume. Comparing at this wider stage catches drift in
features that got selected out of one artifact but could get selected back in on a future
retrain.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DRIFT_DIR = PROJECT_ROOT / "outputs" / "monitoring" / "data_drift"


def run_data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path = DATA_DRIFT_DIR,
    label: str = "report",
) -> dict:
    """feature_columns should never include bookkeeping columns (game_id, home_team, season,
    week, home_covered, cover_margin, ...) - only the actual model-input feature columns
    common to both frames."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    reference = reference_df[feature_columns]
    current = current_df[feature_columns]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{label}_data_drift.html"
    json_path = output_dir / f"{label}_data_drift.json"
    report.save_html(str(html_path))

    raw = report.as_dict()
    summary = _extract_summary(raw, feature_columns)
    json_path.write_text(json.dumps(summary, indent=2))

    return summary


def _extract_summary(raw: dict, feature_columns: list[str]) -> dict:
    """Pulls just the numbers callers actually need out of Evidently's nested `as_dict()`
    output. Isolated in its own function (rather than inlined above) so a future Evidently
    version's differently-shaped JSON only requires changing this one function."""
    drift_metric = next(
        (m for m in raw.get("metrics", []) if m.get("metric") == "DatasetDriftMetric"), None
    )
    dataset_drift = bool(drift_metric["result"]["dataset_drift"]) if drift_metric else False
    share_drifted = float(drift_metric["result"]["share_of_drifted_columns"]) if drift_metric else 0.0

    column_drift_metric = next(
        (m for m in raw.get("metrics", []) if m.get("metric") == "DataDriftTable"), None
    )
    drifted_columns = []
    if column_drift_metric:
        per_column = column_drift_metric["result"].get("drift_by_columns", {})
        drifted_columns = [
            col for col, info in per_column.items() if info.get("drift_detected")
        ]

    return {
        "n_features_compared": len(feature_columns),
        "dataset_drift": dataset_drift,
        "share_drifted_columns": share_drifted,
        "drifted_columns": drifted_columns,
    }
