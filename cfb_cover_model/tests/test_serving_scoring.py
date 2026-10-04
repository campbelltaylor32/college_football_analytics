import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.serving import scoring


def _fake_build_feature_frame(df, feature_columns, transforms, representation, data_cfg):
    """Stands in for the real build_feature_frame (which depends on the full engineered-
    feature pipeline) so score_week's own output-shaping logic - the part actually written
    for this project, as opposed to delegated to prepare_week_frame/build_transform_variant/
    apply_home_away_representation, which are already covered by their own existing tests -
    can be tested in isolation with small, hand-checkable numbers."""
    n = 3
    week_df = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "home_team": ["A", "B", "C"],
            "away_team": ["X", "Y", "Z"],
            "season": [2025, 2025, 2025],
            "week": [10, 10, 10],
            "spread": [-3.5, 7.0, 1.0],
        }
    )
    week_variant = pd.DataFrame(
        {
            "diff_feature_a": [2.0, -2.0, 0.0],
            "diff_feature_b": [1.0, -1.0, 0.0],
            "diff_feature_c": [0.5, -0.5, 0.0],
            "diff_feature_d": [0.1, -0.1, 0.0],
        }
    )
    week_cols = list(week_variant.columns)
    return week_df, week_variant, week_cols


@pytest.fixture(autouse=True)
def _patch_build_feature_frame(monkeypatch):
    monkeypatch.setattr(scoring, "build_feature_frame", _fake_build_feature_frame)


def test_score_week_output_schema(tiny_production_artifact):
    out = scoring.score_week(tiny_production_artifact, pd.DataFrame(), data_cfg={})
    assert list(out.columns) == scoring.OUTPUT_COLUMNS
    assert len(out) == 3
    assert out["artifact_version"].eq(tiny_production_artifact.version).all()


def test_agreement_bet_is_and_of_both_flags(tiny_production_artifact):
    out = scoring.score_week(tiny_production_artifact, pd.DataFrame(), data_cfg={})
    expected = out["logistic_regression_flag"] & out["xgboost_regressor_flag"]
    pd.testing.assert_series_equal(out["agreement_bet"], expected, check_names=False)


def test_flag_boundary_is_inclusive_of_threshold(tiny_production_artifact):
    """Matches scripts/generate_weekly_predictions.py's `proba >= threshold` exactly (not
    strictly greater-than) - a probability exactly equal to the threshold must flag True."""
    artifact = tiny_production_artifact
    out = scoring.score_week(artifact, pd.DataFrame(), data_cfg={})

    at_threshold = out[f"{scoring.CLASSIFIER_MODEL}_probability"] == artifact.classifier_threshold
    if at_threshold.any():
        assert out.loc[at_threshold, f"{scoring.CLASSIFIER_MODEL}_flag"].all()


def test_avg_probability_is_mean_of_both(tiny_production_artifact):
    out = scoring.score_week(tiny_production_artifact, pd.DataFrame(), data_cfg={})
    expected = (out[f"{scoring.CLASSIFIER_MODEL}_probability"] + out[f"{scoring.REGRESSOR_MODEL}_probability"]) / 2
    pd.testing.assert_series_equal(out["avg_probability"], expected, check_names=False)
