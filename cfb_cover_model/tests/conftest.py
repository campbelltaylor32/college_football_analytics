import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_results_csv(tmp_path, rng):
    """A small, hand-checkable results CSV: 40 games across 4 seasons, with a few games
    engineered to be exact pushes so push-handling tests have known ground truth."""
    n = 40
    seasons = np.repeat([2020, 2021, 2022, 2023], 10)
    game_id = np.arange(1000, 1000 + n)
    home_points = rng.integers(10, 45, n)
    away_points = rng.integers(10, 45, n)
    home_minus_away = home_points - away_points
    signed_spread = rng.integers(-14, 14, n).astype(float)

    # Force games 0-2 to be exact pushes: home_minus_away == -signed_spread
    for i in range(3):
        signed_spread[i] = -float(home_minus_away[i])

    df = pd.DataFrame(
        {
            "game_id": game_id,
            "home_points": home_points,
            "away_points": away_points,
            "home_minus_away": home_minus_away,
            "spread": signed_spread,
        }
    )
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path, df, seasons


@pytest.fixture
def tiny_production_artifact(rng):
    """A small but real ProductionArtifact - actual fitted sklearn estimators (LogisticRegression
    + ElasticNet-based ResidualProbabilityRegressor, not xgboost, so this fixture has no
    boosting-extra dependency), 4 feature columns, fit on 60 synthetic rows. Used by
    test_serving_artifact.py (persistence round-trip) and test_serving_scoring.py (score_week's
    output-shaping logic, with build_feature_frame monkeypatched so this fixture never needs to
    survive the real feature-engineering pipeline)."""
    from sklearn.linear_model import ElasticNet, LogisticRegression

    from cfb_cover_model.modeling.regressor import ResidualProbabilityRegressor
    from cfb_cover_model.serving.artifact import ProductionArtifact
    from datetime import datetime, timezone

    n = 60
    selected_columns = ["diff_feature_a", "diff_feature_b", "diff_feature_c", "diff_feature_d"]
    X = pd.DataFrame(rng.normal(0, 1, (n, len(selected_columns))), columns=selected_columns)
    y = (X["diff_feature_a"] + rng.normal(0, 0.5, n) > 0).astype(int)
    margin = X["diff_feature_a"] * 5 + rng.normal(0, 3, n)

    classifier = LogisticRegression(max_iter=1000).fit(X, y)
    regressor = ResidualProbabilityRegressor(ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)).fit(X, margin)

    return ProductionArtifact(
        version="v2025w1_20260101T000000Z",
        trained_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        season_through=2025,
        week_through=1,
        classifier=classifier,
        regressor=regressor,
        selected_columns=selected_columns,
        feature_columns=selected_columns,
        transforms=["prev_week"],
        representation="differential",
        classifier_threshold=0.56,
        regressor_threshold=0.60,
        reference_training_frame=X,
        training_row_count=n,
        git_commit="deadbeef",
        library_versions={"pandas": pd.__version__},
    )


@pytest.fixture
def synthetic_modeling_frame(rng):
    """A minimal frame shaped like the real modeling frame (season/week/home_covered/
    feature columns) for split and leakage tests - no CSV I/O involved."""
    seasons = np.repeat([2015, 2016, 2017, 2018, 2019], 20)
    n = len(seasons)
    df = pd.DataFrame(
        {
            "game_id": np.arange(n),
            "season": seasons,
            "week": rng.integers(4, 12, n),
            "home_covered": rng.integers(0, 2, n),
            "cover_margin": rng.normal(0, 10, n),
            "home_favored": rng.integers(0, 2, n),
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
        }
    ).reset_index(drop=True)
    return df
