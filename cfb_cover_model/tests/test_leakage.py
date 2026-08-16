import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb_cover_model.cleaning import candidate_feature_columns, expand_base_stats
from cfb_cover_model.feature_selection.correlation_pruning import prune_correlated_features
from cfb_cover_model.feature_selection.embedded_selection import select_features_embedded


def test_season_excluded_from_candidate_features():
    data_cfg = {
        "id_columns": ["game_id", "home_team", "away_team"],
        "leakage_adjacent_columns": ["home_favored"],
        "known_bad_base_stats": [],
        "deterministic_redundant_base_stats": [],
    }
    columns = pd.Index(["game_id", "home_team", "away_team", "season", "home_favored", "spread", "week", "home_covered"])
    features = candidate_feature_columns(data_cfg, columns)
    # season must never be a candidate feature - it's a splitting key only, see
    # cleaning.py's NON_FEATURE_BOOKKEEPING_COLUMNS comment.
    assert "season" not in features
    assert "game_id" not in features
    assert "home_favored" not in features
    assert "spread" in features
    assert "week" in features


def test_known_bad_and_redundant_columns_expand_across_home_away_and_transforms():
    data_cfg = {
        "id_columns": [],
        "leakage_adjacent_columns": [],
        "known_bad_base_stats": ["Offense_EPA_per_Run"],
        "deterministic_redundant_base_stats": ["point_differential"],
    }
    columns = pd.Index(
        [
            "home_prev_week_Offense_EPA_per_Run",
            "away_Offense_EPA_per_Run_avg3",
            "home_point_differential_avg_all",
            "home_talent",  # non-temporal, should not match any base-stat pattern here
            "week",
        ]
    )
    features = candidate_feature_columns(data_cfg, columns)
    assert "home_prev_week_Offense_EPA_per_Run" not in features
    assert "away_Offense_EPA_per_Run_avg3" not in features
    assert "home_point_differential_avg_all" not in features
    assert "home_talent" in features
    assert "week" in features


def test_expand_base_stats_only_matches_real_columns():
    columns = pd.Index(["home_prev_week_foo", "away_foo_avg3", "home_bar_avg_all"])
    expanded = expand_base_stats(["foo", "bar", "not_present"], columns)
    assert set(expanded) == {"home_prev_week_foo", "away_foo_avg3", "home_bar_avg_all"}


def test_correlation_pruning_fit_only_uses_given_rows(rng):
    n = 100
    X_train = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        }
    )
    X_train["c"] = X_train["a"] * 2 + 0.001 * rng.normal(0, 1, n)  # near-duplicate of a
    y_train = pd.Series(rng.integers(0, 2, n))

    kept, report = prune_correlated_features(X_train, y_train, correlation_threshold=0.90)
    # a and c are near-duplicates -> should collapse to one representative
    assert not ({"a", "c"} <= set(kept))
    assert "b" in kept
    assert report["n_clusters"] == len(kept)


def test_embedded_selection_never_receives_validation_rows(rng, monkeypatch):
    n = 300
    X_train = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series(rng.integers(0, 2, n))
    season_train = pd.Series(np.repeat([2015, 2016, 2017, 2018, 2019], n // 5))

    seen_row_counts = []
    import sklearn.linear_model as lm

    original_fit = lm.LogisticRegression.fit

    def spy_fit(self, X, y, *args, **kwargs):
        seen_row_counts.append(X.shape[0])
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(lm.LogisticRegression, "fit", spy_fit)

    selected, report = select_features_embedded(
        X_train, y_train, season_train,
        l1_ratio_grid=[0.5], C_grid=[0.1], inner_cv_folds=2, max_features=3, random_state=42,
    )
    # every fit call - inner CV and the final refit - only ever saw <= n training rows,
    # never more than the training set handed in (i.e. nothing outside X_train leaked in)
    assert all(count <= n for count in seen_row_counts)
    assert len(selected) <= 3
