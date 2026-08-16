from __future__ import annotations

import dataclasses

from cfb_spread_model.data import build_feature_matrix, get_feature_columns


def test_real_dataset_shape_matches_config(real_dataset, data_cfg):
    assert len(real_dataset) >= data_cfg.expected_row_count_min
    assert len(real_dataset.columns) == data_cfg.expected_column_count


def test_id_and_label_columns_present(real_dataset, data_cfg):
    for col in data_cfg.id_columns + [data_cfg.label_column] + data_cfg.split_only_columns:
        assert col in real_dataset.columns


def test_get_feature_columns_excludes_ids_label_and_split_columns(real_dataset, data_cfg):
    feature_cols = get_feature_columns(list(real_dataset.columns), data_cfg)
    for col in data_cfg.id_columns:
        assert col not in feature_cols
    assert data_cfg.label_column not in feature_cols
    if not data_cfg.include_split_columns_as_features:
        for col in data_cfg.split_only_columns:
            assert col not in feature_cols


def test_build_feature_matrix_shapes(real_dataset, data_cfg):
    X, y = build_feature_matrix(real_dataset, data_cfg)
    assert len(X) == len(y) == len(real_dataset)
    assert data_cfg.label_column not in X.columns


def test_get_feature_columns_excludes_configured_patterns(real_dataset, data_cfg):
    """config/data.yaml excludes prev_week_* (single-game predictors) -- see the inline comment
    there and docs/project_story.md for why."""
    assert "prev_week_" in data_cfg.excluded_column_patterns
    feature_cols = get_feature_columns(list(real_dataset.columns), data_cfg)
    assert not any("prev_week_" in c for c in feature_cols)
    # Sanity check the exclusion actually did something, not just that it was configured --
    # the real CSV has 168 prev_week_* columns per side (336 total), verified this build.
    prev_week_in_raw = [c for c in real_dataset.columns if "prev_week_" in c]
    assert len(prev_week_in_raw) > 0


def test_differential_representation_on_real_data(real_dataset, data_cfg):
    """config/data.yaml's feature_representation toggle -- verified against the real CSV, not
    just the synthetic fixture (see test_feature_engineering.py for the unit-level checks)."""
    differential_cfg = dataclasses.replace(data_cfg, feature_representation="differential")
    X, y = build_feature_matrix(real_dataset, differential_cfg)

    diff_cols = [c for c in X.columns if c.startswith("diff_")]
    trend_cols = [c for c in X.columns if c.startswith("trend_")]
    raw_avg_cols = [
        c for c in X.columns if (c.startswith("home_") or c.startswith("away_")) and ("avg_all" in c or "avg3" in c)
    ]
    assert len(diff_cols) == 354
    assert len(trend_cols) == 336
    assert raw_avg_cols == []  # fully replaced, not just supplemented

    # Spot-check correctness directly against the raw CSV for one differential and one trend column.
    row = real_dataset.iloc[1]
    expected_diff = row["home_points_avg_all"] - row["away_points_avg_all"]
    assert X.iloc[1]["diff_avg_all_points"] == expected_diff

    expected_trend = row["home_points_avg3"] - row["home_points_avg_all"]
    assert X.iloc[1]["trend_home_points"] == expected_trend

    # raw_dual must be unaffected by the differential toggle -- confirms the toggle doesn't leak
    # state between configs. Built explicitly via dataclasses.replace rather than relying on
    # data_cfg's ambient value, since config/data.yaml's actual feature_representation setting
    # changes across experiments (see docs/project_story.md "Run 3") and this test must hold
    # regardless of what it's currently set to.
    raw_dual_cfg = dataclasses.replace(data_cfg, feature_representation="raw_dual")
    X_raw, _ = build_feature_matrix(real_dataset, raw_dual_cfg)
    assert any(c.startswith("home_") and "avg_all" in c for c in X_raw.columns)
    assert not any(c.startswith("diff_") for c in X_raw.columns)
