from __future__ import annotations

import pytest

from cfb_win_total_model.dataset import NON_FEATURE_COLS, build_feature_registry, build_modeling_dataset
from cfb_win_total_model.targets import get_fbs_teams_by_season


def test_row_count_matches_verified_fbs_team_count(engine, features_cfg):
    df = build_modeling_dataset(engine, target_seasons=[2022], features_cfg=features_cfg)
    assert len(df) == len(get_fbs_teams_by_season(engine, 2022)) == 131


def test_school_season_uniqueness(full_modeling_df):
    assert not full_modeling_df.duplicated(subset=["school", "season"]).any()


def test_wins_within_scheduled_games_bounds(full_modeling_df):
    assert (full_modeling_df["regular_season_wins"] >= 0).all()
    assert (full_modeling_df["regular_season_wins"] <= full_modeling_df["scheduled_games"]).all()


def test_wins_plus_losses_equals_scheduled_games(full_modeling_df):
    """Confirms no ties or incomplete games slipped into the target (post-2000s CFB has no
    ties; see targets.py's tie-warning check)."""
    assert (
        full_modeling_df["regular_season_wins"] + full_modeling_df["regular_season_losses"] == full_modeling_df["scheduled_games"]
    ).all()


def test_feature_registry_matches_dataset_columns_exactly(full_modeling_df, features_cfg):
    registry = build_feature_registry(features_cfg)
    reg_names = set(registry["feature_name"])
    df_cols = set(full_modeling_df.columns) - NON_FEATURE_COLS
    assert df_cols - reg_names == set(), f"columns missing from registry: {df_cols - reg_names}"
    assert reg_names - df_cols == set(), f"registry entries with no matching column: {reg_names - df_cols}"


def test_below_floor_season_raises_or_flags_missing(engine, features_cfg):
    """2013 predates team_talent (2015) and returning_production (2014) -- the per-module
    functions stay permissive (they just return NaN/empty), but scripts/build_modeling_dataset.py
    gates target_seasons >= full_feature_start_season. This test documents the permissive
    module-level behavior; the gate itself is exercised by running the script directly."""
    df = build_modeling_dataset(engine, target_seasons=[2013], features_cfg=features_cfg)
    assert not df.empty
    assert df["talent_missing"].all()
    assert df["returning_production_missing"].all()


def test_build_script_rejects_seasons_below_floor():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_modeling_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--target-seasons", "2013"],
        capture_output=True,
        text=True,
        cwd=script.parents[1],
    )
    assert result.returncode != 0
    assert "full_feature_start_season" in result.stderr
