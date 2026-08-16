"""Assembles the one-row-per-(athlete_id, game_id) modeling table: targets.py's eligible RB-
game rows (player-grain rolling features + rushing_yards target), LEFT JOINed against team
offensive context (own team) and opponent defensive context (joined via `opponent`, not
`team` -- this is what attaches "the upcoming opponent's run defense" and "opposing team's
time of possession" to a player row) and game context. Feature tables are keyed uniquely by
(team, game_id) or (opponent, game_id); joining them onto the many-athletes-per-team-game
`targets` table is an intentional fan-out, not a bug.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model import eligibility, targets
from cfb_rb_rushing_model.config import DataConfig, FeaturesConfig
from cfb_rb_rushing_model.features import game_context, opponent_defense_context, team_offense_context
from cfb_rb_rushing_model.schedule_spine import attach_rest_days, build_schedule_spine
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COL = targets.TARGET_COL

# Columns that are identifiers, context, or the raw REALIZED same-game outcome (carries,
# explosive_runs, red_zone_carries, success_rate, yards_per_carry are this player's own
# ACTUAL stats from the target game itself -- using them as model inputs would leak the
# target; they exist in the table only for diagnostics/breakdowns, e.g. evaluate_by_breakdown
# on `played`).
NON_FEATURE_COLS = {
    "athlete_id", "game_id", "team", "opponent", "season", "week", "start_date",
    "eligible", "played",
    "rushing_yards", "carries", "explosive_runs", "red_zone_carries", "success_rate", "yards_per_carry",
}

_FEATURE_DESCRIBERS = {
    "eligibility": eligibility.describe_features,
    "team_offense_context": team_offense_context.describe_features,
    "opponent_defense_context": opponent_defense_context.describe_features,
    "game_context": game_context.describe_features,
}


def build_modeling_dataset(engine: Engine, target_seasons: list[int], data_cfg: DataConfig, features_cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info(f"Building modeling dataset for target_seasons={target_seasons}")

    spine = build_schedule_spine(engine, target_seasons)
    spine = attach_rest_days(spine, features_cfg.default_rest_days_season_opener)

    df = targets.build_target_table(engine, spine, target_seasons, data_cfg, features_cfg)
    if df.empty:
        logger.warning(f"No eligible RB-game rows produced for target_seasons={target_seasons}")
        return df

    team_off = team_offense_context.build_team_offense_context_features(engine, spine, target_seasons, features_cfg)
    team_off = team_off.drop(columns=["season", "week", "start_date"])
    df = df.merge(team_off, on=["team", "game_id"], how="left")

    opp_def = opponent_defense_context.build_opponent_defense_context_features(engine, spine, target_seasons, features_cfg)
    opp_def = opp_def.drop(columns=["season", "week", "start_date"]).rename(columns={"team": "opponent"})
    df = df.merge(opp_def, on=["opponent", "game_id"], how="left")

    game_ctx = game_context.build_game_context_features(engine, spine, target_seasons, data_cfg)
    game_ctx = game_ctx.drop(columns=["season", "week"])
    df = df.merge(game_ctx, on=["team", "game_id"], how="left")

    if df.duplicated(subset=["athlete_id", "game_id"]).any():
        raise AssertionError("Duplicate (athlete_id, game_id) rows in modeling dataset -- feature join fan-out bug")

    return df.reset_index(drop=True)


def build_feature_registry(features_cfg: FeaturesConfig, data_cfg: DataConfig) -> pd.DataFrame:
    """Concatenates every feature module's describe_features() output -- the auto-generated,
    code-co-located source of truth for outputs/feature_analysis/feature_registry.csv. Only
    modules whose describe_features() output matches column names ACTUALLY present in the
    final modeling table are registered here -- features/rushing_workload.py's own
    describe_features() documents an intermediate (`_lag1`) representation that eligibility.py
    consumes internally but does not expose in the final table (see eligibility.py's
    docstring for why merge_asof's `_avg3_asof`/`_avg_all_asof` columns are used instead)."""
    rows: list[dict] = []
    for name, describer in _FEATURE_DESCRIBERS.items():
        if name == "game_context":
            rows.extend(describer(include_betting_context=data_cfg.include_betting_context))
        else:
            rows.extend(describer())
    registry = pd.DataFrame(rows)
    column_order = ["feature_name", "description", "source_table", "source_season", "transformation", "known_before_kickoff", "missing_value_treatment", "expected_direction", "category"]
    return registry[column_order]
