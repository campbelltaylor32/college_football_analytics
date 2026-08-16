"""Player-grain rolling rushing-workload features, keyed by (athlete_id, game_id).

Source: player_game_rushing.py (raw plays aggregation) + player_resolution.py (name ->
athlete_id), joined onto `spine` (schedule_spine.build_schedule_spine's output) to get each
game's start_date -- rolling windows are sorted chronologically by start_date, not by
week number, since week numbers can occasionally be out of true chronological order
(rescheduled/makeup games).

`carries_avg3_lag1` (and the season-to-date `carries_avg_all_lag1` cold-start fallback) is
also the direct input to eligibility.py's workload-relevance gate -- computed once here,
consumed twice (as a feature AND as the eligibility signal), so the two can never silently
drift apart.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.config import DataConfig, FeaturesConfig
from cfb_rb_rushing_model.features.rolling_utils import attach_games_played_lag1, compute_rolling_and_lag
from cfb_rb_rushing_model.player_game_rushing import build_raw_player_game_rushing
from cfb_rb_rushing_model.player_resolution import resolve_players
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

VALUE_COLS = [
    "carries", "rushing_yards", "yards_per_carry", "success_rate",
    "explosive_runs", "explosive_run_rate", "stuffed_run_rate",
    "red_zone_carries", "avg_epa_per_rush", "first_down_rate",
]

ID_COLS = ["athlete_id", "game_id", "season", "week", "start_date", "team"]


def build_resolved_player_game_rushing(
    engine: Engine, spine: pd.DataFrame, seasons: list[int], data_cfg: DataConfig
) -> pd.DataFrame:
    """Raw player-game rushing, resolved to athlete_id, with start_date/team attached from
    the spine. Rows that fail to resolve (unmatched/ambiguous) are dropped -- shared by this
    module and targets.py so both see an identical realized-rushing population."""
    raw = build_raw_player_game_rushing(
        engine, seasons, data_cfg.rush_play_types, explosive_run_yard_threshold=15
    )
    if raw.empty:
        return pd.DataFrame(columns=VALUE_COLS + ID_COLS)

    distinct_names = raw[["rusher_player_name", "pos_team", "season"]].drop_duplicates()
    resolved = resolve_players(engine, distinct_names, seasons, data_cfg.positions, data_cfg.name_suffixes_to_strip)
    resolved = resolved[resolved["athlete_id"].notna()]

    merged = raw.merge(resolved[["rusher_player_name", "pos_team", "season", "athlete_id"]], on=["rusher_player_name", "pos_team", "season"], how="inner")
    merged = merged.merge(
        spine[["game_id", "team", "start_date"]], left_on=["game_id", "pos_team"], right_on=["game_id", "team"], how="inner"
    )
    return merged


def build_rushing_workload_rolled(
    engine: Engine, spine: pd.DataFrame, seasons: list[int], data_cfg: DataConfig, features_cfg: FeaturesConfig
) -> pd.DataFrame:
    """Full rolled table -- BOTH the `_lag1` (safe-as-feature) columns AND the raw,
    current-game-inclusive `avg3`/`avg_all` columns. The inclusive columns are not safe to use
    directly as a feature for game_id's own row (that would be lookahead), but they ARE exactly
    what eligibility.py needs: "as of the player's most recently played game" for some FUTURE
    target game means the played game's own inclusive value, reached via merge_asof with
    allow_exact_matches=False. Kept as a separate function (not the public feature output) so
    that distinction is explicit at the call site, not implicit.
    """
    resolved = build_resolved_player_game_rushing(engine, spine, seasons, data_cfg)
    if resolved.empty:
        logger.warning(f"No resolved player-game rushing rows for seasons={seasons}")
        return pd.DataFrame(columns=["athlete_id", "game_id", "team", "season", "week", "start_date"])

    window = features_cfg.player_rolling_windows[0]
    rolled = compute_rolling_and_lag(resolved, group_cols=["athlete_id"], sort_col="start_date", value_cols=VALUE_COLS, window=window)
    rolled = attach_games_played_lag1(rolled, group_cols=["athlete_id"], sort_col="start_date", out_col="career_games_played_lag1")
    return rolled.drop_duplicates(subset=["athlete_id", "game_id"])


def build_rushing_workload_features(
    engine: Engine, spine: pd.DataFrame, seasons: list[int], data_cfg: DataConfig, features_cfg: FeaturesConfig
) -> pd.DataFrame:
    rolled = build_rushing_workload_rolled(engine, spine, seasons, data_cfg, features_cfg)
    if rolled.empty:
        return pd.DataFrame(columns=["athlete_id", "game_id"])

    lag1_cols = [c for c in rolled.columns if c.endswith("_lag1")]
    keep_cols = ["athlete_id", "game_id", "season", "week", "team", "start_date"] + lag1_cols
    return rolled[keep_cols]


def describe_features() -> list[dict]:
    base = {
        "source_table": "plays (rusher_player_name/yds_rushed, resolved to athlete_id via team_rosters)",
        "source_season": "player's own trailing games, strictly before the target game (two-step compute-then-lag)",
        "category": "rushing_workload",
        "known_before_kickoff": True,
        "missing_value_treatment": "NaN for a player's first-ever recorded game (no prior game to roll over) -- median-imputed downstream by cleaning.py",
    }
    rows = []
    for col in VALUE_COLS:
        for suffix, desc in [("avg3_lag1", "trailing-3-game average, through the player's most recent prior game"), ("avg_all_lag1", "season-to-date cumulative average, through the player's most recent prior game")]:
            rows.append({**base, "feature_name": f"{col}_{suffix}", "description": f"{col}: {desc}", "transformation": "rolling/expanding mean, shift(1)", "expected_direction": "+" if col != "stuffed_run_rate" else "-"})
    rows.append({**base, "feature_name": "career_games_played_lag1", "description": "Count of this player's own prior recorded games (cold-start indicator)", "transformation": "cumcount", "expected_direction": "context"})
    return rows
