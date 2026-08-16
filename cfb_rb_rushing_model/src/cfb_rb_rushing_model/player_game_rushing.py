"""Raw player-game rushing table, built fresh from `plays` -- no such table exists anywhere
else in the DB or the legacy CSV pipeline. Keyed by (rusher_player_name, pos_team, season,
week, game_id) -- name-keyed, not athlete_id-keyed. Resolution to athlete_id happens in
player_resolution.py, deliberately kept separate so this module has no roster-schema
knowledge and can be unit-tested independently.

Verified live (see docs/assumptions_and_limitations.md): `yards_gained` is 0% NULL on rush
plays across every season 2014-2025, while `yds_rushed` has a small but nonzero NULL rate
(up to 43 rows in 2024) -- yards_gained is used for rushing_yards for this reason.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

RAW_COLUMNS = [
    "rusher_player_name", "pos_team", "season", "week", "game_id",
    "carries", "rushing_yards", "yards_per_carry", "success_rate",
    "explosive_runs", "explosive_run_rate", "stuffed_run_rate",
    "red_zone_carries", "avg_epa_per_rush", "first_down_rate",
]


def _pull_rush_plays(engine: Engine, seasons: list[int], rush_play_types: list[str]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame()
    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    type_placeholders = ", ".join(f":t{i}" for i in range(len(rush_play_types)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    params.update({f"t{i}": t for i, t in enumerate(rush_play_types)})
    sql = f"""
        SELECT rusher_player_name, pos_team, season, week, game_id,
               yards_gained, success, rz_play, stuffed_run, epa, down, distance
        FROM plays
        WHERE season IN ({season_placeholders})
          AND play_type IN ({type_placeholders})
          AND rusher_player_name IS NOT NULL
          AND pos_team IS NOT NULL
    """
    return run_query(sql, params=params, engine=engine)


def build_raw_player_game_rushing(
    engine: Engine, seasons: list[int], rush_play_types: list[str], explosive_run_yard_threshold: int
) -> pd.DataFrame:
    """One row per (rusher_player_name, pos_team, season, week, game_id). `first_down_rate`
    is a documented approximation (yards_gained >= distance) -- `plays` has no explicit
    first-down-achieved flag, so this proxy slightly overcounts penalty-aided first downs and
    undercounts spot-adjustment edge cases (see docs/data_dictionary.md)."""
    raw = _pull_rush_plays(engine, seasons, rush_play_types)
    if raw.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    raw["is_explosive"] = raw["yards_gained"] >= explosive_run_yard_threshold
    raw["is_first_down"] = raw["yards_gained"] >= raw["distance"].fillna(10)

    grouped = raw.groupby(["rusher_player_name", "pos_team", "season", "week", "game_id"], as_index=False).agg(
        carries=("yards_gained", "count"),
        rushing_yards=("yards_gained", "sum"),
        success_rate=("success", "mean"),
        explosive_runs=("is_explosive", "sum"),
        stuffed_run_rate=("stuffed_run", "mean"),
        red_zone_carries=("rz_play", "sum"),
        avg_epa_per_rush=("epa", "mean"),
        first_down_rate=("is_first_down", "mean"),
    )
    grouped["yards_per_carry"] = grouped["rushing_yards"] / grouped["carries"]
    grouped["explosive_run_rate"] = grouped["explosive_runs"] / grouped["carries"]
    grouped["red_zone_carries"] = grouped["red_zone_carries"].astype(int)
    grouped["explosive_runs"] = grouped["explosive_runs"].astype(int)

    return grouped[RAW_COLUMNS]


def describe_features() -> list[dict]:
    base = {
        "source_table": "plays (rush/rushing-touchdown play_types only)",
        "source_season": "realized game outcome -- only ever used as raw material for rolling/lagged features, never a same-game feature",
        "category": "player_game_rushing",
        "known_before_kickoff": False,
        "missing_value_treatment": "n/a -- this is the raw realized table, not a feature table",
    }
    names = [
        ("carries", "Rush attempts in this player-game"),
        ("rushing_yards", "Sum of yards_gained across this player-game's rush attempts"),
        ("yards_per_carry", "rushing_yards / carries"),
        ("success_rate", "Mean of plays.success across this player-game's carries"),
        ("explosive_runs", "Count of carries with yards_gained >= config explosive_run_yard_threshold"),
        ("explosive_run_rate", "explosive_runs / carries"),
        ("stuffed_run_rate", "Mean of plays.stuffed_run across this player-game's carries"),
        ("red_zone_carries", "Count of carries flagged plays.rz_play"),
        ("avg_epa_per_rush", "Mean of plays.epa across this player-game's carries"),
        ("first_down_rate", "Approximation: share of carries with yards_gained >= distance (plays has no explicit first-down-achieved flag)"),
    ]
    return [{**base, "feature_name": n, "description": d, "transformation": "n/a", "expected_direction": "n/a"} for n, d in names]
