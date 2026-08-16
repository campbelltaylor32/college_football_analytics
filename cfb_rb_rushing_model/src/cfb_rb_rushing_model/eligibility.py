"""Workload-relevance eligibility gate AND the authoritative source of player-grain rolling
features for the modeling table.

The row universe here is every (RB roster spot) x (team-game) combination for a season --
i.e. every game an RB's team played, whether or not that RB recorded a carry in it (byes,
injuries, personnel packages, and future/not-yet-played games all produce a row with no
matching player_game_rushing record). For each such row, `pandas.merge_asof(..., backward,
allow_exact_matches=False)` looks up that PLAYER's most recently PLAYED game strictly before
the target game's date, and carries forward that game's own (inclusive) rolling averages.

This is deliberately NOT the same as joining features/rushing_workload.py's `_lag1` columns by
game_id: those only apply to a row where the player themselves recorded a carry THAT game, and
represent "through the row's own prior game." A target game reached after a bye week, an
injury-missed game, or simply being asked to predict several weeks out needs "through
whatever this player's most recent PLAYED game actually was" -- which is what merge_asof
gives. `carries_avg3_asof`/`carries_avg_all_asof` (an inclusive, current-game rolling value AS
OF that last-played game) is both a model feature and the direct input to the eligibility
gate below -- computed once, consumed twice, so the two can never silently drift apart.

A player with NO prior played game at all (true debut / transfer) gets `eligible=False` and
NaN features by construction -- documented in docs/assumptions_and_limitations.md as a real,
unavoidable limitation, not something this design tries to work around.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.config import DataConfig, EligibilityConfig, FeaturesConfig
from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.features.rushing_workload import VALUE_COLS, build_rushing_workload_rolled
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)


def _pull_position_rosters(engine: Engine, seasons: list[int], positions: list[str]) -> pd.DataFrame:
    if not seasons:
        return pd.DataFrame(columns=["athlete_id", "team", "season"])
    season_placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
    pos_placeholders = ", ".join(f":p{i}" for i in range(len(positions)))
    params = {f"s{i}": s for i, s in enumerate(seasons)}
    params.update({f"p{i}": p for i, p in enumerate(positions)})
    sql = f"""
        SELECT DISTINCT athlete_id, team, season FROM team_rosters
        WHERE season IN ({season_placeholders}) AND position IN ({pos_placeholders})
    """
    return run_query(sql, params=params, engine=engine)


def compute_eligibility_from_asof_source(
    candidates: pd.DataFrame, asof_source: pd.DataFrame, eligibility_cfg: EligibilityConfig
) -> pd.DataFrame:
    """Pure, DB-independent core of the eligibility computation -- merge_asof + threshold
    rule only. Split out from build_eligibility_spine specifically so it can be unit-tested
    against small synthetic DataFrames (tests/test_eligibility.py) without a live database.

    candidates: one row per (athlete_id, team, game_id, start_date) candidate to evaluate.
    asof_source: one row per (athlete_id, start_date) PLAYED game, with the current-game-
    inclusive `{col}_avg3`/`{col}_avg_all` columns (VALUE_COLS) and `career_games_played_lag1`.
    """
    candidates_sorted = candidates.sort_values("start_date")
    asof_source_sorted = asof_source.sort_values("start_date")
    asof = pd.merge_asof(
        candidates_sorted, asof_source_sorted, on="start_date", by="athlete_id", direction="backward", allow_exact_matches=False
    )

    asof["prior_games_played"] = asof["career_games_played_lag1"] + 1
    asof.loc[asof["career_games_played_lag1"].isna(), "prior_games_played"] = 0
    asof = asof.drop(columns=["career_games_played_lag1"])

    rename_map = {f"{col}_avg3": f"{col}_avg3_asof" for col in VALUE_COLS}
    rename_map.update({f"{col}_avg_all": f"{col}_avg_all_asof" for col in VALUE_COLS})
    asof = asof.rename(columns=rename_map)

    has_history = asof["prior_games_played"] > 0
    enough_for_avg3 = has_history & (asof["prior_games_played"] >= eligibility_cfg.min_games_played_for_avg3)
    early_season = has_history & ~enough_for_avg3

    total_season_to_date_carries = asof["carries_avg_all_asof"] * asof["prior_games_played"]

    asof["eligible"] = False
    asof.loc[enough_for_avg3, "eligible"] = asof.loc[enough_for_avg3, "carries_avg3_asof"] >= eligibility_cfg.min_trailing3_avg_carries
    asof.loc[early_season, "eligible"] = total_season_to_date_carries.loc[early_season] >= eligibility_cfg.min_season_to_date_carries

    return asof.reset_index(drop=True)


def build_eligibility_spine(
    engine: Engine, spine: pd.DataFrame, seasons: list[int], data_cfg: DataConfig, features_cfg: FeaturesConfig
) -> pd.DataFrame:
    rosters = _pull_position_rosters(engine, seasons, data_cfg.positions)
    if rosters.empty:
        return pd.DataFrame()

    candidates = spine[["team", "opponent", "game_id", "season", "week", "start_date"]].drop_duplicates()
    candidates = candidates.merge(rosters, on=["team", "season"], how="inner")

    workload = build_rushing_workload_rolled(engine, spine, seasons, data_cfg, features_cfg)
    if workload.empty:
        candidates["eligible"] = False
        for col in VALUE_COLS:
            candidates[f"{col}_avg3_asof"] = pd.NA
            candidates[f"{col}_avg_all_asof"] = pd.NA
        candidates["prior_games_played"] = 0
        return candidates

    asof_source_cols = ["athlete_id", "start_date"] + [f"{col}_avg3" for col in VALUE_COLS] + [f"{col}_avg_all" for col in VALUE_COLS] + ["career_games_played_lag1"]
    asof_source = workload[asof_source_cols]

    return compute_eligibility_from_asof_source(candidates, asof_source, features_cfg.eligibility)


def describe_features() -> list[dict]:
    base = {
        "source_table": "features/rushing_workload.py, via merge_asof against the player's most recently PLAYED game",
        "source_season": "player's most recent played game strictly before the target game (merge_asof, allow_exact_matches=False)",
        "category": "eligibility",
        "known_before_kickoff": True,
        "missing_value_treatment": "NaN / eligible=False if the player has no prior played game (debut/transfer -- documented limitation)",
    }
    rows = [{**base, "feature_name": f"{col}_avg3_asof", "description": f"{col}, trailing-3-game average as of the player's last played game", "transformation": "merge_asof", "expected_direction": "context"} for col in VALUE_COLS]
    rows += [{**base, "feature_name": f"{col}_avg_all_asof", "description": f"{col}, season-to-date average as of the player's last played game", "transformation": "merge_asof", "expected_direction": "context"} for col in VALUE_COLS]
    rows.append({**base, "feature_name": "prior_games_played", "description": "Count of this player's own games played at-or-before the matched as-of game", "transformation": "merge_asof + 1", "expected_direction": "context"})
    rows.append({**base, "feature_name": "eligible", "description": "Workload-relevance gate: True if carries_avg3_asof or early-season total carries clears the config threshold (features.yaml eligibility)", "transformation": "threshold rule", "expected_direction": "n/a"})
    return rows
