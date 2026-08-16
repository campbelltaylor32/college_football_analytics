"""Schedule-difficulty features. Source: games for season **t itself** -- opponent identity,
dates, and conference membership only, NEVER season-t results (scores/win-loss are never
read here). Opponent-strength aggregates use the opponent's own **t-1** data (their prior
season's win%, talent, returning production), never the opponent's season-t performance.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.config import FeaturesConfig
from cfb_win_total_model.database import run_query
from cfb_win_total_model.targets import build_target_table
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

BYE_GAP_DAYS = 10
SHORT_REST_GAP_DAYS = 6


def _opponent_strength_source_season(target_season: int) -> int:
    """Opponent-strength columns (avg_opponent_prior_*) use the opponent's t-1 season.
    Schedule STRUCTURE columns (n_games, dates, conference membership) legitimately use
    season=t itself -- opponent identity and dates are known before kickoff, only opponent
    RESULTS would be leakage, and those are never read here."""
    return target_season - 1


def _pull_schedule(engine: Engine, target_season: int) -> pd.DataFrame:
    sql = """
        SELECT game_id, week, start_date, neutral_site, conference_game,
               home_team, home_division, home_conference,
               away_team, away_division, away_conference
        FROM games
        WHERE season = :season AND (home_division = 'fbs' OR away_division = 'fbs')
    """
    return run_query(sql, params={"season": target_season}, engine=engine)


def _stack_team_perspective(games: pd.DataFrame) -> pd.DataFrame:
    home = games.rename(
        columns={
            "home_team": "school",
            "home_division": "division",
            "away_team": "opponent",
            "away_division": "opponent_division",
            "away_conference": "opponent_conference",
        }
    ).assign(is_home=True)
    away = games.rename(
        columns={
            "away_team": "school",
            "away_division": "division",
            "home_team": "opponent",
            "home_division": "opponent_division",
            "home_conference": "opponent_conference",
        }
    ).assign(is_home=False)
    cols = ["game_id", "week", "start_date", "neutral_site", "conference_game", "school", "division", "opponent", "opponent_division", "opponent_conference", "is_home"]
    stacked = pd.concat([home[cols], away[cols]], ignore_index=True)
    stacked = stacked[stacked["division"] == "fbs"].copy()
    stacked["is_road"] = (~stacked["is_home"]) & (~stacked["neutral_site"].astype(bool))
    stacked["is_home"] = stacked["is_home"] & (~stacked["neutral_site"].astype(bool))
    return stacked


def _opponent_strength_lookups(engine: Engine, season_t1: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opp_win_pct = build_target_table(engine, seasons=[season_t1])
    opp_win_pct["opp_prior_win_pct"] = opp_win_pct["regular_season_wins"] / opp_win_pct["scheduled_games"]
    opp_win_pct = opp_win_pct.rename(columns={"school": "opponent"})[["opponent", "opp_prior_win_pct"]]

    opp_talent = run_query(
        "SELECT school AS opponent, talent AS opp_prior_talent FROM team_talent WHERE season = :season",
        params={"season": season_t1},
        engine=engine,
    )

    opp_returning = run_query(
        "SELECT team AS opponent, percent_ppa AS opp_prior_returning_pct_ppa FROM returning_production WHERE season = :season",
        params={"season": season_t1},
        engine=engine,
    )
    return opp_win_pct, opp_talent, opp_returning


def _rest_and_travel(stacked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for school, group in stacked.sort_values("start_date").groupby("school"):
        dates = group["start_date"]
        gaps = dates.diff().dt.days
        bye_week_count = int((gaps > BYE_GAP_DAYS).sum())
        short_rest_count = int((gaps < SHORT_REST_GAP_DAYS).sum())

        back_to_back_road = 0
        consecutive_road = 0
        for is_road in group["is_road"]:
            if is_road:
                consecutive_road += 1
                if consecutive_road >= 2:
                    back_to_back_road += 1
            else:
                consecutive_road = 0

        rows.append(
            {
                "school": school,
                "bye_week_count": bye_week_count,
                "short_rest_count": short_rest_count,
                "back_to_back_road_count": back_to_back_road,
            }
        )
    return pd.DataFrame(rows)


def build_schedule_features(engine: Engine, target_season: int, features_cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info(f"Building schedule features for target_season={target_season} (opponent identity: season {target_season}, opponent strength: season {target_season - 1})")
    season_t1 = target_season - 1

    games = _pull_schedule(engine, target_season)
    if games.empty:
        logger.warning(f"No games rows for season={target_season}; schedule cannot be built (expected if this season's schedule hasn't been ingested yet)")
        return pd.DataFrame(columns=["school", "season"])

    stacked = _stack_team_perspective(games)

    stacked["is_power_opponent"] = stacked.apply(
        lambda r: features_cfg.is_power_conference_opponent(r["opponent_conference"], target_season, school=r["opponent"]), axis=1
    )
    stacked["is_sub_fbs_opponent"] = stacked["opponent_division"] != "fbs"
    stacked["is_group_of_5_opponent"] = (~stacked["is_power_opponent"]) & (~stacked["is_sub_fbs_opponent"])

    opp_win_pct, opp_talent, opp_returning = _opponent_strength_lookups(engine, season_t1)
    stacked = stacked.merge(opp_win_pct, on="opponent", how="left")
    stacked = stacked.merge(opp_talent, on="opponent", how="left")
    stacked = stacked.merge(opp_returning, on="opponent", how="left")

    own_talent = run_query(
        "SELECT school, talent AS own_talent FROM team_talent WHERE season = :season",
        params={"season": target_season},
        engine=engine,
    )
    stacked = stacked.merge(own_talent, on="school", how="left")
    stacked["opponent_above_own_talent"] = stacked["opp_prior_talent"] > stacked["own_talent"]

    early = stacked[stacked["week"] <= features_cfg.early_season_max_week]
    late = stacked[stacked["week"] >= features_cfg.late_season_min_week]
    conf_games = stacked[stacked["conference_game"] == 1]
    nonconf_games = stacked[stacked["conference_game"] == 0]

    g = stacked.groupby("school")
    out = pd.DataFrame(
        {
            "n_games": g["game_id"].count(),
            "n_home": g["is_home"].sum(),
            "n_road": g["is_road"].sum(),
            "n_neutral": g["neutral_site"].sum(),
            "n_conference_games": g["conference_game"].sum(),
            "n_power_opponents": g["is_power_opponent"].sum(),
            "n_group_of_5_opponents": g["is_group_of_5_opponent"].sum(),
            "n_sub_fbs_opponents": g["is_sub_fbs_opponent"].sum(),
            "avg_opponent_prior_win_pct": g["opp_prior_win_pct"].mean(),
            "avg_opponent_prior_talent": g["opp_prior_talent"].mean(),
            "max_opponent_prior_talent": g["opp_prior_talent"].max(),
            "avg_opponent_prior_returning_pct_ppa": g["opp_prior_returning_pct_ppa"].mean(),
            "n_opponents_above_own_talent": g["opponent_above_own_talent"].sum(),
        }
    ).reset_index()
    out["n_nonconference_games"] = out["n_games"] - out["n_conference_games"]

    out = out.merge(early.groupby("school")["is_power_opponent"].sum().rename("n_power_opponents_early"), on="school", how="left")
    out = out.merge(late.groupby("school")["is_power_opponent"].sum().rename("n_power_opponents_late"), on="school", how="left")
    out = out.merge(conf_games.groupby("school")["opp_prior_talent"].mean().rename("avg_opponent_prior_talent_conference"), on="school", how="left")
    out = out.merge(nonconf_games.groupby("school")["opp_prior_talent"].mean().rename("avg_opponent_prior_talent_nonconference"), on="school", how="left")

    rest = _rest_and_travel(stacked)
    out = out.merge(rest, on="school", how="left")

    out["season"] = target_season
    return out


def describe_features() -> list[dict]:
    base = {
        "source_table": "games (opponent identity/dates only)",
        "source_season": "t (identity/schedule structure only, never t results)",
        "category": "schedule",
        "known_before_kickoff": True,
        "missing_value_treatment": "NaN if opponent-strength lookup table lacks a t-1 row for that opponent",
    }
    opp_strength_note = {**base, "source_season": "opponent identity: t; opponent strength values: t-1"}
    rows = [
        {**base, "feature_name": "n_games", "description": "Scheduled FBS games, season t", "transformation": "count", "expected_direction": "context"},
        {**base, "feature_name": "n_home", "description": "Home games, season t", "transformation": "count", "expected_direction": "+"},
        {**base, "feature_name": "n_road", "description": "Road games, season t", "transformation": "count", "expected_direction": "-"},
        {**base, "feature_name": "n_neutral", "description": "Neutral-site games, season t", "transformation": "count", "expected_direction": "context"},
        {**base, "feature_name": "n_conference_games", "description": "Conference games, season t", "transformation": "count", "expected_direction": "context"},
        {**base, "feature_name": "n_nonconference_games", "description": "Non-conference games, season t", "transformation": "n_games - n_conference_games", "expected_direction": "context"},
        {**base, "feature_name": "n_power_opponents", "description": "Opponents in a Power conference (season-aware mapping, config/features.yaml)", "transformation": "count", "expected_direction": "-"},
        {**base, "feature_name": "n_group_of_5_opponents", "description": "FBS opponents not in a Power conference", "transformation": "count", "expected_direction": "+"},
        {**base, "feature_name": "n_sub_fbs_opponents", "description": "Opponents below FBS (FCS/II/III)", "transformation": "count", "expected_direction": "+"},
        {**opp_strength_note, "feature_name": "avg_opponent_prior_win_pct", "description": "Avg opponent win pct, opponents' season t-1", "transformation": "mean", "expected_direction": "-"},
        {**opp_strength_note, "feature_name": "avg_opponent_prior_talent", "description": "Avg opponent talent composite, opponents' season t-1", "transformation": "mean", "expected_direction": "-"},
        {**opp_strength_note, "feature_name": "max_opponent_prior_talent", "description": "Max opponent talent composite, opponents' season t-1", "transformation": "max", "expected_direction": "-"},
        {**opp_strength_note, "feature_name": "avg_opponent_prior_returning_pct_ppa", "description": "Avg opponent returning-production share, opponents' season t-1", "transformation": "mean", "expected_direction": "-"},
        {**opp_strength_note, "feature_name": "n_opponents_above_own_talent", "description": "Count of opponents whose t-1 talent exceeds own season-t talent", "transformation": "count", "expected_direction": "-"},
        {**base, "feature_name": "n_power_opponents_early", "description": f"Power opponents in weeks <= early_season_max_week", "transformation": "count", "expected_direction": "context"},
        {**base, "feature_name": "n_power_opponents_late", "description": f"Power opponents in weeks >= late_season_min_week", "transformation": "count", "expected_direction": "context"},
        {**opp_strength_note, "feature_name": "avg_opponent_prior_talent_conference", "description": "Avg opponent t-1 talent, conference games only", "transformation": "mean", "expected_direction": "-"},
        {**opp_strength_note, "feature_name": "avg_opponent_prior_talent_nonconference", "description": "Avg opponent t-1 talent, nonconference games only", "transformation": "mean", "expected_direction": "-"},
        {**base, "feature_name": "bye_week_count", "description": "Gaps > 10 days between consecutive games, season t", "transformation": "start_date diff", "expected_direction": "+"},
        {**base, "feature_name": "short_rest_count", "description": "Gaps < 6 days between consecutive games, season t", "transformation": "start_date diff", "expected_direction": "-"},
        {**base, "feature_name": "back_to_back_road_count", "description": "Road games immediately following another road game, season t", "transformation": "stateful scan", "expected_direction": "-"},
    ]
    return rows
