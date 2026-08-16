"""Prior-year team performance features. Source: games + game_team_stats + plays, season
**t-1** relative to the target season t. This is the largest and most leakage-sensitive
feature module -- every row is built exclusively from games completed before season t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.database import run_query
from cfb_win_total_model.targets import build_target_table
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

NON_SCORING_PLAY_TYPES = ("Kickoff", "Timeout", "End Period", "End of Half", "End of Game")


def _source_season(target_season: int) -> int:
    return target_season - 1


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def _pull_game_team_stats(engine: Engine, season: int) -> pd.DataFrame:
    sql = """
        SELECT gts.game_id, gts.school, gts.opponent, gts.home_away,
               gts.points, gts.points_allowed, gts.total_yards, gts.total_yards_allowed,
               gts.attempted_passes, gts.rushing_attempts,
               gts.third_down_conversion, gts.third_down_attempts,
               gts.fourth_down_conversion, gts.fourth_down_attempts,
               gts.total_penalties, gts.penalty_yards,
               gts.tackles_for_loss, gts.sacks, gts.interceptions, gts.passes_deflected,
               gts.fumbles_recovered, gts.turnovers,
               gts.punt_return_yards, gts.punt_returns, gts.kick_return_yards, gts.kick_returns,
               gts.kicking_points,
               g.conference_game
        FROM game_team_stats gts
        JOIN games g ON g.game_id = gts.game_id
        WHERE gts.season = :season
    """
    return run_query(sql, params={"season": season}, engine=engine)


def _self_join_opponent_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs each team-game row with its opponent's row for the same game, so opponent
    offensive-play counts (needed as denominators for havoc rate / yards-allowed-per-play)
    are available without a second DB round trip."""
    paired = df.merge(df, on="game_id", suffixes=("", "_opp"))
    paired = paired[paired["school"] != paired["school_opp"]].copy()
    paired["offensive_plays"] = paired["attempted_passes"] + paired["rushing_attempts"]
    paired["opp_offensive_plays"] = paired["attempted_passes_opp"] + paired["rushing_attempts_opp"]
    return paired


def _aggregate_game_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("school")
    out = pd.DataFrame(
        {
            "games_played": g["game_id"].count(),
            "points_per_game": g["points"].mean(),
            "points_allowed_per_game": g["points_allowed"].mean(),
            "yards_per_play": _safe_div(g["total_yards"].sum(), g["offensive_plays"].sum()),
            "yards_per_play_allowed": _safe_div(g["total_yards_allowed"].sum(), g["opp_offensive_plays"].sum()),
            "third_down_pct": _safe_div(g["third_down_conversion"].sum(), g["third_down_attempts"].sum()),
            "fourth_down_pct": _safe_div(g["fourth_down_conversion"].sum(), g["fourth_down_attempts"].sum()),
            "penalty_yards_per_game": g["penalty_yards"].mean(),
            "penalties_per_game": g["total_penalties"].mean(),
            "sack_rate": _safe_div(g["sacks"].sum(), g["opp_offensive_plays"].sum()),
            "havoc_rate": _safe_div(
                g["tackles_for_loss"].sum() + g["interceptions"].sum() + g["passes_deflected"].sum() + g["fumbles_recovered"].sum(),
                g["opp_offensive_plays"].sum(),
            ),
            "turnover_margin_per_game": (
                (df["interceptions"] + df["fumbles_recovered"] - df["turnovers"]).groupby(df["school"]).mean()
            ),
            "punt_return_avg": _safe_div(g["punt_return_yards"].sum(), g["punt_returns"].sum()),
            "kick_return_avg": _safe_div(g["kick_return_yards"].sum(), g["kick_returns"].sum()),
            "kicking_points_per_game": g["kicking_points"].mean(),
        }
    ).reset_index()
    out["point_diff_per_game"] = out["points_per_game"] - out["points_allowed_per_game"]
    return out


def _home_road_splits(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, home_away in (("home", "home"), ("road", "away")):
        subset = df[df["home_away"] == home_away]
        g = subset.groupby("school")
        piece = pd.DataFrame(
            {
                f"points_per_game_{label}": g["points"].mean(),
                f"points_allowed_per_game_{label}": g["points_allowed"].mean(),
            }
        )
        piece[f"point_diff_per_game_{label}"] = piece[f"points_per_game_{label}"] - piece[f"points_allowed_per_game_{label}"]
        frames.append(piece)
    return pd.concat(frames, axis=1).reset_index()


def _conference_performance(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["conference_game"] == 1].copy()
    subset["conf_win"] = subset["points"] > subset["points_allowed"]
    return subset.groupby("school", as_index=False).agg(
        conf_games_played=("game_id", "count"), conf_win_pct=("conf_win", "mean")
    )


def _strength_of_schedule(engine: Engine, df: pd.DataFrame, season_t1: int) -> pd.DataFrame:
    """Opponent strength using opponents' PRIOR season (t-2, i.e. their season before t-1)
    win% and opponents' own t-1 talent (talent is preseason-known for the opponent's own
    season, so no additional lag is needed there)."""
    opponents = df[["school", "opponent"]].rename(columns={"opponent": "opp_school"})

    opp_win_pct = build_target_table(engine, seasons=[season_t1 - 1])
    opp_win_pct = opp_win_pct.rename(columns={"school": "opp_school"})
    opp_win_pct["opp_prior_win_pct"] = opp_win_pct["regular_season_wins"] / opp_win_pct["scheduled_games"]

    opp_talent = run_query(
        "SELECT school AS opp_school, talent AS opp_talent FROM team_talent WHERE season = :season",
        params={"season": season_t1},
        engine=engine,
    )

    merged = opponents.merge(opp_win_pct[["opp_school", "opp_prior_win_pct"]], on="opp_school", how="left")
    merged = merged.merge(opp_talent, on="opp_school", how="left")

    return merged.groupby("school", as_index=False).agg(
        sos_avg_opponent_prior_win_pct=("opp_prior_win_pct", "mean"),
        sos_avg_opponent_prior_talent=("opp_talent", "mean"),
    )


def _pull_plays(engine: Engine, season: int) -> pd.DataFrame:
    placeholders = ", ".join(f":pt{i}" for i in range(len(NON_SCORING_PLAY_TYPES)))
    params = {f"pt{i}": pt for i, pt in enumerate(NON_SCORING_PLAY_TYPES)}
    params["season"] = season
    sql = f"""
        SELECT pos_team, def_pos_team, drive_id, epa, success, rz_play, touchdown
        FROM plays
        WHERE season = :season AND pos_team IS NOT NULL
          AND play_type NOT IN ({placeholders})
    """
    return run_query(sql, params=params, engine=engine)


def _epa_features(plays: pd.DataFrame, explosiveness_threshold: float) -> pd.DataFrame:
    off = plays.groupby("pos_team").agg(
        off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"),
    )
    off["off_explosiveness_rate"] = plays.groupby("pos_team")["epa"].apply(lambda s: (s > explosiveness_threshold).mean())

    defense = plays.groupby("def_pos_team").agg(
        def_epa_per_play=("epa", "mean"),
        def_success_rate_allowed=("success", "mean"),
    )

    rz = plays[plays["rz_play"] == 1]
    rz_drive_td = rz.groupby(["pos_team", "drive_id"])["touchdown"].max()
    red_zone_td_rate = rz_drive_td.groupby("pos_team").mean().rename("red_zone_td_rate")

    combined = off.join(defense, how="outer").join(red_zone_td_rate, how="left")
    combined.index.name = "school"
    return combined.reset_index()


def build_prior_performance_features(engine: Engine, target_season: int, explosiveness_threshold: float = 1.0) -> pd.DataFrame:
    season_t1 = _source_season(target_season)
    logger.info(f"Building prior_performance features for target_season={target_season} (source season {season_t1})")

    gts = _pull_game_team_stats(engine, season_t1)
    if gts.empty:
        logger.warning(f"No game_team_stats rows for season={season_t1}; returning empty frame")
        return pd.DataFrame(columns=["school", "season"])

    paired = _self_join_opponent_plays(gts)
    agg = _aggregate_game_team_stats(paired)
    splits = _home_road_splits(gts)
    conf = _conference_performance(gts)
    sos = _strength_of_schedule(engine, gts, season_t1)

    plays = _pull_plays(engine, season_t1)
    epa = _epa_features(plays, explosiveness_threshold)

    out = agg.merge(splits, on="school", how="left")
    out = out.merge(conf, on="school", how="left")
    out = out.merge(sos, on="school", how="left")
    out = out.merge(epa, on="school", how="left")
    out.insert(1, "season", target_season)
    return out


def describe_features() -> list[dict]:
    base = {"source_table": "game_team_stats + games", "source_season": "t-1", "category": "prior_performance", "missing_value_treatment": "median-impute within season"}
    plays_base = {**base, "source_table": "plays"}
    rows = [
        {**base, "feature_name": "games_played", "description": "Games played in season t-1", "transformation": "count", "known_before_kickoff": True, "expected_direction": "context"},
        {**base, "feature_name": "points_per_game", "description": "Points scored per game, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "points_allowed_per_game", "description": "Points allowed per game, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**base, "feature_name": "point_diff_per_game", "description": "Point differential per game, t-1", "transformation": "points_per_game - points_allowed_per_game", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "yards_per_play", "description": "Offensive yards per play, t-1 (plays proxied by attempted_passes+rushing_attempts)", "transformation": "sum(yards)/sum(plays)", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "yards_per_play_allowed", "description": "Defensive yards allowed per play, t-1", "transformation": "sum(yards_allowed)/sum(opp_plays)", "known_before_kickoff": True, "expected_direction": "-"},
        {**base, "feature_name": "third_down_pct", "description": "3rd down conversion rate, t-1", "transformation": "sum(conv)/sum(att)", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "fourth_down_pct", "description": "4th down conversion rate, t-1", "transformation": "sum(conv)/sum(att)", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "penalty_yards_per_game", "description": "Penalty yards per game, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**base, "feature_name": "penalties_per_game", "description": "Penalties per game, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**base, "feature_name": "sack_rate", "description": "Sacks per opponent offensive play, t-1", "transformation": "sum(sacks)/sum(opp_plays)", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "havoc_rate", "description": "(TFL+INT+PBU+FR)/opponent offensive plays, t-1", "transformation": "ratio", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "turnover_margin_per_game", "description": "(INT+FR-turnovers) per game, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "punt_return_avg", "description": "Punt return yards per return, t-1", "transformation": "sum/sum", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "kick_return_avg", "description": "Kick return yards per return, t-1", "transformation": "sum/sum", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "kicking_points_per_game", "description": "Kicking (FG+XP) points per game, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "points_per_game_home", "description": "Points scored per game, home only, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "points_allowed_per_game_home", "description": "Points allowed per game, home only, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**base, "feature_name": "point_diff_per_game_home", "description": "Point differential, home only, t-1", "transformation": "diff", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "points_per_game_road", "description": "Points scored per game, road only, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "points_allowed_per_game_road", "description": "Points allowed per game, road only, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**base, "feature_name": "point_diff_per_game_road", "description": "Point differential, road only, t-1", "transformation": "diff", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "conf_games_played", "description": "Conference games played, t-1", "transformation": "count", "known_before_kickoff": True, "expected_direction": "context"},
        {**base, "feature_name": "conf_win_pct", "description": "Win pct in conference games, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "sos_avg_opponent_prior_win_pct", "description": "Strength of season t-1's schedule: avg t-1 opponents' win pct in THEIR prior season (t-2)", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**base, "feature_name": "sos_avg_opponent_prior_talent", "description": "Strength of season t-1's schedule: avg t-1 opponents' own talent composite (their season t-1)", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**plays_base, "feature_name": "off_epa_per_play", "description": "Offensive EPA per play, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**plays_base, "feature_name": "off_success_rate", "description": "Offensive success rate, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
        {**plays_base, "feature_name": "off_explosiveness_rate", "description": "Share of offensive plays with EPA above threshold, t-1", "transformation": "mean(epa>threshold)", "known_before_kickoff": True, "expected_direction": "+"},
        {**plays_base, "feature_name": "def_epa_per_play", "description": "Defensive EPA/play allowed, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**plays_base, "feature_name": "def_success_rate_allowed", "description": "Defensive success rate allowed, t-1", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "-"},
        {**plays_base, "feature_name": "red_zone_td_rate", "description": "Share of red-zone drives ending in a TD, t-1 (approximate: drive-level max(touchdown) among rz_play=1 rows)", "transformation": "mean", "known_before_kickoff": True, "expected_direction": "+"},
    ]
    return rows
