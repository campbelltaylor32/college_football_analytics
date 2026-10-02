"""Assemble the play-level modeling dataset: one row per non-garbage-time run/pass call by an
FBS offense, with every feature group joined on."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_playcall_predictability.config import DataConfig, FeaturesConfig
from cfb_playcall_predictability.context import (
    load_games, load_head_coaches, load_lines, team_games as build_team_games,
)
from cfb_playcall_predictability.features.in_game import in_game_features, shrink_in_game
from cfb_playcall_predictability.features.situation import market_features, situation_features
from cfb_playcall_predictability.features.tendencies import opponent_defense, team_tendencies
from cfb_playcall_predictability.plays import build_calls, load_pbp
from cfb_playcall_predictability.utils.logging import get_logger

logger = get_logger(__name__)

ID_COLUMNS = [
    "season", "week", "game_id", "play_seq", "pos_team", "def_pos_team", "head_coach",
    "down", "distance", "yards_to_goal", "period", "pos_score_diff_start", "play_type", "play_text",
]
TARGET = "is_pass"

# Small, league-wide model: the "any coach in this spot" expectation.
SITUATIONAL_FEATURES = [
    "down", "distance", "yards_to_goal", "goal_to_go", "period", "half_secs_rem",
    "game_secs_rem", "score_diff", "score_diff_x_elapsed",
]


SITUATION_FEATURES = [
    "down", "distance", "yards_to_goal", "goal_to_go", "red_zone", "period", "half_secs_rem",
    "game_secs_rem", "two_minute", "score_diff", "score_diff_x_elapsed", "score_diff_per_poss_left",
    "wp_before", "off_timeouts", "def_timeouts", "receives_2h_kickoff", "drive_play_number", "is_home",
]
MARKET_FEATURES = ["exp_margin", "game_total", "implied_team_total", "neutral_site"]
IN_GAME_FEATURES = [
    "g_calls_so_far", "prev_call_pass", "prev_call_success", "prev_call_yards",
    "prev_call_same_drive", "prev_streak_len", "drive_calls_so_far", "drive_passes_so_far",
    "g_pass_rate", "g_pass_rate_vs_std", "g_run_sr", "g_pass_sr", "g_run_epa", "g_pass_epa",
    "g_pass_minus_run_epa", "g_ypc",
]
TEAM_FEATURES = ["new_head_coach", "std_calls_so_far"]
# Tendency / efficiency / opponent columns are generated per config bucket, so match by prefix.
FEATURE_PREFIXES = ("prior_pr_", "std_pr_", "off_prior_", "off_std_", "def_prior_", "def_std_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Explicit whitelist of model features. Raw play columns (EPA, success, yards_gained,
    play_type, ...) describe the play's OUTCOME and must never be features."""
    named = SITUATION_FEATURES + MARKET_FEATURES + IN_GAME_FEATURES + TEAM_FEATURES
    prefixed = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    return [c for c in named if c in df.columns] + prefixed


def build_modeling_dataset(
    seasons: list[int], data_cfg: DataConfig, feat_cfg: FeaturesConfig, engine: Engine | None = None
) -> pd.DataFrame:
    # Tendency priors need the season before the first modeled one.
    pbp = load_pbp(seasons)
    calls = build_calls(pbp, data_cfg)

    games = load_games(seasons, engine=engine)
    tg = build_team_games(games)
    lines = load_lines(seasons, data_cfg, engine=engine)
    coaches = load_head_coaches(seasons, engine=engine)

    n_before = len(calls)
    calls = calls[calls["game_id"].isin(games["game_id"])].reset_index(drop=True)
    logger.info(f"Dropped {n_before - len(calls):,} calls from games not in MySQL `games`")

    # Features that look at the whole game history are computed before any row filtering.
    feats = pd.concat([calls, situation_features(calls), in_game_features(calls)], axis=1)
    feats = feats.loc[:, ~feats.columns.duplicated(keep="last")]

    tend = team_tendencies(calls, tg, feat_cfg)
    opp = opponent_defense(calls, tg, feat_cfg)
    feats = feats.merge(tend, left_on=["game_id", "pos_team"], right_on=["game_id", "team"], how="left").drop(columns="team")
    feats = feats.merge(opp, on=["game_id", "def_pos_team"], how="left")
    feats = feats.merge(games[["game_id", "neutral_site"]], on="game_id", how="left")
    feats = feats.merge(lines, on="game_id", how="left")
    feats = pd.concat([feats, market_features(feats)], axis=1)
    feats = feats.merge(coaches, left_on=["pos_team", "season"], right_on=["team", "season"], how="left").drop(columns="team")
    feats["new_head_coach"] = feats["new_head_coach"].fillna(1)
    feats = pd.concat([feats, shrink_in_game(feats, feat_cfg.shrinkage["in_game_pass_rate"],
                                             feat_cfg.shrinkage["in_game_efficiency"])], axis=1)

    # Model rows: FBS offenses, non-garbage time.
    fbs = tg.loc[tg["is_fbs"], ["game_id", "team"]].rename(columns={"team": "pos_team"})
    feats = feats.merge(fbs.assign(_fbs=1), on=["game_id", "pos_team"], how="left")
    feats = feats[(feats["_fbs"] == 1) & ~feats["garbage_time"]].drop(columns="_fbs")

    keep = list(dict.fromkeys(ID_COLUMNS + [TARGET] + feature_columns(feats)))
    out = feats[keep].reset_index(drop=True)
    logger.info(f"Modeling dataset: {len(out):,} calls, {len(feature_columns(out))} features")
    return out
