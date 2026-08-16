"""Coaching features. Source: coaches. Leakage-sensitive -- see docs/data_leakage_rules.md.

Verified data-quality issue: the most recent season in a coach's tenure can have
games=0/wins=0 (a not-yet-finalized "stub" row) even while sp_overall (a preseason
projection) is already populated. career_win_pct_entering_t therefore filters to
`season < t AND games > 0`, which by construction also excludes the season=t row itself.

Leakage nuance: wins/losses/games/srs/sp_overall/sp_offense/sp_defense for season=t reflect
that season's ACTUAL outcome (computed post-hoc) -- reading them for season=t would leak
(wins is literally derivable from the target). The ONE place a season=t coaches row is
touched is to read the coach's NAME (who currently holds the job), which is public/
preseason-known information, used only for the coaching_change_indicator.

Identity keys: (first_name, last_name) is used for career win-pct accumulation (so it tracks
a coach across school changes); (school, season) is used for tenure/first-year/change
indicators (school-specific).

OC/DC change is explicitly OMITTED -- no assistant-coach table exists in this schema.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.config import FeaturesConfig
from cfb_win_total_model.database import run_query
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

DB_FLOOR_SEASON = 2004


def _source_season(target_season: int) -> int:
    """Primary numeric-stat source season for career/prior-season/SP+ features. The one
    exception is coaching_change_indicator, which also reads the season=t coach-of-record's
    NAME (never season=t wins/games/sp_overall) -- see module docstring."""
    return target_season - 1


def _pull_coaches_history(engine: Engine, target_season: int) -> pd.DataFrame:
    sql = """
        SELECT first_name, last_name, school, season, hire_date, games, wins, losses,
               preseason_rank, sp_overall, sp_offense, sp_defense
        FROM coaches WHERE season <= :target_season
        ORDER BY hire_date ASC
    """
    df = run_query(sql, params={"target_season": target_season}, engine=engine)
    df["games"] = df["games"].fillna(0)
    return df


def _coach_of_record(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (school, season): the coach with the most games that season. Sorted by
    hire_date ascending beforehand so ties (e.g. two games=0 stub rows) resolve to the
    earliest-hired coach, a stable tiebreak."""
    idx = df.groupby(["school", "season"])["games"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_coaching_features(engine: Engine, target_season: int, features_cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info(f"Building coaching features for target_season={target_season}")
    season_t1 = target_season - 1

    history = _pull_coaches_history(engine, target_season)
    if history.empty:
        return pd.DataFrame(columns=["school", "season"])

    cor = _coach_of_record(history)
    cor["coach_name"] = cor["first_name"].fillna("") + " " + cor["last_name"].fillna("")

    finalized = history[history["games"] > 0].copy()

    # Career win pct entering t: cumulative record across all seasons < t with games > 0,
    # grouped by coach identity so it follows the coach across school changes.
    prior = finalized[finalized["season"] < target_season]
    career = prior.groupby(["first_name", "last_name"], as_index=False).agg(
        career_games_entering_t=("games", "sum"), career_wins_entering_t=("wins", "sum")
    )
    career["career_win_pct_entering_t"] = career["career_wins_entering_t"] / career["career_games_entering_t"]

    cor_t = cor[cor["season"] == target_season][["school", "first_name", "last_name", "coach_name"]]
    cor_t1 = cor[cor["season"] == season_t1][["school", "coach_name"]].rename(columns={"coach_name": "coach_name_t1"})

    out = cor_t.merge(career, on=["first_name", "last_name"], how="left")
    out = out.merge(cor_t1, on="school", how="left")
    out["coaching_change_indicator"] = out["coach_name_t1"].notna() & (out["coach_name"] != out["coach_name_t1"])

    # Tenure at current school: earliest season (<= t) this coach identity appears as coach
    # of record for THIS school.
    tenure_start = (
        cor.merge(out[["school", "first_name", "last_name"]], on=["school", "first_name", "last_name"])
        .groupby("school")["season"]
        .min()
        .rename("tenure_start_season")
    )
    out = out.merge(tenure_start, on="school", how="left")
    out["tenure_length_at_school"] = target_season - out["tenure_start_season"]
    out["tenure_left_censored"] = out["tenure_start_season"] <= DB_FLOOR_SEASON
    out["first_year_hc_indicator"] = out["tenure_length_at_school"] == 0

    # Prior-season record + SP+ entering t: from the season=t-1 COACH-OF-RECORD row only
    # (cor, not the raw finalized frame) -- a school with a mid-season coaching change in t-1
    # has two finalized (games>0) rows for that season (e.g. interim + permanent), and using
    # the raw frame directly would fan out the merge below into duplicate (school,season)
    # output rows. games==0 stub rows still yield NaN, not a phantom 0-win season.
    cor_t1_stats = cor[(cor["season"] == season_t1) & (cor["games"] > 0)]
    prior_season = cor_t1_stats[["school", "wins", "losses", "sp_overall", "sp_offense", "sp_defense"]].rename(
        columns={
            "wins": "prior_season_wins",
            "losses": "prior_season_losses",
            "sp_overall": "sp_overall_entering_t",
            "sp_offense": "sp_offense_entering_t",
            "sp_defense": "sp_defense_entering_t",
        }
    )
    out = out.merge(prior_season, on="school", how="left")

    if features_cfg.use_coach_preseason_rank:
        rank_t = cor[cor["season"] == target_season][["school", "preseason_rank"]]
        out = out.merge(rank_t, on="school", how="left")
        out["coach_preseason_rank_missing"] = out["preseason_rank"].isna()

    out["season"] = target_season
    out = out.drop(columns=["first_name", "last_name", "coach_name", "coach_name_t1", "career_wins_entering_t", "tenure_start_season"])
    return out


def describe_features() -> list[dict]:
    base = {"source_table": "coaches", "category": "coaching", "known_before_kickoff": True, "missing_value_treatment": "NaN + left as missing if coach history is left-censored or t-1 row is an unfinalized stub"}
    rows = [
        {**base, "feature_name": "career_win_pct_entering_t", "description": "Coach's cumulative career win pct over all seasons < t with games>0, tracked by (first_name,last_name) across school changes", "source_season": "<t", "transformation": "sum(wins)/sum(games)", "expected_direction": "+"},
        {**base, "feature_name": "career_games_entering_t", "description": "Coach's cumulative career games entering t", "source_season": "<t", "transformation": "sum", "expected_direction": "context"},
        {**base, "feature_name": "coaching_change_indicator", "description": "True if the coach-of-record's name differs between t-1 and t (reads only the season=t NAME, never season=t wins/games/sp_overall)", "source_season": "name at t vs t-1", "transformation": "name comparison", "expected_direction": "-"},
        {**base, "feature_name": "tenure_length_at_school", "description": "Seasons this coach has been at the current school entering t", "source_season": "<=t (identity only)", "transformation": "t - first season as coach of record", "expected_direction": "+"},
        {**base, "feature_name": "tenure_left_censored", "description": "True if tenure_start_season == 2004 (DB floor) -- true tenure may be longer", "source_season": "<=t", "transformation": "flag", "expected_direction": "context"},
        {**base, "feature_name": "first_year_hc_indicator", "description": "True if tenure_length_at_school == 0", "source_season": "<=t", "transformation": "flag", "expected_direction": "-"},
        {**base, "feature_name": "prior_season_wins", "description": "School's coach-of-record wins in season t-1 (only if finalized, games>0)", "source_season": "t-1", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "prior_season_losses", "description": "School's coach-of-record losses in season t-1", "source_season": "t-1", "transformation": "as-is", "expected_direction": "-"},
        {**base, "feature_name": "sp_overall_entering_t", "description": "Team SP+ overall rating, season t-1 (never season t, which reflects t's own outcome)", "source_season": "t-1", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "sp_offense_entering_t", "description": "Team SP+ offense rating, season t-1", "source_season": "t-1", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "sp_defense_entering_t", "description": "Team SP+ defense rating, season t-1", "source_season": "t-1", "transformation": "as-is", "expected_direction": "-"},
    ]
    return rows
