"""Talent and recruiting features. Sources: team_talent (season t, as-is -- sanctioned
exception, same rationale as returning_production.py) and recruiting_players joined to
team_rosters (season t).

Position-code groupings for QB/OL/DL positional talent were verified live against
`SELECT DISTINCT position FROM recruiting_players` (29 distinct codes) -- CFBD's recruiting
taxonomy splits quarterbacks into PRO (pro-style) and DUAL (dual-threat) rather than a single
QB code, unlike team_rosters' position column which does use a plain "QB". See
config/features.yaml `positional_talent_groups`.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.config import FeaturesConfig
from cfb_win_total_model.database import run_query
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

TRAILING_RECRUITING_CLASSES = 4


def _source_season(target_season: int) -> int:
    """team_talent is a sanctioned as-is exception: source season == target season."""
    return target_season


def _team_talent(engine: Engine, target_season: int) -> pd.DataFrame:
    df = run_query(
        "SELECT season, school, talent FROM team_talent WHERE season = :season",
        params={"season": target_season},
        engine=engine,
    )
    if df.empty:
        return df
    df["talent_zscore"] = (df["talent"] - df["talent"].mean()) / df["talent"].std()
    return df


def _roster_recruiting_join(engine: Engine, target_season: int) -> pd.DataFrame:
    """Recruits currently on the season=t roster -- i.e. recruiting talent still present on
    the team, matching the existing R pipeline's blue-chip-ratio convention
    (R Scripts/Full_CFB_Game_Outcome_Historical.R:210-222)."""
    # Positional-talent groups are defined against recruiting_players.position (the
    # recruiting-time taxonomy, e.g. OT/OG/IOL/OC) rather than team_rosters.position (a
    # coarser roster-side grouping, e.g. plain "OL") -- filter on rp.position for consistency
    # with config/features.yaml's positional_talent_groups.
    sql = """
        SELECT tr.team AS school, tr.athlete_id, rp.position AS recruit_position, rp.stars, rp.rating
        FROM team_rosters tr
        JOIN recruiting_players rp
          ON rp.athlete_id = tr.athlete_id AND rp.committed_to = tr.team
        WHERE tr.season = :season
    """
    return run_query(sql, params={"season": target_season}, engine=engine)


def _blue_chip_and_positional(df: pd.DataFrame, features_cfg: FeaturesConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["school"])

    g = df.groupby("school")
    out = pd.DataFrame(
        {
            "n_distinct_recruited_players": g["athlete_id"].nunique(),
            "blue_chip_ratio": g.apply(lambda x: (x["stars"] >= 4).sum() / x["athlete_id"].nunique()),
            "avg_recruit_rating": g["rating"].mean(),
            "n_5_star": g.apply(lambda x: (x["stars"] == 5).sum()),
            "n_4_star": g.apply(lambda x: (x["stars"] == 4).sum()),
            "pct_4_5_star": g.apply(lambda x: (x["stars"] >= 4).sum() / len(x)),
        }
    ).reset_index()

    groups = features_cfg.positional_talent_groups
    for label, positions in groups.items():
        subset = df[df["recruit_position"].isin(positions)]
        piece = subset.groupby("school")["rating"].mean().rename(f"avg_{label}_recruit_rating")
        out = out.merge(piece, on="school", how="left")

    return out


def _trailing_recruiting_class_rank(engine: Engine, target_season: int) -> pd.DataFrame:
    """Trailing 4-class average recruiting-CLASS rank -- a momentum signal distinct from the
    roster-matched blue-chip ratio above (which reflects current roster composition; this
    reflects recent signing-class quality regardless of attrition). Uses recruit_year in
    [t-4, t-1]: the t-1 class is the most recent class fully known to have signed before
    season t in this conservative construction (a class technically finishes signing in
    Dec/Feb before the season, i.e. recruit_year=t could arguably also qualify, but the
    roster-matched features above already capture who from that class is actually enrolled,
    so this rolling metric deliberately stops at t-1 to avoid double-counting ambiguity)."""
    years = list(range(target_season - TRAILING_RECRUITING_CLASSES, target_season))
    if not years:
        return pd.DataFrame(columns=["school", "recruiting_class_rank_avg_4yr"])
    placeholders = ", ".join(f":y{i}" for i in range(len(years)))
    params = {f"y{i}": y for i, y in enumerate(years)}
    sql = f"""
        SELECT recruit_year, committed_to AS school, AVG(ranking) AS class_avg_rank
        FROM recruiting_players
        WHERE recruit_type = 'HighSchool' AND recruit_year IN ({placeholders}) AND committed_to IS NOT NULL
        GROUP BY recruit_year, committed_to
    """
    by_class = run_query(sql, params=params, engine=engine)
    if by_class.empty:
        return pd.DataFrame(columns=["school", "recruiting_class_rank_avg_4yr"])
    out = by_class.groupby("school", as_index=False)["class_avg_rank"].mean()
    out = out.rename(columns={"class_avg_rank": "recruiting_class_rank_avg_4yr"})
    return out


def build_talent_recruiting_features(engine: Engine, target_season: int, features_cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info(f"Building talent_recruiting features for target_season={target_season} (source season {target_season}, as-is)")

    talent = _team_talent(engine, target_season)
    roster_recruits = _roster_recruiting_join(engine, target_season)
    blue_chip = _blue_chip_and_positional(roster_recruits, features_cfg)
    trailing_rank = _trailing_recruiting_class_rank(engine, target_season)

    # Union the school universe across all three sources first, then left-join each piece
    # onto it -- avoids silently dropping a whole piece's columns (e.g. talent/talent_zscore)
    # when that particular piece happens to be empty (seasons < 2015) while another isn't.
    schools: set[str] = set()
    for piece in (talent, blue_chip, trailing_rank):
        if "school" in piece.columns:
            schools |= set(piece["school"])

    if not schools:
        return pd.DataFrame(columns=["school", "season", "talent_missing"])

    out = pd.DataFrame({"school": sorted(schools)})
    for piece in (talent, blue_chip, trailing_rank):
        if not piece.empty:
            out = out.merge(piece, on="school", how="left")

    out["season"] = target_season
    out["talent_missing"] = out["talent"].isna() if "talent" in out.columns else True
    return out


def describe_features() -> list[dict]:
    base = {
        "source_table": "team_talent",
        "source_season": "t (sanctioned exception -- preseason-known for t itself)",
        "category": "talent_recruiting",
        "known_before_kickoff": True,
        "missing_value_treatment": "talent_missing flag; no rows before season 2015",
    }
    roster_base = {**base, "source_table": "recruiting_players JOIN team_rosters", "missing_value_treatment": "0 if no recruited players resolve to the roster"}
    rows = [
        {**base, "feature_name": "talent", "description": "Team talent composite for season t", "transformation": "as-is", "expected_direction": "+"},
        {**base, "feature_name": "talent_zscore", "description": "Within-season z-score of talent (recreates old R pipeline's Scaled_Talent)", "transformation": "(x-mean)/std within season", "expected_direction": "+"},
        {**roster_base, "feature_name": "n_distinct_recruited_players", "description": "Count of season-t roster players with a recruiting record", "transformation": "count distinct", "expected_direction": "context"},
        {**roster_base, "feature_name": "blue_chip_ratio", "description": "Share of recruited roster players rated 4-5 stars (R pipeline formula)", "transformation": "sum(stars>=4)/n_distinct(athlete_id)", "expected_direction": "+"},
        {**roster_base, "feature_name": "avg_recruit_rating", "description": "Avg recruiting rating of roster players with a recruiting record", "transformation": "mean", "expected_direction": "+"},
        {**roster_base, "feature_name": "n_5_star", "description": "Count of 5-star recruits on season-t roster", "transformation": "count", "expected_direction": "+"},
        {**roster_base, "feature_name": "n_4_star", "description": "Count of 4-star recruits on season-t roster", "transformation": "count", "expected_direction": "+"},
        {**roster_base, "feature_name": "pct_4_5_star", "description": "Share of recruited roster rows rated 4+ stars", "transformation": "mean", "expected_direction": "+"},
        {**roster_base, "feature_name": "avg_qb_recruit_rating", "description": "Avg recruiting rating, QB/PRO/DUAL positions on roster", "transformation": "mean", "expected_direction": "+"},
        {**roster_base, "feature_name": "avg_ol_recruit_rating", "description": "Avg recruiting rating, OT/OG/IOL/OC positions on roster", "transformation": "mean", "expected_direction": "+"},
        {**roster_base, "feature_name": "avg_dl_recruit_rating", "description": "Avg recruiting rating, DT/DL/SDE/WDE/EDGE positions on roster", "transformation": "mean", "expected_direction": "+"},
        {
            "feature_name": "recruiting_class_rank_avg_4yr",
            "description": "Trailing 4-class (t-4..t-1) average recruiting-class ranking",
            "source_table": "recruiting_players",
            "source_season": "t-4 through t-1",
            "transformation": "mean of per-class AVG(ranking)",
            "known_before_kickoff": True,
            "missing_value_treatment": "NaN if no classes signed in window",
            "expected_direction": "- (lower rank number is better)",
            "category": "talent_recruiting",
        },
        {**base, "feature_name": "talent_missing", "description": "True if no team_talent row exists (seasons < 2015)", "transformation": "isna flag", "expected_direction": "context"},
    ]
    return rows
