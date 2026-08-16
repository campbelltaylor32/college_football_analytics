"""Roster turnover / transfer-activity features. Source: team_rosters, seasons **t-1 and t**.

No dedicated transfer-portal table exists in this schema. Transfers are INFERRED from
athlete_id roster-membership set differences year-over-year:
  - an athlete_id on team A's t-1 roster but absent from ANY t roster       -> attrition_unknown
    (grayshirt, decommit, exhausted eligibility, injury retirement, and true transfers whose
    destination roster we failed to match are all indistinguishable from this signal alone)
  - an athlete_id on team A's t-1 roster and on a DIFFERENT team's t roster -> transferred_out
    (symmetrically, transferred_in on the receiving side)
This is an approximate proxy, not ground truth -- documented explicitly in
docs/assumptions_and_limitations.md. team_rosters.year (eligibility class) is NOT used as the
returning/departed signal because it is verified dirty (~5.9% of season=2024 rows have the
season number leaked into the class-year field instead of 1-5).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.database import run_query
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)


def _source_seasons(target_season: int) -> tuple[int, int]:
    return target_season - 1, target_season


def _pull_rosters(engine: Engine, season_t1: int, season_t: int) -> pd.DataFrame:
    sql = """
        SELECT athlete_id, team, position, season
        FROM team_rosters
        WHERE season IN (:t1, :t)
    """
    return run_query(sql, params={"t1": season_t1, "t": season_t}, engine=engine)


def _recruit_ratings(engine: Engine) -> pd.DataFrame:
    """athlete_id -> recruiting rating, for the ~74% of roster players who resolve to a
    recruiting record (documented caveat inherited from recruiting_players)."""
    sql = "SELECT athlete_id, AVG(rating) AS rating FROM recruiting_players WHERE athlete_id IS NOT NULL GROUP BY athlete_id"
    return run_query(sql, engine=engine)


def build_roster_turnover_features(engine: Engine, target_season: int) -> pd.DataFrame:
    season_t1, season_t = _source_seasons(target_season)
    logger.info(f"Building roster_turnover features for target_season={target_season} (seasons {season_t1}, {season_t})")

    rosters = _pull_rosters(engine, season_t1, season_t)
    if rosters.empty:
        return pd.DataFrame(columns=["school", "season"])

    prev = rosters[rosters["season"] == season_t1].drop_duplicates("athlete_id", keep="first")
    curr = rosters[rosters["season"] == season_t].drop_duplicates("athlete_id", keep="first")
    prev = prev[["athlete_id", "team", "position"]].rename(columns={"team": "prev_team", "position": "prev_position"})
    curr = curr[["athlete_id", "team"]].rename(columns={"team": "curr_team"})

    merged = prev.merge(curr, on="athlete_id", how="outer")

    merged["returning"] = merged["prev_team"].notna() & (merged["prev_team"] == merged["curr_team"])
    merged["departed"] = merged["prev_team"].notna() & ~merged["returning"]
    merged["incoming"] = merged["curr_team"].notna() & ~merged["returning"]
    merged["transferred_out"] = merged["departed"] & merged["curr_team"].notna()
    merged["transferred_in"] = merged["incoming"] & merged["prev_team"].notna()

    ratings = _recruit_ratings(engine)
    merged = merged.merge(ratings, on="athlete_id", how="left")

    schools = pd.Index(sorted(set(merged["prev_team"].dropna()) | set(merged["curr_team"].dropna())), name="school")

    roster_prev_size = prev.groupby("prev_team").size().reindex(schools, fill_value=0)
    roster_curr_size = curr.groupby("curr_team").size().reindex(schools, fill_value=0)

    departed_rows = merged[merged["departed"]]
    n_departed = departed_rows.groupby("prev_team").size().reindex(schools, fill_value=0)
    n_transferred_out = departed_rows[departed_rows["transferred_out"]].groupby("prev_team").size().reindex(schools, fill_value=0)
    qb_departure = (
        departed_rows[departed_rows["prev_position"] == "QB"].groupby("prev_team").size().reindex(schools, fill_value=0) > 0
    )
    transferred_out_talent = (
        departed_rows[departed_rows["transferred_out"]].groupby("prev_team")["rating"].sum().reindex(schools, fill_value=0.0)
    )

    returning_rows = merged[merged["returning"]]
    n_returning = returning_rows.groupby("prev_team").size().reindex(schools, fill_value=0)

    incoming_rows = merged[merged["incoming"]]
    n_incoming = incoming_rows.groupby("curr_team").size().reindex(schools, fill_value=0)
    n_transferred_in = incoming_rows[incoming_rows["transferred_in"]].groupby("curr_team").size().reindex(schools, fill_value=0)
    transferred_in_talent = (
        incoming_rows[incoming_rows["transferred_in"]].groupby("curr_team")["rating"].sum().reindex(schools, fill_value=0.0)
    )

    out = pd.DataFrame(
        {
            "roster_size_prev": roster_prev_size,
            "roster_size_curr": roster_curr_size,
            "n_returning_players": n_returning,
            "n_departed_players": n_departed,
            "n_transferred_out": n_transferred_out,
            "n_incoming_players": n_incoming,
            "n_transferred_in": n_transferred_in,
            "qb_departure_indicator": qb_departure,
            "net_transfer_talent": transferred_in_talent - transferred_out_talent,
        }
    ).reset_index()

    out["returning_pct"] = out["n_returning_players"] / out["roster_size_prev"].replace(0, pd.NA)
    out["net_roster_turnover_pct"] = out["n_departed_players"] / out["roster_size_prev"].replace(0, pd.NA)
    out["season"] = target_season
    return out


def describe_features() -> list[dict]:
    base = {
        "source_table": "team_rosters (athlete_id set differences, t-1 vs t)",
        "source_season": "t-1 and t",
        "category": "roster_turnover",
        "known_before_kickoff": True,
        "missing_value_treatment": "0 for count columns if a team is missing from one season entirely",
    }
    rows = [
        {**base, "feature_name": "roster_size_prev", "description": "Roster size, season t-1", "transformation": "count distinct athlete_id", "expected_direction": "context"},
        {**base, "feature_name": "roster_size_curr", "description": "Roster size, season t", "transformation": "count distinct athlete_id", "expected_direction": "context"},
        {**base, "feature_name": "n_returning_players", "description": "Players on both t-1 and t roster for the same school", "transformation": "set intersection", "expected_direction": "+"},
        {**base, "feature_name": "n_departed_players", "description": "Players on t-1 roster, absent from this school's t roster", "transformation": "set difference", "expected_direction": "-"},
        {**base, "feature_name": "n_transferred_out", "description": "Departed players who appear on a DIFFERENT school's t roster (approximate transfer signal)", "transformation": "set difference + cross-school match", "expected_direction": "-"},
        {**base, "feature_name": "n_incoming_players", "description": "Players on t roster, absent from this school's t-1 roster", "transformation": "set difference", "expected_direction": "context"},
        {**base, "feature_name": "n_transferred_in", "description": "Incoming players who were on a DIFFERENT school's t-1 roster (approximate transfer signal)", "transformation": "set difference + cross-school match", "expected_direction": "+"},
        {**base, "feature_name": "qb_departure_indicator", "description": "Any QB (team_rosters.position='QB') among departed players -- flags ANY QB departure, not necessarily the starter", "transformation": "any()", "expected_direction": "-"},
        {**base, "feature_name": "net_transfer_talent", "description": "Sum of incoming transfers' recruiting rating minus outgoing transfers' rating (only for the ~74% who resolve to a recruiting record)", "transformation": "sum(rating_in) - sum(rating_out)", "expected_direction": "+"},
        {**base, "feature_name": "returning_pct", "description": "n_returning_players / roster_size_prev", "transformation": "ratio", "expected_direction": "+"},
        {**base, "feature_name": "net_roster_turnover_pct", "description": "n_departed_players / roster_size_prev", "transformation": "ratio", "expected_direction": "context"},
    ]
    return rows
