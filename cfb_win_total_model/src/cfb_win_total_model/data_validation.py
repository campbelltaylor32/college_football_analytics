"""Data-quality checks run against the raw MySQL cache. Every function is a pure,
DB-facing check (via database.run_query) returning a small DataFrame or dict of findings --
no side effects, no writes. scripts/inspect_database.py is the only place these results get
written to outputs/data_inventory/.

Verified baseline counts (see docs/assumptions_and_limitations.md) are compared against with
a WARNING on drift, not a hard failure, since the DB is expected to grow as ingestion re-runs.
Duplicate-key violations are the one hard-failure case -- those indicate a real integrity bug,
not expected growth.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_win_total_model.database import run_query
from cfb_win_total_model.utils.logging import get_logger

logger = get_logger(__name__)

# Verified live against cfb_football on 2026-08-03 (see docs/assumptions_and_limitations.md).
BASELINE_ROW_COUNTS = {
    "teams": 774,
    "games": 28248,
    "betting_lines": 36443,
    "team_talent": 2275,
    "coaches": 2903,
    "team_rosters": 340633,
    "recruiting_players": 77275,
    "returning_production": 1555,
    "game_team_stats": 21340,
    "plays": 2320740,
}

BASELINE_SEASON_RANGES = {
    "games": (2013, 2025),
    "game_team_stats": (2013, 2025),
    "plays": (2013, 2025),
    "team_talent": (2015, 2025),
    "returning_production": (2014, 2025),
    "coaches": (2004, 2025),
    "team_rosters": (2004, 2025),
}

# One query per verified primary/unique key. A non-empty result is an integrity violation.
PRIMARY_KEYS = {
    "teams": ["school"],
    "games": ["game_id"],
    "betting_lines": ["game_id", "provider"],
    "team_talent": ["season", "school"],
    "coaches": ["first_name", "last_name", "school", "season"],
    "team_rosters": ["athlete_id", "season"],
    "recruiting_players": ["recruit_id"],
    "returning_production": ["season", "team"],
    "game_team_stats": ["game_id", "school"],
    "plays": ["season", "game_id", "play_id"],
}

COVID_SEASON = 2020


def check_row_counts(engine: Engine) -> pd.DataFrame:
    rows = []
    for table, baseline in BASELINE_ROW_COUNTS.items():
        actual = run_query(f"SELECT COUNT(*) AS n FROM {table}", engine=engine)["n"].iloc[0]
        status = "PASS" if actual == baseline else "WARN"
        if actual != baseline:
            logger.warning(f"{table}: row count {actual} differs from baseline {baseline}")
        rows.append({"table": table, "baseline_count": baseline, "actual_count": int(actual), "status": status})
    return pd.DataFrame(rows)


def check_season_ranges(engine: Engine) -> pd.DataFrame:
    rows = []
    for table, (bmin, bmax) in BASELINE_SEASON_RANGES.items():
        season_col = "recruit_year" if table == "recruiting_players" else "season"
        df = run_query(f"SELECT MIN({season_col}) AS mn, MAX({season_col}) AS mx FROM {table}", engine=engine)
        actual_min, actual_max = int(df["mn"].iloc[0]), int(df["mx"].iloc[0])
        status = "PASS" if (actual_min <= bmin and actual_max >= bmax) else "WARN"
        rows.append(
            {
                "table": table,
                "baseline_min": bmin,
                "baseline_max": bmax,
                "actual_min": actual_min,
                "actual_max": actual_max,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def check_duplicate_keys(engine: Engine) -> dict[str, pd.DataFrame]:
    results = {}
    for table, key_cols in PRIMARY_KEYS.items():
        cols = ", ".join(key_cols)
        sql = f"SELECT {cols}, COUNT(*) AS n FROM {table} GROUP BY {cols} HAVING COUNT(*) > 1"
        dupes = run_query(sql, engine=engine)
        if not dupes.empty:
            logger.error(f"{table}: {len(dupes)} duplicate key(s) found on ({cols})")
        results[table] = dupes
    return results


def check_fbs_team_counts_by_season(engine: Engine) -> pd.DataFrame:
    sql = """
        SELECT season, COUNT(DISTINCT school) AS n_fbs_teams FROM (
            SELECT season, home_team AS school FROM games WHERE home_division = 'fbs'
            UNION
            SELECT season, away_team AS school FROM games WHERE away_division = 'fbs'
        ) t
        GROUP BY season ORDER BY season
    """
    return run_query(sql, engine=engine)


def check_game_team_stats_coverage(engine: Engine) -> pd.DataFrame:
    """FBS team-game slots (from games) left-joined to game_team_stats -- reports the
    per-season match rate. Verified >99.8% every season; documented as effectively complete,
    not a real gap."""
    sql = """
        SELECT slots.season, COUNT(*) AS n_slots, SUM(gts.game_id IS NOT NULL) AS n_matched
        FROM (
            SELECT game_id, season, home_team AS school FROM games WHERE home_division = 'fbs'
            UNION ALL
            SELECT game_id, season, away_team AS school FROM games WHERE away_division = 'fbs'
        ) slots
        LEFT JOIN game_team_stats gts
            ON gts.game_id = slots.game_id AND gts.school = slots.school
        GROUP BY slots.season ORDER BY slots.season
    """
    df = run_query(sql, engine=engine)
    df["match_rate"] = df["n_matched"] / df["n_slots"]
    return df


def check_covid_season_flag(engine: Engine) -> pd.DataFrame:
    games_by_season = run_query("SELECT season, COUNT(*) AS n_games FROM games GROUP BY season ORDER BY season", engine=engine)
    plays_by_season = run_query("SELECT season, COUNT(*) AS n_plays FROM plays GROUP BY season ORDER BY season", engine=engine)
    merged = games_by_season.merge(plays_by_season, on="season", how="outer").sort_values("season")
    median_games = merged["n_games"].median()
    merged["is_shortened_season"] = merged["n_games"] < (0.6 * median_games)
    if merged.loc[merged["season"] == COVID_SEASON, "is_shortened_season"].any():
        logger.info(f"Confirmed {COVID_SEASON} as a shortened season (games well below the median).")
    return merged


def check_missingness(engine: Engine, table: str, columns: list[str], season_col: str = "season") -> pd.DataFrame:
    rows = []
    for col in columns:
        sql = f"""
            SELECT {season_col} AS season, COUNT(*) AS n_rows, SUM({col} IS NULL) AS n_null
            FROM {table} GROUP BY {season_col} ORDER BY {season_col}
        """
        df = run_query(sql, engine=engine)
        df["column"] = col
        df["null_rate"] = df["n_null"] / df["n_rows"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def check_roster_year_column_reliability(engine: Engine) -> pd.DataFrame:
    """Flags (season, year) pairs where year == season -- the season-leaked-into-class-year
    bug (verified: ~5.9% of season=2024 rows). This is why roster_turnover.py uses athlete_id
    set differences rather than team_rosters.year as its returning/departed signal."""
    sql = """
        SELECT season, year, COUNT(*) AS n
        FROM team_rosters
        WHERE year IS NOT NULL
        GROUP BY season, year ORDER BY season, year
    """
    df = run_query(sql, engine=engine)
    df["season_leaked_into_year"] = df["season"] == df["year"]
    return df


def run_all_checks(engine: Engine) -> dict:
    """Convenience entry point used by scripts/inspect_database.py."""
    return {
        "row_counts": check_row_counts(engine),
        "season_ranges": check_season_ranges(engine),
        "duplicate_keys": check_duplicate_keys(engine),
        "fbs_team_counts_by_season": check_fbs_team_counts_by_season(engine),
        "game_team_stats_coverage": check_game_team_stats_coverage(engine),
        "covid_season_flag": check_covid_season_flag(engine),
        "roster_year_reliability": check_roster_year_column_reliability(engine),
        "missingness_team_rosters_year": check_missingness(engine, "team_rosters", ["year"]),
        "missingness_coaches_preseason_rank": check_missingness(engine, "coaches", ["preseason_rank"]),
        "missingness_returning_production": check_missingness(
            engine,
            "returning_production",
            ["percent_ppa", "percent_passing_ppa", "percent_receiving_ppa", "percent_rushing_ppa"],
        ),
    }
