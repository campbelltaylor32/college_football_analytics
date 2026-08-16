"""Data-quality checks run against the raw MySQL cache. Every function is a pure, DB-facing
check (via database.run_query) returning a small DataFrame -- no side effects, no writes.
scripts/inspect_database.py is the only place these results get written to
outputs/data_inventory/.

Baseline row counts are the same verified numbers documented in the sibling
cfb_win_total_model/cfb_spread_model projects' data_validation.py (same underlying database),
compared with a WARNING on drift, not a hard failure. The rusher-name-completeness check is
new to this project -- see docs/assumptions_and_limitations.md for the full 2025
ingestion-gap investigation these baselines are grounded in.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from cfb_rb_rushing_model.database import run_query
from cfb_rb_rushing_model.player_game_rushing import build_raw_player_game_rushing
from cfb_rb_rushing_model.player_resolution import resolve_players
from cfb_rb_rushing_model.utils.logging import get_logger

logger = get_logger(__name__)

# Verified live against cfb_football during planning (2026-08-09) -- see docs/assumptions_and_limitations.md.
BASELINE_ROW_COUNTS = {
    "teams": 774,
    "games": 28248,
    "team_rosters": 340633,
    "game_team_stats": 21340,
    "plays": 2320740,
}

# Per-season NULL rate of plays.rusher_player_name on rush plays (play_type IN
# ('Rush','Rushing Touchdown')). Verified live 2026-08-09 -- see docs/assumptions_and_limitations.md.
# 2013 is 100% NULL (unusable), 2025 collapses to ~97-99.5% NULL from week 9 on (a live,
# time-bound ingestion gap). This is the actual binding constraint behind
# config/modeling.yaml's full_feature_start_season=2014 and excluded_seasons=[2020, 2025].
MAX_ACCEPTABLE_RUSHER_NAME_NULL_RATE = 0.10


def check_row_counts(engine: Engine) -> pd.DataFrame:
    rows = []
    for table, baseline in BASELINE_ROW_COUNTS.items():
        actual = run_query(f"SELECT COUNT(*) AS n FROM {table}", engine=engine)["n"].iloc[0]
        status = "PASS" if actual >= baseline else "WARN"
        if actual < baseline:
            logger.warning(f"{table}: row count {actual} is BELOW baseline {baseline}")
        rows.append({"table": table, "baseline_count": baseline, "actual_count": int(actual), "status": status})
    return pd.DataFrame(rows)


def check_duplicate_keys(engine: Engine) -> dict[str, pd.DataFrame]:
    primary_keys = {
        "teams": ["school"],
        "games": ["game_id"],
        "team_rosters": ["athlete_id", "season"],
        "game_team_stats": ["game_id", "school"],
        "plays": ["season", "game_id", "play_id"],
    }
    results = {}
    for table, key_cols in primary_keys.items():
        cols = ", ".join(key_cols)
        sql = f"SELECT {cols}, COUNT(*) AS n FROM {table} GROUP BY {cols} HAVING COUNT(*) > 1"
        dupes = run_query(sql, engine=engine)
        if not dupes.empty:
            logger.error(f"{table}: {len(dupes)} duplicate key(s) found on ({cols})")
        results[table] = dupes
    return results


def check_rusher_name_completeness(engine: Engine, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """Per-season (or per-week, if season+week given) NULL rate of plays.rusher_player_name on
    rush plays. This is the hard gate wired into scripts/generate_week_predictions.py --
    a NULL rate above MAX_ACCEPTABLE_RUSHER_NAME_NULL_RATE means the upstream ingestion for
    that season/week silently broke (exactly what happened for 2025 weeks 9+, discovered
    during planning), and any prediction built on it would be scored from a near-empty
    rushing population, not a genuine data gap the model can be expected to handle."""
    group_col = "week" if (season is not None and week is None) else "season"
    filters = []
    params: dict = {}
    if season is not None:
        filters.append("season = :season")
        params["season"] = season
    if week is not None:
        filters.append("week = :week")
        params["week"] = week
    where = f"WHERE play_type IN ('Rush','Rushing Touchdown') AND ({' AND '.join(filters)})" if filters else "WHERE play_type IN ('Rush','Rushing Touchdown')"

    sql = f"""
        SELECT {group_col}, COUNT(*) AS n, SUM(rusher_player_name IS NULL) AS n_null
        FROM plays {where}
        GROUP BY {group_col} ORDER BY {group_col}
    """
    df = run_query(sql, params=params, engine=engine)
    df["null_rate"] = df["n_null"] / df["n"]
    df["status"] = df["null_rate"].apply(lambda r: "FAIL" if r > MAX_ACCEPTABLE_RUSHER_NAME_NULL_RATE else "PASS")
    return df


def check_player_resolution_match_rate(engine: Engine, season: int, positions: list[str], name_suffixes_to_strip: list[str]) -> dict:
    """Two numbers, carries-weighted, for `season`:
      - `any_position_unmatched_rate`: share of rush-play carries that fail to resolve to ANY
        team_rosters row (any position) after the exact+normalized passes -- the real
        resolution-failure floor. Verified live during planning at 2.6-3% for 2022-2024
        (season 2023 carries-weighted: 93.2% exact + 2.1% normalized + 2.0% ambiguous + 2.6%
        unmatched). A WARN above 10% here means the name-matching logic itself has regressed,
        not that most rushers are legitimately non-RB.
      - `rb_position_match_rate`: share of rush-play carries that resolve specifically to an
        RB -- informational (confounded with QB scrambles/other positions, which correctly
        don't match here), not itself a floor.
    """
    raw = build_raw_player_game_rushing(engine, [season], ["Rush", "Rushing Touchdown"], explosive_run_yard_threshold=15)
    if raw.empty:
        return {"season": season, "rb_position_match_rate": None, "any_position_unmatched_rate": None}

    distinct_names = raw[["rusher_player_name", "pos_team", "season"]].drop_duplicates()

    resolved_rb = resolve_players(engine, distinct_names, [season], positions, name_suffixes_to_strip)
    merged_rb = raw.merge(resolved_rb, on=["rusher_player_name", "pos_team", "season"], how="left")
    rb_matched_carries = merged_rb.loc[merged_rb["roster_match_method"].isin(["exact", "normalized"]), "carries"].sum()

    resolved_any = resolve_players(engine, distinct_names, [season], None, name_suffixes_to_strip)
    merged_any = raw.merge(resolved_any, on=["rusher_player_name", "pos_team", "season"], how="left")
    any_unmatched_carries = merged_any.loc[merged_any["roster_match_method"] == "unmatched", "carries"].sum()

    total_carries = raw["carries"].sum()
    return {
        "season": season,
        "rb_position_match_rate": float(rb_matched_carries / total_carries),
        "any_position_unmatched_rate": float(any_unmatched_carries / total_carries),
        "status": "WARN" if (any_unmatched_carries / total_carries) > 0.10 else "PASS",
    }


def check_covid_season_flag(engine: Engine) -> pd.DataFrame:
    games_by_season = run_query("SELECT season, COUNT(*) AS n_games FROM games GROUP BY season ORDER BY season", engine=engine)
    plays_by_season = run_query("SELECT season, COUNT(*) AS n_plays FROM plays GROUP BY season ORDER BY season", engine=engine)
    merged = games_by_season.merge(plays_by_season, on="season", how="outer").sort_values("season")
    median_games = merged["n_games"].median()
    merged["is_shortened_season"] = merged["n_games"] < (0.6 * median_games)
    return merged


def run_all_checks(engine: Engine, positions: list[str], name_suffixes_to_strip: list[str]) -> dict:
    """Convenience entry point used by scripts/inspect_database.py."""
    return {
        "row_counts": check_row_counts(engine),
        "duplicate_keys": check_duplicate_keys(engine),
        "rusher_name_completeness": check_rusher_name_completeness(engine),
        "covid_season_flag": check_covid_season_flag(engine),
        "player_resolution_match_rates": pd.DataFrame(
            [check_player_resolution_match_rate(engine, s, positions, name_suffixes_to_strip) for s in [2022, 2023, 2024]]
        ),
    }
