"""Orchestrates the full port: raw CFBD pulls -> team-week base stats -> rolling windows
-> talent/coach/returning-production join -> home/away pivot against a schedule. Produces
dataframes with the same raw column schema as
../Data/CFB_Gambling_Predictors_Final_PBP.csv / ../Data/CFB_Pred_Week_<N>.csv, so they can
be fed straight into cfb_cover_model.cleaning.prepare_week_frame /
cfb_cover_model.cleaning.build_clean_modeling_frame exactly as those R-generated CSVs are.

Two entry points:
  build_current_week_rows()  - live prediction path, 0-fill policy, one row per team
                                broadcast onto the upcoming week's schedule
  build_historical_rows()    - backfill path, NaN-fill + drop-incomplete policy, one row
                                per completed team-week joined against final scores
"""
from __future__ import annotations

import pandas as pd

from cfb_cover_model.ingest import box_score_features as bsf
from cfb_cover_model.ingest import pbp_features as pf
from cfb_cover_model.ingest import raw_cache
from cfb_cover_model.ingest import rolling_features as rf
from cfb_cover_model.ingest import talent_coach_returning as tcr

MIN_WEEK_HISTORICAL = 3  # matches R's `filter(week >= 3)` before na.omit()
MIN_WEEK_LIVE = 4        # matches R's `filter(week >= as.numeric(4))`
COACH_HISTORY_LOOKBACK_YEARS = 20  # how far back to fetch coach seasons for a cumulative
                                     # career record - R fetches 2004-2025 (~20 years by
                                     # 2024). A coach whose tenure started earlier than this
                                     # window will still have an undercounted career total;
                                     # a byte-exact match would require the exact same
                                     # 2004 start year for every future season, which isn't
                                     # worth hardcoding here - 20 years back covers all but
                                     # the longest-tenured active coaches.


def _fetch_team_week_raw(client, season: int, weeks: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Box-score + EPA/PBP features for every (team, week) in `weeks`, not yet
    rolling-windowed. Returns (frame, stat_cols)."""
    stats = raw_cache.get_weekly_range("game_team_stats", season, weeks, client)
    if stats.empty:
        return pd.DataFrame(), []
    stats = bsf.add_allowed_columns(stats)
    stats = bsf.clean_box_score(stats)
    stats = bsf.add_engineered_ratios(stats)

    plays_frames, drives_frames = [], []
    for w in weeks:
        plays_frames.append(raw_cache.get_weekly("plays", season, w, client))
        drives_frames.append(raw_cache.get_weekly("drives", season, w, client))
    plays = pd.concat(plays_frames, ignore_index=True) if plays_frames else pd.DataFrame()
    drives = pd.concat(drives_frames, ignore_index=True) if drives_frames else pd.DataFrame()
    epa = pf.compute_epa_features(plays, drives) if not plays.empty else pd.DataFrame()

    id_cols = {"game_id", "team", "home_away", "conference", "week", "year"}
    box_stat_cols = [c for c in stats.columns if c not in id_cols]

    merged = stats.merge(epa, on=["team", "week", "year"], how="left") if not epa.empty else stats
    epa_stat_cols = [c for c in epa.columns if c not in ("team", "week", "year")] if not epa.empty else []
    stat_cols = box_stat_cols + epa_stat_cols

    # a team plays at most once per week - collapse the (game_id, team, home_away)
    # granularity down to (team, week, year) for the rolling-window step
    merged = merged.drop(columns=[c for c in ("game_id", "home_away", "conference") if c in merged.columns])
    merged[epa_stat_cols] = merged[epa_stat_cols].fillna(0)

    return merged, stat_cols


def _fetch_talent_coach_returning(client, season: int, min_year: int) -> pd.DataFrame:
    talent = raw_cache.get_season("team_talent", season, client)
    roster = raw_cache.get_season("roster", season, client)
    recruits = raw_cache.get_season("recruits", season, client)
    blue_chip = tcr.compute_blue_chip_ratio(roster, recruits)
    talent_full = tcr.merge_talent(talent, blue_chip)

    coach_years = range(season - COACH_HISTORY_LOOKBACK_YEARS, season + 1)
    coach_frames = []
    for y in coach_years:
        c = raw_cache.get_season("coaches", y, client)
        if not c.empty:
            coach_frames.append(c)
    coaches_multi_year = pd.concat(coach_frames, ignore_index=True) if coach_frames else pd.DataFrame(
        columns=["Name", "team", "year", "games", "wins"]
    )
    coach_record = tcr.compute_coach_cumulative_record(coaches_multi_year)
    coach_lagged = tcr.lag_coach_by_year(coach_record)

    returning = raw_cache.get_season("returning_production", season, client)

    return tcr.merge_talent_coach_returning(talent_full, coach_lagged, returning, min_year)


def _pivot_home_away(team_rows: pd.DataFrame, schedule: pd.DataFrame, match_on_week: bool) -> pd.DataFrame:
    """team_rows: one row per team (match_on_week=False, live path - each team's single
    latest row broadcasts onto every game it plays that week) or one row per (team, week,
    season) (match_on_week=True, historical path - each completed game matched to that
    team's stats for that exact week). schedule always has home_team/away_team columns
    (plus week/season when match_on_week=True). team_rows must already use "season" (not
    "year") as its column name, matching schedule and the final CSV schema - callers
    rename before invoking this."""
    non_id_cols = [c for c in team_rows.columns if c not in ("team", "week", "season")]
    join_keys = ["week", "season"] if match_on_week else []

    home = team_rows.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in non_id_cols}})
    away = team_rows.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in non_id_cols}})

    result = schedule.merge(home, on=["home_team"] + join_keys, how="inner")
    result = result.merge(away, on=["away_team"] + join_keys, how="inner")
    return result


def build_current_week_rows(client, season: int, week: int) -> pd.DataFrame:
    """Live-prediction path: 0-fill missing rolling averages, each team's single most
    recent row broadcast onto every game it plays in `week`.

    Deliberately does NOT lag-and-slice_max the way R's 2025_Pred_Update.R does - that
    approach shifts every column by one row *before* picking the latest available row,
    which (since there's no raw row for the not-yet-played target week to shift into)
    ends up using the *second*-most-recent completed game as "prev_week", one game
    staler than intended. Caught during validation against the real historical CSV
    (see docs/api_ingestion.md): systematic ~100-300-unit discrepancies on raw yardage
    columns, consistent with an off-by-one-game shift. Instead: compute un-lagged rolling
    averages through week-1, take each team's latest (un-lagged) row directly, and rename
    its own raw stat columns to prev_week_{stat} with no further shift - that row's values
    already mean "as of the most recent completed game," which is exactly the correct
    semantic for predicting the next, not-yet-played game, and matches what the model was
    actually trained on (the historical pipeline's non-stale semantics)."""
    prior_weeks = list(range(1, week))
    team_week, stat_cols = _fetch_team_week_raw(client, season, prior_weeks)
    if team_week.empty:
        raise ValueError(f"No team-week data available for season={season} weeks<{week} - has this season started?")

    rolled = rf.add_rolling_averages(team_week, stat_cols, fill_value=0)
    rolled = rolled[rolled["week"] >= MIN_WEEK_LIVE - 1]  # this row itself becomes "last
    # week" for week `week`, so it must be >= (MIN_WEEK_LIVE - 1) to match R's `week >= 4`
    # requirement on the *target* week
    latest = rolled.sort_values("week").groupby("team", as_index=False).tail(1)
    latest = latest.rename(columns={col: f"prev_week_{col}" for col in stat_cols})

    tcr_df = _fetch_talent_coach_returning(client, season, min_year=season)

    team_rows = latest.merge(tcr_df.drop(columns=["year", "Name"], errors="ignore"), on="team", how="left")
    team_rows = team_rows.drop(columns=["week", "year"])  # broadcast onto every game in
    # `week` by team name alone - no week/season column needed on this side of the join

    # Blanket 0-fill, matching 2025_Pred_Update.R's `tot_pred[is.na(tot_pred)] <- 0` (line
    # 49 of that script) applied at this exact point in the pipeline - after the talent/
    # coach/returning-production merge, before the home/away pivot. Covers real missing-
    # data cases the live path can hit that the historical path never needs to (a team with
    # no published talent composite, e.g. a service academy; a first-year coach with no
    # prior record to lag in; 0-attempt down/distance ratios that divide out to NaN) - see
    # docs/api_ingestion.md and the user's explicit "replicate the R live pipeline's 0-fill
    # exactly" decision, made after this NaN gap surfaced during --live smoke testing.
    team_rows = team_rows.fillna(0)

    games = raw_cache.get_weekly("games", season, week, client)
    lines = raw_cache.get_weekly("betting_lines", season, week, client)
    spread = bsf.consensus_or_average_spread(lines) if not lines.empty else pd.DataFrame(
        columns=["game_id", "home_team", "away_team", "spread", "over_under", "formatted_spread"]
    )
    schedule = games.merge(spread[["game_id", "spread", "formatted_spread"]], on="game_id", how="left")
    schedule["spread"] = schedule["spread"].abs()
    schedule["home_favored"] = schedule.apply(
        lambda r: int(str(r["home_team"]) in str(r["formatted_spread"])) if pd.notna(r["formatted_spread"]) else 0,
        axis=1,
    )
    schedule = schedule[
        ["game_id", "home_team", "away_team", "season", "week", "neutral_site", "conference_game", "spread", "home_favored"]
    ]

    return _pivot_home_away(team_rows, schedule, match_on_week=False)


def build_historical_rows(client, season: int, weeks: list[int] | None = None) -> pd.DataFrame:
    """Backfill path: NaN-fill + drop incomplete rows, every completed team-week joined
    against that game's final score/spread by game_id."""
    weeks = weeks or list(range(1, 13))
    team_week, stat_cols = _fetch_team_week_raw(client, season, weeks)
    if team_week.empty:
        return pd.DataFrame()

    rolled = rf.add_rolling_and_lag(team_week, stat_cols, fill_value=None, drop_incomplete=True)
    rolled = rolled[rolled["week"] >= MIN_WEEK_HISTORICAL]

    tcr_df = _fetch_talent_coach_returning(client, season, min_year=season)
    team_rows = rolled.merge(tcr_df.drop(columns=["Name"], errors="ignore"), on=["team", "year"], how="inner")
    team_rows = team_rows.rename(columns={"year": "season"})

    games = raw_cache.get_weekly_range("games", season, weeks, client)
    games = games[games["completed"] == True]  # noqa: E712
    lines = raw_cache.get_weekly_range("betting_lines", season, weeks, client)
    spread = bsf.consensus_or_average_spread(lines) if not lines.empty else pd.DataFrame(
        columns=["game_id", "home_team", "away_team", "spread", "over_under", "formatted_spread"]
    )
    games_with_spread = games.merge(spread[["game_id", "spread", "formatted_spread"]], on="game_id", how="inner")
    games_with_spread["home_minus_away"] = games_with_spread["home_points"] - games_with_spread["away_points"]
    games_with_spread["home_favored"] = games_with_spread.apply(
        lambda r: int(str(r["home_team"]) in str(r["formatted_spread"])), axis=1
    )
    games_with_spread["spread"] = games_with_spread["spread"].abs()
    schedule = games_with_spread[
        ["game_id", "home_team", "away_team", "season", "week", "neutral_site", "conference_game",
         "spread", "home_favored", "home_points", "away_points", "home_minus_away"]
    ]

    return _pivot_home_away(team_rows, schedule, match_on_week=True)
