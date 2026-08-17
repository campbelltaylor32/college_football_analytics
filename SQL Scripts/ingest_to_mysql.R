require(cfbfastR)
require(tidyverse)
require(data.table)
require(DBI)
require(RMariaDB)

# Fills the cfb_football MySQL database (SQL Scripts/schema.sql) from the raw
# cfbfastR endpoints. Idempotent: before pulling a season/week, it checks
# what's already in the DB and skips it, so re-running only fetches new
# weeks instead of re-hitting the API for history already cached.
#
# One-time setup, if not already done:
#   mysql -u root -e "CREATE DATABASE IF NOT EXISTS cfb_football;"
#   mysql -u root cfb_football < "SQL Scripts/schema.sql"

# Load CFBD_API_KEY from the repo root's .env, searching upward from the working directory
# (not commandArgs()'s --file= path -- that mangles paths with spaces, like "SQL Scripts/",
# in some invocation contexts). `Sys.getenv("CFBD_API_KEY")` alone -- the prior form of this
# line -- reads and discards the value; it never actually puts the key where cfbfastR's own
# internal Sys.getenv() calls can see it, which silently made every cfbd_*() call below fail.
find_env_file <- function() {
  dir <- getwd()
  for (i in 1:5) {
    candidate <- file.path(dir, ".env")
    if (file.exists(candidate)) return(candidate)
    dir <- dirname(dir)
  }
  NULL
}
env_file <- find_env_file()
if (!is.null(env_file)) readRenviron(env_file)
if (Sys.getenv("CFBD_API_KEY") == "") {
  stop("CFBD_API_KEY not found -- expected it in the repo root's .env (see .env.example). Run this script from the repo root.")
}

CURRENT_SEASON <- 2026
CURRENT_WEEK   <- 0   # no 2026 games/lines exist yet; 0 skips those endpoints for this season
                      # entirely (WEEKS[WEEKS <= 0] is empty) rather than requesting a
                      # not-yet-played week, which the API/retry-backoff handled very slowly
WEEKS          <- 1:15

# earliest year each endpoint actually has data, per live probe against the
# API (see conversation) -- not guessed from memory
START_YEAR <- list(
  games                = 2013,  # betting_lines (the label source) doesn't exist before this anyway
  betting_lines        = 2013,
  team_talent          = 2015,
  coaches              = 2004,
  team_rosters         = 2004,
  recruiting_players   = 2000,
  returning_production = 2014,
  game_team_stats      = 2013,  # FK -> games(game_id), which only exists from 2013 on;
  plays                = 2013   # earlier rows would 100% fail the FK and get skipped anyway
)

con <- dbConnect(RMariaDB::MariaDB(), host = "localhost", user = "root", password = "", dbname = "cfb_football")
on.exit(dbDisconnect(con), add = TRUE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

upsert_df <- function(con, table, df, key_cols) {
  if (nrow(df) == 0) return(invisible(NULL))
  df[] <- lapply(df, function(x) if (is.factor(x)) as.character(x) else x)
  cols <- names(df)
  update_cols <- setdiff(cols, key_cols)
  placeholders <- paste0("(", paste(rep("?", length(cols)), collapse = ", "), ")")
  col_list <- paste(cols, collapse = ", ")
  update_clause <- paste(sprintf("%s = VALUES(%s)", update_cols, update_cols), collapse = ", ")
  sql <- sprintf(
    "INSERT INTO %s (%s) VALUES %s ON DUPLICATE KEY UPDATE %s",
    table, col_list, placeholders, update_clause
  )
  # bind the whole data frame as one batch (each column is a placeholder's
  # vector of values) -- binding row-by-row in a loop silently only applies
  # a fraction of the rows with RMariaDB, so this must be a single call
  batch_ok <- tryCatch({
    stmt <- dbSendStatement(con, sql)
    dbBind(stmt, unname(as.list(df)))
    dbClearResult(stmt)
    TRUE
  }, error = function(e) {
    log_msg(sprintf(
      "%s: batch insert failed (%s) -- falling back to row-by-row to skip just the bad row(s)",
      table, conditionMessage(e)
    ))
    FALSE
  })
  if (batch_ok) return(invisible(NULL))

  # fallback path: one dbExecute() per row (a fresh prepare/bind/execute each
  # time, NOT a reused dbBind() -- that's the pattern that silently
  # under-executes). Slower, but only runs when the batch above errored, and
  # it's all localhost so per-row round trips are cheap.
  row_sql <- sprintf(
    "INSERT INTO %s (%s) VALUES (%s) ON DUPLICATE KEY UPDATE %s",
    table, col_list, paste(rep("?", length(cols)), collapse = ", "), update_clause
  )
  skipped <- 0
  for (i in seq_len(nrow(df))) {
    row <- df[i, , drop = FALSE]
    ok <- tryCatch({
      dbExecute(con, row_sql, params = unname(as.list(row)))
      TRUE
    }, error = function(e) {
      key_desc <- paste(sprintf("%s=%s", key_cols, row[key_cols]), collapse = ", ")
      log_msg(sprintf("%s: skipping row (%s) -- %s", table, key_desc, conditionMessage(e)))
      FALSE
    })
    if (!ok) skipped <- skipped + 1
  }
  if (skipped > 0) {
    log_msg(sprintf("%s: skipped %d/%d rows that didn't fit the schema", table, skipped, nrow(df)))
  }
}

upsert_teams <- function(con, school_names) {
  school_names <- unique(na.omit(school_names))
  school_names <- setdiff(school_names, dbGetQuery(con, "SELECT school FROM teams")$school)
  if (length(school_names) == 0) return(invisible(NULL))
  df <- data.frame(school = school_names, team_id = NA_integer_)
  dbExecute(con, "INSERT IGNORE INTO teams (school, team_id) VALUES (?, ?)",
            params = list(df$school, df$team_id))
}

existing_season_weeks <- function(con, table) {
  dbGetQuery(con, sprintf("SELECT DISTINCT season, week FROM %s", table))
}

existing_seasons <- function(con, table, season_col = "season") {
  dbGetQuery(con, sprintf("SELECT DISTINCT %s AS season FROM %s", season_col, table))$season
}

needed_weeks <- function(have, season, weeks) {
  if (nrow(have) == 0) return(weeks)
  done <- have$week[have$season == season]
  setdiff(weeks, done)
}

log_msg <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), paste0(...)))

# cfbfastR collapses every non-200 API response (rate limit, auth, server
# error) into the same generic `stop("The API returned an error")`, with no
# status code preserved -- so we can't distinguish "rate limited" from other
# HTTP errors by message text. Retrying any thrown error with backoff is safe
# regardless: a genuine "no data" week (bye week) doesn't throw at all, it
# returns an empty data frame with a warning, handled separately below.
API_MAX_RETRIES <- 5
API_BACKOFF_BASE <- 10  # seconds; doubles each attempt: 10, 20, 40, 80, 160

with_retry <- function(pull_fn, label, max_retries = API_MAX_RETRIES,
                        backoff_base = API_BACKOFF_BASE) {
  attempt <- 1
  repeat {
    result <- tryCatch(list(ok = TRUE, value = pull_fn()),
                        error = function(e) list(ok = FALSE, error = e))
    if (result$ok) return(result$value)
    if (attempt >= max_retries) {
      log_msg(sprintf(
        "giving up on %s after %d attempts: %s",
        label, attempt, conditionMessage(result$error)
      ))
      return(NULL)
    }
    wait <- backoff_base * 2^(attempt - 1)
    log_msg(sprintf(
      "API error on %s (attempt %d/%d): %s -- backing off %ds",
      label, attempt, max_retries, conditionMessage(result$error), wait
    ))
    Sys.sleep(wait)
    attempt <- attempt + 1
  }
}

# cfbd_pbp_data(epa_wpa=TRUE) computes EPA/WPA per game internally, so a
# single bad game -- a missing predictor value, an empty drives pull, or
# some other data gap we haven't hit yet -- breaks the WHOLE week's call
# at once (verified live: ~8-13% of games in a week). None of the causes
# found so far are transient: retrying the identical call doesn't fix
# missing source data. Rather than pattern-matching each new error message
# to decide what's safe to skip, just cap retries low for this endpoint
# specifically and always fall back to pulling team-by-team on any
# failure, whatever it is. That isolates each game, giving a real
# (comparatively rare) transient blip one more short chance per team,
# while whatever's actually broken gets skipped and logged without having
# to know why.
PLAYS_MAX_RETRIES <- 2
PLAYS_BACKOFF_BASE <- 10  # seconds: one retry after a 10s pause, then give up

pull_plays_week <- function(con, season, week) {
  df <- with_retry(
    function() cfbd_pbp_data(year = season, week = week, epa_wpa = TRUE),
    label = sprintf("plays %d wk%d", season, week),
    max_retries = PLAYS_MAX_RETRIES, backoff_base = PLAYS_BACKOFF_BASE
  )
  if (!is.null(df)) return(df)

  teams <- dbGetQuery(
    con, "SELECT DISTINCT home_team AS team FROM games WHERE season = ? AND week = ?",
    params = list(season, week)
  )$team
  if (length(teams) == 0) return(NULL)

  log_msg(sprintf("plays %d wk%d: retrying per-team to salvage the rest of the week", season, week))
  parts <- list()
  for (tm in teams) {
    part <- with_retry(
      function() cfbd_pbp_data(year = season, week = week, team = tm, epa_wpa = TRUE),
      label = sprintf("plays %d wk%d team=%s", season, week, tm),
      max_retries = PLAYS_MAX_RETRIES, backoff_base = PLAYS_BACKOFF_BASE
    )
    if (!is.null(part) && nrow(part) > 0) parts[[tm]] <- part
    Sys.sleep(0.3)
  }
  if (length(parts) == 0) return(NULL)
  dplyr::bind_rows(parts)
}

# ---------------------------------------------------------------------------
# games (cfbd_game_info)
# ---------------------------------------------------------------------------
log_msg("=== games ===")
have <- existing_season_weeks(con, "games")
for (season in START_YEAR$games:CURRENT_SEASON) {
  weeks_to_get <- needed_weeks(have, season, WEEKS[WEEKS <= ifelse(season == CURRENT_SEASON, CURRENT_WEEK, 15)])
  for (week in weeks_to_get) {
    df <- with_retry(function() cfbd_game_info(year = season, week = week),
                      label = sprintf("games %d wk%d", season, week))
    if (is.null(df) || nrow(df) == 0) next
    df <- df %>% filter(completed == TRUE)
    if (nrow(df) == 0) next
    upsert_teams(con, c(df$home_team, df$away_team))
    out <- df %>%
      transmute(
        game_id, season, week, season_type, start_date = format(lubridate::ymd_hms(start_date, quiet = TRUE), "%Y-%m-%d %H:%M:%S"),
        start_time_tbd, completed, neutral_site, conference_game,
        attendance, venue_id, venue,
        home_id, home_team, home_division, home_conference, home_points,
        home_post_win_prob, home_pregame_elo, home_postgame_elo,
        away_id, away_team, away_division, away_conference, away_points,
        away_post_win_prob, away_pregame_elo, away_postgame_elo,
        excitement_index, highlights, notes, playoff
      )
    upsert_df(con, "games", out, key_cols = "game_id")
    log_msg(sprintf("games %d wk%d: %d rows", season, week, nrow(out)))
    Sys.sleep(0.3)
  }
}

# ---------------------------------------------------------------------------
# betting_lines (cfbd_betting_lines)
# ---------------------------------------------------------------------------
log_msg("=== betting_lines ===")
have <- existing_season_weeks(con, "betting_lines")
for (season in START_YEAR$betting_lines:CURRENT_SEASON) {
  weeks_to_get <- needed_weeks(have, season, WEEKS[WEEKS <= ifelse(season == CURRENT_SEASON, CURRENT_WEEK, 15)])
  for (week in weeks_to_get) {
    df <- with_retry(function() cfbd_betting_lines(year = season, week = week),
                      label = sprintf("betting_lines %d wk%d", season, week))
    if (is.null(df) || nrow(df) == 0) next
    upsert_teams(con, c(df$home_team, df$away_team))
    out <- df %>%
      transmute(
        game_id, season, season_type, week, start_date = format(lubridate::ymd_hms(start_date, quiet = TRUE), "%Y-%m-%d %H:%M:%S"),
        home_team_id, home_team, home_conference, home_classification, home_score,
        away_team_id, away_team, away_conference, away_classification, away_score,
        provider, spread, formatted_spread, spread_open, over_under, over_under_open,
        home_moneyline, away_moneyline
      )
    upsert_df(con, "betting_lines", out, key_cols = c("game_id", "provider"))
    log_msg(sprintf("betting_lines %d wk%d: %d rows", season, week, nrow(out)))
    Sys.sleep(1)  # provider is heavier / more rate-limited than other endpoints
  }
}

# ---------------------------------------------------------------------------
# team_talent (cfbd_team_talent)
# ---------------------------------------------------------------------------
log_msg("=== team_talent ===")
have_seasons <- existing_seasons(con, "team_talent")
for (season in setdiff(START_YEAR$team_talent:CURRENT_SEASON, have_seasons)) {
  df <- with_retry(function() cfbd_team_talent(year = season),
                    label = sprintf("team_talent %d", season))
  if (is.null(df) || nrow(df) == 0) { log_msg(sprintf("team_talent %d: no data", season)); next }
  upsert_teams(con, df$school)
  out <- df %>% transmute(season = year, school, talent)
  upsert_df(con, "team_talent", out, key_cols = c("season", "school"))
  log_msg(sprintf("team_talent %d: %d rows", season, nrow(out)))
  Sys.sleep(0.3)
}

# ---------------------------------------------------------------------------
# coaches (cfbd_coaches)
# ---------------------------------------------------------------------------
log_msg("=== coaches ===")
have_seasons <- existing_seasons(con, "coaches")
for (season in setdiff(START_YEAR$coaches:CURRENT_SEASON, have_seasons)) {
  df <- with_retry(function() cfbd_coaches(year = season),
                    label = sprintf("coaches %d", season))
  if (is.null(df) || nrow(df) == 0) { log_msg(sprintf("coaches %d: no data", season)); next }
  upsert_teams(con, df$school)
  out <- df %>%
    transmute(
      first_name, last_name,
      hire_date = format(lubridate::ymd_hms(hire_date, quiet = TRUE), "%Y-%m-%d"),
      school, season = year,
      games, wins, losses, ties, preseason_rank, postseason_rank,
      srs, sp_overall, sp_offense, sp_defense
    )
  upsert_df(con, "coaches", out, key_cols = c("first_name", "last_name", "school", "season"))
  log_msg(sprintf("coaches %d: %d rows", season, nrow(out)))
  Sys.sleep(0.3)
}

# ---------------------------------------------------------------------------
# team_rosters (cfbd_team_roster) -- whole-league pull per year, no team filter
# ---------------------------------------------------------------------------
log_msg("=== team_rosters ===")
have_seasons <- existing_seasons(con, "team_rosters")
for (season in setdiff(START_YEAR$team_rosters:CURRENT_SEASON, have_seasons)) {
  df <- with_retry(function() cfbd_team_roster(year = season),
                    label = sprintf("team_rosters %d", season))
  if (is.null(df) || nrow(df) == 0) { log_msg(sprintf("team_rosters %d: no data", season)); next }
  upsert_teams(con, df$team)
  out <- df %>%
    transmute(
      athlete_id, first_name, last_name, team,
      weight, height, jersey, year, position,
      home_city, home_state, home_country, home_latitude, home_longitude, home_county_fips,
      recruit_ids = sapply(recruit_ids, function(x) jsonlite::toJSON(as.list(x))),
      headshot_url, season
    )
  upsert_df(con, "team_rosters", out, key_cols = c("athlete_id", "season"))
  log_msg(sprintf("team_rosters %d: %d rows", season, nrow(out)))
  Sys.sleep(0.5)
}

# ---------------------------------------------------------------------------
# recruiting_players (cfbd_recruiting_player) -- whole-league pull per year
# ---------------------------------------------------------------------------
log_msg("=== recruiting_players ===")
have_seasons <- existing_seasons(con, "recruiting_players", season_col = "recruit_year")
for (season in setdiff(START_YEAR$recruiting_players:CURRENT_SEASON, have_seasons)) {
  df <- with_retry(function() cfbd_recruiting_player(year = season),
                    label = sprintf("recruiting_players %d", season))
  if (is.null(df) || nrow(df) == 0) { log_msg(sprintf("recruiting_players %d: no data", season)); next }
  upsert_teams(con, df$committed_to)
  out <- df %>%
    transmute(
      recruit_id = id, athlete_id, recruit_type, recruit_year = year, ranking,
      name, school, committed_to, position, height, weight, stars, rating,
      city, state_province, country,
      hometown_latitude = hometown_info_latitude,
      hometown_longitude = hometown_info_longitude,
      hometown_fips_code = hometown_info_fips_code
    )
  upsert_df(con, "recruiting_players", out, key_cols = "recruit_id")
  log_msg(sprintf("recruiting_players %d: %d rows", season, nrow(out)))
  Sys.sleep(0.5)
}

# ---------------------------------------------------------------------------
# returning_production (cfbd_player_returning)
# ---------------------------------------------------------------------------
log_msg("=== returning_production ===")
have_seasons <- existing_seasons(con, "returning_production")
for (season in setdiff(START_YEAR$returning_production:CURRENT_SEASON, have_seasons)) {
  df <- with_retry(function() cfbd_player_returning(season),
                    label = sprintf("returning_production %d", season))
  if (is.null(df) || nrow(df) == 0) { log_msg(sprintf("returning_production %d: no data", season)); next }
  upsert_teams(con, df$team)
  out <- df %>%
    transmute(
      season, team, conference, total_ppa, total_passing_ppa, total_receiving_ppa,
      total_rushing_ppa, percent_ppa, percent_passing_ppa, percent_receiving_ppa,
      percent_rushing_ppa, usage_pct = usage, passing_usage, receiving_usage, rushing_usage
    )
  upsert_df(con, "returning_production", out, key_cols = c("season", "team"))
  log_msg(sprintf("returning_production %d: %d rows", season, nrow(out)))
  Sys.sleep(0.3)
}

# ---------------------------------------------------------------------------
# game_team_stats (cfbd_game_team_stats) -- raw compound strings pass through
# as-is; the *_raw columns feed the generated columns in MySQL
# ---------------------------------------------------------------------------
log_msg("=== game_team_stats ===")
have <- existing_season_weeks(con, "game_team_stats")
for (season in START_YEAR$game_team_stats:CURRENT_SEASON) {
  weeks_to_get <- needed_weeks(have, season, WEEKS[WEEKS <= ifelse(season == CURRENT_SEASON, CURRENT_WEEK, 15)])
  for (week in weeks_to_get) {
    df <- with_retry(function() cfbd_game_team_stats(year = season, week = week),
                      label = sprintf("game_team_stats %d wk%d", season, week))
    if (is.null(df) || nrow(df) == 0) next
    upsert_teams(con, df$school)
    out <- df %>%
      transmute(
        game_id, school, season = season, week = week,
        conference, home_away, opponent, opponent_conference,
        points = as.integer(points),
        total_yards = as.integer(total_yards),
        net_passing_yards = as.integer(net_passing_yards),
        completion_attempts_raw = completion_attempts,
        passing_tds = as.integer(passing_tds),
        yards_per_pass = as.numeric(yards_per_pass),
        passes_intercepted = as.integer(passes_intercepted),
        interception_yards = as.integer(interception_yards),
        interception_tds = as.integer(interception_tds),
        rushing_attempts = as.integer(rushing_attempts),
        rushing_yards = as.integer(rushing_yards),
        rush_tds = as.integer(rush_tds),
        yards_per_rush_attempt = as.numeric(yards_per_rush_attempt),
        first_downs = as.integer(first_downs),
        third_down_eff_raw = third_down_eff,
        fourth_down_eff_raw = fourth_down_eff,
        punt_returns = as.integer(punt_returns),
        punt_return_yards = as.integer(punt_return_yards),
        punt_return_tds = as.integer(punt_return_tds),
        kick_return_yards = as.integer(kick_return_yards),
        kick_return_tds = as.integer(kick_return_tds),
        kick_returns = as.integer(kick_returns),
        kicking_points = as.integer(kicking_points),
        fumbles_recovered = as.integer(fumbles_recovered),
        fumbles_lost = as.integer(fumbles_lost),
        total_fumbles = as.integer(total_fumbles),
        tackles = as.integer(tackles),
        tackles_for_loss = as.integer(tackles_for_loss),
        sacks = as.integer(sacks),
        qb_hurries = as.integer(qb_hurries),
        interceptions = as.integer(interceptions),
        passes_deflected = as.integer(passes_deflected),
        turnovers = as.integer(turnovers),
        defensive_tds = as.integer(defensive_tds),
        total_penalties_yards_raw = total_penalties_yards,
        possession_time_raw = possession_time,
        points_allowed = as.integer(points_allowed),
        total_yards_allowed = as.integer(total_yards_allowed),
        net_passing_yards_allowed = as.integer(net_passing_yards_allowed),
        completion_attempts_allowed_raw = completion_attempts_allowed,
        passing_tds_allowed = as.integer(passing_tds_allowed),
        yards_per_pass_allowed = as.numeric(yards_per_pass_allowed),
        passes_intercepted_allowed = as.integer(passes_intercepted_allowed),
        interception_yards_allowed = as.integer(interception_yards_allowed),
        interception_tds_allowed = as.integer(interception_tds_allowed),
        rushing_attempts_allowed = as.integer(rushing_attempts_allowed),
        rushing_yards_allowed = as.integer(rushing_yards_allowed),
        rush_tds_allowed = as.integer(rush_tds_allowed),
        yards_per_rush_attempt_allowed = as.numeric(yards_per_rush_attempt_allowed),
        first_downs_allowed = as.integer(first_downs_allowed),
        third_down_eff_allowed_raw = third_down_eff_allowed,
        fourth_down_eff_allowed_raw = fourth_down_eff_allowed,
        punt_returns_allowed = as.integer(punt_returns_allowed),
        punt_return_yards_allowed = as.integer(punt_return_yards_allowed),
        punt_return_tds_allowed = as.integer(punt_return_tds_allowed),
        kick_return_yards_allowed = as.integer(kick_return_yards_allowed),
        kick_return_tds_allowed = as.integer(kick_return_tds_allowed),
        kick_returns_allowed = as.integer(kick_returns_allowed),
        kicking_points_allowed = as.integer(kicking_points_allowed),
        fumbles_recovered_allowed = as.integer(fumbles_recovered_allowed),
        fumbles_lost_allowed = as.integer(fumbles_lost_allowed),
        total_fumbles_allowed = as.integer(total_fumbles_allowed),
        tackles_allowed = as.integer(tackles_allowed),
        tackles_for_loss_allowed = as.integer(tackles_for_loss_allowed),
        sacks_allowed = as.integer(sacks_allowed),
        qb_hurries_allowed = as.integer(qb_hurries_allowed),
        interceptions_allowed = as.integer(interceptions_allowed),
        passes_deflected_allowed = as.integer(passes_deflected_allowed),
        turnovers_allowed = as.integer(turnovers_allowed),
        defensive_tds_allowed = as.integer(defensive_tds_allowed),
        penalties_allowed_yards_raw = total_penalties_yards_allowed,
        possession_time_allowed_raw = possession_time_allowed
      )
    upsert_df(con, "game_team_stats", out, key_cols = c("game_id", "school"))
    log_msg(sprintf("game_team_stats %d wk%d: %d rows", season, week, nrow(out)))
    Sys.sleep(0.5)
  }
}

# ---------------------------------------------------------------------------
# plays (cfbd_pbp_data, epa_wpa=TRUE) -- EPA aggregation inputs plus WPA,
# ppa, context flags, participant names, and drive detail (see schema.sql
# note on what's deliberately still excluded)
# ---------------------------------------------------------------------------
log_msg("=== plays ===")
have <- existing_season_weeks(con, "plays")
for (season in START_YEAR$plays:CURRENT_SEASON) {
  weeks_to_get <- needed_weeks(have, season, WEEKS[WEEKS <= ifelse(season == CURRENT_SEASON, CURRENT_WEEK, 15)])
  for (week in weeks_to_get) {
    df <- pull_plays_week(con, season, week)
    if (is.null(df) || nrow(df) == 0) next
    upsert_teams(con, c(df$pos_team, df$def_pos_team))
    out <- df %>%
      transmute(
        play_id = id_play, game_id, season, week = wk, drive_id,
        pos_team, def_pos_team,
        offense_conference, defense_conference,
        play_type, play_text,
        period = as.integer(period),
        half = as.integer(as.character(half)),
        clock_minutes = as.integer(clock_minutes),
        clock_seconds = as.integer(clock_seconds),
        down, distance,
        yard_line = as.integer(yard_line),
        yards_to_goal = as.integer(yards_to_goal),
        yards_gained,
        epa = EPA,
        ep_before = as.numeric(ep_before),
        ep_after = as.numeric(ep_after),
        ppa = as.numeric(ppa),
        wpa = as.numeric(wpa),
        wp_before = as.numeric(wp_before),
        wp_after = as.numeric(wp_after),
        home_wp_before = as.numeric(home_wp_before),
        home_wp_after = as.numeric(home_wp_after),
        away_wp_before = as.numeric(away_wp_before),
        away_wp_after = as.numeric(away_wp_after),
        success,
        rz_play = as.logical(rz_play),
        scoring_opp = as.logical(scoring_opp),
        middle_8,
        stuffed_run = as.logical(stuffed_run),
        turnover = as.logical(turnover),
        downs_turnover = as.logical(downs_turnover),
        touchdown = as.logical(touchdown),
        safety = as.logical(safety),
        penalty_flag, penalty_text,
        rusher_player_name, yds_rushed = as.integer(yds_rushed),
        passer_player_name, receiver_player_name,
        yds_receiving = as.integer(yds_receiving),
        sack_player_name, sack_player_name2,
        yds_sacked = as.integer(yds_sacked),
        interception_player_name,
        yds_int_return = as.integer(yds_int_return),
        fumble_player_name, fumble_forced_player_name, fumble_recovered_player_name,
        yds_fumble_return = as.integer(yds_fumble_return),
        punter_player_name, yds_punted = as.integer(yds_punted),
        punt_returner_player_name, yds_punt_return = as.integer(yds_punt_return),
        fg_kicker_player_name, yds_fg = as.integer(yds_fg),
        kickoff_player_name, kickoff_returner_player_name,
        drive_scoring, drive_pts,
        drive_result_detailed,
        drive_start_yards_to_goal = as.integer(drive_start_yards_to_goal),
        drive_end_yards_to_goal = as.integer(drive_end_yards_to_goal),
        drive_yards = as.integer(drive_yards),
        drive_start_period = as.integer(drive_start_period),
        drive_end_period = as.integer(drive_end_period),
        new_drive_pts = as.integer(new_drive_pts)
      ) %>%
      distinct(season, game_id, play_id, .keep_all = TRUE)
    upsert_df(con, "plays", out, key_cols = c("season", "game_id", "play_id"))
    log_msg(sprintf("plays %d wk%d: %d rows", season, week, nrow(out)))
    Sys.sleep(0.5)
  }
}

log_msg("Ingestion pass complete.")
