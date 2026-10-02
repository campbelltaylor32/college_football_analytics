# Pull + cache play-by-play for the play-call predictability model.
#
# Source: cfbfastR::load_cfb_pbp() release files, NOT the MySQL `plays` table. The `plays`
# table has no score or timeout columns, and rebuilding score from cumulative drive_pts only
# reproduces the final score in ~52-59% of games (checked 2016 + 2024). Joining the release's
# score columns back onto `plays` isn't possible either: the release stores id_play as a
# double, which loses the last 2-3 digits of the 18-digit id. So plays come from the release
# directly (it carries every `plays` column this project uses), while games / betting lines /
# coaches still come from MySQL.
#
# Re-run weekly in-season: completed seasons are cached and skipped; the current season is
# always re-pulled, and any completed week the release doesn't have yet (it lags in-season) is
# back-filled from the CFBD API with cfbd_pbp_data(epa_wpa = TRUE) -- same columns. The
# back-fill needs CFBD_API_KEY in the repo root's .env. Usage (from repo root):
#   Rscript cfb_playcall_predictability/scripts/pull_pbp.R [first_season] [last_season]
# Output (gitignored): data/interim/pbp_<season>.parquet

suppressPackageStartupMessages({
  library(dplyr)
  library(cfbfastR)
  library(arrow)
})

# Release files are 100-300 MB; R's default 60s download timeout is too short.
options(timeout = 900)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
out_dir <- file.path(dirname(script_dir), "data", "interim")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

args <- commandArgs(trailingOnly = TRUE)
first_season <- if (length(args) >= 1) as.integer(args[1]) else 2014L
current_season <- 2026L
last_season <- if (length(args) >= 2) as.integer(args[2]) else current_season

pbp_cols <- c(
  "year", "week", "season_type", "game_id", "game_play_number", "drive_id", "drive_number",
  "drive_play_number", "pos_team", "def_pos_team", "home", "away", "period", "half",
  "TimeSecsRem", "adj_TimeSecsRem", "clock_minutes", "clock_seconds", "down", "distance",
  "yards_to_goal", "play_type", "play_text", "rush", "pass", "sack", "yards_gained", "EPA",
  "success", "wp_before", "pos_score_diff_start", "pos_team_score", "def_pos_team_score",
  "pos_team_timeouts_rem_before", "def_pos_team_timeouts_rem_before", "penalty_flag",
  "pos_team_receives_2H_kickoff", "offense_conference", "defense_conference"
)

# Searches upward from the working directory for the repo root's .env.
find_env_file <- function() {
  dir <- getwd()
  for (i in 1:5) {
    candidate <- file.path(dir, ".env")
    if (file.exists(candidate) && any(grepl("^CFBD_API_KEY=", readLines(candidate)))) return(candidate)
    dir <- dirname(dir)
  }
  NULL
}

trim_pbp <- function(df) {
  df %>%
    as_tibble() %>%
    select(any_of(pbp_cols)) %>%
    mutate(
      game_id = as.integer(game_id),
      drive_id = as.character(drive_id),
      across(any_of(c("rush", "pass", "sack", "success", "penalty_flag", "pos_team_receives_2H_kickoff")), as.integer)
    )
}

# Completed regular-season weeks of `season` that the release file doesn't cover yet.
backfill_current_season <- function(season, pbp) {
  env_file <- find_env_file()
  if (!is.null(env_file)) readRenviron(env_file)
  if (Sys.getenv("CFBD_API_KEY") == "") {
    warning("CFBD_API_KEY not set -- skipping back-fill of weeks the release is missing")
    return(pbp)
  }
  games <- cfbd_game_info(year = season, season_type = "regular")
  done_weeks <- sort(unique(games$week[games$completed]))
  missing_weeks <- setdiff(done_weeks, unique(pbp$week))
  for (wk in missing_weeks) {
    message("  back-filling ", season, " week ", wk, " from the CFBD API")
    Sys.sleep(0.5)
    wk_pbp <- cfbd_pbp_data(year = season, week = wk, season_type = "regular", epa_wpa = TRUE)
    if (is.null(wk_pbp) || nrow(wk_pbp) == 0) next
    wk_pbp <- trim_pbp(wk_pbp) %>% mutate(year = season, week = wk, season_type = "regular")
    pbp <- bind_rows(pbp, wk_pbp)
  }
  pbp
}

for (season in first_season:last_season) {
  out_file <- file.path(out_dir, sprintf("pbp_%d.parquet", season))
  if (file.exists(out_file) && season < current_season) {
    message("Cached: ", season)
    next
  }
  message("Loading PBP release for ", season)
  raw <- load_cfb_pbp(season)
  if (nrow(raw) == 0 || !"game_id" %in% names(raw)) {
    stop("Empty play-by-play pull for ", season, " (download failed?) -- re-run to retry")
  }
  pbp <- trim_pbp(raw)
  if (season == current_season) pbp <- backfill_current_season(season, pbp)
  write_parquet(pbp, out_file)
  message(sprintf("  %d: %d plays, %d games, weeks %s", season, nrow(pbp),
                  n_distinct(pbp$game_id), paste(range(pbp$week), collapse = "-")))
}
