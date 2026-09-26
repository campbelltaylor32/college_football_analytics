# Pull + cache the data behind the Ryan Day offense analysis (2019-2026).
#
# - Play-by-play: cfbfastR::load_cfb_pbp() release files (all FBS games, incl.
#   postseason), trimmed to the columns the analysis uses. Any completed Ohio
#   State game missing from the release (the release lags in-season, and a 2020
#   game is absent) is back-filled from the CFBD API with cfbd_pbp_data().
# - Drives + game results for Ohio State from the CFBD API (season_type = "both",
#   so conference title games, bowls and CFP games are included -- the shared
#   Data/Game_Stats_Averages_CFB_PBP_Added.csv only covers weeks 1-12).
#
# Re-run weekly during the current season to pick up new games.
# Outputs (gitignored): data/pbp_2019_2026.rds, data/osu_drives.rds,
# data/osu_games.rds, data/fbs_teams.rds

library(tidyverse)
library(cfbfastR)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
data_dir <- file.path(script_dir, "data")
dir.create(data_dir, showWarnings = FALSE)

# Searches upward from the working directory for the repo root's .env.
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
  stop("CFBD_API_KEY not found -- expected it in the repo root's .env. Run this script from the repo root.")
}

team <- "Ohio State"
years <- 2019:2026

pbp_cols <- c(
  "year", "week", "season_type", "game_id", "id_play", "drive_id",
  "pos_team", "def_pos_team", "period", "play_type", "down", "distance",
  "yards_to_goal", "yards_gained", "EPA", "success", "wp_before",
  "pos_score_diff_start", "touchdown", "turnover", "home", "away"
)

### Game results (Ohio State) ###

osu_games <- map_dfr(years, function(y) {
  Sys.sleep(0.5)
  cfbd_game_info(year = y, team = team, season_type = "both") %>%
    as_tibble() %>%
    filter(completed) %>%
    transmute(
      year = season, week, season_type, game_id,
      opponent = if_else(home_team == team, away_team, home_team),
      points_for = if_else(home_team == team, home_points, away_points),
      points_against = if_else(home_team == team, away_points, home_points)
    )
})

### FBS team list per season (for the FBS-average benchmark) ###

fbs_teams <- map_dfr(years, function(y) {
  Sys.sleep(0.5)
  cfbd_team_info(year = y, only_fbs = TRUE) %>%
    transmute(year = y, team = school)
})

### Play-by-play ###

pbp <- map_dfr(years, function(y) {
  message("Loading PBP release for ", y)
  load_cfb_pbp(y) %>%
    as_tibble() %>%
    select(any_of(pbp_cols)) %>%
    mutate(game_id = as.integer(game_id), id_play = as.character(id_play),
           drive_id = as.character(drive_id))
})

# Back-fill completed Ohio State games that the release doesn't have yet.
missing_games <- osu_games %>% anti_join(distinct(pbp, game_id), by = "game_id")
if (nrow(missing_games) > 0) {
  message("Back-filling ", nrow(missing_games), " Ohio State game(s) from the CFBD API")
  # cfbd_pbp_data() rejects week > 15, but CFBD files the 2020 Big Ten title game
  # (the one game the release is missing) under week 16. Relax that check here.
  assignInNamespace("validate_week", function(week) invisible(TRUE), "cfbfastR")
  backfill <- pmap_dfr(missing_games, function(year, week, season_type, game_id, ...) {
    Sys.sleep(0.5)
    cfbd_pbp_data(year = year, week = week, team = team, season_type = season_type,
                  epa_wpa = TRUE) %>%
      as_tibble() %>%
      filter(game_id == !!game_id) %>%
      mutate(year = year, week = week, season_type = season_type) %>%
      select(any_of(pbp_cols)) %>%
      mutate(game_id = as.integer(game_id), id_play = as.character(id_play),
             drive_id = as.character(drive_id))
  })
  pbp <- bind_rows(pbp, backfill)
}

still_missing <- osu_games %>% anti_join(distinct(pbp, game_id), by = "game_id")
if (nrow(still_missing) > 0) {
  warning("Still missing PBP for ", nrow(still_missing), " Ohio State game(s): ",
          paste(still_missing$year, still_missing$opponent, collapse = "; "))
}

### Drives (Ohio State) ###

osu_drives <- map_dfr(years, function(y) {
  Sys.sleep(0.5)
  cfbd_drives(year = y, team = team, season_type = "both") %>%
    as_tibble() %>%
    mutate(year = y, game_id = as.integer(game_id))
}) %>%
  semi_join(osu_games, by = "game_id")

saveRDS(pbp, file.path(data_dir, "pbp_2019_2026.rds"))
saveRDS(osu_drives, file.path(data_dir, "osu_drives.rds"))
saveRDS(osu_games, file.path(data_dir, "osu_games.rds"))
saveRDS(fbs_teams, file.path(data_dir, "fbs_teams.rds"))

cat("Ohio State games per season (results / PBP / drives):\n")
print(
  osu_games %>% count(year, name = "games") %>%
    left_join(pbp %>% filter(pos_team == team) %>% distinct(year, game_id) %>%
                count(year, name = "pbp_games"), by = "year") %>%
    left_join(osu_drives %>% distinct(year, game_id) %>%
                count(year, name = "drive_games"), by = "year")
)
