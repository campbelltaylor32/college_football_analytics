# Pull + cache the data behind the 2026 QB cumulative EPA tracker.
#
# - Play-by-play: cfbfastR::load_cfb_pbp() release file for the season, trimmed
#   to the columns the analysis uses.
# - QB roster rows (athlete IDs + ESPN headshot URLs): cfbfastR::load_cfb_rosters().
#   The roster athlete_id matches the ESPN player IDs in the PBP
#   (completion_player_id, sack_taken_player_id, rush_player_id, ...).
# - FBS team info (conference, colors, logos): cfbd_team_info().
#
# Re-run weekly during the season to pick up new games.
# Outputs (gitignored): data/pbp_<season>.rds, data/qb_roster_<season>.rds,
# data/fbs_teams_<season>.rds

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

season <- as.integer(Sys.getenv("CFB_SEASON", "2026"))

pbp_cols <- c(
  "year", "week", "season_type", "game_id", "id_play", "pos_team", "def_pos_team",
  "offense_conference", "period", "play_type", "down", "distance", "yards_gained",
  "EPA", "success", "pos_score_diff_start", "pass", "rush", "sack", "completion",
  "touchdown", "passer_player_name", "rusher_player_name",
  "completion_player_id", "incompletion_player_id", "interception_thrown_player_id",
  "sack_taken_player_id", "rush_player_id"
)

### Play-by-play ###

message("Loading PBP release for ", season)
pbp <- load_cfb_pbp(season) %>%
  as_tibble() %>%
  select(any_of(pbp_cols)) %>%
  mutate(game_id = as.integer(game_id), id_play = as.character(id_play))

### QB roster (IDs + headshots) ###

qb_roster <- load_cfb_rosters(season) %>%
  as_tibble() %>%
  filter(position == "QB") %>%
  transmute(athlete_id = as.character(athlete_id), first_name, last_name,
            team, headshot_url)

### FBS teams ###

fbs_teams <- cfbd_team_info(year = season, only_fbs = TRUE) %>%
  as_tibble() %>%
  transmute(team = school, conference, color, alt_color, logo)

saveRDS(pbp, file.path(data_dir, paste0("pbp_", season, ".rds")))
saveRDS(qb_roster, file.path(data_dir, paste0("qb_roster_", season, ".rds")))
saveRDS(fbs_teams, file.path(data_dir, paste0("fbs_teams_", season, ".rds")))

cat("PBP plays:", nrow(pbp), "| weeks:", paste(sort(unique(pbp$week)), collapse = ", "), "\n")
cat("QB roster rows:", nrow(qb_roster), "| FBS teams:", nrow(fbs_teams), "\n")
