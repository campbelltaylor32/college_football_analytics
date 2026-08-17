# Rebuilds blue_chip_ratio from live CFBD data with two fixes to the
# methodology used in R Scripts/Full_CFB_Game_Outcome_Historical.R:
#
#  1. Denominator: the source pipeline divides by every player on
#     cfbd_team_roster() (up to ~150, including walk-ons with no recruiting
#     profile), which dilutes the ratio far below real-world "Blue Chip
#     Ratio" figures (e.g. it computes Alabama 2021 at 54%, vs. the ~86%
#     widely reported that season). This script instead divides by players
#     who match to *some* recruiting record (rated or unrated) -- i.e.
#     players who were identifiably high-school recruits -- which
#     approximates the scholarship roster without walk-on dilution.
#
#  2. Matching: the source pipeline joins roster to recruiting only on
#     athlete_id, but athlete_id is NULL for a meaningful share of
#     recruiting records (including some blue-chip transfers). This script
#     also uses the roster's own recruit_ids -> recruiting id link and
#     takes the union of both matches, recovering players either method
#     misses alone.
#
# Validated against Alabama 2021 (a well-documented case: Bud Elliott
# reported ~86% BCR, an all-time high): old method = 54.4%, this method =
# 83.95% (68 blue-chip / 81 matched, vs. 68/125 previously).
#
# This does NOT touch Data/CFB_Team_Talent_Data.csv or the shared pipeline
# -- it writes a corrected CSV used only by this project's charts.

library(tidyverse)
library(cfbfastR)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
outputs_dir <- file.path(script_dir, "outputs")
dir.create(outputs_dir, showWarnings = FALSE)

# Searches upward from the working directory (not the commandArgs()-derived script_dir above --
# that path can get mangled with spaces in some invocation contexts, e.g. "SQL Scripts/").
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
  stop("CFBD_API_KEY not found -- expected it in the repo root's .env (see ../.env.example). Run this script from the repo root.")
}

roster_years <- 2015:2025
recruit_years <- 2009:2025

### Pull rosters (national, one call per year) ###
cat("Pulling rosters...\n")
tot_roster <- map_dfr(roster_years, function(y) {
  cat(" ", y, "\n")
  cfbd_team_roster(year = y) %>%
    transmute(
      year = y, team, athlete_id = as.character(athlete_id),
      recruit_ids = lapply(recruit_ids, as.character)
    )
})

### Pull recruiting classes (national, one call per year) ###
cat("Pulling recruiting classes...\n")
tot_recruits <- map_dfr(recruit_years, function(y) {
  cat(" ", y, "\n")
  cfbd_recruiting_player(year = y) %>%
    transmute(id = as.character(id), athlete_id = as.character(athlete_id), stars, rating)
})

by_athlete_id <- tot_recruits %>%
  filter(!is.na(athlete_id)) %>%
  distinct(athlete_id, .keep_all = TRUE) %>%
  select(athlete_id, stars_a = stars, rating_a = rating)

by_recruit_id <- tot_recruits %>%
  distinct(id, .keep_all = TRUE) %>%
  select(id, stars_b = stars, rating_b = rating)

### Union-match each roster row to a recruiting record via athlete_id OR recruit_id ###
cat("Matching roster to recruiting records...\n")
roster_long <- tot_roster %>%
  mutate(roster_row = row_number()) %>%
  unnest_longer(recruit_ids, values_to = "recruit_id", keep_empty = TRUE) %>%
  mutate(recruit_id = ifelse(recruit_id %in% c("0", "", NA), NA, recruit_id))

matched <- roster_long %>%
  left_join(by_athlete_id, by = "athlete_id") %>%
  left_join(by_recruit_id, by = c("recruit_id" = "id")) %>%
  group_by(roster_row, year, team) %>%
  summarise(
    stars = { s <- c(stars_a, stars_b); if (all(is.na(s))) NA_integer_ else max(s, na.rm = TRUE) },
    rating = { r <- c(rating_a, rating_b); if (all(is.na(r))) NA_real_ else max(r, na.rm = TRUE) },
    .groups = "drop"
  )

### Corrected blue-chip ratio: denominator = matched-to-a-recruiting-record players only ###
blue_chip_corrected <- matched %>%
  filter(!is.na(stars)) %>%
  group_by(team, year) %>%
  summarise(
    n_matched = n(),
    blue_chip_ratio = sum(stars >= 4) / n_matched,
    avg_player_rating = mean(rating, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(blue_chip_corrected, file.path(outputs_dir, "blue_chip_ratio_corrected.csv"))
cat("\nWrote", file.path(outputs_dir, "blue_chip_ratio_corrected.csv"), "\n")

cat("\nSanity check -- Alabama 2021:\n")
print(blue_chip_corrected %>% filter(team == "Alabama", year == 2021))
