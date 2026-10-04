# One team-logo horizontal bar chart per FBS conference (SEC, Big Ten, Big 12, ACC, etc.),
# styled identically to plot_preseason_top25.R -- every team in the conference, ranked by
# preseason power rating, not just a top-N cutoff.
#
# Reads outputs/ratings/<season>/week_<NN>_ratings.csv (ALL FBS teams, not just the top 25).
# With no week arg (or week 0) it plots the preseason ratings (generate_preseason_ratings.py's
# week_00_ratings.csv); with a week N >= 1 it plots the in-season blended ratings
# (update_ratings.py's week_<NN>_ratings.csv). Run the corresponding Python script first.
#
# Usage: Rscript scripts/plot_conference_rankings.R [season] [week]   (defaults to 2026, week 0)

require(tidyverse)
require(cfbfastR)
library(ggimage)
library(scales)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
project_dir <- normalizePath(file.path(script_dir, ".."))

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

cli_args <- commandArgs(trailingOnly = TRUE)
SEASON <- if (length(cli_args) > 0) as.integer(cli_args[1]) else 2026
WEEK <- if (length(cli_args) > 1) as.integer(cli_args[2]) else 0
# Optional version tag matching update_ratings.py's output suffix, e.g. "pg5_sw0.5" reads
# week_<NN>_ratings_pg5_sw0.5.csv (older scoring-only runs are tagged just "pg5"). Omit for
# preseason / unversioned (pre-versioning) files.
VERSION <- if (length(cli_args) > 2) cli_args[3] else ""
version_suffix <- if (VERSION != "") paste0("_", VERSION) else ""
sw <- if (grepl("_sw", VERSION)) as.numeric(sub("^.*_sw", "", VERSION)) else NA
# --week N scores week N's slate using games through week N-1.
RESULTS_THROUGH <- WEEK - 1

ratings_file <- sprintf("week_%02d_ratings%s.csv", WEEK, version_suffix)
ratings_path <- file.path(project_dir, "outputs", "ratings", as.character(SEASON), ratings_file)
ratings <- read.csv(ratings_path)

if (WEEK == 0) {
  title_qualifier <- "Preseason Power Ratings"
  chart_subtitle <- "Model-based preseason projection — points above/below an average FBS team on a neutral field"
} else {
  title_qualifier <- paste0("Power Ratings — Through Week ", RESULTS_THROUGH)
  # Share of a typical team's rating still coming from the preseason prior:
  # phantom_games / (phantom_games + games_played), from the ratings CSV itself.
  ratings_all <- read.csv(ratings_path)
  typical_games <- median(ratings_all$games_played)
  prior_pct <- round(100 * median(ratings_all$effective_prior_weight[ratings_all$games_played == typical_games]))
  results_note <- if (!is.na(sw)) {
    sprintf("%d%% final score, %d%% play-by-play efficiency (EPA/play & success rate)", round(100 * sw), round(100 * (1 - sw)))
  } else {
    "final scores"
  }
  chart_subtitle <- paste0(
    sprintf("Opponent- and home-field-adjusted results through Week %d: %s\n", RESULTS_THROUGH, results_note),
    sprintf("Preseason model prior = %d%% of rating (typical team, %d games played) · Points vs. an average FBS team on a neutral field",
            prior_pct, typical_games)
  )
}

teams <- cfbd_team_info(year = SEASON) %>%
  select(school, conference, color, alt_color, logo) %>%
  mutate(color = coalesce(color, alt_color)) %>%
  select(-alt_color) %>%
  mutate(logo = map_chr(logo, ~ if (length(.x) > 0) .x[[1]] else NA_character_)) %>%
  filter(!is.na(conference) & conference != "")

merged_data <- ratings %>%
  inner_join(teams, by = c("team" = "school"))

unmatched <- ratings %>% anti_join(teams, by = c("team" = "school"))
if (nrow(unmatched) > 0) {
  cat("WARNING: teams with no logo/conference match (likely FCS/non-FBS or a name mismatch):\n")
  print(unmatched$team)
}

slugify <- function(x) {
  x %>% tolower() %>% str_replace_all("[^a-z0-9]+", "_") %>% str_replace_all("^_|_$", "")
}

out_dir <- file.path(project_dir, "outputs", "ratings", as.character(SEASON), "conferences")
if (WEEK != 0) out_dir <- file.path(out_dir, sprintf("week_%02d%s", WEEK, version_suffix))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

conferences <- sort(unique(merged_data$conference))
cat("Building charts for", length(conferences), "conferences:\n")
print(conferences)

for (conf in conferences) {
  conf_data <- merged_data %>%
    filter(conference == conf) %>%
    arrange(desc(rating)) %>%
    mutate(
      conf_rank = row_number(),
      team_label = sprintf("%2d. %s", conf_rank, team)
    )

  # Ratings can be negative (a team rated below league-average) -- label/logo placement has
  # to work outward from the bar's actual tip in whichever direction it extends, not assume
  # every bar runs left-to-right from 0. Placing both just past the bar end (rather than the
  # label centered inside, as in an earlier version) also reads correctly even for very short
  # bars, which a centered-inside label does not.
  max_abs <- max(abs(conf_data$rating))
  conf_data <- conf_data %>%
    mutate(
      bar_sign = if_else(rating >= 0, 1, -1),
      label_y = rating + bar_sign * max_abs * 0.05,
      logo_y = rating + bar_sign * max_abs * 0.16
    )

  n_teams <- nrow(conf_data)
  plot_height <- max(4, 1.2 + n_teams * 0.42)

  p <- ggplot(conf_data, aes(x = reorder(team_label, rating), y = rating, fill = color)) +
    geom_col(width = 0.7) +
    geom_image(aes(y = logo_y, image = logo), size = min(0.04, 0.7 / n_teams), by = "height") +
    geom_text(aes(y = label_y, label = sprintf("%.1f", rating)), color = "#003366", fontface = "bold", size = 3.2) +
    coord_flip(clip = "off") +
    scale_fill_identity() +
    scale_y_continuous(expand = expansion(mult = c(0.18, 0.18))) +
    labs(
      title = paste("Campbell Taylor", SEASON, conf, title_qualifier),
      subtitle = chart_subtitle,
      x = "",
      y = "Power rating (points, neutral field)",
      caption = "Data Source: @cfbfastR, Model: cfb_power_ratings, Viz: @campbell_taylor1"
    ) +
    theme_minimal(base_family = "Arial") +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 20, color = "#003366", hjust = 0.5),
      plot.subtitle = element_text(size = 10.5, color = "#003366", hjust = 0.5),
      plot.caption = element_text(size = 8, color = "black"),
      axis.text.y = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(size = 9),
      axis.title.x = element_text(size = 11, face = "bold"),
      plot.background = element_rect(fill = "#F7F7F7", color = NA),
      plot.margin = margin(10, 30, 10, 10),
      legend.position = "none"
    )

  out_path <- file.path(out_dir, paste0(slugify(conf), ".png"))
  ggsave(out_path, plot = p, width = 11, height = plot_height, dpi = 200)
  cat("Wrote", out_path, "(", n_teams, "teams)\n")
}
