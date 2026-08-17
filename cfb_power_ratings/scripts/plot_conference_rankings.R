# One team-logo horizontal bar chart per FBS conference (SEC, Big Ten, Big 12, ACC, etc.),
# styled identically to plot_preseason_top25.R -- every team in the conference, ranked by
# preseason power rating, not just a top-N cutoff.
#
# Reads outputs/ratings/<season>/week_00_ratings.csv (ALL FBS teams, not just the top 25 --
# generate_preseason_ratings.py writes the full list even though it only prints the top 25).
# Run generate_preseason_ratings.py first.

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

SEASON <- 2026

ratings_path <- file.path(project_dir, "outputs", "ratings", as.character(SEASON), "week_00_ratings.csv")
ratings <- read.csv(ratings_path)

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
      title = paste(SEASON, conf, "Preseason Power Ratings"),
      subtitle = "Model-based preseason projection — points above/below average on a neutral field",
      x = "",
      y = "Preseason power rating (points, neutral field)",
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
