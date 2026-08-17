# Team-logo horizontal bar chart of the preseason power ratings Top 25 -- styled to match
# cfb_pythagorean_model/plot_deviation_logos.R and cfb_talent_distribution's theme_talent()
# (navy/light-gray house style, ggimage logos, cfbd_team_info() for colors/logos).
#
# Reads outputs/ratings/<season>/week_00_ratings.csv (scripts/generate_preseason_ratings.py's
# output). Run generate_preseason_ratings.py first.
#
# Usage: Rscript scripts/plot_preseason_top25.R [season]   (defaults to 2026)

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
TOP_N <- 25

ratings_path <- file.path(project_dir, "outputs", "ratings", as.character(SEASON), "week_00_ratings.csv")
ratings <- read.csv(ratings_path) %>%
  arrange(desc(rating)) %>%
  head(TOP_N)

teams <- cfbd_team_info(year = SEASON) %>%
  select(school, color, alt_color, logo) %>%
  mutate(color = coalesce(color, alt_color)) %>%
  select(-alt_color) %>%
  # cfbd_team_info() returns a list-column of logo URLs (dark/light); take the first
  mutate(logo = map_chr(logo, ~ if (length(.x) > 0) .x[[1]] else NA_character_))

merged_data <- ratings %>%
  inner_join(teams, by = c("team" = "school"))

unmatched <- ratings %>% anti_join(teams, by = c("team" = "school"))
if (nrow(unmatched) > 0) {
  cat("WARNING: teams with no logo/color match:\n")
  print(unmatched$team)
}

# Label/logo placement works outward from the bar's actual tip (handles negative ratings --
# a team rated below league-average -- correctly, and reads cleanly even for short bars,
# which a centered-inside label would not).
max_abs <- max(abs(merged_data$rating))
merged_data <- merged_data %>%
  mutate(
    team_label = sprintf("%2d. %s", rank, team),
    bar_sign = if_else(rating >= 0, 1, -1),
    label_y = rating + bar_sign * max_abs * 0.05,
    logo_y = rating + bar_sign * max_abs * 0.16
  )

p <- ggplot(merged_data, aes(x = reorder(team_label, rating), y = rating, fill = color)) +
  geom_col(width = 0.7) +
  geom_image(aes(y = logo_y, image = logo), size = 0.045, by = "height") +
  geom_text(aes(y = label_y, label = sprintf("%.1f", rating)), color = "#003366", fontface = "bold", size = 3.2) +
  coord_flip(clip = "off") +
  scale_fill_identity() +
  scale_y_continuous(expand = expansion(mult = c(0.18, 0.18))) +
  labs(
    title = paste(SEASON, "College Football Preseason Power Ratings"),
    subtitle = "Model-based preseason projection — points above/below average on a neutral field",
    x = "",
    y = "Preseason power rating (points, neutral field)",
    caption = "Data Source: @cfbfastR, Model: cfb_power_ratings, Viz: @campbell_taylor1"
  ) +
  theme_minimal(base_family = "Arial") +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 22, color = "#003366", hjust = 0.5),
    plot.subtitle = element_text(size = 10.5, color = "#003366", hjust = 0.5),
    plot.caption = element_text(size = 8, color = "black"),
    axis.text.y = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(size = 9),
    axis.title.x = element_text(size = 11, face = "bold"),
    plot.background = element_rect(fill = "#F7F7F7", color = NA),
    plot.margin = margin(10, 30, 10, 10),
    legend.position = "none"
  )

out_path <- file.path(project_dir, "outputs", "ratings", as.character(SEASON), "preseason_top25.png")
ggsave(out_path, plot = p, width = 12, height = 10, dpi = 200)
cat("Wrote", out_path, "\n")
