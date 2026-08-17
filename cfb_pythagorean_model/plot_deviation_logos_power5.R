# Power 5 (ACC, Big Ten, Big 12, SEC) + Notre Dame version of the Pythagorean
# deviation logo chart. Unlike plot_deviation_logos.R (which reuses the
# pre-picked all-FBS top/bottom 10), this recomputes the top/bottom 10 from
# scratch within the Power 5 + ND subset, since that ranking shifts once
# Group of 5 teams are excluded.

require(tidyverse)
require(cfbfastR)
library(ggimage)
library(scales)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- dirname(script_path)
outputs_dir <- file.path(script_dir, "outputs")

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

TOP_N <- 10
POWER5_CONFERENCES <- c("ACC", "Big Ten", "Big 12", "SEC")

team_season <- read.csv(file.path(outputs_dir, "team_pythagorean_2025.csv")) %>%
  mutate(deviation = actual_win_pct - pyth_win_pct_exp2)

teams <- cfbd_team_info() %>%
  select(school, conference, color, alt_color, logo) %>%
  mutate(color = coalesce(color, alt_color)) %>%
  select(-alt_color) %>%
  mutate(logo = map_chr(logo, ~ .x[[1]]))

merged_data <- team_season %>%
  inner_join(teams, by = c("team" = "school")) %>%
  filter(conference %in% POWER5_CONFERENCES | team == "Notre Dame")

cat("Power 5 + Notre Dame teams:", nrow(merged_data), "\n")

top_bottom <- bind_rows(
  merged_data %>% slice_max(deviation, n = TOP_N),
  merged_data %>% slice_min(deviation, n = TOP_N)
)

top_bottom <- top_bottom %>%
  mutate(logo_y = ifelse(deviation < 0, deviation - 0.012, deviation + 0.012))

p <- ggplot(top_bottom, aes(x = reorder(team, deviation), y = deviation, fill = color)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_image(aes(y = logo_y, image = logo), size = 0.045, by = "height") +
  geom_hline(yintercept = 0, color = "gray40") +
  scale_fill_identity() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Beating the Pythagorean Odds: 2025 Power 5 + Notre Dame",
    subtitle = "Top 10 Over- and Underperformers vs. Classic Pythagorean Expected Win% (k=2)",
    x = "",
    y = "Actual Win% − Pythagorean Expected Win%",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_minimal(base_family = "Arial") +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 22, color = "#003366", hjust = 0.5),
    plot.subtitle = element_text(size = 13, color = "#003366", hjust = 0.5),
    plot.caption = element_text(size = 8, color = "black"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 10, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    plot.background = element_rect(fill = "#F7F7F7", color = NA),
    legend.position = "none"
  )

ggsave(file.path(outputs_dir, "deviation_top_bottom_10_logos_power5.png"), plot = p, width = 12, height = 8, dpi = 200)
cat("Wrote", file.path(outputs_dir, "deviation_top_bottom_10_logos_power5.png"), "\n")
