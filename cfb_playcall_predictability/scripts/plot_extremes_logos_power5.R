# Power 5 (ACC, Big Ten, Big 12, SEC) + Notre Dame logo chart of the 10 least and 10 most
# predictable run/pass play-calling offenses. Styled to match
# cfb_pythagorean_model/plot_deviation_logos_power5.R (ggimage logos, team-color bars,
# cfbd_team_info() for colors/logos, navy/light-gray house style).
#
# The top/bottom 10 are re-picked within the Power 5 + ND subset, not taken from the
# all-FBS ranking. Reads outputs/rankings/<season>_predictability.csv
# (scripts/rank_predictability.py). Run from the repo root:
#   Rscript cfb_playcall_predictability/scripts/plot_extremes_logos_power5.R
# Output: outputs/figures/<season>_extremes_logos_power5.png

suppressPackageStartupMessages({
  library(tidyverse)
  library(cfbfastR)
  library(ggimage)
})

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
project_dir <- dirname(script_dir)

# Searches upward from the working directory for the repo root's .env.
find_env_file <- function() {
  dir <- getwd()
  for (i in 1:5) {
    candidate <- file.path(dir, ".env")
    if (file.exists(candidate) && any(grepl("^CFBD_API_KEY=", readLines(candidate, warn = FALSE)))) return(candidate)
    dir <- dirname(dir)
  }
  NULL
}
env_file <- find_env_file()
if (!is.null(env_file)) readRenviron(env_file)
if (Sys.getenv("CFBD_API_KEY") == "") {
  stop("CFBD_API_KEY not found -- expected it in the repo root's .env. Run this script from the repo root.")
}

SEASON <- 2026
TOP_N <- 10
POWER5_CONFERENCES <- c("ACC", "Big Ten", "Big 12", "SEC")

rankings <- read.csv(file.path(project_dir, "outputs", "rankings", sprintf("%d_predictability.csv", SEASON)))
through_week <- max(rankings$through_week)
fbs_median <- median(rankings$predictability)

# Near-white primary colors vanish on the light background -- fall back to the alt color.
is_too_light <- function(hex) {
  rgb <- col2rgb(ifelse(is.na(hex), "#000000", hex)) / 255
  (0.2126 * rgb[1, ] + 0.7152 * rgb[2, ] + 0.0722 * rgb[3, ]) > 0.85
}

teams <- cfbd_team_info(year = SEASON) %>%
  select(school, conference, color, alt_color, logo) %>%
  mutate(
    color = if_else(is.na(color) | is_too_light(color), alt_color, color),
    color = coalesce(color, "#5f6b7a"),
    logo = map_chr(logo, ~ if (length(.x) > 0) .x[[1]] else NA_character_)
  ) %>%
  select(-alt_color)

p5 <- rankings %>%
  inner_join(teams, by = c("team" = "school")) %>%
  filter(conference %in% POWER5_CONFERENCES | team == "Notre Dame")

p5_all <- teams %>% filter(conference %in% POWER5_CONFERENCES | school == "Notre Dame")
unranked <- setdiff(p5_all$school, p5$team)
cat("Power 5 + Notre Dame teams ranked:", nrow(p5), "\n")
if (length(unranked) > 0) cat("Not ranked (too few qualifying calls):", paste(unranked, collapse = ", "), "\n")

least_label <- sprintf("Least predictable %d", TOP_N)
most_label <- sprintf("Most predictable %d", TOP_N)

plot_df <- bind_rows(
  p5 %>% slice_min(predictability, n = TOP_N) %>% mutate(group = least_label),
  p5 %>% slice_max(predictability, n = TOP_N) %>% mutate(group = most_label)
) %>%
  mutate(
    group = factor(group, levels = c(least_label, most_label)),
    # Rank 1 of each group at the top of its panel.
    order_key = if_else(group == least_label, -predictability, predictability),
    team_label = team,
    team_label = fct_reorder(team_label, order_key)
  )

# Logos sit in their own lane left of the lowest CI whisker, so whiskers never cross them.
logo_x <- min(0, min(plot_df$predictability_lo)) - 0.04
x_lo <- logo_x - 0.025
x_hi <- max(plot_df$predictability_hi) + 0.06

p <- ggplot(plot_df, aes(y = team_label)) +
  geom_vline(xintercept = 0, color = "gray40") +
  geom_vline(xintercept = fbs_median, color = "gray50", linetype = "dashed") +
  geom_col(aes(x = predictability, fill = color), width = 0.7) +
  geom_errorbarh(aes(xmin = predictability_lo, xmax = predictability_hi),
                 height = 0.25, color = "gray25", linewidth = 0.4) +
  geom_text(aes(x = predictability_hi + 0.008, label = sub("^-0\\.00$", "0.00", sprintf("%.2f", predictability))),
            hjust = 0, size = 3.4, fontface = "bold", color = "#003366") +
  geom_image(aes(x = logo_x, image = logo), size = 0.075, by = "height", asp = 1.1) +
  annotate("text", x = fbs_median, y = Inf, label = "FBS median", vjust = -0.4, hjust = 0.5,
           size = 3, color = "gray35") +
  facet_wrap(~group, scales = "free_y") +
  scale_fill_identity() +
  scale_x_continuous(limits = c(x_lo, x_hi), breaks = seq(0, 0.3, 0.1)) +
  coord_cartesian(clip = "off") +
  labs(
    title = sprintf("Who's Hardest to Call? %d Run/Pass Predictability", SEASON),
    subtitle = sprintf(
      "Power 5 + Notre Dame, through week %d  |  Share of a team's run/pass uncertainty explained by situation, tendencies & game flow\nHigher = easier to call. Error bars = 90%% CI (bootstrapped by game).",
      through_week
    ),
    x = "Predictability score  (1 − model log loss ÷ log loss of guessing the team's own pass rate)",
    y = "",
    caption = paste0(
      if (length(unranked) > 0) paste0("Not ranked (fewer than 150 qualifying calls): ", paste(unranked, collapse = ", "), ".  ") else "",
      "Excludes garbage time, kneels & spikes. Scrambles count as runs.\nData Source: @cfbfastR, Viz: @campbell_taylor1"
    )
  ) +
  theme_minimal(base_family = "Arial") +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.spacing.x = unit(2, "lines"),
    strip.text = element_text(face = "bold", size = 14, color = "#003366", hjust = 0),
    plot.title = element_text(face = "bold", size = 22, color = "#003366", hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = "#003366", hjust = 0.5, margin = margin(b = 14)),
    plot.caption = element_text(size = 8, color = "black"),
    axis.text.y = element_text(size = 10, face = "bold"),
    axis.title.x = element_text(size = 10, face = "bold"),
    plot.background = element_rect(fill = "#F7F7F7", color = NA),
    plot.margin = margin(15, 20, 10, 10),
    legend.position = "none"
  )

out_file <- file.path(project_dir, "outputs", "figures", sprintf("%d_extremes_logos_power5.png", SEASON))
ggsave(out_file, plot = p, width = 15, height = 8, dpi = 200)
cat("Wrote", out_file, "\n")
