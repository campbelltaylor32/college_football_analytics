# Renders scripts/export_schedule_strength.py's 2026 schedule-difficulty ranking as PNG tables
# (not the bar-chart style plot_preseason_top25.R/plot_conference_rankings.R use) -- one overall
# Top 25 table across all FBS, plus one table per conference (every team in that conference, not
# a cutoff), styled to match this project's house style (navy/light-gray, ggimage logos, same
# caption). No table-drawing R package exists elsewhere in this repo (checked: no gt/gridExtra::
# tableGrob/ggtexttable/flextable) -- built here in plain ggplot2 to keep the existing
# tidyverse/cfbfastR/ggimage/scales-only dependency footprint.
#
# Reads outputs/schedule_strength/2026/schedule_strength.csv. Run export_schedule_strength.py first.

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
CAPTION <- "Data Source: @cfbfastR, Model: cfb_power_ratings, Viz: @campbell_taylor1"

# Two schedule-difficulty metrics, both already in schedule_strength.csv: the mean answers "how
# hard is the schedule on the whole" but can be dominated by one or two extreme games; the median
# answers "how hard is a TYPICAL game on this schedule" and is far less moved by a single
# gauntlet matchup. A team can rank very differently on the two (e.g. one marquee road game
# against an elite opponent inflates the mean without moving the median much) -- both are worth
# seeing side by side, not just one.
METRICS <- list(
  mean = list(score_col = "sos_score", rank_col = "sos_rank", label = "SOS (Mean)",
              subtitle = "Schedule strength: MEAN site-adjusted opponent power rating (points, neutral field) across the full 2026 slate",
              overall_file = "hardest_schedules_overall.png", conf_dir = "conferences"),
  median = list(score_col = "sos_median", rank_col = "sos_median_rank", label = "SOS (Median)",
                subtitle = "Schedule strength: MEDIAN site-adjusted opponent power rating (points, neutral field) across the full 2026 slate -- less sensitive to a single extreme game than the mean",
                overall_file = "hardest_schedules_overall_median.png", conf_dir = "conferences_median")
)

slugify <- function(x) {
  x %>% tolower() %>% str_replace_all("[^a-z0-9]+", "_") %>% str_replace_all("^_|_$", "")
}

# Draws one ranked table as a PNG. `data` must have columns: rank, team, logo, n_games,
# n_road_games, n_vs_2026_preseason_top25, `conference` if show_conference=TRUE, and whichever
# column `score_col`/`natl_rank_col` name.
build_schedule_table <- function(data, title, subtitle, out_path, show_conference, score_col, score_label, natl_rank_col) {
  n <- nrow(data)
  data <- data %>% mutate(row_y = n:1, rank_label = sprintf("%d", rank))

  x_rank <- 0.3
  x_logo <- 1.0
  x_team <- 1.6
  x_conf <- 4.7
  x_natl <- 4.7
  x_sos <- if (show_conference) 5.9 else 5.6
  x_games <- x_sos + 1.15
  x_road <- x_games + 1.15
  x_top25 <- x_road + 1.15
  x_max <- x_top25 + 1.0

  header_y <- n + 1
  zebra <- data %>% filter(row_y %% 2 == 0)

  p <- ggplot(data) +
    geom_rect(
      data = zebra, aes(ymin = row_y - 0.5, ymax = row_y + 0.5), xmin = -0.1, xmax = x_max,
      fill = "#E8ECF1", color = NA
    ) +
    geom_rect(
      data = data.frame(x = 1), ymin = header_y - 0.5, ymax = header_y + 0.5, xmin = -0.1, xmax = x_max,
      fill = "#003366", color = NA
    ) +
    geom_text(aes(x = x_rank, y = row_y, label = rank_label), hjust = 0, fontface = "bold", size = 3.3, color = "#003366") +
    geom_image(aes(x = x_logo, y = row_y, image = logo), size = 0.032, by = "height") +
    geom_text(aes(x = x_team, y = row_y, label = team), hjust = 0, size = 3.3) +
    geom_text(aes(x = x_sos, y = row_y, label = sprintf("%.1f", .data[[score_col]])), hjust = 0.5, fontface = "bold", size = 3.3, color = "#003366") +
    geom_text(aes(x = x_games, y = row_y, label = n_games), hjust = 0.5, size = 3.1) +
    geom_text(aes(x = x_road, y = row_y, label = n_road_games), hjust = 0.5, size = 3.1) +
    geom_text(aes(x = x_top25, y = row_y, label = n_vs_2026_preseason_top25), hjust = 0.5, size = 3.1) +
    annotate("text", x = x_rank, y = header_y, label = "Rank", hjust = 0, fontface = "bold", size = 3.3, color = "white") +
    annotate("text", x = x_team, y = header_y, label = "Team", hjust = 0, fontface = "bold", size = 3.3, color = "white") +
    annotate("text", x = x_sos, y = header_y, label = score_label, hjust = 0.5, fontface = "bold", size = 3.3, color = "white") +
    annotate("text", x = x_games, y = header_y, label = "GP", hjust = 0.5, fontface = "bold", size = 3.3, color = "white") +
    annotate("text", x = x_road, y = header_y, label = "Road", hjust = 0.5, fontface = "bold", size = 3.3, color = "white") +
    annotate("text", x = x_top25, y = header_y, label = "Top25", hjust = 0.5, fontface = "bold", size = 3.3, color = "white")

  if (show_conference) {
    p <- p +
      geom_text(aes(x = x_conf, y = row_y, label = conference), hjust = 0, size = 3.1) +
      annotate("text", x = x_conf, y = header_y, label = "Conf", hjust = 0, fontface = "bold", size = 3.3, color = "white")
  } else {
    p <- p +
      geom_text(aes(x = x_natl, y = row_y, label = .data[[natl_rank_col]]), hjust = 0.5, size = 3.1) +
      annotate("text", x = x_natl, y = header_y, label = "Nat'l", hjust = 0.5, fontface = "bold", size = 3.3, color = "white")
  }

  p <- p +
    scale_x_continuous(limits = c(-0.1, x_max), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.3, header_y + 0.6), expand = c(0, 0)) +
    labs(title = title, subtitle = subtitle, caption = CAPTION) +
    theme_void(base_family = "Arial") +
    theme(
      plot.title = element_text(face = "bold", size = 18, color = "#003366", hjust = 0.5),
      plot.subtitle = element_text(size = 9, color = "#003366", hjust = 0.5, margin = margin(b = 8)),
      plot.caption = element_text(size = 7.5, color = "black", margin = margin(t = 8)),
      plot.background = element_rect(fill = "#F7F7F7", color = NA),
      plot.margin = margin(12, 18, 10, 18)
    )

  height <- max(3.5, 1.1 + (n + 1) * 0.32)
  ggsave(out_path, plot = p, width = 9.5, height = height, dpi = 200)
  cat("Wrote", out_path, "(", n, "teams)\n")
}

strength_path <- file.path(project_dir, "outputs", "schedule_strength", as.character(SEASON), "schedule_strength.csv")
strength <- read.csv(strength_path)

teams <- cfbd_team_info(year = SEASON) %>%
  select(school, conference, logo) %>%
  mutate(logo = map_chr(logo, ~ if (length(.x) > 0) .x[[1]] else NA_character_)) %>%
  filter(!is.na(conference) & conference != "")

merged_data <- strength %>%
  inner_join(teams, by = c("team" = "school"))

unmatched <- strength %>% anti_join(teams, by = c("team" = "school"))
if (nrow(unmatched) > 0) {
  cat("WARNING: teams with no logo/conference match (likely FCS/non-FBS or a name mismatch):\n")
  print(unmatched$team)
}

out_dir <- file.path(project_dir, "outputs", "schedule_strength", as.character(SEASON))
conferences <- sort(unique(merged_data$conference))

for (metric_name in names(METRICS)) {
  m <- METRICS[[metric_name]]
  cat("\n=== Metric:", m$label, "===\n")

  conf_out_dir <- file.path(out_dir, m$conf_dir)
  dir.create(conf_out_dir, showWarnings = FALSE, recursive = TRUE)

  overall_data <- merged_data %>%
    arrange(desc(.data[[m$score_col]])) %>%
    head(25) %>%
    mutate(rank = row_number())

  build_schedule_table(
    overall_data,
    title = paste(SEASON, "Hardest Schedules --", m$label, "-- Top 25 (FBS)"),
    subtitle = m$subtitle,
    out_path = file.path(out_dir, m$overall_file),
    show_conference = TRUE,
    score_col = m$score_col, score_label = m$label, natl_rank_col = m$rank_col
  )

  for (conf in conferences) {
    conf_data <- merged_data %>%
      filter(conference == conf) %>%
      arrange(desc(.data[[m$score_col]])) %>%
      mutate(rank = row_number())

    build_schedule_table(
      conf_data,
      title = paste(SEASON, conf, "Schedule Strength --", m$label),
      subtitle = m$subtitle,
      out_path = file.path(conf_out_dir, paste0(slugify(conf), ".png")),
      show_conference = FALSE,
      score_col = m$score_col, score_label = m$label, natl_rank_col = m$rank_col
    )
  }
}
