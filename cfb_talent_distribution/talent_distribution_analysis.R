# Talent distribution analysis: how blue-chip recruiting talent has spread
# across FBS rosters since 2015, and whether it has become more concentrated
# among a handful of programs or more dispersed, since the transfer portal
# (~2018) and NIL (July 2021) eras began.
#
# Run build_corrected_blue_chip_ratio.R first -- this script depends on its
# output (outputs/blue_chip_ratio_corrected.csv). The blue_chip_ratio column
# in Data/CFB_Team_Talent_Data.csv (the shared pipeline's own calculation)
# understates real blue-chip shares by roughly a third (denominator dilution
# from walk-on-inclusive rosters -- see README), so charts 1 and 3 here use
# the corrected figure instead. Chart 2 (Gini) uses `talent`, which comes
# straight from CFBD's own team-talent composite endpoint and is unaffected
# by that issue, so it still reads Data/CFB_Team_Talent_Data.csv directly.
#
# Both data sources are filtered to FBS-only teams per season (the raw
# talent file mixes in FCS programs for 2015-2023 -- see README).

library(tidyverse)
library(cfbfastR)
library(ggridges)
library(scales)

Sys.setenv(CFBD_API_KEY = Sys.getenv("CFBD_API_KEY"))

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
repo_root <- normalizePath(file.path(script_dir, ".."))
outputs_dir <- file.path(script_dir, "outputs")
dir.create(outputs_dir, showWarnings = FALSE)

### Load + filter to FBS-only teams per season ###

talent_raw <- read.csv(file.path(repo_root, "Data", "CFB_Team_Talent_Data.csv"))
years <- sort(unique(talent_raw$year))

fbs_teams <- map_dfr(years, function(y) {
  cfbd_team_info(year = y, only_fbs = TRUE) %>%
    transmute(year = y, team = school)
})

talent <- talent_raw %>%
  inner_join(fbs_teams, by = c("year", "team"))

cat("FBS-filtered team counts by year:\n")
print(talent %>% count(year))

### Metric 1: blue-chip programs over time (corrected blue_chip_ratio) ###

blue_chip_corrected_path <- file.path(outputs_dir, "blue_chip_ratio_corrected.csv")
if (!file.exists(blue_chip_corrected_path)) {
  stop("Missing ", blue_chip_corrected_path, " -- run build_corrected_blue_chip_ratio.R first.")
}

blue_chip_corrected <- read.csv(blue_chip_corrected_path) %>%
  inner_join(fbs_teams, by = c("year", "team")) %>%
  filter(n_matched >= 20)  # drop small-sample team-seasons (e.g. incomplete roster/recruit matches)

blue_chip_by_year <- blue_chip_corrected %>%
  group_by(year) %>%
  summarise(
    n_teams = n(),
    n_blue_chip = sum(blue_chip_ratio > 0.5, na.rm = TRUE),
    pct_blue_chip = n_blue_chip / n_teams,
    median_bcr = median(blue_chip_ratio),
    mean_bcr = mean(blue_chip_ratio)
  ) %>%
  ungroup()

write.csv(blue_chip_by_year, file.path(outputs_dir, "blue_chip_teams_by_year.csv"), row.names = FALSE)

trend_fit <- lm(mean_bcr ~ year, data = blue_chip_by_year)
trend_r2 <- summary(trend_fit)$r.squared
trend_p <- summary(trend_fit)$coefficients["year", "Pr(>|t|)"]
cat(sprintf("\nMean BCR ~ year trend: R^2 = %.3f, p = %.4f\n", trend_r2, trend_p))

### Metric 2: talent concentration (Gini coefficient) ###

gini_coefficient <- function(x) {
  x <- sort(x[!is.na(x) & x >= 0])
  n <- length(x)
  if (n == 0 || sum(x) == 0) return(NA_real_)
  index <- seq_len(n)
  (2 * sum(index * x) / (n * sum(x))) - (n + 1) / n
}

dispersion_by_year <- talent %>%
  group_by(year) %>%
  summarise(
    gini_talent = gini_coefficient(talent),
    cv_talent = sd(talent, na.rm = TRUE) / mean(talent, na.rm = TRUE)
  ) %>%
  ungroup()

write.csv(dispersion_by_year, file.path(outputs_dir, "talent_dispersion_by_year.csv"), row.names = FALSE)

### Shared visual theme ###

theme_talent <- function() {
  theme_minimal(base_family = "Arial") +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 22, color = "#003366", hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = "#003366", hjust = 0.5),
      plot.caption = element_text(size = 8, color = "black"),
      axis.text = element_text(size = 10, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      plot.background = element_rect(fill = "#F7F7F7", color = NA),
      legend.position = "none"
    )
}

era_labels <- tibble(
  x = c(2018, 2021.5),
  label = c("Transfer Portal\nopens", "NIL begins")
)

add_era_markers <- function(p, y_label) {
  p +
    geom_vline(xintercept = era_labels$x, linetype = "dashed", color = "gray55") +
    annotate(
      "text", x = era_labels$x, y = y_label, label = era_labels$label,
      size = 3, color = "gray35", fontface = "italic", lineheight = 0.9, vjust = 0
    )
}

### Chart 1: blue-chip programs over time ###

y1_max <- max(blue_chip_by_year$pct_blue_chip, na.rm = TRUE)

p1 <- ggplot(blue_chip_by_year, aes(x = year, y = pct_blue_chip)) +
  geom_line(color = "#003366", linewidth = 1.1) +
  geom_point(color = "#003366", size = 3) +
  geom_text(aes(label = n_blue_chip), vjust = -1.3, size = 3.4, fontface = "bold", color = "#003366") +
  scale_x_continuous(breaks = years) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0.08, 0.28))) +
  labs(
    title = "The Rise of Blue-Chip Rosters",
    subtitle = "Share of FBS Teams with a Blue-Chip Ratio (4★/5★ Recruits) Above 50%\n(labels = number of teams)",
    x = NULL, y = "% of FBS Teams",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_talent()

p1 <- add_era_markers(p1, y_label = y1_max * 1.18)

ggsave(file.path(outputs_dir, "blue_chip_teams_by_year.png"), plot = p1, width = 11, height = 7, dpi = 200)

### Chart 2: talent concentration (Gini) ###

y2_max <- max(dispersion_by_year$gini_talent, na.rm = TRUE)
y2_min <- min(dispersion_by_year$gini_talent, na.rm = TRUE)

p2 <- ggplot(dispersion_by_year, aes(x = year, y = gini_talent)) +
  geom_line(color = "#003366", linewidth = 1.1) +
  geom_point(color = "#003366", size = 3) +
  scale_x_continuous(breaks = years) +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.25))) +
  labs(
    title = "Is Talent Getting More Concentrated?",
    subtitle = "Gini Coefficient of Team Talent Composite Across FBS Programs\n(higher = more concentrated among fewer teams, lower = more dispersed)",
    x = NULL, y = "Gini Coefficient",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_talent()

p2 <- add_era_markers(p2, y_label = y2_max + (y2_max - y2_min) * 0.35)

ggsave(file.path(outputs_dir, "talent_concentration_gini.png"), plot = p2, width = 11, height = 7, dpi = 200)

### Chart 3: distribution shape over time (ridgeline) ###

p3 <- ggplot(blue_chip_corrected, aes(x = blue_chip_ratio, y = fct_rev(factor(year)), fill = after_stat(x))) +
  geom_density_ridges_gradient(
    scale = 2.4, rel_min_height = 0.01, color = "white", linewidth = 0.3, from = 0,
    quantile_lines = TRUE, quantiles = 2, vline_color = "#d9534f", vline_linetype = "solid"
  ) +
  scale_fill_gradient(low = "#a6c8e0", high = "#003366", name = "Blue-Chip\nRatio", labels = percent_format(accuracy = 1)) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "The Shape of the Talent Gap",
    subtitle = "Distribution of Blue-Chip Ratio Across FBS Teams, by Season (red = median)",
    x = "Blue-Chip Ratio (Share of Roster: 4★/5★ Recruits)", y = NULL,
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_talent() +
  theme(legend.position = "right")

ggsave(file.path(outputs_dir, "blue_chip_ratio_ridgeline.png"), plot = p3, width = 10, height = 8, dpi = 200)

### Chart 4: median/mean blue-chip ratio trend ###

y4_max <- max(blue_chip_by_year$mean_bcr, na.rm = TRUE)
y4_min <- 0

p4 <- ggplot(blue_chip_by_year, aes(x = year)) +
  geom_line(aes(y = median_bcr, color = "Median"), linewidth = 1.1) +
  geom_point(aes(y = median_bcr, color = "Median"), size = 3) +
  geom_line(aes(y = mean_bcr, color = "Mean"), linewidth = 1.1, linetype = "dashed") +
  geom_point(aes(y = mean_bcr, color = "Mean"), size = 2.5) +
  scale_color_manual(values = c("Median" = "#003366", "Mean" = "#8fa8c4"), name = NULL) +
  scale_x_continuous(breaks = years) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0.08, 0.28))) +
  labs(
    title = "The Center of the League Is Shifting",
    subtitle = sprintf(
      "Median & Mean Blue-Chip Ratio Across FBS Teams by Season (mean trend: R² = %.2f, p = %.3f)",
      trend_r2, trend_p
    ),
    x = NULL, y = "Blue-Chip Ratio",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_talent() +
  theme(legend.position = "top")

p4 <- add_era_markers(p4, y_label = y4_max + (y4_max - y4_min) * 0.35)

ggsave(file.path(outputs_dir, "blue_chip_ratio_median_trend.png"), plot = p4, width = 11, height = 7, dpi = 200)

cat("\nWrote outputs to", outputs_dir, "\n")
