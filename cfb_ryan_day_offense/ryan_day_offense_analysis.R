# Ohio State's offense under Ryan Day (2019-2026): has it gotten slower, less
# explosive, less pass-heavy and less efficient on early downs?
#
# Run pull_osu_offense_data.R first -- this script reads its cached .rds files
# from data/. Writes a multi-page PDF of season trend charts plus a season
# summary CSV to outputs/.
#
# Play filter: offensive scrimmage plays (rushes, dropbacks incl. sacks, and
# fumbled plays). Efficiency metrics drop garbage time (score margin > 38 in
# Q2, > 28 in Q3, > 22 in Q4) because Ohio State plays a lot of blowouts;
# volume metrics (plays/drives/points per game) use every snap.

library(tidyverse)
library(scales)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
data_dir <- file.path(script_dir, "data")
outputs_dir <- file.path(script_dir, "outputs")
dir.create(outputs_dir, showWarnings = FALSE)

pbp_path <- file.path(data_dir, "pbp_2019_2026.rds")
if (!file.exists(pbp_path)) {
  stop("Missing ", pbp_path, " -- run pull_osu_offense_data.R first.")
}

team <- "Ohio State"
pbp_raw <- readRDS(pbp_path)
osu_drives_raw <- readRDS(file.path(data_dir, "osu_drives.rds"))
osu_games <- readRDS(file.path(data_dir, "osu_games.rds"))
fbs_teams <- readRDS(file.path(data_dir, "fbs_teams.rds"))

current_year <- max(osu_games$year)
current_games <- sum(osu_games$year == current_year)
partial_label <- paste0(current_year, " in progress (", current_games, " games)")

### Play-level prep ###

rush_types <- c("Rush", "Rushing Touchdown")
pass_types <- c("Pass Reception", "Pass Incompletion", "Passing Touchdown",
                "Pass Completion", "Sack", "Interception Return",
                "Interception Return Touchdown")
fumble_types <- c("Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
                  "Fumble Return Touchdown")

plays <- pbp_raw %>%
  filter(play_type %in% c(rush_types, pass_types, fumble_types)) %>%
  semi_join(fbs_teams, by = c("year", "pos_team" = "team")) %>%
  # Exact-duplicate rows only: id_play is an 18-digit id stored as a double, so
  # distinct plays can share the same (rounded) id_play value.
  distinct() %>%
  mutate(
    is_rush = play_type %in% rush_types,
    is_pass = play_type %in% pass_types,
    is_sack = play_type == "Sack",
    is_td = play_type %in% c("Rushing Touchdown", "Passing Touchdown"),
    is_turnover = play_type %in% c("Interception Return", "Interception Return Touchdown",
                                   "Fumble Recovery (Opponent)", "Fumble Return Touchdown"),
    margin = abs(pos_score_diff_start),
    garbage = (period == 2 & margin > 38) | (period == 3 & margin > 28) |
      (period >= 4 & margin > 22),
    early_down = down %in% c(1, 2),
    neutral = !is.na(wp_before) & between(wp_before, 0.2, 0.8) & period <= 3,
    explosive = (is_rush & yards_gained >= 10) | (is_pass & yards_gained >= 20),
    explosive_20 = yards_gained >= 20,
    third_conv = down == 3 & (yards_gained >= distance | is_td)
  )

### Team-season metrics (every FBS offense, for the FBS benchmark) ###

games_played <- plays %>% distinct(year, pos_team, game_id) %>% count(year, pos_team, name = "games")

volume <- plays %>%
  group_by(year, pos_team) %>%
  summarise(
    explosive_plays = sum(explosive),
    total_plays = n(),
    .groups = "drop"
  ) %>%
  left_join(games_played, by = c("year", "pos_team")) %>%
  transmute(year, pos_team,
            plays_pg = total_plays / games,
            explosive_pg = explosive_plays / games)

efficiency <- plays %>%
  filter(!garbage) %>%
  group_by(year, pos_team) %>%
  summarise(
    pass_rate = sum(is_pass) / sum(is_pass | is_rush),
    neutral_ed_pass_rate = sum(is_pass & early_down & neutral) /
      sum((is_pass | is_rush) & early_down & neutral),
    ypp = mean(yards_gained, na.rm = TRUE),
    yards_per_dropback = mean(yards_gained[is_pass], na.rm = TRUE),
    yards_per_rush = mean(yards_gained[is_rush], na.rm = TRUE),
    epa_pp = mean(EPA, na.rm = TRUE),
    epa_pass = mean(EPA[is_pass], na.rm = TRUE),
    epa_rush = mean(EPA[is_rush], na.rm = TRUE),
    success_rate = mean(success, na.rm = TRUE),
    ed_success_rate = mean(success[early_down], na.rm = TRUE),
    ed_epa = mean(EPA[early_down], na.rm = TRUE),
    third_down_dist = mean(distance[down == 3], na.rm = TRUE),
    third_conv_rate = mean(third_conv[down == 3]),
    expl_rush_rate = mean(explosive[is_rush]),
    expl_pass_rate = mean(explosive[is_pass]),
    expl_rate_20 = mean(explosive_20),
    sack_rate = sum(is_sack) / sum(is_pass),
    turnover_rate = mean(is_turnover),
    .groups = "drop"
  )

red_zone <- plays %>%
  filter(!is.na(drive_id)) %>%
  group_by(year, pos_team, drive_id) %>%
  summarise(reached_rz = any(yards_to_goal <= 20, na.rm = TRUE),
            td = any(is_td), .groups = "drop") %>%
  filter(reached_rz) %>%
  group_by(year, pos_team) %>%
  summarise(rz_td_rate = mean(td), .groups = "drop")

team_seasons <- volume %>%
  left_join(efficiency, by = c("year", "pos_team")) %>%
  left_join(red_zone, by = c("year", "pos_team"))

### Ohio State-only drive / scoring metrics ###

osu_drives <- osu_drives_raw %>%
  filter(offense == team) %>%
  mutate(
    secs = elapsed_minutes * 60 + elapsed_seconds,
    pts = pmax(end_offense_score - start_offense_score, 0),
    # Kneel-downs / clock-killers at the end of a half aren't real possessions
    kneel = drive_result %in% c("END OF HALF", "END OF GAME", "END OF 4TH QUARTER") & plays <= 3,
    # A handful of CFBD drives carry corrupt clock data (e.g. a 54-minute drive in
    # 2024); keep them for scoring but leave them out of the time/pace metrics.
    bad_clock = secs < 0 | secs > 15 * 60
  )

osu_drive_metrics <- osu_drives %>%
  group_by(year) %>%
  summarise(
    drives_pg = sum(!kneel) / n_distinct(game_id),
    plays_per_drive = mean(plays[!kneel]),
    pts_per_drive = mean(pts[!kneel]),
    td_drive_rate = mean(drive_result[!kneel] %in% c("TD")),
    secs_per_play = sum(secs[!bad_clock]) / sum(plays[!bad_clock]),
    top_pg = sum(secs[!bad_clock]) / 60 / n_distinct(game_id),
    .groups = "drop"
  )

osu_scoring <- osu_games %>%
  group_by(year) %>%
  summarise(games = n(), ppg = mean(points_for), opp_ppg = mean(points_against),
            .groups = "drop")

osu_season <- team_seasons %>%
  filter(pos_team == team) %>%
  select(-pos_team) %>%
  left_join(osu_drive_metrics, by = "year") %>%
  left_join(osu_scoring, by = "year") %>%
  relocate(year, games)

fbs_avg <- team_seasons %>%
  group_by(year) %>%
  summarise(across(-pos_team, ~ mean(.x, na.rm = TRUE)), .groups = "drop")

# Percentile of Ohio State among FBS offenses (flip metrics where lower is better)
lower_is_better <- c("sack_rate", "turnover_rate", "third_down_dist")
osu_pctile <- team_seasons %>%
  pivot_longer(-c(year, pos_team), names_to = "metric", values_to = "value") %>%
  group_by(year, metric) %>%
  mutate(pctile = percent_rank(if_else(metric %in% lower_is_better, -value, value))) %>%
  ungroup() %>%
  filter(pos_team == team)

write_csv(
  osu_season %>% mutate(across(where(is.double), ~ round(.x, 4))),
  file.path(outputs_dir, "osu_offense_season_summary.csv")
)

cat("Ohio State season summary:\n")
print(osu_season %>% select(year, games, ppg, plays_pg, secs_per_play, pass_rate,
                            ypp, expl_rush_rate, expl_pass_rate, ed_success_rate),
      width = Inf)

### Plot helpers ###

navy <- "#003366"
light_navy <- "#8fa8c4"
caption_txt <- paste0("Data Source: @cfbfastR, Viz: @campbell_taylor1\n",
                      "Hollow point = ", partial_label,
                      ". Efficiency metrics exclude garbage time.")

theme_day <- function() {
  theme_minimal(base_family = "Arial") +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 22, color = navy, hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = navy, hjust = 0.5),
      plot.caption = element_text(size = 8, color = "black"),
      axis.text = element_text(size = 10, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      plot.background = element_rect(fill = "#F7F7F7", color = NA),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = 11, face = "bold")
    )
}

fmt_pct <- function(x) percent(x, accuracy = 0.1)
fmt_num <- function(digits) function(x) number(x, accuracy = 10^-digits)

# Season trend chart for one metric: Ohio State line (hollow point for the
# in-progress season) plus an optional dashed FBS-average line.
trend_plot <- function(metric, title, subtitle, fmt = fmt_pct, fbs = TRUE) {
  pick <- function(df, series) {
    df %>% transmute(year, value = .data[[metric]], series = series)
  }
  df <- pick(osu_season, team)
  if (fbs) df <- bind_rows(df, pick(fbs_avg, "FBS Average"))
  df <- df %>%
    mutate(series = factor(series, levels = c(team, "FBS Average")),
           partial = year == current_year)
  osu <- df %>% filter(series == team)

  ggplot(df, aes(year, value, color = series, linetype = series)) +
    geom_line(linewidth = 1.1) +
    geom_point(data = filter(df, series == "FBS Average"), size = 2) +
    geom_point(data = osu, aes(shape = partial), size = 3.5, fill = "white",
               stroke = 1.3, show.legend = FALSE) +
    geom_text(data = osu, aes(label = fmt(value)), vjust = -1.2, size = 4,
              fontface = "bold", color = navy, family = "Arial", show.legend = FALSE) +
    scale_color_manual(values = setNames(c(navy, light_navy), c(team, "FBS Average"))) +
    scale_linetype_manual(values = setNames(c("solid", "dashed"), c(team, "FBS Average"))) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 21)) +
    scale_x_continuous(breaks = sort(unique(df$year)), expand = expansion(add = 0.3)) +
    scale_y_continuous(labels = fmt, expand = expansion(mult = c(0.08, 0.15))) +
    labs(title = title, subtitle = subtitle, x = "Season", y = NULL,
         caption = caption_txt) +
    theme_day() +
    theme(legend.position = if (fbs) "top" else "none")
}

plots <- list()

### Page 1: season summary table ###

summary_rows <- tribble(
  ~metric,                   ~label,                          ~fmt,
  "ppg",                     "Points / game",                 fmt_num(1),
  "pts_per_drive",           "Points / drive",                fmt_num(2),
  "plays_pg",                "Offensive plays / game",        fmt_num(1),
  "secs_per_play",           "Seconds / play (on offense)",   fmt_num(1),
  "top_pg",                  "Time of possession (min)",      fmt_num(1),
  "pass_rate",               "Pass rate",                     fmt_pct,
  "neutral_ed_pass_rate",    "Early-down pass rate (neutral)", fmt_pct,
  "ypp",                     "Yards / play",                  fmt_num(2),
  "epa_pp",                  "EPA / play",                    fmt_num(3),
  "success_rate",            "Success rate",                  fmt_pct,
  "ed_success_rate",         "Early-down success rate",       fmt_pct,
  "expl_rush_rate",          "Explosive rush rate (10+)",     fmt_pct,
  "expl_pass_rate",          "Explosive pass rate (20+)",     fmt_pct,
  "third_conv_rate",         "3rd-down conversion rate",      fmt_pct
)

table_df <- summary_rows %>%
  mutate(row = row_number()) %>%
  rowwise() %>%
  reframe(row, label, metric, year = osu_season$year,
          value = osu_season[[metric]], txt = fmt(osu_season[[metric]])) %>%
  left_join(osu_pctile %>% select(year, metric, pctile), by = c("year", "metric"))

plots$table <- ggplot(table_df, aes(factor(year), -row)) +
  geom_tile(aes(fill = pctile), color = "white", linewidth = 1) +
  geom_text(aes(label = txt), family = "Arial", fontface = "bold", size = 3.8,
            color = if_else(coalesce(table_df$pctile, 0) > 0.75, "white", "black")) +
  geom_text(data = distinct(table_df, row, label), aes(x = 0.4, y = -row, label = label),
            hjust = 1, family = "Arial", fontface = "bold", size = 3.8, color = navy) +
  scale_fill_gradient(low = "#E8ECF1", high = navy, na.value = "#E8ECF1",
                      limits = c(0, 1), labels = percent, name = "FBS percentile") +
  scale_x_discrete(position = "top") +
  coord_cartesian(xlim = c(-1.6, length(unique(table_df$year)) + 0.5), clip = "off") +
  labs(title = "Ohio State Offense Under Ryan Day",
       subtitle = paste0("Season-by-season snapshot, 2019-", current_year,
                         "\n(cell shading = rank among FBS offenses; no shading = Ohio State only metric)"),
       caption = caption_txt) +
  theme_void(base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 22, color = navy, hjust = 0.5),
    plot.subtitle = element_text(size = 12, color = navy, hjust = 0.5,
                                 margin = margin(b = 10)),
    plot.caption = element_text(size = 8, color = "black"),
    axis.text.x.top = element_text(size = 12, face = "bold", color = navy),
    plot.background = element_rect(fill = "#F7F7F7", color = NA),
    plot.margin = margin(15, 20, 10, 20),
    legend.position = "bottom",
    legend.title = element_text(size = 10, face = "bold"),
    legend.key.width = unit(1.5, "cm")
  )

### Trend pages (one metric per page) ###

fmt_ed <- function(x) number(x, accuracy = 0.001)
gt <- "garbage time excluded"

trend_specs <- tribble(
  ~metric, ~title, ~subtitle, ~fmt, ~fbs,
  "ppg", "Points per Game",
  "All games, including conference title, bowl and CFP games", fmt_num(1), FALSE,
  "pts_per_drive", "Points per Drive",
  "Offensive points per possession (drives exclude end-of-half kneel-downs)", fmt_num(2), FALSE,
  "td_drive_rate", "Touchdown Rate per Drive",
  "Share of possessions ending in a touchdown (excl. end-of-half kneel-downs)", fmt_pct, FALSE,
  "plays_pg", "Offensive Plays per Game",
  "Offensive scrimmage plays (rushes + dropbacks), all game states", fmt_num(1), TRUE,
  "drives_pg", "Drives per Game",
  "Offensive possessions per game (excl. end-of-half kneel-downs), Ohio State only", fmt_num(1), FALSE,
  "secs_per_play", "Seconds per Offensive Play",
  "Game clock consumed per play on offense (drive time / drive plays) -- higher = slower\nOhio State only (from CFBD drive data)", fmt_num(1), FALSE,
  "plays_per_drive", "Plays per Drive",
  "Average offensive plays per possession, Ohio State only", fmt_num(1), FALSE,
  "top_pg", "Time of Possession",
  "Minutes of possession per game, Ohio State only", fmt_num(1), FALSE,
  "pass_rate", "Pass Rate",
  paste0("Share of plays that are dropbacks (incl. sacks), ", gt), fmt_pct, TRUE,
  "neutral_ed_pass_rate", "Early-Down Pass Rate (Neutral Script)",
  "Dropback share on 1st and 2nd down\n(neutral = win probability 20-80%, quarters 1-3)", fmt_pct, TRUE,
  "ypp", "Yards per Play",
  paste0("All offensive scrimmage plays, ", gt), fmt_num(2), TRUE,
  "yards_per_dropback", "Yards per Dropback",
  paste0("Passing yards per dropback (sacks included), ", gt), fmt_num(2), TRUE,
  "yards_per_rush", "Yards per Rush",
  paste0("Designed runs and QB scrambles logged as rushes, ", gt), fmt_num(2), TRUE,
  "expl_rush_rate", "Explosive Rush Rate",
  paste0("Share of rushes gaining 10+ yards, ", gt), fmt_pct, TRUE,
  "expl_pass_rate", "Explosive Pass Rate",
  paste0("Share of dropbacks gaining 20+ yards, ", gt), fmt_pct, TRUE,
  "explosive_pg", "Explosive Plays per Game",
  "Rushes of 10+ yards plus dropbacks of 20+ yards per game, all game states", fmt_num(1), TRUE,
  "expl_rate_20", "20+ Yard Play Rate",
  paste0("Share of all plays gaining 20+ yards (the cover model's explosive definition), ", gt), fmt_pct, TRUE,
  "epa_pp", "EPA per Play",
  paste0("Expected points added per offensive play, ", gt), fmt_ed, TRUE,
  "epa_pass", "EPA per Dropback",
  paste0("Expected points added per dropback (sacks included), ", gt), fmt_ed, TRUE,
  "epa_rush", "EPA per Rush",
  paste0("Expected points added per rush, ", gt), fmt_ed, TRUE,
  "success_rate", "Success Rate",
  paste0("Success = 50% of yards to go on 1st, 70% on 2nd, 100% on 3rd/4th; ", gt), fmt_pct, TRUE,
  "ed_success_rate", "Early-Down Success Rate",
  paste0("Success rate on 1st and 2nd down, ", gt), fmt_pct, TRUE,
  "ed_epa", "Early-Down EPA per Play",
  paste0("Expected points added on 1st and 2nd down, ", gt), fmt_ed, TRUE,
  "third_down_dist", "Average Distance on 3rd Down",
  paste0("Yards to go on 3rd down -- weaker early downs mean longer 3rd downs; ", gt), fmt_num(1), TRUE,
  "third_conv_rate", "3rd-Down Conversion Rate",
  paste0("Share of 3rd downs converted to a first down or TD, ", gt), fmt_pct, TRUE,
  "rz_td_rate", "Red-Zone TD Rate",
  "Share of drives reaching the opponent 20 that end in a touchdown", fmt_pct, TRUE,
  "sack_rate", "Sack Rate",
  paste0("Sacks per dropback (lower = better), ", gt), fmt_pct, TRUE,
  "turnover_rate", "Turnover Rate",
  paste0("Interceptions + lost fumbles per play (lower = better), ", gt), fmt_pct, TRUE
)

trend_plots <- pmap(trend_specs, trend_plot) %>% set_names(trend_specs$metric)

game_plays <- plays %>%
  filter(pos_team == team) %>%
  count(year, game_id, name = "plays") %>%
  left_join(osu_games %>% select(game_id, points_for, points_against), by = "game_id") %>%
  mutate(game_type = if_else(points_for - points_against >= 21, "Won by 21+", "Everything else"))

plays_games_plot <- ggplot(game_plays, aes(factor(year), plays)) +
  geom_boxplot(fill = "#E8ECF1", color = navy, outlier.shape = NA, width = 0.55) +
  geom_jitter(aes(color = game_type), width = 0.15, size = 2.6, alpha = 0.85) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.5, fill = "white",
               color = navy, stroke = 1.2) +
  scale_color_manual(values = c("Won by 21+" = light_navy, "Everything else" = "#c0392b")) +
  labs(title = "Snaps per Game, Game by Game",
       subtitle = "Each dot is one game; diamond = season average\n(blowouts shorten the game for the starters -- red dots show competitive/closer games)",
       x = "Season", y = "Offensive plays", caption = caption_txt) +
  theme_day()

# The game-by-game snaps chart follows the plays-per-game trend page
metric_pages <- append(trend_plots, list(plays_games = plays_games_plot),
                       after = which(names(trend_plots) == "plays_pg"))

### Relative-to-FBS heatmap ###

heat_metrics <- summary_rows %>%
  filter(metric %in% osu_pctile$metric) %>%
  bind_rows(tribble(~metric, ~label,
                    "explosive_pg", "Explosive plays / game",
                    "rz_td_rate", "Red-zone TD rate",
                    "sack_rate", "Sack rate (lower = better)",
                    "turnover_rate", "Turnover rate (lower = better)"))

heat_df <- osu_pctile %>%
  inner_join(heat_metrics %>% select(metric, label), by = "metric") %>%
  mutate(label = factor(label, levels = rev(heat_metrics$label)))

plots$heatmap <- ggplot(heat_df, aes(factor(year), label, fill = pctile)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = round(pctile * 100)), family = "Arial", fontface = "bold", size = 4,
            color = if_else(heat_df$pctile > 0.6, "white", "black")) +
  scale_fill_gradient2(low = "#c0392b", mid = "#E8ECF1", high = navy, midpoint = 0.5,
                       limits = c(0, 1), labels = percent, name = "FBS percentile") +
  labs(title = "Relative to the Rest of FBS",
       subtitle = "Ohio State's percentile among FBS offenses each season (100 = best)\nSeparates \"Ohio State got worse\" from \"everyone changed\" -- plays/game and pass rates are style, not quality (100 = most)",
       x = NULL, y = NULL, caption = caption_txt) +
  theme_day() +
  theme(panel.grid = element_blank(), legend.position = "right",
        legend.title = element_text(size = 10, face = "bold"),
        axis.text.x = element_text(size = 12, face = "bold", color = navy))

### Write PDF ###

pdf_path <- file.path(outputs_dir, "ryan_day_offense_trends.pdf")
# Quartz (macOS) rather than cairo_pdf: embeds system Arial without needing XQuartz.
quartz(type = "pdf", file = pdf_path, width = 11, height = 7)
pages <- c(list(plots$table), metric_pages, list(plots$heatmap))
walk(pages, print)
dev.off()

cat("Wrote", pdf_path, "\n")
