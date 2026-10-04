# 2026 FBS quarterback cumulative EPA tracker.
#
# Reads the cache written by pull_qb_data.R, attributes every dropback and QB
# rush to a quarterback (ESPN athlete ID), and builds week-by-week cumulative
# EPA, EPA/play and success rate for each qualifying FBS starter.
#
# Outputs:
#   outputs/qb_epa_<season>.csv          one row per QB-week (week 0 = start)
#   outputs/qb_epa_<season>.json         same data, nested per QB, for the page
#   outputs/qb_cumulative_epa_<season>.png  static ggplot with headshots
#   outputs/qb_epa_tracker.html          interactive page (template + JSON)

library(tidyverse)
library(ggimage)
library(jsonlite)
library(httr)

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
data_dir <- file.path(script_dir, "data")
out_dir <- file.path(script_dir, "outputs")
dir.create(out_dir, showWarnings = FALSE)

season <- as.integer(Sys.getenv("CFB_SEASON", "2026"))
min_db_per_game <- 15   # qualifier: dropbacks per team game played
power4_confs <- c("SEC", "Big Ten", "Big 12", "ACC")

pbp <- readRDS(file.path(data_dir, paste0("pbp_", season, ".rds")))
qb_roster <- readRDS(file.path(data_dir, paste0("qb_roster_", season, ".rds")))
fbs_teams <- readRDS(file.path(data_dir, paste0("fbs_teams_", season, ".rds")))

### Plays: FBS offenses, garbage time removed ###

# Same garbage-time thresholds as cfb_ryan_day_offense.
plays <- pbp %>%
  filter(pos_team %in% fbs_teams$team, pass == 1 | rush == 1, !is.na(EPA)) %>%
  mutate(
    margin = abs(pos_score_diff_start),
    garbage = (period == 2 & margin > 38) | (period == 3 & margin > 28) |
      (period == 4 & margin > 22),
    across(ends_with("_player_id"), ~ if_else(is.na(.x), NA_character_, format(.x, scientific = FALSE, trim = TRUE)))
  ) %>%
  filter(!garbage)

### Attribute plays to a QB ###

dropbacks <- plays %>%
  filter(pass == 1) %>%
  mutate(qb_id = coalesce(completion_player_id, incompletion_player_id,
                          interception_thrown_player_id, sack_taken_player_id))

# ~2% of dropbacks have no ID tag but do carry passer_player_name; map those
# through the ID that name is most often tagged with on the same team.
name_to_id <- dropbacks %>%
  filter(!is.na(qb_id), !is.na(passer_player_name)) %>%
  count(pos_team, passer_player_name, qb_id) %>%
  group_by(pos_team, passer_player_name) %>%
  slice_max(n, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(pos_team, passer_player_name, name_id = qb_id)

dropbacks <- dropbacks %>%
  left_join(name_to_id, by = c("pos_team", "passer_player_name")) %>%
  mutate(qb_id = coalesce(qb_id, name_id)) %>%
  select(-name_id) %>%
  filter(!is.na(qb_id))

# A QB is anyone listed at QB on a roster, or anyone with 20+ dropbacks
# (covers roster gaps). QB rushes include scrambles, which cfbfastR codes as rushes.
qb_ids <- union(qb_roster$athlete_id,
                dropbacks %>% count(qb_id) %>% filter(n >= 20) %>% pull(qb_id))

qb_rushes <- plays %>%
  filter(rush == 1, rush_player_id %in% qb_ids) %>%
  mutate(qb_id = rush_player_id)

qb_plays <- bind_rows(
  dropbacks %>% mutate(kind = "dropback"),
  qb_rushes %>% mutate(kind = "rush")
) %>%
  mutate(
    attempt = kind == "dropback" & sack == 0,
    interception = kind == "dropback" & !is.na(interception_thrown_player_id)
  )

### QB-week stats ###

qb_week <- qb_plays %>%
  group_by(qb_id, pos_team, week) %>%
  summarise(
    opponent = first(def_pos_team),
    plays = n(),
    dropbacks = sum(kind == "dropback"),
    rushes = sum(kind == "rush"),
    epa = sum(EPA),
    success = sum(success, na.rm = TRUE),
    attempts = sum(attempt),
    completions = sum(attempt & completion == 1, na.rm = TRUE),
    pass_yards = sum(if_else(attempt, yards_gained, 0), na.rm = TRUE),
    sacks = sum(kind == "dropback" & sack == 1),
    interceptions = sum(interception),
    .groups = "drop"
  )

# A QB's team is the one he took the most snaps for.
qb_team <- qb_week %>%
  group_by(qb_id, pos_team) %>%
  summarise(plays = sum(plays), .groups = "drop") %>%
  group_by(qb_id) %>%
  slice_max(plays, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(qb_id, team = pos_team)

qb_week <- qb_week %>% semi_join(qb_team, by = c("qb_id", "pos_team" = "team"))

### Team schedule (games played by week, opponent) ###

team_games <- pbp %>%
  filter(pos_team %in% fbs_teams$team) %>%
  distinct(team = pos_team, week, game_id, opponent = def_pos_team) %>%
  group_by(team, week) %>%
  summarise(opponent = first(opponent), .groups = "drop")

max_week <- max(team_games$week)
games_played <- team_games %>% count(team, name = "team_games")

### Qualifier: primary starters ###

qualifiers <- qb_week %>%
  group_by(qb_id) %>%
  summarise(dropbacks = sum(dropbacks), .groups = "drop") %>%
  inner_join(qb_team, by = "qb_id") %>%
  inner_join(games_played, by = "team") %>%
  filter(dropbacks >= min_db_per_game * team_games)

### Weekly grid with cumulative stats (week 0 = season start) ###

# Cumulative values at week w only use plays from weeks <= w. Weeks a QB did
# not play (bye or DNP) carry the prior cumulative value forward.
qb_grid <- qualifiers %>%
  select(qb_id, team) %>%
  crossing(week = 0:max_week) %>%
  left_join(team_games, by = c("team", "week")) %>%
  left_join(qb_week %>% select(-opponent, -pos_team), by = c("qb_id", "week")) %>%
  mutate(
    status = case_when(week == 0 ~ "start", !is.na(plays) ~ "played",
                       !is.na(opponent) ~ "dnp", TRUE ~ "bye"),
    across(c(plays, dropbacks, rushes, epa, success, attempts, completions,
             pass_yards, sacks, interceptions), ~ replace_na(.x, 0))
  ) %>%
  arrange(qb_id, week) %>%
  group_by(qb_id) %>%
  mutate(
    cum_plays = cumsum(plays),
    cum_dropbacks = cumsum(dropbacks),
    cum_epa = cumsum(epa),
    cum_epa_play = if_else(cum_plays > 0, cum_epa / cum_plays, NA_real_),
    cum_success_rate = if_else(cum_plays > 0, cumsum(success) / cum_plays, NA_real_),
    cum_comp_pct = if_else(cumsum(attempts) > 0, cumsum(completions) / cumsum(attempts), NA_real_),
    cum_ypa = if_else(cumsum(attempts) > 0, cumsum(pass_yards) / cumsum(attempts), NA_real_),
    cum_sacks = cumsum(sacks),
    cum_ints = cumsum(interceptions),
    week_epa_play = if_else(plays > 0, epa / plays, NA_real_),
    week_success_rate = if_else(plays > 0, success / plays, NA_real_)
  ) %>%
  ungroup()

### QB identity: name, conference, colors, headshot ###

passer_names <- qb_plays %>%
  mutate(name = coalesce(passer_player_name, rusher_player_name)) %>%
  filter(!is.na(name)) %>%
  count(qb_id, name) %>%
  group_by(qb_id) %>%
  slice_max(n, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(qb_id, pbp_name = name)

espn_headshot <- function(id) {
  paste0("https://a.espncdn.com/i/headshots/college-football/players/full/", id, ".png")
}

# Downloads an image once into data/images/ and returns the local path, or NA
# when the URL fails (not every ESPN headshot exists).
cache_image <- function(url, key) {
  path <- file.path(data_dir, "images", paste0(key, ".png"))
  if (file.exists(path)) return(path)
  res <- tryCatch(GET(url, timeout(15)), error = function(e) NULL)
  if (is.null(res) || status_code(res) != 200) return(NA_character_)
  writeBin(content(res, "raw"), path)
  path
}

# Square 96px crop (headshots keep the top of the frame) as a base64 data URI.
# The published page can't load images from other hosts, so they ship inline.
image_data_uri <- function(path, headshot) {
  img <- magick::image_read(path)
  info <- magick::image_info(img)
  side <- min(info$width, info$height)
  x_off <- (info$width - side) %/% 2
  y_off <- if (headshot) 0 else (info$height - side) %/% 2
  img <- img %>%
    magick::image_crop(paste0(side, "x", side, "+", x_off, "+", y_off)) %>%
    magick::image_resize("96x96")
  paste0("data:image/webp;base64,",
         base64_enc(magick::image_write(img, format = "webp", quality = 80)))
}
dir.create(file.path(data_dir, "images"), showWarnings = FALSE)

qb_info <- qualifiers %>%
  select(qb_id, team) %>%
  left_join(qb_roster %>% distinct(athlete_id, .keep_all = TRUE) %>% select(-team),
            by = c("qb_id" = "athlete_id")) %>%
  left_join(passer_names, by = "qb_id") %>%
  left_join(fbs_teams, by = "team") %>%
  mutate(
    name = if_else(!is.na(first_name), paste(first_name, last_name), pbp_name),
    group = if_else(conference %in% power4_confs | team == "Notre Dame", "Power 4", "Group of 5"),
    headshot_url = coalesce(na_if(headshot_url, ""), espn_headshot(qb_id)),
    color = paste0("#", str_remove(coalesce(color, "#888888"), "^#")),
    alt_color = paste0("#", str_remove(coalesce(alt_color, "#888888"), "^#"))
  )

message("Caching ", nrow(qb_info), " headshots")
qb_info <- qb_info %>%
  mutate(
    headshot_path = map2_chr(headshot_url, paste0("qb_", qb_id), cache_image),
    logo_path = map2_chr(logo, paste0("logo_", str_replace_all(team, "[^A-Za-z0-9]", "_")), cache_image),
    has_headshot = !is.na(headshot_path),
    image_path = coalesce(headshot_path, logo_path),
    image = map2_chr(image_path, has_headshot,
                     ~ if (is.na(.x)) NA_character_ else image_data_uri(.x, .y)),
    logo = map_chr(logo_path, ~ if (is.na(.x)) NA_character_ else image_data_uri(.x, FALSE))
  ) %>%
  select(qb_id, name, team, conference, group, color, alt_color, logo, image, image_path, has_headshot)

qb_long <- qb_grid %>%
  left_join(qb_info %>% select(qb_id, name, conference, group), by = "qb_id") %>%
  relocate(qb_id, name, team, conference, group, week, status, opponent)

write_csv(qb_long, file.path(out_dir, paste0("qb_epa_", season, ".csv")))

### JSON for the interactive page ###

week_cols <- c("week", "status", "opponent", "plays", "dropbacks", "epa", "cum_epa",
               "cum_epa_play", "cum_success_rate", "cum_comp_pct", "cum_ypa",
               "cum_dropbacks", "cum_sacks", "cum_ints", "week_epa_play",
               "week_success_rate")

qb_json <- qb_info %>%
  select(-image_path) %>%
  arrange(name) %>%
  mutate(weeks = map(qb_id, function(id) {
    qb_grid %>%
      filter(qb_id == id) %>%
      select(all_of(week_cols)) %>%
      mutate(across(where(is.double), ~ round(.x, 4)))
  }))

payload <- list(
  season = season,
  max_week = max_week,
  min_db_per_game = min_db_per_game,
  updated = format(Sys.Date(), "%Y-%m-%d"),
  qbs = qb_json
)
json_txt <- toJSON(payload, dataframe = "rows", auto_unbox = TRUE, na = "null", digits = NA)
write(json_txt, file.path(out_dir, paste0("qb_epa_", season, ".json")))

template_path <- file.path(script_dir, "qb_epa_tracker_template.html")
if (file.exists(template_path)) {
  template <- paste(readLines(template_path, warn = FALSE), collapse = "\n")
  html <- sub("/*__QB_DATA__*/null", json_txt, template, fixed = TRUE)
  writeLines(html, file.path(out_dir, "qb_epa_tracker.html"))
}

### Static ggplot with headshots ###

# WCAG contrast ratio between two hex colors.
contrast_ratio <- function(a, b) {
  lum <- function(hex) {
    ch <- grDevices::col2rgb(hex)[, 1] / 255
    ch <- ifelse(ch <= 0.03928, ch / 12.92, ((ch + 0.055) / 1.055)^2.4)
    sum(c(0.2126, 0.7152, 0.0722) * ch)
  }
  l <- sort(c(lum(a), lum(b)), decreasing = TRUE)
  (l[1] + 0.05) / (l[2] + 0.05)
}

# A team's line color: primary if it reads on the chart surface, else the
# alternate, else the primary darkened until it reaches 3:1.
readable_team_color <- function(color, alt, surface = "#fcfcfb", min_ratio = 3) {
  for (cand in c(color, alt)) if (contrast_ratio(cand, surface) >= min_ratio) return(cand)
  col <- color
  for (i in 1:20) {
    col <- grDevices::adjustcolor(colorspace::darken(col, 0.1), alpha.f = 1)
    col <- substr(col, 1, 7)
    if (contrast_ratio(col, surface) >= min_ratio) break
  }
  col
}

# Spreads label positions apart so no two are closer than `gap`, keeping
# them as close to their original values as possible.
dodge_positions <- function(y, gap) {
  ord <- order(y)
  ys <- y[ord]
  for (iter in 1:200) {
    moved <- FALSE
    for (i in seq_along(ys)[-1]) {
      overlap <- gap - (ys[i] - ys[i - 1])
      if (overlap > 1e-9) {
        ys[i - 1] <- ys[i - 1] - overlap / 2
        ys[i] <- ys[i] + overlap / 2
        moved <- TRUE
      }
    }
    if (!moved) break
  }
  out <- numeric(length(y))
  out[ord] <- ys
  out
}

final <- qb_grid %>%
  filter(week == max_week) %>%
  left_join(qb_info, by = c("qb_id", "team")) %>%
  arrange(desc(cum_epa)) %>%
  mutate(tier = case_when(row_number() <= 10 ~ "Top 10",
                          row_number() > n() - 10 ~ "Bottom 10",
                          TRUE ~ "Other"))

final$line_color <- map2_chr(final$color, final$alt_color, readable_team_color)
highlight <- final %>% filter(tier != "Other")
y_range <- diff(range(qb_grid$cum_epa))
highlight$label_y <- dodge_positions(highlight$cum_epa, gap = y_range * 0.034)

plot_data <- qb_grid %>%
  left_join(final %>% select(qb_id, tier, line_color), by = "qb_id")

img_x <- max_week + 0.45
label_gap <- y_range * 0.034
group_heads <- highlight %>%
  group_by(tier) %>%
  summarise(y = max(label_y) + label_gap, .groups = "drop") %>%
  mutate(label = toupper(tier))

p <- ggplot(plot_data, aes(week, cum_epa, group = qb_id)) +
  geom_hline(yintercept = 0, color = "#c3c2b7", linewidth = 0.4) +
  geom_line(data = filter(plot_data, tier == "Other"), color = "#d6d5ce", linewidth = 0.35) +
  geom_line(data = filter(plot_data, tier != "Other"), aes(color = line_color), linewidth = 0.8) +
  geom_point(data = filter(plot_data, tier != "Other", week == max_week),
             aes(color = line_color), size = 1.6) +
  geom_segment(data = highlight,
               aes(x = max_week, xend = img_x - 0.12, y = cum_epa, yend = label_y, color = line_color),
               linewidth = 0.3) +
  geom_text(data = group_heads, aes(x = img_x - 0.1, y = y, label = label), hjust = 0,
            size = 2.6, fontface = "bold", color = "#898781", inherit.aes = FALSE) +
  geom_image(data = highlight, aes(x = img_x, y = label_y, image = image_path),
             size = 0.034, asp = 1.5, inherit.aes = FALSE) +
  geom_text(data = highlight,
            aes(x = img_x + 0.18, y = label_y,
                label = paste0(name, "  ", sprintf("%+.1f", cum_epa))),
            hjust = 0, size = 2.9, color = "#0b0b0b", inherit.aes = FALSE) +
  scale_color_identity() +
  scale_x_continuous(breaks = 0:max_week, labels = c("Start", paste("Wk", 1:max_week)),
                     expand = expansion(add = c(0.15, 2.4))) +
  labs(
    title = paste(season, "FBS quarterbacks: cumulative EPA by week"),
    subtitle = paste0("Dropbacks + QB rushes, garbage time removed. ", nrow(qualifiers),
                      " qualifying starters (", min_db_per_game,
                      "+ dropbacks per team game). Top and bottom 10 highlighted in team colors."),
    x = NULL, y = "Cumulative EPA",
    caption = "Data: cfbfastR / CollegeFootballData.com | Headshots: ESPN"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background = element_rect(fill = "#fcfcfb", color = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#e1e0d9", linewidth = 0.3),
    legend.position = "top",
    legend.justification = "left",
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "#52514e", size = 9),
    plot.caption = element_text(color = "#898781", size = 7.5),
    axis.text = element_text(color = "#52514e")
  )

ggsave(file.path(out_dir, paste0("qb_cumulative_epa_", season, ".png")), p,
       width = 12, height = 8, dpi = 200, bg = "#fcfcfb")

### Sanity checks ###

cat("Max week:", max_week, "\n")
cat("Qualifying QBs:", nrow(qualifiers), "(",
    sum(qb_info$group == "Power 4"), "Power 4 /", sum(qb_info$group == "Group of 5"), "Group of 5 )\n")
cat("Headshots found:", sum(qb_info$has_headshot), "of", nrow(qb_info), "(rest use team logo)\n")
cat("Unattributed dropbacks dropped:",
    sum(plays$pass == 1) - nrow(dropbacks), "of", sum(plays$pass == 1), "\n")
spot <- final$qb_id[1]
cat("Spot check", final$name[1], ": cumulative EPA", round(final$cum_epa[1], 2),
    "vs raw play sum", round(sum(qb_plays$EPA[qb_plays$qb_id == spot & qb_plays$pos_team == final$team[1]]), 2), "\n")
cat("\nTop 10 / bottom 10 cumulative EPA through week", max_week, ":\n")
print(final %>% filter(tier != "Other") %>%
        transmute(tier, name, team, cum_epa = round(cum_epa, 1),
                  epa_play = round(cum_epa_play, 3), sr = round(cum_success_rate, 3),
                  plays = cum_plays), n = 20)
