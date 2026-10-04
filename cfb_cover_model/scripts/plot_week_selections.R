# Week cover-pick card: agreement bets (both logistic regression and XGBoost regressor
# flag a home cover) with a posted spread, drawn as a ggplot table with team logos.
#
# The prediction CSV stores |spread|, so the signed line (negative = home favored) comes
# from the cached CFBD lines in data/raw/betting_lines/ -- consensus provider when there
# is one, otherwise the provider average, matching consensus_or_average_spread()
# (src/cfb_cover_model/ingest/box_score_features.py).
#
# Usage (from the repo root):
#   Rscript cfb_cover_model/scripts/plot_week_selections.R --season 2026 --week 4

library(tidyverse)
library(cfbfastR)
library(ggimage)
library(arrow)
library(scales)

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) stop("Missing ", flag, " -- usage: --season 2026 --week 4")
  as.integer(args[i + 1])
}
SEASON <- get_arg("--season")
WEEK <- get_arg("--week")

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
project_dir <- normalizePath(file.path(script_dir, ".."))
pred_dir <- file.path(project_dir, "outputs", "predictions")

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
  stop("CFBD_API_KEY not found -- expected it in the repo root's .env. Run this script from the repo root.")
}

### Picks + signed lines ###

signed_lines <- function(path) {
  lines <- read_parquet(path)
  consensus <- lines %>%
    filter(tolower(provider) == "consensus") %>%
    distinct(game_id, .keep_all = TRUE) %>%
    select(game_id, line = spread)
  lines %>%
    group_by(game_id) %>%
    summarise(line = mean(spread, na.rm = TRUE), .groups = "drop") %>%
    anti_join(consensus, by = "game_id") %>%
    bind_rows(consensus)
}

week_tag <- sprintf("%d_%02d", SEASON, WEEK)
lines_path <- file.path(project_dir, "data", "raw", "betting_lines", paste0(week_tag, ".parquet"))
# (macOS strftime lacks %-d, so strip leading zeros by hand)
lines_pulled <- gsub(" 0", " ", format(file.mtime(lines_path), "%b %d, %I:%M %p"))

preds <- read_csv(file.path(pred_dir, sprintf("live_%d_week_%d_dual_model_predictions.csv", SEASON, WEEK)),
                  show_col_types = FALSE)
artifact_version <- first(preds$artifact_version)

picks <- preds %>%
  filter(agreement_bet, !is.na(spread)) %>%
  left_join(signed_lines(lines_path), by = "game_id") %>%
  arrange(desc(avg_probability))

### Logos ###

teams <- cfbd_team_info(year = SEASON, only_fbs = FALSE) %>%
  select(school, logo) %>%
  mutate(logo = map_chr(logo, ~ if (length(.x) > 0) .x[[1]] else NA_character_)) %>%
  distinct(school, .keep_all = TRUE)

picks <- picks %>%
  left_join(teams %>% rename(home_logo = logo), by = c("home_team" = "school")) %>%
  left_join(teams %>% rename(away_logo = logo), by = c("away_team" = "school"))

missing_logos <- c(picks$home_team[is.na(picks$home_logo)], picks$away_team[is.na(picks$away_logo)])
if (length(missing_logos) > 0) {
  warning("No logo match for: ", paste(unique(missing_logos), collapse = ", "))
}

### Card ###

fmt_line <- function(x) {
  case_when(is.na(x) ~ "--", x == 0 ~ "PK", TRUE ~ sprintf("%+g", round(x * 2) / 2))
}

n <- nrow(picks)
if (n == 0) stop("No agreement bets with a posted spread for season ", SEASON, " week ", WEEK)

picks <- picks %>%
  mutate(
    row_y = n:1,
    line_label = fmt_line(line)
  )

navy <- "#003366"
x_rank <- 0.25
x_home_logo <- 0.85
x_home <- 1.3
x_opp_logo <- 3.85
x_opp <- 4.3
x_line <- 6.9
x_lr <- x_line + 0.9
x_xgb <- x_lr + 0.8
x_bar <- x_xgb + 0.55
bar_w <- 1.6
x_max <- x_bar + bar_w + 0.75

header_y <- n + 1
zebra <- picks %>% filter(row_y %% 2 == 0)

p <- ggplot(picks) +
  geom_rect(data = zebra, aes(ymin = row_y - 0.5, ymax = row_y + 0.5), xmin = -0.1, xmax = x_max,
            fill = "#E8ECF1", color = NA) +
  geom_rect(data = data.frame(x = 1), ymin = header_y - 0.5, ymax = header_y + 0.5, xmin = -0.1,
            xmax = x_max, fill = navy, color = NA) +
  geom_text(aes(x = x_rank, y = row_y, label = seq_len(n)), hjust = 0, fontface = "bold",
            size = 3.3, color = navy) +
  geom_image(aes(x = x_home_logo, y = row_y, image = home_logo), size = 0.045, by = "height") +
  geom_text(aes(x = x_home, y = row_y, label = home_team), hjust = 0, fontface = "bold", size = 3.3) +
  geom_text(aes(x = x_opp_logo - 0.35, y = row_y, label = "vs"), size = 2.8, color = "gray35",
            fontface = "italic") +
  geom_image(aes(x = x_opp_logo, y = row_y, image = away_logo), size = 0.045, by = "height") +
  geom_text(aes(x = x_opp, y = row_y, label = away_team), hjust = 0, size = 3.1) +
  geom_text(aes(x = x_line, y = row_y, label = line_label), fontface = "bold", size = 3.3, color = navy) +
  geom_text(aes(x = x_lr, y = row_y, label = percent(logistic_regression_probability, accuracy = 1)),
            size = 3.1) +
  geom_text(aes(x = x_xgb, y = row_y, label = percent(xgboost_regressor_probability, accuracy = 1)),
            size = 3.1) +
  geom_rect(aes(ymin = row_y - 0.22, ymax = row_y + 0.22), xmin = x_bar, xmax = x_bar + bar_w,
            fill = "#D5DDE7", color = NA) +
  geom_rect(aes(xmax = x_bar + bar_w * avg_probability, ymin = row_y - 0.22, ymax = row_y + 0.22),
            xmin = x_bar, fill = navy, color = NA) +
  geom_text(aes(x = x_bar + bar_w + 0.1, y = row_y, label = percent(avg_probability, accuracy = 0.1)),
            hjust = 0, fontface = "bold", size = 3.2, color = navy) +
  annotate("text", x = x_rank, y = header_y, label = "#", hjust = 0, fontface = "bold", size = 3.3, color = "white") +
  annotate("text", x = x_home_logo - 0.25, y = header_y, label = "Pick (Home)", hjust = 0, fontface = "bold",
           size = 3.3, color = "white") +
  annotate("text", x = x_opp_logo - 0.25, y = header_y, label = "Opponent", hjust = 0, fontface = "bold",
           size = 3.3, color = "white") +
  annotate("text", x = x_line, y = header_y, label = "Line", fontface = "bold", size = 3.3, color = "white") +
  annotate("text", x = x_lr, y = header_y, label = "Logit", fontface = "bold", size = 3.3, color = "white") +
  annotate("text", x = x_xgb, y = header_y, label = "XGB", fontface = "bold", size = 3.3, color = "white") +
  annotate("text", x = x_bar, y = header_y, label = "Avg Cover Prob.", hjust = 0, fontface = "bold",
           size = 3.3, color = "white")

subtitle <- paste0(
  "Week ", WEEK, ", ", SEASON, " -- home team to cover\n",
  "Line = home spread (negative = home favored), pulled ", lines_pulled
)

p <- p +
  scale_x_continuous(limits = c(-0.1, x_max), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.3, header_y + 0.6), expand = c(0, 0)) +
  labs(title = "Campbell Taylor College Football Model Picks", subtitle = subtitle,
       caption = paste0("Data Source: @cfbfastR, Model: cfb_cover_model (", artifact_version,
                        "), Viz: @campbell_taylor1")) +
  theme_void(base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 18, color = navy, hjust = 0.5, margin = margin(b = 6)),
    plot.subtitle = element_text(size = 9, color = navy, hjust = 0.5, margin = margin(b = 8)),
    plot.caption = element_text(size = 7.5, color = "black", margin = margin(t = 8)),
    plot.background = element_rect(fill = "#F7F7F7", color = NA),
    plot.margin = margin(12, 18, 10, 18)
  )

out_path <- file.path(pred_dir, sprintf("live_%d_week_%d_selections.png", SEASON, WEEK))
ggsave(out_path, plot = p, width = 11, height = max(3.5, 1.4 + (n + 1) * 0.36), dpi = 200)
cat("Wrote", out_path, "(", n, "picks)\n")
