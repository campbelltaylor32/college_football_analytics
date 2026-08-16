# Transfer portal flow: which conferences are net winners/losers of
# recruiting talent through transfer-portal movement, 2021-2025 (the only
# seasons CFBD has transfer-portal records for -- verified live, 0 rows
# 2015-2020).
#
# Uses cfbfastR::cfbd_recruiting_transfer_portal(), a dedicated
# origin/destination/rating endpoint not used anywhere else in this repo
# (the existing cfb_win_total_model roster-turnover feature only infers
# transfers by diffing rosters year-over-year, with no team-pair or rating
# data). Conference assignment comes from cfbd_team_info(year, only_fbs =
# TRUE), pulled fresh here.
#
# The flow chart is a circular diagram, not a geographic map -- conference
# realignment during this window (Oklahoma/Texas to the SEC, the Pac-12's
# collapse, etc.) makes "where a conference is" a moving target with no
# stable real-world position, and geography isn't actually informative for
# conference-to-conference flow anyway. Nodes are instead arranged around a
# circle ordered by net talent, so net exporters and net importers cluster
# on opposite sides and the "who benefits" story reads directly from
# position.
#
# Writes the circular flow diagram and two net-talent bar charts to outputs/.

library(tidyverse)
library(cfbfastR)
library(scales)

Sys.setenv(CFBD_API_KEY = Sys.getenv("CFBD_API_KEY"))

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
script_dir <- if (length(script_path) > 0) normalizePath(dirname(script_path)) else getwd()
outputs_dir <- file.path(script_dir, "outputs")
dir.create(outputs_dir, showWarnings = FALSE)

years <- 2021:2025
NON_FBS <- "Non-FBS/Other"

### Pull transfer portal records ###
cat("Pulling transfer portal data...\n")
portal_raw <- map_dfr(years, function(y) {
  cat(" ", y, "\n")
  cfbd_recruiting_transfer_portal(year = y)
})

portal <- portal_raw %>%
  filter(!is.na(origin), !is.na(destination), origin != destination) %>%
  transmute(season = season, origin, destination, rating = as.numeric(rating), stars)

cat("Transfers with both origin and destination:", nrow(portal), "\n")

### Pull team info (conference) per season ###
cat("Pulling team info (conference)...\n")
team_info <- map_dfr(years, function(y) {
  cfbd_team_info(year = y, only_fbs = TRUE) %>%
    transmute(season = y, team = school, conference)
})

### Join origin/destination to conference; unmatched schools -> Non-FBS/Other ###
portal_conf <- portal %>%
  left_join(team_info %>% select(season, team, conference), by = c("season", "origin" = "team")) %>%
  rename(origin_conf = conference) %>%
  left_join(team_info %>% select(season, team, conference), by = c("season", "destination" = "team")) %>%
  rename(dest_conf = conference) %>%
  mutate(
    origin_conf = coalesce(origin_conf, NON_FBS),
    dest_conf = coalesce(dest_conf, NON_FBS)
  )

write_csv(portal_conf, file.path(outputs_dir, "transfer_portal_edges_raw.csv"))

### Per-conference net talent totals ###
talent_out <- portal_conf %>% group_by(conference = origin_conf) %>%
  summarise(talent_out = sum(rating, na.rm = TRUE), n_out = n(), .groups = "drop")
talent_in <- portal_conf %>% group_by(conference = dest_conf) %>%
  summarise(talent_in = sum(rating, na.rm = TRUE), n_in = n(), .groups = "drop")

conf_totals <- full_join(talent_in, talent_out, by = "conference") %>%
  mutate(across(c(talent_in, talent_out, n_in, n_out), ~ replace_na(., 0))) %>%
  mutate(net_talent = talent_in - talent_out, activity = n_in + n_out)

write_csv(conf_totals, file.path(outputs_dir, "net_talent_by_conference.csv"))

### Conference-pair edges (excluding intra-conference moves) ###
pair_edges <- portal_conf %>%
  filter(origin_conf != dest_conf) %>%
  mutate(conf_a = pmin(origin_conf, dest_conf), conf_b = pmax(origin_conf, dest_conf)) %>%
  group_by(conf_a, conf_b) %>%
  summarise(
    total_transfers = n(),
    total_talent = sum(rating, na.rm = TRUE),
    # net flow from a -> b (positive means net movement conf_a -> conf_b)
    net_flow_a_to_b = sum(rating[origin_conf == conf_a & dest_conf == conf_b], na.rm = TRUE) -
      sum(rating[origin_conf == conf_b & dest_conf == conf_a], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    importer = ifelse(net_flow_a_to_b >= 0, conf_b, conf_a),
    exporter = ifelse(net_flow_a_to_b >= 0, conf_a, conf_b)
  )

write_csv(pair_edges, file.path(outputs_dir, "conference_pair_edges.csv"))

### Shared visual theme ###
theme_portal <- function() {
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

### Chart 1: circular conference flow diagram ###

# Order nodes around the circle by net_talent (ascending) rather than
# geography or alphabetically -- net exporters cluster on one side, net
# importers on the other, so the "who benefits" story is visible from
# position alone before you even look at the arcs.
node_order <- conf_totals %>% arrange(net_talent) %>% pull(conference)
n_nodes <- length(node_order)
angles <- tibble(
  conference = node_order,
  angle = seq(pi / 2, pi / 2 - 2 * pi * (n_nodes - 1) / n_nodes, length.out = n_nodes)  # start at top, go clockwise
) %>%
  mutate(x = cos(angle), y = sin(angle))

nodes <- conf_totals %>% inner_join(angles, by = "conference")

edges_plot <- pair_edges %>%
  inner_join(angles %>% select(conference, a_x = x, a_y = y), by = c("conf_a" = "conference")) %>%
  inner_join(angles %>% select(conference, b_x = x, b_y = y), by = c("conf_b" = "conference")) %>%
  # orient every row exporter -> importer so the arrow always points at the net importer
  mutate(
    x_start = ifelse(exporter == conf_a, a_x, b_x),
    y_start = ifelse(exporter == conf_a, a_y, b_y),
    x_end = ifelse(importer == conf_a, a_x, b_x),
    y_end = ifelse(importer == conf_a, a_y, b_y)
  )

# label placement: push text out radially beyond the node, and flip justification
# on the left half of the circle so labels read outward instead of into the plot
label_pos <- nodes %>%
  mutate(
    label_x = x * 1.22, label_y = y * 1.22,
    hjust = ifelse(x >= 0, 0, 1)
  )

p1 <- ggplot() +
  geom_curve(
    data = edges_plot,
    aes(x = x_start, y = y_start, xend = x_end, yend = y_end, linewidth = total_talent, alpha = total_talent),
    curvature = -0.3, color = "#003366",
    arrow = arrow(length = unit(0.18, "cm"), type = "closed")
  ) +
  geom_point(data = nodes, aes(x = x, y = y, size = activity, fill = net_talent), shape = 21, color = "white", stroke = 0.6) +
  geom_text(data = label_pos, aes(x = label_x, y = label_y, label = conference, hjust = hjust), size = 3.4, fontface = "bold", color = "#003366") +
  scale_linewidth(range = c(0.3, 3), guide = "none") +
  scale_alpha(range = c(0.25, 0.85), guide = "none") +
  scale_size_continuous(range = c(4, 16), guide = "none") +
  scale_fill_gradient2(low = "#c0392b", mid = "#f0e6d2", high = "#003366", midpoint = 0, name = "Net Talent\n(In − Out)") +
  coord_fixed(ratio = 1, xlim = c(-1.6, 1.6), ylim = c(-1.35, 1.35), clip = "off") +
  labs(
    title = "Where the Transfer Portal Talent Flows",
    subtitle = "Conference-to-Conference Transfer Movement, 2021–2025\n(nodes run clockwise from most-negative net talent at top to most-positive; node size = total activity; arrow points to net importer)",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_void(base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 22, color = "#003366", hjust = 0.5),
    plot.subtitle = element_text(size = 12, color = "#003366", hjust = 0.5),
    plot.caption = element_text(size = 8, color = "black"),
    plot.background = element_rect(fill = "#F7F7F7", color = NA),
    legend.position = "right"
  )

ggsave(file.path(outputs_dir, "transfer_portal_conference_flow.png"), plot = p1, width = 12, height = 9, dpi = 200)

### Chart 2: net talent import/export by conference ###

conf_totals_plot <- conf_totals %>%
  filter(conference != NON_FBS) %>%
  arrange(net_talent) %>%
  mutate(conference = factor(conference, levels = conference))

p2 <- ggplot(conf_totals_plot) +
  geom_col(aes(x = conference, y = talent_in), fill = "#003366", width = 0.6) +
  geom_col(aes(x = conference, y = -talent_out), fill = "#c0392b", width = 0.6) +
  geom_point(aes(x = conference, y = net_talent), color = "black", size = 2.5) +
  geom_hline(yintercept = 0, color = "gray40") +
  coord_flip() +
  scale_y_continuous(labels = comma_format()) +
  labs(
    title = "The Portal's Volume Story",
    subtitle = "Total Recruiting-Rating Talent Imported (navy) vs. Exported (red) by Conference, 2021–2025\n(black dot = net; this totals ALL transfers, so it's dominated by high-volume depth-player churn, not just star power)",
    x = NULL, y = "Aggregate Recruiting Rating (talent in − talent out)",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_portal()

ggsave(file.path(outputs_dir, "transfer_portal_net_talent_by_conference.png"), plot = p2, width = 11, height = 7, dpi = 200)

### Chart 3: average transfer quality -- trading up or down? ###
# The aggregate-sum view above is dominated by volume (P4 blue bloods shed a lot of
# roster depth). This instead asks, per transfer, whether a conference typically
# upgrades or downgrades: average recruiting rating of players it gains vs. players
# it loses. A cleaner "who's benefiting" signal than raw sums (volume-dominated) or
# blue-chip headcounts (noisy -- includes highly-rated recruits who busted and
# dropped down a level, which isn't really "benefiting").

quality_by_conf <- conf_totals %>%
  filter(conference != NON_FBS, n_in > 0, n_out > 0) %>%
  mutate(
    avg_rating_in = talent_in / n_in,
    avg_rating_out = talent_out / n_out,
    quality_delta = avg_rating_in - avg_rating_out
  ) %>%
  arrange(quality_delta) %>%
  mutate(conference = factor(conference, levels = conference))

write_csv(quality_by_conf, file.path(outputs_dir, "transfer_quality_by_conference.csv"))

p3 <- ggplot(quality_by_conf, aes(x = conference, y = quality_delta, fill = quality_delta > 0)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0, color = "gray40") +
  coord_flip() +
  scale_fill_manual(values = c(`TRUE` = "#003366", `FALSE` = "#c0392b"), guide = "none") +
  labs(
    title = "Trading Up or Trading Down?",
    subtitle = "Avg. Recruiting Rating of Incoming Transfers Minus Avg. Rating of Outgoing Transfers, by Conference, 2021–2025\n(positive = typically upgrades talent per transfer, negative = typically downgrades)",
    x = NULL, y = "Avg. Rating In − Avg. Rating Out",
    caption = "Data Source: @cfbfastR, Viz: @campbell_taylor1"
  ) +
  theme_portal()

ggsave(file.path(outputs_dir, "transfer_portal_quality_delta_by_conference.png"), plot = p3, width = 11, height = 7, dpi = 200)

cat("\nWrote outputs to", outputs_dir, "\n")
