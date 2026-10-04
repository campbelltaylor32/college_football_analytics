# 2026 Quarterback EPA Tracker

## Question
Which FBS quarterbacks are building value week by week, and who is trending up or down? This project tracks cumulative EPA (plus EPA/play and success rate) for every qualifying 2026 starter. It shows the results as an interactive time series with player headshots.

## Data sources
- **Play-by-play:** `cfbfastR::load_cfb_pbp()` release file for 2026 (EPA, success, ESPN player IDs).
- **Quarterbacks and headshots:** `cfbfastR::load_cfb_rosters()`. The roster `athlete_id` matches the ESPN player IDs in the play-by-play, and `headshot_url` points to ESPN's CDN.
- **Teams:** `cfbd_team_info(only_fbs = TRUE)` for conference, team color and logo.

## Method
Run from the repo root (a `.env` with `CFBD_API_KEY` is required). Re-run both scripts weekly. The season defaults to 2026; set `CFB_SEASON` to change it. The Sunday `weekly-results.yml` workflow runs both scripts and publishes the site's CFB QB Tracker.
1. `Rscript cfb_qb_epa_tracker/pull_qb_data.R` caches play-by-play, the QB roster and FBS teams to `data/` (gitignored).
2. `Rscript cfb_qb_epa_tracker/qb_epa_analysis.R` writes to `outputs/`:
   - `qb_epa_tracker.html`: the interactive page. It is built from `qb_epa_tracker_template.html` with the data and images inlined.
     - Filters: All FBS, Power 4, Group of 5, or any one conference.
     - Metrics: cumulative EPA, EPA/play or success rate.
     - Highlights: the top and bottom 10 within the current filter (re-ranked as you filter).
     - Pinning: click any line, headshot or row, or search by name.
     - Also includes a trending panel and a full rankings table.
   - `qb_cumulative_epa_2026.png`: static ggplot with `ggimage` headshots on the top and bottom 10.
   - `qb_epa_2026.csv` / `.json`: one row per QB-week, with week 0 as the season start.

**Attribution.** A QB's plays are:
- His dropbacks: completions, incompletions, interceptions and sacks, matched by ESPN player ID. About 2% of dropbacks have no ID and are matched through the passer name on the same team.
- His rushes. cfbfastR codes scrambles as rushes, so QB rushes include both scrambles and designed runs.

## Caveats
- **Garbage time removed:** plays with a margin over 38 in Q2, over 28 in Q3 or over 22 in Q4. These are the same thresholds as `cfb_ryan_day_offense`.
- **Qualifier:** at least 15 dropbacks per game the QB's team has played. A starter lost to injury drops out once his total falls below that pace.
- **Cumulative vs rate:** cumulative EPA rewards volume, so a QB with an extra game or more plays per game ranks higher. Switch to EPA/play for pure efficiency. Rates are noisy in the first few weeks.
- **No lookahead:** a QB's value at week N only uses plays from weeks 1 through N. Byes and missed games carry the previous value forward.
- **Headshots** are cropped to 96px squares and embedded in the page, because the published page can't load images from other sites. The team logo is used when ESPN has no headshot.
- **Trending** compares EPA/play over the last two weeks with the weeks before. It requires at least 25 plays in each span.

## Snapshot (through 2026 week 5)
- 124 qualifying starters: 67 Power 4 and 57 Group of 5.
- **Cumulative EPA leaders:**
  - Jayden Maiava (USC) +83.2
  - Tayven Jackson (North Texas) +81.3
  - Julian Sayin (Ohio State) +79.7
  - Keelon Russell (Alabama) +76.4
  - Gio Lopez (Wake Forest) +70.3
- **Bottom:** Jayden Denegal (San Diego State) −32.3, Grady Brosterhous (Utah State) −23.3, Caden Pinnick (Washington State) −20.8.
- **Best per-play efficiency among the leaders:** David McComb (Miami (OH)) at +0.61 EPA/play on 113 plays.
