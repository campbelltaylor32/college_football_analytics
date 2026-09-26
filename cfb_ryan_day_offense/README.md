# Ohio State Offense Under Ryan Day (2019–2026)

## Question
Has Ohio State's offense changed under Ryan Day? The early teams (2019–2021) felt fast, explosive, pass-heavy and high-scoring. Recent teams feel slower, with fewer snaps, fewer big plays and less success on early downs. This project tracks season-level trends in scoring, volume and pace, run/pass mix, efficiency, explosiveness and situational metrics. It compares Ohio State with the FBS average and shows its FBS percentile.

## Data sources
- **Play-by-play:** `cfbfastR::load_cfb_pbp()` (sportsdataverse release files) for 2019–2026. This includes all FBS offenses, which gives the FBS benchmark and percentiles. Any Ohio State game missing from the release is back-filled from the CFBD API with `cfbd_pbp_data()`.
- **Drives and game results (Ohio State):** `cfbd_drives()` and `cfbd_game_info()` with `season_type = "both"`.
- Conference title, bowl and CFP games are all included. The shared `Data/Game_Stats_Averages_CFB_PBP_Added.csv` is not used because it only covers weeks 1–12. That would drop the postseason and leave just 4 games for 2020.

## Caveats
- **Garbage time.** Ohio State plays many blowouts. Efficiency metrics (yards/play, EPA, success, explosive rates, pass rates, 3rd downs, sacks, turnovers) exclude plays where the margin is over 38 in Q2, over 28 in Q3 or over 22 in Q4. Volume metrics (plays, drives and points per game) use every snap.
- **Pace** is seconds of game clock per offensive play, computed from CFBD drive data (Ohio State only). It reflects clock stoppages (incompletions, first-down clock stops pre-2023) as well as tempo. The **2023 NCAA clock rule** (clock keeps running after first downs) lowers plays per game across FBS from 2023 on (the FBS average fell about 3 plays per game). Use the FBS average line on the plays chart to separate the rule change from Ohio State's own choices.
- **Clock data.** A few CFBD drives have corrupt clock data (for example, a 54-minute drive in 2024). They are excluded from pace and time-of-possession metrics.
- **Small samples.** 2020 is an 8-game season. 2026 is in progress (3 games at time of writing) and is drawn with a hollow point.
- **Roster turnover.** Each season had a different QB and roster (Fields → Stroud → McCord → Howard → Sayin), so year-to-year swings mix scheme with personnel.
- **Definitions:**
  - Explosive plays are rushes of 10+ yards and dropbacks of 20+ yards. The repo's model features use 20+ for both; that version is in the CSV as `expl_rate_20`.
  - Success uses cfbfastR's `success` flag (50% / 70% / 100% of yards to go on 1st / 2nd / 3rd–4th down).
  - Dropbacks include sacks.

## Method
Run from the repo root (`.env` with `CFBD_API_KEY` required):
1. `Rscript cfb_ryan_day_offense/pull_osu_offense_data.R` pulls and caches play-by-play, drives, game results and the FBS team list to `data/` (gitignored). Re-run weekly to add new 2026 games.
2. `Rscript cfb_ryan_day_offense/ryan_day_offense_analysis.R` computes season metrics for every FBS offense and writes:
   - `outputs/ryan_day_offense_trends.pdf`: a 31-page chart deck: summary table, one page per metric, a game-by-game snaps chart and an FBS-percentile heatmap.
   - `outputs/osu_offense_season_summary.csv`: Ohio State season metrics.

## Results (through 2026 week 3)
- **Snapshot table** / **Relative to FBS heatmap:** Ohio State has ranked in at least the 85th FBS percentile in yards/play, EPA/play and success rate every full season. What has slipped is volume, and explosiveness in 2023 and 2025.
- **Scoring:** points per game fell from 46.9 (2019) and 45.7 (2021) to 30.5 (2023), 35.7 (2024) and 33.4 (2025).
  - Per-drive efficiency recovered in 2024–25 (3.2–3.4 pts/drive, vs 3.6–3.7 at the peak).
  - The drop in points per game after 2023 comes more from fewer possessions than from worse drives.
- **Snaps:** offensive plays per game fell steadily, from 76.0 (2019) to about 62–63 (2023–25). Ohio State went from the 95th FBS percentile in snaps to the 16th in 2024–25. Over the same span the FBS average only dipped from 69 to 66 plays (the 2023 clock rule), so most of Ohio State's 14-play drop is its own doing.
  - The game-by-game view shows the drop in competitive games too, not just blowouts.
- **Pace:** the offense slowed down.
  - Seconds per play rose from 24–27 (2019–23) to 32.4 (2024) and 30.6 (2025).
  - Drives per game fell from 12.6 to 9.0 in 2025.
  - Plays per drive rose to 7.0 and time of possession to 32.9 minutes in 2025: a long-drive, ball-control profile.
  - The early 2026 sample is back to 2019-like pace (26.3 s/play, 12 drives/game).
- **Pass rate:** not what the "air-it-out 2019" memory suggests.
  - 2019–20 were the most run-heavy Day offenses (44–45% dropbacks, below the FBS average) with Fields and Dobbins/Sermon.
  - The pass-heavy peak was 2021 with Stroud (59.6%).
  - From 2022 to 2025 it settled around 51%.
- **Yards per play:** peaked at 8.35 in 2021. The dips were 2023 (6.77) and 2025 (6.87).
  - Yards per rush fell steadily: 7.3 (2020) → 5.1 (2023) → 4.8 (2025).
- **Explosiveness:** explosive plays per game fell from 12.2 (2019–20) to 7.9–9.1 (2023–25).
  - The explosive pass rate hit a Day-era low of 10.6% in 2025 (58th percentile, down from the high-80s/90s).
  - The explosive rush rate cratered in 2023 (12.4%, 25th percentile).
- **EPA:** EPA per rush dropped from +0.25 (2019–20) to about +0.02 to +0.03 (2023, 2025). The running game is where efficiency eroded most. EPA per dropback stayed elite (+0.32 to +0.52).
- **Early downs:** early-down success was similar in 2024–25 (50–51%) to 2019 (49.5%). Early-down EPA, however, fell to 0.09 in 2025, the lowest of the tenure.
  - Early-down plays are still "successful" but produce fewer chunk gains.
- **Situational:** 3rd-down conversion was 61% in 2019 and has been 46–53% since. Red-zone TD rate is volatile (64–80%). Sack rate fell sharply after 2020 (8.6% → 2–4%).
