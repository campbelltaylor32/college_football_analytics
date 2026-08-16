# Talent Distribution Analysis (2015–2025)

## Question

How has recruiting talent been distributed across FBS college football over the last decade, particularly since the transfer portal (opened ~2018) and NIL (July 2021) reshaped how rosters get built?

1. Has the number/share of teams with a **blue-chip roster** (blue-chip ratio — share of roster that are 4★/5★ recruits — above 50%) grown?
2. Is talent **more concentrated** among a handful of elite programs, or **more dispersed** across the league, than it used to be?

## Data sources

- `Data/CFB_Team_Talent_Data.csv` — `talent`/`Scaled_Talent` (CFBD's own team-talent composite, from `cfbd_team_talent()`), used for the concentration/Gini metric. 2015–2025.
- `outputs/blue_chip_ratio_corrected.csv` — a **corrected** blue-chip ratio, built fresh from the CFBD API by `build_corrected_blue_chip_ratio.R` (see below), used for the blue-chip-team and distribution-shape metrics.

Both are filtered to FBS-only teams per season via `cfbfastR::cfbd_team_info(year, only_fbs = TRUE)` — the raw talent file mixes in FCS programs for 2015–2023 (232 "teams" in 2015 vs. 134 in 2024, because `cfbd_team_talent()`'s upstream behavior changed), so an unfiltered comparison across years isn't apples-to-apples.

## Why blue_chip_ratio needed correcting

The shared pipeline (`R Scripts/Full_CFB_Game_Outcome_Historical.R:185–218`) computes `blue_chip_ratio` as `sum(stars ≥ 4) / n_distinct(athlete_id)`, joining the full roster pull to recruiting data on `athlete_id` alone. Reproducing this by hand for Alabama's 2021 team (a well-documented case — Bud Elliott reported ~86% BCR that season, an all-time high) gave **54.4%**, roughly two-thirds of the real figure. Two compounding issues:

1. **Denominator dilution.** `cfbd_team_roster()` returns every player on the roster, including walk-ons with no recruiting profile — 125 players for Alabama 2021, well above the ~85-man scholarship limit the real-world "Blue Chip Ratio" stat is computed against. Dividing by the full roster instead of just recruited players mechanically understates the ratio.
2. **Missed matches.** `athlete_id` is `NULL` for a meaningful share of CFBD's recruiting records (54% of unrated recruits, ~9% even among 4★/5★ recruits) — including some blue-chip transfers whose original high-school recruiting record never got an `athlete_id` populated. Joining only on `athlete_id` silently drops these players from the numerator too.

`build_corrected_blue_chip_ratio.R` fixes both: it matches roster players to recruiting records via **either** `athlete_id` **or** the roster's own `recruit_ids` → recruiting `id` link (taking the union, since each method catches players the other misses), and divides by the number of *matched* players (i.e. identifiable recruits, rated or not) rather than the full walk-on-inclusive roster. This reproduces Alabama 2021 at **84.0%** (68 blue-chip / 81 matched) — much closer to the ~86% publicly reported. Team-seasons with fewer than 20 matched players (mostly service academies and other low-recruited programs with genuinely low ratios, not statistical noise — verified by inspection) are dropped as low-confidence.

This correction is **only applied within this project's outputs** — it does not modify `Data/CFB_Team_Talent_Data.csv` or the shared pipeline, which still feeds the spread-prediction model with the original (diluted) `blue_chip_ratio`. If that model's feature should also be corrected, that's a separate, higher-stakes change (it touches training data and would need a full historical re-pull) and should be scoped on its own.

## Method

1. `build_corrected_blue_chip_ratio.R` — pulls national rosters (2015–2025) and recruiting classes (2009–2025) live from CFBD, computes the corrected `blue_chip_ratio` per team-season, writes `outputs/blue_chip_ratio_corrected.csv`. **Run this first** — it takes a few minutes (network-bound).
2. `talent_distribution_analysis.R` — loads both data sources, filters to FBS-only teams per season, computes yearly aggregates, and renders three ggplot2 charts to `outputs/`, sharing one theme (`theme_talent()`) styled to match `cfb_pythagorean_model/plot_deviation_logos.R`.

## Results

- **`blue_chip_ratio_median_trend.png` — the clearest signal in this analysis.** The median FBS team's blue-chip ratio has more than doubled: 4.0% in 2015 → 10.5% in 2024–2025, with a visible inflection right around the transfer-portal/NIL markers (flat 2015–2018, rising steadily 2018 onward). A linear trend on the yearly mean is statistically significant (R² = 0.38, p = 0.044). This says something the >50% threshold chart below can't: it's not just that a few more teams cross an arbitrary bar — the *entire league's* baseline of blue-chip talent has shifted up. The mean moves the same direction but far more mutedly (dashed line), because it's pulled by the still-large mass of teams near 0%; the median is the more honest read on "the typical team."
- **`blue_chip_teams_by_year.png` / `.csv`** — 14–18 FBS teams per season (11–15% of FBS) carry a blue-chip ratio above 50%. This count alone is cyclical rather than trending (highest in 2015 at 18, lowest in 2023–2024 at 14, back to 17 in 2025) — the extreme tail is noisier than the median, and by itself would understate how broadly talent access has actually shifted. Read alongside the median trend above, not as a standalone conclusion.
- **`talent_concentration_gini.png` / `.csv`** — Gini coefficient of the talent composite bounces in a ~0.18–0.22 band from 2015–2022, then drops sharply to 0.148 (2024) and 0.130 (2025) — its lowest points in the series. Talent has become **more dispersed, not more concentrated**, in the most recent two seasons, consistent with the portal/NIL era making it easier for a wider set of programs to assemble competitive rosters rather than talent pooling at a fixed set of blue bloods. (This metric is unaffected by the blue_chip_ratio correction above — `talent` comes directly from CFBD's own composite score.)
- **`blue_chip_ratio_ridgeline.png`** — the corrected distributions show a clear bimodal shape every season: a tall peak of teams near 0–10% blue-chip share, and a persistent secondary bump around 40–70% (the blue-blood cluster), with a long tail out past 80–90% in most years. The red median line visibly walks rightward from 2018 onward, the same shift the trend chart quantifies. That two-cluster shape — "haves" and "have-nots," with not much of a broad middle class — is stable across all 11 seasons; the last two years show the secondary bump has narrowed rather than disappeared.

**Putting it together:** the median/Gini/ridgeline all agree — talent (and blue-chip share specifically) is spreading further into the middle of the league, not concentrating at the very top. The >50% threshold count is the outlier metric here because it only watches the extreme tail, which is small enough to be dominated by which 2-3 specific blue bloods have a good or bad recruiting cycle in a given year.

**Caveat:** 2024–2025 are only 2 seasons of "post-portal-expansion" data (the portal's modern unlimited/one-time-transfer rules matured around 2023–24), so the recent acceleration should be read as an early signal, not a settled conclusion.
