# Assumptions and Limitations

## 1. The 2025 rusher-name data gap (confirmed persistent at the source, weeks 9+)

`plays.rusher_player_name` NULL rate on rush plays (`play_type IN ('Rush','Rushing
Touchdown')`), by season, verified live against `cfb_football` on 2026-08-09:

| Season | NULL rate | Note |
|---|---|---|
| 2013 | 100.00% | Unusable -- excluded via `full_feature_start_season=2014` |
| 2014-2019 | 0.00-0.03% | Clean |
| 2020 | 0.76% | COVID-shortened, excluded for the standard reason |
| 2021-2024 | 3.38-5.95% | Clean (modest, dominated by unparsed kneel-downs) |
| 2025, weeks 1-8 | 1.57-3.74% | Clean |
| 2025, weeks 9-14 | 97.20-99.49% | **Confirmed persistent gap at the source (see below)** |

`games.completed` shows the 2025 season as fully "completed," so this would NOT be caught by
row-count or season-range checks alone -- only a field-level completeness check
(`data_validation.check_rusher_name_completeness`) catches it.

**Root-caused, not just observed.** Re-investigated on 2026-08-09 (months after the 2025
season ended) by pulling week 5 and week 9 fresh, live, directly via
`cfbfastR::cfbd_pbp_data()` -- the exact function `SQL Scripts/ingest_to_mysql.R` uses:
- Week 5 (Stanford), pulled fresh today: 35/36 rush plays resolve `rusher_player_name`. Matches
  the DB.
- Week 9 (Stanford), pulled fresh today: **0/69** rush plays resolve `rusher_player_name`.
  Matches the DB exactly.

**Conclusion: this is not fixable by re-running the ingestion.** It's a persistent gap in
CollegeFootballData's (sourced from ESPN) structured player-attribution field for 2025 games
from week 9 onward, present in the API today, not a transient ingestion-time failure. The
player's name IS still recoverable as free text in `play_text` (e.g. `"#33 C.Tabb rush middle
for 3 yards..."` -- jersey number + last name), which could be parsed and joined to
`team_rosters` on `(team, season, jersey)` as a future enhancement, but that parser does not
exist yet -- `player_game_rushing.py` still requires the structured field.

**Consequence for season scope**: `config/modeling.yaml` sets `final_holdout_season: 2025`
with `final_holdout_max_week: 8` -- 2025 IS used (as the final holdout, the most realistic
recent-season test), but only its clean weeks. `scripts/build_modeling_dataset.py` drops any
row for the holdout season past that week as a post-build filter, specifically because those
rows' targets would otherwise be silently zero-filled by `targets.py`'s LEFT JOIN (no realized
carries data exists for them) -- indistinguishable from a genuine zero-carry game, which would
quietly corrupt holdout evaluation with fake data. The identical completeness check is wired
into `scripts/generate_week_predictions.py` as a hard abort-on-failure gate for whichever
week is requested, since this failure mode (silent field-level collapse mid-season) is exactly
what a live weekly-inference script is most exposed to -- 2024 slides into the training pool
in 2025's place.

## 2. Name-matching resolution: verified rates and residual risk

`plays.rusher_player_name` is free text with no stable ID; resolved to `team_rosters.athlete_id`
via exact + normalized (accent/punctuation/suffix-stripped) matching (`player_resolution.py`).
Verified live, carries-weighted, across 2022-2024:

- **Matched against the full roster (any position)**: ~93-95% exact + ~2% normalized = **~95-97%
  resolved**, ~2-3% genuinely unmatched, ~2% ambiguous (two same-named players on one roster).
  This is the real resolution-failure floor -- `data_validation.check_player_resolution_match_rate`
  WARNs if the any-position-unmatched rate exceeds 10%.
- **Matched against RB-position roster rows only**: ~68-71% of ALL rush-play carries resolve to
  an RB. This lower number is expected and NOT itself a resolution failure -- decomposed by
  position (season-level, carries-weighted): RB ~68-71%, QB ~18-19% (scrambles/sneaks, correctly
  excluded), other positions ~5% (jet sweeps, trick plays), genuinely unresolved-to-any-position
  ~6-9%.
- **Known residual miss not fixed by the normalized pass**: box scores that abbreviate a
  player's first name to an initial (e.g. `"A. Jeanty"` for Ashton Jeanty, Boise State 2023) --
  verified live at ~1.1% of distinct rusher names league-wide in 2023. A last-name +
  first-initial fallback pass would likely recover most of these but is not implemented in v1;
  flagged as a natural next improvement, not a blocker (the affected carries are a small
  minority and the eligible RB population is dominated by starters whose names are formatted
  consistently).
- Two same-named players on one team-season roster (`ambiguous` classification) are excluded
  rather than guessed -- a real, if rare, source of missing data for a legitimately eligible
  player.

## 3. Eligibility threshold is an explicit, tunable business rule, not a validated optimum

`features.yaml`'s `min_trailing3_avg_carries=8` / `min_season_to_date_carries=15` /
`min_games_played_for_avg3=3` are starting points grounded in the live carries-per-game
distribution, not a tuned-for-accuracy choice. `scripts/run_eda.py` writes
`eligibility_threshold_sensitivity.csv` (row count / unique-player count across candidate
thresholds 4-15) specifically so this can be revisited before being treated as final.

**Cold-start limitation, structural and unavoidable**: a true debut game (true freshman's
first career carry, or a transfer's first game at a new school) has no prior recorded game to
gate on or build rolling features from, so it is `eligible=False` by construction -- this
project cannot and does not attempt to predict a breakout debut game. This is a real,
documented gap, not an oversight.

## 4. Position scope: RB only, strictly `team_rosters.position == 'RB'`

Per the approved plan, QB rushing is explicitly deferred to a later phase. Players tagged
`ATH`/`FB`/similar with real complementary rushing usage are excluded by this same strict
definition -- a deliberate scope decision, not a data gap.

## 5. Betting-line context is optional, off by default

`data.yaml: include_betting_context` (default `false`). Spread/over-under correlate with game
script and therefore rushing volume, but add missingness-handling complexity (a future game
may not have a posted line yet at inference time) for a player-prop model whose other
features already carry substantial signal. Left as an explicit toggle rather than a required
feature.

## 6. No injury-report data exists in this database

An eligible RB who leaves a game early due to an in-game injury, or is a surprise healthy
scratch, produces a genuine `rushing_yards=0` row (see `targets.py`) that no pre-game-known
feature in this schema can anticipate. `median_ae` is reported alongside `mae` specifically
because it is more robust to this exact noise source (see `docs/modeling_methodology.md`).

## 7. `first_down_rate` and `tempo_plays_per_minute` are documented approximations

`plays` has no explicit first-down-achieved flag (`first_down_rate` uses `yards_gained >=
distance` as a proxy) and no per-drive/per-play clock aggregate at team-game grain
(`tempo_plays_per_minute` uses plays-per-minute-of-own-possession instead of true
seconds-per-play). See `docs/data_dictionary.md` for the exact definitions.
