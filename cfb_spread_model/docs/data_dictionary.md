# Data dictionary

This project reads `../Data/CFB_Gambling_Predictors_Final_PBP.csv` directly (see
`config/data.yaml`) and does **not** query the local MySQL database (`cfb_football`) described
in `../SQL Scripts/schema.sql`, populated by `../SQL Scripts/ingest_to_mysql.R`. That database
has no engineered features yet — it's a raw, near-1:1 mirror of the CFBD API endpoints, and its
own README says explicitly that nothing in `R Scripts/` or this project reads from it. This
document exists to trace the CSV's engineered column names back to the raw tables `schema.sql`
documents, for anyone trying to understand where a given predictor originated, and to answer
"which SQL tables does this project relate to" without pretending it queries them.

## Verified structure

`Data/CFB_Gambling_Predictors_Final_PBP.csv`: **2,386 rows × 1,054 columns** (verified this
build). 10 identifier/label columns:

```
game_id, home_team, away_team, season, week, neutral_site, conference_game, spread, home_favored, home_covered
```

The remaining 1,044 columns are `home_*`/`away_*` symmetric pairs — one team-week feature table
(computed once) merged onto the game-outcome table twice, once per side, via `game_id` (see
`../R Scripts/Merge_Predictors_CFB_Historical.R:72-105`). Per side (522 columns):

- **168 `prev_week_*`** — last single game's raw stat. **Excluded from this project's feature
  matrix** by `config/data.yaml`'s `excluded_column_patterns`, as of the run documented in
  `docs/project_story.md` — the noisiest of the three temporal transforms, and the largest
  single category (45%) in this project's first production model. Still present in the raw CSV
  and in `get_feature_columns()`'s input; the exclusion happens at that one choke point.
- **168 `*_avg_all`** — season-to-date cumulative mean of the same stat.
- **168 `*_avg3`** — trailing 3-game rolling mean of the same stat.
- **18 non-temporal**: 4 talent (`talent`, `Scaled_Talent`, `blue_chip_ratio`,
  `avg_player_rating`), 2 coaching (`Total_Games_Coached`, `Winning_Percentage`), 12 returning
  production (`total_ppa`, `total_passing_ppa`, `total_receiving_ppa`, `total_rushing_ppa`,
  `percent_ppa`, `percent_passing_ppa`, `percent_receiving_ppa`, `percent_rushing_ppa`, `usage`,
  `passing_usage`, `receiving_usage`, `rushing_usage`).

Within the 168 base metric names: 70 are EPA/success-rate metrics with paired
`Offense_`/`Defense_` prefixes (35 concept pairs), 41 have an explicit `_allowed` defensive
mirror, 16 are unpaired algebraic-ratio columns (`point_differential`, `turnover_margin`,
`yards_per_pass`, `penalty_yard_margin`, etc.) with no separate raw-component pair among the
168 themselves (their raw components live elsewhere in the set). This is the structure
`feature_selection/correlation_pruning.py`'s Stage 1 targets.

## Naming pattern → source table map

| CSV pattern | Example | Traces to (`../SQL Scripts/schema.sql` table) | R script that builds it |
|---|---|---|---|
| `{home,away}_prev_week_*`, `*_avg_all`, `*_avg3` (box-score metrics) | `home_prev_week_total_yards` | `game_team_stats` (mirrors `cfbd_game_team_stats(year, week)`) | `../R Scripts/Full_CFB_Game_Outcome_Historical.R` (stats pull), `../R Scripts/Merge_Predictors_CFB_Historical.R:37-57` (lag/roll) |
| `{home,away}_{Offense,Defense}_*` (EPA/success-rate metrics) | `home_prev_week_Offense_EPA_per_Play` | `plays` (mirrors `cfbd_pbp_data(..., epa_wpa=TRUE)`) | `../R Scripts/Full_CFB_Game_Outcome_Historical.R` (EPA aggregation) |
| `{home,away}_talent`, `Scaled_Talent`, `blue_chip_ratio`, `avg_player_rating` | `home_talent` | `team_talent`; `blue_chip_ratio`/`avg_player_rating` also draw on `recruiting_players` + `team_rosters` | `../R Scripts/Merge_Predictors_CFB_Historical.R:7,20` |
| `{home,away}_Total_Games_Coached`, `Winning_Percentage` | `home_Winning_Percentage` | `coaches` | `../R Scripts/Merge_Predictors_CFB_Historical.R:8,11-18` |
| `{home,away}_{usage,*_ppa,percent_*}` | `home_total_ppa` | `returning_production` (mirrors `cfbd_player_returning(year)`) | `../R Scripts/Merge_Predictors_CFB_Historical.R:27-30` |
| `spread`, `home_favored` | `spread` | `betting_lines` | `../R Scripts/Full_CFB_Game_Outcome_Historical.R:99-110` |
| `game_id`, `season`, `week`, `neutral_site`, `conference_game` | — | `games` | `../R Scripts/Full_CFB_Game_Outcome_Historical.R` (game pull) |
| `home_covered` (label) | — | derived, not a raw table | `../R Scripts/Full_CFB_Game_Outcome_Historical.R:99-127` |

The `plays` table in `schema.sql` is partitioned by season (`p2015`...`p2025`, `pmax`) and is by
far the largest raw table — this project never touches it directly; the EPA/success-rate
aggregation into team-week summaries already happened upstream in R.

## Engineered columns (this project only, not in the source CSV)

**Pythagorean win%** (always present, not gated by any toggle) —
`scripts/load_and_validate_dataset.py` calls
`feature_engineering.build_pythagorean_features()` right after validating the raw CSV and before
caching to `modeling_dataset.parquet`, adding a 169th base metric (`pythagorean_win_pct`) on top
of the CSV's 168:

| Column | Meaning |
|---|---|
| `{home,away}_pythagorean_win_pct_avg_all` | `PF**2 / (PF**2 + PA**2)` using `{side}_points_avg_all` / `{side}_points_allowed_avg_all` (season-to-date PF/PA) |
| `{home,away}_pythagorean_win_pct_avg3` | Same formula using the `avg3` (trailing 3-game) PF/PA columns |

Bill James-style expected win% from each side's already-lagged rolling scoring margin (see
`cfb_pythagorean_model/` at the repo root for the retrospective analysis this formula and
exponent choice — classic exponent 2 — came from). Built entirely from existing `*_points_avg_all`
/ `*_points_allowed_avg_all` / `*_avg3` columns already in the CSV, no new raw data pull, no new
leakage risk. `modeling_dataset.parquet` therefore has **1,058 columns**, 4 more than the
1,054-column raw CSV `config/data.yaml`'s `expected_column_count` describes (that check runs on
the CSV before these columns are added).

Because these two new columns follow the exact `{side}_{base}_avg_all`/`avg3` naming convention,
they automatically flow into the `diff_`/`trend_` machinery below with no extra code —
`diff_avg_all_pythagorean_win_pct`, `diff_avg3_pythagorean_win_pct`,
`trend_home_pythagorean_win_pct`, `trend_away_pythagorean_win_pct` all exist once
`feature_representation: differential` is selected.

When `config/data.yaml`'s `feature_representation: differential` (not the default —
see `docs/feature_selection_methodology.md` and `docs/project_story.md`'s "Run 3"),
`src/cfb_spread_model/feature_engineering.py` derives two new column families from the raw
`avg_all`/`avg3`/non-temporal columns above (never `prev_week_*`), replacing the raw pairs they're
built from:

| Pattern | Example | Meaning |
|---|---|---|
| `diff_<transform>_<base>` | `diff_avg_all_total_yards` | `home_<base>_<transform> - away_<base>_<transform>` — the matchup differential |
| `diff_<base>` (non-temporal only) | `diff_talent` | `home_<base> - away_<base>` |
| `trend_<side>_<base>` | `trend_home_total_yards` | `<side>_<base>_avg3 - <side>_<base>_avg_all` — recent form vs. season baseline |

356 `diff_*` + 338 `trend_*` = 694 columns (169 base metrics now that `pythagorean_win_pct` is
included, up from the 354/336 figures verified against the 168-base-metric raw CSV alone — see
`tests/test_data_loading.py::test_differential_representation_on_real_data`, which asserts
354/336 against `load_raw_csv`'s output directly, deliberately upstream of where the Pythagorean
columns are added; 356/338 verified against `modeling_dataset.parquet` after the Pythagorean
step). Not a large dimensionality cut by itself — the hypothesis being tested is more directly
relevant signal per column, not fewer columns.

## Known scope decisions inherited from upstream

- The source CSV only covers **weeks 4-12** of each season (see `config/data.yaml`
  `week_range`) — the R pull caps at week 12 (`weeks <- seq(1,12)` in
  `Full_CFB_Game_Outcome_Historical.R`), and the post-lag `na.omit()` in
  `Merge_Predictors_CFB_Historical.R:70` pushes the practical minimum to week 4 (rows without a
  full 3-game trailing window get dropped). Bowls and conference championship games are never
  pulled.
- `spread` in this CSV is stored as `abs(spread)` (`Merge_Predictors_CFB_Historical.R:91`) —
  `home_favored` is the only column carrying the betting line's direction, and is therefore not
  redundant with `spread`.
