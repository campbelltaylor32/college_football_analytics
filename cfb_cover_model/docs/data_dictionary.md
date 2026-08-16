# Data dictionary

Source: `../Data/CFB_Gambling_Predictors_Final_PBP.csv` (1,054 columns, 2,386 games,
2015-2025) joined to `../Data/CFB_Gambling_Results.csv` on `game_id` (see
`src/cfb_cover_model/data.py`). Full generation logic lives in
`../R Scripts/Full_CFB_Game_Outcome_Historical.R` and
`../R Scripts/Merge_Predictors_CFB_Historical.R` - this document describes what survives
into this project's modeling frame and why, not how the R layer computes it.

## Column categories in the source CSV

| category | count (approx.) | example | temporal transforms |
|---|---|---|---|
| game-stat box score / EPA-PBP | ~160 base stats x 2 sides x 3 transforms | `home_prev_week_Offense_EPA_per_Play` | `prev_week_*`, `*_avg_all`, `*_avg3` |
| talent | 4 stats x 2 sides | `home_Scaled_Talent` | none (season-level snapshot) |
| coaching | 2 stats x 2 sides | `home_Winning_Percentage` | none (already lagged 1 year upstream in R) |
| returning production | 12 stats x 2 sides | `home_total_ppa` | none (preseason snapshot) |
| context | 6 | `spread`, `week`, `neutral_site`, `conference_game` | none |

## What this project excludes from the candidate feature set, and why

All exclusion lists live in `config/data.yaml` as base-stat names, expanded to full column
names by `cleaning.expand_base_stats` (handles the home_/away_ x 3-transform Cartesian
product automatically - see `scripts/load_and_validate_dataset.py`'s
`outputs/data_inventory/column_inventory.json` for the exact expanded list from the last run).

- **`id_columns`** (`game_id`, `home_team`, `away_team`) - identifiers, not predictors.
- **`leakage_adjacent_columns`** (`home_favored`) - redundant with `spread`'s sign and used
  to help construct the label; kept in the saved parquet purely to build the
  `always_favorite` diagnostic baseline (`modeling/classifiers.py`), never fed to a model.
- **`known_bad_base_stats`** (`Offense_EPA_per_Run`, `Defense_EPA_per_Run`) - confirmed via
  direct inspection of `Merge_Predictors_CFB_Historical.R`'s `summarise()` blocks and the
  actual CSV that these columns hold **pass-play EPA**, not run-play EPA: the block computes
  `Offense_EPA_per_Run` first, then a second assignment meant to produce
  `Offense_EPA_per_Pass` overwrites it under the same output name due to a duplicate-key
  bug, and `Offense_EPA_per_Pass` never makes it into the final CSV at all. Excluded by
  name rather than silently trained on a mislabeled column. Not fixed at the R layer -
  that pipeline is shared with the production notebook and other sibling projects; fixing
  it there is out of scope here (see `assumptions_and_limitations.md`).
- **`deterministic_redundant_base_stats`** - base stats that are exact linear/ratio
  functions of other raw columns retained in the same table (e.g. `point_differential =
  points - points_allowed`, `yards_per_play = total_yards / total_plays`). Dropped once,
  not twice, so deterministic multicollinearity doesn't enter feature selection before it
  even runs. This list is not claimed to be exhaustive - stage 3 (correlation-cluster
  pruning, see `feature_selection_methodology.md`) is a fold-local safety net for anything
  missed here.
- **`season`** - excluded from the candidate feature set (though retained in the saved
  frame as a splitting key). Using it as a raw numeric predictor would let a model learn a
  "which year" trend rather than an on-field signal, which cannot extrapolate to future
  seasons the model has never seen a season index for.

## The label and continuous target (`src/cfb_cover_model/targets.py`)

- `home_covered` (0/1) - recomputed from `home_minus_away` (final score margin) and
  `signed_spread` (joined from `CFB_Gambling_Results.csv`, not the absolute-value `spread`
  in the predictors file), **not** trusted from the R layer's own column, because that
  column mislabels exact pushes as "did not cover" (see `data_leakage_rules.md` and
  `assumptions_and_limitations.md`).
- `is_push` (bool) - `True` when `home_minus_away == -signed_spread`. Push rows are
  dropped before training/evaluation everywhere in this project.
- `cover_margin` (float) - `home_minus_away + signed_spread`: how many points the home
  team beat (positive) or missed (negative) the spread by. Zero exactly on a push. This is
  Track B's regression target.

## Engineered features (`src/cfb_cover_model/engineered_features.py`)

Applied after the base candidate set is assembled, before feature selection ever runs -
see `docs/feature_importance_findings.md` for the analysis that motivated these and
`docs/project_story.md`'s "Update" section for results:

- **`{side}matchup_adj_{stat}_{transform}`** (additive - raw columns are kept too): for
  down-conversion and EPA/success-rate base stats that have both an offense-side and a
  defense-"allowed" counterpart, `home_offense_rate - away_defense_allowed_rate` (and the
  mirror for away) - a this-game opponent-adjusted version of the raw team-level rate.
  `third_down_rate`/`fourth_down_rate` are computed from conversion/attempts on the fly
  (0-attempt cases fall back to 0.0, not NaN - see the module for why).
- **`{side}special_teams_net_score_{transform}`** (replaces the 14-column raw
  special-teams family per side/transform): a point-value-weighted composite
  (`kicking_points` + return yards/17 + return TDs*6, net of the `_allowed` counterparts) -
  a documented approximation, not a fitted parameter, so it stays deterministic and
  leakage-free.
- **Returning-production trim**: `passing_usage`, `percent_passing_ppa`, `percent_ppa`,
  `percent_receiving_ppa`, `total_passing_ppa`, `total_ppa`, `total_receiving_ppa`, `usage`
  are dropped per side, keeping only `rushing_usage`, `receiving_usage`,
  `percent_rushing_ppa`, `total_rushing_ppa` - the 4 that showed up as fold-stable in the
  original 12-column family.

## Rows dropped

Any row still missing a value in a candidate feature column after all upstream lags is
dropped (`data.yaml`'s `drop_rows_with_any_na: true`), matching the R layer's own
`na.omit()` semantics rather than re-deriving new imputation behavior. As of the last run,
zero rows were dropped at this step (all missingness was already resolved upstream by the
R layer's own `filter(week >= 3)` + `na.omit()`); see
`outputs/data_inventory/column_inventory.json`'s `n_rows_dropped_for_na` for the current
number.
