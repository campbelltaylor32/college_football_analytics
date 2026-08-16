# Data leakage rules

Feature engineering — and its lookahead-bias prevention — already happened upstream in R (see
`../R Scripts/Merge_Predictors_CFB_Historical.R` and `../R Scripts/Full_CFB_Game_Outcome_Historical.R`).
This project reads the already-lagged CSV and does not re-derive that logic; the tests in
`tests/test_leakage.py` are a **regression safety net**, verifying the guarantees below still
hold against the live CSV, not a from-scratch leakage audit.

## Per-category source-season table

| Category | Source | Timing rule | Verified leak-safe? |
|---|---|---|---|
| `prev_week_*`, `*_avg3`, `*_avg_all` (168 base metrics x 2 sides) | `../R Scripts/Merge_Predictors_CFB_Historical.R:44-57` — `dplyr::lag(.x, 1)` applied per `(team, year)`, ordered by `week`, before the `week >= 3` filter | Strictly prior week(s) within the same season | Yes — lag is applied before filtering, verified by direct read |
| Talent (`talent`, `Scaled_Talent`, `blue_chip_ratio`, `avg_player_rating`) | `../R Scripts/Merge_Predictors_CFB_Historical.R:7,20` — merged by `(year, team)`, no lag | Season = t (preseason-known composite) | Yes — sanctioned as-is, same rationale as the sibling `../cfb_win_total_model/` project's talent exception |
| Coaching (`Total_Games_Coached`, `Winning_Percentage`) | `../R Scripts/Merge_Predictors_CFB_Historical.R:11-18` — `lag(year)` per coach `Name`, arranged descending, then `na.omit()` | Season = t-1 (prior season's cumulative record) | Yes — verified by direct read |
| Returning production (12 columns) | `../R Scripts/Merge_Predictors_CFB_Historical.R:27-30` — merged by `(year, team)`, no lag | Season = t (preseason-known — this season's roster returning production, computed from last season's departures, is knowable before week 1) | Yes — sanctioned as-is |
| `home_favored`, `spread` | `../R Scripts/Full_CFB_Game_Outcome_Historical.R:99-110` — string-match of `home_team` against the betting line's `formatted_spread` field | Pre-game (betting market close, not post-game) | Yes — not derived from the outcome |
| `home_covered` (label) | `../R Scripts/Full_CFB_Game_Outcome_Historical.R:112-127` | N/A (target) | N/A |

## Label derivation (`home_covered`)

```r
home_covered <- ifelse(
  (home_favored == 1) & (home_minus_away > (-spread)) |        # favorite covers
    (away_favored == 1) & (home_minus_away >= 0) |               # underdog wins outright
    (away_favored == 1) & (spread >= 0) & (home_minus_away < 0) & # underdog covers, loses
      (home_minus_away > (-spread)),
  1, 0
)
```

`spread` here is the **raw signed value** at that point in the R pipeline (negative = home
favored, standard betting convention) — before `Merge_Predictors_CFB_Historical.R:91` converts
it to `abs(spread)` for the modeling CSV. An exact push (margin equals the spread precisely)
satisfies none of the three OR'd conditions and therefore falls through to `home_covered = 0` —
**there is no separate push class**. This project carries that simplification forward
unchanged (see `docs/assumptions_and_limitations.md`) rather than silently reinterpreting the
existing historical labels. `tests/test_leakage.py::test_home_covered_matches_documented_derivation`
pins this derivation down in executable form against a synthetic fixture (the real CSV doesn't
carry raw `home_points`/`away_points`, so it can't be recomputed from the CSV itself).

## What this project's tests check (`tests/test_leakage.py`)

1. **No post-game columns present** — a denylist check (exact match on `home_points`,
   `away_points`, `home_minus_away`; substring match on `_result`, `final_score`) against the
   live CSV. Note: this must be an *exact* match for the score-like names, not substring —
   `home_points_avg_all`/`home_points_allowed_avg3` are legitimate, correctly-lagged rolling
   averages of *prior* games' scores, not this game's outcome, and a naive substring check
   flags them as false positives (caught by this project's own test suite during the build).
2. **`season`/`week` excluded from the feature matrix by default** — directly targets a
   confirmed bug in the current notebook (`../Python Scripts/CFB_Gambling_Model.ipynb`'s
   `exclude_vars` list only drops `game_id`/`home_team`/`away_team`, leaving `season`/`week`
   inside `X`).
3. **Walk-forward folds never leak future seasons**, and **are expanding, not random** — direct
   checks against `modeling/splits.py`.
4. **Final holdout excludes the COVID season** (2020).
5. **Correlation pruning's function signature** is pinned down to `(X, y, cfg)` — a guard
   against a future refactor that threads in a full unsplit dataframe instead of an
   already-fold-sliced `X`/`y` (Stage 1/2 selection must only ever see a single fold's training
   rows; see `scripts/select_features.py`).
6. **No completeness gap** — every row in the loaded CSV has `week >= 4` and no NA in any
   `prev_week_*`/`*_avg3`/`*_avg_all` column, matching the upstream `na.omit()` guarantee.
7. **Label derivation** — executable spec of the three-condition rule above.
