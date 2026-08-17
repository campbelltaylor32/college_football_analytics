# Methodology

## 1. The rating scale

Every rating produced by this project is in **points, on a neutral field, mean 0 across the
full FBS field** (for whatever season/week snapshot it was computed at). `rating_home -
rating_away + hfa` is a predicted point margin for a hypothetical matchup; positive means the
home team is favored, negative means the away team is favored. This is the same scale
`srs.py::compute_srs` (the historical target) and `rating_engine.py::update_ratings` (the
in-season blend) both produce, by construction — a rating is only ever meaningful relative to
other ratings computed the same way.

## 2. Historical SRS target (`srs.py`)

An **opponent-adjusted, home-field-adjusted Simple Rating System**, generalizing
`cfb_pythagorean_model/opponent_adjusted_analysis.py::compute_srs`:

1. **Home-field advantage** (`estimate_home_field_advantage`): one league-wide constant, the
   mean `home_points - away_points` across every FBS-vs-FBS, non-neutral-site, completed game
   in the full historical window. Verified live: **~4.33 points**. This is backed out of every
   game's margin before the SRS iteration runs (`site_adjusted_margin`), so the resulting
   rating reflects team strength alone, not a home-heavy schedule.
2. **Opponent adjustment**: the standard iterative fixed point —
   `adj_margin[game] = site_adjusted_margin[game] + opponent_rating`,
   `team_rating = mean(adj_margin over that team's games)`, recentered to mean 0 every pass,
   repeated 200 times. Verified to converge to machine precision on the real, densely-connected
   FBS schedule (200 vs. 1000 iterations agree exactly); does **not** converge for a pathological
   2-team/1-game toy schedule (a real property of Jacobi-style fixed-point iteration on
   sparsely-connected graphs — see `tests/test_srs.py`'s explicit test of this — never an issue
   in practice, where every team plays 8+ games against well-connected opponents).
3. **Non-FBS opponents** are pooled into one fixed-rating pseudo-team (`generic_low_major`)
   rather than dropped or individually rated: calibrated so a perfectly-average FBS team's
   expected margin against a non-FBS opponent, plus that pool's rating, nets to 0. This keeps an
   FBS team's money-game results in its profile without trying to accurately rate hundreds of
   FCS/D2/D3 programs.

`build_historical_srs_table` runs this once per season (2013 onward, matching `games`'
earliest coverage) — one row per (team, season), the training **target** for the preseason
model below.

## 3. Preseason model (`dataset.py`, `features/`, `modeling/`)

Predicts a season's actual SRS from only preseason-known information:

- **`talent_recruiting.py`**: the raw team-talent composite, plus a corrected blue-chip ratio
  (ported from `cfb_talent_distribution/build_corrected_blue_chip_ratio.R`'s join-fix — matches
  recruits via `athlete_id` OR the roster's own `recruit_ids` link, divides by matched-recruit
  count rather than the full walk-on-inclusive roster). Verified against the same Alabama 2021
  sanity check that project used (~88% here vs. the ~86% publicly reported, vs. ~54% from the
  uncorrected shared-pipeline version).
- **`returning_production.py`**: direct pass-through of CFBD's own preseason returning-
  production metric.
- **`roster_turnover.py`**: net transfer-portal talent (real origin/destination/rating per
  move, live from `cfbd_recruiting_transfer_portal` for 2021+; roster-set-diff inference for
  earlier seasons, which the portal endpoint has no data for at all).
- **`coaching.py`**: tenure at the school, career win% entering the season (built only from
  seasons strictly before it), and a coaching-change indicator.
- **`program_history.py`**: the team's own trailing 1–3 season SRS (individually and as a
  rolling mean) — verified to be one of the strongest signals in practice, since recent
  on-field performance is normally more informative than any single preseason proxy alone.
- **`pythagorean.py`**: last season's Pythagorean-expected win% (`PF²/(PF²+PA²)`, ported from
  `cfb_pythagorean_model/pythagorean_analysis.py`, which validated the classic k=2 exponent at
  R²=0.797 against actual 2025 win% — a numerically-fit exponent only reached R²=0.801, too
  small a gain to justify fitting one here) and the gap between that and the team's *actual*
  win% (`win_pct_over_pythagorean_lag1`) — a "regression to expectation" signal distinct from
  SRS, which is already opponent-adjusted margin rather than raw scoring-implied win probability.
  **Tested, not just added**: retraining with these two features included changed pooled
  walk-forward MAE by less than 0.01 (6.407 → 6.412) and left market-spread correlation
  essentially unchanged — a real, honestly-reported null result, not a win. Standardized
  coefficient/correlation analysis explains why: `pythagorean_win_pct_lag1` correlates
  reasonably with the target (r=0.55) but is largely redundant with `srs_lag1` (both measure
  "how good was this team last season," just via different math) once both are in the model;
  `win_pct_over_pythagorean_lag1` (the actually-novel "luck" signal) only weakly correlates
  with next season's SRS (r=0.13) — plausibly because a ~12-game college season doesn't give
  the same-sport regression-to-expectation effect (well documented in 82/162-game pro leagues)
  enough games to show up cleanly. Left in the model rather than reverted (per explicit
  instruction), since it doesn't measurably hurt either — but it's not pulling meaningful weight.
- **`roster_experience.py`** (built and tested, **not** included in the model — see below):
  two independent roster age/experience signals were tried on the theory that older, more
  experienced rosters perform better. (1) `class_avg`/`class_valid_row_share`, straight from
  `team_rosters.year` (the eligibility-class field), filtered to plausible values (1–6,
  excluding a confirmed data-corruption pattern where `year` sometimes just equals the season
  itself — verified live: 81.5% of 2015 rows, improving to 0% by 2026) and gated to NaN when
  too few valid rows exist for a team-season. (2) `avg_roster_experience`/`veteran_roster_share`,
  a `year`-independent alternative: how many prior seasons a given `athlete_id` (100% populated
  in every season checked) has appeared in `team_rosters` at all, any team, so a transfer's
  prior experience still counts. **Tested and reverted — this was a real regression, not a null
  result.** Isolated one feature group at a time against the same walk-forward harness: baseline
  ridge MAE 6.412; class-only 6.618 (+0.206, the larger hit, consistent with the known `year`
  corruption); tenure-only 6.450 (+0.038, smaller but still net negative, unlike Pythagorean's
  near-zero effect); both combined 6.709 (worse than either alone). Market-spread correlation
  dropped in all 4 validation seasons with both included (e.g. 2021: 0.807 → 0.769). The module
  and its tests are kept as working, documented infrastructure (`dataset.py` still assembles the
  4 columns into the dataframe) — just excluded from `FEATURE_COLUMNS` — in case a future,
  better-constructed version of either signal is worth re-testing.

Candidates (`modeling/models.py`): `ridge` (with median imputation), `gradient_boosting`
(`HistGradientBoostingRegressor`, chosen for native NaN handling — the feature set has
*structural* NaNs, like `srs_lag3` for a team's first eligible season, not occasional
missingness), `xgboost` (optional). Baselines: `overall_mean`, `prev_season_srs`.

**Validation**: expanding-window walk-forward by season (`modeling/splits.py`), never a random
split — train on every season strictly before the validation season, COVID (2020) excluded.
Primary metric: pooled MAE against actual SRS. **Secondary, target-independent check**
(`evaluate_against_consensus_spread`): does `predicted_rating_home - predicted_rating_away +
hfa` track that season's real betting-market spread (averaged across every provider reporting
that game — CFBD's own `provider='consensus'` field is only populated through 2022, verified
live)? Both are reported in full by `scripts/train_preseason_model.py` — see the README's
"Results" section for the actual numbers from the last real run.

## 4. In-season blending — "SRS with a prior" (`rating_engine.py`)

The one genuinely new numerical idea in this project. Rather than a manual if/else blend
("use the preseason number for the first N weeks, then switch to pure in-season SRS"), each
team's preseason prediction is folded directly into the SRS iteration as `phantom_games`
synthetic games against a fixed, rating-0 "league average" opponent:

- 0 real games played → a team's rating is the average of `phantom_games` copies of its
  preseason prior → equals the prior (shifted by the one constant every week's ratings are
  recentered by — see below).
- *N* real games played → the average is now `phantom_games` prior-copies plus *N* real,
  opponent-adjusted results → the prior's influence shrinks smoothly as *N* grows, with no
  discontinuity. `effective_prior_weight = phantom_games / (phantom_games + games_played)` is
  reported alongside every rating as a descriptive (not load-bearing) measure of how much of a
  team's current rating is still "preseason guess" vs. "actual results."

**On the recentering constant**: every SRS pass recenters ratings to mean 0 across the full FBS
field, including phantom-games-only passes. This means a team with 0 games played gets its raw
preseason prediction *minus the mean of all teams' preseason predictions that season* — not
literally unchanged. This is intentional, not a rounding quirk: it keeps every week's ratings on
the same mean-0 scale the historical SRS target itself uses, which is what makes ratings
comparable across weeks and seasons. Since the model is trained against a mean-0 target, its raw
predictions should average close to 0 already, so this shift is normally small (~1 point on
2025's preseason ratings, verified).

`phantom_games` defaults to **5** (`config/modeling.yaml`) — chosen as a plausible "roughly
faded out by week 8-10" default, then verified (not just assumed) via
`scripts/backtest_season.py`'s 3/5/8 sensitivity sweep on the 2024 season: 5 had the lowest
market-spread MAE and Brier score of the three tested.

## 5. Win probability

`P(home win) = Φ(predicted_margin / residual_std)` — the identical methodology
`cfb_cover_model`'s `ResidualProbabilityRegressor` uses (a normal-CDF conversion of a point
prediction, calibrated by a single `residual_std` fit once from historical residuals), applied
here to this project's own rating differential rather than a spread-relative one.
`residual_std` is fit from `(actual_site_adjusted_margin - (team_srs - opponent_srs))` across a
multi-season historical sample (`rating_engine.historical_site_adjusted_residuals`).

## 6. Backtest design (`scripts/backtest_season.py`)

Reconstructs a full past season week-by-week: for each week *N*, trains a preseason model on
**only** seasons strictly before the target season (a genuinely out-of-sample prior, distinct
from `train_preseason_model.py`'s production artifact, which is trained on all available
history including recent seasons — appropriate for making the best real 2026 prediction, but
not backtest-clean for seasons it was itself trained on), blends in games through week *N-1*,
and scores every real game in week *N* against the actual market-average spread and the actual
final margin. Reports both the pooled and early-vs-late-season breakdown, plus the
`phantom_games` sensitivity sweep — see the README's "Results" section for the real 2024 numbers.
