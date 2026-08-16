# Data Leakage Rules

This document is the canonical reference for what information a player-week row is allowed to
use, and why. Unlike the sibling `cfb_win_total_model` project, **there are no sanctioned
"read season t as-is" exceptions here** -- no talent/returning-production equivalent exists at
player grain in this database. Every feature in this project is either a strictly-prior-game
rolling statistic or a genuinely pre-kickoff-known schedule/context fact. If you find code
that reads a target game's own realized stats as a feature, that is a bug, not a design
choice.

## The general rule

For a target game (a specific `athlete_id` in a specific `game_id`), every feature must be
derivable from information available **before that game kicks off**. In practice:

| Category | Source table(s) | Timing rule |
|---|---|---|
| Player rolling rushing workload | `plays` (resolved via `team_rosters`) | Player's own games strictly before the target game (`eligibility.py` merge_asof, `allow_exact_matches=False`) |
| Team offensive context (rush/pass mix, tempo, ToP) | `game_team_stats` | Team's own games strictly before the target game (two-step compute-then-lag) |
| Opponent defensive context (run defense allowed) | `game_team_stats`, `plays` | Opponent's own games strictly before the target game (two-step compute-then-lag) |
| Game context (home/away, neutral site, conference game, rest days) | `games` (schedule_spine) | Identity/schedule structure for the target game itself -- known before kickoff by construction, never a result |
| Betting context (spread, over/under; optional, off by default) | `betting_lines` | Pre-game posted line for the target game itself |

## The two-step compute-then-lag pattern

Every rolling feature (player, team-offense, opponent-defense) is built in two explicit
steps, mirroring the legacy `Merge_Predictors_CFB_Historical.R` R pipeline's established
pattern rather than computing an already-exclusive rolling window directly:

1. **Compute** trailing-3-game (`avg3`) and cumulative season-to-date (`avg_all`) aggregates
   over each entity's own rows sorted by date, **current row included**
   (`features/rolling_utils.py::compute_rolling_and_lag`).
2. **Lag** every rolled column by `.shift(1)`, grouped by entity -- only the `_lag1` columns
   are ever exposed as features for that entity's OWN next row.

This split is deliberate: it lets `tests/test_feature_shifting.py` independently recompute
"value at row i" and assert it equals "raw rolling value through row i-1," which is a much
simpler, more auditable assertion than re-deriving an already-exclusive window from scratch.

## Why eligibility.py does NOT just reuse features/rushing_workload.py's `_lag1` columns

A target game for an arbitrary future or bye-week-adjacent week is **not necessarily the
player's own next played game** -- a player can miss a week (bye, injury, personnel package)
between their last recorded carry and the game being predicted. `features/rushing_workload.py`'s
`_lag1` columns only answer "through the row's own immediately-prior played game," which is
wrong in exactly that situation.

`eligibility.py` instead uses `pandas.merge_asof(..., direction="backward",
allow_exact_matches=False)` to find, for ANY target game date, that player's most recently
PLAYED game strictly before it, and carries forward THAT game's own inclusive `avg3`/`avg_all`
values (renamed `_avg3_asof`/`_avg_all_asof`). `allow_exact_matches=False` is the specific
guard against lookahead here: without it, a row where the player DID record a carry in the
target game itself would incorrectly match against its own same-date, same-game realized
stats.

These `_avg3_asof`/`_avg_all_asof` columns -- not `rushing_workload.py`'s `_lag1` columns --
are what actually appear in the final modeling table (see `dataset.py`); this is why
`dataset.build_feature_registry` registers `eligibility.describe_features()` and not
`rushing_workload.describe_features()`.

## Target construction is not a leakage vector, but is a deliberate design choice

`targets.py` builds `rushing_yards` via a LEFT JOIN of the (pre-game-known) eligibility spine
against realized per-game rushing, filling unmatched rows to 0 rather than dropping them. This
is a target-population design decision (see `docs/assumptions_and_limitations.md`), not a
leakage concern -- the target itself is, definitionally, the thing being predicted, so there is
no lookahead question about which game's rushing_yards value a row's target is.

## Automated verification

- `tests/test_feature_shifting.py` independently recomputes each rolling module's `_lag1`
  values from raw rows and asserts equality against the module's own output.
- `tests/test_leakage.py` asserts every row in `outputs/feature_analysis/feature_registry.csv`
  is tagged `known_before_kickoff=True`, and exercises `modeling/splits.py`'s walk-forward
  folds for the season-leak guarantee.
- `tests/test_eligibility.py` specifically asserts the `merge_asof` bye-week-carry-forward
  behavior and the `allow_exact_matches=False` same-game exclusion.
