# Data Leakage Rules

This document is the canonical reference for which season each feature category is allowed
to read from, and why. **Two categories are sanctioned exceptions to the general "lag by one
season" rule** — they are called out prominently here specifically so a future contributor
does not "fix" them into an off-by-one bug.

## The general rule

For a target season `t`, every feature must be derivable from information published or
observable **before** season `t`'s first game. In practice this means: any statistic that is
computed from season `t`'s own games (wins, points, EPA, coaching record, etc.) must use
season `t-1` or earlier.

## Per-category source-season table

| Category | Source table(s) | Season rule | Sanctioned exception? |
|---|---|---|---|
| Prior-year team performance | `games`, `game_team_stats`, `plays` | **t-1** | No |
| Returning production | `returning_production` | **t (as-is)** | **YES** |
| Talent (composite) | `team_talent` | **t (as-is)** | **YES** |
| Recruiting (roster-matched, blue-chip ratio, positional talent) | `recruiting_players` + `team_rosters` | **t (as-is)** | Same rationale as talent |
| Recruiting (trailing class rank) | `recruiting_players` | **t-4 through t-1** | No |
| Roster turnover / transfers | `team_rosters` | **t-1 and t** (set differences) | No — uses both, never t's *results* |
| Coaching (career win %, tenure, prior record, SP+) | `coaches` | **< t** (season=t row read ONLY for the coach's name) | No |
| Schedule structure (opponent identity, dates, conference) | `games` | **t** (identity/structure only) | No — see below |
| Schedule opponent-strength | `targets`, `team_talent`, `returning_production` (for opponents) | **t-1** | No |
| Program-level rolling history | multiple, aggregated over seasons | **t-w .. t-1** for window size w | No |

## The two sanctioned as-is exceptions

### `returning_production` (season = t, not t-1)

CFBD's `returning_production` table already represents *"how much of last season's on-field
production is coming back for season t"* — it is inherently a preseason-known quantity,
published before season `t` begins. Using `season = target_season` here is correct. If you
see code that lags this by one more season (`target_season - 1`), that is a bug, not a
leakage fix.

### `team_talent` / roster-matched recruiting features (season = t, not t-1)

The team talent composite and blue-chip/positional-talent ratios (computed from
`recruiting_players` joined to `team_rosters` for season `t`) reflect the roster a team is
bringing INTO season `t` — again a preseason-known quantity. Using `season = target_season`
is correct here too.

## The one place a season=t row is read from a leakage-sensitive table

`coaching.py`'s `coaching_change_indicator` reads the season=t `coaches` row's
**first_name/last_name only** — the identity of the incumbent coach heading into the season
is public information well before kickoff. It never reads that row's `wins`, `losses`,
`games`, `srs`, `sp_overall`, `sp_offense`, or `sp_defense`, all of which reflect season `t`'s
own outcome and would be a hard leak (`wins` is literally derivable from the target).

## Automated verification

- `tests/test_feature_shifting.py` asserts every t-1 module's internal `_source_season`/
  `_source_seasons` helper returns a season strictly less than the target, and asserts the two
  sanctioned exceptions return the target season exactly (a positive check against an
  accidental "fix").
- `tests/test_leakage.py` cross-references `outputs/feature_analysis/feature_registry.csv`'s
  `source_season`/`known_before_kickoff` columns, and verifies `modeling/splits.py`'s
  walk-forward folds never include a validation season in their own training set.
- `outputs/feature_analysis/feature_registry.csv` is generated directly from each feature
  module's `describe_features()` function, co-located with the feature-building code, so the
  registry cannot silently drift out of sync with what the code actually does.
