# Data Dictionary

## Raw source tables (read-only; owned by `SQL Scripts/` in the repo root)

| Table | Role in this project |
|---|---|
| `plays` | Play-by-play. `rusher_player_name`/`yards_gained`/`success`/`stuffed_run`/`rz_play`/`epa` on rush plays are the raw material for player rushing workload; `def_pos_team`-grouped rows give opponent run-defense-allowed stats. |
| `team_rosters` | `athlete_id`, `position`, `team`, `season` -- the join target for resolving `plays.rusher_player_name` to a stable player identity. |
| `game_team_stats` | Team-week box score (rushing_attempts, attempted_passes, possession_time_minutes, and the `*_allowed` mirror columns). Source for team offensive context and the volume side of opponent defensive context. |
| `games` | Schedule (season, week, start_date, home/away team, neutral_site, conference_game, division). Source of `schedule_spine.py`'s row universe -- the only table with rows for not-yet-played games. |
| `betting_lines` | Optional (off by default, `data.yaml: include_betting_context`). Consensus spread/over_under per game. |

See `../SQL Scripts/README.md` and `../SQL Scripts/schema.sql` for full raw-column definitions.

## Player-game rushing (raw, name-keyed) -- `player_game_rushing.py`

One row per `(rusher_player_name, pos_team, season, week, game_id)`, aggregated from `plays`
filtered to `play_type IN ('Rush', 'Rushing Touchdown')`:

| Column | Definition |
|---|---|
| `carries` | Count of rush attempts |
| `rushing_yards` | `SUM(yards_gained)` -- preferred over `yds_rushed` (verified 0% NULL vs. up to 43 NULL rows/season for `yds_rushed`) |
| `yards_per_carry` | `rushing_yards / carries` |
| `success_rate` | `MEAN(plays.success)` |
| `explosive_runs` | Count of carries with `yards_gained >= explosive_run_yard_threshold` (config default 15) |
| `explosive_run_rate` | `explosive_runs / carries` |
| `stuffed_run_rate` | `MEAN(plays.stuffed_run)` |
| `red_zone_carries` | `SUM(plays.rz_play)` |
| `avg_epa_per_rush` | `MEAN(plays.epa)` |
| `first_down_rate` | **Approximation**: `MEAN(yards_gained >= distance)` -- `plays` has no explicit first-down-achieved flag. Slightly overcounts penalty-aided first downs, slightly undercounts spot-adjustment edge cases. |

## Player identity resolution -- `player_resolution.py`

`plays.rusher_player_name` is free text with no stable ID. Resolved to `team_rosters.athlete_id`
via, in order: (1) exact `first_name + " " + last_name` match on `(team, season, position)`,
(2) a normalized fallback (accent/punctuation/suffix-stripped, casefolded) for exact-match
misses, (3) anything still unmatched, or matching >1 distinct athlete_id, is excluded (`ambiguous`/
`unmatched`). See `docs/assumptions_and_limitations.md` for verified match-rate numbers and
residual risk (e.g. box scores that abbreviate a first name to an initial, such as "A. Jeanty"
for Ashton Jeanty, are a known miss the normalized pass does not fix).

## Eligibility / modeling features -- `eligibility.py` (the `_avg3_asof`/`_avg_all_asof` columns)

For every `(carries, rushing_yards, yards_per_carry, success_rate, explosive_runs,
explosive_run_rate, stuffed_run_rate, red_zone_carries, avg_epa_per_rush, first_down_rate)`:
a trailing-3-game (`_avg3_asof`) and season-to-date (`_avg_all_asof`) rolling average, as of
the player's most recently PLAYED game strictly before the target game (see
`data_leakage_rules.md` for why this differs from `rushing_workload.py`'s own `_lag1` columns).
Plus `prior_games_played` (count of the player's own games at-or-before the matched as-of
game) and `eligible` (the workload-relevance gate itself).

## Team offensive context -- `features/team_offense_context.py`

Team-week, rolled/lagged (`_avg3_lag1`/`_avg_all_lag1`):

| Column | Definition |
|---|---|
| `rush_pct` | `rushing_attempts / (rushing_attempts + attempted_passes)` |
| `pass_pct` | `1 - rush_pct` |
| `tempo_plays_per_minute` | **Approximation**: `(rushing_attempts + attempted_passes) / possession_time_minutes` -- a plays-per-minute-of-own-possession tempo proxy, NOT raw seconds-per-play (this DB has no per-drive/per-play clock aggregate at team-game grain). |
| `possession_time_minutes` | `game_team_stats`' own generated column, parsed from the raw `"MM:SS"` string by the DB schema. |

## Opponent defensive context -- `features/opponent_defense_context.py`

Team-week (keyed by the DEFENSE's own team; joined onto a player row via that player's
OPPONENT for the target week), rolled/lagged:

| Column | Definition |
|---|---|
| `rushing_yards_allowed`, `yards_per_rush_attempt_allowed` | `game_team_stats`' own `*_allowed` columns (that table already stores the defense's own row -- no self-join needed) |
| `def_success_rate_allowed`, `def_explosive_runs_allowed`, `def_explosive_rate_allowed`, `def_epa_allowed_per_rush`, `def_stuffed_rate_forced` | `plays` grouped by `def_pos_team`, same rush-play filter and explosive-run threshold as the player side |
| `def_possession_time_minutes` | This team's own possession minutes -- when joined via a player's OPPONENT, this is "opposing team's time of possession," an explicitly requested predictor |

## Game context -- `features/game_context.py`

`is_home`, `neutral_site`, `conference_game`, `rest_days` (days since the team's previous
game, imputed to `features.yaml: default_rest_days_season_opener` for a season opener). Plus
optional `spread`/`over_under` if `data.yaml: include_betting_context` is enabled.

## Target -- `targets.py`

`rushing_yards`: this player's realized rushing yards in the target game, 0-filled (not
dropped) for an eligible player-game with no realized carries. `played` (boolean, diagnostic
only, excluded from features): whether the player actually recorded a carry in the target
game.
