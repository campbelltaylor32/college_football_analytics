# Data leakage rules

Every rule below is a hard constraint: a rating (or a training row) computed for season *t* /
week *N* must never use information that wasn't actually available before that season kicked
off / before week *N*'s games were played.

## Preseason model features (`features/*.py`)

| Feature module | Uses data from | Why it's safe |
|---|---|---|
| `talent_recruiting.py` | `team_talent`, `recruiting_players`/`team_rosters` for season *t* | Recruiting classes sign and the talent composite is published before the season starts |
| `returning_production.py` | `returning_production` for season *t* | CFBD's own metric is itself a preseason-known quantity by construction |
| `roster_turnover.py` | Transfer portal (season *t*) or roster diff (season *t-1* vs *t*) | Portal moves and roster composition for season *t* are both settled before kickoff |
| `coaching.py` | `coaches` — season-*t* coach identity/tenure (known), career win% computed **only from seasons < t** | The career-win% cumulative sum explicitly excludes season *t*'s own games (see `coaching.py`'s `career_wins_prior`/`career_games_prior` — cumsum through *t* minus season *t*'s own contribution) |
| `program_history.py` | Trailing SRS from seasons *t-1*, *t-2*, *t-3* | Only ever looks backward from season *t*, never at season *t* itself |
| `pythagorean.py` | `games` for season *t-1* only | `build_pythagorean_features` queries `season IN {t-1 for each requested t}` explicitly, then relabels the result as season *t*'s feature — season *t*'s own games are never in the query at all, not just filtered out after the fact |
| `roster_experience.py` | `team_rosters` for season *t* (class-based); `team_rosters` for seasons strictly before *t*, within the lookback window (tenure-based) | Season-*t* roster composition is preseason-known (same justification as `roster_turnover.py`); the tenure count explicitly only counts seasons `< t` (see `_build_tenure_features`'s `s < row["season"]` filter) — **not currently in `FEATURE_COLUMNS`** (tested and reverted for hurting accuracy, not for a leakage reason — see `docs/methodology.md`) |

**No feature module reads `games`/`plays`/`game_team_stats` for season *t* itself.** A power
rating's whole purpose is to predict games before they happen; if a feature used season-*t*
game results, the preseason rating would be trivially circular.

## Historical SRS target (`srs.py`)

`build_historical_srs_table`'s target for season *t* is computed **only** from season-*t*'s own
completed games — this is correct and not leakage, because it's the *target being predicted*,
never fed back as a feature for that same season's preseason prediction. `program_history.py`
only ever uses *other* seasons' SRS values as features, never season *t*'s.

## Walk-forward validation (`modeling/splits.py`)

Expanding-window by season: each fold trains on every season strictly before the validation
season (COVID-2020 excluded), never a random split. A model evaluated on season 2023 has never
seen 2023 or later in training. `scripts/backtest_season.py` goes further for its own
out-of-sample prior: it retrains from scratch on seasons strictly before the specific backtest
target, deliberately distinct from `train_preseason_model.py`'s persisted production artifact
(which trains on all available history, appropriate for real 2026 predictions but not a valid
backtest source for recent seasons it was itself trained on).

## In-season blending (`rating_engine.py`)

`update_ratings(preseason_priors, completed_games_so_far, ...)` has no season/week awareness of
its own — **the caller is responsible for only passing games through week N-1** when scoring
week N. `scripts/update_ratings.py`'s `_completed_games_through` helper enforces this via
`WHERE week < :week` (strictly less than, not `<=`). Any future caller of `update_ratings` must
preserve this same discipline — there is no internal guard against accidentally passing a
same-week or future game.

## Live data paths (`live_data.py`, `cfbd_client.py`)

`games`/`betting_lines` in the MySQL DB are completed-games-only by design (verified in
`SQL Scripts/ingest_to_mysql.R`'s `filter(completed == TRUE)`) — there is structurally no way to
leak a future game's result through the DB path. The live CFBD fallback
(`live_data.fetch_completed_games`) explicitly filters to `completed=True` rows for the same
reason. `live_data.fetch_games` (used for the upcoming week's schedule) does **not** filter by
`completed` — by design, since it needs to return the schedule for games that haven't been
played yet — callers must not treat its output as a training/history source.
