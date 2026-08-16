# Assumptions and Limitations

Every item below reflects a concrete, verified finding against the live `cfb_football`
database (not a guess), with the resulting design decision and its consequence.

## 1. No 2026 schedule — this build is a historical backtest, not a live forecast

`SQL Scripts/ingest_to_mysql.R` only pulls `completed = TRUE` games, and the `games` table's
`MAX(season)` is 2025 with zero rows for 2026. Schedule-difficulty features (opponent
identity, home/road/neutral counts, bye weeks) require the target season's schedule to already
be in the database — which it isn't yet for 2026.

**Decision**: `target_season` in `config/modeling.yaml` is set to 2025, the most recently
completed season. The final walk-forward holdout and the "prediction" output
(`outputs/predictions/predicted_win_totals_2025.csv`) are the same season — a legitimate
backtest on real historical data, not a genuine future forecast. Every script accepts
`target_season` as a config value; the exact same code produces real 2026 predictions
automatically once that schedule is ingested (out of scope for this build — would require
extending the R ingestion script to also pull incomplete/future games).

## 2. Conference championship games are indistinguishable from regular-season games

Verified: `games.season_type` is 100% `'regular'` across all 28,248 rows, and `completed` is
100% `TRUE`. There is no separate classification for conference championship games — e.g.
TCU's 2022 Big 12 Championship loss to Kansas State (week 14, `notes = "Dr Pepper Big 12
Championship"`) carries `season_type = 'regular'` identically to every other game and is
counted in `regular_season_wins`. The `notes` field is free text and not reliably populated
across all CCGs, so it cannot be used as a clean filter. Bowl games and CFP games are simply
**absent** from the database entirely (the ingestion script never pulled them), so there is no
risk of those leaking into the target — only conference championship games are ambiguous.

**Decision**: documented as a known limitation of the target definition rather than silently
included. `regular_season_wins` should be read as "regular season + conference championship
game, where applicable."

## 3. Power/Group-of-5 conference mapping is a hardcoded, season-aware assumption

No conference-tier table exists in this schema. `config/features.yaml`'s `power_conferences`
block hardcodes the ACC/Big Ten/Big 12/SEC(/Pac-12) membership, with a season-aware override
for 2024+ (when the Pac-12 collapsed to 2 legacy members) and a manual override treating Notre
Dame as power-tier despite its "FBS Independents" conference label. This mapping was not
independently verified against every season/conference combination in the DB and should be
revisited if used beyond the 2015–2025 window this build covers.

## 4. Roster-based transfer inference is approximate, not ground truth

No dedicated transfer-portal table exists. `features/roster_turnover.py` infers transfers from
`team_rosters` athlete_id set differences year-over-year: a player who leaves team A's roster
and appears on team B's roster the following season is inferred as a transfer. A player who
simply disappears from all rosters is bucketed as `attrition_unknown` — this is genuinely
indistinguishable, from this schema alone, between a transfer we failed to match, a graduating
senior, a grayshirt, a decommit, an injury retirement, or a walk-on who was cut.

## 5. `coaches.games = 0` stub rows, and why `preseason_rank` is excluded by default

Verified: the most recent season in a coach's tenure can carry `games = 0`/`wins = 0` (a
not-yet-finalized row) even while `sp_overall` (a preseason SP+ projection) is already
populated — e.g. Ohio State/Ryan Day season=2025 shows `games=0` but `sp_overall=30.10`.
**All 136 `coaches` rows for season=2025 have `games=0` AND `preseason_rank IS NULL`.**
Historically (2004–2024), `preseason_rank` is populated for only ~20% of coach-seasons (only
ranked teams). `career_win_pct_entering_t` filters to `season < t AND games > 0` specifically
to exclude these stub rows from a coach's career record.
`use_coach_preseason_rank: false` in `config/features.yaml` by default, since the column is
architecturally safe to use (a preseason poll rank is legitimately known before kickoff) but
would be 100% NaN for the primary 2025 demo target in this DB snapshot.

## 6. `team_rosters.year` (eligibility class) is a dirty column

Verified live for season=2024: the value distribution includes `2024`:1,356 rows — i.e. ~5.9%
of that season's roster rows have the season number leaked into the class-year field instead
of a real 1–5 eligibility year. `roster_turnover.py` therefore uses athlete_id
roster-membership set differences as its primary returning/departed signal, never
`team_rosters.year`.

## 7. `game_team_stats` coverage is >99.8% complete — verified, not assumed

A left join of every FBS team-game slot (from `games`) against `game_team_stats` matches at a
>99.8% rate in every season 2013–2025 (e.g. 2024: 1623/1623 matched). This is documented as
"effectively complete," not treated as a data-quality risk requiring special handling.

## 8. Season 2020 (COVID) is excluded from training and validation by default

`games` row count for 2020 (544) is roughly 1/3 of neighboring seasons (~1,500–1,700), and
`plays` shows a corresponding dip. `config/modeling.yaml`'s `excluded_seasons: [2020]` removes
it from every walk-forward fold's training set and ensures it is never a validation season.
Rolling program-history features (`program_history.py`) still include 2020 as a raw historical
data point when it falls within a rolling window — this is a documented simplification, not an
oversight: a 5-year rolling average spanning 2020 will reflect that season's shortened
schedule as-is, without a special adjustment.

## 9. `betting_lines` was evaluated and rejected as a preseason market baseline

The user's original spec allows an optional "preseason-poll or market baseline if such data
already exists and is legitimately available before the season." `betting_lines` DOES exist in
this schema (36,443 rows, 2013–2025) but spreads are set **weekly**, not preseason — even
week-1 opening lines are published only days before kickoff, not before the season as a whole.
This was evaluated and explicitly rejected, not silently omitted.

## 10. OC/DC coaching changes are not derivable from this schema

No assistant-coach table exists. `coaching.py` only tracks head-coach identity and
career/tenure. Offensive/defensive coordinator changes — a real, meaningful predictor in
practice — are simply unavailable and are not fabricated.

## 11. `qb_departure_indicator` is a weak, near-universal-true signal in practice

Verified: FBS teams carry a median of 4 rostered QBs, and given typical roster churn
(especially in the transfer-portal era), essentially every FBS team loses at least one rostered
QB (of any depth-chart position) between consecutive seasons. This flag does not distinguish
"lost QB4" from "lost the starter" — a genuinely useful refinement would cross-reference
`plays.passer_player_name` frequency in season t-1 to identify the actual workhorse QB, which
was not attempted in this build (flagged as a v2 enhancement).

## 12. Recruiting-position taxonomy differs between `recruiting_players` and `team_rosters`

`recruiting_players.position` uses a granular high-school recruiting taxonomy (e.g. `PRO`/
`DUAL` for quarterbacks instead of a plain `QB`; `OT`/`OG`/`IOL`/`OC` for offensive line).
`team_rosters.position` uses a coarser roster-side grouping (e.g. plain `OL`). Positional
talent aggregates in `talent_recruiting.py` are computed against `recruiting_players.position`
consistently — mixing the two taxonomies silently (an early bug in this build, caught and
fixed during development) would undercount every position group except plain `QB`.

## 13. Data cutoff

All verified counts, season ranges, and specific examples in this document reflect the
`cfb_football` database as of **2026-08-03**. Re-running `scripts/inspect_database.py` after a
future ingestion pass may surface different (larger) counts; `data_validation.py` warns rather
than hard-fails on count drift for this reason.
