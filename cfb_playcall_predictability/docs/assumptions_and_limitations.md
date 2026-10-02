# Assumptions and limitations

## Unit of analysis: team-offense, not play caller
CFBD has no offensive-coordinator or play-caller data, so every call is attributed to the offense (`pos_team`). The head coach's name is shown for context only. Teams where the OC calls plays (or where calling changed mid-season) are still ranked as one offense.

## Labels
- **Pass** = dropback outcomes: completions, incompletions, passing TDs, sacks and interceptions.
- **Run** = rushes and rushing TDs.
- **Fumbles and safeties** are labeled from cfbfastR's own `rush`/`pass` flags, and dropped when neither is set.
- **Scrambles are counted as runs.** The source logs them as rushes and the play text doesn't flag them. Some designed passes are therefore labeled runs. This mostly affects teams with running QBs, which will look a little more run-heavy and a little noisier than they are.
- **Excluded:** kneels, spikes, plays with no down (kickoffs, PATs, 2-pt tries), overtime, and penalties with no snap.
- **Garbage time** is excluded from training and scoring: margin over 38 in Q2, over 28 in Q3, over 22 in Q4 (same as `cfb_ryan_day_offense`). Garbage-time snaps still count as in-game history.

## Data source
- Plays come from the cfbfastR release files (`scripts/pull_pbp.R`), not the MySQL `plays` table:
  - `plays` has no score or timeout columns.
  - Rebuilding score from cumulative `drive_pts` matched the final score in only ~52% (2024) to ~59% (2016) of games.
  - The release stores `id_play` as a double, which loses the last digits of the 18-digit id, so release columns can't be joined back onto `plays` exactly.
- Games, betting lines and coaches still come from MySQL.
- **Regular season only.** The MySQL `games` table, which holds neutral site, lines and FBS division, has regular-season games only.
- 2022+ release files include FCS games. Only calls by FBS offenses are modeled; FCS opponents still contribute to defense features.
- Early seasons (2014–2016) often lack a betting total. The spread is ~99% present. The tree models handle the missing values natively.

## Model
- XGBoost and LightGBM come out about even. The pick is whichever has the lower mean walk-forward log loss (`outputs/model_comparison/summary.csv`).
- No post-hoc calibration was added: the GBMs' expected calibration error on validation is under 1%. See `outputs/model_comparison/calibration_holdout.csv`.

## Predictability score
- `predictability = 1 − LL_model / LL_null`. `LL_null` is the log loss of always predicting the team's own pass rate for the season. The score is the share of the team's run/pass uncertainty that the model explains.
- Run/pass mix is accounted for. A 50/50 team and a 30/70 team are both judged on whether you can tell *when* they will pass. Raw accuracy instead rewards lopsided offenses: Army is "predictable" on accuracy just by always guessing run.
- The model already knows each team's tendencies (prior season, season-to-date and in-game). So "predictable" means *predictable given what a well-prepared opponent knows*, not just league-average habits.
- **Small samples.** Early in the season a team has ~4 games and ~250 calls. The 90% CIs (bootstrapped over games) are wide, and middle-of-the-pack differences are not meaningful. Read the extremes.
- Read the year-over-year stability check (`outputs/rankings/stability.csv`) before treating the score as a trait of a coach or offense.
