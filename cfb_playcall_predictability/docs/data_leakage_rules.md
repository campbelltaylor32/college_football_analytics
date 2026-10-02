# Data leakage rules

Every feature on a call's row must be knowable before that snap. Each rule is enforced in code and covered by `tests/test_leakage.py`.

| Feature group | What it may use | Where enforced |
|---|---|---|
| Situation (down, distance, field position, clock, score, timeouts, WP) | the pre-snap state cfbfastR records for the play (`pos_score_diff_start`, `*_timeouts_rem_before`, `wp_before`) | `features/situation.py` |
| Market (spread, total) | closing pregame lines | `context.load_lines` |
| In-game flow (`g_*`, `prev_*`, `drive_*`) | this offense's **earlier** calls in the same game: cumulative sum minus the current row, or `shift(1)` | `features/in_game.py` |
| Tendencies (`prior_pr_*`, `std_pr_*`), offense efficiency (`off_*`), opponent defense (`def_*`) | `prior_*`: the full previous season. `std_*`: this season's games with `game_number` < current game | `features/rates.py` |
| Head coach change | the `coaches` table for this and the previous season, known in the preseason | `context.load_head_coaches` |

Also:
- **Outcome columns are never features.** `feature_columns()` is an explicit whitelist. `EPA`, `success`, `yards_gained`, `play_type` and `play_text` describe the result of the play, so they are excluded.
- **Splits are walk-forward by season.** Each fold trains only on seasons before its validation season (`modeling/splits.py`).
- **The ranking season is never in training.** `rank_predictability.py` refuses to run if it is.
