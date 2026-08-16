# Project Story

## Why this project exists

The repo's two existing modeling projects (`cfb_win_total_model`, `cfb_spread_model`) predict
game/season-level outcomes. This project is the first player-level model in the repo: weekly,
forward-validated running back rushing-yards predictions, intended to extend the same
disciplined walk-forward/no-lookahead conventions down to player-prop grain.

## What made this harder than a straight port of the sibling projects

No player-game rushing table existed anywhere -- not in the MySQL cache, not in the legacy CSV
pipeline. It had to be built fresh from `plays`' free-text `rusher_player_name` field, which
meant solving a real identity-resolution problem (matching a text string to a stable
`athlete_id`) that the sibling team/season-grain projects never had to face.

Two live data-quality findings, made during planning by querying the actual database rather
than assuming the sibling projects' season boundaries would transfer unchanged, materially
changed this project's config defaults from theirs:

1. `plays.rusher_player_name` is ~97-99.5% NULL for 2025 from week 9 onward -- a live,
   time-bound ingestion gap that `games.completed` alone would never surface. This pushed
   `excluded_seasons` to include 2025 (distinct from the COVID exclusion) and added a new,
   production-facing data-quality gate (`check_rusher_name_completeness`) that the sibling
   projects have no equivalent of, wired directly into the weekly inference script as a hard
   abort.
2. Name-to-roster resolution decomposes cleanly once matched against the FULL roster (not just
   RB) -- ~95-97% resolved, only ~2-3% genuinely unmatched -- which is what grounds
   `data_validation.check_player_resolution_match_rate`'s floor. Matching against RB-only
   naturally resolves a lower ~68-71% of all carries, which is expected (QB scrambles and other
   positions' occasional carries correctly don't match), not a resolution failure -- this
   distinction had to be explicitly measured, not assumed, before any match-rate check could be
   written responsibly.

## Design choices made explicitly, not by default

- **RB only for v1** (QB rushing deferred) and **workload-relevance filtering** (not a
  two-stage hurdle model, not unfiltered full-roster prediction) were both deliberate scope
  decisions made with the user before implementation, not assumptions baked in silently.
- **Explosive-run count AND rate, on both the player side and the opposing-defense side**, using
  a shared yardage threshold, were called out explicitly as first-class predictors per the
  user's direct request -- not left implicit inside a general "workload" feature bucket.
- **Eligibility and target construction share one `merge_asof`-based mechanism**
  (`eligibility.py`) rather than two independently-implemented pieces of logic, specifically so
  the workload gate and the player-grain rolling features can never silently drift apart from
  each other.

See `docs/assumptions_and_limitations.md` for the full list of open, explicitly-flagged
judgment calls (eligibility threshold tuning, the initial-only-name matching gap, betting
context) that this build did not resolve unilaterally.
