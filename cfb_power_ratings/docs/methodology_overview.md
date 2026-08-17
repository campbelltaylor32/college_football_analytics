# How the ratings work

A plain-language walkthrough of what goes into the rating and how it changes week to week. For
the full technical writeup (exact formulas, code references, file paths), see `methodology.md`.

## The rating itself

Every team gets one number: **points better or worse than an average FBS team, on a neutral
field.** If Team A is rated +20 and Team B is rated +5, Team A is expected to win by about 15
points on a neutral field — add a few points if Team A is at home.

That number comes from a method called **SRS (Simple Rating System)**. The idea: your score
margin only counts for as much as the team you put it up against. Beating a bad team by 30
shouldn't move your rating as much as beating a great team by 10. The rating adjusts every
team's raw scoring margin by the strength of who they played, so a team's rating reflects who
they beat and lost to, not just the scoreboard.

## Before the season starts: the preseason model

Before a single game is played, the rating comes entirely from a model trained on things that
are already known in the offseason:

- **Recruiting talent** — the team's recruiting composite, plus a corrected blue-chip ratio
  (share of the roster that are 4/5-star recruits — corrected because the standard version of
  this stat dilutes real figures by counting walk-ons in the denominator)
- **Returning production** — how much of last year's snaps/production is coming back
- **Transfer portal activity** — the net talent gained or lost through transfers
- **Coaching** — tenure, career win rate, and whether there's a coaching change this year
- **Recent history** — the team's own rating over the last 1-3 seasons (this turns out to be
  the single strongest predictor — recent form beats any single preseason proxy)
- **Last year's "luck" gap** — whether a team's record ran ahead of or behind what its scoring
  margin implied last season (tested honestly, and it didn't move the needle much — see
  `methodology.md` for the full accounting)

Feed a team's current profile into the model and it predicts a rating — this is what "Week 0"
rankings are built from, before kickoff.

## Once the season starts: blending in real results

The preseason number doesn't get thrown out once games start — it fades out gradually as real
results accumulate. Think of the preseason prediction as counting like **5 "extra" games**
against a league-average opponent, mixed in with the team's real games:

- **0 games played**: the rating is basically just the preseason prediction
- **A few games in**: the preseason number and real results are both pulling roughly equal
  weight
- **8-10 games in**: real results dominate — the preseason guess barely matters anymore

There's no hard cutoff or manual switch-over — the blend shifts smoothly every week as more
real games get added to the average.

## Does it actually work?

Checked against reality, not just assumed:

- The preseason model beats "assume nothing changes from last year" and "assume every team is
  average" by a wide margin at predicting how a team's season actually turns out.
- Its implied point spreads track real Vegas betting lines closely — roughly 0.75-0.85
  correlation across every season tested.
- Replaying a full past season week by week, the blended rating's implied spread lands within
  about 4 points of the real closing line on average — and that error shrinks as the season
  goes on, confirming the "prior fades, results take over" design actually behaves as intended.

## What it can't do

- No preseason model can predict injuries, in-season coaching changes, or a young team suddenly
  clicking — it's a starting point, not a crystal ball.
- Talent/recruiting data for a brand-new season sometimes isn't published by the data provider
  yet at the time rankings are generated — when that happens, the model leans more on returning
  production, coaching, and recent history instead.
- This isn't trying to beat the betting market — it's a standalone power rating, not a betting
  system.
