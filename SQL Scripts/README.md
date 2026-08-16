# SQL Scripts

Raw-data cache for the CollegeFootballData (CFBD) API, accessed through the R
`cfbfastR` package. This exists so the R/Python modeling pipeline in
`R Scripts/` and `Python Scripts/` can eventually read historical data from a
local MySQL database instead of re-hitting the API on every run. It is a
**separate, parallel data store** — nothing in `R Scripts/` reads from it yet;
`schema.sql` and `ingest_to_mysql.R` populate `cfb_football` independently of
the existing `Data/*.csv` pipeline.

Design principle: this mirrors the **raw** API response for each endpoint,
column-for-column, not the aggregated/rolling/lagged features the existing R
scripts derive from it. Feature engineering (moving averages, EPA aggregates,
cumulative coaching records, blue-chip ratios, etc.) stays a downstream
concern — computed from these tables via SQL views or the existing R/Python
scripts, not baked into the raw cache.

## Setup

```bash
mysql -u root -e "CREATE DATABASE IF NOT EXISTS cfb_football;"
mysql -u root cfb_football < "SQL Scripts/schema.sql"
```

## Running the ingest

```bash
cd "College_Football_Gambling_Model"
nohup Rscript "SQL Scripts/ingest_to_mysql.R" > ingest.log 2>&1 &
tail -f ingest.log
```

Idempotent and resumable: before pulling a season/week (or season, for
annual endpoints), it queries the DB for what's already there and skips it.
Re-running after an interruption only fetches what's missing — it never
re-pulls or duplicates existing rows. Safe to `kill` and restart at any time.

Each API call is wrapped in retry-with-backoff (`with_retry()`): any thrown
error (rate limit, auth, transient server error — `cfbfastR` collapses all of
these into the same generic message, so they aren't distinguished by text)
is retried up to 5 times with exponential backoff (30s, 60s, 120s, 240s,
480s) before that one season/week is given up on and logged.

Each insert (`upsert_df()`) tries one batched `INSERT ... ON DUPLICATE KEY
UPDATE` for the whole pulled page first. If any row in that batch doesn't fit
the schema (an out-of-range numeric value, a foreign key with no matching
parent row, etc.), it falls back to inserting row-by-row, logging and
skipping only the specific row(s) that failed rather than losing the whole
page.

## Endpoints pulled

Every table below is one `cfbfastR` function. Columns listed are what the
API actually returns for that call (verified live against the API, not
inferred from the existing CSVs, which are already aggregated) — see
`schema.sql` for exact types and any deliberately-excluded columns.

### `games` ← `cfbd_game_info(year, week)`
One row per game. Pulled per season/week, `2013`–current, filtered to
`completed == TRUE`.
Columns: `game_id`, `season`, `week`, `season_type`, `start_date`,
`start_time_tbd`, `completed`, `neutral_site`, `conference_game`,
`attendance`, `venue_id`, `venue`, `home_id`, `home_team`, `home_division`,
`home_conference`, `home_points`, `home_post_win_prob`, `home_pregame_elo`,
`home_postgame_elo`, and the `away_*` equivalents, plus `excitement_index`,
`highlights`, `notes`, `playoff`.

### `betting_lines` ← `cfbd_betting_lines(year, week)`
One row per (game, sportsbook provider) — a game usually has several rows,
one per book. Pulled per season/week, `2013`–current (CFBD has no betting
data before 2013).
Columns: `game_id`, `season`, `season_type`, `week`, `start_date`,
`home_team_id`, `home_team`, `home_conference`, `home_classification`,
`home_score`, and the `away_*` equivalents, plus `provider`, `spread`,
`formatted_spread`, `spread_open`, `over_under`, `over_under_open`,
`home_moneyline`, `away_moneyline`.

### `team_talent` ← `cfbd_team_talent(year)`
One row per team per season. Whole-league pull, one call per year,
`2015`–current (earliest year CFBD has talent composite data for).
Columns: `season`, `school`, `talent`. That's genuinely all three raw
fields — no `Scaled_Talent`/z-score, that's a derived feature.

### `coaches` ← `cfbd_coaches(year)`
One row per coach per season (coaches with mid-season changes get one row
per school-stint). Whole-league pull, one call per year, `2004`–current.
Columns: `first_name`, `last_name`, `hire_date`, `school`, `season`, `games`,
`wins`, `losses`, `ties`, `preseason_rank`, `postseason_rank`, `srs`,
`sp_overall`, `sp_offense`, `sp_defense`. (Cumulative
games-coached/win-percentage across seasons is a windowed query on read, not
a stored column.)

### `team_rosters` ← `cfbd_team_roster(year)`
One row per player per season. Whole-league pull, one call per year (no
per-team looping), `2004`–current.
Columns: `athlete_id`, `first_name`, `last_name`, `team`, `weight`, `height`,
`jersey`, `year` (eligibility class — sometimes actually the season year,
a raw-data quirk, kept as-is), `position`, `home_city`, `home_state`,
`home_country`, `home_latitude`, `home_longitude`, `home_county_fips`,
`recruit_ids` (list of this player's recruiting-record ids, stored as JSON),
`headshot_url`.

### `recruiting_players` ← `cfbd_recruiting_player(year)`
One row per recruit per class year. Whole-league pull, one call per year,
`2000`–current.
Columns: `recruit_id` (API's `id`), `athlete_id` (joins to
`team_rosters.athlete_id`), `recruit_type`, `recruit_year`, `ranking`,
`name`, `school` (the recruit's *high school*, not the college), `committed_to`,
`position`, `height`, `weight`, `stars`, `rating`, `city`, `state_province`,
`country`, `hometown_latitude`, `hometown_longitude`, `hometown_fips_code`.

### `returning_production` ← `cfbd_player_returning(year)`
One row per team per season. Whole-league pull, one call per year,
`2014`–current (earliest year CFBD has this data for).
Columns: `season`, `team`, `conference`, `total_ppa`, `total_passing_ppa`,
`total_receiving_ppa`, `total_rushing_ppa`, `percent_ppa`,
`percent_passing_ppa`, `percent_receiving_ppa`, `percent_rushing_ppa`,
`usage_pct` (API field is named `usage`, renamed — reserved-adjacent),
`passing_usage`, `receiving_usage`, `rushing_usage`.

### `game_team_stats` ← `cfbd_game_team_stats(year, week)`
One row per team per game (two rows per game: one each side). Pulled per
season/week, `2013`–current — matched to `games`' start year since every row
has a foreign key back to `games(game_id)`.
~78 raw columns per side (`*_allowed` mirrors are the opponent's same stat):
`points`, `total_yards`, `net_passing_yards`, passing/rushing/return/penalty
counting stats, plus four compound "X-Y" or "MM:SS" string fields the API
returns as-is (`completion_attempts` "41-50", `third_down_eff` "7-16",
`fourth_down_eff` "1-3", `total_penalties_yards` "8-60",
`possession_time` "39:15"). Those raw strings are stored verbatim in
`*_raw` columns, and MySQL `GENERATED ALWAYS AS ... STORED` columns parse
them into usable numbers (e.g. `completions`/`attempted_passes` from
`completion_attempts_raw`) — see `schema.sql`.

### `plays` ← `cfbd_pbp_data(year, week, epa_wpa=TRUE)`
One row per play. Pulled per season/week, `2013`–current (also FK'd to
`games`). The raw response is ~363 columns of `cfbfastR`'s own precomputed
EPA/WPA/drive-context output; this table keeps everything with standalone
feature value and drops only `cfbfastR`'s internal `lag_*`/`lead_*` scaffolding
columns (which exist purely to chain its own calculations and have no
meaning as stored data):
- Play identity/context: `play_id`, `game_id`, `season`, `week`, `drive_id`,
  `pos_team`, `def_pos_team`, `offense_conference`, `defense_conference`,
  `play_type`, `play_text`, `period`, `half`, `clock_minutes`,
  `clock_seconds`, `down`, `distance`, `yard_line`, `yards_to_goal`,
  `yards_gained`
- EPA/WPA: `epa`, `ep_before`, `ep_after`, `ppa`, `wpa`, `wp_before`,
  `wp_after`, `home_wp_before`, `home_wp_after`, `away_wp_before`,
  `away_wp_after`
- Context flags: `success`, `rz_play`, `scoring_opp`, `middle_8`,
  `stuffed_run`, `turnover`, `downs_turnover`, `touchdown`, `safety`,
  `penalty_flag`, `penalty_text` (genuinely boolean despite the name)
- Per-play participant names + yardage: rusher, passer, receiver, sack
  (x2), interception, fumble (x3), punter/returner, FG kicker,
  kickoff/returner
- Drive detail: `drive_scoring`, `drive_pts`, `drive_result_detailed`,
  `drive_start_yards_to_goal`, `drive_end_yards_to_goal`, `drive_yards`,
  `drive_start_period`, `drive_end_period`, `new_drive_pts`

`plays` is partitioned by `season` (`PARTITION BY RANGE`), which is why it
has no foreign keys at all (InnoDB doesn't support FKs on partitioned
tables, as either parent or child) — its `game_id`/`pos_team`/`def_pos_team`
relationships to `games`/`teams` are documented, not enforced.

### `teams` (dimension table, not its own endpoint)
Populated opportunistically as other endpoints are pulled — `recruiting`,
`rosters`, `talent`, and `returning_production` only ever give a team name
string, so `school` is the natural join key everything else hangs off of.
`team_id` is filled in when `games`/`betting_lines` (the only two endpoints
carrying a numeric team id) supply it.

## Known scope decisions

- **Regular season only.** No `season_type` filter is applied beyond what
  the API returns by default, which means postseason/bowl games are not
  captured — matches the existing CSV pipeline's scope. Decided
  deliberately, not an oversight.
- **`games`/`betting_lines`/`game_team_stats`/`plays` start at 2013**, even
  though `coaches`/`team_rosters` (2004), `recruiting_players` (2000), and
  the technical earliest for `games` itself (1980) go back further. This
  matches the existing pipeline's actual training window (betting lines,
  the label source, don't exist before 2013) and avoids pulling
  `game_team_stats`/`plays` rows for seasons where no matching `games` row
  will ever exist to satisfy their foreign key.
- **The CFBD API key is currently hardcoded** (as a fallback default) in
  this script and in two existing R scripts
  (`Full_CFB_Game_Outcome_Historical.R`, `2025_Game_Update.R`). It's a real
  key committed to git history — rotating it is a standing recommendation,
  not yet acted on.
