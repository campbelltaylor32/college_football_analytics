-- MySQL schema mirroring the raw cfbfastR / CollegeFootballData API responses,
-- verified against live calls to each endpoint on 2026-07-27 (see conversation).
-- Goal: cache raw pulls so weekly refreshes only fetch new weeks, instead of
-- re-hitting the API for history that's already been ingested.
--
-- Naming: every table's header comment names the cfbfastR function it mirrors.
-- Columns follow the API's own field names/types so this stays a faithful
-- cache, not a re-derivation -- rolling averages, EPA aggregates, cumulative
-- coaching records, blue-chip ratios etc. are feature-engineering output and
-- deliberately NOT modeled here; they get computed from these tables at
-- feature-build time (via views or the R/Python scripts), same as today.

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- Dimension: only games and betting_lines actually carry a numeric team id
-- (home_id/away_id, home_team_id/away_team_id). team_talent, coaches,
-- recruiting, rosters, and returning_production only ever give a team NAME
-- string -- so `school` is the natural key everything joins on; team_id is
-- populated opportunistically from games/betting_lines when known.
-- ---------------------------------------------------------------------------
CREATE TABLE teams (
  school   VARCHAR(100) NOT NULL PRIMARY KEY,
  team_id  INT UNSIGNED UNIQUE
);

-- ---------------------------------------------------------------------------
-- cfbd_game_info(year, week) -- 32 raw columns
-- ---------------------------------------------------------------------------
CREATE TABLE games (
  game_id             INT UNSIGNED PRIMARY KEY,
  season              SMALLINT NOT NULL,
  week                TINYINT NOT NULL,
  season_type         VARCHAR(20),
  start_date          DATETIME,
  start_time_tbd      BOOLEAN,
  completed           BOOLEAN,
  neutral_site        BOOLEAN,
  conference_game     BOOLEAN,
  attendance          INT UNSIGNED,
  venue_id            INT UNSIGNED,
  venue               VARCHAR(150),
  home_id             INT UNSIGNED,
  home_team           VARCHAR(100) NOT NULL,
  home_division       VARCHAR(10),
  home_conference     VARCHAR(100),
  home_points         SMALLINT,
  home_post_win_prob  DECIMAL(6,4),
  home_pregame_elo    SMALLINT,
  home_postgame_elo   SMALLINT,
  away_id             INT UNSIGNED,
  away_team           VARCHAR(100) NOT NULL,
  away_division       VARCHAR(10),
  away_conference     VARCHAR(100),
  away_points         SMALLINT,
  away_post_win_prob  DECIMAL(6,4),
  away_pregame_elo    SMALLINT,
  away_postgame_elo   SMALLINT,
  excitement_index    DECIMAL(6,3),
  highlights          VARCHAR(255),
  notes               VARCHAR(255),
  playoff             BOOLEAN,
  INDEX idx_season_week (season, week),
  INDEX idx_home_team (home_team),
  INDEX idx_away_team (away_team),
  CONSTRAINT fk_games_home_team FOREIGN KEY (home_team) REFERENCES teams(school),
  CONSTRAINT fk_games_away_team FOREIGN KEY (away_team) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_betting_lines(year, week) -- 23 raw columns, one row per (game, provider)
-- ---------------------------------------------------------------------------
CREATE TABLE betting_lines (
  id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
  game_id               INT UNSIGNED NOT NULL,
  season                SMALLINT,
  season_type           VARCHAR(20),
  week                  TINYINT,
  start_date            DATETIME,
  home_team_id          INT UNSIGNED,
  home_team             VARCHAR(100),
  home_conference       VARCHAR(100),
  home_classification   VARCHAR(10),
  home_score            SMALLINT,
  away_team_id          INT UNSIGNED,
  away_team             VARCHAR(100),
  away_conference       VARCHAR(100),
  away_classification   VARCHAR(10),
  away_score            SMALLINT,
  provider              VARCHAR(50) NOT NULL,
  spread                DECIMAL(5,1),
  formatted_spread      VARCHAR(30),
  spread_open           DECIMAL(5,1),
  over_under            DECIMAL(5,1),
  over_under_open       DECIMAL(5,1),
  home_moneyline        MEDIUMINT,
  away_moneyline        MEDIUMINT,
  UNIQUE KEY uq_game_provider (game_id, provider),
  CONSTRAINT fk_betting_lines_game FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- ---------------------------------------------------------------------------
-- cfbd_team_talent(year) -- genuinely just 3 raw columns; Scaled_Talent is
-- computed downstream (z-score across the pulled set), not an API field
-- ---------------------------------------------------------------------------
CREATE TABLE team_talent (
  season   SMALLINT NOT NULL,
  school   VARCHAR(100) NOT NULL,
  talent   DECIMAL(8,2),
  PRIMARY KEY (season, school),
  CONSTRAINT fk_team_talent_school FOREIGN KEY (school) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_coaches(year) -- 15 raw columns, one row per coach per season.
-- Total_Games_Coached / Winning_Percentage (current pipeline's derived
-- output) become a query: SUM(games) OVER (PARTITION BY first_name,
-- last_name ORDER BY season), computed on read instead of recomputed from
-- scratch on every weekly run.
-- ---------------------------------------------------------------------------
CREATE TABLE coaches (
  id                BIGINT AUTO_INCREMENT PRIMARY KEY,
  first_name        VARCHAR(80),
  last_name         VARCHAR(80),
  hire_date         DATE,
  school            VARCHAR(100) NOT NULL,
  season            SMALLINT NOT NULL,
  games             SMALLINT,
  wins              SMALLINT,
  losses            SMALLINT,
  ties              SMALLINT,
  preseason_rank    TINYINT,
  postseason_rank   TINYINT,
  srs               DECIMAL(6,2),
  sp_overall        DECIMAL(6,2),
  sp_offense        DECIMAL(6,2),
  sp_defense        DECIMAL(6,2),
  UNIQUE KEY uq_coach_school_season (first_name, last_name, school, season),
  CONSTRAINT fk_coaches_school FOREIGN KEY (school) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_team_roster(year) -- 17 raw columns. athlete_id is a STRING in the
-- API, not numeric. recruit_ids is a list per player (0+ recruiting record
-- ids) -- stored as JSON to keep that shape rather than force a fake 1:1.
-- ---------------------------------------------------------------------------
CREATE TABLE team_rosters (
  id                BIGINT AUTO_INCREMENT PRIMARY KEY,
  athlete_id        VARCHAR(20) NOT NULL,
  first_name        VARCHAR(80),
  last_name         VARCHAR(80),
  team              VARCHAR(100) NOT NULL,
  weight            SMALLINT,
  height            SMALLINT,
  jersey            SMALLINT,
  year              SMALLINT,         -- eligibility/class year, usually 1-5 but the
                                       -- raw API sometimes returns the season (e.g. 2024)
                                       -- here instead -- verified live, not cleaned up
  position          VARCHAR(10),
  home_city         VARCHAR(100),
  home_state        VARCHAR(10),
  home_country      VARCHAR(50),      -- full names ("Northern Ireland"), not codes
  home_latitude     DECIMAL(9,6),
  home_longitude    DECIMAL(9,6),
  home_county_fips  VARCHAR(10),
  recruit_ids       JSON,
  headshot_url      VARCHAR(255),
  season            SMALLINT NOT NULL,   -- added by the ingestion loop, not the API
  UNIQUE KEY uq_athlete_season (athlete_id, season),
  CONSTRAINT fk_team_rosters_team FOREIGN KEY (team) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_recruiting_player(year) -- 19 raw columns. `id` is the recruiting
-- record id (string); `athlete_id` (also string) is the join key back to
-- team_rosters.athlete_id -- verified live: matches correctly, ~74% of a
-- given team/year's recruits resolve to a later roster row (rest is normal
-- attrition: grayshirts, decommits, transfers).
-- ---------------------------------------------------------------------------
CREATE TABLE recruiting_players (
  recruit_id        VARCHAR(20) PRIMARY KEY,   -- API's `id`
  athlete_id        VARCHAR(20),
  recruit_type      VARCHAR(20),
  recruit_year      SMALLINT NOT NULL,
  ranking           INT,
  name              VARCHAR(150),
  school            VARCHAR(150),              -- high school, NOT the college
  committed_to      VARCHAR(100),
  position          VARCHAR(10),
  height            DECIMAL(7,2),     -- normally 60-84in but the raw API has
                                       -- at least one garbage outlier (5330,
                                       -- 2008) -- widened to fit, not cleaned
  weight            SMALLINT,
  stars             TINYINT,
  rating            DECIMAL(6,4),
  city              VARCHAR(100),
  state_province     VARCHAR(10),
  country           VARCHAR(10),
  hometown_latitude   DECIMAL(9,6),
  hometown_longitude  DECIMAL(9,6),
  hometown_fips_code  VARCHAR(10),
  INDEX idx_athlete (athlete_id),
  INDEX idx_year_team (recruit_year, committed_to),
  CONSTRAINT fk_recruiting_committed_to FOREIGN KEY (committed_to) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_player_returning(year) -- 15 raw columns, matches the persisted CSV
-- exactly (confirmed column-for-column against Data/Returning_Production_CFB.csv)
-- ---------------------------------------------------------------------------
CREATE TABLE returning_production (
  season                 SMALLINT NOT NULL,
  team                   VARCHAR(100) NOT NULL,
  conference             VARCHAR(100),
  total_ppa              DECIMAL(8,3),
  total_passing_ppa      DECIMAL(8,3),
  total_receiving_ppa    DECIMAL(8,3),
  total_rushing_ppa      DECIMAL(8,3),
  percent_ppa            DECIMAL(8,4),   -- ratio of a small sample, can swing
  percent_passing_ppa    DECIMAL(8,4),   -- well outside 0-1 (verified live:
  percent_receiving_ppa  DECIMAL(8,4),   -- percent_passing_ppa hits -129.5)
  percent_rushing_ppa    DECIMAL(8,4),
  usage_pct              DECIMAL(6,4),   -- API field is named `usage`; reserved-adjacent, renamed
  passing_usage          DECIMAL(6,4),
  receiving_usage        DECIMAL(6,4),
  rushing_usage          DECIMAL(6,4),
  PRIMARY KEY (season, team),
  CONSTRAINT fk_returning_production_team FOREIGN KEY (team) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_game_team_stats(year, week) -- 78 raw columns, ALL returned as
-- strings by the API, including compound "X-Y" fields. Generated columns
-- do the same parsing the R script currently does by hand (separate() +
-- as.numeric()), so both the raw string and the usable numeric value are
-- queryable without a separate ETL pass.
-- ---------------------------------------------------------------------------
CREATE TABLE game_team_stats (
  game_id                        INT UNSIGNED NOT NULL,
  school                         VARCHAR(100) NOT NULL,
  season                         SMALLINT NOT NULL,
  week                           TINYINT NOT NULL,
  conference                     VARCHAR(100),
  home_away                      VARCHAR(10),
  opponent                       VARCHAR(100),
  opponent_conference            VARCHAR(100),
  points                         SMALLINT,
  total_yards                    SMALLINT,
  net_passing_yards              SMALLINT,
  completion_attempts_raw        VARCHAR(15),   -- "41-50"
  completions        SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(completion_attempts_raw,'-',1) AS UNSIGNED)) STORED,
  attempted_passes    SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(completion_attempts_raw,'-',-1) AS UNSIGNED)) STORED,
  passing_tds                    SMALLINT,
  yards_per_pass                 DECIMAL(5,2),
  passes_intercepted             SMALLINT,
  interception_yards             SMALLINT,
  interception_tds               SMALLINT,
  rushing_attempts               SMALLINT,
  rushing_yards                  SMALLINT,
  rush_tds                       SMALLINT,
  yards_per_rush_attempt         DECIMAL(5,2),
  first_downs                    SMALLINT,
  third_down_eff_raw             VARCHAR(15),   -- "7-16"
  third_down_conversion SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(third_down_eff_raw,'-',1) AS UNSIGNED)) STORED,
  third_down_attempts   SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(third_down_eff_raw,'-',-1) AS UNSIGNED)) STORED,
  fourth_down_eff_raw             VARCHAR(15),   -- "1-3"
  fourth_down_conversion SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(fourth_down_eff_raw,'-',1) AS UNSIGNED)) STORED,
  fourth_down_attempts   SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(fourth_down_eff_raw,'-',-1) AS UNSIGNED)) STORED,
  punt_returns                    SMALLINT,
  punt_return_yards              SMALLINT,
  punt_return_tds                SMALLINT,
  kick_return_yards              SMALLINT,
  kick_return_tds                SMALLINT,
  kick_returns                    SMALLINT,
  kicking_points                  SMALLINT,
  fumbles_recovered              SMALLINT,
  fumbles_lost                    SMALLINT,
  total_fumbles                   SMALLINT,
  tackles                         SMALLINT,
  tackles_for_loss                SMALLINT,
  sacks                            SMALLINT,
  qb_hurries                      SMALLINT,
  interceptions                   SMALLINT,
  passes_deflected                SMALLINT,
  turnovers                        SMALLINT,
  defensive_tds                    SMALLINT,
  total_penalties_yards_raw       VARCHAR(15),   -- "8-60"
  total_penalties       SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(total_penalties_yards_raw,'-',1) AS UNSIGNED)) STORED,
  penalty_yards         SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(total_penalties_yards_raw,'-',-1) AS UNSIGNED)) STORED,
  possession_time_raw             VARCHAR(10),   -- "39:15"
  possession_time_minutes DECIMAL(6,3) GENERATED ALWAYS AS (
    CAST(SUBSTRING_INDEX(possession_time_raw,':',1) AS DECIMAL(6,3)) +
    CAST(SUBSTRING_INDEX(possession_time_raw,':',-1) AS DECIMAL(6,3)) / 60
  ) STORED,
  points_allowed                   SMALLINT,
  total_yards_allowed              SMALLINT,
  net_passing_yards_allowed        SMALLINT,
  completion_attempts_allowed_raw  VARCHAR(15),   -- "11-27"
  completions_against SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(completion_attempts_allowed_raw,'-',1) AS UNSIGNED)) STORED,
  completion_attempts_against SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(completion_attempts_allowed_raw,'-',-1) AS UNSIGNED)) STORED,
  passing_tds_allowed               SMALLINT,
  yards_per_pass_allowed            DECIMAL(5,2),
  passes_intercepted_allowed        SMALLINT,
  interception_yards_allowed        SMALLINT,
  interception_tds_allowed          SMALLINT,
  rushing_attempts_allowed          SMALLINT,
  rushing_yards_allowed             SMALLINT,
  rush_tds_allowed                  SMALLINT,
  yards_per_rush_attempt_allowed    DECIMAL(5,2),
  first_downs_allowed               SMALLINT,
  third_down_eff_allowed_raw        VARCHAR(15),   -- "2-12"
  third_down_conversion_allowed SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(third_down_eff_allowed_raw,'-',1) AS UNSIGNED)) STORED,
  third_down_attempts_allowed   SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(third_down_eff_allowed_raw,'-',-1) AS UNSIGNED)) STORED,
  fourth_down_eff_allowed_raw       VARCHAR(15),   -- "0-1"
  fourth_down_conversion_allowed SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(fourth_down_eff_allowed_raw,'-',1) AS UNSIGNED)) STORED,
  fourth_down_attempts_allowed   SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(fourth_down_eff_allowed_raw,'-',-1) AS UNSIGNED)) STORED,
  punt_returns_allowed              SMALLINT,
  punt_return_yards_allowed         SMALLINT,
  punt_return_tds_allowed           SMALLINT,
  kick_return_yards_allowed         SMALLINT,
  kick_return_tds_allowed           SMALLINT,
  kick_returns_allowed              SMALLINT,
  kicking_points_allowed            SMALLINT,
  fumbles_recovered_allowed         SMALLINT,
  fumbles_lost_allowed              SMALLINT,
  total_fumbles_allowed             SMALLINT,
  tackles_allowed                   SMALLINT,
  tackles_for_loss_allowed          SMALLINT,
  sacks_allowed                     SMALLINT,
  qb_hurries_allowed                SMALLINT,
  interceptions_allowed             SMALLINT,
  passes_deflected_allowed          SMALLINT,
  turnovers_allowed                 SMALLINT,
  defensive_tds_allowed             SMALLINT,
  penalties_allowed_yards_raw       VARCHAR(15),   -- "3-15"
  penalties_allowed    SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(penalties_allowed_yards_raw,'-',1) AS UNSIGNED)) STORED,
  penalty_yards_allowed SMALLINT GENERATED ALWAYS AS (CAST(SUBSTRING_INDEX(penalties_allowed_yards_raw,'-',-1) AS UNSIGNED)) STORED,
  possession_time_allowed_raw       VARCHAR(10),   -- "20:45"
  possession_time_allowed_minutes DECIMAL(6,3) GENERATED ALWAYS AS (
    CAST(SUBSTRING_INDEX(possession_time_allowed_raw,':',1) AS DECIMAL(6,3)) +
    CAST(SUBSTRING_INDEX(possession_time_allowed_raw,':',-1) AS DECIMAL(6,3)) / 60
  ) STORED,
  PRIMARY KEY (game_id, school),
  CONSTRAINT fk_game_team_stats_game FOREIGN KEY (game_id) REFERENCES games(game_id),
  CONSTRAINT fk_game_team_stats_school FOREIGN KEY (school) REFERENCES teams(school)
);

-- ---------------------------------------------------------------------------
-- cfbd_pbp_data(year, week, epa_wpa=TRUE) -- raw response is 363 columns of
-- cfbfastR's own pre-computed EPA/WPA/drive-context output (lag/lead
-- columns, decomposed WPA, garbage-time flags, etc.), NOT a bare plays
-- payload. This table keeps the offense/defense EPA aggregation inputs plus
-- WPA, ppa, play-level context flags, per-play participant names, and
-- drive-level detail -- everything with standalone feature value. Still
-- deliberately excludes cfbfastR's own internal lag_*/lead_* scaffolding
-- columns (dozens of them), which only exist to chain its own calculations
-- and have no meaning as stored data.
--
-- No FOREIGN KEY constraints here: InnoDB does not support foreign keys on
-- partitioned tables at all (neither as child nor parent), so game_id/
-- pos_team/def_pos_team are documented relationships only, not enforced.
-- ---------------------------------------------------------------------------
CREATE TABLE plays (
  play_id         BIGINT NOT NULL,   -- API's id_play; signed because cfbfastR
                                      -- inserts synthetic placeholder rows with
                                      -- small negative ids -- verified live
  game_id         INT UNSIGNED NOT NULL,      -- logically -> games(game_id), unenforced
  season          SMALLINT NOT NULL,
  week            TINYINT NOT NULL,           -- API field is `wk`; renamed for consistency
  drive_id        BIGINT,            -- also signed for the same reason
  pos_team        VARCHAR(100),               -- logically -> teams(school), unenforced
  def_pos_team    VARCHAR(100),               -- logically -> teams(school), unenforced
  offense_conference  VARCHAR(50),
  defense_conference  VARCHAR(50),
  play_type       VARCHAR(40),
  play_text       TEXT,
  period          TINYINT,
  half            TINYINT,
  clock_minutes   TINYINT,
  clock_seconds   TINYINT,
  down            TINYINT,
  distance        TINYINT,
  yard_line       SMALLINT,
  yards_to_goal   SMALLINT,
  yards_gained    SMALLINT,
  epa             DECIMAL(7,4),
  ep_before       DECIMAL(7,4),
  ep_after        DECIMAL(7,4),
  ppa             DECIMAL(7,4),
  wpa             DECIMAL(7,4),
  wp_before       DECIMAL(8,6),
  wp_after        DECIMAL(8,6),
  home_wp_before  DECIMAL(8,6),
  home_wp_after   DECIMAL(8,6),
  away_wp_before  DECIMAL(8,6),
  away_wp_after   DECIMAL(8,6),
  success         BOOLEAN,
  rz_play         BOOLEAN,
  scoring_opp     BOOLEAN,
  middle_8        BOOLEAN,          -- garbage-time-relevant score/period context flag
  stuffed_run     BOOLEAN,
  turnover        BOOLEAN,
  downs_turnover  BOOLEAN,
  touchdown       BOOLEAN,
  safety          BOOLEAN,
  penalty_flag    BOOLEAN,
  penalty_text    BOOLEAN,          -- genuinely boolean despite the name -- verified live
  rusher_player_name            VARCHAR(100),
  yds_rushed                    SMALLINT,
  passer_player_name             TEXT,   -- occasionally holds a full play
                                          -- description instead of a name
                                          -- (source data quirk, verified live)
  receiver_player_name           TEXT,   -- same quirk, verified live
  yds_receiving                  SMALLINT,
  sack_player_name                VARCHAR(100),
  sack_player_name2               VARCHAR(100),
  yds_sacked                      SMALLINT,
  interception_player_name        TEXT,  -- same quirk, verified live
  yds_int_return                   SMALLINT,
  fumble_player_name                VARCHAR(100),
  fumble_forced_player_name         VARCHAR(100),
  fumble_recovered_player_name      VARCHAR(100),
  yds_fumble_return                 SMALLINT,
  punter_player_name                VARCHAR(100),
  yds_punted                        SMALLINT,
  punt_returner_player_name          VARCHAR(100),
  yds_punt_return                    SMALLINT,
  fg_kicker_player_name              VARCHAR(100),
  yds_fg                             SMALLINT,
  kickoff_player_name                 VARCHAR(100),
  kickoff_returner_player_name        VARCHAR(100),
  drive_scoring   BOOLEAN,
  drive_pts       TINYINT,
  drive_result_detailed       VARCHAR(50),
  drive_start_yards_to_goal   SMALLINT,
  drive_end_yards_to_goal     SMALLINT,
  drive_yards                 SMALLINT,
  drive_start_period          TINYINT,
  drive_end_period            TINYINT,
  new_drive_pts                TINYINT,
  PRIMARY KEY (season, game_id, play_id)
)
PARTITION BY RANGE (season) (
  PARTITION p2015 VALUES LESS THAN (2016),
  PARTITION p2016 VALUES LESS THAN (2017),
  PARTITION p2017 VALUES LESS THAN (2018),
  PARTITION p2018 VALUES LESS THAN (2019),
  PARTITION p2019 VALUES LESS THAN (2020),
  PARTITION p2020 VALUES LESS THAN (2021),
  PARTITION p2021 VALUES LESS THAN (2022),
  PARTITION p2022 VALUES LESS THAN (2023),
  PARTITION p2023 VALUES LESS THAN (2024),
  PARTITION p2024 VALUES LESS THAN (2025),
  PARTITION p2025 VALUES LESS THAN (2026),
  PARTITION pmax  VALUES LESS THAN MAXVALUE
);
