library(cfbfastR)
library(tidyverse)
library(data.table)


### Load in Key
# Searches upward from the working directory (not commandArgs()'s --file= path -- that
# mangles paths with spaces, like "R Scripts/", in some invocation contexts).
find_env_file <- function() {
  dir <- getwd()
  for (i in 1:5) {
    candidate <- file.path(dir, ".env")
    if (file.exists(candidate)) return(candidate)
    dir <- dirname(dir)
  }
  NULL
}
env_file <- find_env_file()
if (!is.null(env_file)) readRenviron(env_file)
if (Sys.getenv("CFBD_API_KEY") == "") {
  stop("CFBD_API_KEY not found -- expected it in the repo root's .env (see .env.example). Run this script from the repo root.")
}

week_update <- 14

### Get Full Games ###
years_download <- 2025
weeks <- seq(1, week_update, 1)

tot_games <- data.frame()
tmp_games <- data.frame()

for (i in seq_along(years_download)) {
  print(paste0("Season: ", years_download[i], "..."))
  for (j in seq_along(weeks)) {
    print(paste0("Week ", weeks[j], "..."))
    tmp_games <- cfbd_game_info(year = years_download[i], week = weeks[j])
    tot_games <- rbind(tot_games, tmp_games)
  }
}

## Volume per Year - Covid Shortened 2020
tot_games %>%
  group_by(
    season
  ) %>%
  summarise(
    n_distinct(game_id)
  )

## Clean Full Games ##
tot_games <- tot_games %>%
  filter(completed == TRUE) %>%
  select(
    game_id, season, week, season_type, start_date,
    neutral_site, conference_game, venue_id,
    home_id, home_team, home_conference,
    home_points, away_id, away_team, away_conference,
    away_points
  ) %>%
  arrange(season, week)


### Load in Betting Data ###
tot_betting <- data.frame()
tmp_betting <- data.frame()

for (i in seq_along(years_download)) {
  print(paste0("Season: ", years_download[i], "..."))
  for (j in seq_along(weeks)) {
    print(paste0("Week ", weeks[j], "..."))
    tmp_betting <- cfbd_betting_lines(year = years_download[i], week = weeks[j])
    tmp_betting$week <- weeks[j]
    tot_betting <- rbind(tot_betting, tmp_betting)
  }
}

## Clean Betting ##
tot_betting <- tot_betting %>%
  select(
    game_id, home_team, away_team, spread, formatted_spread,
    provider, over_under, week
  )

### Pick Consensus Lines if Available
consensus_only <- tot_betting %>%
  filter(provider == "consensus") %>%
  select(game_id, home_team, away_team, spread, over_under, formatted_spread, week)

### If not - Average Out Spreads
fallback <- tot_betting %>%
  group_by(game_id, home_team, away_team, week) %>%
  summarise(
    spread = mean(spread, na.rm = TRUE),
    over_under = mean(over_under, na.rm = TRUE),
    formatted_spread = first(formatted_spread)
  ) %>%
  ungroup()

### Put back together
tot_betting_final <- consensus_only %>%
  bind_rows(
    fallback %>%
      anti_join(consensus_only, by = c("game_id", "home_team", "away_team"))
  )

### Should be Equal ###
nrow(tot_betting_final) == as.numeric(tot_betting %>% summarise(n_distinct(game_id)))


### Merge Together ###
fin_data <- merge(tot_games, tot_betting_final, by = c("game_id", "home_team", "away_team", "week"), all = TRUE)

fin_data <- fin_data %>%
  rowwise() %>%
  mutate(
    home_minus_away = home_points - away_points,
    home_favored = ifelse(
      grepl(home_team, formatted_spread, fixed = TRUE),
      1,
      0
    ),
    away_favored = ifelse(home_favored == 0, 1, 0)
  ) %>%
  ungroup()

fin_data <- fin_data %>%
  mutate(
    home_covered = ifelse(
      ## Home Team Favored & Point Differential Greater than Spread
      #### Spread in format (-3) for home team (3) for away team
      (home_favored == 1) & (home_minus_away > (-spread)) |
        ## Win Outright as Underdog
        (away_favored == 1) & (home_minus_away >= 0) |
        ## Stay within Number as Underdog
        (away_favored == 1) & (spread >= 0) & (home_minus_away < 0) & (
          home_minus_away > (-spread)
        ),
      1,
      0
    )
  )


### Check Distribution of Home Covered - Expect close to 50/50

### Write Gambling Results
fin_data %>%
  write.csv(file = paste0("Data/CFB_Gambling_Results_2025_", week_update, ".csv"), row.names = FALSE)


# fin_data <- fread("CFB_Gambling_Results.csv")



### Get Talent
# tot_talent <- data.frame()
# tmp_talent <- data.frame()
#
# for(i in seq_along(years_download)){
#   tmp_talent <- cfbd_team_talent(year = years_download[i])
#   if(nrow(tmp_talent) > 0){
#     tmp_talent$Scaled_Talent <- as.vector(scale(tmp_talent$talent))
#     tot_talent <- rbind(tot_talent, tmp_talent)
#   }
#   else{
#     print(paste0("No data found for ", years_download[i]))
#   }
#   print(i)
# }
#
# names(tot_talent)[2] <- "team"

### Get Recruits ###
# tot_recruits <- data.frame()
# tmp_recruits <- data.frame()
#
# years_download <- seq(2009,2025)
#
# for(i in seq_along(years_download)){
#   tmp_recruits <- cfbd_recruiting_player(year = years_download[i])
#   if(nrow(tmp_recruits) > 0){
#     tot_recruits <- rbind(tot_recruits, tmp_recruits)
#   }
#   else{
#     print(paste0("No data found for ", years_download[i]))
#   }
#   print(i)
# }
#
# ### Subset on Cleaner ###
# tot_recruits <- tot_recruits %>%
#   select(
#     id, athlete_id,name,position, year, committed_to,ranking ,
#     stars, rating
#   )
#
# names(tot_recruits)[5] <- "recruit_year"
#
# ### Get Full Rosters ###
# tot_roster <- data.frame()
# tmp_roster <- data.frame()
#
# years_download <- seq(2013,2025)
#
# for(i in seq_along(years_download)){
#   tmp_roster <- cfbd_team_roster(year = years_download[i])
#
#   if(nrow(tmp_roster) > 0){
#     tmp_roster$season <- years_download[i]
#     tot_roster <- rbind(tot_roster, tmp_roster)
#   }
#   else{
#     print(paste0("No data found for ", years_download[i]))
#   }
#   print(i)
# }
#
# tot_roster <- tot_roster %>%
#   select(
#     athlete_id, first_name,last_name, team, year, position, season
#   )
# names(tot_roster)[5] <- "class"
# names(tot_roster)[7] <- "year"
#
# ###
# all_roster_recruit <- merge(tot_roster, tot_recruits, by = c("athlete_id"), all.x = TRUE)
#
# blue_chip_ratio <- all_roster_recruit %>%
#   group_by(
#     team, year
#   ) %>%
#   summarise(
#     blue_chip_ratio = sum(stars >= 4, na.rm = TRUE) / n_distinct(athlete_id),
#     avg_player_rating = mean(rating, na.rm = TRUE)
#   ) %>%
#   ungroup()
#
# ### May need to Remove Avg Player Rating
# tot_talent_full <- merge(tot_talent, blue_chip_ratio, by = c("year", "team"), all.x = TRUE)
# tot_talent_full <- as.data.frame(tot_talent_full)
# tot_talent_full[is.na(tot_talent_full)] <- 0
#
# tot_talent_full %>%
#   write.csv(
#     file = "Data/CFB_Team_Talent_Data_2025.csv", row.names = FALSE
#   )


### Coaches ###
# tot_coaches <- data.frame()
# tmp_coaches <- data.frame()
#
# years_download <- seq(2004,2025)
#
# for(i in seq_along(years_download)){
#   tmp_coaches <- cfbd_coaches(year = years_download[i])
#
#   if(nrow(tmp_coaches) > 0){
#     tmp_coaches <- tmp_coaches %>%
#       mutate(
#         Name = paste0(first_name,' ',last_name)
#       ) %>%
#       select(Name,team = school, year, games, wins, losses)
#     tot_coaches <- rbind(tot_coaches, tmp_coaches)
#   }
#   else{
#     print(paste0("No data found for ", years_download[i]))
#   }
#   print(i)
# }
#
# tot_coaches_stat <- tot_coaches %>%
#   arrange(year) %>%
#   group_by(Name) %>%
#   mutate(
#     Total_Games_Coached = cumsum(games),
#     Total_Wins = cumsum(wins),
#     Winning_Percentage = Total_Wins / Total_Games_Coached
#   ) %>%
#   ungroup() %>%
#   select(
#     Name, year, , team, Total_Games_Coached,  Winning_Percentage
#   )
#
# tot_coaches_stat %>%
#   write.csv(file = "Data/Coaches_Winning_CFB_2025.csv", row.names = FALSE)
# ####


### Get Basic Game Stats ###
years_download <- 2025
weeks <- seq(1, 14)

tot_stats <- data.frame()
tmp_stats <- data.frame()

for (i in seq_along(years_download)) {
  print(paste0("Season: ", years_download[i], "..."))
  for (j in seq_along(weeks)) {
    print(paste0("Week ", weeks[j], "..."))
    tmp_stats <- cfbd_game_team_stats(year = years_download[i], week = weeks[j])
    tmp_stats$week <- weeks[j]
    tmp_stats$year <- years_download[i]
    tot_stats <- rbind(tot_stats, tmp_stats)
  }
}

tot_stats <- tot_stats %>%
  select(
    -c(conference, opponent, opponent_conference)
  ) %>%
  select(
    game_id,
    team = school, week, year, everything()
  )

tot_stats <- as.data.frame(tot_stats)
tot_stats[is.na(tot_stats)] <- 0

tot_stats <- tot_stats %>%
  mutate(
    week = as.character(week),
    year = as.character(year),
    game_id = as.character(game_id)
  )

tot_stats <- tot_stats %>%
  separate(
    third_down_eff,
    into = c("third_down_conversion", "third_down_attempts"),
    sep = "-"
  ) %>%
  separate(
    fourth_down_eff,
    into = c("fourth_down_conversion", "fourth_down_attempts"),
    sep = "-"
  ) %>%
  separate(
    total_penalties_yards,
    into = c("total_penalties", "penalty_yards"),
    sep = "-"
  ) %>%
  separate(
    completion_attempts,
    into = c("completions", "attempted_passes"),
    sep = "-"
  ) %>%
  separate(
    completion_attempts_allowed,
    into = c("completions_against", "completion_attempts_against"),
    sep = "-"
  ) %>%
  separate(
    third_down_eff_allowed,
    into = c("third_down_conversion_allowed", "third_down_attempts_allowed"),
    sep = "-"
  ) %>%
  separate(
    fourth_down_eff_allowed,
    into = c("fourth_down_conversion_allowed", "fourth_down_attempts_allowed"),
    sep = "-"
  ) %>%
  separate(
    total_penalties_yards_allowed,
    into = c("penalties_allowed", "penalty_yards_allowed"),
    sep = "-"
  ) %>%
  mutate(
    possession_time = as.numeric(sub("^(\\d+):(\\d+)$", "\\1", possession_time)) +
      as.numeric(sub("^(\\d+):(\\d+)$", "\\2", possession_time)) / 60,
    possession_time_allowed = as.numeric(sub("^(\\d+):(\\d+)$", "\\1", possession_time_allowed)) +
      as.numeric(sub("^(\\d+):(\\d+)$", "\\2", possession_time_allowed)) / 60
  )


tot_stats[, 6:ncol(tot_stats)] <- lapply(tot_stats[, 6:ncol(tot_stats)], as.numeric)

library(RcppRoll)
cummean_na <- function(x) {
  n <- cumsum(!is.na(x))
  s <- cumsum(replace(x, is.na(x), 0))
  out <- s / ifelse(n == 0, NA_real_, n)
  out
}

### Feature Engineer ###
tot_stats <- tot_stats %>%
  mutate(
    third_down_percentage_offense = third_down_conversion / third_down_attempts,
    fourth_down_percentage_offense = fourth_down_conversion / fourth_down_attempts,
    pressure_percentage = qb_hurries / completion_attempts_against,
    sack_percentage = sacks / completion_attempts_against,
    pressure_percentage_allowed = qb_hurries_allowed / attempted_passes,
    sack_percentage_allowed = sacks_allowed / attempted_passes,
    interception_rate_offense = passes_intercepted / attempted_passes,
    intercetpion_rate_defense = interceptions / completion_attempts_against,
    point_differential = points - points_allowed,
    possession_time_difference = possession_time - possession_time_allowed,
    turnover_margin = turnovers - turnovers_allowed,
    penalty_yard_margin = penalty_yards - penalty_yards_allowed,
    total_plays = rushing_attempts + attempted_passes,
    rush_percentage = rushing_attempts / total_plays,
    yards_per_play = total_yards / total_plays,
    total_plays_against = rushing_attempts_allowed + completion_attempts_against,
    rush_percentage_against = rushing_attempts_allowed / total_plays_against,
    yards_per_play_allowed = total_yards_allowed / total_plays_against
  )

tot_stats[is.na(tot_stats)] <- 0
### Add Advanced ###
# years_download <- seq(2015,2024)
# weeks <- seq(1,12)

tot_epa <- data.frame()
tmp_epa <- data.frame()

for (i in seq_along(years_download)) {
  print(paste0("Season: ", years_download[i], "..."))
  for (j in seq_along(weeks)) {
    print(paste0("Week ", weeks[j], "..."))
    tmp_epa <- cfbd_pbp_data(year = years_download[i], week = weeks[j], epa_wpa = TRUE)
    tmp_epa$week <- weeks[j]
    tmp_epa$year <- years_download[i]
    tot_epa <- rbind(tot_epa, tmp_epa)
  }
}

offense_epa <- tot_epa %>%
  filter(play_type %in% c(
    "Rush", "Sack", "Rushing Touchdown",
    "Passing Touchdown", "Interception Return",
    "Interception Return Touchdown", "Pass Incompletion",
    "Pass Reception"
  )) %>%
  group_by(team = pos_team, week, year) %>%
  summarise(
    Total_Offense_Drives = n_distinct(drive_id),
    Total_Offense_Plays = n_distinct(id_play),
    Offense_Total_Run_Plays = n_distinct(id_play[play_type %in%
      c("Rush", "Rushing Touchdown")]),
    Offense_Total_Pass_Plays = n_distinct(id_play[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )]),
    Offense_Pass_Rate = Offense_Total_Pass_Plays / Total_Offense_Plays,
    Offense_Run_Rate = Offense_Total_Run_Plays / Total_Offense_Plays,
    Offense_first_down_pass_rate = n_distinct(id_play[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      ) & down == 1]) / n_distinct(id_play[down == 1]),
    Offense_Avg_3rd_Down_Distance = mean(distance[down == 3]),


    ### Success Related ###
    Total_Offense_EPA = sum(EPA, na.rm = TRUE),
    Offense_EPA_per_Play = Total_Offense_EPA / Total_Offense_Plays,
    Total_Offense_Success = sum(success, na.rm = TRUE),
    Offense_Success_Rate = Total_Offense_Success / Total_Offense_Plays,
    Total_Offense_EPA_Run = sum(EPA[play_type %in%
      c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Offense_EPA_per_Run = Total_Offense_EPA_Run / Offense_Total_Run_Plays,
    Total_Offense_Run_Success = sum(success[play_type %in%
      c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Offense_Run_Success_Rate = Total_Offense_Run_Success / Offense_Total_Run_Plays,
    Total_Offense_EPA_Pass = sum(EPA[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )], na.rm = TRUE),
    Offense_EPA_per_Run = Total_Offense_EPA_Pass / Offense_Total_Pass_Plays,
    Total_Offense_Pass_Success = sum(success[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )], na.rm = TRUE),
    Offense_Pass_Success_Rate = Total_Offense_Pass_Success / Offense_Total_Pass_Plays,
    Total_Offense_Explosives = n_distinct(id_play[yards_gained >= 20], na.rm = TRUE),
    Total_Offense_Explosive_Rate = Total_Offense_Explosives / Total_Offense_Plays,
    Total_Offense_Run_Explosives = n_distinct(id_play[yards_gained >= 20 &
      play_type %in% c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Total_Offense_Run_Explosive_Rate = Total_Offense_Run_Explosives / Offense_Total_Run_Plays,
    Total_Offense_Pass_Explosives = n_distinct(id_play[yards_gained >= 20 &
      play_type %in% c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )], na.rm = TRUE),
    Total_Offense_Pass_Explosive_Rate = Total_Offense_Pass_Explosives / Offense_Total_Pass_Plays,
    Offense_First_Down_Success = sum(success[down == 1], na.rm = TRUE),
    Offense_First_Down_Success_Rate = Offense_First_Down_Success / n_distinct(id_play[down == 1], na.rm = TRUE),
    Offense_First_Down_Run_Success = sum(success[down == 1 & play_type %in% c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Offense_First_Down_Run_Success_Rate = Offense_First_Down_Run_Success / n_distinct(id_play[down == 1 & play_type %in% c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Offense_First_Down_Pass_Success = sum(success[down == 1 & play_type %in% c(
      "Sack", "Passing Touchdown", "Interception Return",
      "Interception Return Touchdown", "Pass Incompletion",
      "Pass Reception"
    )], na.rm = TRUE),
    Offense_First_Down_Pass_Success_Rate = Offense_First_Down_Pass_Success / n_distinct(id_play[down == 1 & play_type %in% c(
      "Sack", "Passing Touchdown", "Interception Return",
      "Interception Return Touchdown", "Pass Incompletion",
      "Pass Reception"
    )], na.rm = TRUE),
    Total_Offense_Scoring_Drives = n_distinct(drive_id[drive_scoring == 1]),
    Total_Offense_Touchdown_Drives = n_distinct(drive_id[drive_pts == 7]),
    Offense_Scoring_Drive_Percentage = Total_Offense_Scoring_Drives / Total_Offense_Drives,
    Offense_Touchdown_Drive_Percentage = Total_Offense_Touchdown_Drives / Total_Offense_Drives
  )

defense_epa <- tot_epa %>%
  filter(play_type %in% c(
    "Rush", "Sack", "Rushing Touchdown",
    "Passing Touchdown", "Interception Return",
    "Interception Return Touchdown", "Pass Incompletion",
    "Pass Reception"
  )) %>%
  group_by(team = def_pos_team, week, year) %>%
  summarise(
    Total_Defense_Drives = n_distinct(drive_id),
    Total_Defense_Plays = n_distinct(id_play),
    Defense_Total_Run_Plays = n_distinct(id_play[play_type %in%
      c("Rush", "Rushing Touchdown")]),
    Defense_Total_Pass_Plays = n_distinct(id_play[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )]),
    Defense_Pass_Rate = Defense_Total_Pass_Plays / Total_Defense_Plays,
    Defense_Run_Rate = Defense_Total_Run_Plays / Total_Defense_Plays,
    Defense_first_down_pass_rate = n_distinct(id_play[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      ) & down == 1]) / n_distinct(id_play[down == 1]),
    Defense_Avg_3rd_Down_Distance = mean(distance[down == 3]),


    ### Success Related ###
    Total_Defense_EPA = sum(EPA, na.rm = TRUE),
    Defense_EPA_per_Play = Total_Defense_EPA / Total_Defense_Plays,
    Total_Defense_Success = sum(success, na.rm = TRUE),
    Defense_Success_Rate = Total_Defense_Success / Total_Defense_Plays,
    Total_Defense_EPA_Run = sum(EPA[play_type %in%
      c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Defense_EPA_per_Run = Total_Defense_EPA_Run / Defense_Total_Run_Plays,
    Total_Defense_Run_Success = sum(success[play_type %in%
      c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Defense_Run_Success_Rate = Total_Defense_Run_Success / Defense_Total_Run_Plays,
    Total_Defense_EPA_Pass = sum(EPA[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )], na.rm = TRUE),
    Defense_EPA_per_Run = Total_Defense_EPA_Pass / Defense_Total_Pass_Plays,
    Total_Defense_Pass_Success = sum(success[play_type %in%
      c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )], na.rm = TRUE),
    Defense_Pass_Success_Rate = Total_Defense_Pass_Success / Defense_Total_Pass_Plays,
    Total_Defense_Explosives = n_distinct(id_play[yards_gained >= 20], na.rm = TRUE),
    Total_Defense_Explosive_Rate = Total_Defense_Explosives / Total_Defense_Plays,
    Total_Defense_Run_Explosives = n_distinct(id_play[yards_gained >= 20 &
      play_type %in% c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Total_Defense_Run_Explosive_Rate = Total_Defense_Run_Explosives / Defense_Total_Run_Plays,
    Total_Defense_Pass_Explosives = n_distinct(id_play[yards_gained >= 20 &
      play_type %in% c(
        "Sack", "Passing Touchdown", "Interception Return",
        "Interception Return Touchdown", "Pass Incompletion",
        "Pass Reception"
      )], na.rm = TRUE),
    Total_Defense_Pass_Explosive_Rate = Total_Defense_Pass_Explosives / Defense_Total_Pass_Plays,
    Defense_First_Down_Success = sum(success[down == 1], na.rm = TRUE),
    Defense_First_Down_Success_Rate = Defense_First_Down_Success / n_distinct(id_play[down == 1], na.rm = TRUE),
    Defense_First_Down_Run_Success = sum(success[down == 1 & play_type %in% c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Defense_First_Down_Run_Success_Rate = Defense_First_Down_Run_Success / n_distinct(id_play[down == 1 & play_type %in% c("Rush", "Rushing Touchdown")], na.rm = TRUE),
    Defense_First_Down_Pass_Success = sum(success[down == 1 & play_type %in% c(
      "Sack", "Passing Touchdown", "Interception Return",
      "Interception Return Touchdown", "Pass Incompletion",
      "Pass Reception"
    )], na.rm = TRUE),
    Defense_First_Down_Pass_Success_Rate = Defense_First_Down_Pass_Success / n_distinct(id_play[down == 1 & play_type %in% c(
      "Sack", "Passing Touchdown", "Interception Return",
      "Interception Return Touchdown", "Pass Incompletion",
      "Pass Reception"
    )], na.rm = TRUE),
    Total_Defense_Scoring_Drives = n_distinct(drive_id[drive_scoring == 1]),
    Total_Defense_Touchdown_Drives = n_distinct(drive_id[drive_pts == 7]),
    Defense_Scoring_Drive_Percentage = Total_Defense_Scoring_Drives / Total_Defense_Drives,
    Defense_Touchdown_Drive_Percentage = Total_Defense_Touchdown_Drives / Total_Defense_Drives
  )

team_epa <- merge(offense_epa, defense_epa, by = c("team", "week", "year"))


all_stats <- merge(tot_stats, team_epa, by = c("team", "week", "year"))
all_stats[is.na(all_stats)] <- 0

###
tot_stats1 <- all_stats %>%
  group_by(year, team) %>%
  arrange(as.numeric(week), .by_group = TRUE) %>%
  mutate(
    # choose numeric feature columns
    across(
      .cols = where(is.numeric) & !any_of(c("year", "week")),
      .fns = list(
        ### Rolling Average
        avg_all = ~ cummean_na(.x),
        ### 3 Game Average
        avg3    = ~ roll_mean(.x, n = 3, align = "right", fill = 0, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

tot_stats1[is.na(tot_stats1)]

tot_stats1 %>%
  arrange(week, year, team) %>%
  write.csv(
    file = "Data/Game_Stats_Averages_CFB_PBP_Added_2025_Week14.csv",
    row.names = FALSE
  )


### Returning Production ###
years_download <- 2025

tot_return <- data.frame()
tmp_return <- data.frame()

for (i in seq_along(years_download)) {
  print(paste0("Season: ", years_download[i], "..."))
  tmp_return <- cfbd_player_returning(years_download[i])
  tot_return <- rbind(tot_return, tmp_return)
  print(i)
}

tot_return %>%
  write.csv(
    file = "Data/Returning_Production_CFB_2025.csv",
    row.names = FALSE
  )
