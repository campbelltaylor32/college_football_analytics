require(tidyverse)
require(data.table)


### Aggregate All Data ###
talent <- fread('Data/CFB_Team_Talent_Data_2025.csv')
coaches <- fread('Data/Coaches_Winning_CFB_2025.csv')

### Coaching Stats Before the Year - No Lookahead Bias ##
coaches <- coaches %>%
  group_by(Name) %>%
  arrange(desc(year)) %>%
  mutate(
    year = lag(year)
  ) %>%
  ungroup() %>%
  na.omit()

talent_coach <- merge(talent,coaches, by = c('year', 'team'), all = TRUE) %>%
  distinct() %>%
  filter(year >= 2025) %>%
  distinct(year, team, .keep_all = TRUE)


### Returning Production 
returning <- fread('Data/Returning_Production_CFB_2025.csv')
names(returning)[1] <- "year"
returning <- returning %>% 
  select(-conference)


talent_coach_ret <- merge(talent_coach, returning, by = c('year', 'team'), all = TRUE)


### Load in Game Averages ###
game_stats <- fread('Data/Game_Stats_Averages_CFB_PBP_Added_2025_Week14.csv')


tot_pred <- merge(game_stats, talent_coach_ret, by = c('team', 'year'))
names(tot_pred)[6:173] <- paste0("prev_week_", names(tot_pred)[6:173])


lag_cols <- names(tot_pred)[
  grepl("^prev_week_", names(tot_pred)) |
    grepl("_avg_all$", names(tot_pred)) |
    grepl("_avg3$", names(tot_pred))
]
tot_pred <- as.data.frame(tot_pred)
tot_pred[is.na(tot_pred)] <- 0

### Lag Game Stats - Avoid Lookahead Bias
tot_pred_lagged <- tot_pred %>%
  group_by(team, year) %>%
  arrange(week, .by_group = TRUE) %>% 
  mutate(
    across(all_of(lag_cols), ~ dplyr::lag(.x, 1))
  ) %>%
  ungroup() %>%
  group_by(
    team
  ) %>% 
  slice_max(week,n = 1)


# Final Predictors per Team Heading into Week 
total_pred_lagged1 <- tot_pred_lagged %>%
  ### Get Full 3 Game Moving Average 
  filter(week >= as.numeric(4)) %>%
  select(
    -c(home_away, Name, year, week, game_id)
  )

home_team_dat <- total_pred_lagged1 %>%
  rename(
    home_team = team
  ) %>%
  rename_with(~ paste0("home_", .x), -c(home_team))




### Load in Game Results ### 
full_outcome <- fread(file ="Data/CFB_Gambling_Results_2025_14.csv")
full_outcome1 <- full_outcome %>% 
  select(
    game_id, home_team, away_team,season,
    week, neutral_site, conference_game,
    spread, home_favored, home_covered
  ) %>%
  mutate(
    spread = abs(spread)
  ) %>% 
  filter(week == 14)

home_full_outcome <- merge(full_outcome1, home_team_dat, by = c("home_team"))

away_team_dat <- total_pred_lagged1 %>%
  rename(
    away_team = team
  ) %>%
  rename_with(~ paste0("away_", .x), -c(away_team))

### Final Dataframe
total_outcome <- merge(home_full_outcome, away_team_dat, by = c("away_team"))
total_outcome <- total_outcome %>% 
  filter(is.na(home_covered))
week = 14

total_outcome %>%
  write.csv(
    file = paste0("Data/CFB_Pred_Week_",week,".csv"), row.names = FALSE
  )


total_outcome %>%
  group_by(season) %>%
  summarise(
    Num_Games = n_distinct(game_id)
  )
