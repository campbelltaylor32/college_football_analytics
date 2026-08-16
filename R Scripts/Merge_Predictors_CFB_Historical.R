
require(tidyverse)
require(data.table)
setwd('Data/')

### Aggregate All Data ###
talent <- fread('CFB_Team_Talent_Data.csv')
coaches <- fread('Coaches_Winning_CFB.csv')

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
  filter(year >= 2015) %>%
  distinct(year, team, .keep_all = TRUE)


### Returning Production 
returning <- fread('Returning_Production_CFB.csv')
names(returning)[1] <- "year"
returning <- returning %>% 
  select(-conference)


talent_coach_ret <- merge(talent_coach, returning, by = c('year', 'team'), all = TRUE)


### Load in Game Averages ###
game_stats <- fread('Game_Stats_Averages_CFB_PBP_Added.csv')


tot_pred <- merge(game_stats, talent_coach_ret, by = c('team', 'year'), allow.cartesian = TRUE)
names(tot_pred)[6:173] <- paste0("prev_week_", names(tot_pred)[6:173])


lag_cols <- names(tot_pred)[
  grepl("^prev_week_", names(tot_pred)) |
    grepl("_avg_all$", names(tot_pred)) |
    grepl("_avg3$", names(tot_pred))
]

### Lag Game Stats - Avoid Lookahead Bias
tot_pred_lagged <- tot_pred %>%
  group_by(team, year) %>%
  arrange(week, .by_group = TRUE) %>% 
  mutate(
    across(all_of(lag_cols), ~ dplyr::lag(.x, 1))
  ) %>%
  ungroup()

osu_check <- tot_pred_lagged %>% 
  filter(team == "Ohio State")


# Final Predictors per Team Heading into Week 
total_pred_lagged1 <- tot_pred_lagged %>%
  ### Get Full 3 Game Moving Average 
  filter(week >= 3) %>%
  select(
    -c(home_away, Name, year, week)
  ) %>%
  na.omit()

home_team_dat <- total_pred_lagged1 %>%
  rename(
    home_team = team
  ) %>%
  rename_with(~ paste0("home_", .x), -c(home_team, game_id))




### Load in Game Results ### 
full_outcome <- fread(file ="CFB_Gambling_Results.csv")
names(full_outcome)
full_outcome1 <- full_outcome %>% 
  select(
    game_id, home_team, away_team,season,
    week, neutral_site, conference_game,
    spread, home_favored, home_covered
  ) %>%
  mutate(
    spread = abs(spread)
  )


home_full_outcome <- merge(full_outcome1, home_team_dat, by = c("home_team", "game_id"))


away_team_dat <- total_pred_lagged1 %>%
  rename(
    away_team = team
  ) %>%
  rename_with(~ paste0("away_", .x), -c(away_team, game_id))

### Final Dataframe
total_outcome <- merge(home_full_outcome, away_team_dat, by = c("away_team", "game_id"))

total_outcome %>%
  write.csv(
    file = "CFB_Gambling_Predictors_Final_PBP.csv", row.names = FALSE
  )


total_outcome %>%
  group_by(season) %>%
  summarise(
    Num_Games = n_distinct(game_id)
  )
