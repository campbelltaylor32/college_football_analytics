# College Football Gambling Model

This project builds a machine learning model to predict whether the home team covers the spread in college football games.  

## Data Sources
- [`cfbfastR`](https://github.com/sportsdataverse/cfbfastR): play-by-play and team statistics  
- [collegefootballdata.com API](https://collegefootballdata.com/): betting lines, game outcomes, and additional team stats  

## Timeframe
- **Training Data:** 2015–2022 seasons  
- **Testing Data:** 2023–2024 seasons  

The final model achieves 57% precision in predicting home team covers.  

---

## Repository Structure

### `Data/`
Single canonical folder for all CSV inputs and outputs: historical training data (`CFB_Gambling_Results.csv`, `CFB_Team_Talent_Data.csv`, `Coaches_Winning_CFB.csv`, `Game_Stats_Averages_CFB_PBP_Added.csv`, `Returning_Production_CFB.csv`, `CFB_Gambling_Predictors_Final(_PBP).csv`), the 2025-season weekly series (`*_2025_N.csv` / `*_2025_WeekN.csv`), and weekly prediction outputs (`CFB_Pred_Week_N.csv`). All scripts read/write here — there should be no loose data CSVs at the repo root.

### `R Scripts/`
- `Full_CFB_Game_Outcome_Historical.R` — pulls raw historical data, engineers moving averages & rolling statistics, defines the cover/push outcome logic, writes to `Data/`
- `Merge_Predictors_CFB_Historical.R` — combines historical predictors into `Data/CFB_Gambling_Predictors_Final_PBP.csv`
- `2025_Game_Update.R` — pulls/refreshes 2025-season gambling spreads and game stats weekly, writes to `Data/`
- `2025_Pred_Update.R` — updates predictors and produces the current week's predictions, writes to `Data/`

### `Python Scripts/`
- `CFB_Gambling_Model.ipynb` — trains classification models, tunes hyperparameters, saves the final trained model
- `Week_Predictions.ipynb` — applies the trained model to 2025 games, produces weekly cover predictions

### `cfb_spread_framework.py`
Standalone feature-selection & model-training framework (reads `Data/CFB_Gambling_Predictors_Final_PBP.csv`, writes to `model_artifacts/`). Outputs from k-sweep runs live in `model_artifacts_sweep/`.

### `Dashboard/`
Streamlit app (`app.py`, `utils.py`) for browsing weekly team stats; reads from `Data/`.

### `Model Information/`
Earlier (Aug/Sep 2025) trained model artifacts, kept for reference alongside the newer `model_artifacts/` outputs.

### `Modeling/`
Feature-search intermediate outputs (`output_feature_search/`).

---

## Performance
- Task: Binary classification (home team covers vs. does not cover)  
- Final Precision: 57% on predicting home covers

---

## Updates
Weekly predictions for the 2025 season will be posted on my Twitter: camtaylor_4

