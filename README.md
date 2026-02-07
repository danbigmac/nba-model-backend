This code started as a way to learn more about modeling and data manipulation in Python.
I added the FastAPI server code to get a bit more familiar with that as well. Associated front-end code is in separate repo of mine.

In its current state the code trains a RandomForest model and a XGBoost model (based on selection) to predict NBA player points scored in individual games.
The evaluation flow (`/predict`) takes a list of players plus train/test seasons, gathers data, builds features, and evaluates a pooled (league‑wide) model by holding out each player’s test season. The models predict residuals on top of Vegas baselines, and per‑player tuning uses a time‑based validation split (last ~15% of games).

The predict‑next flow (`/predict-next`) trains per‑player models from that player’s history, also with time‑based tuning. If a model performs worse than the Vegas baseline on validation, it falls back to baseline for that player/model. Predict‑next responses include `validated`, `used_baseline`, and a top‑level `best_models` summary.

NBA data is pulled mostly via nba_api code (https://github.com/swar/nba_api/). This works, but the source of data throttles requests from the same IP early and often. Note there is a sleep parameter in the fetch script - I discovered that sleeping about 2 seconds between requests prevents throttling. This is better than waiting for a connection to give up / bail. Also, any requests coming from AWS will be blocked unequivocally. So, I added a sqlite cache in the code to store this data and prevent API calls when possible. Current‑season logs refresh every 12 hours (past seasons stay cached for a year), and the current‑season top‑scorers list also refreshes every 12 hours.

Historical Vegas data is used in the model, and you will need to download the csv file from Kaggle that the code uses: https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024 .

For the current season (e.g., 2025-26), the Kaggle data will be missing. This repo includes a small ingestion script that can append daily odds to a live CSV so those games can be modeled with Vegas features. See `scripts/fetch_odds.py`, `scripts/fetch_odds_daily.sh`, and `scripts/cron_example.txt`, which write to `data/nba_2025-26_live.csv` and require `ODDS_API_KEY` to be set. Note: games without Vegas lines are dropped from the dataset, so you will only model dates that have been captured.

To keep current‑season logs fresh (and reduce throttling pain), see `scripts/refresh_top_scorers_logs.py` plus the venv‑aware wrapper `scripts/refresh_logs_daily.sh`. The cron example runs this multiple times daily with a short sleep between players.

Current code / models' metrics won't knock socks off but are not bad, either.

There is also a next-game prediction endpoint that produces an estimated PTS output without evaluation. Example payload:

```json
{
  "games": [
    {
      "player": "LeBron James",
      "opponent": "BOS",
      "game_date": "2026-01-30",
      "home_away": "home"
    }
  ],
  "train_seasons": ["2021-22", "2022-23", "2023-24"],
  "season": "2025-26",
  "models": ["RandomForest"]
}
```

POST to `/predict-next` and fetch results from `/next-results/{task_id}`. `home_away` can be omitted; the service will infer it from the Vegas line (home team listed first) when available. If `vegas_total`/`vegas_spread` are omitted, the service will try to look them up in the live odds CSV for the given matchup/date; if no Vegas line is found, you must provide `home_away`.

Predict‑next responses include per‑model `validated`/`used_baseline` flags and a `best_models` summary per player so you can quickly pick the preferred prediction.

NEXT STEPS for better model: injury data source and usage, lineup data source and usage, better minutes prediction, better usage rate prediction .....
