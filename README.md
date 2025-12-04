This code started as a way to learn more about modeling and data manipulation in Python.
I added the FastAPI server code to get a bit more familiar with that as well. Associated front-end code is in separate repo of mine.

In its current state the code trains a RandomForest model and a XGBoost model (based on selection) to predict NBA player points scored in individual games.
The current flow is receive list of players, training seasons, and test seasons, gather necessary data for those players and seasons, calculate and add features to the model based on that data, run the test, and return the results.

NBA data is pulled mostly via nba_api code (https://github.com/swar/nba_api/). This works, but the source of data throttles requests from the s
ame IP early and often. Also, any requests coming from AWS will be blocked unequivocally. So, I added a sqlite cache in the code to store this data and prevent API calls when possible.

Historical Vegas data is used in the model, and you will need to download the csv file from Kaggle that the code uses: https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024 .

Current code / models' metrics won't knock socks off but are not bad, either.

NEXT STEPS for better model: injury data source and usage, lineup data source and usage, better minutes prediction, better usage rate prediction .....
