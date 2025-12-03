# --- Imports ---
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, LeagueLeaders
from nba_api.stats.static import players, teams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from nbacache import CacheManager
from concurrent.futures import ThreadPoolExecutor, as_completed

def doRf(X_train, y_train, X_test, y_test, sample_weight=None):
    ''' Train Random Forest model and evaluate on test set. '''
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = rf.predict(X_test)

    print("Random Forest Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2:", r2_score(y_test, y_pred))

    return rf, y_pred

def doXgb(X_train, y_train, X_test, y_test, sample_weight=None):
    ''' Train XGBoost model and evaluate on test set. '''
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        min_child_weight=2
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = xgb.predict(X_test)

    print("\nXGBoost Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2:", r2_score(y_test, y_pred))

    return xgb, y_pred

def doRfGridSearchCV(X_train, y_train, rf_model):
    ''' Hyperparameter Tuning: Perform Grid Search CV for Random Forest model. '''
    rf_param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_grid = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        scoring='neg_mean_absolute_error',  # lower MAE is better
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    print("\nBest Random Forest Params:", rf_grid.best_params_)
    print("Best RF MAE:", -rf_grid.best_score_)

def doXgbGridSearchCV(X_train, y_train, xgb_model):
    ''' Hyperparameter Tuning: Perform Grid Search CV for XGBoost model. '''
    xgb_param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    xgb_grid.fit(X_train, y_train)

    print("\nBest XGBoost Params:", xgb_grid.best_params_)
    print("Best XGB MAE:", -xgb_grid.best_score_)

def rolling_cv_evaluate_single_target(model, X_sorted, y_sorted, 
                                      player_name=None, season=None,
                                      n_splits=5, model_name="Model"):
    """
    Rolling CV, but only evaluate on one player's season.
    
    Parameters
    ----------
    model : sklearn-like estimator
    X : pd.DataFrame
        Feature matrix - must contain 'Player' and 'Season' columns
    y : pd.Series
        Target vector
    player_name : str, optional
        If provided, only evaluate this player
    season : str, optional
        If provided, only evaluate this season
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores, r2_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
        X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
        y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]

        # Filter test set down to only target player+season
        mask = np.ones(len(y_test), dtype=bool)
        if player_name:
            mask &= (X_test["Player"] == player_name)
        if season:
            mask &= (X_test["Season"] == season)

        # If no rows match, skip this fold
        if mask.sum() == 0:
            continue

        X_test_filtered = X_test[mask]
        y_test_filtered = y_test[mask]

        # Drop Player and Season columns for modeling
        X_train = X_train.drop(columns=['Player', 'Season'])
        X_test_filtered = X_test_filtered.drop(columns=['Player', 'Season'])

        # Fit on *all training data* (not filtered)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_filtered)

        mae = mean_absolute_error(y_test_filtered, y_pred)
        r2 = r2_score(y_test_filtered, y_pred)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(f"{model_name} | Fold {fold+1}: MAE={mae:.2f}, R^2={r2:.2f} "
              f"({len(y_test_filtered)} games for {player_name}, {season})")

    print(f"\n{model_name} Rolling CV Results for {player_name}, {season}:")
    print(f"Avg MAE: {np.mean(mae_scores):.2f}")
    print(f"Avg R^2: {np.mean(r2_scores):.2f}")
    
    return mae_scores, r2_scores

# --- Helper: Rolling Cross-Validation Evaluation ---
def rolling_cv_evaluate(model, X, y, n_splits=5, model_name="Model"):
    """
    Perform rolling (time-series) cross-validation and evaluate performance.
    
    Parameters
    ----------
    model : sklearn-like estimator
        Must implement fit() and predict()
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    n_splits : int
        Number of rolling folds
    model_name : str
        For printing results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores, r2_scores = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        print(f"{model_name} | Fold {fold+1}: MAE={mae:.2f}, R^2={r2:.2f}")
    
    print(f"\n{model_name} Rolling CV Results:")
    print(f"Avg MAE: {np.mean(mae_scores):.2f}")
    print(f"Avg R^2: {np.mean(r2_scores):.2f}")
    
    return mae_scores, r2_scores

def get_all_players(cache: CacheManager):
    ''' Return all NBA players, using cache if available. '''
    all_players, last_updated = cache.load_players()
    if last_updated is None:
        last_updated = 0
    if all_players is None or (time.time() - last_updated) > 31536000:  # 1 year cache
        all_players = players.get_players()
        cache.save_players(all_players)
    return all_players

# --- Helper: Get Player ID ---
# def get_player_id(name: str) -> int:
#     all_players = get_all_players()
#     player = [p for p in all_players if p['full_name'].lower() == name.lower()]
#     if player:
#         return player[0]['id']
#     else:
#         raise ValueError(f"Player '{name}' not found.")

def bucket_rest(x):
    ''' Bucket rest days into categories. '''
    if x == 0:
        return 0  # season opener
    elif x == 1:
        return 1  # back-to-back
    elif x == 2:
        return 2  # normal rest
    else:
        return 3  # extended rest

def add_multiwindow_streaks(df, stat_col, windows=[3,5,10], threshold=0.0, suffix=None):
    """
    Add multi-window streak features for a given stat column.
    
    Features added:
      - rolling average over last N games
      - hot/cold binary flags relative to season avg
      - deviation (delta) from season avg
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain ['Player','Season', stat_col]
    stat_col : str
        Column to base streaks on (e.g., 'PTS', 'AST')
    windows : list of int
        Rolling window sizes (default = [3,5,10])
    threshold : float
        Margin above/below season avg to define hot/cold
    suffix : str
        Optional suffix for new feature names (default=stat_col)
    """
    if suffix is None:
        suffix = stat_col
    
    # Season average
    df[f'{suffix}_season_avg'] = df.groupby(['Player','Season'])[stat_col].transform('mean')
    
    for w in windows:
        # Rolling mean
        df[f'{suffix}_last{w}'] = df.groupby(['Player','Season'])[stat_col] \
                                   .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        
        # Deviation from season avg
        df[f'{suffix}_delta{w}'] = (df[f'{suffix}_last{w}'] - df[f'{suffix}_season_avg']).astype(float)
        
        # Hot streak flag
        df[f'{suffix}_is_hot{w}'] = (df[f'{suffix}_delta{w}'] > threshold).astype(int)
        
        # Cold streak flag
        df[f'{suffix}_is_cold{w}'] = (df[f'{suffix}_delta{w}'] < -threshold).astype(int)
    
    return df

def add_ewm_features(df: pd.DataFrame,
                     stat_col: str,
                     spans = [3, 5, 10],
                     suffix: str | None = None) -> pd.DataFrame:
    """
    Add exponentially weighted moving averages for `stat_col`
    within each Player+Season.
    """
    if suffix is None:
        suffix = stat_col

    for span in spans:
        col_name = f"{suffix}_ewm{span}"
        df[col_name] = (
            df
            .groupby(["Player", "Season"])[stat_col]
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())
        )
    return df

def add_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add usage-related features. Approximates USG as:
        (FGA + 0.44*FG3A + TOV) / MIN
    (rough proxy when FTA is not available)
    """
    # Avoid division by zero
    df["MIN_safe"] = df["MIN"].replace(0, np.nan)

    df["USG_approx"] = (
        (df["FGA"] + 0.44 * df["FG3A"] + df["TOV"]) / df["MIN_safe"]
    )

    # Fill any NaNs (e.g., weird 0-min games)
    df["USG_approx"] = df["USG_approx"].fillna(df["USG_approx"].median())

    # Rolling + EWM usage
    df["USG_rolling5"] = (
        df.groupby(["Player", "Season"])["USG_approx"]
          .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

    df["USG_ewm5"] = (
        df.groupby(["Player", "Season"])["USG_approx"]
          .transform(lambda x: x.ewm(span=5, adjust=False).mean())
    )

    df.drop(columns=["MIN_safe"], inplace=True)

    return df

def add_opponent_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for how a player performs vs each Opponent.
    """
    df["vsOpp_PTS_avg"] = (
        df.groupby(["Player", "Opponent"])["PTS"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=[0,1], drop=True)
    )

    df["vsOpp_PTS_last5"] = (
        df.groupby(["Player", "Opponent"])["PTS"]
        .apply(lambda s: s.shift().rolling(5, min_periods=1).mean())
        .reset_index(level=[0,1], drop=True)
    )

    return df

def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rest-based flags:
      - is_b2b (back-to-back)
      - is_long_rest (3+ days)
    """
    df["is_b2b"] = (df["RestDays"] == 1).astype(int)
    df["is_long_rest"] = (df["RestDays"] >= 3).astype(int)

    # Rolling PTS after long rest
    df["PTS_after_long_rest_rolling3"] = (
        df
        .assign(PTS_after_long=np.where(df["is_long_rest"] == 1, df["PTS"], np.nan))
        .groupby(["Player", "Season"])["PTS_after_long"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    return df

def add_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-player home/away scoring averages and a simple
    'home boost' feature.
    """
    # HomeAway: 1 = home, 0 = away (from your current code)
    home_map = (
        df[df["HomeAway"] == 1]
        .groupby("Player")["PTS"]
        .mean()
    )
    away_map = (
        df[df["HomeAway"] == 0]
        .groupby("Player")["PTS"]
        .mean()
    )

    df["PTS_home_avg"] = df["Player"].map(home_map)
    df["PTS_away_avg"] = df["Player"].map(away_map)

    # Difference in home vs away scoring for this player
    df["PTS_home_boost"] = df["PTS_home_avg"] - df["PTS_away_avg"]

    # For each row, "expected PTS from home/away split"
    df["PTS_homeaway_expected"] = np.where(
        df["HomeAway"] == 1,
        df["PTS_home_avg"],
        df["PTS_away_avg"],
    )

    return df

def add_season_variability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season-level variability metrics for PTS:
      - PTS_std_season: per-player, per-season standard deviation of PTS
      - PTS_iqr_season: interquartile range (Q3 - Q1) of PTS
    """
    # Std dev
    df["PTS_std_season"] = df.groupby(["Player", "Season"])["PTS"].transform("std")

    # IQR
    def iqr(series):
        return series.quantile(0.75) - series.quantile(0.25)

    df["PTS_iqr_season"] = df.groupby(["Player", "Season"])["PTS"].transform(iqr)

    # Fill potential NaNs (e.g., very short seasons) with median
    for col in ["PTS_std_season", "PTS_iqr_season"]:
        df[col] = df[col].fillna(df[col].median())

    return df

def print_feature_importances(model, feature_cols, top_n=20, player_name="(unknown)"):
    """
    Pretty-print the top N feature importances for a fitted RandomForest model.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    pairs = list(zip(feature_cols, importances))

    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} Feature Importances for {player_name}:")
    for feat, val in sorted_pairs[:top_n]:
        print(f"  {feat:30s} {val:.4f}")

    return sorted_pairs

def get_player_season_log(cache: CacheManager, player: dict, season: str):
    ''' Get a player's game logs for a given season, using cache if available. '''
    pid = player['id']
    # TODO: should all this work be done, then saved in cache?
    #.      --> or should we save raw logs, then process each time? as it is now...
    df, last_updated = cache.load_player_logs(pid, season)
    if last_updated is None:
        last_updated = 0
    if df is None or (time.time() - last_updated) > 31536000:  # 1 year cache
        print(f"[API CALL] Fetching logs for {player['full_name']} ({season})")
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season)
        df = gamelog.get_data_frames()[0]
        cache.save_player_logs(pid, season, df)
    else:
        print(f"[CACHE HIT] Logs for {player['full_name']} ({season})")
    return df

def get_team_stats(cache: CacheManager, season: str):
    ''' Get team stats for a given season, using cache if available. '''
    df, last_updated = cache.load_team_stats(season)
    if last_updated is None:
        last_updated = 0
    if df is None or (time.time() - last_updated) > 31536000:  # 1 year cache
        # Fetch from NBA API
        print(f"[API CALL] Fetching team stats for {season}")
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        # Save to cache
        cache.save_team_stats(season, df)
    else:
        print(f"[CACHE HIT] Team stats for {season}")
    return df

def get_teams(cache: CacheManager):
    ''' Get all NBA teams, using cache if available. '''
    teams_info, last_updated = cache.load_teams()
    if last_updated is None:
        last_updated = 0
    if teams_info is None or (time.time() - last_updated) > 31536000:  # 1 year cache
        # Fetch from NBA API
        print(f"[API CALL] Fetching teams.")
        teams_info = teams.get_teams()
        # Save to cache
        cache.save_teams(teams_info)
    else:
        print(f"[CACHE HIT] Teams")
    return teams_info

def get_top_scorers(cache: CacheManager, season: str, top_n: int = 100, per_mode: str = "PerGame"):
    """
    Return top_n players by scoring (PTS) for a given NBA season. Try cache first.
    per_mode can be 'PerGame', 'Totals', 'Per48', etc.
    """
    # try the cache first
    df, last_updated = cache.load_top_scorers(season)
    if last_updated is None:
        last_updated = 0
    if df is None or (time.time() - last_updated) > 31536000:  # 1 year cache
        print(f"[API CALL] Fetching top scorers for {season}")
        leaders = LeagueLeaders(
            season=season,
            stat_category_abbreviation="PTS",
            per_mode48=per_mode
        )
        df = leaders.get_data_frames()[0]
        df = df[["PLAYER_ID", "PTS", "RANK"]]  # keep relevant
        # Save to cache
        cache.save_top_scorers(season, df)
    # Sort by player rank ascending (i.e. 1,2,3... not 100,99,98...))
    # NOTE KeyError was hitting here because only fields are saved (rather than full raw data) and we save them lowercase
    df_sorted = df.sort_values("rank", ascending=True).reset_index(drop=True)
    # Return top N players
    return df_sorted.head(top_n)

def get_vegas_data():
    '''
    Process csv containing historical Vegas data, return data frame.
    '''
    vegas = pd.read_csv("data/nba_2008-2025.csv")
    # Basic cleanup
    vegas["date"] = pd.to_datetime(vegas["date"])
    # Uppercase team abbreviations
    vegas["home"] = vegas["home"].str.upper()
    vegas["away"] = vegas["away"].str.upper()
    # Fix team abbreviations that differ from NBA API
    TEAM_FIX = {
        "NY": "NYK",
        "GS": "GSW",
        "SA": "SAS",
        "NO": "NOP",
        "UTAH": "UTA",
        "PHO": "PHX",
        "WSH": "WAS",
        "CHO": "CHA",
    }
    vegas["home"] = vegas["home"].replace(TEAM_FIX)
    vegas["away"] = vegas["away"].replace(TEAM_FIX)
    # Keep only columns we need
    vegas_small = vegas[["date", "home", "away", "total", "spread"]].copy()
    for col in ["total", "spread"]:
        vegas_small[col] = pd.to_numeric(vegas_small[col], errors="coerce")

    return vegas_small

def process_player_season(player: dict, season: str, team_stats: pd.DataFrame, cache: CacheManager) -> pd.DataFrame:
    """
    Fetch and process a single player's season log into a DataFrame.
    """
    pname = player['full_name']
    # Fetch player game logs for this season
    df = get_player_season_log(cache, player, season)

    # Sort and add to data
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')

    # Keep useful columns
    df = df[['GAME_DATE', 'MATCHUP', 'PTS', 'AST', 'REB', 'TOV', 'FGA', 'FG3M', 'FG3A', 'MIN']]
    
    # Convert MIN to numeric minutes
    df['MIN'] = (
        df['MIN']
        .astype(str)
        .str.split(':')
        .apply(lambda x: int(x[0]) + int(x[1]) / 60 if len(x) == 2 else int(x[0]))
    )

    # --- New Feature: Rest days ---
    df['RestDays'] = df['GAME_DATE'].diff().dt.days.fillna(0)
    df['RestBucket'] = df['RestDays'].apply(bucket_rest)

    # --- New features: Home/Away + Opponent ---
    df['HomeAway'] = df['MATCHUP'].str.contains("vs").astype(int)
    df['Opponent'] = df['MATCHUP'].str[-3:]
    df["Team"] = df["MATCHUP"].str[:3]

    def parse_matchup(m):
        parts = m.split()
        team1, sep, team2 = parts
        if sep == "@":
            return team2, team1  # home, away
        else: 
            return team1, team2  # home, away

    # HomeTeam / AwayTeam for the game
    df["HomeTeam"], df["AwayTeam"] = zip(
        *df["MATCHUP"].apply(parse_matchup)
    )

    # --- Merge opponent stats ---
    df = df.merge(
        team_stats,
        left_on='Opponent',
        right_on='TEAM_ABBREVIATION',
        how='left'
    )
    df.rename(columns={
        'DEF_RATING': 'Opponent_DEF_RATING',
        'PACE': 'Opponent_PACE'
    }, inplace=True)
    df = df.drop(columns=['TEAM_ABBREVIATION'])

    # Rolling averages (last 5 games)
    for col in ['PTS', 'AST', 'REB', 'TOV', 'FGA', 'FG3A', 'FG3M', 'MIN']:
        df[f'{col}_rolling'] = df[col].rolling(window=5, min_periods=1).mean()

    df = df.dropna().reset_index(drop=True)

    # Target: next game points
    df['PTS_next'] = df['PTS'].shift(-1)
    df = df.dropna()

    # Add identifiers
    df['Player'] = pname
    df['Season'] = season

    return df

def features_and_sort(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add features and sort the data.
    Note we modify the provided DataFrame.
    """
    # Opponent defensive strength rank within each season (0..1, higher = tougher defense)
    # Are we sure this is right? Lower defensive rating = better defense
    data["Opp_DEF_RANK"] = data.groupby("Season")["Opponent_DEF_RATING"] \
        .transform(lambda x: x.rank(pct=True, ascending=False))

    # --- Add streak features for points, assists, rebounds
    data = add_multiwindow_streaks(df=data, stat_col='PTS', windows=[3,5,10], threshold=2)
    data = add_multiwindow_streaks(df=data, stat_col='AST', windows=[3,5], threshold=1.5)
    data = add_multiwindow_streaks(df=data, stat_col='REB', windows=[5], threshold=1.5)
    # --- Residual target: PTS_next - PTS_season_avg ---
    # (keep PTS_next and PTS_season_avg for evaluation, but don't feed PTS_season_avg as input)
    #data["PTS_residual"] = data["PTS_next"] - data["PTS_season_avg"]

    # --- Add EWM features ---
    data = add_ewm_features(data, stat_col="PTS", spans=[3, 5, 10])
    data = add_ewm_features(data, stat_col="FGA", spans=[5])
    data = add_ewm_features(data, stat_col="FG3A", spans=[5])
    data = add_ewm_features(data, stat_col="MIN", spans=[5])

    # === Improved minutes projection ===
    data["MIN_trend"] = (
        0.50 * data["MIN_ewm5"] +
        0.30 * data["MIN_rolling"] +
        0.20 * data.groupby(["Player","Season"])["MIN"].transform("mean")
    )

    # FGA rate per minute
    data["FGA_rate"] = data["FGA_ewm5"] / (data["MIN_ewm5"] + 1e-5)
    # Smooth it
    data["FGA_rate"] = data.groupby(["Player","Season"])["FGA_rate"] \
                                        .transform(lambda x: x.ewm(span=8, adjust=False).mean())
    
    league_avg_def = data.groupby("Season")["Opponent_DEF_RATING"].transform("mean")
    data["DefMultiplier"] = league_avg_def / data["Opponent_DEF_RATING"]
    data["DefMultiplier"] = data["DefMultiplier"].clip(0.85, 1.15)

    # --- Add usage features ---
    data = add_usage_features(data)

    # --- Add opponent history features ---
    data = add_opponent_history_features(data)

    # --- Add rest features ---
    data = add_rest_features(data)

    # --- Add home/away features ---
    data = add_home_away_features(data)

    # --- Add season variability features ---
    data = add_season_variability_features(data)

    print("Dataset shape:", data.shape)
    print(data[['Player','Season']].value_counts())

    # --- 4. Sorting and index features ---
    data_sorted = data.sort_values('GAME_DATE')

    # Season ordering
    season_order = {s: i for i, s in enumerate(sorted(data_sorted['Season'].unique()))}
    data_sorted['SeasonIndex'] = data_sorted['Season'].map(season_order)

    # Game index within season
    data_sorted['GameIndex'] = data_sorted.groupby(['Player', 'Season']).cumcount() + 1
    data_sorted['GameIndex_norm'] = (
        data_sorted['GameIndex'] /
        data_sorted.groupby(['Player', 'Season'])['GameIndex'].transform('max')
    )

    # Player ID for pooled model
    data_sorted['PlayerID'] = data_sorted['Player'].astype('category').cat.codes

    # DaysAgo for recency weighting
    # Not used as a feature, so the global max_date is sufficient
    max_date = data_sorted["GAME_DATE"].max()
    data_sorted["DaysAgo"] = (max_date - data_sorted["GAME_DATE"]).dt.days

    return data_sorted

def do_work(players_of_interest, train_seasons, test_season, models, cache) -> List[Dict[str, Any]]:
    ''' Main work function to fetch data, process features, train and evaluate models. '''

    results = []
    print(f"Players: {players_of_interest}")

    # --- Load Vegas data ---
    vegas_small = get_vegas_data()

    # --- Team lookup dictionaries ---
    teams_info = get_teams(cache)
    team_id_to_abbr = {t['id']: t['abbreviation'] for t in teams_info}
    team_abbr_to_id = {t['abbreviation']: t['id'] for t in teams_info}

    # --- Get top scorers for 2024-25 season ---
    top100 = get_top_scorers(cache, "2024-25", top_n=200, per_mode="PerGame")
    print(top100[["player_id", "pts", "rank"]])

    # --- Grab all player data ---
    all_players = get_all_players(cache)

    all_seasons = train_seasons + [test_season]

    # --- Filter the all player list to just top 100 scorers ---
    top_ids = set(top100["player_id"].tolist())
    filtered_players = [p for p in all_players if p["id"] in top_ids]

    all_data = []

    # --- Process each season, getting data for each player ---
    for season in all_seasons:
        # --- Get team stats for this season, from cache or API ---
        team_stats = get_team_stats(cache, season)

        # Add TEAM_ABBREVIATION for merging
        team_stats['TEAM_ABBREVIATION'] = team_stats['TEAM_ID'].map(team_id_to_abbr)
        team_stats = team_stats[['TEAM_ABBREVIATION', 'DEF_RATING', 'PACE']]

        # --- Player loops ---
        for player in filtered_players:
            try:
                df_player_season = process_player_season(player, season, team_stats, cache)
                all_data.append(df_player_season)
            except Exception as e:
                print(f"Error processing {player['full_name']} ({season}): {e}")
                continue

    # --- Combine all players/seasons ---
    data = pd.concat(all_data, ignore_index=True)

    # --- Add features and sort the data ---
    data_sorted = features_and_sort(data)

    # --- Filter vegas_small to only dates in our data ---
    min_date = data_sorted["GAME_DATE"].min()
    max_date = data_sorted["GAME_DATE"].max()
    vegas_small = vegas_small[(vegas_small["date"] >= min_date) &
                            (vegas_small["date"] <= max_date)]

    # Merge our data with Vegas data on the exact matchup
    data_sorted = data_sorted.merge(
        vegas_small,
        how="left",
        left_on=["GAME_DATE", "Team", "Opponent"],
        right_on=["date", "home", "away"]
    )
    # Second merge for reversed home/away (our team might be the away team)
    data_sorted = data_sorted.merge(
        vegas_small,
        how="left",
        left_on=["GAME_DATE", "Opponent", "Team"],
        right_on=["date", "home", "away"],
        suffixes=("", "_rev")
    )
    # Pick which one matched
    data_sorted["VegasTotal"] = data_sorted["total"].fillna(data_sorted["total_rev"])
    data_sorted["VegasSpread"] = data_sorted["spread"].fillna(data_sorted["spread_rev"])

    # Rows where we did NOT find a Vegas line
    missing_vegas = data_sorted[data_sorted["VegasTotal"].isna()].copy()
    if not missing_vegas.empty:
        print("Total rows with no Vegas match:", len(missing_vegas))
        # See some examples
        print(
            missing_vegas[["GAME_DATE", "Team", "Opponent"]]
            .drop_duplicates()
            .head(20)
        )
        example_date = missing_vegas["GAME_DATE"].iloc[0]
        print("Vegas rows on this date:")
        print(
            vegas_small[vegas_small["date"] == example_date][["date", "home", "away", "total", "spread"]]
        )
        nba_teams = set(data_sorted["Team"].unique()) | set(data_sorted["Opponent"].unique())
        vegas_teams = set(vegas_small["home"].unique()) | set(vegas_small["away"].unique())
        print("Teams in NBA logs but not in Vegas:")
        print(nba_teams - vegas_teams)
        print("Teams in Vegas but not in NBA logs:")
        print(vegas_teams - nba_teams)
        print(
            missing_vegas[["Season"]]
            .value_counts()
            .head(10)
        )
        print(
            missing_vegas[["Player", "Season"]]
            .value_counts()
            .head(10)
        )

    # --- Train vs test mask ---
    train_mask = data_sorted["Season"].isin(train_seasons)
    test_mask  = data_sorted["Season"] == test_season

    # Per-player base scoring from *training seasons only*
    player_base = (
        data_sorted.loc[train_mask]
                .groupby("Player")["PTS"]
                .mean()
                .rename("PTS_base")
    )
    # Merge back onto *all* rows (train + test)
    data_sorted = data_sorted.merge(player_base, on="Player", how="left")
    # Fallback for players that have no train-season history (rookies, etc.)
    global_pts_base = data_sorted.loc[train_mask, "PTS"].mean()
    data_sorted["PTS_base"] = data_sorted["PTS_base"].fillna(global_pts_base)

    # Efficiency (points per shot)
    data_sorted["PPS"] = data_sorted["PTS_ewm5"] / (data_sorted["FGA_ewm5"] + 1e-5)
    data_sorted["PPS"] = data_sorted["PPS"].clip(0.8, 1.7)  # realistic NBA bounds

    data_sorted["VegasPTSProp"] = (
        data_sorted["MIN_trend"] *
        data_sorted["FGA_rate"] *
        data_sorted["PPS"] *
        data_sorted["DefMultiplier"]
    )

    # --- Drop unneeded columns ---
    drop_cols = ["date", "home", "away", "total", "spread", "date_rev", "home_rev", "away_rev", "total_rev", "spread_rev"]
    data_sorted = data_sorted.drop(columns=[c for c in drop_cols if c in data_sorted.columns])
    # --- Compute implied team totals from Vegas lines ---
    T = data_sorted["VegasTotal"]
    S = data_sorted["VegasSpread"]
    data_sorted["HomeImplied"] = (T - S) / 2
    data_sorted["AwayImplied"] = (T + S) / 2
    # HomeAway = 1 if the player is HOME
    data_sorted["TeamImplied"] = data_sorted.apply(
        lambda row: row["HomeImplied"] if row["HomeAway"] == 1 else row["AwayImplied"],
        axis=1
    )
    data_sorted["OppImplied"] = data_sorted.apply(
        lambda row: row["AwayImplied"] if row["HomeAway"] == 1 else row["HomeImplied"],
        axis=1
    )

    # Scale by expected scoring environment
    EnvFactor = data_sorted["TeamImplied"] / 113  # normalize by league average
    data_sorted["VegasPTSProp"] = data_sorted["VegasPTSProp"] * EnvFactor

    player_group = data_sorted.groupby("Player")

    def safe_normalize(colname):
        if colname not in data_sorted.columns:
            print(f"WARNING: Missing column for normalization: {colname}")
            return np.nan  # or skip
        return (data_sorted[colname] - player_group[colname].transform("mean")) / \
            (player_group[colname].transform("std") + 1e-6)

    cols_to_normalize = [
        "PTS_ewm5", "PTS_delta5", "PTS_std_season", "PTS_iqr_season",
        "FGA_ewm5", "FGA_rolling",
        "MIN_ewm5", "MIN_rolling",
        "FG3A_ewm5", "FG3A_rolling",
        "USG_approx", "USG_rolling5", "USG_ewm5",
        "vsOpp_PTS_avg", "vsOpp_PTS_last5",
        "Opponent_DEF_RATING", "Opponent_PACE", "DefMultiplier",
        "PTS_homeaway_expected", "PTS_home_boost",
        "VegasTotal", "VegasSpread",
        "TeamImplied", "OppImplied",
        "VegasPTSProp"
    ]

    for col in cols_to_normalize:
        norm_col = col + "_norm"
        data_sorted[norm_col] = safe_normalize(col)


    # --- Compute over/under hit and margin ---
    data_sorted["OU_Hit"] = (data_sorted["PTS_next"] > data_sorted["VegasPTSProp"]).astype(int)
    data_sorted["OU_Margin"] = data_sorted["PTS_next"] - data_sorted["VegasPTSProp"]

    data_sorted["PTS_residual"] = (
        data_sorted["PTS_next"] - data_sorted["VegasPTSProp"]
    )

    # --- Drop rows with missing Vegas data ---
    vegas_cols = [
        "VegasTotal", "VegasSpread", 
        "TeamImplied", "OppImplied", 
        "VegasPTSProp"
    ]
    print("Missing Vegas rows before converting to numeric:")
    print(data_sorted[vegas_cols].isna().sum())
    for col in vegas_cols:
        if col in data_sorted.columns:
            data_sorted[col] = pd.to_numeric(data_sorted[col], errors="coerce")
    print("Missing Vegas rows after converting to numeric:")
    print(data_sorted[vegas_cols].isna().sum())
    data_sorted = data_sorted.dropna(subset=vegas_cols)
    print(data_sorted[vegas_cols].dtypes)


    core_features = [
        # ---------------------------
        # Temporal context
        # ---------------------------
        "GameIndex_norm",

        # ---------------------------
        # PTS form & streaks (normalized)
        # ---------------------------
        "PTS_ewm5_norm",
        "PTS_delta5_norm",
        "PTS_std_season_norm",
        "PTS_iqr_season_norm",

        # ---------------------------
        # Volume + efficiency (normalized)
        # ---------------------------
        "FGA_ewm5_norm",
        "FGA_rolling_norm",

        "MIN_ewm5_norm",
        "MIN_rolling_norm",

        "FG3A_ewm5_norm",
        "FG3A_rolling_norm",

        # ---------------------------
        # Usage (normalized)
        # ---------------------------
        "USG_approx_norm",
        "USG_rolling5_norm",
        "USG_ewm5_norm",

        # ---------------------------
        # Opponent history (normalized)
        # ---------------------------
        "vsOpp_PTS_avg_norm",
        "vsOpp_PTS_last5_norm",

        # ---------------------------
        # Opponent environment (normalized)
        # ---------------------------
        "Opponent_DEF_RATING_norm",
        "Opponent_PACE_norm",
        "DefMultiplier_norm",

        # ---------------------------
        # Home/Away scoring context (normalized)
        # ---------------------------
        "PTS_homeaway_expected_norm",
        "PTS_home_boost_norm",

        # ---------------------------
        # Vegas features (normalized)
        # ---------------------------
        "VegasTotal_norm",
        "VegasSpread_norm",

        "TeamImplied_norm",
        "OppImplied_norm",

        "VegasPTSProp_norm",

        # ---------------------------
        # Identifiers (not normalized)
        # ---------------------------
        "PlayerID",
        "SeasonIndex"
    ]

    # --- Only keep features that actually exist in data ---
    feature_cols = [c for c in core_features if c in data_sorted.columns]

    print("Using feature cols:", feature_cols)

    # --- Ensure all feature columns are numeric ---
    data_sorted[feature_cols] = data_sorted[feature_cols].apply(pd.to_numeric, errors='coerce')
    data_sorted = data_sorted.dropna(subset=feature_cols + ['PTS_next'])

    # --- Build X_sorted -- with extra columns kept for evaluation but not used as features ---
    X_sorted = data_sorted[feature_cols].copy()
    X_sorted['Player'] = data_sorted['Player']
    X_sorted['Season'] = data_sorted['Season']
    # Keep PTS_next and PTS_season_avg available for baseline reconstruction
    X_sorted['PTS_next'] = data_sorted['PTS_next']
    X_sorted['PTS_season_avg'] = data_sorted['PTS_season_avg']
    X_sorted['DaysAgo'] = data_sorted['DaysAgo']
    X_sorted['VegasPTSProp'] = data_sorted['VegasPTSProp']
    y_sorted = data_sorted['PTS_residual']
    #y_sorted = data_sorted['PTS_next']

    # --- Global constant baseline: always predict global mean points ---
    global_mean = data_sorted['PTS_next'].mean()
    global_mean_mae = mean_absolute_error(data_sorted['PTS_next'], np.full_like(data_sorted['PTS_next'], global_mean))
    print(f"Global mean baseline MAE: {global_mean_mae:.2f}")

    # --- Player-season mean baseline: predict each row by that player's season avg ---
    player_season_mean = data_sorted.groupby(['Player', 'Season'])['PTS_next'].transform('mean')
    ps_mean_mae = mean_absolute_error(data_sorted['PTS_next'], player_season_mean)
    print(f"Player-season mean baseline MAE: {ps_mean_mae:.2f}")

    print("PTS_next mean:", data_sorted['PTS_next'].mean())
    print("PTS_next std:", data_sorted['PTS_next'].std())

    # --- Quick test: Random split ---
    X_all = X_sorted.drop(columns=['Player', 'Season'])
    y_all = y_sorted
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
        X_all, y_all, test_size=0.2, shuffle=True, random_state=42
    )
    rf_tmp, y_pred_tmp = doRf(X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp)
    print("Random split RF MAE:", mean_absolute_error(y_test_tmp, y_pred_tmp))

    # --- Debug ---
    print(X_sorted.head())

    def process_player(player):
        ''' Train model(s) and run predictions for a player's test set. '''
        results = []
        print(f"\n--- Processing player: {player} ---")

        # Train = everything EXCEPT this player's chosen season
        train_mask = ~((X_sorted["Player"] == player) &
                    (X_sorted["Season"] == test_season))

        # Test = ONLY this player's chosen season
        test_mask = ((X_sorted["Player"] == player) &
                    (X_sorted["Season"] == test_season))

        if test_mask.sum() == 0:
            print(f"No test data for {player} in season {test_season}")
            return []

        # Model inputs
        X_train_full = X_sorted[train_mask].copy()
        X_test_full  = X_sorted[test_mask].copy()

        # Targets
        y_train = y_sorted[train_mask]
        y_test  = y_sorted[test_mask]

        # Baseline components in point-space
        baseline_test = X_test_full["PTS_season_avg"].to_numpy()
        y_true_points = X_test_full["PTS_next"].to_numpy()

        # Restrict to feature columns for model input
        X_train = X_train_full[feature_cols]
        X_test  = X_test_full[feature_cols]

        # --- Recency weighting (exponential decay by DaysAgo) ---
        days_ago_train = X_train_full["DaysAgo"].to_numpy()
        half_life = 80.0  # ~80 days -> weight falls by ~e^-1
        sample_weight = np.exp(-days_ago_train / half_life)

        # --- Baseline MAE (season average only) ---
        baseline_mae = mean_absolute_error(y_true_points, baseline_test)
        print(f"Baseline MAE (season avg) for {player} {test_season}: {baseline_mae:.2f}")

        # --- Model Training & Evaluation ---

        if "RandomForest" in models:
            print(f"Training Random Forest for {player}...")
            rf_model, y_pred_rf = doRf(X_train, y_train, X_test, y_test,
                                            sample_weight=sample_weight)
            # Convert residual predictions back to point-space
            #y_pred_points_rf = baseline_test + y_pred_rf

            #mae_points_rf = mean_absolute_error(y_true_points, y_pred_rf)
            #r2_points_rf = r2_score(y_true_points, y_pred_rf)
            
            # Convert residual predictions back to PTS
            y_pred_pts = X_test_full["VegasPTSProp"].to_numpy() + y_pred_rf

            mae_points_rf = mean_absolute_error(y_true_points, y_pred_pts)
            r2_points_rf = r2_score(y_true_points, y_pred_pts)
            print(f"Random Forest POINT Results: MAE={mae_points_rf:.2f}, R^2={r2_points_rf:.3f}")

            # Find and print feature importances
            model_feature_cols = [c for c in feature_cols if c in X_train.columns]
            print_feature_importances(rf_model, model_feature_cols, top_n=20, player_name=player)

            results.append({
                "player": player,
                "model": "RandomForest",
                "train_seasons": train_seasons,
                "test_season": test_season,
                "metrics": {
                    "MAE": mae_points_rf,
                    "R2": r2_points_rf,
                    "Baseline_MAE": baseline_mae,
                },
            })

        if "XGBoost" in models:
            print(f"Training XGBoost for {player}...")
            xgb_model, y_pred_xgb = doXgb(X_train, y_train, X_test, y_test,
                                                sample_weight=sample_weight)
            #y_pred_points_xgb = baseline_test + y_pred_resid_xgb

            # Convert residual predictions back to PTS
            y_pred_pts = X_test_full["VegasPTSProp"].to_numpy() + y_pred_xgb

            mae_points_xgb = mean_absolute_error(y_true_points, y_pred_pts)
            r2_points_xgb = r2_score(y_true_points, y_pred_pts)

            #mae_points_xgb = mean_absolute_error(y_true_points, y_pred_xgb)
            #r2_points_xgb = r2_score(y_true_points, y_pred_xgb)

            print(f"XGBoost POINT Results: MAE={mae_points_xgb:.2f}, R^2={r2_points_xgb:.3f}")

            results.append({
                "player": player,
                "model": "XGBoost",
                "train_seasons": train_seasons,
                "test_season": test_season,
                "metrics": {
                    "MAE": mae_points_xgb,
                    "R2": r2_points_xgb,
                    "Baseline_MAE": baseline_mae,
                },
            })

        return results
    
    # --- TEMP CODE -- hack to test a bunch of players quickly ---
    # --- get list of names of first 25 players from filtered list ---
    #hacked_player_list = filtered_players[:25]
    #players_of_interest_hack = [p['full_name'] for p in hacked_player_list]
    # --- Threaded execution ---
    with ThreadPoolExecutor(max_workers=min(len(players_of_interest), 5)) as executor:
        future_to_player = {executor.submit(process_player, player): player for player in players_of_interest}

        for future in as_completed(future_to_player):
            results.extend(future.result())

    print(results)

    return results
