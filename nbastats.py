# --- Imports ---
import os
import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Tuple
from datetime import datetime
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

# Bump this when feature engineering changes to invalidate cached processed data.
FEATURE_VERSION = "v1"
DEFAULT_HALF_LIFE_DAYS = 80.0
TUNE_VALID_FRACTION = 0.15
CURRENT_SEASON_TTL_SECONDS = 12 * 60 * 60
PAST_SEASON_TTL_SECONDS = 365 * 24 * 60 * 60
TOP_SCORERS_TTL_SECONDS = 12 * 60 * 60

def compute_recency_weights(days_ago: np.ndarray, half_life: float = DEFAULT_HALF_LIFE_DAYS) -> np.ndarray:
    return np.exp(-days_ago / half_life)

def get_current_season_label(now: datetime | None = None) -> str:
    now = now or datetime.now()
    if now.month >= 7:
        start_year = now.year
        end_year = now.year + 1
    else:
        start_year = now.year - 1
        end_year = now.year
    return f"{start_year}-{str(end_year)[-2:]}"

def is_current_season(season: str) -> bool:
    return season == get_current_season_label()

def get_scorer_season(train_seasons: List[str], season: str | None) -> str:
    return season or max(train_seasons)

def build_all_seasons(train_seasons: List[str], season: str | None) -> List[str]:
    return list(dict.fromkeys(train_seasons + [season]))

def prepare_pipeline_inputs(cache: CacheManager, players_of_interest: List[str], train_seasons: List[str],
                            season: str | None, debug_prefix: str = "") -> Dict[str, Any]:
    vegas_small_all = get_vegas_data()
    scorer_season = get_scorer_season(train_seasons, season)
    prefix = f"{debug_prefix} " if debug_prefix else ""
    print(f"{prefix}Using top scorers from season: {scorer_season}")
    filtered_players, top100 = get_filtered_players(cache, players_of_interest, scorer_season)
    all_seasons = build_all_seasons(train_seasons, season)
    return {
        "vegas_small_all": vegas_small_all,
        "filtered_players": filtered_players,
        "top100": top100,
        "all_seasons": all_seasons,
        "scorer_season": scorer_season,
    }

def time_based_split_indices(data_sorted: pd.DataFrame, candidate_idx: pd.Index,
                             valid_frac: float = TUNE_VALID_FRACTION,
                             date_col: str = "GAME_DATE") -> Tuple[pd.Index, pd.Index] | None:
    if candidate_idx.empty:
        return None
    subset = data_sorted.loc[candidate_idx]
    if subset.empty:
        return None
    subset = subset.sort_values(date_col)
    n = len(subset)
    n_val = max(1, int(n * valid_frac))
    n_train = n - n_val
    if n_train < 5 or n_val < 1:
        return None
    train_idx = subset.index[:-n_val]
    val_idx = subset.index[-n_val:]
    return train_idx, val_idx

def tune_model_params(model_name: str,
                      data_sorted: pd.DataFrame,
                      X_sorted: pd.DataFrame,
                      y_sorted: pd.Series,
                      feature_cols: List[str],
                      candidate_idx: pd.Index,
                      player_name: str = "",
                      debug_prefix: str = "") -> Dict[str, Any] | None:
    split = time_based_split_indices(data_sorted, candidate_idx)
    if split is None:
        if debug_prefix:
            print(f"{debug_prefix} {player_name} {model_name}: tuning skipped (insufficient rows)")
        return None

    train_idx, val_idx = split
    X_train = X_sorted.loc[train_idx, feature_cols]
    y_train = y_sorted.loc[train_idx]
    X_val = X_sorted.loc[val_idx, feature_cols]

    vegas_val = data_sorted.loc[val_idx, "VegasPTSProp"].to_numpy()
    y_val_points = data_sorted.loc[val_idx, "PTS_next"].to_numpy()

    days_ago_train = X_sorted.loc[train_idx, "DaysAgo"].to_numpy()
    sample_weight = compute_recency_weights(days_ago_train)

    if model_name == "RandomForest":
        grid = RF_TUNE_GRID
        train_func = train_rf_model
    elif model_name == "XGBoost":
        grid = XGB_TUNE_GRID
        train_func = train_xgb_model
    else:
        return None

    best_params = None
    best_mae = float("inf")
    baseline_mae = mean_absolute_error(y_val_points, vegas_val)

    for params in grid:
        model = train_func(X_train, y_train, sample_weight=sample_weight, params=params)
        pred_resid = model.predict(X_val)
        pred_pts = vegas_val + pred_resid
        mae = mean_absolute_error(y_val_points, pred_pts)
        if mae < best_mae:
            best_mae = mae
            best_params = params

    use_baseline = best_mae >= baseline_mae
    if debug_prefix:
        print(
            f"{debug_prefix} {player_name} {model_name}: "
            f"val_mae={best_mae:.2f}, "
            f"baseline_mae={baseline_mae:.2f}, "
            f"train_rows={len(train_idx)}, "
            f"val_rows={len(val_idx)}, "
            f"best_params={best_params}"
        )
    return {
        "best_params": best_params,
        "val_mae": best_mae,
        "baseline_mae": baseline_mae,
        "use_baseline": use_baseline,
    }

RF_BASE_PARAMS = {
    "n_estimators": 400,
    "max_depth": 12,
    "random_state": 42,
    "n_jobs": -1,
}
XGB_BASE_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "min_child_weight": 2,
}
RF_TUNE_GRID = [
    {"max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"max_depth": 12, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"max_depth": 12, "min_samples_leaf": 2, "max_features": 0.7},
]
XGB_TUNE_GRID = [
    {"max_depth": 6, "min_child_weight": 2, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "min_child_weight": 2, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "min_child_weight": 4, "subsample": 0.9, "colsample_bytree": 0.9},
    {"max_depth": 10, "min_child_weight": 2, "subsample": 1.0, "colsample_bytree": 0.9},
]

def doRf(X_train, y_train, X_test, y_test, sample_weight=None, params: Dict[str, Any] | None = None):
    ''' Train Random Forest model and evaluate on test set. '''
    rf_params = RF_BASE_PARAMS.copy()
    if params:
        rf_params.update(params)
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = rf.predict(X_test)

    print("Random Forest Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2:", r2_score(y_test, y_pred))

    return rf, y_pred

def doXgb(X_train, y_train, X_test, y_test, sample_weight=None, params: Dict[str, Any] | None = None):
    ''' Train XGBoost model and evaluate on test set. '''
    xgb_params = XGB_BASE_PARAMS.copy()
    if params:
        xgb_params.update(params)
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = xgb.predict(X_test)

    print("\nXGBoost Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2:", r2_score(y_test, y_pred))

    return xgb, y_pred

def train_rf_model(X_train, y_train, sample_weight=None, params: Dict[str, Any] | None = None):
    rf_params = RF_BASE_PARAMS.copy()
    if params:
        rf_params.update(params)
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    return rf

def train_xgb_model(X_train, y_train, sample_weight=None, params: Dict[str, Any] | None = None):
    xgb_params = XGB_BASE_PARAMS.copy()
    if params:
        xgb_params.update(params)
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    return xgb

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

def normalize_team_abbr(abbr: str) -> str:
    if not abbr:
        return abbr
    cleaned = abbr.strip().upper()
    return TEAM_FIX.get(cleaned, cleaned)

def parse_home_away(value: str) -> int:
    if value is None:
        raise ValueError("home_away is required.")
    v = value.strip().lower()
    if v in ["home", "h", "1", "true", "yes"]:
        return 1
    if v in ["away", "a", "0", "false", "no"]:
        return 0
    raise ValueError(f"Invalid home_away value: {value}")

def get_player_log_metadata(df: pd.DataFrame):
    if df is None or df.empty:
        return None, 0
    last_game_date = pd.to_datetime(df["GAME_DATE"]).max().date().isoformat()
    return last_game_date, len(df)

def get_filtered_players(cache: CacheManager, players_of_interest, scorer_season: str):
    top100 = get_top_scorers(cache, scorer_season, top_n=200, per_mode="PerGame")
    all_players = get_all_players(cache)

    top_ids = set(top100["player_id"].tolist())
    requested_names = {p.lower() for p in players_of_interest}
    filtered_players = [
        p for p in all_players
        if p["id"] in top_ids or p["full_name"].lower() in requested_names
    ]
    resolved_names = {p["full_name"].lower() for p in filtered_players}
    missing_requested = requested_names - resolved_names
    if missing_requested:
        print(f"WARNING: Requested players not found: {sorted(missing_requested)}")

    return filtered_players, top100

def build_team_stats_by_season(cache: CacheManager, seasons):
    teams_info = get_teams(cache)
    team_id_to_abbr = {t['id']: t['abbreviation'] for t in teams_info}
    team_stats_by_season = {}
    for season in seasons:
        team_stats, last_updated = get_team_stats_with_meta(cache, season)
        team_stats['TEAM_ABBREVIATION'] = team_stats['TEAM_ID'].map(team_id_to_abbr)
        team_stats = team_stats[['TEAM_ABBREVIATION', 'DEF_RATING', 'PACE']]
        team_stats_by_season[season] = {
            "df": team_stats,
            "last_updated": last_updated
        }
    return team_stats_by_season

def get_processed_player_season(cache: CacheManager, player: dict, season: str, team_stats: pd.DataFrame, team_stats_updated: int):
    logs_df = get_player_season_log(cache, player, season)
    last_game_date, row_count = get_player_log_metadata(logs_df)
    cached_df, cached_team_stats_updated, cached_last_date, cached_row_count = cache.load_processed_player_season(
        player["id"], season, FEATURE_VERSION
    )
    if (
        cached_df is not None and
        cached_team_stats_updated == team_stats_updated and
        cached_last_date == last_game_date and
        cached_row_count == row_count
    ):
        return cached_df

    df = process_player_season(player, season, team_stats, cache, logs_df=logs_df)
    cache.save_processed_player_season(
        player["id"],
        season,
        FEATURE_VERSION,
        team_stats_updated,
        last_game_date,
        row_count,
        df
    )
    return df

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

def build_player_season_data(filtered_players, seasons, team_stats_by_season, cache: CacheManager):
    all_data = []
    for season in seasons:
        stats_entry = team_stats_by_season[season]
        team_stats = stats_entry["df"]
        team_stats_updated = stats_entry["last_updated"]
        for player in filtered_players:
            try:
                df_player_season = get_processed_player_season(
                    cache,
                    player,
                    season,
                    team_stats,
                    team_stats_updated,
                )
                all_data.append(df_player_season)
            except Exception as e:
                print(f"Error processing {player['full_name']} ({season}): {e}")
                continue
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def get_feature_definitions(use_vegas_features: bool):
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

    core_features = [
        "GameIndex_norm",
        "PTS_ewm5_norm",
        "PTS_delta5_norm",
        "PTS_std_season_norm",
        "PTS_iqr_season_norm",
        "FGA_ewm5_norm",
        "FGA_rolling_norm",
        "MIN_ewm5_norm",
        "MIN_rolling_norm",
        "FG3A_ewm5_norm",
        "FG3A_rolling_norm",
        "USG_approx_norm",
        "USG_rolling5_norm",
        "USG_ewm5_norm",
        "vsOpp_PTS_avg_norm",
        "vsOpp_PTS_last5_norm",
        "Opponent_DEF_RATING_norm",
        "Opponent_PACE_norm",
        "DefMultiplier_norm",
        "PTS_homeaway_expected_norm",
        "PTS_home_boost_norm",
        "VegasTotal_norm",
        "VegasSpread_norm",
        "TeamImplied_norm",
        "OppImplied_norm",
        "VegasPTSProp_norm",
        "PlayerID",
        "SeasonIndex"
    ]

    if not use_vegas_features:
        vegas_line_features = [
            "VegasTotal_norm",
            "VegasSpread_norm",
            "TeamImplied_norm",
            "OppImplied_norm",
        ]
        core_features = [c for c in core_features if c not in vegas_line_features]

    return core_features, cols_to_normalize

def merge_vegas_lines(data_sorted: pd.DataFrame, vegas_small: pd.DataFrame, debug_prefix: str = "") -> pd.DataFrame:
    data_sorted = data_sorted.merge(
        vegas_small,
        how="left",
        left_on=["GAME_DATE", "Team", "Opponent"],
        right_on=["date", "home", "away"]
    )
    data_sorted = data_sorted.merge(
        vegas_small,
        how="left",
        left_on=["GAME_DATE", "Opponent", "Team"],
        right_on=["date", "home", "away"],
        suffixes=("", "_rev")
    )
    data_sorted["VegasTotal"] = data_sorted["total"].fillna(data_sorted["total_rev"])
    data_sorted["VegasSpread"] = data_sorted["spread"].fillna(data_sorted["spread_rev"])
    if debug_prefix:
        print(f"{debug_prefix} Vegas merge missing totals:",
              int(data_sorted["VegasTotal"].isna().sum()),
              "missing spreads:",
              int(data_sorted["VegasSpread"].isna().sum()))
    return data_sorted

def determine_vegas_usage(data_sorted: pd.DataFrame, debug_prefix: str = ""):
    seasons_with_vegas = (
        data_sorted.groupby("Season")["VegasTotal"]
        .apply(lambda s: s.notna().any())
    )
    seasons_missing_vegas = seasons_with_vegas[~seasons_with_vegas].index.tolist()
    use_vegas_features = (
        data_sorted["VegasTotal"].notna().any() and data_sorted["VegasSpread"].notna().any()
    )
    if not use_vegas_features:
        prefix = f"{debug_prefix} " if debug_prefix else ""
        print(f"{prefix}Skipping Vegas line features; no Vegas lines available.")
    elif seasons_missing_vegas and debug_prefix:
        print(f"{debug_prefix} Vegas line coverage missing for seasons: {seasons_missing_vegas}")
    return use_vegas_features, seasons_missing_vegas

def add_normalized_columns(data_sorted: pd.DataFrame, stats_source: pd.DataFrame, cols_to_normalize, debug_prefix: str = "") -> pd.DataFrame:
    if stats_source is None or stats_source.empty:
        stats_source = data_sorted
    cols_present = [c for c in cols_to_normalize if c in data_sorted.columns and c in stats_source.columns]
    missing_cols = [c for c in cols_to_normalize if c not in cols_present]
    if missing_cols and debug_prefix:
        print(f"{debug_prefix} WARNING: Missing columns for normalization: {missing_cols}")

    if not cols_present:
        return data_sorted

    global_means = stats_source[cols_present].mean(numeric_only=True)
    global_stds = stats_source[cols_present].std(numeric_only=True)

    for col in cols_present:
        mean_map = stats_source.groupby("Player")[col].mean()
        std_map = stats_source.groupby("Player")[col].std()
        mean_series = data_sorted["Player"].map(mean_map).fillna(global_means.get(col, np.nan))
        std_series = data_sorted["Player"].map(std_map).replace(0, np.nan).fillna(global_stds.get(col, np.nan))
        data_sorted[col + "_norm"] = (data_sorted[col] - mean_series) / (std_series + 1e-6)

    return data_sorted

def compute_normalized_values(raw_values: Dict[str, Any], player_data: pd.DataFrame, global_data: pd.DataFrame,
                              cols_to_normalize: List[str]) -> Dict[str, float]:
    cols_present = [c for c in cols_to_normalize if c in player_data.columns and c in global_data.columns]
    if not cols_present:
        return {}

    player_means = player_data[cols_present].mean(numeric_only=True)
    player_stds = player_data[cols_present].std(numeric_only=True)
    global_means = global_data[cols_present].mean(numeric_only=True)
    global_stds = global_data[cols_present].std(numeric_only=True)

    norm_values = {}
    for col in cols_present:
        if col not in raw_values or raw_values[col] is None:
            continue
        mean = player_means.get(col, np.nan)
        std = player_stds.get(col, np.nan)
        if pd.isna(mean):
            mean = global_means.get(col, np.nan)
        if pd.isna(std) or std == 0:
            std = global_stds.get(col, np.nan)
        if pd.isna(mean) or pd.isna(std):
            continue
        norm_values[col + "_norm"] = (raw_values[col] - mean) / (std + 1e-6)

    return norm_values

def resolve_vegas_line(vegas_small: pd.DataFrame, team: str, opponent: str, vegas_date: pd.Timestamp,
                       home_away: int | None, vegas_total: float | None, vegas_spread: float | None,
                       debug_prefix: str = "") -> Dict[str, Any]:
    vegas_row_home = vegas_small[
        (vegas_small["date"] == vegas_date) &
        (vegas_small["home"] == team) &
        (vegas_small["away"] == opponent)
    ]
    vegas_row_away = vegas_small[
        (vegas_small["date"] == vegas_date) &
        (vegas_small["home"] == opponent) &
        (vegas_small["away"] == team)
    ]

    if home_away is None:
        if not vegas_row_home.empty:
            home_away = 1
        elif not vegas_row_away.empty:
            home_away = 0

    home_team = None
    away_team = None
    vegas_row = pd.DataFrame()
    if home_away == 1:
        home_team = team
        away_team = opponent
        vegas_row = vegas_row_home
    elif home_away == 0:
        home_team = opponent
        away_team = team
        vegas_row = vegas_row_away

    if not vegas_row_home.empty and not vegas_row_away.empty:
        print(f"{debug_prefix} WARNING: both home/away Vegas rows exist for {team} vs {opponent} on {vegas_date.date()}")

    if vegas_row.empty and ((home_away == 1 and not vegas_row_away.empty) or (home_away == 0 and not vegas_row_home.empty)):
        print(f"{debug_prefix} WARNING: home_away mismatch; inferring from Vegas row")
        if home_away == 1:
            home_away = 0
            home_team = opponent
            away_team = team
            vegas_row = vegas_row_away
        elif home_away == 0:
            home_away = 1
            home_team = team
            away_team = opponent
            vegas_row = vegas_row_home

    if vegas_total is None or vegas_spread is None:
        if not vegas_row.empty:
            vegas_total = float(vegas_row["total"].iloc[0])
            vegas_spread = float(vegas_row["spread"].iloc[0])

    if (vegas_total is None or vegas_spread is None) and debug_prefix:
        print(f"{debug_prefix} Vegas lines missing for {team} vs {opponent} on {vegas_date.date()}")

    return {
        "home_away": home_away,
        "vegas_total": vegas_total,
        "vegas_spread": vegas_spread,
        "home_team": home_team,
        "away_team": away_team,
    }

def build_pipeline_data(filtered_players, all_seasons, train_seasons, cache: CacheManager, vegas_small: pd.DataFrame,
                        debug_prefix: str = "", normalize_source: str = "train",
                        debug_players: List[str] | None = None):
    team_stats_by_season = build_team_stats_by_season(cache, all_seasons)
    data_raw = build_player_season_data(filtered_players, all_seasons, team_stats_by_season, cache)
    if data_raw.empty:
        return None
    if debug_prefix:
        print(f"{debug_prefix} Combined data rows={len(data_raw)} cols={len(data_raw.columns)}")
    if debug_prefix and debug_players:
        players_lower = {p.lower() for p in debug_players}
        data_debug = data_raw[data_raw["Player"].str.lower().isin(players_lower)].copy()
        if not data_debug.empty:
            if not np.issubdtype(data_debug["GAME_DATE"].dtype, np.datetime64):
                data_debug["GAME_DATE"] = pd.to_datetime(data_debug["GAME_DATE"], errors="coerce")
            summary = (
                data_debug
                .groupby(["Player", "Season"])
                .agg(rows=("Player", "size"), last_game=("GAME_DATE", "max"))
                .reset_index()
            )
            print(f"{debug_prefix} Player-season row counts:")
            for _, row in summary.iterrows():
                last_game = row["last_game"]
                last_game_str = last_game.date().isoformat() if pd.notna(last_game) else "unknown"
                print(f"{debug_prefix} {row['Player']} {row['Season']}: rows={int(row['rows'])}, last_game={last_game_str}")
            for player in sorted({p for p in data_debug["Player"].unique()}):
                seasons_present = set(summary[summary["Player"] == player]["Season"].tolist())
                missing = [s for s in all_seasons if s not in seasons_present]
                if missing:
                    print(f"{debug_prefix} {player}: missing seasons {missing}")
        else:
            print(f"{debug_prefix} No rows found for debug players in data_raw.")

    data_sorted = features_and_sort(data_raw)
    if debug_prefix:
        print(f"{debug_prefix} After feature engineering rows={len(data_sorted)} cols={len(data_sorted.columns)}")

    if vegas_small is not None and not vegas_small.empty and "date" in vegas_small.columns:
        min_date = data_sorted["GAME_DATE"].min()
        max_date = data_sorted["GAME_DATE"].max()
        vegas_small = vegas_small[(vegas_small["date"] >= min_date) &
                                  (vegas_small["date"] <= max_date)]

    data_sorted = merge_vegas_lines(data_sorted, vegas_small, debug_prefix=debug_prefix)
    use_vegas_features, seasons_missing_vegas = determine_vegas_usage(data_sorted, debug_prefix=debug_prefix)
    if use_vegas_features and seasons_missing_vegas and debug_prefix:
        print(f"{debug_prefix} Vegas line coverage missing for seasons: {seasons_missing_vegas}")

    train_mask = data_sorted["Season"].isin(train_seasons)
    player_base = (
        data_sorted.loc[train_mask]
                .groupby("Player")["PTS"]
                .mean()
                .rename("PTS_base")
    )
    data_sorted = data_sorted.merge(player_base, on="Player", how="left")
    global_pts_base = data_sorted.loc[train_mask, "PTS"].mean()
    data_sorted["PTS_base"] = data_sorted["PTS_base"].fillna(global_pts_base)

    data_sorted["PPS"] = data_sorted["PTS_ewm5"] / (data_sorted["FGA_ewm5"] + 1e-5)
    data_sorted["PPS"] = data_sorted["PPS"].clip(0.8, 1.7)

    data_sorted["VegasPTSProp"] = (
        data_sorted["MIN_trend"] *
        data_sorted["FGA_rate"] *
        data_sorted["PPS"] *
        data_sorted["DefMultiplier"]
    )

    drop_cols = ["date", "home", "away", "total", "spread", "date_rev", "home_rev", "away_rev", "total_rev", "spread_rev"]
    data_sorted = data_sorted.drop(columns=[c for c in drop_cols if c in data_sorted.columns])

    if use_vegas_features:
        T = data_sorted["VegasTotal"]
        S = data_sorted["VegasSpread"]
        data_sorted["HomeImplied"] = (T - S) / 2
        data_sorted["AwayImplied"] = (T + S) / 2
        data_sorted["TeamImplied"] = data_sorted.apply(
            lambda row: row["HomeImplied"] if row["HomeAway"] == 1 else row["AwayImplied"],
            axis=1
        )
        data_sorted["OppImplied"] = data_sorted.apply(
            lambda row: row["AwayImplied"] if row["HomeAway"] == 1 else row["HomeImplied"],
            axis=1
        )
        EnvFactor = data_sorted["TeamImplied"] / 113
        data_sorted["VegasPTSProp"] = data_sorted["VegasPTSProp"] * EnvFactor

    data_sorted_full = data_sorted.copy()
    core_features, cols_to_normalize = get_feature_definitions(use_vegas_features)
    norm_source_cols = [c for c in cols_to_normalize if c in data_sorted.columns]

    if normalize_source == "full":
        stats_source = data_sorted_full
    else:
        stats_source = data_sorted_full[data_sorted_full["Season"].isin(train_seasons)]
    data_sorted = add_normalized_columns(data_sorted, stats_source, norm_source_cols, debug_prefix=debug_prefix)

    data_sorted["OU_Hit"] = (data_sorted["PTS_next"] > data_sorted["VegasPTSProp"]).astype(int)
    data_sorted["OU_Margin"] = data_sorted["PTS_next"] - data_sorted["VegasPTSProp"]
    data_sorted["PTS_residual"] = data_sorted["PTS_next"] - data_sorted["VegasPTSProp"]

    vegas_cols = [
        "VegasTotal", "VegasSpread",
        "TeamImplied", "OppImplied",
        "VegasPTSProp"
    ]
    if use_vegas_features:
        rows_2025_before = (data_sorted["Season"] == "2025-26").sum()
        vegas_cols_present = [c for c in vegas_cols if c in data_sorted.columns]
        missing_cols = [c for c in vegas_cols if c not in vegas_cols_present]
        if missing_cols and debug_prefix:
            print(f"{debug_prefix} WARNING: Missing Vegas columns for dropna: {missing_cols}")
        for col in vegas_cols_present:
            data_sorted[col] = pd.to_numeric(data_sorted[col], errors="coerce")
        data_sorted = data_sorted.dropna(subset=vegas_cols_present)
        if debug_prefix:
            print(f"{debug_prefix} After Vegas dropna rows={len(data_sorted)}")
        rows_2025_after = (data_sorted["Season"] == "2025-26").sum()
        if (rows_2025_before or rows_2025_after) and debug_prefix:
            print(f"{debug_prefix} Vegas coverage 2025-26 rows kept: {rows_2025_after}/{rows_2025_before}")
        if debug_prefix and ("2025-26" in data_sorted_full["Season"].unique()) and ("2025-26" not in data_sorted["Season"].unique()):
            print(f"{debug_prefix} Season 2025-26 exists in full data but was dropped by Vegas filtering.")

    feature_cols = [c for c in core_features if c in data_sorted.columns]
    if feature_cols:
        data_sorted[feature_cols] = data_sorted[feature_cols].apply(pd.to_numeric, errors='coerce')
        data_sorted = data_sorted.dropna(subset=feature_cols + ['PTS_next'])
        if debug_prefix:
            print(f"{debug_prefix} After feature dropna rows={len(data_sorted)}")

    X_sorted = data_sorted[feature_cols].copy()
    X_sorted['Player'] = data_sorted['Player']
    X_sorted['Season'] = data_sorted['Season']
    X_sorted['PTS_next'] = data_sorted['PTS_next']
    X_sorted['PTS_season_avg'] = data_sorted['PTS_season_avg']
    X_sorted['DaysAgo'] = data_sorted['DaysAgo']
    X_sorted['VegasPTSProp'] = data_sorted['VegasPTSProp']
    y_sorted = data_sorted['PTS_residual']

    return {
        "data_full": data_sorted_full,
        "data_train": data_sorted,
        "feature_cols": feature_cols,
        "norm_source_cols": norm_source_cols,
        "core_features": core_features,
        "use_vegas_features": use_vegas_features,
        "X_sorted": X_sorted,
        "y_sorted": y_sorted,
        "team_stats_by_season": team_stats_by_season,
    }

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
    ttl_seconds = CURRENT_SEASON_TTL_SECONDS if is_current_season(season) else PAST_SEASON_TTL_SECONDS
    if df is None or (time.time() - last_updated) > ttl_seconds:
        print(f"[API CALL] Fetching logs for {player['full_name']} ({season})")
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season)
        df = gamelog.get_data_frames()[0]
        cache.save_player_logs(pid, season, df)
    else:
        ttl_hours = ttl_seconds / 3600
        print(f"[CACHE HIT] Logs for {player['full_name']} ({season}) (ttl={ttl_hours:.0f}h)")
    return df

def get_team_stats_with_meta(cache: CacheManager, season: str):
    ''' Get team stats for a given season with cache metadata. '''
    df, last_updated = cache.load_team_stats(season)
    if last_updated is None:
        last_updated = 0
    if df is None or (time.time() - last_updated) > 31536000:  # 1 year cache
        print(f"[API CALL] Fetching team stats for {season}")
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0]
        cache.save_team_stats(season, df)
        # Reload to get updated timestamp
        df, last_updated = cache.load_team_stats(season)
    else:
        print(f"[CACHE HIT] Team stats for {season}")
    return df, last_updated

def get_team_stats(cache: CacheManager, season: str):
    ''' Get team stats for a given season, using cache if available. '''
    df, _last_updated = get_team_stats_with_meta(cache, season)
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
    ttl_seconds = TOP_SCORERS_TTL_SECONDS if is_current_season(season) else PAST_SEASON_TTL_SECONDS
    if df is None or (time.time() - last_updated) > ttl_seconds:
        print(f"[API CALL] Fetching top scorers for {season}")
        leaders = LeagueLeaders(
            season=season,
            stat_category_abbreviation="PTS",
            per_mode48=per_mode
        )
        df = leaders.get_data_frames()[0]
        print(f"Fetched {len(df)} players from API.")
        df = df[["PLAYER_ID", "PTS", "RANK"]]  # keep relevant
        df.columns = [c.lower() for c in df.columns]
        # Save to cache
        cache.save_top_scorers(season, df)
        print(f"Saved top scorers for {season} to cache.")
    # Sort by player rank ascending (i.e. 1,2,3... not 100,99,98...))
    # NOTE KeyError was hitting here because only fields are saved (rather than full raw data) and we save them lowercase
    df_sorted = df.sort_values("rank", ascending=True).reset_index(drop=True)
    # Return top N players
    return df_sorted.head(top_n)

def get_vegas_data():
    '''
    Process csv containing historical Vegas data, return data frame.
    '''
    vegas_paths = [
        "data/nba_2008-2025.csv",
        "data/nba_2025-26_live.csv",
    ]
    vegas_frames = []
    for path in vegas_paths:
        if os.path.exists(path):
            vegas_frames.append(pd.read_csv(path))
    if not vegas_frames:
        raise FileNotFoundError("No Vegas odds files found under data/.")
    vegas = pd.concat(vegas_frames, ignore_index=True)
    # Basic cleanup
    vegas["date"] = pd.to_datetime(vegas["date"])
    # Uppercase team abbreviations
    vegas["home"] = vegas["home"].str.upper()
    vegas["away"] = vegas["away"].str.upper()
    # Normalize spread sign based on whos_favored if present (spread = away - home)
    if "whos_favored" in vegas.columns:
        vegas["whos_favored"] = vegas["whos_favored"].astype(str).str.lower()
        vegas["spread"] = pd.to_numeric(vegas["spread"], errors="coerce")
        vegas.loc[vegas["whos_favored"] == "home", "spread"] = -vegas["spread"].abs()
        vegas.loc[vegas["whos_favored"] == "away", "spread"] = vegas["spread"].abs()
    vegas["home"] = vegas["home"].replace(TEAM_FIX)
    vegas["away"] = vegas["away"].replace(TEAM_FIX)
    # Keep only columns we need
    vegas_small = vegas[["date", "home", "away", "total", "spread"]].copy()
    for col in ["total", "spread"]:
        vegas_small[col] = pd.to_numeric(vegas_small[col], errors="coerce")

    return vegas_small

def process_player_season(player: dict, season: str, team_stats: pd.DataFrame, cache: CacheManager, logs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Fetch and process a single player's season log into a DataFrame.
    """
    pname = player['full_name']
    # Fetch player game logs for this season
    df = logs_df if logs_df is not None else get_player_season_log(cache, player, season)

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
    best_models = {}
    print(f"Players: {players_of_interest}")

    pipeline_inputs = prepare_pipeline_inputs(cache, players_of_interest, train_seasons, test_season)
    vegas_small_all = pipeline_inputs["vegas_small_all"]
    filtered_players = pipeline_inputs["filtered_players"]
    top100 = pipeline_inputs["top100"]
    print(top100[["player_id", "pts", "rank"]])

    all_seasons = pipeline_inputs["all_seasons"]

    pipeline = build_pipeline_data(
        filtered_players,
        all_seasons,
        train_seasons,
        cache,
        vegas_small_all,
        normalize_source="train",
        debug_players=players_of_interest,
    )
    if pipeline is None:
        print("No data available after pipeline build.")
        return []

    data_sorted = pipeline["data_train"]
    feature_cols = pipeline["feature_cols"]
    X_sorted = pipeline["X_sorted"]
    y_sorted = pipeline["y_sorted"]

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
        sample_weight = compute_recency_weights(days_ago_train)

        # --- Baseline MAE (season average only) ---
        baseline_mae = mean_absolute_error(y_true_points, baseline_test)
        print(f"Baseline MAE (season avg) for {player} {test_season}: {baseline_mae:.2f}")

        # --- Model Training & Evaluation ---

        model_runners = {
            "RandomForest": {"runner": doRf, "label": "Random Forest"},
            "XGBoost": {"runner": doXgb, "label": "XGBoost"},
        }
        tuned_params = {}
        for model_name in models:
            if model_name not in model_runners:
                continue
            candidate_idx = X_sorted.index[train_mask & (X_sorted["Player"] == player)]
            tuned_params[model_name] = tune_model_params(
                model_name,
                data_sorted,
                X_sorted,
                y_sorted,
                feature_cols,
                candidate_idx,
                player_name=player,
                debug_prefix="[tune]",
            )
        for model_name in models:
            if model_name not in model_runners:
                continue
            tune_info = tuned_params.get(model_name)
            if tune_info and tune_info.get("use_baseline"):
                print(f"[tune] {player} {model_name}: using Vegas baseline (val_mae >= baseline)")
                y_pred = np.zeros_like(y_test)
                y_pred_pts = X_test_full["VegasPTSProp"].to_numpy()
                mae_points = mean_absolute_error(y_true_points, y_pred_pts)
                r2_points = r2_score(y_true_points, y_pred_pts)
                results.append({
                    "player": player,
                    "model": model_name,
                    "train_seasons": train_seasons,
                    "test_season": test_season,
                    "metrics": {
                        "MAE": mae_points,
                        "R2": r2_points,
                        "Baseline_MAE": baseline_mae,
                    },
                    "used_baseline": True,
                })
                continue
            print(f"Training {model_runners[model_name]['label']} for {player}...")
            model, y_pred = model_runners[model_name]["runner"](
                X_train,
                y_train,
                X_test,
                y_test,
                sample_weight=sample_weight,
                params=(tune_info or {}).get("best_params"),
            )
            y_pred_pts = X_test_full["VegasPTSProp"].to_numpy() + y_pred
            mae_points = mean_absolute_error(y_true_points, y_pred_pts)
            r2_points = r2_score(y_true_points, y_pred_pts)
            print(f"{model_runners[model_name]['label']} POINT Results: MAE={mae_points:.2f}, R^2={r2_points:.3f}")

            if model_name == "RandomForest":
                model_feature_cols = [c for c in feature_cols if c in X_train.columns]
                print_feature_importances(model, model_feature_cols, top_n=20, player_name=player)

            results.append({
                "player": player,
                "model": model_name,
                "train_seasons": train_seasons,
                "test_season": test_season,
                "metrics": {
                    "MAE": mae_points,
                    "R2": r2_points,
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

def predict_next_games(games, train_seasons, season, models, cache) -> List[Dict[str, Any]]:
    if not games:
        return []
    
    best_models = {}

    players_of_interest = [g.player for g in games]
    print(f"Next-game players: {players_of_interest}")

    pipeline_inputs = prepare_pipeline_inputs(cache, players_of_interest, train_seasons, season, debug_prefix="[predict_next]")
    vegas_small_all = pipeline_inputs["vegas_small_all"]
    filtered_players = pipeline_inputs["filtered_players"]
    all_seasons = pipeline_inputs["all_seasons"]
    pipeline = build_pipeline_data(
        filtered_players,
        all_seasons,
        train_seasons,
        cache,
        vegas_small_all,
        debug_prefix="[predict_next]",
        normalize_source="full",
        debug_players=players_of_interest,
    )
    if pipeline is None:
        return []

    data_sorted_full = pipeline["data_full"]
    data_sorted = pipeline["data_train"]
    feature_cols = pipeline["feature_cols"]
    norm_source_cols = pipeline["norm_source_cols"]
    use_vegas_features = pipeline["use_vegas_features"]
    X_sorted = pipeline["X_sorted"]
    y_sorted = pipeline["y_sorted"]
    team_stats_by_season = pipeline["team_stats_by_season"]
    print(f"[predict_next] Using {len(feature_cols)} feature columns")

    norm_source_cols_full = [c for c in norm_source_cols if c in data_sorted_full.columns]

    models_to_train = set(models)
    train_func_map = {
        "RandomForest": train_rf_model,
        "XGBoost": train_xgb_model,
    }
    if not (set(train_func_map.keys()) & models_to_train):
        raise ValueError("No valid models requested. Use 'RandomForest' and/or 'XGBoost'.")

    def game_value(game_obj, key):
        return getattr(game_obj, key) if hasattr(game_obj, key) else game_obj.get(key)

    player_models = {}
    player_train_counts = {}
    player_model_fallbacks = {}
    for player_name in {game_value(g, "player") for g in games}:
        player_mask = X_sorted["Player"] == player_name
        if player_mask.sum() == 0:
            player_models[player_name] = None
            player_train_counts[player_name] = 0
            player_model_fallbacks[player_name] = {}
            continue
        X_train_player = X_sorted.loc[player_mask, feature_cols]
        y_train_player = y_sorted.loc[player_mask]
        days_ago_train = X_sorted.loc[player_mask, "DaysAgo"].to_numpy()
        sample_weight = compute_recency_weights(days_ago_train)
        player_train_counts[player_name] = int(player_mask.sum())
        tuned_params = {}
        fallback_flags = {}
        for model_name in models_to_train:
            tuned_params[model_name] = tune_model_params(
                model_name,
                data_sorted,
                X_sorted,
                y_sorted,
                feature_cols,
                X_sorted.index[player_mask],
                player_name=player_name,
                debug_prefix="[tune]",
            )
            tune_info = tuned_params[model_name]
            fallback_flags[model_name] = bool(tune_info and tune_info.get("use_baseline"))
        trained = {}
        for model_name, train_func in train_func_map.items():
            if model_name not in models_to_train:
                continue
            tune_info = tuned_params.get(model_name)
            if tune_info and tune_info.get("use_baseline"):
                continue
            trained[model_name] = train_func(
                X_train_player,
                y_train_player,
                sample_weight=sample_weight,
                params=(tune_info or {}).get("best_params"),
            )
        player_models[player_name] = trained
        player_model_fallbacks[player_name] = fallback_flags
    print(f"[predict_next] Trained per-player models for {len([p for p,v in player_models.items() if v])} players")

    player_id_map = (
        data_sorted_full[["Player", "PlayerID"]]
        .drop_duplicates()
        .set_index("Player")["PlayerID"]
        .to_dict()
    )
    season_order = {s: i for i, s in enumerate(sorted(data_sorted_full['Season'].unique()))}

    results = []

    for game in games:
        player = game_value(game, "player")
        opponent = normalize_team_abbr(game_value(game, "opponent"))
        try:
            game_date = pd.to_datetime(game_value(game, "game_date"))
        except Exception as e:
            print(f"[predict_next] Invalid game_date for {player}: {e}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": game_value(game, "home_away"),
                "model": "N/A",
                "predicted_pts": None,
                "error": f"Invalid game_date: {e}",
            })
            continue
        provided_home_away = game_value(game, "home_away")
        home_away = None
        if provided_home_away is not None:
            try:
                home_away = parse_home_away(provided_home_away)
            except ValueError as e:
                print(f"[predict_next] Invalid home_away for {player}: {e}")
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": provided_home_away,
                    "model": "N/A",
                    "predicted_pts": None,
                    "error": str(e),
                })
                continue

        player_data = data_sorted_full[data_sorted_full["Player"] == player]
        if player_data.empty:
            print(f"[predict_next] No data for player {player}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": game_value(game, "home_away"),
                "model": "N/A",
                "predicted_pts": None,
                "error": f"No data available for player {player}",
            })
            continue

        player_season = player_data[player_data["Season"] == season]
        if player_season.empty:
            print(f"[predict_next] No data for player {player} in season {season}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": game_value(game, "home_away"),
                "model": "N/A",
                "predicted_pts": None,
                "error": f"No season data for player {player} in {season}",
            })
            continue

        last_row = player_season.sort_values("GAME_DATE").iloc[-1]
        if game_date <= last_row["GAME_DATE"]:
            print(f"[predict_next] game_date not after last game for {player}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": game_value(game, "home_away"),
                "model": "N/A",
                "predicted_pts": None,
                "error": "game_date must be after the player's last recorded game",
            })
            continue

        team = last_row["Team"]

        stats_entry = team_stats_by_season.get(season)
        if stats_entry is None:
            print(f"[predict_next] No team stats for season {season}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": game_value(game, "home_away"),
                "model": "N/A",
                "predicted_pts": None,
                "error": f"No team stats found for season {season}",
            })
            continue
        team_stats = stats_entry["df"]
        opp_stats = team_stats[team_stats["TEAM_ABBREVIATION"] == opponent]
        if opp_stats.empty:
            print(f"[predict_next] Opponent not found in team stats: {opponent}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": game_value(game, "home_away"),
                "model": "N/A",
                "predicted_pts": None,
                "error": f"Opponent not found in team stats: {opponent}",
            })
            continue

        opponent_def = float(opp_stats["DEF_RATING"].iloc[0])
        opponent_pace = float(opp_stats["PACE"].iloc[0])

        season_defs = data_sorted_full.loc[data_sorted_full["Season"] == season, "Opponent_DEF_RATING"]
        league_avg_def = season_defs.mean()
        if pd.isna(league_avg_def):
            league_avg_def = data_sorted_full["Opponent_DEF_RATING"].mean()
        def_multiplier = league_avg_def / opponent_def
        if pd.isna(def_multiplier) or pd.isna(opponent_def) or pd.isna(league_avg_def):
            print(
                "[predict_next] DefMultiplier NaN:",
                f"opponent_def={opponent_def}, league_avg_def={league_avg_def}, season={season}, opponent={opponent}"
            )
        def_multiplier = float(np.clip(def_multiplier, 0.85, 1.15))

        vs_data = player_data[player_data["Opponent"] == opponent]
        vs_avg = vs_data["PTS"].mean()
        vs_last5 = vs_data["PTS"].tail(5).mean()
        fallback_pts = player_season["PTS"].mean()
        if pd.isna(vs_avg):
            vs_avg = fallback_pts
        if pd.isna(vs_last5):
            vs_last5 = vs_avg

        home_avg = player_data[player_data["HomeAway"] == 1]["PTS"].mean()
        away_avg = player_data[player_data["HomeAway"] == 0]["PTS"].mean()
        pts_home_boost = home_avg - away_avg
        pts_homeaway_expected = home_avg if home_away == 1 else away_avg

        # Rest-based features are not currently used in core_features

        vegas_total = game_value(game, "vegas_total")
        vegas_spread = game_value(game, "vegas_spread")
        vegas_date = game_date.normalize()

        vegas_info = resolve_vegas_line(
            vegas_small_all,
            team,
            opponent,
            vegas_date,
            home_away,
            vegas_total,
            vegas_spread,
            debug_prefix="[predict_next]",
        )
        home_away = vegas_info["home_away"]
        vegas_total = vegas_info["vegas_total"]
        vegas_spread = vegas_info["vegas_spread"]
        home_team = vegas_info["home_team"]
        away_team = vegas_info["away_team"]

        if home_away is None:
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": None,
                "model": "N/A",
                "predicted_pts": None,
                "error": "home_away is required when Vegas lines are unavailable for inference",
            })
            continue

        vegas_missing = use_vegas_features and (vegas_total is None or vegas_spread is None)

        team_implied = None
        opp_implied = None
        if vegas_total is not None and vegas_spread is not None:
            home_implied = (vegas_total - vegas_spread) / 2
            away_implied = (vegas_total + vegas_spread) / 2
            team_implied = home_implied if home_away == 1 else away_implied
            opp_implied = away_implied if home_away == 1 else home_implied

        def safe_stat(col, fallback_col=None):
            val = last_row.get(col, np.nan)
            if pd.isna(val) and fallback_col and fallback_col in player_season:
                val = player_season[fallback_col].mean()
            if pd.isna(val) and col in player_season:
                val = player_season[col].mean()
            return val

        min_trend = safe_stat("MIN_trend")
        fga_rate = safe_stat("FGA_rate")
        pts_ewm = safe_stat("PTS_ewm5", fallback_col="PTS")
        fga_ewm = safe_stat("FGA_ewm5", fallback_col="FGA")

        pps = pts_ewm / (fga_ewm + 1e-5) if not (pd.isna(pts_ewm) or pd.isna(fga_ewm)) else np.nan
        if not pd.isna(pps):
            pps = float(np.clip(pps, 0.8, 1.7))

        vegas_pts_prop = (
            min_trend *
            fga_rate *
            pps *
            def_multiplier
        )
        if use_vegas_features and team_implied is not None:
            env_factor = team_implied / 113
            vegas_pts_prop = vegas_pts_prop * env_factor
        if pd.isna(vegas_pts_prop):
            fallback_pts = player_season["PTS"].mean()
            print(
                f"[predict_next] VegasPTSProp NaN for {player}; components:",
                f"MIN_trend={min_trend}, FGA_rate={fga_rate}, PTS_ewm5={pts_ewm}, FGA_ewm5={fga_ewm},",
                f"PPS={pps}, DefMultiplier={def_multiplier}, TeamImplied={team_implied}"
            )
            print(f"[predict_next] VegasPTSProp fallback to season avg {fallback_pts:.2f}")
            vegas_pts_prop = float(fallback_pts)

        raw_values = {}
        for col in norm_source_cols:
            if col in last_row:
                raw_values[col] = last_row[col]

        raw_values.update({
            "Opponent_DEF_RATING": opponent_def,
            "Opponent_PACE": opponent_pace,
            "DefMultiplier": def_multiplier,
            "vsOpp_PTS_avg": vs_avg,
            "vsOpp_PTS_last5": vs_last5,
            "PTS_homeaway_expected": pts_homeaway_expected,
            "PTS_home_boost": pts_home_boost,
            "VegasTotal": vegas_total,
            "VegasSpread": vegas_spread,
            "TeamImplied": team_implied,
            "OppImplied": opp_implied,
            "VegasPTSProp": vegas_pts_prop,
        })

        norm_values = compute_normalized_values(
            raw_values,
            player_data,
            data_sorted_full,
            norm_source_cols_full,
        )

        game_index = int(last_row["GameIndex"]) + 1
        season_max_games = int(player_season["GameIndex"].max()) + 1
        game_index_norm = game_index / max(season_max_games, 1)
        season_index = season_order.get(season)
        player_id = player_id_map.get(player)
        if season_index is None or player_id is None:
            print(f"[predict_next] Missing PlayerID or SeasonIndex for {player}")
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": "home" if home_away == 1 else "away",
                "model": "N/A",
                "predicted_pts": None,
                "error": "Missing PlayerID or SeasonIndex for prediction row",
            })
            continue

        row = {}
        for col in feature_cols:
            if col == "GameIndex_norm":
                row[col] = game_index_norm
            elif col.endswith("_norm"):
                row[col] = norm_values.get(col)
            elif col == "SeasonIndex":
                row[col] = season_index
            elif col == "PlayerID":
                row[col] = player_id
            else:
                row[col] = raw_values.get(col)

        missing_cols = [c for c in feature_cols if row.get(c) is None or (isinstance(row.get(c), float) and np.isnan(row.get(c)))]
        if missing_cols:
            non_norm_missing = [c for c in missing_cols if not c.endswith("_norm")]
            print(f"[predict_next] Missing feature cols for {player}: {missing_cols}")
            if "DefMultiplier_norm" in missing_cols:
                player_means = player_data[["DefMultiplier"]].mean(numeric_only=True)
                player_stds = player_data[["DefMultiplier"]].std(numeric_only=True)
                global_means = data_sorted_full[["DefMultiplier"]].mean(numeric_only=True)
                global_stds = data_sorted_full[["DefMultiplier"]].std(numeric_only=True)
                dm_mean = player_means.get("DefMultiplier", np.nan)
                dm_std = player_stds.get("DefMultiplier", np.nan)
                g_dm_mean = global_means.get("DefMultiplier", np.nan)
                g_dm_std = global_stds.get("DefMultiplier", np.nan)
                print(
                    "[predict_next] DefMultiplier stats:",
                    f"player_mean={dm_mean}, player_std={dm_std}, global_mean={g_dm_mean}, global_std={g_dm_std}"
                )
            if non_norm_missing:
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": "home" if home_away == 1 else "away",
                    "model": "N/A",
                    "predicted_pts": None,
                    "error": f"Missing non-normalized features: {non_norm_missing}",
                })
                continue
            # For missing normalized features, fall back to mean (0 in z-score space)
            for col in missing_cols:
                row[col] = 0.0

        X_row = pd.DataFrame([row])
        trained_models = player_models.get(player) or {}
        fallback_flags = player_model_fallbacks.get(player, {})
        for model_name in models_to_train:
            if fallback_flags.get(model_name):
                if vegas_missing:
                    print(f"[predict_next] Vegas missing for {player} ({model_name})")
                    results.append({
                        "player": player,
                        "opponent": opponent,
                        "game_date": game_value(game, "game_date"),
                        "home_away": "home" if home_away == 1 else "away",
                        "model": model_name,
                        "predicted_pts": None,
                        "vegas_total": vegas_total,
                        "vegas_spread": vegas_spread,
                        "error": "Vegas lines not found for this matchup/date",
                    })
                    continue
                if vegas_pts_prop is None or (isinstance(vegas_pts_prop, float) and np.isnan(vegas_pts_prop)):
                    results.append({
                        "player": player,
                        "opponent": opponent,
                        "game_date": game_value(game, "game_date"),
                        "home_away": "home" if home_away == 1 else "away",
                        "model": model_name,
                        "predicted_pts": None,
                        "vegas_total": vegas_total,
                        "vegas_spread": vegas_spread,
                        "error": "VegasPTSProp is missing for prediction row",
                    })
                    continue
                pred_resid = 0.0
                pred_pts = vegas_pts_prop
                print(
                    f"[predict_next] {player} {model_name}: "
                    f"train_rows={player_train_counts.get(player, 0)}, "
                    f"VegasPTSProp={vegas_pts_prop:.2f}, "
                    f"pred_resid={pred_resid:.2f} (baseline), "
                    f"pred_pts={pred_pts:.2f}"
                )
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": "home" if home_away == 1 else "away",
                    "model": model_name,
                    "predicted_pts": float(pred_pts),
                    "vegas_total": vegas_total,
                    "vegas_spread": vegas_spread,
                    "used_baseline": True,
                    "validated": False,
                })
                continue

            model = trained_models.get(model_name)
            if model is None:
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": "home" if home_away == 1 else "away",
                    "model": model_name,
                    "predicted_pts": None,
                    "error": "No training data with Vegas lines for this player",
                })
                continue
            if vegas_missing:
                print(f"[predict_next] Vegas missing for {player} ({model_name})")
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": "home" if home_away == 1 else "away",
                    "model": model_name,
                    "predicted_pts": None,
                    "vegas_total": vegas_total,
                    "vegas_spread": vegas_spread,
                    "error": "Vegas lines not found for this matchup/date",
                })
                continue
            if X_row.isna().any().any():
                print(f"[predict_next] Missing features for {player} ({model_name})")
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": "home" if home_away == 1 else "away",
                    "model": model_name,
                    "predicted_pts": None,
                    "vegas_total": vegas_total,
                    "vegas_spread": vegas_spread,
                    "error": "Missing feature values for prediction row",
                })
                continue
            pred_resid = float(model.predict(X_row)[0])
            if vegas_pts_prop is None or (isinstance(vegas_pts_prop, float) and np.isnan(vegas_pts_prop)):
                results.append({
                    "player": player,
                    "opponent": opponent,
                    "game_date": game_value(game, "game_date"),
                    "home_away": "home" if home_away == 1 else "away",
                    "model": model_name,
                    "predicted_pts": None,
                    "vegas_total": vegas_total,
                    "vegas_spread": vegas_spread,
                    "error": "VegasPTSProp is missing for prediction row",
                })
                continue
            pred_pts = vegas_pts_prop + pred_resid
            print(
                f"[predict_next] {player} {model_name}: "
                f"train_rows={player_train_counts.get(player, 0)}, "
                f"VegasPTSProp={vegas_pts_prop:.2f}, "
                f"pred_resid={pred_resid:.2f}, "
                f"pred_pts={pred_pts:.2f}"
            )
            results.append({
                "player": player,
                "opponent": opponent,
                "game_date": game_value(game, "game_date"),
                "home_away": "home" if home_away == 1 else "away",
                "model": model_name,
                "predicted_pts": float(pred_pts),
                "vegas_total": vegas_total,
                "vegas_spread": vegas_spread,
                "validated": True,
            })

    for player_name in {r.get("player") for r in results if r.get("player")}:
        candidates = [
            r for r in results
            if r.get("player") == player_name and r.get("predicted_pts") is not None
        ]
        validated = [r for r in candidates if r.get("validated")]
        if validated:
            best = validated[0]
        elif candidates:
            best = candidates[0]
        else:
            best = None
        if best:
            best_models[player_name] = {
                "model": best.get("model"),
                "predicted_pts": best.get("predicted_pts"),
            }

    print("Prediction results:", results)
    print("Best models:", best_models)
    return {"predictions": results, "best_models": best_models}
