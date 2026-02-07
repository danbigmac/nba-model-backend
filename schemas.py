from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    players: List[str]          # list of player names
    train_seasons: List[str]    # seasons used for training
    test_season: str            # ONE season applied to all players
    models: List[str]           # ["RandomForest", "XGBoost", ...]

class Metrics(BaseModel):
    MAE: float
    R2: float
    Baseline_MAE: float

class PlayerModelResult(BaseModel):
    player: str
    model: str
    train_seasons: List[str]
    test_season: str
    metrics: Metrics

class PredictResponse(BaseModel):
    task_id: str
    status: str   # "queued" | "running" | "done" | "failed" | "not found"
    results: List[PlayerModelResult]
    error: Optional[str] = None

class NextGameInput(BaseModel):
    player: str
    opponent: str              # NBA team abbreviation (e.g., "BOS")
    game_date: str             # YYYY-MM-DD
    home_away: Optional[str] = None  # "home" or "away" (optional if Vegas line is available)
    vegas_total: Optional[float] = None
    vegas_spread: Optional[float] = None

class PredictNextRequest(BaseModel):
    games: List[NextGameInput]
    train_seasons: List[str]
    season: str
    models: List[str]          # ["RandomForest", "XGBoost", ...]

class NextGamePrediction(BaseModel):
    player: str
    opponent: str
    game_date: str
    home_away: str
    model: str
    predicted_pts: Optional[float] = None
    vegas_total: Optional[float] = None
    vegas_spread: Optional[float] = None
    used_baseline: Optional[bool] = None
    validated: Optional[bool] = None
    error: Optional[str] = None

class PredictNextResponse(BaseModel):
    task_id: str
    status: str   # "queued" | "running" | "done" | "failed" | "not found"
    predictions: List[NextGamePrediction]
    error: Optional[str] = None
    best_models: Optional[dict] = None
