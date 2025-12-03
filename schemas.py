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