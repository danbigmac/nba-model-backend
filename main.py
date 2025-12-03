from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uuid
import reprlib, json
import time
from typing import List
from schemas import PredictRequest, PredictResponse
from nbastats import do_work
from nbacache import CacheManager

DB_FILE = "cache/nba_cache.db"

origins = [
        "http://localhost:3000",
    ]

# --- Lifespan for FastAPI app ---
# --- Creates and tears down the CacheManager ---
# --- The 'with' ensures __exit__ is called on shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing SQLite cache...")
    with CacheManager(DB_FILE) as cache_manager:
        app.state.cache_manager = cache_manager
        yield
    print("Shutting down SQLite cache...")

# --- FastAPI app ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the SQLite connection
async def get_app_cache():
    if not hasattr(app.state, 'cache_manager'):
        raise RuntimeError("Cache manager is not initialized")
    return app.state.cache_manager

def run_prediction_task(task_id: str, request: PredictRequest, cache):
    try:
        # Mark as running
        cache.execute_query(
            "UPDATE tasks SET status=? WHERE task_id=?",
            ("running", task_id)
        )
        
        # Run your blocking do_work() pipeline
        results = do_work(
            request.players,
            request.train_seasons,
            request.test_season,
            request.models,
            cache
        )
        
        # Save results as JSON
        cache.execute_query(
            "UPDATE tasks SET status=?, completed_at=?, result_json=? WHERE task_id=?",
            ("done", int(time.time()), json.dumps(results), task_id)
        )

    except Exception as e:
        error_payload = {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        cache.execute_query(
            "UPDATE tasks SET status=?, completed_at=?, result_json=? WHERE task_id=?",
            ("failed", int(time.time()), json.dumps(error_payload), task_id)
        )


# --- Endpoint ---
@app.post("/predict", response_model=dict)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks, cache: CacheManager = Depends(get_app_cache)):
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    print(f"Received prediction request. Task ID: {task_id}")
    # Store initial task in DB as queued
    cache.execute_query(
        "INSERT INTO tasks (task_id, status, created_at) VALUES (?, ?, ?)",
        (task_id, "queued", int(time.time()))
    )
    # Queue the background task
    background_tasks.add_task(run_prediction_task, task_id, request, cache)
    # Immediately return the task ID and status
    return {"task_id": task_id, "status": "queued"}

@app.get("/results/{task_id}", response_model=PredictResponse)
async def get_results(task_id: str, cache: CacheManager = Depends(get_app_cache)):
    row = cache.execute_query(
        "SELECT status, result_json FROM tasks WHERE task_id=?",
        (task_id,)
    )
    # If task_id entry not found, return response object with status "not found" and empty results
    if not row:
        return {
            "task_id": task_id,
            "status": "not found",
            "results": [],
            "error": None,
        }

    status, result_json = row[0]

    print("DEBUG result_json type:", type(result_json))
    print("DEBUG result_json repr:", reprlib.repr(result_json))
    
    # queued/running, no results yet, no error
    if status in ["queued", "running"]:
        return {
            "task_id": task_id,
            "status": status,
            "results": [],
            "error": None,
        }
    
    # nothing stored
    if not result_json:
        return {
            "task_id": task_id,
            "status": status,
            "results": [],
            "error": None,
        }

    try:
        parsed = json.loads(result_json)
    except json.JSONDecodeError as e:
        print(f"JSON decode error for task {task_id}: {e}, value={reprlib.repr(result_json)}")
        # treat as unknown error
        return {
            "task_id": task_id,
            "status": status,
            "results": [],
            "error": "Failed to parse stored results",
        }
    
    if status == "failed":
        # parsed should be {"error": "...", "error_type": "..."} from our except block
        error_msg = parsed.get("error") if isinstance(parsed, dict) else str(parsed)
        return {
            "task_id": task_id,
            "status": status,
            "results": [],
            "error": error_msg,
        }
    
    # status == "done" -> parsed should be a list of prediction dicts
    if isinstance(parsed, dict):
        # In case do_work returns a dict instead of a list for some reason
        results = [parsed]
    else:
        results = parsed

    return {
        "task_id": task_id,
        "status": status,
        "results": results,
        "error": None,
    }
