# server/app.py
# FastAPI web server that exposes the environment as HTTP endpoints.
# Endpoints: POST /reset, POST /step, GET /state

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from environment import FailureAnalyzerEnvironment

# ── Create FastAPI app ──────────────────────────────────────
app = FastAPI(
    title="Distributed Systems Failure Analyzer",
    description=(
        "An OpenEnv RL environment where an agent analyzes service logs "
        "and identifies failures across 3 difficulty levels: easy, medium, hard."
    ),
    version="1.0.0"
)

# Allow all origins (needed for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single shared environment instance ─────────────────────
env = FailureAnalyzerEnvironment()


def model_to_dict(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


# ── Request/Response models for API ────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"  # "easy", "medium", or "hard"


class StepRequest(BaseModel):
    # Easy task fields
    service_name: Optional[str] = None
    error_code: Optional[str] = None
    # Medium task fields
    root_service: Optional[str] = None
    affected_service: Optional[str] = None
    # Hard task fields
    endpoint: Optional[str] = None
    failure_pattern: Optional[str] = None
    severity: Optional[str] = None


# ── Endpoints ───────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — returns basic info about the environment."""
    return {
        "name": "Distributed Systems Failure Analyzer",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "status": "running",
        "endpoints": ["/reset", "/step", "/state"]
    }


@app.post("/reset")
def reset(request: Optional[dict] = Body(default=None)):
    """
    Start a new episode.
    Body: {"task": "easy"} or {"task": "medium"} or {"task": "hard"}
    Returns the initial observation for the chosen task.
    """
    task = ((request or {}).get("task") or "easy")
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'easy', 'medium', or 'hard'."
        )
    try:
        obs = env.reset(task=task)
        return model_to_dict(obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """
    Submit the agent's answer and receive a reward score.
    For easy task:   {"service_name": "...", "error_code": "..."}
    For medium task: {"root_service": "...", "affected_service": "..."}
    For hard task:   {"root_service": "...", "endpoint": "...", "failure_pattern": "...", "severity": "..."}
    Returns observation with score (0.0-1.0), feedback, and done=True.
    """
    if hasattr(request, "model_dump"):
        action = request.model_dump(exclude_none=True)
    else:
        action = request.dict(exclude_none=True)

    if not action:
        raise HTTPException(
            status_code=400,
            detail="Action cannot be empty. Provide the required fields for the current task."
        )

    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": model_to_dict(obs),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """
    Get the current internal state of the environment.
    Returns: current_task, step_count, score, done, episode_id.
    """
    try:
        return model_to_dict(env.state())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """CLI entrypoint used by OpenEnv multi-mode validation."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
