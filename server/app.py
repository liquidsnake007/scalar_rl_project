# server/app.py
# FastAPI web server that exposes the environment as HTTP endpoints.
# Endpoints: POST /reset, POST /step, GET /state

from typing import Optional

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import uvicorn

from server.environment import FailureAnalyzerEnvironment

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

# ── Session-scoped environment instances ───────────────────
envs: dict[str, FailureAnalyzerEnvironment] = {}


def get_env(session_id: str) -> FailureAnalyzerEnvironment:
    key = session_id.strip() or "default"
    if key not in envs:
        envs[key] = FailureAnalyzerEnvironment()
    return envs[key]


def model_to_dict(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


# ── Request/Response models for API ────────────────────────

class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Optional[str] = "easy"  # "easy", "medium", or "hard"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Optional[str] = None
    target_service: Optional[str] = None
    note: Optional[str] = None
    mitigation_action: Optional[str] = None

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
def reset(request: ResetRequest = Body(default_factory=ResetRequest), x_session_id: Optional[str] = Header(default="default", alias="X-Session-ID")):
    """
    Start a new episode.
    Body: {"task": "easy"} or {"task": "medium"} or {"task": "hard"}
    Returns the initial observation for the chosen task.
    """
    task = (request.task or "easy")
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'easy', 'medium', or 'hard'."
        )
    try:
        env = get_env(x_session_id or "default")
        obs = env.reset(task=task, seed=request.seed)
        return model_to_dict(obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest, x_session_id: Optional[str] = Header(default="default", alias="X-Session-ID")):
    """
    Submit an investigation action and receive shaped reward signal.
    Action types: inspect_logs, inspect_timeline, inspect_trace,
    submit_hypothesis, apply_mitigation, finalize.
    If action_type is omitted and final answer fields are present,
    the action is treated as finalize for backward compatibility.
    For easy task:   {"service_name": "...", "error_code": "..."}
    For medium task: {"root_service": "...", "affected_service": "..."}
    For hard task:   {"root_service": "...", "endpoint": "...", "failure_pattern": "...", "severity": "..."}
    Returns observation with progress score, per-step reward, and done flag.
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
        env = get_env(x_session_id or "default")
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
def state(x_session_id: Optional[str] = Header(default="default", alias="X-Session-ID")):
    """
    Get the current internal state of the environment.
    Returns: current_task, step_count, score, done, episode_id.
    """
    try:
        env = get_env(x_session_id or "default")
        return model_to_dict(env.state())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """CLI entrypoint used by OpenEnv multi-mode validation."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
