# models.py
# All Pydantic models defining what the agent sends and receives.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────
# ACTIONS — what the agent sends
# ─────────────────────────────────────────────

class EasyAction(BaseModel):
    """
    Task 1 (Easy): 1 service, 99% error rate, full clean logs.
    Agent must identify the failing service and error type.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    service_name: str = Field(..., description="Name of the failing service e.g. 'payment-api'")
    error_code: str = Field(..., description="Error type e.g. 'HTTP_500', 'TIMEOUT', 'DB_CONN_FAIL'")


class MediumAction(BaseModel):
    """
    Task 2 (Medium): 3 services, 40% error rate, partial logs.
    Agent must identify root cause service and which service it affected.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    root_service: str = Field(..., description="The service where the problem started")
    affected_service: str = Field(..., description="The downstream service being impacted")


class HardAction(BaseModel):
    """
    Task 3 (Hard): 6 services, 8% intermittent errors, noisy logs with red herrings.
    Agent must identify root service, exact endpoint, failure pattern, and severity.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    root_service: str = Field(..., description="Root cause service name")
    endpoint: str = Field(..., description="Exact endpoint causing degradation e.g. '/api/v1/query'")
    failure_pattern: Literal[
        "intermittent_timeout",
        "error_spike",
        "memory_leak",
        "connection_pool_exhausted",
        "cascading_crash",
    ] = Field(..., description="One of: intermittent_timeout, error_spike, memory_leak, connection_pool_exhausted, cascading_crash")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="One of: low, medium, high, critical")


# ─────────────────────────────────────────────
# OBSERVATIONS — what the environment returns
# ─────────────────────────────────────────────

class EasyObservation(BaseModel):
    task: str = Field(default="easy")
    task_description: str = Field(..., description="What the agent needs to do")
    logs: list = Field(..., description="Log entries from one service with obvious error spike")
    score: float = Field(default=0.0)
    feedback: str = Field(default="")
    done: bool = Field(default=False)


class MediumObservation(BaseModel):
    task: str = Field(default="medium")
    task_description: str = Field(..., description="What the agent needs to do")
    logs: list = Field(..., description="Partial logs from 3 services, some entries missing")
    timeline: list = Field(..., description="Ordered timeline of events")
    score: float = Field(default=0.0)
    feedback: str = Field(default="")
    done: bool = Field(default=False)


class HardObservation(BaseModel):
    task: str = Field(default="hard")
    task_description: str = Field(..., description="What the agent needs to do")
    logs: list = Field(..., description="Noisy logs from 6 services with red herrings")
    trace_graph: dict = Field(..., description="Distributed trace showing service call graph")
    timeline: list = Field(..., description="Full timeline across all 6 services")
    score: float = Field(default=0.0)
    feedback: str = Field(default="")
    done: bool = Field(default=False)


class StateModel(BaseModel):
    current_task: str
    step_count: int
    score: float
    done: bool
    episode_id: str
