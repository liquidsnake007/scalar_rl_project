"""Core environment logic for the Distributed Systems Failure Analyzer tasks."""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, Tuple

from pydantic import ValidationError

from models import (
	EasyAction,
	EasyObservation,
	HardAction,
	HardObservation,
	MediumAction,
	MediumObservation,
	StateModel,
)


class FailureAnalyzerEnvironment:
	"""Single-step scoring environment with easy/medium/hard diagnostics tasks."""

	_MIN_SCORE = 0.01
	_MAX_SCORE = 0.99

	def __init__(self) -> None:
		self.current_task = "easy"
		self.step_count = 0
		self.score = 0.0
		self.done = False
		self.episode_id = self._new_episode_id()
		self._ground_truth: Dict[str, Any] = {}
		self._last_observation: EasyObservation | MediumObservation | HardObservation | None = None

	@staticmethod
	def _new_episode_id() -> str:
		return str(uuid.uuid4())

	@classmethod
	def _strict_score(cls, score: float) -> float:
		"""Clamp score to open interval (0, 1) required by the evaluator."""
		return max(cls._MIN_SCORE, min(cls._MAX_SCORE, float(score)))

	@staticmethod
	def _easy_payload() -> Tuple[EasyObservation, Dict[str, str]]:
		obs = EasyObservation(
			task_description=(
				"Identify the failing service and most likely error code from the log sample."
			),
			logs=[
				{"service": "payment-api", "level": "ERROR", "message": "upstream timeout", "error_code": "TIMEOUT", "count": 198, "window_s": 60},
				{"service": "payment-api", "level": "WARN", "message": "retry budget exhausted", "count": 77, "window_s": 60},
				{"service": "catalog-api", "level": "INFO", "message": "healthy", "count": 2301, "window_s": 60},
			],
		)
		return obs, {"service_name": "payment-api", "error_code": "TIMEOUT"}

	@staticmethod
	def _medium_payload() -> Tuple[MediumObservation, Dict[str, str]]:
		obs = MediumObservation(
			task_description="Find the root-cause service and the downstream affected service.",
			logs=[
				{"service": "auth-service", "level": "ERROR", "message": "DB connection refused", "ratio": 0.42},
				{"service": "gateway", "level": "ERROR", "message": "503 from auth-service", "ratio": 0.37},
				{"service": "profile-service", "level": "WARN", "message": "elevated latency", "ratio": 0.12},
			],
			timeline=[
				"12:00:10 auth-service starts failing DB_CONN_FAIL",
				"12:00:14 gateway sees surge in HTTP_503 from auth-service",
				"12:00:24 profile-service retries gateway and degrades",
			],
		)
		return obs, {"root_service": "auth-service", "affected_service": "gateway"}

	@staticmethod
	def _hard_payload() -> Tuple[HardObservation, Dict[str, str]]:
		obs = HardObservation(
			task_description=(
				"Identify subtle root cause: service, endpoint, failure pattern, and severity."
			),
			logs=[
				{"service": "search-api", "endpoint": "/api/v1/query", "p95_ms": 2120, "error_ratio": 0.08},
				{"service": "cache-router", "endpoint": "/lookup", "p95_ms": 120, "error_ratio": 0.01},
				{"service": "ranking", "endpoint": "/rank", "p95_ms": 190, "error_ratio": 0.02},
			],
			trace_graph={
				"nodes": ["frontend", "search-api", "cache-router", "ranking", "feature-store"],
				"edges": [
					["frontend", "search-api"],
					["search-api", "cache-router"],
					["search-api", "ranking"],
					["ranking", "feature-store"],
				],
			},
			timeline=[
				"13:10 search-api timeout spike appears only on /api/v1/query",
				"13:18 cache hit ratio drops and connection waits increase",
				"13:22 search-api worker saturation returns after pool recycle",
			],
		)
		return obs, {
			"root_service": "search-api",
			"endpoint": "/api/v1/query",
			"failure_pattern": "connection_pool_exhausted",
			"severity": "high",
		}

	def reset(self, task: str = "easy") -> EasyObservation | MediumObservation | HardObservation:
		task_name = (task or "easy").strip().lower()
		if task_name not in {"easy", "medium", "hard"}:
			raise ValueError("Invalid task. Must be one of: easy, medium, hard")

		self.current_task = task_name
		self.step_count = 0
		self.score = 0.0
		self.done = False
		self.episode_id = self._new_episode_id()

		if task_name == "easy":
			obs, truth = self._easy_payload()
		elif task_name == "medium":
			obs, truth = self._medium_payload()
		else:
			obs, truth = self._hard_payload()

		self._ground_truth = truth
		self._last_observation = obs
		return deepcopy(obs)

	def _score_easy(self, action: EasyAction) -> Tuple[float, str]:
		service_ok = action.service_name.strip().lower() == self._ground_truth["service_name"].lower()
		code_ok = action.error_code.strip().upper() == self._ground_truth["error_code"].upper()
		if service_ok and code_ok:
			return self._strict_score(1.0), "Correct service and error code."
		if service_ok or code_ok:
			return self._strict_score(0.5), "Partially correct."
		return self._strict_score(0.0), "Incorrect diagnosis."

	def _score_medium(self, action: MediumAction) -> Tuple[float, str]:
		root_ok = action.root_service.strip().lower() == self._ground_truth["root_service"].lower()
		affected_ok = action.affected_service.strip().lower() == self._ground_truth["affected_service"].lower()
		if root_ok and affected_ok:
			return self._strict_score(1.0), "Correct root-cause chain."
		if root_ok or affected_ok:
			return self._strict_score(0.5), "Partially correct cascade mapping."
		return self._strict_score(0.0), "Incorrect root-cause chain."

	def _score_hard(self, action: HardAction) -> Tuple[float, str]:
		checks = [
			action.root_service.strip().lower() == self._ground_truth["root_service"].lower(),
			action.endpoint.strip() == self._ground_truth["endpoint"],
			action.failure_pattern.strip().lower() == self._ground_truth["failure_pattern"].lower(),
			action.severity.strip().lower() == self._ground_truth["severity"].lower(),
		]
		score = round(sum(1 for ok in checks if ok) / 4.0, 2)
		if score == 1.0:
			return self._strict_score(1.0), "Correct complex failure diagnosis."
		if score >= 0.5:
			return self._strict_score(score), "Mostly correct complex diagnosis."
		return self._strict_score(score), "Diagnosis needs improvement."

	def step(self, action: Dict[str, Any]) -> Tuple[EasyObservation | MediumObservation | HardObservation, float, bool, Dict[str, Any]]:
		if self._last_observation is None:
			raise RuntimeError("Call reset() before step().")

		if self.done:
			return deepcopy(self._last_observation), self.score, True, {"error": "episode_done"}

		self.step_count += 1
		info: Dict[str, Any] = {"error": None}

		try:
			if self.current_task == "easy":
				parsed = EasyAction(**action)
				score, feedback = self._score_easy(parsed)
				obs = EasyObservation(
					task="easy",
					task_description=self._last_observation.task_description,
					logs=self._last_observation.logs,
					score=score,
					feedback=feedback,
					done=True,
				)
			elif self.current_task == "medium":
				parsed = MediumAction(**action)
				score, feedback = self._score_medium(parsed)
				obs = MediumObservation(
					task="medium",
					task_description=self._last_observation.task_description,
					logs=self._last_observation.logs,
					timeline=self._last_observation.timeline,
					score=score,
					feedback=feedback,
					done=True,
				)
			else:
				parsed = HardAction(**action)
				score, feedback = self._score_hard(parsed)
				obs = HardObservation(
					task="hard",
					task_description=self._last_observation.task_description,
					logs=self._last_observation.logs,
					trace_graph=self._last_observation.trace_graph,
					timeline=self._last_observation.timeline,
					score=score,
					feedback=feedback,
					done=True,
				)
		except ValidationError as exc:
			info["error"] = str(exc)
			score = self._strict_score(0.0)
			feedback = "Invalid action payload."
			base = deepcopy(self._last_observation)
			base.score = score
			base.feedback = feedback
			base.done = True
			obs = base

		score = self._strict_score(score)
		self.score = score
		self.done = True
		self._last_observation = obs
		return deepcopy(obs), score, True, info

	def state(self) -> StateModel:
		return StateModel(
			current_task=self.current_task,
			step_count=self.step_count,
			score=self.score,
			done=self.done,
			episode_id=self.episode_id,
		)
