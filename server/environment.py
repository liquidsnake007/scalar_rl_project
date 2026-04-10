"""Core environment logic for the Distributed Systems Failure Analyzer tasks."""

from __future__ import annotations

import random
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

	_EASY_SCENARIOS = [
		{
			"task_description": "Identify the failing service and most likely error code from the log sample.",
			"logs": [
				{"service": "payment-api", "level": "ERROR", "message": "upstream timeout", "error_code": "TIMEOUT", "count": 198, "window_s": 60},
				{"service": "payment-api", "level": "WARN", "message": "retry budget exhausted", "count": 77, "window_s": 60},
				{"service": "catalog-api", "level": "INFO", "message": "healthy", "count": 2301, "window_s": 60},
			],
			"truth": {"service_name": "payment-api", "error_code": "TIMEOUT"},
		},
		{
			"task_description": "Identify the failing service and most likely error code from the log sample.",
			"logs": [
				{"service": "checkout-api", "level": "ERROR", "message": "null dereference in tax calc", "error_code": "NULL_POINTER", "count": 143, "window_s": 60},
				{"service": "checkout-api", "level": "WARN", "message": "fallback computation saturated", "count": 58, "window_s": 60},
				{"service": "inventory-api", "level": "INFO", "message": "healthy", "count": 1897, "window_s": 60},
			],
			"truth": {"service_name": "checkout-api", "error_code": "NULL_POINTER"},
		},
		{
			"task_description": "Identify the failing service and most likely error code from the log sample.",
			"logs": [
				{"service": "session-service", "level": "ERROR", "message": "burst throttling by upstream edge", "error_code": "RATE_LIMIT", "count": 221, "window_s": 60},
				{"service": "session-service", "level": "WARN", "message": "token refresh retries high", "count": 83, "window_s": 60},
				{"service": "user-profile", "level": "INFO", "message": "healthy", "count": 2104, "window_s": 60},
			],
			"truth": {"service_name": "session-service", "error_code": "RATE_LIMIT"},
		},
	]

	_MEDIUM_SCENARIOS = [
		{
			"task_description": "Find the root-cause service and the downstream affected service.",
			"logs": [
				{"service": "auth-service", "level": "ERROR", "message": "DB connection refused", "ratio": 0.42},
				{"service": "gateway", "level": "ERROR", "message": "503 from auth-service", "ratio": 0.37},
				{"service": "profile-service", "level": "WARN", "message": "elevated latency", "ratio": 0.12},
			],
			"timeline": [
				"12:00:10 auth-service starts failing DB_CONN_FAIL",
				"12:00:14 gateway sees surge in HTTP_503 from auth-service",
				"12:00:24 profile-service retries gateway and degrades",
			],
			"truth": {"root_service": "auth-service", "affected_service": "gateway"},
		},
		{
			"task_description": "Find the root-cause service and the downstream affected service.",
			"logs": [
				{"service": "orders-db-proxy", "level": "ERROR", "message": "pool exhausted and queue overflow", "ratio": 0.39},
				{"service": "orders-api", "level": "ERROR", "message": "upstream timeout to orders-db-proxy", "ratio": 0.34},
				{"service": "checkout-api", "level": "WARN", "message": "retry storm observed", "ratio": 0.16},
			],
			"timeline": [
				"09:41:03 orders-db-proxy connection waits spike",
				"09:41:09 orders-api starts timing out on write path",
				"09:41:16 checkout-api latency degrades due to retries",
			],
			"truth": {"root_service": "orders-db-proxy", "affected_service": "orders-api"},
		},
		{
			"task_description": "Find the root-cause service and the downstream affected service.",
			"logs": [
				{"service": "feature-store", "level": "ERROR", "message": "read replicas lagging > 3s", "ratio": 0.31},
				{"service": "ranking", "level": "ERROR", "message": "stale feature fetch failures", "ratio": 0.28},
				{"service": "search-api", "level": "WARN", "message": "tail latency rising", "ratio": 0.15},
			],
			"timeline": [
				"17:05:12 feature-store replication delay crosses threshold",
				"17:05:18 ranking emits stale-read errors",
				"17:05:26 search-api quality and latency regress",
			],
			"truth": {"root_service": "feature-store", "affected_service": "ranking"},
		},
	]

	_HARD_SCENARIOS = [
		{
			"task_description": "Identify subtle root cause: service, endpoint, failure pattern, and severity.",
			"logs": [
				{"service": "search-api", "endpoint": "/api/v1/query", "p95_ms": 2120, "error_ratio": 0.08},
				{"service": "cache-router", "endpoint": "/lookup", "p95_ms": 120, "error_ratio": 0.01},
				{"service": "ranking", "endpoint": "/rank", "p95_ms": 190, "error_ratio": 0.02},
			],
			"trace_graph": {
				"nodes": ["frontend", "search-api", "cache-router", "ranking", "feature-store"],
				"edges": [
					["frontend", "search-api"],
					["search-api", "cache-router"],
					["search-api", "ranking"],
					["ranking", "feature-store"],
				],
			},
			"timeline": [
				"13:10 search-api timeout spike appears only on /api/v1/query",
				"13:18 cache hit ratio drops and connection waits increase",
				"13:22 search-api worker saturation returns after pool recycle",
			],
			"truth": {
				"root_service": "search-api",
				"endpoint": "/api/v1/query",
				"failure_pattern": "connection_pool_exhausted",
				"severity": "high",
			},
		},
		{
			"task_description": "Identify subtle root cause: service, endpoint, failure pattern, and severity.",
			"logs": [
				{"service": "billing-api", "endpoint": "/api/v2/charge", "p95_ms": 1640, "error_ratio": 0.07},
				{"service": "fraud-check", "endpoint": "/risk-score", "p95_ms": 210, "error_ratio": 0.02},
				{"service": "ledger-writer", "endpoint": "/append", "p95_ms": 185, "error_ratio": 0.01},
			],
			"trace_graph": {
				"nodes": ["frontend", "billing-api", "fraud-check", "ledger-writer", "event-bus"],
				"edges": [
					["frontend", "billing-api"],
					["billing-api", "fraud-check"],
					["billing-api", "ledger-writer"],
					["ledger-writer", "event-bus"],
				],
			},
			"timeline": [
				"21:44 intermittent timeout bursts begin on billing-api /api/v2/charge",
				"21:49 retry volume rises without sustained 5xx trend",
				"21:55 p95 remains unstable with transient queue drain events",
			],
			"truth": {
				"root_service": "billing-api",
				"endpoint": "/api/v2/charge",
				"failure_pattern": "intermittent_timeout",
				"severity": "medium",
			},
		},
		{
			"task_description": "Identify subtle root cause: service, endpoint, failure pattern, and severity.",
			"logs": [
				{"service": "stream-ingest", "endpoint": "/events/push", "p95_ms": 930, "error_ratio": 0.06},
				{"service": "schema-registry", "endpoint": "/schema/resolve", "p95_ms": 310, "error_ratio": 0.02},
				{"service": "consumer-router", "endpoint": "/dispatch", "p95_ms": 240, "error_ratio": 0.01},
			],
			"trace_graph": {
				"nodes": ["edge", "stream-ingest", "schema-registry", "consumer-router", "warehouse"],
				"edges": [
					["edge", "stream-ingest"],
					["stream-ingest", "schema-registry"],
					["stream-ingest", "consumer-router"],
					["consumer-router", "warehouse"],
				],
			},
			"timeline": [
				"02:10 stream-ingest error ratio climbs from 0.5% to 6%",
				"02:14 memory usage in stream-ingest increases steadily",
				"02:19 pod recycle temporarily restores throughput",
			],
			"truth": {
				"root_service": "stream-ingest",
				"endpoint": "/events/push",
				"failure_pattern": "memory_leak",
				"severity": "high",
			},
		},
	]

	def __init__(self) -> None:
		self.current_task = "easy"
		self.step_count = 0
		self.score = 0.0
		self.done = False
		self.episode_id = self._new_episode_id()
		self.seed: int | None = None
		self._ground_truth: Dict[str, Any] = {}
		self._last_observation: EasyObservation | MediumObservation | HardObservation | None = None

	@staticmethod
	def _new_episode_id() -> str:
		return str(uuid.uuid4())

	@classmethod
	def _strict_score(cls, score: float) -> float:
		"""Clamp score to strict open interval (0, 1) required by evaluator."""
		return max(cls._MIN_SCORE, min(cls._MAX_SCORE, float(score)))

	def _choose_scenario(self, scenarios: list[Dict[str, Any]], seed: int | None) -> Dict[str, Any]:
		if seed is None:
			return random.choice(scenarios)
		return scenarios[random.Random(seed).randrange(len(scenarios))]

	def _easy_payload(self, seed: int | None) -> Tuple[EasyObservation, Dict[str, str]]:
		scenario = self._choose_scenario(self._EASY_SCENARIOS, seed)
		obs = EasyObservation(
			task_description=scenario["task_description"],
			logs=scenario["logs"],
		)
		return obs, deepcopy(scenario["truth"])

	def _medium_payload(self, seed: int | None) -> Tuple[MediumObservation, Dict[str, str]]:
		scenario = self._choose_scenario(self._MEDIUM_SCENARIOS, seed)
		obs = MediumObservation(
			task_description=scenario["task_description"],
			logs=scenario["logs"],
			timeline=scenario["timeline"],
		)
		return obs, deepcopy(scenario["truth"])

	def _hard_payload(self, seed: int | None) -> Tuple[HardObservation, Dict[str, str]]:
		scenario = self._choose_scenario(self._HARD_SCENARIOS, seed)
		obs = HardObservation(
			task_description=scenario["task_description"],
			logs=scenario["logs"],
			trace_graph=scenario["trace_graph"],
			timeline=scenario["timeline"],
		)
		return obs, deepcopy(scenario["truth"])

	def reset(self, task: str = "easy", seed: int | None = None) -> EasyObservation | MediumObservation | HardObservation:
		task_name = (task or "easy").strip().lower()
		if task_name not in {"easy", "medium", "hard"}:
			raise ValueError("Invalid task. Must be one of: easy, medium, hard")

		self.current_task = task_name
		self.step_count = 0
		self.score = 0.0
		self.done = False
		self.episode_id = self._new_episode_id()
		self.seed = seed

		if task_name == "easy":
			obs, truth = self._easy_payload(seed)
		elif task_name == "medium":
			obs, truth = self._medium_payload(seed)
		else:
			obs, truth = self._hard_payload(seed)

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
