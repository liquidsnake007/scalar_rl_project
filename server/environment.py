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
	"""Multi-step scoring environment with evidence gathering and final diagnosis."""

	_MIN_SCORE = 0.01
	_MAX_SCORE = 0.99
	_MAX_STEPS = {"easy": 4, "medium": 5, "hard": 6}
	_REQUIRED_EVIDENCE = {
		"easy": {"logs"},
		"medium": {"logs", "timeline"},
		"hard": {"logs", "timeline", "trace_graph"},
	}
	_UNSAFE_KEYWORDS = ("drop", "delete", "kill", "restart", "shutdown", "truncate")

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
		self.max_steps = self._MAX_STEPS["easy"]
		self._ground_truth: Dict[str, Any] = {}
		self._scenario: Dict[str, Any] = {}
		self._revealed_sections: set[str] = set()
		self._inspected_targets: set[str] = set()
		self._hypotheses: set[str] = set()
		self._penalties: list[str] = []
		self._reward_history: list[float] = []
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

	def _easy_payload(self, seed: int | None) -> Dict[str, Any]:
		return deepcopy(self._choose_scenario(self._EASY_SCENARIOS, seed))

	def _medium_payload(self, seed: int | None) -> Dict[str, Any]:
		return deepcopy(self._choose_scenario(self._MEDIUM_SCENARIOS, seed))

	def _hard_payload(self, seed: int | None) -> Dict[str, Any]:
		return deepcopy(self._choose_scenario(self._HARD_SCENARIOS, seed))

	def _looks_like_final(self, action: Dict[str, Any]) -> bool:
		if self.current_task == "easy":
			return {"service_name", "error_code"}.issubset(set(action.keys()))
		if self.current_task == "medium":
			return {"root_service", "affected_service"}.issubset(set(action.keys()))
		return {"root_service", "endpoint", "failure_pattern", "severity"}.issubset(set(action.keys()))

	def _available_actions(self) -> list[str]:
		actions = ["inspect_logs", "submit_hypothesis", "finalize", "apply_mitigation"]
		if self.current_task in {"medium", "hard"}:
			actions.insert(1, "inspect_timeline")
		if self.current_task == "hard":
			actions.insert(2, "inspect_trace")
		return actions

	def _safe_graph(self) -> Dict[str, Any]:
		if "trace_graph" in self._revealed_sections:
			return deepcopy(self._scenario.get("trace_graph", {}))
		return {"nodes": [], "edges": []}

	def _safe_timeline(self) -> list[str]:
		if "timeline" in self._revealed_sections:
			return deepcopy(self._scenario.get("timeline", []))
		return []

	def _build_observation(
		self,
		feedback: str,
		done: bool,
	) -> EasyObservation | MediumObservation | HardObservation:
		base = {
			"task": self.current_task,
			"task_description": self._scenario.get("task_description", ""),
			"logs": deepcopy(self._scenario.get("logs", [])),
			"step_index": self.step_count,
			"max_steps": self.max_steps,
			"available_actions": [] if done else self._available_actions(),
			"revealed_sections": sorted(self._revealed_sections),
			"penalties": deepcopy(self._penalties),
			"score": self.score,
			"feedback": feedback,
			"done": done,
		}
		if self.current_task == "easy":
			return EasyObservation(**base)
		if self.current_task == "medium":
			return MediumObservation(**(base | {"timeline": self._safe_timeline()}))
		return HardObservation(
			**(
				base
				| {
					"timeline": self._safe_timeline(),
					"trace_graph": self._safe_graph(),
				}
			)
		)

	def _record_reward(self, reward: float) -> float:
		r = self._strict_score(reward)
		self._reward_history.append(r)
		self.score = self._strict_score(sum(self._reward_history) / len(self._reward_history))
		return r

	def _evidence_ratio(self) -> float:
		required = self._REQUIRED_EVIDENCE[self.current_task]
		return len(required.intersection(self._revealed_sections)) / len(required)

	def _apply_penalty(self, name: str) -> None:
		self._penalties.append(name)

	def _handle_finalize(self, action: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
		final_payload = {
			k: v
			for k, v in action.items()
			if k not in {"action_type", "target_service", "note", "mitigation_action"}
		}
		try:
			if self.current_task == "easy":
				parsed = EasyAction(**final_payload)
				base_score, base_feedback = self._score_easy(parsed)
			elif self.current_task == "medium":
				parsed = MediumAction(**final_payload)
				base_score, base_feedback = self._score_medium(parsed)
			else:
				parsed = HardAction(**final_payload)
				base_score, base_feedback = self._score_hard(parsed)
		except ValidationError as exc:
			self._apply_penalty("invalid_finalize_payload")
			info = {"error": str(exc), "penalties": deepcopy(self._penalties)}
			done = self.step_count >= self.max_steps
			reward = self._record_reward(0.08)
			feedback = "Invalid final answer payload. Provide required fields for this task."
			return reward, feedback, done, info

		evidence_ratio = self._evidence_ratio()
		efficiency_bonus = max(0.0, (self.max_steps - self.step_count) / self.max_steps) * 0.08
		penalty_cost = min(0.24, 0.06 * len(self._penalties))
		final_reward = (base_score * 0.72) + (evidence_ratio * 0.20) + efficiency_bonus - penalty_cost
		reward = self._record_reward(final_reward)
		feedback = f"{base_feedback} Evidence coverage={evidence_ratio:.2f}."
		info = {"error": None, "penalties": deepcopy(self._penalties), "evidence_ratio": evidence_ratio}
		return reward, feedback, True, info

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
		self.max_steps = self._MAX_STEPS[task_name]
		self._revealed_sections = {"logs"}
		self._inspected_targets = set()
		self._hypotheses = set()
		self._penalties = []
		self._reward_history = []

		if task_name == "easy":
			scenario = self._easy_payload(seed)
		elif task_name == "medium":
			scenario = self._medium_payload(seed)
		else:
			scenario = self._hard_payload(seed)

		self._scenario = scenario
		self._ground_truth = deepcopy(scenario["truth"])
		obs = self._build_observation(feedback="Episode reset. Gather evidence before finalizing diagnosis.", done=False)
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
		info: Dict[str, Any] = {"error": None, "penalties": deepcopy(self._penalties)}

		action_type = str(action.get("action_type", "")).strip().lower()
		if not action_type:
			action_type = "finalize" if self._looks_like_final(action) else "inspect_logs"

		done = False
		feedback = ""
		if action_type == "finalize":
			reward, feedback, done, finalize_info = self._handle_finalize(action)
			info.update(finalize_info)
		elif action_type == "inspect_logs":
			target = str(action.get("target_service", "")).strip().lower()
			if target and target in self._inspected_targets:
				self._apply_penalty("redundant_log_probe")
				reward = self._record_reward(0.12)
				feedback = "Redundant log inspection. Probe a new service or finalize."
			else:
				if target:
					self._inspected_targets.add(target)
				reward = self._record_reward(0.62)
				feedback = "Useful evidence gathered from logs."
		elif action_type == "inspect_timeline":
			if self.current_task == "easy":
				self._apply_penalty("invalid_timeline_probe")
				reward = self._record_reward(0.08)
				feedback = "Timeline is not available for easy task."
			elif "timeline" in self._revealed_sections:
				self._apply_penalty("redundant_timeline_probe")
				reward = self._record_reward(0.14)
				feedback = "Timeline already revealed; avoid repeated probes."
			else:
				self._revealed_sections.add("timeline")
				reward = self._record_reward(0.66)
				feedback = "Timeline revealed."
		elif action_type == "inspect_trace":
			if self.current_task != "hard":
				self._apply_penalty("invalid_trace_probe")
				reward = self._record_reward(0.08)
				feedback = "Trace graph is only available for hard task."
			elif "trace_graph" in self._revealed_sections:
				self._apply_penalty("redundant_trace_probe")
				reward = self._record_reward(0.14)
				feedback = "Trace graph already revealed; avoid repeated probes."
			else:
				self._revealed_sections.add("trace_graph")
				reward = self._record_reward(0.68)
				feedback = "Trace graph revealed."
		elif action_type == "submit_hypothesis":
			note = str(action.get("note", "")).strip().lower()
			if len(note) < 10:
				self._apply_penalty("low_quality_hypothesis")
				reward = self._record_reward(0.18)
				feedback = "Hypothesis too weak. Add concrete causal reasoning."
			elif note in self._hypotheses:
				self._apply_penalty("repeated_hypothesis")
				reward = self._record_reward(0.16)
				feedback = "Repeated hypothesis. Provide a new diagnostic angle."
			else:
				self._hypotheses.add(note)
				reward = self._record_reward(0.58)
				feedback = "Hypothesis recorded."
		elif action_type == "apply_mitigation":
			mitigation = str(action.get("mitigation_action", "")).strip().lower()
			if any(word in mitigation for word in self._UNSAFE_KEYWORDS):
				self._apply_penalty("unsafe_mitigation")
				reward = self._record_reward(0.03)
				feedback = "Unsafe mitigation attempt penalized. Diagnose first, then act safely."
			else:
				self._apply_penalty("premature_mitigation")
				reward = self._record_reward(0.22)
				feedback = "Mitigation without strong evidence is discouraged."
		else:
			self._apply_penalty("unknown_action_type")
			reward = self._record_reward(0.08)
			feedback = f"Unknown action_type '{action_type}'."

		if not done and self.step_count >= self.max_steps:
			evidence_ratio = self._evidence_ratio()
			penalty_cost = min(0.24, 0.06 * len(self._penalties))
			reward = self._record_reward(0.18 + (0.25 * evidence_ratio) - penalty_cost)
			feedback = "Episode ended due to step limit before final diagnosis."
			done = True
			info["error"] = info.get("error") or "max_steps_reached"

		self.done = done
		obs = self._build_observation(feedback=feedback, done=done)
		self._last_observation = obs
		info["penalties"] = deepcopy(self._penalties)
		return deepcopy(obs), reward, done, info

	def state(self) -> StateModel:
		return StateModel(
			current_task=self.current_task,
			step_count=self.step_count,
			score=self.score,
			done=self.done,
			episode_id=self.episode_id,
		)
