"""Inference-time Qwen Flow-Director loop over the strict AgentGraph Canvas."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import socket
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStepResult
from .model_registry import ModelRegistry


DIRECTOR_SYSTEM_PROMPT = """You are the Flow-Director. Build an executable AgentGraph for the task, one edit at a time. Follow the latest Canvas feedback and return exactly one JSON object each turn.

Actions:
{"action":"add_agent","agent_id":"...","model_id":"...","contract":"..."}
{"action":"modify_agent","agent_id":"...","model_id":"...","contract":"..."}
{"action":"delete_agent","agent_id":"..."}
{"action":"set_relation","source_id":"...","target_id":"...","source_to_target":true,"target_to_source":false}
{"action":"set_output","agent_id":"..."}
{"action":"finish"}

Use a model_id from the supplied catalog and describe each Agent's job in concise ordinary free text. A useful contract states its objective, expected input or dependency, artifact to produce, and completion condition; do not prefill an upstream result that has not been produced. A relation's two booleans are the two message directions; no relation means independent work, and a bidirectional pair performs one finite draft-and-revision exchange. Choose decomposition and models only when the task needs them, not to make the graph larger.

Only the graph's Output Agent owns the final task answer; other Agents produce intermediate artifacts. Finish only after the Canvas accepts a complete graph. When the latest execution is format-valid and shows no concrete defect, prefer finish. Continue only to address a specific missing evidence hop, unresolved dependency, conflict, format error, execution error, or task mismatch; unused rounds or another catalog model alone are not reasons to edit."""


class DirectorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DirectorResponse:
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DirectorClient(Protocol):
    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
    ) -> DirectorResponse:
        ...


class OpenAIDirectorClient:
    """OpenAI-compatible chat client for the local Qwen3.5-9B endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8015/v1",
        model: str = "supervisor_theta",
        api_key_env: Optional[str] = None,
        policy_version: str = "qwen3.5-9b-sglang-unversioned",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 768,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be absolute HTTP(S)")
        if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Flow-Director must use the local Qwen3.5-9B endpoint")
        if model != "supervisor_theta":
            raise ValueError("Flow-Director model must be supervisor_theta")
        if not model.strip() or not policy_version.strip():
            raise ValueError("model and policy_version must be non-empty")
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Director temperature/top_p are invalid")
        if max_tokens <= 0 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Director token, timeout, and retry limits are invalid")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.policy_version = policy_version
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)

    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
    ) -> DirectorResponse:
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("Director seed must be a non-negative integer or None")
        api_key = "EMPTY"
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env, "")
            if not api_key:
                raise DirectorError(f"missing Director credential environment variable: {self.api_key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            # SkillFlow sends the generation seed through the provider payload.
            payload["seed"] = seed
        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                value = await asyncio.to_thread(self._post, api_key, payload)
                parsed = self._parse(value)
                metadata = dict(parsed.metadata)
                metadata.update(
                    {
                        "latency_ms": max(
                            (time.monotonic() - started_at) * 1000.0,
                            0.0,
                        ),
                        "attempt_count": attempt + 1,
                        "generation_seed": seed,
                    }
                )
                return DirectorResponse(parsed.text, metadata)
            except HTTPError as exc:
                last_error = exc
                if not (exc.code in {408, 409, 425, 429} or exc.code >= 500):
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
            if attempt < self.max_retries:
                await asyncio.sleep(min(2.0**attempt, 4.0))
        detail = f"HTTP {last_error.code}" if isinstance(last_error, HTTPError) else type(last_error).__name__
        raise DirectorError(f"Director request failed: {detail}") from last_error

    def _post(self, api_key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-Director/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise DirectorError("Director returned a non-object response")
        return value

    def _parse(self, value: Mapping[str, Any]) -> DirectorResponse:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise DirectorError("Director response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise DirectorError("Director response has no text content")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return DirectorResponse(
            text=message["content"],
            metadata={
                "policy_version": self.policy_version,
                "model": value.get("model", self.model),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "request_id": value.get("id"),
            },
        )


@dataclass(frozen=True, slots=True)
class DirectorTurn:
    turn_index: int
    prompt: str
    response: DirectorResponse
    canvas_result: AgentWorkflowStepResult


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    final_answer: Optional[str]
    turns: Tuple[DirectorTurn, ...]
    final_graph: Mapping[str, Any]
    termination_reason: str
    explicit_finish: bool


class AgentGraphOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        client: DirectorClient,
        *,
        max_rounds: int = 20,
        seed: int = 42,
        history_window: int = 4,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if isinstance(history_window, bool) or not isinstance(history_window, int) or history_window < 1:
            raise ValueError("history_window must be a positive integer")
        self.registry = registry
        self.client = client
        self.max_rounds = max_rounds
        self.seed = seed
        self.history_window = history_window

    def build_prompt(
        self,
        env: AgentWorkflowEnv,
        turn_index: int,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        catalog = [
            {
                "model_id": model_id,
                "selection_weight": self.registry.require_model(model_id).selection_weight,
                "cheap_weight": self.registry.require_model(model_id).cheap_weight,
                "fast_weight": self.registry.require_model(model_id).fast_weight,
                "routing_metadata": {
                    key: value
                    for key, value in self.registry.require_model(model_id).metadata.items()
                    if key
                    in {
                        "family",
                        "profile",
                        "text_qa_canary",
                        "canary_source",
                    }
                },
            }
            for model_id in self.registry.model_ids
        ]
        complete_validation = env.graph.validate(
            self.registry,
            require_complete=True,
        )
        snapshot = env.snapshot()
        payload = {
            "task": env.problem,
            "turn": turn_index,
            "max_rounds": self.max_rounds,
            "remaining_rounds": max(self.max_rounds - env.turn_count, 0),
            "current_graph": env.graph.to_dict(),
            "topology_statistics": env.graph.topology_statistics(),
            "canvas_feedback": snapshot.last_feedback,
            # SkillFlow presents a bounded visible action-history tail to its
            # ReAct policy; keep the same boundary without adding role recipes.
            "recent_canvas_history": [
                entry.to_dict() for entry in snapshot.history[-self.history_window :]
            ],
            "complete_validation": {
                "valid": complete_validation.valid,
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in complete_validation.issues
                ],
            },
            "model_catalog": catalog,
        }
        if env.max_agents is not None:
            payload["max_agents"] = env.max_agents
        if skills:
            payload["available_skills"] = list(skills)
        return (
            "Choose exactly one next action. Use only observed task, Canvas, and catalog facts.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    async def run(
        self,
        env: AgentWorkflowEnv,
        problem: str,
        *,
        skills: Sequence[Mapping[str, Any]] = (),
    ) -> OrchestrationResult:
        env.reset(problem)
        turns: list[DirectorTurn] = []
        for index in range(self.max_rounds):
            prompt = self.build_prompt(env, index, skills)
            response = await self.client.propose(
                prompt,
                seed=self.seed + index,
            )
            canvas = await env.step(response.text)
            turns.append(DirectorTurn(index, prompt, response, canvas))
            if canvas.done and canvas.final_answer is not None:
                return OrchestrationResult(
                    final_answer=canvas.final_answer,
                    turns=tuple(turns),
                    final_graph=env.graph.to_dict(),
                    termination_reason="finish",
                    explicit_finish=True,
                )
        return OrchestrationResult(
            final_answer=None,
            turns=tuple(turns),
            final_graph=env.graph.to_dict(),
            termination_reason="max_rounds",
            explicit_finish=False,
        )


__all__ = [
    "AgentGraphOrchestrator",
    "DIRECTOR_SYSTEM_PROMPT",
    "DirectorClient",
    "DirectorError",
    "DirectorResponse",
    "DirectorTurn",
    "OpenAIDirectorClient",
    "OrchestrationResult",
]
