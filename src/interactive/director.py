"""Inference-time Qwen Flow-Director loop over the strict AgentGraph Canvas."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import socket
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
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

Use a model_id from the supplied catalog and describe each Agent's job in ordinary free text. Finish only after the Canvas accepts a complete graph."""


class DirectorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DirectorResponse:
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DirectorClient(Protocol):
    async def propose(self, prompt: str) -> DirectorResponse:
        ...


class OpenAIDirectorClient:
    """OpenAI-compatible chat client for the local Qwen3.5-9B endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8005/v1",
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

    async def propose(self, prompt: str) -> DirectorResponse:
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
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                value = await asyncio.to_thread(self._post, api_key, payload)
                return self._parse(value)
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
    final_answer: str
    turns: Tuple[DirectorTurn, ...]
    final_graph: Mapping[str, Any]


class AgentGraphOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        client: DirectorClient,
        *,
        max_rounds: int = 20,
        seed: int = 42,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.registry = registry
        self.client = client
        self.max_rounds = max_rounds
        self.seed = seed

    def build_prompt(
        self,
        env: AgentWorkflowEnv,
        turn_index: int,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        preferred = self.registry.select_weighted(
            seed=self.seed + turn_index,
            cheap_bias=1.0,
            fast_bias=1.0,
        )
        catalog = [
            {
                "model_id": model_id,
                "selection_weight": self.registry.require_model(model_id).selection_weight,
                "cheap_weight": self.registry.require_model(model_id).cheap_weight,
                "fast_weight": self.registry.require_model(model_id).fast_weight,
            }
            for model_id in self.registry.model_ids
        ]
        payload = {
            "task": env.problem,
            "turn": turn_index,
            "current_graph": env.graph.to_dict(),
            "canvas_feedback": env.snapshot().last_feedback,
            "model_catalog": catalog,
            "weighted_preferred_model": preferred.model_id,
        }
        if skills:
            payload["available_skills"] = list(skills)
        return (
            "Choose one next edit. The preferred model is only a cheap/fast suggestion.\n\n"
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
            response = await self.client.propose(prompt)
            canvas = await env.step(response.text)
            turns.append(DirectorTurn(index, prompt, response, canvas))
            if canvas.done and canvas.final_answer is not None:
                return OrchestrationResult(
                    final_answer=canvas.final_answer,
                    turns=tuple(turns),
                    final_graph=env.graph.to_dict(),
                )
        raise DirectorError(f"Director did not finish within {self.max_rounds} turns")


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
