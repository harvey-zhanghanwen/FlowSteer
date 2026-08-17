"""Inference-time Qwen Flow-Director loop over the strict AgentGraph Canvas."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import random
import socket
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStepResult
from .model_registry import ModelRegistry
from .scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
)


DIRECTOR_SYSTEM_PROMPT = """You are the Flow-Director. Build an executable AgentGraph for the task, one edit at a time. Follow the latest Canvas observation and return exactly one JSON object each turn.

Actions:
{"action":"add_agent","agent_id":"...","model_id":"...","contract":"..."}
{"action":"modify_agent","agent_id":"...","model_id":"...","contract":"..."}
{"action":"delete_agent","agent_id":"..."}
{"action":"set_relation","source_id":"...","target_id":"...","source_to_target":true,"target_to_source":false}
{"action":"set_output","agent_id":"..."}
{"action":"finish"}

Use a model_id from the supplied catalog. Before the first edit, inspect whether the task has distinct evidence dependencies and represent only dependencies that need separate artifacts. Describe each Agent's objective, inputs or dependencies, output artifact, and completion condition in concise ordinary text. A directed relation sends the source artifact to the target; the target contract should name the artifact it consumes. Only the Output Agent returns the final task answer, and its contract should request a concise answer span rather than JSON or explanation. Use execution evidence and Canvas issues to decide the next atomic edit or finish; structural or output-format validity alone does not establish task quality."""


DIRECTOR_TRANSCRIPT_SCHEMA = "flowsteer.director.transcript.v1"
DIRECTOR_TRANSCRIPT_HEADER = "Flow-Director chat transcript"


def encode_director_transcript(
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Serialize the exact multi-turn Director messages into a receipt string."""

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Director transcript has an unsupported role")
        if not isinstance(content, str) or not content:
            raise ValueError("Director transcript messages require non-empty content")
        normalized.append({"role": role, "content": content})
    if len(normalized) < 2 or normalized[0] != {
        "role": "system",
        "content": DIRECTOR_SYSTEM_PROMPT,
    }:
        raise ValueError("Director transcript must start with the fixed system prompt")
    if normalized[1]["role"] != "user":
        raise ValueError("Director transcript must start with a user task message")
    payload = {
        "schema_version": DIRECTOR_TRANSCRIPT_SCHEMA,
        "messages": normalized,
    }
    return DIRECTOR_TRANSCRIPT_HEADER + "\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_director_transcript(
    prompt: str,
) -> Optional[Tuple[Mapping[str, str], ...]]:
    """Decode a canonical transcript, or return ``None`` for a legacy prompt."""

    if not isinstance(prompt, str) or not prompt.startswith(
        DIRECTOR_TRANSCRIPT_HEADER + "\n\n"
    ):
        return None
    _, _, raw_payload = prompt.partition("\n\n")
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError) as exc:
        raise DirectorError("Director transcript is not valid JSON") from exc
    if not isinstance(payload, Mapping) or payload.get(
        "schema_version"
    ) != DIRECTOR_TRANSCRIPT_SCHEMA:
        raise DirectorError("Director transcript has an unsupported schema")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise DirectorError("Director transcript has no message list")
    try:
        canonical = encode_director_transcript(raw_messages)
    except (TypeError, ValueError) as exc:
        raise DirectorError("Director transcript violates its message contract") from exc
    if canonical != prompt:
        raise DirectorError("Director transcript is not canonical")
    return tuple(dict(message) for message in raw_messages)


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
        messages = decode_director_transcript(prompt)
        payload = {
            "model": self.model,
            "messages": (
                list(messages)
                if messages is not None
                else [
                    {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            ),
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
        catalog_order_seed: int | str | None = None,
        history_window: int = 4,
        sampling_base_seed: int | None = None,
        sampling_coordinate: ScientificSamplingCoordinate | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if isinstance(history_window, bool) or not isinstance(history_window, int) or history_window < 1:
            raise ValueError("history_window must be a positive integer")
        self.registry = registry
        self.client = client
        self.max_rounds = max_rounds
        self.seed = seed
        if (sampling_base_seed is None) != (sampling_coordinate is None):
            raise ValueError(
                "sampling_base_seed and sampling_coordinate must be supplied together"
            )
        if sampling_base_seed is not None and (
            type(sampling_base_seed) is not int
            or not 0 <= sampling_base_seed < 2**64
        ):
            raise ValueError("sampling_base_seed must be an unsigned 64-bit integer")
        self.sampling_base_seed = sampling_base_seed
        self.sampling_coordinate = sampling_coordinate
        # Sampling varies across rollouts, while a same-task/same-condition
        # group must see the same catalog presentation in its exact prompt.
        self.catalog_order_seed = seed if catalog_order_seed is None else catalog_order_seed
        self.history_window = history_window

    def generation_seed(self, round_index: int) -> int:
        """Return the exact Director action seed for one zero-based Canvas round."""

        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if self.sampling_coordinate is None:
            return self.seed + round_index
        assert self.sampling_base_seed is not None
        return derive_generation_seed(
            base_seed=self.sampling_base_seed,
            coordinate=self.sampling_coordinate,
            step_index=round_index + 1,
            phase=GenerationPhase.ACTION,
        )

    @property
    def sampling_receipt(self) -> Mapping[str, Any]:
        """Return the trajectory-level SkillFlow scientific sampling receipt."""

        if self.sampling_coordinate is None:
            return {}
        assert self.sampling_base_seed is not None
        return {
            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": self.sampling_base_seed,
            "coordinate": self.sampling_coordinate.to_value(),
            "phase": GenerationPhase.ACTION.value,
        }

    def _model_catalog(self) -> list[dict[str, Any]]:
        # Present the frozen set in a deterministic per-condition order.  The
        # previous sorted order made the alphabetically first family the de
        # facto default after the preferred-model hint was removed.  This does
        # not select a model; every action still names the Director's choice.
        catalog_model_ids = list(self.registry.model_ids)
        random.Random(self.catalog_order_seed).shuffle(catalog_model_ids)
        return [
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
            for model_id in catalog_model_ids
        ]

    def _canvas_observation(
        self,
        env: AgentWorkflowEnv,
        *,
        include_task_context: bool,
        skills: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        complete_validation = env.graph.validate(
            self.registry,
            require_complete=True,
        )
        snapshot = env.snapshot()
        directed_edges = [
            {"from": source_id, "to": target_id}
            for relation in env.graph.relations
            for source_id, target_id in relation.directed_edges()
        ]
        payload: dict[str, Any] = {
            "current_graph": env.graph.to_dict(),
            "canvas_feedback": snapshot.last_feedback,
        }
        if directed_edges:
            # The two-bit relation remains the canonical mutation receipt.  A
            # direct edge view avoids making the Director mentally invert a
            # relation after AgentGraph canonicalizes endpoint order.
            payload["directed_edges"] = directed_edges
        if complete_validation.issues:
            payload["structural_issues"] = [
                {
                    "code": issue.code,
                    "message": issue.message,
                }
                for issue in complete_validation.issues
            ]
        if include_task_context:
            payload.update(
                {
                    "task": env.problem,
                    "model_catalog": self._model_catalog(),
                }
            )
            if env.max_agents is not None:
                payload["max_agents"] = env.max_agents
        if skills:
            payload["available_skills"] = list(skills)
        return payload

    @staticmethod
    def _observation_message(payload: Mapping[str, Any]) -> str:
        return (
            "Canvas observation. Choose exactly one next action using only the "
            "task, messages, Canvas, catalog, and any supplied validated Skill facts.\n\n"
            + json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def build_prompt(
        self,
        env: AgentWorkflowEnv,
        turn_index: int,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        """Start one SkillFlow-style persistent Director conversation."""

        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        initial = self._canvas_observation(
            env,
            include_task_context=True,
            skills=skills,
        )
        return encode_director_transcript(
            (
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": self._observation_message(initial)},
            )
        )

    def continue_prompt(
        self,
        previous_prompt: str,
        assistant_content: str,
        env: AgentWorkflowEnv,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        """Append the real sampled action and current Canvas observation."""

        messages = decode_director_transcript(previous_prompt)
        if messages is None:
            raise DirectorError("cannot continue a legacy single-user Director prompt")
        if not isinstance(assistant_content, str) or not assistant_content:
            raise DirectorError("Director continuation requires sampled assistant content")
        observation = self._canvas_observation(
            env,
            include_task_context=False,
            skills=skills,
        )
        continuation = list(messages[2:])
        continuation.extend(
            (
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": self._observation_message(observation),
                },
            )
        )
        # Keep the immutable task/catalog context and a bounded real message
        # continuation.  Unlike the former reconstructed history JSON, these
        # are the exact assistant actions and Canvas observations seen by Qwen.
        continuation = continuation[-2 * self.history_window :]
        return encode_director_transcript(
            (messages[0], messages[1], *continuation)
        )

    @staticmethod
    def consumed_assistant_content(
        response: DirectorResponse,
        canvas: AgentWorkflowStepResult,
    ) -> str:
        action = canvas.action
        if action is None:
            return response.text
        return response.text[: action.consumed_end]

    async def run(
        self,
        env: AgentWorkflowEnv,
        problem: str,
        *,
        skills: Sequence[Mapping[str, Any]] = (),
    ) -> OrchestrationResult:
        env.reset(problem)
        turns: list[DirectorTurn] = []
        prompt = self.build_prompt(env, 0, skills)
        for index in range(self.max_rounds):
            response = await self.client.propose(
                prompt,
                seed=self.generation_seed(index),
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
            prompt = self.continue_prompt(
                prompt,
                self.consumed_assistant_content(response, canvas),
                env,
                skills,
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
    "DIRECTOR_TRANSCRIPT_SCHEMA",
    "DirectorClient",
    "DirectorError",
    "DirectorResponse",
    "DirectorTurn",
    "OpenAIDirectorClient",
    "OrchestrationResult",
    "decode_director_transcript",
    "encode_director_transcript",
]
