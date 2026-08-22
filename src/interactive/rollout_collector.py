"""Exact-receipt AgentGraph rollout collection for the smoke-training path.

This module is a narrow adapter over the existing AgentGraph implementation:

* :class:`SGLangReceiptDirectorClient` uses SGLang's native ``/generate``
  endpoint so the sampled token IDs and behavior log-probabilities are returned
  by the behavior server rather than reconstructed later;
* :func:`select_balanced_tasks` selects the first two aligned training records
  from each of the seven configured sources; and
* :class:`AgentGraphRolloutCollector` drives the existing
  :class:`~src.interactive.agent_workflow_env.AgentWorkflowEnv`, materializes
  the existing versioned record contracts, and optionally appends them to the
  existing :class:`~src.interactive.persistence.EvidenceStore`.

The collector deliberately does not implement an evaluator or a trainer.  It
only records natural-policy trajectories, including incomplete trajectories,
for those downstream boundaries.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import socket
import threading
import time
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_action_parser import AgentAction
from .agent_runtime import AgentCallRecord, AgentRuntimeResult
from .agent_workflow_env import AgentWorkflowEnv
from .director import (
    AgentGraphOrchestrator,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1,
    DIRECTOR_SYSTEM_PROMPT,
    DirectorError,
    DirectorResponse,
    decode_director_transcript,
    director_actions_from_admissible_schema_branch,
    director_model_admissible_sampling_json_schema_text,
    director_model_admissible_sampling_json_schema_text_v1,
    director_modify_agent_field_sampling_json_schema_text,
    director_modify_agent_field_selector_json_schema_text,
    director_state_conditioned_sampling_json_schema_text,
)
from .openai_gateway import build_agent_messages
from .persistence import EvidenceStore, GraphSnapshotEvent, stable_id
from .records import (
    EvaluationReceipt,
    ExecutionRecord,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
    VALID_SPLITS,
    canonical_active_skill_ids,
    canonical_invoked_skill_ids,
    ordered_skill_ids,
)
from .versioning import VersionBundle


AGENTGRAPH_SMOKE_SOURCES: Tuple[str, ...] = (
    "HotpotQA",
    "TriviaQA",
    "AIME 2026",
    "HealthBench Professional",
    "WebShop",
    "ALFWorld",
    "SWE-bench",
)


class ReceiptValidationError(DirectorError):
    """Raised when SGLang cannot prove an exact on-policy token receipt."""


class RolloutGate:
    """Thread-safe pause-and-drain boundary for SGLang weight synchronization.

    SkillFlow pauses Supervisor requests while replacing ``theta_live``.  This
    gate adds an explicit in-flight count so the synchronizer can first block
    new requests and then wait until every request using the old adapter has
    left the server boundary.
    """

    def __init__(self, *, poll_interval_seconds: float = 0.01) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._condition = threading.Condition()
        self._paused = False
        self._in_flight = 0

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def in_flight(self) -> int:
        with self._condition:
            return self._in_flight

    def pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def wait_for_drain(self, timeout_seconds: Optional[float] = None) -> bool:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        with self._condition:
            while self._in_flight:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def drain(self, timeout_seconds: Optional[float] = None) -> None:
        """Block until all admitted requests finish (PolicySyncGate API)."""

        if not self.wait_for_drain(timeout_seconds):
            raise TimeoutError("timed out waiting for in-flight rollout requests")

    def pause_and_drain(self, timeout_seconds: Optional[float] = None) -> bool:
        self.pause()
        return self.wait_for_drain(timeout_seconds)

    async def async_pause_and_drain(self, timeout_seconds: Optional[float] = None) -> bool:
        self.pause()
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        while self.in_flight:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self.poll_interval_seconds)
        return True

    async def acquire(self) -> None:
        # Do not block the event loop on a threading.Condition: the trainer may
        # keep the gate paused for the full adapter load operation.
        while True:
            with self._condition:
                if not self._paused:
                    self._in_flight += 1
                    return
            await asyncio.sleep(self.poll_interval_seconds)

    def release(self) -> None:
        with self._condition:
            if self._in_flight <= 0:
                raise RuntimeError("RolloutGate release without a matching acquire")
            self._in_flight -= 1
            if self._in_flight == 0:
                self._condition.notify_all()

    def require_paused_and_drained(self) -> None:
        with self._condition:
            if not self._paused or self._in_flight:
                raise RuntimeError(
                    "rollout policy may change only while the gate is paused and drained"
                )


def _token_ids(value: object, field_name: str) -> Tuple[int, ...]:
    """Normalize one unbatched token-ID sequence without importing torch."""

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    if not isinstance(value, (list, tuple)):
        raise ReceiptValidationError(f"{field_name} must be a token-ID sequence")
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ReceiptValidationError(
                f"{field_name}[{index}] must be a non-negative integer"
            )
        result.append(item)
    if not result:
        raise ReceiptValidationError(f"{field_name} must not be empty")
    return tuple(result)


def _behavior_receipt(value: object) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """Parse SGLang ``output_token_logprobs`` tuples strictly."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ReceiptValidationError(
            "SGLang response has no output_token_logprobs receipt"
        )
    token_ids: list[int] = []
    log_probs: list[float] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ReceiptValidationError(
                f"output_token_logprobs[{index}] is not an SGLang token tuple"
            )
        raw_log_prob, raw_token_id = item[0], item[1]
        if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
            raise ReceiptValidationError(
                f"output_token_logprobs[{index}] has an invalid token ID"
            )
        if raw_token_id < 0 or isinstance(raw_log_prob, bool) or not isinstance(
            raw_log_prob, (int, float)
        ):
            raise ReceiptValidationError(
                f"output_token_logprobs[{index}] has an invalid log-prob receipt"
            )
        log_prob = float(raw_log_prob)
        if not math.isfinite(log_prob):
            raise ReceiptValidationError(
                f"output_token_logprobs[{index}] has a non-finite log-prob"
            )
        token_ids.append(raw_token_id)
        log_probs.append(log_prob)
    return tuple(token_ids), tuple(log_probs)


def _exact_count(value: object, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptValidationError(f"{field_name} must be a non-negative integer")
    return value


class SGLangReceiptDirectorClient:
    """Qwen3.5 Director client using SGLang's exact native token receipt.

    ``tokenizer`` must be loaded from the same Qwen3.5 checkpoint as the SGLang
    behavior server.  The client intentionally requires
    ``apply_chat_template(..., enable_thinking=False)`` and never falls back to
    an approximately reconstructed prompt.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        base_url: str = "http://127.0.0.1:8015",
        api_key: str = "EMPTY",
        policy_version: str,
        adapter_name: Optional[str] = None,
        expected_server_weight_version: Optional[str] = None,
        rollout_gate: Optional[RolloutGate] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 768,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
    ) -> None:
        if not hasattr(tokenizer, "apply_chat_template") or not hasattr(tokenizer, "decode"):
            raise ValueError("tokenizer must expose apply_chat_template() and decode()")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be absolute HTTP(S)")
        normalized_base = base_url.rstrip("/")
        # The OpenAI-compatible endpoint lives under /v1, while SGLang's native
        # exact-receipt endpoint is rooted at /generate.
        if normalized_base.endswith("/v1"):
            normalized_base = normalized_base[:-3]
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Director temperature/top_p are invalid")
        if top_k == 0 or top_k < -1:
            raise ValueError("top_k must be -1 or a positive integer")
        if max_tokens <= 0 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Director token, timeout, and retry limits are invalid")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version must be non-empty")
        if action_json_schema is not None and (
            not isinstance(action_json_schema, str)
            or not action_json_schema.strip()
        ):
            raise ValueError("action_json_schema must be non-empty text or None")
        if (action_json_schema is None) != (action_json_schema_version is None):
            raise ValueError(
                "action_json_schema and action_json_schema_version must be "
                "supplied together"
            )
        if action_json_schema_version is not None and (
            not isinstance(action_json_schema_version, str)
            or not action_json_schema_version.strip()
        ):
            raise ValueError(
                "action_json_schema_version must be non-empty text or None"
            )
        if (
            expected_server_weight_version is not None
            and not expected_server_weight_version.strip()
        ):
            raise ValueError(
                "expected_server_weight_version must be non-empty when supplied"
            )
        if adapter_name is not None and not adapter_name.strip():
            raise ValueError("adapter_name must be non-empty when supplied")

        self.tokenizer = tokenizer
        self.base_url = normalized_base
        self.api_key = api_key
        self.rollout_gate = rollout_gate or RolloutGate()
        self._route_lock = threading.Lock()
        self._policy_version = policy_version.strip()
        self._adapter_name = adapter_name.strip() if adapter_name is not None else None
        self._expected_server_weight_version = (
            expected_server_weight_version.strip()
            if expected_server_weight_version is not None
            else None
        )
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.action_json_schema = action_json_schema
        self.action_json_schema_version = action_json_schema_version

    @property
    def generate_url(self) -> str:
        return self.base_url + "/generate"

    @property
    def adapter_name(self) -> Optional[str]:
        with self._route_lock:
            return self._adapter_name

    @property
    def policy_version(self) -> str:
        with self._route_lock:
            return self._policy_version

    @property
    def expected_server_weight_version(self) -> Optional[str]:
        with self._route_lock:
            return self._expected_server_weight_version

    def update_policy_route(
        self,
        *,
        policy_version: str,
        adapter_name: Optional[str],
        expected_server_weight_version: Optional[str],
    ) -> None:
        """Atomically switch the rollout route after a paused, drained sync."""

        if adapter_name is not None and not adapter_name.strip():
            raise ValueError("adapter_name must be non-empty when supplied")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version must be non-empty")
        if (
            expected_server_weight_version is not None
            and not expected_server_weight_version.strip()
        ):
            raise ValueError(
                "expected_server_weight_version must be non-empty when supplied"
            )
        self.rollout_gate.require_paused_and_drained()
        with self._route_lock:
            self._policy_version = policy_version.strip()
            self._adapter_name = adapter_name.strip() if adapter_name is not None else None
            self._expected_server_weight_version = (
                expected_server_weight_version.strip()
                if expected_server_weight_version is not None
                else None
            )

    def _policy_route(self) -> Tuple[str, Optional[str], Optional[str]]:
        with self._route_lock:
            return (
                self._policy_version,
                self._adapter_name,
                self._expected_server_weight_version,
            )

    def prompt_token_ids(self, prompt: str) -> Tuple[int, ...]:
        if not isinstance(prompt, str) or not prompt:
            raise ReceiptValidationError("Director prompt must be non-empty")
        transcript = decode_director_transcript(prompt)
        messages = (
            list(transcript)
            if transcript is not None
            else [
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            raise ReceiptValidationError(
                "Qwen3.5 tokenizer must support enable_thinking=False"
            ) from exc
        return _token_ids(encoded, "prompt_token_ids")

    def _request_payload(
        self,
        prompt: str,
        adapter_name: Optional[str],
        seed: Optional[int] = None,
        *,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("Director seed must be a non-negative integer or None")
        prompt_ids = self.prompt_token_ids(prompt)
        payload: dict[str, Any] = {
            "input_ids": list(prompt_ids),
            "sampling_params": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_new_tokens": self.max_tokens,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "no_stop_trim": True,
            },
            "return_logprob": True,
            "logprob_start_len": len(prompt_ids),
            "top_logprobs_num": 0,
            "return_text_in_logprobs": True,
            "stream": False,
        }
        if seed is not None:
            # SkillFlow's OpenAI boundary calls this field ``seed``.  The
            # deployed SGLang 0.5.15 native /generate SamplingParams exposes
            # the equivalent field as ``sampling_seed``.
            payload["sampling_params"]["sampling_seed"] = seed
        (
            resolved_action_schema,
            _,
            _,
        ) = self._resolve_action_schema(
            action_json_schema=action_json_schema,
            action_json_schema_version=action_json_schema_version,
            action_schema_branch=action_schema_branch,
        )
        if resolved_action_schema is not None:
            # NECESSARY_ADAPTATION: deployed SGLang 0.5.15 exposes
            # SamplingParams.json_schema.  Evaluation may use the schema that
            # mirrors the strict AgentActionParser.  Training keeps this off
            # until its HF loss path applies the identical grammar mask and
            # constrained-policy normalization.
            payload["sampling_params"]["json_schema"] = resolved_action_schema
        if adapter_name is not None:
            payload["lora_path"] = adapter_name
        return payload

    def request_payload(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
    ) -> Mapping[str, Any]:
        _, adapter_name, _ = self._policy_route()
        return self._request_payload(
            prompt,
            adapter_name,
            seed,
            action_json_schema=action_json_schema,
            action_json_schema_version=action_json_schema_version,
            action_schema_branch=action_schema_branch,
        )

    def _resolve_action_schema(
        self,
        *,
        action_json_schema: Optional[str],
        action_json_schema_version: Optional[str],
        action_schema_branch: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        override_requested = any(
            value is not None
            for value in (
                action_json_schema,
                action_json_schema_version,
                action_schema_branch,
            )
        )
        if not override_requested:
            return (
                self.action_json_schema,
                self.action_json_schema_version,
                None,
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                action_json_schema,
                action_json_schema_version,
                action_schema_branch,
            )
        ):
            raise ValueError(
                "per-request action schema, version, and branch must be "
                "supplied together as non-empty text"
            )
        assert action_json_schema is not None
        assert action_json_schema_version is not None
        assert action_schema_branch is not None
        try:
            supplied_schema = json.loads(action_json_schema)
            normalized_branch = action_schema_branch.strip()
            if normalized_branch.startswith("admissible-v2:"):
                if (
                    action_json_schema_version.strip()
                    != DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "v2 admissible branch requires its exact schema version"
                    )
                expected_schema_text = (
                    director_model_admissible_sampling_json_schema_text(
                        director_actions_from_admissible_schema_branch(
                            normalized_branch
                        )
                    )
                )
            elif normalized_branch.startswith("admissible:"):
                if (
                    action_json_schema_version.strip()
                    != DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1
                ):
                    raise ValueError(
                        "v1 admissible branch requires its exact schema version"
                    )
                expected_schema_text = (
                    director_model_admissible_sampling_json_schema_text_v1(
                        director_actions_from_admissible_schema_branch(
                            normalized_branch
                        )
                    )
                )
            else:
                expected_schema_text = (
                    director_state_conditioned_sampling_json_schema_text(
                        normalized_branch
                    )
                )
            expected_schema = json.loads(expected_schema_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "per-request action schema must be the strict schema for its branch"
            ) from exc
        if supplied_schema != expected_schema:
            raise ValueError(
                "per-request action schema does not match its declared branch"
            )
        return (
            action_json_schema.strip(),
            action_json_schema_version.strip(),
            action_schema_branch.strip(),
        )

    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
    ) -> DirectorResponse:
        await self.rollout_gate.acquire()
        try:
            policy_version, adapter_name, expected_server_weight_version = (
                self._policy_route()
            )
            (
                resolved_action_schema,
                resolved_action_schema_version,
                resolved_action_schema_branch,
            ) = self._resolve_action_schema(
                action_json_schema=action_json_schema,
                action_json_schema_version=action_json_schema_version,
                action_schema_branch=action_schema_branch,
            )
            payload = self._request_payload(
                prompt,
                adapter_name,
                seed,
                action_json_schema=action_json_schema,
                action_json_schema_version=action_json_schema_version,
                action_schema_branch=action_schema_branch,
            )
            if (
                resolved_action_schema_version
                == DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION
            ):
                assert resolved_action_schema_branch is not None
                actions = director_actions_from_admissible_schema_branch(
                    resolved_action_schema_branch
                )
                return await self._propose_hierarchical_action(
                    prompt=prompt,
                    seed=seed,
                    actions=actions,
                    selector_payload=payload,
                    action_schema_version=resolved_action_schema_version,
                    action_schema_branch=resolved_action_schema_branch,
                    policy_version=policy_version,
                    adapter_name=adapter_name,
                    expected_server_weight_version=expected_server_weight_version,
                )

            value, latency_ms, attempt_count = await self._post_with_retries(payload)
            return self._parse_response(
                prompt,
                payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=(
                    resolved_action_schema_version
                    if resolved_action_schema is not None
                    else None
                ),
                action_schema_branch=resolved_action_schema_branch,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
            )
        finally:
            self.rollout_gate.release()

    async def _post_with_retries(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], float, int]:
        """Submit one exact SGLang generation phase with transport retries."""

        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                value = await asyncio.to_thread(self._post_json, payload)
                return (
                    value,
                    max((time.monotonic() - started_at) * 1000.0, 0.0),
                    attempt + 1,
                )
            except HTTPError as exc:
                last_error = exc
                if not (exc.code in {408, 409, 425, 429} or exc.code >= 500):
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
            if attempt < self.max_retries:
                await asyncio.sleep(min(2.0**attempt, 4.0))
        detail = (
            f"HTTP {last_error.code}"
            if isinstance(last_error, HTTPError)
            else type(last_error).__name__
        )
        raise DirectorError(f"SGLang Director request failed: {detail}") from last_error

    @staticmethod
    def _hierarchical_choice(
        text: str,
        *,
        field_name: str,
        admitted: Sequence[str],
        required_action: str | None = None,
    ) -> str:
        """Parse one constrained discriminator without repairing sampled text."""

        try:
            value, _ = json.JSONDecoder().raw_decode(text.lstrip())
        except (TypeError, ValueError) as exc:
            raise ReceiptValidationError(
                "hierarchical Director discriminator is not JSON"
            ) from exc
        expected_fields = {field_name}
        if required_action is not None:
            expected_fields.add("action")
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ReceiptValidationError(
                "hierarchical Director discriminator has incompatible fields"
            )
        if required_action is not None and value.get("action") != required_action:
            raise ReceiptValidationError(
                "hierarchical Director discriminator changed its action"
            )
        selected = value.get(field_name)
        if not isinstance(selected, str) or selected not in admitted:
            raise ReceiptValidationError(
                "hierarchical Director discriminator selected an inadmissible value"
            )
        return selected

    @staticmethod
    def _hierarchical_phase_receipt(response: DirectorResponse) -> Mapping[str, Any]:
        metadata = response.metadata
        return {
            "text": response.text,
            "prompt_token_ids": metadata.get("prompt_token_ids"),
            "output_token_ids": metadata.get("output_token_ids"),
            "behavior_log_probs": metadata.get("behavior_log_probs"),
            "request_id": metadata.get("request_id"),
            "finish_reason": metadata.get("finish_reason"),
            "prompt_tokens": metadata.get("prompt_tokens"),
            "completion_tokens": metadata.get("completion_tokens"),
            "latency_ms": metadata.get("latency_ms"),
            "attempt_count": metadata.get("attempt_count"),
            "generation_seed": metadata.get("generation_seed"),
            "server_weight_version": metadata.get("server_weight_version"),
            "receipt_verified": metadata.get("receipt_verified"),
        }

    async def _propose_hierarchical_action(
        self,
        *,
        prompt: str,
        seed: Optional[int],
        actions: Sequence[str],
        selector_payload: Mapping[str, Any],
        action_schema_version: str,
        action_schema_branch: str,
        policy_version: str,
        adapter_name: Optional[str],
        expected_server_weight_version: Optional[str],
    ) -> DirectorResponse:
        """Sample action type, optional MODIFY field, then exact parameters."""

        total_latency_ms = 0.0
        total_attempt_count = 0
        phase_receipts: dict[str, Mapping[str, Any]] = {}

        if len(actions) == 1:
            selected_action = actions[0]
        else:
            value, latency_ms, attempt_count = await self._post_with_retries(
                selector_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            selector_response = self._parse_response(
                prompt,
                selector_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
            )
            selected_action = self._hierarchical_choice(
                selector_response.text,
                field_name="action",
                admitted=actions,
            )
            phase_receipts["action_selection"] = self._hierarchical_phase_receipt(
                selector_response
            )

        selected_modify_field: str | None = None
        if selected_action == "modify_agent":
            field_schema = director_modify_agent_field_selector_json_schema_text()
            field_payload = dict(self._request_payload(prompt, adapter_name, seed))
            field_sampling = dict(field_payload["sampling_params"])
            field_sampling["json_schema"] = field_schema
            field_payload["sampling_params"] = field_sampling
            value, latency_ms, attempt_count = await self._post_with_retries(
                field_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            field_response = self._parse_response(
                prompt,
                field_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
            )
            field_selector = json.loads(field_schema)
            admitted_fields = field_selector["properties"]["field"]["enum"]
            selected_modify_field = self._hierarchical_choice(
                field_response.text,
                field_name="field",
                admitted=admitted_fields,
                required_action="modify_agent",
            )
            phase_receipts["modify_field_selection"] = (
                self._hierarchical_phase_receipt(field_response)
            )
            parameter_schema = director_modify_agent_field_sampling_json_schema_text(
                selected_modify_field
            )
        else:
            parameter_schema = director_state_conditioned_sampling_json_schema_text(
                selected_action
            )

        parameter_payload = dict(self._request_payload(prompt, adapter_name, seed))
        parameter_sampling = dict(parameter_payload["sampling_params"])
        parameter_sampling["json_schema"] = parameter_schema
        parameter_payload["sampling_params"] = parameter_sampling
        value, latency_ms, attempt_count = await self._post_with_retries(
            parameter_payload
        )
        total_latency_ms += latency_ms
        total_attempt_count += attempt_count
        response = self._parse_response(
            prompt,
            parameter_payload,
            value,
            policy_version=policy_version,
            adapter_name=adapter_name,
            expected_server_weight_version=expected_server_weight_version,
            action_json_schema_version=action_schema_version,
            action_schema_branch=action_schema_branch,
            latency_ms=latency_ms,
            attempt_count=attempt_count,
        )
        metadata = dict(response.metadata)
        metadata.update(
            {
                "action_decoding_strategy": "hierarchical_json_schema",
                "selected_action": selected_action,
                "selected_modify_field": selected_modify_field,
                "parameter_schema_branch": (
                    selected_action
                    if selected_modify_field is None
                    else f"modify_agent:{selected_modify_field}"
                ),
                "hierarchical_phase_receipts": phase_receipts,
                "request_count": len(phase_receipts) + 1,
                "latency_ms": total_latency_ms,
                "attempt_count": total_attempt_count,
            }
        )
        return DirectorResponse(text=response.text, metadata=metadata)

    def _post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            self.generate_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-SGLang-Receipt/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ReceiptValidationError("SGLang returned a non-object response")
        return value

    def _decode(self, token_ids: Sequence[int]) -> str:
        try:
            text = self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError as exc:
            raise ReceiptValidationError(
                "tokenizer.decode must support exact, non-cleaning decoding"
            ) from exc
        if not isinstance(text, str):
            raise ReceiptValidationError("tokenizer.decode returned non-text output")
        return text

    def _parse_response(
        self,
        prompt: str,
        payload: Mapping[str, Any],
        value: Mapping[str, Any],
        *,
        policy_version: str,
        adapter_name: Optional[str],
        expected_server_weight_version: Optional[str],
        action_json_schema_version: Optional[str],
        action_schema_branch: Optional[str],
        latency_ms: float,
        attempt_count: int,
    ) -> DirectorResponse:
        text = value.get("text")
        meta_info = value.get("meta_info")
        if not isinstance(text, str) or not isinstance(meta_info, Mapping):
            raise ReceiptValidationError("SGLang response is missing text or meta_info")

        output_ids, behavior_log_probs = _behavior_receipt(
            meta_info.get("output_token_logprobs")
        )
        if self._decode(output_ids) != text:
            raise ReceiptValidationError(
                "SGLang output text does not exactly match its sampled token IDs"
            )
        if "output_ids" in value:
            direct_output_ids = _token_ids(value["output_ids"], "output_ids")
            if direct_output_ids != output_ids:
                raise ReceiptValidationError(
                    "SGLang output_ids disagree with output_token_logprobs token IDs"
                )

        prompt_ids = _token_ids(payload.get("input_ids"), "prompt_token_ids")
        prompt_count = _exact_count(meta_info.get("prompt_tokens"), "prompt_tokens")
        completion_count = _exact_count(
            meta_info.get("completion_tokens"), "completion_tokens"
        )
        if prompt_count is not None and prompt_count != len(prompt_ids):
            raise ReceiptValidationError(
                "SGLang prompt token count disagrees with the submitted input_ids"
            )
        if completion_count is not None and completion_count != len(output_ids):
            raise ReceiptValidationError(
                "SGLang completion token count disagrees with its behavior receipt"
            )

        raw_server_weight_version = meta_info.get("weight_version")
        if isinstance(raw_server_weight_version, bool) or raw_server_weight_version is None:
            raise ReceiptValidationError("SGLang response has no weight_version")
        server_weight_version = str(raw_server_weight_version).strip()
        if not server_weight_version:
            raise ReceiptValidationError("SGLang response has an empty weight_version")
        if (
            expected_server_weight_version is not None
            and server_weight_version != expected_server_weight_version
        ):
            raise ReceiptValidationError(
                "SGLang server weight_version does not match the expected server receipt"
            )

        return DirectorResponse(
            text=text,
            metadata={
                "prompt_text": prompt,
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "behavior_log_probs": behavior_log_probs,
                "policy_version": policy_version,
                "server_weight_version": server_weight_version,
                "adapter_name": adapter_name,
                "requested_lora_path": payload.get("lora_path"),
                "request_id": meta_info.get("id"),
                "finish_reason": meta_info.get("finish_reason"),
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(output_ids),
                "latency_ms": latency_ms,
                "attempt_count": attempt_count,
                "generation_seed": payload.get("sampling_params", {}).get(
                    "sampling_seed"
                ),
                "action_json_schema_version": action_json_schema_version,
                "action_schema_branch": action_schema_branch,
                "receipt_verified": True,
            },
        )

    def executed_prefix_tokens(
        self,
        response: DirectorResponse,
        action: AgentAction,
    ) -> int:
        """Return the shortest sampled token prefix covering the consumed action."""

        metadata = response.metadata
        output_ids = _token_ids(metadata.get("output_token_ids"), "output_token_ids")
        if metadata.get("receipt_verified") is not True:
            raise ReceiptValidationError("Director response is not an exact receipt")
        if metadata.get("prompt_text") is None:
            raise ReceiptValidationError("Director receipt has no prompt binding")
        if not (0 <= action.consumed_start < action.consumed_end <= len(response.text)):
            raise ReceiptValidationError("parsed action has an invalid consumed character span")
        if response.text[action.consumed_start : action.consumed_end] != action.raw_json:
            raise ReceiptValidationError("parsed action span disagrees with the sampled text")
        if self._decode(output_ids) != response.text:
            raise ReceiptValidationError("sampled text/token IDs changed after receipt creation")

        consumed_text = response.text[: action.consumed_end]
        for count in range(1, len(output_ids) + 1):
            decoded_prefix = self._decode(output_ids[:count])
            if decoded_prefix.startswith(consumed_text):
                return count
        raise ReceiptValidationError(
            "the Canvas-consumed character prefix is not a sampled token prefix"
        )


def select_balanced_tasks(
    tasks: Iterable[TaskRecord],
    *,
    per_source: int = 2,
    sources: Sequence[str] = AGENTGRAPH_SMOKE_SOURCES,
) -> Tuple[TaskRecord, ...]:
    """Select the first ``per_source`` records from every requested source."""

    if isinstance(per_source, bool) or not isinstance(per_source, int) or per_source <= 0:
        raise ValueError("per_source must be a positive integer")
    normalized_sources = tuple(str(source).strip() for source in sources)
    if not normalized_sources or any(not source for source in normalized_sources):
        raise ValueError("sources must contain non-empty names")
    if len(normalized_sources) != len(set(normalized_sources)):
        raise ValueError("sources must be unique")

    selected = {source: [] for source in normalized_sources}
    for task in tasks:
        raw_source = task.metadata.get("source")
        if not isinstance(raw_source, str) or not raw_source.strip():
            raise ValueError(f"task {task.task_id!r} has no metadata.source")
        source = raw_source.strip()
        if source in selected and len(selected[source]) < per_source:
            selected[source].append(task)
        if all(len(items) == per_source for items in selected.values()):
            break

    missing = {
        source: per_source - len(items)
        for source, items in selected.items()
        if len(items) != per_source
    }
    if missing:
        detail = ", ".join(f"{source}: {count}" for source, count in missing.items())
        raise ValueError(f"insufficient smoke-training tasks by source ({detail})")
    return tuple(task for source in normalized_sources for task in selected[source])


EvaluationValue = Union[EvaluationReceipt, Mapping[str, Any], object]
EvaluatorCallback = Callable[
    [TaskRecord, Optional[str], Mapping[str, Any], Optional[AgentRuntimeResult]],
    Union[EvaluationValue, Awaitable[EvaluationValue]],
]
SkillPromptProvider = Callable[
    [TaskRecord, AgentWorkflowEnv, VersionBundle],
    Sequence[Mapping[str, Any]],
]
ActiveSkillProvider = Callable[
    [TaskRecord, AgentWorkflowEnv, VersionBundle],
    Sequence[str],
]


def _retrieved_skill_ids(
    prompt_context: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    """Project retrieved prompt priors into canonical Skill IDs.

    Forced paired-intervention conditions share the Director prompt boundary
    but are not ACTIVE Skills and therefore cannot appear in a Skill receipt.
    Every ordinary Skill prior must carry the stable identity emitted by
    ``PromptSkillPrior.to_dict``.
    """

    skill_ids: list[str] = []
    for item in prompt_context:
        if item.get("application_mode") == "forced_probe_condition":
            continue
        skill_id = item.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ReceiptValidationError(
                "Director-visible Skill prior has no stable skill_id"
            )
        skill_ids.append(skill_id)
    return ordered_skill_ids(skill_ids, field_name="retrieved_skill_ids")


def _evaluation_receipt(value: EvaluationValue) -> EvaluationReceipt:
    if isinstance(value, EvaluationReceipt):
        return value
    if not isinstance(value, Mapping):
        if all(
            hasattr(value, field_name)
            for field_name in (
                "evaluator_version",
                "valid",
                "reward",
                "metrics",
                "reason",
            )
        ):
            value = {
                "evaluator_version": getattr(value, "evaluator_version"),
                "valid": getattr(value, "valid"),
                "reward": getattr(value, "reward"),
                "metrics": getattr(value, "metrics"),
                "reason": getattr(value, "reason"),
                "details": getattr(value, "details", {}),
            }
        else:
            raise TypeError(
                "evaluator callback must return EvaluationReceipt or an outcome mapping"
            )
    evaluator_version = value.get("evaluator_version")
    valid = value.get("valid")
    reward = value.get("reward")
    if not isinstance(evaluator_version, str) or not evaluator_version.strip():
        raise ValueError("evaluator outcome requires a non-empty evaluator_version")
    if type(valid) is not bool:
        raise ValueError("evaluator outcome valid must be a JSON boolean")
    if reward is not None and (
        isinstance(reward, bool) or not isinstance(reward, (int, float))
    ):
        raise ValueError("evaluator outcome reward must be numeric or null")
    raw_metrics = value.get("metrics", {})
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("evaluator outcome metrics must be a mapping")
    metrics: dict[str, float] = {}
    for key, metric in raw_metrics.items():
        if not isinstance(key, str) or isinstance(metric, bool) or not isinstance(
            metric, (int, float)
        ):
            raise ValueError("evaluator metrics must have string keys and numeric values")
        numeric = float(metric)
        if not math.isfinite(numeric):
            raise ValueError("evaluator metrics must be finite")
        metrics[key] = numeric
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("evaluator outcome reason must be text")
    details = value.get("details", {})
    if not isinstance(details, Mapping):
        raise ValueError("evaluator outcome details must be a mapping")
    return EvaluationReceipt(
        evaluator_version=evaluator_version.strip(),
        valid=valid,
        reward=None if reward is None else float(reward),
        metrics=metrics,
        reason=reason,
        details=dict(details),
    )


def _optional_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _sampling_value(
    response_metadata: Mapping[str, object],
    model_metadata: Mapping[str, str],
    key: str,
    default: Union[int, float],
) -> Union[int, float]:
    value: object = response_metadata.get(key, model_metadata.get(key, default))
    try:
        return int(value) if isinstance(default, int) else float(value)
    except (TypeError, ValueError):
        return default


_JSON_UNSAFE = object()

_PROVIDER_RESPONSE_METADATA_FIELDS: Tuple[str, ...] = (
    "provider_id",
    "model_id",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "temperature",
    "top_p",
    "max_tokens",
)

_UNIFIED_EXECUTION_METADATA_FIELDS: Tuple[str, ...] = (
    "execution_mode",
    "react_turns_used",
    "tool_calls",
    "tool_receipts",
    "react_trace",
    "model_calls",
    "environment_id",
    "task_family",
    "environment_revision",
    "environment_reset_receipt",
    "environment_receipts",
    "environment_terminal",
    "environment_turns_used",
    "environment_steps",
    "evaluator_environment_trace",
)


def _json_safe_value(value: object) -> object:
    """Convert receipt values to JSON types and reject opaque runtime objects."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _JSON_UNSAFE
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            safe_item = _json_safe_value(item)
            if safe_item is not _JSON_UNSAFE:
                converted[key] = safe_item
        return converted
    if isinstance(value, (list, tuple)):
        converted_items: list[object] = []
        for item in value:
            safe_item = _json_safe_value(item)
            if safe_item is not _JSON_UNSAFE:
                converted_items.append(safe_item)
        return converted_items
    return _JSON_UNSAFE


def _copy_json_safe_fields(
    source: Mapping[str, object],
    target: dict[str, object],
    fields: Sequence[str],
) -> None:
    for field in fields:
        if field not in source:
            continue
        safe_value = _json_safe_value(source[field])
        if safe_value is not _JSON_UNSAFE:
            target[field] = safe_value


def _json_safe_or_default(value: object, default: object = None) -> object:
    safe_value = _json_safe_value(value)
    return default if safe_value is _JSON_UNSAFE else safe_value


def _model_call_metadata(
    response_metadata: Mapping[str, object],
) -> Iterable[Mapping[str, object]]:
    raw_calls = response_metadata.get("model_calls", ())
    if not isinstance(raw_calls, (list, tuple)):
        return ()
    return tuple(
        call_metadata
        for call in raw_calls
        if isinstance(call, Mapping)
        and isinstance((call_metadata := call.get("metadata")), Mapping)
    )


def _aggregated_int_metadata(
    response_metadata: Mapping[str, object],
    field: str,
) -> Optional[int]:
    direct = _optional_int(response_metadata.get(field))
    if direct is not None:
        return direct
    values = [
        value
        for metadata in _model_call_metadata(response_metadata)
        if (value := _optional_int(metadata.get(field))) is not None
    ]
    return sum(values) if values else None


def _aggregated_float_metadata(
    response_metadata: Mapping[str, object],
    field: str,
) -> Optional[float]:
    direct = _optional_float(response_metadata.get(field))
    if direct is not None:
        return direct
    values = [
        value
        for metadata in _model_call_metadata(response_metadata)
        if (value := _optional_float(metadata.get(field))) is not None
    ]
    return sum(values) if values else None


def _request_record(call: AgentCallRecord) -> Mapping[str, Any]:
    request = call.request
    return {
        "request_id": request.request_id,
        "run_id": request.run_id,
        "graph_revision": request.graph_revision,
        "problem": request.problem,
        "agent": request.agent.to_dict(),
        "model": request.model.to_dict(),
        "provider_id": request.provider.provider_id,
        "phase": request.phase.value,
        "is_output_agent": request.is_output_agent,
        "execution_role": "format" if request.is_format_agent else "worker",
        "is_format_agent": request.is_format_agent,
        "is_format_predecessor": request.is_format_predecessor,
        "semantic_protocol": request.semantic_protocol,
        "communication_condition": request.communication_condition.value,
        "upstream": [item.to_dict() for item in request.upstream],
        "own_draft": request.own_draft,
        "peer_draft": (
            None
            if request.peer_draft is None
            else request.peer_draft.to_dict()
        ),
        "rendered_messages": build_agent_messages(request),
    }


def _execution_record(call: AgentCallRecord) -> ExecutionRecord:
    request = call.request
    response = call.response
    metadata = dict(response.metadata)
    request_record = _request_record(call)
    model_fingerprint = stable_id(
        "model",
        {
            "model": request.model.to_dict(),
            "provider": request.provider.to_dict(),
        },
    )
    request_hash = stable_id("request", request_record)
    execution_id = stable_id(
        "execution",
        {
            "request": request_record,
            "provider_request_id": metadata.get("provider_request_id"),
            "output": response.text,
        },
    )
    temperature = float(
        _sampling_value(metadata, request.model.metadata, "temperature", 0.0)
    )
    top_p = float(_sampling_value(metadata, request.model.metadata, "top_p", 1.0))
    max_tokens = int(
        _sampling_value(metadata, request.model.metadata, "max_tokens", 4096)
    )
    if temperature < 0:
        temperature = 0.0
    if not 0 < top_p <= 1:
        top_p = 1.0
    if max_tokens <= 0:
        max_tokens = 4096

    input_tokens = _aggregated_int_metadata(metadata, "prompt_tokens")
    output_tokens = _aggregated_int_metadata(metadata, "completion_tokens")
    total_tokens = _aggregated_int_metadata(metadata, "total_tokens")
    latency_ms = _aggregated_float_metadata(metadata, "latency_ms")
    attempt_count = _aggregated_int_metadata(metadata, "attempt_count")
    provider_model = metadata.get("provider_model", request.model.model_name)
    response_receipt: dict[str, object] = {
        "provider_request_id": _json_safe_or_default(
            metadata.get("provider_request_id")
        ),
        "provider_model": _json_safe_or_default(
            provider_model,
            request.model.model_name,
        ),
        "finish_reason": _json_safe_or_default(metadata.get("finish_reason")),
        "attempt_count": attempt_count,
        "generation_seed": _optional_int(metadata.get("generation_seed")),
    }
    _copy_json_safe_fields(
        metadata,
        response_receipt,
        _PROVIDER_RESPONSE_METADATA_FIELDS,
    )
    _copy_json_safe_fields(
        metadata,
        response_receipt,
        _UNIFIED_EXECUTION_METADATA_FIELDS,
    )
    for field, value in (
        ("prompt_tokens", input_tokens),
        ("completion_tokens", output_tokens),
        ("total_tokens", total_tokens),
        ("latency_ms", latency_ms),
    ):
        if value is not None:
            response_receipt[field] = value
    return ExecutionRecord(
        execution_id=execution_id,
        experiment_id=request.run_id,
        graph_revision=request.graph_revision,
        agent_id=request.agent.id,
        model_id=request.model.model_id,
        model_fingerprint=model_fingerprint,
        provider=request.provider.provider_id,
        request_hash=request_hash,
        output=response.text,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        metadata={"request": request_record, "response": response_receipt},
    )


def execution_record_from_call(call: AgentCallRecord) -> ExecutionRecord:
    """Expose the collector's canonical Agent call receipt to eval drivers."""

    return _execution_record(call)


def _runtime_summary(runtime: Optional[AgentRuntimeResult]) -> Mapping[str, Any]:
    """Persist the existing runtime result without duplicating call receipts."""

    if runtime is None:
        return {}
    output_metadata: dict[str, object] = {}
    for agent_id, metadata in runtime.output_metadata.items():
        if not isinstance(agent_id, str) or not isinstance(metadata, Mapping):
            continue
        receipt: dict[str, object] = {}
        for field in ("provider_request_id", "provider_model", "finish_reason"):
            if field in metadata:
                safe_value = _json_safe_value(metadata[field])
                if safe_value is not _JSON_UNSAFE:
                    receipt[field] = safe_value
        attempt_count = _aggregated_int_metadata(metadata, "attempt_count")
        generation_seed = _optional_int(metadata.get("generation_seed"))
        if attempt_count is not None:
            receipt["attempt_count"] = attempt_count
        if generation_seed is not None:
            receipt["generation_seed"] = generation_seed
        _copy_json_safe_fields(
            metadata,
            receipt,
            _PROVIDER_RESPONSE_METADATA_FIELDS,
        )
        _copy_json_safe_fields(
            metadata,
            receipt,
            _UNIFIED_EXECUTION_METADATA_FIELDS,
        )
        output_metadata[agent_id] = receipt
    return {
        "run_id": runtime.run_id,
        "graph_revision": runtime.graph_revision,
        "output_agent_id": runtime.output_agent_id,
        "final_answer": runtime.final_answer,
        "communication_condition": runtime.communication_condition.value,
        "outputs": dict(runtime.outputs),
        "block_completion_order": [
            list(component) for component in runtime.block_completion_order
        ],
        "executed_agent_ids": list(runtime.executed_agent_ids),
        "reused_agent_ids": list(runtime.reused_agent_ids),
        "output_metadata": output_metadata,
    }


class AgentGraphRolloutCollector:
    """Collect one exact-receipt natural-policy AgentGraph trajectory."""

    def __init__(
        self,
        orchestrator: AgentGraphOrchestrator,
        environment: AgentWorkflowEnv,
        versions: VersionBundle,
        evidence_store: Optional[EvidenceStore] = None,
        *,
        condition_id: str = "exploit",
        skills: Sequence[Mapping[str, Any]] = (),
        skill_provider: Optional[SkillPromptProvider] = None,
        active_skill_provider: Optional[ActiveSkillProvider] = None,
        condition_satisfied: bool = True,
        forced_probe: bool = False,
        api_fallback_used: bool = False,
        manual_repair_used: bool = False,
        expected_task_split: str = "train",
    ) -> None:
        if orchestrator.registry is not environment.model_registry:
            raise ValueError("orchestrator and environment must share the model registry")
        if not condition_id.strip():
            raise ValueError("condition_id must be non-empty")
        if expected_task_split not in VALID_SPLITS:
            raise ValueError(
                f"expected_task_split must be one of {sorted(VALID_SPLITS)}"
            )
        prefix_resolver = getattr(orchestrator.client, "executed_prefix_tokens", None)
        if not callable(prefix_resolver):
            raise TypeError("Director client must expose exact executed_prefix_tokens()")
        for name, value in (
            ("condition_satisfied", condition_satisfied),
            ("forced_probe", forced_probe),
            ("api_fallback_used", api_fallback_used),
            ("manual_repair_used", manual_repair_used),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")

        self.orchestrator = orchestrator
        self.environment = environment
        self.versions = versions
        self.evidence_store = evidence_store
        self.condition_id = condition_id.strip()
        if skills and skill_provider is not None:
            raise ValueError("static skills and a dynamic skill_provider are mutually exclusive")
        if active_skill_provider is not None and not callable(active_skill_provider):
            raise TypeError("active_skill_provider must be callable when supplied")
        self.skills = tuple(dict(skill) for skill in skills)
        self.skill_provider = skill_provider
        self.active_skill_provider = active_skill_provider
        self.condition_satisfied = condition_satisfied
        self.forced_probe = forced_probe
        self.api_fallback_used = api_fallback_used
        self.manual_repair_used = manual_repair_used
        self.expected_task_split = expected_task_split
        self._lock = asyncio.Lock()

    async def collect(
        self,
        task: TaskRecord,
        rollout_index: int,
        evaluator_callback: EvaluatorCallback,
        *,
        workflow_problem: Optional[str] = None,
    ) -> TrajectoryRecord:
        """Collect, evaluate, and optionally persist one rollout.

        A policy that reaches ``max_rounds`` without a valid ``finish`` action
        still returns an explicit terminal-failure trajectory with an empty
        final answer.  The evaluator remains the only authority over whether
        that terminal failure has a valid zero reward.
        """

        if task.split != self.expected_task_split:
            raise ValueError(
                "rollout task split mismatch: "
                f"expected {self.expected_task_split!r}, got {task.split!r}"
            )
        if (
            isinstance(rollout_index, bool)
            or not isinstance(rollout_index, int)
            or rollout_index < 0
        ):
            raise ValueError("rollout_index must be a non-negative integer")
        if not callable(evaluator_callback):
            raise TypeError("evaluator_callback must be callable")
        if workflow_problem is not None and (
            not isinstance(workflow_problem, str) or not workflow_problem.strip()
        ):
            raise ValueError("workflow_problem must be non-empty text when supplied")

        async with self._lock:
            return await self._collect_locked(
                task,
                rollout_index,
                evaluator_callback,
                workflow_problem=workflow_problem,
            )

    async def _collect_locked(
        self,
        task: TaskRecord,
        rollout_index: int,
        evaluator_callback: EvaluatorCallback,
        *,
        workflow_problem: Optional[str],
    ) -> TrajectoryRecord:
        env = self.environment
        # SkillFlow separates the immutable public task from the execution
        # context presented to an orchestrator.  Keep the original TaskRecord
        # in the trajectory/evaluator receipt while allowing a thin benchmark
        # adapter to expose a required runtime interface to Flow-Director.
        env.reset(workflow_problem or task.question)
        turns: list[TurnRecord] = []
        snapshots: list[GraphSnapshotEvent] = []
        previous_snapshot_id: Optional[str] = None
        final_answer: Optional[str] = None
        final_runtime: Optional[AgentRuntimeResult] = None
        explicit_finish = False

        group_id = f"{task.task_id}:{self.condition_id}:{self.versions.policy}"
        rollout_id = f"{group_id}:rollout:{rollout_index:04d}"
        trajectory_id = stable_id(
            "trajectory",
            {
                "task_id": task.task_id,
                "group_id": group_id,
                "rollout_id": rollout_id,
                "versions": self.versions.to_dict(),
                "director_sampling": dict(self.orchestrator.sampling_receipt),
            },
        )

        raw_active_skill_ids = (
            ()
            if self.active_skill_provider is None
            else self.active_skill_provider(task, env, self.versions)
        )
        active_skill_ids = canonical_active_skill_ids(
            raw_active_skill_ids,
            field_name="active_skill_ids",
        )

        def visible_skills() -> tuple[Mapping[str, Any], ...]:
            raw = (
                self.skill_provider(task, env, self.versions)
                if self.skill_provider is not None
                else self.skills
            )
            return tuple(dict(skill) for skill in raw)

        current_skills = visible_skills()
        current_retrieved_skill_ids = _retrieved_skill_ids(current_skills)
        if current_retrieved_skill_ids and self.active_skill_provider is None:
            raise ReceiptValidationError(
                "retrieved Skills require an independent ACTIVE-library provider"
            )
        if not set(current_retrieved_skill_ids).issubset(active_skill_ids):
            raise ReceiptValidationError(
                "retrieved Skills are absent from the version-compatible ACTIVE library"
            )
        prompt = self.orchestrator.build_prompt(env, 0, current_skills)
        for round_index in range(self.orchestrator.max_rounds):
            generation_seed = self.orchestrator.generation_seed(round_index)
            schema_request = self.orchestrator.action_schema_request(env)
            response = await self.orchestrator.client.propose(
                prompt,
                seed=generation_seed,
                **schema_request,
            )
            canvas = await env.step(response.text)
            metadata = response.metadata

            if metadata.get("receipt_verified") is not True:
                raise ReceiptValidationError("Director turn lacks an exact behavior receipt")
            if schema_request:
                if metadata.get("action_json_schema_version") != schema_request.get(
                    "action_json_schema_version"
                ):
                    raise ReceiptValidationError(
                        "Director action schema version differs from the request"
                    )
                if metadata.get("action_schema_branch") != schema_request.get(
                    "action_schema_branch"
                ):
                    raise ReceiptValidationError(
                        "Director action schema branch differs from the request"
                    )
            if metadata.get("prompt_text") != prompt:
                raise ReceiptValidationError("Director receipt is bound to a different prompt")
            if _optional_int(metadata.get("generation_seed")) != generation_seed:
                raise ReceiptValidationError(
                    "Director receipt generation seed differs from the request"
                )
            prompt_ids = _token_ids(metadata.get("prompt_token_ids"), "prompt_token_ids")
            output_ids = _token_ids(metadata.get("output_token_ids"), "output_token_ids")
            raw_behavior_log_probs = metadata.get("behavior_log_probs")
            if not isinstance(raw_behavior_log_probs, (list, tuple)) or len(
                raw_behavior_log_probs
            ) != len(output_ids):
                raise ReceiptValidationError(
                    "behavior log-prob receipt must match output token IDs"
                )
            _, behavior_log_probs = _behavior_receipt(
                list(zip(raw_behavior_log_probs, output_ids))
            )
            policy_version = metadata.get("policy_version")
            if not isinstance(policy_version, str) or policy_version != self.versions.policy:
                raise ReceiptValidationError(
                    "Director policy_version differs from VersionBundle.policy"
                )
            adapter_name = metadata.get("adapter_name")
            requested_lora_path = metadata.get("requested_lora_path")
            if adapter_name is not None and (
                not isinstance(adapter_name, str) or not adapter_name.strip()
            ):
                raise ReceiptValidationError(
                    "Director receipt has an invalid policy adapter"
                )
            if requested_lora_path != adapter_name:
                raise ReceiptValidationError(
                    "Director receipt adapter differs from the requested lora_path"
                )
            server_weight_version = metadata.get("server_weight_version")
            if not isinstance(server_weight_version, str) or not server_weight_version.strip():
                raise ReceiptValidationError(
                    "Director receipt has no SGLang server_weight_version"
                )
            raw_director_request_id = metadata.get("request_id")
            director_request_id = (
                raw_director_request_id.strip()
                if isinstance(raw_director_request_id, str)
                and raw_director_request_id.strip()
                else None
            )

            action = canvas.action
            action_decoding_strategy = metadata.get("action_decoding_strategy")
            if action_decoding_strategy is not None:
                if action_decoding_strategy != "hierarchical_json_schema":
                    raise ReceiptValidationError(
                        "Director receipt has an unsupported decoding strategy"
                    )
                selected_action = metadata.get("selected_action")
                if not isinstance(selected_action, str) or not selected_action:
                    raise ReceiptValidationError(
                        "hierarchical Director receipt has no selected action"
                    )
                if action is not None and selected_action != action.action_type.value:
                    raise ReceiptValidationError(
                        "hierarchical action selection differs from the parsed action"
                    )
                phase_receipts = metadata.get("hierarchical_phase_receipts")
                if not isinstance(phase_receipts, Mapping):
                    raise ReceiptValidationError(
                        "hierarchical Director receipt has no phase receipts"
                    )
                for phase_name, phase_receipt in phase_receipts.items():
                    if not isinstance(phase_name, str) or not isinstance(
                        phase_receipt, Mapping
                    ):
                        raise ReceiptValidationError(
                            "hierarchical Director phase receipt is malformed"
                        )
                    if phase_receipt.get("receipt_verified") is not True:
                        raise ReceiptValidationError(
                            "hierarchical Director phase lacks an exact receipt"
                        )
                    phase_output_ids = _token_ids(
                        phase_receipt.get("output_token_ids"),
                        f"{phase_name}.output_token_ids",
                    )
                    phase_log_probs = phase_receipt.get("behavior_log_probs")
                    if not isinstance(phase_log_probs, (list, tuple)) or len(
                        phase_log_probs
                    ) != len(phase_output_ids):
                        raise ReceiptValidationError(
                            "hierarchical phase log-prob receipt is incomplete"
                        )
            executed_prefix_tokens = 0
            if action is not None:
                executed_prefix_tokens = self.orchestrator.client.executed_prefix_tokens(
                    response, action
                )

            snapshot = GraphSnapshotEvent.create(
                canvas.revision,
                canvas.snapshot.graph.to_dict(),
                previous_snapshot_id,
            )
            receipt_execution = canvas.execution or canvas.partial_execution
            execution_records = (
                tuple(_execution_record(call) for call in receipt_execution.calls)
                if receipt_execution is not None and not canvas.execution_reused
                else ()
            )
            runtime_summary = dict(_runtime_summary(receipt_execution))
            if (
                canvas.partial_execution is not None
                or canvas.execution_failure_records
            ):
                runtime_summary["execution_status"] = "failed"
                runtime_summary["failure_records"] = [
                    record.to_dict()
                    for record in canvas.execution_failure_records
                ]
                runtime_summary["unresolved_dirty_agent_ids"] = list(
                    env.unresolved_dirty_agent_ids
                )
            action_schema_version = metadata.get("action_json_schema_version")
            if action_schema_version is not None:
                runtime_summary["director_action_schema_version"] = (
                    action_schema_version
                )
            action_schema_branch = metadata.get("action_schema_branch")
            if action_schema_branch is not None:
                runtime_summary["director_action_schema_branch"] = (
                    action_schema_branch
                )
            if action_decoding_strategy is not None:
                runtime_summary["director_action_decoding"] = {
                    "strategy": action_decoding_strategy,
                    "selected_action": metadata.get("selected_action"),
                    "selected_modify_field": metadata.get(
                        "selected_modify_field"
                    ),
                    "parameter_schema_branch": metadata.get(
                        "parameter_schema_branch"
                    ),
                    "request_count": metadata.get("request_count"),
                    "phase_receipts": metadata.get(
                        "hierarchical_phase_receipts"
                    ),
                }
            turn = TurnRecord(
                turn_id=stable_id(
                    "turn",
                    {
                        "trajectory_id": trajectory_id,
                        "round_index": round_index,
                    },
                ),
                round_index=round_index,
                prompt=prompt,
                policy_response=response.text,
                prompt_token_ids=prompt_ids,
                output_token_ids=output_ids,
                behavior_log_probs=behavior_log_probs,
                executed_prefix_tokens=executed_prefix_tokens,
                action={} if action is None else action.to_dict(),
                canvas_feedback=canvas.feedback,
                graph_revision=canvas.revision,
                graph_snapshot=snapshot.to_dict()["graph"],
                graph_snapshot_id=snapshot.snapshot_id,
                previous_graph_snapshot_id=previous_snapshot_id,
                executions=execution_records,
                runtime_summary=runtime_summary,
                execution_reused=canvas.execution_reused,
                director_request_id=director_request_id,
                director_latency_ms=_optional_float(metadata.get("latency_ms")),
                director_attempt_count=_optional_int(metadata.get("attempt_count")),
                director_generation_seed=generation_seed,
                policy_version=policy_version,
                policy_adapter=adapter_name,
                server_weight_version=server_weight_version,
                reconstructed_context=False,
                receipt_verified=True,
                retrieved_skill_ids=current_retrieved_skill_ids,
                # AgentGraphOrchestrator exposes every ordinary retrieved
                # Skill prior in ``available_skills``. Forced paired-probe
                # conditions are rendered separately and excluded above.
                visible_skill_ids=current_retrieved_skill_ids,
            )
            turns.append(turn)
            snapshots.append(snapshot)
            previous_snapshot_id = snapshot.snapshot_id

            if canvas.done:
                explicit_finish = True
                final_answer = canvas.final_answer
                final_runtime = canvas.execution
                break
            current_skills = visible_skills()
            current_retrieved_skill_ids = _retrieved_skill_ids(current_skills)
            if current_retrieved_skill_ids and self.active_skill_provider is None:
                raise ReceiptValidationError(
                    "retrieved Skills require an independent ACTIVE-library provider"
                )
            if not set(current_retrieved_skill_ids).issubset(active_skill_ids):
                raise ReceiptValidationError(
                    "retrieved Skills are absent from the version-compatible ACTIVE library"
                )
            prompt = self.orchestrator.continue_prompt(
                prompt,
                self.orchestrator.consumed_assistant_content(response, canvas),
                env,
                current_skills,
            )

        termination_reason = "finish" if explicit_finish else "max_rounds"
        if termination_reason == "max_rounds":
            # A progressive execute-on-edit result is Canvas feedback, not an
            # implicit finish.  Preserve the natural truncation as a terminal
            # task failure and let the real evaluator judge the empty answer.
            final_answer = None
            final_runtime = None

        final_graph = env.graph.to_dict()
        raw_evaluation = evaluator_callback(
            task,
            final_answer,
            final_graph,
            final_runtime,
        )
        if inspect.isawaitable(raw_evaluation):
            raw_evaluation = await raw_evaluation
        evaluation = _evaluation_receipt(raw_evaluation)
        trajectory = TrajectoryRecord(
            trajectory_id=trajectory_id,
            task=task,
            group_id=group_id,
            condition_id=self.condition_id,
            rollout_id=rollout_id,
            versions=self.versions,
            turns=tuple(turns),
            final_answer=final_answer,
            evaluation=evaluation,
            termination_reason=termination_reason,
            explicit_finish=explicit_finish,
            director_sampling=dict(self.orchestrator.sampling_receipt),
            condition_satisfied=self.condition_satisfied,
            forced_probe=self.forced_probe,
            api_fallback_used=self.api_fallback_used,
            manual_repair_used=self.manual_repair_used,
            active_skill_ids=active_skill_ids,
            # SkillFlow's trajectory-level retrieved IDs are the ranked H0
            # retrieval. Later stage-conditioned retrievals remain on turns.
            retrieved_skill_ids=(
                tuple(turns[0].retrieved_skill_ids) if turns else ()
            ),
            invoked_skill_ids=canonical_invoked_skill_ids(turns),
        )

        if self.evidence_store is not None:
            for snapshot in snapshots:
                self.evidence_store.append_snapshot(snapshot)
            self.evidence_store.append_trajectory(trajectory)
        return trajectory


__all__ = [
    "AGENTGRAPH_SMOKE_SOURCES",
    "AgentGraphRolloutCollector",
    "ActiveSkillProvider",
    "EvaluatorCallback",
    "ReceiptValidationError",
    "RolloutGate",
    "SGLangReceiptDirectorClient",
    "SkillPromptProvider",
    "execution_record_from_call",
    "select_balanced_tasks",
]
