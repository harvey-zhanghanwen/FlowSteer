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
from dataclasses import replace
import inspect
import json
import math
import socket
import threading
import time
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_action_parser import (
    AgentAction,
    AgentActionParseError,
    AgentActionParser,
)
from .agent_runtime import AgentCallRecord, AgentRuntimeResult
from .agent_workflow_env import AgentWorkflowEnv
from .director import (
    AgentGraphOrchestrator,
    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    DIRECTOR_SYSTEM_PROMPT,
    DirectorError,
    DirectorResponse,
    decode_director_transcript,
    encode_director_transcript,
    director_actions_from_admissible_schema_branch,
    director_model_admissible_sampling_json_schema_text,
    director_model_admissible_sampling_json_schema_text_v1,
    director_model_admissible_sampling_json_schema_text_v3,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_live_add_subgraph_role_selection_from_text,
    director_live_add_subgraph_role_selection_json_schema_text,
    director_live_add_subgraph_relation_candidates,
    director_live_action_parameter_json_schema_text,
    director_live_action_target_domains_json,
    director_live_modify_agent_selector_json_schema_text,
    director_live_relation_candidate_selector_json_schema_text,
    director_modify_agent_field_sampling_json_schema_text,
    director_modify_agent_field_selector_json_schema_text,
    director_state_conditioned_sampling_json_schema_text,
    verified_qa_semantic_protocol,
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


HIERARCHICAL_JSON_SCHEMA_STRATEGY = "hierarchical_json_schema"
ROLE_FIRST_ADD_DECODING_STRATEGY = (
    "hierarchical_json_schema_role_first_add_v1"
)
_ADD_DECLARATION_PARSE_FAILURE_PHASE = "add_agent_declarations"
_ADD_ROLE_SELECTION_PARSE_FAILURE_PHASE = "add_agent_role_selection"
_ADD_ROLE_SELECTION_SERIALIZATION_FAILURE_PHASE = (
    "add_agent_role_selection_serialization_failure"
)
_ACTION_SELECTION_SERIALIZATION_FAILURE_PHASE = (
    "action_selection_serialization_failure"
)
_PARAMETER_SERIALIZATION_FAILURE_PHASE = "parameter_serialization_failure"
_RELATION_CANDIDATE_SERIALIZATION_FAILURE_PHASE = (
    "relation_candidate_serialization_failure"
)
_SGLANG_DETERMINISTIC_SEED_MASK = (1 << 63) - 1
_QWEN_REASONING_END_TOKEN = "</think>"

_ADD_DECLARATION_CONTINUATION = (
    "Complete the Agent declarations for the selected positions and "
    "role_family values. Keep agent_id and role_family unchanged. Return only "
    "the JSON object required by the current schema."
)
_ADD_ACTION_CONTINUATION = (
    "Complete the add_subgraph action for these Agent declarations. Keep "
    "agents unchanged. Select only relations and output_agent_id allowed by "
    "the current schema. Return only the JSON object."
)
_PARAMETER_REGENERATION_CONTINUATION = (
    "Return one complete JSON object that conforms to the current schema."
)


def _sglang_backend_sampling_seed(seed: int) -> int:
    """Project one scientific uint64 seed into SGLang's signed int64 domain.

    The scientific coordinate keeps its original uint64 value in the
    trajectory receipt.  SGLang deterministic inference materializes request
    seeds as ``torch.int64``; masking only the sign bit is therefore the
    minimal transport adaptation and is persisted separately as the backend
    sampling seed.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("Director seed must be a uint64 integer")
    return seed & _SGLANG_DETERMINISTIC_SEED_MASK


def _hierarchical_continuation_prompt(
    prompt: str,
    *,
    committed_json: str,
    instruction: str,
) -> str:
    """Append one sampled hierarchical decision as model-visible context.

    FlowSteer's progressive Canvas exposes every committed edit before the next
    decision.  Hierarchical constrained decoding needs the same conditional
    boundary: JSON Schema constrains tokens, but a later ``const`` field is not
    semantic context for a contract generated earlier in key order.  Preserve
    the exact prior transcript, append the canonical sampled receipt as an
    assistant turn, and request only the next schema-bound phase.
    """

    if not isinstance(committed_json, str) or not committed_json:
        raise ReceiptValidationError(
            "hierarchical continuation requires a non-empty committed receipt"
        )
    if not isinstance(instruction, str) or not instruction:
        raise ReceiptValidationError(
            "hierarchical continuation requires a non-empty instruction"
        )
    transcript = decode_director_transcript(prompt)
    messages = (
        list(transcript)
        if transcript is not None
        else [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    messages.extend(
        (
            {"role": "assistant", "content": committed_json},
            {"role": "user", "content": instruction},
        )
    )
    return encode_director_transcript(messages)


def _action_parameter_serialization_failed(
    text: str,
    *,
    reasoning_end_token: Optional[str] = None,
) -> bool:
    """Return whether the first action object is syntactically incomplete.

    A complete JSON object that violates the action contract remains a Canvas
    rejection; regeneration is only for serialization failures such as an EOS-
    truncated object.  The strict ``AgentActionParser`` remains authoritative
    and is not repaired or bypassed here.
    """

    try:
        AgentActionParser().parse(
            text,
            reasoning_end_token=reasoning_end_token,
        )
    except AgentActionParseError:
        if not isinstance(text, str):
            return True
        scan_start = 0
        if reasoning_end_token is not None:
            if not isinstance(reasoning_end_token, str) or not reasoning_end_token:
                return True
            boundary_start = text.find(reasoning_end_token)
            if boundary_start < 0:
                return True
            scan_start = boundary_start + len(reasoning_end_token)
        stripped_start = scan_start + (
            len(text[scan_start:]) - len(text[scan_start:].lstrip())
        )
        object_start = text.find("{", stripped_start)
        array_start = text.find("[", stripped_start)
        candidates = [
            position
            for position in (object_start, array_start)
            if position >= 0
        ]
        if not candidates:
            return True
        try:
            json.JSONDecoder().raw_decode(text[min(candidates) :])
        except (TypeError, ValueError):
            return True
    return False


def _hierarchical_selector_serialization_failed(
    text: str,
    *,
    reasoning_end_token: Optional[str] = None,
) -> bool:
    """Return whether one schema-bound selector is not a JSON value.

    This only classifies serialization.  Selector fields and admitted values
    remain authoritative in ``_hierarchical_choice`` and
    ``_hierarchical_index_choice``; callers must not infer either from the
    malformed text.
    """

    try:
        _strict_hierarchical_selector_object(
            text,
            reasoning_end_token=reasoning_end_token,
        )
    except ReceiptValidationError:
        return True
    return False


def _strict_hierarchical_selector_object(
    text: str,
    *,
    reasoning_end_token: Optional[str] = None,
) -> Mapping[str, Any]:
    """Decode the first strict selector object after its reasoning boundary.

    SGLang's ``ReasonerGrammarBackend`` applies the selector grammar after
    ``</think>`` while preserving the reasoning prefix in the exact response
    text.  Mirror :class:`AgentActionParser`'s earliest-JSON-value boundary on
    that post-reasoning suffix: never skip a malformed/non-object first value,
    never salvage a later object, and use the same strict decoder so duplicate
    keys and non-finite constants remain invalid.
    """

    if not isinstance(text, str) or not text:
        raise ReceiptValidationError(
            "hierarchical Director selector must be non-empty text"
        )
    scan_start = 0
    if reasoning_end_token is not None:
        if not isinstance(reasoning_end_token, str) or not reasoning_end_token:
            raise ReceiptValidationError(
                "reasoning_end_token must be a non-empty string"
            )
        reasoning_end = text.find(reasoning_end_token)
        if reasoning_end < 0:
            raise ReceiptValidationError(
                "hierarchical Director selector is missing the required "
                f"reasoning boundary {reasoning_end_token!r}"
            )
        scan_start = reasoning_end + len(reasoning_end_token)
    scan_start += len(text[scan_start:]) - len(text[scan_start:].lstrip())
    # AgentActionParser owns the strict duplicate-key/non-finite JSON decoder.
    # Reuse it here because selector receipts are JSON objects but are not full
    # Canvas actions and therefore cannot pass AgentActionParser.parse().
    decoder = AgentActionParser()._decoder
    try:
        leading_value, _ = decoder.raw_decode(text[scan_start:])
    except (AgentActionParseError, json.JSONDecodeError, TypeError, ValueError):
        object_start = text.find("{", scan_start)
        array_start = text.find("[", scan_start)
        candidates = [
            position
            for position in (object_start, array_start)
            if position >= 0
        ]
        if not candidates:
            raise ReceiptValidationError(
                "hierarchical Director selector contains no JSON object"
            )
        value_start = min(candidates)
    else:
        if not isinstance(leading_value, dict):
            raise ReceiptValidationError(
                "hierarchical Director selector's first JSON value is not an object"
            )
        return leading_value
    try:
        value, _ = decoder.raw_decode(text[value_start:])
    except (AgentActionParseError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ReceiptValidationError(
            "hierarchical Director selector's first JSON object is malformed"
        ) from exc
    if not isinstance(value, dict):
        raise ReceiptValidationError(
            "hierarchical Director selector's first JSON value is not an object"
        )
    return value


def _hierarchical_schema_text(
    text: str,
    *,
    reasoning_end_token: Optional[str] = None,
) -> str:
    """Expose only the strict post-reasoning object to legacy phase parsers."""

    if reasoning_end_token is None:
        return text
    value = _strict_hierarchical_selector_object(
        text,
        reasoning_end_token=reasoning_end_token,
    )
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ReceiptValidationError(DirectorError):
    """Raised when SGLang cannot prove an exact on-policy token receipt."""


def _director_reasoning_end_token(metadata: Mapping[str, Any]) -> Optional[str]:
    """Resolve the strict response boundary from the exact Director receipt."""

    enable_thinking = metadata.get("chat_template_enable_thinking")
    if enable_thinking is None:
        return None
    if type(enable_thinking) is not bool:
        raise ReceiptValidationError(
            "Director thinking-mode receipt must be boolean"
        )
    return _QWEN_REASONING_END_TOKEN if enable_thinking else None


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
    behavior server.  The client requires an explicit Qwen chat-template
    thinking mode and never falls back to an approximately reconstructed
    prompt.  ``False`` remains the compatibility default; experiments that
    enable it record the exact mode in every generation receipt.
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
        max_context_tokens: int | None = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
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
        if max_context_tokens is not None and (
            type(max_context_tokens) is not int
            or max_context_tokens <= max_tokens
        ):
            raise ValueError(
                "Director max_context_tokens must be an integer larger than "
                "max_tokens"
            )
        if type(enable_thinking) is not bool:
            raise ValueError("Director enable_thinking must be boolean")
        if thinking_budget_tokens is not None and (
            type(thinking_budget_tokens) is not int
            or thinking_budget_tokens < 1
            or thinking_budget_tokens >= max_tokens
        ):
            raise ValueError(
                "Director thinking_budget_tokens must be a positive integer "
                "smaller than max_tokens"
            )
        if thinking_budget_tokens is not None and not enable_thinking:
            raise ValueError(
                "Director thinking_budget_tokens requires enable_thinking=true"
            )
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
        self.max_context_tokens = max_context_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
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
                enable_thinking=self.enable_thinking,
            )
        except TypeError as exc:
            raise ReceiptValidationError(
                "Qwen3.5 tokenizer must support the configured enable_thinking mode"
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
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if seed is not None:
            _sglang_backend_sampling_seed(seed)
        prompt_ids = self.prompt_token_ids(prompt)
        if (
            self.max_context_tokens is not None
            and len(prompt_ids) + self.max_tokens > self.max_context_tokens
        ):
            raise DirectorError(
                "Director context limit exceeded before SGLang dispatch: "
                f"prompt_tokens={len(prompt_ids)}, "
                f"max_new_tokens={self.max_tokens}, "
                f"max_context_tokens={self.max_context_tokens}"
            )
        payload: dict[str, Any] = {
            "input_ids": list(prompt_ids),
            # SGLang 0.5.15's ReasonerGrammarBackend needs this native
            # GenerateReqInput flag to defer JSON grammar enforcement until
            # after the model's reasoning boundary. Keep it identical to the
            # Qwen chat-template thinking mode so the exact request receipt
            # proves which decoding path was used.
            "require_reasoning": self.enable_thinking,
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
            # the equivalent field as ``sampling_seed`` and deterministic
            # inference materializes it as a signed torch.int64.
            payload["sampling_params"]["sampling_seed"] = (
                _sglang_backend_sampling_seed(seed)
            )
        if self.thinking_budget_tokens is not None:
            # DIRECT_REUSE: SkillFlow bounds Supervisor reasoning separately
            # from its visible action allowance. NECESSARY_ADAPTATION: native
            # SGLang 0.5.15 reads this value from SamplingParams custom_params
            # so ReasonerGrammar can force the </think> boundary before the
            # live JSON grammar consumes the remaining action-token budget.
            payload["sampling_params"]["custom_params"] = {
                "thinking_budget": self.thinking_budget_tokens
            }
        (
            resolved_action_schema,
            _,
            _,
            _,
            _,
        ) = self._resolve_action_schema(
            action_json_schema=action_json_schema,
            action_json_schema_version=action_json_schema_version,
            action_schema_branch=action_schema_branch,
            action_target_domains_json=action_target_domains_json,
            action_target_domain_version=action_target_domain_version,
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
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
    ) -> Mapping[str, Any]:
        _, adapter_name, _ = self._policy_route()
        return self._request_payload(
            prompt,
            adapter_name,
            seed,
            action_json_schema=action_json_schema,
            action_json_schema_version=action_json_schema_version,
            action_schema_branch=action_schema_branch,
            action_target_domains_json=action_target_domains_json,
            action_target_domain_version=action_target_domain_version,
        )

    def _resolve_action_schema(
        self,
        *,
        action_json_schema: Optional[str],
        action_json_schema_version: Optional[str],
        action_schema_branch: Optional[str],
        action_target_domains_json: Optional[str],
        action_target_domain_version: Optional[str],
    ) -> tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        override_requested = any(
            value is not None
            for value in (
                action_json_schema,
                action_json_schema_version,
                action_schema_branch,
                action_target_domains_json,
                action_target_domain_version,
            )
        )
        if not override_requested:
            return (
                self.action_json_schema,
                self.action_json_schema_version,
                None,
                None,
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
            normalized_domains_json: str | None = None
            normalized_domain_version: str | None = None
            if normalized_branch.startswith("admissible-v3:"):
                if (
                    action_json_schema_version.strip()
                    != DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
                ):
                    raise ValueError(
                        "v3 admissible branch requires its exact schema version"
                    )
                if (
                    not isinstance(action_target_domains_json, str)
                    or not action_target_domains_json.strip()
                    or action_target_domain_version
                    != DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "v3 admissible branch requires exact live target domains"
                    )
                actions = director_actions_from_admissible_schema_branch(
                    normalized_branch
                )
                expected_schema_text = (
                    director_model_admissible_sampling_json_schema_text_v3(actions)
                )
                if action_json_schema.strip() != expected_schema_text:
                    raise ValueError("v3 action schema is not canonical")
                parsed_domains = json.loads(action_target_domains_json)
                normalized_domains_json = director_live_action_target_domains_json(
                    actions,
                    parsed_domains,
                )
                if action_target_domains_json.strip() != normalized_domains_json:
                    raise ValueError("v3 live target domains are not canonical")
                normalized_domain_version = action_target_domain_version
            elif normalized_branch.startswith("admissible-v2:"):
                if (
                    action_target_domains_json is not None
                    or action_target_domain_version is not None
                ):
                    raise ValueError("v2 receipts cannot carry v3 target domains")
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
                    action_target_domains_json is not None
                    or action_target_domain_version is not None
                ):
                    raise ValueError("v1 receipts cannot carry v3 target domains")
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
                if (
                    action_target_domains_json is not None
                    or action_target_domain_version is not None
                ):
                    raise ValueError(
                        "state-conditioned receipts cannot carry v3 target domains"
                    )
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
            normalized_domains_json,
            normalized_domain_version,
        )

    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
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
                resolved_target_domains_json,
                resolved_target_domain_version,
            ) = self._resolve_action_schema(
                action_json_schema=action_json_schema,
                action_json_schema_version=action_json_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
            )
            payload = self._request_payload(
                prompt,
                adapter_name,
                seed,
                action_json_schema=action_json_schema,
                action_json_schema_version=action_json_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
            )
            if (
                resolved_action_schema_version
                in {
                    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
                    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
                }
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
                    action_target_domains=(
                        None
                        if resolved_target_domains_json is None
                        else json.loads(resolved_target_domains_json)
                    ),
                    action_target_domains_json=resolved_target_domains_json,
                    action_target_domain_version=resolved_target_domain_version,
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
                action_target_domains_json=resolved_target_domains_json,
                action_target_domain_version=resolved_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
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
        reasoning_end_token: Optional[str] = None,
    ) -> str:
        """Parse one constrained discriminator without repairing sampled text."""

        value = _strict_hierarchical_selector_object(
            text,
            reasoning_end_token=reasoning_end_token,
        )
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
    def _hierarchical_index_choice(
        text: str,
        *,
        admitted: Sequence[int],
        required_action: str,
        reasoning_end_token: Optional[str] = None,
    ) -> int:
        """Parse one integer candidate selector without rewriting its text."""

        value = _strict_hierarchical_selector_object(
            text,
            reasoning_end_token=reasoning_end_token,
        )
        if not isinstance(value, Mapping) or set(value) != {
            "action",
            "candidate_index",
        }:
            raise ReceiptValidationError(
                "hierarchical Director candidate selector has incompatible fields"
            )
        if value.get("action") != required_action:
            raise ReceiptValidationError(
                "hierarchical Director candidate selector changed its action"
            )
        selected = value.get("candidate_index")
        if type(selected) is not int or selected not in admitted:
            raise ReceiptValidationError(
                "hierarchical Director candidate selector selected an inadmissible value"
            )
        return selected

    @staticmethod
    def _hierarchical_phase_receipt(response: DirectorResponse) -> Mapping[str, Any]:
        metadata = response.metadata
        receipt = {
            "text": response.text,
            "prompt_text": metadata.get("prompt_text"),
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
            "backend_sampling_seed": metadata.get("backend_sampling_seed"),
            "chat_template_enable_thinking": metadata.get(
                "chat_template_enable_thinking"
            ),
            "require_reasoning": metadata.get("require_reasoning"),
            "server_weight_version": metadata.get("server_weight_version"),
            "receipt_verified": metadata.get("receipt_verified"),
        }
        if metadata.get("action_target_domain_version") is not None:
            receipt["action_json_schema_version"] = metadata.get(
                "action_json_schema_version"
            )
            receipt["action_schema_branch"] = metadata.get(
                "action_schema_branch"
            )
            receipt["action_target_domain_version"] = metadata.get(
                "action_target_domain_version"
            )
            receipt["action_target_domains_json"] = metadata.get(
                "action_target_domains_json"
            )
        return receipt

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
        action_target_domains: Optional[Mapping[str, Any]],
        action_target_domains_json: Optional[str],
        action_target_domain_version: Optional[str],
    ) -> DirectorResponse:
        """Sample action type, optional MODIFY field, then exact parameters."""

        reasoning_end_token = (
            _QWEN_REASONING_END_TOKEN if self.enable_thinking else None
        )
        total_latency_ms = 0.0
        total_attempt_count = 0
        phase_receipts: dict[str, Mapping[str, Any]] = {}
        action_selection_regeneration_attempted = False
        action_selection_regeneration_succeeded = False
        role_selection_regeneration_attempted = False
        role_selection_regeneration_succeeded = False
        relation_candidate_regeneration_attempted = False
        relation_candidate_regeneration_succeeded = False

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
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            try:
                selected_action = self._hierarchical_choice(
                    selector_response.text,
                    field_name="action",
                    admitted=actions,
                    reasoning_end_token=reasoning_end_token,
                )
            except ReceiptValidationError as exc:
                if not _hierarchical_selector_serialization_failed(
                    selector_response.text,
                    reasoning_end_token=reasoning_end_token,
                ):
                    raise
                # Reuse the bounded serialization regeneration already used
                # by the relation and parameter selectors. The first exact
                # sample remains in the receipt; no action is inferred from
                # reasoning text and no Canvas mutation occurs before a valid
                # live-schema selector is returned.
                action_selection_regeneration_attempted = True
                phase_receipts[
                    _ACTION_SELECTION_SERIALIZATION_FAILURE_PHASE
                ] = self._hierarchical_phase_receipt(selector_response)
                regeneration_prompt = _hierarchical_continuation_prompt(
                    prompt,
                    committed_json=selector_response.text,
                    instruction=_PARAMETER_REGENERATION_CONTINUATION,
                )
                regeneration_payload = dict(
                    self._request_payload(regeneration_prompt, adapter_name, seed)
                )
                regeneration_sampling = dict(
                    regeneration_payload["sampling_params"]
                )
                regeneration_sampling["json_schema"] = selector_payload[
                    "sampling_params"
                ]["json_schema"]
                regeneration_payload["sampling_params"] = regeneration_sampling
                value, latency_ms, attempt_count = await self._post_with_retries(
                    regeneration_payload
                )
                total_latency_ms += latency_ms
                total_attempt_count += attempt_count
                selector_response = self._parse_response(
                    regeneration_prompt,
                    regeneration_payload,
                    value,
                    policy_version=policy_version,
                    adapter_name=adapter_name,
                    expected_server_weight_version=(
                        expected_server_weight_version
                    ),
                    action_json_schema_version=action_schema_version,
                    action_schema_branch=action_schema_branch,
                    action_target_domains_json=action_target_domains_json,
                    action_target_domain_version=action_target_domain_version,
                    latency_ms=latency_ms,
                    attempt_count=attempt_count,
                    generation_seed=seed,
                )
                try:
                    selected_action = self._hierarchical_choice(
                        selector_response.text,
                        field_name="action",
                        admitted=actions,
                        reasoning_end_token=reasoning_end_token,
                    )
                except ReceiptValidationError as regeneration_exc:
                    regeneration_exc.hierarchical_phase_receipts = {
                        **phase_receipts,
                        "action_selection_regeneration_failure": (
                            self._hierarchical_phase_receipt(selector_response)
                        ),
                    }
                    regeneration_exc.failure_phase = "action_selection"
                    raise regeneration_exc from exc
                action_selection_regeneration_succeeded = True
            phase_receipts["action_selection"] = self._hierarchical_phase_receipt(
                selector_response
            )

        selected_add_agent_roles: tuple[dict[str, str], ...] | None = None
        selected_add_agents: tuple[dict[str, Any], ...] | None = None
        selected_modify_field: str | None = None
        selected_modify_agent_id: str | None = None
        selected_relation_candidate: int | None = None
        parameter_prompt = prompt
        role_neutral_add_subgraph = False
        if selected_action == "add_subgraph" and action_target_domains is not None:
            add_subgraph_domain = action_target_domains.get("add_subgraph")
            role_neutral_add_subgraph = (
                isinstance(add_subgraph_domain, Mapping)
                and add_subgraph_domain.get("contract_semantics") == "free_text"
            )
        if (
            selected_action == "add_subgraph"
            and action_target_domains is not None
            and not role_neutral_add_subgraph
        ):
            role_selection_schema = (
                director_live_add_subgraph_role_selection_json_schema_text(
                    action_target_domains
                )
            )
            role_selection_payload = dict(
                self._request_payload(prompt, adapter_name, seed)
            )
            role_selection_sampling = dict(
                role_selection_payload["sampling_params"]
            )
            role_selection_sampling["json_schema"] = role_selection_schema
            role_selection_payload["sampling_params"] = role_selection_sampling
            value, latency_ms, attempt_count = await self._post_with_retries(
                role_selection_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            role_selection_response = self._parse_response(
                prompt,
                role_selection_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            try:
                selected_add_agent_roles = (
                    director_live_add_subgraph_role_selection_from_text(
                        _hierarchical_schema_text(
                            role_selection_response.text,
                            reasoning_end_token=reasoning_end_token,
                        ),
                        action_target_domains,
                    )
                )
            except (ValueError, ReceiptValidationError) as exc:
                if not _hierarchical_selector_serialization_failed(
                    role_selection_response.text,
                    reasoning_end_token=reasoning_end_token,
                ):
                    raise ReceiptValidationError(
                        "v3 add_subgraph Agent role-selection phase is invalid: "
                        f"{exc}"
                    ) from exc
                # Reuse the existing SkillFlow-compatible bounded structured
                # regeneration boundary used by relation/parameter phases.
                # The first exact sample stays in the receipt; one request
                # with the identical live JSON Schema, route, and scientific
                # seed may repair serialization only.  No role is inferred
                # from free text and schema-valid semantic failures are not
                # regenerated.
                role_selection_regeneration_attempted = True
                phase_receipts[
                    _ADD_ROLE_SELECTION_SERIALIZATION_FAILURE_PHASE
                ] = self._hierarchical_phase_receipt(role_selection_response)
                regeneration_prompt = _hierarchical_continuation_prompt(
                    prompt,
                    committed_json=role_selection_response.text,
                    instruction=_PARAMETER_REGENERATION_CONTINUATION,
                )
                regeneration_payload = dict(
                    self._request_payload(regeneration_prompt, adapter_name, seed)
                )
                regeneration_sampling = dict(
                    regeneration_payload["sampling_params"]
                )
                regeneration_sampling["json_schema"] = role_selection_schema
                regeneration_payload["sampling_params"] = regeneration_sampling
                value, latency_ms, attempt_count = await self._post_with_retries(
                    regeneration_payload
                )
                total_latency_ms += latency_ms
                total_attempt_count += attempt_count
                role_selection_response = self._parse_response(
                    regeneration_prompt,
                    regeneration_payload,
                    value,
                    policy_version=policy_version,
                    adapter_name=adapter_name,
                    expected_server_weight_version=(
                        expected_server_weight_version
                    ),
                    action_json_schema_version=action_schema_version,
                    action_schema_branch=action_schema_branch,
                    action_target_domains_json=action_target_domains_json,
                    action_target_domain_version=action_target_domain_version,
                    latency_ms=latency_ms,
                    attempt_count=attempt_count,
                    generation_seed=seed,
                )
                try:
                    selected_add_agent_roles = (
                        director_live_add_subgraph_role_selection_from_text(
                            _hierarchical_schema_text(
                                role_selection_response.text,
                                reasoning_end_token=reasoning_end_token,
                            ),
                            action_target_domains,
                        )
                    )
                except (ValueError, ReceiptValidationError):
                    # Match FlowSteer's existing malformed declaration/final
                    # parameter boundary: preserve both exact samples and
                    # publish the second strict-parser failure as a rejected
                    # Canvas turn.  It cannot declare or execute an Agent and
                    # there is no third generation attempt.
                    phase_receipts["add_agent_role_selection"] = (
                        self._hierarchical_phase_receipt(
                            role_selection_response
                        )
                    )
                    metadata = dict(role_selection_response.metadata)
                    metadata.update(
                        {
                            "base_prompt_text": prompt,
                            "action_decoding_strategy": (
                                ROLE_FIRST_ADD_DECODING_STRATEGY
                            ),
                            "selected_action": selected_action,
                            "selected_modify_field": None,
                            "selected_modify_agent_id": None,
                            "selected_add_agent_ids": None,
                            "selected_add_agent_roles": None,
                            "parameter_schema_branch": None,
                            "parse_failure_phase": (
                                _ADD_ROLE_SELECTION_PARSE_FAILURE_PHASE
                            ),
                            "role_selection_regeneration_attempted": True,
                            "role_selection_regeneration_succeeded": False,
                            "hierarchical_phase_receipts": phase_receipts,
                            "request_count": len(phase_receipts),
                            "latency_ms": total_latency_ms,
                            "attempt_count": total_attempt_count,
                        }
                    )
                    return DirectorResponse(
                        text=role_selection_response.text,
                        metadata=metadata,
                    )
                role_selection_regeneration_succeeded = True
            phase_receipts["add_agent_role_selection"] = (
                self._hierarchical_phase_receipt(role_selection_response)
            )
            selected_roles_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in selected_add_agent_roles],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            declaration_prompt = _hierarchical_continuation_prompt(
                str(role_selection_response.metadata["prompt_text"]),
                committed_json=selected_roles_json,
                instruction=_ADD_DECLARATION_CONTINUATION,
            )
            declaration_schema = (
                director_live_add_subgraph_agent_declarations_json_schema_text(
                    action_target_domains,
                    selected_agent_roles=selected_add_agent_roles,
                )
            )
            declaration_payload = dict(
                self._request_payload(declaration_prompt, adapter_name, seed)
            )
            declaration_sampling = dict(declaration_payload["sampling_params"])
            declaration_sampling["json_schema"] = declaration_schema
            declaration_payload["sampling_params"] = declaration_sampling
            value, latency_ms, attempt_count = await self._post_with_retries(
                declaration_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            declaration_response = self._parse_response(
                declaration_prompt,
                declaration_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            phase_receipts["add_agent_declarations"] = (
                self._hierarchical_phase_receipt(declaration_response)
            )
            try:
                selected_add_agents = (
                    director_live_add_subgraph_agent_declarations_from_text(
                        _hierarchical_schema_text(
                            declaration_response.text,
                            reasoning_end_token=reasoning_end_token,
                        ),
                        action_target_domains,
                        selected_agent_roles=selected_add_agent_roles,
                    )
                )
            except (ValueError, ReceiptValidationError):
                # Match the existing malformed final-parameter boundary: keep
                # the exact sampled text/token/log-prob receipt and let the
                # Canvas publish its parse rejection on the next continuation.
                # No declaration is repaired into an AgentAction and no final
                # ADD parameter request is issued.
                metadata = dict(declaration_response.metadata)
                metadata.update(
                    {
                        "base_prompt_text": prompt,
                        "action_decoding_strategy": (
                            ROLE_FIRST_ADD_DECODING_STRATEGY
                        ),
                        "selected_action": selected_action,
                        "selected_modify_field": None,
                        "selected_modify_agent_id": None,
                        "selected_add_agent_ids": None,
                        "selected_add_agent_roles": [
                            dict(value) for value in selected_add_agent_roles
                        ],
                        "parameter_schema_branch": None,
                        "parse_failure_phase": (
                            _ADD_DECLARATION_PARSE_FAILURE_PHASE
                        ),
                        "hierarchical_phase_receipts": phase_receipts,
                        "request_count": len(phase_receipts),
                        "latency_ms": total_latency_ms,
                        "attempt_count": total_attempt_count,
                    }
                )
                return DirectorResponse(
                    text=declaration_response.text,
                    metadata=metadata,
                )
            selected_declarations_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in selected_add_agents],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            parameter_prompt = _hierarchical_continuation_prompt(
                declaration_prompt,
                committed_json=selected_declarations_json,
                instruction=_ADD_ACTION_CONTINUATION,
            )
            parameter_schema = director_live_action_parameter_json_schema_text(
                "add_subgraph",
                action_target_domains,
                add_agents=selected_add_agents,
            )
        elif (
            selected_action == "add_subgraph"
            and action_target_domains is not None
        ):
            # Role-neutral tasks use the same FlowSteer progressive
            # declaration -> complete-action boundary, but there is no
            # task-authored role inventory to sample first.  Agent count,
            # model, free-text contract, execution profile, relation and
            # Output Agent therefore remain Director decisions constrained by
            # the current Canvas domain.
            declaration_prompt = prompt
            declaration_schema = (
                director_live_add_subgraph_agent_declarations_json_schema_text(
                    action_target_domains,
                )
            )
            declaration_payload = dict(
                self._request_payload(declaration_prompt, adapter_name, seed)
            )
            declaration_sampling = dict(
                declaration_payload["sampling_params"]
            )
            declaration_sampling["json_schema"] = declaration_schema
            declaration_payload["sampling_params"] = declaration_sampling
            value, latency_ms, attempt_count = await self._post_with_retries(
                declaration_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            declaration_response = self._parse_response(
                declaration_prompt,
                declaration_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            phase_receipts["add_agent_declarations"] = (
                self._hierarchical_phase_receipt(declaration_response)
            )
            try:
                selected_add_agents = (
                    director_live_add_subgraph_agent_declarations_from_text(
                        _hierarchical_schema_text(
                            declaration_response.text,
                            reasoning_end_token=reasoning_end_token,
                        ),
                        action_target_domains,
                    )
                )
            except (ValueError, ReceiptValidationError):
                metadata = dict(declaration_response.metadata)
                metadata.update(
                    {
                        "base_prompt_text": prompt,
                        "action_decoding_strategy": (
                            HIERARCHICAL_JSON_SCHEMA_STRATEGY
                        ),
                        "selected_action": selected_action,
                        "selected_modify_field": None,
                        "selected_modify_agent_id": None,
                        "selected_add_agent_ids": None,
                        "selected_add_agent_roles": None,
                        "parameter_schema_branch": None,
                        "parse_failure_phase": (
                            _ADD_DECLARATION_PARSE_FAILURE_PHASE
                        ),
                        "hierarchical_phase_receipts": phase_receipts,
                        "request_count": len(phase_receipts),
                        "latency_ms": total_latency_ms,
                        "attempt_count": total_attempt_count,
                    }
                )
                return DirectorResponse(
                    text=declaration_response.text,
                    metadata=metadata,
                )
            selected_declarations_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in selected_add_agents],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            parameter_prompt = _hierarchical_continuation_prompt(
                declaration_prompt,
                committed_json=selected_declarations_json,
                instruction=_ADD_ACTION_CONTINUATION,
            )
            parameter_schema = director_live_action_parameter_json_schema_text(
                "add_subgraph",
                action_target_domains,
                add_agents=selected_add_agents,
            )
        elif selected_action == "modify_agent":
            admitted_modify_fields: Sequence[str] | None = None
            if action_target_domains is not None:
                modify_domain = action_target_domains.get("modify_agent")
                if not isinstance(modify_domain, Mapping):
                    raise ReceiptValidationError(
                        "v3 modify_agent action has no live target domain"
                    )
                raw_fields = modify_domain.get("mutable_fields")
                raw_candidates = modify_domain.get("per_agent_candidates")
                if not isinstance(raw_fields, (list, tuple)) or not isinstance(
                    raw_candidates, (list, tuple)
                ):
                    raise ReceiptValidationError(
                        "v3 modify_agent action has no live field candidates"
                    )
                candidate_fields = {
                    field
                    for candidate in raw_candidates
                    if isinstance(candidate, Mapping)
                    for field in candidate.get("mutable_fields", ())
                    if isinstance(field, str)
                }
                admitted_modify_fields = tuple(
                    field for field in raw_fields if field in candidate_fields
                )
            field_schema = director_modify_agent_field_selector_json_schema_text(
                admitted_modify_fields
            )
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
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            field_selector = json.loads(field_schema)
            admitted_fields = field_selector["properties"]["field"]["enum"]
            selected_modify_field = self._hierarchical_choice(
                field_response.text,
                field_name="field",
                admitted=admitted_fields,
                required_action="modify_agent",
                reasoning_end_token=reasoning_end_token,
            )
            phase_receipts["modify_field_selection"] = (
                self._hierarchical_phase_receipt(field_response)
            )
            if action_target_domains is None:
                parameter_schema = (
                    director_modify_agent_field_sampling_json_schema_text(
                        selected_modify_field
                    )
                )
            else:
                agent_schema = director_live_modify_agent_selector_json_schema_text(
                    action_target_domains,
                    selected_modify_field,
                )
                admitted_agent_ids = json.loads(agent_schema)["properties"][
                    "agent_id"
                ]["enum"]
                if len(admitted_agent_ids) == 1:
                    selected_modify_agent_id = admitted_agent_ids[0]
                else:
                    agent_payload = dict(
                        self._request_payload(prompt, adapter_name, seed)
                    )
                    agent_sampling = dict(agent_payload["sampling_params"])
                    agent_sampling["json_schema"] = agent_schema
                    agent_payload["sampling_params"] = agent_sampling
                    value, latency_ms, attempt_count = await self._post_with_retries(
                        agent_payload
                    )
                    total_latency_ms += latency_ms
                    total_attempt_count += attempt_count
                    agent_response = self._parse_response(
                        prompt,
                        agent_payload,
                        value,
                        policy_version=policy_version,
                        adapter_name=adapter_name,
                        expected_server_weight_version=(
                            expected_server_weight_version
                        ),
                        action_json_schema_version=action_schema_version,
                        action_schema_branch=action_schema_branch,
                        action_target_domains_json=action_target_domains_json,
                        action_target_domain_version=action_target_domain_version,
                        latency_ms=latency_ms,
                        attempt_count=attempt_count,
                        generation_seed=seed,
                    )
                    selected_modify_agent_id = self._hierarchical_choice(
                        agent_response.text,
                        field_name="agent_id",
                        admitted=admitted_agent_ids,
                        required_action="modify_agent",
                        reasoning_end_token=reasoning_end_token,
                    )
                    phase_receipts["modify_agent_selection"] = (
                        self._hierarchical_phase_receipt(agent_response)
                    )
                parameter_schema = director_live_action_parameter_json_schema_text(
                    "modify_agent",
                    action_target_domains,
                    modify_field=selected_modify_field,
                    modify_agent_id=selected_modify_agent_id,
                )
        elif selected_action == "set_relation" and action_target_domains is not None:
            candidate_schema = (
                director_live_relation_candidate_selector_json_schema_text(
                    action_target_domains
                )
            )
            candidate_payload = dict(self._request_payload(prompt, adapter_name, seed))
            candidate_sampling = dict(candidate_payload["sampling_params"])
            candidate_sampling["json_schema"] = candidate_schema
            candidate_payload["sampling_params"] = candidate_sampling
            value, latency_ms, attempt_count = await self._post_with_retries(
                candidate_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            candidate_response = self._parse_response(
                prompt,
                candidate_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=expected_server_weight_version,
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            candidate_selector = json.loads(candidate_schema)
            admitted_indices = candidate_selector["properties"]["candidate_index"][
                "enum"
            ]
            try:
                selected_relation_candidate = self._hierarchical_index_choice(
                    candidate_response.text,
                    admitted=admitted_indices,
                    required_action="set_relation",
                    reasoning_end_token=reasoning_end_token,
                )
            except ReceiptValidationError:
                if not _hierarchical_selector_serialization_failed(
                    candidate_response.text,
                    reasoning_end_token=reasoning_end_token,
                ):
                    raise
                # Match the existing bounded parameter regeneration boundary:
                # retain the exact invalid selector receipt, then make one
                # schema-bound continuation request with the same route and
                # scientific seed.  No candidate is inferred from malformed
                # text and there is no default index.
                relation_candidate_regeneration_attempted = True
                phase_receipts[
                    _RELATION_CANDIDATE_SERIALIZATION_FAILURE_PHASE
                ] = self._hierarchical_phase_receipt(candidate_response)
                regeneration_prompt = _hierarchical_continuation_prompt(
                    prompt,
                    committed_json=candidate_response.text,
                    instruction=_PARAMETER_REGENERATION_CONTINUATION,
                )
                regeneration_payload = dict(
                    self._request_payload(
                        regeneration_prompt,
                        adapter_name,
                        seed,
                    )
                )
                regeneration_sampling = dict(
                    regeneration_payload["sampling_params"]
                )
                regeneration_sampling["json_schema"] = candidate_schema
                regeneration_payload["sampling_params"] = regeneration_sampling
                value, latency_ms, attempt_count = await self._post_with_retries(
                    regeneration_payload
                )
                total_latency_ms += latency_ms
                total_attempt_count += attempt_count
                candidate_response = self._parse_response(
                    regeneration_prompt,
                    regeneration_payload,
                    value,
                    policy_version=policy_version,
                    adapter_name=adapter_name,
                    expected_server_weight_version=(
                        expected_server_weight_version
                    ),
                    action_json_schema_version=action_schema_version,
                    action_schema_branch=action_schema_branch,
                    action_target_domains_json=action_target_domains_json,
                    action_target_domain_version=action_target_domain_version,
                    latency_ms=latency_ms,
                    attempt_count=attempt_count,
                    generation_seed=seed,
                )
                selected_relation_candidate = self._hierarchical_index_choice(
                    candidate_response.text,
                    admitted=admitted_indices,
                    required_action="set_relation",
                    reasoning_end_token=reasoning_end_token,
                )
                relation_candidate_regeneration_succeeded = True
            phase_receipts["relation_candidate_selection"] = (
                self._hierarchical_phase_receipt(candidate_response)
            )
            parameter_schema = director_live_action_parameter_json_schema_text(
                "set_relation",
                action_target_domains,
                relation_candidate_index=selected_relation_candidate,
            )
        else:
            parameter_schema = (
                director_state_conditioned_sampling_json_schema_text(selected_action)
                if action_target_domains is None
                else director_live_action_parameter_json_schema_text(
                    selected_action,
                    action_target_domains,
                )
            )

        parameter_payload = dict(
            self._request_payload(parameter_prompt, adapter_name, seed)
        )
        parameter_sampling = dict(parameter_payload["sampling_params"])
        parameter_sampling["json_schema"] = parameter_schema
        parameter_payload["sampling_params"] = parameter_sampling
        value, latency_ms, attempt_count = await self._post_with_retries(
            parameter_payload
        )
        total_latency_ms += latency_ms
        total_attempt_count += attempt_count
        response = self._parse_response(
            parameter_prompt,
            parameter_payload,
            value,
            policy_version=policy_version,
            adapter_name=adapter_name,
            expected_server_weight_version=expected_server_weight_version,
            action_json_schema_version=action_schema_version,
            action_schema_branch=action_schema_branch,
            action_target_domains_json=action_target_domains_json,
            action_target_domain_version=action_target_domain_version,
            latency_ms=latency_ms,
            attempt_count=attempt_count,
            generation_seed=seed,
        )
        parameter_regeneration_attempted = False
        parameter_regeneration_succeeded = False
        if _action_parameter_serialization_failed(
            response.text,
            reasoning_end_token=reasoning_end_token,
        ):
            # SGLang may emit EOS before a schema-bound JSON object closes.
            # Preserve that exact failed sample as a phase receipt and make
            # one further request with the same schema, route, and seed.  The
            # continuation does not infer missing fields or alter the already
            # selected action/field/Agent semantics.
            parameter_regeneration_attempted = True
            phase_receipts[_PARAMETER_SERIALIZATION_FAILURE_PHASE] = (
                self._hierarchical_phase_receipt(response)
            )
            regeneration_prompt = _hierarchical_continuation_prompt(
                parameter_prompt,
                committed_json=response.text,
                instruction=_PARAMETER_REGENERATION_CONTINUATION,
            )
            regeneration_payload = dict(
                self._request_payload(regeneration_prompt, adapter_name, seed)
            )
            regeneration_sampling = dict(
                regeneration_payload["sampling_params"]
            )
            regeneration_sampling["json_schema"] = parameter_schema
            regeneration_payload["sampling_params"] = regeneration_sampling
            value, latency_ms, attempt_count = await self._post_with_retries(
                regeneration_payload
            )
            total_latency_ms += latency_ms
            total_attempt_count += attempt_count
            response = self._parse_response(
                regeneration_prompt,
                regeneration_payload,
                value,
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=(
                    expected_server_weight_version
                ),
                action_json_schema_version=action_schema_version,
                action_schema_branch=action_schema_branch,
                action_target_domains_json=action_target_domains_json,
                action_target_domain_version=action_target_domain_version,
                latency_ms=latency_ms,
                attempt_count=attempt_count,
                generation_seed=seed,
            )
            try:
                AgentActionParser().parse(
                    response.text,
                    reasoning_end_token=reasoning_end_token,
                )
            except AgentActionParseError:
                pass
            else:
                parameter_regeneration_succeeded = True
        metadata = dict(response.metadata)
        metadata.update(
            {
                "base_prompt_text": prompt,
                "action_decoding_strategy": (
                    ROLE_FIRST_ADD_DECODING_STRATEGY
                    if selected_add_agent_roles is not None
                    else HIERARCHICAL_JSON_SCHEMA_STRATEGY
                ),
                "selected_action": selected_action,
                "selected_modify_field": selected_modify_field,
                "parameter_schema_branch": (
                    selected_action
                    if selected_modify_field is None
                    and selected_relation_candidate is None
                    else (
                        f"modify_agent:{selected_modify_field}"
                        if selected_modify_field is not None
                        else f"set_relation:{selected_relation_candidate}"
                    )
                ),
                "hierarchical_phase_receipts": phase_receipts,
                "request_count": len(phase_receipts) + 1,
                "latency_ms": total_latency_ms,
                "attempt_count": total_attempt_count,
            }
        )
        if parameter_regeneration_attempted:
            metadata["parameter_regeneration_attempted"] = True
            metadata["parameter_regeneration_succeeded"] = (
                parameter_regeneration_succeeded
            )
        if action_selection_regeneration_attempted:
            metadata["action_selection_regeneration_attempted"] = True
            metadata["action_selection_regeneration_succeeded"] = (
                action_selection_regeneration_succeeded
            )
        if role_selection_regeneration_attempted:
            metadata["role_selection_regeneration_attempted"] = True
            metadata["role_selection_regeneration_succeeded"] = (
                role_selection_regeneration_succeeded
            )
        if relation_candidate_regeneration_attempted:
            metadata["relation_candidate_regeneration_attempted"] = True
            metadata["relation_candidate_regeneration_succeeded"] = (
                relation_candidate_regeneration_succeeded
            )
        if action_target_domains is not None:
            metadata["selected_add_agent_ids"] = (
                None
                if selected_add_agents is None
                else [agent["agent_id"] for agent in selected_add_agents]
            )
            metadata["selected_add_agent_roles"] = (
                None
                if selected_add_agent_roles is None
                else [dict(value) for value in selected_add_agent_roles]
            )
            metadata["selected_modify_agent_id"] = selected_modify_agent_id
        if selected_relation_candidate is not None:
            metadata["selected_relation_candidate"] = selected_relation_candidate
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
        action_target_domains_json: Optional[str],
        action_target_domain_version: Optional[str],
        latency_ms: float,
        attempt_count: int,
        generation_seed: Optional[int] = None,
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

        metadata = {
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
                "generation_seed": (
                    generation_seed
                    if generation_seed is not None
                    else payload.get("sampling_params", {}).get("sampling_seed")
                ),
                "backend_sampling_seed": payload.get(
                    "sampling_params", {}
                ).get("sampling_seed"),
                "chat_template_enable_thinking": self.enable_thinking,
                "thinking_budget_tokens": self.thinking_budget_tokens,
                "require_reasoning": payload.get("require_reasoning"),
                "action_json_schema_version": action_json_schema_version,
                "action_schema_branch": action_schema_branch,
                "receipt_verified": True,
            }
        if action_target_domain_version is not None:
            metadata["action_target_domains_json"] = action_target_domains_json
            metadata["action_target_domain_version"] = action_target_domain_version
        return DirectorResponse(text=text, metadata=metadata)

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


def _validate_v3_hierarchical_action_receipt(
    action: AgentAction | None,
    metadata: Mapping[str, Any],
    schema_request: Mapping[str, str],
) -> set[str]:
    """Validate the exact v3 phase/action/domain correspondence.

    A malformed final parameter, role selection, or role-first ADD declaration
    sample has no parsed ``AgentAction``.  It remains an exact behavior receipt
    and FlowSteer's Canvas returns it as an invalid-action observation for the
    next Director turn.  Hierarchical selections and completed phase receipts
    stay authoritative; sampled text is never repaired into an executed action.
    """

    reasoning_end_token = _director_reasoning_end_token(metadata)
    branch = schema_request.get("action_schema_branch")
    domains_json = schema_request.get("action_target_domains_json")
    if not isinstance(branch, str) or not isinstance(domains_json, str):
        raise ReceiptValidationError("v3 Director request has no branch/domain identity")
    try:
        actions = director_actions_from_admissible_schema_branch(branch)
        domains = json.loads(domains_json)
        director_live_action_target_domains_json(actions, domains)
    except (TypeError, ValueError) as exc:
        raise ReceiptValidationError("v3 Director target-domain receipt is invalid") from exc
    selected_action = metadata.get("selected_action")
    decoding_strategy = metadata.get("action_decoding_strategy")
    parse_failure_phase = metadata.get("parse_failure_phase")
    phase_receipts = metadata.get("hierarchical_phase_receipts")
    action_selection_regeneration_attempted = metadata.get(
        "action_selection_regeneration_attempted"
    )
    if (
        action_selection_regeneration_attempted is not None
        and action_selection_regeneration_attempted is not True
    ):
        raise ReceiptValidationError(
            "v3 action-selection regeneration attempt flag is invalid"
        )
    action_selection_failure_receipt = (
        phase_receipts.get(_ACTION_SELECTION_SERIALIZATION_FAILURE_PHASE)
        if isinstance(phase_receipts, Mapping)
        else None
    )
    parameter_regeneration_attempted = metadata.get(
        "parameter_regeneration_attempted"
    )
    if (
        parameter_regeneration_attempted is not None
        and parameter_regeneration_attempted is not True
    ):
        raise ReceiptValidationError(
            "v3 parameter-regeneration attempt flag is invalid"
        )
    parameter_failure_receipt = (
        phase_receipts.get(_PARAMETER_SERIALIZATION_FAILURE_PHASE)
        if isinstance(phase_receipts, Mapping)
        else None
    )
    role_selection_regeneration_attempted = metadata.get(
        "role_selection_regeneration_attempted"
    )
    if (
        role_selection_regeneration_attempted is not None
        and role_selection_regeneration_attempted is not True
    ):
        raise ReceiptValidationError(
            "v3 role-selection regeneration attempt flag is invalid"
        )
    role_selection_failure_receipt = (
        phase_receipts.get(_ADD_ROLE_SELECTION_SERIALIZATION_FAILURE_PHASE)
        if isinstance(phase_receipts, Mapping)
        else None
    )
    relation_candidate_regeneration_attempted = metadata.get(
        "relation_candidate_regeneration_attempted"
    )
    if (
        relation_candidate_regeneration_attempted is not None
        and relation_candidate_regeneration_attempted is not True
    ):
        raise ReceiptValidationError(
            "v3 relation-candidate regeneration attempt flag is invalid"
        )
    relation_candidate_failure_receipt = (
        phase_receipts.get(_RELATION_CANDIDATE_SERIALIZATION_FAILURE_PHASE)
        if isinstance(phase_receipts, Mapping)
        else None
    )
    if parse_failure_phase is not None and parameter_regeneration_attempted:
        raise ReceiptValidationError(
            "v3 declaration parse failure cannot carry parameter regeneration"
        )
    action_value = None if action is None else action.to_dict()
    if selected_action not in actions or (
        action_value is not None
        and action_value.get("action") != selected_action
    ):
        raise ReceiptValidationError(
            "v3 selected action differs from its branch or parsed Canvas action"
        )
    if (
        decoding_strategy == ROLE_FIRST_ADD_DECODING_STRATEGY
        and selected_action != "add_subgraph"
    ):
        raise ReceiptValidationError(
            "role-first ADD decoding strategy is attached to a non-ADD action"
        )
    if parse_failure_phase is not None and parse_failure_phase not in {
        _ADD_ROLE_SELECTION_PARSE_FAILURE_PHASE,
        _ADD_DECLARATION_PARSE_FAILURE_PHASE,
    }:
        raise ReceiptValidationError(
            "v3 hierarchical receipt has an unsupported parse-failure phase"
        )
    add_domain = domains.get("add_subgraph")
    role_neutral_add_domain = (
        isinstance(add_domain, Mapping)
        and add_domain.get("contract_semantics") == "free_text"
    )
    declaration_only_failure = (
        selected_action == "add_subgraph"
        and parse_failure_phase == _ADD_DECLARATION_PARSE_FAILURE_PHASE
        and decoding_strategy == HIERARCHICAL_JSON_SCHEMA_STRATEGY
        and role_neutral_add_domain
    )
    if parse_failure_phase is not None and not (
        (
            decoding_strategy == ROLE_FIRST_ADD_DECODING_STRATEGY
            and selected_action == "add_subgraph"
        )
        or declaration_only_failure
    ):
        raise ReceiptValidationError(
            "v3 ADD phase parse failure has no matching structured decoding phase"
        )
    if parse_failure_phase is not None and action is not None:
        raise ReceiptValidationError(
            "v3 ADD phase parse failure requires action is None"
        )

    expected_phases: set[str] = set()
    if len(actions) > 1:
        expected_phases.add("action_selection")
    if action_selection_regeneration_attempted:
        action_selection_regeneration_succeeded = metadata.get(
            "action_selection_regeneration_succeeded"
        )
        if action_selection_regeneration_succeeded is not True:
            raise ReceiptValidationError(
                "v3 returned action-selection regeneration did not succeed"
            )
        if len(actions) <= 1 or not isinstance(
            action_selection_failure_receipt, Mapping
        ):
            raise ReceiptValidationError(
                "v3 action-selection regeneration has no initial failure receipt"
            )
        failed_text = action_selection_failure_receipt.get("text")
        failed_prompt = action_selection_failure_receipt.get("prompt_text")
        if (
            not isinstance(failed_text, str)
            or not failed_text
            or not isinstance(failed_prompt, str)
            or not failed_prompt
            or not _hierarchical_selector_serialization_failed(
                failed_text,
                reasoning_end_token=reasoning_end_token,
            )
        ):
            raise ReceiptValidationError(
                "v3 action-selection initial receipt is not a serialization failure"
            )
        final_selector_receipt = (
            phase_receipts.get("action_selection")
            if isinstance(phase_receipts, Mapping)
            else None
        )
        expected_regeneration_prompt = _hierarchical_continuation_prompt(
            failed_prompt,
            committed_json=failed_text,
            instruction=_PARAMETER_REGENERATION_CONTINUATION,
        )
        if (
            not isinstance(final_selector_receipt, Mapping)
            or final_selector_receipt.get("prompt_text")
            != expected_regeneration_prompt
            or final_selector_receipt.get("generation_seed")
            != action_selection_failure_receipt.get("generation_seed")
        ):
            raise ReceiptValidationError(
                "v3 action-selection regeneration is not bound to its failed sample"
            )
        expected_phases.add(_ACTION_SELECTION_SERIALIZATION_FAILURE_PHASE)
    elif (
        metadata.get("action_selection_regeneration_succeeded") is not None
        or action_selection_failure_receipt is not None
    ):
        raise ReceiptValidationError(
            "v3 action-selection regeneration receipt has no attempt flag"
        )
    expected_parameter_branch = selected_action
    if selected_action != "set_relation" and (
        relation_candidate_regeneration_attempted is not None
        or metadata.get("relation_candidate_regeneration_succeeded") is not None
        or relation_candidate_failure_receipt is not None
    ):
        raise ReceiptValidationError(
            "v3 non-relation action carries relation-candidate regeneration"
        )
    if selected_action != "add_subgraph" and (
        role_selection_regeneration_attempted is not None
        or metadata.get("role_selection_regeneration_succeeded") is not None
        or role_selection_failure_receipt is not None
    ):
        raise ReceiptValidationError(
            "v3 non-ADD action carries role-selection regeneration"
        )

    if selected_action == "add_subgraph":
        role_first_add = decoding_strategy == ROLE_FIRST_ADD_DECODING_STRATEGY
        role_selection_parse_failure = (
            parse_failure_phase == _ADD_ROLE_SELECTION_PARSE_FAILURE_PHASE
        )
        if schema_request.get(
            "action_target_domain_version"
        ) == DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION:
            if role_neutral_add_domain and role_first_add:
                raise ReceiptValidationError(
                    "role-neutral live-domain ADD receipt used QA role-first decoding"
                )
            if not role_neutral_add_domain and not role_first_add:
                raise ReceiptValidationError(
                    "role-conditioned live-domain ADD receipt did not use "
                    "role-first decoding"
                )
        if role_first_add:
            expected_phases.add("add_agent_role_selection")
            if not role_selection_parse_failure:
                expected_phases.add("add_agent_declarations")
        elif decoding_strategy == HIERARCHICAL_JSON_SCHEMA_STRATEGY:
            expected_phases.add("add_agent_declarations")
        else:
            raise ReceiptValidationError(
                "v3 add_subgraph receipt has an unsupported decoding strategy"
            )
        role_phase = (
            phase_receipts.get("add_agent_role_selection")
            if isinstance(phase_receipts, Mapping)
            else None
        )
        declaration_phase = (
            phase_receipts.get("add_agent_declarations")
            if isinstance(phase_receipts, Mapping)
            else None
        )
        if role_first_add and (
            not isinstance(role_phase, Mapping)
            or not isinstance(role_phase.get("text"), str)
        ):
            raise ReceiptValidationError(
                "v3 add_subgraph receipt has no Agent role-selection phase"
            )
        if role_selection_parse_failure and declaration_phase is not None:
            raise ReceiptValidationError(
                "v3 role-selection parse failure fabricated an Agent declaration"
            )
        if not role_selection_parse_failure and (
            not isinstance(declaration_phase, Mapping)
            or not isinstance(declaration_phase.get("text"), str)
        ):
            raise ReceiptValidationError(
                "v3 add_subgraph receipt has no Agent declaration phase"
            )
        if role_selection_regeneration_attempted:
            expected_regeneration_succeeded = not role_selection_parse_failure
            observed_regeneration_succeeded = metadata.get(
                "role_selection_regeneration_succeeded"
            )
            if (
                type(observed_regeneration_succeeded) is not bool
                or observed_regeneration_succeeded
                is not expected_regeneration_succeeded
            ):
                raise ReceiptValidationError(
                    "v3 role-selection regeneration result differs from its "
                    "parse-failure phase"
                )
            if not isinstance(role_selection_failure_receipt, Mapping):
                raise ReceiptValidationError(
                    "v3 role-selection regeneration has no initial failure receipt"
                )
            failed_text = role_selection_failure_receipt.get("text")
            failed_prompt = role_selection_failure_receipt.get("prompt_text")
            if (
                not isinstance(failed_text, str)
                or not failed_text
                or not isinstance(failed_prompt, str)
                or not failed_prompt
                or not _hierarchical_selector_serialization_failed(
                    failed_text,
                    reasoning_end_token=reasoning_end_token,
                )
            ):
                raise ReceiptValidationError(
                    "v3 role-selection regeneration initial receipt is not a "
                    "serialization failure"
                )
            expected_regeneration_prompt = _hierarchical_continuation_prompt(
                failed_prompt,
                committed_json=failed_text,
                instruction=_PARAMETER_REGENERATION_CONTINUATION,
            )
            if role_phase.get("prompt_text") != expected_regeneration_prompt:
                raise ReceiptValidationError(
                    "v3 role-selection regeneration is not bound to its "
                    "failed sample"
                )
            if role_selection_failure_receipt.get(
                "generation_seed"
            ) != role_phase.get("generation_seed"):
                raise ReceiptValidationError(
                    "v3 role-selection regeneration changed its generation seed"
                )
            expected_phases.add(
                _ADD_ROLE_SELECTION_SERIALIZATION_FAILURE_PHASE
            )
        elif (
            metadata.get("role_selection_regeneration_succeeded") is not None
            or role_selection_failure_receipt is not None
        ):
            raise ReceiptValidationError(
                "v3 role-selection regeneration receipt has no attempt flag"
            )
        if role_selection_parse_failure:
            if not role_selection_regeneration_attempted:
                raise ReceiptValidationError(
                    "v3 role-selection parse failure has no bounded regeneration"
                )
            assert isinstance(role_phase, Mapping)
            try:
                director_live_add_subgraph_role_selection_from_text(
                    _hierarchical_schema_text(
                        role_phase["text"],
                        reasoning_end_token=reasoning_end_token,
                    ),
                    domains,
                )
            except ValueError:
                pass
            else:
                raise ReceiptValidationError(
                    "v3 role-selection parse-failure sample satisfies the "
                    "strict live domain"
                )
            if metadata.get("prompt_text") != role_phase.get("prompt_text"):
                raise ReceiptValidationError(
                    "v3 role-selection parse-failure receipt is not bound to "
                    "its regeneration prompt"
                )
            if metadata.get("selected_add_agent_roles") is not None:
                raise ReceiptValidationError(
                    "v3 role-selection parse failure fabricated selected roles"
                )
            if metadata.get("selected_add_agent_ids") is not None:
                raise ReceiptValidationError(
                    "v3 role-selection parse failure fabricated Agent declarations"
                )
            if metadata.get("selected_modify_agent_id") is not None:
                raise ReceiptValidationError(
                    "v3 role-selection parse failure carries a MODIFY target"
                )
            if metadata.get("parameter_schema_branch") is not None:
                raise ReceiptValidationError(
                    "v3 role-selection parse failure carries a parameter branch"
                )
            if metadata.get("request_count") != len(expected_phases):
                raise ReceiptValidationError(
                    "v3 role-selection parse-failure request count differs "
                    "from its completed phases"
                )
            return expected_phases
        if parse_failure_phase == _ADD_DECLARATION_PARSE_FAILURE_PHASE:
            if not role_first_add:
                if not role_neutral_add_domain:
                    raise ReceiptValidationError(
                        "role-neutral declaration failure has no free-text domain"
                    )
                declaration_prompt = declaration_phase.get("prompt_text")
                if (
                    not isinstance(declaration_prompt, str)
                    or metadata.get("prompt_text") != declaration_prompt
                ):
                    raise ReceiptValidationError(
                        "v3 role-neutral declaration parse-failure receipt is "
                        "not bound to its prompt"
                    )
                if metadata.get("selected_add_agent_roles") is not None:
                    raise ReceiptValidationError(
                        "v3 role-neutral declaration failure fabricated roles"
                    )
                if metadata.get("selected_add_agent_ids") is not None:
                    raise ReceiptValidationError(
                        "v3 declaration parse failure fabricated Agent declarations"
                    )
                if metadata.get("selected_modify_agent_id") is not None:
                    raise ReceiptValidationError(
                        "v3 declaration parse failure carries a MODIFY target"
                    )
                if metadata.get("parameter_schema_branch") is not None:
                    raise ReceiptValidationError(
                        "v3 declaration parse failure carries a parameter branch"
                    )
                if metadata.get("request_count") != len(expected_phases):
                    raise ReceiptValidationError(
                        "v3 declaration parse-failure request count differs "
                        "from its completed phases"
                    )
                return expected_phases
            assert role_first_add
            assert isinstance(role_phase, Mapping)
            try:
                selected_roles = (
                    director_live_add_subgraph_role_selection_from_text(
                        _hierarchical_schema_text(
                            role_phase["text"],
                            reasoning_end_token=reasoning_end_token,
                        ),
                        domains,
                    )
                )
            except ValueError as exc:
                raise ReceiptValidationError(
                    "v3 declaration parse failure has an invalid prior role receipt"
                ) from exc
            role_prompt = role_phase.get("prompt_text")
            declaration_prompt = declaration_phase.get("prompt_text")
            if not isinstance(role_prompt, str) or not isinstance(
                declaration_prompt,
                str,
            ):
                raise ReceiptValidationError(
                    "v3 declaration parse-failure phase has no prompt binding"
                )
            selected_roles_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in selected_roles],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_declaration_prompt = _hierarchical_continuation_prompt(
                role_prompt,
                committed_json=selected_roles_json,
                instruction=_ADD_DECLARATION_CONTINUATION,
            )
            if declaration_prompt != expected_declaration_prompt:
                raise ReceiptValidationError(
                    "v3 role-first ADD declaration prompt is not conditioned on "
                    "its selected roles"
                )
            if metadata.get("prompt_text") != declaration_prompt:
                raise ReceiptValidationError(
                    "v3 declaration parse-failure receipt is not bound to its "
                    "continuation prompt"
                )
            selected_role_values = [dict(value) for value in selected_roles]
            if metadata.get("selected_add_agent_roles") != selected_role_values:
                raise ReceiptValidationError(
                    "v3 declaration parse-failure role receipt changed"
                )
            if metadata.get("selected_add_agent_ids") is not None:
                raise ReceiptValidationError(
                    "v3 declaration parse failure fabricated Agent declarations"
                )
            if metadata.get("selected_modify_agent_id") is not None:
                raise ReceiptValidationError(
                    "v3 declaration parse failure carries a MODIFY target"
                )
            if metadata.get("parameter_schema_branch") is not None:
                raise ReceiptValidationError(
                    "v3 declaration parse failure carries a parameter branch"
                )
            if metadata.get("request_count") != len(expected_phases):
                raise ReceiptValidationError(
                    "v3 declaration parse-failure request count differs from "
                    "its completed phases"
                )
            return expected_phases
        try:
            if role_first_add:
                assert isinstance(role_phase, Mapping)
                selected_roles = (
                    director_live_add_subgraph_role_selection_from_text(
                        _hierarchical_schema_text(
                            role_phase["text"],
                            reasoning_end_token=reasoning_end_token,
                        ),
                        domains,
                    )
                )
            else:
                selected_roles = None
            declarations = director_live_add_subgraph_agent_declarations_from_text(
                _hierarchical_schema_text(
                    declaration_phase["text"],
                    reasoning_end_token=reasoning_end_token,
                ),
                domains,
                selected_agent_roles=selected_roles,
            )
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=declarations,
            )
        except ValueError as exc:
            raise ReceiptValidationError(
                "v3 add_subgraph declaration receipt violates its live domain"
            ) from exc
        if role_first_add:
            assert selected_roles is not None
            assert isinstance(role_phase, Mapping)
            role_prompt = role_phase.get("prompt_text")
            declaration_prompt = declaration_phase.get("prompt_text")
            if not isinstance(role_prompt, str) or not isinstance(
                declaration_prompt, str
            ):
                raise ReceiptValidationError(
                    "v3 role-first ADD phase has no prompt binding"
                )
            selected_roles_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in selected_roles],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_declaration_prompt = _hierarchical_continuation_prompt(
                role_prompt,
                committed_json=selected_roles_json,
                instruction=_ADD_DECLARATION_CONTINUATION,
            )
            if declaration_prompt != expected_declaration_prompt:
                raise ReceiptValidationError(
                    "v3 role-first ADD declaration prompt is not conditioned on "
                    "its selected roles"
                )
            selected_declarations_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in declarations],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_parameter_prompt = _hierarchical_continuation_prompt(
                declaration_prompt,
                committed_json=selected_declarations_json,
                instruction=_ADD_ACTION_CONTINUATION,
            )
            observed_parameter_prompt = (
                parameter_failure_receipt.get("prompt_text")
                if parameter_regeneration_attempted
                and isinstance(parameter_failure_receipt, Mapping)
                else metadata.get("prompt_text")
            )
            if observed_parameter_prompt != expected_parameter_prompt:
                raise ReceiptValidationError(
                    "v3 role-first ADD parameter prompt is not conditioned on "
                    "its Agent declarations"
                )
        elif role_neutral_add_domain:
            declaration_prompt = declaration_phase.get("prompt_text")
            if not isinstance(declaration_prompt, str):
                raise ReceiptValidationError(
                    "v3 role-neutral ADD declaration has no prompt binding"
                )
            selected_declarations_json = json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [dict(value) for value in declarations],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_parameter_prompt = _hierarchical_continuation_prompt(
                declaration_prompt,
                committed_json=selected_declarations_json,
                instruction=_ADD_ACTION_CONTINUATION,
            )
            observed_parameter_prompt = (
                parameter_failure_receipt.get("prompt_text")
                if parameter_regeneration_attempted
                and isinstance(parameter_failure_receipt, Mapping)
                else metadata.get("prompt_text")
            )
            if observed_parameter_prompt != expected_parameter_prompt:
                raise ReceiptValidationError(
                    "v3 role-neutral ADD parameter prompt is not conditioned "
                    "on its Agent declarations"
                )
        declaration_values = list(declarations)
        if (
            action_value is not None
            and action_value.get("agents") != declaration_values
        ):
            raise ReceiptValidationError(
                "v3 final add_subgraph changed its sampled Agent declarations"
            )
        declared_ids = [agent["agent_id"] for agent in declarations]
        if metadata.get("selected_add_agent_ids") != declared_ids:
            raise ReceiptValidationError(
                "v3 add_subgraph Agent-ID receipt differs from its declarations"
            )
        if role_first_add:
            assert selected_roles is not None
            selected_role_values = [dict(value) for value in selected_roles]
            if metadata.get("selected_add_agent_roles") != selected_role_values:
                raise ReceiptValidationError(
                    "v3 add_subgraph Agent-role receipt differs from its declarations"
                )
        elif metadata.get("selected_add_agent_roles") is not None:
            raise ReceiptValidationError(
                "legacy v3 add_subgraph receipt unexpectedly carries selected roles"
            )
        endpoint_ids = set(domains["add_subgraph"]["existing_agent_ids"])
        endpoint_ids.update(declared_ids)
        for relation in (
            () if action_value is None else action_value.get("relations", ())
        ):  # parser-normalized values
            if (
                not isinstance(relation, Mapping)
                or relation.get("source_id") not in endpoint_ids
                or relation.get("target_id") not in endpoint_ids
            ):
                raise ReceiptValidationError(
                    "v3 add_subgraph relation endpoint is outside the live domain"
                )
        add_domain = domains["add_subgraph"]
        if (
            verified_qa_semantic_protocol(add_domain.get("semantic_protocol"))
            or add_domain.get("contract_semantics") == "free_text"
        ):
            required_relation_count = add_domain.get(
                "required_relation_count"
            )
            required_ingress = add_domain.get(
                "required_existing_ingress_agent_ids"
            )
            exact_free_text_fan_in = bool(
                not verified_qa_semantic_protocol(
                    add_domain.get("semantic_protocol")
                )
                and add_domain.get("contract_semantics") == "free_text"
                and add_domain.get("require_all_existing_ingress") is True
                and type(required_relation_count) is int
                and required_relation_count > 1
            )
            if action_value is not None and exact_free_text_fan_in:
                relations = tuple(action_value.get("relations", ()))
                if (
                    not isinstance(required_ingress, (list, tuple))
                    or any(
                        not isinstance(agent_id, str) or not agent_id
                        for agent_id in required_ingress
                    )
                    or len(set(required_ingress)) != len(required_ingress)
                    or len(required_ingress) != required_relation_count
                    or len(declared_ids) != 1
                ):
                    raise ReceiptValidationError(
                        "v3 exact fan-in live domain is internally inconsistent"
                    )
                if len(relations) != required_relation_count:
                    raise ReceiptValidationError(
                        "v3 exact fan-in relation count differs from its live domain"
                    )
                consumer_id = declared_ids[0]
                observed_sources: set[str] = set()
                for relation in relations:
                    if not (
                        isinstance(relation, Mapping)
                        and relation.get("target_id") == consumer_id
                        and relation.get("source_to_target") is True
                        and relation.get("target_to_source") is False
                        and relation.get("source_id") in required_ingress
                    ):
                        raise ReceiptValidationError(
                            "v3 exact fan-in relation violates its directed live domain"
                        )
                    observed_sources.add(str(relation["source_id"]))
                if observed_sources != set(required_ingress):
                    raise ReceiptValidationError(
                        "v3 exact fan-in sources differ from required ingress"
                    )
            elif (
                action_value is not None
                and len(action_value.get("relations", ())) > 1
            ):
                raise ReceiptValidationError(
                    "v3 add_subgraph exceeds its one-relation edit boundary"
                )
            allowed_relations = {
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for candidate in director_live_add_subgraph_relation_candidates(
                    domains,
                    declarations,
                )
            }
            relation_pairs: set[frozenset[str]] = set()
            for relation in (
                () if action_value is None else action_value.get("relations", ())
            ):
                relation_identity = json.dumps(
                    dict(relation),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if relation_identity not in allowed_relations:
                    raise ReceiptValidationError(
                        "v3 add_subgraph relation violates the live domain"
                    )
                relation_pair = frozenset(
                    (relation["source_id"], relation["target_id"])
                )
                if relation_pair in relation_pairs:
                    raise ReceiptValidationError(
                        "v3 add_subgraph repeats an unordered relation pair"
                    )
                relation_pairs.add(relation_pair)
        output_agent_id = (
            None if action_value is None else action_value.get("output_agent_id")
        )
        if output_agent_id is not None and output_agent_id not in endpoint_ids:
            raise ReceiptValidationError(
                "v3 add_subgraph Output Agent is outside the live domain"
            )
        if (
            output_agent_id is not None
            and verified_qa_semantic_protocol(
                add_domain.get("semantic_protocol")
            )
        ):
            existing_roles = {
                item["agent_id"]: item["role_family"]
                for item in add_domain.get("existing_agents", ())
                if isinstance(item, Mapping)
                and isinstance(item.get("agent_id"), str)
                and isinstance(item.get("role_family"), str)
            }
            existing_roles.update(
                {agent["agent_id"]: agent["role_family"] for agent in declarations}
            )
            if existing_roles.get(output_agent_id) != "format":
                raise ReceiptValidationError(
                    "v3 verified-QA add_subgraph Output Agent is not a Formatter"
                )
            current_output_agent_id = add_domain.get("current_output_agent_id")
            if current_output_agent_id is not None:
                raise ReceiptValidationError(
                    "v3 verified-QA add_subgraph cannot replace the current Output Agent"
                )
        if metadata.get("selected_modify_agent_id") is not None:
            raise ReceiptValidationError(
                "v3 add_subgraph receipt carries a MODIFY target"
            )
    elif selected_action == "modify_agent":
        expected_phases.add("modify_field_selection")
        selected_field = metadata.get("selected_modify_field")
        selected_agent_id = metadata.get("selected_modify_agent_id")
        if not isinstance(selected_field, str) or not isinstance(
            selected_agent_id, str
        ):
            raise ReceiptValidationError(
                "v3 MODIFY field/Agent receipt is incomplete"
            )
        expected_action_fields = (
            {"action", "agent_id", "execution_mode", "allowed_tools"}
            if selected_field == "execution_profile"
            else {"action", "agent_id", selected_field}
        )
        if action_value is not None and (
            set(action_value) != expected_action_fields
            or action_value.get("agent_id") != selected_agent_id
        ):
            raise ReceiptValidationError(
                "v3 MODIFY field/Agent receipt differs from the parsed atomic patch"
            )
        try:
            agent_selector = json.loads(
                director_live_modify_agent_selector_json_schema_text(
                    domains,
                    selected_field,
                )
            )
            admitted_agent_ids = agent_selector["properties"]["agent_id"]["enum"]
            parameter_schema = json.loads(
                director_live_action_parameter_json_schema_text(
                    "modify_agent",
                    domains,
                    modify_field=selected_field,
                    modify_agent_id=selected_agent_id,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReceiptValidationError(
                "v3 MODIFY receipt violates its live parameter domain"
            ) from exc
        if selected_agent_id not in admitted_agent_ids:
            raise ReceiptValidationError("v3 MODIFY selected an inadmissible Agent")
        if len(admitted_agent_ids) > 1:
            expected_phases.add("modify_agent_selection")
        if selected_field == "execution_profile":
            modify_domain = domains.get("modify_agent")
            candidates = (
                modify_domain.get("per_agent_candidates", ())
                if isinstance(modify_domain, Mapping)
                else ()
            )
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and candidate.get("agent_id") == selected_agent_id
                ),
                None,
            )
            raw_profiles = (
                selected_candidate.get("discrete_value_domains", {}).get(
                    "execution_profile", ()
                )
                if isinstance(selected_candidate, Mapping)
                and isinstance(
                    selected_candidate.get("discrete_value_domains"), Mapping
                )
                else ()
            )
            observed_profile = (
                None
                if action_value is None
                else {
                    "execution_mode": action_value.get("execution_mode"),
                    "allowed_tools": action_value.get("allowed_tools"),
                }
            )
            if action_value is not None and observed_profile not in raw_profiles:
                raise ReceiptValidationError(
                    "v3 MODIFY execution profile is outside its exact live domain"
                )
            schema_branches = parameter_schema.get(
                "oneOf", (parameter_schema,)
            )
            if action_value is not None and not any(
                isinstance(branch, Mapping)
                and branch.get("properties", {})
                .get("execution_mode", {})
                .get("const")
                == action_value.get("execution_mode")
                and branch.get("properties", {})
                .get("allowed_tools", {})
                .get("const")
                == action_value.get("allowed_tools")
                for branch in schema_branches
            ):
                raise ReceiptValidationError(
                    "v3 MODIFY execution profile differs from its constrained schema"
                )
        else:
            value_schema = parameter_schema["properties"][selected_field]
            if (
                action_value is not None
                and "enum" in value_schema
                and action_value[selected_field] not in value_schema["enum"]
            ):
                raise ReceiptValidationError(
                    "v3 MODIFY value is outside its discrete live domain"
                )
        expected_parameter_branch = f"modify_agent:{selected_field}"
        if metadata.get("selected_add_agent_ids") is not None:
            raise ReceiptValidationError("v3 MODIFY receipt carries ADD declarations")
        if metadata.get("selected_add_agent_roles") is not None:
            raise ReceiptValidationError("v3 MODIFY receipt carries ADD roles")
    elif selected_action == "set_relation":
        expected_phases.add("relation_candidate_selection")
        selected_index = metadata.get("selected_relation_candidate")
        candidates = domains.get("set_relation", {}).get("candidates")
        if (
            type(selected_index) is not int
            or not isinstance(candidates, list)
            or not 0 <= selected_index < len(candidates)
        ):
            raise ReceiptValidationError(
                "v3 relation candidate receipt is outside the live domain"
            )
        relation_candidate_receipt = (
            phase_receipts.get("relation_candidate_selection")
            if isinstance(phase_receipts, Mapping)
            else None
        )
        if not isinstance(relation_candidate_receipt, Mapping) or not isinstance(
            relation_candidate_receipt.get("text"), str
        ):
            raise ReceiptValidationError(
                "v3 relation candidate selection has no exact phase receipt"
            )
        sampled_index = SGLangReceiptDirectorClient._hierarchical_index_choice(
            relation_candidate_receipt["text"],
            admitted=tuple(range(len(candidates))),
            required_action="set_relation",
            reasoning_end_token=reasoning_end_token,
        )
        if sampled_index != selected_index:
            raise ReceiptValidationError(
                "v3 relation candidate selection differs from its phase receipt"
            )
        if relation_candidate_regeneration_attempted:
            if metadata.get("relation_candidate_regeneration_succeeded") is not True:
                raise ReceiptValidationError(
                    "v3 relation-candidate regeneration did not succeed"
                )
            if not isinstance(relation_candidate_failure_receipt, Mapping):
                raise ReceiptValidationError(
                    "v3 relation-candidate regeneration has no initial failure receipt"
                )
            failed_text = relation_candidate_failure_receipt.get("text")
            failed_prompt = relation_candidate_failure_receipt.get("prompt_text")
            if (
                not isinstance(failed_text, str)
                or not failed_text
                or not isinstance(failed_prompt, str)
                or not failed_prompt
                or not _hierarchical_selector_serialization_failed(
                    failed_text,
                    reasoning_end_token=reasoning_end_token,
                )
            ):
                raise ReceiptValidationError(
                    "v3 relation-candidate regeneration initial receipt is not "
                    "a serialization failure"
                )
            expected_regeneration_prompt = _hierarchical_continuation_prompt(
                failed_prompt,
                committed_json=failed_text,
                instruction=_PARAMETER_REGENERATION_CONTINUATION,
            )
            if relation_candidate_receipt.get("prompt_text") != (
                expected_regeneration_prompt
            ):
                raise ReceiptValidationError(
                    "v3 relation-candidate regeneration is not bound to its "
                    "failed sample"
                )
            if relation_candidate_failure_receipt.get(
                "generation_seed"
            ) != relation_candidate_receipt.get("generation_seed"):
                raise ReceiptValidationError(
                    "v3 relation-candidate regeneration changed its generation seed"
                )
            expected_phases.add(
                _RELATION_CANDIDATE_SERIALIZATION_FAILURE_PHASE
            )
        elif (
            metadata.get("relation_candidate_regeneration_succeeded") is not None
            or relation_candidate_failure_receipt is not None
        ):
            raise ReceiptValidationError(
                "v3 relation-candidate regeneration receipt has no attempt flag"
            )
        expected_action = {"action": "set_relation", **candidates[selected_index]}
        if action_value is not None and action_value != expected_action:
            raise ReceiptValidationError(
                "v3 final relation differs from its selected exact candidate"
            )
        expected_parameter_branch = f"set_relation:{selected_index}"
    else:
        try:
            parameter_schema = json.loads(
                director_live_action_parameter_json_schema_text(
                selected_action,
                domains,
            )
            )
        except ValueError as exc:
            raise ReceiptValidationError(
                "v3 final action violates its live target domain"
            ) from exc
        if selected_action == "add_agent" and action_value is not None:
            add_domain = domains.get("add_agent")
            if not isinstance(add_domain, Mapping):
                raise ReceiptValidationError(
                    "v3 ADD_AGENT receipt has no live target domain"
                )
            registered_profiles = {
                (
                    item.get("execution_mode"),
                    tuple(item.get("allowed_tools", ())),
                )
                for item in add_domain.get(
                    "registered_execution_profiles", ()
                )
                if isinstance(item, Mapping)
                and isinstance(item.get("allowed_tools"), (list, tuple))
            }
            required_fields = {
                "action",
                "agent_id",
                "model_id",
                "contract",
                "execution_mode",
                "allowed_tools",
            }
            observed_profile = (
                action_value.get("execution_mode"),
                tuple(action_value.get("allowed_tools", ())),
            )
            if (
                set(action_value) != required_fields
                or action_value.get("agent_id")
                not in add_domain.get("agent_ids", ())
                or action_value.get("model_id")
                not in add_domain.get("model_ids", ())
                or observed_profile not in registered_profiles
            ):
                raise ReceiptValidationError(
                    "v3 final ADD_AGENT parameters are outside the exact live domain"
                )
            if metadata.get("selected_add_agent_ids") is not None:
                raise ReceiptValidationError(
                    "v3 scalar ADD_AGENT receipt carries subgraph declarations"
                )
            if metadata.get("selected_add_agent_roles") is not None:
                raise ReceiptValidationError(
                    "v3 scalar ADD_AGENT receipt carries role-first declarations"
                )
            # The exact schema is deliberately role neutral; this assertion
            # binds receipt validation to the same request-local ID/model
            # constraints used for constrained decoding.
            schema_branches = parameter_schema.get(
                "oneOf", (parameter_schema,)
            )

            def branch_admits_exact_value(
                branch: Mapping[str, Any],
                field_name: str,
                value: object,
            ) -> bool:
                field_schema = (
                    branch.get("properties", {}).get(field_name, {})
                )
                if not isinstance(field_schema, Mapping):
                    return False
                if "const" in field_schema:
                    return field_schema.get("const") == value
                enum_values = field_schema.get("enum", ())
                return (
                    isinstance(enum_values, (list, tuple))
                    and value in enum_values
                )

            if not any(
                branch_admits_exact_value(
                    branch,
                    "agent_id",
                    action_value.get("agent_id"),
                )
                and branch_admits_exact_value(
                    branch,
                    "model_id",
                    action_value.get("model_id"),
                )
                for branch in schema_branches
                if isinstance(branch, Mapping)
            ):
                raise ReceiptValidationError(
                    "v3 scalar ADD_AGENT receipt differs from its parameter schema"
                )
        if (
            action_value is not None
            and selected_action in {"delete_agent", "set_output"}
        ):
            admitted_ids = domains[selected_action]["agent_ids"]
            if action_value.get("agent_id") not in admitted_ids:
                raise ReceiptValidationError(
                    "v3 final Agent target is outside its live domain"
                )

    if metadata.get("parameter_schema_branch") != expected_parameter_branch:
        raise ReceiptValidationError(
            "v3 parameter-schema branch differs from the sampled action"
        )
    if parameter_regeneration_attempted:
        parameter_regeneration_succeeded = metadata.get(
            "parameter_regeneration_succeeded"
        )
        if type(parameter_regeneration_succeeded) is not bool:
            raise ReceiptValidationError(
                "v3 parameter-regeneration result flag is invalid"
            )
        if not isinstance(parameter_failure_receipt, Mapping):
            raise ReceiptValidationError(
                "v3 parameter regeneration has no initial failure receipt"
            )
        failed_text = parameter_failure_receipt.get("text")
        failed_prompt = parameter_failure_receipt.get("prompt_text")
        if (
            not isinstance(failed_text, str)
            or not failed_text
            or not isinstance(failed_prompt, str)
            or not failed_prompt
            or not _action_parameter_serialization_failed(
                failed_text,
                reasoning_end_token=reasoning_end_token,
            )
        ):
            raise ReceiptValidationError(
                "v3 parameter-regeneration initial receipt is not a "
                "serialization failure"
            )
        if parameter_regeneration_succeeded != (action is not None):
            raise ReceiptValidationError(
                "v3 parameter-regeneration result differs from the parsed action"
            )
        expected_regeneration_prompt = _hierarchical_continuation_prompt(
            failed_prompt,
            committed_json=failed_text,
            instruction=_PARAMETER_REGENERATION_CONTINUATION,
        )
        if metadata.get("prompt_text") != expected_regeneration_prompt:
            raise ReceiptValidationError(
                "v3 parameter regeneration is not bound to its failed sample"
            )
        if parameter_failure_receipt.get("generation_seed") != metadata.get(
            "generation_seed"
        ):
            raise ReceiptValidationError(
                "v3 parameter regeneration changed its generation seed"
            )
        expected_phases.add(_PARAMETER_SERIALIZATION_FAILURE_PHASE)
    elif (
        metadata.get("parameter_regeneration_succeeded") is not None
        or parameter_failure_receipt is not None
    ):
        raise ReceiptValidationError(
            "v3 parameter regeneration receipt has no attempt flag"
        )
    if metadata.get("request_count") != len(expected_phases) + 1:
        raise ReceiptValidationError(
            "v3 hierarchical request count differs from its required phases"
        )
    return expected_phases


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
    "chat_template_enable_thinking",
    "reasoning_content_present",
    "inline_reasoning_present",
    "reasoning_trace_present",
    "reasoning_trace_required",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "temperature",
    "top_p",
    "max_tokens",
)

_UNIFIED_EXECUTION_METADATA_FIELDS: Tuple[str, ...] = (
    "artifact_version",
    "artifact_id",
    "agent_id",
    "graph_revision",
    "model",
    "model_name",
    "provider_id",
    "contract",
    "tool_config",
    "raw_output",
    "upstream_dependencies",
    "input_artifact_versions",
    "input_artifact_provenance",
    "retry_receipts",
    "execution_mode",
    "react_turns_used",
    "new_react_turns_used",
    "continued_action_history_count",
    "continued_tool_receipt_count",
    "continuation_source_agent_id",
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
        "artifact_assessment_protocol": (
            request.artifact_assessment_protocol
        ),
        "continuation_source_agent_id": request.continuation_source_agent_id,
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
    artifact_inputs = list(request.upstream)
    if request.peer_draft is not None:
        artifact_inputs.append(request.peer_draft)
    response_receipt.update(
        {
            # AgentRuntime uses the exact request identity as the immutable
            # artifact identity.  Deriving these fields from this call avoids
            # incorrectly assigning a block's final artifact to its earlier
            # DRAFT/REVISION calls.
            "artifact_version": request.request_id,
            "artifact_id": request.request_id,
            "raw_output": response.text,
            "upstream_dependencies": [
                {
                    "source_agent": item.source_agent_id,
                    "artifact_id": item.artifact_id,
                    "raw_output": item.content,
                }
                for item in artifact_inputs
            ],
        }
    )
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
        "deferred_agent_ids": list(runtime.deferred_agent_ids),
        "agent_statuses": dict(runtime.agent_statuses),
        "output_metadata": output_metadata,
    }


def _lineage_field(lineage: object, field_name: str) -> object:
    """Read one public field from the Env's immutable lineage getter."""

    if isinstance(lineage, Mapping):
        return lineage.get(field_name)
    return getattr(lineage, field_name, None)


def _lineage_graph(lineage: object) -> Optional[Mapping[str, Any]]:
    """Normalize the graph attached to a valid-lineage Env receipt."""

    raw_graph = _lineage_field(lineage, "graph")
    if raw_graph is None:
        raw_graph = _lineage_field(lineage, "graph_snapshot")
    if raw_graph is None:
        snapshot = _lineage_field(lineage, "snapshot")
        if snapshot is not None:
            raw_graph = (
                snapshot.get("graph")
                if isinstance(snapshot, Mapping)
                else getattr(snapshot, "graph", snapshot)
            )
    if raw_graph is None:
        return None
    if not isinstance(raw_graph, Mapping):
        to_dict = getattr(raw_graph, "to_dict", None)
        if not callable(to_dict):
            return None
        raw_graph = to_dict()
    if not isinstance(raw_graph, Mapping):
        return None
    graph = dict(raw_graph)
    nested = graph.get("graph")
    if isinstance(nested, Mapping):
        graph = dict(nested)
    return graph


def _last_valid_evidence_lineage_fallback(
    environment: AgentWorkflowEnv,
) -> Optional[
    tuple[str, AgentRuntimeResult, Mapping[str, Any], Mapping[str, Any]]
]:
    """Return a structurally complete Env-validated lineage, if one exists.

    The environment owns semantic-lineage admission.  This collector only
    checks that its read-only receipt can be passed to the evaluator without
    mixing an answer, Runtime result, or graph from different revisions.
    Supporting both a property and a zero-argument method keeps this boundary
    compatible with the Env implementation while it is introduced.
    """

    lineage = getattr(environment, "last_valid_evidence_lineage", None)
    if callable(lineage):
        lineage = lineage()
    if lineage is None:
        return None

    final_answer = _lineage_field(lineage, "answer")
    if final_answer is None:
        # Mapping-based compatibility adapters may retain ``final_answer``.
        final_answer = _lineage_field(lineage, "final_answer")
    runtime = _lineage_field(lineage, "runtime")
    graph_revision = _lineage_field(lineage, "graph_revision")
    graph = _lineage_graph(lineage)
    if (
        not isinstance(final_answer, str)
        or not final_answer.strip()
        or not isinstance(runtime, AgentRuntimeResult)
        or isinstance(graph_revision, bool)
        or not isinstance(graph_revision, int)
        or graph_revision < 0
        or graph is None
    ):
        return None
    if (
        runtime.graph_revision != graph_revision
        or runtime.final_answer != final_answer
    ):
        return None
    snapshot_revision = graph.get("revision")
    if snapshot_revision is not None and snapshot_revision != graph_revision:
        return None

    receipt: dict[str, Any] = {
        "source": "AgentWorkflowEnv.last_valid_evidence_lineage",
        "final_answer": final_answer,
        "graph_revision": graph_revision,
        "graph_snapshot": dict(graph),
        "runtime_run_id": runtime.run_id,
        "runtime_graph_revision": runtime.graph_revision,
        "runtime_output_agent_id": runtime.output_agent_id,
    }
    return final_answer, runtime, graph, receipt


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
        remains an explicit terminal-failure trajectory.  Only an immutable
        Env receipt for the last complete evidence lineage may supply the
        evaluator answer, Runtime result, and matching graph at that boundary;
        otherwise the answer remains empty.  Such fallback trajectories are
        never admitted to GRPO.
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
        resolved_problem = workflow_problem or task.question
        env.reset(resolved_problem)
        turns: list[TurnRecord] = []
        snapshots: list[GraphSnapshotEvent] = []
        previous_snapshot_id: Optional[str] = None
        final_answer: Optional[str] = None
        final_runtime: Optional[AgentRuntimeResult] = None
        final_graph: Optional[Mapping[str, Any]] = None
        valid_lineage_fallback_used = False
        valid_lineage_fallback_receipt: Mapping[str, Any] = {}
        explicit_finish = False
        natural_terminal_reason: Optional[str] = None

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
        start_round_index = 0
        resume_checkpoint_id: Optional[str] = None
        checkpoint = (
            None
            if self.evidence_store is None
            else self.evidence_store.resolve_rollout_checkpoint(
                trajectory_id
            )
        )
        if checkpoint is None:
            prompt = self.orchestrator.build_prompt(env, 0, current_skills)
        else:
            expected_identity = {
                "task_id": task.task_id,
                "group_id": group_id,
                "rollout_id": rollout_id,
                "condition_id": self.condition_id,
                "versions": self.versions.to_dict(),
                "director_sampling": dict(
                    self.orchestrator.sampling_receipt
                ),
                "workflow_problem": resolved_problem,
                "active_skill_ids": list(active_skill_ids),
            }
            for field_name, expected_value in expected_identity.items():
                if checkpoint.get(field_name) != expected_value:
                    raise ReceiptValidationError(
                        "rollout checkpoint identity mismatch: "
                        + field_name
                    )
            start_round_index = int(checkpoint["next_round_index"])
            if not 0 < start_round_index <= self.orchestrator.max_rounds:
                raise ReceiptValidationError(
                    "rollout checkpoint next round is outside the horizon"
                )
            checkpoint_events = [
                item
                for item in self.evidence_store.rollout_checkpoint_events(
                    trajectory_id
                )
                if item.get("status")
                in {"in_progress", "terminal_pending_evaluation"}
                and isinstance(item.get("next_round_index"), int)
                and int(item["next_round_index"]) <= start_round_index
            ]
            expected_rounds = list(range(1, start_round_index + 1))
            observed_rounds = [
                int(item["next_round_index"])
                for item in checkpoint_events
            ]
            if observed_rounds != expected_rounds:
                raise ReceiptValidationError(
                    "rollout checkpoint turn sequence is not contiguous"
                )
            for item in checkpoint_events:
                raw_turn = item.get("turn")
                raw_snapshot = item.get("graph_snapshot_event")
                if not isinstance(raw_turn, Mapping) or not isinstance(
                    raw_snapshot, Mapping
                ):
                    raise ReceiptValidationError(
                        "rollout checkpoint is missing a completed turn"
                    )
                raw_graph = raw_snapshot.get("graph")
                if not isinstance(raw_graph, Mapping):
                    raise ReceiptValidationError(
                        "rollout checkpoint graph snapshot is malformed"
                    )
                turn = TurnRecord.from_dict(raw_turn)
                snapshot_event = GraphSnapshotEvent(
                    revision=int(raw_snapshot["revision"]),
                    graph=dict(raw_graph),
                    snapshot_id=str(raw_snapshot["snapshot_id"]),
                    previous_snapshot_id=raw_snapshot.get(
                        "previous_snapshot_id"
                    ),
                )
                if (
                    turn.round_index + 1 != item["next_round_index"]
                    or turn.graph_snapshot_id != snapshot_event.snapshot_id
                    or turn.previous_graph_snapshot_id
                    != snapshot_event.previous_snapshot_id
                ):
                    raise ReceiptValidationError(
                        "rollout checkpoint turn/snapshot binding mismatch"
                    )
                turns.append(turn)
                snapshots.append(snapshot_event)
            previous_snapshot_id = snapshots[-1].snapshot_id
            raw_environment = checkpoint.get("environment")
            if not isinstance(raw_environment, Mapping):
                raise ReceiptValidationError(
                    "rollout checkpoint has no Canvas Runtime state"
                )
            env.restore_runtime_checkpoint(raw_environment)
            resume_checkpoint_id = str(checkpoint["checkpoint_id"])
            if env.finished:
                final_runtime = env.current_progressive_execution
                if final_runtime is None or final_runtime.final_answer is None:
                    raise ReceiptValidationError(
                        "terminal checkpoint has no fresh Output artifact"
                    )
                explicit_finish = True
                final_answer = final_runtime.final_answer
                final_graph = env.graph.to_dict()
                prompt = ""
            else:
                next_prompt = checkpoint.get("next_prompt")
                if not isinstance(next_prompt, str) or not next_prompt:
                    raise ReceiptValidationError(
                        "resumable checkpoint has no next Director prompt"
                    )
                prompt = next_prompt
                resumed_skills = visible_skills()
                resumed_skill_ids = _retrieved_skill_ids(resumed_skills)
                if resumed_skill_ids != current_retrieved_skill_ids:
                    raise ReceiptValidationError(
                        "rollout checkpoint Skill condition changed"
                    )

        def persist_completed_turn(
            turn: TurnRecord,
            snapshot_event: GraphSnapshotEvent,
            *,
            next_prompt: Optional[str],
            status: str,
        ) -> None:
            if self.evidence_store is None:
                return
            self.evidence_store.append_snapshot(snapshot_event)
            next_round_index = turn.round_index + 1
            checkpoint_id = stable_id(
                "rollout_checkpoint",
                {
                    "trajectory_id": trajectory_id,
                    "next_round_index": next_round_index,
                    "turn_id": turn.turn_id,
                    "status": status,
                },
            )
            self.evidence_store.append_rollout_checkpoint(
                {
                    "schema_version": (
                        "agentgraph.rollout-turn-checkpoint.v1"
                    ),
                    "checkpoint_id": checkpoint_id,
                    "trajectory_id": trajectory_id,
                    "task_id": task.task_id,
                    "group_id": group_id,
                    "rollout_id": rollout_id,
                    "condition_id": self.condition_id,
                    "versions": self.versions.to_dict(),
                    "director_sampling": dict(
                        self.orchestrator.sampling_receipt
                    ),
                    "workflow_problem": resolved_problem,
                    "active_skill_ids": list(active_skill_ids),
                    "status": status,
                    "round_index": turn.round_index,
                    "next_round_index": next_round_index,
                    "next_prompt": next_prompt,
                    "turn": turn.to_dict(),
                    "graph_snapshot_event": snapshot_event.to_dict(),
                    "environment": env.export_runtime_checkpoint(),
                    "resumed_from_checkpoint_id": resume_checkpoint_id,
                }
            )

        round_indices: Iterable[int] = (
            ()
            if env.finished
            else range(start_round_index, self.orchestrator.max_rounds)
        )
        for round_index in round_indices:
            terminal_diagnosis = self.orchestrator.terminal_canvas_diagnosis(env)
            if terminal_diagnosis is not None:
                natural_terminal_reason = str(
                    terminal_diagnosis.get(
                        "public_error_code",
                        "canvas_action_domain_exhausted",
                    )
                )
                if turns:
                    runtime_summary = dict(turns[-1].runtime_summary)
                    runtime_summary["terminal_canvas_diagnosis"] = dict(
                        terminal_diagnosis
                    )
                    turns[-1] = replace(
                        turns[-1],
                        runtime_summary=runtime_summary,
                    )
                break
            generation_seed = self.orchestrator.generation_seed(round_index)
            schema_request = self.orchestrator.action_schema_request(env)
            response = await self.orchestrator.client.propose(
                prompt,
                seed=generation_seed,
                **schema_request,
            )
            metadata = response.metadata
            reasoning_end_token = _director_reasoning_end_token(metadata)
            parse_failure_phase = metadata.get("parse_failure_phase")
            if parse_failure_phase is not None and parse_failure_phase not in {
                _ADD_ROLE_SELECTION_PARSE_FAILURE_PHASE,
                _ADD_DECLARATION_PARSE_FAILURE_PHASE,
            }:
                raise ReceiptValidationError(
                    "Director receipt has an unsupported parse-failure phase"
                )
            if parse_failure_phase in {
                _ADD_ROLE_SELECTION_PARSE_FAILURE_PHASE,
                _ADD_DECLARATION_PARSE_FAILURE_PHASE,
            }:
                # A role selection or declaration is not a Canvas edit.  Fail
                # closed before Env.step if its raw sample happens to parse as
                # a complete AgentAction; no partial ADD may execute under
                # phase-failure metadata.
                try:
                    env.parser.parse(
                        response.text,
                        reasoning_end_token=reasoning_end_token,
                    )
                except AgentActionParseError:
                    pass
                else:
                    raise ReceiptValidationError(
                        "v3 hierarchical phase-failure sample decoded as a "
                        "Canvas action"
                    )
            if reasoning_end_token is None:
                canvas = await env.step(response.text)
            else:
                canvas = await env.step(
                    response.text,
                    reasoning_end_token=reasoning_end_token,
                )

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
                if metadata.get(
                    "action_target_domain_version"
                ) != schema_request.get("action_target_domain_version"):
                    raise ReceiptValidationError(
                        "Director target-domain version differs from the request"
                    )
                if metadata.get(
                    "action_target_domains_json"
                ) != schema_request.get("action_target_domains_json"):
                    raise ReceiptValidationError(
                        "Director target domains differ from the request"
                    )
            strategy_hint = metadata.get("action_decoding_strategy")
            receipt_base_prompt = (
                metadata.get("base_prompt_text")
                if strategy_hint is not None
                else metadata.get("prompt_text")
            )
            if receipt_base_prompt != prompt:
                raise ReceiptValidationError("Director receipt is bound to a different prompt")
            if strategy_hint == ROLE_FIRST_ADD_DECODING_STRATEGY:
                raw_phases = metadata.get("hierarchical_phase_receipts")
                role_phase = (
                    raw_phases.get("add_agent_role_selection")
                    if isinstance(raw_phases, Mapping)
                    else None
                )
                role_failure_phase = (
                    raw_phases.get(
                        _ADD_ROLE_SELECTION_SERIALIZATION_FAILURE_PHASE
                    )
                    if isinstance(raw_phases, Mapping)
                    else None
                )
                role_selection_regenerated = (
                    metadata.get("role_selection_regeneration_attempted") is True
                )
                root_phase = (
                    role_failure_phase
                    if role_selection_regenerated
                    else role_phase
                )
                if not isinstance(root_phase, Mapping) or root_phase.get(
                    "prompt_text"
                ) != prompt:
                    raise ReceiptValidationError(
                        "role-first ADD receipt is not rooted in the Canvas prompt"
                    )
            elif (
                strategy_hint == HIERARCHICAL_JSON_SCHEMA_STRATEGY
                and metadata.get("selected_action") == "add_subgraph"
                and metadata.get("action_json_schema_version")
                == DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ):
                raw_phases = metadata.get("hierarchical_phase_receipts")
                declaration_phase = (
                    raw_phases.get("add_agent_declarations")
                    if isinstance(raw_phases, Mapping)
                    else None
                )
                if not isinstance(declaration_phase, Mapping) or (
                    declaration_phase.get("prompt_text") != prompt
                ):
                    raise ReceiptValidationError(
                        "role-neutral ADD receipt is not rooted in the Canvas prompt"
                    )
            elif metadata.get("parameter_regeneration_attempted") is True:
                raw_phases = metadata.get("hierarchical_phase_receipts")
                failed_parameter_phase = (
                    raw_phases.get(_PARAMETER_SERIALIZATION_FAILURE_PHASE)
                    if isinstance(raw_phases, Mapping)
                    else None
                )
                if (
                    not isinstance(failed_parameter_phase, Mapping)
                    or failed_parameter_phase.get("prompt_text") != prompt
                ):
                    raise ReceiptValidationError(
                        "parameter regeneration is not rooted in the Canvas prompt"
                    )
            elif metadata.get("prompt_text") != prompt:
                raise ReceiptValidationError(
                    "Director final receipt is bound to a different prompt"
                )
            if _optional_int(metadata.get("generation_seed")) != generation_seed:
                raise ReceiptValidationError(
                    "Director receipt generation seed differs from the request"
                )
            if _optional_int(metadata.get("backend_sampling_seed")) != (
                _sglang_backend_sampling_seed(generation_seed)
            ):
                raise ReceiptValidationError(
                    "Director receipt backend sampling seed differs from the "
                    "signed-int64 SGLang request"
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
                if action_decoding_strategy not in {
                    HIERARCHICAL_JSON_SCHEMA_STRATEGY,
                    ROLE_FIRST_ADD_DECODING_STRATEGY,
                }:
                    raise ReceiptValidationError(
                        "Director receipt has an unsupported decoding strategy"
                    )
                selected_action = metadata.get("selected_action")
                if not isinstance(selected_action, str) or not selected_action:
                    raise ReceiptValidationError(
                        "hierarchical Director receipt has no selected action"
                    )
                live_v3_receipt = (
                    schema_request.get("action_json_schema_version")
                    == DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
                )
                if action_decoding_strategy == ROLE_FIRST_ADD_DECODING_STRATEGY and (
                    not live_v3_receipt or selected_action != "add_subgraph"
                ):
                    raise ReceiptValidationError(
                        "role-first ADD decoding requires a live-v3 ADD action"
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
                expected_v3_phases: set[str] | None = None
                if live_v3_receipt:
                    expected_v3_phases = _validate_v3_hierarchical_action_receipt(
                        action,
                        metadata,
                        schema_request,
                    )
                    if set(phase_receipts) != expected_v3_phases:
                        raise ReceiptValidationError(
                            "v3 hierarchical phase set differs from the required phases"
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
                    if _optional_int(
                        phase_receipt.get("generation_seed")
                    ) != generation_seed or _optional_int(
                        phase_receipt.get("backend_sampling_seed")
                    ) != _sglang_backend_sampling_seed(generation_seed):
                        raise ReceiptValidationError(
                            "hierarchical Director phase seed receipt differs "
                            "from the scientific/backend request pair"
                        )
                    if live_v3_receipt:
                        if phase_receipt.get(
                            "action_json_schema_version"
                        ) != schema_request.get("action_json_schema_version"):
                            raise ReceiptValidationError(
                                "v3 hierarchical phase schema version differs from the request"
                            )
                        if phase_receipt.get(
                            "action_schema_branch"
                        ) != schema_request.get("action_schema_branch"):
                            raise ReceiptValidationError(
                                "v3 hierarchical phase branch differs from the request"
                            )
                        if phase_receipt.get(
                            "action_target_domain_version"
                        ) != schema_request.get("action_target_domain_version"):
                            raise ReceiptValidationError(
                                "v3 hierarchical phase domain version differs from the request"
                            )
                        if phase_receipt.get(
                            "action_target_domains_json"
                        ) != schema_request.get("action_target_domains_json"):
                            raise ReceiptValidationError(
                                "v3 hierarchical phase domains differ from the request"
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
            runtime_summary["director_backend_sampling_seed"] = (
                metadata.get("backend_sampling_seed")
            )
            director_thinking = metadata.get("chat_template_enable_thinking")
            if director_thinking is not None:
                if type(director_thinking) is not bool:
                    raise ReceiptValidationError(
                        "Director thinking-mode receipt must be boolean"
                    )
                runtime_summary["director_chat_template_enable_thinking"] = (
                    director_thinking
                )
            if (
                resume_checkpoint_id is not None
                and round_index == start_round_index
            ):
                runtime_summary["rollout_resume"] = {
                    "checkpoint_id": resume_checkpoint_id,
                    "next_round_index": start_round_index,
                    "exact_completed_turn_boundary": True,
                }
            availability_receipt = env.model_availability_receipt()
            if availability_receipt["failure_receipts"]:
                runtime_summary["model_availability"] = {
                    "model_catalog_version": self.versions.model_catalog,
                    **availability_receipt,
                }
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
            action_target_domain_version = metadata.get(
                "action_target_domain_version"
            )
            if action_target_domain_version is not None:
                runtime_summary["director_action_target_domain_version"] = (
                    action_target_domain_version
                )
                raw_target_domains = metadata.get("action_target_domains_json")
                if not isinstance(raw_target_domains, str):
                    raise ReceiptValidationError(
                        "Director receipt has no serialized target domains"
                    )
                try:
                    runtime_summary["director_action_target_domains"] = json.loads(
                        raw_target_domains
                    )
                except (TypeError, ValueError) as exc:
                    raise ReceiptValidationError(
                        "Director receipt target domains are not JSON"
                    ) from exc
            if action_decoding_strategy is not None:
                action_decoding = {
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
                if metadata.get("selected_relation_candidate") is not None:
                    action_decoding["selected_relation_candidate"] = metadata.get(
                        "selected_relation_candidate"
                    )
                if "selected_add_agent_ids" in metadata:
                    action_decoding["selected_add_agent_ids"] = metadata.get(
                        "selected_add_agent_ids"
                    )
                if "selected_add_agent_roles" in metadata:
                    action_decoding["selected_add_agent_roles"] = metadata.get(
                        "selected_add_agent_roles"
                    )
                if "selected_modify_agent_id" in metadata:
                    action_decoding["selected_modify_agent_id"] = metadata.get(
                        "selected_modify_agent_id"
                    )
                if metadata.get("parse_failure_phase") is not None:
                    action_decoding["parse_failure_phase"] = metadata.get(
                        "parse_failure_phase"
                    )
                if metadata.get("parameter_regeneration_attempted") is not None:
                    action_decoding["parameter_regeneration_attempted"] = (
                        metadata.get("parameter_regeneration_attempted")
                    )
                    action_decoding["parameter_regeneration_succeeded"] = (
                        metadata.get("parameter_regeneration_succeeded")
                    )
                if (
                    metadata.get("role_selection_regeneration_attempted")
                    is not None
                ):
                    action_decoding["role_selection_regeneration_attempted"] = (
                        metadata.get("role_selection_regeneration_attempted")
                    )
                    action_decoding["role_selection_regeneration_succeeded"] = (
                        metadata.get("role_selection_regeneration_succeeded")
                    )
                runtime_summary["director_action_decoding"] = action_decoding
            turn = TurnRecord(
                turn_id=stable_id(
                    "turn",
                    {
                        "trajectory_id": trajectory_id,
                        "round_index": round_index,
                    },
                ),
                round_index=round_index,
                prompt=str(metadata.get("prompt_text")),
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
                feedback_code=canvas.feedback_code,
            )
            turns.append(turn)
            snapshots.append(snapshot)
            previous_snapshot_id = snapshot.snapshot_id

            if canvas.done:
                explicit_finish = True
                final_answer = canvas.final_answer
                final_runtime = canvas.execution
                final_graph = env.graph.to_dict()
                persist_completed_turn(
                    turns[-1],
                    snapshot,
                    next_prompt=None,
                    status="terminal_pending_evaluation",
                )
                break
            terminal_diagnosis = self.orchestrator.terminal_canvas_diagnosis(env)
            if terminal_diagnosis is not None:
                natural_terminal_reason = str(
                    terminal_diagnosis.get(
                        "public_error_code",
                        "canvas_action_domain_exhausted",
                    )
                )
                runtime_summary = dict(turns[-1].runtime_summary)
                runtime_summary["terminal_canvas_diagnosis"] = dict(
                    terminal_diagnosis
                )
                turns[-1] = replace(
                    turns[-1],
                    runtime_summary=runtime_summary,
                )
                persist_completed_turn(
                    turns[-1],
                    snapshot,
                    next_prompt=prompt,
                    status="in_progress",
                )
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
            persist_completed_turn(
                turns[-1],
                snapshot,
                next_prompt=prompt,
                status="in_progress",
            )


        termination_reason = (
            "finish"
            if explicit_finish
            else natural_terminal_reason or "max_rounds"
        )
        if termination_reason != "finish":
            # Progressive artifacts remain available on their immutable turn
            # receipts for diagnosis, but only an explicit FINISH admits a
            # terminal answer to the evaluator.  Never promote historical
            # evidence lineage after max_rounds or a Canvas-domain terminal.
            final_answer = None
            final_runtime = None
            final_graph = env.graph.to_dict()
            valid_lineage_fallback_used = False
            valid_lineage_fallback_receipt = {}

        if final_graph is None:
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
            valid_lineage_fallback_used=valid_lineage_fallback_used,
            valid_lineage_fallback_receipt=valid_lineage_fallback_receipt,
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
            completion_checkpoint_id = stable_id(
                "rollout_checkpoint",
                {
                    "trajectory_id": trajectory_id,
                    "status": "completed",
                },
            )
            self.evidence_store.append_rollout_checkpoint(
                {
                    "schema_version": (
                        "agentgraph.rollout-turn-checkpoint.v1"
                    ),
                    "checkpoint_id": completion_checkpoint_id,
                    "trajectory_id": trajectory_id,
                    "status": "completed",
                    "next_round_index": len(turns),
                    "completed_trajectory_id": trajectory.trajectory_id,
                }
            )
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
