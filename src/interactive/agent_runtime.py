"""Asynchronous, finite AgentGraph execution over a fake-friendly gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from types import MappingProxyType
from typing import Awaitable, Collection, Dict, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Set, Tuple, Union
import uuid

from .agent_graph import (
    AgentGraph,
    AgentGraphValidationError,
    AgentNode,
    AgentRelation,
)
from .model_registry import ModelRegistry, ModelSpec, ProviderSpec
from .tool_runtime import ToolRegistry


class ExecutionPhase(str, Enum):
    SINGLE = "single"
    DRAFT = "draft"
    REVISION = "revision"


class CommunicationCondition(str, Enum):
    """Execution-only communication condition, never a task reward signal."""

    NORMAL = "normal"
    UPSTREAM_MASKED = "upstream_masked"


ARTIFACT_COMMUNICATION_LEGACY = "legacy"
ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1 = (
    "producer_context_exact_dedup_v1"
)
ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2 = (
    "producer_context_structured_evidence_v2"
)
ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3 = (
    "producer_context_structured_evidence_v3"
)
_ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_PROFILES = frozenset(
    {
        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1,
        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2,
        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3,
    }
)
_ARTIFACT_COMMUNICATION_PROFILES = frozenset(
    {
        ARTIFACT_COMMUNICATION_LEGACY,
        *_ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_PROFILES,
    }
)

ARTIFACT_QUALITY_NONE = "none"
ARTIFACT_QUALITY_PUBLIC_TEXT_V1 = "public_text_quality_v1"
ARTIFACT_QUALITY_PUBLIC_TEXT_V2 = "public_text_quality_v2"
_ARTIFACT_QUALITY_PROFILES = frozenset(
    {
        ARTIFACT_QUALITY_NONE,
        ARTIFACT_QUALITY_PUBLIC_TEXT_V1,
        ARTIFACT_QUALITY_PUBLIC_TEXT_V2,
    }
)
_PUBLIC_TEXT_MAX_CHARACTERS_V1 = 12_000


def _public_text_quality_receipt(
    text: str,
    metadata: Mapping[str, object],
    *,
    profile: str,
) -> Mapping[str, object]:
    """Measure truncation and lexical degeneration on a public Artifact.

    The check is intentionally semantic-free: it does not consult the task,
    evaluator, rubric, reference response, Agent role, or graph topology.  It
    detects only provider truncation, invalid characters, excessive length,
    exact paragraph repetition, repeated token n-grams, and extreme run-on
    output.  The immutable v2 profile additionally detects unmistakably
    incomplete headings/lead-ins and CJK sentence boundaries.  The Artifact
    is rejected as a whole; it is never truncated or rewritten by Runtime.
    """

    error_codes: list[str] = []
    finish_reason = metadata.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason.casefold() == "length":
        error_codes.append("provider_length_termination")
    if metadata.get("artifact_complete") is False:
        error_codes.append("provider_incomplete_artifact")
    if len(text) > _PUBLIC_TEXT_MAX_CHARACTERS_V1:
        error_codes.append("artifact_character_limit_exceeded")
    if "\ufffd" in text or any(
        ord(character) < 32 and character not in "\n\r\t"
        for character in text
    ):
        error_codes.append("invalid_text_character")
    if re.search(r"\S{513,}", text) is not None:
        error_codes.append("abnormally_long_token")

    if profile == ARTIFACT_QUALITY_PUBLIC_TEXT_V2:
        stripped_text = text.strip()
        nonempty_lines = [
            line.strip() for line in text.splitlines() if line.strip()
        ]
        # v2 adds one surface-completeness check without changing the
        # published v1 condition.  A lone heading or introductory clause
        # ending in a colon cannot be the complete public Artifact promised by
        # a free-text contract.  The existing SkillFlow-style failure path
        # returns the typed failure to the Director for repair.
        heading_only = bool(
            len(nonempty_lines) == 1
            and re.fullmatch(r"#{1,6}\s+[^\r\n]+", nonempty_lines[0])
        )
        incomplete_leadin = bool(
            len(nonempty_lines) == 1
            and stripped_text.endswith((":", "："))
        )
        if heading_only or incomplete_leadin:
            error_codes.append("incomplete_heading_or_leadin")

    normalized_paragraphs = [
        " ".join(paragraph.split()).casefold()
        for paragraph in re.split(r"\n\s*\n", text)
        if len(" ".join(paragraph.split())) >= 80
    ]
    paragraph_counts: dict[str, int] = {}
    for paragraph in normalized_paragraphs:
        paragraph_counts[paragraph] = paragraph_counts.get(paragraph, 0) + 1
    maximum_paragraph_repetitions = max(paragraph_counts.values(), default=0)
    if maximum_paragraph_repetitions >= 3:
        error_codes.append("repeated_paragraph")

    tokens = re.findall(r"[\w'-]+|[^\w\s]", text.casefold())
    window_size = 24
    window_counts: dict[Tuple[str, ...], int] = {}
    if len(tokens) >= 192:
        for index in range(0, len(tokens) - window_size + 1):
            window = tuple(tokens[index : index + window_size])
            window_counts[window] = window_counts.get(window, 0) + 1
    maximum_ngram_repetitions = max(window_counts.values(), default=0)
    if maximum_ngram_repetitions >= 3:
        error_codes.append("repeated_token_ngram")

    lines = text.splitlines() or [text]
    longest_line_characters = max((len(line) for line in lines), default=0)
    sentence_boundary_pattern = (
        r"[.!?。！？](?:\s|$)?"
        if profile == ARTIFACT_QUALITY_PUBLIC_TEXT_V2
        else r"[.!?](?:\s|$)?"
    )
    sentence_boundaries = len(re.findall(sentence_boundary_pattern, text))
    structured_json = False
    if text.lstrip().startswith(("{", "[")):
        try:
            json.loads(text)
        except (TypeError, ValueError):
            pass
        else:
            structured_json = True
    if (
        longest_line_characters > 5_000
        and sentence_boundaries < 4
        and not structured_json
    ):
        error_codes.append("run_on_text")

    distinct_error_codes = list(dict.fromkeys(error_codes))
    return MappingProxyType(
        {
            "profile": profile,
            "status": "invalid" if distinct_error_codes else "valid",
            "error_codes": distinct_error_codes,
            "character_count": len(text),
            "token_count": len(tokens),
            "longest_line_characters": longest_line_characters,
            "sentence_boundary_count": sentence_boundaries,
            "maximum_exact_paragraph_repetitions": (
                maximum_paragraph_repetitions
            ),
            "maximum_24_token_ngram_repetitions": (
                maximum_ngram_repetitions
            ),
            "structured_json": structured_json,
        }
    )


def _communication_condition(
    value: Union[CommunicationCondition, str],
) -> CommunicationCondition:
    if isinstance(value, CommunicationCondition):
        return value
    if not isinstance(value, str):
        raise TypeError("communication_condition must be a string or CommunicationCondition")
    try:
        return CommunicationCondition(value.strip())
    except ValueError as exc:
        raise ValueError(
            "communication_condition must be normal or upstream_masked"
        ) from exc


@dataclass(frozen=True, slots=True)
class UpstreamMessage:
    """Typed public communication envelope between AgentGraph nodes.

    ``content``/``message_type`` are retained for trajectory compatibility.
    ``artifact_type`` and the remaining receipt fields implement the project's
    minimal typed-envelope extension required because FlowSteer's text-only
    messages cannot represent tool, environment, or coding observations.
    """

    source_agent_id: str
    target_agent_id: str
    content: str
    message_type: str = "artifact"
    graph_revision: Optional[int] = None
    request_or_dependency: Optional[str] = None
    artifact_type: str = "text"
    environment_revision: Optional[int] = None
    tool_receipts: Tuple[Mapping[str, object], ...] = ()
    artifact_version: Optional[str] = None
    source_model_id: Optional[str] = None
    source_contract: Optional[str] = None
    source_execution_mode: Optional[str] = None
    source_role_family: Optional[str] = None
    source_completion_condition: Optional[str] = None
    source_finish_reason: Optional[str] = None
    input_artifact_provenance: Tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_agent_id, "source_agent_id"),
            (self.target_agent_id, "target_agent_id"),
            (self.content, "content"),
            (self.message_type, "message_type"),
            (self.artifact_type, "artifact_type"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.graph_revision is not None and (
            isinstance(self.graph_revision, bool)
            or not isinstance(self.graph_revision, int)
            or self.graph_revision < 0
        ):
            raise ValueError("graph_revision must be non-negative when supplied")
        if self.request_or_dependency is not None and (
            not isinstance(self.request_or_dependency, str)
            or not self.request_or_dependency.strip()
        ):
            raise ValueError(
                "request_or_dependency must be non-empty when supplied"
            )
        if self.environment_revision is not None and (
            isinstance(self.environment_revision, bool)
            or not isinstance(self.environment_revision, int)
            or self.environment_revision < 0
        ):
            raise ValueError("environment_revision must be non-negative when supplied")
        if self.artifact_version is not None and (
            not isinstance(self.artifact_version, str)
            or not self.artifact_version.strip()
        ):
            raise ValueError("artifact_version must be non-empty when supplied")
        for value, name in (
            (self.source_model_id, "source_model_id"),
            (self.source_contract, "source_contract"),
            (self.source_execution_mode, "source_execution_mode"),
            (self.source_role_family, "source_role_family"),
            (
                self.source_completion_condition,
                "source_completion_condition",
            ),
            (self.source_finish_reason, "source_finish_reason"),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be non-empty when supplied")
        if not isinstance(self.tool_receipts, tuple) or any(
            not isinstance(item, Mapping) for item in self.tool_receipts
        ):
            raise TypeError("tool_receipts must be a tuple of mappings")
        if not isinstance(self.input_artifact_provenance, tuple) or any(
            not isinstance(item, Mapping)
            for item in self.input_artifact_provenance
        ):
            raise TypeError(
                "input_artifact_provenance must be a tuple of mappings"
            )
        object.__setattr__(
            self,
            "tool_receipts",
            tuple(MappingProxyType(dict(item)) for item in self.tool_receipts),
        )
        object.__setattr__(
            self,
            "input_artifact_provenance",
            tuple(
                MappingProxyType(dict(item))
                for item in self.input_artifact_provenance
            ),
        )

    @property
    def artifact(self) -> str:
        return self.content

    @property
    def artifact_body(self) -> str:
        return self.content

    @property
    def dependency(self) -> Optional[str]:
        return self.request_or_dependency

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "message_type": self.message_type,
            "artifact_type": self.artifact_type,
            "artifact": self.content,
            "artifact_body": self.content,
            "content": self.content,
            "graph_revision": self.graph_revision,
            "environment_revision": self.environment_revision,
            "artifact_version": self.artifact_version,
            "source_model_id": self.source_model_id,
            "source_contract": self.source_contract,
            "source_execution_mode": self.source_execution_mode,
            "source_role_family": self.source_role_family,
            "source_completion_condition": self.source_completion_condition,
            "source_finish_reason": self.source_finish_reason,
            "request_or_dependency": self.request_or_dependency,
            "dependency": self.request_or_dependency,
            "tool_receipts": [dict(item) for item in self.tool_receipts],
            "input_artifact_provenance": [
                dict(item) for item in self.input_artifact_provenance
            ],
        }


# Public terminology from the project design; the legacy name remains the
# canonical class so persisted trajectories and downstream imports stay valid.
CommunicationEnvelope = UpstreamMessage


@dataclass(frozen=True, slots=True)
class AgentRequest:
    request_id: str
    run_id: str
    graph_revision: int
    problem: str
    agent: AgentNode
    model: ModelSpec
    provider: ProviderSpec
    phase: ExecutionPhase
    is_output_agent: bool = False
    is_format_agent: bool = False
    is_format_predecessor: bool = False
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL
    upstream: Tuple[UpstreamMessage, ...] = ()
    own_draft: Optional[str] = None
    peer_draft: Optional[UpstreamMessage] = None
    semantic_protocol: str = "none"
    artifact_communication_profile: str = ARTIFACT_COMMUNICATION_LEGACY
    # SkillFlow continuation state: a repaired Agent keeps the public
    # Action--Observation history and measured Tool receipts from its failed
    # bounded execution.  These fields never contain hidden reasoning and do
    # not turn a failed completion into a semantic artifact.
    action_history: Tuple[Mapping[str, object], ...] = ()
    prior_tool_receipts: Tuple[Mapping[str, object], ...] = ()
    continuation_source_agent_id: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.is_output_agent) is not bool:
            raise TypeError("is_output_agent must be bool")
        if type(self.is_format_agent) is not bool:
            raise TypeError("is_format_agent must be bool")
        if type(self.is_format_predecessor) is not bool:
            raise TypeError("is_format_predecessor must be bool")
        if self.is_format_agent and not self.is_output_agent:
            raise ValueError("Format Agent must be the Output Agent")
        if self.is_format_agent and self.is_format_predecessor:
            raise ValueError("Format Agent cannot be its own predecessor")
        if self.semantic_protocol not in {
            "none",
            "hotpotqa_verified_answer_slot_v1",
            "qa_verified_answer_lineage_v2",
        }:
            raise ValueError("unsupported AgentRequest semantic_protocol")
        if self.artifact_communication_profile not in (
            _ARTIFACT_COMMUNICATION_PROFILES
        ):
            raise ValueError(
                "unsupported AgentRequest artifact communication profile"
            )
        if any(not isinstance(item, Mapping) for item in self.action_history):
            raise TypeError("AgentRequest.action_history must contain mappings")
        if any(not isinstance(item, Mapping) for item in self.prior_tool_receipts):
            raise TypeError(
                "AgentRequest.prior_tool_receipts must contain mappings"
            )
        if self.continuation_source_agent_id is not None and (
            not isinstance(self.continuation_source_agent_id, str)
            or not self.continuation_source_agent_id.strip()
        ):
            raise ValueError(
                "AgentRequest.continuation_source_agent_id must be non-empty text"
            )
        object.__setattr__(
            self,
            "action_history",
            tuple(MappingProxyType(dict(item)) for item in self.action_history),
        )
        object.__setattr__(
            self,
            "prior_tool_receipts",
            tuple(
                MappingProxyType(dict(item)) for item in self.prior_tool_receipts
            ),
        )
        object.__setattr__(
            self,
            "communication_condition",
            _communication_condition(self.communication_condition),
        )


@dataclass(frozen=True, slots=True)
class AgentResponse:
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("AgentResponse.text must be a string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


GatewayResponse = Union[AgentResponse, str]


class AgentGateway(Protocol):
    """Provider-neutral async boundary; tests can implement one method."""

    async def generate(self, request: AgentRequest) -> GatewayResponse:
        ...


class AgentExecutionAdapter(Protocol):
    """Unified execution boundary for reasoning, ReAct, and coding nodes."""

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        ...


@dataclass(frozen=True, slots=True)
class ReasoningExecutionAdapter:
    """Thin adapter retaining the existing provider-gateway execution path."""

    gateway: AgentGateway

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        return await self.gateway.generate(request)


def _tool_receipts_from_metadata(
    metadata: Mapping[str, object],
) -> Tuple[Mapping[str, object], ...]:
    """Return only concrete serialized receipts emitted by an adapter."""

    raw = metadata.get("tool_receipts", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        MappingProxyType(dict(item)) for item in raw if isinstance(item, Mapping)
    )


def _input_artifact_provenance_from_metadata(
    metadata: Mapping[str, object],
) -> Tuple[Mapping[str, object], ...]:
    """Read nested, source-bound provenance without flattening its graph path."""

    raw = metadata.get("input_artifact_provenance", ())
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        MappingProxyType(dict(item))
        for item in raw
        if isinstance(item, Mapping)
    )


def _environment_revision_from_metadata(
    metadata: Mapping[str, object],
) -> Optional[int]:
    raw = metadata.get("environment_revision")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


def _finish_reason_from_metadata(
    metadata: Mapping[str, object],
) -> Optional[str]:
    raw = metadata.get("finish_reason")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _producer_context(
    node: AgentNode,
    metadata: Mapping[str, object],
    *,
    artifact_communication_profile: str,
) -> Dict[str, Optional[str]]:
    """Project the existing Agent declaration into one routed artifact."""

    if artifact_communication_profile not in (
        _ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_PROFILES
    ):
        return {}
    return {
        "source_model_id": node.model_id,
        "source_contract": node.contract,
        "source_execution_mode": node.execution_mode.value,
        "source_role_family": node.role_family,
        "source_completion_condition": node.completion_condition,
        "source_finish_reason": _finish_reason_from_metadata(metadata),
    }


@dataclass(frozen=True, slots=True)
class AgentCallRecord:
    request: AgentRequest
    response: AgentResponse


@dataclass(frozen=True, slots=True)
class AgentRuntimeResult:
    run_id: str
    graph_revision: int
    output_agent_id: Optional[str]
    final_answer: Optional[str]
    outputs: Mapping[str, str]
    calls: Tuple[AgentCallRecord, ...]
    block_completion_order: Tuple[Tuple[str, ...], ...]
    executed_agent_ids: Tuple[str, ...] = ()
    reused_agent_ids: Tuple[str, ...] = ()
    deferred_agent_ids: Tuple[str, ...] = ()
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL
    output_metadata: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict,
        compare=False,
    )
    execution_reuse_receipts: Tuple[Mapping[str, object], ...] = field(
        default_factory=tuple,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(
            self,
            "communication_condition",
            _communication_condition(self.communication_condition),
        )
        object.__setattr__(
            self,
            "output_metadata",
            MappingProxyType(
                {
                    agent_id: MappingProxyType(dict(metadata))
                    for agent_id, metadata in self.output_metadata.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "execution_reuse_receipts",
            tuple(
                MappingProxyType(dict(receipt))
                for receipt in self.execution_reuse_receipts
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentExecutionCacheEntry:
    """One complete, task-local quotient-component execution artifact.

    The cache key is the canonical semantic input identity constructed by
    :class:`AgentRuntime`.  Transport identifiers are deliberately absent from
    that identity, while the original artifact versions and provenance remain
    attached to the cached outputs.  A caller must scope the mutable cache to
    one task/trajectory and clear it on reset.
    """

    source_graph_revision: int
    outputs: Mapping[str, str]
    output_metadata: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_graph_revision, bool)
            or not isinstance(self.source_graph_revision, int)
            or self.source_graph_revision < 0
        ):
            raise ValueError("source_graph_revision must be non-negative")
        frozen_outputs: Dict[str, str] = {}
        for agent_id, artifact in self.outputs.items():
            if not isinstance(agent_id, str) or not isinstance(artifact, str):
                raise TypeError("cached component outputs must map strings to strings")
            if not artifact.strip():
                raise ValueError("cached component outputs must be non-empty")
            frozen_outputs[agent_id] = artifact
        frozen_metadata: Dict[str, Mapping[str, object]] = {}
        for agent_id, metadata in self.output_metadata.items():
            if not isinstance(agent_id, str) or not isinstance(metadata, Mapping):
                raise TypeError(
                    "cached component output metadata must map strings to mappings"
                )
            frozen_metadata[agent_id] = MappingProxyType(dict(metadata))
        if set(frozen_outputs) != set(frozen_metadata):
            raise ValueError(
                "cached component outputs and output metadata must have identical agents"
            )
        object.__setattr__(self, "outputs", MappingProxyType(frozen_outputs))
        object.__setattr__(
            self,
            "output_metadata",
            MappingProxyType(frozen_metadata),
        )


ComponentExecutionCache = MutableMapping[str, ComponentExecutionCacheEntry]

_COMPONENT_EXECUTION_INPUT_IDENTITY_VERSION = "agentgraph.component-input.v2"
_TOOL_RECEIPT_TRANSPORT_FIELDS = frozenset(
    {
        "ended_at_monotonic",
        "latency_ms",
        "provider_request_id",
        "started_at_monotonic",
    }
)


def _semantic_tool_receipt_value(value: object) -> object:
    """Remove measurement-only fields from one public Tool receipt value."""

    if isinstance(value, Mapping):
        return {
            str(key): _semantic_tool_receipt_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _TOOL_RECEIPT_TRANSPORT_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_tool_receipt_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("Tool receipt identity values must be JSON-compatible")


@dataclass(frozen=True, slots=True)
class AgentFailureRecord:
    """Public execution-failure receipt for one Agent invocation.

    SkillFlow retains the sampled action and public observation when a bounded
    executor fails to complete.  AgentGraph keeps the same public receipt at
    the Runtime boundary so a Canvas correction does not erase already-spent
    Tool calls.  The record never becomes an upstream semantic artifact.
    """

    request_id: str
    agent_id: str
    phase: ExecutionPhase
    graph_revision: int
    error_type: str
    message: str
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "phase": self.phase.value,
            "graph_revision": self.graph_revision,
            "error_type": self.error_type,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


class AgentRuntimeError(RuntimeError):
    """Wraps a gateway or scheduler failure with AgentGraph context."""

    def __init__(
        self,
        message: str,
        *,
        failure_records: Tuple[AgentFailureRecord, ...] = (),
        partial_result: Optional[AgentRuntimeResult] = None,
        blocked_agent_ids: Tuple[str, ...] = (),
        pending_agent_ids: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure_records = tuple(failure_records)
        self.partial_result = partial_result
        self.blocked_agent_ids = tuple(sorted(set(blocked_agent_ids)))
        self.pending_agent_ids = tuple(sorted(set(pending_agent_ids)))


def _public_failure_metadata(exc: BaseException) -> Mapping[str, object]:
    """Copy only adapter-published Action--Observation failure receipts."""

    result: Dict[str, object] = {}
    for field_name in (
        "react_trace",
        "tool_receipts",
        "model_calls",
        "environment_reset_receipt",
        "environment_receipts",
        "evaluator_environment_trace",
    ):
        value = getattr(exc, field_name, None)
        if isinstance(value, Mapping):
            result[field_name] = dict(value)
        elif isinstance(value, (list, tuple)):
            result[field_name] = [
                dict(item) if isinstance(item, Mapping) else item for item in value
            ]
    for field_name in (
        "environment_revision",
        "environment_terminal",
        "cause_error_type",
        "tool_plan_exhausted",
        "provider_id",
        "model_id",
        "http_status",
        "request_status",
    ):
        value = getattr(exc, field_name, None)
        if value is not None:
            result[field_name] = value
    # Failure recovery may delete a node only when the execution boundary has
    # explicitly diagnosed the node itself as unusable.  Provider, Tool, ReAct,
    # timeout, and contract failures do not imply this flag.  Preserve the
    # typed adapter receipt without inferring it from exception text.
    node_unusable = getattr(exc, "node_unusable", None)
    if type(node_unusable) is bool:
        result["node_unusable"] = node_unusable
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class _ExecutionPlan:
    components: Tuple[Tuple[str, ...], ...]
    component_for: Mapping[str, Tuple[str, ...]]
    successors: Mapping[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]
    indegree: Mapping[Tuple[str, ...], int]
    relations: Tuple[AgentRelation, ...]


async def _cancel_and_wait(tasks: List["asyncio.Task[object]"]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _gather_pair(
    left: Awaitable[AgentResponse],
    right: Awaitable[AgentResponse],
) -> Tuple[AgentResponse, AgentResponse]:
    tasks = [asyncio.create_task(left), asyncio.create_task(right)]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        await _cancel_and_wait(tasks)  # type: ignore[arg-type]
        raise
    return results[0], results[1]


class AgentRuntime:
    """Execute quotient-DAG blocks and finite two-agent reciprocal exchanges."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        gateway: AgentGateway,
        *,
        max_concurrency: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        execution_adapters: Optional[Mapping[str, AgentExecutionAdapter]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        dataset_id: Optional[str] = None,
        semantic_protocol: str = "none",
        artifact_communication_profile: str = ARTIFACT_COMMUNICATION_LEGACY,
        artifact_quality_profile: str = ARTIFACT_QUALITY_NONE,
        execution_profile_allowlist: Optional[
            Sequence[Tuple[str, Sequence[str]]]
        ] = None,
    ) -> None:
        if max_concurrency is not None and (
            type(max_concurrency) is not int or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if dataset_id is not None and (
            not isinstance(dataset_id, str) or not dataset_id.strip()
        ):
            raise ValueError("dataset_id must be non-empty when supplied")
        if semantic_protocol not in {
            "none",
            "hotpotqa_verified_answer_slot_v1",
            "qa_verified_answer_lineage_v2",
        }:
            raise ValueError("unsupported AgentRuntime semantic_protocol")
        if artifact_communication_profile not in _ARTIFACT_COMMUNICATION_PROFILES:
            raise ValueError("unsupported AgentRuntime artifact communication profile")
        if artifact_quality_profile not in _ARTIFACT_QUALITY_PROFILES:
            raise ValueError("unsupported AgentRuntime artifact quality profile")
        self.model_registry = model_registry
        self.gateway = gateway
        adapters: Dict[str, AgentExecutionAdapter] = {
            "reasoning": ReasoningExecutionAdapter(gateway)
        }
        for mode, adapter in dict(execution_adapters or {}).items():
            if mode not in {"reasoning", "react", "coding"}:
                raise ValueError(
                    "execution adapter mode must be reasoning, react, or coding"
                )
            if not hasattr(adapter, "execute"):
                raise TypeError("execution adapters must implement execute")
            adapters[mode] = adapter
        self.execution_adapters: Mapping[str, AgentExecutionAdapter] = (
            MappingProxyType(adapters)
        )
        self.tool_registry = tool_registry
        self.dataset_id = None if dataset_id is None else dataset_id.strip()
        self.semantic_protocol = semantic_protocol
        self.artifact_communication_profile = artifact_communication_profile
        self.artifact_quality_profile = artifact_quality_profile
        self._execution_profile_allowlist = (
            self._validate_execution_profile_allowlist(
                execution_profile_allowlist
            )
        )
        self.timeout_seconds = timeout_seconds
        self._global_semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        self._provider_semaphores: Dict[str, asyncio.Semaphore] = {
            provider_id: asyncio.Semaphore(
                self.model_registry.require_provider(provider_id).max_concurrency  # type: ignore[arg-type]
            )
            for provider_id in self.model_registry.provider_ids
            if self.model_registry.require_provider(provider_id).max_concurrency is not None
        }

    def _registered_execution_profiles_unfiltered(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        profiles: list[Tuple[str, Tuple[str, ...]]] = []
        for mode_value in ("reasoning", "react", "coding"):
            if mode_value not in self.execution_adapters:
                continue
            if mode_value != "coding":
                profiles.append((mode_value, ()))
            if mode_value == "reasoning" or self.tool_registry is None:
                continue
            for tool_id in self.tool_registry.resource_ids:
                try:
                    capability = self.tool_registry.require_capability(tool_id)
                except KeyError:
                    continue
                if not capability.availability:
                    continue
                if (
                    self.dataset_id is not None
                    and not capability.supports_dataset(self.dataset_id)
                ):
                    continue
                profiles.append((mode_value, (tool_id,)))
        return tuple(profiles)

    def _validate_execution_profile_allowlist(
        self,
        value: Optional[Sequence[Tuple[str, Sequence[str]]]],
    ) -> Optional[Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        """Bind an optional condition profile to registered Runtime resources."""

        if value is None:
            return None
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("execution_profile_allowlist must be a sequence")
        if not value:
            raise ValueError("execution_profile_allowlist must not be empty")
        normalized: list[Tuple[str, Tuple[str, ...]]] = []
        for item in value:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0].strip()
                or item[0] != item[0].strip()
                or isinstance(item[1], (str, bytes))
                or not isinstance(item[1], Sequence)
            ):
                raise TypeError(
                    "execution_profile_allowlist entries must be "
                    "(execution_mode, allowed_tools) pairs"
                )
            execution_mode = item[0]
            allowed_tools = tuple(item[1])
            if any(
                not isinstance(tool_id, str)
                or not tool_id.strip()
                or tool_id != tool_id.strip()
                for tool_id in allowed_tools
            ):
                raise TypeError(
                    "execution_profile_allowlist Tool IDs must be canonical "
                    "non-empty strings"
                )
            if len(allowed_tools) != len(set(allowed_tools)):
                raise ValueError(
                    "execution_profile_allowlist Tool IDs must be unique"
                )
            profile = (execution_mode, allowed_tools)
            if profile in normalized:
                raise ValueError(
                    "execution_profile_allowlist profiles must be unique"
                )
            normalized.append(profile)

        registered = set(self._registered_execution_profiles_unfiltered())
        unregistered = tuple(
            profile for profile in normalized if profile not in registered
        )
        if unregistered:
            raise ValueError(
                "execution_profile_allowlist contains an unregistered "
                f"Runtime profile: {unregistered[0]!r}"
            )
        return tuple(normalized)

    def registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return execution-mode/Tool pairs exposed by this Runtime.

        This is the shared FlowSteer Canvas capability boundary used by the
        live action domain.  It describes registered executors and available
        task-scoped resources; semantic role names do not create an executor.
        An optional condition allowlist narrows this same boundary without
        changing Agent contracts, relations, counts, or topology.
        """

        registered = self._registered_execution_profiles_unfiltered()
        if self._execution_profile_allowlist is None:
            return registered
        allowed = set(self._execution_profile_allowlist)
        return tuple(profile for profile in registered if profile in allowed)

    def model_supports_execution_profile(
        self,
        model_id: str,
        execution_mode: str,
        allowed_tools: Sequence[str],
    ) -> bool:
        """Return whether catalog metadata admits one executable model/profile.

        FlowSteer's original Runtime assumed a homogeneous model pool and
        exposed model IDs separately from execution profiles.  Heterogeneous
        AgentGraph catalogs need the same SkillFlow capability admission at
        the joint boundary.  Explicit ``false`` metadata is authoritative;
        catalogs without capability metadata retain legacy compatibility.
        """

        model = self.model_registry.require_model(model_id)
        metadata = model.metadata
        if metadata.get("text_capable", "true").casefold() == "false":
            return False
        if execution_mode == "coding":
            coding_capable = metadata.get("coding_capable")
            if (
                coding_capable is not None
                and coding_capable.casefold() != "true"
            ):
                return False
        if execution_mode == "react":
            tool_capable = metadata.get("tool_capable")
            if tool_capable is not None and tool_capable.casefold() != "true":
                return False
        canonical_tools = tuple(allowed_tools)
        if not canonical_tools:
            return True
        tool_capable = metadata.get("tool_capable")
        if tool_capable is not None and tool_capable.casefold() != "true":
            return False
        raw_scope = metadata.get("tool_capability_scope")
        if raw_scope:
            admitted_tools = {
                tool_id.strip()
                for tool_id in raw_scope.split(",")
                if tool_id.strip()
            }
            if not set(canonical_tools) <= admitted_tools:
                return False
        return True

    def registered_execution_profiles_for_model(
        self,
        model_id: str,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Project exact Runtime profiles compatible with one model arm."""

        return tuple(
            profile
            for profile in self.registered_execution_profiles()
            if self.model_supports_execution_profile(
                model_id,
                profile[0],
                profile[1],
            )
        )

    def model_execution_profiles(
        self,
        model_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Tuple[str, str, Tuple[str, ...]], ...]:
        """Return the live model/execution/Tool joint capability domain."""

        selected_model_ids = (
            self.model_registry.model_ids
            if model_ids is None
            else tuple(model_ids)
        )
        return tuple(
            (model_id, execution_mode, allowed_tools)
            for model_id in selected_model_ids
            for execution_mode, allowed_tools in (
                self.registered_execution_profiles_for_model(model_id)
            )
        )

    @staticmethod
    def _component_cache_eligible(
        component: Tuple[str, ...],
        nodes: Mapping[str, AgentNode],
        failure_metadata: Mapping[str, Mapping[str, object]],
    ) -> bool:
        """Admit only complete Tool-free reasoning execution units.

        ReAct and coding adapters may own state outside the model request, so
        equal rendered inputs do not prove that replay is side-effect free.
        Failure continuation likewise represents a new bounded execution state
        and must never be bypassed by a successful historical artifact.
        """

        return bool(component) and all(
            nodes[agent_id].execution_mode.value == "reasoning"
            and not nodes[agent_id].allowed_tools
            and not failure_metadata.get(agent_id)
            for agent_id in component
        )

    def _component_execution_input_identity(
        self,
        component: Tuple[str, ...],
        nodes: Mapping[str, AgentNode],
        plan: _ExecutionPlan,
        outputs: Mapping[str, str],
        output_metadata: Mapping[str, Mapping[str, object]],
        problem: str,
        *,
        output_agent_id: Optional[str],
        format_output_agent: bool,
        communication_condition: CommunicationCondition,
    ) -> Optional[str]:
        """Return a canonical semantic identity for one executable component.

        The identity mirrors FlowSteer's node-cache ``operator + inputs``
        boundary for the heterogeneous AgentGraph scheduler.  It contains every
        field that changes the model-visible task, contract, model, execution
        semantics, or routed public artifact. Run IDs and graph revisions remain
        transport coordinates. Under the versioned producer-context profile,
        the routed artifact version is part of the declared dependency and a
        version change invalidates downstream reuse.
        """

        format_predecessor_ids = {
            source_id
            for relation in plan.relations
            for source_id, target_id in relation.directed_edges()
            if format_output_agent
            and output_agent_id is not None
            and target_id == output_agent_id
        }
        component_ids = set(component)
        agents = []
        external_inputs = []
        try:
            for agent_id in component:
                node = nodes[agent_id]
                model = self.model_registry.require_model(node.model_id)
                provider = self.model_registry.provider_for(node.model_id)
                agents.append(
                    {
                        "agent": node.to_dict(),
                        "model": model.to_dict(),
                        "provider": provider.to_dict(),
                        "is_output_agent": agent_id == output_agent_id,
                        "is_format_agent": (
                            format_output_agent and agent_id == output_agent_id
                        ),
                        "is_format_predecessor": (
                            agent_id in format_predecessor_ids
                        ),
                    }
                )
                for message in self._upstream(
                    agent_id,
                    plan,
                    outputs,
                    nodes=nodes,
                    graph_revision=0,
                    output_metadata=output_metadata,
                ):
                    external_inputs.append(
                        {
                            "source_agent_id": message.source_agent_id,
                            "target_agent_id": message.target_agent_id,
                            "message_type": message.message_type,
                            "artifact_type": message.artifact_type,
                            "artifact_version": (
                                message.artifact_version
                                if self.artifact_communication_profile
                                in _ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_PROFILES
                                else None
                            ),
                            "source_model_id": message.source_model_id,
                            "source_contract": message.source_contract,
                            "source_execution_mode": (
                                message.source_execution_mode
                            ),
                            "source_role_family": message.source_role_family,
                            "source_completion_condition": (
                                message.source_completion_condition
                            ),
                            "source_finish_reason": (
                                message.source_finish_reason
                            ),
                            "request_or_dependency": (
                                message.request_or_dependency
                            ),
                            "environment_revision": message.environment_revision,
                            "content": message.content,
                            "tool_receipts": [
                                _semantic_tool_receipt_value(receipt)
                                for receipt in message.tool_receipts
                            ],
                        }
                    )
            identity = {
                "identity_version": (
                    _COMPONENT_EXECUTION_INPUT_IDENTITY_VERSION
                ),
                "problem": problem,
                "semantic_protocol": self.semantic_protocol,
                "artifact_communication_profile": (
                    self.artifact_communication_profile
                ),
                "artifact_quality_profile": self.artifact_quality_profile,
                "communication_condition": communication_condition.value,
                "component_agent_ids": list(component),
                "agents": agents,
                "internal_relations": sorted(
                    (
                        relation.to_dict()
                        for relation in plan.relations
                        if relation.source_id in component_ids
                        and relation.target_id in component_ids
                    ),
                    key=lambda relation: (
                        str(relation["source_id"]),
                        str(relation["target_id"]),
                    ),
                ),
                "external_inputs": sorted(
                    external_inputs,
                    key=lambda message: (
                        str(message["target_agent_id"]),
                        str(message["source_agent_id"]),
                    ),
                ),
            }
            return json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (KeyError, TypeError, ValueError):
            # An opaque/non-JSON receipt cannot establish input equality.  The
            # ordinary execution path remains authoritative in that case.
            return None

    @staticmethod
    def _component_result_cacheable(
        component: Tuple[str, ...],
        block_outputs: Mapping[str, str],
        calls: Collection[AgentCallRecord],
        *,
        run_id: str,
        graph_revision: int,
    ) -> bool:
        """Return whether one completed component is safe to materialize."""

        if set(block_outputs) != set(component) or any(
            not isinstance(artifact, str) or not artifact.strip()
            for artifact in block_outputs.values()
        ):
            return False
        component_calls = tuple(
            call
            for call in calls
            if call.request.run_id == run_id
            and call.request.graph_revision == graph_revision
            and call.request.agent.id in component
        )
        if not component_calls:
            return False
        return all(
            call.response.text.strip()
            and str(call.response.metadata.get("finish_reason", "")).casefold()
            != "length"
            for call in component_calls
        )

    @staticmethod
    def _component_reuse_receipt(
        component: Tuple[str, ...],
        entry: ComponentExecutionCacheEntry,
        *,
        current_graph_revision: int,
    ) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "execution_reused": True,
                "input_identity_version": (
                    _COMPONENT_EXECUTION_INPUT_IDENTITY_VERSION
                ),
                "component_agent_ids": list(component),
                "source_graph_revision": entry.source_graph_revision,
                "current_graph_revision": current_graph_revision,
                "source_artifact_versions": {
                    agent_id: metadata.get("artifact_version")
                    for agent_id, metadata in entry.output_metadata.items()
                },
            }
        )

    async def execute(
        self,
        graph: AgentGraph,
        problem: str,
        *,
        run_id: Optional[str] = None,
        require_complete: bool = True,
        prior_outputs: Optional[Mapping[str, str]] = None,
        prior_output_metadata: Optional[
            Mapping[str, Mapping[str, object]]
        ] = None,
        prior_failure_metadata: Optional[
            Mapping[str, Mapping[str, object]]
        ] = None,
        dirty_agents: Optional[Collection[str]] = None,
        unavailable_model_ids: Optional[Collection[str]] = None,
        execution_cache: Optional[ComponentExecutionCache] = None,
        format_output_agent: bool = False,
        communication_condition: Union[
            CommunicationCondition, str
        ] = CommunicationCondition.NORMAL,
    ) -> AgentRuntimeResult:
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        if type(format_output_agent) is not bool:
            raise TypeError("format_output_agent must be bool")
        snapshot = graph.snapshot()
        execution_graph = AgentGraph.from_snapshot(snapshot)
        validation = execution_graph.validate(
            self.model_registry,
            require_complete=require_complete,
        )
        validation.raise_if_invalid()
        if require_complete and execution_graph.output_agent_id is None:
            raise AgentGraphValidationError(validation)

        resolved_run_id = run_id or uuid.uuid4().hex
        if not isinstance(resolved_run_id, str) or not resolved_run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        resolved_run_id = resolved_run_id.strip()
        resolved_condition = _communication_condition(communication_condition)
        plan = self._build_plan(execution_graph, validation.components)
        nodes = {node.id: node for node in execution_graph.nodes}
        if unavailable_model_ids is None:
            unavailable_models: set[str] = set()
        else:
            if any(
                not isinstance(model_id, str)
                for model_id in unavailable_model_ids
            ):
                raise TypeError("unavailable_model_ids must contain strings")
            unavailable_models = set(unavailable_model_ids)
        self.validate_execution_contracts(tuple(nodes.values()))
        self._validate_stateful_resource_ownership(nodes, plan)
        if format_output_agent and execution_graph.output_agent_id is not None:
            format_node = nodes[execution_graph.output_agent_id]
            if (format_node.role_family or "").casefold() != "format":
                raise AgentRuntimeError(
                    "format_output_agent requires the Output Agent to carry "
                    "role_family='format'"
                )
            format_mode = getattr(
                format_node.execution_mode,
                "value",
                format_node.execution_mode,
            )
            if format_mode != "reasoning" or format_node.allowed_tools:
                raise AgentRuntimeError(
                    "Format Agent must use reasoning execution without tools"
                )
        outputs: Dict[str, str] = {}
        output_metadata: Dict[str, Mapping[str, object]] = {}
        failure_metadata: Dict[str, Mapping[str, object]] = {}
        for agent_id, output in dict(prior_outputs or {}).items():
            if agent_id in nodes:
                if not isinstance(output, str):
                    raise TypeError("prior_outputs values must be strings")
                outputs[agent_id] = output
        for agent_id, metadata in dict(prior_output_metadata or {}).items():
            if agent_id in outputs:
                if not isinstance(metadata, Mapping):
                    raise TypeError(
                        "prior_output_metadata values must be mappings"
                    )
                output_metadata[agent_id] = MappingProxyType(dict(metadata))
        for agent_id, metadata in dict(prior_failure_metadata or {}).items():
            if agent_id not in nodes:
                continue
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    "prior_failure_metadata values must be mappings"
                )
            failure_metadata[agent_id] = MappingProxyType(dict(metadata))
        if dirty_agents is None:
            dirty_seeds = set(nodes)
        else:
            if any(not isinstance(agent_id, str) for agent_id in dirty_agents):
                raise TypeError("dirty_agents must contain strings")
            dirty_seeds = set(dirty_agents)
        # Missing upstream artifacts invalidate their complete dependent
        # closure just like an explicit Canvas edit.  Computing the closure
        # only before adding missing IDs could otherwise reuse a stale cached
        # downstream artifact.
        dirty_seeds.update(agent_id for agent_id in nodes if agent_id not in outputs)
        dirty = execution_graph.dirty_closure(dirty_seeds)
        # FlowSteer's executor cache is valid only for an unchanged input
        # identity.  A dirty Agent and every dependent successor therefore
        # lose their prior artifact before execution starts; otherwise a
        # failed recomputation could expose a stale value as current state.
        for agent_id in dirty:
            outputs.pop(agent_id, None)
            output_metadata.pop(agent_id, None)
        dirty_components = {
            plan.component_for[agent_id]
            for agent_id in dirty
            if agent_id in plan.component_for
        }
        deferred_components = self._semantic_input_deferred_components(
            execution_graph,
            nodes,
            plan,
            format_output_agent=format_output_agent,
        )
        unavailable_components = {
            component
            for component in plan.components
            if any(nodes[agent_id].model_id in unavailable_models for agent_id in component)
        }
        pending_unavailable_components = list(unavailable_components)
        while pending_unavailable_components:
            component = pending_unavailable_components.pop()
            for successor in plan.successors[component]:
                if successor in unavailable_components:
                    continue
                unavailable_components.add(successor)
                pending_unavailable_components.append(successor)
        deferred_components.update(unavailable_components)
        deferred_agent_ids = tuple(
            sorted(
                agent_id
                for component in deferred_components
                for agent_id in component
            )
        )
        calls: List[AgentCallRecord] = []
        cancelled_failure_records: List[AgentFailureRecord] = []
        execution_reuse_receipts: List[Mapping[str, object]] = []
        completion_order: List[Tuple[str, ...]] = []
        executed_agents: Set[str] = set()
        reused_agents: Set[str] = set()
        indegree = dict(plan.indegree)
        ready = sorted(
            component
            for component, degree in indegree.items()
            if degree == 0 and component not in deferred_components
        )
        active: Dict[
            "asyncio.Task[Tuple[Dict[str, str], bool]]",
            Tuple[str, ...],
        ] = {}

        async def execute_or_reuse(
            component: Tuple[str, ...],
        ) -> Tuple[Dict[str, str], bool]:
            if component not in dirty_components and all(
                agent_id in outputs for agent_id in component
            ):
                return ({agent_id: outputs[agent_id] for agent_id in component}, True)
            cache_key: Optional[str] = None
            cache_eligible = (
                execution_cache is not None
                and self._component_cache_eligible(
                    component,
                    nodes,
                    failure_metadata,
                )
            )
            if cache_eligible:
                cache_key = self._component_execution_input_identity(
                    component,
                    nodes,
                    plan,
                    outputs,
                    output_metadata,
                    problem.strip(),
                    output_agent_id=execution_graph.output_agent_id,
                    format_output_agent=format_output_agent,
                    communication_condition=resolved_condition,
                )
                cached = (
                    None
                    if cache_key is None or execution_cache is None
                    else execution_cache.get(cache_key)
                )
                if (
                    isinstance(cached, ComponentExecutionCacheEntry)
                    and set(cached.outputs) == set(component)
                ):
                    for agent_id in component:
                        output_metadata[agent_id] = MappingProxyType(
                            dict(cached.output_metadata[agent_id])
                        )
                    execution_reuse_receipts.append(
                        self._component_reuse_receipt(
                            component,
                            cached,
                            current_graph_revision=snapshot.revision,
                        )
                    )
                    return (dict(cached.outputs), True)

            block_outputs = await self._execute_block(
                component,
                nodes,
                plan,
                outputs,
                problem.strip(),
                resolved_run_id,
                snapshot.revision,
                calls,
                output_metadata,
                cancelled_failure_records,
                failure_metadata,
                output_agent_id=execution_graph.output_agent_id,
                format_output_agent=format_output_agent,
                communication_condition=resolved_condition,
            )
            if (
                cache_eligible
                and cache_key is not None
                and execution_cache is not None
                and self._component_result_cacheable(
                    component,
                    block_outputs,
                    calls,
                    run_id=resolved_run_id,
                    graph_revision=snapshot.revision,
                )
            ):
                execution_cache[cache_key] = ComponentExecutionCacheEntry(
                    source_graph_revision=snapshot.revision,
                    outputs=block_outputs,
                    output_metadata={
                        agent_id: output_metadata[agent_id]
                        for agent_id in component
                    },
                )
            return (block_outputs, False)

        def start_ready() -> None:
            while ready:
                component = ready.pop(0)
                task = asyncio.create_task(execute_or_reuse(component))
                active[task] = component

        start_ready()
        try:
            while active:
                done, _ = await asyncio.wait(
                    tuple(active), return_when=asyncio.FIRST_COMPLETED
                )
                completed: List[
                    Tuple[Tuple[str, ...], Dict[str, str], bool]
                ] = []
                failures: List[Tuple[Tuple[str, ...], BaseException]] = []
                for task in sorted(done, key=lambda item: active[item]):
                    component = active.pop(task)
                    try:
                        block_outputs, reused = task.result()
                        completed.append((component, block_outputs, reused))
                    except BaseException as exc:
                        failures.append((component, exc))

                # Preserve every block that had already completed in this
                # scheduler tick before handling a sibling failure.  Pending
                # blocks are still cancelled by the fail-fast boundary.
                for component, block_outputs, reused in completed:
                    outputs.update(block_outputs)
                    completion_order.append(component)
                    if reused:
                        reused_agents.update(component)
                    else:
                        executed_agents.update(component)
                if failures:
                    await _cancel_and_wait(list(active))  # type: ignore[arg-type]
                    failure_component, failure = failures[0]
                    if isinstance(failure, asyncio.CancelledError):
                        raise failure
                    failed_component_ids = {
                        agent_id
                        for component, _ in failures
                        for agent_id in component
                    }
                    blocked_agent_ids = tuple(
                        sorted(
                            execution_graph.dirty_closure(failed_component_ids)
                            - failed_component_ids
                        )
                    )
                    partial_result = AgentRuntimeResult(
                        run_id=resolved_run_id,
                        graph_revision=snapshot.revision,
                        output_agent_id=execution_graph.output_agent_id,
                        final_answer=None,
                        outputs=outputs,
                        calls=tuple(
                            sorted(calls, key=lambda record: record.request.request_id)
                        ),
                        block_completion_order=tuple(completion_order),
                        executed_agent_ids=tuple(sorted(executed_agents)),
                        reused_agent_ids=tuple(sorted(reused_agents)),
                        deferred_agent_ids=deferred_agent_ids,
                        communication_condition=resolved_condition,
                        output_metadata=output_metadata,
                        execution_reuse_receipts=tuple(
                            execution_reuse_receipts
                        ),
                    )
                    pending_agent_ids = tuple(
                        sorted(set(nodes) - set(partial_result.outputs))
                    )
                    failure_records = tuple(
                        sorted(
                            (
                                record
                                for _, item in failures
                                if isinstance(item, AgentRuntimeError)
                                for record in item.failure_records
                            ),
                            key=lambda record: (
                                record.request_id,
                                record.error_type,
                            ),
                        )
                    ) + tuple(
                        sorted(
                            cancelled_failure_records,
                            key=lambda record: (
                                record.request_id,
                                record.error_type,
                            ),
                        )
                    )
                    if isinstance(failure, AgentRuntimeError):
                        raise AgentRuntimeError(
                            str(failure),
                            failure_records=failure_records,
                            partial_result=partial_result,
                            blocked_agent_ids=(
                                *failure.blocked_agent_ids,
                                *blocked_agent_ids,
                            ),
                            pending_agent_ids=pending_agent_ids,
                        ) from failure
                    raise AgentRuntimeError(
                        f"AgentGraph block {failure_component!r} execution failed: "
                        f"{failure}",
                        partial_result=partial_result,
                        blocked_agent_ids=blocked_agent_ids,
                        pending_agent_ids=pending_agent_ids,
                    ) from failure

                newly_ready: Set[Tuple[str, ...]] = set()
                for component, _, _ in completed:
                    for successor in plan.successors[component]:
                        indegree[successor] -= 1
                        if (
                            indegree[successor] == 0
                            and successor not in deferred_components
                        ):
                            newly_ready.add(successor)
                ready.extend(sorted(newly_ready))
                ready.sort()
                start_ready()
        except BaseException:
            await _cancel_and_wait(list(active))  # type: ignore[arg-type]
            raise

        output_id = execution_graph.output_agent_id
        final_answer = outputs.get(output_id) if output_id is not None else None
        if require_complete and final_answer is None:
            raise AgentRuntimeError("complete AgentGraph produced no Output Agent artifact")
        return AgentRuntimeResult(
            run_id=resolved_run_id,
            graph_revision=snapshot.revision,
            output_agent_id=output_id,
            final_answer=final_answer,
            outputs=outputs,
            calls=tuple(sorted(calls, key=lambda record: record.request.request_id)),
            block_completion_order=tuple(completion_order),
            executed_agent_ids=tuple(sorted(executed_agents)),
            reused_agent_ids=tuple(sorted(reused_agents)),
            deferred_agent_ids=deferred_agent_ids,
            communication_condition=resolved_condition,
            output_metadata=output_metadata,
            execution_reuse_receipts=tuple(execution_reuse_receipts),
        )

    def _semantic_input_deferred_components(
        self,
        graph: AgentGraph,
        nodes: Mapping[str, AgentNode],
        plan: _ExecutionPlan,
        *,
        format_output_agent: bool,
    ) -> Set[Tuple[str, ...]]:
        """Defer semantic consumers until their declared input is routable.

        FlowSteer's incomplete Canvas is executable after every accepted edit,
        while SkillFlow invokes a bounded Agent only with its current public
        input.  Under the unified evidence-lineage protocol, a Reasoner without
        direct Evidence Retriever ingress has no grounded input.  A disconnected
        Verifier has no Reasoner candidate to check and a disconnected or
        unselected Formatter has no verified answer to serialize.  Defer those
        components and their descendants without deleting nodes or artifacts;
        the next relation or Output edit makes them schedulable under the same
        Canvas revision semantics.
        """

        if self.semantic_protocol == "qa_verified_answer_lineage_v2":
            seeds: Set[Tuple[str, ...]] = set()
            output_agent_id = graph.output_agent_id
            for agent_id, node in nodes.items():
                role = (node.role_family or "").casefold()
                predecessors = graph.directed_predecessors(agent_id)
                has_routed_upstream = bool(predecessors)
                if role == "reasoner" and not any(
                    (
                        nodes[predecessor_id].role_family or ""
                    ).casefold()
                    == "evidence_retriever"
                    for predecessor_id in predecessors
                ):
                    # The unified factual-QA Reasoner is a semantic consumer,
                    # not the initial retrieval root.  A declared Retriever
                    # dependency lets the ordinary execution plan run the
                    # Retriever first; its SkillFlow completion validator then
                    # admits only entity/relation-aligned evidence with a
                    # successful read receipt.  Until that dependency exists,
                    # preserve the partial Canvas and defer the Reasoner rather
                    # than executing it on an ungrounded question-only input.
                    seeds.add(plan.component_for[agent_id])
                elif role == "verifier" and not has_routed_upstream:
                    seeds.add(plan.component_for[agent_id])
                elif role == "format" and (
                    not format_output_agent
                    or agent_id != output_agent_id
                    or not has_routed_upstream
                ):
                    seeds.add(plan.component_for[agent_id])
            deferred = set(seeds)
            frontier = list(seeds)
            while frontier:
                component = frontier.pop()
                for successor in plan.successors[component]:
                    if successor not in deferred:
                        deferred.add(successor)
                        frontier.append(successor)
            return deferred

        if self.semantic_protocol != "hotpotqa_verified_answer_slot_v1":
            return set()
        seeds: Set[Tuple[str, ...]] = set()
        output_agent_id = graph.output_agent_id
        for agent_id, node in nodes.items():
            role = (node.role_family or "").casefold()
            predecessors = graph.directed_predecessors(agent_id)
            if role == "verifier":
                if (
                    len(predecessors) != 1
                    or (
                        nodes[predecessors[0]].role_family or ""
                    ).casefold()
                    != "reasoner"
                ):
                    seeds.add(plan.component_for[agent_id])
            elif role == "format":
                if (
                    not format_output_agent
                    or agent_id != output_agent_id
                    or len(predecessors) != 1
                    or (
                        nodes[predecessors[0]].role_family or ""
                    ).casefold()
                    != "verifier"
                ):
                    seeds.add(plan.component_for[agent_id])
        deferred = set(seeds)
        frontier = list(seeds)
        while frontier:
            component = frontier.pop()
            for successor in plan.successors[component]:
                if successor not in deferred:
                    deferred.add(successor)
                    frontier.append(successor)
        return deferred

    def validate_execution_contracts(
        self,
        nodes: Tuple[AgentNode, ...],
    ) -> None:
        """Validate Director-selected execution semantics before scheduling.

        The Canvas also calls this boundary before committing a graph edit so
        an execution-mode/tool mismatch becomes immediate edit feedback rather
        than a persisted graph revision that can only fail at runtime.
        """

        for node in nodes:
            mode_value = getattr(node.execution_mode, "value", node.execution_mode)
            if mode_value not in self.execution_adapters:
                raise AgentRuntimeError(
                    f"agent {node.id!r} requires unregistered execution adapter "
                    f"{mode_value!r}"
                )
            if (
                self._execution_profile_allowlist is not None
                and (mode_value, tuple(node.allowed_tools))
                not in self._execution_profile_allowlist
            ):
                raise AgentRuntimeError(
                    f"agent {node.id!r} execution profile "
                    f"{(mode_value, tuple(node.allowed_tools))!r} is outside "
                    "the active execution_profile_allowlist"
                )
            if not self.model_supports_execution_profile(
                node.model_id,
                mode_value,
                node.allowed_tools,
            ):
                raise AgentRuntimeError(
                    f"agent {node.id!r} model {node.model_id!r} does not admit "
                    f"execution profile {(mode_value, tuple(node.allowed_tools))!r}"
                )
            if not node.allowed_tools:
                continue
            if mode_value == "reasoning":
                raise AgentRuntimeError(
                    f"reasoning agent {node.id!r} cannot declare allowed_tools; "
                    "set execution_mode='react' or 'coding' for registered tools, "
                    "or clear allowed_tools"
                )
            if self.tool_registry is None:
                raise AgentRuntimeError(
                    f"agent {node.id!r} declares tools but no ToolRegistry is configured"
                )
            for tool_id in node.allowed_tools:
                if tool_id not in self.tool_registry.resource_ids:
                    raise AgentRuntimeError(
                        f"agent {node.id!r} references unknown tool {tool_id!r}"
                    )
                try:
                    capability = self.tool_registry.require_capability(tool_id)
                except KeyError as exc:
                    raise AgentRuntimeError(
                        f"agent {node.id!r} tool {tool_id!r} has no capability metadata"
                    ) from exc
                if not capability.availability:
                    raise AgentRuntimeError(
                        f"agent {node.id!r} tool {tool_id!r} is unavailable"
                    )
                if self.dataset_id is not None and not capability.supports_dataset(
                    self.dataset_id
                ):
                    raise AgentRuntimeError(
                        f"agent {node.id!r} tool {tool_id!r} is outside dataset scope "
                        f"{self.dataset_id!r}"
                    )

    def _validate_stateful_resource_ownership(
        self,
        nodes: Mapping[str, AgentNode],
        plan: _ExecutionPlan,
    ) -> None:
        """Preserve single-episode/single-worktree semantics in a graph run.

        SkillFlow binds one environment episode to one bounded Agent and one
        SWE repository worktree to one coding episode.  FlowSteer's
        reciprocal block executes both members concurrently and then executes
        both again for revision, so a stateful resource cannot legally be
        owned by that block or by multiple graph nodes.  Stateless retrieval
        and process-isolated computation remain unrestricted.
        """

        if self.tool_registry is None:
            return
        exclusive_side_effects = {
            "environment_state_transition",
            "repository_read_write_and_test_process",
        }
        owners: Dict[str, List[str]] = {}
        for node in nodes.values():
            for tool_id in node.allowed_tools:
                capability = self.tool_registry.require_capability(tool_id)
                if capability.side_effect in exclusive_side_effects:
                    owners.setdefault(tool_id, []).append(node.id)

        for tool_id, agent_ids in sorted(owners.items()):
            if len(agent_ids) != 1:
                raise AgentRuntimeError(
                    f"stateful tool {tool_id!r} requires one graph Agent owner; "
                    f"found {len(agent_ids)}"
                )
            owner_id = agent_ids[0]
            component = plan.component_for[owner_id]
            if len(component) != 1:
                raise AgentRuntimeError(
                    f"stateful tool {tool_id!r} cannot execute inside a "
                    "reciprocal Agent block"
                )

    def _build_plan(
        self,
        graph: AgentGraph,
        components: Tuple[Tuple[str, ...], ...],
    ) -> _ExecutionPlan:
        component_for = {
            agent_id: component for component in components for agent_id in component
        }
        successor_sets: Dict[Tuple[str, ...], Set[Tuple[str, ...]]] = {
            component: set() for component in components
        }
        indegree = {component: 0 for component in components}
        for relation in graph.relations:
            for source_id, target_id in relation.directed_edges():
                source = component_for[source_id]
                target = component_for[target_id]
                if source == target or target in successor_sets[source]:
                    continue
                successor_sets[source].add(target)
                indegree[target] += 1
        successors = {
            component: tuple(sorted(targets))
            for component, targets in successor_sets.items()
        }
        return _ExecutionPlan(
            components=components,
            component_for=MappingProxyType(component_for),
            successors=MappingProxyType(successors),
            indegree=MappingProxyType(indegree),
            relations=graph.relations,
        )

    async def _execute_block(
        self,
        component: Tuple[str, ...],
        nodes: Mapping[str, AgentNode],
        plan: _ExecutionPlan,
        outputs: Mapping[str, str],
        problem: str,
        run_id: str,
        graph_revision: int,
        calls: List[AgentCallRecord],
        output_metadata: Dict[str, Mapping[str, object]],
        cancelled_failure_records: List[AgentFailureRecord],
        failure_metadata: Mapping[str, Mapping[str, object]],
        *,
        output_agent_id: Optional[str],
        format_output_agent: bool,
        communication_condition: CommunicationCondition,
    ) -> Dict[str, str]:
        format_predecessor_ids = {
            source_id
            for relation in plan.relations
            for source_id, target_id in relation.directed_edges()
            if format_output_agent
            and output_agent_id is not None
            and target_id == output_agent_id
        }
        if len(component) == 1:
            agent_id = component[0]
            request = self._request(
                agent=nodes[agent_id],
                phase=ExecutionPhase.SINGLE,
                upstream=self._upstream(
                    agent_id,
                    plan,
                    outputs,
                    nodes=nodes,
                    graph_revision=graph_revision,
                    output_metadata=output_metadata,
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=agent_id in format_predecessor_ids,
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(agent_id),
            )
            response = await self._invoke(
                request,
                calls,
                cancelled_failure_records,
            )
            output_metadata[agent_id] = self._response_output_metadata(
                request,
                response,
            )
            return {agent_id: response.text}

        if len(component) != 2:
            raise AgentRuntimeError(f"unsupported reciprocal block size: {len(component)}")
        left_id, right_id = component
        left_upstream = self._upstream(
            left_id,
            plan,
            outputs,
            nodes=nodes,
            graph_revision=graph_revision,
            output_metadata=output_metadata,
        )
        right_upstream = self._upstream(
            right_id,
            plan,
            outputs,
            nodes=nodes,
            graph_revision=graph_revision,
            output_metadata=output_metadata,
        )
        semantic_roles = {
            agent_id: (nodes[agent_id].role_family or "").casefold()
            for agent_id in component
        }
        if (
            self.semantic_protocol == "qa_verified_answer_lineage_v2"
            and set(semantic_roles.values())
            == {"evidence_retriever", "reasoner"}
        ):
            # Thin adaptation of the existing causally ordered
            # Reasoner--Verifier reciprocal block.  FlowSteer's generic
            # reciprocal block has parallel draft/revision barriers, but the
            # Reasoner must not run before receipt-grounded evidence exists.
            # Preserve the Director-selected reciprocal topology and four-call
            # budget while enforcing the semantic dependency order:
            # Retriever DRAFT -> Reasoner DRAFT -> Retriever REVISION ->
            # Reasoner REVISION.
            retriever_id = next(
                agent_id
                for agent_id, role in semantic_roles.items()
                if role == "evidence_retriever"
            )
            reasoner_id = next(
                agent_id
                for agent_id, role in semantic_roles.items()
                if role == "reasoner"
            )
            upstream_by_id = {
                left_id: left_upstream,
                right_id: right_upstream,
            }
            retriever_draft_request = self._request(
                agent=nodes[retriever_id],
                phase=ExecutionPhase.DRAFT,
                upstream=upstream_by_id[retriever_id],
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    retriever_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(retriever_id),
            )
            retriever_draft = await self._invoke(
                retriever_draft_request,
                calls,
                cancelled_failure_records,
            )
            retriever_draft_metadata = self._response_output_metadata(
                retriever_draft_request,
                retriever_draft,
            )
            retriever_draft_message = UpstreamMessage(
                retriever_id,
                reasoner_id,
                retriever_draft.text,
                message_type="evidence",
                graph_revision=graph_revision,
                request_or_dependency=nodes[reasoner_id].contract,
                artifact_type=getattr(
                    nodes[retriever_id], "artifact_type", "text"
                ),
                environment_revision=_environment_revision_from_metadata(
                    retriever_draft_metadata
                ),
                tool_receipts=_tool_receipts_from_metadata(
                    retriever_draft_metadata
                ),
                input_artifact_provenance=(
                    _input_artifact_provenance_from_metadata(
                        retriever_draft_metadata
                    )
                ),
                artifact_version=retriever_draft_request.request_id,
                **_producer_context(
                    nodes[retriever_id],
                    retriever_draft_metadata,
                    artifact_communication_profile=(
                        self.artifact_communication_profile
                    ),
                ),
            )
            reasoner_draft_request = self._request(
                agent=nodes[reasoner_id],
                phase=ExecutionPhase.DRAFT,
                upstream=tuple(
                    sorted(
                        (
                            *upstream_by_id[reasoner_id],
                            retriever_draft_message,
                        ),
                        key=lambda item: (
                            item.source_agent_id,
                            item.target_agent_id,
                        ),
                    )
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    reasoner_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(reasoner_id),
            )
            reasoner_draft = await self._invoke(
                reasoner_draft_request,
                calls,
                cancelled_failure_records,
            )
            reasoner_draft_metadata = self._response_output_metadata(
                reasoner_draft_request,
                reasoner_draft,
            )
            retriever_revision_request = self._request(
                agent=nodes[retriever_id],
                phase=ExecutionPhase.REVISION,
                upstream=upstream_by_id[retriever_id],
                own_draft=retriever_draft.text,
                peer_draft=UpstreamMessage(
                    reasoner_id,
                    retriever_id,
                    reasoner_draft.text,
                    message_type="candidate",
                    graph_revision=graph_revision,
                    request_or_dependency=nodes[retriever_id].contract,
                    artifact_type=getattr(
                        nodes[reasoner_id], "artifact_type", "text"
                    ),
                    environment_revision=_environment_revision_from_metadata(
                        reasoner_draft_metadata
                    ),
                    tool_receipts=_tool_receipts_from_metadata(
                        reasoner_draft_metadata
                    ),
                    input_artifact_provenance=(
                        _input_artifact_provenance_from_metadata(
                            reasoner_draft_metadata
                        )
                    ),
                    artifact_version=reasoner_draft_request.request_id,
                    **_producer_context(
                        nodes[reasoner_id],
                        reasoner_draft_metadata,
                        artifact_communication_profile=(
                            self.artifact_communication_profile
                        ),
                    ),
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    retriever_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(retriever_id),
            )
            retriever_revision = await self._invoke(
                retriever_revision_request,
                calls,
                cancelled_failure_records,
            )
            retriever_revision_metadata = self._response_output_metadata(
                retriever_revision_request,
                retriever_revision,
            )
            reasoner_revision_request = self._request(
                agent=nodes[reasoner_id],
                phase=ExecutionPhase.REVISION,
                upstream=upstream_by_id[reasoner_id],
                own_draft=reasoner_draft.text,
                peer_draft=UpstreamMessage(
                    retriever_id,
                    reasoner_id,
                    retriever_revision.text,
                    message_type="evidence",
                    graph_revision=graph_revision,
                    request_or_dependency=nodes[reasoner_id].contract,
                    artifact_type=getattr(
                        nodes[retriever_id], "artifact_type", "text"
                    ),
                    environment_revision=_environment_revision_from_metadata(
                        retriever_revision_metadata
                    ),
                    tool_receipts=_tool_receipts_from_metadata(
                        retriever_revision_metadata
                    ),
                    input_artifact_provenance=(
                        _input_artifact_provenance_from_metadata(
                            retriever_revision_metadata
                        )
                    ),
                    artifact_version=retriever_revision_request.request_id,
                    **_producer_context(
                        nodes[retriever_id],
                        retriever_revision_metadata,
                        artifact_communication_profile=(
                            self.artifact_communication_profile
                        ),
                    ),
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    reasoner_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(reasoner_id),
            )
            reasoner_revision = await self._invoke(
                reasoner_revision_request,
                calls,
                cancelled_failure_records,
            )
            reasoner_revision_metadata = self._response_output_metadata(
                reasoner_revision_request,
                reasoner_revision,
            )
            output_metadata[retriever_id] = retriever_revision_metadata
            output_metadata[reasoner_id] = reasoner_revision_metadata
            return {
                retriever_id: retriever_revision.text,
                reasoner_id: reasoner_revision.text,
            }
        if (
            self.semantic_protocol
            in {
                "hotpotqa_verified_answer_slot_v1",
                "qa_verified_answer_lineage_v2",
            }
            and set(semantic_roles.values()) == {"reasoner", "verifier"}
        ):
            # FlowSteer's generic reciprocal block uses two parallel barriers.
            # A typed QA Verifier, however, must validate the current Reasoner
            # artifact rather than the Reasoner's previous-phase draft. Keep
            # the Director-selected reciprocal topology and the same four calls,
            # but make this semantic execution unit causally ordered.
            reasoner_id = next(
                agent_id
                for agent_id, role in semantic_roles.items()
                if role == "reasoner"
            )
            verifier_id = next(
                agent_id
                for agent_id, role in semantic_roles.items()
                if role == "verifier"
            )
            upstream_by_id = {
                left_id: left_upstream,
                right_id: right_upstream,
            }
            reasoner_draft_request = self._request(
                agent=nodes[reasoner_id],
                phase=ExecutionPhase.DRAFT,
                upstream=upstream_by_id[reasoner_id],
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    reasoner_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(reasoner_id),
            )
            reasoner_draft = await self._invoke(
                reasoner_draft_request,
                calls,
                cancelled_failure_records,
            )
            reasoner_draft_metadata = self._response_output_metadata(
                reasoner_draft_request,
                reasoner_draft,
            )
            reasoner_draft_message = UpstreamMessage(
                reasoner_id,
                verifier_id,
                reasoner_draft.text,
                message_type="candidate",
                graph_revision=graph_revision,
                request_or_dependency=nodes[verifier_id].contract,
                artifact_type=getattr(
                    nodes[reasoner_id], "artifact_type", "text"
                ),
                environment_revision=_environment_revision_from_metadata(
                    reasoner_draft_metadata
                ),
                tool_receipts=_tool_receipts_from_metadata(
                    reasoner_draft_metadata
                ),
                input_artifact_provenance=(
                    _input_artifact_provenance_from_metadata(
                        reasoner_draft_metadata
                    )
                ),
                artifact_version=reasoner_draft_request.request_id,
                **_producer_context(
                    nodes[reasoner_id],
                    reasoner_draft_metadata,
                    artifact_communication_profile=(
                        self.artifact_communication_profile
                    ),
                ),
            )
            verifier_initial_request = self._request(
                agent=nodes[verifier_id],
                phase=ExecutionPhase.SINGLE,
                upstream=tuple(
                    sorted(
                        (
                            *upstream_by_id[verifier_id],
                            reasoner_draft_message,
                        ),
                        key=lambda item: (
                            item.source_agent_id,
                            item.target_agent_id,
                        ),
                    )
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    verifier_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(verifier_id),
            )
            verifier_initial = await self._invoke(
                verifier_initial_request,
                calls,
                cancelled_failure_records,
            )
            verifier_initial_metadata = self._response_output_metadata(
                verifier_initial_request,
                verifier_initial,
            )
            reasoner_revision_request = self._request(
                agent=nodes[reasoner_id],
                phase=ExecutionPhase.REVISION,
                upstream=upstream_by_id[reasoner_id],
                own_draft=reasoner_draft.text,
                peer_draft=UpstreamMessage(
                    verifier_id,
                    reasoner_id,
                    verifier_initial.text,
                    message_type="candidate",
                    graph_revision=graph_revision,
                    request_or_dependency=nodes[reasoner_id].contract,
                    artifact_type=getattr(
                        nodes[verifier_id], "artifact_type", "text"
                    ),
                    environment_revision=_environment_revision_from_metadata(
                        verifier_initial_metadata
                    ),
                    tool_receipts=_tool_receipts_from_metadata(
                        verifier_initial_metadata
                    ),
                    input_artifact_provenance=(
                        _input_artifact_provenance_from_metadata(
                            verifier_initial_metadata
                        )
                    ),
                    artifact_version=verifier_initial_request.request_id,
                    **_producer_context(
                        nodes[verifier_id],
                        verifier_initial_metadata,
                        artifact_communication_profile=(
                            self.artifact_communication_profile
                        ),
                    ),
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    reasoner_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(reasoner_id),
            )
            reasoner_revision = await self._invoke(
                reasoner_revision_request,
                calls,
                cancelled_failure_records,
            )
            reasoner_revision_metadata = self._response_output_metadata(
                reasoner_revision_request,
                reasoner_revision,
            )
            verifier_revision_request = self._request(
                agent=nodes[verifier_id],
                phase=ExecutionPhase.REVISION,
                upstream=upstream_by_id[verifier_id],
                own_draft=verifier_initial.text,
                peer_draft=UpstreamMessage(
                    reasoner_id,
                    verifier_id,
                    reasoner_revision.text,
                    message_type="candidate",
                    graph_revision=graph_revision,
                    request_or_dependency=nodes[verifier_id].contract,
                    artifact_type=getattr(
                        nodes[reasoner_id], "artifact_type", "text"
                    ),
                    environment_revision=_environment_revision_from_metadata(
                        reasoner_revision_metadata
                    ),
                    tool_receipts=_tool_receipts_from_metadata(
                        reasoner_revision_metadata
                    ),
                    input_artifact_provenance=(
                        _input_artifact_provenance_from_metadata(
                            reasoner_revision_metadata
                        )
                    ),
                    artifact_version=reasoner_revision_request.request_id,
                    **_producer_context(
                        nodes[reasoner_id],
                        reasoner_revision_metadata,
                        artifact_communication_profile=(
                            self.artifact_communication_profile
                        ),
                    ),
                ),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                format_output_agent=format_output_agent,
                is_format_predecessor=(
                    verifier_id in format_predecessor_ids
                ),
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(verifier_id),
            )
            verifier_revision = await self._invoke(
                verifier_revision_request,
                calls,
                cancelled_failure_records,
            )
            verifier_revision_metadata = self._response_output_metadata(
                verifier_revision_request,
                verifier_revision,
            )
            output_metadata[reasoner_id] = reasoner_revision_metadata
            output_metadata[verifier_id] = verifier_revision_metadata
            return {
                reasoner_id: reasoner_revision.text,
                verifier_id: verifier_revision.text,
            }
        left_draft_request = self._request(
            agent=nodes[left_id],
            phase=ExecutionPhase.DRAFT,
            upstream=left_upstream,
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            is_format_predecessor=left_id in format_predecessor_ids,
            communication_condition=communication_condition,
            continuation_metadata=failure_metadata.get(left_id),
        )
        right_draft_request = self._request(
            agent=nodes[right_id],
            phase=ExecutionPhase.DRAFT,
            upstream=right_upstream,
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            is_format_predecessor=right_id in format_predecessor_ids,
            communication_condition=communication_condition,
            continuation_metadata=failure_metadata.get(right_id),
        )
        left_draft, right_draft = await _gather_pair(
            self._invoke(left_draft_request, calls, cancelled_failure_records),
            self._invoke(right_draft_request, calls, cancelled_failure_records),
        )
        # Reuse FlowSteer's standard response binding before either draft is
        # routed to its peer.  This keeps the peer message tied to the exact
        # public artifacts consumed by the producing Agent instead of losing
        # that lineage at the reciprocal-component boundary.
        left_draft_metadata = self._response_output_metadata(
            left_draft_request,
            left_draft,
        )
        right_draft_metadata = self._response_output_metadata(
            right_draft_request,
            right_draft,
        )

        left_revision_request = self._request(
            agent=nodes[left_id],
            phase=ExecutionPhase.REVISION,
            upstream=left_upstream,
            own_draft=left_draft.text,
            peer_draft=UpstreamMessage(
                right_id,
                left_id,
                right_draft.text,
                message_type="candidate",
                graph_revision=graph_revision,
                request_or_dependency=nodes[left_id].contract,
                artifact_type=getattr(nodes[right_id], "artifact_type", "text"),
                environment_revision=_environment_revision_from_metadata(
                    right_draft_metadata
                ),
                tool_receipts=_tool_receipts_from_metadata(
                    right_draft_metadata
                ),
                input_artifact_provenance=(
                    _input_artifact_provenance_from_metadata(
                        right_draft_metadata
                    )
                ),
                artifact_version=right_draft_request.request_id,
                **_producer_context(
                    nodes[right_id],
                    right_draft_metadata,
                    artifact_communication_profile=(
                        self.artifact_communication_profile
                    ),
                ),
            ),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            is_format_predecessor=left_id in format_predecessor_ids,
            communication_condition=communication_condition,
            continuation_metadata=failure_metadata.get(left_id),
        )
        right_revision_request = self._request(
            agent=nodes[right_id],
            phase=ExecutionPhase.REVISION,
            upstream=right_upstream,
            own_draft=right_draft.text,
            peer_draft=UpstreamMessage(
                left_id,
                right_id,
                left_draft.text,
                message_type="candidate",
                graph_revision=graph_revision,
                request_or_dependency=nodes[right_id].contract,
                artifact_type=getattr(nodes[left_id], "artifact_type", "text"),
                environment_revision=_environment_revision_from_metadata(
                    left_draft_metadata
                ),
                tool_receipts=_tool_receipts_from_metadata(
                    left_draft_metadata
                ),
                input_artifact_provenance=(
                    _input_artifact_provenance_from_metadata(
                        left_draft_metadata
                    )
                ),
                artifact_version=left_draft_request.request_id,
                **_producer_context(
                    nodes[left_id],
                    left_draft_metadata,
                    artifact_communication_profile=(
                        self.artifact_communication_profile
                    ),
                ),
            ),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            is_format_predecessor=right_id in format_predecessor_ids,
            communication_condition=communication_condition,
            continuation_metadata=failure_metadata.get(right_id),
        )
        left_revision, right_revision = await _gather_pair(
            self._invoke(left_revision_request, calls, cancelled_failure_records),
            self._invoke(right_revision_request, calls, cancelled_failure_records),
        )
        output_metadata[left_id] = self._response_output_metadata(
            left_revision_request,
            left_revision,
        )
        output_metadata[right_id] = self._response_output_metadata(
            right_revision_request,
            right_revision,
        )
        return {left_id: left_revision.text, right_id: right_revision.text}

    def _upstream(
        self,
        target_agent_id: str,
        plan: _ExecutionPlan,
        outputs: Mapping[str, str],
        *,
        nodes: Mapping[str, AgentNode],
        graph_revision: int,
        output_metadata: Mapping[str, Mapping[str, object]],
    ) -> Tuple[UpstreamMessage, ...]:
        target_component = plan.component_for[target_agent_id]
        messages = []
        for relation in plan.relations:
            for source_id, target_id in relation.directed_edges():
                if target_id != target_agent_id:
                    continue
                if plan.component_for[source_id] == target_component:
                    continue
                if source_id not in outputs:
                    raise AgentRuntimeError(
                        f"upstream output {source_id!r} was not ready for {target_agent_id!r}"
                    )
                # Keep the canonical routed message intact. Diagnostic masking is
                # applied only when the provider prompt is rendered so receipts
                # retain both the true upstream and what the model actually saw.
                messages.append(
                    UpstreamMessage(
                        source_id,
                        target_id,
                        outputs[source_id],
                        message_type="artifact",
                        graph_revision=graph_revision,
                        request_or_dependency=nodes[target_id].contract,
                        artifact_type=getattr(
                            nodes[source_id], "artifact_type", "text"
                        ),
                        environment_revision=_environment_revision_from_metadata(
                            output_metadata.get(source_id, {})
                        ),
                        tool_receipts=_tool_receipts_from_metadata(
                            output_metadata.get(source_id, {})
                        ),
                        input_artifact_provenance=(
                            _input_artifact_provenance_from_metadata(
                                output_metadata.get(source_id, {})
                            )
                        ),
                        artifact_version=(
                            str(
                                output_metadata.get(source_id, {}).get(
                                    "artifact_version"
                                )
                            )
                            if isinstance(
                                output_metadata.get(source_id, {}).get(
                                    "artifact_version"
                                ),
                                str,
                            )
                            and str(
                                output_metadata.get(source_id, {}).get(
                                    "artifact_version"
                                )
                            ).strip()
                            else None
                        ),
                        **_producer_context(
                            nodes[source_id],
                            output_metadata.get(source_id, {}),
                            artifact_communication_profile=(
                                self.artifact_communication_profile
                            ),
                        ),
                    )
                )
        return tuple(
            sorted(messages, key=lambda item: (item.source_agent_id, item.target_agent_id))
        )

    def _request(
        self,
        *,
        agent: AgentNode,
        phase: ExecutionPhase,
        upstream: Tuple[UpstreamMessage, ...],
        problem: str,
        run_id: str,
        graph_revision: int,
        own_draft: Optional[str] = None,
        peer_draft: Optional[UpstreamMessage] = None,
        output_agent_id: Optional[str],
        format_output_agent: bool,
        is_format_predecessor: bool,
        communication_condition: CommunicationCondition,
        continuation_metadata: Optional[Mapping[str, object]] = None,
    ) -> AgentRequest:
        model = self.model_registry.require_model(agent.model_id)
        provider = self.model_registry.provider_for(agent.model_id)
        request_id = f"{run_id}:{graph_revision}:{agent.id}:{phase.value}"
        continuation = dict(continuation_metadata or {})
        continuation_phase = continuation.get("execution_phase")
        if (
            continuation_phase is not None
            and continuation_phase != phase.value
        ):
            # A reciprocal block invokes an Agent in distinct draft/revision
            # phases.  Public ReAct state belongs to the phase that failed and
            # must not be replayed into a different communication contract.
            continuation = {}
        current_input_artifact_versions = {
            message.source_agent_id: message.artifact_version
            for message in (
                *upstream,
                *((peer_draft,) if peer_draft is not None else ()),
            )
            if message.artifact_version is not None
        }
        raw_continuation_input_versions = continuation.get(
            "input_artifact_versions"
        )
        continuation_input_changed = (
            isinstance(raw_continuation_input_versions, Mapping)
            and {
                str(agent_id): str(version)
                for agent_id, version in raw_continuation_input_versions.items()
                if isinstance(agent_id, str) and isinstance(version, str)
            }
            != current_input_artifact_versions
        )
        raw_action_history = continuation.get("react_trace", ())
        raw_tool_receipts = continuation.get("tool_receipts", ())
        raw_continuation_source_agent_id = continuation.get(
            "continuation_source_agent_id"
        )
        # PROJECT_NECESSARY_ADAPTATION: SkillFlow resumes a bounded Agent from
        # its public Action--Observation history. A FlowSteer relation edit can
        # give that Agent a new upstream artifact, so the prior rejection state
        # is no longer conditioned on the current input. Preserve all Tool
        # receipts for provenance, but restart action selection on the new
        # artifact versions instead of replaying a stale terminal diagnosis.
        # Tool receipts are part of the same bounded execution state because
        # the ReAct adapter counts them against its Tool budget.  They remain
        # persisted in the earlier trajectory receipt, but cannot remain active
        # after the dependency-version key changes.  Fresh predecessor receipts
        # are still delivered through ``upstream`` with their artifact version.
        action_history = (
            ()
            if continuation_input_changed
            else (
                tuple(
                    item
                    for item in raw_action_history
                    if isinstance(item, Mapping)
                )
                if isinstance(raw_action_history, (list, tuple))
                else ()
            )
        )
        prior_tool_receipts = (
            ()
            if continuation_input_changed
            else (
                tuple(
                    item
                    for item in raw_tool_receipts
                    if isinstance(item, Mapping)
                )
                if isinstance(raw_tool_receipts, (list, tuple))
                else ()
            )
        )
        continuation_source_agent_id = (
            None
            if continuation_input_changed
            else (
                raw_continuation_source_agent_id.strip()
                if isinstance(raw_continuation_source_agent_id, str)
                and raw_continuation_source_agent_id.strip()
                else None
            )
        )
        return AgentRequest(
            request_id=request_id,
            run_id=run_id,
            graph_revision=graph_revision,
            problem=problem,
            agent=agent,
            model=model,
            provider=provider,
            phase=phase,
            is_output_agent=agent.id == output_agent_id,
            is_format_agent=(
                format_output_agent and agent.id == output_agent_id
            ),
            is_format_predecessor=is_format_predecessor,
            communication_condition=communication_condition,
            upstream=upstream,
            own_draft=own_draft,
            peer_draft=peer_draft,
            semantic_protocol=self.semantic_protocol,
            artifact_communication_profile=(
                self.artifact_communication_profile
            ),
            action_history=action_history,
            prior_tool_receipts=prior_tool_receipts,
            continuation_source_agent_id=continuation_source_agent_id,
        )

    def _response_output_metadata(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> Mapping[str, object]:
        """Bind one artifact to the exact public inputs consumed to produce it."""

        metadata = dict(response.metadata)
        metadata["artifact_version"] = request.request_id
        # FlowSteer's SET_OUTPUT edit is pointer-only. Persist the invocation
        # role so Canvas can distinguish a user-facing Artifact produced under
        # the Output protocol from an intermediate Artifact before promoting an
        # existing node. This records execution provenance; it does not assign
        # an Agent role or alter the generated content.
        metadata["generated_as_output_agent"] = request.is_output_agent
        metadata["generated_as_format_agent"] = request.is_format_agent
        raw_tool_receipts = metadata.get("tool_receipts")
        if isinstance(raw_tool_receipts, (list, tuple)):
            # Directly reuse FlowSteer's public-receipt normalization: remove
            # exact duplicates without changing error/result semantics.
            distinct_tool_receipts: list[dict[str, object]] = []
            for receipt in raw_tool_receipts:
                if not isinstance(receipt, Mapping):
                    continue
                serialized_receipt = dict(receipt)
                if serialized_receipt not in distinct_tool_receipts:
                    distinct_tool_receipts.append(serialized_receipt)
            metadata["tool_receipts"] = distinct_tool_receipts
        inputs = list(request.upstream)
        if request.peer_draft is not None:
            inputs.append(request.peer_draft)
        input_artifact_versions: dict[str, str] = {}
        input_artifact_provenance: list[dict[str, object]] = []
        distinct_inputs: list[UpstreamMessage] = []
        for message in inputs:
            if message.artifact_version is not None:
                previous = input_artifact_versions.get(message.source_agent_id)
                if (
                    previous is not None
                    and previous != message.artifact_version
                ):
                    raise AgentRuntimeError(
                        "one Agent request consumed conflicting artifact versions "
                        f"from {message.source_agent_id!r}"
                    )
                input_artifact_versions[message.source_agent_id] = (
                    message.artifact_version
                )
            serialized_message = message.to_dict()
            if serialized_message in input_artifact_provenance:
                continue
            input_artifact_provenance.append(serialized_message)
            distinct_inputs.append(message)
        metadata["input_artifact_versions"] = input_artifact_versions
        metadata["input_artifact_provenance"] = input_artifact_provenance
        if self.semantic_protocol in {
            "hotpotqa_verified_answer_slot_v1",
            "qa_verified_answer_lineage_v2",
        }:
            # A Reasoner artifact may cite evidence read by an upstream
            # Retriever as well as evidence read during its own bounded ReAct
            # execution.  Preserve that public receipt lineage across every
            # routed edge.  The provider boundary still projects only receipts
            # referenced by the current artifact, while the Runtime and Env
            # retain the complete provenance needed by the downstream
            # Verifier.  This extends FlowSteer's routed artifact envelope with
            # SkillFlow's public Tool receipts; it does not add a workflow or
            # retrieval policy to the Director prompt.
            lineage_tool_receipts: list[dict[str, object]] = []
            for message in distinct_inputs:
                for receipt in message.tool_receipts:
                    serialized = dict(receipt)
                    if serialized not in lineage_tool_receipts:
                        lineage_tool_receipts.append(serialized)
            for receipt in _tool_receipts_from_metadata(metadata):
                serialized = dict(receipt)
                if serialized not in lineage_tool_receipts:
                    lineage_tool_receipts.append(serialized)
            metadata["tool_receipts"] = lineage_tool_receipts
        return MappingProxyType(metadata)

    async def _invoke(
        self,
        request: AgentRequest,
        calls: List[AgentCallRecord],
        cancelled_failure_records: List[AgentFailureRecord],
    ) -> AgentResponse:
        def input_artifact_metadata() -> Mapping[str, object]:
            inputs = list(request.upstream)
            if request.peer_draft is not None:
                inputs.append(request.peer_draft)
            return MappingProxyType(
                {
                    "input_artifact_versions": {
                        message.source_agent_id: message.artifact_version
                        for message in inputs
                        if message.artifact_version is not None
                    },
                    "input_artifact_provenance": [
                        message.to_dict() for message in inputs
                    ],
                }
            )

        def cancelled_adapter_metadata() -> Mapping[str, object]:
            mode_value = getattr(
                request.agent.execution_mode,
                "value",
                request.agent.execution_mode,
            )
            adapter = self.execution_adapters.get(mode_value)
            take_metadata = getattr(
                adapter,
                "take_cancelled_failure_metadata",
                None,
            )
            if not callable(take_metadata):
                return MappingProxyType({})
            value = take_metadata(request.request_id)
            return (
                MappingProxyType(dict(value))
                if isinstance(value, Mapping)
                else MappingProxyType({})
            )

        async def call_gateway() -> GatewayResponse:
            mode_value = getattr(
                request.agent.execution_mode,
                "value",
                request.agent.execution_mode,
            )
            adapter = self.execution_adapters.get(mode_value)
            if adapter is None:
                raise AgentRuntimeError(
                    f"no execution adapter registered for {mode_value!r}"
                )
            provider_semaphore = self._provider_semaphores.get(request.provider.provider_id)
            if provider_semaphore is None:
                return await adapter.execute(request)
            async with provider_semaphore:
                return await adapter.execute(request)

        try:
            if self._global_semaphore is None:
                invocation = call_gateway()
                raw_response = (
                    await invocation
                    if self.timeout_seconds is None
                    else await asyncio.wait_for(invocation, timeout=self.timeout_seconds)
                )
            else:
                async with self._global_semaphore:
                    invocation = call_gateway()
                    raw_response = (
                        await invocation
                        if self.timeout_seconds is None
                        else await asyncio.wait_for(invocation, timeout=self.timeout_seconds)
                    )
        except asyncio.CancelledError as exc:
            metadata = MappingProxyType(
                {
                    **dict(cancelled_adapter_metadata()),
                    **dict(_public_failure_metadata(exc)),
                    **dict(input_artifact_metadata()),
                    **(
                        {
                            "continuation_source_agent_id": (
                                request.continuation_source_agent_id
                            )
                        }
                        if request.continuation_source_agent_id is not None
                        else {}
                    ),
                }
            )
            if metadata:
                cancelled_failure_records.append(
                    AgentFailureRecord(
                        request_id=request.request_id,
                        agent_id=request.agent.id,
                        phase=request.phase,
                        graph_revision=request.graph_revision,
                        error_type=type(exc).__name__,
                        message=(
                            "Agent invocation was cancelled after partial public "
                            "execution"
                        ),
                        metadata=metadata,
                    )
                )
            raise
        except Exception as exc:
            adapter_cancellation_metadata = cancelled_adapter_metadata()
            nested_records = (
                exc.failure_records
                if isinstance(exc, AgentRuntimeError) and exc.failure_records
                else (
                    AgentFailureRecord(
                        request_id=request.request_id,
                        agent_id=request.agent.id,
                        phase=request.phase,
                        graph_revision=request.graph_revision,
                        error_type=type(exc).__name__,
                        message=" ".join(str(exc).split()),
                        metadata=MappingProxyType(
                            {
                                **dict(adapter_cancellation_metadata),
                                **dict(_public_failure_metadata(exc)),
                                **dict(input_artifact_metadata()),
                                **(
                                    {
                                        "continuation_source_agent_id": (
                                            request.continuation_source_agent_id
                                        )
                                    }
                                    if request.continuation_source_agent_id
                                    is not None
                                    else {}
                                ),
                            }
                        ),
                    ),
                )
            )
            raise AgentRuntimeError(
                f"gateway failed for agent {request.agent.id!r} during "
                f"{request.phase.value}: {exc}",
                failure_records=nested_records,
                pending_agent_ids=(request.agent.id,),
            ) from exc
        response = (
            raw_response
            if isinstance(raw_response, AgentResponse)
            else AgentResponse(raw_response)
        )
        artifact_quality_receipt: Mapping[str, object] | None = None
        if self.artifact_quality_profile in {
            ARTIFACT_QUALITY_PUBLIC_TEXT_V1,
            ARTIFACT_QUALITY_PUBLIC_TEXT_V2,
        }:
            artifact_quality_receipt = _public_text_quality_receipt(
                response.text,
                response.metadata,
                profile=self.artifact_quality_profile,
            )
            response = AgentResponse(
                response.text,
                {
                    **dict(response.metadata),
                    "artifact_quality_receipt": dict(
                        artifact_quality_receipt
                    ),
                },
            )
        # SkillFlow records the sampled provider turn before treating an empty
        # completion as a failed execution boundary.  Preserve that receipt,
        # but never publish whitespace-only content as a semantic Artifact:
        # doing so defers the error to the first downstream UpstreamMessage and
        # incorrectly attributes the failure to the consumer rather than this
        # producer.
        calls.append(AgentCallRecord(request=request, response=response))
        if not response.text.strip():
            failure_metadata = MappingProxyType(
                {
                    **dict(response.metadata),
                    **dict(input_artifact_metadata()),
                    "public_error_code": "completion_artifact_empty",
                    "artifact_complete": False,
                    "response_text_characters": len(response.text.strip()),
                }
            )
            record = AgentFailureRecord(
                request_id=request.request_id,
                agent_id=request.agent.id,
                phase=request.phase,
                graph_revision=request.graph_revision,
                error_type="CompletionArtifactEmpty",
                message="Agent produced no non-empty completion Artifact",
                metadata=failure_metadata,
            )
            raise AgentRuntimeError(
                f"agent {request.agent.id!r} produced no non-empty "
                "completion Artifact",
                failure_records=(record,),
                pending_agent_ids=(request.agent.id,),
            )
        if (
            artifact_quality_receipt is not None
            and artifact_quality_receipt.get("status") == "invalid"
        ):
            error_codes = tuple(
                str(item)
                for item in artifact_quality_receipt.get("error_codes", ())
                if isinstance(item, str)
            )
            failure_metadata = MappingProxyType(
                {
                    **dict(response.metadata),
                    **dict(input_artifact_metadata()),
                    "public_error_code": "completion_artifact_quality_invalid",
                    "artifact_complete": False,
                }
            )
            record = AgentFailureRecord(
                request_id=request.request_id,
                agent_id=request.agent.id,
                phase=request.phase,
                graph_revision=request.graph_revision,
                error_type="CompletionArtifactQualityError",
                message=(
                    "Agent completion Artifact failed public text quality "
                    f"admission: {', '.join(error_codes)}"
                ),
                metadata=failure_metadata,
            )
            raise AgentRuntimeError(
                f"agent {request.agent.id!r} produced an invalid completion "
                "Artifact",
                failure_records=(record,),
                pending_agent_ids=(request.agent.id,),
            )
        return response


__all__ = [
    "ARTIFACT_QUALITY_NONE",
    "ARTIFACT_QUALITY_PUBLIC_TEXT_V1",
    "ARTIFACT_QUALITY_PUBLIC_TEXT_V2",
    "AgentCallRecord",
    "AgentFailureRecord",
    "AgentGateway",
    "AgentExecutionAdapter",
    "AgentRequest",
    "AgentResponse",
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentRuntimeResult",
    "CommunicationCondition",
    "CommunicationEnvelope",
    "ComponentExecutionCache",
    "ComponentExecutionCacheEntry",
    "ExecutionPhase",
    "GatewayResponse",
    "ReasoningExecutionAdapter",
    "UpstreamMessage",
]
