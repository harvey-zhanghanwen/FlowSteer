"""Asynchronous, finite AgentGraph execution over a fake-friendly gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Awaitable, Collection, Dict, List, Mapping, Optional, Protocol, Set, Tuple, Union
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


_NON_TERMINAL_PARTIAL_ARTIFACT = "non_terminal_partial"


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
        if not isinstance(self.tool_receipts, tuple) or any(
            not isinstance(item, Mapping) for item in self.tool_receipts
        ):
            raise TypeError("tool_receipts must be a tuple of mappings")
        object.__setattr__(
            self,
            "tool_receipts",
            tuple(MappingProxyType(dict(item)) for item in self.tool_receipts),
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
            "request_or_dependency": self.request_or_dependency,
            "dependency": self.request_or_dependency,
            "tool_receipts": [dict(item) for item in self.tool_receipts],
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
    require_exact_answer_tag: bool = False
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL
    upstream: Tuple[UpstreamMessage, ...] = ()
    own_draft: Optional[str] = None
    peer_draft: Optional[UpstreamMessage] = None
    semantic_protocol: str = "none"
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
        if type(self.require_exact_answer_tag) is not bool:
            raise TypeError("require_exact_answer_tag must be bool")
        if self.is_format_agent and not self.is_output_agent:
            raise ValueError("Format Agent must be the Output Agent")
        if self.is_format_agent and self.is_format_predecessor:
            raise ValueError("Format Agent cannot be its own predecessor")
        if self.semantic_protocol not in {
            "none",
            "hotpotqa_verified_answer_slot_v1",
            "hotpotqa_semantic_lineage_v2",
            "hotpotqa_role_conditional_capabilities_v1",
            "qa_verified_answer_lineage_v2",
        }:
            raise ValueError("unsupported AgentRequest semantic_protocol")
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


def _environment_revision_from_metadata(
    metadata: Mapping[str, object],
) -> Optional[int]:
    raw = metadata.get("environment_revision")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return None
    return raw


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
) -> Tuple[
    Union[AgentResponse, BaseException],
    Union[AgentResponse, BaseException],
]:
    """Settle a reciprocal barrier without discarding a completed peer.

    SkillFlow commits each completed Action--Observation step before a later
    bounded failure is reported.  Apply the same boundary to FlowSteer's
    finite reciprocal exchange: retain a response that completed before its
    peer failed, while preserving fail-fast cancellation for a peer that is
    still running.
    """

    tasks = [asyncio.create_task(left), asyncio.create_task(right)]
    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        if any(
            task.cancelled() or task.exception() is not None
            for task in done
        ):
            await _cancel_and_wait(list(pending))  # type: ignore[arg-type]
        elif pending:
            await asyncio.gather(*pending)
    except BaseException:
        await _cancel_and_wait(tasks)  # type: ignore[arg-type]
        raise

    results: List[Union[AgentResponse, BaseException]] = []
    for task in tasks:
        if task.cancelled():
            results.append(asyncio.CancelledError())
            continue
        failure = task.exception()
        results.append(failure if failure is not None else task.result())
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
            "hotpotqa_semantic_lineage_v2",
            "hotpotqa_role_conditional_capabilities_v1",
            "qa_verified_answer_lineage_v2",
        }:
            raise ValueError("unsupported AgentRuntime semantic_protocol")
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

    def registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return the executable mode/Tool profiles of this Runtime.

        SkillFlow binds a bounded executor to the resources registered for the
        current task, while FlowSteer's Canvas exposes only edits that the
        current executor can run.  Keep those two boundaries correlated here:
        reasoning is Tool-free, ReAct may complete without a Tool or use one
        currently available dataset-compatible resource, and coding is
        exposed only when its adapter is actually registered.  The profile is
        a Runtime capability receipt, not an Agent role or topology template.
        """

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
                if self.dataset_id is not None and not capability.supports_dataset(
                    self.dataset_id
                ):
                    continue
                profiles.append((mode_value, (tool_id,)))
        return tuple(profiles)

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
        format_output_agent: bool = False,
        require_exact_answer_tag: bool = False,
        communication_condition: Union[
            CommunicationCondition, str
        ] = CommunicationCondition.NORMAL,
    ) -> AgentRuntimeResult:
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        if type(format_output_agent) is not bool:
            raise TypeError("format_output_agent must be bool")
        if type(require_exact_answer_tag) is not bool:
            raise TypeError("require_exact_answer_tag must be bool")
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
            format_modes = (
                {"reasoning", "react"}
                if self.semantic_protocol
                == "hotpotqa_role_conditional_capabilities_v1"
                else {"reasoning"}
            )
            if format_mode not in format_modes or format_node.allowed_tools:
                raise AgentRuntimeError(
                    (
                        "Format Agent must use a registered Tool-free reasoning "
                        "or ReAct formatting execution profile"
                        if self.semantic_protocol
                        == "hotpotqa_role_conditional_capabilities_v1"
                        else "Format Agent must use reasoning execution without tools"
                    )
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
        # A successful peer in an incomplete reciprocal block is retained for
        # Canvas feedback and recovery evidence, but it is not a completed
        # block output.  Force a bounded recomputation instead of letting a
        # later topology edit reuse the draft/revision receipt as terminal.
        dirty_seeds.update(
            agent_id
            for agent_id, metadata in output_metadata.items()
            if metadata.get("artifact_status")
            == _NON_TERMINAL_PARTIAL_ARTIFACT
        )
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
        deferred_agent_ids = tuple(
            sorted(
                agent_id
                for component in deferred_components
                for agent_id in component
            )
        )
        calls: List[AgentCallRecord] = []
        cancelled_failure_records: List[AgentFailureRecord] = []
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
            return (
                await self._execute_block(
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
                    require_exact_answer_tag=require_exact_answer_tag,
                    communication_condition=resolved_condition,
                ),
                False,
            )

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
                    # A reciprocal block may have committed one peer response
                    # before the other peer failed.  Preserve that measured
                    # non-terminal artifact in the scheduler-level partial
                    # result just as completed sibling blocks are preserved.
                    for _, item in failures:
                        if not isinstance(item, AgentRuntimeError):
                            continue
                        nested_partial = item.partial_result
                        if nested_partial is None:
                            continue
                        outputs.update(nested_partial.outputs)
                        output_metadata.update(nested_partial.output_metadata)
                        executed_agents.update(
                            nested_partial.executed_agent_ids
                        )
                        reused_agents.update(nested_partial.reused_agent_ids)
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
                    )
                    pending_agent_ids = tuple(
                        sorted(
                            (set(nodes) - set(partial_result.outputs))
                            | {
                                agent_id
                                for agent_id, metadata in (
                                    partial_result.output_metadata.items()
                                )
                                if metadata.get("artifact_status")
                                == _NON_TERMINAL_PARTIAL_ARTIFACT
                            }
                        )
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
        input. For the legacy QA protocols, a disconnected Verifier has no
        Reasoner candidate to check and a disconnected or unselected Formatter
        has no verified answer to serialize. The flexible HotpotQA protocol
        instead asks only whether an actual upstream artifact route exists; it
        does not infer a required producer role, role count, or direct semantic
        chain. Defer unavailable consumers and their descendants without
        deleting nodes or artifacts; the next relation or Output edit makes
        them schedulable under the same Canvas revision semantics.
        """

        if self.semantic_protocol in {
            "hotpotqa_semantic_lineage_v2",
            "hotpotqa_role_conditional_capabilities_v1",
        }:
            seeds: Set[Tuple[str, ...]] = set()
            output_agent_id = graph.output_agent_id
            for agent_id, node in nodes.items():
                role = (node.role_family or "").casefold()
                # A flexible semantic consumer is schedulable whenever the
                # Canvas contains a route that will supply a public upstream
                # artifact. The producer's role and the number of producers
                # are deliberately not topology constraints.
                has_routed_upstream = bool(
                    graph.directed_predecessors(agent_id)
                )
                if role == "verifier" and not has_routed_upstream:
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

        if self.semantic_protocol not in {
            "hotpotqa_verified_answer_slot_v1",
            "qa_verified_answer_lineage_v2",
        }:
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
            if (node.role_family or "").casefold() == "react":
                raise AgentRuntimeError(
                    f"agent {node.id!r} cannot use role_family='react'; "
                    "ReAct is an execution_mode, not an Agent role"
                )
            mode_value = getattr(node.execution_mode, "value", node.execution_mode)
            if mode_value not in self.execution_adapters:
                raise AgentRuntimeError(
                    f"agent {node.id!r} requires unregistered execution adapter "
                    f"{mode_value!r}"
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

    @staticmethod
    def _reciprocal_responses_or_raise(
        results: Tuple[
            Union[AgentResponse, BaseException],
            Union[AgentResponse, BaseException],
        ],
        *,
        agent_ids: Tuple[str, str],
        phase: ExecutionPhase,
        run_id: str,
        graph_revision: int,
        output_agent_id: Optional[str],
        calls: List[AgentCallRecord],
        communication_condition: CommunicationCondition,
    ) -> Tuple[AgentResponse, AgentResponse]:
        """Raise with successful peer receipts when one reciprocal side fails."""

        failures = tuple(
            (agent_id, result)
            for agent_id, result in zip(agent_ids, results)
            if isinstance(result, BaseException)
        )
        if not failures:
            left, right = results
            assert isinstance(left, AgentResponse)
            assert isinstance(right, AgentResponse)
            return left, right

        measured_failures = tuple(
            item
            for item in failures
            if not isinstance(item[1], asyncio.CancelledError)
        )
        if not measured_failures:
            raise failures[0][1]
        for _, failure in measured_failures:
            if not isinstance(failure, Exception):
                raise failure

        successful_responses = {
            agent_id: result
            for agent_id, result in zip(agent_ids, results)
            if isinstance(result, AgentResponse)
        }
        partial_metadata = {
            agent_id: MappingProxyType(
                {
                    **dict(response.metadata),
                    "execution_phase": phase.value,
                    "artifact_status": _NON_TERMINAL_PARTIAL_ARTIFACT,
                    "reciprocal_block_complete": False,
                }
            )
            for agent_id, response in successful_responses.items()
        }
        partial_result = AgentRuntimeResult(
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            # A peer response is evidence from an incomplete reciprocal
            # barrier.  It can inform repair, but cannot become FINISH.
            final_answer=None,
            outputs={
                agent_id: response.text
                for agent_id, response in successful_responses.items()
            },
            calls=tuple(
                sorted(calls, key=lambda record: record.request.request_id)
            ),
            block_completion_order=(),
            executed_agent_ids=tuple(sorted(successful_responses)),
            communication_condition=communication_condition,
            output_metadata=partial_metadata,
        )
        failure_records = tuple(
            record
            for _, failure in measured_failures
            if isinstance(failure, AgentRuntimeError)
            for record in failure.failure_records
        )
        failed_agent_ids = tuple(agent_id for agent_id, _ in failures)
        first_agent_id, first_failure = measured_failures[0]
        raise AgentRuntimeError(
            "reciprocal Agent block failed for "
            f"{first_agent_id!r} during {phase.value}: {first_failure}",
            failure_records=failure_records,
            partial_result=partial_result,
            pending_agent_ids=failed_agent_ids,
        ) from first_failure

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
        require_exact_answer_tag: bool,
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
                require_exact_answer_tag=require_exact_answer_tag,
                is_format_predecessor=agent_id in format_predecessor_ids,
                communication_condition=communication_condition,
                continuation_metadata=failure_metadata.get(agent_id),
            )
            response = await self._invoke(
                request,
                calls,
                cancelled_failure_records,
            )
            output_metadata[agent_id] = response.metadata
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
        left_draft_request = self._request(
            agent=nodes[left_id],
            phase=ExecutionPhase.DRAFT,
            upstream=left_upstream,
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            require_exact_answer_tag=require_exact_answer_tag,
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
            require_exact_answer_tag=require_exact_answer_tag,
            is_format_predecessor=right_id in format_predecessor_ids,
            communication_condition=communication_condition,
            continuation_metadata=failure_metadata.get(right_id),
        )
        left_draft, right_draft = self._reciprocal_responses_or_raise(
            await _gather_pair(
                self._invoke(
                    left_draft_request,
                    calls,
                    cancelled_failure_records,
                ),
                self._invoke(
                    right_draft_request,
                    calls,
                    cancelled_failure_records,
                ),
            ),
            agent_ids=(left_id, right_id),
            phase=ExecutionPhase.DRAFT,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            calls=calls,
            communication_condition=communication_condition,
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
                    right_draft.metadata
                ),
                tool_receipts=_tool_receipts_from_metadata(right_draft.metadata),
            ),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            require_exact_answer_tag=require_exact_answer_tag,
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
                    left_draft.metadata
                ),
                tool_receipts=_tool_receipts_from_metadata(left_draft.metadata),
            ),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            require_exact_answer_tag=require_exact_answer_tag,
            is_format_predecessor=right_id in format_predecessor_ids,
            communication_condition=communication_condition,
            continuation_metadata=failure_metadata.get(right_id),
        )
        left_revision, right_revision = self._reciprocal_responses_or_raise(
            await _gather_pair(
                self._invoke(
                    left_revision_request,
                    calls,
                    cancelled_failure_records,
                ),
                self._invoke(
                    right_revision_request,
                    calls,
                    cancelled_failure_records,
                ),
            ),
            agent_ids=(left_id, right_id),
            phase=ExecutionPhase.REVISION,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            calls=calls,
            communication_condition=communication_condition,
        )
        output_metadata[left_id] = left_revision.metadata
        output_metadata[right_id] = right_revision.metadata
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
        require_exact_answer_tag: bool,
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
        raw_action_history = continuation.get("react_trace", ())
        raw_tool_receipts = continuation.get("tool_receipts", ())
        raw_continuation_source_agent_id = continuation.get(
            "continuation_source_agent_id"
        )
        continuation_source_agent_id = (
            raw_continuation_source_agent_id.strip()
            if isinstance(raw_continuation_source_agent_id, str)
            and raw_continuation_source_agent_id.strip()
            else None
        )
        action_history = (
            tuple(item for item in raw_action_history if isinstance(item, Mapping))
            if isinstance(raw_action_history, (list, tuple))
            else ()
        )
        prior_tool_receipts = (
            tuple(item for item in raw_tool_receipts if isinstance(item, Mapping))
            if isinstance(raw_tool_receipts, (list, tuple))
            else ()
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
            require_exact_answer_tag=(
                require_exact_answer_tag and agent.id == output_agent_id
            ),
            communication_condition=communication_condition,
            upstream=upstream,
            own_draft=own_draft,
            peer_draft=peer_draft,
            semantic_protocol=self.semantic_protocol,
            action_history=action_history,
            prior_tool_receipts=prior_tool_receipts,
            continuation_source_agent_id=continuation_source_agent_id,
        )

    async def _invoke(
        self,
        request: AgentRequest,
        calls: List[AgentCallRecord],
        cancelled_failure_records: List[AgentFailureRecord],
    ) -> AgentResponse:
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
        response = raw_response if isinstance(raw_response, AgentResponse) else AgentResponse(raw_response)
        calls.append(AgentCallRecord(request=request, response=response))
        return response


__all__ = [
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
    "ExecutionPhase",
    "GatewayResponse",
    "ReasoningExecutionAdapter",
    "UpstreamMessage",
]
