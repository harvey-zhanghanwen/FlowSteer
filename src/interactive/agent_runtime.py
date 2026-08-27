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
            "artifact_version": self.artifact_version,
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
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL
    upstream: Tuple[UpstreamMessage, ...] = ()
    own_draft: Optional[str] = None
    peer_draft: Optional[UpstreamMessage] = None

    def __post_init__(self) -> None:
        if type(self.is_output_agent) is not bool:
            raise TypeError("is_output_agent must be bool")
        if type(self.is_format_agent) is not bool:
            raise TypeError("is_format_agent must be bool")
        if self.is_format_agent and not self.is_output_agent:
            raise ValueError("Format Agent must be the Output Agent")
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
    Tool calls.  The record is diagnostic state, not an upstream semantic
    artifact.
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
    """Copy only adapter-published public execution receipts.

    The local HotpotQA adapter currently publishes one immutable ``metadata``
    mapping, while the newer shared adapter exposes the same receipt fields as
    typed exception attributes.  Accepting both forms is the minimal
    compatibility boundary; no hidden reasoning state is inferred.
    """

    result: Dict[str, object] = {}
    published = getattr(exc, "metadata", None)
    if isinstance(published, Mapping):
        result.update(dict(published))
    for field_name in (
        "react_trace",
        "tool_receipts",
        "model_calls",
        "retry_receipts",
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
        """Return execution-mode/Tool pairs runnable by this Runtime.

        This is the shared FlowSteer Canvas capability boundary.  Each item is
        ``(execution_mode, allowed_tools)``; it describes registered executors
        and available task-scoped resources without introducing semantic role
        or workflow constraints.
        """

        profiles: List[Tuple[str, Tuple[str, ...]]] = []
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
        dirty_agents: Optional[Collection[str]] = None,
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
        self._validate_execution_contracts(tuple(nodes.values()))
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
        if dirty_agents is None:
            dirty = set(nodes)
        else:
            if any(not isinstance(agent_id, str) for agent_id in dirty_agents):
                raise TypeError("dirty_agents must contain strings")
            dirty = execution_graph.dirty_closure(dirty_agents)
        dirty.update(agent_id for agent_id in nodes if agent_id not in outputs)
        dirty_components = {
            plan.component_for[agent_id]
            for agent_id in dirty
            if agent_id in plan.component_for
        }
        calls: List[AgentCallRecord] = []
        completion_order: List[Tuple[str, ...]] = []
        executed_agents: Set[str] = set()
        reused_agents: Set[str] = set()
        indegree = dict(plan.indegree)
        ready = sorted(component for component, degree in indegree.items() if degree == 0)
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
                    output_agent_id=execution_graph.output_agent_id,
                    format_output_agent=format_output_agent,
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

                # Preserve every independent block that completed in this
                # scheduler tick before propagating a sibling failure.  This
                # is FlowSteer's partial-execution boundary: valid artifacts
                # and call receipts remain available for Canvas diagnosis.
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
                    failed_agent_ids = {
                        agent_id
                        for component, _ in failures
                        for agent_id in component
                    }
                    blocked_agent_ids = tuple(
                        sorted(
                            execution_graph.dirty_closure(failed_agent_ids)
                            - failed_agent_ids
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
                        communication_condition=resolved_condition,
                        output_metadata=output_metadata,
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
                        if indegree[successor] == 0:
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
            communication_condition=resolved_condition,
            output_metadata=output_metadata,
        )

    def _validate_execution_contracts(
        self,
        nodes: Tuple[AgentNode, ...],
    ) -> None:
        """Validate Director-selected execution semantics before scheduling."""

        for node in nodes:
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
                    f"reasoning agent {node.id!r} cannot declare allowed_tools"
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
        *,
        output_agent_id: Optional[str],
        format_output_agent: bool,
        communication_condition: CommunicationCondition,
    ) -> Dict[str, str]:
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
                communication_condition=communication_condition,
            )
            response = await self._invoke(request, calls)
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
        left_draft_request = self._request(
            agent=nodes[left_id],
            phase=ExecutionPhase.DRAFT,
            upstream=left_upstream,
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            communication_condition=communication_condition,
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
            communication_condition=communication_condition,
        )
        left_draft, right_draft = await _gather_pair(
            self._invoke(left_draft_request, calls),
            self._invoke(right_draft_request, calls),
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
                artifact_version=right_draft_request.request_id,
            ),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            communication_condition=communication_condition,
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
                artifact_version=left_draft_request.request_id,
            ),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            format_output_agent=format_output_agent,
            communication_condition=communication_condition,
        )
        left_revision, right_revision = await _gather_pair(
            self._invoke(left_revision_request, calls),
            self._invoke(right_revision_request, calls),
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
        communication_condition: CommunicationCondition,
    ) -> AgentRequest:
        model = self.model_registry.require_model(agent.model_id)
        provider = self.model_registry.provider_for(agent.model_id)
        request_id = f"{run_id}:{graph_revision}:{agent.id}:{phase.value}"
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
            communication_condition=communication_condition,
            upstream=upstream,
            own_draft=own_draft,
            peer_draft=peer_draft,
        )

    def _response_output_metadata(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> Mapping[str, object]:
        """Bind an artifact to its consumed inputs and QA-memory receipts.

        DIRECT_REUSE: this is the FlowSteer v63 artifact-version and
        ``input_artifact_provenance`` boundary.  When the registered resource
        is HotpotQA QA-memory, SkillFlow Tool receipts are carried across each
        explicit relation so non-chain AgentGraph topologies retain the exact
        search/read lineage seen by downstream Agents.
        """

        metadata = dict(response.metadata)
        metadata["artifact_version"] = request.request_id
        inputs = list(request.upstream)
        if request.peer_draft is not None:
            inputs.append(request.peer_draft)
        input_artifact_versions: dict[str, str] = {}
        input_artifact_provenance: list[dict[str, object]] = []
        for message in inputs:
            if message.artifact_version is not None:
                previous = input_artifact_versions.get(message.source_agent_id)
                if previous is not None and previous != message.artifact_version:
                    raise AgentRuntimeError(
                        "one Agent request consumed conflicting artifact versions "
                        f"from {message.source_agent_id!r}"
                    )
                input_artifact_versions[message.source_agent_id] = (
                    message.artifact_version
                )
            input_artifact_provenance.append(message.to_dict())
        metadata["input_artifact_versions"] = input_artifact_versions
        metadata["input_artifact_provenance"] = input_artifact_provenance

        propagates_qa_memory = bool(
            self.dataset_id is not None
            and self.dataset_id.casefold() == "hotpotqa"
            and self.tool_registry is not None
            and "hotpotqa.qa_memory" in self.tool_registry.resource_ids
        )
        if propagates_qa_memory:
            lineage_tool_receipts: list[dict[str, object]] = []
            for message in inputs:
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
    ) -> AgentResponse:
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
        except asyncio.CancelledError:
            raise
        except Exception as exc:
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
                        metadata=_public_failure_metadata(exc),
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
