"""Asynchronous, finite AgentGraph execution over a fake-friendly gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Awaitable, Dict, List, Mapping, Optional, Protocol, Set, Tuple, Union
import uuid

from .agent_graph import (
    AgentGraph,
    AgentGraphValidationError,
    AgentNode,
    AgentRelation,
)
from .model_registry import ModelRegistry, ModelSpec, ProviderSpec


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
    source_agent_id: str
    target_agent_id: str
    content: str


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
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL
    upstream: Tuple[UpstreamMessage, ...] = ()
    own_draft: Optional[str] = None
    peer_draft: Optional[UpstreamMessage] = None

    def __post_init__(self) -> None:
        if type(self.is_output_agent) is not bool:
            raise TypeError("is_output_agent must be bool")
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


@dataclass(frozen=True, slots=True)
class AgentCallRecord:
    request: AgentRequest
    response: AgentResponse


@dataclass(frozen=True, slots=True)
class AgentRuntimeResult:
    run_id: str
    graph_revision: int
    output_agent_id: str
    final_answer: str
    outputs: Mapping[str, str]
    calls: Tuple[AgentCallRecord, ...]
    block_completion_order: Tuple[Tuple[str, ...], ...]
    communication_condition: CommunicationCondition = CommunicationCondition.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(
            self,
            "communication_condition",
            _communication_condition(self.communication_condition),
        )


class AgentRuntimeError(RuntimeError):
    """Wraps a gateway or scheduler failure with AgentGraph context."""


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
    ) -> None:
        if max_concurrency is not None and (
            type(max_concurrency) is not int or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_registry = model_registry
        self.gateway = gateway
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

    async def execute(
        self,
        graph: AgentGraph,
        problem: str,
        *,
        run_id: Optional[str] = None,
        communication_condition: Union[
            CommunicationCondition, str
        ] = CommunicationCondition.NORMAL,
    ) -> AgentRuntimeResult:
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        snapshot = graph.snapshot()
        execution_graph = AgentGraph.from_snapshot(snapshot)
        validation = execution_graph.validate(self.model_registry, require_complete=True)
        validation.raise_if_invalid()
        if execution_graph.output_agent_id is None:  # narrowed by validation
            raise AgentGraphValidationError(validation)

        resolved_run_id = run_id or uuid.uuid4().hex
        if not isinstance(resolved_run_id, str) or not resolved_run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        resolved_run_id = resolved_run_id.strip()
        resolved_condition = _communication_condition(communication_condition)
        plan = self._build_plan(execution_graph, validation.components)
        nodes = {node.id: node for node in execution_graph.nodes}
        outputs: Dict[str, str] = {}
        calls: List[AgentCallRecord] = []
        completion_order: List[Tuple[str, ...]] = []
        indegree = dict(plan.indegree)
        ready = sorted(component for component, degree in indegree.items() if degree == 0)
        active: Dict["asyncio.Task[Dict[str, str]]", Tuple[str, ...]] = {}

        def start_ready() -> None:
            while ready:
                component = ready.pop(0)
                task = asyncio.create_task(
                    self._execute_block(
                        component,
                        nodes,
                        plan,
                        outputs,
                        problem.strip(),
                        resolved_run_id,
                        snapshot.revision,
                        calls,
                        output_agent_id=execution_graph.output_agent_id,
                        communication_condition=resolved_condition,
                    )
                )
                active[task] = component

        start_ready()
        try:
            while active:
                done, _ = await asyncio.wait(
                    tuple(active), return_when=asyncio.FIRST_COMPLETED
                )
                completed: List[Tuple[Tuple[str, ...], Dict[str, str]]] = []
                failure: Optional[BaseException] = None
                for task in sorted(done, key=lambda item: active[item]):
                    component = active.pop(task)
                    try:
                        completed.append((component, task.result()))
                    except BaseException as exc:
                        failure = exc
                        break
                if failure is not None:
                    await _cancel_and_wait(list(active))  # type: ignore[arg-type]
                    if isinstance(failure, asyncio.CancelledError):
                        raise failure
                    if isinstance(failure, AgentRuntimeError):
                        raise failure
                    raise AgentRuntimeError(f"AgentGraph block execution failed: {failure}") from failure

                for component, block_outputs in completed:
                    outputs.update(block_outputs)
                    completion_order.append(component)
                newly_ready: Set[Tuple[str, ...]] = set()
                for component, _ in completed:
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
        return AgentRuntimeResult(
            run_id=resolved_run_id,
            graph_revision=snapshot.revision,
            output_agent_id=output_id,
            final_answer=outputs[output_id],
            outputs=outputs,
            calls=tuple(sorted(calls, key=lambda record: record.request.request_id)),
            block_completion_order=tuple(completion_order),
            communication_condition=resolved_condition,
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
        *,
        output_agent_id: str,
        communication_condition: CommunicationCondition,
    ) -> Dict[str, str]:
        if len(component) == 1:
            agent_id = component[0]
            request = self._request(
                agent=nodes[agent_id],
                phase=ExecutionPhase.SINGLE,
                upstream=self._upstream(agent_id, plan, outputs),
                problem=problem,
                run_id=run_id,
                graph_revision=graph_revision,
                output_agent_id=output_agent_id,
                communication_condition=communication_condition,
            )
            response = await self._invoke(request, calls)
            return {agent_id: response.text}

        if len(component) != 2:
            raise AgentRuntimeError(f"unsupported reciprocal block size: {len(component)}")
        left_id, right_id = component
        left_upstream = self._upstream(left_id, plan, outputs)
        right_upstream = self._upstream(right_id, plan, outputs)
        left_draft_request = self._request(
            agent=nodes[left_id],
            phase=ExecutionPhase.DRAFT,
            upstream=left_upstream,
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
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
            peer_draft=UpstreamMessage(right_id, left_id, right_draft.text),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            communication_condition=communication_condition,
        )
        right_revision_request = self._request(
            agent=nodes[right_id],
            phase=ExecutionPhase.REVISION,
            upstream=right_upstream,
            own_draft=right_draft.text,
            peer_draft=UpstreamMessage(left_id, right_id, left_draft.text),
            problem=problem,
            run_id=run_id,
            graph_revision=graph_revision,
            output_agent_id=output_agent_id,
            communication_condition=communication_condition,
        )
        left_revision, right_revision = await _gather_pair(
            self._invoke(left_revision_request, calls),
            self._invoke(right_revision_request, calls),
        )
        return {left_id: left_revision.text, right_id: right_revision.text}

    def _upstream(
        self,
        target_agent_id: str,
        plan: _ExecutionPlan,
        outputs: Mapping[str, str],
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
                messages.append(UpstreamMessage(source_id, target_id, outputs[source_id]))
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
        output_agent_id: str,
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
            communication_condition=communication_condition,
            upstream=upstream,
            own_draft=own_draft,
            peer_draft=peer_draft,
        )

    async def _invoke(
        self,
        request: AgentRequest,
        calls: List[AgentCallRecord],
    ) -> AgentResponse:
        async def call_gateway() -> GatewayResponse:
            provider_semaphore = self._provider_semaphores.get(request.provider.provider_id)
            if provider_semaphore is None:
                return await self.gateway.generate(request)
            async with provider_semaphore:
                return await self.gateway.generate(request)

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
            raise AgentRuntimeError(
                f"gateway failed for agent {request.agent.id!r} during {request.phase.value}: {exc}"
            ) from exc
        response = raw_response if isinstance(raw_response, AgentResponse) else AgentResponse(raw_response)
        calls.append(AgentCallRecord(request=request, response=response))
        return response


__all__ = [
    "AgentCallRecord",
    "AgentGateway",
    "AgentRequest",
    "AgentResponse",
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentRuntimeResult",
    "CommunicationCondition",
    "ExecutionPhase",
    "GatewayResponse",
    "UpstreamMessage",
]
