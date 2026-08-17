"""Transactional progressive Canvas environment for AgentGraph actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Optional, Tuple, Union

from .agent_action_parser import (
    AgentAction,
    AgentActionParseError,
    AgentActionParser,
    AgentActionType,
)
from .agent_graph import (
    AgentGraph,
    AgentGraphSnapshot,
    AgentNode,
    GraphMutationError,
    GraphValidationIssue,
    GraphValidationResult,
)
from .agent_runtime import (
    AgentGateway,
    AgentRuntime,
    AgentRuntimeError,
    AgentRuntimeResult,
)
from .model_registry import ModelRegistry


class AgentWorkflowStateError(RuntimeError):
    """Raised for invalid environment construction or restoration."""


def _answer_protocol_state(answer: str) -> tuple[int, bool, bool]:
    """Return tag count, exact-single-wrapper state, and non-empty state."""

    opening_count = answer.count("<answer>")
    closing_count = answer.count("</answer>")
    match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", answer, flags=re.DOTALL)
    exact_single = opening_count == 1 and closing_count == 1 and match is not None
    non_empty = exact_single and bool(match.group(1).strip())
    return opening_count, exact_single, non_empty


@dataclass(frozen=True, slots=True)
class AgentWorkflowHistoryEntry:
    """Canonical adaptation of FlowSteer's per-step Canvas history entry."""

    turn_count: int
    accepted: bool
    done: bool
    action: Optional[AgentAction]
    revision: int
    feedback: str
    execution_reused: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_count": self.turn_count,
            "accepted": self.accepted,
            "done": self.done,
            "action": None if self.action is None else self.action.to_dict(),
            "revision": self.revision,
            "feedback": self.feedback,
            "execution_reused": self.execution_reused,
        }


@dataclass(frozen=True, slots=True)
class AgentWorkflowSnapshot:
    problem: str
    graph: AgentGraphSnapshot
    turn_count: int
    finished: bool
    last_feedback: str
    history: Tuple[AgentWorkflowHistoryEntry, ...] = ()

    @property
    def snapshot_id(self) -> str:
        payload = {
            "problem": self.problem,
            "graph_snapshot_id": self.graph.snapshot_id,
            "turn_count": self.turn_count,
            "finished": self.finished,
            "last_feedback": self.last_feedback,
            "history": [entry.to_dict() for entry in self.history],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentWorkflowStepResult:
    accepted: bool
    done: bool
    action: Optional[AgentAction]
    revision: int
    feedback: str
    snapshot: AgentWorkflowSnapshot
    validation_issues: Tuple[GraphValidationIssue, ...] = ()
    execution: Optional[AgentRuntimeResult] = None
    execution_reused: bool = False

    @property
    def success(self) -> bool:
        return self.accepted

    @property
    def final_answer(self) -> Optional[str]:
        return None if self.execution is None else self.execution.final_answer


class AgentWorkflowEnv:
    """Apply one atomic edit per turn and incrementally execute valid graph blocks."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        gateway: Optional[AgentGateway] = None,
        *,
        runtime: Optional[AgentRuntime] = None,
        problem: str = "",
        graph: Optional[AgentGraph] = None,
        execute_on_edit: bool = False,
        max_agents: Optional[int] = None,
        require_exact_answer_tag: bool = False,
        require_format_agent: bool = False,
    ) -> None:
        if runtime is None and gateway is None:
            raise AgentWorkflowStateError("gateway or runtime is required")
        if runtime is not None and runtime.model_registry is not model_registry:
            raise AgentWorkflowStateError("runtime and environment must share the model registry")
        if max_agents is not None and (
            isinstance(max_agents, bool) or not isinstance(max_agents, int) or max_agents < 1
        ):
            raise AgentWorkflowStateError("max_agents must be a positive integer or None")
        if type(require_exact_answer_tag) is not bool:
            raise AgentWorkflowStateError("require_exact_answer_tag must be bool")
        if type(require_format_agent) is not bool:
            raise AgentWorkflowStateError("require_format_agent must be bool")
        self.model_registry = model_registry
        self.runtime = runtime or AgentRuntime(model_registry, gateway)  # type: ignore[arg-type]
        self.execute_on_edit = execute_on_edit
        self.max_agents = max_agents
        self.require_exact_answer_tag = require_exact_answer_tag
        self.require_format_agent = require_format_agent
        self.parser = AgentActionParser()
        self._problem = problem.strip()
        self._graph = graph.fork() if graph is not None else AgentGraph()
        self._turn_count = 0
        self._finished = False
        self._last_feedback = ""
        self._history: list[AgentWorkflowHistoryEntry] = []
        self._progressive_execution: Optional[AgentRuntimeResult] = None
        self._progressive_execution_revision: Optional[int] = None
        self._progressive_outputs: dict[str, str] = {}
        self._validate_agent_limit(self._graph)
        partial = self._graph.validate(self.model_registry, require_complete=False)
        if not partial.valid:
            raise AgentWorkflowStateError(self._format_issues(partial))

    @property
    def problem(self) -> str:
        return self._problem

    @property
    def graph(self) -> AgentGraph:
        return self._graph

    @property
    def revision(self) -> int:
        return self._graph.revision

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def history(self) -> Tuple[AgentWorkflowHistoryEntry, ...]:
        return tuple(self._history)

    def reset(self, problem: str, graph: Optional[AgentGraph] = None) -> AgentWorkflowSnapshot:
        if not isinstance(problem, str) or not problem.strip():
            raise AgentWorkflowStateError("problem must be a non-empty string")
        candidate = graph.fork() if graph is not None else AgentGraph()
        self._validate_agent_limit(candidate)
        validation = candidate.validate(self.model_registry, require_complete=False)
        if not validation.valid:
            raise AgentWorkflowStateError(self._format_issues(validation))
        self._problem = problem.strip()
        self._graph = candidate
        self._turn_count = 0
        self._finished = False
        self._last_feedback = ""
        self._history.clear()
        self._clear_progressive_execution()
        return self.snapshot()

    def snapshot(self) -> AgentWorkflowSnapshot:
        return AgentWorkflowSnapshot(
            problem=self._problem,
            graph=self._graph.snapshot(),
            turn_count=self._turn_count,
            finished=self._finished,
            last_feedback=self._last_feedback,
            history=tuple(self._history),
        )

    def restore(self, snapshot: AgentWorkflowSnapshot) -> None:
        graph = AgentGraph.from_snapshot(snapshot.graph)
        self._validate_agent_limit(graph)
        validation = graph.validate(self.model_registry, require_complete=False)
        if not validation.valid:
            raise AgentWorkflowStateError(self._format_issues(validation))
        self._problem = snapshot.problem
        self._graph = graph
        self._turn_count = snapshot.turn_count
        self._finished = snapshot.finished
        self._last_feedback = snapshot.last_feedback
        self._history = list(snapshot.history)
        # Runtime results are deliberately not serialized in Canvas snapshots.
        # A restored environment must therefore execute its current graph once
        # before it can establish a revision-local progressive result again.
        self._clear_progressive_execution()

    def fork(self, snapshot: Optional[AgentWorkflowSnapshot] = None) -> "AgentWorkflowEnv":
        state = snapshot or self.snapshot()
        result = AgentWorkflowEnv(
            self.model_registry,
            runtime=self.runtime,
            problem=state.problem,
            graph=AgentGraph.from_snapshot(state.graph),
            execute_on_edit=self.execute_on_edit,
            max_agents=self.max_agents,
            require_exact_answer_tag=self.require_exact_answer_tag,
            require_format_agent=self.require_format_agent,
        )
        result._turn_count = state.turn_count
        result._finished = state.finished
        result._last_feedback = state.last_feedback
        result._history = list(state.history)
        return result

    async def step(self, action_or_response: Union[AgentAction, str]) -> AgentWorkflowStepResult:
        if self._finished:
            return self._reject(None, "workflow already finished")
        if not self._problem:
            return self._reject(None, "environment has no active problem")
        try:
            action = (
                self.parser.parse(action_or_response)
                if isinstance(action_or_response, str)
                else action_or_response
            )
        except AgentActionParseError as exc:
            return self._reject(None, f"invalid action: {exc}")
        if not isinstance(action, AgentAction):
            return self._reject(None, "action must be AgentAction or JSON text")

        self._turn_count += 1
        if action.action_type is AgentActionType.FINISH:
            validation = self._graph.validate(self.model_registry, require_complete=True)
            if not validation.valid:
                return self._reject_after_count(
                    action,
                    f"cannot finish: {self._format_issues(validation)}",
                    validation.issues,
                )
            format_issue = self.format_agent_issue()
            if format_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + format_issue,
                )
            execution = self._cached_progressive_execution()
            execution_reused = execution is not None
            if execution is None:
                try:
                    execution = await self.runtime.execute(
                        self._graph,
                        self._problem,
                        prior_outputs=self._progressive_outputs,
                        format_output_agent=self.require_format_agent,
                    )
                except AgentRuntimeError as exc:
                    return self._reject_after_count(
                        action,
                        "cannot finish: " + self._execution_error_feedback(exc),
                    )
            if execution.final_answer is None:
                return self._reject_after_count(
                    action,
                    "cannot finish: Format Agent produced no terminal artifact",
                )
            terminal_issue = self._terminal_validation_error(execution.final_answer)
            if terminal_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + terminal_issue,
                )
            self._finished = True
            self._last_feedback = "workflow finished"
            self._record_history(
                accepted=True,
                done=True,
                action=action,
                feedback=self._last_feedback,
                execution_reused=execution_reused,
            )
            return AgentWorkflowStepResult(
                accepted=True,
                done=True,
                action=action,
                revision=self._graph.revision,
                feedback=self._last_feedback,
                snapshot=self.snapshot(),
                execution=execution,
                execution_reused=execution_reused,
            )

        previous_revision = self._graph.revision
        candidate = self._graph.fork()
        try:
            dirty_agents = self._apply_mutation(candidate, action)
        except (GraphMutationError, TypeError, ValueError) as exc:
            return self._reject_after_count(action, f"edit rejected: {exc}")
        if candidate.revision == previous_revision:
            return self._reject_after_count(
                action,
                "edit rejected: action made no graph change; modify an Agent "
                "contract/model or another graph field before expecting a new execution",
            )
        validation = candidate.validate(self.model_registry, require_complete=False)
        if not validation.valid:
            return self._reject_after_count(
                action,
                f"edit rejected: {self._format_issues(validation)}",
                validation.issues,
            )
        if (
            action.action_type is AgentActionType.SET_OUTPUT
            and self.require_format_agent
        ):
            format_issue = self._format_agent_issue_for(candidate)
            if format_issue is not None:
                return self._reject_after_count(
                    action,
                    "edit rejected: " + format_issue,
                )

        self._graph = candidate
        execution = None
        execution_reused = False
        execution_error: Optional[AgentRuntimeError] = None
        if self.execute_on_edit:
            if self._graph.nodes:
                try:
                    execution = await self.runtime.execute(
                        self._graph,
                        self._problem,
                        require_complete=False,
                        prior_outputs=self._progressive_outputs,
                        dirty_agents=dirty_agents,
                        format_output_agent=self.require_format_agent,
                    )
                except AgentRuntimeError as exc:
                    # FlowSteer's progressive Canvas treats execution as edit
                    # feedback.  A provider/runtime failure must not roll back
                    # a structurally valid edit or abort the Director rollout.
                    execution_error = exc
                else:
                    self._progressive_outputs = dict(execution.outputs)
                    self._progressive_execution = execution
                    self._progressive_execution_revision = self._graph.revision
            else:
                self._clear_progressive_execution()
        self._last_feedback = self._accepted_feedback(
            action,
            execution,
            execution_error,
        )
        self._record_history(
            accepted=True,
            done=False,
            action=action,
            feedback=self._last_feedback,
            execution_reused=execution_reused,
        )
        return AgentWorkflowStepResult(
            accepted=True,
            done=False,
            action=action,
            revision=self._graph.revision,
            feedback=self._last_feedback,
            snapshot=self.snapshot(),
            execution=execution,
            execution_reused=execution_reused,
        )

    def _accepted_feedback(
        self,
        action: AgentAction,
        execution: Optional[AgentRuntimeResult],
        execution_error: Optional[AgentRuntimeError] = None,
    ) -> str:
        feedback = (
            f"accepted {action.action_type.value} at revision {self._graph.revision}"
        )
        if execution_error is not None:
            return f"{feedback}; {self._execution_error_feedback(execution_error)}"
        if execution is None:
            return feedback

        # FlowSteer's progressive Canvas returns the just-executed workflow
        # result to the policy after an edit.  Keep this receipt deliberately
        # compact: it is state feedback, not a task-specific Director template.
        answer = execution.final_answer
        if answer is not None and len(answer) > 400:
            answer = answer[:397] + "..."
        output_calls = [
            call
            for call in execution.calls
            if call.request.agent.id == execution.output_agent_id
        ]
        output_request = output_calls[-1].request if output_calls else None
        output_inbox = []
        if output_request is not None:
            for message in output_request.upstream[:4]:
                content = " ".join(message.content.split())
                if len(content) > 160:
                    content = content[:157] + "..."
                output_inbox.append(
                    {
                        "source_agent_id": message.source_agent_id,
                        "target_agent_id": message.target_agent_id,
                        "message_type": message.message_type,
                        "graph_revision": message.graph_revision,
                        "request_or_dependency": message.request_or_dependency,
                        "content_preview": content,
                    }
                )
        calls_by_agent = {
            call.request.agent.id: call for call in execution.calls
        }
        agent_artifacts = []
        for agent_id, artifact in sorted(execution.outputs.items()):
            call = calls_by_agent.get(agent_id)
            preview = " ".join(artifact.split())
            if len(preview) > 160:
                preview = preview[:157] + "..."
            agent_artifacts.append(
                {
                    "agent_id": agent_id,
                    "model_id": None if call is None else call.request.model.model_id,
                    "role_family": self._graph.get_node(agent_id).role_family,
                    "execution_role": (
                        "format" if agent_id == execution.output_agent_id else "worker"
                    ),
                    "is_output_agent": agent_id == execution.output_agent_id,
                    "upstream_source_ids": (
                        []
                        if call is None
                        else [item.source_agent_id for item in call.request.upstream]
                    ),
                    "artifact_preview": preview,
                }
            )
        result = json.dumps(
            {
                "output_agent_id": execution.output_agent_id,
                # This is a progressive execution observation.  Only an
                # accepted FINISH produces the trajectory's terminal answer;
                # calling this value ``final_answer`` prematurely made a
                # format-valid singleton look semantically terminal.
                "output": answer,
                "executed_agent_ids": list(execution.executed_agent_ids),
                "reused_agent_ids": list(execution.reused_agent_ids),
                "topology": self._graph.topology_statistics(),
                "output_inbox": output_inbox,
                "agent_artifacts": agent_artifacts,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"{feedback}; execution_result={result}"

    def _terminal_validation_error(self, answer: str) -> Optional[str]:
        """Apply the configured task terminal protocol before accepting FINISH.

        FlowSteer rejects FINISH at the Canvas boundary when a terminal
        constraint is unmet.  The exact answer wrapper is enabled only for QA
        tasks because SkillFlow's interactive environments may terminate with
        an admissible action instead of an XML answer.
        """

        if not self.require_exact_answer_tag:
            return None
        tag_count, exact_wrapper, non_empty = _answer_protocol_state(answer)
        if exact_wrapper and non_empty:
            return None
        return (
            "terminal answer must be exactly one non-empty "
            f"<answer>...</answer> wrapper; answer_tag_count={tag_count}, "
            f"exact_single_answer_tag={exact_wrapper}, non_empty={non_empty}; "
            "modify the Output Agent contract/model or graph before retrying"
        )

    def _cached_progressive_execution(self) -> Optional[AgentRuntimeResult]:
        if self._progressive_execution_revision != self._graph.revision:
            return None
        if self._progressive_execution is None:
            return None
        if self._progressive_execution.final_answer is None:
            return None
        return self._progressive_execution

    def _clear_progressive_execution(self) -> None:
        self._progressive_execution = None
        self._progressive_execution_revision = None
        self._progressive_outputs.clear()

    def format_agent_issue(self) -> Optional[str]:
        """Return the terminal Format-Agent constraint that is still unmet.

        FlowSteer's ``Format`` operator consumes one completed solution and
        performs extraction only.  The free-AgentGraph adaptation keeps
        ``role_family`` as metadata rather than an Operator enum, while the
        factual-QA terminal protocol reserves ``format`` for the distinct
        Output Agent and requires one routed semantic-answer artifact.
        """

        return self._format_agent_issue_for(self._graph)

    def _format_agent_issue_for(self, graph: AgentGraph) -> Optional[str]:
        if not self.require_format_agent:
            return None
        output_agent_id = graph.output_agent_id
        if output_agent_id is None:
            return "Format Agent is not selected as the Output Agent"
        output_node = graph.get_node(output_agent_id)
        if (output_node.role_family or "").casefold() != "format":
            return (
                "Output Agent must be a distinct Format Agent with "
                "role_family='format'; keep semantic-answer computation in "
                "its upstream Agent"
            )
        validation = graph.validate(
            self.model_registry,
            require_complete=False,
        )
        component = next(
            (
                item
                for item in validation.components
                if output_agent_id in item
            ),
            (),
        )
        if len(component) != 1:
            return "Format Agent must be a singleton terminal component"
        predecessors = graph.directed_predecessors(output_agent_id)
        if len(predecessors) != 1:
            return (
                "Format Agent must consume exactly one upstream semantic-answer "
                f"artifact; received {len(predecessors)}"
            )
        return None

    @staticmethod
    def _execution_error_feedback(exc: AgentRuntimeError) -> str:
        message = " ".join(str(exc).split())
        if len(message) > 240:
            message = message[:237] + "..."
        payload = json.dumps(
            {"type": type(exc).__name__, "message": message},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"execution_error={payload}"

    async def execute(self, *, run_id: Optional[str] = None) -> AgentRuntimeResult:
        if not self._problem:
            raise AgentWorkflowStateError("environment has no active problem")
        validation = self._graph.validate(self.model_registry, require_complete=True)
        validation.raise_if_invalid()
        return await self.runtime.execute(
            self._graph,
            self._problem,
            run_id=run_id,
            format_output_agent=self.require_format_agent,
        )

    def _apply_mutation(self, graph: AgentGraph, action: AgentAction) -> set[str]:
        if action.action_type is AgentActionType.ADD_AGENT:
            if action.agent_id is None or action.model_id is None or action.contract is None:
                raise GraphMutationError("add_agent action is incomplete")
            if (
                self.max_agents is not None
                and len(graph.nodes) >= self.max_agents
                and all(node.id != action.agent_id for node in graph.nodes)
            ):
                raise GraphMutationError(
                    f"agent limit reached: max_agents={self.max_agents}"
                )
            graph.add_agent(
                AgentNode(
                    action.agent_id,
                    action.model_id,
                    action.contract,
                    role_family=action.role_family,
                )
            )
            return {action.agent_id}
        elif action.action_type is AgentActionType.MODIFY_AGENT:
            if action.agent_id is None:
                raise GraphMutationError("modify_agent action is incomplete")
            graph.modify_agent(
                action.agent_id,
                model_id=action.model_id,
                contract=action.contract,
                role_family=action.role_family,
            )
            return graph.dirty_closure({action.agent_id})
        elif action.action_type is AgentActionType.DELETE_AGENT:
            if action.agent_id is None:
                raise GraphMutationError("delete_agent action is incomplete")
            dirty = graph.dirty_closure({action.agent_id}) - {action.agent_id}
            graph.delete_agent(action.agent_id)
            return dirty
        elif action.action_type is AgentActionType.SET_RELATION:
            if (
                action.source_id is None
                or action.target_id is None
                or action.source_to_target is None
                or action.target_to_source is None
            ):
                raise GraphMutationError("set_relation action is incomplete")
            previous = graph.relation_bits(action.source_id, action.target_id)
            previous_targets = set()
            if previous.source_to_target:
                previous_targets.add(action.target_id)
            if previous.target_to_source:
                previous_targets.add(action.source_id)
            before = graph.dirty_closure(previous_targets)
            graph.set_relation(
                action.source_id,
                action.target_id,
                action.source_to_target,
                action.target_to_source,
            )
            current_targets = set()
            if action.source_to_target:
                current_targets.add(action.target_id)
            if action.target_to_source:
                current_targets.add(action.source_id)
            return before | graph.dirty_closure(current_targets)
        elif action.action_type is AgentActionType.SET_OUTPUT:
            if action.agent_id is None:
                raise GraphMutationError("set_output action is incomplete")
            previous = graph.output_agent_id
            graph.set_output(action.agent_id)
            seeds = {action.agent_id}
            if previous is not None:
                seeds.add(previous)
            return graph.dirty_closure(seeds)
        else:
            raise GraphMutationError(f"unsupported graph edit: {action.action_type.value}")

    def _reject(
        self,
        action: Optional[AgentAction],
        feedback: str,
        issues: Tuple[GraphValidationIssue, ...] = (),
    ) -> AgentWorkflowStepResult:
        self._turn_count += 1
        return self._reject_after_count(action, feedback, issues)

    def _reject_after_count(
        self,
        action: Optional[AgentAction],
        feedback: str,
        issues: Tuple[GraphValidationIssue, ...] = (),
    ) -> AgentWorkflowStepResult:
        self._last_feedback = feedback
        self._record_history(
            accepted=False,
            done=self._finished,
            action=action,
            feedback=feedback,
        )
        return AgentWorkflowStepResult(
            accepted=False,
            done=self._finished,
            action=action,
            revision=self._graph.revision,
            feedback=feedback,
            snapshot=self.snapshot(),
            validation_issues=issues,
        )

    def _record_history(
        self,
        *,
        accepted: bool,
        done: bool,
        action: Optional[AgentAction],
        feedback: str,
        execution_reused: bool = False,
    ) -> None:
        self._history.append(
            AgentWorkflowHistoryEntry(
                turn_count=self._turn_count,
                accepted=accepted,
                done=done,
                action=action,
                revision=self._graph.revision,
                feedback=feedback,
                execution_reused=execution_reused,
            )
        )

    def _validate_agent_limit(self, graph: AgentGraph) -> None:
        if self.max_agents is not None and len(graph.nodes) > self.max_agents:
            raise AgentWorkflowStateError(
                f"graph has {len(graph.nodes)} agents but max_agents={self.max_agents}"
            )

    @staticmethod
    def _format_issues(validation: GraphValidationResult) -> str:
        return "; ".join(
            f"{issue.code}: {issue.message}" for issue in validation.issues
        ) or "valid"


StepResult = AgentWorkflowStepResult


__all__ = [
    "AgentWorkflowEnv",
    "AgentWorkflowHistoryEntry",
    "AgentWorkflowSnapshot",
    "AgentWorkflowStateError",
    "AgentWorkflowStepResult",
    "StepResult",
]
