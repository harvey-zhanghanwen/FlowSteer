"""Transactional progressive Canvas environment for AgentGraph actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Optional, Sequence, Tuple, Union

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


_PRESERVE_REPAIR_RECOVERY_POLICY = "preserve_diagnose_repair_augment"
_SUPPORTED_RECOVERY_POLICIES = frozenset(
    {"default", _PRESERVE_REPAIR_RECOVERY_POLICY}
)
_SUPPORTED_DIRECTOR_FEEDBACK_MODES = frozenset({"content", "control_plane"})


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
    partial_execution: Optional[AgentRuntimeResult] = None
    execution_failure_records: Tuple[object, ...] = ()

    @property
    def success(self) -> bool:
        return self.accepted

    @property
    def final_answer(self) -> Optional[str]:
        return None if self.execution is None else self.execution.final_answer


class AgentWorkflowEnv:
    """Apply one Canvas action per turn and execute the accepted graph revision."""

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
        max_agents_per_subgraph: int = 3,
        require_exact_answer_tag: bool = False,
        require_format_agent: bool = False,
        required_evidence_tool_id: Optional[str] = None,
        require_evidence_relation: bool = False,
        allowed_actions: Optional[Sequence[str]] = None,
        recovery_policy: str = "default",
        director_feedback_mode: str = "content",
    ) -> None:
        if runtime is None and gateway is None:
            raise AgentWorkflowStateError("gateway or runtime is required")
        if runtime is not None and runtime.model_registry is not model_registry:
            raise AgentWorkflowStateError("runtime and environment must share the model registry")
        if max_agents is not None and (
            isinstance(max_agents, bool) or not isinstance(max_agents, int) or max_agents < 1
        ):
            raise AgentWorkflowStateError("max_agents must be a positive integer or None")
        if (
            isinstance(max_agents_per_subgraph, bool)
            or not isinstance(max_agents_per_subgraph, int)
            or max_agents_per_subgraph < 1
        ):
            raise AgentWorkflowStateError(
                "max_agents_per_subgraph must be a positive integer"
            )
        if type(require_exact_answer_tag) is not bool:
            raise AgentWorkflowStateError("require_exact_answer_tag must be bool")
        if type(require_format_agent) is not bool:
            raise AgentWorkflowStateError("require_format_agent must be bool")
        if required_evidence_tool_id is not None and (
            not isinstance(required_evidence_tool_id, str)
            or not required_evidence_tool_id.strip()
        ):
            raise AgentWorkflowStateError(
                "required_evidence_tool_id must be non-empty text or None"
            )
        if type(require_evidence_relation) is not bool:
            raise AgentWorkflowStateError("require_evidence_relation must be bool")
        if require_evidence_relation and required_evidence_tool_id is None:
            raise AgentWorkflowStateError(
                "require_evidence_relation requires required_evidence_tool_id"
            )
        if (
            not isinstance(recovery_policy, str)
            or recovery_policy not in _SUPPORTED_RECOVERY_POLICIES
        ):
            raise AgentWorkflowStateError(
                "recovery_policy must be default or "
                f"{_PRESERVE_REPAIR_RECOVERY_POLICY}"
            )
        if director_feedback_mode not in _SUPPORTED_DIRECTOR_FEEDBACK_MODES:
            raise AgentWorkflowStateError(
                "director_feedback_mode must be content or control_plane"
            )
        if allowed_actions is None:
            resolved_allowed_actions = tuple(item.value for item in AgentActionType)
        else:
            if isinstance(allowed_actions, (str, bytes)) or not allowed_actions:
                raise AgentWorkflowStateError(
                    "allowed_actions must be a non-empty sequence"
                )
            resolved_allowed_actions = tuple(allowed_actions)
            known_actions = {item.value for item in AgentActionType}
            if (
                any(
                    not isinstance(item, str) or item not in known_actions
                    for item in resolved_allowed_actions
                )
                or len(resolved_allowed_actions) != len(set(resolved_allowed_actions))
            ):
                raise AgentWorkflowStateError(
                    "allowed_actions contains an unknown or duplicate action"
                )
        self.model_registry = model_registry
        self.runtime = runtime or AgentRuntime(model_registry, gateway)  # type: ignore[arg-type]
        self.execute_on_edit = execute_on_edit
        self.max_agents = max_agents
        self.max_agents_per_subgraph = max_agents_per_subgraph
        self.require_exact_answer_tag = require_exact_answer_tag
        self.require_format_agent = require_format_agent
        self.required_evidence_tool_id = (
            None
            if required_evidence_tool_id is None
            else required_evidence_tool_id.strip()
        )
        self.require_evidence_relation = require_evidence_relation
        self.allowed_action_types = resolved_allowed_actions
        self._allowed_action_type_set = frozenset(resolved_allowed_actions)
        self.recovery_policy = recovery_policy
        self.director_feedback_mode = director_feedback_mode
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
        self._progressive_output_metadata: dict[
            str, dict[str, object]
        ] = {}
        self._failed_agent_ids: set[str] = set()
        # A recovery cycle first repairs the failed node once.  If that does
        # not produce a successful artifact, the Canvas may augment the graph
        # on the next step rather than repeatedly modifying the same node.
        self._repair_attempted_failed_agent_ids: set[str] = set()
        self._diagnosed_unusable_agent_ids: set[str] = set()
        self._failed_output_agent_ids: set[str] = set()
        self._unresolved_dirty_agent_ids: set[str] = set()
        self._validate_agent_limit(self._graph)
        self._validate_graph_execution_profiles(self._graph)
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

    @property
    def unresolved_dirty_agent_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._unresolved_dirty_agent_ids))

    def registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return the execution-mode/Tool pairs registered by this Runtime.

        The Runtime is the capability authority.  Role metadata and Director
        text cannot create an executor or grant a Tool capability.
        """

        provider = getattr(self.runtime, "registered_execution_profiles", None)
        if not callable(provider):
            raise AgentWorkflowStateError(
                "AgentRuntime must expose registered_execution_profiles()"
            )
        raw_profiles = provider()
        if not isinstance(raw_profiles, (list, tuple)):
            raise AgentWorkflowStateError(
                "registered execution profiles must be a sequence"
            )
        profiles: list[tuple[str, tuple[str, ...]]] = []
        for raw_profile in raw_profiles:
            if (
                not isinstance(raw_profile, (list, tuple))
                or len(raw_profile) != 2
            ):
                raise AgentWorkflowStateError(
                    "registered execution profile is malformed"
                )
            execution_mode, raw_tools = raw_profile
            if (
                execution_mode not in {"reasoning", "react", "coding"}
                or not isinstance(raw_tools, (list, tuple))
                or any(
                    not isinstance(tool_id, str) or not tool_id
                    for tool_id in raw_tools
                )
                or len(raw_tools) != len(set(raw_tools))
            ):
                raise AgentWorkflowStateError(
                    "registered execution profile is invalid"
                )
            profile = (execution_mode, tuple(raw_tools))
            if (
                self.required_evidence_tool_id is not None
                and execution_mode == "react"
                and profile
                != ("react", (self.required_evidence_tool_id,))
            ):
                # A task-scoped retrieval Runtime may technically execute an
                # empty ReAct loop, but that pair cannot satisfy this Canvas'
                # required evidence capability and is therefore not a live
                # Agent declaration target.
                continue
            if profile in profiles:
                raise AgentWorkflowStateError(
                    "registered execution profiles contain a duplicate"
                )
            profiles.append(profile)
        if not profiles:
            raise AgentWorkflowStateError(
                "AgentRuntime has no registered execution profiles"
            )
        return tuple(profiles)

    def _topology_neutral_registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Project the staged worker/downstream Runtime capability domain.

        DIRECT_REUSE: this is the topology-neutral Tool Canvas projection from
        the TriviaQA v10 architecture.  It fixes only capability ownership: the
        first functional unit materializes one worker Tool artifact and later
        units consume it without the Tool.  Roles, contracts, models, Agent
        count after that first unit, relations and final topology remain under
        the Director's progressive Canvas editing policy.
        """

        profiles = self.registered_execution_profiles()
        tool_id = self.required_evidence_tool_id
        if not self.require_evidence_relation or tool_id is None:
            return profiles
        successful_workers = self._current_successful_evidence_agent_ids()
        if not self._graph.nodes and not successful_workers:
            return tuple(profile for profile in profiles if profile[1] == (tool_id,))
        if successful_workers:
            return tuple(profile for profile in profiles if not profile[1])
        return profiles

    @staticmethod
    def _serialized_execution_profiles(
        profiles: Sequence[Tuple[str, Tuple[str, ...]]],
    ) -> list[dict[str, object]]:
        return [
            {
                "execution_mode": execution_mode,
                "allowed_tools": list(allowed_tools),
            }
            for execution_mode, allowed_tools in profiles
        ]

    def model_admissible_action_types(self) -> Tuple[str, ...]:
        """Project the live, state-conditioned Canvas action domain."""

        node_ids = tuple(node.id for node in self._graph.nodes)
        successful_evidence_ids = set(
            self._current_successful_evidence_agent_ids()
        )
        modifiable_ids = tuple(
            agent_id
            for agent_id in node_ids
            if agent_id not in successful_evidence_ids
            or agent_id in self._failed_agent_ids
        )
        output_target_ids = tuple(
            node.id
            for node in self._graph.nodes
            if not (
                self.require_evidence_relation
                and self.required_evidence_tool_id in node.allowed_tools
            )
        )
        repair_required_ids = (
            self._failed_agent_ids - self._repair_attempted_failed_agent_ids
        )
        if (
            self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY
            and repair_required_ids
            and AgentActionType.MODIFY_AGENT.value in self._allowed_action_type_set
        ):
            # DIRECT_REUSE + NECESSARY_ADAPTATION: FlowSteer's typed provider
            # recovery keeps the failed node and its artifacts in place, and
            # admits one repair before augmentation.  The live target domain
            # below narrows MODIFY to the measured failed workers.
            return (AgentActionType.MODIFY_AGENT.value,)
        can_add = self.max_agents is None or len(node_ids) < self.max_agents
        deletable_ids = {
            agent_id
            for agent_id in node_ids
            if self._delete_admission_issue(agent_id) is None
        }
        finish_admissible = self.finish_admissibility()["admissible"] is True
        if (
            finish_admissible
            and AgentActionType.FINISH.value in self._allowed_action_type_set
        ):
            # FlowSteer's state-conditioned action mask still requires the
            # Director to submit the explicit terminal action.  Once the
            # current executed revision is admissible, later edits can only
            # discard a valid routed artifact.
            return (AgentActionType.FINISH.value,)
        admitted: list[str] = []
        for action_type in self.allowed_action_types:
            if action_type in {
                AgentActionType.ADD_AGENT.value,
                AgentActionType.ADD_SUBGRAPH.value,
            } and not can_add:
                continue
            if (
                action_type == AgentActionType.MODIFY_AGENT.value
                and not modifiable_ids
            ):
                continue
            if action_type == AgentActionType.DELETE_AGENT.value and not deletable_ids:
                continue
            if (
                action_type == AgentActionType.DELETE_AGENT.value
                and self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY
                and self._failed_agent_ids
            ):
                # Failure recovery preserves existing lineage.  A failure
                # never makes deletion an advertised repair action.
                continue
            if action_type == AgentActionType.SET_RELATION.value and len(node_ids) < 2:
                continue
            if (
                action_type == AgentActionType.SET_OUTPUT.value
                and not output_target_ids
            ):
                continue
            if (
                action_type == AgentActionType.FINISH.value
                and not finish_admissible
            ):
                continue
            admitted.append(action_type)
        return tuple(admitted)

    def model_admissible_action_targets(self) -> dict[str, object]:
        """Return the live domains that correspond to admissible actions."""

        admitted = set(self.model_admissible_action_types())
        node_ids = [node.id for node in self._graph.nodes]
        model_ids = list(self.model_registry.model_ids)
        profiles = self.registered_execution_profiles()
        serialized_profiles = self._serialized_execution_profiles(profiles)
        successful_evidence_ids = set(
            self._current_successful_evidence_agent_ids()
        )
        output_target_ids = [
            node.id
            for node in self._graph.nodes
            if not (
                self.require_evidence_relation
                and self.required_evidence_tool_id in node.allowed_tools
            )
        ]
        add_profiles = self._topology_neutral_registered_execution_profiles()
        serialized_add_profiles = self._serialized_execution_profiles(add_profiles)
        result: dict[str, object] = {
            "registered_execution_profiles": serialized_profiles,
            "finish_admissibility": self.finish_admissibility(),
        }
        if AgentActionType.ADD_AGENT.value in admitted:
            result[AgentActionType.ADD_AGENT.value] = {
                "model_ids": model_ids,
                "execution_profiles": serialized_add_profiles,
            }
        if AgentActionType.ADD_SUBGRAPH.value in admitted:
            remaining = (
                self.max_agents_per_subgraph
                if self.max_agents is None
                else min(
                    self.max_agents_per_subgraph,
                    self.max_agents - len(node_ids),
                )
            )
            result[AgentActionType.ADD_SUBGRAPH.value] = {
                "model_ids": model_ids,
                "execution_profiles": serialized_add_profiles,
                "existing_agent_ids": node_ids,
                "max_new_agents": (
                    1
                    if not node_ids
                    and self.require_evidence_relation
                    and self.required_evidence_tool_id is not None
                    else remaining
                ),
                "preserved_input_agent_ids": sorted(successful_evidence_ids),
                "output_agent_ids": output_target_ids,
            }
        if AgentActionType.MODIFY_AGENT.value in admitted:
            result[AgentActionType.MODIFY_AGENT.value] = {
                "agent_ids": (
                    sorted(
                        self._failed_agent_ids
                        - self._repair_attempted_failed_agent_ids
                    )
                    if self.recovery_policy
                    == _PRESERVE_REPAIR_RECOVERY_POLICY
                    and (
                        self._failed_agent_ids
                        - self._repair_attempted_failed_agent_ids
                    )
                    else [
                        agent_id
                        for agent_id in node_ids
                        if agent_id not in successful_evidence_ids
                        or agent_id in self._failed_agent_ids
                    ]
                ),
                "model_ids": model_ids,
                "execution_profiles": serialized_profiles,
                "output_agent_id": self._graph.output_agent_id,
            }
        if AgentActionType.DELETE_AGENT.value in admitted:
            result[AgentActionType.DELETE_AGENT.value] = {
                "agent_ids": [
                    agent_id
                    for agent_id in node_ids
                    if self._delete_admission_issue(agent_id) is None
                ],
            }
        if AgentActionType.SET_RELATION.value in admitted:
            result[AgentActionType.SET_RELATION.value] = {
                "agent_ids": node_ids,
                "preserved_input_agent_ids": sorted(successful_evidence_ids),
            }
        if AgentActionType.SET_OUTPUT.value in admitted:
            result[AgentActionType.SET_OUTPUT.value] = {
                "agent_ids": output_target_ids,
            }
        if AgentActionType.FINISH.value in admitted:
            result[AgentActionType.FINISH.value] = {"admissible": True}
        return result

    def finish_admissibility(self) -> dict[str, object]:
        """Return the current revision's complete terminal admission state."""

        validation = self._graph.validate(
            self.model_registry,
            require_complete=True,
        )
        if not validation.valid:
            return {
                "admissible": False,
                "reason": self._format_issues(validation),
            }
        format_issue = self.format_agent_issue()
        if format_issue is not None:
            return {"admissible": False, "reason": format_issue}
        if not self.execute_on_edit:
            return {"admissible": True, "reason": "execute_on_finish"}
        execution = self._cached_progressive_execution()
        if execution is None:
            return {
                "admissible": False,
                "reason": "current graph revision has no successful execution receipt",
            }
        if execution.final_answer is None:
            return {
                "admissible": False,
                "reason": "current Output Agent produced no terminal artifact",
            }
        terminal_issue = self._terminal_validation_error(execution.final_answer)
        if terminal_issue is not None:
            return {"admissible": False, "reason": terminal_issue}
        evidence_issue = self._required_evidence_issue(execution)
        if evidence_issue is not None:
            return {"admissible": False, "reason": evidence_issue}
        return {
            "admissible": True,
            "reason": "current graph revision passed terminal admission",
        }

    @staticmethod
    def _directed_successors(graph: AgentGraph, agent_id: str) -> set[str]:
        return {
            target_id
            for relation in graph.relations
            for source_id, target_id in relation.directed_edges()
            if source_id == agent_id
        }

    def _tool_dataflow_mutation_issue(
        self,
        action: AgentAction,
        candidate: AgentGraph,
    ) -> Optional[str]:
        """Enforce the worker -> downstream artifact boundary authoritatively.

        DIRECT_REUSE + NECESSARY_ADAPTATION: TriviaQA v10 filters incoming
        relations to successful Tool workers in the live Director domain.  This
        Canvas-side check is the corresponding authoritative boundary for the
        simpler HotpotQA declaration-first decoder and also closes its
        ADD/MODIFY Output-Tool loopholes.
        """

        tool_id = self.required_evidence_tool_id
        if tool_id is None or not self.require_evidence_relation:
            return None
        output_id = candidate.output_agent_id
        if output_id is not None and tool_id in candidate.get_node(output_id).allowed_tools:
            return (
                f"Output Agent {output_id!r} cannot hold retrieval Tool {tool_id!r}; "
                "route the worker artifact to a no-Tool downstream Agent and select "
                "that consumer as Output"
            )
        preserved = set(self._current_successful_evidence_agent_ids())
        if not preserved:
            return None
        if action.action_type in {
            AgentActionType.MODIFY_AGENT,
            AgentActionType.DELETE_AGENT,
        } and action.agent_id in preserved:
            return (
                f"successful Tool worker {action.agent_id!r} and its artifact are "
                "preserved; repair or augment downstream instead"
            )
        previous_edges = {
            edge
            for relation in self._graph.relations
            for edge in relation.directed_edges()
        }
        candidate_edges = {
            edge
            for relation in candidate.relations
            for edge in relation.directed_edges()
        }
        incoming = sorted(
            (source_id, target_id)
            for source_id, target_id in candidate_edges - previous_edges
            if target_id in preserved
        )
        if incoming:
            return (
                "a successful Tool worker cannot receive a new upstream edge; "
                f"preserve worker -> downstream dataflow, rejected_edges={incoming!r}"
            )
        return None

    def _replacement_takeover_agent_ids(self, agent_id: str) -> Tuple[str, ...]:
        if not self._graph.has_node(agent_id):
            return ()
        failed = self._graph.get_node(agent_id)
        failed_role = (failed.role_family or "").casefold()
        failed_artifact_type = failed.artifact_type.casefold()
        failed_downstream = self._directed_successors(self._graph, agent_id)
        output_takeover_required = agent_id in self._failed_output_agent_ids
        replacements: list[str] = []
        for candidate in self._graph.nodes:
            artifact = self._progressive_outputs.get(candidate.id)
            if (
                candidate.id == agent_id
                or candidate.id in self._diagnosed_unusable_agent_ids
                or (candidate.role_family or "").casefold() != failed_role
                or candidate.artifact_type.casefold() != failed_artifact_type
                or not isinstance(artifact, str)
                or not artifact.strip()
                or not failed_downstream
                <= self._directed_successors(self._graph, candidate.id)
                or (
                    output_takeover_required
                    and self._graph.output_agent_id != candidate.id
                )
            ):
                continue
            replacements.append(candidate.id)
        return tuple(replacements)

    def _delete_admission_issue(self, agent_id: Optional[str]) -> Optional[str]:
        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return None
        if agent_id is None or not self._graph.has_node(agent_id):
            return None
        replacements = self._replacement_takeover_agent_ids(agent_id)
        if (
            agent_id in self._diagnosed_unusable_agent_ids
            and replacements
        ):
            return None
        return (
            f"recovery_policy={_PRESERVE_REPAIR_RECOVERY_POLICY} protects Agent "
            f"{agent_id!r}. Use preserve -> diagnose -> repair -> augment; "
            "delete is admitted only after a typed node_unusable diagnosis and "
            "a successful same-role_family/same-artifact_type replacement has "
            "taken over every downstream edge and any previous Output identity"
        )

    def _record_failure_state(self, records: Sequence[object]) -> None:
        """Record typed Runtime failures without inferring node unusability."""

        current_ids = {node.id for node in self._graph.nodes}
        for record in records:
            agent_id = getattr(record, "agent_id", None)
            metadata = getattr(record, "metadata", None)
            if agent_id not in current_ids or not isinstance(metadata, Mapping):
                continue
            self._failed_agent_ids.add(agent_id)
            self._unresolved_dirty_agent_ids.add(agent_id)
            if metadata.get("node_unusable") is True:
                self._diagnosed_unusable_agent_ids.add(agent_id)
                if self._graph.output_agent_id == agent_id:
                    self._failed_output_agent_ids.add(agent_id)
            else:
                self._diagnosed_unusable_agent_ids.discard(agent_id)

    def _mark_agents_recovered(self, agent_ids: Sequence[str]) -> None:
        recovered = set(agent_ids)
        self._failed_agent_ids.difference_update(recovered)
        self._repair_attempted_failed_agent_ids.difference_update(recovered)
        self._diagnosed_unusable_agent_ids.difference_update(recovered)
        self._failed_output_agent_ids.difference_update(recovered)

    def _clear_failure_state(self) -> None:
        self._failed_agent_ids.clear()
        self._repair_attempted_failed_agent_ids.clear()
        self._diagnosed_unusable_agent_ids.clear()
        self._failed_output_agent_ids.clear()
        self._unresolved_dirty_agent_ids.clear()

    def recovery_state(self) -> dict[str, object]:
        """Expose topology-neutral preserve/repair state to the Director."""

        deletable = [
            node.id
            for node in self._graph.nodes
            if self._delete_admission_issue(node.id) is None
        ]
        replacements = {
            agent_id: list(self._replacement_takeover_agent_ids(agent_id))
            for agent_id in sorted(self._diagnosed_unusable_agent_ids)
        }
        return {
            "policy": self.recovery_policy,
            "strategy": "preserve -> diagnose -> repair -> augment",
            "phase": (
                "augment"
                if self._failed_agent_ids
                and self._failed_agent_ids
                <= self._repair_attempted_failed_agent_ids
                else "augment"
                if any(replacements.values())
                else "repair"
                if self._diagnosed_unusable_agent_ids
                else "diagnose"
                if self._failed_agent_ids
                else "preserve"
            ),
            "failed_agent_ids": sorted(self._failed_agent_ids),
            "repair_attempted_failed_agent_ids": sorted(
                self._repair_attempted_failed_agent_ids
            ),
            "diagnosed_unusable_agent_ids": sorted(
                self._diagnosed_unusable_agent_ids
            ),
            "unresolved_dirty_agent_ids": list(
                self.unresolved_dirty_agent_ids
            ),
            "replacement_takeover_agent_ids": replacements,
            "deletable_agent_ids": deletable,
        }

    def reset(self, problem: str, graph: Optional[AgentGraph] = None) -> AgentWorkflowSnapshot:
        if not isinstance(problem, str) or not problem.strip():
            raise AgentWorkflowStateError("problem must be a non-empty string")
        candidate = graph.fork() if graph is not None else AgentGraph()
        self._validate_agent_limit(candidate)
        self._validate_graph_execution_profiles(candidate)
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
        self._clear_failure_state()
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
        self._validate_graph_execution_profiles(graph)
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
        self._clear_failure_state()

    def fork(self, snapshot: Optional[AgentWorkflowSnapshot] = None) -> "AgentWorkflowEnv":
        state = snapshot or self.snapshot()
        result = AgentWorkflowEnv(
            self.model_registry,
            runtime=self.runtime,
            problem=state.problem,
            graph=AgentGraph.from_snapshot(state.graph),
            execute_on_edit=self.execute_on_edit,
            max_agents=self.max_agents,
            max_agents_per_subgraph=self.max_agents_per_subgraph,
            require_exact_answer_tag=self.require_exact_answer_tag,
            require_format_agent=self.require_format_agent,
            required_evidence_tool_id=self.required_evidence_tool_id,
            require_evidence_relation=self.require_evidence_relation,
            allowed_actions=self.allowed_action_types,
            recovery_policy=self.recovery_policy,
            director_feedback_mode=self.director_feedback_mode,
        )
        result._turn_count = state.turn_count
        result._finished = state.finished
        result._last_feedback = state.last_feedback
        result._history = list(state.history)
        result._failed_agent_ids = set(self._failed_agent_ids)
        result._repair_attempted_failed_agent_ids = set(
            self._repair_attempted_failed_agent_ids
        )
        result._diagnosed_unusable_agent_ids = set(
            self._diagnosed_unusable_agent_ids
        )
        result._failed_output_agent_ids = set(self._failed_output_agent_ids)
        result._unresolved_dirty_agent_ids = set(
            self._unresolved_dirty_agent_ids
        )
        result._progressive_outputs = dict(self._progressive_outputs)
        result._progressive_output_metadata = {
            agent_id: dict(metadata)
            for agent_id, metadata in self._progressive_output_metadata.items()
        }
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
        if action.action_type.value not in self._allowed_action_type_set:
            return self._reject_after_count(
                action,
                "action rejected: action type is outside the configured Canvas "
                f"action set {list(self.allowed_action_types)!r}",
            )
        if action.action_type is AgentActionType.DELETE_AGENT:
            delete_issue = self._delete_admission_issue(action.agent_id)
            if delete_issue is not None:
                return self._reject_after_count(
                    action,
                    "edit rejected: " + delete_issue,
                )
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
                        prior_output_metadata=self._progressive_output_metadata,
                        format_output_agent=self.require_format_agent,
                    )
                except AgentRuntimeError as exc:
                    failure_records = tuple(
                        getattr(exc, "failure_records", ())
                    )
                    self._record_failure_state(
                        failure_records,
                    )
                    self._unresolved_dirty_agent_ids.update(
                        agent_id
                        for agent_id in (
                            *getattr(exc, "blocked_agent_ids", ()),
                            *getattr(exc, "pending_agent_ids", ()),
                        )
                        if self._graph.has_node(agent_id)
                    )
                    return self._reject_after_count(
                        action,
                        "cannot finish: " + self._execution_error_feedback(exc),
                        partial_execution=getattr(exc, "partial_result", None),
                        execution_failure_records=failure_records,
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
            evidence_issue = self._required_evidence_issue(execution)
            if evidence_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + evidence_issue,
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
            cached_execution = self._cached_progressive_execution()
            if cached_execution is not None:
                self._last_feedback = (
                    f"accepted {action.action_type.value} at revision "
                    f"{self._graph.revision}; execution_result_reused=true"
                )
                self._record_history(
                    accepted=True,
                    done=False,
                    action=action,
                    feedback=self._last_feedback,
                    execution_reused=True,
                )
                return AgentWorkflowStepResult(
                    accepted=True,
                    done=False,
                    action=action,
                    revision=self._graph.revision,
                    feedback=self._last_feedback,
                    snapshot=self.snapshot(),
                    execution=cached_execution,
                    execution_reused=True,
                )
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
        tool_dataflow_issue = self._tool_dataflow_mutation_issue(action, candidate)
        if tool_dataflow_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + tool_dataflow_issue,
            )
        if (
            action.action_type in {
                AgentActionType.SET_OUTPUT,
                AgentActionType.ADD_SUBGRAPH,
            }
            and candidate.output_agent_id is not None
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
        partial_execution: Optional[AgentRuntimeResult] = None
        execution_failure_records: Tuple[object, ...] = ()
        if self.execute_on_edit:
            if self._graph.nodes:
                try:
                    execution = await self.runtime.execute(
                        self._graph,
                        self._problem,
                        require_complete=False,
                        prior_outputs=self._progressive_outputs,
                        prior_output_metadata=self._progressive_output_metadata,
                        dirty_agents=dirty_agents,
                        format_output_agent=self.require_format_agent,
                    )
                except AgentRuntimeError as exc:
                    # FlowSteer's progressive Canvas treats execution as edit
                    # feedback.  A provider/runtime failure must not roll back
                    # a structurally valid edit or abort the Director rollout.
                    execution_error = exc
                    partial_execution = getattr(exc, "partial_result", None)
                    execution_failure_records = tuple(
                        getattr(exc, "failure_records", ())
                    )
                    if partial_execution is not None:
                        self._progressive_outputs.update(
                            dict(partial_execution.outputs)
                        )
                        self._progressive_output_metadata.update(
                            {
                                agent_id: dict(metadata)
                                for agent_id, metadata in (
                                    partial_execution.output_metadata.items()
                                )
                            }
                        )
                        self._mark_agents_recovered(partial_execution.outputs)
                    self._record_failure_state(
                        execution_failure_records,
                    )
                    self._unresolved_dirty_agent_ids.update(
                        agent_id
                        for agent_id in (
                            *getattr(exc, "blocked_agent_ids", ()),
                            *getattr(exc, "pending_agent_ids", ()),
                        )
                        if self._graph.has_node(agent_id)
                    )
                else:
                    self._progressive_outputs = dict(execution.outputs)
                    self._progressive_output_metadata = {
                        agent_id: dict(metadata)
                        for agent_id, metadata in execution.output_metadata.items()
                    }
                    self._progressive_execution = execution
                    self._progressive_execution_revision = self._graph.revision
                    self._mark_agents_recovered(execution.outputs)
                    self._unresolved_dirty_agent_ids.difference_update(
                        execution.outputs
                    )
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
        if (
            action.action_type is AgentActionType.MODIFY_AGENT
            and action.agent_id in self._failed_agent_ids
        ):
            self._repair_attempted_failed_agent_ids.add(action.agent_id)
        return AgentWorkflowStepResult(
            accepted=True,
            done=False,
            action=action,
            revision=self._graph.revision,
            feedback=self._last_feedback,
            snapshot=self.snapshot(),
            execution=execution,
            execution_reused=execution_reused,
            partial_execution=partial_execution,
            execution_failure_records=execution_failure_records,
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
        expose_content = self.director_feedback_mode == "content"
        answer = execution.final_answer if expose_content else None
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
                successful_actions = sorted(
                    {
                        str(receipt.get("request", {}).get("action"))
                        for receipt in message.tool_receipts
                        if isinstance(receipt, Mapping)
                        and isinstance(receipt.get("request"), Mapping)
                        and receipt.get("error_type") is None
                        and isinstance(receipt.get("result"), Mapping)
                        and receipt.get("request", {}).get("action") is not None
                    }
                )
                failed_actions = sorted(
                    {
                        str(receipt.get("request", {}).get("action"))
                        for receipt in message.tool_receipts
                        if isinstance(receipt, Mapping)
                        and isinstance(receipt.get("request"), Mapping)
                        and receipt.get("error_type") is not None
                        and receipt.get("request", {}).get("action") is not None
                    }
                )
                item = {
                        "source_agent_id": message.source_agent_id,
                        "target_agent_id": message.target_agent_id,
                        "message_type": message.message_type,
                        "artifact_type": message.artifact_type,
                        "graph_revision": message.graph_revision,
                        "environment_revision": message.environment_revision,
                        "request_or_dependency": message.request_or_dependency,
                        "tool_receipt_count": len(message.tool_receipts),
                        "tool_ids": sorted(
                            {
                                str(receipt.get("tool_id"))
                                for receipt in message.tool_receipts
                                if receipt.get("tool_id") is not None
                            }
                        ),
                        "successful_tool_actions": successful_actions,
                        "failed_tool_actions": failed_actions,
                    }
                if expose_content:
                    content = " ".join(message.content.split())
                    if len(content) > 160:
                        content = content[:157] + "..."
                    item["content_preview"] = content
                output_inbox.append(item)
        calls_by_agent = {
            call.request.agent.id: call for call in execution.calls
        }
        agent_artifacts = []
        for agent_id, artifact in sorted(execution.outputs.items()):
            call = calls_by_agent.get(agent_id)
            output_metadata = execution.output_metadata.get(agent_id, {})
            tool_receipts = output_metadata.get("tool_receipts", ())
            if not isinstance(tool_receipts, (list, tuple)):
                tool_receipts = ()
            tool_ids = sorted(
                {
                    str(receipt.get("tool_id"))
                    for receipt in tool_receipts
                    if isinstance(receipt, Mapping)
                    and receipt.get("tool_id") is not None
                }
            )
            successful_tool_actions = sorted(
                {
                    str(receipt.get("request", {}).get("action"))
                    for receipt in tool_receipts
                    if isinstance(receipt, Mapping)
                    and isinstance(receipt.get("request"), Mapping)
                    and receipt.get("error_type") is None
                    and isinstance(receipt.get("result"), Mapping)
                    and receipt.get("request", {}).get("action") is not None
                }
            )
            failed_tool_actions = sorted(
                {
                    str(receipt.get("request", {}).get("action"))
                    for receipt in tool_receipts
                    if isinstance(receipt, Mapping)
                    and isinstance(receipt.get("request"), Mapping)
                    and receipt.get("error_type") is not None
                    and receipt.get("request", {}).get("action") is not None
                }
            )
            item = {
                    "agent_id": agent_id,
                    "artifact_id": f"revision-{self._graph.revision}:{agent_id}",
                    "model_id": None if call is None else call.request.model.model_id,
                    "role_family": self._graph.get_node(agent_id).role_family,
                    "execution_mode": self._graph.get_node(
                        agent_id
                    ).execution_mode.value,
                    "allowed_tools": list(
                        self._graph.get_node(agent_id).allowed_tools
                    ),
                    "artifact_type": self._graph.get_node(
                        agent_id
                    ).artifact_type,
                    "completion_condition": self._graph.get_node(
                        agent_id
                    ).completion_condition,
                    "execution_role": (
                        "format"
                        if self.require_format_agent
                        and agent_id == execution.output_agent_id
                        else "output"
                        if agent_id == execution.output_agent_id
                        else "worker"
                    ),
                    "is_output_agent": agent_id == execution.output_agent_id,
                    "upstream_source_ids": (
                        []
                        if call is None
                        else [item.source_agent_id for item in call.request.upstream]
                    ),
                    "tool_receipt_count": len(tool_receipts),
                    "tool_ids": tool_ids,
                    "successful_tool_actions": successful_tool_actions,
                    "failed_tool_actions": failed_tool_actions,
                }
            if expose_content:
                preview = " ".join(artifact.split())
                if len(preview) > 160:
                    preview = preview[:157] + "..."
                item["artifact_preview"] = preview
            agent_artifacts.append(item)
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
                "feedback_mode": self.director_feedback_mode,
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

    def _required_evidence_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Require successful dynamic search/read on the routed Output path.

        This is the small, topology-neutral part of the existing unified
        architecture's ``required_evidence_tool_id`` FINISH admission.  It
        validates measured Tool receipts rather than requiring any named role
        or a fixed Agent ordering.
        """

        tool_id = self.required_evidence_tool_id
        if tool_id is None:
            return None
        output_id = execution.output_agent_id
        if output_id is None or not self._graph.has_node(output_id):
            return "the current Canvas has no selected Output Agent"
        output_node = self._graph.get_node(output_id)
        if tool_id in output_node.allowed_tools:
            return (
                f"Output Agent {output_id!r} must consume a provenance-bearing "
                "upstream artifact and cannot hold the worker retrieval Tool "
                f"{tool_id!r}"
            )
        # DIRECT_REUSE: the TriviaQA runtime binds each artifact to its actual
        # input_artifact_provenance and propagates public Tool receipts through
        # each explicit relation.  FINISH therefore checks what Output consumed,
        # not merely whether some graph ancestor happened to call a Tool.
        output_metadata = execution.output_metadata.get(output_id)
        if not isinstance(output_metadata, Mapping):
            return "the current Output artifact has no execution provenance"
        successful_actions: set[str] = set()
        receipts = output_metadata.get("tool_receipts", ())
        if isinstance(receipts, (list, tuple)):
            for receipt in receipts:
                if not isinstance(receipt, Mapping):
                    continue
                request = receipt.get("request")
                if (
                    receipt.get("tool_id") == tool_id
                    and receipt.get("error_type") is None
                    and isinstance(receipt.get("result"), Mapping)
                    and isinstance(request, Mapping)
                    and request.get("action") in {"search", "read"}
                ):
                    successful_actions.add(str(request["action"]))
        missing = sorted({"search", "read"} - successful_actions)
        if not missing:
            provenance = output_metadata.get("input_artifact_provenance", ())
            routed_receipt_present = isinstance(provenance, (list, tuple)) and any(
                isinstance(message, Mapping)
                and isinstance(message.get("source_agent_id"), str)
                and message.get("source_agent_id") != output_id
                and isinstance(message.get("tool_receipts"), (list, tuple))
                and any(
                    isinstance(receipt, Mapping)
                    and receipt.get("tool_id") == tool_id
                    and receipt.get("error_type") is None
                    for receipt in message.get("tool_receipts", ())
                )
                for message in provenance
            )
            if self.require_evidence_relation and not routed_receipt_present:
                return (
                    f"the Output artifact must actually consume {tool_id} receipts "
                    "from a distinct upstream artifact routed by an explicit "
                    "AgentGraph relation"
                )
            return None
        return (
            f"the routed Output path requires successful dynamic {tool_id} "
            f"Tool receipts for {missing}; preserve the current graph and "
            "repair or augment it with a Tool-capable ReAct Agent"
        )

    def _current_successful_evidence_agent_ids(self) -> Tuple[str, ...]:
        """Return current-revision workers with successful search and read."""

        tool_id = self.required_evidence_tool_id
        execution = self._progressive_execution
        if (
            tool_id is None
            or execution is None
            or self._progressive_execution_revision != self._graph.revision
        ):
            return ()
        successful: list[str] = []
        for agent_id, metadata in execution.output_metadata.items():
            if not self._graph.has_node(agent_id) or not isinstance(
                metadata, Mapping
            ):
                continue
            receipts = metadata.get("tool_receipts", ())
            if not isinstance(receipts, (list, tuple)):
                continue
            actions = {
                str(request["action"])
                for receipt in receipts
                if isinstance(receipt, Mapping)
                and receipt.get("tool_id") == tool_id
                and receipt.get("error_type") is None
                and isinstance(receipt.get("result"), Mapping)
                and isinstance((request := receipt.get("request")), Mapping)
                and request.get("action") in {"search", "read"}
            }
            if actions == {"search", "read"}:
                successful.append(agent_id)
        return tuple(successful)

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
        self._progressive_output_metadata.clear()

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
        if action.action_type is AgentActionType.ADD_SUBGRAPH:
            if not action.agents:
                raise GraphMutationError("add_subgraph action has no Agents")
            if len(action.agents) > self.max_agents_per_subgraph:
                raise GraphMutationError(
                    "add_subgraph agent limit reached: "
                    f"max_agents_per_subgraph={self.max_agents_per_subgraph}"
                )
            new_ids = {item.agent_id for item in action.agents}
            existing_ids = {item.id for item in graph.nodes}
            if new_ids & existing_ids:
                duplicate = sorted(new_ids & existing_ids)[0]
                raise GraphMutationError(
                    f"add_subgraph Agent already exists: {duplicate}"
                )
            if (
                self.max_agents is not None
                and len(existing_ids) + len(new_ids) > self.max_agents
            ):
                raise GraphMutationError(
                    f"agent limit reached: max_agents={self.max_agents}"
                )

            # FlowSteer's structure ADD mutates one candidate Canvas, validates
            # the complete structure, then executes once.  Reuse the existing
            # AgentGraph scalar mutations inside that transaction so their
            # validation and dirty-closure semantics remain unchanged.
            dirty_agents: set[str] = set()
            for item in action.agents:
                dirty_agents |= self._apply_mutation(
                    graph,
                    AgentAction(
                        action_type=AgentActionType.ADD_AGENT,
                        agent_id=item.agent_id,
                        model_id=item.model_id,
                        contract=item.contract,
                        role_family=item.role_family,
                        allowed_tools=item.allowed_tools,
                        execution_mode=item.execution_mode,
                        artifact_type=item.artifact_type,
                        completion_condition=item.completion_condition,
                    ),
                )
            for item in action.relations:
                dirty_agents |= self._apply_mutation(
                    graph,
                    AgentAction(
                        action_type=AgentActionType.SET_RELATION,
                        source_id=item.source_id,
                        target_id=item.target_id,
                        source_to_target=item.source_to_target,
                        target_to_source=item.target_to_source,
                    ),
                )
            if action.output_agent_id is not None:
                dirty_agents |= self._apply_mutation(
                    graph,
                    AgentAction(
                        action_type=AgentActionType.SET_OUTPUT,
                        agent_id=action.output_agent_id,
                    ),
                )
            return graph.dirty_closure(dirty_agents)
        if action.action_type is AgentActionType.ADD_AGENT:
            if action.agent_id is None or action.model_id is None or action.contract is None:
                raise GraphMutationError("add_agent action is incomplete")
            profile_issue = self._execution_profile_issue(
                action.execution_mode or "reasoning",
                action.allowed_tools or (),
            )
            if profile_issue is not None:
                raise GraphMutationError(profile_issue)
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
                    allowed_tools=action.allowed_tools or (),
                    execution_mode=action.execution_mode or "reasoning",
                    artifact_type=action.artifact_type or "text",
                    completion_condition=action.completion_condition,
                )
            )
            return {action.agent_id}
        elif action.action_type is AgentActionType.MODIFY_AGENT:
            if action.agent_id is None:
                raise GraphMutationError("modify_agent action is incomplete")
            current = graph.get_node(action.agent_id)
            mode_value = (
                current.execution_mode.value
                if action.execution_mode is None
                else action.execution_mode
            )
            allowed_tools = (
                current.allowed_tools
                if action.allowed_tools is None
                else action.allowed_tools
            )
            profile_issue = self._execution_profile_issue(
                mode_value,
                allowed_tools,
            )
            if profile_issue is not None:
                raise GraphMutationError(profile_issue)
            graph.modify_agent(
                action.agent_id,
                model_id=action.model_id,
                contract=action.contract,
                role_family=action.role_family,
                allowed_tools=action.allowed_tools,
                execution_mode=action.execution_mode,
                artifact_type=action.artifact_type,
                completion_condition=action.completion_condition,
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
        *,
        partial_execution: Optional[AgentRuntimeResult] = None,
        execution_failure_records: Tuple[object, ...] = (),
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
            partial_execution=partial_execution,
            execution_failure_records=execution_failure_records,
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

    def _execution_profile_issue(
        self,
        execution_mode: str,
        allowed_tools: Sequence[str],
    ) -> Optional[str]:
        profile = (execution_mode, tuple(allowed_tools))
        registered = self.registered_execution_profiles()
        if profile in registered:
            return None
        return (
            "execution_mode/allowed_tools pair is outside the live Runtime "
            f"capability domain: {profile!r}; registered_profiles="
            f"{list(registered)!r}"
        )

    def _validate_graph_execution_profiles(self, graph: AgentGraph) -> None:
        for node in graph.nodes:
            issue = self._execution_profile_issue(
                node.execution_mode.value,
                node.allowed_tools,
            )
            if issue is not None:
                raise AgentWorkflowStateError(
                    f"Agent {node.id!r} has an invalid execution profile: {issue}"
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
