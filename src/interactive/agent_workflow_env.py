"""Transactional progressive Canvas environment for AgentGraph actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Collection, Mapping, Optional, Sequence, Tuple, Union

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
    AgentFailureRecord,
    AgentGateway,
    AgentRuntime,
    AgentRuntimeError,
    AgentRuntimeResult,
)
from .model_registry import ModelRegistry
from .task_dataset import (
    hotpotqa_answer_cardinality_constraint,
    hotpotqa_answer_type_constraint,
    hotpotqa_question_scope,
)


class AgentWorkflowStateError(RuntimeError):
    """Raised for invalid environment construction or restoration."""


_HOTPOTQA_SEMANTIC_PROTOCOL = "hotpotqa_verified_answer_slot_v1"
_PRESERVE_REPAIR_RECOVERY_POLICY = "preserve_diagnose_repair_augment"
_SUPPORTED_SEMANTIC_PROTOCOLS = frozenset({"none", _HOTPOTQA_SEMANTIC_PROTOCOL})
_SUPPORTED_RECOVERY_POLICIES = frozenset(
    {"default", _PRESERVE_REPAIR_RECOVERY_POLICY}
)
_HOTPOTQA_FORMAT_CONTRACT = (
    "copy the supported Verifier candidate character-for-character into the "
    "required answer wrapper"
)

_REASONER_SEMANTIC_FIELDS = (
    "question_scope",
    "answer_slot",
    "evidence_propositions",
    "multi_hop_chain",
    "candidate_answer",
    "evidence",
)
_VERIFIER_SEMANTIC_FIELDS = (
    "candidate_answer",
    "evidence_supported",
    "entity_attribute_binding_correct",
    "alias_binding_correct",
    "answer_type_cardinality_correct",
    "multi_hop_complete",
    "minimal_answer_surface",
    "scope_preserved",
    "verification_status",
)

# Question-answering titles and suffixes are linguistic entity-surface markers,
# not benchmark entities.  The bounded list keeps the deterministic completion
# gate conservative: it catches a strict subspan such as a name with its title
# removed without treating every sentence-initial capitalized word as part of
# the person mention.
_PERSON_TITLE_PATTERN = (
    r"(?:Dr\.?|Doctor|Professor|President|Vice President|Prime Minister|"
    r"Mr\.?|Mrs\.?|Ms\.?|Miss|Sir|Dame|Lord|Lady|King|Queen|Prince|Princess|"
    r"Pope|Saint|St\.?|General|Colonel|Captain|Reverend|Rev\.?|Judge|Justice|"
    r"Chancellor|Governor|Senator|Representative)"
)
_PERSON_SUFFIX_PATTERN = r"(?:Jr\.?|Sr\.?|II|III|IV|V)"


def _answer_protocol_state(answer: str) -> tuple[int, bool, bool]:
    """Return tag count, exact-single-wrapper state, and non-empty state."""

    opening_count = answer.count("<answer>")
    closing_count = answer.count("</answer>")
    match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", answer, flags=re.DOTALL)
    exact_single = opening_count == 1 and closing_count == 1 and match is not None
    non_empty = exact_single and bool(match.group(1).strip())
    return opening_count, exact_single, non_empty


def _canonical_evidence_text(value: str) -> str:
    """Canonicalize typography only for evidence-provenance comparison.

    The retrieved passage remains the provenance authority.  This accepts the
    presentation-only differences observed at the structured-output boundary
    (Unicode compatibility forms, quotation-mark style, and whitespace) while
    still rejecting lexical paraphrases or unsupported facts.
    """

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(
        str.maketrans("", "", "'\"‘’‚‛“”„‟«»‹›")
    )
    return " ".join(normalized.split())


def _evidence_span_matches_read(evidence_span: str, read_text: str) -> bool:
    span = _canonical_evidence_text(evidence_span)
    return bool(span) and span in _canonical_evidence_text(read_text)


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
    execution_failure_records: Tuple[AgentFailureRecord, ...] = ()

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
        required_tool_id: Optional[str] = None,
        allowed_actions: Optional[Sequence[str]] = None,
        semantic_protocol: str = "none",
        recovery_policy: str = "default",
        required_evidence_tool_id: Optional[str] = None,
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
        if required_tool_id is not None and (
            not isinstance(required_tool_id, str) or not required_tool_id.strip()
        ):
            raise AgentWorkflowStateError(
                "required_tool_id must be non-empty text or None"
            )
        if (
            not isinstance(semantic_protocol, str)
            or semantic_protocol not in _SUPPORTED_SEMANTIC_PROTOCOLS
        ):
            raise AgentWorkflowStateError(
                "semantic_protocol must be none or "
                f"{_HOTPOTQA_SEMANTIC_PROTOCOL}"
            )
        if (
            not isinstance(recovery_policy, str)
            or recovery_policy not in _SUPPORTED_RECOVERY_POLICIES
        ):
            raise AgentWorkflowStateError(
                "recovery_policy must be default or "
                f"{_PRESERVE_REPAIR_RECOVERY_POLICY}"
            )
        if required_evidence_tool_id is not None and (
            not isinstance(required_evidence_tool_id, str)
            or not required_evidence_tool_id.strip()
        ):
            raise AgentWorkflowStateError(
                "required_evidence_tool_id must be non-empty text or None"
            )
        if semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL:
            if recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
                raise AgentWorkflowStateError(
                    "hotpotqa_verified_answer_slot_v1 requires "
                    "recovery_policy=preserve_diagnose_repair_augment"
                )
            if required_evidence_tool_id != "qa-retrieval":
                raise AgentWorkflowStateError(
                    "hotpotqa_verified_answer_slot_v1 requires "
                    "required_evidence_tool_id='qa-retrieval'"
                )
        if runtime is not None and runtime.semantic_protocol != semantic_protocol:
            raise AgentWorkflowStateError(
                "runtime and environment must share semantic_protocol; "
                f"runtime={runtime.semantic_protocol!r}, "
                f"environment={semantic_protocol!r}"
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
        self.runtime = runtime or AgentRuntime(  # type: ignore[arg-type]
            model_registry,
            gateway,
            semantic_protocol=semantic_protocol,
        )
        self.execute_on_edit = execute_on_edit
        self.max_agents = max_agents
        self.max_agents_per_subgraph = max_agents_per_subgraph
        self.require_exact_answer_tag = require_exact_answer_tag
        self.require_format_agent = require_format_agent
        self.required_tool_id = (
            None if required_tool_id is None else required_tool_id.strip()
        )
        self.semantic_protocol = semantic_protocol
        self.recovery_policy = recovery_policy
        self.required_evidence_tool_id = (
            None
            if required_evidence_tool_id is None
            else required_evidence_tool_id.strip()
        )
        self.allowed_action_types = resolved_allowed_actions
        self._allowed_action_type_set = frozenset(resolved_allowed_actions)
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
        self._previous_revision_outputs: dict[str, str] = {}
        self._previous_revision_output_metadata: dict[
            str, dict[str, object]
        ] = {}
        self._unresolved_dirty_agents: set[str] = set()
        self._failed_agent_ids: set[str] = set()
        self._diagnosed_unusable_agent_ids: set[str] = set()
        self._react_exhausted_agent_ids: set[str] = set()
        # Runtime-only SkillFlow continuation state.  Canvas snapshots do not
        # serialize Runtime results, but an in-process repair turn must retain
        # the failed Agent's public Action--Observation history and Tool
        # receipts so a contract edit does not repeat retrieval or discard
        # evidence already obtained on the current task.
        self._failure_continuations: dict[str, dict[str, object]] = {}
        self._validate_agent_limit(self._graph)
        partial = self._graph.validate(self.model_registry, require_complete=False)
        if not partial.valid:
            raise AgentWorkflowStateError(self._format_issues(partial))
        semantic_edit_issue = self._semantic_edit_issue_for(self._graph)
        if semantic_edit_issue is not None:
            raise AgentWorkflowStateError(semantic_edit_issue)

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
        return tuple(sorted(self._unresolved_dirty_agents))

    def model_admissible_action_types(self) -> Tuple[str, ...]:
        """Project state-conditioned Canvas actions for the Flow-Director.

        FlowSteer's progressive Canvas exposes the next legal editing boundary
        after each execute-and-feedback step.  Keep the configured action set
        authoritative in ``step`` while removing actions that cannot change or
        terminate the current public state from the next model observation.
        """

        finish_admitted = self.finish_admissibility().get("admissible") is True
        if (
            self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
            and self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY
            and finish_admitted
            and AgentActionType.FINISH.value in self._allowed_action_type_set
        ):
            # Once the current revision has a verified semantic lineage and a
            # valid terminal artifact, further edits can only endanger the
            # already completed answer.  FlowSteer's action mask still asks the
            # Director to emit the explicit terminal action; it does not finish
            # automatically.
            return (AgentActionType.FINISH.value,)

        mandatory_repair_ids = self._mandatory_repair_agent_ids()
        if (
            mandatory_repair_ids
            and AgentActionType.MODIFY_AGENT.value
            in self._allowed_action_type_set
        ):
            # A typed Runtime failure or a revision-local semantic-artifact
            # failure identifies an existing Agent that can still be repaired.
            # Keep FlowSteer's action mask on that measured repair boundary;
            # augmentation becomes available only after repair succeeds or an
            # explicit ``node_unusable`` receipt admits replacement takeover.
            return (AgentActionType.MODIFY_AGENT.value,)

        node_count = len(self._graph.nodes)
        node_ids = tuple(node.id for node in self._graph.nodes)
        can_add = self.max_agents is None or node_count < self.max_agents
        deletable_ids = tuple(
            node_id
            for node_id in node_ids
            if self._delete_admission_issue(node_id) is None
        )
        can_delete = bool(deletable_ids)
        active_semantic_lineage = set(self._active_semantic_lineage_ids())
        output_target_ids = (
            ()
            if self._graph.output_agent_id in active_semantic_lineage
            else tuple(
                node.id
                for node in self._graph.nodes
                if (
                    self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL
                    or (node.role_family or "").casefold() == "format"
                )
                and node.id != self._graph.output_agent_id
            )
        )
        modifiable_ids = self._model_admissible_modify_agent_ids()
        can_set_output = bool(output_target_ids)
        can_set_relation = bool(self._model_admissible_relation_candidates())
        admitted: list[str] = []
        for action_type in self.allowed_action_types:
            if action_type == AgentActionType.ADD_SUBGRAPH.value and can_add:
                admitted.append(action_type)
            elif action_type == AgentActionType.MODIFY_AGENT.value and modifiable_ids:
                admitted.append(action_type)
            elif action_type == AgentActionType.DELETE_AGENT.value and can_delete:
                admitted.append(action_type)
            elif action_type == AgentActionType.SET_RELATION.value and can_set_relation:
                admitted.append(action_type)
            elif action_type == AgentActionType.SET_OUTPUT.value and can_set_output:
                admitted.append(action_type)
            elif action_type == AgentActionType.FINISH.value and finish_admitted:
                admitted.append(action_type)
        return tuple(admitted)

    def _model_admissible_relation_candidates(self) -> list[dict[str, object]]:
        """Return exact non-self, non-no-op relation edits accepted by Canvas."""

        node_ids = [node.id for node in self._graph.nodes]
        active_lineage = self._active_semantic_lineage_ids()
        protected_edges = tuple(zip(active_lineage, active_lineage[1:]))
        candidates: list[dict[str, object]] = []
        for source_index, source_id in enumerate(node_ids):
            for target_id in node_ids[source_index + 1 :]:
                previous = self._graph.relation_bits(source_id, target_id)
                for source_to_target, target_to_source in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    if (
                        previous.source_to_target == source_to_target
                        and previous.target_to_source == target_to_source
                    ):
                        continue
                    candidate = self._graph.fork()
                    candidate.set_relation(
                        source_id,
                        target_id,
                        source_to_target,
                        target_to_source,
                    )
                    validation = candidate.validate(
                        self.model_registry,
                        require_complete=False,
                    )
                    if not validation.valid:
                        continue
                    if self._semantic_edit_issue_for(candidate) is not None:
                        continue
                    if (
                        candidate.output_agent_id is not None
                        and self._uses_format_agent_protocol()
                        and self._format_agent_issue_for(candidate) is not None
                    ):
                        continue
                    if any(
                        target_id
                        not in self._directed_successors(candidate, source_id)
                        for source_id, target_id in protected_edges
                    ):
                        continue
                    candidates.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_to_target": source_to_target,
                            "target_to_source": target_to_source,
                        }
                    )
        return candidates

    def _model_admissible_modify_agent_ids(self) -> Tuple[str, ...]:
        """Exclude an already verified semantic lineage from repair targets."""

        node_ids = tuple(node.id for node in self._graph.nodes)
        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return node_ids
        mandatory_repair_ids = self._mandatory_repair_agent_ids()
        if mandatory_repair_ids:
            return mandatory_repair_ids
        measured_failed = self._failed_agent_ids.intersection(node_ids)
        if measured_failed:
            # AgentRuntime distinguishes the Agent that raised a typed failure
            # from blocked/pending descendants.  FlowSteer's next Canvas edit
            # should repair every measured root failure, not mutate downstream
            # Agents that merely could not execute because their input is absent.
            return tuple(
                node_id for node_id in node_ids if node_id in measured_failed
            )
        protected = set(self._active_semantic_lineage_ids())
        responsible = set(self._unresolved_dirty_agents)
        responsible.update(self._terminal_unreachable_agent_ids())
        return tuple(
            node_id
            for node_id in node_ids
            if node_id not in protected or node_id in responsible
        )

    def _semantic_artifact_repair_agent_ids(self) -> Tuple[str, ...]:
        """Return the existing Agent responsible for a terminal semantic fault.

        This uses only the current Canvas and progressive Runtime artifacts.  It
        never reads Ground Truth or evaluator state.  Structural and format
        lineage faults remain relation/output repairs; only a complete executed
        terminal lineage with an invalid semantic artifact enters this domain.
        """

        if (
            self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
        ):
            return ()
        validation = self._graph.validate(
            self.model_registry,
            require_complete=True,
        )
        if not validation.valid or self._format_agent_issue_for(self._graph) is not None:
            return ()
        execution = self._cached_progressive_execution()
        if execution is None or execution.final_answer is None:
            return ()
        semantic_issue = self._semantic_protocol_issue(execution)
        if semantic_issue is None:
            return ()
        attribution = self._semantic_repair_attribution(semantic_issue)
        if attribution is None:
            return ()
        agent_id = attribution.get("responsible_agent_id")
        if not isinstance(agent_id, str) or not self._graph.has_node(agent_id):
            return ()
        return (agent_id,)

    def _mandatory_repair_agent_ids(self) -> Tuple[str, ...]:
        """Project the repair-first Agent domain for the current Canvas state."""

        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return ()
        node_ids = tuple(node.id for node in self._graph.nodes)
        repairable_failed = (
            self._failed_agent_ids - self._diagnosed_unusable_agent_ids
        ).intersection(node_ids)
        if repairable_failed:
            return tuple(
                node_id for node_id in node_ids if node_id in repairable_failed
            )
        return self._semantic_artifact_repair_agent_ids()

    def _admissible_augmentation_role_families(self) -> Tuple[str, ...]:
        """Return HotpotQA roles that may be added at this recovery boundary."""

        role_families = (
            "reasoner",
            "verifier",
            "format",
            "evidence_retriever",
            "repair",
        )
        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return role_families
        admitted: list[str] = []
        for role_family in role_families:
            if role_family not in {"reasoner", "verifier", "format"}:
                admitted.append(role_family)
                continue
            existing_ids = tuple(
                node.id
                for node in self._graph.nodes
                if (node.role_family or "").casefold() == role_family
            )
            if not existing_ids or all(
                agent_id in self._diagnosed_unusable_agent_ids
                for agent_id in existing_ids
            ):
                admitted.append(role_family)
        return tuple(admitted)

    def model_admissible_action_targets(self) -> dict[str, object]:
        """Project exact current Canvas target domains for constrained sampling.

        This is a read-only FlowSteer legality projection.  It does not select
        the next action, repair an invalid sample, or prescribe a topology.
        """

        admitted = set(self.model_admissible_action_types())
        node_ids = [node.id for node in self._graph.nodes]
        targets: dict[str, object] = {}
        if AgentActionType.ADD_SUBGRAPH.value in admitted:
            remaining = (
                self.max_agents_per_subgraph
                if self.max_agents is None
                else min(
                    self.max_agents_per_subgraph,
                    max(self.max_agents - len(node_ids), 0),
                )
            )
            targets[AgentActionType.ADD_SUBGRAPH.value] = {
                "min_new_agents": 1,
                "max_new_agents": remaining,
                "existing_agent_ids": node_ids,
                **(
                    {
                        "semantic_protocol": self.semantic_protocol,
                        "existing_agents": [
                            {
                                "agent_id": node.id,
                                "role_family": node.role_family,
                            }
                            for node in self._graph.nodes
                        ],
                        "current_output_agent_id": self._graph.output_agent_id,
                        "output_role_family": "format",
                        "required_agent_fields": [
                            "agent_id",
                            "model_id",
                            "contract",
                            "role_family",
                            "allowed_tools",
                            "execution_mode",
                        ],
                        "model_ids": list(self.model_registry.model_ids),
                        "role_constraints": {
                            "reasoner": {
                                "execution_modes": ["react"],
                                "allowed_tools": [
                                    [self.required_evidence_tool_id]
                                ],
                            },
                            "verifier": {
                                "execution_modes": ["reasoning"],
                                "allowed_tools": [[]],
                            },
                            "format": {
                                "execution_modes": ["reasoning"],
                                "allowed_tools": [[]],
                                "contracts": [_HOTPOTQA_FORMAT_CONTRACT],
                            },
                            "evidence_retriever": {
                                "execution_modes": ["react"],
                                "allowed_tools": [
                                    [self.required_evidence_tool_id]
                                ],
                            },
                            "repair": {
                                "execution_modes": ["reasoning"],
                                "allowed_tools": [[]],
                            },
                        },
                        "admitted_new_role_families": list(
                            self._admissible_augmentation_role_families()
                        ),
                        "endpoint_scope": {
                            "relation_endpoint_sources": [
                                "existing_agent_ids",
                                "same_action_agent_ids",
                            ],
                            "output_agent_id_sources": [
                                "existing_agent_ids",
                                "same_action_agent_ids",
                            ],
                        },
                    }
                    if self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
                    else {}
                ),
            }
        if AgentActionType.MODIFY_AGENT.value in admitted:
            modifiable_node_ids = list(self._model_admissible_modify_agent_ids())
            base_mutable_fields = [
                "model_id",
                "contract",
                "artifact_type",
                "completion_condition",
            ]
            if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
                base_mutable_fields[2:2] = [
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                ]
            measured_failed_ids = self._failed_agent_ids.intersection(node_ids)
            responsible_ids = set(measured_failed_ids)
            if not measured_failed_ids:
                responsible_ids.update(self._unresolved_dirty_agents)
                responsible_ids.update(self._terminal_unreachable_agent_ids())
            if (
                not measured_failed_ids
                and self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
            ):
                execution = self._cached_progressive_execution()
                if execution is not None:
                    semantic_issue = self._semantic_protocol_issue(execution)
                    if semantic_issue is not None:
                        attribution = self._semantic_repair_attribution(
                            semantic_issue
                        )
                        if attribution is not None:
                            responsible = attribution.get("responsible_agent_id")
                            if isinstance(responsible, str):
                                responsible_ids.add(responsible)
            per_agent_mutable_fields = {
                agent_id: [
                    field
                    for field in (
                        ["contract", "completion_condition"]
                        if (
                            self.semantic_protocol
                            == _HOTPOTQA_SEMANTIC_PROTOCOL
                            and agent_id in self._react_exhausted_agent_ids
                        )
                        else base_mutable_fields
                    )
                    if not (
                        self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
                        and (
                            self._graph.get_node(agent_id).role_family or ""
                        ).casefold()
                        == "format"
                        and field == "contract"
                    )
                ]
                for agent_id in modifiable_node_ids
            }
            mutable_fields = [
                field
                for field in base_mutable_fields
                if any(
                    field in fields
                    for fields in per_agent_mutable_fields.values()
                )
            ]
            targets[AgentActionType.MODIFY_AGENT.value] = {
                "agent_ids": modifiable_node_ids,
                "failed_agent_ids": sorted(self._failed_agent_ids),
                "responsible_agent_ids": sorted(
                    responsible_ids.intersection(node_ids)
                ),
                "mutable_fields": mutable_fields,
                "per_agent_candidates": [
                    {
                        "agent_id": agent_id,
                        "mutable_fields": per_agent_mutable_fields[agent_id],
                        "discrete_value_domains": (
                            {
                                "model_id": [
                                    model_id
                                    for model_id in self.model_registry.model_ids
                                    if model_id
                                    != self._graph.get_node(agent_id).model_id
                                ]
                            }
                            if "model_id" in per_agent_mutable_fields[agent_id]
                            else {}
                        ),
                    }
                    for agent_id in modifiable_node_ids
                ],
            }
        if AgentActionType.DELETE_AGENT.value in admitted:
            targets[AgentActionType.DELETE_AGENT.value] = {
                "agent_ids": [
                    node_id
                    for node_id in node_ids
                    if self._delete_admission_issue(node_id) is None
                ]
            }
        if AgentActionType.SET_RELATION.value in admitted:
            targets[AgentActionType.SET_RELATION.value] = {
                "source_agent_ids": node_ids,
                "target_agent_ids": node_ids,
                "endpoints_must_differ": True,
                "candidates": self._model_admissible_relation_candidates(),
            }
        if AgentActionType.SET_OUTPUT.value in admitted:
            active_semantic_lineage = set(self._active_semantic_lineage_ids())
            targets[AgentActionType.SET_OUTPUT.value] = {
                "agent_ids": (
                    []
                    if self._graph.output_agent_id in active_semantic_lineage
                    else [
                        node.id
                        for node in self._graph.nodes
                        if (
                            self.semantic_protocol
                            != _HOTPOTQA_SEMANTIC_PROTOCOL
                            or (node.role_family or "").casefold() == "format"
                        )
                        and node.id != self._graph.output_agent_id
                    ]
                ),
                "current_output_agent_id": self._graph.output_agent_id,
            }
        if AgentActionType.FINISH.value in admitted:
            targets[AgentActionType.FINISH.value] = {
                "admissible": True,
                "submission_semantics": "explicit_finish",
            }
        return targets

    def reset(self, problem: str, graph: Optional[AgentGraph] = None) -> AgentWorkflowSnapshot:
        if not isinstance(problem, str) or not problem.strip():
            raise AgentWorkflowStateError("problem must be a non-empty string")
        candidate = graph.fork() if graph is not None else AgentGraph()
        self._validate_agent_limit(candidate)
        validation = candidate.validate(self.model_registry, require_complete=False)
        if not validation.valid:
            raise AgentWorkflowStateError(self._format_issues(validation))
        semantic_edit_issue = self._semantic_edit_issue_for(candidate)
        if semantic_edit_issue is not None:
            raise AgentWorkflowStateError(semantic_edit_issue)
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
        semantic_edit_issue = self._semantic_edit_issue_for(graph)
        if semantic_edit_issue is not None:
            raise AgentWorkflowStateError(semantic_edit_issue)
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
            max_agents_per_subgraph=self.max_agents_per_subgraph,
            require_exact_answer_tag=self.require_exact_answer_tag,
            require_format_agent=self.require_format_agent,
            required_tool_id=self.required_tool_id,
            allowed_actions=self.allowed_action_types,
            semantic_protocol=self.semantic_protocol,
            recovery_policy=self.recovery_policy,
            required_evidence_tool_id=self.required_evidence_tool_id,
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
        if action.action_type.value not in self._allowed_action_type_set:
            return self._reject_after_count(
                action,
                "action rejected: action type is outside the configured Canvas "
                f"action set {list(self.allowed_action_types)!r}",
            )
        preservation_issue = self._preservation_admission_issue(action)
        if preservation_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + preservation_issue,
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
            required_tool_issue = self.required_tool_issue()
            if required_tool_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + required_tool_issue,
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
                        prior_failure_metadata=self._failure_continuations,
                        format_output_agent=self._uses_format_agent_protocol(),
                    )
                except AgentRuntimeError as exc:
                    if exc.partial_result is not None:
                        self._progressive_outputs = dict(exc.partial_result.outputs)
                        self._progressive_output_metadata = {
                            agent_id: dict(metadata)
                            for agent_id, metadata in (
                                exc.partial_result.output_metadata.items()
                            )
                        }
                    current_agent_ids = {node.id for node in self._graph.nodes}
                    completed_agent_ids = (
                        set()
                        if exc.partial_result is None
                        else (
                            set(exc.partial_result.outputs)
                            if self.recovery_policy
                            == _PRESERVE_REPAIR_RECOVERY_POLICY
                            else set(exc.partial_result.executed_agent_ids)
                        )
                    )
                    self._mark_agents_recovered(completed_agent_ids)
                    self._unresolved_dirty_agents = (
                        current_agent_ids - completed_agent_ids
                    )
                    self._record_failure_state(
                        exc.failure_records,
                        current_agent_ids=current_agent_ids,
                    )
                    return self._reject_after_count(
                        action,
                        "cannot finish: " + self._execution_error_feedback(exc),
                        partial_execution=exc.partial_result,
                        execution_failure_records=exc.failure_records,
                    )
                self._progressive_outputs = dict(execution.outputs)
                self._progressive_output_metadata = {
                    agent_id: dict(metadata)
                    for agent_id, metadata in execution.output_metadata.items()
                }
                self._progressive_execution = execution
                self._progressive_execution_revision = self._graph.revision
                self._unresolved_dirty_agents.clear()
                self._clear_failure_state()
            environment_terminal_issue = self._environment_terminal_issue(execution)
            if environment_terminal_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + environment_terminal_issue,
                    execution=execution,
                    execution_reused=execution_reused,
                )
            if execution.final_answer is None:
                return self._reject_after_count(
                    action,
                    "cannot finish: Format Agent produced no terminal artifact",
                )
            semantic_issue = self._semantic_protocol_issue(execution)
            if semantic_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + semantic_issue,
                    execution=execution,
                    execution_reused=execution_reused,
                )
            terminal_issue = self._terminal_validation_error(execution.final_answer)
            if terminal_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + terminal_issue,
                )
            self._finished = True
            self._unresolved_dirty_agents.clear()
            self._clear_failure_state()
            self._previous_revision_outputs.clear()
            self._previous_revision_output_metadata.clear()
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
        semantic_edit_issue = self._semantic_edit_issue_for(candidate)
        if semantic_edit_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + semantic_edit_issue,
            )
        contract_obligation_issue = self._contract_obligation_issue(action)
        if contract_obligation_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + contract_obligation_issue,
            )
        try:
            # Reuse the Runtime's execution-contract boundary before the
            # candidate Canvas revision is committed.  FlowSteer's edit then
            # remains transactional: an invalid execution declaration is
            # rejected without executing or persisting the candidate graph.
            self.runtime.validate_execution_contracts(candidate.nodes)
        except AgentRuntimeError as exc:
            return self._reject_after_count(
                action,
                "edit rejected: execution contract invalid: "
                + " ".join(str(exc).split()),
            )
        if (
            action.action_type in {
                AgentActionType.SET_OUTPUT,
                AgentActionType.ADD_SUBGRAPH,
                AgentActionType.SET_RELATION,
            }
            and candidate.output_agent_id is not None
            and self._uses_format_agent_protocol()
        ):
            format_issue = self._format_agent_issue_for(candidate)
            if format_issue is not None:
                return self._reject_after_count(
                    action,
                    "edit rejected: " + format_issue,
                )

        self._graph = candidate
        current_agent_ids = {node.id for node in self._graph.nodes}
        self._retain_current_failure_state(current_agent_ids)
        self._unresolved_dirty_agents = (
            self._unresolved_dirty_agents & current_agent_ids
        ) | (set(dirty_agents) & current_agent_ids)
        self._invalidate_progressive_outputs(
            self._unresolved_dirty_agents,
            current_agent_ids=current_agent_ids,
        )
        execution = None
        partial_execution = None
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
                        prior_output_metadata=self._progressive_output_metadata,
                        prior_failure_metadata=self._failure_continuations,
                        dirty_agents=self._unresolved_dirty_agents,
                        format_output_agent=self._uses_format_agent_protocol(),
                    )
                except AgentRuntimeError as exc:
                    # FlowSteer's progressive Canvas treats execution as edit
                    # feedback.  A provider/runtime failure must not roll back
                    # a structurally valid edit or abort the Director rollout.
                    execution_error = exc
                    partial_execution = exc.partial_result
                    if partial_execution is not None:
                        self._progressive_outputs = dict(partial_execution.outputs)
                        self._progressive_output_metadata = {
                            agent_id: dict(metadata)
                            for agent_id, metadata in (
                                partial_execution.output_metadata.items()
                            )
                        }
                        self._unresolved_dirty_agents.difference_update(
                            partial_execution.outputs
                            if self.recovery_policy
                            == _PRESERVE_REPAIR_RECOVERY_POLICY
                            else partial_execution.executed_agent_ids
                        )
                    self._unresolved_dirty_agents.update(
                        agent_id
                        for agent_id in exc.pending_agent_ids
                        if agent_id in current_agent_ids
                    )
                    self._failed_agent_ids.difference_update(
                        ()
                        if partial_execution is None
                        else partial_execution.outputs
                    )
                    self._diagnosed_unusable_agent_ids.difference_update(
                        ()
                        if partial_execution is None
                        else partial_execution.outputs
                    )
                    self._mark_agents_recovered(
                        ()
                        if partial_execution is None
                        else partial_execution.outputs
                    )
                    self._record_failure_state(
                        exc.failure_records,
                        current_agent_ids=current_agent_ids,
                    )
                else:
                    self._progressive_outputs = dict(execution.outputs)
                    self._progressive_output_metadata = {
                        agent_id: dict(metadata)
                        for agent_id, metadata in execution.output_metadata.items()
                    }
                    self._progressive_execution = execution
                    self._progressive_execution_revision = self._graph.revision
                    # A HotpotQA Verifier/Formatter can be structurally present
                    # while its semantic input is not yet routable.  Runtime
                    # deferral is successful progressive execution, not Agent
                    # failure; keep only those unmaterialized nodes unresolved.
                    self._unresolved_dirty_agents = (
                        current_agent_ids - set(execution.outputs)
                    )
                    self._clear_failure_state()
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
            partial_execution=partial_execution,
            execution_failure_records=(
                () if execution_error is None else execution_error.failure_records
            ),
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
            if self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY:
                state = json.dumps(
                    self.recovery_state(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                return f"{feedback}; recovery_state={state}"
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
                "deferred_agent_ids": list(execution.deferred_agent_ids),
                "topology": self._graph.topology_statistics(),
                "output_inbox": output_inbox,
                "agent_artifacts": agent_artifacts,
                **(
                    {"recovery_state": self.recovery_state()}
                    if self.recovery_policy
                    == _PRESERVE_REPAIR_RECOVERY_POLICY
                    else {}
                ),
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

    def finish_admissibility(self) -> dict[str, object]:
        """Return the revision-local explicit-FINISH admission state.

        FlowSteer exposes terminal constraint feedback before accepting
        ``FINISH`` and SkillFlow validates a completion before marking a
        bounded episode terminal.  This read-only projection never executes a
        model and never substitutes an intermediate artifact for an explicit
        Director submission.
        """

        validation = self._graph.validate(
            self.model_registry,
            require_complete=True,
        )
        if not validation.valid:
            result: dict[str, object] = {
                "admissible": False,
                "stage": "graph_validation",
                "reason": self._format_issues(validation),
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "agent_ids": list(issue.agent_ids),
                    }
                    for issue in validation.issues
                ],
            }
            if self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY:
                result["recovery_state"] = self.recovery_state()
                attribution = self._semantic_repair_attribution(
                    result["reason"],
                    stage="graph_validation",
                    validation=validation,
                )
                if attribution is not None:
                    result["failure_attribution"] = attribution
            execution = self._cached_progressive_execution()
            if (
                execution is not None
                and self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
            ):
                semantic_issue = self._semantic_protocol_issue(execution)
                if semantic_issue is not None:
                    result["semantic_lineage_diagnostic"] = semantic_issue
                    # Graph validation is the terminal boundary here.  Retain
                    # the semantic issue as a secondary diagnostic, but never
                    # overwrite the structural repair target with a later
                    # semantic attribution.
            return result
        format_issue = self.format_agent_issue()
        if format_issue is not None:
            result = {
                "admissible": False,
                "stage": "format_lineage",
                "reason": format_issue,
            }
            if self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY:
                result["recovery_state"] = self.recovery_state()
                attribution = self._semantic_repair_attribution(
                    format_issue,
                    stage="format_lineage",
                )
                if attribution is not None:
                    result["failure_attribution"] = attribution
            return result
        required_tool_issue = self.required_tool_issue()
        if required_tool_issue is not None:
            return {
                "admissible": False,
                "stage": "required_tool",
                "reason": required_tool_issue,
            }
        execution = self._cached_progressive_execution()
        if execution is None or execution.final_answer is None:
            result = {
                "admissible": False,
                "stage": "execution",
                "reason": "current graph revision has no successful Output artifact",
            }
            if self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY:
                result["recovery_state"] = self.recovery_state()
                attribution = self._semantic_repair_attribution(
                    result["reason"],
                    stage="execution",
                )
                if attribution is not None:
                    result["failure_attribution"] = attribution
            return result
        environment_issue = self._environment_terminal_issue(execution)
        if environment_issue is not None:
            return {
                "admissible": False,
                "stage": "environment_terminal",
                "reason": environment_issue,
            }
        semantic_issue = self._semantic_protocol_issue(execution)
        if semantic_issue is not None:
            result: dict[str, object] = {
                "admissible": False,
                "stage": "semantic_protocol",
                "reason": semantic_issue,
            }
            repair = self._semantic_repair_attribution(semantic_issue)
            if repair is not None:
                result["failure_attribution"] = repair
            return result
        terminal_issue = self._terminal_validation_error(execution.final_answer)
        if terminal_issue is not None:
            return {
                "admissible": False,
                "stage": "terminal_protocol",
                "reason": terminal_issue,
            }
        return {
            "admissible": True,
            "graph_revision": self._graph.revision,
            "submission_semantics": "explicit_finish",
        }

    def _semantic_repair_attribution(
        self,
        reason: str,
        *,
        stage: Optional[str] = None,
        validation: Optional[GraphValidationResult] = None,
    ) -> Optional[dict[str, object]]:
        """Project the responsible semantic stage for the next Canvas repair."""

        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return None
        formatter_ids = tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold() == "format"
        )
        formatter_id = self._graph.output_agent_id
        verifier_id: Optional[str] = None
        reasoner_id: Optional[str] = None
        if formatter_id is not None:
            predecessors = self._graph.directed_predecessors(formatter_id)
            if len(predecessors) == 1:
                verifier_id = predecessors[0]
                reasoner_predecessors = tuple(
                    agent_id
                    for agent_id in self._graph.directed_predecessors(verifier_id)
                    if (
                        self._graph.get_node(agent_id).role_family or ""
                    ).casefold()
                    == "reasoner"
                )
                if len(reasoner_predecessors) == 1:
                    reasoner_id = reasoner_predecessors[0]
        issue_codes = (
            frozenset()
            if validation is None
            else frozenset(issue.code for issue in validation.issues)
        )
        unreachable_ids = (
            ()
            if validation is None
            else tuple(
                sorted(
                    {
                        agent_id
                        for issue in validation.issues
                        if issue.code == "cannot_reach_output"
                        for agent_id in issue.agent_ids
                    }
                )
            )
        )
        if unreachable_ids:
            target_id = unreachable_ids[0]
            role_family = (
                self._graph.get_node(target_id).role_family
                if self._graph.has_node(target_id)
                else None
            )
            responsible_constraint = "terminal_reachability"
            preferred_actions = ["set_relation", "modify_agent", "add_subgraph"]
        elif (
            self._graph.output_agent_id is None
            or "output_agent_count" in issue_codes
            or "unknown_output_agent" in issue_codes
        ):
            target_id = formatter_ids[0] if len(formatter_ids) == 1 else None
            role_family = "format"
            responsible_constraint = (
                "format_output_assignment"
                if formatter_ids
                else "format_agent_presence_and_output_assignment"
            )
            preferred_actions = (
                ["set_output", "modify_agent", "add_subgraph"]
                if formatter_ids
                else ["add_subgraph", "modify_agent", "set_output"]
            )
        elif stage == "execution" and formatter_id is not None:
            target_id, role_family = formatter_id, "format"
            responsible_constraint = "output_artifact"
            preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        elif stage == "format_lineage" or reason.startswith("Format"):
            target_id = formatter_id or (
                formatter_ids[0] if len(formatter_ids) == 1 else None
            )
            role_family = "format"
            responsible_constraint = "format_lineage"
            preferred_actions = ["set_relation", "set_output", "modify_agent"]
        if reason.startswith("Reasoner"):
            target_id, role_family = reasoner_id, "reasoner"
            responsible_constraint = "reasoner_semantic_artifact"
            preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        elif reason.startswith("Verifier"):
            target_id, role_family = verifier_id, "verifier"
            responsible_constraint = "verifier_semantic_artifact"
            preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        elif reason.startswith("Format"):
            target_id, role_family = formatter_id, "format"
            responsible_constraint = "format_lineage"
            preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        elif not any(
            (
                unreachable_ids,
                self._graph.output_agent_id is None,
                "output_agent_count" in issue_codes,
                "unknown_output_agent" in issue_codes,
                stage in {"execution", "format_lineage"},
            )
        ):
            return None
        preserved = [
            agent_id
            for agent_id in (reasoner_id, verifier_id, formatter_id)
            if agent_id is not None and self._has_successful_artifact(agent_id)
        ]
        result: dict[str, object] = {
            "responsible_constraint": responsible_constraint,
            "responsible_role_family": role_family,
            "responsible_agent_ids": list(unreachable_ids),
            "format_target_agent_ids": list(formatter_ids),
            "preserve_agent_ids": preserved,
            "preferred_action_order": preferred_actions,
            "delete_allowed_before_replacement_takeover": False,
        }
        if target_id is not None:
            result["responsible_agent_id"] = target_id
        return result

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
        self._previous_revision_outputs.clear()
        self._previous_revision_output_metadata.clear()
        self._unresolved_dirty_agents.clear()
        self._clear_failure_state()

    def _retain_current_failure_state(self, current_agent_ids: set[str]) -> None:
        """Drop failure diagnoses for nodes no longer present on the Canvas."""

        self._failed_agent_ids.intersection_update(current_agent_ids)
        self._diagnosed_unusable_agent_ids.intersection_update(current_agent_ids)
        self._react_exhausted_agent_ids.intersection_update(current_agent_ids)
        self._failure_continuations = {
            agent_id: metadata
            for agent_id, metadata in self._failure_continuations.items()
            if agent_id in current_agent_ids
        }

    @staticmethod
    def _failure_continuation_candidate(
        record: AgentFailureRecord,
    ) -> Optional[dict[str, object]]:
        """Return adapter-published continuation state for one failed phase."""

        result: dict[str, object] = {"execution_phase": record.phase.value}
        for field_name in ("react_trace", "tool_receipts"):
            raw_value = record.metadata.get(field_name, ())
            if not isinstance(raw_value, (list, tuple)):
                continue
            public_items = [
                dict(item) for item in raw_value if isinstance(item, Mapping)
            ]
            if public_items:
                result[field_name] = public_items
        return result if len(result) > 1 else None

    @staticmethod
    def _failure_continuation_weight(metadata: Mapping[str, object]) -> tuple[int, int]:
        """Prefer the most advanced public continuation for an Agent."""

        trace = metadata.get("react_trace", ())
        receipts = metadata.get("tool_receipts", ())
        return (
            len(trace) if isinstance(trace, (list, tuple)) else 0,
            len(receipts) if isinstance(receipts, (list, tuple)) else 0,
        )

    def _mark_agents_recovered(self, agent_ids: Collection[str]) -> None:
        """Clear failure-only state after those Agents produced artifacts."""

        recovered = set(agent_ids)
        self._failed_agent_ids.difference_update(recovered)
        self._diagnosed_unusable_agent_ids.difference_update(recovered)
        self._react_exhausted_agent_ids.difference_update(recovered)
        for agent_id in recovered:
            self._failure_continuations.pop(agent_id, None)

    def _record_failure_state(
        self,
        records: Sequence[AgentFailureRecord],
        *,
        current_agent_ids: set[str],
    ) -> None:
        """Record measured failures without inferring node unusability.

        A provider request, ReAct exhaustion, Tool error, or contract error is a
        repair diagnosis, not evidence that the Agent node itself is unusable.
        Deletion eligibility requires the execution adapter's explicit typed
        ``node_unusable=true`` receipt and remains separately gated on artifact
        takeover.
        """

        self._retain_current_failure_state(current_agent_ids)
        recorded_agent_ids: set[str] = set()
        react_exhausted_agent_ids: set[str] = set()
        for record in records:
            if record.agent_id not in current_agent_ids:
                continue
            recorded_agent_ids.add(record.agent_id)
            self._failed_agent_ids.add(record.agent_id)
            category, _, _ = self._execution_failure_diagnosis(record)
            if category == "react_turn_exhaustion":
                react_exhausted_agent_ids.add(record.agent_id)
            if record.metadata.get("node_unusable") is True:
                self._diagnosed_unusable_agent_ids.add(record.agent_id)
            else:
                self._diagnosed_unusable_agent_ids.discard(record.agent_id)
            continuation = self._failure_continuation_candidate(record)
            if continuation is not None:
                current = self._failure_continuations.get(record.agent_id)
                if current is None or self._failure_continuation_weight(
                    continuation
                ) >= self._failure_continuation_weight(current):
                    self._failure_continuations[record.agent_id] = continuation
        self._react_exhausted_agent_ids.difference_update(recorded_agent_ids)
        self._react_exhausted_agent_ids.update(react_exhausted_agent_ids)

    def _clear_failure_state(self) -> None:
        self._failed_agent_ids.clear()
        self._diagnosed_unusable_agent_ids.clear()
        self._react_exhausted_agent_ids.clear()
        self._failure_continuations.clear()

    def _invalidate_progressive_outputs(
        self,
        dirty_agent_ids: set[str],
        *,
        current_agent_ids: set[str],
    ) -> None:
        """Invalidate revision-local cache entries before executing an edit."""

        self._progressive_execution = None
        self._progressive_execution_revision = None
        for agent_id in tuple(self._progressive_outputs):
            if agent_id in dirty_agent_ids or agent_id not in current_agent_ids:
                artifact = self._progressive_outputs.get(agent_id)
                if isinstance(artifact, str) and artifact.strip():
                    self._previous_revision_outputs[agent_id] = artifact
                    metadata = self._progressive_output_metadata.get(agent_id)
                    if isinstance(metadata, Mapping):
                        self._previous_revision_output_metadata[agent_id] = dict(
                            metadata
                        )
                self._progressive_outputs.pop(agent_id, None)
                self._progressive_output_metadata.pop(agent_id, None)

    def format_agent_issue(self) -> Optional[str]:
        """Return the terminal Format-Agent constraint that is still unmet.

        FlowSteer's ``Format`` operator consumes one completed solution and
        performs extraction only.  The free-AgentGraph adaptation keeps
        ``role_family`` as metadata rather than an Operator enum, while the
        factual-QA terminal protocol reserves ``format`` for the distinct
        Output Agent and requires one routed semantic-answer artifact.
        """

        return self._format_agent_issue_for(self._graph)

    def required_tool_issue(self) -> Optional[str]:
        """Return the executor-capability constraint still unmet at FINISH.

        FlowSteer's terminal validation rejects an incomplete Workflow before
        evaluation.  Interactive RAGEN tasks likewise require one stateful
        environment actor; a prose-only graph cannot produce the native replay
        trace consumed by the terminal evaluator.  This constraint fixes only
        the required capability and leaves model, role, topology, and all other
        Agents to the Director search space.
        """

        if self.required_tool_id is None:
            return None
        owners = tuple(
            node.id
            for node in self._graph.nodes
            if node.execution_mode.value == "react"
            and node.allowed_tools == (self.required_tool_id,)
        )
        if len(owners) == 1:
            return None
        return (
            "AgentGraph must contain exactly one ReAct environment actor with "
            f"allowed_tools=['{self.required_tool_id}']; found {len(owners)}. "
            "Add or modify the required executor before retrying FINISH"
        )

    def _environment_terminal_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Require the stateful environment actor's measured terminal receipt.

        SkillFlow marks an interactive rollout complete only after a terminal
        environment observation.  The AgentGraph Output Agent may be a later
        consumer, so terminality is read from the unique environment actor's
        execution metadata rather than inferred from free-form output text.
        """

        if self.required_tool_id is None:
            return None
        owners = tuple(
            node.id
            for node in self._graph.nodes
            if node.execution_mode.value == "react"
            and node.allowed_tools == (self.required_tool_id,)
        )
        if len(owners) != 1:
            return self.required_tool_issue()
        actor_id = owners[0]
        metadata = execution.output_metadata.get(actor_id)
        if not isinstance(metadata, Mapping):
            return (
                f"environment actor {actor_id!r} has no execution receipt for "
                "the current Canvas revision"
            )
        trace = metadata.get("evaluator_environment_trace")
        terminal_transition = (
            isinstance(trace, (list, tuple))
            and bool(trace)
            and isinstance(trace[-1], Mapping)
            and trace[-1].get("done") is True
            and trace[-1].get("state_advanced") is True
        )
        if metadata.get("environment_terminal") is True and terminal_transition:
            return None
        return (
            f"environment actor {actor_id!r} has not produced a terminal "
            "Action--Observation transition for the current Canvas revision"
        )

    def _format_agent_issue_for(self, graph: AgentGraph) -> Optional[str]:
        if not self._uses_format_agent_protocol():
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
        if output_node.execution_mode.value != "reasoning" or output_node.allowed_tools:
            return (
                "Format Agent must use reasoning execution without tools; it only "
                "formats the verified semantic answer and must not invoke a Tool, "
                "reselect the answer, or participate in reasoning"
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
            return (
                "Format Agent must be a singleton terminal component; keep "
                "reciprocal exchange between non-Format Agents and route one "
                "semantic-answer artifact to the Format Agent"
            )
        predecessors = graph.directed_predecessors(output_agent_id)
        if len(predecessors) != 1:
            return (
                "Format Agent must consume exactly one upstream semantic-answer "
                f"artifact; received {len(predecessors)}; add or retain one "
                "semantic-answer Agent and one directed relation from that Agent "
                "to the Format Agent before FINISH"
            )
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL:
            verifier_id = predecessors[0]
            verifier = graph.get_node(verifier_id)
            if (verifier.role_family or "").casefold() != "verifier":
                return (
                    "HotpotQA Format Agent's unique predecessor must have "
                    "role_family='verifier'; the Formatter only wraps an already "
                    "verified semantic answer and must never select or reason over "
                    "an answer"
                )
            reasoner_predecessors = tuple(
                agent_id
                for agent_id in graph.directed_predecessors(verifier_id)
                if (graph.get_node(agent_id).role_family or "").casefold()
                == "reasoner"
            )
            if len(reasoner_predecessors) != 1:
                return (
                    "HotpotQA Verifier must consume exactly one direct Reasoner "
                    "semantic-candidate artifact; found "
                    f"{len(reasoner_predecessors)}. Preserve the original question "
                    "scope and use set_relation or modify_agent before FINISH"
                )
            if tuple(graph.directed_predecessors(verifier_id)) != reasoner_predecessors:
                return (
                    "HotpotQA Verifier must receive its semantic artifact only from "
                    "the unique Reasoner. Route any retrieval or repair evidence "
                    "into the Reasoner first so it can align propositions to the "
                    "original answer slot before verification"
                )
        return None

    def _uses_format_agent_protocol(self) -> bool:
        return (
            self.require_format_agent
            or self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
        )

    def _semantic_edit_issue_for(self, graph: AgentGraph) -> Optional[str]:
        """Enforce the HotpotQA semantic lineage after every Canvas edit.

        The checks are incremental: an unconnected node may exist while the
        Director is still assembling a functional subgraph, but an existing
        edge may not bypass the Reasoner/Verifier/Formatter responsibility
        boundary.  This keeps FlowSteer's edit--execute--feedback transaction
        intact without imposing one fixed graph topology.
        """

        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return None
        missing_role_ids = tuple(
            node.id
            for node in graph.nodes
            if not (node.role_family or "").strip()
        )
        if missing_role_ids:
            return (
                "HotpotQA semantic protocol requires a non-empty role_family "
                "for every Agent; missing role_family for Agents "
                f"{list(missing_role_ids)!r}"
            )
        invalid_role_ids = tuple(
            node.id
            for node in graph.nodes
            if (node.role_family or "").casefold() == "react"
        )
        if invalid_role_ids:
            return (
                "HotpotQA semantic protocol rejects role_family='react' for Agents "
                f"{list(invalid_role_ids)!r}; ReAct is an execution_mode "
                "(Thought -> Action(tool) -> Observation -> Thought -> Final). "
                "Use a semantic role such as reasoner, evidence_retriever, or verifier "
                "and set execution_mode='react' only when Tool orchestration is needed"
            )

        for node in graph.nodes:
            role = (node.role_family or "").casefold()
            normalized_contract = " ".join(node.contract.casefold().split()).rstrip(
                "."
            )
            formatting_only_contract = " ".join(
                _HOTPOTQA_FORMAT_CONTRACT.casefold().split()
            ).rstrip(".")
            if (
                role in {"reasoner", "verifier"}
                and normalized_contract == formatting_only_contract
            ):
                return (
                    f"HotpotQA {role.title()} Agent {node.id!r} has a formatting-only "
                    "role contract. Preserve semantic responsibility: the Reasoner "
                    "determines the evidence-aligned semantic candidate, the Verifier "
                    "checks evidence/binding/hops/scope without changing it, and only "
                    "role_family='format' copies it into the answer wrapper"
                )
            if role == "reasoner" and (
                node.execution_mode.value != "react"
                or node.allowed_tools != (self.required_evidence_tool_id,)
            ):
                return (
                    f"HotpotQA Reasoner Agent {node.id!r} must use "
                    "execution_mode='react' with exactly "
                    f"allowed_tools=['{self.required_evidence_tool_id}']; ReAct is "
                    "the Thought -> Action(tool) -> Observation -> Thought -> Final "
                    "execution schedule, while role_family remains 'reasoner'"
                )
            if role in {"verifier", "format"} and (
                node.execution_mode.value != "reasoning" or node.allowed_tools
            ):
                return (
                    f"HotpotQA {role.title()} Agent {node.id!r} must use "
                    "execution_mode='reasoning' without Tools"
                )
            if role == "format" and node.contract != _HOTPOTQA_FORMAT_CONTRACT:
                return (
                    f"HotpotQA Formatter Agent {node.id!r} must use the neutral "
                    "formatting-only contract and must not name, select, or imply "
                    "a task answer"
                )

            predecessors = graph.directed_predecessors(node.id)
            if role == "verifier":
                invalid_predecessors = tuple(
                    predecessor_id
                    for predecessor_id in predecessors
                    if (
                        graph.get_node(predecessor_id).role_family or ""
                    ).casefold()
                    != "reasoner"
                )
                if invalid_predecessors:
                    return (
                        f"HotpotQA Verifier {node.id!r} must receive its direct "
                        "input only from Reasoners, not from Retriever/Formatter "
                        f"Agents {list(invalid_predecessors)!r}"
                    )
            if role == "format":
                invalid_predecessors = tuple(
                    predecessor_id
                    for predecessor_id in predecessors
                    if (
                        graph.get_node(predecessor_id).role_family or ""
                    ).casefold()
                    != "verifier"
                )
                if invalid_predecessors:
                    return (
                        f"HotpotQA Formatter {node.id!r} must receive its direct "
                        "input only from Verifiers and must not participate in "
                        f"reasoning; invalid predecessors={list(invalid_predecessors)!r}"
                    )
                successors = self._directed_successors(graph, node.id)
                if successors:
                    return (
                        f"HotpotQA Formatter {node.id!r} must be a terminal sink; "
                        f"remove outgoing directed edges to {list(successors)!r}"
                    )
        return None

    @staticmethod
    def _lexical_tokens(value: str) -> Tuple[str, ...]:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return tuple(
            re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", normalized)
        )

    @classmethod
    def _contains_lexical_span(cls, text: str, span: str) -> bool:
        haystack = cls._lexical_tokens(text)
        needle = cls._lexical_tokens(span)
        if not needle or len(needle) > len(haystack):
            return False
        width = len(needle)
        return any(
            haystack[index : index + width] == needle
            for index in range(len(haystack) - width + 1)
        )

    @staticmethod
    def _context_named_phrases(context: str) -> Tuple[str, ...]:
        """Return multi-token proper-name spans from public task context.

        Generate subspans as well as maximal matches so a contract containing
        ``First Last`` is still detected when the passage says ``Title First
        Last``.  Single capitalized words are intentionally excluded here to
        avoid treating an ordinary sentence-initial word as an entity.
        """

        proper_token = r"[A-Z][A-Za-z0-9]*(?:[-'’][A-Za-z0-9]+)*"
        phrases: set[str] = set()
        for match in re.finditer(
            rf"(?<![A-Za-z0-9]){proper_token}(?:\s+{proper_token}){{1,5}}",
            context,
        ):
            tokens = match.group(0).split()
            for width in range(2, len(tokens) + 1):
                for start in range(len(tokens) - width + 1):
                    phrases.add(" ".join(tokens[start : start + width]))
        return tuple(sorted(phrases, key=lambda item: (-len(item), item)))

    def _public_semantic_contract_literals(self) -> Tuple[str, ...]:
        """Collect answer-bearing literals already present in public receipts.

        This projection deliberately ignores Ground Truth and evaluator state.
        It reads only progressive Agent artifacts and SkillFlow's public
        Action--Observation continuation retained by the current Runtime.
        """

        literals: set[str] = set()
        semantic_keys = {
            "candidate_answer",
            "object_or_attribute_value",
            "evidence_span",
            "evidence",
        }

        def visit(value: object, *, semantic_value: bool = False) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if semantic_value and stripped:
                    literals.add(stripped)
                if stripped.startswith(("{", "[")):
                    try:
                        parsed = json.loads(stripped)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        return
                    visit(parsed)
                return
            if isinstance(value, Mapping):
                for raw_key, item in value.items():
                    key = re.sub(
                        r"[ -]+", "_", str(raw_key).strip().casefold()
                    )
                    visit(item, semantic_value=key in semantic_keys)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, semantic_value=semantic_value)

        for artifact in (
            *self._progressive_outputs.values(),
            *self._previous_revision_outputs.values(),
        ):
            visit(artifact)
        for continuation in self._failure_continuations.values():
            visit(continuation)
        return tuple(sorted(literals, key=lambda item: (-len(item), item)))

    def _contract_obligation_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Reject task-answer content in a newly authored Agent contract.

        FlowSteer's Canvas remains transactional and SkillFlow's public
        observations remain available to the next turn.  This project-specific
        HotpotQA admission prevents those observations from being copied into a
        pre-execution contract as a concrete answer.  It never rewrites a
        sampled action and never consults an answer key or evaluator.
        """

        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return None
        contracts: Tuple[str, ...]
        if action.action_type is AgentActionType.ADD_SUBGRAPH:
            contracts = tuple(spec.contract for spec in action.agents)
        elif (
            action.action_type is AgentActionType.MODIFY_AGENT
            and action.contract is not None
        ):
            contracts = (action.contract,)
        else:
            return None

        question = hotpotqa_question_scope(self._problem)
        context = self._problem
        if context.endswith(question) and context != question:
            context = context[: -len(question)]
        public_literals = self._public_semantic_contract_literals()
        named_phrases = self._context_named_phrases(context)
        context_tokens = self._lexical_tokens(context)
        question_tokens = self._lexical_tokens(question)

        def question_contains(span: str) -> bool:
            return self._contains_lexical_span(question, span)

        for contract in contracts:
            # Exact public semantic values include one-word entities that a
            # proper-name phrase detector cannot safely infer from raw prose.
            if any(
                not question_contains(literal)
                and self._contains_lexical_span(contract, literal)
                for literal in public_literals
            ):
                break
            if any(
                not question_contains(phrase)
                and self._contains_lexical_span(contract, phrase)
                for phrase in named_phrases
            ):
                break

            # A copied evidence sentence need not contain a named entity.  Six
            # contiguous lexical tokens are long enough to identify a passage
            # fragment while leaving ordinary scope/relation obligations free.
            contract_tokens = self._lexical_tokens(contract)
            copied_evidence = False
            for width in range(min(10, len(contract_tokens)), 5, -1):
                for start in range(len(contract_tokens) - width + 1):
                    span = contract_tokens[start : start + width]
                    if any(
                        context_tokens[index : index + width] == span
                        for index in range(len(context_tokens) - width + 1)
                    ) and not any(
                        question_tokens[index : index + width] == span
                        for index in range(len(question_tokens) - width + 1)
                    ):
                        copied_evidence = True
                        break
                if copied_evidence:
                    break
            if copied_evidence:
                break

            # Catch a one-token candidate or date only when the contract itself
            # uses answer-selection language; unmarked capitalized words remain
            # available for ordinary natural-language obligations.
            selected_literals = re.findall(
                r"\b(?:answer|candidate|value|return|select|choose|emit|copy)\b"
                r"(?:\s+[A-Za-z_-]+){0,5}?\s+"
                r"(?:as\s+|is\s+)?([A-Z][A-Za-z0-9'’.-]*|\d{3,}(?:[.,]\d+)?)",
                contract,
            )
            if any(
                self._contains_lexical_span(context, literal)
                and not question_contains(literal)
                for literal in selected_literals
            ):
                break
        else:
            return None

        return (
            "HotpotQA Agent contracts are pre-execution obligations only: a new "
            "or modified contract must not embed a concrete candidate, alias, "
            "attribute value, number/date, or evidence span copied from task "
            "context or public Tool/Agent observations outside the original "
            "question. Preserve the original scope and express only relation, "
            "answer-slot, evidence, Tool, and output-schema obligations"
        )

    @staticmethod
    def _structured_semantic_fields(
        artifact: str,
        required_fields: Tuple[str, ...],
    ) -> tuple[Optional[dict[str, object]], Optional[str]]:
        """Parse a JSON object or a strict one-field-per-line labelled artifact."""

        if not isinstance(artifact, str) or not artifact.strip():
            return None, "artifact is empty"
        text = artifact.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced is not None:
            text = fenced.group(1).strip()
        parsed: object
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        aliases = {"fact_propositions": "evidence_propositions"}
        optional_fields = (
            {"evidence", "repair_diagnosis"}
            if required_fields == _VERIFIER_SEMANTIC_FIELDS
            else set()
        )
        allowed_fields = set(required_fields) | optional_fields
        if isinstance(parsed, Mapping):
            fields = {}
            for raw_key, value in parsed.items():
                key = re.sub(r"[ -]+", "_", str(raw_key).strip().casefold())
                key = aliases.get(key, key)
                if key not in allowed_fields:
                    return None, f"unexpected structured field {key!r}"
                if key in fields:
                    return None, f"duplicate structured field {key!r}"
                fields[key] = value
        else:
            labelled_values: dict[str, list[str]] = {}
            current_key: Optional[str] = None
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                match = re.fullmatch(r"([A-Za-z][A-Za-z _-]*):\s*(.*)", line)
                if match is not None:
                    key = re.sub(
                        r"[ -]+",
                        "_",
                        match.group(1).strip().casefold(),
                    )
                    key = aliases.get(key, key)
                    if key not in allowed_fields:
                        return None, f"unexpected structured field {key!r}"
                    if key in labelled_values:
                        return None, f"duplicate structured field {key!r}"
                    current_key = key
                    labelled_values[key] = []
                    inline_value = match.group(2).strip()
                    if inline_value:
                        labelled_values[key].append(inline_value)
                    continue
                if current_key is None:
                    return None, (
                        "labelled artifact must begin with a declared `Field:` label"
                    )
                labelled_values[current_key].append(line)
            fields = {}
            for key, lines in labelled_values.items():
                value_text = "\n".join(lines).strip()
                if key == "candidate_answer":
                    # A bare numeric answer remains text.  JSON scalar coercion
                    # here would turn `Candidate answer: 1844` into an integer
                    # and incorrectly reject an otherwise valid Verifier wire.
                    fields[key] = value_text
                    continue
                try:
                    value = json.loads(value_text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    value = value_text
                fields[key] = value
        missing = tuple(field for field in required_fields if field not in fields)
        if missing:
            return None, f"missing structured fields {list(missing)!r}"
        return fields, None

    @staticmethod
    def _non_empty_semantic_value(value: object) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, (list, tuple)):
            return bool(value)
        return False

    @staticmethod
    def _possessor_surface_issue(
        candidate: str,
        evidence_span: str,
    ) -> Optional[str]:
        """Reject a strict subspan of an explicit possessive person mention."""

        for candidate_match in re.finditer(re.escape(candidate), evidence_span):
            suffix_and_marker = re.match(
                rf"(?P<suffix>(?:,\s*|\s+){_PERSON_SUFFIX_PATTERN})?"
                r"\s*(?:'s|’s)\b",
                evidence_span[candidate_match.end() :],
            )
            if suffix_and_marker is None:
                continue
            title_match = re.search(
                rf"(?<![A-Za-z0-9])(?P<title>{_PERSON_TITLE_PATTERN})\s+$",
                evidence_span[: candidate_match.start()],
            )
            omitted_title = title_match is not None
            omitted_suffix = bool(suffix_and_marker.group("suffix"))
            if omitted_title or omitted_suffix:
                return (
                    "Reasoner candidate_answer is a strict subspan of the "
                    "evidence-aligned possessor entity mention. A who-question "
                    "must preserve the complete referential surface before the "
                    "possessive marker, including any title, honorific, or name "
                    "suffix, while excluding the possessed attribute"
                )
            return None
        return None

    @classmethod
    def _reasoner_candidate(
        cls,
        artifact: str,
        *,
        original_question: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        fields, issue = cls._structured_semantic_fields(
            artifact,
            _REASONER_SEMANTIC_FIELDS,
        )
        if issue is not None or fields is None:
            return None, issue
        for field in (
            "question_scope",
            "answer_slot",
            "evidence_propositions",
            "multi_hop_chain",
            "evidence",
        ):
            if not cls._non_empty_semantic_value(fields[field]):
                return None, f"Reasoner field {field!r} must be non-empty"
        candidate = fields["candidate_answer"]
        if (
            not isinstance(candidate, str)
            or not candidate
            or candidate != candidate.strip()
            or "\n" in candidate
        ):
            return None, (
                "Reasoner field 'candidate_answer' must be one non-empty bare "
                "answer span without surrounding whitespace"
            )

        answer_slot = fields["answer_slot"]
        propositions = fields["evidence_propositions"]
        if original_question is not None and fields["question_scope"] != original_question:
            return None, (
                "Reasoner field 'question_scope' must copy the original question "
                "exactly without adding or removing scope"
            )
        if not isinstance(answer_slot, Mapping):
            return None, "Reasoner field 'answer_slot' must be one structured object"
        slot_fields = {
            "answer_type",
            "answer_cardinality",
            "qualifiers",
            "proposition_index",
            "answer_field",
        }
        if set(answer_slot) != slot_fields:
            return None, (
                "Reasoner field 'answer_slot' must contain exactly "
                f"{sorted(slot_fields)!r}"
            )
        for field in (
            "answer_type",
            "answer_cardinality",
            "answer_field",
        ):
            value = answer_slot[field]
            if not isinstance(value, str) or not value.strip():
                return None, f"Reasoner answer_slot.{field} must be non-empty text"
        expected_answer_type = (
            None
            if original_question is None
            else hotpotqa_answer_type_constraint(original_question)
        )
        if (
            expected_answer_type is not None
            and answer_slot["answer_type"] != expected_answer_type
        ):
            return None, (
                "Reasoner answer_slot.answer_type must equal the original "
                f"question's answer-type constraint {expected_answer_type!r}"
            )
        expected_answer_cardinality = (
            None
            if original_question is None
            else hotpotqa_answer_cardinality_constraint(original_question)
        )
        if (
            expected_answer_cardinality is not None
            and answer_slot["answer_cardinality"]
            != expected_answer_cardinality
        ):
            return None, (
                "Reasoner answer_slot.answer_cardinality must equal the original "
                "question's answer-cardinality constraint "
                f"{expected_answer_cardinality!r}"
            )
        qualifiers = answer_slot["qualifiers"]
        if not isinstance(qualifiers, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in qualifiers
        ):
            return None, "Reasoner answer_slot.qualifiers must be a string array"
        proposition_index = answer_slot["proposition_index"]
        if isinstance(proposition_index, bool) or not isinstance(
            proposition_index, int
        ):
            return None, "Reasoner answer_slot.proposition_index must be an integer"
        if not isinstance(propositions, (list, tuple)) or len(propositions) < 2:
            return None, (
                "Reasoner field 'evidence_propositions' must contain at least two "
                "propositions for HotpotQA multi-hop alignment"
            )
        multi_hop_chain = fields["multi_hop_chain"]
        if (
            not isinstance(multi_hop_chain, (list, tuple))
            or len(multi_hop_chain) < 2
            or any(
                not isinstance(item, str) or not item.strip()
                for item in multi_hop_chain
            )
        ):
            return None, (
                "Reasoner field 'multi_hop_chain' must contain at least two "
                "non-empty hop descriptions"
            )
        proposition_fields = {
            "subject",
            "relation",
            "object_or_attribute_value",
            "qualifiers",
            "evidence_span",
        }
        for index, proposition in enumerate(propositions):
            if not isinstance(proposition, Mapping):
                return None, (
                    f"Reasoner evidence_propositions[{index}] must be an object"
                )
            if set(proposition) != proposition_fields:
                return None, (
                    f"Reasoner evidence_propositions[{index}] must contain exactly "
                    f"{sorted(proposition_fields)!r}"
                )
            for field in (
                "subject",
                "relation",
                "object_or_attribute_value",
                "evidence_span",
            ):
                value = proposition[field]
                if not isinstance(value, str) or not value.strip():
                    return None, (
                        f"Reasoner evidence_propositions[{index}].{field} "
                        "must be non-empty text"
                    )
            proposition_qualifiers = proposition["qualifiers"]
            if not isinstance(proposition_qualifiers, (list, tuple)) or any(
                not isinstance(item, str) or not item.strip()
                for item in proposition_qualifiers
            ):
                return None, (
                    f"Reasoner evidence_propositions[{index}].qualifiers must be "
                    "a string array"
                )
        if proposition_index < 0 or proposition_index >= len(propositions):
            return None, (
                "Reasoner answer_slot.proposition_index is outside "
                "evidence_propositions"
            )
        selected = propositions[proposition_index]
        assert isinstance(selected, Mapping)
        answer_field = answer_slot["answer_field"]
        if answer_field not in {"subject", "object_or_attribute_value"}:
            return None, (
                "Reasoner answer_slot.answer_field must be subject or "
                "object_or_attribute_value"
            )
        if candidate != selected[answer_field]:
            return None, (
                "Reasoner candidate_answer must copy the proposition argument "
                "identified by answer_slot.proposition_index and answer_field exactly"
            )
        if expected_answer_type in {"entity", "person", "location"} and re.fullmatch(
            r"[\d\s.,:/-]+",
            candidate,
        ):
            return None, (
                f"Reasoner candidate_answer is numeric/date-like but the original "
                f"question requires answer type {expected_answer_type!r}"
            )
        if expected_answer_type == "person" and re.search(
            r"(?:'s|’s)\s+\S",
            candidate,
        ):
            return None, (
                "Reasoner candidate_answer is a possessive noun phrase, but a who "
                "question requires the person/possessor entity rather than the "
                "possessed attribute phrase"
            )
        if expected_answer_type == "yes_no" and candidate.casefold() not in {
            "yes",
            "no",
        }:
            return None, (
                "Reasoner candidate_answer must be yes or no for the original "
                "yes/no question"
            )
        evidence_span = selected["evidence_span"]
        assert isinstance(evidence_span, str)
        boolean_answer = candidate.casefold() in {"yes", "no"}
        if not boolean_answer and candidate not in evidence_span:
            return None, (
                "Reasoner candidate_answer must occur verbatim in the selected "
                "evidence_span"
            )
        if expected_answer_type == "person":
            possessor_surface_issue = cls._possessor_surface_issue(
                candidate,
                evidence_span,
            )
            if possessor_surface_issue is not None:
                return None, possessor_surface_issue
        return candidate, None

    @classmethod
    def _verifier_candidate(
        cls,
        artifact: str,
    ) -> tuple[Optional[str], Optional[str]]:
        fields, issue = cls._structured_semantic_fields(
            artifact,
            _VERIFIER_SEMANTIC_FIELDS,
        )
        if issue is not None or fields is None:
            return None, issue
        candidate = fields["candidate_answer"]
        if (
            not isinstance(candidate, str)
            or not candidate
            or candidate != candidate.strip()
            or "\n" in candidate
        ):
            return None, (
                "Verifier field 'candidate_answer' must be one non-empty bare "
                "answer span without surrounding whitespace"
            )
        for field in (
            "evidence_supported",
            "entity_attribute_binding_correct",
            "alias_binding_correct",
            "answer_type_cardinality_correct",
            "multi_hop_complete",
            "minimal_answer_surface",
            "scope_preserved",
        ):
            if fields[field] is not True:
                return None, f"Verifier field {field!r} must be true"
        status = fields["verification_status"]
        if not isinstance(status, str) or status.strip().casefold() != "supported":
            return None, "Verifier field 'verification_status' must be 'supported'"
        return candidate, None

    @staticmethod
    def _successful_read_receipt(
        receipt: Mapping[str, object],
        required_tool_id: str,
    ) -> bool:
        if (
            receipt.get("tool_id") != required_tool_id
            or receipt.get("error_type") is not None
        ):
            return False
        request = receipt.get("request")
        if not isinstance(request, Mapping) or request.get("action") != "read":
            return False
        result = receipt.get("result")
        if not isinstance(result, Mapping):
            return False
        value = result.get("value", result)
        if not isinstance(value, Mapping) or value.get("operation") != "read":
            return False
        passage = value.get("passage")
        return (
            isinstance(passage, Mapping)
            and isinstance(passage.get("text"), str)
            and bool(passage["text"].strip())
        )

    @staticmethod
    def _successful_read_text(
        receipt: Mapping[str, object],
        required_tool_id: str,
    ) -> Optional[str]:
        if not AgentWorkflowEnv._successful_read_receipt(
            receipt,
            required_tool_id,
        ):
            return None
        result = receipt["result"]
        assert isinstance(result, Mapping)
        value = result.get("value", result)
        assert isinstance(value, Mapping)
        passage = value["passage"]
        assert isinstance(passage, Mapping)
        text = passage["text"]
        assert isinstance(text, str)
        return text

    @classmethod
    def _reasoner_evidence_provenance_issue(
        cls,
        artifact: str,
        read_evidence_texts: Sequence[str],
    ) -> Optional[str]:
        """Validate proposition spans against successful read provenance.

        The same typography-only lexical matcher is shared by the bounded
        Reasoner completion gate and the outer explicit-FINISH gate.  Keeping
        one validator prevents a completion from leaving its ReAct loop only
        to fail later under a different provenance rule.
        """

        fields, issue = cls._structured_semantic_fields(
            artifact,
            _REASONER_SEMANTIC_FIELDS,
        )
        if issue is not None or fields is None:
            return issue or "Reasoner semantic artifact is missing"
        propositions = fields.get("evidence_propositions")
        if not isinstance(propositions, (list, tuple)):
            return "Reasoner field 'evidence_propositions' must be an array"
        for index, proposition in enumerate(propositions):
            if not isinstance(proposition, Mapping):
                return f"Reasoner evidence_propositions[{index}] must be an object"
            evidence_span = proposition.get("evidence_span")
            if not isinstance(evidence_span, str) or not any(
                _evidence_span_matches_read(evidence_span, text)
                for text in read_evidence_texts
            ):
                return (
                    f"Reasoner evidence_propositions[{index}].evidence_span "
                    "has no typography-canonical lexical match in any successful "
                    "qa-retrieval read"
                )
        return None

    def _semantic_protocol_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Return the shared HotpotQA FINISH/read-only-admissibility gate."""

        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return None
        structure_issue = self._format_agent_issue_for(self._graph)
        if structure_issue is not None:
            return structure_issue
        formatter_id = self._graph.output_agent_id
        if formatter_id is None:
            return "HotpotQA semantic protocol requires a selected Format Agent"
        verifier_id = self._graph.directed_predecessors(formatter_id)[0]
        reasoner_ids = tuple(
            agent_id
            for agent_id in self._graph.directed_predecessors(verifier_id)
            if (self._graph.get_node(agent_id).role_family or "").casefold()
            == "reasoner"
        )
        if len(reasoner_ids) != 1:
            return (
                "HotpotQA semantic protocol requires exactly one direct Reasoner "
                "semantic candidate for the Verifier"
            )
        reasoner_id = reasoner_ids[0]
        reasoner_artifact = execution.outputs.get(reasoner_id)
        verifier_artifact = execution.outputs.get(verifier_id)
        if reasoner_artifact is None:
            return f"Reasoner {reasoner_id!r} has no current semantic artifact"
        if verifier_artifact is None:
            return f"Verifier {verifier_id!r} has no current verification artifact"
        reasoner_candidate, reasoner_issue = self._reasoner_candidate(
            reasoner_artifact,
            original_question=hotpotqa_question_scope(self._problem),
        )
        if reasoner_issue is not None or reasoner_candidate is None:
            return (
                f"Reasoner {reasoner_id!r} semantic artifact is invalid: "
                f"{reasoner_issue}. Required fields are "
                f"{list(_REASONER_SEMANTIC_FIELDS)!r}"
            )
        # Evidence may be read by the Reasoner itself or by a dedicated
        # Retriever routed into the Reasoner. It may not bypass semantic
        # alignment through a Retriever -> Verifier edge.
        evidence_owner_ids = (
            reasoner_id,
            *self._graph.directed_predecessors(reasoner_id),
        )
        read_evidence_texts: list[str] = []
        assert self.required_evidence_tool_id is not None
        for evidence_owner_id in evidence_owner_ids:
            metadata = execution.output_metadata.get(evidence_owner_id)
            if not isinstance(metadata, Mapping):
                continue
            receipts = metadata.get("tool_receipts", ())
            if not isinstance(receipts, (list, tuple)):
                continue
            for receipt in receipts:
                if not isinstance(receipt, Mapping):
                    continue
                evidence_text = self._successful_read_text(
                    receipt,
                    self.required_evidence_tool_id,
                )
                if evidence_text is not None:
                    read_evidence_texts.append(evidence_text)
        if not read_evidence_texts:
            return (
                "Reasoner lineage has no successful "
                f"{self.required_evidence_tool_id!r} read receipt containing a "
                "non-empty passage. Preserve existing artifacts and add or repair "
                "retrieval into the Reasoner before FINISH"
            )
        provenance_issue = self._reasoner_evidence_provenance_issue(
            reasoner_artifact,
            read_evidence_texts,
        )
        if provenance_issue is not None:
            return (
                provenance_issue
                + ". Preserve the existing candidate and valid evidence; repair or "
                "augment retrieval before FINISH"
            )
        verifier_candidate, verifier_issue = self._verifier_candidate(
            verifier_artifact
        )
        if verifier_issue is not None or verifier_candidate is None:
            return (
                f"Verifier {verifier_id!r} semantic artifact is invalid: "
                f"{verifier_issue}. The Reasoner candidate {reasoner_candidate!r} "
                "already passed answer-slot binding and retrieved-evidence alignment; "
                "preserve that candidate and evidence, diagnose the failed Verifier "
                "check, then repair the existing Verifier or augment retrieval before "
                "FINISH"
            )
        if verifier_candidate != reasoner_candidate:
            return (
                "Verifier changed the Reasoner's candidate_answer: "
                f"reasoner={reasoner_candidate!r}, verifier={verifier_candidate!r}. "
                "The Reasoner candidate already passed answer-slot binding and "
                "retrieved-evidence alignment. Preserve it and repair the Verifier; "
                "the Verifier must not select a replacement answer"
            )
        answer = execution.final_answer
        if answer is None:
            return "Format Agent produced no terminal wrapper"
        wrapper = re.fullmatch(
            r"\s*<answer>(.*?)</answer>\s*",
            answer,
            flags=re.DOTALL,
        )
        if wrapper is None or wrapper.group(1) != reasoner_candidate:
            formatter_value = None if wrapper is None else wrapper.group(1)
            return (
                "Formatter must only wrap the verified candidate and may not "
                "reselect or transform it: "
                f"candidate_answer={reasoner_candidate!r}, "
                f"wrapper_content={formatter_value!r}"
            )
        return None

    @staticmethod
    def _directed_successors(graph: AgentGraph, agent_id: str) -> Tuple[str, ...]:
        return tuple(
            sorted(
                {
                    target_id
                    for relation in graph.relations
                    for source_id, target_id in relation.directed_edges()
                    if source_id == agent_id
                }
            )
        )

    def _has_successful_artifact(self, agent_id: str) -> bool:
        artifact = self._progressive_outputs.get(agent_id)
        return (
            isinstance(artifact, str)
            and bool(artifact.strip())
            and agent_id not in self._unresolved_dirty_agents
        )

    def _terminal_unreachable_agent_ids(self) -> Tuple[str, ...]:
        """Project the existing complete-graph reachability diagnosis.

        ``AgentGraphValidator`` is the terminal-topology authority.  Reuse its
        reciprocal-component contraction and quotient-DAG reachability result
        here instead of maintaining a second graph traversal in recovery.
        """

        validation = self._graph.validate(
            self.model_registry,
            require_complete=True,
        )
        return tuple(
            sorted(
                {
                    agent_id
                    for issue in validation.issues
                    if issue.code == "cannot_reach_output"
                    for agent_id in issue.agent_ids
                }
            )
        )

    def recovery_state(self) -> dict[str, object]:
        """Expose measured preservation state without changing scheduling/cache rules."""

        current_ids = {node.id for node in self._graph.nodes}
        preserved = tuple(
            sorted(
                agent_id
                for agent_id in current_ids
                if self._has_successful_artifact(agent_id)
            )
        )
        terminal_unreachable = self._terminal_unreachable_agent_ids()
        terminal_unreachable_set = set(terminal_unreachable)
        failed = tuple(sorted(self._failed_agent_ids & current_ids))
        react_exhausted = tuple(
            sorted(self._react_exhausted_agent_ids & current_ids)
        )
        diagnosed_unusable = tuple(
            sorted(self._diagnosed_unusable_agent_ids & current_ids)
        )
        diagnosed_unusable_set = set(diagnosed_unusable)
        active_semantic_lineage = self._active_semantic_lineage_ids()
        active_semantic_lineage_set = set(active_semantic_lineage)
        redundant_after_takeover = tuple(
            agent_id
            for agent_id in terminal_unreachable
            if agent_id not in active_semantic_lineage_set
            and bool(active_semantic_lineage)
        )
        deletable = tuple(
            node.id
            for node in self._graph.nodes
            if self._delete_admission_issue(node.id) is None
        )
        deletable_set = set(deletable)
        protected: dict[str, list[str]] = {}
        for node in self._graph.nodes:
            if node.id in deletable_set:
                continue
            reasons: list[str] = []
            if node.id in preserved:
                reasons.append("successful_artifact")
            if self._directed_successors(self._graph, node.id):
                reasons.append("downstream_responsibility")
            if self._graph.output_agent_id == node.id:
                reasons.append("output_identity")
            if node.id in terminal_unreachable_set:
                reasons.append("terminal_unreachable")
            if node.id not in diagnosed_unusable_set:
                reasons.append("not_diagnosed_unusable")
            reasons.append("replacement_takeover_required")
            if reasons:
                protected[node.id] = reasons
        return {
            "policy": self.recovery_policy,
            "strategy": "preserve -> diagnose -> repair -> augment",
            "phase": (
                "diagnose_repair"
                if (
                    self._unresolved_dirty_agents
                    or self._failed_agent_ids
                    or terminal_unreachable
                )
                else "preserve"
            ),
            "preserved_agent_ids": list(preserved),
            "previous_revision_preserved_agent_ids": sorted(
                self._previous_revision_outputs
            ),
            "failed_agent_ids": list(failed),
            "react_turn_exhausted_agent_ids": list(react_exhausted),
            "diagnosed_unusable_agent_ids": list(diagnosed_unusable),
            "unresolved_dirty_agent_ids": list(self.unresolved_dirty_agent_ids),
            "terminal_unreachable_agent_ids": list(terminal_unreachable),
            "active_semantic_lineage_agent_ids": list(active_semantic_lineage),
            "redundant_after_replacement_takeover_agent_ids": list(
                redundant_after_takeover
            ),
            "deletable_agent_ids": list(deletable),
            "deletion_protected": protected,
            "preferred_actions": (
                ["delete_agent", "set_relation", "modify_agent"]
                if deletable
                else ["modify_agent", "set_relation", "add_subgraph"]
            ),
        }

    def _active_semantic_lineage_ids(self) -> Tuple[str, ...]:
        """Return the current verified Reasoner→Verifier→Formatter lineage."""

        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return ()
        formatter_id = self._graph.output_agent_id
        if formatter_id is None or not self._graph.has_node(formatter_id):
            return ()
        formatter = self._graph.get_node(formatter_id)
        if (formatter.role_family or "").casefold() != "format":
            return ()
        verifier_ids = self._graph.directed_predecessors(formatter_id)
        if len(verifier_ids) != 1:
            return ()
        verifier_id = verifier_ids[0]
        verifier = self._graph.get_node(verifier_id)
        if (verifier.role_family or "").casefold() != "verifier":
            return ()
        reasoner_ids = self._graph.directed_predecessors(verifier_id)
        if len(reasoner_ids) != 1:
            return ()
        reasoner_id = reasoner_ids[0]
        reasoner = self._graph.get_node(reasoner_id)
        if (reasoner.role_family or "").casefold() != "reasoner":
            return ()
        for agent_id, role_family in (
            (reasoner_id, "reasoner"),
            (verifier_id, "verifier"),
            (formatter_id, "format"),
        ):
            if not self._has_successful_artifact(agent_id):
                return ()
            if not self._semantic_replacement_has_valid_artifact(
                agent_id,
                role_family,
            ):
                return ()
        return reasoner_id, verifier_id, formatter_id

    def _semantic_replacement_has_valid_artifact(
        self,
        agent_id: str,
        role_family: str,
    ) -> bool:
        """Require a valid current semantic artifact before replacement takeover."""

        if self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL:
            return True
        artifact = self._progressive_outputs.get(agent_id)
        if not isinstance(artifact, str) or not artifact.strip():
            return False
        if role_family == "reasoner":
            candidate, issue = self._reasoner_candidate(
                artifact,
                original_question=hotpotqa_question_scope(self._problem),
            )
            if issue is not None or candidate is None:
                return False
            fields, _ = self._structured_semantic_fields(
                artifact,
                _REASONER_SEMANTIC_FIELDS,
            )
            if fields is None:
                return False
            evidence_texts: list[str] = []
            assert self.required_evidence_tool_id is not None
            for owner_id in (
                agent_id,
                *self._graph.directed_predecessors(agent_id),
            ):
                metadata = self._progressive_output_metadata.get(owner_id)
                if not isinstance(metadata, Mapping):
                    continue
                receipts = metadata.get("tool_receipts", ())
                if not isinstance(receipts, (list, tuple)):
                    continue
                for receipt in receipts:
                    if not isinstance(receipt, Mapping):
                        continue
                    evidence_text = self._successful_read_text(
                        receipt,
                        self.required_evidence_tool_id,
                    )
                    if evidence_text is not None:
                        evidence_texts.append(evidence_text)
            propositions = fields.get("evidence_propositions")
            return bool(evidence_texts) and isinstance(
                propositions,
                (list, tuple),
            ) and all(
                isinstance(proposition, Mapping)
                and isinstance(proposition.get("evidence_span"), str)
                and any(
                    _evidence_span_matches_read(
                        proposition["evidence_span"], evidence_text
                    )
                    for evidence_text in evidence_texts
                )
                for proposition in propositions
            )
        if role_family == "verifier":
            verifier_candidate, verifier_issue = self._verifier_candidate(artifact)
            if verifier_issue is not None or verifier_candidate is None:
                return False
            reasoner_ids = tuple(
                predecessor_id
                for predecessor_id in self._graph.directed_predecessors(agent_id)
                if (
                    self._graph.get_node(predecessor_id).role_family or ""
                ).casefold()
                == "reasoner"
            )
            if len(reasoner_ids) != 1:
                return False
            reasoner_artifact = self._progressive_outputs.get(reasoner_ids[0], "")
            reasoner_candidate, reasoner_issue = self._reasoner_candidate(
                reasoner_artifact,
                original_question=hotpotqa_question_scope(self._problem),
            )
            return (
                reasoner_issue is None
                and reasoner_candidate is not None
                and verifier_candidate == reasoner_candidate
            )
        if role_family == "format":
            predecessors = self._graph.directed_predecessors(agent_id)
            if len(predecessors) != 1:
                return False
            verifier_artifact = self._progressive_outputs.get(predecessors[0], "")
            verifier_candidate, verifier_issue = self._verifier_candidate(
                verifier_artifact
            )
            wrapper = re.fullmatch(
                r"\s*<answer>(.*?)</answer>\s*",
                artifact,
                flags=re.DOTALL,
            )
            return (
                verifier_issue is None
                and verifier_candidate is not None
                and wrapper is not None
                and wrapper.group(1) == verifier_candidate
            )
        return True

    def _delete_admission_issue(self, agent_id: Optional[str]) -> Optional[str]:
        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return None
        if agent_id is None or not self._graph.has_node(agent_id):
            return None
        node = self._graph.get_node(agent_id)
        terminal_unreachable_ids = set(self._terminal_unreachable_agent_ids())
        # Topological disconnection is a relation fault, not evidence that the
        # node itself is unusable.  Deletion therefore requires a measured
        # execution failure plus a same-responsibility replacement takeover.
        diagnosed_unusable = agent_id in self._diagnosed_unusable_agent_ids
        downstream_ids = set(self._directed_successors(self._graph, agent_id))
        protected_reasons: list[str] = []
        if self._has_successful_artifact(agent_id):
            protected_reasons.append("successful artifact/evidence")
        if downstream_ids:
            protected_reasons.append("downstream responsibility")
        if self._graph.output_agent_id == agent_id:
            protected_reasons.append("Output Agent identity")
        if not diagnosed_unusable:
            protected_reasons.append("node has not been diagnosed unusable")
        protected_reasons.append("replacement artifact takeover is required")

        role = (node.role_family or "").casefold()
        artifact_type = node.artifact_type.casefold()
        replacements: list[str] = []
        for candidate in self._graph.nodes:
            if candidate.id == agent_id:
                continue
            if (
                (candidate.role_family or "").casefold() != role
                or candidate.artifact_type.casefold() != artifact_type
                or not self._has_successful_artifact(candidate.id)
                or candidate.id in terminal_unreachable_ids
                or not self._semantic_replacement_has_valid_artifact(
                    candidate.id,
                    role,
                )
            ):
                continue
            candidate_downstream = set(
                self._directed_successors(self._graph, candidate.id)
            )
            if not downstream_ids <= candidate_downstream:
                continue
            # The Director must transfer Output identity before deleting the
            # previous owner; a graph cannot have two simultaneous outputs.
            if self._graph.output_agent_id == agent_id:
                continue
            replacements.append(candidate.id)
        if diagnosed_unusable and replacements:
            return None
        return (
            f"recovery_policy={_PRESERVE_REPAIR_RECOVERY_POLICY} protects Agent "
            f"{agent_id!r} because it has {', '.join(protected_reasons)}. "
            "Use preserve -> diagnose -> repair -> augment: prefer modify_agent, "
            "set_relation, or add_subgraph. Delete is admitted only after a "
            "same-role/same-artifact replacement has executed successfully, taken "
            "every downstream relation, and (when applicable) received Output identity"
        )

    def _preservation_admission_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Protect a verified semantic lineage while recovery remains active."""

        if (
            self.semantic_protocol != _HOTPOTQA_SEMANTIC_PROTOCOL
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            or action.action_type is AgentActionType.FINISH
        ):
            return None
        mandatory_repair_ids = self._mandatory_repair_agent_ids()
        if mandatory_repair_ids and (
            action.action_type is not AgentActionType.MODIFY_AGENT
            or action.agent_id not in mandatory_repair_ids
        ):
            return (
                "repair the measured responsible Agent before augmentation or "
                "other Canvas edits; mandatory_repair_agent_ids="
                f"{list(mandatory_repair_ids)!r}"
            )
        if action.action_type is AgentActionType.ADD_SUBGRAPH:
            admitted_roles = set(self._admissible_augmentation_role_families())
            sampled_roles = [
                (spec.role_family or "").casefold() for spec in action.agents
            ]
            for role_family in ("reasoner", "verifier", "format"):
                sampled_count = sampled_roles.count(role_family)
                if sampled_count > 1:
                    return (
                        "one add_subgraph transaction cannot add multiple Agents "
                        f"with semantic responsibility {role_family!r}"
                    )
                if sampled_count == 1 and role_family not in admitted_roles:
                    return (
                        f"a healthy {role_family!r} Agent already owns that semantic "
                        "responsibility; modify the existing Agent first. A same-role "
                        "replacement is admitted only after an explicit node_unusable "
                        "receipt, while evidence_retriever and repair branches remain "
                        "available for non-linear augmentation"
                    )
        finish_admissible = self.finish_admissibility().get("admissible") is True
        if finish_admissible:
            return (
                "the current revision already has a verified terminal artifact; "
                "preserve its evidence, semantic answer, relations, and Output "
                "identity and emit finish"
            )
        lineage = self._active_semantic_lineage_ids()
        if not lineage:
            return None
        lineage_set = set(lineage)
        if (
            action.action_type is AgentActionType.MODIFY_AGENT
            and action.agent_id in lineage_set
            and action.agent_id not in self._failed_agent_ids
            and action.agent_id not in self._unresolved_dirty_agents
        ):
            return (
                f"Agent {action.agent_id!r} belongs to the verified semantic "
                "lineage and has no measured failure"
            )
        if (
            action.action_type is AgentActionType.SET_OUTPUT
            and self._graph.output_agent_id in lineage_set
            and action.agent_id != self._graph.output_agent_id
        ):
            return "the verified Formatter Output identity must be preserved"
        if action.action_type is AgentActionType.SET_RELATION:
            candidate = self._graph.fork()
            try:
                candidate.set_relation(
                    action.source_id,
                    action.target_id,
                    bool(action.source_to_target),
                    bool(action.target_to_source),
                )
            except GraphMutationError:
                return None
            protected_edges = tuple(zip(lineage, lineage[1:]))
            if any(
                target_id not in self._directed_successors(candidate, source_id)
                for source_id, target_id in protected_edges
            ):
                return "a verified semantic-lineage relation must be preserved"
        return None

    @staticmethod
    def _execution_failure_diagnosis(
        record: AgentFailureRecord,
    ) -> Tuple[str, str, Optional[int]]:
        """Classify one public Runtime failure without inferring node deletion."""

        failure_text = " ".join(record.message.split())
        normalized = f"{record.error_type} {failure_text}".casefold()
        status_match = re.search(
            r"(?:http(?: error)?|status)[ :=]*(\d{3})",
            normalized,
        )
        status_code = None if status_match is None else int(status_match.group(1))
        if (
            "openaicompatiblegatewayerror" in normalized
            or "provider request failed" in normalized
            or status_code is not None
        ):
            return (
                "provider_request_failure",
                (
                    "permanent_configuration"
                    if status_code in {400, 401, 403, 404}
                    else "transient_provider"
                ),
                status_code,
            )
        if "reactexecutionerror" in normalized or (
            "react" in normalized
            and ("turn" in normalized or "exhaust" in normalized)
        ):
            return (
                "react_turn_exhaustion",
                "repair_execution_contract_or_tool_plan",
                status_code,
            )
        if "tool" in normalized:
            return (
                "tool_capability_failure",
                "repair_tool_capability_or_arguments",
                status_code,
            )
        return (
            "execution_contract_or_runtime_failure",
            "diagnose_existing_agent",
            status_code,
        )

    def _react_public_error_summary(
        self,
        record: AgentFailureRecord,
    ) -> dict[str, object]:
        """Compress public ReAct receipts without replaying retrieved passages."""

        raw_trace = record.metadata.get("react_trace", ())
        trace = (
            tuple(item for item in raw_trace if isinstance(item, Mapping))
            if isinstance(raw_trace, (list, tuple))
            else ()
        )
        status_counts: dict[str, int] = {}
        code_counts: dict[str, int] = {}
        last_public_error: Optional[dict[str, str]] = None
        for entry in trace:
            observation = entry.get("observation")
            source = observation if isinstance(observation, Mapping) else entry
            status = source.get("observation_status")
            code = source.get("public_error_code")
            repair_instruction = source.get("repair_instruction")
            if not isinstance(repair_instruction, str):
                repair_instruction = entry.get("repair_instruction")
            if isinstance(status, str) and status:
                status_counts[status] = status_counts.get(status, 0) + 1
            if isinstance(code, str) and code:
                code_counts[code] = code_counts.get(code, 0) + 1
            if (
                isinstance(status, str)
                and status not in {"success", "completed"}
            ):
                last_public_error = {"observation_status": status}
                if isinstance(code, str) and code:
                    last_public_error["public_error_code"] = code
                if isinstance(repair_instruction, str) and repair_instruction.strip():
                    # SkillFlow makes this instruction part of the public
                    # Observation. Keep the generic schema repair, while still
                    # omitting the retrieved passage and complete trace.
                    last_public_error["repair_instruction"] = " ".join(
                        repair_instruction.split()
                    )[:400]

        raw_receipts = record.metadata.get("tool_receipts", ())
        receipts = (
            tuple(item for item in raw_receipts if isinstance(item, Mapping))
            if isinstance(raw_receipts, (list, tuple))
            else ()
        )
        successful_tool_count = sum(
            receipt.get("error_type") is None
            and isinstance(receipt.get("result"), Mapping)
            for receipt in receipts
        )
        successful_evidence_read_count = 0
        if self.required_evidence_tool_id is not None:
            successful_evidence_read_count = sum(
                self._successful_read_receipt(
                    receipt,
                    self.required_evidence_tool_id,
                )
                for receipt in receipts
            )
        summary: dict[str, object] = {
            "react_turn_count": len(trace),
            "observation_status_counts": {
                key: status_counts[key] for key in sorted(status_counts)
            },
            "public_error_code_counts": {
                key: code_counts[key] for key in sorted(code_counts)
            },
            "successful_tool_receipt_count": successful_tool_count,
            "successful_evidence_read_count": successful_evidence_read_count,
        }
        if last_public_error is not None:
            summary["last_public_error"] = last_public_error
        return summary

    def _execution_error_feedback(self, exc: AgentRuntimeError) -> str:
        message = " ".join(str(exc).split())
        if len(message) > 240:
            message = message[:237] + "..."
        failed_agents: list[dict[str, object]] = []
        for record in exc.failure_records[:4]:
            node = (
                self._graph.get_node(record.agent_id)
                if self._graph.has_node(record.agent_id)
                else None
            )
            model_id = None if node is None else node.model_id
            provider_id = (
                None
                if model_id is None
                else self.model_registry.provider_for(model_id).provider_id
            )
            category, retryability, status_code = (
                self._execution_failure_diagnosis(record)
            )
            item: dict[str, object] = {
                "agent_id": record.agent_id,
                "model_id": model_id,
                "provider_id": provider_id,
                "phase": record.phase.value,
                "error_type": record.error_type,
                "failure_category": category,
                "retryability": retryability,
            }
            if status_code is not None:
                item["http_status"] = status_code
            if category == "provider_request_failure" and model_id is not None:
                item["preferred_repair"] = {
                    "action": "modify_agent",
                    "agent_id": record.agent_id,
                    "field": "model_id",
                    "avoid_provider_id": provider_id,
                    "preserve_fields": [
                        "contract",
                        "role_family",
                        "allowed_tools",
                        "execution_mode",
                        "artifact_type",
                        "completion_condition",
                        "relations",
                    ],
                }
            elif category == "react_turn_exhaustion":
                item["react_public_error_summary"] = (
                    self._react_public_error_summary(record)
                )
                item["preferred_repair"] = {
                    "action": "modify_agent",
                    "agent_id": record.agent_id,
                    "field": "contract",
                    "optional_field": "completion_condition",
                    "preserve_fields": [
                        "model_id",
                        "role_family",
                        "allowed_tools",
                        "execution_mode",
                        "artifact_type",
                        "relations",
                    ],
                }
            failed_agents.append(item)
        payload = json.dumps(
            {
                "type": type(exc).__name__,
                "message": message,
                "failed_agents": failed_agents,
                "blocked_agent_ids": list(exc.blocked_agent_ids),
                "pending_agent_ids": list(exc.pending_agent_ids),
                "preserved_agent_ids": (
                    []
                    if exc.partial_result is None
                    else sorted(exc.partial_result.outputs)
                ),
                **(
                    {"recovery_state": self.recovery_state()}
                    if self.recovery_policy
                    == _PRESERVE_REPAIR_RECOVERY_POLICY
                    else {}
                ),
            },
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
            prior_failure_metadata=self._failure_continuations,
            format_output_agent=self._uses_format_agent_protocol(),
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
        execution: Optional[AgentRuntimeResult] = None,
        execution_reused: bool = False,
        partial_execution: Optional[AgentRuntimeResult] = None,
        execution_failure_records: Tuple[AgentFailureRecord, ...] = (),
    ) -> AgentWorkflowStepResult:
        self._last_feedback = feedback
        self._record_history(
            accepted=False,
            done=self._finished,
            action=action,
            feedback=feedback,
            execution_reused=execution_reused,
        )
        return AgentWorkflowStepResult(
            accepted=False,
            done=self._finished,
            action=action,
            revision=self._graph.revision,
            feedback=feedback,
            snapshot=self.snapshot(),
            validation_issues=issues,
            execution=execution,
            execution_reused=execution_reused,
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
