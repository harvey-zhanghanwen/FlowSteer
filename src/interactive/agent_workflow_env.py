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
    qa_answer_type_constraint_accepts,
)


class AgentWorkflowStateError(RuntimeError):
    """Raised for invalid environment construction or restoration."""


_HOTPOTQA_SEMANTIC_PROTOCOL = "hotpotqa_verified_answer_slot_v1"
_HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL = "hotpotqa_semantic_lineage_v2"
_HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL = (
    "hotpotqa_role_conditional_capabilities_v1"
)
_QA_SEMANTIC_PROTOCOL = "qa_verified_answer_lineage_v2"
_PRESERVE_REPAIR_RECOVERY_POLICY = "preserve_diagnose_repair_augment"
_NON_TERMINAL_PARTIAL_ARTIFACT = "non_terminal_partial"
_SEMANTIC_LINEAGE_PROTOCOLS = frozenset(
    {
        _HOTPOTQA_SEMANTIC_PROTOCOL,
        _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL,
        _HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL,
        _QA_SEMANTIC_PROTOCOL,
    }
)
_SUPPORTED_SEMANTIC_PROTOCOLS = frozenset(
    {"none", *_SEMANTIC_LINEAGE_PROTOCOLS}
)
_SUPPORTED_RECOVERY_POLICIES = frozenset(
    {"default", _PRESERVE_REPAIR_RECOVERY_POLICY}
)
_HOTPOTQA_FORMAT_CONTRACT = (
    "copy the supported Verifier candidate character-for-character into the "
    "required answer wrapper"
)
_HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT = (
    "copy the routed semantic candidate character-for-character into the "
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


@dataclass(frozen=True, slots=True)
class AgentWorkflowEvidenceLineageSnapshot:
    """Last execution that passed the complete evidence-lineage FINISH gate.

    The Runtime result and AgentGraph snapshot are both immutable.  Publishing
    one frozen value after every gate succeeds makes the update atomic for
    readers such as the rollout collector; a later rejected or invalid Canvas
    revision cannot partially overwrite the previously valid lineage.
    """

    answer: str
    runtime: AgentRuntimeResult
    graph_revision: int
    graph_snapshot: AgentGraphSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("evidence-lineage answer must be non-empty")
        if self.runtime.final_answer != self.answer:
            raise ValueError("evidence-lineage answer must equal Runtime final_answer")
        if self.graph_revision != self.runtime.graph_revision:
            raise ValueError("evidence-lineage Runtime revision mismatch")
        if self.graph_revision != self.graph_snapshot.revision:
            raise ValueError("evidence-lineage graph snapshot revision mismatch")


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
                "semantic_protocol must be one of "
                f"{sorted(_SUPPORTED_SEMANTIC_PROTOCOLS)!r}"
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
        if semantic_protocol in _SEMANTIC_LINEAGE_PROTOCOLS:
            if recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
                raise AgentWorkflowStateError(
                    f"{semantic_protocol} requires "
                    "recovery_policy=preserve_diagnose_repair_augment"
                )
            if required_evidence_tool_id != "qa-retrieval":
                raise AgentWorkflowStateError(
                    f"{semantic_protocol} requires "
                    "required_evidence_tool_id='qa-retrieval'"
                )
        if semantic_protocol == _QA_SEMANTIC_PROTOCOL:
            dataset_id = None if runtime is None else runtime.dataset_id
            if (
                not isinstance(dataset_id, str)
                or dataset_id.casefold() not in {"hotpotqa", "triviaqa"}
            ):
                raise AgentWorkflowStateError(
                    "qa_verified_answer_lineage_v2 requires runtime.dataset_id "
                    "to be 'hotpotqa' or 'triviaqa'"
                )
        if semantic_protocol == _HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL:
            dataset_id = None if runtime is None else runtime.dataset_id
            if not isinstance(dataset_id, str) or dataset_id.casefold() != "hotpotqa":
                raise AgentWorkflowStateError(
                    "hotpotqa_role_conditional_capabilities_v1 requires "
                    "runtime.dataset_id='hotpotqa'"
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
        self._repair_exhausted_agent_ids: set[str] = set()
        self._latest_failure_record_by_agent: dict[str, AgentFailureRecord] = {}
        self._pending_repair_receipt_count_by_agent: dict[str, int] = {}
        # Runtime-only SkillFlow continuation state.  Canvas snapshots do not
        # serialize Runtime results, but an in-process repair turn must retain
        # the failed Agent's public Action--Observation history and Tool
        # receipts so a contract edit does not repeat retrieval or discard
        # evidence already obtained on the current task.
        self._failure_continuations: dict[str, dict[str, object]] = {}
        self._last_valid_evidence_lineage: Optional[
            AgentWorkflowEvidenceLineageSnapshot
        ] = None
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

    @property
    def last_valid_evidence_lineage(
        self,
    ) -> Optional[AgentWorkflowEvidenceLineageSnapshot]:
        """Return the immutable last complete evidence lineage, if observed."""

        return self._last_valid_evidence_lineage

    def _uses_semantic_lineage_protocol(self) -> bool:
        return self.semantic_protocol in _SEMANTIC_LINEAGE_PROTOCOLS

    def _uses_role_conditional_capabilities(self) -> bool:
        """Return whether semantic roles are optional, per-Agent capabilities."""

        return self.semantic_protocol == _HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL

    def _role_conditional_registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Project the task Runtime's registered HotpotQA profiles.

        FlowSteer's action mask must describe the same executable domain that
        the Runtime validates after a Canvas edit.  SkillFlow supplies the
        bounded reasoning/ReAct executors and the task-scoped Tool registry;
        neither semantic role names nor graph topology create an executor.
        """

        if not self._uses_role_conditional_capabilities():
            return ()
        required_tool_id = self.required_evidence_tool_id
        return tuple(
            (execution_mode, allowed_tools)
            for execution_mode, allowed_tools in (
                self.runtime.registered_execution_profiles()
            )
            if execution_mode in {"reasoning", "react"}
            and allowed_tools in {(), (required_tool_id,)}
        )

    def _role_conditional_execution_profiles_for(
        self,
        role_family: str,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return role-compatible profiles without prescribing role order."""

        registered = self._role_conditional_registered_execution_profiles()
        role = role_family.casefold()
        if role == "format":
            return tuple(profile for profile in registered if not profile[1])
        if role == "evidence_retriever":
            return tuple(
                profile
                for profile in registered
                if profile == ("reasoning", ())
                or profile == (
                    "react",
                    (self.required_evidence_tool_id,),
                )
            )
        return registered

    def _role_conditional_execution_constraint(
        self,
        role_family: str,
    ) -> dict[str, object]:
        """Render one correlated execution-profile domain for the Director."""

        profiles = self._role_conditional_execution_profiles_for(role_family)
        execution_modes = tuple(
            dict.fromkeys(execution_mode for execution_mode, _ in profiles)
        )
        allowed_tool_sets = tuple(
            dict.fromkeys(allowed_tools for _, allowed_tools in profiles)
        )
        return {
            "execution_modes": list(execution_modes),
            "allowed_tools": [list(tool_ids) for tool_ids in allowed_tool_sets],
            "execution_profiles": [
                {
                    "execution_mode": execution_mode,
                    "allowed_tools": list(allowed_tools),
                }
                for execution_mode, allowed_tools in profiles
            ],
        }

    def _semantic_protocol_label(self) -> str:
        return (
            "HotpotQA"
            if self.semantic_protocol
            in {
                _HOTPOTQA_SEMANTIC_PROTOCOL,
                _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL,
                _HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL,
            }
            else "Evidence-grounded QA"
        )

    def _semantic_minimums(self) -> tuple[int, int]:
        """Return dataset-conditioned evidence and reasoning-chain minima."""

        if (
            self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
            and isinstance(self.runtime.dataset_id, str)
            and self.runtime.dataset_id.casefold() == "triviaqa"
        ):
            return 1, 1
        return 2, 2

    def model_admissible_action_types(self) -> Tuple[str, ...]:
        """Project state-conditioned Canvas actions for the Flow-Director.

        FlowSteer's progressive Canvas exposes the next legal editing boundary
        after each execute-and-feedback step.  Keep the configured action set
        authoritative in ``step`` while removing actions that cannot change or
        terminate the current public state from the next model observation.
        """

        finish_admitted = self.finish_admissibility().get("admissible") is True
        if (
            self._uses_semantic_lineage_protocol()
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

        provider_repair_ids = self._provider_repair_agent_ids()
        if (
            provider_repair_ids
            and self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            and AgentActionType.MODIFY_AGENT.value
            in self._allowed_action_type_set
        ):
            # A measured provider failure is an operational repair boundary,
            # independent of any dataset-specific semantic recovery protocol.
            # Keep the next progressive Canvas edit on the failed Agent's
            # model field so a dead provider cannot be routed around by
            # unrelated topology edits while it still blocks execution.
            return (AgentActionType.MODIFY_AGENT.value,)

        selected_output_recovery_relations = (
            self._selected_output_artifact_recovery_relation_candidates()
        )
        if (
            selected_output_recovery_relations
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # A revision-live, receipt-grounded semantic artifact already has
            # an independent route to the selected Output.  Route that exact
            # artifact into the measured failed ancestor before exposing the
            # generic relation/modify domains.  This is an ordinary FlowSteer
            # edit--execute--feedback boundary: the existing dirty closure
            # recomputes the repaired ancestor and terminal downstream.
            return (AgentActionType.SET_RELATION.value,)

        auxiliary_takeover_relations = (
            self._repair_exhausted_auxiliary_takeover_relation_candidates()
        )
        if (
            auxiliary_takeover_relations
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # The isolated replacement has now materialized the failed
            # auxiliary's exact public responsibility.  FlowSteer's next
            # progressive edit transfers one existing downstream edge before
            # any further augmentation.
            return (AgentActionType.SET_RELATION.value,)

        mandatory_repair_ids = self._mandatory_repair_agent_ids()
        if mandatory_repair_ids:
            if (
                AgentActionType.MODIFY_AGENT.value
                in self._allowed_action_type_set
            ):
                # A typed Runtime failure or a revision-local semantic-artifact
                # failure identifies an existing Agent that can still be repaired.
                # Keep FlowSteer's action mask on that measured repair boundary;
                # augmentation becomes available only after repair succeeds or a
                # typed bounded-failure replacement takeover is established.
                return (AgentActionType.MODIFY_AGENT.value,)
            # The authoritative preservation gate rejects every unrelated edit
            # while this measured repair remains unresolved.  Do not expose a
            # broader action which the same step boundary must reject.
            return ()

        takeover_delete_ids = (
            self._repair_exhausted_auxiliary_takeover_delete_ids()
        )
        dirty_replacement_ids = self._dirty_auxiliary_replacement_agent_ids()
        if (
            takeover_delete_ids
            and AgentActionType.DELETE_AGENT.value
            in self._allowed_action_type_set
        ):
            # A bounded ReAct repair has failed without advancing its public
            # Tool receipts and a same-role/same-artifact auxiliary has already
            # taken over every downstream responsibility.  Reuse FlowSteer's
            # existing replacement-takeover delete boundary before adding yet
            # another recovery branch.
            return (AgentActionType.DELETE_AGENT.value,)

        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        if exhausted_reasoner_ids:
            failed_ingress_candidates = (
                self._failed_auxiliary_ingress_relation_candidates()
            )
            if (
                failed_ingress_candidates
                and AgentActionType.SET_RELATION.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.SET_RELATION.value,)
            routing_candidates = self._repair_exhausted_relation_candidates()
            if (
                routing_candidates
                and AgentActionType.SET_RELATION.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.SET_RELATION.value,)

        if (
            dirty_replacement_ids
            and AgentActionType.MODIFY_AGENT.value
            in self._allowed_action_type_set
        ):
            # A replacement which already exists on the Canvas but has not
            # materialized its executable prefix is a live repair target, even
            # when another replacement ADD would exceed max_agents. Preserve
            # the measured node and repair it before considering augmentation.
            return (AgentActionType.MODIFY_AGENT.value,)

        node_count = len(self._graph.nodes)
        node_ids = tuple(node.id for node in self._graph.nodes)
        can_add = self.max_agents is None or node_count < self.max_agents
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        if self._uses_role_conditional_capabilities() and replacement_domains:
            # FlowSteer's next edit must agree with the authoritative
            # preservation gate.  A repair-exhausted auxiliary is replaced by
            # one isolated executable prefix before any relation, Output, or
            # unrelated Canvas mutation can consume another round.
            if (
                can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)
            return ()
        pending_ingress_ids = (
            self._pending_role_conditional_ingress_consumer_ids()
        )
        if pending_ingress_ids:
            ingress_relation_candidates = (
                self._role_conditional_ingress_relation_candidates()
            )
            if (
                ingress_relation_candidates
                and AgentActionType.SET_RELATION.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.SET_RELATION.value,)
            if (
                can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)
            return ()
        evidence_ingress_ids = (
            self._role_conditional_evidence_ingress_consumer_ids()
        )
        if evidence_ingress_ids:
            evidence_relation_candidates = (
                self._role_conditional_evidence_ingress_relation_candidates()
            )
            if (
                evidence_relation_candidates
                and AgentActionType.SET_RELATION.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.SET_RELATION.value,)
            if (
                can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)
            return ()
        missing_role_families = self._missing_semantic_role_families()
        if (
            self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
            and missing_role_families
        ):
            # A missing semantic responsibility is a Canvas-construction
            # boundary after any measured repair/takeover obligation. Complete
            # only the absent capability before SET_OUTPUT or unrelated edits
            # can materialize and protect an incomplete terminal lineage. This
            # fixes neither Agent count nor communication edges.
            if (
                can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)
            return ()

        terminal_reachability_candidates = (
            self._terminal_reachability_relation_candidates()
        )
        prospective_convergence_candidates = (
            self._prospective_terminal_convergence_relation_candidates()
        )
        if (
            prospective_convergence_candidates
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # FlowSteer's downstream Aggregate closes an already materialized
            # parallel block before terminal selection.  AgentGraph represents
            # that same progressive boundary as one exact monotonic relation,
            # repeated only while terminal-unreachable branches remain, then
            # followed by SET_OUTPUT after all branches have converged.
            return (AgentActionType.SET_RELATION.value,)
        if (
            terminal_reachability_candidates
            and self._model_admissible_relation_candidates()
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # FlowSteer's live action mask must expose only an edit that makes
            # progress on the current graph-validation diagnosis.  Falling
            # through to the generic relation domain here permits unrelated
            # rewrites while an orphan recovery branch remains unresolved.
            return (AgentActionType.SET_RELATION.value,)

        required_relation_candidates = (
            self._required_semantic_relation_candidates()
        )
        if (
            required_relation_candidates
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # Canvas admission requires the same exact edge before any recovery
            # augmentation, so expose that edge rather than an ADD action which
            # the authoritative gate would reject.
            return (AgentActionType.SET_RELATION.value,)

        missing_role_families = self._missing_semantic_role_families()
        if (
            exhausted_reasoner_ids
            and node_count
            and missing_role_families
            and can_add
            and AgentActionType.ADD_SUBGRAPH.value
            in self._allowed_action_type_set
        ):
            # Progressively complete a partially declared semantic spine before
            # adding another recovery branch.  The live ADD role domain below
            # exposes only the missing semantic responsibilities.
            return (AgentActionType.ADD_SUBGRAPH.value,)

        output_target_ids = self._model_admissible_output_agent_ids()
        if (
            self._uses_semantic_lineage_protocol()
            and output_target_ids
            and AgentActionType.SET_OUTPUT.value in self._allowed_action_type_set
        ):
            # Select a terminal-compatible capability only after the
            # prospective Canvas passes the same complete-graph and semantic
            # checks used by authoritative admission.  This does not require
            # a Formatter or any fixed role sequence.
            return (AgentActionType.SET_OUTPUT.value,)

        if exhausted_reasoner_ids:
            if (
                self._model_admissible_add_role_families()
                and can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)

        deletable_ids = tuple(
            node_id
            for node_id in node_ids
            if self._delete_admission_issue(node_id) is None
        )
        can_delete = bool(deletable_ids)
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

    def _all_model_admissible_relation_candidates(
        self,
        *,
        terminal_convergence_output_id: Optional[str] = None,
    ) -> list[dict[str, object]]:
        """Return every non-self, non-no-op relation edit accepted by Canvas."""

        node_ids = [node.id for node in self._graph.nodes]
        active_lineage = self._active_semantic_lineage_ids()
        declared_edges = tuple(
            edge
            for edge in self._required_semantic_edges()
            if edge[1] in self._directed_successors(self._graph, edge[0])
        )
        protected_edges = tuple(
            dict.fromkeys(
                (*zip(active_lineage, active_lineage[1:]), *declared_edges)
            )
        )
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
                    if self._output_sink_issue_for(candidate) is not None:
                        continue
                    if self._semantic_edit_issue_for(candidate) is not None:
                        continue
                    if (
                        self._preserved_input_change_issue_for(
                            candidate,
                            terminal_convergence_output_id=(
                                terminal_convergence_output_id
                            ),
                        )
                        is not None
                    ):
                        continue
                    if (
                        candidate.output_agent_id is not None
                        and self._uses_format_agent_protocol(candidate)
                        and self._format_agent_issue_for(candidate) is not None
                    ):
                        continue
                    if any(
                        target_id
                        not in self._directed_successors(candidate, source_id)
                        for source_id, target_id in protected_edges
                    ):
                        continue
                    encoded_source_id = source_id
                    encoded_target_id = target_id
                    encoded_source_to_target = source_to_target
                    encoded_target_to_source = target_to_source
                    if (
                        self._uses_semantic_lineage_protocol()
                        and not source_to_target
                        and target_to_source
                    ):
                        # Reuse the live ADD relation convention: a one-way
                        # semantic handoff is serialized as its actual sender
                        # to receiver with (true,false), not reversed endpoints
                        # with (false,true).
                        encoded_source_id = target_id
                        encoded_target_id = source_id
                        encoded_source_to_target = True
                        encoded_target_to_source = False
                    candidates.append(
                        {
                            "source_id": encoded_source_id,
                            "target_id": encoded_target_id,
                            "source_to_target": encoded_source_to_target,
                            "target_to_source": encoded_target_to_source,
                        }
                    )
        return candidates

    def _model_admissible_relation_candidates(self) -> list[dict[str, object]]:
        """Return the exact state-conditioned FlowSteer relation domain."""

        all_candidates = self._all_model_admissible_relation_candidates()
        selected_output_recovery = (
            self._selected_output_artifact_recovery_relation_candidates(
                all_candidates
            )
        )
        if selected_output_recovery:
            return selected_output_recovery
        auxiliary_takeover = (
            self._repair_exhausted_auxiliary_takeover_relation_candidates(
                all_candidates
            )
        )
        if auxiliary_takeover:
            return auxiliary_takeover
        pending_ingress_ids = (
            self._pending_role_conditional_ingress_consumer_ids()
        )
        if pending_ingress_ids:
            return self._role_conditional_ingress_relation_candidates(
                all_candidates
            )
        evidence_ingress_ids = (
            self._role_conditional_evidence_ingress_consumer_ids()
        )
        if evidence_ingress_ids:
            return self._role_conditional_evidence_ingress_relation_candidates(
                all_candidates
            )
        failed_ingress_candidates = (
            self._failed_auxiliary_ingress_relation_candidates(all_candidates)
        )
        if failed_ingress_candidates:
            return failed_ingress_candidates
        routing_candidates = self._repair_exhausted_relation_candidates(
            all_candidates
        )
        if routing_candidates:
            return routing_candidates
        prospective_convergence_candidates = (
            self._prospective_terminal_convergence_relation_candidates()
        )
        if prospective_convergence_candidates:
            return prospective_convergence_candidates
        terminal_reachability_candidates = (
            self._terminal_reachability_relation_candidates(all_candidates)
        )
        if terminal_reachability_candidates:
            return terminal_reachability_candidates
        required_candidates = self._required_semantic_relation_candidates(
            all_candidates
        )
        if required_candidates:
            return required_candidates
        return [
            item
            for item in all_candidates
            if not self._relation_reintroduces_failed_auxiliary_ingress(item)
        ]

    def _selected_output_artifact_recovery_sources_by_target(
        self,
    ) -> dict[str, Tuple[str, ...]]:
        """Map a blocked failed ancestor to independent grounded artifacts.

        FlowSteer's cached branch result remains public after a sibling
        failure, and SkillFlow keeps its successful Tool receipt.  AgentGraph
        needs one thin projection of that state: after repair of an existing
        failed ancestor has been exhausted, an independently routed semantic
        artifact may be handed to that same ancestor as one ordinary directed
        relation.  This method never changes Output identity, selects an
        answer, or requires a named role.
        """

        output_id = self._graph.output_agent_id
        if (
            not self._uses_role_conditional_capabilities()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            or output_id is None
            or not self._graph.has_node(output_id)
            or self.required_evidence_tool_id is None
        ):
            return {}

        output_ancestors = set(
            self._directed_ancestor_ids(self._graph, output_id)
        )
        failed_targets = tuple(
            node.id
            for node in self._graph.nodes
            if node.id in output_ancestors
            and node.id in self._failed_agent_ids
            and node.id in self._repair_exhausted_agent_ids
            and node.id not in self._diagnosed_unusable_agent_ids
        )
        if not failed_targets:
            return {}

        def reaches_output_without(source_id: str, excluded_id: str) -> bool:
            frontier = [source_id]
            visited: set[str] = set()
            while frontier:
                current = frontier.pop()
                if current == excluded_id or current in visited:
                    continue
                if current == output_id:
                    return True
                visited.add(current)
                frontier.extend(
                    successor_id
                    for successor_id in self._directed_successors(
                        self._graph,
                        current,
                    )
                    if successor_id != excluded_id
                )
            return False

        candidates_by_source: dict[str, str] = {}
        for source in self._graph.nodes:
            source_id = source.id
            source_role = (source.role_family or "").casefold()
            if (
                source_id == output_id
                or source_id in self._failed_agent_ids
                or source_id in self._repair_exhausted_agent_ids
                or source_id in self._unresolved_dirty_agents
                or not self._has_successful_artifact(source_id)
            ):
                continue
            artifact = self._progressive_outputs.get(source_id)
            owner_ids = (
                *self._directed_ancestor_ids(self._graph, source_id),
                source_id,
            )
            evidence_texts: list[str] = []
            for owner_id in owner_ids:
                metadata = self._progressive_output_metadata.get(owner_id)
                if not isinstance(metadata, Mapping):
                    continue
                if (
                    metadata.get("artifact_status")
                    == _NON_TERMINAL_PARTIAL_ARTIFACT
                ):
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
            if not isinstance(artifact, str):
                continue
            if source_role == "reasoner":
                candidate, issue = self._reasoner_candidate_for_current_dataset(
                    artifact
                )
                if (
                    issue is not None
                    or candidate is None
                    or self._reasoner_evidence_provenance_issue(
                        artifact,
                        evidence_texts,
                        require_answer_binding=True,
                    )
                    is not None
                ):
                    continue
            elif source_role == "verifier":
                candidate, issue = self._verifier_candidate(artifact)
                if issue is not None or candidate is None:
                    continue
                producer_candidates: list[str] = []
                for producer_id in self._directed_ancestor_ids(
                    self._graph,
                    source_id,
                ):
                    producer = self._graph.get_node(producer_id)
                    producer_role = (producer.role_family or "").casefold()
                    if producer_role in {
                        "evidence_retriever",
                        "verifier",
                        "format",
                        "output",
                    }:
                        continue
                    producer_artifact = self._progressive_outputs.get(
                        producer_id
                    )
                    if producer_role == "reasoner":
                        producer_candidate, producer_issue = (
                            self._reasoner_candidate_for_current_dataset(
                                producer_artifact or ""
                            )
                        )
                    else:
                        producer_candidate, producer_issue = (
                            self._semantic_candidate_from_artifact(
                                producer_artifact
                            )
                        )
                    if (
                        producer_issue is None
                        and producer_candidate is not None
                    ):
                        producer_candidates.append(producer_candidate)
                if candidate not in producer_candidates:
                    continue
            elif source_role in {"evidence_retriever", "format", "output"}:
                continue
            else:
                candidate, issue = self._semantic_candidate_from_artifact(
                    artifact
                )
                if issue is not None or candidate is None:
                    continue
            if not any(
                self._contains_lexical_span(text, candidate)
                for text in evidence_texts
            ):
                continue
            candidates_by_source[source_id] = candidate

        # Conflicting grounded candidates require diagnosis or verification;
        # routing either one would make the action mask select the answer.
        if not candidates_by_source or len(set(candidates_by_source.values())) != 1:
            return {}
        result: dict[str, Tuple[str, ...]] = {}
        for target_id in failed_targets:
            sources = tuple(
                node.id
                for node in self._graph.nodes
                if node.id in candidates_by_source
                and node.id != target_id
                and target_id
                not in self._directed_ancestor_ids(
                    self._graph,
                    node.id,
                )
                and reaches_output_without(node.id, target_id)
            )
            if sources:
                result[target_id] = sources
        return result

    def _selected_output_artifact_recovery_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Return exact ``artifact source -> failed ancestor`` Canvas edits."""

        sources_by_target = (
            self._selected_output_artifact_recovery_sources_by_target()
        )
        if not sources_by_target:
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        result: list[dict[str, object]] = []
        for item in source_candidates:
            source_id = str(item["source_id"])
            target_id = str(item["target_id"])
            if (
                source_id not in sources_by_target.get(target_id, ())
                or item["source_to_target"] is not True
                or item["target_to_source"] is not False
            ):
                continue
            previous = self._graph.relation_bits(source_id, target_id)
            if not previous.is_independent:
                # Recovery adds one monotonic handoff.  It never removes or
                # reverses an existing communication edge.
                continue
            candidate = self._graph.fork()
            candidate.set_relation(source_id, target_id, True, False)
            if not candidate.validate(
                self.model_registry,
                require_complete=True,
            ).valid:
                continue
            if self._output_sink_issue_for(candidate) is not None:
                continue
            if self._semantic_edit_issue_for(candidate) is not None:
                continue
            result.append(dict(item))
        return result

    def _pending_role_conditional_ingress_consumer_ids(self) -> Tuple[str, ...]:
        """Return selected semantic consumers deferred for lack of routed input."""

        if not self._uses_role_conditional_capabilities():
            return ()
        execution = self._progressive_execution
        if (
            execution is None
            or self._progressive_execution_revision != self._graph.revision
        ):
            return ()
        deferred_ids = set(execution.deferred_agent_ids)
        return tuple(
            node.id
            for node in self._graph.nodes
            if node.id in deferred_ids
            and (node.role_family or "").casefold() in {"verifier", "format"}
            and not self._graph.directed_predecessors(node.id)
        )

    def _role_conditional_evidence_ingress_consumer_ids(
        self,
    ) -> Tuple[str, ...]:
        """Return routed semantic consumers missing required Tool evidence.

        This is a revision-local FlowSteer repair projection over the actual
        SkillFlow execution receipt. It does not require a Reasoner in the
        general search space; it checks only semantic-candidate and checking
        capabilities that the Director has actually selected and executed.
        """

        if (
            not self._uses_role_conditional_capabilities()
            or self.required_evidence_tool_id is None
        ):
            return ()
        execution = self._cached_progressive_execution()
        output_id = self._graph.output_agent_id
        if (
            execution is None
            or execution.final_answer is None
            or output_id is None
            or not self._graph.has_node(output_id)
        ):
            return ()
        routed_ids = (
            *self._directed_ancestor_ids(self._graph, output_id),
            output_id,
        )
        missing_ids: list[str] = []
        for agent_id in routed_ids:
            node = self._graph.get_node(agent_id)
            role = (node.role_family or "").casefold()
            semantic_candidate, _ = self._semantic_candidate_from_artifact(
                execution.outputs.get(agent_id)
            )
            selected_semantic_capability = role == "reasoner" or (
                role
                not in {"evidence_retriever", "verifier", "format", "output"}
                and semantic_candidate is not None
            )
            if (
                agent_id == output_id
                or not selected_semantic_capability
                or not self._has_successful_artifact(agent_id)
            ):
                continue
            evidence_owner_ids = (
                *self._directed_ancestor_ids(self._graph, agent_id),
                agent_id,
            )
            if self._successful_read_texts_for_agents(
                execution,
                evidence_owner_ids,
            ):
                continue
            if any(
                self._graph.get_node(owner_id).execution_mode.value == "react"
                and self.required_evidence_tool_id
                in self._graph.get_node(owner_id).allowed_tools
                for owner_id in evidence_owner_ids
            ):
                # A Tool-capable node already assigned to this semantic
                # consumer is a repair target, not a reason to add a duplicate.
                continue
            missing_ids.append(agent_id)
        if missing_ids:
            return tuple(missing_ids)
        if self._successful_read_texts_for_agents(execution, routed_ids):
            return ()
        output_node = self._graph.get_node(output_id)
        if (
            (output_node.role_family or "").casefold() == "output"
            and self._has_successful_artifact(output_id)
        ):
            # Reasoner is optional. A generic Output capability may consume a
            # newly materialized evidence artifact and rerun in place.
            return (output_id,)
        return ()

    def _role_conditional_evidence_ingress_candidate(
        self,
        candidate: AgentGraph,
    ) -> bool:
        """Match one Retriever-to-selected-consumer augmentation transaction."""

        consumer_ids = set(
            self._role_conditional_evidence_ingress_consumer_ids()
        )
        if (
            not consumer_ids
            or candidate.output_agent_id != self._graph.output_agent_id
        ):
            return False
        current_ids = {node.id for node in self._graph.nodes}
        new_nodes = tuple(
            node for node in candidate.nodes if node.id not in current_ids
        )
        if (
            len(new_nodes) != 1
            or len(candidate.nodes) != len(self._graph.nodes) + 1
        ):
            return False
        retriever = new_nodes[0]
        if (
            (retriever.role_family or "").casefold() != "evidence_retriever"
            or retriever.execution_mode.value != "react"
            or retriever.allowed_tools != (self.required_evidence_tool_id,)
        ):
            return False
        changed_consumer_ids: list[str] = []
        for node in self._graph.nodes:
            before = tuple(self._graph.directed_predecessors(node.id))
            after = tuple(candidate.directed_predecessors(node.id))
            if before == after:
                continue
            if (
                node.id not in consumer_ids
                or set(after) != {*before, retriever.id}
            ):
                return False
            changed_consumer_ids.append(node.id)
        return (
            len(changed_consumer_ids) == 1
            and candidate.relation_bits(
                retriever.id,
                changed_consumer_ids[0],
            ).source_to_target
            and not candidate.relation_bits(
                retriever.id,
                changed_consumer_ids[0],
            ).target_to_source
            and set(self._directed_successors(candidate, retriever.id))
            == {changed_consumer_ids[0]}
            and len(candidate.relations) == len(self._graph.relations) + 1
        )

    def _role_conditional_existing_evidence_ingress_candidate(
        self,
        candidate: AgentGraph,
    ) -> bool:
        """Match one existing receipt-bearing Agent ingress relation."""

        consumer_ids = set(
            self._role_conditional_evidence_ingress_consumer_ids()
        )
        execution = self._cached_progressive_execution()
        if (
            not consumer_ids
            or execution is None
            or candidate.output_agent_id != self._graph.output_agent_id
            or tuple(node.id for node in candidate.nodes)
            != tuple(node.id for node in self._graph.nodes)
        ):
            return False
        source_ids = tuple(
            node.id
            for node in self._graph.nodes
            if self._has_successful_artifact(node.id)
            and self._successful_read_texts_for_agents(execution, (node.id,))
        )
        for source_id in source_ids:
            before_successors = set(
                self._directed_successors(self._graph, source_id)
            )
            after_successors = set(
                self._directed_successors(candidate, source_id)
            )
            added_consumers = (
                after_successors - before_successors
            ).intersection(consumer_ids)
            if len(added_consumers) != 1:
                continue
            consumer_id = next(iter(added_consumers))
            before_bits = self._graph.relation_bits(source_id, consumer_id)
            after_bits = candidate.relation_bits(source_id, consumer_id)
            if (
                before_bits.source_to_target
                or not after_bits.source_to_target
                or after_bits.target_to_source != before_bits.target_to_source
            ):
                continue
            if all(
                tuple(self._graph.directed_predecessors(node.id))
                == tuple(candidate.directed_predecessors(node.id))
                for node in self._graph.nodes
                if node.id != consumer_id
            ):
                return True
        return False

    def _role_conditional_evidence_ingress_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Route one existing receipt-bearing artifact to a missing consumer."""

        if not self._role_conditional_evidence_ingress_consumer_ids():
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        result: list[dict[str, object]] = []
        for item in source_candidates:
            candidate = self._graph.fork()
            candidate.set_relation(
                str(item["source_id"]),
                str(item["target_id"]),
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            if self._role_conditional_existing_evidence_ingress_candidate(
                candidate
            ):
                result.append(dict(item))
        return result

    def _role_conditional_evidence_ingress_admission_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Keep a measured missing-evidence turn on one executable ADD edit."""

        consumer_ids = set(
            self._role_conditional_evidence_ingress_consumer_ids()
        )
        if not consumer_ids:
            return None
        relation_candidates = (
            self._role_conditional_evidence_ingress_relation_candidates()
        )
        if relation_candidates:
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in relation_candidates
            ):
                return None
            return (
                "route one existing successful qa-retrieval artifact into one "
                "selected semantic consumer before adding another Retriever; "
                "required_evidence_ingress_consumer_agent_ids="
                f"{sorted(consumer_ids)!r}"
            )
        if (
            action.action_type is AgentActionType.ADD_SUBGRAPH
            and len(action.agents) == 1
            and (action.agents[0].role_family or "").casefold()
            == "evidence_retriever"
            and action.agents[0].execution_mode == "react"
            and action.agents[0].allowed_tools
            == (self.required_evidence_tool_id,)
            and action.output_agent_id is None
            and len(action.relations) == 1
        ):
            relation = action.relations[0]
            new_id = action.agents[0].agent_id
            supplies_consumer = (
                relation.source_id == new_id
                and relation.target_id in consumer_ids
                and relation.source_to_target is True
                and relation.target_to_source is False
            )
            if supplies_consumer:
                return None
        return (
            "add exactly one qa-retrieval ReAct Evidence Retriever and route "
            "its receipt-grounded artifact into one selected tool-free "
            "semantic consumer while preserving the current Output identity; "
            "required_evidence_ingress_consumer_agent_ids="
            f"{sorted(consumer_ids)!r}"
        )

    def _role_conditional_ingress_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Route one materialized existing artifact into a deferred consumer."""

        pending_ids = set(
            self._pending_role_conditional_ingress_consumer_ids()
        )
        execution = self._progressive_execution
        if not pending_ids or execution is None:
            return []
        materialized_ids = set(execution.outputs)
        semantic_materialized_ids = {
            agent_id
            for agent_id in materialized_ids
            if self._semantic_candidate_from_artifact(
                execution.outputs.get(agent_id)
            )[0]
            is not None
        }
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        result: list[dict[str, object]] = []
        for item in source_candidates:
            candidate = self._graph.fork()
            candidate.set_relation(
                str(item["source_id"]),
                str(item["target_id"]),
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            if any(
                consumer_id in self._directed_successors(candidate, source_id)
                and consumer_id
                not in self._directed_successors(self._graph, source_id)
                and (
                    (
                        self._graph.get_node(consumer_id).role_family or ""
                    ).casefold()
                    not in {"verifier", "format"}
                    or source_id in semantic_materialized_ids
                )
                for source_id in materialized_ids
                for consumer_id in pending_ids
            ):
                result.append(dict(item))
        return result

    @staticmethod
    def _relation_action_matches_candidate(
        action: AgentAction,
        candidate: Mapping[str, object],
    ) -> bool:
        return (
            action.action_type is AgentActionType.SET_RELATION
            and action.source_id == candidate.get("source_id")
            and action.target_id == candidate.get("target_id")
            and action.source_to_target
            is candidate.get("source_to_target")
            and action.target_to_source
            is candidate.get("target_to_source")
        )

    def _semantic_role_agent_ids(self, role_family: str) -> Tuple[str, ...]:
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold() == role_family
            and node.id not in self._diagnosed_unusable_agent_ids
        )

    def _required_semantic_edges(self) -> Tuple[Tuple[str, str], ...]:
        """Return the declared semantic terminal dataflow when unambiguous."""

        if not self._uses_semantic_lineage_protocol():
            return ()
        if self._uses_role_conditional_capabilities():
            return ()
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            # The v2 terminal contract is expressed over routed artifacts and
            # provenance.  It intentionally does not prescribe direct role
            # adjacency: retrieval/repair fan-in, intermediate reasoning, and
            # reciprocal two-Agent blocks remain in FlowSteer's search space.
            return ()
        reasoner_ids = self._semantic_role_agent_ids("reasoner")
        verifier_ids = self._semantic_role_agent_ids("verifier")
        formatter_ids = self._semantic_role_agent_ids("format")
        if not (
            len(reasoner_ids) == len(verifier_ids) == len(formatter_ids) == 1
        ):
            return ()
        return (
            (reasoner_ids[0], verifier_ids[0]),
            (verifier_ids[0], formatter_ids[0]),
        )

    def _required_semantic_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Return edits that close one missing declared semantic dataflow edge."""

        required_edges = self._required_semantic_edges()
        missing_edges = tuple(
            edge
            for edge in required_edges
            if edge[1] not in self._directed_successors(self._graph, edge[0])
        )
        if not missing_edges:
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        result: list[dict[str, object]] = []
        for item in source_candidates:
            candidate = self._graph.fork()
            candidate.set_relation(
                str(item["source_id"]),
                str(item["target_id"]),
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            remaining = tuple(
                edge
                for edge in required_edges
                if edge[1] not in self._directed_successors(candidate, edge[0])
            )
            if len(remaining) < len(missing_edges):
                result.append(dict(item))
        return result

    def _recovery_auxiliary_agent_ids(self) -> Tuple[str, ...]:
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold()
            in {"evidence_retriever", "repair"}
        )

    def _repair_exhausted_reasoner_ids(self) -> Tuple[str, ...]:
        return tuple(
            agent_id
            for agent_id in self._semantic_role_agent_ids("reasoner")
            if agent_id in self._repair_exhausted_agent_ids
        )

    def _has_recovery_ingress(
        self,
        exhausted_reasoner_ids: Sequence[str],
    ) -> bool:
        return any(
            reasoner_id in self._directed_successors(self._graph, auxiliary_id)
            and auxiliary_id
            not in self._directed_successors(self._graph, reasoner_id)
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
            for reasoner_id in exhausted_reasoner_ids
        )

    def _repair_exhausted_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Route one successful auxiliary artifact into an exhausted Reasoner."""

        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        auxiliary_ids = tuple(
            auxiliary_id
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
            if auxiliary_id not in self._failed_agent_ids
            and auxiliary_id not in self._repair_exhausted_agent_ids
            and auxiliary_id not in self._unresolved_dirty_agents
            and self._has_successful_artifact(auxiliary_id)
            and self._semantic_replacement_has_valid_artifact(
                auxiliary_id,
                (
                    self._graph.get_node(auxiliary_id).role_family or ""
                ).casefold(),
            )
        )
        unrouted_auxiliary_ids = tuple(
            auxiliary_id
            for auxiliary_id in auxiliary_ids
            if not any(
                reasoner_id
                in self._directed_successors(self._graph, auxiliary_id)
                and auxiliary_id
                not in self._directed_successors(self._graph, reasoner_id)
                for reasoner_id in exhausted_reasoner_ids
            )
        )
        if (
            not exhausted_reasoner_ids
            or not unrouted_auxiliary_ids
        ):
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        current_unreachable = set(self._terminal_unreachable_agent_ids())
        result: list[dict[str, object]] = []
        for item in source_candidates:
            source_id = str(item["source_id"])
            target_id = str(item["target_id"])
            if (
                source_id not in unrouted_auxiliary_ids
                or target_id not in exhausted_reasoner_ids
                or item["source_to_target"] is not True
                or item["target_to_source"] is not False
            ):
                continue
            candidate = self._graph.fork()
            candidate.set_relation(
                source_id,
                target_id,
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            candidate_unreachable = {
                agent_id
                for issue in candidate.validate(
                    self.model_registry,
                    require_complete=True,
                ).issues
                if issue.code == "cannot_reach_output"
                for agent_id in issue.agent_ids
            }
            if candidate_unreachable < current_unreachable:
                result.append(dict(item))
        return result

    def _repair_exhausted_auxiliary_same_profile_replacements(
        self,
    ) -> dict[str, Tuple[str, ...]]:
        """Map a failed auxiliary to its measured same-profile replacements.

        SkillFlow exposes the continuation source and public Tool receipts on
        the successful replacement artifact.  That receipt is the measured
        handoff required by FlowSteer's next progressive Canvas edit; role,
        artifact type, and execution profile must remain identical.
        """

        if (
            not self._uses_role_conditional_capabilities()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
        ):
            return {}
        replacements_by_source: dict[str, Tuple[str, ...]] = {}
        for source in self._graph.nodes:
            source_role = (source.role_family or "").casefold()
            if (
                source_role not in {"evidence_retriever", "repair"}
                or source.id not in self._failed_agent_ids
                or source.id not in self._repair_exhausted_agent_ids
            ):
                continue
            source_profile = (
                source.execution_mode.value,
                tuple(source.allowed_tools),
            )
            replacement_ids: list[str] = []
            for replacement in self._graph.nodes:
                metadata = self._progressive_output_metadata.get(
                    replacement.id,
                    {},
                )
                replacement_profile = (
                    replacement.execution_mode.value,
                    tuple(replacement.allowed_tools),
                )
                if (
                    replacement.id == source.id
                    or (replacement.role_family or "").casefold()
                    != source_role
                    or replacement.artifact_type.casefold()
                    != source.artifact_type.casefold()
                    or replacement_profile != source_profile
                    or replacement.id in self._failed_agent_ids
                    or replacement.id in self._repair_exhausted_agent_ids
                    or replacement.id in self._unresolved_dirty_agents
                    or not self._has_successful_artifact(replacement.id)
                    or not self._semantic_replacement_has_valid_artifact(
                        replacement.id,
                        source_role,
                    )
                    or not isinstance(metadata, Mapping)
                    or metadata.get("continuation_source_agent_id")
                    != source.id
                    or not self._replacement_preserves_protected_history(
                        source.id,
                        replacement.id,
                    )
                ):
                    continue
                if (
                    self.required_evidence_tool_id is not None
                    and self.required_evidence_tool_id
                    in replacement.allowed_tools
                    and not self._successful_evidence_texts_from_metadata(
                        metadata
                    )
                ):
                    continue
                replacement_ids.append(replacement.id)
                break
            if replacement_ids:
                replacements_by_source[source.id] = tuple(replacement_ids)
        return replacements_by_source

    def _repair_exhausted_auxiliary_takeover_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Route one measured replacement to one existing downstream duty."""

        replacements_by_source = (
            self._repair_exhausted_auxiliary_same_profile_replacements()
        )
        if not replacements_by_source:
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        required_edges = {
            (replacement_id, successor_id)
            for source_id, replacement_ids in replacements_by_source.items()
            for successor_id in self._directed_successors(
                self._graph,
                source_id,
            )
            for replacement_id in replacement_ids
            if successor_id
            not in self._directed_successors(self._graph, replacement_id)
        }
        result: list[dict[str, object]] = []
        for item in source_candidates:
            edge = (str(item["source_id"]), str(item["target_id"]))
            if (
                edge not in required_edges
                or item["source_to_target"] is not True
                or item["target_to_source"] is not False
                or not self._graph.relation_bits(*edge).is_independent
            ):
                continue
            result.append(dict(item))
        return result

    def _terminal_reachability_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Return only relation edits that strictly reduce unreachable Agents."""

        current_unreachable = set(self._terminal_unreachable_agent_ids())
        if not current_unreachable:
            return []
        if self._repair_exhausted_reasoner_ids():
            # Exhausted-Reasoner recovery has a narrower relation projection:
            # detach measured unavailable ingress, or route one already
            # materialized auxiliary through `_repair_exhausted_relation_candidates`.
            # The generic reachability domain must not introduce a peer edge or
            # reciprocal cycle around that measured failure.
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        result: list[dict[str, object]] = []
        for item in source_candidates:
            if self._relation_reintroduces_failed_auxiliary_ingress(item):
                continue
            candidate = self._graph.fork()
            candidate.set_relation(
                str(item["source_id"]),
                str(item["target_id"]),
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            candidate_validation = candidate.validate(
                self.model_registry,
                require_complete=True,
            )
            candidate_unreachable = {
                agent_id
                for issue in candidate_validation.issues
                if issue.code == "cannot_reach_output"
                for agent_id in issue.agent_ids
            }
            if candidate_unreachable < current_unreachable:
                result.append(dict(item))
        return result

    def _prospective_terminal_output_agent_ids(self) -> Tuple[str, ...]:
        """Return successful terminal capabilities blocked only by fan-in."""

        if (
            not self._uses_role_conditional_capabilities()
            or self._graph.output_agent_id is not None
        ):
            return ()
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold() in {"format", "output"}
            and self._has_successful_artifact(node.id)
            and node.id not in self._failed_agent_ids
            and node.id not in self._repair_exhausted_agent_ids
        )

    def _prospective_terminal_convergence_relation_candidates(
        self,
    ) -> list[dict[str, object]]:
        """Project one monotonic fan-in edge before Output assignment.

        FlowSteer places an Aggregate after a parallel block and executes that
        edit before terminal submission.  In AgentGraph, an already selected
        generic Output or optional Formatter can serve as that downstream
        convergence point.  Admit only a single new source-to-sink edge whose
        source artifact is revision-live and which strictly reduces the
        terminal-unreachable set.  The final edge must also pass the unchanged
        complete-Canvas and semantic gates before SET_OUTPUT is exposed.
        """

        if self._graph.output_agent_id is not None:
            return []
        before_edges = {
            edge
            for relation in self._graph.relations
            for edge in relation.directed_edges()
        }
        result: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for output_id in self._prospective_terminal_output_agent_ids():
            before_output = self._graph.fork()
            try:
                before_output.set_output(output_id)
            except GraphMutationError:
                continue
            before_validation = before_output.validate(
                self.model_registry,
                require_complete=True,
            )
            if any(
                issue.code != "cannot_reach_output"
                for issue in before_validation.issues
            ):
                continue
            before_unreachable = {
                agent_id
                for issue in before_validation.issues
                if issue.code == "cannot_reach_output"
                for agent_id in issue.agent_ids
            }
            if not before_unreachable:
                continue
            candidates = self._all_model_admissible_relation_candidates(
                terminal_convergence_output_id=output_id,
            )
            for item in candidates:
                if (
                    item.get("source_to_target") is not True
                    or item.get("target_to_source") is not False
                ):
                    # FlowSteer's downstream Aggregate is a directed fan-in.
                    # Never turn an existing reverse edge into a reciprocal
                    # block while projecting terminal convergence.
                    continue
                candidate = self._graph.fork()
                try:
                    candidate.set_relation(
                        str(item["source_id"]),
                        str(item["target_id"]),
                        bool(item["source_to_target"]),
                        bool(item["target_to_source"]),
                    )
                except GraphMutationError:
                    continue
                after_edges = {
                    edge
                    for relation in candidate.relations
                    for edge in relation.directed_edges()
                }
                added_edges = after_edges - before_edges
                if before_edges - after_edges or len(added_edges) != 1:
                    continue
                source_id, target_id = next(iter(added_edges))
                if (
                    target_id != output_id
                    or source_id not in before_unreachable
                    or not self._has_successful_artifact(source_id)
                    or source_id in self._failed_agent_ids
                    or source_id in self._repair_exhausted_agent_ids
                ):
                    continue
                prospective = candidate.fork()
                try:
                    prospective.set_output(output_id)
                except GraphMutationError:
                    continue
                validation = prospective.validate(
                    self.model_registry,
                    require_complete=True,
                )
                if any(
                    issue.code != "cannot_reach_output"
                    for issue in validation.issues
                ):
                    continue
                after_unreachable = {
                    agent_id
                    for issue in validation.issues
                    if issue.code == "cannot_reach_output"
                    for agent_id in issue.agent_ids
                }
                if not after_unreachable < before_unreachable:
                    continue
                if self._output_sink_issue_for(prospective) is not None:
                    continue
                if after_unreachable:
                    if (
                        candidate.get_node(output_id).role_family or ""
                    ).casefold() != "output":
                        # A pure Formatter cannot aggregate multiple semantic
                        # branches. Leave that optional capability unchanged
                        # and require a generic Output convergence point.
                        continue
                else:
                    if self._semantic_edit_issue_for(prospective) is not None:
                        continue
                    if (
                        self._uses_format_agent_protocol(prospective)
                        and self._format_agent_issue_for(prospective) is not None
                    ):
                        continue
                key = (
                    item["source_id"],
                    item["target_id"],
                    item["source_to_target"],
                    item["target_to_source"],
                )
                if key not in seen:
                    result.append(dict(item))
                    seen.add(key)
        return result

    def _terminal_reachability_admission_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Align authoritative Canvas admission with its live repair domain."""

        candidates = self._terminal_reachability_relation_candidates()
        if not candidates:
            return None
        if any(
            self._relation_action_matches_candidate(action, candidate)
            for candidate in candidates
        ):
            return None
        return (
            "repair terminal reachability with an exact relation that strictly "
            "reduces terminal_unreachable_agent_ids before other Canvas edits; "
            "admissible_relation_candidates="
            + json.dumps(
                candidates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _role_conditional_ingress_admission_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Require one executable upstream handoff for a deferred consumer."""

        pending_ids = set(
            self._pending_role_conditional_ingress_consumer_ids()
        )
        if not pending_ids:
            return None
        relation_candidates = (
            self._role_conditional_ingress_relation_candidates()
        )
        if relation_candidates:
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in relation_candidates
            ):
                return None
            return (
                "route one already materialized Agent artifact into a deferred "
                "semantic consumer before other Canvas edits; "
                "required_ingress_consumer_agent_ids="
                f"{sorted(pending_ids)!r}"
            )
        if action.action_type is not AgentActionType.ADD_SUBGRAPH:
            return (
                "add one schedulable upstream producer and route its artifact "
                "into a deferred semantic consumer; "
                "required_ingress_consumer_agent_ids="
                f"{sorted(pending_ids)!r}"
            )
        schedulable_new_ids = {
            spec.agent_id
            for spec in action.agents
            if (spec.role_family or "").casefold() not in {"verifier", "format"}
        }
        has_ingress = any(
            (
                relation.source_id in schedulable_new_ids
                and relation.target_id in pending_ids
                and relation.source_to_target is True
            )
            or (
                relation.target_id in schedulable_new_ids
                and relation.source_id in pending_ids
                and relation.target_to_source is True
            )
            for relation in action.relations
        )
        if has_ingress:
            return None
        return (
            "ADD_SUBGRAPH must route at least one newly added schedulable "
            "producer into a deferred semantic consumer; no producer role or "
            "serial role order is prescribed. "
            "required_ingress_consumer_agent_ids="
            f"{sorted(pending_ids)!r}"
        )

    def _failed_auxiliary_ingress_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Detach only a failed fan-in edge when terminal reachability survives.

        AgentRuntime executes fan-in as an AND dependency.  A failed auxiliary
        ingress must therefore not remain attached beside an already successful
        auxiliary artifact.  FlowSteer's Canvas relation edit remains the only
        operation here: no OR-join or inactive-node state is introduced.  Since
        FINISH requires every Agent block to reach Output, expose this narrow
        relation repair only when the prospective graph is still complete.
        """

        if (
            not self._uses_semantic_lineage_protocol()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
        ):
            return []
        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        if not exhausted_reasoner_ids:
            return []
        successful_ingress = {
            (auxiliary_id, reasoner_id)
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
            if self._has_successful_artifact(auxiliary_id)
            for reasoner_id in exhausted_reasoner_ids
            if reasoner_id
            in self._directed_successors(self._graph, auxiliary_id)
            and auxiliary_id
            not in self._directed_successors(self._graph, reasoner_id)
        }
        unavailable_ids = (
            self._failed_agent_ids
            | self._repair_exhausted_agent_ids
            | self._unresolved_dirty_agents
        )
        failed_ingress = {
            (auxiliary_id, reasoner_id)
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
            if auxiliary_id in unavailable_ids
            for reasoner_id in exhausted_reasoner_ids
            if reasoner_id
            in self._directed_successors(self._graph, auxiliary_id)
        }
        if not successful_ingress or not failed_ingress:
            return []
        source_candidates = (
            self._all_model_admissible_relation_candidates()
            if candidates is None
            else [dict(item) for item in candidates]
        )
        result: list[dict[str, object]] = []
        for item in source_candidates:
            source_id = str(item["source_id"])
            target_id = str(item["target_id"])
            previous = self._graph.relation_bits(source_id, target_id)
            candidate = self._graph.fork()
            candidate.set_relation(
                source_id,
                target_id,
                bool(item["source_to_target"]),
                bool(item["target_to_source"]),
            )
            if not candidate.validate(
                self.model_registry,
                require_complete=True,
            ).valid:
                continue
            candidate_successful_ingress = {
                edge
                for edge in successful_ingress
                if edge[1]
                in self._directed_successors(candidate, edge[0])
                and edge[0]
                not in self._directed_successors(candidate, edge[1])
            }
            if candidate_successful_ingress != successful_ingress:
                continue
            candidate_failed_ingress = {
                edge
                for edge in failed_ingress
                if edge[1] in self._directed_successors(candidate, edge[0])
            }
            previous_edges = {
                edge
                for edge, present in (
                    ((source_id, target_id), previous.source_to_target),
                    ((target_id, source_id), previous.target_to_source),
                )
                if present
            }
            candidate_bits = candidate.relation_bits(source_id, target_id)
            candidate_edges = {
                edge
                for edge, present in (
                    ((source_id, target_id), candidate_bits.source_to_target),
                    ((target_id, source_id), candidate_bits.target_to_source),
                )
                if present
            }
            if (
                candidate_failed_ingress < failed_ingress
                and candidate_edges < previous_edges
            ):
                result.append(dict(item))
        return result

    def _relation_reintroduces_failed_auxiliary_ingress(
        self,
        item: Mapping[str, object],
    ) -> bool:
        """Reject reattaching an unavailable helper to an exhausted Reasoner."""

        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        if not exhausted_reasoner_ids:
            return False
        unavailable_auxiliary_ids = {
            auxiliary_id
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
            if auxiliary_id in self._failed_agent_ids
            or auxiliary_id in self._repair_exhausted_agent_ids
            or auxiliary_id in self._unresolved_dirty_agents
            or not self._has_successful_artifact(auxiliary_id)
        }
        if not unavailable_auxiliary_ids:
            return False
        candidate = self._graph.fork()
        candidate.set_relation(
            str(item["source_id"]),
            str(item["target_id"]),
            bool(item["source_to_target"]),
            bool(item["target_to_source"]),
        )
        return any(
            reasoner_id
            not in self._directed_successors(self._graph, auxiliary_id)
            and reasoner_id
            in self._directed_successors(candidate, auxiliary_id)
            for auxiliary_id in unavailable_auxiliary_ids
            for reasoner_id in exhausted_reasoner_ids
        )

    def _model_admissible_output_agent_ids(self) -> Tuple[str, ...]:
        """Return Output targets accepted by the same prospective Canvas gate."""

        active_lineage = set(self._active_semantic_lineage_ids())
        if self._graph.output_agent_id in active_lineage:
            return ()
        admitted: list[str] = []
        for node in self._graph.nodes:
            if node.id == self._graph.output_agent_id:
                continue
            if (
                self._uses_semantic_lineage_protocol()
                and not self._uses_role_conditional_capabilities()
                and (node.role_family or "").casefold() != "format"
            ):
                continue
            if (
                self._uses_role_conditional_capabilities()
                and (node.role_family or "").casefold()
                not in {"format", "output"}
            ):
                # ReAct and structured semantic completions are internal
                # artifacts. Only the generic Output capability and the
                # optional pure Formatter have a terminal-compatible wrapper.
                continue
            candidate = self._graph.fork()
            try:
                candidate.set_output(node.id)
            except GraphMutationError:
                continue
            validation = candidate.validate(
                self.model_registry,
                require_complete=False,
            )
            if not validation.valid:
                continue
            if self._output_sink_issue_for(candidate) is not None:
                continue
            if self._semantic_edit_issue_for(candidate) is not None:
                continue
            if (
                self._uses_format_agent_protocol(candidate)
                and self._format_agent_issue_for(candidate) is not None
            ):
                continue
            admitted.append(node.id)
        return tuple(admitted)

    def _output_sink_issue_for(self, graph: AgentGraph) -> Optional[str]:
        """Keep an assigned Output Agent in a terminal quotient-graph block."""

        if graph.output_agent_id is None:
            return None
        validation = graph.validate(
            self.model_registry,
            require_complete=True,
        )
        output_issue = next(
            (
                issue
                for issue in validation.issues
                if issue.code == "output_not_sink"
            ),
            None,
        )
        if output_issue is None:
            return None
        return (
            f"Output Agent {graph.output_agent_id!r} must remain in a "
            "quotient-graph sink block"
        )

    def _model_admissible_modify_agent_ids(self) -> Tuple[str, ...]:
        """Exclude an already verified semantic lineage from repair targets."""

        node_ids = tuple(node.id for node in self._graph.nodes)
        provider_repair_ids = self._provider_repair_agent_ids()
        if (
            provider_repair_ids
            and self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
        ):
            return provider_repair_ids
        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return node_ids
        mandatory_repair_ids = self._mandatory_repair_agent_ids()
        if mandatory_repair_ids:
            return mandatory_repair_ids
        dirty_replacement_ids = self._dirty_auxiliary_replacement_agent_ids()
        if dirty_replacement_ids:
            return dirty_replacement_ids
        measured_failed = (
            self._failed_agent_ids - self._repair_exhausted_agent_ids
        ).intersection(node_ids)
        if measured_failed:
            # AgentRuntime distinguishes the Agent that raised a typed failure
            # from blocked/pending descendants.  FlowSteer's next Canvas edit
            # should repair every measured root failure, not mutate downstream
            # Agents that merely could not execute because their input is absent.
            return tuple(
                node_id for node_id in node_ids if node_id in measured_failed
            )
        protected = set(self._active_semantic_lineage_ids())
        responsible = (
            set(self._unresolved_dirty_agents)
            - self._repair_exhausted_agent_ids
        )
        responsible.update(self._terminal_unreachable_agent_ids())
        return tuple(
            node_id
            for node_id in node_ids
            if node_id not in self._repair_exhausted_agent_ids
            and (node_id not in protected or node_id in responsible)
        )

    def _provider_repair_admission_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Enforce the same provider repair boundary as constrained decoding."""

        if (
            action.action_type is not AgentActionType.MODIFY_AGENT
            or action.agent_id is None
        ):
            return None
        admitted_model_ids = self._provider_repair_model_ids(action.agent_id)
        if not admitted_model_ids:
            return None
        mutable_fields = tuple(
            field_name
            for field_name in (
                "model_id",
                "contract",
                "role_family",
                "allowed_tools",
                "execution_mode",
                "artifact_type",
                "completion_condition",
            )
            if getattr(action, field_name) is not None
        )
        if mutable_fields != ("model_id",):
            return (
                "a provider failure repair must modify only model_id while "
                "preserving the Agent contract, role, tools, execution mode, "
                "artifact type, completion condition, and relations"
            )
        if action.model_id not in admitted_model_ids:
            failed_provider_id = self._provider_repair_avoid_provider_id(
                action.agent_id
            )
            return (
                "provider repair model_id must come from the live admitted "
                "domain; "
                f"avoid_provider_id={failed_provider_id!r}, "
                f"admitted_model_ids={list(admitted_model_ids)!r}"
            )
        return None

    def _react_repair_admission_issue(
        self,
        action: AgentAction,
    ) -> Optional[str]:
        """Keep authoritative ReAct repair equal to the live action domain."""

        if (
            action.action_type is not AgentActionType.MODIFY_AGENT
            or action.agent_id is None
            or action.agent_id not in self._mandatory_repair_agent_ids()
            or action.agent_id not in self._react_exhausted_agent_ids
            or self._provider_repair_model_ids(action.agent_id)
        ):
            return None
        mutable_fields = tuple(
            field_name
            for field_name in (
                "model_id",
                "contract",
                "role_family",
                "allowed_tools",
                "execution_mode",
                "artifact_type",
                "completion_condition",
            )
            if getattr(action, field_name) is not None
        )
        execution_profile_domains = (
            self._repair_exhausted_auxiliary_profile_domains()
        )
        admitted_profiles = execution_profile_domains.get(action.agent_id, ())
        if admitted_profiles:
            sampled_profile = (
                action.execution_mode,
                tuple(action.allowed_tools or ()),
            )
            if (
                mutable_fields == ("allowed_tools", "execution_mode")
                and action.allowed_tools is not None
                and sampled_profile in admitted_profiles
            ):
                return None
            return (
                "a measured execution-profile repair must atomically modify "
                "execution_mode and allowed_tools to one profile that already "
                "materialized the same role/artifact responsibility; "
                f"admitted_execution_profiles={list(admitted_profiles)!r}"
            )
        if (
            len(mutable_fields) != 1
            or mutable_fields[0] not in {"contract", "completion_condition"}
        ):
            return (
                "a measured ReAct repair must modify exactly one of contract "
                "or completion_condition while preserving the Agent model, "
                "role, tools, execution mode, artifact type, and relations"
            )
        return None

    def _semantic_artifact_repair_agent_ids(self) -> Tuple[str, ...]:
        """Return the existing Agent responsible for a terminal semantic fault.

        This uses only the current Canvas and progressive Runtime artifacts.  It
        never reads Ground Truth or evaluator state.  Structural and format
        lineage faults remain relation/output repairs; only a complete executed
        terminal lineage with an invalid semantic artifact enters this domain.
        """

        if (
            not self._uses_semantic_lineage_protocol()
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
        if self._uses_role_conditional_capabilities() and not any(
            marker in semantic_issue
            for marker in (
                "answer_slot",
                "candidate_answer",
                "evidence_propositions[",
                "question_scope",
                "entity_identity",
            )
        ):
            # Missing ingress, terminal reachability, and generic malformed
            # artifacts retain FlowSteer's relation/augmentation search space.
            # Only a measured semantic cross-field inconsistency becomes a
            # repair-first Agent target under the open role-conditional policy.
            return ()
        attribution = self._semantic_repair_attribution(semantic_issue)
        if attribution is None:
            return ()
        if self._uses_role_conditional_capabilities():
            format_serialization_repair = (
                attribution.get("responsible_constraint")
                == "format_serialization"
            )
            raw_responsible_ids = attribution.get("responsible_agent_ids", ())
            responsible_ids = tuple(
                dict.fromkeys(
                    agent_id
                    for agent_id in raw_responsible_ids
                    if isinstance(agent_id, str)
                    and self._graph.has_node(agent_id)
                    and (
                        agent_id != self._graph.output_agent_id
                        or format_serialization_repair
                    )
                )
            )
            if responsible_ids:
                return responsible_ids
            agent_id = attribution.get("responsible_agent_id")
            if (
                isinstance(agent_id, str)
                and self._graph.has_node(agent_id)
                and (
                    agent_id != self._graph.output_agent_id
                    or format_serialization_repair
                )
            ):
                return (agent_id,)
            return ()
        if (
            self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
            and attribution.get("responsible_constraint")
            not in {
                "reasoner_semantic_artifact",
                "verifier_semantic_artifact",
                "format_lineage",
                "candidate_consistency",
            }
        ):
            # Relation construction, missing semantic responsibilities,
            # Output assignment, and terminal reachability are graph faults.
            # Projecting any of them as a mandatory Agent modification traps
            # the progressive Canvas on a healthy/no-op node.
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
        profile_repair_ids = set(
            self._repair_exhausted_auxiliary_profile_domains()
        )
        if profile_repair_ids:
            # An alternate registered execution profile has already completed
            # the same role/artifact responsibility.  Repair the existing node
            # in place before adding another replacement, preserving its
            # relations and public continuation.
            return tuple(
                node_id for node_id in node_ids if node_id in profile_repair_ids
            )
        same_profile_replacements = (
            self._repair_exhausted_auxiliary_same_profile_replacements()
        )
        if (
            same_profile_replacements
            and not self._repair_exhausted_auxiliary_takeover_relation_candidates()
        ):
            # Once every existing downstream responsibility has received the
            # measured same-profile replacement artifact, return to the failed
            # node's bounded contract repair.  Deletion remains unavailable
            # unless Runtime explicitly diagnosed the node as unusable.
            return tuple(
                node_id
                for node_id in node_ids
                if node_id in same_profile_replacements
                and node_id not in self._diagnosed_unusable_agent_ids
            )
        selected_output_recovery = (
            self._selected_output_artifact_recovery_sources_by_target()
        )
        selected_output_recovery_relations = (
            self._selected_output_artifact_recovery_relation_candidates()
            if selected_output_recovery
            else []
        )
        if selected_output_recovery and (
            AgentActionType.SET_RELATION.value
            not in self._allowed_action_type_set
            or not selected_output_recovery_relations
        ):
            # When the exact artifact handoff is already present (or no legal
            # one-edge handoff remains), keep repair on the same failed
            # ancestor.  ReAct exhaustion does not authorize deletion or a
            # broad unrelated Canvas mutation.
            return tuple(
                node_id
                for node_id in node_ids
                if node_id in selected_output_recovery
            )
        if (
            self.max_agents is not None
            and len(self._graph.nodes) >= self.max_agents
        ):
            replacement_domains = (
                self._repair_exhausted_auxiliary_replacement_domains()
            )
            existing_recovery_is_live = bool(
                self._dirty_auxiliary_replacement_agent_ids()
                or self._failed_auxiliary_ingress_relation_candidates()
                or self._repair_exhausted_relation_candidates()
            )
            capacity_blocked_repairs = (
                ()
                if existing_recovery_is_live
                else tuple(
                    node.id
                    for node in self._graph.nodes
                    if node.id in self._failed_agent_ids
                    and node.id in self._repair_exhausted_agent_ids
                    and node.id not in self._diagnosed_unusable_agent_ids
                    and (node.role_family or "").casefold()
                    in replacement_domains
                    and node.artifact_type.casefold()
                    in replacement_domains[
                        (node.role_family or "").casefold()
                    ]
                )
            )
            if capacity_blocked_repairs:
                # FlowSteer's progressive Canvas cannot augment beyond its
                # configured Agent bound. Preserve the failed node and reopen
                # its existing execution contract instead of deleting it or
                # exposing an impossible ADD_SUBGRAPH action.
                return capacity_blocked_repairs
        repairable_failed = (
            self._failed_agent_ids - self._diagnosed_unusable_agent_ids
            - self._repair_exhausted_agent_ids
        ).intersection(node_ids)
        if repairable_failed:
            return tuple(
                node_id for node_id in node_ids if node_id in repairable_failed
            )
        return self._semantic_artifact_repair_agent_ids()

    def _repair_exhausted_auxiliary_takeover_delete_ids(self) -> Tuple[str, ...]:
        """Return bounded failed auxiliaries whose replacement already took over."""

        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return ()
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold()
            in {"evidence_retriever", "repair"}
            and node.id in self._failed_agent_ids
            and node.id in self._repair_exhausted_agent_ids
            and node.id in self._diagnosed_unusable_agent_ids
            and (
                record := self._latest_failure_record_by_agent.get(node.id)
            )
            is not None
            and self._execution_failure_diagnosis(record)[0]
            == "react_turn_exhaustion"
            and self._delete_admission_issue(node.id) is None
        )

    def _admissible_augmentation_role_families(self) -> Tuple[str, ...]:
        """Return semantic QA roles that may be added at this recovery boundary."""

        role_families = (
            "reasoner",
            "verifier",
            "format",
            "evidence_retriever",
            "repair",
        )
        if not self._uses_semantic_lineage_protocol():
            return role_families
        if self._uses_role_conditional_capabilities():
            # These are open search-space choices, not a minimum role set.
            # Multiple semantic workers and a generic Output Agent remain
            # available; only the specialized Formatter remains unique.
            return tuple(
                role_family
                for role_family in (*role_families, "output")
                if role_family != "format"
                or not self._semantic_role_agent_ids("format")
            )
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            # Multiple Reasoners and Verifiers are legal in the v2 search
            # space.  Candidate agreement is checked over the executed lineage
            # at FINISH instead of being encoded as a role-count template.
            return tuple(
                role_family
                for role_family in role_families
                if role_family != "format"
                or not self._semantic_role_agent_ids("format")
            )
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

    def _missing_semantic_role_families(self) -> Tuple[str, ...]:
        """Return semantic responsibilities not owned by a usable Agent."""

        if not self._uses_semantic_lineage_protocol():
            return ()
        if self._uses_role_conditional_capabilities():
            return ()
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            return tuple(
                role_family
                for role_family in ("reasoner", "verifier", "format")
                if not self._semantic_role_agent_ids(role_family)
            )
        return tuple(
            role_family
            for role_family in ("reasoner", "verifier", "format")
            if not self._semantic_role_agent_ids(role_family)
        )

    def _model_admissible_add_role_families(self) -> Tuple[str, ...]:
        """Project the exact live ADD role domain used by Canvas admission."""

        admitted = self._admissible_augmentation_role_families()
        missing = self._missing_semantic_role_families()
        if self._role_conditional_evidence_ingress_consumer_ids():
            return tuple(
                role for role in admitted if role == "evidence_retriever"
            )
        if self._pending_role_conditional_ingress_consumer_ids():
            # A new upstream semantic producer must be schedulable before the
            # deferred consumer. Raw retrieval and terminal wrappers do not
            # satisfy the Runtime semantic-artifact contract.
            return tuple(
                role for role in admitted if role in {"reasoner", "repair"}
            )
        if (
            self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
            and missing
        ):
            return tuple(role for role in admitted if role in missing)
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        if replacement_domains:
            return tuple(
                role for role in admitted if role in replacement_domains
            )
        if (
            self._repair_exhausted_reasoner_ids()
            and self._graph.nodes
            and missing
        ):
            return tuple(role for role in admitted if role in missing)
        if self._repair_exhausted_reasoner_ids():
            return tuple(
                role
                for role in admitted
                if role in {"evidence_retriever", "repair"}
            )
        return admitted

    def _repair_exhausted_auxiliary_replacement_domains(
        self,
    ) -> dict[str, Tuple[str, ...]]:
        """Return same-role/artifact domains still awaiting valid takeover."""

        profile_repair_ids = set(
            self._repair_exhausted_auxiliary_profile_domains()
        )
        same_profile_replacement_ids = set(
            self._repair_exhausted_auxiliary_same_profile_replacements()
        )
        domains: dict[str, list[str]] = {}
        for node in self._graph.nodes:
            role_family = (node.role_family or "").casefold()
            if (
                role_family not in {"evidence_retriever", "repair"}
                or node.id not in self._failed_agent_ids
                or node.id not in self._repair_exhausted_agent_ids
                or node.id in profile_repair_ids
                or node.id in same_profile_replacement_ids
                or self._delete_admission_issue(node.id) is None
            ):
                continue
            artifact_types = domains.setdefault(role_family, [])
            artifact_type = node.artifact_type.casefold()
            if artifact_type not in artifact_types:
                artifact_types.append(artifact_type)
        return {
            role_family: tuple(artifact_types)
            for role_family, artifact_types in domains.items()
        }

    def _repair_exhausted_auxiliary_replacement_ingress_consumer_ids(
        self,
    ) -> Tuple[str, ...]:
        """Return existing downstream duties for one atomic replacement ADD."""

        if (
            not self._uses_role_conditional_capabilities()
            or self._graph.output_agent_id is None
        ):
            return ()
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        source_ids = tuple(
            node.id
            for node in self._graph.nodes
            if node.id in self._failed_agent_ids
            and node.id in self._repair_exhausted_agent_ids
            and (node.role_family or "").casefold() in replacement_domains
            and node.artifact_type.casefold()
            in replacement_domains[(node.role_family or "").casefold()]
        )
        if len(source_ids) != 1:
            return ()
        return self._directed_successors(self._graph, source_ids[0])

    def _repair_exhausted_auxiliary_profile_domains(
        self,
    ) -> dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]]:
        """Return measured same-responsibility execution-profile repairs.

        SkillFlow registers execution as a correlated ``execution_mode`` and
        Tool-set profile.  Once an isolated same-role/same-artifact replacement
        has actually materialized a valid artifact under a different registered
        profile, FlowSteer's next edit repairs the existing failed node with
        that exact profile instead of adding another duplicate.  The failed
        node, its relations, public continuation, and semantic lineage remain
        in place; no answer or evaluator state participates in this domain.
        """

        if (
            not self._uses_role_conditional_capabilities()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
        ):
            return {}
        result: dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {}
        for source in self._graph.nodes:
            source_role = (source.role_family or "").casefold()
            if (
                source_role not in {"evidence_retriever", "repair"}
                or source.id not in self._failed_agent_ids
                or source.id not in self._repair_exhausted_agent_ids
                or source.id in self._diagnosed_unusable_agent_ids
            ):
                continue
            current_profile = (
                source.execution_mode.value,
                tuple(source.allowed_tools),
            )
            registered_profiles = set(
                self._role_conditional_execution_profiles_for(source_role)
            )
            profiles: list[Tuple[str, Tuple[str, ...]]] = []
            for replacement in self._graph.nodes:
                replacement_profile = (
                    replacement.execution_mode.value,
                    tuple(replacement.allowed_tools),
                )
                if (
                    replacement.id == source.id
                    or (replacement.role_family or "").casefold()
                    != source_role
                    or replacement.artifact_type.casefold()
                    != source.artifact_type.casefold()
                    or replacement.id in self._failed_agent_ids
                    or replacement.id in self._repair_exhausted_agent_ids
                    or replacement.id in self._unresolved_dirty_agents
                    or not self._has_successful_artifact(replacement.id)
                    or not self._semantic_replacement_has_valid_artifact(
                        replacement.id,
                        source_role,
                    )
                    or replacement_profile == current_profile
                    or replacement_profile not in registered_profiles
                ):
                    continue
                if replacement_profile not in profiles:
                    profiles.append(replacement_profile)
            if profiles:
                result[source.id] = tuple(profiles)
        return result

    def _dirty_auxiliary_replacement_agent_ids(self) -> Tuple[str, ...]:
        """Return max-capacity Retriever replacements awaiting an artifact."""

        if self.max_agents is None or len(self._graph.nodes) < self.max_agents:
            return ()
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        evidence_artifact_types = set(
            replacement_domains.get("evidence_retriever", ())
        )
        if not evidence_artifact_types:
            return ()
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold() == "evidence_retriever"
            and node.artifact_type.casefold() in evidence_artifact_types
            and node.id in self._unresolved_dirty_agents
            and node.id not in self._repair_exhausted_agent_ids
        )

    def _provider_repair_catalog_domain(
        self,
        current_model_id: str,
    ) -> Tuple[str, ...]:
        """Return cross-provider arms, or same-provider fallback if necessary."""

        current_provider_id = self.model_registry.provider_for(
            current_model_id
        ).provider_id
        alternatives = tuple(
            model_id
            for model_id in self.model_registry.model_ids
            if model_id != current_model_id
        )
        cross_provider = tuple(
            model_id
            for model_id in alternatives
            if self.model_registry.provider_for(model_id).provider_id
            != current_provider_id
        )
        return cross_provider or alternatives

    def _provider_repair_model_ids(self, agent_id: str) -> Tuple[str, ...]:
        """Return the typed provider-failure model repair domain.

        SkillFlow keeps provider identity separate from model identity.  A
        measured provider failure therefore changes only the model field and,
        when the catalog offers one, prefers a different provider.  Falling
        back to another exact model on the same provider keeps recovery live
        for single-provider catalogs.
        """

        record = self._latest_failure_record_by_agent.get(agent_id)
        if record is None or not self._graph.has_node(agent_id):
            return ()
        category, retryability, _ = self._execution_failure_diagnosis(record)
        if (
            category != "provider_request_failure"
            or retryability
            not in {"transient_provider", "permanent_configuration"}
        ):
            return ()
        current_model_id = self._graph.get_node(agent_id).model_id
        return self._provider_repair_catalog_domain(current_model_id)

    def _provider_repair_agent_ids(self) -> Tuple[str, ...]:
        """Return measured provider failures with a live model replacement."""

        return tuple(
            node.id
            for node in self._graph.nodes
            if self._provider_repair_model_ids(node.id)
        )

    def _provider_repair_avoid_provider_id(
        self,
        agent_id: str,
    ) -> Optional[str]:
        """Return the failed provider only when a cross-provider arm exists."""

        if not self._graph.has_node(agent_id):
            return None
        current_model_id = self._graph.get_node(agent_id).model_id
        current_provider_id = self.model_registry.provider_for(
            current_model_id
        ).provider_id
        admitted_model_ids = self._provider_repair_model_ids(agent_id)
        if any(
            self.model_registry.provider_for(model_id).provider_id
            != current_provider_id
            for model_id in admitted_model_ids
        ):
            return current_provider_id
        return None

    def model_admissible_action_targets(self) -> dict[str, object]:
        """Project exact current Canvas target domains for constrained sampling.

        This is a read-only FlowSteer legality projection.  It does not select
        the next action, repair an invalid sample, or prescribe a topology.
        """

        admitted = set(self.model_admissible_action_types())
        node_ids = [node.id for node in self._graph.nodes]
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        replacement_ingress_consumer_ids = (
            self._repair_exhausted_auxiliary_replacement_ingress_consumer_ids()
        )
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
            if replacement_domains:
                # The authoritative recovery admission accepts exactly one
                # same-role/same-artifact executable prefix.
                remaining = min(remaining, 1)
            evidence_ingress_consumer_ids = (
                self._role_conditional_evidence_ingress_consumer_ids()
            )
            if evidence_ingress_consumer_ids:
                remaining = min(remaining, 1)
            missing_role_families = self._missing_semantic_role_families()
            admitted_new_role_families = (
                self._model_admissible_add_role_families()
            )
            pending_ingress_consumer_ids = (
                self._pending_role_conditional_ingress_consumer_ids()
            )
            required_ingress_consumer_ids = tuple(
                dict.fromkeys(
                    (
                        *pending_ingress_consumer_ids,
                        *evidence_ingress_consumer_ids,
                        *replacement_ingress_consumer_ids,
                    )
                )
            )
            current_output_agent_id = self._graph.output_agent_id
            current_execution = self._cached_progressive_execution()
            atomic_output_ingress = bool(
                self._uses_role_conditional_capabilities()
                and current_output_agent_id is not None
                and self._graph.has_node(current_output_agent_id)
                and (
                    self._graph.get_node(current_output_agent_id).role_family
                    or ""
                ).casefold()
                == "output"
                and not replacement_domains
                and not required_ingress_consumer_ids
                and current_execution is not None
                and self._semantic_protocol_issue(current_execution) is not None
            )
            if atomic_output_ingress:
                # FlowSteer's ADD_SUBGRAPH is one atomic Canvas edit followed
                # by one execution.  When Output already exists, a one-Agent
                # augmentation must route its artifact into that existing sink
                # in the same edit; an orphan prefix would be rejected by the
                # unchanged all-Agents-reach-Output invariant.
                remaining = min(remaining, 1)
                admitted_new_role_families = tuple(
                    role_family
                    for role_family in admitted_new_role_families
                    if role_family not in {"format", "verifier"}
                )
            if (
                self.semantic_protocol
                == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
                and missing_role_families
            ):
                # Missing semantic responsibilities are sampled without
                # duplicates at this construction boundary. Multiple
                # Reasoners/Verifiers remain available after the minimum
                # capabilities exist, so this does not prescribe topology.
                remaining = min(remaining, len(missing_role_families))
            if (
                self._repair_exhausted_reasoner_ids()
                and self._graph.nodes
                and missing_role_families
            ):
                remaining = min(remaining, len(missing_role_families))
            elif self._repair_exhausted_reasoner_ids():
                # Recovery augmentation is one executable unit.  Keep the live
                # schema on the same one-Agent boundary enforced by admission.
                remaining = min(remaining, 1)
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
                        **(
                            {
                                "output_role_families": list(
                                    ("format", "output")
                                )
                            }
                            if self._uses_role_conditional_capabilities()
                            else {"output_role_family": "format"}
                        ),
                        "required_agent_fields": [
                            "agent_id",
                            "model_id",
                            "contract",
                            "role_family",
                            "allowed_tools",
                            "execution_mode",
                            *(
                                ["artifact_type"]
                                if replacement_domains
                                else []
                            ),
                        ],
                        "model_ids": list(self.model_registry.model_ids),
                        **(
                            {
                                "registered_execution_profiles": [
                                    {
                                        "execution_mode": execution_mode,
                                        "allowed_tools": list(allowed_tools),
                                    }
                                    for execution_mode, allowed_tools in (
                                        self._role_conditional_registered_execution_profiles()
                                    )
                                ]
                            }
                            if self._uses_role_conditional_capabilities()
                            else {}
                        ),
                        "role_constraints": {
                            "reasoner": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "reasoner"
                                    )
                                    if self._uses_role_conditional_capabilities()
                                    else {
                                        "execution_modes": ["react"],
                                        "allowed_tools": [
                                            [self.required_evidence_tool_id]
                                        ],
                                    }
                                ),
                            },
                            "verifier": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "verifier"
                                    )
                                    if self._uses_role_conditional_capabilities()
                                    else {
                                        "execution_modes": ["reasoning"],
                                        "allowed_tools": [[]],
                                    }
                                ),
                            },
                            "format": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "format"
                                    )
                                    if self._uses_role_conditional_capabilities()
                                    else {
                                        "execution_modes": ["reasoning"],
                                        "allowed_tools": [[]],
                                    }
                                ),
                                "contracts": [
                                    _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT
                                    if self._uses_role_conditional_capabilities()
                                    else _HOTPOTQA_FORMAT_CONTRACT
                                ],
                                **(
                                    {"must_be_output_agent": True}
                                    if self._uses_role_conditional_capabilities()
                                    else {}
                                ),
                            },
                            "evidence_retriever": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "evidence_retriever"
                                    )
                                    if self._uses_role_conditional_capabilities()
                                    else {
                                        "execution_modes": ["react"],
                                        "allowed_tools": [
                                            [self.required_evidence_tool_id]
                                        ],
                                    }
                                ),
                                **(
                                    {
                                        "artifact_types": list(
                                            replacement_domains[
                                                "evidence_retriever"
                                            ]
                                        )
                                    }
                                    if "evidence_retriever"
                                    in replacement_domains
                                    else {}
                                ),
                            },
                            "repair": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "repair"
                                    )
                                    if self._uses_role_conditional_capabilities()
                                    else {
                                        "execution_modes": ["reasoning"],
                                        "allowed_tools": [[]],
                                    }
                                ),
                                **(
                                    {
                                        "artifact_types": list(
                                            replacement_domains["repair"]
                                        )
                                    }
                                    if "repair" in replacement_domains
                                    else {}
                                ),
                            },
                            "output": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "output"
                                    )
                                    if self._uses_role_conditional_capabilities()
                                    else {
                                        "execution_modes": ["reasoning"],
                                        "allowed_tools": [[]],
                                    }
                                ),
                            },
                        },
                        "admitted_new_role_families": list(
                            admitted_new_role_families
                        ),
                        **(
                            {
                        "required_ingress_consumer_agent_ids": list(
                            required_ingress_consumer_ids
                        ),
                        **(
                            {"exact_relation_count": 1}
                            if (
                                evidence_ingress_consumer_ids
                                or replacement_ingress_consumer_ids
                            )
                            else {}
                        ),
                        **(
                            {
                                "required_reachability_output_agent_id": (
                                    current_output_agent_id
                                ),
                                "exact_relation_count": 1,
                            }
                            if atomic_output_ingress
                            else {}
                        ),
                        "explicit_output_assignment_required": False,
                            }
                            if self._uses_role_conditional_capabilities()
                            else {}
                        ),
                        "distinct_new_role_families": bool(
                            self.semantic_protocol
                            == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
                            and missing_role_families
                        ),
                        "defer_output_assignment": bool(
                            self.semantic_protocol
                            == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
                            and missing_role_families
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
                        **(
                            {
                                "relations": [],
                            }
                            if (
                                replacement_domains
                                and not replacement_ingress_consumer_ids
                            )
                            else {}
                        ),
                        **(
                            {"output_agent_id": None}
                            if (
                                replacement_domains
                                and not replacement_ingress_consumer_ids
                            )
                            else {}
                        ),
                    }
                    if self._uses_semantic_lineage_protocol()
                    else {}
                ),
            }
        if AgentActionType.MODIFY_AGENT.value in admitted:
            modifiable_node_ids = list(self._model_admissible_modify_agent_ids())
            profile_repair_domains = (
                self._repair_exhausted_auxiliary_profile_domains()
            )
            base_mutable_fields = [
                "model_id",
                "contract",
                "artifact_type",
                "completion_condition",
            ]
            if not self._uses_semantic_lineage_protocol():
                base_mutable_fields[2:2] = [
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                ]
            measured_failed_ids = self._failed_agent_ids.intersection(node_ids)
            provider_failure_agent_ids = {
                agent_id
                for agent_id in modifiable_node_ids
                if self._provider_repair_model_ids(agent_id)
            }
            dirty_replacement_ids = set(
                self._dirty_auxiliary_replacement_agent_ids()
            )
            model_repair_domains = {
                agent_id: (
                    self._provider_repair_model_ids(agent_id)
                    if agent_id in provider_failure_agent_ids
                    else tuple(
                        model_id
                        for model_id in self.model_registry.model_ids
                        if model_id
                        != self._graph.get_node(agent_id).model_id
                    )
                )
                for agent_id in modifiable_node_ids
            }
            responsible_ids = set(measured_failed_ids)
            if not measured_failed_ids:
                responsible_ids.update(self._unresolved_dirty_agents)
                responsible_ids.update(self._terminal_unreachable_agent_ids())
            if (
                not measured_failed_ids
                and self._uses_semantic_lineage_protocol()
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
                        ["model_id"]
                        if agent_id in provider_failure_agent_ids
                        else ["execution_mode"]
                        if agent_id in profile_repair_domains
                        else ["contract", "completion_condition"]
                        if agent_id in dirty_replacement_ids
                        else ["contract", "completion_condition"]
                        if (
                            self._uses_semantic_lineage_protocol()
                            and agent_id in self._react_exhausted_agent_ids
                        )
                        else base_mutable_fields
                    )
                    if not (
                        self._uses_semantic_lineage_protocol()
                        and (
                            self._graph.get_node(agent_id).role_family or ""
                        ).casefold()
                        == "format"
                        and field == "contract"
                    )
                    and (
                        field != "model_id"
                        or bool(model_repair_domains[agent_id])
                    )
                ]
                for agent_id in modifiable_node_ids
            }
            mutable_fields = [
                field
                for field in (
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                    "artifact_type",
                    "completion_condition",
                )
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
                                    *model_repair_domains[agent_id]
                                ]
                            }
                            if "model_id" in per_agent_mutable_fields[agent_id]
                            else {}
                        ),
                        **(
                            {
                                "execution_profiles": [
                                    {
                                        "execution_mode": execution_mode,
                                        "allowed_tools": list(allowed_tools),
                                    }
                                    for execution_mode, allowed_tools in (
                                        profile_repair_domains[agent_id]
                                    )
                                ]
                            }
                            if agent_id in profile_repair_domains
                            else {}
                        ),
                        **(
                            {"avoid_provider_id": avoid_provider_id}
                            if (
                                agent_id in provider_failure_agent_ids
                                and (
                                    avoid_provider_id
                                    := self._provider_repair_avoid_provider_id(
                                        agent_id
                                    )
                                )
                                is not None
                            )
                            else {}
                        ),
                    }
                    for agent_id in modifiable_node_ids
                ],
                **(
                    {
                        "purpose": "selected_output_artifact_recovery",
                        "artifact_source_agent_ids_by_target": {
                            agent_id: list(source_ids)
                            for agent_id, source_ids in (
                                self._selected_output_artifact_recovery_sources_by_target().items()
                            )
                            if agent_id in modifiable_node_ids
                        },
                    }
                    if self._selected_output_artifact_recovery_sources_by_target()
                    else {}
                ),
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
            relation_candidates = self._model_admissible_relation_candidates()
            selected_output_recovery_candidates = (
                self._selected_output_artifact_recovery_relation_candidates()
            )
            prospective_convergence_candidates = (
                self._prospective_terminal_convergence_relation_candidates()
            )
            targets[AgentActionType.SET_RELATION.value] = {
                "source_agent_ids": node_ids,
                "target_agent_ids": node_ids,
                "endpoints_must_differ": True,
                "candidates": relation_candidates,
                **(
                    {
                        "purpose": "selected_output_artifact_recovery",
                        "selected_output_agent_id": self._graph.output_agent_id,
                    }
                    if selected_output_recovery_candidates
                    and relation_candidates
                    == selected_output_recovery_candidates
                    else {}
                ),
                **(
                    {
                        "purpose": "terminal_branch_convergence",
                        "prospective_output_agent_ids": sorted(
                            {
                                str(candidate["target_id"])
                                for candidate in prospective_convergence_candidates
                            }
                        ),
                    }
                    if prospective_convergence_candidates
                    and relation_candidates
                    == prospective_convergence_candidates
                    else {}
                ),
            }
        if AgentActionType.SET_OUTPUT.value in admitted:
            targets[AgentActionType.SET_OUTPUT.value] = {
                "agent_ids": list(self._model_admissible_output_agent_ids()),
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
        self._last_valid_evidence_lineage = None
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
        self._last_valid_evidence_lineage = None
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
        provider_repair_ids = self._provider_repair_agent_ids()
        if (
            provider_repair_ids
            and self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            and (
                action.action_type is not AgentActionType.MODIFY_AGENT
                or action.agent_id not in provider_repair_ids
            )
        ):
            return self._reject_after_count(
                action,
                "edit rejected: repair the measured provider failure before "
                "other Canvas edits; provider_repair_agent_ids="
                f"{list(provider_repair_ids)!r}",
            )
        provider_repair_issue = self._provider_repair_admission_issue(action)
        if provider_repair_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + provider_repair_issue,
            )
        terminal_reachability_issue = (
            self._terminal_reachability_admission_issue(action)
            if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            else None
        )
        if terminal_reachability_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + terminal_reachability_issue,
            )
        preservation_issue = self._preservation_admission_issue(action)
        if preservation_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + preservation_issue,
            )
        terminal_convergence_output_id = next(
            (
                str(candidate["target_id"])
                for candidate in (
                    self._prospective_terminal_convergence_relation_candidates()
                )
                if self._relation_action_matches_candidate(action, candidate)
            ),
            None,
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
                        require_exact_answer_tag=self.require_exact_answer_tag,
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
                            self._completed_partial_output_agent_ids(
                                exc.partial_result
                            )
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
        if action.action_type in {
            AgentActionType.ADD_SUBGRAPH,
            AgentActionType.SET_RELATION,
            AgentActionType.SET_OUTPUT,
        }:
            output_sink_issue = self._output_sink_issue_for(candidate)
            if output_sink_issue is not None:
                return self._reject_after_count(
                    action,
                    "edit rejected: " + output_sink_issue,
                )
        semantic_edit_issue = self._semantic_edit_issue_for(candidate)
        if semantic_edit_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + semantic_edit_issue,
            )
        preserved_input_issue = self._preserved_input_change_issue_for(
            candidate,
            terminal_convergence_output_id=terminal_convergence_output_id,
        )
        if preserved_input_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + preserved_input_issue,
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
            and self._uses_format_agent_protocol(candidate)
        ):
            format_issue = self._format_agent_issue_for(candidate)
            if format_issue is not None:
                return self._reject_after_count(
                    action,
                    "edit rejected: " + format_issue,
                )

        recovery_continuation_handoff = (
            self._recovery_continuation_handoff(action)
        )
        self._graph = candidate
        current_agent_ids = {node.id for node in self._graph.nodes}
        self._retain_current_failure_state(current_agent_ids)
        # One accepted edit is one FlowSteer execute-and-feedback boundary.
        # A repair baseline must never leak into a later unrelated edit.
        self._pending_repair_receipt_count_by_agent.clear()
        if (
            action.action_type is AgentActionType.MODIFY_AGENT
            and action.agent_id is not None
        ):
            # The typed failure belongs to the pre-edit Agent declaration.
            # Clear that diagnosis once its admitted repair is committed, but
            # retain SkillFlow's public Action--Observation continuation until
            # the repaired Agent executes so successful reads are not repeated.
            if action.agent_id in self._react_exhausted_agent_ids:
                continuation = self._failure_continuations.get(action.agent_id)
                if continuation is not None:
                    self._pending_repair_receipt_count_by_agent[
                        action.agent_id
                    ] = self._failure_continuation_weight(continuation)[0]
            self._failed_agent_ids.discard(action.agent_id)
            self._diagnosed_unusable_agent_ids.discard(action.agent_id)
            self._react_exhausted_agent_ids.discard(action.agent_id)
            self._repair_exhausted_agent_ids.discard(action.agent_id)
            self._latest_failure_record_by_agent.pop(action.agent_id, None)
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
                    prior_failure_metadata = dict(self._failure_continuations)
                    prior_failure_metadata.update(recovery_continuation_handoff)
                    execution = await self.runtime.execute(
                        self._graph,
                        self._problem,
                        require_complete=False,
                        prior_outputs=self._progressive_outputs,
                        prior_output_metadata=self._progressive_output_metadata,
                        prior_failure_metadata=prior_failure_metadata,
                        dirty_agents=self._unresolved_dirty_agents,
                        format_output_agent=self._uses_format_agent_protocol(),
                        require_exact_answer_tag=self.require_exact_answer_tag,
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
                        for agent_id, handoff in (
                            recovery_continuation_handoff.items()
                        ):
                            source_agent_id = handoff.get(
                                "continuation_source_agent_id"
                            )
                            if (
                                agent_id in partial_execution.outputs
                                and isinstance(source_agent_id, str)
                            ):
                                self._progressive_output_metadata.setdefault(
                                    agent_id,
                                    {},
                                ).setdefault(
                                    "continuation_source_agent_id",
                                    source_agent_id,
                                )
                        completed_partial_ids = (
                            self._completed_partial_output_agent_ids(
                                partial_execution
                            )
                            if self.recovery_policy
                            == _PRESERVE_REPAIR_RECOVERY_POLICY
                            else set(partial_execution.executed_agent_ids)
                        )
                        self._unresolved_dirty_agents.difference_update(
                            completed_partial_ids
                        )
                        if (
                            self.recovery_policy
                            == _PRESERVE_REPAIR_RECOVERY_POLICY
                        ):
                            self._unresolved_dirty_agents.update(
                                set(partial_execution.outputs)
                                - completed_partial_ids
                            )
                    self._unresolved_dirty_agents.update(
                        agent_id
                        for agent_id in exc.pending_agent_ids
                        if agent_id in current_agent_ids
                    )
                    self._failed_agent_ids.difference_update(
                        ()
                        if partial_execution is None
                        else self._completed_partial_output_agent_ids(
                            partial_execution
                        )
                    )
                    self._diagnosed_unusable_agent_ids.difference_update(
                        ()
                        if partial_execution is None
                        else self._completed_partial_output_agent_ids(
                            partial_execution
                        )
                    )
                    self._mark_agents_recovered(
                        ()
                        if partial_execution is None
                        else self._completed_partial_output_agent_ids(
                            partial_execution
                        )
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
                    for agent_id, handoff in (
                        recovery_continuation_handoff.items()
                    ):
                        source_agent_id = handoff.get(
                            "continuation_source_agent_id"
                        )
                        if (
                            agent_id in execution.outputs
                            and isinstance(source_agent_id, str)
                        ):
                            self._progressive_output_metadata.setdefault(
                                agent_id,
                                {},
                            ).setdefault(
                                "continuation_source_agent_id",
                                source_agent_id,
                            )
                    self._progressive_execution = execution
                    self._progressive_execution_revision = self._graph.revision
                    # A semantic QA Verifier/Formatter can be structurally present
                    # while its semantic input is not yet routable.  Runtime
                    # deferral is successful progressive execution, not Agent
                    # failure; keep only those unmaterialized nodes unresolved.
                    self._unresolved_dirty_agents = (
                        current_agent_ids - set(execution.outputs)
                    )
                    if (
                        self.recovery_policy
                        == _PRESERVE_REPAIR_RECOVERY_POLICY
                    ):
                        # SkillFlow completion is per Agent Action--Observation
                        # boundary.  A successful auxiliary block does not imply
                        # that a deferred failed Reasoner recovered: clear only
                        # Agents that actually materialized an artifact and keep
                        # the remaining continuation/Tool receipts revision-live.
                        self._mark_agents_recovered(execution.outputs)
                        self._retain_current_failure_state(current_agent_ids)
                        if not self._unresolved_dirty_agents:
                            self._clear_failure_state()
                    else:
                        self._clear_failure_state()
                    self._capture_last_valid_evidence_lineage(execution)
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
                and self._uses_semantic_lineage_protocol()
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

        if not self._uses_semantic_lineage_protocol():
            return None
        if self._uses_role_conditional_capabilities():
            return self._hotpotqa_role_conditional_repair_attribution(
                reason,
                stage=stage,
                validation=validation,
            )
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            return self._hotpotqa_semantic_repair_attribution(
                reason,
                stage=stage,
                validation=validation,
            )
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
        graph_agent_ids = {node.id for node in self._graph.nodes}
        failed_ids = tuple(
            sorted(self._failed_agent_ids.intersection(graph_agent_ids))
        )
        responsible_agent_ids: tuple[str, ...] = unreachable_ids
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
            required_relation_candidates = (
                self._required_semantic_relation_candidates()
            )
            output_target_ids = self._model_admissible_output_agent_ids()
            if required_relation_candidates:
                role_family = "format"
                responsible_constraint = "semantic_lineage_relation"
                preferred_actions = ["set_relation", "set_output", "add_subgraph"]
            elif output_target_ids:
                role_family = "format"
                responsible_constraint = "format_output_assignment"
                preferred_actions = ["set_output", "set_relation", "modify_agent"]
            else:
                role_family = None
                responsible_constraint = "semantic_lineage_construction"
                preferred_actions = ["add_subgraph", "set_relation", "set_output"]
        elif stage == "execution" and failed_ids:
            target_id = failed_ids[0]
            role_family = (
                self._graph.get_node(target_id).role_family
                if self._graph.has_node(target_id)
                else None
            )
            responsible_agent_ids = failed_ids
            responsible_constraint = "execution_contract_or_runtime_failure"
            preferred_actions = ["modify_agent", "add_subgraph", "set_relation"]
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
        coverage_failure = (
            "knowledge_base_coverage_failure" in reason.casefold()
        )
        if coverage_failure:
            target_id, role_family = reasoner_id, "reasoner"
            responsible_constraint = "retrieval_or_database_coverage"
            preferred_actions = ["modify_agent", "add_subgraph", "set_relation"]
        elif reason.startswith("Reasoner"):
            target_id, role_family = reasoner_id, "reasoner"
            responsible_constraint = "reasoner_semantic_artifact"
            preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        elif reason.startswith("Verifier") and any(
            f"{field_name!r} must be true" in reason
            for field_name in (
                "evidence_supported",
                "entity_attribute_binding_correct",
                "alias_binding_correct",
                "answer_type_cardinality_correct",
                "multi_hop_complete",
                "minimal_answer_surface",
                "scope_preserved",
            )
        ):
            # These Verifier booleans are verdicts over the upstream semantic
            # artifact.  The Reasoner owns evidence, relation/alias binding,
            # answer-slot type/cardinality, scope and the candidate surface;
            # repeatedly editing the Verifier cannot repair those fields.
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
            "responsible_agent_ids": list(responsible_agent_ids),
            "format_target_agent_ids": list(formatter_ids),
            "preserve_agent_ids": preserved,
            "preferred_action_order": preferred_actions,
            "delete_allowed_before_replacement_takeover": False,
        }
        if coverage_failure:
            result["operational_diagnosis"] = "knowledge_base_coverage_failure"
            result["corpus_level_oracle_claim"] = False
        if target_id is not None:
            result["responsible_agent_id"] = target_id
        return result

    def _hotpotqa_role_conditional_repair_attribution(
        self,
        reason: str,
        *,
        stage: Optional[str],
        validation: Optional[GraphValidationResult],
    ) -> dict[str, object]:
        """Attribute a measured fault without inventing a required role spine."""

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
        graph_ids = {node.id for node in self._graph.nodes}
        output_id = self._graph.output_agent_id
        routed_ids = (
            ()
            if output_id is None or not self._graph.has_node(output_id)
            else (
                *self._directed_ancestor_ids(self._graph, output_id),
                output_id,
            )
        )
        routed_reasoner_ids = tuple(
            agent_id
            for agent_id in routed_ids
            if (self._graph.get_node(agent_id).role_family or "").casefold()
            == "reasoner"
        )
        failed_ids = tuple(sorted(self._failed_agent_ids & graph_ids))
        target_id: Optional[str] = None
        role_family: Optional[str] = None
        constraint = "output_artifact"
        preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        responsible_ids: Tuple[str, ...] = ()
        if unreachable_ids:
            target_id = unreachable_ids[0]
            role_family = self._graph.get_node(target_id).role_family
            responsible_ids = unreachable_ids
            constraint = "terminal_reachability"
            preferred_actions = ["set_relation", "modify_agent", "add_subgraph"]
        elif stage == "execution" and failed_ids:
            target_id = failed_ids[0]
            role_family = self._graph.get_node(target_id).role_family
            responsible_ids = failed_ids
            constraint = "execution_contract_or_runtime_failure"
        elif (
            self._graph.output_agent_id is None
            or "output_agent_count" in issue_codes
            or "unknown_output_agent" in issue_codes
        ):
            constraint = "output_assignment"
            preferred_actions = ["set_output", "add_subgraph", "set_relation"]
        else:
            quoted = re.search(r"(?:Reasoner|Verifier) '([^']+)'", reason)
            quoted_id = None if quoted is None else quoted.group(1)
            if quoted_id in graph_ids:
                target_id = quoted_id
                role_family = self._graph.get_node(target_id).role_family
            if reason.startswith("Reasoner"):
                role_family = "reasoner"
                constraint = "reasoner_semantic_artifact"
                if target_id is None and routed_reasoner_ids:
                    target_id = routed_reasoner_ids[0]
                diagnosed_reasoner_ids = (
                    (target_id,)
                    if target_id is not None
                    and (
                        self._graph.get_node(target_id).role_family or ""
                    ).casefold()
                    == "reasoner"
                    else routed_reasoner_ids
                )
                responsible_ids = diagnosed_reasoner_ids
            elif reason.startswith("Verifier"):
                verifier_check_failure = any(
                    f"{field_name!r} must be true" in reason
                    for field_name in (
                        "evidence_supported",
                        "entity_attribute_binding_correct",
                        "alias_binding_correct",
                        "answer_type_cardinality_correct",
                        "multi_hop_complete",
                        "minimal_answer_surface",
                        "scope_preserved",
                    )
                )
                semantic_producer_ids: Tuple[str, ...] = ()
                if (
                    self._uses_role_conditional_capabilities()
                    and verifier_check_failure
                    and target_id is not None
                    and (
                        self._graph.get_node(target_id).role_family or ""
                    ).casefold()
                    == "verifier"
                ):
                    verifier_candidate, _ = self._semantic_candidate_from_artifact(
                        self._progressive_outputs.get(target_id, "")
                    )
                    routed_producers: list[tuple[int, str]] = []
                    for ancestor_id in self._directed_ancestor_ids(
                        self._graph,
                        target_id,
                    ):
                        ancestor_role = (
                            self._graph.get_node(ancestor_id).role_family or ""
                        ).casefold()
                        if ancestor_role in {
                            "evidence_retriever",
                            "verifier",
                            "format",
                            "output",
                        }:
                            continue
                        producer_candidate, producer_issue = (
                            self._semantic_candidate_from_artifact(
                                self._progressive_outputs.get(ancestor_id, "")
                            )
                        )
                        path = self._directed_shortest_path(
                            self._graph,
                            ancestor_id,
                            target_id,
                        )
                        if (
                            producer_issue is None
                            and producer_candidate is not None
                            and (
                                verifier_candidate is None
                                or producer_candidate == verifier_candidate
                            )
                            and path
                        ):
                            routed_producers.append((len(path), ancestor_id))
                    if routed_producers:
                        minimum_path_length = min(
                            length for length, _ in routed_producers
                        )
                        semantic_producer_ids = tuple(
                            agent_id
                            for length, agent_id in routed_producers
                            if length == minimum_path_length
                        )
                if (
                    self._uses_role_conditional_capabilities()
                    and verifier_check_failure
                    and semantic_producer_ids
                ):
                    target_id = semantic_producer_ids[0]
                    role_family = self._graph.get_node(target_id).role_family
                    responsible_ids = semantic_producer_ids
                    constraint = "semantic_candidate_artifact"
                else:
                    role_family = "verifier"
                    constraint = "verifier_semantic_artifact"
            elif reason.startswith(("Format", "Formatter")):
                target_id = self._graph.output_agent_id
                role_family = "format"
                constraint = "format_serialization"
            elif "qa-retrieval read receipt" in reason:
                evidence_agents = tuple(
                    node.id
                    for node in self._graph.nodes
                    if node.execution_mode.value == "react"
                    and self.required_evidence_tool_id in node.allowed_tools
                )
                target_id = evidence_agents[0] if evidence_agents else None
                role_family = (
                    None
                    if target_id is None
                    else self._graph.get_node(target_id).role_family
                )
                constraint = "evidence_retrieval"
                preferred_actions = ["modify_agent", "add_subgraph", "set_relation"]
                responsible_ids = tuple(
                    dict.fromkeys((*responsible_ids, *evidence_agents))
                )
        if target_id is not None and target_id not in responsible_ids:
            responsible_ids = (*responsible_ids, target_id)
        preserved = [
            node.id
            for node in self._graph.nodes
            if node.id != target_id and self._has_successful_artifact(node.id)
        ]
        result: dict[str, object] = {
            "responsible_constraint": constraint,
            "responsible_role_family": role_family,
            "responsible_agent_ids": list(responsible_ids),
            "preserve_agent_ids": preserved,
            "preferred_action_order": preferred_actions,
            "delete_allowed_before_replacement_takeover": False,
        }
        if target_id is not None:
            result["responsible_agent_id"] = target_id
        return result

    def _hotpotqa_semantic_repair_attribution(
        self,
        reason: str,
        *,
        stage: Optional[str],
        validation: Optional[GraphValidationResult],
    ) -> Optional[dict[str, object]]:
        """Attribute v2 failures without assuming direct role adjacency."""

        formatter_ids = tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold() == "format"
        )
        formatter_id = self._graph.output_agent_id
        output_ancestors = (
            ()
            if formatter_id is None or not self._graph.has_node(formatter_id)
            else self._directed_ancestor_ids(self._graph, formatter_id)
        )
        verifier_ids = tuple(
            sorted(
                (
                    node.id
                    for node in self._graph.nodes
                    if node.id in output_ancestors
                    and (node.role_family or "").casefold() == "verifier"
                ),
                key=lambda verifier_id: (
                    len(
                        self._directed_shortest_path(
                            self._graph,
                            verifier_id,
                            formatter_id,
                        )
                    ),
                    verifier_id,
                ),
            )
        )
        reasoner_ids = tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold() == "reasoner"
            and any(
                node.id
                in self._directed_ancestor_ids(self._graph, verifier_id)
                for verifier_id in verifier_ids
            )
        )
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
        graph_ids = {node.id for node in self._graph.nodes}
        failed_ids = tuple(sorted(self._failed_agent_ids & graph_ids))

        target_id: Optional[str] = None
        role_family: Optional[str] = None
        responsible_constraint = "semantic_lineage_construction"
        preferred_actions = ["set_relation", "add_subgraph", "modify_agent"]
        responsible_ids: Tuple[str, ...] = ()
        if unreachable_ids:
            target_id = unreachable_ids[0]
            role_family = self._graph.get_node(target_id).role_family
            responsible_ids = unreachable_ids
            responsible_constraint = "terminal_reachability"
            preferred_actions = ["set_relation", "modify_agent", "add_subgraph"]
        elif stage == "execution" and failed_ids:
            target_id = failed_ids[0]
            role_family = self._graph.get_node(target_id).role_family
            responsible_ids = failed_ids
            responsible_constraint = "execution_contract_or_runtime_failure"
            preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
        elif (
            formatter_id is None
            or "output_agent_count" in issue_codes
            or "unknown_output_agent" in issue_codes
        ):
            target_id = formatter_ids[0] if formatter_ids else None
            role_family = "format" if formatter_ids else None
            responsible_constraint = "format_output_assignment"
            preferred_actions = ["set_output", "set_relation", "add_subgraph"]
        else:
            quoted_id = re.search(r"(?:Reasoner|Verifier) '([^']+)'", reason)
            quoted_target = (
                None
                if quoted_id is None or quoted_id.group(1) not in graph_ids
                else quoted_id.group(1)
            )
            reason_folded = reason.casefold()
            verifier_verdict_failure = reason.startswith("Verifier") and any(
                f"{field_name!r} must be true" in reason
                for field_name in (
                    "evidence_supported",
                    "entity_attribute_binding_correct",
                    "alias_binding_correct",
                    "answer_type_cardinality_correct",
                    "multi_hop_complete",
                    "minimal_answer_surface",
                    "scope_preserved",
                )
            )
            if "no routed Verifier" in reason:
                # This is a relation/capability construction fault, not a
                # Formatter artifact fault.  Do not identify a healthy Format
                # Agent as the mandatory MODIFY target.
                target_id = None
                role_family = "verifier"
                responsible_constraint = "semantic_lineage_relation"
                preferred_actions = ["set_relation", "add_subgraph"]
            elif (
                "knowledge_base_coverage_failure" in reason_folded
                or reason.startswith("Reasoner")
                or verifier_verdict_failure
                or "no successful 'qa-retrieval' read receipt" in reason
                or "evidence provenance is invalid" in reason
            ):
                target_id = (
                    quoted_target
                    if quoted_target in reasoner_ids
                    else reasoner_ids[0]
                    if reasoner_ids
                    else None
                )
                role_family = "reasoner"
                responsible_constraint = "reasoner_semantic_artifact"
                preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
            elif reason.startswith("Verifier") or "Verifier changed" in reason:
                target_id = (
                    quoted_target
                    if quoted_target in verifier_ids
                    else verifier_ids[0]
                    if verifier_ids
                    else None
                )
                role_family = "verifier"
                responsible_constraint = "verifier_semantic_artifact"
                preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
            elif reason.startswith(("Format", "Formatter")):
                target_id = formatter_id
                role_family = "format"
                responsible_constraint = "format_lineage"
                preferred_actions = ["modify_agent", "set_relation", "set_output"]
            elif "disagree on candidate_answer" in reason:
                target_id = verifier_ids[0] if verifier_ids else None
                role_family = "verifier"
                responsible_constraint = "candidate_consistency"
                preferred_actions = ["modify_agent", "set_relation", "add_subgraph"]
            elif stage not in {"execution", "format_lineage"}:
                return None

        preserved = tuple(
            node.id
            for node in self._graph.nodes
            if self._has_successful_artifact(node.id)
            and node.id != target_id
        )
        result: dict[str, object] = {
            "responsible_constraint": responsible_constraint,
            "responsible_role_family": role_family,
            "responsible_agent_ids": list(responsible_ids),
            "format_target_agent_ids": list(formatter_ids),
            "preserve_agent_ids": list(preserved),
            "preferred_action_order": preferred_actions,
            "delete_allowed_before_replacement_takeover": False,
        }
        if target_id is not None:
            result["responsible_agent_id"] = target_id
        if "knowledge_base_coverage_failure" in reason.casefold():
            result["operational_diagnosis"] = "knowledge_base_coverage_failure"
            result["corpus_level_oracle_claim"] = False
        return result

    def _cached_progressive_execution(self) -> Optional[AgentRuntimeResult]:
        if self._progressive_execution_revision != self._graph.revision:
            return None
        if self._progressive_execution is None:
            return None
        if self._progressive_execution.final_answer is None:
            return None
        return self._progressive_execution

    def _capture_last_valid_evidence_lineage(
        self,
        execution: AgentRuntimeResult,
    ) -> None:
        """Atomically publish a complete execute-on-edit semantic lineage.

        ``finish_admissibility`` is the single gate authority: graph, Format
        lineage, required Tool ownership, terminal environment receipt,
        semantic provenance, and answer wrapper must all pass.  Invalid later
        revisions leave the previous frozen snapshot untouched.
        """

        if not self.execute_on_edit or not self._uses_semantic_lineage_protocol():
            return
        if (
            execution is not self._progressive_execution
            or execution.graph_revision != self._graph.revision
            or execution.final_answer is None
        ):
            return
        admission = self.finish_admissibility()
        if admission.get("admissible") is not True:
            return
        previous = self._last_valid_evidence_lineage
        if previous is not None and previous.graph_revision >= self._graph.revision:
            return
        candidate = AgentWorkflowEvidenceLineageSnapshot(
            answer=execution.final_answer,
            runtime=execution,
            graph_revision=self._graph.revision,
            graph_snapshot=self._graph.snapshot(),
        )
        # One assignment publishes the fully validated immutable snapshot.
        self._last_valid_evidence_lineage = candidate

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
        self._repair_exhausted_agent_ids.intersection_update(current_agent_ids)
        self._latest_failure_record_by_agent = {
            agent_id: record
            for agent_id, record in self._latest_failure_record_by_agent.items()
            if agent_id in current_agent_ids
        }
        self._pending_repair_receipt_count_by_agent = {
            agent_id: receipt_count
            for agent_id, receipt_count in (
                self._pending_repair_receipt_count_by_agent.items()
            )
            if agent_id in current_agent_ids
        }
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
        source_agent_id = record.metadata.get("continuation_source_agent_id")
        if isinstance(source_agent_id, str) and source_agent_id.strip():
            result["continuation_source_agent_id"] = source_agent_id.strip()
        return result if len(result) > 1 else None

    def _recovery_continuation_handoff(
        self,
        action: AgentAction,
    ) -> dict[str, dict[str, object]]:
        """Project one failed Agent's public Tool state to a new Retriever.

        The source is either a same-role/same-artifact Retriever replacement or
        the existing bounded Reasoner-to-Retriever recovery handoff.  Normal
        repair remains same-Agent and phase-scoped.  This narrowly admitted
        augmentation is public recovery state at FlowSteer's edit--execute
        boundary; it does not create an AgentGraph message edge or transfer a
        semantic artifact, model transcript, Ground Truth, or evaluator state.
        """

        if (
            not self._uses_semantic_lineage_protocol()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            or action.action_type is not AgentActionType.ADD_SUBGRAPH
            or len(action.agents) != 1
            or self.required_evidence_tool_id is None
        ):
            return {}
        target = action.agents[0]
        if (
            (target.role_family or "").casefold() != "evidence_retriever"
            or target.execution_mode != "react"
            or target.allowed_tools is None
            or self.required_evidence_tool_id not in target.allowed_tools
        ):
            return {}
        target_artifact_type = (target.artifact_type or "text").casefold()
        replacement_source_ids = tuple(
            node.id
            for node in self._graph.nodes
            if node.id in self._failed_agent_ids
            and node.id in self._repair_exhausted_agent_ids
            and node.id in self._failure_continuations
            and (node.role_family or "").casefold()
            == "evidence_retriever"
            and node.artifact_type.casefold() == target_artifact_type
        )
        source_ids = replacement_source_ids or tuple(
            source_id
            for source_id in self._repair_exhausted_reasoner_ids()
            if source_id in self._failure_continuations
        )
        if len(source_ids) != 1:
            return {}
        source_id = source_ids[0]
        result = self._tool_continuation_projection(
            self._failure_continuations[source_id],
            execution_phase="single",
            source_agent_id=source_id,
        )
        return {target.agent_id: result} if result is not None else {}

    def _tool_continuation_projection(
        self,
        source: Mapping[str, object],
        *,
        execution_phase: str,
        source_agent_id: str,
    ) -> Optional[dict[str, object]]:
        """Project only public Tool state across an Agent-role boundary."""

        result: dict[str, object] = {
            # The newly added one-way auxiliary Agent executes as a singleton.
            # Bind the envelope to its target phase while preserving measured
            # Tool receipts and dispatched public Tool Action--Observation
            # state.  A source Reasoner's rejected completion is role-specific
            # and must not become a Retriever imitation target.  Tool errors
            # remain public because those dispatches consumed the shared budget.
            "execution_phase": execution_phase,
            "continuation_source_agent_id": source_agent_id,
        }
        raw_trace = source.get("react_trace", ())
        if isinstance(raw_trace, (list, tuple)):
            tool_trace = []
            for raw_item in raw_trace:
                if not isinstance(raw_item, Mapping):
                    continue
                observation = raw_item.get("observation")
                if not isinstance(observation, Mapping):
                    continue
                executed_action = observation.get("executed_action")
                if (
                    not isinstance(executed_action, Mapping)
                    or executed_action.get("kind") != "tool"
                    or executed_action.get("resource_id")
                    != self.required_evidence_tool_id
                ):
                    continue
                tool_trace.append(dict(raw_item))
            if tool_trace:
                result["react_trace"] = tool_trace
        raw_receipts = source.get("tool_receipts", ())
        if isinstance(raw_receipts, (list, tuple)):
            receipts = [
                dict(item)
                for item in raw_receipts
                if isinstance(item, Mapping)
                and item.get("tool_id") == self.required_evidence_tool_id
            ]
            if receipts:
                result["tool_receipts"] = receipts
        if not any(field_name in result for field_name in ("react_trace", "tool_receipts")):
            return None
        return result

    @staticmethod
    def _failure_continuation_weight(metadata: Mapping[str, object]) -> tuple[int, int]:
        """Prefer the most advanced public continuation for an Agent."""

        trace = metadata.get("react_trace", ())
        receipts = metadata.get("tool_receipts", ())
        return (
            len(receipts) if isinstance(receipts, (list, tuple)) else 0,
            len(trace) if isinstance(trace, (list, tuple)) else 0,
        )

    def _mark_agents_recovered(self, agent_ids: Collection[str]) -> None:
        """Clear failure-only state after those Agents produced artifacts."""

        recovered = set(agent_ids)
        self._failed_agent_ids.difference_update(recovered)
        self._diagnosed_unusable_agent_ids.difference_update(recovered)
        self._react_exhausted_agent_ids.difference_update(recovered)
        self._repair_exhausted_agent_ids.difference_update(recovered)
        for agent_id in recovered:
            self._failure_continuations.pop(agent_id, None)
            self._latest_failure_record_by_agent.pop(agent_id, None)
            self._pending_repair_receipt_count_by_agent.pop(agent_id, None)

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
            category, _, _ = self._execution_failure_diagnosis(record)
            continuation = self._failure_continuation_candidate(record)
            if continuation is not None and not any(
                field_name in continuation
                for field_name in ("react_trace", "tool_receipts")
            ):
                source_agent_id = continuation.get(
                    "continuation_source_agent_id"
                )
                source_continuation = (
                    self._failure_continuations.get(source_agent_id)
                    if isinstance(source_agent_id, str)
                    else None
                )
                if source_continuation is not None:
                    projected = self._tool_continuation_projection(
                        source_continuation,
                        execution_phase=record.phase.value,
                        source_agent_id=source_agent_id,
                    )
                    if projected is not None:
                        # The provider failure or scheduler cancellation
                        # occurred before the target adapter published its own
                        # trace. Preserve the same bounded public Tool state.
                        continuation = projected
            if category == "sibling_fail_fast_cancellation":
                # AgentRuntime cancels still-running sibling blocks after the
                # first measured failure.  Preserve any public ReAct prefix or
                # Tool receipts published before cancellation, but do not
                # diagnose the sibling Agent itself as failed or repairable.
                if continuation is not None:
                    current = self._failure_continuations.get(record.agent_id)
                    if current is None or self._failure_continuation_weight(
                        continuation
                    ) >= self._failure_continuation_weight(current):
                        self._failure_continuations[record.agent_id] = continuation
                continue
            recorded_agent_ids.add(record.agent_id)
            self._failed_agent_ids.add(record.agent_id)
            self._latest_failure_record_by_agent[record.agent_id] = record
            if category == "react_turn_exhaustion":
                react_exhausted_agent_ids.add(record.agent_id)
            if record.metadata.get("node_unusable") is True:
                self._diagnosed_unusable_agent_ids.add(record.agent_id)
            else:
                self._diagnosed_unusable_agent_ids.discard(record.agent_id)
            current_continuation = self._failure_continuations.get(
                record.agent_id
            )
            current_receipt_count = (
                0
                if current_continuation is None
                else self._failure_continuation_weight(current_continuation)[0]
            )
            baseline_receipt_count = self._pending_repair_receipt_count_by_agent.pop(
                record.agent_id,
                None,
            )
            new_receipt_count = (
                current_receipt_count
                if continuation is None
                else self._failure_continuation_weight(continuation)[0]
            )
            if (
                category == "react_turn_exhaustion"
                and (
                    (
                        baseline_receipt_count is not None
                        and new_receipt_count <= baseline_receipt_count
                    )
                    or (
                        record.agent_id in self._repair_exhausted_agent_ids
                        and new_receipt_count <= current_receipt_count
                    )
                )
            ):
                self._repair_exhausted_agent_ids.add(record.agent_id)
            else:
                self._repair_exhausted_agent_ids.discard(record.agent_id)
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
        self._repair_exhausted_agent_ids.clear()
        self._failure_continuations.clear()
        self._latest_failure_record_by_agent.clear()
        self._pending_repair_receipt_count_by_agent.clear()

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
        if not self._uses_format_agent_protocol(graph):
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
        admitted_format_modes = (
            {"reasoning", "react"}
            if self._uses_role_conditional_capabilities()
            else {"reasoning"}
        )
        if (
            output_node.execution_mode.value not in admitted_format_modes
            or output_node.allowed_tools
        ):
            return (
                "Format Agent must use a registered reasoning or ReAct execution "
                "profile without tools; it only serializes the determined semantic "
                "answer character-for-character and must not invoke a Tool, reselect "
                "the answer, or participate in reasoning"
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
        if self._uses_role_conditional_capabilities():
            if not predecessors:
                return (
                    "Format Agent must consume at least one routed upstream "
                    "artifact containing an already determined semantic answer"
                )
            return None
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            if not predecessors:
                return (
                    "Format Agent must consume at least one routed upstream "
                    "artifact containing an already verified semantic answer"
                )
            output_ancestors = set(
                self._directed_ancestor_ids(graph, output_agent_id)
            )
            verifier_ids = tuple(
                node.id
                for node in graph.nodes
                if node.id in output_ancestors
                and (node.role_family or "").casefold() == "verifier"
            )
            if not verifier_ids:
                return (
                    "HotpotQA Format Agent must have at least one routed "
                    "Verifier ancestor before Output is assigned; intermediate "
                    "Agents and non-chain topology remain admissible"
                )
            routed_reasoner_ids = tuple(
                node.id
                for node in graph.nodes
                if (node.role_family or "").casefold() == "reasoner"
                and any(
                    node.id in self._directed_ancestor_ids(graph, verifier_id)
                    for verifier_id in verifier_ids
                )
            )
            if not routed_reasoner_ids:
                return (
                    "HotpotQA Verifier lineage must have at least one routed "
                    "Reasoner ancestor before Output is assigned; direct role "
                    "adjacency is not required"
                )
            evidence_capable_reasoner_ids = tuple(
                reasoner_id
                for reasoner_id in routed_reasoner_ids
                if (
                    self._graph_agent_has_evidence_tool(graph, reasoner_id)
                    or any(
                        self._graph_agent_has_evidence_tool(graph, ancestor_id)
                        for ancestor_id in self._directed_ancestor_ids(
                            graph,
                            reasoner_id,
                        )
                    )
                )
            )
            if not evidence_capable_reasoner_ids:
                return (
                    "HotpotQA routed semantic lineage must give at least one "
                    "Reasoner access to explicit retrieval evidence, either "
                    "through its own ReAct qa-retrieval capability or through "
                    "a routed upstream ReAct retrieval Agent"
                )
            return None
        if len(predecessors) != 1:
            return (
                "Format Agent must consume exactly one upstream semantic-answer "
                f"artifact; received {len(predecessors)}; add or retain one "
                "semantic-answer Agent and one directed relation from that Agent "
                "to the Format Agent before FINISH"
            )
        if self._uses_semantic_lineage_protocol():
            protocol_label = self._semantic_protocol_label()
            verifier_id = predecessors[0]
            verifier = graph.get_node(verifier_id)
            if (verifier.role_family or "").casefold() != "verifier":
                return (
                    f"{protocol_label} Format Agent's unique predecessor must have "
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
                    f"{protocol_label} Verifier must consume exactly one direct "
                    "Reasoner "
                    "semantic-candidate artifact; found "
                    f"{len(reasoner_predecessors)}. Preserve the original question "
                    "scope and use set_relation or modify_agent before FINISH"
                )
            if tuple(graph.directed_predecessors(verifier_id)) != reasoner_predecessors:
                return (
                    f"{protocol_label} Verifier must receive its semantic artifact "
                    "only from "
                    "the unique Reasoner. Route any retrieval or repair evidence "
                    "into the Reasoner first so it can align propositions to the "
                    "original answer slot before verification"
                )
        return None

    def _graph_agent_has_evidence_tool(
        self,
        graph: AgentGraph,
        agent_id: str,
    ) -> bool:
        """Return whether one routed Agent can materialize QA evidence."""

        if not graph.has_node(agent_id):
            return False
        node = graph.get_node(agent_id)
        return (
            node.execution_mode.value == "react"
            and self.required_evidence_tool_id is not None
            and node.allowed_tools == (self.required_evidence_tool_id,)
        )

    def _uses_format_agent_protocol(
        self,
        graph: Optional[AgentGraph] = None,
    ) -> bool:
        """Enable FlowSteer's Format boundary only when it is required or selected."""

        resolved_graph = self._graph if graph is None else graph
        if self.require_format_agent:
            return True
        if self._uses_role_conditional_capabilities():
            output_agent_id = resolved_graph.output_agent_id
            return bool(
                output_agent_id is not None
                and resolved_graph.has_node(output_agent_id)
                and (
                    resolved_graph.get_node(output_agent_id).role_family or ""
                ).casefold()
                == "format"
            )
        return self._uses_semantic_lineage_protocol()

    def _semantic_edit_issue_for(self, graph: AgentGraph) -> Optional[str]:
        """Enforce the evidence-grounded QA lineage after every Canvas edit.

        The checks are incremental: an unconnected node may exist while the
        Director is still assembling a functional subgraph, but an existing
        edge may not bypass the Reasoner/Verifier/Formatter responsibility
        boundary.  This keeps FlowSteer's edit--execute--feedback transaction
        intact without imposing one fixed graph topology.
        """

        if not self._uses_semantic_lineage_protocol():
            return None
        if self._uses_role_conditional_capabilities():
            return self._hotpotqa_semantic_lineage_edit_issue_for(graph)
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            return self._hotpotqa_semantic_lineage_edit_issue_for(graph)
        protocol_label = self._semantic_protocol_label()
        missing_role_ids = tuple(
            node.id
            for node in graph.nodes
            if not (node.role_family or "").strip()
        )
        if missing_role_ids:
            return (
                f"{protocol_label} semantic protocol requires a non-empty "
                "role_family "
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
                f"{protocol_label} semantic protocol rejects role_family='react' "
                "for Agents "
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
                    f"{protocol_label} {role.title()} Agent {node.id!r} has a "
                    "formatting-only "
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
                    f"{protocol_label} Reasoner Agent {node.id!r} must use "
                    "execution_mode='react' with exactly "
                    f"allowed_tools=['{self.required_evidence_tool_id}']; ReAct is "
                    "the Thought -> Action(tool) -> Observation -> Thought -> Final "
                    "execution schedule, while role_family remains 'reasoner'"
                )
            if role in {"verifier", "format"} and (
                node.execution_mode.value != "reasoning" or node.allowed_tools
            ):
                return (
                    f"{protocol_label} {role.title()} Agent {node.id!r} must use "
                    "execution_mode='reasoning' without Tools"
                )
            if role == "format" and node.contract != _HOTPOTQA_FORMAT_CONTRACT:
                return (
                    f"{protocol_label} Formatter Agent {node.id!r} must use the "
                    "neutral "
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
                        f"{protocol_label} Verifier {node.id!r} must receive its "
                        "direct "
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
                        f"{protocol_label} Formatter {node.id!r} must receive its "
                        "direct "
                        "input only from Verifiers and must not participate in "
                        f"reasoning; invalid predecessors={list(invalid_predecessors)!r}"
                    )
                successors = self._directed_successors(graph, node.id)
                if successors:
                    return (
                        f"{protocol_label} Formatter {node.id!r} must be a "
                        "terminal sink; "
                        f"remove outgoing directed edges to {list(successors)!r}"
                    )
        return None

    def _hotpotqa_semantic_lineage_edit_issue_for(
        self,
        graph: AgentGraph,
    ) -> Optional[str]:
        """Validate semantic capabilities without prescribing graph topology.

        FlowSteer remains authoritative for graph structure and progressive
        edit--execute--feedback.  These checks only separate semantic roles
        from SkillFlow execution modes and keep the Formatter a pure terminal
        serializer.  Actual Reasoner--Verifier lineage is validated from routed
        artifacts and Tool receipts at FINISH.
        """

        missing_role_ids = tuple(
            node.id
            for node in graph.nodes
            if not (node.role_family or "").strip()
        )
        if missing_role_ids:
            return (
                "HotpotQA semantic protocol requires a non-empty role_family "
                f"for every Agent; missing role_family for {list(missing_role_ids)!r}"
            )
        invalid_role_ids = tuple(
            node.id
            for node in graph.nodes
            if (node.role_family or "").casefold() == "react"
        )
        if invalid_role_ids:
            return (
                "HotpotQA semantic protocol rejects role_family='react' for "
                f"Agents {list(invalid_role_ids)!r}; ReAct is execution_mode='react' "
                "(Thought -> Action(tool) -> Observation -> Thought -> Final), "
                "not an Agent role"
            )
        if (
            self._uses_role_conditional_capabilities()
            and graph.output_agent_id is not None
            and (
                graph.get_node(graph.output_agent_id).role_family or ""
            ).casefold()
            not in {"format", "output"}
        ):
            return (
                "HotpotQA selected Output Agent must use a terminal-compatible "
                "role_family 'output' or the optional formatting-only "
                "role_family 'format'; structured semantic and ReAct artifacts "
                "remain internal capabilities"
            )
        if self._uses_role_conditional_capabilities():
            formatter_ids = tuple(
                node.id
                for node in graph.nodes
                if (node.role_family or "").casefold() == "format"
            )
            if formatter_ids and (
                len(formatter_ids) != 1
                or graph.output_agent_id != formatter_ids[0]
            ):
                return (
                    "A Formatter is optional, but when selected it must be the "
                    "unique Output Agent in the same Canvas revision; it is a "
                    "terminal serializer and cannot remain a deferred non-Output "
                    "node. formatter_agent_ids="
                    f"{list(formatter_ids)!r}, output_agent_id="
                    f"{graph.output_agent_id!r}"
                )

        formatting_only_contract = " ".join(
            (
                _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT
                if self._uses_role_conditional_capabilities()
                else _HOTPOTQA_FORMAT_CONTRACT
            ).casefold().split()
        ).rstrip(".")
        for node in graph.nodes:
            role = (node.role_family or "").casefold()
            normalized_contract = " ".join(
                node.contract.casefold().split()
            ).rstrip(".")
            if (
                role in {"reasoner", "verifier"}
                and normalized_contract == formatting_only_contract
            ):
                return (
                    f"HotpotQA {role.title()} Agent {node.id!r} has a "
                    "formatting-only contract; the Reasoner owns the semantic "
                    "answer and the Verifier checks evidence, binding, hops, and scope"
                )
            if self._uses_role_conditional_capabilities() and role in {
                "reasoner",
                "verifier",
                "format",
                "evidence_retriever",
                "repair",
                "output",
            }:
                profile = (
                    node.execution_mode.value,
                    tuple(node.allowed_tools),
                )
                admitted_profiles = (
                    self._role_conditional_execution_profiles_for(role)
                )
                if profile not in admitted_profiles:
                    return (
                        f"HotpotQA {role.replace('_', ' ').title()} Agent "
                        f"{node.id!r} execution profile is not registered or "
                        "is incompatible with that semantic responsibility; "
                        "execution_mode and allowed_tools must match one "
                        f"published profile {list(admitted_profiles)!r}"
                    )
            elif role == "reasoner":
                react_with_evidence = (
                    node.execution_mode.value == "react"
                    and node.allowed_tools == (self.required_evidence_tool_id,)
                )
                routed_reasoning = (
                    node.execution_mode.value == "reasoning"
                    and not node.allowed_tools
                )
                if not (react_with_evidence or routed_reasoning):
                    return (
                        f"HotpotQA Reasoner Agent {node.id!r} must either use "
                        "execution_mode='react' with exactly "
                        f"allowed_tools=['{self.required_evidence_tool_id}'] or "
                        "execution_mode='reasoning' without Tools and consume "
                        "routed evidence; role_family remains 'reasoner'"
                    )
            if (
                not self._uses_role_conditional_capabilities()
                and role in {"verifier", "format"}
                and (
                node.execution_mode.value != "reasoning" or node.allowed_tools
                )
            ):
                return (
                    f"HotpotQA {role.title()} Agent {node.id!r} must use "
                    "execution_mode='reasoning' without Tools"
                )
            if role == "verifier" and self._uses_role_conditional_capabilities():
                invalid_predecessors = tuple(
                    predecessor_id
                    for predecessor_id in graph.directed_predecessors(node.id)
                    if (
                        graph.get_node(predecessor_id).role_family or ""
                    ).casefold()
                    in {"evidence_retriever", "format", "output"}
                )
                if invalid_predecessors:
                    return (
                        f"HotpotQA Verifier Agent {node.id!r} must consume an "
                        "already determined semantic-candidate artifact, not raw "
                        "retrieval evidence or a terminal wrapper. A named "
                        "Reasoner is optional; any routed non-terminal semantic "
                        "producer is validated from its artifact at FINISH. invalid "
                        f"predecessors={list(invalid_predecessors)!r}"
                    )
            if role == "format":
                expected_format_contract = (
                    _HOTPOTQA_ROLE_CONDITIONAL_FORMAT_CONTRACT
                    if self._uses_role_conditional_capabilities()
                    else _HOTPOTQA_FORMAT_CONTRACT
                )
                if node.contract != expected_format_contract:
                    return (
                        f"HotpotQA Formatter Agent {node.id!r} must use the "
                        "formatting-only contract and must not select or infer an answer"
                    )
                if self._uses_role_conditional_capabilities():
                    invalid_predecessors = tuple(
                        predecessor_id
                        for predecessor_id in graph.directed_predecessors(node.id)
                        if (
                            graph.get_node(predecessor_id).role_family or ""
                        ).casefold()
                        in {"evidence_retriever", "format", "output"}
                    )
                    if invalid_predecessors:
                        return (
                            f"HotpotQA Formatter Agent {node.id!r} must consume "
                            "an already determined semantic-candidate artifact, "
                            "not raw retrieval evidence or another terminal wrapper. "
                            "No named Reasoner or Verifier ancestor is required; "
                            "the routed artifact is authoritative at FINISH. "
                            "invalid predecessors="
                            f"{list(invalid_predecessors)!r}"
                        )
                successors = self._directed_successors(graph, node.id)
                if successors:
                    return (
                        f"HotpotQA Formatter Agent {node.id!r} must be a terminal "
                        f"sink; remove outgoing directed edges to {list(successors)!r}"
                    )
        if (
            self._uses_role_conditional_capabilities()
            and graph.output_agent_id is not None
        ):
            unrouted_verifier_ids = tuple(
                node.id
                for node in graph.nodes
                if (node.role_family or "").casefold() == "verifier"
                and not graph.directed_predecessors(node.id)
            )
            if unrouted_verifier_ids:
                return (
                    "A selected Verifier is a routed semantic consumer and must "
                    "receive at least one upstream artifact before or atomically "
                    "with Output assignment; no Reasoner role or serial topology "
                    "is required. unrouted_verifier_agent_ids="
                    f"{list(unrouted_verifier_ids)!r}"
                )
        if graph.output_agent_id is not None:
            # FlowSteer's complete-Canvas validator is the authority for
            # terminal reachability.  Once Output is assigned, accepting an
            # orphan branch would execute downstream Agents and materialize
            # artifacts which the preservation policy must then protect.  The
            # Director could no longer attach that orphan without invalidating
            # an already successful dependency, producing a terminal deadlock.
            # Require the entire current Canvas to reach Output at the same
            # progressive edit boundary which assigns (or retains) Output.
            # This constrains only completeness, not the topology: fan-in,
            # reciprocal blocks, intermediate Agents, and multiple semantic
            # paths remain admissible.
            unreachable_agent_ids = tuple(
                sorted(
                    {
                        agent_id
                        for issue in graph.validate(
                            self.model_registry,
                            require_complete=True,
                        ).issues
                        if issue.code == "cannot_reach_output"
                        for agent_id in issue.agent_ids
                    }
                )
            )
            if unreachable_agent_ids:
                return (
                    "HotpotQA Output may be assigned only when every current "
                    "Agent can reach that Output in the same Canvas revision; "
                    "route the unresolved branch before SET_OUTPUT or include "
                    "its relation in the same ADD_SUBGRAPH edit. "
                    "terminal_unreachable_agent_ids="
                    f"{list(unreachable_agent_ids)!r}"
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
        observations remain available to the next turn.  This evidence-grounded
        QA admission prevents those observations from being copied into a
        pre-execution contract as a concrete answer.  It never rewrites a sampled
        action and never consults an answer key or evaluator.
        """

        if not self._uses_semantic_lineage_protocol():
            return None
        obligations: Tuple[str, ...]
        scoped_obligations: Tuple[Tuple[str, str], ...]
        if action.action_type is AgentActionType.ADD_SUBGRAPH:
            obligations = tuple(
                value
                for spec in action.agents
                for value in (spec.contract, spec.completion_condition)
                if value is not None
            )
            scoped_obligations = tuple(
                ((spec.role_family or "").casefold(), value)
                for spec in action.agents
                for value in (spec.contract, spec.completion_condition)
                if value is not None
            )
        elif action.action_type is AgentActionType.MODIFY_AGENT:
            obligations = tuple(
                value
                for value in (action.contract, action.completion_condition)
                if value is not None
            )
            existing_role = (
                (
                    self._graph.get_node(action.agent_id).role_family
                    if action.agent_id is not None
                    and self._graph.has_node(action.agent_id)
                    else None
                )
                or ""
            ).casefold()
            scoped_obligations = tuple(
                ((action.role_family or existing_role).casefold(), value)
                for value in (action.contract, action.completion_condition)
                if value is not None
            )
        else:
            return None
        if not obligations:
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

        allowed_protocol_literals = {
            "answer_field",
            "answer_slot",
            "candidate_answer",
            "complete",
            "evidence",
            "evidence_propositions",
            "knowledge_base_coverage_failure",
            "multi_hop_chain",
            "object_or_attribute_value",
            "question_scope",
            "structuredaction",
            "subject",
        }
        if self.required_evidence_tool_id is not None:
            allowed_protocol_literals.add(
                self.required_evidence_tool_id.casefold()
            )
        quoted_directive_literal = re.compile(
            r"\b(?:answer|candidate|value|return|select|choose|emit|output|copy|"
            r"word|string|substring|token|known\s+fact|e\.g\.|for\s+example)\b"
            r"[^.!?\n]{0,96}?[\"']([^\"'\n]{1,80})[\"']",
            flags=re.IGNORECASE,
        )
        bare_directive_literal = re.compile(
            r"(?i:\b(?:return|output|emit)\b)\s+"
            r"(?i:(?:only\s+)?(?:the\s+)?(?:word|answer|candidate|value))\s+"
            r"([A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,5}"
            r"|\d{3,}(?:[.,]\d+)?)",
        )

        # PROJECT_NECESSARY_ADAPTATION: FlowSteer's Director authors semantic
        # Agent obligations, while SkillFlow's request-scoped action schema is
        # the authority for concrete Tool arguments.  Reject only explicit
        # invocations or literal argument values; ordinary responsibility terms
        # such as query rewriting, entity disambiguation, and expanded top-k
        # remain available to the Director.
        concrete_tool_argument_patterns = (
            re.compile(r"\b(?:search|read)\s*\(", flags=re.IGNORECASE),
            re.compile(
                r"\b(?:search|retrieve|read)\b[^.!?\n]{0,80}?"
                r"\bfor\s+[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:using|with)\s+(?:the\s+)?"
                r"(?:query|search\s+phrase)\s+[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"[\"']query[\"']\s*:\s*[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bquery\s*=\s*[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"[\"']?(?:limit|top[_ -]?k)[\"']?\s*(?:=|:)\s*\d+",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"[\"']?passage[_ -]?id[\"']?\s*(?:=|:)\s*"
                r"[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
        )

        if any(
            pattern.search(obligation) is not None
            for obligation in obligations
            for pattern in concrete_tool_argument_patterns
        ):
            return (
                f"{self._semantic_protocol_label()} Agent contract and "
                "completion_condition fields must express semantic responsibility "
                "and terminal predicates without concrete Tool invocation arguments. "
                "The Runtime state-conditioned action schema selects the exact query, "
                "top-k, and passage_id from current public observations"
            )

        # PROJECT_NECESSARY_ADAPTATION: the original FlowSteer Canvas admits
        # free-text contracts, so the HotpotQA semantic protocol must reject a
        # Director-authored scope restriction before it becomes an executable
        # Agent obligation. This check is question-only and never reads the
        # answer key, evaluator, or a task-specific allowed-value list.
        scope_narrowing_verb = re.compile(
            r"\b(?:restrict|limit|narrow|constrain|filter|exclude|omit|ignore|"
            r"disregard)\b",
            flags=re.IGNORECASE,
        )
        implicit_scope_narrowing = re.compile(
            r"\b(?:focus(?:ed)?\s+on|solely|exclusively)\b|"
            r"\bonly\s+(?!(?:a|an|the|one|answer|candidate|evidence|output|"
            r"result|semantic|value|wrapper)\b)",
            flags=re.IGNORECASE,
        )
        scope_neutral_tokens = {
            "a",
            "an",
            "and",
            "answer",
            "answerfield",
            "answerslot",
            "artifact",
            "candidate",
            "comparison",
            "contract",
            "database",
            "evidence",
            "exact",
            "explicit",
            "format",
            "from",
            "in",
            "input",
            "multi",
            "of",
            "one",
            "only",
            "original",
            "output",
            "passage",
            "passages",
            "question",
            "reasoning",
            "relation",
            "retrieval",
            "scope",
            "semantic",
            "slot",
            "source",
            "sources",
            "the",
            "to",
            "tool",
            "using",
            "verified",
            "wrapper",
        }
        question_token_set = set(question_tokens)
        narrowing_tokens = {
            "restrict",
            "limit",
            "narrow",
            "constrain",
            "filter",
            "exclude",
            "omit",
            "ignore",
            "disregard",
        }
        for role_family, obligation in scoped_obligations:
            if role_family == "format" or (
                scope_narrowing_verb.search(obligation) is None
                and implicit_scope_narrowing.search(obligation) is None
            ):
                continue
            unauthorized_qualifier_tokens = tuple(
                token
                for token in self._lexical_tokens(obligation)
                if token not in question_token_set
                and token not in scope_neutral_tokens
                and token not in allowed_protocol_literals
                and token not in narrowing_tokens
            )
            if unauthorized_qualifier_tokens:
                return (
                    f"{self._semantic_protocol_label()} Agent contract must "
                    "preserve the exact original question scope and may not add "
                    "an unrequested category, qualifier, subset, exclusion, or "
                    "comparison restriction; unauthorized_scope_tokens="
                    f"{list(dict.fromkeys(unauthorized_qualifier_tokens))!r}"
                )

        # Comparison contracts may decompose retrieval across operands, but a
        # semantic producer must not preselect one question-side operand before
        # the evidence comparison executes.  This question-only admission is a
        # narrow extension of the existing scope guard; it never consults a
        # Ground Truth value or evaluator state.
        comparison_tail = re.search(
            r",\s*([^,?]+?)\s+or\s+([^,?]+?)\s*\?\s*$",
            question,
            flags=re.IGNORECASE,
        )
        comparison_marker = re.search(
            r"\b(?:more|fewer|less|greater|larger|smaller|higher|lower|older|"
            r"younger|earlier|later|longer|shorter|better|worse|closer|farther)\b",
            question,
            flags=re.IGNORECASE,
        )
        selection_directive = re.compile(
            r"\b(?:extract|select|choose|return|emit|output|copy)\b",
            flags=re.IGNORECASE,
        )
        if comparison_tail is not None and comparison_marker is not None:
            operands = tuple(
                item.strip(" \t\n\r\"'()")
                for item in comparison_tail.groups()
            )
            for role_family, obligation in scoped_obligations:
                if role_family in {"evidence_retriever", "format"} or not (
                    selection_directive.search(obligation)
                ):
                    continue
                mentioned_operands = tuple(
                    operand
                    for operand in operands
                    if operand and self._contains_lexical_span(obligation, operand)
                )
                if len(mentioned_operands) == 1:
                    return (
                        f"{self._semantic_protocol_label()} comparison contract "
                        "must preserve both question operands and the original "
                        "comparison criterion; it may not preselect one operand "
                        "as the answer before evidence execution"
                    )

        concrete_entity_mapping = re.compile(
            r"\b(?:resolve|map|link|normalize|disambiguate)\b"
            r"[^.!?\n]{0,96}?[\"']([^\"'\n]+)[\"']\s+"
            r"(?:to|as|into|->)\s+[\"']([^\"'\n]+)[\"']",
            flags=re.IGNORECASE,
        )
        if any(
            any(
                not all(
                    literal.strip().casefold() in allowed_protocol_literals
                    or question_contains(literal)
                    for literal in match.groups()
                )
                for match in concrete_entity_mapping.finditer(obligation)
            )
            for obligation in obligations
        ):
            return (
                f"{self._semantic_protocol_label()} Agent contract and "
                "completion_condition fields are pre-execution obligations only: "
                "entity linking and alias resolution must remain evidence-grounded "
                "responsibilities, not a concrete precommitted entity mapping"
            )

        for obligation in obligations:
            # Exact public semantic values include one-word entities that a
            # proper-name phrase detector cannot safely infer from raw prose.
            if any(
                not question_contains(literal)
                and self._contains_lexical_span(obligation, literal)
                for literal in public_literals
            ):
                break
            if any(
                not question_contains(phrase)
                and self._contains_lexical_span(obligation, phrase)
                for phrase in named_phrases
            ):
                break

            # The evaluator and Ground Truth remain invisible. Reject only an
            # explicit pre-execution answer directive sampled by the Director.
            # Protocol field names and the declared Tool id remain available
            # for neutral schema obligations.
            directive_literals = [
                *quoted_directive_literal.findall(obligation),
                *bare_directive_literal.findall(obligation),
            ]
            if re.search(r"\bknown\s+fact\s*:", obligation, re.IGNORECASE):
                break
            if any(
                literal.strip().casefold() not in allowed_protocol_literals
                and not question_contains(literal)
                for literal in directive_literals
            ):
                break

            # A copied evidence sentence need not contain a named entity.  Six
            # contiguous lexical tokens are long enough to identify a passage
            # fragment while leaving ordinary scope/relation obligations free.
            contract_tokens = self._lexical_tokens(obligation)
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
                obligation,
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
            f"{self._semantic_protocol_label()} Agent contract and "
            "completion_condition fields are pre-execution obligations only: a "
            "new or modified obligation must not embed a concrete candidate, alias, "
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
        minimum_evidence_propositions: int = 2,
        minimum_reasoning_steps: int = 2,
    ) -> tuple[Optional[str], Optional[str]]:
        if (
            isinstance(minimum_evidence_propositions, bool)
            or not isinstance(minimum_evidence_propositions, int)
            or minimum_evidence_propositions < 1
        ):
            raise ValueError("minimum_evidence_propositions must be positive")
        if (
            isinstance(minimum_reasoning_steps, bool)
            or not isinstance(minimum_reasoning_steps, int)
            or minimum_reasoning_steps < 1
        ):
            raise ValueError("minimum_reasoning_steps must be positive")
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
        normalized_question = (
            ""
            if original_question is None
            else " ".join(
                hotpotqa_question_scope(original_question).casefold().split()
            )
        )
        who_question = bool(
            normalized_question.startswith("who ")
            or re.search(r"\bwho\s*\?\s*$", normalized_question)
        )
        if (
            expected_answer_type is not None
            and original_question is not None
            and not qa_answer_type_constraint_accepts(
                original_question,
                answer_slot["answer_type"],
            )
        ):
            return None, (
                "Reasoner answer_slot.answer_type must be compatible with the "
                "original question's answer-type constraint "
                f"{expected_answer_type!r}"
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
        if (
            not isinstance(propositions, (list, tuple))
            or len(propositions) < minimum_evidence_propositions
        ):
            if minimum_evidence_propositions == 2:
                return None, (
                    "Reasoner field 'evidence_propositions' must contain at least two "
                    "propositions for HotpotQA multi-hop alignment"
                )
            return None, (
                "Reasoner field 'evidence_propositions' must contain at least "
                f"{minimum_evidence_propositions} proposition(s) for "
                "evidence-grounded answer-slot alignment"
            )
        multi_hop_chain = fields["multi_hop_chain"]
        if (
            not isinstance(multi_hop_chain, (list, tuple))
            or len(multi_hop_chain) < minimum_reasoning_steps
            or any(
                not isinstance(item, str) or not item.strip()
                for item in multi_hop_chain
            )
        ):
            if minimum_reasoning_steps == 2:
                return None, (
                    "Reasoner field 'multi_hop_chain' must contain at least two "
                    "non-empty hop descriptions"
                )
            return None, (
                "Reasoner field 'multi_hop_chain' must contain at least "
                f"{minimum_reasoning_steps} non-empty reasoning step(s)"
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
        if expected_answer_type in {"entity", "person", "location"} and re.fullmatch(
            r"[\d\s.,:/-]+",
            candidate,
        ):
            return None, (
                f"Reasoner candidate_answer is numeric/date-like but the original "
                f"question requires answer type {expected_answer_type!r}"
            )
        if who_question and re.search(
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
        if candidate != selected[answer_field]:
            matching_fields = tuple(
                field_name
                for field_name in ("subject", "object_or_attribute_value")
                if selected.get(field_name) == candidate
            )
            if len(matching_fields) == 1:
                return None, (
                    "Reasoner answer_slot.answer_field selects "
                    f"{answer_field!r}, but candidate_answer exactly matches the "
                    f"selected proposition field {matching_fields[0]!r}; set "
                    "answer_field to the proposition field containing "
                    "candidate_answer"
                )
            return None, (
                "Reasoner candidate_answer must copy the proposition argument "
                "identified by answer_slot.proposition_index and answer_field exactly"
            )
        if who_question:
            possessor_surface_issue = cls._possessor_surface_issue(
                candidate,
                evidence_span,
            )
            if possessor_surface_issue is not None:
                return None, possessor_surface_issue
        return candidate, None

    def _reasoner_candidate_for_current_dataset(
        self,
        artifact: str,
    ) -> tuple[Optional[str], Optional[str]]:
        minimum_evidence, minimum_steps = self._semantic_minimums()
        return self._reasoner_candidate(
            artifact,
            original_question=hotpotqa_question_scope(self._problem),
            minimum_evidence_propositions=minimum_evidence,
            minimum_reasoning_steps=minimum_steps,
        )

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
        *,
        require_answer_binding: bool = False,
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
        if not require_answer_binding:
            return None

        answer_slot = fields.get("answer_slot")
        candidate = fields.get("candidate_answer")
        if not isinstance(answer_slot, Mapping):
            return "Reasoner field 'answer_slot' must be one structured object"
        proposition_index = answer_slot.get("proposition_index")
        answer_field = answer_slot.get("answer_field")
        if (
            isinstance(proposition_index, bool)
            or not isinstance(proposition_index, int)
            or proposition_index < 0
            or proposition_index >= len(propositions)
        ):
            return (
                "Reasoner answer_slot.proposition_index is outside "
                "evidence_propositions"
            )
        selected = propositions[proposition_index]
        if not isinstance(selected, Mapping):
            return (
                f"Reasoner evidence_propositions[{proposition_index}] must be "
                "an object"
            )
        evidence_span = selected.get("evidence_span")
        if not isinstance(evidence_span, str):
            return (
                f"Reasoner evidence_propositions[{proposition_index}]."
                "evidence_span must be non-empty text"
            )
        if answer_field not in {"subject", "object_or_attribute_value"}:
            return (
                "Reasoner answer_slot.answer_field must be subject or "
                "object_or_attribute_value"
            )
        if not isinstance(candidate, str) or selected.get(answer_field) != candidate:
            return (
                "Reasoner candidate_answer must copy the selected proposition "
                "argument exactly"
            )

        canonical_span = _canonical_evidence_text(evidence_span)
        for field_name in ("subject", "object_or_attribute_value"):
            value = selected.get(field_name)
            if not isinstance(value, str) or not value.strip():
                return (
                    f"Reasoner answer-bearing proposition {field_name!r} must "
                    "be non-empty text"
                )
            # yes/no is inferred from a grounded relation rather than copied
            # from a passage.  The proposition's remaining entity argument is
            # still required to bind lexically to the successful read span.
            if value.casefold() in {"yes", "no"}:
                continue
            if _canonical_evidence_text(value) not in canonical_span:
                return (
                    "Reasoner answer-bearing proposition has no deterministic "
                    f"entity binding: field {field_name!r} value {value!r} does "
                    "not occur in its evidence_span from a successful "
                    "qa-retrieval read receipt"
                )
        return None

    @staticmethod
    def _reports_knowledge_base_coverage_failure(value: object) -> bool:
        """Recognize an explicit public coverage-failure artifact or receipt.

        This is an operational diagnosis emitted by retrieval execution.  It
        does not establish that the underlying corpus lacks the fact; it only
        records that the bounded configured retrieval/database path did not
        return admissible evidence.
        """

        if isinstance(value, str):
            normalized = value.strip().casefold()
            if "knowledge_base_coverage_failure" in normalized:
                return True
            if normalized.startswith(("{", "[")):
                try:
                    return AgentWorkflowEnv._reports_knowledge_base_coverage_failure(
                        json.loads(value)
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    return False
            return False
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = re.sub(r"[ -]+", "_", str(raw_key).strip().casefold())
                if key == "knowledge_base_coverage_failure" and item is True:
                    return True
                if key in {
                    "failure_category",
                    "failure_type",
                    "error_type",
                    "public_error_code",
                    "status",
                } and isinstance(item, str):
                    if item.strip().casefold() == "knowledge_base_coverage_failure":
                        return True
                if AgentWorkflowEnv._reports_knowledge_base_coverage_failure(item):
                    return True
            return False
        if isinstance(value, (list, tuple)):
            return any(
                AgentWorkflowEnv._reports_knowledge_base_coverage_failure(item)
                for item in value
            )
        return False

    def _semantic_protocol_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Return the shared evidence-grounded QA FINISH admission gate."""

        if not self._uses_semantic_lineage_protocol():
            return None
        if self._uses_role_conditional_capabilities():
            return self._hotpotqa_role_conditional_issue(execution)
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            return self._hotpotqa_semantic_lineage_issue(execution)
        protocol_label = self._semantic_protocol_label()
        structure_issue = self._format_agent_issue_for(self._graph)
        if structure_issue is not None:
            return structure_issue
        formatter_id = self._graph.output_agent_id
        if formatter_id is None:
            return (
                f"{protocol_label} semantic protocol requires a selected "
                "Format Agent"
            )
        verifier_id = self._graph.directed_predecessors(formatter_id)[0]
        reasoner_ids = tuple(
            agent_id
            for agent_id in self._graph.directed_predecessors(verifier_id)
            if (self._graph.get_node(agent_id).role_family or "").casefold()
            == "reasoner"
        )
        if len(reasoner_ids) != 1:
            return (
                f"{protocol_label} semantic protocol requires exactly one direct "
                "Reasoner "
                "semantic candidate for the Verifier"
            )
        reasoner_id = reasoner_ids[0]
        reasoner_artifact = execution.outputs.get(reasoner_id)
        verifier_artifact = execution.outputs.get(verifier_id)
        if reasoner_artifact is None:
            return f"Reasoner {reasoner_id!r} has no current semantic artifact"
        if verifier_artifact is None:
            return f"Verifier {verifier_id!r} has no current verification artifact"
        evidence_owner_ids = (
            reasoner_id,
            *self._graph.directed_predecessors(reasoner_id),
        )
        coverage_failure_agent_ids = tuple(
            agent_id
            for agent_id in evidence_owner_ids
            if self._reports_knowledge_base_coverage_failure(
                execution.outputs.get(agent_id)
            )
            or self._reports_knowledge_base_coverage_failure(
                execution.output_metadata.get(agent_id)
            )
        )
        if coverage_failure_agent_ids:
            return (
                "Reasoner lineage reported knowledge_base_coverage_failure from "
                f"Agents {list(coverage_failure_agent_ids)!r}. This is an "
                "operational retrieval/database coverage diagnosis for the "
                "configured bounded Tool path, not a corpus-level oracle claim. "
                "Preserve valid receipts and semantic artifacts, then repair or "
                "augment retrieval before FINISH; do not guess an answer or "
                "fabricate evidence"
            )
        reasoner_candidate, reasoner_issue = (
            self._reasoner_candidate_for_current_dataset(reasoner_artifact)
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
            require_answer_binding=(
                self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
            ),
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

    @classmethod
    def _semantic_candidate_from_artifact(
        cls,
        artifact: object,
    ) -> tuple[Optional[str], Optional[str]]:
        """Read an explicit semantic-candidate wire without inferring an answer."""

        if not isinstance(artifact, str) or not artifact.strip():
            return None, None
        verifier_candidate, verifier_issue = cls._verifier_candidate(artifact)
        if verifier_issue is None and verifier_candidate is not None:
            return verifier_candidate, None
        try:
            parsed = json.loads(artifact)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, Mapping):
            normalized = {
                re.sub(r"[ -]+", "_", str(raw_key).strip().casefold()): value
                for raw_key, value in parsed.items()
            }
            if len(normalized) != len(parsed):
                return None, "semantic candidate wire contains duplicate fields"
            candidate = normalized.get("candidate_answer")
            if (
                isinstance(candidate, str)
                and candidate
                and candidate == candidate.strip()
                and "\n" not in candidate
            ):
                return candidate, None
        matches = re.findall(
            r"(?im)^Candidate answer:\s*(\S(?:.*\S)?)\s*$",
            artifact,
        )
        if not matches:
            return None, None
        if len(matches) != 1 or "\n" in matches[0]:
            return None, "semantic candidate wire must contain one Candidate answer"
        return matches[0], None

    def _successful_read_texts_for_agents(
        self,
        execution: AgentRuntimeResult,
        agent_ids: Sequence[str],
    ) -> tuple[str, ...]:
        if self.required_evidence_tool_id is None:
            return ()
        texts: list[str] = []
        for agent_id in agent_ids:
            metadata = execution.output_metadata.get(agent_id)
            if not isinstance(metadata, Mapping):
                continue
            receipts = metadata.get("tool_receipts", ())
            if not isinstance(receipts, (list, tuple)):
                continue
            for receipt in receipts:
                if not isinstance(receipt, Mapping):
                    continue
                text = self._successful_read_text(
                    receipt,
                    self.required_evidence_tool_id,
                )
                if text is not None:
                    texts.append(text)
        return tuple(texts)

    def _hotpotqa_role_conditional_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Validate only semantic capabilities actually selected by the Canvas."""

        output_id = self._graph.output_agent_id
        if output_id is None or not self._graph.has_node(output_id):
            return "HotpotQA role-conditional workflow has no selected Output Agent"
        routed_ids = (
            *self._directed_ancestor_ids(self._graph, output_id),
            output_id,
        )
        read_texts = self._successful_read_texts_for_agents(
            execution,
            routed_ids,
        )
        if not read_texts:
            return (
                "HotpotQA routed Output path has no successful qa-retrieval read "
                "receipt containing a non-empty passage"
            )

        routed_semantic_candidates: dict[str, str] = {}
        for agent_id in routed_ids:
            node = self._graph.get_node(agent_id)
            role = (node.role_family or "").casefold()
            artifact = execution.outputs.get(agent_id)
            if role == "reasoner" and agent_id != output_id:
                candidate, issue = self._reasoner_candidate_for_current_dataset(
                    artifact or ""
                )
                if issue is not None or candidate is None:
                    return (
                        f"Reasoner {agent_id!r} semantic artifact is invalid: {issue}"
                    )
                evidence_owner_ids = (
                    *self._directed_ancestor_ids(self._graph, agent_id),
                    agent_id,
                )
                owner_texts = self._successful_read_texts_for_agents(
                    execution,
                    evidence_owner_ids,
                )
                if not owner_texts:
                    return (
                        f"Reasoner {agent_id!r} has no routed successful "
                        "qa-retrieval read receipt"
                    )
                provenance_issue = self._reasoner_evidence_provenance_issue(
                    artifact or "",
                    owner_texts,
                    require_answer_binding=False,
                )
                if provenance_issue is not None:
                    return (
                        f"Reasoner {agent_id!r} evidence provenance is invalid: "
                        f"{provenance_issue}"
                    )
                routed_semantic_candidates[agent_id] = candidate
            elif role == "verifier" and agent_id != output_id:
                candidate, issue = self._verifier_candidate(artifact or "")
                if issue is not None or candidate is None:
                    return (
                        f"Verifier {agent_id!r} semantic artifact is invalid: {issue}"
                    )
                verifier_component = set(
                    next(
                        (
                            component
                            for component in self._graph.validate(
                                self.model_registry,
                                require_complete=False,
                            ).components
                            if agent_id in component
                        ),
                        (agent_id,),
                    )
                )
                ancestor_ids = tuple(
                    upstream_id
                    for upstream_id in self._directed_ancestor_ids(
                        self._graph,
                        agent_id,
                    )
                    if upstream_id != agent_id
                )
                upstream_candidates = tuple(
                    upstream_candidate
                    for upstream_id in ancestor_ids
                    for upstream_candidate, upstream_issue in (
                        self._semantic_candidate_from_artifact(
                            execution.outputs.get(upstream_id, "")
                        ),
                    )
                    if upstream_issue is None
                    and upstream_candidate is not None
                )
                producer_candidates = tuple(
                    upstream_candidate
                    for upstream_id in ancestor_ids
                    if (
                        self._graph.get_node(upstream_id).role_family or ""
                    ).casefold()
                    != "verifier"
                    for upstream_candidate, upstream_issue in (
                        self._semantic_candidate_from_artifact(
                            execution.outputs.get(upstream_id, "")
                        ),
                    )
                    if upstream_issue is None
                    and upstream_candidate is not None
                )
                if not producer_candidates:
                    return (
                        f"Verifier {agent_id!r} has no routed semantic candidate "
                        "from a non-Verifier producer; a Verifier or reciprocal "
                        f"Verifier block {sorted(verifier_component)!r} checks and "
                        "preserves a candidate but must not bootstrap or select one"
                    )
                verifier_evidence_owner_ids = (
                    *self._directed_ancestor_ids(self._graph, agent_id),
                    agent_id,
                )
                if not self._successful_read_texts_for_agents(
                    execution,
                    verifier_evidence_owner_ids,
                ):
                    return (
                        f"Verifier {agent_id!r} has no routed successful "
                        "qa-retrieval read receipt for checking its semantic "
                        "candidate"
                    )
                if upstream_candidates and any(
                    upstream_candidate != candidate
                    for upstream_candidate in upstream_candidates
                ):
                    return (
                        "Verifier changed a routed semantic candidate_answer: "
                        f"verifier={candidate!r}, "
                        f"upstream_candidates={list(upstream_candidates)!r}"
                    )
                routed_semantic_candidates[agent_id] = candidate
            elif role not in {"evidence_retriever", "format", "output"}:
                candidate, issue = self._semantic_candidate_from_artifact(
                    artifact or ""
                )
                if issue is not None:
                    return (
                        f"Semantic producer {agent_id!r} artifact is invalid: "
                        f"{issue}"
                    )
                if candidate is not None:
                    evidence_owner_ids = (
                        *self._directed_ancestor_ids(self._graph, agent_id),
                        agent_id,
                    )
                    if not self._successful_read_texts_for_agents(
                        execution,
                        evidence_owner_ids,
                    ):
                        return (
                            f"Semantic producer {agent_id!r} has no routed "
                            "successful qa-retrieval read receipt"
                        )
                    routed_semantic_candidates[agent_id] = candidate

        output_node = self._graph.get_node(output_id)
        if (output_node.role_family or "").casefold() != "format":
            routed_candidates = tuple(routed_semantic_candidates.values())
            if not routed_candidates:
                return None
            candidate = routed_candidates[0]
            if any(item != candidate for item in routed_candidates):
                return (
                    "Generic Output Agent received disagreeing routed semantic "
                    "candidates: "
                    f"{list(dict.fromkeys(routed_candidates))!r}"
                )
            wrapper = re.fullmatch(
                r"\s*<answer>(.*?)</answer>\s*",
                execution.final_answer or "",
                flags=re.DOTALL,
            )
            if wrapper is None or wrapper.group(1) != candidate:
                output_value = None if wrapper is None else wrapper.group(1)
                return (
                    "Generic Output Agent must preserve the routed semantic "
                    "candidate character-for-character: "
                    f"candidate_answer={candidate!r}, "
                    f"wrapper_content={output_value!r}"
                )
            return None
        direct_candidates: list[str] = []
        for predecessor_id in self._graph.directed_predecessors(output_id):
            candidate, issue = self._semantic_candidate_from_artifact(
                execution.outputs.get(predecessor_id)
            )
            if issue is not None:
                return issue
            if candidate is not None:
                direct_candidates.append(candidate)
        if not direct_candidates:
            return (
                "Format Agent has no routed upstream artifact with one explicit "
                "semantic candidate"
            )
        candidate = direct_candidates[0]
        if any(item != candidate for item in direct_candidates):
            return (
                "Format Agent received disagreeing routed semantic candidates: "
                f"{list(dict.fromkeys(direct_candidates))!r}"
            )
        wrapper = re.fullmatch(
            r"\s*<answer>(.*?)</answer>\s*",
            execution.final_answer or "",
            flags=re.DOTALL,
        )
        if wrapper is None or wrapper.group(1) != candidate:
            formatter_value = None if wrapper is None else wrapper.group(1)
            return (
                "Formatter must only wrap the routed semantic candidate "
                "character-for-character without reselecting or reasoning: "
                f"candidate_answer={candidate!r}, "
                f"wrapper_content={formatter_value!r}"
            )
        return None

    def _hotpotqa_semantic_lineages(
        self,
        execution: AgentRuntimeResult,
    ) -> tuple[Tuple[Tuple[str, Tuple[str, ...]], ...], Tuple[str, ...]]:
        """Resolve valid routed Reasoner--Verifier artifact lineages.

        This is an executed-artifact query, not a topology template.  A lineage
        may contain retrieval, repair, intermediate reasoning, fan-in, or one
        bounded reciprocal block.  Its endpoints are recognized by semantic
        role and its validity comes from the actual artifacts and Tool receipts.
        """

        formatter_id = self._graph.output_agent_id
        if formatter_id is None or not self._graph.has_node(formatter_id):
            return (), ("HotpotQA semantic lineage has no selected Output Agent",)
        output_ancestors = set(
            self._directed_ancestor_ids(self._graph, formatter_id)
        )
        verifier_ids = tuple(
            sorted(
                (
                    node.id
                    for node in self._graph.nodes
                    if node.id in output_ancestors
                    and (node.role_family or "").casefold() == "verifier"
                ),
                key=lambda verifier_id: (
                    len(
                        self._directed_shortest_path(
                            self._graph,
                            verifier_id,
                            formatter_id,
                        )
                    ),
                    verifier_id,
                ),
            )
        )
        if not verifier_ids:
            return (), (
                "HotpotQA Output has no routed Verifier artifact in its directed lineage",
            )

        lineages: list[Tuple[str, Tuple[str, ...]]] = []
        diagnostics: list[str] = []
        for verifier_id in verifier_ids:
            verifier_artifact = execution.outputs.get(verifier_id)
            if verifier_artifact is None:
                diagnostics.append(
                    f"Verifier {verifier_id!r} has no current verification artifact"
                )
                continue
            verifier_candidate, verifier_issue = self._verifier_candidate(
                verifier_artifact
            )
            if verifier_issue is not None or verifier_candidate is None:
                diagnostics.append(
                    f"Verifier {verifier_id!r} semantic artifact is invalid: "
                    f"{verifier_issue}"
                )
                continue
            reasoner_ancestors = set(
                self._directed_ancestor_ids(self._graph, verifier_id)
            )
            reasoner_ids = tuple(
                sorted(
                    (
                        node.id
                        for node in self._graph.nodes
                        if node.id in reasoner_ancestors
                        and (node.role_family or "").casefold() == "reasoner"
                    ),
                    key=lambda reasoner_id: (
                        len(
                            self._directed_shortest_path(
                                self._graph,
                                reasoner_id,
                                verifier_id,
                            )
                        ),
                        reasoner_id,
                    ),
                )
            )
            if not reasoner_ids:
                diagnostics.append(
                    f"Verifier {verifier_id!r} has no routed Reasoner semantic artifact"
                )
                continue
            for reasoner_id in reasoner_ids:
                reasoner_artifact = execution.outputs.get(reasoner_id)
                if reasoner_artifact is None:
                    diagnostics.append(
                        f"Reasoner {reasoner_id!r} has no current semantic artifact"
                    )
                    continue
                reasoner_candidate, reasoner_issue = (
                    self._reasoner_candidate_for_current_dataset(reasoner_artifact)
                )
                if reasoner_issue is not None or reasoner_candidate is None:
                    diagnostics.append(
                        f"Reasoner {reasoner_id!r} semantic artifact is invalid: "
                        f"{reasoner_issue}"
                    )
                    continue
                evidence_owner_ids = (
                    reasoner_id,
                    *self._directed_ancestor_ids(self._graph, reasoner_id),
                )
                coverage_failure_ids = tuple(
                    owner_id
                    for owner_id in evidence_owner_ids
                    if self._reports_knowledge_base_coverage_failure(
                        execution.outputs.get(owner_id)
                    )
                    or self._reports_knowledge_base_coverage_failure(
                        execution.output_metadata.get(owner_id)
                    )
                )
                if coverage_failure_ids:
                    diagnostics.append(
                        "Reasoner lineage reported "
                        "knowledge_base_coverage_failure from Agents "
                        f"{list(coverage_failure_ids)!r}; preserve valid receipts "
                        "and repair or augment retrieval before FINISH"
                    )
                    continue
                read_evidence_texts: list[str] = []
                assert self.required_evidence_tool_id is not None
                for owner_id in evidence_owner_ids:
                    metadata = execution.output_metadata.get(owner_id)
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
                    diagnostics.append(
                        f"Reasoner {reasoner_id!r} lineage has no successful "
                        f"{self.required_evidence_tool_id!r} read receipt"
                    )
                    continue
                provenance_issue = self._reasoner_evidence_provenance_issue(
                    reasoner_artifact,
                    read_evidence_texts,
                    require_answer_binding=True,
                )
                if provenance_issue is not None:
                    diagnostics.append(
                        f"Reasoner {reasoner_id!r} evidence provenance is invalid: "
                        f"{provenance_issue}"
                    )
                    continue
                if verifier_candidate != reasoner_candidate:
                    diagnostics.append(
                        "Verifier changed the Reasoner's candidate_answer: "
                        f"verifier_id={verifier_id!r}, reasoner_id={reasoner_id!r}, "
                        f"reasoner={reasoner_candidate!r}, "
                        f"verifier={verifier_candidate!r}"
                    )
                    continue
                reasoner_to_verifier_paths = self._directed_simple_paths(
                    self._graph,
                    reasoner_id,
                    verifier_id,
                )
                verifier_to_formatter_paths = self._directed_simple_paths(
                    self._graph,
                    verifier_id,
                    formatter_id,
                )
                if (
                    not reasoner_to_verifier_paths
                    or not verifier_to_formatter_paths
                ):
                    diagnostics.append(
                        f"Reasoner {reasoner_id!r} and Verifier {verifier_id!r} "
                        "do not form one routed path to the selected Formatter"
                    )
                    continue
                continuity_issues: list[str] = []
                accepted_path: Optional[Tuple[str, ...]] = None
                for reasoner_to_verifier in reasoner_to_verifier_paths:
                    for verifier_to_formatter in verifier_to_formatter_paths:
                        path = (
                            *reasoner_to_verifier,
                            *verifier_to_formatter[1:],
                        )
                        continuity_issue = (
                            self._semantic_artifact_continuity_issue(
                                execution,
                                path,
                                reasoner_candidate,
                                semantic_endpoint_ids={
                                    reasoner_id,
                                    verifier_id,
                                    formatter_id,
                                },
                            )
                        )
                        if continuity_issue is None:
                            accepted_path = path
                            break
                        continuity_issues.append(continuity_issue)
                    if accepted_path is not None:
                        break
                if accepted_path is None:
                    diagnostics.append(
                        continuity_issues[0]
                        if continuity_issues
                        else (
                            f"Reasoner {reasoner_id!r} semantic artifact did not "
                            "reach Verifier and Formatter through a continuous "
                            "CommunicationEnvelope path"
                        )
                    )
                    continue
                lineages.append((reasoner_candidate, accepted_path))
        return tuple(lineages), tuple(diagnostics)

    def _hotpotqa_semantic_lineage_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Validate HotpotQA semantics over the actually executed free graph."""

        structure_issue = self._format_agent_issue_for(self._graph)
        if structure_issue is not None:
            return structure_issue
        lineages, diagnostics = self._hotpotqa_semantic_lineages(execution)
        if not lineages:
            if diagnostics:
                return diagnostics[0]
            return "HotpotQA has no valid routed Reasoner--Verifier semantic lineage"
        candidates = tuple(dict.fromkeys(candidate for candidate, _ in lineages))
        if len(candidates) != 1:
            return (
                "HotpotQA routed semantic lineages disagree on candidate_answer: "
                f"{list(candidates)!r}; preserve evidence and diagnose entity binding, "
                "question scope, and relation routing before FINISH"
            )
        candidate = candidates[0]
        answer = execution.final_answer
        if answer is None:
            return "Format Agent produced no terminal wrapper"
        wrapper = re.fullmatch(
            r"\s*<answer>(.*?)</answer>\s*",
            answer,
            flags=re.DOTALL,
        )
        if wrapper is None or wrapper.group(1) != candidate:
            formatter_value = None if wrapper is None else wrapper.group(1)
            return (
                "Formatter must only wrap the supported semantic candidate "
                "character-for-character without reselecting or reasoning: "
                f"candidate_answer={candidate!r}, "
                f"wrapper_content={formatter_value!r}"
            )
        return None

    @staticmethod
    def _directed_ancestor_ids(
        graph: AgentGraph,
        agent_id: str,
    ) -> Tuple[str, ...]:
        """Return all routed ancestors in deterministic graph order."""

        pending = list(graph.directed_predecessors(agent_id))
        seen: set[str] = set()
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            pending.extend(graph.directed_predecessors(current))
        return tuple(node.id for node in graph.nodes if node.id in seen)

    @classmethod
    def _directed_shortest_path(
        cls,
        graph: AgentGraph,
        source_id: str,
        target_id: str,
    ) -> Tuple[str, ...]:
        """Return one deterministic routed path, including both endpoints."""

        if source_id == target_id:
            return (source_id,)
        pending: list[Tuple[str, ...]] = [(source_id,)]
        seen = {source_id}
        while pending:
            path = pending.pop(0)
            for successor_id in cls._directed_successors(graph, path[-1]):
                if successor_id == target_id:
                    return (*path, successor_id)
                if successor_id in seen:
                    continue
                seen.add(successor_id)
                pending.append((*path, successor_id))
        return ()

    @classmethod
    def _directed_simple_paths(
        cls,
        graph: AgentGraph,
        source_id: str,
        target_id: str,
    ) -> Tuple[Tuple[str, ...], ...]:
        """Return every deterministic simple directed path between two Agents."""

        if source_id == target_id:
            return ((source_id,),)
        paths: list[Tuple[str, ...]] = []
        pending: list[Tuple[str, ...]] = [(source_id,)]
        while pending:
            path = pending.pop(0)
            for successor_id in cls._directed_successors(graph, path[-1]):
                if successor_id in path:
                    continue
                candidate = (*path, successor_id)
                if successor_id == target_id:
                    paths.append(candidate)
                else:
                    pending.append(candidate)
        return tuple(paths)

    def _semantic_artifact_continuity_issue(
        self,
        execution: AgentRuntimeResult,
        path: Tuple[str, ...],
        candidate_answer: str,
        *,
        semantic_endpoint_ids: set[str],
    ) -> Optional[str]:
        """Validate candidate preservation across routed communication.

        FlowSteer's directed path records topology, while AgentRuntime passes
        only each direct predecessor's artifact in a CommunicationEnvelope.
        Every intermediate Agent on a semantic path must therefore preserve
        the Reasoner's exact candidate in its public artifact. This accepts
        arbitrary bridges, fan-in, repair nodes, and reciprocal blocks, but
        rejects a graph path whose actual messages have lost semantic lineage.
        """

        for agent_id in path[1:-1]:
            if agent_id in semantic_endpoint_ids:
                continue
            artifact = execution.outputs.get(agent_id)
            if not isinstance(artifact, str) or not self._contains_lexical_span(
                artifact,
                candidate_answer,
            ):
                return (
                    "HotpotQA semantic artifact continuity failed: intermediate "
                    f"Agent {agent_id!r} did not preserve candidate_answer "
                    f"{candidate_answer!r} in its routed artifact"
                )

        # When the current runtime result contains the target invocation, also
        # verify that its request receipt carried the exact predecessor output.
        # Reused Agents may have no current-revision call record; their public
        # output continuity is still checked above.
        for source_id, target_id in zip(path, path[1:]):
            target_calls = tuple(
                call
                for call in execution.calls
                if call.request.agent.id == target_id
            )
            if not target_calls:
                continue
            source_artifact = execution.outputs.get(source_id)
            envelope_seen = any(
                any(
                    message.source_agent_id == source_id
                    and message.target_agent_id == target_id
                    and message.content == source_artifact
                    for message in call.request.upstream
                )
                or (
                    call.request.peer_draft is not None
                    and call.request.peer_draft.source_agent_id == source_id
                    and call.request.peer_draft.target_agent_id == target_id
                    and call.request.peer_draft.content == source_artifact
                )
                for call in target_calls
            )
            if not envelope_seen:
                return (
                    "HotpotQA semantic artifact continuity failed: no exact "
                    f"CommunicationEnvelope carried {source_id!r} to "
                    f"{target_id!r} in the current Runtime receipt"
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
        metadata = self._progressive_output_metadata.get(agent_id, {})
        return (
            isinstance(artifact, str)
            and bool(artifact.strip())
            and agent_id not in self._unresolved_dirty_agents
            and metadata.get("artifact_status")
            != _NON_TERMINAL_PARTIAL_ARTIFACT
        )

    @staticmethod
    def _completed_partial_output_agent_ids(
        execution: AgentRuntimeResult,
    ) -> set[str]:
        """Exclude incomplete reciprocal peer artifacts from completion."""

        return {
            agent_id
            for agent_id in execution.outputs
            if execution.output_metadata.get(agent_id, {}).get(
                "artifact_status"
            )
            != _NON_TERMINAL_PARTIAL_ARTIFACT
        }

    def _preserved_input_change_issue_for(
        self,
        candidate: AgentGraph,
        *,
        terminal_convergence_output_id: Optional[str] = None,
    ) -> Optional[str]:
        """Protect the dependency identity of revision-live successful artifacts."""

        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return None
        evidence_ingress_candidate = (
            self._role_conditional_evidence_ingress_candidate(candidate)
            or self._role_conditional_existing_evidence_ingress_candidate(
                candidate
            )
        )
        repairable_ids = self._failed_agent_ids | self._unresolved_dirty_agents
        for node in self._graph.nodes:
            if (
                not self._has_successful_artifact(node.id)
                or node.id in repairable_ids
                or not candidate.has_node(node.id)
            ):
                continue
            before = tuple(self._graph.directed_predecessors(node.id))
            after = tuple(candidate.directed_predecessors(node.id))
            if before != after:
                if evidence_ingress_candidate and node.id in set(
                    self._role_conditional_evidence_ingress_consumer_ids()
                ):
                    continue
                added_predecessors = set(after) - set(before)
                if (
                    terminal_convergence_output_id == node.id
                    and (node.role_family or "").casefold()
                    in {"format", "output"}
                    and set(before) < set(after)
                    and len(added_predecessors) == 1
                    and all(
                        self._has_successful_artifact(source_id)
                        and source_id not in self._failed_agent_ids
                        and source_id not in self._repair_exhausted_agent_ids
                        for source_id in added_predecessors
                    )
                ):
                    # This narrow exception is consumed only by
                    # `_prospective_terminal_convergence_relation_candidates`,
                    # which subsequently requires the same prospective
                    # SET_OUTPUT graph to pass complete and semantic admission.
                    # Existing dependencies are never removed or reversed.
                    continue
                return (
                    f"preserve successful Agent {node.id!r} input dependencies; "
                    f"current_predecessors={list(before)!r}, "
                    f"candidate_predecessors={list(after)!r}. Route its existing "
                    "artifact downstream instead of invalidating it, unless that "
                    "Agent is the measured repair target"
                )
        return None

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
        non_terminal_partial = tuple(
            sorted(
                agent_id
                for agent_id in current_ids
                if self._progressive_output_metadata.get(agent_id, {}).get(
                    "artifact_status"
                )
                == _NON_TERMINAL_PARTIAL_ARTIFACT
            )
        )
        previous_preserved = tuple(
            sorted(
                agent_id
                for agent_id in self._previous_revision_outputs
                if self._previous_revision_output_metadata.get(
                    agent_id,
                    {},
                ).get("artifact_status")
                != _NON_TERMINAL_PARTIAL_ARTIFACT
            )
        )
        previous_non_terminal_partial = tuple(
            sorted(
                agent_id
                for agent_id in self._previous_revision_outputs
                if self._previous_revision_output_metadata.get(
                    agent_id,
                    {},
                ).get("artifact_status")
                == _NON_TERMINAL_PARTIAL_ARTIFACT
            )
        )
        terminal_unreachable = self._terminal_unreachable_agent_ids()
        terminal_unreachable_set = set(terminal_unreachable)
        failed = tuple(sorted(self._failed_agent_ids & current_ids))
        react_exhausted = tuple(
            sorted(self._react_exhausted_agent_ids & current_ids)
        )
        repair_exhausted = tuple(
            sorted(self._repair_exhausted_agent_ids & current_ids)
        )
        mandatory_repair = self._mandatory_repair_agent_ids()
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
        repair_routing_candidates = self._repair_exhausted_relation_candidates()
        auxiliary_takeover_candidates = (
            self._repair_exhausted_auxiliary_takeover_relation_candidates()
        )
        selected_output_recovery_candidates = (
            self._selected_output_artifact_recovery_relation_candidates()
            if AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
            else []
        )
        failed_ingress_relation_candidates = (
            self._failed_auxiliary_ingress_relation_candidates()
        )
        terminal_reachability_relation_candidates = (
            self._terminal_reachability_relation_candidates()
        )
        takeover_delete_ids = (
            self._repair_exhausted_auxiliary_takeover_delete_ids()
        )
        required_relation_candidates = (
            self._required_semantic_relation_candidates()
        )
        output_target_ids = self._model_admissible_output_agent_ids()
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
                "augment"
                if repair_exhausted
                else "diagnose_repair"
                if (
                    self._unresolved_dirty_agents
                    or self._failed_agent_ids
                    or terminal_unreachable
                )
                else "preserve"
            ),
            "preserved_agent_ids": list(preserved),
            "non_terminal_partial_agent_ids": list(non_terminal_partial),
            "previous_revision_preserved_agent_ids": list(
                previous_preserved
            ),
            "previous_revision_non_terminal_partial_agent_ids": list(
                previous_non_terminal_partial
            ),
            "failed_agent_ids": list(failed),
            "react_turn_exhausted_agent_ids": list(react_exhausted),
            "repair_exhausted_agent_ids": list(repair_exhausted),
            "mandatory_repair_agent_ids": list(mandatory_repair),
            "diagnosed_unusable_agent_ids": list(diagnosed_unusable),
            "unresolved_dirty_agent_ids": list(self.unresolved_dirty_agent_ids),
            "terminal_unreachable_agent_ids": list(terminal_unreachable),
            "active_semantic_lineage_agent_ids": list(active_semantic_lineage),
            "redundant_after_replacement_takeover_agent_ids": list(
                redundant_after_takeover
            ),
            "deletable_agent_ids": list(deletable),
            "repair_exhausted_auxiliary_takeover_delete_agent_ids": list(
                takeover_delete_ids
            ),
            "selected_output_artifact_recovery_relation_candidates": [
                dict(candidate)
                for candidate in selected_output_recovery_candidates
            ],
            "repair_exhausted_auxiliary_takeover_relation_candidates": [
                dict(candidate)
                for candidate in auxiliary_takeover_candidates
            ],
            "deletion_protected": protected,
            "preferred_actions": (
                ["set_relation"]
                if selected_output_recovery_candidates
                else ["set_relation"]
                if auxiliary_takeover_candidates
                else ["modify_agent"]
                if (
                    mandatory_repair
                    and AgentActionType.MODIFY_AGENT.value
                    in self._allowed_action_type_set
                )
                else []
                if mandatory_repair
                else ["delete_agent"]
                if takeover_delete_ids
                else ["set_relation"]
                if failed_ingress_relation_candidates
                else ["set_relation"]
                if repair_routing_candidates
                else ["set_relation"]
                if terminal_reachability_relation_candidates
                else ["set_relation"]
                if required_relation_candidates
                else ["add_subgraph"]
                if (
                    repair_exhausted
                    and self._graph.nodes
                    and self._missing_semantic_role_families()
                    and (
                        self.max_agents is None
                        or len(self._graph.nodes) < self.max_agents
                    )
                )
                else ["set_output"]
                if self._uses_semantic_lineage_protocol() and output_target_ids
                else ["add_subgraph"]
                if (
                    repair_exhausted
                    and self._model_admissible_add_role_families()
                    and (
                        self.max_agents is None
                        or len(self._graph.nodes) < self.max_agents
                    )
                )
                else ["set_relation", "modify_agent", "add_subgraph"]
                if repair_exhausted
                else ["delete_agent", "set_relation", "modify_agent"]
                if deletable
                else ["modify_agent", "set_relation", "add_subgraph"]
            ),
        }

    def _active_semantic_lineage_ids(self) -> Tuple[str, ...]:
        """Return IDs in the current terminal-admissible semantic lineage."""

        if not self._uses_semantic_lineage_protocol():
            return ()
        if self._uses_role_conditional_capabilities():
            execution = self._cached_progressive_execution()
            output_id = self._graph.output_agent_id
            if (
                execution is None
                or execution.final_answer is None
                or output_id is None
                or self._hotpotqa_role_conditional_issue(execution) is not None
                or self._terminal_validation_error(execution.final_answer)
                is not None
            ):
                return ()
            routed = (
                *self._directed_ancestor_ids(self._graph, output_id),
                output_id,
            )
            return tuple(
                agent_id
                for agent_id in routed
                if self._has_successful_artifact(agent_id)
            )
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            execution = self._cached_progressive_execution()
            if execution is None or execution.final_answer is None:
                return ()
            lineages, _ = self._hotpotqa_semantic_lineages(execution)
            candidates = tuple(
                dict.fromkeys(candidate for candidate, _ in lineages)
            )
            if len(candidates) != 1:
                return ()
            wrapper = re.fullmatch(
                r"\s*<answer>(.*?)</answer>\s*",
                execution.final_answer,
                flags=re.DOTALL,
            )
            if wrapper is None or wrapper.group(1) != candidates[0]:
                return ()
            return lineages[0][1]
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

        if not self._uses_semantic_lineage_protocol():
            return True
        if self._uses_role_conditional_capabilities():
            artifact = self._progressive_outputs.get(agent_id)
            if not isinstance(artifact, str) or not artifact.strip():
                return False
            if role_family == "reasoner":
                candidate, issue = self._reasoner_candidate_for_current_dataset(
                    artifact
                )
                return issue is None and candidate is not None
            if role_family == "verifier":
                candidate, issue = self._verifier_candidate(artifact)
                return issue is None and candidate is not None
            if role_family == "format":
                return agent_id in self._active_semantic_lineage_ids()
            return True
        if self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL:
            return self._hotpotqa_semantic_replacement_has_valid_artifact(
                agent_id,
                role_family,
            )
        artifact = self._progressive_outputs.get(agent_id)
        if not isinstance(artifact, str) or not artifact.strip():
            return False
        if role_family == "reasoner":
            candidate, issue = self._reasoner_candidate_for_current_dataset(
                artifact
            )
            if issue is not None or candidate is None:
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
            return bool(evidence_texts) and self._reasoner_evidence_provenance_issue(
                artifact,
                evidence_texts,
                require_answer_binding=(
                    self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
                ),
            ) is None
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
            reasoner_candidate, reasoner_issue = (
                self._reasoner_candidate_for_current_dataset(reasoner_artifact)
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

    def _hotpotqa_semantic_replacement_has_valid_artifact(
        self,
        agent_id: str,
        role_family: str,
    ) -> bool:
        """Validate replacement takeover against routed v2 artifacts."""

        artifact = self._progressive_outputs.get(agent_id)
        if not isinstance(artifact, str) or not artifact.strip():
            return False
        if role_family == "reasoner":
            candidate, issue = self._reasoner_candidate_for_current_dataset(
                artifact
            )
            if issue is not None or candidate is None:
                return False
            evidence_texts: list[str] = []
            assert self.required_evidence_tool_id is not None
            for owner_id in (
                agent_id,
                *self._directed_ancestor_ids(self._graph, agent_id),
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
            return bool(evidence_texts) and self._reasoner_evidence_provenance_issue(
                artifact,
                evidence_texts,
                require_answer_binding=True,
            ) is None
        if role_family == "verifier":
            verifier_candidate, verifier_issue = self._verifier_candidate(artifact)
            if verifier_issue is not None or verifier_candidate is None:
                return False
            return any(
                reasoner_issue is None
                and reasoner_candidate == verifier_candidate
                for reasoner_id in self._directed_ancestor_ids(
                    self._graph,
                    agent_id,
                )
                if (
                    self._graph.get_node(reasoner_id).role_family or ""
                ).casefold()
                == "reasoner"
                for reasoner_candidate, reasoner_issue in (
                    self._reasoner_candidate_for_current_dataset(
                        self._progressive_outputs.get(reasoner_id, "")
                    ),
                )
            )
        if role_family == "format":
            return agent_id in self._active_semantic_lineage_ids()
        return True

    def _successful_evidence_texts_from_metadata(
        self,
        metadata: object,
    ) -> Tuple[str, ...]:
        """Return successful public read evidence from one Runtime receipt."""

        if self.required_evidence_tool_id is None or not isinstance(
            metadata,
            Mapping,
        ):
            return ()
        receipts = metadata.get("tool_receipts", ())
        if not isinstance(receipts, (list, tuple)):
            return ()
        texts: list[str] = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            text = self._successful_read_text(
                receipt,
                self.required_evidence_tool_id,
            )
            if text is not None:
                texts.append(text)
        return tuple(dict.fromkeys(texts))

    def _protected_previous_artifact(self, agent_id: str) -> Optional[str]:
        """Return the last complete artifact invalidated by a repair edit."""

        artifact = self._previous_revision_outputs.get(agent_id)
        metadata = self._previous_revision_output_metadata.get(agent_id, {})
        if (
            not isinstance(artifact, str)
            or not artifact.strip()
            or metadata.get("artifact_status")
            == _NON_TERMINAL_PARTIAL_ARTIFACT
        ):
            return None
        return artifact

    def _replacement_preserves_protected_history(
        self,
        source_agent_id: str,
        replacement_agent_id: str,
    ) -> bool:
        """Require replacement takeover to retain prior answer/evidence state."""

        previous_artifact = self._protected_previous_artifact(source_agent_id)
        source_evidence = tuple(
            dict.fromkeys(
                (
                    *self._successful_evidence_texts_from_metadata(
                        self._previous_revision_output_metadata.get(
                            source_agent_id,
                            {},
                        )
                    ),
                    *self._successful_evidence_texts_from_metadata(
                        self._failure_continuations.get(source_agent_id, {})
                    ),
                )
            )
        )
        if previous_artifact is None and not source_evidence:
            return True

        replacement_artifact = self._progressive_outputs.get(
            replacement_agent_id
        )
        if not isinstance(replacement_artifact, str) or not replacement_artifact.strip():
            return False
        if previous_artifact is not None:
            previous_candidate, _ = self._semantic_candidate_from_artifact(
                previous_artifact
            )
            replacement_candidate, _ = self._semantic_candidate_from_artifact(
                replacement_artifact
            )
            if previous_candidate is not None:
                if replacement_candidate != previous_candidate:
                    return False
            elif not source_evidence and (
                replacement_artifact.strip() != previous_artifact.strip()
            ):
                return False

        replacement_owner_ids = (
            replacement_agent_id,
            *self._directed_ancestor_ids(self._graph, replacement_agent_id),
        )
        replacement_evidence = tuple(
            dict.fromkeys(
                text
                for owner_id in replacement_owner_ids
                for text in self._successful_evidence_texts_from_metadata(
                    self._progressive_output_metadata.get(owner_id, {})
                )
            )
        )
        replacement_metadata = self._progressive_output_metadata.get(
            replacement_agent_id,
            {},
        )
        explicit_handoff = (
            isinstance(replacement_metadata, Mapping)
            and replacement_metadata.get("continuation_source_agent_id")
            == source_agent_id
        )
        return (
            not source_evidence
            or explicit_handoff
            or set(source_evidence) <= set(replacement_evidence)
        )

    def _delete_admission_issue(self, agent_id: Optional[str]) -> Optional[str]:
        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return None
        if agent_id is None or not self._graph.has_node(agent_id):
            return None
        node = self._graph.get_node(agent_id)
        terminal_unreachable_ids = set(self._terminal_unreachable_agent_ids())
        # Topological disconnection, provider failure, Tool failure, and
        # bounded ReAct exhaustion are repair diagnoses, not evidence that the
        # node itself is unusable.  Deletion requires the adapter's explicit
        # unusable diagnosis plus a same-responsibility replacement takeover.
        diagnosed_unusable = agent_id in self._diagnosed_unusable_agent_ids
        downstream_ids = set(self._directed_successors(self._graph, agent_id))
        protected_reasons: list[str] = []
        if self._has_successful_artifact(agent_id):
            protected_reasons.append("successful artifact/evidence")
        if self._protected_previous_artifact(agent_id) is not None:
            protected_reasons.append("previous revision artifact")
        if self._successful_evidence_texts_from_metadata(
            self._failure_continuations.get(agent_id, {})
        ):
            protected_reasons.append("successful continuation evidence")
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
                or not self._replacement_preserves_protected_history(
                    agent_id,
                    candidate.id,
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
        if (
            diagnosed_unusable
            and replacements
            and not self._has_successful_artifact(agent_id)
        ):
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
            not self._uses_semantic_lineage_protocol()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            or action.action_type is AgentActionType.FINISH
        ):
            return None
        evidence_ingress_issue = (
            self._role_conditional_evidence_ingress_admission_issue(action)
        )
        if evidence_ingress_issue is not None:
            return evidence_ingress_issue
        ingress_issue = self._role_conditional_ingress_admission_issue(action)
        if ingress_issue is not None:
            return ingress_issue
        selected_output_recovery_candidates = (
            self._selected_output_artifact_recovery_relation_candidates()
        )
        if (
            selected_output_recovery_candidates
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in selected_output_recovery_candidates
            ):
                return None
            return (
                "route the revision-live receipt-grounded artifact into the "
                "failed selected-Output ancestor with one exact admitted "
                "set_relation action before other Canvas edits"
            )
        auxiliary_takeover_candidates = (
            self._repair_exhausted_auxiliary_takeover_relation_candidates()
        )
        if (
            auxiliary_takeover_candidates
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in auxiliary_takeover_candidates
            ):
                return None
            return (
                "route one measured same-profile replacement artifact into "
                "one existing downstream responsibility with the exact "
                "admitted set_relation action before further augmentation"
            )
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
        provider_repair_issue = self._provider_repair_admission_issue(action)
        if provider_repair_issue is not None:
            return provider_repair_issue
        react_repair_issue = self._react_repair_admission_issue(action)
        if react_repair_issue is not None:
            return react_repair_issue
        if mandatory_repair_ids:
            # The live action mask has already reduced this turn to the exact
            # measured repair target, and the provider/ReAct field contract
            # above is authoritative.  A still-incomplete semantic spine must
            # not reject that same repair after constrained decoding selected
            # it; topology closure resumes after the repaired Agent executes.
            return None

        missing_role_families = self._missing_semantic_role_families()
        if (
            self.semantic_protocol == _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL
            and missing_role_families
        ):
            sampled_role_families = tuple(
                (spec.role_family or "").casefold()
                for spec in action.agents
            )
            if (
                action.action_type is AgentActionType.ADD_SUBGRAPH
                and sampled_role_families
                and len(sampled_role_families)
                == len(set(sampled_role_families))
                and set(sampled_role_families) <= set(missing_role_families)
                and sampled_role_families.count("format") <= 1
                and action.output_agent_id is None
            ):
                return None
            return (
                "complete the missing HotpotQA semantic responsibilities with "
                "one add_subgraph transaction, execute that capability block, "
                "and defer Output assignment to a later SET_OUTPUT action; "
                "new roles must be distinct and come only from "
                "admitted_new_role_families="
                f"{list(missing_role_families)!r}"
            )

        takeover_delete_ids = (
            self._repair_exhausted_auxiliary_takeover_delete_ids()
        )
        if takeover_delete_ids:
            if (
                action.action_type is AgentActionType.DELETE_AGENT
                and action.agent_id in takeover_delete_ids
            ):
                return None
            return (
                "remove the bounded repair-exhausted auxiliary only after its "
                "same-role/same-artifact successful replacement has taken over "
                "all downstream relations; admissible_delete_agent_ids="
                f"{list(takeover_delete_ids)!r}"
            )

        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        if exhausted_reasoner_ids:
            failed_ingress_candidates = (
                self._failed_auxiliary_ingress_relation_candidates()
            )
            if failed_ingress_candidates:
                if any(
                    self._relation_action_matches_candidate(action, candidate)
                    for candidate in failed_ingress_candidates
                ):
                    return None
                return (
                    "close only the failed auxiliary ingress while preserving "
                    "the successful ingress and complete semantic dataflow; use "
                    "an exact admitted set_relation candidate"
                )
            routing_candidates = self._repair_exhausted_relation_candidates()
            if routing_candidates:
                if any(
                    self._relation_action_matches_candidate(action, candidate)
                    for candidate in routing_candidates
                ):
                    return None
                return (
                    "route one existing Evidence Retriever or Repair artifact "
                    "into the ReAct-repair-exhausted Reasoner before other Canvas "
                    "edits; use an exact admitted set_relation candidate"
                )

        dirty_replacement_ids = self._dirty_auxiliary_replacement_agent_ids()
        if dirty_replacement_ids:
            mutable_fields = {
                field
                for field in (
                    "model_id",
                    "contract",
                    "role_family",
                    "allowed_tools",
                    "execution_mode",
                    "artifact_type",
                    "completion_condition",
                )
                if getattr(action, field) is not None
            }
            if (
                action.action_type is AgentActionType.MODIFY_AGENT
                and action.agent_id in dirty_replacement_ids
                and len(mutable_fields) == 1
                and mutable_fields <= {"contract", "completion_condition"}
            ):
                return None
            return (
                "repair or execute only the unresolved same-role Evidence "
                "Retriever replacement at max_agents before modifying blocked "
                "downstream Agents; admissible_modify_agent_ids="
                f"{list(dirty_replacement_ids)!r}; mutable_fields="
                "['contract', 'completion_condition']"
            )

        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        if self._uses_role_conditional_capabilities() and replacement_domains:
            replacement_ingress_ids = set(
                self._repair_exhausted_auxiliary_replacement_ingress_consumer_ids()
            )
            replacement_role = (
                (action.agents[0].role_family or "").casefold()
                if len(action.agents) == 1
                else ""
            )
            replacement_artifact_type = (
                (action.agents[0].artifact_type or "text").casefold()
                if len(action.agents) == 1
                else ""
            )
            replacement_agent_id = (
                action.agents[0].agent_id if len(action.agents) == 1 else ""
            )
            has_required_ingress = (
                len(action.relations) == 1
                and action.relations[0].source_id == replacement_agent_id
                and action.relations[0].target_id in replacement_ingress_ids
                and action.relations[0].source_to_target is True
                and action.relations[0].target_to_source is False
            )
            relation_boundary_satisfied = (
                has_required_ingress
                if replacement_ingress_ids
                else not action.relations
            )
            if (
                action.action_type is AgentActionType.ADD_SUBGRAPH
                and len(action.agents) == 1
                and replacement_role in replacement_domains
                and replacement_artifact_type
                in replacement_domains[replacement_role]
                and relation_boundary_satisfied
                and action.output_agent_id is None
            ):
                return None
            return (
                "add the same-role/same-artifact auxiliary replacement "
                + (
                    "and route it to exactly one existing downstream "
                    "responsibility in the same edit"
                    if replacement_ingress_ids
                    else "as an isolated executable prefix with relations=[]"
                )
                + ". Do not assign Output in the replacement ADD"
            )
        if (
            action.action_type is AgentActionType.ADD_SUBGRAPH
            and len(action.agents) == 1
            and (
                replacement_role := (
                    action.agents[0].role_family or ""
                ).casefold()
            )
            in replacement_domains
            and (action.agents[0].artifact_type or "text").casefold()
            in replacement_domains[replacement_role]
            and (action.relations or action.output_agent_id is not None)
        ):
            return (
                "add the same-role/same-artifact auxiliary replacement "
                "as an isolated executable prefix with relations=[] and no "
                "output_agent_id. The accepted ADD executes immediately; route "
                "its artifact to the original downstream consumer only after "
                "that execution succeeds"
            )

        terminal_reachability_candidates = (
            self._terminal_reachability_relation_candidates()
        )
        prospective_convergence_candidates = (
            self._prospective_terminal_convergence_relation_candidates()
        )
        if prospective_convergence_candidates:
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in prospective_convergence_candidates
            ):
                return None
            return (
                "converge the successful parallel branches into an existing "
                "terminal-compatible Agent before Output assignment; use an "
                "exact admitted monotonic set_relation candidate"
            )
        if terminal_reachability_candidates:
            all_relation_candidates = (
                self._all_model_admissible_relation_candidates()
            )
            action_is_otherwise_admissible = any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in all_relation_candidates
            )
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in terminal_reachability_candidates
            ):
                return None
            if action_is_otherwise_admissible:
                return (
                    "repair terminal reachability with an exact admitted relation "
                    "that strictly reduces terminal_unreachable_agent_ids before "
                    "other Canvas edits"
                )

        required_relation_candidates = (
            self._required_semantic_relation_candidates()
        )
        if required_relation_candidates:
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in required_relation_candidates
            ):
                return None
            return (
                "close the declared Reasoner -> Verifier -> Formatter semantic "
                "dataflow before other Canvas edits; use an exact admitted "
                "set_relation candidate"
            )

        if (
            action.action_type is AgentActionType.SET_RELATION
            and self._relation_reintroduces_failed_auxiliary_ingress(
                {
                    "source_id": action.source_id,
                    "target_id": action.target_id,
                    "source_to_target": bool(action.source_to_target),
                    "target_to_source": bool(action.target_to_source),
                }
            )
        ):
            return (
                "do not reintroduce a failed, repair-exhausted, unresolved, or "
                "artifact-free auxiliary ingress after it has been detached"
            )

        missing_role_families = self._missing_semantic_role_families()
        if (
            exhausted_reasoner_ids
            and self._graph.nodes
            and missing_role_families
        ):
            sampled_role_families = tuple(
                (spec.role_family or "").casefold() for spec in action.agents
            )
            if (
                action.action_type is AgentActionType.ADD_SUBGRAPH
                and 1 <= len(action.agents) <= len(missing_role_families)
                and len(sampled_role_families) == len(set(sampled_role_families))
                and set(sampled_role_families) <= set(missing_role_families)
            ):
                return None
            return (
                "complete the missing semantic responsibilities before recovery "
                "augmentation; add only roles from admitted_new_role_families="
                f"{list(missing_role_families)!r}"
            )

        output_target_ids = self._model_admissible_output_agent_ids()
        if output_target_ids:
            if (
                action.action_type is AgentActionType.SET_OUTPUT
                and action.agent_id in output_target_ids
            ):
                return None
            return (
                "select a prospectively valid terminal-compatible Output Agent before "
                "other Canvas edits; admissible_output_agent_ids="
                f"{list(output_target_ids)!r}"
            )

        if (
            exhausted_reasoner_ids
            and self._model_admissible_add_role_families()
            and (
                self.max_agents is None
                or len(self._graph.nodes) < self.max_agents
            )
        ):
            sampled_role_families = tuple(
                (spec.role_family or "").casefold() for spec in action.agents
            )
            replacement_domains = (
                self._repair_exhausted_auxiliary_replacement_domains()
            )
            sampled_artifact_type = (
                (action.agents[0].artifact_type or "text").casefold()
                if len(action.agents) == 1
                else None
            )
            if (
                action.action_type is AgentActionType.ADD_SUBGRAPH
                and len(action.agents) == 1
                and sampled_role_families
                and sampled_role_families[0]
                in self._model_admissible_add_role_families()
                and (
                    not replacement_domains
                    or sampled_artifact_type
                    in replacement_domains.get(
                        sampled_role_families[0],
                        (),
                    )
                )
            ):
                return None
            if replacement_domains:
                return (
                    "bounded auxiliary recovery requires exactly one same-role/"
                    "same-artifact replacement before another recovery role; "
                    f"replacement_domains={replacement_domains!r}"
                )
            return (
                "ReAct-repair-exhausted augmentation admits exactly one new "
                "Evidence Retriever or Repair Agent so the staged relation can "
                "attach it without creating orphan Agents"
            )

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
                candidate = None
            if candidate is not None:
                for source_id, target_id in self._required_semantic_edges():
                    if (
                        target_id
                        in self._directed_successors(self._graph, source_id)
                        and target_id
                        not in self._directed_successors(candidate, source_id)
                    ):
                        return (
                            "the declared semantic-lineage relation "
                            f"{source_id!r} -> {target_id!r} must be preserved"
                        )
        if (
            action.action_type is AgentActionType.ADD_SUBGRAPH
            and self.semantic_protocol
            not in {
                _HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL,
                _HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL,
            }
        ):
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
        if "cancellederror" in normalized or (
            "cancelled" in normalized and "partial public execution" in normalized
        ):
            return (
                "sibling_fail_fast_cancellation",
                "preserve_public_continuation",
                status_code,
            )
        if "knowledge_base_coverage_failure" in normalized:
            return (
                "knowledge_base_coverage_failure",
                "repair_retrieval_or_database_coverage",
                status_code,
            )
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
            if category == "knowledge_base_coverage_failure":
                item["operational_diagnosis"] = {
                    "domain": "retrieval_or_database_coverage",
                    "corpus_level_oracle_claim": False,
                }
                item["preferred_repair"] = {
                    "action_order": [
                        "modify_agent",
                        "add_subgraph",
                        "set_relation",
                    ],
                    "agent_id": record.agent_id,
                    "preserve_fields": [
                        "existing_tool_receipts",
                        "valid_evidence",
                        "semantic_answer",
                        "relations",
                    ],
                }
            elif category == "provider_request_failure" and model_id is not None:
                admitted_model_ids = self._provider_repair_catalog_domain(
                    model_id
                )
                avoid_provider_id = (
                    provider_id
                    if any(
                        self.model_registry.provider_for(candidate_model_id).provider_id
                        != provider_id
                        for candidate_model_id in admitted_model_ids
                    )
                    else None
                )
                item["preferred_repair"] = {
                    "action": "modify_agent",
                    "agent_id": record.agent_id,
                    "field": "model_id",
                    "admitted_model_ids": list(admitted_model_ids),
                    **(
                        {"avoid_provider_id": avoid_provider_id}
                        if avoid_provider_id is not None
                        else {"fallback_provider_id": provider_id}
                    ),
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
                if record.agent_id in self._repair_exhausted_agent_ids:
                    item["preferred_repair"] = {
                        "action_order": [
                            "add_subgraph",
                            "set_relation",
                            "modify_agent",
                        ],
                        "admitted_role_families": [
                            role_family
                            for role_family in ("evidence_retriever", "repair")
                            if role_family
                            in self._admissible_augmentation_role_families()
                        ],
                        "preserve_agent_id": record.agent_id,
                        "preserve_fields": [
                            "existing_tool_receipts",
                            "react_trace",
                            "valid_evidence",
                            "semantic_answer",
                            "relations",
                        ],
                    }
                else:
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
        completed_partial_agent_ids = (
            set()
            if exc.partial_result is None
            else self._completed_partial_output_agent_ids(exc.partial_result)
        )
        non_terminal_partial_agent_ids = (
            set()
            if exc.partial_result is None
            else set(exc.partial_result.outputs) - completed_partial_agent_ids
        )
        payload = json.dumps(
            {
                "type": type(exc).__name__,
                "message": message,
                "failed_agents": failed_agents,
                "blocked_agent_ids": list(exc.blocked_agent_ids),
                "pending_agent_ids": list(exc.pending_agent_ids),
                "preserved_agent_ids": (
                    sorted(completed_partial_agent_ids)
                ),
                "non_terminal_partial_agent_ids": sorted(
                    non_terminal_partial_agent_ids
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
            require_exact_answer_tag=self.require_exact_answer_tag,
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
            previous_output_agent_id = graph.output_agent_id
            previous_output_predecessors = (
                ()
                if previous_output_agent_id is None
                else graph.directed_predecessors(previous_output_agent_id)
            )
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
                preserve_previous_output_artifact = bool(
                    self._uses_role_conditional_capabilities()
                    and previous_output_agent_id is not None
                    and previous_output_agent_id != action.output_agent_id
                    and self._has_successful_artifact(previous_output_agent_id)
                    and graph.directed_predecessors(previous_output_agent_id)
                    == previous_output_predecessors
                )
                if preserve_previous_output_artifact:
                    # The existing artifact becomes an upstream dependency of
                    # the new Output; transferring terminal ownership must not
                    # re-run its already completed Tool interaction.
                    graph.set_output(action.output_agent_id)
                    dirty_agents |= graph.dirty_closure(
                        {action.output_agent_id}
                    )
                else:
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
