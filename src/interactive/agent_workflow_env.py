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
    qa_answer_argument_constraint,
    qa_answer_type_constraint,
    verified_year_to_decade_normalization,
)


class AgentWorkflowStateError(RuntimeError):
    """Raised for invalid environment construction or restoration."""


_HOTPOTQA_SEMANTIC_PROTOCOL = "hotpotqa_verified_answer_slot_v1"
_QA_SEMANTIC_PROTOCOL = "qa_verified_answer_lineage_v2"
_TRIVIAQA_QA_MEMORY_TOOL_ID = "triviaqa.qa_memory"
_PRESERVE_REPAIR_RECOVERY_POLICY = "preserve_diagnose_repair_augment"
_SEMANTIC_LINEAGE_PROTOCOLS = frozenset(
    {_HOTPOTQA_SEMANTIC_PROTOCOL, _QA_SEMANTIC_PROTOCOL}
)
_SUPPORTED_SEMANTIC_PROTOCOLS = frozenset(
    {"none", *_SEMANTIC_LINEAGE_PROTOCOLS}
)
_SUPPORTED_RECOVERY_POLICIES = frozenset(
    {"default", _PRESERVE_REPAIR_RECOVERY_POLICY}
)
_TYPED_RETRIEVAL_FAILURE_RETRYABILITY = {
    "knowledge_base_coverage_failure": (
        "repair_retrieval_or_database_coverage"
    ),
    "retrieval_recall_failure": "repair_retrieval_or_database_coverage",
    "retrieval_strategy_failure": (
        "repair_execution_contract_or_tool_plan"
    ),
}
_BOUNDED_REACT_FAILURE_CATEGORIES = frozenset(
    {
        "react_continuation_request_failure",
        "react_turn_exhaustion",
        *_TYPED_RETRIEVAL_FAILURE_RETRYABILITY,
    }
)
_HOTPOTQA_FORMAT_CONTRACT = (
    "copy the supported Verifier candidate character-for-character into the "
    "required answer wrapper"
)
_QA_ROLE_CONDITIONAL_FORMAT_CONTRACT = (
    "copy the routed semantic candidate character-for-character into the "
    "required answer wrapper"
)
_QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT = (
    "produce answer-free evidence propositions for the original entity and "
    "requested relation, grounded in matching successful read Tool receipts"
)
_QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION = (
    "complete only when entity identity and the requested relation are "
    "supported by a matching successful read Tool receipt"
)
_QA_LOCATION_REASONER_RECOVERY_CONTRACT = (
    "preserve the receipt-grounded first-hop location proposition as "
    "evidence_propositions[0], encode the public location-containment "
    "proposition as evidence_propositions[1], bind answer_slot to "
    "proposition_index 1 and answer_field object_or_attribute_value, and copy "
    "that field exactly into candidate_answer"
)
_QA_LOCATION_REASONER_RECOVERY_COMPLETION = (
    "complete only when evidence_propositions[0] is the preserved first hop, "
    "evidence_propositions[1] is the public containment hop, answer_slot uses "
    "proposition_index 1 and answer_field object_or_attribute_value, and "
    "candidate_answer copies that field exactly"
)
_QA_ROLE_CONTRACT_RESPONSIBILITIES = {
    "evidence_retriever": (
        "ground answer-free evidence for the original entity and requested "
        "relation in matching successful read Tool receipts"
    ),
    "reasoner": (
        "bind grounded evidence to the original answer slot and requested "
        "relation, then derive one semantic candidate"
    ),
    "verifier": (
        "check entity identity, Tool provenance, semantic scope, relation "
        "binding, and answer lineage without changing the candidate"
    ),
    "format": (
        "copy the supported Verifier candidate without semantic reasoning"
    ),
}
_QA_ROLE_EXCLUSIVE_BARE_CONTRACTS = {
    "evidence_retriever": frozenset(
        {
            "format",
            "formatter",
            "formatting",
            "reason",
            "reasoner",
            "reasoning",
            "verification",
            "verifier",
            "verify",
        }
    ),
    "reasoner": frozenset(
        {
            "format",
            "formatter",
            "formatting",
            "read",
            "reader",
            "retrieval",
            "retrieve",
            "retriever",
            "search",
            "serialization",
            "serialize",
            "wrapper",
        }
    ),
    "verifier": frozenset(
        {
            "answer",
            "format",
            "formatter",
            "formatting",
            "reason",
            "reasoner",
            "reasoning",
            "retrieval",
            "retrieve",
            "retriever",
            "serialization",
            "serialize",
            "wrapper",
        }
    ),
}

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


class _ReadReceiptText(str):
    """Passage text carrying its public read-receipt title for entity linking.

    Lexical evidence checks continue to see only the passage body.  The title
    remains separately available to the entity-identity gate, matching the
    QA adapter's existing receipt-backed title/alias rule without admitting a
    title-only evidence proposition.
    """

    passage_title: Optional[str]

    def __new__(
        cls,
        text: str,
        *,
        passage_title: Optional[str] = None,
    ) -> "_ReadReceiptText":
        value = super().__new__(cls, text)
        value.passage_title = passage_title
        return value


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
    return " ".join(normalized.casefold().split())


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
        require_evidence_relation: bool = False,
        director_feedback_mode: str = "artifact_preview",
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
        if type(require_evidence_relation) is not bool:
            raise AgentWorkflowStateError("require_evidence_relation must be bool")
        if require_evidence_relation and required_evidence_tool_id is None:
            raise AgentWorkflowStateError(
                "require_evidence_relation requires required_evidence_tool_id"
            )
        if director_feedback_mode not in {"artifact_preview", "control_plane"}:
            raise AgentWorkflowStateError(
                "director_feedback_mode must be artifact_preview or control_plane"
            )
        if semantic_protocol in _SEMANTIC_LINEAGE_PROTOCOLS:
            if recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
                raise AgentWorkflowStateError(
                    f"{semantic_protocol} requires "
                    "recovery_policy=preserve_diagnose_repair_augment"
                )
            dataset_id = None if runtime is None else runtime.dataset_id
            admitted_evidence_tool_ids = (
                {"qa-retrieval", "triviaqa.qa_memory"}
                if semantic_protocol == _QA_SEMANTIC_PROTOCOL
                and isinstance(dataset_id, str)
                and dataset_id.casefold() == "triviaqa"
                else {"qa-retrieval"}
            )
            if required_evidence_tool_id not in admitted_evidence_tool_ids:
                raise AgentWorkflowStateError(
                    f"{semantic_protocol} requires "
                    "a dataset-admitted required_evidence_tool_id; expected one "
                    f"of {sorted(admitted_evidence_tool_ids)!r}"
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
        self.require_evidence_relation = require_evidence_relation
        self.director_feedback_mode = director_feedback_mode
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
        # Frozen catalog membership remains unchanged during a trajectory.
        # Permanent provider/model failures add only a trajectory-scoped
        # availability overlay used by Runtime scheduling and Canvas admission.
        self._unavailable_model_ids: set[str] = set()
        self._model_availability_receipts: list[dict[str, object]] = []
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

    def director_control_plane_feedback(self) -> dict[str, object]:
        """Return the latest Canvas receipt without data-plane content."""

        if not self._history:
            return {
                "status": "initial_canvas",
                "graph_revision": self._graph.revision,
            }
        entry = self._history[-1]
        action = None if entry.action is None else entry.action.to_dict()
        action_target: object = None
        if isinstance(action, Mapping):
            action_target = action.get("agent_id")
            if action_target is None and action.get("source_id") is not None:
                action_target = {
                    "source_id": action.get("source_id"),
                    "target_id": action.get("target_id"),
                }
        result: dict[str, object] = {
            "status": (
                "finished"
                if entry.done
                else "accepted"
                if entry.accepted
                else "rejected"
            ),
            "accepted": entry.accepted,
            "done": entry.done,
            "graph_revision": entry.revision,
            "action": None if action is None else action.get("action"),
            "target": action_target,
            "execution_reused": entry.execution_reused,
        }
        if self.director_feedback_mode != "control_plane":
            return result
        for marker, field_name in (
            ("execution_result=", "execution_receipt"),
            ("execution_error=", "failure_receipt"),
        ):
            marker_index = entry.feedback.find(marker)
            if marker_index < 0:
                continue
            try:
                value = json.loads(entry.feedback[marker_index + len(marker) :])
            except (TypeError, ValueError, json.JSONDecodeError):
                break
            if isinstance(value, Mapping):
                result[field_name] = dict(value)
            break
        return result

    def director_control_plane_finish_admissibility(self) -> dict[str, object]:
        """Project the FINISH gate to typed, content-free control state."""

        admission = self.finish_admissibility()
        projected: dict[str, object] = {
            field: admission[field]
            for field in (
                "admissible",
                "stage",
                "graph_revision",
                "submission_semantics",
            )
            if field in admission
        }
        issues = admission.get("issues")
        if isinstance(issues, (list, tuple)):
            projected["issues"] = [
                {
                    field: item[field]
                    for field in ("code", "agent_ids")
                    if field in item
                }
                for item in issues
                if isinstance(item, Mapping)
            ]
        attribution = admission.get("failure_attribution")
        if isinstance(attribution, Mapping):
            safe_attribution_fields = (
                "responsible_constraint",
                "responsible_role_family",
                "responsible_agent_id",
                "responsible_agent_ids",
                "format_target_agent_ids",
                "preserve_agent_ids",
                "delete_allowed_before_replacement_takeover",
                "operational_diagnosis",
                "corpus_level_oracle_claim",
            )
            projected["failure_attribution"] = {
                field: attribution[field]
                for field in safe_attribution_fields
                if field in attribution
            }
        if "semantic_lineage_diagnostic" in admission:
            projected["semantic_lineage_diagnostic_present"] = True
        if self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY:
            projected["recovery_state"] = self._control_plane_recovery_state()
        return projected

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
        """Return whether semantic roles are optional Canvas capabilities."""

        return self.semantic_protocol == _QA_SEMANTIC_PROTOCOL

    def _requires_complete_semantic_lineage(self) -> bool:
        """Return whether FINISH requires the full QA responsibility lineage.

        The shared QA protocol can expose optional capabilities for historical
        conditions.  ``require_format_agent`` is the existing configuration
        boundary that selects the already implemented evidence--reasoning--
        verification--formatting lineage.  Keeping this decision in Canvas
        admission, rather than the Director system prompt, preserves a neutral
        topology search space while making the requested semantic
        responsibilities non-optional.
        """

        return bool(
            self._uses_semantic_lineage_protocol()
            and (
                not self._uses_role_conditional_capabilities()
                or self.require_format_agent
            )
        )

    @staticmethod
    def _qa_role_contract_responsibility_issue(node: AgentNode) -> Optional[str]:
        """Reject a contract that names a different QA responsibility.

        FlowSteer keeps the Agent contract model-authored, while SkillFlow's
        role-specific execution schema remains authoritative at Runtime.  This
        admission check only closes a measured mismatch between those two
        boundaries; it does not prescribe contract wording, Agent order, edges,
        Agent count, or graph topology.
        """

        role = (node.role_family or "").casefold()
        conflicting_labels = _QA_ROLE_EXCLUSIVE_BARE_CONTRACTS.get(role)
        if not conflicting_labels:
            return None
        normalized = " ".join(node.contract.casefold().split()).strip(" .,:;")
        normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
        normalized = re.sub(r"\s+(?:agent|role|task)$", "", normalized)
        canonical_conflicts = frozenset(
            " ".join(responsibility.casefold().split()).strip(" .,:;")
            for other_role, responsibility
            in _QA_ROLE_CONTRACT_RESPONSIBILITIES.items()
            if other_role != role
        )
        if (
            normalized not in conflicting_labels
            and normalized not in canonical_conflicts
        ):
            return None
        responsibility = _QA_ROLE_CONTRACT_RESPONSIBILITIES[role]
        return (
            f"{role.replace('_', ' ').title()} Agent {node.id!r} has a "
            f"contract naming another role responsibility ({node.contract!r}); "
            f"its contract must instead describe how it will {responsibility}"
        )

    def _role_conditional_registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return the task Runtime profiles exposed to the QA action domain."""

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

    def _topology_neutral_registered_execution_profiles(
        self,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return registered profiles for a topology-neutral Tool Canvas.

        FlowSteer's live action domain must bind ``execution_mode`` and
        ``allowed_tools`` as one executable profile.  This projection keeps
        every no-Tool Runtime profile and the exact dataset worker profile;
        it does not assign either profile to a role or prescribe an edge.
        """

        required_tool_id = self.required_evidence_tool_id
        return tuple(
            (execution_mode, allowed_tools)
            for execution_mode, allowed_tools in (
                self.runtime.registered_execution_profiles()
            )
            if allowed_tools == ()
            or (
                required_tool_id is not None
                and allowed_tools == (required_tool_id,)
            )
        )

    def _role_conditional_execution_profiles_for(
        self,
        role_family: str,
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Return registered execution profiles compatible with one role."""

        registered = self._role_conditional_registered_execution_profiles()
        role = role_family.casefold()
        if role == "format":
            return tuple(
                profile
                for profile in registered
                if profile == ("reasoning", ())
            )
        if role == "output":
            # Output is a terminal consumer.  A QA-memory Tool principal must
            # be an upstream worker whose receipt-bearing artifact arrives via
            # an explicit AgentGraph relation.
            return tuple(
                profile
                for profile in registered
                if profile == ("reasoning", ())
            )
        if role == "evidence_retriever":
            return tuple(
                profile
                for profile in registered
                if profile
                == ("react", (self.required_evidence_tool_id,))
            )
        if role == "reasoner" and self.require_format_agent:
            return tuple(
                profile
                for profile in registered
                if profile
                == ("react", (self.required_evidence_tool_id,))
            )
        if role == "verifier" and self.require_format_agent:
            return tuple(
                profile
                for profile in registered
                if profile == ("reasoning", ())
            )
        return registered

    def _role_conditional_execution_constraint(
        self,
        role_family: str,
    ) -> dict[str, object]:
        """Render correlated mode/Tool choices for constrained decoding."""

        profiles = self._role_conditional_execution_profiles_for(role_family)
        return {
            "execution_modes": list(
                dict.fromkeys(mode for mode, _ in profiles)
            ),
            "allowed_tools": [
                list(tool_ids)
                for tool_ids in dict.fromkeys(
                    tool_ids for _, tool_ids in profiles
                )
            ],
            "execution_profiles": [
                {
                    "execution_mode": mode,
                    "allowed_tools": list(tool_ids),
                }
                for mode, tool_ids in profiles
            ],
        }

    def _semantic_protocol_label(self) -> str:
        return (
            "HotpotQA"
            if self.semantic_protocol == _HOTPOTQA_SEMANTIC_PROTOCOL
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
            (
                self._uses_semantic_lineage_protocol()
                and self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY
            )
            or self.require_evidence_relation
        ) and (
            finish_admitted
            and AgentActionType.FINISH.value in self._allowed_action_type_set
        ):
            # Once the current revision has either a verified semantic lineage
            # or the configured routed-evidence receipt and a valid terminal
            # artifact, further edits can only endanger the completed answer.
            # FlowSteer's action mask still asks the Director to emit the
            # explicit terminal action; it does not finish automatically.
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
            # augmentation becomes available only after repair succeeds or a
            # typed bounded-failure replacement takeover is established.
            if self._model_admissible_modify_agent_ids():
                return (AgentActionType.MODIFY_AGENT.value,)
            # A measured provider failure with no catalog-backed replacement
            # has no legal parameter repair.  Reuse FlowSteer's empty Canvas
            # action domain so the Orchestrator persists the existing typed
            # ``canvas_action_domain_exhausted`` terminal diagnosis instead of
            # sampling an unrelated contract or topology edit.
            return ()

        stalled_semantic_repair_ids = self._stalled_semantic_repair_agent_ids()
        if stalled_semantic_repair_ids and not any(
            (
                self._graph.get_node(agent_id).role_family or ""
            ).casefold()
            == "reasoner"
            for agent_id in stalled_semantic_repair_ids
        ):
            # One measured repair already re-executed against the same public
            # read receipts and reproduced the same semantic fault. A second
            # blind parameter edit would violate preserve→diagnose→repair/
            # augment. With no grounded Reasoner augmentation boundary, expose
            # natural typed termination rather than an unbounded MODIFY loop.
            return ()

        takeover_delete_ids = (
            self._repair_exhausted_auxiliary_takeover_delete_ids()
        )
        dirty_replacement_ids = self._dirty_auxiliary_replacement_agent_ids()
        node_count = len(self._graph.nodes)
        node_ids = tuple(node.id for node in self._graph.nodes)
        can_add = self.max_agents is None or node_count < self.max_agents
        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        evidence_recovery_reasoner_ids = tuple(
            agent_id
            for agent_id in exhausted_reasoner_ids
            if self._reasoner_failure_requires_evidence_augmentation(agent_id)
        )
        has_single_evidence_recovery_target = bool(
            len(exhausted_reasoner_ids) == 1
            and evidence_recovery_reasoner_ids == exhausted_reasoner_ids
        )
        exhausted_auxiliary_ids = tuple(
            agent_id
            for agent_id in self._recovery_auxiliary_agent_ids()
            if agent_id in self._failed_agent_ids
            and agent_id in self._repair_exhausted_agent_ids
        )
        auxiliary_replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )

        if (
            dirty_replacement_ids
            and AgentActionType.MODIFY_AGENT.value
            in self._allowed_action_type_set
        ):
            # An isolated same-responsibility replacement which has not yet
            # materialized its prefix is the only executable recovery
            # responsibility, regardless of spare Agent capacity. Do not spend
            # the next FlowSteer edit on another augmentation or on a blocked
            # downstream Agent which merely lacks that artifact.
            return (AgentActionType.MODIFY_AGENT.value,)

        missing_role_families = self._missing_semantic_role_families()
        if (
            self._uses_semantic_lineage_protocol()
            and node_count
            and missing_role_families
            and can_add
            and AgentActionType.ADD_SUBGRAPH.value
            in self._allowed_action_type_set
        ):
            if (
                exhausted_reasoner_ids
                and not has_single_evidence_recovery_target
            ):
                # A receipt-grounded Retriever is already routed into the
                # measured Reasoner and its latest public failure belongs to
                # structured semantic binding rather than retrieval.  Adding
                # Verifier/Formatter responsibilities cannot materialize the
                # missing Reasoner artifact, so fail closed at this exact
                # FlowSteer edit boundary instead of constructing a downstream
                # spine that AgentRuntime can only defer.
                return ()
            if (
                isinstance(self.runtime.dataset_id, str)
                and self.runtime.dataset_id.casefold() == "triviaqa"
                and exhausted_auxiliary_ids
                and not auxiliary_replacement_domains
                and not self._has_valid_evidence_retriever_artifact()
            ):
                # A bounded Retriever generation exhausted the only
                # non-destructive declaration repair and produced no valid
                # artifact. Missing downstream responsibilities cannot bypass
                # Evidence Grounding, so the live Canvas domain is exhausted.
                return ()
            required_evidence_ingress_candidates = (
                self._required_evidence_ingress_relation_candidates()
            )
            if (
                not bool(
                    self._failed_agent_ids
                    or self._repair_exhausted_agent_ids
                )
                and required_evidence_ingress_candidates
                and AgentActionType.SET_RELATION.value
                in self._allowed_action_type_set
            ):
                # Match authoritative Canvas admission: after a Retriever has
                # materialized a valid public artifact, route that evidence
                # into the existing Reasoner before adding another semantic
                # responsibility.  A typed failure with a missing role keeps
                # the preserve/repair ADD boundary above, exactly as
                # ``_semantic_edit_issue_for`` requires.  This only orders two
                # live FlowSteer edits; it does not prescribe a topology.
                return (AgentActionType.SET_RELATION.value,)
            # A partial executable Canvas must first materialize every required
            # semantic responsibility. This is a state-conditioned ADD domain,
            # not a fixed Agent count, order, edge set, or topology: one ADD may
            # still contain any legal subset of the missing roles and any legal
            # intra-unit reciprocal communication.
            return (AgentActionType.ADD_SUBGRAPH.value,)

        capacity_recovery_delete_ids = (
            self._capacity_blocking_failed_auxiliary_delete_ids()
        )
        if (
            capacity_recovery_delete_ids
            and AgentActionType.DELETE_AGENT.value
            in self._allowed_action_type_set
        ):
            # FlowSteer's DELETE remains an explicit Canvas edit. Admit it at
            # a full-capacity augmentation boundary only for a typed,
            # repair-exhausted auxiliary which has no artifact, public read,
            # relation, Output identity, or active semantic-lineage identity.
            # The following turn can then ADD the still-missing semantic
            # responsibility without erasing any executed evidence.
            return (AgentActionType.DELETE_AGENT.value,)

        if (
            takeover_delete_ids
            and AgentActionType.DELETE_AGENT.value
            in self._allowed_action_type_set
        ):
            # A bounded ReAct repair has failed without advancing its public
            # Tool receipts and a same-role/same-artifact auxiliary has already
            # taken over every downstream responsibility. Reuse FlowSteer's
            # existing replacement-takeover delete boundary before adding yet
            # another recovery branch.
            return (AgentActionType.DELETE_AGENT.value,)

        if (
            self._uses_semantic_lineage_protocol()
            and node_count
            and missing_role_families
        ):
            # Capacity is full and neither preservation-safe DELETE boundary
            # above can free a slot. No relation or parameter edit can create
            # the missing responsibility, so expose the natural empty Canvas
            # domain for typed terminal persistence.
            return ()

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

        required_relation_candidates = (
            self._required_semantic_relation_candidates()
        )
        required_evidence_ingress_candidates = (
            self._required_evidence_ingress_relation_candidates()
        )
        if (
            required_evidence_ingress_candidates
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # TriviaQA admits the Reasoner execution unit only after a current,
            # receipt-grounded Retriever artifact can be routed into it.  This
            # is a state-conditioned data dependency, not a fixed global graph:
            # any valid Retriever (including one of several parallel branches)
            # may own the ingress and reciprocal communication remains legal.
            return (AgentActionType.SET_RELATION.value,)
        if (
            required_relation_candidates
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # Canvas admission requires the same exact edge before any recovery
            # augmentation, so expose that edge rather than an ADD action which
            # the authoritative gate would reject.
            return (AgentActionType.SET_RELATION.value,)

        output_target_ids = self._model_admissible_output_agent_ids()
        if (
            self._uses_semantic_lineage_protocol()
            and output_target_ids
            and AgentActionType.SET_OUTPUT.value in self._allowed_action_type_set
        ):
            # A Formatter is exposed only after the prospective Canvas passes
            # the same Format-lineage checks used by authoritative admission.
            return (AgentActionType.SET_OUTPUT.value,)

        terminal_reachability_candidates = (
            self._terminal_reachability_relation_candidates()
        )
        if (
            terminal_reachability_candidates
            and self._model_admissible_relation_candidates()
            and AgentActionType.SET_RELATION.value
            in self._allowed_action_type_set
        ):
            # FlowSteer's live action mask must expose only an edit that makes
            # progress on the current graph-validation diagnosis. Falling
            # through to the generic relation domain here permits unrelated
            # rewrites while an orphan recovery branch remains unresolved.
            return (AgentActionType.SET_RELATION.value,)

        if exhausted_reasoner_ids:
            if (
                self._model_admissible_add_role_families()
                and can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)
            # Every evidence/routing/repair boundary above is closed.  Do not
            # fall through to unrelated relation rewrites for a measured
            # structured Reasoner failure; persist the typed empty Canvas
            # domain until a supported same-responsibility repair exists.
            return ()

        if exhausted_auxiliary_ids:
            if (
                self._model_admissible_add_role_families()
                and can_add
                and AgentActionType.ADD_SUBGRAPH.value
                in self._allowed_action_type_set
            ):
                return (AgentActionType.ADD_SUBGRAPH.value,)
            # Every narrower recovery boundary above is empty and the failed
            # bounded auxiliary has no strict-progress replacement domain.
            # Reuse FlowSteer's natural empty Canvas action domain instead of
            # falling through to unrelated peer-edge rewrites.
            return ()

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
            if (
                action_type == AgentActionType.ADD_SUBGRAPH.value
                and can_add
                and (
                    not self._uses_semantic_lineage_protocol()
                    or bool(self._model_admissible_add_role_families())
                )
            ):
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
                    if self._semantic_edit_issue_for(candidate) is not None:
                        continue
                    if self._preserved_input_change_issue_for(candidate) is not None:
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

        if (
            self._uses_semantic_lineage_protocol()
            and self._graph.nodes
            and self._missing_semantic_role_families()
        ):
            # A successful TriviaQA Retriever may need to hand its receipt-
            # grounded artifact to an already declared Reasoner before the
            # Director adds the remaining semantic responsibilities.  The
            # action-type projection already prioritizes this exact ingress;
            # expose the identical parameter domain here so FlowSteer's v3
            # two-phase StructuredAction schema cannot advertise SET_RELATION
            # and then fail with an empty selector.  This orders one live
            # data-dependency edit only; it does not prescribe the remaining
            # Agent count, edges, order, or topology.
            all_candidates = self._all_model_admissible_relation_candidates()
            evidence_ingress_candidates = (
                self._required_evidence_ingress_relation_candidates(
                    all_candidates
                )
            )
            if evidence_ingress_candidates:
                return evidence_ingress_candidates
            # A peer edge cannot materialize a missing semantic
            # responsibility. With capacity, ADD owns the next Canvas edit;
            # at full capacity, only the strict artifact-free auxiliary DELETE
            # boundary may free a slot. If neither is possible the domain is
            # intentionally empty so the collector records a typed terminal
            # state instead of drifting through arbitrary reciprocal edges.
            return []
        all_candidates = self._all_model_admissible_relation_candidates()
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
        evidence_ingress_candidates = (
            self._required_evidence_ingress_relation_candidates(
                all_candidates
            )
        )
        if evidence_ingress_candidates:
            return evidence_ingress_candidates
        required_candidates = self._required_semantic_relation_candidates(
            all_candidates
        )
        if required_candidates:
            return required_candidates
        terminal_reachability_candidates = (
            self._terminal_reachability_relation_candidates(all_candidates)
        )
        if terminal_reachability_candidates:
            return terminal_reachability_candidates
        if any(
            auxiliary_id in self._failed_agent_ids
            and auxiliary_id in self._repair_exhausted_agent_ids
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
        ):
            # The replacement, evidence-ingress, semantic-dataflow, and
            # terminal-reachability projections above are the complete legal
            # relation recovery domain for a bounded auxiliary failure. Do not
            # expose FlowSteer's generic relation-edit fallback once all of
            # those progress candidates are empty.
            return []
        if (
            self._uses_semantic_lineage_protocol()
            and self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY
            and self._repair_exhausted_reasoner_ids()
        ):
            # Exhausted semantic recovery has already tried every narrower
            # relation repair above.  Do not fall through to FlowSteer's
            # unconstrained relation-edit domain and create peer cycles or
            # arbitrary handoffs around a measured terminal failure.
            return []
        return [
            item
            for item in all_candidates
            if not self._relation_reintroduces_failed_auxiliary_ingress(item)
            and not self._relation_adds_failed_artifact_free_auxiliary_source(
                item
            )
            and not self._relation_routes_replacement_outside_reasoner(item)
        ]

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
        if not self._requires_complete_semantic_lineage():
            # Role labels are capabilities selected inside the FlowSteer
            # search space.  Terminal artifact/receipt validation, rather
            # than a fixed role adjacency, determines whether a routed graph
            # may finish.
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

    def _required_evidence_ingress_relation_candidates(
        self,
        candidates: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> list[dict[str, object]]:
        """Return live TriviaQA Retriever-to-Reasoner ingress edits.

        The QA Tool completion validator remains the evidence authority.  This
        projection only asks whether at least one currently valid answer-free
        Retriever artifact can become a direct Reasoner input; it never picks a
        unique Retriever, a relation direction beyond evidence ingress, or an
        otherwise fixed topology.
        """

        if self.semantic_protocol != _QA_SEMANTIC_PROTOCOL:
            return []
        reasoner_ids = self._semantic_role_agent_ids("reasoner")
        if len(reasoner_ids) != 1:
            return []
        reasoner_id = reasoner_ids[0]
        valid_retriever_ids = tuple(
            agent_id
            for agent_id in self._semantic_role_agent_ids(
                "evidence_retriever"
            )
            if self._semantic_replacement_has_valid_artifact(
                agent_id,
                "evidence_retriever",
            )
        )
        if not valid_retriever_ids:
            return []
        if any(
            retriever_id
            in self._graph.directed_predecessors(reasoner_id)
            for retriever_id in valid_retriever_ids
        ):
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
            if any(
                retriever_id
                in candidate.directed_predecessors(reasoner_id)
                for retriever_id in valid_retriever_ids
            ):
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

    def _receipt_valid_routed_evidence_retriever_ids(
        self,
        reasoner_id: str,
    ) -> Tuple[str, ...]:
        """Return current receipt-grounded Retriever inputs to one Reasoner."""

        if (
            not self._graph.has_node(reasoner_id)
            or not self._has_valid_evidence_retriever_artifact()
        ):
            return ()
        return tuple(
            predecessor_id
            for predecessor_id in self._graph.directed_predecessors(reasoner_id)
            if (
                self._graph.get_node(predecessor_id).role_family or ""
            ).casefold()
            == "evidence_retriever"
            and predecessor_id not in self._failed_agent_ids
            and predecessor_id not in self._repair_exhausted_agent_ids
            and predecessor_id not in self._unresolved_dirty_agents
            and self._has_successful_artifact(predecessor_id)
            and self._semantic_replacement_has_valid_artifact(
                predecessor_id,
                "evidence_retriever",
            )
        )

    def _reasoner_failure_requires_evidence_augmentation(
        self,
        reasoner_id: str,
    ) -> bool:
        """Gate one bounded Retriever augmentation on public failure evidence.

        Missing receipt-valid ingress is an evidence deficit.  With an
        existing ingress, only an explicit latest retrieval diagnosis or
        missing-read contract may justify one additional Retriever.  A second
        valid routed Retriever closes that bounded recovery frontier; repeated
        fan-in cannot repair answer-slot, relation, or structured-artifact
        failures.
        """

        valid_ingress_ids = self._receipt_valid_routed_evidence_retriever_ids(
            reasoner_id
        )
        if not valid_ingress_ids:
            return True
        record = self._latest_failure_record_by_agent.get(reasoner_id)
        if record is None:
            return False
        public_summary = self._react_public_error_summary(record)
        last_public_error = public_summary.get("last_public_error", {})
        public_error_code = (
            last_public_error.get("public_error_code")
            if isinstance(last_public_error, Mapping)
            else None
        )
        if isinstance(public_error_code, str):
            if public_error_code not in {
                "qa_completion_requires_successful_read_evidence",
                "qa_read_requires_successful_search",
                "retrieval_recall_failure",
                "retrieval_strategy_failure",
            }:
                return False
        elif self._typed_retrieval_failure_category(record) not in {
            "retrieval_recall_failure",
            "retrieval_strategy_failure",
        }:
            return False
        return len(valid_ingress_ids) == 1

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
            if (
                self._graph.output_agent_id is None
                or candidate_unreachable < current_unreachable
            ):
                # FlowSteer's execute-after-edit recovery can materialize a
                # receipt-grounded auxiliary before the Director selects an
                # Output Agent.  At that partial-Canvas boundary there is no
                # terminal node from which ``cannot_reach_output`` can measure
                # strict progress, but routing the preserved artifact into the
                # measured exhausted Reasoner is itself the exact executable
                # recovery edit.  Once an Output exists, retain the stricter
                # reachability-reduction gate.  This does not prescribe a
                # workflow topology: source and target still come entirely
                # from the live, validated Canvas candidate domain above.
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
            if (
                self._relation_reintroduces_failed_auxiliary_ingress(item)
                or self._relation_adds_failed_artifact_free_auxiliary_source(
                    item
                )
                or self._relation_routes_replacement_outside_reasoner(item)
            ):
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

    def _relation_added_edges(
        self,
        item: Mapping[str, object],
    ) -> Tuple[Tuple[str, str], ...]:
        """Return directed edges newly introduced by one relation edit."""

        source_id = str(item["source_id"])
        target_id = str(item["target_id"])
        previous = self._graph.relation_bits(source_id, target_id)
        result: list[tuple[str, str]] = []
        if item["source_to_target"] is True and not previous.source_to_target:
            result.append((source_id, target_id))
        if item["target_to_source"] is True and not previous.target_to_source:
            result.append((target_id, source_id))
        return tuple(result)

    def _relation_adds_failed_artifact_free_auxiliary_source(
        self,
        item: Mapping[str, object],
    ) -> bool:
        """Reject reachability edges sourced by a measured failed auxiliary."""

        unavailable_sources = {
            auxiliary_id
            for auxiliary_id in self._recovery_auxiliary_agent_ids()
            if auxiliary_id in self._failed_agent_ids
            and auxiliary_id in self._repair_exhausted_agent_ids
            and not self._has_successful_artifact(auxiliary_id)
        }
        return any(
            source_id in unavailable_sources
            for source_id, _ in self._relation_added_edges(item)
        )

    def _successful_auxiliary_replacement_agent_ids(self) -> Tuple[str, ...]:
        """Return valid auxiliaries taking over one failed responsibility."""

        failed_domains = {
            (
                (node.role_family or "").casefold(),
                node.artifact_type.casefold(),
            )
            for node in self._graph.nodes
            if (node.role_family or "").casefold()
            in {"evidence_retriever", "repair"}
            and node.id in self._failed_agent_ids
            and node.id in self._repair_exhausted_agent_ids
        }
        return tuple(
            node.id
            for node in self._graph.nodes
            if (
                (node.role_family or "").casefold(),
                node.artifact_type.casefold(),
            )
            in failed_domains
            and node.id not in self._failed_agent_ids
            and node.id not in self._repair_exhausted_agent_ids
            and node.id not in self._unresolved_dirty_agents
            and self._has_successful_artifact(node.id)
            and self._semantic_replacement_has_valid_artifact(
                node.id,
                (node.role_family or "").casefold(),
            )
        )

    def _relation_routes_replacement_outside_reasoner(
        self,
        item: Mapping[str, object],
    ) -> bool:
        """Keep replacement artifact ingress on a semantic Reasoner consumer."""

        replacement_ids = set(
            self._successful_auxiliary_replacement_agent_ids()
        )
        if not replacement_ids:
            return False
        reasoner_ids = set(self._semantic_role_agent_ids("reasoner"))
        return any(
            source_id in replacement_ids and target_id not in reasoner_ids
            for source_id, target_id in self._relation_added_edges(item)
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
                self.require_evidence_relation
                and self.required_evidence_tool_id in node.allowed_tools
            ):
                # The QA-memory Tool is a worker capability.  Output consumes
                # an explicitly routed artifact and must not retrieve its own
                # answer outside that AgentGraph communication edge.
                continue
            if self._uses_semantic_lineage_protocol():
                role_family = (node.role_family or "").casefold()
                admitted_output_roles = (
                    {"format"}
                    if self._requires_complete_semantic_lineage()
                    else {"format", "output"}
                )
                if role_family not in admitted_output_roles:
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
            if self._semantic_edit_issue_for(candidate) is not None:
                continue
            if (
                self._uses_format_agent_protocol(candidate)
                and self._format_agent_issue_for(candidate) is not None
            ):
                continue
            admitted.append(node.id)
        return tuple(admitted)

    def _model_admissible_modify_agent_ids(self) -> Tuple[str, ...]:
        """Exclude an already verified semantic lineage from repair targets."""

        node_ids = tuple(node.id for node in self._graph.nodes)
        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return node_ids

        def has_non_noop_repair(agent_id: str) -> bool:
            if self._provider_repair_required(agent_id):
                return bool(self._provider_repair_model_ids(agent_id))
            if agent_id not in self._react_exhausted_agent_ids:
                return True
            recovery_values = self._triviaqa_react_recovery_field_values(
                agent_id
            )
            return recovery_values is None or bool(recovery_values)

        mandatory_repair_ids = self._mandatory_repair_agent_ids()
        if mandatory_repair_ids:
            return tuple(
                agent_id
                for agent_id in mandatory_repair_ids
                if has_non_noop_repair(agent_id)
            )
        dirty_replacement_ids = self._dirty_auxiliary_replacement_agent_ids()
        if dirty_replacement_ids:
            return tuple(
                agent_id
                for agent_id in dirty_replacement_ids
                if has_non_noop_repair(agent_id)
            )
        measured_failed = (
            self._failed_agent_ids - self._repair_exhausted_agent_ids
        ).intersection(node_ids)
        if measured_failed:
            # AgentRuntime distinguishes the Agent that raised a typed failure
            # from blocked/pending descendants.  FlowSteer's next Canvas edit
            # should repair every measured root failure, not mutate downstream
            # Agents that merely could not execute because their input is absent.
            return tuple(
                node_id
                for node_id in node_ids
                if node_id in measured_failed and has_non_noop_repair(node_id)
            )
        if self._failed_agent_ids.intersection(node_ids):
            # Every measured failure is terminal for MODIFY at this boundary.
            # Blocked descendants are not substitute repair targets; recovery
            # must use a legal routing, replacement, or fail-closed boundary.
            return ()
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

    def _triviaqa_evidence_retriever_recovery_field_values(
        self,
        agent_id: str,
    ) -> Optional[dict[str, str]]:
        """Return only recovery declaration values that change public state."""

        if (
            not isinstance(self.runtime.dataset_id, str)
            or self.runtime.dataset_id.casefold() != "triviaqa"
            or not self._graph.has_node(agent_id)
        ):
            return None
        node = self._graph.get_node(agent_id)
        if (node.role_family or "").casefold() != "evidence_retriever":
            return None
        candidates = {
            "contract": _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT,
            "completion_condition": _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION,
        }
        return {
            field_name: value
            for field_name, value in candidates.items()
            if getattr(node, field_name) != value
        }

    def _triviaqa_location_reasoner_recovery_field_values(
        self,
        agent_id: str,
    ) -> Optional[dict[str, str]]:
        """Return the bounded public-containment Reasoner repair values.

        The QA adapter remains authoritative for the typed location-lineage
        diagnosis.  This Canvas projection only closes the measured location
        repair boundary after the responsible Reasoner has a successful public
        read whose body explicitly states city/town containment.  It consumes
        neither Ground Truth nor evaluator state and does not prescribe a
        workflow topology.
        """

        if (
            not isinstance(self.runtime.dataset_id, str)
            or self.runtime.dataset_id.casefold() != "triviaqa"
            or not self._graph.has_node(agent_id)
        ):
            return None
        node = self._graph.get_node(agent_id)
        if (node.role_family or "").casefold() != "reasoner":
            return None
        record = self._latest_failure_record_by_agent.get(agent_id)
        if record is None:
            return None
        public_summary = self._react_public_error_summary(record)
        raw_code_counts = public_summary.get("public_error_code_counts", {})
        if not (
            isinstance(raw_code_counts, Mapping)
            and any(
                isinstance(code, str)
                and "qa_location_containment_lineage_missing" in code
                for code in raw_code_counts
            )
        ):
            return None
        raw_trace = record.metadata.get("react_trace", ())
        trace = (
            tuple(item for item in raw_trace if isinstance(item, Mapping))
            if isinstance(raw_trace, (list, tuple))
            else ()
        )
        completion_action: Mapping[str, object] | None = None
        for entry in reversed(trace):
            observation = entry.get("observation")
            source = observation if isinstance(observation, Mapping) else entry
            public_error_code = source.get("public_error_code")
            if not (
                isinstance(public_error_code, str)
                and "qa_location_containment_lineage_missing"
                in public_error_code
            ):
                continue
            candidate_action = entry.get("structured_action")
            if not isinstance(candidate_action, Mapping):
                candidate_action = source.get("structured_action")
            if isinstance(candidate_action, Mapping):
                completion_action = candidate_action
                break
        if completion_action is None:
            return {}
        raw_receipts = record.metadata.get("tool_receipts", ())
        receipts = (
            tuple(item for item in raw_receipts if isinstance(item, Mapping))
            if isinstance(raw_receipts, (list, tuple))
            else ()
        )
        if self.required_evidence_tool_id is None:
            return {}
        read_texts = tuple(
            read_text
            for receipt in receipts
            for read_text in (
                self._successful_read_text(
                    receipt,
                    self.required_evidence_tool_id,
                ),
            )
            if read_text is not None
        )
        from .qa_tool_adapter import (
            _location_containment_repair_anchor,
            _location_resolution_answer_field_constraint,
        )

        original_question = hotpotqa_question_scope(self._problem)
        entity_anchor = _location_containment_repair_anchor(
            original_question=original_question,
            completion_action=completion_action,
        )
        if not isinstance(entity_anchor, str) or not entity_anchor.strip():
            return {}
        answer_field = _location_resolution_answer_field_constraint(
            original_question=original_question,
            entity_anchor=entity_anchor,
            read_evidence_texts=read_texts,
        )
        if answer_field != "object_or_attribute_value":
            return {}
        candidates = {
            "contract": _QA_LOCATION_REASONER_RECOVERY_CONTRACT,
            "completion_condition": _QA_LOCATION_REASONER_RECOVERY_COMPLETION,
        }
        return {
            field_name: value
            for field_name, value in candidates.items()
            if getattr(node, field_name) != value
        }

    def _triviaqa_react_recovery_field_values(
        self,
        agent_id: str,
    ) -> Optional[dict[str, str]]:
        """Reuse the existing discrete ReAct repair domain by role/state."""

        evidence_values = (
            self._triviaqa_evidence_retriever_recovery_field_values(agent_id)
        )
        if evidence_values is not None:
            return evidence_values
        return self._triviaqa_location_reasoner_recovery_field_values(agent_id)

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
        if not self._provider_repair_required(action.agent_id):
            return None
        if not admitted_model_ids:
            return (
                "provider failure repair has no catalog-backed alternative "
                "model_id; modify_agent is outside the live action domain"
            )
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
        if (
            len(mutable_fields) != 1
            or mutable_fields[0] not in {"contract", "completion_condition"}
        ):
            return (
                "a measured ReAct repair must modify exactly one of contract "
                "or completion_condition while preserving the Agent model, "
                "role, tools, execution mode, artifact type, and relations"
            )
        node = self._graph.get_node(action.agent_id)
        recovery_values = self._triviaqa_react_recovery_field_values(
            action.agent_id
        )
        if recovery_values is not None:
            field_name = mutable_fields[0]
            expected_value = recovery_values.get(field_name)
            if (
                expected_value is None
                or getattr(action, field_name) != expected_value
            ):
                if (node.role_family or "").casefold() == "reasoner":
                    return (
                        "a measured location-containment Reasoner repair must "
                        "use one live receipt-conditioned contract or "
                        "completion_condition recovery value and preserve the "
                        "public first-hop and containment propositions"
                    )
                return (
                    "a measured Evidence Retriever repair must preserve its "
                    "evidence-only responsibility and use the live "
                    f"{field_name} recovery value"
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
        attribution = self._semantic_repair_attribution(semantic_issue)
        if attribution is None:
            return ()
        agent_id = attribution.get("responsible_agent_id")
        if not isinstance(agent_id, str) or not self._graph.has_node(agent_id):
            return ()
        if agent_id in self._repair_exhausted_agent_ids:
            return ()
        return (agent_id,)

    def _semantic_failure_progress_state(
        self,
        agent_id: str,
    ) -> Optional[tuple[str, Tuple[Tuple[str, str], ...]]]:
        """Return the current attributed fault and public read-receipt state."""

        execution = self._cached_progressive_execution()
        if execution is None:
            return None
        issue = self._semantic_protocol_issue(execution)
        if issue is None:
            return None
        attribution = self._semantic_repair_attribution(issue)
        if (
            attribution is None
            or attribution.get("responsible_agent_id") != agent_id
        ):
            return None
        receipt_keys: set[tuple[str, str]] = set()

        def collect(value: object) -> None:
            if isinstance(value, Mapping):
                if (
                    self.required_evidence_tool_id is not None
                    and self._successful_read_receipt(
                        value,
                        self.required_evidence_tool_id,
                    )
                ):
                    request = value.get("request")
                    assert isinstance(request, Mapping)
                    arguments = request.get("arguments")
                    record_id_field = (
                        "memory_id"
                        if self.required_evidence_tool_id
                        == _TRIVIAQA_QA_MEMORY_TOOL_ID
                        else "passage_id"
                    )
                    passage_id = (
                        arguments.get(record_id_field)
                        if isinstance(arguments, Mapping)
                        else None
                    )
                    if isinstance(passage_id, str) and passage_id:
                        receipt_keys.add(
                            (self.required_evidence_tool_id, passage_id)
                        )
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(execution.output_metadata)
        return issue, tuple(sorted(receipt_keys))

    def _stalled_semantic_repair_agent_ids(self) -> Tuple[str, ...]:
        """Return semantic repair targets exhausted without new Tool evidence."""

        result: list[str] = []
        for agent_id in self._repair_exhausted_agent_ids:
            if (
                self._graph.has_node(agent_id)
                and (
                    self._graph.get_node(agent_id).role_family or ""
                ).casefold()
                in {"reasoner", "verifier", "format"}
                and self._semantic_failure_progress_state(agent_id) is not None
            ):
                result.append(agent_id)
        return tuple(sorted(result))

    def _mandatory_repair_agent_ids(self) -> Tuple[str, ...]:
        """Project the repair-first Agent domain for the current Canvas state."""

        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return ()
        node_ids = tuple(node.id for node in self._graph.nodes)
        unavailable_model_agents = {
            node.id
            for node in self._graph.nodes
            if node.model_id in self._unavailable_model_ids
        }
        if unavailable_model_agents:
            return tuple(
                node_id
                for node_id in node_ids
                if node_id in unavailable_model_agents
            )
        repairable_failed = (
            self._failed_agent_ids - self._diagnosed_unusable_agent_ids
            - self._repair_exhausted_agent_ids
        ).intersection(node_ids)
        if repairable_failed:
            return tuple(
                node_id for node_id in node_ids if node_id in repairable_failed
            )
        return self._semantic_artifact_repair_agent_ids()

    def _agent_has_successful_read_receipt(self, agent_id: str) -> bool:
        """Return whether any retained public state contains a successful read."""

        if self.required_evidence_tool_id is None:
            return False

        def contains_receipt(value: object) -> bool:
            if isinstance(value, Mapping):
                if self._successful_read_receipt(
                    value,
                    self.required_evidence_tool_id,
                ):
                    return True
                return any(contains_receipt(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains_receipt(item) for item in value)
            return False

        retained_public_state: list[object] = []
        for metadata_by_agent in (
            self._progressive_output_metadata,
            self._previous_revision_output_metadata,
            self._failure_continuations,
        ):
            metadata = metadata_by_agent.get(agent_id)
            if metadata is not None:
                retained_public_state.append(metadata)
        latest_failure = self._latest_failure_record_by_agent.get(agent_id)
        if latest_failure is not None:
            retained_public_state.append(latest_failure.metadata)
        return any(contains_receipt(item) for item in retained_public_state)

    def _auxiliary_retrieval_progress_tokens(
        self,
        agent_id: str,
    ) -> frozenset[tuple[str, str]]:
        """Return answer-free public retrieval progress for one failed Agent."""

        record = self._latest_failure_record_by_agent.get(agent_id)
        if record is None:
            return frozenset()
        result: set[tuple[str, str]] = set()
        diagnosis = self._terminal_retrieval_failure_diagnosis(record)
        if diagnosis is not None:
            schedule = diagnosis.get("retrieval_strategy_schedule_prefix", ())
            if isinstance(schedule, (list, tuple)):
                result.update(
                    ("strategy", value)
                    for value in schedule
                    if isinstance(value, str) and value
                )
            retrieval_attempts = diagnosis.get("retrieval_attempts", ())
            if isinstance(retrieval_attempts, (list, tuple)):
                for attempt in retrieval_attempts:
                    if (
                        not isinstance(attempt, Mapping)
                        or attempt.get("verified") is not True
                        or attempt.get("recall_expansion") is not True
                    ):
                        continue
                    term_set = attempt.get("fts_term_set")
                    observed_top_k = attempt.get("observed_top_k")
                    if (
                        not isinstance(term_set, (list, tuple))
                        or not term_set
                        or not all(
                            isinstance(token, str) and bool(token)
                            for token in term_set
                        )
                        or type(observed_top_k) is not int
                        or observed_top_k < 1
                    ):
                        continue
                    # FlowSteer's next Canvas edit is admitted only after a
                    # new public execution result. A verified SkillFlow
                    # same-FTS-term-set search at a strictly larger top-k is
                    # such a result even though, correctly, it does not claim
                    # completion of another retrieval strategy.
                    result.add(
                        (
                            "recall_expansion",
                            json.dumps(
                                {
                                    "fts_term_set": list(term_set),
                                    "observed_top_k": observed_top_k,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    )

        if self.required_evidence_tool_id is None:
            return frozenset(result)

        def collect(value: object) -> None:
            if isinstance(value, Mapping):
                if self._successful_read_receipt(
                    value,
                    self.required_evidence_tool_id,
                ):
                    request = value.get("request")
                    arguments = (
                        request.get("arguments")
                        if isinstance(request, Mapping)
                        else None
                    )
                    record_id_field = (
                        "memory_id"
                        if self.required_evidence_tool_id
                        == _TRIVIAQA_QA_MEMORY_TOOL_ID
                        else "passage_id"
                    )
                    passage_id = (
                        arguments.get(record_id_field)
                        if isinstance(arguments, Mapping)
                        else None
                    )
                    if isinstance(passage_id, str) and passage_id:
                        result.add(("read", passage_id))
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(record.metadata)
        return frozenset(result)

    def _auxiliary_replacement_has_strict_progress_opportunity(
        self,
        source_id: str,
        previous_source_ids: Sequence[str],
    ) -> bool:
        """Gate another bounded replacement on measured public progress.

        A typed knowledge-base coverage failure is terminal for this frozen
        Tool/catalog condition.  An incomplete retrieval schedule may receive
        another isolated FlowSteer ADD boundary only when its latest generation
        has added a strategy or successful read receipt not already measured in
        earlier same-role/same-artifact generations.  This preserves
        SkillFlow's public Action--Observation boundary without restarting a
        terminal schedule merely because Agent capacity remains.
        """

        record = self._latest_failure_record_by_agent.get(source_id)
        if record is None:
            return False
        category, _, _ = self._execution_failure_diagnosis(record)
        if category == "react_turn_exhaustion":
            diagnosis = self._terminal_retrieval_failure_diagnosis(record)
            remaining_tool_calls = (
                diagnosis.get("remaining_tool_calls")
                if diagnosis is not None
                else None
            )
            if (
                diagnosis is not None
                and diagnosis.get("react_turn_exhausted") is True
                and diagnosis.get("continuation_admissible") is True
                and diagnosis.get("tool_plan_exhausted") is False
                and diagnosis.get("bounded_schedule_exhausted") is False
                and isinstance(remaining_tool_calls, int)
                and not isinstance(remaining_tool_calls, bool)
                and remaining_tool_calls > 0
                and not self._tool_continuation_exhausted(record.metadata)
            ):
                current_progress = self._auxiliary_retrieval_progress_tokens(
                    source_id
                )
                if not current_progress:
                    return False
                previous_progress = frozenset().union(
                    *(
                        self._auxiliary_retrieval_progress_tokens(agent_id)
                        for agent_id in previous_source_ids
                    )
                )
                return not previous_source_ids or bool(
                    current_progress - previous_progress
                )
        if (
            isinstance(self.runtime.dataset_id, str)
            and self.runtime.dataset_id.casefold() == "triviaqa"
            and category == "react_turn_exhaustion"
            and not previous_source_ids
            and not self._tool_continuation_exhausted(record.metadata)
        ):
            # The Retriever may have completed search/read while exhausting
            # its bounded Action--Observation loop on only the structured
            # evidence artifact.  Preserve those public receipts and admit one
            # isolated same-role replacement to finish the artifact; do not
            # open a second replacement generation on unchanged receipts.
            public_summary = self._react_public_error_summary(record)
            last_public_error = public_summary.get("last_public_error", {})
            public_code = (
                last_public_error.get("public_error_code")
                if isinstance(last_public_error, Mapping)
                else None
            )
            return bool(
                public_summary.get("successful_evidence_read_count", 0)
                and isinstance(last_public_error, Mapping)
                and last_public_error.get("observation_status")
                == "schema_invalid"
                and isinstance(public_code, str)
                and public_code.startswith("qa_semantic_artifact_invalid:")
                and isinstance(
                    last_public_error.get("repair_instruction"), str
                )
            )
        if category == "knowledge_base_coverage_failure":
            return False
        if category not in {
            "retrieval_recall_failure",
            "retrieval_strategy_failure",
        }:
            return False
        diagnosis = self._terminal_retrieval_failure_diagnosis(record)
        if (
            diagnosis is None
            or diagnosis.get("bounded_schedule_exhausted") is not False
        ):
            return False
        current_progress = self._auxiliary_retrieval_progress_tokens(source_id)
        if not current_progress:
            return False
        previous_progress = frozenset().union(
            *(
                self._auxiliary_retrieval_progress_tokens(agent_id)
                for agent_id in previous_source_ids
            )
        )
        return not previous_source_ids or bool(
            current_progress - previous_progress
        )

    def _capacity_blocks_required_augmentation(self) -> bool:
        """Return whether the Agent cap alone blocks the next semantic ADD."""

        if (
            not self._uses_semantic_lineage_protocol()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            or self.max_agents is None
            or len(self._graph.nodes) < self.max_agents
        ):
            return False
        if self._missing_semantic_role_families():
            return True
        exhausted_reasoner_ids = self._repair_exhausted_reasoner_ids()
        if not exhausted_reasoner_ids:
            return False
        # Existing exact relation/output progress takes precedence over
        # capacity recovery. DELETE is needed only when the next executable
        # recovery unit must be an augmentation and the cap is its sole blocker.
        return not (
            self._failed_auxiliary_ingress_relation_candidates()
            or self._repair_exhausted_relation_candidates()
            or self._required_semantic_relation_candidates()
            or self._model_admissible_output_agent_ids()
        )

    def _capacity_blocking_failed_auxiliary_delete_ids(self) -> Tuple[str, ...]:
        """Return artifact-free isolated auxiliaries removable to admit one ADD.

        This is a narrow FlowSteer Canvas capacity-recovery boundary. It does
        not infer redundancy from role count or topology preference: every
        public preservation predicate must prove that DELETE cannot discard an
        artifact, Tool read, message edge, Output identity, or active semantic
        lineage. The accepted DELETE remains in ordinary Canvas history.
        """

        if not self._capacity_blocks_required_augmentation():
            return ()
        active_lineage = set(self._active_semantic_lineage_ids())
        result: list[str] = []
        for node in self._graph.nodes:
            agent_id = node.id
            role_family = (node.role_family or "").casefold()
            latest_failure = self._latest_failure_record_by_agent.get(agent_id)
            if (
                role_family not in {"evidence_retriever", "repair"}
                or agent_id not in self._failed_agent_ids
                or agent_id not in self._repair_exhausted_agent_ids
                or latest_failure is None
                or self._execution_failure_diagnosis(latest_failure)[0]
                not in _BOUNDED_REACT_FAILURE_CATEGORIES
                or agent_id in active_lineage
                or self._graph.output_agent_id == agent_id
                or self._graph.directed_predecessors(agent_id)
                or self._directed_successors(self._graph, agent_id)
                or (
                    isinstance(self._progressive_outputs.get(agent_id), str)
                    and bool(self._progressive_outputs[agent_id].strip())
                )
                or (
                    isinstance(self._previous_revision_outputs.get(agent_id), str)
                    and bool(self._previous_revision_outputs[agent_id].strip())
                )
                or self._agent_has_successful_read_receipt(agent_id)
            ):
                continue
            result.append(agent_id)
        return tuple(result)

    def _repair_exhausted_auxiliary_takeover_delete_ids(self) -> Tuple[str, ...]:
        """Return bounded failed auxiliaries whose replacement already took over."""

        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return ()
        capacity_recovery_ids = set(
            self._capacity_blocking_failed_auxiliary_delete_ids()
        )
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold()
            in {"evidence_retriever", "repair"}
            and node.id in self._failed_agent_ids
            and node.id in self._repair_exhausted_agent_ids
            and (
                record := self._latest_failure_record_by_agent.get(node.id)
            )
            is not None
            and self._execution_failure_diagnosis(record)[0]
            in _BOUNDED_REACT_FAILURE_CATEGORIES
            and node.id not in capacity_recovery_ids
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
        if (
            self._uses_role_conditional_capabilities()
            and not self._requires_complete_semantic_lineage()
        ):
            return tuple(
                role_family
                for role_family in (*role_families, "output")
                if role_family != "format"
                or (
                    not self._semantic_role_agent_ids("format")
                    and self._graph.output_agent_id is None
                )
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
        if not self._requires_complete_semantic_lineage():
            return ()
        required_role_families = (
            (
                "evidence_retriever",
                "reasoner",
                "verifier",
                "format",
            )
            if self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
            else ("reasoner", "verifier", "format")
        )
        return tuple(
            role_family
            for role_family in required_role_families
            if not self._semantic_role_agent_ids(role_family)
        )

    def _model_admissible_add_role_families(self) -> Tuple[str, ...]:
        """Project the exact live ADD role domain used by Canvas admission."""

        admitted = self._admissible_augmentation_role_families()
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        if replacement_domains:
            # A bounded failed Retriever with preserved public Tool state owns
            # the next execute-and-feedback boundary.  Complete that existing
            # evidence responsibility before exposing downstream semantic
            # roles; this is state-conditioned admission, not a fixed topology.
            return tuple(
                role for role in admitted if role in replacement_domains
            )
        if (
            isinstance(self.runtime.dataset_id, str)
            and self.runtime.dataset_id.casefold() == "triviaqa"
            and any(
                auxiliary_id in self._failed_agent_ids
                and auxiliary_id in self._repair_exhausted_agent_ids
                for auxiliary_id in self._recovery_auxiliary_agent_ids()
            )
            and not self._has_valid_evidence_retriever_artifact()
        ):
            # No current receipt-grounded evidence artifact can satisfy the
            # downstream lineage, and the bounded same-responsibility
            # replacement domain is closed. Do not expose missing semantic
            # roles as a bypass around the Evidence Grounding gate.
            return ()
        missing = self._missing_semantic_role_families()
        if (
            self._uses_semantic_lineage_protocol()
            and self._graph.nodes
            and missing
        ):
            # Complete the responsibilities still absent from a partial
            # semantic Canvas before admitting an auxiliary branch. The live
            # domain contains responsibilities, not a prescribed topology or
            # fixed Agent sequence.
            return tuple(role for role in admitted if role in missing)
        if self._repair_exhausted_reasoner_ids():
            # A recovery branch must materialize evidence before it is wired
            # into the exhausted Reasoner. A reasoning-only Repair Agent has
            # no admissible input while isolated, so it is not an executable
            # first augmentation unit.
            if self._requires_isolated_reasoner_augmentation():
                return tuple(
                    role for role in admitted if role == "evidence_retriever"
                )
            return ()
        return admitted

    def _repair_exhausted_auxiliary_replacement_domains(
        self,
    ) -> dict[str, Tuple[str, ...]]:
        """Return same-role/artifact domains with strict public progress."""

        capacity_recovery_ids = set(
            self._capacity_blocking_failed_auxiliary_delete_ids()
        )
        domains: dict[str, list[str]] = {}
        failed_generations: dict[tuple[str, str], list[str]] = {}
        for node in self._graph.nodes:
            role_family = (node.role_family or "").casefold()
            if (
                role_family not in {"evidence_retriever", "repair"}
                or node.id not in self._failed_agent_ids
                or node.id not in self._repair_exhausted_agent_ids
            ):
                continue
            artifact_type = node.artifact_type.casefold()
            failed_generations.setdefault(
                (role_family, artifact_type),
                [],
            ).append(node.id)

        for (role_family, artifact_type), agent_ids in failed_generations.items():
            # Only the newest failed generation can justify another
            # replacement. An older incomplete schedule must not remain a
            # permanently reusable ADD ticket after a newer generation stalls.
            source_id = agent_ids[-1]
            if (
                source_id in capacity_recovery_ids
                or self._delete_admission_issue(source_id) is None
                or not self._auxiliary_replacement_has_strict_progress_opportunity(
                    source_id,
                    agent_ids[:-1],
                )
            ):
                continue
            artifact_types = domains.setdefault(role_family, [])
            if artifact_type not in artifact_types:
                artifact_types.append(artifact_type)
        return {
            role_family: tuple(artifact_types)
            for role_family, artifact_types in domains.items()
        }

    def _requires_isolated_reasoner_augmentation(self) -> bool:
        """Return whether recovery needs one newly executed evidence branch.

        FlowSteer's accepted Canvas edit is followed by execution before the
        next policy turn. When a complete semantic spine has an exhausted
        Reasoner and no successful auxiliary artifact can yet be routed, an
        added Evidence Retriever is first executed as one isolated functional
        unit. A later SET_RELATION edit routes only a validated artifact. This
        is a state-conditioned execution boundary, not a prompt-level topology
        prescription.
        """

        return bool(
            self._uses_semantic_lineage_protocol()
            and self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY
            and len(self._repair_exhausted_reasoner_ids()) == 1
            and self._reasoner_failure_requires_evidence_augmentation(
                self._repair_exhausted_reasoner_ids()[0]
            )
            and not self._missing_semantic_role_families()
            and not any(
                auxiliary_id in self._failed_agent_ids
                and auxiliary_id in self._repair_exhausted_agent_ids
                for auxiliary_id in self._recovery_auxiliary_agent_ids()
            )
            and not self._pending_isolated_recovery_auxiliary_agent_ids()
            and not self._repair_exhausted_auxiliary_replacement_domains()
            and not self._repair_exhausted_relation_candidates()
        )

    def _pending_isolated_recovery_auxiliary_agent_ids(
        self,
    ) -> Tuple[str, ...]:
        """Return one admitted recovery unit that still lacks an artifact.

        FlowSteer's next edit is admitted only after the preceding Canvas edit
        has executed.  An isolated recovery Retriever which was cancelled by
        sibling fail-fast or stopped on a repairable provider/contract failure
        still owns that execution boundary.  It must be repaired and executed
        before another equivalent augmentation is admitted.  The predicate is
        derived only from public Runtime state and does not prescribe a fixed
        AgentGraph topology.
        """

        if not self._repair_exhausted_reasoner_ids():
            return ()
        mandatory_repair_ids = set(self._mandatory_repair_agent_ids())
        return tuple(
            node.id
            for node in self._graph.nodes
            if (node.role_family or "").casefold()
            in {"evidence_retriever", "repair"}
            and node.id not in self._repair_exhausted_agent_ids
            and not self._has_successful_artifact(node.id)
            and not self._graph.directed_predecessors(node.id)
            and not self._directed_successors(self._graph, node.id)
            and (
                node.id in self._unresolved_dirty_agents
                or node.id in mandatory_repair_ids
            )
        )

    def _dirty_auxiliary_replacement_agent_ids(self) -> Tuple[str, ...]:
        """Return repairable isolated recovery units awaiting an artifact.

        The replacement is an active SkillFlow-style bounded Agent execution,
        not spare AgentGraph capacity.  It therefore blocks another recovery
        augmentation as soon as it is admitted, independently of max_agents.
        Once that replacement is itself repair-exhausted, it no longer owns a
        legal MODIFY step; the existing FlowSteer ADD boundary can admit the
        next same-role/same-artifact isolated replacement instead.
        """

        pending_recovery_ids = set(
            self._pending_isolated_recovery_auxiliary_agent_ids()
        )
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        if not replacement_domains and not pending_recovery_ids:
            return ()
        return tuple(
            node.id
            for node in self._graph.nodes
            if (
                node.id in pending_recovery_ids
                or (
                    (node.role_family or "").casefold()
                    in replacement_domains
                    and node.artifact_type.casefold()
                    in replacement_domains[
                        (node.role_family or "").casefold()
                    ]
                )
            )
            and node.id in self._unresolved_dirty_agents
            and node.id not in self._repair_exhausted_agent_ids
            and not self._graph.directed_predecessors(node.id)
            and not self._directed_successors(self._graph, node.id)
        )

    def _isolated_auxiliary_execution_scope(
        self,
        candidate: AgentGraph,
        action: AgentAction,
    ) -> Tuple[str, ...]:
        """Return the edit-local bounded replacement execution, if any.

        FlowSteer executes every accepted Canvas edit before the next policy
        turn. SkillFlow executes one bounded Agent from its own public
        Action--Observation continuation. A same-role auxiliary replacement is
        deliberately admitted with no edges or Output identity, so execute
        that one functional unit without rescheduling unrelated historical
        failed roots. Ordinary ADD/MODIFY and all routed graph edits retain the
        full AgentGraph Runtime path.
        """

        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        reasoner_augmentation = (
            self._requires_isolated_reasoner_augmentation()
        )
        dirty_auxiliary_ids = set(
            self._dirty_auxiliary_replacement_agent_ids()
        )
        if (
            not replacement_domains
            and not reasoner_augmentation
            and not dirty_auxiliary_ids
        ):
            return ()
        target_id: Optional[str] = None
        if (
            action.action_type is AgentActionType.ADD_SUBGRAPH
            and len(action.agents) == 1
            and not action.relations
            and action.output_agent_id is None
        ):
            declaration = action.agents[0]
            role_family = (declaration.role_family or "").casefold()
            artifact_type = (declaration.artifact_type or "text").casefold()
            if (
                artifact_type in replacement_domains.get(role_family, ())
                or (
                    reasoner_augmentation
                    and role_family == "evidence_retriever"
                )
            ):
                target_id = declaration.agent_id
        elif (
            action.action_type is AgentActionType.MODIFY_AGENT
            and action.agent_id in dirty_auxiliary_ids
        ):
            target_id = action.agent_id
        if target_id is None or not candidate.has_node(target_id):
            return ()
        if candidate.directed_predecessors(target_id) or self._directed_successors(
            candidate,
            target_id,
        ):
            return ()
        return (target_id,)

    @staticmethod
    def _single_agent_execution_graph(
        graph: AgentGraph,
        agent_id: str,
    ) -> AgentGraph:
        """Project one isolated Canvas node through the existing Runtime."""

        return AgentGraph(
            nodes=(graph.get_node(agent_id),),
            revision=graph.revision,
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
            for model_id in self._available_model_ids()
            if model_id != current_model_id
        )
        cross_provider = tuple(
            model_id
            for model_id in alternatives
            if self.model_registry.provider_for(model_id).provider_id
            != current_provider_id
        )
        return cross_provider or alternatives

    def _provider_repair_required(self, agent_id: str) -> bool:
        """Return whether one Agent has a typed provider/model failure.

        This separates the failure diagnosis from the availability of a
        replacement arm.  In a one-model catalog the diagnosis remains true
        while the repair domain is empty; treating an empty domain as a
        semantic contract repair would violate the live FlowSteer action mask.
        """

        if not self._graph.has_node(agent_id):
            return False
        current_model_id = self._graph.get_node(agent_id).model_id
        if current_model_id in self._unavailable_model_ids:
            return True
        record = self._latest_failure_record_by_agent.get(agent_id)
        if record is None:
            return False
        category, retryability, _ = self._execution_failure_diagnosis(record)
        return (
            category == "provider_request_failure"
            and retryability
            in {"transient_provider", "permanent_configuration"}
        )

    def _available_model_ids(self) -> Tuple[str, ...]:
        return tuple(
            model_id
            for model_id in self.model_registry.model_ids
            if model_id not in self._unavailable_model_ids
        )

    def model_availability_receipt(self) -> dict[str, object]:
        """Return the trajectory-local availability overlay and its evidence."""

        unavailable_providers = tuple(
            provider_id
            for provider_id in self.model_registry.provider_ids
            if (
                provider_models := tuple(
                    model_id
                    for model_id in self.model_registry.model_ids
                    if self.model_registry.provider_for(model_id).provider_id
                    == provider_id
                )
            )
            and all(
                model_id in self._unavailable_model_ids
                for model_id in provider_models
            )
        )
        return {
            "scope": "trajectory",
            "catalog_mutated": False,
            "available_model_ids": list(self._available_model_ids()),
            "unavailable_model_ids": sorted(self._unavailable_model_ids),
            "unavailable_provider_ids": list(unavailable_providers),
            "failure_receipts": [
                dict(item) for item in self._model_availability_receipts
            ],
        }

    def _record_model_unavailability(
        self,
        record: AgentFailureRecord,
        *,
        category: str,
        retryability: str,
        status_code: Optional[int],
    ) -> None:
        """Quarantine one exact model after a permanent public rejection."""

        if (
            category != "provider_request_failure"
            or retryability != "permanent_configuration"
            or status_code not in {401, 403, 404}
            or not self._graph.has_node(record.agent_id)
        ):
            return
        metadata_model_id = record.metadata.get("model_id")
        model_id = (
            metadata_model_id
            if isinstance(metadata_model_id, str)
            and metadata_model_id in self.model_registry.model_ids
            else self._graph.get_node(record.agent_id).model_id
        )
        provider_id = self.model_registry.provider_for(model_id).provider_id
        if model_id in self._unavailable_model_ids:
            return
        self._unavailable_model_ids.add(model_id)
        self._model_availability_receipts.append(
            {
                "agent_id": record.agent_id,
                "model_id": model_id,
                "provider_id": provider_id,
                "http_status": status_code,
                "failure_category": category,
                "retryability": retryability,
                "graph_revision": record.graph_revision,
            }
        )

    def _provider_repair_model_ids(self, agent_id: str) -> Tuple[str, ...]:
        """Return the typed provider-failure model repair domain.

        SkillFlow keeps provider identity separate from model identity.  A
        measured provider failure therefore changes only the model field and,
        when the catalog offers one, prefers a different provider.  Falling
        back to another exact model on the same provider keeps recovery live
        for single-provider catalogs.
        """

        if not self._graph.has_node(agent_id):
            return ()
        current_model_id = self._graph.get_node(agent_id).model_id
        if not self._provider_repair_required(agent_id):
            return ()
        return self._provider_repair_catalog_domain(current_model_id)

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
        role_conditional_capabilities = bool(
            self._uses_role_conditional_capabilities()
            and not self._requires_complete_semantic_lineage()
        )
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        isolated_reasoner_augmentation = (
            self._requires_isolated_reasoner_augmentation()
        )
        evidence_retriever_recovery = bool(
            isolated_reasoner_augmentation
            or "evidence_retriever" in replacement_domains
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
            if replacement_domains or isolated_reasoner_augmentation:
                # One same-role/same-artifact auxiliary replacement is one
                # executable Canvas unit. Keep constrained decoding equal to
                # the authoritative admission boundary below.
                remaining = min(remaining, 1)
            missing_role_families = self._missing_semantic_role_families()
            if (
                self._graph.nodes
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
                        "require_format_agent": self.require_format_agent,
                        "existing_agents": [
                            {
                                "agent_id": node.id,
                                "role_family": node.role_family,
                            }
                            for node in self._graph.nodes
                        ],
                        "preserved_input_agent_ids": list(
                            self._preserved_input_agent_ids()
                        ),
                        "current_output_agent_id": self._graph.output_agent_id,
                        **(
                            {"output_role_families": ["format", "output"]}
                            if role_conditional_capabilities
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
                        "model_ids": list(self._available_model_ids()),
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
                            if role_conditional_capabilities
                            else {}
                        ),
                        "role_constraints": {
                            "reasoner": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "reasoner"
                                    )
                                    if role_conditional_capabilities
                                    else {
                                        "execution_modes": ["react"],
                                        "allowed_tools": [
                                            [self.required_evidence_tool_id]
                                        ],
                                    }
                                ),
                                "contract_responsibility": (
                                    _QA_ROLE_CONTRACT_RESPONSIBILITIES["reasoner"]
                                ),
                            },
                            "verifier": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "verifier"
                                    )
                                    if role_conditional_capabilities
                                    else {
                                        "execution_modes": ["reasoning"],
                                        "allowed_tools": [[]],
                                    }
                                ),
                                "contract_responsibility": (
                                    _QA_ROLE_CONTRACT_RESPONSIBILITIES["verifier"]
                                ),
                            },
                            "format": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "format"
                                    )
                                    if role_conditional_capabilities
                                    else {
                                        "execution_modes": ["reasoning"],
                                        "allowed_tools": [[]],
                                    }
                                ),
                                "contracts": [
                                    _QA_ROLE_CONDITIONAL_FORMAT_CONTRACT
                                    if role_conditional_capabilities
                                    else _HOTPOTQA_FORMAT_CONTRACT
                                ],
                                "contract_responsibility": (
                                    _QA_ROLE_CONTRACT_RESPONSIBILITIES["format"]
                                ),
                                **(
                                    {"must_be_output_agent": True}
                                    if role_conditional_capabilities
                                    else {}
                                ),
                            },
                            "evidence_retriever": {
                                **(
                                    self._role_conditional_execution_constraint(
                                        "evidence_retriever"
                                    )
                                    if role_conditional_capabilities
                                    else {
                                        "execution_modes": ["react"],
                                        "allowed_tools": [
                                            [self.required_evidence_tool_id]
                                        ],
                                    }
                                ),
                                "contract_responsibility": (
                                    _QA_ROLE_CONTRACT_RESPONSIBILITIES[
                                        "evidence_retriever"
                                    ]
                                ),
                                **(
                                    {
                                        "contracts": [
                                            _QA_EVIDENCE_RETRIEVER_RECOVERY_CONTRACT
                                        ],
                                        "completion_conditions": [
                                            _QA_EVIDENCE_RETRIEVER_RECOVERY_COMPLETION
                                        ],
                                    }
                                    if evidence_retriever_recovery
                                    else {}
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
                                    if role_conditional_capabilities
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
                            **(
                                {
                                    "output": (
                                        self._role_conditional_execution_constraint(
                                            "output"
                                        )
                                    )
                                }
                                if role_conditional_capabilities
                                else {}
                            ),
                        },
                        "admitted_new_role_families": list(
                            self._model_admissible_add_role_families()
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
                                "output_agent_id": None,
                            }
                            if (
                                replacement_domains
                                or isolated_reasoner_augmentation
                            )
                            else {}
                        ),
                    }
                    if self._uses_semantic_lineage_protocol()
                    else {
                        # NECESSARY_ADAPTATION: the generic QA-memory profile
                        # keeps semantic_protocol=none, but v3 still needs the
                        # live domains that prevent invented Agent IDs and bind
                        # mode/Tool capability atomically.  Roles, contracts,
                        # Agent count, relations, and topology remain sampled.
                        "topology_neutral": True,
                        "required_agent_fields": [
                            "agent_id",
                            "model_id",
                            "contract",
                            "role_family",
                            "allowed_tools",
                            "execution_mode",
                        ],
                        "model_ids": list(self._available_model_ids()),
                        "registered_execution_profiles": [
                            {
                                "execution_mode": execution_mode,
                                "allowed_tools": list(allowed_tools),
                            }
                            for execution_mode, allowed_tools in (
                                self._topology_neutral_registered_execution_profiles()
                            )
                        ],
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
                if self._provider_repair_required(agent_id)
            }
            dirty_replacement_ids = set(
                self._dirty_auxiliary_replacement_agent_ids()
            )
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
            # A model-admissible MODIFY target owns the next executable repair
            # boundary even when the original measured failure is an exhausted
            # upstream Agent.  Keep the target domain and attribution domain
            # consistent for hierarchical constrained decoding.
            responsible_ids.update(modifiable_node_ids)
            per_agent_model_domains = {
                agent_id: list(
                    self._provider_repair_model_ids(agent_id)
                    if agent_id in provider_failure_agent_ids
                    else tuple(
                        model_id
                        for model_id in self._available_model_ids()
                        if model_id != self._graph.get_node(agent_id).model_id
                    )
                )
                for agent_id in modifiable_node_ids
            }
            per_agent_recovery_field_values = {
                agent_id: (
                    self._triviaqa_react_recovery_field_values(agent_id)
                    if agent_id in self._react_exhausted_agent_ids
                    else None
                )
                for agent_id in modifiable_node_ids
            }
            per_agent_mutable_fields = {
                agent_id: [
                    field
                    for field in (
                        ["model_id"]
                        if agent_id in provider_failure_agent_ids
                        else list(
                            per_agent_recovery_field_values[agent_id]
                        )
                        if per_agent_recovery_field_values[agent_id]
                        is not None
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
                    and not (
                        field == "model_id"
                        and not per_agent_model_domains[agent_id]
                    )
                ]
                for agent_id in modifiable_node_ids
            }
            per_agent_current_values: dict[str, dict[str, object]] = {}
            for agent_id, fields in per_agent_mutable_fields.items():
                node = self._graph.get_node(agent_id)
                current_values: dict[str, object] = {}
                for field in fields:
                    value = getattr(node, field)
                    current_values[field] = (
                        list(value) if isinstance(value, tuple) else value
                    )
                per_agent_current_values[agent_id] = current_values
            per_agent_discrete_domains: dict[
                str, dict[str, list[object]]
            ] = {}
            for agent_id, fields in per_agent_mutable_fields.items():
                node = self._graph.get_node(agent_id)
                discrete_domains: dict[str, list[object]] = {}
                if "model_id" in fields:
                    discrete_domains["model_id"] = list(
                        per_agent_model_domains[agent_id]
                    )
                recovery_field_values = per_agent_recovery_field_values[
                    agent_id
                ]
                if recovery_field_values is not None:
                    for field in fields:
                        if field in recovery_field_values:
                            discrete_domains[field] = [
                                recovery_field_values[field]
                            ]
                per_agent_discrete_domains[agent_id] = discrete_domains
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
                        "role_family": (
                            self._graph.get_node(agent_id).role_family or ""
                        ),
                        **(
                            {
                                "contract_responsibility": (
                                    _QA_ROLE_CONTRACT_RESPONSIBILITIES[
                                        (
                                            self._graph.get_node(
                                                agent_id
                                            ).role_family
                                            or ""
                                        ).casefold()
                                    ]
                                )
                            }
                            if (
                                "contract" in per_agent_mutable_fields[agent_id]
                                and (
                                    self._graph.get_node(agent_id).role_family
                                    or ""
                                ).casefold()
                                in _QA_ROLE_CONTRACT_RESPONSIBILITIES
                            )
                            else {}
                        ),
                        # Neutral current-state receipt for the hierarchical
                        # parameter phase.  The Env still authoritatively
                        # rejects no-op edits; this simply makes the measured
                        # value explicit instead of asking the Director to
                        # rediscover it from the full graph serialization.
                        "current_values": per_agent_current_values[agent_id],
                        "discrete_value_domains": per_agent_discrete_domains[
                            agent_id
                        ],
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
        # Provider availability is a trajectory-local overlay.  The generic
        # FlowSteer collector reuses one Env instance and calls reset for each
        # task, so a prior task's 401/403/404 must not shrink the next task's
        # frozen model catalog or leave a stale failure receipt.
        self._unavailable_model_ids.clear()
        self._model_availability_receipts.clear()
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
            require_evidence_relation=self.require_evidence_relation,
            director_feedback_mode=self.director_feedback_mode,
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
        # Provider/model availability is a Runtime boundary shared by generic
        # FlowSteer Canvas tasks and semantic QA tasks.  Apply its exact live
        # MODIFY domain before any mutation so a raw/manual action cannot bypass
        # the same constrained domain exposed to the Director.
        provider_repair_issue = self._provider_repair_admission_issue(action)
        if provider_repair_issue is not None:
            return self._reject_after_count(
                action,
                "edit rejected: " + provider_repair_issue,
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
            cached_execution = self._cached_progressive_execution()
            if (
                not validation.valid
                and not self._allows_unconsumed_auxiliary_terminal_reachability(
                    validation,
                    cached_execution,
                )
            ):
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
            execution = cached_execution
            execution_reused = execution is not None
            if execution is None:
                try:
                    execution = await self.runtime.execute(
                        self._graph,
                        self._problem,
                        prior_outputs=self._progressive_outputs,
                        prior_output_metadata=self._progressive_output_metadata,
                        prior_failure_metadata=self._failure_continuations,
                        unavailable_model_ids=self._unavailable_model_ids,
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
            evidence_issue = self._required_evidence_issue(execution)
            if evidence_issue is not None:
                return self._reject_after_count(
                    action,
                    "cannot finish: " + evidence_issue,
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

        semantic_repair_baseline = (
            self._semantic_failure_progress_state(action.agent_id)
            if (
                action.action_type is AgentActionType.MODIFY_AGENT
                and action.agent_id is not None
                and self._uses_semantic_lineage_protocol()
            )
            else None
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
        preserved_input_issue = self._preserved_input_change_issue_for(candidate)
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
        isolated_execution_scope = self._isolated_auxiliary_execution_scope(
            candidate,
            action,
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
                execution_graph = (
                    self._single_agent_execution_graph(
                        self._graph,
                        isolated_execution_scope[0],
                    )
                    if isolated_execution_scope
                    else self._graph
                )
                execution_scope_set = set(isolated_execution_scope)
                prior_outputs = (
                    {
                        agent_id: output
                        for agent_id, output in self._progressive_outputs.items()
                        if agent_id in execution_scope_set
                    }
                    if isolated_execution_scope
                    else self._progressive_outputs
                )
                prior_output_metadata = (
                    {
                        agent_id: metadata
                        for agent_id, metadata in (
                            self._progressive_output_metadata.items()
                        )
                        if agent_id in execution_scope_set
                    }
                    if isolated_execution_scope
                    else self._progressive_output_metadata
                )
                try:
                    prior_failure_metadata = dict(self._failure_continuations)
                    prior_failure_metadata.update(recovery_continuation_handoff)
                    execution_dirty_agents = (
                        execution_scope_set
                        if isolated_execution_scope
                        else (
                            set(self._unresolved_dirty_agents)
                            | self._triviaqa_retrievers_requiring_validation(
                                self._graph
                            )
                        )
                    )
                    execution = await self.runtime.execute(
                        execution_graph,
                        self._problem,
                        require_complete=False,
                        prior_outputs=prior_outputs,
                        prior_output_metadata=prior_output_metadata,
                        prior_failure_metadata=prior_failure_metadata,
                        unavailable_model_ids=self._unavailable_model_ids,
                        dirty_agents=execution_dirty_agents,
                        format_output_agent=(
                            False
                            if isolated_execution_scope
                            else self._uses_format_agent_protocol()
                        ),
                    )
                except AgentRuntimeError as exc:
                    # FlowSteer's progressive Canvas treats execution as edit
                    # feedback.  A provider/runtime failure must not roll back
                    # a structurally valid edit or abort the Director rollout.
                    execution_error = exc
                    partial_execution = exc.partial_result
                    if partial_execution is not None:
                        partial_outputs = dict(partial_execution.outputs)
                        partial_metadata = {
                            agent_id: dict(metadata)
                            for agent_id, metadata in (
                                partial_execution.output_metadata.items()
                            )
                        }
                        if isolated_execution_scope:
                            self._progressive_outputs.update(partial_outputs)
                            self._progressive_output_metadata.update(
                                partial_metadata
                            )
                        else:
                            self._progressive_outputs = partial_outputs
                            self._progressive_output_metadata = partial_metadata
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
                        and (
                            not isolated_execution_scope
                            or agent_id in execution_scope_set
                        )
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
                    execution_outputs = dict(execution.outputs)
                    execution_metadata = {
                        agent_id: dict(metadata)
                        for agent_id, metadata in execution.output_metadata.items()
                    }
                    if isolated_execution_scope:
                        self._progressive_outputs.update(execution_outputs)
                        self._progressive_output_metadata.update(
                            execution_metadata
                        )
                        self._unresolved_dirty_agents.difference_update(
                            execution.outputs
                        )
                        self._unresolved_dirty_agents.update(
                            execution_scope_set - set(execution.outputs)
                        )
                    else:
                        self._progressive_outputs = execution_outputs
                        self._progressive_output_metadata = execution_metadata
                        self._progressive_execution = execution
                        self._progressive_execution_revision = self._graph.revision
                    # A semantic QA Verifier/Formatter can be structurally present
                    # while its semantic input is not yet routable.  Runtime
                    # deferral is successful progressive execution, not Agent
                    # failure; keep only those unmaterialized nodes unresolved.
                    if not isolated_execution_scope:
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
                        # A singleton replacement execution only proves that
                        # this isolated functional unit materialized.  It must
                        # not erase the still-live failure diagnosis of the
                        # historical graph (for example the exhausted
                        # Reasoner that now needs the replacement artifact
                        # routed into it).  Full-graph execution retains the
                        # existing clear-on-complete behavior.
                        if (
                            not isolated_execution_scope
                            and not self._unresolved_dirty_agents
                        ):
                            self._clear_failure_state()
                    else:
                        self._clear_failure_state()
                    if not isolated_execution_scope:
                        self._capture_last_valid_evidence_lineage(execution)
            else:
                self._clear_progressive_execution()
        if (
            semantic_repair_baseline is not None
            and action.action_type is AgentActionType.MODIFY_AGENT
            and action.agent_id is not None
            and execution is not None
            and execution_error is None
            and not isolated_execution_scope
            and self._semantic_failure_progress_state(action.agent_id)
            == semantic_repair_baseline
        ):
            # The accepted FlowSteer edit and execution remain in history, but
            # the next action mask must not repeat the same semantic repair
            # against an unchanged SkillFlow public read-receipt lineage.
            self._repair_exhausted_agent_ids.add(action.agent_id)
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
            if self.director_feedback_mode == "control_plane":
                result = json.dumps(
                    {
                        "status": "not_executed",
                        "graph_revision": self._graph.revision,
                        "recovery_state": self._control_plane_recovery_state(),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                return f"{feedback}; execution_result={result}"
            if self.recovery_policy == _PRESERVE_REPAIR_RECOVERY_POLICY:
                state = json.dumps(
                    self.recovery_state(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                return f"{feedback}; recovery_state={state}"
            return feedback

        if self.director_feedback_mode == "control_plane":
            result = json.dumps(
                self._control_plane_execution_receipt(execution),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return f"{feedback}; execution_result={result}"

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

    @staticmethod
    def _control_plane_tool_receipt_summary(
        receipts: object,
    ) -> dict[str, object]:
        """Return Tool call status without request or observation payloads."""

        if not isinstance(receipts, (list, tuple)):
            receipts = ()
        admitted = tuple(item for item in receipts if isinstance(item, Mapping))
        successful = tuple(
            item
            for item in admitted
            if item.get("error_type") is None
            and isinstance(item.get("result"), Mapping)
            and item["result"].get("completed") is not False
        )
        return {
            "receipt_count": len(admitted),
            "successful_receipt_count": len(successful),
            "tool_ids": sorted(
                {
                    str(item["tool_id"])
                    for item in admitted
                    if item.get("tool_id") is not None
                }
            ),
            "actions": sorted(
                {
                    str(request["action"])
                    for item in admitted
                    for request in (item.get("request"),)
                    if isinstance(request, Mapping)
                    and request.get("action") is not None
                }
            ),
            "error_types": sorted(
                {
                    str(item["error_type"])
                    for item in admitted
                    if item.get("error_type") is not None
                }
            ),
        }

    def _control_plane_recovery_state(self) -> dict[str, object]:
        """Project preserve/repair state without semantic or Tool content."""

        state = self.recovery_state()
        safe_fields = (
            "policy",
            "phase",
            "preserved_agent_ids",
            "previous_revision_preserved_agent_ids",
            "failed_agent_ids",
            "react_turn_exhausted_agent_ids",
            "repair_exhausted_agent_ids",
            "mandatory_repair_agent_ids",
            "active_auxiliary_replacement_agent_ids",
            "diagnosed_unusable_agent_ids",
            "unresolved_dirty_agent_ids",
            "terminal_unreachable_agent_ids",
            "active_semantic_lineage_agent_ids",
            "redundant_after_replacement_takeover_agent_ids",
            "deletable_agent_ids",
            "repair_exhausted_auxiliary_takeover_delete_agent_ids",
            "capacity_recovery_delete_agent_ids",
        )
        return {field: state[field] for field in safe_fields if field in state}

    def _control_plane_execution_receipt(
        self,
        execution: AgentRuntimeResult,
    ) -> dict[str, object]:
        """Project execution state without exposing Agent or Tool content.

        The exact worker artifacts, upstream messages, query arguments, Tool
        observations and provenance receipts remain in ``AgentRuntimeResult``
        and the lossless trajectory.  Only scheduling, routing, artifact
        presence and receipt status cross the Director control-plane boundary.
        """

        calls_by_agent = {
            call.request.agent.id: call for call in execution.calls
        }
        output_calls = [
            call
            for call in execution.calls
            if call.request.agent.id == execution.output_agent_id
        ]
        output_request = output_calls[-1].request if output_calls else None
        output_inbox: list[dict[str, object]] = []
        if output_request is not None:
            for message in output_request.upstream:
                output_inbox.append(
                    {
                        "source_agent_id": message.source_agent_id,
                        "target_agent_id": message.target_agent_id,
                        "message_type": message.message_type,
                        "artifact_type": message.artifact_type,
                        "graph_revision": message.graph_revision,
                        "environment_revision": message.environment_revision,
                        "artifact_present": bool(message.content.strip()),
                        "tool_receipt_summary": (
                            self._control_plane_tool_receipt_summary(
                                message.tool_receipts
                            )
                        ),
                    }
                )

        agents: list[dict[str, object]] = []
        known_agent_ids = tuple(
            sorted(
                set(execution.outputs)
                | set(execution.executed_agent_ids)
                | set(execution.reused_agent_ids)
                | set(execution.deferred_agent_ids)
            )
        )
        for agent_id in known_agent_ids:
            if not self._graph.has_node(agent_id):
                continue
            node = self._graph.get_node(agent_id)
            call = calls_by_agent.get(agent_id)
            own_receipts = (
                ()
                if call is None
                else call.response.metadata.get("tool_receipts", ())
            )
            output_metadata = execution.output_metadata.get(agent_id, {})
            artifact_version = output_metadata.get("artifact_version")
            agents.append(
                {
                    "agent_id": agent_id,
                    "model_id": (
                        node.model_id
                        if call is None
                        else call.request.model.model_id
                    ),
                    "role_family": node.role_family,
                    "execution_mode": node.execution_mode.value,
                    "allowed_tools": list(node.allowed_tools),
                    "artifact_type": node.artifact_type,
                    "artifact_present": bool(
                        execution.outputs.get(agent_id, "").strip()
                    ),
                    "artifact_version_present": bool(artifact_version),
                    "is_output_agent": agent_id == execution.output_agent_id,
                    "upstream_source_ids": (
                        []
                        if call is None
                        else [
                            message.source_agent_id
                            for message in call.request.upstream
                        ]
                    ),
                    "tool_receipt_summary": (
                        self._control_plane_tool_receipt_summary(own_receipts)
                    ),
                }
            )
        return {
            "status": "success",
            "graph_revision": execution.graph_revision,
            "output_agent_id": execution.output_agent_id,
            "output_artifact_present": bool(
                execution.final_answer is not None
                and execution.final_answer.strip()
            ),
            "executed_agent_ids": list(execution.executed_agent_ids),
            "reused_agent_ids": list(execution.reused_agent_ids),
            "deferred_agent_ids": list(execution.deferred_agent_ids),
            "topology": self._graph.topology_statistics(),
            "output_inbox": output_inbox,
            "agents": agents,
            "recovery_state": self._control_plane_recovery_state(),
        }

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

    def _allows_unconsumed_auxiliary_terminal_reachability(
        self,
        validation: GraphValidationResult,
        execution: Optional[AgentRuntimeResult],
    ) -> bool:
        """Admit a verified lineage despite an unconsumed auxiliary block.

        FlowSteer's global graph validator remains unchanged.  This narrow QA
        terminal adaptation applies only after the current revision already has
        a complete receipt-grounded Reasoner--Verifier--Formatter lineage.  It
        prevents an unrelated, successfully materialized recovery block from
        invalidating that lineage merely to satisfy whole-Canvas reachability.
        """

        if (
            not self._uses_semantic_lineage_protocol()
            or self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY
            or execution is None
            or execution.final_answer is None
            or validation.valid
            or not validation.issues
        ):
            return False
        if any(issue.code != "cannot_reach_output" for issue in validation.issues):
            return False
        unreachable_ids = {
            agent_id
            for issue in validation.issues
            for agent_id in issue.agent_ids
        }
        if not unreachable_ids:
            return False
        for agent_id in unreachable_ids:
            if not self._graph.has_node(agent_id):
                return False
            role_family = (
                self._graph.get_node(agent_id).role_family or ""
            ).casefold()
            if role_family not in {"evidence_retriever", "repair"}:
                return False
            if not self._semantic_replacement_has_valid_artifact(
                agent_id,
                role_family,
            ):
                return False
        return bool(
            self._environment_terminal_issue(execution) is None
            and self._semantic_protocol_issue(execution) is None
            and self._required_evidence_issue(execution) is None
            and self._terminal_validation_error(execution.final_answer) is None
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
        execution = self._cached_progressive_execution()
        auxiliary_reachability_only = (
            self._allows_unconsumed_auxiliary_terminal_reachability(
                validation,
                execution,
            )
        )
        if not validation.valid and not auxiliary_reachability_only:
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
        evidence_issue = self._required_evidence_issue(execution)
        if evidence_issue is not None:
            return {
                "admissible": False,
                "stage": "evidence_relation",
                "reason": evidence_issue,
            }
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
        elif reason.startswith("Verifier read-receipt lineage"):
            target_id, role_family = reasoner_id, "reasoner"
            responsible_constraint = "evidence_receipt_lineage"
            preferred_actions = ["set_relation", "add_subgraph", "modify_agent"]
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
            if isinstance(raw_value, (list, tuple)):
                public_items = [
                    dict(item)
                    for item in raw_value
                    if isinstance(item, Mapping)
                ]
                if public_items:
                    result[field_name] = public_items
        has_public_tool_state = any(
            field_name in result for field_name in ("react_trace", "tool_receipts")
        )
        # PROJECT_NECESSARY_ADAPTATION: the public Action--Observation
        # continuation is valid only for the exact upstream artifact versions
        # that conditioned it.  Keep the compact version receipt here; the
        # complete input provenance remains in the trajectory failure record
        # and is deliberately not replayed into the next model context.
        raw_input_versions = record.metadata.get("input_artifact_versions")
        if has_public_tool_state and isinstance(raw_input_versions, Mapping):
            result["input_artifact_versions"] = {
                str(agent_id): str(version)
                for agent_id, version in raw_input_versions.items()
                if isinstance(agent_id, str) and isinstance(version, str)
            }
        source_agent_id = record.metadata.get("continuation_source_agent_id")
        if isinstance(source_agent_id, str) and source_agent_id.strip():
            result["continuation_source_agent_id"] = source_agent_id.strip()
        tool_plan_exhausted = record.metadata.get("tool_plan_exhausted")
        if type(tool_plan_exhausted) is bool:
            result["tool_plan_exhausted"] = tool_plan_exhausted
        return result if len(result) > 1 else None

    def _recovery_continuation_handoff(
        self,
        action: AgentAction,
    ) -> dict[str, dict[str, object]]:
        """Project failed Retriever Tool state to a same-role replacement.

        SkillFlow public Action--Observation continuation remains scoped to the
        same Agent responsibility.  A Reasoner's rejected semantic completion
        and Tool frontier stay in its lossless failure trajectory; neither is
        projected into a new Retriever with a different role contract.
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
        if not replacement_source_ids:
            return {}
        # A failed Retriever replacement can itself be replaced.  Reuse the
        # most advanced public SkillFlow Action--Observation continuation; for
        # equal receipt/trace progress, graph declaration order makes the
        # newest same-role replacement the deterministic source.  This selects
        # no private reasoning, semantic answer, label, or evaluator state.
        source_id = max(
            enumerate(replacement_source_ids),
            key=lambda item: (
                self._failure_continuation_weight(
                    self._failure_continuations[item[1]]
                ),
                item[0],
            ),
        )[1]
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

        if self._tool_continuation_exhausted(source):
            # SkillFlow continuation preserves a partial bounded Tool plan.
            # A terminal plan has no remaining Action budget, so transferring
            # its receipts to a replacement would immediately exhaust the new
            # Agent and recursively reproduce the same failure.
            return None

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
    def _tool_continuation_exhausted(source: Mapping[str, object]) -> bool:
        """Return whether public continuation state declares terminal budget."""

        if source.get("tool_plan_exhausted") is True:
            return True
        raw_trace = source.get("react_trace", ())
        if not isinstance(raw_trace, (list, tuple)):
            return False
        for raw_item in raw_trace:
            if not isinstance(raw_item, Mapping):
                continue
            sources = [raw_item]
            observation = raw_item.get("observation")
            if isinstance(observation, Mapping):
                sources.append(observation)
            for item in sources:
                if item.get("tool_plan_exhausted") is True:
                    return True
                diagnosis = item.get("terminal_failure_diagnosis")
                if isinstance(diagnosis, Mapping) and (
                    diagnosis.get("tool_plan_exhausted") is True
                    or diagnosis.get("bounded_schedule_exhausted") is True
                ):
                    return True
        return False

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
            category, retryability, status_code = (
                self._execution_failure_diagnosis(record)
            )
            self._record_model_unavailability(
                record,
                category=category,
                retryability=retryability,
                status_code=status_code,
            )
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
            if category in _BOUNDED_REACT_FAILURE_CATEGORIES:
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
            recovery_field_values = self._triviaqa_react_recovery_field_values(
                record.agent_id
            )
            location_reasoner_recovery_remaining = bool(
                recovery_field_values
                and self._graph.has_node(record.agent_id)
                and (
                    self._graph.get_node(record.agent_id).role_family or ""
                ).casefold()
                == "reasoner"
                and isinstance(self.runtime.dataset_id, str)
                and self.runtime.dataset_id.casefold() == "triviaqa"
            )
            if (
                category in _BOUNDED_REACT_FAILURE_CATEGORIES
                and (
                    recovery_field_values == {}
                    or (
                        not location_reasoner_recovery_remaining
                        and baseline_receipt_count is not None
                        and new_receipt_count <= baseline_receipt_count
                    )
                    or (
                        not location_reasoner_recovery_remaining
                        and record.agent_id in self._repair_exhausted_agent_ids
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
        if (
            self._requires_complete_semantic_lineage()
        ):
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

    def _uses_format_agent_protocol(
        self,
        graph: Optional[AgentGraph] = None,
    ) -> bool:
        """Enable the Format boundary only when configured or selected."""

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

    def _role_conditional_semantic_edit_issue_for(
        self,
        graph: AgentGraph,
    ) -> Optional[str]:
        """Validate selected QA capabilities without prescribing a workflow."""

        missing_role_ids = tuple(
            node.id
            for node in graph.nodes
            if not (node.role_family or "").strip()
        )
        if missing_role_ids:
            return (
                "Evidence-grounded QA requires a non-empty role_family for "
                f"every Agent; missing role_family for {list(missing_role_ids)!r}"
            )
        react_role_ids = tuple(
            node.id
            for node in graph.nodes
            if (node.role_family or "").casefold() == "react"
        )
        if react_role_ids:
            return (
                "Evidence-grounded QA rejects role_family='react' for Agents "
                f"{list(react_role_ids)!r}; ReAct is execution_mode='react', "
                "not an Agent role"
            )

        output_agent_id = graph.output_agent_id
        if output_agent_id is not None:
            output_role = (
                graph.get_node(output_agent_id).role_family or ""
            ).casefold()
            if output_role not in {"format", "output"}:
                return (
                    "The selected Output Agent must use terminal-compatible "
                    "role_family='output' or the optional role_family='format'"
                )

        formatter_ids = tuple(
            node.id
            for node in graph.nodes
            if (node.role_family or "").casefold() == "format"
        )
        if formatter_ids and (
            len(formatter_ids) != 1 or output_agent_id != formatter_ids[0]
        ):
            return (
                "A Format Agent is optional, but when selected it must be the "
                "unique Output Agent in the same Canvas revision; "
                f"format_agent_ids={list(formatter_ids)!r}, "
                f"output_agent_id={output_agent_id!r}"
            )

        known_roles = {
            "reasoner",
            "verifier",
            "format",
            "evidence_retriever",
            "repair",
            "output",
        }
        formatting_contract = " ".join(
            _QA_ROLE_CONDITIONAL_FORMAT_CONTRACT.casefold().split()
        ).rstrip(".")
        for node in graph.nodes:
            role = (node.role_family or "").casefold()
            normalized_contract = " ".join(
                node.contract.casefold().split()
            ).rstrip(".")
            role_contract_issue = self._qa_role_contract_responsibility_issue(node)
            if role_contract_issue is not None:
                return role_contract_issue
            if role in {"reasoner", "verifier"} and (
                normalized_contract == formatting_contract
            ):
                return (
                    f"{role.title()} Agent {node.id!r} has a formatting-only "
                    "contract; semantic and serialization responsibilities "
                    "must remain distinct"
                )
            if role in known_roles:
                profile = (
                    node.execution_mode.value,
                    tuple(node.allowed_tools),
                )
                admitted_profiles = (
                    self._role_conditional_execution_profiles_for(role)
                )
                if profile not in admitted_profiles:
                    return (
                        f"{role.replace('_', ' ').title()} Agent {node.id!r} "
                        "uses an execution profile not registered for that "
                        f"capability; admitted_profiles={list(admitted_profiles)!r}"
                    )
            if role == "verifier":
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
                        f"Verifier Agent {node.id!r} must consume an explicit "
                        "semantic-candidate artifact, not raw retrieval or a "
                        "terminal wrapper; invalid_predecessors="
                        f"{list(invalid_predecessors)!r}"
                    )
            if role == "format":
                if node.contract not in {
                    _QA_ROLE_CONDITIONAL_FORMAT_CONTRACT,
                    _HOTPOTQA_FORMAT_CONTRACT,
                }:
                    return (
                        f"Format Agent {node.id!r} must use the neutral "
                        "copy-only contract"
                    )
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
                        f"Format Agent {node.id!r} must consume an explicit "
                        "semantic candidate, not raw retrieval or another "
                        "terminal wrapper; invalid_predecessors="
                        f"{list(invalid_predecessors)!r}"
                    )
                successors = self._directed_successors(graph, node.id)
                if successors:
                    return (
                        f"Format Agent {node.id!r} must be a terminal sink; "
                        f"outgoing_agent_ids={list(successors)!r}"
                    )

        if output_agent_id is not None:
            unrouted_verifier_ids = tuple(
                node.id
                for node in graph.nodes
                if (node.role_family or "").casefold() == "verifier"
                and not graph.directed_predecessors(node.id)
            )
            if unrouted_verifier_ids:
                return (
                    "A selected Verifier must receive at least one upstream "
                    "semantic artifact; no named Reasoner or serial topology "
                    "is otherwise required. unrouted_verifier_agent_ids="
                    f"{list(unrouted_verifier_ids)!r}"
                )
            # NECESSARY_ADAPTATION: FlowSteer's accepted ADD is executed before
            # the next Canvas turn. SkillFlow likewise materializes one
            # bounded Agent Action--Observation transition before continuation.
            # Permit only the exact typed recovery unit selected by the live
            # recovery domain to remain isolated for this one edit; every other
            # Agent must already reach Output.
            replacement_domains = (
                self._repair_exhausted_auxiliary_replacement_domains()
            )
            reasoner_augmentation = (
                self._requires_isolated_reasoner_augmentation()
            )
            current_agent_ids = {node.id for node in self._graph.nodes}
            allowed_isolated_recovery_ids = set(
                self._pending_isolated_recovery_auxiliary_agent_ids()
            )
            for node in graph.nodes:
                role = (node.role_family or "").casefold()
                if (
                    node.id not in current_agent_ids
                    and node.id != output_agent_id
                    and not graph.directed_predecessors(node.id)
                    and not self._directed_successors(graph, node.id)
                    and (
                        node.artifact_type.casefold()
                        in replacement_domains.get(role, ())
                        or (
                            reasoner_augmentation
                            and role == "evidence_retriever"
                        )
                    )
                ):
                    allowed_isolated_recovery_ids.add(node.id)
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
                        if agent_id not in allowed_isolated_recovery_ids
                    }
                )
            )
            if unreachable_agent_ids:
                return (
                    "Output may be assigned only when every current Agent can "
                    "reach it in the same Canvas revision; "
                    f"terminal_unreachable_agent_ids={list(unreachable_agent_ids)!r}"
                )
        return None

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
        if not self._requires_complete_semantic_lineage():
            return self._role_conditional_semantic_edit_issue_for(graph)
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

        # FLOWSTEER_SOURCE_ADAPTATION: relation edits remain atomic Canvas
        # transactions, but their two direction bits must describe a valid
        # semantic-artifact handoff.  Retrieval evidence may flow into the
        # Reasoner.  A one-way Reasoner -> Evidence Retriever edge has no
        # evidence ingress and therefore reverses that handoff.  Reciprocal
        # communication remains legal because it retains the required
        # Evidence Retriever -> Reasoner channel; no global topology or Agent
        # count is imposed here.
        evidence_retriever_ids = tuple(
            node.id
            for node in graph.nodes
            if (node.role_family or "").casefold() == "evidence_retriever"
        )
        reasoner_ids = tuple(
            node.id
            for node in graph.nodes
            if (node.role_family or "").casefold() == "reasoner"
        )
        for retriever_id in evidence_retriever_ids:
            for reasoner_id in reasoner_ids:
                direction = graph.relation_bits(retriever_id, reasoner_id)
                if direction.target_to_source and not direction.source_to_target:
                    return (
                        f"{protocol_label} relation {reasoner_id!r} -> "
                        f"{retriever_id!r} is an invalid one-way semantic "
                        "handoff: a Reasoner may send repair feedback to an "
                        "Evidence Retriever only when the reciprocal Evidence "
                        "Retriever -> Reasoner channel also carries retrieval "
                        "evidence"
                    )

        for node in graph.nodes:
            role = (node.role_family or "").casefold()
            normalized_contract = " ".join(node.contract.casefold().split()).rstrip(
                "."
            )
            role_contract_issue = self._qa_role_contract_responsibility_issue(node)
            if role_contract_issue is not None:
                return f"{protocol_label} {role_contract_issue}"
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
            if (
                self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
                and role == "evidence_retriever"
                and (
                    node.execution_mode.value != "react"
                    or node.allowed_tools
                    != (self.required_evidence_tool_id,)
                )
            ):
                return (
                    f"{protocol_label} Evidence Retriever Agent {node.id!r} "
                    "must use execution_mode='react' with exactly "
                    f"allowed_tools=['{self.required_evidence_tool_id}']; its "
                    "answer-free completion is admitted only after entity "
                    "identity, requested relation, evidence span, passage_id, "
                    "and successful read receipt agree"
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
        candidate_value_keys = {
            "candidate_answer",
            "object_or_attribute_value",
        }
        evidence_keys = {
            "evidence_span",
            "evidence",
        }

        def visit(value: object, *, semantic_field: Optional[str] = None) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if semantic_field in candidate_value_keys and stripped:
                    literals.add(stripped)
                elif (
                    semantic_field in evidence_keys
                    and len(self._lexical_tokens(stripped)) >= 6
                ):
                    # Reuse the copied-evidence admission threshold below.
                    # Short rejected fragments such as ``For the`` and opaque
                    # passage identifiers are public continuation state, but
                    # they are not answer-bearing evidence spans.
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
                    visit(
                        item,
                        semantic_field=(
                            key
                            if key in candidate_value_keys or key in evidence_keys
                            else None
                        ),
                    )
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, semantic_field=semantic_field)

        for artifact in (
            *self._progressive_outputs.values(),
            *self._previous_revision_outputs.values(),
        ):
            visit(artifact)
        for continuation in self._failure_continuations.values():
            visit(continuation)
        return tuple(sorted(literals, key=lambda item: (-len(item), item)))

    def _public_successful_read_texts(self) -> Tuple[str, ...]:
        """Return only public passage text from successful Tool read receipts."""

        required_tool_id = self.required_evidence_tool_id
        if not isinstance(required_tool_id, str) or not required_tool_id:
            return ()
        texts: set[str] = set()
        visited: set[int] = set()

        def visit(value: object) -> None:
            if isinstance(value, Mapping):
                identity = id(value)
                if identity in visited:
                    return
                visited.add(identity)
                read_text = self._successful_read_text(value, required_tool_id)
                if read_text is not None:
                    texts.add(str(read_text))
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in visited:
                    return
                visited.add(identity)
                for item in value:
                    visit(item)

        for metadata in (
            *self._progressive_output_metadata.values(),
            *self._previous_revision_output_metadata.values(),
            *self._failure_continuations.values(),
        ):
            visit(metadata)
        for record in self._latest_failure_record_by_agent.values():
            visit(record.metadata)
        return tuple(sorted(texts))

    @staticmethod
    def _numeric_literals(value: str) -> Tuple[str, ...]:
        """Extract concrete digit-bearing values without benchmark constants."""

        normalized = unicodedata.normalize("NFKC", value)
        return tuple(
            re.findall(
                r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,:/-]\d+)*"
                r"(?:s|['’]s)?(?![A-Za-z0-9])",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _numeric_literal_key(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        normalized = normalized.replace("’", "'").strip()
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        return normalized.removeprefix("+")

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
        obligation_entries: Tuple[Tuple[str, str], ...]
        if action.action_type is AgentActionType.ADD_SUBGRAPH:
            obligation_entries = tuple(
                ((spec.role_family or "").casefold(), value)
                for spec in action.agents
                for value in (spec.contract, spec.completion_condition)
                if value is not None
            )
        elif action.action_type is AgentActionType.MODIFY_AGENT:
            current_role = ""
            if action.agent_id is not None and self._graph.has_node(action.agent_id):
                current_role = (
                    self._graph.get_node(action.agent_id).role_family or ""
                ).casefold()
            modified_role = (
                action.role_family.casefold()
                if isinstance(action.role_family, str) and action.role_family.strip()
                else current_role
            )
            obligation_entries = tuple(
                (modified_role, value)
                for value in (action.contract, action.completion_condition)
                if value is not None
            )
        else:
            return None
        obligations = tuple(value for _, value in obligation_entries)
        if not obligations:
            return None

        question = hotpotqa_question_scope(self._problem)
        context = self._problem
        if context.endswith(question) and context != question:
            context = context[: -len(question)]
        public_literals = self._public_semantic_contract_literals()
        public_read_texts = self._public_successful_read_texts()
        named_phrases = self._context_named_phrases(context)
        context_tokens = self._lexical_tokens(context)
        question_tokens = self._lexical_tokens(question)

        def question_contains(span: str) -> bool:
            return self._contains_lexical_span(question, span)

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
                r"\b(?:queries|query\s+variations?|search\s+variations?|"
                r"known\s+variations?)\b[^.!?\n]{0,96}?[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"[\"']?(?:limit|top[_ -]?k)[\"']?\s*"
                r"(?:(?:=|:)\s*|(?:of|to)\s+)\d+",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:symbol|operator|syntax)\b[^.!?\n]{0,80}"
                r"\b(?:in|for)\s+(?:the\s+)?(?:search\s+)?query\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"(?<![A-Za-z0-9_]):[A-Za-z][A-Za-z0-9_-]*\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"[\"']?passage[_ -]?id[\"']?\s*(?:=|:)\s*"
                r"[\"'][^\"'\n]+[\"']",
                flags=re.IGNORECASE,
            ),
        )

        # PROJECT_NECESSARY_ADAPTATION: FlowSteer's transactional Canvas
        # admission remains authoritative for sampled Agent declarations.  A
        # factual-QA Evidence Retriever may describe retrieval responsibilities,
        # but it may not precommit a question-external numeric/date candidate or
        # narrow a question head noun with a new hyphenated qualifier.  This is
        # question-only validation; Ground Truth, accepted answers, evaluator
        # state, and hidden model reasoning are not consulted.
        answer_type = (
            qa_answer_type_constraint(question)
            if self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
            else hotpotqa_answer_type_constraint(question)
        )
        grounded_numeric_keys = {
            self._numeric_literal_key(literal)
            for source in (question, *public_read_texts)
            for literal in self._numeric_literals(source)
        }
        candidate_language = re.compile(
            r"\b(?:answer|candidate|value|return|select|choose|emit|output|"
            r"copy|year|date|number|amount|count|decade)\b",
            flags=re.IGNORECASE,
        )
        operational_count_suffix = re.compile(
            r"^\s*(?:(?:evidence|reasoning|retrieval|tool)\s+)?"
            r"(?:propositions?|steps?|hops?|queries|reads?|receipts?|passages?|"
            r"results?|sources?|agents?|rounds?|turns?|attempts?|tokens?|items?|"
            r"documents?|records?|fields?|samples?|models?)\b",
            flags=re.IGNORECASE,
        )
        for role_family, obligation in obligation_entries:
            # The existing Tool-contract gate below owns concrete invocation
            # syntax and returns its specific repair feedback.  Do not
            # misclassify digits inside query/limit/passage_id arguments as a
            # semantic answer candidate before that gate runs.
            if any(
                pattern.search(obligation) is not None
                for pattern in concrete_tool_argument_patterns
            ):
                continue
            obligation_numeric_literals = self._numeric_literals(obligation)
            concrete_literals: list[str] = []
            for literal in obligation_numeric_literals:
                digit_count = len(re.sub(r"\D", "", literal))
                literal_start = obligation.find(literal)
                literal_end = literal_start + len(literal)
                if literal_start >= 0 and operational_count_suffix.search(
                    obligation[literal_end:]
                ) is not None:
                    continue
                context_window = (
                    obligation[
                        max(0, literal_start - 64) : literal_end + 64
                    ]
                    if literal_start >= 0
                    else obligation
                )
                if (
                    answer_type in {"date", "number"}
                    or digit_count >= 3
                    or candidate_language.search(context_window) is not None
                ):
                    concrete_literals.append(literal)
            question_external_numeric = tuple(
                literal
                for literal in concrete_literals
                if self._numeric_literal_key(literal) not in grounded_numeric_keys
            )
            question_external_modifiers: Tuple[str, ...] = ()
            if role_family == "evidence_retriever":
                question_external_modifiers = tuple(
                    phrase
                    for phrase, head in re.findall(
                        r"\b([A-Za-z][A-Za-z0-9]*-[A-Za-z][A-Za-z0-9]*)\s+"
                        r"([A-Za-z][A-Za-z0-9]*)\b",
                        obligation,
                    )
                    if question_contains(head)
                    and not question_contains(phrase)
                )
            if question_external_numeric or question_external_modifiers:
                return (
                    f"{self._semantic_protocol_label()} Agent "
                    "contract and completion_condition fields are pre-execution "
                    "obligations only: reject question-external semantic literals "
                    f"{list((*question_external_numeric, *question_external_modifiers))!r}. "
                    "A concrete numeric candidate must already occur in the "
                    "original question or a successful public Tool read receipt. "
                    "Preserve the original entity, relation, qualifiers, and "
                    "answer type; spelling normalization, alias expansion, "
                    "entity disambiguation, query rewriting, and larger top-k "
                    "must not precommit a candidate answer or narrow question scope"
                )

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
            r"word|string|substring|token|known\s+fact|e\.g\.|for\s+example|"
            r"canonical\s+match(?:\s+for)?)\b"
            r"[^.!?\n]{0,96}?[\"']([^\"'\n]{1,80})[\"']",
            flags=re.IGNORECASE,
        )
        bare_directive_literal = re.compile(
            r"(?i:\b(?:return|output|emit)\b)\s+"
            r"(?i:(?:only\s+)?(?:the\s+)?(?:word|answer|candidate|value))\s+"
            r"([A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,5}"
            r"|\d{3,}(?:[.,]\d+)?)",
        )
        pre_answer_directive_literal = re.compile(
            r"(?:\(\s*)?"
            r"([A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,5}"
            r"(?:\s*,\s*[A-Z][A-Za-z0-9'’.-]*"
            r"(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,5}){0,3})"
            r"(?:\s*\))?\s+as\s+(?:the\s+)?answer\b",
        )
        answer_assertion_clause = re.compile(
            r"\b(?:answer|candidate)\b[^.!?\n]{0,192}",
            flags=re.IGNORECASE,
        )
        assertion_bound_single_entity = re.compile(
            r"\b(?:is|was|in|at)\s+"
            r"(?:(?:an?|the)\s+)?([A-Z][A-Za-z0-9'’.-]*)\b"
        )
        operational_single_literals = {
            "agent",
            "agentgraph",
            "canvas",
            "director",
            "evidence",
            "formatter",
            "json",
            "markdown",
            "reasoner",
            "retriever",
            "structuredaction",
            "tool",
            "verifier",
            "workflow",
            "xml",
            "yaml",
        }

        # PROJECT_NECESSARY_ADAPTATION: FlowSteer's Director authors semantic
        # Agent obligations, while SkillFlow's request-scoped action schema is
        # the authority for concrete Tool arguments.  Reject only explicit
        # invocations or literal argument values; ordinary responsibility terms
        # such as query rewriting, entity disambiguation, and expanded top-k
        # remain available to the Director.
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
            # Catch the measured one-token precommit form without treating
            # every capitalized word as an entity.  The candidate must occur
            # inside an explicit answer/candidate assertion and immediately
            # after a binding preposition/copula.  Question terms and terms in
            # successful public read receipts remain admissible; evaluator and
            # accepted-answer fields are never consulted.
            asserted_external_literals = tuple(
                literal
                for clause_match in answer_assertion_clause.finditer(obligation)
                for literal in assertion_bound_single_entity.findall(
                    clause_match.group(0)
                )
                if literal.casefold() not in operational_single_literals
                and not question_contains(literal)
                and not any(
                    self._contains_lexical_span(read_text, literal)
                    for read_text in public_read_texts
                )
            )
            if asserted_external_literals:
                return (
                    f"{self._semantic_protocol_label()} Agent contract and "
                    "completion_condition fields are pre-execution obligations "
                    "only: an explicit answer/candidate assertion must not bind "
                    "a question-external single-token entity without a matching "
                    "successful public read Tool receipt; reject literals "
                    f"{list(asserted_external_literals)!r}"
                )

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
                *pre_answer_directive_literal.findall(obligation),
            ]
            if re.search(r"\bknown\s+fact\s*:", obligation, re.IGNORECASE):
                break
            if any(
                literal.strip().casefold() not in allowed_protocol_literals
                for literal in directive_literals
            ):
                break

            # A multi-token proper-name phrase that introduces at least one
            # token absent from the immutable question is a concrete entity
            # precommit, even when the contract avoids explicit answer words.
            # This closes the measured ``Nicolas Sinclair`` narrowing without
            # rejecting a reordered responsibility description made entirely
            # from question terms.  Tool/Agent protocol names remain ordinary
            # execution vocabulary rather than task entities.
            operational_named_phrases = {
                "agent graph",
                "evidence retriever",
                "flow director",
                "ground truth",
                "structured action",
                "tool receipt",
            }
            question_token_set = set(question_tokens)
            external_named_phrases = tuple(
                phrase
                for phrase in self._context_named_phrases(obligation)
                if phrase.casefold() not in operational_named_phrases
                and not question_contains(phrase)
                and any(
                    token not in question_token_set
                    for token in self._lexical_tokens(phrase)
                )
            )
            if external_named_phrases:
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
            {"evidence", "repair_diagnosis", "reasoning"}
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
        preserve_question_derived_answer_field: bool = False,
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
        expected_answer_field = (
            None
            if original_question is None
            else qa_answer_argument_constraint(original_question)
        )
        if (
            expected_answer_field is not None
            and answer_field != expected_answer_field
        ):
            return None, (
                "Reasoner answer_slot.answer_field must preserve the original "
                "question's overt wh-dependency and bind candidate_answer to "
                f"the selected proposition field {expected_answer_field!r}"
            )
        selected_subject = selected["subject"]
        selected_object = selected["object_or_attribute_value"]
        assert isinstance(selected_subject, str)
        assert isinstance(selected_object, str)
        if (
            _canonical_evidence_text(selected_subject)
            == _canonical_evidence_text(selected_object)
        ):
            bound_answer_field = (
                expected_answer_field
                if expected_answer_field is not None
                else answer_field
            )
            non_answer_field = (
                "object_or_attribute_value"
                if bound_answer_field == "subject"
                else "subject"
                if bound_answer_field == "object_or_attribute_value"
                else None
            )
            if (
                non_answer_field is not None
                and selected.get(bound_answer_field) == candidate
            ):
                return None, (
                    "Reasoner selected evidence proposition at "
                    f"evidence_propositions[{proposition_index}] must bind "
                    "distinct subject and object_or_attribute_value arguments; "
                    f"candidate_answer already matches the fixed answer field "
                    f"{bound_answer_field!r}, so repair only the non-answer "
                    f"field {non_answer_field!r} from the same read receipt. "
                    "The same entity or candidate cannot occupy both answer-slot "
                    "fields; self-reported entity binding does not establish the "
                    "other relation argument"
                )
            return None, (
                "Reasoner selected evidence proposition at "
                f"evidence_propositions[{proposition_index}] must bind distinct "
                "subject and object_or_attribute_value arguments. The same "
                "entity or candidate cannot occupy both answer-slot fields; "
                "self-reported entity binding does not establish which field "
                "answers the original question"
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
        temporal_normalization = verified_year_to_decade_normalization(
            original_question=original_question,
            source_value=selected[answer_field],
            candidate_answer=candidate,
        )
        alternate_temporal_fields = tuple(
            field_name
            for field_name in ("subject", "object_or_attribute_value")
            if field_name != answer_field
            and verified_year_to_decade_normalization(
                original_question=original_question,
                source_value=selected.get(field_name),
                candidate_answer=candidate,
            )
        )
        if not temporal_normalization and len(alternate_temporal_fields) == 1:
            if (
                preserve_question_derived_answer_field
                and expected_answer_field is not None
            ):
                return None, (
                    "Reasoner candidate_answer is bound to the alternate "
                    "proposition argument "
                    f"{alternate_temporal_fields[0]!r} at "
                    f"evidence_propositions[{proposition_index}], but the original "
                    "question's overt wh-dependency fixes answer_slot.answer_field "
                    f"to {expected_answer_field!r}; preserve candidate_answer and "
                    "answer_slot, and repair the selected proposition's subject "
                    "and object_or_attribute_value as distinct receipt-grounded "
                    "arguments"
                )
            return None, (
                "Reasoner answer_slot.answer_field selects "
                f"{answer_field!r}, but candidate_answer is the verified "
                "year-to-decade normalization of the selected proposition field "
                f"{alternate_temporal_fields[0]!r}; set answer_field to that "
                "proposition field"
            )
        if (
            not boolean_answer
            and not temporal_normalization
            and candidate not in evidence_span
        ):
            return None, (
                "Reasoner candidate_answer must occur verbatim in the selected "
                "evidence_span unless it is a verified year-to-decade temporal "
                "normalization requested by the question"
            )
        if candidate != selected[answer_field] and not temporal_normalization:
            matching_fields = tuple(
                field_name
                for field_name in ("subject", "object_or_attribute_value")
                if selected.get(field_name) == candidate
            )
            if len(matching_fields) == 1:
                if (
                    preserve_question_derived_answer_field
                    and expected_answer_field is not None
                ):
                    return None, (
                        "Reasoner candidate_answer is bound to the alternate "
                        "proposition argument "
                        f"{matching_fields[0]!r} at "
                        f"evidence_propositions[{proposition_index}], but the "
                        "original question's overt wh-dependency fixes "
                        "answer_slot.answer_field to "
                        f"{expected_answer_field!r}; preserve candidate_answer "
                        "and answer_slot, and repair the selected proposition's "
                        "subject and object_or_attribute_value as distinct "
                        "receipt-grounded arguments"
                    )
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
            preserve_question_derived_answer_field=(
                self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
            ),
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
        arguments = request.get("arguments")
        if not isinstance(arguments, Mapping):
            return False
        if (
            required_tool_id == _TRIVIAQA_QA_MEMORY_TOOL_ID
            and set(arguments) == {"memory_id"}
        ):
            record_id_field = "memory_id"
            record_field = "memory"
        elif set(arguments) == {"passage_id"}:
            # Internal-only compatibility view used by the unchanged v63
            # semantic validators. The canonical worker receipt remains
            # read(memory_id) and is never rewritten in the trajectory.
            record_id_field = "passage_id"
            record_field = "passage"
        else:
            return False
        if (
            not isinstance(arguments.get(record_id_field), str)
            or not arguments[record_id_field].strip()
        ):
            return False
        request_record_id = arguments[record_id_field]
        result = receipt.get("result")
        if (
            not isinstance(result, Mapping)
            or result.get("completed") is not True
        ):
            return False
        value = result.get("value", result)
        if not isinstance(value, Mapping) or value.get("operation") != "read":
            return False
        record = value.get(record_field)
        return (
            isinstance(record, Mapping)
            and value.get(record_id_field) == request_record_id
            and record.get(record_id_field) == request_record_id
            and isinstance(record.get("text"), str)
            and bool(record["text"].strip())
        )

    @staticmethod
    def _successful_search_receipt(
        receipt: Mapping[str, object],
        required_tool_id: str,
    ) -> bool:
        """Recognize one completed local QA-memory search receipt.

        This is the same Tool receipt boundary used by the QA-memory adapter:
        an Agent must first issue a query and then read one returned memory.
        It deliberately inspects only the executing worker's public receipt;
        the Director never owns a Tool request or retrieval payload.
        """

        if (
            receipt.get("tool_id") != required_tool_id
            or receipt.get("error_type") is not None
        ):
            return False
        request = receipt.get("request")
        if not isinstance(request, Mapping) or request.get("action") != "search":
            return False
        arguments = request.get("arguments")
        if (
            not isinstance(arguments, Mapping)
            or not isinstance(arguments.get("query"), str)
            or not arguments["query"].strip()
        ):
            return False
        result = receipt.get("result")
        if not isinstance(result, Mapping) or result.get("completed") is not True:
            return False
        value = result.get("value", result)
        return isinstance(value, Mapping) and value.get("operation") == "search"

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
        request = receipt["request"]
        assert isinstance(request, Mapping)
        arguments = request["arguments"]
        assert isinstance(arguments, Mapping)
        record = value[
            "memory"
            if set(arguments) == {"memory_id"}
            else "passage"
        ]
        assert isinstance(record, Mapping)
        text = record["text"]
        assert isinstance(text, str)
        raw_title = record.get("title")
        passage_title = (
            raw_title.strip()
            if isinstance(raw_title, str) and raw_title.strip()
            else None
        )
        return _ReadReceiptText(text, passage_title=passage_title)

    @classmethod
    def _reasoner_evidence_provenance_issue(
        cls,
        artifact: str,
        read_evidence_texts: Sequence[str],
        *,
        require_answer_binding: bool = False,
        original_question: Optional[str] = None,
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
        from .qa_tool_adapter import (
            _ENTITY_COREFERENCE_PRONOUNS,
            _explicit_named_geographic_scope,
            _location_surface_component_aliases,
            _proposition_preserves_requested_relation,
            _question_scope_modifier_issue,
            _relation_surface_matches_evidence,
            _relation_surfaces_share_content,
        )

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
            canonical_span = _canonical_evidence_text(evidence_span)
            relation = proposition.get("relation")
            if (
                not isinstance(relation, str)
                or not relation.strip()
                or not _relation_surface_matches_evidence(relation, evidence_span)
            ):
                return (
                    f"Reasoner evidence_propositions[{index}].relation is not "
                    "grounded in its evidence_span from the same successful "
                    "qa-retrieval read receipt"
                )
            for field_name in ("subject", "object_or_attribute_value"):
                argument = proposition.get(field_name)
                if not isinstance(argument, str) or not argument.strip():
                    return (
                        f"Reasoner evidence_propositions[{index}].{field_name} "
                        "must be non-empty text"
                    )
                if argument.casefold() in {"yes", "no"}:
                    continue
                if _canonical_evidence_text(argument) not in canonical_span:
                    return (
                        f"Reasoner evidence_propositions[{index}].{field_name} "
                        "is not grounded in its evidence_span from the same "
                        "successful qa-retrieval read receipt"
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
        scope_modifier_issue = _question_scope_modifier_issue(
            original_question or "",
            evidence_span,
        )
        if scope_modifier_issue is not None:
            return (
                "Reasoner semantic scope is not grounded in the selected "
                f"read-receipt evidence: {scope_modifier_issue}"
            )
        selected_relation = selected.get("relation")
        if (
            not isinstance(selected_relation, str)
            or not selected_relation.strip()
            or not _relation_surface_matches_evidence(
                selected_relation,
                evidence_span,
            )
        ):
            return (
                "Reasoner target relation is not grounded in the selected "
                "evidence_span from a successful qa-retrieval read receipt"
            )
        relation_aligned_propositions = tuple(
            proposition
            for proposition in propositions
            if isinstance(proposition, Mapping)
            and isinstance(proposition.get("relation"), str)
            and isinstance(
                proposition.get("object_or_attribute_value"),
                str,
            )
            and isinstance(proposition.get("evidence_span"), str)
            and (
                _relation_surfaces_share_content(
                    proposition["relation"],
                    original_question,
                )
                or _proposition_preserves_requested_relation(
                    requested_relation=original_question,
                    predicate=proposition["relation"],
                    object_or_attribute_value=proposition[
                        "object_or_attribute_value"
                    ],
                    original_question=original_question,
                    evidence_span=proposition["evidence_span"],
                )
            )
        ) if original_question else tuple(propositions)
        if not relation_aligned_propositions:
            return (
                "Reasoner evidence propositions do not preserve the requested "
                "relation from the original question"
            )
        if original_question:
            ignored_question_heads = {
                "a",
                "an",
                "are",
                "did",
                "do",
                "does",
                "how",
                "in",
                "is",
                "on",
                "the",
                "what",
                "when",
                "where",
                "which",
                "who",
                "whom",
                "whose",
                "why",
            }
            question_entity_anchor_list: list[str] = []
            for match in re.finditer(
                r"(?<![A-Za-z0-9])(?:[A-Z][A-Za-z0-9'’.-]*"
                r"(?:\s+[A-Z][A-Za-z0-9'’.-]*)*)",
                original_question,
            ):
                anchor_tokens = match.group(0).split()
                while (
                    anchor_tokens
                    and anchor_tokens[0].casefold() in ignored_question_heads
                ):
                    anchor_tokens.pop(0)
                if anchor_tokens:
                    question_entity_anchor_list.append(" ".join(anchor_tokens))
            named_geographic_scope = _explicit_named_geographic_scope(
                original_question
            )
            canonical_named_scope = (
                _canonical_evidence_text(named_geographic_scope)
                if named_geographic_scope is not None
                else None
            )
            question_entity_anchors = tuple(
                anchor
                for anchor in question_entity_anchor_list
                if _canonical_evidence_text(anchor) != canonical_named_scope
            )

            def proposition_argument_binds_question_entity(
                proposition: Mapping[str, object],
                field_name: str,
                anchor: str,
            ) -> bool:
                argument = proposition.get(field_name)
                evidence_span = proposition.get("evidence_span")
                if (
                    not isinstance(argument, str)
                    or not argument.strip()
                    or not isinstance(evidence_span, str)
                    or not evidence_span.strip()
                ):
                    return False
                canonical_anchor = _canonical_evidence_text(anchor)
                canonical_argument = _canonical_evidence_text(argument)
                canonical_span = _canonical_evidence_text(evidence_span)
                if (
                    canonical_anchor
                    and canonical_argument
                    and re.search(
                        rf"(?<!\w){re.escape(canonical_anchor)}(?!\w)",
                        canonical_argument,
                    )
                    and re.search(
                        rf"(?<!\w){re.escape(canonical_argument)}(?!\w)",
                        canonical_span,
                    )
                ):
                    return True
                honorific_normalized_anchor = re.sub(
                    rf"^(?:{_PERSON_TITLE_PATTERN})\s+",
                    "",
                    canonical_anchor,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()
                for read_text in read_evidence_texts:
                    if not _evidence_span_matches_read(evidence_span, read_text):
                        continue
                    raw_title = getattr(read_text, "passage_title", None)
                    if not isinstance(raw_title, str) or not raw_title.strip():
                        continue
                    canonical_title = _canonical_evidence_text(raw_title)
                    if canonical_title not in {
                        canonical_anchor,
                        honorific_normalized_anchor,
                    }:
                        continue
                    argument_in_title = bool(
                        canonical_argument
                        and re.search(
                            rf"(?<!\w){re.escape(canonical_argument)}(?!\w)",
                            canonical_title,
                        )
                    )
                    title_coreference = bool(
                        canonical_argument in _ENTITY_COREFERENCE_PRONOUNS
                        and canonical_title
                        and canonical_title in _canonical_evidence_text(read_text)
                    )
                    if argument_in_title or title_coreference:
                        return True
                return False

            def proposition_argument_occurs_in_question(
                proposition: Mapping[str, object],
                field_name: str,
            ) -> bool:
                argument = proposition.get(field_name)
                if not isinstance(argument, str) or not argument.strip():
                    return False
                canonical_argument = _canonical_evidence_text(argument)
                canonical_question = _canonical_evidence_text(original_question)
                return bool(
                    canonical_argument
                    and re.search(
                        rf"(?<!\w){re.escape(canonical_argument)}(?!\w)",
                        canonical_question,
                    )
                )

            seeded_argument_aliases: set[str] = set()
            for proposition in propositions:
                assert isinstance(proposition, Mapping)
                seed_fields = tuple(
                    field_name
                    for field_name in (
                        "subject",
                        "object_or_attribute_value",
                    )
                    if proposition is not selected or field_name != answer_field
                )
                for field_name in seed_fields:
                    if not (
                        (
                            not question_entity_anchors
                            and proposition_argument_occurs_in_question(
                                proposition,
                                field_name,
                            )
                        )
                        or any(
                            proposition_argument_binds_question_entity(
                                proposition,
                                field_name,
                                anchor,
                            )
                            for anchor in question_entity_anchors
                        )
                    ):
                        continue
                    argument = proposition.get(field_name)
                    assert isinstance(argument, str)
                    seeded_argument_aliases.add(
                        _canonical_evidence_text(argument)
                    )

            if not seeded_argument_aliases:
                return (
                    "Reasoner requested-relation proposition has no deterministic "
                    "entity binding: its non-answer proposition argument does not "
                    "bind an explicit question-side entity, event, or topic through "
                    "the same successful qa-retrieval read receipt"
                )

            if seeded_argument_aliases:
                answer_slot_type = answer_slot.get("answer_type")

                def proposition_argument_aliases(value: object) -> set[str]:
                    if not isinstance(value, str) or not value.strip():
                        return set()
                    aliases = {_canonical_evidence_text(value)}
                    if answer_slot_type == "location":
                        aliases.update(
                            _canonical_evidence_text(alias)
                            for alias in _location_surface_component_aliases(value)
                        )
                    aliases.discard("")
                    return aliases

                reachable_aliases = set(seeded_argument_aliases)
                proposition_edges = tuple(
                    (
                        proposition_argument_aliases(
                            proposition.get("subject")
                        ),
                        proposition_argument_aliases(
                            proposition.get("object_or_attribute_value")
                        ),
                    )
                    for proposition in propositions
                    if isinstance(proposition, Mapping)
                )
                changed = True
                while changed:
                    changed = False
                    for subject_aliases, object_aliases in proposition_edges:
                        if reachable_aliases & subject_aliases:
                            new_aliases = object_aliases - reachable_aliases
                            if new_aliases:
                                reachable_aliases.update(new_aliases)
                                changed = True
                        if reachable_aliases & object_aliases:
                            new_aliases = subject_aliases - reachable_aliases
                            if new_aliases:
                                reachable_aliases.update(new_aliases)
                                changed = True
                answer_argument_aliases = proposition_argument_aliases(
                    selected.get(answer_field)
                )
                if not reachable_aliases & answer_argument_aliases:
                    return (
                        "Reasoner answer-bearing proposition has no deterministic "
                        "entity binding: answer_slot is not reachable from the "
                        "question-side entity through receipt-grounded evidence "
                        "propositions"
                    )
        if answer_field not in {"subject", "object_or_attribute_value"}:
            return (
                "Reasoner answer_slot.answer_field must be subject or "
                "object_or_attribute_value"
            )
        temporal_normalization = verified_year_to_decade_normalization(
            original_question=original_question,
            source_value=selected.get(answer_field),
            candidate_answer=candidate,
        )
        alternate_temporal_fields = tuple(
            field_name
            for field_name in ("subject", "object_or_attribute_value")
            if field_name != answer_field
            and verified_year_to_decade_normalization(
                original_question=original_question,
                source_value=selected.get(field_name),
                candidate_answer=candidate,
            )
        )
        if not temporal_normalization and len(alternate_temporal_fields) == 1:
            return (
                "Reasoner answer_slot.answer_field selects "
                f"{answer_field!r}, but candidate_answer is the verified "
                "year-to-decade normalization of the selected proposition field "
                f"{alternate_temporal_fields[0]!r}; set answer_field to that "
                "proposition field"
            )
        if not isinstance(candidate, str) or (
            selected.get(answer_field) != candidate
            and not temporal_normalization
        ):
            return (
                "Reasoner candidate_answer must copy the selected proposition "
                "argument exactly unless it is a verified year-to-decade "
                "temporal normalization requested by the question"
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

    @classmethod
    def _semantic_candidate_from_artifact(
        cls,
        artifact: object,
    ) -> tuple[Optional[str], Optional[str]]:
        """Read an explicit semantic candidate without inferring an answer."""

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
                re.sub(r"[ -]+", "_", str(key).strip().casefold()): value
                for key, value in parsed.items()
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

    def _role_conditional_semantic_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Validate free QA topology at the worker Tool/route boundary.

        The role-conditional protocol intentionally does not impose a
        Retriever -> Reasoner -> Verifier -> Formatter spine.  FINISH requires
        only one non-Output ReAct worker with a completed QA-memory search and
        read, a non-empty evidence artifact, and an explicit directed path from
        that worker into the selected Output Agent.  Any further reasoning,
        verification, formatting, fan-out, or reciprocal exchange remains a
        Director-selected Canvas capability rather than a terminal template.
        """

        output_id = self._graph.output_agent_id
        if output_id is None or not self._graph.has_node(output_id):
            return "Evidence-grounded QA has no selected Output Agent"
        if self.required_evidence_tool_id is None:
            return "Evidence-grounded QA has no configured retrieval Tool"

        routed_worker_ids = self._directed_ancestor_ids(self._graph, output_id)
        for worker_id in routed_worker_ids:
            worker = self._graph.get_node(worker_id)
            if (
                worker.execution_mode.value != "react"
                or self.required_evidence_tool_id not in worker.allowed_tools
            ):
                continue
            artifact = execution.outputs.get(worker_id)
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            metadata = execution.output_metadata.get(worker_id)
            if not isinstance(metadata, Mapping):
                continue
            receipts = metadata.get("tool_receipts", ())
            if not isinstance(receipts, (list, tuple)):
                continue
            public_receipts = tuple(
                receipt for receipt in receipts if isinstance(receipt, Mapping)
            )
            if any(
                self._successful_search_receipt(
                    receipt, self.required_evidence_tool_id
                )
                for receipt in public_receipts
            ) and any(
                self._successful_read_receipt(
                    receipt, self.required_evidence_tool_id
                )
                for receipt in public_receipts
            ):
                return None

        return (
            "The selected Output Agent has no explicitly routed upstream ReAct "
            "worker artifact with completed "
            f"{self.required_evidence_tool_id!r} search and read receipts"
        )

    def _required_evidence_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Apply the topology-neutral dynamic evidence FINISH gate.

        This is the direct thin adaptation of FlowSteer's existing
        ``require_evidence_relation`` boundary: it observes only executed Tool
        receipts and explicit Canvas relations.  It neither assigns a role nor
        requires a particular worker chain.  The selected Output Agent cannot
        satisfy this requirement with its own Tool receipt because only strict
        directed ancestors are considered.
        """

        if not self.require_evidence_relation:
            return None
        output_agent_id = self._graph.output_agent_id
        if output_agent_id is None or not self._graph.has_node(output_agent_id):
            return "The current Canvas has no selected Output Agent"
        output_agent = self._graph.get_node(output_agent_id)
        if self.required_evidence_tool_id in output_agent.allowed_tools:
            return (
                f"Output Agent {output_agent_id!r} must consume a provenance-"
                "bearing upstream artifact and cannot hold the worker retrieval "
                f"Tool {self.required_evidence_tool_id!r}"
            )
        return self._role_conditional_semantic_issue(execution)

    def _semantic_protocol_issue(
        self,
        execution: AgentRuntimeResult,
    ) -> Optional[str]:
        """Return the shared evidence-grounded QA FINISH admission gate."""

        if not self._uses_semantic_lineage_protocol():
            return None
        if not self._requires_complete_semantic_lineage():
            return self._role_conditional_semantic_issue(execution)
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
        if self.semantic_protocol == _QA_SEMANTIC_PROTOCOL:
            # NECESSARY_ADAPTATION: TriviaQA's evidence-grounding boundary is
            # an executed data dependency rather than a Director prompt
            # recipe.  Any one of several direct Retriever predecessors may
            # satisfy it, but the artifact, current version and strict read
            # receipt must all be valid before the Reasoner lineage can FINISH.
            from .qa_tool_adapter import QARetrievalReactExecutionAdapter

            valid_retriever_ingress = False
            for predecessor_id in self._graph.directed_predecessors(
                reasoner_id
            ):
                predecessor = self._graph.get_node(predecessor_id)
                if (
                    predecessor.role_family or ""
                ).casefold() != "evidence_retriever":
                    continue
                artifact = execution.outputs.get(predecessor_id)
                metadata = execution.output_metadata.get(predecessor_id)
                if (
                    not isinstance(artifact, str)
                    or not artifact.strip()
                    or not isinstance(metadata, Mapping)
                ):
                    continue
                receipts = metadata.get("tool_receipts", ())
                if not isinstance(receipts, (list, tuple)):
                    continue
                public_receipts = tuple(
                    receipt
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                )
                if (
                    QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                        original_question=hotpotqa_question_scope(
                            self._problem
                        ),
                        artifact=artifact,
                        tool_receipts=public_receipts,
                        retrieval_tool_id=self.required_evidence_tool_id,
                    )
                    is not None
                ):
                    continue
                if self._artifact_version_binding_issue(
                    execution.output_metadata,
                    producer_id=predecessor_id,
                    consumer_id=reasoner_id,
                    consumer_role="Reasoner",
                ) is not None:
                    continue
                valid_retriever_ingress = True
                break
            if not valid_retriever_ingress:
                return (
                    "TriviaQA Reasoner lineage has no current direct "
                    "Evidence Retriever artifact whose entity identity, "
                    "requested relation, evidence span, passage_id, successful "
                    "read receipt, and artifact version all match. Preserve any "
                    "valid Retriever artifact and route it into the Reasoner "
                    "before FINISH"
                )
        verifier_binding_issue = self._artifact_version_binding_issue(
            execution.output_metadata,
            producer_id=reasoner_id,
            consumer_id=verifier_id,
            consumer_role="Verifier",
        )
        if verifier_binding_issue is not None:
            return verifier_binding_issue
        formatter_binding_issue = self._artifact_version_binding_issue(
            execution.output_metadata,
            producer_id=verifier_id,
            consumer_id=formatter_id,
            consumer_role="Formatter",
        )
        if formatter_binding_issue is not None:
            return formatter_binding_issue
        verifier_receipt_texts, verifier_receipt_issue = (
            self._verifier_read_receipt_lineage(
                execution.output_metadata,
                reasoner_id=reasoner_id,
                verifier_id=verifier_id,
            )
        )
        if verifier_receipt_issue is not None:
            return verifier_receipt_issue
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
            original_question=hotpotqa_question_scope(self._problem),
        )
        if provenance_issue is not None:
            completion_only_repair = (
                "no deterministic entity binding" in provenance_issue
                or bool(
                    re.search(
                        r"Reasoner evidence_propositions\[\d+\]\."
                        r"(?:subject|object_or_attribute_value|relation) is not "
                        r"grounded",
                        provenance_issue,
                    )
                )
            )
            if completion_only_repair:
                return (
                    provenance_issue
                    + ". Preserve every successful read receipt and all valid "
                    "semantic fields; repair only the implicated proposition or "
                    "binding fields before FINISH, without another Tool call"
                )
            return (
                provenance_issue
                + ". Preserve the existing candidate and valid evidence; repair or "
                "augment retrieval before FINISH"
            )
        verifier_lineage_issue = self._reasoner_evidence_provenance_issue(
            reasoner_artifact,
            verifier_receipt_texts,
            require_answer_binding=(
                self.semantic_protocol == _QA_SEMANTIC_PROTOCOL
            ),
            original_question=hotpotqa_question_scope(self._problem),
        )
        if verifier_lineage_issue is not None:
            return (
                "Verifier read-receipt lineage does not ground the current "
                f"Reasoner artifact: {verifier_lineage_issue}. Preserve valid "
                "artifacts and receipts, then repair or augment the evidence "
                "handoff before FINISH"
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
    def _artifact_version_binding_issue(
        output_metadata: Mapping[str, Mapping[str, object]],
        *,
        producer_id: str,
        consumer_id: str,
        consumer_role: str,
    ) -> Optional[str]:
        """Require a consumer artifact to bind the producer's current version."""

        producer_metadata = output_metadata.get(producer_id)
        consumer_metadata = output_metadata.get(consumer_id)
        producer_version = (
            producer_metadata.get("artifact_version")
            if isinstance(producer_metadata, Mapping)
            else None
        )
        input_versions = (
            consumer_metadata.get("input_artifact_versions")
            if isinstance(consumer_metadata, Mapping)
            else None
        )
        consumed_version = (
            input_versions.get(producer_id)
            if isinstance(input_versions, Mapping)
            else None
        )
        if (
            not isinstance(producer_version, str)
            or not producer_version.strip()
            or not isinstance(consumed_version, str)
            or not consumed_version.strip()
            or consumed_version != producer_version
        ):
            return (
                f"{consumer_role} {consumer_id!r} is not bound to the current "
                f"artifact version from {producer_id!r}: "
                f"consumed={consumed_version!r}, current={producer_version!r}. "
                "Preserve the current artifacts and re-execute the declared "
                "dependency before FINISH"
            )
        return None

    def _verifier_read_receipt_lineage(
        self,
        output_metadata: Mapping[str, Mapping[str, object]],
        *,
        reasoner_id: str,
        verifier_id: str,
    ) -> tuple[Tuple[str, ...], Optional[str]]:
        """Return successful reads carried by the exact Reasoner→Verifier wire."""

        producer_metadata = output_metadata.get(reasoner_id)
        verifier_metadata = output_metadata.get(verifier_id)
        producer_version = (
            producer_metadata.get("artifact_version")
            if isinstance(producer_metadata, Mapping)
            else None
        )
        provenance = (
            verifier_metadata.get("input_artifact_provenance")
            if isinstance(verifier_metadata, Mapping)
            else None
        )
        if not isinstance(provenance, (list, tuple)):
            return (), (
                f"Verifier {verifier_id!r} has no input_artifact_provenance "
                f"for Reasoner {reasoner_id!r}"
            )
        read_texts: list[str] = []
        matched_wire = False
        assert self.required_evidence_tool_id is not None
        for raw_message in provenance:
            if (
                not isinstance(raw_message, Mapping)
                or raw_message.get("source_agent_id") != reasoner_id
                or raw_message.get("artifact_version") != producer_version
            ):
                continue
            matched_wire = True
            receipts = raw_message.get("tool_receipts", ())
            if not isinstance(receipts, (list, tuple)):
                continue
            for receipt in receipts:
                if not isinstance(receipt, Mapping):
                    continue
                read_text = self._successful_read_text(
                    receipt,
                    self.required_evidence_tool_id,
                )
                if read_text is not None:
                    read_texts.append(read_text)
        if not matched_wire:
            return (), (
                f"Verifier {verifier_id!r} input provenance is not bound to "
                f"the current Reasoner {reasoner_id!r} artifact wire"
            )
        if not read_texts and isinstance(producer_metadata, Mapping):
            producer_receipt_sources: list[object] = [
                producer_metadata.get("tool_receipts", ())
            ]
            producer_inputs = producer_metadata.get(
                "input_artifact_provenance",
                (),
            )
            if isinstance(producer_inputs, (list, tuple)):
                producer_receipt_sources.extend(
                    raw_input.get("tool_receipts", ())
                    for raw_input in producer_inputs
                    if isinstance(raw_input, Mapping)
                )
            for receipt_source in producer_receipt_sources:
                if not isinstance(receipt_source, (list, tuple)):
                    continue
                for receipt in receipt_source:
                    if not isinstance(receipt, Mapping):
                        continue
                    read_text = self._successful_read_text(
                        receipt,
                        self.required_evidence_tool_id,
                    )
                    if read_text is not None:
                        read_texts.append(read_text)
        if not read_texts:
            return (), (
                f"Verifier {verifier_id!r} received no successful "
                f"{self.required_evidence_tool_id!r} read receipt through the "
                f"current Reasoner {reasoner_id!r} artifact lineage"
            )
        return tuple(read_texts), None

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

    def _has_valid_evidence_retriever_artifact(self) -> bool:
        """Return whether any current Retriever owns valid grounded evidence."""

        if self.required_evidence_tool_id is None:
            return False
        return any(
            (node.role_family or "").casefold() == "evidence_retriever"
            and node.id not in self._failed_agent_ids
            and node.id not in self._repair_exhausted_agent_ids
            and self._has_successful_artifact(node.id)
            and self._semantic_replacement_has_valid_artifact(
                node.id,
                "evidence_retriever",
            )
            for node in self._graph.nodes
        )

    def _triviaqa_retriever_has_current_grounded_artifact(
        self,
        graph: AgentGraph,
        retriever_id: str,
    ) -> bool:
        """Validate one direct Retriever cache entry for TriviaQA reuse.

        FlowSteer's accepted edit still enters AgentRuntime once. SkillFlow's
        completion boundary validates Retriever Action--Observation state
        before either a one-way successor or the existing reciprocal four-phase
        block advances to Reasoner. This state-conditioned check only prevents
        an invalid cached artifact from bypassing that causal execution path.
        """

        if (
            self.semantic_protocol != _QA_SEMANTIC_PROTOCOL
            or not isinstance(self.runtime.dataset_id, str)
            or self.runtime.dataset_id.casefold() != "triviaqa"
            or not graph.has_node(retriever_id)
        ):
            return True
        retriever = graph.get_node(retriever_id)
        if (retriever.role_family or "").casefold() != "evidence_retriever":
            return False
        if (
            retriever_id in self._failed_agent_ids
            or retriever_id in self._repair_exhausted_agent_ids
            or not self._has_successful_artifact(retriever_id)
        ):
            return False
        metadata = self._progressive_output_metadata.get(retriever_id)
        artifact_version = (
            metadata.get("artifact_version")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            not isinstance(artifact_version, str)
            or not artifact_version.strip()
        ):
            return False
        return self._semantic_replacement_has_valid_artifact(
            retriever_id,
            "evidence_retriever",
        )

    def _triviaqa_retrievers_requiring_validation(
        self,
        graph: AgentGraph,
    ) -> set[str]:
        """Return cached Retriever inputs that must execute before Reasoning.

        Missing Retriever outputs are already dirty in AgentRuntime.  This
        adds only direct TriviaQA Retriever inputs whose cached artifact fails
        current entity/relation/read-receipt admission, so a stale cache cannot
        bypass the ordinary Retriever -> Reasoner execution dependency.
        """

        if (
            self.semantic_protocol != _QA_SEMANTIC_PROTOCOL
            or not isinstance(self.runtime.dataset_id, str)
            or self.runtime.dataset_id.casefold() != "triviaqa"
        ):
            return set()
        result: set[str] = set()
        for reasoner in graph.nodes:
            if (reasoner.role_family or "").casefold() != "reasoner":
                continue
            for predecessor_id in graph.directed_predecessors(reasoner.id):
                predecessor = graph.get_node(predecessor_id)
                if (
                    predecessor.role_family or ""
                ).casefold() != "evidence_retriever":
                    continue
                if not self._triviaqa_retriever_has_current_grounded_artifact(
                    graph,
                    predecessor_id,
                ):
                    result.add(predecessor_id)
        return result

    def _allows_grounded_terminal_reachability_ingress(
        self,
        candidate: AgentGraph,
        agent_id: str,
        before: Sequence[str],
        after: Sequence[str],
    ) -> bool:
        """Allow one grounded Retriever edge into the active Reasoner.

        FlowSteer's relation edit invalidates and re-executes the changed
        downstream closure.  When a valid semantic lineage already exists but
        one successful public-evidence Retriever is terminal-unreachable, the
        non-destructive repair is therefore the exact one-way
        Retriever -> Reasoner handoff: the Retriever artifact remains live and
        the prior Reasoner/Verifier/Formatter artifacts are retained in the
        previous-revision preservation store before being recomputed.  No
        arbitrary predecessor change is admitted here.
        """

        active_lineage = self._active_semantic_lineage_ids()
        reasoner_ids = self._semantic_role_agent_ids("reasoner")
        target_reasoner_id = (
            active_lineage[0]
            if active_lineage
            else reasoner_ids[0]
            if len(reasoner_ids) == 1
            else None
        )
        if target_reasoner_id is None or agent_id != target_reasoner_id:
            return False
        before_ids = set(before)
        after_ids = set(after)
        if not before_ids < after_ids or len(after_ids - before_ids) != 1:
            return False
        source_id = next(iter(after_ids - before_ids))
        if source_id not in self._terminal_unreachable_agent_ids():
            return False
        if not self._graph.has_node(source_id) or not candidate.has_node(source_id):
            return False
        source = self._graph.get_node(source_id)
        if (source.role_family or "").casefold() != "evidence_retriever":
            return False
        return self._semantic_replacement_has_valid_artifact(
            source_id,
            "evidence_retriever",
        )

    def _preserved_input_change_issue_for(
        self,
        candidate: AgentGraph,
    ) -> Optional[str]:
        """Protect the dependency identity of revision-live successful artifacts."""

        preserved_input_ids = set(self._preserved_input_agent_ids())
        for node in self._graph.nodes:
            if node.id not in preserved_input_ids or not candidate.has_node(node.id):
                continue
            before = tuple(self._graph.directed_predecessors(node.id))
            after = tuple(candidate.directed_predecessors(node.id))
            if before != after:
                if self._allows_grounded_terminal_reachability_ingress(
                    candidate,
                    node.id,
                    before,
                    after,
                ):
                    continue
                return (
                    f"preserve successful Agent {node.id!r} input dependencies; "
                    f"current_predecessors={list(before)!r}, "
                    f"candidate_predecessors={list(after)!r}. Route its existing "
                    "artifact downstream instead of invalidating it, unless that "
                    "Agent is the measured repair target"
                )
        return None

    def _preserved_input_agent_ids(self) -> Tuple[str, ...]:
        """Return successful Agents whose current input identity is immutable."""

        if self.recovery_policy != _PRESERVE_REPAIR_RECOVERY_POLICY:
            return ()
        repairable_ids = self._failed_agent_ids | self._unresolved_dirty_agents
        return tuple(
            node.id
            for node in self._graph.nodes
            if self._has_successful_artifact(node.id)
            and node.id not in repairable_ids
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
        repair_exhausted = tuple(
            sorted(self._repair_exhausted_agent_ids & current_ids)
        )
        mandatory_repair = self._mandatory_repair_agent_ids()
        active_auxiliary_replacements = (
            self._dirty_auxiliary_replacement_agent_ids()
        )
        auxiliary_replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        triviaqa_failed_auxiliary_blocks_missing_roles = bool(
            isinstance(self.runtime.dataset_id, str)
            and self.runtime.dataset_id.casefold() == "triviaqa"
            and any(
                node.id in self._failed_agent_ids
                and node.id in self._repair_exhausted_agent_ids
                and (node.role_family or "").casefold()
                in {"evidence_retriever", "repair"}
                for node in self._graph.nodes
            )
            and not auxiliary_replacement_domains
            and not self._has_valid_evidence_retriever_artifact()
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
        repair_routing_candidates = self._repair_exhausted_relation_candidates()
        failed_ingress_relation_candidates = (
            self._failed_auxiliary_ingress_relation_candidates()
        )
        terminal_reachability_relation_candidates = (
            self._terminal_reachability_relation_candidates()
        )
        takeover_delete_ids = (
            self._repair_exhausted_auxiliary_takeover_delete_ids()
        )
        capacity_recovery_delete_ids = (
            self._capacity_blocking_failed_auxiliary_delete_ids()
        )
        required_evidence_ingress_candidates = (
            self._required_evidence_ingress_relation_candidates()
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
            "model_availability": self.model_availability_receipt(),
            "phase": (
                "diagnose_repair"
                if active_auxiliary_replacements
                else "augment"
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
            "previous_revision_preserved_agent_ids": sorted(
                self._previous_revision_outputs
            ),
            "failed_agent_ids": list(failed),
            "react_turn_exhausted_agent_ids": list(react_exhausted),
            "repair_exhausted_agent_ids": list(repair_exhausted),
            "mandatory_repair_agent_ids": list(mandatory_repair),
            "active_auxiliary_replacement_agent_ids": list(
                active_auxiliary_replacements
            ),
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
            "capacity_recovery_delete_agent_ids": list(
                capacity_recovery_delete_ids
            ),
            "deletion_protected": protected,
            "preferred_actions": (
                ["modify_agent"]
                if mandatory_repair or active_auxiliary_replacements
                else ["add_subgraph"]
                if (
                    self._graph.nodes
                    and self._missing_semantic_role_families()
                    and not triviaqa_failed_auxiliary_blocks_missing_roles
                    and (
                        self.max_agents is None
                        or len(self._graph.nodes) < self.max_agents
                    )
                )
                else ["delete_agent"]
                if capacity_recovery_delete_ids or takeover_delete_ids
                else ["set_relation"]
                if failed_ingress_relation_candidates
                else ["set_relation"]
                if repair_routing_candidates
                else ["set_relation"]
                if required_evidence_ingress_candidates
                else ["set_relation"]
                if required_relation_candidates
                else ["set_output"]
                if self._uses_semantic_lineage_protocol() and output_target_ids
                else ["set_relation"]
                if terminal_reachability_relation_candidates
                else ["add_subgraph"]
                if (
                    repair_exhausted
                    and self._model_admissible_add_role_families()
                    and (
                        self.max_agents is None
                        or len(self._graph.nodes) < self.max_agents
                    )
                )
                else []
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
        if not self._requires_complete_semantic_lineage():
            execution = self._cached_progressive_execution()
            output_id = self._graph.output_agent_id
            if (
                execution is None
                or execution.final_answer is None
                or output_id is None
                or self._role_conditional_semantic_issue(execution) is not None
                or self._terminal_validation_error(execution.final_answer)
                is not None
            ):
                return ()
            routed_ids = (
                *self._directed_ancestor_ids(self._graph, output_id),
                output_id,
            )
            return tuple(
                agent_id
                for agent_id in routed_ids
                if self._has_successful_artifact(agent_id)
            )
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
        artifact = self._progressive_outputs.get(agent_id)
        if not isinstance(artifact, str) or not artifact.strip():
            return False
        if not self._requires_complete_semantic_lineage():
            if role_family == "reasoner":
                candidate, issue = self._reasoner_candidate_for_current_dataset(
                    artifact
                )
                if issue is not None or candidate is None:
                    return False
                owner_ids = (
                    *self._directed_ancestor_ids(self._graph, agent_id),
                    agent_id,
                )
                execution = self._cached_progressive_execution()
                if execution is None:
                    return False
                evidence_texts = self._successful_read_texts_for_agents(
                    execution,
                    owner_ids,
                )
                return bool(evidence_texts) and (
                    self._reasoner_evidence_provenance_issue(
                        artifact,
                        evidence_texts,
                        require_answer_binding=True,
                        original_question=hotpotqa_question_scope(self._problem),
                    )
                    is None
                )
            if role_family == "verifier":
                verifier_candidate, verifier_issue = self._verifier_candidate(
                    artifact
                )
                if verifier_issue is not None or verifier_candidate is None:
                    return False
                upstream_candidates = tuple(
                    candidate
                    for upstream_id in self._directed_ancestor_ids(
                        self._graph,
                        agent_id,
                    )
                    if (
                        self._graph.get_node(upstream_id).role_family or ""
                    ).casefold()
                    != "verifier"
                    for candidate, issue in (
                        self._semantic_candidate_from_artifact(
                            self._progressive_outputs.get(upstream_id, "")
                        ),
                    )
                    if issue is None and candidate is not None
                )
                return bool(upstream_candidates) and all(
                    candidate == verifier_candidate
                    for candidate in upstream_candidates
                )
            if role_family == "format":
                return agent_id in self._active_semantic_lineage_ids()
        if role_family == "evidence_retriever":
            metadata = self._progressive_output_metadata.get(agent_id)
            if not isinstance(metadata, Mapping):
                return False
            receipts = metadata.get("tool_receipts", ())
            if not isinstance(receipts, (list, tuple)):
                return False
            assert self.required_evidence_tool_id is not None
            public_receipts = tuple(
                receipt
                for receipt in receipts
                if isinstance(receipt, Mapping)
            )
            if not any(
                self._successful_read_text(
                    receipt,
                    self.required_evidence_tool_id,
                )
                is not None
                for receipt in public_receipts
            ):
                return False
            # Use the same answer-free entity/relation/read-receipt gate that
            # admitted the Retriever completion. A successful read alone is
            # not replacement takeover: its artifact must still bind the
            # original question entity and requested relation to that receipt.
            from .qa_tool_adapter import QARetrievalReactExecutionAdapter

            completion_issue = QARetrievalReactExecutionAdapter._evidence_retriever_completion_issue(
                original_question=hotpotqa_question_scope(self._problem),
                artifact=artifact,
                tool_receipts=public_receipts,
                retrieval_tool_id=self.required_evidence_tool_id,
            )
            return completion_issue is None
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
                original_question=hotpotqa_question_scope(self._problem),
            ) is None
        if role_family == "verifier":
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
            if self._artifact_version_binding_issue(
                self._progressive_output_metadata,
                producer_id=reasoner_ids[0],
                consumer_id=agent_id,
                consumer_role="Verifier",
            ) is not None:
                return False
            verifier_candidate, verifier_issue = self._verifier_candidate(artifact)
            if verifier_issue is not None or verifier_candidate is None:
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
            if self._artifact_version_binding_issue(
                self._progressive_output_metadata,
                producer_id=predecessors[0],
                consumer_id=agent_id,
                consumer_role="Formatter",
            ) is not None:
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
        if agent_id in self._capacity_blocking_failed_auxiliary_delete_ids():
            return None
        node = self._graph.get_node(agent_id)
        terminal_unreachable_ids = set(self._terminal_unreachable_agent_ids())
        # Topological disconnection is a relation fault, not evidence that the
        # node itself is unusable.  Deletion therefore requires either the
        # adapter's explicit unusable diagnosis or bounded ReAct exhaustion,
        # plus a same-responsibility replacement takeover.
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
        latest_failure = self._latest_failure_record_by_agent.get(agent_id)
        bounded_auxiliary_exhaustion = (
            role in {"evidence_retriever", "repair"}
            and agent_id in self._failed_agent_ids
            and agent_id in self._repair_exhausted_agent_ids
            and latest_failure is not None
            and self._execution_failure_diagnosis(latest_failure)[0]
            in _BOUNDED_REACT_FAILURE_CATEGORIES
        )
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
        if (diagnosed_unusable or bounded_auxiliary_exhaustion) and replacements:
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
        replacement_domains = (
            self._repair_exhausted_auxiliary_replacement_domains()
        )
        if (
            self._graph.nodes
            and missing_role_families
            and not replacement_domains
            and not self._dirty_auxiliary_replacement_agent_ids()
            and (
                self.max_agents is None
                or len(self._graph.nodes) < self.max_agents
            )
        ):
            sampled_role_families = tuple(
                (spec.role_family or "").casefold() for spec in action.agents
            )
            exact_missing_role_add = (
                action.action_type is AgentActionType.ADD_SUBGRAPH
                and 1 <= len(action.agents) <= len(missing_role_families)
                and len(sampled_role_families) == len(set(sampled_role_families))
                and set(sampled_role_families) <= set(missing_role_families)
            )
            if (
                exact_missing_role_add
                and not self._required_evidence_ingress_relation_candidates()
            ):
                # Match the live FlowSteer action mask for both initial
                # progressive construction and recovery: one admitted
                # functional subgraph may materialize the still-missing
                # semantic responsibility before a later Canvas edit closes
                # any remaining relation.  The accepted edit executes and
                # feeds back immediately; this orders live edits without
                # prescribing a complete workflow topology.
                return None
            if (
                bool(self._failed_agent_ids or self._repair_exhausted_agent_ids)
                and not self._required_evidence_ingress_relation_candidates()
            ):
                return (
                    "complete the missing semantic responsibilities before "
                    "auxiliary augmentation or generic relation edits; add only "
                    "roles from admitted_new_role_families="
                    f"{list(missing_role_families)!r}"
                )

        capacity_recovery_delete_ids = (
            self._capacity_blocking_failed_auxiliary_delete_ids()
        )
        if capacity_recovery_delete_ids:
            if (
                action.action_type is AgentActionType.DELETE_AGENT
                and action.agent_id in capacity_recovery_delete_ids
            ):
                return None
            return (
                "free one full-capacity augmentation slot only by deleting an "
                "artifact-free isolated typed repair-exhausted auxiliary; "
                "admissible_delete_agent_ids="
                f"{list(capacity_recovery_delete_ids)!r}"
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
        exhausted_auxiliary_ids = tuple(
            agent_id
            for agent_id in self._recovery_auxiliary_agent_ids()
            if agent_id in self._failed_agent_ids
            and agent_id in self._repair_exhausted_agent_ids
        )
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

        isolated_reasoner_augmentation = (
            self._requires_isolated_reasoner_augmentation()
        )
        if (
            isolated_reasoner_augmentation
            and action.action_type is AgentActionType.ADD_SUBGRAPH
        ):
            if (
                len(action.agents) != 1
                or (action.agents[0].role_family or "").casefold()
                != "evidence_retriever"
            ):
                return (
                    "exhausted-Reasoner recovery requires one executable "
                    "Evidence Retriever augmentation"
                )
            if action.relations or action.output_agent_id is not None:
                return (
                    "execute the recovery Evidence Retriever as an isolated "
                    "Canvas unit with relations=[] and no output_agent_id; after "
                    "its artifact passes entity/relation/Tool-receipt validation, "
                    "route it in a later set_relation edit"
                )

        if (
            replacement_domains
            and action.action_type is AgentActionType.ADD_SUBGRAPH
        ):
            if len(action.agents) != 1:
                return (
                    "bounded auxiliary recovery requires exactly one same-role/"
                    "same-artifact replacement Agent"
                )
            replacement = action.agents[0]
            replacement_role = (replacement.role_family or "").casefold()
            replacement_artifact_type = (
                replacement.artifact_type or "text"
            ).casefold()
            if replacement_artifact_type not in replacement_domains.get(
                replacement_role,
                (),
            ):
                return (
                    "bounded auxiliary recovery requires exactly one same-role/"
                    "same-artifact replacement; replacement_domains="
                    f"{replacement_domains!r}"
                )
            if action.relations or action.output_agent_id is not None:
                return (
                    "add the same-role/same-artifact auxiliary replacement "
                    "as an isolated executable prefix with relations=[] and no "
                    "output_agent_id. The accepted ADD executes immediately; route "
                    "its artifact to the Reasoner only after that execution succeeds"
                )
            return None

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
                "repair or execute only the unresolved same-role auxiliary "
                "replacement before augmentation or modification of blocked "
                "downstream Agents; admissible_modify_agent_ids="
                f"{list(dirty_replacement_ids)!r}; mutable_fields="
                "['contract', 'completion_condition']"
            )

        missing_role_families = self._missing_semantic_role_families()
        if (
            self._graph.nodes
            and missing_role_families
            and bool(self._failed_agent_ids or self._repair_exhausted_agent_ids)
            and not replacement_domains
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
                "complete the missing semantic responsibilities before "
                "auxiliary augmentation or generic relation edits; add only "
                "roles from admitted_new_role_families="
                f"{list(missing_role_families)!r}"
            )

        required_evidence_ingress_candidates = (
            self._required_evidence_ingress_relation_candidates()
        )
        if required_evidence_ingress_candidates:
            if any(
                self._relation_action_matches_candidate(action, candidate)
                for candidate in required_evidence_ingress_candidates
            ):
                return None
            return (
                "route one current receipt-grounded Evidence Retriever artifact "
                "into the Reasoner before other Canvas edits; use an exact "
                "admitted set_relation candidate"
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

        output_target_ids = self._model_admissible_output_agent_ids()
        if output_target_ids:
            if (
                action.action_type is AgentActionType.SET_OUTPUT
                and action.agent_id in output_target_ids
            ):
                return None
            return (
                "select the prospectively valid Formatter Output Agent before "
                "other Canvas edits; admissible_output_agent_ids="
                f"{list(output_target_ids)!r}"
            )

        terminal_reachability_candidates = (
            self._terminal_reachability_relation_candidates()
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

        if action.action_type is AgentActionType.SET_RELATION:
            relation_item = {
                "source_id": action.source_id,
                "target_id": action.target_id,
                "source_to_target": bool(action.source_to_target),
                "target_to_source": bool(action.target_to_source),
            }
            if self._relation_adds_failed_artifact_free_auxiliary_source(
                relation_item
            ):
                return (
                    "do not use a failed, repair-exhausted, artifact-free "
                    "auxiliary as a new terminal-reachability source"
                )
            if self._relation_routes_replacement_outside_reasoner(
                relation_item
            ):
                return (
                    "route a validated auxiliary replacement artifact only to "
                    "an active semantic Reasoner consumer"
                )

        if (
            (exhausted_reasoner_ids or exhausted_auxiliary_ids)
            and action.action_type is AgentActionType.SET_RELATION
        ):
            # The failed-ingress, grounded-routing, terminal-reachability, and
            # required semantic-spine candidates above are the complete legal
            # relation recovery domain.  Once it is empty, reject generic graph
            # rewrites instead of letting a terminal Tool failure drift into
            # arbitrary peer edges or reciprocal cycles.
            return (
                "no admissible semantic recovery relation remains for the "
                "repair-exhausted Agent; preserve existing evidence and use a "
                "legal bounded replacement when capacity is available"
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
    def _terminal_retrieval_failure_diagnosis(
        record: AgentFailureRecord,
    ) -> Optional[Mapping[str, object]]:
        """Return the latest public terminal retrieval diagnosis, if any."""

        raw_trace = record.metadata.get("react_trace", ())
        if not isinstance(raw_trace, (list, tuple)):
            return None
        for entry in reversed(raw_trace):
            if not isinstance(entry, Mapping):
                continue
            diagnosis = entry.get("terminal_failure_diagnosis")
            if not isinstance(diagnosis, Mapping):
                observation = entry.get("observation")
                if isinstance(observation, Mapping):
                    diagnosis = observation.get(
                        "terminal_failure_diagnosis"
                    )
            if not isinstance(diagnosis, Mapping):
                continue
            return diagnosis
        return None

    @staticmethod
    def _typed_retrieval_failure_category(
        record: AgentFailureRecord,
    ) -> Optional[str]:
        """Return the adapter's typed terminal retrieval diagnosis, if any."""

        diagnosis = AgentWorkflowEnv._terminal_retrieval_failure_diagnosis(
            record
        )
        if diagnosis is None:
            return None
        public_error_code = diagnosis.get("public_error_code")
        return (
            public_error_code
            if public_error_code in _TYPED_RETRIEVAL_FAILURE_RETRYABILITY
            else None
        )

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
        typed_retrieval_failure = (
            AgentWorkflowEnv._typed_retrieval_failure_category(record)
        )
        if typed_retrieval_failure is not None:
            return (
                typed_retrieval_failure,
                _TYPED_RETRIEVAL_FAILURE_RETRYABILITY[
                    typed_retrieval_failure
                ],
                status_code,
            )
        raw_react_trace = record.metadata.get("react_trace", ())
        raw_model_calls = record.metadata.get("model_calls", ())
        late_react_bad_request = bool(
            status_code == 400
            and isinstance(raw_react_trace, (list, tuple))
            and len(raw_react_trace) > 0
            and isinstance(raw_model_calls, (list, tuple))
            and len(raw_model_calls) > 1
        )
        if late_react_bad_request:
            # A provider 400 after an already-materialized public ReAct prefix
            # is not evidence that the catalog model is permanently invalid.
            # Preserve the Action--Observation continuation and expose the
            # ordinary bounded repair boundary. Startup/configuration 400s and
            # 401/403/404 keep their existing provider diagnosis below.
            return (
                "react_continuation_request_failure",
                "preserve_public_continuation",
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
        last_terminal_failure_diagnosis: Optional[dict[str, object]] = None
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
            terminal_diagnosis = entry.get("terminal_failure_diagnosis")
            if not isinstance(terminal_diagnosis, Mapping) and source is not entry:
                terminal_diagnosis = source.get("terminal_failure_diagnosis")
            if isinstance(terminal_diagnosis, Mapping):
                projected: dict[str, object] = {}
                for field_name in (
                    "observation_status",
                    "public_error_code",
                    "react_turn_exhausted",
                    "tool_plan_exhausted",
                    "bounded_schedule_exhausted",
                    "continuation_admissible",
                    "remaining_tool_calls",
                    "retrieval_attempt_count",
                    "retrieval_strategy_progress_count",
                    "recall_expansion_count",
                    "normalized_query_novelty_verified",
                    "strategy_semantics_verified",
                    "successful_search_with_hits_count",
                    "successful_empty_search_count",
                    "tool_error_count",
                ):
                    value = terminal_diagnosis.get(field_name)
                    if isinstance(value, (str, int, bool)):
                        projected[field_name] = value
                schedule = terminal_diagnosis.get(
                    "retrieval_strategy_schedule_prefix"
                )
                if isinstance(schedule, list) and all(
                    isinstance(value, str) for value in schedule
                ):
                    projected["retrieval_strategy_schedule_prefix"] = list(
                        schedule[:8]
                    )
                for coverage_field in (
                    "verified_retrieval_strategy_coverage",
                    "missing_retrieval_strategy_coverage",
                ):
                    coverage = terminal_diagnosis.get(coverage_field)
                    if isinstance(coverage, list) and all(
                        isinstance(value, str) for value in coverage
                    ):
                        projected[coverage_field] = list(coverage[:8])
                terminal_code = projected.get("public_error_code")
                if isinstance(terminal_code, str) and terminal_code:
                    code_counts[terminal_code] = (
                        code_counts.get(terminal_code, 0) + 1
                    )
                if projected:
                    last_terminal_failure_diagnosis = projected

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
        if last_terminal_failure_diagnosis is not None:
            summary["terminal_failure_diagnosis"] = (
                last_terminal_failure_diagnosis
            )
        return summary

    def _execution_error_feedback(self, exc: AgentRuntimeError) -> str:
        if self.director_feedback_mode == "control_plane":
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
                react_summary = self._react_public_error_summary(record)
                item: dict[str, object] = {
                    "agent_id": record.agent_id,
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "phase": record.phase.value,
                    "error_type": record.error_type,
                    "failure_category": category,
                    "retryability": retryability,
                    "tool_receipt_summary": (
                        self._control_plane_tool_receipt_summary(
                            record.metadata.get("tool_receipts", ())
                        )
                    ),
                    "react_summary": {
                        field: react_summary[field]
                        for field in (
                            "react_turn_count",
                            "observation_status_counts",
                            "public_error_code_counts",
                            "successful_tool_receipt_count",
                            "successful_evidence_read_count",
                            "terminal_failure_diagnosis",
                        )
                        if field in react_summary
                    },
                }
                if status_code is not None:
                    item["http_status"] = status_code
                failed_agents.append(item)
            payload = json.dumps(
                {
                    "type": type(exc).__name__,
                    "code": "agent_runtime_execution_failed",
                    "failed_agents": failed_agents,
                    "blocked_agent_ids": list(exc.blocked_agent_ids),
                    "pending_agent_ids": list(exc.pending_agent_ids),
                    "preserved_agent_ids": (
                        []
                        if exc.partial_result is None
                        else sorted(exc.partial_result.outputs)
                    ),
                    "recovery_state": self._control_plane_recovery_state(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return f"execution_error={payload}"
        message = " ".join(str(exc).split())
        if len(message) > 240:
            message = message[:237] + "..."
        live_action_types = set(self.model_admissible_action_types())
        live_modify_agent_ids = (
            set(self._model_admissible_modify_agent_ids())
            if AgentActionType.MODIFY_AGENT.value in live_action_types
            else set()
        )
        live_recovery_role_families = tuple(
            role_family
            for role_family in (
                self._model_admissible_add_role_families()
                if AgentActionType.ADD_SUBGRAPH.value in live_action_types
                else ()
            )
            if role_family in {"evidence_retriever", "repair"}
        )
        live_recovery_relation_available = bool(
            AgentActionType.SET_RELATION.value in live_action_types
            and (
                self._failed_auxiliary_ingress_relation_candidates()
                or self._repair_exhausted_relation_candidates()
                or self._required_evidence_ingress_relation_candidates()
            )
        )
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
                item["react_public_error_summary"] = (
                    self._react_public_error_summary(record)
                )
                item["operational_diagnosis"] = {
                    "domain": "retrieval_or_database_coverage",
                    "corpus_level_oracle_claim": False,
                }
                action_order = [
                    action_type
                    for action_type in (
                        AgentActionType.MODIFY_AGENT.value,
                        AgentActionType.ADD_SUBGRAPH.value,
                        AgentActionType.SET_RELATION.value,
                    )
                    if (
                        action_type == AgentActionType.MODIFY_AGENT.value
                        and record.agent_id in live_modify_agent_ids
                    )
                    or (
                        action_type == AgentActionType.ADD_SUBGRAPH.value
                        and bool(live_recovery_role_families)
                    )
                    or (
                        action_type == AgentActionType.SET_RELATION.value
                        and live_recovery_relation_available
                    )
                ]
                if action_order:
                    item["preferred_repair"] = {
                        "action_order": action_order,
                        "agent_id": record.agent_id,
                        **(
                            {
                                "admitted_role_families": list(
                                    live_recovery_role_families
                                )
                            }
                            if live_recovery_role_families
                            else {}
                        ),
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
                if admitted_model_ids:
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
                else:
                    # Do not advertise an impossible MODIFY action.  The next
                    # collector boundary observes the empty legal action domain
                    # and persists FlowSteer's typed natural terminal.
                    item["admitted_model_ids"] = []
            elif category in _BOUNDED_REACT_FAILURE_CATEGORIES:
                item["react_public_error_summary"] = (
                    self._react_public_error_summary(record)
                )
                if record.agent_id in self._repair_exhausted_agent_ids:
                    action_order = [
                        action_type
                        for action_type in (
                            AgentActionType.ADD_SUBGRAPH.value,
                            AgentActionType.SET_RELATION.value,
                            AgentActionType.MODIFY_AGENT.value,
                        )
                        if (
                            action_type == AgentActionType.ADD_SUBGRAPH.value
                            and bool(live_recovery_role_families)
                        )
                        or (
                            action_type == AgentActionType.SET_RELATION.value
                            and live_recovery_relation_available
                        )
                        or (
                            action_type == AgentActionType.MODIFY_AGENT.value
                            and record.agent_id in live_modify_agent_ids
                        )
                    ]
                    if action_order:
                        item["preferred_repair"] = {
                            "action_order": action_order,
                            **(
                                {
                                    "admitted_role_families": list(
                                        live_recovery_role_families
                                    )
                                }
                                if live_recovery_role_families
                                else {}
                            ),
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
            unavailable_model_ids=self._unavailable_model_ids,
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
                if item.model_id in self._unavailable_model_ids:
                    raise GraphMutationError(
                        "add_subgraph model_id is unavailable for the current "
                        f"trajectory: {item.model_id!r}"
                    )
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
            if action.model_id in self._unavailable_model_ids:
                raise GraphMutationError(
                    "add_agent model_id is unavailable for the current "
                    f"trajectory: {action.model_id!r}"
                )
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
            if (
                action.model_id is not None
                and action.model_id in self._unavailable_model_ids
            ):
                raise GraphMutationError(
                    "modify_agent model_id is unavailable for the current "
                    f"trajectory: {action.model_id!r}"
                )
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
