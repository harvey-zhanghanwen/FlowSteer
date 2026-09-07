"""Translate a ledger candidate into a paired AgentGraph intervention.

The translator is deliberately a control-plane component.  It does not mutate
the Canvas or execute either branch: it receives the immutable pre-execution
snapshot and returns the two atomic :class:`AgentAction` values that the probe
runner must apply to independently restored forks.

Role-cluster interventions cross the free-contract boundary through an
injected LLM rewriter.  No keyword, regular-expression, or role template is
used here.  The protocol is structurally compatible with
``role_classifier.ContractRoleRewriter`` without importing its concrete
network client.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol, Sequence

from ..agent_action_parser import AgentAction, AgentActionType
from ..agent_workflow_env import AgentWorkflowSnapshot
from .combination_ledger import DecisionKey, ROLE_CLUSTERS


INTERVENTION_RECEIPT_VERSION = "paired-agentgraph-intervention-v1"
_DECISION_FIELDS = (
    "role_cluster",
    "model_id",
    "edge_type",
    "same_model_as_upstream",
)


class ProbeInterventionError(ValueError):
    """The requested one-field intervention cannot be represented safely."""


class ContractRewriteResult(Protocol):
    """Return type implemented by ``role_classifier.ContractRewrite``."""

    contract: str
    receipt: object


class ContractRewriter(Protocol):
    """Injected LLM boundary for changing a free contract's role cluster."""

    async def rewrite(
        self,
        original_contract: str,
        *,
        target_role_cluster: str,
        task_family: str,
        seed: Optional[int] = None,
    ) -> ContractRewriteResult:
        ...


@dataclass(frozen=True, slots=True)
class ProbeInterventionReceipt:
    """Auditable translation receipt for a same-snapshot paired probe."""

    version: str
    snapshot_id: str
    changed_field: str
    current_key: DecisionKey
    candidate_key: DecisionKey
    keep_action: AgentAction
    switch_action: AgentAction
    target_agent_id: Optional[str]
    relation_endpoints: Optional[tuple[str, str]]
    upstream_agent_id: Optional[str]
    downstream_agent_id: Optional[str]
    keep_upstream_model_id: Optional[str]
    selected_upstream_model_id: Optional[str]
    model_selection_reason: Optional[str]
    model_catalog_version: Optional[str]
    contract_rewrite_receipt: Optional[Mapping[str, Any]]
    snapshot_unchanged: bool
    problem_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "changed_field": self.changed_field,
            "current_key": self.current_key.to_dict(),
            "candidate_key": self.candidate_key.to_dict(),
            "keep_action": self.keep_action.to_dict(),
            "switch_action": self.switch_action.to_dict(),
            "target_agent_id": self.target_agent_id,
            "relation_endpoints": (
                None if self.relation_endpoints is None else list(self.relation_endpoints)
            ),
            "upstream_agent_id": self.upstream_agent_id,
            "downstream_agent_id": self.downstream_agent_id,
            "keep_upstream_model_id": self.keep_upstream_model_id,
            "selected_upstream_model_id": self.selected_upstream_model_id,
            "model_selection_reason": self.model_selection_reason,
            "model_catalog_version": self.model_catalog_version,
            "contract_rewrite_receipt": (
                None
                if self.contract_rewrite_receipt is None
                else dict(self.contract_rewrite_receipt)
            ),
            "snapshot_unchanged": self.snapshot_unchanged,
            "problem_unchanged": self.problem_unchanged,
        }


@dataclass(frozen=True, slots=True)
class PairedAgentGraphIntervention:
    """Atomic keep/switch actions plus their translation receipt."""

    keep: AgentAction
    switch: AgentAction
    receipt: ProbeInterventionReceipt


def _changed_decision_field(current: DecisionKey, candidate: DecisionKey) -> str:
    if not isinstance(current, DecisionKey) or not isinstance(candidate, DecisionKey):
        raise TypeError("current and candidate must be DecisionKey values")
    if current.task_family != candidate.task_family:
        raise ProbeInterventionError("paired intervention cannot change task_family")
    if current.stage != candidate.stage:
        raise ProbeInterventionError("paired intervention cannot change stage")
    changed = [
        field_name
        for field_name in _DECISION_FIELDS
        if getattr(current, field_name) != getattr(candidate, field_name)
    ]
    if len(changed) != 1:
        raise ProbeInterventionError(
            "paired intervention must change exactly one DecisionKey field; "
            f"found {len(changed)}"
        )
    return changed[0]


def _without_parse_provenance(action: AgentAction, **changes: object) -> AgentAction:
    """Return a generated action whose raw text offsets cannot be mistaken for policy output."""

    return replace(
        action,
        raw_json="",
        consumed_start=0,
        consumed_end=0,
        **changes,
    )


def _snapshot_node(snapshot: AgentWorkflowSnapshot, agent_id: str):
    matches = [node for node in snapshot.graph.nodes if node.id == agent_id]
    if len(matches) != 1:
        raise ProbeInterventionError(
            f"pre-execution snapshot must contain exactly one agent {agent_id!r}"
        )
    return matches[0]


def _action_agent_id(action: AgentAction) -> str:
    if action.action_type not in {AgentActionType.ADD_AGENT, AgentActionType.MODIFY_AGENT}:
        raise ProbeInterventionError(
            "model and role interventions require add_agent or modify_agent"
        )
    if action.agent_id is None:
        raise ProbeInterventionError("Agent action has no target agent_id")
    return action.agent_id


def _edge_type(action: AgentAction) -> str:
    if action.action_type is not AgentActionType.SET_RELATION:
        raise ProbeInterventionError("edge_type intervention requires set_relation")
    if action.source_to_target is None or action.target_to_source is None:
        raise ProbeInterventionError("set_relation action is missing its two direction bits")
    if action.source_to_target and action.target_to_source:
        return "bidirectional"
    if action.source_to_target or action.target_to_source:
        return "unidirectional"
    return "independent"


def _relation_bits_for(
    candidate_edge_type: str,
    *,
    action: AgentAction,
    snapshot: AgentWorkflowSnapshot,
) -> tuple[bool, bool]:
    if candidate_edge_type == "independent":
        return False, False
    if candidate_edge_type == "bidirectional":
        return True, True
    if candidate_edge_type != "unidirectional":
        raise ProbeInterventionError(f"unsupported edge_type: {candidate_edge_type!r}")

    # Preserve an explicit direction whenever either the actual action or the
    # pre-action relation supplies one.  If neither does, AgentAction's endpoint
    # names are the only available orientation: source_id -> target_id.
    if bool(action.source_to_target) != bool(action.target_to_source):
        return bool(action.source_to_target), bool(action.target_to_source)
    if action.source_id is None or action.target_id is None:
        raise ProbeInterventionError("set_relation action has no relation endpoints")
    matching = [
        relation
        for relation in snapshot.graph.relations
        if relation.unordered_key
        == tuple(sorted((action.source_id, action.target_id)))
    ]
    if len(matching) > 1:
        raise ProbeInterventionError("pre-execution snapshot has duplicate relation records")
    if matching:
        bits = matching[0].oriented(action.source_id, action.target_id)
        if bits.source_to_target != bits.target_to_source:
            return bits.source_to_target, bits.target_to_source
    return True, False


def _receipt_mapping(receipt: object) -> Mapping[str, Any]:
    to_dict = getattr(receipt, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if not isinstance(value, Mapping):
            raise ProbeInterventionError("contract rewrite receipt to_dict() was not a mapping")
        return dict(value)
    if isinstance(receipt, Mapping):
        return dict(receipt)
    raise ProbeInterventionError("contract rewriter did not return a serializable receipt")


class ProbeInterventionTranslator:
    """Compile a one-field ``DecisionKey`` switch to two atomic Canvas edits."""

    def __init__(
        self,
        *,
        contract_rewriter: Optional[ContractRewriter] = None,
        model_catalog: Sequence[str] = (),
        model_catalog_version: Optional[str] = None,
    ) -> None:
        normalized: list[str] = []
        for value in model_catalog:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model_catalog entries must be non-empty strings")
            model_id = value.strip()
            if model_id not in normalized:
                normalized.append(model_id)
        if normalized and (
            not isinstance(model_catalog_version, str) or not model_catalog_version.strip()
        ):
            raise ValueError(
                "a non-empty model_catalog_version is required for a frozen model_catalog"
            )
        if not normalized and model_catalog_version is not None:
            raise ValueError("model_catalog_version requires a non-empty model_catalog")
        self.contract_rewriter = contract_rewriter
        self.model_catalog = tuple(normalized)
        self.model_catalog_version = (
            None if model_catalog_version is None else model_catalog_version.strip()
        )

    async def translate(
        self,
        action: AgentAction,
        snapshot: AgentWorkflowSnapshot,
        current: DecisionKey,
        candidate: DecisionKey,
        *,
        seed: Optional[int] = None,
    ) -> PairedAgentGraphIntervention:
        if not isinstance(action, AgentAction):
            raise TypeError("action must be AgentAction")
        if not isinstance(snapshot, AgentWorkflowSnapshot):
            raise TypeError("snapshot must be AgentWorkflowSnapshot")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")

        original_snapshot = snapshot
        original_problem = snapshot.problem
        changed_field = _changed_decision_field(current, candidate)
        keep = action
        target_agent_id: Optional[str] = None
        relation_endpoints: Optional[tuple[str, str]] = None
        upstream_agent_id: Optional[str] = None
        downstream_agent_id: Optional[str] = None
        keep_upstream_model_id: Optional[str] = None
        selected_upstream_model_id: Optional[str] = None
        model_selection_reason: Optional[str] = None
        rewrite_receipt: Optional[Mapping[str, Any]] = None

        if changed_field == "model_id":
            target_agent_id = _action_agent_id(action)
            if action.model_id != current.model_id:
                raise ProbeInterventionError(
                    "actual Agent action model_id does not match the current DecisionKey"
                )
            if self.model_catalog and candidate.model_id not in self.model_catalog:
                raise ProbeInterventionError("candidate model_id is absent from the frozen catalog")
            switch = _without_parse_provenance(action, model_id=candidate.model_id)
            self._assert_action_delta(keep, switch, {"model_id"})

        elif changed_field == "role_cluster":
            target_agent_id = _action_agent_id(action)
            if action.contract is None:
                raise ProbeInterventionError(
                    "role_cluster intervention requires an explicit free-text contract action"
                )
            if candidate.role_cluster not in ROLE_CLUSTERS:
                raise ProbeInterventionError("candidate role_cluster is outside the fixed catalog")
            if self.contract_rewriter is None:
                raise ProbeInterventionError(
                    "role_cluster intervention requires an injected LLM contract rewriter"
                )
            rewritten = await self.contract_rewriter.rewrite(
                action.contract,
                target_role_cluster=candidate.role_cluster,
                task_family=current.task_family,
                seed=seed,
            )
            contract = getattr(rewritten, "contract", None)
            if not isinstance(contract, str) or not contract.strip():
                raise ProbeInterventionError("contract rewriter returned an empty contract")
            if contract.strip() == action.contract.strip():
                raise ProbeInterventionError("contract rewriter did not change the free contract")
            rewrite_receipt = _receipt_mapping(getattr(rewritten, "receipt", None))
            switch = _without_parse_provenance(action, contract=contract.strip())
            self._assert_action_delta(keep, switch, {"contract"})

        elif changed_field == "edge_type":
            if action.action_type is not AgentActionType.SET_RELATION:
                raise ProbeInterventionError("edge_type intervention requires set_relation")
            if action.source_id is None or action.target_id is None:
                raise ProbeInterventionError("set_relation action has no relation endpoints")
            if _edge_type(action) != current.edge_type:
                raise ProbeInterventionError(
                    "actual relation bits do not match the current DecisionKey edge_type"
                )
            relation_endpoints = (action.source_id, action.target_id)
            source_to_target, target_to_source = _relation_bits_for(
                candidate.edge_type,
                action=action,
                snapshot=snapshot,
            )
            switch = _without_parse_provenance(
                action,
                source_to_target=source_to_target,
                target_to_source=target_to_source,
            )
            self._assert_action_delta(
                keep,
                switch,
                {"source_to_target", "target_to_source"},
                require_all=False,
            )
            if _edge_type(switch) != candidate.edge_type:
                raise ProbeInterventionError("translated relation does not match candidate edge_type")

        else:  # same_model_as_upstream
            (
                switch,
                upstream_agent_id,
                downstream_agent_id,
                keep_upstream_model_id,
                selected_upstream_model_id,
                model_selection_reason,
            ) = self._translate_same_model(action, snapshot, current, candidate)
            target_agent_id = upstream_agent_id
            self._assert_action_delta(keep, switch, {"model_id"})

        snapshot_unchanged = snapshot == original_snapshot
        problem_unchanged = snapshot.problem == original_problem
        if not snapshot_unchanged or not problem_unchanged:
            raise AssertionError("probe translation mutated the pre-execution Canvas snapshot")
        receipt = ProbeInterventionReceipt(
            version=INTERVENTION_RECEIPT_VERSION,
            snapshot_id=snapshot.snapshot_id,
            changed_field=changed_field,
            current_key=current,
            candidate_key=candidate,
            keep_action=keep,
            switch_action=switch,
            target_agent_id=target_agent_id,
            relation_endpoints=relation_endpoints,
            upstream_agent_id=upstream_agent_id,
            downstream_agent_id=downstream_agent_id,
            keep_upstream_model_id=keep_upstream_model_id,
            selected_upstream_model_id=selected_upstream_model_id,
            model_selection_reason=model_selection_reason,
            model_catalog_version=self.model_catalog_version,
            contract_rewrite_receipt=rewrite_receipt,
            snapshot_unchanged=snapshot_unchanged,
            problem_unchanged=problem_unchanged,
        )
        return PairedAgentGraphIntervention(keep=keep, switch=switch, receipt=receipt)

    def _translate_same_model(
        self,
        action: AgentAction,
        snapshot: AgentWorkflowSnapshot,
        current: DecisionKey,
        candidate: DecisionKey,
    ) -> tuple[AgentAction, str, str, str, str, str]:
        # The specification defines this candidate as an intervention on the
        # direct upstream model at that upstream edit's pre-execution snapshot.
        # Therefore the actual action must target that upstream node, while the
        # unchanged DecisionKey.model_id identifies its unique direct successor.
        if action.action_type is not AgentActionType.MODIFY_AGENT or action.agent_id is None:
            raise ProbeInterventionError(
                "same_model_as_upstream intervention requires the upstream modify_agent action"
            )
        upstream = _snapshot_node(snapshot, action.agent_id)
        downstream_ids: list[str] = []
        for relation in snapshot.graph.relations:
            for source_id, target_id in relation.directed_edges():
                if source_id == upstream.id:
                    downstream_ids.append(target_id)
        unique_downstream = list(dict.fromkeys(downstream_ids))
        if len(unique_downstream) != 1:
            raise ProbeInterventionError(
                "same_model_as_upstream requires exactly one direct downstream node "
                "at the upstream pre-execution snapshot"
            )
        downstream = _snapshot_node(snapshot, unique_downstream[0])
        if downstream.model_id != current.model_id:
            raise ProbeInterventionError(
                "the direct downstream model does not match DecisionKey.model_id"
            )
        direct_upstream_ids: list[str] = []
        for relation in snapshot.graph.relations:
            for source_id, target_id in relation.directed_edges():
                if target_id == downstream.id:
                    direct_upstream_ids.append(source_id)
        if list(dict.fromkeys(direct_upstream_ids)) != [upstream.id]:
            raise ProbeInterventionError(
                "the focal node must have exactly one direct upstream node in the "
                "pre-execution snapshot"
            )
        effective_keep_model = action.model_id or upstream.model_id
        observed_same = effective_keep_model == downstream.model_id
        if observed_same is not current.same_model_as_upstream:
            raise ProbeInterventionError(
                "pre-execution upstream binding disagrees with current DecisionKey"
            )

        if candidate.same_model_as_upstream:
            selected_model = downstream.model_id
            selection_reason = (
                "candidate requires the directly upstream node to use the focal "
                "node's frozen model_id"
            )
        else:
            selected_model = next(
                (model_id for model_id in self.model_catalog if model_id != downstream.model_id),
                "",
            )
            if not selected_model:
                raise ProbeInterventionError(
                    "a frozen alternative model is required to make the upstream model different"
                )
            selection_reason = (
                "candidate requires a different upstream model; selected the first "
                "eligible different model in frozen catalog order"
            )
        switch = _without_parse_provenance(action, model_id=selected_model)
        if (selected_model == downstream.model_id) is not candidate.same_model_as_upstream:
            raise AssertionError("translated upstream model does not realize the candidate binding")
        return (
            switch,
            upstream.id,
            downstream.id,
            effective_keep_model,
            selected_model,
            selection_reason,
        )

    @staticmethod
    def _assert_action_delta(
        keep: AgentAction,
        switch: AgentAction,
        allowed_fields: set[str],
        *,
        require_all: bool = True,
    ) -> None:
        # Parse offsets and raw JSON are provenance, not Canvas semantics; a
        # generated switch deliberately clears them.  Every semantic action
        # field outside the translated factor must remain byte-for-byte equal.
        semantic_fields = (
            "action_type",
            "agent_id",
            "model_id",
            "contract",
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
        )
        changed = {
            field_name
            for field_name in semantic_fields
            if getattr(keep, field_name) != getattr(switch, field_name)
        }
        if not changed or not changed.issubset(allowed_fields):
            raise ProbeInterventionError(
                "translated actions changed fields outside the selected intervention: "
                + ",".join(sorted(changed))
            )
        if require_all and changed != allowed_fields:
            raise ProbeInterventionError(
                "translated action did not realize the complete selected intervention"
            )


async def translate_probe_intervention(
    action: AgentAction,
    snapshot: AgentWorkflowSnapshot,
    current: DecisionKey,
    candidate: DecisionKey,
    *,
    contract_rewriter: Optional[ContractRewriter] = None,
    model_catalog: Sequence[str] = (),
    model_catalog_version: Optional[str] = None,
    seed: Optional[int] = None,
) -> PairedAgentGraphIntervention:
    """Functional wrapper around :class:`ProbeInterventionTranslator`."""

    return await ProbeInterventionTranslator(
        contract_rewriter=contract_rewriter,
        model_catalog=model_catalog,
        model_catalog_version=model_catalog_version,
    ).translate(action, snapshot, current, candidate, seed=seed)


__all__ = [
    "ContractRewriter",
    "INTERVENTION_RECEIPT_VERSION",
    "PairedAgentGraphIntervention",
    "ProbeInterventionError",
    "ProbeInterventionReceipt",
    "ProbeInterventionTranslator",
    "translate_probe_intervention",
]
