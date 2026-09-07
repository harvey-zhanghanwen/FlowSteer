"""Mock-only tests for one-field AgentGraph paired interventions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_workflow_env import AgentWorkflowSnapshot
from src.interactive.exploration.combination_ledger import DecisionKey
from src.interactive.exploration.probe_intervention import (
    INTERVENTION_RECEIPT_VERSION,
    ProbeInterventionError,
    ProbeInterventionTranslator,
)
from src.interactive.exploration.role_classifier import (
    ContractRoleRewriter,
    RoleCompletionResponse,
)


def key(**changes: object) -> DecisionKey:
    payload: dict[str, object] = {
        "task_family": "hotpotqa",
        "role_cluster": "solve",
        "model_id": "model-a",
        "edge_type": "unidirectional",
        "same_model_as_upstream": True,
        "stage": "other",
    }
    payload.update(changes)
    return DecisionKey(**payload)  # type: ignore[arg-type]


def snapshot(
    *,
    upstream_model: str = "model-a",
    downstream_model: str = "model-a",
    extra_downstream: bool = False,
) -> AgentWorkflowSnapshot:
    nodes = [
        AgentNode("upstream", upstream_model, "Produce a supported candidate."),
        AgentNode("focal", downstream_model, "Derive the requested answer."),
        AgentNode("other", "model-c", "Independently inspect the evidence."),
    ]
    relations = [AgentRelation("upstream", "focal", True, False)]
    if extra_downstream:
        nodes.append(AgentNode("focal-2", downstream_model, "Derive another answer."))
        relations.append(AgentRelation("upstream", "focal-2", True, False))
    graph = AgentGraph(
        nodes=nodes,
        relations=relations,
        output_agent_id="focal",
        revision=7,
    ).snapshot()
    return AgentWorkflowSnapshot(
        problem="Which entity satisfies both supplied relations?",
        graph=graph,
        turn_count=4,
        finished=False,
        last_feedback="accepted",
    )


@dataclass(frozen=True)
class MockRewriteReceipt:
    request_id: str
    target_role: str

    def to_dict(self) -> dict[str, str]:
        return {"request_id": self.request_id, "target_role": self.target_role}


@dataclass(frozen=True)
class MockRewriteResult:
    contract: str
    receipt: MockRewriteReceipt


class MockContractRewriter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def rewrite(
        self,
        original_contract: str,
        *,
        target_role_cluster: str,
        task_family: str,
        seed: int | None = None,
    ) -> MockRewriteResult:
        self.calls.append(
            {
                "original_contract": original_contract,
                "target_role_cluster": target_role_cluster,
                "task_family": task_family,
                "seed": seed,
            }
        )
        return MockRewriteResult(
            contract="Check the supplied evidence and return only a supported conclusion.",
            receipt=MockRewriteReceipt("rewrite-1", target_role_cluster),
        )


class ConcreteRewriteCompletion:
    async def complete(self, request: object) -> RoleCompletionResponse:
        del request
        return RoleCompletionResponse(
            '{"contract":"Inspect the evidence before returning the supported answer."}',
            {"provider_request_id": "mock-provider"},
        )


def test_model_intervention_changes_only_target_node_model() -> None:
    before = snapshot()
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="focal",
        model_id="model-a",
        contract="Keep this free contract unchanged.",
        raw_json='{"action":"modify_agent"}',
        consumed_start=3,
        consumed_end=37,
    )
    result = asyncio.run(
        ProbeInterventionTranslator(
            model_catalog=("model-a", "model-b"),
            model_catalog_version="catalog-e0",
        ).translate(
            action,
            before,
            key(),
            key(model_id="model-b"),
        )
    )

    assert result.keep is action
    assert result.switch.agent_id == "focal"
    assert result.switch.model_id == "model-b"
    assert result.switch.contract == action.contract
    assert result.switch.raw_json == ""
    assert result.receipt.changed_field == "model_id"
    assert result.receipt.target_agent_id == "focal"
    assert result.receipt.version == INTERVENTION_RECEIPT_VERSION
    assert result.receipt.snapshot_unchanged is True
    assert result.receipt.problem_unchanged is True
    assert before == snapshot()


def test_edge_intervention_changes_only_corresponding_two_bit_relation() -> None:
    before = snapshot()
    action = AgentAction(
        AgentActionType.SET_RELATION,
        source_id="upstream",
        target_id="focal",
        source_to_target=True,
        target_to_source=False,
    )
    result = asyncio.run(
        ProbeInterventionTranslator().translate(
            action,
            before,
            key(),
            key(edge_type="bidirectional"),
        )
    )

    assert result.switch.source_id == "upstream"
    assert result.switch.target_id == "focal"
    assert result.switch.source_to_target is True
    assert result.switch.target_to_source is True
    assert result.receipt.relation_endpoints == ("upstream", "focal")
    assert result.receipt.target_agent_id is None
    assert before.graph.relations == snapshot().graph.relations


def test_unidirectional_switch_uses_existing_orientation_when_available() -> None:
    before = snapshot()
    # The current action makes the relation bidirectional.  Switching it back
    # to unidirectional must recover the pre-snapshot upstream -> focal bit.
    action = AgentAction(
        AgentActionType.SET_RELATION,
        source_id="upstream",
        target_id="focal",
        source_to_target=True,
        target_to_source=True,
    )
    result = asyncio.run(
        ProbeInterventionTranslator().translate(
            action,
            before,
            key(edge_type="bidirectional"),
            key(edge_type="unidirectional"),
        )
    )
    assert (result.switch.source_to_target, result.switch.target_to_source) == (
        True,
        False,
    )


def test_role_intervention_uses_injected_llm_rewriter_without_template() -> None:
    rewriter = MockContractRewriter()
    original = "Derive a concise answer from the material passed by upstream Agents."
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="focal",
        model_id="model-a",
        contract=original,
    )
    result = asyncio.run(
        ProbeInterventionTranslator(contract_rewriter=rewriter).translate(
            action,
            snapshot(),
            key(),
            key(role_cluster="verify"),
            seed=31,
        )
    )

    assert result.switch.contract == (
        "Check the supplied evidence and return only a supported conclusion."
    )
    assert result.switch.model_id == action.model_id
    assert rewriter.calls == [
        {
            "original_contract": original,
            "target_role_cluster": "verify",
            "task_family": "hotpotqa",
            "seed": 31,
        }
    ]
    assert result.receipt.contract_rewrite_receipt == {
        "request_id": "rewrite-1",
        "target_role": "verify",
    }


def test_role_intervention_fails_closed_without_llm_rewriter() -> None:
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="focal",
        contract="Produce the answer.",
    )
    with pytest.raises(ProbeInterventionError, match="injected LLM"):
        asyncio.run(
            ProbeInterventionTranslator().translate(
                action,
                snapshot(),
                key(),
                key(role_cluster="verify"),
            )
        )


def test_role_intervention_is_compatible_with_concrete_contract_rewriter_api() -> None:
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="focal",
        contract="Derive the answer from the supplied evidence.",
    )
    rewriter = ContractRoleRewriter(
        ConcreteRewriteCompletion(),
        request_id_factory=lambda: "rewrite-concrete-1",
    )
    result = asyncio.run(
        ProbeInterventionTranslator(contract_rewriter=rewriter).translate(
            action,
            snapshot(),
            key(),
            key(role_cluster="verify"),
            seed=7,
        )
    )
    assert result.switch.contract.startswith("Inspect the evidence")
    assert result.receipt.contract_rewrite_receipt is not None
    assert result.receipt.contract_rewrite_receipt["status"] == "rewritten"


def test_same_model_switch_is_bound_to_unique_direct_upstream_snapshot() -> None:
    before = snapshot(upstream_model="model-a", downstream_model="model-a")
    # The actual action is the keep choice at the upstream node's
    # pre-execution snapshot.  The DecisionKey model remains that of its direct
    # downstream node; only the equality indicator changes.
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="upstream",
        model_id="model-a",
        contract="Produce a supported candidate.",
    )
    result = asyncio.run(
        ProbeInterventionTranslator(
            model_catalog=("model-a", "model-b", "model-c"),
            model_catalog_version="catalog-e0",
        ).translate(
            action,
            before,
            key(same_model_as_upstream=True),
            key(same_model_as_upstream=False),
        )
    )

    assert result.switch.action_type is AgentActionType.MODIFY_AGENT
    assert result.switch.agent_id == "upstream"
    assert result.switch.model_id == "model-b"
    assert result.switch.contract == action.contract
    assert result.receipt.changed_field == "same_model_as_upstream"
    assert result.receipt.upstream_agent_id == "upstream"
    assert result.receipt.downstream_agent_id == "focal"
    assert result.receipt.keep_upstream_model_id == "model-a"
    assert result.receipt.selected_upstream_model_id == "model-b"
    assert result.receipt.model_catalog_version == "catalog-e0"
    assert "frozen catalog order" in result.receipt.model_selection_reason
    assert before == snapshot(upstream_model="model-a", downstream_model="model-a")


def test_same_model_switch_to_equal_uses_downstream_model() -> None:
    before = snapshot(upstream_model="model-b", downstream_model="model-a")
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="upstream",
        model_id="model-b",
    )
    result = asyncio.run(
        ProbeInterventionTranslator(
            model_catalog=("model-a", "model-b"),
            model_catalog_version="catalog-e0",
        ).translate(
            action,
            before,
            key(same_model_as_upstream=False),
            key(same_model_as_upstream=True),
        )
    )
    assert result.switch.model_id == "model-a"
    assert result.receipt.selected_upstream_model_id == "model-a"


def test_same_model_switch_rejects_ambiguous_or_unfrozen_alternative() -> None:
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="upstream",
        model_id="model-a",
    )
    with pytest.raises(ProbeInterventionError, match="one direct downstream"):
        asyncio.run(
            ProbeInterventionTranslator(
                model_catalog=("model-a", "model-b"),
                model_catalog_version="catalog-e0",
            ).translate(
                action,
                snapshot(extra_downstream=True),
                key(),
                key(same_model_as_upstream=False),
            )
        )
    with pytest.raises(ProbeInterventionError, match="frozen alternative"):
        asyncio.run(
            ProbeInterventionTranslator(
                model_catalog=("model-a",),
                model_catalog_version="catalog-e0",
            ).translate(
                action,
                snapshot(),
                key(),
                key(same_model_as_upstream=False),
            )
        )


def test_intervention_rejects_multiple_decision_fields_or_context_change() -> None:
    action = AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id="focal",
        model_id="model-a",
    )
    translator = ProbeInterventionTranslator(
        model_catalog=("model-a", "model-b"),
        model_catalog_version="catalog-e0",
    )
    with pytest.raises(ProbeInterventionError, match="exactly one"):
        asyncio.run(
            translator.translate(
                action,
                snapshot(),
                key(),
                key(model_id="model-b", edge_type="bidirectional"),
            )
        )
    with pytest.raises(ProbeInterventionError, match="task_family"):
        asyncio.run(
            translator.translate(
                action,
                snapshot(),
                key(),
                key(task_family="triviaqa", model_id="model-b"),
            )
        )
