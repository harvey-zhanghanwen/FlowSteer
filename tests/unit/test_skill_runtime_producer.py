"""Focused tests for the real probe-to-Skill runtime producer."""

from __future__ import annotations

import pytest

from src.interactive.exploration.combination_ledger import (
    CombinationPosterior,
    DecisionKey,
)
from src.interactive.exploration.ledger_probe import make_probe_branch_ids
from src.interactive.records import ProbeRecord
from src.interactive.skills.runtime_producer import (
    SkillRuntimeProducerConfig,
    produce_skill_runtime_update,
)
from src.interactive.skills.schema import SkillStatus
from src.interactive.versioning import VersionBundle


def _versions(policy: str) -> VersionBundle:
    return VersionBundle(
        policy=policy,
        model_catalog="catalog-v1",
        evaluator="skillflow.training.reward.v1",
        prompt="agentgraph.director.minimal.v2",
        tool="agentgraph.executor-react.v2",
        encoder="ledger-role-classifier-v1",
        feature_schema="dynamic-combination-decision-key-v1",
    )


def _key(model_id: str) -> DecisionKey:
    return DecisionKey(
        task_family="hotpotqa",
        role_cluster="retrieve",
        model_id=model_id,
        edge_type="independent",
        same_model_as_upstream=False,
        stage="other",
    )


def _probe(
    index: int,
    *,
    split: str,
    policy: str,
    problem_id: str | None = None,
    positive: bool = True,
) -> ProbeRecord:
    probe_id = f"{split}-probe-{index:03d}"
    branches = make_probe_branch_ids(probe_id, seed=index)
    return ProbeRecord(
        probe_id=probe_id,
        problem_id=problem_id or f"{split}-problem-{index:03d}",
        task_split=split,
        snapshot_id=f"{split}-snapshot-{index:03d}",
        policy_version=policy,
        state_features={"is_audit": False},
        incumbent_action=_key("model-a").to_dict(),
        candidate_action=_key("model-b").to_dict(),
        sampling_probability=0.1,
        incumbent_returns=(0, 0, 0) if positive else (1, 1, 1),
        candidate_returns=(1, 1, 1) if positive else (0, 0, 0),
        executor_versions={"model-a": "v1", "model-b": "v1"},
        evaluator_version="skillflow.training.reward.v1",
        feature_schema_version="dynamic-combination-decision-key-v1",
        branch_order=branches.execution_order,
    )


def _publication_ledger(discovery: tuple[ProbeRecord, ...]) -> CombinationPosterior:
    ledger = CombinationPosterior("policy-e0", epoch=0)
    ledger.register_key(_key("model-a"))
    ledger.register_key(_key("model-b"))
    for record in discovery:
        ledger.update_probe(
            DecisionKey.from_dict(record.incumbent_action),
            DecisionKey.from_dict(record.candidate_action),
            record.incumbent_returns,
            record.candidate_returns,
        )
    ledger.empirical_bayes_update()
    ledger.refresh_policy("policy-e1")
    return ledger


def test_no_discovery_qualified_rule_completes_without_fake_confirmation() -> None:
    discovery = tuple(
        _probe(index, split="train", policy="policy-e0") for index in range(9)
    )
    result = produce_skill_runtime_update(
        _publication_ledger(discovery),
        discovery_probes=discovery,
        confirmation_probes=(),
        publication_versions=_versions("policy-e1"),
        discovery_epoch=0,
    )
    assert result.skills == ()
    assert result.receipt.status == "complete_no_qualified_candidate"
    assert result.receipt.producer_complete is True
    assert result.receipt.confirmation_probe_count == 0
    assert result.receipt.grpo_reward_contribution == 0.0
    assert result.receipt.ttb_enabled is False
    assert result.receipt.to_dict()["skill_status_counts"]["active"] == 0


def test_qualified_discovery_without_twenty_heldout_problems_stays_pending() -> None:
    discovery = tuple(
        _probe(index, split="train", policy="policy-e0") for index in range(10)
    )
    result = produce_skill_runtime_update(
        _publication_ledger(discovery),
        discovery_probes=discovery,
        confirmation_probes=tuple(
            _probe(index, split="validation", policy="policy-e1")
            for index in range(19)
        ),
        publication_versions=_versions("policy-e1"),
        discovery_epoch=0,
    )
    assert result.receipt.status == "pending_heldout_confirmation"
    assert result.receipt.producer_complete is False
    assert result.receipt.discovery_qualified_rule_count == 1
    assert len(result.receipt.pending_confirmation_rule_ids) == 1
    assert [skill.status for skill in result.skills] == [SkillStatus.CANDIDATE]


def test_real_twenty_problem_confirmation_activates_through_publication_gate() -> None:
    discovery = tuple(
        _probe(index, split="train", policy="policy-e0") for index in range(10)
    )
    confirmation = tuple(
        _probe(index, split="validation", policy="policy-e1")
        for index in range(20)
    )
    result = produce_skill_runtime_update(
        _publication_ledger(discovery),
        discovery_probes=discovery,
        confirmation_probes=confirmation,
        publication_versions=_versions("policy-e1"),
        discovery_epoch=0,
        config=SkillRuntimeProducerConfig(),
    )
    assert result.receipt.status == "complete"
    assert result.receipt.active_rule_count == 1
    assert len(result.skills) == 1
    active = result.skills[0]
    assert active.status is SkillStatus.ACTIVE
    assert active.activated_epoch == 1
    assert active.gate_receipt
    assert active.evidence.effective_pairs == 20
    assert len(active.evidence.independent_problem_ids) == 20
    assert set(active.evidence.evidence_ids) == set(result.evidence_records)


def test_confirmation_split_policy_and_problem_isolation_fail_closed() -> None:
    discovery = tuple(
        _probe(index, split="train", policy="policy-e0") for index in range(10)
    )
    ledger = _publication_ledger(discovery)
    with pytest.raises(ValueError, match="publication policy"):
        produce_skill_runtime_update(
            ledger,
            discovery_probes=discovery,
            confirmation_probes=(
                _probe(0, split="validation", policy="wrong-policy"),
            ),
            publication_versions=_versions("policy-e1"),
            discovery_epoch=0,
        )
    with pytest.raises(ValueError, match="problems overlap"):
        produce_skill_runtime_update(
            ledger,
            discovery_probes=discovery,
            confirmation_probes=(
                _probe(
                    0,
                    split="validation",
                    policy="policy-e1",
                    problem_id="train-problem-000",
                ),
            ),
            publication_versions=_versions("policy-e1"),
            discovery_epoch=0,
        )
