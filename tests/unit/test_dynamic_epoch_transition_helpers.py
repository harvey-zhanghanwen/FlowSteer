"""Unit tests for runner-facing dynamic epoch transition helpers."""

from __future__ import annotations

import json

import pytest

from src.interactive.exploration.combination_ledger import DecisionKey, SurfaceSignal
from src.interactive.exploration.dynamic_training_runtime import (
    DynamicLedgerBatch,
    DynamicLedgerEpochCoordinator,
    close_dynamic_epoch,
    freeze_initial_epoch,
)
from src.interactive.exploration.ledger_epoch import (
    LedgerEpoch,
    LedgerTrajectoryRecord,
    StepRecord,
)
from src.interactive.exploration.ledger_probe import make_probe_branch_ids
from src.interactive.exploration.role_classifier import RoleClassifier
from src.interactive.exploration.rollout_hook import HotpotQADynamicLedgerHook
from src.interactive.persistence.ids import stable_id
from src.interactive.records import (
    EvaluationReceipt,
    ProbeRecord,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.skills.schema import SkillEvidence, SkillRecord, SkillStatus
from src.interactive.versioning import VersionBundle


def _versions(policy: str = "policy-e0") -> VersionBundle:
    return VersionBundle(
        policy=policy,
        model_catalog="catalog-v1",
        evaluator="hotpotqa-em-f1-v1",
        prompt="prompt-v1",
        tool="tool-v1",
        encoder="encoder-v1",
        feature_schema="hotpotqa-decision-key-v1",
    )


def _key(model_id: str) -> DecisionKey:
    return DecisionKey(
        task_family="hotpotqa",
        role_cluster="solve",
        model_id=model_id,
        edge_type="unidirectional",
        same_model_as_upstream=False,
        stage="other",
    )


def _ledger_step(trajectory_id: str) -> StepRecord:
    current = _key("model-a")
    return StepRecord(
        step_id=f"step-{trajectory_id}",
        trajectory_id=trajectory_id,
        snapshot_id=f"snapshot-{trajectory_id}",
        key=current,
        candidates=(_key("model-b"),),
        ledger_readout={"pre": {}, "post": None, "skills": None},
        director_action="continue",
        surface=SurfaceSignal.for_key("verifier_pass", True, current),
    )


def _turn(policy: str) -> TurnRecord:
    graph_snapshot = {"nodes": [], "edges": [], "output_node_id": None}
    snapshot_id = stable_id(
        "snapshot",
        {
            "revision": 0,
            "graph": graph_snapshot,
            "previous_snapshot_id": None,
        },
    )
    return TurnRecord(
        turn_id="turn-0",
        round_index=0,
        prompt="prompt",
        policy_response='{"action":"finish"}',
        prompt_token_ids=(10,),
        output_token_ids=(11,),
        behavior_log_probs=(-0.2,),
        executed_prefix_tokens=1,
        action={"action": "finish"},
        canvas_feedback="finished",
        graph_revision=0,
        graph_snapshot=graph_snapshot,
        policy_version=policy,
        graph_snapshot_id=snapshot_id,
        previous_graph_snapshot_id=None,
        receipt_verified=True,
    )


def _runtime_trajectory(
    epoch: LedgerEpoch,
    trajectory_id: str,
    *,
    reward: int,
    forced_probe: bool,
    probe_id: str | None = None,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        task=TaskRecord(
            task_id="problem-1",
            question="Which entity is requested?",
            ground_truth="answer",
            split="train",
        ),
        group_id="problem-1:condition",
        condition_id=epoch.condition.condition_id,
        rollout_id=f"rollout-{trajectory_id}",
        versions=epoch.condition.versions,
        turns=(_turn(epoch.condition.versions.policy),),
        final_answer="answer" if reward else "wrong",
        evaluation=EvaluationReceipt(
            evaluator_version=epoch.condition.versions.evaluator,
            valid=True,
            reward=float(reward),
            metrics={"exact_match": float(reward), "token_f1": float(reward)},
        ),
        termination_reason="finish",
        explicit_finish=True,
        forced_probe=forced_probe,
        intervention={} if not forced_probe else {"probe_id": probe_id},
    )


def _ledger_trajectory(
    epoch: LedgerEpoch,
    trajectory_id: str,
    *,
    reward: int,
    natural: bool,
    probe_id: str | None = None,
) -> LedgerTrajectoryRecord:
    return LedgerTrajectoryRecord(
        trajectory_id=trajectory_id,
        problem_id="problem-1",
        task_split="train",
        group_id="problem-1:condition",
        condition_id=epoch.condition.condition_id,
        policy_version=epoch.condition.versions.policy,
        steps=(_ledger_step(trajectory_id),),
        is_natural=natural,
        terminal_reward=reward,
        evaluator_valid=True,
        grpo_eligible=natural,
        probe_id=probe_id,
    )


def _probe(epoch: LedgerEpoch) -> tuple[ProbeRecord, tuple[str, ...]]:
    probe_id = "probe-1"
    branch_ids = make_probe_branch_ids(probe_id, seed=17)
    probe = ProbeRecord(
        probe_id=probe_id,
        problem_id="problem-1",
        task_split="train",
        snapshot_id="shared-snapshot",
        policy_version=epoch.condition.versions.policy,
        state_features={"is_audit": False},
        incumbent_action=_key("model-a").to_dict(),
        candidate_action=_key("model-b").to_dict(),
        sampling_probability=0.1,
        incumbent_returns=(0, 0, 1),
        candidate_returns=(1, 1, 1),
        executor_versions={"model-a": "v1", "model-b": "v1"},
        evaluator_version=epoch.condition.versions.evaluator,
        feature_schema_version=epoch.condition.versions.feature_schema,
        branch_order=branch_ids.execution_order,
    )
    return probe, branch_ids.all_ids


def _reward_for_branch(probe: ProbeRecord, branch_id: str) -> int:
    index = int(branch_id.rsplit(":", 1)[1])
    values = (
        probe.incumbent_returns
        if ":keep:" in branch_id
        else probe.candidate_returns
    )
    return int(values[index])


def _batch() -> tuple[DynamicLedgerBatch, tuple[str, ...]]:
    epoch = freeze_initial_epoch(
        versions=_versions(), model_ids=("model-a", "model-b")
    )
    natural_id = "natural-1"
    natural = _runtime_trajectory(
        epoch, natural_id, reward=1, forced_probe=False
    )
    natural_ledger = _ledger_trajectory(
        epoch, natural_id, reward=1, natural=True
    )
    probe, branch_ids = _probe(epoch)
    branches = tuple(
        _runtime_trajectory(
            epoch,
            branch_id,
            reward=_reward_for_branch(probe, branch_id),
            forced_probe=True,
            probe_id=probe.probe_id,
        )
        for branch_id in branch_ids
    )
    branch_ledgers = tuple(
        _ledger_trajectory(
            epoch,
            branch_id,
            reward=_reward_for_branch(probe, branch_id),
            natural=False,
            probe_id=probe.probe_id,
        )
        for branch_id in branch_ids
    )
    return (
        DynamicLedgerBatch(
            epoch=epoch,
            natural_trajectories=(natural,),
            natural_ledger_records=(natural_ledger,),
            natural_sidecars=(),
            intervention_trajectories=branches,
            intervention_ledger_records=branch_ledgers,
            probe_records=(probe,),
        ),
        branch_ids,
    )


def _skill_evidence() -> SkillEvidence:
    return SkillEvidence(
        baseline="same-snapshot paired intervention",
        paired_effect_mean=0.2,
        calibrated_lower=0.1,
        calibrated_upper=0.3,
        effective_pairs=1,
        independent_problem_ids=(),
        discovery_problem_ids=("train-evidence-1",),
        validation_problem_ids=(),
        validation_splits=(),
        heldout_task_families=(),
        empirical_coverage=0.0,
        harm_probability=0.1,
    )


def _skill(status: SkillStatus) -> SkillRecord:
    kwargs = {}
    if status is SkillStatus.ACTIVE:
        kwargs = {
            "activated_epoch": 1,
            "gate_config": {"alpha": 0.05},
            "gate_receipt": "gate-receipt",
        }
    elif status in {SkillStatus.SUSPENDED, SkillStatus.RETIRED}:
        kwargs = {"suspended_reason": "held-out gate result"}
    return SkillRecord(
        skill_id=f"skill-{status.value}",
        version=1,
        status=status,
        condition={
            "task_family": "hotpotqa",
            "graph_stage": "other",
            "role_cluster": "solve",
        },
        action={"model_id": "model-b"},
        evidence=_skill_evidence(),
        versions=_versions("policy-e1"),
        created_epoch=0,
        eligible_epoch=1,
        **kwargs,
    )


def test_close_dynamic_epoch_persists_json_receipts_and_all_skill_counts(
    tmp_path,
) -> None:
    batch, branch_ids = _batch()
    skills = tuple(_skill(status) for status in SkillStatus)
    result = close_dynamic_epoch(
        batch,
        next_policy_version="policy-e1",
        next_skills=skills,
        natural_grpo_trajectory_ids=("natural-1",),
        intervention_exclusion_trajectory_ids=tuple(reversed(branch_ids)),
        next_epoch_directory=tmp_path / "epoch-1",
    )

    assert result.next_epoch.condition.versions.policy == "policy-e1"
    assert result.receipt.skill_status_counts == {
        "active": 1,
        "candidate": 1,
        "retired": 1,
        "suspended": 1,
    }
    assert result.receipt.heldout_calibration == {
        "status": "pending",
        "coverage": None,
        "source": None,
        "applied_to_policy_refresh": False,
    }
    assert result.transition.summary.grpo_trajectory_ids == ("natural-1",)
    assert set(result.transition.summary.excluded_intervention_ids) == set(branch_ids)
    recovery = result.recovery_mapping()
    assert recovery["status"] == "ready"
    assert recovery["next_epoch_receipt_path"] == str(
        tmp_path / "epoch-1" / "receipt.json"
    )
    assert LedgerEpoch.load(recovery["next_epoch_receipt_path"]).condition == (
        result.next_epoch.condition
    )
    metrics = result.metrics_mapping()
    assert metrics["natural_grpo_trajectory_count"] == 1
    assert metrics["intervention_exclusion_trajectory_count"] == 6
    assert metrics["heldout_calibration_status"] == "pending"
    assert metrics["heldout_coverage"] is None
    json.dumps(result.receipt.to_dict())
    json.dumps(recovery)
    json.dumps(metrics)


def test_close_dynamic_epoch_rejects_runner_id_partition_mismatches() -> None:
    batch, branch_ids = _batch()
    with pytest.raises(ValueError, match="natural GRPO trajectory IDs"):
        close_dynamic_epoch(
            batch,
            next_policy_version="policy-e1",
            next_skills=(),
            natural_grpo_trajectory_ids=("wrong-natural-id",),
            intervention_exclusion_trajectory_ids=branch_ids,
        )
    with pytest.raises(ValueError, match="intervention exclusion IDs"):
        close_dynamic_epoch(
            batch,
            next_policy_version="policy-e1",
            next_skills=(),
            natural_grpo_trajectory_ids=("natural-1",),
            intervention_exclusion_trajectory_ids=branch_ids[:-1],
        )


class _UnusedCompletion:
    async def complete(self, request):  # pragma: no cover - not called here
        raise AssertionError("public hook construction must not call a model")


class _UnusedRewriter:
    async def rewrite(self, *args, **kwargs):  # pragma: no cover - not called here
        raise AssertionError("public hook construction must not call a model")


def test_coordinator_exposes_public_make_hook_without_model_io() -> None:
    epoch = freeze_initial_epoch(versions=_versions(), model_ids=("model-a",))
    coordinator = DynamicLedgerEpochCoordinator(
        epoch=epoch,
        role_classifier=RoleClassifier(_UnusedCompletion()),
        contract_rewriter=_UnusedRewriter(),
        model_catalog=("model-a",),
        model_catalog_version="catalog-v1",
        max_rounds=4,
        seed=31,
    )
    hook = coordinator.make_hook(7)
    assert isinstance(hook, HotpotQADynamicLedgerHook)
    assert hook.epoch.condition == epoch.condition
    assert hook.random_seed == 38
    assert coordinator._hook(7).random_seed == 38
