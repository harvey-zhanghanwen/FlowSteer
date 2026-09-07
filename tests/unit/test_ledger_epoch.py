"""Unit coverage for frozen dynamic-ledger epoch coordination."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.interactive.exploration.combination_ledger import (
    CombinationPosterior,
    DecisionKey,
    SurfaceSignal,
)
from src.interactive.exploration.ledger_epoch import (
    LedgerEpoch,
    LedgerTrajectoryRecord,
    StepRecord,
    combine_step_readouts,
)
from src.interactive.exploration.ledger_probe import make_probe_branch_ids
from src.interactive.records import ProbeRecord
from src.interactive.skills.lifecycle import SkillLifecycleManager
from src.interactive.skills.schema import SkillEvidence, SkillRecord, SkillStatus
from src.interactive.skills.validator import SkillEvidenceGate
from src.interactive.versioning import VersionBundle


def versions(policy: str = "policy-e0") -> VersionBundle:
    return VersionBundle(
        policy=policy,
        model_catalog="catalog-v1",
        evaluator="hotpot-em-v1",
        prompt="director-prompt-v1",
        tool="tool-v1",
        encoder="encoder-v1",
        feature_schema="decision-key-v1",
    )


def key(*, model: str = "model-a") -> DecisionKey:
    return DecisionKey(
        task_family="hotpotqa",
        role_cluster="solve",
        model_id=model,
        edge_type="unidirectional",
        same_model_as_upstream=False,
        stage="other",
    )


def step(trajectory_id: str, *, model: str = "model-a", signal: bool = True) -> StepRecord:
    current = key(model=model)
    alternative = key(model="model-b" if model == "model-a" else "model-a")
    return StepRecord(
        step_id=f"step-{trajectory_id}",
        trajectory_id=trajectory_id,
        snapshot_id=f"snapshot-{trajectory_id}",
        key=current,
        candidates=(alternative,),
        ledger_readout={"pre": {"text": "frozen"}, "post": None},
        director_action="continue",
        surface=SurfaceSignal.for_key("verifier_pass", signal, current),
    )


def trajectory(
    epoch: LedgerEpoch,
    trajectory_id: str,
    *,
    natural: bool,
    reward: int,
    probe_id: str | None = None,
    audit: bool = False,
    valid: bool = True,
) -> LedgerTrajectoryRecord:
    return LedgerTrajectoryRecord(
        trajectory_id=trajectory_id,
        problem_id="problem-1",
        task_split="train",
        group_id="group-1",
        condition_id=epoch.condition.condition_id,
        policy_version=epoch.condition.versions.policy,
        steps=(step(trajectory_id),),
        is_natural=natural,
        terminal_reward=reward,
        evaluator_valid=valid,
        grpo_eligible=natural and valid,
        probe_id=probe_id,
        audit=audit,
    )


def make_probe(epoch: LedgerEpoch, probe_id: str = "probe-1") -> tuple[ProbeRecord, tuple[str, ...]]:
    branch_ids = make_probe_branch_ids(probe_id, seed=11)
    record = ProbeRecord(
        probe_id=probe_id,
        problem_id="problem-1",
        task_split="train",
        snapshot_id="shared-snapshot",
        policy_version=epoch.condition.versions.policy,
        state_features={"is_audit": False},
        incumbent_action=key(model="model-a").to_dict(),
        candidate_action=key(model="model-b").to_dict(),
        sampling_probability=0.1,
        incumbent_returns=(0, 0, 1),
        candidate_returns=(1, 1, 1),
        executor_versions={"model-a": "v1", "model-b": "v1"},
        evaluator_version=epoch.condition.versions.evaluator,
        feature_schema_version=epoch.condition.versions.feature_schema,
        branch_order=branch_ids.execution_order,
    )
    return record, branch_ids.execution_order


def active_skill(policy: str, *, skill_id: str = "skill-model-b") -> SkillRecord:
    evidence_ids = tuple(f"evidence-{index}" for index in range(20))
    problem_ids = tuple(f"heldout-{index}" for index in range(20))
    evidence = SkillEvidence(
        baseline="same-snapshot paired intervention",
        paired_effect_mean=0.30,
        calibrated_lower=0.20,
        calibrated_upper=0.40,
        effective_pairs=20,
        independent_problem_ids=problem_ids,
        discovery_problem_ids=tuple(f"train-{index}" for index in range(10)),
        validation_problem_ids=problem_ids,
        validation_splits=("validation",),
        heldout_task_families=("hotpotqa",),
        empirical_coverage=0.95,
        harm_probability=0.01,
        evidence_ids=evidence_ids,
    )
    candidate = SkillRecord(
        skill_id=skill_id,
        version=1,
        status=SkillStatus.CANDIDATE,
        condition={
            "task_family": "hotpotqa",
            "graph_stage": "other",
            "role_cluster": "solve",
        },
        action={"model_id": "model-b"},
        evidence=evidence,
        versions=versions(policy),
        readable_text="prefer model-b for this condition",
        created_epoch=0,
        eligible_epoch=1,
    )
    by_id = {
        evidence_id: {
            "problem_id": problem_id,
            "task_split": "validation",
            "paired_effect": 0.30,
            "policy_version": policy,
            "evaluator_version": "hotpot-em-v1",
            "feature_schema_version": "decision-key-v1",
        }
        for evidence_id, problem_id in zip(evidence_ids, problem_ids)
    }
    gate = SkillEvidenceGate(evidence_lookup=by_id.get)
    return SkillLifecycleManager(gate).activate(candidate, current_epoch=1)


def test_step_and_trajectory_schema_round_trip() -> None:
    current = key()
    ledger = CombinationPosterior("policy-e0")
    epoch = LedgerEpoch.freeze(ledger, (), versions())
    pre = epoch.pre_execution(step_id="s0", current=current, candidates=(key(model="model-b"),))
    post = epoch.post_execution(
        step_id="s0",
        prefix_before=(),
        current=current,
        candidates=(key(model="model-b"),),
        surface=SurfaceSignal.for_key("verifier_pass", True, current),
        remaining_rounds=3,
        rng=np.random.default_rng(7),
    )
    original = StepRecord(
        step_id="s0",
        trajectory_id="t0",
        snapshot_id="snap0",
        key=current,
        candidates=(key(model="model-b"),),
        ledger_readout=combine_step_readouts(pre, post),
        director_action="continue",
        surface=SurfaceSignal.for_key("verifier_pass", True, current),
    )
    restored = StepRecord.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    wrapped = LedgerTrajectoryRecord(
        trajectory_id="t0",
        problem_id="p0",
        task_split="train",
        group_id="g0",
        condition_id=epoch.condition.condition_id,
        policy_version="policy-e0",
        steps=(original,),
        is_natural=True,
        terminal_reward=1,
        evaluator_valid=True,
        grpo_eligible=True,
    )
    assert LedgerTrajectoryRecord.from_dict(wrapped.to_dict()).to_dict() == wrapped.to_dict()


def test_epoch_close_enforces_natural_probe_signal_isolation() -> None:
    epoch = LedgerEpoch.freeze(CombinationPosterior("policy-e0"), (), versions())
    probe, branch_ids = make_probe(epoch)
    natural = trajectory(epoch, "natural-1", natural=True, reward=1)
    branches = tuple(
        trajectory(
            epoch,
            branch_id,
            natural=False,
            reward=(
                probe.candidate_returns[int(branch_id.rsplit(":", 1)[1])]
                if ":switch:" in branch_id
                else probe.incumbent_returns[int(branch_id.rsplit(":", 1)[1])]
            ),
            probe_id=probe.probe_id,
        )
        for branch_id in branch_ids
    )
    transition = epoch.close(
        trajectories=(natural, *branches),
        probes=(probe,),
        next_policy_version="policy-e1",
        next_skills=(),
    )
    summary = transition.summary
    assert summary.contrast_probe_updates == 1
    assert summary.baseline_updates == 7
    assert summary.sensor_updates == 7
    assert summary.grpo_trajectory_ids == ("natural-1",)
    assert set(summary.excluded_intervention_ids) == set(branch_ids)
    next_ledger = transition.next_epoch.ledger
    assert next_ledger.probe_count == 1
    # Prior count four plus all seven evaluator-valid absolute outcomes.
    assert next_ledger.task_baseline["hotpotqa"][1] == 11

    natural_only_epoch = LedgerEpoch.freeze(
        CombinationPosterior("policy-e0"), (), versions()
    )
    natural_only = trajectory(
        natural_only_epoch, "natural-only", natural=True, reward=1
    )
    natural_transition = natural_only_epoch.close(
        trajectories=(natural_only,),
        probes=(),
        next_policy_version="policy-e1",
        next_skills=(),
    )
    assert natural_transition.next_epoch.ledger.probe_count == 0


def test_condition_is_frozen_and_rejects_mixed_policy_or_condition() -> None:
    source = CombinationPosterior("policy-e0")
    epoch = LedgerEpoch.freeze(source, (), versions())
    source.update_probe(key(model="model-a"), key(model="model-b"), (0, 0, 0), (1, 1, 1))
    assert epoch.ledger.probe_count == 0

    valid = trajectory(epoch, "natural-1", natural=True, reward=1)
    wrong_policy = replace(valid, policy_version="policy-other")
    with pytest.raises(ValueError, match="trajectory policy"):
        epoch.close(
            trajectories=(wrong_policy,),
            probes=(),
            next_policy_version="policy-e1",
            next_skills=(),
        )
    wrong_condition = replace(valid, condition_id="condition-other")
    with pytest.raises(ValueError, match="trajectory condition"):
        epoch.close(
            trajectories=(wrong_condition,),
            probes=(),
            next_policy_version="policy-e1",
            next_skills=(),
        )


def test_new_active_skill_is_visible_only_in_next_epoch() -> None:
    epoch = LedgerEpoch.freeze(CombinationPosterior("policy-e0"), (), versions())
    assert epoch.select_skill_texts(
        task_family="hotpotqa",
        role_cluster="solve",
        stage="other",
        available_models=("model-a", "model-b"),
    ).skill_ids == ()

    skill = active_skill("policy-e1")
    natural = trajectory(epoch, "natural-1", natural=True, reward=1)
    transition = epoch.close(
        trajectories=(natural,),
        probes=(),
        next_policy_version="policy-e1",
        next_skills=(skill,),
    )
    selection = transition.next_epoch.select_skill_texts(
        task_family="hotpotqa",
        role_cluster="solve",
        stage="other",
        available_models=("model-a", "model-b"),
    )
    assert selection.skill_ids == (skill.skill_id,)
    assert selection.text.startswith("[Skill]")
    # The source epoch remains frozen and cannot see next-epoch publication.
    assert epoch.condition.visible_skill_ids == ()


def test_save_load_preserves_recoverable_ledger_and_skill_snapshots(tmp_path) -> None:
    ledger = CombinationPosterior("policy-e1")
    ledger.update_probe(
        key(model="model-a"), key(model="model-b"), (0, 0, 1), (1, 1, 1)
    )
    ledger.refresh_policy("policy-e2")
    skill = active_skill("policy-e2")
    epoch = LedgerEpoch.freeze(ledger, (skill,), versions("policy-e2"))
    receipt = epoch.save(tmp_path / "epoch-1")
    restored = LedgerEpoch.load(tmp_path / "epoch-1" / "receipt.json")
    assert restored.condition == epoch.condition
    assert restored.skills[0].to_dict() == skill.to_dict()
    np.testing.assert_allclose(restored.ledger.Lambda, epoch.ledger.Lambda)
    np.testing.assert_allclose(restored.ledger.eta, epoch.ledger.eta)
    assert receipt.ledger_snapshot_id == restored.condition.ledger_snapshot_id
    assert receipt.skill_snapshot_id == restored.condition.skill_snapshot_id
