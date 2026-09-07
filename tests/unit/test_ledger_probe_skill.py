"""Unit tests for LatentLoss Sections 6 and 7 pure-function gates."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.interactive.exploration.ledger_probe import (
    StraddleCandidate,
    assert_probe_data_partition,
    grpo_training_trajectories,
    make_probe_branch_ids,
    natural_trajectories,
    probe_budget,
    select_seeded_audits,
    select_straddle_candidates,
    standard_metric_trajectories,
    validate_probe_record,
)
from src.interactive.records import ProbeRecord
from src.interactive.skills.ledger_miner import (
    ConfirmationEvidence,
    RuleContrast,
    benjamini_hochberg,
    calibrated_interval,
    decide_skill_status,
    empirical_calibration_quantile,
    enumerate_rule_candidates,
)
from src.interactive.skills.schema import SkillStatus


def decision(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task_family": "hotpotqa",
        "role_cluster": "solve",
        "model_id": "model-a",
        "edge_type": "unidirectional",
        "same_model_as_upstream": False,
        "stage": "other",
    }
    value.update(overrides)
    return value


def site(
    trajectory: str,
    step: str,
    risk: float,
    score: float,
) -> StraddleCandidate:
    return StraddleCandidate(
        trajectory_id=trajectory,
        step_id=step,
        snapshot_id=f"snapshot-{trajectory}-{step}",
        risk_probability=risk,
        score=score,
        key_keep=decision(),
        key_switch=decision(model_id="model-b"),
    )


def probe_record(
    *,
    candidate_action: dict[str, object] | None = None,
    branch_order: tuple[str, ...] | None = None,
    incumbent_returns: tuple[int, ...] = (0, 1, 0),
    candidate_returns: tuple[int, ...] = (1, 1, 1),
    is_audit: bool = False,
) -> ProbeRecord:
    branches = make_probe_branch_ids("probe-1", seed=9)
    return ProbeRecord(
        probe_id="probe-1",
        problem_id="problem-1",
        task_split="train",
        snapshot_id="snapshot-1",
        policy_version="policy-e0",
        state_features={"is_audit": is_audit},
        incumbent_action=decision(),
        candidate_action=candidate_action or decision(model_id="model-b"),
        sampling_probability=0.1,
        incumbent_returns=incumbent_returns,
        candidate_returns=candidate_returns,
        executor_versions={"model-a": "v1", "model-b": "v1"},
        evaluator_version="hotpot-em-v1",
        feature_schema_version="ledger-decision-key-v1",
        branch_order=branch_order or branches.execution_order,
    )


def rule(
    rule_id: str,
    *,
    mean: float = 0.30,
    variance: float = 0.0025,
    pairs: int = 10,
    order: int = 1,
    activated: bool = True,
) -> RuleContrast:
    return RuleContrast(
        rule_id=rule_id,
        order=order,
        condition={
            "task_family": "hotpotqa",
            "role_cluster": "verify",
            "stage": "before_output",
        },
        action={"model_id": "model-b"},
        delta_mean=mean,
        posterior_variance=variance,
        n_eff_pairs=pairs,
        interaction_activated=activated,
    )


def test_straddle_selects_one_peak_per_trajectory_and_ten_percent_batch() -> None:
    candidates = [
        site("t1", "s1", 0.5, 0.3),
        site("t1", "s2", 0.5, 0.8),
        site("t2", "s1", 0.5, 0.7),
        site("t3", "s1", 0.8, 5.0),  # warning, not a straddle candidate
        site("t4", "s1", 0.2, 5.0),  # no-hint boundary
    ]
    assert probe_budget(20) == 2
    selected = select_straddle_candidates(candidates, natural_trajectory_count=20)
    assert [(item.trajectory_id, item.step_id) for item in selected] == [
        ("t1", "s2"),
        ("t2", "s1"),
    ]


def test_seeded_warning_audit_is_reproducible_and_only_uses_warnings() -> None:
    warnings = [site(f"t{index:02d}", "s0", 0.9, 0.0) for index in range(100)]
    warnings.append(site("middle", "s0", 0.5, 1.0))
    first = select_seeded_audits(warnings, seed=37)
    second = select_seeded_audits(reversed(warnings), seed=37)
    assert first == second
    assert all(item.risk_probability >= 0.8 for item in first)
    assert 0 < len(first) < len(warnings)


def test_probe_record_requires_one_field_k3_binary_and_six_branch_ids() -> None:
    validated = validate_probe_record(probe_record(is_audit=True))
    assert validated.changed_field == "model_id"
    assert validated.is_audit is True
    assert len(validated.branch_ids) == 6

    with pytest.raises(ValueError, match="exactly one field"):
        validate_probe_record(
            probe_record(candidate_action=decision(model_id="model-b", edge_type="bidirectional"))
        )
    with pytest.raises(ValueError, match="K=3"):
        validate_probe_record(probe_record(incumbent_returns=(0, 1)))
    with pytest.raises(ValueError, match="binary"):
        validate_probe_record(probe_record(candidate_returns=(1, 1, 0.5)))
    with pytest.raises(ValueError, match="six unique"):
        validate_probe_record(probe_record(branch_order=("duplicate",) * 6))


@dataclass(frozen=True)
class Evaluation:
    valid: bool


@dataclass(frozen=True)
class BranchTrajectory:
    trajectory_id: str
    forced_probe: bool
    grpo_eligible: bool
    evaluation: Evaluation


def test_probe_and_audit_branches_are_excluded_from_grpo_and_standard_metrics() -> None:
    record = probe_record()
    branches = [
        BranchTrajectory(branch_id, True, False, Evaluation(True))
        for branch_id in record.branch_order
    ]
    natural = BranchTrajectory("natural-1", False, True, Evaluation(True))
    assert_probe_data_partition(record, branches)
    mixed = [natural, *branches]
    assert natural_trajectories(mixed) == (natural,)
    assert grpo_training_trajectories(mixed) == (natural,)
    assert standard_metric_trajectories(mixed) == (natural,)

    leaked = list(branches)
    leaked[0] = BranchTrajectory(leaked[0].trajectory_id, False, True, Evaluation(True))
    with pytest.raises(ValueError, match="marked natural"):
        assert_probe_data_partition(record, leaked)


def test_empirical_calibration_quantile_and_interval_follow_specification() -> None:
    quantile = empirical_calibration_quantile(
        observed_deltas=[0.0, 1.0, 2.0, 3.0],
        predicted_means=[0.0, 0.0, 0.0, 0.0],
        posterior_variances=[0.5, 0.5, 0.5, 0.5],
        noise_variances=[0.5, 0.5, 0.5, 0.5],
        alpha=0.25,
    )
    assert quantile == 3.0
    assert calibrated_interval(0.3, 0.01, quantile) == pytest.approx((0.0, 0.6))


def test_rule_enumeration_requires_ten_pairs_and_activated_interaction() -> None:
    candidates = enumerate_rule_candidates(
        [
            rule("eligible"),
            rule("too-few", pairs=9),
            rule("inactive-interaction", order=2, activated=False),
            rule("active-interaction", order=2, activated=True),
        ],
        calibration_quantile=1.0,
    )
    assert [item.rule_id for item in candidates] == ["active-interaction", "eligible"]
    assert candidates[1].lower == pytest.approx(0.25)
    assert candidates[1].upper == pytest.approx(0.35)


def test_bh_fdr_uses_step_up_threshold() -> None:
    selected = benjamini_hochberg(
        {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20},
        fdr=0.10,
    )
    assert selected == frozenset({"a", "b", "c"})


def test_skill_status_requires_confirmation_harm_and_bh_then_suspends_and_retires() -> None:
    candidate = enumerate_rule_candidates(
        [rule("positive")],
        calibration_quantile=1.0,
    )[0]
    confirmation = ConfirmationEvidence(
        calibrated_lower=0.20,
        problem_ids=tuple(f"heldout-{index}" for index in range(20)),
        harm_probability=0.01,
    )
    activated = decide_skill_status(
        candidate,
        confirmation,
        bh_selected=True,
    )
    assert activated.status is SkillStatus.ACTIVE
    assert activated.requires_publication_gate is True

    weak = enumerate_rule_candidates(
        [rule("weak", mean=0.04, variance=0.0)],
        calibration_quantile=1.0,
    )[0]
    suspended = decide_skill_status(
        weak,
        confirmation,
        previous=SkillStatus.ACTIVE,
        bh_selected=True,
    )
    assert suspended.status is SkillStatus.SUSPENDED
    retired = decide_skill_status(
        weak,
        confirmation,
        previous=SkillStatus.SUSPENDED,
        consecutive_suspended_epochs=3,
    )
    assert retired.status is SkillStatus.RETIRED


def test_no_value_rule_retires_and_unconfirmed_rule_stays_candidate() -> None:
    no_value = enumerate_rule_candidates(
        [rule("no-value", mean=0.0, variance=0.0001)],
        calibration_quantile=1.0,
    )[0]
    assert decide_skill_status(no_value, None).status is SkillStatus.RETIRED

    positive = enumerate_rule_candidates(
        [rule("positive")], calibration_quantile=1.0
    )[0]
    assert decide_skill_status(positive, None, bh_selected=True).status is SkillStatus.CANDIDATE
