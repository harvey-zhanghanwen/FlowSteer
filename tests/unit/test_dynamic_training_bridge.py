from __future__ import annotations

from dataclasses import replace

import pytest

from src.interactive.exploration.combination_ledger import DecisionKey
from src.interactive.exploration.ledger_epoch import FrozenEpochCondition, StepRecord
from src.interactive.exploration.training_bridge import (
    binary_exact_match,
    build_probe_record,
    intervention_step,
    natural_ledger_trajectory,
    probe_ledger_trajectory,
)
from src.interactive.records import EvaluationReceipt, TaskRecord, TrajectoryRecord
from src.interactive.versioning import VersionBundle


def _condition() -> FrozenEpochCondition:
    versions = VersionBundle(
        policy="policy-0",
        model_catalog="catalog-0",
        evaluator="eval-0",
        prompt="prompt-0",
        tool="tool-0",
        feature_schema="dynamic-combination-decision-key-v1",
        posterior="ledger-0",
        skill_library="skills-0",
    )
    return FrozenEpochCondition(0, "condition-0", versions, "ledger-0", "skills-0")


def _record(identifier: str, *, forced: bool = False, em: float = 1.0):
    condition = _condition()
    return TrajectoryRecord(
        trajectory_id=identifier,
        task=TaskRecord("hotpot-1", "question", ["answer"], "train"),
        group_id="group-0",
        condition_id=condition.condition_id,
        rollout_id=identifier,
        versions=condition.versions,
        turns=(),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            "eval-0", True, em, {"exact_match": em, "token_f1": em}
        ),
        termination_reason="finish",
        explicit_finish=True,
        forced_probe=forced,
        intervention=({"branch": identifier} if forced else {}),
    )


def _key(model: str) -> DecisionKey:
    return DecisionKey("hotpotqa", "solve", model, "independent", False, "other")


def test_binary_exact_match_is_strict_and_does_not_read_f1() -> None:
    assert binary_exact_match(_record("natural")) == 1
    with pytest.raises(ValueError, match="binary"):
        binary_exact_match(_record("fractional", em=0.5))


def test_natural_and_probe_planes_remain_disjoint() -> None:
    condition = _condition()
    natural = _record("natural")
    step = StepRecord(
        "s0", "natural", "snapshot", _key("a"), (_key("b"),), {},
        "not_triggered", None,
    )
    ledger = natural_ledger_trajectory(natural, condition=condition, steps=(step,))
    assert ledger.is_natural and not ledger.forced_probe

    forced = intervention_step(
        trajectory_id="probe:keep:0",
        step_id="s0:forced",
        snapshot_id="snapshot",
        key=_key("a"),
        alternative=_key("b"),
        translation_receipt={"changed_field": "model_id"},
        audit=False,
    )
    branch = _record("probe:keep:0", forced=True)
    branch_ledger = probe_ledger_trajectory(
        branch,
        condition=condition,
        probe_id="probe",
        audit=False,
        forced_step=forced,
    )
    assert branch_ledger.forced_probe and not branch_ledger.grpo_eligible


def test_probe_record_binds_six_binary_branch_receipts() -> None:
    branches = {}
    order = []
    for arm in ("keep", "switch"):
        for index in range(3):
            identifier = f"probe-1:{arm}:{index}"
            branches[identifier] = _record(
                identifier,
                forced=True,
                em=float(arm == "switch"),
            )
            order.append(identifier)
    receipt = build_probe_record(
        probe_id="probe-1",
        problem_id="hotpot-1",
        task_split="train",
        snapshot_id="snapshot",
        condition=_condition(),
        key_keep=_key("a"),
        key_switch=_key("b"),
        branch_order=tuple(reversed(order)),
        branches=branches,
        sampling_probability=0.1,
        is_audit=False,
    )
    assert tuple(receipt.incumbent_returns) == (0, 0, 0)
    assert tuple(receipt.candidate_returns) == (1, 1, 1)
    assert receipt.paired_effect == 1.0
