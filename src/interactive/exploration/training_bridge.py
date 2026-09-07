"""Strict record bridge from AgentGraph rollouts to the dynamic ledger plane.

This is a schema adapter, not a second trainer.  The existing natural
``TrajectoryRecord`` remains the only object consumed by Action-Masked
One-Pass GRPO.  These helpers copy its evaluator-valid binary exact-match
outcome and the hook's decision sidecar into ``LedgerTrajectoryRecord``;
forced probe branches stay explicitly ineligible for GRPO and ordinary
metrics.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

from ..records import ProbeRecord, TrajectoryRecord
from .combination_ledger import DecisionKey
from .ledger_epoch import FrozenEpochCondition, LedgerTrajectoryRecord, StepRecord


LEDGER_FEATURE_SCHEMA_VERSION = "dynamic-combination-decision-key-v1"


def binary_exact_match(record: TrajectoryRecord) -> int:
    """Return the ledger's binary ``R_T`` without changing GRPO reward."""

    if not isinstance(record, TrajectoryRecord):
        raise TypeError("record must be TrajectoryRecord")
    if not record.evaluation.valid:
        raise ValueError("an evaluator-invalid trajectory has no ledger outcome")
    value = record.evaluation.metrics.get("exact_match")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ledger outcome requires the official exact_match metric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError("ledger exact_match outcome must be binary")
    return int(numeric)


def _bind_steps(
    steps: Sequence[StepRecord],
    trajectory_id: str,
) -> tuple[StepRecord, ...]:
    return tuple(
        step
        if step.trajectory_id == trajectory_id
        else replace(step, trajectory_id=trajectory_id)
        for step in steps
    )


def natural_ledger_trajectory(
    record: TrajectoryRecord,
    *,
    condition: FrozenEpochCondition,
    steps: Sequence[StepRecord],
) -> LedgerTrajectoryRecord:
    """Bind one natural rollout to its frozen ledger condition."""

    if record.forced_probe:
        raise ValueError("a forced probe cannot be converted as natural evidence")
    if record.condition_id != condition.condition_id:
        raise ValueError("natural rollout condition differs from the ledger epoch")
    if record.versions.policy != condition.versions.policy:
        raise ValueError("natural rollout policy differs from the ledger epoch")
    return LedgerTrajectoryRecord(
        trajectory_id=record.trajectory_id,
        problem_id=record.task.task_id,
        task_split=record.task.split,
        group_id=record.group_id,
        condition_id=record.condition_id,
        policy_version=record.versions.policy,
        steps=_bind_steps(steps, record.trajectory_id),
        is_natural=True,
        terminal_reward=binary_exact_match(record),
        evaluator_valid=record.evaluation.valid,
        grpo_eligible=record.grpo_eligible,
    )


def intervention_step(
    *,
    trajectory_id: str,
    step_id: str,
    snapshot_id: str,
    key: DecisionKey,
    alternative: DecisionKey,
    translation_receipt: Mapping[str, Any],
    audit: bool,
) -> StepRecord:
    """Represent the externally assigned, no-policy-credit probe edit."""

    return StepRecord(
        step_id=step_id,
        trajectory_id=trajectory_id,
        snapshot_id=snapshot_id,
        key=key,
        candidates=(alternative,),
        ledger_readout={
            "forced_intervention": True,
            "translation_receipt": dict(translation_receipt),
        },
        director_action="not_triggered",
        surface=None,
        forked=True,
        audit=audit,
    )


def probe_ledger_trajectory(
    record: TrajectoryRecord,
    *,
    condition: FrozenEpochCondition,
    probe_id: str,
    audit: bool,
    forced_step: StepRecord,
    downstream_steps: Sequence[StepRecord] = (),
) -> LedgerTrajectoryRecord:
    """Bind a forced branch while proving it cannot enter GRPO."""

    if not record.forced_probe or record.grpo_eligible:
        raise ValueError("probe branch must be forced and GRPO-ineligible")
    if record.trajectory_id != forced_step.trajectory_id:
        raise ValueError("forced intervention step belongs to another trajectory")
    if record.condition_id != condition.condition_id:
        raise ValueError("probe branch condition differs from the ledger epoch")
    if record.versions.policy != condition.versions.policy:
        raise ValueError("probe branch policy differs from the ledger epoch")
    return LedgerTrajectoryRecord(
        trajectory_id=record.trajectory_id,
        problem_id=record.task.task_id,
        task_split=record.task.split,
        group_id=record.group_id,
        condition_id=record.condition_id,
        policy_version=record.versions.policy,
        steps=(forced_step, *_bind_steps(downstream_steps, record.trajectory_id)),
        is_natural=False,
        terminal_reward=binary_exact_match(record),
        evaluator_valid=record.evaluation.valid,
        grpo_eligible=False,
        probe_id=probe_id,
        audit=audit,
    )


def build_probe_record(
    *,
    probe_id: str,
    problem_id: str,
    task_split: str,
    snapshot_id: str,
    condition: FrozenEpochCondition,
    key_keep: DecisionKey,
    key_switch: DecisionKey,
    branch_order: Sequence[str],
    branches: Mapping[str, TrajectoryRecord],
    sampling_probability: float,
    is_audit: bool,
) -> ProbeRecord:
    """Create the paired K=3 receipt from completed branch trajectories."""

    expected = {
        *(f"{probe_id}:keep:{index}" for index in range(3)),
        *(f"{probe_id}:switch:{index}" for index in range(3)),
    }
    if set(branches) != expected or set(branch_order) != expected:
        raise ValueError("probe branch set differs from the frozen K=3 plan")

    def arm_returns(arm: str) -> tuple[int, int, int]:
        return tuple(
            binary_exact_match(branches[f"{probe_id}:{arm}:{index}"])
            for index in range(3)
        )  # type: ignore[return-value]

    executor_versions: dict[str, str] = {}
    for branch in branches.values():
        for turn in branch.turns:
            for execution in turn.executions:
                executor_versions[execution.model_id] = condition.versions.model_catalog
    return ProbeRecord(
        probe_id=probe_id,
        problem_id=problem_id,
        task_split=task_split,
        snapshot_id=snapshot_id,
        policy_version=condition.versions.policy,
        state_features={"is_audit": is_audit},
        incumbent_action=key_keep.to_dict(),
        candidate_action=key_switch.to_dict(),
        sampling_probability=float(sampling_probability),
        incumbent_returns=arm_returns("keep"),
        candidate_returns=arm_returns("switch"),
        executor_versions=executor_versions,
        evaluator_version=condition.versions.evaluator,
        feature_schema_version=condition.versions.feature_schema,
        branch_order=tuple(branch_order),
    )


__all__ = [
    "LEDGER_FEATURE_SCHEMA_VERSION",
    "binary_exact_match",
    "build_probe_record",
    "intervention_step",
    "natural_ledger_trajectory",
    "probe_ledger_trajectory",
]
