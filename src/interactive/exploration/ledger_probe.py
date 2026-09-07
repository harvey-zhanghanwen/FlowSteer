"""Pure scheduling and data-plane guards for ledger paired probes.

This module implements Section 6 of ``LatentLoss_Implementation_Spec.md``.
It intentionally does not execute a fork or mutate an AgentGraph.  The caller
owns snapshot restoration, branch-local caches/tools, and frozen-Director
continuation.  These helpers only choose probe/audit sites, validate the
resulting :class:`~src.interactive.records.ProbeRecord`, and keep intervention
branches out of GRPO and ordinary task metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from ..records import ProbeRecord


DEFAULT_PROBE_FRACTION = 0.10
DEFAULT_AUDIT_PROBABILITY = 0.05
DEFAULT_BRANCH_REPEATS = 3

_CONTEXT_FIELDS = ("task_family", "stage")
_DECISION_FIELDS = (
    "role_cluster",
    "model_id",
    "edge_type",
    "same_model_as_upstream",
)


@dataclass(frozen=True)
class StraddleCandidate:
    """One post-execution step eligible for a same-snapshot intervention."""

    trajectory_id: str
    step_id: str
    snapshot_id: str
    risk_probability: float
    score: float
    key_keep: Mapping[str, Any]
    key_switch: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in ("trajectory_id", "step_id", "snapshot_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not math.isfinite(float(self.risk_probability)) or not (
            0.0 <= float(self.risk_probability) <= 1.0
        ):
            raise ValueError("risk_probability must be finite and in [0, 1]")
        if not math.isfinite(float(self.score)) or float(self.score) < 0.0:
            raise ValueError("straddle score must be finite and non-negative")
        differing_decision_field(self.key_keep, self.key_switch)


@dataclass(frozen=True)
class ProbeBranchIds:
    """The six branch trajectory IDs and their seeded execution order."""

    keep: tuple[str, str, str]
    switch: tuple[str, str, str]
    execution_order: tuple[str, str, str, str, str, str]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.keep + self.switch


@dataclass(frozen=True)
class ProbeValidation:
    """Validated properties needed by the ledger update path."""

    changed_field: str
    is_audit: bool
    branch_ids: tuple[str, ...]


def _validate_rate(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def differing_decision_field(
    key_keep: Mapping[str, Any],
    key_switch: Mapping[str, Any],
) -> str:
    """Return the sole changed DecisionKey field or reject the pair.

    ``task_family`` and ``stage`` identify the intervention context and must
    remain fixed.  The mutable fields are exactly those allowed by Section
    5.1, including ``same_model_as_upstream`` for an upstream-model switch.
    """

    if not isinstance(key_keep, Mapping) or not isinstance(key_switch, Mapping):
        raise TypeError("paired decisions must be mappings")
    required = set(_CONTEXT_FIELDS + _DECISION_FIELDS)
    missing_keep = required.difference(key_keep)
    missing_switch = required.difference(key_switch)
    if missing_keep or missing_switch:
        missing = sorted(missing_keep | missing_switch)
        raise ValueError(f"DecisionKey is missing required fields: {missing}")
    changed_context = [
        name for name in _CONTEXT_FIELDS if key_keep[name] != key_switch[name]
    ]
    if changed_context:
        raise ValueError(
            "paired probe cannot change context fields: " + ",".join(changed_context)
        )
    changed = [
        name for name in _DECISION_FIELDS if key_keep[name] != key_switch[name]
    ]
    if len(changed) != 1:
        raise ValueError(
            "paired probe decisions must differ in exactly one field; "
            f"found {len(changed)}"
        )
    return changed[0]


def probe_budget(natural_trajectory_count: int, fraction: float = DEFAULT_PROBE_FRACTION) -> int:
    """Return the Section 6.1 floor budget for a natural-trajectory batch."""

    if (
        isinstance(natural_trajectory_count, bool)
        or not isinstance(natural_trajectory_count, int)
        or natural_trajectory_count < 0
    ):
        raise ValueError("natural_trajectory_count must be a non-negative integer")
    fraction = _validate_rate(fraction, "probe fraction")
    return int(math.floor(natural_trajectory_count * fraction))


def select_straddle_candidates(
    candidates: Iterable[StraddleCandidate],
    *,
    natural_trajectory_count: int,
    tau: float = 0.8,
    fraction: float = DEFAULT_PROBE_FRACTION,
) -> tuple[StraddleCandidate, ...]:
    """Choose one highest-straddle step per trajectory, then the batch top-B.

    Only risks strictly inside ``(1-tau, tau)`` are eligible: values at the
    upper boundary are warnings and values at the lower boundary are the
    no-hint region from Section 5.5.
    """

    tau = _validate_rate(tau, "tau")
    if tau <= 0.5:
        raise ValueError("tau must exceed 0.5 to define a middle region")
    budget = probe_budget(natural_trajectory_count, fraction)
    if budget == 0:
        return ()

    lower = 1.0 - tau
    eligible = [
        item
        for item in candidates
        if lower < float(item.risk_probability) < tau
        and not math.isclose(float(item.risk_probability), lower, abs_tol=1e-12)
        and not math.isclose(float(item.risk_probability), tau, abs_tol=1e-12)
        and item.score > 0.0
    ]
    eligible.sort(
        key=lambda item: (
            item.trajectory_id,
            -float(item.score),
            item.step_id,
            item.snapshot_id,
        )
    )
    best_by_trajectory: dict[str, StraddleCandidate] = {}
    for item in eligible:
        best_by_trajectory.setdefault(item.trajectory_id, item)
    ranked = sorted(
        best_by_trajectory.values(),
        key=lambda item: (
            -float(item.score),
            item.trajectory_id,
            item.step_id,
            item.snapshot_id,
        ),
    )
    return tuple(ranked[:budget])


def select_seeded_audits(
    warning_candidates: Iterable[StraddleCandidate],
    *,
    seed: int,
    probability: float = DEFAULT_AUDIT_PROBABILITY,
    tau: float = 0.8,
) -> tuple[StraddleCandidate, ...]:
    """Independently sample high-confidence warning steps for 5% audits."""

    probability = _validate_rate(probability, "audit probability")
    tau = _validate_rate(tau, "tau")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    warnings = sorted(
        (
            item
            for item in warning_candidates
            if float(item.risk_probability) >= tau
        ),
        key=lambda item: (item.trajectory_id, item.step_id, item.snapshot_id),
    )
    rng = np.random.default_rng(seed)
    return tuple(item for item in warnings if float(rng.random()) < probability)


def make_probe_branch_ids(
    probe_id: str,
    *,
    seed: int,
    repeats: int = DEFAULT_BRANCH_REPEATS,
) -> ProbeBranchIds:
    """Create K=3 IDs for each arm and a seeded randomized run order."""

    if not isinstance(probe_id, str) or not probe_id.strip():
        raise ValueError("probe_id must be a non-empty string")
    if repeats != DEFAULT_BRANCH_REPEATS:
        raise ValueError("the frozen paired-probe protocol requires K=3 per branch")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    keep = tuple(f"{probe_id}:keep:{index}" for index in range(repeats))
    switch = tuple(f"{probe_id}:switch:{index}" for index in range(repeats))
    all_ids = keep + switch
    rng = np.random.default_rng(seed)
    order = tuple(all_ids[int(index)] for index in rng.permutation(len(all_ids)))
    return ProbeBranchIds(keep=keep, switch=switch, execution_order=order)


def validate_probe_record(
    record: ProbeRecord,
    *,
    repeats: int = DEFAULT_BRANCH_REPEATS,
) -> ProbeValidation:
    """Validate one-field intervention, binary K=3 outcomes, and branch IDs."""

    if not isinstance(record, ProbeRecord):
        raise TypeError("record must be an interactive.records.ProbeRecord")
    if repeats != DEFAULT_BRANCH_REPEATS:
        raise ValueError("the frozen paired-probe protocol requires K=3 per branch")
    changed_field = differing_decision_field(
        record.incumbent_action,
        record.candidate_action,
    )
    if len(record.incumbent_returns) != repeats or len(record.candidate_returns) != repeats:
        raise ValueError("paired probe must contain exactly K=3 returns per branch")
    returns = tuple(record.incumbent_returns) + tuple(record.candidate_returns)
    if any(float(value) not in (0.0, 1.0) for value in returns):
        raise ValueError("paired probe terminal returns must be binary")

    expected = {
        *(f"{record.probe_id}:keep:{index}" for index in range(repeats)),
        *(f"{record.probe_id}:switch:{index}" for index in range(repeats)),
    }
    branch_ids = tuple(str(value) for value in record.branch_order)
    if len(branch_ids) != 2 * repeats or len(set(branch_ids)) != 2 * repeats:
        raise ValueError("branch_order must contain six unique branch trajectory IDs")
    if set(branch_ids) != expected:
        raise ValueError("branch_order IDs do not match the probe keep/switch arms")
    is_audit = record.state_features.get("is_audit", False)
    if type(is_audit) is not bool:
        raise ValueError("state_features.is_audit must be bool when supplied")
    return ProbeValidation(changed_field, is_audit, branch_ids)


Trajectory = TypeVar("Trajectory")


def trajectory_is_natural(record: object) -> bool:
    """Read either the new ``is_natural`` flag or the legacy forced-probe flag."""

    if hasattr(record, "is_natural"):
        value = getattr(record, "is_natural")
        if type(value) is not bool:
            raise ValueError("trajectory is_natural must be bool")
        if hasattr(record, "forced_probe") and bool(getattr(record, "forced_probe")) == value:
            raise ValueError("is_natural and forced_probe flags are inconsistent")
        return value
    if not hasattr(record, "forced_probe"):
        raise ValueError("trajectory must expose is_natural or forced_probe")
    value = getattr(record, "forced_probe")
    if type(value) is not bool:
        raise ValueError("trajectory forced_probe must be bool")
    return not value


def natural_trajectories(records: Iterable[Trajectory]) -> tuple[Trajectory, ...]:
    """The only records eligible for ordinary success-rate reporting."""

    return tuple(record for record in records if trajectory_is_natural(record))


def grpo_training_trajectories(records: Iterable[Trajectory]) -> tuple[Trajectory, ...]:
    """Select valid natural trajectories; probe/audit branches cannot enter GRPO."""

    selected: list[Trajectory] = []
    for record in records:
        if trajectory_is_natural(record) and bool(getattr(record, "grpo_eligible", False)):
            selected.append(record)
    return tuple(selected)


def standard_metric_trajectories(records: Iterable[Trajectory]) -> tuple[Trajectory, ...]:
    """Select evaluator-valid natural trajectories for standard task metrics."""

    selected: list[Trajectory] = []
    for record in records:
        if not trajectory_is_natural(record):
            continue
        evaluation = getattr(record, "evaluation", None)
        evaluator_valid = getattr(record, "evaluator_valid", None)
        valid = (
            bool(getattr(evaluation, "valid", False))
            if evaluation is not None
            else bool(evaluator_valid)
        )
        if valid:
            selected.append(record)
    return tuple(selected)


def assert_probe_data_partition(
    record: ProbeRecord,
    branch_trajectories: Sequence[object],
) -> None:
    """Assert that all six recorded branches are intervention-only trajectories."""

    validation = validate_probe_record(record)
    by_id = {str(getattr(item, "trajectory_id", "")): item for item in branch_trajectories}
    if set(by_id) != set(validation.branch_ids) or len(by_id) != len(branch_trajectories):
        raise ValueError("branch trajectories do not exactly match ProbeRecord branch IDs")
    for branch_id in validation.branch_ids:
        branch = by_id[branch_id]
        if trajectory_is_natural(branch):
            raise ValueError("probe/audit branch is incorrectly marked natural")
        if bool(getattr(branch, "grpo_eligible", False)):
            raise ValueError("probe/audit branch is incorrectly GRPO-eligible")
    if natural_trajectories(branch_trajectories):
        raise AssertionError("probe/audit branches leaked into standard task metrics")
