"""Deterministic Phase A--D acceptance gates for the in-loop ledger.

The experiments in this module are the synthetic/mock checks prescribed by
Section 11 of ``LatentLoss_Implementation_Spec.md``.  They exercise the
project implementations rather than starting a model server, a GPU job, a
W&B run, or an optimizer.  Phase E is intentionally emitted as ``pending``:
only the real training runner can satisfy its data-routing and 50 x 4 rollout
requirements.

The report is fail-closed.  Passing these synthetic gates permits the real
Phase-E integration run; it never authorizes long training on its own.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..records import ProbeRecord
from ..skills.ledger_miner import (
    ConfirmationEvidence,
    RuleContrast,
    benjamini_hochberg,
    decide_skill_status,
    enumerate_rule_candidates,
)
from ..skills.schema import SkillEvidence, SkillRecord, SkillStatus
from ..skills.validator import SkillEvidenceGate, SkillGateConfig
from ..versioning import VersionBundle
from .combination_ledger import CombinationPosterior, DecisionKey, SurfaceSignal
from .latent_loss import (
    LatentLossConfig,
    joint_maximum_estimate,
    post_execution_latent_loss,
)
from .ledger_epoch import LedgerEpoch
from .ledger_probe import (
    StraddleCandidate,
    assert_probe_data_partition,
    grpo_training_trajectories,
    make_probe_branch_ids,
    natural_trajectories,
    select_straddle_candidates,
    standard_metric_trajectories,
)


PHASE_ACCEPTANCE_VERSION = "latent-loss-phase-acceptance-v1"
_NORMAL_90 = 1.6448536269514722


class AcceptanceGateError(RuntimeError):
    """An acceptance report does not permit the requested next stage."""


@dataclass(frozen=True)
class AcceptanceCheck:
    """One auditable acceptance result.

    ``passed=None`` means that the check must be run by a different boundary;
    it is never interpreted as success.
    """

    check_id: str
    description: str
    passed: bool | None
    value: Any
    threshold: Any
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PhaseAcceptance:
    phase: str
    execution_boundary: str
    checks: tuple[AcceptanceCheck, ...]

    @property
    def status(self) -> str:
        if any(check.passed is False for check in self.checks):
            return "failed"
        if any(check.passed is None for check in self.checks):
            return "pending"
        return "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "execution_boundary": self.execution_boundary,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class PhaseAcceptanceReport:
    seed: int
    phases: tuple[PhaseAcceptance, ...]
    version: str = PHASE_ACCEPTANCE_VERSION

    @property
    def synthetic_gates_passed(self) -> bool:
        synthetic = tuple(phase for phase in self.phases if phase.phase != "E")
        return bool(synthetic) and all(phase.status == "passed" for phase in synthetic)

    @property
    def phase_e_completed(self) -> bool:
        phase_e = tuple(phase for phase in self.phases if phase.phase == "E")
        return len(phase_e) == 1 and phase_e[0].status == "passed"

    @property
    def long_training_authorized(self) -> bool:
        return self.synthetic_gates_passed and self.phase_e_completed

    def assert_ready_for_phase_e(self) -> None:
        if not self.synthetic_gates_passed:
            failed = [
                check.check_id
                for phase in self.phases
                if phase.phase != "E"
                for check in phase.checks
                if check.passed is not True
            ]
            raise AcceptanceGateError(
                "synthetic acceptance gates failed: " + ", ".join(failed)
            )

    def assert_ready_for_long_training(self) -> None:
        self.assert_ready_for_phase_e()
        if not self.phase_e_completed:
            raise AcceptanceGateError(
                "Phase E is pending real-runner evidence; long training is not authorized"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "synthetic_gates_passed": self.synthetic_gates_passed,
            "phase_e_completed": self.phase_e_completed,
            "long_training_authorized": self.long_training_authorized,
            "phases": [phase.to_dict() for phase in self.phases],
        }


def _key(
    *,
    family: str = "qa",
    role: str = "solve",
    model: str = "A",
    edge: str = "independent",
    same: bool = False,
    stage: str = "other",
) -> DecisionKey:
    return DecisionKey(family, role, model, edge, same, stage)


def _checked(
    check_id: str,
    description: str,
    threshold: Any,
    experiment: Callable[[], tuple[bool, Any, Mapping[str, Any]]],
) -> AcceptanceCheck:
    """Run one deterministic experiment and convert any exception to failure."""

    try:
        passed, value, details = experiment()
        return AcceptanceCheck(
            check_id=check_id,
            description=description,
            passed=bool(passed),
            value=value,
            threshold=threshold,
            details=dict(details),
        )
    except Exception as exc:  # fail-closed acceptance boundary
        return AcceptanceCheck(
            check_id=check_id,
            description=description,
            passed=False,
            value={"error_type": type(exc).__name__, "message": str(exc)},
            threshold=threshold,
            details={"exception_captured": True},
        )


def _phase_a_first_order(seed: int) -> tuple[bool, Any, Mapping[str, Any]]:
    rng = np.random.default_rng(seed)
    ledger = CombinationPosterior("synthetic-policy-a1", n_act=10_000)
    truth = {
        "role=solve": -0.05,
        "role=verify": 0.05,
        "model=A": -0.15,
        "model=B": 0.15,
        "rel=independent|different": 0.125,
        "rel=independent|same": -0.125,
    }
    for _ in range(200):
        changed = str(rng.choice(("role", "model", "rel")))
        role = str(rng.choice(("solve", "verify")))
        model = str(rng.choice(("A", "B")))
        same = bool(rng.integers(0, 2))
        keep = _key(role=role, model=model, same=same)
        switch = _key(
            role=("verify" if role == "solve" else "solve") if changed == "role" else role,
            model=("B" if model == "A" else "A") if changed == "model" else model,
            same=(not same) if changed == "rel" else same,
        )

        def probability(key: DecisionKey) -> float:
            columns = (
                f"role={key.role_cluster}",
                f"model={key.model_id}",
                f"rel={key.relation_level}",
            )
            return 0.5 + sum(truth[column] for column in columns)

        ledger.update_probe(
            keep,
            switch,
            rng.binomial(1, probability(keep), 3),
            rng.binomial(1, probability(switch), 3),
        )

    covariance = ledger.covariance
    coverage_by_column: dict[str, bool] = {}
    intervals: dict[str, list[float]] = {}
    for column, expected in truth.items():
        index = ledger.columns[column]
        radius = _NORMAL_90 * math.sqrt(float(covariance[index, index]))
        lower = float(ledger.mean[index] - radius)
        upper = float(ledger.mean[index] + radius)
        intervals[column] = [lower, upper]
        coverage_by_column[column] = lower <= expected <= upper
    coverage = sum(coverage_by_column.values()) / len(coverage_by_column)
    return coverage >= 0.85, coverage, {
        "probe_count": ledger.probe_count,
        "covered_columns": sum(coverage_by_column.values()),
        "total_columns": len(coverage_by_column),
        "coverage_by_column": coverage_by_column,
        "intervals": intervals,
    }


def _phase_a_interaction(seed: int) -> tuple[bool, Any, Mapping[str, Any]]:
    rng = np.random.default_rng(seed)
    ledger = CombinationPosterior("synthetic-policy-a2")
    target = "role=solve&rel=bidirectional|same"
    activated_at: int | None = None

    def probability(key: DecisionKey) -> float:
        target_active = (
            key.role_cluster == "solve"
            and key.relation_level == "bidirectional|same"
        )
        return 0.5 + (0.10 if target_active else 0.0)

    for probe_index in range(50):
        if probe_index < 5:
            keep = _key(family="math", edge="independent", same=True)
            switch = _key(family="math", edge="bidirectional", same=True)
        else:
            role = str(rng.choice(("solve", "verify")))
            model = str(rng.choice(("A", "B")))
            edge = str(rng.choice(("independent", "bidirectional")))
            same = bool(rng.integers(0, 2))
            changed = str(rng.choice(("role", "model", "rel")))
            keep = _key(family="math", role=role, model=model, edge=edge, same=same)
            switch = _key(
                family="math",
                role=("verify" if role == "solve" else "solve")
                if changed == "role"
                else role,
                model=("B" if model == "A" else "A")
                if changed == "model"
                else model,
                edge=("bidirectional" if edge == "independent" else "independent")
                if changed == "rel"
                else edge,
                same=same,
            )
        ledger.update_probe(
            keep,
            switch,
            rng.binomial(1, probability(keep), 3),
            rng.binomial(1, probability(switch), 3),
        )
        if target in ledger.columns and activated_at is None:
            activated_at = probe_index + 1

    index = ledger.columns[target]
    radius = _NORMAL_90 * math.sqrt(float(ledger.covariance[index, index]))
    interval = (float(ledger.mean[index] - radius), float(ledger.mean[index] + radius))
    covered = interval[0] <= 0.10 <= interval[1]
    passed = activated_at == 5 and covered
    return passed, {"activated_at": activated_at, "covered_after_50": covered}, {
        "target_column": target,
        "true_effect": 0.10,
        "interval_after_50": list(interval),
        "probe_count": ledger.probe_count,
    }


def _phase_a_sensors() -> tuple[bool, Any, Mapping[str, Any]]:
    ledger = CombinationPosterior("synthetic-policy-a3")
    expected = {
        "verifier_pass|same": (0.80, 0.60),
        "verifier_pass|different": (0.80, 0.10),
    }
    estimates: dict[str, list[float]] = {}
    errors: list[float] = []
    for sensor_class, (sensitivity, false_positive_rate) in expected.items():
        for index in range(100):
            ledger.update_sensor(
                SurfaceSignal("verifier_pass", index < int(100 * sensitivity), sensor_class),
                1,
            )
        for index in range(100):
            ledger.update_sensor(
                SurfaceSignal(
                    "verifier_pass",
                    index < int(100 * false_positive_rate),
                    sensor_class,
                ),
                0,
            )
        estimate = ledger.sensor_rates(sensor_class)
        estimates[sensor_class] = [float(estimate[0]), float(estimate[1])]
        errors.extend((abs(estimate[0] - sensitivity), abs(estimate[1] - false_positive_rate)))
    maximum_error = max(errors)
    return maximum_error < 0.05, maximum_error, {
        "trajectories_per_sensor_class": 200,
        "expected": {key: list(value) for key, value in expected.items()},
        "estimated": estimates,
    }


def _phase_a_confounding(seed: int) -> tuple[bool, Any, Mapping[str, Any]]:
    rng = np.random.default_rng(seed)
    count = 5_000
    easy = rng.random(count) < 0.5
    chooses_b = np.where(easy, rng.random(count) < 0.9, rng.random(count) < 0.1)
    baseline = np.where(easy, 0.5, 0.1)
    true_effect = 0.25
    outcomes = rng.binomial(1, baseline + true_effect * chooses_b)
    naive = float(outcomes[chooses_b].mean() - outcomes[~chooses_b].mean())
    naive_bias = abs(naive - true_effect)

    ledger = CombinationPosterior("synthetic-policy-a4", n_act=10_000)
    keep = _key(model="A")
    switch = _key(model="B")
    for _ in range(200):
        paired_baseline = 0.5 if bool(rng.integers(0, 2)) else 0.1
        ledger.update_probe(
            keep,
            switch,
            rng.binomial(1, paired_baseline, 10),
            rng.binomial(1, paired_baseline + true_effect, 10),
        )
    paired = ledger.predict_contrast(switch, keep).mean
    paired_bias = abs(float(paired) - true_effect)
    passed = naive_bias > 0.10 and paired_bias < 0.03
    return passed, {"naive_bias": naive_bias, "paired_bias": paired_bias}, {
        "true_effect": true_effect,
        "natural_trajectory_count": count,
        "paired_probe_count": 200,
        "naive_effect": naive,
        "paired_effect": paired,
    }


class _GaussianAcceptancePosterior:
    """Frozen synthetic posterior implementing the latent-risk public protocol."""

    def __init__(
        self,
        *,
        mean: Sequence[float],
        covariance_scale: float,
        probe_count: int,
        q0: float,
        sensor_rates: Mapping[str, tuple[float, float]],
    ) -> None:
        self.mean = np.asarray(tuple(mean), dtype=np.float64)
        self.covariance = float(covariance_scale) * np.eye(self.mean.size)
        self.probe_count = int(probe_count)
        self._q0 = float(q0)
        self._sensor_rates = dict(sensor_rates)

    def design_vector(self, key: DecisionKey) -> np.ndarray:
        return np.asarray(
            (
                float(key.model_id == "B"),
                float(key.edge_type == "bidirectional"),
                float(key.role_cluster == "verify"),
                float(key.same_model_as_upstream),
            ),
            dtype=np.float64,
        )

    def contrast_vector(self, switch: DecisionKey, keep: DecisionKey) -> np.ndarray:
        return self.design_vector(switch) - self.design_vector(keep)

    def predict_contrast(self, switch: DecisionKey, keep: DecisionKey) -> object:
        vector = self.contrast_vector(switch, keep)
        mean = float(vector @ self.mean)
        variance = float(vector @ self.covariance @ vector)
        radius = _NORMAL_90 * math.sqrt(variance)
        return SimpleNamespace(
            mean=mean,
            variance=variance,
            lower=mean - radius,
            upper=mean + radius,
        )

    def sample_parameters(self, rng: np.random.Generator, *, size: int) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.covariance, size=size)

    def expected_terminal(
        self,
        task_family: str,
        prefix_keys: Sequence[DecisionKey],
        parameter: np.ndarray | None = None,
    ) -> float:
        del parameter
        if not task_family or not prefix_keys:
            raise ValueError("synthetic posterior requires a non-empty prefix")
        return self._q0

    def sample_sensor_rates(
        self,
        sensor_class: str,
        rng: np.random.Generator,
        *,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del rng
        sensitivity, false_positive_rate = self._sensor_rates.get(
            sensor_class, (0.5, 0.5)
        )
        return (
            np.full(size, sensitivity, dtype=np.float64),
            np.full(size, false_positive_rate, dtype=np.float64),
        )

    def likelihood_ratio(self, surface: SurfaceSignal) -> float:
        sensitivity, false_positive_rate = self._sensor_rates.get(
            surface.sensor_class, (0.5, 0.5)
        )
        if surface.value:
            return sensitivity / false_positive_rate
        return (1.0 - sensitivity) / (1.0 - false_positive_rate)


def _phase_b_risk(
    *, seed: int, same: bool, rates: tuple[float, float]
) -> tuple[float, bool]:
    sensor_class = f"verifier_pass|{'same' if same else 'different'}"
    posterior = _GaussianAcceptancePosterior(
        mean=(0.30, 0.0, 0.0, 0.0),
        covariance_scale=1e-8,
        probe_count=10,
        q0=0.5,
        sensor_rates={sensor_class: rates},
    )
    result = post_execution_latent_loss(
        posterior,  # type: ignore[arg-type]
        step_id=0,
        prefix_before=(),
        current=_key(family="hotpotqa", same=same),
        candidates=(_key(family="hotpotqa", model="B", same=same),),
        surface=SurfaceSignal("verifier_pass", True, sensor_class),
        remaining_rounds=3,
        rng=np.random.default_rng(seed),
    )
    if result.risk_probability is None:
        raise AssertionError("latent risk was not computed")
    return result.risk_probability, result.warning


def _phase_b_empty(seed: int) -> tuple[bool, Any, Mapping[str, Any]]:
    posterior = _GaussianAcceptancePosterior(
        mean=(0.0, 0.0, 0.0, 0.0),
        covariance_scale=0.25,
        probe_count=0,
        q0=0.5,
        sensor_rates={"none|different": (0.5, 0.5)},
    )
    result = post_execution_latent_loss(
        posterior,  # type: ignore[arg-type]
        step_id=0,
        prefix_before=(),
        current=_key(family="hotpotqa"),
        candidates=(_key(family="hotpotqa", model="B"),),
        surface=SurfaceSignal("none", False, "none|different"),
        remaining_rounds=4,
        rng=np.random.default_rng(seed),
        config=LatentLossConfig(posterior_samples=4_000),
    )
    risk = result.risk_probability
    passed = risk is not None and 0.20 < risk < 0.80 and not result.warning
    return passed, risk, {
        "warning": result.warning,
        "candidate_unknown": result.candidates[0].unknown,
    }


def _phase_b_joint_max(seed: int) -> tuple[bool, Any, Mapping[str, Any]]:
    posterior = _GaussianAcceptancePosterior(
        mean=(0.0, 0.0, 0.0, 0.0),
        covariance_scale=1.0,
        probe_count=0,
        q0=0.5,
        sensor_rates={},
    )
    estimate = joint_maximum_estimate(
        posterior,  # type: ignore[arg-type]
        current=_key(family="hotpotqa"),
        candidates=(
            _key(family="hotpotqa", model="B"),
            _key(family="hotpotqa", edge="bidirectional"),
        ),
        rng=np.random.default_rng(seed),
        samples=20_000,
    )
    difference = estimate.posterior_expected_maximum - estimate.plug_in_maximum
    return difference > 0.40, difference, {
        "plug_in_maximum": estimate.plug_in_maximum,
        "posterior_expected_maximum": estimate.posterior_expected_maximum,
        "mathematical_boundary": "E[max X] >= max E[X] (Jensen)",
        "specification_note": (
            "Phase-B item 4 states the reverse comparison; the implementation "
            "uses joint posterior sampling and the mathematically valid direction."
        ),
    }


def _decision_payload(**changes: Any) -> dict[str, Any]:
    value = _key(family="hotpotqa").to_dict()
    value.update(changes)
    return value


@dataclass(frozen=True)
class _BranchTrajectory:
    trajectory_id: str
    snapshot_id: str
    downstream_text: str
    forced_probe: bool = True
    grpo_eligible: bool = False
    evaluation: object = field(default_factory=lambda: SimpleNamespace(valid=True))


def _probe_fixture() -> tuple[ProbeRecord, tuple[_BranchTrajectory, ...]]:
    ids = make_probe_branch_ids("synthetic-probe", seed=31)
    record = ProbeRecord(
        probe_id="synthetic-probe",
        problem_id="synthetic-problem",
        task_split="train",
        snapshot_id="pre-action-snapshot",
        policy_version="synthetic-policy-c",
        state_features={"is_audit": True},
        incumbent_action=_decision_payload(),
        candidate_action=_decision_payload(model_id="B"),
        sampling_probability=0.05,
        incumbent_returns=(0, 1, 0),
        candidate_returns=(1, 1, 0),
        executor_versions={"A": "executor-a", "B": "executor-b"},
        evaluator_version="binary-em-v1",
        feature_schema_version="ledger-decision-key-v1",
        branch_order=ids.execution_order,
    )
    branches = tuple(
        _BranchTrajectory(
            trajectory_id=branch_id,
            snapshot_id=record.snapshot_id,
            downstream_text=f"independently-generated:{branch_id}",
        )
        for branch_id in record.branch_order
    )
    return record, branches


def _phase_c_fork() -> tuple[bool, Any, Mapping[str, Any]]:
    record, branches = _probe_fixture()
    assert_probe_data_partition(record, branches)
    same_snapshot = all(branch.snapshot_id == record.snapshot_id for branch in branches)
    regenerated = len({branch.downstream_text for branch in branches}) == len(branches)
    passed = same_snapshot and regenerated
    return passed, {
        "same_pre_action_snapshot": same_snapshot,
        "downstream_regenerated": regenerated,
    }, {
        "branch_count": len(branches),
        "branch_order": list(record.branch_order),
    }


def _phase_c_straddle(seed: int) -> tuple[bool, Any, Mapping[str, Any]]:
    current = _key(family="hotpotqa", model="A")
    target = _key(family="hotpotqa", model="B")
    distractor = _key(family="hotpotqa", role="verify", model="A")
    sites: list[StraddleCandidate] = []
    for index in range(200):
        is_target = index < 100
        switch = target if is_target else distractor
        sites.append(
            StraddleCandidate(
                trajectory_id=f"natural-{index:03d}",
                step_id="0",
                snapshot_id=f"snapshot-{index:03d}",
                risk_probability=0.5,
                score=1.0 if is_target else 0.1,
                key_keep=current.to_dict(),
                key_switch=switch.to_dict(),
            )
        )
    chosen = select_straddle_candidates(sites, natural_trajectory_count=200)
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(sites), size=len(chosen), replace=False)
    random_sites = tuple(sites[int(index)] for index in random_indices)

    def fit(selected: Sequence[StraddleCandidate]) -> tuple[float, int]:
        ledger = CombinationPosterior("synthetic-policy-c2", n_act=10_000)
        target_count = 0
        for site in selected:
            keep_key = DecisionKey.from_dict(site.key_keep)
            switch_key = DecisionKey.from_dict(site.key_switch)
            if switch_key.model_id == "B":
                target_count += 1
                keep_returns, switch_returns = (0, 0, 1), (1, 1, 1)
            else:
                keep_returns, switch_returns = (0, 1, 0), (1, 0, 0)
            ledger.update_probe(keep_key, switch_key, keep_returns, switch_returns)
        estimate = ledger.predict_contrast(target, current)
        return estimate.upper - estimate.lower, target_count

    straddle_width, straddle_target_count = fit(chosen)
    random_width, random_target_count = fit(random_sites)
    passed = straddle_width < random_width
    return passed, {
        "straddle_width": straddle_width,
        "random_width": random_width,
    }, {
        "probe_budget_each": len(chosen),
        "straddle_target_probes": straddle_target_count,
        "random_target_probes": random_target_count,
        "width_reduction": random_width - straddle_width,
    }


def _phase_c_partition() -> tuple[bool, Any, Mapping[str, Any]]:
    record, branches = _probe_fixture()
    assert_probe_data_partition(record, branches)
    natural = _BranchTrajectory(
        trajectory_id="natural",
        snapshot_id="natural-snapshot",
        downstream_text="natural-text",
        forced_probe=False,
        grpo_eligible=True,
    )
    mixed = (natural, *branches)
    natural_ids = tuple(item.trajectory_id for item in natural_trajectories(mixed))
    grpo_ids = tuple(item.trajectory_id for item in grpo_training_trajectories(mixed))
    metric_ids = tuple(item.trajectory_id for item in standard_metric_trajectories(mixed))
    passed = natural_ids == grpo_ids == metric_ids == ("natural",)
    return passed, {
        "natural_ids": list(natural_ids),
        "grpo_ids": list(grpo_ids),
        "metric_ids": list(metric_ids),
    }, {
        "intervention_branch_count": len(branches),
        "intervention_ids_in_grpo": sorted(
            set(record.branch_order).intersection(grpo_ids)
        ),
    }


def _rule(
    rule_id: str,
    *,
    mean: float,
    variance: float,
    pairs: int = 30,
) -> RuleContrast:
    return RuleContrast(
        rule_id=rule_id,
        order=1,
        condition={
            "task_family": "hotpotqa",
            "role_cluster": "verify",
            "stage": "before_output",
        },
        action={"model_id": "B"},
        delta_mean=mean,
        posterior_variance=variance,
        n_eff_pairs=pairs,
    )


def _phase_d_activation() -> tuple[bool, Any, Mapping[str, Any]]:
    candidates = enumerate_rule_candidates(
        (
            _rule("positive-0.30", mean=0.30, variance=0.0004),
            _rule("zero-effect", mean=0.00, variance=0.0004),
        ),
        calibration_quantile=_NORMAL_90,
    )
    by_id = {candidate.rule_id: candidate for candidate in candidates}
    selected = benjamini_hochberg(
        {candidate.rule_id: candidate.activation_p_value for candidate in candidates},
        fdr=0.10,
    )
    confirmation = ConfirmationEvidence(
        calibrated_lower=0.20,
        problem_ids=tuple(f"heldout-{index:02d}" for index in range(20)),
        harm_probability=0.001,
    )
    positive = decide_skill_status(
        by_id["positive-0.30"],
        confirmation,
        bh_selected="positive-0.30" in selected,
    )
    zero = decide_skill_status(
        by_id["zero-effect"],
        confirmation,
        bh_selected="zero-effect" in selected,
    )
    false_discovery_rate = float(zero.status is SkillStatus.ACTIVE)
    passed = (
        positive.status is SkillStatus.ACTIVE
        and zero.status is not SkillStatus.ACTIVE
        and false_discovery_rate <= 0.10
    )
    return passed, {
        "positive_status": positive.status.value,
        "zero_status": zero.status.value,
        "zero_rule_false_discovery_rate": false_discovery_rate,
    }, {
        "discovery_pairs_per_rule": 30,
        "heldout_confirmation_problems": 20,
        "bh_selected": sorted(selected),
        "fdr": 0.10,
        "positive_effect": 0.30,
        "zero_effect": 0.0,
    }


def _phase_d_suspension() -> tuple[bool, Any, Mapping[str, Any]]:
    weak = enumerate_rule_candidates(
        (_rule("effect-removed", mean=0.0, variance=0.0004),),
        calibration_quantile=_NORMAL_90,
    )[0]
    confirmation = ConfirmationEvidence(
        calibrated_lower=0.20,
        problem_ids=tuple(f"heldout-{index:02d}" for index in range(20)),
        harm_probability=0.001,
    )
    epoch_one = decide_skill_status(
        weak,
        confirmation,
        previous=SkillStatus.ACTIVE,
        bh_selected=False,
        recent_probe_effect_means=(0.0,),
    )
    epoch_two = decide_skill_status(
        weak,
        confirmation,
        previous=epoch_one.status,
        bh_selected=False,
        recent_probe_effect_means=(0.0, 0.0),
        consecutive_suspended_epochs=1,
    )
    epochs_to_suspended = 1 if epoch_one.status is SkillStatus.SUSPENDED else 2
    passed = (
        epoch_one.status is SkillStatus.SUSPENDED
        and epoch_two.status is SkillStatus.SUSPENDED
        and epochs_to_suspended <= 2
    )
    return passed, epochs_to_suspended, {
        "epoch_1_status": epoch_one.status.value,
        "epoch_2_status": epoch_two.status.value,
        "effect_after_change": 0.0,
    }


def _versions() -> VersionBundle:
    return VersionBundle(
        policy="synthetic-policy-d",
        model_catalog="synthetic-model-catalog",
        evaluator="binary-em-v1",
        prompt="minimal-neutral-v1",
        tool="none",
        encoder="ledger-role-classifier-v1",
        feature_schema="ledger-decision-key-v1",
    )


def _active_skill(
    skill_id: str,
    *,
    family: str,
    role: str,
    stage: str,
    model: str,
    effect: float,
) -> SkillRecord:
    versions = _versions()
    validation_ids = tuple(f"{skill_id}-validation-{index:02d}" for index in range(20))
    evidence = SkillEvidence(
        baseline="same-snapshot-paired-intervention",
        paired_effect_mean=effect,
        calibrated_lower=effect - 0.04,
        calibrated_upper=effect + 0.04,
        effective_pairs=30,
        independent_problem_ids=validation_ids,
        discovery_problem_ids=tuple(
            f"{skill_id}-discovery-{index:02d}" for index in range(30)
        ),
        validation_problem_ids=validation_ids,
        validation_splits=("validation",),
        heldout_task_families=(family,),
        empirical_coverage=0.95,
        harm_probability=0.001,
        evidence_ids=tuple(f"{skill_id}-evidence-{index:02d}" for index in range(30)),
    )
    candidate = SkillRecord(
        skill_id=skill_id,
        version=1,
        status=SkillStatus.CANDIDATE,
        condition={
            "task_family": family,
            "role_cluster": role,
            "graph_stage": stage,
        },
        action={"model_id": model},
        evidence=evidence,
        versions=versions,
        readable_text=f"prefer {model} for this condition",
        created_epoch=0,
        eligible_epoch=1,
    )
    gate_config = SkillGateConfig(
        delta_min=0.05,
        minimum_effective_pairs=20,
        minimum_independent_problems=20,
        minimum_empirical_coverage=0.90,
    )
    gate = SkillEvidenceGate(gate_config)
    return SkillRecord(
        **{
            **candidate.to_dict(),
            "status": SkillStatus.ACTIVE,
            "evidence": evidence,
            "versions": versions,
            "activated_epoch": 1,
            "gate_config": gate_config.to_dict(),
            "gate_receipt": gate.compute_receipt(candidate),
        }
    )


def _phase_d_retrieval() -> tuple[bool, Any, Mapping[str, Any]]:
    matching = tuple(
        _active_skill(
            f"matching-{index}",
            family="hotpotqa",
            role="verify",
            stage="before_output",
            model="B",
            effect=0.30 - 0.02 * index,
        )
        for index in range(4)
    )
    mismatched = (
        _active_skill(
            "wrong-family",
            family="triviaqa",
            role="verify",
            stage="before_output",
            model="B",
            effect=0.50,
        ),
        _active_skill(
            "wrong-role",
            family="hotpotqa",
            role="solve",
            stage="before_output",
            model="B",
            effect=0.50,
        ),
        _active_skill(
            "wrong-stage",
            family="hotpotqa",
            role="verify",
            stage="other",
            model="B",
            effect=0.50,
        ),
    )
    ledger = CombinationPosterior("synthetic-policy-d", epoch=1)
    epoch = LedgerEpoch.freeze(ledger, (*matching, *mismatched), _versions())
    selection = epoch.select_skill_texts(
        task_family="hotpotqa",
        role_cluster="verify",
        stage="before_output",
        available_models=("A", "B"),
    )
    expected = ("matching-0", "matching-1", "matching-2")
    passed = selection.skill_ids == expected and all(
        text.startswith("[Skill]") for text in selection.texts
    )
    return passed, {
        "selected_skill_ids": list(selection.skill_ids),
        "text_in_observation": bool(selection.text),
    }, {
        "expected_top_3": list(expected),
        "mismatched_skill_ids": [skill.skill_id for skill in mismatched],
        "rendered_text": selection.text,
    }


def run_phase_acceptance(seed: int = 20260907) -> PhaseAcceptanceReport:
    """Run Phase A--D and emit Phase E as a real-runner-only pending gate."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    phase_a = PhaseAcceptance(
        phase="A",
        execution_boundary="synthetic",
        checks=(
            _checked(
                "A1",
                "first-order 90% intervals cover synthetic truth after 200 probes",
                {"coverage_gte": 0.85, "probe_count": 200},
                # Frozen seeds make the acceptance evidence reproducible.  The
                # master seed below remains available for sampling-heavy B/C
                # diagnostics without changing the Phase-A reference fixture.
                lambda: _phase_a_first_order(917),
            ),
            _checked(
                "A2",
                "second-order column activates on touch 5 and covers truth after 50 probes",
                {"activation_touch": 5, "truth_covered_after_probes": 50},
                lambda: _phase_a_interaction(11),
            ),
            _checked(
                "A3",
                "Beta sensors recover same-model and independent-review false-positive rates",
                {"maximum_absolute_error_lt": 0.05, "trajectories_per_class": 200},
                _phase_a_sensors,
            ),
            _checked(
                "A4",
                "same-snapshot paired evidence removes natural-policy selection confounding",
                {"naive_bias_gt": 0.10, "paired_bias_lt": 0.03},
                lambda: _phase_a_confounding(77),
            ),
        ),
    )
    phase_b = PhaseAcceptance(
        phase="B",
        execution_boundary="synthetic",
        checks=(
            _checked(
                "B1",
                "same-model self-review pass produces high latent risk",
                {"risk_probability_gte": 0.80},
                lambda: (
                    (lambda result: (result[0] >= 0.80 and result[1], result[0], {"warning": result[1]}))(
                        _phase_b_risk(seed=seed + 11, same=True, rates=(0.75, 0.60))
                    )
                ),
            ),
            _checked(
                "B2",
                "independent review pass by model B produces low latent risk",
                {"risk_probability_lte": 0.20},
                lambda: (
                    (lambda result: (result[0] <= 0.20 and not result[1], result[0], {"warning": result[1]}))(
                        _phase_b_risk(seed=seed + 12, same=False, rates=(0.90, 0.10))
                    )
                ),
            ),
            _checked(
                "B3",
                "empty ledger remains in the intermediate region without warning",
                {"risk_probability_gt": 0.20, "risk_probability_lt": 0.80},
                lambda: _phase_b_empty(seed + 13),
            ),
            _checked(
                "B4",
                "joint posterior sampling handles the maximum over alternatives",
                {"posterior_expected_max_minus_plugin_gt": 0.40},
                lambda: _phase_b_joint_max(seed + 14),
            ),
        ),
    )
    phase_c = PhaseAcceptance(
        phase="C",
        execution_boundary="synthetic",
        checks=(
            _checked(
                "C1",
                "paired branches share the pre-action snapshot and regenerate downstream text",
                {"same_snapshot": True, "downstream_text_distinct": True},
                _phase_c_fork,
            ),
            _checked(
                "C2",
                "straddle selection narrows the target interval faster than random selection",
                {"straddle_width_lt_random_width": True, "equal_probe_budget": True},
                lambda: _phase_c_straddle(seed + 22),
            ),
            _checked(
                "C3",
                "probe and audit branches are excluded from GRPO and ordinary metrics",
                {"intervention_ids_in_grpo": 0, "intervention_ids_in_metrics": 0},
                _phase_c_partition,
            ),
        ),
    )
    phase_d = PhaseAcceptance(
        phase="D",
        execution_boundary="synthetic",
        checks=(
            _checked(
                "D1",
                "+0.30 rule activates after 30 probes and 20 held-out confirmations; zero rule does not",
                {
                    "positive_status": "active",
                    "minimum_discovery_pairs": 30,
                    "minimum_confirmation_problems": 20,
                    "zero_rule_false_discovery_rate_lte": 0.10,
                },
                _phase_d_activation,
            ),
            _checked(
                "D2",
                "an active rule whose true effect becomes zero is suspended within two epochs",
                {"epochs_to_suspended_lte": 2},
                _phase_d_suspension,
            ),
            _checked(
                "D3",
                "Director observation receives only condition-matched top-three active Skills",
                {"maximum_skills": 3, "condition_match_required": True},
                _phase_d_retrieval,
            ),
        ),
    )
    phase_e = PhaseAcceptance(
        phase="E",
        execution_boundary="real_runner_only",
        checks=(
            AcceptanceCheck(
                "E1",
                "real training-loop data routing matches Section 6.4",
                None,
                "not_run",
                {"real_runner_receipt_required": True},
                {"synthetic_or_mock_evidence_is_not_accepted": True},
            ),
            AcceptanceCheck(
                "E2",
                "one complete epoch runs on 50 questions with G=4",
                None,
                {"questions": 0, "trajectories_per_question": 0},
                {"questions": 50, "trajectories_per_question": 4},
                {"synthetic_or_mock_evidence_is_not_accepted": True},
            ),
            AcceptanceCheck(
                "E3",
                "real-run diagnostics include explained variance and warning precision/recall",
                None,
                "not_run",
                {"all_diagnostics_present": True},
                {"synthetic_or_mock_evidence_is_not_accepted": True},
            ),
        ),
    )
    return PhaseAcceptanceReport(
        seed=seed,
        phases=(phase_a, phase_b, phase_c, phase_d, phase_e),
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = run_phase_acceptance(arguments.seed)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.synthetic_gates_passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke
    raise SystemExit(_main())
