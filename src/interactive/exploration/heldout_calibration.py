"""Held-out paired calibration receipts for the dynamic Combination Posterior.

This module implements the validation-only calibration boundary described in
Section 13.1 / Phase 2 of ``FlowSteer_MACE_Bayesian_Skill_Design.md`` and
Sections 7.2--7.3 of ``LatentLoss_Implementation_Spec.md``.  It consumes the
existing :class:`ProbeRecord` and intervention trajectories produced by the
FlowSteer same-snapshot continuation path; it does not introduce another
rollout, optimizer, or Skill publication path.

The builder is deliberately fail closed.  Calibration problems must be a
frozen set of unique validation tasks, disjoint from discovery tasks and
probes, and every paired probe must be bound to the frozen policy, condition,
evaluator, feature schema, and executor-model catalog.  Intervention branches
are revalidated as non-natural and GRPO-ineligible before any receipt is
reported as complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..records import ProbeRecord
from .combination_ledger import CombinationPosterior, DecisionKey
from .ledger_probe import assert_probe_data_partition, validate_probe_record
from ..skills.ledger_miner import empirical_calibration_quantile


HELDOUT_PAIRED_CALIBRATION_MANIFEST_VERSION = (
    "heldout-paired-calibration-manifest-v1"
)
HELDOUT_PAIRED_CALIBRATION_RECEIPT_VERSION = (
    "heldout-paired-calibration-receipt-v1"
)


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unique_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    identifiers = tuple(_non_empty(value, name) for value in values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} must contain unique IDs")
    return identifiers


def _finite(value: object, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class HeldoutPairedCalibrationManifest:
    """Frozen validation problem set and version boundary for calibration."""

    dataset: str
    problem_ids: Sequence[str]
    discovery_problem_ids: Sequence[str]
    discovery_probe_ids: Sequence[str]
    policy_version: str
    condition_id: str
    evaluator_version: str
    feature_schema_version: str
    model_catalog_version: str
    posterior_snapshot_id: str
    alpha: float = 0.05
    minimum_unique_problems: int = 20
    task_split: str = "validation"
    schema_version: str = HELDOUT_PAIRED_CALIBRATION_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HELDOUT_PAIRED_CALIBRATION_MANIFEST_VERSION:
            raise ValueError(
                f"unsupported held-out manifest schema: {self.schema_version}"
            )
        for name in (
            "dataset",
            "policy_version",
            "condition_id",
            "evaluator_version",
            "feature_schema_version",
            "model_catalog_version",
            "posterior_snapshot_id",
        ):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if self.task_split != "validation":
            raise ValueError("held-out paired calibration must use validation split")
        if (
            isinstance(self.minimum_unique_problems, bool)
            or not isinstance(self.minimum_unique_problems, int)
            or self.minimum_unique_problems < 1
        ):
            raise ValueError("minimum_unique_problems must be a positive integer")
        alpha = _finite(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        object.__setattr__(self, "alpha", alpha)
        problems = _unique_ids(self.problem_ids, "problem_ids")
        discovery_problems = _unique_ids(
            self.discovery_problem_ids, "discovery_problem_ids"
        )
        discovery_probes = _unique_ids(
            self.discovery_probe_ids, "discovery_probe_ids"
        )
        if len(problems) < self.minimum_unique_problems:
            raise ValueError(
                "held-out manifest has fewer than minimum_unique_problems"
            )
        if not discovery_problems:
            raise ValueError("discovery_problem_ids cannot be empty")
        if not discovery_probes:
            raise ValueError("discovery_probe_ids cannot be empty")
        overlap = set(problems).intersection(discovery_problems)
        if overlap:
            raise ValueError(
                "held-out and discovery problem IDs overlap: "
                + ",".join(sorted(overlap))
            )
        object.__setattr__(self, "problem_ids", problems)
        object.__setattr__(self, "discovery_problem_ids", discovery_problems)
        object.__setattr__(self, "discovery_probe_ids", discovery_probes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "task_split": self.task_split,
            "problem_ids": list(self.problem_ids),
            "discovery_problem_ids": list(self.discovery_problem_ids),
            "discovery_probe_ids": list(self.discovery_probe_ids),
            "policy_version": self.policy_version,
            "condition_id": self.condition_id,
            "evaluator_version": self.evaluator_version,
            "feature_schema_version": self.feature_schema_version,
            "model_catalog_version": self.model_catalog_version,
            "posterior_snapshot_id": self.posterior_snapshot_id,
            "alpha": self.alpha,
            "minimum_unique_problems": self.minimum_unique_problems,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HeldoutPairedCalibrationManifest":
        return cls(
            schema_version=str(value["schema_version"]),
            dataset=str(value["dataset"]),
            task_split=str(value["task_split"]),
            problem_ids=tuple(str(item) for item in value["problem_ids"]),
            discovery_problem_ids=tuple(
                str(item) for item in value["discovery_problem_ids"]
            ),
            discovery_probe_ids=tuple(
                str(item) for item in value["discovery_probe_ids"]
            ),
            policy_version=str(value["policy_version"]),
            condition_id=str(value["condition_id"]),
            evaluator_version=str(value["evaluator_version"]),
            feature_schema_version=str(value["feature_schema_version"]),
            model_catalog_version=str(value["model_catalog_version"]),
            posterior_snapshot_id=str(value["posterior_snapshot_id"]),
            alpha=float(value["alpha"]),
            minimum_unique_problems=int(value["minimum_unique_problems"]),
        )


@dataclass(frozen=True)
class HeldoutPairedCalibrationObservation:
    """One validation-problem contrast and its frozen posterior prediction."""

    probe_id: str
    problem_id: str
    snapshot_id: str
    condition_id: str
    policy_version: str
    evaluator_version: str
    feature_schema_version: str
    incumbent_action: Mapping[str, Any]
    candidate_action: Mapping[str, Any]
    branch_trajectory_ids: Sequence[str]
    observed_delta: float
    predicted_mean: float
    posterior_variance: float
    noise_variance: float
    standardized_residual: float

    def __post_init__(self) -> None:
        for name in (
            "probe_id",
            "problem_id",
            "snapshot_id",
            "condition_id",
            "policy_version",
            "evaluator_version",
            "feature_schema_version",
        ):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        branch_ids = _unique_ids(
            self.branch_trajectory_ids, "branch_trajectory_ids"
        )
        if len(branch_ids) != 6:
            raise ValueError("a held-out paired observation requires six branches")
        object.__setattr__(self, "branch_trajectory_ids", branch_ids)
        object.__setattr__(
            self, "incumbent_action", MappingProxyType(dict(self.incumbent_action))
        )
        object.__setattr__(
            self, "candidate_action", MappingProxyType(dict(self.candidate_action))
        )
        for name in (
            "observed_delta",
            "predicted_mean",
            "posterior_variance",
            "noise_variance",
            "standardized_residual",
        ):
            numeric = _finite(getattr(self, name), name)
            if name in {"posterior_variance", "noise_variance", "standardized_residual"}:
                if numeric < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, numeric)
        if self.posterior_variance + self.noise_variance <= 0.0:
            raise ValueError("total calibration variance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "problem_id": self.problem_id,
            "snapshot_id": self.snapshot_id,
            "condition_id": self.condition_id,
            "policy_version": self.policy_version,
            "evaluator_version": self.evaluator_version,
            "feature_schema_version": self.feature_schema_version,
            "incumbent_action": dict(self.incumbent_action),
            "candidate_action": dict(self.candidate_action),
            "branch_trajectory_ids": list(self.branch_trajectory_ids),
            "observed_delta": self.observed_delta,
            "predicted_mean": self.predicted_mean,
            "posterior_variance": self.posterior_variance,
            "noise_variance": self.noise_variance,
            "standardized_residual": self.standardized_residual,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "HeldoutPairedCalibrationObservation":
        return cls(
            probe_id=str(value["probe_id"]),
            problem_id=str(value["problem_id"]),
            snapshot_id=str(value["snapshot_id"]),
            condition_id=str(value["condition_id"]),
            policy_version=str(value["policy_version"]),
            evaluator_version=str(value["evaluator_version"]),
            feature_schema_version=str(value["feature_schema_version"]),
            incumbent_action=dict(value["incumbent_action"]),
            candidate_action=dict(value["candidate_action"]),
            branch_trajectory_ids=tuple(
                str(item) for item in value["branch_trajectory_ids"]
            ),
            observed_delta=float(value["observed_delta"]),
            predicted_mean=float(value["predicted_mean"]),
            posterior_variance=float(value["posterior_variance"]),
            noise_variance=float(value["noise_variance"]),
            standardized_residual=float(value["standardized_residual"]),
        )


@dataclass(frozen=True)
class HeldoutPairedCalibrationReceipt:
    """Persistable result of one independent validation paired set."""

    manifest: HeldoutPairedCalibrationManifest
    observations: Sequence[HeldoutPairedCalibrationObservation]
    calibration_quantile: float
    empirical_coverage: float
    uncalibrated_coverage_80: float
    uncalibrated_coverage_90: float
    uncalibrated_coverage_95: float
    calibrated_interval_width_mean: float
    gaussian_nll_mean: float
    direction_accuracy: float
    status: str = "complete"
    schema_version: str = HELDOUT_PAIRED_CALIBRATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HELDOUT_PAIRED_CALIBRATION_RECEIPT_VERSION:
            raise ValueError(
                f"unsupported held-out receipt schema: {self.schema_version}"
            )
        if self.status != "complete":
            raise ValueError("held-out calibration receipt status must be complete")
        if not isinstance(self.manifest, HeldoutPairedCalibrationManifest):
            raise TypeError("manifest must be HeldoutPairedCalibrationManifest")
        observations = tuple(self.observations)
        if not observations or any(
            not isinstance(item, HeldoutPairedCalibrationObservation)
            for item in observations
        ):
            raise TypeError("observations must contain calibration observations")
        object.__setattr__(self, "observations", observations)
        for name in (
            "calibration_quantile",
            "empirical_coverage",
            "uncalibrated_coverage_80",
            "uncalibrated_coverage_90",
            "uncalibrated_coverage_95",
            "calibrated_interval_width_mean",
            "gaussian_nll_mean",
            "direction_accuracy",
        ):
            numeric = _finite(getattr(self, name), name)
            if name in {
                "empirical_coverage",
                "uncalibrated_coverage_80",
                "uncalibrated_coverage_90",
                "uncalibrated_coverage_95",
                "direction_accuracy",
            } and not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            if name in {"calibration_quantile", "calibrated_interval_width_mean"}:
                if numeric < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, numeric)

    @property
    def problem_count(self) -> int:
        return len(self.observations)

    @property
    def probe_count(self) -> int:
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "scope": "heldout_paired_calibration_only",
            "manifest": self.manifest.to_dict(),
            "problem_count": self.problem_count,
            "probe_count": self.probe_count,
            "calibration_quantile": self.calibration_quantile,
            "empirical_coverage": self.empirical_coverage,
            "uncalibrated_coverage": {
                "0.80": self.uncalibrated_coverage_80,
                "0.90": self.uncalibrated_coverage_90,
                "0.95": self.uncalibrated_coverage_95,
            },
            "calibrated_interval_width_mean": self.calibrated_interval_width_mean,
            "gaussian_nll_mean": self.gaussian_nll_mean,
            "direction_accuracy": self.direction_accuracy,
            "brier_score": None,
            "brier_status": "not_identified_by_contrast_only_probe_receipt",
            "heldout_calibration_gate_passed": True,
            "does_not_authorize_training_by_itself": True,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HeldoutPairedCalibrationReceipt":
        coverage = dict(value["uncalibrated_coverage"])
        return cls(
            schema_version=str(value["schema_version"]),
            status=str(value["status"]),
            manifest=HeldoutPairedCalibrationManifest.from_dict(value["manifest"]),
            observations=tuple(
                HeldoutPairedCalibrationObservation.from_dict(item)
                for item in value["observations"]
            ),
            calibration_quantile=float(value["calibration_quantile"]),
            empirical_coverage=float(value["empirical_coverage"]),
            uncalibrated_coverage_80=float(coverage["0.80"]),
            uncalibrated_coverage_90=float(coverage["0.90"]),
            uncalibrated_coverage_95=float(coverage["0.95"]),
            calibrated_interval_width_mean=float(
                value["calibrated_interval_width_mean"]
            ),
            gaussian_nll_mean=float(value["gaussian_nll_mean"]),
            direction_accuracy=float(value["direction_accuracy"]),
        )


def _noise_variance(record: ProbeRecord, variance_floor: float) -> float:
    keep = np.asarray(record.incumbent_returns, dtype=np.float64)
    switch = np.asarray(record.candidate_returns, dtype=np.float64)
    if keep.shape != switch.shape:
        raise ValueError("held-out probe arms must use the same repeat count")
    keep_rate = float(np.mean(keep))
    switch_rate = float(np.mean(switch))
    return float(
        (
            switch_rate * (1.0 - switch_rate)
            + keep_rate * (1.0 - keep_rate)
        )
        / keep.size
        + variance_floor
    )


def _metrics(
    observations: Sequence[HeldoutPairedCalibrationObservation],
    *,
    alpha: float,
) -> dict[str, float]:
    observed = np.asarray(
        [item.observed_delta for item in observations], dtype=np.float64
    )
    predicted = np.asarray(
        [item.predicted_mean for item in observations], dtype=np.float64
    )
    posterior = np.asarray(
        [item.posterior_variance for item in observations], dtype=np.float64
    )
    noise = np.asarray(
        [item.noise_variance for item in observations], dtype=np.float64
    )
    total = posterior + noise
    quantile = empirical_calibration_quantile(
        observed,
        predicted,
        posterior,
        noise,
        alpha=alpha,
    )
    residuals = np.abs(observed - predicted) / np.sqrt(total)
    gaussian_nll = 0.5 * (
        np.log(2.0 * math.pi * total) + np.square(observed - predicted) / total
    )

    def coverage(q: float) -> float:
        return float(np.mean(residuals <= q + 1e-12))

    predicted_direction = np.sign(predicted)
    observed_direction = np.sign(observed)
    return {
        "calibration_quantile": float(quantile),
        "empirical_coverage": coverage(float(quantile)),
        "uncalibrated_coverage_80": coverage(1.2815515655446004),
        "uncalibrated_coverage_90": coverage(1.6448536269514722),
        "uncalibrated_coverage_95": coverage(1.959963984540054),
        "calibrated_interval_width_mean": float(
            np.mean(2.0 * float(quantile) * np.sqrt(posterior))
        ),
        "gaussian_nll_mean": float(np.mean(gaussian_nll)),
        "direction_accuracy": float(
            np.mean(predicted_direction == observed_direction)
        ),
    }


def _branch_condition_id(branch: object) -> str:
    return _non_empty(getattr(branch, "condition_id", None), "branch.condition_id")


def _validate_branch_versions(
    branch: object,
    manifest: HeldoutPairedCalibrationManifest,
) -> None:
    task = getattr(branch, "task", None)
    if task is None or getattr(task, "split", None) != "validation":
        raise ValueError("held-out intervention branch must use validation split")
    if str(getattr(task, "task_id", "")) not in set(manifest.problem_ids):
        raise ValueError("held-out branch task is absent from frozen manifest")
    versions = getattr(branch, "versions", None)
    if versions is None:
        raise ValueError("held-out branch is missing version bindings")
    if getattr(versions, "policy", None) != manifest.policy_version:
        raise ValueError("held-out branch policy version mismatch")
    if getattr(versions, "evaluator", None) != manifest.evaluator_version:
        raise ValueError("held-out branch evaluator version mismatch")
    if getattr(versions, "feature_schema", None) != manifest.feature_schema_version:
        raise ValueError("held-out branch feature schema version mismatch")
    if _branch_condition_id(branch) != manifest.condition_id:
        raise ValueError("held-out branch condition ID mismatch")


def build_heldout_paired_calibration(
    ledger: CombinationPosterior,
    manifest: HeldoutPairedCalibrationManifest,
    probes: Sequence[ProbeRecord],
    branch_trajectories: Sequence[object],
) -> HeldoutPairedCalibrationReceipt:
    """Validate and calibrate one frozen independent paired validation set.

    There must be exactly one paired probe per manifest problem.  This makes
    the conformal calibration unit the complete problem rather than treating
    repeated branches (or multiple graph nodes from one problem) as independent
    observations.  Each probe still contains the prescribed K=3 independent
    continuations for both arms.
    """

    if not isinstance(ledger, CombinationPosterior):
        raise TypeError("ledger must be CombinationPosterior")
    if not isinstance(manifest, HeldoutPairedCalibrationManifest):
        raise TypeError("manifest must be HeldoutPairedCalibrationManifest")
    if ledger.policy_version != manifest.policy_version:
        raise ValueError("posterior policy version differs from held-out manifest")
    records = tuple(probes)
    if not records or any(not isinstance(item, ProbeRecord) for item in records):
        raise TypeError("probes must contain ProbeRecord values")
    if len(records) != len(manifest.problem_ids):
        raise ValueError("held-out set requires exactly one probe per manifest problem")
    probe_ids = _unique_ids(
        tuple(item.probe_id for item in records), "heldout_probe_ids"
    )
    if set(probe_ids).intersection(manifest.discovery_probe_ids):
        raise ValueError("held-out and discovery probe IDs overlap")
    problem_ids = _unique_ids(
        tuple(item.problem_id for item in records), "heldout_probe_problem_ids"
    )
    if set(problem_ids) != set(manifest.problem_ids):
        raise ValueError("held-out probe problems differ from frozen manifest")
    snapshots = _unique_ids(
        tuple(item.snapshot_id for item in records), "heldout_snapshot_ids"
    )
    if len(snapshots) != len(records):
        raise AssertionError("held-out snapshot cardinality mismatch")

    branches_by_id = {
        str(getattr(item, "trajectory_id", "")): item
        for item in branch_trajectories
    }
    if "" in branches_by_id or len(branches_by_id) != len(branch_trajectories):
        raise ValueError("held-out branch trajectory IDs must be unique and non-empty")
    expected_branch_ids = {
        branch_id for record in records for branch_id in record.branch_order
    }
    if set(branches_by_id) != expected_branch_ids:
        raise ValueError("held-out branches differ from the exact ProbeRecord set")

    # Prediction must not mutate the frozen fitted posterior if an unexpected
    # validation action contains an unregistered feature level.
    prediction_ledger = CombinationPosterior.from_state_dict(ledger.state_dict())
    initial_columns = prediction_ledger.columns
    observations: list[HeldoutPairedCalibrationObservation] = []
    for record in sorted(records, key=lambda item: item.problem_id):
        validation = validate_probe_record(record)
        if record.task_split != "validation":
            raise ValueError("held-out paired evidence must use validation split")
        if bool(record.state_features.get("is_audit", False)):
            raise ValueError("warning audits cannot serve as held-out confirmation")
        if record.policy_version != manifest.policy_version:
            raise ValueError("held-out probe policy version mismatch")
        if record.evaluator_version != manifest.evaluator_version:
            raise ValueError("held-out probe evaluator version mismatch")
        if record.feature_schema_version != manifest.feature_schema_version:
            raise ValueError("held-out probe feature schema version mismatch")
        if not record.executor_versions:
            raise ValueError("held-out probe is missing executor versions")
        if set(record.executor_versions.values()) != {manifest.model_catalog_version}:
            raise ValueError("held-out probe executor model catalog mismatch")
        branch_values = tuple(branches_by_id[item] for item in validation.branch_ids)
        assert_probe_data_partition(record, branch_values)
        for branch in branch_values:
            _validate_branch_versions(branch, manifest)
            if str(getattr(branch.task, "task_id", "")) != record.problem_id:
                raise ValueError("held-out branch problem does not match ProbeRecord")

        key_keep = DecisionKey.from_dict(record.incumbent_action)
        key_switch = DecisionKey.from_dict(record.candidate_action)
        if key_keep.task_family != manifest.dataset:
            raise ValueError("held-out probe task family differs from dataset")
        estimate = prediction_ledger.predict_contrast(key_switch, key_keep)
        if prediction_ledger.columns != initial_columns:
            raise ValueError(
                "held-out probe introduced a feature level absent from frozen posterior"
            )
        noise_variance = _noise_variance(record, prediction_ledger.variance_floor)
        total_variance = estimate.variance + noise_variance
        residual = abs(record.paired_effect - estimate.mean) / math.sqrt(
            total_variance
        )
        observations.append(
            HeldoutPairedCalibrationObservation(
                probe_id=record.probe_id,
                problem_id=record.problem_id,
                snapshot_id=record.snapshot_id,
                condition_id=manifest.condition_id,
                policy_version=record.policy_version,
                evaluator_version=record.evaluator_version,
                feature_schema_version=record.feature_schema_version,
                incumbent_action=record.incumbent_action,
                candidate_action=record.candidate_action,
                branch_trajectory_ids=validation.branch_ids,
                observed_delta=record.paired_effect,
                predicted_mean=estimate.mean,
                posterior_variance=estimate.variance,
                noise_variance=noise_variance,
                standardized_residual=residual,
            )
        )
    metrics = _metrics(observations, alpha=manifest.alpha)
    if metrics["empirical_coverage"] + 1e-12 < 1.0 - manifest.alpha:
        raise ValueError("held-out empirical coverage is below the calibrated target")
    return HeldoutPairedCalibrationReceipt(
        manifest=manifest,
        observations=tuple(observations),
        **metrics,
    )


def validate_heldout_paired_calibration(
    receipt: HeldoutPairedCalibrationReceipt,
    *,
    expected_manifest: HeldoutPairedCalibrationManifest,
    ledger: CombinationPosterior | None = None,
) -> None:
    """Fail closed on stale/tampered metrics or version/manifest mismatch."""

    if not isinstance(receipt, HeldoutPairedCalibrationReceipt):
        raise TypeError("receipt must be HeldoutPairedCalibrationReceipt")
    if receipt.manifest.to_dict() != expected_manifest.to_dict():
        raise ValueError("held-out calibration manifest mismatch")
    if receipt.problem_count != len(expected_manifest.problem_ids):
        raise ValueError("held-out calibration problem count mismatch")
    observations = tuple(receipt.observations)
    if {item.problem_id for item in observations} != set(expected_manifest.problem_ids):
        raise ValueError("held-out calibration observations do not cover the manifest")
    if {item.probe_id for item in observations}.intersection(
        expected_manifest.discovery_probe_ids
    ):
        raise ValueError("held-out receipt overlaps discovery probes")
    metrics = _metrics(observations, alpha=expected_manifest.alpha)
    for name, expected in metrics.items():
        actual = float(getattr(receipt, name))
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"held-out receipt metric mismatch: {name}")
    if ledger is not None:
        if ledger.policy_version != expected_manifest.policy_version:
            raise ValueError("validation ledger policy version mismatch")
        prediction_ledger = CombinationPosterior.from_state_dict(ledger.state_dict())
        original_columns = prediction_ledger.columns
        for item in observations:
            estimate = prediction_ledger.predict_contrast(
                DecisionKey.from_dict(item.candidate_action),
                DecisionKey.from_dict(item.incumbent_action),
            )
            if prediction_ledger.columns != original_columns:
                raise ValueError(
                    "held-out receipt action is absent from frozen posterior"
                )
            if not math.isclose(
                item.predicted_mean, estimate.mean, rel_tol=1e-12, abs_tol=1e-12
            ) or not math.isclose(
                item.posterior_variance,
                estimate.variance,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("held-out receipt posterior prediction mismatch")


def save_heldout_paired_calibration(
    receipt: HeldoutPairedCalibrationReceipt,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically persist a validated JSON receipt."""

    validate_heldout_paired_calibration(
        receipt, expected_manifest=receipt.manifest
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_heldout_paired_calibration(
    path: str | os.PathLike[str],
    *,
    expected_manifest: HeldoutPairedCalibrationManifest,
    ledger: CombinationPosterior | None = None,
) -> HeldoutPairedCalibrationReceipt:
    """Load and fully validate a persisted held-out calibration receipt."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("held-out calibration receipt must be a JSON object")
    if value.get("heldout_calibration_gate_passed") is not True:
        raise ValueError("held-out calibration gate is not passed")
    if value.get("does_not_authorize_training_by_itself") is not True:
        raise ValueError("held-out receipt has invalid authorization scope")
    receipt = HeldoutPairedCalibrationReceipt.from_dict(value)
    validate_heldout_paired_calibration(
        receipt,
        expected_manifest=expected_manifest,
        ledger=ledger,
    )
    return receipt


__all__ = [
    "HELDOUT_PAIRED_CALIBRATION_MANIFEST_VERSION",
    "HELDOUT_PAIRED_CALIBRATION_RECEIPT_VERSION",
    "HeldoutPairedCalibrationManifest",
    "HeldoutPairedCalibrationObservation",
    "HeldoutPairedCalibrationReceipt",
    "build_heldout_paired_calibration",
    "load_heldout_paired_calibration",
    "save_heldout_paired_calibration",
    "validate_heldout_paired_calibration",
]
