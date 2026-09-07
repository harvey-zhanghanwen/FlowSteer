"""Runtime coordinator for one frozen dynamic-ledger evidence epoch.

This is a narrow adapter over the existing FlowSteer progressive Canvas
collector.  Natural trajectories retain their original exact behavior
receipts and are the only records returned to Action-Masked One-Pass GRPO.
The coordinator optionally forks selected same-snapshot paired interventions;
those branches are persisted in a separate evidence plane and can update only
the Combination Posterior, task baseline, surface-signal sensors, and Skill
evidence at the epoch boundary.

The module owns no optimizer, model server, reward, or W&B lifecycle.  Those
remain in the existing HotpotQA runner and SkillFlow-derived runtime adapters.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from ..records import ProbeRecord, TaskRecord, TrajectoryRecord
from ..skills.schema import SkillRecord, SkillStatus
from ..versioning import VersionBundle
from .combination_ledger import CombinationPosterior, DecisionKey
from .latent_loss import LatentLossConfig
from .ledger_epoch import (
    EpochSnapshotReceipt,
    EpochTransition,
    LedgerEpoch,
    LedgerTrajectoryRecord,
)
from .ledger_probe import (
    StraddleCandidate,
    assert_probe_data_partition,
    grpo_training_trajectories,
    make_probe_branch_ids,
    select_seeded_audits,
    select_straddle_candidates,
    standard_metric_trajectories,
)
from .probe_intervention import ContractRewriter, ProbeInterventionTranslator
from .role_classifier import ROLE_CLUSTERS, RoleClassifier
from .rollout_hook import HotpotQADynamicLedgerHook, ProbeSite, TrajectoryLedgerSidecar
from .training_bridge import (
    LEDGER_FEATURE_SCHEMA_VERSION,
    build_probe_record,
    intervention_step,
    natural_ledger_trajectory,
    probe_ledger_trajectory,
)


DYNAMIC_EPOCH_RUNTIME_VERSION = "hotpotqa-dynamic-ledger-runtime-v1"
DYNAMIC_EPOCH_TRANSITION_RECEIPT_VERSION = (
    "hotpotqa-dynamic-ledger-epoch-transition-v1"
)
_EDGE_TYPES = ("independent", "unidirectional", "bidirectional")
_STAGES = ("before_output", "other")


class DynamicLedgerBackend(Protocol):
    """Existing collector surface used by the coordinator."""

    model_catalog_version: str

    async def collect(
        self,
        task: TaskRecord,
        rollout_index: int,
        versions: VersionBundle,
        *,
        expected_task_split: str = "train",
        observation_hook: object | None = None,
        skills: Sequence[Mapping[str, Any]] = (),
        condition_id: Optional[str] = None,
    ) -> TrajectoryRecord: ...

    async def collect_probe_branch(
        self,
        task: TaskRecord,
        rollout_index: int,
        versions: VersionBundle,
        *,
        initial_snapshot: object,
        intervention: object,
        branch_id: str,
        expected_task_split: str = "train",
        observation_hook: object | None = None,
        skills: Sequence[Mapping[str, Any]] = (),
        condition_id: Optional[str] = None,
    ) -> TrajectoryRecord: ...


@dataclass(frozen=True)
class SelectedProbeSite:
    """One immutable site selected for either a probe or an audit."""

    site: ProbeSite
    audit: bool
    sampling_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.site, ProbeSite):
            raise TypeError("site must be ProbeSite")
        if type(self.audit) is not bool:
            raise TypeError("audit must be bool")
        if not 0.0 <= float(self.sampling_probability) <= 1.0:
            raise ValueError("sampling_probability must lie in [0, 1]")


@dataclass(frozen=True)
class DynamicLedgerBatch:
    """Natural and intervention evidence from one frozen policy condition."""

    epoch: LedgerEpoch
    natural_trajectories: tuple[TrajectoryRecord, ...]
    natural_ledger_records: tuple[LedgerTrajectoryRecord, ...]
    natural_sidecars: tuple[TrajectoryLedgerSidecar, ...]
    selected_sites: tuple[SelectedProbeSite, ...] = ()
    intervention_trajectories: tuple[TrajectoryRecord, ...] = ()
    intervention_ledger_records: tuple[LedgerTrajectoryRecord, ...] = ()
    probe_records: tuple[ProbeRecord, ...] = ()

    @property
    def all_ledger_records(self) -> tuple[LedgerTrajectoryRecord, ...]:
        return self.natural_ledger_records + self.intervention_ledger_records

    def __post_init__(self) -> None:
        natural_ids = {item.trajectory_id for item in self.natural_trajectories}
        ledger_natural_ids = {
            item.trajectory_id for item in self.natural_ledger_records
        }
        if natural_ids != ledger_natural_ids:
            raise ValueError("natural trajectory and ledger sidecar IDs differ")
        if any(item.forced_probe for item in self.natural_trajectories):
            raise ValueError("natural batch contains a forced probe")
        if any(
            not item.forced_probe or item.grpo_eligible
            for item in self.intervention_trajectories
        ):
            raise ValueError("intervention batch contains a GRPO-eligible branch")
        intervention_ids = {
            item.trajectory_id for item in self.intervention_trajectories
        }
        ledger_intervention_ids = {
            item.trajectory_id for item in self.intervention_ledger_records
        }
        if intervention_ids != ledger_intervention_ids:
            raise ValueError("intervention trajectory and ledger sidecar IDs differ")
        if set(natural_ids) & set(intervention_ids):
            raise ValueError("natural and intervention trajectory IDs overlap")
        for probe in self.probe_records:
            branch_ids = set(probe.branch_order)
            branches = tuple(
                item
                for item in self.intervention_trajectories
                if item.trajectory_id in branch_ids
            )
            assert_probe_data_partition(probe, branches)


def _trajectory_ids(
    values: Sequence[str],
    *,
    name: str,
    preserve_order: bool,
) -> tuple[str, ...]:
    identifiers = tuple(str(value).strip() for value in values)
    if any(not value for value in identifiers):
        raise ValueError(f"{name} must contain only non-empty trajectory IDs")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} must contain unique trajectory IDs")
    return identifiers if preserve_order else tuple(sorted(identifiers))


def _skill_status_counts(skills: Sequence[SkillRecord]) -> dict[str, int]:
    counts = {status.value: 0 for status in SkillStatus}
    for skill in skills:
        if not isinstance(skill, SkillRecord):
            raise TypeError("next_skills must contain only SkillRecord values")
        counts[skill.status.value] += 1
    return counts


def _heldout_calibration_receipt(
    coverage: float | None,
) -> dict[str, Any]:
    """Describe caller-supplied calibration without synthesizing evidence."""

    if coverage is None:
        return {
            "status": "pending",
            "coverage": None,
            "source": None,
            "applied_to_policy_refresh": False,
        }
    value = float(coverage)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("heldout_coverage must be finite and lie in [0, 1]")
    return {
        "status": "complete",
        "coverage": value,
        "source": "caller_provided_heldout_calibration",
        "applied_to_policy_refresh": True,
    }


@dataclass(frozen=True)
class DynamicEpochTransitionReceipt:
    """JSON receipt for one strict natural/probe epoch transition.

    The receipt intentionally contains no inferred held-out statistics.  A
    missing held-out calibration remains ``pending`` with a JSON ``null``
    coverage value.
    """

    source_condition_id: str
    next_condition_id: str
    source_policy_version: str
    next_policy_version: str
    natural_grpo_trajectory_ids: tuple[str, ...]
    intervention_exclusion_trajectory_ids: tuple[str, ...]
    update_summary: Mapping[str, Any]
    skill_status_counts: Mapping[str, int]
    heldout_calibration: Mapping[str, Any]
    next_epoch_snapshot: EpochSnapshotReceipt | None = None
    next_epoch_receipt_path: str | None = None
    schema_version: str = DYNAMIC_EPOCH_TRANSITION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_EPOCH_TRANSITION_RECEIPT_VERSION:
            raise ValueError(
                f"unsupported dynamic epoch receipt schema: {self.schema_version}"
            )
        for name in (
            "source_condition_id",
            "next_condition_id",
            "source_policy_version",
            "next_policy_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        natural_ids = _trajectory_ids(
            self.natural_grpo_trajectory_ids,
            name="natural_grpo_trajectory_ids",
            preserve_order=True,
        )
        intervention_ids = _trajectory_ids(
            self.intervention_exclusion_trajectory_ids,
            name="intervention_exclusion_trajectory_ids",
            preserve_order=False,
        )
        if set(natural_ids) & set(intervention_ids):
            raise ValueError("natural GRPO and intervention exclusion IDs overlap")
        object.__setattr__(self, "natural_grpo_trajectory_ids", natural_ids)
        object.__setattr__(
            self, "intervention_exclusion_trajectory_ids", intervention_ids
        )
        expected_statuses = {status.value for status in SkillStatus}
        if set(self.skill_status_counts) != expected_statuses:
            raise ValueError("skill_status_counts must contain all four SkillStatus values")
        counts = {
            status: int(self.skill_status_counts[status])
            for status in sorted(expected_statuses)
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError("SkillStatus counts must be non-negative")
        object.__setattr__(self, "skill_status_counts", MappingProxyType(counts))
        calibration = dict(self.heldout_calibration)
        if calibration.get("status") not in {"pending", "complete"}:
            raise ValueError("heldout calibration status must be pending or complete")
        if calibration["status"] == "pending" and calibration.get("coverage") is not None:
            raise ValueError("pending heldout calibration cannot carry coverage")
        object.__setattr__(
            self, "heldout_calibration", MappingProxyType(calibration)
        )
        object.__setattr__(self, "update_summary", MappingProxyType(dict(self.update_summary)))
        if (self.next_epoch_snapshot is None) != (self.next_epoch_receipt_path is None):
            raise ValueError(
                "next epoch snapshot and receipt path must both be present or absent"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": "closed",
            "source_condition_id": self.source_condition_id,
            "next_condition_id": self.next_condition_id,
            "source_policy_version": self.source_policy_version,
            "next_policy_version": self.next_policy_version,
            "natural_grpo_trajectory_ids": list(
                self.natural_grpo_trajectory_ids
            ),
            "intervention_exclusion_trajectory_ids": list(
                self.intervention_exclusion_trajectory_ids
            ),
            "update_summary": dict(self.update_summary),
            "skill_status_counts": dict(self.skill_status_counts),
            "heldout_calibration": dict(self.heldout_calibration),
            "next_epoch_snapshot": (
                None
                if self.next_epoch_snapshot is None
                else self.next_epoch_snapshot.to_dict()
            ),
            "next_epoch_receipt_path": self.next_epoch_receipt_path,
            "routing_assertions": {
                "natural_grpo_ids_exact": True,
                "intervention_exclusion_ids_exact": True,
                "intervention_excluded_from_grpo": True,
            },
        }

    def recovery_mapping(self) -> dict[str, Any]:
        """Return the runner state needed to recover the next frozen epoch."""

        snapshot = self.next_epoch_snapshot
        return {
            "schema_version": self.schema_version,
            "status": "ready" if snapshot is not None else "not_persisted",
            "next_policy_version": self.next_policy_version,
            "next_condition_id": self.next_condition_id,
            "next_epoch_receipt_path": self.next_epoch_receipt_path,
            "ledger_snapshot_id": (
                None if snapshot is None else snapshot.ledger_snapshot_id
            ),
            "skill_snapshot_id": (
                None if snapshot is None else snapshot.skill_snapshot_id
            ),
            "skill_status_counts": dict(self.skill_status_counts),
            "heldout_calibration": dict(self.heldout_calibration),
        }

    def metrics_mapping(self) -> dict[str, Any]:
        """Return JSON-compatible transition metrics for logs and W&B adapters."""

        summary = self.update_summary
        return {
            "schema_version": self.schema_version,
            "natural_grpo_trajectory_count": len(
                self.natural_grpo_trajectory_ids
            ),
            "intervention_exclusion_trajectory_count": len(
                self.intervention_exclusion_trajectory_ids
            ),
            "baseline_update_count": int(summary["baseline_updates"]),
            "surface_sensor_update_count": int(summary["sensor_updates"]),
            "contrast_probe_update_count": int(
                summary["contrast_probe_updates"]
            ),
            "invalid_trajectory_count": int(summary["invalid_trajectories"]),
            "skill_status_counts": dict(self.skill_status_counts),
            "heldout_calibration_status": self.heldout_calibration["status"],
            "heldout_coverage": self.heldout_calibration["coverage"],
        }


@dataclass(frozen=True)
class DynamicEpochCloseResult:
    """Runtime object plus its serializable transition receipt."""

    transition: EpochTransition
    receipt: DynamicEpochTransitionReceipt

    @property
    def next_epoch(self) -> LedgerEpoch:
        return self.transition.next_epoch

    def recovery_mapping(self) -> dict[str, Any]:
        return self.receipt.recovery_mapping()

    def metrics_mapping(self) -> dict[str, Any]:
        return self.receipt.metrics_mapping()


def close_dynamic_epoch(
    batch: DynamicLedgerBatch,
    *,
    next_policy_version: str,
    next_skills: Sequence[SkillRecord],
    natural_grpo_trajectory_ids: Sequence[str],
    intervention_exclusion_trajectory_ids: Sequence[str],
    next_epoch_directory: str | Path | None = None,
    heldout_coverage: float | None = None,
) -> DynamicEpochCloseResult:
    """Close a frozen epoch through :meth:`LedgerEpoch.close` and bind receipts.

    The two ID sequences are supplied by the real runner: the first is the
    exact one-pass GRPO input order; the second is the complete intervention
    exclusion set.  This adapter rejects any discrepancy before posterior
    state is advanced and verifies the same partition again from
    :class:`EpochUpdateSummary` after ``LedgerEpoch.close``.
    """

    if not isinstance(batch, DynamicLedgerBatch):
        raise TypeError("batch must be DynamicLedgerBatch")
    natural_grpo_ids = _trajectory_ids(
        natural_grpo_trajectory_ids,
        name="natural_grpo_trajectory_ids",
        preserve_order=True,
    )
    excluded_intervention_ids = _trajectory_ids(
        intervention_exclusion_trajectory_ids,
        name="intervention_exclusion_trajectory_ids",
        preserve_order=False,
    )
    observed_grpo_ids = tuple(
        record.trajectory_id
        for record in grpo_training_trajectories(batch.natural_trajectories)
    )
    observed_natural_ids = tuple(
        record.trajectory_id for record in batch.natural_trajectories
    )
    ledger_grpo_ids = tuple(
        record.trajectory_id
        for record in batch.natural_ledger_records
        if record.grpo_eligible
    )
    if not (
        natural_grpo_ids
        == observed_grpo_ids
        == observed_natural_ids
        == ledger_grpo_ids
    ):
        raise ValueError(
            "natural GRPO trajectory IDs do not exactly match the frozen natural batch"
        )
    observed_intervention_ids = tuple(
        sorted(record.trajectory_id for record in batch.intervention_trajectories)
    )
    ledger_intervention_ids = tuple(
        sorted(record.trajectory_id for record in batch.intervention_ledger_records)
    )
    if not (
        excluded_intervention_ids
        == observed_intervention_ids
        == ledger_intervention_ids
    ):
        raise ValueError(
            "intervention exclusion IDs do not exactly match the frozen probe batch"
        )
    if set(natural_grpo_ids) & set(excluded_intervention_ids):
        raise ValueError("intervention evidence overlaps the natural GRPO batch")

    skills = tuple(next_skills)
    calibration = _heldout_calibration_receipt(heldout_coverage)
    transition = batch.epoch.close(
        trajectories=batch.all_ledger_records,
        probes=batch.probe_records,
        next_policy_version=next_policy_version,
        next_skills=skills,
        heldout_coverage=heldout_coverage,
    )
    summary = transition.summary
    if tuple(summary.grpo_trajectory_ids) != natural_grpo_ids:
        raise AssertionError(
            "LedgerEpoch.close returned different natural GRPO trajectory IDs"
        )
    if tuple(sorted(summary.excluded_intervention_ids)) != excluded_intervention_ids:
        raise AssertionError(
            "LedgerEpoch.close returned different intervention exclusion IDs"
        )

    snapshot: EpochSnapshotReceipt | None = None
    snapshot_path: str | None = None
    if next_epoch_directory is not None:
        destination = Path(next_epoch_directory)
        snapshot = transition.next_epoch.save(destination)
        snapshot_path = str(destination / "receipt.json")
    skill_counts = _skill_status_counts(transition.next_epoch.skills)
    receipt = DynamicEpochTransitionReceipt(
        source_condition_id=batch.epoch.condition.condition_id,
        next_condition_id=transition.next_epoch.condition.condition_id,
        source_policy_version=batch.epoch.condition.versions.policy,
        next_policy_version=transition.next_epoch.condition.versions.policy,
        natural_grpo_trajectory_ids=natural_grpo_ids,
        intervention_exclusion_trajectory_ids=excluded_intervention_ids,
        update_summary=summary.to_dict(),
        skill_status_counts=skill_counts,
        heldout_calibration=calibration,
        next_epoch_snapshot=snapshot,
        next_epoch_receipt_path=snapshot_path,
    )
    return DynamicEpochCloseResult(transition=transition, receipt=receipt)


def freeze_initial_epoch(
    *,
    versions: VersionBundle,
    model_ids: Sequence[str],
    skills: Sequence[SkillRecord] = (),
    epoch: int = 0,
) -> LedgerEpoch:
    """Create the empty ledger and freeze its complete first-order domain."""

    if versions.feature_schema != LEDGER_FEATURE_SCHEMA_VERSION:
        versions = replace(versions, feature_schema=LEDGER_FEATURE_SCHEMA_VERSION)
    ledger = CombinationPosterior(versions.policy, epoch=epoch)
    normalized_models = tuple(dict.fromkeys(str(value).strip() for value in model_ids))
    if not normalized_models or any(not value for value in normalized_models):
        raise ValueError("model_ids must contain non-empty stable catalog IDs")
    for model_id in normalized_models:
        for role in sorted(ROLE_CLUSTERS):
            for edge_type in _EDGE_TYPES:
                same_options = (False,) if edge_type == "independent" else (False, True)
                for same_model in same_options:
                    for stage in _STAGES:
                        ledger.register_key(
                            DecisionKey(
                                task_family="hotpotqa",
                                role_cluster=role,
                                model_id=model_id,
                                edge_type=edge_type,
                                same_model_as_upstream=same_model,
                                stage=stage,
                            )
                        )
    return LedgerEpoch.freeze(ledger, skills, versions)


def _site_identity(site: ProbeSite) -> tuple[str, int, str]:
    return (site.trajectory_id, site.step_id, site.pre_snapshot.snapshot_id)


def select_probe_sites(
    sidecars: Sequence[TrajectoryLedgerSidecar],
    *,
    natural_trajectory_count: int,
    seed: int,
    probe_fraction: float = 0.10,
    audit_probability: float = 0.05,
    tau: float = 0.8,
) -> tuple[SelectedProbeSite, ...]:
    """Apply the specified straddle and warning-audit selection rules."""

    sites = tuple(site for sidecar in sidecars for site in sidecar.probe_sites)
    by_identity = {_site_identity(site): site for site in sites}
    if len(by_identity) != len(sites):
        raise ValueError("probe site identities must be unique")
    candidates = tuple(
        StraddleCandidate(
            trajectory_id=site.trajectory_id,
            step_id=str(site.step_id),
            snapshot_id=site.pre_snapshot.snapshot_id,
            risk_probability=float(site.risk_probability),
            score=float(site.straddle_score),
            key_keep=site.current_key.to_dict(),
            key_switch=site.candidate_key.to_dict(),
        )
        for site in sites
        if site.risk_probability is not None
    )
    selected = select_straddle_candidates(
        candidates,
        natural_trajectory_count=natural_trajectory_count,
        tau=tau,
        fraction=probe_fraction,
    )
    audits = select_seeded_audits(
        candidates,
        seed=seed,
        probability=audit_probability,
        tau=tau,
    )
    values: list[SelectedProbeSite] = []
    for item in selected:
        identity = (item.trajectory_id, int(item.step_id), item.snapshot_id)
        values.append(
            SelectedProbeSite(
                site=by_identity[identity],
                audit=False,
                sampling_probability=probe_fraction,
            )
        )
    for item in audits:
        identity = (item.trajectory_id, int(item.step_id), item.snapshot_id)
        if any(_site_identity(value.site) == identity for value in values):
            continue
        values.append(
            SelectedProbeSite(
                site=by_identity[identity],
                audit=True,
                sampling_probability=audit_probability,
            )
        )
    return tuple(values)


class DynamicLedgerEpochCoordinator:
    """Collect natural rollouts and isolated paired continuations."""

    def __init__(
        self,
        *,
        epoch: LedgerEpoch,
        role_classifier: RoleClassifier,
        contract_rewriter: ContractRewriter,
        model_catalog: Sequence[str],
        model_catalog_version: str,
        max_rounds: int,
        seed: int,
        rollout_concurrency: int = 28,
        probe_concurrency: int = 4,
        latent_config: LatentLossConfig = LatentLossConfig(),
    ) -> None:
        if not isinstance(epoch, LedgerEpoch):
            raise TypeError("epoch must be LedgerEpoch")
        if not isinstance(role_classifier, RoleClassifier):
            raise TypeError("role_classifier must be RoleClassifier")
        if not all(callable(getattr(contract_rewriter, name, None)) for name in ("rewrite",)):
            raise TypeError("contract_rewriter must expose rewrite()")
        catalog = tuple(dict.fromkeys(str(value).strip() for value in model_catalog))
        if not catalog or any(not value for value in catalog):
            raise ValueError("model_catalog must contain non-empty model IDs")
        if not model_catalog_version.strip():
            raise ValueError("model_catalog_version must be non-empty")
        for name, value in {
            "max_rounds": max_rounds,
            "rollout_concurrency": rollout_concurrency,
            "probe_concurrency": probe_concurrency,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.epoch = epoch
        self.role_classifier = role_classifier
        self.contract_rewriter = contract_rewriter
        self.model_catalog = catalog
        self.model_catalog_version = model_catalog_version.strip()
        self.max_rounds = max_rounds
        self.seed = seed
        self.rollout_concurrency = rollout_concurrency
        self.probe_concurrency = probe_concurrency
        self.latent_config = latent_config

    def make_hook(self, rollout_index: int) -> HotpotQADynamicLedgerHook:
        """Build a hook bound to this frozen epoch and rollout seed."""

        return HotpotQADynamicLedgerHook(
            epoch=self.epoch,
            role_classifier=self.role_classifier,
            model_catalog=self.model_catalog,
            max_rounds=self.max_rounds,
            latent_config=self.latent_config,
            random_seed=self.seed + rollout_index,
        )

    def _hook(self, rollout_index: int) -> HotpotQADynamicLedgerHook:
        """Backward-compatible alias for callers predating the public hook API."""

        return self.make_hook(rollout_index)

    async def collect_natural(
        self,
        backend: DynamicLedgerBackend,
        tasks: Sequence[TaskRecord],
        *,
        rollouts_per_task: int,
        start_rollout_index: int = 0,
        expected_task_split: str = "train",
    ) -> DynamicLedgerBatch:
        if rollouts_per_task < 1:
            raise ValueError("rollouts_per_task must be positive")
        if expected_task_split not in {"train", "validation"}:
            raise ValueError(
                "dynamic ledger collection accepts only train or validation tasks"
            )
        if any(task.split != expected_task_split for task in tasks):
            raise ValueError(
                "dynamic ledger tasks differ from expected_task_split"
            )
        semaphore = asyncio.Semaphore(self.rollout_concurrency)

        async def one(task: TaskRecord, rollout_index: int):
            hook = self.make_hook(rollout_index)
            async with semaphore:
                record = await backend.collect(
                    task,
                    rollout_index,
                    self.epoch.condition.versions,
                    expected_task_split=expected_task_split,
                    observation_hook=hook,
                    condition_id=self.epoch.condition.condition_id,
                )
            sidecar = hook.sidecar(record.trajectory_id)
            ledger_record = natural_ledger_trajectory(
                record,
                condition=self.epoch.condition,
                steps=sidecar.steps,
            )
            return record, sidecar, ledger_record

        jobs = []
        rollout_index = start_rollout_index
        for task in tasks:
            for _ in range(rollouts_per_task):
                jobs.append(one(task, rollout_index))
                rollout_index += 1
        results = tuple(await asyncio.gather(*jobs))
        trajectories = tuple(value[0] for value in results)
        sidecars = tuple(value[1] for value in results)
        ledger_records = tuple(value[2] for value in results)
        grpo_records = grpo_training_trajectories(trajectories)
        if expected_task_split == "train":
            if len(grpo_records) != len(trajectories):
                raise ValueError(
                    "a natural discovery epoch contains a non-GRPO trajectory"
                )
        elif grpo_records:
            raise ValueError(
                "held-out calibration trajectories must be GRPO-ineligible"
            )
        return DynamicLedgerBatch(
            epoch=self.epoch,
            natural_trajectories=trajectories,
            natural_ledger_records=ledger_records,
            natural_sidecars=sidecars,
        )

    async def collect_selected_probes(
        self,
        backend: DynamicLedgerBackend,
        natural: DynamicLedgerBatch,
        *,
        selected_sites: Sequence[SelectedProbeSite],
        start_rollout_index: int,
        expected_task_split: str = "train",
    ) -> DynamicLedgerBatch:
        if natural.epoch.condition != self.epoch.condition:
            raise ValueError("natural batch belongs to another frozen epoch")
        if expected_task_split not in {"train", "validation"}:
            raise ValueError(
                "paired probes accept only train or validation tasks"
            )
        if any(
            record.task.split != expected_task_split
            for record in natural.natural_trajectories
        ):
            raise ValueError(
                "paired-probe natural tasks differ from expected_task_split"
            )
        task_by_trajectory = {
            record.trajectory_id: record.task for record in natural.natural_trajectories
        }
        translator = ProbeInterventionTranslator(
            contract_rewriter=self.contract_rewriter,
            model_catalog=self.model_catalog,
            model_catalog_version=self.model_catalog_version,
        )
        semaphore = asyncio.Semaphore(self.probe_concurrency)

        async def one_probe(selection: SelectedProbeSite, ordinal: int):
            site = selection.site
            task = task_by_trajectory.get(site.trajectory_id)
            if task is None:
                raise ValueError("selected probe site has no natural task binding")
            async with semaphore:
                intervention = await translator.translate(
                    site.actual_action,
                    site.pre_snapshot,
                    site.current_key,
                    site.candidate_key,
                    seed=self.seed + start_rollout_index + ordinal,
                )
                probe_id = f"probe:{site.trajectory_id}:{site.step_id}"
                identifiers = make_probe_branch_ids(
                    probe_id,
                    seed=self.seed + start_rollout_index + ordinal,
                )
                branches: dict[str, TrajectoryRecord] = {}
                branch_ledgers: list[LedgerTrajectoryRecord] = []
                for branch_offset, branch_id in enumerate(identifiers.execution_order):
                    keep_arm = branch_id in identifiers.keep
                    action = intervention.keep if keep_arm else intervention.switch
                    hook = self.make_hook(
                        start_rollout_index + ordinal * 10 + branch_offset
                    )
                    branch = await backend.collect_probe_branch(
                        task,
                        start_rollout_index + ordinal * 10 + branch_offset,
                        self.epoch.condition.versions,
                        initial_snapshot=site.pre_snapshot,
                        intervention=action,
                        branch_id=branch_id,
                        expected_task_split=expected_task_split,
                        observation_hook=hook,
                        condition_id=self.epoch.condition.condition_id,
                    )
                    branches[branch_id] = branch
                    sidecar = hook.sidecar(branch_id)
                    executed_key = (
                        site.current_key if keep_arm else site.candidate_key
                    )
                    alternative_key = (
                        site.candidate_key if keep_arm else site.current_key
                    )
                    forced = intervention_step(
                        trajectory_id=branch_id,
                        step_id=f"{site.step_id}:forced",
                        snapshot_id=site.pre_snapshot.snapshot_id,
                        key=executed_key,
                        alternative=alternative_key,
                        translation_receipt=intervention.receipt.to_dict(),
                        audit=selection.audit,
                    )
                    branch_ledgers.append(
                        probe_ledger_trajectory(
                            branch,
                            condition=self.epoch.condition,
                            probe_id=probe_id,
                            audit=selection.audit,
                            forced_step=forced,
                            downstream_steps=sidecar.steps,
                        )
                    )
                probe = build_probe_record(
                    probe_id=probe_id,
                    problem_id=task.task_id,
                    task_split=expected_task_split,
                    snapshot_id=site.pre_snapshot.snapshot_id,
                    condition=self.epoch.condition,
                    key_keep=site.current_key,
                    key_switch=site.candidate_key,
                    branch_order=identifiers.execution_order,
                    branches=branches,
                    sampling_probability=selection.sampling_probability,
                    is_audit=selection.audit,
                )
                branch_values = tuple(branches[value] for value in identifiers.all_ids)
                assert_probe_data_partition(probe, branch_values)
                return probe, branch_values, tuple(branch_ledgers)

        results = tuple(
            await asyncio.gather(
                *(one_probe(site, index) for index, site in enumerate(selected_sites))
            )
        )
        probes = tuple(value[0] for value in results)
        branches = tuple(item for value in results for item in value[1])
        branch_ledgers = tuple(item for value in results for item in value[2])
        return DynamicLedgerBatch(
            epoch=self.epoch,
            natural_trajectories=natural.natural_trajectories,
            natural_ledger_records=natural.natural_ledger_records,
            natural_sidecars=natural.natural_sidecars,
            selected_sites=tuple(selected_sites),
            intervention_trajectories=branches,
            intervention_ledger_records=branch_ledgers,
            probe_records=probes,
        )


def dynamic_epoch_metrics(batch: DynamicLedgerBatch) -> Mapping[str, Any]:
    """Return task-only metrics plus the required ledger diagnostics."""

    natural = tuple(standard_metric_trajectories(batch.natural_trajectories))
    if not natural:
        raise ValueError("dynamic epoch has no evaluator-valid natural trajectories")
    exact_values = [float(item.evaluation.metrics["exact_match"]) for item in natural]
    f1_values = [float(item.evaluation.metrics["token_f1"]) for item in natural]
    sidecar_by_id = {item.trajectory_id: item for item in batch.natural_sidecars}
    failures = {
        item.trajectory_id for item, exact in zip(natural, exact_values) if exact == 0.0
    }
    warning_ids = {
        trajectory_id
        for trajectory_id, sidecar in sidecar_by_id.items()
        if any(site.warning for site in sidecar.probe_sites)
    }
    ignored_warning_ids = {
        trajectory_id
        for trajectory_id, sidecar in sidecar_by_id.items()
        if any(step.director_action == "continue" for step in sidecar.steps)
    }
    warning_precision = (
        len(ignored_warning_ids & failures) / len(ignored_warning_ids)
        if ignored_warning_ids
        else 0.0
    )
    warning_recall = (
        len(warning_ids & failures) / len(failures) if failures else 0.0
    )
    explained_variance = _decision_key_explained_variance(
        batch.natural_ledger_records
    )
    probe_count = sum(not item.audit for item in batch.selected_sites)
    audit_count = sum(item.audit for item in batch.selected_sites)
    grpo_ids = {
        item.trajectory_id
        for item in grpo_training_trajectories(batch.natural_trajectories)
    }
    intervention_ids = {
        item.trajectory_id for item in batch.intervention_trajectories
    }
    if grpo_ids & intervention_ids:
        raise ValueError("intervention evidence leaked into the GRPO batch")
    return MappingProxyType(
        {
            "schema_version": DYNAMIC_EPOCH_RUNTIME_VERSION,
            "condition": batch.epoch.condition.to_dict(),
            "natural_trajectory_count": len(batch.natural_trajectories),
            "valid_natural_trajectory_count": len(natural),
            "grpo_trajectory_count": len(grpo_ids),
            "intervention_trajectory_count": len(batch.intervention_trajectories),
            "standard_metric_intervention_count": 0,
            "probe_count": probe_count,
            "audit_count": audit_count,
            "exact_match": float(np.mean(exact_values)),
            "token_f1": float(np.mean(f1_values)),
            "terminal_failure_count": sum(
                item.termination_reason != "finish" for item in natural
            ),
            "warning_precision": float(warning_precision),
            "warning_recall": float(warning_recall),
            "warning_trajectory_count": len(warning_ids),
            "ignored_warning_trajectory_count": len(ignored_warning_ids),
            "decision_key_explained_variance": float(explained_variance),
            "routing_assertions": {
                "natural_only_grpo": grpo_ids
                == {item.trajectory_id for item in batch.natural_trajectories},
                "probe_and_audit_excluded_from_grpo": not bool(
                    grpo_ids & intervention_ids
                ),
                "probe_and_audit_excluded_from_standard_metrics": True,
                "comparison_fit_source": "same_snapshot_paired_probes_only",
                "latent_risk_reward_contribution": 0.0,
                "skill_reward_contribution": 0.0,
            },
        }
    )


def _decision_key_explained_variance(
    records: Sequence[LedgerTrajectoryRecord],
) -> float:
    """In-sample diagnostic R²; it never updates the comparison posterior."""

    valid = tuple(record for record in records if record.is_natural and record.evaluator_valid)
    if len(valid) < 2:
        return 0.0
    labels: list[tuple[str, str]] = []
    for record in valid:
        for step in record.steps:
            for factor, level in step.key.factor_levels().items():
                label = (factor, level)
                if label not in labels:
                    labels.append(label)
    if not labels:
        return 0.0
    x = np.ones((len(valid), 1 + len(labels)), dtype=np.float64)
    for row, record in enumerate(valid):
        counts: dict[tuple[str, str], int] = {}
        for step in record.steps:
            for factor, level in step.key.factor_levels().items():
                key = (factor, level)
                counts[key] = counts.get(key, 0) + 1
        for column, label in enumerate(labels, start=1):
            x[row, column] = float(counts.get(label, 0))
    y = np.asarray([record.terminal_reward for record in valid], dtype=np.float64)
    centered = y - float(np.mean(y))
    denominator = float(centered @ centered)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - fitted
    return float(np.clip(1.0 - float(residual @ residual) / denominator, 0.0, 1.0))


__all__ = [
    "DYNAMIC_EPOCH_RUNTIME_VERSION",
    "DYNAMIC_EPOCH_TRANSITION_RECEIPT_VERSION",
    "DynamicEpochCloseResult",
    "DynamicEpochTransitionReceipt",
    "DynamicLedgerBatch",
    "DynamicLedgerEpochCoordinator",
    "SelectedProbeSite",
    "close_dynamic_epoch",
    "dynamic_epoch_metrics",
    "freeze_initial_epoch",
    "select_probe_sites",
]
