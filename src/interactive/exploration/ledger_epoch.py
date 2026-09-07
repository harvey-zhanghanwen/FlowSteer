"""Epoch-scoped state coordination for the dynamic combination ledger.

This module is the data-plane boundary described by Sections 3.3, 6.4, and 8
of ``LatentLoss_Implementation_Spec.md``.  It deliberately contains no model
calls, Canvas execution, fork runner, reward shaping, or optimizer logic.

An :class:`LedgerEpoch` freezes the policy, ledger, and Skill snapshots used by
every natural rollout in one condition.  At the epoch boundary it applies the
strict evidence routing rules:

* natural trajectories may be selected by the existing GRPO path but never
  update the comparison posterior here;
* paired probe/audit records are the only comparison-posterior evidence;
* every evaluator-valid binary trajectory, natural or intervention-only,
  updates the task baseline and surface-signal sensors;
* newly activated Skills are visible only in the returned next epoch.

The resulting ledger and Skill snapshots can be saved with a small receipt and
loaded without reconstructing any runtime/model state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..persistence.ids import canonical_json, stable_id
from ..records import ProbeRecord, utc_now
from ..skills.retrieval import SkillQuery, SkillRetriever
from ..skills.schema import SkillRecord, SkillStatus
from ..versioning import VersionBundle
from .combination_ledger import CombinationPosterior, DecisionKey, SurfaceSignal
from .latent_loss import (
    LatentLossConfig,
    PostExecutionReadout,
    PreExecutionReadout,
    changed_field,
    post_execution_latent_loss,
    pre_execution_readout,
)
from .ledger_probe import assert_probe_data_partition, validate_probe_record


LEDGER_EPOCH_SCHEMA_VERSION = "flowsteer.ledger-epoch.v1"
_DIRECTOR_ACTIONS = frozenset({"continue", "revise", "not_triggered"})


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_copy(value: Any) -> Any:
    """Return a detached JSON value and reject non-serializable state."""

    return json.loads(canonical_json(value))


def _skill_copies(skills: Iterable[SkillRecord]) -> tuple[SkillRecord, ...]:
    copies = tuple(
        SkillRecord.from_dict(skill.to_dict())
        for skill in sorted(skills, key=lambda item: (item.skill_id, item.version))
    )
    identifiers = [skill.skill_id for skill in copies]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("a frozen Skill snapshot may contain only one version per skill_id")
    return copies


def _ledger_snapshot_id(ledger: CombinationPosterior) -> str:
    return stable_id("ledger_snapshot", ledger.state_dict())


def _skill_snapshot_id(skills: Sequence[SkillRecord]) -> str:
    return stable_id("skill_snapshot", [skill.to_dict() for skill in skills])


def _ledger_copy(ledger: CombinationPosterior) -> CombinationPosterior:
    """Copy a ledger, including the valid zero-column initial state.

    ``LedgerState.to_dict`` represents a ``(0, 0)`` array as ``[]``.  Until a
    first-order level has been registered, a JSON round-trip therefore loses
    the array rank.  A deepcopy keeps the initial state exact without changing
    the upstream ledger implementation.
    """

    return copy.deepcopy(ledger)


def _ledger_from_json_state(value: Mapping[str, Any]) -> CombinationPosterior:
    payload = dict(value)
    if not payload.get("columns") and payload.get("Lambda") == []:
        payload["Lambda"] = np.zeros((0, 0), dtype=np.float64)
    return CombinationPosterior.from_state_dict(payload)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class StepRecord:
    """Section 3.3 sidecar for one progressive Canvas decision."""

    step_id: int | str
    trajectory_id: str
    snapshot_id: str
    key: DecisionKey
    candidates: Sequence[DecisionKey]
    ledger_readout: Mapping[str, Any]
    director_action: str
    surface: SurfaceSignal | None
    forked: bool = False
    audit: bool = False
    schema_version: str = LEDGER_EPOCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_EPOCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported StepRecord schema: {self.schema_version}")
        if isinstance(self.step_id, bool) or not isinstance(self.step_id, (int, str)):
            raise TypeError("step_id must be a non-negative integer or non-empty string")
        if isinstance(self.step_id, int):
            if self.step_id < 0:
                raise ValueError("integer step_id must be non-negative")
        else:
            object.__setattr__(self, "step_id", _non_empty(self.step_id, "step_id"))
        for name in ("trajectory_id", "snapshot_id"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if not isinstance(self.key, DecisionKey):
            raise TypeError("key must be DecisionKey")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, DecisionKey) for candidate in candidates):
            raise TypeError("candidates must contain only DecisionKey values")
        if any(
            candidate.task_family != self.key.task_family
            or candidate.stage != self.key.stage
            for candidate in candidates
        ):
            raise ValueError("candidate context must match the executed DecisionKey")
        for candidate in candidates:
            changed_field(self.key, candidate)
        object.__setattr__(self, "candidates", candidates)
        readout = _json_copy(dict(self.ledger_readout))
        object.__setattr__(self, "ledger_readout", MappingProxyType(readout))
        if self.director_action not in _DIRECTOR_ACTIONS:
            raise ValueError(
                "director_action must be continue, revise, or not_triggered"
            )
        if self.surface is not None and not isinstance(self.surface, SurfaceSignal):
            raise TypeError("surface must be SurfaceSignal or None")
        if self.surface is not None:
            expected_sensor_class = SurfaceSignal.for_key(
                self.surface.kind, self.surface.value, self.key
            ).sensor_class
            if self.surface.sensor_class != expected_sensor_class:
                raise ValueError(
                    "surface sensor_class does not match DecisionKey model relation"
                )
        if type(self.forked) is not bool or type(self.audit) is not bool:
            raise TypeError("forked and audit must be bool")
        if self.audit and not self.forked:
            raise ValueError("an audit step must be marked forked")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "step_id": self.step_id,
            "trajectory_id": self.trajectory_id,
            "snapshot_id": self.snapshot_id,
            "key": self.key.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "ledger_readout": _json_copy(dict(self.ledger_readout)),
            "director_action": self.director_action,
            "surface": None if self.surface is None else self.surface.to_dict(),
            "forked": self.forked,
            "audit": self.audit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepRecord":
        payload = dict(value)
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            step_id=payload["step_id"],
            trajectory_id=str(payload["trajectory_id"]),
            snapshot_id=str(payload["snapshot_id"]),
            key=DecisionKey.from_dict(payload["key"]),
            candidates=tuple(
                DecisionKey.from_dict(candidate)
                for candidate in payload.get("candidates", ())
            ),
            ledger_readout=dict(payload.get("ledger_readout", {})),
            director_action=str(payload["director_action"]),
            surface=(
                None
                if payload.get("surface") is None
                else SurfaceSignal.from_dict(payload["surface"])
            ),
            forked=payload.get("forked", False),
            audit=payload.get("audit", False),
        )


@dataclass(frozen=True)
class LedgerTrajectoryRecord:
    """Minimal natural/probe trajectory wrapper used only by the ledger plane."""

    trajectory_id: str
    problem_id: str
    task_split: str
    group_id: str
    condition_id: str
    policy_version: str
    steps: Sequence[StepRecord]
    is_natural: bool
    terminal_reward: int
    evaluator_valid: bool
    grpo_eligible: bool = False
    probe_id: str | None = None
    audit: bool = False
    schema_version: str = LEDGER_EPOCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_EPOCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported LedgerTrajectoryRecord schema: {self.schema_version}"
            )
        for name in (
            "trajectory_id",
            "problem_id",
            "group_id",
            "condition_id",
            "policy_version",
        ):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if self.task_split not in {"train", "validation", "test"}:
            raise ValueError("task_split must be train, validation, or test")
        steps = tuple(self.steps)
        if any(not isinstance(step, StepRecord) for step in steps):
            raise TypeError("steps must contain only StepRecord values")
        if any(step.trajectory_id != self.trajectory_id for step in steps):
            raise ValueError("every StepRecord must reference its parent trajectory")
        families = {step.key.task_family for step in steps}
        if len(families) > 1:
            raise ValueError("all steps in a trajectory must share one task_family")
        object.__setattr__(self, "steps", steps)
        for name in ("is_natural", "evaluator_valid", "grpo_eligible", "audit"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.terminal_reward not in (0, 1) or type(self.terminal_reward) is not int:
            raise ValueError("terminal_reward must be the binary evaluator outcome")
        if self.evaluator_valid and not steps:
            raise ValueError("an evaluator-valid ledger trajectory requires at least one step")
        if self.grpo_eligible and (
            not self.is_natural or not self.evaluator_valid
        ):
            raise ValueError("only evaluator-valid natural trajectories can be GRPO-eligible")
        if self.is_natural:
            if self.probe_id is not None or self.audit:
                raise ValueError("a natural trajectory cannot carry probe/audit identity")
        else:
            object.__setattr__(self, "probe_id", _non_empty(self.probe_id, "probe_id"))
        if self.audit and self.is_natural:
            raise ValueError("an audit branch cannot be natural")

    @property
    def forced_probe(self) -> bool:
        """Compatibility with the existing probe-partition assertions."""

        return not self.is_natural

    @property
    def task_family(self) -> str:
        if not self.steps:
            raise ValueError("task_family is unavailable without a StepRecord")
        return self.steps[0].key.task_family

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "problem_id": self.problem_id,
            "task_split": self.task_split,
            "group_id": self.group_id,
            "condition_id": self.condition_id,
            "policy_version": self.policy_version,
            "steps": [step.to_dict() for step in self.steps],
            "is_natural": self.is_natural,
            "terminal_reward": self.terminal_reward,
            "evaluator_valid": self.evaluator_valid,
            "grpo_eligible": self.grpo_eligible,
            "probe_id": self.probe_id,
            "audit": self.audit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerTrajectoryRecord":
        payload = dict(value)
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            trajectory_id=str(payload["trajectory_id"]),
            problem_id=str(payload["problem_id"]),
            task_split=str(payload["task_split"]),
            group_id=str(payload["group_id"]),
            condition_id=str(payload["condition_id"]),
            policy_version=str(payload["policy_version"]),
            steps=tuple(StepRecord.from_dict(step) for step in payload.get("steps", ())),
            is_natural=payload["is_natural"],
            terminal_reward=payload["terminal_reward"],
            evaluator_valid=payload["evaluator_valid"],
            grpo_eligible=payload.get("grpo_eligible", False),
            probe_id=payload.get("probe_id"),
            audit=payload.get("audit", False),
        )


@dataclass(frozen=True)
class FrozenEpochCondition:
    """Immutable ``(policy, ledger, Skill)`` binding shared by an epoch group."""

    epoch: int
    condition_id: str
    versions: VersionBundle
    ledger_snapshot_id: str
    skill_snapshot_id: str
    visible_skill_ids: Sequence[str] = field(default_factory=tuple)
    schema_version: str = LEDGER_EPOCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_EPOCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported frozen-condition schema: {self.schema_version}")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        for name in ("condition_id", "ledger_snapshot_id", "skill_snapshot_id"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if not isinstance(self.versions, VersionBundle):
            raise TypeError("versions must be VersionBundle")
        if self.versions.posterior != self.ledger_snapshot_id:
            raise ValueError("versions.posterior must bind the frozen ledger snapshot")
        if self.versions.skill_library != self.skill_snapshot_id:
            raise ValueError("versions.skill_library must bind the frozen Skill snapshot")
        identifiers = tuple(str(value) for value in self.visible_skill_ids)
        if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("visible_skill_ids must be unique non-empty identifiers")
        object.__setattr__(self, "visible_skill_ids", identifiers)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "condition_id": self.condition_id,
            "versions": self.versions.to_dict(),
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "skill_snapshot_id": self.skill_snapshot_id,
            "visible_skill_ids": list(self.visible_skill_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenEpochCondition":
        payload = dict(value)
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            epoch=int(payload["epoch"]),
            condition_id=str(payload["condition_id"]),
            versions=VersionBundle.from_dict(payload["versions"]),
            ledger_snapshot_id=str(payload["ledger_snapshot_id"]),
            skill_snapshot_id=str(payload["skill_snapshot_id"]),
            visible_skill_ids=tuple(payload.get("visible_skill_ids", ())),
        )


@dataclass(frozen=True)
class SkillTextSelection:
    """Top-three condition-matched active Skills for a Director observation."""

    skill_ids: tuple[str, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.skill_ids) != len(self.texts) or len(self.skill_ids) > 3:
            raise ValueError("Skill selection must contain at most three aligned entries")

    @property
    def text(self) -> str:
        return "\n".join(self.texts)

    def to_dict(self) -> dict[str, object]:
        return {"skill_ids": list(self.skill_ids), "texts": list(self.texts), "text": self.text}


@dataclass(frozen=True)
class EpochUpdateSummary:
    """Auditable counts proving the Section 6.4 routing decisions."""

    natural_trajectories: int
    intervention_trajectories: int
    invalid_trajectories: int
    baseline_updates: int
    sensor_updates: int
    contrast_probe_updates: int
    grpo_trajectory_ids: tuple[str, ...]
    excluded_intervention_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "natural_trajectories": self.natural_trajectories,
            "intervention_trajectories": self.intervention_trajectories,
            "invalid_trajectories": self.invalid_trajectories,
            "baseline_updates": self.baseline_updates,
            "sensor_updates": self.sensor_updates,
            "contrast_probe_updates": self.contrast_probe_updates,
            "grpo_trajectory_ids": list(self.grpo_trajectory_ids),
            "excluded_intervention_ids": list(self.excluded_intervention_ids),
        }


@dataclass(frozen=True)
class EpochSnapshotReceipt:
    """Receipt locating an independently recoverable ledger/Skill epoch snapshot."""

    receipt_id: str
    condition: FrozenEpochCondition
    ledger_file: str
    skill_file: str
    ledger_snapshot_id: str
    skill_snapshot_id: str
    created_at: str
    schema_version: str = LEDGER_EPOCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_EPOCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot receipt schema: {self.schema_version}")
        for name in ("receipt_id", "ledger_file", "skill_file", "created_at"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if Path(self.ledger_file).is_absolute() or Path(self.skill_file).is_absolute():
            raise ValueError("snapshot receipt paths must be relative to the receipt")
        if self.ledger_snapshot_id != self.condition.ledger_snapshot_id:
            raise ValueError("receipt ledger ID does not match its condition")
        if self.skill_snapshot_id != self.condition.skill_snapshot_id:
            raise ValueError("receipt Skill ID does not match its condition")

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "condition": self.condition.to_dict(),
            "ledger_file": self.ledger_file,
            "skill_file": self.skill_file,
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "skill_snapshot_id": self.skill_snapshot_id,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, object]:
        return {"receipt_id": self.receipt_id, **self.unsigned_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpochSnapshotReceipt":
        payload = dict(value)
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            receipt_id=str(payload["receipt_id"]),
            condition=FrozenEpochCondition.from_dict(payload["condition"]),
            ledger_file=str(payload["ledger_file"]),
            skill_file=str(payload["skill_file"]),
            ledger_snapshot_id=str(payload["ledger_snapshot_id"]),
            skill_snapshot_id=str(payload["skill_snapshot_id"]),
            created_at=str(payload["created_at"]),
        )


@dataclass(frozen=True)
class EpochTransition:
    """Completed ledger-plane transition; GRPO remains owned by the trainer."""

    next_epoch: "LedgerEpoch"
    summary: EpochUpdateSummary


class LedgerEpoch:
    """Frozen epoch read model plus the strict epoch-boundary update operation."""

    def __init__(
        self,
        *,
        ledger: CombinationPosterior,
        skills: Iterable[SkillRecord],
        condition: FrozenEpochCondition,
    ) -> None:
        if not isinstance(ledger, CombinationPosterior):
            raise TypeError("ledger must be CombinationPosterior")
        if ledger.epoch != condition.epoch:
            raise ValueError("ledger epoch does not match frozen condition")
        if ledger.policy_version != condition.versions.policy:
            raise ValueError("ledger policy version does not match frozen condition")
        copied_ledger = _ledger_copy(ledger)
        copied_skills = _skill_copies(skills)
        if _ledger_snapshot_id(copied_ledger) != condition.ledger_snapshot_id:
            raise ValueError("ledger contents do not match frozen condition")
        if _skill_snapshot_id(copied_skills) != condition.skill_snapshot_id:
            raise ValueError("Skill contents do not match frozen condition")
        visible = tuple(
            skill.skill_id
            for skill in copied_skills
            if skill.status is SkillStatus.ACTIVE
            and skill.activated_epoch is not None
            and skill.activated_epoch <= condition.epoch
        )
        if visible != tuple(condition.visible_skill_ids):
            raise ValueError("visible Skill IDs do not match frozen epoch snapshot")
        self._ledger = copied_ledger
        self._skills = copied_skills
        self.condition = condition

    @classmethod
    def freeze(
        cls,
        ledger: CombinationPosterior,
        skills: Iterable[SkillRecord],
        versions: VersionBundle,
    ) -> "LedgerEpoch":
        """Copy and bind the policy, ledger, and Skill state for one epoch."""

        if not isinstance(ledger, CombinationPosterior):
            raise TypeError("ledger must be CombinationPosterior")
        if not isinstance(versions, VersionBundle):
            raise TypeError("versions must be VersionBundle")
        if ledger.policy_version != versions.policy:
            raise ValueError("ledger and VersionBundle policy versions must match")
        copied_ledger = _ledger_copy(ledger)
        copied_skills = _skill_copies(skills)
        ledger_id = _ledger_snapshot_id(copied_ledger)
        skill_id = _skill_snapshot_id(copied_skills)
        bound_versions = replace(
            versions,
            posterior=ledger_id,
            skill_library=skill_id,
        )
        visible = tuple(
            skill.skill_id
            for skill in copied_skills
            if skill.status is SkillStatus.ACTIVE
            and skill.activated_epoch is not None
            and skill.activated_epoch <= copied_ledger.epoch
        )
        condition_payload = {
            "epoch": copied_ledger.epoch,
            "versions": bound_versions.to_dict(),
            "ledger_snapshot_id": ledger_id,
            "skill_snapshot_id": skill_id,
            "visible_skill_ids": list(visible),
        }
        condition = FrozenEpochCondition(
            epoch=copied_ledger.epoch,
            condition_id=stable_id("ledger_condition", condition_payload),
            versions=bound_versions,
            ledger_snapshot_id=ledger_id,
            skill_snapshot_id=skill_id,
            visible_skill_ids=visible,
        )
        return cls(ledger=copied_ledger, skills=copied_skills, condition=condition)

    @property
    def ledger(self) -> CombinationPosterior:
        """Return a detached ledger so callers cannot mutate the frozen snapshot."""

        return _ledger_copy(self._ledger)

    @property
    def skills(self) -> tuple[SkillRecord, ...]:
        return _skill_copies(self._skills)

    def pre_execution(
        self,
        *,
        step_id: int | str,
        current: DecisionKey,
        candidates: Sequence[DecisionKey],
        config: LatentLossConfig = LatentLossConfig(),
    ) -> PreExecutionReadout:
        """Read the frozen ledger before an edit; no epoch state is mutated."""

        return pre_execution_readout(
            self.ledger,
            step_id=step_id,
            current=current,
            candidates=candidates,
            config=config,
        )

    def post_execution(
        self,
        *,
        step_id: int | str,
        prefix_before: Sequence[DecisionKey],
        current: DecisionKey,
        candidates: Sequence[DecisionKey],
        surface: SurfaceSignal | None,
        remaining_rounds: int,
        rng: np.random.Generator,
        config: LatentLossConfig = LatentLossConfig(),
    ) -> PostExecutionReadout:
        """Read posterior latent risk after execution; no reward is changed."""

        return post_execution_latent_loss(
            self.ledger,
            step_id=step_id,
            prefix_before=prefix_before,
            current=current,
            candidates=candidates,
            surface=surface,
            remaining_rounds=remaining_rounds,
            rng=rng,
            config=config,
        )

    def select_skill_texts(
        self,
        *,
        task_family: str,
        role_cluster: str,
        stage: str,
        available_models: Sequence[str],
        tags: Sequence[str] = (),
    ) -> SkillTextSelection:
        """Select up to three applicable active Skills from this frozen snapshot."""

        task_family = _non_empty(task_family, "task_family")
        role_cluster = _non_empty(role_cluster, "role_cluster")
        stage = _non_empty(stage, "stage")
        eligible_ids = set(self.condition.visible_skill_ids)
        role_matched = tuple(
            skill
            for skill in self._skills
            if skill.skill_id in eligible_ids
            and skill.condition.get("role_cluster") in ("*", role_cluster)
        )
        selected = SkillRetriever(top_k=3).retrieve(
            role_matched,
            SkillQuery(
                task_family=task_family,
                graph_stage=stage,
                tags=tuple(tags),
                available_models=tuple(available_models),
                current_epoch=self.condition.epoch,
            ),
            self.condition.versions,
        )
        texts = tuple(self._render_skill(skill) for skill in selected)
        return SkillTextSelection(
            skill_ids=tuple(skill.skill_id for skill in selected),
            texts=texts,
        )

    @staticmethod
    def _render_skill(skill: SkillRecord) -> str:
        if skill.readable_text.strip():
            detail = skill.readable_text.strip()
        else:
            detail = f"prefer {dict(skill.action)}"
        evidence = skill.evidence
        failure_scope = ", ".join(skill.failure_scope) if skill.failure_scope else "none"
        return (
            f"[Skill] {dict(skill.condition)}: {detail}; expected gain "
            f"{evidence.paired_effect_mean:+.2f} "
            f"[{evidence.calibrated_lower:.2f},{evidence.calibrated_upper:.2f}], "
            f"n={evidence.effective_pairs}. Not applicable to: {failure_scope}."
        )

    def close(
        self,
        *,
        trajectories: Iterable[LedgerTrajectoryRecord],
        probes: Iterable[ProbeRecord],
        next_policy_version: str,
        next_skills: Iterable[SkillRecord],
        heldout_coverage: float | None = None,
    ) -> EpochTransition:
        """Apply the Section 6.4 routing rules and return a frozen next epoch.

        This method does not consume, alter, or derive GRPO rewards.  The
        ``grpo_trajectory_ids`` field merely mirrors caller-validated eligibility
        so the trainer can assert that no intervention branch crossed planes.
        """

        next_policy_version = _non_empty(next_policy_version, "next_policy_version")
        if next_policy_version == self.condition.versions.policy:
            raise ValueError("next_policy_version must identify the optimizer-updated policy")
        trajectories = tuple(trajectories)
        probes = tuple(probes)
        if len({record.trajectory_id for record in trajectories}) != len(trajectories):
            raise ValueError("trajectory IDs must be unique within an epoch")

        for record in trajectories:
            if not isinstance(record, LedgerTrajectoryRecord):
                raise TypeError("trajectories must contain LedgerTrajectoryRecord values")
            if record.condition_id != self.condition.condition_id:
                raise ValueError("trajectory condition does not match the frozen epoch")
            if record.policy_version != self.condition.versions.policy:
                raise ValueError("trajectory policy does not match the frozen epoch")
            if record.task_split != "train":
                raise ValueError("epoch ledger updates accept only train trajectories")

        intervention_by_id = {
            record.trajectory_id: record
            for record in trajectories
            if not record.is_natural
        }
        expected_intervention_ids: set[str] = set()
        for probe in probes:
            if not isinstance(probe, ProbeRecord):
                raise TypeError("probes must contain ProbeRecord values")
            validation = validate_probe_record(probe)
            if probe.policy_version != self.condition.versions.policy:
                raise ValueError("probe policy does not match the frozen epoch")
            if probe.evaluator_version != self.condition.versions.evaluator:
                raise ValueError("probe evaluator does not match the frozen epoch")
            if probe.feature_schema_version != self.condition.versions.feature_schema:
                raise ValueError("probe feature schema does not match the frozen epoch")
            if probe.task_split != "train":
                raise ValueError("held-out probes are reserved for calibration")
            branch_records = tuple(
                intervention_by_id[branch_id]
                for branch_id in validation.branch_ids
                if branch_id in intervention_by_id
            )
            if len(branch_records) != len(validation.branch_ids):
                raise ValueError("probe is missing one or more intervention trajectories")
            assert_probe_data_partition(probe, branch_records)
            if any(record.probe_id != probe.probe_id for record in branch_records):
                raise ValueError("probe branch sidecar references the wrong probe_id")
            if any(record.audit != validation.is_audit for record in branch_records):
                raise ValueError("probe branch audit flag is inconsistent")
            if any(not record.evaluator_valid for record in branch_records):
                raise ValueError("a ProbeRecord cannot include evaluator-invalid branches")
            for branch in branch_records:
                fields = branch.trajectory_id.rsplit(":", 2)
                if len(fields) != 3 or fields[0] != probe.probe_id:
                    raise ValueError("probe branch trajectory ID is malformed")
                arm, repeat_text = fields[1], fields[2]
                try:
                    repeat_index = int(repeat_text)
                except ValueError as error:
                    raise ValueError("probe branch repeat index is malformed") from error
                expected_returns = (
                    probe.incumbent_returns if arm == "keep" else probe.candidate_returns
                )
                if arm not in {"keep", "switch"} or not 0 <= repeat_index < len(
                    expected_returns
                ):
                    raise ValueError("probe branch arm or repeat index is malformed")
                if branch.terminal_reward != int(expected_returns[repeat_index]):
                    raise ValueError(
                        "probe branch terminal outcome does not match ProbeRecord"
                    )
            overlap = expected_intervention_ids.intersection(validation.branch_ids)
            if overlap:
                raise ValueError("an intervention trajectory belongs to multiple probes")
            expected_intervention_ids.update(validation.branch_ids)
        if set(intervention_by_id) != expected_intervention_ids:
            raise ValueError("every intervention trajectory must belong to one ProbeRecord")

        updated = self.ledger
        for probe in probes:
            updated.update_probe(
                DecisionKey.from_dict(probe.incumbent_action),
                DecisionKey.from_dict(probe.candidate_action),
                probe.incumbent_returns,
                probe.candidate_returns,
            )

        baseline_updates = 0
        sensor_updates = 0
        for record in trajectories:
            if not record.evaluator_valid:
                continue
            surfaces = tuple(
                step.surface for step in record.steps if step.surface is not None
            )
            updated.update_trajectory(
                record.task_family,
                tuple(step.key for step in record.steps),
                surfaces,
                record.terminal_reward,
            )
            baseline_updates += 1
            sensor_updates += len(surfaces)

        updated.empirical_bayes_update()
        updated.refresh_policy(
            next_policy_version,
            heldout_coverage=heldout_coverage,
        )
        next_versions = replace(
            self.condition.versions,
            policy=next_policy_version,
            posterior="pending-next-ledger",
            skill_library="pending-next-skill-snapshot",
        )
        next_epoch = LedgerEpoch.freeze(updated, next_skills, next_versions)
        natural = tuple(record for record in trajectories if record.is_natural)
        intervention = tuple(record for record in trajectories if not record.is_natural)
        summary = EpochUpdateSummary(
            natural_trajectories=len(natural),
            intervention_trajectories=len(intervention),
            invalid_trajectories=sum(not record.evaluator_valid for record in trajectories),
            baseline_updates=baseline_updates,
            sensor_updates=sensor_updates,
            contrast_probe_updates=len(probes),
            grpo_trajectory_ids=tuple(
                record.trajectory_id for record in natural if record.grpo_eligible
            ),
            excluded_intervention_ids=tuple(
                sorted(record.trajectory_id for record in intervention)
            ),
        )
        return EpochTransition(next_epoch=next_epoch, summary=summary)

    def save(self, directory: str | os.PathLike[str]) -> EpochSnapshotReceipt:
        """Persist this frozen epoch as ledger, Skill, and receipt JSON files."""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        ledger_file = "ledger.json"
        skill_file = "skills.json"
        _atomic_json(destination / ledger_file, self._ledger.state_dict())
        _atomic_json(
            destination / skill_file,
            {
                "schema_version": LEDGER_EPOCH_SCHEMA_VERSION,
                "skills": [skill.to_dict() for skill in self._skills],
            },
        )
        created_at = utc_now()
        unsigned = {
            "schema_version": LEDGER_EPOCH_SCHEMA_VERSION,
            "condition": self.condition.to_dict(),
            "ledger_file": ledger_file,
            "skill_file": skill_file,
            "ledger_snapshot_id": self.condition.ledger_snapshot_id,
            "skill_snapshot_id": self.condition.skill_snapshot_id,
            "created_at": created_at,
        }
        receipt = EpochSnapshotReceipt(
            receipt_id=stable_id("ledger_epoch_receipt", unsigned),
            condition=self.condition,
            ledger_file=ledger_file,
            skill_file=skill_file,
            ledger_snapshot_id=self.condition.ledger_snapshot_id,
            skill_snapshot_id=self.condition.skill_snapshot_id,
            created_at=created_at,
        )
        _atomic_json(destination / "receipt.json", receipt.to_dict())
        return receipt

    @classmethod
    def load(cls, receipt_path: str | os.PathLike[str]) -> "LedgerEpoch":
        """Load a snapshot written by :meth:`save`."""

        path = Path(receipt_path)
        if path.is_dir():
            path = path / "receipt.json"
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("epoch receipt must be a JSON object")
        receipt = EpochSnapshotReceipt.from_dict(value)
        if receipt.receipt_id != stable_id(
            "ledger_epoch_receipt", receipt.unsigned_dict()
        ):
            raise ValueError("epoch receipt contents do not match receipt_id")
        with (path.parent / receipt.ledger_file).open("r", encoding="utf-8") as handle:
            ledger_payload = json.load(handle)
        with (path.parent / receipt.skill_file).open("r", encoding="utf-8") as handle:
            skill_payload = json.load(handle)
        if not isinstance(skill_payload, Mapping) or skill_payload.get(
            "schema_version"
        ) != LEDGER_EPOCH_SCHEMA_VERSION:
            raise ValueError("unsupported Skill snapshot schema")
        ledger = _ledger_from_json_state(ledger_payload)
        skills = tuple(
            SkillRecord.from_dict(item) for item in skill_payload.get("skills", ())
        )
        return cls(ledger=ledger, skills=skills, condition=receipt.condition)


def combine_step_readouts(
    pre: PreExecutionReadout,
    post: PostExecutionReadout | None,
    skills: SkillTextSelection | None = None,
) -> dict[str, object]:
    """Build the JSON sidecar inserted into ``StepRecord.ledger_readout``."""

    return {
        "pre": pre.to_dict(),
        "post": None if post is None else post.to_dict(),
        "skills": None if skills is None else skills.to_dict(),
    }


__all__ = [
    "EpochSnapshotReceipt",
    "EpochTransition",
    "EpochUpdateSummary",
    "FrozenEpochCondition",
    "LEDGER_EPOCH_SCHEMA_VERSION",
    "LedgerEpoch",
    "LedgerTrajectoryRecord",
    "SkillTextSelection",
    "StepRecord",
    "combine_step_readouts",
]
