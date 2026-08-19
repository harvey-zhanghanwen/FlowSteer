"""Structured Skills: conditions, actions, evidence, versions, and limits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence

from ..records import utc_now
from ..versioning import VersionBundle


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Skill fields must be JSON-compatible, got {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class SkillEvidence:
    baseline: str
    paired_effect_mean: float
    calibrated_lower: float
    calibrated_upper: float
    effective_pairs: int
    independent_problem_ids: Sequence[str]
    discovery_problem_ids: Sequence[str]
    validation_problem_ids: Sequence[str]
    validation_splits: Sequence[str]
    heldout_task_families: Sequence[str]
    empirical_coverage: float
    harm_probability: float
    slice_effects: Mapping[str, float] = field(default_factory=dict)
    evidence_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        numeric = (
            self.paired_effect_mean,
            self.calibrated_lower,
            self.calibrated_upper,
            self.empirical_coverage,
            self.harm_probability,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Skill evidence statistics must be finite")
        if self.calibrated_lower > self.calibrated_upper:
            raise ValueError("calibrated interval is reversed")
        if type(self.effective_pairs) is not int or self.effective_pairs < 0:
            raise ValueError("effective_pairs must be non-negative")
        if not 0 <= self.empirical_coverage <= 1:
            raise ValueError("empirical_coverage must be in [0, 1]")
        if not 0 <= self.harm_probability <= 1:
            raise ValueError("harm_probability must be in [0, 1]")
        if "test" in self.validation_splits:
            raise ValueError("test data cannot be used as Skill evidence")
        sequence_fields = (
            "independent_problem_ids",
            "discovery_problem_ids",
            "validation_problem_ids",
            "validation_splits",
            "heldout_task_families",
            "evidence_ids",
        )
        for name in sequence_fields:
            values = tuple(str(value) for value in getattr(self, name))
            if any(not value for value in values):
                raise ValueError(f"{name} cannot contain empty values")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        independent = set(self.independent_problem_ids)
        validation = set(self.validation_problem_ids)
        discovery = set(self.discovery_problem_ids)
        if not independent.issubset(validation):
            raise ValueError("independent problems must be held-out validation problems")
        if discovery & validation:
            raise ValueError("discovery and validation problem IDs must be disjoint")
        frozen_slices = {str(key): float(value) for key, value in self.slice_effects.items()}
        if not all(math.isfinite(value) for value in frozen_slices.values()):
            raise ValueError("slice effects must be finite")
        object.__setattr__(self, "slice_effects", MappingProxyType(frozen_slices))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "paired_effect_mean": self.paired_effect_mean,
            "calibrated_lower": self.calibrated_lower,
            "calibrated_upper": self.calibrated_upper,
            "effective_pairs": self.effective_pairs,
            "independent_problem_ids": list(self.independent_problem_ids),
            "discovery_problem_ids": list(self.discovery_problem_ids),
            "validation_problem_ids": list(self.validation_problem_ids),
            "validation_splits": list(self.validation_splits),
            "heldout_task_families": list(self.heldout_task_families),
            "empirical_coverage": self.empirical_coverage,
            "harm_probability": self.harm_probability,
            "slice_effects": dict(self.slice_effects),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillEvidence":
        data = dict(value)
        return cls(**data)


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    version: int
    status: SkillStatus
    condition: Mapping[str, Any]
    action: Mapping[str, Any]
    evidence: SkillEvidence
    versions: VersionBundle
    failure_scope: Sequence[str] = field(default_factory=tuple)
    readable_text: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_epoch: int = 0
    eligible_epoch: int = 1
    activated_epoch: Optional[int] = None
    suspended_reason: Optional[str] = None
    gate_config: Mapping[str, Any] = field(default_factory=dict)
    gate_receipt: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("skill_id must be non-empty")
        if self.version < 1:
            raise ValueError("Skill version must be >= 1")
        if self.created_epoch < 0:
            raise ValueError("created_epoch must be non-negative")
        if self.eligible_epoch <= self.created_epoch:
            raise ValueError("a new Skill can only become visible in a later epoch")
        if not self.condition:
            raise ValueError("Skill condition must be structured and non-empty")
        if not self.action:
            raise ValueError("Skill action must be structured and non-empty")
        condition = _freeze(self.condition)
        action = _freeze(self.action)
        provenance = _freeze(self.provenance)
        gate_config = _freeze(self.gate_config)
        failure_scope = tuple(str(value) for value in self.failure_scope)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "gate_config", gate_config)
        object.__setattr__(self, "failure_scope", failure_scope)
        for key in ("task_family", "graph_stage"):
            value = condition.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Skill condition requires explicit {key}")
        required_tools = condition.get("required_tools", ())
        if not isinstance(required_tools, tuple):
            raise TypeError("Skill condition required_tools must be a sequence")
        if any(
            not isinstance(tool_id, str)
            or not tool_id.strip()
            or tool_id != tool_id.strip()
            for tool_id in required_tools
        ):
            raise ValueError(
                "Skill condition required_tools must contain non-empty tool IDs"
            )
        if tuple(sorted(set(required_tools))) != required_tools:
            raise ValueError(
                "Skill condition required_tools must be sorted and unique"
            )
        if not any(key in action for key in ("model_id", "relation", "instruction", "instruction_template")):
            raise ValueError("Skill action must recommend a model, relation, or bounded instruction")
        model_id = action.get("model_id")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            raise ValueError("Skill action model_id must be non-empty when present")
        instruction = action.get("instruction", action.get("instruction_template", ""))
        if instruction and (not isinstance(instruction, str) or len(instruction) > 2000):
            raise ValueError("Skill instruction must be text no longer than 2000 characters")
        if self.status is SkillStatus.CANDIDATE:
            if self.activated_epoch is not None or self.gate_receipt is not None or self.gate_config:
                raise ValueError("candidate Skill cannot carry activation state")
        elif self.status is SkillStatus.ACTIVE:
            if self.activated_epoch is None or self.activated_epoch < self.eligible_epoch:
                raise ValueError("active Skill requires a valid delayed activation epoch")
            if not self.gate_receipt or not self.gate_config:
                raise ValueError("active Skill requires a deterministic gate receipt")
        elif self.status in {SkillStatus.SUSPENDED, SkillStatus.RETIRED}:
            if not self.suspended_reason:
                raise ValueError(f"{self.status.value} Skill requires a reason")

    def _with_status(
        self,
        status: SkillStatus,
        *,
        epoch: Optional[int] = None,
        reason: Optional[str] = None,
        gate_config: Optional[Mapping[str, Any]] = None,
        gate_receipt: Optional[str] = None,
    ) -> "SkillRecord":
        return replace(
            self,
            status=status,
            activated_epoch=epoch if status is SkillStatus.ACTIVE else self.activated_epoch,
            suspended_reason=reason if status in {SkillStatus.SUSPENDED, SkillStatus.RETIRED} else None,
            gate_config=gate_config if status is SkillStatus.ACTIVE else self.gate_config,
            gate_receipt=gate_receipt if status is SkillStatus.ACTIVE else self.gate_receipt,
            updated_at=utc_now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "status": self.status.value,
            "condition": _thaw(self.condition),
            "action": _thaw(self.action),
            "evidence": self.evidence.to_dict(),
            "versions": self.versions.to_dict(),
            "failure_scope": list(self.failure_scope),
            "readable_text": self.readable_text,
            "provenance": _thaw(self.provenance),
            "created_epoch": self.created_epoch,
            "eligible_epoch": self.eligible_epoch,
            "activated_epoch": self.activated_epoch,
            "suspended_reason": self.suspended_reason,
            "gate_config": _thaw(self.gate_config),
            "gate_receipt": self.gate_receipt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillRecord":
        data = dict(value)
        data["status"] = SkillStatus(data["status"])
        data["evidence"] = SkillEvidence.from_dict(data["evidence"])
        data["versions"] = VersionBundle.from_dict(data["versions"])
        return cls(**data)
