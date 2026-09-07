"""Supplemental admission receipt for evidence-producing long training.

The first real optimizer-step closure is an immutable Phase-E acceptance
artifact.  It must not be edited after a later Skill producer or held-out
calibration implementation becomes available.  This module therefore builds
one separately versioned, fail-closed admission receipt that binds the
immutable closure to the committed step, recoverable run state, next ledger
epoch, and Skill evidence decision.

This is a project control-plane adapter.  FlowSteer supplies the progressive
Canvas / terminal-reward GRPO boundary, while the existing Skill lifecycle and
adapter publication path follow the SkillFlow implementation shape.  Neither
upstream project defines this post-closure admission receipt.

The receipt authorizes only evidence-producing training.  It never declares
the Skill publication phase or ordinary deployment/evaluation phase complete,
never changes a reward, and rejects TTB.  If no discovery rule qualifies, a
completed ``complete_no_qualified_candidate`` producer decision is sufficient
to continue gathering evidence.  Once a discovery candidate exists, a real,
version-bound held-out paired-calibration receipt is mandatory, and a pending
Skill producer remains a blocker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..skills.runtime_producer import (
    SKILL_RUNTIME_UPDATE_SCHEMA,
    SkillRuntimeUpdateReceipt,
)
from .heldout_calibration import (
    HeldoutPairedCalibrationReceipt,
    validate_heldout_paired_calibration,
)


SUPPLEMENTAL_LONG_TRAINING_ADMISSION_SCHEMA = (
    "flowsteer.supplemental-long-training-admission.v1"
)
EVIDENCE_PRODUCING_TRAINING_SCOPE = "evidence-producing-training-only"
_ONE_STEP_CLOSURE_SCHEMA = "flowsteer.dynamic-grpo-one-step-closure.v1"
_STEP_MANIFEST_SCHEMA = "flowsteer.hotpotqa.grpo_step_manifest.v1"
_RUN_STATE_SCHEMA = "flowsteer.hotpotqa.grpo_run_state.v1"
_LEDGER_EPOCH_SCHEMA = "flowsteer.ledger-epoch.v1"
_OBJECTIVE = "action_masked_one_pass_grpo"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _nested(value: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _non_empty(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _skill_receipt_dict(
    value: SkillRuntimeUpdateReceipt | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(value, SkillRuntimeUpdateReceipt):
        return value.to_dict()
    return _mapping(value, "skill_runtime_receipt")


def _heldout_receipt(
    value: HeldoutPairedCalibrationReceipt | Mapping[str, Any] | None,
) -> HeldoutPairedCalibrationReceipt | None:
    if value is None:
        return None
    if isinstance(value, HeldoutPairedCalibrationReceipt):
        return value
    serialized = _mapping(value, "heldout_calibration_receipt")
    if serialized.get("heldout_calibration_gate_passed") is not True:
        raise ValueError("held-out calibration gate is not passed")
    if serialized.get("does_not_authorize_training_by_itself") is not True:
        raise ValueError("held-out receipt has invalid authorization scope")
    return HeldoutPairedCalibrationReceipt.from_dict(serialized)


@dataclass(frozen=True)
class SupplementalLongTrainingAdmissionReceipt:
    """JSON-serializable result of the post-one-step admission checks."""

    status: str
    long_training_authorized: bool
    behavior_policy_version: str
    updated_policy_version: str
    ledger_epoch: int
    ledger_condition_id: str
    ledger_snapshot_id: str
    skill_snapshot_id: str
    skill_runtime_status: str
    heldout_calibration_status: str
    checks: Mapping[str, bool]
    blockers: tuple[str, ...]
    evidence_references: Mapping[str, str]
    created_at: str
    scope: str = EVIDENCE_PRODUCING_TRAINING_SCOPE
    objective: str = _OBJECTIVE
    ttb_enabled: bool = False
    phase4_complete: bool = False
    phase5_complete: bool = False
    schema_version: str = SUPPLEMENTAL_LONG_TRAINING_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SUPPLEMENTAL_LONG_TRAINING_ADMISSION_SCHEMA:
            raise ValueError(
                f"unsupported supplemental admission schema: {self.schema_version}"
            )
        if self.status not in {"authorized", "blocked"}:
            raise ValueError("status must be authorized or blocked")
        if type(self.long_training_authorized) is not bool:
            raise TypeError("long_training_authorized must be bool")
        if self.long_training_authorized != (self.status == "authorized"):
            raise ValueError("status and long_training_authorized disagree")
        if self.scope != EVIDENCE_PRODUCING_TRAINING_SCOPE:
            raise ValueError("supplemental admission has an invalid scope")
        if self.objective != _OBJECTIVE:
            raise ValueError("supplemental admission must use Action-Masked One-Pass GRPO")
        if self.ttb_enabled:
            raise ValueError("TTB must remain disabled")
        if self.phase4_complete or self.phase5_complete:
            raise ValueError("this receipt cannot declare Phase 4 or Phase 5 complete")
        if isinstance(self.ledger_epoch, bool) or not isinstance(self.ledger_epoch, int):
            raise TypeError("ledger_epoch must be an integer")
        if self.ledger_epoch < 0:
            raise ValueError("ledger_epoch must be non-negative")
        for name in (
            "behavior_policy_version",
            "updated_policy_version",
            "ledger_condition_id",
            "ledger_snapshot_id",
            "skill_snapshot_id",
            "skill_runtime_status",
            "heldout_calibration_status",
            "created_at",
        ):
            if not _non_empty(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty string")
        checks = {str(key): value for key, value in self.checks.items()}
        if not checks or any(type(value) is not bool for value in checks.values()):
            raise TypeError("checks must be a non-empty string-to-bool mapping")
        blockers = tuple(str(item) for item in self.blockers)
        expected_blockers = tuple(key for key, passed in checks.items() if not passed)
        if blockers != expected_blockers:
            raise ValueError("blockers must exactly enumerate failed checks")
        if self.long_training_authorized != all(checks.values()):
            raise ValueError("authorization must equal the conjunction of all checks")
        references = {str(key): str(value) for key, value in self.evidence_references.items()}
        if any(not key or not value for key, value in references.items()):
            raise ValueError("evidence reference keys and values must be non-empty")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "evidence_references", MappingProxyType(references))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "scope": self.scope,
            "long_training_authorized": self.long_training_authorized,
            "objective": self.objective,
            "ttb_enabled": self.ttb_enabled,
            "phase4_complete": self.phase4_complete,
            "phase5_complete": self.phase5_complete,
            "behavior_policy_version": self.behavior_policy_version,
            "updated_policy_version": self.updated_policy_version,
            "ledger_epoch": self.ledger_epoch,
            "ledger_condition_id": self.ledger_condition_id,
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "skill_snapshot_id": self.skill_snapshot_id,
            "skill_runtime_status": self.skill_runtime_status,
            "heldout_calibration_status": self.heldout_calibration_status,
            "checks": dict(self.checks),
            "blockers": list(self.blockers),
            "evidence_references": dict(self.evidence_references),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SupplementalLongTrainingAdmissionReceipt":
        return cls(
            schema_version=str(value["schema_version"]),
            status=str(value["status"]),
            scope=str(value["scope"]),
            long_training_authorized=value["long_training_authorized"],
            objective=str(value["objective"]),
            ttb_enabled=value["ttb_enabled"],
            phase4_complete=value["phase4_complete"],
            phase5_complete=value["phase5_complete"],
            behavior_policy_version=str(value["behavior_policy_version"]),
            updated_policy_version=str(value["updated_policy_version"]),
            ledger_epoch=int(value["ledger_epoch"]),
            ledger_condition_id=str(value["ledger_condition_id"]),
            ledger_snapshot_id=str(value["ledger_snapshot_id"]),
            skill_snapshot_id=str(value["skill_snapshot_id"]),
            skill_runtime_status=str(value["skill_runtime_status"]),
            heldout_calibration_status=str(value["heldout_calibration_status"]),
            checks=dict(value["checks"]),
            blockers=tuple(str(item) for item in value["blockers"]),
            evidence_references={
                str(key): str(item)
                for key, item in dict(value.get("evidence_references", {})).items()
            },
            created_at=str(value["created_at"]),
        )


def build_supplemental_long_training_admission(
    *,
    one_step_closure: Mapping[str, Any],
    run_state: Mapping[str, Any],
    step_manifest: Mapping[str, Any],
    ledger_epoch_receipt: Mapping[str, Any],
    skill_runtime_receipt: SkillRuntimeUpdateReceipt | Mapping[str, Any],
    heldout_calibration_receipt: (
        HeldoutPairedCalibrationReceipt | Mapping[str, Any] | None
    ) = None,
    evidence_references: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> SupplementalLongTrainingAdmissionReceipt:
    """Bind immutable one-step evidence to a fail-closed continuation decision."""

    closure = _mapping(one_step_closure, "one_step_closure")
    state = _mapping(run_state, "run_state")
    step = _mapping(step_manifest, "step_manifest")
    ledger_receipt = _mapping(ledger_epoch_receipt, "ledger_epoch_receipt")
    skill = _skill_receipt_dict(skill_runtime_receipt)

    behavior_policy = _non_empty(closure.get("behavior_policy_version"))
    updated_policy = _non_empty(closure.get("updated_policy_version"))
    condition = _mapping(ledger_receipt.get("condition", {}), "ledger condition")
    versions = _mapping(condition.get("versions", {}), "ledger versions")
    ledger_epoch_value = condition.get("epoch", -1)
    ledger_epoch = (
        ledger_epoch_value
        if isinstance(ledger_epoch_value, int) and not isinstance(ledger_epoch_value, bool)
        else -1
    )
    condition_id = _non_empty(condition.get("condition_id"))
    ledger_snapshot_id = _non_empty(condition.get("ledger_snapshot_id"))
    skill_snapshot_id = _non_empty(condition.get("skill_snapshot_id"))

    training = _mapping(step.get("training", {}), "step training")
    sync = _mapping(step.get("policy_sync", {}), "step policy_sync")
    metrics = _mapping(step.get("metrics", {}), "step metrics")
    phase_acceptance = _mapping(
        state.get("dynamic_phase_e_acceptance", {}), "phase-E acceptance"
    )
    phase_e = _mapping(phase_acceptance.get("phase_e", {}), "Phase-E receipt")
    phase_checks = _mapping(phase_e.get("checks", {}), "Phase-E checks")
    routing = _mapping(
        _nested(phase_e, "metrics", "routing_assertions", default={}),
        "routing assertions",
    )

    checkpoint_path = _non_empty(training.get("checkpoint_dir"))
    training_state_path = _non_empty(training.get("training_state_checkpoint"))
    optimizer_state_path = _non_empty(training.get("optimizer_state_checkpoint"))

    checks: dict[str, bool] = {}
    checks["immutable_one_step_closure_passed"] = bool(
        closure.get("schema_version") == _ONE_STEP_CLOSURE_SCHEMA
        and closure.get("status") == "passed"
        and closure.get("long_training_authorized") is False
    )
    checks["committed_real_optimizer_step"] = bool(
        step.get("schema_version") == _STEP_MANIFEST_SCHEMA
        and step.get("status") == "committed"
        and step.get("optimizer_step") == 1
        and closure.get("optimizer_updates") == 1
        and state.get("optimizer_updates_completed") == 1
        and training.get("optimizer_updates") == 1
        and _positive_integer(training.get("trained_trajectories"))
    )
    checks["action_masked_one_pass_grpo"] = bool(
        closure.get("objective") == _OBJECTIVE
    )
    checks["nonzero_gradient"] = bool(
        closure.get("nonzero_gradient") is True
        and _positive_finite(training.get("grad_norm"))
        and _positive_finite(metrics.get("train/grad_norm"))
    )
    checks["nonzero_lora_update"] = bool(
        closure.get("nonzero_lora_update") is True
        and _positive_finite(training.get("trainable_update_l2"))
        and _positive_finite(metrics.get("train/lora_update_l2"))
    )
    checks["checkpoint_recoverable"] = bool(
        closure.get("checkpoint_recoverable") is True
        and training.get("checkpoint_recoverable") is True
        and training.get("training_state_saved") is True
        and training.get("optimizer_state_saved") is True
        and training.get("scheduler_state_saved") is True
        and training.get("rng_state_saved") is True
        and checkpoint_path
        and training_state_path
        and optimizer_state_path
        and state.get("behavior_adapter_checkpoint") == checkpoint_path
        and state.get("training_state_checkpoint") == training_state_path
        and state.get("optimizer_state_checkpoint") == optimizer_state_path
    )
    checks["natural_only_same_problem_same_condition_grpo"] = bool(
        phase_checks.get("natural_only_grpo") is True
        and phase_checks.get("same_condition_groups") is True
        and phase_checks.get("unique_question_count") is True
        and phase_checks.get("probe_and_audit_excluded_from_grpo") is True
        and routing.get("natural_only_grpo") is True
        and routing.get("probe_and_audit_excluded_from_grpo") is True
        and _positive_integer(training.get("exact_groups"))
        and training.get("record_eligible_trajectories")
        == training.get("input_trajectories")
    )
    checks["auxiliary_signals_excluded_from_grpo_reward"] = bool(
        routing.get("comparison_fit_source")
        == "same_snapshot_paired_probes_only"
        and routing.get("latent_risk_reward_contribution") == 0
        and routing.get("skill_reward_contribution") == 0
        and skill.get("grpo_reward_contribution") == 0.0
    )
    checks["ttb_disabled"] = bool(
        closure.get("ttb_enabled") is False
        and phase_acceptance.get("ttb_enabled") is False
        and skill.get("ttb_enabled") is False
    )

    policy_values = {
        updated_policy,
        _non_empty(state.get("behavior_policy_version")),
        _non_empty(step.get("candidate_policy_version")),
        _non_empty(training.get("updated_policy_version")),
        _non_empty(sync.get("candidate_policy_version")),
        _non_empty(sync.get("new_policy_version")),
        _non_empty(sync.get("route_policy_version")),
        _non_empty(versions.get("policy")),
        _non_empty(skill.get("publication_policy_version")),
    }
    checks["updated_policy_binding"] = bool(
        updated_policy and policy_values == {updated_policy}
    )
    behavior_values = {
        behavior_policy,
        _non_empty(step.get("behavior_policy_version")),
        _non_empty(training.get("behavior_policy_version")),
        _non_empty(sync.get("behavior_policy_version")),
    }
    checks["behavior_policy_binding"] = bool(
        behavior_policy and behavior_values == {behavior_policy}
    )
    checks["checkpoint_publish_route_switch"] = bool(
        closure.get("publish_success") is True
        and closure.get("route_switch_success") is True
        and sync.get("success") is True
        and sync.get("status") == "published"
        and sync.get("route_switch_success") is True
        and sync.get("checkpoint_path") == checkpoint_path
    )
    checks["updated_policy_canary_and_next_rollout"] = bool(
        closure.get("next_policy_rollout_verified") is True
        and sync.get("canary_succeeded") is True
        and bool(step.get("canary_trajectory_ids"))
    )
    checks["wandb_binding"] = bool(
        closure.get("wandb_logged") is True
        and _non_empty(closure.get("wandb_run_id"))
        and closure.get("wandb_run_id") == state.get("wandb_run_id")
        and _non_empty(
            _nested(state, "latest_wandb_checkpoint_artifact", "name", default="")
        )
    )
    checks["ledger_epoch_binding"] = bool(
        state.get("schema_version") == _RUN_STATE_SCHEMA
        and ledger_receipt.get("schema_version") == _LEDGER_EPOCH_SCHEMA
        and ledger_receipt.get("ledger_snapshot_id") == ledger_snapshot_id
        and ledger_receipt.get("skill_snapshot_id") == skill_snapshot_id
        and state.get("ledger_condition_id") == condition_id
        and state.get("ledger_snapshot_id") == ledger_snapshot_id
        and state.get("skill_snapshot_id") == skill_snapshot_id
        and versions.get("posterior") == ledger_snapshot_id
        and versions.get("skill_library") == skill_snapshot_id
        and ledger_epoch >= 0
    )

    candidate_count = skill.get("discovery_qualified_rule_count")
    producer_status = _non_empty(skill.get("status"))
    discovery_probe_count = skill.get("discovery_probe_count")
    evidence_policy_versions = tuple(
        str(item) for item in skill.get("evidence_policy_versions", ())
    )
    skill_core_valid = bool(
        skill.get("schema_version") == SKILL_RUNTIME_UPDATE_SCHEMA
        and skill.get("publication_epoch") == ledger_epoch
        and skill.get("discovery_epoch") == ledger_epoch - 1
        and skill.get("producer_complete")
        == (producer_status != "pending_heldout_confirmation")
        and isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count >= 0
        and isinstance(discovery_probe_count, int)
        and not isinstance(discovery_probe_count, bool)
        and discovery_probe_count >= 0
        and all(_non_empty(item) for item in evidence_policy_versions)
        and (discovery_probe_count == 0 or bool(evidence_policy_versions))
    )
    no_qualified_candidate = bool(
        skill_core_valid
        and candidate_count == 0
        and producer_status == "complete_no_qualified_candidate"
        and skill.get("producer_complete") is True
        and skill.get("active_rule_count") == 0
        and not skill.get("pending_confirmation_rule_ids")
    )

    heldout: HeldoutPairedCalibrationReceipt | None = None
    heldout_valid = False
    if heldout_calibration_receipt is not None:
        try:
            heldout = _heldout_receipt(heldout_calibration_receipt)
            if heldout is None:
                raise ValueError("held-out receipt unexpectedly absent")
            validate_heldout_paired_calibration(
                heldout,
                expected_manifest=heldout.manifest,
            )
            manifest = heldout.manifest
            heldout_valid = bool(
                manifest.policy_version == updated_policy
                and manifest.condition_id == condition_id
                and manifest.posterior_snapshot_id == ledger_snapshot_id
                and manifest.evaluator_version == versions.get("evaluator")
                and manifest.feature_schema_version == versions.get("feature_schema")
                and manifest.model_catalog_version == versions.get("model_catalog")
                and heldout.problem_count >= 20
                and skill.get("confirmation_probe_count") == heldout.probe_count
            )
        except (KeyError, TypeError, ValueError):
            heldout_valid = False

    if no_qualified_candidate:
        heldout_status = "not_applicable_no_qualified_candidate"
        skill_ready = True
        heldout_requirement_satisfied = True
    else:
        heldout_status = (
            "complete" if heldout_valid else "required_or_invalid"
        )
        skill_ready = bool(
            skill_core_valid
            and candidate_count > 0
            and producer_status in {"complete", "complete_no_active_skill"}
            and skill.get("producer_complete") is True
            and not skill.get("pending_confirmation_rule_ids")
        )
        heldout_requirement_satisfied = heldout_valid
    checks["skill_producer_decision_complete"] = skill_ready
    checks["heldout_confirmation_requirement"] = heldout_requirement_satisfied

    blockers = tuple(key for key, passed in checks.items() if not passed)
    authorized = not blockers
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    return SupplementalLongTrainingAdmissionReceipt(
        status="authorized" if authorized else "blocked",
        long_training_authorized=authorized,
        behavior_policy_version=behavior_policy or "missing",
        updated_policy_version=updated_policy or "missing",
        ledger_epoch=max(ledger_epoch, 0),
        ledger_condition_id=condition_id or "missing",
        ledger_snapshot_id=ledger_snapshot_id or "missing",
        skill_snapshot_id=skill_snapshot_id or "missing",
        skill_runtime_status=producer_status or "missing",
        heldout_calibration_status=heldout_status,
        checks=checks,
        blockers=blockers,
        evidence_references=evidence_references or {},
        created_at=timestamp,
    )


def validate_supplemental_long_training_admission(
    receipt: SupplementalLongTrainingAdmissionReceipt,
    *,
    one_step_closure: Mapping[str, Any],
    run_state: Mapping[str, Any],
    step_manifest: Mapping[str, Any],
    ledger_epoch_receipt: Mapping[str, Any],
    skill_runtime_receipt: SkillRuntimeUpdateReceipt | Mapping[str, Any],
    heldout_calibration_receipt: (
        HeldoutPairedCalibrationReceipt | Mapping[str, Any] | None
    ) = None,
) -> None:
    """Recompute all checks and reject a stale or altered admission receipt."""

    if not isinstance(receipt, SupplementalLongTrainingAdmissionReceipt):
        raise TypeError("receipt must be SupplementalLongTrainingAdmissionReceipt")
    expected = build_supplemental_long_training_admission(
        one_step_closure=one_step_closure,
        run_state=run_state,
        step_manifest=step_manifest,
        ledger_epoch_receipt=ledger_epoch_receipt,
        skill_runtime_receipt=skill_runtime_receipt,
        heldout_calibration_receipt=heldout_calibration_receipt,
        evidence_references=receipt.evidence_references,
        created_at=receipt.created_at,
    )
    if receipt.to_dict() != expected.to_dict():
        raise ValueError("supplemental long-training admission receipt mismatch")


def save_supplemental_long_training_admission(
    receipt: SupplementalLongTrainingAdmissionReceipt,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically persist the supplemental receipt without touching its inputs."""

    if not isinstance(receipt, SupplementalLongTrainingAdmissionReceipt):
        raise TypeError("receipt must be SupplementalLongTrainingAdmissionReceipt")
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


def load_supplemental_long_training_admission(
    path: str | os.PathLike[str],
) -> SupplementalLongTrainingAdmissionReceipt:
    """Load the independently versioned supplemental receipt."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return SupplementalLongTrainingAdmissionReceipt.from_dict(
        _mapping(value, "supplemental long-training admission")
    )


__all__ = [
    "EVIDENCE_PRODUCING_TRAINING_SCOPE",
    "SUPPLEMENTAL_LONG_TRAINING_ADMISSION_SCHEMA",
    "SupplementalLongTrainingAdmissionReceipt",
    "build_supplemental_long_training_admission",
    "load_supplemental_long_training_admission",
    "save_supplemental_long_training_admission",
    "validate_supplemental_long_training_admission",
]
