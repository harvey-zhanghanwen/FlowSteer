"""CPU-only tests for the supplemental long-training admission boundary."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from src.interactive.exploration.heldout_calibration import (
    HeldoutPairedCalibrationManifest,
    HeldoutPairedCalibrationObservation,
    HeldoutPairedCalibrationReceipt,
)
from src.interactive.exploration.long_training_admission import (
    build_supplemental_long_training_admission,
    load_supplemental_long_training_admission,
    save_supplemental_long_training_admission,
    validate_supplemental_long_training_admission,
)


BEHAVIOR_POLICY = "policy-e0"
UPDATED_POLICY = "policy-e1"
CONDITION = "condition-e1"
LEDGER = "ledger-e1"
SKILLS = "skills-e1"
CHECKPOINT = "/checkpoint/theta"
TRAINING_STATE = "/checkpoint/theta/training_state.pt"


def _closure() -> dict:
    return {
        "schema_version": "flowsteer.dynamic-grpo-one-step-closure.v1",
        "status": "passed",
        "long_training_authorized": False,
        "objective": "action_masked_one_pass_grpo",
        "optimizer_updates": 1,
        "behavior_policy_version": BEHAVIOR_POLICY,
        "updated_policy_version": UPDATED_POLICY,
        "nonzero_gradient": True,
        "nonzero_lora_update": True,
        "checkpoint_recoverable": True,
        "publish_success": True,
        "route_switch_success": True,
        "next_policy_rollout_verified": True,
        "wandb_logged": True,
        "wandb_run_id": "wandb-run",
        "ttb_enabled": False,
    }


def _run_state() -> dict:
    return {
        "schema_version": "flowsteer.hotpotqa.grpo_run_state.v1",
        "optimizer_updates_completed": 1,
        "behavior_policy_version": UPDATED_POLICY,
        "behavior_adapter_checkpoint": CHECKPOINT,
        "training_state_checkpoint": TRAINING_STATE,
        "optimizer_state_checkpoint": TRAINING_STATE,
        "ledger_condition_id": CONDITION,
        "ledger_snapshot_id": LEDGER,
        "skill_snapshot_id": SKILLS,
        "wandb_run_id": "wandb-run",
        "latest_wandb_checkpoint_artifact": {"name": "checkpoint:v0"},
        "dynamic_phase_e_acceptance": {
            "ttb_enabled": False,
            "phase_e": {
                "checks": {
                    "natural_only_grpo": True,
                    "same_condition_groups": True,
                    "unique_question_count": True,
                    "probe_and_audit_excluded_from_grpo": True,
                },
                "metrics": {
                    "routing_assertions": {
                        "comparison_fit_source": "same_snapshot_paired_probes_only",
                        "natural_only_grpo": True,
                        "probe_and_audit_excluded_from_grpo": True,
                        "latent_risk_reward_contribution": 0,
                        "skill_reward_contribution": 0,
                    }
                },
            },
        },
    }


def _step() -> dict:
    return {
        "schema_version": "flowsteer.hotpotqa.grpo_step_manifest.v1",
        "status": "committed",
        "optimizer_step": 1,
        "behavior_policy_version": BEHAVIOR_POLICY,
        "candidate_policy_version": UPDATED_POLICY,
        "canary_trajectory_ids": ["canary-1"],
        "training": {
            "behavior_policy_version": BEHAVIOR_POLICY,
            "updated_policy_version": UPDATED_POLICY,
            "optimizer_updates": 1,
            "trained_trajectories": 12,
            "exact_groups": 7,
            "record_eligible_trajectories": 28,
            "input_trajectories": 28,
            "grad_norm": 0.5,
            "trainable_update_l2": 0.25,
            "checkpoint_dir": CHECKPOINT,
            "checkpoint_recoverable": True,
            "training_state_checkpoint": TRAINING_STATE,
            "optimizer_state_checkpoint": TRAINING_STATE,
            "training_state_saved": True,
            "optimizer_state_saved": True,
            "scheduler_state_saved": True,
            "rng_state_saved": True,
        },
        "policy_sync": {
            "behavior_policy_version": BEHAVIOR_POLICY,
            "candidate_policy_version": UPDATED_POLICY,
            "new_policy_version": UPDATED_POLICY,
            "route_policy_version": UPDATED_POLICY,
            "checkpoint_path": CHECKPOINT,
            "success": True,
            "status": "published",
            "route_switch_success": True,
            "canary_succeeded": True,
        },
        "metrics": {
            "train/grad_norm": 0.5,
            "train/lora_update_l2": 0.25,
        },
    }


def _ledger_receipt() -> dict:
    return {
        "schema_version": "flowsteer.ledger-epoch.v1",
        "ledger_snapshot_id": LEDGER,
        "skill_snapshot_id": SKILLS,
        "condition": {
            "schema_version": "flowsteer.ledger-epoch.v1",
            "epoch": 1,
            "condition_id": CONDITION,
            "ledger_snapshot_id": LEDGER,
            "skill_snapshot_id": SKILLS,
            "versions": {
                "policy": UPDATED_POLICY,
                "posterior": LEDGER,
                "skill_library": SKILLS,
                "evaluator": "evaluator-v1",
                "feature_schema": "features-v1",
                "model_catalog": "catalog-v1",
            },
        },
    }


def _skill_receipt(*, candidate: bool = False, pending: bool = False) -> dict:
    if not candidate:
        status = "complete_no_qualified_candidate"
    elif pending:
        status = "pending_heldout_confirmation"
    else:
        status = "complete"
    return {
        "schema_version": "flowsteer.skill-runtime-update.v1",
        "status": status,
        "producer_complete": not pending,
        "discovery_epoch": 0,
        "publication_epoch": 1,
        "discovery_probe_count": 10 if candidate else 2,
        "confirmation_probe_count": 20 if candidate and not pending else 0,
        "discovery_qualified_rule_count": 1 if candidate else 0,
        "active_rule_count": 1 if candidate and not pending else 0,
        "pending_confirmation_rule_ids": ["rule-1"] if pending else [],
        "evidence_policy_versions": [BEHAVIOR_POLICY],
        "publication_policy_version": UPDATED_POLICY,
        "grpo_reward_contribution": 0.0,
        "ttb_enabled": False,
    }


def _heldout_receipt() -> HeldoutPairedCalibrationReceipt:
    problems = tuple(f"validation-{index:02d}" for index in range(20))
    manifest = HeldoutPairedCalibrationManifest(
        dataset="hotpotqa",
        problem_ids=problems,
        discovery_problem_ids=tuple(f"train-{index:02d}" for index in range(20)),
        discovery_probe_ids=("discovery-probe",),
        policy_version=UPDATED_POLICY,
        condition_id=CONDITION,
        evaluator_version="evaluator-v1",
        feature_schema_version="features-v1",
        model_catalog_version="catalog-v1",
        posterior_snapshot_id=LEDGER,
    )
    observations = tuple(
        HeldoutPairedCalibrationObservation(
            probe_id=f"validation-probe-{index:02d}",
            problem_id=problem,
            snapshot_id=f"snapshot-{index:02d}",
            condition_id=CONDITION,
            policy_version=UPDATED_POLICY,
            evaluator_version="evaluator-v1",
            feature_schema_version="features-v1",
            incumbent_action={"model_id": "a"},
            candidate_action={"model_id": "b"},
            branch_trajectory_ids=tuple(
                f"branch-{index:02d}-{branch}" for branch in range(6)
            ),
            observed_delta=0.0,
            predicted_mean=0.0,
            posterior_variance=1.0,
            noise_variance=1.0,
            standardized_residual=0.0,
        )
        for index, problem in enumerate(problems)
    )
    return HeldoutPairedCalibrationReceipt(
        manifest=manifest,
        observations=observations,
        calibration_quantile=0.0,
        empirical_coverage=1.0,
        uncalibrated_coverage_80=1.0,
        uncalibrated_coverage_90=1.0,
        uncalibrated_coverage_95=1.0,
        calibrated_interval_width_mean=0.0,
        gaussian_nll_mean=0.5 * math.log(4.0 * math.pi),
        direction_accuracy=1.0,
    )


def _build(**overrides):
    inputs = {
        "one_step_closure": _closure(),
        "run_state": _run_state(),
        "step_manifest": _step(),
        "ledger_epoch_receipt": _ledger_receipt(),
        "skill_runtime_receipt": _skill_receipt(),
        "created_at": "2026-09-07T00:00:00+00:00",
    }
    inputs.update(overrides)
    return build_supplemental_long_training_admission(**inputs)


def test_no_qualified_candidate_authorizes_only_evidence_producing_training(
    tmp_path,
) -> None:
    closure = _closure()
    before = deepcopy(closure)
    receipt = _build(
        one_step_closure=closure,
        evidence_references={"one_step_closure": "/immutable/closure.json"},
    )

    assert receipt.status == "authorized"
    assert receipt.long_training_authorized is True
    assert receipt.scope == "evidence-producing-training-only"
    assert receipt.heldout_calibration_status == (
        "not_applicable_no_qualified_candidate"
    )
    assert receipt.phase4_complete is False
    assert receipt.phase5_complete is False
    assert receipt.ttb_enabled is False
    assert receipt.blockers == ()
    assert closure == before

    path = save_supplemental_long_training_admission(receipt, tmp_path / "gate.json")
    restored = load_supplemental_long_training_admission(path)
    assert restored.to_dict() == receipt.to_dict()


def test_pending_candidate_without_real_heldout_receipt_is_blocked() -> None:
    receipt = _build(
        skill_runtime_receipt=_skill_receipt(candidate=True, pending=True)
    )
    assert receipt.status == "blocked"
    assert receipt.long_training_authorized is False
    assert "skill_producer_decision_complete" in receipt.blockers
    assert "heldout_confirmation_requirement" in receipt.blockers


def test_confirmed_candidate_requires_version_bound_real_heldout_receipt() -> None:
    without = _build(skill_runtime_receipt=_skill_receipt(candidate=True))
    assert without.status == "blocked"
    assert without.heldout_calibration_status == "required_or_invalid"

    heldout = _heldout_receipt()
    receipt = _build(
        skill_runtime_receipt=_skill_receipt(candidate=True),
        heldout_calibration_receipt=heldout,
    )
    assert receipt.status == "authorized"
    assert receipt.heldout_calibration_status == "complete"

    mismatched = deepcopy(heldout.to_dict())
    mismatched["manifest"]["policy_version"] = "wrong-policy"
    for observation in mismatched["observations"]:
        observation["policy_version"] = "wrong-policy"
    blocked = _build(
        skill_runtime_receipt=_skill_receipt(candidate=True),
        heldout_calibration_receipt=mismatched,
    )
    assert blocked.status == "blocked"
    assert "heldout_confirmation_requirement" in blocked.blockers


def test_policy_or_training_evidence_mismatch_fails_closed() -> None:
    state = _run_state()
    state["behavior_policy_version"] = "stale-policy"
    step = _step()
    step["training"]["trainable_update_l2"] = 0.0
    receipt = _build(run_state=state, step_manifest=step)
    assert receipt.status == "blocked"
    assert "updated_policy_binding" in receipt.blockers
    assert "nonzero_lora_update" in receipt.blockers


def test_revalidation_rejects_altered_serialized_receipt() -> None:
    receipt = _build()
    validate_supplemental_long_training_admission(
        receipt,
        one_step_closure=_closure(),
        run_state=_run_state(),
        step_manifest=_step(),
        ledger_epoch_receipt=_ledger_receipt(),
        skill_runtime_receipt=_skill_receipt(),
    )
    changed = receipt.to_dict()
    changed["updated_policy_version"] = "another-policy"
    with pytest.raises(ValueError, match="receipt mismatch"):
        validate_supplemental_long_training_admission(
            type(receipt).from_dict(changed),
            one_step_closure=_closure(),
            run_state=_run_state(),
            step_manifest=_step(),
            ledger_epoch_receipt=_ledger_receipt(),
            skill_runtime_receipt=_skill_receipt(),
        )
