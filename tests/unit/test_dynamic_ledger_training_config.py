"""Static admission checks for the HotpotQA dynamic-ledger profile.

These tests parse configuration only.  They never start SGLang, call an API,
create a W&B run, allocate a GPU, or execute an optimizer step.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.interactive.config_loader import load_yaml, validate_agent_graph_config


ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_CONFIG = ROOT / "config" / "training_hotpotqa_dynamic_ledger_grpo.yaml"
BASE_CONFIG = ROOT / "config" / "training_hotpotqa_grpo.yaml"
COMPLIANCE_DOC = ROOT / "docs" / "LATENT_LOSS_DYNAMIC_LEDGER_COMPLIANCE.md"


def _raw(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_dynamic_profile_keeps_grpo_as_the_only_differentiable_objective() -> None:
    config = load_yaml(DYNAMIC_CONFIG)
    validate_agent_graph_config(config)

    method = config["method_boundary"]
    assert method["compliance_marker"] == "MD_FULL_COMPLIANCE_20260906_V2"
    assert method["latent_loss_marker"] == "LATENT_LOSS_DYNAMIC_LEDGER_V1"
    assert method["primary_objective"] == "action_masked_one_pass_grpo"
    assert method["differentiable_objectives"] == ["action_masked_one_pass_grpo"]
    assert method["ttb_enabled"] is False
    assert method["mixing_losses_allowed"] is False
    assert method["latent_loss_is_differentiable_objective"] is False
    assert method["probe_and_skill_reward_contribution"] == 0.0

    grpo = config["grpo"]
    assert grpo["enabled"] is True
    assert grpo["objective"] == "action_masked_one_pass"
    assert grpo["terminal_reward_field"] == "token_f1"
    assert grpo["group_key"] == ["task_id", "condition_id", "policy_version"]
    assert grpo["optimization_passes_per_rollout_batch"] == 1
    assert grpo["max_optimizer_updates_per_step"] == 1
    for field in (
        "structural_reward",
        "exploration_reward",
        "skill_usage_reward",
        "latent_loss_reward",
        "probe_reward",
    ):
        assert grpo[field] == 0.0
    assert grpo["reject_forced_probes"] is True
    assert grpo["reject_audits"] is True


def test_combination_posterior_parameters_and_data_routing_match_spec() -> None:
    config = _raw(DYNAMIC_CONFIG)
    ledger = config["dynamic_ledger"]
    assert ledger["enabled"] is True
    assert ledger["differentiable"] is False
    assert ledger["contributes_to_grpo_reward"] is False
    assert ledger["terminal_binary_outcome"] == "exact_match"
    assert ledger["comparison_fit_source"] == "same_snapshot_paired_probes_only"
    assert set(ledger["absolute_baseline_sources"]) == {
        "natural_trajectory",
        "probe_branch",
        "audit_branch",
    }
    assert set(ledger["sensor_sources"]) == set(ledger["absolute_baseline_sources"])

    key = ledger["decision_key"]
    assert key["role_clusters"] == [
        "solve",
        "verify",
        "plan",
        "summarize",
        "arbitrate",
        "retrieve",
        "code",
        "test",
    ]
    assert key["edge_types"] == ["independent", "unidirectional", "bidirectional"]
    assert key["same_model_as_upstream"] == [False, True]
    assert key["stages"] == ["before_output", "other"]
    assert key["role_assignment"] == "llm_strict_json"
    assert key["keyword_fallback_allowed"] is False

    posterior = ledger["posterior"]
    assert posterior["family"] == "gaussian_conjugate"
    assert posterior["interaction_activation_probes"] == 5
    assert posterior["noise_variance_floor"] == 0.01
    assert posterior["empirical_bayes_once_per_epoch"] is True
    assert posterior["sensor_prior"] == "Beta(1,1)"

    risk = ledger["latent_risk"]
    assert (risk["tau"], risk["delta_min"]) == (0.8, 0.05)
    assert (risk["candidate_top_k"], risk["posterior_samples"]) == (5, 200)
    assert risk["director_action_space_unchanged"] is True

    exploration = config["exploration"]
    assert exploration["enabled"] is True
    assert exploration["legacy_mace_enabled"] is False
    assert exploration["legacy_joint_bayesian_enabled"] is False
    assert exploration["legacy_particle_evsi_enabled"] is False
    probe = exploration["probe"]
    assert probe["budget_fraction_of_natural_trajectories"] == 0.10
    assert probe["repeats_per_arm"] == 3
    assert probe["same_snapshot"] is True
    assert probe["exactly_one_decision_field_changed"] is True
    assert probe["enters_grpo"] is False
    assert probe["enters_standard_metrics"] is False
    audit = exploration["audit"]
    assert audit["probability"] == 0.05
    assert audit["repeats_per_arm"] == 3
    assert audit["enters_grpo"] is False
    assert audit["enters_standard_metrics"] is False


def test_skill_lifecycle_is_calibrated_and_does_not_change_reward() -> None:
    config = _raw(DYNAMIC_CONFIG)
    skills = config["skills"]
    assert skills["enabled"] is True
    assert skills["source"] == "dynamic_combination_posterior"
    assert skills["statuses"] == ["candidate", "active", "suspended", "retired"]
    assert skills["minimum_discovery_pairs"] == 10
    assert skills["minimum_confirmation_problems"] == 20
    assert skills["calibration_alpha"] == 0.05
    assert skills["benjamini_hochberg_fdr"] == 0.10
    assert skills["maximum_active_injected_per_step"] == 3
    assert skills["newly_active_visible_from"] == "next_epoch"
    assert skills["hard_gate_director_action"] is False
    assert skills["reward_contribution"] == 0.0
    assert config["role_classification"]["method"] == "llm"
    assert config["role_classification"]["keyword_or_regex_fallback_allowed"] is False


def test_phase_a_through_e_remain_fail_closed_and_phase_e_is_not_claimed() -> None:
    config = _raw(DYNAMIC_CONFIG)
    compliance = config["latent_loss_compliance"]
    assert compliance["fail_closed"] is True
    assert compliance["all_required_gates_passed"] is False
    assert compliance["optimizer_step_authorized"] is False
    assert config["experiment"]["training_enabled"] is False
    assert config["gpu"]["training_enabled"] is False

    gates = compliance["gates"]
    assert tuple(gates) == (
        "phase_a_combination_posterior",
        "phase_b_latent_risk",
        "phase_c_paired_probe_and_audit",
        "phase_d_skill_lifecycle",
        "phase_e_integration",
    )
    assert all(gate["required"] is True for gate in gates.values())
    assert all(gate["receipt"] is None for gate in gates.values())
    assert all(not gate["status"].startswith("passed") for gate in gates.values())
    phase_e = gates["phase_e_integration"]
    assert phase_e["status"] == "pending_not_run"
    assert phase_e["required_unique_questions"] == 50
    assert phase_e["rollouts_per_question"] == 4
    assert phase_e["expected_natural_trajectories"] == 200


def test_gpu4_profile_is_single_worker_sequential_and_uses_task_port() -> None:
    config = load_yaml(DYNAMIC_CONFIG)
    gpu = config["gpu"]
    assert gpu["execution_layout"] == "single_gpu_sequential"
    assert gpu["gradient_worker_count"] == 1
    assert {
        gpu["learner_physical"],
        gpu["rollout_physical"],
        gpu["gradient_replica_physical"],
        gpu["supervisor_gpu_id"],
    } == {4}
    assert gpu["learner_device"] == gpu["gradient_replica_device"] == "cuda:4"
    assert gpu["supervisor_port"] == 8016
    assert config["director"]["api_base"] == "http://127.0.0.1:8016/v1"
    assert config["policy_sync"]["api_base"] == "http://127.0.0.1:8016"
    assert gpu["lifecycle"]["sequence"] == [
        "rollout",
        "pause",
        "drain",
        "stop_sglang",
        "train",
        "restart_sglang",
        "publish",
        "canary",
        "resume",
    ]
    assert gpu["lifecycle"]["fail_closed"] is True
    assert gpu["lifecycle"]["stop_only_runtime_owned_process"] is True
    assert gpu["oom_policy"]["micro_batch_schedule"] == [4, 2, 1]


def test_frozen_validation_monitor_and_round01_start_are_not_changed() -> None:
    dynamic = _raw(DYNAMIC_CONFIG)
    baseline = _raw(BASE_CONFIG)
    assert dynamic["source"]["backup_branch"] == baseline["source"]["backup_branch"]
    assert dynamic["source"]["backup_commit"] == baseline["source"]["backup_commit"]
    assert dynamic["director"]["behavior_adapter_checkpoint"] == baseline["director"][
        "behavior_adapter_checkpoint"
    ]
    assert dynamic["tracking"]["validation_protocol"]["monitor_task_ids"] == baseline[
        "tracking"
    ]["validation_protocol"]["monitor_task_ids"]


def test_compliance_document_names_sources_and_pending_phase_e() -> None:
    text = COMPLIANCE_DOC.read_text(encoding="utf-8")
    for required in (
        "Action-Masked One-Pass GRPO",
        "Tempered Trajectory Balance",
        "CombinationPosterior",
        "post_execution_latent_loss",
        "collect_probe_branch",
        "SequentialGpuRuntime",
        "pending_not_run",
        "50 questions × 4",
    ):
        assert required in text
