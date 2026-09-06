from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TTB_CONFIG = ROOT / "config" / "training_hotpotqa_skillflow_ttb.yaml"
GRPO_CONFIG = ROOT / "config" / "training_hotpotqa_grpo.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_primary_method_is_ttb_and_grpo_is_disabled() -> None:
    config = _load(TTB_CONFIG)
    method = config["method_and_conflict_resolution"]
    assert method["primary"] == "tempered_trajectory_balance"
    assert method["paper_formal_name"] == "Tempered Trajectory Balance (TTB)"
    assert method["grpo"] == {
        "enabled": False,
        "role": "baseline_or_ablation_only",
    }
    assert config["run_identity"]["training_enabled"] is False
    assert config["actual_status"]["launch_allowed"] is False


def test_ttb_scoring_and_joint_trainable_contract() -> None:
    config = _load(TTB_CONFIG)
    method = config["method_and_conflict_resolution"]
    paper = config["paper_hyperparameters"]
    assert method["score_only"] == "structured_action_tokens"
    assert method["reasoning_tokens"] == "context_only_zero_loss_mask"
    assert method["edge_log_probability"] == "mean_over_action_tokens"
    assert method["rollout_policy"] == "current_theta_on_policy"
    assert method["executor"] == "frozen"
    assert method["terminal_reward"]["primary"] == "official_exact_match"
    assert method["terminal_reward"]["auxiliary_metric"] == "token_f1"
    assert paper["joint_updates"] == ["theta", "partition_function_Z", "phi"]
    assert paper["partition_function"]["trainable"] is True


def test_paper_parameters_and_phi_ambiguity_are_explicit() -> None:
    config = _load(TTB_CONFIG)
    paper = config["paper_hyperparameters"]
    assert paper["optimizer_steps"] == 250
    assert (paper["questions_per_step"], paper["trajectories_per_question"]) == (7, 4)
    assert paper["effective_batch_size"] == 28
    assert paper["max_trajectory_steps"] == 12
    assert (paper["beta"], paper["epsilon_min"], paper["kl_coefficient"]) == (
        1.0,
        0.1,
        0.01,
    )
    assert paper["theta_lora"] == {
        "rank": 64,
        "alpha": 128,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    assert paper["phi_lora"] == {
        "selected_rank": 16,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    }
    ambiguity = config["paper_ambiguities"]["phi_lora_rank"]
    assert (ambiguity["main_table"], ambiguity["appendix_variant"]) == (16, 32)


def test_hardware_and_project_engineering_provenance_are_separate() -> None:
    config = _load(TTB_CONFIG)
    hardware = config["paper_hardware"]
    for field in (
        "tensor_parallel_layout",
        "data_parallel_layout",
        "zero_stage",
        "rollout_worker_count",
        "sampling_parameters",
        "gpu_role_mapping",
    ):
        assert hardware[field] == "not_specified_by_paper"
    assert config["resource_admission"]["resolved_gpu_plan"] is None
    assert config["resource_admission"]["status"] == "resource_blocked"
    assert "project engineering" in config["project_adaptations"]["provenance_note"]


def test_unimplemented_components_and_real_closure_remain_blocked() -> None:
    config = _load(TTB_CONFIG)
    wiring = config["wiring_status"]
    assert wiring["skillflow_ttb_action_token_adapter"] == "not_implemented"
    assert wiring["theta_phi_Z_joint_training_on_agentgraph_trajectory"] == "not_implemented"
    assert wiring["real_two_step_closure"] == "not_run"
    for name in (
        "mace",
        "joint_bayesian_posterior",
        "evsi",
        "paired_intervention",
        "automatic_skill_publication",
    ):
        assert wiring[name] == "present_unwired"


def test_grpo_runner_is_only_a_baseline() -> None:
    config = _load(GRPO_CONFIG)
    boundary = config["method_boundary"]
    assert boundary["role"] == "baseline_or_ablation_only"
    assert boundary["eligible_as_skillflow_main"] is False
    assert boundary["primary_skillflow_method"] == "tempered_trajectory_balance"
