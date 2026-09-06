from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TTB_CONFIG = ROOT / "config" / "training_hotpotqa_skillflow_ttb.yaml"
GRPO_CONFIG = ROOT / "config" / "training_hotpotqa_grpo.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_md_selects_one_pass_grpo_and_disables_ttb() -> None:
    config = _load(TTB_CONFIG)
    method = config["method_and_conflict_resolution"]
    assert method["compliance_marker"] == "MD_FULL_COMPLIANCE_20260906_V2"
    assert method["decision_status"] == "selected"
    assert method["primary"] == "action_masked_one_pass_grpo"
    assert method["paper_formal_name"] == "Tempered Trajectory Balance (TTB)"
    assert method["grpo"]["enabled"] is True
    assert method["grpo"]["role"] == "primary_task_learning_objective"
    assert method["ttb"]["enabled"] is False
    assert method["ttb"]["role"] == "disabled_not_permitted_in_md_training"
    assert method["mixing_losses_allowed"] is False
    assert config["run_identity"]["training_enabled"] is False
    assert config["actual_status"]["launch_allowed"] is False


def test_disabled_ttb_reference_is_not_a_launch_contract() -> None:
    config = _load(TTB_CONFIG)
    assert config["schema_version"] == "flowsteer.disabled_method_reference.v1"
    assert config["run_identity"]["status"] == (
        "disabled_by_MD_FULL_COMPLIANCE_20260906_V2"
    )
    assert config["closure_gate"]["require_nonzero_gradients"] == ["theta"]
    assert config["closure_gate"]["require_nonzero_parameter_updates"] == ["theta"]


def test_disabled_ttb_paper_reference_remains_attributed() -> None:
    config = _load(TTB_CONFIG)
    paper = config["disabled_ttb_reference_hyperparameters"]
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
    ambiguity = config["disabled_ttb_reference_ambiguities"]["phi_lora_rank"]
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
    assert wiring["skillflow_ttb_action_token_adapter"] == "disabled_not_permitted"
    assert wiring["theta_phi_Z_joint_training_on_agentgraph_trajectory"] == (
        "disabled_not_permitted"
    )
    assert wiring["real_two_step_closure"] == "not_run"
    for name in (
        "mace",
        "joint_bayesian_posterior",
        "evsi",
        "paired_intervention",
        "automatic_skill_publication",
    ):
        assert wiring[name] == "present_unwired"


def test_grpo_runner_is_selected_but_phase_gated() -> None:
    config = _load(GRPO_CONFIG)
    boundary = config["method_boundary"]
    assert boundary["compliance_marker"] == "MD_FULL_COMPLIANCE_20260906_V2"
    assert boundary["decision_status"] == "selected"
    assert boundary["role"] == "primary_task_learning_objective"
    assert boundary["primary_objective"] == "action_masked_one_pass_grpo"
    assert boundary["ttb_enabled"] is False
    assert boundary["mixing_losses_allowed"] is False
    compliance = config["md_compliance"]
    assert compliance["phase_0_status"] == "incomplete"
    assert compliance["real_step_authorized"] is False
    assert compliance["one_step_closure_status"] == "not_run"
    assert compliance["long_training_authorized"] is False
    assert config["experiment"]["training_enabled"] is False
    assert config["gpu"]["training_enabled"] is False
