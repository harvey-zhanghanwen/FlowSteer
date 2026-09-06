from __future__ import annotations

from dataclasses import replace

import pytest

from src.interactive.ttb_config import (
    GPUAssignment,
    LoraProfile,
    ObjectiveSelection,
    OptimizerProfile,
    PartitionFunctionProfile,
    ProjectRuntimeProfile,
    TTBConfigurationError,
    TTBTrainingConfig,
    validate_ttb_config,
)


def test_formal_skillflow_ttb_profile_is_valid_and_manifest_ready() -> None:
    config = validate_ttb_config(TTBTrainingConfig())
    manifest = config.to_manifest_dict()

    assert manifest["algorithm"] == "tempered_trajectory_balance"
    assert manifest["benchmark"] == "mbpp_plus"
    assert manifest["protocol_scope"] == "project_dataset_adaptation"
    assert manifest["skillflow_joint_iid_reproduction"] is False
    assert manifest["optimizer_steps"] == 250
    assert manifest["effective_batch_size"] == 7 * 4 == 28
    assert manifest["max_trajectory_steps"] == 12
    assert manifest["theta_lora"] == {
        "adapter_name": "theta",
        "rank": 64,
        "alpha": 128,
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "dropout": 0.05,
        "appendix_rank_ambiguity": None,
    }
    assert manifest["phi_lora"] == {
        "adapter_name": "phi",
        "rank": 16,
        "alpha": 32,
        "target_modules": ("q_proj", "v_proj"),
        "dropout": 0.05,
        "appendix_rank_ambiguity": 32,
    }
    assert manifest["partition_function"] == {
        "separate_parameter": True,
        "optimized": True,
    }
    assert manifest["executor_frozen"] is True
    assert manifest["grpo_enabled"] is False
    assert manifest["skill_evolution_enabled"] is False
    assert manifest["mace_enabled"] is False
    assert manifest["bayesian_posterior_enabled"] is False


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("algorithm", "action_masked_one_pass_grpo"),
        ("benchmark", "skillflow_seven_iid"),
        ("protocol_scope", "bit_exact_reproduction"),
        ("skillflow_joint_iid_reproduction", True),
        ("optimizer_steps", 300),
        ("questions_per_batch", 6),
        ("trajectories_per_question", 3),
        ("effective_batch_size", 24),
        ("max_trajectory_steps", 30),
        ("on_policy_rollouts", False),
        ("beta", 0.5),
        ("epsilon_min", 0.0),
        ("edge_logprob_normalization", "trajectory_token_mean"),
        ("reasoning_token_treatment", "optimized"),
        ("checkpoint_interval_steps", 1),
        ("executor_frozen", False),
        ("grpo_enabled", True),
        ("skill_evolution_enabled", True),
        ("mace_enabled", True),
        ("bayesian_posterior_enabled", True),
    ),
)
def test_formal_protocol_fields_cannot_drift(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(replace(TTBTrainingConfig(), **{field_name: invalid_value}))


@pytest.mark.parametrize(
    "profile",
    (
        LoraProfile("theta", 32, 128, ("q_proj", "k_proj", "v_proj", "o_proj")),
        LoraProfile("theta", 64, 64, ("q_proj", "k_proj", "v_proj", "o_proj")),
        LoraProfile("theta", 64, 128, ("q_proj", "v_proj")),
        LoraProfile(
            "theta",
            64,
            128,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            dropout=0.0,
        ),
    ),
)
def test_theta_lora_profile_is_fixed(profile: LoraProfile) -> None:
    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(replace(TTBTrainingConfig(), theta_lora=profile))


@pytest.mark.parametrize(
    "profile",
    (
        LoraProfile(
            "phi", 32, 32, ("q_proj", "v_proj"), appendix_rank_ambiguity=32
        ),
        LoraProfile(
            "phi", 16, 16, ("q_proj", "v_proj"), appendix_rank_ambiguity=32
        ),
        LoraProfile(
            "phi",
            16,
            32,
            ("q_proj", "k_proj", "v_proj"),
            appendix_rank_ambiguity=32,
        ),
        LoraProfile(
            "phi", 16, 32, ("q_proj", "v_proj"), appendix_rank_ambiguity=None
        ),
        LoraProfile(
            "phi",
            16,
            32,
            ("q_proj", "v_proj"),
            dropout=0.0,
            appendix_rank_ambiguity=32,
        ),
    ),
)
def test_phi_main_table_profile_and_appendix_ambiguity_are_fixed(
    profile: LoraProfile,
) -> None:
    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(replace(TTBTrainingConfig(), phi_lora=profile))


def test_partition_function_and_optimizer_are_strict() -> None:
    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(
            replace(
                TTBTrainingConfig(),
                partition_function=PartitionFunctionProfile(optimized=False),
            )
        )

    invalid_optimizers = (
        OptimizerProfile(name="SGD"),
        OptimizerProfile(learning_rate=1.0e-5),
        OptimizerProfile(max_grad_norm=1.0),
        OptimizerProfile(kl_coefficient=0.0),
    )
    for optimizer in invalid_optimizers:
        with pytest.raises(TTBConfigurationError):
            validate_ttb_config(replace(TTBTrainingConfig(), optimizer=optimizer))


def test_objective_conflict_is_explicit_and_losses_cannot_be_mixed() -> None:
    config = TTBTrainingConfig()
    selection = config.objective_selection
    assert selection.design_document_objective == "action_masked_one_pass_grpo"
    assert selection.skillflow_primary_objective == "tempered_trajectory_balance"
    assert selection.selected_objective == "tempered_trajectory_balance"
    assert selection.losses_mixed is False

    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(
            replace(
                config,
                objective_selection=replace(selection, losses_mixed=True),
            )
        )
    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(replace(config, grpo_enabled=True))


def test_runtime_choices_are_explicitly_project_implemented() -> None:
    runtime = ProjectRuntimeProfile(
        gpu_assignments=(
            GPUAssignment("learner", 2),
            GPUAssignment("rollout", 6),
        ),
        rollout_workers=8,
        micro_batch_size=2,
    )
    config = replace(TTBTrainingConfig(), runtime=runtime)

    validate_ttb_config(config, require_runtime_resources=True)
    assert config.to_manifest_dict()["runtime"]["implementation_source"] == (
        "project_implementation"
    )


def test_launch_validation_requires_actual_runtime_resource_values() -> None:
    validate_ttb_config(TTBTrainingConfig(), require_runtime_resources=False)
    with pytest.raises(TTBConfigurationError, match="not configured"):
        validate_ttb_config(TTBTrainingConfig(), require_runtime_resources=True)


@pytest.mark.parametrize(
    "runtime",
    (
        ProjectRuntimeProfile(implementation_source="skillflow_paper"),
        ProjectRuntimeProfile(
            gpu_assignments=(GPUAssignment("learner", 1), GPUAssignment("learner", 2))
        ),
        ProjectRuntimeProfile(
            gpu_assignments=(GPUAssignment("learner", 1), GPUAssignment("rollout", 1))
        ),
        ProjectRuntimeProfile(rollout_workers=0),
        ProjectRuntimeProfile(micro_batch_size=True),
        ProjectRuntimeProfile(policy_sync_interval_steps=2),
        ProjectRuntimeProfile(require_updated_theta_before_next_rollout=False),
    ),
)
def test_project_runtime_cannot_claim_paper_or_bypass_sync(
    runtime: ProjectRuntimeProfile,
) -> None:
    with pytest.raises(TTBConfigurationError):
        validate_ttb_config(replace(TTBTrainingConfig(), runtime=runtime))
