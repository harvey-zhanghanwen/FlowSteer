"""Strict configuration contract for the SkillFlow primary TTB candidate.

The project design document specifies Action-Masked One-Pass GRPO, while
SkillFlow's primary training objective is tempered trajectory balance (TTB).
This module validates candidate B and prevents the two losses from being
enabled in the same training condition. It does not resolve the project-level
method decision; the external method-decision record remains authoritative.

MBPP+ is a project dataset adaptation; it is not part of SkillFlow's reported
seven-IID-task joint training protocol.  GPU assignments, rollout worker count,
micro-batch size, and the synchronization barrier are therefore recorded as
project implementation choices based on the resources available at launch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


TTB_ALGORITHM = "tempered_trajectory_balance"
PROJECT_IMPLEMENTATION = "project_implementation"
MBPP_PLUS_BENCHMARK = "mbpp_plus"
PROJECT_DATASET_ADAPTATION = "project_dataset_adaptation"


class TTBConfigurationError(ValueError):
    """A proposed training condition violates the formal TTB profile."""


@dataclass(frozen=True)
class LoraProfile:
    """One named LoRA profile used by the TTB trainer."""

    adapter_name: str
    rank: int
    alpha: int
    target_modules: tuple[str, ...]
    # The main-run table does not report dropout.  The released SkillFlow
    # implementation uses 0.05 for both named adapters.
    dropout: float = 0.05
    appendix_rank_ambiguity: int | None = None


@dataclass(frozen=True)
class PartitionFunctionProfile:
    """Training status of SkillFlow's separate partition function Z."""

    separate_parameter: bool = True
    optimized: bool = True


@dataclass(frozen=True)
class OptimizerProfile:
    """Optimizer and regularization values reported for the primary run."""

    name: str = "AdamW"
    learning_rate: float = 1.0e-4
    max_grad_norm: float = 3.0
    kl_coefficient: float = 0.01


@dataclass(frozen=True)
class ObjectiveSelection:
    """Objective identity inside this TTB candidate, not project approval."""

    design_document_objective: str = "action_masked_one_pass_grpo"
    skillflow_primary_objective: str = TTB_ALGORITHM
    selected_objective: str = TTB_ALGORITHM
    losses_mixed: bool = False


@dataclass(frozen=True)
class GPUAssignment:
    """One GPU role chosen from the actual launch-time resource inventory."""

    role: str
    physical_gpu: int


@dataclass(frozen=True)
class ProjectRuntimeProfile:
    """Engineering choices not specified by the SkillFlow paper."""

    implementation_source: str = PROJECT_IMPLEMENTATION
    gpu_assignments: tuple[GPUAssignment, ...] = ()
    rollout_workers: int | None = None
    micro_batch_size: int | None = None
    policy_sync_interval_steps: int = 1
    require_updated_theta_before_next_rollout: bool = True


@dataclass(frozen=True)
class TTBTrainingConfig:
    """Formal SkillFlow TTB profile with an explicit MBPP+ scope boundary."""

    algorithm: str = TTB_ALGORITHM
    benchmark: str = MBPP_PLUS_BENCHMARK
    protocol_scope: str = PROJECT_DATASET_ADAPTATION
    skillflow_joint_iid_reproduction: bool = False

    optimizer_steps: int = 250
    questions_per_batch: int = 7
    trajectories_per_question: int = 4
    effective_batch_size: int = 28
    max_trajectory_steps: int = 12
    on_policy_rollouts: bool = True

    beta: float = 1.0
    epsilon_min: float = 0.1
    edge_logprob_normalization: str = "per_action_token"
    reasoning_token_treatment: str = "context_only"

    theta_lora: LoraProfile = field(
        default_factory=lambda: LoraProfile(
            adapter_name="theta",
            rank=64,
            alpha=128,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        )
    )
    phi_lora: LoraProfile = field(
        default_factory=lambda: LoraProfile(
            adapter_name="phi",
            rank=16,
            alpha=32,
            target_modules=("q_proj", "v_proj"),
            appendix_rank_ambiguity=32,
        )
    )
    partition_function: PartitionFunctionProfile = field(
        default_factory=PartitionFunctionProfile
    )
    optimizer: OptimizerProfile = field(default_factory=OptimizerProfile)
    objective_selection: ObjectiveSelection = field(default_factory=ObjectiveSelection)

    checkpoint_interval_steps: int = 10
    executor_frozen: bool = True
    grpo_enabled: bool = False
    skill_evolution_enabled: bool = False
    mace_enabled: bool = False
    bayesian_posterior_enabled: bool = False
    runtime: ProjectRuntimeProfile = field(default_factory=ProjectRuntimeProfile)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return the complete non-secret protocol record for a run manifest."""

        return asdict(self)


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise TTBConfigurationError(
            f"{name} must be {expected!r} for the formal TTB profile; got {actual!r}"
        )


def _require_finite_equal(name: str, actual: Any, expected: float) -> None:
    if isinstance(actual, bool):
        raise TTBConfigurationError(f"{name} must be numeric")
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as exc:
        raise TTBConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric) or numeric != expected:
        raise TTBConfigurationError(
            f"{name} must be {expected!r} for the formal TTB profile; got {actual!r}"
        )


def _validate_lora(
    profile: LoraProfile,
    *,
    name: str,
    adapter_name: str,
    rank: int,
    alpha: int,
    target_modules: tuple[str, ...],
    dropout: float,
    appendix_rank_ambiguity: int | None,
) -> None:
    if not isinstance(profile, LoraProfile):
        raise TTBConfigurationError(f"{name} must be a LoraProfile")
    _require_equal(f"{name}.adapter_name", profile.adapter_name, adapter_name)
    _require_equal(f"{name}.rank", profile.rank, rank)
    _require_equal(f"{name}.alpha", profile.alpha, alpha)
    _require_equal(f"{name}.target_modules", profile.target_modules, target_modules)
    _require_finite_equal(f"{name}.dropout", profile.dropout, dropout)
    _require_equal(
        f"{name}.appendix_rank_ambiguity",
        profile.appendix_rank_ambiguity,
        appendix_rank_ambiguity,
    )


def _validate_runtime(
    runtime: ProjectRuntimeProfile,
    *,
    require_runtime_resources: bool,
) -> None:
    if not isinstance(runtime, ProjectRuntimeProfile):
        raise TTBConfigurationError("runtime must be a ProjectRuntimeProfile")
    _require_equal(
        "runtime.implementation_source",
        runtime.implementation_source,
        PROJECT_IMPLEMENTATION,
    )
    _require_equal("runtime.policy_sync_interval_steps", runtime.policy_sync_interval_steps, 1)
    _require_equal(
        "runtime.require_updated_theta_before_next_rollout",
        runtime.require_updated_theta_before_next_rollout,
        True,
    )

    roles: set[str] = set()
    devices: set[int] = set()
    for assignment in runtime.gpu_assignments:
        if not isinstance(assignment, GPUAssignment):
            raise TTBConfigurationError(
                "runtime.gpu_assignments must contain GPUAssignment values"
            )
        if not isinstance(assignment.role, str) or not assignment.role.strip():
            raise TTBConfigurationError("GPU assignment role must be non-empty")
        if (
            isinstance(assignment.physical_gpu, bool)
            or not isinstance(assignment.physical_gpu, int)
            or assignment.physical_gpu < 0
        ):
            raise TTBConfigurationError("physical_gpu must be a non-negative integer")
        if assignment.role in roles:
            raise TTBConfigurationError("GPU assignment roles must be unique")
        if assignment.physical_gpu in devices:
            raise TTBConfigurationError("physical GPU assignments must be exclusive")
        roles.add(assignment.role)
        devices.add(assignment.physical_gpu)

    for name, value in (
        ("runtime.rollout_workers", runtime.rollout_workers),
        ("runtime.micro_batch_size", runtime.micro_batch_size),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise TTBConfigurationError(f"{name} must be a positive integer when set")

    if require_runtime_resources:
        missing = []
        if not runtime.gpu_assignments:
            missing.append("gpu_assignments")
        if runtime.rollout_workers is None:
            missing.append("rollout_workers")
        if runtime.micro_batch_size is None:
            missing.append("micro_batch_size")
        if missing:
            raise TTBConfigurationError(
                "launch-time project runtime resources are not configured: "
                + ", ".join(missing)
            )


def validate_ttb_config(
    config: TTBTrainingConfig,
    *,
    require_runtime_resources: bool = False,
) -> TTBTrainingConfig:
    """Validate and return one unmixed SkillFlow-primary TTB condition."""

    if not isinstance(config, TTBTrainingConfig):
        raise TTBConfigurationError("config must be a TTBTrainingConfig")

    _require_equal("algorithm", config.algorithm, TTB_ALGORITHM)
    _require_equal("benchmark", config.benchmark, MBPP_PLUS_BENCHMARK)
    _require_equal("protocol_scope", config.protocol_scope, PROJECT_DATASET_ADAPTATION)
    _require_equal(
        "skillflow_joint_iid_reproduction",
        config.skillflow_joint_iid_reproduction,
        False,
    )
    _require_equal("optimizer_steps", config.optimizer_steps, 250)
    _require_equal("questions_per_batch", config.questions_per_batch, 7)
    _require_equal(
        "trajectories_per_question", config.trajectories_per_question, 4
    )
    _require_equal("effective_batch_size", config.effective_batch_size, 28)
    if config.effective_batch_size != (
        config.questions_per_batch * config.trajectories_per_question
    ):
        raise TTBConfigurationError(
            "effective_batch_size must equal questions_per_batch multiplied by "
            "trajectories_per_question"
        )
    _require_equal("max_trajectory_steps", config.max_trajectory_steps, 12)
    _require_equal("on_policy_rollouts", config.on_policy_rollouts, True)

    _require_finite_equal("beta", config.beta, 1.0)
    _require_finite_equal("epsilon_min", config.epsilon_min, 0.1)
    _require_equal(
        "edge_logprob_normalization",
        config.edge_logprob_normalization,
        "per_action_token",
    )
    _require_equal(
        "reasoning_token_treatment", config.reasoning_token_treatment, "context_only"
    )

    _validate_lora(
        config.theta_lora,
        name="theta_lora",
        adapter_name="theta",
        rank=64,
        alpha=128,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        dropout=0.05,
        appendix_rank_ambiguity=None,
    )
    _validate_lora(
        config.phi_lora,
        name="phi_lora",
        adapter_name="phi",
        rank=16,
        alpha=32,
        target_modules=("q_proj", "v_proj"),
        dropout=0.05,
        appendix_rank_ambiguity=32,
    )

    if not isinstance(config.partition_function, PartitionFunctionProfile):
        raise TTBConfigurationError(
            "partition_function must be a PartitionFunctionProfile"
        )
    _require_equal(
        "partition_function.separate_parameter",
        config.partition_function.separate_parameter,
        True,
    )
    _require_equal("partition_function.optimized", config.partition_function.optimized, True)

    if not isinstance(config.optimizer, OptimizerProfile):
        raise TTBConfigurationError("optimizer must be an OptimizerProfile")
    _require_equal("optimizer.name", config.optimizer.name, "AdamW")
    _require_finite_equal("optimizer.learning_rate", config.optimizer.learning_rate, 1.0e-4)
    _require_finite_equal("optimizer.max_grad_norm", config.optimizer.max_grad_norm, 3.0)
    _require_finite_equal("optimizer.kl_coefficient", config.optimizer.kl_coefficient, 0.01)

    if not isinstance(config.objective_selection, ObjectiveSelection):
        raise TTBConfigurationError(
            "objective_selection must be an ObjectiveSelection"
        )
    _require_equal(
        "objective_selection.design_document_objective",
        config.objective_selection.design_document_objective,
        "action_masked_one_pass_grpo",
    )
    _require_equal(
        "objective_selection.skillflow_primary_objective",
        config.objective_selection.skillflow_primary_objective,
        TTB_ALGORITHM,
    )
    _require_equal(
        "objective_selection.selected_objective",
        config.objective_selection.selected_objective,
        TTB_ALGORITHM,
    )
    _require_equal(
        "objective_selection.losses_mixed",
        config.objective_selection.losses_mixed,
        False,
    )

    _require_equal("checkpoint_interval_steps", config.checkpoint_interval_steps, 10)
    _require_equal("executor_frozen", config.executor_frozen, True)
    _require_equal("grpo_enabled", config.grpo_enabled, False)
    _require_equal("skill_evolution_enabled", config.skill_evolution_enabled, False)
    _require_equal("mace_enabled", config.mace_enabled, False)
    _require_equal(
        "bayesian_posterior_enabled", config.bayesian_posterior_enabled, False
    )
    _validate_runtime(
        config.runtime,
        require_runtime_resources=require_runtime_resources,
    )
    return config


__all__ = [
    "GPUAssignment",
    "LoraProfile",
    "ObjectiveSelection",
    "OptimizerProfile",
    "PartitionFunctionProfile",
    "ProjectRuntimeProfile",
    "TTB_ALGORITHM",
    "TTBConfigurationError",
    "TTBTrainingConfig",
    "validate_ttb_config",
]
