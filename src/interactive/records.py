"""Versioned records for the AgentGraph execution and learning planes.

These records deliberately keep natural-policy rollouts, forced probes, and
Skill evidence distinct.  No terminal score is implicitly copied to nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .persistence.ids import stable_id
from .versioning import VersionBundle


SCHEMA_VERSION = "flowsteer.agentgraph.v1"
VALID_SPLITS = frozenset({"train", "validation", "test"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    question: str
    ground_truth: Any
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    experiment_id: str
    graph_revision: int
    agent_id: str
    model_id: str
    model_fingerprint: str
    provider: str
    request_hash: str
    output: str
    temperature: float
    top_p: float
    max_tokens: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.graph_revision < 0:
            raise ValueError("graph_revision must be non-negative")
        if not 0 <= self.temperature:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurnRecord:
    turn_id: str
    round_index: int
    prompt: str
    policy_response: str
    prompt_token_ids: Sequence[int]
    output_token_ids: Sequence[int]
    behavior_log_probs: Sequence[float]
    executed_prefix_tokens: int
    action: Mapping[str, Any]
    canvas_feedback: str
    graph_revision: int
    graph_snapshot: Mapping[str, Any]
    policy_version: str
    policy_adapter: Optional[str] = None
    server_weight_version: Optional[str] = None
    graph_snapshot_id: str = ""
    previous_graph_snapshot_id: Optional[str] = None
    executions: Sequence[ExecutionRecord] = field(default_factory=tuple)
    runtime_summary: Mapping[str, Any] = field(default_factory=dict)
    execution_reused: bool = False
    director_request_id: Optional[str] = None
    director_latency_ms: Optional[float] = None
    director_attempt_count: Optional[int] = None
    director_generation_seed: Optional[int] = None
    reconstructed_context: bool = False
    receipt_verified: bool = False
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.executed_prefix_tokens < 0:
            raise ValueError("executed_prefix_tokens must be non-negative")
        if self.executed_prefix_tokens > len(self.output_token_ids):
            raise ValueError("executed prefix exceeds sampled output")
        if len(self.behavior_log_probs) != len(self.output_token_ids):
            raise ValueError("behavior log-prob receipt must match output token count")
        if not all(math.isfinite(float(value)) for value in self.behavior_log_probs):
            raise ValueError("behavior log-prob receipt must contain only finite values")
        if self.policy_adapter is not None and not self.policy_adapter.strip():
            raise ValueError("policy_adapter must be non-empty when supplied")
        if self.server_weight_version is not None and not self.server_weight_version.strip():
            raise ValueError("server_weight_version must be non-empty when supplied")
        if type(self.execution_reused) is not bool:
            raise ValueError("execution_reused must be bool")
        if self.director_request_id is not None and not self.director_request_id.strip():
            raise ValueError("director_request_id must be non-empty when supplied")
        if self.director_latency_ms is not None and self.director_latency_ms < 0:
            raise ValueError("director_latency_ms must be non-negative when supplied")
        if self.director_attempt_count is not None and (
            isinstance(self.director_attempt_count, bool)
            or not isinstance(self.director_attempt_count, int)
            or self.director_attempt_count < 1
        ):
            raise ValueError("director_attempt_count must be positive when supplied")
        if self.director_generation_seed is not None and (
            isinstance(self.director_generation_seed, bool)
            or not isinstance(self.director_generation_seed, int)
            or self.director_generation_seed < 0
        ):
            raise ValueError("director_generation_seed must be non-negative when supplied")
        if not isinstance(self.runtime_summary, Mapping):
            raise ValueError("runtime_summary must be a mapping")

    @property
    def snapshot_receipt_verified(self) -> bool:
        if not self.graph_snapshot_id:
            return False
        expected = stable_id(
            "snapshot",
            {
                "revision": self.graph_revision,
                "graph": self.graph_snapshot,
                "previous_snapshot_id": self.previous_graph_snapshot_id,
            },
        )
        return expected == self.graph_snapshot_id

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["executions"] = [item.to_dict() for item in self.executions]
        return result


@dataclass(frozen=True)
class EvaluationReceipt:
    evaluator_version: str
    valid: bool
    reward: Optional[float]
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid and self.reward is None:
            raise ValueError("valid evaluator receipt requires a reward")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("evaluator reward must be finite")
        if not isinstance(self.details, Mapping):
            raise ValueError("evaluator details must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    task: TaskRecord
    group_id: str
    condition_id: str
    rollout_id: str
    versions: VersionBundle
    turns: Sequence[TurnRecord]
    final_answer: Optional[str]
    evaluation: EvaluationReceipt
    termination_reason: str
    explicit_finish: bool
    condition_satisfied: bool = True
    forced_probe: bool = False
    api_fallback_used: bool = False
    manual_repair_used: bool = False
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    @property
    def group_key(self) -> Tuple[str, str, str]:
        return (self.task.task_id, self.condition_id, self.versions.fingerprint)

    @property
    def terminal_failure(self) -> bool:
        """Whether the natural policy exhausted its edit budget without finish."""

        return bool(
            not self.explicit_finish
            and self.termination_reason == "max_rounds"
            and self.final_answer in (None, "")
        )

    @property
    def natural_policy_terminal(self) -> bool:
        return bool(
            (self.explicit_finish and self.termination_reason == "finish")
            or self.terminal_failure
        )

    def _snapshot_chain_valid(self) -> bool:
        previous: Optional[str] = None
        for index, turn in enumerate(self.turns):
            if not turn.snapshot_receipt_verified:
                return False
            if index > 0 and turn.previous_graph_snapshot_id != previous:
                return False
            previous = turn.graph_snapshot_id
        return True

    @property
    def grpo_eligible(self) -> bool:
        return bool(
            self.task.split == "train"
            and self.natural_policy_terminal
            and self.evaluation.valid
            and self.evaluation.reward is not None
            and (
                not self.terminal_failure
                or float(self.evaluation.reward) == 0.0
            )
            and self.evaluation.evaluator_version == self.versions.evaluator
            and not self.forced_probe
            and not self.api_fallback_used
            and not self.manual_repair_used
            and self.turns
            and self._snapshot_chain_valid()
            and all(
                turn.receipt_verified
                and not turn.reconstructed_context
                and turn.policy_version == self.versions.policy
                and turn.executed_prefix_tokens > 0
                for turn in self.turns
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task": self.task.to_dict(),
            "group_id": self.group_id,
            "condition_id": self.condition_id,
            "rollout_id": self.rollout_id,
            "versions": self.versions.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "final_answer": self.final_answer,
            "evaluation": self.evaluation.to_dict(),
            "termination_reason": self.termination_reason,
            "explicit_finish": self.explicit_finish,
            "terminal_failure": self.terminal_failure,
            "condition_satisfied": self.condition_satisfied,
            "forced_probe": self.forced_probe,
            "api_fallback_used": self.api_fallback_used,
            "manual_repair_used": self.manual_repair_used,
            "grpo_eligible": self.grpo_eligible,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SelectionReceipt:
    selection_id: str
    snapshot_id: str
    strategy: str
    candidate_ids: Sequence[str]
    selected_id: str
    predicted_means: Mapping[str, float]
    predicted_stds: Mapping[str, float]
    sampling_probability: Optional[float]
    posterior_id: str
    feature_schema_version: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeRecord:
    probe_id: str
    problem_id: str
    task_split: str
    snapshot_id: str
    policy_version: str
    state_features: Mapping[str, Any]
    incumbent_action: Mapping[str, Any]
    candidate_action: Mapping[str, Any]
    sampling_probability: float
    incumbent_returns: Sequence[float]
    candidate_returns: Sequence[float]
    executor_versions: Mapping[str, str]
    evaluator_version: str
    feature_schema_version: str
    branch_order: Sequence[str]
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.task_split == "test":
            raise ValueError("test tasks cannot be used as probe evidence")
        if self.task_split not in {"train", "validation"}:
            raise ValueError("probe task_split must be train or validation")
        if not 0 < self.sampling_probability <= 1:
            raise ValueError("sampling_probability must be in (0, 1]")
        if not self.incumbent_returns or not self.candidate_returns:
            raise ValueError("paired probes need results for both arms")
        if not all(
            math.isfinite(float(value))
            for value in (*self.incumbent_returns, *self.candidate_returns)
        ):
            raise ValueError("paired probe returns must be finite")

    @property
    def paired_effect(self) -> float:
        left = sum(self.candidate_returns) / len(self.candidate_returns)
        right = sum(self.incumbent_returns) / len(self.incumbent_returns)
        return float(left - right)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["paired_effect"] = self.paired_effect
        return result


@dataclass(frozen=True)
class PosteriorSnapshotRecord:
    posterior_id: str
    epoch: int
    policy_version: str
    encoder_version: str
    feature_schema_version: str
    mean_parameters: Sequence[float]
    precision_parameters: Sequence[Sequence[float]]
    observation_noise: float
    calibration_quantile: float
    coverage_metrics: Mapping[str, float]
    valid_task_slices: Sequence[str]
    observation_ids: Sequence[str]
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.observation_noise <= 0:
            raise ValueError("observation_noise must be positive")
        numeric = [*self.mean_parameters, self.observation_noise, self.calibration_quantile]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("posterior snapshot scalars must be finite")
        dimension = len(self.mean_parameters)
        if dimension < 1:
            raise ValueError("posterior mean must be non-empty")
        if len(self.precision_parameters) != dimension or any(
            len(row) != dimension for row in self.precision_parameters
        ):
            raise ValueError("posterior precision must be square and match the mean")
        if not all(
            math.isfinite(float(value))
            for row in self.precision_parameters
            for value in row
        ):
            raise ValueError("posterior precision must be finite")
        if len(self.observation_ids) != len(set(self.observation_ids)):
            raise ValueError("posterior observation IDs must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
