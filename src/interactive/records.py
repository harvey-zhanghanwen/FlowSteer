"""Versioned records for the AgentGraph execution and learning planes.

These records deliberately keep natural-policy rollouts, forced probes, and
Skill evidence distinct.  No terminal score is implicitly copied to nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .persistence.ids import stable_id
from .scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
)
from .versioning import VersionBundle


SCHEMA_VERSION = "flowsteer.agentgraph.v2"
VALID_SPLITS = frozenset({"train", "validation", "test"})


def canonical_active_skill_ids(
    values: Sequence[str],
    *,
    field_name: str,
) -> Tuple[str, ...]:
    """Return SkillFlow's sorted, unique ACTIVE-library identifier set.

    This is the dependency-light part of SkillFlow's canonical Skill
    invocation contract.  IDs, rather than rendered Skill instructions, are
    persisted so trajectory validation never needs to inspect model reasoning
    or reconstruct prompt text.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be a sequence of Skill IDs")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain non-empty Skill IDs")
        if value != value.strip():
            raise ValueError(f"{field_name} Skill IDs must not contain whitespace")
        normalized.append(value)
    canonical = tuple(normalized)
    if tuple(sorted(set(canonical))) != canonical:
        raise ValueError(f"{field_name} must be sorted and unique")
    return canonical


def ordered_skill_ids(
    values: Sequence[str],
    *,
    field_name: str,
) -> Tuple[str, ...]:
    """Validate a unique Skill ranking without changing its prompt order."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be a sequence of Skill IDs")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain non-empty Skill IDs")
        if value != value.strip():
            raise ValueError(f"{field_name} Skill IDs must not contain whitespace")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique Skill IDs")
    return tuple(normalized)


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskRecord":
        """Rebuild one persisted task using the same immutable contract."""

        if not isinstance(value, Mapping):
            raise ValueError("serialized task must be a mapping")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("serialized task metadata must be a mapping")
        return cls(
            task_id=value["task_id"],
            question=value["question"],
            ground_truth=value.get("ground_truth"),
            split=value["split"],
            metadata=dict(metadata),
        )


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionRecord":
        if not isinstance(value, Mapping):
            raise ValueError("serialized execution must be a mapping")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("serialized execution metadata must be a mapping")
        return cls(
            execution_id=value["execution_id"],
            experiment_id=value["experiment_id"],
            graph_revision=value["graph_revision"],
            agent_id=value["agent_id"],
            model_id=value["model_id"],
            model_fingerprint=value["model_fingerprint"],
            provider=value["provider"],
            request_hash=value["request_hash"],
            output=value["output"],
            temperature=value["temperature"],
            top_p=value["top_p"],
            max_tokens=value["max_tokens"],
            input_tokens=value.get("input_tokens"),
            output_tokens=value.get("output_tokens"),
            latency_ms=value.get("latency_ms"),
            error_type=value.get("error_type"),
            created_at=value.get("created_at", utc_now()),
            metadata=dict(metadata),
        )


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
    retrieved_skill_ids: Sequence[str] = field(default_factory=tuple)
    visible_skill_ids: Sequence[str] = field(default_factory=tuple)

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
        retrieved_skill_ids = ordered_skill_ids(
            self.retrieved_skill_ids,
            field_name="retrieved_skill_ids",
        )
        visible_skill_ids = ordered_skill_ids(
            self.visible_skill_ids,
            field_name="visible_skill_ids",
        )
        if visible_skill_ids != retrieved_skill_ids:
            raise ValueError(
                "visible Skill IDs must equal the ranked retrieved Skill IDs"
            )
        object.__setattr__(self, "retrieved_skill_ids", retrieved_skill_ids)
        object.__setattr__(self, "visible_skill_ids", visible_skill_ids)

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
        result["retrieved_skill_ids"] = list(self.retrieved_skill_ids)
        result["visible_skill_ids"] = list(self.visible_skill_ids)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TurnRecord":
        if not isinstance(value, Mapping):
            raise ValueError("serialized turn must be a mapping")
        action = value.get("action", {})
        graph_snapshot = value.get("graph_snapshot", {})
        runtime_summary = value.get("runtime_summary", {})
        if not isinstance(action, Mapping):
            raise ValueError("serialized turn action must be a mapping")
        if not isinstance(graph_snapshot, Mapping):
            raise ValueError("serialized turn graph snapshot must be a mapping")
        if not isinstance(runtime_summary, Mapping):
            raise ValueError("serialized turn runtime summary must be a mapping")
        raw_executions = value.get("executions", ())
        if not isinstance(raw_executions, Sequence) or isinstance(
            raw_executions, (str, bytes)
        ):
            raise ValueError("serialized turn executions must be a sequence")
        return cls(
            turn_id=value["turn_id"],
            round_index=value["round_index"],
            prompt=value["prompt"],
            policy_response=value["policy_response"],
            prompt_token_ids=tuple(value.get("prompt_token_ids", ())),
            output_token_ids=tuple(value.get("output_token_ids", ())),
            behavior_log_probs=tuple(value.get("behavior_log_probs", ())),
            executed_prefix_tokens=value["executed_prefix_tokens"],
            action=dict(action),
            canvas_feedback=value["canvas_feedback"],
            graph_revision=value["graph_revision"],
            graph_snapshot=dict(graph_snapshot),
            policy_version=value["policy_version"],
            policy_adapter=value.get("policy_adapter"),
            server_weight_version=value.get("server_weight_version"),
            graph_snapshot_id=value.get("graph_snapshot_id", ""),
            previous_graph_snapshot_id=value.get("previous_graph_snapshot_id"),
            executions=tuple(
                ExecutionRecord.from_dict(item) for item in raw_executions
            ),
            runtime_summary=dict(runtime_summary),
            execution_reused=value.get("execution_reused", False),
            director_request_id=value.get("director_request_id"),
            director_latency_ms=value.get("director_latency_ms"),
            director_attempt_count=value.get("director_attempt_count"),
            director_generation_seed=value.get("director_generation_seed"),
            reconstructed_context=value.get("reconstructed_context", False),
            receipt_verified=value.get("receipt_verified", False),
            created_at=value.get("created_at", utc_now()),
            retrieved_skill_ids=tuple(value.get("retrieved_skill_ids", ())),
            visible_skill_ids=tuple(value.get("visible_skill_ids", ())),
        )


def canonical_invoked_skill_ids(turns: Sequence[TurnRecord]) -> Tuple[str, ...]:
    """Derive credited Skill invocations from admitted execution receipts.

    Executor-side Skill actions are intentionally not admitted in the current
    runtime.  A model may still emit ``ActionKind.SKILL`` and receive the
    public ``skill_action_not_admitted`` observation; that rejected attempt is
    behavior data, not Skill credit.  Until the SkillFlow executor invocation
    boundary is implemented, any successful-looking or explicitly credited
    Skill action fails closed.
    """

    if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
        raise ValueError("turns must be a sequence")
    for turn in turns:
        if not isinstance(turn, TurnRecord):
            raise ValueError("turns must contain TurnRecord values")
        for execution in turn.executions:
            metadata = execution.metadata
            nested_response = metadata.get("response")
            receipt_sources = [metadata]
            if isinstance(nested_response, Mapping):
                receipt_sources.append(nested_response)
            for receipt_source in receipt_sources:
                raw_credited = receipt_source.get("invoked_skill_ids", ())
                credited = canonical_active_skill_ids(
                    raw_credited,  # type: ignore[arg-type]
                    field_name="execution invoked_skill_ids",
                )
                if credited:
                    raise ValueError(
                        "Executor Skill invocation credit is not admitted by this runtime"
                    )
                raw_trace = receipt_source.get("react_trace", ())
                if not isinstance(raw_trace, Sequence) or isinstance(
                    raw_trace, (str, bytes)
                ):
                    continue
                for entry in raw_trace:
                    if not isinstance(entry, Mapping):
                        continue
                    action_text = entry.get("action_text")
                    if not isinstance(action_text, str):
                        continue
                    try:
                        action = json.loads(action_text)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(action, Mapping) or set(action) != {
                        "arguments",
                        "kind",
                        "name",
                        "resource_id",
                        "skill_id",
                    }:
                        continue
                    kind = action.get("kind")
                    name = action.get("name")
                    resource_id = action.get("resource_id")
                    skill_id = action.get("skill_id")
                    if kind not in {"tool", "skill", "complete"}:
                        continue
                    if not isinstance(name, str) or not name:
                        continue
                    if kind in {"tool", "skill"} and (
                        not isinstance(resource_id, str) or not resource_id
                    ):
                        continue
                    if kind == "complete" and resource_id is not None:
                        continue
                    if kind != "skill":
                        continue
                    if not isinstance(skill_id, str) or not skill_id:
                        continue
                    if (
                        entry.get("observation_status") == "schema_invalid"
                        and entry.get("public_error_code")
                        == "skill_action_not_admitted"
                    ):
                        continue
                    raise ValueError(
                        "ActionKind.SKILL has no admitted invocation receipt"
                    )
    return ()


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationReceipt":
        if not isinstance(value, Mapping):
            raise ValueError("serialized evaluation receipt must be a mapping")
        metrics = value.get("metrics", {})
        details = value.get("details", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("serialized evaluation metrics must be a mapping")
        if not isinstance(details, Mapping):
            raise ValueError("serialized evaluation details must be a mapping")
        return cls(
            evaluator_version=value["evaluator_version"],
            valid=value["valid"],
            reward=value.get("reward"),
            metrics=dict(metrics),
            reason=value.get("reason", ""),
            details=dict(details),
        )


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
    director_sampling: Mapping[str, Any] = field(default_factory=dict)
    condition_satisfied: bool = True
    forced_probe: bool = False
    api_fallback_used: bool = False
    manual_repair_used: bool = False
    valid_lineage_fallback_used: bool = False
    valid_lineage_fallback_receipt: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION
    active_skill_ids: Sequence[str] = field(default_factory=tuple)
    retrieved_skill_ids: Sequence[str] = field(default_factory=tuple)
    invoked_skill_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.director_sampling, Mapping):
            raise ValueError("director_sampling must be a mapping")
        if type(self.valid_lineage_fallback_used) is not bool:
            raise ValueError("valid_lineage_fallback_used must be bool")
        if not isinstance(self.valid_lineage_fallback_receipt, Mapping):
            raise ValueError("valid_lineage_fallback_receipt must be a mapping")
        fallback_receipt = dict(self.valid_lineage_fallback_receipt)
        if self.valid_lineage_fallback_used:
            if (
                self.explicit_finish
                or self.termination_reason != "max_rounds"
                or not isinstance(self.final_answer, str)
                or not self.final_answer.strip()
                or not fallback_receipt
            ):
                raise ValueError(
                    "valid lineage fallback requires a non-empty max_rounds "
                    "answer without explicit finish and a fallback receipt"
                )
        elif fallback_receipt:
            raise ValueError(
                "valid_lineage_fallback_receipt requires fallback_used=true"
            )
        object.__setattr__(
            self,
            "valid_lineage_fallback_receipt",
            fallback_receipt,
        )
        initial_retrieved = (
            tuple(self.turns[0].retrieved_skill_ids) if self.turns else ()
        )
        retrieved_skill_ids = ordered_skill_ids(
            self.retrieved_skill_ids,
            field_name="trajectory retrieved_skill_ids",
        )
        if retrieved_skill_ids != initial_retrieved:
            raise ValueError(
                "trajectory retrieved Skill IDs differ from the initial Director turn"
            )
        active_skill_ids = canonical_active_skill_ids(
            self.active_skill_ids,
            field_name="active_skill_ids",
        )
        all_turn_retrieved = {
            skill_id
            for turn in self.turns
            for skill_id in turn.retrieved_skill_ids
        }
        if not all_turn_retrieved.issubset(active_skill_ids):
            raise ValueError(
                "every turn's retrieved Skill IDs must be a subset of active Skill IDs"
            )
        invoked_skill_ids = canonical_active_skill_ids(
            self.invoked_skill_ids,
            field_name="invoked_skill_ids",
        )
        canonical_invoked = canonical_invoked_skill_ids(self.turns)
        if invoked_skill_ids != canonical_invoked:
            raise ValueError(
                "trajectory invoked Skill IDs differ from admitted execution receipts"
            )
        if not set(invoked_skill_ids).issubset(all_turn_retrieved):
            raise ValueError(
                "invoked Skill IDs must be a subset of turn-retrieved Skill IDs"
            )
        object.__setattr__(self, "active_skill_ids", active_skill_ids)
        object.__setattr__(self, "retrieved_skill_ids", retrieved_skill_ids)
        object.__setattr__(self, "invoked_skill_ids", invoked_skill_ids)

    @property
    def skill_receipt_verified(self) -> bool:
        """Whether exposure and invocation IDs match the persisted turns."""

        try:
            initial_retrieved = (
                tuple(self.turns[0].retrieved_skill_ids) if self.turns else ()
            )
            all_turn_retrieved = {
                skill_id
                for turn in self.turns
                for skill_id in turn.retrieved_skill_ids
            }
            return bool(
                tuple(self.retrieved_skill_ids) == initial_retrieved
                and all_turn_retrieved.issubset(self.active_skill_ids)
                and tuple(self.invoked_skill_ids)
                == canonical_invoked_skill_ids(self.turns)
                and set(self.invoked_skill_ids).issubset(all_turn_retrieved)
            )
        except (TypeError, ValueError):
            return False

    @property
    def group_key(self) -> Tuple[str, str, str]:
        return (self.task.task_id, self.condition_id, self.versions.fingerprint)

    @property
    def terminal_failure(self) -> bool:
        """Whether the natural policy exhausted its edit budget without finish."""

        return bool(
            not self.explicit_finish
            and self.termination_reason == "max_rounds"
            and (
                self.final_answer in (None, "")
                or self.valid_lineage_fallback_used
            )
        )

    @property
    def natural_policy_terminal(self) -> bool:
        return bool(
            (self.explicit_finish and self.termination_reason == "finish")
            or self.terminal_failure
        )

    @property
    def sampling_receipt_verified(self) -> bool:
        receipt = self.director_sampling
        if set(receipt) != {"algorithm", "base_seed", "coordinate", "phase"}:
            return False
        if receipt.get("algorithm") != SCIENTIFIC_SAMPLING_ALGORITHM:
            return False
        if receipt.get("phase") != GenerationPhase.ACTION.value:
            return False
        base_seed = receipt.get("base_seed")
        if type(base_seed) is not int or not 0 <= base_seed < 2**64:
            return False
        try:
            coordinate = ScientificSamplingCoordinate.from_value(
                receipt.get("coordinate")
            )
        except (TypeError, ValueError):
            return False
        if coordinate.task_id != self.task.task_id:
            return False
        if coordinate.sampling_schedule_hash != scientific_sampling_schedule_hash(
            base_seed=base_seed
        ):
            return False
        return all(
            turn.director_generation_seed
            == derive_generation_seed(
                base_seed=base_seed,
                coordinate=coordinate,
                step_index=turn.round_index + 1,
                phase=GenerationPhase.ACTION,
            )
            for turn in self.turns
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
            and self.sampling_receipt_verified
            and not self.forced_probe
            and not self.api_fallback_used
            and not self.manual_repair_used
            and not self.valid_lineage_fallback_used
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
            "director_sampling": dict(self.director_sampling),
            "sampling_receipt_verified": self.sampling_receipt_verified,
            "condition_satisfied": self.condition_satisfied,
            "forced_probe": self.forced_probe,
            "api_fallback_used": self.api_fallback_used,
            "manual_repair_used": self.manual_repair_used,
            "valid_lineage_fallback_used": self.valid_lineage_fallback_used,
            "valid_lineage_fallback_receipt": dict(
                self.valid_lineage_fallback_receipt
            ),
            "active_skill_ids": list(self.active_skill_ids),
            "retrieved_skill_ids": list(self.retrieved_skill_ids),
            "invoked_skill_ids": list(self.invoked_skill_ids),
            "skill_receipt_verified": self.skill_receipt_verified,
            "grpo_eligible": self.grpo_eligible,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        """Rebuild and revalidate a persisted rollout for exact resume.

        SkillFlow resumes persisted trajectories through an explicit
        deserialization boundary.  This project adaptation additionally checks
        every serialized derived eligibility flag instead of trusting it.
        """

        if not isinstance(value, Mapping):
            raise ValueError("serialized trajectory must be a mapping")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("serialized trajectory schema version differs")
        raw_turns = value.get("turns", ())
        if not isinstance(raw_turns, Sequence) or isinstance(raw_turns, (str, bytes)):
            raise ValueError("serialized trajectory turns must be a sequence")
        director_sampling = value.get("director_sampling", {})
        if not isinstance(director_sampling, Mapping):
            raise ValueError("serialized Director sampling receipt must be a mapping")
        fallback_receipt = value.get("valid_lineage_fallback_receipt", {})
        if not isinstance(fallback_receipt, Mapping):
            raise ValueError(
                "serialized valid lineage fallback receipt must be a mapping"
            )
        record = cls(
            trajectory_id=value["trajectory_id"],
            task=TaskRecord.from_dict(value["task"]),
            group_id=value["group_id"],
            condition_id=value["condition_id"],
            rollout_id=value["rollout_id"],
            versions=VersionBundle.from_dict(value["versions"]),
            turns=tuple(TurnRecord.from_dict(item) for item in raw_turns),
            final_answer=value.get("final_answer"),
            evaluation=EvaluationReceipt.from_dict(value["evaluation"]),
            termination_reason=value["termination_reason"],
            explicit_finish=value["explicit_finish"],
            director_sampling=dict(director_sampling),
            condition_satisfied=value.get("condition_satisfied", True),
            forced_probe=value.get("forced_probe", False),
            api_fallback_used=value.get("api_fallback_used", False),
            manual_repair_used=value.get("manual_repair_used", False),
            valid_lineage_fallback_used=value.get(
                "valid_lineage_fallback_used",
                False,
            ),
            valid_lineage_fallback_receipt=dict(fallback_receipt),
            created_at=value.get("created_at", utc_now()),
            schema_version=value["schema_version"],
            active_skill_ids=tuple(value.get("active_skill_ids", ())),
            retrieved_skill_ids=tuple(value.get("retrieved_skill_ids", ())),
            invoked_skill_ids=tuple(value.get("invoked_skill_ids", ())),
        )
        derived = {
            "terminal_failure": record.terminal_failure,
            "sampling_receipt_verified": record.sampling_receipt_verified,
            "skill_receipt_verified": record.skill_receipt_verified,
            "grpo_eligible": record.grpo_eligible,
        }
        for name, expected in derived.items():
            if name in value and value[name] != expected:
                raise ValueError(f"serialized trajectory derived field {name!r} differs")
        return record


@dataclass(frozen=True)
class CommunicationDiagnosticRecord:
    """Execution-only communication ablation, never a policy trajectory."""

    diagnostic_id: str
    pair_id: str
    source_trajectory_id: str
    task: TaskRecord
    condition_id: str
    communication_condition: str
    versions: VersionBundle
    graph_snapshot: Mapping[str, Any]
    output_agent_id: str
    runtime_run_id: str
    executions: Sequence[ExecutionRecord]
    final_answer: str
    evaluation: EvaluationReceipt
    mask_applied_call_ids: Sequence[str] = field(default_factory=tuple)
    mask_scope: str = "all_inter_agent_content"
    created_at: str = field(default_factory=utc_now)
    schema_version: str = "flowsteer.agentgraph.communication_diagnostic.v1"
    diagnostic_only: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("diagnostic_id", self.diagnostic_id),
            ("pair_id", self.pair_id),
            ("source_trajectory_id", self.source_trajectory_id),
            ("condition_id", self.condition_id),
            ("output_agent_id", self.output_agent_id),
            ("runtime_run_id", self.runtime_run_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.communication_condition not in {"normal", "upstream_masked"}:
            raise ValueError(
                "communication_condition must be normal or upstream_masked"
            )
        if self.condition_id != self.communication_condition:
            raise ValueError("condition_id must match communication_condition")
        if self.task.split not in {"validation", "test"}:
            raise ValueError("communication diagnostics require a held-out task")

    @property
    def grpo_eligible(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "diagnostic_id": self.diagnostic_id,
            "pair_id": self.pair_id,
            "source_trajectory_id": self.source_trajectory_id,
            "task": self.task.to_dict(),
            "condition_id": self.condition_id,
            "communication_condition": self.communication_condition,
            "versions": self.versions.to_dict(),
            "graph_snapshot": dict(self.graph_snapshot),
            "output_agent_id": self.output_agent_id,
            "runtime_run_id": self.runtime_run_id,
            "executions": [item.to_dict() for item in self.executions],
            "final_answer": self.final_answer,
            "evaluation": self.evaluation.to_dict(),
            "mask_applied_call_ids": list(self.mask_applied_call_ids),
            "mask_scope": self.mask_scope,
            "diagnostic_only": self.diagnostic_only,
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
