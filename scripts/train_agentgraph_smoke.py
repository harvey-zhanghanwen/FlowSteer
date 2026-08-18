#!/usr/bin/env python3
"""Run the bounded 7x2 AgentGraph Qwen3.5 smoke-training transaction."""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import (
    ConfigurationError,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.director import AgentGraphOrchestrator
from src.interactive.grpo_objective import same_condition_advantages
from src.interactive.hotpot_training_schedule import (
    FrozenHotpotTrainingSchedule,
    HotpotTrainingCursorState,
    HotpotTrainingProgress,
)
from src.interactive.joint_qa_training_schedule import (
    FrozenJointQATrainingSchedule,
    JointQATrainingCursorState,
    JointQATrainingProgress,
)
from src.interactive.openai_gateway import OpenAICompatibleGateway
from src.interactive.persistence import EvidenceStore
from src.interactive.qa_retrieval import (
    SkillFlowQARetriever,
    augment_task_with_retrieval,
    build_keyword_query,
    receipt_from_mapping,
)
from src.interactive.policy_sync import (
    PolicySyncConfig,
    PolicySyncError,
    SGLangPolicyPublisher,
)
from src.interactive.records import TaskRecord, TrajectoryRecord
from src.interactive.rollout_collector import (
    AgentGraphRolloutCollector,
    RolloutGate,
    SGLangReceiptDirectorClient,
)
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.skills import SkillEvidencePipeline, SkillQuery, SkillStore
from src.interactive.smoke_trainer import (
    Qwen35OnePassSmokeTrainer,
    SmokeTrainerConfig,
    trajectory_to_grpo,
)
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import (
    EvaluationOutcome,
    HEALTHBENCH_EVALUATOR_VERSION,
    HOTPOTQA_ANSWER_EVALUATOR_VERSION,
    RAGEN_EVALUATOR_VERSION,
    SKILLFLOW_REWARD_VERSION,
    SWEBENCH_EVALUATOR_VERSION,
    TRIVIAQA_ANSWER_EVALUATOR_VERSION,
    evaluate_task,
)
from src.interactive.versioning import VersionBundle


PROMPT_VERSION = "agentgraph.director.minimal.v1"
TOOL_VERSION = "agentgraph.atomic-actions.v1"

EXPECTED_SOURCE_ORDER = (
    "hotpotqa",
    "triviaqa",
    "aime_2026",
    "healthbench_professional",
    "webshop",
    "alfworld",
    "swe_bench",
)


class SmokeRunError(RuntimeError):
    """The bounded run could not prove a complete training transaction."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _is_hotpot_micro(config: Mapping[str, Any]) -> bool:
    experiment = _mapping(config.get("experiment"), "experiment")
    return experiment.get("phase") == "hotpotqa_micro_training"


def _is_joint_qa_micro(config: Mapping[str, Any]) -> bool:
    experiment = _mapping(config.get("experiment"), "experiment")
    return experiment.get("phase") == "joint_qa_micro_training"


def _is_frozen_micro(config: Mapping[str, Any]) -> bool:
    return _is_hotpot_micro(config) or _is_joint_qa_micro(config)


def validate_smoke_bounds(config: Mapping[str, Any]) -> None:
    """Reject configs that expand either supported one-update transaction."""

    validate_agent_graph_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    hotpot_micro = _is_hotpot_micro(config)
    joint_qa_micro = _is_joint_qa_micro(config)
    frozen_micro = hotpot_micro or joint_qa_micro
    selection_key = (
        "hotpot_micro" if hotpot_micro else "joint_qa_micro" if joint_qa_micro else "smoke"
    )
    selection_name = f"data.{selection_key}"
    selection = _mapping(
        data.get(selection_key),
        selection_name,
    )
    grpo = _mapping(config.get("grpo"), "grpo")
    director = _mapping(config.get("director"), "director")
    policy_sync = _mapping(config.get("policy_sync"), "policy_sync")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")

    checks = {
        "experiment.phase": experiment.get("phase")
        == (
            "hotpotqa_micro_training"
            if hotpot_micro
            else "joint_qa_micro_training"
            if joint_qa_micro
            else "smoke_training"
        ),
        "experiment.training_enabled": experiment.get("training_enabled") is True,
        f"{selection_name}.split": selection.get("split") == "train",
        f"{selection_name}.selection": selection.get("selection")
        == (
            "frozen_hotpot_schedule"
            if hotpot_micro
            else "frozen_joint_qa_schedule"
            if joint_qa_micro
            else "sequential_per_source"
        ),
        f"{selection_name}.expected_total_tasks": selection.get(
            "expected_total_tasks"
        )
        == (1 if hotpot_micro else 2 if joint_qa_micro else 14),
        "grpo.enabled": grpo.get("enabled") is True,
        "grpo.samples_per_problem": type(grpo.get("samples_per_problem")) is int
        and int(grpo["samples_per_problem"]) >= 2
        and (
            hotpot_micro
            or (joint_qa_micro and grpo.get("samples_per_problem") == 8)
            or (not frozen_micro and grpo.get("samples_per_problem") == 2)
        ),
        "grpo.expected_rollout_count": grpo.get("expected_rollout_count")
        == (
            grpo.get("samples_per_problem")
            if hotpot_micro
            else 16
            if joint_qa_micro
            else 28
        ),
        "grpo.optimization_passes_per_rollout_batch": (
            grpo.get("optimization_passes_per_rollout_batch") == 1
        ),
        "grpo.max_optimizer_updates": grpo.get("max_optimizer_updates") == 1,
        "grpo.terminal_task_reward_only": grpo.get("terminal_task_reward_only") is True,
        "policy_sync.enabled": policy_sync.get("enabled") is True,
        "policy_sync.post_update_canary_count": (
            policy_sync.get("post_update_canary_count")
            == (2 if joint_qa_micro else 1)
        ),
        "evaluation.healthbench_judge_model": frozen_micro
        or bool(str(evaluation.get("healthbench_judge_model", "")).strip()),
        "evaluation.max_environment_steps": (
            evaluation.get("max_environment_steps") == 12
        ),
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "director.temperature": float(director.get("temperature", -1)) == 1.0,
        "director.top_p": float(director.get("top_p", -1)) == 1.0,
        "director.top_k": director.get("top_k") == -1,
        "exploration.enabled": exploration.get("enabled") is False,
        "grpo.structural_reward": float(grpo.get("structural_reward", -1.0))
        == 0.0,
        "grpo.exploration_reward": float(grpo.get("exploration_reward", -1.0))
        == 0.0,
        "grpo.skill_usage_reward": float(grpo.get("skill_usage_reward", -1.0))
        == 0.0,
        "skills.enabled": hotpot_micro or skills.get("enabled") is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "smoke config violates fixed bounds: " + ", ".join(failed)
        )

    if frozen_micro:
        expected_dataset_keys = (
            ("hotpotqa",)
            if hotpot_micro
            else ("hotpotqa", "triviaqa")
        )
        configured_dataset_keys = (
            (selection.get("dataset_key"),)
            if hotpot_micro
            else tuple(str(value) for value in selection.get("dataset_keys", ()))
        )
        if configured_dataset_keys != expected_dataset_keys:
            raise ConfigurationError(
                f"{selection_name} dataset keys must be {expected_dataset_keys!r}"
            )
        for field_name in ("schedule_path", "cursor_path", "next_cursor_path"):
            if not str(selection.get(field_name, "")).strip():
                raise ConfigurationError(
                    f"{selection_name}.{field_name} must be non-empty"
                )
        if len(
            {
                str(selection["schedule_path"]),
                str(selection["cursor_path"]),
                str(selection["next_cursor_path"]),
            }
        ) != 3:
            raise ConfigurationError(
                "frozen schedule, cursor, and next cursor paths must differ"
            )
        if joint_qa_micro:
            retrieval = _mapping(
                selection.get("retrieval"),
                f"{selection_name}.retrieval",
            )
            if (
                retrieval.get("enabled") is not True
                or retrieval.get("mode")
                != "deterministic_question_query_prefetch"
                or int(retrieval.get("search_limit", 0)) < 1
            ):
                raise ConfigurationError(
                    "joint QA micro-training requires the frozen SkillFlow public retrieval condition"
                )
            terminal = _mapping(
                _mapping(config["agent_graph"], "agent_graph").get(
                    "terminal_protocol_by_source"
                ),
                "agent_graph.terminal_protocol_by_source",
            )
            if any(
                terminal.get(source) != "exact_single_answer_tag"
                for source in expected_dataset_keys
            ):
                raise ConfigurationError(
                    "joint QA micro-training requires exact_single_answer_tag for both datasets"
                )
    else:
        if selection.get("tasks_per_dataset") != 2:
            raise ConfigurationError("data.smoke.tasks_per_dataset must be 2")
        source_order = tuple(
            str(value) for value in selection.get("source_order", ())
        )
        if source_order != EXPECTED_SOURCE_ORDER:
            raise ConfigurationError(
                "data.smoke.source_order must contain the fixed seven-source order"
            )
    for field_name in (
        "behavior_policy_version",
        "updated_policy_version",
        "expected_server_weight_version",
    ):
        value = director.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"director.{field_name} must be non-empty")
    if director["behavior_policy_version"] == director["updated_policy_version"]:
        raise ConfigurationError("Director behavior and updated policy versions must differ")
    oom = _mapping(_mapping(config["gpu"], "gpu")["oom_policy"], "gpu.oom_policy")
    if tuple(oom.get("micro_batch_schedule", ())) != (4, 2, 1):
        raise ConfigurationError("gpu.oom_policy.micro_batch_schedule must be [4, 2, 1]")


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _dataset_key(task: TaskRecord) -> str:
    value = task.metadata.get("dataset_key")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task {task.task_id!r} has no metadata.dataset_key")
    return value.strip()


def _skill_context_tag(namespace: str, field_name: str, value: Any) -> str:
    """Encode one exact, decision-time Skill condition as a namespaced tag."""

    return (
        f"{namespace}.{field_name}="
        + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _skill_query_tags(
    task: TaskRecord,
    graph: AgentGraph,
    *,
    task_family: str,
    graph_stage: str,
    validation_issue_codes: Sequence[str],
) -> tuple[str, ...]:
    """Describe only the task context and current AgentGraph prefix.

    SkillFlow retrieves against the current task query and task type.  This
    evidence-gated path keeps its existing exact ``SkillQuery.tags`` matching,
    and supplies the context product required by the project design: task
    context, prefix graph, optional ``role_family``, model, relation motif,
    and graph position.  No final-graph or evaluator fields are read here.
    """

    tags = {str(code) for code in validation_issue_codes}

    task_context: dict[str, Any] = {
        "dataset_key": _dataset_key(task),
        "task_family": task_family,
    }
    metadata_task_type = task.metadata.get("task_type")
    if isinstance(metadata_task_type, str) and metadata_task_type.strip():
        task_context["task_type"] = metadata_task_type.strip()
    skillflow = task.metadata.get("skillflow")
    if isinstance(skillflow, Mapping):
        skillflow_task_type = skillflow.get("task_type")
        if (
            "task_type" not in task_context
            and isinstance(skillflow_task_type, str)
            and skillflow_task_type.strip()
        ):
            task_context["task_type"] = skillflow_task_type.strip()
        extra = skillflow.get("extra")
        if isinstance(extra, Mapping):
            for source_key, condition_key in (
                ("type", "task_subtype"),
                ("level", "difficulty"),
            ):
                value = extra.get(source_key)
                if isinstance(value, str) and value.strip():
                    task_context[condition_key] = value.strip()
    for field_name, value in task_context.items():
        tags.add(_skill_context_tag("task_context", field_name, value))

    statistics = graph.topology_statistics()
    prefix_fields = {
        "graph_stage": graph_stage,
        "topology_family": statistics["topology_family"],
        "structural_depth": statistics["structural_depth"],
        "output_state": (
            "set" if graph.output_agent_id is not None else "unset"
        ),
    }
    for field_name, value in prefix_fields.items():
        tags.add(_skill_context_tag("graph_prefix", field_name, value))
    relation_motifs: set[str] = set()
    pair_count = len(graph.nodes) * (len(graph.nodes) - 1) // 2
    if pair_count > len(graph.relations):
        relation_motifs.add("independent")
    for relation in graph.relations:
        relation_motifs.add(
            "bidirectional" if relation.bits.is_bidirectional else "unidirectional"
        )
    for motif in relation_motifs:
        tags.add(_skill_context_tag("relation_motif", "kind", motif))

    roots = set(statistics["root_agent_ids"])
    sinks = set(statistics["sink_agent_ids"])
    fan_in = set(statistics["fan_in_agent_ids"])
    fan_out = set(statistics["fan_out_agent_ids"])
    for node in graph.nodes:
        tags.add(_skill_context_tag("model", "model_id", node.model_id))
        role_family = node.role_family
        if role_family is not None:
            tags.add(_skill_context_tag("role_family", "value", role_family))

        positions: set[str] = set()
        if node.id in roots:
            positions.add("root")
        if node.id in sinks:
            positions.add("sink")
        if node.id not in roots and node.id not in sinks:
            positions.add("intermediate")
        if node.id in fan_in:
            positions.add("fan_in")
        if node.id in fan_out:
            positions.add("fan_out")
        if node.id == graph.output_agent_id:
            positions.add("output")
        for position in positions:
            tags.add(_skill_context_tag("graph_position", "kind", position))

        local_relation_motifs: set[str] = set()
        if len(graph.nodes) > 1:
            related_ids = {
                endpoint
                for relation in graph.relations
                if node.id in (relation.source_id, relation.target_id)
                for endpoint in (relation.source_id, relation.target_id)
                if endpoint != node.id
            }
            if len(related_ids) < len(graph.nodes) - 1:
                local_relation_motifs.add("independent")
        for relation in graph.relations:
            if node.id in (relation.source_id, relation.target_id):
                local_relation_motifs.add(
                    "bidirectional"
                    if relation.bits.is_bidirectional
                    else "unidirectional"
                )
        if role_family is not None:
            for relation_motif in local_relation_motifs:
                for position in positions:
                    tags.add(
                        _skill_context_tag(
                            "agent_context",
                            "role_model_relation_position",
                            {
                                "role_family": role_family,
                                "model_id": node.model_id,
                                "relation_motif": relation_motif,
                                "graph_position": position,
                            },
                        )
                    )

    return tuple(sorted(tags))


def _base_task_id(task: TaskRecord) -> str:
    sampling = task.metadata.get("sampling", {})
    if isinstance(sampling, Mapping):
        value = sampling.get("base_task_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return task.task_id


def select_smoke_tasks(
    tasks: Sequence[TaskRecord] | Any,
    *,
    source_order: Sequence[str],
    per_source: int,
    require_unique_base_tasks: bool,
    skip_per_source: int = 0,
    expected_split: str = "train",
) -> tuple[TaskRecord, ...]:
    """Select the first N records per dataset key in config order."""

    if type(per_source) is not int or per_source <= 0:
        raise ValueError("per_source must be a positive integer")
    if type(skip_per_source) is not int or skip_per_source < 0:
        raise ValueError("skip_per_source must be a non-negative integer")
    if not isinstance(expected_split, str) or not expected_split.strip():
        raise ValueError("expected_split must be a non-empty string")
    ordered_sources = tuple(str(value).strip() for value in source_order)
    if not ordered_sources or any(not value for value in ordered_sources):
        raise ValueError("source_order must contain non-empty dataset keys")
    if len(ordered_sources) != len(set(ordered_sources)):
        raise ValueError("source_order contains duplicate dataset keys")

    selected: dict[str, list[TaskRecord]] = {source: [] for source in ordered_sources}
    base_ids: dict[str, set[str]] = {source: set() for source in ordered_sources}
    skipped: dict[str, int] = {source: 0 for source in ordered_sources}
    for task in tasks:
        if task.split != expected_split:
            raise ValueError(
                f"bounded task {task.task_id!r} is not in the {expected_split} split"
            )
        source = _dataset_key(task)
        if source not in selected or len(selected[source]) >= per_source:
            continue
        if skipped[source] < skip_per_source:
            skipped[source] += 1
            continue
        base_id = _base_task_id(task)
        if require_unique_base_tasks and base_id in base_ids[source]:
            continue
        selected[source].append(task)
        base_ids[source].add(base_id)
        if all(len(items) == per_source for items in selected.values()):
            break

    missing = {
        source: per_source - len(items)
        for source, items in selected.items()
        if len(items) != per_source
    }
    if missing:
        detail = ", ".join(f"{source}: {count}" for source, count in missing.items())
        raise ValueError(f"insufficient smoke tasks by source ({detail})")
    return tuple(task for source in ordered_sources for task in selected[source])


def evaluator_version_for(task: TaskRecord) -> str:
    source = _dataset_key(task)
    if source == "hotpotqa":
        return HOTPOTQA_ANSWER_EVALUATOR_VERSION
    if source == "triviaqa":
        return TRIVIAQA_ANSWER_EVALUATOR_VERSION
    if source == "aime_2026":
        return SKILLFLOW_REWARD_VERSION
    if source == "healthbench_professional":
        return HEALTHBENCH_EVALUATOR_VERSION
    if source in {"webshop", "alfworld"}:
        return RAGEN_EVALUATOR_VERSION
    if source == "swe_bench":
        return SWEBENCH_EVALUATOR_VERSION
    raise ValueError(f"unsupported smoke dataset key: {source}")


def version_bundle_for(
    task: TaskRecord,
    *,
    policy_version: str,
    model_catalog_version: str,
    prompt_version: str = PROMPT_VERSION,
    tool_version: str = TOOL_VERSION,
    encoder_version: str = "none",
    feature_schema_version: str = "none",
    posterior_version: str = "none",
    skill_library_version: str = "none",
) -> VersionBundle:
    return VersionBundle(
        policy=policy_version,
        model_catalog=model_catalog_version,
        evaluator=evaluator_version_for(task),
        prompt=prompt_version,
        tool=tool_version,
        encoder=encoder_version,
        feature_schema=feature_schema_version,
        posterior=posterior_version,
        skill_library=skill_library_version,
    )


def _validate_resumed_initial_rollouts(
    records: Sequence[TrajectoryRecord],
    expected_jobs: Sequence[tuple[TaskRecord, int, VersionBundle]],
    *,
    condition_id: str,
    sampling_anchor_ordinal: int,
    behavior_adapter_name: Optional[str],
    expected_server_weight_version: str,
) -> None:
    """Fail closed unless persisted rollouts are the exact frozen job list."""

    if len(records) != len(expected_jobs):
        raise SmokeRunError("resumed rollout count differs from the frozen job list")
    if len({record.trajectory_id for record in records}) != len(records):
        raise SmokeRunError("resumed rollout artifact contains duplicate trajectories")
    if len({record.rollout_id for record in records}) != len(records):
        raise SmokeRunError("resumed rollout artifact contains duplicate rollout IDs")
    for index, (record, expected) in enumerate(
        zip(records, expected_jobs, strict=True)
    ):
        task, rollout_ordinal, versions = expected
        if record.task.to_dict() != task.to_dict():
            raise SmokeRunError(f"resumed rollout {index} has the wrong frozen task")
        if record.versions != versions:
            raise SmokeRunError(f"resumed rollout {index} has the wrong version bundle")
        if record.condition_id != condition_id:
            raise SmokeRunError(f"resumed rollout {index} has the wrong condition")
        expected_group = f"{task.task_id}:{condition_id}:{versions.policy}"
        if record.group_id != expected_group:
            raise SmokeRunError(f"resumed rollout {index} has the wrong GRPO group")
        expected_rollout_id = f"{expected_group}:rollout:{rollout_ordinal:04d}"
        if record.rollout_id != expected_rollout_id:
            raise SmokeRunError(f"resumed rollout {index} has the wrong rollout ID")
        if task.split != "train":
            raise SmokeRunError(f"resumed rollout {index} is not a training sample")
        if not record.natural_policy_terminal:
            raise SmokeRunError(f"resumed rollout {index} is not a natural terminal")
        if (
            not record.condition_satisfied
            or record.forced_probe
            or record.api_fallback_used
            or record.manual_repair_used
        ):
            raise SmokeRunError(
                f"resumed rollout {index} is not an unmodified natural-policy sample"
            )
        if (
            not record.evaluation.valid
            or record.evaluation.reward is None
            or record.evaluation.evaluator_version != versions.evaluator
        ):
            raise SmokeRunError(f"resumed rollout {index} has an invalid evaluator")
        if not record.sampling_receipt_verified:
            raise SmokeRunError(f"resumed rollout {index} has an invalid sampling receipt")
        if not record.turns:
            raise SmokeRunError(f"resumed rollout {index} contains no policy turns")
        for turn in record.turns:
            if (
                not turn.prompt
                or not turn.policy_response
                or not turn.output_token_ids
                or not turn.receipt_verified
                or turn.reconstructed_context
                or turn.policy_version != versions.policy
                or turn.policy_adapter != behavior_adapter_name
                or turn.server_weight_version != expected_server_weight_version
            ):
                raise SmokeRunError(
                    f"resumed rollout {index} has an invalid exact policy receipt"
                )
            if turn.executed_prefix_tokens == 0 and (
                bool(turn.action)
                or not turn.canvas_feedback.startswith("invalid action:")
            ):
                raise SmokeRunError(
                    f"resumed rollout {index} has an unexplained zero action mask"
                )
        coordinate = ScientificSamplingCoordinate.from_value(
            record.director_sampling["coordinate"]
        )
        if (
            coordinate.task_id != task.task_id
            or coordinate.sequence_position != rollout_ordinal
            or coordinate.schedule_purpose != condition_id
            or coordinate.optimizer_step_or_anchor_ordinal
            != sampling_anchor_ordinal
        ):
            raise SmokeRunError(
                f"resumed rollout {index} has the wrong frozen sampling coordinate"
            )
    if sum(record.grpo_eligible for record in records) < 2:
        raise SmokeRunError("resumed rollout batch has fewer than two eligible samples")


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SmokeRunError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise SmokeRunError(f"{label} must be a mapping")
    return value


def _validate_resume_preconditions(
    *,
    paths: Mapping[str, Path],
    next_cursor_path: Optional[Path],
) -> Mapping[str, Any]:
    """Prove the prior attempt persisted no model or schedule state change.

    SkillFlow resumes only from an explicit persisted boundary.  This bounded
    adaptation accepts exactly the known zero-update failure state: no learner
    checkpoint, optimizer state, policy publication, canary, or next cursor.
    """

    prior_manifest = _read_json_mapping(
        paths["manifest"], label="prior failed training manifest"
    )
    prior_status = prior_manifest.get("status")
    retry_after_runtime_failure = prior_status == "failed_training"
    if prior_status not in {"failed_no_optimizer_update", "failed_training"}:
        raise SmokeRunError(
            "strict rollout resume requires a proven zero-persistence failure manifest"
        )
    embedded_preconditions = prior_manifest.get("resume_preconditions")
    fresh_runtime_failure = False
    if retry_after_runtime_failure:
        initial_source = prior_manifest.get("initial_rollout_source")
        nested_strict_resume_failure = (
            prior_manifest.get("resume_initial_rollouts_requested") is True
            and isinstance(initial_source, Mapping)
            and initial_source.get("mode") == "strict_persisted_resume"
            and isinstance(embedded_preconditions, Mapping)
            and embedded_preconditions.get("optimizer_updates") == 0
        )
        fresh_runtime_failure = bool(
            prior_manifest.get("resume_initial_rollouts_requested") is False
            and isinstance(initial_source, Mapping)
            and initial_source.get("mode") == "live_collection"
            and int(initial_source.get("new_collections", 0)) > 0
            and embedded_preconditions is None
        )
        if (
            prior_manifest.get("training") is not None
            or prior_manifest.get("policy_sync") is not None
            or not (nested_strict_resume_failure or fresh_runtime_failure)
        ):
            raise SmokeRunError(
                "failed training manifest does not preserve a strict zero-update resume proof"
            )
        manifest_training: Optional[Mapping[str, Any]] = None
    else:
        raw_training = prior_manifest.get("training")
        if not isinstance(raw_training, Mapping):
            raise SmokeRunError("prior failed manifest has no training receipt")
        manifest_training = raw_training
    summary_path = paths["training_root"] / "training_summary.json"
    receipts: list[tuple[str, Mapping[str, Any]]] = []
    if fresh_runtime_failure:
        if summary_path.exists():
            raise SmokeRunError(
                "fresh runtime failure unexpectedly contains a training summary"
            )
    else:
        persisted_summary = _read_json_mapping(
            summary_path, label="prior failed training summary"
        )
        receipts.append(("summary", persisted_summary))
    if manifest_training is not None:
        receipts.insert(0, ("manifest", manifest_training))
    for receipt_name, receipt in receipts:
        if (
            receipt.get("optimizer_updates") != 0
            or float(receipt.get("trainable_update_l2", 0.0)) != 0.0
            or receipt.get("checkpoint_dir") not in (None, "")
            or receipt.get("optimizer_state_saved", False) is not False
        ):
            raise SmokeRunError(
                f"prior {receipt_name} does not prove a zero-update failure"
            )
    if fresh_runtime_failure:
        if paths["sync"].exists():
            raise SmokeRunError(
                "fresh runtime failure unexpectedly contains a policy sync receipt"
            )
        sync: Mapping[str, Any] = {
            "success": False,
            "status": "absent_before_training_completed",
        }
    else:
        sync = _read_json_mapping(paths["sync"], label="prior policy sync receipt")
        if (
            sync.get("success") is not False
            or sync.get("status") != "not_attempted_no_optimizer_update"
        ):
            raise SmokeRunError("prior attempt may have published a policy adapter")
    manifest_sync = prior_manifest.get("policy_sync")
    if fresh_runtime_failure:
        pass
    elif retry_after_runtime_failure:
        if embedded_preconditions.get("policy_sync_status") != sync.get("status"):
            raise SmokeRunError("retry manifest and policy sync receipts disagree")
    elif not isinstance(manifest_sync, Mapping) or dict(manifest_sync) != dict(sync):
        raise SmokeRunError("prior manifest and policy sync receipts disagree")
    if (paths["training_root"] / "checkpoint_final").exists():
        raise SmokeRunError("prior attempt already contains a final learner checkpoint")
    if tuple(paths["training_root"].rglob("optimizer_state.pt")):
        raise SmokeRunError("prior attempt already contains optimizer state")
    if paths["post_update"].exists():
        raise SmokeRunError("prior attempt already contains a post-update canary")
    if next_cursor_path is not None and next_cursor_path.exists():
        raise SmokeRunError("prior attempt already committed the next training cursor")
    return {
        "source_manifest_status": prior_status,
        "root_zero_update_status": (
            "failed_training_before_persistence"
            if fresh_runtime_failure
            else embedded_preconditions.get("source_manifest_status")
            if retry_after_runtime_failure
            else prior_status
        ),
        "optimizer_updates": 0,
        "policy_sync_status": sync["status"],
        "checkpoint_absent": True,
        "optimizer_state_absent": True,
        "post_update_canary_absent": True,
        "next_cursor_absent": next_cursor_path is None or not next_cursor_path.exists(),
    }


def _validate_resume_evidence_stream(
    records: Sequence[TrajectoryRecord],
    *,
    evidence_path: Path,
) -> None:
    """Require the EvidenceStore stream to contain the same frozen batch."""

    if not evidence_path.is_file():
        raise FileNotFoundError(
            f"resume EvidenceStore trajectory stream does not exist: {evidence_path}"
        )
    event_ids: list[str] = []
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SmokeRunError(
                    f"resume evidence line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise SmokeRunError(
                    f"resume evidence line {line_number} is not a mapping"
                )
            payload = value.get("payload")
            event_id = value.get("event_id")
            if not isinstance(payload, Mapping) or payload.get("trajectory_id") != event_id:
                raise SmokeRunError(
                    f"resume evidence line {line_number} has an invalid trajectory envelope"
                )
            if not isinstance(event_id, str) or not event_id:
                raise SmokeRunError(
                    f"resume evidence line {line_number} has no trajectory event ID"
                )
            event_ids.append(event_id)
    expected_ids = [record.trajectory_id for record in records]
    if len(event_ids) != len(set(event_ids)) or set(event_ids) != set(expected_ids):
        raise SmokeRunError(
            "resume EvidenceStore stream differs from the frozen trajectory batch"
        )


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("artifact values must be mappings, dataclasses, or expose to_dict()")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value) if not isinstance(value, Mapping) else dict(value),
                   ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _read_jsonl_mappings(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    values: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SmokeRunError(
                    f"JSONL line {line_number} is not valid JSON: {path}"
                ) from exc
            if not isinstance(value, Mapping):
                raise SmokeRunError(
                    f"JSONL line {line_number} is not a mapping: {path}"
                )
            values.append(value)
    return tuple(values)


def _augment_joint_trivia_tasks(
    tasks: Sequence[TaskRecord],
    *,
    selection: Mapping[str, Any],
    receipt_path: Path,
) -> tuple[TaskRecord, ...]:
    """Reuse the frozen SkillFlow public retrieval condition for joint rollout."""

    trivia_tasks = tuple(task for task in tasks if _dataset_key(task) == "triviaqa")
    if len(trivia_tasks) != 1:
        raise SmokeRunError("each joint QA step must contain exactly one TriviaQA task")
    cached_by_task = {
        str(row.get("task_id")): row for row in _read_jsonl_mappings(receipt_path)
    }
    receipts: dict[str, Any] = {}
    persisted: dict[str, Mapping[str, Any]] = {}
    for task in trivia_tasks:
        cached = cached_by_task.get(task.task_id)
        if cached is None or cached.get("question") != task.question:
            continue
        receipt_value = cached.get("retrieval")
        if not isinstance(receipt_value, Mapping):
            continue
        restored = receipt_from_mapping(receipt_value)
        if restored.query != build_keyword_query(task.question):
            continue
        receipts[task.task_id] = restored
        persisted[task.task_id] = cached

    retrieval = _mapping(
        selection.get("retrieval"),
        "data.joint_qa_micro.retrieval",
    )
    missing = [task for task in trivia_tasks if task.task_id not in receipts]
    if missing:
        with SkillFlowQARetriever(
            index_path=str(retrieval["index_path"]),
            skillflow_source=str(retrieval["skillflow_source"]),
            search_limit=int(retrieval["search_limit"]),
        ) as retriever:
            for task in missing:
                receipt = retriever.retrieve(build_keyword_query(task.question))
                receipts[task.task_id] = receipt
                persisted[task.task_id] = {
                    "schema_version": "flowsteer.triviaqa.public_retrieval.v1",
                    "task_id": task.task_id,
                    "question": task.question,
                    "retrieval": receipt.to_dict(),
                    "created_at": _utc_now(),
                }
        _write_jsonl(
            receipt_path,
            [persisted[task.task_id] for task in trivia_tasks],
        )

    return tuple(
        augment_task_with_retrieval(task, receipts[task.task_id])
        if _dataset_key(task) == "triviaqa"
        else task
        for task in tasks
    )


def _read_trajectory_records(path: Path) -> tuple[TrajectoryRecord, ...]:
    """Load persisted rollouts through the immutable record contract."""

    if not path.is_file():
        raise FileNotFoundError(f"resume trajectory artifact does not exist: {path}")
    records: list[TrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SmokeRunError(
                    f"resume trajectory line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise SmokeRunError(
                    f"resume trajectory line {line_number} is not a mapping"
                )
            try:
                records.append(TrajectoryRecord.from_dict(value))
            except (KeyError, TypeError, ValueError) as exc:
                raise SmokeRunError(
                    f"resume trajectory line {line_number} violates its record contract"
                ) from exc
    return tuple(records)


def _safe_error(error: BaseException) -> str:
    message = str(error)
    secret = os.environ.get("VECTOR_ENGINE_API_KEY", "")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return f"{type(error).__name__}: {message}"


def _graph_from_mapping(value: Mapping[str, Any]) -> AgentGraph:
    raw_nodes = value.get("nodes", ())
    raw_relations = value.get("relations", ())
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        raise ValueError("final graph nodes are malformed")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        raise ValueError("final graph relations are malformed")
    nodes = []
    for item in raw_nodes:
        if not isinstance(item, Mapping):
            raise ValueError("final graph node is malformed")
        nodes.append(
            AgentNode(
                str(item.get("id", "")),
                str(item.get("model_id", "")),
                str(item.get("contract", item.get("prompt", ""))),
            )
        )
    relations = []
    for item in raw_relations:
        if not isinstance(item, Mapping):
            raise ValueError("final graph relation is malformed")
        relations.append(
            AgentRelation(
                str(item.get("source_id", "")),
                str(item.get("target_id", "")),
                item.get("source_to_target"),
                item.get("target_to_source"),
            )
        )
    revision = value.get("revision", 0)
    if type(revision) is not int:
        raise ValueError("final graph revision is malformed")
    output = value.get("output_agent_id")
    if output is not None and not isinstance(output, str):
        raise ValueError("final graph output_agent_id is malformed")
    return AgentGraph(nodes, relations, output_agent_id=output, revision=revision)


class SmokeBackend(Protocol):
    model_catalog_version: str

    async def collect(
        self,
        task: TaskRecord,
        rollout_index: int,
        versions: VersionBundle,
        *,
        expected_task_split: str = "train",
        condition_id: Optional[str] = None,
        sampling_schedule_purpose: Optional[str] = None,
        prompt_priors: Sequence[Mapping[str, Any]] = (),
        forced_probe: bool = False,
        condition_satisfied: bool = True,
        sampling_anchor_ordinal: Optional[int] = None,
    ) -> TrajectoryRecord:
        ...

    def train(
        self,
        trajectories: Sequence[TrajectoryRecord],
        output_dir: Path,
    ) -> Any:
        ...

    async def publish(self, summary: Any) -> Any:
        ...


JudgeCallback = Callable[[Sequence[Mapping[str, str]], str], Awaitable[Any]]


class LiveSmokeBackend:
    """Thin wiring layer over the existing collector, trainer, and publisher."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        registry: Any,
        runtime: AgentRuntime,
        director_client: SGLangReceiptDirectorClient,
        rollout_gate: RolloutGate,
        evidence_store: EvidenceStore,
        trainer: Optional[Qwen35OnePassSmokeTrainer],
        publisher: SGLangPolicyPublisher,
        judge: Optional[JudgeCallback],
        judge_model: str,
        skill_pipeline: Optional[SkillEvidencePipeline] = None,
        skill_epoch: int = 0,
    ) -> None:
        self.config = config
        self.registry = registry
        self.runtime = runtime
        self.director_client = director_client
        self.rollout_gate = rollout_gate
        self.evidence_store = evidence_store
        self.trainer = trainer
        self.publisher = publisher
        self.judge = judge
        self.judge_model = judge_model
        self.skill_pipeline = skill_pipeline
        self.skill_epoch = skill_epoch

    @property
    def model_catalog_version(self) -> str:
        return self.registry.catalog_id

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        root: Path,
        *,
        evaluation_only: bool = False,
    ) -> "LiveSmokeBackend":
        director = _mapping(config["director"], "director")
        experiment = _mapping(config["experiment"], "experiment")
        graph_config = _mapping(config["agent_graph"], "agent_graph")
        grpo = _mapping(config["grpo"], "grpo")
        gpu = _mapping(config["gpu"], "gpu")
        oom = _mapping(gpu["oom_policy"], "gpu.oom_policy")
        storage = _mapping(config["storage"], "storage")
        sync = _mapping(config["policy_sync"], "policy_sync")
        evaluation = _mapping(config["evaluation"], "evaluation")
        skills_config = _mapping(config["skills"], "skills")

        catalog_path = _resolve(root, str(graph_config["model_catalog_path"]))
        if not catalog_path.is_file():
            raise ConfigurationError(
                f"model catalog does not exist: {catalog_path}; copy the example first"
            )
        registry = load_model_registry(catalog_path)
        required_credentials = tuple(
            sorted(
                {
                    provider.api_key_env
                    for provider_id in registry.provider_ids
                    for provider in (registry.require_provider(provider_id),)
                    if provider.api_key_env is not None
                }
            )
        )
        missing_credentials = tuple(
            name for name in required_credentials if not os.environ.get(name, "")
        )
        if missing_credentials:
            raise ConfigurationError(
                "missing required provider environment variable(s): "
                + ", ".join(missing_credentials)
            )
        secret = os.environ.get("VECTOR_ENGINE_API_KEY", "")

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - heavy runtime only
            raise RuntimeError("transformers is required for Qwen3.5 smoke rollout") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(director["tokenizer_path"]),
            trust_remote_code=True,
        )

        gate = RolloutGate()
        behavior_adapter = director.get("behavior_adapter_name")
        if behavior_adapter is not None:
            behavior_adapter = str(behavior_adapter).strip() or None
        director_client = SGLangReceiptDirectorClient(
            tokenizer,
            base_url=str(director["api_base"]),
            api_key=os.environ.get("SGLANG_API_KEY", "EMPTY"),
            policy_version=str(director["behavior_policy_version"]),
            adapter_name=behavior_adapter,
            expected_server_weight_version=str(
                director["expected_server_weight_version"]
            ),
            rollout_gate=gate,
            temperature=float(director["temperature"]),
            top_p=float(director["top_p"]),
            top_k=int(director["top_k"]),
            max_tokens=int(director["max_action_tokens"]),
        )

        gateway = OpenAICompatibleGateway(default_seed=int(experiment["seed"]))
        runtime = AgentRuntime(registry, gateway)
        evidence_store = EvidenceStore(_resolve(root, str(storage["root"])))
        skill_pipeline: Optional[SkillEvidencePipeline] = None
        skill_epoch = int(skills_config.get("current_epoch", 0))
        if bool(skills_config.get("enabled", False)):
            raw_store_path = skills_config.get("store_path")
            if not isinstance(raw_store_path, str) or not raw_store_path.strip():
                raise ConfigurationError(
                    "skills.store_path is required when validated Skill retrieval is enabled"
                )
            store_path = _resolve(root, raw_store_path)
            if not store_path.is_file():
                raise ConfigurationError(
                    "enabled Skill retrieval requires an existing evidence-gated store: "
                    f"{store_path}"
                )
            retrieval_top_k = skills_config.get("retrieval_top_k")
            if (
                isinstance(retrieval_top_k, bool)
                or not isinstance(retrieval_top_k, int)
                or retrieval_top_k < 1
            ):
                raise ConfigurationError(
                    "skills.retrieval_top_k must be positive when Skill retrieval is enabled"
                )
            if skill_epoch < 0:
                raise ConfigurationError("skills.current_epoch must be non-negative")
            skill_pipeline = SkillEvidencePipeline(
                evidence_store=evidence_store,
                skill_store=SkillStore(store_path),
                retrieval_top_k=retrieval_top_k,
            )

        trainer: Optional[Qwen35OnePassSmokeTrainer] = None
        if not evaluation_only:
            lora = _mapping(director["lora"], "director.lora")
            trainer = Qwen35OnePassSmokeTrainer(
                SmokeTrainerConfig(
                    model_path=str(director["base_model"]),
                    tokenizer_path=str(director["tokenizer_path"]),
                    behavior_policy_version=str(director["behavior_policy_version"]),
                    updated_policy_version=str(director["updated_policy_version"]),
                    behavior_policy_adapter=behavior_adapter,
                    behavior_server_weight_version=str(
                        director["expected_server_weight_version"]
                    ),
                    behavior_adapter_checkpoint=(
                        str(_resolve(root, str(director["behavior_adapter_checkpoint"])))
                        if director.get("behavior_adapter_checkpoint")
                        else None
                    ),
                    update_step=int(experiment.get("update_step", 1)),
                    optimizer_state_checkpoint=(
                        str(_resolve(root, str(director["optimizer_state_checkpoint"])))
                        if director.get("optimizer_state_checkpoint")
                        else None
                    ),
                    exact_optimizer_continuation=bool(
                        grpo.get("exact_optimizer_continuation", False)
                    ),
                    learner_device=str(gpu["learner_device"]),
                    gradient_replica_device=str(gpu["gradient_replica_device"]),
                    lora_rank=int(lora["rank"]),
                    lora_alpha=int(lora["alpha"]),
                    lora_dropout=float(lora["dropout"]),
                    lora_target_modules=tuple(
                        str(value) for value in lora["target_modules"]
                    ),
                    learning_rate=float(grpo["learning_rate"]),
                    max_grad_norm=float(grpo["max_grad_norm"]),
                    advantage_epsilon=float(grpo["advantage_epsilon"]),
                    gradient_checkpointing=bool(grpo["gradient_checkpointing"]),
                    micro_batch_backoff=tuple(
                        int(value) for value in oom["micro_batch_schedule"]
                    ),
                )
            )
        publisher = SGLangPolicyPublisher(
            PolicySyncConfig(
                api_base=str(sync["api_base"]),
                api_key=os.environ.get("SGLANG_API_KEY", "EMPTY"),
                adapter_name_prefix=str(sync["adapter_name_prefix"]),
                request_timeout_seconds=float(sync["request_timeout_seconds"]),
                max_retries=int(sync["max_retries"]),
                retry_backoff_seconds=float(sync["retry_backoff_seconds"]),
            )
        )
        judge: Optional[JudgeCallback] = None
        judge_model = ""
        if not evaluation_only and not _is_frozen_micro(config):
            judge, judge_model = cls._build_healthbench_judge(
                registry,
                secret,
                str(evaluation["healthbench_judge_model"]),
            )
        return cls(
            config=config,
            registry=registry,
            runtime=runtime,
            director_client=director_client,
            rollout_gate=gate,
            evidence_store=evidence_store,
            trainer=trainer,
            publisher=publisher,
            judge=judge,
            judge_model=judge_model,
            skill_pipeline=skill_pipeline,
            skill_epoch=skill_epoch,
        )

    @staticmethod
    def _build_healthbench_judge(
        registry: Any,
        secret: str,
        configured_model_id: str,
    ) -> tuple[JudgeCallback, str]:
        if not configured_model_id.strip():
            raise ConfigurationError("evaluation.healthbench_judge_model is empty")
        model = registry.require_model(configured_model_id)
        provider = registry.provider_for(model.model_id)
        if provider.api_key_env != "VECTOR_ENGINE_API_KEY":
            raise ConfigurationError(
                "configured HealthBench judge must use the VectorEngine provider"
            )
        if not provider.endpoint:
            raise ConfigurationError("HealthBench judge provider has no endpoint")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("openai is required for the HealthBench judge") from exc
        client = AsyncOpenAI(api_key=secret, base_url=provider.endpoint)

        async def judge(messages: Sequence[Mapping[str, str]], model_name: str) -> Any:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[dict(message) for message in messages],
                max_tokens=2048,
                temperature=0,
            )
            if not response.choices or response.choices[0].message.content is None:
                return ""
            return response.choices[0].message.content

        return judge, model.model_name

    def _visible_skill_priors(
        self,
        task: TaskRecord,
        environment: AgentWorkflowEnv,
        versions: VersionBundle,
    ) -> Sequence[Mapping[str, Any]]:
        """Retrieve only ACTIVE, evidence-gated Skills for the current graph stage."""

        if self.skill_pipeline is None:
            return ()
        graph = environment.graph
        if not graph.nodes:
            graph_stage = "empty_graph"
        elif graph.validate(self.registry, require_complete=True).valid:
            graph_stage = "before_final_answer"
        else:
            graph_stage = "construction"
        task_family = str(
            task.metadata.get("task_family", _dataset_key(task))
        ).strip()
        if not task_family:
            task_family = _dataset_key(task)
        complete_validation = graph.validate(
            self.registry,
            require_complete=True,
        )
        issue_tags = tuple(sorted(issue.code for issue in complete_validation.issues))
        query_tags = _skill_query_tags(
            task,
            graph,
            task_family=task_family,
            graph_stage=graph_stage,
            validation_issue_codes=issue_tags,
        )
        priors = self.skill_pipeline.retrieve_prompt_priors(
            SkillQuery(
                task_family=task_family,
                graph_stage=graph_stage,
                tags=query_tags,
                available_models=tuple(self.registry.model_ids),
                current_epoch=self.skill_epoch,
            ),
            versions,
        )
        return tuple(prior.to_dict() for prior in priors)

    async def collect(
        self,
        task: TaskRecord,
        rollout_index: int,
        versions: VersionBundle,
        *,
        expected_task_split: str = "train",
        condition_id: Optional[str] = None,
        sampling_schedule_purpose: Optional[str] = None,
        prompt_priors: Sequence[Mapping[str, Any]] = (),
        forced_probe: bool = False,
        condition_satisfied: bool = True,
        sampling_anchor_ordinal: Optional[int] = None,
    ) -> TrajectoryRecord:
        director = _mapping(self.config["director"], "director")
        graph_config = _mapping(self.config["agent_graph"], "agent_graph")
        experiment = _mapping(self.config["experiment"], "experiment")
        if condition_id is None:
            resolved_condition_id = str(
                experiment.get("condition_id", "natural_smoke")
            ).strip()
        elif isinstance(condition_id, str):
            resolved_condition_id = condition_id.strip()
        else:
            raise TypeError("condition_id must be text or None")
        if not resolved_condition_id:
            raise ConfigurationError("condition_id must be non-empty")
        if sampling_schedule_purpose is None:
            resolved_schedule_purpose = str(
                experiment.get(
                    "sampling_schedule_purpose",
                    resolved_condition_id,
                )
            ).strip()
        elif isinstance(sampling_schedule_purpose, str):
            resolved_schedule_purpose = sampling_schedule_purpose.strip()
        else:
            raise TypeError("sampling_schedule_purpose must be text or None")
        if not resolved_schedule_purpose:
            raise ConfigurationError("sampling_schedule_purpose must be non-empty")
        for name, value in (
            ("forced_probe", forced_probe),
            ("condition_satisfied", condition_satisfied),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        if isinstance(prompt_priors, (str, bytes)) or not isinstance(
            prompt_priors, Sequence
        ):
            raise TypeError("prompt_priors must be a sequence of mappings")
        normalized_conditions: list[dict[str, Any]] = []
        for item in prompt_priors:
            if not isinstance(item, Mapping):
                raise TypeError("prompt_priors must contain only mappings")
            normalized_conditions.append(dict(item))
        static_conditions = tuple(normalized_conditions)
        if static_conditions:
            if self.skill_pipeline is not None:
                raise ConfigurationError(
                    "forced-probe prompt conditions and dynamic ACTIVE Skill "
                    "retrieval are mutually exclusive"
                )
            if not forced_probe:
                raise ConfigurationError(
                    "static prompt conditions require forced_probe=true"
                )
            if any(
                item.get("application_mode") != "forced_probe_condition"
                for item in static_conditions
            ):
                raise ConfigurationError(
                    "static prompt conditions must use "
                    "application_mode=forced_probe_condition"
                )
        terminal_protocols = graph_config.get("terminal_protocol_by_source", {})
        if not isinstance(terminal_protocols, Mapping):
            raise ConfigurationError(
                "agent_graph.terminal_protocol_by_source must be a mapping"
            )
        terminal_protocol = str(
            terminal_protocols.get(_dataset_key(task), "none")
        ).strip()
        if terminal_protocol not in {"none", "exact_single_answer_tag"}:
            raise ConfigurationError(
                "terminal protocol must be none or exact_single_answer_tag"
            )
        catalog_order_namespace = str(
            experiment.get(
                "catalog_order_namespace",
                experiment.get("condition_id", "natural_smoke"),
            )
        ).strip()
        if not catalog_order_namespace:
            raise ConfigurationError("experiment.catalog_order_namespace must be non-empty")
        base_seed = int(experiment["seed"])
        if sampling_anchor_ordinal is None:
            resolved_sampling_anchor_ordinal = int(
                experiment.get(
                    "sampling_anchor_ordinal",
                    experiment.get("update_step", 0),
                )
            )
        elif (
            type(sampling_anchor_ordinal) is int
            and sampling_anchor_ordinal >= 0
        ):
            resolved_sampling_anchor_ordinal = sampling_anchor_ordinal
        else:
            raise ValueError("sampling_anchor_ordinal must be non-negative or None")
        sampling_coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(
                base_seed=base_seed
            ),
            schedule_purpose=resolved_schedule_purpose,
            ordered_sequence_hash=stable_hash([task.task_id]),
            # SkillFlow's position is a rollout coordinate.  It must be the
            # ordinal within this task, never the task's selected-list index.
            sequence_position=rollout_index,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=resolved_sampling_anchor_ordinal,
        )
        orchestrator = AgentGraphOrchestrator(
            self.registry,
            self.director_client,
            max_rounds=int(director["max_rounds"]),
            seed=base_seed,
            catalog_order_seed=(
                f"{experiment['seed']}:{catalog_order_namespace}:"
                f"{task.task_id}"
            ),
            history_window=int(director["history_window"]),
            sampling_base_seed=base_seed,
            sampling_coordinate=sampling_coordinate,
        )
        environment = AgentWorkflowEnv(
            self.registry,
            runtime=self.runtime,
            execute_on_edit=bool(director["execute_on_edit"]),
            max_agents=int(graph_config["max_agents"]),
            max_agents_per_subgraph=int(
                graph_config.get("max_agents_per_subgraph", 3)
            ),
            require_exact_answer_tag=(
                terminal_protocol == "exact_single_answer_tag"
            ),
            require_format_agent=(
                terminal_protocol == "exact_single_answer_tag"
            ),
        )
        collector = AgentGraphRolloutCollector(
            orchestrator,
            environment,
            versions,
            self.evidence_store,
            condition_id=resolved_condition_id,
            skills=static_conditions,
            skill_provider=(
                self._visible_skill_priors
                if self.skill_pipeline is not None and not static_conditions
                else None
            ),
            condition_satisfied=condition_satisfied,
            forced_probe=forced_probe,
            expected_task_split=expected_task_split,
        )

        async def evaluator_callback(
            evaluated_task: TaskRecord,
            final_answer: Optional[str],
            final_graph: Mapping[str, Any],
            final_runtime: Any,
        ) -> Any:
            del final_runtime
            source_key = _dataset_key(evaluated_task)
            if source_key in {"webshop", "alfworld"} and final_answer is None:
                # A natural Director budget exhaustion is already a real
                # terminal failure in the MD/SkillFlow boundary.  Do not start
                # a fresh interactive environment after the workflow itself
                # failed to finish.
                return EvaluationOutcome(
                    valid=True,
                    reward=0.0,
                    metrics={"success": 0.0},
                    reason="director_max_rounds_without_explicit_finish",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                )
            environment_graph = _graph_from_mapping(final_graph)
            environment_step = 0

            async def run_graph(observation: str) -> str:
                nonlocal environment_step
                environment_step += 1
                result = await self.runtime.execute(
                    environment_graph,
                    observation,
                    run_id=(
                        f"environment:{evaluated_task.task_id}:"
                        f"{rollout_index:04d}:{environment_step:04d}"
                    ),
                )
                return result.final_answer

            configured_steps = _mapping(
                self.config["evaluation"], "evaluation"
            ).get("max_environment_steps_by_source", {})
            if not isinstance(configured_steps, Mapping):
                configured_steps = {}
            return await evaluate_task(
                evaluated_task,
                final_answer or "",
                judge=self.judge,
                judge_model=self.judge_model,
                run_graph=run_graph,
                max_environment_steps=int(
                    configured_steps.get(
                        source_key,
                        _mapping(self.config["evaluation"], "evaluation")[
                            "max_environment_steps"
                        ],
                    )
                ),
            )

        return await collector.collect(task, rollout_index, evaluator_callback)

    def train(
        self,
        trajectories: Sequence[TrajectoryRecord],
        output_dir: Path,
    ) -> Any:
        if self.trainer is None:
            raise RuntimeError("training is disabled for this evaluation-only backend")
        return self.trainer.train(trajectories, output_dir)

    async def publish(self, summary: Any) -> Any:
        director = _mapping(self.config["director"], "director")
        experiment = _mapping(self.config["experiment"], "experiment")
        checkpoint_version = f"checkpoint:{summary.updated_policy_version}"
        previous_policy_version = self.director_client.policy_version
        previous_adapter = self.director_client.adapter_name
        previous_server_weight_version = (
            self.director_client.expected_server_weight_version
        )

        def switch_route(policy_version: str, adapter_name: str) -> None:
            self.director_client.update_policy_route(
                policy_version=policy_version,
                adapter_name=adapter_name,
                expected_server_weight_version=str(
                    director["expected_server_weight_version"]
                ),
            )

        def restore_route() -> None:
            self.director_client.update_policy_route(
                policy_version=previous_policy_version,
                adapter_name=previous_adapter,
                expected_server_weight_version=previous_server_weight_version,
            )

        try:
            receipt = await asyncio.to_thread(
                self.publisher.publish,
                checkpoint_path=summary.checkpoint_dir,
                checkpoint_version=checkpoint_version,
                behavior_policy_version=summary.behavior_policy_version,
                candidate_policy_version=summary.updated_policy_version,
                step=int(experiment.get("update_step", 1)),
                previous_adapter=previous_adapter,
                gate=self.rollout_gate,
                route_switch=switch_route,
                route_rollback=restore_route,
            )
        except PolicySyncError:
            raise
        return receipt


def _summary_dict(summary: Any) -> dict[str, Any]:
    value = _json_value(summary)
    return dict(value)


def _write_grpo_groups(path: Path, trajectories: Sequence[TrajectoryRecord]) -> None:
    grouped: dict[tuple[str, str, str], list[tuple[TrajectoryRecord, Any]]] = defaultdict(
        list
    )
    for record in trajectories:
        item = trajectory_to_grpo(record)
        grouped[item.group_key].append((record, item))
    rows = []
    for key, entries in sorted(grouped.items()):
        eligible = [
            item for record, item in entries if record.grpo_eligible and item.eligible
        ]
        eligible_advantages = same_condition_advantages(eligible)
        advantages_by_id = {
            item.trajectory_id: float(value)
            for item, value in zip(eligible, eligible_advantages, strict=True)
        }
        rows.append(
            {
                "group_key": list(key),
                "trajectory_ids": [item.trajectory_id for _, item in entries],
                "rewards": [item.terminal_reward for _, item in entries],
                "eligible": [
                    record.grpo_eligible and item.eligible for record, item in entries
                ],
                "advantages": [
                    advantages_by_id.get(item.trajectory_id) for _, item in entries
                ],
                "informative": bool(
                    len(eligible) >= 2
                    and any(float(value) != 0.0 for value in eligible_advantages)
                ),
            }
        )
    _write_jsonl(path, rows)


def _artifact_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config["storage"], "storage")
    paths = {
        "selected": _resolve(root, str(storage["selected_tasks_path"])),
        "trajectories": _resolve(root, str(storage["trajectories_path"])),
        "groups": _resolve(root, str(storage["grpo_groups_path"])),
        "manifest": _resolve(root, str(storage["manifest_path"])),
        "sync": _resolve(root, str(storage["sync_receipt_path"])),
        "post_update": _resolve(root, str(storage["post_update_trajectories_path"])),
        "training_root": _resolve(
            root, str(_mapping(config["experiment"], "experiment")["output_dir"])
        ),
    }
    for name, field_name in (
        ("retrieval", "retrieval_receipts_path"),
        ("behavior_preflight", "behavior_policy_preflight_path"),
    ):
        value = storage.get(field_name)
        if isinstance(value, str) and value.strip():
            paths[name] = _resolve(root, value)
    return paths


def _select_run_scope(
    config: Mapping[str, Any],
    root: Path,
) -> tuple[
    tuple[TaskRecord, ...],
    tuple[int, ...],
    Optional[HotpotTrainingProgress | JointQATrainingProgress],
    Optional[Path],
    Mapping[str, Any],
]:
    """Resolve the legacy smoke scope or one exact frozen micro-training step."""

    data = _mapping(config["data"], "data")
    train_path = _resolve(root, str(data["train_path"]))
    if not _is_frozen_micro(config):
        smoke = _mapping(data["smoke"], "data.smoke")
        selected = select_smoke_tasks(
            iter_task_records(train_path, expected_split="train"),
            source_order=tuple(str(value) for value in smoke["source_order"]),
            per_source=int(smoke["tasks_per_dataset"]),
            require_unique_base_tasks=bool(smoke["require_unique_base_tasks"]),
        )
        rollout_ordinals = tuple(
            range(int(_mapping(config["grpo"], "grpo")["samples_per_problem"]))
        )
        return selected, rollout_ordinals, None, None, {
            "selection": "sequential_per_source",
        }

    if _is_joint_qa_micro(config):
        joint = _mapping(data["joint_qa_micro"], "data.joint_qa_micro")
        schedule_path = _resolve(root, str(joint["schedule_path"]))
        cursor_path = _resolve(root, str(joint["cursor_path"]))
        next_cursor_path = _resolve(root, str(joint["next_cursor_path"]))
        if next_cursor_path.exists():
            raise FileExistsError(
                f"write-once next joint-QA cursor already exists: {next_cursor_path}"
            )
        schedule = FrozenJointQATrainingSchedule.read(schedule_path)
        cursor = JointQATrainingCursorState.read(cursor_path)
        progress = JointQATrainingProgress.from_state(schedule, cursor)
        resolved = schedule.resolve(
            train_path=train_path,
            validation_path=_resolve(root, str(data["validation_path"])),
            test_path=_resolve(root, str(data["test_path"])),
        )
        step = progress.current_step
        tasks = resolved[cursor.cursor]
        if tuple(task.task_id for task in tasks) != tuple(
            binding.task_id for binding in step.tasks
        ):
            raise SmokeRunError(
                "current frozen joint-QA tasks do not match their cursor"
            )
        rollout_ordinals = step.rollout_ordinals
        grpo = _mapping(config["grpo"], "grpo")
        if len(rollout_ordinals) != int(grpo["samples_per_problem"]):
            raise ConfigurationError(
                "frozen rollout ordinals differ from grpo.samples_per_problem"
            )
        storage = _mapping(config["storage"], "storage")
        retrieval_path = _resolve(
            root,
            str(storage["retrieval_receipts_path"]),
        )
        augmented = _augment_joint_trivia_tasks(
            tasks,
            selection=joint,
            receipt_path=retrieval_path,
        )
        return augmented, rollout_ordinals, progress, next_cursor_path, {
            "selection": "frozen_joint_qa_schedule",
            "schedule_path": str(schedule_path),
            "schedule_id": schedule.content_hash,
            "cursor_path": str(cursor_path),
            "cursor_before": cursor.to_value(),
            "step": step.to_value(),
            "retrieval_receipts_path": str(retrieval_path),
            "next_cursor_path": str(next_cursor_path),
        }

    hotpot = _mapping(data["hotpot_micro"], "data.hotpot_micro")
    schedule_path = _resolve(root, str(hotpot["schedule_path"]))
    cursor_path = _resolve(root, str(hotpot["cursor_path"]))
    next_cursor_path = _resolve(root, str(hotpot["next_cursor_path"]))
    if next_cursor_path.exists():
        raise FileExistsError(
            f"write-once next HotpotQA cursor already exists: {next_cursor_path}"
        )
    schedule = FrozenHotpotTrainingSchedule.read(schedule_path)
    cursor = HotpotTrainingCursorState.read(cursor_path)
    progress = HotpotTrainingProgress.from_state(schedule, cursor)
    resolved = schedule.resolve(
        train_path=train_path,
        validation_path=_resolve(root, str(data["validation_path"])),
        test_path=_resolve(root, str(data["test_path"])),
    )
    step = progress.current_step
    task = resolved[cursor.cursor]
    if task.task_id != step.task_id:
        raise SmokeRunError("current frozen HotpotQA task does not match its cursor")
    rollout_ordinals = step.rollout_ordinals
    grpo = _mapping(config["grpo"], "grpo")
    if len(rollout_ordinals) != int(grpo["samples_per_problem"]):
        raise ConfigurationError(
            "frozen rollout ordinals differ from grpo.samples_per_problem"
        )
    return (task,), rollout_ordinals, progress, next_cursor_path, {
        "selection": "frozen_hotpot_schedule",
        "schedule_path": str(schedule_path),
        "schedule_id": schedule.content_hash,
        "cursor_path": str(cursor_path),
        "cursor_before": cursor.to_value(),
        "step": step.to_value(),
        "next_cursor_path": str(next_cursor_path),
    }


async def run_smoke(
    config_path: str | Path,
    *,
    prepare_only: bool = False,
    resume_initial_rollouts: bool = False,
    backend: Optional[SmokeBackend] = None,
    project_root: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """Execute the exact bounded pipeline and return its persisted manifest."""

    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_smoke_bounds(config)
    if resume_initial_rollouts and not _is_frozen_micro(config):
        raise ConfigurationError(
            "exact initial-rollout resume is limited to frozen micro-training steps"
        )
    paths = _artifact_paths(config, root)

    data = _mapping(config["data"], "data")
    train_path = _resolve(root, str(data["train_path"]))
    selected, rollout_ordinals, progress, next_cursor_path, scope_receipt = (
        _select_run_scope(config, root)
    )
    resume_preconditions: Optional[Mapping[str, Any]] = None
    if resume_initial_rollouts:
        resume_preconditions = _validate_resume_preconditions(
            paths=paths,
            next_cursor_path=next_cursor_path,
        )
    selection_key = (
        "hotpot_micro"
        if _is_hotpot_micro(config)
        else "joint_qa_micro"
        if _is_joint_qa_micro(config)
        else "smoke"
    )
    selection = _mapping(data[selection_key], "data selection")
    if len(selected) != int(selection["expected_total_tasks"]):
        raise SmokeRunError("selected task count differs from the fixed run bound")
    _write_jsonl(
        paths["selected"],
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            for task in selected
        ],
    )

    source_counts = Counter(_dataset_key(task) for task in selected)
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.agentgraph.smoke_manifest.v1",
        "status": "prepared" if prepare_only else "collecting",
        "config_path": str(resolved_config),
        "train_path": str(train_path),
        "started_at": _utc_now(),
        "bounds": {
            "tasks_per_dataset": (
                None
                if _is_hotpot_micro(config)
                else 1
                if _is_joint_qa_micro(config)
                else 2
            ),
            "selected_tasks": len(selected),
            "rollouts_per_task": len(rollout_ordinals),
            "expected_initial_rollouts": len(selected) * len(rollout_ordinals),
            "max_optimizer_updates": 1,
            "post_update_canaries": 1,
        },
        "selection_receipt": dict(scope_receipt),
        "selected_by_source": dict(sorted(source_counts.items())),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "training_root"},
        "exploration_enabled": False,
        "skills_enabled": bool(_mapping(config["skills"], "skills")["enabled"]),
        "resume_initial_rollouts_requested": bool(resume_initial_rollouts),
    }
    if resume_preconditions is not None:
        manifest["resume_preconditions"] = dict(resume_preconditions)
    if prepare_only:
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        return manifest

    if not resume_initial_rollouts:
        _write_json(paths["manifest"], manifest)
    try:
        live_backend = backend or LiveSmokeBackend.from_config(config, root)
    except Exception as exc:
        # Do not destroy the only persisted proof of a zero-update attempt
        # merely because a strict resume could not construct its runtime.
        if not resume_initial_rollouts:
            manifest["status"] = "failed_runtime_setup"
            manifest["error"] = _safe_error(exc)
            manifest["completed_at"] = _utc_now()
            _write_json(paths["manifest"], manifest)
        raise SmokeRunError("smoke runtime setup failed") from exc
    director = _mapping(config["director"], "director")
    grpo = _mapping(config["grpo"], "grpo")
    behavior_policy = str(director["behavior_policy_version"])
    updated_policy = str(director["updated_policy_version"])
    behavior_adapter = director.get("behavior_adapter_name")
    behavior_checkpoint = director.get("behavior_adapter_checkpoint")
    if (
        isinstance(live_backend, LiveSmokeBackend)
        and isinstance(behavior_adapter, str)
        and behavior_adapter.strip()
        and isinstance(behavior_checkpoint, str)
        and behavior_checkpoint.strip()
    ):
        try:
            behavior_preflight = await asyncio.to_thread(
                live_backend.publisher.ensure_loaded_adapter,
                checkpoint_path=str(_resolve(root, behavior_checkpoint)),
                adapter_name=behavior_adapter,
            )
        except Exception as exc:
            manifest["status"] = "failed_behavior_policy_preflight"
            manifest["error"] = _safe_error(exc)
            manifest["completed_at"] = _utc_now()
            _write_json(paths["manifest"], manifest)
            raise SmokeRunError("behavior policy adapter preflight failed") from exc
        manifest["behavior_policy_preflight"] = dict(behavior_preflight)
        if "behavior_preflight" in paths:
            _write_json(paths["behavior_preflight"], behavior_preflight)
        _write_json(paths["manifest"], manifest)
    expected_jobs: list[tuple[TaskRecord, int, VersionBundle]] = []
    experiment = _mapping(config["experiment"], "experiment")
    for task in selected:
        versions = version_bundle_for(
            task,
            policy_version=behavior_policy,
            model_catalog_version=live_backend.model_catalog_version,
            prompt_version=str(experiment.get("prompt_version", PROMPT_VERSION)),
            tool_version=str(experiment.get("tool_version", TOOL_VERSION)),
        )
        for rollout_index in rollout_ordinals:
            expected_jobs.append((task, rollout_index, versions))
    if resume_initial_rollouts:
        initial = _read_trajectory_records(paths["trajectories"])
        storage = _mapping(config["storage"], "storage")
        behavior_adapter = director.get("behavior_adapter_name")
        if behavior_adapter is not None:
            behavior_adapter = str(behavior_adapter).strip() or None
        _validate_resumed_initial_rollouts(
            initial,
            expected_jobs,
            condition_id=str(experiment.get("condition_id", "natural_smoke")),
            sampling_anchor_ordinal=int(
                experiment.get(
                    "sampling_anchor_ordinal",
                    experiment.get("update_step", 0),
                )
            ),
            behavior_adapter_name=behavior_adapter,
            expected_server_weight_version=str(
                director["expected_server_weight_version"]
            ),
        )
        _validate_resume_evidence_stream(
            initial,
            evidence_path=_resolve(root, str(storage["root"])) / "trajectories.jsonl",
        )
        manifest["initial_rollout_source"] = {
            "mode": "strict_persisted_resume",
            "path": str(paths["trajectories"]),
            "reused": len(initial),
            "new_collections": 0,
        }
        _write_json(paths["manifest"], manifest)
    else:
        initial_jobs = [
            live_backend.collect(task, rollout_index, versions)
            for task, rollout_index, versions in expected_jobs
        ]
        try:
            initial = tuple(await asyncio.gather(*initial_jobs))
        except Exception as exc:
            manifest["status"] = "failed_initial_rollout"
            manifest["error"] = _safe_error(exc)
            manifest["completed_at"] = _utc_now()
            _write_json(paths["manifest"], manifest)
            raise SmokeRunError("initial rollout collection failed") from exc
        manifest["initial_rollout_source"] = {
            "mode": "live_collection",
            "reused": 0,
            "new_collections": len(initial),
        }
    if len(initial) != int(grpo["expected_rollout_count"]):
        raise SmokeRunError("initial rollout count differs from the fixed smoke bound")
    if any(record.versions.policy != behavior_policy for record in initial):
        raise SmokeRunError("initial trajectories contain a non-behavior policy version")
    _write_jsonl(paths["trajectories"], initial)
    _write_grpo_groups(paths["groups"], initial)

    try:
        summary = await asyncio.to_thread(
            live_backend.train, initial, paths["training_root"]
        )
    except Exception as exc:
        manifest["status"] = "failed_training"
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("one-pass smoke training failed") from exc
    summary_value = _summary_dict(summary)
    manifest["initial_rollouts"] = {
        "collected": len(initial),
        "valid_evaluators": sum(record.evaluation.valid for record in initial),
        "grpo_eligible": sum(record.grpo_eligible for record in initial),
        "evaluator_versions": dict(
            sorted(Counter(record.evaluation.evaluator_version for record in initial).items())
        ),
    }
    manifest["training"] = summary_value
    if int(summary_value.get("optimizer_updates", 0)) != 1:
        sync_value = {
            "status": "not_attempted_no_optimizer_update",
            "success": False,
            "behavior_policy_version": behavior_policy,
            "candidate_policy_version": updated_policy,
        }
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_no_optimizer_update"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("smoke trainer completed zero optimizer updates")

    try:
        sync_receipt = await live_backend.publish(summary)
    except PolicySyncError as exc:
        sync_value = exc.receipt.to_dict()
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_policy_sync"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("SGLang policy publication failed") from exc
    except Exception as exc:
        sync_value = {
            "status": "failed",
            "success": False,
            "behavior_policy_version": behavior_policy,
            "candidate_policy_version": updated_policy,
            "error": _safe_error(exc),
        }
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_policy_sync"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("SGLang policy publication failed") from exc
    sync_value = _summary_dict(sync_receipt)
    if sync_value.get("success") is not True:
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_policy_sync"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("policy publisher returned an unsuccessful receipt")
    adapter_name = sync_value.get("adapter_name")
    new_policy = sync_value.get("new_policy_version")
    if not isinstance(adapter_name, str) or not adapter_name.strip():
        raise SmokeRunError("policy sync receipt has no adapter_name")
    if new_policy != updated_policy:
        raise SmokeRunError("policy sync receipt has the wrong updated policy version")
    _write_json(paths["sync"], sync_value)

    canary_count = int(
        _mapping(config["policy_sync"], "policy_sync")["post_update_canary_count"]
    )
    canary_jobs = []
    for index in range(canary_count):
        task = selected[index % len(selected)]
        versions = version_bundle_for(
            task,
            policy_version=updated_policy,
            model_catalog_version=live_backend.model_catalog_version,
            prompt_version=str(experiment.get("prompt_version", PROMPT_VERSION)),
            tool_version=str(experiment.get("tool_version", TOOL_VERSION)),
        )
        canary_jobs.append(live_backend.collect(task, 10_000 + index, versions))
    try:
        canaries = tuple(await asyncio.gather(*canary_jobs))
    except Exception as exc:
        manifest["status"] = "failed_post_update_canary"
        manifest["policy_sync"] = sync_value
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("post-update canary collection failed") from exc
    canary_valid = len(canaries) == canary_count and all(
        record.versions.policy == updated_policy
        and bool(record.turns)
        and all(turn.policy_version == updated_policy for turn in record.turns)
        and all(turn.policy_adapter == adapter_name for turn in record.turns)
        for record in canaries
    )
    if not canary_valid:
        manifest["status"] = "failed_post_update_canary"
        manifest["policy_sync"] = sync_value
        manifest["error"] = "canary policy or adapter receipt mismatch"
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("post-update canary did not use the published adapter")
    _write_jsonl(paths["post_update"], canaries)

    cursor_value: Optional[Mapping[str, Any]] = None
    if progress is not None:
        assert next_cursor_path is not None
        step = progress.current_step
        next_state = progress.preview_step(step_ordinal=step.step_ordinal)
        try:
            next_cursor_path.parent.mkdir(parents=True, exist_ok=True)
            next_state.write_once(next_cursor_path)
        except Exception as exc:
            manifest["status"] = "failed_cursor_commit"
            manifest["policy_sync"] = sync_value
            manifest["error"] = _safe_error(exc)
            manifest["completed_at"] = _utc_now()
            _write_json(paths["manifest"], manifest)
            raise SmokeRunError(
                "policy was updated but the exact HotpotQA cursor could not commit"
            ) from exc
        progress.commit_step_state(next_state)
        cursor_value = next_state.to_value()

    manifest.update(
        status="completed",
        policy_sync=sync_value,
        post_update_canaries={
            "collected": len(canaries),
            "adapter_name": adapter_name,
            "policy_version": updated_policy,
            "trajectory_ids": [record.trajectory_id for record in canaries],
        },
        completed_at=_utc_now(),
    )
    if cursor_value is not None:
        manifest["selection_receipt"]["cursor_after"] = dict(cursor_value)
    _write_json(paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/training_agentgraph_smoke.yaml",
        help="bounded smoke-training YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="select and persist the 14 tasks without model, API, or GPU work",
    )
    parser.add_argument(
        "--resume-initial-rollouts",
        action="store_true",
        help=(
            "strictly reuse the persisted frozen HotpotQA rollout batch; "
            "never recollect its paid Executor calls"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve(PROJECT_ROOT, args.config)
    try:
        manifest = asyncio.run(
            run_smoke(
                config_path,
                prepare_only=bool(args.prepare_only),
                resume_initial_rollouts=bool(args.resume_initial_rollouts),
                project_root=PROJECT_ROOT,
            )
        )
    except (ConfigurationError, SmokeRunError, ValueError, RuntimeError) as exc:
        print(f"smoke run failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_tasks": manifest["bounds"]["selected_tasks"],
                "expected_initial_rollouts": manifest["bounds"][
                    "expected_initial_rollouts"
                ],
                "max_optimizer_updates": manifest["bounds"]["max_optimizer_updates"],
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
