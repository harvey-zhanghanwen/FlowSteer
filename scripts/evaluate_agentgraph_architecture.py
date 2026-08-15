#!/usr/bin/env python3
"""Run one frozen AgentGraph architecture task per configured dataset.

This is an evaluation-only loop.  It reuses the smoke runner's live Director,
AgentGraph runtime, exact receipts, and terminal evaluators, but it never calls
the trainer or policy publisher.  The resulting seven-task batch is therefore
an architecture diagnostic, not an optimizer update or a held-out benchmark.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_agentgraph_smoke import (
    EXPECTED_SOURCE_ORDER,
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _write_json,
    _write_jsonl,
    select_smoke_tasks,
    version_bundle_for,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.records import TrajectoryRecord
from src.interactive.task_dataset import iter_task_records


class ArchitectureEvaluationError(RuntimeError):
    """The bounded architecture diagnostic is malformed or wholly unavailable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def validate_architecture_evaluation_config(config: Mapping[str, Any]) -> None:
    """Keep this loop at exactly one task and one rollout per source."""

    validate_agent_graph_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    evaluation = _mapping(
        config.get("architecture_evaluation"), "architecture_evaluation"
    )
    director = _mapping(config.get("director"), "director")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")
    checks = {
        "experiment.phase": experiment.get("phase") == "architecture_evaluation",
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "architecture_evaluation.split": evaluation.get("split") == "train",
        "architecture_evaluation.tasks_per_dataset": (
            evaluation.get("tasks_per_dataset") == 1
        ),
        "architecture_evaluation.rollouts_per_task": (
            evaluation.get("rollouts_per_task") == 1
        ),
        "architecture_evaluation.expected_total_tasks": (
            evaluation.get("expected_total_tasks") == 7
        ),
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.enabled": skills.get("enabled") is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "architecture evaluation violates fixed bounds: " + ", ".join(failed)
        )
    source_order = tuple(str(value) for value in evaluation.get("source_order", ()))
    if source_order != EXPECTED_SOURCE_ORDER:
        raise ConfigurationError(
            "architecture_evaluation.source_order must contain the fixed seven-source order"
        )
    skip = evaluation.get("skip_per_dataset", 0)
    if type(skip) is not int or skip < 0:
        raise ConfigurationError(
            "architecture_evaluation.skip_per_dataset must be a non-negative integer"
        )
    for field_name in (
        "behavior_policy_version",
        "expected_server_weight_version",
    ):
        value = director.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"director.{field_name} must be non-empty")


def _metric_name(source: str) -> str:
    return {
        "hotpotqa": "token_f1",
        "triviaqa": "token_f1",
        "aime_2026": "exact_match",
        "healthbench_professional": "grpo_reward",
        "webshop": "success",
        "alfworld": "success",
        "swe_bench": "resolved",
    }[source]


def _trajectory_mapping(value: TrajectoryRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, TrajectoryRecord):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("trajectory must be a TrajectoryRecord or mapping")
    return dict(value)


def _read_trajectory_mappings(path: Path) -> list[dict[str, Any]]:
    """Read only complete JSONL records; a truncated artifact is not resumable."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArchitectureEvaluationError(
                    f"{path}:{line_number}: invalid trajectory JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ArchitectureEvaluationError(
                    f"{path}:{line_number}: trajectory must be a mapping"
                )
            records.append(dict(value))
    return records


def _persist_trajectory_mappings(
    path: Path,
    selected: Sequence[Any],
    by_task: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically checkpoint completed rollouts in frozen task order."""

    ordered = [by_task[task.task_id] for task in selected if task.task_id in by_task]
    temporary = path.with_name(f".{path.name}.partial")
    _write_jsonl(temporary, ordered)
    temporary.replace(path)


def _resume_matches(
    value: Mapping[str, Any],
    *,
    task_id: str,
    condition_id: str,
    versions: Mapping[str, str],
) -> bool:
    """Require the exact task, condition, and full VersionBundle receipt."""

    task = value.get("task")
    evaluation = value.get("evaluation")
    return bool(
        isinstance(task, Mapping)
        and task.get("task_id") == task_id
        and value.get("condition_id") == condition_id
        and value.get("versions") == versions
        and isinstance(value.get("turns"), list)
        and isinstance(evaluation, Mapping)
        and "termination_reason" in value
        and isinstance(value.get("explicit_finish"), bool)
    )


def _report(
    selected: Sequence[Any],
    trajectories: Sequence[TrajectoryRecord | Mapping[str, Any]],
    failures: Mapping[str, str],
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for trajectory in trajectories:
        value = _trajectory_mapping(trajectory)
        task = value.get("task")
        if isinstance(task, Mapping) and isinstance(task.get("task_id"), str):
            by_task[str(task["task_id"])] = value
    rows: list[dict[str, Any]] = []
    measurable_scores: list[float] = []
    for task in selected:
        source = _dataset_key(task)
        record = by_task.get(task.task_id)
        metric = _metric_name(source)
        if record is None:
            rows.append(
                {
                    "dataset_key": source,
                    "source": task.metadata.get("source", source),
                    "task_id": task.task_id,
                    "valid_samples": 0,
                    "correct_or_successful": 0,
                    "metric": metric,
                    "metric_value": None,
                    "status": "collection_failed",
                    "reason": failures.get(task.task_id, "missing trajectory"),
                }
            )
            continue
        evaluation = record.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            evaluation = {}
        explicit_finish = record.get("explicit_finish") is True
        termination_reason = str(record.get("termination_reason", ""))
        max_rounds_failure = not explicit_finish and termination_reason == "max_rounds"
        evaluator_valid = evaluation.get("valid") is True
        raw_reward = evaluation.get("reward")
        score = (
            0.0
            if max_rounds_failure
            else float(raw_reward)
            if evaluator_valid and raw_reward is not None
            else None
        )
        if score is not None:
            measurable_scores.append(score)
        turns = record.get("turns", ())
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
            turns = ()
        executor_calls = sum(
            len(turn.get("executions", ()))
            for turn in turns
            if isinstance(turn, Mapping)
            and isinstance(turn.get("executions", ()), Sequence)
        )
        versions = record.get("versions", {})
        if not isinstance(versions, Mapping):
            versions = {}
        first_turn = turns[0] if turns and isinstance(turns[0], Mapping) else {}
        evaluator_reason = str(evaluation.get("reason", ""))
        rows.append(
            {
                "dataset_key": source,
                "source": task.metadata.get("source", source),
                "task_id": task.task_id,
                "valid_samples": int(score is not None),
                "correct_or_successful": int(score is not None and score >= 1.0),
                "metric": metric,
                "metric_value": score,
                "status": (
                    "terminal_failure"
                    if max_rounds_failure
                    else "measured"
                    if score is not None
                    else "unmeasurable"
                ),
                "reason": (
                    "director_max_rounds_without_explicit_finish"
                    if max_rounds_failure
                    else evaluator_reason
                ),
                "evaluator_reason": evaluator_reason,
                "evaluator_reward_ignored": (
                    raw_reward
                    if max_rounds_failure and raw_reward not in (None, 0, 0.0)
                    else None
                ),
                "termination_reason": termination_reason,
                "explicit_finish": explicit_finish,
                "director_turns": len(turns),
                "executor_calls": executor_calls,
                "policy_version": versions.get("policy"),
                "policy_adapter": first_turn.get("policy_adapter"),
                "evaluator_version": evaluation.get("evaluator_version"),
            }
        )
    return {
        "schema_version": "flowsteer.agentgraph.architecture_evaluation.v1",
        "scope": "one_train_task_per_dataset_architecture_diagnostic",
        "heldout_validation": False,
        "datasets": rows,
        "measurable_dataset_count": len(measurable_scores),
        "diagnostic_macro_mean": (
            sum(measurable_scores) / len(measurable_scores)
            if measurable_scores
            else None
        ),
        "stop_threshold_assessed": False,
    }


async def run_architecture_evaluation(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_architecture_evaluation_config(config)
    experiment = _mapping(config["experiment"], "experiment")
    bounded = _mapping(
        config["architecture_evaluation"], "architecture_evaluation"
    )
    storage = _mapping(config["storage"], "storage")
    data = _mapping(config["data"], "data")
    director = _mapping(config["director"], "director")

    train_path = _resolve(root, str(data["train_path"]))
    selected = select_smoke_tasks(
        iter_task_records(train_path, expected_split="train"),
        source_order=tuple(str(value) for value in bounded["source_order"]),
        per_source=1,
        require_unique_base_tasks=True,
        skip_per_source=int(bounded.get("skip_per_dataset", 0)),
        expected_split="train",
    )
    if len(selected) != 7:
        raise ArchitectureEvaluationError("the diagnostic did not select seven tasks")

    selected_path = _resolve(root, str(storage["selected_tasks_path"]))
    trajectories_path = _resolve(root, str(storage["trajectories_path"]))
    report_path = _resolve(root, str(storage["report_path"]))
    manifest_path = _resolve(root, str(storage["manifest_path"]))
    _write_jsonl(
        selected_path,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            for task in selected
        ],
    )

    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.agentgraph.architecture_manifest.v1",
        "status": "collecting",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "train_path": str(train_path),
        "training_enabled": False,
        "optimizer_updates": 0,
        "selected_task_ids": [task.task_id for task in selected],
        "selected_by_source": dict(
            sorted(Counter(_dataset_key(task) for task in selected).items())
        ),
        "behavior_policy_version": str(director["behavior_policy_version"]),
        "behavior_policy_adapter": director.get("behavior_adapter_name"),
        "condition_id": str(experiment["condition_id"]),
        "artifacts": {
            "selected_tasks": str(selected_path),
            "trajectories": str(trajectories_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }
    _write_json(manifest_path, manifest)

    try:
        backend = LiveSmokeBackend.from_config(config, root)
    except Exception as exc:
        manifest.update(
            status="failed_runtime_setup",
            error=_safe_error(exc),
            completed_at=_utc_now(),
        )
        _write_json(manifest_path, manifest)
        raise ArchitectureEvaluationError("runtime setup failed") from exc

    policy = str(director["behavior_policy_version"])
    prompt_version = str(experiment["prompt_version"])
    tool_version = str(experiment["tool_version"])
    condition_id = str(experiment["condition_id"])
    expected_versions = {
        task.task_id: version_bundle_for(
            task,
            policy_version=policy,
            model_catalog_version=backend.model_catalog_version,
            prompt_version=prompt_version,
            tool_version=tool_version,
        )
        for task in selected
    }

    trajectories_by_task: dict[str, dict[str, Any]] = {}
    for value in _read_trajectory_mappings(trajectories_path):
        task = value.get("task")
        task_id = task.get("task_id") if isinstance(task, Mapping) else None
        if not isinstance(task_id, str) or task_id in trajectories_by_task:
            continue
        versions = expected_versions.get(task_id)
        if versions is not None and _resume_matches(
            value,
            task_id=task_id,
            condition_id=condition_id,
            versions=versions.to_dict(),
        ):
            trajectories_by_task[task_id] = value

    reused = len(trajectories_by_task)
    _persist_trajectory_mappings(
        trajectories_path,
        selected,
        trajectories_by_task,
    )
    missing = [
        (index, task)
        for index, task in enumerate(selected)
        if task.task_id not in trajectories_by_task
    ]
    failures: dict[str, str] = {}
    fresh_collected = 0

    async def collect_one(index: int, task: Any) -> tuple[Any, Any]:
        try:
            result = await backend.collect(task, index, expected_versions[task.task_id])
        except BaseException as exc:  # preserve independent task failure isolation
            return task, exc
        return task, result

    jobs = [asyncio.create_task(collect_one(index, task)) for index, task in missing]
    for completed in asyncio.as_completed(jobs):
        task, result = await completed
        if isinstance(result, BaseException):
            failures[task.task_id] = _safe_error(result)
        else:
            trajectories_by_task[task.task_id] = _trajectory_mapping(result)
            fresh_collected += 1
            _persist_trajectory_mappings(
                trajectories_path,
                selected,
                trajectories_by_task,
            )
        manifest.update(
            resumed_trajectories=reused,
            fresh_collected=fresh_collected,
            collection_failures=failures,
        )
        _write_json(manifest_path, manifest)

    trajectories = [
        trajectories_by_task[task.task_id]
        for task in selected
        if task.task_id in trajectories_by_task
    ]
    report = _report(selected, trajectories, failures)
    _write_json(report_path, report)
    manifest.update(
        status=("completed" if trajectories else "failed_all_collections"),
        collected=len(trajectories),
        resumed_trajectories=reused,
        fresh_collected=fresh_collected,
        collection_failures=failures,
        measurable_dataset_count=report["measurable_dataset_count"],
        diagnostic_macro_mean=report["diagnostic_macro_mean"],
        completed_at=_utc_now(),
    )
    _write_json(manifest_path, manifest)
    if not trajectories:
        raise ArchitectureEvaluationError("all seven architecture tasks failed")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_agentgraph_architecture.yaml",
        help="bounded one-task-per-dataset architecture evaluation YAML",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve(PROJECT_ROOT, args.config)
    try:
        manifest = asyncio.run(
            run_architecture_evaluation(config_path, project_root=PROJECT_ROOT)
        )
    except (ArchitectureEvaluationError, ConfigurationError, ValueError) as exc:
        print(f"architecture evaluation failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
