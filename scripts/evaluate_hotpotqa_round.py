#!/usr/bin/env python3
"""Run the fixed HotpotQA Direct-vs-AgentGraph architecture validation.

The runner is intentionally evaluation-only.  It reuses the SkillFlow-derived
SGLang/Supervisor boundary and the existing FlowSteer AgentGraph collector; it
does not call a trainer, optimizer, backward pass, policy update, or Skill loop.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _write_json,
    _write_jsonl,
    version_bundle_for,
)
from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentCallRecord,
    AgentRequest,
    ExecutionPhase,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.records import TaskRecord, TrajectoryRecord
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import evaluate_task


class HotpotRoundError(RuntimeError):
    """The fixed evaluation protocol could not complete as declared."""


DIRECT_CONTRACT = (
    "Chain evidence across passages to answer multi-hop questions. Answer briefly: "
    "name, date, number, or short phrase."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _git_state(root: Path) -> Mapping[str, Optional[str]]:
    def read(*args: str) -> Optional[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None

    return {
        "branch": read("branch", "--show-current"),
        "commit": read("rev-parse", "HEAD"),
    }


def validate_hotpot_config(config: Mapping[str, Any]) -> None:
    """Reject any accidental expansion beyond the declared held-out round."""

    validate_agent_graph_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    bounded = _mapping(config.get("hotpotqa_evaluation"), "hotpotqa_evaluation")
    director = _mapping(config.get("director"), "director")
    grpo = _mapping(config.get("grpo"), "grpo")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")
    gpu = _mapping(config.get("gpu"), "gpu")
    checks = {
        "experiment.phase": experiment.get("phase") == "hotpotqa_evaluation",
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "dataset_key": bounded.get("dataset_key") == "hotpotqa",
        "split": bounded.get("split") == "validation",
        "selection": bounded.get("selection") in {"sequential", "task_ids"},
        "rollouts_per_task": bounded.get("rollouts_per_task") == 1,
        "direct_model_id": bounded.get("direct_model_id") == "qwen3.5-9b-local",
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "grpo.enabled": grpo.get("enabled") is False,
        "optimizer_passes": grpo.get("optimization_passes_per_rollout_batch") == 0,
        "optimizer_updates": grpo.get("max_optimizer_updates") == 0,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.enabled": skills.get("enabled") is False,
        "gpu.training_enabled": gpu.get("training_enabled") is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "HotpotQA round violates fixed evaluation bounds: " + ", ".join(failed)
        )
    sample_count = bounded.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= 128
    ):
        raise ConfigurationError(
            "hotpotqa_evaluation.sample_count must be between 1 and 128"
        )
    if bounded.get("selection") == "task_ids":
        task_ids = bounded.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or len(task_ids) != sample_count
            or not all(isinstance(item, str) and item.strip() for item in task_ids)
            or len(set(task_ids)) != len(task_ids)
        ):
            raise ConfigurationError(
                "task_ids selection requires sample_count unique non-empty task IDs"
            )
    concurrency = bounded.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError("hotpotqa_evaluation.concurrency must be positive")
    for name in (
        "behavior_policy_version",
        "behavior_adapter_name",
        "behavior_adapter_checkpoint",
        "expected_server_weight_version",
    ):
        if not str(director.get(name, "")).strip():
            raise ConfigurationError(f"director.{name} must be non-empty")


def _paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config["storage"], "storage")
    names = {
        "selected": "selected_tasks_path",
        "direct": "direct_predictions_path",
        "trajectories": "trajectories_path",
        "failures": "failures_path",
        "paired": "paired_results_path",
        "wrong": "wrong_demos_path",
        "manifest": "manifest_path",
        "preflight": "preflight_receipt_path",
        "report_json": "report_json_path",
        "report_markdown": "report_markdown_path",
    }
    return {name: _resolve(root, str(storage[field])) for name, field in names.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HotpotRoundError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise HotpotRoundError(f"{path}:{line_number}: expected an object")
            values.append(dict(value))
    return values


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    _write_jsonl(temporary, values)
    temporary.replace(path)


def _select_tasks(config: Mapping[str, Any], root: Path, selected_path: Path) -> tuple[TaskRecord, ...]:
    data = _mapping(config["data"], "data")
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    count = int(bounded["sample_count"])
    source_path = _resolve(root, str(data["validation_path"]))
    candidates = tuple(
        task
        for task in iter_task_records(source_path, expected_split="validation")
        if _dataset_key(task) == "hotpotqa"
    )
    if len(candidates) < count:
        raise HotpotRoundError(
            f"validation contains only {len(candidates)} HotpotQA tasks; expected {count}"
        )
    if bounded.get("selection") == "task_ids":
        candidates_by_id = {task.task_id: task for task in candidates}
        requested = [str(task_id) for task_id in bounded["task_ids"]]
        missing = [task_id for task_id in requested if task_id not in candidates_by_id]
        if missing:
            raise HotpotRoundError(
                "requested HotpotQA task IDs are absent from validation: "
                + ", ".join(missing)
            )
        expected = tuple(candidates_by_id[task_id] for task_id in requested)
    else:
        expected = candidates[:count]
    if selected_path.exists():
        frozen = tuple(iter_task_records(selected_path, expected_split="validation"))
        if len(frozen) != count:
            raise HotpotRoundError("frozen HotpotQA selection has the wrong size")
        for expected_task, frozen_task in zip(expected, frozen, strict=True):
            if (
                expected_task.task_id != frozen_task.task_id
                or expected_task.question != frozen_task.question
                or expected_task.ground_truth != frozen_task.ground_truth
            ):
                raise HotpotRoundError(
                    "frozen HotpotQA task batch differs from the aligned validation data"
                )
        return frozen
    _atomic_jsonl(
        selected_path,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            for task in expected
        ],
    )
    return expected


def _by_task(values: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        task_id = value.get("task_id")
        if not isinstance(task_id, str):
            task = value.get("task")
            task_id = task.get("task_id") if isinstance(task, Mapping) else None
        if isinstance(task_id, str) and task_id not in result:
            result[task_id] = dict(value)
    return result


def _persist_ordered(
    path: Path,
    selected: Sequence[TaskRecord],
    values: Mapping[str, Mapping[str, Any]],
) -> None:
    _atomic_jsonl(path, [values[task.task_id] for task in selected if task.task_id in values])


async def _direct_one(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    index: int,
    *,
    model_id: str,
    protocol: str,
    seed: int,
    run_label: str,
) -> Mapping[str, Any]:
    model = backend.registry.require_model(model_id)
    provider = backend.registry.provider_for(model_id)
    run_id = f"{run_label}-direct-{index:04d}"
    request = AgentRequest(
        request_id=f"{run_id}:direct:single",
        run_id=run_id,
        graph_revision=0,
        problem=task.question,
        agent=AgentNode("direct", model_id, DIRECT_CONTRACT),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
        is_output_agent=True,
    )
    started_at = _utc_now()
    response = await backend.runtime.gateway.generate(request)
    execution = execution_record_from_call(AgentCallRecord(request, response))
    actual_seed = execution.metadata.get("response", {}).get("generation_seed")
    if actual_seed != seed:
        raise HotpotRoundError("Direct generation seed receipt differs from config")
    evaluation = await evaluate_task(task, response.text)
    return {
        "schema_version": "flowsteer.hotpotqa.direct_prediction.v1",
        "task_id": task.task_id,
        "task": task.to_dict(),
        "condition": "direct_local_qwen35_9b",
        "protocol": protocol,
        "model_id": model_id,
        "provider_id": provider.provider_id,
        "provider_model": model.model_name,
        "generation_seed": actual_seed,
        "final_answer": response.text,
        "evaluation": asdict(evaluation),
        "execution": execution.to_dict(),
        "started_at": started_at,
        "completed_at": _utc_now(),
    }


def _direct_resume_matches(
    value: Mapping[str, Any],
    *,
    task: TaskRecord,
    model_id: str,
    protocol: str,
    seed: int,
) -> bool:
    evaluation = value.get("evaluation")
    execution = value.get("execution")
    return bool(
        value.get("task_id") == task.task_id
        and value.get("model_id") == model_id
        and value.get("protocol") == protocol
        and value.get("generation_seed") == seed
        and isinstance(evaluation, Mapping)
        and isinstance(execution, Mapping)
        and evaluation.get("valid") is True
    )


def _trajectory_resume_matches(
    value: Mapping[str, Any],
    *,
    task: TaskRecord,
    condition_id: str,
    versions: Mapping[str, str],
) -> bool:
    embedded = value.get("task")
    return bool(
        isinstance(embedded, Mapping)
        and embedded.get("task_id") == task.task_id
        and value.get("condition_id") == condition_id
        and value.get("versions") == versions
        and isinstance(value.get("turns"), list)
        and isinstance(value.get("evaluation"), Mapping)
        and isinstance(value.get("explicit_finish"), bool)
    )


async def _collect_direct(
    backend: LiveSmokeBackend,
    selected: Sequence[TaskRecord],
    config: Mapping[str, Any],
    root: Path,
    path: Path,
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    experiment = _mapping(config["experiment"], "experiment")
    model_id = str(bounded["direct_model_id"])
    protocol = str(bounded["direct_protocol"])
    seed = int(experiment["seed"])
    run_label = str(experiment["name"])
    concurrency = int(bounded["concurrency"])
    direct_candidates = _read_jsonl(path)
    reuse_source = bounded.get("direct_reused_from")
    reuse_path: Optional[Path] = None
    if isinstance(reuse_source, str) and reuse_source.strip():
        reuse_path = _resolve(root, reuse_source)
        if not reuse_path.is_file():
            raise HotpotRoundError(
                f"declared Direct reuse source does not exist: {reuse_path}"
            )
        if reuse_path.resolve() != path.resolve():
            for value in _read_jsonl(reuse_path):
                copied = dict(value)
                copied["reuse_receipt"] = {
                    "reused": True,
                    "source": str(reuse_path),
                }
                direct_candidates.append(copied)
    by_task = {
        task_id: value
        for task_id, value in _by_task(direct_candidates).items()
        for task in selected
        if task.task_id == task_id
        and _direct_resume_matches(
            value,
            task=task,
            model_id=model_id,
            protocol=protocol,
            seed=seed,
        )
    }
    _persist_ordered(path, selected, by_task)
    manifest["direct_progress"] = {
        "completed": len(by_task),
        "reused_from": None if reuse_path is None else str(reuse_path),
        "reused_records": sum(
            value.get("reuse_receipt", {}).get("reused") is True
            for value in by_task.values()
        ),
        "newly_collected_records": sum(
            value.get("reuse_receipt", {}).get("reused") is not True
            for value in by_task.values()
        ),
        "failed_attempts": sum(
            item.get("condition") == "direct_local_qwen35_9b" for item in failures
        ),
    }
    _write_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int, task: TaskRecord) -> tuple[TaskRecord, Any]:
        async with semaphore:
            try:
                return task, await _direct_one(
                    backend,
                    task,
                    index,
                    model_id=model_id,
                    protocol=protocol,
                    seed=seed,
                    run_label=run_label,
                )
            except BaseException as exc:
                return task, exc

    jobs = [
        asyncio.create_task(run(index, task))
        for index, task in enumerate(selected)
        if task.task_id not in by_task
    ]
    for completed in asyncio.as_completed(jobs):
        task, result = await completed
        if isinstance(result, BaseException):
            failures.append(
                {
                    "task_id": task.task_id,
                    "condition": "direct_local_qwen35_9b",
                    "stage": "generation_or_evaluator",
                    "error": _safe_error(result),
                    "recorded_at": _utc_now(),
                }
            )
        else:
            by_task[task.task_id] = dict(result)
            _persist_ordered(path, selected, by_task)
        manifest["direct_progress"] = {
            "completed": len(by_task),
            "reused_from": None if reuse_path is None else str(reuse_path),
            "reused_records": sum(
                value.get("reuse_receipt", {}).get("reused") is True
                for value in by_task.values()
            ),
            "newly_collected_records": sum(
                value.get("reuse_receipt", {}).get("reused") is not True
                for value in by_task.values()
            ),
            "failed_attempts": sum(
                item.get("condition") == "direct_local_qwen35_9b" for item in failures
            ),
        }
        _write_json(manifest_path, manifest)
    return by_task


async def _collect_graph(
    backend: LiveSmokeBackend,
    selected: Sequence[TaskRecord],
    config: Mapping[str, Any],
    path: Path,
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    experiment = _mapping(config["experiment"], "experiment")
    director = _mapping(config["director"], "director")
    policy = str(director["behavior_policy_version"])
    condition_id = str(experiment["condition_id"])
    versions = {
        task.task_id: version_bundle_for(
            task,
            policy_version=policy,
            model_catalog_version=backend.model_catalog_version,
            prompt_version=str(experiment["prompt_version"]),
            tool_version=str(experiment["tool_version"]),
        )
        for task in selected
    }
    by_task = {
        task_id: value
        for task_id, value in _by_task(_read_jsonl(path)).items()
        for task in selected
        if task.task_id == task_id
        and _trajectory_resume_matches(
            value,
            task=task,
            condition_id=condition_id,
            versions=versions[task_id].to_dict(),
        )
    }
    # The collector commits to EvidenceStore before returning to this runner.
    # Recover that authoritative payload first if the process stopped between
    # the append and its ordered mirror checkpoint; never repay a successful
    # provider call merely because the mirror write was interrupted.
    selected_by_id = {task.task_id: task for task in selected}
    for value in backend.evidence_store.trajectories.payloads():
        embedded = value.get("task")
        task_id = embedded.get("task_id") if isinstance(embedded, Mapping) else None
        selected_task = selected_by_id.get(task_id)
        if (
            selected_task is not None
            and task_id not in by_task
            and _trajectory_resume_matches(
                value,
                task=selected_task,
                condition_id=condition_id,
                versions=versions[task_id].to_dict(),
            )
        ):
            by_task[task_id] = dict(value)
    _persist_ordered(path, selected, by_task)
    manifest["agentgraph_progress"] = {
        "completed": len(by_task),
        "failed_attempts": sum(
            item.get("condition") == "agentgraph" for item in failures
        ),
    }
    _write_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(int(bounded["concurrency"]))

    async def run(task: TaskRecord) -> tuple[TaskRecord, Any]:
        async with semaphore:
            try:
                trajectory = await backend.collect(
                    task,
                    0,
                    versions[task.task_id],
                    expected_task_split="validation",
                )
                return task, trajectory
            except BaseException as exc:
                return task, exc

    jobs = [
        asyncio.create_task(run(task))
        for task in selected
        if task.task_id not in by_task
    ]
    for completed in asyncio.as_completed(jobs):
        task, result = await completed
        if isinstance(result, BaseException):
            failures.append(
                {
                    "task_id": task.task_id,
                    "condition": "agentgraph",
                    "stage": "director_canvas_runtime_or_evaluator",
                    "error": _safe_error(result),
                    "recorded_at": _utc_now(),
                }
            )
        else:
            if not isinstance(result, TrajectoryRecord):
                raise HotpotRoundError("backend returned a non-trajectory result")
            by_task[task.task_id] = result.to_dict()
            _persist_ordered(path, selected, by_task)
        manifest["agentgraph_progress"] = {
            "completed": len(by_task),
            "failed_attempts": sum(
                item.get("condition") == "agentgraph" for item in failures
            ),
        }
        _write_json(manifest_path, manifest)
    return by_task


def _metrics(value: Optional[Mapping[str, Any]]) -> tuple[bool, float, float]:
    if value is None:
        return False, 0.0, 0.0
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("valid") is not True:
        return False, 0.0, 0.0
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, 0.0, 0.0
    return (
        True,
        float(metrics.get("exact_match", 0.0)),
        float(metrics.get("token_f1", 0.0)),
    )


def _output_inbox(trajectory: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if trajectory is None:
        return None
    turns = trajectory.get("turns")
    final_graph = turns[-1].get("graph_snapshot") if isinstance(turns, list) and turns else None
    if not isinstance(final_graph, Mapping):
        return None
    output_id = final_graph.get("output_agent_id")
    if not isinstance(output_id, str):
        return None
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        executions = turn.get("executions", ())
        if not isinstance(executions, list):
            continue
        for execution in reversed(executions):
            if not isinstance(execution, Mapping) or execution.get("agent_id") != output_id:
                continue
            metadata = execution.get("metadata")
            request = metadata.get("request") if isinstance(metadata, Mapping) else None
            if not isinstance(request, Mapping):
                continue
            return {
                "output_agent_id": output_id,
                "execution_id": execution.get("execution_id"),
                "agent": request.get("agent"),
                "model": request.get("model"),
                "phase": request.get("phase"),
                "upstream": request.get("upstream"),
                "own_draft": request.get("own_draft"),
                "peer_draft": request.get("peer_draft"),
                "rendered_messages": request.get("rendered_messages"),
            }
    return None


def _graph_telemetry(trajectory: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if trajectory is None:
        return {
            "director_turns": 0,
            "executor_calls": 0,
            "api_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        turns = []
    calls = [
        execution
        for turn in turns
        if isinstance(turn, Mapping)
        for execution in turn.get("executions", ())
        if isinstance(execution, Mapping)
    ]
    director_attempts = sum(
        int(turn.get("director_attempt_count") or 0)
        for turn in turns
        if isinstance(turn, Mapping)
    )
    executor_attempts = 0
    for call in calls:
        metadata = call.get("metadata")
        response = metadata.get("response") if isinstance(metadata, Mapping) else None
        if isinstance(response, Mapping):
            executor_attempts += int(response.get("attempt_count") or 0)
    return {
        "director_turns": len(turns),
        "executor_calls": len(calls),
        "api_attempts": director_attempts + executor_attempts,
        "input_tokens": sum(
            len(turn.get("prompt_token_ids", ()))
            for turn in turns
            if isinstance(turn, Mapping)
        )
        + sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(
            len(turn.get("output_token_ids", ()))
            for turn in turns
            if isinstance(turn, Mapping)
        )
        + sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_ms": sum(
            float(turn.get("director_latency_ms") or 0.0)
            for turn in turns
            if isinstance(turn, Mapping)
        )
        + sum(float(call.get("latency_ms") or 0.0) for call in calls),
    }


def _failure_type(
    direct: Optional[Mapping[str, Any]],
    trajectory: Optional[Mapping[str, Any]],
    direct_em: float,
    graph_em: float,
    graph_f1: float,
) -> str:
    if trajectory is None:
        return "agentgraph_operational_failure"
    if trajectory.get("explicit_finish") is not True:
        return "director_max_rounds"
    turns = trajectory.get("turns", ())
    feedback = " ".join(
        str(turn.get("canvas_feedback", ""))
        for turn in turns
        if isinstance(turn, Mapping)
    )
    if "execution_error=" in feedback:
        return "executor_or_provider_failure"
    if direct is None:
        return "direct_operational_failure_comparison_unavailable"
    if direct_em == 1.0 and graph_em == 0.0:
        return "architecture_regression_candidate"
    if direct_em == 0.0 and graph_em == 1.0:
        return "architecture_gain"
    if graph_em == 0.0 and graph_f1 > 0.0:
        return "partial_or_overlong_answer"
    if direct_em == 0.0 and graph_em == 0.0:
        return "shared_reasoning_or_model_failure_candidate"
    return "correct"


def _paired_rows(
    selected: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for task in selected:
        direct_value = direct.get(task.task_id)
        graph_value = trajectories.get(task.task_id)
        direct_valid, direct_em, direct_f1 = _metrics(direct_value)
        graph_valid, graph_em, graph_f1 = _metrics(graph_value)
        direct_execution = direct_value.get("execution") if direct_value else None
        direct_response = (
            direct_execution.get("metadata", {}).get("response", {})
            if isinstance(direct_execution, Mapping)
            else {}
        )
        direct_telemetry = {
            "api_attempts": int(direct_response.get("attempt_count") or 0),
            "input_tokens": int(direct_execution.get("input_tokens") or 0)
            if isinstance(direct_execution, Mapping)
            else 0,
            "output_tokens": int(direct_execution.get("output_tokens") or 0)
            if isinstance(direct_execution, Mapping)
            else 0,
            "latency_ms": float(direct_execution.get("latency_ms") or 0.0)
            if isinstance(direct_execution, Mapping)
            else 0.0,
        }
        rows.append(
            {
                "task_id": task.task_id,
                "question": task.question,
                "ground_truth": task.ground_truth,
                "direct": {
                    "available": direct_value is not None,
                    "valid": direct_valid,
                    "final_answer": direct_value.get("final_answer") if direct_value else None,
                    "exact_match": direct_em,
                    "token_f1": direct_f1,
                    "telemetry": direct_telemetry,
                    "execution": direct_execution,
                },
                "agentgraph": {
                    "available": graph_value is not None,
                    "valid": graph_valid,
                    "final_answer": graph_value.get("final_answer") if graph_value else None,
                    "exact_match": graph_em,
                    "token_f1": graph_f1,
                    "explicit_finish": graph_value.get("explicit_finish")
                    if graph_value
                    else False,
                    "termination_reason": graph_value.get("termination_reason")
                    if graph_value
                    else "collection_failed",
                    "trajectory_id": graph_value.get("trajectory_id") if graph_value else None,
                    "final_graph": (
                        graph_value.get("turns", [{}])[-1].get("graph_snapshot")
                        if graph_value and graph_value.get("turns")
                        else None
                    ),
                    "output_agent_inbox": _output_inbox(graph_value),
                    "telemetry": _graph_telemetry(graph_value),
                },
                "delta_exact_match": graph_em - direct_em,
                "delta_token_f1": graph_f1 - direct_f1,
                "failure_type": _failure_type(
                    direct_value,
                    graph_value,
                    direct_em,
                    graph_em,
                    graph_f1,
                ),
            }
        )
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]], condition: str) -> Mapping[str, Any]:
    total = len(rows)
    values = [row[condition] for row in rows]
    valid = [value for value in values if value.get("valid") is True]
    return {
        "denominator": total,
        "completed": sum(value.get("available") is True for value in values),
        "evaluator_valid": len(valid),
        "strict_exact_match": (
            sum(float(value.get("exact_match", 0.0)) for value in values) / total
            if total
            else 0.0
        ),
        "strict_token_f1": (
            sum(float(value.get("token_f1", 0.0)) for value in values) / total
            if total
            else 0.0
        ),
        "completed_only_exact_match": (
            sum(float(value.get("exact_match", 0.0)) for value in valid) / len(valid)
            if valid
            else None
        ),
        "completed_only_token_f1": (
            sum(float(value.get("token_f1", 0.0)) for value in valid) / len(valid)
            if valid
            else None
        ),
    }


def _report(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _aggregate(rows, "direct")
    graph = _aggregate(rows, "agentgraph")
    failure_counts = Counter(str(row["failure_type"]) for row in rows)
    wrong = [row for row in rows if float(row["agentgraph"]["exact_match"]) < 1.0]
    explicit_finished = sum(
        row["agentgraph"].get("explicit_finish") is True for row in rows
    )
    terminal_failures = sum(
        row["agentgraph"].get("available") is True
        and row["agentgraph"].get("termination_reason") == "max_rounds"
        for row in rows
    )
    operational_failures = sum(
        row["direct"].get("available") is not True
        or row["direct"].get("valid") is not True
        or row["agentgraph"].get("available") is not True
        or row["agentgraph"].get("valid") is not True
        for row in rows
    )
    return {
        "schema_version": "flowsteer.hotpotqa.round_report.v1",
        "dataset": "HotpotQA",
        "project_split": "validation",
        "native_source_split": "train",
        "input_context": "full_10_passages",
        "sample_count": len(rows),
        "evaluation_name": config["experiment"]["name"],
        "direct_local_baseline": direct,
        "agentgraph": graph,
        "agentgraph_minus_direct": {
            "exact_match": graph["strict_exact_match"] - direct["strict_exact_match"],
            "token_f1": graph["strict_token_f1"] - direct["strict_token_f1"],
        },
        "paper_references_percent": {
            "qwen3.5_9b_direct": {"exact_match": 60.94, "token_f1": 75.70},
            "flowsteer": {"exact_match": 89.84, "token_f1": 91.20},
            "skillflow": {"exact_match": 92.19, "token_f1": 93.95},
            "comparability_note": (
                "Reference numbers are contextual only; the paired Local Direct "
                "baseline on these fixed project-held-out samples is primary."
            ),
        },
        "failure_types": dict(sorted(failure_counts.items())),
        "wrong_demo_count": len(wrong),
        "explicit_finished_count": explicit_finished,
        "terminal_failure_count": terminal_failures,
        "operational_failure_count": operational_failures,
        "typical_wrong_demo_task_ids": [row["task_id"] for row in wrong[:10]],
        "policy_version": config["director"]["behavior_policy_version"],
        "policy_adapter": config["director"]["behavior_adapter_name"],
        "model_catalog_path": config["agent_graph"]["model_catalog_path"],
        "training_performed": False,
        "method_level_changes_performed": False,
        "known_limitations": [
            "A Director/API exception before terminal collection may not preserve partial turns.",
            "Paper references use a different published evaluation setup and are not a paired baseline.",
        ],
        "completed_at": _utc_now(),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    direct = report["direct_local_baseline"]
    graph = report["agentgraph"]
    delta = report["agentgraph_minus_direct"]
    failures = report["failure_types"]
    failure_lines = "\n".join(f"- `{name}`: {count}" for name, count in failures.items())
    return f"""# HotpotQA Architecture Validation — {report['evaluation_name']}

Fixed project-held-out samples: **{report['sample_count']}**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

AgentGraph explicit FINISH: **{report['explicit_finished_count']}/{report['sample_count']}**; natural max-round terminal failures: **{report['terminal_failure_count']}**; operational/evaluator failures: **{report['operational_failure_count']}**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | {direct['completed']} | {direct['evaluator_valid']} | {100 * direct['strict_exact_match']:.2f} | {100 * direct['strict_token_f1']:.2f} |
| AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * graph['strict_exact_match']:.2f} | {100 * graph['strict_token_f1']:.2f} |

AgentGraph − Direct: **{100 * delta['exact_match']:+.2f} EM**, **{100 * delta['token_f1']:+.2f} F1**.

## Failure types

{failure_lines or '- None'}

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
"""


def _stable_zero_check(
    tasks: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    checks: list[dict[str, Any]] = []
    for task in tasks:
        direct_value = direct.get(task.task_id)
        trajectory = trajectories.get(task.task_id)
        turns = trajectory.get("turns", ()) if trajectory else ()
        full_turn_receipts = bool(turns) and all(
            isinstance(turn, Mapping)
            and turn.get("receipt_verified") is True
            and turn.get("director_attempt_count")
            and turn.get("director_generation_seed") is not None
            and turn.get("director_latency_ms") is not None
            for turn in turns
        )
        passed = bool(
            direct_value
            and trajectory
            and trajectory.get("explicit_finish") is True
            and trajectory.get("final_answer") not in (None, "")
            and trajectory.get("evaluation", {}).get("valid") is True
            and _output_inbox(trajectory) is not None
            and full_turn_receipts
        )
        checks.append(
            {
                "task_id": task.task_id,
                "passed": passed,
                "direct_complete": direct_value is not None,
                "agentgraph_complete": trajectory is not None,
                "explicit_finish": trajectory.get("explicit_finish") if trajectory else False,
                "output_inbox_saved": _output_inbox(trajectory) is not None,
                "full_turn_receipts": full_turn_receipts,
            }
        )
    passed = bool(checks) and all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "criterion": "every_fixed_task_completed_the_full_chain",
        "checks": checks,
    }


async def run_hotpot_round(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prepare_only: bool = False,
    canary_only: bool = False,
) -> Mapping[str, Any]:
    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_hotpot_config(config)
    paths = _paths(config, root)
    selected = _select_tasks(config, root, paths["selected"])
    failures = _read_jsonl(paths["failures"])
    gpu = _mapping(config["gpu"], "gpu")
    configured_rollout_gpu = int(gpu["rollout_physical"])
    effective_rollout_gpu = int(
        os.environ.get("FLOWSTEER_ROLLOUT_GPU", configured_rollout_gpu)
    )
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.round_manifest.v1",
        "status": "prepared" if prepare_only else "runtime_preflight",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "git_start": _git_state(root),
        "selected_task_ids": [task.task_id for task in selected],
        "sample_count": len(selected),
        "fixed_split": "validation",
        "input_context": "full_10_passages",
        "training_enabled": False,
        "optimizer_updates": 0,
        "runtime_resource": {
            "configured_rollout_physical": configured_rollout_gpu,
            "effective_rollout_physical": effective_rollout_gpu,
            "resource_adaptation": effective_rollout_gpu != configured_rollout_gpu,
            "supervisor_port": int(os.environ.get("FLOWSTEER_SUPERVISOR_PORT", "8015")),
            "context_length": int(
                os.environ.get("FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH", "32768")
            ),
            "mem_fraction_static": float(
                os.environ.get("FLOWSTEER_SUPERVISOR_MEM_FRACTION", "0.82")
            ),
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    if prepare_only:
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        return manifest

    try:
        backend = LiveSmokeBackend.from_config(config, root, evaluation_only=True)
        director = _mapping(config["director"], "director")
        preflight = await asyncio.to_thread(
            backend.publisher.ensure_loaded_adapter,
            checkpoint_path=str(director["behavior_adapter_checkpoint"]),
            adapter_name=str(director["behavior_adapter_name"]),
        )
        known_answer = await evaluate_task(
            selected[0], f"<answer>{selected[0].ground_truth}</answer>"
        )
        if (
            not known_answer.valid
            or known_answer.metrics.get("exact_match") != 1.0
            or known_answer.metrics.get("token_f1") != 1.0
        ):
            raise HotpotRoundError("known-answer Hotpot evaluator preflight failed")
        preflight = {
            **dict(preflight),
            "evaluator_known_answer": asdict(known_answer),
        }
        _write_json(paths["preflight"], preflight)
    except Exception as exc:
        manifest.update(
            status="failed_runtime_preflight",
            error=_safe_error(exc),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        raise HotpotRoundError("HotpotQA runtime preflight failed") from exc

    active = selected[:2] if canary_only else selected
    manifest["status"] = "direct_baseline"
    _write_json(paths["manifest"], manifest)
    direct = await _collect_direct(
        backend,
        active,
        config,
        root,
        paths["direct"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    manifest["status"] = "agentgraph"
    _write_json(paths["manifest"], manifest)
    trajectories = await _collect_graph(
        backend,
        active,
        config,
        paths["trajectories"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    rows = _paired_rows(active, direct, trajectories)
    _atomic_jsonl(paths["paired"], rows)
    wrong = [row for row in rows if float(row["agentgraph"]["exact_match"]) < 1.0]
    _atomic_jsonl(paths["wrong"], wrong)
    stable_zero = _stable_zero_check(active, direct, trajectories)
    manifest["stable_zero"] = stable_zero
    if canary_only and not stable_zero["passed"]:
        manifest.update(status="failed_stable_zero", completed_at=_utc_now())
        _write_json(paths["manifest"], manifest)
        raise HotpotRoundError("one or more canary tasks failed the Stable Zero chain")

    if canary_only:
        manifest.update(
            status="stable_zero_confirmed",
            canary_task_count=len(active),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        return manifest

    report = _report(rows, config)
    _write_json(paths["report_json"], report)
    paths["report_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_markdown"].write_text(_report_markdown(report), encoding="utf-8")
    direct_progress = dict(manifest.get("direct_progress", {}))
    direct_progress["completed"] = len(direct)
    if report["operational_failure_count"]:
        final_status = "completed_with_operational_failures"
    elif report["terminal_failure_count"]:
        final_status = "completed_with_terminal_failures"
    else:
        final_status = "completed"
    manifest.update(
        status=final_status,
        direct_progress=direct_progress,
        agentgraph_progress={"completed": len(trajectories)},
        metrics={
            "direct": report["direct_local_baseline"],
            "agentgraph": report["agentgraph"],
            "delta": report["agentgraph_minus_direct"],
        },
        failure_type_counts=report["failure_types"],
        git_end=_git_state(root),
        completed_at=_utc_now(),
    )
    _write_json(paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_round_01.yaml",
        help="fixed HotpotQA Round-01 YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze the 128 validation tasks without starting a model or API",
    )
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="run the first two frozen tasks to prove the Stable Zero chain",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = asyncio.run(
            run_hotpot_round(
                _resolve(PROJECT_ROOT, args.config),
                project_root=PROJECT_ROOT,
                prepare_only=bool(args.prepare_only),
                canary_only=bool(args.canary_only),
            )
        )
    except (ConfigurationError, HotpotRoundError, ValueError, RuntimeError) as exc:
        print(f"HotpotQA round failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
