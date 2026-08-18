#!/usr/bin/env python3
"""Run a fixed AIME 2026 or HealthBench Professional evaluation round.

This runner is evaluation-only.  It keeps the frozen task, resume, Canvas,
trajectory-receipt, and Stable Zero boundaries from
``evaluate_hotpotqa_round.py`` and uses ``LiveSmokeBackend`` for both the
Director and AgentGraph execution.  It never trains, performs a backward pass,
updates an optimizer, or publishes model/Skill state.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import evaluate_hotpotqa_round as hotpot_round
from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _write_json,
    evaluator_version_for,
)
from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentCallRecord,
    AgentRequest,
    ExecutionPhase,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.graph_diagnostics import (
    aggregate_trajectory_diagnostics,
    diagnose_trajectory,
)
from src.interactive.records import TaskRecord
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import evaluate_task


# These are aliases, rather than local reimplementations, so the completion
# benchmarks retain the HotpotQA runner's checkpoint and receipt boundaries.
_atomic_jsonl = hotpot_round._atomic_jsonl
_by_task = hotpot_round._by_task
_collect_graph = hotpot_round._collect_graph
_git_state = hotpot_round._git_state
_graph_telemetry = hotpot_round._graph_telemetry
_output_inbox = hotpot_round._output_inbox
_read_jsonl = hotpot_round._read_jsonl
_stable_zero_check = hotpot_round._stable_zero_check
_trajectory_resume_matches = hotpot_round._trajectory_resume_matches


class CompletionBenchmarkRoundError(RuntimeError):
    """The fixed AIME/HealthBench evaluation protocol could not complete."""


_BENCHMARKS: Mapping[str, Mapping[str, Any]] = {
    "aime_2026": {
        "label": "AIME 2026",
        "section_names": ("aime2026_evaluation", "aime_2026_evaluation"),
        "phase_names": ("aime2026_evaluation", "aime_2026_evaluation"),
        "primary_metric": "exact_match",
    },
    "healthbench_professional": {
        "label": "HealthBench Professional",
        "section_names": (
            "healthbench_evaluation",
            "healthbench_professional_evaluation",
        ),
        "phase_names": (
            "healthbench_evaluation",
            "healthbench_professional_evaluation",
        ),
        "primary_metric": "raw_score",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _evaluation_section(
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    """Return the one declared dataset-specific evaluation section."""

    known_names = {
        name
        for specification in _BENCHMARKS.values()
        for name in specification["section_names"]
    }
    present = [name for name in sorted(known_names) if name in config]
    if len(present) != 1:
        raise ConfigurationError(
            "exactly one AIME/HealthBench dataset-specific evaluation section "
            "must be configured"
        )
    section_name = present[0]
    bounded = _mapping(config[section_name], section_name)
    dataset_key = str(bounded.get("dataset_key", "")).strip()
    specification = _BENCHMARKS.get(dataset_key)
    if specification is None or section_name not in specification["section_names"]:
        raise ConfigurationError(
            f"{section_name}.dataset_key is not compatible with the section name"
        )
    return section_name, bounded


def _skill_evaluation_mode(config: Mapping[str, Any]) -> bool:
    skills = _mapping(config.get("skills"), "skills")
    deployment = _mapping(config.get("deployment"), "deployment")
    if skills.get("enabled") is False:
        return True
    return bool(
        skills.get("enabled") is True
        and isinstance(skills.get("store_path"), str)
        and str(skills.get("store_path", "")).strip()
        and type(skills.get("retrieval_top_k")) is int
        and int(skills["retrieval_top_k"]) > 0
        and type(skills.get("current_epoch")) is int
        and int(skills["current_epoch"]) >= 1
        and deployment.get("exploration_beta") == 0.0
        and deployment.get("allow_forced_probes") is False
        and deployment.get("active_skills_only") is True
        and deployment.get("require_version_compatible_skills") is True
    )


def validate_completion_benchmark_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless the config describes a bounded evaluation-only run."""

    validate_agent_graph_config(config)
    section_name, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    specification = _BENCHMARKS[dataset_key]
    experiment = _mapping(config.get("experiment"), "experiment")
    director = _mapping(config.get("director"), "director")
    grpo = _mapping(config.get("grpo"), "grpo")
    exploration = _mapping(config.get("exploration"), "exploration")
    gpu = _mapping(config.get("gpu"), "gpu")
    checks = {
        "experiment.phase": experiment.get("phase")
        in specification["phase_names"],
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "split": bounded.get("split") in {"train", "validation", "test"},
        "selection": bounded.get("selection") in {"sequential", "task_ids"},
        "rollouts_per_task": bounded.get("rollouts_per_task") == 1,
        "direct_model_id": bounded.get("direct_model_id") == "qwen3.5-9b-local",
        "direct_protocol": bool(str(bounded.get("direct_protocol", "")).strip()),
        "direct_contract": bool(str(bounded.get("direct_contract", "")).strip()),
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "director.execute_on_edit": director.get("execute_on_edit") is True,
        "grpo.enabled": grpo.get("enabled") is False,
        "optimizer_passes": grpo.get("optimization_passes_per_rollout_batch") == 0,
        "optimizer_updates": grpo.get("max_optimizer_updates") == 0,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.evaluation_mode": _skill_evaluation_mode(config),
        "gpu.training_enabled": gpu.get("training_enabled") is False,
    }
    if dataset_key == "healthbench_professional":
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        checks["evaluation.healthbench_judge_model"] = bool(
            str(evaluation.get("healthbench_judge_model", "")).strip()
        )
        checks["evaluation.healthbench_judge_catalog_path"] = bool(
            str(evaluation.get("healthbench_judge_catalog_path", "")).strip()
        )
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            f"{section_name} violates fixed evaluation bounds: " + ", ".join(failed)
        )

    sample_count = bounded.get("sample_count")
    official_aime = dataset_key == "aime_2026" and (
        bounded.get("official_2026_only") is True
        or bounded.get("benchmark_slice") == "official_aime_2026"
    )
    maximum = 30 if official_aime else 128
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= maximum
    ):
        raise ConfigurationError(
            f"{section_name}.sample_count must be between 1 and {maximum}"
        )
    concurrency = bounded.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError(f"{section_name}.concurrency must be positive")
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
    if bounded.get("official_2026_only") is True and dataset_key != "aime_2026":
        raise ConfigurationError("official_2026_only is valid only for AIME 2026")
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


def _benchmark_slice(task: TaskRecord) -> str:
    value = task.metadata.get("benchmark_slice")
    return str(value).strip() if value is not None else ""


def _select_tasks(
    config: Mapping[str, Any], root: Path, selected_path: Path
) -> tuple[TaskRecord, ...]:
    """Freeze exactly the aligned records selected by the benchmark section."""

    _, bounded = _evaluation_section(config)
    data = _mapping(config["data"], "data")
    dataset_key = str(bounded["dataset_key"])
    split = str(bounded["split"])
    source_field = {
        "train": "train_path",
        "validation": "validation_path",
        "test": "test_path",
    }[split]
    source = _resolve(root, str(data[source_field]))
    requested_slice = bounded.get("benchmark_slice")
    if bounded.get("official_2026_only") is True:
        requested_slice = "official_aime_2026"
    candidates = tuple(
        task
        for task in iter_task_records(source, expected_split=split)
        if _dataset_key(task) == dataset_key
        and (
            not requested_slice
            or _benchmark_slice(task) == str(requested_slice)
        )
    )
    count = int(bounded["sample_count"])
    if len(candidates) < count:
        suffix = f" in benchmark_slice={requested_slice}" if requested_slice else ""
        raise CompletionBenchmarkRoundError(
            f"aligned {split} contains only {len(candidates)} {dataset_key} tasks"
            f"{suffix}; expected {count}"
        )
    if bounded.get("selection") == "task_ids":
        by_id = {task.task_id: task for task in candidates}
        requested = [str(task_id) for task_id in bounded["task_ids"]]
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise CompletionBenchmarkRoundError(
                f"requested {dataset_key} task IDs are absent from {split}: "
                + ", ".join(missing)
            )
        expected = tuple(by_id[task_id] for task_id in requested)
    else:
        expected = candidates[:count]

    if selected_path.exists():
        frozen = tuple(iter_task_records(selected_path, expected_split=split))
        if len(frozen) != count:
            raise CompletionBenchmarkRoundError(
                f"frozen {dataset_key} selection has the wrong size"
            )
        for expected_task, frozen_task in zip(expected, frozen, strict=True):
            if (
                expected_task.task_id != frozen_task.task_id
                or expected_task.question != frozen_task.question
                or expected_task.ground_truth != frozen_task.ground_truth
            ):
                raise CompletionBenchmarkRoundError(
                    f"frozen {dataset_key} selection differs from aligned {split} data"
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


async def _evaluate_prediction(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    prediction: str,
) -> Any:
    """Use the existing evaluator, including an already configured HB judge."""

    if _dataset_key(task) == "healthbench_professional":
        return await evaluate_task(
            task,
            prediction,
            judge=backend.judge,
            judge_model=backend.judge_model,
        )
    return await evaluate_task(task, prediction)


async def _direct_one(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    index: int,
    *,
    model_id: str,
    protocol: str,
    contract: str,
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
        agent=AgentNode("direct", model_id, contract),
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
        raise CompletionBenchmarkRoundError(
            "Direct generation seed receipt differs from config"
        )
    evaluation = await _evaluate_prediction(backend, task, response.text)
    dataset_key = _dataset_key(task)
    return {
        "schema_version": "flowsteer.completion_benchmark.direct_prediction.v1",
        "dataset_key": dataset_key,
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
    """Checkpoint and resume the paired Direct condition task-by-task."""

    _, bounded = _evaluation_section(config)
    experiment = _mapping(config["experiment"], "experiment")
    model_id = str(bounded["direct_model_id"])
    protocol = str(bounded["direct_protocol"])
    contract = str(bounded["direct_contract"])
    seed = int(bounded.get("direct_generation_seed", experiment["seed"]))
    run_label = str(experiment["name"])
    concurrency = int(bounded["concurrency"])
    direct_candidates = _read_jsonl(path)
    reuse_source = bounded.get("direct_reused_from")
    reuse_path: Optional[Path] = None
    if isinstance(reuse_source, str) and reuse_source.strip():
        reuse_path = _resolve(root, reuse_source)
        if not reuse_path.is_file():
            raise CompletionBenchmarkRoundError(
                f"declared Direct reuse source does not exist: {reuse_path}"
            )
        if reuse_path.resolve() != path.resolve():
            reused = []
            for value in _read_jsonl(reuse_path):
                copied = dict(value)
                copied["reuse_receipt"] = {"reused": True, "source": str(reuse_path)}
                reused.append(copied)
            direct_candidates = reused + direct_candidates

    selected_by_id = {task.task_id: task for task in selected}
    rescored: list[dict[str, Any]] = []
    for candidate in direct_candidates:
        task = selected_by_id.get(candidate.get("task_id"))
        evaluation = candidate.get("evaluation")
        if (
            task is not None
            and candidate.get("model_id") == model_id
            and candidate.get("protocol") == protocol
            and candidate.get("generation_seed") == seed
            and isinstance(candidate.get("final_answer"), str)
            and (
                not isinstance(evaluation, Mapping)
                or evaluation.get("evaluator_version") != evaluator_version_for(task)
            )
        ):
            updated = dict(candidate)
            updated["evaluation"] = asdict(
                await _evaluate_prediction(backend, task, str(candidate["final_answer"]))
            )
            updated["rescore_receipt"] = {
                "mode": "offline_existing_prediction",
                "source_evaluator_version": (
                    evaluation.get("evaluator_version")
                    if isinstance(evaluation, Mapping)
                    else None
                ),
                "target_evaluator_version": evaluator_version_for(task),
            }
            rescored.append(updated)
        else:
            rescored.append(dict(candidate))

    by_task = {
        task_id: value
        for task_id, value in _by_task(rescored).items()
        for task in selected
        if task.task_id == task_id
        and hotpot_round._direct_resume_matches(
            value,
            task=task,
            model_id=model_id,
            protocol=protocol,
            seed=seed,
        )
    }
    hotpot_round._persist_ordered(path, selected, by_task)

    def checkpoint() -> None:
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
                item.get("condition") == "direct_local_qwen35_9b"
                for item in failures
            ),
        }
        _write_json(manifest_path, manifest)

    checkpoint()
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
                    contract=contract,
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
            hotpot_round._persist_ordered(path, selected, by_task)
        checkpoint()
    return by_task


def _metric(
    value: Optional[Mapping[str, Any]], dataset_key: str
) -> tuple[bool, float]:
    if value is None:
        return False, 0.0
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("valid") is not True:
        return False, 0.0
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, 0.0
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    raw = metrics.get(metric_name)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return False, 0.0
    value_float = float(raw)
    return (math.isfinite(value_float), value_float if math.isfinite(value_float) else 0.0)


def _direct_telemetry(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    execution = value.get("execution") if value else None
    response = (
        execution.get("metadata", {}).get("response", {})
        if isinstance(execution, Mapping)
        else {}
    )
    return {
        "api_attempts": int(response.get("attempt_count") or 0),
        "input_tokens": (
            int(execution.get("input_tokens") or 0)
            if isinstance(execution, Mapping)
            else 0
        ),
        "output_tokens": (
            int(execution.get("output_tokens") or 0)
            if isinstance(execution, Mapping)
            else 0
        ),
        "latency_ms": (
            float(execution.get("latency_ms") or 0.0)
            if isinstance(execution, Mapping)
            else 0.0
        ),
    }


def _failure_type(
    direct_value: Optional[Mapping[str, Any]],
    graph_value: Optional[Mapping[str, Any]],
    *,
    direct_valid: bool,
    graph_valid: bool,
    direct_score: float,
    graph_score: float,
    dataset_key: str,
) -> str:
    if direct_value is None or not direct_valid:
        return "direct_operational_or_evaluator_failure"
    if graph_value is None or not graph_valid:
        return "agentgraph_operational_or_evaluator_failure"
    if graph_value.get("explicit_finish") is not True:
        return "agentgraph_terminal_failure"
    if dataset_key == "aime_2026":
        if graph_score == 1.0 and direct_score == 0.0:
            return "agentgraph_exact_match_gain"
        if graph_score == 0.0 and direct_score == 1.0:
            return "agentgraph_exact_match_regression"
        return "both_exact" if graph_score == 1.0 else "both_incorrect"
    if graph_score > direct_score:
        return "agentgraph_higher_raw_score"
    if graph_score < direct_score:
        return "direct_higher_raw_score"
    return "equal_raw_score"


def _paired_rows(
    selected: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    dataset_key: str,
) -> list[dict[str, Any]]:
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    rows: list[dict[str, Any]] = []
    for task in selected:
        direct_value = direct.get(task.task_id)
        graph_value = trajectories.get(task.task_id)
        direct_valid, direct_score = _metric(direct_value, dataset_key)
        graph_valid, graph_score = _metric(graph_value, dataset_key)
        direct_execution = direct_value.get("execution") if direct_value else None
        rows.append(
            {
                "schema_version": "flowsteer.completion_benchmark.paired_result.v1",
                "dataset_key": dataset_key,
                "task_id": task.task_id,
                "question": task.question,
                "ground_truth": task.ground_truth,
                "primary_metric": metric_name,
                "direct": {
                    "available": direct_value is not None,
                    "valid": direct_valid,
                    "final_answer": (
                        direct_value.get("final_answer") if direct_value else None
                    ),
                    metric_name: direct_score,
                    "evaluation": (
                        direct_value.get("evaluation") if direct_value else None
                    ),
                    "telemetry": _direct_telemetry(direct_value),
                    "execution": direct_execution,
                },
                "agentgraph": {
                    "available": graph_value is not None,
                    "valid": graph_valid,
                    "final_answer": (
                        graph_value.get("final_answer") if graph_value else None
                    ),
                    metric_name: graph_score,
                    "evaluation": (
                        graph_value.get("evaluation") if graph_value else None
                    ),
                    "explicit_finish": (
                        graph_value.get("explicit_finish") if graph_value else False
                    ),
                    "termination_reason": (
                        graph_value.get("termination_reason")
                        if graph_value
                        else "collection_failed"
                    ),
                    "trajectory_id": (
                        graph_value.get("trajectory_id") if graph_value else None
                    ),
                    "final_graph": (
                        graph_value.get("turns", [{}])[-1].get("graph_snapshot")
                        if graph_value and graph_value.get("turns")
                        else None
                    ),
                    "output_agent_inbox": _output_inbox(graph_value),
                    "telemetry": _graph_telemetry(graph_value),
                    "graph_diagnostic": (
                        diagnose_trajectory(graph_value).to_dict()
                        if graph_value is not None
                        else None
                    ),
                },
                f"delta_{metric_name}": graph_score - direct_score,
                "failure_type": _failure_type(
                    direct_value,
                    graph_value,
                    direct_valid=direct_valid,
                    graph_valid=graph_valid,
                    direct_score=direct_score,
                    graph_score=graph_score,
                    dataset_key=dataset_key,
                ),
            }
        )
    return rows


def _aggregate(
    rows: Sequence[Mapping[str, Any]], condition: str, dataset_key: str
) -> Mapping[str, Any]:
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    total = len(rows)
    values = [row[condition] for row in rows]
    valid = [value for value in values if value.get("valid") is True]
    strict = (
        sum(float(value.get(metric_name, 0.0)) for value in values) / total
        if total
        else 0.0
    )
    completed_only = (
        sum(float(value.get(metric_name, 0.0)) for value in valid) / len(valid)
        if valid
        else None
    )
    result = {
        "denominator": total,
        "completed": sum(value.get("available") is True for value in values),
        "evaluator_valid": len(valid),
        f"strict_{metric_name}": strict,
        f"completed_only_{metric_name}": completed_only,
    }
    # Official AIME accuracy is the dataset average of per-problem exact match.
    if dataset_key == "aime_2026":
        result["strict_accuracy"] = strict
        result["completed_only_accuracy"] = completed_only
    return result


def _report(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    _, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    specification = _BENCHMARKS[dataset_key]
    metric_name = str(specification["primary_metric"])
    direct = _aggregate(rows, "direct", dataset_key)
    graph = _aggregate(rows, "agentgraph", dataset_key)
    strict_key = f"strict_{metric_name}"
    failure_counts = Counter(str(row["failure_type"]) for row in rows)
    below_full = [
        row
        for row in rows
        if float(row["agentgraph"].get(metric_name, 0.0)) < 1.0
    ]
    return {
        "schema_version": "flowsteer.completion_benchmark.round_report.v1",
        "dataset_key": dataset_key,
        "dataset": specification["label"],
        "project_split": str(bounded["split"]),
        "benchmark_slice": bounded.get("benchmark_slice"),
        "sample_count": len(rows),
        "primary_metric": metric_name,
        "metric_scope": (
            "SkillFlow_exact_answer_extraction_and_exact_match"
            if dataset_key == "aime_2026"
            else "OpenAI_simple_evals_HealthBench_rubric_raw_score"
        ),
        "direct_local_baseline": direct,
        "agentgraph": graph,
        "agentgraph_minus_direct": {
            metric_name: float(graph[strict_key]) - float(direct[strict_key])
        },
        "graph_search_diagnostics": aggregate_trajectory_diagnostics(trajectories),
        "failure_types": dict(sorted(failure_counts.items())),
        "below_full_score_demo_count": len(below_full),
        "typical_below_full_score_task_ids": [
            row["task_id"] for row in below_full[:10]
        ],
        "explicit_finished_count": sum(
            row["agentgraph"].get("explicit_finish") is True for row in rows
        ),
        "terminal_failure_count": sum(
            row["agentgraph"].get("available") is True
            and row["agentgraph"].get("explicit_finish") is not True
            for row in rows
        ),
        "operational_failure_count": sum(
            row["direct"].get("available") is not True
            or row["direct"].get("valid") is not True
            or row["agentgraph"].get("available") is not True
            or row["agentgraph"].get("valid") is not True
            for row in rows
        ),
        "policy_version": str(config["director"]["behavior_policy_version"]),
        "policy_adapter": config["director"].get("behavior_adapter_name"),
        "model_catalog_path": str(config["agent_graph"]["model_catalog_path"]),
        "training_performed": False,
        "skill_injection_performed": bool(config.get("skills", {}).get("enabled", False)),
        "skill_evaluation_mode": (
            "memory_on_active_only"
            if bool(config.get("skills", {}).get("enabled", False))
            else "memory_off"
        ),
        "skill_store_path": config.get("skills", {}).get("store_path"),
        "completed_at": _utc_now(),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    metric_name = str(report["primary_metric"])
    strict_key = f"strict_{metric_name}"
    direct = report["direct_local_baseline"]
    graph = report["agentgraph"]
    delta = report["agentgraph_minus_direct"][metric_name]
    failures = "\n".join(
        f"- `{name}`: {count}" for name, count in report["failure_types"].items()
    ) or "- None"
    skill_sentence = (
        "Only evidence-gated ACTIVE Skills were retrieved as rejectable prompt priors."
        if report.get("skill_injection_performed")
        else "No Skill was injected."
    )
    return f"""# {report['dataset']} Architecture Validation

Fixed {report['project_split']} samples: **{report['sample_count']}**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. {skill_sentence}

Primary metric: **{metric_name}** (`{report['metric_scope']}`). AgentGraph explicit FINISH: **{report['explicit_finished_count']}/{report['sample_count']}**; terminal failures: **{report['terminal_failure_count']}**; operational/evaluator failures: **{report['operational_failure_count']}**.

| Condition | Completed | Evaluator valid | Strict {metric_name} |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | {direct['completed']} | {direct['evaluator_valid']} | {100 * float(direct[strict_key]):.2f}% |
| AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * float(graph[strict_key]):.2f}% |

AgentGraph - Direct: **{100 * float(delta):+.2f} percentage points**.

## Failure types

{failures}
"""


def _compatibility_config(
    config: Mapping[str, Any], bounded: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Present the dataset section to the unchanged HotpotQA graph collector."""

    value = dict(config)
    value["hotpotqa_evaluation"] = bounded
    return value


def _attach_healthbench_reference_judge(
    backend: LiveSmokeBackend,
    config: Mapping[str, Any],
    root: Path,
) -> Mapping[str, str]:
    """Attach the existing HealthBench grader without exposing it to Director."""

    evaluation = _mapping(config["evaluation"], "evaluation")
    catalog_path = _resolve(root, str(evaluation["healthbench_judge_catalog_path"]))
    if not catalog_path.is_file():
        raise ConfigurationError(
            f"HealthBench judge catalog does not exist: {catalog_path}"
        )
    judge_registry = load_model_registry(catalog_path)
    judge, provider_model = LiveSmokeBackend._build_healthbench_judge(
        judge_registry,
        os.environ.get("VECTOR_ENGINE_API_KEY", ""),
        str(evaluation["healthbench_judge_model"]),
    )
    backend.judge = judge
    backend.judge_model = provider_model
    return {
        "mode": "openai_simple_evals_compatible_reference",
        "configured_model_id": str(evaluation["healthbench_judge_model"]),
        "provider_model": provider_model,
        "catalog_path": str(catalog_path),
        "official_private_evaluator": "unavailable",
    }


def _completion_stable_zero_check(
    tasks: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    dataset_key: str,
    judge_model: str = "",
) -> Mapping[str, Any]:
    """Extend the reused chain check with dataset evaluator receipts."""

    base = _stable_zero_check(tasks, direct, trajectories)
    checks: list[dict[str, Any]] = []
    for task, base_check in zip(tasks, base["checks"], strict=True):
        direct_value = direct.get(task.task_id)
        graph_value = trajectories.get(task.task_id)
        direct_evaluation = (
            direct_value.get("evaluation", {}) if direct_value else {}
        )
        graph_evaluation = graph_value.get("evaluation", {}) if graph_value else {}
        evaluator_version = evaluator_version_for(task)
        direct_evaluator_valid = bool(
            direct_evaluation.get("valid") is True
            and direct_evaluation.get("evaluator_version") == evaluator_version
        )
        graph_evaluator_valid = bool(
            graph_evaluation.get("valid") is True
            and graph_evaluation.get("evaluator_version") == evaluator_version
        )
        judge_receipts_valid = True
        if dataset_key == "healthbench_professional":
            judge_receipts_valid = all(
                isinstance(value.get("details"), Mapping)
                and value["details"].get("judge_model") == judge_model
                and isinstance(value["details"].get("rubric_grades"), list)
                and bool(value["details"]["rubric_grades"])
                for value in (direct_evaluation, graph_evaluation)
            )
        check = {
            **dict(base_check),
            "direct_evaluator_valid": direct_evaluator_valid,
            "agentgraph_evaluator_valid": graph_evaluator_valid,
            "judge_receipts_valid": judge_receipts_valid,
        }
        check["passed"] = bool(
            check.get("passed")
            and direct_evaluator_valid
            and graph_evaluator_valid
            and judge_receipts_valid
        )
        checks.append(check)
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "criterion": (
            "every_fixed_task_completed_the_full_chain_with_valid_evaluator_receipts"
        ),
        "checks": checks,
    }


async def run_completion_benchmark_round(
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
    validate_completion_benchmark_config(config)
    section_name, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    paths = _paths(config, root)
    selected = _select_tasks(config, root, paths["selected"])
    failures = _read_jsonl(paths["failures"])
    gpu = _mapping(config["gpu"], "gpu")
    director_config = _mapping(config["director"], "director")
    configured_rollout_gpu = int(gpu["rollout_physical"])
    effective_rollout_gpu = int(
        os.environ.get("FLOWSTEER_ROLLOUT_GPU", configured_rollout_gpu)
    )
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.completion_benchmark.round_manifest.v1",
        "status": "prepared" if prepare_only else "runtime_preflight",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "evaluation_section": section_name,
        "dataset_key": dataset_key,
        "git_start": _git_state(root),
        "selected_task_ids": [task.task_id for task in selected],
        "sample_count": len(selected),
        "fixed_split": str(bounded["split"]),
        "training_enabled": False,
        "optimizer_updates": 0,
        "runtime_resource": {
            "configured_rollout_physical": configured_rollout_gpu,
            "effective_rollout_physical": effective_rollout_gpu,
            "resource_adaptation": effective_rollout_gpu != configured_rollout_gpu,
            "supervisor_port": int(
                os.environ.get("FLOWSTEER_SUPERVISOR_PORT", "8015")
            ),
            "context_length": int(
                os.environ.get(
                    "FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH",
                    str(director_config.get("max_context_tokens", 8192)),
                )
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
        judge_receipt = None
        if dataset_key == "healthbench_professional":
            judge_receipt = _attach_healthbench_reference_judge(
                backend,
                config,
                root,
            )
        director = _mapping(config["director"], "director")
        adapter_preflight = await asyncio.to_thread(
            backend.publisher.ensure_loaded_adapter,
            checkpoint_path=str(
                _resolve(root, str(director["behavior_adapter_checkpoint"]))
            ),
            adapter_name=str(director["behavior_adapter_name"]),
        )
        known_prediction = (
            f"<answer>{selected[0].ground_truth}</answer>"
            if dataset_key == "aime_2026"
            else selected[0].ground_truth
        )
        known_answer = await _evaluate_prediction(
            backend, selected[0], known_prediction
        )
        metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
        known_value = known_answer.metrics.get(metric_name)
        known_valid = (
            known_answer.valid
            and isinstance(known_value, (int, float))
            and not isinstance(known_value, bool)
            and math.isfinite(float(known_value))
        )
        if dataset_key == "aime_2026":
            known_valid = known_valid and float(known_value) == 1.0
        if not known_valid:
            raise CompletionBenchmarkRoundError(
                f"known-answer {dataset_key} evaluator preflight failed"
            )
        preflight = {
            **dict(adapter_preflight),
            "evaluator_known_answer": asdict(known_answer),
            "healthbench_judge_model": (
                backend.judge_model
                if dataset_key == "healthbench_professional"
                else None
            ),
            "healthbench_judge_receipt": judge_receipt,
        }
        _write_json(paths["preflight"], preflight)
    except Exception as exc:
        manifest.update(
            status="failed_runtime_preflight",
            error=_safe_error(exc),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        raise CompletionBenchmarkRoundError(
            f"{dataset_key} runtime preflight failed"
        ) from exc

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
        _compatibility_config(config, bounded),
        paths["trajectories"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    rows = _paired_rows(active, direct, trajectories, dataset_key)
    _atomic_jsonl(paths["paired"], rows)
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    wrong = [
        row
        for row in rows
        if float(row["agentgraph"].get(metric_name, 0.0)) < 1.0
    ]
    _atomic_jsonl(paths["wrong"], wrong)
    stable_zero = _completion_stable_zero_check(
        active,
        direct,
        trajectories,
        dataset_key=dataset_key,
        judge_model=backend.judge_model,
    )
    manifest["stable_zero"] = stable_zero
    if canary_only:
        manifest.update(
            status=(
                "stable_zero_confirmed"
                if stable_zero["passed"]
                else "failed_stable_zero"
            ),
            canary_task_count=len(active),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        if not stable_zero["passed"]:
            raise CompletionBenchmarkRoundError(
                f"{dataset_key} canary failed the Stable Zero chain"
            )
        return manifest

    report = _report(rows, config, tuple(trajectories.values()))
    _write_json(paths["report_json"], report)
    paths["report_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_markdown"].write_text(
        _report_markdown(report), encoding="utf-8"
    )
    if report["operational_failure_count"]:
        final_status = "completed_with_operational_failures"
    elif report["terminal_failure_count"]:
        final_status = "completed_with_terminal_failures"
    else:
        final_status = "completed"
    manifest.update(
        status=final_status,
        direct_progress={**dict(manifest.get("direct_progress", {})), "completed": len(direct)},
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
        required=True,
        help="fixed AIME 2026 or HealthBench Professional evaluation YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze selected tasks without starting a model or API",
    )
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="run the first two frozen tasks through the Stable Zero chain",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = asyncio.run(
            run_completion_benchmark_round(
                _resolve(PROJECT_ROOT, args.config),
                project_root=PROJECT_ROOT,
                prepare_only=bool(args.prepare_only),
                canary_only=bool(args.canary_only),
            )
        )
    except (
        CompletionBenchmarkRoundError,
        ConfigurationError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Completion benchmark round failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_key": manifest["dataset_key"],
                "sample_count": manifest["sample_count"],
                "metrics": manifest.get("metrics"),
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
