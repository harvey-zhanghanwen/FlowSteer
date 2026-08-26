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
from dataclasses import asdict, replace
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
    evaluator_version_for,
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
from src.interactive.graph_diagnostics import (
    aggregate_trajectory_diagnostics,
    diagnose_trajectory,
)
from src.interactive.persistence import stable_id
from src.interactive.records import EvaluationReceipt, TaskRecord, TrajectoryRecord
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
    deployment = _mapping(config.get("deployment"), "deployment")
    gpu = _mapping(config.get("gpu"), "gpu")
    skills_enabled = skills.get("enabled") is True
    skill_evaluation_mode = (
        skills.get("enabled") is False
        or (
            skills_enabled
            and isinstance(skills.get("store_path"), str)
            and bool(str(skills.get("store_path")).strip())
            and type(skills.get("retrieval_top_k")) is int
            and int(skills.get("retrieval_top_k")) > 0
            and type(skills.get("current_epoch")) is int
            and int(skills.get("current_epoch")) >= 1
            and deployment.get("exploration_beta") == 0.0
            and deployment.get("allow_forced_probes") is False
            and deployment.get("active_skills_only") is True
            and deployment.get("require_version_compatible_skills") is True
        )
    )
    checks = {
        "experiment.phase": experiment.get("phase") == "hotpotqa_evaluation",
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "dataset_key": bounded.get("dataset_key") == "hotpotqa",
        "split": bounded.get("split") in {"train", "validation", "test"},
        "selection": bounded.get("selection") in {"sequential", "task_ids"},
        "rollouts_per_task": bounded.get("rollouts_per_task") == 1,
        "direct_model_id": bounded.get("direct_model_id") == "qwen3.5-9b-local",
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "grpo.enabled": grpo.get("enabled") is False,
        "optimizer_passes": grpo.get("optimization_passes_per_rollout_batch") == 0,
        "optimizer_updates": grpo.get("max_optimizer_updates") == 0,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.evaluation_mode": skill_evaluation_mode,
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
    split = str(bounded["split"])
    source_fields = {
        "train": "train_path",
        "validation": "validation_path",
        "test": "test_path",
    }
    try:
        source_field = source_fields[split]
    except KeyError as exc:
        raise HotpotRoundError(f"unsupported HotpotQA split: {split}") from exc
    source_path = _resolve(root, str(data[source_field]))
    candidates = tuple(
        task
        for task in iter_task_records(source_path, expected_split=split)
        if _dataset_key(task) == "hotpotqa"
    )
    if len(candidates) < count:
        raise HotpotRoundError(
            f"{split} contains only {len(candidates)} HotpotQA tasks; expected {count}"
        )
    if bounded.get("selection") == "task_ids":
        candidates_by_id = {task.task_id: task for task in candidates}
        requested = [str(task_id) for task_id in bounded["task_ids"]]
        missing = [task_id for task_id in requested if task_id not in candidates_by_id]
        if missing:
            raise HotpotRoundError(
                f"requested HotpotQA task IDs are absent from {split}: "
                + ", ".join(missing)
            )
        expected = tuple(candidates_by_id[task_id] for task_id in requested)
    else:
        expected = candidates[:count]
    if selected_path.exists():
        frozen = tuple(iter_task_records(selected_path, expected_split=split))
        if len(frozen) != count:
            raise HotpotRoundError("frozen HotpotQA selection has the wrong size")
        for expected_task, frozen_task in zip(expected, frozen, strict=True):
            if (
                expected_task.task_id != frozen_task.task_id
                or expected_task.question != frozen_task.question
                or expected_task.ground_truth != frozen_task.ground_truth
            ):
                raise HotpotRoundError(
                    f"frozen HotpotQA task batch differs from the aligned {split} data"
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
        and evaluation.get("evaluator_version") == evaluator_version_for(task)
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
    evaluation = value.get("evaluation")
    return bool(
        _trajectory_identity_matches(
            value,
            task=task,
            condition_id=condition_id,
            versions=versions,
        )
        and isinstance(evaluation, Mapping)
        and evaluation.get("valid") is True
        and evaluation.get("evaluator_version") == versions.get("evaluator")
    )


def _reportable_terminal_failure_matches(
    value: Mapping[str, Any],
    *,
    task: TaskRecord,
    condition_id: str,
    versions: Mapping[str, str],
) -> bool:
    """Admit one frozen AIME terminal failure to reporting, not evaluation.

    A trajectory that naturally exhausts the Director budget without a legal
    ``FINISH`` is a completed execution receipt under the project terminal
    semantics.  It must remain evaluator-invalid and must not enter terminal
    evaluator retry, but dropping it would erase the real failure from paired
    analysis and cause the frozen workflow to be sampled again.
    """

    evaluation = value.get("evaluation")
    details = evaluation.get("details") if isinstance(evaluation, Mapping) else None
    return bool(
        _dataset_key(task) == "aime_2026"
        and _trajectory_identity_matches(
            value,
            task=task,
            condition_id=condition_id,
            versions=versions,
        )
        and value.get("explicit_finish") is False
        and value.get("termination_reason")
        in {"max_rounds", "canvas_action_domain_exhausted"}
        and value.get("final_answer") in (None, "")
        and isinstance(evaluation, Mapping)
        and evaluation.get("valid") is False
        and evaluation.get("reward") is None
        and evaluation.get("reason")
        == "not_evaluated_without_explicit_finish"
        and evaluation.get("evaluator_version") == versions.get("evaluator")
        and isinstance(details, Mapping)
        and details.get("formal_evaluator_called") is False
    )


def _trajectory_identity_matches(
    value: Mapping[str, Any],
    *,
    task: TaskRecord,
    condition_id: str,
    versions: Mapping[str, str],
) -> bool:
    """Match the frozen rollout independently of evaluator admission."""

    embedded = value.get("task")
    return bool(
        isinstance(value.get("trajectory_id"), str)
        and isinstance(embedded, Mapping)
        and embedded.get("task_id") == task.task_id
        and value.get("condition_id") == condition_id
        and value.get("versions") == versions
        and isinstance(value.get("turns"), list)
        and isinstance(value.get("evaluation"), Mapping)
        and isinstance(value.get("explicit_finish"), bool)
    )


def _environment_replay_trace(
    value: Mapping[str, Any],
) -> Optional[tuple[dict[str, Any], ...]]:
    """Return only a structurally ordered evaluator trace prefix."""

    evaluation = value.get("evaluation")
    details = evaluation.get("details") if isinstance(evaluation, Mapping) else None
    raw_trace = details.get("trace") if isinstance(details, Mapping) else None
    if raw_trace is None:
        return ()
    if not isinstance(raw_trace, list):
        return None
    trace: list[dict[str, Any]] = []
    for step_index, entry in enumerate(raw_trace):
        if not isinstance(entry, Mapping) or entry.get("step") != step_index:
            return None
        trace.append(dict(entry))
    return tuple(trace)


def _environment_replay_trace_length(value: Mapping[str, Any]) -> int:
    trace = _environment_replay_trace(value)
    return -1 if trace is None else len(trace)


def _evaluation_retry_attempt(value: Mapping[str, Any]) -> int:
    receipt = value.get("evaluation_retry_receipt")
    attempt = receipt.get("attempt") if isinstance(receipt, Mapping) else 0
    return attempt if type(attempt) is int and attempt >= 1 else 0


def _final_graph(value: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    final_turn = turns[-1]
    graph = final_turn.get("graph_snapshot") if isinstance(final_turn, Mapping) else None
    return dict(graph) if isinstance(graph, Mapping) else None


async def _retry_terminal_evaluator(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    source: Mapping[str, Any],
    *,
    versions: Mapping[str, str],
    attempt: int,
) -> Mapping[str, Any]:
    """Append one evaluator-only attempt for a frozen AgentGraph rollout."""

    source_record = TrajectoryRecord.from_dict(source)
    source_trajectory_id = source_record.trajectory_id
    expected_evaluator = str(versions["evaluator"])
    source_evaluation = source.get("evaluation")
    source_details = (
        source_evaluation.get("details")
        if isinstance(source_evaluation, Mapping)
        else None
    )
    replay_trace_missing = not isinstance(source_details, Mapping) or (
        "trace" not in source_details or source_details.get("trace") is None
    )
    replay_trace = _environment_replay_trace(source)
    final_graph = _final_graph(source)
    repository_patch: Optional[str] = None
    if _dataset_key(task) == "swe_bench":
        terminal_artifact = (
            source_details.get("terminal_artifact")
            if isinstance(source_details, Mapping)
            else None
        )
        candidate_patch = (
            terminal_artifact.get("repository_patch")
            if isinstance(terminal_artifact, Mapping)
            else None
        )
        if isinstance(candidate_patch, str) and candidate_patch.strip():
            repository_patch = candidate_patch
    started_at = _utc_now()
    if _dataset_key(task) in {"webshop", "alfworld"} and replay_trace_missing:
        evaluation = EvaluationReceipt(
            evaluator_version=expected_evaluator,
            valid=False,
            reward=None,
            reason="environment_replay_trace_unavailable",
            details={
                "error": "interactive evaluator attempt has no persisted trace",
                "trace": None,
            },
        )
    elif replay_trace is None:
        evaluation = EvaluationReceipt(
            evaluator_version=expected_evaluator,
            valid=False,
            reward=None,
            reason="environment_replay_trace_invalid",
            details={
                "error": "persisted evaluator trace is structurally invalid",
                "trace": (
                    source_details.get("trace")
                    if isinstance(source_details, Mapping)
                    else None
                ),
            },
        )
    elif _dataset_key(task) == "swe_bench" and repository_patch is None:
        evaluation = EvaluationReceipt(
            evaluator_version=expected_evaluator,
            valid=False,
            reward=None,
            reason="terminal_repository_patch_unavailable",
            details={
                "error": (
                    "frozen SWE-bench rollout has no non-empty authoritative "
                    "workspace diff"
                ),
            },
        )
    elif final_graph is None and source_record.final_answer is not None:
        evaluation = EvaluationReceipt(
            evaluator_version=expected_evaluator,
            valid=False,
            reward=None,
            reason="terminal_evaluator_retry_unavailable",
            details={"error": "frozen final graph snapshot is unavailable"},
        )
    else:
        try:
            evaluator_kwargs: dict[str, Any] = {}
            if _dataset_key(task) == "swe_bench":
                evaluator_kwargs["repository_patch"] = repository_patch
            outcome = await backend.evaluate_final_graph(
                task,
                source_record.final_answer,
                final_graph or {},
                rollout_index=0,
                environment_replay_trace=replay_trace,
                **evaluator_kwargs,
            )
            evaluation = EvaluationReceipt.from_dict(asdict(outcome))
        except Exception as exc:
            evaluation = EvaluationReceipt(
                evaluator_version=expected_evaluator,
                valid=False,
                reward=None,
                reason="terminal_evaluator_retry_failed",
                details={
                    "error": _safe_error(exc),
                    "trace": list(replay_trace),
                },
            )
    completed_at = _utc_now()
    retry_trajectory_id = stable_id(
        "trajectory",
        {
            "source_trajectory_id": source_trajectory_id,
            "attempt": attempt,
            "evaluator_version": expected_evaluator,
            "mode": "terminal_evaluator_only",
        },
    )
    retry_record = replace(
        source_record,
        trajectory_id=retry_trajectory_id,
        evaluation=evaluation,
        created_at=completed_at,
    )
    payload = retry_record.to_dict()
    final_turn = source.get("turns", ())[-1] if source.get("turns") else None
    payload["evaluation_retry_receipt"] = {
        "schema_version": "flowsteer.agentgraph.evaluation_retry.v1",
        "source_trajectory_id": source_trajectory_id,
        "attempt": attempt,
        "mode": "terminal_evaluator_only",
        "evaluator_version": expected_evaluator,
        "reused_director_canvas": True,
        "reused_turn_count": len(source_record.turns),
        "final_graph_snapshot_id": (
            final_turn.get("graph_snapshot_id")
            if isinstance(final_turn, Mapping)
            else None
        ),
        "environment_replay_steps": (
            None
            if replay_trace is None or replay_trace_missing
            else len(replay_trace)
        ),
        "repository_patch_reused": repository_patch is not None,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    backend.evidence_store.append_trajectory(payload)
    return payload


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
    # The paired Direct baseline has its own frozen sampling seed.  A new
    # Director/Canvas condition may change the policy sampling seed without
    # repaying or silently replacing the fixed Local Direct comparator.
    seed = int(bounded.get("direct_generation_seed", experiment["seed"]))
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
            reused_candidates = []
            for value in _read_jsonl(reuse_path):
                copied = dict(value)
                copied["reuse_receipt"] = {
                    "reused": True,
                    "source": str(reuse_path),
                }
                reused_candidates.append(copied)
            # ``_by_task`` intentionally keeps the first complete record so a
            # resume cannot silently replace a successful request.  A declared
            # fixed comparator is more authoritative than stale records from a
            # prior canary with another Direct seed, so place it first.
            direct_candidates = reused_candidates + direct_candidates
    selected_by_id = {task.task_id: task for task in selected}
    rescored_candidates: list[dict[str, Any]] = []
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
                or evaluation.get("evaluator_version")
                != evaluator_version_for(task)
            )
        ):
            updated = dict(candidate)
            updated["evaluation"] = asdict(
                await evaluate_task(task, str(candidate["final_answer"]))
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
            rescored_candidates.append(updated)
        else:
            rescored_candidates.append(dict(candidate))
    direct_candidates = rescored_candidates
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
    failure_path: Optional[Path] = None,
) -> dict[str, dict[str, Any]]:
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    experiment = _mapping(config["experiment"], "experiment")
    director = _mapping(config["director"], "director")
    exploration = _mapping(config.get("exploration", {}), "exploration")
    skills = _mapping(config.get("skills", {}), "skills")
    policy = str(director["behavior_policy_version"])
    condition_id = str(experiment["condition_id"])
    versions = {
        task.task_id: version_bundle_for(
            task,
            policy_version=policy,
            model_catalog_version=backend.model_catalog_version,
            prompt_version=str(experiment["prompt_version"]),
            tool_version=str(experiment["tool_version"]),
            encoder_version=str(exploration.get("encoder_version", "none")),
            feature_schema_version=str(
                exploration.get("feature_schema_version", "none")
            ),
            posterior_version=str(exploration.get("posterior_version", "none")),
            skill_library_version=str(skills.get("library_version", "none")),
        )
        for task in selected
    }
    selected_by_id = {task.task_id: task for task in selected}
    # The collector commits to EvidenceStore before returning to this runner.
    # Read both the ordered checkpoint and the authoritative append-only stream
    # before scheduling any work.  An exact invalid evaluator attempt reserves
    # its frozen rollout just like a valid one: only its terminal evaluator may
    # be retried, never the Director/Canvas/Agent construction.
    candidates = [*_read_jsonl(path), *backend.evidence_store.trajectories.payloads()]
    valid_candidates: dict[str, tuple[int, dict[str, Any]]] = {}
    terminal_failure_candidates: dict[str, tuple[int, dict[str, Any]]] = {}
    invalid_candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for order, value in enumerate(candidates):
        embedded = value.get("task")
        task_id = embedded.get("task_id") if isinstance(embedded, Mapping) else None
        selected_task = selected_by_id.get(task_id)
        if selected_task is None or not _trajectory_identity_matches(
            value,
            task=selected_task,
            condition_id=condition_id,
            versions=versions[task_id].to_dict(),
        ):
            continue
        if _trajectory_resume_matches(
            value,
            task=selected_task,
            condition_id=condition_id,
            versions=versions[task_id].to_dict(),
        ):
            valid_candidates[task_id] = (order, dict(value))
        elif _reportable_terminal_failure_matches(
            value,
            task=selected_task,
            condition_id=condition_id,
            versions=versions[task_id].to_dict(),
        ):
            terminal_failure_candidates[task_id] = (order, dict(value))
        else:
            invalid_candidates.setdefault(task_id, []).append((order, dict(value)))

    by_task = {
        task_id: value for task_id, (_, value) in valid_candidates.items()
    }
    for task_id, (_, value) in terminal_failure_candidates.items():
        by_task.setdefault(task_id, value)
    retry_sources: dict[str, tuple[dict[str, Any], int]] = {}
    for task_id, values in invalid_candidates.items():
        if task_id in by_task:
            continue
        # Reuse the longest structurally valid environment prefix; if several
        # attempts reached the same step, the latest append-only event wins.
        _, source = max(
            values,
            key=lambda item: (_environment_replay_trace_length(item[1]), item[0]),
        )
        next_attempt = 1 + max(
            _evaluation_retry_attempt(value) for _, value in values
        )
        retry_sources[task_id] = (source, next_attempt)
    pending_retry_task_ids = set(retry_sources)

    # Remove stale invalid rows from the ordered mirror.  They remain in the
    # append-only EvidenceStore, while this checkpoint contains at most one
    # admitted trajectory per task in the frozen selection order.
    _persist_ordered(path, selected, by_task)
    manifest["agentgraph_progress"] = {
        "completed": len(by_task),
        "reportable_terminal_failures": sum(
            _reportable_terminal_failure_matches(
                value,
                task=selected_by_id[task_id],
                condition_id=condition_id,
                versions=versions[task_id].to_dict(),
            )
            for task_id, value in by_task.items()
        ),
        "pending_evaluator_retries": len(pending_retry_task_ids),
        "failed_attempts": sum(
            item.get("condition") == "agentgraph" for item in failures
        ),
    }
    _write_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(int(bounded["concurrency"]))
    task_timeout_raw = bounded.get("task_timeout_seconds")
    task_timeout_seconds = (
        None if task_timeout_raw is None else float(task_timeout_raw)
    )

    async def run(task: TaskRecord) -> tuple[TaskRecord, str, Any]:
        async with semaphore:
            mode = "terminal_evaluator_retry" if task.task_id in retry_sources else "collect"
            try:
                if mode == "terminal_evaluator_retry":
                    source, attempt = retry_sources[task.task_id]
                    invocation = _retry_terminal_evaluator(
                        backend,
                        task,
                        source,
                        versions=versions[task.task_id].to_dict(),
                        attempt=attempt,
                    )
                else:
                    invocation = backend.collect(
                        task,
                        0,
                        versions[task.task_id],
                        expected_task_split=str(bounded.get("split", "validation")),
                    )
                result = (
                    await invocation
                    if task_timeout_seconds is None
                    else await asyncio.wait_for(
                        invocation,
                        timeout=task_timeout_seconds,
                    )
                )
                return task, mode, result
            except BaseException as exc:
                return task, mode, exc

    jobs = [
        asyncio.create_task(run(task))
        for task in selected
        if task.task_id not in by_task
    ]
    for completed in asyncio.as_completed(jobs):
        task, mode, result = await completed
        if isinstance(result, BaseException):
            failures.append(
                {
                    "task_id": task.task_id,
                    "condition": "agentgraph",
                    "stage": mode,
                    "error": _safe_error(result),
                    "recorded_at": _utc_now(),
                }
            )
        else:
            if isinstance(result, TrajectoryRecord):
                payload = result.to_dict()
            elif isinstance(result, Mapping):
                payload = dict(result)
            else:
                raise HotpotRoundError("backend returned a non-trajectory result")
            if _trajectory_resume_matches(
                payload,
                task=task,
                condition_id=condition_id,
                versions=versions[task.task_id].to_dict(),
            ) or _reportable_terminal_failure_matches(
                payload,
                task=task,
                condition_id=condition_id,
                versions=versions[task.task_id].to_dict(),
            ):
                by_task[task.task_id] = payload
                pending_retry_task_ids.discard(task.task_id)
                _persist_ordered(path, selected, by_task)
            else:
                pending_retry_task_ids.add(task.task_id)
                evaluation = payload.get("evaluation")
                reason = (
                    evaluation.get("reason", "invalid_evaluator_receipt")
                    if isinstance(evaluation, Mapping)
                    else "missing_evaluator_receipt"
                )
                failures.append(
                    {
                        "task_id": task.task_id,
                        "condition": "agentgraph",
                        "stage": (
                            "terminal_evaluator_retry"
                            if mode == "terminal_evaluator_retry"
                            else "terminal_evaluator"
                        ),
                        "error": str(reason),
                        "trajectory_id": payload.get("trajectory_id"),
                        "recorded_at": _utc_now(),
                    }
                )
        manifest["agentgraph_progress"] = {
            "completed": len(by_task),
            "reportable_terminal_failures": sum(
                _reportable_terminal_failure_matches(
                    value,
                    task=selected_by_id[task_id],
                    condition_id=condition_id,
                    versions=versions[task_id].to_dict(),
                )
                for task_id, value in by_task.items()
            ),
            "pending_evaluator_retries": sum(
                task.task_id not in by_task
                and task.task_id in pending_retry_task_ids
                for task in selected
            ),
            "failed_attempts": sum(
                item.get("condition") == "agentgraph" for item in failures
            ),
        }
        _write_json(manifest_path, manifest)
        if failure_path is not None:
            _atomic_jsonl(failure_path, failures)
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
    # A transient execution error can be followed by a valid graph edit and a
    # correct terminal answer.  Keep that receipt in the trajectory, but do
    # not classify a recovered, correct task as an evaluation failure.  Keep
    # paired gains distinct so the Direct-vs-AgentGraph diagnostic is not
    # silently collapsed into the generic correct bucket.
    if graph_em == 1.0:
        if direct is not None and direct_em == 0.0:
            return "architecture_gain"
        return "correct"
    if "execution_error=" in feedback:
        return "executor_or_provider_failure"
    if direct is None:
        return "direct_operational_failure_comparison_unavailable"
    if direct_em == 1.0 and graph_em == 0.0:
        return "architecture_regression_candidate"
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
                    "graph_diagnostic": (
                        diagnose_trajectory(graph_value).to_dict()
                        if graph_value is not None
                        else None
                    ),
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


def _report(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
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
        "metric_scope": "official_compatible_answer_only",
        "supporting_fact_metrics_available": False,
        "dataset": "HotpotQA",
        "project_split": str(
            config.get("hotpotqa_evaluation", {}).get("split", "validation")
        ),
        "native_source_split": "train",
        "input_context": "full_10_passages",
        "sample_count": len(rows),
        "evaluation_name": config["experiment"]["name"],
        "direct_local_baseline": direct,
        "agentgraph": graph,
        "graph_search_diagnostics": aggregate_trajectory_diagnostics(trajectories),
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
        "skill_injection_performed": bool(config.get("skills", {}).get("enabled", False)),
        "skill_evaluation_mode": (
            "memory_on_active_only"
            if bool(config.get("skills", {}).get("enabled", False))
            else "memory_off"
        ),
        "skill_store_path": config.get("skills", {}).get("store_path"),
        "known_limitations": [
            "Only answer EM/F1 is available; supporting-fact and joint metrics are not emitted.",
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
    split = str(report.get("project_split", "validation"))
    sample_role = (
        "frozen architecture-development samples"
        if split == "train"
        else "fixed project validation samples"
    )
    skill_sentence = (
        "Only evidence-gated ACTIVE Skills were retrieved as rejectable prompt priors; "
        "no forced intervention or Skill update ran."
        if report.get("skill_injection_performed")
        else "No Skill was injected."
    )
    return f"""# HotpotQA Architecture Validation — {report['evaluation_name']}

Evaluation split: **{split}**; {sample_role}: **{report['sample_count']}**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian update, or Skill publication ran. {skill_sentence}

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

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
    *,
    require_non_empty_final_answer: bool = True,
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
        final_answer = trajectory.get("final_answer") if trajectory else None
        terminal_artifact_saved = bool(
            final_answer is not None
            and (
                not require_non_empty_final_answer
                or final_answer != ""
            )
        )
        passed = bool(
            direct_value
            and trajectory
            and trajectory.get("explicit_finish") is True
            and terminal_artifact_saved
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
                "terminal_artifact_saved": terminal_artifact_saved,
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
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
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
        "schema_version": "flowsteer.hotpotqa.round_manifest.v1",
        "status": "prepared" if prepare_only else "runtime_preflight",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "git_start": _git_state(root),
        "selected_task_ids": [task.task_id for task in selected],
        "sample_count": len(selected),
        "fixed_split": str(bounded["split"]),
        "input_context": "full_10_passages",
        "training_enabled": False,
        "optimizer_updates": 0,
        "runtime_resource": {
            "configured_rollout_physical": configured_rollout_gpu,
            "effective_rollout_physical": effective_rollout_gpu,
            "resource_adaptation": effective_rollout_gpu != configured_rollout_gpu,
            "supervisor_port": int(os.environ.get("FLOWSTEER_SUPERVISOR_PORT", "8015")),
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
        failure_path=paths["failures"],
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

    report = _report(rows, config, tuple(trajectories.values()))
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
