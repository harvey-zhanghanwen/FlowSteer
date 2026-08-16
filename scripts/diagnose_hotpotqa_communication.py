#!/usr/bin/env python3
"""Replay frozen multi-Agent HotpotQA graphs with normal vs masked upstream.

This is an execution-only diagnostic.  It skips the Director, Direct baseline,
collector, trainer, publisher, MACE, Bayesian, and Skill paths.  Both arms use
the same frozen graph, task, model routes, sampling seed, and evaluator.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_hotpotqa_round import (
    _atomic_jsonl,
    _git_state,
    _read_jsonl,
    validate_hotpot_config,
)
from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _graph_from_mapping,
    _safe_error,
    _write_json,
)
from src.interactive.agent_runtime import CommunicationCondition
from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.openai_gateway import MASKED_UPSTREAM_CONTENT
from src.interactive.persistence.ids import stable_id
from src.interactive.records import (
    CommunicationDiagnosticRecord,
    EvaluationReceipt,
    TaskRecord,
)
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.task_evaluator import _normalize_answer, evaluate_task
from src.interactive.versioning import VersionBundle


class CommunicationDiagnosticError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _final_graph(trajectory: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    turns = trajectory.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    graph = turns[-1].get("graph_snapshot") if isinstance(turns[-1], Mapping) else None
    return graph if isinstance(graph, Mapping) else None


def _is_multi_agent_graph(graph: Optional[Mapping[str, Any]]) -> bool:
    if graph is None:
        return False
    nodes = graph.get("nodes")
    relations = graph.get("relations")
    output_id = graph.get("output_agent_id")
    if not isinstance(nodes, list) or len(nodes) < 2:
        return False
    if not isinstance(relations, list) or not isinstance(output_id, str):
        return False
    return any(
        isinstance(relation, Mapping)
        and (
            (
                relation.get("target_id") == output_id
                and relation.get("source_to_target") is True
            )
            or (
                relation.get("source_id") == output_id
                and relation.get("target_to_source") is True
            )
        )
        for relation in relations
    )


def select_diagnostic_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    max_pairs: int,
) -> tuple[Mapping[str, Any], ...]:
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    selected = [
        item
        for item in trajectories
        if item.get("explicit_finish") is True
        and item.get("final_answer") not in (None, "")
        and _is_multi_agent_graph(_final_graph(item))
    ]
    return tuple(selected[:max_pairs])


def _validate_isolated_paths(
    output_paths: Sequence[Path],
    protected_paths: Sequence[Path],
) -> None:
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise CommunicationDiagnosticError("diagnostic output paths must be distinct")
    protected = {path.resolve() for path in protected_paths}
    collision = [path for path in resolved_outputs if path in protected]
    if collision:
        raise CommunicationDiagnosticError(
            "diagnostic output path overlaps a benchmark or trajectory artifact"
        )


def _task_from_trajectory(value: Mapping[str, Any]) -> TaskRecord:
    task = _mapping(value.get("task"), "trajectory.task")
    return TaskRecord(
        task_id=str(task["task_id"]),
        question=str(task["question"]),
        ground_truth=task.get("ground_truth"),
        split=str(task["split"]),
        metadata=dict(_mapping(task.get("metadata", {}), "trajectory.task.metadata")),
    )


def _evaluation_receipt(outcome: Any) -> EvaluationReceipt:
    return EvaluationReceipt(
        evaluator_version=str(outcome.evaluator_version),
        valid=bool(outcome.valid),
        reward=None if outcome.reward is None else float(outcome.reward),
        metrics={str(key): float(value) for key, value in outcome.metrics.items()},
        reason=str(outcome.reason),
        details=dict(outcome.details),
    )


def _scored_answer(record: Mapping[str, Any]) -> str:
    evaluation = record.get("evaluation")
    details = evaluation.get("details") if isinstance(evaluation, Mapping) else None
    scored = details.get("scored_prediction") if isinstance(details, Mapping) else None
    return str(scored if scored is not None else record.get("final_answer", ""))


def paired_result(
    *,
    pair_id: str,
    source_trajectory_id: str,
    task_id: str,
    normal: Mapping[str, Any],
    masked: Mapping[str, Any],
) -> Mapping[str, Any]:
    normal_eval = _mapping(normal.get("evaluation"), "normal.evaluation")
    masked_eval = _mapping(masked.get("evaluation"), "masked.evaluation")
    normal_metrics = _mapping(normal_eval.get("metrics"), "normal.metrics")
    masked_metrics = _mapping(masked_eval.get("metrics"), "masked.metrics")
    normal_answer = _scored_answer(normal)
    masked_answer = _scored_answer(masked)
    normal_em = float(normal_metrics.get("exact_match", 0.0))
    masked_em = float(masked_metrics.get("exact_match", 0.0))
    normal_f1 = float(normal_metrics.get("token_f1", 0.0))
    masked_f1 = float(masked_metrics.get("token_f1", 0.0))
    return {
        "schema_version": "flowsteer.agentgraph.communication_pair.v1",
        "pair_id": pair_id,
        "source_trajectory_id": source_trajectory_id,
        "task_id": task_id,
        "normal_diagnostic_id": normal["diagnostic_id"],
        "masked_diagnostic_id": masked["diagnostic_id"],
        "raw_answer_changed": normal.get("final_answer") != masked.get("final_answer"),
        "normalized_answer_changed": (
            _normalize_answer(normal_answer) != _normalize_answer(masked_answer)
        ),
        "normal": {
            "scored_answer": normal_answer,
            "exact_match": normal_em,
            "token_f1": normal_f1,
        },
        "upstream_masked": {
            "scored_answer": masked_answer,
            "exact_match": masked_em,
            "token_f1": masked_f1,
        },
        "delta_masked_minus_normal": {
            "exact_match": masked_em - normal_em,
            "token_f1": masked_f1 - normal_f1,
        },
        "normal_correct_to_masked_wrong": normal_em == 1.0 and masked_em == 0.0,
        "diagnostic_only": True,
        "grpo_eligible": False,
        "created_at": _utc_now(),
    }


def _summary(
    selected_count: int,
    arms: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    normal = [item for item in arms if item.get("communication_condition") == "normal"]
    masked = [
        item
        for item in arms
        if item.get("communication_condition") == "upstream_masked"
    ]

    def average(records: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
        values = []
        for record in records:
            evaluation = record.get("evaluation")
            metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
            if isinstance(metrics, Mapping) and evaluation.get("valid") is True:
                values.append(float(metrics.get(metric, 0.0)))
        return sum(values) / len(values) if values else None

    normal_em = average(normal, "exact_match")
    masked_em = average(masked, "exact_match")
    normal_f1 = average(normal, "token_f1")
    masked_f1 = average(masked, "token_f1")
    return {
        "schema_version": "flowsteer.agentgraph.communication_summary.v1",
        "diagnostic_only": True,
        "grpo_eligible": False,
        "selected_frozen_multi_agent_graphs": selected_count,
        "completed_normal_arms": len(normal),
        "completed_upstream_masked_arms": len(masked),
        "completed_pairs": len(pairs),
        "failure_count": len(failures),
        "normal": {"exact_match": normal_em, "token_f1": normal_f1},
        "upstream_masked": {"exact_match": masked_em, "token_f1": masked_f1},
        "delta_masked_minus_normal": {
            "exact_match": (
                None if normal_em is None or masked_em is None else masked_em - normal_em
            ),
            "token_f1": (
                None if normal_f1 is None or masked_f1 is None else masked_f1 - normal_f1
            ),
        },
        "raw_answer_changed_pairs": sum(
            item.get("raw_answer_changed") is True for item in pairs
        ),
        "normalized_answer_changed_pairs": sum(
            item.get("normalized_answer_changed") is True for item in pairs
        ),
        "normal_correct_to_masked_wrong": sum(
            item.get("normal_correct_to_masked_wrong") is True for item in pairs
        ),
        "interpretation": (
            "A masked-score drop demonstrates behavioral dependence on upstream content. "
            "No change does not prove non-use because Question and Context remain intact."
        ),
        "completed_at": _utc_now(),
    }


async def run_diagnostic(
    config_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Mapping[str, Any]:
    diagnostic = load_yaml(config_path)
    base_path = _resolve(project_root, str(diagnostic["base_evaluation_config"]))
    base = load_yaml(base_path)
    validate_hotpot_config(base)
    bounds = _mapping(diagnostic.get("diagnostic"), "diagnostic")
    storage = _mapping(diagnostic.get("storage"), "storage")
    if bounds.get("diagnostic_only") is not True or bounds.get("grpo_eligible") is not False:
        raise ConfigurationError("communication diagnostic must be diagnostic-only and GRPO-ineligible")
    if tuple(bounds.get("conditions", ())) != ("normal", "upstream_masked"):
        raise ConfigurationError("conditions must be [normal, upstream_masked]")
    if bounds.get("selection") != "first_frozen_multi_agent_graphs":
        raise ConfigurationError("unsupported communication diagnostic selection")
    max_pairs = int(bounds["max_pairs"])

    source_path = _resolve(project_root, str(diagnostic["source_trajectories_path"]))
    arm_path = _resolve(project_root, str(storage["arms_path"]))
    pair_path = _resolve(project_root, str(storage["pairs_path"]))
    failure_path = _resolve(project_root, str(storage["failures_path"]))
    summary_path = _resolve(project_root, str(storage["summary_path"]))
    manifest_path = _resolve(project_root, str(storage["manifest_path"]))
    base_storage = _mapping(base["storage"], "base.storage")
    protected = [
        source_path,
        *(
            _resolve(project_root, str(value))
            for key, value in base_storage.items()
            if key.endswith("_path")
        ),
    ]
    _validate_isolated_paths(
        [arm_path, pair_path, failure_path, summary_path, manifest_path],
        protected,
    )
    trajectories = _read_jsonl(source_path)
    selected = select_diagnostic_trajectories(trajectories, max_pairs)
    if not selected:
        raise CommunicationDiagnosticError(
            "source trajectories contain no completed multi-Agent output graph"
        )

    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.agentgraph.communication_manifest.v1",
        "status": "runtime",
        "started_at": _utc_now(),
        "git_start": _git_state(project_root),
        "source_trajectories_path": str(source_path),
        "selected_source_trajectory_ids": [
            item.get("trajectory_id") for item in selected
        ],
        "conditions": ["normal", "upstream_masked"],
        "diagnostic_only": True,
        "grpo_eligible": False,
        "director_calls": 0,
        "direct_baseline_calls": 0,
        "training_enabled": False,
        "optimizer_updates": 0,
        "runtime_resource": {
            "configured_rollout_physical": int(base["gpu"]["rollout_physical"]),
            "effective_rollout_physical": int(
                os.environ.get(
                    "FLOWSTEER_ROLLOUT_GPU",
                    str(base["gpu"]["rollout_physical"]),
                )
            ),
            "supervisor_port": int(
                os.environ.get("FLOWSTEER_SUPERVISOR_PORT", "8015")
            ),
        },
    }
    _write_json(manifest_path, manifest)
    backend = LiveSmokeBackend.from_config(base, project_root, evaluation_only=True)
    existing_arms = {
        str(item["diagnostic_id"]): item
        for item in _read_jsonl(arm_path)
        if isinstance(item.get("diagnostic_id"), str)
    }
    arm_order: list[str] = []
    pairs: list[Mapping[str, Any]] = []
    failures = _read_jsonl(failure_path)

    for index, source in enumerate(selected):
        graph_value = _final_graph(source)
        if graph_value is None:
            continue
        graph = _graph_from_mapping(graph_value)
        graph.validate(backend.registry, require_complete=True).raise_if_invalid()
        task = _task_from_trajectory(source)
        versions = VersionBundle.from_dict(
            dict(_mapping(source.get("versions"), "trajectory.versions"))
        )
        source_id = str(source["trajectory_id"])
        pair_id = stable_id(
            "communication_pair",
            {
                "source_trajectory_id": source_id,
                "graph": graph_value,
                "versions": versions.to_dict(),
            },
        )
        pair_arms: dict[str, Mapping[str, Any]] = {}
        for condition in (
            CommunicationCondition.NORMAL,
            CommunicationCondition.UPSTREAM_MASKED,
        ):
            diagnostic_id = stable_id(
                "communication_diagnostic",
                {"pair_id": pair_id, "condition": condition.value},
            )
            arm_order.append(diagnostic_id)
            cached = existing_arms.get(diagnostic_id)
            if (
                cached is not None
                and cached.get("source_trajectory_id") == source_id
                and cached.get("communication_condition") == condition.value
                and cached.get("diagnostic_only") is True
                and cached.get("grpo_eligible") is False
            ):
                pair_arms[condition.value] = cached
                continue
            run_id = f"communication:{index:04d}:{condition.value}:{task.task_id}"
            try:
                runtime = await backend.runtime.execute(
                    graph,
                    task.question,
                    run_id=run_id,
                    communication_condition=condition,
                )
                executions = tuple(
                    execution_record_from_call(call) for call in runtime.calls
                )
                masked_ids = [
                    call.request.request_id
                    for call in runtime.calls
                    if condition is CommunicationCondition.UPSTREAM_MASKED
                    and (call.request.upstream or call.request.peer_draft is not None)
                ]
                if condition is CommunicationCondition.UPSTREAM_MASKED and not masked_ids:
                    raise CommunicationDiagnosticError(
                        "masked multi-Agent arm had no inter-Agent content to mask"
                    )
                if condition is CommunicationCondition.UPSTREAM_MASKED:
                    for execution, call in zip(executions, runtime.calls, strict=True):
                        if call.request.request_id not in masked_ids:
                            continue
                        rendered = execution.metadata["request"]["rendered_messages"]
                        if MASKED_UPSTREAM_CONTENT not in json.dumps(rendered):
                            raise CommunicationDiagnosticError(
                                "masked call receipt does not contain the visible sentinel"
                            )
                outcome = await evaluate_task(task, runtime.final_answer)
                record = CommunicationDiagnosticRecord(
                    diagnostic_id=diagnostic_id,
                    pair_id=pair_id,
                    source_trajectory_id=source_id,
                    task=task,
                    condition_id=condition.value,
                    communication_condition=condition.value,
                    versions=versions,
                    graph_snapshot=graph_value,
                    output_agent_id=runtime.output_agent_id,
                    runtime_run_id=runtime.run_id,
                    executions=executions,
                    final_answer=runtime.final_answer,
                    evaluation=_evaluation_receipt(outcome),
                    mask_applied_call_ids=masked_ids,
                ).to_dict()
                existing_arms[diagnostic_id] = record
                pair_arms[condition.value] = record
                _atomic_jsonl(
                    arm_path,
                    [existing_arms[item] for item in arm_order if item in existing_arms],
                )
            except BaseException as exc:
                failures.append(
                    {
                        "pair_id": pair_id,
                        "task_id": task.task_id,
                        "condition": condition.value,
                        "error": _safe_error(exc),
                        "recorded_at": _utc_now(),
                    }
                )
                _atomic_jsonl(failure_path, failures)

        normal = pair_arms.get("normal")
        masked = pair_arms.get("upstream_masked")
        if normal is not None and masked is not None:
            pairs.append(
                paired_result(
                    pair_id=pair_id,
                    source_trajectory_id=source_id,
                    task_id=task.task_id,
                    normal=normal,
                    masked=masked,
                )
            )
            _atomic_jsonl(pair_path, pairs)

    arms = [existing_arms[item] for item in arm_order if item in existing_arms]
    summary = _summary(len(selected), arms, pairs, failures)
    _write_json(summary_path, summary)
    manifest.update(
        status="completed" if len(pairs) == len(selected) else "incomplete",
        summary=summary,
        git_end=_git_state(project_root),
        completed_at=_utc_now(),
    )
    _write_json(manifest_path, manifest)
    if len(pairs) != len(selected):
        raise CommunicationDiagnosticError(
            f"completed {len(pairs)} of {len(selected)} diagnostic pairs"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/diagnostic_hotpotqa_training_ready_step0_communication.yaml",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = asyncio.run(
            run_diagnostic(_resolve(PROJECT_ROOT, args.config), project_root=PROJECT_ROOT)
        )
    except (CommunicationDiagnosticError, ConfigurationError, ValueError) as exc:
        print(f"HotpotQA communication diagnostic failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], "summary": manifest["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
