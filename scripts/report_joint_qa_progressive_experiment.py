#!/usr/bin/env python3
"""Build a read-only Chinese report for progressive HotpotQA/TriviaQA runs.

The reporter consumes the JSON/JSONL artifacts already emitted by the fixed
evaluation runners.  It deliberately does not execute a model, an evaluator,
or a training step.  Missing stages and missing artifact sections are retained
as ``unavailable`` instead of being converted to zero.

Input example::

    {
      "schema_version": "flowsteer.joint_qa.progressive_report_input.v1",
      "title": "HotpotQA + TriviaQA progressive experiment",
      "steps": [
        {
          "step": 0,
          "label": "Step0",
          "policy_version": "joint-step-000000",
          "policy_adapter": "theta_joint_step_000000",
          "skill_publication_path": ".../publication_results.json",
          "skill_store_path": ".../skills.json",
          "training_manifest_path": ".../training_manifest.json",
          "training_summary_path": ".../checkpoint/training_summary.json",
          "sync_receipt_path": ".../sync_receipt.json",
          "datasets": {
            "hotpotqa": {
              "metrics_path": ".../paired_results.jsonl",
              "trajectory_path": ".../agentgraph_trajectories.jsonl",
              "wrong_demos_path": ".../wrong_demos.jsonl",
              "skill_comparison": {
                "off_metrics_path": ".../skill_off/paired_results.jsonl",
                "on_metrics_path": ".../skill_on/paired_results.jsonl"
              },
              "demo_task_ids": ["hotpotqa:..."]
            }
          }
        }
      ]
    }

``metrics_path`` accepts either a round-report JSON or the paired-results
JSONL used by ``evaluate_hotpotqa_round.py`` and
``evaluate_triviaqa_round.py``.  JSONL aggregation reuses their strict
denominator implementation.  Graph statistics reuse
``aggregate_trajectory_diagnostics`` and therefore preserve the distinction
between structural depth and evidence-backed effective dependency depth.
The optional Skill/training paths only project existing receipts.  One LoRA
update is reported as a single-point training diagnostic, never as a
multi-point training curve.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_hotpotqa_round import _aggregate
from src.interactive.graph_diagnostics import aggregate_trajectory_diagnostics


INPUT_SCHEMA_VERSION = "flowsteer.joint_qa.progressive_report_input.v1"
OUTPUT_SCHEMA_VERSION = "flowsteer.joint_qa.progressive_report.v1"
DATASETS = ("hotpotqa", "triviaqa")
DATASET_LABELS = {"hotpotqa": "HotpotQA", "triviaqa": "TriviaQA"}
UNAVAILABLE = "unavailable"


class ProgressiveReportError(RuntimeError):
    """An available artifact is malformed or contradicts its step receipt."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgressiveReportError(f"{name} must be an object")
    return value


def _unavailable(reason: str, *, source_path: Optional[Path] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": UNAVAILABLE, "reason": reason}
    if source_path is not None:
        result["source_path"] = str(source_path)
    return result


def _resolve(base: Path, value: Any, name: str) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProgressiveReportError(f"{name} must be a non-empty path or null")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProgressiveReportError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ProgressiveReportError(f"{path}: expected a JSON object")
    return dict(value)


def _optional_json_artifact(
    base: Path, value: Any, *, name: str
) -> tuple[dict[str, Any], Optional[Path]]:
    """Read an optional JSON receipt without turning absence into zero values."""

    path = _resolve(base, value, name)
    if path is None:
        return _unavailable(f"{name} 未配置"), None
    if not path.is_file():
        return _unavailable(f"{name} 文件不存在", source_path=path), path
    result = _read_json(path)
    result["_source_path"] = str(path)
    return result, path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return rows
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProgressiveReportError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ProgressiveReportError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(dict(value))
    return rows


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgressiveReportError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProgressiveReportError(f"{name} must be finite and in [0, 1]")
    return result


def _metric_payload(
    aggregate: Mapping[str, Any],
    *,
    path: Path,
    source_kind: str,
    task_ids: Sequence[str] = (),
    rows: Sequence[Mapping[str, Any]] = (),
    policy_version: Optional[str] = None,
    policy_adapter: Optional[str] = None,
) -> dict[str, Any]:
    denominator = aggregate.get("denominator")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
    ):
        raise ProgressiveReportError(f"{path}: denominator must be positive")
    exact_match = _unit_float(
        aggregate.get("strict_exact_match"), f"{path}.strict_exact_match"
    )
    token_f1 = _unit_float(
        aggregate.get("strict_token_f1"), f"{path}.strict_token_f1"
    )
    return {
        "status": "available",
        "source_kind": source_kind,
        "source_path": str(path),
        "task_count": denominator,
        "completed": int(aggregate.get("completed", 0)),
        "evaluator_valid": int(aggregate.get("evaluator_valid", 0)),
        "strict_exact_match": exact_match,
        "strict_token_f1": token_f1,
        "strict_exact_match_percent": 100.0 * exact_match,
        "strict_token_f1_percent": 100.0 * token_f1,
        "policy_version": policy_version,
        "policy_adapter": policy_adapter,
        "_task_ids": tuple(task_ids),
        "_rows": tuple(dict(row) for row in rows),
    }


def _load_metrics(
    path: Optional[Path], *, dataset: str, name: str
) -> dict[str, Any]:
    if path is None:
        return _unavailable(f"{name} 未配置")
    if not path.is_file():
        return _unavailable(f"{name} 文件不存在", source_path=path)
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
        if not rows:
            return _unavailable(f"{name} JSONL 为空", source_path=path)
        task_ids: list[str] = []
        for index, row in enumerate(rows, start=1):
            task_id = row.get("task_id")
            graph = row.get("agentgraph")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ProgressiveReportError(f"{path}:{index}: missing task_id")
            if not task_id.startswith(f"{dataset}:"):
                raise ProgressiveReportError(
                    f"{path}:{index}: task_id is not from {dataset}"
                )
            if not isinstance(graph, Mapping):
                raise ProgressiveReportError(
                    f"{path}:{index}: missing agentgraph result"
                )
            task_ids.append(task_id)
        if len(set(task_ids)) != len(task_ids):
            raise ProgressiveReportError(f"{path}: duplicate task_id")
        return _metric_payload(
            _aggregate(rows, "agentgraph"),
            path=path,
            source_kind="paired_results_jsonl",
            task_ids=task_ids,
            rows=rows,
        )
    if path.suffix.lower() != ".json":
        raise ProgressiveReportError(f"{path}: metrics must be JSON or JSONL")
    report = _read_json(path)
    reported_dataset = report.get("dataset")
    if isinstance(reported_dataset, str) and reported_dataset.strip():
        if reported_dataset.strip().lower() != dataset:
            raise ProgressiveReportError(
                f"{path}: report dataset {reported_dataset!r} does not match {dataset!r}"
            )
    aggregate = _mapping(report.get("agentgraph"), f"{path}.agentgraph")
    sample_count = report.get("sample_count")
    if sample_count is not None and sample_count != aggregate.get("denominator"):
        raise ProgressiveReportError(
            f"{path}: sample_count differs from agentgraph.denominator"
        )
    raw_task_ids = report.get("task_ids", ())
    task_ids: tuple[str, ...] = ()
    if isinstance(raw_task_ids, Sequence) and not isinstance(
        raw_task_ids, (str, bytes)
    ):
        task_ids = tuple(str(item) for item in raw_task_ids)
    return _metric_payload(
        aggregate,
        path=path,
        source_kind="round_report_json",
        task_ids=task_ids,
        policy_version=(
            str(report["policy_version"])
            if isinstance(report.get("policy_version"), str)
            and str(report["policy_version"]).strip()
            else None
        ),
        policy_adapter=(
            str(report["policy_adapter"])
            if isinstance(report.get("policy_adapter"), str)
            and str(report["policy_adapter"]).strip()
            else None
        ),
    )


def _public_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _task_id(record: Mapping[str, Any]) -> str:
    task = record.get("task")
    task_id = task.get("task_id") if isinstance(task, Mapping) else None
    return task_id if isinstance(task_id, str) else ""


def _recorded_family(execution: Mapping[str, Any]) -> Optional[str]:
    metadata = execution.get("metadata")
    request = metadata.get("request") if isinstance(metadata, Mapping) else None
    model = request.get("model") if isinstance(request, Mapping) else None
    model_metadata = model.get("metadata") if isinstance(model, Mapping) else None
    family = model_metadata.get("family") if isinstance(model_metadata, Mapping) else None
    if isinstance(family, str) and family.strip():
        return family.strip()
    return None


def _numeric_sum(values: Iterable[Any]) -> dict[str, Any]:
    present = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not present:
        return {"status": UNAVAILABLE, "value": None, "receipt_count": 0}
    return {
        "status": "available",
        "value": sum(present),
        "receipt_count": len(present),
    }


def _numeric_mean(values: Iterable[Any]) -> dict[str, Any]:
    present = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not present:
        return {"status": UNAVAILABLE, "value": None, "receipt_count": 0}
    return {
        "status": "available",
        "value": fmean(present),
        "receipt_count": len(present),
    }


def _trajectory_sections(
    path: Optional[Path], *, name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        missing = _unavailable(f"{name} 未配置")
        return missing, dict(missing), dict(missing), [], []
    if not path.is_file():
        missing = _unavailable(f"{name} 文件不存在", source_path=path)
        return missing, dict(missing), dict(missing), [], []
    records = _read_jsonl(path)
    if not records:
        missing = _unavailable(f"{name} JSONL 为空", source_path=path)
        return missing, dict(missing), dict(missing), [], []
    try:
        diagnostic = aggregate_trajectory_diagnostics(records)
    except (TypeError, ValueError) as exc:
        raise ProgressiveReportError(f"{path}: invalid trajectory graph receipt") from exc
    trajectories = diagnostic.get("trajectories", [])
    if not isinstance(trajectories, list):
        raise ProgressiveReportError(f"{path}: malformed graph diagnostics")
    graph = {
        "status": "available",
        "source_path": str(path),
        "task_count": int(diagnostic.get("task_count", 0)),
        "mean_agent_count": fmean(
            float(item.get("agent_count", 0)) for item in trajectories
        ),
        "mean_structural_depth": fmean(
            float(item.get("structural_depth", 0)) for item in trajectories
        ),
        "mean_effective_dependency_depth": fmean(
            float(item.get("effective_dependency_depth", 0)) for item in trajectories
        ),
        "agent_count_distribution": diagnostic.get("agent_count_distribution", {}),
        "structural_depth_distribution": diagnostic.get(
            "structural_depth_distribution", {}
        ),
        "effective_dependency_depth_distribution": diagnostic.get(
            "effective_dependency_depth_distribution", {}
        ),
        "effective_dependency_status_distribution": diagnostic.get(
            "effective_dependency_status_distribution", {}
        ),
        "topology_family_distribution": diagnostic.get(
            "topology_family_distribution", {}
        ),
        "role_family_distribution": diagnostic.get("role_family_distribution", {}),
        "execution_turn_count": int(diagnostic.get("execution_turn_count", 0)),
        "executor_call_count": int(diagnostic.get("executor_call_count", 0)),
    }

    model_ids: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    families: Counter[str] = Counter()
    cooccurrences: Counter[str] = Counter()
    missing_family = 0
    director_input_tokens: list[int] = []
    director_output_tokens: list[int] = []
    director_latencies: list[float] = []
    executor_input_tokens: list[int] = []
    executor_output_tokens: list[int] = []
    executor_latencies: list[float] = []
    director_calls = 0
    executor_calls = 0
    observed_policies: set[str] = set()
    observed_adapters: set[str] = set()

    for record in records:
        versions = record.get("versions")
        policy = versions.get("policy") if isinstance(versions, Mapping) else None
        if isinstance(policy, str) and policy.strip():
            observed_policies.add(policy.strip())
        record_families: set[str] = set()
        turns = record.get("turns", ())
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            adapter = turn.get("policy_adapter")
            if isinstance(adapter, str) and adapter.strip():
                observed_adapters.add(adapter.strip())
            if isinstance(turn.get("director_request_id"), str) and str(
                turn["director_request_id"]
            ).strip():
                director_calls += 1
            prompt_ids = turn.get("prompt_token_ids")
            if isinstance(prompt_ids, Sequence) and not isinstance(
                prompt_ids, (str, bytes)
            ):
                director_input_tokens.append(len(prompt_ids))
            output_ids = turn.get("output_token_ids")
            if isinstance(output_ids, Sequence) and not isinstance(
                output_ids, (str, bytes)
            ):
                director_output_tokens.append(len(output_ids))
            latency = turn.get("director_latency_ms")
            if isinstance(latency, (int, float)) and not isinstance(latency, bool):
                director_latencies.append(float(latency))
            executions = turn.get("executions", ())
            if not isinstance(executions, Sequence) or isinstance(
                executions, (str, bytes)
            ):
                continue
            for execution in executions:
                if not isinstance(execution, Mapping):
                    continue
                executor_calls += 1
                model_id = execution.get("model_id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids[model_id.strip()] += 1
                provider = execution.get("provider")
                if isinstance(provider, str) and provider.strip():
                    providers[provider.strip()] += 1
                family = _recorded_family(execution)
                if family is None:
                    missing_family += 1
                else:
                    families[family] += 1
                    record_families.add(family)
                if isinstance(execution.get("input_tokens"), int) and not isinstance(
                    execution.get("input_tokens"), bool
                ):
                    executor_input_tokens.append(int(execution["input_tokens"]))
                if isinstance(execution.get("output_tokens"), int) and not isinstance(
                    execution.get("output_tokens"), bool
                ):
                    executor_output_tokens.append(int(execution["output_tokens"]))
                execution_latency = execution.get("latency_ms")
                if isinstance(execution_latency, (int, float)) and not isinstance(
                    execution_latency, bool
                ):
                    executor_latencies.append(float(execution_latency))
        if record_families:
            key = " + ".join(sorted(record_families))
            cooccurrences[key] += 1

    models = {
        "status": "available",
        "source_path": str(path),
        "model_id_call_distribution": dict(sorted(model_ids.items())),
        "provider_call_distribution": dict(sorted(providers.items())),
        "recorded_model_family_call_distribution": dict(sorted(families.items())),
        "recorded_model_family_cooccurrence_by_trajectory": dict(
            sorted(cooccurrences.items())
        ),
        "family_unavailable_execution_count": missing_family,
        "family_source": "execution.metadata.request.model.metadata.family",
    }
    usage = {
        "status": "available",
        "source_path": str(path),
        "director_api_call_receipt_count": director_calls,
        "executor_api_call_receipt_count": executor_calls,
        "api_call_receipt_count": director_calls + executor_calls,
        "director_input_tokens": _numeric_sum(director_input_tokens),
        "director_output_tokens": _numeric_sum(director_output_tokens),
        "executor_input_tokens": _numeric_sum(executor_input_tokens),
        "executor_output_tokens": _numeric_sum(executor_output_tokens),
        "director_latency_ms_total": _numeric_sum(director_latencies),
        "director_latency_ms_mean": _numeric_mean(director_latencies),
        "executor_latency_ms_total": _numeric_sum(executor_latencies),
        "executor_latency_ms_mean": _numeric_mean(executor_latencies),
        "latency_scope": "sum_and_mean_of_available_call_receipts_not_wall_clock",
        "observed_policy_versions": sorted(observed_policies),
        "observed_policy_adapters": sorted(observed_adapters),
    }
    return graph, models, usage, records, [dict(item) for item in trajectories]


def _skill_effect(
    value: Any,
    *,
    dataset: str,
    base: Path,
    name: str,
) -> dict[str, Any]:
    if value is None:
        return _unavailable("未提供 Skill-off/Skill-on 对照")
    comparison = _mapping(value, name)
    off_path = _resolve(
        base, comparison.get("off_metrics_path"), f"{name}.off_metrics_path"
    )
    on_path = _resolve(
        base, comparison.get("on_metrics_path"), f"{name}.on_metrics_path"
    )
    off = _load_metrics(off_path, dataset=dataset, name="Skill-off metrics")
    on = _load_metrics(on_path, dataset=dataset, name="Skill-on metrics")
    if off.get("status") != "available" or on.get("status") != "available":
        return {
            "status": UNAVAILABLE,
            "reason": "Skill-off 或 Skill-on 指标不可用",
            "skill_off": _public_metrics(off),
            "skill_on": _public_metrics(on),
        }
    off_ids = tuple(off.get("_task_ids", ()))
    on_ids = tuple(on.get("_task_ids", ()))
    if off_ids and on_ids and off_ids != on_ids:
        return {
            "status": UNAVAILABLE,
            "reason": "Skill-off 与 Skill-on 的 task_id 不一致",
            "skill_off": _public_metrics(off),
            "skill_on": _public_metrics(on),
        }
    off_policy = off.get("policy_version")
    on_policy = on.get("policy_version")
    if off_policy and on_policy and off_policy != on_policy:
        return {
            "status": UNAVAILABLE,
            "reason": "Skill-off 与 Skill-on 的 policy_version 不一致",
            "skill_off": _public_metrics(off),
            "skill_on": _public_metrics(on),
        }
    delta_em = float(on["strict_exact_match"]) - float(off["strict_exact_match"])
    delta_f1 = float(on["strict_token_f1"]) - float(off["strict_token_f1"])
    return {
        "status": "available",
        "skill_off": _public_metrics(off),
        "skill_on": _public_metrics(on),
        "delta_exact_match": delta_em,
        "delta_token_f1": delta_f1,
        "delta_exact_match_percentage_points": 100.0 * delta_em,
        "delta_token_f1_percentage_points": 100.0 * delta_f1,
        "paired_task_ids_verified": bool(off_ids and on_ids and off_ids == on_ids),
        "same_policy_verified": bool(off_policy and on_policy and off_policy == on_policy),
        "interpretation": (
            "paired_same_task_delta"
            if off_ids and on_ids and off_ids == on_ids
            else "aggregate_delta_task_alignment_unavailable"
        ),
    }


def _metric_rows_by_task(metrics: Mapping[str, Any]) -> dict[str, tuple[int, Mapping[str, Any]]]:
    result: dict[str, tuple[int, Mapping[str, Any]]] = {}
    rows = metrics.get("_rows", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return result
    for line_number, row in enumerate(rows, start=1):
        if isinstance(row, Mapping) and isinstance(row.get("task_id"), str):
            result[str(row["task_id"])] = (line_number, row)
    return result


def _wrong_demo_rows(
    path: Optional[Path],
) -> dict[str, tuple[int, Mapping[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            task_id = _task_id(row)
        if task_id:
            result[task_id] = (line_number, row)
    return result


def _final_graph_snapshot(record: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = record.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return {}
    for turn in reversed(turns):
        snapshot = turn.get("graph_snapshot") if isinstance(turn, Mapping) else None
        if isinstance(snapshot, Mapping):
            return snapshot
    return {}


def _director_actions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    turns = record.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return result
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            continue
        result.append(
            {
                "turn_index": turn_index,
                "round_index": turn.get("round_index", UNAVAILABLE),
                "action": turn.get("action", UNAVAILABLE),
                "policy_response": turn.get("policy_response", UNAVAILABLE),
                "canvas_feedback": turn.get("canvas_feedback", UNAVAILABLE),
                "graph_revision": turn.get("graph_revision", UNAVAILABLE),
                "director_request_id": turn.get("director_request_id", UNAVAILABLE),
            }
        )
    return result


def _final_agents_and_relations(
    record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[str]]:
    snapshot = _final_graph_snapshot(record)
    raw_nodes = snapshot.get("nodes", ())
    nodes = raw_nodes if isinstance(raw_nodes, Sequence) and not isinstance(
        raw_nodes, (str, bytes)
    ) else ()
    agents = [
        {
            "agent_id": node.get("id", UNAVAILABLE),
            "role_family": node.get("role_family", UNAVAILABLE),
            "model_id": node.get("model_id", UNAVAILABLE),
            "contract": node.get("contract", UNAVAILABLE),
        }
        for node in nodes
        if isinstance(node, Mapping)
    ]
    directed_relations: list[dict[str, Any]] = []
    raw_relations = snapshot.get("relations", ())
    relations = raw_relations if isinstance(raw_relations, Sequence) and not isinstance(
        raw_relations, (str, bytes)
    ) else ()
    for relation in relations:
        if not isinstance(relation, Mapping):
            continue
        source = relation.get("source_id")
        target = relation.get("target_id")
        if relation.get("source_to_target") is True:
            directed_relations.append(
                {"source_agent_id": source, "target_agent_id": target}
            )
        if relation.get("target_to_source") is True:
            directed_relations.append(
                {"source_agent_id": target, "target_agent_id": source}
            )
    output_agent_id = snapshot.get("output_agent_id")
    return (
        agents,
        directed_relations,
        output_agent_id if isinstance(output_agent_id, str) else None,
    )


def _communication_and_output_inbox(
    record: Mapping[str, Any], *, output_agent_id: Optional[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    communications: list[dict[str, Any]] = []
    output_receipts: list[dict[str, Any]] = []
    turns = record.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return communications, _unavailable("trajectory turns 不可用")
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            continue
        executions = turn.get("executions", ())
        if not isinstance(executions, Sequence) or isinstance(
            executions, (str, bytes)
        ):
            continue
        for execution in executions:
            if not isinstance(execution, Mapping):
                continue
            metadata = execution.get("metadata")
            request = metadata.get("request") if isinstance(metadata, Mapping) else None
            request = request if isinstance(request, Mapping) else {}
            upstream = request.get("upstream", ())
            upstream = upstream if isinstance(upstream, Sequence) and not isinstance(
                upstream, (str, bytes)
            ) else ()
            for message in upstream:
                if not isinstance(message, Mapping):
                    continue
                communications.append(
                    {
                        "turn_index": turn_index,
                        "execution_id": execution.get("execution_id", UNAVAILABLE),
                        "consumer_agent_id": execution.get("agent_id", UNAVAILABLE),
                        "source_agent_id": message.get(
                            "source_agent_id", UNAVAILABLE
                        ),
                        "target_agent_id": message.get(
                            "target_agent_id", UNAVAILABLE
                        ),
                        "message_type": message.get("message_type", UNAVAILABLE),
                        "content": message.get(
                            "content", message.get("artifact", UNAVAILABLE)
                        ),
                        "request_or_dependency": message.get(
                            "request_or_dependency", UNAVAILABLE
                        ),
                        "graph_revision": message.get(
                            "graph_revision", execution.get("graph_revision", UNAVAILABLE)
                        ),
                    }
                )
            if output_agent_id is not None and execution.get("agent_id") == output_agent_id:
                rendered = request.get("rendered_messages", ())
                output_receipts.append(
                    {
                        "turn_index": turn_index,
                        "execution_id": execution.get("execution_id", UNAVAILABLE),
                        "agent_id": execution.get("agent_id", UNAVAILABLE),
                        "model_id": execution.get("model_id", UNAVAILABLE),
                        "phase": request.get("phase", UNAVAILABLE),
                        "upstream": [dict(item) for item in upstream if isinstance(item, Mapping)],
                        "rendered_messages": (
                            [dict(item) for item in rendered if isinstance(item, Mapping)]
                            if isinstance(rendered, Sequence)
                            and not isinstance(rendered, (str, bytes))
                            else []
                        ),
                    }
                )
    if not output_receipts:
        inbox = _unavailable("Output Agent execution receipt 不可用")
    else:
        inbox = {
            "status": "available",
            "selection": "last_output_agent_execution_receipt",
            **output_receipts[-1],
        }
    return communications, inbox


def _failure_origin(
    record: Mapping[str, Any],
    *,
    metric_row: Optional[Mapping[str, Any]],
    wrong_row: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_executions: list[dict[str, Any]] = []
    turns = record.get("turns", ())
    if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes)):
        for turn_index, turn in enumerate(turns):
            executions = turn.get("executions", ()) if isinstance(turn, Mapping) else ()
            if not isinstance(executions, Sequence) or isinstance(
                executions, (str, bytes)
            ):
                continue
            for execution in executions:
                if not isinstance(execution, Mapping):
                    continue
                error_type = execution.get("error_type")
                if isinstance(error_type, str) and error_type.strip():
                    failed_executions.append(
                        {
                            "turn_index": turn_index,
                            "execution_id": execution.get("execution_id", UNAVAILABLE),
                            "agent_id": execution.get("agent_id", UNAVAILABLE),
                            "error_type": error_type,
                        }
                    )
    evaluation = record.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    graph_result = metric_row.get("agentgraph") if isinstance(metric_row, Mapping) else None
    exact_match = graph_result.get("exact_match") if isinstance(graph_result, Mapping) else None
    if failed_executions:
        observed_boundary = "executor_execution_receipt"
    elif evaluation.get("valid") is False:
        observed_boundary = "evaluator_receipt"
    elif exact_match == 0 or exact_match == 0.0:
        observed_boundary = "final_answer_vs_ground_truth"
    else:
        observed_boundary = "no_failure_recorded"
    failure_type = None
    if isinstance(wrong_row, Mapping) and isinstance(wrong_row.get("failure_type"), str):
        failure_type = str(wrong_row["failure_type"])
    elif isinstance(metric_row, Mapping) and isinstance(
        metric_row.get("failure_type"), str
    ):
        failure_type = str(metric_row["failure_type"])
    return {
        "observed_failure_boundary": observed_boundary,
        "recorded_failure_type": failure_type or UNAVAILABLE,
        "evaluation_valid": evaluation.get("valid", UNAVAILABLE),
        "evaluation_reason": evaluation.get("reason", UNAVAILABLE),
        "terminal_failure": record.get("terminal_failure", UNAVAILABLE),
        "failed_execution_receipts": failed_executions,
        "causal_root_cause": UNAVAILABLE,
        "attribution_scope": (
            "recorded_failure_signals_only; no causal root cause is inferred"
        ),
    }


def _demo_references(
    *,
    trajectory_path: Optional[Path],
    records: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    wrong_demos_path: Optional[Path],
    requested_task_ids: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    if trajectory_path is None or not records or limit < 1:
        return []
    record_lines = {
        _task_id(record): line_number
        for line_number, record in enumerate(records, start=1)
        if _task_id(record)
    }
    diagnostic_by_task = {
        str(item.get("task_id")): item
        for item in diagnostics
        if isinstance(item.get("task_id"), str)
    }
    record_by_task = {
        _task_id(record): record for record in records if _task_id(record)
    }
    metric_rows = _metric_rows_by_task(metrics)
    wrong_rows = _wrong_demo_rows(wrong_demos_path)
    candidates: list[str] = []
    candidates.extend(task_id for task_id in requested_task_ids if task_id in record_lines)
    candidates.extend(task_id for task_id in wrong_rows if task_id in record_lines)
    candidates.extend(
        str(item.get("task_id"))
        for item in sorted(
            diagnostics,
            key=lambda entry: (
                int(entry.get("structural_depth", 0)),
                int(entry.get("agent_count", 0)),
            ),
            reverse=True,
        )
        if str(item.get("task_id")) in record_lines
    )
    candidates.extend(record_lines)
    selected: list[str] = []
    for task_id in candidates:
        if task_id not in selected:
            selected.append(task_id)
        if len(selected) >= limit:
            break
    result: list[dict[str, Any]] = []
    for task_id in selected:
        diagnostic = diagnostic_by_task.get(task_id, {})
        record = record_by_task[task_id]
        task = record.get("task")
        task = task if isinstance(task, Mapping) else {}
        agents, directed_relations, output_agent_id = _final_agents_and_relations(
            record
        )
        directed_communication, output_agent_inbox = (
            _communication_and_output_inbox(
                record, output_agent_id=output_agent_id
            )
        )
        metric_entry = metric_rows.get(task_id)
        metric_row = metric_entry[1] if metric_entry is not None else None
        wrong_entry = wrong_rows.get(task_id)
        wrong_row = wrong_entry[1] if wrong_entry is not None else None
        graph_result = (
            metric_row.get("agentgraph") if isinstance(metric_row, Mapping) else None
        )
        value: dict[str, Any] = {
            "task_id": task_id,
            "question": task.get(
                "question",
                metric_row.get("question", UNAVAILABLE)
                if isinstance(metric_row, Mapping)
                else UNAVAILABLE,
            ),
            "ground_truth": task.get(
                "ground_truth",
                metric_row.get("ground_truth", UNAVAILABLE)
                if isinstance(metric_row, Mapping)
                else UNAVAILABLE,
            ),
            "final_answer": record.get(
                "final_answer",
                graph_result.get("final_answer", UNAVAILABLE)
                if isinstance(graph_result, Mapping)
                else UNAVAILABLE,
            ),
            "trajectory_artifact": {
                "path": str(trajectory_path),
                "jsonl_line": record_lines[task_id],
                "record_scope": "complete_TrajectoryRecord",
            },
            "topology_family": diagnostic.get("topology_family", UNAVAILABLE),
            "agent_count": diagnostic.get("agent_count", UNAVAILABLE),
            "structural_depth": diagnostic.get("structural_depth", UNAVAILABLE),
            "effective_dependency_depth": diagnostic.get(
                "effective_dependency_depth", UNAVAILABLE
            ),
            "director_actions": _director_actions(record),
            "agents": agents,
            "directed_relations": directed_relations,
            "directed_communication": directed_communication,
            "output_agent_inbox": output_agent_inbox,
            "failure_origin": _failure_origin(
                record, metric_row=metric_row, wrong_row=wrong_row
            ),
        }
        if metric_entry is not None:
            line_number, row = metric_entry
            graph = row.get("agentgraph")
            value["metrics_artifact"] = {
                "path": metrics.get("source_path"),
                "jsonl_line": line_number,
            }
            if isinstance(graph, Mapping):
                value["exact_match"] = graph.get("exact_match", UNAVAILABLE)
                value["token_f1"] = graph.get("token_f1", UNAVAILABLE)
        if wrong_entry is not None and wrong_demos_path is not None:
            value["wrong_demo_artifact"] = {
                "path": str(wrong_demos_path),
                "jsonl_line": wrong_entry[0],
            }
        result.append(value)
    return result


def _dataset_report(
    value: Any,
    *,
    dataset: str,
    base: Path,
    default_demo_limit: int,
) -> dict[str, Any]:
    if value is None:
        return {
            "dataset": DATASET_LABELS[dataset],
            "status": UNAVAILABLE,
            "reason": "该 Step 未配置此数据集",
            "metrics": _unavailable("metrics 未配置"),
            "graph": _unavailable("trajectory 未配置"),
            "models": _unavailable("trajectory 未配置"),
            "usage": _unavailable("trajectory 未配置"),
            "skill_effect": _unavailable("未提供 Skill-off/Skill-on 对照"),
            "demos": [],
        }
    config = _mapping(value, f"datasets.{dataset}")
    raw_metrics_path = config.get("metrics_path", config.get("report_path"))
    metrics_path = _resolve(
        base, raw_metrics_path, f"datasets.{dataset}.metrics_path"
    )
    trajectory_path = _resolve(
        base, config.get("trajectory_path"), f"datasets.{dataset}.trajectory_path"
    )
    wrong_demos_path = _resolve(
        base,
        config.get("wrong_demos_path"),
        f"datasets.{dataset}.wrong_demos_path",
    )
    metrics = _load_metrics(metrics_path, dataset=dataset, name="metrics")
    graph, models, usage, records, diagnostics = _trajectory_sections(
        trajectory_path, name="trajectory"
    )
    skill_effect = _skill_effect(
        config.get("skill_comparison"),
        dataset=dataset,
        base=base,
        name=f"datasets.{dataset}.skill_comparison",
    )
    raw_demo_ids = config.get("demo_task_ids", ())
    if not isinstance(raw_demo_ids, Sequence) or isinstance(
        raw_demo_ids, (str, bytes)
    ):
        raise ProgressiveReportError(
            f"datasets.{dataset}.demo_task_ids must be an array"
        )
    raw_limit = config.get("demo_limit", default_demo_limit)
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit < 0:
        raise ProgressiveReportError(
            f"datasets.{dataset}.demo_limit must be non-negative"
        )
    demos = _demo_references(
        trajectory_path=trajectory_path,
        records=records,
        diagnostics=diagnostics,
        metrics=metrics,
        wrong_demos_path=wrong_demos_path,
        requested_task_ids=tuple(str(item) for item in raw_demo_ids),
        limit=raw_limit,
    )
    available_sections = sum(
        section.get("status") == "available"
        for section in (metrics, graph, models, usage)
    )
    status = (
        UNAVAILABLE
        if available_sections == 0
        else "available"
        if available_sections == 4
        else "partial"
    )
    observed_policies = set(usage.get("observed_policy_versions", ()))
    if isinstance(metrics.get("policy_version"), str):
        observed_policies.add(str(metrics["policy_version"]))
    observed_adapters = set(usage.get("observed_policy_adapters", ()))
    if isinstance(metrics.get("policy_adapter"), str):
        observed_adapters.add(str(metrics["policy_adapter"]))
    return {
        "dataset": DATASET_LABELS[dataset],
        "status": status,
        "metrics": _public_metrics(metrics),
        "graph": graph,
        "models": models,
        "usage": usage,
        "skill_effect": skill_effect,
        "demos": demos,
        "observed_policy_versions": sorted(observed_policies),
        "observed_policy_adapters": sorted(observed_adapters),
    }


def _declared_version(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProgressiveReportError(f"{name} must be a non-empty string or null")
    return value.strip()


def _resolve_step_version(
    declared: Optional[str], observed: set[str], *, step: int, name: str
) -> tuple[str, str]:
    if declared is not None:
        mismatches = observed - {declared}
        if mismatches:
            raise ProgressiveReportError(
                f"Step{step} {name} {declared!r} contradicts receipts {sorted(observed)!r}"
            )
        return declared, "declared_and_receipt_consistent" if observed else "declared_only"
    if len(observed) == 1:
        return next(iter(observed)), "receipt"
    if not observed:
        return UNAVAILABLE, UNAVAILABLE
    return UNAVAILABLE, "conflicting_receipts"


def _macro_metrics(datasets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    available: list[str] = []
    missing: list[str] = []
    exact_matches: list[float] = []
    token_f1s: list[float] = []
    task_count = 0
    for dataset in DATASETS:
        metrics = datasets[dataset].get("metrics", {})
        if not isinstance(metrics, Mapping) or metrics.get("status") != "available":
            missing.append(dataset)
            continue
        available.append(dataset)
        exact_matches.append(float(metrics["strict_exact_match"]))
        token_f1s.append(float(metrics["strict_token_f1"]))
        task_count += int(metrics["task_count"])
    if missing:
        return {
            "status": UNAVAILABLE,
            "reason": "HotpotQA 与 TriviaQA 两个数据集指标必须同时可用",
            "available_datasets": available,
            "missing_datasets": missing,
            "weighting": "unweighted_dataset_macro",
        }
    exact_match = fmean(exact_matches)
    token_f1 = fmean(token_f1s)
    return {
        "status": "available",
        "dataset_count": len(DATASETS),
        "component_datasets": list(DATASETS),
        "component_task_count": task_count,
        "strict_exact_match": exact_match,
        "strict_token_f1": token_f1,
        "strict_exact_match_percent": 100.0 * exact_match,
        "strict_token_f1_percent": 100.0 * token_f1,
        "weighting": "unweighted_dataset_macro",
    }


def _skill_evidence_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _unavailable("Skill evidence 不可用")
    keys = (
        "baseline",
        "effective_pairs",
        "paired_effect_mean",
        "calibrated_lower",
        "calibrated_upper",
        "harm_probability",
        "empirical_coverage",
        "heldout_task_families",
        "validation_splits",
    )
    result = {key: value.get(key, UNAVAILABLE) for key in keys}
    result["status"] = "available"
    result["evidence_id_count"] = (
        len(value["evidence_ids"])
        if isinstance(value.get("evidence_ids"), Sequence)
        and not isinstance(value.get("evidence_ids"), (str, bytes))
        else UNAVAILABLE
    )
    return result


def _skill_projection(
    step_config: Mapping[str, Any], *, base: Path, step_index: int
) -> dict[str, Any]:
    publication, publication_path = _optional_json_artifact(
        base,
        step_config.get("skill_publication_path"),
        name=f"steps[{step_index}].skill_publication_path",
    )
    store, store_path = _optional_json_artifact(
        base,
        step_config.get("skill_store_path"),
        name=f"steps[{step_index}].skill_store_path",
    )
    publication_available = publication_path is not None and publication_path.is_file()
    store_available = store_path is not None and store_path.is_file()
    if not publication_available and not store_available:
        return _unavailable("Skill publication/store 未配置或不可用")

    raw_publications = publication.get("publications", {})
    publications = raw_publications if isinstance(raw_publications, Mapping) else {}
    raw_current = store.get("current", {})
    current = raw_current if isinstance(raw_current, Mapping) else {}
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        raw_publication = publications.get(dataset)
        dataset_publication = (
            raw_publication if isinstance(raw_publication, Mapping) else {}
        )
        embedded = dataset_publication.get("skill")
        embedded = embedded if isinstance(embedded, Mapping) else {}
        embedded_id = embedded.get("skill_id")
        store_skill: Mapping[str, Any] = {}
        if isinstance(embedded_id, str) and isinstance(current.get(embedded_id), Mapping):
            store_skill = current[embedded_id]
        else:
            matches = [
                candidate
                for candidate in current.values()
                if isinstance(candidate, Mapping)
                and isinstance(candidate.get("condition"), Mapping)
                and candidate["condition"].get("task_family") == dataset
            ]
            if len(matches) == 1:
                store_skill = matches[0]
        skill = store_skill or embedded
        gate = dataset_publication.get("gate")
        gate = gate if isinstance(gate, Mapping) else {}
        if not skill and not dataset_publication:
            datasets[dataset] = _unavailable(
                f"{DATASET_LABELS[dataset]} Skill publication 不可用"
            )
            continue
        datasets[dataset] = {
            "status": "available",
            "skill_id": skill.get("skill_id", embedded_id or UNAVAILABLE),
            "skill_status": skill.get("status", UNAVAILABLE),
            "skill_version": skill.get("version", UNAVAILABLE),
            "selected_candidate": dataset_publication.get(
                "selected_candidate", UNAVAILABLE
            ),
            "gate": {
                "approved": gate.get("approved", UNAVAILABLE),
                "no_practical_value": gate.get("no_practical_value", UNAVAILABLE),
                "reasons": gate.get("reasons", []),
            },
            "activated_epoch": skill.get("activated_epoch", UNAVAILABLE),
            "eligible_epoch": skill.get("eligible_epoch", UNAVAILABLE),
            "condition": skill.get("condition", UNAVAILABLE),
            "evidence": _skill_evidence_projection(skill.get("evidence")),
        }
    available_count = sum(
        value.get("status") == "available" for value in datasets.values()
    )
    return {
        "status": (
            "available"
            if available_count == len(DATASETS)
            else "partial"
            if available_count
            else UNAVAILABLE
        ),
        "publication_source_path": (
            str(publication_path) if publication_available else UNAVAILABLE
        ),
        "store_source_path": str(store_path) if store_available else UNAVAILABLE,
        "experiment_version": publication.get("experiment_version", UNAVAILABLE),
        "active_datasets": publication.get("active_datasets", UNAVAILABLE),
        "causal_estimand": publication.get("causal_estimand", UNAVAILABLE),
        "datasets": datasets,
    }


def _training_projection(
    step_config: Mapping[str, Any], *, base: Path, step_index: int
) -> dict[str, Any]:
    manifest, manifest_path = _optional_json_artifact(
        base,
        step_config.get("training_manifest_path"),
        name=f"steps[{step_index}].training_manifest_path",
    )
    summary, summary_path = _optional_json_artifact(
        base,
        step_config.get("training_summary_path"),
        name=f"steps[{step_index}].training_summary_path",
    )
    sync, sync_path = _optional_json_artifact(
        base,
        step_config.get("sync_receipt_path"),
        name=f"steps[{step_index}].sync_receipt_path",
    )
    manifest_available = manifest_path is not None and manifest_path.is_file()
    summary_available = summary_path is not None and summary_path.is_file()
    sync_available = sync_path is not None and sync_path.is_file()
    embedded_training = manifest.get("training")
    training = summary if summary_available else (
        embedded_training if isinstance(embedded_training, Mapping) else {}
    )
    embedded_sync = manifest.get("policy_sync")
    sync_receipt = sync if sync_available else (
        embedded_sync if isinstance(embedded_sync, Mapping) else {}
    )
    post_update = manifest.get("post_update_canaries")
    post_update = post_update if isinstance(post_update, Mapping) else {}
    if not manifest_available and not summary_available and not sync_available:
        return _unavailable("training/sync receipts 未配置或不可用")

    diagnostic_keys = (
        "optimizer_updates",
        "loss",
        "grad_norm",
        "trainable_update_l2",
        "behavior_policy_version",
        "updated_policy_version",
        "informative_groups",
        "trained_groups",
        "trained_trajectories",
        "oom_backoff_count",
    )
    optimizer = {
        key: training.get(key, UNAVAILABLE) if isinstance(training, Mapping) else UNAVAILABLE
        for key in diagnostic_keys
    }
    optimizer["status"] = (
        "available"
        if isinstance(training, Mapping) and "optimizer_updates" in training
        else UNAVAILABLE
    )
    synchronization = (
        {
            "status": sync_receipt.get("status", UNAVAILABLE),
            "success": sync_receipt.get("success", UNAVAILABLE),
            "adapter_name": sync_receipt.get("adapter_name", UNAVAILABLE),
            "behavior_policy_version": sync_receipt.get(
                "behavior_policy_version", UNAVAILABLE
            ),
            "candidate_policy_version": sync_receipt.get(
                "candidate_policy_version",
                sync_receipt.get("new_policy_version", UNAVAILABLE),
            ),
            "checkpoint_path": sync_receipt.get("checkpoint_path", UNAVAILABLE),
            "canary_succeeded": sync_receipt.get("canary_succeeded", UNAVAILABLE),
        }
        if isinstance(sync_receipt, Mapping) and sync_receipt
        else _unavailable("adapter synchronization receipt 不可用")
    )
    canary_available = (
        isinstance(synchronization, Mapping)
        and synchronization.get("canary_succeeded") != UNAVAILABLE
    ) or bool(post_update)
    canary = (
        {
            "status": "available",
            "sync_canary_succeeded": synchronization.get(
                "canary_succeeded", UNAVAILABLE
            ),
            "post_update_collected": post_update.get("collected", UNAVAILABLE),
            "post_update_policy_version": post_update.get(
                "policy_version", UNAVAILABLE
            ),
            "post_update_adapter_name": post_update.get(
                "adapter_name", UNAVAILABLE
            ),
            "post_update_trajectory_ids": post_update.get(
                "trajectory_ids", UNAVAILABLE
            ),
        }
        if canary_available
        else _unavailable("post-update canary receipt 不可用")
    )
    available_sections = sum(
        (
            optimizer.get("status") == "available",
            synchronization.get("status") != UNAVAILABLE,
            canary.get("status") == "available",
        )
    )
    return {
        "status": (
            "available"
            if available_sections == 3
            else "partial"
            if available_sections
            else UNAVAILABLE
        ),
        "manifest_source_path": (
            str(manifest_path) if manifest_available else UNAVAILABLE
        ),
        "summary_source_path": str(summary_path) if summary_available else UNAVAILABLE,
        "sync_source_path": str(sync_path) if sync_available else UNAVAILABLE,
        "manifest_status": manifest.get("status", UNAVAILABLE),
        "skills_enabled": manifest.get("skills_enabled", UNAVAILABLE),
        "optimizer": optimizer,
        "synchronization": synchronization,
        "canary": canary,
        "interpretation": "single_point_training_diagnostics",
        "multi_point_training_curve_claimed": False,
    }


def build_progressive_report(
    spec: Mapping[str, Any], *, base_dir: str | Path
) -> dict[str, Any]:
    """Aggregate progressive evaluation artifacts without filling missing values."""

    if spec.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ProgressiveReportError(
            f"schema_version must be {INPUT_SCHEMA_VERSION!r}"
        )
    raw_steps = spec.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(
        raw_steps, (str, bytes)
    ) or not raw_steps:
        raise ProgressiveReportError("steps must be a non-empty array")
    raw_demo_limit = spec.get("demo_limit", 3)
    if (
        isinstance(raw_demo_limit, bool)
        or not isinstance(raw_demo_limit, int)
        or raw_demo_limit < 0
    ):
        raise ProgressiveReportError("demo_limit must be non-negative")
    base = Path(base_dir).expanduser().resolve()
    seen: set[int] = set()
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        step_config = _mapping(raw_step, f"steps[{index}]")
        ordinal = step_config.get("step")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ProgressiveReportError(f"steps[{index}].step must be non-negative")
        if ordinal in seen:
            raise ProgressiveReportError(f"duplicate step {ordinal}")
        seen.add(ordinal)
        raw_datasets = step_config.get("datasets", {})
        datasets = _mapping(raw_datasets, f"steps[{index}].datasets")
        unknown = set(datasets) - set(DATASETS)
        if unknown:
            raise ProgressiveReportError(
                f"steps[{index}] contains unknown datasets: {sorted(unknown)!r}"
            )
        values = {
            dataset: _dataset_report(
                datasets.get(dataset),
                dataset=dataset,
                base=base,
                default_demo_limit=raw_demo_limit,
            )
            for dataset in DATASETS
        }
        observed_policies = {
            policy
            for dataset in DATASETS
            for policy in values[dataset].get("observed_policy_versions", ())
        }
        observed_adapters = {
            adapter
            for dataset in DATASETS
            for adapter in values[dataset].get("observed_policy_adapters", ())
        }
        policy, policy_source = _resolve_step_version(
            _declared_version(
                step_config.get("policy_version"), f"steps[{index}].policy_version"
            ),
            observed_policies,
            step=ordinal,
            name="policy_version",
        )
        adapter, adapter_source = _resolve_step_version(
            _declared_version(
                step_config.get("policy_adapter"), f"steps[{index}].policy_adapter"
            ),
            observed_adapters,
            step=ordinal,
            name="policy_adapter",
        )
        macro = _macro_metrics(values)
        skill = _skill_projection(step_config, base=base, step_index=index)
        training = _training_projection(step_config, base=base, step_index=index)
        steps.append(
            {
                "step": ordinal,
                "label": str(step_config.get("label", f"Step{ordinal}")),
                "policy_version": policy,
                "policy_version_source": policy_source,
                "policy_adapter": adapter,
                "policy_adapter_source": adapter_source,
                "training_curve_coordinate": {
                    "step": ordinal,
                    "policy": policy,
                    "adapter": adapter,
                },
                "macro_metrics": macro,
                "skill": skill,
                "training_diagnostics": training,
                "datasets": values,
            }
        )
    steps.sort(key=lambda item: int(item["step"]))
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "title": str(
            spec.get("title", "HotpotQA + TriviaQA 渐进式编排实验报告")
        ),
        "metric_scope": "strict_answer_exact_match_and_token_f1",
        "missing_value_policy": "missing_artifacts_are_unavailable_not_zero",
        "graph_diagnostics_source": (
            "src.interactive.graph_diagnostics.aggregate_trajectory_diagnostics"
        ),
        "training_diagnostic_scope": (
            "each optimizer update is a single-point diagnostic; no multi-point "
            "training curve is inferred"
        ),
        "multi_point_training_curve_claimed": False,
        "steps": steps,
        "generated_at": _utc_now(),
    }


def _value_or_unavailable(section: Mapping[str, Any], key: str) -> Any:
    if section.get("status") != "available":
        return UNAVAILABLE
    value = section.get(key)
    return UNAVAILABLE if value is None else value


def _usage_value(section: Mapping[str, Any], key: str) -> Any:
    if section.get("status") != "available":
        return UNAVAILABLE
    value = section.get(key)
    if isinstance(value, Mapping):
        return value.get("value") if value.get("status") == "available" else UNAVAILABLE
    return UNAVAILABLE if value is None else value


def _projection_value(section: Any, key: str) -> Any:
    if not isinstance(section, Mapping) or section.get("status") == UNAVAILABLE:
        return UNAVAILABLE
    value = section.get(key)
    return UNAVAILABLE if value is None else value


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in report["steps"]:
        macro = step["macro_metrics"]
        step_skill = step["skill"]
        skill_datasets = (
            step_skill.get("datasets", {})
            if isinstance(step_skill, Mapping)
            else {}
        )
        training = step["training_diagnostics"]
        optimizer = (
            training.get("optimizer", {}) if isinstance(training, Mapping) else {}
        )
        synchronization = (
            training.get("synchronization", {})
            if isinstance(training, Mapping)
            else {}
        )
        canary = (
            training.get("canary", {}) if isinstance(training, Mapping) else {}
        )
        for dataset in DATASETS:
            value = step["datasets"][dataset]
            metrics = value["metrics"]
            graph = value["graph"]
            models = value["models"]
            usage = value["usage"]
            skill = value["skill_effect"]
            published_skill = skill_datasets.get(dataset, {})
            skill_evidence = (
                published_skill.get("evidence", {})
                if isinstance(published_skill, Mapping)
                else {}
            )
            off = skill.get("skill_off", {}) if isinstance(skill, Mapping) else {}
            on = skill.get("skill_on", {}) if isinstance(skill, Mapping) else {}
            rows.append(
                {
                    "step": step["step"],
                    "label": step["label"],
                    "policy_version": step["policy_version"],
                    "policy_adapter": step["policy_adapter"],
                    "dataset": DATASET_LABELS[dataset],
                    "status": value["status"],
                    "task_count": _value_or_unavailable(metrics, "task_count"),
                    "strict_exact_match": _value_or_unavailable(
                        metrics, "strict_exact_match"
                    ),
                    "strict_token_f1": _value_or_unavailable(
                        metrics, "strict_token_f1"
                    ),
                    "macro_strict_exact_match": _projection_value(
                        macro, "strict_exact_match"
                    ),
                    "macro_strict_token_f1": _projection_value(
                        macro, "strict_token_f1"
                    ),
                    "published_skill_id": _projection_value(
                        published_skill, "skill_id"
                    ),
                    "published_skill_status": _projection_value(
                        published_skill, "skill_status"
                    ),
                    "skill_evidence_effective_pairs": _projection_value(
                        skill_evidence, "effective_pairs"
                    ),
                    "skill_evidence_paired_effect_mean": _projection_value(
                        skill_evidence, "paired_effect_mean"
                    ),
                    "skill_off_exact_match": _value_or_unavailable(
                        off, "strict_exact_match"
                    ),
                    "skill_off_token_f1": _value_or_unavailable(
                        off, "strict_token_f1"
                    ),
                    "skill_on_exact_match": _value_or_unavailable(
                        on, "strict_exact_match"
                    ),
                    "skill_on_token_f1": _value_or_unavailable(
                        on, "strict_token_f1"
                    ),
                    "skill_delta_exact_match": _value_or_unavailable(
                        skill, "delta_exact_match"
                    ),
                    "skill_delta_token_f1": _value_or_unavailable(
                        skill, "delta_token_f1"
                    ),
                    "mean_agent_count": _value_or_unavailable(
                        graph, "mean_agent_count"
                    ),
                    "mean_structural_depth": _value_or_unavailable(
                        graph, "mean_structural_depth"
                    ),
                    "mean_effective_dependency_depth": _value_or_unavailable(
                        graph, "mean_effective_dependency_depth"
                    ),
                    "topology_family_distribution": (
                        json.dumps(
                            graph.get("topology_family_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if graph.get("status") == "available"
                        else UNAVAILABLE
                    ),
                    "model_family_call_distribution": (
                        json.dumps(
                            models.get("recorded_model_family_call_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if models.get("status") == "available"
                        else UNAVAILABLE
                    ),
                    "model_family_cooccurrence": (
                        json.dumps(
                            models.get(
                                "recorded_model_family_cooccurrence_by_trajectory", {}
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if models.get("status") == "available"
                        else UNAVAILABLE
                    ),
                    "director_input_tokens": _usage_value(
                        usage, "director_input_tokens"
                    ),
                    "director_output_tokens": _usage_value(
                        usage, "director_output_tokens"
                    ),
                    "executor_input_tokens": _usage_value(
                        usage, "executor_input_tokens"
                    ),
                    "executor_output_tokens": _usage_value(
                        usage, "executor_output_tokens"
                    ),
                    "director_latency_ms_total": _usage_value(
                        usage, "director_latency_ms_total"
                    ),
                    "executor_latency_ms_total": _usage_value(
                        usage, "executor_latency_ms_total"
                    ),
                    "api_call_receipt_count": _usage_value(
                        usage, "api_call_receipt_count"
                    ),
                    "optimizer_updates": _projection_value(
                        optimizer, "optimizer_updates"
                    ),
                    "loss": _projection_value(optimizer, "loss"),
                    "grad_norm": _projection_value(optimizer, "grad_norm"),
                    "trainable_update_l2": _projection_value(
                        optimizer, "trainable_update_l2"
                    ),
                    "sync_status": _projection_value(
                        synchronization, "status"
                    ),
                    "sync_success": _projection_value(
                        synchronization, "success"
                    ),
                    "canary_succeeded": _projection_value(
                        canary, "sync_canary_succeeded"
                    ),
                    "demo_artifact_references": json.dumps(
                        value.get("demos", []), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return rows


def _fmt_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{100.0 * float(value):.2f}%"
    return UNAVAILABLE


def _fmt_number(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return UNAVAILABLE


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['title']}",
        "",
        "本报告只聚合已保存的评测与 TrajectoryRecord；缺失文件或缺失字段标记为 `unavailable`，不按 0 计。EM/F1 使用现有固定评测脚本的严格分母。HotpotQA + TriviaQA 联合指标为两数据集等权宏平均。结构深度来自 AgentGraph，effective dependency depth 只反映 trajectory 中可核验的依赖传递证据。",
        "",
        "LoRA/GRPO 字段只是每个 optimizer update 的 single-point training diagnostics；本报告不将一次 update 表述为 multi-point training curve。",
        "",
        "## Step0–N 评测序列与任务指标",
        "",
        "| Step | Policy | Adapter | 数据集 | 状态 | N | EM | F1 | 平均 Agent 数 | structural depth | effective dependency depth | Skill ΔEM | Skill ΔF1 | API call receipts |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in report["steps"]:
        for dataset in DATASETS:
            value = step["datasets"][dataset]
            metrics = value["metrics"]
            graph = value["graph"]
            skill = value["skill_effect"]
            usage = value["usage"]
            lines.append(
                "| {step} | `{policy}` | `{adapter}` | {dataset} | {status} | {n} | {em} | {f1} | {agents} | {structural} | {effective} | {delta_em} | {delta_f1} | {calls} |".format(
                    step=step["step"],
                    policy=step["policy_version"],
                    adapter=step["policy_adapter"],
                    dataset=DATASET_LABELS[dataset],
                    status=value["status"],
                    n=_value_or_unavailable(metrics, "task_count"),
                    em=_fmt_percent(_value_or_unavailable(metrics, "strict_exact_match")),
                    f1=_fmt_percent(_value_or_unavailable(metrics, "strict_token_f1")),
                    agents=_fmt_number(_value_or_unavailable(graph, "mean_agent_count")),
                    structural=_fmt_number(
                        _value_or_unavailable(graph, "mean_structural_depth")
                    ),
                    effective=_fmt_number(
                        _value_or_unavailable(
                            graph, "mean_effective_dependency_depth"
                        )
                    ),
                    delta_em=(
                        f"{float(skill['delta_exact_match_percentage_points']):+.2f} pp"
                        if skill.get("status") == "available"
                        else UNAVAILABLE
                    ),
                    delta_f1=(
                        f"{float(skill['delta_token_f1_percentage_points']):+.2f} pp"
                        if skill.get("status") == "available"
                        else UNAVAILABLE
                    ),
                    calls=_usage_value(usage, "api_call_receipt_count"),
                )
            )
        macro = step["macro_metrics"]
        lines.append(
            "| {step} | `{policy}` | `{adapter}` | HotpotQA + TriviaQA macro | {status} | {n} | {em} | {f1} | — | — | — | — | — | — |".format(
                step=step["step"],
                policy=step["policy_version"],
                adapter=step["policy_adapter"],
                status=macro.get("status", UNAVAILABLE),
                n=_projection_value(macro, "component_task_count"),
                em=_fmt_percent(_projection_value(macro, "strict_exact_match")),
                f1=_fmt_percent(_projection_value(macro, "strict_token_f1")),
            )
        )

    for step in report["steps"]:
        lines.extend(["", f"## {step['label']} 诊断", ""])
        macro = step["macro_metrics"]
        if macro.get("status") == "available":
            lines.append(
                "- HotpotQA + TriviaQA macro：EM {em}，F1 {f1}（两数据集等权）。".format(
                    em=_fmt_percent(macro.get("strict_exact_match")),
                    f1=_fmt_percent(macro.get("strict_token_f1")),
                )
            )
        else:
            lines.append(
                f"- HotpotQA + TriviaQA macro：`unavailable`（{macro.get('reason', '')}）。"
            )
        training = step["training_diagnostics"]
        if training.get("status") in {"available", "partial"}:
            optimizer = training.get("optimizer", {})
            synchronization = training.get("synchronization", {})
            canary = training.get("canary", {})
            lines.append(
                "- single-point training diagnostics：optimizer_updates={updates}，loss={loss}，grad_norm={grad}，trainable_update_l2={update_l2}。".format(
                    updates=_projection_value(optimizer, "optimizer_updates"),
                    loss=_projection_value(optimizer, "loss"),
                    grad=_projection_value(optimizer, "grad_norm"),
                    update_l2=_projection_value(optimizer, "trainable_update_l2"),
                )
            )
            lines.append(
                "- adapter synchronization：status={status}，success={success}；post-update canary={canary}，collected={collected}。".format(
                    status=_projection_value(synchronization, "status"),
                    success=_projection_value(synchronization, "success"),
                    canary=_projection_value(canary, "sync_canary_succeeded"),
                    collected=_projection_value(canary, "post_update_collected"),
                )
            )
        else:
            lines.append("- single-point training diagnostics：`unavailable`。")
        step_skill = step["skill"]
        skill_datasets = (
            step_skill.get("datasets", {})
            if isinstance(step_skill, Mapping)
            else {}
        )
        for dataset in DATASETS:
            published = skill_datasets.get(dataset, {})
            if isinstance(published, Mapping) and published.get("status") == "available":
                evidence = published.get("evidence", {})
                lines.append(
                    "- {dataset} Skill：ID=`{skill_id}`，status=`{status}`，effective_pairs={pairs}，paired_effect_mean={effect}，calibrated interval=[{lower}, {upper}]，harm_probability={harm}。".format(
                        dataset=DATASET_LABELS[dataset],
                        skill_id=published.get("skill_id", UNAVAILABLE),
                        status=published.get("skill_status", UNAVAILABLE),
                        pairs=_projection_value(evidence, "effective_pairs"),
                        effect=_projection_value(evidence, "paired_effect_mean"),
                        lower=_projection_value(evidence, "calibrated_lower"),
                        upper=_projection_value(evidence, "calibrated_upper"),
                        harm=_projection_value(evidence, "harm_probability"),
                    )
                )
            else:
                lines.append(
                    f"- {DATASET_LABELS[dataset]} Skill publication：`unavailable`。"
                )
        lines.append("")
        for dataset in DATASETS:
            value = step["datasets"][dataset]
            lines.extend([f"### {DATASET_LABELS[dataset]}", ""])
            if value["status"] == UNAVAILABLE:
                lines.extend(["- 阶段状态：`unavailable`。", ""])
                continue
            graph = value["graph"]
            models = value["models"]
            skill = value["skill_effect"]
            usage = value["usage"]
            if graph.get("status") == "available":
                lines.append(
                    "- topology family 分布：`{}`。".format(
                        json.dumps(
                            graph.get("topology_family_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                )
                lines.append(
                    "- Agent 数分布：`{}`；structural depth 分布：`{}`；effective dependency depth 分布：`{}`。".format(
                        json.dumps(
                            graph.get("agent_count_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            graph.get("structural_depth_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            graph.get("effective_dependency_depth_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            else:
                lines.append(f"- 图结构诊断：`unavailable`（{graph.get('reason', '')}）。")
            if models.get("status") == "available":
                lines.append(
                    "- 模型 family 调用分布：`{}`；trajectory 内 family 共现：`{}`。".format(
                        json.dumps(
                            models.get("recorded_model_family_call_distribution", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            models.get(
                                "recorded_model_family_cooccurrence_by_trajectory", {}
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            if usage.get("status") == "available":
                lines.append(
                    "- 调用收据：Director {director}，Executor {executor}，合计 {total}；token 与 latency 的完整性见 JSON/CSV。".format(
                        director=usage.get("director_api_call_receipt_count"),
                        executor=usage.get("executor_api_call_receipt_count"),
                        total=usage.get("api_call_receipt_count"),
                    )
                )
            if skill.get("status") == "available":
                lines.append(
                    "- Skill-on/off：ΔEM {em:+.2f} pp，ΔF1 {f1:+.2f} pp；task_id 对齐={tasks}，同 policy={policy}。".format(
                        em=float(skill["delta_exact_match_percentage_points"]),
                        f1=float(skill["delta_token_f1_percentage_points"]),
                        tasks=skill["paired_task_ids_verified"],
                        policy=skill["same_policy_verified"],
                    )
                )
            else:
                lines.append(
                    f"- Skill-on/off：`unavailable`（{skill.get('reason', '')}）。"
                )
            if value.get("demos"):
                lines.extend(["", "#### 完整 Demo 展开", ""])
                for demo in value["demos"]:
                    trajectory = demo["trajectory_artifact"]
                    lines.extend(
                        [
                            "##### `{task}`".format(task=demo["task_id"]),
                            "",
                            "- TrajectoryRecord：`{path}:{line}`；topology=`{topology}`，structural depth=`{depth}`，effective dependency depth=`{effective}`。".format(
                            task=demo["task_id"],
                            path=trajectory["path"],
                            line=trajectory["jsonl_line"],
                            topology=demo["topology_family"],
                            depth=demo["structural_depth"],
                                effective=demo["effective_dependency_depth"],
                            ),
                            "- Ground Truth：`{}`。".format(
                                json.dumps(demo.get("ground_truth"), ensure_ascii=False)
                            ),
                            "- Final Answer：`{}`；EM=`{}`，F1=`{}`。".format(
                                json.dumps(demo.get("final_answer"), ensure_ascii=False),
                                demo.get("exact_match", UNAVAILABLE),
                                demo.get("token_f1", UNAVAILABLE),
                            ),
                            "",
                            "<details><summary>Question</summary>",
                            "",
                            "```text",
                            str(demo.get("question", UNAVAILABLE)),
                            "```",
                            "",
                            "</details>",
                            "",
                            "Director actions：",
                            "",
                            "```json",
                            json.dumps(
                                demo.get("director_actions", []),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "```",
                            "",
                            "Agent roles/models 与 directed relations：",
                            "",
                            "```json",
                            json.dumps(
                                {
                                    "agents": demo.get("agents", []),
                                    "directed_relations": demo.get(
                                        "directed_relations", []
                                    ),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "```",
                            "",
                            "Actual directed communication：",
                            "",
                            "```json",
                            json.dumps(
                                demo.get("directed_communication", []),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "```",
                            "",
                            "Output Agent inbox：",
                            "",
                            "```json",
                            json.dumps(
                                demo.get("output_agent_inbox", {}),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "```",
                            "",
                            "Failure origin receipt：",
                            "",
                            "```json",
                            json.dumps(
                                demo.get("failure_origin", {}),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            "```",
                            "",
                        ]
                    )
            else:
                lines.append("- 完整 Demo 展开：`unavailable`。")
            lines.append("")

    lines.extend(
        [
            "## 解释边界",
            "",
            "- Skill-on/off 只有在同一 task_id 顺序可核验时才标记 `paired_task_ids_verified=true`；仅有汇总报告时仍可给出 aggregate delta，但不会宣称 paired effect。",
            "- `api_call_receipt_count` 是已保存的 Director request receipt 与 Executor ExecutionRecord 数量，不代表并行请求的 wall-clock 数量。",
            "- latency 是可用调用收据的求和与均值，不是端到端 wall-clock latency。",
            "- 模型 family 只读取 `execution.metadata.request.model.metadata.family`；缺失时不根据 model_id 猜测。",
            "- Demo 的 failure origin 只展示 trajectory/evaluator/wrong-demo receipt 中的最早可观测失败边界，不自动归因为因果 root cause。",
            "- optimizer_updates/loss/grad_norm/trainable_update_l2 是 single-point training diagnostics；只有多个完整、可比的 update 点时才能构成 training curve。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_progressive_report_outputs(
    report: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Write canonical JSON, CSV, and Chinese Markdown report artifacts."""

    target = Path(output_dir).expanduser().resolve()
    json_path = target / "joint_qa_progressive_report.json"
    csv_path = target / "joint_qa_progressive_curve.csv"
    markdown_path = target / "joint_qa_progressive_report.zh-CN.md"
    rows = _csv_rows(report)
    if not rows:
        raise ProgressiveReportError("report has no CSV rows")
    target.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_path.with_name(f".{csv_path.name}.partial")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_temporary.replace(csv_path)
    output = dict(report)
    output["artifacts"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "chinese_markdown": str(markdown_path),
    }
    _atomic_text(
        json_path,
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(markdown_path, _markdown(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="progressive report input JSON")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec).expanduser().resolve()
    try:
        if not spec_path.is_file():
            raise ProgressiveReportError(f"missing spec: {spec_path}")
        report = build_progressive_report(
            _read_json(spec_path), base_dir=spec_path.parent
        )
        output = write_progressive_report_outputs(report, args.output_dir)
    except ProgressiveReportError as exc:
        parser.error(str(exc))
    print(json.dumps(output["artifacts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
