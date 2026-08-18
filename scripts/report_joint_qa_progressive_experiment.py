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


def _wrong_demo_lines(path: Optional[Path]) -> dict[str, int]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, int] = {}
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            task_id = _task_id(row)
        if task_id:
            result[task_id] = line_number
    return result


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
    metric_rows = _metric_rows_by_task(metrics)
    wrong_lines = _wrong_demo_lines(wrong_demos_path)
    candidates: list[str] = []
    candidates.extend(task_id for task_id in requested_task_ids if task_id in record_lines)
    candidates.extend(task_id for task_id in wrong_lines if task_id in record_lines)
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
        value: dict[str, Any] = {
            "task_id": task_id,
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
        }
        metric_line = metric_rows.get(task_id)
        if metric_line is not None:
            line_number, row = metric_line
            graph = row.get("agentgraph")
            value["metrics_artifact"] = {
                "path": metrics.get("source_path"),
                "jsonl_line": line_number,
            }
            if isinstance(graph, Mapping):
                value["exact_match"] = graph.get("exact_match", UNAVAILABLE)
                value["token_f1"] = graph.get("token_f1", UNAVAILABLE)
        if task_id in wrong_lines and wrong_demos_path is not None:
            value["wrong_demo_artifact"] = {
                "path": str(wrong_demos_path),
                "jsonl_line": wrong_lines[task_id],
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


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in report["steps"]:
        for dataset in DATASETS:
            value = step["datasets"][dataset]
            metrics = value["metrics"]
            graph = value["graph"]
            models = value["models"]
            usage = value["usage"]
            skill = value["skill_effect"]
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
        "本报告只聚合已保存的评测与 TrajectoryRecord；缺失文件或缺失字段标记为 `unavailable`，不按 0 计。EM/F1 使用现有固定评测脚本的严格分母。结构深度来自 AgentGraph，effective dependency depth 只反映 trajectory 中可核验的依赖传递证据。",
        "",
        "## Step0–N 训练曲线坐标与任务指标",
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

    for step in report["steps"]:
        lines.extend(["", f"## {step['label']} 诊断", ""])
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
                lines.append("- 完整 Demo artifact 引用：")
                for demo in value["demos"]:
                    trajectory = demo["trajectory_artifact"]
                    lines.append(
                        "  - `{task}`：`{path}:{line}`，topology=`{topology}`，structural depth=`{depth}`。".format(
                            task=demo["task_id"],
                            path=trajectory["path"],
                            line=trajectory["jsonl_line"],
                            topology=demo["topology_family"],
                            depth=demo["structural_depth"],
                        )
                    )
            else:
                lines.append("- 完整 Demo artifact 引用：`unavailable`。")
            lines.append("")

    lines.extend(
        [
            "## 解释边界",
            "",
            "- Skill-on/off 只有在同一 task_id 顺序可核验时才标记 `paired_task_ids_verified=true`；仅有汇总报告时仍可给出 aggregate delta，但不会宣称 paired effect。",
            "- `api_call_receipt_count` 是已保存的 Director request receipt 与 Executor ExecutionRecord 数量，不代表并行请求的 wall-clock 数量。",
            "- latency 是可用调用收据的求和与均值，不是端到端 wall-clock latency。",
            "- 模型 family 只读取 `execution.metadata.request.model.metadata.family`；缺失时不根据 model_id 猜测。",
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
