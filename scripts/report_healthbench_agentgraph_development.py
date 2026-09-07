#!/usr/bin/env python3
"""Offline, receipt-backed reports for completed AgentGraph-only development runs.

No Direct/paired artifacts, model, Tool, evaluator, or training invocation is
needed. Public files contain numeric/structural projections only. Full inputs,
turns, messages and evaluator receipts are written under evaluator_private.
Re-running this command recovers interrupted report generation atomically per
file. --verify-only builds everything in memory without writing any output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_healthbench_failure_demos import (
    FailureDemoReportError,
    _execution_view,
    _index,
    _json_details,
    _load_json_object,
    _turn_view,
)
from report_joint_qa_progressive_experiment import _atomic_text
from report_multidataset_stable_zero import (
    _communication_envelopes,
    _execution_records,
    _load_jsonl,
    _react_trace_entries,
    _tool_receipts,
)

SCHEMA = "flowsteer.healthbench.agentgraph-development-report.v1"
DEFAULT_EVALUATION = ROOT / "artifacts" / (
    "healthbench_professional_clinical_reference_sources_v2_39_dev5"
) / "evaluation"
DEFAULT_REPORT = ROOT / "reports" / (
    "healthbench_professional_clinical_reference_sources_v2_39"
)
METRICS = ("overall_score", "overall_score_length_adjusted")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _objects(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _deduplicate(rows: Sequence[Mapping[str, Any]], identity: Any) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    result = []
    for index, row in enumerate(rows):
        key = _canonical(identity(row, index))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _covered_sum(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, Any]:
    values = [row[field] for row in rows if _number(row.get(field))]
    return {"sum": sum(values) if values else None, "covered": len(values), "total": len(rows)}


def _usage(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {field: _covered_sum(rows, field) for field in (
        "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms", "attempt_count"
    )}


def _failure_records(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [record for turn in _objects(trajectory.get("turns"))
            for record in _objects(_mapping(turn.get("runtime_summary")).get("failure_records"))]
    return _deduplicate(rows, lambda row, index: (
        [row.get("request_id"), row.get("agent_id"), row.get("phase"), row.get("error_type")]
        if row.get("request_id") else ["receipt", row]
    ))


def _evidence_trajectory(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    """Adapt saved failure metadata to existing offline extractors, not an arm."""
    failures = [{
        "agent_id": record.get("agent_id"),
        "metadata": {
            "request": {"agent": {"id": record.get("agent_id")}},
            "response": _mapping(record.get("metadata")),
        },
    } for record in _failure_records(trajectory)]
    return {"turns": [*_objects(trajectory.get("turns")), {"executions": failures}]}


def _tool_identity(receipt: Mapping[str, Any], index: int) -> Any:
    start, end = receipt.get("started_at_monotonic"), receipt.get("ended_at_monotonic")
    if _number(start) and _number(end):
        return [receipt.get("tool_id"), start, end]
    request_id = receipt.get("receipt_id") or receipt.get("tool_call_id")
    if request_id:
        return [receipt.get("tool_id"), request_id]
    # Exact receipt equality is the only available fallback; report uncertainty.
    return ["exact_receipt_only", receipt]


def _tool_summary(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _tool_receipts(_evidence_trajectory(trajectory))
    unique = _deduplicate(raw, _tool_identity)
    ambiguous = sum(_tool_identity(row, i)[0] == "exact_receipt_only" for i, row in enumerate(unique))
    return {
        "receipt_occurrences": len(raw), "distinct_receipt_events": len(unique),
        "duplicate_receipt_occurrences": len(raw) - len(unique),
        "events_without_call_identity": ambiguous,
        "actual_calls": len(unique) if not ambiguous else None,
        "by_tool": dict(sorted(Counter(str(row.get("tool_id", "unknown")) for row in unique).items())),
        "error_types": dict(Counter(str(row["error_type"]) for row in unique if row.get("error_type"))),
        "latency_ms": _covered_sum(unique, "latency_ms"),
    }


def _model_receipts(trajectory: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int]:
    records = []
    missing = 0
    executions = _execution_records(trajectory)
    sources = [(row, _mapping(_mapping(row.get("metadata")).get("response"))) for row in executions]
    sources.extend((row, _mapping(row.get("metadata"))) for row in _failure_records(trajectory))
    for source_index, (source, response) in enumerate(sources):
        calls = _objects(response.get("model_calls"))
        if calls:
            for call_index, call in enumerate(calls):
                metadata = _mapping(call.get("metadata"))
                records.append({
                    **metadata,
                    "model_id": metadata.get("model_id") or source.get("model_id") or response.get("model_id") or "unknown",
                    "request_status": call.get("request_status", metadata.get("request_status")),
                    "_identity": call.get("request_id") or metadata.get("provider_request_id")
                    or [source_index, call_index],
                })
        elif "execution_id" in source and response and "model_calls" not in response:
            # A plain reasoning response is itself a saved model receipt. Do not
            # synthesize model calls from execution counts or absent usage fields.
            records.append({
                **response, "model_id": response.get("model_id") or source.get("model_id") or "unknown",
                "_identity": response.get("provider_request_id")
                or _mapping(_mapping(source.get("metadata")).get("request")).get("request_id")
                or source.get("execution_id"),
            })
        else:
            missing += 1
    return _deduplicate(records, lambda row, i: [row.get("model_id"), row["_identity"]]), missing


def _director_summary(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = _objects(trajectory.get("turns"))
    phases = []
    for turn_index, turn in enumerate(turns):
        saved = _mapping(_mapping(turn.get("runtime_summary")).get("director_generation_phase_receipts"))
        for name, value in saved.items():
            if isinstance(value, Mapping):
                phases.append({**value, "_identity": value.get("request_id") or [turn_index, name]})
    phases = _deduplicate(phases, lambda row, i: row["_identity"])
    return {
        "turns": len(turns), "recorded_phase_calls": len(phases),
        "turns_with_phase_receipts": sum(bool(_mapping(_mapping(t.get("runtime_summary")).get("director_generation_phase_receipts"))) for t in turns),
        "turn_attempt_count": _covered_sum(turns, "director_attempt_count"),
        "turn_latency_ms": _covered_sum(turns, "director_latency_ms"),
        "recorded_phase_usage": _usage(phases),
        "model_id": None,
        "policy_versions": sorted({str(t["policy_version"]) for t in turns if t.get("policy_version")}),
    }


def _graph_summary(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshots = [_mapping(turn.get("graph_snapshot")) for turn in _objects(trajectory.get("turns"))
                 if isinstance(turn.get("graph_snapshot"), Mapping)]
    if not snapshots:
        return {"available": False, "node_count": None, "topology": None}
    graph = snapshots[-1]
    nodes = _objects(graph.get("nodes"))
    # Contracts and other content-bearing node fields are intentionally omitted.
    public_nodes = [{key: node.get(key) for key in ("id", "model_id", "execution_mode")} for node in nodes]
    relations = [{key: relation.get(key) for key in (
        "source_id", "target_id", "source_to_target", "target_to_source"
    )} for relation in _objects(graph.get("relations"))]
    arrows = []
    for relation in relations:
        if relation["source_to_target"]:
            arrows.append(f"{relation['source_id']} → {relation['target_id']}")
        if relation["target_to_source"]:
            arrows.append(f"{relation['target_id']} → {relation['source_id']}")
    return {
        "available": True, "snapshot_scope": "last saved Canvas graph_snapshot; not inferred execution topology",
        "node_count": len(nodes), "max_saved_node_count": max(len(_objects(s.get("nodes"))) for s in snapshots),
        "relation_count": len(relations), "directed_edge_count": len(arrows),
        "revision": graph.get("revision"), "output_agent_id": graph.get("output_agent_id"),
        "nodes": public_nodes, "relations": relations,
        "topology": "; ".join(arrows) if arrows else "no directed relations",
    }


def _row_summary(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = _mapping(trajectory.get("evaluation"))
    details = _mapping(evaluation.get("details"))
    telemetry = _mapping(details.get("grader_telemetry"))
    calls, missing = _model_receipts(trajectory)
    failures = _failure_records(trajectory)
    provider_failures = [record for record in failures
                         if _mapping(record.get("metadata")).get("http_status") is not None
                         or _mapping(record.get("metadata")).get("provider_id") is not None]
    grader_usage = _mapping(telemetry.get("token_usage"))
    return {
        "task_id": _mapping(trajectory.get("task")).get("task_id", trajectory.get("task_id")),
        "trajectory_available": True,
        "trajectory_id": trajectory.get("trajectory_id"),
        "native_metrics": {key: _mapping(evaluation.get("metrics")).get(key) for key in METRICS},
        "valid": evaluation.get("valid"), "explicit_finish": trajectory.get("explicit_finish"),
        "terminal_failure": trajectory.get("terminal_failure"),
        "termination_reason": trajectory.get("termination_reason"),
        "graph": _graph_summary(trajectory),
        "agent_model": {
            "recorded_calls": len(calls), "sources_without_model_receipts": missing,
            "by_model": dict(sorted(Counter(str(call["model_id"]) for call in calls).items())),
            "request_status_counts": dict(Counter(str(call.get("request_status", "not_recorded")) for call in calls)),
            "usage": _usage(calls),
        },
        "director": _director_summary(trajectory), "tools": _tool_summary(trajectory),
        "errors": {
            "runtime_failure_receipts": len(failures),
            "runtime_failure_types": dict(Counter(str(row.get("error_type", "unknown")) for row in failures)),
            "provider_identified_runtime_failure_receipts": len(provider_failures),
            "execution_error_types": dict(Counter(str(row["error_type"]) for row in _execution_records(trajectory) if row.get("error_type"))),
            "grader_provider_errors": len(_objects(telemetry.get("provider_errors"))),
            "grader_error_present": bool(details.get("grader_error")),
        },
        "grader": {
            "model_id": details.get("judge_model"), "api_calls": telemetry.get("api_calls"),
            "api_call_receipts": len(_objects(telemetry.get("api_call_receipts"))),
            "latency_ms": telemetry.get("latency_ms"),
            "token_usage": {key: grader_usage.get(key) for key in (
                "input_tokens", "output_tokens", "total_tokens", "input_cached_tokens", "output_reasoning_tokens"
            )},
            "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries",
        },
        "communication_envelopes": len(_communication_envelopes(trajectory)),
    }


def _private_demo(trajectory: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    evidence = _evidence_trajectory(trajectory)
    sections = [f"## {summary['task_id']}", "", _json_details("Public summary", summary), "",
                _json_details("完整 TaskRecord / 原始模型输入 / evaluator-only target（若原件提供）", trajectory.get("task")), "",
                _json_details("最终完整输出", trajectory.get("final_answer")), ""]
    for turn in _objects(trajectory.get("turns")):
        sections += [_json_details(f"Director / Canvas turn {turn.get('round_index')}", _turn_view(turn)), ""]
        for execution in _objects(turn.get("executions")):
            sections += [_json_details(f"Agent input/output: {execution.get('execution_id')}", _execution_view(execution)), ""]
    sections += [
        _json_details("Agent-to-Agent CommunicationEnvelope", _communication_envelopes(trajectory)), "",
        _json_details("ReAct Action / Tool Observation（含失败执行已保存部分）", _react_trace_entries(evidence)), "",
        _json_details("去重真实 Tool receipts", _deduplicate(_tool_receipts(evidence), _tool_identity)), "",
        _json_details("Runtime failure receipts（完整，非医疗归因）", _failure_records(trajectory)), "",
        _json_details("原生 evaluator receipt（不重评、不重算评分）", trajectory.get("evaluation")), "",
        "无病例失分原因自动分类；由人工阅读上述完整 evidence 后决定。", "",
    ]
    return "\n".join(sections)


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _coverage(value: Mapping[str, Any]) -> str:
    return f"{_display(value['sum'])} ({value['covered']}/{value['total']})"


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise FailureDemoReportError("operational-failure evidence lacks a timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise FailureDemoReportError("operational-failure timestamps require time zones")
    return parsed


def _operational_evidence(
    evaluation_dir: Path, manifest: Mapping[str, Any], missing: Sequence[str],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    """Bind unmeasured tasks to this finished run, never to historical errors."""
    started = _timestamp(manifest.get("started_at"))
    completed = _timestamp(manifest.get("completed_at"))
    attempt_id = manifest.get("run_attempt_id")
    if completed < started or not isinstance(attempt_id, str) or not attempt_id:
        raise FailureDemoReportError("invalid completed run identity/time interval")
    selected = _index(_load_jsonl(evaluation_dir / "selected_tasks.jsonl"), name="selected tasks")
    if set(selected) != set(manifest["selected_task_ids"]):
        raise FailureDemoReportError("selected task evidence differs from fixed manifest population")
    progress = [row for row in _load_jsonl(evaluation_dir / "rollout_progress.jsonl")
                if row.get("run_attempt_id") == attempt_id]
    conditions = {row.get("condition_id") for row in progress}
    if len(conditions) != 1 or not all(isinstance(value, str) and value for value in conditions):
        raise FailureDemoReportError("current-run progress lacks one exact condition identity")
    if any(row.get("condition_id") not in conditions for row in trajectories.values()):
        raise FailureDemoReportError("terminal trajectories and current-run progress differ in condition")
    failures = _load_jsonl(evaluation_dir / "collection_failures.jsonl")
    result = {}
    for task_id in missing:
        task = selected[task_id]
        if not isinstance(task.get("question"), str) or not task["question"]:
            raise FailureDemoReportError(f"missing selected question for operational failure: {task_id}")
        task_progress = [row for row in progress if _mapping(row.get("event")).get("task_id") == task_id
                         and started <= _timestamp(_mapping(row.get("event")).get("timestamp")) <= completed]
        task_failures = [row for row in failures
                         if row.get("task_id") == task_id and row.get("condition") == "agentgraph"
                         and row.get("stage") == "collect"
                         and isinstance(row.get("error"), str) and bool(row["error"])
                         and row.get("run_attempt_id", attempt_id) == attempt_id
                         and started <= _timestamp(row.get("recorded_at")) <= completed]
        if not task_progress or not task_failures:
            raise FailureDemoReportError(f"missing current-run task/condition collection-failure evidence: {task_id}")
        result[task_id] = {"task_id": task_id, "selected_task": task, "progress": task_progress,
                           "collection_failures": task_failures, "run_attempt_id": attempt_id,
                           "condition_id": next(iter(conditions)),
                           "trajectory_available": False,
                           "recoverability": "No terminal trajectory or evaluator receipt is available. Progress/failure records cannot reconstruct missing calls. If evaluator_private/partial_trajectories.jsonl exists, its matching run/task diagnostics may preserve returned intermediate calls; they are non-scoreable and are not included in this report's terminal-trajectory totals."}
    return result


def _missing_summary(task_id: str, evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    # This is explicitly a public report row, never a synthetic trajectory.
    return {"task_id": task_id, "trajectory_available": False, "operational_failure": True,
            "native_metrics": {key: None for key in METRICS}, "valid": None,
            "explicit_finish": None, "terminal_failure": None, "termination_reason": None,
            "graph": None, "agent_model": None, "director": None, "tools": None, "grader": None,
            "collection_failure_receipts": len(evidence["collection_failures"]),
            "unavailable_reason": "terminal trajectory not persisted; collection failure supported by current-run task/condition evidence"}


def _private_operational_demo(evidence: Mapping[str, Any]) -> str:
    return "\n".join([
        f"## {evidence['task_id']} — operational failure / no terminal trajectory", "",
        "本题仍计入固定五题分母；raw、length-adjusted、valid、FINISH、完整调用/token/latency 均 N/A。",
        "终局 trajectory 与 evaluator receipt 未落盘，不能评分；progress/collect failure 不能恢复未返回的调用。如存在同目录 partial_trajectories.jsonl，请按本轮 run_attempt_id/task_id 查看已返回的中间调用诊断；这些不是终局轨迹，不计入本报告调用总量。", "",
        _json_details("本轮 selected question、progress 与 collect failure（完整原始证据）", evidence), "",
    ])


def _public_report(summary: Mapping[str, Any]) -> str:
    rows = summary["tasks"]
    lines = ["# HealthBench Professional AgentGraph-only development report", "",
             "仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。",
             "分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。", "",
             f"- selected / persisted terminal trajectories：{summary['sample_count']} / {summary['completed_count']}；operational failures without trajectory：{summary['operational_failure_count']}",
             f"- evaluator valid：{summary['valid_count']}；explicit FINISH：{summary['finish_count']}；terminal failure：{summary['terminal_failure_count']}",
             f"- 严格全五题 native mean raw：{_display(summary['native_metric_means'][METRICS[0]]['mean'])}；length-adjusted：{_display(summary['native_metric_means'][METRICS[1]]['mean'])}；固定 denominator={summary['sample_count']}，任一缺失则 N/A，绝不置零。",
             f"- completed-only native mean raw：{_display(summary['completed_only_native_metric_means'][METRICS[0]]['mean'])} ({summary['completed_only_native_metric_means'][METRICS[0]]['observed_count']}/{summary['sample_count']})；length-adjusted：{_display(summary['completed_only_native_metric_means'][METRICS[1]]['mean'])} ({summary['completed_only_native_metric_means'][METRICS[1]]['observed_count']}/{summary['sample_count']})",
             "- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。",
             f"- 已落盘 trajectory 内 Agent recorded model calls：{summary['totals']['agent_recorded_model_calls']}；Director recorded phase calls：{summary['totals']['director_recorded_phase_calls']}；observed Tool calls（题内去重后求和）：{_display(summary['totals']['observed_actual_tool_calls'])}；全五题 actual Tool calls：{_display(summary['totals']['actual_tool_calls'])}", "",
             "| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |",
             "|---|---:|---:|---|---|---|---:|---|---:|---:|---:|"]
    for row in rows:
        if not row["trajectory_available"]:
            lines.append("| " + row["task_id"] + " | " + " | ".join(["N/A"] * 10) + " |")
            continue
        graph = row["graph"]
        fields = [row["task_id"], *[row["native_metrics"][key] for key in METRICS], row["valid"], row["explicit_finish"], row["terminal_failure"],
                  f"{graph['node_count']} / {graph.get('max_saved_node_count', 'N/A')}", graph["topology"], row["agent_model"]["recorded_calls"], row["tools"]["actual_calls"], row["errors"]["grader_provider_errors"]]
        lines.append("| " + " | ".join(_display(value) for value in fields) + " |")
    lines += ["", "## 模型、调用与覆盖范围", "",
              "- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。",
              "- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。",
              "- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。",
              "- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。",
              "- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。",
              "- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。",
              f"- 调用/token/latency 只覆盖已落盘的 {summary['completed_count']}/{summary['sample_count']} 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。", ""]
    for row in rows:
        if not row["trajectory_available"]:
            lines += [f"### {row['task_id']}", "",
                      "本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。", ""]
            continue
        lines += [f"### {row['task_id']}", "",
                  f"- Agent model calls：`{json.dumps(row['agent_model']['by_model'], ensure_ascii=False, sort_keys=True)}`；missing-source receipts={row['agent_model']['sources_without_model_receipts']}",
                  f"- Director turns={row['director']['turns']}；recorded phase calls={row['director']['recorded_phase_calls']}；turn attempts={_coverage(row['director']['turn_attempt_count'])}；turn latency_ms={_coverage(row['director']['turn_latency_ms'])}",
                  f"- Tool calls：`{json.dumps(row['tools']['by_tool'], ensure_ascii=False, sort_keys=True)}`；raw receipts={row['tools']['receipt_occurrences']}；duplicates={row['tools']['duplicate_receipt_occurrences']}；latency_ms={_coverage(row['tools']['latency_ms'])}",
                  f"- Terminal reason：{_display(row['termination_reason'])}；errors=`{json.dumps(row['errors'], ensure_ascii=False, sort_keys=True)}`；tool errors=`{json.dumps(row['tools']['error_types'], ensure_ascii=False, sort_keys=True)}`", "",
                  "| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |",
                  "|---|---:|---:|---:|---:|"]
        for label, usage in (("Agent recorded calls", row["agent_model"]["usage"]), ("Director recorded phases", row["director"]["recorded_phase_usage"])):
            lines.append("| " + label + " | " + " | ".join(_coverage(usage[field]) for field in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_ms")) + " |")
        lines += ["", f"- Grader model={row['grader']['model_id']}；saved telemetry=`{json.dumps(row['grader'], ensure_ascii=False, sort_keys=True)}`", ""]
    lines += ["## 解释边界", "", "图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。", ""]
    return "\n".join(lines)


def build_reports(evaluation_dir: Path, expected_count: int = 5, *, allow_operational_failures: bool = False) -> tuple[Mapping[str, Any], str, list[Mapping[str, Any]], Mapping[str, Any]]:
    manifest = _load_json_object(evaluation_dir / "run_manifest.json", name="run manifest")
    if manifest.get("collection_arm") != "agentgraph" or manifest.get("dataset_key") != "healthbench_professional":
        raise FailureDemoReportError("requires a HealthBench Professional AgentGraph-only collection")
    operational_mode = (
        allow_operational_failures
        and manifest.get("status") == "collection_completed_with_operational_failures"
        and manifest.get("collection_arm_completed") is False
        and bool(manifest.get("completed_at"))
    )
    if allow_operational_failures and expected_count != 5:
        raise FailureDemoReportError("operational-failure mode requires the original fixed five-task denominator")
    if manifest.get("collection_arm_completed") is not True and not operational_mode:
        raise FailureDemoReportError("collection is not complete; rerun after the fixed population finishes")
    selected = manifest.get("selected_task_ids")
    if not isinstance(selected, list) or any(not isinstance(task_id, str) or not task_id for task_id in selected):
        raise FailureDemoReportError("manifest selected_task_ids must be nonempty strings")
    if len(selected) != expected_count or len(set(selected)) != expected_count or manifest.get("sample_count") != expected_count:
        raise FailureDemoReportError("manifest fixed population differs from expected count")
    trajectories = _index(_load_jsonl(evaluation_dir / "agentgraph_trajectories.jsonl"), name="AgentGraph trajectories")
    if set(trajectories) - set(selected):
        raise FailureDemoReportError("trajectory contains tasks outside selected_task_ids")
    missing = [task_id for task_id in selected if task_id not in trajectories]
    if missing and not operational_mode:
        raise FailureDemoReportError("trajectory task population does not exactly match selected_task_ids")
    operational = _operational_evidence(evaluation_dir, manifest, missing, trajectories) if missing else {}
    ordered = [trajectories[task_id] for task_id in selected if task_id in trajectories]
    for trajectory in ordered:
        if not isinstance(trajectory.get("evaluation"), Mapping):
            raise FailureDemoReportError("trajectory lacks native evaluation receipt")
        metrics = _mapping(trajectory["evaluation"].get("metrics"))
        if trajectory["evaluation"].get("valid") is True and not all(_number(metrics.get(key)) for key in METRICS):
            raise FailureDemoReportError("valid evaluation lacks finite native raw/length-adjusted metrics")
        if not _objects(trajectory.get("turns")) or not _mapping(trajectory.get("task")).get("question"):
            raise FailureDemoReportError("trajectory lacks saved turns or original task input")
    observed_rows = [_row_summary(trajectory) for trajectory in ordered]
    by_id = {row["task_id"]: row for row in observed_rows}
    rows = [by_id[task_id] if task_id in by_id else _missing_summary(task_id, operational[task_id]) for task_id in selected]
    means = {}
    completed_means = {}
    for key in METRICS:
        values = [row["native_metrics"][key] for row in rows if _number(row["native_metrics"][key])]
        completed_means[key] = {"mean": sum(values) / len(values) if values else None, "observed_count": len(values), "selected_count": expected_count}
        means[key] = {"mean": sum(values) / expected_count if len(values) == expected_count else None,
                      "observed_count": len(values), "selected_count": expected_count, "scope": "strict full fixed population; missing is not zero"}
    summary = {
        "schema_version": SCHEMA, "collection_arm": "agentgraph", "sample_count": expected_count,
        "scope": "development observations; not a final benchmark estimate",
        "task_population_validation": "exact selected_task_ids; finished collection with current-run operational evidence" if missing else "exact selected_task_ids; completed collection",
        "completed_count": len(ordered), "operational_failure_count": len(missing),
        "missing_trajectory_task_ids": missing,
        "source_evaluation_dir": str(evaluation_dir),
        "valid_count": sum(row["valid"] is True for row in rows),
        "finish_count": sum(row["explicit_finish"] is True for row in rows),
        "terminal_failure_count": sum(row["terminal_failure"] is True for row in rows),
        "native_metric_means": means, "completed_only_native_metric_means": completed_means, "tasks": rows,
        "totals": {
            "scope": "persisted terminal trajectories only; not full-run billing",
            "covered_tasks": len(ordered), "selected_tasks": expected_count,
            "agent_recorded_model_calls": sum(row["agent_model"]["recorded_calls"] for row in observed_rows),
            "director_recorded_phase_calls": sum(row["director"]["recorded_phase_calls"] for row in observed_rows),
            "actual_tool_calls": sum(row["tools"]["actual_calls"] for row in observed_rows)
            if not missing and all(row["tools"]["actual_calls"] is not None for row in observed_rows) else None,
            "observed_actual_tool_calls": sum(row["tools"]["actual_calls"] for row in observed_rows)
            if observed_rows and all(row["tools"]["actual_calls"] is not None for row in observed_rows) else None,
            "grader_provider_error_receipts": sum(row["errors"]["grader_provider_errors"] for row in observed_rows),
            "runtime_failure_receipts": sum(row["errors"]["runtime_failure_receipts"] for row in observed_rows),
        },
        "report_generation": {"model_calls": 0, "tool_calls": 0, "grader_calls": 0, "training_performed": False},
    }
    private = "\n".join([
        "# Evaluator-private AgentGraph development demos", "",
        "> EVALUATOR-PRIVATE：不得提交 Git，不得进入 Director/Agent prompt 或训练输入。", "",
        "完整原始轨迹另存 agentgraph_development_evidence.jsonl；未记录证据不补造。", "",
        *[_private_demo(trajectories[task_id], by_id[task_id]) if task_id in trajectories
          else _private_operational_demo(operational[task_id]) for task_id in selected],
    ])
    return summary, private, ordered, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-count", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-operational-failures", action="store_true",
                        help="allow only a finished operational-failure collection; retain all five tasks with missing metrics as N/A")
    args = parser.parse_args(argv)
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    try:
        if args.expected_count < 1:
            raise FailureDemoReportError("expected count must be positive")
        if report_dir == evaluation_dir or evaluation_dir in report_dir.parents or report_dir in evaluation_dir.parents:
            raise FailureDemoReportError("public report directory must not overlap evaluation artifacts")
        summary, private, rows, manifest = build_reports(evaluation_dir, args.expected_count,
                                                       allow_operational_failures=args.allow_operational_failures)
        public = _public_report(summary)
        if not args.verify_only:
            private_dir = evaluation_dir / "evaluator_private"
            _atomic_text(private_dir / "agentgraph_development_evidence.jsonl", "\n".join(_canonical(row) for row in rows) + "\n")
            _atomic_text(private_dir / "agentgraph_development_run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            _atomic_text(private_dir / "agentgraph_development_demos.md", private)
            _atomic_text(report_dir / "agentgraph_development_report.md", public)
            _atomic_text(report_dir / "agentgraph_development_report.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"schema_version": SCHEMA, "verified_only": args.verify_only,
                          "sample_count": summary["sample_count"], "valid_count": summary["valid_count"],
                          "completed_count": summary["completed_count"], "operational_failure_count": summary["operational_failure_count"],
                          "finish_count": summary["finish_count"], "native_metric_means": summary["native_metric_means"],
                          "completed_only_native_metric_means": summary["completed_only_native_metric_means"],
                          "public_report_dir": str(report_dir), "private_report_dir": str(evaluation_dir / "evaluator_private")}, ensure_ascii=False, sort_keys=True))
    except (FailureDemoReportError, OSError, ValueError) as exc:
        print(f"offline report error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
