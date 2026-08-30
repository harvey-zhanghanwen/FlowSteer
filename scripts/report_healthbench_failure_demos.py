#!/usr/bin/env python3
"""Render receipt-backed HealthBench Professional failure-demo reports.

This is an offline reporting adapter.  It joins the completion runner's
``wrong_demos.jsonl`` and frozen trajectory receipts with the evaluator-only
HealthBench case store.  It never calls a model, Tool, environment, or grader
and never mutates policy, Skill, or training state.

The public report contains aggregate counts and redacted receipt summaries.
The evaluator-private report contains the full conversation, rubric target,
candidate outputs, and execution trace and must remain outside Git/model input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# Reuse the repository's existing receipt readers, communication/Tool
# extractors, and atomic Markdown writer instead of defining a second report
# artifact boundary.
from report_joint_qa_progressive_experiment import _atomic_text
from report_multidataset_stable_zero import (
    _communication_envelopes,
    _load_jsonl,
    _react_trace_entries,
    _tool_receipts,
)


SCHEMA_VERSION = "flowsteer.healthbench.failure-demo-report.v3"
DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "healthbench_professional_official_v1"
    / "evaluation"
)
DEFAULT_PRIVATE_CASES = (
    PROJECT_ROOT
    / "data"
    / "healthbench_professional_official_v1"
    / "private_cases.jsonl"
)
DEFAULT_PUBLIC_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "healthbench_professional_official_v1"
    / "failure_taxonomy_report_zh.md"
)
DEFAULT_PRIVATE_REPORT = (
    DEFAULT_EVALUATION_DIR
    / "evaluator_private"
    / "failure_taxonomy_private_zh.md"
)
DEFAULT_PRIVATE_MANIFEST = (
    DEFAULT_EVALUATION_DIR
    / "evaluator_private"
    / "failure_taxonomy_manifest.json"
)

CATEGORY_ORDER = (
    "retrieval_tool_failure",
    "agent_runtime_failure",
    "terminal_output_extraction_failure",
    "evaluator_operational_failure",
    "rubric_response_quality",
    "terminal_response_length_adjustment",
    "finished_graph_relation_anomaly",
    "finished_director_action_parsing_anomaly",
    "terminal_max_rounds",
)

CATEGORY_LABELS = {
    "retrieval_tool_failure": "Retrieval / Tool execution",
    "agent_runtime_failure": "Agent runtime / provider execution",
    "terminal_output_extraction_failure": "Terminal output extraction",
    "evaluator_operational_failure": "Evaluator / grader operational",
    "rubric_response_quality": "Rubric / response quality",
    "terminal_response_length_adjustment": "Terminal response length adjustment",
    "finished_graph_relation_anomaly": "已 FINISH 的 Canvas / graph / relation edit anomaly",
    "finished_director_action_parsing_anomaly": "已 FINISH 的 Director action parsing / recovery anomaly",
    "terminal_max_rounds": "Terminal / max_rounds",
}

NOT_SEPARATELY_OBSERVABLE = (
    (
        "Agent communication semantic use",
        "N/A",
        "receipt 能证明 artifact 经过 relation 传输，但不能证明下游模型在语义上正确使用，因此不伪报因果失败数。",
    ),
    (
        "隐藏 reasoning step",
        "N/A",
        "rubric receipt 能确认终局 response-quality shortfall，不能反推未记录的内部推理步骤。",
    ),
    (
        "独立 Verifier Agent",
        "N/A",
        "统一 search space 未强制 Verifier role，不能把 rubric miss 追溯为不存在的固定验证节点故障。",
    ),
    (
        "QA answer canonicalization",
        "N/A",
        "HealthBench Professional 使用 rubric-level grading，不使用 EM/F1 或 QA answer canonicalization。",
    ),
)


class FailureDemoReportError(RuntimeError):
    """The frozen evidence cannot support a complete failure-demo report."""


def _index(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            task = row.get("task")
            task_id = task.get("task_id") if isinstance(task, Mapping) else None
        if not isinstance(task_id, str) or not task_id:
            raise FailureDemoReportError(f"{name} row is missing task_id")
        if task_id in result:
            raise FailureDemoReportError(f"{name} contains duplicate task_id: {task_id}")
        result[task_id] = dict(row)
    return result


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FailureDemoReportError(f"missing {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FailureDemoReportError(f"{name} must be a JSON object: {path}")
    return dict(value)


def _validate_task_populations(
    *,
    paired: Mapping[str, Mapping[str, Any]],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    private_cases: Mapping[str, Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    historical_failures: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate exact task identity, including declared Direct strict-zero terminals.

    A Direct response may be absent only when the completion runner froze that task
    as a strict-zero ReAct terminal failure.  The manifest declaration, paired zero
    record, and append-only terminal receipt must all agree.  This does not relax
    population checks for ordinary missing predictions.
    """

    paired_ids = set(paired)
    trajectory_ids = set(trajectories)
    private_ids = set(private_cases)
    direct_ids = set(direct)
    if trajectory_ids != paired_ids or private_ids != paired_ids:
        raise FailureDemoReportError(
            "paired/trajectory/private task populations do not match"
        )
    extra_direct_ids = direct_ids - paired_ids
    if extra_direct_ids:
        raise FailureDemoReportError(
            "Direct predictions contain tasks outside the paired population: "
            + ", ".join(sorted(extra_direct_ids))
        )

    missing_direct_ids = paired_ids - direct_ids
    if not missing_direct_ids:
        return {}

    manifest_sample_count = run_manifest.get("sample_count")
    direct_progress = run_manifest.get("direct_progress")
    direct_progress = direct_progress if isinstance(direct_progress, Mapping) else {}
    completed_count = direct_progress.get("completed")
    strict_zero_count = direct_progress.get("strict_zero_terminal_failures")
    frozen_count = direct_progress.get("frozen_react_terminal_failures")
    if (
        manifest_sample_count != len(paired_ids)
        or completed_count != len(direct_ids)
        or not isinstance(strict_zero_count, int)
        or isinstance(strict_zero_count, bool)
        or not isinstance(frozen_count, int)
        or isinstance(frozen_count, bool)
        or strict_zero_count != len(missing_direct_ids)
        or frozen_count != len(missing_direct_ids)
    ):
        raise FailureDemoReportError(
            "Direct population and manifest progress do not exactly declare the "
            "missing responses as frozen strict-zero ReAct terminal failures"
        )

    receipts_by_task: dict[str, list[dict[str, Any]]] = {
        task_id: [] for task_id in missing_direct_ids
    }
    for failure in historical_failures:
        task_id = failure.get("task_id")
        if task_id not in receipts_by_task:
            continue
        condition = str(failure.get("condition") or "").casefold()
        stage = str(failure.get("stage") or "").casefold()
        error = str(failure.get("error") or "").casefold()
        if (
            condition.startswith("direct")
            and stage == "generation_or_evaluator"
            and "react agent" in error
            and "exhausted" in error
            and "without a valid completion" in error
        ):
            receipts_by_task[str(task_id)].append(dict(failure))

    validated: dict[str, dict[str, Any]] = {}
    for task_id in sorted(missing_direct_ids):
        paired_direct = paired[task_id].get("direct")
        paired_direct = paired_direct if isinstance(paired_direct, Mapping) else {}
        score = paired_direct.get("overall_score")
        adjusted_score = paired_direct.get("overall_score_length_adjusted")
        if not (
            paired_direct.get("available") is False
            and paired_direct.get("valid") is False
            and score == 0
            and adjusted_score == 0
        ):
            raise FailureDemoReportError(
                f"{task_id} missing Direct response lacks a paired strict-zero record"
            )
        receipts = receipts_by_task[task_id]
        if not receipts:
            raise FailureDemoReportError(
                f"{task_id} missing Direct response lacks a frozen ReAct terminal receipt"
            )
        validated[task_id] = receipts[-1]
    return validated


def _diagnosis(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("wrong_demo_diagnosis")
    return value if isinstance(value, Mapping) else {}


def primary_category(row: Mapping[str, Any]) -> str:
    """Return one mutually exclusive category, with terminal failure first."""

    failure_type = row.get("failure_type")
    diagnosis = _diagnosis(row)
    layer = diagnosis.get("failure_layer")
    if failure_type == "agentgraph_terminal_failure":
        return "terminal_max_rounds"
    if layer == "tool":
        return "retrieval_tool_failure"
    if layer == "runtime":
        return "agent_runtime_failure"
    if layer == "output_extraction":
        return "terminal_output_extraction_failure"
    if layer == "evaluator":
        return "evaluator_operational_failure"
    if layer == "rubric_evaluation":
        return "rubric_response_quality"
    if layer == "terminal_response_length_adjustment":
        return "terminal_response_length_adjustment"
    if layer == "graph":
        return "finished_graph_relation_anomaly"
    if layer == "director":
        return "finished_director_action_parsing_anomaly"
    raise FailureDemoReportError(
        f"unsupported HealthBench failure layer: {layer!r} / {failure_type!r}"
    )


def _percentage(count: int, denominator: int) -> float:
    return 100.0 * count / denominator if denominator else 0.0


def _grader_provider_error_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        evaluation = row.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        details = evaluation.get("details")
        details = details if isinstance(details, Mapping) else {}
        telemetry = details.get("grader_telemetry")
        telemetry = telemetry if isinstance(telemetry, Mapping) else {}
        errors = telemetry.get("provider_errors")
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
            count += len(errors)
    return count


def _rejected_turn_count(trajectory: Mapping[str, Any]) -> int:
    count = 0
    turns = trajectory.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return count
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        feedback = str(turn.get("canvas_feedback") or "").casefold()
        if any(
            marker in feedback
            for marker in (
                "rejected",
                "[invalid]",
                "invalid action:",
                "cannot finish:",
                "parse_error",
                "schema_invalid",
            )
        ):
            count += 1
    return count


def _delta(row: Mapping[str, Any]) -> float:
    value = row.get("delta_overall_score_length_adjusted")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rubric_subcategory(row: Mapping[str, Any]) -> str:
    summary = _diagnosis(row).get("rubric_receipt_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    unmet = int(summary.get("unmet_positive_rubric_count") or 0) > 0
    negative = int(summary.get("triggered_negative_rubric_count") or 0) > 0
    if unmet and negative:
        return "unmet_positive_and_triggered_negative"
    if unmet:
        return "unmet_positive_only"
    if negative:
        return "triggered_negative_only"
    return "rubric_shortfall_without_counted_flag"


def select_representatives(
    categorized: Mapping[str, Sequence[Mapping[str, Any]]],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Select deterministic severe examples without task-ID hard-coding."""

    selected: dict[str, list[dict[str, Any]]] = {}
    for category in (
        "retrieval_tool_failure",
        "agent_runtime_failure",
        "terminal_output_extraction_failure",
        "evaluator_operational_failure",
    ):
        rows = list(categorized.get(category, ()))
        selected[category] = [dict(min(rows, key=_delta))] if rows else []

    rubric_rows = list(categorized.get("rubric_response_quality", ()))
    selected["rubric_response_quality"] = []
    for subcategory in (
        "unmet_positive_only",
        "triggered_negative_only",
        "unmet_positive_and_triggered_negative",
        "rubric_shortfall_without_counted_flag",
    ):
        candidates = [
            row for row in rubric_rows if _rubric_subcategory(row) == subcategory
        ]
        if candidates:
            selected["rubric_response_quality"].append(
                dict(min(candidates, key=_delta))
            )

    length_rows = list(categorized.get("terminal_response_length_adjustment", ()))
    selected["terminal_response_length_adjustment"] = (
        [dict(min(length_rows, key=_delta))] if length_rows else []
    )

    graph_rows = list(categorized.get("finished_graph_relation_anomaly", ()))
    selected["finished_graph_relation_anomaly"] = []
    for rejected_action in ("set_relation", "modify_agent", "add_agent", "unparsed"):
        candidates = [
            row
            for row in graph_rows
            if str(_diagnosis(row).get("first_error_action") or "unparsed")
            == rejected_action
        ]
        if candidates:
            selected["finished_graph_relation_anomaly"].append(
                dict(min(candidates, key=_delta))
            )

    director_rows = list(
        categorized.get("finished_director_action_parsing_anomaly", ())
    )
    selected["finished_director_action_parsing_anomaly"] = (
        [dict(min(director_rows, key=_delta))] if director_rows else []
    )

    terminal = list(categorized.get("terminal_max_rounds", ()))
    terminal_selected: list[dict[str, Any]] = []
    terminal_layers = sorted(
        {
            str(_diagnosis(row).get("failure_layer") or "unknown")
            for row in terminal
        }
    )
    for layer in terminal_layers:
        candidates = [row for row in terminal if _diagnosis(row).get("failure_layer") == layer]
        if not candidates:
            continue
        terminal_selected.append(
            dict(
                max(
                    candidates,
                    key=lambda row: (
                        _rejected_turn_count(trajectories[str(row["task_id"])]),
                        -_delta(row),
                    ),
                )
            )
        )
    selected["terminal_max_rounds"] = terminal_selected
    return selected


def _json_details(title: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"<details><summary>{escape(title)}</summary>\n\n"
        f"<pre><code>{escape(payload)}</code></pre>\n\n"
        "</details>"
    )


def _execution_view(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = execution.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    request = metadata.get("request")
    response = metadata.get("response")
    request_agent = request.get("agent") if isinstance(request, Mapping) else None
    request_agent = request_agent if isinstance(request_agent, Mapping) else {}
    return {
        "execution_id": execution.get("execution_id"),
        "agent_id": execution.get("agent_id"),
        "model_id": execution.get("model_id"),
        "provider": execution.get("provider"),
        "graph_revision": execution.get("graph_revision"),
        "error_type": execution.get("error_type"),
        "input_tokens": execution.get("input_tokens"),
        "output_tokens": execution.get("output_tokens"),
        "latency_ms": execution.get("latency_ms"),
        "execution_mode": request_agent.get("execution_mode"),
        "allowed_tools": request_agent.get("allowed_tools"),
        "request": dict(request) if isinstance(request, Mapping) else request,
        "output": execution.get("output"),
        "response_receipt": (
            dict(response) if isinstance(response, Mapping) else response
        ),
    }


def _execution_mode_call_counts(
    trajectories: Sequence[Mapping[str, Any]],
) -> Mapping[str, int]:
    """Count executed Agent calls by execution_mode, never by role label."""

    counts: Counter[str] = Counter()
    for trajectory in trajectories:
        turns = trajectory.get("turns")
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            executions = turn.get("executions")
            if not isinstance(executions, Sequence) or isinstance(
                executions, (str, bytes)
            ):
                continue
            for execution in executions:
                if not isinstance(execution, Mapping):
                    continue
                metadata = execution.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                request = metadata.get("request")
                request = request if isinstance(request, Mapping) else {}
                agent = request.get("agent")
                agent = agent if isinstance(agent, Mapping) else {}
                mode = agent.get("execution_mode")
                counts[str(mode) if isinstance(mode, str) and mode else "unspecified"] += 1
    return dict(sorted(counts.items()))


def _turn_view(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = turn.get("runtime_summary")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    return {
        "round_index": turn.get("round_index"),
        "director_input": turn.get("prompt"),
        "director_raw_output": turn.get("policy_response"),
        "parsed_action": turn.get("action"),
        "canvas_feedback": turn.get("canvas_feedback"),
        "graph_revision": turn.get("graph_revision"),
        "graph_snapshot": turn.get("graph_snapshot"),
        "reconstructed_context": turn.get("reconstructed_context"),
        "director_receipt": {
            "attempt_count": turn.get("director_attempt_count"),
            "generation_seed": turn.get("director_generation_seed"),
            "latency_ms": turn.get("director_latency_ms"),
            "request_id": turn.get("director_request_id"),
            "receipt_verified": turn.get("receipt_verified"),
        },
        "runtime_summary": dict(runtime),
        "agent_executions": [
            _execution_view(execution)
            for execution in turn.get("executions", ())
            if isinstance(execution, Mapping)
        ],
    }


def _metric_view(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "valid": value.get("valid"),
        "overall_score": value.get("overall_score"),
        "overall_score_length_adjusted": value.get(
            "overall_score_length_adjusted"
        ),
        "final_answer": value.get("final_answer"),
        "explicit_finish": value.get("explicit_finish"),
        "termination_reason": value.get("termination_reason"),
    }


def _causal_boundary(row: Mapping[str, Any]) -> str:
    category = primary_category(row)
    diagnosis = _diagnosis(row)
    if category == "retrieval_tool_failure":
        return (
            f"round {diagnosis.get('first_error_turn')} 的 Agent Tool Action–Observation "
            f"receipt 首次记录 `{diagnosis.get('error')}`；这是可观察的 retrieval/Tool "
            "execution boundary，不把 ReAct 当作 Agent role，也不据此虚构隐藏医学推理错误。"
        )
    if category == "agent_runtime_failure":
        return (
            f"round {diagnosis.get('first_error_turn')} 的 Agent runtime/provider receipt "
            f"首次记录 `{diagnosis.get('error')}`；后续影响只按保存的 receipt span 描述。"
        )
    if category == "terminal_output_extraction_failure":
        return (
            "AgentGraph 已到达终局，但 evaluator receipt 明确记录 final-response "
            f"output extraction 失败：`{diagnosis.get('error')}`；该类不等同于 rubric shortfall。"
        )
    if category == "evaluator_operational_failure":
        return (
            "终局 response 已产生，但 HealthBench grader/evaluator receipt 无效："
            f"`{diagnosis.get('error')}`；没有有效 rubric-level score，不能改报为模型答案错误。"
        )
    if category == "rubric_response_quality":
        return (
            "此前没有保存到 runtime/Canvas fault；首个可观察短板位于 Output "
            "Agent 的完整终局响应，rubric receipt 显示 positive criterion 未满足或 "
            "negative criterion 被触发。现有 receipt 不能把语义缺失唯一归因到更早 Agent。"
        )
    if category == "terminal_response_length_adjustment":
        return (
            "raw rubric score 已满，首个可观察损失发生在 Professional character-length "
            "adjustment；不是 Formatter、canonicalization 或 terminal parsing failure。"
        )
    if category == "finished_graph_relation_anomaly":
        return (
            f"round {diagnosis.get('first_error_turn')} 的 "
            f"{diagnosis.get('first_error_action')} Canvas edit 首次被拒；workflow 后续恢复并 "
            "FINISH，因此该 rejection 是最早 fault receipt，但不自动证明它导致最终 rubric 回退。"
        )
    if category == "finished_director_action_parsing_anomaly":
        return (
            f"round {diagnosis.get('first_error_turn')} 的 Director action 首次无法解析；"
            "后续 Canvas 恢复并 FINISH，语义影响只作为 causal hypothesis。"
        )
    return (
        f"首个可观察 fault 位于 round {diagnosis.get('first_error_turn')} 的 "
        f"{diagnosis.get('first_error_action')}；后续持续 graph editing，最终到达 "
        "max_rounds 而没有合法 FINISH。terminal failure 为确定结果。"
    )


def _category_rows(
    wrong_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result = {category: [] for category in CATEGORY_ORDER}
    for row in wrong_rows:
        result[primary_category(row)].append(dict(row))
    if sum(len(rows) for rows in result.values()) != len(wrong_rows):
        raise FailureDemoReportError("failure taxonomy is not exhaustive")
    return result


def _subcategory_counts(
    categorized: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Mapping[str, int]]:
    rubric = categorized["rubric_response_quality"]
    rubric_counts = Counter()
    for row in rubric:
        rubric_counts[_rubric_subcategory(row)] += 1

    graph_counts = Counter(
        str(_diagnosis(row).get("first_error_action") or "unparsed")
        for row in categorized["finished_graph_relation_anomaly"]
    )
    terminal_counts = Counter(
        str(_diagnosis(row).get("failure_layer") or "unknown")
        for row in categorized["terminal_max_rounds"]
    )
    operational_counts = {
        category: dict(
            sorted(
                Counter(
                    str(_diagnosis(row).get("error") or "unspecified")
                    for row in categorized[category]
                ).items()
            )
        )
        for category in (
            "retrieval_tool_failure",
            "agent_runtime_failure",
            "terminal_output_extraction_failure",
            "evaluator_operational_failure",
        )
    }
    return {
        "first_observable_operational_errors": operational_counts,
        "rubric_response_quality": dict(sorted(rubric_counts.items())),
        "finished_graph_relation_anomaly": dict(sorted(graph_counts.items())),
        "terminal_max_rounds_first_observable_layer": dict(
            sorted(terminal_counts.items())
        ),
    }


def _representative_subcategory(row: Mapping[str, Any]) -> str:
    category = primary_category(row)
    if category in {
        "retrieval_tool_failure",
        "agent_runtime_failure",
        "terminal_output_extraction_failure",
        "evaluator_operational_failure",
    }:
        return str(_diagnosis(row).get("error") or "unspecified")
    if category == "rubric_response_quality":
        return _rubric_subcategory(row)
    if category == "finished_graph_relation_anomaly":
        return str(_diagnosis(row).get("first_error_action") or "unparsed")
    if category == "terminal_max_rounds":
        return str(_diagnosis(row).get("failure_layer") or "unknown")
    return category


def _public_report(
    *,
    categorized: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    sample_count: int,
    historical_failures: Sequence[Mapping[str, Any]],
    grader_provider_retries: Mapping[str, int],
    execution_mode_call_counts: Mapping[str, int],
    direct_record_count: int | None = None,
    direct_strict_zero_terminal_failures: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> str:
    wrong_count = sum(len(rows) for rows in categorized.values())
    strict_zero_failures = dict(direct_strict_zero_terminal_failures or {})
    lines = [
        "# HealthBench Professional Failure Taxonomy 与脱敏 Wrong Demo",
        "",
        "本报告完全来自冻结的 paired-result、trajectory 与 evaluator receipt；没有重新调用模型、Tool 或 grader，也没有训练。完整 conversation、rubric、physician response、candidate output 和逐轮 prompt 只保存在 evaluator-private 本地报告，不进入 Git 或模型输入。Wrong Demo 的固定定义是 `agentgraph.overall_score_length_adjusted < 1.0`，不是 Direct-vs-AgentGraph 高低。",
        "",
        "## 任务分母完整性",
        "",
        f"- paired / trajectory / evaluator-private task：`{sample_count}`。",
        f"- Direct 完成响应记录：`{direct_record_count if direct_record_count is not None else sample_count}`。",
        f"- Direct 冻结 strict-zero ReAct terminal failure：`{len(strict_zero_failures)}`。这些任务没有伪造 response；只有 manifest、paired strict-zero 和 append-only terminal receipt 三者一致时才允许作为缺失响应计入固定分母。",
        "",
        "## 互斥分类",
        "",
        f"Terminal failure 优先分类，因此 terminal case 不会再次计入 graph/director 类。百分比分别以 {wrong_count} 个 wrong demo 和 {sample_count} 个 public-test task 为分母。",
        "",
        "| 类别 | 数量 | Wrong Demo 占比 | 全体占比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in CATEGORY_ORDER:
        count = len(categorized[category])
        lines.append(
            f"| {CATEGORY_LABELS[category]} | {count} | "
            f"{_percentage(count, wrong_count):.4f}% | "
            f"{_percentage(count, sample_count):.4f}% |"
        )
    lines.extend(
        [
            f"| **合计** | **{wrong_count}** | **100.0000%** | **{_percentage(wrong_count, sample_count):.4f}%** |",
            "",
            "## 子类分布",
            "",
        ]
    )
    subcategories = _subcategory_counts(categorized)
    lines.extend(
        [
            f"- Tool/runtime/output-extraction/evaluator 首个可观察 error：`{json.dumps(subcategories['first_observable_operational_errors'], ensure_ascii=False, sort_keys=True)}`。",
            f"- Rubric / response quality：`{json.dumps(subcategories['rubric_response_quality'], ensure_ascii=False, sort_keys=True)}`。",
            f"- 已 FINISH 的 graph/relation anomaly 首个 rejected action：`{json.dumps(subcategories['finished_graph_relation_anomaly'], ensure_ascii=False, sort_keys=True)}`。",
            f"- Terminal/max_rounds 首个可观察 layer：`{json.dumps(subcategories['terminal_max_rounds_first_observable_layer'], ensure_ascii=False, sort_keys=True)}`。",
            "",
            "## Agent execution_mode",
            "",
            f"- 实际 Agent call 的 `execution_mode` 分布：`{json.dumps(execution_mode_call_counts, ensure_ascii=False, sort_keys=True)}`。",
            "- `react` 只表示 Agent 的执行模式（StructuredAction → Tool Observation → completion）；它不是 Agent role，也不作为 role family 统计。",
            "",
            "## 不可从 receipt 单独识别的类别",
            "",
            "| 类别 | 数量 | 依据 |",
            "| --- | ---: | --- |",
        ]
    )
    for label, count, evidence in NOT_SEPARATELY_OBSERVABLE:
        lines.append(f"| {label} | {count} | {evidence} |")
    lines.extend(
        [
            "",
            "上述 N/A 类别没有可支持独立因果计数的 receipt，因此不生成 demo。",
        ]
    )

    stage_counts = Counter(str(row.get("stage") or "unknown") for row in historical_failures)
    strict_zero_count = len(strict_zero_failures)
    recovered_attempt_count = max(0, len(historical_failures) - strict_zero_count)
    lines.extend(
        [
            "",
            "## 历史失败 attempts 与终局状态",
            "",
            f"append-only `collection_failures.jsonl` 保留 {len(historical_failures)} 个历史 attempt：`{json.dumps(dict(sorted(stage_counts.items())), ensure_ascii=False)}`。其中 {recovered_attempt_count} 个 attempt 已由最终有效 receipt 取代；另有 {strict_zero_count} 个 manifest-declared Direct ReAct terminal failure 没有 response，按冻结协议严格计 0，不能称为已恢复。",
            f"valid terminal grader receipt 内另记录已恢复的 provider error attempts：Direct={grader_provider_retries.get('direct', 0)}，AgentGraph={grader_provider_retries.get('agentgraph', 0)}。它们是物理调用 retry，不是最终 task failure。",
            "",
            "## 各类代表样本（脱敏）",
            "",
        ]
    )
    for category in CATEGORY_ORDER:
        rows = list(selected.get(category, ()))
        if not rows:
            lines.extend(
                [f"### {CATEGORY_LABELS[category]}", "", "- 数量：0；不虚构 demo。", ""]
            )
            continue
        lines.extend([f"### {CATEGORY_LABELS[category]}", ""])
        for row in rows:
            diagnosis = _diagnosis(row)
            graph = row.get("agentgraph")
            graph = graph if isinstance(graph, Mapping) else {}
            final_graph = graph.get("final_graph")
            final_graph = final_graph if isinstance(final_graph, Mapping) else {}
            nodes = final_graph.get("nodes")
            relations = final_graph.get("relations")
            lines.extend(
                [
                    f"- task_id：`{row.get('task_id')}`",
                    f"- subcategory：`{_representative_subcategory(row)}`",
                    f"- Direct / AgentGraph length-adjusted：`{row.get('direct', {}).get('overall_score_length_adjusted')} / {graph.get('overall_score_length_adjusted')}`",
                    f"- first observable layer：`{diagnosis.get('failure_layer')}`；turn=`{diagnosis.get('first_error_turn')}`；action=`{diagnosis.get('first_error_action')}`；agent=`{diagnosis.get('first_error_agent_id')}`",
                    f"- terminal：`{graph.get('termination_reason')}`；explicit_finish=`{graph.get('explicit_finish')}`；nodes=`{len(nodes) if isinstance(nodes, Sequence) else 0}`；relations=`{len(relations) if isinstance(relations, Sequence) else 0}`",
                    f"- failure boundary：{_causal_boundary(row)}",
                    "",
                ]
            )
    lines.extend(
        [
            "## 解释边界",
            "",
            "- HealthBench Professional 没有单一 reference answer；正式 target 是 signed `rubric_items`。`physician_response` 是 evaluator-only reference material，不直接参与该 public scorer。",
            "- `first observable failure` 来自实际 receipt。若 workflow 后续恢复，不能仅凭时间顺序宣称该 fault 唯一导致终局分数变化。",
            "- Rubric failure 是 evaluator-visible response-quality shortfall；没有保存证据时，不把它凭空细分成某个隐藏 reasoning 或 verification step。",
            "- 完整 private demo 报告是 evaluator-side diagnostic artifact，禁止拼入 Director/Agent prompt 或作为训练样本。",
            "",
        ]
    )
    return "\n".join(lines)


def _private_demo(
    row: Mapping[str, Any],
    *,
    private_case: Mapping[str, Any],
    direct_record: Mapping[str, Any] | None,
    direct_terminal_failure: Mapping[str, Any] | None,
    trajectory: Mapping[str, Any],
) -> str:
    task_id = str(row["task_id"])
    category = primary_category(row)
    graph = row.get("agentgraph")
    graph = graph if isinstance(graph, Mapping) else {}
    direct = row.get("direct")
    direct = direct if isinstance(direct, Mapping) else {}
    turns = trajectory.get("turns")
    turns = (
        [turn for turn in turns if isinstance(turn, Mapping)]
        if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes))
        else []
    )
    action_sequence = [
        turn.get("action", {}).get("action")
        for turn in turns
        if isinstance(turn.get("action"), Mapping)
    ]
    final_graph = graph.get("terminal_canvas_graph") or graph.get("final_graph")
    direct_record = direct_record if isinstance(direct_record, Mapping) else {}
    direct_execution = direct_record.get("execution")
    direct_execution = (
        _execution_view(direct_execution)
        if isinstance(direct_execution, Mapping)
        else direct_execution
    )
    sections = [
        f"## {CATEGORY_LABELS[category]} / {_representative_subcategory(row)} — `{task_id}`",
        "",
        f"- schema_version：`{SCHEMA_VERSION}`",
        f"- trajectory_id：`{trajectory.get('trajectory_id')}`",
        f"- condition_id：`{trajectory.get('condition_id')}`",
        f"- action sequence：`{' → '.join(str(value) for value in action_sequence)}`",
        f"- Direct metrics：`{json.dumps(_metric_view(direct), ensure_ascii=False, sort_keys=True)}`",
        f"- AgentGraph metrics：`{json.dumps(_metric_view(graph), ensure_ascii=False, sort_keys=True)}`",
        f"- 首个可观察 failure boundary：{_causal_boundary(row)}",
        "",
        _json_details("完整 conversation（模型可见输入）", private_case.get("prompt")),
        "",
        _json_details(
            "正式 evaluator target：signed rubric_items（仅 evaluator 可见）",
            private_case.get("rubric_items"),
        ),
        "",
        _json_details(
            "physician completion（非 ground truth、非本轮评分输入，仅供人工对照）",
            private_case.get("physician_response"),
        ),
        "",
        _json_details("Direct 完整执行输入/输出与 provider receipt", direct_execution),
        "",
        _json_details(
            "Direct strict-zero terminal failure receipt（若无 Direct response）",
            direct_terminal_failure,
        ),
        "",
        _json_details("AgentGraph terminal / evaluated graph", final_graph),
        "",
        _json_details("Output Agent 最终 inbox", graph.get("output_agent_inbox")),
        "",
        _json_details("首个 failure diagnosis 与后续传播 receipt", _diagnosis(row)),
        "",
    ]
    for turn in turns:
        sections.extend(
            [
                _json_details(
                    f"Director / Canvas / Agent 完整 turn {turn.get('round_index')}",
                    _turn_view(turn),
                ),
                "",
            ]
        )
    communication = _communication_envelopes(trajectory)
    react_entries = _react_trace_entries(trajectory)
    tool_receipts = _tool_receipts(trajectory)
    sections.extend(
        [
            _json_details("Agent-to-Agent CommunicationEnvelope", communication),
            "",
            _json_details(
                "Agent execution_mode=react StructuredAction–Observation receipt",
                react_entries,
            ),
            "",
            _json_details("Tool receipts", tool_receipts),
            "",
            (
                "执行模式结论：ReAct 仅为 Agent `execution_mode`，不是 role；"
                f"保存到的 Tool receipt 数量={len(tool_receipts)}，"
                f"StructuredAction–Observation 数量={len(react_entries)}。"
            ),
            "",
            _json_details("AgentGraph terminal evaluator receipt", trajectory.get("evaluation")),
            "",
            _json_details("Direct evaluator receipt", direct_record.get("evaluation")),
            "",
        ]
    )
    return "\n".join(sections)


def _private_report(
    *,
    categorized: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    private_cases: Mapping[str, Mapping[str, Any]],
    direct_records: Mapping[str, Mapping[str, Any]],
    direct_strict_zero_terminal_failures: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    sample_count: int,
) -> str:
    wrong_count = sum(len(rows) for rows in categorized.values())
    lines = [
        "# HealthBench Professional Evaluator-Private Failure Demo Report",
        "",
        "> **EVALUATOR-PRIVATE：不得提交 Git，不得进入 Director/Agent prompt，不得作为训练输入。** 本文件由冻结 evidence 离线生成，没有模型、Tool、grader 或训练调用。",
        "",
        f"- public-test population：{sample_count}",
        f"- Direct response records：{len(direct_records)}",
        f"- Direct frozen strict-zero terminal failures：{len(direct_strict_zero_terminal_failures)}",
        f"- wrong demo：{wrong_count}",
        "- 正式目标：signed rubric；没有单一 reference answer。",
        "- physician_response：physician completion；非 ground truth、非本轮评分输入，仅供人工对照。",
        "",
    ]
    for category in CATEGORY_ORDER:
        rows = list(selected.get(category, ()))
        if not rows:
            lines.extend(
                [f"## {CATEGORY_LABELS[category]}", "", "该类为 0；不虚构 demo。", ""]
            )
            continue
        for row in rows:
            task_id = str(row["task_id"])
            missing = [
                name
                for name, index in (
                    ("private case", private_cases),
                    ("trajectory", trajectories),
                )
                if task_id not in index
            ]
            if (
                task_id not in direct_records
                and task_id not in direct_strict_zero_terminal_failures
            ):
                missing.append("direct record or declared strict-zero terminal receipt")
            if missing:
                raise FailureDemoReportError(
                    f"{task_id} missing private evidence: {', '.join(missing)}"
                )
            lines.extend(
                [
                    _private_demo(
                        row,
                        private_case=private_cases[task_id],
                        direct_record=direct_records.get(task_id),
                        direct_terminal_failure=direct_strict_zero_terminal_failures.get(
                            task_id
                        ),
                        trajectory=trajectories[task_id],
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def _direct_predictions_path(evaluation_dir: Path) -> Path:
    """Resolve legacy reasoning and v3 ReAct Direct artifact names."""

    candidates = (
        evaluation_dir / "direct_predictions.jsonl",
        evaluation_dir / "single_agent_react_predictions.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FailureDemoReportError(
        "missing Direct predictions artifact; expected one of: "
        + ", ".join(path.name for path in candidates)
    )


def build_reports(
    *,
    evaluation_dir: Path,
    private_cases_path: Path,
) -> tuple[str, str, Mapping[str, Any]]:
    wrong_rows = _load_jsonl(evaluation_dir / "wrong_demos.jsonl")
    paired_rows = _load_jsonl(evaluation_dir / "paired_results.jsonl")
    direct_rows = _load_jsonl(_direct_predictions_path(evaluation_dir))
    trajectory_rows = _load_jsonl(evaluation_dir / "agentgraph_trajectories.jsonl")
    private_rows = _load_jsonl(private_cases_path)
    historical_failures = _load_jsonl(evaluation_dir / "collection_failures.jsonl")
    run_manifest = _load_json_object(
        evaluation_dir / "run_manifest.json", name="run manifest"
    )

    paired = _index(paired_rows, name="paired results")
    direct = _index(direct_rows, name="direct predictions")
    trajectories = _index(trajectory_rows, name="agentgraph trajectories")
    private_cases = _index(private_rows, name="private cases")
    direct_strict_zero_terminal_failures = _validate_task_populations(
        paired=paired,
        direct=direct,
        trajectories=trajectories,
        private_cases=private_cases,
        run_manifest=run_manifest,
        historical_failures=historical_failures,
    )
    for row in wrong_rows:
        task_id = str(row.get("task_id"))
        if (
            task_id not in paired
            or task_id not in trajectories
            or task_id not in private_cases
            or (
                task_id not in direct
                and task_id not in direct_strict_zero_terminal_failures
            )
        ):
            raise FailureDemoReportError(f"wrong demo lacks frozen evidence: {task_id}")

    categorized = _category_rows(wrong_rows)
    selected = select_representatives(categorized, trajectories)
    sample_count = len(paired)
    grader_provider_retries = {
        "direct": _grader_provider_error_count(direct_rows),
        "agentgraph": _grader_provider_error_count(trajectory_rows),
    }
    execution_mode_call_counts = _execution_mode_call_counts(trajectory_rows)
    public = _public_report(
        categorized=categorized,
        selected=selected,
        sample_count=sample_count,
        historical_failures=historical_failures,
        grader_provider_retries=grader_provider_retries,
        execution_mode_call_counts=execution_mode_call_counts,
        direct_record_count=len(direct),
        direct_strict_zero_terminal_failures=direct_strict_zero_terminal_failures,
    )
    private = _private_report(
        categorized=categorized,
        selected=selected,
        private_cases=private_cases,
        direct_records=direct,
        direct_strict_zero_terminal_failures=direct_strict_zero_terminal_failures,
        trajectories=trajectories,
        sample_count=sample_count,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sample_count": sample_count,
        "direct_response_record_count": len(direct),
        "direct_strict_zero_terminal_failure_count": len(
            direct_strict_zero_terminal_failures
        ),
        "direct_strict_zero_terminal_failure_task_ids": sorted(
            direct_strict_zero_terminal_failures
        ),
        "task_population_validation": "exact_with_declared_direct_strict_zero_terminals",
        "wrong_demo_count": len(wrong_rows),
        "category_counts": {
            category: len(categorized[category]) for category in CATEGORY_ORDER
        },
        "subcategory_counts": _subcategory_counts(categorized),
        "representative_task_ids": {
            category: [str(row["task_id"]) for row in selected[category]]
            for category in CATEGORY_ORDER
        },
        "historical_attempt_count": len(historical_failures),
        "recovered_grader_provider_error_attempts": grader_provider_retries,
        "agent_execution_mode_call_counts": execution_mode_call_counts,
        "model_calls": 0,
        "tool_calls": 0,
        "grader_calls": 0,
        "training_performed": False,
    }
    return public, private, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--private-cases", type=Path, default=DEFAULT_PRIVATE_CASES)
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC_REPORT)
    parser.add_argument("--private-output", type=Path, default=DEFAULT_PRIVATE_REPORT)
    parser.add_argument(
        "--private-manifest-output",
        type=Path,
        default=DEFAULT_PRIVATE_MANIFEST,
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    public, private, manifest = build_reports(
        evaluation_dir=args.evaluation_dir.expanduser().resolve(),
        private_cases_path=args.private_cases.expanduser().resolve(),
    )
    if not args.verify_only:
        _atomic_text(args.public_output.expanduser().resolve(), public)
        _atomic_text(args.private_output.expanduser().resolve(), private)
        _atomic_text(
            args.private_manifest_output.expanduser().resolve(),
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
