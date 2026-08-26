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


SCHEMA_VERSION = "flowsteer.healthbench.failure-demo-report.v1"
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
    "rubric_response_quality",
    "terminal_response_length_adjustment",
    "finished_graph_relation_anomaly",
    "finished_director_action_parsing_anomaly",
    "terminal_max_rounds",
)

CATEGORY_LABELS = {
    "rubric_response_quality": "Rubric / response quality",
    "terminal_response_length_adjustment": "Terminal response length adjustment",
    "finished_graph_relation_anomaly": "已 FINISH 的 Canvas / graph / relation edit anomaly",
    "finished_director_action_parsing_anomaly": "已 FINISH 的 Director action parsing / recovery anomaly",
    "terminal_max_rounds": "Terminal / max_rounds",
}

ZERO_OR_NOT_APPLICABLE = (
    (
        "Retrieval / Tool",
        "0",
        "官方 public base condition 为 no-Tool；全部 final node 的 allowed_tools=[]，无 Tool 或 ReAct Action–Observation receipt。",
    ),
    (
        "Agent communication transport/runtime",
        "0",
        "没有 Agent runtime failed turn、execution error 或 message transport failure；relation construction anomaly 单列，不与 transport failure 混算。",
    ),
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
        "Formatter / terminal output parsing",
        "0",
        "require_format_agent=false，terminal parsing failure=0；长度校正不是格式解析失败。",
    ),
    (
        "Final evaluator / canonicalization",
        "0",
        "最终 operational/evaluator failure=0；HealthBench 使用 rubric score，不使用 QA canonicalization。",
    ),
    (
        "Final provider / collection",
        "0",
        "历史 provider/collection attempts 已恢复并单列，不能计作最终 task failure。",
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
    for layer in ("graph", "director"):
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
        "request": dict(request) if isinstance(request, Mapping) else request,
        "output": execution.get("output"),
        "response_receipt": (
            dict(response) if isinstance(response, Mapping) else response
        ),
    }


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
        f"{diagnosis.get('first_error_action')}；后续持续 graph editing，最终 20 turns "
        "内没有合法 FINISH，formal evaluator 未调用。terminal failure 为确定结果。"
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
    return {
        "rubric_response_quality": dict(sorted(rubric_counts.items())),
        "finished_graph_relation_anomaly": dict(sorted(graph_counts.items())),
        "terminal_max_rounds_first_observable_layer": dict(
            sorted(terminal_counts.items())
        ),
    }


def _representative_subcategory(row: Mapping[str, Any]) -> str:
    category = primary_category(row)
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
) -> str:
    wrong_count = sum(len(rows) for rows in categorized.values())
    lines = [
        "# HealthBench Professional Failure Taxonomy 与脱敏 Wrong Demo",
        "",
        "本报告完全来自冻结的 paired-result、trajectory 与 evaluator receipt；没有重新调用模型、Tool 或 grader，也没有训练。完整 conversation、rubric、physician response、candidate output 和逐轮 prompt 只保存在 evaluator-private 本地报告，不进入 Git 或模型输入。Wrong Demo 的固定定义是 `agentgraph.overall_score_length_adjusted < 1.0`，不是 Direct-vs-AgentGraph 高低。",
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
            f"- Rubric / response quality：`{json.dumps(subcategories['rubric_response_quality'], ensure_ascii=False, sort_keys=True)}`。",
            f"- 已 FINISH 的 graph/relation anomaly 首个 rejected action：`{json.dumps(subcategories['finished_graph_relation_anomaly'], ensure_ascii=False, sort_keys=True)}`。",
            f"- Terminal/max_rounds 首个可观察 layer：`{json.dumps(subcategories['terminal_max_rounds_first_observable_layer'], ensure_ascii=False, sort_keys=True)}`。",
            "",
            "## 适用性为 0 或 N/A 的类别",
            "",
            "| 类别 | 数量 | 依据 |",
            "| --- | ---: | --- |",
        ]
    )
    for label, count, evidence in ZERO_OR_NOT_APPLICABLE:
        lines.append(f"| {label} | {count} | {evidence} |")
    lines.extend(
        [
            "",
            "上述 0/N/A 类别没有对应真实 failure receipt，因此不生成 demo。",
        ]
    )

    stage_counts = Counter(str(row.get("stage") or "unknown") for row in historical_failures)
    lines.extend(
        [
            "",
            "## 历史已恢复 attempts",
            "",
            f"append-only `collection_failures.jsonl` 保留 {len(historical_failures)} 个历史 attempt：`{json.dumps(dict(sorted(stage_counts.items())), ensure_ascii=False)}`。这些 attempt 已被最终 receipt 取代，最终 provider/collection/evaluator operational failure 为 0。",
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
    direct_record: Mapping[str, Any],
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
            _json_details("Executor ReAct StructuredAction–Observation receipt", react_entries),
            "",
            _json_details("Tool receipts", tool_receipts),
            "",
            (
                "Tool/ReAct 结论：本 condition 为 `no_tool`；保存到的 Tool receipt "
                f"数量={len(tool_receipts)}，ReAct Action–Observation 数量={len(react_entries)}。"
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
                    ("direct record", direct_records),
                    ("trajectory", trajectories),
                )
                if task_id not in index
            ]
            if missing:
                raise FailureDemoReportError(
                    f"{task_id} missing private evidence: {', '.join(missing)}"
                )
            lines.extend(
                [
                    _private_demo(
                        row,
                        private_case=private_cases[task_id],
                        direct_record=direct_records[task_id],
                        trajectory=trajectories[task_id],
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def build_reports(
    *,
    evaluation_dir: Path,
    private_cases_path: Path,
) -> tuple[str, str, Mapping[str, Any]]:
    wrong_rows = _load_jsonl(evaluation_dir / "wrong_demos.jsonl")
    paired_rows = _load_jsonl(evaluation_dir / "paired_results.jsonl")
    direct_rows = _load_jsonl(evaluation_dir / "direct_predictions.jsonl")
    trajectory_rows = _load_jsonl(evaluation_dir / "agentgraph_trajectories.jsonl")
    private_rows = _load_jsonl(private_cases_path)
    historical_failures = _load_jsonl(evaluation_dir / "collection_failures.jsonl")

    paired = _index(paired_rows, name="paired results")
    direct = _index(direct_rows, name="direct predictions")
    trajectories = _index(trajectory_rows, name="agentgraph trajectories")
    private_cases = _index(private_rows, name="private cases")
    if not (len(paired) == len(direct) == len(trajectories) == len(private_cases)):
        raise FailureDemoReportError(
            "paired/direct/trajectory/private task populations do not match"
        )
    for row in wrong_rows:
        task_id = str(row.get("task_id"))
        if task_id not in paired or task_id not in trajectories:
            raise FailureDemoReportError(f"wrong demo lacks frozen evidence: {task_id}")

    categorized = _category_rows(wrong_rows)
    selected = select_representatives(categorized, trajectories)
    sample_count = len(paired)
    grader_provider_retries = {
        "direct": _grader_provider_error_count(direct_rows),
        "agentgraph": _grader_provider_error_count(trajectory_rows),
    }
    public = _public_report(
        categorized=categorized,
        selected=selected,
        sample_count=sample_count,
        historical_failures=historical_failures,
        grader_provider_retries=grader_provider_retries,
    )
    private = _private_report(
        categorized=categorized,
        selected=selected,
        private_cases=private_cases,
        direct_records=direct,
        trajectories=trajectories,
        sample_count=sample_count,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sample_count": sample_count,
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
