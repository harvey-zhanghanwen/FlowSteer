#!/usr/bin/env python3
"""Build an offline, same-task AIME 2026 runtime v1 -> v2 report.

The adapter reads only the two persisted ``paired_results.jsonl`` files.  It
does not import the evaluation runtime and cannot call a model, API, evaluator,
environment, Skill pipeline, or training path.  Alignment follows the strict
same-order ``task_id`` check used by ``report_joint_qa_progressive_experiment``;
the AIME-specific projections reuse fields emitted by
``evaluate_completion_benchmark_round._paired_rows``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DENOMINATOR = 30
EXPECTED_TASK_IDS = tuple(
    f"aime-2026/{index:02d}" for index in range(1, EXPECTED_DENOMINATOR + 1)
)
SCHEMA_VERSION = "flowsteer.aime2026.runtime_comparison.v1"

DEFAULT_V1_PAIRED = (
    PROJECT_ROOT
    / "artifacts"
    / "aime2026_unified_initial_v1"
    / "evaluation"
    / "paired_results.jsonl"
)
DEFAULT_V2_PAIRED = (
    PROJECT_ROOT
    / "artifacts"
    / "aime2026_runtime_v2"
    / "evaluation"
    / "paired_results.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "aime2026_runtime_comparison"

_OPERATIONAL_FAILURE_TYPES = {
    "direct_operational_or_evaluator_failure",
    "agentgraph_operational_or_evaluator_failure",
}
_TRANSITIONS = (
    "repaired",
    "regressed",
    "unchanged_correct",
    "unchanged_incorrect",
)


class AIMEComparisonReportError(RuntimeError):
    """Raised when persisted inputs cannot support a strict 30-task pair."""


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AIMEComparisonReportError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AIMEComparisonReportError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise AIMEComparisonReportError(
                    f"{path}:{line_number}: paired row must be an object"
                )
            rows.append(dict(value))
    return rows


def _binary_accuracy(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIMEComparisonReportError(f"{location}: accuracy must be 0 or 1")
    score = float(value)
    if score not in {0.0, 1.0}:
        raise AIMEComparisonReportError(f"{location}: accuracy must be 0 or 1")
    return score


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str, path: Path
) -> tuple[str, ...]:
    if len(rows) != EXPECTED_DENOMINATOR:
        raise AIMEComparisonReportError(
            f"{label} must retain exactly {EXPECTED_DENOMINATOR} paired rows; "
            f"found {len(rows)} in {path}"
        )
    task_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        location = f"{path}:{index}"
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise AIMEComparisonReportError(f"{location}: missing task_id")
        task_ids.append(task_id)
        if row.get("dataset_key") != "aime_2026":
            raise AIMEComparisonReportError(
                f"{location}: dataset_key must be 'aime_2026'"
            )
        if row.get("primary_metric") != "accuracy":
            raise AIMEComparisonReportError(
                f"{location}: primary_metric must be 'accuracy'"
            )
        graph = row.get("agentgraph")
        if not isinstance(graph, Mapping):
            raise AIMEComparisonReportError(
                f"{location}: missing agentgraph result"
            )
        if not isinstance(graph.get("available"), bool):
            raise AIMEComparisonReportError(
                f"{location}: agentgraph.available must be boolean"
            )
        if not isinstance(graph.get("valid"), bool):
            raise AIMEComparisonReportError(
                f"{location}: agentgraph.valid must be boolean"
            )
        _binary_accuracy(
            graph.get("accuracy"), location=f"{location}.agentgraph"
        )
        direct = row.get("direct")
        if not isinstance(direct, Mapping):
            raise AIMEComparisonReportError(f"{location}: missing direct result")
        if not isinstance(direct.get("available"), bool):
            raise AIMEComparisonReportError(
                f"{location}: direct.available must be boolean"
            )
        if not isinstance(direct.get("valid"), bool):
            raise AIMEComparisonReportError(
                f"{location}: direct.valid must be boolean"
            )
        _binary_accuracy(direct.get("accuracy"), location=f"{location}.direct")
    if len(set(task_ids)) != len(task_ids):
        raise AIMEComparisonReportError(f"{path}: duplicate task_id")
    observed = tuple(task_ids)
    if observed != EXPECTED_TASK_IDS:
        raise AIMEComparisonReportError(
            f"{label} task_id order is not the frozen AIME 2026 /01../30 order"
        )
    return observed


def _details(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = graph.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return {}
    details = evaluation.get("details")
    return details if isinstance(details, Mapping) else {}


def _diagnosis(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("wrong_demo_diagnosis")
    return value if isinstance(value, Mapping) else {}


def _first_failure(
    row: Mapping[str, Any], *, accuracy: float
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    if accuracy == 1.0:
        return None, None
    diagnosis = _diagnosis(row)
    raw_layer = diagnosis.get("failure_layer")
    layer = (
        raw_layer.strip()
        if isinstance(raw_layer, str) and raw_layer.strip()
        else "unavailable"
    )
    if not diagnosis:
        return layer, None
    return layer, {
        key: diagnosis.get(key)
        for key in (
            "diagnosis_scope",
            "failure_layer",
            "failure_boundary",
            "first_error_turn",
            "first_error_action",
            "first_error_agent_id",
            "error",
            "terminal_result",
        )
        if key in diagnosis
    }


def _condition_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    graph = row["agentgraph"]
    assert isinstance(graph, Mapping)
    accuracy = _binary_accuracy(
        graph.get("accuracy"), location=f"task {row.get('task_id')!r}"
    )
    available = graph.get("available") is True
    valid = graph.get("valid") is True
    failure_type = str(row.get("failure_type") or "unavailable")
    details = _details(graph)
    layer, first_failure = _first_failure(row, accuracy=accuracy)
    terminal_failure = bool(
        available
        and (
            graph.get("explicit_finish") is not True
            or failure_type == "agentgraph_terminal_failure"
        )
    )
    parsing_failure = bool(
        details.get("parsing_succeeded") is False
        or failure_type == "output_parsing_failure"
    )
    runtime_failure = bool(
        not available
        or failure_type in _OPERATIONAL_FAILURE_TYPES
        or layer in {"runtime", "tool"}
    )
    graph_failure = layer == "graph"
    failure_flags = {
        "terminal_failure": terminal_failure,
        "parsing_failure": parsing_failure,
        "runtime_failure": runtime_failure,
        "graph_failure": graph_failure,
    }
    if accuracy == 1.0:
        result_state = "correct"
    elif runtime_failure:
        result_state = "runtime_failure"
    elif terminal_failure:
        result_state = "terminal_failure"
    elif parsing_failure:
        result_state = "parsing_failure"
    elif graph_failure:
        result_state = "graph_failure"
    else:
        result_state = "incorrect"
    evaluation = graph.get("evaluation")
    evaluator_reason = (
        evaluation.get("reason") if isinstance(evaluation, Mapping) else None
    )
    return {
        "available": available,
        "valid": valid,
        "accuracy": accuracy,
        "result_state": result_state,
        "final_answer": graph.get("final_answer"),
        "failure_type": failure_type,
        "failure_flags": failure_flags,
        "explicit_finish": graph.get("explicit_finish") is True,
        "termination_reason": graph.get("termination_reason"),
        "trajectory_id": graph.get("trajectory_id"),
        "evaluator_reason": evaluator_reason,
        "parsing_succeeded": details.get("parsing_succeeded"),
        "parsing_failure_reason": details.get("parsing_failure_reason"),
        "first_observable_failure_layer": layer,
        "first_observable_failure": first_failure,
    }


def _transition(v1_accuracy: float, v2_accuracy: float) -> str:
    if v1_accuracy == 0.0 and v2_accuracy == 1.0:
        return "repaired"
    if v1_accuracy == 1.0 and v2_accuracy == 0.0:
        return "regressed"
    if v1_accuracy == 1.0:
        return "unchanged_correct"
    return "unchanged_incorrect"


def _condition_summary(tasks: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [task[key] for task in tasks]
    correct = sum(float(value["accuracy"]) == 1.0 for value in values)
    flag_counts = {
        flag: sum(value["failure_flags"][flag] is True for value in values)
        for flag in (
            "terminal_failure",
            "parsing_failure",
            "runtime_failure",
            "graph_failure",
        )
    }
    layers = Counter(
        str(value["first_observable_failure_layer"])
        for value in values
        if value["first_observable_failure_layer"] is not None
    )
    return {
        "denominator": EXPECTED_DENOMINATOR,
        "available_count": sum(value["available"] is True for value in values),
        "valid_count": sum(value["valid"] is True for value in values),
        "correct_count": correct,
        "strict_accuracy": correct / EXPECTED_DENOMINATOR,
        "failure_flag_counts_nonexclusive": flag_counts,
        "failure_type_distribution": dict(
            sorted(Counter(str(value["failure_type"]) for value in values).items())
        ),
        "first_observable_failure_layer_distribution": dict(sorted(layers.items())),
        "result_state_distribution": dict(
            sorted(Counter(str(value["result_state"]) for value in values).items())
        ),
    }


def _direct_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [row["direct"] for row in rows]
    correct = sum(float(value["accuracy"]) == 1.0 for value in values)
    return {
        "denominator": EXPECTED_DENOMINATOR,
        "available_count": sum(value["available"] is True for value in values),
        "valid_count": sum(value["valid"] is True for value in values),
        "correct_count": correct,
        "strict_accuracy": correct / EXPECTED_DENOMINATOR,
    }


def build_comparison_report(
    v1_paired_path: Path | str,
    v2_paired_path: Path | str,
) -> dict[str, Any]:
    """Build a deterministic comparison from two already-scored paired files."""

    v1_path = Path(v1_paired_path).expanduser().resolve()
    v2_path = Path(v2_paired_path).expanduser().resolve()
    v1_rows = _read_jsonl(v1_path, label="v1 paired results")
    v2_rows = _read_jsonl(v2_path, label="v2 paired results")
    v1_ids = _validate_rows(v1_rows, label="v1", path=v1_path)
    v2_ids = _validate_rows(v2_rows, label="v2", path=v2_path)
    if v1_ids != v2_ids:
        raise AIMEComparisonReportError(
            "v1 and v2 task_id order differs; paired delta is unavailable"
        )

    tasks: list[dict[str, Any]] = []
    for v1_row, v2_row in zip(v1_rows, v2_rows, strict=True):
        task_id = str(v1_row["task_id"])
        for field in ("question", "ground_truth"):
            if v1_row.get(field) != v2_row.get(field):
                raise AIMEComparisonReportError(
                    f"{task_id}: v1 and v2 {field} differ"
                )
        v1 = _condition_projection(v1_row)
        v2 = _condition_projection(v2_row)
        transition = _transition(float(v1["accuracy"]), float(v2["accuracy"]))
        tasks.append(
            {
                "task_id": task_id,
                "ground_truth": v1_row.get("ground_truth"),
                "transition": transition,
                "accuracy_delta_v2_minus_v1": (
                    float(v2["accuracy"]) - float(v1["accuracy"])
                ),
                "v1": v1,
                "v2": v2,
            }
        )

    v1_summary = _condition_summary(tasks, "v1")
    v2_summary = _condition_summary(tasks, "v2")
    direct_v1_summary = _direct_summary(v1_rows)
    direct_v2_summary = _direct_summary(v2_rows)
    transition_counts = Counter(str(task["transition"]) for task in tasks)
    transitions = {
        f"{name}_count": transition_counts.get(name, 0) for name in _TRANSITIONS
    }
    transitions.update(
        {
            f"{name}_task_ids": [
                str(task["task_id"])
                for task in tasks
                if task["transition"] == name
            ]
            for name in _TRANSITIONS
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_key": "aime_2026",
        "primary_metric": "accuracy",
        "comparison": "agentgraph_runtime_v2_minus_unified_initial_v1",
        "denominator": EXPECTED_DENOMINATOR,
        "paired_task_ids_verified": True,
        "task_ids": list(v1_ids),
        "sources": {
            "v1_paired_results": str(v1_path),
            "v2_paired_results": str(v2_path),
        },
        "controls": {
            "offline_persisted_artifacts_only": True,
            "model_calls": 0,
            "api_calls": 0,
            "evaluator_calls": 0,
            "training_steps": 0,
            "failure_flags_are_nonexclusive": True,
            "missing_or_invalid_agentgraph_rows_remain_in_denominator": True,
        },
        "source_mapping": {
            "alignment": (
                "scripts/report_joint_qa_progressive_experiment.py: strict "
                "same-order task_id comparison"
            ),
            "paired_row_schema": (
                "scripts/evaluate_completion_benchmark_round.py::_paired_rows"
            ),
            "aime_failure_type": (
                "scripts/evaluate_completion_benchmark_round.py::_failure_type"
            ),
            "first_observable_failure": (
                "scripts/evaluate_completion_benchmark_round.py::"
                "_aime_wrong_demo_diagnosis"
            ),
        },
        "v1": v1_summary,
        "v2": v2_summary,
        "direct_v1": direct_v1_summary,
        "direct_v2": direct_v2_summary,
        "v2_minus_v1": {
            "correct_count": (
                int(v2_summary["correct_count"])
                - int(v1_summary["correct_count"])
            ),
            "strict_accuracy": (
                float(v2_summary["strict_accuracy"])
                - float(v1_summary["strict_accuracy"])
            ),
            "strict_accuracy_percentage_points": 100.0
            * (
                float(v2_summary["strict_accuracy"])
                - float(v1_summary["strict_accuracy"])
            ),
        },
        "transitions": transitions,
        "tasks": tasks,
    }


def _cell(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "\\|").replace(
        "\n", " "
    )


def _flag_text(value: Mapping[str, Any]) -> str:
    names = [
        name.removesuffix("_failure")
        for name, enabled in value["failure_flags"].items()
        if enabled is True
    ]
    return ",".join(names) if names else "-"


def comparison_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Chinese report while preserving every task row."""

    v1 = report["v1"]
    v2 = report["v2"]
    delta = report["v2_minus_v1"]
    transitions = report["transitions"]
    direct_v1 = report["direct_v1"]
    direct_v2 = report["direct_v2"]
    lines = [
        "# AIME 2026 Runtime v1 → v2 离线配对报告",
        "",
        (
            f"固定同序题目：**{report['denominator']}**；"
            f"task_id 对齐：**{report['paired_task_ids_verified']}**。"
            "缺失、无效或 collection_failed 的 AgentGraph 行仍保留在固定分母中。"
        ),
        "",
        (
            f"v1：**{v1['correct_count']}/{v1['denominator']}**；"
            f"v2：**{v2['correct_count']}/{v2['denominator']}**；"
            f"v2-v1：**{delta['strict_accuracy_percentage_points']:+.2f} pp**。"
        ),
        (
            f"同批 Direct：v1 run **{direct_v1['correct_count']}/30**，"
            f"v2 run **{direct_v2['correct_count']}/30**；"
            "主配对结论使用 AgentGraph v1→v2，不把两次随机 Direct "
            "run 的差异归因于 runtime patch。"
        ),
        "",
        (
            f"修复 **{transitions['repaired_count']}**，"
            f"回退 **{transitions['regressed_count']}**，"
            f"持续正确 **{transitions['unchanged_correct_count']}**，"
            f"持续未正确 **{transitions['unchanged_incorrect_count']}**。"
        ),
        "",
        "故障计数为非互斥标记；首个 failure layer 仅投影已有 receipt，不推断隐藏因果。",
        "",
        "| Task | v1 | v2 | 变化 | v1 failure | v1 首层 | v2 failure | v2 首层 |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for task in report["tasks"]:
        before = task["v1"]
        after = task["v2"]
        lines.append(
            "| {task_id} | {v1} | {v2} | {transition} | {v1_flags} | "
            "{v1_layer} | {v2_flags} | {v2_layer} |".format(
                task_id=_cell(task["task_id"]),
                v1=int(float(before["accuracy"])),
                v2=int(float(after["accuracy"])),
                transition=_cell(task["transition"]),
                v1_flags=_cell(_flag_text(before)),
                v1_layer=_cell(before["first_observable_failure_layer"]),
                v2_flags=_cell(_flag_text(after)),
                v2_layer=_cell(after["first_observable_failure_layer"]),
            )
        )
    lines.extend(
        [
            "",
            "## 来源映射",
            "",
            *[
                f"- `{name}`：{value}"
                for name, value in report["source_mapping"].items()
            ],
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_report_outputs(
    report: Mapping[str, Any], output_dir: Path | str
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "comparison_report.json"
    markdown_path = output / "comparison_report_zh.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        comparison_report_markdown(report), encoding="utf-8"
    )
    return {"json": str(json_path), "chinese_markdown": str(markdown_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-paired", default=str(DEFAULT_V1_PAIRED))
    parser.add_argument("--v2-paired", default=str(DEFAULT_V2_PAIRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_comparison_report(args.v1_paired, args.v2_paired)
        artifacts = write_comparison_report_outputs(report, args.output_dir)
    except AIMEComparisonReportError as exc:
        parser.error(str(exc))
    print(json.dumps(artifacts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
