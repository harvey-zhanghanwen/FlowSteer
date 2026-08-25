from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "report_aime2026_runtime_comparison.py"
_SPEC = importlib.util.spec_from_file_location(
    "report_aime2026_runtime_comparison", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    index: int,
    *,
    accuracy: int = 1,
    available: bool = True,
    valid: bool = True,
    failure_type: str = "both_correct",
    explicit_finish: bool = True,
    termination_reason: str = "finish",
    failure_layer: str | None = None,
    error: str | None = None,
    parsing_succeeded: bool | None = True,
) -> dict:
    task_id = f"aime-2026/{index:02d}"
    diagnosis = None
    if accuracy == 0:
        diagnosis = {
            "diagnosis_scope": "first_observable_failure",
            "failure_layer": failure_layer or "agent",
            "first_error_turn": 2,
            "first_error_action": "finish",
            "first_error_agent_id": "solver",
            "error": error or "incorrect_terminal_integer",
            "terminal_result": termination_reason,
        }
    evaluation = None
    if available:
        evaluation = {
            "valid": valid,
            "reason": (
                "evaluated"
                if explicit_finish
                else "not_evaluated_without_explicit_finish"
            ),
            "metrics": {"accuracy": accuracy} if valid else {},
            "details": {
                "parsing_succeeded": parsing_succeeded,
                "parsing_failure_reason": (
                    error if parsing_succeeded is False else None
                ),
            },
        }
    return {
        "schema_version": "flowsteer.completion_benchmark.paired_result.v1",
        "dataset_key": "aime_2026",
        "task_id": task_id,
        "question": f"question-{index}",
        "ground_truth": str(100 + index),
        "primary_metric": "accuracy",
        "direct": {
            "available": True,
            "valid": True,
            "accuracy": 1,
        },
        "agentgraph": {
            "available": available,
            "valid": valid,
            "accuracy": accuracy,
            "final_answer": str(100 + index) if accuracy else "999",
            "evaluation": evaluation,
            "explicit_finish": explicit_finish,
            "termination_reason": termination_reason,
            "trajectory_id": f"trajectory-{index}" if available else None,
        },
        "failure_type": failure_type,
        "wrong_demo_diagnosis": diagnosis,
    }


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    v1_rows = [_row(index) for index in range(1, 31)]
    v2_rows = [_row(index) for index in range(1, 31)]

    # Parsing failure repaired by v2.
    v1_rows[0] = _row(
        1,
        accuracy=0,
        failure_type="output_parsing_failure",
        failure_layer="output_extraction",
        error="integer_conversion_failed",
        parsing_succeeded=False,
    )
    # A v1 correct answer regresses to a graph-observed terminal failure.
    v2_rows[1] = _row(
        2,
        accuracy=0,
        valid=False,
        failure_type="agentgraph_terminal_failure",
        explicit_finish=False,
        termination_reason="max_rounds",
        failure_layer="graph",
        error="canvas_action_rejected",
        parsing_succeeded=None,
    )
    # Graph failure remains wrong and becomes a runtime failure.
    v1_rows[2] = _row(
        3,
        accuracy=0,
        failure_type="both_incorrect",
        failure_layer="graph",
        error="canvas_action_rejected",
    )
    v2_rows[2] = _row(
        3,
        accuracy=0,
        available=False,
        valid=False,
        failure_type="agentgraph_operational_or_evaluator_failure",
        explicit_finish=False,
        termination_reason="collection_failed",
        failure_layer="runtime",
        error="trajectory_missing",
        parsing_succeeded=None,
    )
    # An ordinary wrong v1 answer becomes a parsing failure in v2.
    v1_rows[3] = _row(4, accuracy=0, failure_type="both_incorrect")
    v2_rows[3] = _row(
        4,
        accuracy=0,
        failure_type="output_parsing_failure",
        failure_layer="output_extraction",
        error="ambiguous_integer_markers",
        parsing_succeeded=False,
    )
    # Preserve the real v1 denominator boundary: task 13 has no trajectory.
    v1_rows[12] = _row(
        13,
        accuracy=0,
        available=False,
        valid=False,
        failure_type="agentgraph_operational_or_evaluator_failure",
        explicit_finish=False,
        termination_reason="collection_failed",
        failure_layer="runtime",
        error="trajectory_missing",
        parsing_succeeded=None,
    )

    v1_path = tmp_path / "v1.jsonl"
    v2_path = tmp_path / "v2.jsonl"
    _write_jsonl(v1_path, v1_rows)
    _write_jsonl(v2_path, v2_rows)
    return v1_path, v2_path


def test_comparison_retains_all_30_and_projects_transitions_and_failures(tmp_path):
    v1_path, v2_path = _fixtures(tmp_path)

    report = _MODULE.build_comparison_report(v1_path, v2_path)

    assert report["denominator"] == 30
    assert report["paired_task_ids_verified"] is True
    assert len(report["tasks"]) == 30
    assert report["v1"]["denominator"] == 30
    assert report["v1"]["available_count"] == 29
    assert report["v1"]["correct_count"] == 26
    assert report["v2"]["correct_count"] == 27
    assert report["direct_v1"]["correct_count"] == 30
    assert report["direct_v2"]["correct_count"] == 30
    assert report["v2_minus_v1"]["correct_count"] == 1
    assert report["v2_minus_v1"]["strict_accuracy"] == pytest.approx(1 / 30)
    assert report["transitions"]["repaired_count"] == 2
    assert report["transitions"]["repaired_task_ids"] == [
        "aime-2026/01",
        "aime-2026/13",
    ]
    assert report["transitions"]["regressed_task_ids"] == ["aime-2026/02"]
    assert report["transitions"]["unchanged_incorrect_count"] == 2
    assert report["transitions"]["unchanged_correct_count"] == 25

    task13 = report["tasks"][12]
    assert task13["transition"] == "repaired"
    assert task13["v1"]["available"] is False
    assert task13["v1"]["failure_flags"]["runtime_failure"] is True
    assert task13["v1"]["first_observable_failure_layer"] == "runtime"
    assert task13["v1"]["first_observable_failure"]["error"] == (
        "trajectory_missing"
    )

    assert report["v1"]["failure_flag_counts_nonexclusive"] == {
        "terminal_failure": 0,
        "parsing_failure": 1,
        "runtime_failure": 1,
        "graph_failure": 1,
    }
    assert report["v2"]["failure_flag_counts_nonexclusive"] == {
        "terminal_failure": 1,
        "parsing_failure": 1,
        "runtime_failure": 1,
        "graph_failure": 1,
    }
    assert report["v2"]["first_observable_failure_layer_distribution"] == {
        "graph": 1,
        "output_extraction": 1,
        "runtime": 1,
    }


def test_comparison_fails_closed_if_fixed_denominator_or_ids_change(tmp_path):
    v1_path, v2_path = _fixtures(tmp_path)
    v2_rows = [json.loads(line) for line in v2_path.read_text().splitlines()]
    _write_jsonl(v2_path, v2_rows[:-1])

    with pytest.raises(_MODULE.AIMEComparisonReportError, match="exactly 30"):
        _MODULE.build_comparison_report(v1_path, v2_path)

    _write_jsonl(v2_path, v2_rows)
    v2_rows[0]["task_id"], v2_rows[1]["task_id"] = (
        v2_rows[1]["task_id"],
        v2_rows[0]["task_id"],
    )
    _write_jsonl(v2_path, v2_rows)
    with pytest.raises(_MODULE.AIMEComparisonReportError, match="task_id"):
        _MODULE.build_comparison_report(v1_path, v2_path)


def test_comparison_fails_closed_on_ground_truth_mismatch(tmp_path):
    v1_path, v2_path = _fixtures(tmp_path)
    v2_rows = [json.loads(line) for line in v2_path.read_text().splitlines()]
    v2_rows[4]["ground_truth"] = "different"
    _write_jsonl(v2_path, v2_rows)

    with pytest.raises(_MODULE.AIMEComparisonReportError, match="ground_truth"):
        _MODULE.build_comparison_report(v1_path, v2_path)


def test_writer_emits_json_and_chinese_per_task_report(tmp_path):
    v1_path, v2_path = _fixtures(tmp_path)
    report = _MODULE.build_comparison_report(v1_path, v2_path)

    artifacts = _MODULE.write_comparison_report_outputs(
        report, tmp_path / "report"
    )

    json_path = Path(artifacts["json"])
    markdown_path = Path(artifacts["chinese_markdown"])
    assert json_path.is_file()
    assert markdown_path.is_file()
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert persisted["denominator"] == 30
    assert persisted["controls"]["model_calls"] == 0
    assert "aime-2026/13" in markdown
    assert "repaired" in markdown
    assert "runtime" in markdown
    assert "来源映射" in markdown
    assert "同批 Direct" in markdown
