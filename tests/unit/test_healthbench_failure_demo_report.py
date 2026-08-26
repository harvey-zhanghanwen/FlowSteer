from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "report_healthbench_failure_demos.py"
_SPEC = importlib.util.spec_from_file_location(
    "report_healthbench_failure_demos", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(
    task_id: str,
    *,
    layer: str,
    failure_type: str = "direct_higher_overall_score_length_adjusted",
    terminal: str = "finish",
    delta: float = -0.5,
) -> dict:
    return {
        "task_id": task_id,
        "failure_type": failure_type,
        "delta_overall_score_length_adjusted": delta,
        "wrong_demo_diagnosis": {
            "failure_layer": layer,
            "terminal_result": terminal,
            "first_error_turn": 2,
            "first_error_action": "set_relation",
        },
        "direct": {"overall_score_length_adjusted": 0.5},
        "agentgraph": {
            "overall_score_length_adjusted": 0.0,
            "explicit_finish": terminal == "finish",
            "termination_reason": terminal,
            "final_graph": {"nodes": [], "relations": []},
        },
    }


def test_terminal_failure_takes_precedence_over_first_observable_graph_layer():
    row = _row(
        "healthbench-professional:terminal",
        layer="graph",
        failure_type="agentgraph_terminal_failure",
        terminal="max_rounds",
    )
    assert _MODULE.primary_category(row) == "terminal_max_rounds"


def test_healthbench_failure_categories_are_mutually_exclusive():
    rows = [
        _row("rubric", layer="rubric_evaluation"),
        _row("length", layer="terminal_response_length_adjustment"),
        _row("graph", layer="graph"),
        _row("director", layer="director"),
        _row(
            "terminal",
            layer="graph",
            failure_type="agentgraph_terminal_failure",
            terminal="max_rounds",
        ),
    ]
    categorized = _MODULE._category_rows(rows)
    assert {key: len(value) for key, value in categorized.items()} == {
        "rubric_response_quality": 1,
        "terminal_response_length_adjustment": 1,
        "finished_graph_relation_anomaly": 1,
        "finished_director_action_parsing_anomaly": 1,
        "terminal_max_rounds": 1,
    }


def test_public_report_contains_no_private_case_payload():
    rows = [_row("rubric", layer="rubric_evaluation")]
    categorized = _MODULE._category_rows(rows)
    selected = {
        category: list(values) for category, values in categorized.items()
    }
    report = _MODULE._public_report(
        categorized=categorized,
        selected=selected,
        sample_count=1,
        historical_failures=[],
        grader_provider_retries={"direct": 0, "agentgraph": 0},
    )
    assert "rubric_items" in report  # protocol name only
    assert "physician_response" in report  # field name only
    assert "secret conversation" not in report
    assert "secret criterion" not in report
    assert "secret candidate response" not in report


def test_execution_view_preserves_agent_input_output_and_communication_receipt():
    value = _MODULE._execution_view(
        {
            "execution_id": "execution-1",
            "agent_id": "agent-a",
            "model_id": "qwen3.5-9b-local",
            "provider": "local",
            "output": "candidate",
            "metadata": {
                "request": {
                    "rendered_messages": [{"role": "user", "content": "input"}],
                    "upstream": [{"source_agent_id": "agent-b"}],
                },
                "response": {"attempt_count": 1},
            },
        }
    )
    assert value["request"]["rendered_messages"][0]["content"] == "input"
    assert value["request"]["upstream"][0]["source_agent_id"] == "agent-b"
    assert value["output"] == "candidate"
    assert value["response_receipt"]["attempt_count"] == 1


def test_representatives_cover_each_nonzero_rubric_subcategory():
    rows = []
    for task_id, unmet, negative in (
        ("positive", 1, 0),
        ("negative", 0, 1),
        ("both", 1, 1),
    ):
        row = _row(task_id, layer="rubric_evaluation")
        row["wrong_demo_diagnosis"]["rubric_receipt_summary"] = {
            "unmet_positive_rubric_count": unmet,
            "triggered_negative_rubric_count": negative,
        }
        rows.append(row)
    categorized = _MODULE._category_rows(rows)
    selected = _MODULE.select_representatives(categorized, {})
    assert [
        _MODULE._rubric_subcategory(row)
        for row in selected["rubric_response_quality"]
    ] == [
        "unmet_positive_only",
        "triggered_negative_only",
        "unmet_positive_and_triggered_negative",
    ]
