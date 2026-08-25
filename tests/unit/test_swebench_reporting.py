from __future__ import annotations

from src.interactive.swebench_reporting import (
    aggregate_swebench_receipts,
    classify_swebench_evaluation,
    diagnose_swebench_wrong_demo,
)


def _receipt(action: str, value: dict | None = None, *, error_type: str | None = None):
    return {
        "tool_id": "swebench_repository",
        "tool_version": "skillflow.repository-tools.v2",
        "request": {"action": action, "arguments": {}},
        "result": None if error_type else {"value": value or {}, "completed": True},
        "error_type": error_type,
    }


def _execution(
    *,
    receipts=(),
    trace=(),
    model_calls=(),
    continued_receipts: int = 0,
    continued_trace: int = 0,
    error_type: str | None = None,
    agent_id: str = "agent_1",
):
    return {
        "agent_id": agent_id,
        "error_type": error_type,
        "metadata": {
            "response": {
                "tool_receipts": list(receipts),
                "react_trace": list(trace),
                "model_calls": list(model_calls),
                "continued_tool_receipt_count": continued_receipts,
                "continued_action_history_count": continued_trace,
            }
        },
    }


def _evaluation(status: str, *, valid: bool = True):
    resolved = status == "resolved"
    return {
        "valid": valid,
        "metrics": {"resolved": float(resolved)} if valid else {},
        "reason": "evaluated" if valid else "swebench_harness_failed",
        "details": {
            "resolved": resolved,
            "harness_details": status,
            "proxy_metric_used": False,
        },
    }


def _row(task_id: str, *, direct_status: str, graph_status: str):
    return {
        "task_id": task_id,
        "direct": {
            "available": True,
            "valid": True,
            "resolved": float(direct_status == "resolved"),
            "evaluation": _evaluation(direct_status),
        },
        "agentgraph": {
            "available": True,
            "valid": True,
            "resolved": float(graph_status == "resolved"),
            "evaluation": _evaluation(graph_status),
            "explicit_finish": True,
            "final_answer": "diff --git a/a.py b/a.py\n",
        },
    }


def _trajectory(
    task_id: str,
    turns,
    *,
    final_answer="diff --git a/a.py b/a.py\n",
    finish=True,
):
    return {
        "task": {"task_id": task_id},
        "turns": list(turns),
        "final_answer": final_answer,
        "explicit_finish": finish,
        "evaluation": _evaluation("unresolved"),
    }


def _accepted_turn(executions=(), *, round_index=0):
    return {
        "round_index": round_index,
        "action": {"action": "add_subgraph"},
        "canvas_feedback": "accepted add_subgraph",
        "executions": list(executions),
        "runtime_summary": {},
    }


def test_aggregate_uses_structured_receipts_and_deduplicates_continuation_prefix():
    direct_trace = [
        {
            "turn": 1,
            "observation_status": "parse_error",
            "public_error_code": "ValueError",
        },
        {
            "turn": 2,
            "observation_status": "schema_invalid",
            "public_error_code": "tool_arguments_schema_invalid",
        },
        {
            "turn": 3,
            "observation_status": "budget_exhausted",
            "public_error_code": "tool_call_budget_exhausted",
        },
        {
            "turn": 4,
            "structured_action": {"kind": "tool", "name": "view_file"},
            "observation": {
                "observation_status": "tool_error",
                "error_type": "TimeoutError",
            },
        },
    ]
    direct_receipts = [
        _receipt("search_code", {"ok": True}),
        _receipt("view_file", {"ok": True}),
        _receipt("exact_edit", {"ok": True, "changed": True}),
        _receipt("bash", {"ok": True}),
        _receipt("run_tests", {"ok": True, "passed": True, "timed_out": False}),
        _receipt("run_tests", {"ok": True, "passed": False, "timed_out": False}),
        _receipt("run_tests", {"ok": True, "passed": False, "timed_out": True}),
        _receipt("run_tests", {"ok": False, "error": "cannot start"}),
        _receipt("diff", error_type="RuntimeError"),
    ]
    row = _row("swe:1", direct_status="patch_apply_failed", graph_status="unresolved")
    row["direct"]["execution"] = _execution(
        receipts=direct_receipts,
        trace=direct_trace,
    )

    first_receipt = _receipt("search_code", {"ok": True})
    first_trace = {
        "turn": 1,
        "structured_action": {"kind": "tool", "name": "search_code"},
        "observation": {"observation_status": "success", "result": {"ok": True}},
    }
    failing_test_trace = {
        "turn": 4,
        "structured_action": {"kind": "tool", "name": "run_tests"},
        "observation": {
            "observation_status": "success",
            "result": {"ok": True, "passed": False, "timed_out": False},
        },
    }
    second_receipts = [
        first_receipt,
        _receipt("view_file", {"ok": True}),
        _receipt("apply_patch", {"ok": True, "changed": True}),
        _receipt("run_tests", {"ok": True, "passed": False, "timed_out": False}),
    ]
    trajectory = _trajectory(
        "swe:1",
        [
            _accepted_turn(
                [
                    _execution(receipts=[first_receipt], trace=[first_trace]),
                    _execution(
                        receipts=second_receipts,
                        trace=[first_trace, failing_test_trace],
                        continued_receipts=1,
                        continued_trace=1,
                    ),
                ]
            )
        ],
    )

    report = aggregate_swebench_receipts([row], [trajectory])

    direct = report["arms"]["direct"]
    assert direct["raw_tool_action_counts"] == {
        "bash": 1,
        "diff": 1,
        "exact_edit": 1,
        "run_tests": 4,
        "search_code": 1,
        "view_file": 1,
    }
    assert direct["tool_action_group_counts"] == {
        "search": 1,
        "view": 1,
        "edit": 1,
        "test": 4,
        "command": 1,
    }
    assert direct["invalid_structured_action_count"] == 2
    assert direct["invalid_tool_action_code_counts"] == {
        "tool_arguments_schema_invalid": 1
    }
    assert direct["structured_action_budget_exhausted_code_counts"] == {
        "tool_call_budget_exhausted": 1
    }
    assert direct["tool_dispatch_error_type_counts"] == {"TimeoutError": 1}
    assert direct["tool_backend_error_type_counts"] == {"RuntimeError": 1}
    assert direct["tool_result_failure_action_counts"] == {"run_tests": 1}
    assert direct["local_test_receipts"] == {
        "observed_count": 3,
        "passed_count": 1,
        "failed_count": 1,
        "timeout_count": 1,
        "backend_failure_count": 1,
    }
    assert direct["evaluation_outcome_counts"] == {"patch_apply_failure": 1}

    graph = report["arms"]["agentgraph"]
    assert graph["raw_tool_action_counts"] == {
        "apply_patch": 1,
        "run_tests": 1,
        "search_code": 1,
        "view_file": 1,
    }
    assert graph["tool_action_group_counts"] == {
        "search": 1,
        "view": 1,
        "edit": 1,
        "test": 1,
        "command": 0,
    }
    assert graph["local_test_receipts"]["failed_count"] == 1
    assert graph["evaluation_outcome_counts"] == {"test_failure": 1}
    assert report["receipt_policy"]["passed_string_parsing_used"] is False
    assert report["wrong_demo_diagnoses"] == [
        {
            "task_id": "swe:1",
            "diagnosis_scope": "first_observable_failure",
            "causal_attribution": False,
            "interpretation": (
                "earliest persisted failure receipt; not proof of causal attribution"
            ),
            "failure_layer": "tool_result",
            "failure_type": "local_test_failed",
            "turn_index": 0,
            "inner_turn": 4,
            "agent_id": "agent_1",
            "action": "run_tests",
            "receipt_source": "execution_receipt",
        }
    ]


def test_classify_swebench_evaluation_uses_official_receipt_fields_only():
    assert (
        classify_swebench_evaluation(_evaluation("resolved"))["outcome"]
        == "resolved"
    )
    assert (
        classify_swebench_evaluation(_evaluation("patch_apply_failed"))["outcome"]
        == "patch_apply_failure"
    )
    assert (
        classify_swebench_evaluation(_evaluation("test_timeout"))["outcome"]
        == "test_failure"
    )
    structured_environment_failure = {
        "valid": False,
        "metrics": {},
        "reason": "swebench_harness_failed",
        "details": {"phase": "docker", "classification": "permission_denied"},
    }
    assert classify_swebench_evaluation(structured_environment_failure) == {
        "outcome": "environment_failure",
        "failure_phase": "environment_failure",
        "resolved": None,
        "evaluator_reason": "swebench_harness_failed",
    }
    unresolved_without_status = {
        "valid": True,
        "metrics": {"resolved": 0.0},
        "details": {"resolved": False, "proxy_metric_used": False},
    }
    assert classify_swebench_evaluation(unresolved_without_status)["outcome"] == (
        "official_harness_unresolved"
    )


def test_wrong_demo_first_observable_layers_follow_persisted_execution_order():
    row = _row("swe:parse", direct_status="resolved", graph_status="unresolved")
    parse_trajectory = _trajectory(
        "swe:parse",
        [
            {
                "round_index": 0,
                "action": {},
                "canvas_feedback": "parse_error",
                "executions": [],
            }
        ],
    )
    assert diagnose_swebench_wrong_demo(
        row, trajectory=parse_trajectory
    )["failure_layer"] == "director_action_parsing"

    provider_trajectory = _trajectory(
        "swe:parse",
        [
            _accepted_turn(
                [
                    _execution(
                        model_calls=[
                            {
                                "turn": 1,
                                "request_status": "failed",
                                "error_type": "ProviderRequestError",
                            }
                        ],
                        error_type="ReactGenerationError",
                    )
                ]
            )
        ],
    )
    provider = diagnose_swebench_wrong_demo(row, trajectory=provider_trajectory)
    assert provider["failure_layer"] == "provider_execution"
    assert provider["failure_type"] == "ProviderRequestError"

    schema_trajectory = _trajectory(
        "swe:parse",
        [
            _accepted_turn(
                [
                    _execution(
                        trace=[
                            {
                                "turn": 1,
                                "observation_status": "schema_invalid",
                                "public_error_code": "tool_arguments_schema_invalid",
                            }
                        ]
                    )
                ]
            )
        ],
    )
    schema = diagnose_swebench_wrong_demo(row, trajectory=schema_trajectory)
    assert schema["failure_layer"] == "structured_tool_action"
    assert schema["failure_type"] == "tool_arguments_schema_invalid"


def test_wrong_demo_terminal_and_official_harness_layers_are_distinct():
    row = _row("swe:terminal", direct_status="resolved", graph_status="unresolved")
    accepted = [_accepted_turn()]

    no_patch = diagnose_swebench_wrong_demo(
        row,
        trajectory=_trajectory("swe:terminal", accepted, final_answer=None),
    )
    assert no_patch["failure_layer"] == "workspace_patch"

    no_finish = diagnose_swebench_wrong_demo(
        row,
        trajectory=_trajectory("swe:terminal", accepted, finish=False),
    )
    assert no_finish["failure_layer"] == "finish"

    patch_apply_row = _row(
        "swe:terminal", direct_status="resolved", graph_status="patch_apply_failed"
    )
    patch_apply = diagnose_swebench_wrong_demo(
        patch_apply_row,
        trajectory={
            **_trajectory("swe:terminal", accepted),
            "evaluation": _evaluation("patch_apply_failed"),
        },
    )
    assert patch_apply["failure_layer"] == "official_harness_patch_apply"

    missing = diagnose_swebench_wrong_demo(
        row,
        trajectory=None,
        collection_failures=[
            {
                "task_id": "swe:terminal",
                "condition": "swebench_agentgraph",
                "stage": "repository_setup",
            }
        ],
    )
    assert missing["failure_layer"] == "repository_environment"
    assert missing["causal_attribution"] is False
