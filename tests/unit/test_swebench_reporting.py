from __future__ import annotations

import json

from src.interactive.swebench_reporting import (
    SWEBENCH_FAILURE_TAXONOMY,
    aggregate_swebench_receipts,
    build_swebench_wrong_demo_taxonomy,
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
        _receipt("edit_file", {"ok": True, "changed": True}),
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
        "edit_file": 1,
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


def test_failure_taxonomy_reports_explicit_zero_and_na_share_without_wrong_tasks():
    taxonomy = build_swebench_wrong_demo_taxonomy([], [])

    assert taxonomy["denominator"] == 0
    assert taxonomy["denominator_status"] == "no_evaluated_wrong_demo"
    assert taxonomy["run_preflight_blockers_included"] is False
    assert taxonomy["representative_demos"] == []
    assert [category["code"] for category in taxonomy["categories"]] == [
        category["code"] for category in SWEBENCH_FAILURE_TAXONOMY
    ]
    assert all(category["count"] == 0 for category in taxonomy["categories"])
    assert all(category["share"] is None for category in taxonomy["categories"])
    assert all(
        category["share_status"]
        == "not_applicable_no_evaluated_wrong_demo"
        for category in taxonomy["categories"]
    )


def test_failure_taxonomy_uses_terminal_official_outcome_without_inferring_recovery():
    row = _row("swe:recovered-debug", direct_status="resolved", graph_status="unresolved")
    trajectory = _trajectory(
        "swe:recovered-debug",
        [
            _accepted_turn(
                [
                    _execution(
                        trace=[
                            {
                                "turn": 1,
                                "structured_action": {
                                    "kind": "tool",
                                    "name": "run_tests",
                                },
                                "observation": {
                                    "observation_status": "success",
                                    "result": {
                                        "ok": True,
                                        "passed": False,
                                        "timed_out": False,
                                    },
                                },
                            }
                        ]
                    )
                ]
            )
        ],
    )

    taxonomy = build_swebench_wrong_demo_taxonomy([row], [trajectory])
    counts = {category["code"]: category["count"] for category in taxonomy["categories"]}

    assert taxonomy["denominator"] == 1
    assert sum(counts.values()) == 1
    assert counts["repository_tool_failure"] == 0
    assert counts["official_test_failure_unclassified"] == 1
    demo = taxonomy["representative_demos"][0]
    assert demo["failure_analysis"]["first_observable_failure"]["failure_type"] == (
        "local_test_failed"
    )
    assert demo["failure_analysis"]["first_causal_failure"] is None
    assert demo["failure_analysis"]["causality_status"] == "not_established"


def test_failure_taxonomy_excludes_preflight_placeholder_rows():
    row = _row("swe:preflight", direct_status="resolved", graph_status="unresolved")
    row["agentgraph"].update(
        available=False,
        valid=False,
        evaluation=None,
        trajectory_id=None,
    )
    taxonomy = build_swebench_wrong_demo_taxonomy(
        [row],
        [],
        [
            {
                "task_id": "swe:preflight",
                "condition": "swebench_agentgraph",
                "stage": "repository_setup",
                "execution_started": False,
            }
        ],
    )

    assert taxonomy["denominator"] == 0
    assert taxonomy["representative_demos"] == []
    assert all(category["count"] == 0 for category in taxonomy["categories"])
    report = aggregate_swebench_receipts(
        [row],
        [],
        [
            {
                "task_id": "swe:preflight",
                "condition": "swebench_agentgraph",
                "stage": "repository_setup",
                "execution_started": False,
            }
        ],
    )
    assert report["wrong_demo_count"] == 0
    assert report["wrong_demo_diagnoses"] == []


def test_failure_taxonomy_prioritizes_max_rounds_over_not_called_evaluator():
    row = _row("swe:max-rounds", direct_status="resolved", graph_status="unresolved")
    not_called = {
        "valid": False,
        "metrics": {},
        "reason": "not_evaluated_without_explicit_finish",
        "details": {},
    }
    row["agentgraph"].update(
        valid=False,
        evaluation=not_called,
        explicit_finish=False,
        termination_reason="max_rounds",
    )
    trajectory = {
        **_trajectory("swe:max-rounds", [_accepted_turn()], finish=False),
        "termination_reason": "max_rounds",
        "evaluation": not_called,
    }

    taxonomy = build_swebench_wrong_demo_taxonomy([row], [trajectory])
    counts = {category["code"]: category["count"] for category in taxonomy["categories"]}

    assert counts["terminal_budget_failure"] == 1
    assert counts["evaluator_runtime_failure"] == 0


def test_failure_taxonomy_treats_failed_local_test_as_validation_not_tool_backend():
    row = _row("swe:local-test", direct_status="resolved", graph_status="unresolved")
    row["agentgraph"]["evaluation"] = None
    trajectory = _trajectory(
        "swe:local-test",
        [
            _accepted_turn(
                [
                    _execution(
                        trace=[
                            {
                                "turn": 1,
                                "structured_action": {
                                    "kind": "tool",
                                    "name": "run_tests",
                                },
                                "observation": {
                                    "observation_status": "success",
                                    "result": {
                                        "ok": True,
                                        "passed": False,
                                        "timed_out": False,
                                    },
                                },
                            }
                        ]
                    )
                ]
            )
        ],
    )
    trajectory.pop("evaluation")

    taxonomy = build_swebench_wrong_demo_taxonomy([row], [trajectory])
    counts = {category["code"]: category["count"] for category in taxonomy["categories"]}

    assert counts["local_validation_failure"] == 1
    assert counts["repository_tool_failure"] == 0


def test_failure_taxonomy_rejects_duplicate_task_rows():
    row = _row("swe:duplicate", direct_status="resolved", graph_status="unresolved")
    trajectory = _trajectory("swe:duplicate", [_accepted_turn()])

    try:
        build_swebench_wrong_demo_taxonomy([row, row], [trajectory])
    except ValueError as exc:
        assert "duplicate SWE-bench paired task_id" in str(exc)
    else:
        raise AssertionError("duplicate task IDs must fail closed")


def test_failure_taxonomy_distinguishes_target_test_failure_and_regression():
    target_row = _row("swe:target", direct_status="resolved", graph_status="unresolved")
    target_evaluation = _evaluation("unresolved")
    target_evaluation["details"]["tests_status"] = {
        "FAIL_TO_PASS": {"success": [], "failure": ["SECRET_F2P_ID"]},
        "PASS_TO_PASS": {"success": ["SECRET_P2P_ID"], "failure": []},
    }
    target_row["agentgraph"]["evaluation"] = target_evaluation
    target_trajectory = {
        **_trajectory("swe:target", [_accepted_turn()]),
        "evaluation": target_evaluation,
    }

    regression_row = _row(
        "swe:regression", direct_status="resolved", graph_status="unresolved"
    )
    regression_evaluation = _evaluation("unresolved")
    regression_evaluation["details"]["tests_status"] = {
        "FAIL_TO_PASS": {"success": ["SECRET_F2P_ID"], "failure": []},
        "PASS_TO_PASS": {"success": [], "failure": ["SECRET_P2P_ID"]},
    }
    regression_row["agentgraph"]["evaluation"] = regression_evaluation
    regression_trajectory = {
        **_trajectory("swe:regression", [_accepted_turn()]),
        "evaluation": regression_evaluation,
    }

    taxonomy = build_swebench_wrong_demo_taxonomy(
        [target_row, regression_row],
        [target_trajectory, regression_trajectory],
    )
    counts = {category["code"]: category["count"] for category in taxonomy["categories"]}

    assert taxonomy["denominator"] == 2
    assert counts["official_target_test_failure"] == 1
    assert counts["official_regression_failure"] == 1
    assert sum(counts.values()) == 2
    serialized = json.dumps(taxonomy)
    assert "SECRET_F2P_ID" not in serialized
    assert "SECRET_P2P_ID" not in serialized


def test_representative_demo_contains_full_receipt_chain_and_redacts_gold_target():
    row = _row("swe:demo", direct_status="resolved", graph_status="unresolved")
    row.update(
        question="Fix the public issue",
        ground_truth="GOLD_PATCH_SENTINEL",
        repository_state={
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "base",
        },
    )
    row["agentgraph"]["output_agent_inbox"] = {
        "evaluator_payload": {"test_patch": "SECRET_OUTPUT_INBOX"}
    }
    secret_evaluation = _evaluation("unresolved")
    secret_evaluation["details"]["harness_details"] = "SECRET_TEST_CONTRACT"
    row["agentgraph"]["evaluation"] = secret_evaluation
    execution = _execution(
        receipts=[_receipt("search_code", {"ok": True})],
        trace=[
            {
                "turn": 1,
                "structured_action": {"kind": "tool", "name": "search_code"},
                "observation": {
                    "observation_status": "success",
                    "result": {"ok": True},
                },
            }
        ],
        agent_id="agent_search",
    )
    execution.update(
        execution_id="execution-1",
        model_id="qwen3.5-9b-local",
        provider="local",
        output="candidate patch",
    )
    execution["metadata"]["request"] = {
        "agent": {
            "id": "agent_search",
            "model_id": "qwen3.5-9b-local",
            "contract": "Inspect the repository and propose a patch.",
        },
        "problem": "Fix the public issue",
        "phase": "single",
        "rendered_messages": [{"role": "user", "content": "Fix the public issue"}],
        "upstream": [],
        "evaluator_payload": {"gold_patch": "GOLD_PATCH_SENTINEL"},
    }
    execution["metadata"]["response"]["model_calls"] = [
        {"request_status": "success", "evaluator_payload": "SECRET_MODEL_CALL"}
    ]
    execution["metadata"]["response"]["react_trace"][0][
        "evaluator_payload"
    ] = "SECRET_REACT_TRACE"
    execution["metadata"]["response"]["tool_receipts"][0][
        "evaluator_payload"
    ] = "SECRET_TOOL_RECEIPT"
    turn = _accepted_turn([execution])
    turn.update(
        prompt="Director input",
        policy_response='{"action":"add_subgraph"}',
        policy_version="policy-v1",
        director_request_id="director-1",
        graph_revision=1,
        graph_snapshot={
            "nodes": [
                {
                    "id": "agent_search",
                    "model_id": "qwen3.5-9b-local",
                    "contract": "Inspect the repository and propose a patch.",
                }
            ],
            "relations": [],
            "output_agent_id": "agent_search",
            "revision": 1,
            "evaluator_payload": "SECRET_GRAPH_SNAPSHOT",
        },
        runtime_summary={"evaluator_payload": "SECRET_RUNTIME"},
    )
    trajectory = {
        **_trajectory("swe:demo", [turn]),
        "trajectory_id": "trajectory-1",
        "evaluation": secret_evaluation,
        "task": {
            "task_id": "swe:demo",
            "ground_truth": "GOLD_PATCH_SENTINEL",
            "metadata": {"evaluator_payload": {"test_patch": "SECRET_TEST"}},
        },
    }

    taxonomy = build_swebench_wrong_demo_taxonomy([row], [trajectory])
    demo = taxonomy["representative_demos"][0]

    assert demo["task_id"] == "swe:demo"
    assert demo["sample_id"] == "owner__repo-1"
    assert demo["input"]["question"] == "Fix the public issue"
    assert demo["reference_target"] == {
        "metric": "official_swebench_harness_resolved",
        "expected_resolved": True,
        "gold_patch": None,
        "fail_to_pass": None,
        "pass_to_pass": None,
        "redaction_role": "evaluator_only_redacted",
    }
    chain_turn = demo["execution_chain"]["director_canvas_turns"][0]
    assert chain_turn["director"]["parsed_action"] == {"action": "add_subgraph"}
    assert chain_turn["canvas_edit"]["feedback"] == "accepted add_subgraph"
    agent = chain_turn["agent_executions"][0]
    assert agent["agent_input"]["problem"] == "Fix the public issue"
    assert agent["agent_input"]["evaluator_payload"] is None
    assert agent["agent_input"]["evaluator_payload_role"] == (
        "evaluator_only_redacted"
    )
    assert agent["agent_output"] == "candidate patch"
    assert agent["react_tool_execution"]["tool_receipts"][0]["request"][
        "action"
    ] == "search_code"
    assert demo["execution_chain"]["terminal_receipt"] is not None
    assert demo["execution_chain"]["evaluator_receipt"] is not None
    serialized = json.dumps(taxonomy)
    assert "GOLD_PATCH_SENTINEL" not in serialized
    assert "SECRET_TEST" not in serialized
    assert "SECRET_MODEL_CALL" not in serialized
    assert "SECRET_REACT_TRACE" not in serialized
    assert "SECRET_TOOL_RECEIPT" not in serialized
    assert "SECRET_GRAPH_SNAPSHOT" not in serialized
    assert "SECRET_RUNTIME" not in serialized
    assert "SECRET_OUTPUT_INBOX" not in serialized
    assert "SECRET_TEST_CONTRACT" not in serialized


def test_representative_demo_keeps_missing_chain_receipts_null():
    row = _row("swe:missing", direct_status="resolved", graph_status="unresolved")
    row["agentgraph"].update(
        available=False,
        valid=False,
        evaluation=None,
        trajectory_id=None,
    )
    failures = [
        {
            "task_id": "swe:missing",
            "condition": "swebench_agentgraph",
            "stage": "provider_request",
            "execution_started": True,
            "failure_phase": "provider",
        }
    ]

    taxonomy = build_swebench_wrong_demo_taxonomy([row], [], failures)
    demo = taxonomy["representative_demos"][0]

    assert demo["execution_chain"]["director_canvas_turns"] is None
    assert demo["execution_chain"]["terminal_receipt"] is None
    assert demo["failure_analysis"]["subsequent_observed_receipts"][
        "subsequent_turn_receipts"
    ] is None
