from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_triviaqa_qa_memory_results.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_test_triviaqa_qa_memory_result_analysis", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def _receipt(action: str, ordinal: int) -> dict[str, object]:
    arguments: dict[str, object]
    if action == "search":
        arguments = {"query": "rewritten public question", "limit": 3}
        value: dict[str, object] = {
            "hits": [
                {
                    "memory_id": "memory-train-1",
                    "source_train_task_id": "triviaqa:train:1",
                    "similarity": 0.91,
                    "paraphrase_question": "A semantically equivalent train question",
                    "paraphrase_answer_statement": "The answer is train answer",
                    "canonical_answer": "train answer",
                }
            ]
        }
    else:
        arguments = {"memory_id": "memory-train-1"}
        value = {
            "memory": {
                "memory_id": "memory-train-1",
                "source_train_task_id": "triviaqa:train:1",
                "paraphrase_question": "A semantically equivalent train question",
                "paraphrase_answer_statement": "The answer is train answer",
                "canonical_answer": "train answer",
            }
        }
    return {
        "tool_id": analysis.QA_MEMORY_TOOL_ID,
        "tool_version": "qa-memory-test-v1",
        "request": {"action": action, "arguments": arguments},
        "result": {"completed": True, "value": value},
        "started_at_monotonic": float(ordinal),
        "ended_at_monotonic": float(ordinal) + 0.1,
        "latency_ms": 100.0,
        "error_type": None,
    }


def _trajectory(task_id: str) -> dict[str, object]:
    search = _receipt("search", 1)
    read = _receipt("read", 2)
    relations = [
        {
            "source_id": "retriever",
            "target_id": "reasoner",
            "source_to_target": True,
            "target_to_source": False,
        },
        {
            "source_id": "reasoner",
            "target_id": "formatter",
            "source_to_target": True,
            "target_to_source": False,
        },
    ]
    graph = {
        "nodes": [
            {
                "id": "retriever",
                "execution_mode": "react",
                "allowed_tools": [analysis.QA_MEMORY_TOOL_ID],
            },
            {"id": "reasoner", "execution_mode": "reasoning", "allowed_tools": []},
            {"id": "formatter", "execution_mode": "reasoning", "allowed_tools": []},
        ],
        "relations": relations,
        "output_agent_id": "formatter",
        "revision": 3,
    }
    common_request = {
        "execution_role": "worker",
        "graph_revision": 3,
        "problem": "public validation question",
    }
    return {
        "schema_version": "flowsteer.agentgraph.trajectory.v1",
        "task": {
            "task_id": task_id,
            "question": "public validation question",
            "ground_truth": "reference answer",
            "split": "validation",
            "metadata": {},
        },
        "turns": [
            {
                "round_index": 0,
                "director_request_id": "director-1",
                "prompt": "Public task and control-plane Canvas state only.",
                "policy_response": '{"action":"add_subgraph"}',
                "action": {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "retriever",
                            "allowed_tools": [analysis.QA_MEMORY_TOOL_ID],
                            "execution_mode": "react",
                        }
                    ],
                    "relations": relations,
                    "output_agent_id": "formatter",
                },
                "canvas_feedback": (
                    "accepted add_subgraph; execution_result="
                    '{"status":"success","artifact_id":"artifact-1"}'
                ),
                "graph_revision": 3,
                "graph_snapshot": graph,
                "executions": [
                    {
                        "agent_id": "retriever",
                        "execution_id": "execution-retriever",
                        "output": "evidence artifact",
                        "metadata": {
                            "request": {
                                **common_request,
                                "agent": {
                                    "id": "retriever",
                                    "execution_mode": "react",
                                    "allowed_tools": [analysis.QA_MEMORY_TOOL_ID],
                                },
                                "upstream": [],
                            },
                            "response": {
                                "tool_receipts": [search, read],
                                "react_trace": [],
                            },
                        },
                    },
                    {
                        "agent_id": "reasoner",
                        "execution_id": "execution-reasoner",
                        "output": "semantic wrong answer",
                        "metadata": {
                            "request": {
                                **common_request,
                                "agent": {
                                    "id": "reasoner",
                                    "execution_mode": "reasoning",
                                    "allowed_tools": [],
                                },
                                "upstream": [
                                    {
                                        "source_agent_id": "retriever",
                                        "target_agent_id": "reasoner",
                                        "artifact_type": "evidence",
                                        "artifact": "evidence artifact",
                                        "tool_receipts": [search, read],
                                    }
                                ],
                            },
                            "response": {},
                        },
                    },
                    {
                        "agent_id": "formatter",
                        "execution_id": "execution-formatter",
                        "output": "<answer>wrong answer</answer>",
                        "metadata": {
                            "request": {
                                **common_request,
                                "agent": {
                                    "id": "formatter",
                                    "execution_mode": "reasoning",
                                    "allowed_tools": [],
                                },
                                "upstream": [
                                    {
                                        "source_agent_id": "reasoner",
                                        "target_agent_id": "formatter",
                                        "artifact_type": "semantic_answer",
                                        "artifact": "semantic wrong answer",
                                        "tool_receipts": [],
                                    }
                                ],
                            },
                            "response": {},
                        },
                    },
                ],
                "runtime_summary": {
                    "output_agent_id": "formatter",
                    "final_answer": "<answer>wrong answer</answer>",
                },
            },
            {
                "round_index": 1,
                "director_request_id": "director-2",
                "prompt": "Public task and typed execution receipt only.",
                "policy_response": '{"action":"finish"}',
                "action": {"action": "finish"},
                "canvas_feedback": "workflow finished",
                "graph_revision": 3,
                "graph_snapshot": graph,
                "executions": [],
                "runtime_summary": {
                    "output_agent_id": "formatter",
                    "final_answer": "<answer>wrong answer</answer>",
                },
            },
        ],
        "final_answer": "<answer>wrong answer</answer>",
        "termination_reason": "finish",
        "explicit_finish": True,
        "evaluation": {
            "evaluator_version": "triviaqa.official.answer.v1",
            "valid": True,
            "reason": "evaluated",
            "reward": 0.0,
            "metrics": {"exact_match": 0.0, "token_f1": 0.0},
            "details": {"structured_answer_extracted": True},
        },
    }


def _paired(task_id: str, direct_correct: bool) -> dict[str, object]:
    def condition(score: float, answer: str) -> dict[str, object]:
        return {
            "available": True,
            "valid": True,
            "exact_match": score,
            "token_f1": score,
            "final_answer": answer,
            "evaluation": {
                "evaluator_version": "triviaqa.official.answer.v1",
                "valid": True,
                "metrics": {"exact_match": score, "token_f1": score},
            },
        }

    return {
        "task_id": task_id,
        "direct": condition(1.0 if direct_correct else 0.0, "<answer>direct</answer>"),
        "agentgraph": condition(0.0, "<answer>wrong answer</answer>"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _assert_formal_report_metrics_tool_ownership_isolation_routing_and_demos() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        task_ids = [f"triviaqa:validation:{index}" for index in range(3)]
        selected = [
            {
                "task_id": task_id,
                "question": "public validation question",
                "ground_truth": "reference answer",
                "split": "validation",
                "metadata": {"evaluator_payload": {"accepted_answers": ["reference answer"]}},
            }
            for task_id in task_ids
        ]
        selected_path = root / "selected.jsonl"
        trajectories_path = root / "trajectories.jsonl"
        paired_path = root / "paired.jsonl"
        manifest_path = root / "run_manifest.json"
        index_manifest_path = root / "index_manifest.json"
        _write_jsonl(selected_path, selected)
        _write_jsonl(trajectories_path, [_trajectory(task_id) for task_id in task_ids])
        _write_jsonl(
            paired_path,
            [_paired(task_id, index == 0) for index, task_id in enumerate(task_ids)],
        )
        manifest_path.write_text(
            json.dumps({"status": "completed", "sample_count": 3}), encoding="utf-8"
        )
        index_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "flowsteer.triviaqa.qa_memory.manifest.v1",
                    "tool_id": analysis.QA_MEMORY_TOOL_ID,
                    "source_split": "train",
                    "memory_count": 512,
                    "unique_source_count": 512,
                    "cycled_count": 0,
                    "paraphrase_count": 512,
                    "validation_content_indexed": False,
                    "validation_isolation_count": 128,
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            selected_tasks=str(selected_path),
            trajectories=str(trajectories_path),
            paired_results=str(paired_path),
            run_manifest=str(manifest_path),
            index_manifest=str(index_manifest_path),
            demo_count=3,
        )
        report = analysis.build_report(args)

    assert report["run"]["status"] == "complete"
    assert report["metrics"]["direct"]["strict_exact_match"] == 1 / 3
    assert report["metrics"]["agentgraph"]["strict_exact_match"] == 0.0
    retrieval = report["retrieval_diagnostics"]
    assert retrieval["tool_call_count"] == 6
    assert retrieval["search_call_count"] == 3
    assert retrieval["read_call_count"] == 3
    assert retrieval["search_candidate_accepted_answer_match_count"] == 0
    assertions = report["control_plane_and_tool_routing"]["assertions"]
    assert assertions["director_tool_calls"] == 0
    assert assertions["director_data_plane_isolated"] is True
    assert assertions["retrieval_tool_calls_by_worker"] == 6
    assert assertions["retrieval_tool_calls_by_worker_gt_0"] is True
    assert assertions["worker_ownership_violation_count"] == 0
    assert assertions["retrieval_artifact_routed_via_relation"] is True
    assert assertions["output_inbox_receipt_lineage"] is False
    assert assertions["retrieval_tasks_with_output_inbox_receipt_lineage"] == 0
    assert report["wrong_demo_selection"]["actual_count"] == 3
    assert report["wrong_demo_selection"]["minimum_formal_count_met"] is True
    assert all(demo["actual_execution_chain"] for demo in report["wrong_demos"])


def _assert_control_plane_assertions_fail_closed_on_payload_leak_and_missing_relation() -> None:
    trajectory = _trajectory("triviaqa:validation:leak")
    trajectory["turns"][0]["prompt"] = (  # type: ignore[index]
        "Leaked paraphrase_answer_statement and memory-train-1"
    )
    trajectory["turns"][0]["graph_snapshot"]["relations"] = []  # type: ignore[index]

    result = analysis._trajectory_control_plane(  # noqa: SLF001
        "triviaqa:validation:leak", trajectory
    )

    assert result["director_retrieval_payload_markers"] == [
        "paraphrase_answer_statement"
    ]
    assert result["director_exposed_memory_ids"] == ["memory-train-1"]
    assert result["director_exposed_retrieval_values"] == [
        {"field": "memory_id", "value": "memory-train-1"}
    ]
    assert result["retrieval_artifact_routed_via_relation"] is False


def _assert_field_names_are_diagnostic_only_and_output_inbox_lineage_is_independent() -> None:
    task_id = "triviaqa:validation:field-only"
    trajectory = _trajectory(task_id)
    trajectory["turns"][0]["prompt"] = (  # type: ignore[index]
        "The ordinary worker contract may name memory_id, canonical_answer, "
        "paraphrase_question, paraphrase_answer_statement, and source_train_task_id "
        "without containing any retrieved value."
    )
    control, _ = analysis._aggregate_control_plane(  # noqa: SLF001
        [task_id], {task_id: trajectory}
    )
    assertions = control["assertions"]
    assert control["director_field_name_diagnostics"][task_id]
    assert control["director_payload_exposures"] == {}
    assert assertions["director_data_plane_isolated"] is True
    assert assertions["retrieval_artifact_routed_via_relation"] is True
    assert assertions["output_inbox_receipt_lineage"] is False

    worker_receipts = trajectory["turns"][0]["executions"][0]["metadata"][  # type: ignore[index]
        "response"
    ]["tool_receipts"]
    trajectory["turns"][0]["executions"][2]["metadata"]["request"][  # type: ignore[index]
        "upstream"
    ][0]["tool_receipts"] = worker_receipts
    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    assert result["retrieval_artifact_routed_via_relation"] is True
    assert result["output_inbox_receipt_lineage"] is True
    assert result["output_inbox_canonical_receipt_count"] == 2
    assert result["output_inbox_missing_canonical_receipt_signatures"] == []


def _assert_bare_answer_collision_is_not_a_payload_exposure() -> None:
    task_id = "triviaqa:validation:answer-collision"
    trajectory = _trajectory(task_id)
    trajectory["turns"][0]["prompt"] = (  # type: ignore[index]
        "The public question contains train answer as ordinary text."
    )

    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001

    assert result["director_exposed_retrieval_values"] == []
    assert result["director_retrieval_value_collisions"] == [
        {"field": "canonical_answer", "value": "train answer"}
    ]

    trajectory["turns"][0]["prompt"] = (  # type: ignore[index]
        'Leaked record: {"canonical_answer":"train answer"}'
    )
    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    assert result["director_exposed_retrieval_values"] == [
        {"field": "canonical_answer", "value": "train answer"}
    ]

    trajectory["turns"][0]["prompt"] = (  # type: ignore[index]
        "Flow-Director chat transcript\n\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Public Canvas state only."},
                    {
                        "role": "assistant",
                        "content": '{"canonical_answer":"train answer"}',
                    },
                ]
            }
        )
    )
    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    assert result["director_exposed_retrieval_values"] == []
    assert result["director_retrieval_value_collisions"] == [
        {"field": "canonical_answer", "value": "train answer"}
    ]

    short_receipt = trajectory["turns"][0]["executions"][0]["metadata"][  # type: ignore[index]
        "response"
    ]["tool_receipts"][0]
    short_receipt["result"]["value"]["hits"][0]["canonical_answer"] = "H"
    trajectory["turns"][0]["prompt"] = '{"canonical_answer":"H"}'  # type: ignore[index]
    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    assert result["director_exposed_retrieval_values"] == [
        {"field": "canonical_answer", "value": "H"}
    ]


def _assert_post_hoc_retrieval_coverage_uses_official_normalization() -> None:
    task_id = "triviaqa:validation:coverage"
    selected = {
        task_id: {
            "task_id": task_id,
            "ground_truth": "The Train Answer",
            "metadata": {
                "evaluator_payload": {"accepted_answers": ["The Train Answer"]}
            },
        }
    }
    report = analysis._retrieval_coverage(  # noqa: SLF001
        selected,
        {task_id: _trajectory(task_id)},
        [{"canonical_answer": "train answer"}],
    )

    assert report["analysis_scope"] == "post_hoc_offline_only_not_model_visible"
    assert report["corpus_accepted_answer_match_count"] == 1
    assert report["search_top1_accepted_answer_match_count"] == 1
    assert report["search_candidate_accepted_answer_match_count"] == 1
    assert report["read_candidate_accepted_answer_match_count"] == 1


def _assert_failed_worker_receipts_count_without_claiming_artifact_route() -> None:
    task_id = "triviaqa:validation:failed-worker"
    trajectory = _trajectory(task_id)
    turn = trajectory["turns"][0]  # type: ignore[index]
    search = _receipt("search", 11)
    read = _receipt("read", 12)
    turn["executions"] = []
    turn["runtime_summary"] = {
        "failure_records": [
            {
                "agent_id": "retriever",
                "error_type": "ReactExecutionError",
                "metadata": {"tool_receipts": [search, read]},
            }
        ],
        "output_agent_id": "formatter",
    }

    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    assert result["retrieval_tool_call_count"] == 2
    assert result["retrieval_artifact_receipt_count"] == 0
    assert result["worker_ownership_violations"] == []
    assert result["retrieval_artifact_routed_via_relation"] is False

    control, _ = analysis._aggregate_control_plane(  # noqa: SLF001
        [task_id], {task_id: trajectory}
    )
    assertions = control["assertions"]
    assert assertions["retrieval_tool_calls_by_worker"] == 2
    assert assertions["retrieval_tool_calls_by_worker_gt_0"] is True
    assert assertions["retrieval_artifact_tasks"] == 0
    assert assertions["retrieval_artifact_routed_via_relation"] is False


class TriviaQAQAMemoryResultAnalysisTests(unittest.TestCase):
    def test_formal_report_metrics_tool_ownership_isolation_routing_and_demos(
        self,
    ) -> None:
        _assert_formal_report_metrics_tool_ownership_isolation_routing_and_demos()

    def test_control_plane_assertions_fail_closed_on_payload_leak_and_missing_relation(
        self,
    ) -> None:
        _assert_control_plane_assertions_fail_closed_on_payload_leak_and_missing_relation()

    def test_field_names_are_diagnostic_only_and_output_inbox_lineage_is_independent(
        self,
    ) -> None:
        _assert_field_names_are_diagnostic_only_and_output_inbox_lineage_is_independent()

    def test_failed_worker_receipts_count_without_claiming_artifact_route(
        self,
    ) -> None:
        _assert_failed_worker_receipts_count_without_claiming_artifact_route()

    def test_bare_answer_collision_is_not_a_payload_exposure(self) -> None:
        _assert_bare_answer_collision_is_not_a_payload_exposure()

    def test_post_hoc_retrieval_coverage_uses_official_normalization(self) -> None:
        _assert_post_hoc_retrieval_coverage_uses_official_normalization()
