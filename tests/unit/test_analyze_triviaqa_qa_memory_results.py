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


def _ordered_top_k_fixture(
    top_k: int = 3,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    rows = [
        {
            "memory_id": f"memory-train-{index}",
            "source_train_task_id": f"triviaqa:train:{index}",
            "paraphrase_question": f"Equivalent train question {index}",
            "paraphrase_answer_statement": f"The answer is train answer {index}",
            "canonical_answer": f"train answer {index}",
            "rank": index,
            "similarity": round(0.95 - index * 0.05, 2),
        }
        for index in range(1, top_k + 1)
    ]
    memory_ids = [str(row["memory_id"]) for row in rows]
    query = "rewritten public question"
    search: dict[str, object] = {
        "tool_id": analysis.QA_MEMORY_TOOL_ID,
        "tool_version": "qa-memory-test-v2",
        "request": {
            "action": "search",
            "arguments": {"query": query, "limit": top_k},
        },
        "result": {
            "completed": True,
            "value": {
                "operation": "search",
                "query": query,
                "top_k": top_k,
                "memory_ids": memory_ids,
                "hits": [
                    {
                        "memory_id": row["memory_id"],
                        "rank": row["rank"],
                        "similarity": row["similarity"],
                    }
                    for row in rows
                ],
            },
        },
        "error_type": None,
    }
    reads: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for row in rows:
        memory = {
            "memory_id": row["memory_id"],
            "source_train_task_id": row["source_train_task_id"],
            "paraphrase_question": row["paraphrase_question"],
            "paraphrase_answer_statement": row[
                "paraphrase_answer_statement"
            ],
            "canonical_answer": row["canonical_answer"],
            "text": (
                f"Question: {row['paraphrase_question']}\n"
                f"Answer: {row['paraphrase_answer_statement']}"
            ),
        }
        reads.append(
            {
                "tool_id": analysis.QA_MEMORY_TOOL_ID,
                "tool_version": "qa-memory-test-v2",
                "request": {
                    "action": "read",
                    "arguments": {"memory_id": row["memory_id"]},
                },
                "result": {
                    "completed": True,
                    "value": {
                        "operation": "read",
                        "memory_id": row["memory_id"],
                        "memory": memory,
                    },
                },
                "error_type": None,
            }
        )
        candidates.append(
            {
                field: row[field]
                for field in analysis.QA_MEMORY_BATCH_CANDIDATE_FIELDS
            }
        )
    artifact = {
        "question_scope": "public validation question",
        "retrieval_query": query,
        "top_k": top_k,
        "candidates": candidates,
    }
    return search, reads, artifact


def _director_prompt(
    *,
    allowed_tools: list[str] | None = None,
    tool_calls_enabled: bool = False,
    diagnostic: str = "public control-plane state",
    staged_completion: bool = False,
) -> str:
    observation = {
        "director_execution_profile": {
            "allowed_tools": list(allowed_tools or []),
            "tool_calls_enabled": tool_calls_enabled,
        },
        "diagnostic": diagnostic,
    }
    messages = [
        {"role": "system", "content": "test Flow-Director policy"},
        {"role": "user", "content": "Canvas observation\n\n" + json.dumps(observation)},
    ]
    if staged_completion:
        messages.extend(
            [
                {"role": "assistant", "content": '{"action":"add_subgraph"}'},
                {
                    "role": "user",
                    "content": (
                        "Complete the add_subgraph action for these Agent "
                        "declarations. Return only the JSON object."
                    ),
                },
            ]
        )
    return analysis.DIRECTOR_TRANSCRIPT_HEADER + "\n\n" + json.dumps(
        {
            "schema_version": analysis.DIRECTOR_TRANSCRIPT_SCHEMA,
            "messages": messages,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class DirectorExecutionProfileParsingTest(unittest.TestCase):
    def test_staged_structured_action_keeps_canvas_profile(self) -> None:
        trajectory = {
            "turns": [
                {
                    "round_index": 0,
                    "prompt": _director_prompt(staged_completion=True),
                }
            ]
        }

        profiles, violations = analysis._director_execution_profiles(trajectory)

        self.assertEqual([], violations)
        self.assertEqual(
            [
                {
                    "round_index": 0,
                    "allowed_tools": [],
                    "tool_calls_enabled": False,
                }
            ],
            profiles,
        )


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
                "role_family": "evidence_retriever",
                "execution_mode": "react",
                "allowed_tools": [analysis.QA_MEMORY_TOOL_ID],
            },
            {
                "id": "reasoner",
                "role_family": "reasoner",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "id": "formatter",
                "role_family": "format",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
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
                "prompt": _director_prompt(),
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
                                    "role_family": "evidence_retriever",
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
                                    "role_family": "reasoner",
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
                                    "role_family": "format",
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
                "prompt": _director_prompt(
                    diagnostic="public terminal control-plane state"
                ),
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


def _ordered_top_k_trajectory(
    task_id: str, top_k: int = 3
) -> dict[str, object]:
    trajectory = _trajectory(task_id)
    search, reads, artifact = _ordered_top_k_fixture(top_k)
    receipts = [search, *reads]
    artifact_text = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    turn = trajectory["turns"][0]  # type: ignore[index]
    retriever = turn["executions"][0]
    retriever["output"] = artifact_text
    retriever["metadata"]["response"] = {
        "tool_receipts": receipts,
        "react_trace": [
            {
                "structured_action": {
                    "kind": "complete",
                    "arguments": {
                        "value": {
                            "memory_ids": [
                                candidate["memory_id"]
                                for candidate in artifact["candidates"]
                            ]
                        }
                    },
                }
            }
        ],
    }
    reasoner_upstream = turn["executions"][1]["metadata"]["request"][
        "upstream"
    ][0]
    reasoner_upstream["artifact"] = artifact_text
    reasoner_upstream["tool_receipts"] = receipts
    return trajectory


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
    assertions = report["control_plane_and_tool_routing"]["assertions"]
    assert assertions["director_tool_calls"] == 0
    assert assertions["director_request_allowed_tools"] == []
    assert assertions["director_requests_toolless"] is True
    assert assertions["director_data_plane_isolated"] is True
    assert assertions["retrieval_tool_calls_by_worker"] == 6
    assert assertions["retrieval_tool_calls_by_worker_gt_0"] is True
    assert assertions["worker_ownership_violation_count"] == 0
    assert assertions["reasoner_qamemory_tool_unassigned"] is True
    assert assertions["retrieval_artifact_routed_via_relation"] is True
    assert assertions["output_inbox_receipt_lineage"] is False
    assert assertions["retrieval_tasks_with_output_inbox_receipt_lineage"] == 0
    assert report["wrong_demo_selection"]["actual_count"] == 3
    assert report["wrong_demo_selection"]["minimum_formal_count_met"] is True
    assert all(demo["actual_execution_chain"] for demo in report["wrong_demos"])


def _assert_control_plane_assertions_fail_closed_on_payload_leak_and_missing_relation() -> None:
    trajectory = _trajectory("triviaqa:validation:leak")
    trajectory["turns"][0]["prompt"] = _director_prompt(  # type: ignore[index]
        diagnostic="Leaked paraphrase_answer_statement and memory-train-1"
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
    trajectory["turns"][0]["prompt"] = _director_prompt(  # type: ignore[index]
        diagnostic=(
            "The ordinary worker contract may name memory_id, canonical_answer, "
            "paraphrase_question, paraphrase_answer_statement, and "
            "source_train_task_id without containing any retrieved value."
        )
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


def _assert_director_outputs_are_not_misclassified_as_request_payload() -> None:
    task_id = "triviaqa:validation:director-output"
    trajectory = _trajectory(task_id)
    turn = trajectory["turns"][0]  # type: ignore[index]
    turn["prompt"] = _director_prompt(
        diagnostic=(
            "The public task or graph state may contain the bare span "
            "train answer without containing a QA-memory record."
        )
    )
    turn["policy_response"] = (
        '{"action":"add_subgraph","diagnostic":"memory-train-1"}'
    )
    turn["action"]["diagnostic"] = "The emitted candidate was train answer"
    turn["canvas_feedback"] = "memory-train-1"
    turn["reconstructed_context"] = "train answer"

    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001

    assert result["director_exposed_retrieval_values"] == []
    assert result["director_exposed_memory_ids"] == []


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


def _assert_director_profile_and_reasoner_tool_assignment_fail_closed() -> None:
    task_id = "triviaqa:validation:ownership"
    trajectory = _trajectory(task_id)
    trajectory["turns"][0]["prompt"] = _director_prompt(  # type: ignore[index]
        allowed_tools=[analysis.QA_MEMORY_TOOL_ID],
        tool_calls_enabled=True,
    )
    reasoner_node = trajectory["turns"][0]["graph_snapshot"]["nodes"][1]  # type: ignore[index]
    reasoner_node["allowed_tools"] = [analysis.QA_MEMORY_TOOL_ID]
    reasoner_request = trajectory["turns"][0]["executions"][1]["metadata"][  # type: ignore[index]
        "request"
    ]["agent"]
    reasoner_request["allowed_tools"] = [analysis.QA_MEMORY_TOOL_ID]

    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    assert result["director_allowed_tools"] == [analysis.QA_MEMORY_TOOL_ID]
    assert result["director_tool_calls_enabled_count"] == 1
    assert len(result["reasoner_qamemory_tool_assignment_violations"]) == 2

    control, _ = analysis._aggregate_control_plane(  # noqa: SLF001
        [task_id], {task_id: trajectory}
    )
    assertions = control["assertions"]
    assert assertions["director_requests_toolless"] is False
    assert assertions["director_data_plane_isolated"] is False
    assert assertions["reasoner_qamemory_tool_assignment_violation_count"] == 2
    assert assertions["reasoner_qamemory_tool_unassigned"] is False


def _assert_ordered_top_k_projection_and_k_completeness() -> None:
    task_id = "triviaqa:validation:ordered-top-k"
    trajectory = _ordered_top_k_trajectory(task_id, top_k=3)

    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    projections = result["native_artifact_receipt_projections"]
    assert len(projections) == 1
    projection = projections[0]
    assert projection["projection_kind"] == "ordered_top_k_batch"
    assert projection["expected_top_k"] == 3
    assert projection["completion_memory_id_count"] == 3
    assert projection["search_hit_count"] == 3
    assert projection["artifact_candidate_count"] == 3
    assert projection["successful_read_count"] == 3
    assert projection["successful_search_count"] == 1
    assert projection["qa_tool_action_count"] == 4
    assert projection["one_search_plus_k_ordered_reads"] is True
    assert projection["ordered_memory_ids_match"] is True
    assert projection["rank_similarity_exact"] is True
    assert projection["read_records_exact"] is True
    assert projection["k_complete"] is True
    assert projection["receipt_exact"] is True
    assert all(
        candidate["receipt_exact"] is True
        for candidate in projection["candidate_receipt_diagnostics"]
    )
    assert result["native_top_k_batch_projection_count"] == 1
    assert result["native_top_k_batch_complete_count"] == 1
    assert result["native_top_k_batch_incomplete_count"] == 0

    control, _ = analysis._aggregate_control_plane(  # noqa: SLF001
        [task_id], {task_id: trajectory}
    )
    assertions = control["assertions"]
    assert assertions["director_tool_calls_eq_0"] is True
    assert assertions["retrieval_tool_calls_by_worker_gt_0"] is True
    assert assertions["retrieval_artifact_routed_via_relation"] is True
    assert assertions["native_top_k_batch_projection_count"] == 1
    assert assertions["native_top_k_batch_complete_count"] == 1
    assert assertions["native_top_k_batch_incomplete_count"] == 0
    assert assertions["native_top_k_batch_expected_k_values"] == [3]
    assert assertions["native_top_k_batches_complete"] is True


def _assert_ordered_top_k_projection_fails_closed_on_receipt_mismatch() -> None:
    task_id = "triviaqa:validation:ordered-top-k-mismatch"
    trajectory = _ordered_top_k_trajectory(task_id, top_k=3)
    turn = trajectory["turns"][0]  # type: ignore[index]
    retriever = turn["executions"][0]
    artifact = json.loads(retriever["output"])
    artifact["candidates"][1]["similarity"] = 0.01
    artifact["candidates"][2]["paraphrase_answer_statement"] = (
        "tampered answer statement"
    )
    artifact_text = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    retriever["output"] = artifact_text
    response = retriever["metadata"]["response"]
    response["tool_receipts"] = response["tool_receipts"][:-1]
    reasoner_upstream = turn["executions"][1]["metadata"]["request"][
        "upstream"
    ][0]
    reasoner_upstream["artifact"] = artifact_text
    reasoner_upstream["tool_receipts"] = response["tool_receipts"]

    result = analysis._trajectory_control_plane(task_id, trajectory)  # noqa: SLF001
    projection = result["native_artifact_receipt_projections"][0]
    assert projection["projection_kind"] == "ordered_top_k_batch"
    assert projection["successful_read_count"] == 2
    assert projection["one_search_plus_k_ordered_reads"] is False
    assert projection["rank_similarity_exact"] is False
    assert projection["read_records_exact"] is False
    assert projection["k_complete"] is False
    assert projection["receipt_exact"] is False
    assert "search_hit.similarity" in projection[
        "candidate_receipt_diagnostics"
    ][1]["mismatched_receipt_fields"]
    assert "read_request.memory_id" in projection[
        "candidate_receipt_diagnostics"
    ][2]["mismatched_receipt_fields"]

    control, _ = analysis._aggregate_control_plane(  # noqa: SLF001
        [task_id], {task_id: trajectory}
    )
    assertions = control["assertions"]
    assert assertions["native_artifact_receipt_projection_violation_count"] == 1
    assert assertions["native_top_k_batch_complete_count"] == 0
    assert assertions["native_top_k_batch_incomplete_count"] == 1
    assert assertions["native_top_k_batches_complete"] is False


def _assert_legacy_singular_projection_remains_supported() -> None:
    trajectory = _trajectory("triviaqa:validation:legacy-singular")
    turn = trajectory["turns"][0]  # type: ignore[index]
    retriever = turn["executions"][0]
    response = retriever["metadata"]["response"]
    search = response["tool_receipts"][0]
    search["result"]["value"]["memory_ids"] = ["memory-train-1"]
    artifact = {
        "memory_id": "memory-train-1",
        "canonical_answer": "train answer",
        "paraphrase_question": "A semantically equivalent train question",
        "paraphrase_answer_statement": "The answer is train answer",
        "source_train_task_id": "triviaqa:train:1",
    }
    retriever["output"] = json.dumps(artifact, sort_keys=True)
    response["react_trace"] = [
        {
            "structured_action": {
                "kind": "complete",
                "arguments": {"value": {"memory_id": "memory-train-1"}},
            }
        }
    ]

    projections = analysis._native_artifact_receipt_projections(  # noqa: SLF001
        trajectory
    )
    assert len(projections) == 1
    projection = projections[0]
    assert projection["projection_kind"] == "legacy_singular"
    assert projection["selection_matches_artifact"] is True
    assert projection["exact_search_read_receipt_found"] is True
    assert projection["k_completeness_applicable"] is False
    assert projection["k_complete"] is None
    assert projection["receipt_exact"] is True


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

    def test_director_outputs_are_not_misclassified_as_request_payload(self) -> None:
        _assert_director_outputs_are_not_misclassified_as_request_payload()

    def test_failed_worker_receipts_count_without_claiming_artifact_route(
        self,
    ) -> None:
        _assert_failed_worker_receipts_count_without_claiming_artifact_route()

    def test_director_profile_and_reasoner_tool_assignment_fail_closed(
        self,
    ) -> None:
        _assert_director_profile_and_reasoner_tool_assignment_fail_closed()

    def test_ordered_top_k_projection_and_k_completeness(self) -> None:
        _assert_ordered_top_k_projection_and_k_completeness()

    def test_ordered_top_k_projection_fails_closed_on_receipt_mismatch(
        self,
    ) -> None:
        _assert_ordered_top_k_projection_fails_closed_on_receipt_mismatch()

    def test_legacy_singular_projection_remains_supported(self) -> None:
        _assert_legacy_singular_projection_remains_supported()
