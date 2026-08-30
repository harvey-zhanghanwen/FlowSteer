from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_triviaqa_fact_memory_results.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_test_triviaqa_fact_memory_result_analysis", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def _fact_receipts(
    *, top_k: int = 2
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    query = "rewritten public task"
    hits = [
        {
            "memory_id": f"fact-{index}",
            "rank": index,
            "similarity": round(0.95 - index * 0.05, 2),
            "fact_text": f"Entity {index} has a self-contained public fact.",
        }
        for index in range(1, top_k + 1)
    ]
    memory_ids = [str(hit["memory_id"]) for hit in hits]
    search: dict[str, object] = {
        "tool_id": analysis.FACT_MEMORY_TOOL_ID,
        "tool_version": "fact-memory-test-v1",
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
                "hits": hits,
            },
        },
        "error_type": None,
    }
    reads: list[dict[str, object]] = []
    for hit in hits:
        memory_id = str(hit["memory_id"])
        reads.append(
            {
                "tool_id": analysis.FACT_MEMORY_TOOL_ID,
                "tool_version": "fact-memory-test-v1",
                "request": {
                    "action": "read",
                    "arguments": {"memory_id": memory_id},
                },
                "result": {
                    "completed": True,
                    "value": {
                        "operation": "read",
                        "memory_id": memory_id,
                        "memory": {
                            "memory_id": memory_id,
                            "fact_text": hit["fact_text"],
                        },
                    },
                },
                "error_type": None,
            }
        )
    artifact = {
        "question_scope": "public task scope",
        "retrieval_query": query,
        "top_k": top_k,
        "candidates": hits,
        "retrieval_status": "evidence_found",
        "relevant_memory_ids": memory_ids,
    }
    return search, reads, artifact


def _trajectory(
    task_id: str,
    *,
    exact_match: float = 0.0,
    token_f1: float = 0.2,
) -> dict[str, object]:
    search, reads, artifact = _fact_receipts()
    receipts = [search, *reads]
    artifact_text = json.dumps(artifact, ensure_ascii=False, sort_keys=True)
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
                "allowed_tools": [analysis.FACT_MEMORY_TOOL_ID],
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
        "revision": 1,
    }
    common_request = {
        "execution_role": "worker",
        "graph_revision": 1,
        "problem": "public validation task",
    }
    routed_evidence = {
        "source_agent_id": "retriever",
        "target_agent_id": "reasoner",
        "artifact_type": "evidence",
        "artifact": artifact_text,
        "tool_receipts": receipts,
    }
    routed_semantics = {
        "source_agent_id": "reasoner",
        "target_agent_id": "formatter",
        "artifact_type": "semantic_answer",
        "artifact": "semantic candidate",
        "tool_receipts": receipts,
    }
    final_answer = "<answer>wrong span</answer>"
    return {
        "schema_version": "flowsteer.agentgraph.trajectory.v1",
        "task": {
            "task_id": task_id,
            "question": "original public validation question",
            "ground_truth": "private reference answer",
            "split": "validation",
            "metadata": {
                "evaluator_payload": {
                    "accepted_answers": ["private reference answer"]
                }
            },
        },
        "turns": [
            {
                "round_index": 0,
                "director_request_id": f"director-{task_id}",
                "prompt": "public control-plane state only",
                "policy_response": '{"action":"add_subgraph"}',
                "action": {
                    "action": "add_subgraph",
                    "agents": ["retriever", "reasoner", "formatter"],
                    "relations": relations,
                    "output_agent_id": "formatter",
                },
                "canvas_feedback": "accepted and executed",
                "graph_revision": 1,
                "graph_snapshot": graph,
                "executions": [
                    {
                        "agent_id": "retriever",
                        "execution_id": f"retriever-{task_id}",
                        "output": artifact_text,
                        "metadata": {
                            "request": {
                                **common_request,
                                "agent": graph["nodes"][0],
                                "upstream": [],
                            },
                            "response": {
                                "tool_receipts": receipts,
                                "react_trace": [],
                            },
                        },
                    },
                    {
                        "agent_id": "reasoner",
                        "execution_id": f"reasoner-{task_id}",
                        "output": "semantic candidate",
                        "metadata": {
                            "request": {
                                **common_request,
                                "agent": graph["nodes"][1],
                                "upstream": [routed_evidence],
                            },
                            "response": {},
                        },
                    },
                    {
                        "agent_id": "formatter",
                        "execution_id": f"formatter-{task_id}",
                        "output": final_answer,
                        "metadata": {
                            "request": {
                                **common_request,
                                "agent": graph["nodes"][2],
                                "upstream": [routed_semantics],
                            },
                            "response": {},
                        },
                    },
                ],
                "runtime_summary": {
                    "output_agent_id": "formatter",
                    "final_answer": final_answer,
                },
            }
        ],
        "final_answer": final_answer,
        "termination_reason": "finish",
        "explicit_finish": True,
        "evaluation": {
            "evaluator_version": "triviaqa.official.answer.v1",
            "valid": True,
            "reason": "evaluated",
            "reward": exact_match,
            "metrics": {"exact_match": exact_match, "token_f1": token_f1},
        },
    }


def _paired(
    task_id: str,
    *,
    graph_em: float = 0.0,
    graph_f1: float = 0.2,
) -> dict[str, object]:
    def condition(em: float, f1: float) -> dict[str, object]:
        return {
            "available": True,
            "valid": True,
            "exact_match": em,
            "token_f1": f1,
            "evaluation": {
                "evaluator_version": "triviaqa.official.answer.v1",
                "valid": True,
                "metrics": {"exact_match": em, "token_f1": f1},
            },
        }

    return {
        "task_id": task_id,
        "direct": condition(1.0, 1.0),
        "agentgraph": condition(graph_em, graph_f1),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(
    root: Path,
    *,
    expected_count: int = 3,
    wrong_count: int | None = None,
) -> tuple[argparse.Namespace, list[dict[str, object]]]:
    wrong_count = expected_count if wrong_count is None else wrong_count
    task_ids = [f"triviaqa:validation:{index}" for index in range(expected_count)]
    selected = [
        {
            "task_id": task_id,
            "question": "original public validation question",
            "ground_truth": "private reference answer",
            "split": "validation",
            "metadata": {
                "evaluator_payload": {
                    "accepted_answers": ["private reference answer"]
                }
            },
        }
        for task_id in task_ids
    ]
    trajectories = [
        _trajectory(
            task_id,
            exact_match=0.0 if index < wrong_count else 1.0,
            token_f1=0.2 if index < wrong_count else 1.0,
        )
        for index, task_id in enumerate(task_ids)
    ]
    paired = [
        _paired(
            task_id,
            graph_em=0.0 if index < wrong_count else 1.0,
            graph_f1=0.2 if index < wrong_count else 1.0,
        )
        for index, task_id in enumerate(task_ids)
    ]
    selected_path = root / "selected.jsonl"
    trajectories_path = root / "trajectories.jsonl"
    paired_path = root / "paired.jsonl"
    manifest_path = root / "run_manifest.json"
    index_manifest_path = root / "index_manifest.json"
    materialization_manifest_path = root / "materialization_manifest.json"
    top_k_selection_receipt_path = root / "top_k_selection_receipt.json"
    _write_jsonl(selected_path, selected)
    _write_jsonl(trajectories_path, trajectories)
    _write_jsonl(paired_path, paired)
    manifest_path.write_text(
        json.dumps({"status": "completed", "sample_count": expected_count}),
        encoding="utf-8",
    )
    index_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "flowsteer.fact-memory.index-manifest.v1",
                "record_kind": "fact_memory",
                "tool_id": analysis.FACT_MEMORY_TOOL_ID,
                "memory_count": analysis.EXPECTED_FACT_MEMORY_COUNT,
                "fact_only": True,
                "embedding_input_field": "fact_text",
                "provenance_loaded_by_index": False,
                "embedding_model": "frozen-test-embedding",
                "embedding_dimension": 3,
                "normalization": "l2",
                "similarity": "cosine",
                "frozen_top_k": 2,
                "tool_budget": {"search": 1, "read": 2},
            }
        ),
        encoding="utf-8",
    )
    materialization_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": analysis.MATERIALIZATION_SCHEMA_VERSION,
                "record_count": analysis.EXPECTED_FACT_MEMORY_COUNT,
                "unique_source_count": analysis.EXPECTED_FACT_MEMORY_COUNT,
                "strict_semantic_paraphrase_count": (
                    analysis.EXPECTED_FACT_MEMORY_COUNT
                ),
                "lexical_or_phrase_replacement_count": (
                    analysis.EXPECTED_FACT_MEMORY_COUNT
                ),
                "fact_text_count": analysis.EXPECTED_FACT_MEMORY_COUNT,
                "dataset_pair_fallback_count": 0,
                "bounded_failure_fallback_count": 0,
                "pending_gap_fallback_count": 0,
                "legacy_fallback_regeneration_count": 0,
                "exact_original_question_substring_count": 0,
            }
        ),
        encoding="utf-8",
    )
    top_k_selection_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": analysis.TOP_K_SELECTION_SCHEMA_VERSION,
                "selection_split": "architecture_development",
                "selection_rule": (
                    "smallest top-k with maximal development recall"
                ),
                "selected_top_k": 2,
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
        materialization_manifest=str(materialization_manifest_path),
        top_k_selection_receipt=str(top_k_selection_receipt_path),
        expected_count=expected_count,
        expected_fact_memory_count=analysis.EXPECTED_FACT_MEMORY_COUNT,
        demo_count=3,
    )
    return args, trajectories


def test_complete_snapshot_reports_formal_metrics_and_protocol(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)

    report = analysis.build_report(args)

    assert report["run"]["status"] == "complete"
    assert report["metrics"]["denominator"] == 3
    assert report["metrics"]["formal"]["direct"]["strict_exact_match"] == 1.0
    assert report["metrics"]["formal"]["agentgraph"]["strict_exact_match"] == 0.0
    graph_f1 = report["metrics"]["formal"]["agentgraph"]["strict_token_f1"]
    assert graph_f1 is not None and abs(graph_f1 - 0.2) < 1e-12
    assert report["terminal"]["strict_failure_count"] == 0
    assertions = report["protocol_assertions"]
    assert assertions["director_tool_calls"] == 0
    assert assertions["retrieval_tool_calls_by_worker_gt_0"] is True
    assert assertions["first_data_plane_action_is_search"] is True
    assert assertions["complete_top_k_read_by_rank"] is True
    assert assertions["fact_artifact_routed_via_explicit_relation"] is True
    assert assertions["output_lineage"] is True
    assert assertions["web_search_count"] == 0
    assert assertions["agent_facing_data_plane_violation_count"] == 0
    assert assertions["materialization_manifest_valid"] is True
    assert all(assertions["materialization_count_checks"].values())
    assert all(assertions["materialization_fallback_checks"].values())
    assert assertions["exact_original_question_substring_count_eq_0"] is True
    assert assertions["final_index_memory_count_valid"] is True
    assert assertions["top_k_selection_receipt_valid"] is True
    assert assertions["selected_top_k"] == 2
    assert assertions["final_index_frozen_top_k"] == 2
    assert assertions["selected_top_k_matches_final_index"] is True
    assert assertions["protocol_valid"] is True
    assert report["wrong_demo_selection"]["actual_count"] == 3
    assert report["wrong_demo_selection"]["shortfall"] == 0
    assert len({demo["task_id"] for demo in report["wrong_demos"]}) == 3
    assert all(demo["actual_execution_chain"] for demo in report["wrong_demos"])


def test_agent_facing_private_field_leak_withholds_formal_metrics(
    tmp_path: Path,
) -> None:
    args, trajectories = _fixture(tmp_path)
    leaked = copy.deepcopy(trajectories)
    first_turn = leaked[0]["turns"][0]  # type: ignore[index]
    response = first_turn["executions"][0]["metadata"]["response"]  # type: ignore[index]
    response["tool_receipts"][0]["result"]["value"]["canonical_answer"] = "leak"
    response["tool_receipts"][0]["result"]["value"]["qa_pair"] = {
        "Question": "leaked prompt wrapper",
        "Answer": "leaked response wrapper",
    }
    _write_jsonl(Path(args.trajectories), leaked)

    report = analysis.build_report(args)

    assert report["run"]["snapshot_complete"] is True
    assert report["run"]["protocol_valid"] is False
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None
    task = report["per_task_protocol"]["triviaqa:validation:0"]
    assert task["data_plane_violation_count"] > 0
    assert any(
        item["marker"] == "canonical_answer"
        for item in task["data_plane_violations"]
    )
    assert any(
        item["marker"] == "Question/Answer mapping"
        for item in task["data_plane_violations"]
    )


def test_semantic_agent_answer_wrapper_is_not_a_fact_memory_leak(
    tmp_path: Path,
) -> None:
    args, trajectories = _fixture(tmp_path)
    semantic_output = copy.deepcopy(trajectories)
    first_turn = semantic_output[0]["turns"][0]  # type: ignore[index]
    reasoner = first_turn["executions"][1]  # type: ignore[index]
    reasoner["output"] = "Answer: model-authored semantic candidate"
    _write_jsonl(Path(args.trajectories), semantic_output)

    report = analysis.build_report(args)

    task = report["per_task_protocol"]["triviaqa:validation:0"]
    assert task["data_plane_violation_count"] == 0


def test_read_before_search_and_incomplete_top_k_fail_closed(tmp_path: Path) -> None:
    args, trajectories = _fixture(tmp_path)
    broken = copy.deepcopy(trajectories)
    first_turn = broken[0]["turns"][0]  # type: ignore[index]
    response = first_turn["executions"][0]["metadata"]["response"]  # type: ignore[index]
    receipts = response["tool_receipts"]
    response["tool_receipts"] = [receipts[1], receipts[0]]
    _write_jsonl(Path(args.trajectories), broken)

    report = analysis.build_report(args)

    task = report["per_task_protocol"]["triviaqa:validation:0"]
    assert task["first_data_plane_action"] == "read"
    assert task["first_data_plane_action_is_search"] is False
    assert task["complete_top_k_read_by_rank"] is False
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None


def test_missing_top_k_selection_receipt_fails_closed(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    Path(args.top_k_selection_receipt).unlink()

    report = analysis.build_report(args)

    assertions = report["protocol_assertions"]
    assert assertions["top_k_selection_receipt_valid"] is False
    assert assertions["selected_top_k_matches_final_index"] is False
    assert assertions["protocol_valid"] is False
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None
    assert report["run"]["top_k_selection_receipt_read_error"].startswith(
        "missing file:"
    )


def test_missing_materialization_manifest_fails_closed(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    Path(args.materialization_manifest).unlink()

    report = analysis.build_report(args)

    assertions = report["protocol_assertions"]
    assert assertions["materialization_manifest_valid"] is False
    assert assertions["protocol_valid"] is False
    assert report["run"]["snapshot_complete"] is True
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None
    assert report["run"]["materialization_manifest_read_error"].startswith(
        "missing file:"
    )


def test_materialization_manifest_requires_every_strict_count_and_zero_fallback(
    tmp_path: Path,
) -> None:
    args, _ = _fixture(tmp_path)
    materialization_path = Path(args.materialization_manifest)
    valid = json.loads(materialization_path.read_text(encoding="utf-8"))
    invalid_values = {
        "schema_version": "flowsteer.triviaqa.fact_memory.materialization.old",
        "record_count": analysis.EXPECTED_FACT_MEMORY_COUNT - 1,
        "unique_source_count": analysis.EXPECTED_FACT_MEMORY_COUNT - 1,
        "strict_semantic_paraphrase_count": (
            analysis.EXPECTED_FACT_MEMORY_COUNT - 1
        ),
        "lexical_or_phrase_replacement_count": (
            analysis.EXPECTED_FACT_MEMORY_COUNT - 1
        ),
        "fact_text_count": analysis.EXPECTED_FACT_MEMORY_COUNT - 1,
        "dataset_pair_fallback_count": 1,
        "bounded_failure_fallback_count": 1,
        "pending_gap_fallback_count": 1,
        "legacy_fallback_regeneration_count": 1,
        "exact_original_question_substring_count": 1,
    }

    for field, invalid_value in invalid_values.items():
        invalid = {**valid, field: invalid_value}
        materialization_path.write_text(json.dumps(invalid), encoding="utf-8")

        report = analysis.build_report(args)

        assert report["protocol_assertions"][
            "materialization_manifest_valid"
        ] is False, field
        assert report["protocol_assertions"]["protocol_valid"] is False, field
        assert report["run"]["formal_metrics_available"] is False, field
        assert report["metrics"]["formal"] is None, field

    missing = dict(valid)
    missing.pop("fact_text_count")
    materialization_path.write_text(json.dumps(missing), encoding="utf-8")
    report = analysis.build_report(args)
    assert report["protocol_assertions"][
        "materialization_manifest_valid"
    ] is False
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None


def test_final_index_memory_count_must_equal_full_materialization(
    tmp_path: Path,
) -> None:
    args, _ = _fixture(tmp_path)
    index_path = Path(args.index_manifest)
    index_manifest = json.loads(index_path.read_text(encoding="utf-8"))
    index_manifest["memory_count"] = analysis.EXPECTED_FACT_MEMORY_COUNT - 1
    index_path.write_text(json.dumps(index_manifest), encoding="utf-8")

    report = analysis.build_report(args)

    assertions = report["protocol_assertions"]
    assert assertions["final_index_memory_count_valid"] is False
    assert assertions["protocol_valid"] is False
    assert report["run"]["snapshot_complete"] is False
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None


def test_selected_top_k_must_match_final_index_manifest(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    selection_path = Path(args.top_k_selection_receipt)
    receipt = json.loads(selection_path.read_text(encoding="utf-8"))
    receipt["selected_top_k"] = 3
    selection_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = analysis.build_report(args)

    assertions = report["protocol_assertions"]
    assert assertions["top_k_selection_receipt_valid"] is True
    assert assertions["selected_top_k"] == 3
    assert assertions["final_index_frozen_top_k"] == 2
    assert assertions["selected_top_k_matches_final_index"] is False
    assert assertions["protocol_valid"] is False
    assert report["run"]["snapshot_complete"] is True
    assert report["run"]["formal_metrics_available"] is False
    assert report["metrics"]["formal"] is None


def test_incomplete_snapshot_and_wrong_demo_shortfall_are_not_fabricated(
    tmp_path: Path,
) -> None:
    args, trajectories = _fixture(tmp_path, expected_count=2, wrong_count=1)
    manifest = Path(args.run_manifest)
    manifest.write_text(
        json.dumps({"status": "prepared", "sample_count": 2}),
        encoding="utf-8",
    )

    report = analysis.build_report(args)

    assert report["run"]["snapshot_complete"] is False
    assert report["metrics"]["formal"] is None
    assert report["terminal"]["strict_failure_count"] is None
    assert report["wrong_demo_selection"]["actual_count"] == 1
    assert report["wrong_demo_selection"]["shortfall"] == 2
    assert [demo["task_id"] for demo in report["wrong_demos"]] == [
        trajectories[0]["task"]["task_id"]  # type: ignore[index]
    ]
