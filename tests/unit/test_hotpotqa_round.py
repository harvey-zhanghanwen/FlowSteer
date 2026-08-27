from __future__ import annotations

import importlib.util
from pathlib import Path

from src.interactive.config_loader import load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_hotpotqa_round.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_hotpotqa_round", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_round_config_is_fixed_heldout_and_training_disabled():
    config = load_yaml(_ROOT / "config" / "evaluation_hotpotqa_round_01.yaml")
    _MODULE.validate_hotpot_config(config)


def test_embedding_v4_config_freezes_question_only_development_profile():
    config = load_yaml(
        _ROOT / "config" / "evaluation_hotpotqa_embedding_retrieval_v4.yaml"
    )

    _MODULE.validate_hotpot_config(config)

    retrieval = config["qa_embedding_retrieval"]
    assert retrieval["question_scope"] == "question_only"
    assert retrieval["search_top_k"] == 4
    assert retrieval["max_tool_calls_per_agent_call"] == 4
    assert retrieval["max_turns_per_agent_call"] == 6
    assert retrieval["web_search_enabled"] is False


def test_strict_aggregate_keeps_failed_task_in_denominator():
    rows = [
        {
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 0.8,
            }
        },
        {
            "agentgraph": {
                "available": False,
                "valid": False,
                "exact_match": 0.0,
                "token_f1": 0.0,
            }
        },
    ]

    result = _MODULE._aggregate(rows, "agentgraph")

    assert result["denominator"] == 2
    assert result["completed"] == 1
    assert result["evaluator_valid"] == 1
    assert result["strict_exact_match"] == 0.5
    assert result["strict_token_f1"] == 0.4
    assert result["completed_only_exact_match"] == 1.0


def _search_and_read_receipts():
    search = {
        "tool_id": "hotpotqa.qa_memory",
        "tool_version": "qa-memory-test-v1",
        "started_at_monotonic": 1.0,
        "ended_at_monotonic": 2.0,
        "request": {
            "action": "search",
            "arguments": {"query": "Alpha author", "k": 2},
        },
        "result": {
            "completed": True,
            "value": {
                "memory_ids": ["memory-1"],
                "hits": [
                    {
                        "memory_id": "memory-1",
                        "source_train_task_id": "train-1",
                    }
                ],
            },
        },
        "error_type": None,
    }
    read = {
        "tool_id": "hotpotqa.qa_memory",
        "tool_version": "qa-memory-test-v1",
        "started_at_monotonic": 3.0,
        "ended_at_monotonic": 4.0,
        "request": {
            "action": "read",
            "arguments": {"memory_id": "memory-1"},
        },
        "result": {
            "completed": True,
            "value": {"memory_id": "memory-1"},
        },
        "error_type": None,
    }
    return search, read


def _retrieval_trajectory(*, include_output_inbox: bool):
    search, read = _search_and_read_receipts()
    worker_request = {
        "request_id": "run:1:worker:single",
        "graph_revision": 1,
        "is_output_agent": False,
        "execution_role": "worker",
        "agent": {
            "id": "worker",
            "execution_mode": "react",
            "allowed_tools": ["hotpotqa.qa_memory"],
        },
        "upstream": [],
        "peer_draft": None,
    }
    executions = [
        {
            "agent_id": "worker",
            "output": "grounded evidence artifact",
            "metadata": {
                "request": worker_request,
                "response": {"tool_receipts": [search, read]},
            },
        }
    ]
    if include_output_inbox:
        executions.append(
            {
                "agent_id": "output",
                "output": "<answer>Ada Lovelace</answer>",
                "metadata": {
                    "request": {
                        "request_id": "run:1:output:single",
                        "graph_revision": 1,
                        "is_output_agent": True,
                        "execution_role": "worker",
                        "agent": {
                            "id": "output",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        "upstream": [
                            {
                                "source_agent_id": "worker",
                                "target_agent_id": "output",
                                "graph_revision": 1,
                                "tool_receipts": [search, read],
                            }
                        ],
                        "peer_draft": None,
                    },
                    "response": {"tool_receipts": [search, read]},
                },
            }
        )
    return {
        "explicit_finish": True,
        "turns": [
            {
                "action": {"action": "add_agent"},
                "graph_snapshot": {
                    "output_agent_id": "output",
                    "relations": [
                        {
                            "source_id": "worker",
                            "target_id": "output",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                },
                "executions": executions,
            }
        ],
    }


def test_retrieval_boundary_requires_worker_search_read_in_actual_output_inbox():
    trajectories = {
        "task-1": _retrieval_trajectory(include_output_inbox=True),
    }

    result = _MODULE._retrieval_boundary_statistics(
        trajectories,
        tool_id="hotpotqa.qa_memory",
    )

    assert result["director_tool_calls"] == 0
    assert result["retrieval_tool_calls_by_worker"] == 2
    assert result["retrieval_artifact_routed_via_relation"] is True
    assert result["retrieval_output_inbox_lineage_tasks"] == 1
    assert result["unique_train_base_task_ids_retrieved"] == 1


def test_graph_reachability_without_output_inbox_receipts_is_not_routed():
    result = _MODULE._retrieval_boundary_statistics(
        {"task-1": _retrieval_trajectory(include_output_inbox=False)},
        tool_id="hotpotqa.qa_memory",
    )

    assert result["retrieval_tool_calls_by_worker"] == 2
    assert result["finished_retrieval_invoked_tasks"] == 1
    assert result["retrieval_output_inbox_lineage_tasks"] == 0
    assert result["retrieval_artifact_routed_via_relation"] is False


def test_output_inbox_search_without_matching_read_is_not_lineage():
    trajectory = _retrieval_trajectory(include_output_inbox=True)
    executions = trajectory["turns"][0]["executions"]
    executions[0]["metadata"]["response"]["tool_receipts"] = executions[0][
        "metadata"
    ]["response"]["tool_receipts"][:1]
    executions[1]["metadata"]["request"]["upstream"][0][
        "tool_receipts"
    ] = executions[1]["metadata"]["request"]["upstream"][0][
        "tool_receipts"
    ][:1]

    result = _MODULE._retrieval_boundary_statistics(
        {"task-1": trajectory},
        tool_id="hotpotqa.qa_memory",
    )

    assert result["retrieval_tool_calls_by_worker"] == 1
    assert result["retrieval_output_inbox_lineage_tasks"] == 0
    assert result["retrieval_artifact_routed_via_relation"] is False
