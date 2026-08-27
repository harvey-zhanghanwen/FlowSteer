from __future__ import annotations

import importlib.util
from pathlib import Path

from src.interactive.config_loader import load_yaml
from src.interactive.task_dataset import iter_task_records


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


def test_qa_memory_v10_separates_director_and_worker_action_budgets():
    config = load_yaml(
        _ROOT / "config" / "evaluation_hotpotqa_qa_memory_v10.yaml"
    )

    _MODULE.validate_hotpot_config(config)

    assert config["director"]["max_action_tokens"] == 1024
    assert config["qa_embedding_retrieval"]["max_action_tokens"] == 4096


def test_qa_memory_v11_freezes_worker_only_local_retrieval_dataflow():
    config = load_yaml(
        _ROOT / "config" / "evaluation_hotpotqa_qa_memory_v11.yaml"
    )

    _MODULE.validate_hotpot_config(config)

    retrieval = config["qa_embedding_retrieval"]
    graph = config["agent_graph"]
    assert retrieval["corpus_kind"] == "train_qa_memory"
    assert retrieval["train_sample_count"] == 512
    assert retrieval["validation_sample_count"] == 128
    assert retrieval["search_top_k"] == 2
    assert retrieval["web_search_enabled"] is False
    assert graph["required_evidence_tool_id"] == "hotpotqa.qa_memory"
    assert graph["require_evidence_relation"] is True
    assert graph["terminal_protocol"] == "exact_single_answer_tag"
    assert graph["require_format_agent"] is False
    assert "add_agent" not in graph["actions"]


def test_qa_memory_v13_keeps_public_task_and_question_only_worker_query():
    config = load_yaml(
        _ROOT / "config" / "evaluation_hotpotqa_qa_memory_v13.yaml"
    )

    _MODULE.validate_hotpot_config(config)

    retrieval = config["qa_embedding_retrieval"]
    assert retrieval["task_scope"] == "public_task"
    assert retrieval["retrieval_query_scope"] == "question_only"
    assert retrieval["search_top_k"] == 2
    assert retrieval["max_tool_calls_per_agent_call"] == 4
    assert retrieval["web_search_enabled"] is False
    assert retrieval["index_dir"].endswith(
        "hotpotqa_qa_memory_source_freeze_v2/index"
    )
    assert config["hotpotqa_evaluation"]["sample_count"] == 128

    source_freeze = list(
        iter_task_records(
            _ROOT / config["data"]["validation_path"],
            expected_split="validation",
        )
    )
    round01 = list(
        iter_task_records(
            _ROOT / "artifacts/hotpotqa_round_01/selected_tasks.jsonl",
            expected_split="validation",
        )
    )
    assert len(source_freeze) == 128
    assert [task.task_id for task in source_freeze] == [
        task.task_id for task in round01
    ]
    assert all("\n\nQuestion:" in task.question for task in source_freeze)


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


def test_retrieval_boundary_requires_worker_receipt_and_relation_path():
    trajectories = {
        "task-1": {
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
                    "executions": [
                        {
                            "agent_id": "worker",
                            "metadata": {
                                "response": {
                                    "tool_receipts": [
                                        {
                                            "tool_id": "hotpotqa.qa_memory",
                                            "result": {
                                                "value": {
                                                    "memory_ids": ["memory-1"],
                                                    "hits": [
                                                        {
                                                            "source_train_task_id": "train-1"
                                                        }
                                                    ],
                                                }
                                            },
                                        }
                                    ]
                                }
                            },
                        }
                    ],
                }
            ],
        }
    }

    result = _MODULE._retrieval_boundary_statistics(
        trajectories,
        tool_id="hotpotqa.qa_memory",
    )

    assert result["director_tool_calls"] == 0
    assert result["retrieval_tool_calls_by_worker"] == 1
    assert result["retrieval_artifact_routed_via_relation"] is True
    assert result["unique_train_base_task_ids_retrieved"] == 1
