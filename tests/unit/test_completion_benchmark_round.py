from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.interactive.config_loader import load_model_registry, load_yaml
from src.interactive.task_evaluator import EvaluationOutcome


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _evaluation_config(dataset_key: str) -> dict:
    config = deepcopy(load_yaml(_ROOT / "config" / "evaluation_hotpotqa_round_01.yaml"))
    config.pop("hotpotqa_evaluation")
    if dataset_key in {"hotpotqa", "triviaqa"}:
        section_name = f"{dataset_key}_evaluation"
        phase = f"{dataset_key}_evaluation"
        split = "validation"
        sample_count = 2
        extra = {"protocol_equivalent_to_direct": False}
    elif dataset_key == "aime_2026":
        section_name = "aime2026_evaluation"
        phase = "aime2026_evaluation"
        split = "test"
        sample_count = 30
        extra = {
            "official_2026_only": True,
            "benchmark_slice": "official_aime_2026",
        }
    elif dataset_key == "healthbench_professional":
        section_name = "healthbench_professional_evaluation"
        phase = "healthbench_professional_evaluation"
        split = "validation"
        sample_count = 2
        extra = {}
    else:
        section_name = f"{dataset_key}_evaluation"
        phase = f"{dataset_key}_evaluation"
        split = "validation"
        sample_count = 2
        extra = {}
    config["experiment"]["phase"] = phase
    config[section_name] = {
        "dataset_key": dataset_key,
        "split": split,
        "selection": "sequential",
        "sample_count": sample_count,
        "rollouts_per_task": 1,
        "concurrency": 2,
        "direct_model_id": "qwen3.5-9b-local",
        "direct_protocol": "single_call_v1",
        "direct_contract": "Complete the task and return the requested final answer.",
        **extra,
    }
    if dataset_key == "healthbench_professional":
        config["evaluation"]["healthbench_judge_catalog_path"] = (
            "config/model_catalog_healthbench_reference_judge.yaml"
        )
    if dataset_key in {"webshop", "alfworld"}:
        config["evaluation"]["max_environment_steps_by_source"] = {
            dataset_key: 10
        }
    return config


def test_runner_reuses_hotpot_graph_and_stable_zero_boundaries():
    assert _MODULE._collect_graph is _MODULE.hotpot_round._collect_graph
    assert _MODULE._stable_zero_check is _MODULE.hotpot_round._stable_zero_check
    assert (
        _MODULE._trajectory_resume_matches
        is _MODULE.hotpot_round._trajectory_resume_matches
    )


def test_supported_configs_are_evaluation_only():
    for dataset_key in (
        "hotpotqa",
        "triviaqa",
        "aime_2026",
        "healthbench_professional",
        "webshop",
        "alfworld",
    ):
        config = _evaluation_config(dataset_key)
        _MODULE.validate_completion_benchmark_config(config)

        invalid = deepcopy(config)
        invalid["grpo"]["max_optimizer_updates"] = 1
        try:
            _MODULE.validate_completion_benchmark_config(invalid)
        except Exception as exc:
            assert "optimizer_updates" in str(exc)
        else:  # pragma: no cover - fail-closed guard
            raise AssertionError("an optimizer-enabled evaluation config was accepted")


def test_graph_task_timeout_must_be_positive_when_configured():
    config = _evaluation_config("alfworld")
    config["alfworld_evaluation"]["task_timeout_seconds"] = 300
    _MODULE.validate_completion_benchmark_config(config)

    invalid = deepcopy(config)
    invalid["alfworld_evaluation"]["task_timeout_seconds"] = 0
    try:
        _MODULE.validate_completion_benchmark_config(invalid)
    except Exception as exc:
        assert "task_timeout_seconds" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("a non-positive graph task timeout was accepted")


def test_webshop_round04_production_configs_use_the_frozen_live_catalog():
    for name, split, count in (
        ("development_webshop_round_04.yaml", "train", 16),
        ("evaluation_webshop_round_04.yaml", "validation", 128),
    ):
        config = load_yaml(_ROOT / "config" / name)
        _MODULE.validate_completion_benchmark_config(config)
        section = config["webshop_evaluation"]
        assert section["split"] == split
        assert section["sample_count"] == count
        assert config["data"]["catalog_path"] == "config/datasets_webshop.yaml"
        assert config["data"]["validation_path"] == "data/webshop_v2/validation.jsonl"
        assert config["experiment"]["training_enabled"] is False
        assert config["skills"]["enabled"] is False


def test_aime_selection_filters_the_official_2026_slice(tmp_path):
    config = _evaluation_config("aime_2026")
    config["aime2026_evaluation"]["sample_count"] = 2
    source = tmp_path / "test.jsonl"
    selected = tmp_path / "selected.jsonl"
    config["data"]["test_path"] = str(source)

    def task(task_id: str, benchmark_slice: str, answer: str):
        return _MODULE.TaskRecord(
            task_id=task_id,
            question=f"question {task_id}",
            ground_truth=answer,
            split="test",
            metadata={
                "dataset_key": "aime_2026",
                "benchmark_slice": benchmark_slice,
                "evaluator_payload": {"accepted_answers": [answer]},
            },
        )

    records = (
        task("aime-historical:one", "historical_aime_1983_2025", "1"),
        task("aime-2026:01", "official_aime_2026", "2"),
        task("aime-2026:02", "official_aime_2026", "3"),
    )
    _MODULE._atomic_jsonl(
        source,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **item.to_dict()}
            for item in records
        ],
    )

    frozen = _MODULE._select_tasks(config, tmp_path, selected)

    assert [item.task_id for item in frozen] == ["aime-2026:01", "aime-2026:02"]
    assert [item.task_id for item in _MODULE.iter_task_records(selected)] == [
        "aime-2026:01",
        "aime-2026:02",
    ]


def test_selection_fails_closed_on_declared_partition_mismatch(tmp_path):
    config = _evaluation_config("hotpotqa")
    section = config["hotpotqa_evaluation"]
    section.update(
        {
            "stage": "development",
            "required_partition": "development",
            "sample_count": 2,
            "stable_zero_sample_count": 2,
        }
    )
    source = tmp_path / "validation.jsonl"
    selected = tmp_path / "selected.jsonl"
    config["data"]["validation_path"] = str(source)
    records = [
        _MODULE.TaskRecord(
            task_id=f"hotpotqa:{index}",
            question=f"question {index}",
            ground_truth=str(index),
            split="validation",
            metadata={
                "dataset_key": "hotpotqa",
                "joint_qa_partition": partition,
            },
        )
        for index, partition in enumerate(("development", "test"))
    ]
    _MODULE._atomic_jsonl(
        source,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **item.to_dict()}
            for item in records
        ],
    )

    try:
        _MODULE._select_tasks(config, tmp_path, selected)
    except Exception as exc:
        assert "belongs to partition 'test', expected 'development'" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("cross-partition task selection was accepted")


def test_final_evaluation_stage_requires_test_split():
    config = _evaluation_config("hotpotqa")
    config["hotpotqa_evaluation"]["stage"] = "final_evaluation"

    try:
        _MODULE.validate_completion_benchmark_config(config)
    except Exception as exc:
        assert "evaluation.final_split" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("final evaluation accepted a validation split")


def test_healthbench_evaluation_receives_the_backend_judge():
    calls = []

    async def judge(messages, model):
        calls.append((messages, model))
        return {"criteria_met": True, "explanation": "criterion met"}

    backend = type("Backend", (), {})()
    backend.judge = judge
    backend.judge_model = "configured-healthbench-judge"
    task = _MODULE.TaskRecord(
        task_id="healthbench-professional:one",
        question="Conversation:\n\n[user] What should I do?\n\n[assistant]",
        ground_truth="Seek medical advice.",
        split="validation",
        metadata={
            "dataset_key": "healthbench_professional",
            "evaluator_payload": {
                "rubric_items": [
                    {"criterion_text": "Recommends medical advice", "points": 5}
                ]
            },
        },
    )

    outcome = asyncio.run(
        _MODULE._evaluate_prediction(backend, task, "Seek medical advice.")
    )

    assert outcome.valid is True
    assert outcome.metrics["raw_score"] == 1.0
    assert len(calls) == 1
    assert calls[0][1] == "configured-healthbench-judge"


def test_evaluator_preflight_static_fixtures_are_synthetic_and_non_test():
    for dataset_key in (
        "hotpotqa",
        "triviaqa",
        "aime_2026",
        "healthbench_professional",
        "swe_bench",
    ):
        task, _prediction = _MODULE._synthetic_evaluator_preflight_fixture(
            dataset_key
        )

        assert task.task_id == f"evaluator-preflight:{dataset_key}:synthetic-v1"
        assert task.split == "train"
        assert task.metadata["dataset_key"] == dataset_key


def test_evaluator_preflight_receipt_drops_answer_bearing_outcome_fields():
    outcome = EvaluationOutcome(
        valid=True,
        reward=1.0,
        metrics={"exact_match": 1.0, "token_f1": 1.0},
        reason="evaluated",
        details={
            "raw_prediction": "fixture answer",
            "scored_prediction": "fixture answer",
        },
        evaluator_version="hotpotqa.official.answer.v1",
    )

    receipt = _MODULE._evaluator_preflight_receipt(outcome, "hotpotqa")

    assert receipt == {
        "passed": True,
        "evaluator_version": "hotpotqa.official.answer.v1",
    }
    assert "metrics" not in receipt
    assert "details" not in receipt
    assert "reward" not in receipt


def test_environment_preflight_uses_fixed_training_record(tmp_path):
    config = _evaluation_config("webshop")
    train_path = tmp_path / "train.jsonl"
    config["data"]["train_path"] = str(train_path)
    task = _MODULE.TaskRecord(
        task_id="webshop:train:fixture",
        question="fixed non-test environment fixture",
        ground_truth="environment_success",
        split="train",
        metadata={"dataset_key": "webshop"},
    )
    _MODULE._atomic_jsonl(
        train_path,
        [{"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}],
    )

    fixture = _MODULE._non_test_environment_preflight_task(
        config, tmp_path, "webshop"
    )

    assert fixture.task_id == "webshop:train:fixture"
    assert fixture.split == "train"


def test_swebench_evaluator_preflight_does_not_call_selected_task_harness():
    backend = SimpleNamespace(
        swe_harness=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the selected-task harness must not be used")
        )
    )

    receipt = asyncio.run(
        _MODULE._run_evaluator_preflight(
            backend,
            _evaluation_config("hotpotqa"),
            _ROOT,
            "swe_bench",
        )
    )

    assert receipt == {
        "passed": True,
        "evaluator_version": "swebench.harness.v1",
    }


def test_stable_zero_requires_dataset_evaluator_receipts():
    task = _MODULE.TaskRecord(
        task_id="aime-2026:01",
        question="question",
        ground_truth="7",
        split="test",
        metadata={"dataset_key": "aime_2026"},
    )
    direct = {
        task.task_id: {
            "evaluation": {
                "valid": True,
                "evaluator_version": "wrong-evaluator",
            }
        }
    }
    trajectory = {
        "explicit_finish": True,
        "final_answer": "<answer>7</answer>",
        "evaluation": {
            "valid": True,
            "evaluator_version": _MODULE.evaluator_version_for(task),
        },
        "turns": [
            {
                "receipt_verified": True,
                "director_attempt_count": 1,
                "director_generation_seed": 1,
                "director_latency_ms": 1.0,
                "action": {"action_type": "finish"},
                "graph_snapshot": {"output_agent_id": "actor"},
                "executions": [
                    {
                        "agent_id": "actor",
                        "execution_id": "execution-1",
                        "metadata": {
                            "request": {
                                "agent": {"agent_id": "actor"},
                                "model": {"model_id": "m"},
                                "phase": "single",
                                "upstream": [],
                                "own_draft": None,
                                "peer_draft": None,
                                "rendered_messages": [],
                            }
                        },
                    }
                ],
            }
        ],
    }

    result = _MODULE._completion_stable_zero_check(
        (task,),
        direct,
        {task.task_id: trajectory},
        dataset_key="aime_2026",
    )

    assert result["passed"] is False
    assert result["checks"][0]["direct_evaluator_valid"] is False


def test_environment_stable_zero_uses_terminal_receipt_not_free_text_answer():
    task = _MODULE.TaskRecord(
        task_id="webshop:00500",
        question="buy the requested item",
        ground_truth="environment_terminal_success",
        split="validation",
        metadata={"dataset_key": "webshop"},
    )
    evaluator_version = _MODULE.evaluator_version_for(task)
    direct = {
        task.task_id: {
            "evaluation": {
                "valid": True,
                "evaluator_version": evaluator_version,
            }
        }
    }
    trajectory = {
        "explicit_finish": True,
        "final_answer": "",
        "evaluation": {
            "valid": True,
            "evaluator_version": evaluator_version,
        },
        "turns": [
            {
                "receipt_verified": True,
                "director_attempt_count": 1,
                "director_generation_seed": 1,
                "director_latency_ms": 1.0,
                "action": {"action_type": "finish"},
                "graph_snapshot": {"output_agent_id": "actor"},
                "executions": [
                    {
                        "agent_id": "actor",
                        "execution_id": "execution-1",
                        "metadata": {
                            "request": {
                                "agent": {"agent_id": "actor"},
                                "model": {"model_id": "m"},
                                "phase": "single",
                                "upstream": [],
                                "own_draft": None,
                                "peer_draft": None,
                                "rendered_messages": [],
                            }
                        },
                    }
                ],
                "runtime_summary": {
                    "output_metadata": {
                        "actor": {
                            "environment_terminal": True,
                            "evaluator_environment_trace": [
                                {
                                    "action": "click[buy now]",
                                    "done": True,
                                    "state_advanced": True,
                                }
                            ],
                        }
                    }
                },
            }
        ],
    }

    passed = _MODULE._completion_stable_zero_check(
        (task,),
        direct,
        {task.task_id: trajectory},
        dataset_key="webshop",
    )
    assert passed["passed"] is True
    assert passed["checks"][0]["terminal_artifact_saved"] is True
    assert passed["checks"][0]["environment_terminal_receipt_valid"] is True

    trajectory["turns"][0]["runtime_summary"]["output_metadata"]["actor"][
        "environment_terminal"
    ] = False
    failed = _MODULE._completion_stable_zero_check(
        (task,),
        direct,
        {task.task_id: trajectory},
        dataset_key="webshop",
    )
    assert failed["passed"] is False
    assert failed["checks"][0]["environment_terminal_receipt_valid"] is False


def test_reports_aime_exact_match_and_healthbench_raw_score():
    aime_rows = [
        {
            "direct": {"available": True, "valid": True, "exact_match": 0.0},
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "explicit_finish": True,
            },
        },
        {
            "direct": {"available": True, "valid": True, "exact_match": 1.0},
            "agentgraph": {
                "available": False,
                "valid": False,
                "exact_match": 0.0,
                "explicit_finish": False,
            },
        },
    ]
    aime = _MODULE._aggregate(aime_rows, "agentgraph", "aime_2026")
    assert aime["denominator"] == 2
    assert aime["strict_exact_match"] == 0.5
    assert aime["completed_only_exact_match"] == 1.0
    assert aime["strict_accuracy"] == 0.5

    health_rows = [
        {
            "direct": {"available": True, "valid": True, "raw_score": 0.25},
            "agentgraph": {
                "available": True,
                "valid": True,
                "raw_score": 0.75,
                "explicit_finish": True,
            },
        },
        {
            "direct": {"available": True, "valid": True, "raw_score": 0.50},
            "agentgraph": {
                "available": True,
                "valid": True,
                "raw_score": -0.25,
                "explicit_finish": True,
            },
        },
    ]
    health = _MODULE._aggregate(
        health_rows, "agentgraph", "healthbench_professional"
    )
    assert health["denominator"] == 2
    assert health["strict_raw_score"] == 0.25
    assert health["completed_only_raw_score"] == 0.25

    environment = _MODULE._aggregate(
        [
            {
                "direct": {"available": True, "valid": True, "success": 0.0},
                "agentgraph": {
                    "available": True,
                    "valid": True,
                    "success": 1.0,
                    "explicit_finish": True,
                },
            }
        ],
        "agentgraph",
        "webshop",
    )
    assert environment["strict_success"] == 1.0


def test_qa_reports_native_exact_match_and_token_f1_together():
    rows = [
        {
            "direct": {
                "available": True,
                "valid": True,
                "exact_match": 0.0,
                "token_f1": 0.5,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 1.0,
                "explicit_finish": True,
            },
        },
        {
            "direct": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 1.0,
            },
            "agentgraph": {
                "available": False,
                "valid": False,
                "exact_match": 0.0,
                "token_f1": 0.0,
                "explicit_finish": False,
            },
        },
    ]

    result = _MODULE._aggregate(rows, "agentgraph", "hotpotqa")

    assert result["strict_exact_match"] == 0.5
    assert result["strict_token_f1"] == 0.5
    assert result["completed_only_exact_match"] == 1.0
    assert result["completed_only_token_f1"] == 1.0


def test_qa_metric_receipt_requires_both_native_metrics():
    valid, values = _MODULE._metrics(
        {
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 1.0, "token_f1": 0.75},
            }
        },
        "triviaqa",
    )
    missing_f1, missing_values = _MODULE._metrics(
        {"evaluation": {"valid": True, "metrics": {"exact_match": 1.0}}},
        "triviaqa",
    )

    assert valid is True
    assert values == {"exact_match": 1.0, "token_f1": 0.75}
    assert missing_f1 is False
    assert missing_values == {"exact_match": 0.0, "token_f1": 0.0}


def test_qa_paired_rows_preserve_exact_match_and_token_f1():
    task = _MODULE.TaskRecord(
        task_id="triviaqa:one",
        question="Who wrote Main Street?",
        ground_truth="Sinclair Lewis",
        split="validation",
        metadata={"dataset_key": "triviaqa"},
    )
    direct = {
        task.task_id: {
            "final_answer": "John Galsworthy",
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 0.0, "token_f1": 0.0},
            },
        }
    }
    trajectories = {
        task.task_id: {
            "final_answer": "Sinclair Lewis",
            "explicit_finish": True,
            "termination_reason": "explicit_finish",
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 1.0, "token_f1": 1.0},
            },
            "turns": [],
        }
    }

    row = _MODULE._paired_rows(
        (task,), direct, trajectories, "triviaqa"
    )[0]

    assert row["direct"]["exact_match"] == 0.0
    assert row["direct"]["token_f1"] == 0.0
    assert row["agentgraph"]["exact_match"] == 1.0
    assert row["agentgraph"]["token_f1"] == 1.0
    assert row["delta_exact_match"] == 1.0
    assert row["delta_token_f1"] == 1.0


def test_trivia_failure_taxonomy_uses_evaluator_and_public_runtime_receipts():
    common = {
        "explicit_finish": True,
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 0.0, "token_f1": 0.6},
            "details": {},
        },
        "turns": [],
    }
    canonicalization = {
        **common,
        "evaluation": {
            **common["evaluation"],
            "details": {
                "answer_mismatch_type": (
                    "accepted_answer_canonicalization_mismatch"
                )
            },
        },
    }
    coverage = {
        **common,
        "turns": [
            {
                "canvas_feedback": "knowledge_base_coverage_failure",
                "runtime_summary": {},
            }
        ],
    }
    strategy_failure = {
        **common,
        "explicit_finish": False,
        "turns": [
            {
                "canvas_feedback": "bounded ReAct execution failed",
                "runtime_summary": {
                    "react_public_error_summary": {
                        "terminal_failure_diagnosis": {
                            "public_error_code": "retrieval_strategy_failure"
                        }
                    }
                },
            }
        ],
    }

    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        canonicalization,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.6,
        dataset_key="triviaqa",
    ) == "accepted_answer_canonicalization_mismatch"
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        canonicalization,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "accepted_answer_canonicalization_mismatch"
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        coverage,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.6,
        dataset_key="triviaqa",
    ) == "knowledge_base_coverage_failure"
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        strategy_failure,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "retrieval_strategy_failure"

    partial_overlap_after_recovery = {
        **common,
        "evaluation": {
            **common["evaluation"],
            "details": {"answer_mismatch_type": "partial_answer_overlap"},
        },
        "turns": [
            {
                "canvas_feedback": "semantic_evidence_provenance_invalid",
                "runtime_summary": {},
            }
        ],
    }
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        partial_overlap_after_recovery,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.5,
        dataset_key="triviaqa",
    ) == "partial_answer_overlap"

    coverage_without_finish = {
        **coverage,
        "explicit_finish": False,
        "termination_reason": "max_rounds",
    }
    assert _MODULE._failure_type(
        {"evaluation": {"valid": True}},
        coverage_without_finish,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "knowledge_base_coverage_failure"


def test_trivia_failure_taxonomy_prioritizes_final_legal_lineage():
    direct = {"evaluation": {"valid": True}}
    no_overlap_evaluation = {
        "valid": True,
        "metrics": {"exact_match": 0.0, "token_f1": 0.0},
        "details": {"answer_mismatch_type": "no_accepted_answer_overlap"},
    }
    recall_failure_turn = {
        "canvas_feedback": (
            "accepted modify_agent at revision 4; "
            "execution_error=bounded ReAct execution failed"
        ),
        "runtime_summary": {
            "execution_status": "failed",
            "failure_records": [
                {
                    "request_id": "trivia:recall:reader:single",
                    "agent_id": "reader",
                    "phase": "single",
                    "graph_revision": 4,
                    "error_type": "ReactExecutionError",
                    "message": "bounded ReAct execution failed",
                    "metadata": {
                        "react_trace": [
                            {
                                "turn": 10,
                                "terminal_failure_diagnosis": {
                                    "observation_status": "budget_exhausted",
                                    "public_error_code": (
                                        "retrieval_recall_failure"
                                    ),
                                    "tool_plan_exhausted": True,
                                    "bounded_schedule_exhausted": True,
                                },
                            }
                        ],
                        "tool_plan_exhausted": True,
                    },
                }
            ],
        },
    }

    unresolved_recall = {
        "explicit_finish": False,
        "termination_reason": "max_rounds",
        "evaluation": no_overlap_evaluation,
        "turns": [recall_failure_turn],
    }
    assert _MODULE._failure_type(
        direct,
        unresolved_recall,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "retrieval_recall_failure"

    final_reasoning_failure = {
        "explicit_finish": True,
        "termination_reason": "finish",
        "evaluation": no_overlap_evaluation,
        "turns": [],
    }
    assert _MODULE._failure_type(
        direct,
        final_reasoning_failure,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "reasoning_failure"

    recovered_then_reasoning_failure = {
        **final_reasoning_failure,
        # The exact failed Runtime receipt remains losslessly available after
        # a later Canvas revision repairs retrieval and explicitly finishes.
        "turns": [recall_failure_turn],
    }
    assert _MODULE._failure_type(
        direct,
        recovered_then_reasoning_failure,
        direct_valid=True,
        graph_valid=True,
        direct_score=0.0,
        graph_score=0.0,
        dataset_key="triviaqa",
    ) == "reasoning_failure"


def test_paired_row_uses_the_evaluated_lineage_graph_for_fallback():
    task = _MODULE.TaskRecord(
        task_id="triviaqa:fallback",
        question="Who wrote Main Street?",
        ground_truth="Sinclair Lewis",
        split="validation",
        metadata={"dataset_key": "triviaqa"},
    )
    evaluated_graph = {
        "revision": 1,
        "nodes": [],
        "relations": [],
        "output_agent_id": None,
    }
    terminal_graph = {
        "revision": 2,
        "nodes": [],
        "relations": [],
        "output_agent_id": None,
    }
    direct = {
        task.task_id: {
            "final_answer": "Sinclair Lewis",
            "evaluation": {
                "valid": True,
                "metrics": {"exact_match": 1.0, "token_f1": 1.0},
            },
        }
    }
    trajectory = {
        "task": task.to_dict(),
        "trajectory_id": "trajectory:fallback",
        "final_answer": "Sinclair Lewis",
        "explicit_finish": False,
        "termination_reason": "max_rounds",
        "valid_lineage_fallback_used": True,
        "valid_lineage_fallback_receipt": {
            "graph_snapshot": evaluated_graph,
        },
        "evaluation": {
            "valid": True,
            "metrics": {"exact_match": 1.0, "token_f1": 1.0},
        },
        "turns": [
            {
                "action": {"action": "modify_agent"},
                "canvas_feedback": "accepted modify_agent at revision 2",
                "graph_snapshot": terminal_graph,
            }
        ],
    }

    row = _MODULE._paired_rows(
        (task,), direct, {task.task_id: trajectory}, "triviaqa"
    )[0]

    assert row["agentgraph"]["final_graph"] == evaluated_graph
    assert row["agentgraph"]["terminal_canvas_graph"] == terminal_graph
    assert row["agentgraph"]["graph_diagnostic"]["graph_revision"] == 1


def test_interactive_direct_condition_records_every_environment_policy_call():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )

    class Gateway:
        def __init__(self):
            self.calls = 0

        async def generate(self, request):
            self.calls += 1
            return SimpleNamespace(text=f"click[action-{self.calls}]")

    gateway = Gateway()
    backend = SimpleNamespace(
        registry=registry,
        runtime=SimpleNamespace(gateway=gateway),
        config={"evaluation": {"max_environment_steps": 10}},
    )
    task = _MODULE.TaskRecord(
        task_id="webshop:00001",
        question="buy an item",
        ground_truth="environment_success",
        split="validation",
        metadata={"dataset_key": "webshop"},
    )
    execution_index = 0

    class Execution:
        def __init__(self, output):
            nonlocal execution_index
            execution_index += 1
            self.output = output
            self.metadata = {"response": {"generation_seed": 17}}

        def to_dict(self):
            return {
                "output": self.output,
                "metadata": self.metadata,
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 1.0,
            }

    async def fake_evaluate(_backend, _task, _prediction, *, run_graph=None):
        assert run_graph is not None
        await run_graph("environment step one")
        await run_graph("environment step two")
        return EvaluationOutcome(
            valid=True,
            reward=1.0,
            metrics={"success": 1.0},
            reason="evaluated",
            evaluator_version="skillflow.ragen_adapter.v1",
        )

    with patch.object(
        _MODULE,
        "execution_record_from_call",
        side_effect=lambda call: Execution(call.response.text),
    ), patch.object(_MODULE, "_evaluate_prediction", new=fake_evaluate):
        result = asyncio.run(
            _MODULE._direct_one(
                backend,
                task,
                0,
                model_id="qwen3.5-9b-local",
                protocol="skillflow_ragen_webshop_react_v1",
                contract="Return one legal action.",
                seed=17,
                run_label="webshop-test",
            )
        )

    assert len(result["executions"]) == 2
    assert result["final_answer"] == "click[action-2]"
    assert result["evaluation"]["metrics"]["success"] == 1.0


def test_swe_direct_condition_is_one_coding_agent_with_repository_tools():
    registry = load_model_registry(
        _ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml"
    )
    observed = {}
    closed = []

    class Runtime:
        async def execute(self, graph, problem, *, run_id):
            node = graph.nodes[0]
            observed.update(
                problem=problem,
                run_id=run_id,
                node=node,
                output_agent_id=graph.output_agent_id,
            )
            call = SimpleNamespace(
                response=SimpleNamespace(
                    metadata={
                        "model_calls": [
                            {"metadata": {"generation_seed": 23}}
                        ]
                    }
                )
            )
            return SimpleNamespace(
                final_answer="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py",
                calls=(call,),
                run_id=run_id,
                output_agent_id=node.id,
                block_completion_order=((node.id,),),
                executed_agent_ids=(node.id,),
            )

    backend = SimpleNamespace(
        registry=registry,
        config={
            "experiment": {"condition_id": "swe-coding"},
            "swebench_evaluation": {
                "dataset_key": "swe_bench",
                "direct_completion_condition": (
                    "Inspect, edit, run tests, inspect a changed diff, then complete."
                ),
            },
            "swe_coding_runtime": {"enabled": True},
        },
        _runtime_for_task=lambda task, condition_id: (
            Runtime(),
            SimpleNamespace(resource_ids=("swebench_repository",)),
            lambda: closed.append(True),
        ),
    )
    task = _MODULE.TaskRecord(
        task_id="swe_bench:one",
        question="Fix the public issue description.",
        ground_truth="official_harness_only",
        split="validation",
        metadata={"dataset_key": "swe_bench"},
    )

    async def fake_evaluate(_backend, _task, prediction, *, run_graph=None):
        assert prediction.startswith("diff --git")
        assert run_graph is None
        return EvaluationOutcome(
            valid=True,
            reward=1.0,
            metrics={"resolved": 1.0},
            reason="official harness",
            evaluator_version="swebench.harness.v1",
        )

    with patch.object(
        _MODULE,
        "execution_record_from_call",
        return_value=SimpleNamespace(
            to_dict=lambda: {
                "output": "diff --git a/a.py b/a.py",
                "metadata": {"response": {"attempt_count": 1}},
            }
        ),
    ), patch.object(_MODULE, "_evaluate_prediction", new=fake_evaluate):
        result = asyncio.run(
            _MODULE._direct_one(
                backend,
                task,
                0,
                model_id="qwen3.5-9b-local",
                protocol="single_coding_agent_v1",
                contract="Use repository tools to resolve the issue.",
                seed=23,
                run_label="test",
            )
        )

    node = observed["node"]
    assert node.execution_mode.value == "coding"
    assert node.allowed_tools == ("swebench_repository",)
    assert observed["output_agent_id"] == "direct_coding_agent"
    assert result["simple_baseline_topology"] == "single_coding_agent"
    assert result["final_answer"].startswith("diff --git")
    assert closed == [True]
