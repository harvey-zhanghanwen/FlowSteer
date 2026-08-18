from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path

from src.interactive.config_loader import load_yaml


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
    if dataset_key == "aime_2026":
        section_name = "aime2026_evaluation"
        phase = "aime2026_evaluation"
        split = "test"
        sample_count = 30
        extra = {
            "official_2026_only": True,
            "benchmark_slice": "official_aime_2026",
        }
    else:
        section_name = "healthbench_professional_evaluation"
        phase = "healthbench_professional_evaluation"
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
    return config


def test_runner_reuses_hotpot_graph_and_stable_zero_boundaries():
    assert _MODULE._collect_graph is _MODULE.hotpot_round._collect_graph
    assert _MODULE._stable_zero_check is _MODULE.hotpot_round._stable_zero_check
    assert (
        _MODULE._trajectory_resume_matches
        is _MODULE.hotpot_round._trajectory_resume_matches
    )


def test_aime_and_health_configs_are_evaluation_only():
    for dataset_key in ("aime_2026", "healthbench_professional"):
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


def test_alfworld_round01_configs_are_interactive_evaluation_only():
    for name, split, sample_count in (
        ("development_alfworld_round_01.yaml", "train", 16),
        ("evaluation_alfworld_round_01.yaml", "validation", 128),
    ):
        config = load_yaml(_ROOT / "config" / name)
        _MODULE.validate_completion_benchmark_config(config)
        section = config["alfworld_evaluation"]
        assert section["split"] == split
        assert section["sample_count"] == sample_count
        assert config["data"]["catalog_path"] == "config/datasets_alfworld.yaml"
        assert config["evaluation"]["max_environment_steps_by_source"]["alfworld"] == 50
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
                "execution_records": [
                    {
                        "is_output_agent": True,
                        "input": {"inbox": []},
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
