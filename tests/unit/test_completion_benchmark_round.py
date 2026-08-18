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
    if dataset_key == "aime_2026":
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
    if dataset_key in {"webshop"}:
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
        "aime_2026",
        "healthbench_professional",
        "webshop",
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
