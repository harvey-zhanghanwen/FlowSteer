from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.interactive.config_loader import load_yaml
from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import MBPPPLUS_EVALUATOR_VERSION


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_mbppplus_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _task(number: int = 2) -> TaskRecord:
    return TaskRecord(
        task_id=f"mbpp-plus:Mbpp/{number}",
        question="Write a function that returns the common elements of two lists.",
        ground_truth=None,
        split="test",
        metadata={
            "dataset_key": "mbpp_plus",
            "source_task_id": f"Mbpp/{number}",
            "entry_point": "similar_elements",
            "ground_truth_role": "evaluator_only_redacted",
        },
    )


def test_mbppplus_config_is_evaluation_only_and_bounded_to_fixed_100():
    config = load_yaml(_ROOT / "config" / "evaluation_mbppplus_initial_v1.yaml")

    _MODULE.validate_completion_benchmark_config(config)
    assert config["mbppplus_evaluation"]["sample_count"] == 100
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["skills"]["enabled"] is False

    invalid = deepcopy(config)
    invalid["mbppplus_evaluation"]["sample_count"] = 101
    try:
        _MODULE.validate_completion_benchmark_config(invalid)
    except Exception as exc:
        assert "between 1 and 100" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("MBPP+ selection larger than fixed-100 was accepted")

    missing_evaluator = deepcopy(config)
    missing_evaluator["evaluation"]["evalplus_enabled"] = False
    try:
        _MODULE.validate_completion_benchmark_config(missing_evaluator)
    except Exception as exc:
        assert "evalplus_enabled" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("MBPP+ config without official EvalPlus was accepted")


def test_mbppplus_workflow_problem_only_specifies_terminal_artifact():
    problem = _MODULE._workflow_problem(_task(), {})

    assert "complete Python source file" in problem
    assert "`similar_elements`" in problem
    assert "repository patch" in problem
    for fixed_role in ("Coder", "Reviewer", "Tester"):
        assert fixed_role not in problem
    assert "Coder -> Reviewer -> Tester" not in problem


def test_mbppplus_evaluator_version_and_callback_are_terminal_only():
    task = _task()

    async def official_callback(record, prediction):
        assert record is task
        assert "def similar_elements" in prediction
        return {
            "task_id": "Mbpp/2",
            "base_status": "pass",
            "plus_status": "pass",
            "base_passed": True,
            "plus_passed": True,
            "pass_at_1": 1.0,
            "format_diagnostics": {"entry_point": "similar_elements"},
            "evaluator_protocol": MBPPPLUS_EVALUATOR_VERSION,
            "runtime_version": "0.3.1",
            "dataset_version": "v0.2.0",
        }

    backend = SimpleNamespace(
        config={"evaluation": {"max_environment_steps": 1}},
        mbppplus_harness=official_callback,
    )
    outcome = asyncio.run(
        _MODULE._evaluate_prediction(
            backend,
            task,
            "def similar_elements(test_tup1, test_tup2):\n    return tuple(set(test_tup1) & set(test_tup2))\n",
        )
    )

    assert _MODULE.evaluator_version_for(task) == MBPPPLUS_EVALUATOR_VERSION
    assert outcome.valid is True
    assert outcome.metrics == {"base_pass_at_1": 1.0, "pass_at_1": 1.0}


def test_mbppplus_attach_freezes_official_ids_and_preflight_redacts_trusted_data(tmp_path):
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def evaluate(self, record, prediction):  # pragma: no cover
            raise AssertionError("preflight must not evaluate a selected task")

        async def preflight(self):
            return {
                "ready": True,
                "task_id": "Mbpp/999",
                "base_passed": True,
                "plus_passed": True,
                "pass_at_1": 1.0,
                "selection_disjoint": True,
                "selected_task_count": 1,
                "evaluator_protocol": MBPPPLUS_EVALUATOR_VERSION,
                "runtime_version": "0.3.1",
                "dataset_version": "v0.2.0",
                "canonical_solution": "must-not-leak",
                "plus_input": ["must-not-leak"],
            }

    config = {
        "evaluation": {
            "evalplus_runtime_path": "runtime",
            "evalplus_dataset_path": "MbppPlus.jsonl",
            "evalplus_cache_root": "cache",
        }
    }
    backend = SimpleNamespace()
    with patch(
        "src.interactive.mbppplus_adapter.MBPPPlusOfficialEvaluator",
        FakeEvaluator,
    ):
        receipt = _MODULE._attach_mbppplus_official_evaluator(
            backend,
            config,
            tmp_path,
            (_task(),),
        )

    assert captured["selected_task_ids"] == ("Mbpp/2",)
    assert backend.mbppplus_harness == backend.mbppplus_evaluator.evaluate
    assert receipt["hidden_tests_model_visible"] is False
    public_preflight = asyncio.run(
        _MODULE._run_evaluator_preflight(
            backend,
            config,
            tmp_path,
            "mbpp_plus",
        )
    )
    assert public_preflight["passed"] is True
    assert "canonical_solution" not in public_preflight
    assert "plus_input" not in public_preflight
    assert "task_id" not in public_preflight


def test_mbppplus_failure_receipt_preserves_python_source_artifact():
    task = _task()
    prediction = "def similar_elements(a, b):\n    return ()\n"

    outcome = _MODULE._evaluator_runtime_failure_outcome(
        task,
        prediction,
        RuntimeError("evaluator unavailable"),
    )

    artifact = outcome.details["terminal_artifact"]
    assert artifact == {
        "kind": "python_source",
        "source": "Output Agent terminal text",
        "python_source": prediction,
        "non_empty": True,
    }
