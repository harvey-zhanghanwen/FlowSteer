from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interactive import mbpp_training_adapter
from src.interactive.mbpp_training_adapter import (
    MBPP_TRAINING_EVALUATOR_VERSION,
    SkillFlowRewardUnavailable,
    evaluate_mbpp_public_tests,
)
from src.interactive.records import TaskRecord


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="mbpp-training:601",
        question=(
            "Write a function that increments a number.\n\n"
            "assert increment(1) == 2\n"
            "assert increment(-1) == 0\n"
        ),
        ground_truth=None,
        split="train",
        metadata={
            "dataset_key": "mbpp_plus",
            "entry_point": "increment",
            "training_population": "flowsteer_mbpp_train",
        },
    )


def test_public_test_reward_matches_deployed_skillflow_pass_rate() -> None:
    outcome = evaluate_mbpp_public_tests(
        _task(),
        "def increment(value):\n    return value + 1\n",
        timeout_seconds=2.0,
    )

    assert outcome.valid is True
    assert outcome.reward == 1.0
    assert outcome.metrics == {"pass_at_1": 1.0, "public_test_pass_rate": 1.0}
    assert outcome.evaluator_version == MBPP_TRAINING_EVALUATOR_VERSION


def test_adapter_calls_skillflow_function_with_only_public_task_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def code_test_pass_rate(prediction: str, tests: list[str]) -> float:
        observed["prediction"] = prediction
        observed["tests"] = tests
        return 0.5

    monkeypatch.setattr(
        mbpp_training_adapter,
        "_load_skillflow_reward",
        lambda _path: SimpleNamespace(code_test_pass_rate=code_test_pass_rate),
    )
    task = _task()
    task = TaskRecord(
        task_id=task.task_id,
        question=task.question,
        ground_truth="PRIVATE_REFERENCE_MUST_NOT_BE_USED",
        split=task.split,
        metadata={
            **dict(task.metadata),
            "evaluator_payload": {"hidden_tests": ["PRIVATE_HIDDEN_TEST"]},
        },
    )

    outcome = evaluate_mbpp_public_tests(
        task,
        "def increment(value):\n    return value\n",
        timeout_seconds=2.0,
    )

    assert observed == {
        "prediction": "def increment(value):\n    return value\n",
        "tests": ["assert increment(1) == 2", "assert increment(-1) == 0"],
    }
    assert outcome.reward == 0.5
    assert "PRIVATE_REFERENCE" not in repr(outcome)
    assert "PRIVATE_HIDDEN_TEST" not in repr(outcome)


def test_public_test_reward_keeps_failed_terminal_trajectory() -> None:
    outcome = evaluate_mbpp_public_tests(
        _task(),
        "def increment(value):\n    return value\n",
        timeout_seconds=2.0,
    )

    assert outcome.valid is True
    assert outcome.reward == 0.0
    assert outcome.metrics["pass_at_1"] == 0.0
    assert outcome.details["passed"] == 0
    assert outcome.details["total"] == 2


def test_empty_max_round_artifact_is_valid_zero_reward() -> None:
    outcome = evaluate_mbpp_public_tests(
        _task(),
        "",
        timeout_seconds=2.0,
    )

    assert outcome.valid is True
    assert outcome.reward == 0.0
    assert outcome.details["total"] == 2


def test_multiline_public_test_program_matches_skillflow_line_splitting() -> None:
    task = TaskRecord(
        task_id="mbpp-training:927",
        question="Write max_height.\n",
        ground_truth=None,
        split="train",
        metadata={
            "dataset_key": "mbpp_plus",
            "entry_point": "max_height",
            "training_population": "flowsteer_mbpp_train",
            "public_test_program": (
                "tree = [1, [2], [3]]\n"
                "assert max_height(tree) == 2\n"
            ),
        },
    )

    outcome = evaluate_mbpp_public_tests(
        task,
        "def max_height(tree):\n    return 2\n",
        timeout_seconds=2.0,
    )

    assert outcome.valid is True
    # This intentionally follows SkillFlow training/reward.py: each non-empty
    # line is passed as an independent test case. The setup line passes and the
    # assertion lacks that setup in its fresh namespace.
    assert outcome.reward == 0.5
    assert outcome.details["total"] == 2


def test_missing_upstream_function_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mbpp_training_adapter,
        "_load_skillflow_reward",
        lambda _path: SimpleNamespace(),
    )

    with pytest.raises(
        SkillFlowRewardUnavailable,
        match="no callable code_test_pass_rate",
    ):
        evaluate_mbpp_public_tests(
            _task(),
            "def increment(value):\n    return value + 1\n",
            timeout_seconds=2.0,
        )
