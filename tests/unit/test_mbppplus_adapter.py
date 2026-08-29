from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.interactive.mbppplus_adapter import (
    MBPPPLUS_EVALUATOR_VERSION,
    MBPPPlusEvaluatorUnavailable,
    MBPPPlusOfficialEvaluator,
)
from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import evaluate_task


def record(task_id: str, *, selected_id: str | None = None) -> TaskRecord:
    official_id = selected_id or task_id
    return TaskRecord(
        task_id=task_id,
        question="Write the requested Python function.",
        ground_truth="",
        split="test",
        metadata={
            "dataset_key": "mbpp_plus",
            "evaluator_payload": {"task_id": official_id},
        },
    )


class FakeEvalPlus:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.pass_status = "pass"
        self.output_not_none_tasks = ()
        self.runtime_version = "0.3.1"
        self.dataset_version = "v0.2.0"

    @staticmethod
    def sanitize(*, code: str, entrypoint: str) -> str:
        return (
            code.replace("```python", "")
            .replace("```", "")
            .strip()
        )

    @staticmethod
    def get_groundtruth(problems, cache_key, output_not_none_tasks):
        return {
            task_id: {
                "base": ["hidden-base-output"],
                "base_time": [0.01],
                "plus": ["hidden-plus-output"],
                "plus_time": [0.01],
            }
            for task_id in problems
        }

    def check_correctness(
        self,
        dataset,
        completion_id,
        problem,
        solution,
        expected_output,
        *,
        base_only,
        fast_check,
        identifier,
    ):
        self.calls.append(
            {
                "dataset": dataset,
                "completion_id": completion_id,
                "task_id": problem["task_id"],
                "solution": solution,
                "base_only": base_only,
                "fast_check": fast_check,
                "identifier": identifier,
            }
        )
        if "return 1" in solution:
            return {
                "base": ("pass", [{"hidden_input": "must-not-leak"}]),
                "plus": ("pass", [{"hidden_input": "must-not-leak"}]),
            }
        if "return 2" in solution:
            return {
                "base": ("pass", [{"hidden_input": "must-not-leak"}]),
                "plus": ("fail", [{"hidden_input": "must-not-leak"}]),
            }
        if "return 3" in solution:
            return {
                "base": ("fail", [{"hidden_input": "must-not-leak"}]),
                "plus": ("pass", [{"hidden_input": "must-not-leak"}]),
            }
        return {
            "base": ("fail", [{"hidden_input": "must-not-leak"}]),
            "plus": ("fail", [{"hidden_input": "must-not-leak"}]),
        }


class MBPPPlusOfficialEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    def backend(
        self,
        root: Path,
        *,
        selected_task_ids: tuple[str, ...] = ("Mbpp/2",),
    ) -> tuple[MBPPPlusOfficialEvaluator, FakeEvalPlus]:
        backend = MBPPPlusOfficialEvaluator(
            runtime_path=root / "runtime",
            dataset_path=root / "MbppPlus-v0.2.0.jsonl",
            cache_root=root / "cache",
            selected_task_ids=selected_task_ids,
        )
        fake = FakeEvalPlus()
        backend._runtime = fake  # type: ignore[assignment]
        backend._problems = {
            "Mbpp/2": {
                "task_id": "Mbpp/2",
                "entry_point": "solve",
                "canonical_solution": "def solve():\n    return 1\n",
            },
            "Mbpp/3": {
                "task_id": "Mbpp/3",
                "entry_point": "solve",
                "canonical_solution": "def solve():\n    return 1  # private-preflight\n",
            },
        }
        return backend, fake

    async def test_selected_task_uses_official_base_and_plus_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, fake = self.backend(Path(directory))
            result = await backend.evaluate(
                record("mbpp_plus:Mbpp/2", selected_id="Mbpp/2"),
                "```python\ndef solve():\n    return 1\n```",
            )

        self.assertEqual("pass", result["base_status"])
        self.assertEqual("pass", result["plus_status"])
        self.assertTrue(result["base_passed"])
        self.assertTrue(result["plus_passed"])
        self.assertEqual(1.0, result["pass_at_1"])
        self.assertTrue(result["format_diagnostics"]["sanitization_changed"])
        self.assertNotIn("hidden", repr(result).lower())
        self.assertEqual("mbpp", fake.calls[0]["dataset"])
        self.assertFalse(fake.calls[0]["base_only"])
        self.assertTrue(fake.calls[0]["fast_check"])

    async def test_plus_failure_is_zero_and_hidden_details_are_not_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _ = self.backend(Path(directory))
            result = await backend.evaluate(
                record("Mbpp/2"),
                "def solve():\n    return 2\n",
            )

        self.assertTrue(result["base_passed"])
        self.assertFalse(result["plus_passed"])
        self.assertEqual(0.0, result["pass_at_1"])
        self.assertNotIn("hidden", repr(result).lower())

    async def test_plus_status_alone_defines_mbpp_plus_pass_at_1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _ = self.backend(Path(directory))
            result = await backend.evaluate(
                record("Mbpp/2"),
                "def solve():\n    return 3\n",
            )

        self.assertFalse(result["base_passed"])
        self.assertTrue(result["plus_passed"])
        self.assertEqual(1.0, result["pass_at_1"])

    async def test_evaluate_rejects_task_outside_frozen_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, _ = self.backend(Path(directory))
            with self.assertRaisesRegex(
                MBPPPlusEvaluatorUnavailable,
                "outside the frozen evaluation selection",
            ):
                await backend.evaluate(record("Mbpp/3"), "def solve(): return 1")

    async def test_preflight_uses_disjoint_canonical_task_without_leaking_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend, fake = self.backend(Path(directory))
            result = await backend.preflight()

        self.assertTrue(result["ready"])
        self.assertEqual("Mbpp/3", result["task_id"])
        self.assertTrue(result["selection_disjoint"])
        self.assertEqual(1, result["selected_task_count"])
        self.assertEqual("0.3.1", result["runtime_version"])
        self.assertEqual("v0.2.0", result["dataset_version"])
        self.assertNotIn("canonical", repr(result).lower())
        self.assertNotIn("private-preflight", repr(result))
        self.assertEqual("Mbpp/3", fake.calls[0]["task_id"])


class MBPPPlusTaskEvaluatorDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_reports_official_pass_at_1_metrics(self) -> None:
        async def harness(task, prediction):
            return {
                "task_id": "Mbpp/2",
                "base_status": "pass",
                "plus_status": "pass",
                "base_passed": True,
                "plus_passed": True,
                "pass_at_1": 1.0,
                "format_diagnostics": {
                    "sanitization_changed": True,
                    "entry_point": "solve",
                    "hidden_inputs": ["must-not-pass-through"],
                },
                "evaluator_protocol": MBPPPLUS_EVALUATOR_VERSION,
                "runtime_version": "0.3.1",
                "dataset_version": "v0.2.0",
            }

        outcome = await evaluate_task(
            record("Mbpp/2"),
            "def solve(): return 1",
            mbppplus_harness=harness,
        )

        self.assertTrue(outcome.valid)
        self.assertEqual(1.0, outcome.reward)
        self.assertEqual(1.0, outcome.metrics["base_pass_at_1"])
        self.assertEqual(1.0, outcome.metrics["pass_at_1"])
        self.assertEqual(MBPPPLUS_EVALUATOR_VERSION, outcome.evaluator_version)
        self.assertNotIn(
            "hidden_inputs",
            outcome.details["format_diagnostics"],
        )

    async def test_dispatch_uses_plus_status_without_base_containment_assumption(
        self,
    ) -> None:
        async def harness(task, prediction):
            return {
                "task_id": "Mbpp/2",
                "base_status": "fail",
                "plus_status": "pass",
                "base_passed": False,
                "plus_passed": True,
                "pass_at_1": 1.0,
            }

        outcome = await evaluate_task(
            record("Mbpp/2"),
            "def solve(): return 3",
            mbppplus_harness=harness,
        )

        self.assertTrue(outcome.valid)
        self.assertEqual(0.0, outcome.metrics["base_pass_at_1"])
        self.assertEqual(1.0, outcome.metrics["pass_at_1"])

    async def test_missing_callback_and_inconsistent_status_fail_closed(self) -> None:
        missing = await evaluate_task(
            record("Mbpp/2"),
            "def solve(): return 1",
        )

        async def inconsistent(task, prediction):
            return {
                "task_id": "Mbpp/2",
                "base_status": "pass",
                "plus_status": "fail",
                "base_passed": True,
                "plus_passed": False,
                "pass_at_1": 1.0,
            }

        inconsistent_result = await evaluate_task(
            record("Mbpp/2"),
            "def solve(): return 1",
            mbppplus_harness=inconsistent,
        )

        self.assertFalse(missing.valid)
        self.assertIsNone(missing.reward)
        self.assertEqual("mbppplus_harness_unavailable", missing.reason)
        self.assertFalse(inconsistent_result.valid)
        self.assertEqual(
            "mbppplus_harness_result_inconsistent",
            inconsistent_result.reason,
        )


if __name__ == "__main__":
    unittest.main()
