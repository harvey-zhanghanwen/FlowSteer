from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.interactive.records import TaskRecord
from src.interactive.swebench_adapter import (
    OfficialSWEbenchHarness,
    SWEBENCH_DATASET_SOURCE_REGULAR_DEV,
    SWEBENCH_DATASET_SOURCE_VERIFIED,
    SWEbenchHarnessUnavailable,
)


def record(
    instance_id: str = "owner__repo-1",
    *,
    dataset_source: str = SWEBENCH_DATASET_SOURCE_VERIFIED,
) -> TaskRecord:
    return TaskRecord(
        task_id=f"swe-bench:{instance_id}",
        question="Fix the issue",
        ground_truth="",
        split=(
            "validation"
            if dataset_source == SWEBENCH_DATASET_SOURCE_REGULAR_DEV
            else "test"
        ),
        metadata={
            "dataset_key": "swe_bench",
            "dataset_source": dataset_source,
            "benchmark_slice": dataset_source,
            "evaluator_payload": {
                "instance_id": instance_id,
                "repo": "owner/repo",
                "base_commit": "base-commit",
                "test_patch": "test patch",
                "version": "1.0",
                "FAIL_TO_PASS": ["test_target"],
                "PASS_TO_PASS": [],
                "environment_setup_commit": "environment-commit",
                "dataset_source": dataset_source,
            },
        },
    )


class OfficialSWEbenchHarnessTests(unittest.IsolatedAsyncioTestCase):
    def harness(
        self,
        root: Path,
        *,
        dataset_source: str = SWEBENCH_DATASET_SOURCE_VERIFIED,
    ) -> OfficialSWEbenchHarness:
        evaluator = root / "swe_bench_eval.py"
        evaluator.write_text("# fake upstream\n", encoding="utf-8")
        harness = root / "harness"
        dataset = root / "verified"
        harness.mkdir()
        if dataset_source == SWEBENCH_DATASET_SOURCE_VERIFIED:
            dataset.mkdir()
        else:
            dataset.write_text("{}\n", encoding="utf-8")
        return OfficialSWEbenchHarness(
            evaluator_path=evaluator,
            harness_path=harness,
            dataset_source=dataset_source,
            dataset_path=dataset,
            evaluation_root=root / "evaluation",
            timeout_seconds=17,
        )

    async def test_callback_delegates_to_skillflow_official_evaluator(self) -> None:
        calls: list[tuple[str, str, int]] = []

        def evaluate_patch(instance_id, prediction, *, timeout):
            calls.append((instance_id, prediction, timeout))
            return True, 1.0, "resolved"

        module = SimpleNamespace(
            VERIFIED_DS_PATH="",
            _verified_cache={"owner__repo-1": {}},
            _load_verified_dataset=lambda: None,
            _flowsteer_dataset_source_receipt=(
                SWEBENCH_DATASET_SOURCE_VERIFIED,
                "unused",
            ),
            evaluate_patch=evaluate_patch,
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(Path(directory))
            module._flowsteer_dataset_source_receipt = (
                SWEBENCH_DATASET_SOURCE_VERIFIED,
                str((Path(directory) / "verified").resolve()),
            )
            with patch(
                "src.interactive.swebench_adapter._load_skillflow_evaluator",
                return_value=module,
            ):
                outcome = await harness(record(), "diff --git a/x b/x")

        self.assertEqual([("owner__repo-1", "diff --git a/x b/x", 17)], calls)
        self.assertTrue(outcome["resolved"])
        self.assertEqual(1.0, outcome["official_score"])
        self.assertFalse(outcome["proxy_metric_used"])

    def test_preflight_fails_closed_when_docker_is_unavailable(self) -> None:
        module = SimpleNamespace(
            VERIFIED_DS_PATH="",
            _verified_cache={"owner__repo-1": {}},
            _load_verified_dataset=lambda: None,
        )
        docker_module = ModuleType("docker")
        docker_module.from_env = lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError("daemon unavailable")
        )
        swebench_module = ModuleType("swebench")
        swebench_module.__path__ = []
        harness_module = ModuleType("swebench.harness")
        harness_module.run_evaluation = object()

        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(Path(directory))
            module._flowsteer_dataset_source_receipt = (
                SWEBENCH_DATASET_SOURCE_VERIFIED,
                str((Path(directory) / "verified").resolve()),
            )
            with (
                patch(
                    "src.interactive.swebench_adapter._load_skillflow_evaluator",
                    return_value=module,
                ),
                patch.dict(
                    sys.modules,
                    {
                        "docker": docker_module,
                        "swebench": swebench_module,
                        "swebench.harness": harness_module,
                    },
                ),
            ):
                with self.assertRaises(SWEbenchHarnessUnavailable):
                    harness.preflight([record()])

    def test_preflight_rejects_instances_outside_verified(self) -> None:
        module = SimpleNamespace(
            VERIFIED_DS_PATH="",
            _verified_cache={"another-instance": {}},
            _load_verified_dataset=lambda: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(Path(directory))
            module._flowsteer_dataset_source_receipt = (
                SWEBENCH_DATASET_SOURCE_VERIFIED,
                str((Path(directory) / "verified").resolve()),
            )
            with patch(
                "src.interactive.swebench_adapter._load_skillflow_evaluator",
                return_value=module,
            ):
                with self.assertRaises(SWEbenchHarnessUnavailable):
                    harness.preflight([record()])

    async def test_regular_dev_primes_skillflow_cache_from_evaluator_payload(
        self,
    ) -> None:
        calls: list[tuple[str, str, int]] = []

        def evaluate_patch(instance_id, prediction, *, timeout):
            calls.append((instance_id, prediction, timeout))
            assert module._verified_cache[instance_id]["repo"] == "owner/repo"
            assert module._verified_cache[instance_id]["patch"] == "gold patch"
            return False, 0.0, "unresolved"

        module = SimpleNamespace(
            VERIFIED_DS_PATH="",
            _verified_cache={},
            evaluate_patch=evaluate_patch,
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(
                Path(directory),
                dataset_source=SWEBENCH_DATASET_SOURCE_REGULAR_DEV,
            )
            task = record(dataset_source=SWEBENCH_DATASET_SOURCE_REGULAR_DEV)
            task = TaskRecord(
                task_id=task.task_id,
                question=task.question,
                ground_truth="gold patch",
                split=task.split,
                metadata=task.metadata,
            )
            with patch(
                "src.interactive.swebench_adapter._load_skillflow_evaluator",
                return_value=module,
            ):
                outcome = await harness(task, "model patch")

        self.assertEqual([("owner__repo-1", "model patch", 17)], calls)
        self.assertFalse(outcome["resolved"])

    async def test_dataset_source_mismatch_fails_before_upstream_evaluation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(Path(directory))
            with self.assertRaises(SWEbenchHarnessUnavailable):
                await harness(
                    record(dataset_source=SWEBENCH_DATASET_SOURCE_REGULAR_DEV),
                    "model patch",
                )


if __name__ == "__main__":
    unittest.main()
