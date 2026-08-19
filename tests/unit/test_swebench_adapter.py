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
    SWEbenchHarnessUnavailable,
)


def record(instance_id: str = "owner__repo-1") -> TaskRecord:
    return TaskRecord(
        task_id=f"swe-bench:{instance_id}",
        question="Fix the issue",
        ground_truth="",
        split="validation",
        metadata={
            "dataset_key": "swe_bench",
            "evaluator_payload": {"instance_id": instance_id},
        },
    )


class OfficialSWEbenchHarnessTests(unittest.IsolatedAsyncioTestCase):
    def harness(self, root: Path) -> OfficialSWEbenchHarness:
        evaluator = root / "swe_bench_eval.py"
        evaluator.write_text("# fake upstream\n", encoding="utf-8")
        harness = root / "harness"
        verified = root / "verified"
        harness.mkdir()
        verified.mkdir()
        return OfficialSWEbenchHarness(
            evaluator_path=evaluator,
            harness_path=harness,
            verified_path=verified,
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
            evaluate_patch=evaluate_patch,
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(Path(directory))
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
                    harness.preflight(["owner__repo-1"])

    def test_preflight_rejects_instances_outside_verified(self) -> None:
        module = SimpleNamespace(
            VERIFIED_DS_PATH="",
            _verified_cache={"another-instance": {}},
            _load_verified_dataset=lambda: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = self.harness(Path(directory))
            with patch(
                "src.interactive.swebench_adapter._load_skillflow_evaluator",
                return_value=module,
            ):
                with self.assertRaises(SWEbenchHarnessUnavailable):
                    harness.preflight(["owner__repo-1"])


if __name__ == "__main__":
    unittest.main()
