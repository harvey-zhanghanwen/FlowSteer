from __future__ import annotations

from collections import Counter
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.interactive.records import EvaluationReceipt, TaskRecord, TrajectoryRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_SCRIPT = SCRIPTS_DIR / "evaluate_agentgraph_architecture.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_agentgraph_architecture", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

EXPECTED_SOURCE_ORDER = _MODULE.EXPECTED_SOURCE_ORDER
run_architecture_evaluation = _MODULE.run_architecture_evaluation
architecture_report = _MODULE._report


SOURCE_NAMES = {
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "aime_2026": "AIME 2026",
    "healthbench_professional": "HealthBench Professional",
    "webshop": "WebShop",
    "alfworld": "ALFWorld",
    "swe_bench": "SWE-bench",
}


def make_task(source: str, index: int) -> TaskRecord:
    task_id = f"{source}:{index}"
    return TaskRecord(
        task_id=task_id,
        question=f"Question {source} {index}?",
        ground_truth="answer",
        split="train",
        metadata={
            "dataset_key": source,
            "source": SOURCE_NAMES[source],
            "sampling": {"base_task_id": task_id},
        },
    )


def aligned_row(task: TaskRecord) -> dict:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        **task.to_dict(),
    }


def make_trajectory(task: TaskRecord, rollout_index: int, versions, *, valid: bool):
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{rollout_index}",
        task=task,
        group_id=f"{task.task_id}:architecture:{versions.policy}",
        condition_id="natural_architecture_iteration_02",
        rollout_id=f"rollout:{rollout_index}",
        versions=versions,
        turns=(),
        final_answer="answer" if valid else None,
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=valid,
            reward=1.0 if valid else None,
            metrics={"score": 1.0} if valid else {},
            reason="" if valid else "synthetic_evaluator_unavailable",
        ),
        termination_reason="finish" if valid else "evaluator_unavailable",
        explicit_finish=valid,
    )


class FakeBackend:
    model_catalog_version = "catalog-test-v1"

    def __init__(self, *, fail_sources=("triviaqa",)) -> None:
        self.collected_task_ids: list[str] = []
        self.train_calls = 0
        self.publish_calls = 0
        self.fail_sources = frozenset(fail_sources)

    async def collect(self, task, rollout_index, versions):
        self.collected_task_ids.append(task.task_id)
        source = task.metadata["dataset_key"]
        if source in self.fail_sources:
            raise RuntimeError("synthetic collection failure")
        return make_trajectory(
            task,
            rollout_index,
            versions,
            valid=source != "swe_bench",
        )

    def train(self, trajectories, output_dir):
        self.train_calls += 1
        raise AssertionError("architecture evaluation must not train")

    async def publish(self, summary):
        self.publish_calls += 1
        raise AssertionError("architecture evaluation must not publish")


def create_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()

    source_config = PROJECT_ROOT / "config" / "evaluation_agentgraph_architecture.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["data"]["train_path"] = "data/train.jsonl"
    config["storage"].update(
        selected_tasks_path="artifacts/data/selected_tasks.jsonl",
        trajectories_path="artifacts/data/trajectories.jsonl",
        report_path="artifacts/data/evaluation_report.json",
        manifest_path="artifacts/data/evaluation_manifest.json",
    )
    config_path = root / "config" / "architecture.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with (root / "data" / "train.jsonl").open("w", encoding="utf-8") as handle:
        for source in EXPECTED_SOURCE_ORDER:
            for index in range(4):
                handle.write(
                    json.dumps(aligned_row(make_task(source, index)), ensure_ascii=False)
                    + "\n"
                )
    return root, config_path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ArchitectureEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_collection_is_failure_isolated_and_never_trains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config_path = create_project(Path(directory))
            backend = FakeBackend()
            with patch.object(
                _MODULE.LiveSmokeBackend,
                "from_config",
                return_value=backend,
            ):
                manifest = await run_architecture_evaluation(
                    config_path,
                    project_root=root,
                )

            selected = read_jsonl(root / "artifacts/data/selected_tasks.jsonl")
            selected_sources = [row["metadata"]["dataset_key"] for row in selected]
            self.assertEqual(7, len(selected))
            self.assertEqual(
                {source: 1 for source in EXPECTED_SOURCE_ORDER},
                dict(Counter(selected_sources)),
            )
            self.assertEqual(
                [f"{source}:3" for source in EXPECTED_SOURCE_ORDER],
                [row["task_id"] for row in selected],
            )
            self.assertEqual(7, len(backend.collected_task_ids))
            self.assertEqual(0, backend.train_calls)
            self.assertEqual(0, backend.publish_calls)

            self.assertEqual("completed", manifest["status"])
            self.assertEqual(6, manifest["collected"])
            self.assertIn("triviaqa:3", manifest["collection_failures"])

            report = read_json(root / "artifacts/data/evaluation_report.json")
            rows = {row["dataset_key"]: row for row in report["datasets"]}
            self.assertFalse(report["heldout_validation"])
            self.assertFalse(report["stop_threshold_assessed"])
            self.assertEqual("collection_failed", rows["triviaqa"]["status"])
            self.assertIsNone(rows["triviaqa"]["metric_value"])
            self.assertEqual("unmeasurable", rows["swe_bench"]["status"])
            self.assertEqual(0, rows["swe_bench"]["valid_samples"])
            self.assertIsNone(rows["swe_bench"]["metric_value"])
            self.assertEqual(
                "synthetic_evaluator_unavailable",
                rows["swe_bench"]["reason"],
            )
            self.assertEqual(5, report["measurable_dataset_count"])

    async def test_resume_reuses_only_exact_condition_and_version_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config_path = create_project(Path(directory))
            initial_backend = FakeBackend()
            with patch.object(
                _MODULE.LiveSmokeBackend,
                "from_config",
                return_value=initial_backend,
            ):
                await run_architecture_evaluation(config_path, project_root=root)

            resume_backend = FakeBackend()
            with patch.object(
                _MODULE.LiveSmokeBackend,
                "from_config",
                return_value=resume_backend,
            ):
                resumed = await run_architecture_evaluation(
                    config_path,
                    project_root=root,
                )

            self.assertEqual(["triviaqa:3"], resume_backend.collected_task_ids)
            self.assertEqual(6, resumed["resumed_trajectories"])
            self.assertEqual(0, resumed["fresh_collected"])

            trajectories_path = root / "artifacts/data/trajectories.jsonl"
            saved = read_jsonl(trajectories_path)
            saved[0]["versions"]["prompt"] = "mismatched-prompt"
            saved[1]["condition_id"] = "mismatched-condition"
            trajectories_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in saved
                ),
                encoding="utf-8",
            )

            mismatch_backend = FakeBackend()
            with patch.object(
                _MODULE.LiveSmokeBackend,
                "from_config",
                return_value=mismatch_backend,
            ):
                mismatch_run = await run_architecture_evaluation(
                    config_path,
                    project_root=root,
                )

            self.assertEqual(
                {"hotpotqa:3", "triviaqa:3", "aime_2026:3"},
                set(mismatch_backend.collected_task_ids),
            )
            self.assertEqual(4, mismatch_run["resumed_trajectories"])
            self.assertEqual(2, mismatch_run["fresh_collected"])

    def test_max_rounds_without_explicit_finish_is_reported_as_zero(self) -> None:
        task = make_task("webshop", 3)
        versions = _MODULE.version_bundle_for(
            task,
            policy_version="policy-test",
            model_catalog_version="catalog-test-v1",
            prompt_version="prompt-test",
            tool_version="tool-test",
        )
        successful_environment_receipt = make_trajectory(
            task,
            0,
            versions,
            valid=True,
        )
        max_rounds = replace(
            successful_environment_receipt,
            final_answer=None,
            termination_reason="max_rounds",
            explicit_finish=False,
        )

        report = architecture_report([task], [max_rounds], {})
        row = report["datasets"][0]
        self.assertEqual("terminal_failure", row["status"])
        self.assertEqual(0.0, row["metric_value"])
        self.assertEqual(0, row["correct_or_successful"])
        self.assertEqual(1, row["valid_samples"])
        self.assertEqual(1.0, row["evaluator_reward_ignored"])
        self.assertEqual(
            "director_max_rounds_without_explicit_finish",
            row["reason"],
        )


if __name__ == "__main__":
    unittest.main()
