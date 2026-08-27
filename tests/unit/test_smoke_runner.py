from __future__ import annotations

import json
import importlib.util
import os
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from src.interactive.persistence import stable_id
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_agentgraph_smoke.py"
_SPEC = importlib.util.spec_from_file_location("train_agentgraph_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

EXPECTED_SOURCE_ORDER = _MODULE.EXPECTED_SOURCE_ORDER
SmokeRunError = _MODULE.SmokeRunError
evaluator_version_for = _MODULE.evaluator_version_for
run_smoke = _MODULE.run_smoke
select_smoke_tasks = _MODULE.select_smoke_tasks
validate_smoke_bounds = _MODULE.validate_smoke_bounds
qa_retrieval_runtime_task = _MODULE.qa_retrieval_runtime_task
qa_retrieval_scopes = _MODULE.qa_retrieval_scopes


SOURCE_NAMES = {
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "aime_2026": "AIME 2026",
    "healthbench_professional": "HealthBench Professional",
    "webshop": "WebShop",
    "alfworld": "ALFWorld",
    "swe_bench": "SWE-bench",
}


def make_task(source: str, index: int, *, base_id: str | None = None) -> TaskRecord:
    task_id = f"{source}:{index}"
    return TaskRecord(
        task_id=task_id,
        question=f"Question {source} {index}?",
        ground_truth="answer",
        split="train",
        metadata={
            "dataset_key": source,
            "source": SOURCE_NAMES[source],
            "sampling": {"base_task_id": base_id or task_id},
        },
    )


def aligned_row(task: TaskRecord) -> dict:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        **task.to_dict(),
    }


def trajectory(task: TaskRecord, rollout_index: int, versions) -> TrajectoryRecord:
    graph = {}
    snapshot_id = stable_id(
        "snapshot",
        {"revision": 0, "graph": graph, "previous_snapshot_id": None},
    )
    turn = TurnRecord(
        turn_id=f"turn:{task.task_id}:{rollout_index}",
        round_index=0,
        prompt="prompt",
        policy_response='{"action":"finish"}',
        prompt_token_ids=(1,),
        output_token_ids=(2,),
        behavior_log_probs=(-0.1,),
        executed_prefix_tokens=1,
        action={"action": "finish"},
        canvas_feedback="workflow finished",
        graph_revision=0,
        graph_snapshot=graph,
        graph_snapshot_id=snapshot_id,
        previous_graph_snapshot_id=None,
        policy_version=versions.policy,
        receipt_verified=True,
        server_weight_version="default",
        policy_adapter=(
            "theta_smoke_step_000001" if rollout_index >= 10_000 else None
        ),
    )
    reward = float(rollout_index % 2)
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{versions.policy}:{rollout_index}",
        task=task,
        group_id=f"{task.task_id}:natural_smoke:{versions.policy}",
        condition_id="natural_smoke",
        rollout_id=f"rollout:{rollout_index}",
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=reward,
            metrics={"score": reward},
        ),
        termination_reason="finish",
        explicit_finish=True,
    )


class Summary:
    def __init__(self, checkpoint: Path, updates: int = 1) -> None:
        self.optimizer_updates = updates
        self.behavior_policy_version = "qwen35-9b-base-step-0000"
        self.updated_policy_version = (
            "qwen35-9b-smoke-step-0001" if updates else ""
        )
        self.checkpoint_dir = str(checkpoint) if updates else ""

    def to_dict(self):
        return {
            "optimizer_updates": self.optimizer_updates,
            "behavior_policy_version": self.behavior_policy_version,
            "updated_policy_version": self.updated_policy_version,
            "checkpoint_dir": self.checkpoint_dir,
        }


class Receipt:
    adapter_name = "theta_smoke_step_000001"
    new_policy_version = "qwen35-9b-smoke-step-0001"

    def to_dict(self):
        return {
            "success": True,
            "status": "published",
            "adapter_name": self.adapter_name,
            "behavior_policy_version": "qwen35-9b-base-step-0000",
            "candidate_policy_version": self.new_policy_version,
            "new_policy_version": self.new_policy_version,
        }


class FakeBackend:
    model_catalog_version = "catalog-test-v1"

    def __init__(self, *, updates: int = 1) -> None:
        self.updates = updates
        self.events: list[str] = []
        self.train_inputs = []
        self.publish_summary = None

    async def collect(self, task, rollout_index, versions):
        self.events.append(f"collect:{rollout_index}")
        return trajectory(task, rollout_index, versions)

    def train(self, trajectories, output_dir):
        self.events.append("train")
        self.train_inputs = list(trajectories)
        checkpoint = output_dir / "checkpoint_final" / "supervisor_lora"
        if self.updates:
            checkpoint.mkdir(parents=True, exist_ok=True)
        return Summary(checkpoint, self.updates)

    async def publish(self, summary):
        self.events.append("publish")
        self.publish_summary = summary
        return Receipt()


def create_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    source_config = Path("config/training_agentgraph_smoke.yaml")
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["data"]["train_path"] = "data/train.jsonl"
    config["experiment"]["output_dir"] = "artifacts/smoke"
    config["storage"].update(
        root="artifacts/smoke/evidence",
        selected_tasks_path="artifacts/smoke/data/selected_tasks.jsonl",
        trajectories_path="artifacts/smoke/data/trajectories.jsonl",
        grpo_groups_path="artifacts/smoke/data/grpo_groups.jsonl",
        manifest_path="artifacts/smoke/data/training_manifest.json",
        sync_receipt_path="artifacts/smoke/data/sync_receipt.json",
        post_update_trajectories_path=(
            "artifacts/smoke/data/post_update_trajectories.jsonl"
        ),
    )
    config_path = root / "config" / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (root / "data" / "train.jsonl").open("w", encoding="utf-8") as handle:
        for source in EXPECTED_SOURCE_ORDER:
            for index in range(3):
                handle.write(json.dumps(aligned_row(make_task(source, index))) + "\n")
    return root, config_path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SelectionTests(unittest.TestCase):
    def test_explicit_public_task_and_question_only_retrieval_scopes_are_separate(self) -> None:
        public_question = (
            "Based on the following passages, answer the question.\n\n"
            "[[Delhi] Delhi is the capital of India.]\n\n"
            "Question: Which city is the capital of India?"
        )
        task = TaskRecord(
            task_id="hotpotqa:public-task",
            question=public_question,
            ground_truth="Delhi",
            split="validation",
            metadata={"dataset_key": "hotpotqa"},
        )
        retrieval = {
            "task_scope": "public_task",
            "retrieval_query_scope": "question_only",
        }

        runtime_task = qa_retrieval_runtime_task(task, retrieval)

        self.assertIs(runtime_task, task)
        self.assertIn("[[Delhi]", runtime_task.question)
        self.assertEqual(
            ("public_task", "question_only"),
            qa_retrieval_scopes(retrieval),
        )

    def test_explicit_retrieval_scope_split_fails_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "retrieval_query_scope"):
            qa_retrieval_scopes({"task_scope": "public_task"})
        with self.assertRaisesRegex(Exception, "task_scope"):
            qa_retrieval_scopes(
                {
                    "task_scope": "question_only",
                    "retrieval_query_scope": "question_only",
                }
            )

    def test_bounds_require_raw_on_policy_sampling_and_fixed_oom_schedule(self) -> None:
        config = yaml.safe_load(
            Path("config/training_agentgraph_smoke.yaml").read_text(encoding="utf-8")
        )
        invalid_sampling = copy.deepcopy(config)
        invalid_sampling["director"]["top_p"] = 0.95
        with self.assertRaisesRegex(Exception, "director.top_p"):
            validate_smoke_bounds(invalid_sampling)

        invalid_backoff = copy.deepcopy(config)
        invalid_backoff["gpu"]["oom_policy"]["micro_batch_schedule"] = [2, 1]
        with self.assertRaisesRegex(Exception, "micro_batch_schedule"):
            validate_smoke_bounds(invalid_backoff)

    def test_bounds_require_explicit_healthbench_judge(self) -> None:
        config = yaml.safe_load(
            Path("config/training_agentgraph_smoke.yaml").read_text(encoding="utf-8")
        )
        config["evaluation"]["healthbench_judge_model"] = ""
        with self.assertRaisesRegex(Exception, "healthbench_judge_model"):
            validate_smoke_bounds(config)

    def test_source_order_and_unique_base_task_are_enforced(self) -> None:
        tasks = [
            make_task("triviaqa", 0),
            make_task("hotpotqa", 0, base_id="same"),
            make_task("hotpotqa", 1, base_id="same"),
            make_task("hotpotqa", 2, base_id="different"),
            make_task("triviaqa", 1),
        ]
        selected = select_smoke_tasks(
            tasks,
            source_order=("hotpotqa", "triviaqa"),
            per_source=2,
            require_unique_base_tasks=True,
        )
        self.assertEqual(
            ["hotpotqa:0", "hotpotqa:2", "triviaqa:0", "triviaqa:1"],
            [item.task_id for item in selected],
        )

    def test_each_source_has_the_required_evaluator_version(self) -> None:
        versions = {
            source: evaluator_version_for(make_task(source, 0))
            for source in EXPECTED_SOURCE_ORDER
        }
        self.assertEqual(versions["hotpotqa"], versions["aime_2026"])
        self.assertEqual(versions["webshop"], versions["alfworld"])
        self.assertEqual(4, len(set(versions.values())))


class SmokeRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_only_needs_no_secret_and_writes_exactly_14(self) -> None:
        root, config_path = create_project(Path(self._temp_dir.name))
        with patch.dict(os.environ, {}, clear=True):
            manifest = await run_smoke(
                config_path,
                prepare_only=True,
                project_root=root,
            )
        self.assertEqual("prepared", manifest["status"])
        selected = read_jsonl(root / "artifacts/smoke/data/selected_tasks.jsonl")
        self.assertEqual(14, len(selected))
        self.assertEqual(
            list(EXPECTED_SOURCE_ORDER),
            [selected[index * 2]["metadata"]["dataset_key"] for index in range(7)],
        )

    async def test_full_pipeline_orders_update_publish_and_updated_canary(self) -> None:
        root, config_path = create_project(Path(self._temp_dir.name))
        backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            backend=backend,
            project_root=root,
        )
        self.assertEqual("completed", manifest["status"])
        self.assertEqual(28, len(backend.train_inputs))
        self.assertEqual(28, len(read_jsonl(root / "artifacts/smoke/data/trajectories.jsonl")))
        canaries = read_jsonl(
            root / "artifacts/smoke/data/post_update_trajectories.jsonl"
        )
        self.assertEqual(1, len(canaries))
        self.assertEqual(
            "qwen35-9b-smoke-step-0001",
            canaries[0]["versions"]["policy"],
        )
        self.assertEqual(
            "theta_smoke_step_000001",
            canaries[0]["turns"][0]["policy_adapter"],
        )
        self.assertLess(backend.events.index("train"), backend.events.index("publish"))
        self.assertLess(
            backend.events.index("publish"), backend.events.index("collect:10000")
        )
        groups = read_jsonl(root / "artifacts/smoke/data/grpo_groups.jsonl")
        self.assertEqual(14, len(groups))
        self.assertTrue(all(row["informative"] for row in groups))

    async def test_zero_update_is_a_failed_run_and_never_publishes(self) -> None:
        root, config_path = create_project(Path(self._temp_dir.name))
        backend = FakeBackend(updates=0)
        with self.assertRaisesRegex(SmokeRunError, "zero optimizer updates"):
            await run_smoke(config_path, backend=backend, project_root=root)
        self.assertNotIn("publish", backend.events)
        sync = json.loads(
            (root / "artifacts/smoke/data/sync_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("not_attempted_no_optimizer_update", sync["status"])
        manifest = json.loads(
            (root / "artifacts/smoke/data/training_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("failed_no_optimizer_update", manifest["status"])

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
