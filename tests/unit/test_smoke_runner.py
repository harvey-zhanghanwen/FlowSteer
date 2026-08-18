from __future__ import annotations

import json
import importlib.util
import os
import copy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from src.interactive.persistence import stable_id
from src.interactive.hotpot_training_schedule import (
    HotpotTrainingCursorState,
    freeze_hotpot_training_schedule,
)
from src.interactive.joint_qa_training_schedule import (
    JointQATrainingCursorState,
    freeze_joint_qa_training_schedule,
)
from src.interactive.qa_retrieval import QARetrievalReceipt, build_keyword_query
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
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
validate_resumed_initial_rollouts = _MODULE._validate_resumed_initial_rollouts


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
    base_seed = 42
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(
            base_seed=base_seed
        ),
        schedule_purpose="natural_smoke",
        ordered_sequence_hash=stable_hash([task.task_id]),
        sequence_position=rollout_index,
        task_id=task.task_id,
        optimizer_step_or_anchor_ordinal=0,
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
        director_generation_seed=derive_generation_seed(
            base_seed=base_seed,
            coordinate=coordinate,
            step_index=1,
            phase=GenerationPhase.ACTION,
        ),
    )
    reward = float(rollout_index % 2)
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{versions.policy}:{rollout_index}",
        task=task,
        group_id=f"{task.task_id}:natural_smoke:{versions.policy}",
        condition_id="natural_smoke",
        rollout_id=(
            f"{task.task_id}:natural_smoke:{versions.policy}:"
            f"rollout:{rollout_index:04d}"
        ),
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
        director_sampling={
            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": base_seed,
            "coordinate": coordinate.to_value(),
            "phase": GenerationPhase.ACTION.value,
        },
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
            "optimizer_state_saved": False,
            "trainable_update_l2": 1.0 if self.optimizer_updates else 0.0,
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
        summary = Summary(checkpoint, self.updates)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary.to_dict()) + "\n",
            encoding="utf-8",
        )
        return summary

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


def create_hotpot_micro_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, config_path = create_project(tmp_path)
    validation_path = root / "data" / "validation.jsonl"
    test_path = root / "data" / "test.jsonl"
    for path, split, index in (
        (validation_path, "validation", 100),
        (test_path, "test", 101),
    ):
        task = TaskRecord(
            task_id=f"hotpotqa:{split}-{index}",
            question="Held-out question?",
            ground_truth="answer",
            split=split,
            metadata={"dataset_key": "hotpotqa", "source": "HotpotQA"},
        )
        path.write_text(json.dumps(aligned_row(task)) + "\n", encoding="utf-8")

    schedule = freeze_hotpot_training_schedule(
        train_path=root / "data" / "train.jsonl",
        validation_path=validation_path,
        test_path=test_path,
        task_positions=(0,),
        rollouts_per_task=2,
    )
    schedule_path = root / "artifacts" / "schedule.json"
    cursor_path = root / "artifacts" / "cursor0.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_once(schedule_path)
    HotpotTrainingCursorState.fresh(schedule).write_once(cursor_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"].update(
        phase="hotpotqa_micro_training",
        update_step=1,
        output_dir="artifacts/hotpot_step1/training",
    )
    config["data"].update(
        validation_path="data/validation.jsonl",
        test_path="data/test.jsonl",
        hotpot_micro={
            "split": "train",
            "dataset_key": "hotpotqa",
            "selection": "frozen_hotpot_schedule",
            "expected_total_tasks": 1,
            "schedule_path": "artifacts/schedule.json",
            "cursor_path": "artifacts/cursor0.json",
            "next_cursor_path": "artifacts/cursor1.json",
        },
    )
    config["evaluation"]["healthbench_judge_model"] = ""
    config["grpo"].update(samples_per_problem=2, expected_rollout_count=2)
    for field in (
        "root",
        "selected_tasks_path",
        "trajectories_path",
        "grpo_groups_path",
        "manifest_path",
        "sync_receipt_path",
        "post_update_trajectories_path",
    ):
        leaf = Path(str(config["storage"][field])).name
        config["storage"][field] = f"artifacts/hotpot_step1/{leaf}"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config_path, root / "artifacts" / "cursor1.json"


def create_joint_qa_micro_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, _ = create_project(tmp_path)
    validation_path = root / "data" / "validation.jsonl"
    test_path = root / "data" / "test.jsonl"
    for path, split, index in (
        (validation_path, "validation", 100),
        (test_path, "test", 101),
    ):
        rows = []
        for source in ("hotpotqa", "triviaqa"):
            task = TaskRecord(
                task_id=f"{source}:{split}-{index}",
                question="Held-out question?",
                ground_truth="answer",
                split=split,
                metadata={"dataset_key": source, "source": SOURCE_NAMES[source]},
            )
            rows.append(json.dumps(aligned_row(task)) + "\n")
        path.write_text("".join(rows), encoding="utf-8")

    schedule = freeze_joint_qa_training_schedule(
        train_path=root / "data" / "train.jsonl",
        validation_path=validation_path,
        test_path=test_path,
        task_positions_by_dataset={"hotpotqa": (0,), "triviaqa": (0,)},
        rollouts_per_task=8,
    )
    schedule_path = root / "artifacts" / "joint_schedule.json"
    cursor_path = root / "artifacts" / "joint_cursor0.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_once(schedule_path)
    JointQATrainingCursorState.fresh(schedule).write_once(cursor_path)

    config = yaml.safe_load(
        Path("config/training_joint_qa_step1.yaml").read_text(encoding="utf-8")
    )
    config["data"].update(
        train_path="data/train.jsonl",
        validation_path="data/validation.jsonl",
        test_path="data/test.jsonl",
    )
    config["data"]["joint_qa_micro"].update(
        schedule_path="artifacts/joint_schedule.json",
        cursor_path="artifacts/joint_cursor0.json",
        next_cursor_path="artifacts/joint_cursor1.json",
    )
    config["experiment"]["output_dir"] = "artifacts/joint_step1/training"
    for field in (
        "root",
        "selected_tasks_path",
        "retrieval_receipts_path",
        "trajectories_path",
        "grpo_groups_path",
        "manifest_path",
        "behavior_policy_preflight_path",
        "sync_receipt_path",
        "post_update_trajectories_path",
    ):
        leaf = Path(str(config["storage"][field])).name
        config["storage"][field] = f"artifacts/joint_step1/{leaf}"
    trivia = make_task("triviaqa", 0)
    receipt = QARetrievalReceipt(
        query=build_keyword_query(trivia.question),
        search_limit=5,
        passages=(),
    )
    retrieval_path = root / config["storage"]["retrieval_receipts_path"]
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(
        json.dumps(
            {
                "schema_version": "flowsteer.triviaqa.public_retrieval.v1",
                "task_id": trivia.task_id,
                "question": trivia.question,
                "retrieval": receipt.to_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = root / "config" / "joint.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config_path, root / "artifacts" / "joint_cursor1.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_fake_evidence(trajectory_path: Path, evidence_path: Path) -> None:
    trajectory_rows = read_jsonl(trajectory_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        "".join(
            json.dumps(
                {
                    "record_kind": "trajectory",
                    "event_id": row["trajectory_id"],
                    "payload": row,
                }
            )
            + "\n"
            for row in trajectory_rows
        ),
        encoding="utf-8",
    )


class SelectionTests(unittest.TestCase):
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

    def test_hotpot_micro_bounds_require_zero_shaping_rewards(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            _, config_path, _ = create_hotpot_micro_project(Path(directory))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            validate_smoke_bounds(config)
            config["grpo"]["structural_reward"] = 0.1
            with self.assertRaisesRegex(Exception, "structural_reward"):
                validate_smoke_bounds(config)

    def test_joint_qa_bounds_require_two_groups_and_two_canaries(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            _, config_path, _ = create_joint_qa_micro_project(Path(directory))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            validate_smoke_bounds(config)
            config["grpo"]["expected_rollout_count"] = 8
            with self.assertRaisesRegex(Exception, "expected_rollout_count"):
                validate_smoke_bounds(config)

    def test_joint_qa_prepare_freezes_both_tasks_and_trivia_retrieval(self) -> None:
        import asyncio
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root, config_path, _ = create_joint_qa_micro_project(Path(directory))
            manifest = asyncio.run(
                run_smoke(config_path, prepare_only=True, project_root=root)
            )
            self.assertEqual("prepared", manifest["status"])
            self.assertEqual(
                {"hotpotqa": 1, "triviaqa": 1},
                manifest["selected_by_source"],
            )
            selected = read_jsonl(
                root / "artifacts/joint_step1/selected_tasks.jsonl"
            )
            self.assertEqual(2, len(selected))
            self.assertIn(
                "Public retrieval observations (SkillFlow search/read)",
                selected[1]["question"],
            )

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
        self.assertEqual("hotpotqa.official.answer.v1", versions["hotpotqa"])
        self.assertEqual("triviaqa.official.answer.v1", versions["triviaqa"])
        self.assertEqual("skillflow.training.reward.v1", versions["aime_2026"])
        self.assertEqual(versions["webshop"], versions["alfworld"])
        self.assertEqual(6, len(set(versions.values())))

    def test_exact_resume_preserves_but_excludes_a_malformed_atomic_action(self) -> None:
        task = make_task("hotpotqa", 0)
        versions = _MODULE.version_bundle_for(
            task,
            policy_version="qwen35-9b-base-step-0000",
            model_catalog_version="catalog-test-v1",
        )
        invalid = trajectory(task, 0, versions)
        invalid_turn = replace(
            invalid.turns[0],
            executed_prefix_tokens=0,
            action={},
            canvas_feedback="invalid action: malformed JSON",
        )
        invalid = replace(invalid, turns=(invalid_turn,))
        valid = trajectory(task, 1, versions)
        third = trajectory(task, 2, versions)

        validate_resumed_initial_rollouts(
            (invalid, valid, third),
            ((task, 0, versions), (task, 1, versions), (task, 2, versions)),
            condition_id="natural_smoke",
            sampling_anchor_ordinal=0,
            behavior_adapter_name=None,
            expected_server_weight_version="default",
        )

        self.assertFalse(invalid.grpo_eligible)
        self.assertTrue(valid.grpo_eligible)


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

    async def test_hotpot_micro_executes_only_current_frozen_step_and_commits_cursor(
        self,
    ) -> None:
        root, config_path, next_cursor_path = create_hotpot_micro_project(
            Path(self._temp_dir.name)
        )
        backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            backend=backend,
            project_root=root,
        )
        self.assertEqual("completed", manifest["status"])
        self.assertEqual(1, manifest["bounds"]["selected_tasks"])
        self.assertEqual(2, manifest["bounds"]["expected_initial_rollouts"])
        self.assertEqual(["collect:0", "collect:1"], backend.events[:2])
        self.assertEqual(2, len(backend.train_inputs))
        committed = HotpotTrainingCursorState.read(next_cursor_path)
        self.assertEqual(1, committed.cursor)
        self.assertEqual(
            committed.to_value(),
            manifest["selection_receipt"]["cursor_after"],
        )

    async def test_hotpot_micro_strict_resume_skips_initial_collection(self) -> None:
        root, config_path, _ = create_hotpot_micro_project(
            Path(self._temp_dir.name)
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # The fake trajectory helper predates formal step ordinals; pin its
        # existing sampling anchor explicitly for this persistence test.
        config["experiment"]["sampling_anchor_ordinal"] = 0
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        failed_backend = FakeBackend(updates=0)
        with self.assertRaisesRegex(SmokeRunError, "zero optimizer updates"):
            await run_smoke(
                config_path,
                backend=failed_backend,
                project_root=root,
            )
        self.assertEqual(
            ["collect:0", "collect:1", "train"], failed_backend.events
        )
        write_fake_evidence(
            root / "artifacts/hotpot_step1/trajectories.jsonl",
            root / "artifacts/hotpot_step1/evidence/trajectories.jsonl",
        )

        resumed_backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            resume_initial_rollouts=True,
            backend=resumed_backend,
            project_root=root,
        )
        self.assertEqual("completed", manifest["status"])
        self.assertEqual(
            ["train", "publish", "collect:10000"], resumed_backend.events
        )
        self.assertEqual(
            {
                "mode": "strict_persisted_resume",
                "path": str(
                    root / "artifacts/hotpot_step1/trajectories.jsonl"
                ),
                "reused": 2,
                "new_collections": 0,
            },
            manifest["initial_rollout_source"],
        )

    async def test_hotpot_micro_resume_after_precheckpoint_runtime_failure(
        self,
    ) -> None:
        class PrecheckpointFailureBackend(FakeBackend):
            def train(self, trajectories, output_dir):
                del output_dir
                self.events.append("train")
                self.train_inputs = list(trajectories)
                raise RuntimeError("replica unavailable before checkpoint")

        root, config_path, _ = create_hotpot_micro_project(
            Path(self._temp_dir.name)
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["experiment"]["sampling_anchor_ordinal"] = 0
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        failed_backend = PrecheckpointFailureBackend()
        with self.assertRaisesRegex(SmokeRunError, "one-pass smoke training failed"):
            await run_smoke(
                config_path,
                backend=failed_backend,
                project_root=root,
            )
        write_fake_evidence(
            root / "artifacts/hotpot_step1/trajectories.jsonl",
            root / "artifacts/hotpot_step1/evidence/trajectories.jsonl",
        )

        resumed_backend = FakeBackend()
        manifest = await run_smoke(
            config_path,
            resume_initial_rollouts=True,
            backend=resumed_backend,
            project_root=root,
        )

        self.assertEqual("completed", manifest["status"])
        self.assertEqual(
            ["train", "publish", "collect:10000"], resumed_backend.events
        )
        self.assertEqual(
            "failed_training_before_persistence",
            manifest["resume_preconditions"]["root_zero_update_status"],
        )

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

    async def test_live_backend_switches_director_route_inside_publisher_gate(
        self,
    ) -> None:
        class RecordingGate(_MODULE.RolloutGate):
            def __init__(self) -> None:
                super().__init__(poll_interval_seconds=0.001)
                self.events: list[str] = []

            def pause(self) -> None:
                self.events.append("pause")
                super().pause()

            def drain(self, timeout_seconds=None) -> None:
                self.events.append("drain")
                super().drain(timeout_seconds)

            def resume(self) -> None:
                self.events.append("resume")
                super().resume()

        class RouteClient:
            def __init__(self, gate) -> None:
                self.gate = gate
                self.policy_version = "behavior-v0"
                self.adapter_name = "theta_smoke_step_000000"
                self.expected_server_weight_version = "server-v0"
                self.updates: list[tuple[str, str | None, str | None]] = []

            def update_policy_route(
                self,
                *,
                policy_version,
                adapter_name,
                expected_server_weight_version,
            ) -> None:
                self.gate.require_paused_and_drained()
                self.policy_version = policy_version
                self.adapter_name = adapter_name
                self.expected_server_weight_version = expected_server_weight_version
                self.updates.append(
                    (
                        policy_version,
                        adapter_name,
                        expected_server_weight_version,
                    )
                )

        class TransactionalPublisher:
            def __init__(self, client) -> None:
                self.client = client

            def publish(self, **kwargs):
                gate = kwargs["gate"]
                gate.pause()
                try:
                    gate.drain()
                    kwargs["route_switch"](
                        "qwen35-9b-smoke-step-0001",
                        "theta_smoke_step_000001",
                    )
                    assert self.client.adapter_name == "theta_smoke_step_000001"
                finally:
                    gate.resume()
                return Receipt()

        gate = RecordingGate()
        client = RouteClient(gate)
        backend = object.__new__(_MODULE.LiveSmokeBackend)
        backend.config = {
            "director": {
                "behavior_adapter_name": "theta_smoke_step_000000",
                "expected_server_weight_version": "server-v1",
            },
            "experiment": {"update_step": 1},
        }
        backend.director_client = client
        backend.rollout_gate = gate
        backend.publisher = TransactionalPublisher(client)

        summary = Summary(Path(self._temp_dir.name))
        receipt = await backend.publish(summary)

        self.assertIsInstance(receipt, Receipt)
        self.assertEqual(gate.events, ["pause", "drain", "resume"])
        self.assertFalse(gate.paused)
        self.assertEqual(
            client.updates,
            [
                (
                    "qwen35-9b-smoke-step-0001",
                    "theta_smoke_step_000001",
                    "server-v1",
                )
            ],
        )

    def setUp(self) -> None:
        import tempfile

        self._temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
