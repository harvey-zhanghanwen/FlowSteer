from __future__ import annotations

import importlib.util
import json
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
from src.interactive.step_transaction import StepPhase, StepTransaction


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_hotpotqa_grpo.py"
_SPEC = importlib.util.spec_from_file_location("train_hotpotqa_grpo", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_CONFIG = _SCRIPT.parents[1] / "config" / "training_hotpotqa_grpo.yaml"

HotpotTrainingError = _MODULE.HotpotTrainingError
run_hotpotqa_training = _MODULE.run_hotpotqa_training
sample_hotpotqa_tasks = _MODULE.sample_hotpotqa_tasks
validate_hotpotqa_training_config = _MODULE.validate_hotpotqa_training_config


class FakeTracker:
    run_id = "wandb-test-run"
    run_url = "https://wandb.invalid/test"

    def __init__(self) -> None:
        self.logs: list[tuple[int, dict]] = []
        self.summary: dict = {}
        self.finished: list[int] = []

    def log(self, values, *, step: int) -> None:
        self.logs.append((step, dict(values)))

    def update_summary(self, values) -> None:
        self.summary.update(values)

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class FakeSummary:
    def __init__(self, values: dict) -> None:
        self.values = values

    def to_dict(self) -> dict:
        return dict(self.values)


class FakeReceipt:
    def __init__(self, values: dict) -> None:
        self.values = values

    def to_dict(self) -> dict:
        return dict(self.values)


def _trajectory(task, rollout_index, versions, *, adapter: str) -> TrajectoryRecord:
    graph: dict = {}
    snapshot_id = stable_id(
        "snapshot",
        {"revision": 0, "graph": graph, "previous_snapshot_id": None},
    )
    turn = TurnRecord(
        turn_id=f"turn:{task.task_id}:{rollout_index}",
        round_index=0,
        prompt="ordinary prompt",
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
        policy_adapter=adapter,
        server_weight_version="default",
        receipt_verified=True,
    )
    reward = float(rollout_index % 2)
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{rollout_index}:{versions.policy}",
        task=task,
        group_id=f"{task.task_id}:hotpotqa_grpo_natural_v1:{versions.policy}",
        condition_id="hotpotqa_grpo_natural_v1",
        rollout_id=f"rollout:{rollout_index}",
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=reward,
            metrics={"exact_match": reward, "token_f1": reward},
        ),
        termination_reason="finish",
        explicit_finish=True,
    )


class FakeBackend:
    model_catalog_version = "catalog-test-v1"

    def __init__(self, config: dict, events: list[str]) -> None:
        self.config = config
        self.events = events
        self.training_step = int(config["experiment"]["training_step"])
        self.absolute_step = int(config["experiment"]["update_step"])
        self.active_policy = str(config["director"]["behavior_policy_version"])
        self.active_adapter = str(config["director"]["behavior_adapter_name"])

    async def ensure_behavior_ready(self):
        self.events.append(f"ready:{self.training_step}:{self.active_policy}")
        return {"success": True, "status": "ready"}

    async def collect(
        self,
        task,
        rollout_index,
        versions,
        *,
        expected_task_split="train",
    ):
        assert expected_task_split == "train"
        kind = "canary" if rollout_index >= 1_000_000 else "rollout"
        self.events.append(
            f"collect:{self.training_step}:{kind}:{versions.policy}:{self.active_adapter}"
        )
        return _trajectory(
            task,
            rollout_index,
            versions,
            adapter=self.active_adapter,
        )

    def train(self, trajectories, output_dir):
        self.events.append(f"train:{self.training_step}")
        checkpoint = Path(output_dir) / "checkpoint_final" / "supervisor_lora" / "theta"
        checkpoint.mkdir(parents=True)
        optimizer_state = checkpoint / "optimizer_state.pt"
        optimizer_state.write_bytes(b"state")
        values = {
            "optimizer_updates": 1,
            "input_trajectories": len(trajectories),
            "informative_groups": 7,
            "trained_trajectories": len(trajectories),
            "loss": 0.25 / self.training_step,
            "grad_norm": 1.0,
            "trainable_update_l2": 0.1,
            "behavior_policy_version": self.active_policy,
            "updated_policy_version": self.config["director"][
                "updated_policy_version"
            ],
            "micro_batch_size_used": 4,
            "oom_backoff_count": 0,
            "checkpoint_dir": str(checkpoint),
            "optimizer_state_checkpoint": str(optimizer_state),
            "optimizer_state_saved": True,
            "committed_step": self.absolute_step,
        }
        return FakeSummary(values)

    async def publish(self, summary):
        self.events.append(f"publish:{self.training_step}")
        candidate = summary.to_dict()["updated_policy_version"]
        adapter = (
            self.config["policy_sync"]["adapter_name_prefix"]
            + f"{self.absolute_step:06d}"
        )
        previous = self.active_policy
        self.active_policy = candidate
        self.active_adapter = adapter
        return FakeReceipt(
            {
                "success": True,
                "status": "published",
                "behavior_policy_version": previous,
                "candidate_policy_version": candidate,
                "new_policy_version": candidate,
                "adapter_name": adapter,
            }
        )


def _task(index: int, *, source: str = "hotpotqa") -> TaskRecord:
    return TaskRecord(
        task_id=f"{source}:{index}",
        question=f"Question {source} {index}?",
        ground_truth="answer",
        split="train",
        metadata={
            "dataset_key": source,
            "source": "HotpotQA" if source == "hotpotqa" else "TriviaQA",
            "sampling": {"base_task_id": f"{source}:base:{index}"},
        },
    )


def _write_tasks(path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(12):
            task = _task(index)
            stream.write(
                json.dumps(
                    {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                )
                + "\n"
            )
        for index in range(2):
            task = _task(index, source="triviaqa")
            stream.write(
                json.dumps(
                    {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                )
                + "\n"
            )


def _create_project(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    config["data"]["train_path"] = "data/train.jsonl"
    config["data"]["validation_path"] = "data/validation.jsonl"
    config["experiment"]["output_dir"] = "artifacts/training"
    config["storage"].update(
        root="artifacts/training/evidence",
        manifest_path="artifacts/training/run_manifest.json",
        state_path="artifacts/training/run_state.json",
        training_log_path="artifacts/training/training_log.jsonl",
    )
    config_path = root / "config" / "training.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_tasks(root / "data" / "train.jsonl")
    return config_path


class ConfigAndSamplingTests(unittest.TestCase):
    def test_config_fixes_grpo_and_disables_other_learning_flows(self) -> None:
        config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
        validate_hotpotqa_training_config(config)
        config["grpo"]["ttb_enabled"] = True
        with self.assertRaisesRegex(Exception, "TTB"):
            validate_hotpotqa_training_config(config)

    def test_skillflow_seeded_sample_is_repeatable_and_unique(self) -> None:
        pool = tuple(_task(index) for index in range(10))
        first = sample_hotpotqa_tasks(
            pool,
            optimizer_step=3,
            tasks_per_step=7,
            seed=42,
            seed_offset=0,
        )
        second = sample_hotpotqa_tasks(
            pool,
            optimizer_step=3,
            tasks_per_step=7,
            seed=42,
            seed_offset=0,
        )
        self.assertEqual([item.task_id for item in first], [item.task_id for item in second])
        self.assertEqual(7, len({_MODULE._base_task_id(item) for item in first}))


class StepTransactionTests(unittest.TestCase):
    def test_journal_and_recovery_state_are_durable_and_clearable(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            transaction = StepTransaction(directory)
            transaction.transition(step=1, phase=StepPhase.PREPARED, policy="v0")
            transaction.seal(step=1, state={"phase": "rollout_complete"})
            self.assertEqual(1, transaction.load()["step"])
            journal = [
                json.loads(line)
                for line in transaction.journal.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("prepared", journal[0]["phase"])
            transaction.clear()
            self.assertIsNone(transaction.load())


class SequentialRunnerTests(unittest.TestCase):
    def test_two_step_boundary_has_no_cross_step_prefetch(self) -> None:
        import tempfile

        async def inline_to_thread(function, /, *args, **kwargs):
            return function(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _create_project(root)
            events: list[str] = []
            tracker = FakeTracker()

            def factory(config, project_root):
                self.assertEqual(root, project_root)
                return FakeBackend(config, events)

            # Keep production fsync semantics while avoiding the workspace
            # filesystem's sync latency in this dependency-light unit test.
            with (
                patch.object(_MODULE.os, "fsync"),
                patch.object(_MODULE.asyncio, "to_thread", new=inline_to_thread),
            ):
                import asyncio

                manifest = asyncio.run(
                    run_hotpotqa_training(
                        config_path,
                        project_root=root,
                        stop_after_optimizer_steps=2,
                        backend_factory=factory,
                        tracker=tracker,
                    )
                )

            self.assertEqual("paused_at_requested_boundary", manifest["status"])
            self.assertEqual(2, manifest["optimizer_updates_completed"])
            self.assertEqual([1, 2], [step for step, _ in tracker.logs])
            self.assertEqual([0], tracker.finished)
            for _, values in tracker.logs:
                self.assertGreater(values["train/grad_norm"], 0)
                self.assertGreater(values["train/update_l2"], 0)
                self.assertTrue(values["policy/sync_success"])
                self.assertTrue(values["policy/canary_success"])

            first_publish = events.index("publish:1")
            first_canary = next(
                index
                for index, event in enumerate(events)
                if event.startswith("collect:1:canary:")
            )
            second_rollout = next(
                index
                for index, event in enumerate(events)
                if event.startswith("collect:2:rollout:")
            )
            self.assertLess(first_publish, first_canary)
            self.assertLess(first_canary, second_rollout)
            self.assertEqual(
                28,
                sum(event.startswith("collect:1:rollout:") for event in events),
            )
            self.assertEqual(
                28,
                sum(event.startswith("collect:2:rollout:") for event in events),
            )

            state = json.loads(
                (root / "artifacts/training/run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, state["optimizer_updates_completed"])
            self.assertEqual(
                "qwen35-9b-hotpotqa-grpo-step-000002",
                state["behavior_policy_version"],
            )
            self.assertTrue(Path(state["optimizer_state_checkpoint"]).is_file())
            journal = [
                json.loads(line)
                for line in (
                    root / "artifacts/training/recovery/step_journal.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            expected = [phase.value for phase in StepPhase]
            for step in (1, 2):
                self.assertEqual(
                    expected,
                    [item["phase"] for item in journal if item["step"] == step],
                )

    def test_prepare_only_does_not_construct_backend_or_tracker(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _create_project(root)

            def forbidden_backend(config, project_root):
                raise AssertionError("backend must not be constructed")

            with patch.object(_MODULE.os, "fsync"):
                import asyncio

                manifest = asyncio.run(
                    run_hotpotqa_training(
                        config_path,
                        project_root=root,
                        prepare_only=True,
                        backend_factory=forbidden_backend,
                    )
                )
            self.assertEqual("prepared", manifest["status"])
            self.assertEqual(0, manifest["optimizer_updates_completed"])


if __name__ == "__main__":
    unittest.main()
