from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.interactive.persistence import stable_id
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.smoke_trainer import (
    Qwen35OnePassSmokeTrainer,
    SmokeTrainerConfig,
)
from src.interactive.versioning import VersionBundle


BEHAVIOR_POLICY = "qwen35-9b-base-step-0000"


def _config(**changes: object) -> SmokeTrainerConfig:
    values: dict[str, object] = {
        "model_path": "/models/Qwen3.5-9B",
        "tokenizer_path": "/tokenizers/Qwen3.5-9B",
    }
    values.update(changes)
    return SmokeTrainerConfig(**values)


def _turn(
    turn_id: str,
    *,
    policy: str = BEHAVIOR_POLICY,
    adapter: str | None = None,
    server_weight: str = "default",
) -> TurnRecord:
    graph: dict[str, object] = {}
    snapshot_id = stable_id(
        "snapshot",
        {"revision": 0, "graph": graph, "previous_snapshot_id": None},
    )
    return TurnRecord(
        turn_id=turn_id,
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
        policy_version=policy,
        policy_adapter=adapter,
        server_weight_version=server_weight,
        graph_snapshot_id=snapshot_id,
        previous_graph_snapshot_id=None,
        receipt_verified=True,
    )


def _trajectory(index: int, turn: TurnRecord) -> TrajectoryRecord:
    versions = VersionBundle(
        policy=BEHAVIOR_POLICY,
        model_catalog="catalog-v1",
        evaluator="evaluator-v1",
        prompt="minimal-director-v1",
        tool="agentgraph-v1",
    )
    task = TaskRecord(
        task_id="hotpotqa:0",
        question="Question?",
        ground_truth="answer",
        split="train",
        metadata={"dataset_key": "hotpotqa", "source": "HotpotQA"},
    )
    return TrajectoryRecord(
        trajectory_id=f"trajectory-{index}",
        task=task,
        group_id=f"hotpotqa:0:natural:{BEHAVIOR_POLICY}",
        condition_id="natural_smoke",
        rollout_id=f"rollout-{index}",
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt("evaluator-v1", True, float(index)),
        termination_reason="finish",
        explicit_finish=True,
    )


class SmokeTrainerConfigTests(unittest.TestCase):
    def test_default_config_is_deterministic_and_has_strict_backoff(self) -> None:
        config = _config()

        self.assertEqual(config.lora_dropout, 0.0)
        self.assertEqual(config.micro_batch_backoff, (4, 2, 1))
        self.assertNotEqual(config.learner_device, config.gradient_replica_device)

    def test_backoff_must_be_integer_strictly_decreasing_and_end_at_one(self) -> None:
        invalid_schedules = (
            (),
            (4, 2),
            (4, 4, 1),
            (1, 2, 1),
            (4, 0, 1),
            (4, 2.0, 1),
            [4, 2, 1],
        )
        for schedule in invalid_schedules:
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, "micro-batch backoff"):
                    _config(micro_batch_backoff=schedule)

    def test_oom_replay_rejects_stochastic_lora_even_without_checkpointing(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            _config(lora_dropout=0.1, gradient_checkpointing=False)

    def test_training_replicas_must_use_distinct_devices(self) -> None:
        with self.assertRaisesRegex(ValueError, "devices must differ"):
            _config(learner_device="cuda:3", gradient_replica_device="cuda:3")


class BehaviorRouteGateTests(unittest.TestCase):
    def _assert_rejected_before_load(
        self,
        turns: tuple[TurnRecord, TurnRecord],
        message: str,
    ) -> None:
        trainer = Qwen35OnePassSmokeTrainer(_config())
        trajectories = tuple(
            _trajectory(index, turn) for index, turn in enumerate(turns)
        )
        with TemporaryDirectory() as directory:
            with patch.object(trainer, "_load_models") as load_models:
                with self.assertRaisesRegex(ValueError, message):
                    trainer.train(trajectories, Path(directory))
                load_models.assert_not_called()

    def test_wrong_and_mixed_adapter_server_routes_are_hard_rejected(self) -> None:
        cases = {
            "wrong_server": (
                _turn("turn-0", server_weight="stale"),
                _turn("turn-1", server_weight="stale"),
            ),
            "mixed_server": (
                _turn("turn-0"),
                _turn("turn-1", server_weight="stale"),
            ),
            "mixed_adapter": (
                _turn("turn-0"),
                _turn("turn-1", adapter="theta_stale"),
            ),
        }
        for name, turns in cases.items():
            with self.subTest(name=name):
                self._assert_rejected_before_load(
                    turns,
                    "exact behavior route receipt",
                )

    def test_wrong_logical_policy_turn_receipt_is_hard_rejected(self) -> None:
        self._assert_rejected_before_load(
            (
                _turn("turn-0"),
                replace(_turn("turn-1"), policy_version="stale-policy"),
            ),
            "logical behavior policy",
        )


if __name__ == "__main__":
    unittest.main()
