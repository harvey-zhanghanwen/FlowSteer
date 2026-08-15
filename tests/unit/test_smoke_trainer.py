from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, call, patch

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
    _partition_groups_by_token_cost,
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

    def test_step_two_requires_the_behavior_adapter_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, r"step 2\+"):
            _config(update_step=2)

        config = _config(
            update_step=2,
            behavior_adapter_checkpoint="/checkpoints/step-1/theta",
        )
        self.assertEqual(config.update_step, 2)

    def test_optimizer_continuation_requires_adapter_continuation(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer continuation"):
            _config(optimizer_state_checkpoint="/checkpoints/optimizer_state.pt")


class CostBalancedPartitionTests(unittest.TestCase):
    def test_exact_groups_use_skillflow_greedy_token_cost_partition(self) -> None:
        costs = (10, 9, 8, 7)
        records = {}
        groups = []
        for index, token_cost in enumerate(costs):
            trajectory_id = f"trajectory-{index}"
            records[trajectory_id] = SimpleNamespace(
                turns=(
                    SimpleNamespace(
                        prompt_token_ids=tuple(range(token_cost - 1)),
                        output_token_ids=(1,),
                    ),
                )
            )
            groups.append(
                (
                    (f"task-{index}", "condition", "policy"),
                    [SimpleNamespace(trajectory_id=trajectory_id)],
                )
            )

        partitions, partition_costs = _partition_groups_by_token_cost(
            groups,
            records,
        )

        self.assertEqual(partition_costs, (17, 17))
        self.assertEqual(
            [[entry[0][0] for entry in partition] for partition in partitions],
            [["task-0", "task-3"], ["task-1", "task-2"]],
        )

    def test_token_partition_requires_positive_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker_count"):
            _partition_groups_by_token_cost([], {}, worker_count=0)


class ContinuationStateTests(unittest.TestCase):
    def test_step_one_adapter_without_optimizer_is_explicit_warm_start(self) -> None:
        trainer = Qwen35OnePassSmokeTrainer(
            _config(
                update_step=2,
                behavior_adapter_checkpoint="/checkpoints/step-1/theta",
            )
        )

        status = trainer._restore_optimizer_state(MagicMock(), MagicMock())

        self.assertEqual(status, "warm_start_fresh_optimizer")

    def test_optimizer_restore_requires_immediately_previous_committed_step(self) -> None:
        torch = MagicMock()
        torch.load.return_value = {
            "format": "flowsteer-smoke-optimizer-v1",
            "committed_step": 2,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        }
        optimizer = MagicMock()
        trainer = Qwen35OnePassSmokeTrainer(
            _config(
                update_step=3,
                behavior_adapter_checkpoint="/checkpoints/step-2/theta",
                optimizer_state_checkpoint=(
                    "/checkpoints/step-2/theta/optimizer_state.pt"
                ),
            )
        )

        status = trainer._restore_optimizer_state(torch, optimizer)

        self.assertEqual(status, "restored_optimizer")
        torch.load.assert_called_once_with(
            "/checkpoints/step-2/theta/optimizer_state.pt",
            map_location="cuda:3",
            weights_only=False,
        )
        optimizer.load_state_dict.assert_called_once_with(
            {"state": {}, "param_groups": []}
        )

        torch.load.return_value["committed_step"] = 1
        with self.assertRaisesRegex(ValueError, "immediately precede"):
            trainer._restore_optimizer_state(torch, optimizer)

    def test_step_two_saves_optimizer_state_and_committed_step(self) -> None:
        torch = MagicMock()
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"state": {}}
        trainer = Qwen35OnePassSmokeTrainer(
            _config(
                update_step=2,
                behavior_adapter_checkpoint="/checkpoints/step-1/theta",
            )
        )
        with TemporaryDirectory() as directory:
            path, saved = trainer._save_optimizer_state(
                torch,
                optimizer,
                Path(directory),
            )

        self.assertTrue(saved)
        self.assertTrue(path.endswith("optimizer_state.pt"))
        payload, saved_path = torch.save.call_args.args
        self.assertEqual(payload["committed_step"], 2)
        self.assertEqual(payload["optimizer_state_dict"], {"state": {}})
        self.assertEqual(saved_path.name, "optimizer_state.pt")

    def test_continuation_loads_same_trainable_theta_on_both_replicas(self) -> None:
        models = [MagicMock(), MagicMock()]
        for model in models:
            model.config = SimpleNamespace(use_cache=True)
            model.named_parameters.return_value = ()
        bases = [object(), object()]
        auto_model = MagicMock()
        auto_model.from_pretrained.side_effect = bases
        peft_model = MagicMock()
        peft_model.from_pretrained.side_effect = models

        torch_module = ModuleType("torch")
        torch_module.bfloat16 = object()
        peft_module = ModuleType("peft")
        peft_module.LoraConfig = MagicMock()
        peft_module.PeftModel = peft_model
        peft_module.get_peft_model = MagicMock()
        transformers_module = ModuleType("transformers")
        transformers_module.AutoModelForMultimodalLM = auto_model
        trainer = Qwen35OnePassSmokeTrainer(
            _config(
                update_step=2,
                behavior_adapter_checkpoint="/checkpoints/step-1/theta",
                gradient_checkpointing=False,
            )
        )

        with patch.dict(
            sys.modules,
            {
                "torch": torch_module,
                "peft": peft_module,
                "transformers": transformers_module,
            },
        ):
            loaded_torch, learner, replica = trainer._load_models()

        self.assertIs(loaded_torch, torch_module)
        self.assertIs(learner, models[0])
        self.assertIs(replica, models[1])
        self.assertEqual(
            peft_model.from_pretrained.call_args_list,
            [
                call(
                    bases[0],
                    "/checkpoints/step-1/theta",
                    adapter_name="theta",
                    is_trainable=True,
                ),
                call(
                    bases[1],
                    "/checkpoints/step-1/theta",
                    adapter_name="theta",
                    is_trainable=True,
                ),
            ],
        )
        for model in models:
            model.set_adapter.assert_called_once_with("theta")
            self.assertFalse(model.config.use_cache)


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
