from __future__ import annotations

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import call, MagicMock, patch

import numpy as np
import torch

from src.interactive.smoke_trainer import (
    Qwen35OnePassSmokeTrainer,
    SmokeTrainerConfig,
    SmokeTrainingSummary,
)


BASE_POLICY = "qwen35-9b-base-step-0000"
STEP_ONE_POLICY = "qwen35-9b-md-grpo-step-0001"


def _config(**changes: object) -> SmokeTrainerConfig:
    values: dict[str, object] = {
        "model_path": "/models/Qwen3.5-9B",
        "tokenizer_path": "/tokenizers/Qwen3.5-9B",
        "behavior_policy_version": BASE_POLICY,
        "updated_policy_version": STEP_ONE_POLICY,
        "learner_device": "cuda:3",
        "gradient_replica_device": "cuda:5",
    }
    values.update(changes)
    return SmokeTrainerConfig(**values)


class CompleteCheckpointStateTests(unittest.TestCase):
    def test_cuda_rng_is_scoped_to_the_two_training_devices(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
        trainer = Qwen35OnePassSmokeTrainer(_config(update_step=1))
        states = [
            torch.tensor([1, 2], dtype=torch.uint8),
            torch.tensor([3, 4], dtype=torch.uint8),
        ]
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step_000001_test" / "theta"
            checkpoint.mkdir(parents=True)
            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "get_rng_state", side_effect=states) as get_rng,
            ):
                state_path = trainer._save_training_state(
                    torch,
                    optimizer,
                    checkpoint,
                    checkpoint_version="step_000001_test",
                    training_metadata={"dataset": "triviaqa"},
                )
            self.assertEqual(
                get_rng.call_args_list,
                [call("cuda:3"), call("cuda:5")],
            )

            resumed = Qwen35OnePassSmokeTrainer(
                _config(
                    update_step=2,
                    behavior_policy_version=STEP_ONE_POLICY,
                    updated_policy_version="qwen35-9b-md-grpo-step-0002",
                    behavior_adapter_checkpoint=str(checkpoint),
                    optimizer_state_checkpoint=state_path,
                    exact_optimizer_continuation=True,
                )
            )
            resumed_optimizer = torch.optim.AdamW(
                [torch.nn.Parameter(torch.tensor([1.0]))],
                lr=1.0e-3,
            )
            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "set_rng_state") as set_rng,
            ):
                resumed._restore_optimizer_state(torch, resumed_optimizer)
            self.assertEqual(set_rng.call_count, 2)
            for actual, expected_state, expected_device in zip(
                set_rng.call_args_list,
                states,
                ("cuda:3", "cuda:5"),
            ):
                self.assertTrue(torch.equal(actual.args[0], expected_state))
                self.assertEqual(actual.kwargs["device"], expected_device)

    def test_complete_state_round_trip_restores_optimizer_and_rng(self) -> None:
        random.seed(17)
        np.random.seed(19)
        torch.manual_seed(23)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
        parameter.grad = torch.tensor([0.5])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        trainer = Qwen35OnePassSmokeTrainer(_config(update_step=1))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step_000001_test" / "theta"
            checkpoint.mkdir(parents=True)
            with patch.object(torch.cuda, "is_available", return_value=False):
                state_path = trainer._save_training_state(
                    torch,
                    optimizer,
                    checkpoint,
                    checkpoint_version="step_000001_test",
                    training_metadata={
                        "objective": "action_masked_one_pass_grpo",
                        "dataset": "triviaqa",
                    },
                )

            payload = torch.load(
                state_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["global_step"], 1)
            self.assertEqual(payload["update_step"], 1)
            self.assertEqual(payload["committed_step"], 1)
            self.assertEqual(payload["behavior_policy_version"], BASE_POLICY)
            self.assertEqual(payload["updated_policy_version"], STEP_ONE_POLICY)
            self.assertEqual(payload["checkpoint_version"], "step_000001_test")
            self.assertEqual(payload["checkpoint_dir"], ".")
            self.assertIsNone(payload["scheduler_state_dict"])
            self.assertEqual(
                payload["scheduler_contract"],
                {
                    "enabled": False,
                    "scheduler_class": None,
                    "status": "disabled_constant_learning_rate",
                },
            )
            self.assertIn("random_state", payload)
            self.assertIn("np_random_state", payload)
            self.assertIn("torch_random_state", payload)
            self.assertIn("torch_cuda_random_state", payload)
            self.assertEqual(payload["torch_cuda_rng_status"], "not_available")

            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(3)
            for _ in range(10):
                random.random()
                np.random.random()
                torch.rand(3)

            resumed_parameter = torch.nn.Parameter(torch.tensor([1.0]))
            resumed_optimizer = torch.optim.AdamW(
                [resumed_parameter],
                lr=1.0e-3,
            )
            resumed = Qwen35OnePassSmokeTrainer(
                _config(
                    update_step=2,
                    behavior_policy_version=STEP_ONE_POLICY,
                    updated_policy_version="qwen35-9b-md-grpo-step-0002",
                    behavior_adapter_checkpoint=str(checkpoint),
                    optimizer_state_checkpoint=state_path,
                    exact_optimizer_continuation=True,
                )
            )
            with patch.object(torch.cuda, "is_available", return_value=False):
                status = resumed._restore_optimizer_state(
                    torch,
                    resumed_optimizer,
                )

            self.assertEqual(status, "restored_complete_training_state")
            self.assertTrue(resumed._rng_state_restored)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            self.assertTrue(resumed_optimizer.state_dict()["state"])

    def test_incomplete_state_is_rejected_before_optimizer_restore(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
        trainer = Qwen35OnePassSmokeTrainer(_config(update_step=1))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step_000001_test" / "theta"
            checkpoint.mkdir(parents=True)
            with patch.object(torch.cuda, "is_available", return_value=False):
                state_path = trainer._save_training_state(
                    torch,
                    optimizer,
                    checkpoint,
                    checkpoint_version="step_000001_test",
                    training_metadata={"dataset": "triviaqa"},
                )
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            del payload["np_random_state"]
            torch.save(payload, state_path)

            resumed_optimizer = torch.optim.AdamW(
                [torch.nn.Parameter(torch.tensor([1.0]))],
                lr=1.0e-3,
            )
            resumed = Qwen35OnePassSmokeTrainer(
                _config(
                    update_step=2,
                    behavior_policy_version=STEP_ONE_POLICY,
                    updated_policy_version="qwen35-9b-md-grpo-step-0002",
                    behavior_adapter_checkpoint=str(checkpoint),
                    optimizer_state_checkpoint=state_path,
                    exact_optimizer_continuation=True,
                )
            )
            with self.assertRaisesRegex(ValueError, "missing fields"):
                resumed._restore_optimizer_state(torch, resumed_optimizer)

    def test_adapter_receipt_state_and_directory_are_cross_checked(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
        saver = Qwen35OnePassSmokeTrainer(_config(update_step=1))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step_000001_test" / "theta"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            with patch.object(torch.cuda, "is_available", return_value=False):
                state_path = saver._save_training_state(
                    torch,
                    optimizer,
                    checkpoint,
                    checkpoint_version="step_000001_test",
                    training_metadata={"dataset": "triviaqa"},
                )
            receipt = {
                "behavior_policy_version": BASE_POLICY,
                "updated_policy_version": STEP_ONE_POLICY,
                "global_step": 1,
                "update_step": 1,
                "committed_step": 1,
                "checkpoint_ready": True,
                "checkpoint_version": "step_000001_test",
                "checkpoint_dir": ".",
                "training_state_checkpoint": "training_state.pt",
                "training_state_format": "flowsteer-one-pass-grpo-training-state-v2",
            }
            metadata_path = checkpoint / "policy_version.json"
            metadata_path.write_text(json.dumps(receipt), encoding="utf-8")
            resumed = Qwen35OnePassSmokeTrainer(
                _config(
                    update_step=2,
                    behavior_policy_version=STEP_ONE_POLICY,
                    updated_policy_version="qwen35-9b-md-grpo-step-0002",
                    behavior_adapter_checkpoint=str(checkpoint),
                    optimizer_state_checkpoint=state_path,
                    exact_optimizer_continuation=True,
                )
            )
            resumed._validate_behavior_checkpoint_metadata()

            receipt["checkpoint_version"] = "wrong-version"
            metadata_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint version"):
                resumed._validate_behavior_checkpoint_metadata()

    def test_backward_returns_exact_consumed_ids_and_group_keys(self) -> None:
        trainer = Qwen35OnePassSmokeTrainer(_config())
        key = ("triviaqa:0", "natural", "versions")
        group = [
            SimpleNamespace(trajectory_id="trajectory-0"),
            SimpleNamespace(trajectory_id="trajectory-1"),
        ]
        turn = SimpleNamespace(output_token_ids=(1,), executed_prefix_tokens=1)
        records = {
            item.trajectory_id: SimpleNamespace(turns=(turn,)) for item in group
        }
        with patch.object(
            trainer,
            "_turn_log_probs",
            side_effect=lambda *_: torch.tensor([-0.2], requires_grad=True),
        ):
            _, consumed = trainer._backward_partition(
                MagicMock(),
                "cpu",
                [(key, group)],
                records,
                {"trajectory-0": 1.0, "trajectory-1": -1.0},
                total_groups=1,
                micro_batch_size=1,
            )
        self.assertEqual(
            consumed,
            ((key, "trajectory-0"), (key, "trajectory-1")),
        )

    def test_updated_summary_cannot_be_checkpoint_ready_without_full_state(self) -> None:
        required = {
            "optimizer_updates": 1,
            "input_trajectories": 4,
            "record_eligible_trajectories": 4,
            "exact_groups": 1,
            "informative_groups": 1,
            "trained_groups": 1,
            "trained_trajectories": 4,
            "zero_information_groups": 0,
            "excluded_groups": 0,
            "loss": 0.1,
            "grad_norm": 1.0,
            "max_behavior_logprob_delta": 0.0,
            "behavior_policy_version": BASE_POLICY,
            "updated_policy_version": STEP_ONE_POLICY,
            "micro_batch_size_used": 1,
            "oom_backoff_count": 0,
            "trainable_update_l2": 0.01,
            "checkpoint_dir": "/checkpoint/theta",
            "exclusions": {},
            "update_step": 1,
            "committed_step": 1,
            "consumed_trajectory_ids": (
                "trajectory-0",
                "trajectory-1",
                "trajectory-2",
                "trajectory-3",
            ),
            "consumed_group_keys": (("task", "condition", "policy"),),
        }
        with self.assertRaisesRegex(ValueError, "complete recoverable checkpoint"):
            SmokeTrainingSummary(**required)

        summary = SmokeTrainingSummary(
            **required,
            optimizer_state_checkpoint="/checkpoint/theta/training_state.pt",
            optimizer_state_saved=True,
            checkpoint_ready=True,
            checkpoint_version="step_000001_test",
            training_state_checkpoint="/checkpoint/theta/training_state.pt",
            training_state_format="flowsteer-one-pass-grpo-training-state-v2",
            scheduler_state_status="disabled_constant_learning_rate",
            rng_state_saved=True,
        )
        self.assertTrue(summary.checkpoint_ready)


if __name__ == "__main__":
    unittest.main()
