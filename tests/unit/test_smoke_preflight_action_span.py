"""CPU-only checks of receipt preflight against the existing action mask."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.interactive.smoke_trainer import (
    Qwen35OnePassSmokeTrainer,
    SmokeTrainerConfig,
)


class SmokePreflightActionSpanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = Qwen35OnePassSmokeTrainer(
            SmokeTrainerConfig(
                model_path="unused-in-cpu-test",
                tokenizer_path="unused-in-cpu-test",
                learner_device="cpu",
                gradient_replica_device="cpu",
                gradient_worker_count=1,
            )
        )
        self.group_key = ("task", "condition", "policy")

    def preflight(self, computed, behavior, action_tokens=2, output_count=3):
        turn = SimpleNamespace(
            output_token_ids=tuple(range(output_count)),
            behavior_log_probs=tuple(behavior),
            executed_prefix_tokens=action_tokens,
        )
        record = SimpleNamespace(turns=(turn,))
        item = SimpleNamespace(trajectory_id="trajectory")
        model = MagicMock()
        with patch.object(
            self.trainer,
            "_turn_log_probs",
            return_value=torch.tensor(computed, dtype=torch.float32, device="cpu"),
        ) as log_probs:
            result = self.trainer._preflight_partition(
                model,
                "cpu",
                [(self.group_key, [item])],
                {"trajectory": record},
            )[self.group_key]
        model.eval.assert_called_once_with()
        return result, log_probs

    def test_large_unused_suffix_delta_does_not_reject_group(self):
        result, log_probs = self.preflight(
            [-0.5, -0.25, -20.0], [-0.5, -0.25, -0.1]
        )
        self.assertEqual(result, (0.0, True, ""))
        log_probs.assert_called_once()
        self.assertEqual(self.trainer.config.behavior_logprob_tolerance, 0.25)

    def test_executed_action_delta_still_rejects_entire_group(self):
        result, _ = self.preflight([-0.5, -0.6, -0.1], [-0.5, -0.25, -0.1])
        self.assertFalse(result[1])
        self.assertEqual(result[2], "behavior_logprob_tolerance_exceeded")
        self.assertGreater(result[0], 0.25)

    def test_action_delta_at_existing_threshold_is_accepted(self):
        result, _ = self.preflight([-0.5, -0.5, -3.0], [-0.5, -0.25, -0.1])
        self.assertEqual(result, (0.25, True, ""))

    def test_action_nonfinite_still_rejects(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                result, _ = self.preflight([-0.5, invalid, -0.1], [-0.5, -0.25, -0.1])
                self.assertFalse(result[1])
                self.assertEqual(result[2], "behavior_logprob_tolerance_exceeded")

    def test_complete_receipt_shape_checked_before_action_slice(self):
        # A slice alone would hide the missing or additional suffix entry.
        for computed, behavior in (
            ([-0.5, -0.25], [-0.5, -0.25, -0.1]),
            ([-0.5, -0.25, -0.1, -0.2], [-0.5, -0.25, -0.1]),
        ):
            with self.subTest(computed=computed):
                result, _ = self.preflight(computed, behavior)
                self.assertEqual(result, (0.0, False, "behavior_receipt_shape_mismatch"))

    def test_invalid_action_span_rejected_before_teacher_forcing(self):
        for action_tokens in (0, -1, 4, True, 2.0, "2", None):
            with self.subTest(action_tokens=action_tokens):
                result, log_probs = self.preflight(
                    [-0.5, -0.25, -0.1], [-0.5, -0.25, -0.1], action_tokens
                )
                self.assertEqual(result, (0.0, False, "invalid_executed_action_span"))
                log_probs.assert_not_called()

    def test_group_without_sampled_tokens_rejected(self):
        result, log_probs = self.preflight([], [], action_tokens=0, output_count=0)
        self.assertEqual(result, (0.0, False, "invalid_executed_action_span"))
        log_probs.assert_not_called()

    def test_all_sampled_tokens_can_be_the_executed_action(self):
        result, _ = self.preflight(
            [-0.5, -0.25, -0.1], [-0.5, -0.25, -0.1], action_tokens=3
        )
        self.assertEqual(result, (0.0, True, ""))


if __name__ == "__main__":
    unittest.main()
