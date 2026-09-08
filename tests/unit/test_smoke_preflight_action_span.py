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
from src.interactive.grpo_objective import torch_action_masked_one_pass_loss


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
        for action_tokens in (-1, 4, True, 2.0, "2", None):
            with self.subTest(action_tokens=action_tokens):
                result, log_probs = self.preflight(
                    [-0.5, -0.25, -0.1], [-0.5, -0.25, -0.1], action_tokens
                )
                self.assertEqual(result, (0.0, False, "invalid_executed_action_span"))
                log_probs.assert_not_called()

    def test_zero_action_turn_has_no_preflight_delta(self):
        result, log_probs = self.preflight(
            [-20.0, -10.0, -3.0], [-0.5, -0.25, -0.1], action_tokens=0
        )
        self.assertEqual(result, (0.0, True, ""))
        log_probs.assert_called_once()

    def test_zero_action_turn_still_requires_complete_receipt_shape(self):
        result, _ = self.preflight(
            [-0.5, -0.25], [-0.5, -0.25, -0.1], action_tokens=0
        )
        self.assertEqual(result, (0.0, False, "behavior_receipt_shape_mismatch"))

    def test_empty_context_turn_has_valid_zero_span(self):
        # Whole-trajectory eligibility is enforced before preflight; a turn
        # with no consumed action must not invalidate its other valid turns.
        result, log_probs = self.preflight([], [], action_tokens=0, output_count=0)
        self.assertEqual(result, (0.0, True, ""))
        log_probs.assert_called_once()

    def test_parser_failure_then_executed_action_preserves_group(self):
        turns = (
            SimpleNamespace(
                output_token_ids=(1, 2),
                behavior_log_probs=(-0.2, -0.4),
                executed_prefix_tokens=0,
                action={},
                canvas_feedback="invalid action: malformed JSON",
            ),
            SimpleNamespace(
                output_token_ids=(3, 4, 5),
                behavior_log_probs=(-0.5, -0.25, -0.1),
                executed_prefix_tokens=2,
                action={"action": "add_agent"},
            ),
        )
        item = SimpleNamespace(trajectory_id="trajectory")
        with patch.object(
            self.trainer,
            "_turn_log_probs",
            side_effect=[
                torch.tensor([-20.0, -10.0]),
                torch.tensor([-0.5, -0.25, -30.0]),
            ],
        ) as log_probs:
            result = self.trainer._preflight_partition(
                MagicMock(),
                "cpu",
                [(self.group_key, [item])],
                {"trajectory": SimpleNamespace(turns=turns)},
            )[self.group_key]
        self.assertEqual(result, (0.0, True, ""))
        self.assertEqual(log_probs.call_count, 2)

    def test_zero_action_turn_contributes_no_gradient_in_mixed_trajectory(self):
        context = torch.tensor([-20.0, -10.0], requires_grad=True)
        action = torch.tensor([-0.5, -0.25, -30.0], requires_grad=True)
        loss = torch_action_masked_one_pass_loss(
            [torch.cat([context, action])],
            [torch.tensor([0, 0, 1, 1, 0])],
            [1.0],
        )
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.375)
        torch.testing.assert_close(context.grad, torch.zeros(2))
        torch.testing.assert_close(action.grad, torch.tensor([-0.5, -0.5, 0.0]))

    def test_whole_trajectory_without_consumed_action_remains_invalid(self):
        with self.assertRaisesRegex(ValueError, "at least one consumed action token"):
            torch_action_masked_one_pass_loss(
                [torch.tensor([-0.5, -0.25], requires_grad=True)],
                [torch.tensor([0, 0])],
                [1.0],
            )

    def test_all_sampled_tokens_can_be_the_executed_action(self):
        result, _ = self.preflight(
            [-0.5, -0.25, -0.1], [-0.5, -0.25, -0.1], action_tokens=3
        )
        self.assertEqual(result, (0.0, True, ""))


if __name__ == "__main__":
    unittest.main()
