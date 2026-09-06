from __future__ import annotations

import math
import unittest

from src.interactive.ttb_objective import (
    edge_action_logprob_mean,
    tempered_trajectory_balance_loss,
    tempered_trajectory_balance_residual,
    torch_tempered_trajectory_balance_loss,
)


class TTBObjectiveTests(unittest.TestCase):
    def test_edge_mean_uses_only_structured_action_tokens(self) -> None:
        # The reasoning log-probability is deliberately not an input to the
        # edge objective.  It belongs to the model context.
        reasoning_logprob_sum = -1000.0
        action_logprob_sum = -6.0
        self.assertAlmostEqual(edge_action_logprob_mean(action_logprob_sum, 3), -2.0)
        self.assertNotAlmostEqual(
            edge_action_logprob_mean(action_logprob_sum, 3),
            (reasoning_logprob_sum + action_logprob_sum) / 4,
        )

    def test_python_residual_and_step_normalized_loss(self) -> None:
        kwargs = dict(
            log_z=0.3,
            forward_action_logprob_sums=[-4.0, -2.0],
            backward_action_logprob_sums=[-1.0, -3.0],
            action_token_counts=[2, 1],
            r_tilde=0.5,
            beta=1.0,
        )
        expected_residual = 0.3 - 4.0 - math.log(0.5) + 3.5
        self.assertAlmostEqual(
            tempered_trajectory_balance_residual(**kwargs),
            expected_residual,
        )
        self.assertAlmostEqual(
            tempered_trajectory_balance_loss(**kwargs),
            (expected_residual / 2) ** 2,
        )

    def test_reward_shift_or_clip_is_explicit_at_call_site(self) -> None:
        base = dict(
            log_z=0.0,
            forward_action_logprob_sums=[-1.0],
            backward_action_logprob_sums=[-1.0],
            action_token_counts=[1],
        )
        with self.assertRaisesRegex(ValueError, "epsilon shift or clip"):
            tempered_trajectory_balance_loss(**base, r_tilde=0.0)
        shifted = tempered_trajectory_balance_loss(**base, r_tilde=0.1)
        self.assertAlmostEqual(shifted, math.log(10.0) ** 2)

    def test_python_input_validation_is_strict(self) -> None:
        base = dict(
            log_z=0.0,
            forward_action_logprob_sums=[-1.0],
            backward_action_logprob_sums=[-1.0],
            action_token_counts=[1],
            r_tilde=1.0,
        )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            tempered_trajectory_balance_loss(
                **{**base, "backward_action_logprob_sums": [-1.0, -2.0]}
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            tempered_trajectory_balance_loss(
                **{**base, "action_token_counts": [0]}
            )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            tempered_trajectory_balance_loss(
                **{**base, "forward_action_logprob_sums": [float("nan")]}
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            tempered_trajectory_balance_loss(
                **{
                    **base,
                    "forward_action_logprob_sums": [],
                    "backward_action_logprob_sums": [],
                    "action_token_counts": [],
                }
            )

    def test_torch_loss_matches_python_and_backpropagates(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - depends on test environment
            self.skipTest("PyTorch is not installed")

        log_z = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        forward = torch.tensor([-4.0, -2.0], dtype=torch.float64, requires_grad=True)
        backward = torch.tensor([-1.0, -3.0], dtype=torch.float64, requires_grad=True)
        reward = torch.tensor(0.5, dtype=torch.float64)
        counts = torch.tensor([2, 1], dtype=torch.int64)

        loss = torch_tempered_trajectory_balance_loss(
            log_z=log_z,
            forward_action_logprob_sums=forward,
            backward_action_logprob_sums=backward,
            action_token_counts=counts,
            r_tilde=reward,
            beta=1.0,
        )
        expected = tempered_trajectory_balance_loss(
            log_z=0.3,
            forward_action_logprob_sums=[-4.0, -2.0],
            backward_action_logprob_sums=[-1.0, -3.0],
            action_token_counts=[2, 1],
            r_tilde=0.5,
            beta=1.0,
        )
        self.assertAlmostEqual(loss.detach().item(), expected)
        loss.backward()
        self.assertIsNotNone(log_z.grad)
        self.assertIsNotNone(forward.grad)
        self.assertIsNotNone(backward.grad)
        self.assertTrue(torch.isfinite(log_z.grad).item())
        self.assertGreater(torch.linalg.vector_norm(forward.grad).item(), 0.0)
        self.assertGreater(torch.linalg.vector_norm(backward.grad).item(), 0.0)

    def test_torch_input_validation_rejects_invalid_edge_shapes(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - depends on test environment
            self.skipTest("PyTorch is not installed")

        with self.assertRaisesRegex(ValueError, "equal shapes"):
            torch_tempered_trajectory_balance_loss(
                log_z=torch.tensor(0.0),
                forward_action_logprob_sums=torch.tensor([-1.0]),
                backward_action_logprob_sums=torch.tensor([-1.0, -2.0]),
                action_token_counts=torch.tensor([1]),
                r_tilde=torch.tensor(1.0),
            )

        with self.assertRaisesRegex(TypeError, "non-boolean"):
            torch_tempered_trajectory_balance_loss(
                log_z=torch.tensor(0.0),
                forward_action_logprob_sums=torch.tensor([-1.0]),
                backward_action_logprob_sums=torch.tensor([-1.0]),
                action_token_counts=torch.tensor([True]),
                r_tilde=torch.tensor(1.0),
            )


if __name__ == "__main__":
    unittest.main()
