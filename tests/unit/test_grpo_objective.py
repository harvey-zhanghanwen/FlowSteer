from __future__ import annotations

import unittest

from src.interactive.grpo_objective import (
    GRPOTrajectory,
    action_masked_one_pass_loss,
    same_condition_advantages,
)


def trajectory(
    trajectory_id: str,
    reward: float,
    *,
    condition: str = "exploit",
    policy: str = "p1",
    **flags: object,
) -> GRPOTrajectory:
    return GRPOTrajectory(
        trajectory_id=trajectory_id,
        task_id="q1",
        condition_id=condition,
        policy_version=policy,
        terminal_reward=reward,
        token_log_probs=[-1.0, -2.0, -100.0],
        action_mask=[1, 1, 0],
        **flags,
    )


class GRPOObjectiveTests(unittest.TestCase):
    def test_same_condition_advantage_and_length_normalization(self) -> None:
        left = trajectory("a", 0.0)
        right = trajectory("b", 1.0)
        advantages = same_condition_advantages([left, right])
        self.assertEqual(advantages.tolist(), [-1.0, 1.0])
        result = action_masked_one_pass_loss([left, right])
        self.assertAlmostEqual(result.loss, 0.0)
        self.assertEqual(result.eligible_trajectories, 2)

    def test_singleton_and_constant_groups_have_zero_information(self) -> None:
        result = action_masked_one_pass_loss(
            [trajectory("a", 1.0), trajectory("b", 1.0), trajectory("c", 0.2, condition="other")]
        )
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(result.zero_information_groups, 2)

    def test_group_key_includes_condition_and_policy(self) -> None:
        result = action_masked_one_pass_loss(
            [trajectory("a", 0.0), trajectory("b", 1.0, condition="probe-visible")]
        )
        self.assertEqual(result.groups, 2)
        self.assertTrue(all(value == 0.0 for value in result.advantages.values()))

    def test_forced_or_reconstructed_trajectory_is_excluded(self) -> None:
        result = action_masked_one_pass_loss(
            [
                trajectory("a", 0.0),
                trajectory("b", 1.0, forced_probe=True),
                trajectory("c", 1.0, reconstructed_context=True),
            ]
        )
        self.assertEqual(result.eligible_trajectories, 1)
        self.assertEqual(set(result.advantages), {"a"})

    def test_group_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            same_condition_advantages(
                [trajectory("a", 0.0), trajectory("b", 1.0, policy="p2")]
            )


if __name__ == "__main__":
    unittest.main()
