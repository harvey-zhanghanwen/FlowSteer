from __future__ import annotations

from dataclasses import replace
import math
import unittest

from src.interactive.persistence import GraphSnapshotEvent
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)
from src.interactive.versioning import VersionBundle


def versions(prompt: str = "prompt-v1") -> VersionBundle:
    return VersionBundle(
        policy="policy-v1",
        model_catalog="catalog-v1",
        evaluator="eval-v1",
        prompt=prompt,
        tool="tools-v1",
    )


def turn(*, receipt: bool = True, policy: str = "policy-v1") -> TurnRecord:
    snapshot = GraphSnapshotEvent.create(1, {"nodes": [{"id": "a"}]})
    return TurnRecord(
        turn_id="turn-1",
        round_index=0,
        prompt="prompt",
        policy_response='{"action":"finish"}',
        prompt_token_ids=[1, 2],
        output_token_ids=[3, 4],
        behavior_log_probs=[-0.1, -0.2],
        executed_prefix_tokens=2,
        action={"action": "finish"},
        canvas_feedback="done",
        graph_revision=1,
        graph_snapshot=snapshot.to_dict()["graph"],
        policy_version=policy,
        graph_snapshot_id=snapshot.snapshot_id,
        previous_graph_snapshot_id=None,
        receipt_verified=receipt,
    )


def trajectory(task_split: str = "train", **changes: object) -> TrajectoryRecord:
    values = dict(
        trajectory_id="trajectory-1",
        task=TaskRecord("q1", "question", "answer", task_split),
        group_id="q1:exploit:policy-v1",
        condition_id="exploit",
        rollout_id="r1",
        versions=versions(),
        turns=[turn()],
        final_answer="answer",
        evaluation=EvaluationReceipt("eval-v1", True, 1.0),
        termination_reason="finish",
        explicit_finish=True,
    )
    values.update(changes)
    return TrajectoryRecord(**values)


class RecordTests(unittest.TestCase):
    def test_eligible_trajectory_requires_exact_receipts_and_snapshot_hash(self) -> None:
        self.assertTrue(trajectory().grpo_eligible)
        self.assertFalse(trajectory(turns=[turn(receipt=False)]).grpo_eligible)
        self.assertFalse(trajectory(turns=[turn(policy="other")]).grpo_eligible)
        self.assertFalse(
            trajectory(evaluation=EvaluationReceipt("different-evaluator", True, 1.0)).grpo_eligible
        )
        tampered = replace(turn(), graph_snapshot={"nodes": [{"id": "tampered"}]})
        self.assertFalse(trajectory(turns=[tampered]).grpo_eligible)

    def test_group_key_fingerprints_full_external_regime(self) -> None:
        left = trajectory()
        right = trajectory(versions=versions(prompt="prompt-v2"))
        self.assertNotEqual(left.group_key, right.group_key)

    def test_test_and_forced_probe_rollouts_are_ineligible(self) -> None:
        self.assertFalse(trajectory("test").grpo_eligible)
        self.assertFalse(trajectory(forced_probe=True).grpo_eligible)

    def test_nonfinite_evaluator_reward_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationReceipt("eval", True, math.nan)


if __name__ == "__main__":
    unittest.main()
