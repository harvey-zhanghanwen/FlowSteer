from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

from src.interactive.persistence import GraphSnapshotEvent
from src.interactive.records import (
    CommunicationDiagnosticRecord,
    EvaluationReceipt,
    ExecutionRecord,
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
from src.interactive.versioning import VersionBundle


def versions(prompt: str = "prompt-v1") -> VersionBundle:
    return VersionBundle(
        policy="policy-v1",
        model_catalog="catalog-v1",
        evaluator="eval-v1",
        prompt=prompt,
        tool="tools-v1",
    )


def sampling(task_id: str = "q1", position: int = 0) -> dict:
    base_seed = 17
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(
            base_seed=base_seed
        ),
        schedule_purpose="exploit",
        ordered_sequence_hash=stable_hash([task_id]),
        sequence_position=position,
        task_id=task_id,
        optimizer_step_or_anchor_ordinal=0,
    )
    return {
        "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
        "base_seed": base_seed,
        "coordinate": coordinate.to_value(),
        "phase": GenerationPhase.ACTION.value,
    }


def turn(*, receipt: bool = True, policy: str = "policy-v1") -> TurnRecord:
    snapshot = GraphSnapshotEvent.create(1, {"nodes": [{"id": "a"}]})
    director_sampling = sampling()
    coordinate = ScientificSamplingCoordinate.from_value(
        director_sampling["coordinate"]
    )
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
        feedback_code="accepted_finish",
        graph_revision=1,
        graph_snapshot=snapshot.to_dict()["graph"],
        policy_version=policy,
        graph_snapshot_id=snapshot.snapshot_id,
        previous_graph_snapshot_id=None,
        director_generation_seed=derive_generation_seed(
            base_seed=director_sampling["base_seed"],
            coordinate=coordinate,
            step_index=1,
            phase=GenerationPhase.ACTION,
        ),
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
        director_sampling=sampling(),
    )
    values.update(changes)
    return TrajectoryRecord(**values)


class RecordTests(unittest.TestCase):
    def test_persisted_trajectory_round_trip_revalidates_derived_fields(self) -> None:
        original = trajectory()
        restored = TrajectoryRecord.from_dict(original.to_dict())
        self.assertEqual(
            json.dumps(original.to_dict(), sort_keys=True),
            json.dumps(restored.to_dict(), sort_keys=True),
        )
        self.assertEqual("accepted_finish", restored.turns[0].feedback_code)

        legacy = original.to_dict()
        legacy["turns"][0].pop("feedback_code")
        self.assertIsNone(TrajectoryRecord.from_dict(legacy).turns[0].feedback_code)

        tampered = original.to_dict()
        tampered["grpo_eligible"] = False
        with self.assertRaisesRegex(ValueError, "derived field"):
            TrajectoryRecord.from_dict(tampered)

    def test_communication_diagnostic_is_structurally_excluded_from_grpo(self) -> None:
        record = CommunicationDiagnosticRecord(
            diagnostic_id="diag-1",
            pair_id="pair-1",
            source_trajectory_id="trajectory-source",
            task=TaskRecord("q-heldout", "question", "answer", "validation"),
            condition_id="upstream_masked",
            communication_condition="upstream_masked",
            versions=versions(),
            graph_snapshot={"nodes": [], "relations": [], "output_agent_id": "out"},
            output_agent_id="out",
            runtime_run_id="diag-run",
            executions=[
                ExecutionRecord(
                    execution_id="execution-1",
                    experiment_id="diag-run",
                    graph_revision=1,
                    agent_id="out",
                    model_id="m1",
                    model_fingerprint="model-1",
                    provider="fake",
                    request_hash="request-1",
                    output="<answer>answer</answer>",
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=64,
                )
            ],
            final_answer="<answer>answer</answer>",
            evaluation=EvaluationReceipt(
                "eval-v1",
                True,
                1.0,
                metrics={"exact_match": 1.0},
            ),
            mask_applied_call_ids=["execution-1"],
        )

        serialized = record.to_dict()
        self.assertTrue(serialized["diagnostic_only"])
        self.assertFalse(serialized["grpo_eligible"])
        self.assertEqual("upstream_masked", serialized["communication_condition"])
        with self.assertRaises(ValueError):
            replace(record, task=TaskRecord("q-train", "q", "a", "train"))

    def test_eligible_trajectory_requires_exact_receipts_and_snapshot_hash(self) -> None:
        self.assertTrue(trajectory().grpo_eligible)
        self.assertFalse(trajectory(turns=[turn(receipt=False)]).grpo_eligible)
        self.assertFalse(trajectory(turns=[turn(policy="other")]).grpo_eligible)
        self.assertFalse(
            trajectory(evaluation=EvaluationReceipt("different-evaluator", True, 1.0)).grpo_eligible
        )
        tampered = replace(turn(), graph_snapshot={"nodes": [{"id": "tampered"}]})
        self.assertFalse(trajectory(turns=[tampered]).grpo_eligible)
        self.assertFalse(trajectory(director_sampling={}).grpo_eligible)
        wrong_seed = replace(turn(), director_generation_seed=1)
        self.assertFalse(trajectory(turns=[wrong_seed]).grpo_eligible)

    def test_group_key_fingerprints_full_external_regime(self) -> None:
        left = trajectory()
        right = trajectory(versions=versions(prompt="prompt-v2"))
        self.assertNotEqual(left.group_key, right.group_key)

    def test_test_and_forced_probe_rollouts_are_ineligible(self) -> None:
        self.assertFalse(trajectory("test").grpo_eligible)
        self.assertFalse(trajectory(forced_probe=True).grpo_eligible)

    def test_natural_max_round_failure_with_real_zero_reward_is_eligible(self) -> None:
        failure = trajectory(
            explicit_finish=False,
            termination_reason="max_rounds",
            final_answer=None,
            evaluation=EvaluationReceipt("eval-v1", True, 0.0),
        )
        self.assertTrue(failure.terminal_failure)
        self.assertTrue(failure.natural_policy_terminal)
        self.assertTrue(failure.grpo_eligible)
        self.assertTrue(failure.to_dict()["terminal_failure"])

        self.assertFalse(
            trajectory(
                explicit_finish=False,
                termination_reason="max_rounds",
                final_answer=None,
                evaluation=EvaluationReceipt("eval-v1", True, 0.5),
            ).grpo_eligible
        )
        self.assertFalse(
            trajectory(
                explicit_finish=False,
                termination_reason="max_rounds",
                final_answer=None,
                evaluation=EvaluationReceipt("eval-v1", False, None),
            ).grpo_eligible
        )
        self.assertFalse(replace(failure, api_fallback_used=True).grpo_eligible)
        self.assertFalse(replace(failure, manual_repair_used=True).grpo_eligible)
        self.assertFalse(replace(failure, forced_probe=True).grpo_eligible)

    def test_valid_lineage_fallback_round_trips_and_is_never_grpo_eligible(self) -> None:
        fallback = trajectory(
            explicit_finish=False,
            termination_reason="max_rounds",
            final_answer="answer",
            evaluation=EvaluationReceipt("eval-v1", True, 0.0),
            valid_lineage_fallback_used=True,
            valid_lineage_fallback_receipt={
                "source": "AgentWorkflowEnv.last_valid_evidence_lineage",
                "graph_revision": 1,
                "runtime_run_id": "runtime-1",
            },
        )

        self.assertTrue(fallback.terminal_failure)
        self.assertTrue(fallback.natural_policy_terminal)
        self.assertFalse(fallback.grpo_eligible)
        restored = TrajectoryRecord.from_dict(fallback.to_dict())
        self.assertTrue(restored.valid_lineage_fallback_used)
        self.assertEqual(
            restored.valid_lineage_fallback_receipt["runtime_run_id"],
            "runtime-1",
        )

    def test_old_trajectory_without_lineage_fallback_fields_remains_readable(self) -> None:
        serialized = trajectory().to_dict()
        serialized.pop("valid_lineage_fallback_used")
        serialized.pop("valid_lineage_fallback_receipt")

        restored = TrajectoryRecord.from_dict(serialized)

        self.assertFalse(restored.valid_lineage_fallback_used)
        self.assertEqual(restored.valid_lineage_fallback_receipt, {})

        old_max_rounds = trajectory(
            explicit_finish=False,
            termination_reason="max_rounds",
            final_answer="legacy intermediate answer",
        ).to_dict()
        old_max_rounds.pop("valid_lineage_fallback_used")
        old_max_rounds.pop("valid_lineage_fallback_receipt")
        self.assertFalse(
            TrajectoryRecord.from_dict(old_max_rounds).terminal_failure
        )

    def test_nonfinite_evaluator_reward_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationReceipt("eval", True, math.nan)

    def test_evaluator_details_and_director_adapter_receipts_are_serialized(self) -> None:
        evaluation = EvaluationReceipt(
            "eval-v1",
            True,
            0.5,
            metrics={"score": 0.5},
            details={"rubric": [{"criterion": "correct", "met": True}]},
        )
        self.assertEqual(
            evaluation.to_dict()["details"]["rubric"][0]["criterion"],
            "correct",
        )

        adapter_turn = replace(
            turn(),
            policy_adapter="theta_live",
            server_weight_version="default",
            director_request_id="director-request-1",
            director_latency_ms=12.5,
            director_attempt_count=2,
            director_generation_seed=9,
            runtime_summary={"block_completion_order": [["solver"]]},
        )
        serialized = adapter_turn.to_dict()
        self.assertEqual(serialized["policy_adapter"], "theta_live")
        self.assertEqual(serialized["server_weight_version"], "default")
        self.assertEqual(serialized["director_request_id"], "director-request-1")
        self.assertEqual(serialized["director_latency_ms"], 12.5)
        self.assertEqual(serialized["director_attempt_count"], 2)
        self.assertEqual(serialized["director_generation_seed"], 9)
        self.assertEqual(
            serialized["runtime_summary"]["block_completion_order"],
            [["solver"]],
        )

        with self.assertRaises(ValueError):
            EvaluationReceipt("eval-v1", False, None, details=[])


if __name__ == "__main__":
    unittest.main()
