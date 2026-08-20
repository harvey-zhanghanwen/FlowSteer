from __future__ import annotations

from dataclasses import replace
import json
import unittest

from src.interactive.persistence import GraphSnapshotEvent
from src.interactive.records import (
    EvaluationReceipt,
    ExecutionRecord,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
    canonical_invoked_skill_ids,
)
from src.interactive.versioning import VersionBundle


def _versions() -> VersionBundle:
    return VersionBundle(
        policy="policy-v1",
        model_catalog="catalog-v1",
        evaluator="evaluator-v1",
        prompt="prompt-v1",
        tool="tool-v1",
    )


def _execution(*, admitted: bool) -> ExecutionRecord:
    trace_entry = {
        "turn": 1,
        "action_text": json.dumps(
            {
                "arguments": {},
                "kind": "skill",
                "name": "search",
                "resource_id": "qa.search",
                "skill_id": "skill.qa.search",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if admitted:
        trace_entry["observation_status"] = "success"
    else:
        trace_entry.update(
            {
                "observation_status": "schema_invalid",
                "public_error_code": "skill_action_not_admitted",
            }
        )
    return ExecutionRecord(
        execution_id="execution-1",
        experiment_id="run-1",
        graph_revision=1,
        agent_id="solver",
        model_id="model-1",
        model_fingerprint="fingerprint-1",
        provider="fake",
        request_hash="request-1",
        output="artifact",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        metadata={"response": {"react_trace": [trace_entry]}},
    )


def _turn(
    *,
    executions: tuple[ExecutionRecord, ...] = (),
    retrieved: tuple[str, ...] = ("skill.qa.search",),
    visible: tuple[str, ...] = ("skill.qa.search",),
) -> TurnRecord:
    snapshot = GraphSnapshotEvent.create(1, {"nodes": [{"id": "solver"}]})
    return TurnRecord(
        turn_id="turn-1",
        round_index=0,
        prompt="prompt",
        policy_response='{"action":"finish"}',
        prompt_token_ids=(1,),
        output_token_ids=(2,),
        behavior_log_probs=(-0.1,),
        executed_prefix_tokens=1,
        action={"action": "finish"},
        canvas_feedback="done",
        graph_revision=1,
        graph_snapshot=snapshot.to_dict()["graph"],
        policy_version="policy-v1",
        graph_snapshot_id=snapshot.snapshot_id,
        executions=executions,
        receipt_verified=True,
        retrieved_skill_ids=retrieved,
        visible_skill_ids=visible,
    )


def _trajectory(turn: TurnRecord, **changes: object) -> TrajectoryRecord:
    values = {
        "trajectory_id": "trajectory-1",
        "task": TaskRecord("q1", "question", "answer", "validation"),
        "group_id": "q1:condition:policy-v1",
        "condition_id": "condition",
        "rollout_id": "rollout-1",
        "versions": _versions(),
        "turns": (turn,),
        "final_answer": "answer",
        "evaluation": EvaluationReceipt("evaluator-v1", True, 1.0),
        "termination_reason": "finish",
        "explicit_finish": True,
        "active_skill_ids": ("skill.qa.search",),
        "retrieved_skill_ids": ("skill.qa.search",),
        "invoked_skill_ids": (),
    }
    values.update(changes)
    return TrajectoryRecord(**values)


class SkillInvocationReceiptTests(unittest.TestCase):
    def test_structured_receipt_round_trip_and_legacy_defaults(self) -> None:
        original = _trajectory(_turn())
        serialized = original.to_dict()
        restored = TrajectoryRecord.from_dict(serialized)

        self.assertEqual(("skill.qa.search",), restored.turns[0].retrieved_skill_ids)
        self.assertEqual(("skill.qa.search",), restored.turns[0].visible_skill_ids)
        self.assertEqual(("skill.qa.search",), restored.active_skill_ids)
        self.assertEqual(("skill.qa.search",), restored.retrieved_skill_ids)
        self.assertEqual((), restored.invoked_skill_ids)
        self.assertTrue(restored.skill_receipt_verified)

        legacy = original.to_dict()
        for field_name in (
            "active_skill_ids",
            "retrieved_skill_ids",
            "invoked_skill_ids",
            "skill_receipt_verified",
        ):
            legacy.pop(field_name)
        legacy["turns"][0].pop("retrieved_skill_ids")
        legacy["turns"][0].pop("visible_skill_ids")
        restored_legacy = TrajectoryRecord.from_dict(legacy)
        self.assertEqual((), restored_legacy.active_skill_ids)
        self.assertEqual((), restored_legacy.retrieved_skill_ids)
        self.assertEqual((), restored_legacy.invoked_skill_ids)
        self.assertTrue(restored_legacy.skill_receipt_verified)

    def test_visibility_and_false_invocation_credit_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            _turn(retrieved=("skill.a",), visible=("skill.b",))

        with self.assertRaisesRegex(ValueError, "invoked Skill IDs"):
            _trajectory(_turn(), invoked_skill_ids=("skill.qa.search",))

        with self.assertRaisesRegex(ValueError, "active Skill IDs"):
            _trajectory(_turn(), active_skill_ids=())

    def test_active_ids_are_canonical_and_retrieval_order_is_preserved(self) -> None:
        ranked_turn = _turn(
            retrieved=("skill.b", "skill.a"),
            visible=("skill.b", "skill.a"),
        )
        ranked = _trajectory(
            ranked_turn,
            active_skill_ids=("skill.a", "skill.b"),
            retrieved_skill_ids=("skill.b", "skill.a"),
        )
        self.assertEqual(("skill.a", "skill.b"), ranked.active_skill_ids)
        self.assertEqual(("skill.b", "skill.a"), ranked.retrieved_skill_ids)
        self.assertEqual(("skill.b", "skill.a"), ranked.turns[0].visible_skill_ids)
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            _trajectory(
                ranked_turn,
                active_skill_ids=("skill.b", "skill.a"),
                retrieved_skill_ids=("skill.b", "skill.a"),
            )

    def test_rejected_actionkind_skill_attempt_receives_no_credit(self) -> None:
        rejected_turn = _turn(executions=(_execution(admitted=False),))
        trajectory = _trajectory(rejected_turn)
        self.assertEqual((), canonical_invoked_skill_ids(trajectory.turns))
        self.assertEqual((), trajectory.invoked_skill_ids)
        self.assertTrue(trajectory.skill_receipt_verified)

        admitted_turn = replace(
            rejected_turn,
            executions=(_execution(admitted=True),),
        )
        with self.assertRaisesRegex(ValueError, "no admitted invocation receipt"):
            _trajectory(admitted_turn)


if __name__ == "__main__":
    unittest.main()
