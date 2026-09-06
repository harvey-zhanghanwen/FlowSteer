from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from scripts.validate_md_phase0 import (
    build_phase0_acceptance_receipt,
    validate_trajectory_payload,
)
from src.interactive.persistence import EvidenceStore, GraphSnapshotEvent
from src.interactive.records import (
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


def _sampling(task_id: str) -> dict:
    base_seed = 71
    coordinate = ScientificSamplingCoordinate(
        sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=base_seed),
        schedule_purpose="exploit",
        ordered_sequence_hash=stable_hash([task_id]),
        sequence_position=0,
        task_id=task_id,
        optimizer_step_or_anchor_ordinal=0,
    )
    return {
        "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
        "base_seed": base_seed,
        "coordinate": coordinate.to_value(),
        "phase": GenerationPhase.ACTION.value,
    }


def _trajectory() -> TrajectoryRecord:
    task_id = "triviaqa:phase0-example"
    sampling = _sampling(task_id)
    coordinate = ScientificSamplingCoordinate.from_value(sampling["coordinate"])
    first_snapshot = GraphSnapshotEvent.create(
        4,
        {"nodes": [{"id": "retriever"}, {"id": "output"}], "output": "output"},
    )
    second_snapshot = GraphSnapshotEvent.create(
        6,
        {
            "nodes": [{"id": "retriever"}, {"id": "output"}],
            "relations": [{"source": "retriever", "target": "output"}],
            "output": "output",
        },
        first_snapshot.snapshot_id,
    )
    final_snapshot = GraphSnapshotEvent.create(
        6,
        second_snapshot.to_dict()["graph"],
        second_snapshot.snapshot_id,
    )

    def turn(
        index: int,
        snapshot: GraphSnapshotEvent,
        action: dict,
        execution: ExecutionRecord | None,
        *,
        execution_reused: bool = False,
    ) -> TurnRecord:
        return TurnRecord(
            turn_id=f"turn-{index}",
            round_index=index,
            prompt=f"prompt-{index}",
            policy_response=json.dumps(action),
            prompt_token_ids=[10, 11, 12],
            output_token_ids=[20, 21],
            behavior_log_probs=[-0.1, -0.2],
            executed_prefix_tokens=2,
            action=action,
            canvas_feedback=f"feedback-{index}",
            graph_revision=snapshot.revision,
            graph_snapshot=snapshot.to_dict()["graph"],
            policy_version="policy-e0",
            server_weight_version="weights-e0",
            graph_snapshot_id=snapshot.snapshot_id,
            previous_graph_snapshot_id=snapshot.previous_snapshot_id,
            executions=() if execution is None else (execution,),
            execution_reused=execution_reused,
            director_request_id=f"director-request-{index}",
            director_latency_ms=10.0 + index,
            director_attempt_count=1,
            director_generation_seed=derive_generation_seed(
                base_seed=sampling["base_seed"],
                coordinate=coordinate,
                step_index=index + 1,
                phase=GenerationPhase.ACTION,
            ),
            receipt_verified=True,
        )

    first_execution = ExecutionRecord(
        execution_id="execution-1",
        experiment_id="runtime-1",
        graph_revision=4,
        agent_id="retriever",
        model_id="qwen3.5-9b-local",
        model_fingerprint="qwen35-checkpoint-e0",
        provider="sglang",
        request_hash="request-1",
        output="evidence",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        input_tokens=12,
        output_tokens=3,
        latency_ms=7.0,
    )
    second_execution = ExecutionRecord(
        execution_id="execution-2",
        experiment_id="runtime-2",
        graph_revision=6,
        agent_id="output",
        model_id="qwen3.5-9b-local",
        model_fingerprint="qwen35-checkpoint-e0",
        provider="sglang",
        request_hash="request-2",
        output="final answer",
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        input_tokens=20,
        output_tokens=4,
        latency_ms=8.0,
    )
    turns = (
        turn(0, first_snapshot, {"action": "add_subgraph"}, first_execution),
        turn(1, second_snapshot, {"action": "set_relation"}, second_execution),
        turn(
            2,
            final_snapshot,
            {"action": "finish"},
            None,
            execution_reused=True,
        ),
    )
    versions = VersionBundle(
        policy="policy-e0",
        model_catalog="catalog-e0",
        evaluator="triviaqa.official.answer.v1",
        prompt="director-prompt-e0",
        tool="triviaqa-tool-e0",
    )
    return TrajectoryRecord(
        trajectory_id="trajectory-phase0-example",
        task=TaskRecord(task_id, "question", "final answer", "train"),
        group_id=f"{task_id}:exploit:policy-e0",
        condition_id="exploit",
        rollout_id="rollout-0",
        versions=versions,
        turns=turns,
        final_answer="final answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=1.0,
            metrics={"exact_match": 1.0},
        ),
        termination_reason="finish",
        explicit_finish=True,
        director_sampling=sampling,
    )


def test_phase0_accepts_raw_and_append_only_trajectory_jsonl() -> None:
    trajectory = _trajectory()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        raw_path = root / "raw.jsonl"
        raw_path.write_text(
            json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raw_receipt = build_phase0_acceptance_receipt(raw_path)
        assert raw_receipt["accepted"] is True
        assert raw_receipt["counts"]["input_trajectories"] == 1
        assert raw_receipt["counts"]["director_turns"] == 3
        assert raw_receipt["counts"]["agent_execution_calls"] == 2

        store = EvidenceStore(root / "evidence")
        store.append_trajectory(trajectory)
        envelope_receipt = build_phase0_acceptance_receipt(
            root / "evidence" / "trajectories.jsonl"
        )
        assert envelope_receipt["accepted"] is True


def test_phase0_rejects_cached_execution_copy() -> None:
    payload = _trajectory().to_dict()
    copied_execution = deepcopy(payload["turns"][0]["executions"][0])
    copied_execution["graph_revision"] = payload["turns"][-1]["graph_revision"]
    payload["turns"][-1]["executions"] = [copied_execution]

    receipt = None
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "trajectory.jsonl"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        receipt = build_phase0_acceptance_receipt(path)

    assert receipt["accepted"] is False
    assert receipt["checks"]["execution_recorded_once"] is False
    assert "duplicates cached execution" in receipt["failures"][0]["reason"]


def test_phase0_rejects_policy_or_evaluator_version_mismatch() -> None:
    policy_payload = _trajectory().to_dict()
    policy_payload["turns"][1]["policy_version"] = "policy-other"
    policy_payload["grpo_eligible"] = False
    try:
        validate_trajectory_payload(policy_payload)
    except ValueError as exc:
        assert "behavior policy version" in str(exc)
    else:  # pragma: no cover - explicit fail path
        raise AssertionError("policy mismatch was accepted")

    evaluator_payload = _trajectory().to_dict()
    evaluator_payload["evaluation"]["evaluator_version"] = "evaluator-other"
    evaluator_payload["grpo_eligible"] = False
    try:
        validate_trajectory_payload(evaluator_payload)
    except ValueError as exc:
        assert "evaluator version" in str(exc)
    else:  # pragma: no cover - explicit fail path
        raise AssertionError("evaluator mismatch was accepted")
