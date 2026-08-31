from __future__ import annotations

import asyncio

import pytest

from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.persistence import EvidenceStore
from src.interactive.rollout_collector import AgentGraphRolloutCollector
from tests.unit.test_rollout_collector import (
    EVALUATOR_VERSION,
    FakeGateway,
    ScriptedSGLangClient,
    _orchestrator,
    _registry,
    _task,
    _versions,
)


_EVENT_FIELDS = {
    "task_id",
    "rollout_id",
    "round_index",
    "stage",
    "action",
    "accepted",
    "done",
    "graph_revision",
    "timestamp",
    "error_type",
}


def _collector(events, *, evidence_store=None, progress_callback=None):
    registry = _registry()
    client = ScriptedSGLangClient(
        [
            (
                '{"action":"add_agent","agent_id":"solver",'
                '"model_id":"cheap-model","contract":"solve directly"}'
            ),
            '{"action":"set_output","agent_id":"solver"}',
            '{"action":"finish"}',
        ],
        policy_version=_versions().policy,
        expected_server_weight_version="default",
    )
    return AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=3),
        AgentWorkflowEnv(registry, gateway=FakeGateway()),
        _versions(),
        evidence_store,
        progress_callback=(
            progress_callback
            if progress_callback is not None
            else lambda event: events.append(dict(event))
        ),
    )


def _evaluation():
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "valid": True,
        "reward": 1.0,
        "metrics": {"score": 1.0},
    }


def test_progress_events_follow_validated_turn_and_evaluator_boundaries():
    events = []
    collector = _collector(events)

    trajectory = asyncio.run(
        collector.collect(_task(), 0, lambda *_args: _evaluation())
    )

    assert trajectory.explicit_finish is True
    assert [event["stage"] for event in events] == [
        "rollout_started",
        "director_started",
        "rollout_step_committed",
        "director_started",
        "rollout_step_committed",
        "director_started",
        "rollout_step_committed",
        "evaluator_started",
        "evaluator_completed",
        "rollout_completed",
    ]
    committed = [
        event for event in events if event["stage"] == "rollout_step_committed"
    ]
    assert [event["round_index"] for event in committed] == [0, 1, 2]
    assert [event["action"] for event in committed] == [
        "add_agent",
        "set_output",
        "finish",
    ]
    assert [event["accepted"] for event in committed] == [True, True, True]
    assert [event["done"] for event in committed] == [False, False, True]
    assert [event["graph_revision"] for event in committed] == [1, 2, 2]
    assert events[-1]["done"] is True
    assert events[-1]["accepted"] is True
    assert all(set(event) == _EVENT_FIELDS for event in events)
    assert all(event["task_id"] == _task().task_id for event in events)
    assert all(event["rollout_id"] == trajectory.rollout_id for event in events)
    assert all(
        isinstance(event["timestamp"], str) and event["timestamp"].endswith("Z")
        for event in events
    )
    assert all(event["error_type"] is None for event in events)


def test_cancelled_evaluator_emits_diagnostic_and_persists_no_evidence(tmp_path):
    async def scenario():
        events = []
        evaluator_started = asyncio.Event()
        evidence = EvidenceStore(tmp_path)
        collector = _collector(events, evidence_store=evidence)

        async def evaluator(*_args):
            evaluator_started.set()
            await asyncio.Future()

        rollout = asyncio.create_task(collector.collect(_task(), 0, evaluator))
        await asyncio.wait_for(evaluator_started.wait(), timeout=1.0)
        rollout.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rollout
        return events, evidence

    events, evidence = asyncio.run(scenario())

    assert [event["stage"] for event in events][-2:] == [
        "evaluator_started",
        "rollout_cancelled",
    ]
    assert events[-1]["error_type"] == "CancelledError"
    assert events[-1]["done"] is True
    assert "evaluator_completed" not in {event["stage"] for event in events}
    assert "rollout_completed" not in {event["stage"] for event in events}
    assert len(evidence.trajectories) == 0
    assert len(evidence.snapshots) == 0


def test_evaluator_error_is_reported_as_rollout_rejected():
    events = []
    collector = _collector(events)

    def evaluator(*_args):
        raise ValueError("grader unavailable")

    with pytest.raises(ValueError, match="grader unavailable"):
        asyncio.run(collector.collect(_task(), 0, evaluator))

    assert [event["stage"] for event in events][-2:] == [
        "evaluator_started",
        "rollout_rejected",
    ]
    assert events[-1]["error_type"] == "ValueError"
    assert "rollout_completed" not in {event["stage"] for event in events}


def test_diagnostic_sink_failure_does_not_change_rollout_behavior():
    def failing_sink(_event):
        raise asyncio.CancelledError

    collector = _collector([], progress_callback=failing_sink)
    trajectory = asyncio.run(
        collector.collect(_task(), 0, lambda *_args: _evaluation())
    )

    assert trajectory.explicit_finish is True
    assert trajectory.evaluation.valid is True


def test_lifecycle_completion_does_not_claim_canvas_finish():
    events = []
    registry = _registry()
    client = ScriptedSGLangClient(
        ['{"action":"finish"}'],
        policy_version=_versions().policy,
        expected_server_weight_version="default",
    )
    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=1),
        AgentWorkflowEnv(registry, gateway=FakeGateway()),
        _versions(),
        progress_callback=lambda event: events.append(dict(event)),
    )

    trajectory = asyncio.run(
        collector.collect(_task(), 0, lambda *_args: _evaluation())
    )

    assert trajectory.explicit_finish is False
    by_stage = {event["stage"]: event for event in events}
    assert by_stage["rollout_step_committed"]["done"] is False
    assert by_stage["evaluator_started"]["done"] is False
    assert by_stage["evaluator_completed"]["done"] is False
    assert by_stage["rollout_completed"]["done"] is False
