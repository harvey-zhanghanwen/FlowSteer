from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import ConfigurationError
from src.interactive.persistence import EvidenceStore
from src.interactive.rollout_collector import (
    AgentGraphRolloutCollector,
    partial_trajectory_diagnostic_scope,
)
from tests.unit.test_completion_benchmark_round import _MODULE as runner
from tests.unit.test_rollout_collector import (
    FakeGateway,
    ScriptedSGLangClient,
    _orchestrator,
    _registry,
    _task,
    _versions,
)
from tests.unit.test_rollout_progress_events import _evaluation


ADD = ('{"action":"add_agent","agent_id":"solver",'
       '"model_id":"cheap-model","contract":"solve directly"}')
MODIFY = ('{"action":"modify_agent","agent_id":"solver",'
          '"contract":"repair from existing feedback"}')
SUCCESS = [ADD, '{"action":"set_output","agent_id":"solver"}', '{"action":"finish"}']


class _BlockingClient(ScriptedSGLangClient):
    def __init__(self, actions, *, error=None):
        super().__init__(actions, policy_version=_versions().policy,
                         expected_server_weight_version="default")
        self.blocked = asyncio.Event()
        self.error = error

    async def propose(self, *args, **kwargs):
        if not self.actions:
            self.blocked.set()
            if self.error is not None:
                raise self.error
            await asyncio.Future()
        return await super().propose(*args, **kwargs)


def _collector(actions, *, sink=None, evidence=None, gateway=None, error=None):
    registry = _registry()
    client = _BlockingClient(actions, error=error)
    collector = AgentGraphRolloutCollector(
        _orchestrator(registry, client, max_rounds=len(actions) + 1),
        AgentWorkflowEnv(registry, gateway=gateway or FakeGateway(), execute_on_edit=True),
        _versions(), evidence, condition_id="v240-test",
        partial_trajectory_callback=sink,
    )
    return collector, client


async def _cancel_at_next_director(collector, client, task, evaluator):
    job = asyncio.create_task(collector.collect(task, 0, evaluator))
    await asyncio.wait_for(client.blocked.wait(), timeout=2)
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job


def test_cancel_preserves_real_turns_runtime_failure_and_no_grade(tmp_path):
    class FailOnceGateway(FakeGateway):
        calls = 0

        async def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("observed mock provider failure")
            return await super().generate(request)

    async def scenario():
        diagnostics = []
        evidence = EvidenceStore(tmp_path / "evidence")
        evaluator = Mock(side_effect=AssertionError("partial answer reached evaluator"))
        collector, client = _collector(
            [ADD, MODIFY], sink=diagnostics.append, evidence=evidence,
            gateway=FailOnceGateway(),
        )
        task = replace(_task(), ground_truth="PRIVATE TARGET NOT FOR DIAGNOSTICS")
        await _cancel_at_next_director(collector, client, task, evaluator)
        evaluator.assert_not_called()
        assert len(evidence.trajectories) == len(evidence.snapshots) == 0
        return diagnostics[0]

    row = asyncio.run(scenario())
    assert row["complete"] is False and row["non_scoreable"] is True
    assert row["evaluation"] is row["final_answer"] is None
    assert row["termination_reason"] == "cancelled"
    assert row["error"]["type"] == "CancelledError"
    assert len(row["turns"]) == len(row["public_history"]) == 2
    assert row["turns"][0]["runtime_summary"]["execution_status"] == "failed"
    assert "observed mock provider failure" in row["last_known_runtime_failure"]["message"]
    execution = row["turns"][1]["executions"][0]
    assert execution["output"] == "final answer"
    assert execution["metadata"]["request"]["rendered_messages"]
    assert row["current_graph"]["revision"] == row["public_history"][-1]["revision"]
    assert row["pending_turn"]["policy_response"] is None
    assert row["pending_turn"]["missing_receipts"] == ["director_response"]
    assert row["canvas_finished"] is False
    assert "PRIVATE TARGET NOT FOR DIAGNOSTICS" not in json.dumps(row)


def test_cancel_during_canvas_keeps_returned_director_and_marks_missing_runtime():
    class BlockingGateway(FakeGateway):
        def __init__(self):
            self.calls = 0
            self.blocked = asyncio.Event()

        async def generate(self, request):
            self.calls += 1
            if self.calls == 2:
                self.blocked.set()
                await asyncio.Future()
            return await super().generate(request)

    async def scenario():
        rows = []
        gateway = BlockingGateway()
        collector, _ = _collector([ADD, MODIFY], sink=rows.append, gateway=gateway)
        evaluator = Mock()
        job = asyncio.create_task(collector.collect(_task(), 0, evaluator))
        await asyncio.wait_for(gateway.blocked.wait(), timeout=2)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        evaluator.assert_not_called()
        return rows[0]

    row = asyncio.run(scenario())
    assert len(row["turns"]) == 1
    pending = row["pending_turn"]
    assert pending["director_response_returned"] is True
    assert pending["policy_response"] == MODIFY
    assert pending["director_metadata"]["output_token_ids"]
    assert pending["canvas_result_returned"] is False
    assert pending["missing_receipts"] == ["canvas_result"]
    assert pending["executions"] == []
    assert "solver" in row["unresolved_dirty_agent_ids"]


def test_scope_saves_two_concurrent_tasks_without_cross_task_or_attempt_mix(tmp_path):
    async def scenario():
        path = tmp_path / "evaluator_private" / "partial_trajectories.jsonl"
        sink = runner._partial_trajectory_callback(
            {"partial_trajectories": path}, run_attempt_id="attempt-a",
            condition_id="v240-test",
        )
        evaluator = Mock()
        with partial_trajectory_diagnostic_scope(sink):
            pairs = [_collector([ADD]) for _ in range(2)]
            await asyncio.gather(*[
                _cancel_at_next_director(
                    collector, client,
                    replace(_task(f"task-{index}"), question=f"public question {index}"),
                    evaluator,
                )
                for index, (collector, client) in enumerate(pairs)
            ])
        evaluator.assert_not_called()
        # The scope reset prevents an unrelated later collection inheriting it.
        collector, client = _collector([ADD])
        await _cancel_at_next_director(collector, client, _task("outside"), evaluator)
        rows = runner._read_jsonl(path)
        assert len(rows) == 2
        for row in rows:
            index = row["task_id"].split("-")[-1]
            assert row["public_task_input"] == f"public question {index}"
            assert f"public question {index}" in row["turns"][0]["prompt"]
            assert row["rollout_id"].startswith(f"task-{index}:v240-test:")
            assert row["run_attempt_id"] == "attempt-a"
        sink(rows[0])
        assert len(runner._read_jsonl(path)) == 2
        runner._PartialTrajectoryJsonlSink(
            path, run_attempt_id="attempt-b", condition_id="v240-test",
        )(rows[0])
        assert len(runner._read_jsonl(path)) == 3

    asyncio.run(scenario())


def test_success_path_and_disabled_config_never_write_partial_checkpoint(tmp_path):
    async def scenario():
        path = tmp_path / "evaluator_private" / "partial_trajectories.jsonl"
        evidence = EvidenceStore(tmp_path / "evidence")
        sink = runner._partial_trajectory_callback(
            {"partial_trajectories": path}, run_attempt_id="attempt-a",
            condition_id="v240-test",
        )
        collector, _ = _collector(SUCCESS, sink=sink, evidence=evidence)
        trajectory = await collector.collect(_task(), 0, lambda *_: _evaluation())
        assert trajectory.explicit_finish and trajectory.evaluation.valid
        assert len(evidence.trajectories) == 1
        assert not path.exists()
        assert runner._partial_trajectory_callback(
            {}, run_attempt_id="attempt-a", condition_id="v240-test",
        ) is None

    asyncio.run(scenario())
    storage = {field: str(tmp_path / f"{name}.jsonl") for name, field in {
        "selected": "selected_tasks_path", "direct": "direct_predictions_path",
        "trajectories": "trajectories_path", "failures": "failures_path",
        "paired": "paired_results_path", "wrong": "wrong_demos_path",
        "manifest": "manifest_path", "preflight": "preflight_receipt_path",
        "report_json": "report_json_path", "report_markdown": "report_markdown_path",
    }.items()}
    assert "partial_trajectories" not in runner._paths({"storage": storage}, tmp_path)
    storage["partial_trajectories_path"] = str(tmp_path / "unrestricted.jsonl")
    with pytest.raises(ConfigurationError, match="evaluator_private"):
        runner._paths({"storage": storage}, tmp_path)
    shared = str(tmp_path / "evaluator_private" / "scored.jsonl")
    storage.update(partial_trajectories_path=shared, trajectories_path=shared)
    with pytest.raises(ConfigurationError, match="overlap"):
        runner._paths({"storage": storage}, tmp_path)


def test_failure_receipt_and_broken_sink_preserve_original_exception():
    async def scenario():
        rows = []
        original = ValueError("real director failure")
        evaluator = Mock()
        collector, _ = _collector([ADD], sink=rows.append, error=original)
        with pytest.raises(ValueError) as caught:
            await collector.collect(_task(), 0, evaluator)
        assert caught.value is original
        assert rows[0]["error"] == {"type": "ValueError", "message": str(original)}
        assert rows[0]["termination_reason"] == "failed"
        assert len(rows[0]["turns"]) == 1
        for sink_error in (RuntimeError("disk failure"), asyncio.CancelledError()):
            def broken_sink(_row):
                raise sink_error

            collector, _ = _collector([ADD], sink=broken_sink, error=original)
            with pytest.raises(ValueError) as caught:
                await collector.collect(_task(), 0, evaluator)
            assert caught.value is original
            collector, client = _collector([ADD], sink=broken_sink)
            await _cancel_at_next_director(collector, client, _task(), evaluator)
        evaluator.assert_not_called()

    asyncio.run(scenario())


def test_cancelled_lock_waiter_never_snapshots_the_active_task():
    async def scenario():
        rows = []
        collector, client = _collector([ADD], sink=rows.append)
        evaluator = Mock()
        owner = asyncio.create_task(collector.collect(_task("owner"), 0, evaluator))
        await asyncio.wait_for(client.blocked.wait(), timeout=2)
        waiter = asyncio.create_task(collector.collect(_task("waiter"), 0, evaluator))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert rows == []
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert len(rows) == 1 and rows[0]["task_id"] == "owner"
        evaluator.assert_not_called()

    asyncio.run(scenario())
