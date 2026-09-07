"""No model calls: measured deadline feedback must not alter Canvas admission."""
import asyncio
from unittest.mock import patch

import pytest

from src.interactive.director import (
    director_time_budget_observation, director_time_budget_scope,
    record_director_step_timing,
    decode_director_transcript,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from tests.unit.test_rollout_collector import FakeGateway, _orchestrator, _registry
from tests.unit.test_partial_rollout_diagnostics import _collector, SUCCESS
from tests.unit.test_rollout_progress_events import _evaluation
from tests.unit.test_rollout_collector import _task


def test_elapsed_remaining_step_cost_and_scope_reset():
    assert director_time_budget_observation() is None
    with patch("src.interactive.director.time.monotonic", return_value=10):
        with director_time_budget_scope(900):
            record_director_step_timing(round_index=1, director_seconds=30, canvas_seconds=150)
            with patch("src.interactive.director.time.monotonic", return_value=610):
                value = director_time_budget_observation()
                assert value["elapsed_seconds"] == 600
                assert value["remaining_seconds"] == 300
                assert value["last_completed_step"]["canvas_seconds"] == 150
            with director_time_budget_scope(None):
                assert director_time_budget_observation() is None
            assert director_time_budget_observation()["limit_seconds"] == 900
    assert director_time_budget_observation() is None
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            with director_time_budget_scope(invalid):
                pass


def test_budget_visible_in_canvas_but_does_not_force_finish_or_repeat_execution():
    async def scenario():
        registry = _registry()
        env = AgentWorkflowEnv(registry, gateway=FakeGateway(), execute_on_edit=True)
        env.reset("Complete the supplied public task")
        director = _orchestrator(registry, None, max_rounds=20)
        assert "time_budget" not in director._canvas_observation(env, include_task_context=False, skills=[])
        with director_time_budget_scope(0.001):
            for action in SUCCESS[:2]:
                assert (await env.step(action)).accepted
            before = env.graph.to_dict()
            repeated = await env.step(SUCCESS[0].replace('"add_agent"', '"modify_agent"'))
            assert not repeated.accepted
            assert env.graph.to_dict() == before
            allowed_before = env.model_admissible_action_types()
            view = director._canvas_observation(env, include_task_context=False, skills=[])
            assert "time_budget" in view
            assert view["time_budget"]["remaining_seconds"] <= 0.001
            assert not env.finished
            assert env.model_admissible_action_types() == allowed_before
            assert (await env.step(SUCCESS[2])).done
    asyncio.run(scenario())


def test_concurrent_task_budgets_and_real_collector_observations_are_isolated():
    async def one(limit):
        with director_time_budget_scope(limit):
            collector, _ = _collector(SUCCESS)
            result = await collector.collect(_task(), 0, lambda *_: _evaluation())
            assert result.explicit_finish
            value = director_time_budget_observation()
            assert value["limit_seconds"] == limit
            assert value["last_completed_step"]["action"] == "finish"
            first_messages = decode_director_transcript(result.turns[0].prompt)
            next_messages = decode_director_transcript(result.turns[1].prompt)
            assert any('"time_budget"' in message["content"] for message in first_messages)
            assert any('"last_completed_step"' in message["content"] for message in next_messages)
        assert director_time_budget_observation() is None

    async def scenario():
        await asyncio.gather(one(300), one(900))
    asyncio.run(scenario())
