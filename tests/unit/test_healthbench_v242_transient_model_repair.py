"""Synthetic model/provider recovery domains; no network or model inference."""

import asyncio
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentFailureRecord, AgentResponse, AgentRuntimeError, ExecutionPhase
from src.interactive.agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStateError
from tests.unit.test_agent_graph import make_multi_provider_registry


class NoCallGateway:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse("A synthetic response after an explicitly selected repair.")


def environment(*, enabled=True, status=429, typed=True):
    model_registry = make_multi_provider_registry()
    graph = AgentGraph([AgentNode("source", "balanced", "Produce the requested complete response.")],
                       output_agent_id="source")
    gateway = NoCallGateway()
    env = AgentWorkflowEnv(model_registry, gateway, graph=graph, problem="Synthetic task.",
        execute_on_edit=False, recovery_policy="preserve_diagnose_repair_augment",
        allow_same_provider_transient_repair=enabled)
    record = AgentFailureRecord(
        request_id="synthetic-failure", agent_id="source", phase=ExecutionPhase.SINGLE,
        graph_revision=graph.revision, error_type="OpenAICompatibleGatewayError",
        message=f"provider request failed for provider-a: HTTP {status}",
        metadata={"http_status": status, "model_id": "balanced", "provider_id": "provider-a"} if typed else {},
    )
    env._record_failure_state((record,), current_agent_ids={"source"})
    return env, gateway, record


def feedback(env, record):
    error = AgentRuntimeError("synthetic request failure", failure_records=(record,))
    return json.loads(env._execution_error_feedback(error).split("=", 1)[1])["failed_agents"][0]


def target(env):
    return env.model_admissible_action_targets()["modify_agent"]["per_agent_candidates"][0]


def modify(model, **extra):
    return AgentAction(action_type=AgentActionType.MODIFY_AGENT, agent_id="source", model_id=model, **extra)


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_opt_in_is_strictly_boolean(value):
    with pytest.raises(AgentWorkflowStateError, match="allow_same_provider_transient_repair"):
        environment(enabled=value)


def test_429_opt_in_exposes_all_compatible_alternatives_in_all_three_boundaries():
    env, gateway, record = environment()
    expected = ["alternate", "cheap", "fast"]
    revision = env.revision
    assert env._provider_repair_model_ids("source") == tuple(expected)
    preferred = feedback(env, record)["preferred_repair"]
    assert preferred["admitted_model_ids"] == expected
    assert "avoid_provider_id" not in preferred
    assert "fallback_provider_id" not in preferred
    candidate = target(env)
    assert candidate["discrete_value_domains"]["model_id"] == expected
    assert "avoid_provider_id" not in candidate
    for model in expected:
        assert env._provider_repair_admission_issue(modify(model)) is None
    assert env._provider_repair_admission_issue(modify("balanced")) is not None
    assert env._provider_repair_admission_issue(modify("cheap", contract="Change the task.")) is not None
    availability = env.model_availability_receipt()
    assert availability["available_model_ids"] == list(env.model_registry.model_ids)
    assert availability["unavailable_model_ids"] == []
    assert availability["unavailable_provider_ids"] == []
    assert env.revision == revision and env.graph.get_node("source").model_id == "balanced"
    assert gateway.requests == []


def test_default_429_retains_cross_provider_only_recovery():
    env, _, record = environment(enabled=False)
    expected = ["alternate", "fast"]
    assert env._provider_repair_model_ids("source") == tuple(expected)
    assert feedback(env, record)["preferred_repair"]["admitted_model_ids"] == expected
    assert feedback(env, record)["preferred_repair"]["avoid_provider_id"] == "provider-a"
    assert target(env)["discrete_value_domains"]["model_id"] == expected
    assert env._provider_repair_admission_issue(modify("cheap")) is not None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 500])
def test_other_statuses_keep_exact_old_domain_and_availability(status):
    old, _, old_record = environment(enabled=False, status=status)
    new, _, new_record = environment(enabled=True, status=status)
    assert new._provider_repair_model_ids("source") == old._provider_repair_model_ids("source")
    assert new.model_availability_receipt() == old.model_availability_receipt()
    assert feedback(new, new_record) == feedback(old, old_record)
    assert target(new) == target(old)


def test_error_text_alone_does_not_expand_the_repair_domain():
    env, _, record = environment(typed=False)
    assert env._provider_repair_model_ids("source") == ("alternate", "fast")
    assert feedback(env, record)["preferred_repair"]["avoid_provider_id"] == "provider-a"
    for value in ("429", True, None):
        altered = replace(record, metadata={"http_status": value})
        env._record_failure_state((altered,), current_agent_ids={"source"})
        assert env._provider_repair_model_ids("source") == ("alternate", "fast")


def test_profile_incompatible_or_already_unavailable_models_are_not_admitted():
    env, _, record = environment()
    # Simulate the existing trajectory-local quarantine of another model,
    # without changing the catalog, adding a provider, or invoking anything.
    env._unavailable_model_ids.add("alternate")
    with patch.object(env.runtime, "model_supports_execution_profile", side_effect=lambda model_id, *_: model_id != "fast"):
        assert env._provider_repair_model_ids("source") == ("cheap",)
        assert feedback(env, record)["preferred_repair"]["admitted_model_ids"] == ["cheap"]
        assert target(env)["discrete_value_domains"]["model_id"] == ["cheap"]
        assert env._provider_repair_admission_issue(modify("cheap")) is None
        assert env._provider_repair_admission_issue(modify("fast")) is not None
        assert env._provider_repair_admission_issue(modify("alternate")) is not None


def test_unknown_429_scope_does_not_mark_provider_avoided_if_only_cross_provider_remains():
    env, _, record = environment()
    env._unavailable_model_ids.add("cheap")
    assert env._provider_repair_model_ids("source") == ("alternate", "fast")
    assert env._provider_repair_avoid_provider_id("source") is None
    assert "avoid_provider_id" not in feedback(env, record)["preferred_repair"]


def test_same_provider_change_is_director_selected_and_preserves_contract_output_and_catalog():
    env, gateway, _ = environment()
    original = env.graph.get_node("source")
    catalog = env.model_registry.model_ids
    assert gateway.requests == []
    result = asyncio.run(env.step(json.dumps({"action": "modify_agent", "agent_id": "source", "model_id": "cheap"})))
    assert result.accepted
    assert env.graph.get_node("source") == replace(original, model_id="cheap")
    assert env.graph.output_agent_id == "source"
    assert env.model_registry.model_ids == catalog


def test_fork_preserves_the_opt_in_without_fabricating_a_failure_receipt():
    env, _, record = environment()
    forked = env.fork()
    assert forked.allow_same_provider_transient_repair is True
    # Existing fork snapshots do not carry transient failure bookkeeping.
    # Preserve that behavior; a real failure must be recorded independently.
    assert forked._provider_repair_model_ids("source") == ()
    forked._record_failure_state((record,), current_agent_ids={"source"})
    assert forked._provider_repair_model_ids("source") == env._provider_repair_model_ids("source")
