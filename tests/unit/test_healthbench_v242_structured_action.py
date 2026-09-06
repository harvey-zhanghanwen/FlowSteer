"""Offline per-model action budget and state-aware completion-hook checks.

No replacement of the existing action schema/domain and no model calls.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from src.interactive.react_execution import ToolReactExecutionAdapter, ReactExecutionError
from tests.unit.test_react_execution import request, registry, SequenceGateway, action


def completion(value="fixture artifact"):
    return action("complete", name="complete", arguments={"value": value}, resource_id=None)


def adapter(gateway=None, **kwargs):
    return ToolReactExecutionAdapter(
        gateway=gateway or SequenceGateway([]), tool_registry=registry(),
        max_turns=2, max_tool_calls=1, max_action_tokens=4096, **kwargs,
    )


@pytest.mark.parametrize("enabled,declared,expected", [
    (False, "16384", 4096), (True, "16384", 16384),
    (True, "2048", 2048), (True, None, 4096),
])
def test_configured_budget_is_exact_in_request_and_sampling_receipt(enabled, declared, expected):
    gateway = SequenceGateway([completion()])
    req = request()
    metadata = {"chat_template_enable_thinking": "true"}
    if declared is not None:
        metadata["max_tokens"] = declared
    req = replace(req, model=replace(req.model, metadata=metadata))
    response = asyncio.run(adapter(gateway, respect_model_action_token_budget=enabled).execute(req))
    sent = gateway.requests[0]
    assert sent.model.metadata["max_tokens"] == str(expected)
    assert sent.model.metadata["chat_template_enable_thinking"] == "true"
    assert req.model.metadata == metadata
    assert response.metadata["model_calls"][0]["requested_sampling"]["max_tokens"] == expected
    assert response.metadata["new_react_turns_used"] == 1
    assert response.metadata["tool_calls"] == 0


@pytest.mark.parametrize("declared", ["0", "-1", "4096.0", "True", "none", "", "１２３", "1e4"])
def test_invalid_declared_string_budget_fails_before_generation(declared):
    gateway = SequenceGateway([completion()])
    req = request()
    req = replace(req, model=replace(req.model, metadata={"max_tokens": declared}))
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(adapter(gateway, respect_model_action_token_budget=True).execute(req))
    assert gateway.requests == []


@pytest.mark.parametrize("declared", [True, False, 1.2, None, 0, -2])
def test_typed_invalid_budget_is_not_coerced(declared):
    req = SimpleNamespace(model=SimpleNamespace(metadata={"max_tokens": declared}))
    with pytest.raises(ValueError, match="positive integer"):
        adapter(respect_model_action_token_budget=True)._action_token_budget(req)


def test_typed_positive_integer_budget_is_supported():
    req = SimpleNamespace(model=SimpleNamespace(metadata={"max_tokens": 8192}))
    assert adapter(respect_model_action_token_budget=True)._action_token_budget(req) == 8192


def test_default_generic_multi_action_schema_remains_unchanged():
    instance = adapter()
    assert instance._state_conditioned_response_schema(request(), []) is None
    assert instance._completion_arguments_schema_for_state(request(), []) == instance._completion_arguments_schema(request())


def test_existing_single_action_envelope_calls_state_hook_without_changing_domain():
    class StateHookAdapter(ToolReactExecutionAdapter):
        def _state_conditioned_action_domain(self, request, observations):
            return frozenset(), True

        def _completion_arguments_schema_for_state(self, request, observations):
            return {"type": "object", "required": ["value"],
                "properties": {"value": {"type": "string", "enum": [observations[-1]["public_reference"]]}},
                "additionalProperties": False}

    instance = StateHookAdapter(gateway=SequenceGateway([]), tool_registry=registry(), max_turns=2, max_tool_calls=1)
    observations = [{"observation_status": "success", "public_reference": "observed-source-id"}]
    schema = instance._state_conditioned_response_schema(request(), observations)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(json.loads(completion("observed-source-id")))
    assert not validator.is_valid(json.loads(completion("invented-source-id")))
    assert "oneOf" not in schema and "anyOf" not in schema
    assert set(schema["required"]) == {"arguments", "kind", "name", "resource_id", "skill_id"}
    visible_contract = instance._contract(request(), observations)
    assert '"enum":["observed-source-id"]' in visible_contract


def test_truncated_json_remains_parse_failure_not_automatically_completed():
    truncated = '{"arguments":{"value":"an unfinished artifact'
    gateway = SequenceGateway([truncated, truncated])
    with pytest.raises(ReactExecutionError) as caught:
        asyncio.run(adapter(gateway, respect_model_action_token_budget=True).execute(request()))
    assert len(gateway.requests) == 2
    assert all(item["observation_status"] == "parse_error" for item in caught.value.react_trace)
    assert all(item["action_text"] == truncated for item in caught.value.react_trace)
