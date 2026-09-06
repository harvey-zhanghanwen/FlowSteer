"""Mock-only regression for SkillFlow-style invalid Action observations."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json

import pytest

from src.interactive.agent_runtime import AgentResponse
from src.interactive.react_execution import (
    ReactExecutionError,
    ToolReactExecutionAdapter,
    _parse_structured_action,
)
from tests.unit.test_react_execution import action, registry, request


COMPLETE = action(
    "complete", name="complete", arguments={"value": "retained answer"}, resource_id=None
)
TOOL_ACTION = {
    "kind": "tool",
    "name": "search",
    "arguments": {"query": "UNREPLAYABLE_ACTION_BODY"},
    "resource_id": "wiki.search",
    "skill_id": None,
}


class ReceiptGateway:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, current_request):
        self.requests.append(current_request)
        text, metadata = self.outputs.pop(0)
        return AgentResponse(text, metadata)


def _adapter(gateway, *, max_turns=2):
    return ToolReactExecutionAdapter(
        gateway=gateway,
        tool_registry=registry(),
        max_turns=max_turns,
        max_tool_calls=1,
        max_action_tokens=384,
    )


def _public_observations(contract):
    suffix = contract.split("\nPublic observations: ", 1)[1]
    return json.JSONDecoder().raw_decode(suffix)[0]


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (
            {**TOOL_ACTION, "skill_id": "null"},
            "Only Skill actions may carry a skill ID",
        ),
        (
            {**TOOL_ACTION, "resource_id": None},
            "Executable actions require a resource ID",
        ),
        (
            {"action_envelope": TOOL_ACTION},
            "Structured action has an incompatible field set",
        ),
        (
            {**TOOL_ACTION, "resource_id": {"const": "wiki.search"}},
            "Structured action resource_id must be text or null",
        ),
    ],
    ids=["string-null-skill-id", "missing-resource", "nested-envelope", "schema-as-value"],
)
def test_actual_parser_error_reaches_trace_and_next_agent_input(sample, message):
    raw = json.dumps(sample)
    with pytest.raises((TypeError, ValueError)) as caught:
        _parse_structured_action(raw)
    assert str(caught.value) == message
    gateway = ReceiptGateway([(raw, {"finish_reason": "stop"}), (COMPLETE, {})])
    response = asyncio.run(_adapter(gateway).execute(request()))

    assert response.text == "retained answer"
    assert len(gateway.requests) == 2 and gateway.outputs == []
    assert response.metadata["tool_calls"] == 0
    trace_entry = response.metadata["react_trace"][0]
    assert trace_entry["action_text"] == raw
    assert trace_entry["observation_status"] == "parse_error"
    assert trace_entry["public_error_code"] == type(caught.value).__name__
    assert trace_entry["error_message"] == message
    assert trace_entry["finish_reason"] == "stop"
    visible = _public_observations(gateway.requests[1].agent.contract)[0]
    assert visible["error_message"] == message
    assert visible["public_error_code"] == type(caught.value).__name__
    assert visible["finish_reason"] == "stop"
    assert "error_message" in visible["repair_instruction"]
    assert "action_envelope" not in visible["repair_instruction"]
    assert "finish_reason=length" not in visible["repair_instruction"]
    assert "action_text" not in visible
    assert "UNREPLAYABLE_ACTION_BODY" not in gateway.requests[1].agent.contract
    assert all(item.model.metadata["max_tokens"] == "384" for item in gateway.requests)


@pytest.mark.parametrize(
    ("metadata", "length_reported"),
    [
        ({"finish_reason": "length"}, True),
        ({"finish_reason": {"type": "length", "length": 384}}, True),
        ({"finish_reason": "stop"}, False),
        ({}, False),
    ],
    ids=["provider-length", "native-length", "stop-not-length", "no-finish-receipt"],
)
def test_length_diagnosis_requires_actual_finish_receipt(metadata, length_reported):
    raw = '{"kind":"tool","arguments":{"query":"UNREPLAYABLE_ACTION_BODY"'
    gateway = ReceiptGateway([(raw, metadata), (COMPLETE, {})])
    response = asyncio.run(_adapter(gateway).execute(request()))
    trace_entry = response.metadata["react_trace"][0]
    assert trace_entry["error_message"] == "action text is not one JSON object"
    visible = _public_observations(gateway.requests[1].agent.contract)[0]
    assert ("finish_reason=length" in visible["repair_instruction"]) is length_reported
    if "finish_reason" in metadata:
        assert trace_entry["finish_reason"] == visible["finish_reason"] == metadata["finish_reason"]
    else:
        assert "finish_reason" not in trace_entry and "finish_reason" not in visible
    assert "action_text" not in visible
    assert trace_entry["action_text"] == raw
    assert len(gateway.requests) == 2  # No parser retry generation was added.


def test_parse_diagnosis_survives_canvas_continuation_without_action_replay():
    raw = json.dumps({**TOOL_ACTION, "skill_id": "null"})
    first_gateway = ReceiptGateway([(raw, {"finish_reason": "stop"})])
    with pytest.raises(ReactExecutionError) as caught:
        asyncio.run(_adapter(first_gateway, max_turns=1).execute(request()))
    prefix = caught.value.react_trace
    assert len(first_gateway.requests) == 1 and len(prefix) == 1
    assert prefix[0]["action_text"] == raw

    resumed_gateway = ReceiptGateway([(COMPLETE, {})])
    resumed = replace(
        request(),
        graph_revision=2,
        action_history=prefix,
        continuation_source_agent_id="r",
    )
    response = asyncio.run(_adapter(resumed_gateway, max_turns=1).execute(resumed))
    visible = _public_observations(resumed_gateway.requests[0].agent.contract)[0]
    assert visible["error_message"] == "Only Skill actions may carry a skill ID"
    assert visible["finish_reason"] == "stop"
    assert visible["public_error_code"] == "ValueError"
    assert "action_text" not in visible
    assert "UNREPLAYABLE_ACTION_BODY" not in resumed_gateway.requests[0].agent.contract
    assert response.metadata["continued_action_history_count"] == 1
    assert response.metadata["react_turns_used"] == 2
    assert response.metadata["new_react_turns_used"] == 1
    assert response.metadata["react_trace"][0]["error_message"] == visible["error_message"]
    assert len(resumed_gateway.requests) == 1


def test_parser_error_reaches_director_summary_feedback_and_step_projection():
    from src.interactive.agent_graph import AgentGraph, AgentNode
    from src.interactive.agent_runtime import (
        AgentFailureRecord, AgentRuntimeError, ExecutionPhase,
    )
    from src.interactive.agent_workflow_env import AgentWorkflowEnv
    from tests.unit.test_rollout_collector import (
        POLICY_VERSION, ScriptedSGLangClient, _orchestrator, _registry,
    )

    catalog = _registry()
    graph = AgentGraph([AgentNode("node_1", "cheap-model", "Use the public evidence.")])
    environment = AgentWorkflowEnv(
        catalog, gateway=ReceiptGateway([]), graph=graph, problem="Unmodified task."
    )
    client = ScriptedSGLangClient([], policy_version=POLICY_VERSION)
    orchestrator = _orchestrator(catalog, client, max_rounds=2)
    original_error = "Only Skill actions may carry a skill ID"
    for message in (original_error, original_error + "; validation detail" * 70):
        observation = {
            "observation_status": "parse_error",
            "public_error_code": "ValueError",
            "error_message": message,
            "repair_instruction": "Correct the parser error reported in error_message.",
        }
        metadata = {
            "react_trace": [
                {
                    "turn": 1,
                    "action_text": "UNREPLAYABLE_ACTION_BODY",
                    "hidden_state": "NEVER_PUBLIC_HIDDEN_STATE",
                    **observation,
                },
                {
                    "turn": 2,
                    "action_text": "UNREPLAYABLE_ACTION_BODY",
                    "observation": observation,
                },
            ]
        }
        record = AgentFailureRecord(
            request_id="mock-react-failure",
            agent_id="node_1",
            phase=ExecutionPhase.SINGLE,
            graph_revision=graph.revision,
            error_type="ReactExecutionError",
            message="bounded ReAct turns exhausted",
            metadata=metadata,
        )
        expected_summary = " ".join(message.split())[:400]
        summary = environment._react_public_error_summary(record)
        assert summary["last_public_error"]["error_message"] == expected_summary
        assert summary["observation_status_counts"] == {"parse_error": 2}
        assert summary["public_error_code_counts"] == {"ValueError": 2}
        step_projection = environment._compact_react_action_observations(metadata)
        for step in step_projection:
            projected_message = step["observation"]["error_message"]
            assert projected_message.startswith(original_error)
            assert len(projected_message) <= 320
            if len(message) <= 320:
                assert projected_message == message
        feedback_text = environment._execution_error_feedback(
            AgentRuntimeError("bounded ReAct failure", failure_records=(record,))
        )
        feedback = json.loads(feedback_text.split("=", 1)[1])
        attributed = feedback["failed_agents"][0]
        assert attributed["failure_category"] == "react_turn_exhaustion"
        assert attributed["react_public_error_summary"]["last_public_error"]["error_message"] == expected_summary
        environment._reject(None, feedback_text)
        prompt = orchestrator.build_prompt(environment, 1, ())
        assert expected_summary in prompt
        assert "UNREPLAYABLE_ACTION_BODY" not in prompt
        assert "NEVER_PUBLIC_HIDDEN_STATE" not in prompt
        assert "UNREPLAYABLE_ACTION_BODY" not in json.dumps(step_projection)
        assert "NEVER_PUBLIC_HIDDEN_STATE" not in json.dumps(step_projection)
        assert metadata["react_trace"][0]["error_message"] == message
    assert client.payloads == []
