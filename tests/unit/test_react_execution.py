from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import (
    ReactExecutionError,
    ToolReactExecutionAdapter,
)
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)


def action(
    kind: str,
    *,
    name: str,
    arguments: object,
    resource_id: str | None,
) -> str:
    return json.dumps(
        {
            "kind": kind,
            "name": name,
            "arguments": arguments,
            "resource_id": resource_id,
            "skill_id": None,
        }
    )


def registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolRegistration(
                "wiki.search",
                FakeTool(
                    {
                        "search": lambda arguments: {
                            "passage_ids": ["p1"],
                            "query": arguments["query"],
                        }
                    }
                ),
                ToolCapability(
                    tool_id="wiki.search",
                    dataset_scope=("triviaqa",),
                    action_schemas={
                        "search": {
                            "type": "object",
                            "required": ["query"],
                            "properties": {"query": {"type": "string"}},
                        }
                    },
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    side_effect="none",
                    timeout_seconds=1.0,
                    version="wiki-test-v1",
                ),
            ),
        )
    )


def request() -> AgentRequest:
    return AgentRequest(
        request_id="run:1:r:single",
        run_id="run",
        graph_revision=1,
        problem="Who wrote the first published algorithm?",
        agent=AgentNode(
            "r",
            "m",
            "retrieve evidence and answer",
            allowed_tools=("wiki.search",),
            execution_mode="react",
            artifact_type="evidence",
            completion_condition="return an answer supported by the observation",
        ),
        model=ModelSpec("m", "fake"),
        provider=ProviderSpec("fake", kind="test"),
        phase=ExecutionPhase.SINGLE,
    )


class SequenceGateway:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(self.outputs.pop(0), {"provider_request_id": len(self.requests)})


class ToolReactExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_action_adapter_rejects_dynamic_native_capability(self) -> None:
        gateway = SequenceGateway([])
        dynamic_registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    FakeTool({}),
                    ToolCapability(
                        tool_id="wiki.search",
                        dataset_scope=("triviaqa",),
                        action_schemas={},
                        input_schema={"type": "object"},
                        output_schema={"type": "object"},
                        side_effect="environment_state_transition",
                        timeout_seconds=1.0,
                        version="dynamic-test-v1",
                    ),
                ),
            )
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=dynamic_registry,
            max_turns=1,
            max_tool_calls=1,
        )

        with self.assertRaisesRegex(ReactExecutionError, "fixed action schemas"):
            await adapter.execute(request())
        self.assertEqual([], gateway.requests)

    async def test_parse_error_tool_observation_and_explicit_completion(self) -> None:
        gateway = SequenceGateway(
            [
                "not JSON",
                action(
                    "tool",
                    name="search",
                    arguments={"query": "first published algorithm author"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=4,
            max_tool_calls=2,
            max_action_tokens=256,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(3, response.metadata["react_turns_used"])
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(
            "parse_error", response.metadata["react_trace"][0]["observation_status"]
        )
        self.assertEqual(
            "not JSON", response.metadata["react_trace"][0]["action_text"]
        )
        self.assertEqual(
            "wiki.search", response.metadata["tool_receipts"][0]["tool_id"]
        )
        self.assertIn('"name":"search"', gateway.requests[0].agent.contract)
        self.assertIn('"required":["query"]', gateway.requests[0].agent.contract)
        self.assertNotIn('"name":"tool action"', gateway.requests[0].agent.contract)
        self.assertIn("passage_ids", gateway.requests[2].agent.contract)
        self.assertIn("executed_action", gateway.requests[2].agent.contract)
        self.assertEqual("256", gateway.requests[0].model.metadata["max_tokens"])

    async def test_argument_schema_is_enforced_before_tool_dispatch(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={},
                    resource_id="wiki.search",
                ),
                action(
                    "tool",
                    name="search",
                    arguments={"query": "Ada Lovelace"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )

        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=3,
            max_tool_calls=1,
        ).execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        invalid = response.metadata["react_trace"][0]
        self.assertEqual(
            "tool_arguments_schema_invalid",
            invalid["public_error_code"],
        )
        self.assertIn("query", invalid["argument_validation"]["message"])

    async def test_unregistered_action_name_is_rejected_before_backend(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="tool action",
                    arguments={"query": "wrong action name"},
                    resource_id="wiki.search",
                ),
                action(
                    "tool",
                    name="search",
                    arguments={"query": "Ada Lovelace"},
                    resource_id="wiki.search",
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=3,
            max_tool_calls=2,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(1, len(response.metadata["tool_receipts"]))
        self.assertEqual(
            "tool_action_not_registered",
            response.metadata["react_trace"][0]["public_error_code"],
        )
        self.assertEqual(
            ["search"],
            response.metadata["react_trace"][0]["allowed_action_names"],
        )

    async def test_skill_action_remains_unavailable_to_executor(self) -> None:
        gateway = SequenceGateway(
            [
                json.dumps(
                    {
                        "kind": "skill",
                        "name": "invoke",
                        "arguments": {},
                        "resource_id": "wiki.search",
                        "skill_id": "candidate.skill",
                    }
                ),
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "Ada Lovelace"},
                    resource_id=None,
                ),
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=1,
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(0, response.metadata["tool_calls"])
        self.assertEqual(
            "skill_action_not_admitted",
            response.metadata["react_trace"][0]["public_error_code"],
        )

    async def test_turn_budget_requires_explicit_completion(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={"query": "x"},
                    resource_id="wiki.search",
                )
            ]
        )
        adapter = ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
        )
        with self.assertRaisesRegex(ReactExecutionError, "exhausted") as raised:
            await adapter.execute(request())
        self.assertEqual(1, len(raised.exception.react_trace))
        self.assertEqual(1, len(raised.exception.tool_receipts))
        self.assertEqual(1, len(raised.exception.model_calls))
        self.assertEqual(
            "wiki.search", raised.exception.tool_receipts[0]["tool_id"]
        )

    async def test_identical_failed_tool_request_is_not_dispatched_twice(self) -> None:
        class FailingTool:
            def __init__(self) -> None:
                self.calls = 0

            async def invoke(self, request):
                del request
                self.calls += 1
                raise TimeoutError("public timeout")

        backend = FailingTool()
        tool_registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    backend,
                    registry().require_capability("wiki.search"),
                ),
            )
        )
        repeated = action(
            "tool",
            name="search",
            arguments={"query": "same failed query"},
            resource_id="wiki.search",
        )
        gateway = SequenceGateway(
            [
                repeated,
                repeated,
                action(
                    "complete",
                    name="complete",
                    arguments={"value": "insufficient evidence"},
                    resource_id=None,
                ),
            ]
        )
        response = await ToolReactExecutionAdapter(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=3,
            max_tool_calls=3,
        ).execute(request())

        self.assertEqual(1, backend.calls)
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(
            "duplicate_tool_request",
            response.metadata["react_trace"][1]["public_error_code"],
        )
        self.assertIn("executed_action", gateway.requests[2].agent.contract)


if __name__ == "__main__":
    unittest.main()
