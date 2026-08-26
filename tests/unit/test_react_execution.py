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
        )

        response = await adapter.execute(request())

        self.assertEqual("Ada Lovelace", response.text)
        self.assertEqual(3, response.metadata["react_turns_used"])
        self.assertEqual(1, response.metadata["tool_calls"])
        self.assertEqual(
            "parse_error", response.metadata["react_trace"][0]["observation_status"]
        )
        self.assertEqual(
            "wiki.search", response.metadata["tool_receipts"][0]["tool_id"]
        )
        self.assertIn("passage_ids", gateway.requests[2].agent.contract)

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
        with self.assertRaisesRegex(ReactExecutionError, "exhausted"):
            await adapter.execute(request())


if __name__ == "__main__":
    unittest.main()
