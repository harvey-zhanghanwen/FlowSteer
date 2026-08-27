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
from src.interactive.scientific_sampling import (
    GenerationPhase,
    ScientificSamplingCoordinate,
    derive_generation_seed,
    scientific_sampling_schedule_hash,
    stable_hash,
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
                            "additionalProperties": False,
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
    async def test_scientific_sampling_uses_distinct_deterministic_turn_seeds(self) -> None:
        base_seed = 20260815
        coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(
                base_seed=base_seed
            ),
            schedule_purpose="hotpotqa_qa_memory_v7",
            ordered_sequence_hash=stable_hash(["task-1"]),
            sequence_position=0,
            task_id="task-1",
            optimizer_step_or_anchor_ordinal=0,
        )
        gateway = SequenceGateway(
            [
                "not JSON",
                "still not JSON",
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
            sampling_base_seed=base_seed,
            sampling_coordinate=coordinate,
        ).execute(request())

        seeds = [
            int(item.model.metadata["generation_seed"])
            for item in gateway.requests
        ]
        expected = [
            derive_generation_seed(
                base_seed=base_seed,
                coordinate=coordinate,
                step_index=turn,
                phase=GenerationPhase.ACTION,
            )
            for turn in (1, 2, 3)
        ]
        self.assertEqual(expected, seeds)
        self.assertEqual(3, len(set(seeds)))
        self.assertEqual(
            expected,
            [
                item["scientific_sampling"]["generation_seed"]
                for item in response.metadata["model_calls"]
            ],
        )

    async def test_narrowed_action_without_registered_schema_fails_before_model(self) -> None:
        class MissingSchemaAdapter(ToolReactExecutionAdapter):
            def _state_conditioned_action_domain(self, request, observations):
                del request, observations
                return frozenset({("wiki.search", "read")}), False

        gateway = SequenceGateway([])
        adapter = MissingSchemaAdapter(
            gateway=gateway,
            tool_registry=registry(),
            max_turns=1,
            max_tool_calls=1,
        )

        with self.assertRaisesRegex(ReactExecutionError, "no registered argument schema"):
            await adapter.execute(request())
        self.assertEqual([], gateway.requests)

    async def test_tool_arguments_schema_is_enforced_before_dispatch(self) -> None:
        gateway = SequenceGateway(
            [
                action(
                    "tool",
                    name="search",
                    arguments={},
                    resource_id="wiki.search",
                ),
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
            tool_registry=registry(),
            max_turns=2,
            max_tool_calls=1,
        ).execute(request())

        self.assertEqual(0, response.metadata["tool_calls"])
        invalid = response.metadata["react_trace"][0]
        self.assertEqual("tool_arguments_schema_invalid", invalid["public_error_code"])
        self.assertIn("query", invalid["argument_validation"]["message"])

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
