from __future__ import annotations

import unittest

from src.interactive.agent_action_parser import AgentActionParser
from src.interactive.agent_graph import AgentExecutionMode, AgentGraph, AgentNode
from src.interactive.agent_runtime import (
    AgentResponse,
    AgentRuntime,
    CommunicationEnvelope,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
)


def capability(tool_id: str = "wiki.search") -> ToolCapability:
    return ToolCapability(
        tool_id=tool_id,
        dataset_scope=("triviaqa",),
        action_schemas={"search": {"type": "object"}},
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        side_effect="none",
        timeout_seconds=1.0,
        version="wiki-snapshot-2026-08",
        availability=True,
    )


class AgentExecutionContractTests(unittest.TestCase):
    def test_parser_and_snapshot_preserve_execution_contract(self) -> None:
        action = AgentActionParser().parse(
            '{"action":"add_agent","agent_id":"retriever","model_id":"m",'
            '"contract":"retrieve public evidence","allowed_tools":["wiki.search"],'
            '"execution_mode":"react","artifact_type":"retrieved_document",'
            '"completion_condition":"return cited evidence"}'
        )
        node = AgentNode(
            action.agent_id or "",
            action.model_id or "",
            action.contract,
            allowed_tools=action.allowed_tools or (),
            execution_mode=action.execution_mode or "reasoning",
            artifact_type=action.artifact_type or "text",
            completion_condition=action.completion_condition,
        )
        graph = AgentGraph([node], output_agent_id="retriever")
        restored = AgentGraph.from_snapshot(graph.snapshot())

        restored_node = restored.get_node("retriever")
        self.assertEqual(("wiki.search",), restored_node.allowed_tools)
        self.assertIs(AgentExecutionMode.REACT, restored_node.execution_mode)
        self.assertEqual("retrieved_document", restored_node.artifact_type)
        self.assertEqual("return cited evidence", restored_node.completion_condition)
        self.assertEqual(graph.to_dict(), restored.to_dict())

    def test_legacy_node_defaults_to_reasoning_without_tools(self) -> None:
        node = AgentNode("reasoner", "m", "reason over supplied context")
        self.assertEqual((), node.allowed_tools)
        self.assertIs(AgentExecutionMode.REASONING, node.execution_mode)
        self.assertEqual("text", node.artifact_type)
        self.assertIsNone(node.completion_condition)

    def test_strict_parser_rejects_unknown_mode_and_duplicate_tools(self) -> None:
        parser = AgentActionParser()
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            parser.parse(
                '{"action":"add_agent","agent_id":"a","model_id":"m",'
                '"contract":"x","execution_mode":"planner"}'
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            parser.parse(
                '{"action":"add_agent","agent_id":"a","model_id":"m",'
                '"contract":"x","allowed_tools":["t","t"]}'
            )


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    def test_capability_serializes_and_validates_fixed_action_schema(self) -> None:
        fixed = capability()
        self.assertEqual(("search",), fixed.action_names)
        self.assertEqual(
            {"search": {"type": "object"}},
            fixed.to_value()["action_schemas"],
        )
        self.assertIsNone(fixed.argument_validation_error("search", {}))
        self.assertIsNotNone(fixed.argument_validation_error("read", {}))

        with self.assertRaisesRegex(ValueError, "invalid"):
            ToolCapability(
                tool_id="broken",
                dataset_scope=("triviaqa",),
                action_schemas={"search": {"type": "not-a-json-schema-type"}},
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                side_effect="none",
                timeout_seconds=1.0,
                version="broken-v1",
            )

    async def test_skillflow_registry_port_emits_measured_receipt(self) -> None:
        backend = FakeTool({"search": lambda arguments: {"query": arguments["query"]}})
        registry = ToolRegistry(
            (
                ToolRegistration(
                    "wiki.search",
                    backend,
                    capability("wiki.search"),
                ),
            )
        )
        request = ToolRequest("search", {"query": "Ada Lovelace"})
        result, receipt = await registry.ainvoke_with_receipt(
            "wiki.search", request
        )

        self.assertIsNotNone(result)
        self.assertEqual({"query": "Ada Lovelace"}, result.value)
        self.assertEqual("wiki-snapshot-2026-08", receipt.tool_version)
        self.assertGreaterEqual(receipt.latency_ms, 0.0)
        self.assertIsNone(receipt.error_type)

        envelope = CommunicationEnvelope(
            "retriever",
            "reasoner",
            "Ada Lovelace evidence",
            artifact_type="retrieved_document",
            graph_revision=2,
            request_or_dependency="resolve the named entity",
            tool_receipts=(receipt.to_value(),),
        )
        serialized = envelope.to_dict()
        self.assertEqual("retrieved_document", serialized["artifact_type"])
        self.assertEqual(
            "resolve the named entity", serialized["dependency"]
        )
        self.assertEqual("wiki.search", serialized["tool_receipts"][0]["tool_id"])

    async def test_runtime_dispatches_react_node_and_routes_receipt(self) -> None:
        tool = FakeTool({"search": lambda arguments: {"query": arguments["query"]}})
        registry = ToolRegistry(
            (ToolRegistration("wiki.search", tool, capability("wiki.search")),)
        )
        catalog = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )

        class UnusedGateway:
            async def generate(self, request):  # type: ignore[no-untyped-def]
                raise AssertionError("react node must not use reasoning gateway")

        class ReactAdapter:
            async def execute(self, request):  # type: ignore[no-untyped-def]
                result, receipt = await registry.ainvoke_with_receipt(
                    "wiki.search",
                    ToolRequest("search", {"query": "Ada Lovelace"}),
                )
                assert result is not None
                return AgentResponse(
                    "retrieved Ada Lovelace evidence",
                    {
                        "tool_receipts": [receipt.to_value()],
                        "environment_revision": 1,
                    },
                )

        graph = AgentGraph(
            [
                AgentNode(
                    "retriever",
                    "m",
                    "retrieve evidence",
                    allowed_tools=("wiki.search",),
                    execution_mode="react",
                    artifact_type="retrieved_document",
                )
            ],
            output_agent_id="retriever",
        )
        runtime = AgentRuntime(
            catalog,
            UnusedGateway(),
            execution_adapters={"react": ReactAdapter()},
            tool_registry=registry,
            dataset_id="triviaqa",
        )
        execution = await runtime.execute(graph, "Who was Ada Lovelace?")

        self.assertEqual("retrieved Ada Lovelace evidence", execution.final_answer)
        self.assertEqual(
            "wiki.search",
            execution.output_metadata["retriever"]["tool_receipts"][0]["tool_id"],
        )

    async def test_runtime_rejects_unregistered_react_adapter(self) -> None:
        catalog = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )

        class Gateway:
            async def generate(self, request):  # type: ignore[no-untyped-def]
                return "unused"

        graph = AgentGraph(
            [AgentNode("a", "m", "interact", execution_mode="react")],
            output_agent_id="a",
        )
        with self.assertRaisesRegex(RuntimeError, "unregistered execution adapter"):
            await AgentRuntime(catalog, Gateway()).execute(graph, "task")


if __name__ == "__main__":
    unittest.main()
