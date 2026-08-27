from __future__ import annotations

import asyncio
import unittest

from src.interactive.agent_graph import AgentGraph, AgentGraphValidationError, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
    ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.tool_runtime import (
    FakeTool,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
)


def registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("m1", "fake"), ModelSpec("m2", "fake")],
    )


class RecordingGateway:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []
        self.active = 0
        self.max_active = 0

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            upstream = ",".join(
                f"{message.source_agent_id}:{message.content}" for message in request.upstream
            )
            return f"{request.agent.id}[{upstream}]"
        finally:
            self.active -= 1


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_chain_routes_final_outputs_and_models(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        runtime = AgentRuntime(catalog, gateway)
        graph = AgentGraph(
            [AgentNode("a", "m1", "draft"), AgentNode("b", "m2", "answer")],
            [AgentRelation("a", "b", True, False)],
            output_agent_id="b",
        )
        snapshot = graph.snapshot()
        result = await runtime.execute(graph, "question", run_id="chain")

        self.assertEqual("b[a:a[]]", result.final_answer)
        self.assertEqual(["a", "b"], [request.agent.id for request in gateway.requests])
        self.assertEqual(["m1", "m2"], [request.model.model_id for request in gateway.requests])
        self.assertEqual(snapshot.to_dict(), graph.snapshot().to_dict())

    async def test_fanin_has_sorted_inputs(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [
                AgentNode("b", "m1", "right"),
                AgentNode("a", "m1", "left"),
                AgentNode("c", "m2", "merge"),
            ],
            [AgentRelation("b", "c", True, False), AgentRelation("a", "c", True, False)],
            output_agent_id="c",
        )
        result = await AgentRuntime(catalog, gateway).execute(graph, "question", run_id="fanin")
        request_c = next(request for request in gateway.requests if request.agent.id == "c")
        self.assertEqual(["a", "b"], [message.source_agent_id for message in request_c.upstream])
        self.assertEqual("c[a:a[],b:b[]]", result.final_answer)

    async def test_event_driven_successor_starts_while_unrelated_branch_is_slow(self) -> None:
        catalog = registry()

        class EventGateway:
            def __init__(self) -> None:
                self.b_started = asyncio.Event()
                self.release_b = asyncio.Event()
                self.c_started = asyncio.Event()

            async def generate(self, request: AgentRequest) -> str:
                if request.agent.id == "b":
                    self.b_started.set()
                    await self.release_b.wait()
                if request.agent.id == "c":
                    self.c_started.set()
                return request.agent.id

        gateway = EventGateway()
        graph = AgentGraph(
            [AgentNode(name, "m1", name) for name in ("a", "b", "c", "d")],
            [
                AgentRelation("a", "c", True, False),
                AgentRelation("c", "d", True, False),
                AgentRelation("b", "d", True, False),
            ],
            output_agent_id="d",
        )
        task = asyncio.create_task(AgentRuntime(catalog, gateway).execute(graph, "q"))
        await asyncio.wait_for(gateway.b_started.wait(), timeout=1.0)
        await asyncio.wait_for(gateway.c_started.wait(), timeout=1.0)
        self.assertFalse(task.done())
        gateway.release_b.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        self.assertEqual("d", result.final_answer)

    async def test_bidirectional_block_has_two_barriers_and_immutable_drafts(self) -> None:
        catalog = registry()

        class ReciprocalGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.drafts_started = 0
                self.release_drafts = asyncio.Event()

            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.phase is ExecutionPhase.DRAFT:
                    self.drafts_started += 1
                    if self.drafts_started == 2:
                        self.release_drafts.set()
                    await self.release_drafts.wait()
                    return f"draft-{request.agent.id}"
                self.assert_revision_request(request)
                return f"revision-{request.agent.id}"

            @staticmethod
            def assert_revision_request(request: AgentRequest) -> None:
                expected_peer = "b" if request.agent.id == "a" else "a"
                if request.own_draft != f"draft-{request.agent.id}":
                    raise AssertionError("revision did not receive its own immutable draft")
                if request.peer_draft is None or request.peer_draft.content != f"draft-{expected_peer}":
                    raise AssertionError("revision did not receive the peer draft")

        gateway = ReciprocalGateway()
        graph = AgentGraph(
            [AgentNode("a", "m1", "draft"), AgentNode("b", "m2", "verify")],
            [AgentRelation("a", "b", True, True)],
            output_agent_id="b",
        )
        result = await AgentRuntime(catalog, gateway).execute(graph, "q", run_id="pair")
        self.assertEqual("revision-b", result.final_answer)
        self.assertEqual(4, len(gateway.requests))
        phases = [request.phase for request in gateway.requests]
        self.assertEqual(2, phases.count(ExecutionPhase.DRAFT))
        self.assertEqual(2, phases.count(ExecutionPhase.REVISION))

    async def test_bidirectional_members_receive_only_their_external_inputs(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [AgentNode(name, "m1", name) for name in ("x", "y", "a", "b")],
            [
                AgentRelation("x", "a", True, False),
                AgentRelation("y", "b", True, False),
                AgentRelation("a", "b", True, True),
            ],
            output_agent_id="b",
        )
        await AgentRuntime(catalog, gateway).execute(graph, "q")
        drafts = {
            request.agent.id: request
            for request in gateway.requests
            if request.phase is ExecutionPhase.DRAFT
        }
        self.assertEqual(["x"], [message.source_agent_id for message in drafts["a"].upstream])
        self.assertEqual(["y"], [message.source_agent_id for message in drafts["b"].upstream])

    async def test_invalid_graph_fails_before_gateway_call(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph([AgentNode("a", "m1", "answer")])
        with self.assertRaises(AgentGraphValidationError):
            await AgentRuntime(catalog, gateway).execute(graph, "q")
        self.assertEqual([], gateway.requests)

    async def test_gateway_failure_cancels_other_blocks_and_never_runs_downstream(self) -> None:
        catalog = registry()

        class FailingGateway:
            def __init__(self) -> None:
                self.b_started = asyncio.Event()
                self.b_cancelled = asyncio.Event()
                self.called: list[str] = []

            async def generate(self, request: AgentRequest) -> str:
                self.called.append(request.agent.id)
                if request.agent.id == "a":
                    await self.b_started.wait()
                    raise RuntimeError("boom")
                if request.agent.id == "b":
                    self.b_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.b_cancelled.set()
                return request.agent.id

        gateway = FailingGateway()
        graph = AgentGraph(
            [AgentNode(name, "m1", name) for name in ("a", "b", "c")],
            [AgentRelation("a", "c", True, False), AgentRelation("b", "c", True, False)],
            output_agent_id="c",
        )
        with self.assertRaises(AgentRuntimeError):
            await AgentRuntime(catalog, gateway).execute(graph, "q")
        await asyncio.wait_for(gateway.b_cancelled.wait(), timeout=1.0)
        self.assertNotIn("c", gateway.called)

    async def test_react_failure_preserves_receipts_and_completed_sibling(self) -> None:
        catalog = registry()

        class PartialFailureGateway:
            def __init__(self) -> None:
                self.sibling_completed = asyncio.Event()
                self.called: list[str] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.called.append(request.agent.id)
                if request.agent.id == "a":
                    await self.sibling_completed.wait()
                    raise ReactExecutionError(
                        "bounded execution exhausted",
                        metadata={
                            "react_trace": [
                                {
                                    "turn": 1,
                                    "observation_status": "success",
                                }
                            ],
                            "tool_receipts": [
                                {"tool_id": "qa.search", "success": True}
                            ],
                            "model_calls": [
                                {
                                    "turn": 1,
                                    "request_id": request.request_id + ":react:1",
                                }
                            ],
                        },
                    )
                if request.agent.id == "b":
                    self.sibling_completed.set()
                    return AgentResponse(
                        "durable sibling artifact",
                        {"provider_request_id": "sibling-ok"},
                    )
                return AgentResponse("must not run")

        gateway = PartialFailureGateway()
        graph = AgentGraph(
            [AgentNode(name, "m1", name) for name in ("a", "b", "c")],
            [
                AgentRelation("a", "c", True, False),
                AgentRelation("b", "c", True, False),
            ],
            output_agent_id="c",
        )

        with self.assertRaises(AgentRuntimeError) as raised:
            await AgentRuntime(catalog, gateway).execute(
                graph,
                "question",
                require_complete=False,
            )

        failure = raised.exception.failure_records[0]
        self.assertEqual("a", failure.agent_id)
        self.assertEqual("ReactExecutionError", failure.error_type)
        self.assertEqual(
            "success",
            failure.metadata["react_trace"][0]["observation_status"],
        )
        self.assertEqual(
            "qa.search",
            failure.metadata["tool_receipts"][0]["tool_id"],
        )
        self.assertTrue(
            failure.metadata["model_calls"][0]["request_id"].endswith(
                ":react:1"
            )
        )
        partial = raised.exception.partial_result
        self.assertIsNotNone(partial)
        assert partial is not None
        self.assertEqual(
            {"b": "durable sibling artifact"},
            dict(partial.outputs),
        )
        self.assertEqual(("b",), partial.executed_agent_ids)
        self.assertEqual(1, len(partial.calls))
        self.assertEqual(("c",), raised.exception.blocked_agent_ids)
        self.assertEqual(("a", "c"), raised.exception.pending_agent_ids)
        self.assertNotIn("c", gateway.called)

    async def test_registered_execution_profiles_expose_task_scoped_tools(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        tools = ToolRegistry(
            (
                ToolRegistration(
                    "qa.search",
                    FakeTool({"search": lambda arguments: arguments}),
                    ToolCapability(
                        tool_id="qa.search",
                        dataset_scope=("hotpotqa",),
                        action_schemas={
                            "search": {
                                "type": "object",
                                "additionalProperties": True,
                            }
                        },
                        input_schema={"type": "object"},
                        output_schema={"type": "object"},
                        side_effect="none",
                        timeout_seconds=1.0,
                        version="qa-search-v1",
                    ),
                ),
            )
        )
        runtime = AgentRuntime(
            catalog,
            gateway,
            execution_adapters={
                "react": ReasoningExecutionAdapter(gateway),
            },
            tool_registry=tools,
            dataset_id="hotpotqa",
        )

        self.assertEqual(
            (
                ("reasoning", ()),
                ("react", ()),
                ("react", ("qa.search",)),
            ),
            runtime.registered_execution_profiles(),
        )

    async def test_global_concurrency_limit(self) -> None:
        catalog = registry()

        class LimitedGateway:
            def __init__(self) -> None:
                self.active = 0
                self.maximum = 0

            async def generate(self, request: AgentRequest) -> str:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                await asyncio.sleep(0)
                self.active -= 1
                return request.agent.id

        gateway = LimitedGateway()
        graph = AgentGraph(
            [AgentNode(name, "m1", name) for name in ("a", "b", "c")],
            [AgentRelation("a", "c", True, False), AgentRelation("b", "c", True, False)],
            output_agent_id="c",
        )
        await AgentRuntime(catalog, gateway, max_concurrency=1).execute(graph, "q")
        self.assertEqual(1, gateway.maximum)


if __name__ == "__main__":
    unittest.main()
