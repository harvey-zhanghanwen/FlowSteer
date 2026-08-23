from __future__ import annotations

import asyncio
import unittest

from src.interactive.agent_graph import AgentGraph, AgentGraphValidationError, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentRuntimeError,
    CommunicationCondition,
    ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.environment_execution import (
    build_environment_execution_resources,
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
    @staticmethod
    def _stateful_tool_registry() -> ToolRegistry:
        tool_id = "webshop.environment"
        capability = ToolCapability(
            tool_id=tool_id,
            dataset_scope=("webshop",),
            action_schemas={"step": {"type": "object"}},
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            side_effect="environment_state_transition",
            timeout_seconds=None,
            version="test-v1",
        )
        return ToolRegistry(
            (
                ToolRegistration(
                    tool_id,
                    FakeTool({"step": lambda arguments: dict(arguments)}),
                    capability,
                ),
            )
        )

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
        self.assertFalse(gateway.requests[0].is_output_agent)
        self.assertTrue(gateway.requests[1].is_output_agent)
        routed = gateway.requests[1].upstream[0]
        self.assertEqual("artifact", routed.message_type)
        self.assertEqual(snapshot.revision, routed.graph_revision)
        self.assertEqual("answer", routed.request_or_dependency)
        self.assertEqual(routed.content, routed.artifact)
        self.assertEqual(snapshot.to_dict(), graph.snapshot().to_dict())

    async def test_environment_routes_exact_answer_contract_only_to_output_request(
        self,
    ) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [AgentNode("a", "m1", "collect"), AgentNode("b", "m2", "answer")],
            [AgentRelation("a", "b", True, False)],
            output_agent_id="b",
        )
        env = AgentWorkflowEnv(
            catalog,
            gateway,
            problem="question",
            graph=graph,
            require_exact_answer_tag=True,
        )

        await env.execute(run_id="exact-answer-wire")

        self.assertEqual(["a", "b"], [item.agent.id for item in gateway.requests])
        self.assertFalse(gateway.requests[0].require_exact_answer_tag)
        self.assertTrue(gateway.requests[1].require_exact_answer_tag)
        self.assertFalse(gateway.requests[0].is_output_agent)
        self.assertTrue(gateway.requests[1].is_output_agent)

    async def test_runtime_routes_target_keyed_public_failure_continuation(
        self,
    ) -> None:
        catalog = registry()

        class ContinuationAdapter:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []

            async def execute(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse("repaired")

        adapter = ContinuationAdapter()
        runtime = AgentRuntime(
            catalog,
            RecordingGateway(),
            execution_adapters={"react": adapter},
        )
        graph = AgentGraph(
            [
                AgentNode(
                    "reasoner",
                    "m1",
                    "repair the semantic artifact",
                    execution_mode="react",
                )
            ],
            output_agent_id="reasoner",
        )
        await runtime.execute(
            graph,
            "question",
            prior_failure_metadata={
                "reasoner": {
                    "execution_phase": "single",
                    "react_trace": [
                        {
                            "turn": 1,
                            "observation_status": "schema_invalid",
                            "public_error_code": "candidate binding mismatch",
                        }
                    ],
                    "tool_receipts": [
                        {"tool_id": "qa-retrieval", "success": True}
                    ],
                    "continuation_source_agent_id": "failed_reasoner",
                }
            },
        )

        self.assertEqual(1, len(adapter.requests))
        self.assertEqual(
            "candidate binding mismatch",
            adapter.requests[0].action_history[0]["public_error_code"],
        )
        self.assertEqual(
            "qa-retrieval",
            adapter.requests[0].prior_tool_receipts[0]["tool_id"],
        )
        self.assertEqual(
            "failed_reasoner",
            adapter.requests[0].continuation_source_agent_id,
        )

        await runtime.execute(
            graph,
            "question",
            prior_failure_metadata={
                "reasoner": {
                    "execution_phase": "revision",
                    "react_trace": [
                        {
                            "turn": 1,
                            "observation_status": "schema_invalid",
                            "public_error_code": "wrong communication phase",
                        }
                    ],
                    "tool_receipts": [
                        {"tool_id": "qa-retrieval", "success": True}
                    ],
                    "continuation_source_agent_id": "failed_reasoner",
                }
            },
        )
        self.assertEqual(2, len(adapter.requests))
        self.assertEqual((), adapter.requests[1].action_history)
        self.assertEqual((), adapter.requests[1].prior_tool_receipts)
        self.assertIsNone(adapter.requests[1].continuation_source_agent_id)

    async def test_semantic_protocol_is_propagated_to_every_agent_request(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        runtime = AgentRuntime(
            catalog,
            gateway,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )
        graph = AgentGraph(
            [AgentNode("a", "m1", "reason"), AgentNode("b", "m2", "verify")],
            [AgentRelation("a", "b", True, False)],
            output_agent_id="b",
        )

        await runtime.execute(graph, "question", run_id="semantic-protocol")

        self.assertTrue(gateway.requests)
        self.assertEqual(
            {"hotpotqa_verified_answer_slot_v1"},
            {request.semantic_protocol for request in gateway.requests},
        )

    async def test_hotpot_defers_verifier_and_formatter_until_inputs_are_routed(
        self,
    ) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        runtime = AgentRuntime(
            catalog,
            gateway,
            semantic_protocol="hotpotqa_verified_answer_slot_v1",
        )
        graph = AgentGraph(
            [
                AgentNode("reasoner", "m1", "reason", role_family="reasoner"),
                AgentNode("verifier", "m2", "verify", role_family="verifier"),
                AgentNode("format", "m2", "format", role_family="format"),
            ]
        )

        first = await runtime.execute(
            graph,
            "question",
            require_complete=False,
            format_output_agent=True,
        )

        self.assertEqual(("reasoner",), first.executed_agent_ids)
        self.assertEqual(("format", "verifier"), first.deferred_agent_ids)
        self.assertEqual(["reasoner"], [item.agent.id for item in gateway.requests])

        graph.set_relation("reasoner", "verifier", True, False)
        gateway.requests.clear()
        second = await runtime.execute(
            graph,
            "question",
            require_complete=False,
            prior_outputs=first.outputs,
            dirty_agents={"verifier", "format"},
            format_output_agent=True,
        )
        self.assertEqual(("verifier",), second.executed_agent_ids)
        self.assertEqual(("format",), second.deferred_agent_ids)
        self.assertEqual(["verifier"], [item.agent.id for item in gateway.requests])

        graph.set_relation("verifier", "format", True, False)
        graph.set_output("format")
        gateway.requests.clear()
        third = await runtime.execute(
            graph,
            "question",
            require_complete=False,
            prior_outputs=second.outputs,
            dirty_agents={"format"},
            format_output_agent=True,
        )
        self.assertEqual(("format",), third.executed_agent_ids)
        self.assertEqual((), third.deferred_agent_ids)
        formatter_request = gateway.requests[0]
        self.assertEqual("format", formatter_request.agent.id)
        self.assertTrue(formatter_request.is_format_agent)
        self.assertEqual(
            ["verifier"],
            [item.source_agent_id for item in formatter_request.upstream],
        )

    async def test_masked_condition_keeps_canonical_upstream_and_marks_requests(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [AgentNode("a", "m1", "evidence"), AgentNode("b", "m2", "answer")],
            [AgentRelation("a", "b", True, False)],
            output_agent_id="b",
        )
        result = await AgentRuntime(catalog, gateway).execute(
            graph,
            "question",
            run_id="masked",
            communication_condition=CommunicationCondition.UPSTREAM_MASKED,
        )
        target = next(item for item in gateway.requests if item.agent.id == "b")
        self.assertEqual(CommunicationCondition.UPSTREAM_MASKED, result.communication_condition)
        self.assertEqual(CommunicationCondition.UPSTREAM_MASKED, target.communication_condition)
        self.assertEqual("a[]", target.upstream[0].content)

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

    async def test_partial_execution_reuses_clean_branch_and_recomputes_dirty_closure(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        runtime = AgentRuntime(catalog, gateway)
        graph = AgentGraph(
            [
                AgentNode("a", "m1", "left"),
                AgentNode("b", "m1", "right"),
                AgentNode("c", "m2", "merge"),
            ],
            [
                AgentRelation("a", "c", True, False),
                AgentRelation("b", "c", True, False),
            ],
        )
        initial = await runtime.execute(
            graph,
            "question",
            require_complete=False,
            run_id="partial-initial",
        )
        graph.modify_agent("a", contract="revised left")
        gateway.requests.clear()

        updated = await runtime.execute(
            graph,
            "question",
            require_complete=False,
            prior_outputs=initial.outputs,
            dirty_agents={"a"},
            run_id="partial-updated",
        )

        self.assertIsNone(updated.output_agent_id)
        self.assertIsNone(updated.final_answer)
        self.assertEqual(("a", "c"), updated.executed_agent_ids)
        self.assertEqual(("b",), updated.reused_agent_ids)
        self.assertEqual(["a", "c"], [item.agent.id for item in gateway.requests])
        request_c = next(item for item in gateway.requests if item.agent.id == "c")
        self.assertEqual(
            ["a", "b"],
            [message.source_agent_id for message in request_c.upstream],
        )

    async def test_missing_upstream_invalidates_cached_downstream(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [AgentNode("a", "m1", "evidence"), AgentNode("c", "m2", "answer")],
            [AgentRelation("a", "c", True, False)],
            output_agent_id="c",
        )

        result = await AgentRuntime(catalog, gateway).execute(
            graph,
            "question",
            prior_outputs={"c": "stale-cached-answer"},
            dirty_agents=set(),
        )

        self.assertEqual(("a", "c"), result.executed_agent_ids)
        self.assertEqual((), result.reused_agent_ids)
        self.assertNotEqual("stale-cached-answer", result.final_answer)

    async def test_failure_evicts_dirty_prior_and_preserves_completed_clean_branch(self) -> None:
        catalog = registry()

        class FailingGateway(RecordingGateway):
            async def generate(self, request: AgentRequest) -> str:
                self.requests.append(request)
                if request.agent.id == "a":
                    await asyncio.sleep(0.01)
                    raise RuntimeError("revised branch failed")
                return request.agent.id

        gateway = FailingGateway()
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
                prior_outputs={
                    "a": "stale-a",
                    "b": "cached-b",
                    "c": "stale-c",
                },
                dirty_agents={"a"},
            )

        partial = raised.exception.partial_result
        self.assertIsNotNone(partial)
        assert partial is not None
        self.assertEqual({"b": "cached-b"}, dict(partial.outputs))
        self.assertEqual(("b",), partial.reused_agent_ids)
        self.assertEqual(("c",), raised.exception.blocked_agent_ids)
        self.assertEqual(("a", "c"), raised.exception.pending_agent_ids)
        self.assertEqual("a", raised.exception.failure_records[0].agent_id)
        self.assertNotIn("c", [item.agent.id for item in gateway.requests])

    async def test_react_failure_keeps_public_action_observation_receipts(self) -> None:
        catalog = registry()

        class ReactFailureGateway:
            async def generate(self, request: AgentRequest) -> str:
                raise ReactExecutionError(
                    "bounded execution exhausted",
                    react_trace=(
                        {
                            "turn": 1,
                            "observation_status": "success",
                        },
                    ),
                    tool_receipts=(
                        {"tool_id": "qa.search", "success": True},
                    ),
                    model_calls=(
                        {"turn": 1, "request_id": request.request_id},
                    ),
                )

        graph = AgentGraph([AgentNode("a", "m1", "answer")])
        with self.assertRaises(AgentRuntimeError) as raised:
            await AgentRuntime(catalog, ReactFailureGateway()).execute(
                graph,
                "question",
                require_complete=False,
            )

        failure = raised.exception.failure_records[0]
        self.assertEqual("ReactExecutionError", failure.error_type)
        self.assertEqual(
            "success",
            failure.metadata["react_trace"][0]["observation_status"],
        )
        self.assertEqual("qa.search", failure.metadata["tool_receipts"][0]["tool_id"])

    async def test_format_execution_role_is_explicit_and_terminal(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [
                AgentNode("solver", "m1", "solve", role_family="reasoning"),
                AgentNode("fmt", "m2", "extract", role_family="format"),
            ],
            [AgentRelation("solver", "fmt", True, False)],
            output_agent_id="fmt",
        )

        result = await AgentRuntime(catalog, gateway).execute(
            graph,
            "question",
            format_output_agent=True,
        )
        formatter = next(item for item in gateway.requests if item.agent.id == "fmt")
        solver = next(item for item in gateway.requests if item.agent.id == "solver")

        self.assertTrue(formatter.is_output_agent)
        self.assertTrue(formatter.is_format_agent)
        self.assertFalse(solver.is_format_agent)
        self.assertTrue(solver.is_format_predecessor)
        self.assertFalse(formatter.is_format_predecessor)
        self.assertEqual("fmt[solver:solver[]]", result.final_answer)

    async def test_format_execution_rejects_non_format_output_metadata(self) -> None:
        catalog = registry()
        graph = AgentGraph(
            [AgentNode("solver", "m1", "solve"), AgentNode("out", "m2", "solve again")],
            [AgentRelation("solver", "out", True, False)],
            output_agent_id="out",
        )

        with self.assertRaisesRegex(AgentRuntimeError, "role_family='format'"):
            await AgentRuntime(catalog, RecordingGateway()).execute(
                graph,
                "question",
                format_output_agent=True,
            )

    async def test_format_execution_rejects_react_mode(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        graph = AgentGraph(
            [
                AgentNode("solver", "m1", "solve"),
                AgentNode(
                    "fmt",
                    "m2",
                    "extract",
                    role_family="format",
                    execution_mode="react",
                ),
            ],
            [AgentRelation("solver", "fmt", True, False)],
            output_agent_id="fmt",
        )
        runtime = AgentRuntime(
            catalog,
            gateway,
            execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
        )

        with self.assertRaisesRegex(
            AgentRuntimeError,
            "Format Agent must use reasoning execution without tools",
        ):
            await runtime.execute(graph, "question", format_output_agent=True)

    async def test_stateful_tool_requires_one_graph_agent_owner(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        tools = self._stateful_tool_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "actor_a",
                    "m1",
                    "act",
                    allowed_tools=("webshop.environment",),
                    execution_mode="react",
                ),
                AgentNode(
                    "actor_b",
                    "m2",
                    "act",
                    allowed_tools=("webshop.environment",),
                    execution_mode="react",
                ),
            ],
            [AgentRelation("actor_a", "actor_b", True, False)],
            output_agent_id="actor_b",
        )
        runtime = AgentRuntime(
            catalog,
            gateway,
            execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
            tool_registry=tools,
            dataset_id="webshop",
        )

        with self.assertRaisesRegex(AgentRuntimeError, "one graph Agent owner"):
            await runtime.execute(graph, "question")
        self.assertEqual([], gateway.requests)

    async def test_stateful_tool_owner_cannot_join_reciprocal_block(self) -> None:
        catalog = registry()
        gateway = RecordingGateway()
        tools = self._stateful_tool_registry()
        graph = AgentGraph(
            [
                AgentNode(
                    "actor",
                    "m1",
                    "act",
                    allowed_tools=("webshop.environment",),
                    execution_mode="react",
                ),
                AgentNode("critic", "m2", "critique"),
            ],
            [AgentRelation("actor", "critic", True, True)],
            output_agent_id="critic",
        )
        runtime = AgentRuntime(
            catalog,
            gateway,
            execution_adapters={"react": ReasoningExecutionAdapter(gateway)},
            tool_registry=tools,
            dataset_id="webshop",
        )

        with self.assertRaisesRegex(AgentRuntimeError, "reciprocal Agent block"):
            await runtime.execute(graph, "question")
        self.assertEqual([], gateway.requests)

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

    async def test_fail_fast_cancellation_preserves_environment_public_prefix(self) -> None:
        catalog = registry()

        class Session:
            environment_id = "fake:alfworld"
            task_family = "alfworld"
            available_actions = ("look",)

            def reset(self) -> str:
                return "room zero"

            def step(self, action: str):  # type: ignore[no-untyped-def]
                self.available_actions = ("wait",)
                return "room one", 0.0, False, {"won": False}

        class Gateway:
            def __init__(self) -> None:
                self.actor_calls = 0
                self.actor_second_call = asyncio.Event()
                self.actor_cancelled = asyncio.Event()
                self.called: list[str] = []

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.called.append(request.agent.id)
                if request.agent.id == "failure":
                    await self.actor_second_call.wait()
                    raise RuntimeError("sibling failed")
                if request.agent.id == "actor":
                    self.actor_calls += 1
                    if self.actor_calls == 1:
                        return AgentResponse(
                            "look",
                            {"provider_request_id": "actor-turn-1"},
                        )
                    self.actor_second_call.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.actor_cancelled.set()
                        raise
                return AgentResponse(request.agent.id)

        gateway = Gateway()
        environment = build_environment_execution_resources(
            gateway=gateway,
            session_factory=lambda _request: Session(),
            task_family="alfworld",
            max_turns=3,
        )
        runtime = AgentRuntime(
            catalog,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="alfworld",
            timeout_seconds=10.0,
        )
        graph = AgentGraph(
            [
                AgentNode(
                    "actor",
                    "m1",
                    "act",
                    allowed_tools=(environment.tool_id,),
                    execution_mode="react",
                    artifact_type="environment_observation",
                ),
                AgentNode("failure", "m1", "fail"),
                AgentNode("output", "m1", "summarize"),
            ],
            [
                AgentRelation("actor", "output", True, False),
                AgentRelation("failure", "output", True, False),
            ],
            output_agent_id="output",
        )

        with self.assertRaises(AgentRuntimeError) as caught:
            await runtime.execute(graph, "complete the task")

        await asyncio.wait_for(gateway.actor_cancelled.wait(), timeout=1.0)
        cancellation = next(
            record
            for record in caught.exception.failure_records
            if record.agent_id == "actor"
        )
        self.assertEqual("CancelledError", cancellation.error_type)
        self.assertEqual(1, cancellation.metadata["environment_revision"])
        self.assertFalse(cancellation.metadata["environment_terminal"])
        self.assertEqual(
            "room zero",
            cancellation.metadata["environment_reset_receipt"]["observation"],
        )
        self.assertEqual(
            ["look"],
            [
                item["action"]
                for item in cancellation.metadata["environment_receipts"]
            ],
        )
        self.assertEqual(1, len(cancellation.metadata["tool_receipts"]))
        self.assertEqual(1, len(cancellation.metadata["model_calls"]))
        self.assertIsNotNone(caught.exception.partial_result)
        self.assertIsNone(caught.exception.partial_result.final_answer)
        self.assertNotIn("actor", caught.exception.partial_result.outputs)
        self.assertNotIn("output", gateway.called)

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
