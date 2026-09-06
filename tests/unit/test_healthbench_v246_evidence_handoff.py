"""Synthetic public-evidence handoff checks; no model, evaluator or network."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
    AgentResponse,
    AgentRuntime,
    CommunicationCondition,
)
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.react_execution import ToolReactExecutionAdapter
from src.interactive.tool_runtime import FakeTool, ToolCapability, ToolRegistration, ToolRegistry


SEARCH = "healthbench-authoritative.search"
PROBLEM = render_model_visible_conversation(({
    "role": "user", "content": "Explain the evidence for the specified population.",
},))
SOURCE_TEXT = "SOURCE_OBSERVATION: only the specified population was observed."


def receipt(document_id="public-source-246"):
    return {
        "tool_id": SEARCH,
        "error_type": None,
        "request": {"action": "search", "arguments": {"query": "specified population"}},
        "result": {"completed": True, "value": {
            "operation": "search", "query": "specified population",
            "evidence": [{
                "document_id": document_id, "source": "Synthetic public source",
                "title": "Synthetic evidence", "date": None, "url": None,
                "excerpt": SOURCE_TEXT,
            }],
        }},
    }


def continuation():
    return {
        "execution_phase": "single", "input_artifact_versions": {},
        "continuation_source_agent_id": "a",
        "react_trace": [{
            "turn": 1, "observation_status": "parse_error",
            "public_error_code": "ValueError", "error_message": "invalid action",
            "action_text": "FAILED_COMPLETION_NOT_EVIDENCE",
        }],
        "tool_receipts": [receipt()],
        "evaluator_private": "PRIVATE_EVALUATOR_SENTINEL",
    }


class Gateway:
    def __init__(self, response="Fresh response based on public observations."):
        self.requests = []
        self.response = response

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse(self.response)


def runtime(gateway, **kwargs):
    registry = ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("m1", "fake"), ModelSpec("m2", "fake")],
    )
    return AgentRuntime(
        registry, gateway,
        artifact_communication_profile=ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
        **kwargs,
    )


def graph(*, react=False):
    return AgentGraph([AgentNode(
        "a", "m2", "Use observed sources to address the original request.",
        execution_mode="react" if react else "reasoning",
        allowed_tools=(SEARCH,) if react else (),
    )], output_agent_id="a")


def rendered(request):
    return json.dumps(build_agent_messages(request), ensure_ascii=False)


class EvidenceHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_react_receipts_reach_same_node_reasoning_input(self):
        gateway = Gateway()
        executor = runtime(gateway)
        result = await executor.execute(
            graph(), PROBLEM, prior_failure_metadata={"a": continuation()},
        )
        request = gateway.requests[0]
        self.assertEqual(1, len(request.prior_tool_receipts))
        self.assertIn(SOURCE_TEXT, rendered(request))
        self.assertNotIn("FAILED_COMPLETION_NOT_EVIDENCE", rendered(request))
        self.assertNotIn("PRIVATE_EVALUATOR_SENTINEL", rendered(request))
        self.assertIn("public-source-246", json.dumps(
            dict(result.output_metadata["a"]), default=dict,
        ))

    async def test_singleton_reexecution_preserves_versioned_source_not_old_answer(self):
        gateway = Gateway()
        executor = runtime(gateway)
        await executor.execute(
            graph(), PROBLEM,
            historical_outputs={"a": "OLD_ANSWER_REQUIRING_REVALIDATION"},
            historical_output_metadata={"a": {
                "artifact_version": "previous-run:2:a:single", "graph_revision": 2,
                "tool_receipts": [receipt()],
                "react_trace": [{"action_text": "FAILED_COMPLETION_NOT_EVIDENCE"}],
                "evaluator_private": "PRIVATE_EVALUATOR_SENTINEL",
            }},
        )
        request = gateway.requests[0]
        history = [item for item in request.upstream if item.message_type == "historical_evidence"]
        self.assertEqual(1, len(history))
        self.assertEqual(("a", "a", "previous-run:2:a:single"), (
            history[0].source_agent_id, history[0].target_agent_id, history[0].artifact_version,
        ))
        self.assertIn(SOURCE_TEXT, rendered(request))
        self.assertNotIn("OLD_ANSWER_REQUIRING_REVALIDATION", rendered(request))
        self.assertNotIn("FAILED_COMPLETION_NOT_EVIDENCE", rendered(request))
        self.assertNotIn("PRIVATE_EVALUATOR_SENTINEL", rendered(request))
        self.assertEqual((), request.action_history)
        self.assertEqual((), request.prior_tool_receipts)

    async def test_wrong_phase_foreign_node_and_new_task_have_no_evidence_handoff(self):
        for metadata in (
            {"a": {**continuation(), "execution_phase": "revision"}},
            {"a": {**continuation(), "continuation_source_agent_id": "other"}},
            {"other": continuation()},
            {},
        ):
            with self.subTest(metadata_keys=list(metadata)):
                gateway = Gateway()
                await runtime(gateway).execute(
                    graph(), PROBLEM, prior_failure_metadata=metadata,
                )
                self.assertNotIn(SOURCE_TEXT, rendered(gateway.requests[0]))
        gateway = Gateway()
        executor = runtime(gateway)
        await executor.execute(graph(), PROBLEM, prior_failure_metadata={"a": continuation()})
        await executor.execute(graph(), "A different public task with no historical state.")
        self.assertNotIn(SOURCE_TEXT, rendered(gateway.requests[-1]))

    async def test_singleton_does_not_promote_unversioned_caller_text(self):
        gateway = Gateway()
        await runtime(gateway).execute(
            graph(), PROBLEM,
            historical_outputs={"a": "Unversioned caller text."},
            historical_output_metadata={"a": {"tool_receipts": [receipt()]}},
        )
        self.assertNotIn(SOURCE_TEXT, rendered(gateway.requests[0]))

    async def test_failed_tool_result_is_not_presented_as_evidence(self):
        failed = {**receipt("failed-source"), "error_type": "SyntheticError"}
        incomplete = {**receipt("incomplete-source"), "result": {"completed": False, "value": {}}}
        state = {**continuation(), "tool_receipts": [failed, incomplete]}
        gateway = Gateway()
        await runtime(gateway).execute(graph(), PROBLEM, prior_failure_metadata={"a": state})
        text = rendered(gateway.requests[0])
        self.assertNotIn("failed-source", text)
        self.assertNotIn("incomplete-source", text)
        self.assertNotIn(SOURCE_TEXT, text)

    async def test_communication_mask_still_masks_retained_sources(self):
        gateway = Gateway()
        await runtime(gateway).execute(graph(), PROBLEM, prior_failure_metadata={"a": continuation()})
        request = replace(gateway.requests[0], communication_condition=CommunicationCondition.UPSTREAM_MASKED)
        self.assertNotIn(SOURCE_TEXT, rendered(request))

    async def test_react_continuation_keeps_spent_tool_budget_without_requery(self):
        tool_calls = []
        def search(arguments):
            tool_calls.append(arguments)
            return {}
        tools = ToolRegistry((ToolRegistration(
            SEARCH, FakeTool({"search": search}),
            ToolCapability(SEARCH, ("healthbench_professional",),
                           {"search": {"type": "object"}}, {"type": "object"},
                           {"type": "object"}, "read_only", 1.0, "synthetic"),
        ),))
        gateway = Gateway(json.dumps({
            "kind": "complete", "name": "complete", "resource_id": None,
            "skill_id": None, "arguments": {"value": "A bounded public response."},
        }))
        adapter = ToolReactExecutionAdapter(
            gateway=gateway, tool_registry=tools, max_turns=1, max_tool_calls=1,
        )
        state = continuation()
        state["react_trace"] = [{
            "turn": 1, "observation": {
                "tool_id": SEARCH, "observation_status": "success", "completed": True,
                "result": receipt()["result"]["value"],
            },
        }]
        result = await runtime(gateway, tool_registry=tools, execution_adapters={"react": adapter}).execute(
            graph(react=True), PROBLEM, prior_failure_metadata={"a": state},
        )
        request = gateway.requests[0]
        self.assertEqual(1, len(request.prior_tool_receipts))
        self.assertEqual(1, len(request.action_history))
        self.assertIn(SOURCE_TEXT, rendered(request))
        self.assertEqual([], tool_calls)
        self.assertEqual(1, len(result.output_metadata["a"]["tool_receipts"]))

    async def test_react_singleton_sees_historical_sources_without_resuming_old_actions(self):
        tool_calls = []
        tools = ToolRegistry((ToolRegistration(
            SEARCH, FakeTool({"search": lambda arguments: tool_calls.append(arguments) or {}}),
            ToolCapability(SEARCH, ("healthbench_professional",),
                           {"search": {"type": "object"}}, {"type": "object"},
                           {"type": "object"}, "read_only", 1.0, "synthetic"),
        ),))
        gateway = Gateway(json.dumps({
            "kind": "complete", "name": "complete", "resource_id": None,
            "skill_id": None, "arguments": {"value": "A revalidated public response."},
        }))
        adapter = ToolReactExecutionAdapter(
            gateway=gateway, tool_registry=tools, max_turns=1, max_tool_calls=1,
        )
        await runtime(gateway, tool_registry=tools, execution_adapters={"react": adapter}).execute(
            graph(react=True), PROBLEM,
            historical_outputs={"a": "OLD_ANSWER_REQUIRING_REVALIDATION"},
            historical_output_metadata={"a": {
                "artifact_version": "old:1:a:single", "tool_receipts": [receipt()],
                "react_trace": [{"turn": 99, "action_text": "FAILED_COMPLETION_NOT_EVIDENCE"}],
            }},
        )
        request = gateway.requests[0]
        self.assertIn(SOURCE_TEXT, rendered(request))
        self.assertNotIn("OLD_ANSWER_REQUIRING_REVALIDATION", rendered(request))
        self.assertNotIn("FAILED_COMPLETION_NOT_EVIDENCE", rendered(request))
        self.assertEqual((), request.action_history)
        self.assertEqual((), request.prior_tool_receipts)
        self.assertEqual([], tool_calls)


if __name__ == "__main__":
    unittest.main()
