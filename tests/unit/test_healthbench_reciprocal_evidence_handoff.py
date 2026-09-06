"""Synthetic source handoff checks; no model, network, benchmark or grader calls."""

from __future__ import annotations

import json
import unittest

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
    AgentRequest, AgentResponse, AgentRuntime, AgentRuntimeError, ExecutionPhase,
    ReasoningExecutionAdapter,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.tool_runtime import FakeTool, ToolCapability, ToolRegistration, ToolRegistry


SEARCH = "healthbench-authoritative.search"
PROBLEM = render_model_visible_conversation(({
    "role": "user", "content": "Explain the evidence for the specified clinical group.",
},))


def source_receipt(document_id="synthetic-source-101"):
    return {
        "tool_id": SEARCH, "error_type": None,
        "request": {"action": "search", "arguments": {"query": "specified clinical group"}},
        "result": {"completed": True, "value": {
            "operation": "search", "query": "specified clinical group",
            "evidence": [{
                "source": "Synthetic source", "source_type": "pubmed",
                "document_id": document_id, "title": "Synthetic clinical evidence",
                "url": "https://example.invalid/" + document_id,
                "date": "2030", "excerpt": "The source reports a finding for the specified group.",
            }],
        }},
    }


def catalog():
    return ModelRegistry([ProviderSpec("fake", kind="test")], [ModelSpec("m", "fake")])


def pair(*, reciprocal=True, react=False):
    kwargs = {"execution_mode": "react", "allowed_tools": (SEARCH,)} if react else {}
    return AgentGraph(
        [AgentNode("a", "m", "Analyze sources.", **kwargs), AgentNode("b", "m", "Check the sources.")],
        [AgentRelation("a", "b", True, reciprocal)], output_agent_id="b",
    )


class Gateway:
    def __init__(self, *, fail_draft=False):
        self.requests = []
        self.fail_draft = fail_draft

    async def generate(self, request: AgentRequest):
        self.requests.append(request)
        if self.fail_draft and request.phase is ExecutionPhase.DRAFT:
            raise RuntimeError("synthetic recomputation failure")
        if request.agent.id == "a" and request.phase is ExecutionPhase.SINGLE:
            return AgentResponse("Earlier producer interpretation requiring revalidation.", {
                "tool_receipts": [source_receipt()],
                "react_trace": [{"turn": 3, "observation_status": "success", "action_text": "old completion"}],
            })
        return AgentResponse(f"fresh {request.agent.id} {request.phase.value}")


def runtime(gateway, **kwargs):
    return AgentRuntime(
        catalog(), gateway,
        artifact_communication_profile=ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
        **kwargs,
    )


def rendered(request):
    return json.dumps(build_agent_messages(request), ensure_ascii=False)


class ReciprocalEvidenceHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_canvas_single_to_reciprocal_preserves_sources_at_two_phase_barrier(self):
        gateway = Gateway()
        executor = runtime(gateway)
        env = AgentWorkflowEnv(executor.model_registry, runtime=executor,
                               problem=PROBLEM, graph=pair(reciprocal=False), execute_on_edit=True)
        first = await env.step(AgentAction(AgentActionType.MODIFY_AGENT,
                                         agent_id="a", contract="Inspect the specified sources."))
        self.assertTrue(first.accepted, first.feedback)
        old_version = first.execution.output_metadata["a"]["artifact_version"]
        gateway.requests.clear()
        second = await env.step(AgentAction(AgentActionType.SET_RELATION, source_id="a", target_id="b",
                                          source_to_target=True, target_to_source=True))
        self.assertTrue(second.accepted, second.feedback)
        self.assertEqual(4, len(gateway.requests))
        requests = {(item.agent.id, item.phase): item for item in gateway.requests}
        own = requests["a", ExecutionPhase.DRAFT]
        peer_draft = requests["b", ExecutionPhase.DRAFT]
        peer_revision = requests["b", ExecutionPhase.REVISION]
        history = next(item for item in own.upstream if item.message_type == "historical_evidence")
        self.assertEqual("a", history.source_agent_id)
        self.assertEqual("a", history.target_agent_id)
        self.assertEqual(old_version, history.artifact_version)
        self.assertIn("revalidation", history.content)
        self.assertIn("synthetic-source-101", rendered(own))
        # b already received a's source during the earlier legal SINGLE edge;
        # retaining b's own historical inputs is not reading a's new draft.
        self.assertIn("synthetic-source-101", rendered(peer_draft))
        self.assertNotIn("fresh a draft", rendered(peer_draft))
        self.assertIn("fresh a draft", rendered(peer_revision))
        self.assertIn("synthetic-source-101", rendered(peer_revision))
        self.assertEqual("fresh a draft", peer_revision.peer_draft.content)
        self.assertNotEqual(old_version, peer_revision.peer_draft.artifact_version)
        for request in gateway.requests:
            self.assertEqual((), request.action_history)
            self.assertEqual((), request.prior_tool_receipts)
            self.assertIsNone(request.continuation_source_agent_id)
        self.assertEqual("fresh b revision", second.execution.final_answer)
        self.assertNotEqual(old_version, second.execution.output_metadata["a"]["artifact_version"])
        self.assertEqual({}, second.execution.output_metadata["a"]["input_artifact_provenance"][0]["input_artifact_provenance"][0].get("react_trace", {}))

    async def test_runtime_invalidates_current_answers_but_keeps_dirty_source_material(self):
        gateway = Gateway()
        executor = runtime(gateway)
        first = await executor.execute(pair(reciprocal=False), PROBLEM, run_id="before")
        gateway.fail_draft = True
        with self.assertRaises(AgentRuntimeError) as failure:
            await executor.execute(pair(), PROBLEM, run_id="after",
                                   prior_outputs=first.outputs, prior_output_metadata=first.output_metadata,
                                   dirty_agents={"a", "b"})
        self.assertEqual({}, dict(failure.exception.partial_result.outputs))
        draft = next(item for item in gateway.requests if item.agent.id == "a" and item.phase is ExecutionPhase.DRAFT)
        self.assertIn("synthetic-source-101", rendered(draft))

    async def test_history_does_not_broadcast_disconnected_or_deleted_producers(self):
        gateway = Gateway()
        graph = AgentGraph(
            [AgentNode(name, "m", "Inspect sources.") for name in ("a", "b", "c", "out")],
            [AgentRelation("a", "b", True, True), AgentRelation("b", "out", True, False),
             AgentRelation("c", "out", True, False)], output_agent_id="out",
        )
        nested = [{"source_agent_id": source, "target_agent_id": "a", "artifact": "old unrelated evidence",
                   "tool_receipts": [source_receipt(source + "-private-source")]} for source in ("c", "deleted")]
        await runtime(gateway).execute(
            graph, PROBLEM, historical_outputs={"a": "old a", "c": "old c", "deleted": "old deleted"},
            historical_output_metadata={
                "a": {"artifact_version": "old:a", "tool_receipts": [source_receipt()],
                      "input_artifact_provenance": nested},
                "c": {"tool_receipts": [source_receipt("c-private-source")]},
                "deleted": {"tool_receipts": [source_receipt("deleted-private-source")]},
            },
        )
        for request in gateway.requests:
            self.assertNotIn("c-private-source", rendered(request))
            self.assertNotIn("deleted-private-source", rendered(request))
        own_draft = next(item for item in gateway.requests
                         if item.agent.id == "a" and item.phase is ExecutionPhase.DRAFT)
        self.assertIn("synthetic-source-101", rendered(own_draft))

    async def test_historical_reference_does_not_reset_same_phase_continuation_budget(self):
        gateway = Gateway()
        tools = ToolRegistry((ToolRegistration(SEARCH, FakeTool({"search": lambda arguments: {}}),
            ToolCapability(SEARCH, ("healthbench_professional",), {"search": {"type": "object"}},
                           {"type": "object"}, {"type": "object"}, "read_only", 1.0, "synthetic")),))
        trace = [{"turn": 1, "observation_status": "schema_invalid", "public_error_code": "synthetic rejection"}]
        spent = [source_receipt("spent-one"), {"tool_id": SEARCH, "error_type": "synthetic_error", "result": None}]
        await runtime(gateway, tool_registry=tools, execution_adapters={"react": ReasoningExecutionAdapter(gateway)}).execute(
            pair(react=True), PROBLEM,
            historical_outputs={"a": "earlier answer"},
            historical_output_metadata={"a": {"artifact_version": "old:a", "tool_receipts": [source_receipt()]}},
            prior_failure_metadata={"a": {"execution_phase": "draft", "react_trace": trace,
                                          "tool_receipts": spent, "input_artifact_versions": {},
                                          "continuation_source_agent_id": "a"}},
        )
        draft = next(item for item in gateway.requests if item.agent.id == "a" and item.phase is ExecutionPhase.DRAFT)
        revision = next(item for item in gateway.requests if item.agent.id == "a" and item.phase is ExecutionPhase.REVISION)
        self.assertEqual(trace, [dict(item) for item in draft.action_history])
        self.assertEqual(spent, [dict(item) for item in draft.prior_tool_receipts])
        self.assertEqual("a", draft.continuation_source_agent_id)
        adapter = HealthBenchClinicalReactExecutionAdapter(gateway=gateway, tool_registry=tools,
                                                         max_turns=4, max_tool_calls=2)
        domain, completion = adapter._state_conditioned_action_domain(draft, adapter._continuation_observations(draft.action_history))
        self.assertEqual(frozenset(), domain)
        self.assertTrue(completion)
        self.assertEqual((), revision.action_history)
        self.assertEqual((), revision.prior_tool_receipts)
        self.assertIsNone(revision.continuation_source_agent_id)
        self.assertIn("synthetic-source-101", rendered(revision))

    async def test_repeated_reciprocal_edits_keep_receipts_in_existing_projection_bounds(self):
        gateway = Gateway()
        executor = runtime(gateway)
        result = await executor.execute(pair(reciprocal=False), PROBLEM)
        for _ in range(5):
            gateway.requests.clear()
            result = await executor.execute(pair(), PROBLEM, prior_outputs=result.outputs,
                                            prior_output_metadata=result.output_metadata)
            draft = next(item for item in gateway.requests if item.agent.id == "a" and item.phase is ExecutionPhase.DRAFT)
            self.assertIn("synthetic-source-101", rendered(draft))


if __name__ == "__main__":
    unittest.main()
