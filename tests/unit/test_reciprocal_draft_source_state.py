"""Offline phase-source regression; no model, network or evaluator calls."""

from __future__ import annotations

import json
import unittest

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
    AgentRequest, AgentResponse, AgentRuntime, ExecutionPhase,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.openai_gateway import build_agent_messages
from src.interactive.tool_runtime import FakeTool, ToolCapability, ToolRegistration, ToolRegistry


def receipt():
    return {
        "tool_id": "healthbench-authoritative.search", "error_type": None,
        "request": {"action": "search", "arguments": {"query": "synthetic group"}},
        "result": {"completed": True, "value": {
            "operation": "search", "query": "synthetic group", "evidence": [{
                "document_id": "synthetic-draft-only-101", "source": "Synthetic source",
                "source_type": "pubmed", "title": "A synthetic observation", "date": None,
                "url": None, "excerpt": "The synthetic source reports a qualified finding.",
            }],
        }},
    }


class DraftSourceGateway:
    def __init__(self):
        self.requests = []

    async def generate(self, request: AgentRequest):
        self.requests.append(request)
        metadata = {}
        if request.agent.id == "a" and request.phase is ExecutionPhase.DRAFT:
            metadata = {"tool_receipts": [receipt()], "react_trace": [{
                "turn": 1, "observation_status": "completed",
                "action_text": "old phase control text must not resume",
            }]}
        return AgentResponse(f"Complete {request.agent.id} {request.phase.value} artifact.", metadata)


def graph():
    return AgentGraph(
        [AgentNode("a", "m", "Analyze available sources."),
         AgentNode("b", "m", "Produce the complete response.")],
        [AgentRelation("a", "b", True, True)], output_agent_id="b",
    )


def runtime(gateway):
    catalog = ModelRegistry([ProviderSpec("fake", kind="test")], [ModelSpec("m", "fake")])
    return AgentRuntime(
        catalog, gateway,
        artifact_communication_profile=ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
    )


class ReciprocalDraftSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_react_revision_can_bind_own_draft_source_without_another_search(self):
        source = receipt()["result"]["value"]["evidence"][0]
        supported = {
            "schema_version": "healthbench.structured-evidence.v1", "status": "supported",
            "summary": "The synthetic source reports a qualified finding.",
            "evidence_items": [{
                **{key: source[key] for key in ("document_id", "source", "title", "date", "url")},
                "evidence_span": source["excerpt"], "supported_claim": source["excerpt"],
                "conditions_or_qualifiers": "This is a synthetic test fixture.",
            }], "uncertainties": [],
        }

        class ReactGateway:
            def __init__(self):
                self.requests = []

            async def generate(self, request):
                key = (request.agent.id, request.phase)
                prior_calls = sum((item.agent.id, item.phase) == key for item in self.requests)
                self.requests.append(request)
                action = {"kind": "complete", "name": "complete", "resource_id": None,
                          "skill_id": None, "arguments": {"value": supported if key[0] == "a"
                                                        else "The synthetic source supports a qualified finding."}}
                if key == ("a", ExecutionPhase.DRAFT) and prior_calls == 0:
                    action.update(kind="tool", name="search", resource_id="healthbench-authoritative.search",
                                  arguments={"query": "synthetic group"})
                return AgentResponse(json.dumps(action))

        dispatched = []

        def search(arguments):
            dispatched.append(arguments)
            return receipt()["result"]["value"]

        tool_id = "healthbench-authoritative.search"
        tools = ToolRegistry((ToolRegistration(tool_id, FakeTool({"search": search}), ToolCapability(
            tool_id, ("healthbench_professional",), {"search": {"type": "object"}},
            {"type": "object"}, {"type": "object"}, "read_only", 1.0, "synthetic-v1",
        )),))
        gateway = ReactGateway()
        adapter = HealthBenchClinicalReactExecutionAdapter(
            gateway=gateway, tool_registry=tools, max_turns=2, max_tool_calls=1,
            require_structured_evidence_artifact=True,
        )
        catalog = ModelRegistry([ProviderSpec("fake", kind="test")], [ModelSpec("m", "fake")])
        executor = AgentRuntime(catalog, gateway, tool_registry=tools, execution_adapters={"react": adapter},
                                artifact_communication_profile=ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4)
        canvas = AgentGraph([AgentNode(name, "m", "Use the synthetic source.", execution_mode="react",
                                      allowed_tools=(tool_id,)) for name in ("a", "b")],
                            [AgentRelation("a", "b", True, True)], output_agent_id="b")
        env = AgentWorkflowEnv(catalog, runtime=executor, graph=canvas, problem="Explain the synthetic group.",
                               execute_on_edit=True, finish_only_when_admissible=True)
        edited = await env.step(AgentAction(AgentActionType.MODIFY_AGENT, agent_id="a",
                                            contract="Analyze the synthetic source finding."))
        self.assertIsNotNone(edited.execution, edited.feedback)
        self.assertEqual(1, len(dispatched))
        self.assertEqual(0, edited.execution.output_metadata["a"]["tool_calls"])
        self.assertEqual([], edited.execution.output_metadata["a"]["tool_receipts"])
        self.assertEqual("completed", edited.execution.output_metadata["a"]["react_trace"][0]["observation_status"])
        self.assertEqual(1, edited.execution.output_metadata["a"]["react_turns_used"])
        finished = await env.step(AgentAction(AgentActionType.FINISH))
        self.assertTrue(finished.accepted, finished.feedback)
        self.assertTrue(finished.done)
        self.assertEqual(5, len(gateway.requests))

    async def test_own_draft_sources_reach_revision_without_control_state_or_self_dependency(self):
        gateway = DraftSourceGateway()
        result = await runtime(gateway).execute(graph(), "Explain the synthetic group.")
        requests = {(item.agent.id, item.phase): item for item in gateway.requests}
        self.assertEqual(4, len(requests))
        for agent_id in ("a", "b"):
            self.assertEqual((), requests[agent_id, ExecutionPhase.DRAFT].upstream)
        revision = requests["a", ExecutionPhase.REVISION]
        own_source = next(item for item in revision.upstream if item.source_agent_id == "a")
        self.assertEqual("historical_evidence", own_source.message_type)
        self.assertEqual(requests["a", ExecutionPhase.DRAFT].request_id, own_source.artifact_version)
        self.assertIn("this graph revision", own_source.content)
        self.assertEqual((receipt(),), own_source.tool_receipts)
        self.assertIn("synthetic-draft-only-101", json.dumps(build_agent_messages(revision)))
        self.assertNotIn("old phase control text", json.dumps(build_agent_messages(revision)))
        self.assertEqual((), revision.action_history)
        self.assertEqual((), revision.prior_tool_receipts)
        self.assertIsNone(revision.continuation_source_agent_id)
        self.assertEqual({"b": revision.peer_draft.artifact_version},
                         result.output_metadata["a"]["input_artifact_versions"])
        self.assertFalse(result.output_metadata["a"].get("tool_receipts"))
        self.assertEqual([receipt()], result.output_metadata["a"]["input_artifact_provenance"][0]["tool_receipts"])

    async def test_real_canvas_edit_then_finish_reuses_revision_artifacts(self):
        gateway = DraftSourceGateway()
        executor = runtime(gateway)
        env = AgentWorkflowEnv(
            executor.model_registry, runtime=executor, graph=graph(),
            problem="Explain the synthetic group.", execute_on_edit=True,
            finish_only_when_admissible=True,
            require_output_protocol_artifact_for_set_output=True,
            recovery_policy="preserve_diagnose_repair_augment",
        )
        edited = await env.step(AgentAction(
            AgentActionType.MODIFY_AGENT, agent_id="a", contract="Analyze the available source findings.",
        ))
        self.assertTrue(edited.accepted, edited.feedback)
        self.assertIsNotNone(edited.execution)
        self.assertEqual(4, len(gateway.requests))
        self.assertIn("finish", env.model_admissible_action_types())
        finished = await env.step(AgentAction(AgentActionType.FINISH))
        self.assertTrue(finished.accepted, finished.feedback)
        self.assertTrue(finished.done)
        self.assertEqual("Complete b revision artifact.", finished.execution.final_answer)
        self.assertEqual(4, len(gateway.requests))

    async def test_new_own_phase_evidence_does_not_reset_unchanged_revision_budget(self):
        gateway = DraftSourceGateway()
        executor = runtime(gateway)
        revision_kwargs = {
            "agent": graph().get_node("a"), "phase": ExecutionPhase.REVISION,
            "upstream": (), "problem": "Explain the synthetic group.", "run_id": "run",
            "graph_revision": 2, "own_draft": "Current immutable draft.", "peer_draft": None,
            "own_draft_metadata": {"artifact_version": "run:2:a:draft", "tool_receipts": [receipt()]},
            "output_agent_id": "b", "format_output_agent": False, "is_format_predecessor": False,
            "communication_condition": "normal",
            "continuation_metadata": {"execution_phase": "revision", "input_artifact_versions": {},
                "react_trace": [{"turn": 2, "observation_status": "schema_invalid"}],
                "tool_receipts": [receipt()], "continuation_source_agent_id": "a"},
        }
        request = executor._request(**revision_kwargs)
        self.assertEqual(2, request.action_history[0]["turn"])
        self.assertEqual((receipt(),), request.prior_tool_receipts)
        self.assertEqual("a", request.continuation_source_agent_id)


if __name__ == "__main__":
    unittest.main()
