"""Offline completion-boundary regressions: no HTTP, model service or grader."""

import asyncio
from dataclasses import replace
import json
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, ExecutionPhase
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_evidence_adapter import (
    HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2,
    HealthBenchAuthoritativeReactExecutionAdapter,
)
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.react_execution import ReactExecutionError
from src.interactive.tool_runtime import StructuredAction, ToolCapability, ToolRegistration, ToolRegistry


TOOL = "healthbench-authoritative.search"
ERROR = "completion_artifact_requires_substantive_body"
TITLE = "Evidence Summary and Management Considerations for the Specified Condition"
BODY = "The available evidence is insufficient to support the proposed plan."


def complete(text):
    return {"kind": "complete", "name": "complete", "resource_id": None,
            "skill_id": None, "arguments": {"value": text}}


def request(question="Explain management of the specified condition.", *, output=True):
    return AgentRequest(
        request_id="fixture", run_id="fixture", graph_revision=1,
        problem=render_model_visible_conversation(({"role": "user", "content": question},)),
        agent=AgentNode("node", "model", "Complete the requested artifact.",
                        execution_mode="react", allowed_tools=()),
        model=ModelSpec("model", "provider"), provider=ProviderSpec("provider", kind="test"),
        phase=ExecutionPhase.SINGLE, is_output_agent=output,
    )


class NoToolBackend:
    def invoke(self, _request):
        raise AssertionError("Completion repair must not require a Tool call")


def registry():
    capability = ToolCapability(
        TOOL, ("healthbench_professional",),
        {"search": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}},
        {"type": "object"}, {"type": "object"}, "read_only", 1.0, "synthetic-v1",
    )
    return ToolRegistry((ToolRegistration(TOOL, NoToolBackend(), capability),))


class SequenceGateway:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, node_request):
        self.requests.append(node_request)
        return AgentResponse(json.dumps(complete(self.outputs.pop(0))))


class SubstantiveCompletionTests(unittest.TestCase):
    def adapter(self, gateway, *, adapter_class=HealthBenchClinicalReactExecutionAdapter, **settings):
        return adapter_class(
            gateway=gateway, tool_registry=registry(), max_turns=2, max_tool_calls=1,
            require_complete_natural_language_artifact=settings.pop("guard", True),
            completion_quality_profile=HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2,
            enforce_state_conditioned_completion_admission=True, **settings,
        )

    def test_observed_plain_titles_repair_at_existing_same_agent_boundary(self):
        # Public question/answer surfaces only, never task IDs or grader fields.
        cases = (
            ("How to manage concomitant varicecs and barrette's esophagus in the same patient",
             "Management Considerations for Concomitant Varices and Barrett's Esophagus in Cirrhosis—Confirmed Guideline Limitations Based on Current Evidence"),
            ("Write an MDT board conclusion for a patient with laryngeal squamous cell carcinoma, cT3N+M0, aimed at organ preservation.",
             "MDT BOARD CONCLUSION – LARYNGEAL SQUAMOUS CELL CARCINOMA ORGANS PRESERVATION CARE PATHWAY"),
        )
        for adapter_class in (HealthBenchAuthoritativeReactExecutionAdapter,
                              HealthBenchClinicalReactExecutionAdapter):
            for question, title in cases:
                with self.subTest(adapter=adapter_class.__name__, title=title):
                    gateway = SequenceGateway(title, BODY)
                    adapter = self.adapter(gateway, adapter_class=adapter_class)
                    response = asyncio.run(adapter.execute(request(question)))
                    self.assertEqual(BODY, response.text)
                    trace = response.metadata["react_trace"]
                    self.assertEqual(ERROR, trace[0]["public_error_code"])
                    self.assertEqual("schema_invalid", trace[0]["observation_status"])
                    self.assertEqual("completed", trace[1]["observation_status"])
                    self.assertEqual(2, len(gateway.requests))
                    self.assertEqual("node", gateway.requests[1].agent.id)
                    self.assertIn("Complete the requested body", gateway.requests[1].agent.contract)
                    self.assertEqual([], response.metadata["tool_receipts"])

    def test_all_heading_outline_is_not_a_requested_document_body(self):
        gateway = SequenceGateway("# Evidence Summary\n## Findings\n## Limitations", BODY)
        response = asyncio.run(self.adapter(gateway).execute(request()))
        self.assertEqual(ERROR, response.metadata["react_trace"][0]["public_error_code"])
        self.assertEqual(BODY, response.text)

    def test_short_answers_and_requested_fragment_forms_remain_legal(self):
        cases = (
            ("Explain the limitation.", "No Evidence Was Found"),
            ("How should this be managed?", "Seek urgent care"),
            ("Explain the finding.", "It is uncertain."),
            ("How common is the finding?", "0.7%"),
            ("Which category applies?", "Unclassified"),
            ("Is there sufficient evidence?", "No"),
            ("Write a title for this management document.", TITLE),
            ("Give an outline of the proposed report.", "# Findings\n## Limitations"),
            ("List the available categories.", "- First Category\n- Second Category"),
            ("Explain the finding.", "# Finding\nThe finding is uncertain."),
            ("Explain the finding.", "**The finding is uncertain.**"),
        )
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                gateway = SequenceGateway(answer)
                response = asyncio.run(self.adapter(gateway).execute(request(question)))
                self.assertEqual(answer, response.text)
                self.assertEqual(1, len(gateway.requests))

    def test_final_public_message_not_prior_conversation_controls_output_form(self):
        node_request = replace(request(), problem=render_model_visible_conversation((
            {"role": "user", "content": "Write a detailed report."},
            {"role": "assistant", "content": BODY},
            {"role": "user", "content": "Now give just a title."},
        )))
        gateway = SequenceGateway(TITLE)
        self.assertEqual(TITLE, asyncio.run(self.adapter(gateway).execute(node_request)).text)

    def test_opt_out_and_intermediate_contract_are_unchanged(self):
        for guard, output in ((False, True), (True, False)):
            gateway = SequenceGateway(TITLE)
            response = asyncio.run(self.adapter(gateway, guard=guard).execute(request(output=output)))
            self.assertEqual(TITLE, response.text)
            self.assertEqual(1, len(gateway.requests))

    def test_unrepaired_scaffold_exhausts_original_budget_without_publishing_output(self):
        gateway = SequenceGateway(TITLE, TITLE)
        adapter = self.adapter(gateway)
        with self.assertRaises(ReactExecutionError) as failure:
            asyncio.run(adapter.execute(request()))
        self.assertEqual(2, len(gateway.requests))
        self.assertEqual([ERROR, ERROR], [row["public_error_code"] for row in failure.exception.react_trace])
        # No task context may leak after either a success or an exception.
        self.assertIsNone(adapter._completion_error(
            action=StructuredAction.from_value(complete(TITLE)), artifact=TITLE, tool_receipts=[],
        ))

    def test_concurrent_requests_do_not_share_completion_expectations(self):
        class InterleavedGateway:
            def __init__(self):
                self.counts = {}

            async def generate(self, node_request):
                key = node_request.run_id
                self.counts[key] = self.counts.get(key, 0) + 1
                await asyncio.sleep(0)
                answer = BODY if self.counts[key] > 1 else TITLE
                return AgentResponse(json.dumps(complete(answer)))

        async def run():
            gateway = InterleavedGateway()
            adapter = self.adapter(gateway)
            body_request = replace(request(), run_id="body", request_id="body")
            title_request = replace(request("Write a title for the report."), run_id="title", request_id="title")
            results = await asyncio.gather(adapter.execute(body_request), adapter.execute(title_request))
            return results, gateway.counts

        results, counts = asyncio.run(run())
        self.assertEqual([BODY, TITLE], [result.text for result in results])
        self.assertEqual({"body": 2, "title": 1}, counts)


if __name__ == "__main__":
    unittest.main()
