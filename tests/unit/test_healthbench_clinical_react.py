"""Synthetic optional-tool checks; no model, HTTP, benchmark or grader calls."""

import asyncio
from dataclasses import replace
import json
import unittest

from jsonschema import Draft202012Validator

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3,
    AgentRequest, AgentResponse, ExecutionPhase, UpstreamMessage,
)
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_evidence_adapter import HealthBenchAuthoritativeReactExecutionAdapter
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import (
    _healthbench_search_candidates, _healthbench_v3_receipts, build_agent_messages,
)
from src.interactive.tool_runtime import (
    StructuredAction, ToolCapability, ToolRegistration, ToolRegistry, ToolRequest, ToolResult,
)


SEARCH = "healthbench-authoritative.search"
LOCAL = "healthbench-medrag.search"
READ = "healthbench-source.read"
DRUG = "healthbench-drug.lookup"
CALC = "healthbench-computation.calculator"
SPECS = {
    SEARCH: ("search", "query"), LOCAL: ("search", "query"),
    READ: ("read_source", "source_id"), DRUG: ("drug_lookup", "drug_name"),
    CALC: ("calculator", "expression"),
}


def evidence(document_id="fixture-1"):
    return {
        "source_type": "fixture", "source": "Synthetic source",
        "document_id": document_id, "title": "Synthetic relation",
        "date": "2030", "url": "https://example.invalid/source",
        "excerpt": "The source describes the relation only for the specified group.",
    }


def artifact(row):
    return {
        "schema_version": "healthbench.structured-evidence.v1", "status": "supported",
        "summary": "The requested relation is documented for the stated group.",
        "evidence_items": [{
            **{key: row[key] for key in ("document_id", "source", "title", "date", "url")},
            "evidence_span": row["excerpt"],
            "supported_claim": "The relation is documented for the stated group.",
            "conditions_or_qualifiers": "Only for the specified group.",
        }],
        "uncertainties": [],
    }


def action(tool, value, **extra):
    name, field = SPECS[tool]
    return {"kind": "tool", "name": name, "resource_id": tool,
            "skill_id": None, "arguments": {field: value, **extra}}


def complete(value):
    return {"kind": "complete", "name": "complete", "resource_id": None,
            "skill_id": None, "arguments": {"value": value}}


def receipt(tool, row=None, value="synthetic"):
    sampled = action(tool, value)
    return {
        "tool_id": tool, "error_type": None,
        "request": {"action": sampled["name"], "arguments": sampled["arguments"]},
        "result": {"completed": True, "value": {
            "operation": sampled["name"], "query": value,
            "evidence": [row or evidence()],
        }},
    }


def observation(tool, value, *, error=False, **extra):
    return {
        "observation_status": "tool_error" if error else "success",
        "executed_action": action(tool, value, **extra),
        "result": {"operation": SPECS[tool][0], "evidence": [evidence()]},
        "completed": not error,
    }


class SequenceGateway:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return AgentResponse(json.dumps(self.outputs.pop(0)))


class SyntheticBackend:
    def __init__(self, tool):
        self.tool = tool
        self.calls = []

    def invoke(self, request: ToolRequest):
        self.calls.append(request)
        if self.tool == CALC:
            return ToolResult({"operation": "calculator", "observation": "[RESULT] 2 + 3 = 5"})
        if self.tool == LOCAL:
            return ToolResult({
                "operation": "search", "query": request.arguments["query"],
                "frozen_corpus": {"source": "Synthetic frozen textbook"},
                "ranked_chunks": [{"document_id": "local-1", "title": "Synthetic textbook",
                                   "text": evidence()["excerpt"], "score": 2.0, "matched_terms": ["relation"]}],
            })
        row = evidence()
        if self.tool == READ:
            row.update({"source_id": "pubmed:fixture-1", "content_type": "abstract",
                        "version": None, "offset": request.arguments.get("offset", 0),
                        "next_offset": None, "truncated": False})
        return ToolResult(receipt(self.tool, row)["result"]["value"])


def registry():
    registrations = []
    for tool, (name, field) in SPECS.items():
        arguments = {"type": "object", "required": [field],
                     "properties": {field: {"type": "string", "minLength": 1}},
                     "additionalProperties": False}
        if tool == READ:
            arguments["properties"]["offset"] = {"type": "integer", "minimum": 0}
        cap = ToolCapability(tool, ("healthbench_professional",), {name: arguments},
                             {"type": "object"}, {"type": "object"}, "read_only", 1.0,
                             "synthetic-v1")
        registrations.append(ToolRegistration(tool, SyntheticBackend(tool), cap))
    return ToolRegistry(tuple(registrations))


def request(*allowed, output=False, **extra):
    return AgentRequest(
        "fixture:agent", "fixture", 1,
        render_model_visible_conversation(({"role": "user", "content": "Explain the requested relation for the specified group."},)),
        AgentNode("node", "fixture-model", "Complete the requested analysis.",
                  allowed_tools=allowed or tuple(SPECS), execution_mode="react"),
        ModelSpec("fixture-model", "fixture-provider"), ProviderSpec("fixture-provider", kind="test"),
        ExecutionPhase.SINGLE, is_output_agent=output, **extra,
    )


class HealthBenchClinicalReactTests(unittest.TestCase):
    def adapter(self, *outputs, **settings):
        self.gateway = SequenceGateway(*outputs)
        self.registry = registry()
        return HealthBenchClinicalReactExecutionAdapter(
            gateway=self.gateway, tool_registry=self.registry, max_turns=settings.pop("max_turns", 6),
            max_tool_calls=settings.pop("max_tool_calls", 3),
            require_structured_evidence_artifact=True,
            max_completion_artifact_characters=6000,
            **settings,
        )

    def test_domain_is_exactly_node_declared_tools_and_completion_is_immediate(self):
        adapter = self.adapter()
        node_request = request(READ, CALC)
        admitted, completion = adapter._state_conditioned_action_domain(node_request, [])
        self.assertEqual(frozenset({(READ, "read_source"), (CALC, "calculator")}), admitted)
        self.assertTrue(completion)
        schema = adapter._state_conditioned_response_schema(node_request, [])
        validator = Draft202012Validator(schema)
        self.assertTrue(validator.is_valid(action(READ, "pubmed:fixture-1")))
        self.assertTrue(validator.is_valid(action(CALC, "2 + 3")))
        self.assertFalse(validator.is_valid(action(SEARCH, "synthetic")))
        self.assertTrue(validator.is_valid(complete(artifact(evidence()))))

    def test_two_successful_searches_still_allow_source_read(self):
        adapter = self.adapter()
        seen = [observation(SEARCH, "first"), observation(LOCAL, "second")]
        admitted, completion = adapter._state_conditioned_action_domain(request(), seen)
        self.assertNotIn((SEARCH, "search"), admitted)
        self.assertNotIn((LOCAL, "search"), admitted)
        self.assertIn((READ, "read_source"), admitted)
        self.assertIn((DRUG, "drug_lookup"), admitted)
        self.assertIn((CALC, "calculator"), admitted)
        self.assertTrue(completion)
        seen.append(observation(READ, "pubmed:fixture-1"))
        self.assertEqual((frozenset(), True), adapter._state_conditioned_action_domain(request(), seen))

    def test_unrelated_nonempty_searches_do_not_consume_relevant_search_slots(self):
        adapter = self.adapter(require_relevant_evidence=True)
        node_request = replace(request(), problem="Explain the synthetic TRIAL-ZETA study.")
        seen = [
            observation(SEARCH, "TRIAL-ZETA study findings"),
            observation(SEARCH, "TRIAL-ZETA study design"),
        ]
        admitted, completion = adapter._state_conditioned_action_domain(node_request, seen)
        self.assertIn((SEARCH, "search"), admitted)
        self.assertIn((LOCAL, "search"), admitted)
        self.assertIn((READ, "read_source"), admitted)
        self.assertTrue(completion)
        # A relevance check must never replenish the original dispatch budget.
        seen.append(observation(SEARCH, "TRIAL-ZETA study population"))
        self.assertEqual((frozenset(), True), adapter._state_conditioned_action_domain(node_request, seen))

    def test_medrag_ranked_chunks_use_the_same_complete_entity_relevance_check(self):
        adapter = self.adapter(require_relevant_evidence=True)
        node_request = replace(request(), problem="Explain the synthetic TRIAL-ZETA study.")
        raw = self.registry._backend(LOCAL).invoke(ToolRequest("search", {"query": "TRIAL-ZETA study"})).value
        local = {**observation(LOCAL, "TRIAL-ZETA study"), "result": raw}
        unrelated = observation(SEARCH, "TRIAL-ZETA design")
        admitted, completion = adapter._state_conditioned_action_domain(node_request, [local, unrelated])
        self.assertIn((SEARCH, "search"), admitted)
        self.assertIn((LOCAL, "search"), admitted)
        self.assertTrue(completion)
        # Same raw schema, now with a genuine complete surface in both sources.
        local["result"]["ranked_chunks"][0]["title"] = "Synthetic TRIAL-ZETA publication"
        unrelated["result"]["evidence"][0]["title"] = "Synthetic TRIAL-ZETA study design"
        admitted, completion = adapter._state_conditioned_action_domain(node_request, [local, unrelated])
        self.assertNotIn((SEARCH, "search"), admitted)
        self.assertNotIn((LOCAL, "search"), admitted)
        self.assertIn((READ, "read_source"), admitted)
        self.assertTrue(completion)

    def test_relevance_setting_false_preserves_historical_nonempty_count(self):
        adapter = self.adapter(require_relevant_evidence=False)
        node_request = replace(request(), problem="Explain synthetic TRIAL-ZETA.")
        admitted, completion = adapter._state_conditioned_action_domain(node_request, [
            observation(SEARCH, "TRIAL-ZETA results"), observation(LOCAL, "TRIAL-ZETA design"),
        ])
        self.assertNotIn((SEARCH, "search"), admitted)
        self.assertNotIn((LOCAL, "search"), admitted)
        self.assertTrue(completion)

    def test_medrag_search_reuses_opted_in_public_task_anchor_without_rewriting(self):
        adapter = self.adapter(require_task_query_anchor=True)
        node_request = replace(request(LOCAL), problem="Compare synthetic TRIAL-ZETA results and applicability.")
        drift = StructuredAction.from_value(action(LOCAL, "unrelated enzyme metabolism"))
        grounded = StructuredAction.from_value(action(LOCAL, "TRIAL-ZETA applicability"))
        self.assertEqual("query_does_not_preserve_public_task_anchor", adapter._tool_action_error(
            request=node_request, action=drift, observations=[]))
        self.assertIsNone(adapter._tool_action_error(request=node_request, action=grounded, observations=[]))
        self.assertEqual("TRIAL-ZETA applicability", grounded.arguments["query"])
        # A different source is a legitimate retry, not a repeated local call.
        self.assertIsNone(adapter._tool_action_error(
            request=node_request, action=grounded,
            observations=[observation(SEARCH, "TRIAL-ZETA applicability")]))
        self.assertEqual("duplicate_tool_request", adapter._tool_action_error(
            request=node_request, action=grounded,
            observations=[observation(LOCAL, "TRIAL-ZETA applicability")]))

    def test_medrag_anchor_guard_is_opt_in_and_does_not_constrain_other_tool_arguments(self):
        node_request = replace(request(), problem="Compare synthetic TRIAL-ZETA results.")
        adapter = self.adapter(require_task_query_anchor=False)
        self.assertIsNone(adapter._tool_action_error(
            request=node_request, action=StructuredAction.from_value(action(LOCAL, "unrelated enzyme metabolism")),
            observations=[]))
        adapter = self.adapter(require_task_query_anchor=True)
        for sampled in (action(READ, "pubmed:fixture-1"), action(DRUG, "synthetic label"), action(CALC, "2 + 3")):
            with self.subTest(tool=sampled["resource_id"]):
                self.assertIsNone(adapter._tool_action_error(
                    request=node_request, action=StructuredAction.from_value(sampled), observations=[]))
        chinese = replace(request(LOCAL), problem='[{"role":"user","content":"合成中文问题"}]')
        self.assertIsNone(adapter._tool_action_error(
            request=chinese, action=StructuredAction.from_value(action(LOCAL, "synthetic translated query")), observations=[]))

    def test_failed_dispatches_share_budget_but_invalid_attempts_do_not(self):
        adapter = self.adapter(max_tool_calls=2)
        invalid = {"observation_status": "schema_invalid", "executed_action": action(READ, "missing")}
        seen = [observation(DRUG, "synthetic", error=True), invalid]
        self.assertTrue(adapter._state_conditioned_action_domain(request(), seen)[0])
        seen.append(observation(CALC, "2 + 3"))
        self.assertFalse(adapter._state_conditioned_action_domain(request(), seen)[0])

    def test_continuation_budget_uses_receipts_without_double_count(self):
        adapter = self.adapter()
        old = observation(SEARCH, "first")
        node_request = request(prior_tool_receipts=(receipt(SEARCH),),
                               action_history=({"turn": 1, "observation": old},))
        self.assertTrue(adapter._state_conditioned_action_domain(node_request, [old, observation(READ, "id")])[0])
        self.assertFalse(adapter._state_conditioned_action_domain(node_request, [old, observation(READ, "id"), observation(CALC, "2 + 3")])[0])

    def test_repeated_read_rejected_but_next_page_allowed(self):
        adapter = self.adapter()
        seen = [observation(READ, "pubmed:fixture-1"), observation(CALC, "2 + 3")]
        self.assertEqual("duplicate_tool_request", adapter._tool_action_error(
            request=request(), action=StructuredAction.from_value(action(READ, "pubmed:fixture-1")), observations=seen))
        self.assertIsNone(adapter._tool_action_error(
            request=request(), action=StructuredAction.from_value(action(READ, "pubmed:fixture-1", offset=1000)), observations=seen))

    def test_search_read_completion_keeps_lossless_receipts_and_observations(self):
        adapter = self.adapter(action(SEARCH, "synthetic relation"), action(READ, "pubmed:fixture-1"), complete(artifact(evidence())))
        response = asyncio.run(adapter.execute(request()))
        self.assertEqual(artifact(evidence()), json.loads(response.text))
        self.assertEqual([SEARCH, READ], [item["tool_id"] for item in response.metadata["tool_receipts"]])
        self.assertEqual(["success", "success", "completed"], [
            turn.get("observation_status") or turn["observation"]["observation_status"]
            for turn in response.metadata["react_trace"]])
        self.assertIn("pubmed:fixture-1", self.gateway.requests[2].agent.contract)
        self.assertIn("The source describes", self.gateway.requests[2].agent.contract)

    def test_pure_calculator_node_has_no_forced_search_or_literature_schema(self):
        adapter = self.adapter(action(CALC, "2 + 3"), complete("The computed result is 5."))
        node_request = request(CALC)
        self.assertEqual("string", adapter._completion_arguments_schema(node_request)["properties"]["value"]["type"])
        response = asyncio.run(adapter.execute(node_request))
        self.assertEqual("The computed result is 5.", response.text)
        self.assertEqual([CALC], [item["tool_id"] for item in response.metadata["tool_receipts"]])
        self.assertEqual((), adapter._successful_search_evidence(list(response.metadata["tool_receipts"])))

    def test_read_and_label_bind_exact_evidence_while_calculator_cannot(self):
        adapter = self.adapter()
        value = artifact(evidence())
        for tool in (READ, DRUG):
            with self.subTest(tool=tool):
                self.assertIsNone(adapter._structured_evidence_artifact_error(value, [receipt(tool)]))
        self.assertEqual("structured_evidence_item_receipt_binding_invalid", adapter._structured_evidence_artifact_error(value, [receipt(CALC)]))
        wrong = artifact(evidence())
        wrong["evidence_items"][0]["evidence_span"] = "Invented excerpt."
        self.assertEqual("structured_evidence_item_span_not_in_receipt", adapter._structured_evidence_artifact_error(wrong, [receipt(READ)]))

    def test_source_error_or_incomplete_receipt_is_not_evidence(self):
        adapter = self.adapter()
        failed = receipt(DRUG)
        failed["error_type"] = "SyntheticFailure"
        incomplete = receipt(READ)
        incomplete["result"]["completed"] = False
        self.assertEqual((), adapter._successful_search_evidence([failed, incomplete]))

    def test_medrag_projection_is_shared_by_agent_completion_and_gateway(self):
        adapter = self.adapter()
        raw = self.registry._backend(LOCAL).invoke(ToolRequest("search", {"query": "relation"})).value
        original = {"observation_status": "success", "executed_action": action(LOCAL, "relation"), "result": raw}
        visible = adapter._model_visible_observations([original])[0]["result"]
        self.assertIn("ranked_chunks", original["result"])
        self.assertNotIn("ranked_chunks", visible)
        row = visible["evidence"][0]
        persisted = receipt(LOCAL)
        persisted["result"]["value"] = raw
        self.assertEqual(row, _healthbench_search_candidates([persisted])[0][2])
        self.assertIsNone(adapter._structured_evidence_artifact_error(artifact(row), [persisted]))

    def test_optional_tools_do_not_enable_forced_search_flags(self):
        for name in ("require_initial_search", "require_refinement_on_insufficient_evidence"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.adapter(**{name: True})

    def test_old_authoritative_adapter_retains_search_only_domain(self):
        self.adapter()
        old = HealthBenchAuthoritativeReactExecutionAdapter(
            gateway=self.gateway, tool_registry=self.registry, max_turns=6, max_tool_calls=3,
            require_initial_search=True,
        )
        self.assertEqual((frozenset({(SEARCH, "search")}), False), old._state_conditioned_action_domain(request(), []))

    def test_source_and_label_provenance_reaches_director_projection(self):
        adapter = self.adapter(action(READ, "pubmed:fixture-1"), complete(artifact(evidence())))
        response = asyncio.run(adapter.execute(request(READ)))
        message = UpstreamMessage("node", "output", response.text, artifact_version="fixture-v1",
                                  source_execution_mode="react", tool_receipts=tuple(response.metadata["tool_receipts"]))
        projected = _healthbench_v3_receipts(message, seen_sources=set(), char_budget=6000)
        row = projected["evidence_receipts"][0]
        self.assertEqual("pubmed:fixture-1", row["source_retrieval"][0]["source_id"])
        self.assertEqual("abstract", row["source_retrieval"][0]["content_type"])
        self.assertFalse(row["source_retrieval"][0]["truncated"])
        self.assertEqual(evidence()["excerpt"], row["excerpts"][0])
        self.assertEqual(1, projected["receipt_bound_reference_count"])
        # The same projection is used by the downstream V4 Agent payload and
        # AgentWorkflowEnv's current_artifact_receipts Director feedback.
        output = request(output=True, upstream=(message,),
                         artifact_communication_profile=ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3)
        rendered = json.dumps(build_agent_messages(output))
        self.assertIn("pubmed:fixture-1", rendered)
        self.assertIn("retrieved-not-endorsed", rendered)

    def test_source_pages_survive_prior_search_and_other_agents_without_duplicate_page(self):
        seen = set()
        source = evidence()
        search_message = UpstreamMessage("searcher", "output", json.dumps(artifact(source)),
                                        tool_receipts=(receipt(SEARCH, source),))
        _healthbench_v3_receipts(search_message, seen_sources=seen, char_budget=6000)
        for offset in (0, 1000):
            row = {**source, "excerpt": f"Different information on the page at offset {offset}.",
                   "source_id": "pubmed:fixture-1", "content_type": "abstract",
                   "version": None, "offset": offset, "next_offset": offset + 1000,
                   "truncated": True}
            message = UpstreamMessage(f"reader-{offset}", "output", json.dumps(artifact(row)),
                                      tool_receipts=(receipt(READ, row),))
            projected = _healthbench_v3_receipts(message, seen_sources=seen, char_budget=6000)
            self.assertEqual([row["excerpt"]], projected["evidence_receipts"][0]["excerpts"])
            self.assertEqual(offset, projected["evidence_receipts"][0]["source_retrieval"][0]["offset"])
            self.assertEqual([row["excerpt"]], projected["evidence_receipts"][0]["artifact_cited_spans"])
            repeated = _healthbench_v3_receipts(replace(message, source_agent_id="another-reader"),
                                               seen_sources=seen, char_budget=6000)
            self.assertEqual([], repeated["evidence_receipts"])

    def test_multi_tool_budget_and_duplicate_are_enforced_in_actual_dispatch(self):
        adapter = self.adapter(
            action(READ, "pubmed:fixture-1"), action(CALC, "2 + 3"),
            action(READ, "pubmed:fixture-1"), action(DRUG, "synthetic"),
            action(READ, "pubmed:fixture-2"), complete("The completed response preserves uncertainty."),
        )
        response = asyncio.run(adapter.execute(request(output=True)))
        self.assertEqual([READ, CALC, DRUG], [item["tool_id"] for item in response.metadata["tool_receipts"]])
        self.assertEqual(1, len(self.registry._backend(READ).calls))
        errors = [turn.get("public_error_code") for turn in response.metadata["react_trace"]]
        self.assertIn("duplicate_tool_request", errors)
        self.assertIn("state_action_not_admitted", errors)

    def test_existing_completion_quality_recovers_with_same_agent(self):
        adapter = self.adapter(complete("# Heading"), complete("The completed response preserves uncertainty."),
                               require_complete_natural_language_artifact=True)
        response = asyncio.run(adapter.execute(request(output=True)))
        self.assertEqual("The completed response preserves uncertainty.", response.text)
        self.assertEqual("completion_artifact_is_heading_only", response.metadata["react_trace"][0]["public_error_code"])
        self.assertEqual(0, len(response.metadata["tool_receipts"]))


if __name__ == "__main__":
    unittest.main()
