"""Synthetic regression checks for the model-visible HealthBench receipt path."""

from dataclasses import replace
import json
import unittest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2 as V2,
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3 as V3,
    AgentRequest,
    CommunicationCondition,
    ExecutionPhase,
    UpstreamMessage,
)
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.model_registry import ModelSpec, ProviderSpec
from src.interactive.openai_gateway import (
    MASKED_UPSTREAM_CONTENT,
    OpenAICompatibleGateway,
    _HEALTHBENCH_V3_EXECUTION_SUPPLEMENT,
    _HEALTHBENCH_V3_PROMPT_CHARS,
    _HEALTHBENCH_V3_UPSTREAM_CHARS,
    _format_upstream,
    build_agent_messages,
)


def evidence(document_id="synthetic-a", excerpt="A source supports the requested relation."):
    return {
        "document_id": document_id,
        "source": "Synthetic literature",
        "title": "Synthetic source " + document_id,
        "date": "2031",
        "url": "https://example.invalid/" + document_id,
        "excerpt": excerpt,
    }


def receipt(*rows, **extra):
    return {
        "tool_id": "healthbench-authoritative.search",
        "error_type": None,
        "request": {"action": "search", "arguments": {"query": "synthetic relation"}},
        "result": {"completed": True, "value": {
            "operation": "search", "query": "synthetic relation", "evidence": list(rows),
            "irrelevant_raw_body": "DO_NOT_REPLAY_RAW_BODY",
        }},
        **extra,
    }


def artifact(*cited, summary="The producer found no evidence."):
    return json.dumps({
        "schema_version": "healthbench.structured-evidence.v1",
        "status": "insufficient", "summary": summary,
        "evidence_items": [
            {**{key: value for key, value in row.items() if key != "excerpt"}, "evidence_span": row["excerpt"]}
            for row in cited
        ],
        "uncertainties": ["Independent source appraisal is still needed."],
    })


def upstream(*rows, content=None, source="retriever", **extra):
    return UpstreamMessage(
        source, "output", content if content is not None else artifact(),
        artifact_version=source + "-v1", source_execution_mode="react",
        source_model_id="synthetic-model", source_contract="Retrieve the requested relation.",
        tool_receipts=(receipt(*rows),), **extra,
    )


def request(*messages, profile=V3, **extra):
    provider = ProviderSpec("synthetic", endpoint="https://example.invalid/v1", kind="openai-compatible")
    model = ModelSpec("synthetic-model", "synthetic", model_name="synthetic",
                      metadata={"healthbench_execution_protocol": "contract-is-not-evidence.output-complete.slot-binding.closure.v5"})
    base = AgentRequest(
        "run:1:output", "synthetic-run", 1,
        render_model_visible_conversation(({"role": "user", "content": "Explain the requested relation, preserving the population."},)),
        AgentNode("output", "synthetic-model", "Answer the original conversation."),
        model, provider, ExecutionPhase.SINGLE,
        is_output_agent=True, upstream=messages,
        artifact_communication_profile=profile,
    )
    return replace(base, **extra)


def projection(message):
    text = build_agent_messages(request(message))[-1]["content"]
    return json.loads(text.split("[Upstream artifact]\n", 1)[1].split("\n\nProduce", 1)[0])


class HealthBenchEvidenceProjectionV3Tests(unittest.TestCase):
    def test_empty_citations_do_not_erase_successful_observation(self):
        row = evidence()
        original = upstream(row)
        result = projection(original)
        self.assertEqual("The producer found no evidence.", result["producer_artifact"]["summary"])
        projected = result["retrieval_evidence"]
        self.assertEqual(0, projected["artifact_reference_count"])
        self.assertEqual(0, projected["receipt_bound_reference_count"])
        self.assertEqual(row["excerpt"], projected["evidence_receipts"][0]["excerpts"][0])
        self.assertEqual("retrieved-not-endorsed", projected["evidence_receipts"][0]["evidence_status"])
        self.assertEqual(row, original.tool_receipts[0]["result"]["value"]["evidence"][0])

    def test_wrong_reference_is_not_promoted_but_receipt_survives(self):
        row = evidence()
        wrong = {**row, "excerpt": "FABRICATED_REFERENCE", "supported_claim": "FABRICATED_CLAIM", "conditions_or_qualifiers": "FABRICATED_QUALIFIER"}
        result = projection(upstream(row, content=artifact(wrong)))["retrieval_evidence"]
        self.assertEqual(1, result["artifact_reference_count"])
        self.assertEqual(0, result["receipt_bound_reference_count"])
        self.assertEqual([], result["evidence_receipts"][0]["artifact_cited_spans"])
        self.assertNotIn("FABRICATED_REFERENCE", json.dumps(result))
        self.assertNotIn("FABRICATED_CLAIM", json.dumps(result))
        self.assertEqual([], result["evidence_receipts"][0]["producer_interpretations"])

    def test_downstream_payload_keeps_bound_claim_and_qualifier(self):
        row = evidence("qualified", "Benefit applies to population A in the stated time window, except group B.")
        claim = {**row, "supported_claim": "Population A may benefit in the documented window.",
                 "conditions_or_qualifiers": "Only population A, within the documented interval; except group B."}
        agent_request = request(upstream(row, content=artifact(claim)))
        rendered_messages = OpenAICompatibleGateway().request_payload(agent_request)["messages"]
        text = json.dumps(rendered_messages)
        self.assertIn(claim["supported_claim"], text)
        self.assertIn(claim["conditions_or_qualifiers"], text)
        result = projection(agent_request.upstream[0])["retrieval_evidence"]["evidence_receipts"][0]
        self.assertEqual("qualified", result["document_id"])
        self.assertEqual("retrieved-not-endorsed", result["evidence_status"])
        interpretation = result["producer_interpretations"][0]
        self.assertEqual("producer-interpretation-not-independently-verified", interpretation["claim_status"])
        self.assertEqual(row["excerpt"], result["artifact_cited_spans"][interpretation["receipt_bound_span_index"]])

    def test_long_qualifier_keeps_ending_and_discloses_middle_omission(self):
        row = evidence("long-qualified")
        qualifier = "For population A only. " + "Repeated detail. " * 160 + " EXCLUDE POPULATION B AT THE FINAL TIME WINDOW."
        claim = {**row, "supported_claim": "A producer interpretation.", "conditions_or_qualifiers": qualifier}
        result = projection(upstream(row, content=artifact(claim)))
        visible = result["retrieval_evidence"]["evidence_receipts"][0]["producer_interpretations"][0]["conditions_or_qualifiers"]
        self.assertIn("For population A only.", visible)
        self.assertIn("EXCLUDE POPULATION B AT THE FINAL TIME WINDOW.", visible)
        self.assertIn("projection truncated", visible)
        self.assertLessEqual(len(visible), 1600)

    def test_producer_summary_and_uncertainty_omissions_are_explicit(self):
        value = json.loads(artifact(summary="Beginning. " + "detail " * 1000 + " FINAL QUALIFIER."))
        value["uncertainties"] = [f"Uncertainty {i}" for i in range(7)]
        result = projection(upstream(evidence(), content=json.dumps(value)))["producer_artifact"]
        self.assertEqual("partial", result["projection_status"])
        self.assertEqual(3, result["omitted_uncertainties_count"])
        self.assertIn("FINAL QUALIFIER.", result["summary"])
        self.assertIn("projection truncated", result["summary"])
        self.assertEqual(["Uncertainty 0", "Uncertainty 1", "Uncertainty 5", "Uncertainty 6"], result["uncertainties"])

    def test_support_and_conflict_are_both_preserved(self):
        support = evidence("support", "The relation is supported for population A.")
        conflict = evidence("conflict", "The relation is not supported for population B.")
        result = projection(upstream(support, conflict, content=artifact(support, support)))
        rows = result["retrieval_evidence"]["evidence_receipts"]
        self.assertEqual(["support", "conflict"], [row["document_id"] for row in rows])
        self.assertEqual([support["excerpt"]], rows[0]["artifact_cited_spans"])
        self.assertEqual(1, result["retrieval_evidence"]["receipt_bound_reference_count"])
        for row in rows:
            for key in ("source", "title", "date", "url", "document_id"):
                self.assertTrue(row[key])

    def test_deduplicates_documents_across_repeated_receipts_and_sources(self):
        row = evidence()
        first = upstream(row)
        first = replace(first, tool_receipts=(receipt(row), receipt(row, latency_ms=999)))
        text = build_agent_messages(request(first, first, upstream(row, source="peer")))[-1]["content"]
        self.assertEqual(1, text.count('"document_id":"synthetic-a"'))
        self.assertNotIn("latency_ms", text)
        self.assertNotIn("DO_NOT_REPLAY_RAW_BODY", text)

    def test_document_dedup_does_not_erase_later_producer_qualifier(self):
        row = evidence("shared")
        later = {**row, "supported_claim": "A distinct interpretation from the second producer.",
                 "conditions_or_qualifiers": "THE LATER PRODUCER EXCLUDES POPULATION C."}
        text = build_agent_messages(request(upstream(row), upstream(row, source="reviewer", content=artifact(later))))[-1]["content"]
        self.assertEqual(1, text.count('"title":"Synthetic source shared"'))
        self.assertIn(later["conditions_or_qualifiers"], text)
        self.assertIn('"previously_projected_source_interpretations"', text)
        self.assertIn('"claim_status":"producer-interpretation-not-independently-verified"', text)

    def test_nested_receipts_survive_without_recursive_raw_body(self):
        raw = upstream(evidence(), content=artifact()).to_dict()
        message = upstream(source="synthesizer", content="A producer synthesis.",
                           input_artifact_provenance=(raw, raw))
        text = build_agent_messages(request(message))[-1]["content"]
        self.assertEqual(1, text.count('"document_id":"synthetic-a"'))
        self.assertIn('"producer_agent_id":"retriever"', text)
        self.assertNotIn("DO_NOT_REPLAY_RAW_BODY", text)
        self.assertNotIn('"input_artifact_provenance"', text)

    def test_failed_and_nonsearch_receipts_are_not_evidence(self):
        message = replace(upstream(), tool_receipts=(
            receipt(evidence("failed"), error_type="provider_error"),
            {**receipt(evidence("incomplete")), "result": {"completed": False}},
            receipt(evidence("other-tool"), tool_id="other.search"),
        ))
        self.assertEqual([], projection(message)["retrieval_evidence"]["evidence_receipts"])

    def test_unstructured_producer_still_receives_bounded_receipt_projection(self):
        result = projection(upstream(evidence(), content="The producer rejected the source."))
        self.assertEqual("The producer rejected the source.", result["producer_artifact"])
        self.assertEqual(1, len(result["retrieval_evidence"]["evidence_receipts"]))

    def test_per_upstream_and_shared_peer_projection_are_bounded(self):
        messages = tuple(
            upstream(*(evidence(f"doc-{i}-{j}", "source excerpt " * 3000) for j in range(30)),
                     source=f"producer-{i}", content="summary " * 3000)
            for i in range(8)
        )
        state = {"remaining": _HEALTHBENCH_V3_PROMPT_CHARS, "sources": set(), "envelopes": set()}
        text = _format_upstream(messages, CommunicationCondition.NORMAL,
                                project_healthbench_structured_evidence=True,
                                artifact_communication_profile=V3, healthbench_projection_state=state)
        peer = _format_upstream((upstream(evidence("peer"), source="peer"),), CommunicationCondition.NORMAL,
                                project_healthbench_structured_evidence=True,
                                artifact_communication_profile=V3, healthbench_projection_state=state)
        self.assertLessEqual(len(text) + len(peer), _HEALTHBENCH_V3_PROMPT_CHARS)
        for envelope in text.split("\n\n"):
            self.assertLessEqual(len(envelope), _HEALTHBENCH_V3_UPSTREAM_CHARS)
        self.assertIn("truncated", text)

    def test_mask_removes_artifact_receipt_and_nested_content(self):
        message = upstream(evidence("SECRET_DOCUMENT"), content=artifact(summary="SECRET_SUMMARY"),
                           input_artifact_provenance=(upstream(evidence("SECRET_NESTED")).to_dict(),))
        message = replace(message, source_contract="SECRET_CONTRACT", request_or_dependency="SECRET_DEPENDENCY")
        rendered = build_agent_messages(request(
            message, communication_condition=CommunicationCondition.UPSTREAM_MASKED,
            phase=ExecutionPhase.REVISION, own_draft="Own public draft.", peer_draft=message,
        ))
        text = json.dumps(rendered)
        self.assertIn(MASKED_UPSTREAM_CONTENT, text)
        self.assertNotIn("SECRET_", text)
        self.assertNotIn('"retrieval_evidence"', text)

    def test_v2_retains_historical_citation_only_rendering(self):
        message = upstream(evidence())
        text = json.dumps(build_agent_messages(request(message, profile=V2)))
        self.assertIn("complete-no-evidence-references", text)
        self.assertNotIn("synthetic-a", text)
        self.assertNotIn(_HEALTHBENCH_V3_EXECUTION_SUPPLEMENT, text)

    def test_non_healthbench_and_direct_do_not_receive_new_protocol(self):
        message = upstream(evidence())
        v3 = request(message, problem="A non-HealthBench QA task.")
        self.assertEqual(build_agent_messages(replace(v3, artifact_communication_profile=V2)), build_agent_messages(v3))
        direct = build_agent_messages(request(profile="legacy"))
        self.assertNotIn(_HEALTHBENCH_V3_EXECUTION_SUPPLEMENT, direct[0]["content"])

    def test_actual_request_payload_includes_projection_and_brief_protocol(self):
        agent_request = request(upstream(evidence()))
        rendered_messages = OpenAICompatibleGateway().request_payload(agent_request)["messages"]
        self.assertEqual(build_agent_messages(agent_request), rendered_messages)
        self.assertIn(_HEALTHBENCH_V3_EXECUTION_SUPPLEMENT, rendered_messages[0]["content"])
        text = json.dumps(rendered_messages)
        self.assertIn("synthetic-a", text)
        self.assertIn("retrieved-not-endorsed", text)
        self.assertNotIn("DO_NOT_REPLAY_RAW_BODY", text)


if __name__ == "__main__":
    unittest.main()
