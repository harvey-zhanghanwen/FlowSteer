"""Bounded public evidence repair using synthetic sources only; no API calls."""

import asyncio
from dataclasses import replace
import json

import pytest

from src.interactive.agent_runtime import AgentResponse, CommunicationCondition, UpstreamMessage
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_evidence_adapter import _COMPLETION_REQUEST
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.react_execution import ReactExecutionError
from src.interactive.tool_runtime import StructuredAction
from tests.unit.test_healthbench_clinical_react import (
    READ, SEARCH, SequenceGateway, action, artifact, complete, evidence,
    observation, receipt, registry, request,
)


def adapter(*outputs, enabled=True, max_turns=6):
    return HealthBenchClinicalReactExecutionAdapter(
        gateway=SequenceGateway(*outputs), tool_registry=registry(),
        max_turns=max_turns, max_tool_calls=3,
        require_structured_evidence_artifact=True,
        enable_evidence_repair_feedback=enabled,
    )


def check(executor, node, value, receipts):
    token = _COMPLETION_REQUEST.set(node)
    try:
        error = executor._structured_evidence_artifact_error(value, receipts)
    finally:
        _COMPLETION_REQUEST.reset(token)
    detail = executor._action_error_feedback(
        request=node, action=StructuredAction.from_value(complete(value)),
        public_error_code=error, tool_receipts=receipts, observations=[],
    )
    return error, detail


@pytest.mark.parametrize("field", ["source", "title", "date", "url"])
def test_exact_receipt_metadata_remains_strict_and_repair_names_real_field(field):
    row = evidence()
    submitted = artifact(row)
    submitted["evidence_items"][0][field] = "invented decorated metadata"
    executor = adapter()
    error, detail = check(executor, request(), submitted, [receipt(SEARCH, row)])
    assert error == "structured_evidence_item_receipt_binding_invalid"
    context = detail["repair_context"]
    assert context["evidence_item_index"] == 0
    assert context["receipt_candidates"][0]["mismatched_fields"] == [field]
    assert context["receipt_candidates"][0]["canonical_metadata"][field] == row[field]
    assert context["receipt_candidates"][0]["verbatim_excerpt_preview"] == row["excerpt"]
    assert "invented decorated metadata" not in json.dumps(detail)
    assert submitted["evidence_items"][0][field] == "invented decorated metadata"


def test_invalid_quote_is_not_accepted_and_exact_preview_is_bounded():
    row = {**evidence(), "excerpt": "Original first sentence. " + "Original next sentence. " * 60,
           "source_id": "pubmed:123456", "truncated": True, "next_offset": 16000}
    submitted = artifact(row)
    submitted["evidence_items"][0]["evidence_span"] = "Original first sentence ... Original next sentence"
    error, detail = check(adapter(), request(), submitted, [receipt(SEARCH, row)])
    assert error == "structured_evidence_item_span_not_in_receipt"
    candidate = detail["repair_context"]["receipt_candidates"][0]
    assert candidate["mismatched_fields"] == ["evidence_span"]
    assert candidate["verbatim_excerpt_preview"] == row["excerpt"][:400]
    assert candidate["next_offset"] == 16000
    assert candidate["preview_is_complete_excerpt"] is False


def test_item_index_points_to_first_failure_without_repeating_all_sources():
    first, second = evidence("fixture-a"), evidence("fixture-b")
    submitted = artifact(first)
    submitted["evidence_items"].append(artifact(second)["evidence_items"][0])
    submitted["evidence_items"][1]["date"] = "invented date"
    receipts = [receipt(SEARCH, row) for row in [first, second] + [evidence(str(i)) for i in range(30)]]
    _, detail = check(adapter(), request(), submitted, receipts)
    assert detail["repair_context"]["evidence_item_index"] == 1
    assert len(detail["repair_context"]["receipt_candidates"]) == 1


def test_repeated_routed_receipts_are_not_copied_twice_into_repair_feedback():
    row = evidence()
    source = UpstreamMessage("upstream", "node", "A source artifact.", tool_receipts=(receipt(SEARCH, row),))
    bad = artifact(row)
    bad["evidence_items"][0]["source"] = "wrong metadata"
    _, detail = check(adapter(), replace(request(), upstream=(source,)), bad, [receipt(SEARCH, row)])
    assert len(detail["repair_context"]["receipt_candidates"]) == 1


@pytest.mark.parametrize("masked", [False, True])
def test_upstream_receipts_are_feedback_sources_but_free_text_is_not(masked):
    row = evidence()
    source = UpstreamMessage("upstream", "node", "Untrusted rewritten evidence summary.",
                             tool_receipts=(receipt(SEARCH, row),))
    node = replace(request(), upstream=(source,))
    if masked:
        node = replace(node, communication_condition=CommunicationCondition.UPSTREAM_MASKED)
    submitted = artifact(row)
    submitted["evidence_items"][0]["title"] = "a different title"
    _, detail = check(adapter(), node, submitted, [])
    if masked:
        assert detail["repair_context"]["observed_document_ids"] == []
    else:
        assert detail["repair_context"]["receipt_candidates"][0]["canonical_metadata"]["title"] == row["title"]


def test_feedback_is_opt_in_and_valid_artifact_unchanged():
    row = evidence()
    submitted = artifact(row)
    submitted["evidence_items"][0]["title"] = "a different title"
    error, detail = check(adapter(enabled=False), request(), submitted, [receipt(SEARCH, row)])
    assert error == "structured_evidence_item_receipt_binding_invalid" and detail == {}
    valid = artifact(row)
    error, detail = check(adapter(), request(), valid, [receipt(SEARCH, row)])
    assert error is None and detail == {} and valid == artifact(row)


def test_complete_repair_happens_inside_existing_loop_without_another_tool_call():
    row = evidence()
    bad = artifact(row)
    bad["evidence_items"][0]["source"] = "decorated wrong source"
    executor = adapter(action(SEARCH, "synthetic relation"), complete(bad), complete(artifact(row)))
    response = asyncio.run(executor.execute(request()))
    assert json.loads(response.text) == artifact(row)
    assert response.metadata["tool_calls"] == 1
    assert response.metadata["react_turns_used"] == 3
    repair_contract = executor._gateway.requests[2].agent.contract
    assert '"evidence_item_index":0' in repair_contract
    assert '"source":"Synthetic source"' in repair_contract
    assert "decorated wrong source" not in repair_contract
    failed = response.metadata["react_trace"][1]
    assert failed["structured_action"]["arguments"]["value"] == bad
    restored = executor._continuation_observations((failed,))
    assert restored[0]["repair_context"] == failed["repair_context"]
    assert executor._model_visible_observations(restored)[0]["repair_context"] == failed["repair_context"]


def test_guessed_source_read_rejected_without_network_or_budget_then_model_repairs():
    row = {**evidence(), "source_id": "pubmed:123456", "full_text_source_id": "pmc:PMC987654"}
    source = UpstreamMessage("upstream", "node", "An interpretation.", tool_receipts=(receipt(SEARCH, row),))
    executor = adapter(action(READ, "pmc:PMC999999"), action(READ, "pmc:PMC987654"), complete("Retrieved source is available with its stated limitations."))
    response = asyncio.run(executor.execute(replace(request(output=True), upstream=(source,))))
    assert response.metadata["tool_calls"] == 1
    calls = executor._tool_registry._backend(READ).calls
    assert len(calls) == 1 and calls[0].arguments["source_id"] == "pmc:PMC987654"
    repair = executor._gateway.requests[1].agent.contract
    assert '"observed_source_ids":["pubmed:123456","pmc:PMC987654","fixture-1"]' in repair
    assert "PMC999999" not in repair


@pytest.mark.parametrize("source_id", ["pubmed:123456", "pmc:PMC987654", "https://example.invalid/actual-source"])
def test_original_public_references_keep_existing_direct_read_admission(source_id):
    node = replace(request(), problem=render_model_visible_conversation(({
        "role": "user", "content": f"Please read ({source_id}) and explain its limitations.",
    },)))
    executor = adapter()
    assert executor._tool_action_error(request=node, action=StructuredAction.from_value(action(READ, source_id)), observations=[]) is None
    assert executor._tool_action_error(request=node, action=StructuredAction.from_value(action(READ, source_id[:-1])), observations=[]) == "source_read_identifier_not_observed"


def test_original_id_followed_by_sentence_punctuation_is_still_a_reference():
    node = replace(request(), problem=render_model_visible_conversation(({
        "role": "user", "content": "Read pubmed:123456. Explain its limitations.",
    },)))
    assert adapter()._tool_action_error(request=node,
        action=StructuredAction.from_value(action(READ, "pubmed:123456")), observations=[]) is None


def test_metadata_only_source_does_not_become_supported_evidence_through_repair():
    row = {**evidence(), "excerpt": "", "source_id": "pubmed:123456"}
    submitted = artifact(evidence())
    error, detail = check(adapter(), request(), submitted, [receipt(SEARCH, row)])
    assert error == "structured_evidence_item_span_not_in_receipt"
    assert detail["repair_context"]["receipt_candidates"][0]["verbatim_excerpt_preview"] == ""
    assert "metadata-only hits do not establish clinical findings" in detail["repair_instruction"]


def test_source_read_old_default_is_unchanged_and_sources_can_be_observed_locally():
    executor = adapter()
    read = StructuredAction.from_value(action(READ, "fixture-1"))
    assert executor._tool_action_error(request=request(), action=read,
        observations=[observation(SEARCH, "synthetic relation")]) is None
    assert adapter(enabled=False)._tool_action_error(request=request(), action=read, observations=[]) is None


def test_truncated_json_is_not_silently_completed_or_scored():
    class TruncatedGateway:
        async def generate(self, request):
            return AgentResponse('{"kind":"complete","arguments":{"value":"A partial answer')
    executor = adapter(max_turns=1)
    executor._gateway = TruncatedGateway()
    with pytest.raises(ReactExecutionError) as caught:
        asyncio.run(executor.execute(request(output=True)))
    assert caught.value.react_trace[0]["observation_status"] == "parse_error"
    assert not caught.value.tool_receipts
