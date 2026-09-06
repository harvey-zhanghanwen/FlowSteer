"""Synthetic source-fidelity and routed binding regressions; no live calls."""

import asyncio
from dataclasses import replace
import json

import pytest

from src.interactive.agent_runtime import CommunicationCondition, UpstreamMessage
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_evidence_adapter import _COMPLETION_REQUEST
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.tool_runtime import StructuredAction
from tests.unit.test_healthbench_clinical_react import (
    LOCAL, READ, SEARCH, SequenceGateway, action, artifact, complete, evidence,
    observation, receipt, registry, request,
)


def adapter(*outputs, fidelity=True, **settings):
    return HealthBenchClinicalReactExecutionAdapter(
        gateway=SequenceGateway(*outputs), tool_registry=registry(),
        max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True,
        require_initial_query_fidelity=fidelity, **settings,
    )


def short_request(**changes):
    return replace(request(SEARCH, LOCAL, READ), problem=render_model_visible_conversation(({
        "role": "user", "content": "velorin nebulark trial",
    },)), **changes)


def admission(executor, node, query, observations=()):
    return executor._tool_action_error(request=node,
        action=StructuredAction.from_value(action(LOCAL, query)), observations=list(observations))


def test_initial_fidelity_is_opt_in_and_does_not_require_search_or_choose_source():
    node = short_request()
    changed = "velorin nebulark population efficacy"
    assert admission(adapter(fidelity=False), node, changed) is None
    executor = adapter()
    assert admission(executor, node, changed) == "initial_query_must_preserve_public_input"
    for query in ("velorin nebulark trial", '"NEBULARK" trial velorin'):
        assert admission(executor, node, query) is None
    admitted, can_complete = executor._state_conditioned_action_domain(node, [])
    assert can_complete
    assert {(LOCAL, "search"), (SEARCH, "search"), (READ, "read_source")} <= admitted
    assert admission(executor, request(), changed) is None  # Full sentence unchanged.


def test_rejected_initial_query_repairs_through_existing_observation_without_dispatch():
    executor = adapter(action(LOCAL, "velorin nebulark population efficacy"),
        action(LOCAL, "velorin nebulark trial"), complete("The retrieved text needs interpretation in its stated context."))
    response = asyncio.run(executor.execute(short_request(is_output_agent=True)))
    assert len(response.metadata["tool_receipts"]) == 1
    calls = executor._tool_registry._backend(LOCAL).calls
    assert len(calls) == 1 and calls[0].arguments["query"] == "velorin nebulark trial"
    repair_contract = executor._gateway.requests[1].agent.contract
    assert '"original_public_query":"velorin nebulark trial"' in repair_contract
    assert "initial_query_must_preserve_public_input" in repair_contract
    assert "population efficacy" not in repair_contract


@pytest.mark.parametrize("route", ["observed", "upstream", "peer", "summary_only", "masked"])
def test_first_query_can_refine_real_sources_but_not_contract_or_summary_assertions(route):
    row = evidence()
    source = UpstreamMessage("producer", "node", "A proposed interpretation.",
        tool_receipts=(receipt(SEARCH, row),))
    node = short_request()
    history = []
    if route == "observed":
        history = [observation(LOCAL, "velorin nebulark trial")]
    elif route == "upstream":
        node = replace(node, upstream=(source,))
    elif route == "peer":
        node = replace(node, peer_draft=source)
    elif route == "masked":
        node = replace(node, upstream=(source,), communication_condition=CommunicationCondition.UPSTREAM_MASKED)
    else:
        node = replace(node, upstream=(replace(source, content=json.dumps(artifact(row)), tool_receipts=()),))
    error = admission(adapter(), node, "velorin nebulark source refinement", history)
    assert error == ("initial_query_must_preserve_public_input" if route in {"summary_only", "masked"} else None)


@pytest.mark.parametrize("route", ["historical", "nested", "peer"])
def test_completion_binds_routed_source_without_replaying_its_tool_budget(route):
    row = evidence()
    source = UpstreamMessage("producer", "node", "Earlier interpretation, not evidence.",
        message_type="historical_evidence", tool_receipts=(receipt(SEARCH, row),), artifact_version="source-v1")
    node = short_request()
    if route == "peer":
        node = replace(node, peer_draft=source)
    elif route == "nested":
        node = replace(node, upstream=(replace(source, tool_receipts=(),
            input_artifact_provenance=(source.to_dict(),), artifact_version="source-v2"),))
    else:
        node = replace(node, upstream=(source,))
    executor = adapter(complete(artifact(row)))
    response = asyncio.run(executor.execute(node))
    assert json.loads(response.text)["status"] == "supported"
    assert response.metadata["tool_receipts"] == []
    assert node.prior_tool_receipts == () and node.action_history == ()
    assert len(executor._gateway.requests) == 1


@pytest.mark.parametrize("invalid", ["summary_only", "same_id_false_span", "metadata_only", "masked"])
def test_routed_binding_does_not_endorse_unretrieved_claims(invalid):
    row = evidence()
    claimed = artifact(row)
    source = UpstreamMessage("producer", "node", json.dumps(claimed),
        tool_receipts=(receipt(SEARCH, row),))
    node = short_request(upstream=(source,))
    if invalid == "summary_only":
        node = replace(node, upstream=(replace(source, tool_receipts=()),))
    elif invalid == "same_id_false_span":
        claimed["evidence_items"][0]["evidence_span"] = "An invented result absent from the real source."
        node = replace(node, upstream=(replace(source, content=json.dumps(claimed)),))
    elif invalid == "metadata_only":
        node = replace(node, upstream=(replace(source, tool_receipts=(receipt(SEARCH, {**row, "excerpt": ""}),)),))
    else:
        node = replace(node, communication_condition=CommunicationCondition.UPSTREAM_MASKED)
    executor = adapter()
    token = _COMPLETION_REQUEST.set(node)
    try:
        error = executor._structured_evidence_artifact_error(claimed, [])
    finally:
        _COMPLETION_REQUEST.reset(token)
    expected = "structured_evidence_item_span_not_in_receipt" if invalid in {"same_id_false_span", "metadata_only"} else "structured_evidence_item_receipt_binding_invalid"
    assert error == expected


def test_metadata_hits_remain_visible_without_consuming_content_evidence_slots():
    executor = adapter(require_relevant_evidence=True, max_successful_searches=1)
    found = observation(SEARCH, "velorin nebulark trial")
    found["result"]["evidence"][0].update({"title": "Velorin NEBULARK trial", "excerpt": ""})
    admitted, can_complete = executor._state_conditioned_action_domain(short_request(), [found])
    assert (SEARCH, "search") in admitted and can_complete
    assert executor._model_visible_observations([found])[0]["result"]["evidence"]
    found["result"]["evidence"][0]["excerpt"] = "The original source supplies text."
    admitted, can_complete = executor._state_conditioned_action_domain(short_request(), [found])
    assert (SEARCH, "search") not in admitted and can_complete
