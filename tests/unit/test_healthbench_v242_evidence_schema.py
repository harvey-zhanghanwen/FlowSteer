"""Synthetic receipt-derived schema constraints; no models, API or rubric data."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import CommunicationCondition, UpstreamMessage
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_evidence_adapter import (
    HealthBenchAuthoritativeReactExecutionAdapter, _COMPLETION_REQUEST,
)
from tests.unit.test_healthbench_clinical_react import (
    CALC, READ, SEARCH, SequenceGateway, action, artifact, complete, evidence,
    observation, receipt, registry, request,
)


def adapter(*outputs, enabled=True, authoritative=False, **settings):
    cls = HealthBenchAuthoritativeReactExecutionAdapter if authoritative else HealthBenchClinicalReactExecutionAdapter
    return cls(gateway=SequenceGateway(*outputs), tool_registry=registry(),
        max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True,
        constrain_evidence_metadata=enabled, **settings)


def source_observation(row, **changes):
    result = observation(SEARCH, "synthetic relationship")
    result["result"]["evidence"] = [row]
    result.update(changes)
    return result


def completion_valid(schema, value):
    return Draft202012Validator(schema).is_valid({"value": value})


def insufficient():
    return {"schema_version": "healthbench.structured-evidence.v1", "status": "insufficient",
        "summary": "The available sources do not establish the requested relationship.",
        "evidence_items": [], "uncertainties": ["The relationship remains unresolved."]}


@pytest.mark.parametrize("invalid", ["true", "false", 1, 0, None])
def test_metadata_constraint_is_strictly_typed(invalid):
    with pytest.raises(TypeError, match="constrain_evidence_metadata"):
        adapter(enabled=invalid)


def test_old_default_schema_and_full_assistant_response_are_unchanged():
    node = request()
    executor = adapter(enabled=False)
    assert executor._completion_arguments_schema_for_state(node, []) == executor._completion_arguments_schema(node)
    executor = adapter()
    for node in (request(output=True), request(CALC)):
        assert executor._completion_arguments_schema_for_state(node, [source_observation(evidence())]) == executor._completion_arguments_schema(node)


def test_metadata_is_one_observed_tuple_not_independent_field_enums():
    left = evidence("source-a")
    right = {**evidence("source-b"), "source": "Another exact source", "title": "Another exact title",
        "date": None, "url": None, "excerpt": "Another source describes a distinct relationship."}
    executor = adapter()
    observations = [source_observation(left), source_observation(right)]
    snapshot = deepcopy(observations)
    schema = executor._completion_arguments_schema_for_state(request(), observations)
    assert completion_valid(schema, artifact(left))
    assert completion_valid(schema, artifact(right))
    for field in ("document_id", "source", "title", "date", "url"):
        mixed = artifact(left)
        mixed["evidence_items"][0][field] = right[field]
        assert not completion_valid(schema, mixed), field
    assert observations == snapshot
    assert "anyOf" not in executor._completion_arguments_schema(request())["properties"]["value"]["properties"]["evidence_items"]["items"]


def test_every_real_metadata_variant_remains_available_and_duplicates_are_removed():
    sources = [{**evidence(str(index)), "title": f"Observed source {index}"} for index in range(17)]
    sources.append({**sources[0], "source": "Alternate observed repository"})
    observed = [source_observation(source) for source in sources + sources]
    schema = adapter()._completion_arguments_schema_for_state(request(), observed)
    choices = schema["properties"]["value"]["properties"]["evidence_items"]["items"]["anyOf"]
    assert len(choices) == 18
    assert all(completion_valid(schema, artifact(source)) for source in sources)


@pytest.mark.parametrize("route", ["own_receipt", "upstream", "peer", "nested", "masked", "summary_only"])
def test_only_real_visible_receipts_constrain_sources(route):
    source = evidence()
    message = UpstreamMessage("producer", "node", json.dumps(artifact(source)),
        artifact_version="a1", tool_receipts=(receipt(SEARCH, source),))
    node = request()
    if route == "own_receipt":
        node = replace(node, prior_tool_receipts=(receipt(SEARCH, source),))
    elif route == "peer":
        node = replace(node, peer_draft=message)
    elif route == "nested":
        node = replace(node, upstream=(replace(message, tool_receipts=(),
            artifact_version="a2", input_artifact_provenance=(message.to_dict(),)),))
    else:
        node = replace(node, upstream=(message,))
        if route == "masked":
            node = replace(node, communication_condition=CommunicationCondition.UPSTREAM_MASKED)
        elif route == "summary_only":
            node = replace(node, upstream=(replace(message, tool_receipts=()),))
    before = (node.prior_tool_receipts, node.action_history)
    schema = adapter()._completion_arguments_schema_for_state(node, [])
    assert completion_valid(schema, artifact(source)) is (route not in {"masked", "summary_only"})
    assert completion_valid(schema, insufficient())
    assert (node.prior_tool_receipts, node.action_history) == before


@pytest.mark.parametrize("state", ["no_sources", "metadata_only", "failed_tool", "incomplete_tool", "invalid_action"])
def test_no_read_source_keeps_explicit_insufficient_completion_available(state):
    source = evidence()
    observations = []
    if state == "metadata_only":
        observations = [source_observation({**source, "excerpt": "", "source_id": "pubmed:123456"})]
    elif state == "failed_tool":
        observations = [source_observation(source, observation_status="tool_error")]
    elif state == "incomplete_tool":
        observations = [source_observation(source, completed=False)]
    elif state == "invalid_action":
        observations = [source_observation(source, observation_status="schema_invalid")]
    executor = adapter()
    schema = executor._completion_arguments_schema_for_state(request(), observations)
    assert completion_valid(schema, insufficient())
    assert not completion_valid(schema, artifact(source))
    assert not completion_valid(schema, {**insufficient(), "status": "supported"})
    assert not completion_valid(schema, {**insufficient(), "uncertainties": []})
    assert executor._state_conditioned_action_domain(request(), observations)[1] is True


def test_schema_does_not_replace_strict_span_validation_or_write_claims():
    source = evidence()
    executor = adapter()
    node = request()
    schema = executor._completion_arguments_schema_for_state(node, [source_observation(source)])
    submitted = artifact(source)
    submitted["evidence_items"][0]["evidence_span"] = "The source ... a conclusion never observed."
    submitted["evidence_items"][0]["supported_claim"] = "A claim authored by the model, not the schema."
    before = deepcopy(submitted)
    assert completion_valid(schema, submitted)  # Grammar constrains copying, not evidence entailment.
    token = _COMPLETION_REQUEST.set(node)
    try:
        error = executor._structured_evidence_artifact_error(submitted, [receipt(SEARCH, source)])
    finally:
        _COMPLETION_REQUEST.reset(token)
    assert error == "structured_evidence_item_span_not_in_receipt"
    assert submitted == before
    items = schema["properties"]["value"]["properties"]["evidence_items"]["items"]
    old_items = executor._completion_arguments_schema(node)["properties"]["value"]["properties"]["evidence_items"]["items"]
    assert items["properties"]["evidence_span"] == old_items["properties"]["evidence_span"]
    assert items["properties"]["supported_claim"] == old_items["properties"]["supported_claim"]


def test_reading_metadata_only_hit_makes_its_actual_metadata_available():
    source = {**evidence(), "source_id": "pubmed:123456"}
    metadata = source_observation({**source, "excerpt": ""})
    reading = {"observation_status": "success", "completed": True,
        "executed_action": action(READ, "pubmed:123456"),
        "result": {"operation": "read_source", "evidence": [source]}}
    executor = adapter()
    assert not completion_valid(executor._completion_arguments_schema_for_state(request(), [metadata]), artifact(source))
    assert completion_valid(executor._completion_arguments_schema_for_state(request(), [metadata, reading]), artifact(source))


def test_authoritative_provider_schema_uses_observed_metadata_without_changing_action_domain():
    source = evidence()
    executor = adapter(action(SEARCH, "synthetic relationship"), complete(artifact(source)), authoritative=True)
    response = asyncio.run(executor.execute(request(SEARCH)))
    assert json.loads(response.text) == artifact(source)
    assert response.metadata["tool_calls"] == 1
    schema = json.loads(executor._gateway.requests[1].model.metadata["response_json_schema"])
    assert set(schema) == {"oneOf"}
    assert len(schema["oneOf"]) == 2
    validator = Draft202012Validator(schema)
    assert validator.is_valid(action(SEARCH, "another public relationship"))
    assert validator.is_valid(complete(artifact(source)))
    bad = artifact(source)
    bad["evidence_items"][0]["title"] = "Decorated invented title"
    assert not validator.is_valid(complete(bad))
