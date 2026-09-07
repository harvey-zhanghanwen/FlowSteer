"""Synthetic status/quote protocol regressions; CPU grammar, no live services."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import CommunicationCondition, UpstreamMessage
from tests.unit.test_healthbench_clinical_react import (
    SEARCH, artifact, complete, evidence, receipt, request,
)
from tests.unit.test_healthbench_v241_evidence_repair import (
    adapter as repair_adapter, check,
)
from tests.unit.test_healthbench_v242_evidence_schema import (
    adapter, insufficient, source_observation,
)
from tests.unit.test_healthbench_v247_evidence_schema import (
    compiler, compile_schema, completion_schema, grammar_accepts,
)


def state_schema(*rows):
    executor = adapter()
    node = request()
    return executor, executor._completion_arguments_schema_for_state(
        node, [source_observation(row) for row in rows],
    )


def test_status_branches_preserve_complete_root_and_nested_item_access():
    row = evidence()
    executor, schema = state_schema(row)
    root = schema["properties"]["value"]
    base = executor._completion_arguments_schema(request())["properties"]["value"]
    assert root["properties"]["evidence_items"]["items"]["properties"] == (
        base["properties"]["evidence_items"]["items"]["properties"]
    )
    assert len(root["properties"]["evidence_items"]["items"]["anyOf"]) == 1
    assert "anyOf" not in base
    for branch, status, collection in zip(
        root["anyOf"], ("supported", "insufficient"),
        ("evidence_items", "uncertainties"), strict=True,
    ):
        expected = deepcopy({key: value for key, value in root.items() if key != "anyOf"})
        expected["properties"]["status"] = {"const": status}
        expected["properties"][collection]["minItems"] = 1
        assert branch == expected
        assert branch["type"] == "object" and branch["additionalProperties"] is False
        assert branch["required"] == base["required"]


@pytest.mark.parametrize("status,has_evidence,has_uncertainty,expected", [
    ("supported", True, False, True),
    ("supported", True, True, True),
    ("supported", False, False, False),
    ("supported", False, True, False),
    ("insufficient", False, True, True),
    ("insufficient", True, True, True),
    ("insufficient", False, False, False),
    ("insufficient", True, False, False),
])
def test_status_cardinality_matches_validator_and_cpu_grammar(
    compiler, status, has_evidence, has_uncertainty, expected,
):
    row = evidence()
    executor, schema = state_schema(row)
    value = artifact(row)
    value.update(status=status,
        evidence_items=value["evidence_items"] if has_evidence else [],
        uncertainties=["An unresolved synthetic limitation."] if has_uncertainty else [])
    assert Draft202012Validator(schema).is_valid({"value": value}) is expected
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    assert grammar_accepts(compiler, grammar, complete(value)) is expected
    error = executor._structured_evidence_artifact_error(value, [receipt(SEARCH, row)])
    assert (error is None) is expected
    if not expected:
        assert error == ("structured_evidence_artifact_requires_evidence" if status == "supported"
                         else "structured_evidence_artifact_requires_uncertainty")


@pytest.mark.parametrize("missing", [
    "schema_version", "status", "summary", "evidence_items", "uncertainties",
])
def test_cpu_status_union_does_not_drop_required_siblings(compiler, missing):
    row = evidence()
    executor, schema = state_schema(row)
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    for value in (artifact(row), insufficient()):
        del value[missing]
        assert not Draft202012Validator(schema).is_valid({"value": value})
        assert not grammar_accepts(compiler, grammar, complete(value))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "other.protocol"), ("status", "unknown"),
    ("summary", 7), ("extra", "not permitted"),
])
def test_cpu_status_union_preserves_root_property_constraints(compiler, field, value):
    row = evidence()
    executor, schema = state_schema(row)
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    for submitted in (artifact(row), insufficient()):
        submitted[field] = value
        assert not grammar_accepts(compiler, grammar, complete(submitted))


@pytest.mark.parametrize("metadata_only", [False, True])
def test_no_excerpt_keeps_only_explicit_insufficient_cpu_branch(compiler, metadata_only):
    rows = [{**evidence(), "excerpt": ""}] if metadata_only else []
    executor, schema = state_schema(*rows)
    root = schema["properties"]["value"]
    assert len(root["anyOf"]) == 1
    assert root["anyOf"][0]["properties"]["status"] == {"const": "insufficient"}
    grammar = compile_schema(compiler, completion_schema(executor, schema))
    assert grammar_accepts(compiler, grammar, complete(insufficient()))
    assert not grammar_accepts(compiler, grammar, complete({**insufficient(), "uncertainties": []}))
    assert not grammar_accepts(compiler, grammar, complete({**insufficient(), "status": "supported"}))
    assert not grammar_accepts(compiler, grammar, complete(artifact(evidence())))


def test_disabled_schema_and_output_text_are_unchanged():
    row = evidence()
    for executor, node in ((adapter(enabled=False), request()), (adapter(), request(output=True))):
        assert executor._completion_arguments_schema_for_state(node, [source_observation(row)]) == (
            executor._completion_arguments_schema(node)
        )


@pytest.mark.parametrize("actual,submitted", [
    ("The synthetic modules remain co-operating during the entire observation.",
     "The synthetic modules remain cooperating during the entire observation."),
    ("The synthetic modules remain connected. PAGE 42 HEADER Their labels stay unchanged.",
     "The synthetic modules remain connected. Their labels stay unchanged."),
    ("The synthetic modules retain their labels throughout the observation.",
     "The synthetic modules retain all their labels throughout the observation."),
    ("The synthetic modules retain their labels throughout the observation.",
     "The synthetic modules retain their labls throughout the observation."),
    ("The synthetic labels read ‘naïve’—the spelling remains unchanged.",
     "The synthetic labels read 'naïve'-the spelling remains unchanged."),
    ("The synthetic modules remain connected. Their labels stay unchanged.",
     "The synthetic modules remain connected ... Their labels stay unchanged."),
])
def test_late_quote_mismatch_gets_raw_local_window_without_acceptance(actual, submitted):
    excerpt = "Unrelated opening context. " * 35 + actual + " Closing context." * 30
    row = {**evidence(), "excerpt": excerpt, "source_id": "fixture-raw-page",
           "truncated": True, "next_offset": 7200}
    value = artifact(row)
    value["evidence_items"][0]["evidence_span"] = submitted
    value["evidence_items"][0]["supported_claim"] = "MODEL_AUTHORED_CLAIM_NOT_A_REPAIR"
    before = deepcopy(value)
    executor = repair_adapter()
    error, feedback = check(executor, request(), value, [receipt(SEARCH, row)])
    assert error == "structured_evidence_item_span_not_in_receipt"
    candidate = feedback["repair_context"]["receipt_candidates"][0]
    start, end = candidate["preview_start_offset"], candidate["preview_end_offset"]
    assert start > 400 and end - start <= 400
    assert candidate["verbatim_excerpt_preview"] == excerpt[start:end]
    assert actual in candidate["verbatim_excerpt_preview"]
    assert candidate["preview_offset_basis"] == "receipt_excerpt_characters_0_based_end_exclusive"
    assert candidate["preview_is_complete_excerpt"] is False
    assert start <= candidate["diagnostic_alignment"]["receipt_excerpt_offset"] <= end
    assert candidate["next_offset"] == 7200  # Source pagination is not a preview character index.
    assert "MODEL_AUTHORED_CLAIM_NOT_A_REPAIR" not in json.dumps(feedback)
    assert value == before
    assert executor._structured_evidence_artifact_error(value, [receipt(SEARCH, row)]) == error
    exact = deepcopy(value)
    exact["evidence_items"][0]["evidence_span"] = actual
    assert check(executor, request(), exact, [receipt(SEARCH, row)]) == (None, {})


def test_no_shared_text_falls_back_to_bounded_raw_prefix():
    row = {**evidence(), "excerpt": "lowercase actual receipt text. " * 40}
    value = artifact(row)
    value["evidence_items"][0]["evidence_span"] = "ZZZZZZZZZZ"
    error, feedback = check(repair_adapter(), request(), value, [receipt(SEARCH, row)])
    assert error == "structured_evidence_item_span_not_in_receipt"
    candidate = feedback["repair_context"]["receipt_candidates"][0]
    assert candidate["verbatim_excerpt_preview"] == row["excerpt"][:400]
    assert candidate["preview_start_offset"] == 0
    assert "diagnostic_alignment" not in candidate


def test_local_window_uses_only_routed_receipts_and_keeps_candidate_bound():
    actual = "Synthetic components remain co-operating while their labels stay unchanged."
    row = {**evidence(), "excerpt": "Opening context. " * 70 + actual}
    hidden = {**row, "excerpt": "UNROUTED_PRIVATE_TEXT " + actual}
    upstream = UpstreamMessage("producer", "node", "An interpretation.",
        tool_receipts=(receipt(SEARCH, hidden),))
    node = replace(request(), upstream=(upstream,),
        communication_condition=CommunicationCondition.UPSTREAM_MASKED)
    value = artifact(row)
    value["evidence_items"][0]["evidence_span"] = actual.replace("co-operating", "cooperating")
    receipts = [receipt(SEARCH, {**row, "excerpt": row["excerpt"] + f" Variant {i}."}) for i in range(3)]
    error, feedback = check(repair_adapter(), node, value, receipts)
    assert error == "structured_evidence_item_span_not_in_receipt"
    candidates = feedback["repair_context"]["receipt_candidates"]
    assert len(candidates) == 2
    assert all(len(candidate["verbatim_excerpt_preview"]) <= 400 for candidate in candidates)
    assert "UNROUTED_PRIVATE_TEXT" not in json.dumps(feedback)
    _, hidden_only = check(repair_adapter(), node, value, [])
    assert hidden_only["repair_context"]["observed_document_ids"] == []
