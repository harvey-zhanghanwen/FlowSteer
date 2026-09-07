"""Synthetic quality-boundary checks; no model, Tool, HTTP or task data."""

from copy import deepcopy
import asyncio
import json

import pytest

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    ARTIFACT_QUALITY_NONE,
    ARTIFACT_QUALITY_PUBLIC_TEXT_V1,
    ARTIFACT_QUALITY_PUBLIC_TEXT_V2,
    AgentRequest, AgentResponse, AgentRuntime, AgentRuntimeError, ExecutionPhase,
    _public_text_quality_receipt,
)
from src.interactive.healthbench_evidence_adapter import HealthBenchAuthoritativeReactExecutionAdapter
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from tests.unit.test_healthbench_clinical_react import registry as synthetic_registry


PROFILES = (ARTIFACT_QUALITY_PUBLIC_TEXT_V1, ARTIFACT_QUALITY_PUBLIC_TEXT_V2)
REPEATED = " ".join(f"word{index}" for index in range(24))


def evidence_artifact():
    sentences = (
        "Amber samples retained their original shape during the overnight observation period.",
        "Blue panels absorbed light evenly across the freshly painted surface under inspection.",
        "Copper fittings remained secure after several gentle adjustments by the workshop staff.",
        "Diamond patterns appeared clearly along the outer edge of the finished fabric.",
        "Emerald leaves developed gradually as spring sunlight reached the sheltered garden bed.",
        "Flax fibers separated cleanly when the operator loosened the surrounding wooden frame.",
    )
    return {
        "schema_version": "healthbench.structured-evidence.v1",
        "status": "supported",
        "summary": "Six separate observations document distinct material properties in the synthetic collection.",
        "evidence_items": [
            {
                "supported_claim": sentence,
                "conditions_or_qualifiers": f"Observation {index} concerns its individually named specimen.",
                "document_id": f"fixture-{index}",
                "source": "Synthetic source",
                "title": "Shared synthetic handbook",
                "date": None,
                "url": None,
                "evidence_span": sentence,
            }
            for index, sentence in enumerate(sentences)
        ],
        "uncertainties": ["The collection does not establish performance outside the recorded conditions."],
    }


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def completion_metadata(value):
    return {
        "execution_mode": "react",
        "react_trace": [{
            "turn": 1,
            "observation_status": "completed",
            "structured_action": {
                "kind": "complete", "name": "complete", "resource_id": None,
                "skill_id": None, "arguments": {"value": deepcopy(value)},
            },
        }],
    }


class NoCalls:
    async def generate(self, request):
        raise AssertionError("This test must not make a model or Tool call")


def validator():
    return HealthBenchAuthoritativeReactExecutionAdapter(
        gateway=NoCalls(), tool_registry=synthetic_registry(),
        max_turns=1, max_tool_calls=3, require_structured_evidence_artifact=True,
    )


def validate_fixture(value):
    rows = [
        {**{key: item[key] for key in ("document_id", "source", "title", "date", "url")},
         "excerpt": item["evidence_span"]}
        for item in value["evidence_items"]
    ]
    receipts = [{
        "tool_id": "healthbench-authoritative.search", "error_type": None,
        "result": {"completed": True, "value": {"operation": "search", "evidence": rows}},
    }]
    assert validator()._structured_evidence_artifact_error(value, receipts) is None


@pytest.mark.parametrize("profile", PROFILES)
def test_distinct_evidence_with_shared_metadata_is_not_repetition(profile):
    value = evidence_artifact()
    validate_fixture(value)
    text, metadata = serialized(value), completion_metadata(value)
    old = _public_text_quality_receipt(text, metadata, profile=profile)
    result = _public_text_quality_receipt(
        text, metadata, profile=profile, validated_structured_evidence=True,
    )
    assert "repeated_token_ngram" in old["error_codes"]
    assert result["status"] == "valid"
    assert result["maximum_24_token_ngram_repetitions"] < 3
    assert result["structured_json"] is True
    assert set(result) == set(old)
    for key in ("profile", "character_count", "token_count", "longest_line_characters", "sentence_boundary_count"):
        assert result[key] == old[key]
    assert text == serialized(value) and metadata == completion_metadata(value)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("field", ("summary", "supported_claim", "conditions_or_qualifiers", "evidence_span", "uncertainties"))
def test_real_repetition_in_every_protocol_prose_field_remains_rejected(profile, field):
    value = evidence_artifact()
    if field == "summary":
        value[field] = (REPEATED + " ") * 9
    elif field == "uncertainties":
        value[field] = [REPEATED] * 3
    else:
        for item in value["evidence_items"]:
            item[field] = REPEATED
    validate_fixture(value)
    result = _public_text_quality_receipt(
        serialized(value), completion_metadata(value), profile=profile,
        validated_structured_evidence=True,
    )
    assert "repeated_token_ngram" in result["error_codes"]
    assert result["maximum_24_token_ngram_repetitions"] >= 3


@pytest.mark.parametrize("profile", PROFILES)
def test_ngram_start_threshold_counts_prose_not_metadata(profile):
    value = evidence_artifact()
    value["summary"] = (REPEATED + " ") * 3
    value["evidence_items"] = [value["evidence_items"][0]]
    item = value["evidence_items"][0]
    item["supported_claim"] = "The observation concerns one sample."
    item["conditions_or_qualifiers"] = ""
    item["evidence_span"] = "One sample was recorded."
    item["title"] = " ".join(f"metadata{index}" for index in range(150))
    validate_fixture(value)
    result = _public_text_quality_receipt(
        serialized(value), completion_metadata(value), profile=profile,
        validated_structured_evidence=True,
    )
    assert result["token_count"] >= 192
    assert result["maximum_24_token_ngram_repetitions"] == 0
    assert result["status"] == "valid"


@pytest.mark.parametrize("mutation", ("malformed", "other_version", "extra_field", "missing_field", "uncompleted", "mismatched_value"))
def test_invalid_or_unbound_schema_tagged_json_does_not_get_projection(mutation):
    value = evidence_artifact()
    metadata = completion_metadata(value)
    if mutation == "other_version":
        value["schema_version"] = "other.schema.v1"
    elif mutation == "extra_field":
        value["extra"] = "not part of the evidence protocol"
    elif mutation == "missing_field":
        del value["evidence_items"][0]["supported_claim"]
    elif mutation == "uncompleted":
        metadata["react_trace"][-1]["observation_status"] = "schema_invalid"
    elif mutation == "mismatched_value":
        metadata["react_trace"][-1]["structured_action"]["arguments"]["value"]["summary"] = "Different artifact."
    if mutation in {"other_version", "extra_field", "missing_field"}:
        metadata = completion_metadata(value)
    text = serialized(value)
    if mutation == "malformed":
        text = text[:-1]
    expected = _public_text_quality_receipt(text, metadata, profile=PROFILES[1])
    actual = _public_text_quality_receipt(
        text, metadata, profile=PROFILES[1], validated_structured_evidence=True,
    )
    assert actual == expected
    assert "repeated_token_ngram" in actual["error_codes"]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("text", ((REPEATED + " ") * 10, " / ->" * 250, json.dumps({"rows": [REPEATED] * 10})))
def test_plain_text_garbage_and_other_json_keep_the_original_detection(profile, text):
    original = _public_text_quality_receipt(text, {}, profile=profile)
    actual = _public_text_quality_receipt(text, {}, profile=profile, validated_structured_evidence=True)
    assert actual == original
    assert "repeated_token_ngram" in actual["error_codes"]


@pytest.mark.parametrize("profile", (*PROFILES, ARTIFACT_QUALITY_NONE))
def test_runtime_uses_selected_validating_adapter_and_keeps_profile_none(profile):
    value = evidence_artifact()

    class ValidatedFixtureAdapter(HealthBenchAuthoritativeReactExecutionAdapter):
        async def execute(self, request):
            validate_fixture(value)
            return AgentResponse(serialized(value), completion_metadata(value))

    adapter = ValidatedFixtureAdapter(
        gateway=NoCalls(), tool_registry=synthetic_registry(), max_turns=1,
        max_tool_calls=3, require_structured_evidence_artifact=True,
    )
    provider, model = ProviderSpec("fixture-provider", kind="test"), ModelSpec("fixture-model", "fixture-provider")
    request = AgentRequest(
        "fixture:node", "fixture", 1, "Describe the synthetic observations.",
        AgentNode("node", model.model_id, "Collect evidence.", execution_mode="react",
                  allowed_tools=("healthbench-authoritative.search",)),
        model, provider, ExecutionPhase.SINGLE,
    )
    runtime = AgentRuntime(
        ModelRegistry([provider], [model]), NoCalls(), execution_adapters={"react": adapter},
        artifact_quality_profile=profile,
    )
    response = asyncio.run(runtime._invoke(request, [], []))
    assert response.text == serialized(value)
    if profile == ARTIFACT_QUALITY_NONE:
        assert "artifact_quality_receipt" not in response.metadata
    else:
        assert response.metadata["artifact_quality_receipt"]["status"] == "valid"
        # A completed-looking response cannot opt in through metadata alone.
        adapter._require_structured_evidence_artifact = False
        with pytest.raises(AgentRuntimeError) as caught:
            asyncio.run(runtime._invoke(request, [], []))
        assert "repeated_token_ngram" in caught.value.failure_records[0].metadata["artifact_quality_receipt"]["error_codes"]
