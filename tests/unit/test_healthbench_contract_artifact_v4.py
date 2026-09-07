"""Synthetic contract/receipt compatibility checks; no benchmark answers or API."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import yaml

from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_CONTRACT_ARTIFACT_V4 as V4,
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3 as V3,
    CommunicationCondition,
)
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.openai_gateway import build_agent_messages, _format_upstream
from src.interactive.tool_runtime import StructuredAction
from tests.unit.test_healthbench_clinical_react import (
    SequenceGateway, registry, request, complete, artifact, evidence, receipt,
    SEARCH, READ, CALC,
)
from tests.unit.test_healthbench_evidence_projection_v3 import (
    request as downstream_request, upstream,
)


def adapter(*values):
    return HealthBenchClinicalReactExecutionAdapter(
        gateway=SequenceGateway(*values), tool_registry=registry(),
        max_turns=6, max_tool_calls=3, require_initial_search=False,
        require_structured_evidence_artifact=True,
        require_complete_natural_language_artifact=True,
        max_completion_artifact_characters=12000,
    )


def project(message, profile=V4):
    text = build_agent_messages(downstream_request(message, profile=profile))[-1]["content"]
    return json.loads(text.split("[Upstream artifact]\n", 1)[1].split("\n\nProduce", 1)[0])


def test_completion_union_preserves_legacy_output_and_receipt_schema():
    runtime = adapter()
    public_text = "A completed contract artifact. " * 340
    req = request(SEARCH, artifact_communication_profile=V4)
    schema = runtime._completion_arguments_schema(req)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"value": public_text})
    assert not validator.is_valid({"value": "x" * 12001})
    assert not validator.is_valid({"value": ""})
    assert validator.is_valid({"value": artifact(evidence())})
    # The evidence branch still binds every claim to observed source receipts.
    value = artifact(evidence())
    args = StructuredAction.from_value(complete(value))
    assert runtime._completion_error(action=args, artifact=json.dumps(value), tool_receipts=[receipt(SEARCH)]) is None
    assert runtime._completion_error(action=args, artifact=json.dumps(value), tool_receipts=[]) is not None
    legacy = runtime._completion_arguments_schema(replace(req, artifact_communication_profile=V3))
    assert not Draft202012Validator(legacy).is_valid({"value": public_text})
    for profile in (V3, V4):
        output = runtime._completion_arguments_schema(replace(req, is_output_agent=True, artifact_communication_profile=profile))
        assert output["properties"]["value"]["type"] == "string"


def test_clinical_read_and_calculation_keep_their_existing_schema_boundaries():
    runtime = adapter()
    read_schema = runtime._completion_arguments_schema(request(READ, artifact_communication_profile=V4))
    fields = read_schema["properties"]["value"]["anyOf"][1]["properties"]["evidence_items"]["items"]["properties"]
    assert "successful retrieval" in fields["document_id"]["description"]
    calculation = runtime._completion_arguments_schema(request(CALC, artifact_communication_profile=V4))
    assert calculation["properties"]["value"]["type"] == "string"


def test_long_contract_text_completes_once_and_is_delivered_to_downstream_intact():
    # The decisive middle marker is outside the old 3600-character projection.
    text = "Opening. " + "αβγ complete paragraph. " * 230 + " MIDDLE_DELIVERABLE " + "Second complete paragraph. " * 180 + " End."
    assert 9000 < len(text) < 12000
    runtime = adapter(complete(text))
    response = asyncio.run(runtime.execute(request(SEARCH, artifact_communication_profile=V4)))
    assert response.text == text
    assert len(response.metadata["react_trace"]) == 1
    assert response.metadata["react_trace"][0]["observation_status"] == "completed"
    assert response.metadata["tool_calls"] == 0
    message = replace(upstream(content=response.text), tool_receipts=())
    projected = project(message)
    assert projected["producer_artifact"] == text
    assert "MIDDLE_DELIVERABLE" not in project(message, V3)["producer_artifact"]
    assert project(message, V3)["producer_artifact"] != text


def test_plain_text_does_not_erase_independent_evidence_or_duplicate_messages():
    row = evidence()
    message = upstream(row, content="The completed task product, with the source's applicability stated.")
    projected = project(message)
    assert projected["producer_artifact"] == message.artifact
    assert projected["retrieval_evidence"]["evidence_receipts"][0]["excerpts"] == [row["excerpt"]]
    text = build_agent_messages(downstream_request(message, message, profile=V4))[-1]["content"]
    assert text.count("[Upstream artifact]") == 1


def test_masked_condition_and_shared_budget_remain_effective():
    message = upstream(evidence(), content="DO_NOT_REVEAL_SYNTHETIC_PAYLOAD")
    masked = build_agent_messages(downstream_request(
        message, profile=V4, communication_condition=CommunicationCondition.UPSTREAM_MASKED,
    ))[-1]["content"]
    assert "DO_NOT_REVEAL_SYNTHETIC_PAYLOAD" not in masked
    state = {"remaining": 900, "sources": set(), "envelopes": set()}
    text = _format_upstream(
        [upstream(content="A complete paragraph. " * 600)], CommunicationCondition.NORMAL,
        project_healthbench_structured_evidence=True, artifact_communication_profile=V4,
        healthbench_projection_state=state,
    )
    assert len(text) <= 900
    assert "projection truncated" in text


def test_single_rerun_config_wires_graph_profile_without_changing_direct_or_budgets():
    root = Path(__file__).resolve().parents[2]
    old = yaml.safe_load((root / "config/evaluation_healthbench_deadline_recovery_single.yaml").read_text())
    new = yaml.safe_load((root / "config/evaluation_healthbench_contract_artifact_single.yaml").read_text())
    assert new["agent_graph"]["artifact_communication_profile"] == V4
    assert new["healthbench_professional_evaluation"] == old["healthbench_professional_evaluation"]
    assert new["director"] == old["director"]
    assert new["candidate_skill_evaluation"] == old["candidate_skill_evaluation"]
    assert new["evaluation"] == old["evaluation"]
    assert new["experiment"]["training_enabled"] is False
