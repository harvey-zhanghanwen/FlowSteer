"""Existing ReAct -> source receipts -> FTS5 -> routed evidence, without HTTP/LLMs."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import UpstreamMessage
from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.healthbench_clinical_tools import (
    HEALTHBENCH_LITERATURE_SEARCH_TOOL_ID as LITERATURE,
    HEALTHBENCH_TRIAL_SEARCH_TOOL_ID as TRIALS,
    HEALTHBENCH_SOURCE_READ_TOOL_ID as READ,
    _source_page, build_healthbench_clinical_tool_registry,
)
from src.interactive.healthbench_knowledge_tools import (
    HEALTHBENCH_KNOWLEDGE_TOOL_ID as KNOWLEDGE,
    HealthBenchKnowledgeReactExecutionAdapter, build_healthbench_knowledge_tool_registry,
)
from src.interactive.openai_gateway import _healthbench_search_candidates, _healthbench_v3_receipts
from src.interactive.tool_runtime import StructuredAction, ToolRequest
from tests.unit.test_healthbench_clinical_react import SequenceGateway, request, complete, artifact, evidence
from tests.unit.test_healthbench_clinical_tools import fixture_corpus
from tests.unit.test_healthbench_optional_tools_wiring import SMOKE, optional_config, registry as model_registry, _task, _settings


class SourceClient:
    def __init__(self, source, source_id):
        self.calls = []
        self.row = {
            **evidence("synthetic-source"), "source": source, "source_id": source_id,
            "content_type": "synthetic_public_source", "version": None,
            "pmid": "12345", "pmcid": "PMC12345", "doi": "10.fixture/example",
            "publication_types": ["Journal Article"], "is_open_access": True,
            "is_preprint": False, "full_text_source_id": "pmc:PMC12345",
        }

    def search(self, query):
        self.calls.append(("search", query))
        return {"operation": "search", "query": query, "evidence": [_source_page(self.row)],
                "source_receipts": [{"source_type": self.row["source_type"], "source": self.row["source"],
                                     "status": "success", "result_count": 1, "error_type": None}]}

    def read(self, source_id):
        self.calls.append(("read", source_id))
        return self.row


def registry(enabled=True):
    literature = SourceClient("Europe PMC", "europepmc:MED:12345")
    trials = SourceClient("ClinicalTrials.gov", "clinicaltrials:NCT12345678")
    original = build_healthbench_clinical_tool_registry(fixture_corpus(),
        external_medical_sources_enabled=enabled, literature_client=literature, trial_client=trials)
    return build_healthbench_knowledge_tool_registry(original), literature, trials


def action(tool, **arguments):
    return {"kind": "tool", "name": "read_source" if tool == READ else "search",
            "resource_id": tool, "skill_id": None, "arguments": arguments}


@pytest.mark.parametrize("tool", [LITERATURE, TRIALS])
def test_external_search_registered_and_receipt_admitted(tool):
    tools, literature, trials = registry()
    assert len(tools.resource_ids) == 8
    source = literature if tool == LITERATURE else trials
    result = asyncio.run(tools.ainvoke(tool, ToolRequest("search", {"query": "relation"})))
    Draft202012Validator(tools.require_capability(tool).output_schema).validate(result.value)
    receipts = [{"tool_id": tool, "error_type": None,
                 "request": {"action": "search", "arguments": {"query": "relation"}},
                 "result": result.to_value()}]
    assert _healthbench_search_candidates(receipts)[0][2]["source"] == source.row["source"]
    assert source.calls == [("search", "relation")]


def test_old_toolset_is_unchanged_and_cannot_access_new_sources():
    tools, literature, trials = registry(False)
    assert len(tools.resource_ids) == 6
    assert LITERATURE not in tools.resource_ids and TRIALS not in tools.resource_ids
    assert not literature.calls and not trials.calls


@pytest.mark.parametrize("tool", [LITERATURE, TRIALS])
def test_existing_source_read_dispatch_and_explicit_pagination(tool):
    tools, literature, trials = registry()
    client = literature if tool == LITERATURE else trials
    client.row["excerpt"] = "A" * 16005
    result = asyncio.run(tools.ainvoke(READ, ToolRequest("read_source", {"source_id": client.row["source_id"]})))
    Draft202012Validator(tools.require_capability(READ).output_schema).validate(result.value)
    first = result.value["evidence"][0]
    assert first["truncated"] and first["next_offset"] == 16000
    second = asyncio.run(tools.ainvoke(READ, ToolRequest("read_source", {"source_id": client.row["source_id"], "offset": 16000}))).value["evidence"][0]
    assert first["excerpt"] + second["excerpt"] == client.row["excerpt"]


@pytest.mark.parametrize("tool", [LITERATURE, TRIALS])
def test_search_index_query_complete_and_downstream_communication(tmp_path, tool):
    tools, literature, trials = registry()
    source = literature if tool == LITERATURE else trials
    gateway = SequenceGateway(action(tool, query="relation"),
        action(KNOWLEDGE, database="medical_references", query="relation"), complete(artifact(source.row)))
    adapter = HealthBenchKnowledgeReactExecutionAdapter(gateway=gateway, tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True)
    response = asyncio.run(adapter.execute(request(tool, KNOWLEDGE)))
    assert len(gateway.requests) == 3
    assert response.metadata["knowledge_index"]["counts"]["medical_references"] == 1
    receipts = response.metadata["tool_receipts"]
    retrieved = _healthbench_search_candidates(receipts)
    assert retrieved[-1][2]["source"] == source.row["source"]
    assert retrieved[-1][2]["publication_types"] == ["Journal Article"]
    assert retrieved[-1][2]["full_text_source_id"] == "pmc:PMC12345"
    message = UpstreamMessage("producer", "node", response.text, tool_receipts=tuple(receipts))
    projection = _healthbench_v3_receipts(message, seen_sources=set(), char_budget=14000)
    assert projection["evidence_receipts"][0]["source_retrieval"][0]["full_text_source_id"] == "pmc:PMC12345"
    downstream = SequenceGateway(action(KNOWLEDGE, database="medical_references", query="relation"),
                                 complete(artifact(source.row)))
    adapter = HealthBenchKnowledgeReactExecutionAdapter(gateway=downstream, tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True)
    next_response = asyncio.run(adapter.execute(request(KNOWLEDGE, upstream=(message,))))
    assert next_response.metadata["knowledge_index"]["counts"]["medical_references"] == 1
    assert source.calls == [("search", "relation")]


@pytest.mark.parametrize("tool", [LITERATURE, TRIALS])
def test_new_tools_share_react_query_and_budget_boundaries(tmp_path, tool):
    tools, _, _ = registry()
    adapter = HealthBenchKnowledgeReactExecutionAdapter(gateway=SequenceGateway(), tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3, require_task_query_anchor=True)
    sampled = action(tool, query="relation")
    req = request(tool)
    assert adapter._tool_action_error(request=req, action=StructuredAction.from_value(sampled), observations=[]) is None
    bad = action(tool, query="unrelated astronomy")
    assert adapter._tool_action_error(request=req, action=StructuredAction.from_value(bad), observations=[]) == "query_does_not_preserve_public_task_anchor"
    observations = [{"observation_status": "success", "executed_action": sampled,
                     "result": {"operation": "search", "evidence": []}}]
    assert adapter._tool_action_error(request=req, action=StructuredAction.from_value(sampled), observations=observations) == "duplicate_tool_request"
    admitted, completion = adapter._state_conditioned_action_domain(req, observations * 3)
    assert not admitted and completion


def test_external_flag_reaches_the_existing_runtime_factory(tmp_path):
    config = optional_config()
    section = config["healthbench_tool_runtime"]
    section.update(toolset="source_separated_clinical_v1", knowledge_root="artifacts/indexes",
                   skillflow_source="/upstream/src", external_medical_sources_enabled=True)
    assert _settings(config)["external_medical_sources_enabled"] is True
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config, backend.registry, backend.project_root = config, model_registry(), tmp_path
    backend.runtime = SimpleNamespace(gateway=object(), timeout_seconds=30,
        artifact_communication_profile="producer_context_structured_evidence_v4")
    opened = SimpleNamespace(registry=object(), close=Mock())
    with (patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
          patch.object(SMOKE, "open_healthbench_knowledge_tool_registry", return_value=opened) as op,
          patch.object(SMOKE, "HealthBenchKnowledgeReactExecutionAdapter", return_value=SimpleNamespace(execute=Mock()))):
        backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert op.call_args.kwargs["external_medical_sources_enabled"] is True
    section["external_medical_sources_enabled"] = "true"
    with pytest.raises(ConfigurationError):
        _settings(config)


def test_v238_changes_only_tool_condition_and_keeps_comparable_future_arms():
    from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
    root = Path(__file__).resolve().parents[2]
    old = load_yaml(root / "config/evaluation_healthbench_professional_grounded_retrieval_v2_37_dev5.yaml")
    new = load_yaml(root / "config/evaluation_healthbench_professional_external_medical_sources_v2_38_dev5.yaml")
    validate_completion_benchmark_config(new)
    assert new["evaluation"] == old["evaluation"]
    assert new["experiment"]["prompt_version"] == old["experiment"]["prompt_version"]
    graph = dict(new["agent_graph"])
    graph["model_catalog_path"] = old["agent_graph"]["model_catalog_path"]
    assert graph == old["agent_graph"]
    section = new["healthbench_professional_evaluation"]
    assert section["task_ids"] == old["healthbench_professional_evaluation"]["task_ids"]
    assert section["sample_count"] == 5
    assert section["direct_allowed_tools"] == new["healthbench_tool_runtime"]["execution_profile_allowlist"][-1]["allowed_tools"]
    assert len(section["direct_allowed_tools"]) == 8
    catalog = load_yaml(root / new["agent_graph"]["model_catalog_path"])
    for model in catalog["models"]:
        if model["metadata"]["tool_capable"] == "true":
            assert {LITERATURE, TRIALS} <= set(model["metadata"]["tool_capability_scope"].split(","))
    assert all(not new[name]["enabled"] for name in ("grpo", "policy_sync", "exploration", "skills"))
