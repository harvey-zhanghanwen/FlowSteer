"""No-model regressions: reuse v2.38 Tool/ReAct/FTS5 and evidence wire fixtures."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import UpstreamMessage
from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.healthbench_clinical_tools import (
    HEALTHBENCH_BOOKSHELF_SEARCH_TOOL_ID as BOOKS,
    HEALTHBENCH_PDQ_SEARCH_TOOL_ID as PDQ,
    HEALTHBENCH_AHRQ_SEARCH_TOOL_ID as AHRQ,
    HEALTHBENCH_TERMINOLOGY_SEARCH_TOOL_ID as MESH,
    HEALTHBENCH_SOURCE_READ_TOOL_ID as READ,
    build_healthbench_clinical_tool_registry,
)
from src.interactive.healthbench_knowledge_tools import (
    HEALTHBENCH_KNOWLEDGE_TOOL_ID as KNOWLEDGE,
    HealthBenchKnowledgeReactExecutionAdapter,
    build_healthbench_knowledge_tool_registry,
)
from src.interactive.openai_gateway import _healthbench_search_candidates, _healthbench_v3_receipts
from src.interactive.tool_runtime import StructuredAction, ToolRequest
from tests.unit.test_healthbench_external_sources_wiring import SourceClient, action
from tests.unit.test_healthbench_clinical_react import SequenceGateway, request, complete, artifact
from tests.unit.test_healthbench_clinical_tools import fixture_corpus
from tests.unit.test_healthbench_optional_tools_wiring import (
    SMOKE, optional_config, registry as model_registry, _task, _settings,
)


TOOLS = (BOOKS, PDQ, AHRQ, MESH)


@pytest.mark.parametrize("tool", [BOOKS, MESH])
def test_real_client_projections_fit_registered_search_and_read_schema(tool):
    from src.interactive.healthbench_bookshelf import BookshelfClient
    from src.interactive.healthbench_mesh import MeSHClient
    from tests.unit.test_healthbench_bookshelf import FixtureOpener as BooksOpener
    from tests.unit.test_healthbench_mesh import FixtureOpener as MeSHOpener
    client = BookshelfClient(opener=BooksOpener()) if tool == BOOKS else MeSHClient(opener=MeSHOpener())
    tools = build_healthbench_clinical_tool_registry(fixture_corpus(),
        clinical_reference_sources_enabled=True,
        **({"bookshelf_client": client} if tool == BOOKS else {"terminology_client": client}))
    with patch("src.interactive.healthbench_bookshelf._wait_for_pubmed_request_slot"):
        search = asyncio.run(tools.ainvoke(tool, ToolRequest("search", {"query": "synthetic"}))).value
    Draft202012Validator(tools.require_capability(tool).output_schema).validate(search)
    read = asyncio.run(tools.ainvoke(READ, ToolRequest("read_source", {
        "source_id": search["evidence"][0]["source_id"],
    }))).value
    Draft202012Validator(tools.require_capability(READ).output_schema).validate(read)
    assert read["evidence"][0]["excerpt"]
    if tool == BOOKS:
        assert search["evidence"][0]["excerpt"] == ""
        assert read["evidence"][0]["collection"] == "pdq"
    else:
        assert read["evidence"][0]["source_type"] == "controlled_vocabulary"


def test_new_retrieval_feedback_preserves_scope_without_a_fixed_role(tmp_path):
    tools, _ = registry()
    adapter = HealthBenchKnowledgeReactExecutionAdapter(gateway=SequenceGateway(), tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3)
    text = adapter._contract(request(BOOKS, READ), [])
    assert "An empty search is specific to that query/source" in text
    assert "Read metadata-only hits" in text
    old = adapter._contract(request(READ), [])
    assert "An empty search is specific to that query/source" not in old


def registry(enabled=True):
    clients = {
        tool: SourceClient("NLM MeSH" if tool == MESH else "NCBI Bookshelf",
                           "mesh:D012345" if tool == MESH else "bookshelf:NBK12345")
        for tool in TOOLS
    }
    for tool, client in clients.items():
        client.row = {key: value for key, value in client.row.items() if key not in {
            "pmid", "pmcid", "doi", "publication_types", "is_open_access", "is_preprint", "full_text_source_id",
        }}
        client.row["source_type"] = "controlled_vocabulary" if tool == MESH else "biomedical_reference"
    tools = build_healthbench_clinical_tool_registry(fixture_corpus(),
        external_medical_sources_enabled=True, clinical_reference_sources_enabled=enabled,
        bookshelf_client=clients[BOOKS], pdq_client=clients[PDQ],
        ahrq_client=clients[AHRQ], terminology_client=clients[MESH])
    return build_healthbench_knowledge_tool_registry(tools), clients


@pytest.mark.parametrize("tool", TOOLS)
def test_new_reference_sources_use_existing_schema_and_receipt_projection(tool):
    tools, clients = registry()
    assert len(tools.resource_ids) == 12
    result = asyncio.run(tools.ainvoke(tool, ToolRequest("search", {"query": "relation"})))
    Draft202012Validator(tools.require_capability(tool).output_schema).validate(result.value)
    receipts = [{"tool_id": tool, "error_type": None,
                 "request": {"action": "search", "arguments": {"query": "relation"}},
                 "result": result.to_value()}]
    candidate = _healthbench_search_candidates(receipts)[0][2]
    assert candidate["source"] == clients[tool].row["source"]
    assert candidate["source_type"] == clients[tool].row["source_type"]
    assert clients[tool].calls == [("search", "relation")]


def test_v238_registry_unchanged_when_reference_sources_disabled():
    tools, clients = registry(False)
    assert len(tools.resource_ids) == 8
    assert not set(TOOLS).intersection(tools.resource_ids)
    assert all(not client.calls for client in clients.values())
    with pytest.raises(LookupError):
        asyncio.run(tools.ainvoke(READ, ToolRequest("read_source", {"source_id": "bookshelf:NBK12345"})))


@pytest.mark.parametrize("tool", [BOOKS, MESH])
def test_reference_read_reuses_explicit_source_pagination(tool):
    tools, clients = registry()
    client = clients[tool]
    client.row["excerpt"] = "X" * 16005
    first = asyncio.run(tools.ainvoke(READ, ToolRequest("read_source", {
        "source_id": client.row["source_id"],
    }))).value
    Draft202012Validator(tools.require_capability(READ).output_schema).validate(first)
    second = asyncio.run(tools.ainvoke(READ, ToolRequest("read_source", {
        "source_id": client.row["source_id"], "offset": first["evidence"][0]["next_offset"],
    }))).value
    assert first["evidence"][0]["excerpt"] + second["evidence"][0]["excerpt"] == client.row["excerpt"]


@pytest.mark.parametrize("tool", TOOLS)
def test_reference_search_fts5_and_downstream_keep_source_identity(tmp_path, tool):
    tools, clients = registry()
    client = clients[tool]
    gateway = SequenceGateway(action(tool, query="relation"),
        action(KNOWLEDGE, database="medical_references", query="relation"),
        complete(artifact(client.row)))
    adapter = HealthBenchKnowledgeReactExecutionAdapter(gateway=gateway, tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True)
    response = asyncio.run(adapter.execute(request(tool, KNOWLEDGE)))
    assert response.metadata["knowledge_index"]["counts"]["medical_references"] == 1
    receipts = response.metadata["tool_receipts"]
    message = UpstreamMessage("producer", "node", response.text, tool_receipts=tuple(receipts))
    projection = _healthbench_v3_receipts(message, seen_sources=set(), char_budget=14000)
    assert projection["evidence_receipts"][0]["source"] == client.row["source"]
    downstream = SequenceGateway(action(KNOWLEDGE, database="medical_references", query="relation"),
                                 complete(artifact(client.row)))
    consumer = HealthBenchKnowledgeReactExecutionAdapter(gateway=downstream, tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True)
    response = asyncio.run(consumer.execute(request(KNOWLEDGE, upstream=(message,))))
    assert response.metadata["knowledge_index"]["counts"]["medical_references"] == 1
    assert client.calls == [("search", "relation")]


@pytest.mark.parametrize("tool", TOOLS)
def test_new_sources_share_query_anchor_and_repetition_budgets(tmp_path, tool):
    tools, _ = registry()
    adapter = HealthBenchKnowledgeReactExecutionAdapter(gateway=SequenceGateway(), tool_registry=tools,
        knowledge_root=tmp_path, max_turns=6, max_tool_calls=3, require_task_query_anchor=True)
    req = request(tool)
    good = action(tool, query="relation")
    assert adapter._tool_action_error(request=req, action=StructuredAction.from_value(good), observations=[]) is None
    assert adapter._tool_action_error(request=req,
        action=StructuredAction.from_value(action(tool, query="unrelated astronomy")), observations=[]) == "query_does_not_preserve_public_task_anchor"
    observations = [{"observation_status": "success", "executed_action": good,
                     "result": {"operation": "search", "evidence": []}}]
    assert adapter._tool_action_error(request=req, action=StructuredAction.from_value(good), observations=observations) == "duplicate_tool_request"
    assert adapter._state_conditioned_action_domain(req, observations * 3) == (frozenset(), True)


def test_reference_flag_reaches_existing_factory_and_is_not_a_truthy_string(tmp_path):
    config = optional_config()
    section = config["healthbench_tool_runtime"]
    section.update(toolset="source_separated_clinical_v1", knowledge_root="artifacts/indexes",
                   skillflow_source="/upstream/src", clinical_reference_sources_enabled=True)
    assert _settings(config)["clinical_reference_sources_enabled"] is True
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config, backend.registry, backend.project_root = config, model_registry(), tmp_path
    backend.runtime = SimpleNamespace(gateway=object(), timeout_seconds=30,
        artifact_communication_profile="producer_context_structured_evidence_v4")
    opened = SimpleNamespace(registry=object(), close=Mock())
    with (patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
          patch.object(SMOKE, "open_healthbench_knowledge_tool_registry", return_value=opened) as op,
          patch.object(SMOKE, "HealthBenchKnowledgeReactExecutionAdapter", return_value=SimpleNamespace(execute=Mock()))):
        backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert op.call_args.kwargs["clinical_reference_sources_enabled"] is True
    section["clinical_reference_sources_enabled"] = "true"
    with pytest.raises(ConfigurationError):
        _settings(config)


def test_v239_preserves_graph_evaluator_and_budgets_while_allowing_search_then_read():
    from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
    root = Path(__file__).resolve().parents[2]
    old = load_yaml(root / "config/evaluation_healthbench_professional_external_medical_sources_v2_38_dev5.yaml")
    new = load_yaml(root / "config/evaluation_healthbench_professional_clinical_reference_sources_v2_39_dev5.yaml")
    validate_completion_benchmark_config(new)
    assert new["evaluation"] == old["evaluation"]
    assert new["director"] == old["director"]
    graph = dict(new["agent_graph"])
    graph["model_catalog_path"] = old["agent_graph"]["model_catalog_path"]
    assert graph == old["agent_graph"]
    section = new["healthbench_professional_evaluation"]
    assert section["task_ids"] == old["healthbench_professional_evaluation"]["task_ids"]
    assert section["sample_count"] == 5
    profiles = new["healthbench_tool_runtime"]["execution_profile_allowlist"]
    assert section["direct_allowed_tools"] == profiles[-1]["allowed_tools"]
    assert len(section["direct_allowed_tools"]) == 12
    assert len(profiles) == 2
    assert profiles[0] == {"execution_mode": "reasoning", "allowed_tools": []}
    assert profiles[1]["execution_mode"] == "react"
    assert set(TOOLS) <= set(profiles[1]["allowed_tools"])
    for key in ("max_turns_per_agent_call", "max_tool_calls_per_agent_call", "max_successful_queries"):
        assert new["healthbench_tool_runtime"][key] == old["healthbench_tool_runtime"][key]
    catalog = load_yaml(root / new["agent_graph"]["model_catalog_path"])
    for model in catalog["models"]:
        if model["metadata"]["tool_capable"] == "true":
            assert set(TOOLS) <= set(model["metadata"]["tool_capability_scope"].split(","))
    assert all(not new[name]["enabled"] for name in ("grpo", "policy_sync", "exploration", "skills"))
    profiles[1]["allowed_tools"] = profiles[1]["allowed_tools"][:-1]
    with pytest.raises(ConfigurationError):
        validate_completion_benchmark_config(new)
