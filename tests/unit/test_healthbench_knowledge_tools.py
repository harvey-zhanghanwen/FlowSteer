"""Synthetic routed-evidence checks; real SkillFlow FTS5, no model or network."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import CommunicationCondition, UpstreamMessage
from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore
from src.interactive.healthbench_knowledge_tools import (
    HEALTHBENCH_KNOWLEDGE_TOOL_ID as KNOWLEDGE, _CURRENT_STORE,
    HealthBenchKnowledgeReactExecutionAdapter, build_healthbench_knowledge_tool_registry,
    _routed_receipts,
)
from src.interactive.openai_gateway import _healthbench_search_candidates
from src.interactive.tool_runtime import ToolRegistration, ToolRegistry, ToolRequest, ToolResult, StructuredAction
from tests.unit.test_healthbench_clinical_react import (
    SEARCH, DRUG, SequenceGateway, action, complete, artifact, evidence,
    receipt, registry, request,
)


def source(tool=SEARCH, document_id="synthetic-paper"):
    return {**evidence(document_id),
            "source": "NLM DailyMed" if tool == DRUG else "NCBI PubMed",
            "source_type": "drug_label" if tool == DRUG else "abstract",
            "source_id": "dailymed:synthetic-label" if tool == DRUG else "pubmed:synthetic-paper",
            "version": "2" if tool == DRUG else None, "offset": 0}


class ObservedSource:
    def __init__(self, tool):
        self.tool, self.calls = tool, 0

    def invoke(self, req):
        self.calls += 1
        return ToolResult(receipt(self.tool, source(self.tool))["result"]["value"])


def wrapped_registry():
    base = registry()
    return build_healthbench_knowledge_tool_registry(ToolRegistry(tuple(
        ToolRegistration(tool, ObservedSource(tool), base.require_capability(tool))
        for tool in (SEARCH, DRUG)
    )))


def lookup(database="medical_references", query="relation"):
    return {"kind": "tool", "name": "search", "resource_id": KNOWLEDGE,
            "skill_id": None, "arguments": {"database": database, "query": query}}


def adapter(root, *outputs, **kwargs):
    gateway = SequenceGateway(*outputs)
    obj = HealthBenchKnowledgeReactExecutionAdapter(
        gateway=gateway, tool_registry=wrapped_registry(), knowledge_root=root,
        max_turns=6, max_tool_calls=3, require_structured_evidence_artifact=True,
        **kwargs,
    )
    return obj, gateway


def test_real_tool_observation_is_indexed_and_queried_in_existing_react_loop(tmp_path):
    obj, gateway = adapter(tmp_path, action(SEARCH, "relation"), lookup(), complete(artifact(source())))
    response = asyncio.run(obj.execute(request(SEARCH, KNOWLEDGE)))
    assert len(gateway.requests) == 3
    assert response.metadata["knowledge_index"]["counts"] == {
        "conversation": 1, "medical_references": 1, "drug_labels": 0,
    }
    directory = Path(response.metadata["knowledge_index"]["directory"])
    assert json.loads((directory / "request_receipt.json").read_text())["agent_id"] == "node"
    saved = json.loads((directory / "medical_references/records.jsonl").read_text())
    assert saved["excerpt"] == source()["excerpt"]
    assert saved["source_id"] == source()["source_id"]
    assert list((directory / "medical_references").glob("*.sqlite3"))
    assert _CURRENT_STORE.get() is None


def upstream():
    return UpstreamMessage("producer", "node", "Unverified summary is not evidence.",
                           tool_receipts=(receipt(SEARCH, source()),))


def test_only_routed_receipts_populate_downstream_index(tmp_path):
    obj, _ = adapter(tmp_path, lookup(), complete(artifact(source())))
    response = asyncio.run(obj.execute(request(KNOWLEDGE, upstream=(upstream(),))))
    assert response.metadata["knowledge_index"]["counts"]["medical_references"] == 1
    # Same parent directory is not an off-graph, cross-task shared memory.
    obj, _ = adapter(tmp_path, complete("The supplied conversation is available."))
    isolated = asyncio.run(obj.execute(request(KNOWLEDGE, output=True)))
    assert isolated.metadata["knowledge_index"]["counts"]["medical_references"] == 0
    assert isolated.metadata["knowledge_index"]["directory"] != response.metadata["knowledge_index"]["directory"]


def test_masked_upstream_does_not_enter_index_but_own_receipts_do(tmp_path):
    obj, _ = adapter(tmp_path, complete("The supplied conversation is available."))
    response = asyncio.run(obj.execute(request(KNOWLEDGE, output=True, upstream=(upstream(),),
        communication_condition=CommunicationCondition.UPSTREAM_MASKED)))
    assert response.metadata["knowledge_index"]["counts"]["medical_references"] == 0
    obj, _ = adapter(tmp_path, complete("The retained source describes the requested relation."))
    response = asyncio.run(obj.execute(request(KNOWLEDGE, output=True, upstream=(upstream(),),
        prior_tool_receipts=(receipt(DRUG, source(DRUG)),),
        communication_condition=CommunicationCondition.UPSTREAM_MASKED)))
    assert response.metadata["knowledge_index"]["counts"] == {
        "conversation": 1, "medical_references": 0, "drug_labels": 1,
    }


def test_peer_and_transitive_provenance_use_existing_envelopes_only():
    parent = upstream()
    forwarded = replace(parent, source_agent_id="middle", tool_receipts=(),
                        input_artifact_provenance=(parent.to_dict(),))
    rows = list(_routed_receipts(request(KNOWLEDGE, upstream=(forwarded,), peer_draft=parent)))
    assert len(rows) == 1
    assert rows[0]["result"]["value"]["evidence"][0]["source"] == "NCBI PubMed"


def test_conversation_query_is_not_projected_as_external_medical_evidence(tmp_path):
    store = HealthBenchKnowledgeStore(tmp_path, [{"role": "user", "content": "I report a relation."}])
    token = _CURRENT_STORE.set(store)
    try:
        tools = wrapped_registry()
        result = asyncio.run(tools.ainvoke(KNOWLEDGE, ToolRequest("search", lookup("conversation")["arguments"])))
        assert result.value["conversation_matches"][0]["role"] == "user"
        assert result.value["evidence"] == []
        row = {"tool_id": KNOWLEDGE, "error_type": None,
               "request": {"action": "search", "arguments": lookup("conversation")["arguments"]},
               "result": result.to_value()}
        assert list(_healthbench_search_candidates([row])) == []
        Draft202012Validator(tools.require_capability(KNOWLEDGE).output_schema).validate(result.value)
    finally:
        _CURRENT_STORE.reset(token)
        store.close()


def test_tool_indexing_failure_preserves_original_tool_output(tmp_path):
    store = HealthBenchKnowledgeStore(tmp_path, [])
    token = _CURRENT_STORE.set(store)
    try:
        tools = wrapped_registry()
        with patch.object(store, "ingest_evidence", side_effect=OSError("synthetic failure")):
            result = asyncio.run(tools.ainvoke(SEARCH, ToolRequest("search", {"query": "relation"})))
        assert result.completed
        assert result.value["evidence"] == [source()]
        assert result.value["knowledge_index"]["status"] == "partially_indexed"
        assert result.value["knowledge_index"]["error_types"] == ["OSError"]
    finally:
        _CURRENT_STORE.reset(token)
        store.close()


def test_unchanged_queries_blocked_but_new_evidence_permits_requery(tmp_path):
    obj, _ = adapter(tmp_path)
    store = HealthBenchKnowledgeStore(tmp_path, [])
    token = _CURRENT_STORE.set(store)
    try:
        sampled = StructuredAction.from_value(lookup())
        previous = [{"executed_action": lookup(), "result": {"database_counts": store.counts()},
                     "observation_status": "success"}]
        assert obj._tool_action_error(request=request(KNOWLEDGE), action=sampled, observations=previous) == "duplicate_tool_request"
        store.ingest_evidence(source(DRUG))
        assert obj._tool_action_error(request=request(KNOWLEDGE), action=sampled, observations=previous) == "duplicate_tool_request"
        store.ingest_evidence(source())
        assert obj._tool_action_error(request=request(KNOWLEDGE), action=sampled, observations=previous) is None
    finally:
        _CURRENT_STORE.reset(token)
        store.close()


def test_concurrent_invocations_do_not_share_context_or_sources(tmp_path):
    first, _ = adapter(tmp_path, action(DRUG, "synthetic"), complete("The label describes the requested relation."))
    second, _ = adapter(tmp_path, complete("No external source is assumed for this response."))
    async def run():
        return await asyncio.gather(first.execute(request(DRUG, KNOWLEDGE, output=True)),
                                    second.execute(request(KNOWLEDGE, output=True)))
    a, b = asyncio.run(run())
    assert a.metadata["knowledge_index"]["counts"]["drug_labels"] == 1
    assert b.metadata["knowledge_index"]["counts"]["drug_labels"] == 0
    assert _CURRENT_STORE.get() is None


def test_factory_uses_knowledge_adapter_and_original_resource_cleanup(tmp_path):
    from tests.unit.test_healthbench_optional_tools_wiring import SMOKE, optional_config, registry as model_registry, _task, _settings
    config = optional_config()
    config["healthbench_tool_runtime"].update(toolset="source_separated_clinical_v1",
        knowledge_root="artifacts/local-indexes", skillflow_source="/upstream/src")
    assert _settings(config)["toolset"] == "source_separated_clinical_v1"
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config, backend.registry, backend.project_root = config, model_registry(), tmp_path
    backend.runtime = SimpleNamespace(gateway=object(), timeout_seconds=30,
        artifact_communication_profile="producer_context_structured_evidence_v3")
    opened = SimpleNamespace(registry=object(), close=Mock())
    execution = SimpleNamespace(execute=Mock())
    with (patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
          patch.object(SMOKE, "open_healthbench_knowledge_tool_registry", return_value=opened) as op,
          patch.object(SMOKE, "open_healthbench_clinical_tool_registry") as old,
          patch.object(SMOKE, "HealthBenchKnowledgeReactExecutionAdapter", return_value=execution) as impl):
        runtime, tools, close = backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert op.call_count == 1
    old.assert_not_called()
    assert impl.call_args.kwargs["knowledge_root"] == tmp_path / "artifacts/local-indexes"
    assert runtime.execution_adapters["react"] is execution
    assert tools is opened.registry and close is opened.close


