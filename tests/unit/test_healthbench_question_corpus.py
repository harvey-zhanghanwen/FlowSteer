"""Synthetic corpus, scope and FINISH regressions; no model or remote calls."""
import asyncio
import json
from pathlib import Path

import pytest

from scripts.build_healthbench_question_corpus import public_queries, public_questions
from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore, load_frozen_public_evidence
from src.interactive.healthbench_professional_adapter import render_model_visible_conversation
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import load_yaml
from tests.unit.test_scope_neutral_contract_admission import _registry, _add_contract
from tests.unit.test_healthbench_knowledge_store import evidence


def test_525_public_conversations_only_are_used_for_queries(tmp_path):
    question = render_model_visible_conversation([{"role": "user", "content": "Find synthetic evidence about magnesium."}])
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps({"task_id": f"task-{i}", "question": question,
        "ground_truth": "NEVER_INDEX_REFERENCE", "rubric": "NEVER_INDEX_RUBRIC"}) + "\n" for i in range(525)))
    rows = public_questions(path)
    assert len(rows) == 525
    assert all(public_queries(q) == ("Find synthetic evidence about magnesium.",) for _, q in rows)
    assert "NEVER_INDEX" not in json.dumps(rows)


def test_frozen_library_uses_source_records_not_query_to_answer_mapping(tmp_path):
    store = HealthBenchKnowledgeStore(tmp_path / "build", [])
    try:
        store.ingest_evidence(evidence(agent_summary="NEVER_INDEX", rubric="NEVER_INDEX"))
        result = store.search("medical_references", "magnesium")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "flowsteer.healthbench.public-question-evidence-corpus.v1",
            "status": "frozen", "databases": {"medical_references": result["index_receipt"]},
            "queries": "NEVER_INDEX", "reference_response": "NEVER_INDEX",
        }))
        frozen = load_frozen_public_evidence(manifest)
        assert len(frozen) == 1 and "NEVER_INDEX" not in json.dumps(frozen)
        fresh = HealthBenchKnowledgeStore(tmp_path / "runtime", [{"role": "user", "content": "New conversation"}])
        try:
            for record in frozen:
                assert fresh.ingest_evidence(record)
            hit = fresh.search("medical_references", "magnesium")
            assert hit["evidence"][0]["source_id"] == evidence()["source_id"]
            assert fresh.counts()["conversation"] == 1
        finally:
            fresh.close()
    finally:
        store.close()


def test_source_unicode_is_normalized_only_in_upstream_index(tmp_path):
    raw = evidence(title="Synthetic cafe\u0301 publication", excerpt="magnesium cafe\u0301 observation. ")
    store = HealthBenchKnowledgeStore(tmp_path, [])
    try:
        assert store.ingest_evidence(raw)
        result = store.search("medical_references", "magnesium")
        assert result["evidence"][0]["excerpt"] == raw["excerpt"]
        receipt = result["index_receipt"]
        saved = json.loads(Path(receipt["records_path"]).read_text())
        assert saved["title"] == raw["title"] and saved["excerpt"] == raw["excerpt"]
        with store._module.RetrievalIndex.open(Path(receipt["index_path"])) as index:
            assert index.read(saved["passage_id"]).text == "magnesium caf\u00e9 observation. "
    finally:
        store.close()


def test_resume_publishes_saved_sources_without_retrieving_again(tmp_path, monkeypatch):
    import scripts.build_healthbench_question_corpus as builder
    config = tmp_path / "config.yaml"
    from src.interactive.qa_retrieval import DEFAULT_SKILLFLOW_SOURCE
    config.write_text(json.dumps({"data": {"test_path": "unused"},
        "healthbench_tool_runtime": {"skillflow_source": str(DEFAULT_SKILLFLOW_SOURCE)}}))
    question = render_model_visible_conversation([{"role": "user", "content": "magnesium"}])
    monkeypatch.setattr(builder, "public_questions", lambda path: [(str(i), question) for i in range(525)])
    def no_retrieval(*args, **kwargs):
        raise AssertionError("resume must not retrieve again")
    monkeypatch.setattr(builder.FrozenMedRAGBM25Corpus, "open", no_retrieval)
    records = tmp_path / "previous" / "medical_references"
    records.mkdir(parents=True)
    (records / "records.jsonl").write_text(json.dumps(evidence()) + "\n")
    manifest = builder.build(config, tmp_path / "published", [], records.parent)
    assert manifest["question_count"] == 525
    assert manifest["question_retrieval_nonempty"] is None
    assert not manifest["retrieval_executed_this_invocation"]
    assert manifest["source_counts"] == {"NCBI PubMed": 1}
    assert len(load_frozen_public_evidence(tmp_path / "published/manifest.json")) == 1


@pytest.mark.parametrize("contract", [
    "Identify sources and output: 1) list evidence 2) state uncertainty.",
    "Identify sources and output: (1) list evidence (2) state uncertainty.",
])
def test_numbered_contract_is_not_an_unsupported_clinical_value(contract):
    env = AgentWorkflowEnv(_registry(), object(), problem="Find evidence for the question.",
        require_scope_neutral_contracts=True, execute_on_edit=False)
    result = asyncio.run(env.step(_add_contract(contract)))
    assert result.accepted, result.feedback
    assert env.graph.get_node("node_1").contract == contract


def test_numbered_list_does_not_exempt_actual_dose():
    env = AgentWorkflowEnv(_registry(), object(), problem="Find evidence for the question.",
        require_scope_neutral_contracts=True, execute_on_edit=False)
    result = asyncio.run(env.step(_add_contract("Output: 1) recommend 47 mg 2) list sources.")))
    assert not result.accepted and "47" in result.feedback


def test_valid_output_can_be_repaired_instead_of_forcing_finish():
    class Gateway:
        async def generate(self, request):
            return "Below is the summary of the requested evidence."
    env = AgentWorkflowEnv(_registry(), Gateway(), problem="Summarize evidence.",
        finish_only_when_admissible=False, execute_on_edit=True)
    action = json.loads(_add_contract("Summarize all requested evidence."))
    action["output_agent_id"] = "node_1"
    result = asyncio.run(env.step(json.dumps(action)))
    assert result.accepted and env.finish_admissibility()["admissible"]
    assert {"finish", "modify_agent"}.issubset(env.model_admissible_action_types())
    assert not env.finished


def test_new_profile_keeps_fixed_samples_models_and_nontraining_boundary():
    root = Path(__file__).resolve().parents[2]
    new = load_yaml(root / "config/evaluation_healthbench_question_corpus_dev5.yaml")
    old = load_yaml(root / "config/evaluation_healthbench_professional_v233_all_sources_dev5.yaml")
    assert new["healthbench_professional_evaluation"] == old["healthbench_professional_evaluation"]
    assert new["agent_graph"]["model_catalog_path"] == old["agent_graph"]["model_catalog_path"]
    assert new["evaluation"] == old["evaluation"]
    assert new["candidate_skill_evaluation"] == old["candidate_skill_evaluation"]
    assert not new["agent_graph"]["finish_only_when_admissible"]
    assert new["director"]["max_prompt_tokens"] == 24000 and new["director"]["context_projection"]
    assert not new["experiment"]["training_enabled"] and new["grpo"]["max_optimizer_updates"] == 0


def test_seeded_source_is_visible_through_actual_react_tool_observation(tmp_path):
    from tests.unit.test_healthbench_knowledge_tools import adapter, lookup, source, KNOWLEDGE
    from tests.unit.test_healthbench_clinical_react import request, complete, artifact
    store = HealthBenchKnowledgeStore(tmp_path / "build", [])
    try:
        store.ingest_evidence(source())
        result = store.search("medical_references", "relation")
        manifest = tmp_path / "library.json"
        manifest.write_text(json.dumps({
            "schema_version": "flowsteer.healthbench.public-question-evidence-corpus.v1",
            "status": "frozen", "databases": {"medical_references": result["index_receipt"]},
        }))
    finally:
        store.close()
    executor, gateway = adapter(tmp_path / "runtime", lookup(), complete(artifact(source())),
        frozen_corpus_manifest=manifest)
    response = asyncio.run(executor.execute(request(KNOWLEDGE)))
    assert len(gateway.requests) == 2
    assert response.metadata["knowledge_index"]["frozen_record_count"] == 1
    receipt = response.metadata["tool_receipts"][0]
    assert receipt["tool_id"] == KNOWLEDGE
    assert receipt["result"]["value"]["evidence"][0]["source_id"] == source()["source_id"]
