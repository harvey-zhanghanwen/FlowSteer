"""Deterministic synthetic retrieval/interface tests; no medical QA evaluation."""
import asyncio
import json
from pathlib import Path
import re

import numpy as np
import pytest

from src.interactive import healthbench_semantic_retrieval as semantic
from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore
from src.interactive.config_loader import load_yaml
from scripts.healthbench_candidate_skill_profile import load_candidate_skill_profile, build_candidate_prompt_priors
from tests.unit.test_healthbench_knowledge_store import evidence


class SyntheticEncoder:
    max_seq_length = 512
    def tokenizer(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}
    def encode(self, texts, **kwargs):
        return np.asarray([[1., 0.] if any(word in text for word in ("myocardial", "heart attack"))
                           else [0., 1.] for text in texts], dtype=np.float32)


@pytest.fixture
def index(tmp_path, monkeypatch):
    model = SyntheticEncoder()
    monkeypatch.setattr(semantic, "encoder", lambda path: model)
    records = [evidence(title="Synthetic source A", excerpt="myocardial infarction observation", document_id="a"),
               evidence(title="Synthetic source B", excerpt="weather observation", document_id="b")]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    np.save(tmp_path / "vectors.npy", np.eye(2, dtype=np.float32))
    np.save(tmp_path / "owners.npy", np.asarray([0, 1], dtype=np.int64))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": semantic.SCHEMA, "status": "frozen",
        "encoder_path": "synthetic", "record_count": 2}))
    return semantic.SemanticEvidenceIndex(manifest), records


def test_semantic_match_does_not_require_shared_surface_words(index):
    retriever, records = index
    assert retriever._bm25_score("heart attack", [semantic.source_text(r) for r in records]) == [0., 0.]
    hits, receipt = retriever.rank("heart attack", records)
    assert len(hits) == 1 and hits[0]["document_id"] == "a"
    assert hits[0]["retrieval_scores"]["hybrid"] == pytest.approx(0.6)
    assert receipt["similarity_is_not_medical_verification"]
    assert hits[0]["excerpt"] == records[0]["excerpt"]


def test_semantic_backend_cannot_return_unrouted_records(index):
    retriever, records = index
    hits, _ = retriever.rank("heart attack", [records[1]])
    assert not hits


def test_new_tool_evidence_is_ranked_without_rebuilding_frozen_index(index):
    retriever, _ = index
    fresh = evidence(title="New observation", document_id="new", excerpt="myocardial source")
    hits, receipt = retriever.rank("heart attack", [fresh])
    assert hits[0]["document_id"] == "new"
    assert receipt["new_source_windows_encoded"] == 1


def test_long_source_and_query_windows_keep_tail():
    text = "word " * 1100 + "TAIL_MARKER"
    parts = semantic.windows(SyntheticEncoder(), text)
    assert len(parts) > 1 and parts[-1][0].endswith("TAIL_MARKER")
    assert parts[0][1] == 0 and parts[-1][2] == len(text)
    assert all(a[2] >= b[1] for a, b in zip(parts, parts[1:]))


def test_store_uses_semantic_for_external_sources_not_patient_conversation(index, tmp_path):
    retriever, records = index
    store = HealthBenchKnowledgeStore(tmp_path / "store", [{"role": "user", "content": "weather context"}],
                                     semantic_index=retriever)
    try:
        for record in records: store.ingest_evidence(record)
        result = store.search("medical_references", "heart attack")
        assert result["evidence"][0]["document_id"] == "a"
        assert result["index_receipt"]["backend"] == "skillflow-bm25-bge-hybrid"
        context = store.search("conversation", "weather")
        assert context["conversation_matches"][0]["content"] == "weather context"
        assert not context["evidence"]
    finally:
        store.close()


def test_existing_react_loop_observes_semantic_hit(index, tmp_path, monkeypatch):
    from tests.unit.test_healthbench_knowledge_tools import adapter, lookup, KNOWLEDGE, source
    from tests.unit.test_healthbench_clinical_react import request, complete, artifact
    retriever, _ = index
    monkeypatch.setattr(semantic, "load_semantic_index", lambda path: retriever)
    build = HealthBenchKnowledgeStore(tmp_path / "seed", [])
    try:
        build.ingest_evidence(source())
        result = build.search("medical_references", "relation")
        manifest = tmp_path / "seed.json"
        manifest.write_text(json.dumps({"schema_version": "flowsteer.healthbench.public-question-evidence-corpus.v1",
            "status": "frozen", "databases": {"medical_references": result["index_receipt"]}}))
    finally:
        build.close()
    obj, gateway = adapter(tmp_path / "runtime", lookup(), complete(artifact(source())),
        frozen_corpus_manifest=manifest, semantic_index_manifest="synthetic")
    response = asyncio.run(obj.execute(request(KNOWLEDGE)))
    assert len(gateway.requests) == 2
    receipt = response.metadata["tool_receipts"][0]["result"]["value"]
    assert receipt["index_receipt"]["backend"] == "skillflow-bm25-bge-hybrid"
    assert receipt["evidence"][0]["source_id"] == source()["source_id"]


def test_new_condition_retires_old_candidates_and_uses_different_five():
    root = Path(__file__).resolve().parents[2]
    new = load_yaml(root / "config/evaluation_healthbench_semantic_skills_new5.yaml")
    old = load_yaml(root / "config/evaluation_healthbench_question_corpus_dev5.yaml")
    section = "healthbench_professional_evaluation"
    assert len(new[section]["task_ids"]) == 5
    assert set(new[section]["task_ids"]).isdisjoint(old[section]["task_ids"])
    for key in ("concurrency", "task_timeout_seconds", "rollouts_per_task"):
        assert new[section][key] == old[section][key]
    profile = load_candidate_skill_profile(root / new["candidate_skill_evaluation"]["profile_path"], run_config=new)
    priors = build_candidate_prompt_priors(profile)
    previous = load_yaml(root / old["candidate_skill_evaluation"]["profile_path"])
    assert {r["condition_id"] for r in priors}.isdisjoint(r["condition_id"] for r in previous["candidates"])
    assert sum(len(r["action"]["instruction"]) for r in priors) < sum(len(r["action"]["instruction"]) for r in previous["candidates"])
    assert all(r["rejectable"] for r in priors)
