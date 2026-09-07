"""Offline external-source fixtures: title lookup, full passages and no answers."""
import json
from pathlib import Path

from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore, load_frozen_public_evidence
from scripts.refine_healthbench_evidence_corpus import refine
from tests.unit.test_healthbench_knowledge_store import evidence


def test_title_identifier_match_is_distinct_from_clinical_verification(tmp_path):
    store = HealthBenchKnowledgeStore(tmp_path, (), metadata_aware_retrieval=True)
    try:
        correct = evidence(document_id="321123", title="NOVA trial study group", source="NCBI PubMed",
                           excerpt="An intervention was studied in the documented population.")
        correct.update(pmid="321123", source_id="pubmed:321123")
        distractor = evidence(document_id="other", title="General handbook", excerpt="NOVA trial terminology. " * 80)
        store.ingest_evidence(distractor)
        store.ingest_evidence(correct)
        result = store.search("medical_references", "NOVA trial")
        assert result["evidence"][0]["document_id"] == "321123"
        assert result["evidence"][0]["retrieval_match"]["clinical_correctness_verified"] is False
        assert store.search("medical_references", "321123")["evidence"][0]["retrieval_match"]["type"] == "exact_identifier"
        assert store.search("medical_references", "other-topic")["index_receipt"]["metadata_match_count"] == 0
        assert correct["excerpt"] == result["evidence"][0]["excerpt"]
    finally:
        store.close()


def test_metadata_promotion_preserves_hybrid_fallback_and_no_duplicate_evidence(tmp_path):
    class Hybrid:
        def rank(self, query, records, limit):
            return [{**row, "rank": i+1} for i, row in enumerate(records[:limit])], {"backend": "synthetic-hybrid"}
    store = HealthBenchKnowledgeStore(tmp_path, (), semantic_index=Hybrid(), metadata_aware_retrieval=True)
    try:
        store.ingest_evidence(evidence(document_id="b", title="Related source", excerpt="Related context."))
        store.ingest_evidence(evidence(document_id="a", title="NOVA trial", excerpt="The actual reported finding."))
        result = store.search("medical_references", "NOVA trial", limit=2)
        assert [row["document_id"] for row in result["evidence"]] == ["a", "b"]
        assert result["index_receipt"]["metadata_lookup"]["metadata_match_count"] == 1
        assert result["index_receipt"]["backend"] == "synthetic-hybrid"
        no_title = store.search("medical_references", "unmatched topic", limit=2)
        assert [row["document_id"] for row in no_title["evidence"]] == ["b", "a"]
    finally:
        store.close()


def test_enrichment_uses_original_external_passage_and_does_not_overwrite_old_corpus(tmp_path):
    seed = HealthBenchKnowledgeStore(tmp_path / "old", ())
    original = evidence(document_id="textbook-1", title="Synthetic textbook", excerpt="First sentence.")
    original["source"] = "MedRAG/textbooks"
    try:
        seed.ingest_evidence(original)
        result = seed.search("medical_references", "sentence")
        manifest = tmp_path / "old.json"
        manifest.write_text(json.dumps({"schema_version": "flowsteer.healthbench.public-question-evidence-corpus.v1",
            "status": "frozen", "databases": {"medical_references": result["index_receipt"]}}))
    finally:
        seed.close()
    source = tmp_path / "chunks.jsonl"
    full = "First sentence. Important applicability and limitations in the original source."
    source.write_text(json.dumps({"id": "textbook-1", "title": "Synthetic textbook", "contents": full}) + "\n")
    target = tmp_path / "new"
    outcome = refine(manifest, source, target)
    rows = load_frozen_public_evidence(target / "manifest.json")
    assert len(rows) == 1 and rows[0]["excerpt"] == full
    assert rows[0]["text_scope"] == "full_local_source_passage"
    assert rows[0]["truncated"] is False
    assert load_frozen_public_evidence(manifest)[0]["excerpt"] == "First sentence."
    assert outcome["expanded_source_records"] == 1
    assert outcome["new_network_calls"] == outcome["grader_calls"] == 0
