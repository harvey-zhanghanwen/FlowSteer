"""Real upstream FTS5 tests over synthetic data; no models, HTTP, or rubric."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore
from src.interactive.qa_retrieval import _load_retrieval_module, DEFAULT_SKILLFLOW_SOURCE


def evidence(source="NCBI PubMed", **changes):
    return {
        "source_type": "peer_reviewed_literature", "source": source,
        "source_id": "pubmed:1234", "document_id": "1234",
        "title": "Synthetic magnesium publication", "date": "2026",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1234/",
        "excerpt": "Synthetic magnesium observation. No clinical recommendation.",
        "content_type": "pubmed_abstract_not_full_text", "version": None,
        **changes,
    }


class HealthBenchKnowledgeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="healthbench-store-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = self.new_store([
            {"role": "user", "content": "Synthetic patient reports magnesium exposure."},
            {"role": "assistant", "content": "Synthetic prior conversation; not a guideline."},
        ])

    def new_store(self, conversation=()):
        store = HealthBenchKnowledgeStore(self.root, conversation)
        self.addCleanup(store.close)
        return store

    def test_source_databases_are_empty_until_real_observations_arrive(self):
        self.assertEqual(self.store.counts(), {
            "conversation": 2, "medical_references": 0, "drug_labels": 0,
        })
        result = self.store.search("medical_references", "magnesium")
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["evidence"], [])
        self.assertIsNone(result["index_receipt"])
        self.assertEqual(list(self.store.directory.rglob("*.sqlite3")), [])

    def test_complete_conversation_is_separate_from_external_evidence(self):
        text = "Synthetic magnesium context. " * 70
        store = self.new_store([{"role": "user", "content": text, "ignored": "not copied"}])
        result = store.search("conversation", "magnesium")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["conversation_matches"][0]["content"], text)
        self.assertEqual(result["conversation_matches"][0]["role"], "user")
        self.assertEqual(result["conversation_matches"][0]["turn_index"], 0)
        self.assertNotIn("ignored", result["conversation_matches"][0])
        self.assertEqual(store.search("medical_references", "magnesium")["status"], "empty")

    def test_source_routes_and_original_provenance_are_preserved(self):
        pubmed = evidence()
        textbook = evidence("MedRAG/textbooks", document_id="book-7", source_id="medrag:book-7",
                            source_type="frozen_medical_textbook", version="fixture-v1")
        label = evidence("NLM DailyMed", document_id="label-7", source_id="dailymed:label-7",
                         source_type="drug_label", version="2", published_date="2026-09-01")
        for row in (pubmed, textbook, label):
            self.assertTrue(self.store.ingest_evidence(row))
        references = self.store.search("medical_references", "magnesium")
        labels = self.store.search("drug_labels", "magnesium")
        self.assertEqual({row["source"] for row in references["evidence"]}, {"NCBI PubMed", "MedRAG/textbooks"})
        self.assertEqual(len(labels["evidence"]), 1)
        for key, value in label.items():
            self.assertEqual(labels["evidence"][0][key], value)
        self.assertEqual(labels["conversation_matches"], [])

    def test_unrecognised_sources_and_nested_non_evidence_fields_are_not_ingested(self):
        for row in ({"source": "agent", "excerpt": "a model answer"},
                    evidence(source="benchmark answer database"),
                    evidence(excerpt=""), {"conversation": []}, None):
            self.assertFalse(self.store.ingest_evidence(row))
        self.assertTrue(self.store.ingest_evidence(evidence(
            private_evaluation={"unrelated": "field"}, agent_summary="not source material",
        )))
        result = self.store.search("medical_references", "magnesium")
        record = result["evidence"][0]
        self.assertNotIn("private_evaluation", record)
        self.assertNotIn("agent_summary", record)
        persisted = Path(result["index_receipt"]["metadata_path"]).read_text()
        self.assertNotIn("private_evaluation", persisted)
        self.assertNotIn("agent_summary", persisted)

    def test_versions_pages_and_long_excerpts_are_not_overwritten(self):
        first = evidence("NLM DailyMed", source_id="dailymed:label", version="1",
                         excerpt="magnesium " * 80, offset=0, next_offset=800,
                         truncated=True, total_characters=1600)
        self.assertTrue(self.store.ingest_evidence(first))
        self.assertFalse(self.store.ingest_evidence(first))
        initial = self.store.search("drug_labels", "magnesium")
        first_index = Path(initial["index_receipt"]["index_path"])
        self.assertEqual(initial["evidence"][0]["excerpt"], first["excerpt"])
        self.assertTrue(self.store.ingest_evidence({**first, "offset": 800, "next_offset": None,
                                                  "truncated": False, "excerpt": "magnesium continuation"}))
        self.assertTrue(self.store.ingest_evidence({**first, "version": "2"}))
        updated = self.store.search("drug_labels", "magnesium", limit=5)
        self.assertEqual(len(updated["evidence"]), 3)
        self.assertNotEqual(updated["index_receipt"]["index_path"], str(first_index))
        self.assertTrue(first_index.exists())
        self.assertEqual(self.store.search("drug_labels", "magnesium")["index_receipt"], updated["index_receipt"])
        upstream = _load_retrieval_module(DEFAULT_SKILLFLOW_SOURCE)
        with upstream.RetrievalIndex.open(first_index) as old:
            old_hits = old.search("magnesium", limit=5)
            self.assertEqual(len(old_hits), 1)
            self.assertEqual(old.read(old_hits[0].passage_id).text, first["excerpt"])

    def test_actual_upstream_builder_search_read_and_metadata_sidecar(self):
        self.store.ingest_evidence(evidence())
        result = self.store.search("medical_references", "magnesium")
        receipt = result["index_receipt"]
        module = _load_retrieval_module(DEFAULT_SKILLFLOW_SOURCE)
        with module.RetrievalIndex.open(Path(receipt["index_path"])) as index:
            hits = index.search("magnesium", limit=3)
            self.assertEqual(hits[0].passage_id, result["evidence"][0]["passage_id"])
            self.assertEqual(index.read(hits[0].passage_id).text, evidence()["excerpt"])
            self.assertEqual(index.manifest.retrieval_backend, "sqlite-fts5-lexical")
        metadata = json.loads(Path(receipt["metadata_path"]).read_text())
        self.assertEqual(metadata["index_receipt"], receipt)
        self.assertEqual(metadata["records"][0]["url"], evidence()["url"])

    def test_concurrent_callers_and_active_async_loop_reuse_thread_affinity(self):
        self.store.ingest_evidence(evidence())
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: self.store.search("medical_references", "magnesium"), range(6)))
        self.assertTrue(all(len(item["evidence"]) == 1 for item in results))
        self.assertEqual(len(list(self.store.directory.rglob("*.sqlite3"))), 1)
        async def use_sync_backend():
            return self.store.search("medical_references", "magnesium")
        self.assertEqual(asyncio.run(use_sync_backend())["status"], "ok")

    def test_instances_never_read_other_request_database_even_same_parent(self):
        self.store.ingest_evidence(evidence())
        self.store.search("medical_references", "magnesium")
        other = self.new_store()
        self.assertNotEqual(other.directory, self.store.directory)
        self.assertEqual(other.search("medical_references", "magnesium")["status"], "empty")
        self.assertEqual(other.search("conversation", "magnesium")["conversation_matches"], [])

    def test_public_records_persist_without_forcing_unused_index_build(self):
        self.store.ingest_evidence(evidence())
        records = self.store.directory / "medical_references" / "records.jsonl"
        self.store.close()
        self.assertEqual(json.loads(records.read_text())["excerpt"], evidence()["excerpt"])
        self.assertEqual(list(self.store.directory.rglob("*.sqlite3")), [])
        self.store.close()
        with self.assertRaises(RuntimeError):
            self.store.search("medical_references", "magnesium")

    def test_empty_results_and_invalid_queries_are_explicit(self):
        self.store.ingest_evidence(evidence())
        result = self.store.search("medical_references", "unrelatedzztoken")
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["evidence"], [])
        for database, query, limit in (("unknown", "magnesium", 1), ("conversation", "", 1),
                                       ("conversation", "x", 0), ("conversation", "x", True)):
            with self.subTest(database=database, query=query, limit=limit):
                with self.assertRaises(ValueError):
                    self.store.search(database, query, limit)


if __name__ == "__main__":
    unittest.main()
