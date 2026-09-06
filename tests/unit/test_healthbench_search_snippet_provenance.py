"""Synthetic search-fragment provenance; real local retrieval, no network."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator

from src.interactive.agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4,
    UpstreamMessage,
)
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_clinical_tools import HealthBenchSourceReadToolBackend
from src.interactive.healthbench_knowledge_store import HealthBenchKnowledgeStore
from src.interactive.healthbench_evidence_adapter import (
    PubMedEUtilitiesClient, build_healthbench_authoritative_tool_registry,
)
from src.interactive.healthbench_tool_adapter import (
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
    FrozenMedRAGBM25Corpus,
    build_healthbench_medrag_tool_registry,
)
from src.interactive.openai_gateway import (
    _healthbench_medrag_evidence, _healthbench_search_candidates,
    _healthbench_v3_receipts, build_agent_messages,
)
from src.interactive.tool_runtime import ToolRequest
from tests.unit.test_healthbench_clinical_react import request


SEARCH = HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID
PAGE_FIELDS = (
    "source_id", "content_type", "version", "offset", "total_characters",
    "truncated", "next_offset",
)


def corpus(text):
    return FrozenMedRAGBM25Corpus(
        source_identity="MedRAG/textbooks", source_revision="synthetic-v1", corpus_rows=1,
        _corpus=(text,), _document_ids=("book-1",), _titles=("Synthetic textbook",),
        _index={"idf": {"fixture": 2.0}, "inverted_index": {"fixture": [(0, 1)]},
                "doc_lens": [10], "avg_dl": 10},
    )


def search_result(text):
    source = corpus(text)
    registry = build_healthbench_medrag_tool_registry(source)
    result = registry.invoke(SEARCH, ToolRequest("search", {"query": "fixture"}))
    Draft202012Validator(registry.require_capability(SEARCH).output_schema).validate(result.value)
    return source, registry, result


def receipt(result):
    return {"tool_id": SEARCH, "error_type": None,
            "request": {"action": "search", "arguments": {"query": "fixture"}},
            "result": result.to_value()}


class HealthBenchSearchSnippetProvenanceTests(unittest.TestCase):
    def test_authoritative_aggregation_keeps_snippet_fields_and_strict_schema(self):
        source = corpus("fixture " + "x" * 650)
        self.addCleanup(source.close)
        registry = build_healthbench_authoritative_tool_registry(source)
        with patch.object(PubMedEUtilitiesClient, "search", return_value=[]):
            result = registry.invoke("healthbench-authoritative.search", ToolRequest("search", {"query": "fixture"}))
        Draft202012Validator(registry.require_capability("healthbench-authoritative.search").output_schema).validate(result.value)
        evidence = result.value["evidence"][0]
        assert evidence["source_id"] == "medrag:book-1"
        assert evidence["truncated"] is True
        assert evidence["next_offset"] == 500
        assert evidence["total_characters"] == 658

    def test_actual_chunk_length_controls_499_500_501_character_boundaries(self):
        for length in (499, 500, 501):
            with self.subTest(length=length):
                text = "fixture " + "x" * (length - len("fixture "))
                source, _, result = search_result(text)
                self.addCleanup(source.close)
                chunk = result.value["ranked_chunks"][0]
                self.assertEqual(chunk["text"], text[:500])
                self.assertEqual(chunk["total_characters"], length)
                self.assertIs(chunk["truncated"], length > 500)
                self.assertEqual(chunk["next_offset"], 500 if length > 500 else None)
                self.assertEqual(chunk["offset"], 0)
                self.assertEqual(chunk["source_id"], "medrag:book-1")
                self.assertEqual(chunk["content_type"], "frozen_textbook_chunk")
                self.assertEqual(chunk["version"], "synthetic-v1")
                self.assertEqual(chunk["document_id"], "book-1")
                self.assertEqual(chunk["score"], 2.0)
                self.assertEqual(chunk["matched_terms"], ["fixture"])
                self.assertEqual(chunk["rank"], 1)
                self.assertEqual(result.value["top_k"], 3)

    def test_cut_relation_is_explicit_and_existing_read_source_recovers_context(self):
        prefix = "fixture " + "x" * (500 - len("fixture ") - len(" before")) + " before"
        text = prefix + " the synthetic observation, not after it."
        self.assertEqual(len(prefix), 500)
        source, _, result = search_result(text)
        self.addCleanup(source.close)
        chunk = result.value["ranked_chunks"][0]
        self.assertTrue(chunk["text"].endswith(" before"))
        self.assertTrue(chunk["truncated"])
        pubmed, drugs = Mock(), Mock()
        reader = HealthBenchSourceReadToolBackend(source, pubmed, drugs)
        read = reader.invoke(ToolRequest("read_source", {"source_id": chunk["source_id"]})).value["evidence"][0]
        self.assertEqual(read["excerpt"], text)
        self.assertFalse(read["truncated"])
        self.assertEqual(read["source_id"], chunk["source_id"])
        self.assertEqual(read["version"], chunk["version"])
        following = reader.invoke(ToolRequest("read_source", {
            "source_id": chunk["source_id"], "offset": chunk["next_offset"],
        })).value["evidence"][0]
        self.assertEqual(following["excerpt"], text[500:])
        self.assertEqual(chunk["text"] + following["excerpt"], text)
        self.assertEqual(pubmed.mock_calls, [])
        self.assertEqual(drugs.mock_calls, [])

    def test_source_metadata_survives_medrag_agent_observation_and_receipt_projection(self):
        source, _, result = search_result("fixture " + "x" * 650)
        self.addCleanup(source.close)
        chunk = result.value["ranked_chunks"][0]
        raw = {"observation_status": "success", "executed_action": {
            "kind": "tool", "resource_id": SEARCH, "name": "search",
            "arguments": {"query": "fixture"},
        }, "result": result.value}
        visible = HealthBenchClinicalReactExecutionAdapter._model_visible_observations([raw])[0]
        projected = visible["result"]["evidence"][0]
        self.assertNotIn("ranked_chunks", visible["result"])
        self.assertIn("ranked_chunks", raw["result"])
        for field in PAGE_FIELDS:
            self.assertEqual(projected[field], chunk[field])
        self.assertEqual(projected["excerpt"], chunk["text"])
        self.assertEqual(projected, _healthbench_search_candidates([receipt(result)])[0][2])

    def test_v3_shared_director_projection_and_v4_downstream_keep_truncation_flags(self):
        source, _, result = search_result("fixture " + "x" * 650)
        self.addCleanup(source.close)
        message = UpstreamMessage(
            "producer", "node", "The retrieved fragment is incomplete.",
            source_execution_mode="react", artifact_version="synthetic-1",
            tool_receipts=(receipt(result),),
        )
        projected = _healthbench_v3_receipts(message, seen_sources=set(), char_budget=6000)
        metadata = projected["evidence_receipts"][0]["source_retrieval"][0]
        chunk = result.value["ranked_chunks"][0]
        for field in PAGE_FIELDS:
            self.assertEqual(metadata[field], chunk[field])
        downstream = request(SEARCH, output=True, upstream=(message,),
            artifact_communication_profile=ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V4)
        rendered = json.dumps(build_agent_messages(downstream), ensure_ascii=False)
        self.assertIn("medrag:book-1", rendered)
        self.assertIn("source_retrieval", rendered)
        self.assertIn("truncated", rendered)
        self.assertIn("next_offset", rendered)
        self.assertIn("retrieved-not-endorsed", rendered)

    def test_source_separated_store_retains_actual_excerpt_completeness(self):
        source, _, result = search_result("fixture " + "x" * 650)
        self.addCleanup(source.close)
        evidence = _healthbench_medrag_evidence(result.value)[0]
        with TemporaryDirectory(prefix="healthbench-snippet-test-") as temporary:
            store = HealthBenchKnowledgeStore(Path(temporary), [])
            try:
                self.assertTrue(store.ingest_evidence(evidence))
                found = store.search("medical_references", "fixture")["evidence"][0]
                self.assertEqual(found["excerpt"], evidence["excerpt"])
                for field in PAGE_FIELDS:
                    self.assertEqual(found[field], evidence[field])
                self.assertTrue(found["truncated"])
                self.assertEqual(store.counts()["drug_labels"], 0)
            finally:
                store.close()

    def test_historical_snippet_without_metadata_is_not_declared_complete(self):
        value = {"operation": "search", "query": "fixture",
                 "frozen_corpus": {"source": "MedRAG/textbooks"},
                 "ranked_chunks": [{"document_id": "book-1", "title": "Synthetic textbook",
                                    "text": "fixture " + "x" * 492, "rank": 1}]}
        projected = _healthbench_medrag_evidence(value)[0]
        self.assertNotIn("truncated", projected)
        self.assertNotIn("next_offset", projected)
        self.assertNotIn("total_characters", projected)

    def test_tool_description_preserves_names_and_keeps_reading_optional(self):
        source, registry, _ = search_result("fixture complete synthetic statement.")
        self.addCleanup(source.close)
        description = registry.require_capability(SEARCH).action_schemas["search"]["properties"]["query"]["description"]
        self.assertIn("Preserve the conversation's original names", description)
        self.assertIn("source evidence supports the equivalence", description)
        self.assertIn("not that the entity does not exist", description)
        self.assertIn("when that tool is available", description)


if __name__ == "__main__":
    unittest.main()
