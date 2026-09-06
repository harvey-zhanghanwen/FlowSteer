"""Offline tool contracts; synthetic documents, no benchmark or HTTP calls."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from jsonschema import Draft202012Validator

from src.interactive.computation_tools import AIMEComputationToolBackend
from src.interactive.healthbench_clinical_tools import (
    DAILYMED_BASE_URL,
    HEALTHBENCH_CALCULATOR_TOOL_ID,
    HEALTHBENCH_DRUG_LOOKUP_TOOL_ID,
    HEALTHBENCH_SOURCE_READ_TOOL_ID,
    SOURCE_PAGE_CHARACTERS,
    DailyMedClient,
    build_healthbench_clinical_tool_registry,
    open_healthbench_clinical_tool_registry,
)
from src.interactive.healthbench_evidence_adapter import (
    HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
    PubMedEUtilitiesClient,
    build_healthbench_authoritative_tool_registry,
)
from src.interactive.healthbench_tool_adapter import (
    HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
    FrozenMedRAGBM25Corpus,
    build_healthbench_medrag_tool_registry,
)
from src.interactive.tool_runtime import ToolRequest


SETID = "12345678-1111-2222-3333-123456789abc"
OTHER_SETID = "12345678-1111-2222-3333-123456789abd"
LABEL_XML = f"""<document xmlns="urn:hl7-org:v3">
<setId root="{SETID}"/><versionNumber value="7"/><effectiveTime value="20260901"/>
<title>SYNTHETIC LABEL — formulation A</title>
<component><structuredBody><component><section>
<title>EXAMPLE SECTION</title><text><paragraph>Fixture source paragraph.</paragraph>
<paragraph>Second paragraph with <content>explicit condition</content>.</paragraph></text>
<component><section><title>SUBSECTION</title><text>Fixture exception.</text></section></component>
</section></component></structuredBody></component></document>""".encode()
ABSTRACT = "Public abstract information. " * 80
PUBMED_XML = ("<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID>"
              "<Article><ArticleTitle>Fixture article</ArticleTitle>"
              "<Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>"
              f"<Abstract><AbstractText Label='METHOD'>{ABSTRACT}</AbstractText>"
              "<AbstractText Label='RESULT'>Fixture result.</AbstractText></Abstract>"
              "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>").encode()


class BytesResponse:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.closed = False

    def read(self) -> bytes:
        return self.value

    def close(self) -> None:
        self.closed = True


class FixtureOpener:
    def __init__(self) -> None:
        self.calls = []
        self.responses = []
        self.records = [{"setid": SETID, "spl_version": "7", "published_date": "Sep 02, 2026", "title": "Metadata is not evidence"}]
        self.label = LABEL_XML
        self.pubmed = PUBMED_XML
        self.fail_label = False

    def __call__(self, request, *, timeout):
        self.calls.append((request.full_url, timeout))
        path = urlparse(request.full_url).path
        if path.endswith("spls.json"):
            payload = json.dumps({"metadata": {"total_elements": "1"}, "data": self.records}).encode()
        elif path.endswith(".xml"):
            if self.fail_label:
                raise HTTPError(request.full_url, 503, "fixture unavailable", {}, None)
            payload = self.label
        elif path.endswith("efetch.fcgi"):
            payload = self.pubmed
        elif path.endswith("esearch.fcgi"):
            payload = b'{"esearchresult":{"idlist":[]}}'
        else:
            raise AssertionError(f"unexpected fixture path: {path}")
        response = BytesResponse(payload)
        self.responses.append(response)
        return response


def fixture_corpus(text: str | None = None) -> FrozenMedRAGBM25Corpus:
    return FrozenMedRAGBM25Corpus(
        source_identity="synthetic-textbook", source_revision="fixture-v1", corpus_rows=1,
        _corpus=(text or ("Fixture evidence with conditions. " * 40),),
        _document_ids=("book-1",), _titles=("Fixture textbook",),
        _index={"idf": {"fixture": 2.0}, "inverted_index": {"fixture": [(0, 1)]}, "doc_lens": [10], "avg_dl": 10},
    )


class ClinicalToolsTests(unittest.TestCase):
    def setUp(self):
        self.opener = FixtureOpener()
        self.corpus = fixture_corpus()
        self.pubmed = PubMedEUtilitiesClient(opener=self.opener, minimum_interval_seconds=0)
        self.drugs = DailyMedClient(opener=self.opener)
        self.registry = build_healthbench_clinical_tool_registry(self.corpus, pubmed_client=self.pubmed, drug_client=self.drugs)

    def invoke(self, tool_id, action, arguments):
        value = self.registry.invoke(tool_id, ToolRequest(action, arguments)).value
        Draft202012Validator(self.registry.require_capability(tool_id).output_schema).validate(value)
        return value

    def test_exact_five_optional_tools_no_fixed_roles_or_patient_environment(self):
        self.assertEqual(set(self.registry.resource_ids), {
            HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID, HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
            HEALTHBENCH_SOURCE_READ_TOOL_ID, HEALTHBENCH_DRUG_LOOKUP_TOOL_ID,
            HEALTHBENCH_CALCULATOR_TOOL_ID,
        })
        for capability in self.registry.capabilities:
            self.assertEqual(capability.dataset_scope, ("healthbench_professional",))
            self.assertEqual(capability.side_effect, "none")

    def test_existing_authoritative_and_medrag_registrations_are_unchanged(self):
        for registry in (
            build_healthbench_authoritative_tool_registry(self.corpus, pubmed_client=self.pubmed, timeout_seconds=30),
            build_healthbench_medrag_tool_registry(self.corpus, timeout_seconds=30),
        ):
            for tool_id in registry.resource_ids:
                self.assertEqual(self.registry.require_capability(tool_id), registry.require_capability(tool_id))
                expected = registry.invoke(tool_id, ToolRequest("search", {"query": "fixture"})).value
                self.assertEqual(self.invoke(tool_id, "search", {"query": "fixture"}), expected)

    def test_medrag_read_preserves_full_chunk_not_search_excerpt(self):
        search = self.invoke(HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID, "search", {"query": "fixture"})
        self.assertEqual(len(search["ranked_chunks"][0]["text"]), 500)
        result = self.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", {"source_id": "book-1"})
        evidence = result["evidence"][0]
        self.assertEqual(evidence["excerpt"], self.corpus._corpus[0])
        self.assertEqual(evidence["source_id"], "medrag:book-1")
        self.assertEqual(evidence["version"], "fixture-v1")
        self.assertFalse(evidence["truncated"])
        self.assertEqual(len(self.opener.calls), 0)
        self.assertEqual(self.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", {"source_id": "medrag:book-1"})["evidence"], result["evidence"])

    def test_source_paging_explicitly_retains_continuation(self):
        self.corpus._corpus = ("x" * (SOURCE_PAGE_CHARACTERS + 13),)
        first = self.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", {"source_id": "book-1"})["evidence"][0]
        self.assertTrue(first["truncated"])
        self.assertEqual(first["next_offset"], SOURCE_PAGE_CHARACTERS)
        second = self.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", {"source_id": first["source_id"], "offset": first["next_offset"]})["evidence"][0]
        self.assertEqual(first["excerpt"] + second["excerpt"], self.corpus._corpus[0])
        self.assertFalse(second["truncated"])
        self.assertIsNone(second["next_offset"])

    def test_pubmed_exact_id_read_retains_complete_abstract_and_labels(self):
        result = self.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", {"source_id": "pubmed:12345"})
        item = result["evidence"][0]
        self.assertGreater(len(item["excerpt"]), 1200)
        self.assertIn("METHOD:", item["excerpt"])
        self.assertIn("RESULT: Fixture result.", item["excerpt"])
        self.assertEqual(item["content_type"], "pubmed_abstract_not_full_text")
        self.assertEqual(item["date"], "2026")
        self.assertEqual(len(self.opener.calls), 1)
        params = parse_qs(urlparse(self.opener.calls[0][0]).query)
        self.assertEqual(params["id"], ["12345"])
        self.assertEqual(params["rettype"], ["abstract"])
        self.assertTrue(self.opener.responses[0].closed)

    def test_pubmed_without_abstract_is_transparent_tool_error(self):
        self.opener.pubmed = b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID></MedlineCitation></PubmedArticle></PubmedArticleSet>"
        result, receipt = asyncio.run(self.registry.ainvoke_with_receipt(HEALTHBENCH_SOURCE_READ_TOOL_ID, ToolRequest("read_source", {"source_id": "12345"})))
        self.assertIsNone(result)
        self.assertEqual(receipt.error_type, "RuntimeError")

    def test_drug_lookup_fetches_label_text_with_versions_and_dates(self):
        result = self.invoke(HEALTHBENCH_DRUG_LOOKUP_TOOL_ID, "drug_lookup", {"drug_name": "Fixture Drug"})
        self.assertEqual(result["query"], "Fixture Drug")
        item = result["evidence"][0]
        self.assertIn("Fixture source paragraph.", item["excerpt"])
        self.assertIn("explicit condition", item["excerpt"])
        self.assertEqual(item["excerpt"].count("Fixture exception."), 1)
        self.assertNotIn("Metadata is not evidence", item["excerpt"])
        self.assertEqual(item["version"], "7")
        self.assertEqual(item["date"], "20260901")
        self.assertEqual(item["published_date"], "Sep 02, 2026")
        self.assertEqual(item["source_id"], f"dailymed:{SETID}")
        self.assertEqual(len(self.opener.calls), 2)
        self.assertTrue(all(url.startswith(DAILYMED_BASE_URL) for url, _ in self.opener.calls))
        params = parse_qs(urlparse(self.opener.calls[0][0]).query)
        self.assertEqual(params["drug_name"], ["Fixture Drug"])
        self.assertEqual(params["pagesize"], ["2"])
        self.assertTrue(all(response.closed for response in self.opener.responses))

    def test_daily_med_exact_source_id_is_readable(self):
        item = self.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", {"source_id": SETID})["evidence"][0]
        self.assertEqual(item["source_id"], f"dailymed:{SETID}")
        self.assertEqual(item["content_type"], "structured_product_label_sections")

    def test_failed_label_is_not_presented_as_metadata_evidence(self):
        self.opener.fail_label = True
        result = self.invoke(HEALTHBENCH_DRUG_LOOKUP_TOOL_ID, "drug_lookup", {"drug_name": "Fixture Drug"})
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["source_receipts"][0]["status"], "error")
        self.assertEqual(result["source_receipts"][0]["error_type"], "HTTPError")

    def test_wrong_product_or_empty_label_is_not_admitted(self):
        for label in (LABEL_XML.replace(SETID.encode(), OTHER_SETID.encode()), f'<document xmlns="urn:hl7-org:v3"><setId root="{SETID}"/></document>'.encode()):
            with self.subTest(label=label[:30]):
                self.opener.label = label
                result = self.invoke(HEALTHBENCH_DRUG_LOOKUP_TOOL_ID, "drug_lookup", {"drug_name": "Fixture Drug"})
                self.assertEqual(result["evidence"], [])
                self.assertEqual(result["source_receipts"][0]["error_type"], "RuntimeError")

    def test_no_matching_label_is_empty_not_clinical_conclusion(self):
        self.opener.records = []
        result = self.invoke(HEALTHBENCH_DRUG_LOOKUP_TOOL_ID, "drug_lookup", {"drug_name": "Fixture Drug"})
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["source_receipts"][0]["result_count"], 0)
        self.assertEqual(len(self.opener.calls), 1)

    def test_calculator_is_actual_existing_skillflow_backend(self):
        self.assertIsInstance(self.registry._backend(HEALTHBENCH_CALCULATOR_TOOL_ID), AIMEComputationToolBackend)
        result = self.invoke(HEALTHBENCH_CALCULATOR_TOOL_ID, "calculator", {"expression": "2 * 3 + 1"})
        self.assertEqual(result, {"action": "calculator", "ok": True, "observation": "[RESULT] 2 * 3 + 1 = 7"})
        self.assertEqual(self.opener.calls, [])

    def test_invalid_read_arguments_do_not_dispatch_http(self):
        for args in ({"source_id": "unknown"}, {"source_id": "book-1", "offset": -1}, {"source_id": "book-1", "offset": True}, {"source_id": "book-1", "offset": 1000000}, {"source_id": "pubmed:bad"}, {"source_id": "book-1", "extra": "unused"}):
            with self.subTest(args=args), self.assertRaises((ValueError, LookupError)):
                self.registry.invoke(HEALTHBENCH_SOURCE_READ_TOOL_ID, ToolRequest("read_source", args))
        self.assertEqual(self.opener.calls, [])

    def test_owned_resource_lifecycle_is_reused(self):
        with patch.object(FrozenMedRAGBM25Corpus, "open", return_value=self.corpus):
            with open_healthbench_clinical_tool_registry(corpus_root="unused", source_identity="fixture", expected_source_revision="fixture", expected_rows=1, pubmed_client=self.pubmed, drug_client=self.drugs) as resources:
                self.assertEqual(resources.registry.resource_ids, self.registry.resource_ids)
                self.assertFalse(resources.closed)
            self.assertTrue(resources.closed)
            self.assertTrue(self.corpus.closed)

    def test_registry_build_failure_closes_owned_corpus(self):
        with patch.object(FrozenMedRAGBM25Corpus, "open", return_value=self.corpus), self.assertRaises(ValueError):
            open_healthbench_clinical_tool_registry(corpus_root="unused", source_identity="fixture", expected_source_revision="fixture", expected_rows=1, timeout_seconds=-1)
        self.assertTrue(self.corpus.closed)


if __name__ == "__main__":
    unittest.main()
