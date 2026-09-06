"""Offline Europe PMC protocol fixtures; no model or external HTTP requests."""

from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from src.interactive.healthbench_clinical_tools import _source_page
from src.interactive.healthbench_europe_pmc import (
    EUROPE_PMC_BASE_URL,
    EUROPE_PMC_SEARCH_EXCERPT_CHARACTERS,
    EuropePMCClient,
)


def record(**overrides):
    return {
        "id": "12345", "source": "MED", "pmid": "12345", "pmcid": "PMC111",
        "doi": "10.0000/fixture", "title": "Synthetic publication", "firstPublicationDate": "2026-01-02",
        "abstractText": "<h4>METHODS</h4><p>Fixture conditions &amp; outcome.</p>",
        "pubTypeList": {"pubType": ["Journal Article"]}, "isOpenAccess": "Y",
        **overrides,
    }


JATS = b"""<article article-type="research-article"><front><article-meta>
<article-id pub-id-type="pmc">111</article-id><article-id pub-id-type="pmid">12345</article-id>
<article-id pub-id-type="doi">10.0000/fixture</article-id>
<title-group><article-title>Synthetic publication</article-title></title-group>
<pub-date pub-type="epub"><day>02</day><month>01</month><year>2026</year></pub-date>
<abstract><p>Fixture abstract.</p></abstract></article-meta></front>
<body><sec><title>Results</title><p>Body condition.</p><sec><title>Limitations</title>
<p>Important uncertainty.</p></sec><table-wrap><caption>Data</caption><table><tr><td>42</td></tr></table>
</table-wrap></sec></body><back><ref-list><ref>Source citation.</ref></ref-list></back></article>"""


class BytesResponse:
    def __init__(self, payload):
        self.payload, self.closed = payload, False

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


class FixtureOpener:
    def __init__(self, rows=None, xml=JATS, error=None):
        self.rows = rows if rows is not None else [record()]
        self.xml, self.error = xml, error
        self.calls, self.responses = [], []

    def __call__(self, request, *, timeout):
        self.calls.append((request.full_url, timeout))
        if self.error:
            raise self.error
        path = urlparse(request.full_url).path
        payload = self.xml if path.endswith("/fullTextXML") else json.dumps({"resultList": {"result": self.rows}}).encode()
        response = BytesResponse(payload)
        self.responses.append(response)
        return response


class EuropePMCClientTests(unittest.TestCase):
    def test_search_preserves_standard_evidence_and_document_metadata(self):
        opener = FixtureOpener()
        result = EuropePMCClient(opener=opener).search("synthetic trial")
        self.assertEqual(result["operation"], "search")
        self.assertEqual(result["query"], "synthetic trial")
        evidence = result["evidence"][0]
        self.assertEqual(evidence["source_id"], "europepmc:MED:12345")
        self.assertEqual(evidence["full_text_source_id"], "pmc:PMC111")
        self.assertEqual(evidence["pmid"], "12345")
        self.assertEqual(evidence["pmcid"], "PMC111")
        self.assertEqual(evidence["doi"], "10.0000/fixture")
        self.assertEqual(evidence["date"], "2026-01-02")
        self.assertEqual(evidence["excerpt"], "METHODS Fixture conditions & outcome.")
        self.assertEqual(evidence["content_type"], "europepmc_abstract_not_full_text")
        self.assertEqual(result["source_receipts"][0]["result_count"], 1)
        url, timeout = opener.calls[0]
        self.assertTrue(url.startswith(EUROPE_PMC_BASE_URL))
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["pageSize"], ["3"])
        self.assertEqual(query["resultType"], ["core"])
        self.assertIn("NOT PUB_TYPE:Preprint", query["query"][0])
        self.assertEqual(timeout, 8.0)
        self.assertTrue(opener.responses[0].closed)
        self.assertEqual(len(opener.calls), 1)

    def test_search_truncation_is_explicit_and_read_restores_abstract(self):
        abstract = "Fixture data. " * 200
        opener = FixtureOpener([record(abstractText=abstract)])
        client = EuropePMCClient(opener=opener)
        snippet = client.search("synthetic trial")["evidence"][0]
        self.assertEqual(len(snippet["excerpt"]), EUROPE_PMC_SEARCH_EXCERPT_CHARACTERS)
        self.assertTrue(snippet["truncated"])
        self.assertEqual(snippet["next_offset"], EUROPE_PMC_SEARCH_EXCERPT_CHARACTERS)
        full = client.read(snippet["source_id"])
        self.assertEqual(full["excerpt"], abstract.strip())
        self.assertFalse(full["truncated"])
        self.assertEqual(parse_qs(urlparse(opener.calls[1][0]).query)["query"], ["EXT_ID:12345 AND SRC:MED"])

    def test_non_oa_record_with_pmcid_does_not_advertise_xml(self):
        row = record(isOpenAccess="N", fullTextUrlList={"fullTextUrl": [{"availability": "Free"}]})
        evidence = EuropePMCClient(opener=FixtureOpener([row])).search("synthetic trial")["evidence"][0]
        self.assertEqual(evidence["pmcid"], "PMC111")
        self.assertFalse(evidence["is_open_access"])
        self.assertIsNone(evidence["full_text_source_id"])

    def test_missing_abstract_is_not_mislabeled_as_article_text(self):
        client = EuropePMCClient(opener=FixtureOpener([record(abstractText=None)]))
        item = client.search("synthetic trial")["evidence"][0]
        self.assertEqual(item["excerpt"], "")
        self.assertEqual(item["content_type"], "publication_metadata_no_abstract")
        with self.assertRaisesRegex(RuntimeError, "no abstract"):
            client.read("europepmc:MED:12345")

    def test_default_filters_preprints_even_if_returned_by_server(self):
        rows = [record(id="PPR111", source="PPR", pubTypeList={"pubType": ["Preprint"]}), record()]
        result = EuropePMCClient(opener=FixtureOpener(rows)).search("synthetic trial")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertFalse(result["evidence"][0]["is_preprint"])

    def test_optional_preprints_are_labeled_and_follow_other_publications(self):
        rows = [record(id="PPR111", source="PPR", pubTypeList={"pubType": ["Preprint"]}), record()]
        items = EuropePMCClient(include_preprints=True, opener=FixtureOpener(rows)).search("synthetic trial")["evidence"]
        self.assertEqual([item["is_preprint"] for item in items], [False, True])
        self.assertEqual(items[1]["source_type"], "preprint")
        self.assertEqual([item["rank"] for item in items], [1, 2])

    def test_full_text_retains_body_limitations_tables_and_references(self):
        opener = FixtureOpener()
        item = EuropePMCClient(opener=opener).read("pmc:PMC111")
        self.assertEqual(item["content_type"], "pmc_open_access_jats_full_text")
        self.assertEqual(item["pmid"], "12345")
        self.assertEqual(item["pmcid"], "PMC111")
        self.assertEqual(item["date"], "2026-01-02")
        for text in ("Fixture abstract.", "Body condition.", "Important uncertainty.", "42", "Source citation."):
            self.assertIn(text, item["excerpt"])
        self.assertEqual(opener.calls[0][0], f"{EUROPE_PMC_BASE_URL}/PMC111/fullTextXML")
        self.assertEqual(len(opener.calls), 1)

    def test_caller_source_page_handles_full_text_without_silent_loss(self):
        xml = JATS.replace(b"Body condition.", b"Body condition. " * 1500)
        item = EuropePMCClient(opener=FixtureOpener(xml=xml)).read("pmc:PMC111")
        first = _source_page(item)
        self.assertTrue(first["truncated"])
        second = _source_page(item, first["next_offset"])
        self.assertEqual(first["excerpt"] + second["excerpt"], item["excerpt"])
        self.assertIsNone(second["next_offset"])

    def test_abstract_only_xml_is_not_called_full_text(self):
        xml = b'<article><front><article-meta><article-id pub-id-type="pmc">111</article-id><abstract>Only abstract.</abstract></article-meta></front></article>'
        with self.assertRaisesRegex(RuntimeError, "no body"):
            EuropePMCClient(opener=FixtureOpener(xml=xml)).read("pmc:PMC111")

    def test_exact_id_response_mismatch_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "requested record"):
            EuropePMCClient(opener=FixtureOpener([record(id="999")])).read("europepmc:MED:12345")
        with self.assertRaisesRegex(RuntimeError, "requested PMCID"):
            EuropePMCClient(opener=FixtureOpener(xml=JATS.replace(b">111<", b">999<"))).read("pmc:PMC111")

    def test_transport_failure_is_not_retried_or_disguised_as_no_evidence(self):
        error = HTTPError(EUROPE_PMC_BASE_URL, 503, "fixture unavailable", {}, None)
        opener = FixtureOpener(error=error)
        with self.assertRaises(HTTPError):
            EuropePMCClient(opener=opener).search("synthetic trial")
        self.assertEqual(len(opener.calls), 1)

    def test_empty_search_has_success_receipt_with_zero_results(self):
        result = EuropePMCClient(opener=FixtureOpener([])).search("synthetic trial")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["source_receipts"][0]["status"], "success")
        self.assertEqual(result["source_receipts"][0]["result_count"], 0)

    def test_record_count_is_bounded_and_duplicates_not_repeated(self):
        rows = [record(), record(), record(id="45678"), record(id="99999")]
        result = EuropePMCClient(opener=FixtureOpener(rows)).search("synthetic trial")
        self.assertEqual([item["document_id"] for item in result["evidence"]], ["12345", "45678"])


if __name__ == "__main__":
    unittest.main()
