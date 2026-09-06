"""Offline MeSH protocol fixtures; no benchmark or external HTTP requests."""

from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from src.interactive.healthbench_clinical_tools import _source_page
from src.interactive.healthbench_mesh import MESH_BASE_URL, MESH_SOURCE, MeSHClient


def descriptor(**overrides):
    # Compact JSON-LD shape confirmed at NLM's documented D015242.json URI;
    # literal content and IDs below are synthetic fixtures, not medical claims.
    return {
        "@id": "http://id.nlm.nih.gov/mesh/D000001",
        "@type": "http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor",
        "identifier": "D000001",
        "label": {"@language": "en", "@value": "Synthetic heading"},
        "preferredConcept": "http://id.nlm.nih.gov/mesh/M000001",
        "concept": ["http://id.nlm.nih.gov/mesh/M000002"],
        "preferredTerm": "http://id.nlm.nih.gov/mesh/T000001",
        "lastUpdated": "2026-01-02",
        **overrides,
    }


def concept(**overrides):
    return {
        "@id": "http://id.nlm.nih.gov/mesh/M000001",
        "identifier": "M000001",
        "label": {"@language": "en", "@value": "Synthetic concept"},
        "scopeNote": {"@language": "en", "@value": "Fixture vocabulary scope; no measured outcome."},
        "preferredTerm": "http://id.nlm.nih.gov/mesh/T000001",
        "term": ["http://id.nlm.nih.gov/mesh/T000002"],
        **overrides,
    }


class BytesResponse:
    def __init__(self, payload):
        self.payload, self.closed = payload, False

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


class FixtureOpener:
    def __init__(self, *, rows=None, records=None, error=None, raw=None):
        self.rows = [{"resource": "http://id.nlm.nih.gov/mesh/D000001", "label": "Synthetic heading"}] if rows is None else rows
        self.records = {"D000001": descriptor(), "M000001": concept()} if records is None else records
        self.error, self.raw = error, raw
        self.calls, self.responses = [], []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        path = urlparse(request.full_url).path
        value = self.rows if path.endswith("/lookup/descriptor") else self.records[path.rsplit("/", 1)[1].removesuffix(".json")]
        response = BytesResponse(self.raw if self.raw is not None else json.dumps(value).encode())
        self.responses.append(response)
        return response


class MeSHClientTests(unittest.TestCase):
    def test_search_uses_only_documented_descriptor_label_contains_lookup(self):
        opener = FixtureOpener()
        result = MeSHClient(opener=opener).search("  synthetic heading  ")
        self.assertEqual(result["operation"], "search")
        self.assertEqual(result["query"], "synthetic heading")
        item = result["evidence"][0]
        self.assertEqual(item["source_id"], "mesh:D000001")
        self.assertEqual(item["source_type"], "controlled_vocabulary")
        self.assertEqual(item["content_type"], "mesh_descriptor_label_match")
        self.assertEqual(item["title"], "Synthetic heading")
        self.assertIn("not clinical-effectiveness evidence", item["excerpt"])
        self.assertIn("no definition or synonym expansion", item["excerpt"])
        self.assertIn("does not guarantee trial-acronym", item["excerpt"])
        self.assertEqual(result["source_receipts"], [{
            "source_type": "controlled_vocabulary", "source": MESH_SOURCE,
            "status": "success", "result_count": 1, "error_type": None,
        }])
        request, timeout = opener.calls[0]
        self.assertEqual(urlparse(request.full_url).path, "/mesh/lookup/descriptor")
        self.assertEqual(parse_qs(urlparse(request.full_url).query), {
            "label": ["synthetic heading"], "match": ["contains"], "limit": ["3"],
        })
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(timeout, 8.0)
        self.assertTrue(opener.responses[0].closed)
        self.assertEqual(len(opener.calls), 1)

    def test_read_preserves_preferred_label_definition_concept_and_term_ids(self):
        opener = FixtureOpener()
        item = MeSHClient(opener=opener).read("mesh:D000001")
        self.assertEqual(item["title"], "Synthetic heading")
        self.assertEqual(item["source_id"], "mesh:D000001")
        self.assertEqual(item["source_type"], "controlled_vocabulary")
        self.assertEqual(item["date"], "2026-01-02")
        self.assertIsNone(item["version"])
        self.assertEqual(item["url"], f"{MESH_BASE_URL}/D000001")
        for value in ("Synthetic concept", "Fixture vocabulary scope; no measured outcome.",
                      "M000001", "M000002", "T000001", "T000002", '"@language": "en"'):
            self.assertIn(value, item["excerpt"])
        self.assertIn("Term IDs are not resolved synonym labels", item["excerpt"])
        self.assertEqual([call[0].full_url for call in opener.calls], [
            f"{MESH_BASE_URL}/D000001.json", f"{MESH_BASE_URL}/M000001.json",
        ])
        self.assertTrue(all(response.closed for response in opener.responses))

    def test_missing_scope_note_is_not_invented(self):
        missing = concept()
        missing.pop("scopeNote")
        opener = FixtureOpener(records={"D000001": descriptor(), "M000001": missing})
        item = MeSHClient(opener=opener).read("mesh:D000001")
        projection = json.loads(item["excerpt"][item["excerpt"].index("{"):])
        self.assertNotIn("scopeNote", projection["preferred_concept"])
        self.assertNotIn("Fixture vocabulary scope", item["excerpt"])
        self.assertIn("Missing fields are unavailable", item["excerpt"])

    def test_missing_preferred_concept_does_not_trigger_guessed_lookup(self):
        row = descriptor()
        row.pop("preferredConcept")
        opener = FixtureOpener(records={"D000001": row})
        item = MeSHClient(opener=opener).read("mesh:D000001")
        self.assertIn("Synthetic heading", item["excerpt"])
        self.assertEqual(len(opener.calls), 1)

    def test_full_definition_uses_existing_source_pagination_without_loss(self):
        note = "Synthetic vocabulary context. " * 1600
        opener = FixtureOpener(records={
            "D000001": descriptor(), "M000001": concept(scopeNote={"@value": note, "@language": "en"}),
        })
        item = MeSHClient(opener=opener).read("mesh:D000001")
        pages, offset = [], 0
        while True:
            page = _source_page(item, offset)
            pages.append(page["excerpt"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        self.assertGreater(len(pages), 1)
        self.assertEqual("".join(pages), item["excerpt"])
        self.assertIn(note, item["excerpt"])
        self.assertEqual(len(opener.calls), 2)

    def test_empty_search_receipt_is_success_not_a_medical_negative(self):
        result = MeSHClient(opener=FixtureOpener(rows=[])).search("synthetic heading")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["source_receipts"][0]["status"], "success")
        self.assertEqual(result["source_receipts"][0]["result_count"], 0)

    def test_search_is_bounded_deduplicated_and_keeps_server_order(self):
        rows = [
            {"resource": "http://id.nlm.nih.gov/mesh/D000003", "label": "Third"},
            {"resource": "http://id.nlm.nih.gov/mesh/D000003", "label": "Third"},
            {"resource": "http://id.nlm.nih.gov/mesh/D000001", "label": "First"},
            {"resource": "http://id.nlm.nih.gov/mesh/D000002", "label": "Second"},
        ]
        items = MeSHClient(opener=FixtureOpener(rows=rows)).search("synthetic heading")["evidence"]
        self.assertEqual([item["document_id"] for item in items], ["D000003", "D000001"])
        self.assertEqual([item["rank"] for item in items], [1, 2])

    def test_nondefault_limit_and_timeout_are_forwarded(self):
        opener = FixtureOpener()
        MeSHClient(retmax=1, timeout_seconds=2.5, opener=opener).search("synthetic heading")
        request, timeout = opener.calls[0]
        self.assertEqual(parse_qs(urlparse(request.full_url).query)["limit"], ["1"])
        self.assertEqual(timeout, 2.5)

    def test_invalid_source_ids_fail_before_http(self):
        opener = FixtureOpener()
        for source_id in (None, 42, "D000001", "mesh:M000001", "mesh:T000001", "mesh:Dabc", "mesh:D000001/extra"):
            with self.subTest(source_id=source_id), self.assertRaises(ValueError):
                MeSHClient(opener=opener).read(source_id)
        self.assertEqual(opener.calls, [])

    def test_descriptor_response_must_identify_requested_record(self):
        for row in (descriptor(**{"@id": "http://id.nlm.nih.gov/mesh/D000002"}),
                    descriptor(identifier="D000002")):
            with self.subTest(row=row), self.assertRaisesRegex(RuntimeError, "requested record"):
                MeSHClient(opener=FixtureOpener(records={"D000001": row})).read("mesh:D000001")

    def test_preferred_concept_response_must_identify_requested_record(self):
        opener = FixtureOpener(records={
            "D000001": descriptor(), "M000001": concept(**{"@id": "http://id.nlm.nih.gov/mesh/M000002"}),
        })
        with self.assertRaisesRegex(RuntimeError, "requested record"):
            MeSHClient(opener=opener).read("mesh:D000001")
        self.assertEqual(len(opener.calls), 2)

    def test_wrong_preferred_concept_reference_is_not_fetched(self):
        opener = FixtureOpener(records={"D000001": descriptor(preferredConcept="http://id.nlm.nih.gov/mesh/D000002")})
        with self.assertRaisesRegex(RuntimeError, "M resource ID"):
            MeSHClient(opener=opener).read("mesh:D000001")
        self.assertEqual(len(opener.calls), 1)

    def test_malformed_lookup_is_not_disguised_as_no_results(self):
        for rows in ({"error": "fixture"}, [None], [{"resource": "http://id.nlm.nih.gov/mesh/M000001", "label": "Concept"}],
                     [{"resource": "http://id.nlm.nih.gov/mesh/D000001", "label": ""}]):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                MeSHClient(opener=FixtureOpener(rows=rows)).search("synthetic heading")

    def test_missing_preferred_label_does_not_fall_back_to_query_or_id(self):
        opener = FixtureOpener(records={"D000001": descriptor(label=None)})
        with self.assertRaisesRegex(RuntimeError, "preferred label"):
            MeSHClient(opener=opener).read("mesh:D000001")
        self.assertEqual(len(opener.calls), 1)

    def test_transport_failure_is_not_retried_or_returned_as_empty_search(self):
        opener = FixtureOpener(error=HTTPError(MESH_BASE_URL, 503, "fixture unavailable", {}, None))
        with self.assertRaises(HTTPError):
            MeSHClient(opener=opener).search("synthetic heading")
        self.assertEqual(len(opener.calls), 1)

    def test_invalid_json_and_nonbytes_responses_are_closed(self):
        for payload, error_type in ((b"not-json", json.JSONDecodeError), ("[]", RuntimeError)):
            opener = FixtureOpener(raw=payload)
            with self.subTest(payload=payload), self.assertRaises(error_type):
                MeSHClient(opener=opener).search("synthetic heading")
            self.assertTrue(opener.responses[0].closed)

    def test_invalid_queries_fail_before_http(self):
        opener = FixtureOpener()
        for query in (None, 42, "", "  ", "x" * 161):
            with self.subTest(query=query), self.assertRaises(ValueError):
                MeSHClient(opener=opener).search(query)
        self.assertEqual(opener.calls, [])

    def test_invalid_configuration_is_rejected(self):
        for timeout in (0, -1, float("inf"), float("nan"), True, "8"):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                MeSHClient(timeout_seconds=timeout)
        for retmax in (0, 4, True, 1.5):
            with self.subTest(retmax=retmax), self.assertRaises(ValueError):
                MeSHClient(retmax=retmax)
        with self.assertRaises(TypeError):
            MeSHClient(opener=None)


if __name__ == "__main__":
    unittest.main()
