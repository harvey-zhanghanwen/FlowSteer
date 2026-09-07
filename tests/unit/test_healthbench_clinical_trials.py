"""Offline ClinicalTrials.gov v2 schema fixtures; no benchmark or HTTP calls."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from src.interactive.healthbench_clinical_trials import (
    CLINICAL_TRIALS_BASE_URL,
    ClinicalTrialsClient,
)
from src.interactive.healthbench_clinical_tools import SOURCE_PAGE_CHARACTERS, _source_page


NCT_ID = "NCT01234567"
STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": NCT_ID, "briefTitle": "Synthetic study", "acronym": "FIXTURE"},
        "statusModule": {"overallStatus": "COMPLETED", "lastUpdatePostDateStruct": {"date": "2026-01-02", "type": "ACTUAL"}},
        "descriptionModule": {"briefSummary": "Synthetic registered study description."},
        "conditionsModule": {"conditions": ["Fixture condition"]},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"]},
        "armsInterventionsModule": {
            "armGroups": [{"label": "Group A", "type": "EXPERIMENTAL", "interventionNames": ["Drug: Fixture compound"]}],
            "interventions": [{"name": "Fixture compound", "type": "DRUG", "armGroupLabels": ["Group A"]}],
        },
        "outcomesModule": {
            "primaryOutcomes": [{"measure": "Registered measure", "timeFrame": "Day 10"}],
            "secondaryOutcomes": [{"measure": "Another measure", "timeFrame": "Day 20"}],
        },
        "eligibilityModule": {"eligibilityCriteria": "Synthetic inclusion and exclusion conditions.", "minimumAge": "18 Years"},
        "referencesModule": {"references": [{"pmid": "12345", "type": "RESULT", "citation": "Synthetic citation"}]},
    },
    "hasResults": False,
}
RESULTS = {
    "outcomeMeasuresModule": {"outcomeMeasures": [{
        "title": "Registered measure",
        "groups": [{"id": "OG000", "title": "Group A"}],
        "classes": [{"categories": [{"measurements": [{"groupId": "OG000", "value": "7"}]}]}],
    }]},
    "adverseEventsModule": {"eventGroups": [{"id": "EG000", "title": "Group A", "deathsNumAffected": 0}]},
    "moreInfoModule": {"limitationsAndCaveats": "Synthetic study limitation."},
}


class BytesResponse:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


class FixtureOpener:
    def __init__(self):
        self.calls = []
        self.responses = []
        self.studies = [deepcopy(STUDY)]
        self.record = deepcopy(STUDY)
        self.error = None
        self.override = None

    def __call__(self, request, *, timeout):
        self.calls.append((request.full_url, timeout))
        if self.error:
            raise self.error
        data = {"studies": self.studies} if urlparse(request.full_url).path.endswith("/studies") else self.record
        payload = self.override if self.override is not None else json.dumps(data).encode()
        response = BytesResponse(payload)
        self.responses.append(response)
        return response


class ClinicalTrialsClientTests(unittest.TestCase):
    def setUp(self):
        self.opener = FixtureOpener()
        self.client = ClinicalTrialsClient(opener=self.opener)

    def test_search_preserves_query_and_registered_fields_in_one_request(self):
        result = self.client.search("FIXTURE trial")
        self.assertEqual(result["operation"], "search")
        self.assertEqual(result["query"], "FIXTURE trial")
        self.assertEqual(len(self.opener.calls), 1)
        url, timeout = self.opener.calls[0]
        self.assertTrue(url.startswith(CLINICAL_TRIALS_BASE_URL + "/studies?"))
        self.assertEqual(parse_qs(urlparse(url).query), {"query.term": ["FIXTURE trial"], "pageSize": ["3"], "format": ["json"]})
        self.assertEqual(timeout, 8.0)
        self.assertTrue(self.opener.responses[0].closed)
        item = result["evidence"][0]
        self.assertEqual(item["source_id"], "clinicaltrials:" + NCT_ID)
        self.assertEqual(item["source_type"], "clinical_trial_registry")
        self.assertEqual(item["date"], "2026-01-02")
        self.assertEqual(item["content_type"], "clinical_trial_registry_protocol_no_posted_results")
        projected = json.loads(item["excerpt"].split("\n", 1)[1])
        self.assertEqual(projected, STUDY)
        self.assertEqual(result["source_receipts"][0]["result_count"], 1)
        self.assertFalse(item["truncated"])

    def test_read_retains_actual_results_without_claiming_publication_or_efficacy(self):
        self.opener.record["hasResults"] = True
        self.opener.record["resultsSection"] = deepcopy(RESULTS)
        item = self.client.read("clinicaltrials:" + NCT_ID)
        self.assertEqual(item["content_type"], "clinical_trial_registry_protocol_and_posted_results")
        self.assertIn("Registration does not establish efficacy", item["excerpt"])
        self.assertIn("not a journal article", item["excerpt"])
        projected = json.loads(item["excerpt"].split("\n", 1)[1])
        self.assertEqual(projected["resultsSection"], RESULTS)
        self.assertEqual(projected["protocolSection"], STUDY["protocolSection"])
        self.assertIn("/studies/" + NCT_ID, self.opener.calls[0][0])

    def test_absent_results_flag_is_not_reported_as_no_results(self):
        del self.opener.record["hasResults"]
        item = self.client.read(NCT_ID)
        self.assertEqual(item["content_type"], "clinical_trial_registry_protocol_results_status_unknown")
        self.assertIn('"hasResults": null', item["excerpt"])

    def test_results_flag_without_payload_is_distinct_from_posted_results_evidence(self):
        self.opener.record["hasResults"] = True
        item = self.client.read(NCT_ID)
        self.assertEqual(item["content_type"], "clinical_trial_registry_protocol_results_not_returned")
        self.assertNotIn('"resultsSection"', item["excerpt"])

    def test_search_uses_existing_explicit_pagination_and_read_recovers_full_record(self):
        record = deepcopy(STUDY)
        record["protocolSection"]["descriptionModule"]["briefSummary"] = "Fixture passage. " * 2000
        self.opener.studies = [record]
        self.opener.record = record
        first = self.client.search("fixture")["evidence"][0]
        self.assertEqual(len(first["excerpt"]), SOURCE_PAGE_CHARACTERS)
        self.assertTrue(first["truncated"])
        self.assertEqual(first["next_offset"], SOURCE_PAGE_CHARACTERS)
        full = self.client.read(first["source_id"])
        second = _source_page(full, first["next_offset"])
        self.assertEqual(first["excerpt"] + second["excerpt"], full["excerpt"][:2 * SOURCE_PAGE_CHARACTERS])
        self.assertEqual(first["total_characters"], len(full["excerpt"]))

    def test_empty_search_has_successful_zero_result_receipt(self):
        self.opener.studies = []
        result = self.client.search("fixture")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["source_receipts"], [{"source_type": "clinical_trial_registry", "source": "ClinicalTrials.gov", "status": "success", "result_count": 0, "error_type": None}])

    def test_duplicate_records_do_not_create_duplicate_evidence(self):
        self.opener.studies *= 3
        result = self.client.search("fixture")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["rank"], 1)

    def test_malformed_record_has_error_receipt_not_invented_evidence(self):
        self.opener.studies = [{"protocolSection": {}}, deepcopy(STUDY)]
        result = self.client.search("fixture")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["source_receipts"][0]["error_type"], "RuntimeError")

    def test_http_failure_does_not_retry_or_return_empty_success(self):
        self.opener.error = HTTPError(CLINICAL_TRIALS_BASE_URL, 503, "fixture unavailable", {}, None)
        with self.assertRaises(HTTPError):
            self.client.search("fixture")
        self.assertEqual(len(self.opener.calls), 1)

    def test_wrong_returned_id_is_not_labeled_as_requested_record(self):
        self.opener.record["protocolSection"]["identificationModule"]["nctId"] = "NCT01234568"
        with self.assertRaisesRegex(RuntimeError, "different NCT ID"):
            self.client.read(NCT_ID)

    def test_invalid_query_and_source_id_do_not_make_requests(self):
        for query in ("", "  ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.client.search(query)
        for source_id in ("NCT123", "pubmed:12345", None):
            with self.subTest(source_id=source_id), self.assertRaises(ValueError):
                self.client.read(source_id)
        self.assertEqual(self.opener.calls, [])

    def test_invalid_response_is_visible_and_response_is_closed(self):
        for payload in (b"[]", b"{}", b"invalid json"):
            self.opener.override = payload
            with self.subTest(payload=payload), self.assertRaises((RuntimeError, ValueError)):
                self.client.search("fixture")
            self.assertTrue(self.opener.responses[-1].closed)

    def test_configuration_bounds(self):
        for retmax in (0, 4, True):
            with self.subTest(retmax=retmax), self.assertRaises(ValueError):
                ClinicalTrialsClient(retmax=retmax)
        for timeout in (0, -1, float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                ClinicalTrialsClient(timeout_seconds=timeout)


if __name__ == "__main__":
    unittest.main()

