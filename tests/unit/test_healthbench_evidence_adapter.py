from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import re
import tempfile
import unittest
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

from jsonschema import Draft202012Validator

from src.interactive.healthbench_evidence_adapter import (
    HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
    NCBI_EUTILS_HOST,
    NCBI_EUTILS_TOOL_PARAMETER,
    PUBMED_HTTP_TIMEOUT_SECONDS,
    PUBMED_RETMAX,
    HealthBenchAuthoritativeReactExecutionAdapter,
    PubMedEUtilitiesClient,
    open_healthbench_authoritative_tool_registry,
)
from src.interactive.tool_runtime import StructuredAction, ToolRequest


FIXTURE_SOURCE_REVISION = "3" * 40


def _write_bm25_fixture(root: Path) -> tuple[str, ...]:
    documents = (
        "Aspirin irreversibly inhibits platelets and may cause gastrointestinal bleeding.",
        "Warfarin requires INR monitoring because anticoagulation increases bleeding risk.",
        "Insulin lowers blood glucose in patients with diabetes mellitus.",
    )
    inverted: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    for document_id, contents in enumerate(documents):
        term_frequency = Counter(re.findall(r"\b\w+\b", contents.lower()))
        document_lengths.append(sum(term_frequency.values()))
        for term, frequency in term_frequency.items():
            inverted[term].append((document_id, frequency))
        document_frequency.update(term_frequency)
    index = {
        "avg_dl": sum(document_lengths) / len(documents),
        "doc_lens": document_lengths,
        "idf": {
            term: math.log(
                1.0
                + (len(documents) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        },
        "inverted_index": dict(inverted),
    }
    with (root / "bm25_index.pkl").open("wb") as output:
        pickle.dump(index, output)
    with (root / "all_chunks.jsonl").open("w", encoding="utf-8") as output:
        for document_id, contents in enumerate(documents):
            output.write(
                json.dumps(
                    {
                        "contents": contents,
                        "id": f"textbook-{document_id}",
                        "title": f"Clinical Textbook {document_id}",
                    }
                )
                + "\n"
            )
    (root / ".source_revision").write_text(
        FIXTURE_SOURCE_REVISION + "\n", encoding="utf-8"
    )
    return documents


class _BytesResponse:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.closed = False

    def read(self) -> bytes:
        return self.value

    def close(self) -> None:
        self.closed = True


class _MockPubMedOpener:
    def __init__(self) -> None:
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float) -> _BytesResponse:
        self.calls.append((request, timeout))
        path = urlparse(request.full_url).path  # type: ignore[attr-defined]
        if path.endswith("/esearch.fcgi"):
            return _BytesResponse(
                json.dumps(
                    {"esearchresult": {"idlist": ["111", "222"]}}
                ).encode("utf-8")
            )
        if path.endswith("/efetch.fcgi"):
            return _BytesResponse(
                b"""<?xml version='1.0' encoding='UTF-8'?>
                <PubmedArticleSet>
                  <PubmedArticle>
                    <MedlineCitation><PMID>111</PMID><Article>
                      <ArticleTitle>Aspirin and <i>gastrointestinal</i> bleeding</ArticleTitle>
                      <Abstract><AbstractText>Evidence about aspirin bleeding risk.</AbstractText></Abstract>
                      <Journal><JournalIssue><PubDate><Year>2024</Year><Month>Jan</Month></PubDate></JournalIssue></Journal>
                    </Article></MedlineCitation>
                  </PubmedArticle>
                  <PubmedArticle>
                    <MedlineCitation><PMID>222</PMID><Article>
                      <ArticleTitle>Antiplatelet safety review</ArticleTitle>
                      <Abstract><AbstractText>Review of antiplatelet safety.</AbstractText></Abstract>
                      <ArticleDate><Year>2025</Year><Month>02</Month><Day>03</Day></ArticleDate>
                    </Article></MedlineCitation>
                  </PubmedArticle>
                </PubmedArticleSet>"""
            )
        raise AssertionError(f"unexpected mocked NCBI endpoint: {path}")


class _FailingPubMedOpener:
    def __call__(self, request: object, *, timeout: float) -> object:
        del request, timeout
        raise URLError("offline fixture")


def _successful_observation(query: str) -> dict[str, object]:
    return {
        "observation_status": "success",
        "executed_action": {
            "kind": "tool",
            "name": "search",
            "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
            "skill_id": None,
            "arguments": {"query": query},
        },
        "result": {"evidence": [{"excerpt": "public evidence"}]},
    }


class HealthBenchEvidenceAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.documents = _write_bm25_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _open(self, opener: object):
        return open_healthbench_authoritative_tool_registry(
            corpus_root=self.root,
            source_identity="fixture-medrag-textbooks",
            expected_source_revision=FIXTURE_SOURCE_REVISION,
            expected_rows=len(self.documents),
            pubmed_client=PubMedEUtilitiesClient(
                opener=opener,
                minimum_interval_seconds=0.0,
            ),
            timeout_seconds=1.0,
        )

    def test_single_tool_aggregates_medrag_and_mocked_pubmed_evidence(
        self,
    ) -> None:
        opener = _MockPubMedOpener()
        with self._open(opener) as opened:
            result = opened.registry.invoke(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                ToolRequest(
                    "search",
                    {"query": "aspirin gastrointestinal bleeding risk"},
                ),
            )

        self.assertEqual("search", result.value["operation"])
        evidence = result.value["evidence"]
        self.assertGreaterEqual(len(evidence), 3)
        self.assertEqual(list(range(1, len(evidence) + 1)), [
            item["rank"] for item in evidence
        ])
        self.assertEqual("frozen_medical_textbook", evidence[0]["source_type"])
        self.assertEqual("peer_reviewed_literature", evidence[1]["source_type"])
        self.assertEqual("111", evidence[1]["document_id"])
        self.assertEqual(
            "Aspirin and gastrointestinal bleeding",
            evidence[1]["title"],
        )
        self.assertEqual("2024-Jan", evidence[1]["date"])
        self.assertEqual(
            "https://pubmed.ncbi.nlm.nih.gov/111/",
            evidence[1]["url"],
        )
        self.assertEqual(
            ["success", "success"],
            [item["status"] for item in result.value["source_receipts"]],
        )
        self.assertEqual(2, len(opener.calls))

        for request, timeout in opener.calls:
            parsed = urlparse(request.full_url)  # type: ignore[attr-defined]
            parameters = parse_qs(parsed.query)
            self.assertEqual("https", parsed.scheme)
            self.assertEqual(NCBI_EUTILS_HOST, parsed.hostname)
            self.assertEqual(["pubmed"], parameters["db"])
            self.assertEqual(
                [NCBI_EUTILS_TOOL_PARAMETER], parameters["tool"]
            )
            self.assertEqual(PUBMED_HTTP_TIMEOUT_SECONDS, timeout)
        search_parameters = parse_qs(
            urlparse(opener.calls[0][0].full_url).query  # type: ignore[attr-defined]
        )
        self.assertEqual([str(PUBMED_RETMAX)], search_parameters["retmax"])

    def test_pubmed_failure_preserves_local_evidence_and_receipt(self) -> None:
        with self._open(_FailingPubMedOpener()) as opened:
            result = opened.registry.invoke(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                ToolRequest("search", {"query": "aspirin bleeding"}),
            )

        self.assertTrue(result.value["evidence"])
        self.assertTrue(all(
            item["source_type"] == "frozen_medical_textbook"
            for item in result.value["evidence"]
        ))
        pubmed_receipt = result.value["source_receipts"][1]
        self.assertEqual("error", pubmed_receipt["status"])
        self.assertEqual("URLError", pubmed_receipt["error_type"])

    def test_schema_rejects_evaluator_only_fields(self) -> None:
        with self._open(_MockPubMedOpener()) as opened:
            capability = opened.registry.require_capability(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            )
            for forbidden in (
                "task_id",
                "rubric",
                "reference_response",
                "ground_truth",
                "evaluator_payload",
            ):
                with self.assertRaises(ValueError):
                    opened.registry.invoke(
                        HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                        ToolRequest(
                            "search",
                            {"query": "aspirin bleeding", forbidden: "hidden"},
                        ),
                    )

        description = capability.input_schema["properties"]["query"][
            "description"
        ]
        self.assertIn("English clinical terminology", description)
        evidence_schema = capability.output_schema["properties"]["evidence"][
            "items"
        ]
        self.assertEqual(
            {
                "source_type",
                "source",
                "document_id",
                "title",
                "date",
                "url",
                "excerpt",
                "rank",
            },
            set(evidence_schema["required"]),
        )

    def test_react_state_allows_one_complementary_search_then_completion(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=3,
            )
            initial = adapter._state_conditioned_action_domain(None, [])
            first = _successful_observation("aspirin gastrointestinal bleeding")
            after_first = adapter._state_conditioned_action_domain(None, [first])
            second = _successful_observation("aspirin renal safety chronic disease")
            after_second = adapter._state_conditioned_action_domain(
                None, [first, second]
            )

        search = frozenset(
            {(HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID, "search")}
        )
        self.assertEqual((search, True), initial)
        self.assertEqual((search, True), after_first)
        self.assertEqual((frozenset(), True), after_second)

    def test_react_rejects_exact_and_near_duplicate_but_allows_novel_query(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        prior = _successful_observation("aspirin gastrointestinal bleeding")
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=2,
            )
            exact = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": " ASPIRIN   gastrointestinal bleeding "},
                }
            )
            near = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {
                        "query": "gastrointestinal bleeding from aspirin"
                    },
                }
            )
            novel = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": "aspirin renal safety chronic disease"},
                }
            )
            exact_error = adapter._tool_action_error(
                request=None, action=exact, observations=[prior]
            )
            near_error = adapter._tool_action_error(
                request=None, action=near, observations=[prior]
            )
            novel_error = adapter._tool_action_error(
                request=None, action=novel, observations=[prior]
            )

        self.assertEqual("duplicate_tool_request", exact_error)
        self.assertEqual("duplicate_tool_request", near_error)
        self.assertIsNone(novel_error)

    def test_multi_branch_schema_is_strict_oneof_and_initial_search_is_optional(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=2,
            )
            schema = adapter._state_conditioned_response_schema(None, [])
            required_adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=2,
                require_initial_search=True,
            )
            required_schema = required_adapter._state_conditioned_response_schema(
                None, []
            )

        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertEqual(2, len(schema["oneOf"]))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(
            {
                "kind": "tool",
                "name": "search",
                "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "acute kidney injury management"},
            }
        )
        validator.validate(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": "Complete public assistant response."},
            }
        )
        self.assertNotIn("oneOf", required_schema)
        self.assertEqual(
            "tool", required_schema["properties"]["kind"]["const"]
        )
        self.assertEqual(
            (
                frozenset(
                    {(HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID, "search")}
                ),
                False,
            ),
            required_adapter._state_conditioned_action_domain(None, []),
        )

    def test_tool_budget_masks_search_even_before_two_successes(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        failed = {
            "observation_status": "tool_error",
            "executed_action": {
                "kind": "tool",
                "name": "search",
                "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "acute kidney injury"},
            },
        }
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=3,
                max_tool_calls=1,
            )
            at_budget = adapter._state_conditioned_action_domain(None, [failed])

        self.assertEqual((frozenset(), True), at_budget)


if __name__ == "__main__":
    unittest.main()
