from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import re
import socket
import tempfile
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from jsonschema import Draft202012Validator

from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    ExecutionPhase,
)
from src.interactive.healthbench_evidence_adapter import (
    AUTHORITATIVE_QUERY_MAX_CHARACTERS,
    AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
    HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
    HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION,
    HEALTHBENCH_COMPLETION_ARTIFACT_MAX_CHARACTERS_V1,
    HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2,
    NCBI_EUTILS_HOST,
    NCBI_EUTILS_TOOL_PARAMETER,
    PUBMED_HTTP_TIMEOUT_SECONDS,
    PUBMED_RETMAX,
    HealthBenchAuthoritativeReactExecutionAdapter,
    PubMedEUtilitiesClient,
    open_healthbench_authoritative_tool_registry,
)
from src.interactive.model_registry import ModelSpec, ProviderSpec
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


class _ScriptedPubMedOpener:
    """Return or raise one local scripted outcome per transport attempt."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float) -> object:
        self.calls.append((request, timeout))
        if not self.outcomes:
            raise AssertionError("scripted PubMed opener exhausted")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FailOnceThenMockPubMedOpener:
    def __init__(self) -> None:
        self.calls: list[tuple[object, float]] = []
        self.delegate = _MockPubMedOpener()

    def __call__(self, request: object, *, timeout: float) -> object:
        self.calls.append((request, timeout))
        if len(self.calls) == 1:
            raise URLError("transient offline fixture")
        return self.delegate(request, timeout=timeout)


def _http_error(status: int) -> HTTPError:
    return HTTPError(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        status,
        f"fixture HTTP {status}",
        hdrs=None,
        fp=None,
    )


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


def _search_observation(
    query: str,
    *,
    title: str,
    excerpt: str,
) -> dict[str, object]:
    return {
        "observation_status": "success",
        "executed_action": {
            "kind": "tool",
            "name": "search",
            "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
            "skill_id": None,
            "arguments": {"query": query},
        },
        "result": {
            "evidence": [{"title": title, "excerpt": excerpt}],
        },
    }


def _agent_request(problem: str) -> AgentRequest:
    provider = ProviderSpec("fixture-provider", kind="test")
    model = ModelSpec("fixture-model", provider.provider_id)
    return AgentRequest(
        request_id="fixture:request",
        run_id="fixture",
        graph_revision=0,
        problem=problem,
        agent=AgentNode("agent", model.model_id, "Search public evidence."),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
    )


class HealthBenchEvidenceAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.documents = _write_bm25_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _open(
        self,
        opener: object,
        *,
        max_query_content_tokens: int = (
            AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS
        ),
    ):
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
            max_query_content_tokens=max_query_content_tokens,
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
        self.assertGreater(evidence[0]["score"], 0.0)
        self.assertIn("aspirin", evidence[0]["matched_terms"])
        self.assertEqual("peer_reviewed_literature", evidence[1]["source_type"])
        self.assertNotIn("score", evidence[1])
        self.assertNotIn("matched_terms", evidence[1])
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
        self.assertTrue(all(
            isinstance(item.get("score"), float)
            and isinstance(item.get("matched_terms"), list)
            for item in result.value["evidence"]
        ))
        pubmed_receipt = result.value["source_receipts"][1]
        self.assertEqual("error", pubmed_receipt["status"])
        self.assertEqual("URLError", pubmed_receipt["error_type"])

    def test_pubmed_historical_default_does_not_retry_transport_failure(
        self,
    ) -> None:
        opener = _ScriptedPubMedOpener(
            URLError("transient fixture"),
            _BytesResponse(b"would have succeeded"),
        )
        sleeps: list[float] = []
        client = PubMedEUtilitiesClient(
            opener=opener,
            minimum_interval_seconds=0.0,
            sleep=sleeps.append,
        )

        with self.assertRaises(URLError):
            client._get("esearch.fcgi", {"term": "aspirin bleeding"})

        self.assertEqual(0, client.max_retries)
        self.assertEqual(1, len(opener.calls))
        self.assertEqual([], sleeps)

    def test_pubmed_retries_only_supported_transport_failures(self) -> None:
        retryable_errors: tuple[BaseException, ...] = (
            URLError("temporary DNS fixture"),
            TimeoutError("temporary timeout fixture"),
            socket.timeout("temporary socket timeout fixture"),
            *(
                _http_error(status)
                for status in (408, 409, 425, 429, 500, 503, 599)
            ),
        )
        for error in retryable_errors:
            with self.subTest(
                error=type(error).__name__,
                status=getattr(error, "code", None),
            ):
                opener = _ScriptedPubMedOpener(
                    error,
                    _BytesResponse(b"retry succeeded"),
                )
                sleeps: list[float] = []
                client = PubMedEUtilitiesClient(
                    opener=opener,
                    minimum_interval_seconds=0.0,
                    max_retries=1,
                    retry_backoff_seconds=0.25,
                    sleep=sleeps.append,
                )

                payload = client._get(
                    "esearch.fcgi", {"term": "aspirin bleeding"}
                )

                self.assertEqual(b"retry succeeded", payload)
                self.assertEqual(2, len(opener.calls))
                self.assertEqual([0.25], sleeps)

    def test_pubmed_does_not_retry_other_http_or_application_failures(
        self,
    ) -> None:
        non_retryable_errors: tuple[BaseException, ...] = (
            _http_error(400),
            _http_error(403),
            _http_error(600),
            RuntimeError("invalid response fixture"),
            ValueError("invalid payload fixture"),
        )
        for error in non_retryable_errors:
            with self.subTest(
                error=type(error).__name__,
                status=getattr(error, "code", None),
            ):
                opener = _ScriptedPubMedOpener(
                    error,
                    _BytesResponse(b"must not be reached"),
                )
                sleeps: list[float] = []
                client = PubMedEUtilitiesClient(
                    opener=opener,
                    minimum_interval_seconds=0.0,
                    max_retries=2,
                    retry_backoff_seconds=0.25,
                    sleep=sleeps.append,
                )

                with self.assertRaises(type(error)):
                    client._get(
                        "esearch.fcgi", {"term": "aspirin bleeding"}
                    )

                self.assertEqual(1, len(opener.calls))
                self.assertEqual([], sleeps)

    def test_pubmed_retry_uses_bounded_exponential_backoff(self) -> None:
        opener = _ScriptedPubMedOpener(
            URLError("first fixture"),
            URLError("second fixture"),
            URLError("third fixture"),
            URLError("fourth fixture"),
            URLError("fifth fixture"),
            _BytesResponse(b"eventual success"),
        )
        sleeps: list[float] = []
        client = PubMedEUtilitiesClient(
            opener=opener,
            minimum_interval_seconds=0.0,
            max_retries=5,
            retry_backoff_seconds=0.1,
            sleep=sleeps.append,
        )

        payload = client._get(
            "esearch.fcgi", {"term": "aspirin bleeding"}
        )

        self.assertEqual(b"eventual success", payload)
        self.assertEqual([0.1, 0.2, 0.4, 0.4, 0.4], sleeps)

    def test_retry_policy_version_and_receipt_match_eventual_success(
        self,
    ) -> None:
        opener = _FailOnceThenMockPubMedOpener()
        sleeps: list[float] = []
        client = PubMedEUtilitiesClient(
            opener=opener,
            minimum_interval_seconds=0.0,
            max_retries=1,
            retry_backoff_seconds=0.125,
            sleep=sleeps.append,
        )
        with open_healthbench_authoritative_tool_registry(
            corpus_root=self.root,
            source_identity="fixture-medrag-textbooks",
            expected_source_revision=FIXTURE_SOURCE_REVISION,
            expected_rows=len(self.documents),
            pubmed_client=client,
            timeout_seconds=1.0,
        ) as opened:
            capability = opened.registry.require_capability(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            )
            result = opened.registry.invoke(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                ToolRequest("search", {"query": "aspirin bleeding"}),
            )

        self.assertEqual(
            f"{HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION}+"
            f"medrag-{FIXTURE_SOURCE_REVISION}+"
            "pubmed-eutils-retmax-3+"
            "pubmed-transport-retry-max-1-exp2-base-0.125s",
            capability.version,
        )
        self.assertEqual([0.125], sleeps)
        self.assertEqual(3, len(opener.calls))
        pubmed_receipt = result.value["source_receipts"][1]
        self.assertEqual("success", pubmed_receipt["status"])
        self.assertEqual(2, pubmed_receipt["result_count"])
        self.assertIsNone(pubmed_receipt["error_type"])

    def test_retry_exhaustion_keeps_truthful_error_receipt(self) -> None:
        opener = _ScriptedPubMedOpener(
            URLError("first fixture"),
            URLError("second fixture"),
            URLError("final fixture"),
        )
        sleeps: list[float] = []
        client = PubMedEUtilitiesClient(
            opener=opener,
            minimum_interval_seconds=0.0,
            max_retries=2,
            retry_backoff_seconds=0.1,
            sleep=sleeps.append,
        )
        with open_healthbench_authoritative_tool_registry(
            corpus_root=self.root,
            source_identity="fixture-medrag-textbooks",
            expected_source_revision=FIXTURE_SOURCE_REVISION,
            expected_rows=len(self.documents),
            pubmed_client=client,
            timeout_seconds=1.0,
        ) as opened:
            result = opened.registry.invoke(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                ToolRequest("search", {"query": "aspirin bleeding"}),
            )

        self.assertEqual([0.1, 0.2], sleeps)
        self.assertEqual(3, len(opener.calls))
        self.assertTrue(result.value["evidence"])
        pubmed_receipt = result.value["source_receipts"][1]
        self.assertEqual("error", pubmed_receipt["status"])
        self.assertEqual(0, pubmed_receipt["result_count"])
        self.assertEqual("URLError", pubmed_receipt["error_type"])

    def test_pubmed_retry_configuration_validation(self) -> None:
        for invalid in (-1, 1.5, True):
            with self.subTest(max_retries=invalid):
                with self.assertRaises(ValueError):
                    PubMedEUtilitiesClient(max_retries=invalid)  # type: ignore[arg-type]
        for invalid in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(retry_backoff_seconds=invalid):
                with self.assertRaises(ValueError):
                    PubMedEUtilitiesClient(
                        retry_backoff_seconds=invalid,  # type: ignore[arg-type]
                    )
        with self.assertRaises(TypeError):
            PubMedEUtilitiesClient(sleep=None)  # type: ignore[arg-type]

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
        self.assertIn("12 clinical content terms", description)
        self.assertNotIn("pubmed-transport-retry", capability.version)
        self.assertEqual(
            AUTHORITATIVE_QUERY_MAX_CHARACTERS,
            capability.input_schema["properties"]["query"]["maxLength"],
        )
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
        self.assertEqual("number", evidence_schema["properties"]["score"]["type"])
        self.assertEqual(
            {"type": "array", "items": {"type": "string"}},
            evidence_schema["properties"]["matched_terms"],
        )

    def test_backend_rejects_empty_or_over_broad_content_queries(self) -> None:
        with self._open(_MockPubMedOpener()) as opened:
            for query in (
                "and the or with",
                (
                    "pediatric pulmonary hypertension diagnosis management "
                    "treatment algorithm screening catheterization vasodilator "
                    "prostacyclin surgery monitoring safety"
                ),
            ):
                with self.assertRaises(ValueError):
                    opened.registry.invoke(
                        HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                        ToolRequest("search", {"query": query}),
                    )

    def test_six_term_query_cap_is_consistent_across_schema_backend_and_react(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        legal_query = "aspirin bleeding risk prevention stroke safety"
        broad_query = (
            "aspirin platelet bleeding prevention stroke safety monitoring"
        )
        with self._open(
            _MockPubMedOpener(),
            max_query_content_tokens=6,
        ) as opened:
            capability = opened.registry.require_capability(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            )
            description = capability.action_schemas["search"]["properties"][
                "query"
            ]["description"]
            self.assertIn("short named clinical entity", description)
            self.assertIn("one target relation", description)
            self.assertIn("6 clinical content terms", description)
            self.assertNotIn("12 clinical content terms", description)

            with self.assertRaisesRegex(
                ValueError,
                "authoritative clinical-term budget",
            ):
                opened.registry.invoke(
                    HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    ToolRequest("search", {"query": broad_query}),
                )

            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=3,
                max_tool_calls=2,
                max_query_content_tokens=6,
            )
            broad_action = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": broad_query},
                }
            )
            self.assertEqual(
                "query_too_broad_use_at_most_6_clinical_terms",
                adapter._tool_action_error(
                    request=None,
                    action=broad_action,
                    observations=[],
                ),
            )

            result, receipt = asyncio.run(
                opened.registry.ainvoke_with_receipt(
                    HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    ToolRequest("search", {"query": legal_query}),
                )
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(legal_query, result.value["query"])
        self.assertEqual(
            f"{HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION}+"
            f"medrag-{FIXTURE_SOURCE_REVISION}+"
            "pubmed-eutils-retmax-3+query-content-cap-6",
            receipt.tool_version,
        )

    def test_query_cap_defaults_to_historical_twelve_and_requires_exact_wiring(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open(_MockPubMedOpener()) as historical:
            capability = historical.registry.require_capability(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            )
            description = capability.action_schemas["search"]["properties"][
                "query"
            ]["description"]
            self.assertIn("12 clinical content terms", description)
            self.assertNotIn("query-content-cap", capability.version)
            HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=historical.registry,
                max_turns=2,
                max_tool_calls=1,
            )

        with self._open(
            _MockPubMedOpener(),
            max_query_content_tokens=6,
        ) as shortened:
            with self.assertRaisesRegex(
                ValueError,
                "must match the registered HealthBench authoritative Tool",
            ):
                HealthBenchAuthoritativeReactExecutionAdapter(
                    gateway=gateway,
                    tool_registry=shortened.registry,
                    max_turns=2,
                    max_tool_calls=1,
                )

    def test_query_cap_rejects_non_positive_non_integer_and_relaxed_values(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        invalid_values = (0, -1, 13, 1.5, True)
        for invalid in invalid_values:
            with self.subTest(registry_value=invalid):
                with self.assertRaises(ValueError):
                    self._open(
                        _MockPubMedOpener(),
                        max_query_content_tokens=invalid,  # type: ignore[arg-type]
                    )

        with self._open(_MockPubMedOpener()) as opened:
            for invalid in invalid_values:
                with self.subTest(adapter_value=invalid):
                    with self.assertRaises(ValueError):
                        HealthBenchAuthoritativeReactExecutionAdapter(
                            gateway=gateway,
                            tool_registry=opened.registry,
                            max_turns=2,
                            max_tool_calls=1,
                            max_query_content_tokens=invalid,  # type: ignore[arg-type]
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

    def test_relevant_evidence_gate_preserves_complete_named_entity(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        request = _agent_request(
            "Explain the STOP-DAPT trial design and outcomes."
        )
        irrelevant = _search_observation(
            "STOP-DAPT trial outcomes",
            title="General DAPT outcomes",
            excerpt="A broad review of dual antiplatelet therapy.",
        )
        relevant = _search_observation(
            "STOP-DAPT safety endpoints",
            title="STOP-DAPT trial safety endpoints",
            excerpt="The STOP-DAPT study reported prespecified outcomes.",
        )
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=3,
                require_initial_search=True,
                require_relevant_evidence=True,
            )
            after_irrelevant = adapter._state_conditioned_action_domain(
                request,
                [irrelevant],
            )
            after_relevant = adapter._state_conditioned_action_domain(
                request,
                [irrelevant, relevant],
            )

        search = frozenset(
            {(HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID, "search")}
        )
        self.assertEqual((search, False), after_irrelevant)
        self.assertEqual((search, True), after_relevant)

    def test_relevant_evidence_gate_allows_insufficient_completion_at_budget(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        request = _agent_request("Clarify unresolved PSIF terminology.")
        observations = [
            _search_observation(
                f"PSIF meaning context {index}",
                title=f"Unrelated terminology {index}",
                excerpt="No occurrence of the unresolved surface form.",
            )
            for index in range(3)
        ]
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=3,
                require_initial_search=True,
                require_relevant_evidence=True,
            )
            at_budget = adapter._state_conditioned_action_domain(
                request,
                observations,
            )

        self.assertEqual((frozenset(), True), at_budget)

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
            broad = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {
                        "query": (
                            "pediatric pulmonary hypertension diagnosis management "
                            "treatment algorithm screening catheterization vasodilator "
                            "prostacyclin surgery monitoring safety"
                        )
                    },
                }
            )
            empty_content = StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": "and the or with"},
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
            broad_error = adapter._tool_action_error(
                request=None, action=broad, observations=[]
            )
            empty_content_error = adapter._tool_action_error(
                request=None, action=empty_content, observations=[]
            )

        self.assertEqual("duplicate_tool_request", exact_error)
        self.assertEqual("duplicate_tool_request", near_error)
        self.assertIsNone(novel_error)
        self.assertEqual(
            f"query_too_broad_use_at_most_{AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS}_clinical_terms",
            broad_error,
        )
        self.assertEqual(
            "query_must_include_clinical_content_term",
            empty_content_error,
        )

    def test_task_query_anchor_guard_rejects_scope_drift_and_allows_anchor(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        request = type(
            "Request",
            (),
            {
                "problem": (
                    "A patient with prior prostatectomy has constipation and "
                    "cramp-like pain; contrast CT shows no bowel obstruction."
                )
            },
        )()

        def search(query: str) -> StructuredAction:
            return StructuredAction.from_value(
                {
                    "kind": "tool",
                    "name": "search",
                    "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    "skill_id": None,
                    "arguments": {"query": query},
                }
            )

        with self._open(_MockPubMedOpener()) as opened:
            guarded = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=3,
                max_tool_calls=2,
                require_task_query_anchor=True,
            )
            drift = guarded._tool_action_error(
                request=request,
                action=search("stent placement acute diarrhea management"),
                observations=[],
            )
            grounded = guarded._tool_action_error(
                request=request,
                action=search("constipation bowel obstruction CT admission"),
                observations=[],
            )

        self.assertEqual(
            "query_does_not_preserve_public_task_anchor",
            drift,
        )
        self.assertIsNone(grounded)

    def test_task_query_anchor_ignores_nonsemantic_json_scaffolding(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        request = type(
            "Request",
            (),
            {
                "problem": (
                    '[{"role":"user","content":"我最近头痛，应该怎么办？"}]'
                )
            },
        )()
        action = StructuredAction.from_value(
            {
                "kind": "tool",
                "name": "search",
                "resource_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                "skill_id": None,
                "arguments": {"query": "headache clinical evaluation"},
            }
        )

        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_task_query_anchor=True,
            )
            error = adapter._tool_action_error(
                request=request,
                action=action,
                observations=[],
            )

        self.assertIsNone(error)

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

    def test_completion_guard_is_opt_in_and_historical_schema_is_unchanged(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        request = type("Request", (), {"is_output_agent": True})()
        with self._open(_MockPubMedOpener()) as opened:
            historical = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
            )
            guarded = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                max_completion_artifact_characters=(
                    HEALTHBENCH_COMPLETION_ARTIFACT_MAX_CHARACTERS_V1
                ),
            )
            historical_schema = historical._completion_arguments_schema(request)
            guarded_schema = guarded._completion_arguments_schema(request)

        historical_value = historical_schema["properties"]["value"]
        guarded_value = guarded_schema["properties"]["value"]
        self.assertNotIn("type", historical_value)
        self.assertNotIn("maxLength", historical_value)
        self.assertEqual("string", guarded_value["type"])
        self.assertEqual(1, guarded_value["minLength"])
        self.assertEqual(
            HEALTHBENCH_COMPLETION_ARTIFACT_MAX_CHARACTERS_V1,
            guarded_value["maxLength"],
        )
        self.assertIn("complete assistant response", guarded_value["description"])
        Draft202012Validator.check_schema(guarded_schema)

    def test_completion_guard_describes_intermediate_contract_artifact(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        request = type("Request", (), {"is_output_agent": False})()
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                max_completion_artifact_characters=120,
            )
            schema = adapter._completion_arguments_schema(request)

        description = schema["properties"]["value"]["description"]
        self.assertIn("intermediate artifact", description)
        self.assertIn("AgentGraph communication", description)
        self.assertNotIn("assistant response", description)

    def test_completion_guard_rejects_oversize_without_truncating_artifact(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        completion = StructuredAction.from_value(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": "artifact"},
            }
        )
        with self._open(_MockPubMedOpener()) as opened:
            historical = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
            )
            guarded = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                max_completion_artifact_characters=12,
            )
            historical_error = historical._completion_error(
                action=completion,
                artifact="x" * 13,
                tool_receipts=[],
            )
            at_limit_error = guarded._completion_error(
                action=completion,
                artifact="x" * 12,
                tool_receipts=[],
            )
            oversize = "x" * 13
            oversize_error = guarded._completion_error(
                action=completion,
                artifact=oversize,
                tool_receipts=[],
            )

        self.assertIsNone(historical_error)
        self.assertIsNone(at_limit_error)
        self.assertEqual(
            "completion_artifact_exceeds_12_characters",
            oversize_error,
        )
        self.assertEqual(13, len(oversize))

    def test_completion_guard_rejects_only_obvious_heading_or_label_when_enabled(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()

        def completion(value: str) -> StructuredAction:
            return StructuredAction.from_value(
                {
                    "kind": "complete",
                    "name": "complete",
                    "resource_id": None,
                    "skill_id": None,
                    "arguments": {"value": value},
                }
            )

        with self._open(_MockPubMedOpener()) as opened:
            historical = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
            )
            guarded = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_complete_natural_language_artifact=True,
                completion_quality_profile=(
                    HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2
                ),
            )

            self.assertIsNone(
                historical._completion_error(
                    action=completion("score"),
                    artifact="score",
                    tool_receipts=[],
                )
            )
            for artifact, error in (
                ("score", "completion_artifact_is_label_only"),
                (
                    "# Meta-Analysis of Enteral Nutrition: Key Evidence Summary",
                    "completion_artifact_is_heading_only",
                ),
                (
                    "**Mallory-Weiss Tear Outpatient Management:**",
                    "completion_artifact_is_heading_only",
                ),
            ):
                self.assertEqual(
                    error,
                    guarded._completion_error(
                        action=completion(artifact),
                        artifact=artifact,
                        tool_receipts=[],
                    ),
                )
            complete = "A single intramuscular dose is recommended."
            self.assertIsNone(
                guarded._completion_error(
                    action=completion(complete),
                    artifact=complete,
                    tool_receipts=[],
                )
            )

    def test_completion_guard_retries_post_tool_keyword_fragment(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        action = StructuredAction.from_value(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {
                    "value": (
                        "guidelines dental clearance oral examination extraction "
                        "repair concurrent radiation treatment syndrome osteo"
                    )
                },
            }
        )
        receipt = {
            "tool_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
            "error_type": None,
            "result": {"completed": True, "value": {"evidence": []}},
        }

        with self._open(_MockPubMedOpener()) as opened:
            guarded = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_complete_natural_language_artifact=True,
                completion_quality_profile=(
                    HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2
                ),
            )

            self.assertEqual(
                "completion_artifact_is_post_tool_keyword_fragment",
                guarded._completion_error(
                    action=action,
                    artifact=action.arguments["value"],
                    tool_receipts=[receipt],
                ),
            )
            finding = (
                "The retrieved evidence indicates that dental assessment "
                "should occur before radiotherapy begins"
            )
            self.assertIsNone(
                guarded._completion_error(
                    action=action,
                    artifact=finding,
                    tool_receipts=[receipt],
                )
            )

    def test_oversize_completion_is_public_observation_then_same_agent_repairs(
        self,
    ) -> None:
        class SequenceGateway:
            def __init__(self) -> None:
                self.requests: list[AgentRequest] = []
                self.outputs = [
                    json.dumps(
                        {
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                            "arguments": {"value": "x" * 13},
                        }
                    ),
                    json.dumps(
                        {
                            "kind": "complete",
                            "name": "complete",
                            "resource_id": None,
                            "skill_id": None,
                            "arguments": {"value": "repaired"},
                        }
                    ),
                ]

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                return AgentResponse(self.outputs.pop(0))

        gateway = SequenceGateway()
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                max_completion_artifact_characters=12,
            )
            request = AgentRequest(
                request_id="run:healthbench:output",
                run_id="run",
                graph_revision=1,
                problem="Provide a complete response to this conversation.",
                agent=AgentNode(
                    "output",
                    "model",
                    "Answer the user's conversation using available evidence.",
                    allowed_tools=(HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,),
                    execution_mode="react",
                    completion_condition="return the complete response",
                ),
                model=ModelSpec("model", "provider"),
                provider=ProviderSpec("provider", kind="test"),
                phase=ExecutionPhase.SINGLE,
                is_output_agent=True,
            )
            response = asyncio.run(adapter.execute(request))

        self.assertEqual("repaired", response.text)
        trace = response.metadata["react_trace"]
        self.assertEqual("schema_invalid", trace[0]["observation_status"])
        self.assertEqual(
            "completion_artifact_exceeds_12_characters",
            trace[0]["public_error_code"],
        )
        self.assertEqual("completed", trace[1]["observation_status"])
        self.assertEqual(2, len(gateway.requests))
        self.assertIn(
            "completion_artifact_exceeds_12_characters",
            gateway.requests[1].agent.contract,
        )

    def test_completion_guard_configuration_validation(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open(_MockPubMedOpener()) as opened:
            for invalid in (0, -1, 1.5, True):
                with self.subTest(value=invalid):
                    with self.assertRaises(ValueError):
                        HealthBenchAuthoritativeReactExecutionAdapter(
                            gateway=gateway,
                            tool_registry=opened.registry,
                            max_turns=2,
                            max_tool_calls=1,
                            max_completion_artifact_characters=invalid,
                        )

    def test_completion_guard_rejects_fullwidth_colon_leadin(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        completion = StructuredAction.from_value(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": "接下来的治疗计划如下："},
            }
        )
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_complete_natural_language_artifact=True,
                completion_quality_profile=(
                    HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2
                ),
            )
            error = adapter._completion_error(
                action=completion,
                artifact="接下来的治疗计划如下：",
                tool_receipts=[],
            )

        self.assertEqual("completion_artifact_is_heading_only", error)

    def test_completion_guard_v1_keeps_historical_fullwidth_error_code(self) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        completion = StructuredAction.from_value(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": "接下来的治疗计划如下："},
            }
        )
        with self._open(_MockPubMedOpener()) as opened:
            historical = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_complete_natural_language_artifact=True,
            )
            error = historical._completion_error(
                action=completion,
                artifact="接下来的治疗计划如下：",
                tool_receipts=[],
            )

        self.assertEqual("completion_artifact_is_label_only", error)

    def test_structured_evidence_schema_only_applies_to_intermediate_agent(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_structured_evidence_artifact=True,
                max_completion_artifact_characters=12_000,
            )
            intermediate = adapter._completion_arguments_schema(
                type("Request", (), {"is_output_agent": False})()
            )
            output = adapter._completion_arguments_schema(
                type("Request", (), {"is_output_agent": True})()
            )

        self.assertEqual(
            "healthbench.structured-evidence.v1",
            intermediate["properties"]["value"]["properties"]
            ["schema_version"]["const"],
        )
        self.assertEqual("string", output["properties"]["value"]["type"])
        Draft202012Validator.check_schema(intermediate)
        Draft202012Validator.check_schema(output)

    def test_structured_evidence_artifact_requires_exact_receipt_binding(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        evidence = {
            "document_id": "111",
            "source": "NCBI PubMed",
            "title": "Aspirin safety",
            "date": "2024-Jan",
            "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
            "excerpt": "Aspirin may increase gastrointestinal bleeding risk.",
        }
        receipt = {
            "tool_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
            "error_type": None,
            "result": {
                "completed": True,
                "value": {
                    "operation": "search",
                    "evidence": [evidence],
                },
            },
        }
        artifact = {
            "schema_version": "healthbench.structured-evidence.v1",
            "status": "supported",
            "summary": "Aspirin can increase gastrointestinal bleeding risk.",
            "evidence_items": [
                {
                    "supported_claim": "Aspirin can increase bleeding risk.",
                    "conditions_or_qualifiers": "Assess individual risk.",
                    "document_id": "111",
                    "source": "NCBI PubMed",
                    "title": "Aspirin safety",
                    "date": "2024-Jan",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
                    "evidence_span": "increase gastrointestinal bleeding risk",
                }
            ],
            "uncertainties": [],
        }
        completion = StructuredAction.from_value(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": artifact},
            }
        )
        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=2,
                max_tool_calls=1,
                require_structured_evidence_artifact=True,
                max_completion_artifact_characters=12_000,
            )
            accepted = adapter._completion_error(
                action=completion,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt],
            )
            changed = json.loads(json.dumps(artifact))
            changed["evidence_items"][0]["evidence_span"] = (
                "A claim that is absent from the receipt."
            )
            rejected = adapter._completion_error(
                action=StructuredAction.from_value(
                    {
                        "kind": "complete",
                        "name": "complete",
                        "resource_id": None,
                        "skill_id": None,
                        "arguments": {"value": changed},
                    }
                ),
                artifact=json.dumps(changed),
                tool_receipts=[receipt],
            )

        self.assertIsNone(accepted)
        self.assertEqual(
            "structured_evidence_item_span_not_in_receipt",
            rejected,
        )

    def test_insufficient_structured_evidence_requires_one_refined_search(
        self,
    ) -> None:
        gateway = type("Gateway", (), {"generate": lambda *_: None})()
        artifact = {
            "schema_version": "healthbench.structured-evidence.v1",
            "status": "insufficient",
            "summary": "The first search did not resolve the question.",
            "evidence_items": [],
            "uncertainties": ["The requested comparison remains unresolved."],
        }
        completion = StructuredAction.from_value(
            {
                "kind": "complete",
                "name": "complete",
                "resource_id": None,
                "skill_id": None,
                "arguments": {"value": artifact},
            }
        )

        def receipt(index: int) -> dict[str, object]:
            return {
                "tool_id": HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                "error_type": None,
                "request": {
                    "action": "search",
                    "arguments": {"query": f"distinct query {index}"},
                },
                "result": {
                    "completed": True,
                    "value": {
                        "operation": "search",
                        "evidence": [{"document_id": str(index)}],
                    },
                },
            }

        with self._open(_MockPubMedOpener()) as opened:
            adapter = HealthBenchAuthoritativeReactExecutionAdapter(
                gateway=gateway,
                tool_registry=opened.registry,
                max_turns=6,
                max_tool_calls=3,
                max_successful_searches=2,
                require_structured_evidence_artifact=True,
                require_refinement_on_insufficient_evidence=True,
            )
            needs_refinement = adapter._completion_error(
                action=completion,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt(1)],
            )
            exhausted = adapter._completion_error(
                action=completion,
                artifact=json.dumps(artifact),
                tool_receipts=[receipt(1), receipt(2)],
            )

        self.assertEqual(
            "insufficient_evidence_requires_distinct_refined_search",
            needs_refinement,
        )
        self.assertIsNone(exhausted)


if __name__ == "__main__":
    unittest.main()
