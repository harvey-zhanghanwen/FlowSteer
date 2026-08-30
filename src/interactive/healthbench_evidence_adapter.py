"""HealthBench authoritative evidence Tool over MedRAG and NCBI PubMed.

``FrozenMedRAGBM25Corpus`` is reused directly from the existing HealthBench
adapter, whose loading and BM25 ranking path is a thin port of SkillFlow's
``training/environment.py`` external-corpus search.  The registered Tool and
bounded Action--Observation execution use the existing SkillFlow-compatible
``ToolRegistry`` and FlowSteer-compatible ``ToolReactExecutionAdapter``
boundaries.

The only project-specific adaptation is source aggregation: one public search
action returns evidence from the frozen local medical corpus and the official
NCBI PubMed E-utilities service.  It never accepts or reads HealthBench task
IDs, rubrics, reference responses, ground truth, or evaluator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from threading import Lock
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .agent_runtime import AgentRequest
from .healthbench_tool_adapter import (
    HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
    FrozenMedRAGBM25Corpus,
)
from .react_execution import ToolReactExecutionAdapter
from .tool_runtime import (
    StructuredAction,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID = "healthbench-authoritative.search"
HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION = "healthbench-authoritative-v1"

# NECESSARY_PROJECT_ADAPTATION: these constants are runtime-owned rather than
# Agent-controlled.  Every request is restricted to the official E-utilities
# host, the PubMed database, a bounded result count, and a named NCBI client.
NCBI_EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"
NCBI_EUTILS_BASE_URL = f"https://{NCBI_EUTILS_HOST}/entrez/eutils"
NCBI_EUTILS_TOOL_PARAMETER = "FlowSteerHealthBenchEvidence"
PUBMED_DATABASE = "pubmed"
PUBMED_RETMAX = 3
PUBMED_HTTP_TIMEOUT_SECONDS = 8.0
PUBMED_EXCERPT_CHARACTERS = 1200
AUTHORITATIVE_MAX_SUCCESSFUL_SEARCHES = 2

_PUBMED_RATE_LOCK = Lock()
_PUBMED_LAST_REQUEST_AT = 0.0


def _wait_for_pubmed_request_slot(minimum_interval_seconds: float) -> None:
    """Serialize request starts to respect the configured NCBI request rate."""

    global _PUBMED_LAST_REQUEST_AT
    with _PUBMED_RATE_LOCK:
        now = time.monotonic()
        remaining = minimum_interval_seconds - (now - _PUBMED_LAST_REQUEST_AT)
        if remaining > 0:
            time.sleep(remaining)
        _PUBMED_LAST_REQUEST_AT = time.monotonic()


def _required_query(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("search query must be non-empty text")
    return value.strip()


def _node_text(node: ElementTree.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _publication_date(article: ElementTree.Element) -> str | None:
    article_date = article.find(".//Article/ArticleDate")
    publication_date = article.find(".//Journal/JournalIssue/PubDate")
    date_node = article_date if article_date is not None else publication_date
    if date_node is None:
        return None
    medline_date = _node_text(date_node.find("MedlineDate"))
    if medline_date:
        return medline_date
    components = [
        _node_text(date_node.find(component))
        for component in ("Year", "Month", "Day")
    ]
    rendered = "-".join(component for component in components if component)
    return rendered or None


@dataclass(frozen=True, slots=True)
class PubMedEUtilitiesClient:
    """Small synchronous client for the official NCBI PubMed endpoints."""

    base_url: str = NCBI_EUTILS_BASE_URL
    tool_parameter: str = NCBI_EUTILS_TOOL_PARAMETER
    retmax: int = PUBMED_RETMAX
    timeout_seconds: float = PUBMED_HTTP_TIMEOUT_SECONDS
    minimum_interval_seconds: float = 0.4
    # Tests inject a completely local opener.  Production uses only stdlib
    # urllib and the fixed HTTPS endpoint above.
    opener: Callable[..., object] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        normalized_base = self.base_url.rstrip("/")
        if normalized_base != NCBI_EUTILS_BASE_URL:
            raise ValueError("PubMed client must use the official NCBI endpoint")
        if not self.tool_parameter.strip():
            raise ValueError("PubMed client tool parameter cannot be empty")
        if type(self.retmax) is not int or not 1 <= self.retmax <= 5:
            raise ValueError("PubMed retmax must be an integer from 1 through 5")
        if self.timeout_seconds <= 0:
            raise ValueError("PubMed timeout must be positive")
        if self.minimum_interval_seconds < 0:
            raise ValueError("PubMed minimum interval cannot be negative")
        object.__setattr__(self, "base_url", normalized_base)
        object.__setattr__(self, "tool_parameter", self.tool_parameter.strip())

    def _url(self, endpoint: str, parameters: Mapping[str, object]) -> str:
        fixed_parameters = {
            **parameters,
            "db": PUBMED_DATABASE,
            "tool": self.tool_parameter,
        }
        return f"{self.base_url}/{endpoint}?{urlencode(fixed_parameters)}"

    def _get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes:
        _wait_for_pubmed_request_slot(self.minimum_interval_seconds)
        request = Request(
            self._url(endpoint, parameters),
            headers={
                "Accept": "application/json, application/xml;q=0.9",
                "User-Agent": f"{self.tool_parameter}/1.0",
            },
            method="GET",
        )
        response = self.opener(
            request,
            timeout=self.timeout_seconds,
        )
        try:
            payload = response.read()  # type: ignore[attr-defined]
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if not isinstance(payload, bytes):
            raise RuntimeError("NCBI E-utilities response is not bytes")
        return payload

    def search(self, query: str) -> list[dict[str, object]]:
        """Return a bounded PubMed result projection with no benchmark fields."""

        query = _required_query(query)
        search_payload = json.loads(
            self._get(
                "esearch.fcgi",
                {
                    "term": query,
                    "retmode": "json",
                    "retmax": self.retmax,
                    "sort": "relevance",
                },
            ).decode("utf-8")
        )
        if not isinstance(search_payload, Mapping):
            raise RuntimeError("NCBI PubMed search response is invalid")
        search_result = search_payload.get("esearchresult")
        identifiers = (
            search_result.get("idlist")
            if isinstance(search_result, Mapping)
            else None
        )
        if not isinstance(identifiers, list) or any(
            not isinstance(identifier, str) or not identifier.isdigit()
            for identifier in identifiers
        ):
            raise RuntimeError("NCBI PubMed identifier response is invalid")
        identifiers = list(dict.fromkeys(identifiers))[: self.retmax]
        if not identifiers:
            return []

        article_payload = self._get(
            "efetch.fcgi",
            {
                "id": ",".join(identifiers),
                "retmode": "xml",
                "rettype": "abstract",
            },
        )
        try:
            root = ElementTree.fromstring(article_payload)
        except ElementTree.ParseError as exc:
            raise RuntimeError("NCBI PubMed article response is invalid") from exc

        by_identifier: dict[str, dict[str, object]] = {}
        for article in root.findall(".//PubmedArticle"):
            identifier = _node_text(article.find(".//MedlineCitation/PMID"))
            if not identifier or identifier not in identifiers:
                continue
            title = _node_text(article.find(".//Article/ArticleTitle"))
            abstract_parts = [
                _node_text(part)
                for part in article.findall(".//Article/Abstract/AbstractText")
            ]
            excerpt = " ".join(part for part in abstract_parts if part)
            by_identifier[identifier] = {
                "source_type": "peer_reviewed_literature",
                "source": "NCBI PubMed",
                "document_id": identifier,
                "title": title,
                "date": _publication_date(article),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/",
                "excerpt": excerpt[:PUBMED_EXCERPT_CHARACTERS],
            }
        return [
            by_identifier[identifier]
            for identifier in identifiers
            if identifier in by_identifier
        ]


def _interleave_evidence(
    local: list[dict[str, object]],
    pubmed: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Interleave source-local ranks without comparing incompatible scores."""

    combined: list[dict[str, object]] = []
    for source_rank in range(max(len(local), len(pubmed))):
        for source_items in (local, pubmed):
            if source_rank >= len(source_items):
                continue
            combined.append(
                {
                    **source_items[source_rank],
                    "rank": len(combined) + 1,
                }
            )
    return combined


@dataclass(frozen=True, slots=True)
class HealthBenchAuthoritativeSearchToolBackend:
    """Aggregate frozen textbook evidence and live official PubMed evidence."""

    corpus: FrozenMedRAGBM25Corpus
    pubmed: PubMedEUtilitiesClient = field(default_factory=PubMedEUtilitiesClient)

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "search":
            raise ValueError(
                "HealthBench authoritative backend received an incompatible action"
            )
        if set(request.arguments) != {"query"}:
            raise ValueError("search arguments must contain exactly query")
        query = _required_query(request.arguments["query"])

        # DIRECT_REUSE_SKILLFLOW: preserve FrozenMedRAGBM25Corpus.search rather
        # than introducing another local retrieval or ranking implementation.
        local_chunks = self.corpus.search(query)
        local_evidence = [
            {
                "source_type": "frozen_medical_textbook",
                "source": self.corpus.source_identity,
                "document_id": cast(str, chunk["document_id"]),
                "title": cast(str, chunk["title"]),
                "date": None,
                "url": None,
                "excerpt": cast(str, chunk["text"]),
            }
            for chunk in local_chunks
        ]
        source_receipts: list[dict[str, object]] = [
            {
                "source_type": "frozen_medical_textbook",
                "source": self.corpus.source_identity,
                "status": "success",
                "result_count": len(local_evidence),
                "error_type": None,
            }
        ]
        try:
            pubmed_evidence = self.pubmed.search(query)
        except Exception as exc:
            # Preserve useful frozen evidence when the optional network source
            # is unavailable, while making the source failure observable.
            pubmed_evidence = []
            source_receipts.append(
                {
                    "source_type": "peer_reviewed_literature",
                    "source": "NCBI PubMed",
                    "status": "error",
                    "result_count": 0,
                    "error_type": type(exc).__name__,
                }
            )
        else:
            source_receipts.append(
                {
                    "source_type": "peer_reviewed_literature",
                    "source": "NCBI PubMed",
                    "status": "success",
                    "result_count": len(pubmed_evidence),
                    "error_type": None,
                }
            )

        return ToolResult(
            {
                "operation": "search",
                "query": query,
                "evidence": _interleave_evidence(local_evidence, pubmed_evidence),
                "source_receipts": source_receipts,
                "frozen_corpus": self.corpus.identity,
                "pubmed": {
                    "host": NCBI_EUTILS_HOST,
                    "database": PUBMED_DATABASE,
                    "retmax": self.pubmed.retmax,
                    "tool": self.pubmed.tool_parameter,
                },
            }
        )


def _normalized_query_tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    stop_words = {
        "a",
        "an",
        "and",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in stop_words
    )


def _queries_are_near_duplicates(left: object, right: object) -> bool:
    left_tokens = _normalized_query_tokens(left)
    right_tokens = _normalized_query_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    jaccard = intersection / union
    # NECESSARY_PROJECT_ADAPTATION: lexical novelty is the only admissible
    # public signal here; no rubric, reference answer, or evaluator feedback is
    # consulted. High Jaccard overlap rejects a reordered/paraphrased query,
    # while a query that adds a genuinely new clinical concept remains legal.
    return jaccard >= 0.8


class HealthBenchAuthoritativeReactExecutionAdapter(ToolReactExecutionAdapter):
    """Bound authoritative retrieval to two distinct searches and completion."""

    def __init__(
        self,
        *,
        require_initial_search: bool = False,
        max_successful_searches: int | None = None,
        **kwargs: Any,
    ) -> None:
        if type(require_initial_search) is not bool:
            raise TypeError("require_initial_search must be boolean")
        super().__init__(**kwargs)
        if max_successful_searches is None:
            max_successful_searches = min(
                AUTHORITATIVE_MAX_SUCCESSFUL_SEARCHES,
                self._max_tool_calls,
            )
        if (
            type(max_successful_searches) is not int
            or not 1 <= max_successful_searches <= self._max_tool_calls
        ):
            raise ValueError(
                "max_successful_searches must be between 1 and max_tool_calls"
            )
        self._require_initial_search = require_initial_search
        self._max_successful_searches = max_successful_searches

    @staticmethod
    def _executed_search(
        observation: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        action = observation.get("executed_action")
        if not isinstance(action, Mapping):
            return None
        if (
            action.get("kind") != "tool"
            or action.get("name") != "search"
            or action.get("resource_id")
            != HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
        ):
            return None
        return action

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Expose search and completion from measured public retrieval state."""

        del request
        dispatched = 0
        successful = 0
        for observation in observations:
            if self._executed_search(observation) is None:
                continue
            if observation.get("observation_status") in {"success", "tool_error"}:
                dispatched += 1
            if observation.get("observation_status") == "success":
                result = observation.get("result")
                evidence = (
                    result.get("evidence")
                    if isinstance(result, Mapping)
                    else None
                )
                if isinstance(evidence, list) and evidence:
                    successful += 1

        search_admitted = (
            dispatched < self._max_tool_calls
            and successful < self._max_successful_searches
        )
        admitted = (
            frozenset(
                {(HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID, "search")}
            )
            if search_admitted
            else frozenset()
        )
        completion_admitted = not (
            self._require_initial_search and dispatched == 0 and search_admitted
        )
        return admitted, completion_admitted

    @staticmethod
    def _action_schema(
        *,
        arguments_schema: Mapping[str, object],
        kind: str,
        name: str,
        resource_id: str | None,
    ) -> dict[str, object]:
        return {
            "type": "object",
            "required": [
                "arguments",
                "kind",
                "name",
                "resource_id",
                "skill_id",
            ],
            "properties": {
                "arguments": dict(arguments_schema),
                "kind": {"const": kind},
                "name": {"const": name},
                "resource_id": {"const": resource_id},
                "skill_id": {"const": None},
            },
            "additionalProperties": False,
        }

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> dict[str, object] | None:
        """Use strict ``oneOf`` when search and completion are both legal."""

        admitted, completion_admitted = self._state_conditioned_action_domain(
            request,
            observations,
        )
        if (
            admitted
            == frozenset({(HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID, "search")})
            and completion_admitted
        ):
            capability = self._tool_registry.require_capability(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            )
            return {
                "oneOf": [
                    self._action_schema(
                        arguments_schema=capability.action_schemas["search"],
                        kind="tool",
                        name="search",
                        resource_id=HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                    ),
                    self._action_schema(
                        arguments_schema=self._completion_arguments_schema(request),
                        kind="complete",
                        name="complete",
                        resource_id=None,
                    ),
                ]
            }
        # FLOWSTEER_BOUNDARY: the generic executor remains authoritative for
        # single-branch strict schemas and all Action--Observation receipts.
        return super()._state_conditioned_response_schema(request, observations)

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        """Reject exact and lexical near-duplicate searches without dispatch."""

        del request
        if (
            action.resource_id != HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            or action.name != "search"
            or not isinstance(action.arguments, dict)
        ):
            return None
        query = action.arguments.get("query")
        for observation in observations:
            prior = self._executed_search(observation)
            if prior is None:
                continue
            arguments = prior.get("arguments")
            if isinstance(arguments, Mapping) and _queries_are_near_duplicates(
                arguments.get("query"), query
            ):
                return "duplicate_tool_request"
        return None


def _evidence_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_type",
            "source",
            "document_id",
            "title",
            "date",
            "url",
            "excerpt",
            "rank",
        ],
        "properties": {
            "source_type": {"type": "string"},
            "source": {"type": "string"},
            "document_id": {"type": "string"},
            "title": {"type": "string"},
            "date": {"type": ["string", "null"]},
            "url": {"type": ["string", "null"]},
            "excerpt": {"type": "string"},
            "rank": {"type": "integer", "minimum": 1},
        },
    }


def build_healthbench_authoritative_tool_registry(
    corpus: FrozenMedRAGBM25Corpus,
    *,
    pubmed_client: PubMedEUtilitiesClient | None = None,
    timeout_seconds: float = 20.0,
) -> ToolRegistry:
    """Register the single HealthBench-only aggregate evidence Tool."""

    if not isinstance(corpus, FrozenMedRAGBM25Corpus):
        raise TypeError("corpus must be a FrozenMedRAGBM25Corpus")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    client = pubmed_client or PubMedEUtilitiesClient()
    if not isinstance(client, PubMedEUtilitiesClient):
        raise TypeError("pubmed_client must be a PubMedEUtilitiesClient")

    search_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
                "description": (
                    "Search with concise English clinical terminology. Use "
                    "specific conditions, interventions, outcomes, populations, "
                    "or safety concepts rather than copying the full conversation. "
                    "A supplemental search must introduce a materially different "
                    "clinical concept, not repeat or paraphrase the prior query."
                ),
            }
        },
    }
    corpus_identity_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source",
            "source_revision",
            "corpus_rows",
            "retrieval_backend",
        ],
        "properties": {
            "source": {"type": "string"},
            "source_revision": {"type": "string"},
            "corpus_rows": {"type": "integer"},
            "retrieval_backend": {"const": "bm25"},
        },
    }
    source_receipt_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_type",
            "source",
            "status",
            "result_count",
            "error_type",
        ],
        "properties": {
            "source_type": {"type": "string"},
            "source": {"type": "string"},
            "status": {"enum": ["success", "error"]},
            "result_count": {"type": "integer", "minimum": 0},
            "error_type": {"type": ["string", "null"]},
        },
    }
    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "operation",
            "query",
            "evidence",
            "source_receipts",
            "frozen_corpus",
            "pubmed",
        ],
        "properties": {
            "operation": {"const": "search"},
            "query": {"type": "string"},
            "evidence": {"type": "array", "items": _evidence_schema()},
            "source_receipts": {
                "type": "array",
                "items": source_receipt_schema,
            },
            "frozen_corpus": corpus_identity_schema,
            "pubmed": {
                "type": "object",
                "additionalProperties": False,
                "required": ["host", "database", "retmax", "tool"],
                "properties": {
                    "host": {"const": NCBI_EUTILS_HOST},
                    "database": {"const": PUBMED_DATABASE},
                    "retmax": {"const": client.retmax},
                    "tool": {"const": client.tool_parameter},
                },
            },
        },
    }
    capability = ToolCapability(
        tool_id=HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
        dataset_scope=HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
        action_schemas={"search": search_input_schema},
        input_schema=search_input_schema,
        output_schema=output_schema,
        side_effect="none",
        timeout_seconds=timeout_seconds,
        version=(
            f"{HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION}+"
            f"medrag-{corpus.source_revision}+"
            f"pubmed-eutils-retmax-{client.retmax}"
        ),
    )
    return ToolRegistry(
        (
            ToolRegistration(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                HealthBenchAuthoritativeSearchToolBackend(corpus, client),
                capability,
            ),
        )
    )


@dataclass(slots=True)
class OpenHealthBenchAuthoritativeToolRegistry:
    """Owned frozen corpus plus one immutable aggregate Tool registration."""

    registry: ToolRegistry
    frozen_corpus_identity: Mapping[str, object]
    _corpus: FrozenMedRAGBM25Corpus = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._corpus.close()
            self._closed = True

    def __enter__(self) -> "OpenHealthBenchAuthoritativeToolRegistry":
        if self._closed:
            raise RuntimeError("HealthBench authoritative ToolRegistry is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def open_healthbench_authoritative_tool_registry(
    *,
    corpus_root: str | Path,
    source_identity: str,
    expected_source_revision: str,
    expected_rows: int,
    pubmed_client: PubMedEUtilitiesClient | None = None,
    timeout_seconds: float = 20.0,
) -> OpenHealthBenchAuthoritativeToolRegistry:
    """Open the frozen SkillFlow corpus and the aggregate Tool lifecycle."""

    corpus = FrozenMedRAGBM25Corpus.open(
        corpus_root,
        source_identity=source_identity,
        expected_source_revision=expected_source_revision,
        expected_rows=expected_rows,
    )
    try:
        registry = build_healthbench_authoritative_tool_registry(
            corpus,
            pubmed_client=pubmed_client,
            timeout_seconds=timeout_seconds,
        )
    except BaseException:
        corpus.close()
        raise
    return OpenHealthBenchAuthoritativeToolRegistry(
        registry=registry,
        frozen_corpus_identity=MappingProxyType(corpus.identity),
        _corpus=corpus,
    )


__all__ = [
    "AUTHORITATIVE_MAX_SUCCESSFUL_SEARCHES",
    "HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID",
    "HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION",
    "HealthBenchAuthoritativeReactExecutionAdapter",
    "HealthBenchAuthoritativeSearchToolBackend",
    "NCBI_EUTILS_BASE_URL",
    "NCBI_EUTILS_HOST",
    "NCBI_EUTILS_TOOL_PARAMETER",
    "OpenHealthBenchAuthoritativeToolRegistry",
    "PUBMED_DATABASE",
    "PUBMED_HTTP_TIMEOUT_SECONDS",
    "PUBMED_RETMAX",
    "PubMedEUtilitiesClient",
    "build_healthbench_authoritative_tool_registry",
    "open_healthbench_authoritative_tool_registry",
]
