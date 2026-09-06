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

from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
import socket
from threading import Lock
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .agent_runtime import AgentRequest, CommunicationCondition, GatewayResponse
from .healthbench_professional_adapter import parse_model_visible_conversation
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
HEALTHBENCH_AUTHORITATIVE_TOOL_VERSION = "healthbench-authoritative-v3"

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
AUTHORITATIVE_QUERY_MAX_CHARACTERS = 160
AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS = 12
# NECESSARY_PROJECT_ADAPTATION: the historical HealthBench conditions remain
# unbounded unless they explicitly opt into this versioned completion guard.
# The value is above every admitted v2.16 artifact (maximum 10,299 characters)
# while rejecting the observed 53,358-character degeneration.  The limit is
# applied to both constrained decoding and post-generation admission; it never
# truncates or rewrites a semantic artifact.
HEALTHBENCH_COMPLETION_ARTIFACT_MAX_CHARACTERS_V1 = 12_000
HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V1 = (
    "healthbench_completion_quality_v1"
)
HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2 = (
    "healthbench_completion_quality_v2"
)
_HEALTHBENCH_COMPLETION_QUALITY_PROFILES = frozenset(
    {
        HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V1,
        HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2,
    }
)
_COMPLETION_REQUEST: ContextVar[AgentRequest | None] = ContextVar(
    "healthbench_completion_request", default=None
)

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


def _validated_max_query_content_tokens(value: object) -> int:
    if (
        type(value) is not int
        or not 1 <= value <= AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS
    ):
        raise ValueError(
            "max_query_content_tokens must be a positive integer no greater "
            f"than {AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS}"
        )
    return value


def _required_query(
    value: object,
    *,
    max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
) -> str:
    maximum = _validated_max_query_content_tokens(max_query_content_tokens)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("search query must be non-empty text")
    query = value.strip()
    if len(query) > AUTHORITATIVE_QUERY_MAX_CHARACTERS:
        raise ValueError(
            "search query exceeds the authoritative character budget"
        )
    content_terms = _normalized_query_tokens(query)
    if not content_terms:
        raise ValueError("search query must include a clinical content term")
    if len(content_terms) > maximum:
        raise ValueError(
            "search query exceeds the authoritative clinical-term budget"
        )
    return query


def _query_content_tokens_from_tool_version(version: str) -> int:
    match = re.search(r"(?:^|\+)query-content-cap-(\d+)(?:\+|$)", version)
    if match is None:
        return AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS
    return _validated_max_query_content_tokens(int(match.group(1)))


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
    # DIRECT_REUSE_FLOWSTEER: the OpenAI-compatible gateway uses a bounded
    # retry loop for transport failures and retryable HTTP status codes.  Keep
    # zero retries as the historical PubMed behavior unless the evaluation
    # condition opts into the same transport policy explicitly.
    max_retries: int = 0
    retry_backoff_seconds: float = 1.0
    # Tests inject a completely local opener.  Production uses only stdlib
    # urllib and the fixed HTTPS endpoint above.
    opener: Callable[..., object] = field(default=urlopen, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

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
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("PubMed max_retries must be a non-negative integer")
        if (
            isinstance(self.retry_backoff_seconds, bool)
            or not isinstance(self.retry_backoff_seconds, (int, float))
            or not math.isfinite(float(self.retry_backoff_seconds))
            or self.retry_backoff_seconds < 0
        ):
            raise ValueError(
                "PubMed retry_backoff_seconds must be a finite non-negative number"
            )
        if not callable(self.opener):
            raise TypeError("PubMed opener must be callable")
        if not callable(self.sleep):
            raise TypeError("PubMed sleep must be callable")
        object.__setattr__(self, "base_url", normalized_base)
        object.__setattr__(self, "tool_parameter", self.tool_parameter.strip())
        object.__setattr__(
            self,
            "retry_backoff_seconds",
            float(self.retry_backoff_seconds),
        )

    def _url(self, endpoint: str, parameters: Mapping[str, object]) -> str:
        fixed_parameters = {
            **parameters,
            "db": PUBMED_DATABASE,
            "tool": self.tool_parameter,
        }
        return f"{self.base_url}/{endpoint}?{urlencode(fixed_parameters)}"

    def _get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes:
        request = Request(
            self._url(endpoint, parameters),
            headers={
                "Accept": "application/json, application/xml;q=0.9",
                "User-Agent": f"{self.tool_parameter}/1.0",
            },
            method="GET",
        )
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                # A retry is a new NCBI request and therefore passes through
                # the same global request-rate boundary as the first attempt.
                _wait_for_pubmed_request_slot(self.minimum_interval_seconds)
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
            except HTTPError as exc:
                last_error = exc
                retryable = exc.code in {408, 409, 425, 429} or (
                    500 <= exc.code <= 599
                )
                if not retryable or attempt >= self.max_retries:
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break

            if attempt < self.max_retries:
                # DIRECT_REUSE_FLOWSTEER: match the gateway's exponential
                # 1x/2x/4x bounded retry schedule, with an injected base delay
                # and sleep function so the behavior is deterministic in
                # local tests.
                delay = min(2.0**attempt, 4.0) * self.retry_backoff_seconds
                if delay > 0:
                    self.sleep(delay)

        if last_error is None:  # pragma: no cover - loop exhaustiveness guard
            raise RuntimeError("NCBI E-utilities request failed without an error")
        raise last_error

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
    max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS

    def __post_init__(self) -> None:
        _validated_max_query_content_tokens(self.max_query_content_tokens)

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "search":
            raise ValueError(
                "HealthBench authoritative backend received an incompatible action"
            )
        if set(request.arguments) != {"query"}:
            raise ValueError("search arguments must contain exactly query")
        query = _required_query(
            request.arguments["query"],
            max_query_content_tokens=self.max_query_content_tokens,
        )

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
                # DIRECT_REUSE_SKILLFLOW: the frozen BM25 implementation
                # already emits these public relevance fields.  Preserve them
                # in the aggregate Tool Observation so the acting Agent can
                # distinguish a strong lexical match from an incidental top-k
                # result without consulting evaluator-only information.
                "score": float(cast(float, chunk["score"])),
                "matched_terms": [
                    str(term)
                    for term in cast(list[object], chunk["matched_terms"])
                ],
                # Reuse the source-read pagination contract for search
                # snippets; do not present an incomplete clause as full text.
                **{
                    key: chunk[key]
                    for key in (
                        "source_id", "version", "content_type", "offset",
                        "total_characters", "truncated", "next_offset",
                    )
                    if key in chunk
                },
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


_GENERIC_TASK_QUERY_TERMS = frozenset(
    {
        "assistant",
        "authoritative",
        "case",
        "care",
        "clinical",
        "content",
        "conversation",
        "current",
        "evidence",
        "final",
        "guideline",
        "guidelines",
        "health",
        "history",
        "management",
        "medical",
        "message",
        "messages",
        "option",
        "options",
        "patient",
        "patients",
        "recommendation",
        "recommendations",
        "role",
        "respond",
        "response",
        "search",
        "therapy",
        "treatment",
        "user",
    }
)


def _query_preserves_task_surface(problem: object, query: object) -> bool:
    """Require one public task anchor in an opted-in retrieval query.

    FlowSteer's QA Tool adapter already rejects query scope loss before a read.
    HealthBench queries are free text rather than entity/relation tuples, so
    this thin adapter checks lexical overlap with the model-visible
    conversation, allowing a narrowly bounded long-word spelling correction.
    It never expands an abbreviation, chooses a diagnosis, or consults a
    rubric/reference answer. Non-English tasks without comparable ASCII
    anchors remain admitted.
    """

    problem_tokens = {
        token
        for token in _normalized_query_tokens(problem)
        if len(token) >= 3 and token not in _GENERIC_TASK_QUERY_TERMS
    }
    if not problem_tokens:
        return True
    query_tokens = {
        token
        for token in _normalized_query_tokens(query)
        if token not in _GENERIC_TASK_QUERY_TERMS
    }
    if problem_tokens & query_tokens:
        return True

    # NECESSARY_PROJECT_ADAPTATION: SkillFlow's lower-case lexical retrieval
    # has no task-surface admission rule. Our exact-only guard can reject a
    # corrected spelling before retrieval. Keep its original exact path and
    # permit only one character edit in a long ordinary word, not acronym or
    # numeric/code substitution. This neither rewrites the query nor changes
    # the upstream BM25 tokenization, scores, or returned evidence.
    from difflib import SequenceMatcher

    def ordinary_long_words(value: object, tokens: set[str]) -> set[str]:
        if not isinstance(value, str):
            return set()
        return {
            word.casefold()
            for word in re.findall(
                r"(?<![A-Za-z0-9_-])[A-Za-z]{6,}(?![A-Za-z0-9_-])",
                value,
            )
            if not word.isupper() and word.casefold() in tokens
        }

    for source in ordinary_long_words(problem, problem_tokens):
        for candidate in ordinary_long_words(query, query_tokens):
            if abs(len(source) - len(candidate)) > 1:
                continue
            comparison = SequenceMatcher(None, source, candidate, autojunk=False)
            if comparison.ratio() < 0.90:
                continue
            edits = sum(
                max(source_end - source_start, target_end - target_start)
                for operation, source_start, source_end, target_start, target_end
                in comparison.get_opcodes()
                if operation != "equal"
            )
            if edits == 1:
                return True
    return False


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


def _normalized_surface(value: object) -> str:
    """Normalize a public entity surface without expanding its semantics."""

    if not isinstance(value, str):
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _retrieval_surface_anchors(
    problem: object,
    query: object,
) -> tuple[str, ...]:
    """Return unresolved/acronym surfaces that this query must preserve.

    This is a deterministic public-input check, not a relevance model.  It
    protects exact named trials, codes, drugs and abbreviations from partial
    lexical matches such as treating a hit on ``DAPT`` as evidence about the
    complete ``STOP-DAPT`` entity.
    """

    if not isinstance(query, str):
        return ()
    surface_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9]*|"
        r"[A-Za-z0-9]+(?:[-–—][A-Za-z0-9]+)+)"
        r"(?![A-Za-z0-9])"
    )
    raw_candidates = list(surface_pattern.findall(query))
    normalized_query = _normalized_surface(query)
    if isinstance(problem, str):
        for candidate in surface_pattern.findall(problem):
            normalized_candidate = _normalized_surface(candidate)
            if (
                normalized_candidate
                and normalized_candidate in normalized_query
            ):
                raw_candidates.append(candidate)
    return tuple(
        dict.fromkeys(
            normalized
            for candidate in raw_candidates
            if (normalized := _normalized_surface(candidate))
        )
    )


def _evidence_preserves_query_anchors(
    request: AgentRequest | None,
    observation: Mapping[str, object],
) -> bool:
    """Distinguish a non-empty retrieval result from anchor-bearing evidence."""

    result = observation.get("result")
    evidence = result.get("evidence") if isinstance(result, Mapping) else None
    if not isinstance(evidence, list) or not evidence:
        return False
    # A discoverable title/identifier is not retrieved source content. Keep
    # metadata hits visible, but do not spend a content-bearing search slot
    # before a source read has supplied an actual excerpt.
    evidence = [item for item in evidence if isinstance(item, Mapping)
                and isinstance(item.get("excerpt"), str) and item["excerpt"].strip()]
    if not evidence:
        return False
    executed = observation.get("executed_action")
    arguments = (
        executed.get("arguments")
        if isinstance(executed, Mapping)
        else None
    )
    query = arguments.get("query") if isinstance(arguments, Mapping) else None
    problem = request.problem if isinstance(request, AgentRequest) else None
    anchors = _retrieval_surface_anchors(problem, query)
    if not anchors:
        return True
    evidence_surfaces = tuple(
        _normalized_surface(
            " ".join(
                str(item.get(field, ""))
                for field in ("title", "excerpt")
            )
        )
        for item in evidence
        if isinstance(item, Mapping)
    )
    return all(
        any(anchor in surface for surface in evidence_surfaces)
        for anchor in anchors
    )


def _routed_evidence_receipts(request: AgentRequest):
    """Reuse the knowledge adapter's bounded, graph-delivered source walk.

    These are evidence inputs, not this invocation's Tool/control history.
    Only Runtime envelopes and their provenance are traversed; never parse an
    Agent's contract or summary into a source, or read global graph outputs.
    """
    yield from request.prior_tool_receipts
    if request.communication_condition is CommunicationCondition.UPSTREAM_MASKED:
        return
    pending = [item.to_dict() for item in request.upstream]
    if request.peer_draft is not None:
        pending.append(request.peer_draft.to_dict())
    seen = set()
    examined = 0
    while pending and examined < 16:
        item = pending.pop(0)
        identity = (item.get("source_agent_id"), item.get("artifact_version"),
                    item.get("graph_revision"), item.get("content", item.get("artifact", "")))
        if identity in seen:
            continue
        seen.add(identity)
        examined += 1
        yield from (receipt for receipt in item.get("tool_receipts", ()) if isinstance(receipt, Mapping))
        pending.extend(value for value in item.get("input_artifact_provenance", ()) if isinstance(value, Mapping))


class HealthBenchAuthoritativeReactExecutionAdapter(ToolReactExecutionAdapter):
    """Bound authoritative retrieval to two distinct searches and completion."""

    def __init__(
        self,
        *,
        require_initial_search: bool = False,
        max_successful_searches: int | None = None,
        require_relevant_evidence: bool = False,
        require_task_query_anchor: bool = False,
        require_refinement_on_insufficient_evidence: bool = False,
        require_structured_evidence_artifact: bool = False,
        require_complete_natural_language_artifact: bool = False,
        enable_evidence_repair_feedback: bool = False,
        completion_quality_profile: str = (
            HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V1
        ),
        max_completion_artifact_characters: int | None = None,
        max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
        **kwargs: Any,
    ) -> None:
        if type(require_initial_search) is not bool:
            raise TypeError("require_initial_search must be boolean")
        if type(enable_evidence_repair_feedback) is not bool:
            raise TypeError("enable_evidence_repair_feedback must be boolean")
        if type(require_relevant_evidence) is not bool:
            raise TypeError("require_relevant_evidence must be boolean")
        if type(require_task_query_anchor) is not bool:
            raise TypeError("require_task_query_anchor must be boolean")
        if type(require_refinement_on_insufficient_evidence) is not bool:
            raise TypeError(
                "require_refinement_on_insufficient_evidence must be boolean"
            )
        if type(require_structured_evidence_artifact) is not bool:
            raise TypeError(
                "require_structured_evidence_artifact must be boolean"
            )
        if type(require_complete_natural_language_artifact) is not bool:
            raise TypeError(
                "require_complete_natural_language_artifact must be boolean"
            )
        if completion_quality_profile not in (
            _HEALTHBENCH_COMPLETION_QUALITY_PROFILES
        ):
            raise ValueError(
                "completion_quality_profile must be a supported HealthBench "
                "completion quality profile"
            )
        if max_completion_artifact_characters is not None and (
            type(max_completion_artifact_characters) is not int
            or max_completion_artifact_characters < 1
        ):
            raise ValueError(
                "max_completion_artifact_characters must be a positive "
                "integer or None"
            )
        maximum_query_content_tokens = _validated_max_query_content_tokens(
            max_query_content_tokens
        )
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
        self._enable_evidence_repair_feedback = enable_evidence_repair_feedback
        self._require_relevant_evidence = require_relevant_evidence
        self._require_task_query_anchor = require_task_query_anchor
        self._require_refinement_on_insufficient_evidence = (
            require_refinement_on_insufficient_evidence
        )
        self._require_structured_evidence_artifact = (
            require_structured_evidence_artifact
        )
        self._require_complete_natural_language_artifact = (
            require_complete_natural_language_artifact
        )
        self._completion_quality_profile = completion_quality_profile
        self._max_successful_searches = max_successful_searches
        self._max_completion_artifact_characters = (
            max_completion_artifact_characters
        )
        self._max_query_content_tokens = maximum_query_content_tokens
        capability = self._tool_registry.require_capability(
            HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
        )
        registered_maximum = _query_content_tokens_from_tool_version(
            capability.version
        )
        if registered_maximum != maximum_query_content_tokens:
            raise ValueError(
                "max_query_content_tokens must match the registered "
                "HealthBench authoritative Tool capability"
            )

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        """Scope public task context to the existing completion-validation hook.

        The shared SkillFlow-compatible hook does not accept a request. Keep
        the request coroutine-local, not mutable adapter state: independent
        nodes may execute concurrently through the same adapter. Parsing,
        recovery, turn budgets and terminal admission remain in the base loop.
        """

        token = _COMPLETION_REQUEST.set(request)
        try:
            return await super().execute(request)
        finally:
            _COMPLETION_REQUEST.reset(token)

    def _completion_arguments_schema(
        self,
        request: AgentRequest,
    ) -> Mapping[str, object]:
        """Bound an opted-in public artifact without choosing its semantics.

        ``None`` deliberately delegates to the exact SkillFlow-compatible
        historical schema.  A configured bound makes the provider-visible
        value type and length explicit.  Output nodes produce the complete
        user-facing assistant response; other nodes produce the intermediate
        contract artifact routed through AgentGraph.
        """

        maximum = self._max_completion_artifact_characters
        if self._require_structured_evidence_artifact and not request.is_output_agent:
            return {
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": "object",
                        "required": [
                            "schema_version",
                            "status",
                            "summary",
                            "evidence_items",
                            "uncertainties",
                        ],
                        "properties": {
                            "schema_version": {
                                "const": "healthbench.structured-evidence.v1"
                            },
                            "status": {
                                "enum": ["supported", "insufficient"]
                            },
                            "summary": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 4000,
                                "description": (
                                    "A concise synthesis that preserves the "
                                    "Agent contract's material findings, "
                                    "conditions, and limitations."
                                ),
                            },
                            "evidence_items": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {
                                    "type": "object",
                                    "required": [
                                        "supported_claim",
                                        "conditions_or_qualifiers",
                                        "document_id",
                                        "source",
                                        "title",
                                        "date",
                                        "url",
                                        "evidence_span",
                                    ],
                                    "properties": {
                                        "supported_claim": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 1200,
                                            "description": (
                                                "One claim supported by the "
                                                "cited evidence span."
                                            ),
                                        },
                                        "conditions_or_qualifiers": {
                                            "type": "string",
                                            "maxLength": 1000,
                                            "description": (
                                                "Material population, timing, "
                                                "dose, exception, or limitation."
                                            ),
                                        },
                                        "document_id": {
                                            "type": "string",
                                            "minLength": 1,
                                            "description": (
                                                "Copy document_id exactly from "
                                                "a successful search Observation."
                                            ),
                                        },
                                        "source": {
                                            "type": "string",
                                            "minLength": 1,
                                            "description": (
                                                "Copy source exactly from the "
                                                "same search evidence item."
                                            ),
                                        },
                                        "title": {
                                            "type": "string",
                                            "minLength": 1,
                                            "description": (
                                                "Copy title exactly from the "
                                                "same search evidence item."
                                            ),
                                        },
                                        "date": {
                                            "type": ["string", "null"],
                                            "description": (
                                                "Copy date exactly from the "
                                                "same search evidence item."
                                            ),
                                        },
                                        "url": {
                                            "type": ["string", "null"],
                                            "description": (
                                                "Copy url exactly from the same "
                                                "search evidence item."
                                            ),
                                        },
                                        "evidence_span": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": 600,
                                            "description": (
                                                "Copy one relevant contiguous "
                                                "span exactly from excerpt."
                                            ),
                                        },
                                    },
                                    "additionalProperties": False,
                                },
                            },
                            "uncertainties": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 600,
                                },
                            },
                        },
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            }
        if maximum is None:
            return super()._completion_arguments_schema(request)
        description = (
            "The complete assistant response to the user's conversation; "
            "preserve every explicit task part and necessary qualification."
            if request.is_output_agent
            else (
                "The completed intermediate artifact required by the Agent "
                "contract for downstream AgentGraph communication."
            )
        )
        if self._require_complete_natural_language_artifact and request.is_output_agent:
            description += (
                " When the user requests an explanation, plan or document, "
                "include its substantive body, not only a title or contents "
                "outline. A genuinely complete short answer remains valid."
            )
        return {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": maximum,
                    "description": description,
                }
            },
            "additionalProperties": False,
        }

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        """Publish an over-limit completion error for same-Agent repair."""

        inherited_error = super()._completion_error(
            action=action,
            artifact=artifact,
            tool_receipts=tool_receipts,
        )
        if inherited_error is not None:
            return inherited_error
        maximum = self._max_completion_artifact_characters
        if maximum is not None and len(artifact) > maximum:
            return f"completion_artifact_exceeds_{maximum}_characters"
        # NECESSARY_PROJECT_ADAPTATION: ToolReactExecutionAdapter already
        # routes completion validation errors back as a public Observation so
        # the same Agent can repair them on its next ReAct turn.  HealthBench
        # trajectories exposed a narrower schema gap: a heading or a label
        # such as ``score`` satisfied the generic non-empty string schema and
        # was therefore published as the user-facing response.  Reject only
        # unmistakably incomplete surface forms here; do not impose a medical
        # role, minimum essay length, rubric vocabulary, or answer content.
        if self._require_complete_natural_language_artifact:
            incomplete_error = self._obviously_incomplete_completion_error(
                artifact,
                completion_quality_profile=self._completion_quality_profile,
                request=_COMPLETION_REQUEST.get(),
            )
            if incomplete_error is not None:
                return incomplete_error
            # SkillFlow's completion-validation boundary turns this public
            # error into an Observation and gives the same ReAct Agent another
            # bounded turn.  A successful Tool call followed by a one-line
            # English keyword fragment is still a retrieval plan, not the
            # completed finding required by the Agent contract.  Keep the
            # check surface-form only: it does not inspect task answers,
            # evaluator fields, medical entities, Agent roles, or topology.
            if any(
                isinstance(receipt, Mapping)
                and receipt.get("error_type") in {None, ""}
                and isinstance(receipt.get("result"), Mapping)
                and receipt["result"].get("completed") is True
                for receipt in tool_receipts
            ):
                fragment_error = self._post_tool_keyword_fragment_error(
                    artifact
                )
                if fragment_error is not None:
                    return fragment_error
        if self._require_structured_evidence_artifact:
            value = (
                action.arguments.get("value")
                if isinstance(action.arguments, Mapping)
                else None
            )
            # Output Agents and Direct use the unchanged natural-language
            # completion schema.  Only an object value opts into this
            # intermediate Artifact validation boundary.
            if isinstance(value, Mapping):
                return self._structured_evidence_artifact_error(
                    value,
                    tool_receipts,
                )
        return None

    @staticmethod
    def _obviously_incomplete_completion_error(
        artifact: str,
        *,
        completion_quality_profile: str = (
            HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V1
        ),
        request: AgentRequest | None = None,
    ) -> str | None:
        """Reject a heading/label without rewriting or scoring its content."""

        normalized = "\n".join(
            line.strip() for line in artifact.strip().splitlines() if line.strip()
        )
        if not normalized:
            return "completion_artifact_is_empty"
        # Structured intermediate evidence has its own authoritative schema
        # below and must not be mistaken for Markdown or a short text label.
        if normalized.startswith("{") or normalized.startswith("["):
            return None
        task = ""
        if request is not None and request.is_output_agent:
            try:
                task = parse_model_visible_conversation(request.problem)[-1]["content"]
            except ValueError:
                task = request.problem
            # Fragment-shaped outputs are legitimate for explicitly requested
            # titles, outlines, lists or short factual values. Never infer a
            # long-answer requirement from the Director's contract instead of
            # the actual public user request.
            if HealthBenchAuthoritativeReactExecutionAdapter._task_allows_fragment_completion(task):
                return None
            # Numeric results and direct yes/no answers are not empty labels;
            # no arbitrary minimum character or word count is an answer test.
            if re.fullmatch(
                r"(?:yes|no|unknown|unclear|none|\d+(?:[.,]\d+)?\s*(?:%|[A-Za-zµμ/]+)?)\.?",
                normalized,
                flags=re.IGNORECASE,
            ):
                return None
        if "\n" not in normalized and re.fullmatch(
            r"#{1,6}\s+[^\n]+",
            normalized,
        ):
            return "completion_artifact_is_heading_only"
        unwrapped = normalized
        while len(unwrapped) >= 4 and (
            (unwrapped.startswith("**") and unwrapped.endswith("**"))
            or (unwrapped.startswith("__") and unwrapped.endswith("__"))
        ):
            unwrapped = unwrapped[2:-2].strip()
        lexical_tokens = re.findall(r"[^\W_]+", unwrapped, flags=re.UNICODE)
        has_sentence_boundary = re.search(r"[.!?。！？](?:[\"'’”)]*)$", unwrapped)
        incomplete_suffixes = (
            (":", "：")
            if completion_quality_profile
            == HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2
            else (":",)
        )
        if "\n" not in normalized and unwrapped.rstrip().endswith(
            incomplete_suffixes
        ):
            return "completion_artifact_is_heading_only"
        # Retain the historical fallback when no Output task is available.
        # With the public task in hand, word count alone cannot distinguish
        # a label from a correct short answer (e.g. a name or an imperative).
        if not task and len(lexical_tokens) <= 3 and has_sentence_boundary is None:
            return "completion_artifact_is_label_only"
        if task:
            return HealthBenchAuthoritativeReactExecutionAdapter._task_body_completion_error(
                normalized, task
            )
        return None

    @staticmethod
    def _task_allows_fragment_completion(task: str) -> bool:
        """Recognize requested output forms, not clinical answers or roles."""

        return re.search(
            r"\b(?:title|heading|headings|outline|table of contents)\s+only\b|"
            r"\b(?:only|just)\s+(?:a\s+|the\s+)?(?:title|heading|outline|name|number|value|list)\b|"
            r"\b(?:write|give|suggest|create|provide|generate)\s+(?:(?:me|a|an|the|short|brief)\s+)*"
            r"(?:title|heading|outline|table of contents)\b|"
            r"\b(?:list|name|enumerate)\s+(?:the|all|some|a|\d+)\b|"
            r"\b(?:one[- ]word|single[- ]word|yes[- /]or[- /]no|yes/no|number only)\b|"
            r"(?:只(?:要|需|给|写|列出)|仅(?:需|给|写|列出)).{0,12}(?:标题|提纲|目录|名称|数字)|"
            r"(?:拟|写|给出|生成)(?:一个)?(?:标题|提纲|目录)",
            task,
            flags=re.IGNORECASE,
        ) is not None

    @staticmethod
    def _task_body_completion_error(artifact: str, task: str) -> str | None:
        """Reject all-heading scaffolds when the public task asks for a body.

        NECESSARY_PROJECT_ADAPTATION: upstream BoundedAgent calls environment
        validate_completion and returns a non-terminal schema_invalid
        Observation on rejection. This is a narrow HealthBench surface-form
        validator at that same boundary, not a medical correctness judge.
        It does not require named roles, tools, sections or a minimum length.
        """

        if re.search(
            r"\b(?:explain|describe|discuss|compare|summari[sz]e|write|draft|prepare|"
            r"recommend|manage|management)\b|\bhow\s+(?:to|do|does|should|would|can)\b|"
            r"(?:解释|说明|讨论|比较|总结|撰写|制定|如何|怎样)",
            task,
            flags=re.IGNORECASE,
        ) is None:
            return None
        small_words = {
            "a", "an", "the", "and", "or", "but", "for", "nor", "of", "on", "in",
            "to", "at", "by", "from", "with", "as", "vs", "versus", "without",
        }
        # A finite predicate or imperative can make even a very short line a
        # completed answer. Title case alone must not reject "No Evidence Was
        # Found" or "Seek Urgent Care" just because they use capitals.
        predicate = re.compile(
            r"\b(?:is|are|was|were|has|have|had|should|must|may|might|can|could|"
            r"cannot|will|would|recommend(?:s|ed)?|avoid|seek|refer|start|stop|"
            r"continue|consider|monitor|use|give|take|call|consult|assess|treat|"
            r"shows?|indicates?|supports?|suggests?|requires?)\b",
            flags=re.IGNORECASE,
        )
        for line in artifact.splitlines():
            stripped = line.strip()
            if not stripped or re.fullmatch(r"[-=_*]{3,}", stripped):
                continue
            if re.fullmatch(r"#{1,6}\s+.+", stripped):
                continue
            stripped = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", stripped)
            stripped = stripped.strip("*_ ")
            if predicate.search(stripped):
                return None
            if stripped.endswith((":", "：")):
                continue
            words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", stripped)
            content_words = [word for word in words if word.casefold() not in small_words]
            if not content_words or not all(word[0].isupper() for word in content_words):
                return None
            # Mixed-language text is not safely classified by English casing.
            if re.search(r"[^\x00-\x7f]", re.sub(r"[–—‘’“”]", "", stripped)):
                return None
        return "completion_artifact_requires_substantive_body"

    @staticmethod
    def _model_visible_observations(
        observations: list[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        visible = ToolReactExecutionAdapter._model_visible_observations(observations)
        for observation in visible:
            if observation.get("public_error_code") == "completion_artifact_requires_substantive_body":
                observation["repair_instruction"] = (
                    "The submitted artifact contains only a title, headings or "
                    "contents outline, but the user requested an explanation, "
                    "plan or document. Complete the requested body in your next "
                    "completion, using available evidence and explicit limits "
                    "where needed. A concise substantive answer is valid; no "
                    "minimum length, extra Agent or Tool call is required."
                )
        return visible

    @staticmethod
    def _post_tool_keyword_fragment_error(artifact: str) -> str | None:
        """Reject an unmistakable query-like fragment after Tool execution."""

        normalized = " ".join(artifact.strip().split())
        if not normalized or "\n" in artifact:
            return None
        lexical_tokens = re.findall(
            r"[A-Za-z]+(?:['’-][A-Za-z]+)?|\d+(?:\.\d+)?",
            normalized,
        )
        if len(lexical_tokens) < 9:
            return None
        # Full sentences, clauses, and structured list items remain legal.
        if re.search(r"[.!?;:。！？；：]", normalized) is not None:
            return None
        predicate_or_finding = re.compile(
            r"\b(?:is|are|was|were|be|been|being|has|have|had|"
            r"shows?|showed|shown|indicates?|indicated|supports?|supported|"
            r"finds?|found|reports?|reported|recommends?|recommended|"
            r"suggests?|suggested|confirms?|confirmed|requires?|required|"
            r"should|must|may|might|can|could|cannot|insufficient|"
            r"unclear|unknown|unresolved)\b",
            flags=re.IGNORECASE,
        )
        if predicate_or_finding.search(normalized) is None:
            return "completion_artifact_is_post_tool_keyword_fragment"
        return None

    @staticmethod
    def _successful_search_evidence(
        tool_receipts: list[dict[str, object]],
    ) -> tuple[Mapping[str, object], ...]:
        evidence_items: list[Mapping[str, object]] = []
        for receipt in tool_receipts:
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("tool_id")
                != HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
                or receipt.get("error_type") is not None
            ):
                continue
            result = receipt.get("result")
            if not isinstance(result, Mapping) or result.get("completed") is not True:
                continue
            value = result.get("value") if isinstance(result, Mapping) else None
            if not isinstance(value, Mapping) or value.get("operation") != "search":
                continue
            evidence = value.get("evidence")
            if not isinstance(evidence, list):
                continue
            evidence_items.extend(
                item for item in evidence if isinstance(item, Mapping)
            )
        return tuple(evidence_items)

    @staticmethod
    def _normalized_evidence_text(value: object) -> str:
        return " ".join(str(value).split()) if isinstance(value, str) else ""

    def _action_error_feedback(
        self, *, request, action, public_error_code, tool_receipts, observations,
    ) -> Mapping[str, object]:
        """Explain exact receipt mismatches without choosing an answer.

        Necessary HealthBench adaptation of SkillFlow BoundedAgent's public
        invalid-action Observation. Sources come only from observed Tool
        results and Runtime-delivered provenance. Never repair a submitted
        claim, silently canonicalize its metadata, or relax span validation.
        """
        inherited = super()._action_error_feedback(
            request=request, action=action, public_error_code=public_error_code,
            tool_receipts=tool_receipts, observations=observations,
        )
        if not self._enable_evidence_repair_feedback or public_error_code not in {
            "structured_evidence_item_receipt_binding_invalid",
            "structured_evidence_item_span_not_in_receipt",
            "structured_evidence_item_duplicate",
        }:
            return inherited
        value = action.arguments.get("value")
        items = value.get("evidence_items") if isinstance(value, Mapping) else None
        if not isinstance(items, list):
            return inherited
        sources = self._successful_search_evidence(tool_receipts)
        sources += self._successful_search_evidence(list(_routed_evidence_receipts(request)))
        fields = ("document_id", "source", "title", "date", "url")
        seen = set()
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            span = self._normalized_evidence_text(item.get("evidence_span"))
            identity = (item.get("document_id"), span.casefold())
            same_document = tuple(row for row in sources if row.get("document_id") == item.get("document_id"))
            matching = tuple(row for row in same_document if all(row.get(field) == item.get(field) for field in fields))
            mismatch = not matching
            span_invalid = matching and not any(span and span in self._normalized_evidence_text(row.get("excerpt")) for row in matching)
            if identity in seen or mismatch or span_invalid:
                context: dict[str, object] = {"evidence_item_index": index, "index_base": 0}
                if identity in seen:
                    context["mismatched_fields"] = ["duplicate_evidence_item"]
                elif not same_document:
                    context["mismatched_fields"] = ["document_id"]
                    context["observed_document_ids"] = list(dict.fromkeys(
                        str(row["document_id"]) for row in sources if row.get("document_id")
                    ))[:6]
                else:
                    # At most two actual variants of this document, not the
                    # full source collection. A preview is continuous source
                    # text, not an assertion that it supports the model claim.
                    candidates = []
                    for row in matching or same_document:
                        if not any(all(previous.get(field) == row.get(field) for field in (*fields, "excerpt")) for previous in candidates):
                            candidates.append(row)
                        if len(candidates) == 2:
                            break
                    context["receipt_candidates"] = [{
                        "canonical_metadata": {field: row.get(field) for field in fields},
                        "mismatched_fields": [field for field in fields if row.get(field) != item.get(field)]
                            + (["evidence_span"] if span_invalid else []),
                        "verbatim_excerpt_preview": str(row.get("excerpt") or "")[:400],
                        "preview_is_complete_excerpt": len(str(row.get("excerpt") or "")) <= 400,
                        **{field: row[field] for field in ("source_id", "full_text_source_id", "truncated", "next_offset") if field in row},
                    } for row in candidates]
                return {
                    "repair_context": context,
                    "repair_instruction": (
                        "Repair the indicated evidence item using one actual receipt: copy its metadata exactly "
                        "and one contiguous span from its excerpt; do not join phrases with ellipses or paraphrase a quote. "
                        "The preview is source text, not a suggested claim. Use source.read with an observed source_id "
                        "and next_offset if needed and admitted; metadata-only hits do not establish clinical findings. "
                        "Keep supported content; state unresolved limits instead of inventing evidence."
                    ),
                }
            seen.add(identity)
        return inherited

    def _structured_evidence_artifact_error(
        self,
        value: Mapping[str, object],
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        """Bind every model-visible evidence item to a successful receipt."""

        required_keys = {
            "schema_version",
            "status",
            "summary",
            "evidence_items",
            "uncertainties",
        }
        if set(value) != required_keys:
            return "structured_evidence_artifact_fields_invalid"
        if value.get("schema_version") != "healthbench.structured-evidence.v1":
            return "structured_evidence_artifact_version_invalid"
        status = value.get("status")
        summary = value.get("summary")
        evidence_items = value.get("evidence_items")
        uncertainties = value.get("uncertainties")
        if status not in {"supported", "insufficient"}:
            return "structured_evidence_artifact_status_invalid"
        if not isinstance(summary, str) or not summary.strip():
            return "structured_evidence_artifact_summary_invalid"
        if (
            not isinstance(evidence_items, list)
            or len(evidence_items) > 6
            or not isinstance(uncertainties, list)
            or len(uncertainties) > 6
            or any(
                not isinstance(item, str) or not item.strip()
                for item in uncertainties
            )
        ):
            return "structured_evidence_artifact_collection_invalid"
        if status == "supported" and not evidence_items:
            return "structured_evidence_artifact_requires_evidence"
        if status == "insufficient" and not uncertainties:
            return "structured_evidence_artifact_requires_uncertainty"
        if (
            self._require_refinement_on_insufficient_evidence
            and status == "insufficient"
        ):
            dispatched_searches, successful_searches = (
                self._search_receipt_counts(tool_receipts)
            )
            if (
                dispatched_searches < self._max_tool_calls
                and successful_searches < self._max_successful_searches
            ):
                return "insufficient_evidence_requires_distinct_refined_search"

        receipt_evidence = self._successful_search_evidence(tool_receipts)
        request = _COMPLETION_REQUEST.get()
        if request is not None:
            # DIRECT_REUSE: successful-source projection plus the existing
            # knowledge adapter's routed-envelope boundary. Historical and
            # peer sources are visible inputs, but not newly dispatched calls:
            # do not add them to tool_receipts or the budget checks above.
            receipt_evidence += self._successful_search_evidence(
                list(_routed_evidence_receipts(request))
            )
        expected_item_keys = {
            "supported_claim",
            "conditions_or_qualifiers",
            "document_id",
            "source",
            "title",
            "date",
            "url",
            "evidence_span",
        }
        seen: set[tuple[str, str]] = set()
        for item in evidence_items:
            if not isinstance(item, Mapping) or set(item) != expected_item_keys:
                return "structured_evidence_item_fields_invalid"
            if any(
                not isinstance(item.get(field), str)
                or not str(item.get(field)).strip()
                for field in (
                    "supported_claim",
                    "document_id",
                    "source",
                    "title",
                    "evidence_span",
                )
            ) or not isinstance(item.get("conditions_or_qualifiers"), str):
                return "structured_evidence_item_text_invalid"
            if item.get("date") is not None and not isinstance(item.get("date"), str):
                return "structured_evidence_item_date_invalid"
            if item.get("url") is not None and not isinstance(item.get("url"), str):
                return "structured_evidence_item_url_invalid"

            document_id = str(item["document_id"])
            normalized_span = self._normalized_evidence_text(
                item["evidence_span"]
            )
            identity = (document_id, normalized_span.casefold())
            if identity in seen:
                return "structured_evidence_item_duplicate"
            seen.add(identity)
            matching_receipts = tuple(
                receipt_item
                for receipt_item in receipt_evidence
                if receipt_item.get("document_id") == item.get("document_id")
                and receipt_item.get("source") == item.get("source")
                and receipt_item.get("title") == item.get("title")
                and receipt_item.get("date") == item.get("date")
                and receipt_item.get("url") == item.get("url")
            )
            if not matching_receipts:
                return "structured_evidence_item_receipt_binding_invalid"
            if not any(
                normalized_span
                and normalized_span
                in self._normalized_evidence_text(
                    receipt_item.get("excerpt")
                )
                for receipt_item in matching_receipts
            ):
                return "structured_evidence_item_span_not_in_receipt"
        return None

    @staticmethod
    def _search_receipt_counts(
        tool_receipts: list[dict[str, object]],
    ) -> tuple[int, int]:
        """Count dispatched and evidence-bearing public search receipts."""

        dispatched = 0
        successful = 0
        for receipt in tool_receipts:
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("tool_id")
                != HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            ):
                continue
            dispatched += 1
            if receipt.get("error_type") is not None:
                continue
            result = receipt.get("result")
            if not isinstance(result, Mapping) or result.get("completed") is not True:
                continue
            value = result.get("value", result)
            evidence = value.get("evidence") if isinstance(value, Mapping) else None
            if isinstance(evidence, list) and evidence:
                successful += 1
        return dispatched, successful

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
                if (
                    isinstance(evidence, list)
                    and evidence
                    and (
                        not self._require_relevant_evidence
                        or _evidence_preserves_query_anchors(
                            request,
                            observation,
                        )
                    )
                ):
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
        unresolved_initial_search = (
            successful == 0
            if self._require_relevant_evidence
            else dispatched == 0
        )
        completion_admitted = not (
            self._require_initial_search
            and unresolved_initial_search
            and search_admitted
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

        if (
            action.resource_id != HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID
            or action.name != "search"
            or not isinstance(action.arguments, dict)
        ):
            return None
        query = action.arguments.get("query")
        query_tokens = _normalized_query_tokens(query)
        if not query_tokens:
            return "query_must_include_clinical_content_term"
        if len(query_tokens) > self._max_query_content_tokens:
            return (
                "query_too_broad_use_at_most_"
                f"{self._max_query_content_tokens}_clinical_terms"
            )
        if self._require_task_query_anchor and not _query_preserves_task_surface(
            request.problem,
            query,
        ):
            return "query_does_not_preserve_public_task_anchor"
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
            "source_id": {"type": "string"},
            "version": {"type": "string"},
            "content_type": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "total_characters": {"type": "integer", "minimum": 0},
            "truncated": {"type": "boolean"},
            "next_offset": {"type": ["integer", "null"], "minimum": 0},
            # Optional publication metadata from the new external adapters.
            "pmid": {"type": ["string", "null"]},
            "pmcid": {"type": ["string", "null"]},
            "doi": {"type": ["string", "null"]},
            "full_text_source_id": {"type": ["string", "null"]},
            "publication_types": {"type": "array", "items": {"type": "string"}},
            "is_open_access": {"type": "boolean"},
            "is_preprint": {"type": "boolean"},
            # Bookshelf search metadata is not chapter text; source/version
            # distinctions survive source.read and downstream projections.
            "matched_title": {"type": "string"},
            "book_title": {"type": ["string", "null"]},
            "full_text_availability": {"type": "string"},
            "text_scope": {"type": "string"},
            "collection": {"type": "string"},
            "repository_datestamp": {"type": ["string", "null"]},
            # SkillFlow BM25 fields are present only on frozen-textbook
            # evidence.  PubMed evidence therefore keeps them optional under
            # the shared evidence schema.
            "score": {"type": "number"},
            "matched_terms": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def build_healthbench_authoritative_tool_registry(
    corpus: FrozenMedRAGBM25Corpus,
    *,
    pubmed_client: PubMedEUtilitiesClient | None = None,
    timeout_seconds: float = 20.0,
    max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
) -> ToolRegistry:
    """Register the single HealthBench-only aggregate evidence Tool."""

    if not isinstance(corpus, FrozenMedRAGBM25Corpus):
        raise TypeError("corpus must be a FrozenMedRAGBM25Corpus")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    maximum_query_content_tokens = _validated_max_query_content_tokens(
        max_query_content_tokens
    )
    client = pubmed_client or PubMedEUtilitiesClient()
    if not isinstance(client, PubMedEUtilitiesClient):
        raise TypeError("pubmed_client must be a PubMedEUtilitiesClient")

    query_description = (
        (
            "Search with a short named clinical entity and one target "
            "relation. Do not copy the full conversation. Use no more than "
            f"{maximum_query_content_tokens} clinical content terms. "
        )
        if maximum_query_content_tokens
        != AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS
        else (
            "Search with concise English clinical terminology. Use specific "
            "conditions, interventions, outcomes, populations, or safety "
            "concepts rather than copying the full conversation. Use no more "
            f"than {maximum_query_content_tokens} clinical content terms. "
        )
    )
    search_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": AUTHORITATIVE_QUERY_MAX_CHARACTERS,
                "description": query_description
                + (
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
            + (
                "+pubmed-transport-retry-"
                f"max-{client.max_retries}-exp2-"
                f"base-{client.retry_backoff_seconds:g}s"
                if client.max_retries > 0
                else ""
            )
            + (
                "+query-content-cap-"
                f"{maximum_query_content_tokens}"
                if maximum_query_content_tokens
                != AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS
                else ""
            )
        ),
    )
    return ToolRegistry(
        (
            ToolRegistration(
                HEALTHBENCH_AUTHORITATIVE_SEARCH_TOOL_ID,
                HealthBenchAuthoritativeSearchToolBackend(
                    corpus,
                    client,
                    maximum_query_content_tokens,
                ),
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
    max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
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
            max_query_content_tokens=max_query_content_tokens,
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
    "HEALTHBENCH_COMPLETION_ARTIFACT_MAX_CHARACTERS_V1",
    "HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V1",
    "HEALTHBENCH_COMPLETION_QUALITY_PROFILE_V2",
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
