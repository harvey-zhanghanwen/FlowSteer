"""Optional HealthBench evidence tools, not native benchmark capabilities.

DIRECT_REUSE: the frozen SkillFlow BM25/NCBI aggregate registrations, existing
ToolRegistry lifetime, and SkillFlow ``training/tools.py`` calculator backend.
NECESSARY_PROJECT_ADAPTATION: source-ID lookup and DailyMed's documented
``/spls.json`` -> ``/spls/{SETID}.xml`` protocol have no upstream adapter.
Neither tool takes task records, physician responses, nor evaluator fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .computation_tools import (
    AIME_CALCULATOR_TOOL_ID,
    create_aime_computation_registry,
)
from .healthbench_evidence_adapter import (
    AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
    OpenHealthBenchAuthoritativeToolRegistry,
    PubMedEUtilitiesClient,
    _evidence_schema,
    _node_text,
    _publication_date,
    build_healthbench_authoritative_tool_registry,
)
from .healthbench_tool_adapter import (
    HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
    FrozenMedRAGBM25Corpus,
    build_healthbench_medrag_tool_registry,
)
from .tool_runtime import ToolCapability, ToolRegistration, ToolRegistry, ToolRequest, ToolResult


HEALTHBENCH_SOURCE_READ_TOOL_ID = "healthbench-source.read"
HEALTHBENCH_DRUG_LOOKUP_TOOL_ID = "healthbench-drug.lookup"
HEALTHBENCH_CALCULATOR_TOOL_ID = "healthbench-computation.calculator"
HEALTHBENCH_LITERATURE_SEARCH_TOOL_ID = "healthbench-literature.search"
HEALTHBENCH_TRIAL_SEARCH_TOOL_ID = "healthbench-trials.search"
HEALTHBENCH_BOOKSHELF_SEARCH_TOOL_ID = "healthbench-bookshelf.search"
HEALTHBENCH_PDQ_SEARCH_TOOL_ID = "healthbench-pdq.search"
HEALTHBENCH_AHRQ_SEARCH_TOOL_ID = "healthbench-ahrq.search"
HEALTHBENCH_TERMINOLOGY_SEARCH_TOOL_ID = "healthbench-terminology.search"
HEALTHBENCH_CLINICAL_TOOL_VERSION = "healthbench-clinical-tools-v1"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
SOURCE_PAGE_CHARACTERS = 16_000
_SETID_PATTERN = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_SPL_NS = {"h": "urn:hl7-org:v3"}


def _text_argument(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _source_page(evidence: dict[str, object], offset: int = 0) -> dict[str, object]:
    """Bound model context with explicit continuation, not silent truncation."""
    text = str(evidence["excerpt"])
    if type(offset) is not int or offset < 0 or offset > len(text):
        raise ValueError("source offset is outside the document")
    end = min(len(text), offset + SOURCE_PAGE_CHARACTERS)
    return {
        **evidence,
        "excerpt": text[offset:end],
        "rank": 1,
        "offset": offset,
        "total_characters": len(text),
        "truncated": end < len(text),
        "next_offset": end if end < len(text) else None,
    }


def _source_receipt(source_type: str, source: str, count: int, error: Exception | None = None) -> dict[str, object]:
    return {
        "source_type": source_type,
        "source": source,
        "status": "error" if error else "success",
        "result_count": count,
        "error_type": type(error).__name__ if error else None,
    }


@dataclass(frozen=True, slots=True)
class DailyMedClient:
    """Read FDA SPL label text through the official NLM DailyMed API.

    Sources: DailyMed webservices-help/v2/spls_api.cfm and
    spls_setid_api.cfm. Search metadata is never presented as label evidence.
    The candidate count is bounded; a name match is not an indication that
    products/formulations are interchangeable or that interactions are absent.
    """

    timeout_seconds: float = 8.0
    retmax: int = 2
    opener: Callable[..., object] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("DailyMed timeout must be positive")
        if type(self.retmax) is not int or not 1 <= self.retmax <= 3:
            raise ValueError("DailyMed retmax must be from 1 through 3")
        if not callable(self.opener):
            raise TypeError("DailyMed opener must be callable")

    def _get(self, endpoint: str, parameters: Mapping[str, object] | None = None) -> bytes:
        url = f"{DAILYMED_BASE_URL}/{endpoint}"
        if parameters:
            url += "?" + urlencode(parameters)
        request = Request(url, headers={"Accept": "application/json, application/xml;q=0.9"}, method="GET")
        response = self.opener(request, timeout=self.timeout_seconds)
        try:
            payload = response.read()  # type: ignore[attr-defined]
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if not isinstance(payload, bytes):
            raise RuntimeError("DailyMed response is not bytes")
        return payload

    def read(self, setid: str, *, published_date: str | None = None) -> dict[str, object]:
        if not _SETID_PATTERN.fullmatch(setid):
            raise ValueError("DailyMed source requires an SPL SETID")
        root = ElementTree.fromstring(self._get(f"spls/{setid}.xml"))
        actual_setid = root.find("h:setId", _SPL_NS)
        if actual_setid is None or actual_setid.get("root", "").casefold() != setid.casefold():
            raise RuntimeError("DailyMed label does not identify the requested SETID")
        sections = []
        for section in root.findall(".//h:structuredBody//h:section", _SPL_NS):
            body = section.find("h:text", _SPL_NS)
            if body is None:
                continue
            body_text = " ".join(" ".join(body.itertext()).split())
            if body_text:
                title = _node_text(section.find("h:title", _SPL_NS))
                sections.append((title + "\n" if title else "") + body_text)
        if not sections:
            raise RuntimeError("DailyMed SPL contains no readable label sections")
        effective = root.find("h:effectiveTime", _SPL_NS)
        version = root.find("h:versionNumber", _SPL_NS)
        return {
            "source_type": "drug_label",
            "source": "NLM DailyMed",
            "document_id": setid,
            "source_id": f"dailymed:{setid}",
            "title": _node_text(root.find("h:title", _SPL_NS)),
            "date": effective.get("value") if effective is not None else None,
            "published_date": published_date,
            "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
            "excerpt": "\n\n".join(sections),
            "content_type": "structured_product_label_sections",
            "version": version.get("value") if version is not None else None,
        }

    def lookup(self, drug_name: str) -> dict[str, object]:
        drug_name = _text_argument(drug_name, "drug_name")
        if len(drug_name) > 160:
            raise ValueError("drug_name exceeds 160 characters")
        payload = json.loads(self._get("spls.json", {"drug_name": drug_name, "pagesize": self.retmax, "page": 1}))
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise RuntimeError("DailyMed search response is invalid")
        evidence, receipts = [], []
        seen: set[str] = set()
        for row in rows[:self.retmax]:
            if not isinstance(row, Mapping) or not isinstance(row.get("setid"), str):
                raise RuntimeError("DailyMed search entry lacks a SETID")
            setid = row["setid"]
            if setid in seen:
                continue
            seen.add(setid)
            try:
                published = row.get("published_date")
                label = self.read(setid, published_date=published if isinstance(published, str) else None)
            except Exception as exc:
                receipts.append(_source_receipt("drug_label", f"NLM DailyMed:{setid}", 0, exc))
                continue
            evidence.append({**_source_page(label), "rank": len(evidence) + 1})
            receipts.append(_source_receipt("drug_label", f"NLM DailyMed:{setid}", 1))
        if not rows:
            receipts.append(_source_receipt("drug_label", "NLM DailyMed", 0))
        return {
            "operation": "drug_lookup",
            "query": drug_name,
            "evidence": evidence,
            "source_receipts": receipts,
        }


@dataclass(frozen=True, slots=True)
class HealthBenchSourceReadToolBackend:
    corpus: FrozenMedRAGBM25Corpus
    pubmed: PubMedEUtilitiesClient
    drug_client: DailyMedClient
    literature_client: object | None = None
    trial_client: object | None = None
    bookshelf_client: object | None = None
    terminology_client: object | None = None

    def _read_pubmed(self, pmid: str) -> dict[str, object]:
        if not pmid.isdigit():
            raise ValueError("PubMed source requires a PMID")
        # DIRECT_REUSE: same NCBI transport, endpoint and date projection as
        # PubMedEUtilitiesClient.search; exact ID avoids a second search.
        root = ElementTree.fromstring(self.pubmed._get("efetch.fcgi", {"id": pmid, "retmode": "xml", "rettype": "abstract"}))
        for article in root.findall(".//PubmedArticle"):
            if _node_text(article.find(".//MedlineCitation/PMID")) != pmid:
                continue
            parts = []
            for part in article.findall(".//Article/Abstract/AbstractText"):
                content = _node_text(part)
                if content:
                    label = part.get("Label")
                    parts.append((label + ": " if label else "") + content)
            if not parts:
                raise RuntimeError("PubMed record contains no abstract")
            return {
                "source_type": "peer_reviewed_literature",
                "source": "NCBI PubMed",
                "document_id": pmid,
                "source_id": f"pubmed:{pmid}",
                "title": _node_text(article.find(".//Article/ArticleTitle")),
                "date": _publication_date(article),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "excerpt": "\n".join(parts),
                "content_type": "pubmed_abstract_not_full_text",
                "version": None,
            }
        raise LookupError("requested PubMed record was not returned")

    def _read(self, source_id: str) -> dict[str, object]:
        # Minimal adapter: the upstream frozen corpus exposes search snippets
        # but no full-chunk ID lookup. Do not modify its ranking or lifecycle.
        document_id = source_id.removeprefix("medrag:")
        with self.corpus._lock:
            if self.corpus.closed:
                raise RuntimeError("MedRAG BM25 corpus is closed")
            try:
                index = self.corpus._document_ids.index(document_id)
            except ValueError:
                index = None
            if index is not None:
                return {
                    "source_type": "frozen_medical_textbook",
                    "source": self.corpus.source_identity,
                    "document_id": document_id,
                    "source_id": f"medrag:{document_id}",
                    "title": self.corpus._titles[index],
                    "date": None,
                    "url": None,
                    "excerpt": self.corpus._corpus[index],
                    "content_type": "frozen_textbook_chunk",
                    "version": self.corpus.source_revision,
                }
        if source_id.startswith("pubmed:") or source_id.isdigit():
            return self._read_pubmed(source_id.removeprefix("pubmed:"))
        if source_id.startswith("dailymed:") or _SETID_PATTERN.fullmatch(source_id):
            return self.drug_client.read(source_id.removeprefix("dailymed:"))
        if self.literature_client is not None and source_id.startswith(("europepmc:", "pmc:")):
            return self.literature_client.read(source_id)
        if self.trial_client is not None and source_id.startswith("clinicaltrials:"):
            return self.trial_client.read(source_id)
        if self.bookshelf_client is not None and source_id.startswith("bookshelf:"):
            return self.bookshelf_client.read(source_id)
        if self.terminology_client is not None and source_id.startswith("mesh:"):
            return self.terminology_client.read(source_id)
        raise LookupError("source_id is not a known MedRAG document, PMID, or DailyMed SETID")

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "read_source" or not {"source_id"}.issubset(request.arguments) or set(request.arguments) - {"source_id", "offset"}:
            raise ValueError("read_source requires source_id and optional offset")
        source_id = _text_argument(request.arguments["source_id"], "source_id")
        offset = request.arguments.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("source offset must be a non-negative integer")
        evidence = _source_page(self._read(source_id), offset)
        return ToolResult({
            "operation": "read_source",
            "query": source_id,
            "evidence": [evidence],
            "source_receipts": [_source_receipt(str(evidence["source_type"]), str(evidence["source"]), 1)],
        })


@dataclass(frozen=True, slots=True)
class HealthBenchDrugLookupToolBackend:
    client: DailyMedClient

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "drug_lookup" or set(request.arguments) != {"drug_name"}:
            raise ValueError("drug_lookup requires exactly drug_name")
        return ToolResult(self.client.lookup(_text_argument(request.arguments["drug_name"], "drug_name")))


@dataclass(frozen=True, slots=True)
class HealthBenchExternalSearchToolBackend:
    """Thin transport adapter; reuse ToolRegistry/ReAct and evidence receipts."""

    client: object

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "search" or set(request.arguments) != {"query"}:
            raise ValueError("external search requires exactly query")
        query = _text_argument(request.arguments["query"], "query")
        if len(query) > 160:
            raise ValueError("external search query exceeds 160 characters")
        return ToolResult(self.client.search(query))


def _clinical_output_schema() -> dict[str, object]:
    evidence = _evidence_schema()
    evidence["properties"].update({
        "source_id": {"type": "string"},
        "content_type": {"type": "string"},
        "version": {"type": ["string", "null"]},
        "published_date": {"type": ["string", "null"]},
        "offset": {"type": "integer", "minimum": 0},
        "next_offset": {"type": ["integer", "null"]},
        "total_characters": {"type": "integer", "minimum": 0},
        "truncated": {"type": "boolean"},
    })
    return {
        "type": "object", "additionalProperties": False,
        "required": ["operation", "query", "evidence", "source_receipts"],
        "properties": {
            "operation": {"enum": ["read_source", "drug_lookup"]},
            "query": {"type": "string"},
            "evidence": {"type": "array", "items": evidence},
            "source_receipts": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["source_type", "source", "status", "result_count", "error_type"],
                "properties": {
                    "source_type": {"type": "string"}, "source": {"type": "string"},
                    "status": {"enum": ["success", "error"]},
                    "result_count": {"type": "integer", "minimum": 0},
                    "error_type": {"type": ["string", "null"]},
                },
            }},
        },
    }


def build_healthbench_clinical_tool_registry(
    corpus: FrozenMedRAGBM25Corpus,
    *,
    pubmed_client: PubMedEUtilitiesClient | None = None,
    timeout_seconds: float = 30.0,
    max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
    drug_client: DailyMedClient | None = None,
    external_medical_sources_enabled: bool = False,
    literature_client: object | None = None,
    trial_client: object | None = None,
    clinical_reference_sources_enabled: bool = False,
    bookshelf_client: object | None = None,
    pdq_client: object | None = None,
    ahrq_client: object | None = None,
    terminology_client: object | None = None,
) -> ToolRegistry:
    """Compose existing registrations plus optional source/label/calculator."""
    pubmed = pubmed_client or PubMedEUtilitiesClient()
    drugs = drug_client or DailyMedClient()
    if external_medical_sources_enabled:
        # Necessary API adapters: neither upstream implements these services.
        from .healthbench_europe_pmc import EuropePMCClient
        from .healthbench_clinical_trials import ClinicalTrialsClient
        literature_client = literature_client or EuropePMCClient()
        trial_client = trial_client or ClinicalTrialsClient()
    else:
        literature_client = trial_client = None
    if clinical_reference_sources_enabled:
        # Necessary transport adapters; reuse the same registry/receipt loop.
        from .healthbench_bookshelf import BookshelfClient
        from .healthbench_mesh import MeSHClient
        bookshelf_client = bookshelf_client or BookshelfClient()
        pdq_client = pdq_client or BookshelfClient(collection="pdq")
        ahrq_client = ahrq_client or BookshelfClient(collection="ahrq")
        terminology_client = terminology_client or MeSHClient()
    else:
        bookshelf_client = pdq_client = ahrq_client = terminology_client = None
    registries = (
        build_healthbench_authoritative_tool_registry(corpus, pubmed_client=pubmed, timeout_seconds=timeout_seconds, max_query_content_tokens=max_query_content_tokens),
        build_healthbench_medrag_tool_registry(corpus, timeout_seconds=timeout_seconds),
    )
    registrations = [ToolRegistration(tool_id, registry._backend(tool_id), registry.require_capability(tool_id)) for registry in registries for tool_id in registry.resource_ids]
    inputs = {
        "read_source": {
            "type": "object", "additionalProperties": False,
            "required": ["source_id"],
            "properties": {
                "source_id": {"type": "string", "minLength": 1, "description": "Exact MedRAG document_id, pubmed:PMID (abstract only), or dailymed:SETID (product label). Not an arbitrary URL."},
                "offset": {"type": "integer", "minimum": 0, "description": "Use next_offset from a truncated source page; omitted means 0."},
            },
        },
        "drug_lookup": {
            "type": "object", "additionalProperties": False,
            "required": ["drug_name"],
            "properties": {"drug_name": {"type": "string", "minLength": 1, "maxLength": 160, "description": "Generic or brand name; returns bounded DailyMed product-label candidates, not patient-specific advice or a complete interaction checker."}},
        },
    }
    if external_medical_sources_enabled:
        inputs["read_source"]["properties"]["source_id"]["description"] += (
            " Also accepts europepmc:MED:PMID for an abstract, pmc:PMCID for available "
            "open-access article XML, and clinicaltrials:NCT######## for a registered "
            "study and any posted results. Use IDs returned by search."
        )
    if clinical_reference_sources_enabled:
        inputs["read_source"]["properties"]["source_id"]["description"] += (
            " Also accepts bookshelf:NBK identifiers for available public-domain/open-access "
            "Bookshelf documents, and mesh:D identifiers for terminology definitions. "
            "A catalogue hit is not a clinical recommendation; terminology is not evidence of efficacy."
        )
    for tool_id, action, backend in (
        (HEALTHBENCH_SOURCE_READ_TOOL_ID, "read_source", HealthBenchSourceReadToolBackend(
            corpus, pubmed, drugs, literature_client, trial_client, bookshelf_client, terminology_client)),
        (HEALTHBENCH_DRUG_LOOKUP_TOOL_ID, "drug_lookup", HealthBenchDrugLookupToolBackend(drugs)),
    ):
        registrations.append(ToolRegistration(tool_id, backend, ToolCapability(
            tool_id=tool_id, dataset_scope=HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
            action_schemas={action: inputs[action]}, input_schema=inputs[action],
            output_schema=_clinical_output_schema(), side_effect="none",
            timeout_seconds=timeout_seconds, version=HEALTHBENCH_CLINICAL_TOOL_VERSION,
        )))
    # Direct backend/schema reuse: only the dataset capability/ID changes.
    calculator_registry = create_aime_computation_registry()
    calculator = calculator_registry.require_capability(AIME_CALCULATOR_TOOL_ID)
    registrations.append(ToolRegistration(HEALTHBENCH_CALCULATOR_TOOL_ID, calculator_registry._backend(AIME_CALCULATOR_TOOL_ID), ToolCapability(
        tool_id=HEALTHBENCH_CALCULATOR_TOOL_ID, dataset_scope=HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
        action_schemas=calculator.action_schemas, input_schema=calculator.input_schema,
        output_schema=calculator.output_schema, side_effect=calculator.side_effect,
        timeout_seconds=calculator.timeout_seconds, version=calculator.version,
    )))
    source_registrations = []
    if external_medical_sources_enabled:
        source_registrations.extend((
            (HEALTHBENCH_LITERATURE_SEARCH_TOOL_ID, literature_client,
             "Search Europe PMC literature abstracts using the original named entity and requested relation. "
             "Preprints are excluded. An abstract is not a full paper; read available OA full text by full_text_source_id. "
             "Do not send the full conversation or search benchmark questions, rubrics, or reference answers."),
            (HEALTHBENCH_TRIAL_SEARCH_TOOL_ID, trial_client,
             "Search ClinicalTrials.gov for registered study names, interventions, populations and outcomes. "
             "Registration describes a protocol, not proof of efficacy; distinguish posted results from planned outcomes. "
             "Use short clinical terms, not the full conversation or benchmark/reference-answer content."),
        ))
    if clinical_reference_sources_enabled:
        source_registrations.extend((
            (HEALTHBENCH_BOOKSHELF_SEARCH_TOOL_ID, bookshelf_client,
             "Search NCBI Bookshelf medical books, guidelines and reviews. Results are catalogue metadata; "
             "read the returned source_id for available public-domain/OA text before citing its clinical content."),
            (HEALTHBENCH_PDQ_SEARCH_TOOL_ID, pdq_client,
             "Search the NCI PDQ collection in Bookshelf. Read the source and check the intended audience "
             "and update date. PDQ evidence summaries are not clinical practice guidelines."),
            (HEALTHBENCH_AHRQ_SEARCH_TOOL_ID, ahrq_client,
             "Search AHRQ EPC Systematic Reviews (formerly Comparative Effectiveness Reviews) "
             "in Bookshelf, not all AHRQ publications. "
             "Read matching evidence reviews for population, outcomes, harms and limitations."),
            (HEALTHBENCH_TERMINOLOGY_SEARCH_TOOL_ID, terminology_client,
             "Look up MeSH descriptor labels to clarify medical terminology. Read definitions using source_id. "
             "Label matching is not clinical evidence or an exhaustive synonym/trial-acronym lookup."),
        ))
    for tool_id, client, description in source_registrations:
        arguments = {
            "type": "object", "additionalProperties": False,
            "required": ["query"], "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 160,
                          "description": description},
            },
        }
        output_schema = _clinical_output_schema()
        output_schema["properties"]["operation"] = {"const": "search"}
        if tool_id in {HEALTHBENCH_BOOKSHELF_SEARCH_TOOL_ID, HEALTHBENCH_PDQ_SEARCH_TOOL_ID, HEALTHBENCH_AHRQ_SEARCH_TOOL_ID}:
            output_schema["properties"]["search_collection"] = {"enum": ["bookshelf", "pdq", "ahrq"]}
        registrations.append(ToolRegistration(tool_id, HealthBenchExternalSearchToolBackend(client), ToolCapability(
            tool_id=tool_id, dataset_scope=HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
            action_schemas={"search": arguments}, input_schema=arguments,
            output_schema=output_schema, side_effect="none",
            timeout_seconds=timeout_seconds, version="healthbench-external-medical-sources-v1",
        )))
    return ToolRegistry(tuple(registrations))


def open_healthbench_clinical_tool_registry(
    *,
    corpus_root: str | Path,
    source_identity: str,
    expected_source_revision: str,
    expected_rows: int,
    pubmed_client: PubMedEUtilitiesClient | None = None,
    timeout_seconds: float = 30.0,
    max_query_content_tokens: int = AUTHORITATIVE_QUERY_MAX_CONTENT_TOKENS,
    drug_client: DailyMedClient | None = None,
    external_medical_sources_enabled: bool = False,
    literature_client: object | None = None,
    trial_client: object | None = None,
    clinical_reference_sources_enabled: bool = False,
    bookshelf_client: object | None = None,
    pdq_client: object | None = None,
    ahrq_client: object | None = None,
    terminology_client: object | None = None,
) -> OpenHealthBenchAuthoritativeToolRegistry:
    """Reuse the existing owned corpus lifetime without another resource type."""
    corpus = FrozenMedRAGBM25Corpus.open(corpus_root, source_identity=source_identity, expected_source_revision=expected_source_revision, expected_rows=expected_rows)
    try:
        registry = build_healthbench_clinical_tool_registry(corpus, pubmed_client=pubmed_client, timeout_seconds=timeout_seconds, max_query_content_tokens=max_query_content_tokens, drug_client=drug_client,
            external_medical_sources_enabled=external_medical_sources_enabled,
            literature_client=literature_client, trial_client=trial_client,
            clinical_reference_sources_enabled=clinical_reference_sources_enabled,
            bookshelf_client=bookshelf_client, pdq_client=pdq_client,
            ahrq_client=ahrq_client, terminology_client=terminology_client)
    except BaseException:
        corpus.close()
        raise
    return OpenHealthBenchAuthoritativeToolRegistry(registry, MappingProxyType(corpus.identity), corpus)
