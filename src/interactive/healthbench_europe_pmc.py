"""Optional Europe PMC literature transport, not a benchmark answer store.

NECESSARY_PROJECT_ADAPTATION: Europe PMC's ``search`` (``resultType=core``)
and ``/{PMCID}/fullTextXML`` have no SkillFlow/FlowSteer adapter. This follows
the existing ``PubMedEUtilitiesClient`` / ``DailyMedClient`` injected urllib
transport and evidence projection, preserving SkillFlow's search/read split.
It does not introduce a retrieval runtime, ranking model, or Agent roles.

Official protocol: https://europepmc.org/RestfulWebService and
https://europepmc.org/help. Full-text XML is available only for the OA subset;
an abstract, a PMCID, or a free web page is not proof of XML availability.
Source pagination remains the caller's existing ``_source_page`` operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
import json
import re
from typing import Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .healthbench_evidence_adapter import _node_text, _required_query


EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUROPE_PMC_SOURCE = "Europe PMC"
EUROPE_PMC_SEARCH_EXCERPT_CHARACTERS = 1200
_PMC_ID = re.compile(r"PMC[0-9]+\Z")
_RECORD_ID = re.compile(r"europepmc:(MED|PMC|PPR):([A-Za-z0-9]+)\Z")


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _plain_abstract(value: object) -> str:
    if not isinstance(value, str):
        return ""
    # Europe PMC abstractText contains inline XML/HTML, not always a complete
    # XML document. Retain its words and paragraph boundaries as plain text.
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _publication_types(row: Mapping[str, object]) -> list[str]:
    value = row.get("pubTypeList")
    entries = value.get("pubType") if isinstance(value, Mapping) else None
    return [str(item) for item in entries if isinstance(item, str)] if isinstance(entries, list) else []


def _is_preprint(source: str, publication_types: list[str]) -> bool:
    return source == "PPR" or any("preprint" in item.casefold() for item in publication_types)


@dataclass(frozen=True, slots=True)
class EuropePMCClient:
    """Bounded official search, exact-record abstracts, and OA JATS text.

    Search excludes preprints by default and preserves the server's relevance
    order. Optional preprints are explicitly labelled and ordered after other
    returned publications; being PubMed-indexed is not itself peer review.
    A transport failure raises to the existing Tool runtime, without retries.
    """

    timeout_seconds: float = 8.0
    retmax: int = 3
    include_preprints: bool = False
    opener: Callable[..., object] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("Europe PMC timeout must be positive")
        if type(self.retmax) is not int or not 1 <= self.retmax <= 3:
            raise ValueError("Europe PMC retmax must be from 1 through 3")
        if type(self.include_preprints) is not bool:
            raise TypeError("include_preprints must be boolean")
        if not callable(self.opener):
            raise TypeError("Europe PMC opener must be callable")

    def _get(self, endpoint: str, parameters: Mapping[str, object] | None = None) -> bytes:
        url = f"{EUROPE_PMC_BASE_URL}/{endpoint}"
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
            raise RuntimeError("Europe PMC response is not bytes")
        return payload

    def _records(self, query: str, *, limit: int) -> list[Mapping[str, object]]:
        payload = json.loads(self._get("search", {
            "query": query, "format": "json", "resultType": "core", "pageSize": limit,
        }))
        result_list = payload.get("resultList") if isinstance(payload, Mapping) else None
        rows = result_list.get("result") if isinstance(result_list, Mapping) else None
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise RuntimeError("Europe PMC core search response is invalid")
        return rows[:limit]

    @staticmethod
    def _abstract_evidence(row: Mapping[str, object], *, truncate: bool) -> dict[str, object]:
        source = str(row.get("source", ""))
        identifier = str(row.get("id", ""))
        source_id = f"europepmc:{source}:{identifier}"
        if not _RECORD_ID.fullmatch(source_id):
            raise RuntimeError("Europe PMC record lacks a supported source and ID")
        if source == "MED" and not identifier.isdigit():
            raise RuntimeError("Europe PMC MED record lacks a PMID")
        pmcid = _optional_text(row.get("pmcid"))
        if pmcid is not None and not _PMC_ID.fullmatch(pmcid):
            raise RuntimeError("Europe PMC record has an invalid PMCID")
        publication_types = _publication_types(row)
        preprint = _is_preprint(source, publication_types)
        abstract = _plain_abstract(row.get("abstractText"))
        excerpt = abstract[:EUROPE_PMC_SEARCH_EXCERPT_CHARACTERS] if truncate else abstract
        is_open_access = row.get("isOpenAccess") == "Y"
        return {
            "source_type": "preprint" if preprint else "indexed_biomedical_literature",
            "source": EUROPE_PMC_SOURCE,
            "document_id": identifier,
            "source_id": source_id,
            "title": _plain_abstract(row.get("title")),
            "date": _optional_text(row.get("firstPublicationDate")) or _optional_text(row.get("pubYear")),
            "url": f"https://europepmc.org/article/{source}/{identifier}",
            "excerpt": excerpt,
            "rank": 1,
            "content_type": "europepmc_abstract_not_full_text" if abstract else "publication_metadata_no_abstract",
            "pmid": _optional_text(row.get("pmid")) or (identifier if source == "MED" else None),
            "pmcid": pmcid,
            "doi": _optional_text(row.get("doi")),
            "publication_types": publication_types,
            "is_preprint": preprint,
            "is_open_access": is_open_access,
            "full_text_source_id": f"pmc:{pmcid}" if pmcid and is_open_access else None,
            "offset": 0,
            "total_characters": len(abstract),
            "truncated": len(excerpt) < len(abstract),
            "next_offset": len(excerpt) if len(excerpt) < len(abstract) else None,
        }

    def search(self, query: str) -> dict[str, object]:
        query = _required_query(query)
        sources = "(SRC:MED OR SRC:PMC OR SRC:PPR)" if self.include_preprints else "(SRC:MED OR SRC:PMC)"
        scoped_query = f"({query}) AND {sources}"
        if not self.include_preprints:
            scoped_query += " NOT PUB_TYPE:Preprint"
        rows = self._records(scoped_query, limit=self.retmax)
        evidence: list[dict[str, object]] = []
        seen: set[str] = set()
        for row in rows:
            item = self._abstract_evidence(row, truncate=True)
            if item["is_preprint"] and not self.include_preprints:
                continue
            source_id = str(item["source_id"])
            if source_id in seen:
                continue
            seen.add(source_id)
            evidence.append(item)
        evidence.sort(key=lambda item: bool(item["is_preprint"]))
        evidence = [{**item, "rank": rank} for rank, item in enumerate(evidence, 1)]
        return {
            "operation": "search",
            "query": query,
            "evidence": evidence,
            "source_receipts": [{
                "source_type": "indexed_biomedical_literature",
                "source": EUROPE_PMC_SOURCE,
                "status": "success",
                "result_count": len(evidence),
                "error_type": None,
            }],
        }

    def read(self, source_id: str) -> dict[str, object]:
        """Return one complete source projection; caller handles pagination."""
        if not isinstance(source_id, str):
            raise ValueError("Europe PMC source ID must be text")
        if source_id.startswith("pmc:"):
            pmcid = source_id.removeprefix("pmc:")
            if not _PMC_ID.fullmatch(pmcid):
                raise ValueError("PMC full text requires an exact PMCID")
            return self._read_full_text(pmcid)
        match = _RECORD_ID.fullmatch(source_id)
        if match is None:
            raise ValueError("Europe PMC abstract requires europepmc:<source>:<ID>")
        source, identifier = match.groups()
        if source == "MED" and not identifier.isdigit():
            raise ValueError("Europe PMC MED source requires a PMID")
        rows = self._records(f"EXT_ID:{identifier} AND SRC:{source}", limit=1)
        if not rows or rows[0].get("source") != source or str(rows[0].get("id")) != identifier:
            raise RuntimeError("Europe PMC response does not identify the requested record")
        evidence = self._abstract_evidence(rows[0], truncate=False)
        if not evidence["excerpt"]:
            raise RuntimeError("Europe PMC record contains no abstract; metadata is not full text")
        return evidence

    def _read_full_text(self, pmcid: str) -> dict[str, object]:
        root = ElementTree.fromstring(self._get(f"{pmcid}/fullTextXML"))
        article = root if root.tag == "article" else root.find("article")
        if article is None:
            raise RuntimeError("Europe PMC fullTextXML did not return a JATS article")
        meta = article.find("front/article-meta")
        if meta is None:
            raise RuntimeError("Europe PMC JATS article lacks source metadata")
        ids = {str(item.get("pub-id-type")): _node_text(item) for item in meta.findall("article-id")}
        actual_pmcid = ids.get("pmc", ids.get("pmcid", ""))
        if actual_pmcid.isdigit():
            actual_pmcid = "PMC" + actual_pmcid
        if actual_pmcid != pmcid:
            raise RuntimeError("Europe PMC JATS article does not identify the requested PMCID")
        body = article.find("body")
        if body is None or not _node_text(body):
            raise RuntimeError("Europe PMC article has no body; not presenting its abstract as full text")
        # Project the complete readable JATS text, not figures or PDF layout.
        # Keep abstract, body sections, and back matter once each. Pagination
        # belongs to the existing source.read adapter, not this transport.
        parts = [_node_text(item) for item in meta.findall("abstract")]
        parts.extend(_node_text(item) for item in body)
        if body.text and body.text.strip():
            parts.insert(len(meta.findall("abstract")), body.text.strip())
        for section in ("back", "floats-group"):
            item = article.find(section)
            if item is not None:
                parts.append(_node_text(item))
        dates = meta.findall("pub-date")
        date = next((item for item in dates if item.get("pub-type") == "epub"), dates[0] if dates else None)
        published = "-".join(_node_text(date.find(key)) for key in ("year", "month", "day") if date is not None and _node_text(date.find(key)))
        publication_types = [article.get("article-type")] if article.get("article-type") else []
        preprint = _is_preprint("PMC", publication_types)
        return {
            "source_type": "preprint" if preprint else "indexed_biomedical_literature",
            "source": EUROPE_PMC_SOURCE,
            "document_id": pmcid,
            "source_id": f"pmc:{pmcid}",
            "title": _node_text(meta.find("title-group/article-title")),
            "date": published or None,
            "url": f"https://europepmc.org/articles/{pmcid}",
            "excerpt": "\n\n".join(part for part in parts if part),
            "rank": 1,
            "content_type": "pmc_open_access_jats_full_text",
            "pmid": ids.get("pmid") or None,
            "pmcid": pmcid,
            "doi": ids.get("doi") or None,
            "publication_types": publication_types,
            "is_preprint": preprint,
            "is_open_access": True,
            "full_text_source_id": f"pmc:{pmcid}",
        }

