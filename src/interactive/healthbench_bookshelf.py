"""Bounded Bookshelf discovery and permitted, on-demand XML text reading.

NECESSARY_PROJECT_ADAPTATION: SkillFlow RetrievalIndex has a search/read split
but no Bookshelf transport. Reuse the existing injected urllib/evidence shape;
do not add a retrieval runtime, ranking model, corpus download, or Agent role.

Official protocols: https://www.ncbi.nlm.nih.gov/books/NBK45615/ and
https://www.ncbi.nlm.nih.gov/books/about/oai/. ESearch UIDs are NOT NBK IDs:
ESummary's accessionid resolves an indexed section to its readable document.
Search returns explicitly labelled metadata, never purported clinical text.
Read uses one Books-OAI GetRecord with nbk_ftext, available only for public
domain/OA content. No HTML/FTP/PDF fallback, retries, or automatic harvesting.

Verified scopes: PDQ NBK82221 uses pdqcis[book]; AHRQ NBK42934's official
search form uses collection_hscompeffcollect[filter]. In July 2026 that AHRQ
collection was renamed from Comparative Effectiveness Reviews to AHRQ EPC
Systematic Reviews. It is not every AHRQ publication. XML may use BITS or the
older NLM Book Tag Set. Images and linked supplements are not downloaded;
their available captions/alternative text remain in the text projection.
Caller-owned _source_page provides text pagination. Missing versions remain
None; an OAI datestamp is repository metadata, not a publication version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
from html import unescape
from io import BytesIO
import json
import math
import re
from threading import Lock
from typing import Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree
import zlib

from .healthbench_evidence_adapter import (
    NCBI_EUTILS_BASE_URL,
    _node_text,
    _required_query,
    _wait_for_pubmed_request_slot,
)


BOOKSHELF_SOURCE = "NCBI Bookshelf"
BOOKSHELF_SOURCE_TYPE = "biomedical_reference"
BOOKSHELF_OAI_BASE_URL = "https://api.ncbi.nlm.nih.gov/lit/oai/books/"
BOOKSHELF_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
BOOKSHELF_COLLECTION_FILTERS = {
    "bookshelf": None,
    "pdq": "pdqcis[book]",
    "ahrq": "collection_hscompeffcollect[filter]",
}
_NBK_ID = re.compile(r"NBK[0-9]+\Z")
_OAI_LOCK = Lock()
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_TEXT_BLOCKS = frozenset({
    "title", "subtitle", "p", "sec", "list-item", "label", "caption",
    "alt-text", "tr", "td", "th", "ref", "fn", "book-part", "def-item",
})


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _plain_title(value: object) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", str(value or ""))).split())


def _readable_text(node: ElementTree.Element | None) -> str:
    if node is None:
        return ""
    parts: list[str] = []

    def visit(item: ElementTree.Element) -> None:
        block = item.tag in _TEXT_BLOCKS
        if block:
            parts.append("\n")
        if item.text:
            parts.append(item.text)
        for child in item:
            visit(child)
            if child.tail:
                parts.append(child.tail)
        if block:
            parts.append("\n")

    visit(node)
    return "\n".join(line for raw in "".join(parts).splitlines() if (line := " ".join(raw.split())))


def _xml_date(meta: ElementTree.Element | None) -> str | None:
    if meta is None:
        return None
    dates = meta.findall("pub-date")
    preferred = next((node for node in dates if node.get("date-type") == "pub"), None)
    if preferred is None:
        preferred = next((node for node in dates if node.get("pub-type") == "epub"), None)
    if preferred is None:
        preferred = dates[0] if dates else None
    if preferred is None:
        return None
    return "-".join(_node_text(preferred.find(key)) for key in ("year", "month", "day") if _node_text(preferred.find(key))) or None


@dataclass(frozen=True, slots=True)
class BookshelfClient:
    """Two-request metadata search; one-request permitted XML text read.

    collection scopes discovery only. All source IDs have the same reader and
    stable NCBI Bookshelf attribution, regardless of which search found them.
    A 4 MiB wire/decoded response ceiling rejects oversized documents without
    silently truncating text. Read is not a promise that every hit permits XML.
    """

    timeout_seconds: float = 8.0
    retmax: int = 3
    collection: str = "bookshelf"
    opener: Callable[..., object] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Bookshelf timeout must be a positive finite number")
        if type(self.retmax) is not int or not 1 <= self.retmax <= 3:
            raise ValueError("Bookshelf retmax must be from 1 through 3")
        if not isinstance(self.collection, str) or self.collection not in BOOKSHELF_COLLECTION_FILTERS:
            raise ValueError("Bookshelf collection must be bookshelf, pdq, or ahrq")
        if not callable(self.opener):
            raise TypeError("Bookshelf opener must be callable")

    def _get(self, url: str, parameters: Mapping[str, object], *, oai: bool = False) -> bytes:
        request = Request(url + "?" + urlencode(parameters), headers={
            "Accept": "application/json, application/xml;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }, method="GET")
        if not oai:
            # DIRECT_REUSE: share the existing NCBI request-start rate limit.
            _wait_for_pubmed_request_slot(0.4)
        response = self.opener(request, timeout=self.timeout_seconds)
        try:
            payload = response.read(BOOKSHELF_MAX_RESPONSE_BYTES + 1)  # type: ignore[attr-defined]
            headers = getattr(response, "headers", {})
            encoding = str(headers.get("Content-Encoding", "")).lower()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if not isinstance(payload, bytes):
            raise RuntimeError("Bookshelf response is not bytes")
        if len(payload) > BOOKSHELF_MAX_RESPONSE_BYTES:
            raise RuntimeError("Bookshelf response exceeds the bounded document size")
        if encoding == "gzip":
            with gzip.GzipFile(fileobj=BytesIO(payload)) as compressed:
                payload = compressed.read(BOOKSHELF_MAX_RESPONSE_BYTES + 1)
        elif encoding == "deflate":
            decompressor = zlib.decompressobj()
            payload = decompressor.decompress(payload, BOOKSHELF_MAX_RESPONSE_BYTES + 1)
            if not decompressor.eof and len(payload) <= BOOKSHELF_MAX_RESPONSE_BYTES:
                raise RuntimeError("Bookshelf compressed response is incomplete")
        elif encoding not in {"", "identity"}:
            raise RuntimeError("Bookshelf response uses an unsupported content encoding")
        if len(payload) > BOOKSHELF_MAX_RESPONSE_BYTES:
            raise RuntimeError("Bookshelf decoded response exceeds the bounded document size")
        return payload

    def _eutils(self, endpoint: str, parameters: Mapping[str, object]) -> object:
        return json.loads(self._get(f"{NCBI_EUTILS_BASE_URL}/{endpoint}", {
            **parameters, "db": "books", "retmode": "json", "tool": "flowsteer-bookshelf",
        }))

    @staticmethod
    def _metadata_evidence(row: Mapping[str, object], rank: int) -> dict[str, object]:
        nbk = row.get("accessionid")
        if not isinstance(nbk, str) or not _NBK_ID.fullmatch(nbk):
            raise RuntimeError("Bookshelf summary lacks an exact NBK accessionid")
        matched_title = _plain_title(row.get("title"))
        title = matched_title
        book_title = None
        info = row.get("bookinfo")
        if isinstance(info, str) and info.strip():
            root = ElementTree.fromstring(info)
            for parent in root.findall("Path/Parent"):
                if parent.get("role") == "source":
                    book_title = _node_text(parent.find("Title")) or None
                # Indexed subsections share their parent chapter's NBK. Name
                # the actual read target and retain the matched section title.
                if parent.get("role") == "document":
                    title = _node_text(parent.find("Title")) or title
        return {
            "source_type": BOOKSHELF_SOURCE_TYPE,
            "source": BOOKSHELF_SOURCE,
            "source_id": f"bookshelf:{nbk}",
            "document_id": nbk,
            "title": title,
            "matched_title": matched_title,
            "book_title": book_title,
            "date": _optional_text(row.get("pubdate")),
            "version": None,
            "url": f"https://www.ncbi.nlm.nih.gov/books/{nbk}/",
            "excerpt": "",
            "rank": rank,
            "content_type": "bookshelf_metadata_not_full_text",
            "full_text_availability": "unverified_until_source_read",
        }

    def search(self, query: str) -> dict[str, object]:
        query = _required_query(query)
        scope = BOOKSHELF_COLLECTION_FILTERS[self.collection]
        term = f"({query}) AND {scope}" if scope else query
        payload = self._eutils("esearch.fcgi", {"term": term, "retmax": self.retmax, "sort": "relevance"})
        result = payload.get("esearchresult") if isinstance(payload, Mapping) else None
        identifiers = result.get("idlist") if isinstance(result, Mapping) else None
        if not isinstance(identifiers, list) or any(not isinstance(uid, str) or not uid.isdigit() for uid in identifiers):
            raise RuntimeError("Bookshelf search identifier response is invalid")
        if result.get("errorlist") or result.get("ERROR"):
            raise RuntimeError("Bookshelf rejected the search expression")
        identifiers = list(dict.fromkeys(identifiers))[:self.retmax]
        evidence: list[dict[str, object]] = []
        if identifiers:
            payload = self._eutils("esummary.fcgi", {"id": ",".join(identifiers)})
            summaries = payload.get("result") if isinstance(payload, Mapping) else None
            if not isinstance(summaries, Mapping):
                raise RuntimeError("Bookshelf summary response is invalid")
            seen: set[str] = set()
            for uid in identifiers:
                row = summaries.get(uid)
                if not isinstance(row, Mapping) or str(row.get("uid")) != uid or row.get("error"):
                    raise RuntimeError("Bookshelf summary does not identify the requested UID")
                item = self._metadata_evidence(row, len(evidence) + 1)
                if item["source_id"] not in seen:
                    seen.add(str(item["source_id"]))
                    evidence.append(item)
        return {
            "operation": "search", "query": query, "search_collection": self.collection,
            "evidence": evidence,
            "source_receipts": [{
                "source_type": BOOKSHELF_SOURCE_TYPE, "source": BOOKSHELF_SOURCE,
                "status": "success", "result_count": len(evidence), "error_type": None,
            }],
        }

    def read(self, source_id: str) -> dict[str, object]:
        if not isinstance(source_id, str) or not source_id.startswith("bookshelf:"):
            raise ValueError("Bookshelf source requires bookshelf:NBK<digits>")
        nbk = source_id.removeprefix("bookshelf:")
        if not _NBK_ID.fullmatch(nbk):
            raise ValueError("Bookshelf source requires an exact unversioned NBK accession")
        oai_id = "oai:books.ncbi.nlm.nih.gov:" + nbk.removeprefix("NBK")
        # Official OAI guidance forbids concurrent harvesting requests.
        with _OAI_LOCK:
            root = ElementTree.fromstring(self._get(BOOKSHELF_OAI_BASE_URL, {
                "verb": "GetRecord", "identifier": oai_id, "metadataPrefix": "nbk_ftext",
            }, oai=True))
        for node in root.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        if root.tag != "OAI-PMH":
            raise RuntimeError("Bookshelf did not return an OAI record")
        error = root.find("error")
        if error is not None:
            raise LookupError(f"Bookshelf permitted XML unavailable ({error.get('code', 'unknown')}): {_node_text(error)}")
        request = root.find("request")
        if request is not None and request.get("metadataPrefix") not in {None, "nbk_ftext"}:
            raise RuntimeError("Bookshelf returned metadata instead of permitted full text")
        records = root.findall("GetRecord/record")
        if len(records) != 1:
            raise RuntimeError("Bookshelf must return exactly one requested record")
        record = records[0]
        header = record.find("header")
        if header is None or header.get("status") == "deleted" or _node_text(header.find("identifier")) != oai_id:
            raise RuntimeError("Bookshelf record does not identify the requested accession")
        metadata = record.find("metadata")
        if metadata is None or len(metadata) != 1:
            raise RuntimeError("Bookshelf record lacks a single XML document")
        document = metadata[0]
        if document.tag not in {"book", "book-part", "book-part-wrapper"}:
            raise RuntimeError("Bookshelf XML is not a supported book or book part")
        book_meta = document.find("book-meta")
        part = document.find("book-part") if document.tag == "book-part-wrapper" else document
        if part is None:
            raise RuntimeError("Bookshelf wrapper contains no book part")
        part_meta = part.find("book-part-meta")
        source_meta = part_meta if part_meta is not None else book_meta
        if source_meta is None:
            raise RuntimeError("Bookshelf document has no source metadata")
        declared_ids = [_node_text(node) for node in source_meta.findall("book-part-id") if node.get("book-part-id-type") == "art-access-id"]
        declared_ids.extend(node.get(_XLINK_HREF, "").removeprefix("art-access-id://") for node in source_meta.findall("uri") if node.get(_XLINK_HREF, "").startswith("art-access-id://"))
        if declared_ids and nbk not in declared_ids:
            raise RuntimeError("Bookshelf XML identifies a different NBK accession")
        body = part.find("body")
        if body is None:
            body = part.find("book-body")
        if body is None or not _readable_text(body):
            raise RuntimeError("Bookshelf document has no readable body; metadata or abstract is not full text")
        pieces = [_readable_text(node) for node in source_meta.findall("abstract")]
        pieces.append(_readable_text(body))
        for tag in ("back", "book-back", "floats-group"):
            pieces.append(_readable_text(part.find(tag)))
        title_group = source_meta.find("title-group")
        if title_group is None:
            title_group = source_meta.find("book-title-group")
        title = ": ".join(_node_text(node) for node in title_group if _node_text(node)) if title_group is not None else ""
        sets = [_node_text(node) for node in header.findall("setSpec")]
        collections = [node.get("source-id") for node in document.findall(".//related-object") if node.get("link-type") == "collection-link"]
        collection = "pdq" if "pdqcis" in sets else "ahrq" if "hscompeffcollect" in collections else "bookshelf"
        return {
            "source_type": BOOKSHELF_SOURCE_TYPE, "source": BOOKSHELF_SOURCE,
            "source_id": source_id, "document_id": nbk,
            "title": title, "book_title": _node_text(book_meta.find("book-title-group/book-title")) if book_meta is not None else None,
            "date": _xml_date(source_meta) or _xml_date(book_meta),
            "version": _node_text(book_meta.find("edition")) or None if book_meta is not None else None,
            "url": f"https://www.ncbi.nlm.nih.gov/books/{nbk}/",
            "excerpt": "\n\n".join(piece for piece in pieces if piece), "rank": 1,
            "content_type": "bookshelf_open_access_xml_text",
            "text_scope": "book" if document.tag == "book" else "book_part",
            "collection": collection,
            "repository_datestamp": _node_text(header.find("datestamp")) or None,
            "full_text_availability": "permitted_xml_returned",
        }
