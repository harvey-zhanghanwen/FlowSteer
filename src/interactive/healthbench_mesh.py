"""Optional NLM MeSH terminology transport, not clinical-effectiveness evidence.

NECESSARY_PROJECT_ADAPTATION: MeSH has no upstream SkillFlow/FlowSteer
transport. This follows EuropePMCClient/DailyMedClient's injected urllib
search/read split, directly reuses the clinical source receipt, and leaves
pagination to the existing source.read ``_source_page``. No retrieval runtime,
ranking, Agent role, benchmark data, or medical inference is introduced.

Official protocols: https://id.nlm.nih.gov/mesh/swagger/ui and
https://hhs.github.io/meshrdf/sparql-and-uri-requests . NLM links the RDF API
from https://www.nlm.nih.gov/databases/download/mesh.html . Search uses only
``lookup/descriptor`` with ``match=contains`` against descriptor labels, not
entry-term lookup, concept search, natural-language retrieval, or guaranteed
trial-acronym disambiguation. Read fetches an exact descriptor's JSON-LD and,
if linked, its preferred concept's JSON-LD: at most two HTTP requests.

The preferred concept's scopeNote is vocabulary documentation, not a study
result. Linked term IDs are retained but are not fetched as synonym labels;
other concepts' definitions, term lexical variants, and the full ontology are
not expanded. Missing fields are left missing, never medically completed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .healthbench_evidence_adapter import _required_query


MESH_BASE_URL = "https://id.nlm.nih.gov/mesh"
MESH_SOURCE = "NLM MeSH"
_DESCRIPTOR_SOURCE_ID = re.compile(r"mesh:(D[0-9]+)\Z")
_RESOURCE_ID = re.compile(r"https?://id\.nlm\.nih\.gov/mesh/([DMT][0-9]+)\Z")
_VOCABULARY_NOTICE = (
    "NLM MeSH controlled vocabulary; not clinical-effectiveness evidence or "
    "a treatment recommendation. Label matching does not guarantee trial-acronym "
    "disambiguation."
)
_DESCRIPTOR_FIELDS = (
    "identifier", "label", "preferredConcept", "concept", "preferredTerm",
    "dateIntroduced", "lastUpdated", "dateRevised",
)
_CONCEPT_FIELDS = ("identifier", "label", "scopeNote", "preferredTerm", "term")


def _resource_id(value: object, prefix: str) -> str:
    match = _RESOURCE_ID.fullmatch(value) if isinstance(value, str) else None
    if match is None or not match.group(1).startswith(prefix):
        raise RuntimeError(f"MeSH response lacks a valid {prefix} resource ID")
    return match.group(1)


def _literal_text(value: object) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("@value")
    return value if isinstance(value, str) and value.strip() else None


def _evidence(identifier: str, label: str, excerpt: str, content_type: str,
              date: str | None = None) -> dict[str, object]:
    return {
        "source_type": "controlled_vocabulary",
        "source": MESH_SOURCE,
        "document_id": identifier,
        "source_id": f"mesh:{identifier}",
        "title": label,
        "date": date,
        "url": f"{MESH_BASE_URL}/{identifier}",
        "excerpt": excerpt,
        "rank": 1,
        "content_type": content_type,
        "version": None,
    }


@dataclass(frozen=True, slots=True)
class MeSHClient:
    """Bounded descriptor-label search and exact terminology reads, no retries."""

    timeout_seconds: float = 8.0
    retmax: int = 3
    opener: Callable[..., object] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("MeSH timeout must be finite and positive")
        if type(self.retmax) is not int or not 1 <= self.retmax <= 3:
            raise ValueError("MeSH retmax must be from 1 through 3")
        if not callable(self.opener):
            raise TypeError("MeSH opener must be callable")

    def _get(self, endpoint: str, parameters: Mapping[str, object] | None = None) -> object:
        url = f"{MESH_BASE_URL}/{endpoint}"
        if parameters:
            url += "?" + urlencode(parameters)
        request = Request(url, headers={"Accept": "application/json, application/ld+json;q=0.9"}, method="GET")
        response = self.opener(request, timeout=self.timeout_seconds)
        try:
            payload = response.read()  # type: ignore[attr-defined]
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if not isinstance(payload, bytes):
            raise RuntimeError("MeSH response is not bytes")
        return json.loads(payload)

    def search(self, query: str) -> dict[str, object]:
        # DIRECT_REUSE: same successful/empty receipt contract as source.read.
        from .healthbench_clinical_tools import _source_receipt

        query = _required_query(query)
        rows = self._get("lookup/descriptor", {"label": query, "match": "contains", "limit": self.retmax})
        if not isinstance(rows, list):
            raise RuntimeError("MeSH descriptor lookup response is invalid")
        evidence: list[dict[str, object]] = []
        seen: set[str] = set()
        for row in rows[:self.retmax]:
            if not isinstance(row, Mapping):
                raise RuntimeError("MeSH descriptor lookup entry is invalid")
            identifier = _resource_id(row.get("resource"), "D")
            label = row.get("label")
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError("MeSH descriptor lookup entry lacks a label")
            if identifier in seen:
                continue
            seen.add(identifier)
            item = _evidence(identifier, label, (
                _VOCABULARY_NOTICE + "\nDescriptor-label contains match only; "
                "no definition or synonym expansion has been fetched.\n"
                + json.dumps({"resource": row["resource"], "label": label}, ensure_ascii=False)
            ), "mesh_descriptor_label_match")
            evidence.append({**item, "rank": len(evidence) + 1})
        return {
            "operation": "search",
            "query": query,
            "evidence": evidence,
            "source_receipts": [_source_receipt("controlled_vocabulary", MESH_SOURCE, len(evidence))],
        }

    def _record(self, identifier: str) -> Mapping[str, object]:
        row = self._get(f"{identifier}.json")
        if not isinstance(row, Mapping) or _resource_id(row.get("@id"), identifier[0]) != identifier:
            raise RuntimeError("MeSH JSON-LD does not identify the requested record")
        if "identifier" in row and row["identifier"] != identifier:
            raise RuntimeError("MeSH JSON-LD identifier does not match the requested record")
        return row

    def read(self, source_id: str) -> dict[str, object]:
        """Return one complete terminology projection; caller handles paging."""
        match = _DESCRIPTOR_SOURCE_ID.fullmatch(source_id) if isinstance(source_id, str) else None
        if match is None:
            raise ValueError("MeSH source requires mesh:D<digits> for an exact descriptor")
        identifier = match.group(1)
        descriptor = self._record(identifier)
        label = _literal_text(descriptor.get("label"))
        if label is None:
            raise RuntimeError("MeSH descriptor has no preferred label")
        projected: dict[str, object] = {
            "descriptor": {"@id": descriptor["@id"], **{
                name: descriptor[name] for name in _DESCRIPTOR_FIELDS if name in descriptor
            }},
        }
        preferred = descriptor.get("preferredConcept")
        if preferred is not None:
            concept = self._record(_resource_id(preferred, "M"))
            projected["preferred_concept"] = {"@id": concept["@id"], **{
                name: concept[name] for name in _CONCEPT_FIELDS if name in concept
            }}
        excerpt = (
            _VOCABULARY_NOTICE + "\nOfficial descriptor label and returned preferred-concept "
            "fields follow. scopeNote, when present, is a terminology definition, not a "
            "clinical outcome. Term IDs are not resolved synonym labels; other concept "
            "definitions and term lexical variants are not fetched. Missing fields are "
            "unavailable in this projection, not negative medical findings.\n"
            + json.dumps(projected, ensure_ascii=False, indent=2)
        )
        return _evidence(identifier, label, excerpt, "mesh_descriptor_terminology", (
            _literal_text(descriptor.get("lastUpdated"))
            or _literal_text(descriptor.get("dateRevised"))
        ))

