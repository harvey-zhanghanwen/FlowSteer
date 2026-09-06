"""Optional ClinicalTrials.gov evidence through its official v2 API.

NECESSARY_PROJECT_ADAPTATION: SkillFlow's public-document search/read and
FlowSteer's tool receipts have no ClinicalTrials.gov transport.  This client
follows the existing DailyMedClient/PubMedEUtilitiesClient transport shape and
reuses the clinical tools' evidence pagination/receipt functions.  It does not
change AgentGraph, infer treatment recommendations, or consume evaluator data.

Official schema and endpoint references:
https://clinicaltrials.gov/data-api/about-api/study-data-structure
https://clinicaltrials.gov/data-api/about-api/api-migration
https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CLINICAL_TRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2"
_NCT_ID = re.compile(r"NCT[0-9]{8}\Z")
_PROTOCOL_MODULES = (
    "identificationModule",
    "statusModule",
    "descriptionModule",
    "conditionsModule",
    "designModule",
    "armsInterventionsModule",
    "outcomesModule",
    "eligibilityModule",
    "referencesModule",
)


def _study_evidence(study: Mapping[str, object]) -> dict[str, object]:
    """Project API fields without converting planned outcomes into results."""
    protocol = study.get("protocolSection")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("ClinicalTrials.gov record lacks protocolSection")
    identification = protocol.get("identificationModule")
    if not isinstance(identification, Mapping):
        raise RuntimeError("ClinicalTrials.gov record lacks identificationModule")
    nct_id = identification.get("nctId")
    if not isinstance(nct_id, str) or not _NCT_ID.fullmatch(nct_id):
        raise RuntimeError("ClinicalTrials.gov record lacks a valid NCT ID")
    title = identification.get("officialTitle") or identification.get("briefTitle")
    if not isinstance(title, str) or not title.strip():
        raise RuntimeError("ClinicalTrials.gov record lacks a study title")

    # hasResults is the official availability flag.  Its absence must not be
    # silently interpreted as a negative finding.  The actual results payload
    # remains distinct from protocolSection.outcomesModule (planned measures).
    has_results = study.get("hasResults")
    if has_results is not None and type(has_results) is not bool:
        raise RuntimeError("ClinicalTrials.gov hasResults must be boolean")
    results = study.get("resultsSection")
    if results is not None and not isinstance(results, Mapping):
        raise RuntimeError("ClinicalTrials.gov resultsSection must be an object")
    if results:
        content_type = "clinical_trial_registry_protocol_and_posted_results"
    elif has_results is True:
        content_type = "clinical_trial_registry_protocol_results_not_returned"
    elif has_results is False:
        content_type = "clinical_trial_registry_protocol_no_posted_results"
    else:
        content_type = "clinical_trial_registry_protocol_results_status_unknown"

    projected = {
        "hasResults": has_results,
        "protocolSection": {
            name: protocol[name] for name in _PROTOCOL_MODULES if name in protocol
        },
    }
    if results is not None:
        # Retain group labels, outcome analyses, adverse events and limitations
        # exactly as supplied; no model-generated compression or interpretation.
        projected["resultsSection"] = results
    status = protocol.get("statusModule")
    last_update = status.get("lastUpdatePostDateStruct") if isinstance(status, Mapping) else None
    date = last_update.get("date") if isinstance(last_update, Mapping) else None
    return {
        "source_type": "clinical_trial_registry",
        "source": "ClinicalTrials.gov",
        "document_id": nct_id,
        "source_id": f"clinicaltrials:{nct_id}",
        "title": title,
        "date": date if isinstance(date, str) else None,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
        "excerpt": (
            "ClinicalTrials.gov registry record, not a journal article or clinical recommendation. "
            "Registration does not establish efficacy. Protocol outcomes describe registered "
            "measures, not observed effects; only a returned resultsSection contains posted "
            "summary results. hasResults reports result availability, not a favorable outcome.\n"
            + json.dumps(projected, ensure_ascii=False, indent=2)
        ),
        "content_type": content_type,
        "version": date if isinstance(date, str) else None,
    }


@dataclass(frozen=True, slots=True)
class ClinicalTrialsClient:
    """Bounded search/read client; injected opener supports entirely local tests."""

    timeout_seconds: float = 8.0
    retmax: int = 3
    opener: Callable[..., object] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("ClinicalTrials.gov timeout must be finite and positive")
        if type(self.retmax) is not int or not 1 <= self.retmax <= 3:
            raise ValueError("ClinicalTrials.gov retmax must be from 1 through 3")
        if not callable(self.opener):
            raise TypeError("ClinicalTrials.gov opener must be callable")

    def _get(self, endpoint: str, parameters: Mapping[str, object] | None = None) -> Mapping[str, object]:
        # NECESSARY_ADAPTATION of DailyMedClient._get: same single-request
        # urllib lifecycle, with the ClinicalTrials.gov JSON endpoint/schema.
        url = f"{CLINICAL_TRIALS_BASE_URL}/{endpoint}"
        if parameters:
            url += "?" + urlencode(parameters)
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        response = self.opener(request, timeout=self.timeout_seconds)
        try:
            payload = response.read()  # type: ignore[attr-defined]
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if not isinstance(payload, bytes):
            raise RuntimeError("ClinicalTrials.gov response is not bytes")
        result = json.loads(payload)
        if not isinstance(result, Mapping):
            raise RuntimeError("ClinicalTrials.gov response must be a JSON object")
        return result

    def search(self, query: str) -> dict[str, object]:
        # Direct reuse at call time avoids a module cycle when clinical_tools
        # registers this adapter alongside DailyMedClient and source.read.
        from .healthbench_clinical_tools import _source_page, _source_receipt, _text_argument

        query = _text_argument(query, "query")
        payload = self._get("studies", {"query.term": query, "pageSize": self.retmax, "format": "json"})
        rows = payload.get("studies")
        if not isinstance(rows, list):
            raise RuntimeError("ClinicalTrials.gov search response lacks studies")
        evidence: list[dict[str, object]] = []
        receipts: list[dict[str, object]] = []
        seen: set[str] = set()
        for row in rows[:self.retmax]:
            try:
                if not isinstance(row, Mapping):
                    raise RuntimeError("ClinicalTrials.gov study entry must be an object")
                item = _study_evidence(row)
            except (RuntimeError, TypeError, ValueError) as exc:
                receipts.append(_source_receipt("clinical_trial_registry", "ClinicalTrials.gov", 0, exc))
                continue
            source_id = str(item["source_id"])
            if source_id in seen:
                continue
            seen.add(source_id)
            evidence.append({**_source_page(item), "rank": len(evidence) + 1})
            receipts.append(_source_receipt("clinical_trial_registry", source_id, 1))
        if not rows:
            receipts.append(_source_receipt("clinical_trial_registry", "ClinicalTrials.gov", 0))
        return {"operation": "search", "query": query, "evidence": evidence, "source_receipts": receipts}

    def read(self, source_id: str) -> dict[str, object]:
        """Return the complete projected record; existing source.read pages it."""
        if not isinstance(source_id, str):
            raise ValueError("ClinicalTrials.gov source requires an NCT ID")
        nct_id = source_id.removeprefix("clinicaltrials:")
        if not _NCT_ID.fullmatch(nct_id):
            raise ValueError("ClinicalTrials.gov source requires an NCT ID")
        item = _study_evidence(self._get(f"studies/{nct_id}", {"format": "json"}))
        if item["document_id"] != nct_id:
            raise RuntimeError("ClinicalTrials.gov returned a different NCT ID")
        return item
