"""Synthetic external-search preflight checks; no model or network calls."""

import asyncio
from dataclasses import replace
from io import BytesIO
import json
from urllib.parse import urlparse

import pytest

from src.interactive.healthbench_bookshelf import BookshelfClient
from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
from src.interactive.healthbench_clinical_tools import (
    DailyMedClient, build_healthbench_clinical_tool_registry,
)
from src.interactive.healthbench_clinical_trials import ClinicalTrialsClient
from src.interactive.healthbench_europe_pmc import EuropePMCClient
from src.interactive.healthbench_evidence_adapter import PubMedEUtilitiesClient
from src.interactive.healthbench_mesh import MeSHClient
from src.interactive.react_execution import ReactExecutionError
from src.interactive.tool_runtime import StructuredAction
from tests.unit.test_healthbench_clinical_react import SequenceGateway, complete, request
from tests.unit.test_healthbench_clinical_tools import fixture_corpus


LITERATURE = "healthbench-literature.search"
BOOKSHELF = "healthbench-bookshelf.search"
BOUNDED = (LITERATURE, BOOKSHELF, "healthbench-pdq.search",
           "healthbench-ahrq.search", "healthbench-terminology.search")
TRIALS = "healthbench-trials.search"
MEDRAG = "healthbench-medrag.search"
KNOWLEDGE = "healthbench-knowledge.search"
LONG_TERMS = " ".join(f"term{index}" for index in range(13))
TWELVE_TERMS = " ".join(f"term{index}" for index in range(12))
DONE = complete("The synthetic search returned no source records.")


def search(tool, query):
    return {"kind": "tool", "name": "search", "resource_id": tool,
            "skill_id": None, "arguments": {"query": query}}


class OfflineOpener:
    """Exercise the real client validators and parsers with empty local data."""

    def __init__(self):
        self.calls = []

    def __call__(self, req, *, timeout):
        self.calls.append(req.full_url)
        host = urlparse(req.full_url).hostname
        if host == "www.ebi.ac.uk":
            payload = {"resultList": {"result": []}}
        elif host == "eutils.ncbi.nlm.nih.gov":
            payload = {"esearchresult": {"idlist": []}}
        elif host == "id.nlm.nih.gov":
            payload = []
        elif host == "clinicaltrials.gov":
            payload = {"studies": []}
        else:
            raise AssertionError(f"Unexpected synthetic request: {req.full_url}")
        return BytesIO(json.dumps(payload).encode())


@pytest.fixture
def setup(monkeypatch):
    # Real Bookshelf transport shares a rate limiter; no delay is needed for
    # this injected local opener, and none of these fixtures can contact HTTP.
    monkeypatch.setattr(
        "src.interactive.healthbench_bookshelf._wait_for_pubmed_request_slot",
        lambda interval: None,
    )
    opener = OfflineOpener()
    clients = {
        "literature_client": EuropePMCClient(opener=opener),
        "trial_client": ClinicalTrialsClient(opener=opener),
        "bookshelf_client": BookshelfClient(opener=opener),
        "pdq_client": BookshelfClient(collection="pdq", opener=opener),
        "ahrq_client": BookshelfClient(collection="ahrq", opener=opener),
        "terminology_client": MeSHClient(opener=opener),
    }

    def build(*outputs, max_tool_calls=1, max_turns=6, authoritative_cap=12):
        registry = build_healthbench_clinical_tool_registry(
            fixture_corpus(), pubmed_client=PubMedEUtilitiesClient(
                opener=opener, minimum_interval_seconds=0,
            ), drug_client=DailyMedClient(opener=opener),
            external_medical_sources_enabled=True,
            clinical_reference_sources_enabled=True,
            max_query_content_tokens=authoritative_cap, **clients,
        )
        gateway = SequenceGateway(*outputs)
        adapter = HealthBenchClinicalReactExecutionAdapter(
            gateway=gateway, tool_registry=registry, max_turns=max_turns,
            max_tool_calls=max_tool_calls, max_query_content_tokens=authoritative_cap,
            max_completion_artifact_characters=6000,
            enforce_state_conditioned_completion_admission=True,
        )
        return adapter, gateway, registry

    return build, opener


@pytest.mark.parametrize("tool", BOUNDED)
@pytest.mark.parametrize(("query", "error"), [
    (LONG_TERMS, "query_too_broad_use_at_most_12_clinical_terms"),
    ("x" * 161, "query_too_long_use_at_most_160_characters"),
    ("and of the", "query_must_include_clinical_content_term"),
    ("   ", "query_must_include_clinical_content_term"),
])
def test_client_invalid_query_is_rejected_without_dispatch(setup, tool, query, error):
    build, opener = setup
    adapter, gateway, _ = build(search(tool, query), DONE)
    response = asyncio.run(adapter.execute(request(tool, output=True)))
    failed = response.metadata["react_trace"][0]
    assert failed["observation_status"] == "schema_invalid"
    assert failed["public_error_code"] == error
    assert failed["executed_action"] == search(tool, query)
    assert response.metadata["tool_calls"] == 0
    assert response.metadata["tool_receipts"] == []
    assert opener.calls == []
    assert error in gateway.requests[1].agent.contract


@pytest.mark.parametrize("tool", BOUNDED)
@pytest.mark.parametrize("valid", [TWELVE_TERMS, "x" * 160])
def test_invalid_attempt_leaves_budget_for_one_boundary_valid_query(setup, tool, valid):
    build, opener = setup
    adapter, _, _ = build(search(tool, LONG_TERMS), search(tool, valid), DONE)
    response = asyncio.run(adapter.execute(request(tool, output=True)))
    assert response.metadata["tool_calls"] == 1
    assert len(response.metadata["tool_receipts"]) == 1
    receipt = response.metadata["tool_receipts"][0]
    assert receipt["error_type"] is None
    assert receipt["request"]["arguments"] == {"query": valid}
    assert len(opener.calls) == 1


def test_valid_same_query_across_sources_is_not_a_duplicate(setup):
    build, opener = setup
    query = "synthetic relation"
    adapter, _, _ = build(search(LITERATURE, query), search(BOOKSHELF, query),
                          DONE, max_tool_calls=2)
    response = asyncio.run(adapter.execute(request(LITERATURE, BOOKSHELF, output=True)))
    assert [row["tool_id"] for row in response.metadata["tool_receipts"]] == [
        LITERATURE, BOOKSHELF,
    ]
    assert all(row["error_type"] is None for row in response.metadata["tool_receipts"])
    assert len(opener.calls) == 2


def test_external_cap_does_not_inherit_configurable_authoritative_cap(setup):
    build, opener = setup
    adapter, _, _ = build(search(LITERATURE, TWELVE_TERMS), DONE,
                          authoritative_cap=6)
    response = asyncio.run(adapter.execute(request(LITERATURE, output=True)))
    assert response.metadata["tool_calls"] == 1
    assert response.metadata["tool_receipts"][0]["error_type"] is None
    assert len(opener.calls) == 1


def test_trials_has_no_new_content_term_cap(setup):
    build, opener = setup
    adapter, _, _ = build(search(TRIALS, LONG_TERMS), DONE)
    response = asyncio.run(adapter.execute(request(TRIALS, output=True)))
    assert response.metadata["tool_calls"] == 1
    assert response.metadata["tool_receipts"][0]["error_type"] is None
    assert len(opener.calls) == 1


def test_medrag_retains_wider_query_domain(setup):
    build, opener = setup
    query = LONG_TERMS + " " + "x" * 161
    adapter, _, _ = build(search(MEDRAG, query), DONE)
    response = asyncio.run(adapter.execute(request(MEDRAG, output=True)))
    assert response.metadata["tool_calls"] == 1
    receipt = response.metadata["tool_receipts"][0]
    assert receipt["error_type"] is None
    assert receipt["request"]["arguments"] == {"query": query}
    assert opener.calls == []


def test_local_knowledge_query_is_not_subject_to_external_preflight(setup):
    build, _ = setup
    adapter, _, _ = build()
    sampled = search(KNOWLEDGE, LONG_TERMS + " " + "x" * 161)
    sampled["arguments"]["database"] = "medical_references"
    assert adapter._tool_action_error(
        request=request(KNOWLEDGE), action=StructuredAction.from_value(sampled),
        observations=[],
    ) is None


def test_continuation_retains_shared_budget_and_invalid_action_history(setup):
    build, opener = setup
    adapter, _, _ = build(search(LITERATURE, LONG_TERMS),
                          search(LITERATURE, "synthetic first"),
                          max_tool_calls=2, max_turns=2)
    initial = request(LITERATURE, BOOKSHELF, output=True)
    with pytest.raises(ReactExecutionError) as failed:
        asyncio.run(adapter.execute(initial))
    assert len(failed.value.tool_receipts) == 1
    assert len(opener.calls) == 1
    continued = replace(
        initial, request_id="fixture:continued", graph_revision=2,
        prior_tool_receipts=failed.value.tool_receipts,
        action_history=failed.value.react_trace, continuation_source_agent_id="node",
    )
    repaired, _, _ = build(search(BOOKSHELF, LONG_TERMS),
                           search(BOOKSHELF, "synthetic second"),
                           search(LITERATURE, "synthetic third"), DONE,
                           max_tool_calls=2)
    response = asyncio.run(repaired.execute(continued))
    assert response.metadata["continued_tool_receipt_count"] == 1
    assert response.metadata["continued_action_history_count"] == 2
    assert response.metadata["tool_calls"] == 2
    assert len(response.metadata["tool_receipts"]) == 2
    assert len(opener.calls) == 2
    trace = response.metadata["react_trace"]
    assert trace[2]["public_error_code"] == "query_too_broad_use_at_most_12_clinical_terms"
    assert trace[4]["public_error_code"] == "state_action_not_admitted"


def test_schema_only_publishes_existing_client_content_caps(setup):
    build, _ = setup
    _, _, registry = build()
    for tool in BOUNDED:
        query = registry.require_capability(tool).action_schemas["search"]["properties"]["query"]
        assert query["maxLength"] == 160
        assert "at most 12 clinical content terms" in query["description"]
    trials = registry.require_capability(TRIALS).action_schemas["search"]["properties"]["query"]
    assert trials["maxLength"] == 160
    assert "12 clinical content terms" not in trials["description"]
