"""Concrete OpenAI-compatible Agent gateway for local vLLM and API pools."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.prompts.prompt import FORMAT_PROMPT

from .agent_runtime import (
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2,
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3,
    ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1,
    AgentRequest,
    AgentResponse,
    CommunicationCondition,
    ExecutionPhase,
    UpstreamMessage,
)
from .healthbench_professional_adapter import (
    parse_model_visible_conversation,
)


class OpenAICompatibleGatewayError(RuntimeError):
    """Provider failure with public routing metadata for typed recovery."""

    provider_id: str | None = None
    model_id: str | None = None
    http_status: int | None = None
    request_status: str | None = None


MASKED_UPSTREAM_CONTENT = "[UPSTREAM CONTENT MASKED FOR COMMUNICATION DIAGNOSTIC]"


# NECESSARY_ADAPTATION: HealthBench supplies a native healthcare conversation
# and expects a complete assistant response. This is a task execution boundary,
# not a medical Agent role or workflow template. It contains no rubric,
# reference response, benchmark answer, or evaluator state.
HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL = """Treat diagnoses, plans, drug identities, doses, trial or guideline claims, and routed Agent artifacts as unverified until supported by the supplied conversation or reliable evidence available to this execution. Resolve ambiguous shorthand, entity binding, and internal contradictions before committing to an interpretation; if support is insufficient, state the uncertainty or ask the necessary clarification instead of inventing a fact. When using a Tool, preserve unresolved names and abbreviations from the task verbatim in the query; do not replace them with an inferred expansion. In a completed Artifact, distinguish observed support from inference or missing evidence. Compare material claims across the conversation and routed artifacts. When a supported upstream claim is consistent with the task, preserve its entity-property binding, quantity, unit, time condition, diagnosis, and billing code; replace it only when the conversation or routed evidence contains a concrete conflict, and do not introduce a new decisive claim without such support. When clinically relevant, check red flags, contraindications, interactions, dosing, follow-up, and urgent escalation, and do not endorse unsafe content merely because the user asks to translate, summarize, or reformat it. Preserve the user's language, scope, requested output form, and clinically important quantities. A review must identify concrete support, conflict, or insufficiency rather than treating model agreement as verification. The user-facing response must be self-contained and must not expose Agent IDs, contracts, artifact labels, provenance headings, or workflow instructions."""

# NECESSARY_PROJECT_ADAPTATION: v2 keeps the role- and topology-neutral
# HealthBench boundary above and makes one already-declared Canvas invariant
# explicit to the Executor.  A Director-authored contract is an instruction,
# not an evidence source.  Model metadata selects this version so historical
# catalogs keep their exact rendered prompt when reconstructed.
HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V2 = (
    HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL
    + " The Agent contract describes work to perform, not evidence or an "
    "established clinical conclusion. Never copy a diagnosis, expansion, "
    "quantity, level, treatment, or recommendation from the contract merely "
    "because it appears there; derive material claims from the conversation, "
    "a successful Tool Observation, or a routed artifact with matching "
    "evidence. Before completing, answer every explicit part of the user's "
    "request and do not let an incomplete upstream artifact narrow it."
)

_HEALTHBENCH_EXECUTION_PROTOCOL_V2_METADATA = "contract-is-not-evidence.v2"

# NECESSARY_PROJECT_ADAPTATION: v3 keeps the v2 contract/evidence boundary and
# adds a terminal semantic invariant only for a HealthBench ReAct Output Agent.
# The suffix is separate from v2 so completed v1/v2 conditions reconstruct the
# exact prompts they originally used.
_HEALTHBENCH_EXECUTION_PROTOCOL_V3_METADATA = (
    "contract-is-not-evidence.output-complete.v3"
)

HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V3 = (
    HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V2
    + " When choosing a literature search, use a short query that preserves "
    "the exact unresolved condition, intervention, trial, guideline, drug, "
    "or abbreviation from the conversation and adds only the relation needed "
    "for the task. If a search returns no relevant result, broaden it by "
    "removing restrictive terms instead of adding unrelated terms."
)

# NECESSARY_PROJECT_ADAPTATION: v4 preserves every v3 execution boundary and
# adds a role- and topology-neutral answer-slot binding invariant observed in
# the held-out HealthBench canary.  It applies to every HealthBench Agent so an
# intermediate artifact cannot silently narrow an unresolved item before it
# reaches the Output Agent.  Historical v1-v3 catalogs still render their exact
# original protocol text.
_HEALTHBENCH_EXECUTION_PROTOCOL_V4_METADATA = (
    "contract-is-not-evidence.output-complete.slot-binding.v4"
)

HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V4 = (
    HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V3
    + " Before completing, cover each explicit noun phrase and unresolved "
    "abbreviation in the final user message; do not silently omit one because "
    "an upstream artifact did. Bind every decisive value to the exact entity, "
    "attribute, condition, and procedural stage asked about, and never "
    "substitute a related but different property. For a procedure, explicitly "
    "distinguish the access or entry site, target or tip position, and coverage "
    "or treatment level."
)

# NECESSARY_PROJECT_ADAPTATION: v5 preserves the v4 task/evidence boundary and
# makes three failure classes observed in public HealthBench trajectories
# explicit at the existing SkillFlow Executor boundary.  It remains a
# role- and topology-neutral execution protocol: the Director still chooses
# every Agent, relation, model, Tool, and Output identity.
_HEALTHBENCH_EXECUTION_PROTOCOL_V5_METADATA = (
    "contract-is-not-evidence.output-complete.slot-binding.closure.v5"
)

HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V5 = (
    HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V4
    + " After a successful Tool Observation, complete with an "
    "evidence-derived finding or an explicit statement that the evidence is "
    "insufficient; never complete with a query, keyword list, heading, or "
    "retrieval plan. Before producing a user-facing response, compare the "
    "original conversation, routed findings, and proposed response once. "
    "Resolve internal contradictions that can change disposition or safety, "
    "and preserve the strongest supported requirement for urgent or in-person "
    "evaluation without weakening its necessity or timing. If a requested "
    "translation, summary, or reformatting would reproduce unsafe instructions, "
    "preserve the requested form while adding the necessary safety correction. "
    "If an intermediate review identifies a concrete error, apply the correction "
    "in the requested deliverable instead of returning the review report."
)

_HEALTHBENCH_OUTPUT_REACT_PROTOCOL_V3 = (
    " For this HealthBench Professional ReAct Output Agent, keep the "
    "StructuredAction JSON protocol above. A tool action is only a retrieval "
    "request: its arguments must contain only the admitted `query` field, not "
    "an answer, evidence artifact, plan, or analysis. When choosing the "
    "complete action, `arguments.value` must be the complete, self-contained, "
    "user-facing assistant response that answers every explicit request in the "
    "final user message of the original conversation. It must not be merely a "
    "retrieval plan, evidence artifact, or internal analysis, and it must not "
    "expose AgentGraph, Agent IDs, contracts, or workflow state."
)


def _healthbench_execution_protocol(request: AgentRequest) -> str:
    """Resolve the versioned HealthBench Executor boundary from the catalog."""

    protocol_version = request.model.metadata.get(
        "healthbench_execution_protocol"
    )
    if protocol_version == _HEALTHBENCH_EXECUTION_PROTOCOL_V5_METADATA:
        return HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V5
    if protocol_version == _HEALTHBENCH_EXECUTION_PROTOCOL_V4_METADATA:
        return HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V4
    if protocol_version == _HEALTHBENCH_EXECUTION_PROTOCOL_V3_METADATA:
        return HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V3
    if protocol_version == _HEALTHBENCH_EXECUTION_PROTOCOL_V2_METADATA:
        return HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL_V2
    return HEALTHBENCH_PROFESSIONAL_EXECUTION_PROTOCOL


def _number(metadata: Mapping[str, str], key: str, default: float) -> float:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICompatibleGatewayError(f"model metadata {key} must be numeric") from exc


def _integer(metadata: Mapping[str, str], key: str, default: int) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICompatibleGatewayError(f"model metadata {key} must be an integer") from exc
    if parsed <= 0:
        raise OpenAICompatibleGatewayError(f"model metadata {key} must be positive")
    return parsed


def _non_negative_integer(
    metadata: Mapping[str, str],
    key: str,
    default: Optional[int],
) -> Optional[int]:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICompatibleGatewayError(
            f"model metadata {key} must be an integer"
        ) from exc
    if not 0 <= parsed < 2**64:
        raise OpenAICompatibleGatewayError(
            f"model metadata {key} must be an unsigned 64-bit integer"
        )
    return parsed


def supports_local_sglang_top_k(request: AgentRequest) -> bool:
    """Return whether this exact model arm declares local SGLang ``top_k``.

    ``top_k`` is not part of the portable OpenAI Chat Completions contract.
    SkillFlow sends ``top_k=-1`` to its native SGLang rollout endpoint, so the
    compatible field is forwarded only when the frozen provider/model metadata
    explicitly declares the same local backend and capability.  An endpoint or
    provider name alone is not treated as a capability receipt.
    """

    provider_metadata = request.provider.metadata
    model_metadata = request.model.metadata

    def declared_value(key: str) -> str:
        value = model_metadata.get(key, provider_metadata.get(key, ""))
        return value.strip().casefold() if isinstance(value, str) else ""

    return bool(
        declared_value("sampling_backend") == "sglang"
        and declared_value("deployment_locality") == "local"
        and declared_value("supports_top_k") == "true"
    )


def supports_local_sglang_repetition_penalty(request: AgentRequest) -> bool:
    """Return whether this model arm admits SGLang ``repetition_penalty``.

    SkillFlow's Qwen3.5 direct client sends this native SGLang decoding field,
    but it is not part of the portable OpenAI Chat Completions contract.  Keep
    the same explicit capability boundary used by ``top_k``: neither an
    endpoint name nor a supplied value is sufficient without a declared local
    SGLang deployment and capability.
    """

    provider_metadata = request.provider.metadata
    model_metadata = request.model.metadata

    def declared_value(key: str) -> str:
        value = model_metadata.get(key, provider_metadata.get(key, ""))
        return value.strip().casefold() if isinstance(value, str) else ""

    return bool(
        declared_value("sampling_backend") == "sglang"
        and declared_value("deployment_locality") == "local"
        and declared_value("supports_repetition_penalty") == "true"
    )


def _requested_sampling(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Project only the decoding fields placed on the provider request."""

    chat_template_kwargs = payload.get("chat_template_kwargs")
    chat_template_enable_thinking = (
        chat_template_kwargs.get("enable_thinking")
        if isinstance(chat_template_kwargs, Mapping)
        else None
    )

    return {
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "repetition_penalty": payload.get("repetition_penalty"),
        "max_tokens": payload.get("max_tokens"),
        "seed": payload.get("seed"),
        "chat_template_enable_thinking": chat_template_enable_thinking,
    }


def _sglang_backend_sampling_seed(seed: int) -> int:
    """Project a scientific uint64 seed into SGLang's signed int64 domain."""

    return seed & ((1 << 63) - 1)


def _visible_message_content(
    content: str,
    condition: CommunicationCondition,
) -> str:
    if condition is CommunicationCondition.UPSTREAM_MASKED:
        return MASKED_UPSTREAM_CONTENT
    return content


def _artifact_receipt_references(
    artifact: str,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return public passage IDs and evidence spans cited by one artifact."""

    try:
        value = json.loads(artifact)
    except (TypeError, ValueError, json.JSONDecodeError):
        return frozenset(), ()
    passage_ids: set[str] = set()
    evidence_spans: list[str] = []

    def visit(item: object, *, field_name: str | None = None) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, field_name=str(key))
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, field_name=field_name)
            return
        if not isinstance(item, str) or not item.strip():
            return
        if field_name == "passage_id":
            passage_ids.add(item.strip())
        elif field_name in {"evidence", "evidence_span"}:
            evidence_spans.append(item.strip())

    visit(value)
    return frozenset(passage_ids), tuple(dict.fromkeys(evidence_spans))


def _successful_read_receipt_projection(
    receipts: Sequence[Mapping[str, object]],
    *,
    artifact: str,
) -> tuple[dict[str, object], ...]:
    """Project only artifact-referenced successful read receipts for a model.

    The immutable UpstreamMessage and trajectory keep every Tool receipt.  This
    projection changes only the model-visible communication envelope so a
    semantic-lineage fan-in does not replay unrelated search results or read
    bodies.  Backend provenance validation continues to consume the complete
    receipt tuple from AgentRequest.
    """

    referenced_ids, evidence_spans = _artifact_receipt_references(artifact)
    successful_reads: list[tuple[dict[str, object], str | None, str | None]] = []
    for receipt in receipts:
        if receipt.get("error_type") is not None:
            continue
        request = receipt.get("request")
        result = receipt.get("result")
        if not isinstance(request, Mapping) or request.get("action") != "read":
            continue
        if not isinstance(result, Mapping):
            continue
        value = result.get("value", result)
        if not isinstance(value, Mapping) or value.get("operation") != "read":
            continue
        passage = value.get("passage")
        if not isinstance(passage, Mapping):
            continue
        passage_id = value.get("passage_id", passage.get("passage_id"))
        if not isinstance(passage_id, str):
            arguments = request.get("arguments")
            if isinstance(arguments, Mapping):
                passage_id = arguments.get("passage_id")
        passage_text = passage.get("text")
        successful_reads.append(
            (
                dict(receipt),
                passage_id if isinstance(passage_id, str) else None,
                passage_text if isinstance(passage_text, str) else None,
            )
        )
    selected = [
        receipt
        for receipt, passage_id, passage_text in successful_reads
        if (
            passage_id is not None
            and passage_id in referenced_ids
        )
        or (
            passage_text is not None
            and any(span in passage_text for span in evidence_spans)
        )
    ]
    # A malformed/non-JSON intermediate artifact still needs evidence for a
    # downstream repair diagnosis.  Fall back to successful reads only; never
    # replay search receipts in the semantic-lineage model envelope.
    if not selected and not referenced_ids and not evidence_spans:
        selected = [receipt for receipt, _, _ in successful_reads]
    return tuple(selected)


_HEALTHBENCH_SEARCH_TOOL_ID = "healthbench-authoritative.search"
_HEALTHBENCH_STRUCTURED_EVIDENCE_SCHEMA_V1 = (
    "healthbench.structured-evidence.v1"
)
_HEALTHBENCH_STRUCTURED_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "summary",
        "evidence_items",
        "uncertainties",
    }
)
_PRODUCER_CONTEXT_PROFILES = frozenset(
    {
        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_V1,
        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2,
        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3,
    }
)


def _uses_producer_context(artifact_communication_profile: str) -> bool:
    return artifact_communication_profile in _PRODUCER_CONTEXT_PROFILES


def _healthbench_structured_evidence_references(
    artifact: str,
) -> tuple[dict[str, object], ...] | None:
    """Parse only the versioned, public HealthBench evidence Artifact shape."""

    try:
        value = json.loads(artifact)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping) or not (
        _HEALTHBENCH_STRUCTURED_EVIDENCE_FIELDS <= set(value)
    ):
        return None
    if (
        value.get("schema_version")
        != _HEALTHBENCH_STRUCTURED_EVIDENCE_SCHEMA_V1
    ):
        return None
    if not isinstance(value.get("status"), str):
        return None
    if not isinstance(value.get("summary"), str):
        return None
    uncertainties = value.get("uncertainties")
    if not isinstance(uncertainties, list) or any(
        not isinstance(item, str) for item in uncertainties
    ):
        return None
    evidence_items = value.get("evidence_items")
    if not isinstance(evidence_items, list):
        return None

    references: list[dict[str, object]] = []
    seen_references: set[tuple[str, str]] = set()
    for item in evidence_items:
        if not isinstance(item, Mapping):
            return None
        document_id = item.get("document_id")
        evidence_span = item.get("evidence_span")
        if (
            not isinstance(document_id, str)
            or not document_id.strip()
            or not isinstance(evidence_span, str)
            or not evidence_span.strip()
        ):
            return None
        reference_key = (document_id.strip(), evidence_span.strip())
        if reference_key in seen_references:
            continue
        seen_references.add(reference_key)
        references.append(
            {
                "document_id": reference_key[0],
                "evidence_span": reference_key[1],
                "source": item.get("source"),
                "title": item.get("title"),
                "date": item.get("date"),
                "url": item.get("url"),
            }
        )
    return tuple(references)


def _is_healthbench_search_receipt(receipt: Mapping[str, object]) -> bool:
    return receipt.get("tool_id") == _HEALTHBENCH_SEARCH_TOOL_ID


def _healthbench_search_candidates(
    receipts: Sequence[Mapping[str, object]],
) -> list[tuple[str, str, Mapping[str, object]]]:
    """Reuse the v2 successful, public search-Observation admission boundary."""
    candidates: list[tuple[str, str, Mapping[str, object]]] = []
    for receipt in receipts:
        if not _is_healthbench_search_receipt(receipt):
            continue
        if receipt.get("error_type") is not None:
            continue
        request = receipt.get("request")
        result = receipt.get("result")
        if not isinstance(request, Mapping) or request.get("action") != "search":
            continue
        if not isinstance(result, Mapping) or result.get("completed") is not True:
            continue
        value = result.get("value", result)
        if not isinstance(value, Mapping) or value.get("operation") != "search":
            continue
        query = value.get("query")
        if not isinstance(query, str) or not query.strip():
            arguments = request.get("arguments")
            query = (
                arguments.get("query")
                if isinstance(arguments, Mapping)
                else None
            )
        if not isinstance(query, str) or not query.strip():
            continue
        evidence = value.get("evidence")
        if not isinstance(evidence, list):
            continue
        for result_item in evidence:
            if isinstance(result_item, Mapping):
                candidates.append(
                    (
                        str(receipt.get("tool_id")),
                        " ".join(query.split()),
                        result_item,
                    )
                )
    return candidates


def _healthbench_evidence_receipt_projection(
    receipts: Sequence[Mapping[str, object]],
    *,
    artifact: str,
) -> tuple[tuple[dict[str, object], ...], int | None]:
    """Return compact receipt facts for evidence explicitly cited by Artifact.

    Complete receipts remain immutable on ``AgentRequest`` and in persisted
    execution records.  This projection is derived from successful search
    receipts, contains no timing or unrelated result bodies, and never falls
    back to replaying every search result when the Artifact is malformed.
    """

    references = _healthbench_structured_evidence_references(artifact)
    if references is None:
        return (), None
    candidates = _healthbench_search_candidates(receipts)

    projected: list[dict[str, object]] = []
    seen_projection: set[str] = set()
    for reference in references:
        for tool_id, query, result_item in candidates:
            if result_item.get("document_id") != reference["document_id"]:
                continue
            if any(
                reference.get(field_name) != result_item.get(field_name)
                for field_name in ("source", "title", "date", "url")
            ):
                continue
            excerpt = result_item.get("excerpt")
            evidence_span = reference["evidence_span"]
            normalized_excerpt = (
                " ".join(excerpt.split()) if isinstance(excerpt, str) else ""
            )
            normalized_span = (
                " ".join(evidence_span.split())
                if isinstance(evidence_span, str)
                else ""
            )
            if (
                not normalized_span
                or normalized_span not in normalized_excerpt
            ):
                continue
            compact = {
                "tool_id": tool_id,
                "query": query,
                "document_id": result_item.get("document_id"),
                "source": result_item.get("source"),
                "title": result_item.get("title"),
                "date": result_item.get("date"),
                "url": result_item.get("url"),
                "evidence_span": evidence_span,
            }
            serialized = json.dumps(
                compact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if serialized not in seen_projection:
                seen_projection.add(serialized)
                projected.append(compact)
            break
    return tuple(projected), len(references)


def _healthbench_projection_status(
    matched_count: int,
    reference_count: int | None,
) -> str:
    if reference_count is None:
        return "unavailable-invalid-structured-artifact"
    if reference_count == 0:
        return "complete-no-evidence-references"
    if matched_count == reference_count:
        return "complete"
    if matched_count:
        return "partial"
    return "unavailable-no-receipt-match"


_COMPACT_PROVENANCE_FIELDS = (
    "source_agent_id",
    "target_agent_id",
    "message_type",
    "artifact_type",
    "artifact_version",
    "graph_revision",
    "environment_revision",
    "source_model_id",
    "source_execution_mode",
    "source_finish_reason",
)


def _compact_healthbench_input_provenance(
    provenance_items: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Remove recursive receipt bodies and duplicate Artifact aliases."""

    compact_items: list[dict[str, object]] = []
    seen_items: set[str] = set()
    for provenance in provenance_items:
        compact = {
            field_name: provenance[field_name]
            for field_name in _COMPACT_PROVENANCE_FIELDS
            if field_name in provenance and provenance[field_name] is not None
        }
        artifact = provenance.get("artifact")
        if not isinstance(artifact, str):
            artifact = provenance.get("artifact_body")
        if not isinstance(artifact, str):
            artifact = provenance.get("content")
        if isinstance(artifact, str):
            compact["artifact"] = artifact

        raw_receipts = provenance.get("tool_receipts")
        receipts = (
            tuple(
                receipt
                for receipt in raw_receipts
                if isinstance(receipt, Mapping)
            )
            if isinstance(raw_receipts, (list, tuple))
            else ()
        )
        if receipts and any(
            _is_healthbench_search_receipt(receipt) for receipt in receipts
        ):
            projection, reference_count = (
                _healthbench_evidence_receipt_projection(
                    receipts,
                    artifact=artifact if isinstance(artifact, str) else "",
                )
            )
            compact["tool_receipt_projection"] = (
                "artifact-referenced-healthbench-evidence-v1"
            )
            compact["tool_receipt_projection_status"] = (
                _healthbench_projection_status(
                    len(projection),
                    reference_count,
                )
            )
            if projection:
                compact["evidence_receipts"] = list(projection)

        raw_nested = provenance.get("input_artifact_provenance")
        if isinstance(raw_nested, (list, tuple)):
            nested = _compact_healthbench_input_provenance(
                tuple(
                    item for item in raw_nested if isinstance(item, Mapping)
                )
            )
            if nested:
                compact["input_artifact_provenance"] = list(nested)

        serialized = json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if serialized not in seen_items:
            seen_items.add(serialized)
            compact_items.append(compact)
    return tuple(compact_items)


# NECESSARY_ADAPTATION: SkillFlow GenericTaskEnvironment.step retains public
# Tool Observations separately from the Agent's answer; BoundedAgent records
# them without claiming their truth. FlowSteer's routed result/feedback path
# is retained, but its text-only result cannot represent receipt provenance.
# v3 fixes that projection gap without changing execution, roles or topology.
_HEALTHBENCH_V3_UPSTREAM_CHARS = 12000
_HEALTHBENCH_V3_PROMPT_CHARS = 24000
_HEALTHBENCH_V3_TRUNCATED = "[projection truncated; full receipt retained in trajectory]"
_HEALTHBENCH_V3_EXECUTION_SUPPLEMENT = (
    " Retrieved excerpts are not automatically endorsed facts. Compare their "
    "source, population, relation and date with the conversation and producer "
    "summary; do not discard a sourced result merely because it conflicts with "
    "model memory or appears newer. Missing evidence does not establish "
    "nonexistence. Answer the relation actually asked, not a neighboring "
    "question; do not invent patient facts. A requested format does not cancel "
    "material risk or contradiction handling. Preserve supported conclusions "
    "and explicitly distinguish unresolved conflicts from verified findings."
)


def _healthbench_v3_text(value: object, limit: int) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) <= limit:
        return text
    # Preserve the ending as well: exceptions and temporal qualifiers often
    # occur after a recommendation. Explicitly disclose any omitted middle.
    available = max(0, limit - len(_HEALTHBENCH_V3_TRUNCATED))
    head = (available + 1) // 2
    tail = available - head
    return text[:head] + _HEALTHBENCH_V3_TRUNCATED + (text[-tail:] if tail else "")


def _healthbench_v3_artifact(artifact: str) -> object:
    # Citation text is projected once, from matching receipts, below. Keep
    # producer conclusions (including a mistaken rejection) visibly separate.
    if _healthbench_structured_evidence_references(artifact) is not None:
        value = json.loads(artifact)
        uncertainties = list(dict.fromkeys(value["uncertainties"]))
        visible_uncertainties = uncertainties if len(uncertainties) <= 4 else uncertainties[:2] + uncertainties[-2:]
        partial = (
            len(value["status"]) > 120 or len(value["summary"]) > 2400
            or len(uncertainties) > 4
            or any(len(item) > 300 for item in visible_uncertainties)
        )
        return {
            "schema_version": value["schema_version"],
            "status": _healthbench_v3_text(value["status"], 120),
            "summary": _healthbench_v3_text(value["summary"], 2400),
            "uncertainties": [
                _healthbench_v3_text(item, 300)
                for item in visible_uncertainties
            ],
            "projection_status": "partial" if partial else "complete",
            "omitted_uncertainties_count": max(0, len(uncertainties) - 4),
        }
    return _healthbench_v3_text(artifact, 3600)


def _healthbench_v3_interpretations(
    citations: Sequence[Mapping[str, object]], match: Mapping[str, object], producer: str,
) -> tuple[dict[str, object], ...]:
    interpretations: list[dict[str, object]] = []
    for citation in citations:
        if any(citation.get(field) != match.get(field) for field in ("document_id", "source", "title", "date", "url")):
            continue
        if str(citation.get("evidence_span", "")).strip() != match["evidence_span"]:
            continue
        interpretation = {
            "producer_agent_id": producer,
            "claim_status": "producer-interpretation-not-independently-verified",
            **{
                field: _healthbench_v3_text(citation.get(field), limit)
                for field, limit in (("supported_claim", 1000), ("conditions_or_qualifiers", 1600))
            },
        }
        if (interpretation["supported_claim"] or interpretation["conditions_or_qualifiers"]) and interpretation not in interpretations:
            interpretations.append(interpretation)
    return tuple(interpretations)


def _healthbench_v3_receipts(
    item: UpstreamMessage,
    *,
    seen_sources: set[tuple[str, str]],
    char_budget: int,
) -> dict[str, object]:
    """Bounded non-recursive projection, not a new retrieval or grading step."""

    pending = [(item.source_agent_id, item.artifact, item.tool_receipts, item.input_artifact_provenance, 0)]
    documents: dict[tuple[str, str], dict[str, object]] = {}
    source_reference_interpretations: list[dict[str, object]] = []
    visited: set[tuple[str, str]] = set()
    references_total = matched_total = 0
    examined = 0
    while pending and examined < 16:
        producer, artifact, receipts, nested, depth = pending.pop(0)
        identity = (producer, artifact)
        if identity in visited:
            continue
        visited.add(identity)
        examined += 1
        matches, reference_count = _healthbench_evidence_receipt_projection(receipts, artifact=artifact)
        producer_items = (
            json.loads(artifact)["evidence_items"]
            if reference_count is not None else []
        )
        references_total += reference_count or 0
        matched_total += len(matches)
        for tool_id, query, result in _healthbench_search_candidates(receipts):
            document_id = result.get("document_id")
            excerpt = result.get("excerpt")
            if not isinstance(document_id, str) or not document_id.strip() or not isinstance(excerpt, str) or not excerpt.strip():
                continue
            key = (str(result.get("source") or ""), document_id)
            if key in seen_sources:
                # Do not replay the same source body, but retain a later
                # producer's new interpretation/qualifier of that source.
                for match in matches:
                    if match["document_id"] != document_id or match["source"] != result.get("source"):
                        continue
                    for interpretation in _healthbench_v3_interpretations(producer_items, match, producer):
                        reference = {
                            **interpretation,
                            "source_reference": {"document_id": document_id, "source": result.get("source")},
                            "receipt_bound_evidence_span": _healthbench_v3_text(match["evidence_span"], 1200),
                        }
                        if reference not in source_reference_interpretations:
                            source_reference_interpretations.append(reference)
                continue
            if key not in documents:
                documents[key] = {
                    "evidence_status": "retrieved-not-endorsed",
                    "producer_agent_id": producer,
                    "tool_id": tool_id,
                    "query": _healthbench_v3_text(query, 320),
                    **{
                        field: _healthbench_v3_text(result.get(field), limit)
                        for field, limit in (("document_id", 256), ("source", 180), ("title", 360), ("date", 80), ("url", 512))
                    },
                    "excerpts": [],
                    "artifact_cited_spans": [],
                    "producer_interpretations": [],
                }
            document = documents[key]
            excerpts = document["excerpts"]
            assert isinstance(excerpts, list)
            bounded_excerpt = _healthbench_v3_text(" ".join(excerpt.split()), 1600)
            if bounded_excerpt not in excerpts and len(excerpts) < 2:
                excerpts.append(bounded_excerpt)
            spans = document["artifact_cited_spans"]
            assert isinstance(spans, list)
            for match in matches:
                if match["document_id"] == document_id and match["source"] == result.get("source"):
                    span = _healthbench_v3_text(match["evidence_span"], 1200)
                    if span not in spans and len(spans) < 2:
                        spans.append(span)
                    interpretations = document["producer_interpretations"]
                    assert isinstance(interpretations, list)
                    for interpretation in _healthbench_v3_interpretations(producer_items, match, producer):
                        if span not in spans:
                            document["producer_interpretations_truncated"] = True
                            continue
                        # Matching an excerpt establishes provenance, not the
                        # truth of its producer's clinical interpretation.
                        interpretation["receipt_bound_span_index"] = spans.index(span)
                        if interpretation not in interpretations:
                            if len(interpretations) < 6:
                                interpretations.append(interpretation)
                            else:
                                document["producer_interpretations_truncated"] = True
        if depth < 3:
            for provenance in nested[:16]:
                artifact_value = provenance.get("artifact", provenance.get("artifact_body", provenance.get("content", "")))
                raw_receipts = provenance.get("tool_receipts", ())
                raw_nested = provenance.get("input_artifact_provenance", ())
                pending.append((
                    str(provenance.get("source_agent_id", "unknown")),
                    artifact_value if isinstance(artifact_value, str) else "",
                    tuple(r for r in raw_receipts if isinstance(r, Mapping)) if isinstance(raw_receipts, (tuple, list)) else (),
                    tuple(p for p in raw_nested if isinstance(p, Mapping)) if isinstance(raw_nested, (tuple, list)) else (),
                    depth + 1,
                ))

    result: dict[str, object] = {
        "tool_receipt_projection": "bounded-healthbench-retrieved-evidence-v3",
        "artifact_reference_count": references_total,
        "receipt_bound_reference_count": matched_total,
        "evidence_receipts": [],
        "projection_truncated": bool(pending),
    }
    if source_reference_interpretations:
        visible_interpretations: list[dict[str, object]] = []
        result["previously_projected_source_interpretations"] = visible_interpretations
        for interpretation in source_reference_interpretations:
            visible_interpretations.append(interpretation)
            if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) > char_budget:
                visible_interpretations.pop()
                result["projection_truncated"] = True
    rows = result["evidence_receipts"]
    assert isinstance(rows, list)
    # Receipt-bound citations are preserved first; unendorsed results still
    # remain available when the producer rejects or omits all citations.
    ordered = sorted(documents.items(), key=lambda pair: not bool(pair[1]["artifact_cited_spans"]))
    for key, document in ordered:
        rows.append(document)
        if len(rows) > 6 or len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) > char_budget:
            rows.pop()
            result["projection_truncated"] = True
            continue
        seen_sources.add(key)
    return result


def _format_healthbench_upstream_v3(
    messages: Sequence[UpstreamMessage],
    condition: CommunicationCondition,
    *,
    include_dependency: bool,
    state: dict[str, Any],
) -> str:
    rendered: list[str] = []
    for item in messages:
        identity = (item.source_agent_id, item.target_agent_id, item.artifact_version, item.artifact)
        if identity in state["envelopes"]:
            continue
        state["envelopes"].add(identity)
        allowance = min(_HEALTHBENCH_V3_UPSTREAM_CHARS, state["remaining"] - 100)
        if allowance < 512:
            marker = _HEALTHBENCH_V3_TRUNCATED[:max(0, state["remaining"] - 2)]
            if marker:
                rendered.append(marker)
                state["remaining"] -= len(marker) + 2
            break
        envelope: dict[str, object] = {
            "source_agent": item.source_agent_id,
            "target_agent": item.target_agent_id,
            "artifact_version": item.artifact_version,
            "source_execution_mode": item.source_execution_mode,
            "source_model_id": item.source_model_id,
        }
        if condition is CommunicationCondition.UPSTREAM_MASKED:
            envelope["artifact"] = MASKED_UPSTREAM_CONTENT
        else:
            envelope["source_contract_provenance"] = _healthbench_v3_text(item.source_contract, 700)
            if include_dependency:
                envelope["request_or_dependency"] = _healthbench_v3_text(item.request_or_dependency, 500)
            envelope["producer_artifact"] = _healthbench_v3_artifact(item.artifact)
            base_chars = len(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))
            if allowance - base_chars > 800:
                envelope["retrieval_evidence"] = _healthbench_v3_receipts(
                    item, seen_sources=state["sources"], char_budget=allowance - base_chars - 120,
                )
        text = "[Upstream artifact]\n" + json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        text = _healthbench_v3_text(text, allowance)
        state["remaining"] -= len(text) + 2
        rendered.append(text)
    return "\n\n".join(rendered) if rendered else "(none)"


def _format_upstream(
    messages: Sequence[UpstreamMessage],
    condition: CommunicationCondition,
    *,
    include_dependency: bool = True,
    project_artifact_read_receipts: bool = False,
    project_healthbench_structured_evidence: bool = False,
    artifact_communication_profile: str = "legacy",
    healthbench_projection_state: dict[str, Any] | None = None,
) -> str:
    if not messages:
        return "(none)"
    if (
        project_healthbench_structured_evidence
        and artifact_communication_profile
        == ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3
    ):
        return _format_healthbench_upstream_v3(
            messages, condition, include_dependency=include_dependency,
            state=healthbench_projection_state if healthbench_projection_state is not None else {
                "remaining": _HEALTHBENCH_V3_PROMPT_CHARS,
                "sources": set(), "envelopes": set(),
            },
        )
    rendered = []
    seen_exact_envelopes: set[str] = set()
    for item in messages:
        if (
            _uses_producer_context(artifact_communication_profile)
            and item.artifact_version is not None
        ):
            try:
                exact_envelope = json.dumps(
                    item.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                exact_envelope = ""
            if exact_envelope and exact_envelope in seen_exact_envelopes:
                continue
            if exact_envelope:
                seen_exact_envelopes.add(exact_envelope)
        envelope = [
            "[Upstream artifact]",
            f"source_agent: {item.source_agent_id}",
            f"target_agent: {item.target_agent_id}",
            f"message_type: {item.message_type}",
            f"artifact_type: {item.artifact_type}",
        ]
        if item.graph_revision is not None:
            envelope.append(f"graph_revision: {item.graph_revision}")
        if item.environment_revision is not None:
            envelope.append(f"environment_revision: {item.environment_revision}")
        if _uses_producer_context(artifact_communication_profile):
            if item.artifact_version is not None:
                envelope.append(f"artifact_version: {item.artifact_version}")
            if item.source_model_id is not None:
                envelope.append(f"source_model_id: {item.source_model_id}")
            if item.source_execution_mode is not None:
                envelope.append(
                    f"source_execution_mode: {item.source_execution_mode}"
                )
            if item.source_role_family is not None:
                envelope.append(
                    f"source_role_family: {item.source_role_family}"
                )
            if item.source_completion_condition is not None:
                envelope.append(
                    "source_completion_condition: "
                    + item.source_completion_condition
                )
            if item.source_finish_reason is not None:
                envelope.append(
                    f"source_finish_reason: {item.source_finish_reason}"
                )
            if item.source_contract is not None:
                envelope.extend(
                    [
                        "source_contract_provenance:",
                        item.source_contract,
                    ]
                )
        if include_dependency and item.request_or_dependency is not None:
            envelope.append(
                f"request_or_dependency: {item.request_or_dependency}"
            )
        healthbench_projection_candidate = (
            project_healthbench_structured_evidence
            and bool(item.tool_receipts)
            and (
                item.source_execution_mode == "react"
                or any(
                    _is_healthbench_search_receipt(receipt)
                    for receipt in item.tool_receipts
                )
            )
        )
        if healthbench_projection_candidate:
            evidence_projection, reference_count = (
                _healthbench_evidence_receipt_projection(
                    item.tool_receipts,
                    artifact=item.artifact,
                )
            )
            envelope.append(
                "tool_receipt_projection: "
                "artifact-referenced-healthbench-evidence-v1"
            )
            envelope.append(
                "tool_receipt_projection_status: "
                + _healthbench_projection_status(
                    len(evidence_projection),
                    reference_count,
                )
            )
            if evidence_projection:
                envelope.append(
                    "evidence_receipts: "
                    + json.dumps(
                        list(evidence_projection),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            visible_tool_receipts: tuple[dict[str, object], ...] = ()
        elif project_artifact_read_receipts:
            visible_tool_receipts = _successful_read_receipt_projection(
                item.tool_receipts,
                artifact=item.artifact,
            )
        else:
            visible_tool_receipts = tuple(
                dict(receipt) for receipt in item.tool_receipts
            )
        if (
            project_artifact_read_receipts
            and item.tool_receipts
            and not healthbench_projection_candidate
        ):
            envelope.append(
                "tool_receipt_projection: artifact-referenced-successful-reads"
            )
        if visible_tool_receipts:
            envelope.append(
                "tool_receipts: "
                + json.dumps(
                    list(visible_tool_receipts),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        if item.input_artifact_provenance:
            if project_healthbench_structured_evidence:
                visible_input_provenance = (
                    _compact_healthbench_input_provenance(
                        item.input_artifact_provenance
                    )
                )
                envelope.append(
                    "input_artifact_provenance_projection: "
                    "compact-structured-evidence-v2"
                )
            else:
                visible_input_provenance = tuple(
                    dict(provenance)
                    for provenance in item.input_artifact_provenance
                )
            envelope.append(
                "input_artifact_provenance: "
                + json.dumps(
                    list(visible_input_provenance),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        envelope.extend(
            [
                # Keep FlowSteer's model-visible label stable; the persisted
                # communication envelope carries the canonical artifact_body.
                "artifact:",
                _visible_message_content(item.artifact, condition),
            ]
        )
        rendered.append("\n".join(envelope))
    return "\n\n".join(rendered)


def _semantic_role(request: AgentRequest) -> str:
    role_family = request.agent.role_family
    return role_family.casefold() if isinstance(role_family, str) else ""


_HOTPOTQA_COMPLETE_ENTITY_SURFACE_RULE = (
    "A single-entity answer surface is one complete, evidence-aligned referential "
    "surface. Minimality removes only alias lists, explanations, redundant "
    "question-head nouns, and text outside the selected entity mention; it never "
    "permits truncating that entity mention. For a who-question licensed by a "
    "possessive construction, the answer slot is the full possessor entity mention "
    "immediately before the possessive marker ('s or its typographic equivalent), "
    "not the possessed attribute. Preserve every component of that evidence-aligned "
    "mention, including any title, honorific, or name suffix; no component may be "
    "dropped or reclassified as an unrequested qualifier, even when a shorter form is "
    "coreferential. Outside that possessive construction, a shorter canonical name or "
    "alias is admissible only when an explicit identity proposition supports that exact "
    "surface choice. "
)


_HOTPOTQA_REASONER_PROTOCOL = (
    "You are the semantic Reasoner, not a formatter or verifier. Preserve the "
    "question's original scope, relation, qualifiers, comparison criterion, and "
    "answer type and answer cardinality. Align every database or retrieved fact "
    "to a proposition with "
    "subject/entity, predicate/relation, object or attribute value, and qualifiers; "
    "preserve the sentence's asserted semantic roles instead of placing the desired "
    "candidate into an unrelated field. For a comparison fact, the compared entity "
    "is normally the proposition subject and its date, number, or other compared "
    "attribute is the object_or_attribute_value. Then align that proposition to the "
    "answer slot actually requested. Apply the "
    "question's wh-word answer-type constraint: a Which-comparison returns the "
    "compared entity rather than the comparison value, and a who-question returns "
    "the evidence-supported answer-bearing entity, which may be a person or "
    "organization, rather than a possessive attribute phrase. You alone "
    "determine the semantic candidate. Return exactly the six structured fields "
    "question_scope, answer_slot, evidence_propositions, multi_hop_chain, "
    "candidate_answer, and evidence. Copy question_scope exactly from the original "
    "question. answer_slot contains exactly answer_type, answer_cardinality, qualifiers, "
    "proposition_index, and answer_field. proposition_index selects one item in "
    "evidence_propositions and answer_field selects either its subject or its "
    "object_or_attribute_value; candidate_answer must equal that selected value, "
    "except for a deterministic year-to-decade normalization when the original "
    "question explicitly requests a decade. "
    + _HOTPOTQA_COMPLETE_ENTITY_SURFACE_RULE
    + "If a comparison produces unexpectedly "
    "equal values, recheck the "
    "original scope, both entity-attribute bindings, retrieved evidence, and any "
    "upstream contract narrowing before concluding a tie."
)

_HOTPOTQA_VERIFIER_PROTOCOL = (
    "You are the semantic Verifier, not a Reasoner or formatter. Inspect the routed "
    "Reasoner candidate against explicit database or retrieved evidence. Check all "
    "seven gates: evidence explicitly supports the candidate; each entity is bound "
    "to the correct attribute/value; every required multi-hop bridge is complete; "
    "the original question scope was not narrowed or changed; the answer type and "
    "cardinality match the original question; the answer surface is one complete, "
    "evidence-aligned referential surface without an alias list, redundant "
    "question-head noun, or text outside the selected entity mention; "
    "and every canonical-name or alias choice has an explicit identity binding in the "
    "evidence propositions. In a Which-comparison, reject a numeric/date comparison "
    "value as the candidate. "
    + _HOTPOTQA_COMPLETE_ENTITY_SURFACE_RULE
    + "In that possessive construction, reject the possessed attribute, an incomplete "
    "possessor entity mention, or any candidate that shortens the full possessor mention. "
    "You must not "
    "select, replace, canonicalize, or invent a different candidate. Return exactly "
    "these labeled fields: `Candidate answer:`, `Evidence supported:`, "
    "`Entity attribute binding correct:`, `Multi-hop complete:`, `Scope preserved:`, "
    "`Answer type cardinality correct:`, `Minimal answer surface:`, "
    "`Alias binding correct:`, and `Verification status:`. Copy the Reasoner's Candidate "
    "answer character-for-character. Every check field must contain only the literal "
    "boolean `true` or `false`; never put the candidate or an explanation in a check "
    "field. Set each check to true only when explicitly supported; set Verification "
    "status to supported only when all seven checks pass, "
    "otherwise set it to repair_required. The false check fields are the repair "
    "diagnosis; do not supply a substitute candidate for the Formatter."
)

_QA_COMPLETE_ENTITY_SURFACE_RULE = (
    "A semantic answer must use one concise evidence-grounded surface form. "
    "Preserve the entity identity, requested relation, answer type, cardinality, "
    "and question qualifiers. A spelling variant, alias, abbreviation, or "
    "canonical name is admissible only when an explicit identity binding in the "
    "retrieved evidence supports that surface. A candidate that exactly copies "
    "the selected proposition argument and occurs verbatim as one complete entity "
    "mention in its evidence span satisfies the concise-surface check even when "
    "the passage also contains a shorter coreferential name; do not reject that "
    "complete mention merely because a shorter alias or subspan exists. Reject "
    "alias lists, appositive glosses, redundant answer-type nouns, text outside a "
    "single complete entity mention, and surfaces without explicit identity "
    "support. The Formatter must not perform canonicalization. "
)

_QA_REASONER_PROTOCOL = (
    "You are the semantic Reasoner, not a Retriever, Verifier, or Formatter. "
    "Preserve the original question scope and bind its answer slot to the target "
    "entity and requested relation. Use only successful qa-retrieval read evidence "
    "and its Tool receipts. Represent each supporting fact as subject/entity, "
    "predicate/relation, object or attribute value, qualifiers, and an exact "
    "evidence span. Resolve spelling variants, aliases, and entity ambiguity only "
    "when retrieved evidence supplies the identity binding. You alone determine "
    "the semantic candidate. In every evidence proposition, copy each entity "
    "surface exactly as it occurs in that proposition's evidence span; do not "
    "silently replace a surname, pronoun, or shortened mention with the longer "
    "question entity. Represent that coreference or alias only through a separate "
    "evidence-supported identity proposition. Return exactly the six structured "
    "fields "
    "question_scope, answer_slot, evidence_propositions, multi_hop_chain, "
    "candidate_answer, and evidence. Copy question_scope exactly. answer_slot "
    "contains exactly answer_type, answer_cardinality, qualifiers, "
    "proposition_index, and answer_field. proposition_index selects one evidence "
    "proposition and answer_field selects its subject or "
    "object_or_attribute_value; candidate_answer must equal that selected value, "
    "except for a deterministic year-to-decade normalization when the original "
    "question explicitly requests a decade. "
    + _QA_COMPLETE_ENTITY_SURFACE_RULE
    + "If the retrieved evidence does not bind both entity identity and target "
    "relation, do not guess or fabricate a candidate; continue the admitted "
    "retrieval policy or report knowledge_base_coverage_failure."
)

_QA_VERIFIER_PROTOCOL = (
    "You are the semantic Verifier, not a Retriever, Reasoner, or Formatter. "
    "Check the routed Reasoner artifact against successful qa-retrieval read Tool "
    "receipts. Verify evidence support, entity-to-relation binding, complete "
    "reasoning lineage, unchanged question scope, answer type and cardinality, "
    "concise evidence-grounded answer surface, and explicit alias or canonical-name "
    "binding. "
    + _QA_COMPLETE_ENTITY_SURFACE_RULE
    + "You must not select, replace, canonicalize, or invent a candidate. "
    "Return exactly these labeled fields: `Candidate answer:`, `Evidence supported:`, "
    "`Entity attribute binding correct:`, `Multi-hop complete:`, `Scope preserved:`, "
    "`Answer type cardinality correct:`, `Minimal answer surface:`, "
    "`Alias binding correct:`, and `Verification status:`. Copy the Reasoner's "
    "Candidate answer character-for-character. Every check is the literal boolean "
    "`true` or `false`. Set Verification status to supported only when every check "
    "passes; otherwise set it to repair_required and do not provide a substitute."
)


def build_agent_messages(request: AgentRequest) -> list[dict[str, str]]:
    """Build finite-phase prompts without exposing provider credentials."""

    execution_mode = getattr(
        request.agent.execution_mode,
        "value",
        request.agent.execution_mode,
    )
    semantic_role = _semantic_role(request)
    hotpot_semantic = request.semantic_protocol == "hotpotqa_verified_answer_slot_v1"
    unified_qa_semantic = request.semantic_protocol == "qa_verified_answer_lineage_v2"
    semantic_lineage = hotpot_semantic or unified_qa_semantic
    if execution_mode in {"react", "coding"}:
        # SkillFlow's BoundedAgent asks the policy for one StructuredAction per
        # model turn.  The execution adapter, not this provider boundary,
        # decides when a ``complete`` action becomes the node artifact.  The
        # generic Output-Agent answer wrapper would otherwise override the
        # JSON action contract and make an Output ReAct/Coding node
        # unexecutable.
        protocol = (
            "This is one bounded execution-policy turn. Return exactly one "
            "StructuredAction JSON object using the schema and admitted "
            "resources in the assigned contract, with no Markdown or text "
            "outside that object. A tool action requests one public "
            "observation; a complete action supplies the declared node "
            "artifact. The current state-conditioned action domain in the "
            "assigned contract is authoritative: choose only an action shown "
            "as currently admissible, and put only its declared keys in "
            "arguments. If completion is not currently admissible, do not put "
            "the eventual artifact or answer fields into a Tool action. Do not "
            "emit <answer> tags in this internal action."
        )
        if semantic_lineage and semantic_role == "reasoner":
            protocol += (
                " ReAct is only this node's execution schedule, not its role. "
                "Never place semantic-answer fields in search/read arguments. "
                "Only when the assigned contract marks completion currently "
                "admissible, put the structured semantic Reasoner artifact defined "
                "there in arguments.value. "
                + (
                    _HOTPOTQA_COMPLETE_ENTITY_SURFACE_RULE
                    if hotpot_semantic
                    else (
                        _QA_COMPLETE_ENTITY_SURFACE_RULE
                        + "In every evidence proposition, copy each entity surface "
                        "exactly as it occurs in that proposition's evidence span; "
                        "represent a surname, pronoun, shortened mention, or longer "
                        "question entity coreference only through a separate "
                        "evidence-supported identity proposition. "
                        + "Bind entity identity and the requested relation to "
                        "successful qa-retrieval read Tool receipts; if that "
                        "grounding is absent, do not guess or fabricate evidence."
                    )
                )
            )
        elif semantic_lineage and semantic_role == "evidence_retriever":
            protocol += (
                " ReAct is only this node's execution schedule, not its role. "
                "You are the Evidence Retriever and own only public retrieval "
                "provenance. Use the admitted search/read actions; only when "
                "completion is admitted, submit the exact grounded evidence "
                "artifact defined by the assigned schema in arguments.value. "
                "Cite a successful read receipt and do not select or emit "
                "candidate_answer, answer_slot, or final_answer."
            )
        elif semantic_lineage and semantic_role == "verifier":
            protocol += (
                " ReAct is only this node's execution schedule, not its role. "
                "When completing, put the full labeled Verifier artifact required "
                "below in arguments.value. "
                + (
                    _HOTPOTQA_VERIFIER_PROTOCOL
                    if hotpot_semantic
                    else _QA_VERIFIER_PROTOCOL
                )
            )
        elif request.is_format_predecessor:
            protocol += (
                " In a complete action that supplies the semantic answer to the "
                "terminal Format Agent, set arguments.value to exactly two fields: one line "
                "`Candidate answer: ...` containing only the answer value, followed by "
                "one `Evidence: ...` field. Do not put a sentence or question restatement "
                "in the Candidate answer field."
            )
    elif request.is_format_agent and semantic_lineage:
        protocol = (
            "You are the terminal FlowSteer Format Operator. The solution has already "
            + (
                "been computed and passed by a Verifier in exactly one routed "
                "upstream artifact. "
                if hotpot_semantic
                else "been computed as one explicit candidate in a routed upstream "
                "artifact. "
            )
            + "You will not receive the original question. Follow the copying "
            "instructions in the user message; do not solve, verify, or extend the "
            "answer; do not canonicalize or reselect it."
        )
    elif semantic_lineage and semantic_role == "reasoner":
        protocol = (
            _HOTPOTQA_REASONER_PROTOCOL
            if hotpot_semantic
            else _QA_REASONER_PROTOCOL
        ) + " Do not use <answer> tags."
    elif semantic_lineage and semantic_role == "verifier":
        protocol = (
            _HOTPOTQA_VERIFIER_PROTOCOL
            if hotpot_semantic
            else _QA_VERIFIER_PROTOCOL
        ) + " Do not use <answer> tags."
    elif request.is_format_agent:
        protocol = (
            "You are the terminal FlowSteer Format Operator. The solution has already "
            "been computed in exactly one routed upstream artifact. Follow the extraction "
            "instructions in the user message; do not solve, verify, or extend the answer."
        )
    elif request.is_format_predecessor:
        protocol = (
            "You are the direct semantic predecessor of the terminal Format Agent. "
            "Follow your assigned contract and compute the answer from the task and routed "
            "evidence. Return exactly two fields: one `Candidate answer: ...` line containing "
            "only the answer value, followed by one `Evidence: ...` field. Do not use "
            "<answer> tags."
        )
    elif unified_qa_semantic and request.is_output_agent:
        protocol = (
            "You are the unique Output Agent. Use only routed upstream artifacts "
            "and successful qa-retrieval read receipts. If an explicit semantic "
            "candidate is routed to you, preserve it character-for-character; if "
            "multiple candidates are present, they must agree. Otherwise derive the "
            "short answer directly from the routed receipt-grounded evidence. Return "
            "exactly one <answer>...</answer> wrapper with no explanation. Do not "
            "invent evidence or assume a fixed Retriever, Reasoner, Verifier, or "
            "Formatter sequence."
        )
    elif request.is_output_agent:
        # DIRECT_REUSE: restore the generic Output/intermediate execution
        # boundary used by the highest completed HealthBench official_v1
        # condition.  The final sentence is the minimal project adaptation
        # evidenced by that condition's residual internal-Artifact leakage.
        protocol = (
            "You are the unique Output Agent. Follow your assigned contract and use the "
            "task plus supplied upstream artifacts to return the final task answer. Treat "
            "each routed upstream artifact as the declared dependency for this node; do "
            "not silently redo or ignore an upstream responsibility unless its artifact "
            "has a concrete conflict with the task. Preserve a concise answer when the "
            "artifacts support it and resolve concrete conflicts against the task. Preserve "
            "the output form and level of detail required by the task and Agent contract; "
            "do not collapse a required long-form, structured, code, or environment artifact "
            "to a short answer span. If the task supplies legal or admissible actions and asks "
            "for one action, return exactly one listed executable action with no explanation. "
            "Do not expose AgentGraph identifiers, internal Artifact or provenance labels, "
            "or intermediate-analysis headings, and do not repeat the same rationale unless "
            "the task explicitly requests them."
        )
    else:
        protocol = (
            "You are an intermediate AgentGraph node. Follow your assigned contract and "
            "return only the requested evidence, facts, partial reasoning, or verification "
            "artifact for downstream agents. When routed upstream artifacts are present, "
            "consume them as this node's declared dependencies instead of silently redoing "
            "their responsibilities, unless the contract explicitly asks for verification. "
            "Preserve the task's original relation, qualifiers, comparison criterion, and "
            "answer type. Ground each semantic candidate in the relevant source passage or "
            "span; when the contract asks for verification, independently reconstruct that "
            "evidence and report agreement, conflict, or insufficiency rather than merely "
            "restating the upstream artifact. Do not present a task-level final answer and "
            "do not use <answer> tags."
        )
    healthbench_messages: tuple[dict[str, str], ...] | None = None
    if not request.is_format_agent:
        try:
            healthbench_messages = parse_model_visible_conversation(
                request.problem
            )
        except ValueError:
            # Every non-HealthBench task retains the existing text transport.
            healthbench_messages = None
    if (
        healthbench_messages is not None
        and execution_mode == "react"
        and request.is_output_agent
        and request.model.metadata.get("healthbench_execution_protocol")
        in {
            _HEALTHBENCH_EXECUTION_PROTOCOL_V3_METADATA,
            _HEALTHBENCH_EXECUTION_PROTOCOL_V4_METADATA,
            _HEALTHBENCH_EXECUTION_PROTOCOL_V5_METADATA,
        }
    ):
        protocol += _HEALTHBENCH_OUTPUT_REACT_PROTOCOL_V3
    if request.is_format_agent:
        # FlowSteer's Format Operator normally receives the problem and the
        # computed solution under its fixed extraction prompt.  Do not inject
        # the graph-authored free-text contract into the terminal invocation:
        # it is retained in the Canvas/trajectory receipt, but may contain an
        # explanatory target sentence that conflicts with answer-span
        # extraction.  This is the minimal free-AgentGraph adaptation of the
        # upstream Operator boundary.
        system = (
            f"Agent ID: {request.agent.id}\nRole: Format\n\n"
            f"Execution protocol:\n{protocol}"
        )
    else:
        # Keep the graph-authored free-text contract, then append the execution
        # boundary so a contract cannot accidentally reassign final-answer ownership.
        system = (
            f"Agent ID: {request.agent.id}\nContract:\n{request.agent.contract}\n\n"
            f"Execution protocol (takes precedence):\n{protocol}"
        )
        if _uses_producer_context(request.artifact_communication_profile):
            system += (
                "\n\nA routed source contract is provenance describing why its "
                "artifact was produced. Follow this Agent's own contract; use "
                "source provenance only to interpret and validate the artifact."
            )
    if healthbench_messages is not None:
        system += (
            "\n\nHealthBench Professional execution protocol "
            "(takes precedence over an Agent contract):\n"
            + _healthbench_execution_protocol(request)
        )
        if request.artifact_communication_profile == ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3:
            system += _HEALTHBENCH_V3_EXECUTION_SUPPLEMENT
    healthbench_projection_state: dict[str, Any] = {
        "remaining": _HEALTHBENCH_V3_PROMPT_CHARS,
        "sources": set(), "envelopes": set(),
    }
    upstream_text = _format_upstream(
        request.upstream,
        request.communication_condition,
        include_dependency=not request.is_format_agent,
        project_artifact_read_receipts=(
            semantic_lineage
            and semantic_role in {"reasoner", "verifier"}
        ),
        project_healthbench_structured_evidence=(
            healthbench_messages is not None
            and request.artifact_communication_profile
            in {
                ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2,
                ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3,
            }
        ),
        artifact_communication_profile=(
            request.artifact_communication_profile
        ),
        healthbench_projection_state=healthbench_projection_state,
    )
    if request.is_format_agent:
        # Directly reuse FlowSteer's Format Operator prompt and its clean
        # ``problem + computed solution`` call boundary.  The AgentGraph
        # adaptation changes only the terminal wrapper; typed communication
        # envelopes remain intact in trajectory receipts but do not burden the
        # extraction-only model input.
        solution = (
            _visible_message_content(
                request.upstream[0].artifact,
                request.communication_condition,
            )
            if len(request.upstream) == 1
            else ""
        )
        if semantic_lineage:
            common = FORMAT_PROMPT.format(
                problem_description=(
                    "the formatting-only transfer of one explicit Candidate answer"
                ),
                solution=solution,
            ) + (
                (
                    "\nFor this AgentGraph terminal protocol, the rules below take "
                    "precedence over every normalization or transformation example "
                    "above. The solution must be a Verifier artifact whose "
                    "`Verification status:` is exactly `supported`. "
                )
                if hotpot_semantic
                else (
                    "\nFor this role-conditional QA terminal protocol, the rules "
                    "below take precedence over every normalization or "
                    "transformation example above. The solution must contain one "
                    "explicit routed semantic candidate. "
                )
            ) + (
                (
                    "Copy character-for-character only the value following its "
                    "single `Candidate answer:` label; never select another name "
                    "or value, and never change an alias, abbreviation, unit, "
                    "date, spelling, or symbolic form. Enclose that exact copied "
                    "value in exactly one <answer>...</answer> wrapper, with no "
                    "explanation. If the supported status or exactly one Candidate "
                    "answer is absent, return exactly <answer></answer>."
                )
                if hotpot_semantic
                else (
                    "Copy character-for-character only its candidate value; never "
                    "select another name or value, and never change an alias, "
                    "abbreviation, unit, date, spelling, or symbolic form. Enclose "
                    "that exact copied value in exactly one <answer>...</answer> "
                    "wrapper, with no explanation. If one explicit candidate is "
                    "absent, return exactly <answer></answer>."
                )
            )
        else:
            common = FORMAT_PROMPT.format(
                problem_description=request.problem,
                solution=solution,
            ) + (
                "\nFor this AgentGraph terminal protocol, enclose only that extracted "
                "answer value in exactly one <answer>...</answer> wrapper. Do not put a "
                "sentence or explanation inside the wrapper. If the computed solution "
                "does not contain one answer candidate, return exactly <answer></answer>."
            )
    elif healthbench_messages is not None:
        # SkillEval's HealthBench task source preserves the native multi-turn
        # roles. Keep those messages intact, then append only the current
        # AgentGraph execution context. Rubrics and reference responses never
        # enter ``request.problem`` and therefore cannot cross this boundary.
        common = "External upstream messages:\n" + upstream_text
    else:
        common = (
            f"Task:\n{request.problem}\n\n"
            "External upstream messages:\n"
            f"{upstream_text}"
        )
    if request.phase is ExecutionPhase.SINGLE:
        phase = "Produce your response now."
    elif request.phase is ExecutionPhase.DRAFT:
        phase = (
            "This is the independent draft phase of a finite bidirectional exchange. "
            "Produce a draft without assuming access to the peer's current draft."
        )
    elif request.phase is ExecutionPhase.REVISION:
        if request.own_draft is None or request.peer_draft is None:
            raise OpenAICompatibleGatewayError("revision request is missing immutable drafts")
        phase_prefix = (
            "This is the revision phase. Revise your own draft after reading the "
            "peer's previous-phase draft. You cannot observe the peer's current "
            f"revision.\n\nYour draft:\n{request.own_draft}\n\n"
            "Peer artifact envelope:\n"
        )
        if _uses_producer_context(request.artifact_communication_profile):
            phase = phase_prefix + _format_upstream(
                (request.peer_draft,),
                request.communication_condition,
                project_healthbench_structured_evidence=(
                    healthbench_messages is not None
                    and request.artifact_communication_profile
                    in {
                        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V2,
                        ARTIFACT_COMMUNICATION_PRODUCER_CONTEXT_STRUCTURED_EVIDENCE_V3,
                    }
                ),
                artifact_communication_profile=(
                    request.artifact_communication_profile
                ),
                healthbench_projection_state=healthbench_projection_state,
            )
        else:
            phase = (
                phase_prefix
                + f"source_agent: {request.peer_draft.source_agent_id}\n"
                f"target_agent: {request.peer_draft.target_agent_id}\n"
                f"message_type: {request.peer_draft.message_type}\n"
                f"artifact_type: {request.peer_draft.artifact_type}\n"
                f"graph_revision: {request.peer_draft.graph_revision}\n"
                + (
                    "environment_revision: "
                    f"{request.peer_draft.environment_revision}\n"
                    if request.peer_draft.environment_revision is not None
                    else ""
                )
                + (
                    "tool_receipts: "
                    + json.dumps(
                        [
                            dict(receipt)
                            for receipt in request.peer_draft.tool_receipts
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                    if request.peer_draft.tool_receipts
                    else ""
                )
                + "artifact:\n"
                + _visible_message_content(
                    request.peer_draft.content,
                    request.communication_condition,
                )
            )
    else:  # pragma: no cover - enum exhaustiveness guard
        raise OpenAICompatibleGatewayError(f"unsupported execution phase: {request.phase}")
    if healthbench_messages is not None:
        return [
            {"role": "system", "content": system},
            *[dict(message) for message in healthbench_messages],
            {
                "role": "user",
                "content": "AgentGraph execution context:\n" + common + "\n\n" + phase,
            },
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": common + "\n\n" + phase},
    ]


class OpenAICompatibleGateway:
    """A small dependency-free `/chat/completions` client.

    Provider records carry only the *name* of an API-key environment variable.
    The resolved key stays in memory and is never included in errors/metadata.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        default_temperature: float = 0.0,
        default_top_p: float = 1.0,
        default_max_tokens: int = 4096,
        default_seed: Optional[int] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.default_max_tokens = int(default_max_tokens)
        self.default_seed = default_seed
        if self.default_temperature < 0:
            raise ValueError("default_temperature must be non-negative")
        if not 0 < self.default_top_p <= 1:
            raise ValueError("default_top_p must be in (0, 1]")
        if self.default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be positive")
        if self.default_seed is not None and (
            isinstance(self.default_seed, bool)
            or not isinstance(self.default_seed, int)
            or not 0 <= self.default_seed < 2**64
        ):
            raise ValueError(
                "default_seed must be an unsigned 64-bit integer or None"
            )

    def request_payload(self, request: AgentRequest) -> Dict[str, Any]:
        metadata = request.model.metadata
        temperature = _number(metadata, "temperature", self.default_temperature)
        top_p = _number(metadata, "top_p", self.default_top_p)
        if temperature < 0:
            raise OpenAICompatibleGatewayError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise OpenAICompatibleGatewayError("top_p must be in (0, 1]")
        visible_max_tokens = _integer(
            metadata,
            "max_tokens",
            self.default_max_tokens,
        )
        payload: Dict[str, Any] = {
            "model": request.model.model_name,
            "messages": build_agent_messages(request),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": visible_max_tokens,
        }
        generation_seed = _non_negative_integer(
            metadata,
            "generation_seed",
            self.default_seed,
        )
        if generation_seed is not None:
            # DIRECT_REUSE: SkillFlow derives one scientific seed for each
            # bounded rollout step. A request-level seed therefore takes
            # precedence over the gateway's legacy fixed-run default.
            payload["seed"] = (
                _sglang_backend_sampling_seed(generation_seed)
                if supports_local_sglang_top_k(request)
                else generation_seed
            )
        if supports_local_sglang_top_k(request):
            raw_top_k = metadata.get("top_k")
            if raw_top_k is not None:
                try:
                    top_k = int(raw_top_k)
                except (TypeError, ValueError) as exc:
                    raise OpenAICompatibleGatewayError(
                        "model metadata top_k must be an integer"
                    ) from exc
                if top_k != -1 and top_k <= 0:
                    raise OpenAICompatibleGatewayError(
                        "model metadata top_k must be -1 or a positive integer"
                    )
                payload["top_k"] = top_k
        if supports_local_sglang_repetition_penalty(request):
            raw_repetition_penalty = metadata.get("repetition_penalty")
            if raw_repetition_penalty is not None:
                repetition_penalty = _number(
                    metadata,
                    "repetition_penalty",
                    1.0,
                )
                if not 0 < repetition_penalty <= 2:
                    raise OpenAICompatibleGatewayError(
                        "model metadata repetition_penalty must be in (0, 2]"
                    )
                payload["repetition_penalty"] = repetition_penalty
        thinking = metadata.get("chat_template_enable_thinking")
        if thinking is not None:
            normalized = thinking.strip().lower()
            if normalized not in {"true", "false"}:
                raise OpenAICompatibleGatewayError(
                    "model metadata chat_template_enable_thinking must be true or false"
                )
            # SGLang's Qwen3.5 OpenAI surface accepts the Hugging Face chat
            # template toggle under chat_template_kwargs.  This keeps Agent
            # answers in message.content instead of an empty content field
            # accompanied only by reasoning_content.
            payload["chat_template_kwargs"] = {
                "enable_thinking": normalized == "true"
            }
            raw_thinking_budget = metadata.get("thinking_budget")
            if raw_thinking_budget is not None:
                if normalized != "true":
                    raise OpenAICompatibleGatewayError(
                        "model metadata thinking_budget requires "
                        "chat_template_enable_thinking=true"
                    )
                # DIRECT_REUSE + NECESSARY_ADAPTATION: SkillFlow reserves a
                # separate Qwen thinking budget and sends the provider the
                # visible response allowance plus that hidden budget.  The
                # OpenAI-compatible SGLang surface exposes only max_tokens, so
                # retain the two components in the receipt while sending their
                # sum to the provider.
                thinking_budget = _integer(metadata, "thinking_budget", 1)
                payload["max_tokens"] = visible_max_tokens + thinking_budget
        response_schema_text = metadata.get("response_json_schema")
        if response_schema_text is not None:
            if not isinstance(response_schema_text, str) or not response_schema_text.strip():
                raise OpenAICompatibleGatewayError(
                    "model metadata response_json_schema must be non-empty JSON text"
                )
            try:
                response_schema = json.loads(response_schema_text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OpenAICompatibleGatewayError(
                    "model metadata response_json_schema is not valid JSON"
                ) from exc
            if not isinstance(response_schema, dict):
                raise OpenAICompatibleGatewayError(
                    "model metadata response_json_schema must decode to an object"
                )
            # DIRECT_REUSE: SkillFlow runtime/openai_provider.py sends the
            # request-scoped ModelRequest.response_schema through the standard
            # OpenAI structured-output boundary.
            payload["response_format"] = {
                "json_schema": {
                    "name": "skillev_action",
                    "schema": response_schema,
                    "strict": True,
                },
                "type": "json_schema",
            }
        return payload

    async def generate(self, request: AgentRequest) -> AgentResponse:
        endpoint = request.provider.endpoint
        if not endpoint:
            raise OpenAICompatibleGatewayError(
                f"provider {request.provider.provider_id!r} has no endpoint"
            )
        api_key = "EMPTY"
        if request.provider.api_key_env:
            api_key = os.getenv(request.provider.api_key_env, "")
            if not api_key:
                raise OpenAICompatibleGatewayError(
                    f"missing provider credential environment variable: "
                    f"{request.provider.api_key_env}"
                )
        payload = self.request_payload(request)
        scientific_generation_seed = _non_negative_integer(
            request.model.metadata,
            "generation_seed",
            self.default_seed,
        )
        requested_sampling = _requested_sampling(payload)
        if "thinking_budget" in request.model.metadata:
            requested_sampling.update(
                {
                    "visible_max_tokens": _integer(
                        request.model.metadata,
                        "max_tokens",
                        self.default_max_tokens,
                    ),
                    "thinking_budget": _integer(
                        request.model.metadata,
                        "thinking_budget",
                        1,
                    ),
                }
            )
        if (
            scientific_generation_seed is not None
            and payload.get("seed") != scientific_generation_seed
        ):
            requested_sampling["seed"] = scientific_generation_seed
            requested_sampling["backend_seed"] = payload.get("seed")
        url = endpoint.rstrip("/") + "/chat/completions"

        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(self._post_json, url, api_key, payload)
                parsed = self._parse_response(response, request)
                metadata = dict(parsed.metadata)
                metadata.update(
                    {
                        "latency_ms": max(
                            (time.monotonic() - started_at) * 1000.0,
                            0.0,
                        ),
                        "attempt_count": attempt + 1,
                        "generation_seed": scientific_generation_seed,
                        "backend_sampling_seed": payload.get("seed"),
                        "requested_sampling": requested_sampling,
                        "request_status": "completed",
                    }
                )
                return AgentResponse(parsed.text, metadata)
            except HTTPError as exc:
                last_error = exc
                retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(min(2.0**attempt, 4.0))

        if isinstance(last_error, HTTPError):
            detail = f"HTTP {last_error.code}"
        else:
            detail = type(last_error).__name__ if last_error is not None else "unknown error"
        error = OpenAICompatibleGatewayError(
            f"provider request failed for {request.provider.provider_id}: {detail}"
        )
        # This is the exact decoding projection from the already-materialized
        # provider payload.  It states what was requested, not what a failed
        # server necessarily applied.
        error.requested_sampling = requested_sampling
        error.request_status = "failed"
        error.provider_id = request.provider.provider_id
        error.model_id = request.model.model_id
        error.http_status = (
            last_error.code if isinstance(last_error, HTTPError) else None
        )
        raise error from last_error

    def _post_json(self, url: str, api_key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-AgentGraph/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise OpenAICompatibleGatewayError("provider returned a non-object response")
        return value

    @staticmethod
    def _parse_response(response: Mapping[str, Any], request: AgentRequest) -> AgentResponse:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenAICompatibleGatewayError("provider response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OpenAICompatibleGatewayError("provider response has no text message content")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        completion_token_details = (
            usage.get("completion_tokens_details")
            if isinstance(usage.get("completion_tokens_details"), Mapping)
            else {}
        )
        reasoning_tokens = completion_token_details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int):
            reasoning_tokens = 0
        metadata = {
            "provider_id": request.provider.provider_id,
            "model_id": request.model.model_id,
            "provider_model": response.get("model", request.model.model_name),
            "finish_reason": choices[0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "provider_request_id": response.get("id"),
            # Keep the hidden reasoning body out of Agent artifacts while
            # retaining a public receipt that the provider returned one.
            "reasoning_content_present": isinstance(
                message.get("reasoning_content"), str
            )
            and bool(message.get("reasoning_content")),
            "reasoning_content_characters": (
                len(message["reasoning_content"])
                if isinstance(message.get("reasoning_content"), str)
                else 0
            ),
            "reasoning_tokens": reasoning_tokens,
        }
        return AgentResponse(text=message["content"], metadata=metadata)


__all__ = [
    "OpenAICompatibleGateway",
    "OpenAICompatibleGatewayError",
    "build_agent_messages",
]
