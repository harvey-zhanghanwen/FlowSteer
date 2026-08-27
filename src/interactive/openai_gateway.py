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
    AgentRequest,
    AgentResponse,
    CommunicationCondition,
    ExecutionPhase,
    UpstreamMessage,
)


class OpenAICompatibleGatewayError(RuntimeError):
    """Provider failure with public routing metadata for typed recovery."""

    provider_id: str | None = None
    model_id: str | None = None
    http_status: int | None = None
    request_status: str | None = None


MASKED_UPSTREAM_CONTENT = "[UPSTREAM CONTENT MASKED FOR COMMUNICATION DIAGNOSTIC]"


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


def _requested_sampling(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Project only the decoding fields placed on the provider request."""

    return {
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "max_tokens": payload.get("max_tokens"),
        "seed": payload.get("seed"),
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
    """Return public read-record IDs and evidence spans cited by one artifact."""

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
        if field_name in {"passage_id", "memory_id"}:
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
        passage = value.get("memory", value.get("passage"))
        if not isinstance(passage, Mapping):
            continue
        passage_id = value.get(
            "memory_id",
            value.get(
                "passage_id",
                passage.get("memory_id", passage.get("passage_id")),
            ),
        )
        if not isinstance(passage_id, str):
            arguments = request.get("arguments")
            if isinstance(arguments, Mapping):
                passage_id = arguments.get(
                    "memory_id", arguments.get("passage_id")
                )
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


def _format_upstream(
    messages: Sequence[UpstreamMessage],
    condition: CommunicationCondition,
    *,
    include_dependency: bool = True,
    project_artifact_read_receipts: bool = False,
) -> str:
    if not messages:
        return "(none)"
    rendered = []
    for item in messages:
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
        if include_dependency and item.request_or_dependency is not None:
            envelope.append(
                f"request_or_dependency: {item.request_or_dependency}"
            )
        visible_tool_receipts = (
            _successful_read_receipt_projection(
                item.tool_receipts,
                artifact=item.artifact,
            )
            if project_artifact_read_receipts
            else tuple(dict(receipt) for receipt in item.tool_receipts)
        )
        if project_artifact_read_receipts and item.tool_receipts:
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
            "for one action, return exactly one listed executable action with no explanation."
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
            "restating the upstream artifact. "
            "Do not present a task-level final answer and "
            "do not use <answer> tags."
        )
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
    upstream_text = _format_upstream(
        request.upstream,
        request.communication_condition,
        include_dependency=not request.is_format_agent,
        project_artifact_read_receipts=(
            semantic_lineage
            and semantic_role in {"reasoner", "verifier"}
        ),
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
        phase = (
            "This is the revision phase. Revise your own draft after reading the peer's "
            "previous-phase draft. You cannot observe the peer's current revision.\n\n"
            f"Your draft:\n{request.own_draft}\n\n"
            "Peer artifact envelope:\n"
            f"source_agent: {request.peer_draft.source_agent_id}\n"
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
                    [dict(receipt) for receipt in request.peer_draft.tool_receipts],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                if request.peer_draft.tool_receipts
                else ""
            )
            + "artifact:\n"
            f"{_visible_message_content(request.peer_draft.content, request.communication_condition)}"
        )
    else:  # pragma: no cover - enum exhaustiveness guard
        raise OpenAICompatibleGatewayError(f"unsupported execution phase: {request.phase}")
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
        payload: Dict[str, Any] = {
            "model": request.model.model_name,
            "messages": build_agent_messages(request),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": _integer(metadata, "max_tokens", self.default_max_tokens),
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
        metadata = {
            "provider_id": request.provider.provider_id,
            "model_id": request.model.model_id,
            "provider_model": response.get("model", request.model.model_name),
            "finish_reason": choices[0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "provider_request_id": response.get("id"),
        }
        return AgentResponse(text=message["content"], metadata=metadata)


__all__ = [
    "OpenAICompatibleGateway",
    "OpenAICompatibleGatewayError",
    "build_agent_messages",
]
