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
    pass


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


def _visible_message_content(
    content: str,
    condition: CommunicationCondition,
) -> str:
    if condition is CommunicationCondition.UPSTREAM_MASKED:
        return MASKED_UPSTREAM_CONTENT
    return content


def _format_upstream(
    messages: Sequence[UpstreamMessage],
    condition: CommunicationCondition,
    *,
    include_dependency: bool = True,
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
        if item.tool_receipts:
            envelope.append(
                "tool_receipts: "
                + json.dumps(
                    [dict(receipt) for receipt in item.tool_receipts],
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


_HOTPOTQA_SEMANTIC_PROTOCOLS = {
    "hotpotqa_verified_answer_slot_v1",
    "hotpotqa_semantic_lineage_v2",
    "hotpotqa_role_conditional_capabilities_v1",
}


def _single_labeled_value(artifact: str, label: str) -> Optional[str]:
    """Return one non-empty line value from a structured semantic artifact."""

    prefix = f"{label}:"
    values = [
        line.strip()[len(prefix) :].strip()
        for line in artifact.splitlines()
        if line.strip().startswith(prefix)
    ]
    if len(values) != 1 or not values[0]:
        return None
    return values[0]


def _hotpotqa_supported_verifier_candidate(artifact: str) -> Optional[str]:
    """Return a supported candidate, ``None`` for auxiliary, or ``""`` if invalid."""

    try:
        fields = json.loads(artifact)
    except (TypeError, ValueError, json.JSONDecodeError):
        fields = None
    verifier_keys = {
        "verification_status",
        "evidence_supported",
        "entity_attribute_binding_correct",
        "multi_hop_complete",
        "scope_preserved",
        "answer_type_cardinality_correct",
        "minimal_answer_surface",
        "alias_binding_correct",
    }
    if isinstance(fields, Mapping) and verifier_keys.intersection(fields):
        candidate = fields.get("candidate_answer")
        status = fields.get("verification_status")
        checks = (
            "evidence_supported",
            "entity_attribute_binding_correct",
            "multi_hop_complete",
            "scope_preserved",
            "answer_type_cardinality_correct",
            "minimal_answer_surface",
            "alias_binding_correct",
        )
        if (
            not isinstance(candidate, str)
            or not candidate
            or candidate != candidate.strip()
            or "\n" in candidate
            or not isinstance(status, str)
            or status.strip().casefold() != "supported"
            or any(fields.get(check) is not True for check in checks)
        ):
            return ""
        return candidate

    candidate = _single_labeled_value(artifact, "Candidate answer")
    status = _single_labeled_value(artifact, "Verification status")
    if status is None:
        return None
    if candidate is None or status is None or status.casefold() != "supported":
        return ""
    return candidate


def _hotpotqa_supported_consensus(
    messages: Sequence[UpstreamMessage],
    condition: CommunicationCondition,
    *,
    allow_role_conditional_sources: bool = False,
) -> str:
    """Build a formatting-only transfer from agreeing Verifier artifacts.

    Auxiliary evidence and Reasoner artifacts may share a flexible fan-in with
    one or more Verifier artifacts. Only Verifier-shaped artifacts participate
    in consensus; every such artifact must be supported and must copy the same
    already-determined semantic candidate.
    """

    verifier_candidates: list[str] = []
    for message in messages:
        artifact = _visible_message_content(message.artifact, condition)
        candidate = _hotpotqa_supported_verifier_candidate(artifact)
        if candidate is None and allow_role_conditional_sources:
            try:
                fields = json.loads(artifact)
            except (TypeError, ValueError, json.JSONDecodeError):
                fields = None
            if isinstance(fields, Mapping) and {
                "question_scope",
                "answer_slot",
                "evidence_propositions",
                "multi_hop_chain",
                "candidate_answer",
                "evidence",
            } <= set(fields):
                raw_candidate = fields.get("candidate_answer")
                candidate = (
                    raw_candidate
                    if isinstance(raw_candidate, str)
                    and raw_candidate
                    and raw_candidate == raw_candidate.strip()
                    and "\n" not in raw_candidate
                    else ""
                )
            elif _single_labeled_value(artifact, "Verification status") is None:
                candidate = _single_labeled_value(artifact, "Candidate answer")
        if candidate is None:
            continue
        if not candidate:
            return ""
        verifier_candidates.append(candidate)
    if not verifier_candidates:
        return ""
    candidate = verifier_candidates[0]
    if any(other_candidate != candidate for other_candidate in verifier_candidates):
        return ""
    return f"Candidate answer: {candidate}\nVerification status: supported"


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
    "answer type and answer cardinality. First align every database or retrieved "
    "fact's sentence-level syntactic predicate-argument structure (grammatical "
    "subject, predicate, complements, modifiers, and comparison operands) with "
    "its semantic proposition. Represent that proposition with "
    "subject/entity, predicate/relation, object or attribute value, and qualifiers; "
    "preserve the sentence's asserted semantic roles instead of placing the desired "
    "candidate into an unrelated field. For a comparison fact, the compared entity "
    "is normally the proposition subject and its date, number, or other compared "
    "attribute is the object_or_attribute_value. Then align that proposition to the "
    "answer slot actually requested. Apply the "
    "question's wh-word answer-type constraint: a Which-comparison returns the "
    "compared entity rather than the comparison value, and a who-question returns "
    "the evidence-supported answer-bearing entity, which may be a person or "
    "organization, rather than a possessive attribute phrase. You alone determine "
    "the semantic candidate and own the final semantic answer; no downstream "
    "Formatter may reselect it. "
    "Return exactly the six structured fields "
    "question_scope, answer_slot, evidence_propositions, multi_hop_chain, "
    "candidate_answer, and evidence. Copy question_scope exactly from the original "
    "question. answer_slot contains exactly answer_type, answer_cardinality, qualifiers, "
    "proposition_index, and answer_field. proposition_index selects one item in "
    "evidence_propositions and answer_field selects either its subject or its "
    "object_or_attribute_value; candidate_answer must equal that selected value. "
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
    "value as the candidate. If both comparison sides have unexpectedly equal "
    "values, do not accept a tie until the original question scope, both "
    "entity-attribute bindings, explicit evidence, and any upstream contract "
    "narrowing have all been rechecked. "
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

_HOTPOTQA_ROLE_CONDITIONAL_VERIFIER_PROTOCOL = (
    _HOTPOTQA_VERIFIER_PROTOCOL.replace(
        "Inspect the routed Reasoner candidate",
        "Inspect the routed semantic candidate",
        1,
    ).replace(
        "Copy the Reasoner's Candidate answer character-for-character.",
        "Copy the routed Candidate answer character-for-character.",
        1,
    )
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
    "object_or_attribute_value; candidate_answer must equal that selected value. "
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
    hotpot_semantic = request.semantic_protocol in _HOTPOTQA_SEMANTIC_PROTOCOLS
    flexible_hotpot_semantic = (
        request.semantic_protocol
        in {
            "hotpotqa_semantic_lineage_v2",
            "hotpotqa_role_conditional_capabilities_v1",
        }
    )
    role_conditional_hotpot = (
        request.semantic_protocol
        == "hotpotqa_role_conditional_capabilities_v1"
    )
    unified_qa_semantic = request.semantic_protocol == "qa_verified_answer_lineage_v2"
    semantic_lineage = hotpot_semantic or unified_qa_semantic
    exact_answer_output = (
        request.is_output_agent and request.require_exact_answer_tag
    )
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
            "put terminal syntax outside the StructuredAction JSON object."
        )
        if semantic_lineage and semantic_role == "reasoner":
            protocol += (
                " ReAct is only this node's execution schedule, not its role. "
                "Never place semantic-answer fields in search/read arguments. "
                "Only when the assigned contract marks completion currently "
                "admissible, put the structured semantic Reasoner artifact defined "
                "there in arguments.value. "
                + (
                    _HOTPOTQA_REASONER_PROTOCOL
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
        if flexible_hotpot_semantic:
            protocol = (
                "You are the terminal FlowSteer Format Operator. The semantic answer "
                + (
                    "has already been determined in an explicit routed semantic "
                    "candidate artifact. "
                    if role_conditional_hotpot
                    else (
                        "has already been determined by a Reasoner and supported by "
                        "one or more agreeing Verifier artifacts routed through the "
                        "current graph. "
                    )
                )
                + "You will receive only the candidate transfer, never the original "
                "question. Serialize that candidate character-for-character; do not "
                "solve, reason, verify, canonicalize, or reselect it."
            )
        else:
            protocol = (
                "You are the terminal FlowSteer Format Operator. The solution has "
                "already been computed and passed by a Verifier in exactly one routed "
                "upstream artifact. You will not receive the original question. Follow "
                "the copying instructions in the user message; do not solve, verify, "
                "or extend the answer; do not canonicalize or reselect it."
            )
    elif semantic_lineage and semantic_role == "reasoner":
        protocol = (
            _HOTPOTQA_REASONER_PROTOCOL
            if hotpot_semantic
            else _QA_REASONER_PROTOCOL
        ) + " Do not use <answer> tags."
    elif semantic_lineage and semantic_role == "verifier":
        protocol = (
            (
                _HOTPOTQA_ROLE_CONDITIONAL_VERIFIER_PROTOCOL
                if role_conditional_hotpot
                else _HOTPOTQA_VERIFIER_PROTOCOL
            )
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
        if role_conditional_hotpot:
            protocol += (
                " If routed inputs contain one or more explicit semantic-candidate "
                "artifacts, copy their agreeing candidate character-for-character "
                "into the terminal answer. Never reselect, canonicalize, or rewrite "
                "that candidate. If routed candidates disagree, do not choose among "
                "them; the upstream semantic conflict must be repaired before this "
                "completion is admissible."
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
    if execution_mode in {"react", "coding"}:
        if exact_answer_output:
            protocol += (
                " When completion is admissible, set only complete.arguments.value "
                "to exactly one non-empty <answer>...</answer> wrapper and put no "
                "text outside that wrapper value. This is terminal serialization "
                "only; it does not assign an Agent role or workflow topology."
            )
        else:
            protocol += " Do not emit <answer> tags in this internal action."
    elif exact_answer_output:
        protocol += (
            " Serialize the final artifact as exactly one non-empty "
            "<answer>...</answer> wrapper with no text outside it. This is only "
            "the terminal output syntax; preserve the answer determined from the "
            "task and routed artifacts, and do not infer a workflow role from it."
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
    )
    if request.is_format_agent:
        # Directly reuse FlowSteer's Format Operator prompt and its clean
        # ``problem + computed solution`` call boundary.  The AgentGraph
        # adaptation changes only the terminal wrapper; typed communication
        # envelopes remain intact in trajectory receipts but do not burden the
        # extraction-only model input.
        if flexible_hotpot_semantic:
            solution = _hotpotqa_supported_consensus(
                request.upstream,
                request.communication_condition,
                allow_role_conditional_sources=(
                    request.semantic_protocol
                    == "hotpotqa_role_conditional_capabilities_v1"
                ),
            )
        else:
            solution = (
                _visible_message_content(
                    request.upstream[0].artifact,
                    request.communication_condition,
                )
                if len(request.upstream) == 1
                else ""
            )
        if semantic_lineage:
            semantic_transfer_requirement = (
                "must contain exactly one explicit routed `Candidate answer:` value. "
                if role_conditional_hotpot
                else (
                    "must be a Verifier artifact whose `Verification status:` is "
                    "exactly `supported`. "
                )
            )
            common = FORMAT_PROMPT.format(
                problem_description=(
                    "the formatting-only transfer of one verified Candidate answer"
                ),
                solution=solution,
            ) + (
                "\nFor this AgentGraph terminal protocol, the rules below take precedence "
                "over every normalization or transformation example above. The solution "
                + semantic_transfer_requirement
                + "Copy character-for-character only the value following its "
                "single `Candidate answer:` label; never select another name or value, "
                "and never change an alias, abbreviation, "
                "unit, date, spelling, or symbolic form. Enclose that exact copied value "
                "in exactly one <answer>...</answer> wrapper, with no explanation. If the "
                "required semantic candidate or exactly one Candidate answer is absent, return "
                "exactly <answer></answer>."
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
            or self.default_seed < 0
        ):
            raise ValueError("default_seed must be a non-negative integer or None")

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
        if self.default_seed is not None:
            # SkillFlow's OpenAI-compatible provider sends the configured seed
            # to the serving boundary.  Keep the same fixed-run contract here.
            payload["seed"] = self.default_seed
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
                        "generation_seed": payload.get("seed"),
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
        raise OpenAICompatibleGatewayError(
            f"provider request failed for {request.provider.provider_id}: {detail}"
        ) from last_error

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
