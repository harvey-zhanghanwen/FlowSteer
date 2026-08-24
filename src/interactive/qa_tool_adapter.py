"""SkillFlow public-retrieval backends for the unified ToolRegistry.

This module is a thin adapter over SkillFlow
``skillev.benchmarks.retrieval.RetrievalIndex``.  It preserves the upstream
read-only ``open -> search/read`` call chain and only projects public corpus
observations.  Benchmark task IDs, accepted answers, and evaluator state are
not accepted by either tool schema.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Event
from typing import Callable, Mapping, Optional, Protocol, Sequence
import unicodedata

from .agent_runtime import AgentGateway, AgentRequest, GatewayResponse
from .react_execution import ReactExecutionError, ToolReactExecutionAdapter
from .scientific_sampling import ScientificSamplingCoordinate
from .task_dataset import (
    hotpotqa_answer_cardinality_constraint,
    hotpotqa_answer_type_constraint,
    hotpotqa_question_scope,
    qa_answer_argument_constraint,
    qa_answer_cardinality_constraint,
    qa_answer_type_constraint,
    qa_question_scope,
)
from .qa_retrieval import (
    DEFAULT_QA_RETRIEVAL_INDEX,
    DEFAULT_SKILLFLOW_SOURCE,
    SkillFlowRetrievalError,
    _load_retrieval_index_class,
    _load_retrieval_module,
)
from .tool_runtime import (
    ActionKind,
    StructuredAction,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


# DIRECT_REUSE: SkillFlow's QARetrievalEnvironment publishes one resource ID
# with the two executable action names ``search`` and ``read``.  Keeping them
# under one capability prevents the Canvas from assigning ``read`` without the
# search action that produces its opaque passage_id.
QA_RETRIEVAL_TOOL_ID = "qa-retrieval"
QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL = "qa_verified_answer_lineage_v2"
DEFAULT_QA_DATASET_SCOPE = ("hotpotqa", "triviaqa")
_PROVIDED_PASSAGE = re.compile(
    r"^\[(?P<title>[^\]]+)\]\s*(?P<text>.+)$",
    flags=re.DOTALL,
)

SKILLFLOW_MULTI_HOP_QA_GUIDANCE = (
    "Chain evidence across passages to answer multi-hop questions. "
    "Search for specific entity names (not the full question). "
    "If a search has no match or repeats prior evidence, pivot with synonyms. "
    "Complete with a brief name, date, number, or short phrase."
)
SKILLFLOW_FACTUAL_QA_GUIDANCE = (
    "Extract the answer from retrieved passages and do not guess from memory. "
    "Complete with a concise name, place, number, or short phrase."
)
HOTPOTQA_VERIFIED_ANSWER_SLOT_GUIDANCE = (
    "Treat ReAct only as the execution schedule Thought -> Action(tool) -> "
    "Observation -> Thought -> Final, never as an Agent role. Preserve the "
    "question's exact scope and answer slot. Resolve entity aliases and coreference "
    "from the supplied passages before composing the retrieval query, and retain "
    "that entity binding through every hop. Apply the original wh-word answer type: "
    "a Which-comparison returns the compared entity, not the numeric/date comparison "
    "value; a who-question returns the evidence-supported answer-bearing entity "
    "(which may be a person or organization), not a possessive attribute phrase. "
    "Represent retrieved facts as "
    "subject/entity, predicate/relation, object or attribute value, and qualifiers. "
    "Preserve the sentence's asserted semantic roles instead of placing the desired "
    "candidate into an unrelated proposition field. In a comparison proposition, "
    "the compared entity is normally the subject and its compared date, number, or "
    "attribute is object_or_attribute_value. "
    "Bind candidate_answer to exactly one evidence_propositions item through "
    "answer_slot.proposition_index and answer_field; keep the entity-to-attribute "
    "binding explicit and show every bridge in the multi-hop chain. Return one "
    "minimal but complete evidence-aligned referential surface when "
    "answer_cardinality is single; do not return an alias list, appositive gloss, "
    "or the question's answer-type head noun. For a who-question whose evidence "
    "expresses the requested person through a possessive construction, exclude the "
    "possessive marker and possessed attribute, but retain the complete possessor "
    "entity mention immediately before the marker, including any title, honorific, "
    "or name suffix present in the evidence. If compared values "
    "are unexpectedly equal, recheck scope, "
    "both bindings, retrieved passages, and contract narrowing before calling a tie."
)
QA_VERIFIED_ANSWER_LINEAGE_GUIDANCE = (
    "Treat ReAct only as the execution schedule Thought -> Action(tool) -> "
    "Observation -> Thought -> Final, never as an Agent role. Preserve the "
    "question's exact semantic scope and answer slot. Bind the target entity, "
    "requested relation, and every answer-bearing proposition to successful "
    "qa-retrieval read receipts. Keep spelling variants, aliases, and canonical "
    "names explicit in the propositions; do not guess an entity identity or "
    "relation that is absent from retrieved evidence. Bind every explicit "
    "ordinal or superlative scope modifier to the same evidence proposition as "
    "the target entity, requested relation, and answer-bearing argument; a "
    "local or subtype first does not establish an entity-global first. Return a concise semantic "
    "answer and leave surface-only output formatting to the Format Agent."
)

_HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX = (
    "hotpotqa_semantic_artifact_invalid:"
)
_HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX = (
    "hotpotqa_semantic_evidence_provenance_invalid:"
)
_QA_SEMANTIC_STRUCTURE_ERROR_PREFIX = "qa_semantic_artifact_invalid:"
_QA_SEMANTIC_EVIDENCE_ERROR_PREFIX = (
    "qa_semantic_evidence_provenance_invalid:"
)
_QA_MISSING_EVIDENCE_ERROR = "qa_completion_requires_successful_read_evidence"
_KNOWLEDGE_BASE_COVERAGE_FAILURE = "knowledge_base_coverage_failure"
_RETRIEVAL_RECALL_FAILURE = "retrieval_recall_failure"
_RETRIEVAL_STRATEGY_FAILURE = "retrieval_strategy_failure"
_RETRIEVAL_QUERY_CANDIDATE_INJECTION = (
    "qa_retrieval_query_candidate_answer_injection"
)
_RETRIEVAL_QUERY_STRATEGY_LABEL_INJECTION = (
    "qa_retrieval_query_strategy_label_injection"
)
_RETRIEVAL_QUERY_SCOPE_MODIFIER_LOSS = (
    "qa_retrieval_query_scope_modifier_loss"
)
_QA_ORDINAL_RELATION_SCOPE_MISMATCH = (
    "qa_ordinal_relation_scope_mismatch"
)
_QA_LOCATION_CONTAINMENT_LINEAGE_MISSING = (
    "qa_location_containment_lineage_missing"
)
_QA_LOCATION_RELATION_GROUNDING_QUERY_MISMATCH = (
    "qa_location_relation_grounding_query_mismatch"
)

# PROJECT_NECESSARY_ADAPTATION: a successful factual read can ground the
# requested first-hop relation while exposing a finer locality whose
# city/town containment is still unresolved.  SkillFlow's public
# Action--Observation continuation must then query the receipt-grounded object,
# not restart from the question entity.  These are relation-class terms for
# action admission only; they neither select a candidate parent nor impose a
# Director topology.
_LOCATION_RELATION_GROUNDING_TERMS = frozenset(
    {
        "belong",
        "belongs",
        "city",
        "contained",
        "district",
        "municipality",
        "part",
        "sublocality",
        "suburb",
        "town",
    }
)

# These are action-domain descriptions rather than useful SkillFlow FTS
# surfaces.  RetrievalIndex compiles public query tokens with OR semantics, so
# admitting them would let orchestration language displace the grounded
# locality/type terms that are supposed to drive relation retrieval.
_LOCATION_RELATION_GROUNDING_METAWORDS = frozenset(
    {
        "administrative",
        "containment",
        "geographic",
        "grounding",
        "relation",
        "type",
    }
)

# PROJECT_NECESSARY_ADAPTATION: SkillFlow supplies the public search/read
# actions and bounded continuation, while unified_architecture_v2 requires a
# bounded factual-QA recovery policy.  The initial top-k is the existing
# TriviaQA search limit; each public retry broadens it without changing the
# upstream retrieval backend.
_FACTUAL_QA_RETRIEVAL_STRATEGIES = (
    "initial_retrieval",
    "spelling_normalization",
    "alias_expansion",
    "entity_disambiguation",
    "query_rewriting",
)
_FACTUAL_QA_RETRIEVAL_STRATEGY_GUIDANCE = {
    "initial_retrieval": (
        "Keep only the target entity/topic anchor and requested-relation "
        "content words. Remove wh-phrases, answer-slot wording that does not "
        "express the relation, auxiliary verbs, and generic question syntax."
    ),
    "spelling_normalization": (
        "Correct spelling, tokenization, and grammatical noise in the "
        "entity/relation surfaces while preserving entity identity, relation, "
        "and scope. Replace the noisy surface; do not append a strategy label "
        "or introduce a new topic."
    ),
    "alias_expansion": (
        "Replace, rather than append to, an entity surface with a common "
        "alternate name or a relation surface with an established domain "
        "expression or near-synonym of the same predicate. Because this is a "
        "discovery query rather than evidence, the alternate lexical surface "
        "need not already occur verbatim in the original question or returned "
        "title/snippet. Preserve the same entity/topic, requested relation, and "
        "scope; do not introduce an unrelated subtype, work, event, other "
        "subtopic, or candidate answer."
    ),
    "entity_disambiguation": (
        "Prefer an exact title anchor from a prior returned hit only when that "
        "title and its snippet jointly identify the target entity/topic. Add only "
        "question-supported type or context terms that distinguish homonyms, "
        "preserve the requested relation, and do not reuse a title from an "
        "irrelevant subtopic."
    ),
    "query_rewriting": (
        "Preserve the exact chosen entity anchor and reformulate the requested "
        "relation between verbal and nominal paraphrases with the same predicate "
        "meaning. Remove interrogative, auxiliary, answer-slot, and other noisy "
        "terms; do not introduce a new subtopic or candidate answer."
    ),
}
_FACTUAL_QA_SEARCH_LIMITS = (5, 10, 15, 20, 25)


def _factual_transition_frontier_guidance(strategy_progress_count: int) -> str:
    """Describe legal retrieval transitions without an ordinal role template."""

    if strategy_progress_count <= 0:
        return _FACTUAL_QA_RETRIEVAL_STRATEGY_GUIDANCE["initial_retrieval"]
    return (
        "Choose one adjacent public-evidence-supported spelling normalization, "
        "alias expansion, entity disambiguation, or query rewriting transition. "
        "A query rewrite may remove only question syntax/noise while preserving "
        "content-bearing tokens, or add relation context copied from a prior "
        "mirror-valid snippet that binds entity, relation, and scope. The "
        "transition label is derived after execution from adjacent public "
        "Action--Observation receipts, never from attempt order."
    )

# PROJECT_NECESSARY_ADAPTATION: SkillFlow's lexical RetrievalIndex compiles
# every query token into the FTS5/BM25 match expression.  A policy that copies
# an action-mask label such as ``alias_expansion`` into ``query`` therefore
# changes retrieval rather than expressing that strategy.  Reject only these
# orchestration metawords; the model remains responsible for choosing the
# task-grounded entity/relation paraphrase and no answer surface is supplied.
_FACTUAL_QA_STRATEGY_METAWORDS = frozenset(
    {
        "alias",
        "aliases",
        "disambiguate",
        "disambiguation",
        "normalisation",
        "normalise",
        "normalization",
        "normalize",
        "rewrite",
        "rewriting",
        "spelling",
        "strategy",
        "synonym",
        "synonyms",
    }
)

# Function words do not carry the requested relation.  The remaining tokens
# must still be present in the exact read-backed evidence span below.
_RELATION_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "its",
        "of",
        "on",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)
_RELATION_IRREGULAR_LEMMAS = {
    "are": "be",
    "became": "become",
    "began": "begin",
    "begun": "begin",
    "came": "come",
    "comes": "come",
    "gone": "go",
    "held": "hold",
    "is": "be",
    "made": "make",
    "was": "be",
    "were": "be",
    "went": "go",
    "won": "win",
    "wrote": "write",
    "written": "write",
}

# PROJECT_NECESSARY_ADAPTATION: the public SkillFlow RetrievalIndex and read
# receipt ground lexical predicates but deliberately do not decide whether two
# predicates express the same question relation.  Keep this normalization
# deliberately small and linguistic: it covers common argument-preserving
# alternations, requires a shared public question/read context below, and never
# contains a benchmark entity, answer, date, or passage identifier.
_CONTROLLED_RELATION_PARAPHRASE_CLASSES = (
    frozenset({"award", "receive", "win"}),
    frozenset({"beat", "defeat", "win"}),
    frozenset({"come", "from", "originate"}),
)

# Retrieval-query paraphrases are discovery actions, not evidence entailment.
# Keep their lexical admission separate from the stricter proposition-level
# relation classes above: ``birthplace`` can be useful for retrieving evidence
# about where someone comes from, but does not by itself prove that the person
# was born there.
_RETRIEVAL_QUERY_RELATION_PARAPHRASE_CLASSES = (
    *_CONTROLLED_RELATION_PARAPHRASE_CLASSES,
    frozenset(
        {
            "birth",
            "birthplace",
            "born",
            "come",
            "from",
            "originate",
            "origin",
        }
    ),
)

# PROJECT_NECESSARY_ADAPTATION: SkillFlow intentionally leaves retrieval-query
# lexical choice to the policy.  The bounded factual-QA schedule nevertheless
# needs an answer-free way to verify that ``alias_expansion`` and
# ``query_rewriting`` changed a relation surface instead of merely reordering
# or appending FTS terms.  These are small linguistic equivalence classes; they
# contain no benchmark entity, date, answer, task ID, or corpus passage.  A
# generic relation head such as ``chart`` remains available to the lexical
# retriever, but is deliberately not a strong alias: on its own it cannot
# preserve the narrower ``hit chart`` predicate or admit a read candidate.
_FACTUAL_RELATION_ALIAS_SURFACES = {
    "hit_chart": (
        ("hit", "chart"),
        ("hit", "parade"),
        ("music", "hit", "parade"),
    ),
    "publication": (
        ("publish",),
        ("publication",),
    ),
}
_RETRIEVAL_QUERY_STRATEGY_SEMANTICS_MISMATCH = (
    "qa_retrieval_query_strategy_semantics_mismatch"
)
_RETRIEVAL_QUERY_RELATION_CLASS_LOSS = (
    "qa_retrieval_query_target_relation_loss"
)
_RETRIEVAL_QUERY_ENTITY_ANCHOR_LOSS = (
    "qa_retrieval_query_entity_anchor_loss"
)
_RETRIEVAL_QUERY_NAMED_SCOPE_LOSS = (
    "qa_retrieval_query_named_scope_loss"
)

_RELATION_CONTEXT_STOPWORDS = frozenset(
    {
        *_RELATION_FUNCTION_WORDS,
        "city",
        "date",
        "day",
        "decade",
        "how",
        "name",
        "number",
        "place",
        "time",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "year",
    }
)

_ENTITY_COREFERENCE_PRONOUNS = frozenset(
    {
        "he",
        "her",
        "hers",
        "him",
        "his",
        "it",
        "its",
        "she",
        "their",
        "theirs",
        "them",
        "they",
    }
)

_CALENDAR_MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)"
)

_QUESTION_ANCHOR_WH_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
    }
)

# PROJECT_NECESSARY_ADAPTATION: a query-rewrite transition may remove only
# question syntax that does not carry entity, relation, answer-slot scope, or
# ordinal content.  This list is linguistic and benchmark-answer-free; public
# search receipts remain the authority for any newly introduced context term.
_QUERY_REWRITE_NOISE_TOKENS = frozenset(
    {
        *_RELATION_FUNCTION_WORDS,
        *_QUESTION_ANCHOR_WH_WORDS,
        "answer",
        "give",
        "please",
        "question",
        "tell",
    }
)


def _strong_answer_surface_type(surface: str) -> str | None:
    """Classify only unambiguous answer surfaces without external knowledge.

    This deliberately leaves names and places unclassified: deciding whether
    an arbitrary proper noun is a person or location requires knowledge that
    the answer-free Retriever does not own.  Calendar and scalar surfaces are
    sufficiently explicit to reject a read that fills the wrong relation
    argument (for example, a birth date for a birthplace question).
    """

    normalized = " ".join(surface.casefold().split()).strip(" .,:;()[]")
    if normalized in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", normalized):
        return "year_or_number"
    if re.search(rf"\b{_CALENDAR_MONTH_PATTERN}\b", normalized) and re.search(
        r"\b\d{1,4}\b", normalized
    ):
        return "date"
    if re.fullmatch(
        r"\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?",
        normalized,
    ):
        return "date"
    if re.fullmatch(
        r"(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)|[a-z-]+)\s+century",
        normalized,
    ):
        return "date"
    if re.fullmatch(
        r"(?:the\s+)?(?:\d{2}|\d{3}0)[’']?s(?:\s*(?:ad|ce|bc|bce))?",
        normalized,
    ):
        return "date"
    if re.fullmatch(
        r"[+-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:\s*(?:%|percent))?",
        normalized,
    ):
        return "number"
    return None


def _answer_surface_type_issue(*, expected_type: str, surface: str) -> str | None:
    """Return a conservative question-slot/evidence-argument mismatch."""

    observed_type = _strong_answer_surface_type(surface)
    admitted_observed_types = {
        "date": {"date", "year_or_number"},
        "number": {"number", "year_or_number"},
        "yes_no": {"yes_no"},
    }
    if expected_type in admitted_observed_types:
        if observed_type not in admitted_observed_types[expected_type]:
            return (
                "Evidence Retriever evidence proposition does not supply the "
                f"question's requested {expected_type} relation argument"
            )
        return None
    if expected_type in {"location", "entity", "nationality"} and observed_type in {
        "date",
        "number",
        "year_or_number",
        "yes_no",
    }:
        return (
            "Evidence Retriever evidence proposition supplies a "
            f"{observed_type.replace('_', '/')} relation argument, not the "
            "question's requested "
            f"{expected_type} relation argument"
        )
    return None


def _explicit_named_geographic_scope(question: str) -> str | None:
    """Return a conservative ``Where in <named place>`` scope, if present.

    This is a lexical question-scope check, not a gazetteer.  Requiring title-
    cased name tokens and an auxiliary boundary avoids treating an arbitrary
    lower-case prepositional phrase as a geographic entity.
    """

    match = re.search(
        r"(?i:\bwhere\s+in\s+(?:the\s+)?)"
        r"(?P<scope>[A-Z][A-Za-z'’.-]*"
        r"(?:\s+(?:(?:of|the|and)\s+)?[A-Z][A-Za-z'’.-]*){0,4})"
        r"(?=\s+(?i:was|were|is|are|did|does|do|has|have|had|can|"
        r"could|would|will)\b|\s*[,?])",
        question,
    )
    if match is None:
        return None
    scope = " ".join(match.group("scope").split())
    return scope or None


def _canonical_location_surface(surface: str) -> str:
    """Canonicalize a public location surface for receipt-lineage matching."""

    return " ".join(
        re.findall(
            r"\w+(?:['’]\w+)?",
            unicodedata.normalize("NFKC", surface).casefold(),
            flags=re.UNICODE,
        )
    )


def _contains_canonical_location_surface(text: str, surface: str) -> bool:
    canonical_text = _canonical_location_surface(text)
    canonical_surface = _canonical_location_surface(surface)
    return bool(
        canonical_surface
        and re.search(
            rf"(?<!\w){re.escape(canonical_surface)}(?!\w)",
            canonical_text,
        )
    )


def _finer_locality_surface_in_span(candidate: str, evidence_span: str) -> bool:
    """Recognize receipt-explicit locality surfaces requiring resolution."""

    if not _contains_canonical_location_surface(evidence_span, candidate):
        return False
    # A retrieval proposition may preserve the complete public location
    # surface as its object (for example ``<locality>, <administrative area>``)
    # instead of selecting only the leading locality.  Treat that surface as
    # unresolved, not as proof that either comma component is the requested
    # canonical settlement.  The receipt-backed resolution branches below
    # decide whether to preserve the leading city/town or follow an explicit
    # containment relation to its parent.
    if _comma_qualified_location_components(candidate):
        return True
    escaped_candidate = re.escape(candidate.strip())
    comma_qualified = re.search(
        rf"(?<![\w]){escaped_candidate}(?![\w])\s*,\s*\S",
        evidence_span,
        flags=re.IGNORECASE,
    )
    explicitly_typed = re.search(
        rf"\b(?:district|suburb|sublocality)\s+(?:of|called|named)\s+"
        rf"{escaped_candidate}(?![\w])|"
        rf"(?<![\w]){escaped_candidate}(?![\w])\s+(?:is|was)\s+"
        r"(?:an?\s+|the\s+)?(?:district|suburb|sublocality)\b",
        evidence_span,
        flags=re.IGNORECASE,
    )
    return comma_qualified is not None or explicitly_typed is not None


def _comma_qualified_location_components(surface: str) -> tuple[str, ...]:
    """Return exact non-empty comma components for one public surface."""

    components = tuple(component.strip() for component in surface.split(","))
    if len(components) < 2 or any(not component for component in components):
        return ()
    return components


def _location_surface_component_aliases(surface: str) -> frozenset[str]:
    """Return only the full surface and its exact leading comma component."""

    aliases = {_canonical_location_surface(surface)}
    components = _comma_qualified_location_components(surface)
    if components:
        aliases.add(_canonical_location_surface(components[0]))
    aliases.discard("")
    return frozenset(aliases)


def _location_resolution_answer_field_constraint(
    *,
    original_question: str,
    entity_anchor: str,
    read_evidence_texts: Sequence[str],
) -> str | None:
    """Derive branch-A/B answer binding only from public read bodies.

    A finer locality explicitly contained by a city/town selects the
    containment proposition's object.  A locality whose own body types it as
    a city/town selects that proposition's subject.  Conflicting or incomplete
    public bodies deliberately leave the schema unconstrained.
    """

    named_scope = _explicit_named_geographic_scope(original_question)
    anchor_components = _comma_qualified_location_components(entity_anchor)
    anchor = anchor_components[0] if anchor_components else entity_anchor
    if (
        not isinstance(named_scope, str)
        or not named_scope.strip()
        or not isinstance(anchor, str)
        or not anchor.strip()
    ):
        return None
    escaped_anchor = re.escape(anchor.strip())
    branches: set[str] = set()
    for read_text in read_evidence_texts:
        if not isinstance(read_text, str):
            continue
        for clause in _evidence_proposition_clauses(read_text):
            if (
                not _contains_canonical_location_surface(clause, anchor)
                or not _contains_canonical_location_surface(
                    clause,
                    named_scope,
                )
            ):
                continue
            explicit_containment = bool(
                re.search(
                    rf"(?<![\w]){escaped_anchor}(?![\w])\s+"
                    r"(?:is|was)\s+[^.!?]{0,100}"
                    r"(?:\bpart\s+of\b|\bbelongs?\s+to\b|"
                    r"\b(?:district|suburb|sublocality)\b[^.!?]{0,40}\bof\b)"
                    r"[^.!?]{0,100}\b(?:city|town)\b|"
                    rf"(?<![\w]){escaped_anchor}(?![\w])\s+"
                    r"(?:is|was)\s+(?:an?\s+|the\s+)?"
                    r"(?:district|suburb|sublocality)\b[^.!?]{0,100}\bof\b",
                    clause,
                    flags=re.IGNORECASE,
                )
            )
            typed_as_city_or_town = bool(
                re.search(
                    rf"(?<![\w]){escaped_anchor}(?![\w])\s+"
                    r"(?:is|was)\s+(?:an?\s+|the\s+)?(?:city|town)\b",
                    clause,
                    flags=re.IGNORECASE,
                )
            )
            if explicit_containment:
                branches.add("object_or_attribute_value")
            if typed_as_city_or_town:
                branches.add("subject")
    return next(iter(branches)) if len(branches) == 1 else None


def _public_reads_matching_span(
    evidence_span: str,
    read_evidence_texts: Sequence[str],
) -> tuple[str, ...]:
    """Return public read bodies that contain the exact lexical span."""

    return tuple(
        read_text
        for read_text in read_evidence_texts
        if isinstance(read_text, str)
        and _contains_canonical_location_surface(read_text, evidence_span)
    )


def _public_read_title_binds_subject(
    read_text: str,
    subject: str,
) -> bool:
    """Use a public passage title only as an exact entity-identity signal."""

    raw_title = getattr(read_text, "passage_title", None)
    if not isinstance(raw_title, str) or not raw_title.strip():
        return False
    canonical_subject = _canonical_location_surface(subject)
    if not canonical_subject:
        return False
    title_components = _comma_qualified_location_components(raw_title)
    title_surfaces = {
        _canonical_location_surface(raw_title),
        *(
            (_canonical_location_surface(title_components[0]),)
            if title_components
            else ()
        ),
    }
    return canonical_subject in title_surfaces


def _chain_binds_location_resolution(
    *,
    multi_hop_chain: object,
    first_hop: Mapping[str, object],
    selected_subject: str,
    resolution_surface: str,
) -> bool:
    """Require distinct first-hop and resolution entries in the public chain."""

    if not isinstance(multi_hop_chain, (list, tuple)):
        return False
    entries = tuple(item for item in multi_hop_chain if isinstance(item, str))
    if len(entries) < 2:
        return False
    first_subject = first_hop.get("subject")
    first_locality = first_hop.get("object_or_attribute_value")
    if not isinstance(first_subject, str) or not isinstance(first_locality, str):
        return False
    first_locality_aliases = _location_surface_component_aliases(
        first_locality
    )
    first_indices = {
        index
        for index, entry in enumerate(entries)
        if _contains_canonical_location_surface(entry, first_subject)
        and any(
            _contains_canonical_location_surface(entry, alias)
            for alias in first_locality_aliases
        )
    }
    resolution_indices = {
        index
        for index, entry in enumerate(entries)
        if _contains_canonical_location_surface(entry, selected_subject)
        and _contains_canonical_location_surface(entry, resolution_surface)
    }
    return bool(
        first_indices
        and resolution_indices
        and any(
            first_index != resolution_index
            for first_index in first_indices
            for resolution_index in resolution_indices
        )
    )


def _location_containment_lineage_issue(
    *,
    original_question: str,
    reasoner_fields: Mapping[str, object],
    read_evidence_texts: Sequence[str] = (),
) -> str | None:
    """Require read-backed locality type/containment resolution when needed.

    The gate is deliberately state-conditioned and answer-free.  It activates
    only for an explicit named geographic scope when a proposition exposes an
    unresolved or finer locality whose direct evidence does not bind that
    scope.  A public body may prove either that the leading locality is itself
    the city/town answer or that it is contained by one.  Existing one-hop
    answers otherwise retain the unified factual protocol.
    """

    answer_slot = reasoner_fields.get("answer_slot")
    propositions = reasoner_fields.get("evidence_propositions")
    multi_hop_chain = reasoner_fields.get("multi_hop_chain")
    candidate_answer = reasoner_fields.get("candidate_answer")
    if (
        not isinstance(answer_slot, Mapping)
        or answer_slot.get("answer_type") != "location"
        or qa_answer_type_constraint(original_question) != "location"
        or not isinstance(propositions, (list, tuple))
        or not isinstance(candidate_answer, str)
        or not candidate_answer.strip()
    ):
        return None
    named_scope = _explicit_named_geographic_scope(original_question)
    if named_scope is None:
        return None

    finer_first_hops: list[Mapping[str, object]] = []
    for proposition in propositions:
        if not isinstance(proposition, Mapping):
            continue
        subject = proposition.get("subject")
        locality = proposition.get("object_or_attribute_value")
        evidence_span = proposition.get("evidence_span")
        relation = proposition.get("relation")
        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in (subject, locality, evidence_span, relation)
        ):
            continue
        assert isinstance(subject, str)
        assert isinstance(locality, str)
        assert isinstance(evidence_span, str)
        assert isinstance(relation, str)
        supporting_clauses = tuple(
            clause
            for clause in _evidence_proposition_clauses(evidence_span)
            if _contains_canonical_location_surface(clause, subject)
            and _contains_canonical_location_surface(clause, locality)
            and _relation_surface_matches_evidence(relation, clause)
        )
        scope_bound_in_supporting_clause = any(
            _contains_canonical_location_surface(clause, named_scope)
            for clause in supporting_clauses
        )
        if (
            scope_bound_in_supporting_clause
            or not _finer_locality_surface_in_span(locality, evidence_span)
            or re.search(r"\b(?:part\s+of|belongs?\s+to)\b", relation, re.I)
        ):
            continue
        finer_first_hops.append(proposition)
    if not finer_first_hops:
        return None

    proposition_index = answer_slot.get("proposition_index")
    answer_field = answer_slot.get("answer_field")
    if (
        isinstance(proposition_index, bool)
        or not isinstance(proposition_index, int)
        or proposition_index < 0
        or proposition_index >= len(propositions)
        or answer_field not in {"subject", "object_or_attribute_value"}
    ):
        return _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
    selected = propositions[proposition_index]
    if not isinstance(selected, Mapping):
        return _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
    selected_subject = selected.get("subject")
    selected_relation = selected.get("relation")
    selected_object = selected.get("object_or_attribute_value")
    selected_span = selected.get("evidence_span")
    if not all(
        isinstance(value, str) and bool(value.strip())
        for value in (
            selected_subject,
            selected_relation,
            selected_object,
            selected_span,
        )
    ):
        return _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
    assert isinstance(selected_subject, str)
    assert isinstance(selected_relation, str)
    assert isinstance(selected_object, str)
    assert isinstance(selected_span, str)

    matching_public_reads = _public_reads_matching_span(
        selected_span,
        read_evidence_texts,
    )
    public_title_binds_subject = any(
        _public_read_title_binds_subject(read_text, selected_subject)
        for read_text in matching_public_reads
    )
    selected_parent_is_city_or_town = bool(
        re.search(
            rf"\b(?:city|town)\s+of\s+"
            rf"{re.escape(selected_object.strip())}(?![\w])",
            selected_span,
            flags=re.IGNORECASE,
        )
    )
    selected_child_is_finer_locality = bool(
        re.search(
            rf"(?<![\w]){re.escape(selected_subject.strip())}(?![\w])\s+"
            r"(?:is|was)\s+(?:an?\s+|the\s+)?"
            r"(?:district|suburb|sublocality)\b|"
            r"\b(?:district|suburb|sublocality)\s+of\s+"
            rf"{re.escape(selected_subject.strip())}(?![\w])",
            selected_span,
            flags=re.IGNORECASE,
        )
    )
    selected_relation_is_containment = bool(
        re.search(
            r"\b(?:part\s+of|belongs?\s+to)\b",
            selected_relation,
            flags=re.IGNORECASE,
        )
    )
    selected_body_is_containment = bool(
        re.search(
            r"\b(?:part\s+of|belongs?\s+to)\b",
            selected_span,
            flags=re.IGNORECASE,
        )
    )
    selected_binds_scope = _contains_canonical_location_surface(
        selected_span,
        named_scope,
    )
    linked_first_hop = next(
        (
            proposition
            for proposition in finer_first_hops
            if isinstance(proposition.get("object_or_attribute_value"), str)
            and _canonical_location_surface(
                selected_subject
            )
            in _location_surface_component_aliases(
                str(proposition["object_or_attribute_value"])
            )
        ),
        None,
    )
    if linked_first_hop is None:
        return _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING

    # Resolution branch A: the public body explicitly states that the leading
    # locality is contained by a city/town in the named scope.  A matching
    # public title may bind the leading comma component to that body, but it
    # never supplies the containment predicate or geographic scope itself.
    containment_chain_binds = _chain_binds_location_resolution(
        multi_hop_chain=multi_hop_chain,
        first_hop=linked_first_hop,
        selected_subject=selected_subject,
        resolution_surface=candidate_answer,
    )
    if all(
        (
            answer_field == "object_or_attribute_value",
            selected_object == candidate_answer,
            _contains_canonical_location_surface(
                selected_span,
                candidate_answer,
            ),
            selected_parent_is_city_or_town,
            selected_child_is_finer_locality or selected_body_is_containment,
            selected_relation_is_containment,
            selected_body_is_containment,
            selected_binds_scope,
            public_title_binds_subject,
            containment_chain_binds,
        )
    ):
        return None

    # Resolution branch B: a comma-qualified surface can already denote the
    # canonical city/town.  Preserve its exact leading component only when a
    # successful public read body types that subject as a city/town inside the
    # named scope.  The administrative suffix is never promoted by punctuation.
    escaped_subject = re.escape(selected_subject.strip())
    selected_body_types_subject_as_city_or_town = bool(
        re.search(
            rf"(?<![\w]){escaped_subject}(?![\w])\s+(?:is|was)\s+"
            r"[^.!?]{0,120}\b(?:city|town)\b",
            selected_span,
            flags=re.IGNORECASE,
        )
    )
    selected_relation_types_city_or_town = bool(
        re.search(r"\b(?:city|town)\b", selected_relation, flags=re.IGNORECASE)
    )
    settlement_chain_binds = _chain_binds_location_resolution(
        multi_hop_chain=multi_hop_chain,
        first_hop=linked_first_hop,
        selected_subject=selected_subject,
        resolution_surface=named_scope,
    )
    if all(
        (
            answer_field == "subject",
            selected_subject == candidate_answer,
            selected_body_types_subject_as_city_or_town,
            selected_relation_types_city_or_town,
            selected_binds_scope,
            public_title_binds_subject,
            settlement_chain_binds,
        )
    ):
        return None
    return _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING


def _location_containment_repair_anchor(
    *,
    original_question: str,
    completion_action: Mapping[str, object],
) -> str | None:
    """Recover the receipt-grounded finer locality from a rejected artifact.

    This helper is called only after semantic admission returned the typed
    ``qa_location_containment_lineage_missing`` diagnosis.  Consequently the
    proposition and exact evidence span have already passed the public Tool
    receipt provenance checks.  The projected anchor is the leading locality
    component from that public model artifact; no accepted answer or evaluator
    field is consulted.
    """

    arguments = completion_action.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    value = arguments.get("value")
    if not isinstance(value, Mapping):
        return None
    propositions = value.get("evidence_propositions")
    if not isinstance(propositions, (list, tuple)):
        return None
    named_scope = _explicit_named_geographic_scope(original_question)
    if named_scope is None:
        return None
    for proposition in propositions:
        if not isinstance(proposition, Mapping):
            continue
        locality = proposition.get("object_or_attribute_value")
        evidence_span = proposition.get("evidence_span")
        relation = proposition.get("relation")
        if not all(
            isinstance(item, str) and bool(item.strip())
            for item in (locality, evidence_span, relation)
        ):
            continue
        assert isinstance(locality, str)
        assert isinstance(evidence_span, str)
        assert isinstance(relation, str)
        if (
            _contains_canonical_location_surface(evidence_span, named_scope)
            or not _finer_locality_surface_in_span(locality, evidence_span)
            or re.search(r"\b(?:part\s+of|belongs?\s+to)\b", relation, re.I)
        ):
            continue
        leading_component = locality.split(",", maxsplit=1)[0].strip()
        return leading_component or locality.strip()
    return None


def _location_relation_grounding_query_issue(
    *,
    query: str,
    entity_anchor: str | None,
    named_scope: str | None,
    original_question: str | None = None,
) -> str | None:
    """Validate an answer-free location relation-grounding search query."""

    constraint_violations: list[str] = []
    if not isinstance(entity_anchor, str) or not entity_anchor.strip():
        constraint_violations.append("missing_receipt_grounded_entity_anchor")
    elif not _contains_canonical_location_surface(query, entity_anchor):
        constraint_violations.append("missing_receipt_grounded_entity_anchor")
    if not isinstance(named_scope, str) or not named_scope.strip():
        constraint_violations.append("missing_question_geographic_scope")
    elif not _contains_canonical_location_surface(query, named_scope):
        constraint_violations.append("missing_question_geographic_scope")
    query_terms = frozenset(_normalized_retrieval_query(query).split())
    if not query_terms.intersection(_LOCATION_RELATION_GROUNDING_TERMS):
        constraint_violations.append(
            "missing_lexical_geographic_type_or_containment_relation"
        )
    if query_terms.intersection(_LOCATION_RELATION_GROUNDING_METAWORDS):
        constraint_violations.append("orchestration_class_label_in_query")
    if isinstance(original_question, str) and original_question.strip():
        original_entity_tokens = set(
            _question_entity_anchor_tokens(original_question)
        )
        locality_tokens = set(
            _normalized_retrieval_query(entity_anchor or "").split()
        )
        scope_tokens = set(
            _normalized_retrieval_query(named_scope or "").split()
        )
        restarted_entity_tokens = (
            original_entity_tokens - locality_tokens - scope_tokens
        ).intersection(query_terms)
        if restarted_entity_tokens:
            constraint_violations.append("original_question_entity_restart")
    if not constraint_violations:
        return None
    return _QA_LOCATION_RELATION_GROUNDING_QUERY_MISMATCH + ": constraints=" + (
        json.dumps(constraint_violations, ensure_ascii=False)
    )


def _location_relation_candidate_compatible(
    *,
    title: object,
    snippet: object,
    entity_anchor: str | None,
    named_scope: str | None,
) -> bool:
    """Return whether a public hit can support the location-resolution read.

    Search snippets are not proof, but the next opaque passage ID must at least
    bind the receipt-grounded locality, the named geographic scope, and one
    natural location type/containment surface.  The strict read receipt and
    proposition validator remain authoritative after the passage is opened.
    """

    if (
        not isinstance(title, str)
        or not isinstance(snippet, str)
        or not isinstance(entity_anchor, str)
        or not entity_anchor.strip()
        or not isinstance(named_scope, str)
        or not named_scope.strip()
    ):
        return False
    public_surface = f"{title} {snippet}"
    anchor_aliases = _location_surface_component_aliases(entity_anchor)
    snippet_terms = frozenset(_normalized_retrieval_query(snippet).split())
    return bool(
        any(
            _contains_canonical_location_surface(public_surface, alias)
            for alias in anchor_aliases
        )
        and _contains_canonical_location_surface(snippet, named_scope)
        and snippet_terms.intersection(_LOCATION_RELATION_GROUNDING_TERMS)
    )


def _normalized_retrieval_query(query: str) -> str:
    """Canonicalize only for duplicate-request admission, not retrieval."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _retrieval_query_term_set_signature(query: str) -> tuple[str, ...]:
    """Mirror SkillFlow FTS query equivalence without changing its backend.

    ``RetrievalIndex._compile_match_query`` removes duplicate terms and joins
    the remaining terms with ``OR``.  Token order therefore does not express a
    new retrieval strategy.  This signature is used only for action admission;
    the original query remains unchanged in the Tool request and receipt.
    """

    return tuple(sorted(set(_normalized_retrieval_query(query).split())))


def _retrieval_query_strategy_metawords(
    query: str,
    *,
    original_question: str,
) -> tuple[str, ...]:
    """Return question-external control labels that would pollute retrieval."""

    tokens = set(_normalized_retrieval_query(query).split())
    question_tokens = set(_normalized_retrieval_query(original_question).split())
    return tuple(
        sorted((tokens - question_tokens) & _FACTUAL_QA_STRATEGY_METAWORDS)
    )


def _relation_token_variants(token: str) -> frozenset[str]:
    """Return conservative English inflection variants for one relation token.

    This is deliberately not semantic paraphrase matching.  It admits only
    deterministic inflectional forms needed at the structured evidence
    boundary (for example ``publish``/``published``), while the exact Tool read
    receipt, entity arguments, and relation predicate remain authoritative.
    """

    normalized = token.casefold()
    variants = {normalized}
    irregular = _RELATION_IRREGULAR_LEMMAS.get(normalized)
    if irregular is not None:
        variants.add(irregular)
    if len(normalized) > 4 and normalized.endswith("ied"):
        variants.add(normalized[:-3] + "y")
    if len(normalized) > 3 and normalized.endswith("ed"):
        base = normalized[:-2]
        variants.add(base)
        variants.add(base + "e")
        if len(base) > 2 and base[-1] == base[-2]:
            variants.add(base[:-1])
    if len(normalized) > 4 and normalized.endswith("ing"):
        base = normalized[:-3]
        variants.add(base)
        variants.add(base + "e")
        if len(base) > 2 and base[-1] == base[-2]:
            variants.add(base[:-1])
    if len(normalized) > 3 and normalized.endswith("ies"):
        variants.add(normalized[:-3] + "y")
    if len(normalized) > 3 and normalized.endswith("es"):
        variants.add(normalized[:-2])
    if len(normalized) > 2 and normalized.endswith("s"):
        variants.add(normalized[:-1])
    return frozenset(variant for variant in variants if variant)


def _relation_surface_matches_evidence(
    relation_surface: str,
    evidence_surface: str,
) -> bool:
    """Check exact or conservative inflectional relation grounding."""

    canonical_relation = " ".join(relation_surface.casefold().split())
    canonical_evidence = " ".join(evidence_surface.casefold().split())
    if canonical_relation and canonical_relation in canonical_evidence:
        return True
    relation_tokens = tuple(
        token
        for token in re.findall(r"\w+", canonical_relation, flags=re.UNICODE)
        if token not in _RELATION_FUNCTION_WORDS
    )
    evidence_variants = tuple(
        _relation_token_variants(token)
        for token in re.findall(r"\w+", canonical_evidence, flags=re.UNICODE)
    )
    return bool(relation_tokens) and all(
        any(
            _relation_token_variants(relation_token) & evidence_token_variants
            for evidence_token_variants in evidence_variants
        )
        for relation_token in relation_tokens
    )


def _relation_surface_tokens(surface: str) -> tuple[str, ...]:
    """Return content-bearing lexical relation tokens."""

    return tuple(
        token
        for token in re.findall(
            r"\w+(?:['’]\w+)?",
            unicodedata.normalize("NFKC", surface).casefold(),
            flags=re.UNICODE,
        )
        if token not in _RELATION_FUNCTION_WORDS
    )


def _relation_surfaces_share_content(left: str, right: str) -> bool:
    """Check whether two relation surfaces share an inflectional content token."""

    return any(
        _relation_token_variants(left_token)
        & _relation_token_variants(right_token)
        for left_token in _relation_surface_tokens(left)
        for right_token in _relation_surface_tokens(right)
    )


def _surface_uses_relation_class(
    surface: str,
    relation_class: frozenset[str],
) -> bool:
    return any(
        _relation_token_variants(token) & relation_class
        for token in _scope_tokens(surface)
    )


def _controlled_relation_paraphrase(
    *,
    question_relation: str,
    evidence_predicate: str,
    original_question: str,
    evidence_span: str,
) -> bool:
    """Admit only a controlled paraphrase with a shared public event anchor.

    The Retriever does not prove semantic entailment.  It publishes the
    question-side relation and the exact receipt-side predicate separately for
    the Reasoner and Verifier.  This gate merely prevents an unrelated exact
    predicate from being labelled as a paraphrase: both surfaces must belong to
    one small linguistic alternation class and the original question/read span
    must share at least one non-relational public context token.
    """

    matching_classes = tuple(
        relation_class
        for relation_class in _CONTROLLED_RELATION_PARAPHRASE_CLASSES
        if _surface_uses_relation_class(question_relation, relation_class)
        and _surface_uses_relation_class(evidence_predicate, relation_class)
    )
    # A public history sentence can realize the first publication event as an
    # introduction followed by later chart events.  Keep this narrower than a
    # global ``publish == introduce`` synonym: the same receipt must contain
    # the explicit in-sequence ``introduced ... followed by ...`` construction
    # and the original question must request the first event.  Entity, object,
    # answer type and exact-span checks remain authoritative below.
    onset_publication_class = frozenset({"introduce", "publish"})
    if (
        "first" in _question_ordinal_classes(original_question)
        and _receipt_first_in_sequence_onset(evidence_span)
        and _surface_uses_relation_class(
            question_relation,
            onset_publication_class,
        )
        and _surface_uses_relation_class(
            evidence_predicate,
            onset_publication_class,
        )
    ):
        matching_classes = (*matching_classes, onset_publication_class)
    if not matching_classes:
        return False
    excluded = set(_RELATION_CONTEXT_STOPWORDS)
    for relation_class in matching_classes:
        excluded.update(relation_class)
    question_context = {
        token
        for token in re.findall(
            r"\w+(?:['’]\w+)?",
            unicodedata.normalize("NFKC", original_question).casefold(),
            flags=re.UNICODE,
        )
        if token not in excluded and len(token) > 1
    }
    evidence_context = {
        token
        for token in re.findall(
            r"\w+(?:['’]\w+)?",
            unicodedata.normalize("NFKC", evidence_span).casefold(),
            flags=re.UNICODE,
        )
        if token not in excluded and len(token) > 1
    }
    return bool(question_context & evidence_context)


def _receipt_first_in_sequence_onset(surface: str) -> bool:
    """Recognize one receipt-explicit introduction followed by a later event.

    ``introduced`` alone does not prove a global ordinal.  The bounded
    same-sentence ``followed by`` continuation is the public chronology signal
    that makes the introduction an admissible realization of ``first``.
    """

    normalized = unicodedata.normalize("NFKC", surface)
    return bool(
        re.search(
            r"\bintroduc(?:e|ed|es|ing)\b"
            r"[^.!?]{0,240}\b(?:was\s+|were\s+)?followed\s+by\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )


_QUESTION_SCOPE_ORDINAL_MODIFIERS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "earliest",
        "latest",
        "last",
        "initial",
        "original",
    }
)

# PROJECT_NECESSARY_ADAPTATION: the upstream SkillFlow retrieval environment
# publishes exact search/read observations but deliberately does not decide
# semantic entailment.  The unified QA completion boundary therefore needs a
# small answer-free equivalence relation for explicit ordinal scope.  These
# are linguistic realizations of order/onset, not dataset entities, candidate
# answers, or corpus-specific retrieval recipes.
_ORDINAL_SCOPE_EQUIVALENTS = {
    "first": frozenset(
        {
            "1st",
            "began",
            "begin",
            "beginning",
            "begun",
            "commence",
            "commenced",
            "commencing",
            "earliest",
            "first",
            "inaugural",
            "initial",
            "initially",
            "start",
            "started",
            "starting",
        }
    ),
    "second": frozenset({"2nd", "second"}),
    "third": frozenset({"3rd", "third"}),
    "fourth": frozenset({"4th", "fourth"}),
    "fifth": frozenset({"5th", "fifth"}),
    "last": frozenset({"final", "last", "latest"}),
}
_ORDINAL_SCOPE_CANONICAL = {
    surface: canonical
    for canonical, surfaces in _ORDINAL_SCOPE_EQUIVALENTS.items()
    for surface in surfaces
}
_ORDINAL_SCOPE_DETERMINERS = frozenset(
    {
        "a",
        "an",
        "any",
        "her",
        "his",
        "its",
        "our",
        "the",
        "their",
        "this",
        "that",
    }
)


def _scope_tokens(surface: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"\w+(?:['’]\w+)?",
            unicodedata.normalize("NFKC", surface).casefold(),
            flags=re.UNICODE,
        )
    )


def _tokens_contain_alias_surface(
    tokens: Sequence[str],
    alias_surface: Sequence[str],
) -> bool:
    """Match one relation surface with conservative inflection handling."""

    width = len(alias_surface)
    if width == 0 or len(tokens) < width:
        return False
    return any(
        all(
            _relation_token_variants(token) & _relation_token_variants(alias)
            for token, alias in zip(tokens[offset : offset + width], alias_surface)
        )
        for offset in range(len(tokens) - width + 1)
    )


def _relation_alias_surfaces_in(
    surface: str,
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Return maximal answer-free relation surfaces present in text."""

    tokens = _scope_tokens(surface)
    matches: dict[str, frozenset[tuple[str, ...]]] = {}
    for relation_class, aliases in _FACTUAL_RELATION_ALIAS_SURFACES.items():
        present = {
            alias
            for alias in aliases
            if _tokens_contain_alias_surface(tokens, alias)
        }
        # Retain only maximal spans so replacing one multi-token surface with
        # another is observable, while appending both is not.
        maximal = {
            alias
            for alias in present
            if not any(
                alias != other
                and len(alias) < len(other)
                and any(
                    tuple(other[offset : offset + len(alias)]) == alias
                    for offset in range(len(other) - len(alias) + 1)
                )
                for other in present
            )
        }
        if maximal:
            matches[relation_class] = frozenset(maximal)
    return matches


def _question_relation_surface_alternatives(
    original_question: str,
) -> dict[str, tuple[str, ...]]:
    """Project answer-free strong alternatives for question relation classes."""

    required_classes = frozenset(
        _relation_alias_surfaces_in(original_question)
    )
    return {
        relation_class: tuple(
            " ".join(alias_surface)
            for alias_surface in _FACTUAL_RELATION_ALIAS_SURFACES[relation_class]
        )
        for relation_class in sorted(required_classes)
    }


def _missing_required_relation_classes(
    original_question: str,
    candidate_surface: str,
) -> tuple[str, ...]:
    """Return strong question relation classes absent from a query/candidate."""

    required_classes = frozenset(
        _relation_alias_surfaces_in(original_question)
    )
    candidate_classes = frozenset(
        _relation_alias_surfaces_in(candidate_surface)
    )
    return tuple(sorted(required_classes - candidate_classes))


def _question_entity_anchor_tokens(original_question: str) -> tuple[str, ...]:
    """Extract a bounded lexical entity/topic anchor from the public question.

    Proper-name tokens before the requested ordinal/relation are preferred.
    For generic questions (for example ``Which institution first ...``), the
    head noun between the wh-word and relation is retained.  This is only an
    action-domain compatibility check and never infers an answer entity.
    """

    raw_tokens = tuple(
        re.findall(
            r"[^\W\d_]+(?:[-'’][^\W\d_]+)*",
            unicodedata.normalize("NFKC", original_question),
            flags=re.UNICODE,
        )
    )
    normalized = tuple(token.casefold() for token in raw_tokens)
    ordinal_positions = tuple(
        index
        for index, token in enumerate(normalized)
        if token in _ORDINAL_SCOPE_CANONICAL
        or token in _QUESTION_SCOPE_ORDINAL_MODIFIERS
    )
    relation_matches = _relation_alias_surfaces_in(original_question)
    relation_tokens = {
        token
        for aliases in relation_matches.values()
        for alias in aliases
        for token in alias
    }
    relation_positions = tuple(
        index
        for index, token in enumerate(normalized)
        if any(
            _relation_token_variants(token)
            & _relation_token_variants(relation_token)
            for relation_token in relation_tokens
        )
    )
    # An entity name may itself contain a relation-class noun (for example a
    # publication named ``... Chart``).  The explicit ordinal is therefore the
    # authoritative boundary when present; only ordinal-free questions fall
    # back to the first recognized relation surface.
    cutoff = (
        min(ordinal_positions)
        if ordinal_positions
        else min(relation_positions)
        if relation_positions
        else len(raw_tokens)
    )
    proper_positions = tuple(
        index
        for index, token in enumerate(raw_tokens[:cutoff])
        if token[:1].isupper()
        and normalized[index] not in _QUESTION_ANCHOR_WH_WORDS
        and normalized[index] not in _RELATION_CONTEXT_STOPWORDS
    )
    if proper_positions:
        # Bind the proper-name phrase nearest the requested relation, rather
        # than merging separate proper spans such as a location constraint and
        # the target person into one impossible ordered anchor.
        phrase_start = len(proper_positions) - 1
        while (
            phrase_start > 0
            and proper_positions[phrase_start - 1]
            == proper_positions[phrase_start] - 1
        ):
            phrase_start -= 1
        entity_positions = proper_positions[phrase_start:]
        proper = [normalized[index] for index in entity_positions]
        # Preserve one immediately adjacent lower-case entity type noun.  This
        # keeps a public anchor such as ``<proper name> magazine`` ordered and
        # prevents a broad proper-name token from matching unrelated subtypes.
        # The token is derived only from the question; no entity/type catalogue
        # or benchmark surface is introduced here.
        adjacent_index = entity_positions[-1] + 1
        if (
            ordinal_positions
            and adjacent_index < cutoff
            and adjacent_index == entity_positions[-1] + 1
            and raw_tokens[adjacent_index][:1].islower()
            and normalized[adjacent_index] not in _RELATION_CONTEXT_STOPWORDS
            and normalized[adjacent_index]
            not in _QUESTION_SCOPE_ORDINAL_MODIFIERS
            and normalized[adjacent_index] not in _ORDINAL_SCOPE_CANONICAL
            and not any(
                _relation_token_variants(normalized[adjacent_index])
                & _relation_token_variants(relation_token)
                for relation_token in relation_tokens
            )
        ):
            proper.append(normalized[adjacent_index])
        return tuple(proper)

    wh_positions = tuple(
        index
        for index, token in enumerate(normalized)
        if token in _QUESTION_ANCHOR_WH_WORDS
    )
    start = wh_positions[-1] + 1 if wh_positions else 0
    ignored = {
        *_RELATION_CONTEXT_STOPWORDS,
        *_QUESTION_SCOPE_ORDINAL_MODIFIERS,
        *_ORDINAL_SCOPE_CANONICAL,
        "did",
    }
    generic = tuple(
        token
        for token in normalized[start:cutoff]
        if token not in ignored
        and not any(
            _relation_token_variants(token)
            & _relation_token_variants(relation_token)
            for relation_token in relation_tokens
        )
    )
    return generic[-2:]


def _question_has_proper_entity_anchor(original_question: str) -> bool:
    """Return whether the question publishes an explicit proper-name anchor."""

    return any(
        token[:1].isupper()
        and token.casefold() not in _QUESTION_ANCHOR_WH_WORDS
        and token.casefold() not in _RELATION_CONTEXT_STOPWORDS
        for token in re.findall(
            r"[^\W\d_]+(?:[-'’][^\W\d_]+)*",
            unicodedata.normalize("NFKC", original_question),
            flags=re.UNICODE,
        )
    )


def _surface_binds_entity_anchor(
    surface: str,
    entity_anchor_tokens: Sequence[str],
) -> bool:
    if not entity_anchor_tokens:
        return False
    surface_tokens = _scope_tokens(surface)
    width = len(entity_anchor_tokens)
    return any(
        all(
            _relation_token_variants(anchor)
            & _relation_token_variants(surface_token)
            for anchor, surface_token in zip(
                entity_anchor_tokens,
                surface_tokens[offset : offset + width],
            )
        )
        for offset in range(max(0, len(surface_tokens) - width + 1))
    )


@dataclass(frozen=True, slots=True)
class _FactualRetrievalStrategyProof:
    """Answer-free observability for one bounded retrieval attempt."""

    strategy: str
    verified: bool
    proof_strength: str
    source_passage_ids: tuple[str, ...] = ()

    def to_value(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "verified": self.verified,
            "proof_strength": self.proof_strength,
            "source_passage_ids": list(self.source_passage_ids),
        }


@dataclass(frozen=True, slots=True)
class _FactualRetrievalAttemptRecord:
    """Compact answer-free record of one executed public search transition.

    SkillFlow keeps the authoritative StructuredAction, public Observation,
    and Tool receipt.  This record does not add a new retrieval operation or
    infer an entity alias.  It only projects the fields that must agree when
    that existing transition is replayed from the receipt, and states whether
    the query advanced the bounded strategy schedule or was a top-k recall
    expansion of the latest query variant.
    """

    attempt_index: int
    required_strategy: str | None
    query_variant: str
    normalized_query: str
    fts_term_set: tuple[str, ...]
    required_top_k: int
    observed_top_k: int | None
    strategy_advanced: bool
    query_variant_verified: bool
    tool_transition_verified: bool
    hit_count: int
    recall_expansion: bool = False

    @property
    def verified(self) -> bool:
        return (
            self.query_variant_verified
            and self.tool_transition_verified
            and self.observed_top_k == self.required_top_k
        )

    def to_value(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "required_strategy": self.required_strategy,
            "query_variant": self.query_variant,
            "normalized_query": self.normalized_query,
            "fts_term_set": list(self.fts_term_set),
            "required_top_k": self.required_top_k,
            "observed_top_k": self.observed_top_k,
            "strategy_advanced": self.strategy_advanced,
            "recall_expansion": self.recall_expansion,
            "query_variant_verified": self.query_variant_verified,
            "tool_transition_verified": self.tool_transition_verified,
            "hit_count": self.hit_count,
            "verified": self.verified,
        }


def _relation_surface_replacement_classes(
    *,
    original_question: str,
    previous_query: str,
    query: str,
) -> frozenset[str]:
    """Return question-derived relation classes replaced between two queries."""

    if not _surface_binds_entity_anchor(
        query,
        _question_entity_anchor_tokens(original_question),
    ):
        return frozenset()
    if _missing_question_scope_modifiers(original_question, query):
        return frozenset()
    question_classes = frozenset(_relation_alias_surfaces_in(original_question))
    previous = _relation_alias_surfaces_in(previous_query)
    current = _relation_alias_surfaces_in(query)
    if (
        not question_classes
        or not question_classes <= previous.keys()
        or not question_classes <= current.keys()
    ):
        return frozenset()
    # A controlled relation replacement changes only a surface belonging to
    # one already-required relation class.  Every other normalized query token
    # (including repetitions) must remain identical; otherwise a model could
    # smuggle a new subtopic, guessed entity, or candidate into an apparent
    # alias/rewrite transition.
    def non_relation_token_multiset(surface: str) -> tuple[str, ...]:
        tokens = _scope_tokens(surface)
        occupied_relation_positions: set[int] = set()
        present_surfaces = _relation_alias_surfaces_in(surface)
        for relation_class in question_classes:
            for alias_surface in present_surfaces.get(
                relation_class,
                frozenset(),
            ):
                width = len(alias_surface)
                for offset in range(len(tokens) - width + 1):
                    if all(
                        _relation_token_variants(token)
                        & _relation_token_variants(alias)
                        for token, alias in zip(
                            tokens[offset : offset + width],
                            alias_surface,
                        )
                    ):
                        occupied_relation_positions.update(
                            range(offset, offset + width)
                        )
        return tuple(
            sorted(
                token
                for index, token in enumerate(tokens)
                if index not in occupied_relation_positions
            )
        )

    if non_relation_token_multiset(previous_query) != non_relation_token_multiset(
        query
    ):
        return frozenset()
    return frozenset(
        relation_class
        for relation_class in question_classes
        if previous[relation_class] - current[relation_class]
        and current[relation_class] - previous[relation_class]
    )


def _query_replaces_relation_surface(
    *,
    original_question: str,
    previous_query: str,
    query: str,
) -> bool:
    """Verify a same-class relation replacement, never addition/reordering."""

    return bool(
        _relation_surface_replacement_classes(
            original_question=original_question,
            previous_query=previous_query,
            query=query,
        )
    )


def _query_transition_is_inflectional_normalization(
    previous_query: str,
    query: str,
) -> bool:
    """Recognize a lexical normalization without assigning it by ordinal.

    The two public search queries must differ as FTS term sets while retaining
    a one-to-one conservative inflectional token alignment.  This intentionally
    recognizes forms such as ``publish`` -> ``published`` but not a semantic
    alias, added context term, or term-order-only recall retry.
    """

    previous_tokens = list(_scope_tokens(previous_query))
    query_tokens = list(_scope_tokens(query))
    if (
        not previous_tokens
        or len(previous_tokens) != len(query_tokens)
        or _retrieval_query_term_set_signature(previous_query)
        == _retrieval_query_term_set_signature(query)
    ):
        return False
    unmatched = list(query_tokens)
    for previous_token in previous_tokens:
        match_index = next(
            (
                index
                for index, query_token in enumerate(unmatched)
                if _relation_token_variants(previous_token)
                & _relation_token_variants(query_token)
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _query_transition_is_ordinal_surface_normalization(
    previous_query: str,
    query: str,
) -> bool:
    """Recognize an answer-free replacement within one ordinal class."""

    previous_tokens = _scope_tokens(previous_query)
    query_tokens = _scope_tokens(query)
    if (
        not previous_tokens
        or _retrieval_query_term_set_signature(previous_query)
        == _retrieval_query_term_set_signature(query)
    ):
        return False

    def canonical_multiset(tokens: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            sorted(_ORDINAL_SCOPE_CANONICAL.get(token, token) for token in tokens)
        )

    return canonical_multiset(previous_tokens) == canonical_multiset(query_tokens)


def _query_transition_is_controlled_relation_paraphrase(
    *,
    original_question: str,
    previous_query: str,
    query: str,
) -> bool:
    """Verify a question-grounded relation paraphrase without answer injection.

    Both adjacent queries and the original question must realize one controlled
    linguistic relation class.  Outside that class, the transition may only
    delete question syntax or restore tokens already published by the question.
    """

    question_tokens = _scope_tokens(original_question)

    def token_is_question_backed(token: str) -> bool:
        return any(
            _relation_token_variants(token)
            & _relation_token_variants(question_token)
            for question_token in question_tokens
        )

    def residual_tokens(
        surface: str,
        relation_class: frozenset[str],
    ) -> list[str]:
        return [
            token
            for token in _scope_tokens(surface)
            if not (_relation_token_variants(token) & relation_class)
        ]

    for relation_class in _RETRIEVAL_QUERY_RELATION_PARAPHRASE_CLASSES:
        if not all(
            _surface_uses_relation_class(surface, relation_class)
            for surface in (original_question, previous_query, query)
        ):
            continue
        previous_relation_tokens = {
            token
            for token in _scope_tokens(previous_query)
            if _relation_token_variants(token) & relation_class
        }
        query_relation_tokens = {
            token
            for token in _scope_tokens(query)
            if _relation_token_variants(token) & relation_class
        }
        if previous_relation_tokens == query_relation_tokens:
            continue
        previous_residual = residual_tokens(previous_query, relation_class)
        remaining_query_residual = residual_tokens(query, relation_class)
        removed_tokens: list[str] = []
        for token in previous_residual:
            try:
                remaining_query_residual.remove(token)
            except ValueError:
                removed_tokens.append(token)
        if any(
            token not in _QUERY_REWRITE_NOISE_TOKENS
            for token in removed_tokens
        ):
            continue
        if all(
            token_is_question_backed(token)
            for token in remaining_query_residual
        ):
            return True
    return False


def _question_retrieval_relation_context_tokens(
    original_question: str,
) -> tuple[str, ...]:
    """Project answer-free relation/scope content from the public question."""

    question_tokens = _scope_tokens(original_question)
    entity_anchor = _question_entity_anchor_tokens(original_question)
    content = tuple(
        token
        for token in question_tokens
        if token not in _RELATION_CONTEXT_STOPWORDS
        and token not in _ORDINAL_SCOPE_CANONICAL
        and not any(
            _relation_token_variants(token)
            & _relation_token_variants(anchor)
            for anchor in entity_anchor
        )
    )
    if content:
        return content
    return tuple(
        token
        for token in question_tokens
        if token not in _RELATION_CONTEXT_STOPWORDS
        and token not in _ORDINAL_SCOPE_CANONICAL
    )


def _question_named_constraint_tokens(
    original_question: str,
) -> frozenset[str]:
    """Return explicit question-side named scope tokens outside the entity."""

    entity_anchor = _question_entity_anchor_tokens(original_question)
    named: set[str] = set()
    for index, match in enumerate(
        re.finditer(
            r"[^\W\d_]+(?:[-'’][^\W\d_]+)*",
            unicodedata.normalize("NFKC", original_question),
            flags=re.UNICODE,
        )
    ):
        surface = match.group(0)
        token = surface.casefold()
        if (
            not surface[:1].isupper()
            or (index == 0 and token in _QUESTION_ANCHOR_WH_WORDS)
            or token in _RELATION_CONTEXT_STOPWORDS
            or any(
                _relation_token_variants(token)
                & _relation_token_variants(anchor)
                for anchor in entity_anchor
            )
        ):
            continue
        named.add(token)
    return frozenset(named)


def _missing_question_named_constraints(
    original_question: str,
    candidate_surface: str,
) -> tuple[str, ...]:
    candidate_tokens = _scope_tokens(candidate_surface)
    return tuple(
        constraint
        for constraint in sorted(
            _question_named_constraint_tokens(original_question)
        )
        if not any(
            _relation_token_variants(constraint)
            & _relation_token_variants(candidate)
            for candidate in candidate_tokens
        )
    )


def _snippet_relation_context_supports_tokens(
    *,
    snippet: str,
    added_tokens: Sequence[str],
    relation_context_tokens: Sequence[str],
) -> bool:
    """Verify lowercase relation context near a question relation anchor.

    This is a conservative public-receipt lexical check, not semantic
    entailment.  It rejects Title-Case entity/value import and requires every
    newly introduced token to occur in the same clause and local window as a
    question-derived relation/scope token.
    """

    if not added_tokens or not relation_context_tokens:
        return False
    for clause in re.split(r"[.!?;]+", unicodedata.normalize("NFKC", snippet)):
        raw_tokens = re.findall(
            r"[^\W\d_]+(?:[-'’][^\W\d_]+)*",
            clause,
            flags=re.UNICODE,
        )
        normalized_tokens = tuple(token.casefold() for token in raw_tokens)
        anchor_positions = tuple(
            index
            for index, token in enumerate(normalized_tokens)
            if any(
                _relation_token_variants(token)
                & _relation_token_variants(anchor)
                for anchor in relation_context_tokens
            )
        )
        if not anchor_positions:
            continue
        supported = True
        for added in added_tokens:
            added_positions = tuple(
                index
                for index, token in enumerate(normalized_tokens)
                if _relation_token_variants(token)
                & _relation_token_variants(added)
                and raw_tokens[index][:1].islower()
            )
            if not added_positions or not any(
                abs(added_index - anchor_index) <= 6
                for added_index in added_positions
                for anchor_index in anchor_positions
            ):
                supported = False
                break
        if supported:
            return True
    return False


def _public_title_transition_support(
    *,
    original_question: str,
    previous_query: str,
    query: str,
    prior_observation: Mapping[str, object] | None,
) -> tuple[bool, tuple[str, ...]]:
    """Return exact prior-hit support for a title-conditioned transition."""

    if not isinstance(prior_observation, Mapping):
        return False, ()
    transition_verified, verified_passage_ids = (
        _public_search_transition_mirror(prior_observation)
    )
    result = prior_observation.get("result")
    raw_hits = result.get("hits") if isinstance(result, Mapping) else None
    if not transition_verified or not isinstance(raw_hits, list):
        return False, ()
    query_tokens = _scope_tokens(query)
    previous_query_tokens = _scope_tokens(previous_query)
    entity_anchor = _question_entity_anchor_tokens(original_question)
    relation_context_tokens = _question_retrieval_relation_context_tokens(
        original_question
    )
    source_ids: list[str] = []
    for hit in raw_hits:
        if not isinstance(hit, Mapping):
            continue
        title = hit.get("title")
        snippet = hit.get("snippet")
        passage_id = hit.get("passage_id")
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(snippet, str)
            or not isinstance(passage_id, str)
            or passage_id.strip() not in verified_passage_ids
        ):
            continue
        title_tokens = _scope_tokens(title)
        public_surface = f"{title} {snippet}"
        if (
            title_tokens
            and not any(
                tuple(
                    previous_query_tokens[
                        offset : offset + len(title_tokens)
                    ]
                )
                == title_tokens
                for offset in range(
                    max(
                        0,
                        len(previous_query_tokens) - len(title_tokens) + 1,
                    )
                )
            )
            and any(
                tuple(query_tokens[offset : offset + len(title_tokens)])
                == title_tokens
                for offset in range(
                    max(0, len(query_tokens) - len(title_tokens) + 1)
                )
            )
            and (
                not entity_anchor
                or _surface_binds_entity_anchor(public_surface, entity_anchor)
            )
            and any(
                _relation_token_variants(public_token)
                & _relation_token_variants(context_token)
                for public_token in _scope_tokens(public_surface)
                for context_token in relation_context_tokens
            )
            and not _missing_required_relation_classes(
                original_question,
                public_surface,
            )
            and not _missing_question_scope_modifiers(
                original_question,
                public_surface,
            )
        ):
            source_ids.append(passage_id.strip())
    return bool(source_ids), tuple(dict.fromkeys(source_ids))


def _query_rewriting_transition_support(
    *,
    original_question: str,
    previous_query: str,
    query: str,
    prior_observation: Mapping[str, object] | None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify one answer-free rewrite from adjacent public search state.

    A rewrite either deletes only question syntax/noise while retaining every
    content-bearing query token, or adds lexical context copied from a prior
    mirror-valid snippet that itself binds the unchanged entity, relation and
    scope.  Ground truth, accepted aliases and evaluator state are not inputs.
    """

    if not isinstance(prior_observation, Mapping):
        return False, ()
    prior_verified, verified_passage_ids = _public_search_transition_mirror(
        prior_observation
    )
    if not prior_verified:
        return False, ()
    if (
        not _surface_binds_entity_anchor(
            query,
            _question_entity_anchor_tokens(original_question),
        )
        or _missing_required_relation_classes(original_question, query)
        or _missing_question_scope_modifiers(original_question, query)
        or (
            not _missing_question_named_constraints(
                original_question,
                previous_query,
            )
            and _missing_question_named_constraints(
                original_question,
                query,
            )
        )
    ):
        return False, ()

    previous_tokens = list(_scope_tokens(previous_query))
    query_tokens = list(_scope_tokens(query))
    remaining_query_tokens = list(query_tokens)
    removed_tokens: list[str] = []
    for token in previous_tokens:
        try:
            remaining_query_tokens.remove(token)
        except ValueError:
            removed_tokens.append(token)
    added_tokens = tuple(remaining_query_tokens)
    if not removed_tokens and not added_tokens:
        return False, ()
    relation_context_tokens = _question_retrieval_relation_context_tokens(
        original_question
    )
    named_constraint_tokens = _question_named_constraint_tokens(
        original_question
    )
    content_removed_tokens = tuple(
        token
        for token in removed_tokens
        if token not in _QUERY_REWRITE_NOISE_TOKENS
    )
    if any(
        token in named_constraint_tokens
        or not any(
            _relation_token_variants(token)
            & _relation_token_variants(context_token)
            for context_token in relation_context_tokens
        )
        for token in content_removed_tokens
    ):
        return False, ()
    if not added_tokens:
        return (not content_removed_tokens), ()
    if any(
        token in _QUERY_REWRITE_NOISE_TOKENS
        or token in _FACTUAL_QA_STRATEGY_METAWORDS
        or not re.search(r"[^\W\d_]", token, flags=re.UNICODE)
        for token in added_tokens
    ):
        return False, ()

    question_tokens = _scope_tokens(original_question)
    receipt_conditioned_added_tokens = tuple(
        token
        for token in added_tokens
        if not any(
            _relation_token_variants(token)
            & _relation_token_variants(question_token)
            for question_token in question_tokens
        )
    )
    if not receipt_conditioned_added_tokens:
        return (not content_removed_tokens), ()
    # Local snippet co-occurrence can justify additive discovery context, but
    # it cannot prove that an arbitrary new token is synonymous with a removed
    # predicate. Predicate replacement remains limited to the inflectional,
    # ordinal-equivalence and controlled same-predicate branches above.
    if content_removed_tokens:
        return False, ()

    result = prior_observation.get("result")
    raw_hits = result.get("hits") if isinstance(result, Mapping) else None
    if not isinstance(raw_hits, list):
        return False, ()
    entity_anchor = _question_entity_anchor_tokens(original_question)
    source_ids: list[str] = []
    for hit in raw_hits:
        if not isinstance(hit, Mapping):
            continue
        passage_id = hit.get("passage_id")
        snippet = hit.get("snippet")
        if (
            not isinstance(passage_id, str)
            or passage_id.strip() not in verified_passage_ids
            or not isinstance(snippet, str)
            or not snippet.strip()
            or not _surface_binds_entity_anchor(snippet, entity_anchor)
            or _missing_required_relation_classes(original_question, snippet)
            or _missing_question_scope_modifiers(original_question, snippet)
        ):
            continue
        if _snippet_relation_context_supports_tokens(
            snippet=snippet,
            added_tokens=receipt_conditioned_added_tokens,
            relation_context_tokens=relation_context_tokens,
        ):
            source_ids.append(passage_id.strip())
    return bool(source_ids), tuple(dict.fromkeys(source_ids))


def _factual_transition_strategy_identification(
    *,
    original_question: str,
    previous_query: str | None,
    query: str,
    prior_observation: Mapping[str, object] | None = None,
) -> tuple[str, bool, tuple[str, ...]]:
    """Identify the strongest transition strategy from public invariants.

    One distinct search transition yields one proof record and one strongest
    public-invariant label.  No label is inferred from the transition's
    ordinal position.
    """

    entity_preserved = _surface_binds_entity_anchor(
        query,
        _question_entity_anchor_tokens(original_question),
    )
    scope_preserved = not _missing_question_scope_modifiers(
        original_question,
        query,
    )
    named_scope_preserved = bool(
        _missing_question_named_constraints(
            original_question,
            previous_query or "",
        )
        or not _missing_question_named_constraints(
            original_question,
            query,
        )
    )
    relation_preserved = not _missing_required_relation_classes(
        original_question,
        query,
    )
    invariants_verified = bool(
        entity_preserved
        and scope_preserved
        and named_scope_preserved
        and relation_preserved
    )
    if previous_query is None:
        return "initial_retrieval", invariants_verified, ()

    replacement_classes = _relation_surface_replacement_classes(
        original_question=original_question,
        previous_query=previous_query,
        query=query,
    )
    if replacement_classes:
        # A deterministic same-class relation-surface replacement is the
        # strongest answer-free evidence available here.  More general query
        # rewriting remains unverified unless another controlled invariant is
        # added; do not duplicate one transition into two proof records.
        return "alias_expansion", invariants_verified, ()

    if _query_transition_is_inflectional_normalization(
        previous_query,
        query,
    ):
        return "spelling_normalization", invariants_verified, ()

    if _query_transition_is_ordinal_surface_normalization(
        previous_query,
        query,
    ):
        return "query_rewriting", invariants_verified, ()

    if _query_transition_is_controlled_relation_paraphrase(
        original_question=original_question,
        previous_query=previous_query,
        query=query,
    ):
        return "alias_expansion", invariants_verified, ()

    rewrite_supported, rewrite_source_ids = (
        _query_rewriting_transition_support(
            original_question=original_question,
            previous_query=previous_query,
            query=query,
            prior_observation=prior_observation,
        )
    )
    if rewrite_supported:
        return (
            "query_rewriting",
            invariants_verified,
            rewrite_source_ids,
        )

    title_supported, source_ids = _public_title_transition_support(
        original_question=original_question,
        previous_query=previous_query,
        query=query,
        prior_observation=prior_observation,
    )

    previous_tokens = set(_scope_tokens(previous_query))
    query_tokens = set(_scope_tokens(query))
    if query_tokens - previous_tokens and title_supported:
        return (
            "entity_disambiguation",
            invariants_verified,
            source_ids,
        )

    # A distinct transition which merely removes or substitutes unsupported
    # lexical material remains observable, but public evidence has not
    # identified a legal spelling, alias, disambiguation, or rewrite stage.
    return "query_rewriting", False, ()


def _factual_strategy_semantics_verified(
    *,
    original_question: str,
    distinct_queries: Sequence[str],
) -> tuple[bool, ...]:
    """Measure adjacent-query semantics without ordinal stage assignment."""

    verified: list[bool] = []
    for index, query in enumerate(distinct_queries):
        if index >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES):
            break
        _, transition_verified, _ = (
            _factual_transition_strategy_identification(
                original_question=original_question,
                previous_query=(distinct_queries[index - 1] if index else None),
                query=query,
            )
        )
        verified.append(transition_verified)
    return tuple(verified)


def _public_search_transition_mirror(
    observation: Mapping[str, object],
) -> tuple[bool, tuple[str, ...]]:
    """Verify the exact SkillFlow search Action--Observation receipt mirror."""

    executed_action = observation.get("executed_action")
    result = observation.get("result")
    if not isinstance(executed_action, Mapping) or not isinstance(result, Mapping):
        return False, ()
    arguments = executed_action.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, ()
    action_query = arguments.get("query")
    action_limit = arguments.get("limit")
    raw_passage_ids = result.get("passage_ids")
    raw_hits = result.get("hits")
    if not isinstance(raw_passage_ids, list) or not isinstance(raw_hits, list):
        return False, ()
    if len(raw_passage_ids) != len(raw_hits):
        return False, ()

    passage_ids: list[str] = []
    hit_passage_ids: list[str] = []
    hits_complete = True
    expected_hit_fields = {
        "document_id",
        "passage_id",
        "rank",
        "snippet",
        "title",
    }
    for expected_rank, (passage_id, hit) in enumerate(
        zip(raw_passage_ids, raw_hits),
        start=1,
    ):
        if not isinstance(passage_id, str) or not passage_id.strip():
            hits_complete = False
            continue
        passage_ids.append(passage_id.strip())
        if not isinstance(hit, Mapping) or set(hit) != expected_hit_fields:
            hits_complete = False
            continue
        hit_passage_id = hit.get("passage_id")
        document_id = hit.get("document_id")
        title = hit.get("title")
        snippet = hit.get("snippet")
        rank = hit.get("rank")
        if (
            not isinstance(hit_passage_id, str)
            or not hit_passage_id.strip()
            or not isinstance(document_id, str)
            or not document_id.strip()
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(snippet, str)
            or not snippet.strip()
            or type(rank) is not int
            or rank != expected_rank
        ):
            hits_complete = False
            continue
        hit_passage_ids.append(hit_passage_id.strip())

    verified = bool(
        observation.get("observation_status") == "success"
        and executed_action.get("kind") == "tool"
        and executed_action.get("name") == "search"
        and executed_action.get("resource_id") == QA_RETRIEVAL_TOOL_ID
        and set(arguments) == {"query", "limit"}
        and isinstance(action_query, str)
        and bool(action_query.strip())
        and action_query == result.get("query")
        and type(action_limit) is int
        and action_limit > 0
        and action_limit == result.get("top_k")
        and result.get("operation") == "search"
        and hits_complete
        and len(passage_ids) == len(raw_passage_ids)
        and len(hit_passage_ids) == len(raw_hits)
        and tuple(passage_ids) == tuple(hit_passage_ids)
    )
    return verified, tuple(passage_ids) if verified else ()


def _public_read_transition_mirror(
    observation: Mapping[str, object],
) -> tuple[bool, str | None]:
    """Verify one public SkillFlow read Action--Observation mirror."""

    executed_action = observation.get("executed_action")
    result = observation.get("result")
    if not isinstance(executed_action, Mapping) or not isinstance(result, Mapping):
        return False, None
    arguments = executed_action.get("arguments")
    passage = result.get("passage")
    if not isinstance(arguments, Mapping) or not isinstance(passage, Mapping):
        return False, None
    action_passage_id = arguments.get("passage_id")
    result_passage_id = result.get("passage_id", passage.get("passage_id"))
    passage_passage_id = passage.get("passage_id")
    passage_text = passage.get("text")
    verified = bool(
        observation.get("observation_status") == "success"
        and executed_action.get("kind") == "tool"
        and executed_action.get("name") == "read"
        and executed_action.get("resource_id") == QA_RETRIEVAL_TOOL_ID
        and set(arguments) == {"passage_id"}
        and isinstance(action_passage_id, str)
        and bool(action_passage_id.strip())
        and isinstance(result_passage_id, str)
        and action_passage_id == result_passage_id
        and (
            passage_passage_id is None
            or passage_passage_id == action_passage_id
        )
        and result.get("operation") == "read"
        and isinstance(passage_text, str)
        and bool(passage_text.strip())
    )
    return verified, passage_text.strip() if verified else None


def _factual_strategy_proofs(
    *,
    original_question: str,
    distinct_queries: Sequence[str],
    search_observations: Sequence[Mapping[str, object]] = (),
) -> tuple[_FactualRetrievalStrategyProof, ...]:
    """Expose one receipt-verifiable proof per distinct query transition.

    DIRECT_REUSE: SkillFlow supplies the ordered search Action--Observation
    receipts.  The project adaptation classifies each query against its
    immediate predecessor and the adjacent public receipts; it never assigns a
    stage from the query's ordinal position.
    """

    proofs: list[_FactualRetrievalStrategyProof] = []
    for index, query in enumerate(distinct_queries):
        if index >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES):
            break
        current_observation = (
            search_observations[index]
            if index < len(search_observations)
            else None
        )
        prior_observation = (
            search_observations[index - 1]
            if index > 0 and index - 1 < len(search_observations)
            else None
        )
        strategy, semantic_verified, source_ids = (
            _factual_transition_strategy_identification(
                original_question=original_question,
                previous_query=(distinct_queries[index - 1] if index else None),
                query=query,
                prior_observation=prior_observation,
            )
        )
        current_transition_verified, _ = (
            _public_search_transition_mirror(current_observation)
            if isinstance(current_observation, Mapping)
            else (False, ())
        )
        prior_transition_verified = True
        if index > 0:
            prior_transition_verified, prior_passage_ids = (
                _public_search_transition_mirror(prior_observation)
                if isinstance(prior_observation, Mapping)
                else (False, ())
            )
            if not source_ids and prior_transition_verified:
                source_ids = prior_passage_ids
        stage_verified = bool(
            semantic_verified
            and current_transition_verified
            and prior_transition_verified
        )
        strength = (
            "unverified_strategy_attempt"
            if not stage_verified
            else (
                "tool_receipt_conditioned_strategy_attempt"
                if strategy
                in {"spelling_normalization", "entity_disambiguation"}
                or (strategy == "query_rewriting" and bool(source_ids))
                else "deterministic_relation_invariant_strategy_attempt"
                if strategy in {"alias_expansion", "query_rewriting"}
                else "question_invariant_strategy_attempt"
            )
        )
        proofs.append(
            _FactualRetrievalStrategyProof(
                strategy,
                stage_verified,
                strength,
                tuple(dict.fromkeys(source_ids)),
            )
        )
    return tuple(proofs)


def _factual_retrieval_attempt_records(
    *,
    original_question: str,
    search_observations: Sequence[Mapping[str, object]],
) -> tuple[_FactualRetrievalAttemptRecord, ...]:
    """Project every successful search Action--Observation transition.

    DIRECT_REUSE: SkillFlow's ``QARetrievalEnvironment`` admits only
    ``search(query, limit)`` and publishes the same query plus ranked public
    hits in its Observation; the generic FlowSteer Tool runtime serializes
    that request and result unchanged in the Tool receipt.  This thin adapter
    verifies that mirror and the existing five-stage/top-k action mask.  It
    never reads benchmark answers, accepted aliases, or evaluator state.
    """

    records: list[_FactualRetrievalAttemptRecord] = []
    distinct_queries: list[str] = []
    distinct_observations: list[Mapping[str, object]] = []
    distinct_signatures: list[tuple[str, ...]] = []
    prior_top_ks_by_signature: dict[tuple[str, ...], list[int]] = {}
    proof_verified_by_signature: dict[tuple[str, ...], bool] = {}
    proof_by_signature: dict[
        tuple[str, ...], _FactualRetrievalStrategyProof
    ] = {}

    for attempt_index, observation in enumerate(search_observations, start=1):
        executed_action = observation.get("executed_action")
        result = observation.get("result")
        action_arguments = (
            executed_action.get("arguments")
            if isinstance(executed_action, Mapping)
            else None
        )
        action_query = (
            action_arguments.get("query")
            if isinstance(action_arguments, Mapping)
            else None
        )
        action_limit = (
            action_arguments.get("limit")
            if isinstance(action_arguments, Mapping)
            else None
        )
        result_query = result.get("query") if isinstance(result, Mapping) else None
        result_top_k = result.get("top_k") if isinstance(result, Mapping) else None
        query_variant = (
            action_query.strip()
            if isinstance(action_query, str) and action_query.strip()
            else result_query.strip()
            if isinstance(result_query, str) and result_query.strip()
            else ""
        )
        observed_top_k = (
            action_limit
            if type(action_limit) is int and action_limit > 0
            else result_top_k
            if type(result_top_k) is int and result_top_k > 0
            else None
        )
        signature = _retrieval_query_term_set_signature(query_variant)
        # Replay the same monotonic search-attempt schedule admitted live.  A
        # same-query recall expansion consumes the next top-k (5 -> 10 -> 15)
        # without consuming or fabricating a distinct strategy label.
        required_top_k = _FACTUAL_QA_SEARCH_LIMITS[
            min(attempt_index - 1, len(_FACTUAL_QA_SEARCH_LIMITS) - 1)
        ]
        query_transition_advanced = bool(
            query_variant and signature not in distinct_signatures
        )

        if query_transition_advanced:
            candidate_distinct_queries = (*distinct_queries, query_variant)
            proofs = _factual_strategy_proofs(
                original_question=original_question,
                distinct_queries=candidate_distinct_queries,
                search_observations=(*distinct_observations, observation),
            )
            current_proof = (
                proofs[-1]
                if len(proofs) == len(candidate_distinct_queries)
                else None
            )
            query_variant_verified = bool(
                current_proof is not None and current_proof.verified
            )
            required_strategy = (
                current_proof.strategy if current_proof is not None else None
            )
        else:
            # The existing action admission allows only the latest normalized
            # FTS term set at a strictly larger top-k.  Such an attempt expands
            # recall but deliberately does not claim completion of the next
            # spelling/alias/disambiguation/rewrite strategy.
            prior_limits = prior_top_ks_by_signature.get(signature, [])
            query_variant_verified = bool(
                distinct_signatures
                and signature == distinct_signatures[-1]
                and observed_top_k is not None
                and prior_limits
                and observed_top_k > max(prior_limits)
                and proof_verified_by_signature.get(signature, False)
            )
            current_proof = proof_by_signature.get(signature)
            required_strategy = None

        tool_transition_verified, passage_ids = (
            _public_search_transition_mirror(observation)
        )
        records.append(
            _FactualRetrievalAttemptRecord(
                attempt_index=attempt_index,
                required_strategy=required_strategy,
                query_variant=query_variant,
                normalized_query=_normalized_retrieval_query(query_variant),
                fts_term_set=signature,
                required_top_k=required_top_k,
                observed_top_k=observed_top_k,
                strategy_advanced=query_transition_advanced,
                query_variant_verified=query_variant_verified,
                tool_transition_verified=tool_transition_verified,
                hit_count=len(passage_ids),
                recall_expansion=not query_transition_advanced,
            )
        )
        if observed_top_k is not None:
            prior_top_ks_by_signature.setdefault(signature, []).append(
                observed_top_k
            )
        if query_transition_advanced:
            distinct_signatures.append(signature)
            distinct_queries.append(query_variant)
            distinct_observations.append(observation)
            proof_verified_by_signature[signature] = query_variant_verified
            if current_proof is not None:
                proof_by_signature[signature] = current_proof

    return tuple(records)


def _verified_transformed_relation_classes(
    *,
    original_question: str,
    distinct_queries: Sequence[str],
) -> frozenset[str]:
    """Return relation classes covered by verified relation-rewrite stages."""

    transformed: set[str] = set()
    for index, query in enumerate(distinct_queries):
        if index == 0:
            continue
        strategy, stage_verified, _ = (
            _factual_transition_strategy_identification(
                original_question=original_question,
                previous_query=distinct_queries[index - 1],
                query=query,
            )
        )
        if (
            not stage_verified
            or strategy not in {"alias_expansion", "query_rewriting"}
        ):
            continue
        transformed.update(
            _relation_surface_replacement_classes(
                original_question=original_question,
                previous_query=distinct_queries[index - 1],
                query=distinct_queries[index],
            )
        )
    return frozenset(transformed)


def _question_ordinal_classes(original_question: str) -> tuple[str, ...]:
    classes: list[str] = []
    for token in _scope_tokens(original_question):
        canonical = _ORDINAL_SCOPE_CANONICAL.get(token)
        if canonical is None and token in _QUESTION_SCOPE_ORDINAL_MODIFIERS:
            canonical = token
        if canonical is not None and canonical not in classes:
            classes.append(canonical)
    return tuple(classes)


def _ordinal_surface_matches(
    token: str,
    ordinal_classes: Sequence[str],
) -> bool:
    return any(
        token
        in _ORDINAL_SCOPE_EQUIVALENTS.get(
            ordinal_class,
            frozenset({ordinal_class}),
        )
        for ordinal_class in ordinal_classes
    )


def _relation_content_tokens(relation_surface: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _scope_tokens(relation_surface)
        if token not in _RELATION_FUNCTION_WORDS
        and token not in _ORDINAL_SCOPE_CANONICAL
    )


def _token_matches_relation(token: str, relation_token: str) -> bool:
    return bool(
        _relation_token_variants(token)
        & _relation_token_variants(relation_token)
    )


def _evidence_proposition_clauses(evidence_span: str) -> tuple[str, ...]:
    """Split only boundaries that cannot belong to one asserted proposition."""

    normalized = unicodedata.normalize("NFKC", evidence_span)
    clauses = re.split(
        r"(?:[.!?;:]+|,\s*(?=(?:but|while|whereas|although|however|yet|and)\b))",
        normalized,
        flags=re.IGNORECASE,
    )
    return tuple(clause.strip(" ,\t\r\n") for clause in clauses if clause.strip())


def _question_relation_scope_tokens(
    original_question: str,
    relation_surface: str,
) -> tuple[str, ...]:
    """Project the question-side argument governed by an ordinal relation.

    This is deliberately lexical and conservative.  It never infers an alias
    or answer: a semantically different object scope needs an explicit
    receipt-grounded surface instead of being silently broadened.
    """

    question_tokens = _scope_tokens(original_question)
    relation_tokens = _relation_content_tokens(relation_surface)
    relation_positions = tuple(
        index
        for index, question_token in enumerate(question_tokens)
        if any(
            _token_matches_relation(question_token, relation_token)
            for relation_token in relation_tokens
        )
    )
    ordinal_positions = tuple(
        index
        for index, token in enumerate(question_tokens)
        if token in _ORDINAL_SCOPE_CANONICAL
        or token in _QUESTION_SCOPE_ORDINAL_MODIFIERS
    )
    if not ordinal_positions:
        return ()
    if relation_positions:
        relation_index = min(
            relation_positions,
            key=lambda index: min(
                abs(index - ordinal_index)
                for ordinal_index in ordinal_positions
            ),
        )
        start = relation_index + 1
    else:
        # A receipt-grounded semantic relation alias may have no lexical token
        # in the question (for example an onset construction).  The ordinal's
        # following argument is still observable without guessing the answer.
        start = ordinal_positions[-1] + 2
    ignored = {
        *_RELATION_FUNCTION_WORDS,
        *_QUESTION_ANCHOR_WH_WORDS,
        *_ORDINAL_SCOPE_CANONICAL,
        "date",
        "day",
        "decade",
        "month",
        "time",
        "week",
        "when",
        "year",
    }
    return tuple(
        token
        for token in question_tokens[start:]
        if token not in ignored
        and not any(
            _token_matches_relation(token, relation_token)
            for relation_token in relation_tokens
        )
    )


def _relation_aware_ordinal_scope_issue(
    *,
    original_question: str,
    evidence_span: str,
    relation_surface: str,
    proposition_subject: str,
    proposition_object: str,
) -> str | None:
    """Require ordinal, entity, relation, and answer argument in one proposition.

    A token such as ``first`` anywhere in a paragraph is not evidence that the
    selected relation is globally first.  Admission requires one clause that
    contains the structured proposition arguments, a receipt-grounded relation
    realization, and an ordinal/onset realization bound to that relation.  A
    question-global generic argument also cannot be narrowed to an unrequested
    subtype solely by the evidence clause.
    """

    ordinal_classes = _question_ordinal_classes(original_question)
    if not ordinal_classes:
        return None
    relation_tokens = _relation_content_tokens(relation_surface)
    if not relation_tokens:
        return (
            _QA_ORDINAL_RELATION_SCOPE_MISMATCH
            + ": Evidence proposition has no content-bearing relation surface for "
            "the question's ordinal scope"
        )
    question_scope_tokens = _question_relation_scope_tokens(
        original_question,
        relation_surface,
    )
    subject_tokens = _scope_tokens(proposition_subject)
    object_tokens = _scope_tokens(proposition_object)

    for clause in _evidence_proposition_clauses(evidence_span):
        clause_tokens = _scope_tokens(clause)
        if not clause_tokens:
            continue
        clause_surface = " ".join(clause_tokens)
        if not _relation_surface_matches_evidence(
            " ".join(relation_tokens),
            clause_surface,
        ):
            continue
        relation_positions = tuple(
            index
            for index, token in enumerate(clause_tokens)
            if any(
                _token_matches_relation(token, relation_token)
                for relation_token in relation_tokens
            )
        )
        ordinal_positions = tuple(
            index
            for index, token in enumerate(clause_tokens)
            if _ordinal_surface_matches(token, ordinal_classes)
        )
        if (
            not ordinal_positions
            and "first" in ordinal_classes
            and _receipt_first_in_sequence_onset(evidence_span)
        ):
            ordinal_positions = tuple(
                index
                for index, token in enumerate(clause_tokens)
                if "introduce" in _relation_token_variants(token)
            )
        if not relation_positions or not ordinal_positions:
            continue
        subject_grounded = bool(subject_tokens) and " ".join(
            subject_tokens
        ) in clause_surface
        object_grounded = (
            proposition_object.casefold() in {"yes", "no"}
            or (bool(object_tokens) and " ".join(object_tokens) in clause_surface)
        )
        if not subject_grounded or not object_grounded:
            continue

        scope_positions = tuple(
            index
            for index, token in enumerate(clause_tokens)
            if any(
                _token_matches_relation(token, scope_token)
                for scope_token in question_scope_tokens
            )
        )
        ordinal_relation_bound = any(
            abs(ordinal_index - relation_index) <= 3
            or any(
                (
                    relation_index < ordinal_index <= scope_index
                    and scope_index - relation_index <= 8
                )
                or (
                    ordinal_index <= scope_index < relation_index
                    and relation_index - ordinal_index <= 10
                )
                for scope_index in scope_positions
            )
            for ordinal_index in ordinal_positions
            for relation_index in relation_positions
        )
        if not ordinal_relation_bound:
            continue

        if len(question_scope_tokens) > 1:
            distinctive_scope_tokens = question_scope_tokens[:-1]
            if distinctive_scope_tokens and not any(
                any(
                    _token_matches_relation(clause_token, scope_token)
                    for clause_token in clause_tokens
                )
                for scope_token in distinctive_scope_tokens
            ):
                return (
                    _QA_ORDINAL_RELATION_SCOPE_MISMATCH
                    + ": Evidence proposition narrows or changes the question's "
                    "ordinal relation argument scope"
                )
        elif len(question_scope_tokens) == 1:
            head = question_scope_tokens[0]
            head_positions = tuple(
                index
                for index, token in enumerate(clause_tokens)
                if _token_matches_relation(token, head)
            )
            if not head_positions:
                return (
                    _QA_ORDINAL_RELATION_SCOPE_MISMATCH
                    + ": Evidence proposition changes the question-global "
                    "ordinal relation argument"
                )
            for head_index in head_positions:
                nearest_relation = min(
                    relation_positions,
                    key=lambda index: abs(index - head_index),
                )
                nearest_ordinal = min(
                    ordinal_positions,
                    key=lambda index: abs(index - head_index),
                )
                lower = min(nearest_relation, nearest_ordinal, head_index)
                upper = max(nearest_relation, nearest_ordinal, head_index)
                intervening = clause_tokens[lower + 1 : upper]
                narrowing_tokens = tuple(
                    token
                    for token in intervening
                    if token not in _ORDINAL_SCOPE_DETERMINERS
                    and token not in _RELATION_FUNCTION_WORDS
                    and token not in _ORDINAL_SCOPE_CANONICAL
                    and token != head
                    and not any(
                        _token_matches_relation(token, relation_token)
                        for relation_token in relation_tokens
                    )
                )
                if narrowing_tokens:
                    return (
                        _QA_ORDINAL_RELATION_SCOPE_MISMATCH
                        + ": Evidence proposition narrows the question-global "
                        "ordinal relation to an unrequested local/subtype scope"
                    )
                break
        return None
    return (
        _QA_ORDINAL_RELATION_SCOPE_MISMATCH
        + ": Evidence proposition does not bind the question's ordinal scope, "
        "entity, requested relation, and answer-bearing argument in the same "
        "receipt-grounded proposition"
    )


def _missing_question_scope_modifiers(
    original_question: str,
    candidate_surface: str,
) -> tuple[str, ...]:
    ordinal_classes = _question_ordinal_classes(original_question)
    candidate_tokens = _scope_tokens(candidate_surface)
    return tuple(
        ordinal_class
        for ordinal_class in ordinal_classes
        if not any(
            _ordinal_surface_matches(token, (ordinal_class,))
            for token in candidate_tokens
        )
    )


def _question_scope_modifier_issue(
    original_question: str,
    evidence_span: str,
) -> str | None:
    """Require explicit ordinal scope modifiers to survive evidence binding.

    Relation paraphrases such as ``come from``/``was born in`` or
    ``won``/``defeated`` cannot be validated by surface overlap alone.  The
    semantic Verifier remains responsible for that entailment.  This
    answer-free gate instead rejects a read that drops an explicit ordinal
    scope restriction, which is the measured ``first published`` versus
    ``introduced`` failure, without encoding dataset answers or a retrieval
    recipe.
    """

    ordinal_classes = _question_ordinal_classes(original_question)
    evidence_tokens = _scope_tokens(evidence_span)
    missing = tuple(
        ordinal_class
        for ordinal_class in ordinal_classes
        if not any(
            _ordinal_surface_matches(token, (ordinal_class,))
            for token in evidence_tokens
        )
        and not (
            ordinal_class == "first"
            and _receipt_first_in_sequence_onset(evidence_span)
        )
    )
    if not missing:
        return None
    return (
        "Evidence Retriever evidence_span does not preserve the original "
        f"question's ordinal scope modifier(s) {missing!r}"
    )


def _candidate_question_content_tokens(question: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _scope_tokens(question)
        if token not in _RELATION_CONTEXT_STOPWORDS
        and token not in _ORDINAL_SCOPE_CANONICAL
        and len(token) > 1
    )


def _candidate_matched_question_tokens(
    question_tokens: Sequence[str],
    candidate_surface: str,
) -> frozenset[str]:
    candidate_tokens = _scope_tokens(candidate_surface)
    return frozenset(
        question_token
        for question_token in question_tokens
        if any(
            _token_matches_relation(question_token, candidate_token)
            for candidate_token in candidate_tokens
        )
    )


def _public_search_candidate_compatibility(
    *,
    original_question: str,
    title: str,
    snippet: str,
) -> tuple[bool, int, int, int]:
    """Rank public search hits by entity/relation/qualifier compatibility.

    This is an answer-free action-domain projection over fields already
    published by SkillFlow's RetrievalIndex.  For ordinal questions, a strong
    hit must bind the public question entity/topic anchor, every recognized
    target-relation class (including a controlled alias), and the ordinal in
    the same ``title + one snippet clause`` unit.  This prevents a song or
    performer's local ``first hit`` from masquerading as an entity-global
    publication fact merely because other query words occur elsewhere in the
    passage snippet.  Non-ordinal questions are ordered by overlap but never
    narrowed here.
    """

    question_tokens = _candidate_question_content_tokens(original_question)
    title_matches = _candidate_matched_question_tokens(question_tokens, title)
    snippet_matches = _candidate_matched_question_tokens(question_tokens, snippet)
    ordinal_classes = _question_ordinal_classes(original_question)
    entity_anchor_tokens = _question_entity_anchor_tokens(original_question)
    required_relation_classes = frozenset(
        _relation_alias_surfaces_in(original_question)
    )
    best_clause_matches = 0
    qualifier_clause_compatible = False
    for clause in _evidence_proposition_clauses(snippet):
        clause_tokens = _scope_tokens(clause)
        title_and_clause = f"{title} {clause}"
        clause_matches = _candidate_matched_question_tokens(
            question_tokens,
            title_and_clause,
        )
        best_clause_matches = max(best_clause_matches, len(clause_matches))
        candidate_relation_classes = frozenset(
            _relation_alias_surfaces_in(title_and_clause)
        )
        if (
            ordinal_classes
            and entity_anchor_tokens
            and required_relation_classes
            and _surface_binds_entity_anchor(
                title_and_clause,
                entity_anchor_tokens,
            )
            and required_relation_classes <= candidate_relation_classes
            and all(
                any(
                    _ordinal_surface_matches(token, (ordinal_class,))
                    for token in clause_tokens
                )
                for ordinal_class in ordinal_classes
            )
        ):
            qualifier_clause_compatible = True
    strong = bool(ordinal_classes) and qualifier_clause_compatible
    return (
        strong,
        best_clause_matches,
        len(title_matches | snippet_matches),
        len(title_matches),
    )


@dataclass(frozen=True, slots=True)
class _RequiredEvidenceState:
    """Public SkillFlow search/read state used by the QA action mask."""

    required: bool
    search_queries: tuple[str, ...]
    normalized_search_queries: tuple[str, ...]
    search_top_ks: tuple[int, ...]
    search_attempt_count: int
    strategy_progress_count: int
    strategy_semantics: tuple[bool, ...]
    strategy_proofs: tuple[_FactualRetrievalStrategyProof, ...]
    retrieval_attempts: tuple[_FactualRetrievalAttemptRecord, ...]
    recall_expansion_count: int
    successful_search_hit_counts: tuple[int, ...]
    tool_error_count: int
    searched_passage_ids: tuple[str, ...]
    latest_search_passage_ids: tuple[str, ...]
    read_passage_ids: tuple[str, ...]
    upstream_read_passage_ids: tuple[str, ...]
    dispatched_tool_calls: int
    latest_successful_operation: str | None
    semantic_repair_kind: str | None
    semantic_repair_error_code: str | None
    semantic_repair_attempt_count: int
    semantic_repair_observation_index: int
    location_containment_repair_anchor: str | None
    location_containment_repair_queries: tuple[str, ...]
    location_containment_repair_top_ks: tuple[int, ...]
    location_containment_repair_hit_counts: tuple[int, ...]
    location_containment_repair_candidate_ids: tuple[str, ...]
    location_containment_repair_search_count: int
    location_containment_repair_read_count: int

    @property
    def unread_passage_ids(self) -> tuple[str, ...]:
        read = frozenset(self.read_passage_ids)
        return tuple(
            passage_id
            for passage_id in self.searched_passage_ids
            if passage_id not in read
        )

    @property
    def latest_unread_passage_ids(self) -> tuple[str, ...]:
        read = frozenset(self.read_passage_ids)
        return tuple(
            passage_id
            for passage_id in self.latest_search_passage_ids
            if passage_id not in read
        )

    @property
    def successful_read_count(self) -> int:
        return len(
            dict.fromkeys(
                (*self.read_passage_ids, *self.upstream_read_passage_ids)
            )
        )

    @property
    def strategy_semantics_verified(self) -> bool:
        return (
            self.strategy_progress_count > 0
            and len(self.strategy_semantics) == self.strategy_progress_count
            and all(self.strategy_semantics)
        )

    @property
    def verified_strategy_coverage(self) -> tuple[str, ...]:
        """Return the unordered set of receipt-verified strategy labels."""

        covered = {
            proof.strategy for proof in self.strategy_proofs if proof.verified
        }
        return tuple(
            strategy
            for strategy in _FACTUAL_QA_RETRIEVAL_STRATEGIES
            if strategy in covered
        )

    @property
    def missing_strategy_coverage(self) -> tuple[str, ...]:
        covered = frozenset(self.verified_strategy_coverage)
        return tuple(
            strategy
            for strategy in _FACTUAL_QA_RETRIEVAL_STRATEGIES
            if strategy not in covered
        )

    @property
    def retrieval_attempts_verified(self) -> bool:
        return (
            bool(self.retrieval_attempts)
            and len(self.retrieval_attempts) == self.search_attempt_count
            and all(attempt.verified for attempt in self.retrieval_attempts)
        )


class _RetrievalIndex(Protocol):
    @property
    def manifest(self) -> object:
        ...

    def search(self, query: str, *, limit: int) -> Sequence[object]:
        ...

    def read(self, passage_id: str) -> object:
        ...

    def close(self) -> None:
        ...


class _ThreadAffineRetrievalWorker:
    """Necessary adapter for SkillFlow SQLite's connection-thread affinity."""

    _SQLITE_PROGRESS_INTERVAL = 10_000

    def __init__(self, retrieval_index_class: object, index_path: Path) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="flowsteer-qa-retrieval",
        )
        try:
            self._index = self._executor.submit(
                getattr(retrieval_index_class, "open"), index_path
            ).result()
            self.manifest = self._executor.submit(
                lambda: self._index.manifest
            ).result()
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._closed = False

    def _cancellable_call(
        self,
        operation: Callable[[], object],
        cancel_requested: Event,
    ) -> object:
        """Run one upstream call with cooperative SQLite cancellation.

        SkillFlow keeps a private read-only SQLite connection on its
        ``RetrievalIndex``.  The outer ToolRegistry already supplies the
        authoritative timeout with ``asyncio.wait_for``; this progress handler
        only makes cancellation reach an FTS query that is already running on
        the dedicated thread.  It does not change the query, ranking, top-k or
        public observation.
        """

        connection = getattr(self._index, "_connection", None)
        set_progress_handler = getattr(
            connection,
            "set_progress_handler",
            None,
        )
        if not callable(set_progress_handler):
            return operation()
        set_progress_handler(
            cancel_requested.is_set,
            self._SQLITE_PROGRESS_INTERVAL,
        )
        try:
            return operation()
        finally:
            set_progress_handler(None, 0)

    async def _await_cancellable(
        self,
        operation: Callable[[], object],
    ) -> object:
        loop = asyncio.get_running_loop()
        cancel_requested = Event()
        future = loop.run_in_executor(
            self._executor,
            self._cancellable_call,
            operation,
            cancel_requested,
        )
        try:
            # Keep the executor Future live long enough for the progress
            # handler to observe cancellation and clear itself on its owning
            # thread.  The caller still receives the original CancelledError,
            # which asyncio.wait_for converts into the existing TimeoutError
            # Tool receipt.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            cancel_requested.set()
            if callable(
                getattr(
                    getattr(self._index, "_connection", None),
                    "set_progress_handler",
                    None,
                )
            ):
                try:
                    await asyncio.shield(future)
                except BaseException:
                    pass
            else:
                future.cancel()
            raise

    async def search(self, query: str, *, limit: int) -> Sequence[object]:
        return await self._await_cancellable(
            lambda: self._index.search(query, limit=limit)
        )

    async def read(self, passage_id: str) -> object:
        return await self._await_cancellable(
            lambda: self._index.read(passage_id)
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._index.close).result()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"retrieval manifest {field_name} must be non-empty text")
    return value.strip()


def _index_identity(index: _RetrievalIndex) -> dict[str, object]:
    """Project the frozen public-corpus identity without its content digest."""

    manifest = index.manifest
    return {
        "source": _required_text(
            getattr(manifest, "corpus_name", None), field_name="corpus_name"
        ),
        "corpus_version": _required_text(
            getattr(manifest, "corpus_version", None),
            field_name="corpus_version",
        ),
        "index_id": _required_text(
            getattr(manifest, "index_id", None), field_name="index_id"
        ),
        "index_format": _required_text(
            getattr(manifest, "format", None), field_name="format"
        ),
        "retrieval_backend": _required_text(
            getattr(manifest, "retrieval_backend", None),
            field_name="retrieval_backend",
        ),
    }


def _validate_action(request: ToolRequest, expected_action: str) -> None:
    if request.action != expected_action:
        raise ValueError(
            f"{expected_action} backend received incompatible action {request.action!r}"
        )


@dataclass(frozen=True, slots=True)
class QASearchToolBackend:
    """Execute SkillFlow ``RetrievalIndex.search`` without evaluator access."""

    index: _RetrievalIndex
    index_identity: Mapping[str, object]

    async def invoke(self, request: ToolRequest) -> ToolResult:
        # SkillFlow opens SQLite with its default thread affinity.  Keeping the
        # call on the event-loop thread matches QARetrievalEnvironment.execute
        # and avoids moving the connection through ``asyncio.to_thread``.
        _validate_action(request, "search")
        if set(request.arguments) != {"query", "limit"}:
            raise ValueError("search arguments must contain exactly query and limit")
        query = request.arguments["query"]
        limit = request.arguments["limit"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be non-empty text")
        if type(limit) is not int or limit < 1:
            raise ValueError("search limit must be a positive integer")

        raw_hits = self.index.search(query, limit=limit)
        hits = await raw_hits if inspect.isawaitable(raw_hits) else raw_hits
        public_hits = [
            {
                "passage_id": str(getattr(hit, "passage_id")),
                "document_id": str(getattr(hit, "document_id")),
                "title": str(getattr(hit, "title")),
                "snippet": str(getattr(hit, "snippet")),
                "rank": int(getattr(hit, "rank")),
            }
            for hit in hits
        ]
        return ToolResult(
            {
                "operation": "search",
                "retrieval_index": dict(self.index_identity),
                "query": query,
                "top_k": limit,
                "passage_ids": [hit["passage_id"] for hit in public_hits],
                "hits": public_hits,
            }
        )


@dataclass(frozen=True, slots=True)
class QAReadToolBackend:
    """Execute SkillFlow ``RetrievalIndex.read`` and return one public passage."""

    index: _RetrievalIndex
    index_identity: Mapping[str, object]

    async def invoke(self, request: ToolRequest) -> ToolResult:
        # See QASearchToolBackend.invoke for the upstream SQLite thread
        # affinity preserved by this async adapter boundary.
        _validate_action(request, "read")
        if set(request.arguments) != {"passage_id"}:
            raise ValueError("read arguments must contain exactly passage_id")
        passage_id = request.arguments["passage_id"]
        if not isinstance(passage_id, str) or not passage_id.strip():
            raise ValueError("read passage_id must be non-empty text")

        raw_passage = self.index.read(passage_id)
        passage = (
            await raw_passage if inspect.isawaitable(raw_passage) else raw_passage
        )
        public_passage = {
            "passage_id": str(getattr(passage, "passage_id")),
            "document_id": str(getattr(passage, "document_id")),
            "title": str(getattr(passage, "title")),
            "text": str(getattr(passage, "text")),
        }
        return ToolResult(
            {
                "operation": "read",
                "retrieval_index": dict(self.index_identity),
                "passage_id": public_passage["passage_id"],
                "passage": public_passage,
            }
        )


@dataclass(frozen=True, slots=True)
class QARetrievalToolBackend:
    """Dispatch SkillFlow's unified QA retrieval action domain."""

    index: _RetrievalIndex
    index_identity: Mapping[str, object]

    async def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action == "search":
            return await QASearchToolBackend(
                self.index,
                self.index_identity,
            ).invoke(request)
        if request.action == "read":
            return await QAReadToolBackend(
                self.index,
                self.index_identity,
            ).invoke(request)
        raise ValueError(
            f"retrieval backend received unsupported action {request.action!r}"
        )


class QARetrievalReactExecutionAdapter(ToolReactExecutionAdapter):
    """Bounded QA retrieval with canonical search-to-read admission."""

    def __init__(
        self,
        *,
        gateway: AgentGateway,
        tool_registry: ToolRegistry,
        max_turns: int,
        max_tool_calls: int,
        max_action_tokens: int = 512,
        task_type: str | None = None,
        completion_policy: str = "required_tool_call",
        sampling_base_seed: int | None = None,
        sampling_coordinate: ScientificSamplingCoordinate | None = None,
    ) -> None:
        if task_type not in {None, "multi_hop_qa", "factual_qa"}:
            raise ValueError("QA task_type must be multi_hop_qa, factual_qa, or None")
        if completion_policy not in {
            "optional",
            "required_tool_call",
            "required_evidence",
        }:
            raise ValueError(
                "QA completion_policy must be optional, required_tool_call, "
                "or required_evidence"
            )
        super().__init__(
            gateway=gateway,
            tool_registry=tool_registry,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_action_tokens=max_action_tokens,
            sampling_base_seed=sampling_base_seed,
            sampling_coordinate=sampling_coordinate,
        )
        self._task_type = task_type
        self._completion_policy = completion_policy
        self._retrieval_completion_required: ContextVar[bool] = ContextVar(
            f"qa_retrieval_completion_required_{id(self)}",
            default=False,
        )
        self._semantic_reasoner_question: ContextVar[str | None] = ContextVar(
            f"qa_semantic_reasoner_question_{id(self)}",
            default=None,
        )
        self._semantic_reasoner_protocol: ContextVar[str | None] = ContextVar(
            f"qa_semantic_reasoner_protocol_{id(self)}",
            default=None,
        )
        self._semantic_evidence_retriever_question: ContextVar[
            str | None
        ] = ContextVar(
            f"qa_semantic_evidence_retriever_question_{id(self)}",
            default=None,
        )
        self._semantic_evidence_retriever_protocol: ContextVar[
            str | None
        ] = ContextVar(
            f"qa_semantic_evidence_retriever_protocol_{id(self)}",
            default=None,
        )
        self._semantic_upstream_tool_receipts: ContextVar[
            tuple[Mapping[str, object], ...]
        ] = ContextVar(
            f"qa_semantic_upstream_tool_receipts_{id(self)}",
            default=(),
        )

    def _unified_factual_protocol(self, request: AgentRequest) -> bool:
        return (
            self._task_type == "factual_qa"
            and request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
        )

    @staticmethod
    def _direct_upstream_tool_receipts(
        request: AgentRequest,
    ) -> tuple[Mapping[str, object], ...]:
        """Return direct predecessor receipts without changing own Tool budget."""

        return tuple(
            receipt
            for message in request.upstream
            for receipt in message.tool_receipts
            if isinstance(receipt, Mapping)
        )

    def _validated_upstream_evidence_receipts(
        self,
        request: AgentRequest,
    ) -> tuple[Mapping[str, object], ...]:
        """Return receipts carried by valid unified factual evidence artifacts.

        TriviaQA Reasoner completion must not be admitted by an unrelated read
        receipt alone.  Validate each direct predecessor and peer-revision
        artifact through the existing Evidence Retriever completion boundary;
        only receipts belonging to an artifact with complete entity, relation,
        evidence-span, passage-id, and strict read-receipt grounding enter the
        Reasoner's evidence context.
        """

        if (
            not self._unified_factual_protocol(request)
            or (request.agent.role_family or "").casefold() != "reasoner"
        ):
            return ()
        messages = (
            *request.upstream,
            *((request.peer_draft,) if request.peer_draft is not None else ()),
        )
        original_question = qa_question_scope(request.problem)
        validated_receipts: list[Mapping[str, object]] = []
        for message in messages:
            message_receipts = tuple(
                receipt
                for receipt in message.tool_receipts
                if isinstance(receipt, Mapping)
            )
            if self._evidence_retriever_completion_issue(
                original_question=original_question,
                artifact=message.content,
                tool_receipts=message_receipts,
            ) is not None:
                continue
            validated_receipts.extend(message_receipts)
        return tuple(validated_receipts)

    @staticmethod
    def _successful_read_receipt(receipt: Mapping[str, object]) -> bool:
        from .agent_workflow_env import AgentWorkflowEnv

        return AgentWorkflowEnv._successful_read_receipt(
            receipt,
            QA_RETRIEVAL_TOOL_ID,
        )

    def _hotpot_tool_plan_exhausted(
        self,
        request: AgentRequest,
        state: _RequiredEvidenceState,
    ) -> bool:
        """Return a typed bounded-retrieval diagnosis, never a task oracle."""

        if (
            request.semantic_protocol != "hotpotqa_verified_answer_slot_v1"
            or (request.agent.role_family or "").casefold() != "reasoner"
            or state.semantic_repair_kind != "evidence"
        ):
            return False
        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        if (
            state.latest_successful_operation == "search"
            and state.latest_unread_passage_ids
            and remaining_tool_calls >= 1
        ):
            return False
        return remaining_tool_calls < 2

    @staticmethod
    def _factual_retrieval_strategy(search_attempt_count: int) -> str:
        index = min(
            max(search_attempt_count, 0),
            len(_FACTUAL_QA_RETRIEVAL_STRATEGIES) - 1,
        )
        return _FACTUAL_QA_RETRIEVAL_STRATEGIES[index]

    @staticmethod
    def _factual_search_limit(search_attempt_count: int) -> int:
        index = min(
            max(search_attempt_count, 0),
            len(_FACTUAL_QA_SEARCH_LIMITS) - 1,
        )
        return _FACTUAL_QA_SEARCH_LIMITS[index]

    @staticmethod
    def _factual_exhaustion_diagnosis(
        *,
        strategy_progress_count: int,
        strategy_semantics_verified: bool,
        successful_search_hit_counts: Sequence[int],
        tool_error_count: int,
        verified_strategy_coverage: Sequence[str] | None = None,
    ) -> str:
        """Classify the measured bounded retrieval schedule.

        The label is scoped to this frozen public knowledge base and bounded
        strategy schedule.  It does not claim that the fact is absent from all
        possible corpora.  Search hits alone do not establish coverage: after
        every strategy has completed, the required outcome is an entity- and
        relation-aligned successful read receipt.
        """

        required_strategy_coverage = frozenset(
            _FACTUAL_QA_RETRIEVAL_STRATEGIES
        )
        measured_strategy_coverage = (
            frozenset(verified_strategy_coverage)
            if verified_strategy_coverage is not None
            else None
        )
        if (
            strategy_progress_count
            < len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
            or tool_error_count > 0
            or not strategy_semantics_verified
            or (
                measured_strategy_coverage is not None
                and not required_strategy_coverage
                <= measured_strategy_coverage
            )
        ):
            return _RETRIEVAL_STRATEGY_FAILURE
        if any(hit_count > 0 for hit_count in successful_search_hit_counts):
            return _RETRIEVAL_RECALL_FAILURE
        return _KNOWLEDGE_BASE_COVERAGE_FAILURE

    @staticmethod
    def _semantic_rejection_kind(public_error_code: object) -> str | None:
        """Classify public completion feedback without inspecting hidden labels."""

        if not isinstance(public_error_code, str):
            return None
        if (
            public_error_code == _QA_MISSING_EVIDENCE_ERROR
            or public_error_code.startswith(
                _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
            )
            or public_error_code.startswith(
                _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
            )
        ):
            return "evidence"
        if public_error_code.startswith(
            _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
        ) or public_error_code.startswith(_QA_SEMANTIC_STRUCTURE_ERROR_PREFIX):
            return "structure"
        if public_error_code == _KNOWLEDGE_BASE_COVERAGE_FAILURE:
            return "coverage"
        if public_error_code == _RETRIEVAL_RECALL_FAILURE:
            return "recall"
        if public_error_code == _RETRIEVAL_STRATEGY_FAILURE:
            return "strategy"
        return None

    @staticmethod
    def _semantic_repair_instruction(repair_kind: str) -> str:
        """Return the public SkillFlow continuation instruction for one fault."""

        if repair_kind == "structure":
            return (
                "Preserve all successful qa-retrieval read evidence and every "
                "semantic field not implicated by this public_error_code. Repair "
                "only the diagnosed structured semantic artifact fields, then "
                "emit a complete action; do not add a search or read."
            )
        if repair_kind == "evidence":
            return (
                "Preserve the current semantic work and use the admitted "
                "qa-retrieval search/read continuation to obtain the missing "
                "evidence or provenance before completing again."
            )
        if repair_kind == "coverage":
            return (
                "The bounded retrieval strategies and Tool budget did not produce "
                "evidence that binds the target entity and relation. Do not guess "
                "or fabricate an answer or evidence."
            )
        if repair_kind == "recall":
            return (
                "Distinct bounded retrieval queries returned public results, but "
                "no read receipt bound the target entity and relation. Preserve "
                "all successful Tool receipts, repair entity/relation retrieval "
                "or augment with a replacement Retriever, and do not guess."
            )
        if repair_kind == "strategy":
            return (
                "The bounded execution ended before all distinct retrieval-query "
                "strategies completed. Preserve public Tool receipts and repair "
                "query reformulation or Tool execution; do not guess or fabricate "
                "evidence."
            )
        raise ValueError(f"unsupported semantic repair kind {repair_kind!r}")

    @classmethod
    def _public_semantic_repair_instruction(
        cls,
        public_error_code: object,
    ) -> str | None:
        """Return the exact public repair attached to one QA Observation.

        SkillFlow carries the same public Action--Observation record into the
        next bounded turn and into an exhausted continuation receipt. Keep the
        diagnosis in one helper so the model-visible continuation and the
        outer FlowSteer Canvas cannot disagree about the required repair.
        """

        if isinstance(public_error_code, str) and (
            "candidate_answer is the verified year-to-decade normalization"
            in public_error_code
        ):
            return (
                "Preserve every successful qa-retrieval read receipt, "
                "question_scope, candidate_answer, evidence proposition, and "
                "multi_hop_chain. Do not search or read again. Locate the "
                "zero-based evidence_propositions index and the subject or "
                "object_or_attribute_value field containing the source year or "
                "date whose deterministic year-to-decade normalization equals "
                "candidate_answer; set answer_slot.proposition_index and "
                "answer_slot.answer_field to that source field. Preserve the "
                "normalized candidate_answer, repair only those answer_slot "
                "fields, then emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Reasoner answer_slot.answer_field selects" in public_error_code
        ):
            return (
                "Preserve every successful qa-retrieval read receipt, "
                "question_scope, candidate_answer, evidence proposition, and "
                "multi_hop_chain. Do not search or read again. Locate the "
                "zero-based evidence_propositions index and the subject or "
                "object_or_attribute_value field whose value equals "
                "candidate_answer exactly; set answer_slot.proposition_index "
                "and answer_slot.answer_field to that same proposition field. "
                "Repair only those answer_slot fields, then emit a complete "
                "action."
            )
        if isinstance(public_error_code, str) and re.search(
            r"Reasoner evidence_propositions\[\d+\]\."
            r"(?:subject|object_or_attribute_value) is not grounded",
            public_error_code,
        ):
            return (
                "Preserve every successful qa-retrieval read receipt, the exact "
                "evidence_span, question_scope, candidate_answer, answer_slot, "
                "and all unimplicated propositions. Do not search or read again. "
                "Repair only the named proposition argument by copying its exact "
                "subject or object_or_attribute_value surface from that same "
                "evidence_span, then emit a complete action."
            )
        if isinstance(public_error_code, str) and re.search(
            r"Reasoner evidence_propositions\[\d+\]\.relation is not grounded",
            public_error_code,
        ):
            return (
                "Preserve every successful qa-retrieval read receipt, the exact "
                "evidence_span, question_scope, candidate_answer, answer_slot, "
                "and all unimplicated propositions. Do not search or read again. "
                "Repair only the named proposition relation by copying the exact "
                "predicate surface asserted in that same evidence_span, then "
                "emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Reasoner candidate_answer must copy the proposition argument "
            "identified by answer_slot.proposition_index and answer_field "
            "exactly"
        ) in public_error_code:
            return (
                "Preserve every successful qa-retrieval read receipt, "
                "question_scope, evidence proposition, and multi_hop_chain. "
                "Do not search or read again. Bind answer_slot to the "
                "zero-based evidence_propositions index and the subject or "
                "object_or_attribute_value field that fills the original "
                "question's requested answer slot, then copy that field value "
                "character-for-character into candidate_answer. Repair only "
                "candidate_answer and answer_slot, then emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "entity_identity.evidence_surface is not supported by the "
            "cited passage title identity chain"
        ) in public_error_code:
            return (
                "Preserve the same successful read receipt, passage_id, passage "
                "title, target_relation, evidence_span, and proposition. Repair "
                "only entity_identity: keep question_surface on the original "
                "question entity and set evidence_surface to the complete public "
                "passage-title identity when that title occurs inside the exact "
                "evidence_span and the unchanged proposition argument. Do not "
                "shorten it to a surname, copy a descriptor, or replace an "
                "expanded full-name proposition argument with the shorter title. "
                "Emit a complete action; do not search or read again."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever entity_identity.evidence_surface does not "
            "occur in evidence_span"
        ) in public_error_code:
            return (
                "Preserve the same successful read receipt, passage_id, passage "
                "title, question_surface, target_relation, and evidence "
                "proposition. Do not search or read again. If that read body "
                "begins with a coreferential pronoun whose public passage title "
                "binds the original question entity, keep the exact body span, "
                "copy that pronoun into entity_identity.evidence_surface and the "
                "same proposition argument, and do not copy the title into the "
                "body evidence. Otherwise, from that same read receipt, expand "
                "evidence_span only as needed to include both the exact entity "
                "mention and the sentence expressing the requested relation, "
                "then copy that exact mention into evidence_surface. Keep the "
                "span contiguous and receipt-grounded, and emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever answer-bearing entity surface is a strict "
            "subset of the resolved passage-title identity"
        ) in public_error_code:
            return (
                "Preserve the original question, requested relation, every "
                "successful qa-retrieval read receipt, and all already valid "
                "semantic fields. First select an existing successful read whose "
                "contiguous body span contains the complete public passage-title "
                "identity inside the proposition field that answers the original "
                "wh-dependency. Copy that complete title identity into "
                "entity_identity.evidence_surface while preserving the exact "
                "receipt-grounded proposition argument, including any expanded "
                "full-name surface. If no preserved read contains that binding, "
                "continue the admitted bounded retrieval rather than guessing or "
                "emitting an ambiguous short-name answer."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever target_relation must preserve the "
            "requested relation from the original question"
        ) in public_error_code:
            return (
                "Preserve the same successful read receipt, passage_id, passage "
                "title, question_scope, entity_identity, answer_type_constraint, "
                "evidence_span, and evidence_proposition. Do not search or read "
                "again. Repair only target_relation by restoring the requested "
                "relation and its scope modifiers from the unchanged original "
                "question; do not change the receipt-grounded proposition "
                "predicate or any other structured field. Then emit a complete "
                "action."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever entity_identity.question_surface does not "
            "occur in the original question"
        ) in public_error_code:
            return (
                "Preserve the same successful read receipt, passage_id, passage "
                "title, evidence_surface, target_relation, evidence_span, and "
                "evidence proposition. Do not search or read again. Copy one "
                "concise exact entity or event mention from the unchanged "
                "original question into entity_identity.question_surface; do "
                "not expand it with title-only tokens. Repair only "
                "question_surface, then emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever entity_identity.question_surface must be a "
            "question-side entity/event anchor"
        ) in public_error_code:
            return (
                "Preserve the same successful read receipt, passage_id, passage "
                "title, evidence_surface, target_relation, evidence_span, and "
                "evidence proposition. Do not search or read again. Replace "
                "only entity_identity.question_surface with one concise exact "
                "entity or event mention from the unchanged original question, "
                "excluding the wh-word or wh-phrase, then emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever entity_identity.evidence_surface must bind to "
            "exactly one evidence_proposition relation argument"
        ) in public_error_code:
            return (
                "Preserve the same successful read receipt, passage_id, passage "
                "title, question_surface, target_relation, evidence_span, and "
                "evidence_surface. Do not search or read again. Copy that exact "
                "evidence_surface into exactly one of evidence_proposition.subject "
                "or evidence_proposition.object_or_attribute_value according to "
                "the relation expressed by the same evidence_span; keep the "
                "other argument distinct and receipt-grounded. Repair only the "
                "implicated proposition argument, then emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Reasoner requested-relation proposition has no deterministic "
            "entity binding"
        ) in public_error_code:
            return (
                "Preserve every successful qa-retrieval read receipt, "
                "question_scope, evidence_span, requested relation, and all "
                "unimplicated propositions; do not add a search or read. In the "
                "proposition that expresses the requested relation, bind one "
                "subject or object_or_attribute_value argument to the original "
                "question-side entity using an exact evidence-span surface or a "
                "passage-title-supported alias/coreference; copy the other "
                "relation argument from that same span. Keep candidate_answer "
                "and answer_slot attached to the field that actually answers the "
                "question. Repair only the implicated proposition and binding "
                "fields, then emit a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Reasoner answer-bearing proposition has no deterministic entity "
            "binding: answer_slot is not reachable"
        ) in public_error_code:
            return (
                "Preserve every successful qa-retrieval read receipt, the "
                "question-side entity proposition, requested relation, and "
                "candidate_answer. Do not search or read again. Repair only the "
                "receipt-grounded evidence propositions and answer_slot so that "
                "successive propositions share an explicit entity surface from "
                "the question-side entity through to the proposition field equal "
                "to candidate_answer; do not invent an alias or bridge. Then emit "
                "a complete action."
            )
        if isinstance(public_error_code, str) and (
            "Evidence Retriever evidence_span has no typography-canonical "
            "lexical match"
        ) in public_error_code:
            return (
                "Preserve the cited passage_id and all successful Tool receipts. "
                "Copy one contiguous exact evidence_span from that same public "
                "qa-retrieval read receipt, allowing only typography/whitespace "
                "canonicalization; do not paraphrase, concatenate spans, search, "
                "or read again."
            )
        if isinstance(public_error_code, str) and (
            _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING in public_error_code
        ):
            return (
                "Preserve the receipt-grounded first-hop location proposition. "
                "Use the admitted qa-retrieval search/read continuation to resolve "
                "the leading locality's identity and geographic type from a public "
                "read body. If that body types the leading locality itself as a city "
                "or town inside the question's named scope, preserve it and bind the "
                "answer slot to that proposition's subject. Only when the read body "
                "types it as a district, suburb, or sublocality, or explicitly states "
                "that it is part of or belongs to a city or town in scope, bind the "
                "answer slot to the containing proposition's object. A passage title "
                "may support identity only; it cannot supply type, containment, or "
                "scope. Retain both receipt-backed hops in multi_hop_chain, and do "
                "not guess a parent, promote a comma suffix, or discard first-hop "
                "evidence. Encode the first hop and the location-resolution hop "
                "as two distinct evidence_propositions. Set answer_slot to the "
                "proposition field that equals candidate_answer; do not leave the "
                "slot bound to the unresolved first-hop locality."
            )
        if isinstance(public_error_code, str) and public_error_code.startswith(
            _QA_LOCATION_RELATION_GROUNDING_QUERY_MISMATCH
        ):
            return (
                "Preserve the receipt-grounded first-hop proposition and all "
                "successful Tool receipts. Reformulate only the current search "
                "query so it contains the public first-hop locality anchor, the "
                "original question's named geographic scope, and a geographic "
                "type or administrative-containment relation such as city, town, "
                "district, suburb, part of, or belongs to. Do not restart from "
                "the original person/entity query, guess a containing place, or "
                "discard the first-hop evidence."
            )
        if public_error_code == "qa_retrieval_duplicate_normalized_query":
            return (
                "Preserve all successful Tool receipts. Use a distinct normalized "
                "entity-and-relation query with a distinct FTS term set for the "
                "current retrieval strategy; reordering the same terms is not a "
                "new strategy. A "
                "repeat of only the latest query is admitted at each strictly "
                "larger top_k required by the bounded action schema as recall "
                "expansion, but it does not advance retrieval-strategy progress. Do not "
                "repeat a prior (query, top_k) pair or cycle to an older query."
            )
        if public_error_code == _RETRIEVAL_QUERY_CANDIDATE_INJECTION:
            return (
                "Preserve the original question entity, requested relation, and "
                "all successful Tool receipts. Remove every question-external "
                "numeric, year, or decade candidate from the query, then perform "
                "the required scope-preserving retrieval strategy. Reformulate "
                "the entity and relation only; do not guess the answer in the query."
            )
        if public_error_code == _RETRIEVAL_QUERY_STRATEGY_LABEL_INJECTION:
            return (
                "Preserve all successful Tool receipts and the required retrieval "
                "strategy. Express that strategy by changing the lexical entity or "
                "relation surface in query; never copy orchestration labels such as "
                "spelling normalization, alias expansion, entity disambiguation, "
                "query rewriting, or synonym into the Tool query."
            )
        if public_error_code == _RETRIEVAL_QUERY_SCOPE_MODIFIER_LOSS:
            return (
                "Preserve all successful Tool receipts, the original entity, and "
                "the requested relation. Restore every explicit ordinal scope "
                "class from the original question using that surface or a "
                "controlled equivalent in the retrieval query; do "
                "not broaden `first`, `last`, `earliest`, or a corresponding "
                "ordinal into an unscoped relation."
            )
        if isinstance(public_error_code, str) and public_error_code.startswith(
            _RETRIEVAL_QUERY_NAMED_SCOPE_LOSS
        ):
            return (
                "Preserve all successful Tool receipts, the original entity, and "
                "the requested relation. Restore every explicit named scope "
                "constraint copied from the question and listed by the public "
                "error. Do not broaden a nationality, jurisdiction, language, or "
                "other named restriction into an unscoped relation."
            )
        if isinstance(public_error_code, str) and public_error_code.startswith(
            _RETRIEVAL_QUERY_RELATION_CLASS_LOSS
        ):
            return (
                "Preserve all successful Tool receipts, the ordered public entity "
                "anchor, and every ordinal scope modifier. Restore every missing "
                "question-derived strong relation class using one of the answer-free "
                "required_relation_surface_alternatives in the current public Tool "
                "continuation state. A generic relation head alone cannot replace a "
                "narrower multi-token relation; do not add an answer candidate or "
                "change the question scope."
            )
        if public_error_code == _RETRIEVAL_QUERY_ENTITY_ANCHOR_LOSS:
            return (
                "Preserve all successful Tool receipts. Restore the complete "
                "ordered question-derived entity anchor, including an adjacent "
                "question-side type noun, before changing the relation surface. "
                "Do not replace it with a returned subtype title or add an answer "
                "candidate."
            )
        if isinstance(public_error_code, str) and public_error_code.startswith(
            _RETRIEVAL_QUERY_STRATEGY_SEMANTICS_MISMATCH
        ):
            return (
                "Preserve all successful Tool receipts, the public entity anchor, "
                "every ordinal scope modifier, and every question-derived strong "
                "relation class. Choose one still-uncovered strategy from "
                "missing_strategy_coverage and express it through a distinct, "
                "answer-free adjacent query transition supported by the public "
                "question or the latest mirror-valid search receipt. Do not repeat a "
                "covered strategy while another remains. If the public error names "
                "remaining_relation_classes, replace at least one of those classes "
                "with a same-predicate alternative while preserving every other "
                "relation and scope constraint; consult "
                "remaining_relation_transformation_classes in the current public "
                "continuation state. Do not merely append an alternate "
                "surface, retain both surfaces, import a candidate answer, or reorder "
                "the existing FTS terms."
            )
        repair_kind = cls._semantic_rejection_kind(public_error_code)
        if repair_kind is not None:
            return cls._semantic_repair_instruction(repair_kind)
        return None

    @classmethod
    def _model_visible_observations(
        cls,
        observations: list[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Attach the public continuation repair to semantic rejections.

        SkillFlow carries Action--Observation feedback into the next model
        turn.  Keep the sampled invalid completion only in the trajectory and
        expose a diagnosis that says whether the existing read evidence must
        be augmented or only the structured completion must be repaired.
        """

        visible = ToolReactExecutionAdapter._model_visible_observations(
            observations
        )
        for observation in visible:
            public_error_code = observation.get("public_error_code")
            repair_instruction = cls._public_semantic_repair_instruction(
                public_error_code
            )
            if repair_instruction is not None:
                observation["repair_instruction"] = repair_instruction

        # SkillFlow persists every sampled Action--Observation turn in the
        # trajectory, but the next model input need not replay an unbounded run
        # of identical invalid-action observations. Preserve every Tool result
        # and every change of diagnosis while collapsing only consecutive
        # duplicate public errors.
        compacted: list[dict[str, object]] = []
        last_invalid_key: tuple[str, str] | None = None
        for observation in visible:
            status = observation.get("observation_status")
            public_error_code = observation.get("public_error_code")
            if (
                status in {"parse_error", "schema_invalid"}
                and isinstance(public_error_code, str)
            ):
                invalid_key = (str(status), public_error_code)
                if invalid_key == last_invalid_key:
                    prior = compacted[-1]
                    repeat_count = prior.get("repeat_count", 1)
                    prior["repeat_count"] = (
                        repeat_count + 1
                        if isinstance(repeat_count, int)
                        and not isinstance(repeat_count, bool)
                        else 2
                    )
                    continue
                last_invalid_key = invalid_key
            else:
                last_invalid_key = None
            compacted.append(observation)

        # A successful read action has already been consumed by SkillFlow's
        # bounded Action--Observation loop.  Preserve its public result in the
        # next model input, but do not replay the stale executable action: an
        # exact copy can otherwise be sampled again after a completion-field
        # repair.  The lossless trajectory and Tool receipts still retain the
        # original action, and unread search candidates remain available from
        # their search Observation.
        for observation in compacted:
            result = observation.get("result")
            executed_action = observation.get("executed_action")
            if (
                observation.get("observation_status") == "success"
                and isinstance(result, Mapping)
                and result.get("operation") == "read"
                and isinstance(executed_action, Mapping)
                and executed_action.get("kind") == "tool"
                and executed_action.get("resource_id")
                == QA_RETRIEVAL_TOOL_ID
                and executed_action.get("name") == "read"
            ):
                observation.pop("executed_action", None)
        return compacted

    def _required_evidence_state(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> _RequiredEvidenceState:
        required = (
            self._completion_policy == "required_evidence"
            and self._max_tool_calls > 0
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        )
        search_queries: list[str] = []
        successful_search_observations: list[Mapping[str, object]] = []
        normalized_search_queries: list[str] = []
        search_top_ks: list[int] = []
        successful_search_hit_counts: list[int] = []
        searched_passage_ids: list[str] = []
        latest_search_passage_ids: list[str] = []
        read_passage_ids: list[str] = []
        upstream_read_passage_ids: list[str] = []
        dispatched_tool_calls = 0
        tool_error_count = 0
        latest_successful_operation: str | None = None
        latest_successful_read_index = -1
        latest_semantic_rejection_index = -1
        latest_semantic_rejection_kind: str | None = None
        latest_semantic_rejection_code: str | None = None
        latest_location_containment_repair_anchor: str | None = None
        latest_location_containment_repair_error_code: str | None = None
        latest_location_containment_repair_observation_index = -1
        location_relation_grounding_pending = False
        location_containment_repair_queries: list[str] = []
        location_containment_repair_top_ks: list[int] = []
        location_containment_repair_hit_counts: list[int] = []
        location_containment_repair_search_count = 0
        location_containment_repair_read_count = 0
        # A direct Retriever predecessor supplies evidence to the Reasoner via
        # the public Agent message/receipt lineage.  Those reads admit semantic
        # completion but never consume this Reasoner's own bounded Tool budget.
        for receipt in self._semantic_upstream_tool_receipts.get():
            if not isinstance(receipt, Mapping) or not self._successful_read_receipt(
                receipt
            ):
                continue
            receipt_request = receipt.get("request")
            receipt_result = receipt.get("result")
            if (
                not isinstance(receipt_request, Mapping)
                or receipt_request.get("action") != "read"
                or not isinstance(receipt_result, Mapping)
            ):
                continue
            value = receipt_result.get("value", receipt_result)
            if not isinstance(value, Mapping) or value.get("operation") != "read":
                continue
            passage = value.get("passage")
            if (
                not isinstance(passage, Mapping)
                or not isinstance(passage.get("text"), str)
                or not passage["text"].strip()
            ):
                continue
            arguments = receipt_request.get("arguments")
            assert isinstance(arguments, Mapping)
            passage_id = arguments.get("passage_id")
            if (
                isinstance(passage_id, str)
                and passage_id.strip()
                and passage_id.strip() not in upstream_read_passage_ids
            ):
                upstream_read_passage_ids.append(passage_id.strip())
        for observation_index, observation in enumerate(observations):
            status = observation.get("observation_status")
            public_error_code = observation.get("public_error_code")
            semantic_rejection_kind = self._semantic_rejection_kind(
                public_error_code
            )
            if status == "schema_invalid" and semantic_rejection_kind is not None:
                latest_semantic_rejection_index = observation_index
                latest_semantic_rejection_kind = semantic_rejection_kind
                assert isinstance(public_error_code, str)
                latest_semantic_rejection_code = public_error_code
                if _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING in public_error_code:
                    executed_action = observation.get("executed_action")
                    if isinstance(executed_action, Mapping):
                        candidate_anchor = (
                            _location_containment_repair_anchor(
                                original_question=qa_question_scope(
                                    request.problem
                                ),
                                completion_action=executed_action,
                            )
                        )
                        if isinstance(candidate_anchor, str) and candidate_anchor:
                            if (
                                latest_location_containment_repair_anchor
                                is not None
                                and _canonical_location_surface(candidate_anchor)
                                != _canonical_location_surface(
                                    latest_location_containment_repair_anchor
                                )
                            ):
                                location_containment_repair_queries.clear()
                                location_containment_repair_top_ks.clear()
                                location_containment_repair_hit_counts.clear()
                                location_containment_repair_search_count = 0
                                location_containment_repair_read_count = 0
                            latest_location_containment_repair_anchor = (
                                candidate_anchor
                            )
                            latest_location_containment_repair_error_code = (
                                public_error_code
                            )
                            latest_location_containment_repair_observation_index = (
                                observation_index
                            )
                            # A later rejected completion over an already grounded
                            # containment read is a structured-artifact repair, not
                            # a reason to repeat retrieval.  Otherwise the typed
                            # relation-grounding generation remains active until a
                            # matching read succeeds.
                            location_relation_grounding_pending = (
                                location_containment_repair_read_count == 0
                            )
            if status not in {"success", "tool_error"}:
                continue
            executed_action = observation.get("executed_action")
            if isinstance(executed_action, Mapping):
                if (
                    executed_action.get("kind") == "tool"
                    and executed_action.get("resource_id") == QA_RETRIEVAL_TOOL_ID
                ):
                    dispatched_tool_calls += 1
                    if status == "tool_error":
                        tool_error_count += 1
            elif status == "success":
                # Unit fixtures and restored legacy public observations may
                # omit executed_action.  A successful retrieval result still
                # represents exactly one dispatched SkillFlow Tool action.
                dispatched_tool_calls += 1
            if status != "success":
                continue
            result = observation.get("result")
            if not isinstance(result, Mapping):
                continue
            if result.get("operation") == "search":
                location_relation_grounding_search = (
                    location_relation_grounding_pending
                )
                if location_relation_grounding_search:
                    location_containment_repair_search_count += 1
                latest_successful_operation = "search"
                latest_search_passage_ids = []
                query: object = result.get("query")
                top_k: object = result.get("top_k")
                if isinstance(executed_action, Mapping):
                    arguments = executed_action.get("arguments")
                    if isinstance(arguments, Mapping):
                        query = arguments.get("query", query)
                        top_k = arguments.get("limit", top_k)
                raw_ids = result.get("passage_ids")
                if isinstance(raw_ids, list):
                    for value in raw_ids:
                        if not isinstance(value, str) or not value.strip():
                            continue
                        passage_id = value.strip()
                        if passage_id not in searched_passage_ids:
                            searched_passage_ids.append(passage_id)
                        if passage_id not in latest_search_passage_ids:
                            latest_search_passage_ids.append(passage_id)
                # Strategy progress is based only on successful public search
                # Observations. Tool errors still consume the bounded Tool
                # budget above, but cannot masquerade as query reformulation.
                if (
                    not location_relation_grounding_search
                    and isinstance(query, str)
                    and query.strip()
                ):
                    query = query.strip()
                    search_queries.append(query)
                    successful_search_observations.append(observation)
                    normalized_search_queries.append(
                        _normalized_retrieval_query(query)
                    )
                    search_top_ks.append(
                        top_k if type(top_k) is int and top_k > 0 else 0
                    )
                    successful_search_hit_counts.append(
                        len(latest_search_passage_ids)
                    )
                elif (
                    location_relation_grounding_search
                    and isinstance(query, str)
                    and query.strip()
                ):
                    location_containment_repair_queries.append(query.strip())
                    location_containment_repair_top_ks.append(
                        top_k if type(top_k) is int and top_k > 0 else 0
                    )
                    location_containment_repair_hit_counts.append(
                        len(latest_search_passage_ids)
                    )
            elif (
                result.get("operation") == "read"
                and isinstance(result.get("passage"), Mapping)
                and isinstance(result["passage"].get("text"), str)
                and bool(result["passage"]["text"].strip())
            ):
                latest_successful_operation = "read"
                latest_successful_read_index = observation_index
                read_transition_verified, _ = _public_read_transition_mirror(
                    observation
                )
                if (
                    location_relation_grounding_pending
                    and read_transition_verified
                    and isinstance(
                        latest_location_containment_repair_anchor,
                        str,
                    )
                ):
                    passage = result["passage"]
                    assert isinstance(passage, Mapping)
                    passage_text = passage.get("text")
                    passage_title = passage.get("title")
                    named_scope = _explicit_named_geographic_scope(
                        qa_question_scope(request.problem)
                    )
                    anchor_aliases = _location_surface_component_aliases(
                        latest_location_containment_repair_anchor
                    )
                    public_surface = " ".join(
                        value
                        for value in (passage_title, passage_text)
                        if isinstance(value, str) and value.strip()
                    )
                    relation_grounded = bool(
                        isinstance(passage_text, str)
                        and isinstance(named_scope, str)
                        and any(
                            _contains_canonical_location_surface(
                                public_surface,
                                alias,
                            )
                            for alias in anchor_aliases
                        )
                        and _contains_canonical_location_surface(
                            passage_text,
                            named_scope,
                        )
                        and re.search(
                            r"\b(?:city|town)\b",
                            passage_text,
                            flags=re.IGNORECASE,
                        )
                    )
                    if relation_grounded:
                        location_containment_repair_read_count += 1
                        location_relation_grounding_pending = False
                raw_passage_id = result.get("passage_id")
                if not isinstance(raw_passage_id, str):
                    raw_passage_id = result["passage"].get("passage_id")
                if not isinstance(raw_passage_id, str) and isinstance(
                    executed_action, Mapping
                ):
                    arguments = executed_action.get("arguments")
                    if isinstance(arguments, Mapping):
                        raw_passage_id = arguments.get("passage_id")
                if isinstance(raw_passage_id, str) and raw_passage_id.strip():
                    passage_id = raw_passage_id.strip()
                    if passage_id not in read_passage_ids:
                        read_passage_ids.append(passage_id)
        distinct_queries: list[str] = []
        distinct_search_observations: list[Mapping[str, object]] = []
        distinct_term_set_signatures: list[tuple[str, ...]] = []
        for query, search_observation in zip(
            search_queries,
            successful_search_observations,
        ):
            signature = _retrieval_query_term_set_signature(query)
            if signature in distinct_term_set_signatures:
                continue
            distinct_term_set_signatures.append(signature)
            distinct_queries.append(query)
            distinct_search_observations.append(search_observation)
        strategy_proofs = _factual_strategy_proofs(
            original_question=qa_question_scope(request.problem),
            distinct_queries=distinct_queries,
            search_observations=distinct_search_observations,
        )
        strategy_semantics = tuple(
            proof.verified for proof in strategy_proofs
        )
        retrieval_attempts = _factual_retrieval_attempt_records(
            original_question=qa_question_scope(request.problem),
            search_observations=successful_search_observations,
        )
        completed_location_evidence_repair = bool(
            location_containment_repair_read_count > 0
            and isinstance(
                latest_location_containment_repair_error_code,
                str,
            )
            and latest_location_containment_repair_observation_index
            > latest_successful_read_index
        )
        unresolved_location_repair = bool(
            location_relation_grounding_pending
            and not completed_location_evidence_repair
            and isinstance(latest_location_containment_repair_anchor, str)
            and latest_location_containment_repair_anchor.strip()
            and isinstance(
                latest_location_containment_repair_error_code,
                str,
            )
        )
        if completed_location_evidence_repair:
            # The location-resolution Tool receipt already exists. A repeated
            # containment-lineage rejection now diagnoses only the Reasoner's
            # structured proposition/answer-slot binding; another retrieval
            # action cannot add the missing field alignment. Preserve both
            # reads and keep SkillFlow's next bounded turn completion-only.
            active_semantic_repair_kind = "structure"
            active_semantic_repair_error_code = (
                latest_location_containment_repair_error_code
            )
            active_semantic_rejection_index = (
                latest_location_containment_repair_observation_index
            )
        elif unresolved_location_repair:
            active_semantic_repair_kind = "evidence"
            active_semantic_repair_error_code = (
                latest_location_containment_repair_error_code
            )
            active_semantic_rejection_index = (
                latest_location_containment_repair_observation_index
            )
        else:
            active_semantic_repair_kind = (
                latest_semantic_rejection_kind
                if latest_semantic_rejection_index > latest_successful_read_index
                else None
            )
            active_semantic_repair_error_code = (
                latest_semantic_rejection_code
                if latest_semantic_rejection_index > latest_successful_read_index
                else None
            )
            active_semantic_rejection_index = (
                latest_semantic_rejection_index
                if active_semantic_repair_error_code is not None
                else -1
            )
        semantic_repair_attempt_count = 0
        if active_semantic_repair_error_code is not None:
            for observation in observations[latest_successful_read_index + 1 :]:
                if (
                    observation.get("observation_status") != "schema_invalid"
                    or observation.get("public_error_code")
                    != active_semantic_repair_error_code
                ):
                    continue
                repeat_count = observation.get("repeat_count", 1)
                semantic_repair_attempt_count += (
                    repeat_count
                    if isinstance(repeat_count, int)
                    and not isinstance(repeat_count, bool)
                    and repeat_count > 0
                    else 1
                )

        # A structured semantic rejection diagnoses the completion artifact,
        # not retrieval coverage. Preserve the successful read receipt and
        # keep the continuation completion-only even when the same field error
        # repeats. FlowSteer's outer Canvas may repair/replace the responsible
        # Agent; a repeated serialization or binding error is not evidence that
        # the public knowledge base lacks the requested fact.
        return _RequiredEvidenceState(
            required=required,
            search_queries=tuple(search_queries),
            normalized_search_queries=tuple(normalized_search_queries),
            search_top_ks=tuple(search_top_ks),
            search_attempt_count=len(search_queries),
            strategy_progress_count=len(distinct_term_set_signatures),
            strategy_semantics=strategy_semantics,
            strategy_proofs=strategy_proofs,
            retrieval_attempts=retrieval_attempts,
            recall_expansion_count=(
                len(search_queries) - len(distinct_term_set_signatures)
            ),
            successful_search_hit_counts=tuple(successful_search_hit_counts),
            tool_error_count=tool_error_count,
            searched_passage_ids=tuple(searched_passage_ids),
            latest_search_passage_ids=tuple(latest_search_passage_ids),
            read_passage_ids=tuple(read_passage_ids),
            upstream_read_passage_ids=tuple(upstream_read_passage_ids),
            dispatched_tool_calls=dispatched_tool_calls,
            latest_successful_operation=latest_successful_operation,
            semantic_repair_kind=active_semantic_repair_kind,
            semantic_repair_error_code=active_semantic_repair_error_code,
            semantic_repair_attempt_count=semantic_repair_attempt_count,
            semantic_repair_observation_index=(
                active_semantic_rejection_index
            ),
            location_containment_repair_anchor=(
                latest_location_containment_repair_anchor
                if active_semantic_repair_error_code is not None
                or location_containment_repair_read_count > 0
                else None
            ),
            location_containment_repair_queries=tuple(
                location_containment_repair_queries
            ),
            location_containment_repair_top_ks=tuple(
                location_containment_repair_top_ks
            ),
            location_containment_repair_hit_counts=tuple(
                location_containment_repair_hit_counts
            ),
            location_containment_repair_candidate_ids=tuple(
                candidate["passage_id"]
                for candidate in self._latest_public_search_candidates(
                    observations,
                    unread_passage_ids=tuple(
                        passage_id
                        for passage_id in latest_search_passage_ids
                        if passage_id not in read_passage_ids
                    ),
                    original_question=" ".join(
                        part
                        for part in (
                            latest_location_containment_repair_anchor,
                            _explicit_named_geographic_scope(
                                qa_question_scope(request.problem)
                            ),
                        )
                        if isinstance(part, str) and part.strip()
                    ),
                )
                if isinstance(candidate.get("passage_id"), str)
                and _location_relation_candidate_compatible(
                    title=candidate.get("title"),
                    snippet=candidate.get("snippet"),
                    entity_anchor=latest_location_containment_repair_anchor,
                    named_scope=_explicit_named_geographic_scope(
                        qa_question_scope(request.problem)
                    ),
                )
            ),
            location_containment_repair_search_count=(
                location_containment_repair_search_count
            ),
            location_containment_repair_read_count=(
                location_containment_repair_read_count
            ),
        )

    def _unified_factual_action_domain(
        self,
        state: _RequiredEvidenceState,
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Apply bounded factual-QA retrieval recovery to public state."""

        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        strategies_exhausted = (
            state.strategy_progress_count
            >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
        )

        # A structured semantic error is repaired on preserved evidence.  A
        # provenance/entity/relation mismatch instead advances the public
        # search strategy; it never admits a guessed completion.
        if state.semantic_repair_kind == "structure":
            return frozenset(), state.successful_read_count > 0
        if state.semantic_repair_kind in {"coverage", "recall", "strategy"}:
            return frozenset(), False
        if state.semantic_repair_kind == "evidence":
            location_containment_repair = (
                isinstance(state.semantic_repair_error_code, str)
                and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
                in state.semantic_repair_error_code
                and isinstance(
                    state.location_containment_repair_anchor,
                    str,
                )
                and bool(state.location_containment_repair_anchor.strip())
                and state.successful_read_count > 0
            )
            if location_containment_repair:
                if state.location_containment_repair_read_count > 0:
                    return frozenset(), True
                # DIRECT_REUSE: continue SkillFlow's one-Action/one-Observation
                # loop and reuse this adapter's existing bounded five-strategy
                # schedule.  Search snippets only select opaque read candidates;
                # the read receipt remains the evidence authority.
                if (
                    state.location_containment_repair_candidate_ids
                    and remaining_tool_calls >= 1
                ):
                    return frozenset(
                        {(QA_RETRIEVAL_TOOL_ID, "read")}
                    ), False
                if (
                    state.location_containment_repair_search_count
                    < len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                    and remaining_tool_calls >= 2
                ):
                    return frozenset(
                        {(QA_RETRIEVAL_TOOL_ID, "search")}
                    ), False
                return frozenset(), False
            actions: set[tuple[str, str]] = set()
            if state.latest_unread_passage_ids and remaining_tool_calls >= 1:
                actions.add((QA_RETRIEVAL_TOOL_ID, "read"))
            if not strategies_exhausted and remaining_tool_calls >= 2:
                actions.add((QA_RETRIEVAL_TOOL_ID, "search"))
            if not actions:
                return frozenset(), False
            return frozenset(actions), False

        if state.successful_read_count > 0:
            return frozenset(), True
        if state.latest_unread_passage_ids and remaining_tool_calls >= 1:
            actions = {(QA_RETRIEVAL_TOOL_ID, "read")}
            # The latest public title/snippet list is retrieval evidence for
            # candidate selection, not proof.  If none of those candidates
            # jointly matches entity, relation, and scope, admit the next
            # strategy search without forcing an irrelevant read first.
            if not strategies_exhausted and remaining_tool_calls >= 2:
                actions.add((QA_RETRIEVAL_TOOL_ID, "search"))
            return frozenset(actions), False
        if strategies_exhausted or remaining_tool_calls < 2:
            return frozenset(), False
        return frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False

    def _state_conditioned_action_domain(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
        """Expose a bounded multi-hop search/read continuation domain."""

        state = self._required_evidence_state(request, observations)
        if not state.required:
            return super()._state_conditioned_action_domain(request, observations)
        if (
            self._unified_factual_protocol(request)
            and (request.agent.role_family or "").casefold() == "reasoner"
        ):
            upstream_completion_admitted = bool(
                self._validated_upstream_evidence_receipts(request)
            )
        else:
            upstream_completion_admitted = any(
                self._successful_read_receipt(receipt)
                for receipt in self._direct_upstream_tool_receipts(request)
            )

        def admitted(
            tool_actions: Optional[frozenset[tuple[str, str]]],
            completion: bool,
        ) -> tuple[Optional[frozenset[tuple[str, str]]], bool]:
            # A direct Retriever predecessor may satisfy semantic provenance,
            # but its receipts remain outside this Reasoner's own Tool budget.
            # Completion validation below still checks every cited span.  Once
            # that exact upstream evidence has produced an evidence/provenance
            # rejection, preserve it but revoke completion until this Reasoner
            # obtains a new successful read through the public SkillFlow
            # Action--Observation continuation.  Otherwise one irrelevant
            # predecessor read permanently masks the search/read recovery
            # domain and causes repeated invalid completion actions.
            upstream_can_complete = (
                upstream_completion_admitted
                and state.semantic_repair_kind
                not in {"evidence", "coverage", "recall", "strategy"}
            )
            return tool_actions, completion or upstream_can_complete

        if self._unified_factual_protocol(request):
            tool_actions, completion = self._unified_factual_action_domain(state)
            if (
                (QA_RETRIEVAL_TOOL_ID, "read") in tool_actions
                and _question_ordinal_classes(qa_question_scope(request.problem))
                and self._latest_search_has_public_candidate_metadata(
                    observations
                )
                and not self._latest_public_search_candidates(
                    observations,
                    unread_passage_ids=state.latest_unread_passage_ids,
                    original_question=qa_question_scope(request.problem),
                )
            ):
                tool_actions = frozenset(
                    action
                    for action in tool_actions
                    if action != (QA_RETRIEVAL_TOOL_ID, "read")
                )
            return admitted(tool_actions, completion)

        remaining_tool_calls = max(
            0,
            self._max_tool_calls - state.dispatched_tool_calls,
        )
        minimum_reads = (
            2
            if self._task_type == "multi_hop_qa"
            and request.semantic_protocol == "hotpotqa_verified_answer_slot_v1"
            else 1
        )
        hotpot_multi_hop = (
            self._task_type == "multi_hop_qa"
            and request.semantic_protocol == "hotpotqa_verified_answer_slot_v1"
        )

        # SkillFlow's public Action--Observation continuation distinguishes a
        # missing evidence/provenance diagnosis from a structured completion
        # diagnosis.  The latter must repair on the already-read evidence;
        # blindly retrieving again cannot fix answer-slot or schema binding.
        if state.semantic_repair_kind == "structure":
            return admitted(frozenset(), state.successful_read_count > 0)
        if state.semantic_repair_kind == "evidence":
            if (
                state.latest_successful_operation == "search"
                and state.latest_unread_passage_ids
                and remaining_tool_calls >= 1
            ):
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False
                )
            if remaining_tool_calls >= 2:
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False
                )
            return admitted(frozenset(), state.successful_read_count > 0)

        if (
            state.successful_read_count >= minimum_reads
            or remaining_tool_calls == 0
            and state.successful_read_count > 0
        ):
            return admitted(frozenset(), True)

        # HotpotQA multi-hop retrieval uses the newest read Observation to
        # formulate the next missing-hop search.  This preserves SkillFlow's
        # public Action--Observation continuation while avoiding two blind
        # reads from one initial query.
        if hotpot_multi_hop and state.successful_read_count > 0:
            if (
                state.latest_successful_operation == "search"
                and state.latest_unread_passage_ids
            ):
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "read")}), False
                )
            if remaining_tool_calls >= 2:
                return admitted(
                    frozenset({(QA_RETRIEVAL_TOOL_ID, "search")}), False
                )
            return admitted(frozenset(), True)

        if state.latest_unread_passage_ids:
            action_name = "read"
        elif remaining_tool_calls >= 2 or state.successful_read_count == 0:
            action_name = "search"
        else:
            # One Tool call cannot complete a new search->read transition.
            # Preserve the successfully read evidence and admit completion.
            return admitted(frozenset(), state.successful_read_count > 0)
        return admitted(
            frozenset({(QA_RETRIEVAL_TOOL_ID, action_name)}), False
        )

    def _completion_arguments_schema(
        self,
        request: AgentRequest,
    ) -> Mapping[str, object]:
        semantic_protocol = request.semantic_protocol
        semantic_role = (request.agent.role_family or "").casefold()
        if semantic_role == "evidence_retriever" and semantic_protocol in {
            "hotpotqa_verified_answer_slot_v1",
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        }:
            non_empty_text = {"type": "string", "minLength": 1}
            question_scope = (
                hotpotqa_question_scope(request.problem)
                if semantic_protocol == "hotpotqa_verified_answer_slot_v1"
                else qa_question_scope(request.problem)
            )
            answer_type = (
                hotpotqa_answer_type_constraint(request.problem)
                if semantic_protocol == "hotpotqa_verified_answer_slot_v1"
                else qa_answer_type_constraint(request.problem)
            )
            # PROJECT_NECESSARY_ADAPTATION: SkillFlow constrains every public
            # StructuredAction and read receipt, while the shared AgentGraph
            # recovery branch needs an answer-free intermediate artifact.  The
            # Retriever owns evidence provenance only; semantic answer
            # selection remains exclusively with the Reasoner.
            return {
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {
                        "type": "object",
                        "required": [
                            "question_scope",
                            "entity_identity",
                            "target_relation",
                            "answer_type_constraint",
                            "evidence_proposition",
                            "evidence_span",
                            "passage_id",
                        ],
                        "properties": {
                            "question_scope": {
                                "const": question_scope,
                                "description": (
                                    "Copy the original question exactly without "
                                    "narrowing its semantic scope."
                                ),
                            },
                            "entity_identity": {
                                "type": "object",
                                "required": [
                                    "question_surface",
                                    "evidence_surface",
                                ],
                                "properties": {
                                    "question_surface": {
                                        **non_empty_text,
                                        "description": (
                                            "The concise question-side entity or "
                                            "event anchor copied from the original "
                                            "question. It must not be a wh-word or "
                                            "wh-phrase, the whole question, or a "
                                            "candidate/final answer."
                                            + (
                                                " For factual QA, an exact cited "
                                                "passage-title binding may normalize "
                                                "only one leading honorific and must "
                                                "come from the same successful read "
                                                "receipt."
                                                if semantic_protocol
                                                == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                                                else ""
                                            )
                                        ),
                                    },
                                    "evidence_surface": {
                                        **non_empty_text,
                                        "description": (
                                            "The coreferential surface of the same "
                                            "question-side entity or event anchor, "
                                            "copied from the receipt-grounded exact "
                                            "evidence_span, not the whole span."
                                            + (
                                                " For factual QA, the same successful "
                                                "read receipt may bind a public passage "
                                                "title/name to a coreferential pronoun "
                                                "or short-name surface in the exact "
                                                "evidence_span. The title supplies only "
                                                "identity context and must never be used "
                                                "as evidence_span. When a read body chunk "
                                                "starts with that pronoun, copy the pronoun "
                                                "rather than the title as evidence_surface "
                                                "and as the corresponding proposition "
                                                "argument. When that proposition "
                                                "field answers the original wh-dependency, "
                                                "evidence_surface must preserve the complete "
                                                "resolved title identity rather than an "
                                                "ambiguous strict subset. The exact "
                                                "receipt-grounded proposition argument may "
                                                "retain an expanded full-name surface that "
                                                "contains that title identity."
                                                if semantic_protocol
                                                == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                                                else ""
                                            )
                                        ),
                                    },
                                },
                                "additionalProperties": False,
                            },
                            "target_relation": {
                                **non_empty_text,
                                "description": (
                                    "The question-side requested relation, copied or "
                                    "inflectionally normalized from the original "
                                    "question. Do not replace it with the wording from "
                                    "the read receipt. evidence_proposition.predicate "
                                    "separately records the exact receipt-grounded "
                                    "predicate; the Reasoner and Verifier check their "
                                    "controlled semantic binding."
                                ),
                            },
                            "answer_type_constraint": {
                                "const": answer_type,
                                "description": (
                                    "The question-only answer type returned by "
                                    "qa_answer_type_constraint. When the anchor "
                                    "binds exactly one binary proposition argument, "
                                    "it checks the other argument without selecting "
                                    "an answer; answer-slot binding belongs to the "
                                    "Reasoner."
                                ),
                            },
                            "evidence_proposition": {
                                "type": "object",
                                "required": [
                                    "subject",
                                    "predicate",
                                    "object_or_attribute_value",
                                ],
                                "properties": {
                                    "subject": {
                                        **non_empty_text,
                                        "description": (
                                            "Exact subject/entity surface copied "
                                            "from evidence_span."
                                        ),
                                    },
                                    "predicate": {
                                        **non_empty_text,
                                        "description": (
                                            "Exact predicate surface copied from "
                                            "evidence_span. It may be a controlled "
                                            "relation paraphrase of target_relation, "
                                            "but it must remain verbatim receipt text."
                                        ),
                                    },
                                    "object_or_attribute_value": {
                                        **non_empty_text,
                                        "description": (
                                            "Exact object or attribute-value "
                                            "surface copied from evidence_span. "
                                            "Together with subject and predicate, "
                                            "this records evidence provenance; it "
                                            "does not select a candidate answer."
                                        ),
                                    },
                                },
                                "additionalProperties": False,
                            },
                            "evidence_span": {
                                **non_empty_text,
                                "description": (
                                    "An exact supporting span from the cited read "
                                    "receipt."
                                ),
                            },
                            "passage_id": dict(non_empty_text),
                        },
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            }
        if semantic_role != "reasoner" or (
            semantic_protocol
            not in {
                "hotpotqa_verified_answer_slot_v1",
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            }
        ):
            return super()._completion_arguments_schema(request)
        unified_factual = self._unified_factual_protocol(request)
        minimum_reasoning_items = 1 if unified_factual else 2
        non_empty_text = {"type": "string", "minLength": 1}
        non_empty_text_list = {
            "type": "array",
            "minItems": 1,
            "items": dict(non_empty_text),
        }
        qualifier_list = {
            "type": "array",
            "items": dict(non_empty_text),
        }
        if semantic_protocol == "hotpotqa_verified_answer_slot_v1":
            entity_surface_description = (
                "When this field supplies an entity answer, copy one minimal but "
                "complete evidence-aligned referential surface. Do not truncate a "
                "title, honorific, or name suffix that belongs to the source entity "
                "mention. For a possessive construction, retain the complete possessor "
                "mention before the possessive marker and exclude the marker plus the "
                "possessed attribute."
            )
        else:
            entity_surface_description = (
                "When this field supplies an entity answer, use one concise "
                "evidence-grounded entity surface. Any spelling variant, alias, or "
                "canonical-name choice must be supported by an explicit identity "
                "binding in the evidence propositions."
            )
        unified_answer_argument = (
            qa_answer_argument_constraint(request.problem)
            if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
            else None
        )
        answer_field_constraint: dict[str, object] = (
            {
                "const": unified_answer_argument,
                "description": (
                    "The question-only wh-dependency fixes which proposition "
                    "argument supplies candidate_answer."
                ),
            }
            if unified_answer_argument
            in {"subject", "object_or_attribute_value"}
            else {
                "type": "string",
                "enum": ["subject", "object_or_attribute_value"],
                "description": (
                    "The selected proposition field copied as candidate_answer. "
                    "For an entity comparison, select the winning entity from "
                    "the subject field while its compared date, number, or "
                    "attribute remains object_or_attribute_value. When the "
                    "question explicitly asks for a decade, this field may "
                    "instead select an evidence-grounded year that is "
                    "deterministically normalized to the candidate decade."
                ),
            }
        )
        answer_slot_schema = {
            "type": "object",
            "required": [
                "answer_type",
                "answer_cardinality",
                "qualifiers",
                "proposition_index",
                "answer_field",
            ],
            "properties": {
                "answer_type": {
                    "const": (
                        hotpotqa_answer_type_constraint(request.problem)
                        if semantic_protocol
                        == "hotpotqa_verified_answer_slot_v1"
                        else qa_answer_type_constraint(request.problem)
                    ),
                    "description": "The answer type requested by the original question.",
                },
                "answer_cardinality": {
                    "const": (
                        hotpotqa_answer_cardinality_constraint(request.problem)
                        if semantic_protocol
                        == "hotpotqa_verified_answer_slot_v1"
                        else qa_answer_cardinality_constraint(request.problem)
                    ),
                    "description": (
                        "Whether the original question requests one answer value or "
                        "multiple answer values."
                    ),
                },
                "qualifiers": dict(qualifier_list),
                "proposition_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Zero-based index of the evidence proposition whose selected "
                        "field supplies candidate_answer."
                    ),
                },
                "answer_field": answer_field_constraint,
            },
            "additionalProperties": False,
        }

        proposition_schema = {
            "type": "object",
            "required": [
                "subject",
                "relation",
                "object_or_attribute_value",
                "qualifiers",
                "evidence_span",
            ],
            "properties": {
                "subject": {
                    **non_empty_text,
                    "description": entity_surface_description,
                },
                "relation": {
                    **non_empty_text,
                    "description": (
                        "The predicate asserted by the evidence sentence between "
                        "subject and object_or_attribute_value."
                    ),
                },
                "object_or_attribute_value": {
                    **non_empty_text,
                    "description": (
                        entity_surface_description
                        + " Preserve the value attributed to the subject; do not "
                        "repeat the subject merely to make it candidate_answer."
                    ),
                },
                "qualifiers": dict(qualifier_list),
                "evidence_span": {
                    **non_empty_text,
                    "description": "An exact supporting span from a read passage.",
                },
            },
            "additionalProperties": False,
        }
        # NECESSARY_ADAPTATION: SkillFlow constrains every StructuredAction at
        # its provider boundary.  The shared QA protocol additionally gives
        # the semantic Reasoner ownership of one exact six-field artifact. Nest that
        # artifact under completion arguments.value so structured serving
        # cannot confuse semantic fields with action-envelope fields.
        return {
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {
                    "type": "object",
                    "required": [
                        "question_scope",
                        "answer_slot",
                        "evidence_propositions",
                        "multi_hop_chain",
                        "candidate_answer",
                        "evidence",
                    ],
                    "properties": {
                        "question_scope": {
                            "const": (
                                hotpotqa_question_scope(request.problem)
                                if semantic_protocol
                                == "hotpotqa_verified_answer_slot_v1"
                                else qa_question_scope(request.problem)
                            ),
                            "description": (
                                "Copy the original question exactly; do not narrow "
                                "or add qualifiers."
                            ),
                        },
                        "answer_slot": answer_slot_schema,
                        "evidence_propositions": {
                            "type": "array",
                            "minItems": minimum_reasoning_items,
                            "description": (
                                "Explicit answer-bearing and supporting propositions. "
                                "answer_slot.proposition_index selects the "
                                "answer-bearing proposition."
                            ),
                            "items": proposition_schema,
                        },
                        "multi_hop_chain": {
                            "type": "array",
                            "minItems": minimum_reasoning_items,
                            "items": dict(non_empty_text),
                        },
                        "candidate_answer": {
                            **non_empty_text,
                            "description": (
                                "Copy the selected proposition field exactly. For an "
                                "entity answer, it must be the minimal but complete "
                                "evidence-aligned referential surface described by "
                                "that field, not a strict subspan of the source entity "
                                "mention. The only non-verbatim normalization admitted "
                                "is a deterministic year-to-decade mapping when the "
                                "original question explicitly asks for a decade."
                            ),
                        },
                        "evidence": dict(non_empty_text_list),
                    },
                    "additionalProperties": False,
                }
            },
            "additionalProperties": False,
        }

    @staticmethod
    def _latest_public_search_candidates(
        observations: Sequence[Mapping[str, object]],
        *,
        unread_passage_ids: Sequence[str],
        original_question: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Return public title/snippet metadata for the latest search only.

        SkillFlow's ``SearchHit`` already publishes these fields.  Repeating a
        compact projection beside the current opaque-ID action domain preserves
        the upstream Tool contract while making the measured candidate-selection
        state explicit to the next bounded turn.
        """

        admitted_ids = frozenset(unread_passage_ids)
        if not admitted_ids:
            return ()
        for observation in reversed(observations):
            if observation.get("observation_status") != "success":
                continue
            result = observation.get("result")
            if not isinstance(result, Mapping) or result.get("operation") != "search":
                continue
            raw_hits = result.get("hits")
            if not isinstance(raw_hits, list):
                return ()
            ranked_candidates: list[
                tuple[tuple[bool, int, int, int], int, dict[str, object]]
            ] = []
            for hit_index, raw_hit in enumerate(raw_hits):
                if not isinstance(raw_hit, Mapping):
                    continue
                passage_id = raw_hit.get("passage_id")
                if not isinstance(passage_id, str) or passage_id not in admitted_ids:
                    continue
                title = raw_hit.get("title")
                snippet = raw_hit.get("snippet")
                rank = raw_hit.get("rank")
                if not isinstance(title, str) or not isinstance(snippet, str):
                    continue
                candidate: dict[str, object] = {
                    "passage_id": passage_id,
                    "title": title,
                    "snippet": snippet,
                }
                if type(rank) is int and rank > 0:
                    candidate["rank"] = rank
                compatibility = (
                    _public_search_candidate_compatibility(
                        original_question=original_question,
                        title=title,
                        snippet=snippet,
                    )
                    if isinstance(original_question, str)
                    and original_question.strip()
                    else (False, 0, 0, 0)
                )
                encounter_rank = (
                    rank
                    if type(rank) is int and rank > 0
                    else hit_index + 1
                )
                ranked_candidates.append(
                    (compatibility, encounter_rank, candidate)
                )
            ranked_candidates.sort(
                key=lambda item: (
                    -int(item[0][0]),
                    -item[0][1],
                    -item[0][2],
                    -item[0][3],
                    item[1],
                )
            )
            if (
                isinstance(original_question, str)
                and _question_ordinal_classes(original_question)
            ):
                # Search snippets are bounded public previews and may end
                # before the answer-bearing proposition.  Keep strongly
                # compatible previews first, but do not make snippet
                # entailment an exhaustive read gate when the public passage
                # title itself binds the complete question entity anchor.
                # The successful read receipt and semantic evidence gate
                # remain authoritative after the passage is opened.
                entity_anchor_tokens = _question_entity_anchor_tokens(
                    original_question
                )
                ranked_candidates = [
                    item
                    for item in ranked_candidates
                    if item[0][0]
                    or _surface_binds_entity_anchor(
                        str(item[2]["title"]),
                        entity_anchor_tokens,
                    )
                ]
            return tuple(item[2] for item in ranked_candidates)
        return ()

    @staticmethod
    def _latest_search_has_public_candidate_metadata(
        observations: Sequence[Mapping[str, object]],
    ) -> bool:
        """Distinguish an empty compatibility result from a legacy fixture."""

        for observation in reversed(observations):
            if observation.get("observation_status") != "success":
                continue
            result = observation.get("result")
            if not isinstance(result, Mapping) or result.get("operation") != "search":
                continue
            return isinstance(result.get("hits"), list)
        return False

    def _condition_qa_action_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
        schema: Mapping[str, object],
    ) -> dict[str, object]:
        """Apply request-local QA constraints to one action-schema branch."""

        # The generic builder shallow-copies a normalized Tool schema. Copy
        # this request-scoped schema before narrowing it so later Agents retain
        # the published SkillFlow capability unchanged.
        schema = deepcopy(schema)
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return schema
        name = properties.get("name")
        if not isinstance(name, Mapping):
            return schema
        action_name = name.get("const")
        arguments = properties.get("arguments")
        if not isinstance(arguments, dict):
            return schema
        argument_properties = arguments.get("properties")
        if not isinstance(argument_properties, dict):
            return schema
        if (
            action_name == "search"
            and request.semantic_protocol == "hotpotqa_verified_answer_slot_v1"
        ):
            argument_properties["limit"] = {"const": 10}
            return schema
        if action_name == "search" and self._unified_factual_protocol(request):
            state = self._required_evidence_state(request, observations)
            original_question = qa_question_scope(request.problem)
            location_containment_repair = (
                state.semantic_repair_kind == "evidence"
                and isinstance(state.semantic_repair_error_code, str)
                and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
                in state.semantic_repair_error_code
                and isinstance(
                    state.location_containment_repair_anchor,
                    str,
                )
            )
            if location_containment_repair:
                named_scope = _explicit_named_geographic_scope(
                    original_question
                )
                argument_properties["query"] = {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "An answer-free relation-grounding query derived from "
                        "the current public Action--Observation state. Preserve "
                        "the receipt-grounded first-hop locality anchor "
                        + json.dumps(
                            state.location_containment_repair_anchor,
                            ensure_ascii=False,
                        )
                        + " and the original named geographic scope "
                        + json.dumps(named_scope, ensure_ascii=False)
                        + ". Express a geographic type or administrative-"
                        "containment relation (city, town, district, suburb, "
                        "part of, or belongs to). Do not restart from the "
                        "original person/entity query and do not add a guessed "
                        "containing-place candidate."
                    ),
                }
                argument_properties["limit"] = {
                    "const": self._factual_search_limit(
                        state.strategy_progress_count
                        + state.location_containment_repair_search_count
                    )
                }
                return schema
            required_scope_modifiers = _missing_question_scope_modifiers(
                original_question,
                "",
            )
            required_relation_alternatives = (
                _question_relation_surface_alternatives(original_question)
            )
            argument_properties["query"] = {
                "type": "string",
                "minLength": 1,
                "description": (
                    "A focused entity-and-relation query. "
                    + _factual_transition_frontier_guidance(
                        state.strategy_progress_count
                    )
                    + " To advance retrieval, "
                    "change at least one normalized lexical item and keep both "
                    "the target entity and requested relation represented. Use "
                    "every explicit ordinal scope class from the original "
                    "question, using its original surface or a controlled "
                    "equivalent: "
                    + json.dumps(required_scope_modifiers, ensure_ascii=False)
                    + ". Preserve every question-derived strong relation class. "
                    "Answer-free strong surface alternatives admitted by the "
                    "live Tool action domain are: "
                    + json.dumps(
                        required_relation_alternatives,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + ". A generic relation head alone does not preserve a "
                    "narrower multi-token relation. Use "
                    "content-bearing entity/relation terms rather than generic "
                    "question syntax or auxiliary noise, and do not import an "
                    "unrelated topic from a returned hit. Its normalized FTS "
                    "term set must differ from every prior distinct query; term "
                    "reordering is not a strategy change. Repeating only the latest "
                    "normalized query at the required larger top_k is a recall "
                    "expansion and does not advance strategy progress. Express "
                    "a transition through a changed lexical entity/relation surface; "
                    "do not include an orchestration strategy name or metaword in query."
                ),
            }
            argument_properties["limit"] = {
                "const": self._factual_search_limit(
                    state.search_attempt_count
                )
            }
            return schema
        if action_name == "complete" and self._unified_factual_protocol(request):
            state = self._required_evidence_state(request, observations)
            if (
                state.location_containment_repair_read_count > 0
                and isinstance(state.location_containment_repair_anchor, str)
            ):
                value_schema = argument_properties.get("value")
                value_properties = (
                    value_schema.get("properties")
                    if isinstance(value_schema, Mapping)
                    else None
                )
                if isinstance(value_properties, dict):
                    propositions = value_properties.get(
                        "evidence_propositions"
                    )
                    if isinstance(propositions, dict):
                        propositions["minItems"] = 2
                        propositions["description"] = (
                            "Encode the preserved first-hop location proposition "
                            "and the successful-read-backed geographic type or "
                            "administrative-containment proposition as the first two "
                            "distinct items in that order. Preserve any additional "
                            "receipt-grounded propositions after them. Bind answer_slot "
                            "to the second item and the field whose value equals "
                            "candidate_answer."
                        )
                    chain = value_properties.get("multi_hop_chain")
                    if isinstance(chain, dict):
                        chain["minItems"] = 2
                        chain["description"] = (
                            "Include the first two distinct entries in order for the "
                            "first-hop entity-to-locality relation and the receipt-"
                            "backed locality type/containment resolution, preserving "
                            "any additional valid lineage entries after them. The leading component "
                            "of a comma-qualified first-hop locality is an admitted "
                            "identity surface; an administrative suffix is not an "
                            "answer."
                        )
                    answer_slot = value_properties.get("answer_slot")
                    if isinstance(answer_slot, dict):
                        answer_slot["description"] = (
                            "Select the receipt-backed location-resolution "
                            "proposition field that equals candidate_answer, not "
                            "the unresolved first-hop locality field."
                        )
                        answer_slot_properties = answer_slot.get("properties")
                        if isinstance(answer_slot_properties, dict):
                            read_evidence_texts = tuple(
                                passage_text
                                for observation in observations
                                for verified, passage_text in (
                                    _public_read_transition_mirror(observation),
                                )
                                if verified and passage_text is not None
                            )
                            answer_field = (
                                _location_resolution_answer_field_constraint(
                                    original_question=qa_question_scope(
                                        request.problem
                                    ),
                                    entity_anchor=(
                                        state.location_containment_repair_anchor
                                    ),
                                    read_evidence_texts=read_evidence_texts,
                                )
                            )
                            if answer_field is not None:
                                answer_slot_properties[
                                    "proposition_index"
                                ] = {
                                    "const": 1,
                                    "description": (
                                        "The successful public location-"
                                        "resolution read fixes the second "
                                        "proposition as the answer-bearing hop."
                                    ),
                                }
                                answer_slot_properties["answer_field"] = {
                                    "const": answer_field,
                                    "description": (
                                        "The successful public location-"
                                        "resolution read fixes the answer "
                                        "argument for the admitted branch."
                                    ),
                                }
                return schema
        if action_name != "read":
            return schema
        state = self._required_evidence_state(request, observations)
        if not state.latest_unread_passage_ids:
            return schema
        original_question = qa_question_scope(request.problem)
        location_containment_repair = (
            state.semantic_repair_kind == "evidence"
            and isinstance(state.semantic_repair_error_code, str)
            and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
            in state.semantic_repair_error_code
            and isinstance(state.location_containment_repair_anchor, str)
        )
        candidate_selection_scope = original_question
        if location_containment_repair:
            candidate_selection_scope = " ".join(
                part
                for part in (
                    state.location_containment_repair_anchor,
                    _explicit_named_geographic_scope(original_question),
                    "city town district suburb sublocality part belongs",
                )
                if isinstance(part, str) and part.strip()
            )
        public_candidates = self._latest_public_search_candidates(
            observations,
            unread_passage_ids=state.latest_unread_passage_ids,
            original_question=candidate_selection_scope,
        )
        admitted_passage_ids = (
            state.location_containment_repair_candidate_ids
            if location_containment_repair
            else (
                tuple(
                    candidate["passage_id"]
                    for candidate in public_candidates
                    if isinstance(candidate.get("passage_id"), str)
                )
                or state.latest_unread_passage_ids
            )
        )
        argument_properties["passage_id"] = {
            "type": "string",
            "enum": list(admitted_passage_ids),
            "description": (
                "Choose one exact opaque passage_id only when its public title "
                "and snippet jointly match "
                + (
                    "the receipt-grounded locality, its geographic type or "
                    "administrative-containment relation, and the original named "
                    "scope. "
                    if location_containment_repair
                    else (
                        "the unchanged question entity, requested relation, and "
                        "scope modifiers. "
                    )
                )
                + "Rank is retrieval order, "
                "not evidence or proof of entity/relation alignment. "
                "Public candidates from the latest successful search: "
                + json.dumps(
                    public_candidates,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
        return schema

    def _state_conditioned_response_schema(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> dict[str, object] | None:
        """Constrain every admitted Tool branch from the measured public state.

        SkillFlow's generic adapter emits a strict schema only for a singleton
        action domain.  Factual QA can deliberately admit both ``read`` and the
        next recovery ``search``.  Preserve that public choice as ``oneOf`` and
        still bind the read branch to the filtered opaque passage IDs.
        """

        schema = super()._state_conditioned_response_schema(request, observations)
        if schema is not None:
            return self._condition_qa_action_response_schema(
                request,
                observations,
                schema,
            )
        admitted_actions, completion_admitted = (
            self._state_conditioned_action_domain(request, observations)
        )
        if (
            not self._unified_factual_protocol(request)
            or admitted_actions is None
            or len(admitted_actions) < 2
            or completion_admitted
        ):
            return None
        branches: list[dict[str, object]] = []
        for resource_id, action_name in sorted(admitted_actions):
            capability = self._tool_registry.require_capability(resource_id)
            argument_schema = capability.action_schemas.get(action_name)
            if not isinstance(argument_schema, Mapping):
                continue
            branch = {
                "type": "object",
                "required": [
                    "arguments",
                    "kind",
                    "name",
                    "resource_id",
                    "skill_id",
                ],
                "properties": {
                    "arguments": dict(argument_schema),
                    "kind": {"const": "tool"},
                    "name": {"const": action_name},
                    "resource_id": {"const": resource_id},
                    "skill_id": {"const": None},
                },
                "additionalProperties": False,
            }
            branches.append(
                self._condition_qa_action_response_schema(
                    request,
                    observations,
                    branch,
                )
            )
        return {"oneOf": branches} if len(branches) >= 2 else None

    def _public_retrieval_continuation_state(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> dict[str, object] | None:
        """Project the current action-mask state without synthesizing a query.

        SkillFlow's bounded loop conditions every next Action on the ordered
        public Action--Observation history.  The factual-QA adapter additionally
        exposes its derived action-mask state in one compact record so the model
        need not reconstruct strategy progress from a long receipt transcript.
        """

        if not self._unified_factual_protocol(request):
            return None
        state = self._required_evidence_state(request, observations)
        original_question = qa_question_scope(request.problem)
        required_relation_alternatives = (
            _question_relation_surface_alternatives(original_question)
        )
        distinct_relation_queries: list[str] = []
        distinct_relation_signatures: set[tuple[str, ...]] = set()
        for query in state.search_queries:
            signature = _retrieval_query_term_set_signature(query)
            if signature in distinct_relation_signatures:
                continue
            distinct_relation_signatures.add(signature)
            distinct_relation_queries.append(query)
        transformed_relation_classes = (
            _verified_transformed_relation_classes(
                original_question=original_question,
                distinct_queries=distinct_relation_queries,
            )
        )
        remaining_relation_classes = tuple(
            sorted(
                frozenset(required_relation_alternatives)
                - transformed_relation_classes
            )
        )
        schedule_exhausted = (
            state.strategy_progress_count
            >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
        )
        transition_admissible = bool(
            not schedule_exhausted
            and state.semantic_repair_kind != "structure"
        )
        strategy = (
            "initial_retrieval"
            if transition_admissible and state.strategy_progress_count == 0
            else None
        )
        remaining_transition_strategies = tuple(
            candidate
            for candidate in state.missing_strategy_coverage
            if candidate != "initial_retrieval"
        )
        public_state = {
            "required_strategy": strategy,
            "required_top_k": (
                None
                if not transition_admissible
                else self._factual_search_limit(
                    state.search_attempt_count
                )
            ),
            "admissible_transition_strategies": (
                list(remaining_transition_strategies)
                + ["recall_expansion"]
                if transition_admissible and state.strategy_progress_count > 0
                else ["initial_retrieval"]
                if transition_admissible
                else []
            ),
            "prior_normalized_queries": list(
                dict.fromkeys(state.normalized_search_queries)
            ),
            "prior_fts_term_sets": [
                list(signature)
                for signature in dict.fromkeys(
                    _retrieval_query_term_set_signature(query)
                    for query in state.search_queries
                )
            ],
            "required_scope_modifiers": list(
                _missing_question_scope_modifiers(
                    original_question,
                    "",
                )
            ),
            "required_relation_classes": list(
                required_relation_alternatives
            ),
            "required_relation_surface_alternatives": {
                relation_class: list(alternatives)
                for relation_class, alternatives
                in required_relation_alternatives.items()
            },
            "transformed_relation_classes": list(
                sorted(transformed_relation_classes)
            ),
            "remaining_relation_transformation_classes": list(
                remaining_relation_classes
            ),
            "recall_expansion_count": state.recall_expansion_count,
            "strategy_progress_count": state.strategy_progress_count,
            "strategy_semantics_verified": (
                state.strategy_semantics_verified
            ),
            "verified_strategy_coverage": list(
                state.verified_strategy_coverage
            ) if state.strategy_progress_count > 0 else [],
            "missing_strategy_coverage": list(
                state.missing_strategy_coverage
            ) if state.strategy_progress_count > 0 else [],
            "strategy_semantics_prefix": list(state.strategy_semantics),
            "strategy_proofs": [
                proof.to_value() for proof in state.strategy_proofs
            ],
            "retrieval_attempts": [
                attempt.to_value() for attempt in state.retrieval_attempts
            ],
            "retrieval_attempts_verified": (
                state.retrieval_attempts_verified
            ),
            "strategy_schedule_length": len(
                _FACTUAL_QA_RETRIEVAL_STRATEGIES
            ),
            "semantic_repair_kind": state.semantic_repair_kind,
            "semantic_repair_attempt_count": (
                state.semantic_repair_attempt_count
            ),
        }
        location_containment_repair = (
            state.semantic_repair_kind == "evidence"
            and isinstance(state.semantic_repair_error_code, str)
            and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
            in state.semantic_repair_error_code
            and isinstance(state.location_containment_repair_anchor, str)
        )
        if location_containment_repair:
            location_signatures = tuple(
                dict.fromkeys(
                    _retrieval_query_term_set_signature(query)
                    for query in state.location_containment_repair_queries
                )
            )
            public_state.update(
                {
                    "required_strategy": None,
                    "required_top_k": self._factual_search_limit(
                        state.strategy_progress_count
                        + state.location_containment_repair_search_count
                    ),
                    "required_operation": "location_relation_grounding",
                    "receipt_grounded_entity_anchor": (
                        state.location_containment_repair_anchor
                    ),
                    "question_geographic_scope": (
                        _explicit_named_geographic_scope(original_question)
                    ),
                    "required_relation_classes": [
                        "geographic_type",
                        "administrative_containment",
                    ],
                    "location_relation_grounding_search_count": (
                        state.location_containment_repair_search_count
                    ),
                    "prior_normalized_queries": list(
                        dict.fromkeys(
                            _normalized_retrieval_query(query)
                            for query in (
                                state.location_containment_repair_queries
                            )
                        )
                    ),
                    "prior_fts_term_sets": [
                        list(signature) for signature in location_signatures
                    ],
                    "relation_compatible_unread_passage_ids": list(
                        state.location_containment_repair_candidate_ids
                    ),
                }
            )
        if strategy is not None and not location_containment_repair:
            public_state["required_strategy_semantics"] = (
                _FACTUAL_QA_RETRIEVAL_STRATEGY_GUIDANCE[strategy]
            )
        return public_state

    async def execute(self, request: AgentRequest) -> GatewayResponse:
        # NECESSARY_ADAPTATION: the generic AgentGraph completion hook receives
        # Tool receipts but not the current AgentRequest.  Bind only the QA
        # request-scoped admission bit here so concurrent executions cannot
        # leak allowed-tool state across Agents.  A zero Tool budget preserves
        # the generic direct-completion boundary because no dispatch is legal.
        requires_retrieval = (
            self._completion_policy != "optional"
            and self._max_tool_calls > 0
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        )
        semantic_reasoner_protocol = (
            request.semantic_protocol
            if request.semantic_protocol
            in {
                "hotpotqa_verified_answer_slot_v1",
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            }
            and (request.agent.role_family or "").casefold() == "reasoner"
            else None
        )
        semantic_evidence_retriever_protocol = (
            request.semantic_protocol
            if request.semantic_protocol
            in {
                "hotpotqa_verified_answer_slot_v1",
                QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
            }
            and (request.agent.role_family or "").casefold()
            == "evidence_retriever"
            else None
        )
        semantic_reasoner_question = (
            (
                hotpotqa_question_scope(request.problem)
                if semantic_reasoner_protocol
                == "hotpotqa_verified_answer_slot_v1"
                else qa_question_scope(request.problem)
            )
            if semantic_reasoner_protocol is not None
            else None
        )
        semantic_evidence_retriever_question = (
            (
                hotpotqa_question_scope(request.problem)
                if semantic_evidence_retriever_protocol
                == "hotpotqa_verified_answer_slot_v1"
                else qa_question_scope(request.problem)
            )
            if semantic_evidence_retriever_protocol is not None
            else None
        )
        if semantic_reasoner_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
            semantic_upstream_tool_receipts = (
                self._validated_upstream_evidence_receipts(request)
            )
        elif semantic_reasoner_protocol is not None:
            semantic_upstream_tool_receipts = (
                self._direct_upstream_tool_receipts(request)
            )
        else:
            semantic_upstream_tool_receipts = ()
        retrieval_token = self._retrieval_completion_required.set(requires_retrieval)
        semantic_token = self._semantic_reasoner_question.set(
            semantic_reasoner_question
        )
        semantic_protocol_token = self._semantic_reasoner_protocol.set(
            semantic_reasoner_protocol
        )
        evidence_retriever_question_token = (
            self._semantic_evidence_retriever_question.set(
                semantic_evidence_retriever_question
            )
        )
        evidence_retriever_protocol_token = (
            self._semantic_evidence_retriever_protocol.set(
                semantic_evidence_retriever_protocol
            )
        )
        upstream_receipts_token = self._semantic_upstream_tool_receipts.set(
            semantic_upstream_tool_receipts
        )
        try:
            try:
                return await super().execute(request)
            except ReactExecutionError as exc:
                public_observations = self._continuation_observations(
                    exc.react_trace
                )
                state = self._required_evidence_state(
                    request,
                    public_observations,
                )
                if self._hotpot_tool_plan_exhausted(request, state):
                    remaining_tool_calls = max(
                        0,
                        self._max_tool_calls - state.dispatched_tool_calls,
                    )
                    trace = [dict(item) for item in exc.react_trace]
                    terminal_diagnosis = {
                        "observation_status": "budget_exhausted",
                        "public_error_code": "qa_retrieval_tool_plan_exhausted",
                        "tool_plan_exhausted": True,
                        "remaining_tool_calls": remaining_tool_calls,
                        "successful_tool_receipt_count": (
                            state.dispatched_tool_calls
                        ),
                        "successful_evidence_read_count": (
                            state.successful_read_count
                        ),
                    }
                    if trace:
                        trace[-1].update(terminal_diagnosis)
                        trace[-1]["terminal_failure_diagnosis"] = terminal_diagnosis
                    else:  # pragma: no cover - bounded execution samples a turn
                        trace.append(terminal_diagnosis)
                    raise ReactExecutionError(
                        str(exc),
                        react_trace=tuple(trace),
                        tool_receipts=exc.tool_receipts,
                        model_calls=exc.model_calls,
                        tool_plan_exhausted=True,
                    ) from exc
                if not self._unified_factual_protocol(request):
                    raise
                location_containment_repair = (
                    state.semantic_repair_kind == "evidence"
                    and isinstance(state.semantic_repair_error_code, str)
                    and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
                    in state.semantic_repair_error_code
                    and isinstance(
                        state.location_containment_repair_anchor,
                        str,
                    )
                )
                location_schedule_exhausted = bool(
                    location_containment_repair
                    and state.location_containment_repair_search_count
                    >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                )
                exhausted = (
                    state.strategy_progress_count
                    >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                    or location_schedule_exhausted
                    or state.dispatched_tool_calls >= self._max_tool_calls
                )
                evidence_unresolved = (
                    state.successful_read_count == 0
                    or state.semantic_repair_kind
                    in {"evidence", "coverage", "recall", "strategy"}
                )
                if not evidence_unresolved:
                    if state.semantic_repair_kind is None:
                        raise
                    trace = [dict(item) for item in exc.react_trace]
                    if trace:
                        repair_instruction = (
                            self._public_semantic_repair_instruction(
                                state.semantic_repair_error_code
                            )
                        )
                        if repair_instruction is None:
                            repair_instruction = self._semantic_repair_instruction(
                                state.semantic_repair_kind
                            )
                        trace[-1]["repair_instruction"] = repair_instruction
                    raise ReactExecutionError(
                        str(exc),
                        react_trace=tuple(trace),
                        tool_receipts=exc.tool_receipts,
                        model_calls=exc.model_calls,
                        tool_plan_exhausted=exc.tool_plan_exhausted,
                    ) from exc
                trace = [dict(item) for item in exc.react_trace]
                if location_containment_repair:
                    diagnosis_progress_count = (
                        state.location_containment_repair_search_count
                    )
                    diagnosis_hit_counts = (
                        state.location_containment_repair_hit_counts
                    )
                    diagnosis_semantics_verified = bool(
                        diagnosis_progress_count > 0
                    )
                else:
                    diagnosis_progress_count = state.strategy_progress_count
                    diagnosis_hit_counts = state.successful_search_hit_counts
                    diagnosis_semantics_verified = (
                        state.strategy_semantics_verified
                    )
                failure_code = self._factual_exhaustion_diagnosis(
                    strategy_progress_count=diagnosis_progress_count,
                    strategy_semantics_verified=diagnosis_semantics_verified,
                    successful_search_hit_counts=diagnosis_hit_counts,
                    tool_error_count=state.tool_error_count,
                    verified_strategy_coverage=(
                        None
                        if location_containment_repair
                        else state.verified_strategy_coverage
                    ),
                )
                repair_kind = {
                    _KNOWLEDGE_BASE_COVERAGE_FAILURE: "coverage",
                    _RETRIEVAL_RECALL_FAILURE: "recall",
                    _RETRIEVAL_STRATEGY_FAILURE: "strategy",
                }[failure_code]
                terminal_diagnosis = {
                    "observation_status": "budget_exhausted",
                    "public_error_code": failure_code,
                    "tool_plan_exhausted": True,
                    "bounded_schedule_exhausted": exhausted,
                    "retrieval_attempt_count": state.search_attempt_count,
                    "retrieval_strategy_progress_count": (
                        state.strategy_progress_count
                    ),
                    "recall_expansion_count": state.recall_expansion_count,
                    "retrieval_strategy_schedule_prefix": list(
                        proof.strategy for proof in state.strategy_proofs
                    ),
                    "verified_retrieval_strategy_coverage": list(
                        state.verified_strategy_coverage
                    ),
                    "missing_retrieval_strategy_coverage": list(
                        state.missing_strategy_coverage
                    ),
                    "normalized_query_novelty_verified": (
                        state.recall_expansion_count == 0
                    ),
                    "strategy_semantics_verified": (
                        state.strategy_semantics_verified
                    ),
                    "strategy_semantics_prefix": list(
                        state.strategy_semantics
                    ),
                    "retrieval_attempts": [
                        attempt.to_value()
                        for attempt in state.retrieval_attempts
                    ],
                    "retrieval_attempts_verified": (
                        state.retrieval_attempts_verified
                    ),
                    "successful_search_with_hits_count": sum(
                        hit_count > 0
                        for hit_count in state.successful_search_hit_counts
                    ),
                    "successful_empty_search_count": sum(
                        hit_count == 0
                        for hit_count in state.successful_search_hit_counts
                    ),
                    "tool_error_count": state.tool_error_count,
                    "search_queries": list(state.search_queries),
                    "search_top_ks": list(state.search_top_ks),
                    "location_relation_grounding_search_count": (
                        state.location_containment_repair_search_count
                    ),
                    "location_relation_grounding_queries": list(
                        state.location_containment_repair_queries
                    ),
                    "location_relation_grounding_top_ks": list(
                        state.location_containment_repair_top_ks
                    ),
                }
                if trace:
                    trace[-1]["repair_instruction"] = (
                        self._semantic_repair_instruction(repair_kind)
                    )
                    trace[-1].update(terminal_diagnosis)
                    trace[-1]["terminal_failure_diagnosis"] = terminal_diagnosis
                else:  # pragma: no cover - bounded execution always samples a turn
                    trace.append(terminal_diagnosis)
                raise ReactExecutionError(
                    str(exc),
                    react_trace=tuple(trace),
                    tool_receipts=exc.tool_receipts,
                    model_calls=exc.model_calls,
                    tool_plan_exhausted=True,
                ) from exc
        finally:
            self._semantic_evidence_retriever_protocol.reset(
                evidence_retriever_protocol_token
            )
            self._semantic_evidence_retriever_question.reset(
                evidence_retriever_question_token
            )
            self._semantic_upstream_tool_receipts.reset(
                upstream_receipts_token
            )
            self._semantic_reasoner_protocol.reset(semantic_protocol_token)
            self._semantic_reasoner_question.reset(semantic_token)
            self._retrieval_completion_required.reset(retrieval_token)

    def _contract(
        self,
        request: AgentRequest,
        observations: list[Mapping[str, object]],
    ) -> str:
        contract = super()._contract(request, observations)
        evidence_state = self._required_evidence_state(request, observations)
        admitted_actions, completion_admitted = self._state_conditioned_action_domain(
            request,
            observations,
        )
        location_containment_repair = (
            evidence_state.semantic_repair_kind == "evidence"
            and isinstance(evidence_state.semantic_repair_error_code, str)
            and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
            in evidence_state.semantic_repair_error_code
            and isinstance(
                evidence_state.location_containment_repair_anchor,
                str,
            )
        )
        original_question = qa_question_scope(request.problem)
        candidate_selection_scope = original_question
        if location_containment_repair:
            candidate_selection_scope = " ".join(
                part
                for part in (
                    evidence_state.location_containment_repair_anchor,
                    _explicit_named_geographic_scope(original_question),
                    "city town district suburb sublocality part belongs",
                )
                if isinstance(part, str) and part.strip()
            )
        searched_passage_ids = (
            evidence_state.location_containment_repair_candidate_ids
            if location_containment_repair
            else evidence_state.latest_unread_passage_ids
        )
        public_search_candidates = self._latest_public_search_candidates(
            observations,
            unread_passage_ids=searched_passage_ids,
            original_question=candidate_selection_scope,
        )
        if public_search_candidates and not location_containment_repair:
            searched_passage_ids = tuple(
                candidate["passage_id"]
                for candidate in public_search_candidates
                if isinstance(candidate.get("passage_id"), str)
            )
        public_search_candidates_text = json.dumps(
            public_search_candidates,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if self._task_type == "multi_hop_qa":
            guidance = SKILLFLOW_MULTI_HOP_QA_GUIDANCE
            if request.semantic_protocol == "hotpotqa_verified_answer_slot_v1":
                guidance += " " + HOTPOTQA_VERIFIED_ANSWER_SLOT_GUIDANCE
            elif request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                guidance += " " + QA_VERIFIED_ANSWER_LINEAGE_GUIDANCE
        elif self._task_type == "factual_qa":
            guidance = SKILLFLOW_FACTUAL_QA_GUIDANCE
            if request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                guidance += " " + QA_VERIFIED_ANSWER_LINEAGE_GUIDANCE
        else:
            guidance = ""
        # DIRECT_REUSE: SkillFlow training/task_prompts.py::{MULTI_HOP_QA,
        # FACTUAL_QA}.  Only the terminal wire is adapted from answer(response=)
        # to this runtime's already-declared StructuredAction completion.
        terminal_wire = (
            " On completion, arguments.value is the completed QA artifact, "
            "not a schema label or placeholder."
            if completion_admitted
            else (
                " Completion is not admitted in this Tool-only state; preserve "
                "the eventual artifact responsibility but emit only the current "
                "Tool action and its declared arguments."
            )
        )
        semantic_role = (request.agent.role_family or "").casefold()
        if request.semantic_protocol in {
            "hotpotqa_verified_answer_slot_v1",
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        } and semantic_role == "reasoner":
            if completion_admitted:
                terminal_wire += (
                    " As the Reasoner, arguments.value must contain exactly the six "
                    "structured fields question_scope, answer_slot, "
                    "evidence_propositions, multi_hop_chain, candidate_answer, and "
                    "evidence. Copy question_scope exactly from the original question. "
                    "Use answer_slot.proposition_index and answer_field to bind the "
                    "candidate to one explicit evidence proposition; candidate_answer "
                    "must equal that proposition's selected field, except for a "
                    "deterministic year-to-decade normalization when the original "
                    "question explicitly requests a decade. Set answer_type and "
                    "answer_cardinality from the original question and preserve its "
                    "qualifiers. For single-answer questions, return one minimal but "
                    "complete evidence-aligned referential surface rather than an alias "
                    "list, appositive gloss, or the question's answer-type head noun. In "
                    "a who-question with a possessive construction, exclude the "
                    "possessive marker and possessed attribute but retain the complete "
                    "possessor entity mention before it, including any title, honorific, "
                    "or name suffix present in the evidence."
                )
                if (
                    location_containment_repair
                    and evidence_state.location_containment_repair_read_count
                    > 0
                ):
                    terminal_wire += (
                        " The public continuation already contains a matching "
                        "location type/containment read. Encode the preserved "
                        "first hop and that resolution hop as two distinct "
                        "evidence_propositions and two distinct multi_hop_chain "
                        "entries. Bind answer_slot to the resolution proposition "
                        "field that exactly equals candidate_answer; do not "
                        "restart retrieval or keep the slot on the unresolved "
                        "first-hop locality."
                    )
            else:
                terminal_wire += (
                    " The Reasoner remains responsible for semantic alignment after "
                    "evidence is read, but no semantic-answer field belongs in the "
                    "current Tool arguments."
                )
        elif request.semantic_protocol in {
            "hotpotqa_verified_answer_slot_v1",
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        } and semantic_role == "evidence_retriever":
            if completion_admitted:
                terminal_wire += (
                    " As the Evidence Retriever, arguments.value must contain "
                    "exactly question_scope, entity_identity, target_relation, "
                    "answer_type_constraint, evidence_proposition, evidence_span, "
                    "and passage_id. entity_identity contains exactly "
                    "question_surface and evidence_surface. question_surface is "
                    "the question-side entity/event anchor copied from the original "
                    "question, never a wh-word/wh-phrase, the whole question, or an "
                    "answer. evidence_surface is the coreferential surface of that "
                    "same anchor copied from the receipt-grounded exact evidence_span. "
                    "Cite one successful qa-retrieval read receipt and copy one exact "
                    "evidence span containing the anchor surface, exact predicate, and "
                    "proposition surfaces. Keep target_relation as the question-side "
                    "requested relation and copy evidence_proposition.predicate "
                    "verbatim from the evidence span; do not overwrite one with the "
                    "other. A different question/evidence surface may use the same "
                    "read receipt's passage title for explicit coreference binding, "
                    "but the title is identity context and is never evidence_span."
                )
                if request.semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                    terminal_wire += (
                        " For factual QA, when question_surface and evidence_surface "
                        "differ and question_surface identifies that same read "
                        "receipt's public passage title, evidence_surface may be a "
                        "coreferential pronoun or short-name surface present in the "
                        "exact evidence_span. If that same named proposition field "
                        "answers the original wh-dependency, evidence_surface must "
                        "preserve the complete resolved passage-title identity rather "
                        "than an ambiguous strict subset. Keep any expanded full-name "
                        "surface in the exact receipt-grounded proposition argument. "
                        "The entity/event anchor may occupy the subject, "
                        "the object, or neither binary proposition argument. The "
                        "Reasoner alone owns relation binding, answer-slot binding, and "
                        "semantic answer selection."
                    )
                else:
                    terminal_wire += (
                        " The anchor need not occupy a binary proposition argument; "
                        "the Reasoner owns answer-slot binding. If the anchor occupies "
                        "exactly one argument, the other argument must match the "
                        "question-only answer_type_constraint; it must not occupy both "
                        "arguments."
                    )
                terminal_wire += (
                    " evidence_proposition records only receipt-grounded subject, "
                    "predicate, and object_or_attribute_value surfaces. Do not select "
                    "or emit candidate_answer, answer_slot, or final_answer."
                )
            else:
                terminal_wire += (
                    " The Evidence Retriever owns only receipt-grounded entity, "
                    "relation, and evidence provenance; no answer field belongs "
                    "in the current Tool arguments."
                )
        elif request.semantic_protocol in {
            "hotpotqa_verified_answer_slot_v1",
            QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
        } and semantic_role == "verifier":
            terminal_wire += (
                " As the Verifier, check explicit retrieved evidence, entity-to-"
                "attribute binding, multi-hop completeness, and unchanged question "
                "scope; also check answer type/cardinality, a minimal but complete "
                "evidence-aligned referential surface, and alias binding. For a who-"
                "question with a possessive construction, reject a candidate that "
                "drops part of the possessor entity mention before the possessive "
                "marker, including a source title, honorific, or name suffix. Do not "
                "replace the Reasoner's candidate. Return "
                "Candidate answer, the seven explicit boolean check fields, and "
                "Verification status. Every check field must be the literal boolean "
                "true or false, never the candidate text or an explanation. Set "
                "supported only when all checks pass; "
                "otherwise request repair without a substitute candidate."
            )
        elif request.is_format_predecessor:
            terminal_wire += (
                " As the direct predecessor of the Format Agent, put the brief "
                "answer span after `Candidate answer:` and the supporting "
                "retrieved passage span after `Evidence:` in arguments.value."
            )
        # DIRECT_REUSE: SkillFlow's bounded Agent resumes from public
        # Action--Observation history rather than discarding completed Tool
        # transitions after a rejected completion.  The HotpotQA wording below
        # narrows only the semantic repair obligation; the retained search/read
        # receipts and the next state-conditioned action remain that upstream
        # continuation boundary.
        evidence_continuation = ""
        if (
            self._completion_policy == "required_evidence"
            and QA_RETRIEVAL_TOOL_ID in request.agent.allowed_tools
        ):
            evidence_continuation = (
                "\nRequired-evidence ReAct continuation: preserve any semantic "
                "candidate already present in the public action history; do not "
                "discard or replace it merely because completion was rejected. "
            )
            if completion_admitted:
                evidence_continuation += (
                    "The required successful non-empty qa-retrieval reads are present, "
                    "so the next action must complete after aligning every required "
                    "hop to the original answer slot. This turn use kind=complete, "
                    "name=complete, resource_id=null, and skill_id=null; arguments "
                    "must contain exactly one key, value, whose value is the full "
                    "structured semantic artifact. Do not use kind=completion, "
                    "name=answer, or resource_id=qa-retrieval."
                )
            elif (
                location_containment_repair
                and admitted_actions
                == frozenset({(QA_RETRIEVAL_TOOL_ID, "search")})
            ):
                evidence_continuation += (
                    "The typed feedback requires a public location relation-"
                    "grounding continuation, not another question-entity "
                    "retrieval strategy. The next action must be qa-retrieval "
                    "search with limit exactly "
                    f"{self._factual_search_limit(evidence_state.strategy_progress_count + evidence_state.location_containment_repair_search_count)}. "
                    "Preserve the receipt-"
                    "grounded first-hop locality anchor "
                    + json.dumps(
                        evidence_state.location_containment_repair_anchor,
                        ensure_ascii=False,
                    )
                    + " and the original named geographic scope "
                    + json.dumps(
                        _explicit_named_geographic_scope(original_question),
                        ensure_ascii=False,
                    )
                    + ". Express a geographic type or administrative-"
                    "containment relation. Do not restart from the original "
                    "person/entity query, guess a parent location, or discard "
                    "the receipt-grounded first hop."
                )
            elif (
                self._unified_factual_protocol(request)
                and searched_passage_ids
                and admitted_actions is not None
                and (QA_RETRIEVAL_TOOL_ID, "read") in admitted_actions
                and (QA_RETRIEVAL_TOOL_ID, "search") in admitted_actions
            ):
                transition_guidance = _factual_transition_frontier_guidance(
                    evidence_state.strategy_progress_count
                )
                expected_top_k = self._factual_search_limit(
                    evidence_state.search_attempt_count
                )
                evidence_continuation += (
                    "The latest public search candidates do not by themselves bind "
                    "the unchanged question entity and requested relation. Preserve "
                    "every receipt. Inspect the title/snippet candidates and either "
                    "read one exact unread passage_id when its title and snippet "
                    "jointly support entity identity, relation, and scope modifiers, "
                    "or, when none does, issue a new qa-retrieval search with "
                    f"limit exactly {expected_top_k}. For search, "
                    + transition_guidance
                    + " The query must either be a distinct, answer-free entity-and-"
                    "relation reformulation, or repeat only the latest normalized "
                    "query at each strictly larger required top_k in the bounded "
                    "schedule as recall expansion. Recall expansion does not advance "
                    "strategy coverage. Do not copy a strategy label into query. "
                    "For read, arguments contains only one passage_id from: "
                    + json.dumps(searched_passage_ids, ensure_ascii=False)
                    + ". Public candidates from the latest successful search are: "
                    + public_search_candidates_text
                    + ". Rank is retrieval order, not evidence or proof of entity/"
                    "relation alignment. Do not complete until a read "
                    "receipt binds entity identity and requested relation."
                )
            elif searched_passage_ids and admitted_actions == frozenset(
                {(QA_RETRIEVAL_TOOL_ID, "read")}
            ):
                read_wire = {
                    "arguments": {"passage_id": searched_passage_ids[0]},
                    "kind": "tool",
                    "name": "read",
                    "resource_id": QA_RETRIEVAL_TOOL_ID,
                    "skill_id": None,
                }
                read_alignment = (
                    "the receipt-grounded locality's geographic type or "
                    "administrative containment inside the original named scope"
                    if location_containment_repair
                    else (
                        "the unchanged question's entity identity and target "
                        "relation"
                    )
                )
                evidence_continuation += (
                    "The next action must be qa-retrieval read using one exact "
                    "passage_id returned by the successful search observation: "
                    + json.dumps(searched_passage_ids, ensure_ascii=False)
                    + ". Public candidates from the latest successful search are: "
                    + public_search_candidates_text
                    + ". Inspect every returned title and snippet jointly, then "
                    "select a passage only when that joint match supports "
                    + read_alignment
                    + ". "
                    "Rank is retrieval order, not evidence or proof of identity "
                    "and relation. Its "
                    "arguments object contains only passage_id; never put "
                    "Question scope, Answer slot, Evidence propositions, Multi-hop "
                    "chain, Candidate answer, Evidence, JSON-Schema properties, or "
                    "additionalProperties into read arguments. One syntactically "
                    "valid exact wire using a returned passage_id is: "
                    + json.dumps(
                        read_wire,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + ". Do not call complete before that read succeeds."
                )
            else:
                if self._unified_factual_protocol(request):
                    transition_guidance = (
                        _factual_transition_frontier_guidance(
                            evidence_state.strategy_progress_count
                        )
                    )
                    expected_top_k = self._factual_search_limit(
                        evidence_state.search_attempt_count
                    )
                    evidence_continuation += (
                        "The next action must be qa-retrieval search with limit "
                        f"exactly {expected_top_k}. Preserve the original semantic "
                        "scope and answer slot while reformulating only the "
                        "entity/relation query. "
                        + transition_guidance
                        + " A returned title may be reused as an exact entity anchor "
                        "only when that title and its snippet jointly identify the "
                        "target entity/topic; never import an anchor from an "
                        "irrelevant subtype or subtopic. If that supported title "
                        "anchor is retained but its passage misses the requested "
                        "relation, change only the relation surface. Express the "
                        "transition through the changed "
                        "lexical entity/relation surface; never copy a strategy label "
                        "or metaword into query. To advance retrieval, the "
                        "query FTS term set must differ from every prior strategy "
                        "term set; reordering terms is not a strategy change. Only "
                        "the latest normalized query may be repeated at each "
                        "strictly larger required top_k in the bounded schedule as "
                        "recall expansion; that expansion does not advance strategy "
                        "progress, and the next "
                        "search must use the next schema-required top_k. Prior "
                        "normalized queries are: "
                        + json.dumps(
                            list(evidence_state.normalized_search_queries),
                            ensure_ascii=False,
                        )
                        + ". Then read one exact returned passage_id. Search "
                        "arguments contain exactly query and limit; do not put the "
                        "eventual semantic artifact in Tool arguments. If all "
                        "bounded strategies fail to bind entity identity and target "
                        "relation to a successful Tool receipt, report "
                        "knowledge_base_coverage_failure and never guess or fabricate "
                        "evidence."
                    )
                else:
                    evidence_continuation += (
                        "The next action must be qa-retrieval search with a concise "
                        "entity-and-relation query for a missing hop. Then read an exact returned "
                        "passage_id. Search arguments contain exactly query and limit; "
                        "never copy JSON-Schema properties/additionalProperties or the "
                        "eventual semantic artifact into those arguments. Set query to "
                        "one non-empty focused entity-and-relation string selected from "
                        "the original question or newest read evidence. For HotpotQA, "
                        "set limit to exactly 10 so the bounded continuation can inspect "
                        "two distinct ranked passages from the same search result. Keep "
                        "the outer constants kind=tool, name=search, "
                        "resource_id=qa-retrieval, and skill_id=null. Do not call "
                        "complete before a non-empty read succeeds."
                    )
        qa_guidance = (
            "\nSkillFlow QA execution guidance: " + guidance + terminal_wire
            if guidance
            else ""
        )
        public_retrieval_state = self._public_retrieval_continuation_state(
            request,
            observations,
        )
        public_state_text = (
            ""
            if public_retrieval_state is None
            else "\nCurrent public retrieval continuation state: "
            + json.dumps(
                public_retrieval_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return (
            contract
            + qa_guidance
            + evidence_continuation
            + public_state_text
        )

    @staticmethod
    def _evidence_retriever_completion_issue(
        *,
        original_question: str,
        artifact: str,
        tool_receipts: Sequence[Mapping[str, object]],
    ) -> str | None:
        """Validate one answer-free Retriever artifact against one read receipt.

        An exact same-receipt title binding can ground a distinct question and
        evidence entity surface without weakening the contextual-anchor path.
        """

        from .agent_workflow_env import (
            AgentWorkflowEnv,
            _PERSON_TITLE_PATTERN,
            _canonical_evidence_text,
            _evidence_span_matches_read,
        )

        required_fields = (
            "question_scope",
            "entity_identity",
            "target_relation",
            "answer_type_constraint",
            "evidence_proposition",
            "evidence_span",
            "passage_id",
        )
        fields, issue = AgentWorkflowEnv._structured_semantic_fields(
            artifact,
            required_fields,
        )
        if issue is not None or fields is None:
            return issue or "Evidence Retriever artifact is missing"
        if fields.get("question_scope") != original_question:
            return "Evidence Retriever question_scope must equal the original question"

        passage_id = fields.get("passage_id")
        evidence_span = fields.get("evidence_span")
        target_relation = fields.get("target_relation")
        answer_type_constraint = fields.get("answer_type_constraint")
        evidence_proposition = fields.get("evidence_proposition")
        identity = fields.get("entity_identity")
        if (
            not isinstance(passage_id, str)
            or not passage_id.strip()
            or passage_id != passage_id.strip()
            or not isinstance(evidence_span, str)
            or not evidence_span.strip()
            or not isinstance(target_relation, str)
            or not target_relation.strip()
        ):
            return (
                "Evidence Retriever passage_id must be non-empty trimmed text; "
                "target_relation and evidence_span must be non-empty text"
            )
        if not isinstance(identity, Mapping) or set(identity) != {
            "question_surface",
            "evidence_surface",
        }:
            return (
                "Evidence Retriever entity_identity must contain exactly "
                "question_surface and evidence_surface"
            )
        question_surface = identity.get("question_surface")
        evidence_surface = identity.get("evidence_surface")
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in (question_surface, evidence_surface)
        ):
            return "Evidence Retriever entity surfaces must be non-empty trimmed text"
        assert isinstance(question_surface, str)
        assert isinstance(evidence_surface, str)

        expected_answer_type = qa_answer_type_constraint(original_question)
        if answer_type_constraint != expected_answer_type:
            return (
                "Evidence Retriever answer_type_constraint must equal the "
                "question-only qa_answer_type_constraint"
            )
        if not isinstance(evidence_proposition, Mapping) or set(
            evidence_proposition
        ) != {
            "subject",
            "predicate",
            "object_or_attribute_value",
        }:
            return (
                "Evidence Retriever evidence_proposition must contain exactly "
                "subject, predicate, and object_or_attribute_value"
            )
        proposition_values = tuple(
            evidence_proposition.get(field)
            for field in ("subject", "predicate", "object_or_attribute_value")
        )
        if any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in proposition_values
        ):
            return (
                "Evidence Retriever evidence proposition surfaces must be "
                "non-empty trimmed text"
            )
        proposition_subject, proposition_predicate, proposition_object = (
            proposition_values
        )
        assert isinstance(proposition_subject, str)
        assert isinstance(proposition_predicate, str)
        assert isinstance(proposition_object, str)

        cited_read_text: str | None = None
        cited_read_title: str | None = None
        for receipt in tool_receipts:
            if not AgentWorkflowEnv._successful_read_receipt(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            ):
                continue
            receipt_request = receipt.get("request")
            assert isinstance(receipt_request, Mapping)
            receipt_arguments = receipt_request.get("arguments")
            if not isinstance(receipt_arguments, Mapping):
                continue
            request_passage_id = receipt_arguments.get("passage_id")
            if request_passage_id != passage_id:
                continue
            result = receipt.get("result")
            assert isinstance(result, Mapping)
            value = result.get("value", result)
            assert isinstance(value, Mapping)
            passage = value.get("passage")
            assert isinstance(passage, Mapping)
            result_ids = (
                value.get("passage_id"),
                passage.get("passage_id"),
            )
            if any(
                result_id is not None and result_id != passage_id
                for result_id in result_ids
            ):
                continue
            cited_read_text = AgentWorkflowEnv._successful_read_text(
                receipt,
                QA_RETRIEVAL_TOOL_ID,
            )
            if cited_read_text is not None:
                raw_title = passage.get("title")
                if isinstance(raw_title, str) and raw_title.strip():
                    cited_read_title = raw_title.strip()
                break
        if cited_read_text is None:
            return (
                "Evidence Retriever passage_id has no matching successful "
                "qa-retrieval read receipt"
            )
        if not _evidence_span_matches_read(evidence_span, cited_read_text):
            return (
                "Evidence Retriever evidence_span has no typography-canonical "
                "lexical match in the cited qa-retrieval read receipt"
            )

        canonical_question = _canonical_evidence_text(original_question)
        canonical_span = _canonical_evidence_text(evidence_span)
        canonical_read = _canonical_evidence_text(cited_read_text)
        canonical_title = (
            _canonical_evidence_text(cited_read_title)
            if cited_read_title is not None
            else ""
        )
        canonical_question_surface = _canonical_evidence_text(question_surface)
        canonical_evidence_surface = _canonical_evidence_text(evidence_surface)
        canonical_relation = _canonical_evidence_text(target_relation)
        canonical_subject = _canonical_evidence_text(proposition_subject)
        canonical_predicate = _canonical_evidence_text(proposition_predicate)
        canonical_object = _canonical_evidence_text(proposition_object)
        honorific_normalized_question = re.sub(
            rf"^(?:{_PERSON_TITLE_PATTERN})\s+",
            "",
            canonical_question_surface,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        question_name_tokens = frozenset(
            token
            for token in _scope_tokens(honorific_normalized_question)
            if token not in _RELATION_CONTEXT_STOPWORDS and len(token) > 1
        )
        title_name_tokens = frozenset(
            token
            for token in _scope_tokens(canonical_title)
            if token not in _RELATION_CONTEXT_STOPWORDS and len(token) > 1
        )
        canonical_title_identity = _canonical_evidence_text(
            re.sub(
                r"\s*\([^()]*\)\s*$",
                "",
                cited_read_title or "",
            )
        )
        title_identity_tokens = frozenset(
            token
            for token in _scope_tokens(canonical_title_identity)
            if token not in _RELATION_CONTEXT_STOPWORDS and len(token) > 1
        )
        question_title_binding = bool(
            canonical_title
            and (
                canonical_question_surface == canonical_title
                or (
                    honorific_normalized_question != canonical_question_surface
                    and honorific_normalized_question == canonical_title
                )
                or (
                    bool(question_name_tokens)
                    and question_name_tokens <= title_name_tokens
                )
            )
        )
        evidence_name_tokens = frozenset(
            token
            for token in _scope_tokens(canonical_evidence_surface)
            if token not in _RELATION_CONTEXT_STOPWORDS and len(token) > 1
        )
        read_tokens = _scope_tokens(canonical_read)
        leading_receipt_coreference = bool(
            canonical_evidence_surface in _ENTITY_COREFERENCE_PRONOUNS
            and any(
                token == canonical_evidence_surface and index <= 4
                for index, token in enumerate(read_tokens)
            )
        )
        receipt_coreference_binding = bool(
            question_title_binding
            and (
                (
                    bool(evidence_name_tokens)
                    and evidence_name_tokens <= title_name_tokens
                )
                or (
                    canonical_evidence_surface
                    in _ENTITY_COREFERENCE_PRONOUNS
                    and (
                        canonical_title in canonical_read
                        or leading_receipt_coreference
                    )
                )
            )
        )
        if canonical_question_surface not in canonical_question:
            return (
                "Evidence Retriever entity_identity.question_surface does not "
                "occur in the original question"
            )
        if canonical_evidence_surface not in canonical_span:
            return (
                "Evidence Retriever entity_identity.evidence_surface does not "
                "occur in evidence_span"
            )
        if canonical_question_surface == canonical_question:
            return (
                "Evidence Retriever entity_identity.question_surface must be a "
                "concise entity mention, not the whole original question"
            )
        question_anchor_tokens = re.findall(
            r"\w+",
            question_surface.casefold(),
            flags=re.UNICODE,
        )
        if (
            question_anchor_tokens
            and question_anchor_tokens[0] in _QUESTION_ANCHOR_WH_WORDS
        ):
            return (
                "Evidence Retriever entity_identity.question_surface must be a "
                "question-side entity/event anchor, not a wh-word or wh-phrase"
            )
        if canonical_evidence_surface == canonical_span:
            return (
                "Evidence Retriever entity_identity.evidence_surface must be a "
                "concise entity mention, not the whole evidence_span"
            )
        if canonical_title and canonical_span == canonical_title:
            return (
                "Evidence Retriever evidence_span must be read-body evidence, "
                "not the cited passage title alone"
            )
        # Diagnose a read for the wrong public entity before inspecting its
        # predicate. Otherwise an irrelevant passage whose predicate also
        # drifts can be misclassified as a completion-field repair and prevent
        # the Entity Linking/Retriever continuation from advancing.
        if (
            not question_title_binding
            and canonical_question_surface != canonical_evidence_surface
            and canonical_question_surface not in canonical_span
        ):
            return (
                "Evidence Retriever alias identity lacks an explicit binding "
                "between question_surface and evidence_surface in the cited "
                "read receipt title/evidence_span"
            )
        scope_modifier_issue = _question_scope_modifier_issue(
            original_question,
            evidence_span,
        )
        if scope_modifier_issue is not None:
            return scope_modifier_issue
        if not (
            _relation_surface_matches_evidence(
                canonical_relation,
                canonical_question,
            )
            or _relation_surfaces_share_content(
                canonical_relation,
                canonical_question,
            )
            or _controlled_relation_paraphrase(
                question_relation=canonical_question,
                evidence_predicate=canonical_relation,
                original_question=original_question,
                evidence_span=evidence_span,
            )
        ):
            return (
                "Evidence Retriever target_relation must preserve the "
                "requested relation from the original question"
            )
        if canonical_question_surface == canonical_relation or (
            canonical_evidence_surface == canonical_relation
        ):
            return (
                "Evidence Retriever entity surface must not be the relation "
                "predicate surface"
            )
        for field_name, proposition_surface in (
            ("subject", canonical_subject),
            ("predicate", canonical_predicate),
            ("object_or_attribute_value", canonical_object),
        ):
            if proposition_surface not in canonical_span:
                return (
                    "Evidence Retriever evidence_proposition."
                    f"{field_name} does not occur verbatim in evidence_span"
                )
        # PROJECT_NECESSARY_ADAPTATION: SkillFlow's exact read receipt remains
        # authoritative, but a proposition can realize its relation across the
        # predicate and object fields (for example ``became`` + ``... to
        # receive ...``).  Admit that complete, still-verbatim proposition
        # realization through the same conservative lexical/paraphrase gates;
        # all three proposition fields were independently checked against the
        # same evidence_span above.
        proposition_relation_realizations = tuple(
            dict.fromkeys(
                (
                    canonical_predicate,
                    f"{canonical_predicate} {canonical_object}".strip(),
                )
            )
        )
        if not any(
            _relation_surface_matches_evidence(
                canonical_relation,
                relation_realization,
            )
            or _relation_surface_matches_evidence(
                relation_realization,
                canonical_relation,
            )
            or _controlled_relation_paraphrase(
                question_relation=canonical_relation,
                evidence_predicate=relation_realization,
                original_question=original_question,
                evidence_span=evidence_span,
            )
            for relation_realization in proposition_relation_realizations
        ):
            return (
                "Evidence Retriever question target_relation and receipt-grounded "
                "evidence_proposition relation realization lack controlled "
                "relation alignment"
            )
        ordinal_scope_issue = _relation_aware_ordinal_scope_issue(
            original_question=original_question,
            evidence_span=evidence_span,
            relation_surface=proposition_predicate,
            proposition_subject=proposition_subject,
            proposition_object=proposition_object,
        )
        if ordinal_scope_issue is not None:
            return ordinal_scope_issue

        entity_pattern = re.compile(
            rf"(?<!\w){re.escape(canonical_evidence_surface)}(?!\w)",
            flags=re.UNICODE,
        )
        entity_in_subject = entity_pattern.search(canonical_subject) is not None
        entity_in_object = entity_pattern.search(canonical_object) is not None
        if entity_in_subject and entity_in_object:
            return (
                "Evidence Retriever evidence proposition must not bind "
                "entity_identity.evidence_surface to both relation arguments"
            )
        elif (
            not entity_in_subject
            and not entity_in_object
            and not (
                canonical_question_surface == canonical_evidence_surface
                and question_title_binding
                and canonical_subject not in _ENTITY_COREFERENCE_PRONOUNS
                and canonical_object not in _ENTITY_COREFERENCE_PRONOUNS
            )
        ):
            return (
                "Evidence Retriever entity_identity.evidence_surface must bind "
                "to exactly one evidence_proposition relation argument"
            )
        elif entity_in_subject != entity_in_object:
            evidence_identity_field = (
                "subject" if entity_in_subject else "object_or_attribute_value"
            )
            if (
                qa_answer_argument_constraint(original_question)
                == evidence_identity_field
                and question_title_binding
                and canonical_evidence_surface
                not in _ENTITY_COREFERENCE_PRONOUNS
                and len(title_identity_tokens) > 1
                and bool(evidence_name_tokens)
                and evidence_name_tokens < title_identity_tokens
            ):
                return (
                    "Evidence Retriever answer-bearing entity surface is a "
                    "strict subset of the resolved passage-title identity; "
                    "preserve the complete receipt-grounded entity mention "
                    "before Reasoner answer-slot binding"
                )
            open_argument = (
                proposition_object if entity_in_subject else proposition_subject
            )
            answer_type_issue = _answer_surface_type_issue(
                expected_type=expected_answer_type,
                surface=open_argument,
            )
            if answer_type_issue is not None:
                return answer_type_issue

        # A strong answer-slot mismatch cannot be repaired by rewriting only
        # identity fields. Diagnose it before the title-chain repair so the
        # bounded factual-QA retrieval policy can augment evidence instead of
        # looping over an irrelevant passage. If the same receipt otherwise
        # supplies the requested answer type, identity repair remains the
        # non-destructive completion-only path below.
        if (
            question_title_binding
            and canonical_question_surface != canonical_evidence_surface
            and canonical_evidence_surface not in canonical_title
            and not receipt_coreference_binding
        ):
            return (
                "Evidence Retriever entity_identity.evidence_surface is not "
                "supported by the cited passage title identity chain"
            )

        if (
            canonical_question_surface != canonical_evidence_surface
            and canonical_question_surface not in canonical_span
        ):
            canonical_receipt_identity_surface = " ".join(
                part for part in (canonical_title, canonical_span) if part
            )
            explicit_receipt_binding = not (
                canonical_question_surface
                not in canonical_receipt_identity_surface
                or canonical_evidence_surface not in canonical_receipt_identity_surface
            )

            # PROJECT_NECESSARY_ADAPTATION: preserve SkillFlow's exact read-
            # receipt boundary while normalizing only one leading linguistic
            # honorific already enumerated by the shared semantic validator.
            title_identity_binding = bool(
                question_title_binding
                and (
                    canonical_evidence_surface in canonical_title
                    or receipt_coreference_binding
                )
            )
            if not explicit_receipt_binding and not title_identity_binding:
                return (
                    "Evidence Retriever alias identity lacks an explicit binding "
                    "between question_surface and evidence_surface in the cited "
                    "read receipt title/evidence_span"
                )
        return None

    def _completion_error(
        self,
        *,
        action: StructuredAction,
        artifact: str,
        tool_receipts: list[dict[str, object]],
    ) -> str | None:
        semantic_protocol = self._semantic_reasoner_protocol.get()
        unified_factual = (
            semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
            and self._task_type == "factual_qa"
        )
        qa_receipts = tuple(
            receipt
            for receipt in tool_receipts
            if receipt.get("tool_id") == QA_RETRIEVAL_TOOL_ID
        )
        successful_search_queries: list[str] = []
        successful_search_observations: list[Mapping[str, object]] = []
        successful_search_hit_counts: list[int] = []
        tool_error_count = 0
        for receipt in qa_receipts:
            if receipt.get("error_type") is not None:
                tool_error_count += 1
                continue
            receipt_request = receipt.get("request")
            receipt_result = receipt.get("result")
            if (
                not isinstance(receipt_request, Mapping)
                or receipt_request.get("action") != "search"
                or not isinstance(receipt_result, Mapping)
            ):
                continue
            value = receipt_result.get("value", receipt_result)
            if not isinstance(value, Mapping) or value.get("operation") != "search":
                continue
            arguments = receipt_request.get("arguments")
            query = value.get("query")
            if isinstance(arguments, Mapping):
                query = arguments.get("query", query)
            if not isinstance(query, str) or not query.strip():
                continue
            successful_search_queries.append(query.strip())
            successful_search_observations.append(
                {
                    "observation_status": "success",
                    "executed_action": {
                        "kind": "tool",
                        "name": "search",
                        "resource_id": QA_RETRIEVAL_TOOL_ID,
                        "skill_id": None,
                        "arguments": dict(arguments)
                        if isinstance(arguments, Mapping)
                        else {},
                    },
                    "result": dict(value),
                }
            )
            raw_passage_ids = value.get("passage_ids")
            successful_search_hit_counts.append(
                len(
                    {
                        passage_id.strip()
                        for passage_id in raw_passage_ids
                        if isinstance(passage_id, str) and passage_id.strip()
                    }
                )
                if isinstance(raw_passage_ids, list)
                else 0
            )
        distinct_successful_queries: list[str] = []
        distinct_search_observations: list[Mapping[str, object]] = []
        distinct_signatures: list[tuple[str, ...]] = []
        for query, observation in zip(
            successful_search_queries,
            successful_search_observations,
        ):
            signature = _retrieval_query_term_set_signature(query)
            if signature in distinct_signatures:
                continue
            distinct_signatures.append(signature)
            distinct_successful_queries.append(query)
            distinct_search_observations.append(observation)
        strategy_progress_count = len(distinct_successful_queries)
        strategy_question = (
            self._semantic_reasoner_question.get()
            or self._semantic_evidence_retriever_question.get()
            or ""
        )
        strategy_proofs = _factual_strategy_proofs(
            original_question=strategy_question,
            distinct_queries=distinct_successful_queries,
            search_observations=distinct_search_observations,
        )
        strategy_semantics_verified = (
            len(strategy_proofs) == strategy_progress_count
            and all(proof.verified for proof in strategy_proofs)
        )
        tool_call_budget_exhausted = len(qa_receipts) >= self._max_tool_calls
        retrieval_budget_exhausted = (
            tool_call_budget_exhausted
            or strategy_progress_count
            >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
        )
        retrieval_failure_code = self._factual_exhaustion_diagnosis(
            strategy_progress_count=strategy_progress_count,
            strategy_semantics_verified=strategy_semantics_verified,
            successful_search_hit_counts=successful_search_hit_counts,
            tool_error_count=tool_error_count,
            verified_strategy_coverage=tuple(
                proof.strategy for proof in strategy_proofs if proof.verified
            ),
        )
        upstream_tool_receipts = list(
            self._semantic_upstream_tool_receipts.get()
        )
        evidence_tool_receipts = [
            *tool_receipts,
            *upstream_tool_receipts,
        ]

        # DIRECT_REUSE: SkillFlow training/environment.py::step admits the
        # terminal answer after at least one non-answer Tool turn.  Preserve
        # that historical ``required_tool_call`` boundary, including a failed
        # receipt, so a Tool outage cannot discard a usable upstream answer.
        # ``required_evidence`` is an explicit stricter experiment condition:
        # only a successful read receipt carrying non-empty public text counts.
        if self._retrieval_completion_required.get():
            retrieval_admitted = False
            for receipt in evidence_tool_receipts:
                if receipt.get("tool_id") != QA_RETRIEVAL_TOOL_ID:
                    continue
                receipt_request = receipt.get("request")
                if not isinstance(receipt_request, Mapping) or receipt_request.get(
                    "action"
                ) not in {"search", "read"}:
                    continue
                if self._completion_policy == "required_tool_call":
                    retrieval_admitted = True
                    break
                if self._successful_read_receipt(receipt):
                    retrieval_admitted = True
                    break
            if not retrieval_admitted:
                if self._completion_policy == "required_tool_call":
                    return "qa_completion_requires_retrieval_dispatch"
                if unified_factual and retrieval_budget_exhausted:
                    return retrieval_failure_code
                return "qa_completion_requires_successful_read_evidence"

        evidence_retriever_question = (
            self._semantic_evidence_retriever_question.get()
        )
        if evidence_retriever_question is not None:
            issue = self._evidence_retriever_completion_issue(
                original_question=evidence_retriever_question,
                artifact=artifact,
                tool_receipts=tool_receipts,
            )
            if issue is None:
                return None
            retriever_protocol = self._semantic_evidence_retriever_protocol.get()
            # A receipt/span/alias-lineage failure needs another search/read.
            # A malformed or lexically misaligned Retriever field can be
            # corrected against the already-successful read.  Route that case
            # through the existing structured-artifact repair branch so the
            # SkillFlow Action--Observation continuation does not mislabel a
            # usable passage as database coverage failure.
            structured_repair = issue.startswith(
                (
                    "artifact is empty",
                    "labelled artifact",
                    "artifact must",
                    "field ",
                    "Evidence Retriever question_scope",
                    "Evidence Retriever passage_id must be non-empty trimmed text",
                    "Evidence Retriever entity_identity must contain",
                    "Evidence Retriever entity surfaces",
                    (
                        "Evidence Retriever evidence_span has no "
                        "typography-canonical lexical match"
                    ),
                    "Evidence Retriever evidence_span must be read-body evidence",
                    "Evidence Retriever entity_identity.question_surface",
                    "Evidence Retriever entity_identity.evidence_surface",
                    "Evidence Retriever target_relation must preserve",
                    "Evidence Retriever question target_relation",
                    "Evidence Retriever answer_type_constraint",
                    "Evidence Retriever evidence_proposition must contain",
                    "Evidence Retriever evidence proposition surfaces",
                    "Evidence Retriever entity surface must not be",
                    "Evidence Retriever evidence_proposition.",
                    "Evidence Retriever evidence_proposition.predicate",
                    "Evidence Retriever evidence proposition must not bind",
                    (
                        "Evidence Retriever entity_identity.evidence_surface is not "
                        "supported by the cited passage title identity chain"
                    ),
                )
            )
            if retriever_protocol == "hotpotqa_verified_answer_slot_v1":
                prefix = (
                    _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                    if structured_repair
                    else _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                )
            else:
                prefix = (
                    _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                    if structured_repair
                    else _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                )
            return prefix + " " + issue

        original_question = self._semantic_reasoner_question.get()
        if original_question is None:
            return None
        # NECESSARY_ADAPTATION: completion admission in the generic bounded
        # executor has no AgentRequest parameter.  Reuse AgentWorkflowEnv's
        # semantic artifact parser/validator under a request-scoped ContextVar
        # so an invalid Reasoner completion becomes public repair feedback in
        # the same ReAct loop instead of breaking the outer semantic lineage.
        from .agent_workflow_env import AgentWorkflowEnv

        reasoner_kwargs: dict[str, object] = {
            "original_question": original_question,
        }
        if unified_factual:
            reasoner_kwargs.update(
                minimum_evidence_propositions=1,
                minimum_reasoning_steps=1,
            )
        candidate, issue = AgentWorkflowEnv._reasoner_candidate(
            artifact,
            **reasoner_kwargs,
        )
        if issue is not None or candidate is None:
            detail = issue or "Reasoner candidate_answer is missing"
            evidence_binding_issue = (
                "must occur verbatim in the selected evidence_span" in detail
            )
            if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL:
                prefix = (
                    _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                    if evidence_binding_issue
                    else _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                )
            else:
                prefix = (
                    _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                    if evidence_binding_issue
                    else _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                )
            return prefix + " " + detail
        read_evidence_texts = tuple(
            text
            for receipt in evidence_tool_receipts
            if isinstance(receipt, Mapping)
            for text in (
                AgentWorkflowEnv._successful_read_text(
                    receipt,
                    QA_RETRIEVAL_TOOL_ID,
                ),
            )
            if text is not None
        )
        provenance_issue = AgentWorkflowEnv._reasoner_evidence_provenance_issue(
            artifact,
            read_evidence_texts,
            require_answer_binding=(
                semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
            ),
            original_question=original_question,
        )
        if provenance_issue is not None:
            entity_binding_repair = (
                "no deterministic entity binding" in provenance_issue
            )
            proposition_field_repair = bool(
                re.search(
                    r"Reasoner evidence_propositions\[\d+\]\."
                    r"(?:subject|object_or_attribute_value|relation) is not "
                    r"grounded",
                    provenance_issue,
                )
            )
            if entity_binding_repair or proposition_field_repair:
                prefix = (
                    _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                    if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                    else _HOTPOTQA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                )
                return prefix + " " + provenance_issue
            if unified_factual and retrieval_budget_exhausted:
                return retrieval_failure_code
            prefix = (
                _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                if semantic_protocol == QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL
                else _HOTPOTQA_SEMANTIC_EVIDENCE_ERROR_PREFIX
            )
            return prefix + " " + provenance_issue
        if unified_factual:
            reasoner_fields, reasoner_fields_issue = (
                AgentWorkflowEnv._structured_semantic_fields(
                    artifact,
                    (
                        "question_scope",
                        "answer_slot",
                        "evidence_propositions",
                        "multi_hop_chain",
                        "candidate_answer",
                        "evidence",
                    ),
                )
            )
            if reasoner_fields_issue is None and reasoner_fields is not None:
                answer_slot = reasoner_fields.get("answer_slot")
                propositions = reasoner_fields.get("evidence_propositions")
                if isinstance(answer_slot, Mapping) and isinstance(
                    propositions,
                    (list, tuple),
                ):
                    proposition_index = answer_slot.get("proposition_index")
                    if (
                        isinstance(proposition_index, int)
                        and not isinstance(proposition_index, bool)
                        and 0 <= proposition_index < len(propositions)
                        and isinstance(propositions[proposition_index], Mapping)
                    ):
                        selected = propositions[proposition_index]
                        evidence_span = selected.get("evidence_span")
                        relation_surface = selected.get("relation")
                        proposition_subject = selected.get("subject")
                        proposition_object = selected.get(
                            "object_or_attribute_value"
                        )
                        if all(
                            isinstance(value, str) and bool(value.strip())
                            for value in (
                                evidence_span,
                                relation_surface,
                                proposition_subject,
                                proposition_object,
                            )
                        ):
                            assert isinstance(evidence_span, str)
                            assert isinstance(relation_surface, str)
                            assert isinstance(proposition_subject, str)
                            assert isinstance(proposition_object, str)
                            ordinal_scope_issue = (
                                _relation_aware_ordinal_scope_issue(
                                    original_question=original_question,
                                    evidence_span=evidence_span,
                                    relation_surface=relation_surface,
                                    proposition_subject=proposition_subject,
                                    proposition_object=proposition_object,
                                )
                            )
                            if ordinal_scope_issue is not None:
                                if retrieval_budget_exhausted:
                                    return retrieval_failure_code
                                return (
                                    _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                                    + " "
                                    + ordinal_scope_issue
                                )
                location_lineage_issue = _location_containment_lineage_issue(
                    original_question=original_question,
                    reasoner_fields=reasoner_fields,
                    read_evidence_texts=read_evidence_texts,
                )
                if location_lineage_issue is not None:
                    location_anchor = _location_containment_repair_anchor(
                        original_question=original_question,
                        completion_action=action.to_value(),
                    )
                    named_scope = _explicit_named_geographic_scope(
                        original_question
                    )
                    location_resolution_read_available = bool(
                        isinstance(location_anchor, str)
                        and location_anchor.strip()
                        and isinstance(named_scope, str)
                        and named_scope.strip()
                        and any(
                            any(
                                _contains_canonical_location_surface(
                                    " ".join(
                                        part
                                        for part in (
                                            getattr(
                                                read_text,
                                                "passage_title",
                                                None,
                                            ),
                                            read_text,
                                        )
                                        if isinstance(part, str)
                                        and part.strip()
                                    ),
                                    alias,
                                )
                                for alias in _location_surface_component_aliases(
                                    location_anchor
                                )
                            )
                            and _contains_canonical_location_surface(
                                read_text,
                                named_scope,
                            )
                            and re.search(
                                r"\b(?:city|town)\b",
                                read_text,
                                flags=re.IGNORECASE,
                            )
                            for read_text in read_evidence_texts
                        )
                    )
                    # The ordinary entity/relation schedule may have produced a
                    # valid first-hop read.  It cannot suppress the separate,
                    # receipt-grounded location-relation continuation while
                    # Tool budget remains.  The state-conditioned action domain
                    # applies that continuation's own bounded retry schedule.
                    if location_resolution_read_available:
                        return (
                            _QA_SEMANTIC_STRUCTURE_ERROR_PREFIX
                            + " "
                            + location_lineage_issue
                        )
                    if tool_call_budget_exhausted:
                        return retrieval_failure_code
                    return (
                        _QA_SEMANTIC_EVIDENCE_ERROR_PREFIX
                        + " "
                        + location_lineage_issue
                    )
        return None

    def _tool_action_error(
        self,
        *,
        request: AgentRequest,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        state = self._required_evidence_state(request, observations)
        if (
            self._unified_factual_protocol(request)
            and action.kind is ActionKind.TOOL
            and action.resource_id == QA_RETRIEVAL_TOOL_ID
            and action.name == "search"
        ):
            location_containment_repair = (
                state.semantic_repair_kind == "evidence"
                and isinstance(state.semantic_repair_error_code, str)
                and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
                in state.semantic_repair_error_code
                and isinstance(
                    state.location_containment_repair_anchor,
                    str,
                )
                and bool(state.location_containment_repair_anchor.strip())
                and state.successful_read_count > 0
            )
            if (
                (
                    state.strategy_progress_count
                    >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                    and not location_containment_repair
                )
                or (
                    location_containment_repair
                    and state.location_containment_repair_search_count
                    >= len(_FACTUAL_QA_RETRIEVAL_STRATEGIES)
                )
                or state.dispatched_tool_calls >= self._max_tool_calls
            ):
                if location_containment_repair:
                    diagnosis_progress_count = (
                        state.location_containment_repair_search_count
                    )
                    diagnosis_semantics_verified = bool(
                        diagnosis_progress_count > 0
                    )
                    diagnosis_hit_counts = (
                        state.location_containment_repair_hit_counts
                    )
                else:
                    diagnosis_progress_count = state.strategy_progress_count
                    diagnosis_semantics_verified = (
                        state.strategy_semantics_verified
                    )
                    diagnosis_hit_counts = state.successful_search_hit_counts
                return self._factual_exhaustion_diagnosis(
                    strategy_progress_count=diagnosis_progress_count,
                    strategy_semantics_verified=diagnosis_semantics_verified,
                    successful_search_hit_counts=diagnosis_hit_counts,
                    tool_error_count=state.tool_error_count,
                    verified_strategy_coverage=(
                        None
                        if location_containment_repair
                        else state.verified_strategy_coverage
                    ),
                )
            arguments = action.arguments
            if isinstance(arguments, dict):
                query = arguments.get("query")
                limit = arguments.get("limit")
                if isinstance(query, str) and query.strip():
                    original_question = qa_question_scope(request.problem)
                    if not location_containment_repair and (
                        _retrieval_query_strategy_metawords(
                            query,
                            original_question=original_question,
                        )
                    ):
                        return _RETRIEVAL_QUERY_STRATEGY_LABEL_INJECTION
                    if _missing_question_scope_modifiers(
                        original_question,
                        query,
                    ):
                        return _RETRIEVAL_QUERY_SCOPE_MODIFIER_LOSS
                    if location_containment_repair:
                        location_query_issue = (
                            _location_relation_grounding_query_issue(
                                query=query,
                                entity_anchor=(
                                    state.location_containment_repair_anchor
                                ),
                                named_scope=(
                                    _explicit_named_geographic_scope(
                                        original_question
                                    )
                                ),
                                original_question=original_question,
                            )
                        )
                        if location_query_issue is not None:
                            return location_query_issue
                    else:
                        entity_anchor_tokens = _question_entity_anchor_tokens(
                            original_question
                        )
                        if (
                            entity_anchor_tokens
                            and (
                                _question_has_proper_entity_anchor(
                                    original_question
                                )
                                or bool(
                                    _question_ordinal_classes(
                                        original_question
                                    )
                                )
                            )
                            and not _surface_binds_entity_anchor(
                                query,
                                entity_anchor_tokens,
                            )
                        ):
                            return _RETRIEVAL_QUERY_ENTITY_ANCHOR_LOSS
                        missing_relation_classes = (
                            _missing_required_relation_classes(
                                original_question,
                                query,
                            )
                        )
                        if missing_relation_classes:
                            return (
                                _RETRIEVAL_QUERY_RELATION_CLASS_LOSS
                                + ": missing_relation_classes="
                                + json.dumps(
                                    missing_relation_classes,
                                    ensure_ascii=False,
                                )
                            )
                    expected_answer_type = qa_answer_type_constraint(
                        original_question
                    )
                    candidate_literals: tuple[str, ...] = ()
                    if expected_answer_type == "date":
                        candidate_literals = tuple(
                            re.findall(
                                r"(?<![A-Za-z0-9])(?:\d{3,4}(?:s|['’]s)?|"
                                r"\d{2}(?:s|['’]s))(?![A-Za-z0-9])",
                                query,
                                flags=re.IGNORECASE,
                            )
                        )
                    elif expected_answer_type == "number":
                        candidate_literals = tuple(
                            re.findall(
                                r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,:/-]\d+)*"
                                r"(?![A-Za-z0-9])",
                                query,
                            )
                        )
                    if any(
                        _normalized_retrieval_query(literal)
                        not in _normalized_retrieval_query(original_question)
                        for literal in candidate_literals
                    ):
                        return _RETRIEVAL_QUERY_CANDIDATE_INJECTION
                    normalized_query = _normalized_retrieval_query(query)
                    query_term_set_signature = (
                        _retrieval_query_term_set_signature(query)
                    )
                    prior_queries = (
                        state.location_containment_repair_queries
                        if location_containment_repair
                        else state.search_queries
                    )
                    prior_top_ks = (
                        state.location_containment_repair_top_ks
                        if location_containment_repair
                        else state.search_top_ks
                    )
                    prior_limits = tuple(
                        top_k
                        for prior_query, top_k in zip(
                            prior_queries,
                            prior_top_ks,
                        )
                        if _retrieval_query_term_set_signature(prior_query)
                        == query_term_set_signature
                    )
                    if (
                        not location_containment_repair
                        and prior_queries
                        and not prior_limits
                    ):
                        previous_query = prior_queries[-1]
                        lost_named_constraints = (
                            _missing_question_named_constraints(
                                original_question,
                                query,
                            )
                            if not _missing_question_named_constraints(
                                original_question,
                                previous_query,
                            )
                            else ()
                        )
                        if lost_named_constraints:
                            return (
                                _RETRIEVAL_QUERY_NAMED_SCOPE_LOSS
                                + ": missing_named_constraints="
                                + json.dumps(
                                    lost_named_constraints,
                                    ensure_ascii=False,
                                )
                            )
                        previous_signature = (
                            _retrieval_query_term_set_signature(previous_query)
                        )
                        prior_observation = next(
                            (
                                observation
                                for observation in reversed(observations)
                                if isinstance(observation.get("result"), Mapping)
                                and observation["result"].get("operation")
                                == "search"
                                and _retrieval_query_term_set_signature(
                                    str(observation["result"].get("query", ""))
                                )
                                == previous_signature
                            ),
                            None,
                        )
                        transition_strategy, transition_verified, _ = (
                            _factual_transition_strategy_identification(
                                original_question=original_question,
                                previous_query=previous_query,
                                query=query,
                                prior_observation=prior_observation,
                            )
                        )
                        if not transition_verified:
                            issue = _RETRIEVAL_QUERY_STRATEGY_SEMANTICS_MISMATCH
                            required_relation_classes = frozenset(
                                _relation_alias_surfaces_in(original_question)
                            )
                            if required_relation_classes:
                                distinct_prior_queries: list[str] = []
                                prior_signatures: set[tuple[str, ...]] = set()
                                for prior_query in prior_queries:
                                    signature = (
                                        _retrieval_query_term_set_signature(
                                            prior_query
                                        )
                                    )
                                    if signature in prior_signatures:
                                        continue
                                    prior_signatures.add(signature)
                                    distinct_prior_queries.append(prior_query)
                                transformed_relation_classes = (
                                    _verified_transformed_relation_classes(
                                        original_question=original_question,
                                        distinct_queries=distinct_prior_queries,
                                    )
                                )
                                remaining_relation_classes = (
                                    required_relation_classes
                                    - transformed_relation_classes
                                )
                                if remaining_relation_classes:
                                    issue += ": remaining_relation_classes=" + (
                                        json.dumps(
                                            sorted(remaining_relation_classes),
                                            ensure_ascii=False,
                                        )
                                    )
                            return issue
                        covered_transition_strategies = frozenset(
                            state.verified_strategy_coverage
                        )
                        remaining_transition_strategies = tuple(
                            candidate
                            for candidate in _FACTUAL_QA_RETRIEVAL_STRATEGIES[1:]
                            if candidate
                            not in covered_transition_strategies
                        )
                        if (
                            transition_strategy
                            in covered_transition_strategies
                            and remaining_transition_strategies
                        ):
                            return (
                                _RETRIEVAL_QUERY_STRATEGY_SEMANTICS_MISMATCH
                                + ": remaining_transition_strategies="
                                + json.dumps(
                                    remaining_transition_strategies,
                                    ensure_ascii=False,
                                )
                            )
                        if transition_strategy == "alias_expansion":
                            distinct_prior_queries: list[str] = []
                            prior_signatures: set[tuple[str, ...]] = set()
                            for prior_query in prior_queries:
                                signature = (
                                    _retrieval_query_term_set_signature(
                                        prior_query
                                    )
                                )
                                if signature in prior_signatures:
                                    continue
                                prior_signatures.add(signature)
                                distinct_prior_queries.append(prior_query)
                            transformed_relation_classes = (
                                _verified_transformed_relation_classes(
                                    original_question=original_question,
                                    distinct_queries=distinct_prior_queries,
                                )
                            )
                            remaining_relation_classes = frozenset(
                                _relation_alias_surfaces_in(original_question)
                            ) - transformed_relation_classes
                            replacement_classes = (
                                _relation_surface_replacement_classes(
                                    original_question=original_question,
                                    previous_query=previous_query,
                                    query=query,
                                )
                            )
                            if (
                                remaining_relation_classes
                                and replacement_classes
                                and not (
                                    replacement_classes
                                    & remaining_relation_classes
                                )
                            ):
                                return (
                                    _RETRIEVAL_QUERY_STRATEGY_SEMANTICS_MISMATCH
                                    + ": remaining_relation_classes="
                                    + json.dumps(
                                        sorted(remaining_relation_classes),
                                        ensure_ascii=False,
                                    )
                                )
                    latest_normalized_query = (
                        _normalized_retrieval_query(prior_queries[-1])
                        if prior_queries
                        else None
                    )
                    if (
                        prior_limits
                        and not (
                            normalized_query
                            == latest_normalized_query
                            and type(limit) is int
                            and limit > max(prior_limits)
                        )
                    ):
                        return "qa_retrieval_duplicate_normalized_query"
                expected_limit = (
                    self._factual_search_limit(
                        state.strategy_progress_count
                        + state.location_containment_repair_search_count
                    )
                    if location_containment_repair
                    else self._factual_search_limit(
                        state.search_attempt_count
                    )
                )
                attempt_number = (
                    state.strategy_progress_count
                    + state.location_containment_repair_search_count
                    + 1
                    if location_containment_repair
                    else state.search_attempt_count + 1
                )
                if type(limit) is int and limit != expected_limit:
                    return (
                        "qa_retrieval_top_k_mismatch: expected "
                        f"{expected_limit} for attempt "
                        f"{attempt_number}"
                    )
        admitted_actions, completion_admitted = self._state_conditioned_action_domain(
            request,
            observations,
        )
        if state.required and action.kind is ActionKind.TOOL:
            if admitted_actions is not None and (
                action.resource_id,
                action.name,
            ) not in admitted_actions:
                if completion_admitted:
                    return "qa_required_evidence_next_action_complete"
                if (
                    self._unified_factual_protocol(request)
                    and not admitted_actions
                ):
                    location_containment_repair = (
                        state.semantic_repair_kind == "evidence"
                        and isinstance(state.semantic_repair_error_code, str)
                        and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
                        in state.semantic_repair_error_code
                        and isinstance(
                            state.location_containment_repair_anchor,
                            str,
                        )
                    )
                    return self._factual_exhaustion_diagnosis(
                        strategy_progress_count=(
                            state.location_containment_repair_search_count
                            if location_containment_repair
                            else state.strategy_progress_count
                        ),
                        strategy_semantics_verified=(
                            bool(
                                state.location_containment_repair_search_count
                                > 0
                            )
                            if location_containment_repair
                            else state.strategy_semantics_verified
                        ),
                        successful_search_hit_counts=(
                            state.location_containment_repair_hit_counts
                            if location_containment_repair
                            else state.successful_search_hit_counts
                        ),
                        tool_error_count=state.tool_error_count,
                        verified_strategy_coverage=(
                            None
                            if location_containment_repair
                            else state.verified_strategy_coverage
                        ),
                    )
                expected_action = (
                    next(iter(admitted_actions))[1] if admitted_actions else "complete"
                )
                return f"qa_required_evidence_next_action_{expected_action}"
        if (
            action.kind is not ActionKind.TOOL
            or action.resource_id != QA_RETRIEVAL_TOOL_ID
            or action.name != "read"
        ):
            return None
        arguments = action.arguments
        if not isinstance(arguments, dict):
            return None
        passage_id = arguments.get("passage_id")
        if not isinstance(passage_id, str) or not passage_id.strip():
            return None

        location_containment_repair = (
            state.semantic_repair_kind == "evidence"
            and isinstance(state.semantic_repair_error_code, str)
            and _QA_LOCATION_CONTAINMENT_LINEAGE_MISSING
            in state.semantic_repair_error_code
            and isinstance(state.location_containment_repair_anchor, str)
        )
        if location_containment_repair:
            admitted_passage_ids = set(
                state.location_containment_repair_candidate_ids
            )
        else:
            public_candidates = self._latest_public_search_candidates(
                observations,
                unread_passage_ids=state.latest_unread_passage_ids,
                original_question=qa_question_scope(request.problem),
            )
            admitted_passage_ids = {
                candidate["passage_id"]
                for candidate in public_candidates
                if isinstance(candidate.get("passage_id"), str)
            } or set(state.latest_unread_passage_ids)
        if not admitted_passage_ids:
            return "qa_read_requires_successful_search"
        if passage_id not in admitted_passage_ids:
            return "qa_read_passage_id_not_from_search"
        return None


def build_qa_tool_registry(
    index: _RetrievalIndex,
    *,
    dataset_scope: Sequence[str] = DEFAULT_QA_DATASET_SCOPE,
    timeout_seconds: float = 10.0,
) -> ToolRegistry:
    """Register explicit search/read capabilities over an open frozen index.

    The caller owns ``index``.  Use :func:`open_qa_tool_registry` when this
    adapter should own and close the SkillFlow index connection.
    """

    if isinstance(dataset_scope, (str, bytes)):
        raise TypeError("dataset_scope must be a sequence of dataset IDs")
    scope = tuple(dataset_scope)
    identity = _index_identity(index)
    version = str(identity["index_id"])
    identity_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source",
            "corpus_version",
            "index_id",
            "index_format",
            "retrieval_backend",
        ],
        "properties": {
            "source": {"type": "string"},
            "corpus_version": {"type": "string"},
            "index_id": {"type": "string"},
            "index_format": {"type": "string"},
            "retrieval_backend": {"type": "string"},
        },
    }
    search_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
    }
    search_input_schema["properties"]["query"]["description"] = (
        "A focused query for the missing public fact or relation."
    )
    search_input_schema["properties"]["limit"]["description"] = (
        "The positive number of ranked public passages to return."
    )
    read_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["passage_id"],
        "properties": {
            "passage_id": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "The exact canonical passage_id returned by a successful "
                    "search action in this execution; a title or document_id "
                    "is not a passage_id."
                ),
            },
        },
    }
    retrieval_capability = ToolCapability(
        tool_id=QA_RETRIEVAL_TOOL_ID,
        dataset_scope=scope,
        action_schemas={
            "search": search_input_schema,
            "read": read_input_schema,
        },
        input_schema={
            "oneOf": [search_input_schema, read_input_schema],
        },
        output_schema={
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "retrieval_index",
                        "query",
                        "top_k",
                        "passage_ids",
                        "hits",
                    ],
                    "properties": {
                        "operation": {"const": "search"},
                        "retrieval_index": identity_schema,
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "passage_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "hits": {"type": "array", "items": {"type": "object"}},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "retrieval_index",
                        "passage_id",
                        "passage",
                    ],
                    "properties": {
                        "operation": {"const": "read"},
                        "retrieval_index": identity_schema,
                        "passage_id": {"type": "string"},
                        "passage": {"type": "object"},
                    },
                },
            ],
        },
        side_effect="none",
        timeout_seconds=timeout_seconds,
        version=version,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                QA_RETRIEVAL_TOOL_ID,
                QARetrievalToolBackend(index, identity),
                retrieval_capability,
            ),
        )
    )


@dataclass(slots=True)
class OpenQAToolRegistry:
    """Owned SkillFlow index connection plus its registered Tool resources."""

    registry: ToolRegistry
    retrieval_index_identity: Mapping[str, object]
    _index: _RetrievalIndex = field(repr=False)
    _cleanup: Callable[[], None] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if not self._closed:
            try:
                self._index.close()
            finally:
                self._closed = True
                if self._cleanup is not None:
                    self._cleanup()

    def __enter__(self) -> "OpenQAToolRegistry":
        if self._closed:
            raise RuntimeError("QA retrieval ToolRegistry is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def open_qa_tool_registry(
    *,
    index_path: str | Path = DEFAULT_QA_RETRIEVAL_INDEX,
    skillflow_source: str | Path = DEFAULT_SKILLFLOW_SOURCE,
    dataset_scope: Sequence[str] = DEFAULT_QA_DATASET_SCOPE,
    timeout_seconds: float = 10.0,
) -> OpenQAToolRegistry:
    """Open SkillFlow's immutable index and register search/read resources."""

    retrieval_index_class = _load_retrieval_index_class(Path(skillflow_source))
    try:
        index = _ThreadAffineRetrievalWorker(
            retrieval_index_class,
            Path(index_path),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SkillFlowRetrievalError(
            f"SkillFlow retrieval index could not be opened: {Path(index_path)}"
        ) from exc
    try:
        registry = build_qa_tool_registry(
            index,
            dataset_scope=dataset_scope,
            timeout_seconds=timeout_seconds,
        )
        identity = _index_identity(index)
    except BaseException:
        index.close()
        raise
    return OpenQAToolRegistry(registry, identity, index)


def _provided_context_passages(
    values: Sequence[str],
    *,
    document_passage_class: object,
) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("provided QA context must be a non-empty passage sequence")
    passages: list[object] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("provided QA context contains an empty passage")
        normalized = value.strip()
        matched = _PROVIDED_PASSAGE.fullmatch(normalized)
        title = (
            matched.group("title").strip()
            if matched is not None
            else f"Provided passage {index + 1}"
        )
        text = matched.group("text").strip() if matched is not None else normalized
        passages.append(
            document_passage_class(
                passage_id=f"provided-context-{index + 1:02d}",
                document_id=f"provided-document-{index + 1:02d}",
                title=title,
                text=text,
            )
        )
    return tuple(passages)


def open_provided_context_qa_tool_registry(
    passages: Sequence[str],
    *,
    skillflow_source: str | Path = DEFAULT_SKILLFLOW_SOURCE,
    dataset_scope: Sequence[str] = ("hotpotqa",),
    timeout_seconds: float = 10.0,
) -> OpenQAToolRegistry:
    """Build and open SkillFlow's FTS index over one task's supplied context.

    SkillFlow ``GenericTaskEnvironment._search_passages`` searches ``context``
    before its external corpus.  The free-AgentGraph runtime uses SkillFlow's
    newer public ``DocumentPassage -> build_retrieval_index -> RetrievalIndex``
    boundary to preserve that behavior without implementing another ranker.
    """

    module = _load_retrieval_module(Path(skillflow_source))
    try:
        document_passage_class = module.DocumentPassage
        build_retrieval_index = module.build_retrieval_index
        retrieval_index_class = module.RetrievalIndex
    except AttributeError as exc:
        raise SkillFlowRetrievalError(
            "SkillFlow task-context retrieval components are unavailable"
        ) from exc

    normalized_passages = _provided_context_passages(
        passages,
        document_passage_class=document_passage_class,
    )
    temporary = TemporaryDirectory(prefix="flowsteer-provided-qa-context-")
    index_path = Path(temporary.name) / "retrieval.sqlite3"
    try:
        build_retrieval_index(
            index_path,
            normalized_passages,
            corpus_name="benchmark-provided-context",
            corpus_version="task-scoped-v1",
        )
        index = _ThreadAffineRetrievalWorker(
            retrieval_index_class,
            index_path,
        )
        registry = build_qa_tool_registry(
            index,
            dataset_scope=dataset_scope,
            timeout_seconds=timeout_seconds,
        )
        identity = _index_identity(index)
    except BaseException:
        temporary.cleanup()
        raise
    return OpenQAToolRegistry(
        registry,
        identity,
        index,
        _cleanup=temporary.cleanup,
    )


__all__ = [
    "DEFAULT_QA_DATASET_SCOPE",
    "OpenQAToolRegistry",
    "QARetrievalReactExecutionAdapter",
    "QARetrievalToolBackend",
    "QAReadToolBackend",
    "QA_RETRIEVAL_TOOL_ID",
    "QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL",
    "QASearchToolBackend",
    "build_qa_tool_registry",
    "open_qa_tool_registry",
    "open_provided_context_qa_tool_registry",
]
