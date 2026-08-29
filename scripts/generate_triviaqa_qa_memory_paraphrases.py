#!/usr/bin/env python3
"""Materialize semantic-preserving TriviaQA fact memories and provenance.

Source binding is checked before the local Qwen3.5 OpenAI-compatible endpoint
is contacted.  The model generates a reworded question and a self-contained,
relation-bearing declarative fact from each frozen canonical Q-A pair.  The
question, canonical answer, accepted aliases, and generation provenance remain
outside the Agent-facing corpus.  Only the declarative ``fact_text`` projection
is eligible for embedding and child-Agent retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from itertools import islice
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    TriviaQAQAMemoryRecord,
    TriviaQATrainSource,
    canonical_is_original_spelling_variant,
    exact_canonical_span_preserved,
    load_triviaqa_qa_memory_sources,
    relation_bearing_answer_statement,
    validate_qa_memory_against_sources,
    write_materialized_qa_memory,
)
from scripts.prepare_agentgraph_datasets import (  # noqa: E402
    CONVERTERS,
    _trivia_records,
    _path as resolve_dataset_path,
)
from scripts.materialize_triviaqa_full_train_qa_memory import (  # noqa: E402
    project_unique_nonheldout_train,
)


PROMPT_TEMPLATE_VERSION = "triviaqa.qa_memory.qa_paraphrase.v12"
PARAPHRASE_VERSION = "triviaqa.qa_memory.paraphrase.v12"
SEMANTIC_ADMISSION_VERSION = "triviaqa.qa_memory.semantic_admission.v14"
PARAPHRASE_METHOD = "semantic-preserving-question-and-answer-paraphrase"
GENERATOR_PROVIDER = "local-openai-compatible"
GENERATION_ROUND_SEED_STRIDE = 100_000_000
FULL_NATIVE_UNIQUE_QA_COUNT = 76_523
FACT_MEMORY_SCHEMA_VERSION = "flowsteer.triviaqa.fact_memory.record.v1"
FACT_PROVENANCE_SCHEMA_VERSION = (
    "flowsteer.triviaqa.fact_memory.provenance.v1"
)
FACT_MEMORY_TOOL_ID = "triviaqa.qa_memory"
DATASET_PAIR_FALLBACK_METHOD = "dataset-question-answer-pair-association"
DATASET_PAIR_FALLBACK_PROVIDER = "deterministic-dataset-binding"
DATASET_PAIR_FALLBACK_MODEL_ID = "deterministic-dataset-pair-association"
DATASET_PAIR_FALLBACK_MODEL_REVISION = "not-applicable"
DATASET_PAIR_FALLBACK_PROMPT_VERSION = (
    "triviaqa.qa_memory.dataset_pair_association.v1"
)

SYSTEM_PROMPT = """Paraphrase one TriviaQA training question and its training answer.
Preserve the exact entity identity, requested relation, answer type, temporal or geographic scope, and every constraint. Replace at least one non-entity content word or multiword expression with a true synonym or equivalent phrase; changing only word order is not enough. lexical_replacement_source_tokens lists eligible original content wording, and at least one listed token or its containing phrase must be replaced rather than merely reordered. A dangling generic interrogative may instead be expanded by at least two content words that state its existing answer type without adding another relation. Do not put the answer into the question, including generic-looking words from the canonical answer: forbidden_question_canonical_tokens lists the answer tokens absent from the original question, and none may occur in paraphrase_question. Write one complete declarative answer statement by binding the supplied canonical answer span character-for-character to the original question's wh-dependency or explicitly listed-choice slot. The answer statement must copy at least one non-answer content token from original_question so its relation lineage is explicit. Preserve the original subject/object direction, relation, scope, and constraints; prefer a minimal question-to-declarative transformation and do not require the canonical span to begin the sentence. Never inflect, lowercase, or paraphrase the canonical span. If natural grammar requires an inflected form for a listed choice, use the natural relation wording and also include the exact canonical span as the selected option label in the same declarative statement. The answer statement must not be only the canonical span or a generic wrapper such as 'The answer is ...'. Do not add facts, broaden or narrow the meaning, or invent aliases.
Preserve participation markers inside relations. In particular, co-founded must remain co-founded or use an explicitly equivalent phrase such as helped found, helped establish, or jointly founded; established/founded alone is broader and is not equivalent.
Treat every supplied immutable entity token, number/date token, and quoted span as an exact dataset string even when it looks unusual or factually mistaken. Copy those strings into the paraphrased question; never correct, replace, delete, or complete them from world knowledge. The only morphology exception is a token listed in multiword_possessive_identity_bases: an original multiword person possessive such as "Andy Warhol's" may become the non-possessive "Andy Warhol" only inside an explicit "of Andy Warhol" construction.
When leading_answer_slot_anchor is non-null, it is a disambiguating token in the original leading wh-phrase. Keep it verbatim in the initial interrogative or imperative answer-slot phrase of paraphrase_question; never move it into a downstream participant, object, title, or contextual clause.
Only when original_interrogative_head_omitted is true, the source grammar has a structurally missing head immediately after an interrogative. You may add the minimal answer-type head needed to express the single existing answer slot, using the supplied training answer type, without adding another relation or constraint. Never alter an explicit answer-type head.
Return exactly one JSON object with this schema and no other text:
{"paraphrase_question":"...","paraphrase_answer_statement":"..."}"""

VERIFICATION_SYSTEM_PROMPT = """Verify one TriviaQA question paraphrase against its original question.
Treat the original question as an authoritative dataset string even when its grammar, answer type, or named entity appears factually unusual or inconsistent with world knowledge. Do not correct the dataset and do not infer an expected answer. Compare only whether the paraphrased question preserves the original literal entity, requested relation, scope, constraints, and answer slot. Canonical-answer leakage has already been checked deterministically and the answer is intentionally withheld from this semantic-equivalence check. Evaluate each boolean independently; question_changed concerns surface wording only.
When leading_answer_slot_anchor is non-null, reject a paraphrase that moves that exact token out of the initial interrogative or imperative answer-slot phrase and reuses it as a participant, object, title, or contextual modifier.
If and only if original_interrogative_head_omitted is true, the source has one structurally incomplete interrogative slot; a paraphrase may add one minimal answer-type head to make that same slot grammatical. Treat this as preserving one answer slot, not as adding scope or another requested relation. This exception never applies when the source already states an explicit answer type.
If canonical_answer_is_explicit_compound is true, changing a generic Who/What slot to a plural answer-type phrase such as 'Which nations' still requests one compound response and does not change answer cardinality. The answer itself remains withheld; do not infer identities.
Return exactly one JSON object with these boolean fields and no other text:
{"semantic_preserved":true,"entity_identity_preserved":true,"relation_and_scope_preserved":true,"answer_cardinality_preserved":true,"answer_not_revealed":true,"question_changed":true}
Set a field to false if the paraphrase substitutes an entity, adds or removes a requested relation, adds another answer slot, changes temporal/geographic constraints, reveals any part of the supplied answer that was absent from the original question, or changes only word order without an equivalent lexical/phrase replacement."""

ANSWER_VERIFICATION_SYSTEM_PROMPT = """Verify whether a declarative answer statement is produced by binding the exact supplied canonical answer to the wh-dependency or explicitly listed-choice slot of the original TriviaQA training question while preserving relation direction and constraints. Treat both as authoritative dataset strings even if the question answer-type word is inconsistent with world knowledge. Do not fact-check. The statement may omit the wh answer-type noun after replacing the whole wh phrase. For a listed-choice question whose natural predicate inflects the canonical label, a relation-bearing statement may retain that natural predicate and include the exact canonical span as the selected option label.
Return exactly one JSON object with these boolean fields and no other text:
{"canonical_span_preserved":true,"answer_slot_bound":true,"relation_direction_preserved":true,"scope_and_constraints_preserved":true,"no_new_fact_or_relation":true}"""

ANSWER_REPAIR_SYSTEM_PROMPT = """Repair only one TriviaQA declarative answer statement by literal slot substitution. Treat the original question and canonical answer as authoritative dataset strings; never fact-check or correct them. Never return only the canonical answer, even when that span already looks like a sentence or explanation. For a wh-question, copy every non-wh token and its order from original_question, including auxiliaries such as was, is, or did; replace only the complete wh-constituent with the exact canonical span and convert question punctuation to a declarative period. For a leading 'What/Which <answer type> is/was <predicate>?' question, use the relation-bearing declarative form 'The <answer type> that is/was <predicate> is/was <exact canonical span>.' When the exact canonical span begins with a preposition such as 'In', 'On', 'At', 'By', or 'From', place that exact span first as a fronted phrase, then a comma, then render the remaining subject and predicate in declarative order; never lowercase or duplicate its preposition. When the exact canonical span is possessive and ends in "'s", and original_question asks 'the <relation noun> of what/who ...', front the exact possessive span and follow it with that relation noun and the remaining predicate, for example '<exact possessive canonical> diet ...'; never return the possessive span alone. For an explicitly listed-choice question, preserve the proposition and alternatives, state the selected relation naturally, and include the exact uninflected canonical span as the selected option label in that same declarative statement. Do not use wording from rejected_answer_statement. Do not paraphrase the relation clause, reverse subject/object direction, write '<answer type> of <canonical answer>', or add a fact, relation, scope, or constraint.
When canonical_training_answer is itself a complete clause and original_question has the form 'What <slot> did <subject> have?', preserve the clause character-for-character in the relation-bearing form 'The statement "<canonical>" identifies the <slot> that <subject> had.'
Return exactly one JSON object with this schema and no other text:
{"paraphrase_answer_statement":"..."}"""

QUESTION_REPAIR_SYSTEM_PROMPT = """Repair only one TriviaQA paraphrase_question that changed only syntax or word order. Preserve every entity, relation, answer slot, number/date, quoted span, scope, and constraint from original_question. Keep all immutable_original_entity_tokens exactly, except that a multiword person possessive listed in multiword_possessive_identity_bases may drop its final possessive suffix only in an explicit 'of <full person name>' construction. Replace required_source_token_to_replace, or a phrase containing that exact token, with a genuine synonym or equivalent phrase; do not merely reorder it, replace an entity instead, add a new relation, infer the answer, or use any forbidden_question_canonical_tokens. Report required_source_token_to_replace verbatim as replaced_source_token and the new phrase that visibly occurs in paraphrase_question. Return exactly one JSON object with this schema and no other text:
{"paraphrase_question":"...","replaced_source_token":"...","replacement_phrase":"..."}"""

SYNONYM_REPAIR_SYSTEM_PROMPT = """Produce one context-appropriate synonym or equivalent phrase for required_source_token as it is used in original_question. Preserve part of speech and meaning. Do not rewrite the question, name an answer, replace an entity, or return the same token. Return exactly one JSON object with this schema and no other text:
{"source_token":"...","replacement_phrase":"..."}"""

_VERIFICATION_FIELDS = frozenset(
    {
        "semantic_preserved",
        "entity_identity_preserved",
        "relation_and_scope_preserved",
        "answer_cardinality_preserved",
        "answer_not_revealed",
        "question_changed",
    }
)
_ANSWER_VERIFICATION_FIELDS = frozenset(
    {
        "canonical_span_preserved",
        "answer_slot_bound",
        "relation_direction_preserved",
        "scope_and_constraints_preserved",
        "no_new_fact_or_relation",
    }
)

_LEXICAL_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_NUMBER_OR_DATE_TOKEN = re.compile(
    r"\b(?:\d(?:[\d,./-]*\d)?|[IVXLCDM]{2,})\b"
)
_DOUBLE_QUOTED_SPAN = re.compile(r'"([^\"]+)"|“([^”]+)”')
_SINGLE_QUOTED_SPAN = re.compile(
    r"(?<!\w)'((?:[^']|(?<=\w)'(?=\w))+)'(?!\w)"
    + r"|(?<!\w)‘((?:[^‘’]|(?<=\w)’(?=\w)|"
    r"‘(?:[^‘’]|(?<=\w)’(?=\w))+’)+)’(?!\w)"
)
_ORDERED_QUOTED_SLOT = re.compile(
    r'"([^\"]+)"|“([^”]+)”|'
    r"(?<!\w)'((?:[^']|(?<=\w)'(?=\w))+)'(?!\w)"
    + r"|(?<!\w)‘((?:[^‘’]|(?<=\w)’(?=\w)|"
    r"‘(?:[^‘’]|(?<=\w)’(?=\w))+’)+)’(?!\w)"
)
_LEADING_DOT_LITERAL = re.compile(
    r"(?<!\w)\.([^\W_]+(?:['’-][^\W_]+)*)",
    re.UNICODE,
)
_LEADING_INTERROGATIVE_CONTRACTION = re.compile(
    r"(?:what|which|who|whom|whose|where|when|why|how)"
    r"(?:['’](?:d|ll|re|s|ve))",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_CONTRACTION = re.compile(
    r"(?:i['’]m|you['’]re|they['’]re|we['’]re|"
    r"do(?:n't|n’t)|ca(?:n't|n’t)|wo(?:n't|n’t))",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_IMPERATIVE = frozenset(
    {
        "complete",
        "finish",
        "give",
        "identify",
        "name",
        "specify",
        "state",
        "supply",
        "tell",
    }
)
_SENTENCE_BOUNDARY_NON_ENTITY_WORDS = frozenset(
    {
        "according",
        "following",
        "given",
        "his",
        "her",
        "its",
        "located",
        "our",
        "their",
        "these",
        "this",
        "those",
        "under",
        "upon",
        "your",
    }
)
_FACT_QA_WRAPPER = re.compile(
    r"(?:\b(?:question|answer|prompt|response)\s*:|"
    r"\b(?:dataset\s+source\s+prompt|paired\s+response|"
    r"corresponding\s+answer)\b|\bthe\s+answer\s+is\b)",
    re.IGNORECASE,
)
_FACT_ANAPHORIC_SUBJECT = re.compile(
    r"^(?:it|he|she|they|this|that|these|those)\b",
    re.IGNORECASE,
)
_FACT_INTERROGATIVE_HEAD = re.compile(
    r"^(?:(?:who|what|which|where|when|why|how)\s+"
    r"(?:is|are|was|were|did|does|do|has|have|had|can|could|"
    r"would|will)\b|(?:name|identify|tell)\b)",
    re.IGNORECASE,
)
_GENERIC_FACT_SUBJECT = (
    r"(?:answer|response|person|man|woman|place|country|city|river|"
    r"number|year|date|title|name|word|term|thing)"
)
_OBSERVED_RESPONSE_KEY_TYPOS = {
    ".paraphrase_question": "paraphrase_question",
    ".paraphrase_answer_statement": "paraphrase_answer_statement",
    "parphrase_question": "paraphrase_question",
}
_QUESTION_SLOT_TOKEN = re.compile(
    r"\b(?:what|which|who|whom|whose|where|when|why|how)\b",
    re.IGNORECASE,
)
_COORDINATED_QUESTION_SLOT = re.compile(
    r"(?:,\s*)?\b(?:and|or)\s+"
    r"(?:what|which|who|whom|whose|where|when|why|how)\b",
    re.IGNORECASE,
)
_SOURCE_VIEW_INTERROGATIVE_START = re.compile(
    r"^(?:what|which|who|whom|whose|where|when|why|how)\b",
    re.IGNORECASE,
)
_SOURCE_VIEW_LOWERCASE_INTERROGATIVE = re.compile(
    r"\b(?:what|which|who|whom|whose|where|when|why|how)\b"
)
_SOURCE_VIEW_SENTENCE_INTERROGATIVE = re.compile(
    r"[.!?]\s+(?:What|Which|Who|Whom|Whose|Where|When|Why|How)\b"
)
_SOURCE_VIEW_DIGIT_MOJIBAKE_MULTIPLY = re.compile(
    r"(?<=\d)\u0102\u0097(?=\d)"
)
# These repairs are deliberately bound to the complete observed surface or a
# corpus-unique syntactic frame.  Broader suffix/token splitting would corrupt
# names such as Glenis, Boris, Dennis, and Ennis.
_SOURCE_VIEW_TRUNCATED_WHICH = re.compile(
    r"^Whic(?=\s+(?:is|are|was|were|did|does|do|has|have|had)\b)"
)
_SOURCE_VIEW_FUSED_BY_WHAT = re.compile(
    r"^Bywhat(?=\s+name\s+(?:is|was)\b)"
)
_SOURCE_VIEW_FUSED_CALCANEUS_IS = re.compile(
    r"\bCalcaneusis(?=\s+the\s+medical\s+name\b)"
)
_SOURCE_VIEW_FUSED_EGO_FIRST = re.compile(
    r"\begofirst(?=\s+appeared\b)"
)
_SOURCE_VIEW_STRAY_INITIAL_PERIOD = re.compile(
    r"^\.(?: )?(?=[A-Z][a-z])"
)
_SOURCE_VIEW_TRAILING_CLOSERS = "\"\u201d\u2019')]}"


def _decode_csv_transport_wrapper(text: str) -> str:
    """Decode only an unambiguous outer CSV quote representation.

    A single outer quote pair without doubled inner delimiters can be a
    semantic quotation and therefore remains untouched. Malformed inner
    quote pairs are not inferred or repaired: this function only removes the
    transport wrapper and decodes the explicit doubled delimiters.
    """

    stripped = text.strip()
    if (
        len(stripped) >= 2
        and stripped.startswith('"')
        and stripped.endswith('"')
        and '""' in stripped
    ):
        return stripped[1:-1].replace('""', '"')
    return text


def _source_transport_question_view(question: str) -> str:
    """Return a fail-closed generation/comparison view of one source string.

    The authoritative dataset string remains unchanged in the source record
    and provenance. Only observed transport representations are normalized;
    apostrophe glyphs, curly quote delimiters, and ambiguous quote corruption
    are deliberately outside this boundary.
    """

    stripped_question = question.strip()
    ambiguous_outer_quote = (
        len(stripped_question) >= 2
        and stripped_question.startswith(('"', "\u201c", "\u2018"))
        and stripped_question.endswith(('"', "\u201d", "\u2019"))
        and '""' not in stripped_question
    )
    view = _decode_csv_transport_wrapper(question)
    view = view.replace("\u0085", "\u2026").replace("\u00a0", " ")
    view = _SOURCE_VIEW_DIGIT_MOJIBAKE_MULTIPLY.sub("\u00d7", view)
    # The initial-period guard excludes ellipses, lower-case internet domain
    # labels such as .za/.uk, and all-uppercase dot-prefixed identifiers.
    view = _SOURCE_VIEW_STRAY_INITIAL_PERIOD.sub("", view, count=1)
    view = _SOURCE_VIEW_TRUNCATED_WHICH.sub("Which", view, count=1)
    view = _SOURCE_VIEW_FUSED_BY_WHAT.sub("By what", view, count=1)
    view = _SOURCE_VIEW_FUSED_CALCANEUS_IS.sub(
        "Calcaneus is",
        view,
        count=1,
    )
    view = _SOURCE_VIEW_FUSED_EGO_FIRST.sub("ego first", view, count=1)
    # Dataset transport contains a small number of repeated ASCII spaces,
    # including inside immutable quoted spans. Model responses are normalized
    # to single spaces before quote-scope admission, so compare against the
    # same transport-only view while retaining the raw source in provenance.
    view = " ".join(view.split())

    explicit_interrogative = bool(
        _SOURCE_VIEW_INTERROGATIVE_START.search(view.lstrip())
        or _SOURCE_VIEW_LOWERCASE_INTERROGATIVE.search(view)
        or _SOURCE_VIEW_SENTENCE_INTERROGATIVE.search(view)
    )
    if explicit_interrogative and "?" not in view and not ambiguous_outer_quote:
        stripped = view.rstrip()
        if stripped.endswith("/"):
            view = stripped[:-1].rstrip() + "?"
        else:
            punctuation_surface = stripped.rstrip(
                _SOURCE_VIEW_TRAILING_CLOSERS
            )
            if not punctuation_surface.endswith((".", "!", "?")):
                view = stripped + "?"
    return view


def source_transport_view(
    source: TriviaQATrainSource,
) -> TriviaQATrainSource:
    """Bind generation and admission to a normalized view without mutation."""

    question = _source_transport_question_view(source.original_question)
    if question == source.original_question:
        return source
    return replace(source, original_question=question)


_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "after",
        "before",
        "between",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
        "during",
    }
)
_NAME_PARTICLES = frozenset(
    {"da", "de", "del", "di", "du", "la", "le", "van", "von"}
)


class SemanticPreservationError(ValueError):
    """A checkpoint row passed syntax checks but changed the requested fact.

    Resume handling catches only this exception when deciding which otherwise
    valid rows may be regenerated.  Schema, provenance, answer leakage, and
    every pre-existing admission error remain ordinary ``ValueError`` failures
    and therefore stay fail-closed.
    """


class FactProjectionAdmissionError(ValueError):
    """A semantically paired row is not an Agent-facing declarative fact.

    The exception is a narrow resume-repair boundary.  It permits a previously
    admitted Q-A row to be regenerated under the fact-only contract without
    converting unrelated schema or provenance corruption into model work.
    """


_ENTITY_ANSWER_HEADS = frozenset(
    {
        "actor",
        "actress",
        "artist",
        "athlete",
        "author",
        "band",
        "brother",
        "champion",
        "character",
        "child",
        "company",
        "composer",
        "country",
        "creator",
        "daughter",
        "designer",
        "detective",
        "deity",
        "director",
        "executive",
        "figure",
        "filmmaker",
        "firm",
        "founder",
        "general",
        "god",
        "goddess",
        "golfer",
        "group",
        "human",
        "husband",
        "individual",
        "inventor",
        "journalist",
        "king",
        "leader",
        "legend",
        "man",
        "member",
        "minister",
        "monarch",
        "musician",
        "nation",
        "nephew",
        "organization",
        "performer",
        "person",
        "player",
        "pope",
        "president",
        "producer",
        "queen",
        "relative",
        "role",
        "ruler",
        "scientist",
        "secretary",
        "sibling",
        "singer",
        "sister",
        "son",
        "spouse",
        "star",
        "team",
        "woman",
        "writer",
        "winner",
    }
)
_LOCATION_SLOT_HEADS = (
    "city",
    "country",
    "location",
    "nation",
    "place",
    "region",
    "site",
    "state",
    "territory",
)
_BIRTH_EVENT = re.compile(
    r"\b(?:born|birth(?:place)?|came into (?:the )?world|"
    r"brought into (?:the )?world|(?:come|came) into existence)\b",
    re.IGNORECASE,
)
_BIRTH_LOCATION_OR_TIME_HEADS = (
    *_LOCATION_SLOT_HEADS,
    "date",
    "day",
    "month",
    "year",
)
_RECORDING_CONTRACT_SOURCE = re.compile(
    r"\bsign(?:ed|s|ing)?\s+to\b",
    re.IGNORECASE,
)
_RECORDING_CONTRACT_TARGET = re.compile(
    r"\b(?:sign(?:ed|s|ing)?|join(?:ed|s|ing)?|contract(?:ed|s|ing)?|"
    r"enter(?:ed|s|ing)?\b[^?.]{0,40}\b(?:contract|agreement))\b",
    re.IGNORECASE,
)
_SINGLES_SCOPE = re.compile(
    r"\b(?:singles|individual(?:[- ]event)?\s+"
    r"(?:championship|competition|event|title|tournament))\b",
    re.IGNORECASE,
)
_FIELD_HOCKEY_SCOPE = re.compile(
    r"\b(?:field hockey|hockey (?:played|staged|conducted) on (?:a )?field)\b",
    re.IGNORECASE,
)


def _primary_answer_slot(question: str) -> tuple[str, tuple[str, ...]] | None:
    """Return the surface family of the first answer-bearing construction."""

    tokens = tuple(_LEXICAL_TOKEN.findall(" ".join(question.split())))
    if not tokens:
        return None
    folded = tuple(token.casefold() for token in tokens)
    if folded[0] in {"identify", "name", "specify", "state"}:
        return "imperative", folded[1:9]
    for index, token in enumerate(folded):
        if token in {"what", "which", "who", "whom", "whose", "where", "when"}:
            return token, folded[index + 1 : index + 9]
        if token == "how":
            return token, folded[index + 1 : index + 9]
    return None


def _birth_event_is_target_relation(question: str) -> bool:
    """Require birth lineage only when the question asks through ``born``.

    TriviaQA frequently mentions that a contextual subject was born somewhere
    before asking a different relation, such as a stage name, a featured
    singer, or a team.  Treating every occurrence of ``born`` as the target
    relation incorrectly forces those unrelated answer statements to assert a
    birth fact.  This detector remains deliberately narrow: it admits direct
    who/where/when birth questions, explicit location/time birth slots, and a
    leading identity slot whose ``born`` constraint identifies the requested
    person.  A background birth clause before the primary question does not
    match.
    """

    normalized = " ".join(question.strip().lstrip('"“').split())
    birth_match = re.search(r"\bborn\b", normalized, re.IGNORECASE)
    if birth_match is None:
        return False
    # A later interrogative is the operative request in contextual forms such
    # as ``When X was born, ... Where was he left vulnerable?``.  Do not bind
    # that later slot to the background birth clause.
    if re.search(
        r"\b(?:what|which|who|whom|whose|where|when|why|how)\b[^?]*\?",
        normalized[birth_match.end() :],
        re.IGNORECASE,
    ) is not None:
        return False

    location_or_time = "|".join(
        re.escape(head) for head in _BIRTH_LOCATION_OR_TIME_HEADS
    )
    if re.search(
        rf"\b(?:where|when)\b[^?.;]{{0,120}}\bborn\b|"
        rf"\b(?:in|on|at)\s+(?:what|which)\s+"
        rf"(?:{location_or_time})\b[^?.;]{{0,120}}\bborn\b|"
        rf"\b(?:what|which)\s+(?:{location_or_time})\b"
        rf"[^?.;]{{0,120}}\bborn\b",
        normalized,
        re.IGNORECASE,
    ) is not None:
        return True

    entity_heads = "|".join(
        re.escape(head) for head in sorted(_ENTITY_ANSWER_HEADS)
    )
    return re.match(
        rf"^(?:who\b[^?.;]{{0,100}}\bborn\b|"
        rf"(?:what|which)\s+(?:[^\W_]+(?:['’-][^\W_]+)*\s+){{0,4}}"
        rf"(?:{entity_heads})\b[^?.;]{{0,100}}\bborn\b|"
        rf"(?:what|which)\s+(?:[^\W_]+(?:['’-][^\W_]+)*\s+){{0,3}}"
        rf"born\s+(?:{entity_heads})\b)",
        normalized,
        re.IGNORECASE,
    ) is not None


def _who_answer_slot_family_preserved(original: str, paraphrase: str) -> bool:
    """Keep a person/entity WH slot from becoming an object or label slot."""

    original_slot = _primary_answer_slot(original)
    if original_slot is None or original_slot[0] not in {"who", "whom", "whose"}:
        return True
    paraphrase_slot = _primary_answer_slot(paraphrase)
    if paraphrase_slot is None:
        return False
    head, descriptors = paraphrase_slot
    if head in {"who", "whom", "whose"}:
        return True
    if head not in {"what", "which", "imperative"}:
        return False
    noun_phrase = list(descriptors)
    if head != "imperative":
        boundary_tokens = {
            "are", "can", "could", "did", "do", "does", "had", "has",
            "have", "is", "may", "might", "must", "shall", "should",
            "was", "were", "will", "would",
        }
        noun_phrase = []
        for descriptor in descriptors:
            if descriptor in boundary_tokens:
                break
            noun_phrase.append(descriptor)
    normalized_descriptors: set[str] = set()
    for descriptor in noun_phrase:
        normalized = (
            descriptor[:-2]
            if descriptor.endswith(("'s", "’s"))
            else descriptor
        )
        normalized_descriptors.add(normalized)
        normalized_descriptors.update(normalized.split("-"))
        if normalized.endswith("ies") and len(normalized) > 3:
            normalized_descriptors.add(normalized[:-3] + "y")
        elif normalized.endswith("es") and len(normalized) > 2:
            normalized_descriptors.add(normalized[:-2])
        if normalized.endswith("s") and not normalized.endswith("ss"):
            normalized_descriptors.add(normalized[:-1])
    return bool(normalized_descriptors.intersection(_ENTITY_ANSWER_HEADS))


def _adjacent_transposition(left: str, right: str) -> bool:
    """Recognize a one-swap spelling correction without a word list."""

    if len(left) != len(right) or len(left) < 4 or left == right:
        return False
    mismatches = [
        index
        for index, (left_char, right_char) in enumerate(zip(left, right))
        if left_char != right_char
    ]
    if len(mismatches) != 2 or mismatches[1] != mismatches[0] + 1:
        return False
    first, second = mismatches
    return left[first] == right[second] and left[second] == right[first]


def _contains_source_spelling_correction(original: str, candidate: str) -> bool:
    """Reject silently correcting a source token by transposing its letters."""

    original_tokens = {
        token.casefold() for token in _LEXICAL_TOKEN.findall(original)
    }
    candidate_tokens = {
        token.casefold() for token in _LEXICAL_TOKEN.findall(candidate)
    }
    removed = original_tokens - candidate_tokens
    added = candidate_tokens - original_tokens
    return any(
        _adjacent_transposition(source_token, candidate_token)
        for source_token in removed
        for candidate_token in added
    )


def _restore_authoritative_source_transpositions(
    original: str,
    candidate: str,
    *,
    protected_span: str | None = None,
) -> str:
    """Undo only unambiguous adjacent-transposition source corrections.

    TriviaQA source wording remains authoritative even when a language model
    believes it is misspelled.  This repair preserves the generated sentence
    and substitutes an original surface token only when exactly one missing
    source token is an adjacent-transposition match for the introduced token.
    The repaired candidate still passes through the complete admission gate.
    """

    original_surfaces: dict[str, str] = {}
    for token in _LEXICAL_TOKEN.findall(original):
        original_surfaces.setdefault(token.casefold(), token)
    candidate_tokens = {
        token.casefold() for token in _LEXICAL_TOKEN.findall(candidate)
    }
    missing = {
        folded: surface
        for folded, surface in original_surfaces.items()
        if folded not in candidate_tokens
    }

    def restore(match: re.Match[str]) -> str:
        surface = match.group(0)
        folded = surface.casefold()
        matches = [
            original_surface
            for original_folded, original_surface in missing.items()
            if _adjacent_transposition(original_folded, folded)
        ]
        return matches[0] if len(matches) == 1 else surface

    if protected_span and protected_span in candidate:
        return protected_span.join(
            _LEXICAL_TOKEN.sub(restore, segment)
            for segment in candidate.split(protected_span)
        )
    return _LEXICAL_TOKEN.sub(restore, candidate)


def _source_destination_relation_preserved(
    original: str,
    paraphrase: str,
) -> bool:
    """Keep a location slot attached to the same motion-event argument."""

    normalized_original = " ".join(original.split())
    normalized_paraphrase = " ".join(paraphrase.split())
    source_slot = re.search(
        r"\b(?:leave|left|depart(?:ed|s|ing)?)\s+"
        r"(?:where|(?:which|what)\s+(?:place|location|country|nation|city|state))\b",
        normalized_original,
        re.IGNORECASE,
    )
    destination_rewrite = re.search(
        r"\b(?:(?:in|to)\s+(?:which|what)\s+(?:"
        + "|".join(_LOCATION_SLOT_HEADS)
        + r")|where)\b[^?]{0,120}\b(?:go|travel|arrive|return)\b",
        normalized_paraphrase,
        re.IGNORECASE,
    )
    if source_slot is not None and destination_rewrite is not None:
        return False

    # Some raw TriviaQA questions place a location WH at the end of an
    # ``after <event> where?`` subordinate clause.  Do not reattach that WH to
    # a main-clause motion destination (``where did X return/go ...``).
    subordinate_location_slot = re.search(
        r"\bafter\b[^?]+\bwhere\s*\?\s*$",
        normalized_original,
        re.IGNORECASE,
    )
    motion_destination_slot = re.search(
        r"\bwhere\s+did\s+(?:(?:the|a|an)\s+\w+\s+|"
        r"[A-Z][^\s?,]*\s+){1,5}(?:return|go|travel|arrive)\b",
        normalized_paraphrase,
    )
    return not (
        subordinate_location_slot is not None
        and motion_destination_slot is not None
    )


def _semantic_relation_and_scope_preserved(
    *,
    original_question: str,
    paraphrase_question: str,
    paraphrase_answer_statement: str,
    canonical_answer: str | None = None,
) -> None:
    """Fail closed on deterministic, high-confidence semantic drift."""

    if not _who_answer_slot_family_preserved(
        original_question,
        paraphrase_question,
    ):
        raise SemanticPreservationError(
            "paraphrase changed the answer-slot family or predicate argument"
        )
    if not _source_destination_relation_preserved(
        original_question,
        paraphrase_question,
    ):
        raise SemanticPreservationError(
            "paraphrase changed a source/destination relation"
        )
    if _birth_event_is_target_relation(original_question):
        if _BIRTH_EVENT.search(paraphrase_question) is None or _BIRTH_EVENT.search(
            paraphrase_answer_statement
        ) is None:
            raise SemanticPreservationError(
                "paraphrase weakened the birth-event relation"
            )
    if _RECORDING_CONTRACT_SOURCE.search(original_question) is not None:
        if _RECORDING_CONTRACT_TARGET.search(
            paraphrase_question
        ) is None or _RECORDING_CONTRACT_TARGET.search(
            paraphrase_answer_statement
        ) is None:
            raise SemanticPreservationError(
                "paraphrase changed the recording-contract relation"
            )
    if re.search(r"\bsingles\b", original_question, re.IGNORECASE) is not None:
        if _SINGLES_SCOPE.search(paraphrase_question) is None:
            raise SemanticPreservationError(
                "paraphrase broadened the singles-event scope"
            )
    if re.search(r"\bfield hockey\b", original_question, re.IGNORECASE) is not None:
        if _FIELD_HOCKEY_SCOPE.search(paraphrase_question) is None:
            raise SemanticPreservationError(
                "paraphrase removed the field-hockey discipline"
            )
    if re.search(r"\bstar sign\b", original_question, re.IGNORECASE) is not None:
        if re.search(
            r"\b(?:star|zodiac) sign\b",
            paraphrase_question,
            re.IGNORECASE,
        ) is None:
            raise SemanticPreservationError(
                "paraphrase changed the zodiac star-sign scope"
            )
    if re.search(
        r"\btype of what\s*\?\s*$",
        original_question,
        re.IGNORECASE,
    ) is not None:
        introduced_kind = re.search(
            r"\b(?:what|which)\s+(?:kind|type|category)\s+of\s+"
            r"([A-Za-z][A-Za-z'-]*)",
            paraphrase_question,
            re.IGNORECASE,
        )
        if (
            introduced_kind is not None
            and re.search(
                rf"\b{re.escape(introduced_kind.group(1))}\b",
                original_question,
                re.IGNORECASE,
            )
            is None
        ):
            raise SemanticPreservationError(
                "paraphrase narrowed a generic type-of-what answer slot"
            )
    if (
        re.search(r"\b(?:what|which)\s+date\b", original_question, re.IGNORECASE)
        is not None
        and re.search(
            r"\bday of (?:the )?month\b",
            paraphrase_question,
            re.IGNORECASE,
        )
        is not None
    ):
        raise SemanticPreservationError(
            "paraphrase narrowed calendar-date granularity to day-of-month"
        )
    statement_for_spelling_check = paraphrase_answer_statement
    if canonical_answer and canonical_answer in statement_for_spelling_check:
        statement_for_spelling_check = "".join(
            statement_for_spelling_check.split(canonical_answer)
        )
    if _contains_source_spelling_correction(
        original_question,
        paraphrase_question,
    ) or _contains_source_spelling_correction(
        original_question,
        statement_for_spelling_check,
    ):
        raise SemanticPreservationError(
            "paraphrase silently corrected an authoritative source token"
        )


def _content_token_counts(text: str) -> Counter[str]:
    return Counter(
        token.casefold()
        for token in _LEXICAL_TOKEN.findall(text)
        if token.casefold() not in _FUNCTION_WORDS
    )


def _identity_token_preserved(
    required: str,
    observed_tokens: frozenset[str],
) -> bool:
    """Permit adding possessive syntax, but never remove it from an entity."""

    normalized_required = required.replace("’", "'")
    normalized_observed = frozenset(
        token.replace("’", "'") for token in observed_tokens
    )
    if normalized_required in normalized_observed:
        return True
    if normalized_required.endswith("'s"):
        return False
    return f"{normalized_required}'s" in normalized_observed


def _has_lexical_or_phrase_replacement(original: str, paraphrase: str) -> bool:
    """Reject a word-order-only rewrite at the materialization boundary."""

    def comparison_token(token: str) -> str:
        normalized = token.casefold()
        return (
            normalized[:-2]
            if normalized.endswith(("'s", "’s"))
            else normalized
        )

    original_surface = Counter(
        comparison_token(token) for token in _LEXICAL_TOKEN.findall(original)
    )
    paraphrase_surface = Counter(
        comparison_token(token) for token in _LEXICAL_TOKEN.findall(paraphrase)
    )
    original_counts = Counter(
        comparison_token(token)
        for token in _LEXICAL_TOKEN.findall(original)
        if token.casefold() not in _FUNCTION_WORDS
    )
    paraphrase_counts = Counter(
        comparison_token(token)
        for token in _LEXICAL_TOKEN.findall(paraphrase)
        if token.casefold() not in _FUNCTION_WORDS
    )
    # A relation-bearing phrase can replace a copula or interrogative phrase
    # without removing another content token, for example ``What is the
    # capital ...`` -> ``Which city serves as the capital ...``. Require a
    # removed surface token plus a newly introduced content token. A dangling
    # generic interrogative can also be expanded by two or more content words
    # without removing another token. Pure word reordering has identical
    # multisets and therefore still fails closed.
    removed_surface = original_surface - paraphrase_surface
    added_content = paraphrase_counts - original_counts
    return bool(added_content) and (
        bool(removed_surface) or sum(added_content.values()) >= 2
    )


def _participation_marker_preserved(original: str, paraphrase: str) -> bool:
    """Preserve co-participation when paraphrasing a founding relation."""

    if re.search(r"\bco[- ]?found(?:ed|er|ing)?\b", original, re.IGNORECASE) is None:
        return True
    return re.search(
        r"\b(?:co[- ]?found(?:ed|er|ing)?|"
        r"help(?:ed|ing|s)?\s+(?:to\s+)?(?:establish|found)|"
        r"jointly\s+(?:establish(?:ed|es|ing)?|found(?:ed|s|ing)?))\b",
        paraphrase,
        re.IGNORECASE,
    ) is not None


def _answer_slot_count(question: str) -> int:
    """Count requested answers without treating relative clauses as slots."""

    # TriviaQA records request one answer unless another interrogative is
    # explicitly coordinated (for example, "which film, and who directed
    # it?").  Counting every wh-token incorrectly treats relative clauses such
    # as "the story upon which ..." as a second answer request.
    wh_count = len(_QUESTION_SLOT_TOKEN.findall(question))
    if wh_count <= 1:
        return 1
    return 1 + len(_COORDINATED_QUESTION_SLOT.findall(question))


def _leading_answer_slot_anchor(source: TriviaQATrainSource) -> str | None:
    """Return an alias-grounded disambiguator from a leading wh-phrase."""

    tokens = _LEXICAL_TOKEN.findall(" ".join(source.original_question.split()))
    if (
        len(tokens) < 2
        or tokens[0].casefold() not in {"what", "which"}
        or not tokens[1][0].isupper()
    ):
        return None
    anchor = tokens[1]
    # ``Which Iain Banks novel ...`` and ``Which Great Fire ...`` begin with
    # a multi-token proper-name modifier, not an alias-grounded answer-slot
    # restriction.  Requiring the second source token to end that proper-name
    # run preserves the original ``Which Gloria co-founded ...`` contract
    # without pinning an entity modifier to the front of every paraphrase.
    if len(tokens) > 2 and tokens[2][:1].isupper():
        return None
    canonical_tokens = {
        token.casefold() for token in _LEXICAL_TOKEN.findall(source.canonical_answer)
    }
    for accepted_answer in source.accepted_answers_for_admission:
        accepted_tokens = {
            token.casefold() for token in _LEXICAL_TOKEN.findall(accepted_answer)
        }
        if anchor.casefold() in accepted_tokens and canonical_tokens.issubset(
            accepted_tokens
        ):
            return anchor
    return None


def _leading_answer_slot_anchor_preserved(
    source: TriviaQATrainSource,
    paraphrase_question: str,
) -> bool:
    """Keep a leading disambiguator bound to the rewritten answer slot.

    This prevents a rewrite such as ``Which Gloria ...?`` -> ``Who ...
    alongside Gloria?``.  The latter preserves the surface token but changes
    its semantic role from answer-slot restriction to downstream participant.
    """

    anchor = _leading_answer_slot_anchor(source)
    if anchor is None:
        return True
    tokens = _LEXICAL_TOKEN.findall(" ".join(paraphrase_question.split()))
    folded = [token.casefold() for token in tokens]
    anchor_folded = anchor.casefold()
    if len(folded) >= 2 and folded[0] in {"what", "which"}:
        return folded[1] == anchor_folded
    if len(folded) >= 3 and folded[0] in {"who", "whom", "whose"}:
        if folded[1] not in {"are", "is", "was", "were"}:
            return False
        offset = 3 if folded[2] in {"a", "an", "the"} else 2
        return len(folded) > offset and folded[offset] == anchor_folded
    if folded and folded[0] in {"identify", "name", "specify", "state"}:
        offset = 1
        if len(folded) > offset and folded[offset] in {
            "a",
            "an",
            "the",
            "what",
            "which",
        }:
            offset += 1
        return len(folded) > offset and folded[offset] == anchor_folded
    if (
        len(folded) >= 3
        and folded[0] in {"at", "by", "from", "in", "on"}
        and folded[1] in {"what", "which"}
    ):
        return folded[2] == anchor_folded
    return False


def _subject_wh_anchor_requires_literal_binding(
    source: TriviaQATrainSource,
) -> bool:
    """Detect an alias-grounded subject-WH slot followed by its predicate."""

    anchor = _leading_answer_slot_anchor(source)
    tokens = _LEXICAL_TOKEN.findall(source.original_question)
    if anchor is None or len(tokens) < 3:
        return False
    return tokens[2].casefold() not in {
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
    }


def _literal_subject_wh_answer_statement(
    source: TriviaQATrainSource,
) -> str | None:
    """Bind the canonical span to an admitted leading subject-WH slot."""

    if not _subject_wh_anchor_requires_literal_binding(source):
        return None
    match = re.fullmatch(
        r"(?i:what|which)\s+\S+\s+(.+?)\?",
        " ".join(source.original_question.split()),
    )
    if match is None:
        return None
    return f"{source.canonical_answer} {match.group(1)}."


def _possessive_name_answer_statement(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Return the source possessive span and its literal name-relation answer."""

    match = re.fullmatch(
        r"(?i:what|which)\s+(?P<copula>is|was)\s+"
        r"(?P<subject>.+?['’]s)\s+"
        r"(?P<modifier>actual|birth|full|real)\s+name\?",
        " ".join(source.original_question.split()),
    )
    if match is None:
        return None
    subject = match.group("subject")
    statement = (
        f"{subject} {match.group('modifier')} name "
        f"{match.group('copula')} {source.canonical_answer}."
    )
    return subject, statement


_ATOMIC_COUNT_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "dozen",
        "score",
        "and",
    }
)
_DETERMINISTIC_SUBJECT_WH_ANSWER_HEADS = frozenset(
    {
        *_ENTITY_ANSWER_HEADS,
        *_LOCATION_SLOT_HEADS,
        "album",
        "animal",
        "book",
        "club",
        "film",
        "language",
        "movie",
        "novel",
        "play",
        "rank",
        "river",
        "score",
        "song",
        "species",
        "station",
        "term",
        "title",
        "word",
    }
)


def _atomic_count_canonical(canonical_answer: str) -> bool:
    """Admit only a short numeric or number-word answer to a count slot."""

    if re.search(r"[;?!]", canonical_answer) is not None:
        return False
    tokens = tuple(
        token.casefold() for token in _LEXICAL_TOKEN.findall(canonical_answer)
    )
    if not tokens or len(tokens) > 5:
        return False
    if any(character.isdigit() for character in canonical_answer):
        return re.fullmatch(
            r"\s*[+-]?(?:\d[\d,.]*)(?:\s+(?:hundred|thousand|million|"
            r"billion))?\s*",
            canonical_answer,
            re.IGNORECASE,
        ) is not None
    return all(
        part in _ATOMIC_COUNT_WORDS
        for token in tokens
        for part in token.replace("-", " ").split()
    )


def _declarative_statement(text: str) -> str:
    """Terminate a copied declarative without altering canonical punctuation."""

    normalized = " ".join(text.split()).rstrip()
    if re.match(r"(?i:a|an|the)\b", normalized) is not None:
        normalized = normalized[0].upper() + normalized[1:]
    if re.search(r"[.!?][\"'”’]?\Z", normalized) is not None:
        return normalized
    return normalized + "."


def _simple_subject_wh_head(head: str) -> bool:
    """Require one short answer-type noun phrase, not a hidden possessor slot."""

    tokens = tuple(
        token.casefold() for token in _LEXICAL_TOKEN.findall(head)
    )
    if not tokens or len(tokens) > 12:
        return False
    if any(token.endswith(("'s", "’s")) for token in tokens):
        return False
    return tokens[-1] in _DETERMINISTIC_SUBJECT_WH_ANSWER_HEADS


def _deterministic_answer_slot_statement(
    source: TriviaQATrainSource,
) -> str | None:
    """Fill one structurally unambiguous TriviaQA answer slot.

    These templates are a bounded repair inside the existing materialization
    path.  They copy the authoritative relation and scope from the source,
    insert the exact canonical span once, and fail closed for coordinated or
    dependency-ambiguous questions.  The returned candidate is still parsed
    by :func:`parse_paraphrase_response` and checked by the existing answer
    verifier before admission.
    """

    original = " ".join(source.original_question.split())
    if not original or _answer_slot_count(original) != 1:
        return None
    body = original[:-1].rstrip() if original.endswith(("?", ".")) else original
    canonical = source.canonical_answer
    candidate: str | None = None
    internal_question_mark = "?" in body
    if _is_explicit_listed_choice_question(
        original_question=original,
        canonical_answer=canonical,
    ):
        return None

    name_of = re.fullmatch(
        r"(?i:what)\s+(?P<copula>is|was)\s+the\s+name\s+of\s+"
        r"(?P<referent>.+)",
        body,
    )
    if name_of is not None and not internal_question_mark:
        candidate = _declarative_statement(
            f"The name of {name_of.group('referent')} "
            f"{name_of.group('copula')} {canonical}"
        )

    if candidate is None:
        leading_called = re.fullmatch(
            r"(?i:what)\s+(?P<copula>is|was|are|were)\s+"
            r"(?P<subject>.+?)\s+"
            r"(?P<predicate>(?:(?:better|commonly|formerly|popularly)\s+)?"
            r"(?:called|named|known\s+as))",
            body,
        )
        if leading_called is not None and not internal_question_mark:
            candidate = _declarative_statement(
                f"{leading_called.group('subject')} "
                f"{leading_called.group('copula')} "
                f"{leading_called.group('predicate')} {canonical}"
            )

    if candidate is None:
        trailing_known_name = re.fullmatch(
            r"(?P<subject>.+?)\s+(?P<copula>(?i:is|was|are|were))\s+"
            r"(?P<predicate>(?i:(?:(?:better|commonly|formerly|popularly)\s+)?"
            r"known))\s+(?P<preposition>(?i:by|under))\s+"
            r"(?i:what)\s+(?i:name)",
            body,
        )
        if trailing_known_name is not None and not internal_question_mark:
            candidate = _declarative_statement(
                f"{trailing_known_name.group('subject')} "
                f"{trailing_known_name.group('copula')} "
                f"{trailing_known_name.group('predicate')} "
                f"{trailing_known_name.group('preposition')} the name "
                f"{canonical}"
            )

    if candidate is None:
        fronted_known_name = re.fullmatch(
            r"(?i:what)\s+(?i:name)\s+"
            r"(?P<copula>(?i:is|was|are|were))\s+"
            r"(?P<subject>.+?)\s+"
            r"(?P<predicate>(?i:(?:(?:better|commonly|formerly|popularly)\s+)?"
            r"known))\s+(?P<preposition>(?i:by|under))",
            body,
        )
        if fronted_known_name is not None and not internal_question_mark:
            candidate = _declarative_statement(
                f"{fronted_known_name.group('subject')} "
                f"{fronted_known_name.group('copula')} "
                f"{fronted_known_name.group('predicate')} "
                f"{fronted_known_name.group('preposition')} the name "
                f"{canonical}"
            )

    if candidate is None:
        subject_called = re.fullmatch(
            r"(?i:what|which)\s+(?P<head>.+?)\s+"
            r"(?P<copula>is|was|are|were)\s+"
            r"(?P<predicate>(?:(?:better|commonly|formerly|popularly)\s+)?"
            r"(?:called|named|known\s+as))\s+"
            r"(?P<complement>.+)",
            body,
        )
        if (
            subject_called is not None
            and not internal_question_mark
            and _simple_subject_wh_head(subject_called.group("head"))
        ):
            candidate = _declarative_statement(
                f"{canonical} {subject_called.group('copula')} "
                f"{subject_called.group('predicate')} "
                f"{subject_called.group('complement')}"
            )

    if candidate is None and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1:
        how_many = re.fullmatch(r"(?i:how\s+many)\s+(?P<tail>.+)", body)
        if (
            how_many is not None
            and not internal_question_mark
            and _atomic_count_canonical(canonical)
        ):
            tail = how_many.group("tail")
            perfect_existential = re.fullmatch(
                r"(?P<head>.+?)\s+(?P<aux>have|has|had)\s+there\s+been"
                r"(?P<rest>.*)",
                tail,
                re.IGNORECASE,
            )
            existential = re.fullmatch(
                r"(?P<head>.+?)\s+(?P<aux>are|were|is|was)\s+there"
                r"(?P<rest>.*)",
                tail,
                re.IGNORECASE,
            )
            predicative = re.fullmatch(
                r"(?P<head>.+?)\s+(?P<aux>are|were|is|was)\s+"
                r"(?P<rest>.+)",
                tail,
                re.IGNORECASE,
            )
            if perfect_existential is not None:
                candidate = _declarative_statement(
                    f"There {perfect_existential.group('aux')} been {canonical} "
                    f"{perfect_existential.group('head')}"
                    f"{perfect_existential.group('rest')}"
                )
            elif existential is not None:
                candidate = _declarative_statement(
                    f"There {existential.group('aux')} {canonical} "
                    f"{existential.group('head')}{existential.group('rest')}"
                )
            elif predicative is not None:
                candidate = _declarative_statement(
                    f"{canonical} {predicative.group('head')} "
                    f"{predicative.group('aux')} {predicative.group('rest')}"
                )

    if candidate is None and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1:
        trailing_slot = re.fullmatch(
            r"(?P<prefix>.+?)\s+(?P<slot>(?i:what|where))",
            body,
        )
        if trailing_slot is not None and not internal_question_mark:
            literal_candidate = _declarative_statement(
                f"{trailing_slot.group('prefix')} {canonical}"
            )
            if _literal_slot_substitution_preserved(
                original_question=original,
                canonical_answer=canonical,
                answer_statement=literal_candidate,
            ):
                candidate = literal_candidate

    if candidate is None and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1:
        # A sentence-final prepositional object is the one embedded-WH shape
        # for which literal substitution is dependency-unambiguous.  The
        # non-empty clause before the preposition deliberately excludes
        # fronted forms such as ``On which street did ...?``.  The existing
        # token-level proof below must still show that one contiguous WH
        # constituent, and nothing else, was replaced by the canonical span.
        trailing_preposition_slot = re.fullmatch(
            r"(?P<prefix>.+?\s+(?:in|from|by|at|on|for|under|with|to)\s+)"
            r"(?P<slot>(?:what|which)\s+.+)",
            body,
            re.IGNORECASE,
        )
        if (
            trailing_preposition_slot is not None
            and not internal_question_mark
            and _COORDINATED_QUESTION_SLOT.search(original) is None
        ):
            canonical_object = (
                f'"{canonical}"'
                if re.search(r"[?!]", canonical) is not None
                and '"' not in canonical
                else canonical
            )
            literal_candidate = _declarative_statement(
                f"{trailing_preposition_slot.group('prefix')}"
                f"{canonical_object}"
            )
            if _literal_slot_substitution_preserved(
                original_question=original,
                canonical_answer=canonical,
                answer_statement=literal_candidate,
            ):
                candidate = literal_candidate

    if candidate is None:
        completion = re.fullmatch(
            r"(?P<operator>(?i:complete|finish))\s+(?P<object>.+)",
            body,
        )
        if (
            completion is not None
            and _quoted_spans(original)
            and re.search(
                r"\b(?:title|quote|quotation|proverb|lyric|line|phrase|"
                r"saying|sentence)\b",
                completion.group("object"),
                re.IGNORECASE,
            )
            is not None
            and _COORDINATED_QUESTION_SLOT.search(original) is None
        ):
            completion_object = re.sub(
                r"\s+(?:what|who|where)\s*\Z",
                "",
                completion.group("object"),
                flags=re.IGNORECASE,
            ).rstrip()
            if completion_object:
                if completion.group("operator").casefold() == "complete":
                    candidate = _declarative_statement(
                        f"The completion of {completion_object} is {canonical}"
                    )
                else:
                    candidate = _declarative_statement(
                        f"The ending of {completion_object} is {canonical}"
                    )

    if candidate is None and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1:
        who_subject = re.fullmatch(r"(?i:who)\s+(?P<predicate>.+)", body)
        if who_subject is not None and not internal_question_mark:
            first = _LEXICAL_TOKEN.search(who_subject.group("predicate"))
            if first is not None and first.group(0).casefold() not in {
                "did",
                "do",
                "does",
            }:
                candidate = _declarative_statement(
                    f"{canonical} {who_subject.group('predicate')}"
                )

    if candidate is None and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1:
        what_copular = re.fullmatch(
            r"(?i:what)\s+(?P<copula>is|was)\s+"
            r"(?P<complement>the\s+.+)",
            body,
        )
        if what_copular is not None and not internal_question_mark:
            candidate = _declarative_statement(
                f"{what_copular.group('complement')} "
                f"{what_copular.group('copula')} {canonical}"
            )

    if candidate is None and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1:
        subject_wh = re.fullmatch(
            r"(?i:what|which)\s+(?P<head>.+?)\s+"
            rf"(?P<verb>{_DETERMINISTIC_SUBJECT_RELATIVE_VERB.pattern}|"
            r"is|was|are|were)\s+(?P<tail>.+)",
            body,
            re.IGNORECASE,
        )
        if subject_wh is not None:
            if (
                not internal_question_mark
                and _simple_subject_wh_head(subject_wh.group("head"))
            ):
                candidate = _declarative_statement(
                    f"{canonical} is the {subject_wh.group('head')} that "
                    f"{subject_wh.group('verb')} {subject_wh.group('tail')}"
                )

    if candidate is None:
        return None
    candidate = _quote_terminal_punctuated_canonical_span(
        candidate,
        canonical,
    )
    candidate = _repair_fact_terminal_punctuation(
        candidate,
        protected_span=canonical,
    )
    source_quotes = _quoted_spans(source.original_question)
    if source_quotes and not source_quotes.issubset(_quoted_spans(candidate)):
        return None
    if not exact_canonical_span_preserved(candidate, canonical):
        return None
    if not relation_bearing_answer_statement(candidate, canonical):
        return None
    if not _answer_statement_has_lexical_relation_lineage(
        original_question=original,
        canonical_answer=canonical,
        answer_statement=candidate,
    ):
        return None
    return candidate


def _clausal_canonical_relation_statement(
    source: TriviaQATrainSource,
) -> str | None:
    """Bind a clausal canonical label to a ``What X did Y have`` slot."""

    canonical_tokens = _LEXICAL_TOKEN.findall(source.canonical_answer)
    if (
        len(canonical_tokens) < 3
        or canonical_tokens[0].casefold()
        not in {"he", "i", "it", "she", "they", "we", "you"}
    ):
        return None
    match = re.fullmatch(
        r"(?i:what)\s+(?P<slot>.+?)\s+(?i:did)\s+"
        r"(?P<subject>.+?)\s+(?i:have)\?",
        " ".join(source.original_question.split()),
    )
    if match is None:
        return None
    return (
        f'The statement "{source.canonical_answer}" identifies the '
        f"{match.group('slot')} that {match.group('subject')} had."
    )


def _original_interrogative_head_omitted(question: str) -> bool:
    """Detect ``preposition + which + named subject`` malformed questions."""

    return re.search(
        r"\b(?i:in|on|at|by|from|during)\s+(?i:which)\s+[A-Z][^?]*\?\Z",
        " ".join(question.split()),
    ) is not None


def _canonical_answer_is_explicit_compound(canonical_answer: str) -> bool:
    normalized = " ".join(canonical_answer.split()).casefold()
    return " & " in normalized or " and " in normalized


def _literal_slot_substitution_preserved(
    *,
    original_question: str,
    canonical_answer: str,
    answer_statement: str,
) -> bool:
    """Prove exact wh-constituent replacement without model semantics.

    This is a fail-closed fallback for malformed dataset questions that make a
    world-knowledge verifier unreliable.  Apart from one contiguous canonical
    answer span, the declarative statement must copy an exact prefix and suffix
    of the original token sequence; the removed middle must contain a wh-token.
    """

    original_tokens = tuple(
        token.casefold() for token in _LEXICAL_TOKEN.findall(original_question)
    )
    canonical_tokens = tuple(
        token.casefold() for token in _LEXICAL_TOKEN.findall(canonical_answer)
    )
    statement_tokens = tuple(
        token.casefold() for token in _LEXICAL_TOKEN.findall(answer_statement)
    )
    if not original_tokens or not canonical_tokens or not statement_tokens:
        return False
    wh_tokens = {
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
    width = len(canonical_tokens)
    for start in range(len(statement_tokens) - width + 1):
        if statement_tokens[start : start + width] != canonical_tokens:
            continue
        prefix = statement_tokens[:start]
        suffix = statement_tokens[start + width :]
        if original_tokens[: len(prefix)] != prefix:
            continue
        if suffix and original_tokens[-len(suffix) :] != suffix:
            continue
        middle_start = len(prefix)
        middle_stop = len(original_tokens) - len(suffix)
        if middle_stop <= middle_start:
            continue
        removed = original_tokens[middle_start:middle_stop]
        if any(token in wh_tokens for token in removed):
            return True
    return False


def _called_relation_substitution_preserved(
    *,
    original_question: str,
    canonical_answer: str,
    answer_statement: str,
) -> bool:
    """Prove the deterministic ``What was X called?`` declarative form."""

    match = re.fullmatch(
        r"(?i:what)\s+(?i:was)\s+(.+?)\s+(?i:called)\?",
        " ".join(original_question.split()),
    )
    if match is None or not exact_canonical_span_preserved(
        answer_statement,
        canonical_answer,
    ):
        return False
    expected_tokens = tuple(
        token.casefold()
        for token in _LEXICAL_TOKEN.findall(
            f"{match.group(1)} was called {canonical_answer}"
        )
    )
    observed_tokens = tuple(
        token.casefold() for token in _LEXICAL_TOKEN.findall(answer_statement)
    )
    return observed_tokens == expected_tokens


def _answer_statement_has_lexical_relation_lineage(
    *,
    original_question: str,
    canonical_answer: str,
    answer_statement: str,
) -> bool:
    """Require at least one non-answer relation token from the source question."""

    relation_tokens = set(_content_token_counts(original_question)) - set(
        _content_token_counts(canonical_answer)
    )
    statement_tokens = set(_content_token_counts(answer_statement)) - set(
        _content_token_counts(canonical_answer)
    )
    return bool(relation_tokens & statement_tokens)


def _canonicalize_answer_statement_from_accepted_alias(
    source: TriviaQATrainSource,
    answer_statement: str,
) -> str | None:
    """Replace one unambiguous train alias with the exact canonical span.

    This is an admission-only repair over aliases already supplied by the
    TriviaQA training row.  It does not infer aliases or rewrite surrounding
    statement text.  Nested aliases are resolved by their unique longest span;
    repeated, disjoint, or partially overlapping candidates remain fail-closed.
    ASCII and curly apostrophe glyphs are the sole generated orthographic
    variants because they preserve the same lexical span.
    """

    if not isinstance(answer_statement, str) or not answer_statement.strip():
        return None
    canonical = source.canonical_answer
    if exact_canonical_span_preserved(answer_statement, canonical):
        return None

    def apostrophe_variants(text: str) -> tuple[str, ...]:
        variants = {text}
        if "'" in text:
            variants.add(text.replace("'", "’"))
        if "’" in text:
            variants.add(text.replace("’", "'"))
        return tuple(sorted(variants, key=lambda value: (-len(value), value)))

    aliases: list[str] = []
    for admitted_answer in source.accepted_answers_for_admission:
        for alias in apostrophe_variants(admitted_answer):
            if alias == canonical or alias in aliases:
                continue
            alias_tokens = tuple(_LEXICAL_TOKEN.findall(alias))
            if not (
                any(character.isdigit() for character in alias)
                or any(
                    token.casefold() not in _FUNCTION_WORDS
                    for token in alias_tokens
                )
            ):
                continue
            aliases.append(alias)
    if not aliases:
        return None

    matches: list[tuple[int, int, str]] = []
    for alias in aliases:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        alias_matches = tuple(pattern.finditer(answer_statement))
        if not alias_matches:
            continue
        if any(
            re.search(
                rf"(?<!\w){re.escape(question_alias)}(?!\w)",
                source.original_question,
                re.IGNORECASE,
            )
            is not None
            for question_alias in apostrophe_variants(alias)
        ):
            return None
        matches.extend(
            (match.start(), match.end(), alias) for match in alias_matches
        )
    if not matches:
        return None

    longest_width = max(stop - start for start, stop, _ in matches)
    longest_spans = {
        (start, stop)
        for start, stop, _ in matches
        if stop - start == longest_width
    }
    if len(longest_spans) != 1:
        return None
    selected_start, selected_stop = next(iter(longest_spans))
    # Shorter aliases wholly nested in the unique longest span are the reason
    # for longest-span selection, not an ambiguity.  Any other match means the
    # statement contains a second or partially overlapping candidate.
    if any(
        (start, stop) != (selected_start, selected_stop)
        and not (
            selected_start <= start
            and stop <= selected_stop
        )
        for start, stop, _ in matches
    ):
        return None
    return (
        answer_statement[:selected_start]
        + canonical
        + answer_statement[selected_stop:]
    )


def _listed_choice_answer_binding_preserved(
    *,
    original_question: str,
    canonical_answer: str,
    answer_statement: str,
) -> bool:
    """Require an explicit option binding for listed-choice questions.

    The canonical TriviaQA label can be an uninflected form such as
    ``Triple`` while the natural declarative predicate is ``tripled``.  A
    question restatement followed by a bare label is not a grounded answer
    statement, so the exact label must be attached to an explicit selected
    option/choice/answer phrase.
    """

    question = " ".join(original_question.split())
    canonical = " ".join(canonical_answer.split())
    statement = " ".join(answer_statement.split())
    is_listed_choice = _is_explicit_listed_choice_question(
        original_question=question,
        canonical_answer=canonical,
    )
    if not is_listed_choice:
        return True
    if not exact_canonical_span_preserved(statement, canonical) or "?" in statement:
        return False
    canonical_pattern = re.escape(canonical)
    selected_label = re.compile(
        rf"\bselected\b[^.;:]*\b(?:option|choice|answer)\b[^.;:]*"
        rf"{canonical_pattern}(?:\b|\Z)",
        re.IGNORECASE,
    )
    label_selected = re.compile(
        rf"{canonical_pattern}(?:\b|\Z)[^.;:]*\b(?:option|choice|answer)\b"
        rf"[^.;:]*\bselected\b",
        re.IGNORECASE,
    )
    return bool(selected_label.search(statement) or label_selected.search(statement))


def _is_explicit_listed_choice_question(
    *,
    original_question: str,
    canonical_answer: str,
) -> bool:
    question = " ".join(original_question.split()).casefold().replace("’", "'")
    canonical = " ".join(canonical_answer.split()).casefold().replace("’", "'")
    return " or " in question and canonical in question


def _augment_listed_choice_answer_statement(
    *,
    original_question: str,
    canonical_answer: str,
    rejected_answer_statement: str,
) -> str | None:
    """Attach the exact training label to an otherwise declarative predicate."""

    if not _is_explicit_listed_choice_question(
        original_question=original_question,
        canonical_answer=canonical_answer,
    ):
        return None
    statement = " ".join(rejected_answer_statement.split()).strip()
    if not statement or "?" in statement:
        return None
    statement = statement.rstrip(" .;:")
    if not statement:
        return None
    return (
        f"{statement}; the selected listed option is {canonical_answer}."
    )


def _capitalized_identity_tokens(question: str) -> frozenset[str]:
    matches = tuple(_LEXICAL_TOKEN.finditer(question))
    tokens = tuple(match.group(0) for match in matches)
    return frozenset(
        token.casefold()
        for index, (token, match) in enumerate(zip(tokens, matches))
        if (
            (
                token[:1].isupper()
                and token.casefold() not in _FUNCTION_WORDS
                and not _sentence_boundary_non_entity_token(
                    question,
                    match,
                )
            )
            or (
                token.casefold() in _NAME_PARTICLES
                and index > 0
                and index + 1 < len(tokens)
                and tokens[index - 1][:1].isupper()
                and tokens[index + 1][:1].isupper()
            )
        )
    )


def _sentence_boundary_non_entity_token(
    text: str,
    token_match: re.Match[str],
) -> bool:
    """Exclude only clear discourse operators at a sentence boundary.

    Capitalization alone makes a leading imperative or ordinary contraction
    look like a named entity.  Keep the exception position-sensitive instead
    of globally whitelisting the surface: the same token elsewhere remains
    eligible for identity protection, while quoted spans continue to be
    protected independently by the immutable quote gate.
    """

    prefix = text[: token_match.start()].rstrip()
    while prefix.endswith(('"', "“", "'", "‘", "(", "[", "{")):
        prefix = prefix[:-1].rstrip()
    at_boundary = not prefix or prefix[-1] in ".!?;:"
    if not at_boundary:
        return False
    token = token_match.group(0)
    if _SENTENCE_BOUNDARY_CONTRACTION.fullmatch(token) is not None:
        next_token = _LEXICAL_TOKEN.search(text, token_match.end())
        # Preserve title-like capitalization such as ``Don't Panic``.  The
        # exception is only for ordinary sentence-boundary contractions such
        # as ``Don't guess`` or ``I'm ready``.
        return next_token is None or not next_token.group(0)[:1].isupper()
    return bool(
        _LEADING_INTERROGATIVE_CONTRACTION.fullmatch(token)
        or token.casefold() in _SENTENCE_BOUNDARY_IMPERATIVE
        or token.casefold() in _SENTENCE_BOUNDARY_NON_ENTITY_WORDS
    )


def _multiword_possessive_identity_bases(question: str) -> frozenset[str]:
    """Permit ``X Y's`` to become ``of X Y`` without weakening brand names."""

    tokens = _LEXICAL_TOKEN.findall(question)
    bases: set[str] = set()
    for index, token in enumerate(tokens):
        if index == 0 or not token.endswith(("'s", "’s")):
            continue
        previous = tokens[index - 1]
        if (
            previous[:1].isupper()
            and previous.casefold() not in _FUNCTION_WORDS
        ):
            bases.add(token[:-2].casefold())
    return frozenset(bases)


def _capitalized_identity_surfaces(question: str) -> tuple[str, ...]:
    required = _capitalized_identity_tokens(question)
    return tuple(
        dict.fromkeys(
            token
            for token in _LEXICAL_TOKEN.findall(question)
            if token.casefold() in required
        )
    )


def _quoted_spans(text: str) -> frozenset[str]:
    normalized = text.strip()
    # Some TriviaQA source strings retain CSV-style outer quoting and doubled
    # inner quotation marks, e.g. ``"Which song ... ""lyric"""``.  Decode
    # that transport representation before extracting semantic quoted spans;
    # otherwise the outer wrapper is misclassified as a second immutable
    # title that includes the interrogative itself.
    normalized = _decode_csv_transport_wrapper(normalized)
    spans = [
        next(group for group in match.groups() if group is not None).strip()
        for match in _DOUBLE_QUOTED_SPAN.finditer(normalized)
    ]
    spans.extend(
        next(group for group in match.groups() if group is not None).strip()
        for match in _SINGLE_QUOTED_SPAN.finditer(normalized)
    )
    return frozenset(span for span in spans if span)


def _ordered_quoted_slots(
    text: str,
    *,
    decode_csv_transport: bool = False,
) -> tuple[str, tuple[tuple[int, int, str], ...]]:
    """Locate semantic quote slots without normalizing their content.

    ``decode_csv_transport`` is used only for the authoritative source text,
    whose native TriviaQA representation may retain an outer CSV quote and
    doubled inner delimiters.  Candidate text is never decoded here, so a
    recovery cannot silently rewrite any of its non-slot text.
    """

    located_text = text
    stripped = text.strip()
    if decode_csv_transport:
        located_text = _decode_csv_transport_wrapper(stripped)
    slots: list[tuple[int, int, str]] = []
    for match in _ORDERED_QUOTED_SLOT.finditer(located_text):
        content_group = next(
            index
            for index, value in enumerate(match.groups(), start=1)
            if value is not None
        )
        slots.append(
            (
                match.start(content_group),
                match.end(content_group),
                match.group(content_group),
            )
        )
    return located_text, tuple(slots)


def _quote_slot_identity_tokens(content: str) -> frozenset[str]:
    tokens = frozenset(
        token.casefold().replace("\u2019", "'")
        for token in _LEXICAL_TOKEN.findall(content)
    )
    content_tokens = tokens - _FUNCTION_WORDS
    return content_tokens or tokens


def _quote_slots_have_unambiguous_ordered_identity(
    source_contents: Sequence[str],
    candidate_contents: Sequence[str],
) -> bool:
    """Reject multi-slot repair unless every slot has a unique ordered match."""

    if len(source_contents) != len(candidate_contents) or not source_contents:
        return False
    if len(source_contents) == 1:
        return True

    source_tokens = tuple(
        _quote_slot_identity_tokens(content) for content in source_contents
    )
    candidate_tokens = tuple(
        _quote_slot_identity_tokens(content) for content in candidate_contents
    )

    def score(source_index: int, candidate_index: int) -> tuple[int, int]:
        source = (
            " ".join(source_contents[source_index].split())
            .casefold()
            .replace("\u2019", "'")
        )
        candidate = " ".join(
            candidate_contents[candidate_index].split()
        ).casefold().replace("\u2019", "'")
        return (
            int(source == candidate),
            len(source_tokens[source_index] & candidate_tokens[candidate_index]),
        )

    matrix = tuple(
        tuple(
            score(source_index, candidate_index)
            for candidate_index in range(len(candidate_contents))
        )
        for source_index in range(len(source_contents))
    )
    for position in range(len(source_contents)):
        candidate_column = tuple(
            matrix[source_index][position]
            for source_index in range(len(source_contents))
        )
        best_source_score = max(candidate_column)
        if best_source_score == (0, 0):
            return False
        if candidate_column.count(best_source_score) != 1:
            return False
        if candidate_column.index(best_source_score) != position:
            return False

        source_row = matrix[position]
        best_candidate_score = max(source_row)
        if best_candidate_score == (0, 0):
            return False
        if source_row.count(best_candidate_score) != 1:
            return False
        if source_row.index(best_candidate_score) != position:
            return False
    return True


def _restore_immutable_quoted_slots(
    source_text: str,
    candidate_text: str,
) -> str | None:
    """Restore quote contents only when slot cardinality and order are clear.

    Delimiters remain those emitted by the candidate, so ASCII and Unicode
    curly quotes are interchangeable representations while the authoritative
    source content is copied character-for-character.  No text outside a
    located quote content span is changed.  ``None`` is a fail-closed result.
    """

    _, source_slots = _ordered_quoted_slots(
        source_text,
        decode_csv_transport=True,
    )
    located_candidate, candidate_slots = _ordered_quoted_slots(candidate_text)
    source_contents = tuple(content for _, _, content in source_slots)
    candidate_contents = tuple(content for _, _, content in candidate_slots)
    if not _quote_slots_have_unambiguous_ordered_identity(
        source_contents,
        candidate_contents,
    ):
        return None

    pieces: list[str] = []
    cursor = 0
    for (_, _, source_content), (start, end, _) in zip(
        source_slots,
        candidate_slots,
        strict=True,
    ):
        pieces.append(located_candidate[cursor:start])
        pieces.append(source_content)
        cursor = end
    pieces.append(located_candidate[cursor:])
    return "".join(pieces)


def _lexical_replacement_source_tokens(
    source: TriviaQATrainSource,
) -> frozenset[str]:
    """Return mutable source wording shared by generation and repair.

    Numeric/date strings, quoted content, and leading-dot literals are literal
    dataset constraints, not synonym targets.  Tokenize their full spans with
    the same lexer used for the candidate pool so punctuation inside a date or
    number cannot leave a partially eligible token behind.
    """

    question = source.original_question
    protected_tokens = set(_capitalized_identity_tokens(question))
    protected_tokens.update(_content_token_counts(source.canonical_answer))
    question_tokens = _LEXICAL_TOKEN.findall(question)
    if (
        question_tokens
        and _LEADING_INTERROGATIVE_CONTRACTION.fullmatch(question_tokens[0])
        is not None
    ):
        protected_tokens.add(question_tokens[0].casefold())
    for token_match in _LEXICAL_TOKEN.finditer(question):
        if _NUMBER_OR_DATE_TOKEN.search(token_match.group(0)) is not None:
            protected_tokens.add(token_match.group(0).casefold())
    protected_spans = list(_quoted_spans(question))
    protected_spans.extend(_LEADING_DOT_LITERAL.findall(question))
    for span in protected_spans:
        protected_tokens.update(
            token.casefold() for token in _LEXICAL_TOKEN.findall(span)
        )
    return frozenset(
        set(_content_token_counts(question)) - protected_tokens
    )


def _quoted_scope_preserved(original: str, paraphrase: str) -> bool:
    """Preserve quoted content while allowing quotes around existing titles."""

    original_spans = _quoted_spans(original)
    paraphrase_spans = _quoted_spans(paraphrase)
    if not original_spans.issubset(paraphrase_spans):
        return False
    newly_quoted = paraphrase_spans - original_spans
    return all(span in original for span in newly_quoted)


def _quoted_attribution_qa(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Paraphrase a CSV-quoted ``Who said`` item without editing its quote."""

    spans = tuple(_quoted_spans(source.original_question))
    if len(spans) != 1:
        return None
    normalized = " ".join(source.original_question.split())
    if (
        len(normalized) >= 2
        and normalized.startswith('"')
        and normalized.endswith('"')
        and '""' in normalized
    ):
        normalized = normalized[1:-1].replace('""', '"')
    attribution = re.match(
        r"(?i:who|which\s+(?P<answer_type>[^,?]+?))\s+(?i:said)\s*,",
        normalized,
    )
    if attribution is None:
        return None
    quote = spans[0]
    answer_type = attribution.group("answer_type") or "person"
    return (
        f'Identify the {answer_type} who said, "{quote}"',
        f'{source.canonical_answer} said, "{quote}".',
    )


def _canonical_question_leakage(
    *,
    original_question: str,
    paraphrase_question: str,
    canonical_answer: str,
) -> frozenset[str]:
    original_tokens = set(_content_token_counts(original_question))
    paraphrase_tokens = set(_content_token_counts(paraphrase_question))
    canonical_tokens = set(_content_token_counts(canonical_answer))
    return frozenset((canonical_tokens - original_tokens) & paraphrase_tokens)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _generation_round_indices(
    *,
    generation_round_start: int,
    generation_round_count: int,
) -> range:
    """Return only the fresh deterministic seed rounds for this resume."""

    if generation_round_start < 0 or generation_round_count < 1:
        raise ValueError("generation round start/count are invalid")
    return range(
        generation_round_start,
        generation_round_start + generation_round_count,
    )


def _admitted_generation_seeds(
    *,
    base_seed: int,
    selection_index: int,
    generation_round_start: int,
    generation_round_count: int,
    max_retries: int,
) -> frozenset[int]:
    """Admit historical rounds plus this resume's newly available rounds."""

    if base_seed < 0 or selection_index < 0 or max_retries < 0:
        raise ValueError("generation seed inputs are invalid")
    stop = generation_round_start + generation_round_count
    return frozenset(
        base_seed
        + selection_index
        + generation_round * GENERATION_ROUND_SEED_STRIDE
        + retry
        for generation_round in range(stop)
        for retry in range(max_retries + 1)
    )


def build_paraphrase_messages(source: TriviaQATrainSource) -> list[dict[str, str]]:
    """Build the bounded model request without accepted aliases."""

    source = source_transport_view(source)
    # Only the frozen training projection is model-visible. Held-out validation
    # content and accepted-answer aliases are never included.
    user_payload = {
        "original_question": source.original_question,
        "canonical_training_answer": source.canonical_answer,
        "immutable_original_entity_tokens": list(
            _capitalized_identity_surfaces(source.original_question)
        ),
        "multiword_possessive_identity_bases": sorted(
            _multiword_possessive_identity_bases(source.original_question)
        ),
        "immutable_number_or_date_tokens": sorted(
            set(_NUMBER_OR_DATE_TOKEN.findall(source.original_question))
        ),
        "immutable_quoted_spans": sorted(
            _quoted_spans(source.original_question)
        ),
        "required_answer_slot_count": _answer_slot_count(
            source.original_question
        ),
        "leading_answer_slot_anchor": _leading_answer_slot_anchor(source),
        "original_interrogative_head_omitted": (
            _original_interrogative_head_omitted(source.original_question)
        ),
        "canonical_answer_is_explicit_compound": (
            _canonical_answer_is_explicit_compound(source.canonical_answer)
        ),
        "forbidden_question_canonical_tokens": sorted(
            set(_content_token_counts(source.canonical_answer))
            - set(_content_token_counts(source.original_question))
        ),
        "lexical_replacement_source_tokens": sorted(
            _lexical_replacement_source_tokens(source)
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                user_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def build_verification_messages(
    source: TriviaQATrainSource,
    *,
    paraphrase_question: str,
) -> list[dict[str, str]]:
    """Build a second local-Qwen semantic-equivalence check."""

    source = source_transport_view(source)
    payload = {
        "original_question": source.original_question,
        "paraphrase_question": paraphrase_question,
        "canonical_answer_leakage_checked_deterministically": True,
        "leading_answer_slot_anchor": _leading_answer_slot_anchor(source),
        "original_interrogative_head_omitted": (
            _original_interrogative_head_omitted(source.original_question)
        ),
        "canonical_answer_is_explicit_compound": (
            _canonical_answer_is_explicit_compound(source.canonical_answer)
        ),
    }
    return [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def parse_verification_response(text: str) -> Mapping[str, bool]:
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("verification response is not strict JSON") from exc
    if not isinstance(value, Mapping) or set(value) != _VERIFICATION_FIELDS:
        raise ValueError("verification response fields are incompatible")
    if any(type(value[field]) is not bool for field in _VERIFICATION_FIELDS):
        raise ValueError("verification response fields must be boolean")
    return {field: bool(value[field]) for field in sorted(_VERIFICATION_FIELDS)}


def build_answer_verification_messages(
    source: TriviaQATrainSource,
    *,
    paraphrase_answer_statement: str,
) -> list[dict[str, str]]:
    """Build the local-Qwen answer-slot and relation-direction check."""

    source = source_transport_view(source)
    payload = {
        "original_question": source.original_question,
        "canonical_training_answer": source.canonical_answer,
        "paraphrase_answer_statement": paraphrase_answer_statement,
    }
    return [
        {"role": "system", "content": ANSWER_VERIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def parse_answer_verification_response(text: str) -> Mapping[str, bool]:
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "answer verification response is not strict JSON"
        ) from exc
    if not isinstance(value, Mapping) or set(value) != _ANSWER_VERIFICATION_FIELDS:
        raise ValueError("answer verification response fields are incompatible")
    if any(
        type(value[field]) is not bool for field in _ANSWER_VERIFICATION_FIELDS
    ):
        raise ValueError("answer verification response fields must be boolean")
    return {
        field: bool(value[field])
        for field in sorted(_ANSWER_VERIFICATION_FIELDS)
    }


def build_answer_repair_messages(
    source: TriviaQATrainSource,
    *,
    rejected_answer_statement: str,
) -> list[dict[str, str]]:
    source = source_transport_view(source)
    payload = {
        "original_question": source.original_question,
        "canonical_training_answer": source.canonical_answer,
        "rejected_answer_statement": rejected_answer_statement,
    }
    return [
        {"role": "system", "content": ANSWER_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def _normalize_observed_response_keys(
    value: object,
    *,
    expected_fields: frozenset[str],
    response_name: str,
) -> Mapping[str, object]:
    """Normalize only known structured-output typos without key collisions."""

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{response_name} fields are incompatible; "
            f"observed_type={type(value).__name__}"
        )
    normalized: dict[str, object] = {}
    observed_fields: list[str] = []
    unexpected_fields: list[str] = []
    collision_fields: list[str] = []
    non_text_key_types: list[str] = []
    for raw_field, field_value in value.items():
        if not isinstance(raw_field, str):
            non_text_key_types.append(type(raw_field).__name__)
            continue
        observed_fields.append(raw_field)
        normalized_field = _OBSERVED_RESPONSE_KEY_TYPOS.get(
            raw_field,
            raw_field,
        )
        if normalized_field not in expected_fields:
            unexpected_fields.append(raw_field)
            continue
        if normalized_field in normalized:
            collision_fields.append(normalized_field)
            continue
        normalized[normalized_field] = field_value
    missing_fields = sorted(expected_fields - frozenset(normalized))
    if (
        missing_fields
        or unexpected_fields
        or collision_fields
        or non_text_key_types
    ):
        raise ValueError(
            f"{response_name} fields are incompatible; "
            f"observed_fields={sorted(observed_fields)!r}; "
            f"missing_fields={missing_fields!r}; "
            f"unexpected_fields={sorted(unexpected_fields)!r}; "
            f"collision_fields={sorted(collision_fields)!r}; "
            f"non_text_key_types={sorted(non_text_key_types)!r}"
        )
    return normalized


def parse_answer_repair_response(text: str) -> str:
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("answer repair response is not strict JSON") from exc
    value = _normalize_observed_response_keys(
        value,
        expected_fields=frozenset({"paraphrase_answer_statement"}),
        response_name="answer repair response",
    )
    statement = value["paraphrase_answer_statement"]
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("answer repair statement must be non-empty text")
    return " ".join(statement.split())


def build_question_repair_messages(
    source: TriviaQATrainSource,
    *,
    rejected_question: str,
    required_source_token: str | None = None,
) -> list[dict[str, str]]:
    payload = json.loads(build_paraphrase_messages(source)[1]["content"])
    payload.pop("canonical_training_answer", None)
    payload["rejected_question"] = rejected_question
    eligible = payload.get("lexical_replacement_source_tokens")
    if not isinstance(eligible, list) or not eligible:
        raise ValueError("question repair has no eligible lexical source token")
    eligible_tokens = tuple(str(token) for token in eligible)
    selected = (
        required_source_token
        if required_source_token is not None
        else min(eligible_tokens, key=lambda token: (len(token), token))
    )
    if selected not in eligible_tokens:
        raise ValueError("required question-repair source token is ineligible")
    payload["required_source_token_to_replace"] = selected
    return [
        {"role": "system", "content": QUESTION_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def parse_question_repair_response(
    text: str,
    *,
    eligible_source_tokens: Sequence[str],
    required_source_token: str,
) -> str:
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("question repair response is not strict JSON") from exc
    value = _normalize_observed_response_keys(
        value,
        expected_fields=frozenset(
            {
                "paraphrase_question",
                "replaced_source_token",
                "replacement_phrase",
            }
        ),
        response_name="question repair response",
    )
    question = value["paraphrase_question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question repair must be non-empty text")
    replaced = value["replaced_source_token"]
    replacement = value["replacement_phrase"]
    if not isinstance(replaced, str) or not isinstance(replacement, str):
        raise ValueError("question repair replacement receipt must be text")
    eligible = {token.casefold() for token in eligible_source_tokens}
    if replaced.strip().casefold() not in eligible:
        raise ValueError("question repair replaced_source_token is ineligible")
    if replaced.strip().casefold() != required_source_token.strip().casefold():
        raise ValueError("question repair replaced the wrong source token")
    normalized_question = " ".join(question.split())
    if replacement.strip().casefold() not in normalized_question.casefold():
        raise ValueError("question repair replacement_phrase is absent")
    if replaced.strip().casefold() == replacement.strip().casefold():
        raise ValueError("question repair did not change lexical wording")
    return normalized_question


def build_synonym_repair_messages(
    source: TriviaQATrainSource,
    *,
    required_source_token: str,
) -> list[dict[str, str]]:
    source = source_transport_view(source)
    payload = {
        "original_question": source.original_question,
        "required_source_token": required_source_token,
        "immutable_original_entity_tokens": list(
            _capitalized_identity_surfaces(source.original_question)
        ),
        "forbidden_question_canonical_tokens": sorted(
            set(_content_token_counts(source.canonical_answer))
            - set(_content_token_counts(source.original_question))
        ),
    }
    return [
        {"role": "system", "content": SYNONYM_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def parse_synonym_repair_response(
    text: str,
    *,
    required_source_token: str,
) -> str:
    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("synonym repair response is not strict JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "source_token",
        "replacement_phrase",
    }:
        raise ValueError("synonym repair response fields are incompatible")
    source_token = value["source_token"]
    replacement = value["replacement_phrase"]
    if not isinstance(source_token, str) or not isinstance(replacement, str):
        raise ValueError("synonym repair fields must be text")
    if source_token.strip().casefold() != required_source_token.casefold():
        raise ValueError("synonym repair source token is incompatible")
    replacement = " ".join(replacement.split())
    if not replacement or replacement.casefold() == required_source_token.casefold():
        raise ValueError("synonym repair did not change lexical wording")
    return replacement


def verify_original_dataset_binding(
    *,
    dataset_catalog_path: Path,
    sources: Sequence[TriviaQATrainSource],
    validation_ids: frozenset[str],
    expected_train_count: int,
    expected_validation_count: int,
) -> Mapping[str, object]:
    """Rebind the frozen projection to the original TriviaQA converter."""

    with dataset_catalog_path.resolve().open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("dataset catalog root must be a mapping")
    raw_sources = catalog.get("sources")
    trivia_config = (
        raw_sources.get("triviaqa")
        if isinstance(raw_sources, Mapping)
        else None
    )
    converter = CONVERTERS.get("triviaqa")
    if not isinstance(trivia_config, Mapping) or converter is None:
        raise ValueError("dataset catalog has no TriviaQA converter configuration")
    prefix = tuple(
        islice(
            converter(trivia_config),
            expected_validation_count + expected_train_count,
        )
    )
    if len(prefix) != expected_validation_count + expected_train_count:
        raise ValueError("original TriviaQA candidate prefix is incomplete")
    raw_validation_ids = frozenset(
        str(record.get("task_id", ""))
        for record in prefix[:expected_validation_count]
    )
    if raw_validation_ids != validation_ids:
        raise ValueError(
            "frozen validation identities differ from original TriviaQA order"
        )
    raw_train = prefix[
        expected_validation_count : expected_validation_count
        + expected_train_count
    ]
    if len(raw_train) != len(sources):
        raise ValueError("frozen train count differs from original TriviaQA slice")
    for source, record in zip(sources, raw_train):
        metadata = record.get("metadata")
        payload = (
            metadata.get("evaluator_payload")
            if isinstance(metadata, Mapping)
            else None
        )
        answers = (
            payload.get("accepted_answers")
            if isinstance(payload, Mapping)
            else None
        )
        canonical = answers[0] if isinstance(answers, list) and answers else None
        comparisons = {
            "task_id": (source.base_task_id, record.get("task_id")),
            "question": (source.original_question, record.get("question")),
            "canonical_answer": (source.canonical_answer, canonical),
        }
        for field_name, (frozen, original) in comparisons.items():
            if frozen != original:
                raise ValueError(
                    "frozen TriviaQA train field differs from original "
                    f"dataset at selection_index={source.selection_index}: "
                    f"{field_name}"
                )
    return {
        "dataset": "TriviaQA",
        "configuration": "rc.nocontext",
        "candidate_sequence": list(
            trivia_config.get("candidate_sequence", ())
        ),
        "original_source_path": str(
            resolve_dataset_path(str(trivia_config.get("path", "")))
        ),
        "validation_prefix_count": expected_validation_count,
        "train_slice_start": expected_validation_count,
        "train_slice_stop": expected_validation_count + expected_train_count,
        "train_record_count": expected_train_count,
        "validation_overlap_count": 0,
        "canonical_answer_field": "answer.value/accepted_answers[0]",
    }


def verify_full_native_unique_dataset_binding(
    *,
    dataset_catalog_path: Path,
    sources: Sequence[TriviaQATrainSource],
    validation_ids: frozenset[str],
    expected_train_count: int,
    expected_validation_count: int,
    include_validation_qa: bool,
) -> Mapping[str, object]:
    """Bind a full unique native-train projection to the existing converter.

    This is the scale-only counterpart of ``verify_original_dataset_binding``.
    It reuses the same TriviaQA converter and the full-train de-duplication
    adapter instead of assuming the historical 128+512 prefix layout.
    """

    with dataset_catalog_path.resolve().open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    raw_sources = catalog.get("sources") if isinstance(catalog, Mapping) else None
    trivia_config = (
        raw_sources.get("triviaqa")
        if isinstance(raw_sources, Mapping)
        else None
    )
    if not isinstance(trivia_config, Mapping):
        raise ValueError("dataset catalog has no TriviaQA converter configuration")
    projected, stats = project_unique_nonheldout_train(
        _trivia_records(trivia_config),
        heldout_base_ids=tuple(sorted(validation_ids)),
        include_heldout=include_validation_qa,
    )
    if len(projected) != expected_train_count or len(sources) != expected_train_count:
        raise ValueError("full native TriviaQA projection count is incompatible")
    if len(validation_ids) != expected_validation_count:
        raise ValueError("full native TriviaQA validation identity count is incompatible")
    for source, record in zip(sources, projected):
        metadata = record.get("metadata")
        sampling = metadata.get("sampling") if isinstance(metadata, Mapping) else None
        payload = (
            metadata.get("evaluator_payload")
            if isinstance(metadata, Mapping)
            else None
        )
        answers = payload.get("accepted_answers") if isinstance(payload, Mapping) else None
        canonical = answers[0] if isinstance(answers, list) and answers else None
        comparisons = {
            "base_task_id": (
                source.base_task_id,
                sampling.get("base_task_id") if isinstance(sampling, Mapping) else None,
            ),
            "selection_index": (
                source.selection_index,
                sampling.get("selection_index") if isinstance(sampling, Mapping) else None,
            ),
            "question": (source.original_question, record.get("question")),
            "canonical_answer": (source.canonical_answer, canonical),
        }
        for field_name, (frozen, original) in comparisons.items():
            if frozen != original:
                raise ValueError(
                    "full native TriviaQA field differs from original dataset at "
                    f"selection_index={source.selection_index}: {field_name}"
                )
    overlap_count = len(
        {source.base_task_id for source in sources}.intersection(validation_ids)
    )
    expected_overlap = expected_validation_count if include_validation_qa else 0
    if overlap_count != expected_overlap:
        raise ValueError("full native TriviaQA evaluation overlap is incompatible")
    return {
        "dataset": "TriviaQA",
        "configuration": "rc.nocontext",
        "projection": "full_native_train_unique_question_id",
        "original_source_path": str(
            resolve_dataset_path(str(trivia_config.get("path", "")))
        ),
        "raw_train_row_count": stats["raw_train_row_count"],
        "raw_unique_question_id_count": stats[
            "raw_unique_question_id_count"
        ],
        "duplicate_raw_row_count": stats["duplicate_raw_row_count"],
        "train_record_count": expected_train_count,
        "validation_overlap_count": overlap_count,
        "validation_content_indexed": include_validation_qa,
        "canonical_answer_field": "answer.value/accepted_answers[0]",
    }


def reject_exact_question_identity_shortcut(
    source: TriviaQATrainSource,
    *,
    paraphrase_question: str,
    paraphrase_answer_statement: str,
) -> None:
    """Reject a stored QA record that contains the complete evaluation query."""

    source = source_transport_view(source)
    contaminated_fields = _exact_question_identity_contaminated_fields(
        source,
        paraphrase_question=paraphrase_question,
        paraphrase_answer_statement=paraphrase_answer_statement,
    )
    if contaminated_fields:
        raise FactProjectionAdmissionError(
            "semantic paraphrase retained the complete original question substring"
        )


def _exact_question_identity_contaminated_fields(
    source: TriviaQATrainSource,
    *,
    paraphrase_question: str,
    paraphrase_answer_statement: str,
) -> frozenset[str]:
    """Locate an exact-query shortcut without weakening its rejection gate."""

    original = " ".join(source.original_question.split()).casefold()
    question = " ".join(paraphrase_question.split()).casefold()
    statement = " ".join(paraphrase_answer_statement.split()).casefold()
    contaminated: set[str] = set()
    if original in question:
        contaminated.add("paraphrase_question")
    if original in statement:
        contaminated.add("paraphrase_answer_statement")
    return frozenset(contaminated)


def validate_self_contained_declarative_fact(
    source: TriviaQATrainSource,
    fact_text: object,
) -> str:
    """Admit one fact-only retrieval payload using offline Q-A metadata.

    The original question and canonical answer are used only here, before the
    Agent-facing projection is written.  They never become projected fields or
    embedding text.  This boundary deliberately rejects Q-A wrappers and
    question fragments even when an answer span happens to be present.
    """

    source = source_transport_view(source)
    if not isinstance(fact_text, str) or not fact_text.strip():
        raise FactProjectionAdmissionError("fact_text must be non-empty text")
    fact = " ".join(fact_text.split())
    if _FACT_QA_WRAPPER.search(fact):
        raise FactProjectionAdmissionError(
            "fact_text contains a Question/Answer/prompt/response wrapper"
        )

    # A question mark inside an immutable quoted title or quotation remains
    # part of that fact.  Any question mark in the surrounding clause means the
    # record is still an interrogative or a question-plus-answer concatenation.
    unquoted_surface = _ORDERED_QUOTED_SLOT.sub(" quoted material ", fact)
    if "?" in unquoted_surface or _FACT_INTERROGATIVE_HEAD.match(fact):
        raise FactProjectionAdmissionError(
            "fact_text must be a declarative statement, not a question"
        )
    if _FACT_ANAPHORIC_SUBJECT.match(fact):
        raise FactProjectionAdmissionError(
            "fact_text begins with an unbound anaphoric subject"
        )

    punctuation_surface = fact.rstrip('"\'\u2019\u201d)]} ')
    container_surface = fact.rstrip(")]} ")
    quoted_terminal_punctuation = re.search(
        r"[?!.][\"'\u2019\u201d]\Z",
        container_surface,
    )
    if (
        not punctuation_surface.endswith((".", "!"))
        and quoted_terminal_punctuation is None
    ):
        raise FactProjectionAdmissionError(
            "fact_text must end as a complete declarative sentence"
        )

    canonical = " ".join(source.canonical_answer.split())
    if not exact_canonical_span_preserved(fact, canonical):
        raise FactProjectionAdmissionError(
            "fact_text does not preserve the canonical fact value"
        )
    if not relation_bearing_answer_statement(fact, canonical):
        raise FactProjectionAdmissionError(
            "fact_text is a bare answer without a relation-bearing context"
        )

    canonical_pattern = re.escape(canonical)
    generic_binding = re.compile(
        rf"^(?:(?:the\s+)?{_GENERIC_FACT_SUBJECT}\s+"
        rf"(?:is|are|was|were)\s+(?:the\s+)?{canonical_pattern}|"
        rf"{canonical_pattern}\s+(?:is|are|was|were)\s+"
        rf"(?:the\s+)?{_GENERIC_FACT_SUBJECT})[.!]\Z",
        re.IGNORECASE,
    )
    if generic_binding.fullmatch(fact):
        raise FactProjectionAdmissionError(
            "fact_text binds the answer only to a generic subject"
        )
    if not _answer_statement_has_lexical_relation_lineage(
        original_question=source.original_question,
        canonical_answer=canonical,
        answer_statement=fact,
    ):
        raise FactProjectionAdmissionError(
            "fact_text has no relation lineage from the source question"
        )
    return fact


def _quote_terminal_punctuated_canonical_span(
    fact_text: str,
    canonical_answer: str,
) -> str:
    """Quote one terminal-punctuated canonical value without changing it.

    TriviaQA contains titles whose canonical span itself includes ``?`` or
    ``!``.  Quoting that exact, unique span distinguishes title punctuation
    from an interrogative fact.  Ambiguous repeated spans and already-quoted
    values remain unchanged and continue through the fail-closed fact gate.
    """

    fact = " ".join(fact_text.split()).strip()
    canonical = " ".join(canonical_answer.split()).strip()
    if not fact or not canonical or re.search(r"[?!]", canonical) is None:
        return fact
    if '"' in canonical:
        return fact
    matches = tuple(
        re.finditer(rf"(?<!\w){re.escape(canonical)}(?!\w)", fact)
    )
    if len(matches) != 1:
        return fact
    match = matches[0]
    if (
        match.start() > 0
        and match.end() < len(fact)
        and fact[match.start() - 1] in {'"', "\u201c"}
        and fact[match.end()] in {'"', "\u201d"}
    ):
        return fact
    return fact[: match.start()] + '"' + canonical + '"' + fact[match.end() :]


def _repair_fact_terminal_punctuation(
    fact_text: str,
    *,
    protected_span: str | None = None,
) -> str:
    """Repair only a missing or comma-like terminal sentence delimiter.

    A terminal comma, semicolon, or colon that belongs to the authoritative
    canonical span is retained character-for-character; sentence punctuation
    is appended after it.  Unprotected delimiter repair keeps the historical
    normalization behavior.
    """

    fact = " ".join(fact_text.split()).strip()
    if not fact:
        return fact
    match = re.fullmatch(r"(?P<body>.*?)(?P<closers>[\"'\u2019\u201d)\]}]*)", fact)
    assert match is not None
    body = match.group("body").rstrip()
    closers = match.group("closers")
    if body.endswith((".", "!", "?")):
        return fact
    normalized_protected = (
        " ".join(protected_span.split()).strip()
        if isinstance(protected_span, str)
        else ""
    )
    if (
        normalized_protected
        and normalized_protected.endswith((",", ";", ":"))
        and body.endswith(normalized_protected)
    ):
        return f"{body}{closers}."
    if body.endswith((",", ";", ":")):
        body = body[:-1].rstrip()
    return f"{body}{closers}."


def _dataset_pair_fallback_text(
    source: TriviaQATrainSource,
) -> tuple[str, str]:
    """Render an explicit dataset Q-A association, not a semantic paraphrase."""

    original = " ".join(source.original_question.split())
    canonical = " ".join(source.canonical_answer.split())
    return (
        f"TriviaQA dataset source prompt (verbatim): {original}",
        "For this TriviaQA dataset source prompt, the paired response is "
        f"{canonical}.",
    )


def _is_dataset_pair_fallback_record(record: TriviaQAQAMemoryRecord) -> bool:
    return (
        record.paraphrase_method == DATASET_PAIR_FALLBACK_METHOD
        and record.generator_provider == DATASET_PAIR_FALLBACK_PROVIDER
        and record.model_id == DATASET_PAIR_FALLBACK_MODEL_ID
        and record.model_revision == DATASET_PAIR_FALLBACK_MODEL_REVISION
        and record.prompt_template_version
        == DATASET_PAIR_FALLBACK_PROMPT_VERSION
    )


def _validate_dataset_pair_fallback_record(
    record: TriviaQAQAMemoryRecord,
    source: TriviaQATrainSource,
) -> None:
    """Validate the exact deterministic fallback and its separate provenance."""

    expected_question, expected_statement = _dataset_pair_fallback_text(source)
    if not _is_dataset_pair_fallback_record(record):
        raise ValueError("dataset-pair fallback provenance is incompatible")
    if record.paraphrase_question != expected_question:
        raise ValueError("dataset-pair fallback changed the source prompt")
    if record.paraphrase_answer_statement != expected_statement:
        raise ValueError("dataset-pair fallback changed the paired response")
    if record.canonical_answer != source.canonical_answer:
        raise ValueError("dataset-pair fallback changed the canonical answer")
    if not exact_canonical_span_preserved(
        record.paraphrase_answer_statement,
        source.canonical_answer,
    ) or not relation_bearing_answer_statement(
        record.paraphrase_answer_statement,
        source.canonical_answer,
    ):
        raise ValueError("dataset-pair fallback lost its Q-A association")


def _create_dataset_pair_fallback_record(
    source: TriviaQATrainSource,
    *,
    paraphrase_version: str,
    generation_seed: int,
) -> TriviaQAQAMemoryRecord:
    question, statement = _dataset_pair_fallback_text(source)
    record = TriviaQAQAMemoryRecord.create(
        source=source,
        paraphrase_question=question,
        paraphrase_answer_statement=statement,
        paraphrase_version=paraphrase_version,
        paraphrase_method=DATASET_PAIR_FALLBACK_METHOD,
        generator_provider=DATASET_PAIR_FALLBACK_PROVIDER,
        model_id=DATASET_PAIR_FALLBACK_MODEL_ID,
        model_revision=DATASET_PAIR_FALLBACK_MODEL_REVISION,
        prompt_template_version=DATASET_PAIR_FALLBACK_PROMPT_VERSION,
        generation_seed=generation_seed,
    )
    _validate_dataset_pair_fallback_record(record, source)
    return record


def _decode_paraphrase_object(text: str) -> Mapping[str, object]:
    """Decode the exact Qwen paraphrase object and observed key typos."""

    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("paraphrase response is not strict JSON") from exc
    # Normalize only the three observed structured-output typos.  The shared
    # normalizer rejects unknown keys and any alias/canonical collision before
    # the complete deterministic admission gate runs below.
    return _normalize_observed_response_keys(
        value,
        expected_fields=frozenset(
            {"paraphrase_question", "paraphrase_answer_statement"}
        ),
        response_name="paraphrase response",
    )


def parse_paraphrase_response(
    text: str,
    source: TriviaQATrainSource,
) -> tuple[str, str]:
    """Accept only the declared question-and-statement JSON response."""

    source = source_transport_view(source)
    value = _decode_paraphrase_object(text)
    question = value["paraphrase_question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("paraphrase_question must be non-empty text")
    question = " ".join(question.split())
    if question.casefold() == " ".join(source.original_question.split()).casefold():
        raise FactProjectionAdmissionError(
            "model returned the original question unchanged"
        )
    if not _has_lexical_or_phrase_replacement(
        source.original_question,
        question,
    ):
        raise ValueError(
            "paraphrase_question changed only syntax or word order without "
            "a lexical or phrase replacement"
        )
    if not _participation_marker_preserved(
        source.original_question,
        question,
    ):
        raise ValueError(
            "paraphrase_question broadened a co-participation relation"
        )
    original_identity_tokens = _capitalized_identity_tokens(
        source.original_question
    )
    paraphrase_tokens = frozenset(
        token.casefold()
        for token in _LEXICAL_TOKEN.findall(question)
        if token.casefold() not in _FUNCTION_WORDS
    )
    missing_identity_tokens = frozenset(
        token
        for token in original_identity_tokens
        if not _identity_token_preserved(token, paraphrase_tokens)
        and not (
            token.endswith(("'s", "’s"))
            and token[:-2] in _multiword_possessive_identity_bases(
                source.original_question
            )
            and token[:-2] in paraphrase_tokens
        )
    )
    if missing_identity_tokens:
        raise ValueError(
            "paraphrase_question removed or replaced an original entity token: "
            + ", ".join(sorted(missing_identity_tokens))
        )
    if not _leading_answer_slot_anchor_preserved(
        source,
        question,
    ):
        raise ValueError(
            "paraphrase_question moved the leading answer-slot anchor out of "
            "the interrogative or imperative answer-slot phrase"
        )
    original_numbers = Counter(
        _NUMBER_OR_DATE_TOKEN.findall(source.original_question)
    )
    paraphrase_numbers = Counter(_NUMBER_OR_DATE_TOKEN.findall(question))
    if original_numbers != paraphrase_numbers:
        raise ValueError(
            "paraphrase_question added, removed, or changed a numeric/date constraint"
        )
    if not _quoted_scope_preserved(source.original_question, question):
        raise SemanticPreservationError(
            "paraphrase_question removed, changed, or invented quoted content"
        )
    if _answer_slot_count(question) != _answer_slot_count(
        source.original_question
    ):
        raise ValueError(
            "paraphrase_question changed the requested answer cardinality"
        )
    leaked_tokens = _canonical_question_leakage(
        original_question=source.original_question,
        paraphrase_question=question,
        canonical_answer=source.canonical_answer,
    )
    if leaked_tokens and not canonical_is_original_spelling_variant(
        source.original_question,
        source.canonical_answer,
    ):
        raise ValueError(
            "paraphrase_question introduced answer-bearing canonical tokens: "
            + ", ".join(sorted(leaked_tokens))
        )
    canonical = source.canonical_answer
    if (
        canonical.casefold() not in source.original_question.casefold()
        and canonical.casefold() in question.casefold()
        and not canonical_is_original_spelling_variant(
            source.original_question,
            source.canonical_answer,
        )
    ):
        raise ValueError("paraphrase_question introduced the canonical answer")
    statement = value["paraphrase_answer_statement"]
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("paraphrase_answer_statement must be non-empty text")
    statement = " ".join(statement.split())
    if not exact_canonical_span_preserved(statement, canonical):
        canonicalized_statement = (
            _canonicalize_answer_statement_from_accepted_alias(
                source,
                statement,
            )
        )
        if canonicalized_statement is not None:
            statement = canonicalized_statement
    possessive_name_binding = _possessive_name_answer_statement(source)
    if (
        possessive_name_binding is not None
        and re.search(
            rf"\bof\s+{re.escape(possessive_name_binding[0])}\s+"
            r"(?i:is|was|are|were)\b",
            statement,
        )
        is not None
    ):
        raise ValueError(
            "paraphrase_answer_statement contains a dangling possessive "
            "in the name relation"
        )
    if not exact_canonical_span_preserved(statement, canonical):
        raise ValueError(
            "paraphrase_answer_statement does not preserve the exact canonical span"
        )
    if not relation_bearing_answer_statement(statement, canonical):
        raise ValueError(
            "paraphrase_answer_statement must be declarative and express the "
            "question relation beyond the canonical answer span"
        )
    if not _answer_statement_has_lexical_relation_lineage(
        original_question=source.original_question,
        canonical_answer=canonical,
        answer_statement=statement,
    ):
        raise ValueError(
            "paraphrase_answer_statement lost lexical relation lineage from "
            "the original question"
        )
    if (
        _subject_wh_anchor_requires_literal_binding(source)
        and not _literal_slot_substitution_preserved(
            original_question=source.original_question,
            canonical_answer=canonical,
            answer_statement=statement,
        )
    ):
        raise ValueError(
            "paraphrase_answer_statement does not preserve deterministic "
            "leading answer-slot binding"
        )
    if not _listed_choice_answer_binding_preserved(
        original_question=source.original_question,
        canonical_answer=canonical,
        answer_statement=statement,
    ):
        raise ValueError(
            "paraphrase_answer_statement does not bind the exact canonical "
            "label to the selected listed option"
        )
    statement = validate_self_contained_declarative_fact(source, statement)
    # Run the new semantic boundary last.  Resume partitioning is therefore
    # allowed to repair only an otherwise valid row; every older deterministic
    # admission failure remains fail-closed.
    _semantic_relation_and_scope_preserved(
        original_question=source.original_question,
        paraphrase_question=question,
        paraphrase_answer_statement=statement,
        canonical_answer=canonical,
    )
    return question, statement


def load_resume_records(
    path: Path,
) -> tuple[tuple[TriviaQAQAMemoryRecord, ...], tuple[str, ...]]:
    """Load a checkpoint while dropping only v4 answer-only records.

    This preserves every already-valid local generation.  Any unrelated row
    corruption still fails closed instead of being silently regenerated.
    """

    if not path.is_file():
        return (), ()
    records: list[TriviaQAQAMemoryRecord] = []
    rejected_source_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"existing paraphrase JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"existing paraphrase row {line_number} is not an object"
                )
            if value.get("prompt_template_version") in {
                "triviaqa.qa_memory.qa_paraphrase.v4",
                "triviaqa.qa_memory.qa_paraphrase.v5",
                PROMPT_TEMPLATE_VERSION,
            }:
                statement = value.get("paraphrase_answer_statement")
                canonical = value.get("canonical_answer")
                if not relation_bearing_answer_statement(statement, canonical):
                    source_id = value.get("source_train_task_id")
                    if not isinstance(source_id, str) or not source_id.strip():
                        raise ValueError(
                            "answer-only checkpoint row has no source_train_task_id"
                        )
                    rejected_source_ids.append(source_id)
                    continue
            records.append(TriviaQAQAMemoryRecord.from_value(value))
    if len(set(rejected_source_ids)) != len(rejected_source_ids):
        raise ValueError("answer-only checkpoint source IDs are not unique")
    return tuple(records), tuple(rejected_source_ids)


def validate_resume_record_admission(
    records: Sequence[TriviaQAQAMemoryRecord],
    sources: Sequence[TriviaQATrainSource],
) -> None:
    """Re-run the strict semantic/fact boundary on every checkpoint row."""

    source_by_id = {
        source.source_train_task_id: source for source in sources
    }
    for record in records:
        source = source_by_id.get(record.source_train_task_id)
        if source is None:
            raise ValueError("resume row references a non-train source")
        if _is_dataset_pair_fallback_record(record):
            raise FactProjectionAdmissionError(
                "dataset-pair fallback is not a semantic paraphrase or fact"
            )
        parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": record.paraphrase_question,
                    "paraphrase_answer_statement": (
                        record.paraphrase_answer_statement
                    ),
                },
                ensure_ascii=False,
            ),
            source,
        )


def partition_resume_records_for_semantic_repair(
    records: Sequence[TriviaQAQAMemoryRecord],
    sources: Sequence[TriviaQATrainSource],
) -> tuple[tuple[TriviaQAQAMemoryRecord, ...], tuple[str, ...]]:
    """Partition semantic drift and legacy non-facts into regeneration.

    Catching only the two dedicated semantic/fact exceptions, rather than
    ``ValueError``, is the explicit repair boundary: malformed JSON/schema,
    source mismatches, answer leakage, provenance errors, and all unrelated
    admission failures abort resume instead of becoming silent model work.
    """

    source_by_id = {
        source.source_train_task_id: source for source in sources
    }
    accepted: list[TriviaQAQAMemoryRecord] = []
    repair_source_ids: list[str] = []
    for record in records:
        source = source_by_id.get(record.source_train_task_id)
        if source is None:
            raise ValueError("resume row references a non-train source")
        if _is_dataset_pair_fallback_record(record):
            _validate_dataset_pair_fallback_record(record, source)
            repair_source_ids.append(record.source_train_task_id)
            continue
        try:
            # Run the fact-only admission boundary first.  An older strict
            # Q-A row can otherwise fail an earlier generic parser check
            # (wrapper, question fragment, missing canonical span) and abort
            # the entire resume before it is classified as a fact repair.
            validate_self_contained_declarative_fact(
                source,
                record.paraphrase_answer_statement,
            )
            question, statement = parse_paraphrase_response(
                json.dumps(
                    {
                        "paraphrase_question": record.paraphrase_question,
                        "paraphrase_answer_statement": (
                            record.paraphrase_answer_statement
                        ),
                    },
                    ensure_ascii=False,
                ),
                source,
            )
            reject_exact_question_identity_shortcut(
                source,
                paraphrase_question=question,
                paraphrase_answer_statement=statement,
            )
        except (SemanticPreservationError, FactProjectionAdmissionError):
            repair_source_ids.append(record.source_train_task_id)
            continue
        accepted.append(record)
    if len(set(repair_source_ids)) != len(repair_source_ids):
        raise ValueError("semantic-repair source IDs are not unique")
    return tuple(accepted), tuple(repair_source_ids)


def order_pending_sources_for_resume(
    sources: Sequence[TriviaQATrainSource],
    records: Sequence[TriviaQAQAMemoryRecord],
) -> tuple[TriviaQATrainSource, ...]:
    """Run never-attempted tail rows before deterministic resume gaps.

    A missing row at or below the highest admitted ``selection_index`` has
    already crossed the bounded generation/admission boundary in an earlier
    resumable pass.  Keeping those hard gaps behind the untouched source tail
    prevents a restart from replaying the same deterministic seeds before it
    reaches new work.  Every gap remains in the returned sequence and is still
    processed in frozen source order after the tail.
    """

    admitted_ids = {
        record.source_train_task_id for record in records
    }
    frontier = max(
        (record.selection_index for record in records),
        default=-1,
    )
    untouched_tail: list[TriviaQATrainSource] = []
    resume_gaps: list[TriviaQATrainSource] = []
    for source in sources:
        if source.source_train_task_id in admitted_ids:
            continue
        target = (
            untouched_tail
            if source.selection_index > frontier
            else resume_gaps
        )
        target.append(source)
    return tuple((*untouched_tail, *resume_gaps))


def build_fact_memory_projection(
    records: Sequence[TriviaQAQAMemoryRecord],
    sources: Sequence[TriviaQATrainSource],
    *,
    expected_count: int | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Split strict materialization into fact corpus and external provenance.

    The first tuple is the complete Agent-facing/indexable allowlist.  The
    second tuple is an offline provenance sidecar and is never an input to the
    embedding model or retrieval Tool.
    """

    if expected_count is not None and (
        type(expected_count) is not int or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")
    validate_qa_memory_against_sources(records, sources, require_complete=True)
    record_by_id = {
        record.source_train_task_id: record for record in records
    }
    if len(record_by_id) != len(records):
        raise ValueError("fact projection source identities are not unique")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"fact projection count is {len(records)}, expected {expected_count}"
        )

    fact_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda item: item.selection_index):
        record = record_by_id[source.source_train_task_id]
        if _is_dataset_pair_fallback_record(record):
            raise FactProjectionAdmissionError(
                "dataset-pair fallback cannot enter a fact-memory release"
            )
        question, statement = parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": record.paraphrase_question,
                    "paraphrase_answer_statement": (
                        record.paraphrase_answer_statement
                    ),
                },
                ensure_ascii=False,
            ),
            source,
        )
        reject_exact_question_identity_shortcut(
            source,
            paraphrase_question=question,
            paraphrase_answer_statement=statement,
        )
        if not _has_lexical_or_phrase_replacement(
            source.original_question,
            question,
        ):
            raise FactProjectionAdmissionError(
                "fact release question has no lexical or phrase replacement"
            )
        fact_text = validate_self_contained_declarative_fact(source, statement)
        fact_rows.append(
            {
                "schema_version": FACT_MEMORY_SCHEMA_VERSION,
                "memory_id": record.memory_id,
                "tool_id": FACT_MEMORY_TOOL_ID,
                "fact_text": fact_text,
            }
        )
        accepted_answers = (
            source.accepted_answers_for_admission
            or (source.canonical_answer,)
        )
        provenance_rows.append(
            {
                "schema_version": FACT_PROVENANCE_SCHEMA_VERSION,
                "memory_id": record.memory_id,
                "source_train_task_id": source.source_train_task_id,
                "base_task_id": source.base_task_id,
                "selection_index": source.selection_index,
                "original_question": source.original_question,
                "accepted_answers": list(accepted_answers),
                "canonical_answer": source.canonical_answer,
                "paraphrase_question": question,
                "fact_text": fact_text,
                "paraphrase_version": record.paraphrase_version,
                "paraphrase_method": record.paraphrase_method,
                "generator_provider": record.generator_provider,
                "model_id": record.model_id,
                "model_revision": record.model_revision,
                "prompt_template_version": record.prompt_template_version,
                "generation_seed": record.generation_seed,
            }
        )

    memory_ids = [str(row["memory_id"]) for row in fact_rows]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("fact projection memory_id values are not unique")
    if len(fact_rows) != len(sources) or len(provenance_rows) != len(sources):
        raise ValueError("fact/provenance projection coverage is incomplete")
    return tuple(fact_rows), tuple(provenance_rows)


def write_jsonl_projection(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Atomically write a deterministic projection without partial release."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(target)


_DETERMINISTIC_SUBJECT_RELATIVE_VERB = re.compile(
    r"(?:appears?|became|becomes?|begins?|belongs?|co[- ]?founded|"
    r"contains?|created|creates?|features?|has|have|had|includes?|"
    r"inspired|inspires?|played|plays?|recorded|records?|reads?|"
    r"said|sang|stars?|starred|supplies?|translates?|won|wrote)",
    re.IGNORECASE,
)

def _finish_deterministic_question_candidate(
    source: TriviaQATrainSource,
    candidate: str,
) -> str | None:
    """Run question-level admission checks before returning a fallback.

    Every caller immediately combines the returned question with its current
    answer statement and invokes :func:`parse_paraphrase_response`; this
    preflight is intentionally not a parallel materialization boundary.  It
    only avoids returning a deterministic sentence transformation that is
    already known to violate an existing question-side invariant.
    """

    original = " ".join(source.original_question.split())
    candidate = " ".join(candidate.split())
    if not candidate or original.casefold() in candidate.casefold():
        return None
    if not _has_lexical_or_phrase_replacement(original, candidate):
        return None
    if not _who_answer_slot_family_preserved(original, candidate):
        return None
    if not _leading_answer_slot_anchor_preserved(source, candidate):
        return None
    if not _participation_marker_preserved(original, candidate):
        return None
    if Counter(_NUMBER_OR_DATE_TOKEN.findall(original)) != Counter(
        _NUMBER_OR_DATE_TOKEN.findall(candidate)
    ):
        return None
    if not _quoted_scope_preserved(original, candidate):
        return None
    if _answer_slot_count(original) != _answer_slot_count(candidate):
        return None
    original_identity_tokens = _capitalized_identity_tokens(original)
    candidate_tokens = frozenset(
        token.casefold()
        for token in _LEXICAL_TOKEN.findall(candidate)
        if token.casefold() not in _FUNCTION_WORDS
    )
    if any(
        not _identity_token_preserved(token, candidate_tokens)
        for token in original_identity_tokens
    ):
        return None
    if _canonical_question_leakage(
        original_question=original,
        paraphrase_question=candidate,
        canonical_answer=source.canonical_answer,
    ) and not canonical_is_original_spelling_variant(
        original,
        source.canonical_answer,
    ):
        return None
    if _contains_source_spelling_correction(original, candidate):
        return None
    return candidate


def _deterministic_question_paraphrase(
    source: TriviaQATrainSource,
) -> str | None:
    """Apply conservative semantic-preserving sentence transformations.

    This is a final repair fallback after bounded model and synonym repair.
    It transforms the authoritative leading answer-slot construction itself;
    it never wraps or embeds the complete original question.  Patterns that
    would require uncertain dependency parsing (for example object-WH
    inversion after ``did``) fail closed.
    """

    original = " ".join(source.original_question.split())
    body = original[:-1].rstrip() if original.endswith(("?", ".")) else original
    candidate: str | None = None

    def declarative(text: str) -> str:
        stripped = text.rstrip()
        if re.search(r"[.!?][\"'”’]?\Z", stripped):
            return stripped
        return stripped + "."

    who = re.fullmatch(r"(?i:who)\s+(?P<tail>.+)", body)
    if (
        who is not None
        and _answer_slot_count(original) == 1
        and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1
    ):
        tail = who.group("tail")
        first = _LEXICAL_TOKEN.search(tail)
        # A leading do-support auxiliary usually marks an object-WH question;
        # converting it safely requires dependency parsing and inflection.
        if first is not None and first.group(0).casefold() not in {
            "did",
            "do",
            "does",
        }:
            candidate = declarative(f"Identify the person who {tail}")

    if candidate is None:
        where = re.fullmatch(r"(?i:where)\s+(?P<tail>.+)", body)
        if (
            where is not None
            and _answer_slot_count(original) == 1
            and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1
        ):
            candidate = f"At what location {where.group('tail')}?"

    if candidate is None:
        when = re.fullmatch(r"(?i:when)\s+(?P<tail>.+)", body)
        if (
            when is not None
            and _answer_slot_count(original) == 1
            and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1
        ):
            first_tail_token = _LEXICAL_TOKEN.search(when.group("tail"))
            safe_auxiliary = (
                first_tail_token is not None
                and first_tail_token.group(0).casefold()
                in {
                    "are",
                    "can",
                    "did",
                    "do",
                    "does",
                    "had",
                    "has",
                    "have",
                    "is",
                    "was",
                    "were",
                    "will",
                    "would",
                }
            )
        else:
            safe_auxiliary = False
        if when is not None and safe_auxiliary:
            candidate = f"At what time {when.group('tail')}?"

    if candidate is None:
        how_many = re.fullmatch(r"(?i:how\s+many)\s+(?P<tail>.+)", body)
        if (
            how_many is not None
            and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1
        ):
            tail = how_many.group("tail")
            existential = re.fullmatch(
                r"(?P<head>.+?)\s+(?P<aux>are|were|is|was)\s+there"
                r"(?P<rest>.*)",
                tail,
                re.IGNORECASE,
            )
            perfect_existential = re.fullmatch(
                r"(?P<head>.+?)\s+(?P<aux>have|has|had)\s+there\s+been"
                r"(?P<rest>.*)",
                tail,
                re.IGNORECASE,
            )
            predicative = re.fullmatch(
                r"(?P<head>.+?)\s+(?P<aux>are|were|is|was)\s+"
                r"(?P<rest>.+)",
                tail,
                re.IGNORECASE,
            )
            if perfect_existential is not None:
                candidate = declarative(
                    "State the number of "
                    f"{perfect_existential.group('head')} there "
                    f"{perfect_existential.group('aux')} been"
                    f"{perfect_existential.group('rest')}"
                )
            elif existential is not None:
                candidate = declarative(
                    "State the number of "
                    f"{existential.group('head')} there "
                    f"{existential.group('aux')}"
                    f"{existential.group('rest')}"
                )
            elif predicative is not None:
                candidate = declarative(
                    "State the number of "
                    f"{predicative.group('head')} that "
                    f"{predicative.group('aux')} "
                    f"{predicative.group('rest')}"
                )
            elif re.search(
                r"\b(?:are|were|is|was|do|does|did|have|has|had)\b",
                tail,
                re.IGNORECASE,
            ) is None:
                connector = "" if tail.casefold().startswith("of ") else "of "
                candidate = declarative(f"State the number {connector}{tail}")

    if candidate is None:
        imperative = re.fullmatch(
            r"(?P<head>(?i:name))\s+(?P<tail>.+)",
            body,
        )
        if imperative is not None:
            candidate = declarative(f"Identify {imperative.group('tail')}")

    if candidate is None:
        complete = re.fullmatch(r"(?i:complete)\s+(?P<tail>.+)", body)
        if complete is not None:
            candidate = declarative(
                "Supply the missing completion for "
                f"{complete.group('tail')}"
            )

    if candidate is None:
        finish = re.fullmatch(r"(?i:finish)\s+(?P<tail>.+)", body)
        if finish is not None:
            candidate = declarative(
                f"Supply the ending of {finish.group('tail')}"
            )

    if candidate is None:
        what_copular = re.fullmatch(
            r"(?i:what)\s+(?P<copula>is|was)\s+"
            r"(?P<complement>the\s+.+)",
            body,
        )
        if (
            what_copular is not None
            and _answer_slot_count(original) == 1
            and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1
        ):
            candidate = declarative(
                f"Identify {what_copular.group('complement')}"
            )

    if candidate is None:
        subject_wh = re.fullmatch(
            r"(?P<wh>(?i:what|which))\s+"
            r"(?P<head>.+?)\s+"
            rf"(?P<verb>{_DETERMINISTIC_SUBJECT_RELATIVE_VERB.pattern})\s+"
            r"(?P<tail>.+)",
            body,
            re.IGNORECASE,
        )
        if (
            subject_wh is not None
            and _answer_slot_count(original) == 1
            and len(_QUESTION_SLOT_TOKEN.findall(original)) == 1
        ):
            candidate = declarative(
                f"Identify the {subject_wh.group('head')} that "
                f"{subject_wh.group('verb')} {subject_wh.group('tail')}"
            )

    if (
        candidate is None
        and _primary_answer_slot(original) is None
        and not _quoted_spans(original)
    ):
        # Native TriviaQA contains a small number of answer-slot-free clue
        # fragments.  Give only those fragments a neutral denotation frame;
        # no entity token or clue content is paraphrased or inferred.
        if original.endswith(("?", ".")):
            clue = original[:-1].rstrip()
            if len(_LEXICAL_TOKEN.findall(clue)) >= 2:
                candidate = f'State what the clue "{clue}" denotes.'
        else:
            address = re.fullmatch(
                r"(?P<street>\d+[^,]+),\s*(?P<locality>[^,]+),\s*"
                r"(?P<region>[^,]+)",
                original,
            )
            if address is not None:
                candidate = (
                    "State the entity associated with "
                    f"{address.group('region')}, {address.group('locality')}, "
                    f"at {address.group('street')}."
                )

    if candidate is None:
        return None
    return _finish_deterministic_question_candidate(source, candidate)


_BOUNDED_SUBJECT_WH_HEADS = frozenset(
    {
        "dictator",
        "footballer",
        "law",
        "playwright",
        "port",
        "procedure",
        "revolutionary",
        "series",
    }
)
_BOUNDED_SUBJECT_WH_PREDICATE = re.compile(
    r"(?:is|was|are|were|returned|returns|says?|runs?|wrote|writes|"
    r"co[- ]?starred)\b",
    re.IGNORECASE,
)


def _bounded_subject_wh_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind one leading typed subject-WH to its copied finite predicate."""

    original = " ".join(source.original_question.split())
    body = original[:-1].rstrip() if original.endswith(("?", ".")) else original
    if (
        not body
        or "?" in body
        or _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
    ):
        return None
    match = re.fullmatch(
        r"(?i:what|which)\s+(?P<head>.+?)\s+"
        rf"(?P<predicate>{_BOUNDED_SUBJECT_WH_PREDICATE.pattern}.+)",
        body,
        re.IGNORECASE,
    )
    if match is None:
        return None
    head = match.group("head")
    head_tokens = tuple(_LEXICAL_TOKEN.findall(head))
    if (
        not head_tokens
        or len(head_tokens) > 12
        or any(token.endswith(("'s", "’s")) for token in head_tokens)
        or not (
            _simple_subject_wh_head(head)
            or head_tokens[-1].casefold() in _BOUNDED_SUBJECT_WH_HEADS
        )
    ):
        return None
    predicate = match.group("predicate")
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(f"Identify the {head} that {predicate}"),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{source.canonical_answer} {predicate}"
    )


def _leading_copular_object_wh_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind one leading object-WH to its complete copied complement."""

    original = " ".join(source.original_question.split())
    body = original[:-1].rstrip() if original.endswith(("?", ".")) else original
    if (
        not body
        or "?" in body
        or _answer_slot_count(original) != 1
    ):
        return None
    match = re.fullmatch(
        r"(?i:what|which)\s+(?P<copula>is|was|are|were)\s+"
        r"(?P<complement>.+)",
        body,
    )
    if match is None:
        return None
    complement = match.group("complement")
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(f"Identify {complement}"),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{complement} {match.group('copula')} {source.canonical_answer}"
    )


def _network_identifier_contrast_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind the target country in the complete UK identifier contrast."""

    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    match = re.fullmatch(
        r"(?P<reference>'\.uk'|\.uk(?: \(dot uk\))?)\s+"
        r"is the network identifier for the United Kingdom[.,]\s+"
        r"which country uses the identifier\s+"
        r"(?P<target>'\.(?P<quoted_label>[a-z]{2})'|"
        r"\.(?P<bare_label>[a-z]{2})"
        r"(?: \(dot (?P<dot_label>[a-z]{2})\))?)\?",
        original,
        re.IGNORECASE,
    )
    if match is None:
        return None
    dot_label = match.group("dot_label")
    bare_label = match.group("bare_label")
    if (
        dot_label is not None
        and bare_label is not None
        and dot_label.casefold() != bare_label.casefold()
    ):
        return None
    reference = match.group("reference")
    target = match.group("target")
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(
            "The network identifier for the United Kingdom is "
            f"{reference}. Identify the country that uses the identifier "
            f"{target}"
        ),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        "The network identifier for the United Kingdom is "
        f"{reference}, and {source.canonical_answer} uses the identifier "
        f"{target}"
    )


_FRONTED_CONTEXT_PERSON_PREDICATE = re.compile(
    r"(?:became\s+the\s+first\b.*\bgolfer\b|"
    r"(?:is|was)\s+the\s+(?:mother|uncle)\b)",
    re.IGNORECASE,
)


def _fronted_context_subject_wh_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind one singular subject-WH while retaining its fronted context."""

    original = " ".join(source.original_question.split())
    body = original[:-1].rstrip() if original.endswith(("?", ".")) else original
    if (
        not body
        or "?" in body
        or _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _canonical_answer_is_explicit_compound(source.canonical_answer)
    ):
        return None
    match = re.fullmatch(
        r"(?P<context>In .+?)(?:,\s+|\s+)who\s+(?P<predicate>.+)",
        body,
        re.IGNORECASE,
    )
    if match is None:
        return None
    context = match.group("context")
    predicate = match.group("predicate")
    if (
        _QUESTION_SLOT_TOKEN.search(context) is not None
        or re.match(r"(?:are|were)\b", predicate, re.IGNORECASE) is not None
        or _FRONTED_CONTEXT_PERSON_PREDICATE.match(predicate) is None
    ):
        return None
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(
            f"{context}, identify the person who {predicate}"
        ),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{context}, {source.canonical_answer} {predicate}"
    )



def _latin_translation_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted Latin-translation relation and surface change."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    match = re.fullmatch(
        r"What does the Latin phrase (?P<phrase>'[^']+'|‘[^’]+’) "
        r"translate to in English\?", original, re.IGNORECASE
    )
    if match is None:
        return None
    phrase = match.group("phrase")
    question = _finish_deterministic_question_candidate(
        source, f"What is the English translation of the Latin phrase {phrase}?"
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The English translation of the Latin phrase {phrase} is {source.canonical_answer}"
    )


def _circle_line_name_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted ``joins`` to ``connects`` circle relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    if re.fullmatch(
        r"What name is given to a straight line that joins any two points "
        r"on the circumference of a circle\?", original, re.IGNORECASE
    ):
        question = _finish_deterministic_question_candidate(
            source,
            "What name is given to a straight line that connects any two "
            "points on the circumference of a circle?",
        )
        if question is None:
            return None
        return question, _declarative_statement(
            "The name given to a straight line that joins any two points on "
            f"the circumference of a circle is {source.canonical_answer}"
        )
    if not re.fullmatch(
        r"What is a line that joins two points of a circle\?", original, re.IGNORECASE
    ):
        return None
    question = _finish_deterministic_question_candidate(
        source, "What is a line that connects two points of a circle?"
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The line that connects two points of a circle is {source.canonical_answer}"
    )


def _english_theatre_location_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted ``find`` to ``locate`` theatre relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    match = re.fullmatch(
        r"In which English town or city would you find "
        r"(?P<theatre>the [^?]{1,80} theatre)\?", original, re.IGNORECASE
    )
    if match is None:
        return None
    theatre = match.group("theatre")
    question = _finish_deterministic_question_candidate(
        source, f"In which English town or city could you locate {theatre}?"
    )
    if question is None:
        return None
    return question, _declarative_statement(f"{theatre} is located in {source.canonical_answer}")


def _alcock_brown_year_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted Alcock-and-Brown year/crossing relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    match = re.fullmatch(
        r"In which year did Alcock and Brown make (?P<event>their Atlantic crossing|"
        r"the first non-stop trans- Atlantic flight)\?", original, re.IGNORECASE
    )
    if match is None:
        return None
    event = match.group("event")
    question = _finish_deterministic_question_candidate(
        source, f"During what year did Alcock and Brown complete {event}?"
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The year Alcock and Brown completed {event} is {source.canonical_answer}"
    )


def _tells_lies_description_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted ``lies`` to ``falsehoods`` description relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    if re.fullmatch(r"Someone who tells lies is what sort of person\?", original, re.IGNORECASE):
        question = _finish_deterministic_question_candidate(
            source, "Someone who tells falsehoods is what sort of person?"
        )
        if question is None:
            return None
        return question, _declarative_statement(
            f"Someone who tells lies is a {source.canonical_answer} sort of person"
        )
    if not re.fullmatch(r"Someone who tells lies is said to be what\?", original, re.IGNORECASE):
        return None
    question = _finish_deterministic_question_candidate(
        source, "Someone who tells falsehoods is said to be what?"
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"Someone who tells falsehoods is said to be {source.canonical_answer}"
    )


def _eating_relation_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted ``eating`` to ``consuming`` food relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    match = re.fullmatch(
        r"If you are eating (?P<food>salted, unfertilized sturgeon roe), "
        r"what are you eating\?", original, re.IGNORECASE
    )
    if match is not None:
        food = match.group("food")
        question = _finish_deterministic_question_candidate(
            source, f"If you are consuming {food}, what are you consuming?"
        )
        if question is None:
            return None
        return question, _declarative_statement(
            f"If you are eating {food}, you are eating {source.canonical_answer}"
        )
    if not re.fullmatch(r"If you are eating tripe, what are you eating\?", original, re.IGNORECASE):
        return None
    question = _finish_deterministic_question_candidate(
        source, "If you are consuming tripe, identify the organ you are consuming?"
    )
    if question is None:
        return None
    return question, _declarative_statement(f"The organ you are consuming is {source.canonical_answer}")


def _darts_shanghai_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted ``same`` to ``identical`` darts relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    if re.fullmatch(
        r"What name is given to scoring a single, double and treble of the same "
        r"number in three darts\?", original, re.IGNORECASE
    ):
        question = _finish_deterministic_question_candidate(
            source,
            "What name is given to scoring a single, double and treble of the "
            "identical number in three darts?",
        )
        if question is None:
            return None
        return question, _declarative_statement(
            "The name given to scoring a single, double and treble of the same "
            f"number in three darts is {source.canonical_answer}"
        )
    if re.fullmatch(
        r"What name is given in darts when a player hits a single, double and "
        r"treble of the same number in his turn\?", original, re.IGNORECASE
    ):
        question = _finish_deterministic_question_candidate(
            source,
            "What name is given in darts when a player hits a single, double and "
            "treble of the identical number in his turn?",
        )
        if question is None:
            return None
        return question, _declarative_statement(
            "The name given in darts when a player hits a single, double and treble "
            f"of the same number in his turn is {source.canonical_answer}"
        )
    if not re.fullmatch(
        r"In darts, a three dart finish requiring a treble, single and double of "
        r"the same number is given what name\?", original, re.IGNORECASE
    ):
        return None
    question = _finish_deterministic_question_candidate(
        source,
        "In darts, a three dart finish requiring a treble, single and double of "
        "the identical number is given what name?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        "In darts, the name given to a three dart finish requiring a treble, single "
        f"and double of the same number is {source.canonical_answer}"
    )


def _acted_role_analogue_pair(source: TriviaQATrainSource) -> tuple[str, str] | None:
    """Reuse the admitted ``part`` to ``role`` acting relation."""
    original = " ".join(source.original_question.split())
    if _answer_slot_count(original) != 1:
        return None
    match = re.fullmatch(
        r"What part did (?P<actor>[^?]{1,80}?) play in "
        r"(?P<work>'[^']+'|‘[^’]+’)(?P<year> \([0-9]{4}\))?\?",
        original, re.IGNORECASE,
    )
    if match is None:
        return None
    actor, work, year = match.group("actor"), match.group("work"), match.group("year") or ""
    question = _finish_deterministic_question_candidate(
        source, f"Which role did {actor} play in {work}{year}?"
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{actor} played the role of {source.canonical_answer} in {work}{year}"
    )


def _first_word_relation_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind one singular ``first word of`` relation to its value."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or len(_LEXICAL_TOKEN.findall(source.canonical_answer)) != 1
    ):
        return None
    match = re.fullmatch(
        r"(?i:what(?:['’]s|\s+is))\s+the\s+first\s+word\s+of\s+"
        r"(?P<referent>[^?]{1,240})\?",
        original,
    )
    if match is None:
        return None
    referent = match.group("referent")
    question = _finish_deterministic_question_candidate(
        source,
        f"What is the initial word of {referent}?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The first word of {referent} is {source.canonical_answer}"
    )


def _name_given_relation_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind the exact leading ``What name is given to`` construction."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
    ):
        return None
    match = re.fullmatch(
        r"(?i:what name is given to)\s+(?P<referent>[^?]{1,500})\?",
        original,
    )
    if match is None:
        return None
    referent = match.group("referent")
    question = _finish_deterministic_question_candidate(
        source,
        f"What designation is given to {referent}?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The name given to {referent} is {source.canonical_answer}"
    )


_SIMPLE_NP_COPULAR_CLAUSE = re.compile(
    r"\b(?:and|or|if|that|when|where|which|who|whom|whose)\b",
    re.IGNORECASE,
)


def _simple_np_terminal_copular_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind terminal ``The <simple NP> <copula> what`` without a clause."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
    ):
        return None
    match = re.fullmatch(
        r"(?P<subject>The [^,;:?!]{1,240})\s+"
        r"(?P<copula>is|was|are|were)\s+what\?",
        original,
        re.IGNORECASE,
    )
    if match is None:
        return None
    subject = match.group("subject")
    if (
        not subject.startswith("The ")
        or _SIMPLE_NP_COPULAR_CLAUSE.search(subject) is not None
    ):
        return None
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(f"Identify the {subject[4:]}"),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{subject} {match.group('copula')} {source.canonical_answer}"
    )


def _object_wh_represent_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind one object-WH under singular do-support ``represent``."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = re.fullmatch(
        r"(?i:what does)\s+(?P<subject>[^?]{1,240}?)\s+represent"
        r"(?P<context>(?:\s+(?:as|at|for|in|on|within)\s+[^?]+)?)\?",
        original,
    )
    if match is None:
        return None
    subject = match.group("subject")
    context = match.group("context")
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(
            f"Identify what {subject} represents{context}"
        ),
    )
    if question is None:
        return None
    # Preserve lower-case scientific symbols (for example ``c``) exactly;
    # ``_declarative_statement`` sentence-cases only an initial article.
    return question, _declarative_statement(
        f"{subject} represents {source.canonical_answer}{context}"
    )


def _fronted_domain_passive_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind the domain-code country slot in one exact fronted passive."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = re.fullmatch(
        r"(?P<context>In internet domain names)\s+what\s+"
        r"(?P<head>country|nation)\s+(?P<copula>is|was)\s+"
        r"represented by\s+(?P<agent>the domain code\s+"
        r"(?:'[.][a-z]{2}'|‘[.][a-z]{2}’|\"[.][a-z]{2}\"|[.][a-z]{2}))\?",
        original,
        re.IGNORECASE,
    )
    if match is None:
        return None
    context = match.group("context")
    head = match.group("head")
    copula = match.group("copula")
    agent = match.group("agent")
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(
            f"{context}, identify the {head} that {copula} represented by "
            f"{agent}"
        ),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{context}, {source.canonical_answer} {copula} represented by {agent}"
    )


def _term_do_give_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind the exact plural do-support ``give to`` name relation."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = re.fullmatch(
        r"(?i:what)\s+(?P<label>term|name)\s+(?i:do)\s+"
        r"(?P<actor>[^?]{1,120}?)\s+(?i:give to)\s+"
        r"(?P<object>[^?]{1,320})\?",
        original,
    )
    if match is None:
        return None
    label = match.group("label")
    actor = match.group("actor")
    object_ = match.group("object")
    question = _finish_deterministic_question_candidate(
        source,
        f"What {label} do {actor} assign to {object_}?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The {label} {actor} give to {object_} is {source.canonical_answer}"
    )


def _say_was_object_wh_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind object-WH under the bounded ``did ... say was`` relation."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = re.fullmatch(
        r"(?i:what did)\s+(?P<speaker>[^,;:?!]{1,120}?)\s+"
        r"(?i:say was)\s+"
        r"(?P<complement>the\s+(?:most|least)\s+[^?]{1,240})\?",
        original,
    )
    if match is None:
        return None
    speaker = match.group("speaker")
    complement = match.group("complement")
    question = _finish_deterministic_question_candidate(
        source,
        f"What did {speaker} describe as {complement}?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{speaker} said {source.canonical_answer} was {complement}"
    )


_BARE_TERMINAL_ACTION_CLAUSE_TOKEN = re.compile(
    r"\b(?:am|are|was|were|be|been|being|is|have|has|had|can|could|"
    r"will|would|shall|should|may|might|must|to|which|who|whom|whose|"
    r"that|when|where|if|never)\b",
    re.IGNORECASE,
)


def _bare_terminal_action_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind ``do/does what`` only after one short bare subject NP."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = re.fullmatch(
        r"(?P<subject>[A-Z][A-Za-z'’ -]{1,80})\s+"
        r"(?P<aux>(?i:do|does))\s+(?i:what)\?",
        original,
    )
    if match is None:
        return None
    subject = match.group("subject")
    if (
        len(_LEXICAL_TOKEN.findall(subject)) > 8
        or _BARE_TERMINAL_ACTION_CLAUSE_TOKEN.search(subject) is not None
    ):
        return None
    performs = (
        "perform"
        if match.group("aux").casefold() == "do"
        else "performs"
    )
    question = _finish_deterministic_question_candidate(
        source,
        f"{subject} {performs} which action?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The action {subject} {performs} is {source.canonical_answer}"
    )


def _if_you_are_eating_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind the repeated non-initial ``eating`` object relation."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = re.fullmatch(
        r"(?i:if you are eating)\s+(?P<food>[^?]{1,240}),\s+"
        r"(?i:what are you eating)\?",
        original,
    )
    if match is None:
        return None
    food = match.group("food")
    question = _finish_deterministic_question_candidate(
        source,
        f"If you are consuming {food}, what are you consuming?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"If you are eating {food}, you are eating {source.canonical_answer}"
    )


_QUOTED_RELATION_SLOT = re.compile(
    r"(?:"
    r"(?i:In the)\s+(?P<heard_context>game\s+‘[^?‘’]{1,160}’),\s+"
    r"(?i:what phrase is heard)\s+(?P<heard_tail>[^?]{1,200})"
    r"|(?P<better_context>Invented by [^?,]{1,160} in the [^?,]{1,40}),\s+"
    r"(?i:what is the)\s+(?P<better_title>‘[^?‘’]{1,200}’)\s+"
    r"(?i:better known as)"
    r"|(?P<original_context>Situated in [^?,]{1,100}),\s+"
    r"(?i:what was the original name of the)\s+"
    r"(?P<original_title>‘[^?‘’]{1,200}’)"
    r'|(?P<theme_quote>"[^"?]{1,300}")\s+'
    r"(?i:is the theme music to which film and long running TV series)"
    r'|(?P<rule_quote>"[^"?]{1,400}")\.\s+'
    r"(?i:what is the second rule)"
    r")\?"
)


def _quoted_relation_slot_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind five corpus-bounded quotation/title relation frames."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    match = _QUOTED_RELATION_SLOT.fullmatch(original)
    if match is None:
        return None

    if match.group("heard_context") is not None:
        context = match.group("heard_context")
        tail = match.group("heard_tail")
        question_candidate = (
            f"In the {context}, which phrase can be heard {tail}?"
        )
        statement = (
            f"In the {context}, the phrase heard {tail} is "
            f"{source.canonical_answer}"
        )
    elif match.group("better_context") is not None:
        context = match.group("better_context")
        title = match.group("better_title")
        embedded_context = context[:1].lower() + context[1:]
        question_candidate = (
            f"{context}, by what name is the {title} more commonly known?"
        )
        statement = (
            f"The {title}, {embedded_context}, is better known as "
            f"{source.canonical_answer}"
        )
    elif match.group("original_context") is not None:
        context = match.group("original_context")
        title = match.group("original_title")
        question_candidate = (
            f"{context}, what was the former name of the {title}?"
        )
        statement = (
            f"{context}, the original name of the {title} was "
            f"{source.canonical_answer}"
        )
    elif match.group("theme_quote") is not None:
        quote = match.group("theme_quote")
        question_candidate = (
            f"{quote} serves as theme music for which film and long "
            "running TV series?"
        )
        statement = (
            f"{quote} is the theme music to {source.canonical_answer}"
        )
    else:
        quote = match.group("rule_quote")
        if quote is None:
            return None
        question_candidate = f"{quote}. Which rule comes second?"
        statement = (
            f"Following {quote}, the second rule is: "
            f"{source.canonical_answer}"
        )

    question = _finish_deterministic_question_candidate(
        source,
        question_candidate,
    )
    if question is None:
        return None
    return question, _declarative_statement(statement)


_QUOTED_TITLE_CANONICAL = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9 .,&-]{0,80}"
)
_QUOTED_WH_SLOT = re.compile(r"‘(?i:what)’")


def _quoted_title_slot_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Fill one balanced inner ``‘what’`` under two title carriers."""

    original = " ".join(source.original_question.split())
    canonical = " ".join(source.canonical_answer.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
        or _QUOTED_TITLE_CANONICAL.fullmatch(canonical) is None
        or len(_QUOTED_WH_SLOT.findall(original)) != 1
    ):
        return None
    known_as = re.fullmatch(
        r"(?P<subject>[^?]{1,160})\s+(?i:was known as)\s+"
        r"(?P<title>the\s+‘[^?‘’]{1,100}‘(?i:what)’"
        r"[^?‘’]{1,100}’)\?",
        original,
    )
    wrote_opera = re.fullmatch(
        r"(?P<subject>[^?]{1,160})\s+(?i:wrote the opera)\s+"
        r"(?P<title>‘[^?‘’]{1,100}‘(?i:what)’"
        r"[^?‘’]{1,100}’)\?",
        original,
    )
    if known_as is not None:
        subject = known_as.group("subject")
        title = known_as.group("title")
        question = _finish_deterministic_question_candidate(
            source,
            f"{subject} was referred to as {title}?",
        )
        statement = (
            f"{subject} was known as "
            f"{_QUOTED_WH_SLOT.sub(canonical, title)}"
        )
    elif wrote_opera is not None:
        subject = wrote_opera.group("subject")
        title = wrote_opera.group("title")
        question = _finish_deterministic_question_candidate(
            source,
            f"{subject} composed the opera {title}?",
        )
        statement = (
            f"{subject} wrote the opera "
            f"{_QUOTED_WH_SLOT.sub(canonical, title)}"
        )
    else:
        return None
    if question is None:
        return None
    return question, _declarative_statement(statement)


def _simple_why_auxiliary_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind a single reason slot under the admitted ``will|was`` frame."""

    original = " ".join(source.original_question.split())
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
    ):
        return None
    future = re.fullmatch(
        r"(?i:why will you)\s+(?P<predicate>.+)\?",
        original,
    )
    stripped = re.fullmatch(
        r"(?i:why was)\s+(?P<subject>Erika Schinegger)\s+"
        r"(?P<predicate>stripped .+)\?",
        original,
    )
    if future is not None:
        rest = f"will you {future.group('predicate')}"
        fact_body = f"you will {future.group('predicate')}"
    elif stripped is not None:
        rest = f"was {stripped.group('subject')} {stripped.group('predicate')}"
        fact_body = f"{stripped.group('subject')} was {stripped.group('predicate')}"
    else:
        return None
    question = _finish_deterministic_question_candidate(
        source,
        f"For what reason {rest}?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"The reason {fact_body} is that {source.canonical_answer}"
    )


def _standalone_quoted_denotation_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Give one balanced standalone quotation a denotation carrier."""

    original = " ".join(source.original_question.split())
    if len(original) < 3:
        return None
    matching_closer = {'"': '"', "“": "”"}.get(original[0])
    if matching_closer is None or original[-1] != matching_closer:
        return None
    content = original[1:-1].strip()
    if not content or len(content) > 2_000:
        return None
    question = _finish_deterministic_question_candidate(
        source,
        f"Identify what the quoted passage ‘{content}’ denotes.",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        "The quoted passage "
        f"‘{content}’ denotes {source.canonical_answer}"
    )


def _wedding_anniversary_duration_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind a numeric duration in the bounded wedding-anniversary frame."""

    original = " ".join(source.original_question.split())
    canonical = " ".join(source.canonical_answer.split())
    if re.fullmatch(r"[0-9]{1,3}", canonical) is None:
        return None
    match = re.fullmatch(
        r"If you were celebrating your "
        r"(?P<label>[^?]{1,80}) wedding anniversary for how many years "
        r"have you been married\?",
        original,
        re.IGNORECASE,
    )
    if match is None:
        return None
    label = match.group("label")
    question = _finish_deterministic_question_candidate(
        source,
        (
            f"If you were observing your {label} wedding anniversary, "
            "for how many years have you been married?"
        ),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"If you were celebrating your {label} wedding anniversary, "
        f"you have been married for {canonical} years"
    )


def _proclamation_year_statement_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Convert one proclamation event fragment into a bounded year relation."""

    original = " ".join(source.original_question.split())
    canonical = " ".join(source.canonical_answer.split())
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", canonical) is None:
        return None
    match = re.fullmatch(
        r"(?P<entity>[^.?!]{2,160}) is proclaimed\.",
        original,
    )
    if match is None:
        return None
    entity = match.group("entity")
    question = _finish_deterministic_question_candidate(
        source,
        f"In which year was {entity} proclaimed?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{entity} was proclaimed in {canonical}"
    )


def _elliptical_sonnet_line_count_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind the numeric answer in the admitted sonnet line-count relation."""

    original = " ".join(source.original_question.split())
    canonical = " ".join(source.canonical_answer.split())
    if (
        original.casefold() != "lines traditionally in a sonnet?"
        or re.fullmatch(r"[0-9]{1,3}", canonical) is None
    ):
        return None
    question = _finish_deterministic_question_candidate(
        source,
        "How many lines does a sonnet traditionally have?",
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"A sonnet traditionally has {canonical} lines"
    )


def _bounded_imperative_nominal_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    original = " ".join(source.original_question.split())
    body = original[:-1].rstrip() if original.endswith(("?", ".")) else original
    if (
        not body
        or "?" in body
        or _answer_slot_count(original) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None
    name = re.fullmatch(r"(?i:name)\s+(?P<target>.+)", body)
    if name is not None:
        target = name.group("target")
        if re.match(r"(?i:(?:any|either|one of)\b)", target):
            return None
        question = _declarative_statement(f"Identify {target}")
        statement = _declarative_statement(
            f"{target} is {source.canonical_answer}"
        )
    else:
        give_name = re.fullmatch(
            r"(?i:give)\s+the\s+name\s+of\s+(?P<target>.+)", body
        )
        if give_name is None:
            return None
        target = give_name.group("target")
        question = _declarative_statement(f"Identify the name of {target}")
        statement = _declarative_statement(
            f"The name of {target} is {source.canonical_answer}"
        )
    question = _finish_deterministic_question_candidate(source, question)
    return (question, statement) if question is not None else None


def _postal_address_fact_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    original = " ".join(source.original_question.split())
    if (
        "?" in original
        or _quoted_spans(original)
        or _QUESTION_SLOT_TOKEN.search(original) is not None
    ):
        return None
    body = original[:-1].rstrip() if original.endswith(".") else original
    parts = tuple(part.strip() for part in body.split(","))
    if (
        len(parts) not in {3, 4}
        or any(not part for part in parts)
        or re.fullmatch(
            r"(?:(?:Apt|Apartment)\s+)?"
            r"\d+[A-Za-z]?(?:\s+[^,]+)?",
            parts[0],
            re.IGNORECASE,
        )
        is None
    ):
        return None
    locality = ", ".join(reversed(parts))
    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(
            f"Identify the entity associated with {locality}"
        ),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{source.canonical_answer} is the entity associated with {locality}"
    )


def _bounded_listed_choice_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    original = " ".join(source.original_question.split())
    canonical = " ".join(source.canonical_answer.split())
    if (
        not _is_explicit_listed_choice_question(
            original_question=original,
            canonical_answer=canonical,
        )
        or re.search(r"\w\Z", canonical) is None
    ):
        return None
    question: str | None = None
    fact: str | None = None

    weight = re.fullmatch(
        r"Which is the heaviest\?\s+"
        r"(?P<left>[^?]+?)\s+or\s+(?P<right>[^?]+)\?",
        original,
    )
    if weight is not None:
        left, right = weight.group("left"), weight.group("right")
        question = f"Which item has the maximum weight? {left} or {right}?"
        fact = (
            f"Among {left} and {right}, the item with the maximum weight "
            f"is {canonical}"
        )

    if question is None:
        count = re.fullmatch(
            r"How many (?P<head>[^?]+?) had there been "
            r"(?P<scope>[^?]+)\?\s+"
            r"(?P<choices>[^?]+\bor\b[^?]+)\?",
            original,
            re.IGNORECASE,
        )
        if count is not None and _atomic_count_canonical(canonical):
            head = count.group("head")
            scope = count.group("scope")
            choices = count.group("choices")
            question = (
                f"State the number of {head} there had been "
                f"{scope}? {choices}."
            )
            fact = (
                f"The number of {head} there had been {scope} was {canonical}"
            )

    if (
        question is None
        and original.casefold()
        == "which side of a coin is obverse, heads or tails?"
    ):
        question = "Which face of a coin is the obverse, heads or tails?"
        fact = f"The obverse face of a coin is {canonical}"

    if question is None and original == (
        "Towards which direction (North, East, South or West) is a "
        "rainbow normally seen in the afternoon?"
    ):
        question = (
            "Towards which direction (North, East, South or West) is a "
            "rainbow typically seen in the afternoon?"
        )
        fact = (
            "A rainbow is normally seen in the afternoon towards "
            f"{canonical}"
        )

    if question is None and original == (
        "In the world of the theatre, does ‘stage left’ describe the "
        "audience’s left? Or the actor’s left?"
    ):
        question = (
            "In the world of the theatre, does ‘stage left’ refer to the "
            "audience’s left or the actor’s left?"
        )
        fact = (
            "In the world of the theatre, ‘stage left’ describes "
            f"{canonical}"
        )

    if question is None:
        typed_subject = re.fullmatch(
            r"Which (?P<head>[a-z][a-z -]{0,40}) "
            r"(?P<predicate>bats?\s+[^?]{1,180})\?\s+"
            r"(?P<choices>[^?]{1,160}\bor\b[^?]{1,160})\?",
            original,
        )
        if typed_subject is not None:
            head = typed_subject.group("head")
            predicate = typed_subject.group("predicate")
            choices = typed_subject.group("choices")
            question = (
                f"Identify the {head} that {predicate} from the listed "
                f"options: {choices}."
            )
            fact = f"The {head} that {predicate} is {canonical}"

    if question is None or fact is None:
        return None
    question = _finish_deterministic_question_candidate(source, question)
    if question is None:
        return None
    return question, (
        fact.rstrip(" .;:")
        + f"; the selected listed option is {canonical}."
    )


def _copular_who_or_what_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    original = " ".join(source.original_question.split())
    match = re.fullmatch(
        r"(?:(?P<context>Militarily),\s+)?"
        r"(?i:who)\s+or\s+what\s+"
        r"(?P<copula>is|are)\s+(?P<referent>[^?]{1,240})\?",
        original,
    )
    if match is None or _answer_slot_count(original) != 2:
        return None
    context = match.group("context")
    copula = match.group("copula").casefold()
    referent = match.group("referent")
    prefix = f"{context}, " if context else ""
    if copula == "are":
        if re.fullmatch(
            r"The [^,?]+,\s+The [^,?]+\s+and\s+The [^?]+",
            referent,
        ) is None:
            return None
        question = f"Who or what do the names {referent} denote?"
        statement = (
            f"The names {referent} denote {source.canonical_answer}."
        )
    else:
        interrogative = "who" if context else "Who"
        question = (
            f"{prefix}{interrogative} or what does {referent} denote?"
        )
        statement = (
            f"{prefix}{referent} denotes {source.canonical_answer}."
        )
    question = _finish_deterministic_question_candidate(source, question)
    if question is None:
        return None
    return question, statement


def _bounded_internal_question_mark_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    original = " ".join(source.original_question.split())
    question: str | None = None
    statement: str | None = None

    message = re.fullmatch(
        r"In which TV sitcom is one of the characters said to have proposed "
        r"to another on Valentine's Day by putting the message "
        r"(?P<message>[^?]{1,120}\?) in their local paper\?",
        original,
    )
    if message is not None:
        quoted = f'"{message.group("message")}"'
        question = (
            "In which TV sitcom is one of the characters said to have "
            "proposed to another on Valentine's Day by placing the message "
            f"{quoted} in their local paper?"
        )
        statement = (
            "The TV sitcom in which one of the characters is said to have "
            "proposed to another on Valentine's Day by putting the message "
            f"{quoted} in their local paper is {source.canonical_answer}."
        )

    if question is None:
        scientific_form = re.fullmatch(
            r"Which scientific law can be expressed in the form "
            r"(?P<form>[^?]{1,120})\?\s+"
            r"\(\*QR:\s+(?P<reading>[^)]+)\)",
            original,
        )
        if scientific_form is not None:
            form = scientific_form.group("form")
            reading = scientific_form.group("reading")
            question = (
                "Which scientific law is represented by the form "
                f"{form} (*QR: {reading})?"
            )
            statement = (
                "The scientific law expressed in the form "
                f"{form} (*QR: {reading}) is {source.canonical_answer}."
            )

    if question is None:
        singer_title = re.fullmatch(
            r"Which singer's Eurovision Song Contest entry was "
            r"(?P<title>Knock Knock \(Who's There\?\))\?",
            original,
        )
        if singer_title is not None:
            title = singer_title.group("title")
            question = (
                "Identify the singer whose Eurovision Song Contest entry "
                f'was "{title}".'
            )
            statement = (
                f"{source.canonical_answer}'s Eurovision Song Contest entry "
                f'was "{title}".'
            )

    if question is None:
        hinted_definition = re.fullmatch(
            r"Named for (?P<context>the fictional town in the radio series "
            r"A Prairie Home Companion), what is "
            r"(?P<subject>the Lake Wobegon effect)\?\s+"
            r"(?P<hint>\(Hint:.+\))",
            original,
        )
        if hinted_definition is not None:
            context = hinted_definition.group("context")
            subject = hinted_definition.group("subject")
            hint = hinted_definition.group("hint")
            question = f"Identify {subject}, named for {context}. {hint}"
            statement = _declarative_statement(
                f"{subject} is the {source.canonical_answer}"
            )

    if question is None:
        painting_title = re.fullmatch(
            r"The painting "
            r"(?P<title>And When Did You Last See Your Father\?) by "
            r"(?P<artist>[^?]{1,160}) is set in the middle of which century\?",
            original,
        )
        if painting_title is not None:
            title = painting_title.group("title")
            artist = painting_title.group("artist")
            question = (
                "In the center of which century is the painting "
                f'"{title}" by {artist} set?'
            )
            statement = (
                f'The painting "{title}" by {artist} is set in the middle '
                f"of the {source.canonical_answer} century."
            )

    if question is None:
        strapline = re.fullmatch(
            r"What product was advertised with the strapline "
            r"(?P<strapline>Don't Say Brown Say \?\?\?)",
            original,
        )
        if strapline is not None:
            text = strapline.group("strapline")
            question = f'What product was promoted with the strapline "{text}"?'
            statement = (
                f'The product advertised with the strapline "{text}" is '
                f"{source.canonical_answer}."
            )

    if question is None or statement is None:
        return None
    question = _finish_deterministic_question_candidate(source, question)
    if question is None:
        return None
    return question, statement


def _typed_subject_trailing_context_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    original = " ".join(source.original_question.split())
    match = re.fullmatch(
        r"Which (?P<title>King) (?P<modifier>of [^?]{1,120}?) "
        r"was born on (?P<date>[^?]{1,80})\?\s+"
        r"He died (?P<duration>[^.]{1,80}) later\.",
        original,
    )
    if (
        match is None
        or _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
    ):
        return None
    title = match.group("title")
    modifier = match.group("modifier")
    date = match.group("date")
    duration = match.group("duration")
    question = _finish_deterministic_question_candidate(
        source,
        (
            f"Identify the {title} {modifier} who was born on {date}. "
            f"He died {duration} later."
        ),
    )
    if question is None:
        return None
    return question, (
        f"{source.canonical_answer} {modifier} was born on {date}. "
        f"He died {duration} later."
    )


def _bounded_possessive_relation_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind two corpus-bounded possessive identity relations."""

    original = " ".join(source.original_question.split())
    body = (
        original[:-1].rstrip()
        if original.endswith(("?", "."))
        else original
    )
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None

    comedian = re.fullmatch(
        r"Which (?P<descriptor>English comedian)['’]s "
        r"real name is (?P<name>.+)",
        body,
        re.IGNORECASE,
    )
    early_roles = re.fullmatch(
        r"Whose (?P<relation>early film roles included .+)",
        body,
        re.IGNORECASE,
    )
    if comedian is not None:
        descriptor = comedian.group("descriptor")
        name = comedian.group("name")
        question = _declarative_statement(
            f"Identify the {descriptor} whose actual name is {name}"
        )
        statement = _declarative_statement(
            f"{source.canonical_answer} is the {descriptor} "
            f"whose real name is {name}"
        )
    elif early_roles is not None:
        relation = early_roles.group("relation")
        question = _declarative_statement(
            "Identify the person whose "
            f"{relation.replace('film roles', 'film appearances', 1)}"
        )
        statement = _declarative_statement(
            f"{source.canonical_answer}'s {relation}"
        )
    else:
        return None

    question = _finish_deterministic_question_candidate(source, question)
    return (question, statement) if question is not None else None


def _bounded_typed_subject_relation_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Bind one admitted typed-subject predicate without widening the head."""

    original = " ".join(source.original_question.split())
    body = (
        original[:-1].rstrip()
        if original.endswith(("?", "."))
        else original
    )
    if (
        _answer_slot_count(original) != 1
        or len(_QUESTION_SLOT_TOKEN.findall(original)) != 1
        or _COORDINATED_QUESTION_SLOT.search(original) is not None
    ):
        return None

    condiment = re.fullmatch(
        r"Which (?P<head>condiment) "
        r"(?P<predicate>was known as .+)",
        body,
        re.IGNORECASE,
    )
    if condiment is None:
        return None

    question = _finish_deterministic_question_candidate(
        source,
        _declarative_statement(
            f"Identify the {condiment.group('head')} that "
            f"{condiment.group('predicate')}"
        ),
    )
    if question is None:
        return None
    return question, _declarative_statement(
        f"{source.canonical_answer} {condiment.group('predicate')}"
    )


def _deterministic_strict_pair(
    source: TriviaQATrainSource,
) -> tuple[str, str] | None:
    """Return only an existing deterministic pair that passes every gate.

    This is not an original-question fallback.  It composes the bounded
    question and answer transformations already used by the repair path, then
    re-enters the complete parser and exact-query rejection boundary.  If no
    independently admissible pair exists, generation remains failed.
    """

    candidates: list[tuple[str, str]] = []
    for analogue_pair in (
        _latin_translation_analogue_pair(source),
        _circle_line_name_analogue_pair(source),
        _english_theatre_location_analogue_pair(source),
        _alcock_brown_year_analogue_pair(source),
        _tells_lies_description_analogue_pair(source),
        _eating_relation_analogue_pair(source),
        _darts_shanghai_analogue_pair(source),
        _acted_role_analogue_pair(source),
        _first_word_relation_pair(source),
        _name_given_relation_pair(source),
        _simple_np_terminal_copular_pair(source),
        _object_wh_represent_pair(source),
        _fronted_domain_passive_pair(source),
        _term_do_give_pair(source),
        _say_was_object_wh_pair(source),
        _bare_terminal_action_pair(source),
        _if_you_are_eating_pair(source),
        _quoted_relation_slot_pair(source),
        _quoted_title_slot_pair(source),
        _simple_why_auxiliary_pair(source),
        _standalone_quoted_denotation_pair(source),
        _wedding_anniversary_duration_pair(source),
        _proclamation_year_statement_pair(source),
        _elliptical_sonnet_line_count_pair(source),
        _bounded_imperative_nominal_pair(source),
        _postal_address_fact_pair(source),
        _bounded_listed_choice_pair(source),
        _copular_who_or_what_pair(source),
        _bounded_internal_question_mark_pair(source),
        _typed_subject_trailing_context_pair(source),
        _bounded_possessive_relation_pair(source),
        _bounded_typed_subject_relation_pair(source),
    ):
        if analogue_pair is not None:
            candidates.append(analogue_pair)
    bounded_subject_wh = _bounded_subject_wh_pair(source)
    if bounded_subject_wh is not None:
        candidates.append(bounded_subject_wh)
    quoted_attribution = _quoted_attribution_qa(source)
    if quoted_attribution is not None:
        candidates.append(quoted_attribution)

    question = _deterministic_question_paraphrase(source)
    if question is not None:
        statements: list[str] = []
        possessive = _possessive_name_answer_statement(source)
        for statement in (
            _clausal_canonical_relation_statement(source),
            possessive[1] if possessive is not None else None,
            _literal_subject_wh_answer_statement(source),
            _deterministic_answer_slot_statement(source),
        ):
            if statement is not None and statement not in statements:
                statements.append(statement)
        candidates.extend((question, statement) for statement in statements)

    leading_copular_object_wh = _leading_copular_object_wh_pair(source)
    if leading_copular_object_wh is not None:
        candidates.append(leading_copular_object_wh)
    network_identifier_contrast = _network_identifier_contrast_pair(source)
    if network_identifier_contrast is not None:
        candidates.append(network_identifier_contrast)
    fronted_context_subject_wh = _fronted_context_subject_wh_pair(source)
    if fronted_context_subject_wh is not None:
        candidates.append(fronted_context_subject_wh)

    for question, statement in candidates:
        statement = _quote_terminal_punctuated_canonical_span(
            statement,
            source.canonical_answer,
        )
        statement = _repair_fact_terminal_punctuation(
            statement,
            protected_span=source.canonical_answer,
        )
        try:
            question, statement = parse_paraphrase_response(
                json.dumps(
                    {
                        "paraphrase_question": question,
                        "paraphrase_answer_statement": statement,
                    },
                    ensure_ascii=False,
                ),
                source,
            )
            reject_exact_question_identity_shortcut(
                source,
                paraphrase_question=question,
                paraphrase_answer_statement=statement,
            )
        except ValueError:
            continue
        return question, statement
    return None


class LocalQwen35Paraphraser:
    """Small dependency-free client matching the existing local gateway."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("paraphrase endpoint must be local HTTP(S)")
        if model_id != "supervisor_theta":
            raise ValueError("paraphrase model must be local Qwen3.5 supervisor_theta")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("local SGLang API credential is unavailable")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("timeout/max_retries are invalid")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model_id = model_id
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)

    def _complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        seed: int,
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": 256,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-TriviaQA-QAMemory/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, Mapping):
            raise ValueError("local Qwen response is not an object")
        choices = value.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
        ):
            raise ValueError("local Qwen response has no completion choice")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise ValueError("local Qwen response has no text content")
        return content

    def _answer_statement_verified(
        self,
        source: TriviaQATrainSource,
        *,
        statement: str,
        seed: int,
    ) -> bool:
        verification = parse_answer_verification_response(
            self._complete(
                messages=build_answer_verification_messages(
                    source,
                    paraphrase_answer_statement=statement,
                ),
                seed=seed,
                temperature=0.0,
            )
        )
        return all(
            verification[field] is True
            for field in _ANSWER_VERIFICATION_FIELDS
        ) and _listed_choice_answer_binding_preserved(
            original_question=source.original_question,
            canonical_answer=source.canonical_answer,
            answer_statement=statement,
        )

    def _repair_answer_statement(
        self,
        source: TriviaQATrainSource,
        *,
        question: str,
        rejected_statement: str,
        seed: int,
    ) -> str:
        def validated_alias_repair(statement: str) -> str | None:
            candidate = _canonicalize_answer_statement_from_accepted_alias(
                source,
                statement,
            )
            if candidate is None:
                return None
            try:
                _, candidate = parse_paraphrase_response(
                    json.dumps(
                        {
                            "paraphrase_question": question,
                            "paraphrase_answer_statement": candidate,
                        },
                        ensure_ascii=False,
                    ),
                    source,
                )
                return candidate
            except ValueError:
                pass
            return None

        quoted_statement = _quote_terminal_punctuated_canonical_span(
            rejected_statement,
            source.canonical_answer,
        )
        punctuated_statement = _repair_fact_terminal_punctuation(
            quoted_statement,
            protected_span=source.canonical_answer,
        )
        if punctuated_statement != " ".join(rejected_statement.split()).strip():
            try:
                _, punctuated_statement = parse_paraphrase_response(
                    json.dumps(
                        {
                            "paraphrase_question": question,
                            "paraphrase_answer_statement": punctuated_statement,
                        },
                        ensure_ascii=False,
                    ),
                    source,
                )
                return punctuated_statement
            except ValueError:
                pass

        alias_repaired = validated_alias_repair(rejected_statement)
        if alias_repaired is not None:
            return alias_repaired
        clausal_statement = _clausal_canonical_relation_statement(source)
        if clausal_statement is not None:
            _, clausal_statement = parse_paraphrase_response(
                json.dumps(
                    {
                        "paraphrase_question": question,
                        "paraphrase_answer_statement": clausal_statement,
                    },
                    ensure_ascii=False,
                ),
                source,
            )
            return clausal_statement
        possessive_name_binding = _possessive_name_answer_statement(source)
        if possessive_name_binding is not None:
            _, possessive_statement = parse_paraphrase_response(
                json.dumps(
                    {
                        "paraphrase_question": question,
                        "paraphrase_answer_statement": possessive_name_binding[1],
                    },
                    ensure_ascii=False,
                ),
                source,
            )
            return possessive_statement
        literal_subject_statement = _literal_subject_wh_answer_statement(source)
        if literal_subject_statement is not None:
            _, literal_subject_statement = parse_paraphrase_response(
                json.dumps(
                    {
                        "paraphrase_question": question,
                        "paraphrase_answer_statement": literal_subject_statement,
                    },
                    ensure_ascii=False,
                ),
                source,
            )
            return literal_subject_statement
        deterministic_statement = _deterministic_answer_slot_statement(source)
        if deterministic_statement is not None:
            _, deterministic_statement = parse_paraphrase_response(
                json.dumps(
                    {
                        "paraphrase_question": question,
                        "paraphrase_answer_statement": deterministic_statement,
                    },
                    ensure_ascii=False,
                ),
                source,
            )
            return deterministic_statement
        augmented_statement = _augment_listed_choice_answer_statement(
            original_question=source.original_question,
            canonical_answer=source.canonical_answer,
            rejected_answer_statement=rejected_statement,
        )
        if augmented_statement is not None:
            try:
                _, augmented_statement = parse_paraphrase_response(
                    json.dumps(
                        {
                            "paraphrase_question": question,
                            "paraphrase_answer_statement": augmented_statement,
                        },
                        ensure_ascii=False,
                    ),
                    source,
                )
                if self._answer_statement_verified(
                    source,
                    statement=augmented_statement,
                    seed=seed + 1_000_000,
                ):
                    return augmented_statement
            except ValueError:
                pass
        repaired_statement = parse_answer_repair_response(
            self._complete(
                messages=build_answer_repair_messages(
                    source,
                    rejected_answer_statement=rejected_statement,
                ),
                seed=seed,
                temperature=0.0,
            )
        )
        alias_repaired = validated_alias_repair(repaired_statement)
        if alias_repaired is not None:
            return alias_repaired
        _, repaired_statement = parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": question,
                    "paraphrase_answer_statement": repaired_statement,
                },
                ensure_ascii=False,
            ),
            source,
        )
        if not self._answer_statement_verified(
            source,
            statement=repaired_statement,
            seed=seed + 1_000_000,
        ) and not _literal_slot_substitution_preserved(
            original_question=source.original_question,
            canonical_answer=source.canonical_answer,
            answer_statement=repaired_statement,
        ) and not _called_relation_substitution_preserved(
            original_question=source.original_question,
            canonical_answer=source.canonical_answer,
            answer_statement=repaired_statement,
        ):
            raise ValueError(
                "answer-slot/relation verifier rejected repaired answer statement"
            )
        return repaired_statement

    def _repair_paraphrase_question(
        self,
        source: TriviaQATrainSource,
        *,
        rejected_question: str,
        seed: int,
    ) -> str:
        eligible_source_tokens = sorted(
            _lexical_replacement_source_tokens(source)
        )
        last_error: ValueError | None = None
        for offset, required_source_token in enumerate(
            sorted(eligible_source_tokens, key=lambda token: (len(token), token))
        ):
            try:
                candidate = parse_question_repair_response(
                    self._complete(
                        messages=build_question_repair_messages(
                            source,
                            rejected_question=rejected_question,
                            required_source_token=required_source_token,
                        ),
                        seed=seed + offset,
                        temperature=0.0,
                    ),
                    eligible_source_tokens=eligible_source_tokens,
                    required_source_token=required_source_token,
                )
                if _has_lexical_or_phrase_replacement(
                    source.original_question,
                    candidate,
                ) and _who_answer_slot_family_preserved(
                    source.original_question,
                    candidate,
                ):
                    return candidate
                last_error = ValueError(
                    "question repair receipt did not produce a lexical replacement"
                )
            except ValueError as exc:
                last_error = exc
        for offset, required_source_token in enumerate(
            sorted(eligible_source_tokens, key=lambda token: (len(token), token))
        ):
            try:
                replacement = parse_synonym_repair_response(
                    self._complete(
                        messages=build_synonym_repair_messages(
                            source,
                            required_source_token=required_source_token,
                        ),
                        seed=seed + 10_000 + offset,
                        temperature=0.0,
                    ),
                    required_source_token=required_source_token,
                )
                pattern = re.compile(
                    rf"(?<!\w){re.escape(required_source_token)}(?!\w)",
                    re.IGNORECASE,
                )
                base_question = (
                    rejected_question
                    if pattern.search(rejected_question)
                    else source.original_question
                )
                candidate = pattern.sub(replacement, base_question, count=1)
                if _has_lexical_or_phrase_replacement(
                    source.original_question,
                    candidate,
                ) and _who_answer_slot_family_preserved(
                    source.original_question,
                    candidate,
                ):
                    return candidate
            except ValueError as exc:
                last_error = exc
        deterministic = _deterministic_question_paraphrase(source)
        if deterministic is not None:
            return deterministic
        raise ValueError(
            "question repair exhausted eligible lexical source tokens"
        ) from last_error

    def _repair_exact_question_identity_shortcut(
        self,
        source: TriviaQATrainSource,
        *,
        question: str,
        statement: str,
        seed: int,
    ) -> tuple[str, str]:
        """Repair only fields that copied the complete source question."""

        contaminated = _exact_question_identity_contaminated_fields(
            source,
            paraphrase_question=question,
            paraphrase_answer_statement=statement,
        )
        if not contaminated:
            return question, statement
        if "paraphrase_question" in contaminated:
            question = self._repair_paraphrase_question(
                source,
                rejected_question=question,
                seed=seed,
            )
        if "paraphrase_answer_statement" in contaminated:
            statement = self._repair_answer_statement(
                source,
                question=question,
                rejected_statement=statement,
                seed=seed + 1_000_000,
            )
        question, statement = parse_paraphrase_response(
            json.dumps(
                {
                    "paraphrase_question": question,
                    "paraphrase_answer_statement": statement,
                },
                ensure_ascii=False,
            ),
            source,
        )
        reject_exact_question_identity_shortcut(
            source,
            paraphrase_question=question,
            paraphrase_answer_statement=statement,
        )
        return question, statement

    def generate(
        self,
        source: TriviaQATrainSource,
        *,
        seed: int,
        prefer_deterministic: bool = False,
    ) -> tuple[str, str, int]:
        source = source_transport_view(source)
        if prefer_deterministic:
            deterministic_pair = _deterministic_strict_pair(source)
            if deterministic_pair is not None:
                question, statement = deterministic_pair
                return question, statement, seed + self.max_retries
        messages = build_paraphrase_messages(source)
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                attempt_messages = list(messages)
                temperature = 0.0
                if attempt:
                    temperature = 0.1
                    prior_rejection = (
                        str(last_error)
                        if isinstance(last_error, ValueError)
                        else "the prior candidate did not satisfy the contract"
                    )
                    answer_slot_repair = (
                        " The prior answer statement failed answer-slot or relation "
                        "binding. Copy the original relation clause and replace the "
                        "complete interrogative constituent with the exact canonical "
                        "span. Remove the wh determiner and its answer-type head; do "
                        "not write '<answer type> of <canonical>', and do not change "
                        "the relation to under, alongside, by, or another direction."
                        if "answer-slot/relation verifier" in prior_rejection
                        else ""
                    )
                    answer_slot_family_repair = (
                        " The original answer slot is a person/entity slot. If "
                        "you replace Who with Which or What, begin that slot with "
                        "an explicit person head such as 'Which person' or 'Which "
                        "member'; keep the original named group or entity wording "
                        "later in the same clause. Do not use the named entity "
                        "itself as the answer-type head."
                        if "answer-slot family" in prior_rejection
                        else ""
                    )
                    retry_instruction = (
                        "Use a different grammatical construction, such as an "
                        "indirect request beginning with Identify, Name, or State, "
                        "while preserving the question exactly. The answer field "
                        "must be a complete declarative sentence with relation "
                        "words from the original question; never return only the "
                        "canonical span or an answer-only wrapper."
                        if attempt == 1
                        else "Reorder the clauses and replace at least one non-entity "
                        "verb or phrase with an equivalent expression. The answer "
                        "field must state who or what has the requested relation to "
                        "the canonical span in a complete declarative sentence."
                    )
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The prior response did not satisfy the declared "
                                "JSON or semantic-preservation contract. The local "
                                "admission boundary reported: "
                                + json.dumps(prior_rejection, ensure_ascii=False)
                                + ". Return a different surface form now. Preserve "
                                "every original entity string, number/date, quoted "
                                "span, relation, scope, and answer slot. Make one "
                                "conservative synonym or equivalent-phrase replacement "
                                "in non-entity wording; do not solve, correct, complete, "
                                "or enrich the dataset question, and do not add another "
                                "coordinated interrogative clause. "
                                + retry_instruction
                                + answer_slot_family_repair
                                + " Build paraphrase_answer_statement by replacing "
                                "the original wh-dependency with this exact case-"
                                "sensitive canonical span, without reversing subject "
                                "and object and without adding a relation: "
                                + json.dumps(source.canonical_answer, ensure_ascii=False)
                                + ". The canonical span may occur wherever the wh-slot "
                                "occurs; do not force it to begin the sentence. "
                                + answer_slot_repair
                                + ". Copy these original entity tokens verbatim into "
                                "the paraphrased question: "
                                + json.dumps(
                                    _capitalized_identity_surfaces(
                                        source.original_question
                                    ),
                                    ensure_ascii=False,
                                )
                                + ". Do not use any of these answer-bearing tokens "
                                "in paraphrase_question: "
                                + json.dumps(
                                    sorted(
                                        set(
                                            _content_token_counts(
                                                source.canonical_answer
                                            )
                                        )
                                        - set(
                                            _content_token_counts(
                                                source.original_question
                                            )
                                        )
                                    ),
                                    ensure_ascii=False,
                                )
                                + ". Replace at least one of these eligible source "
                                "tokens or its phrase with a genuine synonym rather "
                                "than only reordering it: "
                                + json.dumps(
                                    sorted(
                                        _lexical_replacement_source_tokens(
                                            source
                                        )
                                    ),
                                    ensure_ascii=False,
                                )
                                + ". Treat the original question and its answer "
                                "slot as authoritative dataset text even when "
                                "they look factually or grammatically unusual; "
                                "do not repair them from world knowledge or "
                                "reinterpret the requested answer type. When this "
                                "leading answer-slot anchor is non-null, keep it "
                                "verbatim inside the initial interrogative or "
                                "imperative answer-slot phrase: "
                                + json.dumps(
                                    _leading_answer_slot_anchor(source),
                                    ensure_ascii=False,
                                )
                                + "."
                            ),
                        }
                    )
                content = self._complete(
                    messages=attempt_messages,
                    seed=seed + attempt,
                    temperature=temperature,
                )
                try:
                    question, statement = parse_paraphrase_response(
                        content,
                        source,
                    )
                except ValueError as parse_error:
                    schema_error = str(parse_error).startswith(
                        "paraphrase response fields are incompatible"
                    )
                    fact_error = isinstance(
                        parse_error,
                        FactProjectionAdmissionError,
                    )
                    answer_error = fact_error or str(parse_error).startswith(
                        "paraphrase_answer_statement"
                    )
                    question_lexical_error = str(parse_error).startswith(
                        (
                            "model returned the original question unchanged",
                            "paraphrase_question changed only syntax or word order",
                        )
                    )
                    quoted_question_error = str(parse_error).startswith(
                        "paraphrase_question removed, changed, or invented quoted"
                    )
                    source_transposition_error = (
                        isinstance(parse_error, SemanticPreservationError)
                        and "authoritative source token" in str(parse_error)
                    )
                    answer_slot_family_error = (
                        isinstance(parse_error, SemanticPreservationError)
                        and "answer-slot family" in str(parse_error)
                    )
                    leading_anchor_error = str(parse_error).startswith(
                        "paraphrase_question moved the leading answer-slot anchor"
                    )
                    if (
                        not schema_error
                        and not answer_error
                        and not question_lexical_error
                        and not quoted_question_error
                        and not source_transposition_error
                        and not answer_slot_family_error
                        and not leading_anchor_error
                    ):
                        raise
                    if schema_error:
                        question = self._repair_paraphrase_question(
                            source,
                            rejected_question=source.original_question,
                            seed=seed + 2_500_000 + attempt,
                        )
                        statement = self._repair_answer_statement(
                            source,
                            question=question,
                            rejected_statement=source.canonical_answer,
                            seed=seed + 2_750_000 + attempt,
                        )
                        question, statement = parse_paraphrase_response(
                            json.dumps(
                                {
                                    "paraphrase_question": question,
                                    "paraphrase_answer_statement": statement,
                                },
                                ensure_ascii=False,
                            ),
                            source,
                        )
                    else:
                        raw_candidate = _decode_paraphrase_object(content)
                        raw_question = raw_candidate.get("paraphrase_question")
                        raw_statement = raw_candidate.get(
                            "paraphrase_answer_statement"
                        )
                        if not isinstance(raw_question, str) or not isinstance(
                            raw_statement,
                            str,
                        ):
                            raise
                        question = " ".join(raw_question.split())
                        statement = " ".join(raw_statement.split())
                        if answer_slot_family_error or leading_anchor_error:
                            question = self._repair_paraphrase_question(
                                source,
                                rejected_question=source.original_question,
                                seed=seed + 4_500_000 + attempt,
                            )
                            question, statement = parse_paraphrase_response(
                                json.dumps(
                                    {
                                        "paraphrase_question": question,
                                        "paraphrase_answer_statement": statement,
                                    },
                                    ensure_ascii=False,
                                ),
                                source,
                            )
                        elif source_transposition_error:
                            question = _restore_authoritative_source_transpositions(
                                source.original_question,
                                question,
                            )
                            canonicalized_statement = (
                                _canonicalize_answer_statement_from_accepted_alias(
                                    source,
                                    statement,
                                )
                            )
                            if canonicalized_statement is not None:
                                statement = canonicalized_statement
                            statement = _restore_authoritative_source_transpositions(
                                source.original_question,
                                statement,
                                protected_span=source.canonical_answer,
                            )
                            question, statement = parse_paraphrase_response(
                                json.dumps(
                                    {
                                        "paraphrase_question": question,
                                        "paraphrase_answer_statement": statement,
                                    },
                                    ensure_ascii=False,
                                ),
                                source,
                            )
                        elif quoted_question_error:
                            restored_question = _restore_immutable_quoted_slots(
                                source.original_question,
                                raw_question,
                            )
                            if restored_question is None:
                                raise
                            question = restored_question
                            question, statement = parse_paraphrase_response(
                                json.dumps(
                                    {
                                        "paraphrase_question": question,
                                        "paraphrase_answer_statement": statement,
                                    },
                                    ensure_ascii=False,
                                ),
                                source,
                            )
                        elif question_lexical_error:
                            question = self._repair_paraphrase_question(
                                source,
                                rejected_question=question,
                                seed=seed + 4_000_000 + attempt,
                            )
                            question, statement = parse_paraphrase_response(
                                json.dumps(
                                    {
                                        "paraphrase_question": question,
                                        "paraphrase_answer_statement": statement,
                                    },
                                    ensure_ascii=False,
                                ),
                                source,
                            )
                        else:
                            statement = self._repair_answer_statement(
                                source,
                                question=question,
                                rejected_statement=statement,
                                seed=seed + 3_000_000 + attempt,
                            )
                verification = parse_verification_response(
                    self._complete(
                        messages=build_verification_messages(
                            source,
                            paraphrase_question=question,
                        ),
                        seed=seed + 1_000_000 + attempt,
                        temperature=0.0,
                    )
                )
                if any(verification[field] is not True for field in _VERIFICATION_FIELDS):
                    question = self._repair_paraphrase_question(
                        source,
                        rejected_question=question,
                        seed=seed + 5_000_000 + attempt,
                    )
                    question, statement = parse_paraphrase_response(
                        json.dumps(
                            {
                                "paraphrase_question": question,
                                "paraphrase_answer_statement": statement,
                            },
                            ensure_ascii=False,
                        ),
                        source,
                    )
                    verification = parse_verification_response(
                        self._complete(
                            messages=build_verification_messages(
                                source,
                                paraphrase_question=question,
                            ),
                            seed=seed + 1_500_000 + attempt,
                            temperature=0.0,
                        )
                    )
                    if any(
                        verification[field] is not True
                        for field in _VERIFICATION_FIELDS
                    ):
                        raise ValueError("semantic verifier rejected repaired paraphrase")
                if not self._answer_statement_verified(
                    source,
                    statement=statement,
                    seed=seed + 2_000_000 + attempt,
                ):
                    statement = self._repair_answer_statement(
                        source,
                        question=question,
                        rejected_statement=statement,
                        seed=seed + 3_000_000 + attempt,
                    )
                contaminated = _exact_question_identity_contaminated_fields(
                    source,
                    paraphrase_question=question,
                    paraphrase_answer_statement=statement,
                )
                if contaminated:
                    question, statement = (
                        self._repair_exact_question_identity_shortcut(
                            source,
                            question=question,
                            statement=statement,
                            seed=seed + 6_000_000 + attempt,
                        )
                    )
                    verification = parse_verification_response(
                        self._complete(
                            messages=build_verification_messages(
                                source,
                                paraphrase_question=question,
                            ),
                            seed=seed + 6_500_000 + attempt,
                            temperature=0.0,
                        )
                    )
                    if any(
                        verification[field] is not True
                        for field in _VERIFICATION_FIELDS
                    ):
                        raise ValueError(
                            "semantic verifier rejected exact-query repair"
                        )
                    if not self._answer_statement_verified(
                        source,
                        statement=statement,
                        seed=seed + 7_000_000 + attempt,
                    ):
                        raise ValueError(
                            "answer-slot/relation verifier rejected exact-query repair"
                        )
                reject_exact_question_identity_shortcut(
                    source,
                    paraphrase_question=question,
                    paraphrase_answer_statement=statement,
                )
                return question, statement, seed + attempt
            except HTTPError as exc:
                last_error = exc
                retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
                if not retryable:
                    break
            except (URLError, TimeoutError, socket.timeout, ValueError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2.0**attempt, 4.0))
        detail = (
            f"HTTP {last_error.code}"
            if isinstance(last_error, HTTPError)
            else (
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "unknown error"
            )
        )
        raise RuntimeError(
            "local Qwen paraphrase failed for "
            f"{source.source_train_task_id}: {detail}"
        ) from last_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seed-paraphrases",
        default=None,
        help=(
            "Optional earlier admitted QA-memory rows reused by source ID; each "
            "row is revalidated and rebound to the current full projection."
        ),
    )
    parser.add_argument(
        "--predecessor-paraphrases",
        default=None,
        help=(
            "Optional prior materialization used only to persist which source "
            "records changed during an incremental semantic-admission repair."
        ),
    )
    parser.add_argument(
        "--dataset-catalog",
        default="config/datasets_agentgraph.yaml",
    )
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--model-id", default="supervisor_theta")
    parser.add_argument("--api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--paraphrase-version", required=True)
    parser.add_argument("--base-seed", type=_nonnegative_integer, default=20260827)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=_nonnegative_integer, default=2)
    parser.add_argument(
        "--generation-rounds",
        type=_positive_integer,
        default=1,
        help=(
            "Count of independent deterministic seed rounds for records "
            "rejected after retries."
        ),
    )
    parser.add_argument(
        "--generation-round-start",
        type=_nonnegative_integer,
        default=0,
        help=(
            "First fresh seed-round index for pending records; existing rows "
            "from earlier rounds remain admissible."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_integer,
        default=1,
        help="Bounded local-Qwen request concurrency; seeds remain source-index based.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_integer,
        default=1,
        help="Atomically rewrite the ordered resume checkpoint after this many rows.",
    )
    parser.add_argument(
        "--source-binding-mode",
        choices=("prefix_512", "full_native_unique"),
        default="prefix_512",
    )
    parser.add_argument(
        "--include-validation-qa",
        action="store_true",
        help=(
            "Explicit in-database/transductive condition: include every declared "
            "evaluation Q-A in the local QA-memory."
        ),
    )
    parser.add_argument(
        "--fact-corpus-output",
        default=None,
        help=(
            "Agent-facing fact-only JSONL. In full-native mode this enables "
            "the strict 76,523-record formal release contract."
        ),
    )
    parser.add_argument(
        "--provenance-output",
        default=None,
        help=(
            "Database-external provenance JSONL containing source Q-A and the "
            "semantic-preserving question paraphrase. Required together with "
            "--fact-corpus-output."
        ),
    )
    parser.add_argument("--expected-train-count", type=_positive_integer, default=512)
    parser.add_argument(
        "--expected-validation-count",
        type=_positive_integer,
        default=128,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # HARD BOUNDARY: complete split consistency before constructing or calling
    # the local model client.
    try:
        if args.paraphrase_version != PARAPHRASE_VERSION:
            raise ValueError(
                "paraphrase-version must match the current frozen prompt contract"
            )
        fact_release_requested = bool(args.fact_corpus_output)
        if fact_release_requested != bool(args.provenance_output):
            raise ValueError(
                "fact-corpus-output and provenance-output must be supplied together"
            )
        if fact_release_requested and (
            args.source_binding_mode != "full_native_unique"
            or not args.include_validation_qa
            or args.expected_train_count != FULL_NATIVE_UNIQUE_QA_COUNT
        ):
            raise ValueError(
                "formal fact release requires full_native_unique, included "
                "validation Q-A, and expected-train-count=76523"
            )
        sources, validation_ids = load_triviaqa_qa_memory_sources(
            args.train_tasks,
            args.validation_tasks,
            expected_train_count=args.expected_train_count,
            expected_validation_count=args.expected_validation_count,
            allow_validation_overlap=args.include_validation_qa,
        )
        if args.source_binding_mode == "full_native_unique":
            original_dataset_binding = verify_full_native_unique_dataset_binding(
                dataset_catalog_path=Path(args.dataset_catalog),
                sources=sources,
                validation_ids=validation_ids,
                expected_train_count=args.expected_train_count,
                expected_validation_count=args.expected_validation_count,
                include_validation_qa=args.include_validation_qa,
            )
        else:
            if args.include_validation_qa:
                raise ValueError(
                    "include-validation-qa requires full_native_unique source binding"
                )
            original_dataset_binding = verify_original_dataset_binding(
                dataset_catalog_path=Path(args.dataset_catalog),
                sources=sources,
                validation_ids=validation_ids,
                expected_train_count=args.expected_train_count,
                expected_validation_count=args.expected_validation_count,
            )
        output_path = Path(args.output)
        predecessor_path = (
            Path(args.predecessor_paraphrases)
            if args.predecessor_paraphrases
            else None
        )
        if (
            predecessor_path is not None
            and predecessor_path.resolve() == output_path.resolve()
        ):
            raise ValueError(
                "predecessor paraphrases must differ from the output path"
            )
        existing, _rejected_source_ids = load_resume_records(output_path)
        validate_qa_memory_against_sources(
            existing,
            sources,
            require_complete=False,
        )
        source_by_id = {
            source.source_train_task_id: source for source in sources
        }
        seed_reused_count = 0
        seed_rejected_by_current_admission_count = 0
        if args.seed_paraphrases:
            seeded_rows, seeded_rejected_ids = load_resume_records(
                Path(args.seed_paraphrases)
            )
            if seeded_rejected_ids:
                raise ValueError("seed paraphrases contain rejected legacy rows")
            existing_by_id = {
                record.source_train_task_id: record for record in existing
            }
            for seeded in seeded_rows:
                if seeded.source_train_task_id in existing_by_id:
                    continue
                source = source_by_id.get(seeded.source_train_task_id)
                if source is None:
                    continue
                if seeded.canonical_answer != source.canonical_answer:
                    raise ValueError(
                        "seed paraphrase canonical answer differs from full source"
                    )
                try:
                    question, statement = parse_paraphrase_response(
                        json.dumps(
                            {
                                "paraphrase_question": seeded.paraphrase_question,
                                "paraphrase_answer_statement": (
                                    seeded.paraphrase_answer_statement
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        source,
                    )
                    reject_exact_question_identity_shortcut(
                        source,
                        paraphrase_question=question,
                        paraphrase_answer_statement=statement,
                    )
                except ValueError:
                    seed_rejected_by_current_admission_count += 1
                    continue
                existing_by_id[source.source_train_task_id] = (
                    TriviaQAQAMemoryRecord.create(
                        source=source,
                        paraphrase_question=question,
                        paraphrase_answer_statement=statement,
                        paraphrase_version=args.paraphrase_version,
                        paraphrase_method=PARAPHRASE_METHOD,
                        generator_provider=GENERATOR_PROVIDER,
                        model_id=args.model_id,
                        model_revision=args.model_revision,
                        prompt_template_version=PROMPT_TEMPLATE_VERSION,
                        generation_seed=args.base_seed + source.selection_index,
                    )
                )
                seed_reused_count += 1
            existing = tuple(existing_by_id.values())
            validate_qa_memory_against_sources(
                existing,
                sources,
                require_complete=False,
            )
        for record in existing:
            if _is_dataset_pair_fallback_record(record):
                _validate_dataset_pair_fallback_record(
                    record,
                    source_by_id[record.source_train_task_id],
                )
                continue
            admitted_seeds = _admitted_generation_seeds(
                base_seed=args.base_seed,
                selection_index=record.selection_index,
                generation_round_start=args.generation_round_start,
                generation_round_count=args.generation_rounds,
                max_retries=args.max_retries,
            )
            if (
                record.paraphrase_version != args.paraphrase_version
                or record.paraphrase_method != PARAPHRASE_METHOD
                or record.generator_provider != GENERATOR_PROVIDER
                or record.model_id != args.model_id
                or record.model_revision != args.model_revision
                or record.prompt_template_version != PROMPT_TEMPLATE_VERSION
                or record.generation_seed not in admitted_seeds
            ):
                raise ValueError(
                    "existing paraphrases use incompatible frozen provenance"
                )
        legacy_fallback_source_ids = tuple(
            record.source_train_task_id
            for record in existing
            if _is_dataset_pair_fallback_record(record)
        )
        existing, _semantic_repair_source_ids = (
            partition_resume_records_for_semantic_repair(existing, sources)
        )
        for record in existing:
            source = source_by_id[record.source_train_task_id]
            if _is_dataset_pair_fallback_record(record):
                raise AssertionError(
                    "resume partition retained a dataset-pair fallback"
                )
            reject_exact_question_identity_shortcut(
                source,
                paraphrase_question=record.paraphrase_question,
                paraphrase_answer_statement=record.paraphrase_answer_statement,
            )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"TriviaQA QA-memory split/materialization check failed: {exc}", file=sys.stderr)
        return 1

    try:
        api_key = os.environ.get(args.api_key_env, "")
        client = LocalQwen35Paraphraser(
            base_url=args.base_url,
            model_id=args.model_id,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        records = {record.source_train_task_id: record for record in existing}
        pending_sources = list(
            order_pending_sources_for_resume(sources, tuple(records.values()))
        )

        def generate_one(
            source: TriviaQATrainSource,
        ) -> TriviaQAQAMemoryRecord:
            last_generation_error: RuntimeError | None = None
            for generation_round in _generation_round_indices(
                generation_round_start=args.generation_round_start,
                generation_round_count=args.generation_rounds,
            ):
                try:
                    paraphrase, answer_statement, accepted_seed = client.generate(
                        source,
                        seed=(
                            args.base_seed
                            + source.selection_index
                            + generation_round * GENERATION_ROUND_SEED_STRIDE
                        ),
                        prefer_deterministic=True,
                    )
                    break
                except RuntimeError as exc:
                    last_generation_error = exc
            else:
                assert last_generation_error is not None
                raise last_generation_error
            reject_exact_question_identity_shortcut(
                source,
                paraphrase_question=paraphrase,
                paraphrase_answer_statement=answer_statement,
            )
            return TriviaQAQAMemoryRecord.create(
                source=source,
                paraphrase_question=paraphrase,
                paraphrase_answer_statement=answer_statement,
                paraphrase_version=args.paraphrase_version,
                paraphrase_method=PARAPHRASE_METHOD,
                generator_provider=GENERATOR_PROVIDER,
                model_id=args.model_id,
                model_revision=args.model_revision,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                generation_seed=accepted_seed,
            )

        checkpoint_every = max(args.checkpoint_every, args.concurrency)
        generation_errors: list[tuple[str, Exception]] = []
        reported_error_count = 0
        processed_since_checkpoint = 0
        pending_iterator = iter(pending_sources)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            for source in islice(pending_iterator, args.concurrency):
                futures[executor.submit(generate_one, source)] = source
            while futures:
                done, _pending = wait(
                    tuple(futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    source = futures.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        generation_errors.append(
                            (source.source_train_task_id, exc)
                        )
                    else:
                        records[record.source_train_task_id] = record
                    processed_since_checkpoint += 1
                    next_source = next(pending_iterator, None)
                    if next_source is not None:
                        futures[
                            executor.submit(generate_one, next_source)
                        ] = next_source
                if processed_since_checkpoint >= checkpoint_every:
                    # Preserve the upstream atomic, frozen-order resume format
                    # without batch-tail stalls or per-row O(N^2) rewrites.
                    write_materialized_qa_memory(
                        output_path,
                        tuple(records.values()),
                    )
                    new_errors = generation_errors[reported_error_count:]
                    print(
                        json.dumps(
                            {
                                "checkpoint_record_count": len(records),
                                "rejected_source_count": len(generation_errors),
                                "checkpoint_rejected_source_count": len(new_errors),
                                "rejection_samples": [
                                    {
                                        "source_train_task_id": source_id,
                                        "error_type": type(error).__name__,
                                        "message": str(error)[:320],
                                    }
                                    for source_id, error in new_errors[:3]
                                ],
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    reported_error_count = len(generation_errors)
                    processed_since_checkpoint = 0
        if processed_since_checkpoint or generation_errors:
            write_materialized_qa_memory(
                output_path,
                tuple(records.values()),
            )
        if generation_errors:
            source_id, error = generation_errors[0]
            raise RuntimeError(
                "strict fact generation retained all successful checkpoint rows "
                f"but rejected {len(generation_errors)} source(s); "
                f"first={source_id}: {type(error).__name__}: {error}; "
                "resume the checkpoint instead of publishing a fallback"
            ) from error
        completed = tuple(records.values())
        validate_qa_memory_against_sources(
            completed,
            sources,
            require_complete=True,
        )
        source_by_id = {
            source.source_train_task_id: source for source in sources
        }
        strict_paraphrases = tuple(
            record
            for record in completed
            if not _is_dataset_pair_fallback_record(record)
        )
        dataset_pair_fallbacks = tuple(
            record
            for record in completed
            if _is_dataset_pair_fallback_record(record)
        )
        if dataset_pair_fallbacks:
            raise FactProjectionAdmissionError(
                "formal materialization contains dataset-pair fallback records"
            )
        lexical_replacement_count = sum(
            _has_lexical_or_phrase_replacement(
                source_by_id[record.source_train_task_id].original_question,
                record.paraphrase_question,
            )
            for record in strict_paraphrases
        )
        literal_slot_substitution_count = sum(
            _literal_slot_substitution_preserved(
                original_question=(
                    source_by_id[record.source_train_task_id].original_question
                ),
                canonical_answer=record.canonical_answer,
                answer_statement=record.paraphrase_answer_statement,
            )
            for record in strict_paraphrases
        )
        exact_original_question_substring_count = sum(
            " ".join(
                source_by_id[record.source_train_task_id]
                .original_question.split()
            ).casefold()
            in " ".join(record.paraphrase_question.split()).casefold()
            or " ".join(
                source_by_id[record.source_train_task_id]
                .original_question.split()
            ).casefold()
            in " ".join(
                record.paraphrase_answer_statement.split()
            ).casefold()
            for record in completed
        )
        if len(strict_paraphrases) != len(sources):
            raise FactProjectionAdmissionError(
                "strict semantic paraphrase coverage is incomplete"
            )
        if lexical_replacement_count != len(sources):
            raise FactProjectionAdmissionError(
                "every source question must contain a lexical or phrase replacement"
            )
        if exact_original_question_substring_count != 0:
            raise FactProjectionAdmissionError(
                "formal materialization retained an exact original question"
            )
        fact_rows, provenance_rows = build_fact_memory_projection(
            completed,
            sources,
            expected_count=len(sources),
        )
        if len(fact_rows) != len(sources):
            raise FactProjectionAdmissionError(
                "fact_text coverage is incomplete"
            )
        if fact_release_requested and len(fact_rows) != FULL_NATIVE_UNIQUE_QA_COUNT:
            raise FactProjectionAdmissionError(
                "formal full-dataset release must contain 76523 fact_text rows"
            )
        # Checkpoints are written throughout generation, but no final corpus or
        # manifest is published until every strict semantic/fact invariant has
        # closed over the complete source set.
        write_materialized_qa_memory(output_path, completed)
        if fact_release_requested:
            write_jsonl_projection(args.fact_corpus_output, fact_rows)
            write_jsonl_projection(args.provenance_output, provenance_rows)
        semantic_repair_source_ids = tuple(_semantic_repair_source_ids)
        semantic_repair_provenance = "current_resume_partition"
        if predecessor_path is not None:
            predecessor_records, predecessor_rejected_ids = load_resume_records(
                predecessor_path
            )
            if predecessor_rejected_ids:
                raise ValueError(
                    "predecessor materialization contains rejected legacy rows"
                )
            validate_qa_memory_against_sources(
                predecessor_records,
                sources,
                require_complete=True,
            )
            predecessor_by_source = {
                record.source_train_task_id: record
                for record in predecessor_records
            }
            completed_by_source = {
                record.source_train_task_id: record for record in completed
            }
            if set(predecessor_by_source) != set(completed_by_source):
                raise ValueError(
                    "predecessor and completed source identities differ"
                )
            semantic_repair_source_ids = tuple(
                sorted(
                    source_id
                    for source_id, record in completed_by_source.items()
                    if record.memory_id
                    != predecessor_by_source[source_id].memory_id
                )
            )
            semantic_repair_provenance = "predecessor_record_diff"
        manifest_path = (
            Path(args.manifest_output)
            if args.manifest_output
            else output_path.with_name("materialization_manifest.json")
        )
        manifest = {
            "schema_version": "flowsteer.triviaqa.fact_memory.materialization.v1",
            "record_count": len(completed),
            "unique_source_count": len(
                {record.base_task_id for record in completed}
            ),
            "cycled_count": sum(
                record.cycled_training_sample for record in completed
            ),
            "paraphrase_version": args.paraphrase_version,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "paraphrase_method": PARAPHRASE_METHOD,
            "strict_semantic_paraphrase_count": len(strict_paraphrases),
            "dataset_pair_fallback_count": len(dataset_pair_fallbacks),
            "dataset_pair_fallback_method": DATASET_PAIR_FALLBACK_METHOD,
            "dataset_pair_fallback_prompt_template_version": (
                DATASET_PAIR_FALLBACK_PROMPT_VERSION
            ),
            "dataset_pair_fallback_source_ids": sorted(
                record.source_train_task_id
                for record in dataset_pair_fallbacks
            ),
            "bounded_failure_fallback_count": 0,
            "pending_gap_fallback_count": 0,
            "bypass_generation_for_pending": False,
            "fallback_disabled": True,
            "legacy_fallback_regeneration_count": len(
                legacy_fallback_source_ids
            ),
            "lexical_or_phrase_replacement_count": (
                lexical_replacement_count
            ),
            "semantic_verification_required": True,
            "semantic_verification_model": args.model_id,
            "semantic_admission_version": SEMANTIC_ADMISSION_VERSION,
            "semantic_admission_checked_count": len(strict_paraphrases),
            "semantic_repair_count": len(semantic_repair_source_ids),
            "semantic_repair_source_ids": sorted(
                semantic_repair_source_ids
            ),
            "semantic_repair_provenance": semantic_repair_provenance,
            "retained_checkpoint_count": (
                len(completed) - len(semantic_repair_source_ids)
            ),
            "answer_statement_verification_policy": (
                "local_qwen_then_exact_literal_wh_slot_substitution"
            ),
            "exact_literal_slot_substitution_record_count": (
                literal_slot_substitution_count
            ),
            "evaluation_scope": (
                "in_database_transductive"
                if args.include_validation_qa
                else "held_out_generalization"
            ),
            "validation_content_indexed": args.include_validation_qa,
            "exact_original_question_substring_count": (
                exact_original_question_substring_count
            ),
            "fact_memory_schema_version": FACT_MEMORY_SCHEMA_VERSION,
            "fact_text_count": len(fact_rows),
            "fact_projection_fields": [
                "schema_version",
                "memory_id",
                "tool_id",
                "fact_text",
            ],
            "embedding_text_field": "fact_text",
            "original_question_indexed": False,
            "canonical_answer_indexed": False,
            "accepted_answers_indexed": False,
            "paraphrase_question_indexed": False,
            "fact_corpus_output": (
                str(Path(args.fact_corpus_output).resolve())
                if fact_release_requested
                else None
            ),
            "provenance_output": (
                str(Path(args.provenance_output).resolve())
                if fact_release_requested
                else None
            ),
            "generation_concurrency": args.concurrency,
            "generation_rounds": args.generation_rounds,
            "generation_round_start": args.generation_round_start,
            "generation_round_count": args.generation_rounds,
            "generation_round_seed_stride": GENERATION_ROUND_SEED_STRIDE,
            "checkpoint_every": checkpoint_every,
            "seed_paraphrase_reused_count": seed_reused_count,
            "seed_paraphrase_rejected_by_current_admission_count": (
                seed_rejected_by_current_admission_count
            ),
            "original_dataset_binding": dict(original_dataset_binding),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_name(
            f".{manifest_path.name}.partial"
        )
        temporary_manifest.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"TriviaQA QA-memory paraphrase generation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": (
                    "flowsteer.triviaqa.fact_memory.materialization.v1"
                ),
                "output": str(output_path.resolve()),
                "fact_corpus_output": (
                    str(Path(args.fact_corpus_output).resolve())
                    if fact_release_requested
                    else None
                ),
                "provenance_output": (
                    str(Path(args.provenance_output).resolve())
                    if fact_release_requested
                    else None
                ),
                "record_count": len(completed),
                "unique_source_count": len(
                    {record.base_task_id for record in completed}
                ),
                "cycled_count": sum(
                    record.cycled_training_sample for record in completed
                ),
                "paraphrase_version": args.paraphrase_version,
                "generation_round_start": args.generation_round_start,
                "generation_round_count": args.generation_rounds,
                "validation_content_used": args.include_validation_qa,
                "original_dataset_binding": original_dataset_binding,
                "materialization_manifest": str(manifest_path.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
