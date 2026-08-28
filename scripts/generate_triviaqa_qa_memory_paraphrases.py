#!/usr/bin/env python3
"""Materialize semantic-preserving TriviaQA native-train Q-A paraphrases.

Source binding is checked before the local Qwen3.5 OpenAI-compatible endpoint
is contacted.  The model generates a reworded question and a relation-bearing
declarative answer statement from each frozen canonical Q-A pair.  Depending
on the explicit CLI protocol, fixed evaluation Q-A records are either excluded
for split-isolated retrieval or retained for the in-database protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    r"(?<!\w)'([^']+)'(?!\w)|(?<!\w)‘([^’]+)’(?!\w)"
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
    if re.search(r"\bborn\b", original_question, re.IGNORECASE) is not None:
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
    if _contains_source_spelling_correction(
        original_question,
        paraphrase_question,
    ) or _contains_source_spelling_correction(
        original_question,
        paraphrase_answer_statement,
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

    if required in observed_tokens:
        return True
    if required.endswith(("'s", "’s")):
        return False
    return f"{required}'s" in observed_tokens or f"{required}’s" in observed_tokens


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
    question = " ".join(original_question.split()).casefold()
    canonical = " ".join(canonical_answer.split()).casefold()
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
    tokens = _LEXICAL_TOKEN.findall(question)
    return frozenset(
        token.casefold()
        for index, token in enumerate(tokens)
        if (
            (
                token[:1].isupper()
                and token.casefold() not in _FUNCTION_WORDS
                and not (
                    index == 0
                    and _LEADING_INTERROGATIVE_CONTRACTION.fullmatch(token)
                    is not None
                )
                and not (
                    index == 0
                    and token.casefold()
                    in {
                        "identify",
                        "name",
                        "state",
                        "give",
                        "specify",
                        "tell",
                    }
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
    if (
        len(normalized) >= 2
        and normalized.startswith('"')
        and normalized.endswith('"')
        and '""' in normalized
    ):
        normalized = normalized[1:-1].replace('""', '"')
    spans = [
        next(group for group in match.groups() if group is not None).strip()
        for match in _DOUBLE_QUOTED_SPAN.finditer(normalized)
    ]
    spans.extend(
        next(group for group in match.groups() if group is not None).strip()
        for match in _SINGLE_QUOTED_SPAN.finditer(normalized)
    )
    return frozenset(span for span in spans if span)


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


def build_paraphrase_messages(source: TriviaQATrainSource) -> list[dict[str, str]]:
    """Build the bounded model request without accepted aliases."""

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
        raise ValueError(f"{response_name} fields are incompatible")
    normalized: dict[str, object] = {}
    for raw_field, field_value in value.items():
        if not isinstance(raw_field, str):
            raise ValueError(f"{response_name} fields are incompatible")
        normalized_field = _OBSERVED_RESPONSE_KEY_TYPOS.get(
            raw_field,
            raw_field,
        )
        if (
            normalized_field not in expected_fields
            or normalized_field in normalized
        ):
            raise ValueError(f"{response_name} fields are incompatible")
        normalized[normalized_field] = field_value
    if frozenset(normalized) != expected_fields:
        raise ValueError(f"{response_name} fields are incompatible")
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

    original = " ".join(source.original_question.split()).casefold()
    question = " ".join(paraphrase_question.split()).casefold()
    statement = " ".join(paraphrase_answer_statement.split()).casefold()
    if original in question or original in statement:
        raise ValueError(
            "semantic paraphrase retained the complete original question substring"
        )


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

    value = _decode_paraphrase_object(text)
    question = value["paraphrase_question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("paraphrase_question must be non-empty text")
    question = " ".join(question.split())
    if question.casefold() == " ".join(source.original_question.split()).casefold():
        raise ValueError("model returned the original question unchanged")
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
        raise ValueError(
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
    # Run the new semantic boundary last.  Resume partitioning is therefore
    # allowed to repair only an otherwise valid row; every older deterministic
    # admission failure remains fail-closed.
    _semantic_relation_and_scope_preserved(
        original_question=source.original_question,
        paraphrase_question=question,
        paraphrase_answer_statement=statement,
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
    """Re-run the current deterministic v7 boundary on every checkpoint row."""

    source_by_id = {
        source.source_train_task_id: source for source in sources
    }
    for record in records:
        source = source_by_id.get(record.source_train_task_id)
        if source is None:
            raise ValueError("resume row references a non-train source")
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
    """Partition only newly detected semantic drift into regeneration.

    Catching the dedicated exception, rather than ``ValueError``, is the
    explicit repair boundary: malformed JSON/schema, source mismatches,
    answer leakage, provenance errors, and all older admission failures abort
    resume instead of being silently converted into new model work.
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
        try:
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
        except SemanticPreservationError:
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
        raise ValueError(
            "question repair exhausted eligible lexical source tokens"
        ) from last_error

    def generate(
        self,
        source: TriviaQATrainSource,
        *,
        seed: int,
    ) -> tuple[str, str, int]:
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
                    answer_error = str(parse_error).startswith(
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
                    if (
                        not answer_error
                        and not question_lexical_error
                        and not quoted_question_error
                        and not source_transposition_error
                        and not answer_slot_family_error
                    ):
                        raise
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
                    if answer_slot_family_error:
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
                        statement = _restore_authoritative_source_transpositions(
                            source.original_question,
                            statement,
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
                        quoted_repair = _quoted_attribution_qa(source)
                        if quoted_repair is None:
                            raise
                        question, statement = quoted_repair
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
        help="Independent deterministic seed rounds for records rejected after retries.",
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
            admitted_seeds = {
                args.base_seed
                + record.selection_index
                + generation_round * GENERATION_ROUND_SEED_STRIDE
                + retry
                for generation_round in range(args.generation_rounds)
                for retry in range(args.max_retries + 1)
            }
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
        existing, _semantic_repair_source_ids = (
            partition_resume_records_for_semantic_repair(existing, sources)
        )
        for record in existing:
            source = source_by_id[record.source_train_task_id]
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
            for generation_round in range(args.generation_rounds):
                try:
                    paraphrase, answer_statement, accepted_seed = client.generate(
                        source,
                        seed=(
                            args.base_seed
                            + source.selection_index
                            + generation_round * GENERATION_ROUND_SEED_STRIDE
                        ),
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
                "bounded paraphrase generation retained all successful rows but "
                f"rejected {len(generation_errors)} source(s); first={source_id}: "
                f"{type(error).__name__}: {error}"
            ) from error
        completed = tuple(records.values())
        validate_qa_memory_against_sources(
            completed,
            sources,
            require_complete=True,
        )
        write_materialized_qa_memory(output_path, completed)
        source_by_id = {
            source.source_train_task_id: source for source in sources
        }
        lexical_replacement_count = sum(
            _has_lexical_or_phrase_replacement(
                source_by_id[record.source_train_task_id].original_question,
                record.paraphrase_question,
            )
            for record in completed
        )
        literal_slot_substitution_count = sum(
            _literal_slot_substitution_preserved(
                original_question=(
                    source_by_id[record.source_train_task_id].original_question
                ),
                canonical_answer=record.canonical_answer,
                answer_statement=record.paraphrase_answer_statement,
            )
            for record in completed
        )
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
            "schema_version": (
                "flowsteer.triviaqa.qa_memory.materialization.v2"
            ),
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
            "lexical_or_phrase_replacement_count": (
                lexical_replacement_count
            ),
            "semantic_verification_required": True,
            "semantic_verification_model": args.model_id,
            "semantic_admission_version": SEMANTIC_ADMISSION_VERSION,
            "semantic_admission_checked_count": len(completed),
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
            "exact_original_question_substring_count": 0,
            "generation_concurrency": args.concurrency,
            "generation_rounds": args.generation_rounds,
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
                "schema_version": "flowsteer.triviaqa.qa_memory.materialization.v2",
                "output": str(output_path.resolve()),
                "record_count": len(completed),
                "unique_source_count": len(
                    {record.base_task_id for record in completed}
                ),
                "cycled_count": sum(
                    record.cycled_training_sample for record in completed
                ),
                "paraphrase_version": args.paraphrase_version,
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
