"""Global HotpotQA declarative-fact embedding index.

The record and ``search``/``read`` boundary follow SkillFlow's
``DocumentPassage`` retrieval contract.  FlowSteer's normalized BGE encoder
and deterministic cosine ranking are reused.  Raw questions, canonical
answers, paraphrased questions, evaluator metadata, and generation receipts
remain outside this runtime index; only a self-contained declarative fact is
embedded and exposed to worker Agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
import json
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import yaml

from scripts.prepare_agentgraph_datasets import _hotpot_records, _path

from .hotpotqa_embedding_index import _encode, _load_sentence_transformer
from .hotpotqa_qa_memory_index import HotpotQATrainQASource
from .task_dataset import qa_question_scope


FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION = (
    "flowsteer.hotpotqa.full_dataset_fact_memory_index.v1"
)
FULL_DATASET_FACT_MEMORY_CORPUS_VERSION = (
    "flowsteer.hotpotqa.native_train_validation_declarative_facts.v1"
)
FULL_DATASET_FACT_DOCUMENT_TEMPLATE = "{fact_text}"
FULL_DATASET_FACT_DOCUMENT_FORMAT = "declarative_fact_only"
FULL_DATASET_FACT_INDEXED_TEXT_FIELD = "fact_text"
FULL_DATASET_EVALUATION_SCOPE = "in_database_transductive"

_MATERIALIZATION_FIELDS = frozenset(
    {
        "source_train_task_id",
        "paraphrase_question",
        "fact_statement",
        "paraphrase_provenance",
        "paraphrase_version",
        "semantic_preservation_attested",
    }
)
_FACT_FIELDS = frozenset({"memory_id", "fact_text"})

# Thin adaptation of TriviaQA's lexical/immutable-field admission boundary.
# These checks stay outside the Agent-facing index and use Q-A only as source
# provenance while deciding whether a generated fact may be projected.
_LEXICAL_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
# NECESSARY_HOTPOT_ADAPTATION: the native corpus contains attached forms such
# as ``a2002`` and equivalent range separators such as ``1980-1991`` versus
# ``1980 to 1991``.  TriviaQA protects the entire surface token exactly; here
# we retain that immutable-field boundary over the ordered numeric atoms so a
# separator or an attached article cannot create a false rejection.  Actual
# changed, added, or removed numbers still fail deterministically.
_ARABIC_NUMBER_ATOM = re.compile(r"(?<!\d)\d[\d,]*(?!\d)")
_DATE_COMMA_SEPARATOR = re.compile(
    r"(?<!\d)(?P<day>\d{1,2}),(?P<year>\d{4})(?!\d)"
)
_YEAR_COMMA_SEPARATOR = re.compile(r"(?P<year>\d{4}),(?=\d{4}\b)")
_ROMAN_NUMERAL_TOKEN = re.compile(r"\b[IVXLCDM]{2,}\b")
_VALID_ROMAN_NUMERAL = re.compile(
    r"M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})"
    r"(?:IX|IV|V?I{0,3})"
)
_WORLD_WAR_ROMAN_TOKEN = re.compile(
    r"\bworld\s+war\s+(?P<roman>[ivxlcdm]+)\b",
    re.IGNORECASE,
)
_WORLD_WAR_ABBREVIATION = re.compile(r"\bWW(?P<roman>I{1,3})\b", re.IGNORECASE)
_QUOTED_SPAN = re.compile(r'"([^\"]+)"|“([^”]+)”|(?<!\w)[\'‘]([^\'’]+)[\'’](?!\w)')
_FINITE_CLAUSE_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|does|do|did|can|could|may|"
    r"might|must|shall|should|will|would)\b",
    re.IGNORECASE,
)
# NECESSARY_HOTPOT_ADAPTATION: a small set of ordinary finite predicates is
# required for upstream answer strings such as ``Founded in 1868, WSU
# consists ...``.  The auxiliary-only predicate boundary inherited from the
# TriviaQA adapter classified those complete propositions as answer labels.
# This remains a sentence-shape signal only; semantic equivalence is still
# checked by the model verifier before a fact is admitted.
_COMMON_FINITE_CLAUSE_PREDICATE = re.compile(
    r"\b(?:became|began|built|consists?|created|died|directed|educated|fell|"
    r"founded|gave|made|played|produced|released|reigned|served|took|won|"
    r"wrote)\b",
    re.IGNORECASE,
)
# DIRECT_REUSE: TriviaQA ``_FACT_ANAPHORIC_SUBJECT``.  A retrieval fact must
# bind its subject inside the indexed sentence; a leading personal or
# demonstrative pronoun otherwise depends on context that is absent from the
# fact-only runtime record.
_FACT_ANAPHORIC_SUBJECT = re.compile(
    r"^(?:it|he|she|they|this|that|these|those)\b",
    re.IGNORECASE,
)
# A fact must bind the answer slot rather than preserve an interrogative
# constituent inside an otherwise punctuated surface.  This rejects generated
# QA concatenations such as ``X was written by which person. A was that
# person.`` while retaining ordinary relative clauses (``A was the person who
# wrote X``) and titles containing an internal question mark.
_FACT_UNBOUND_ANSWER_SLOT = re.compile(
    r"(?:\b(?:by|for|from|to|with|at|in|on|of|as)\s+(?:which|what)\s+"
    r"(?:person|individual|entity|thing|place|city|country|state|year|date|"
    r"number|film|movie|song|book|work|organization|company|team|one)\b|"
    r"\b(?:is|are|was|were|did|does|do|has|have|had|can|could|will|would|"
    r"should)\s+(?:who|whom|where|when|why|how)\b|"
    r"\b(?:is|are|was|were)\s+(?:which|what)\s+"
    r"(?:person|individual|entity|thing|place|city|country|state|year|date|"
    r"number|film|movie|song|book|work|organization|company|team|one)\b)",
    re.IGNORECASE,
)
_FACT_LEADING_INTERROGATIVE = re.compile(
    r"^(?:who|what|which|where|when|why|how|name|identify|"
    r"are|can|could|did|do|does|had|has|have|is|should|"
    r"was|were|will|would)\b",
    re.IGNORECASE,
)
# DIRECT_REUSE: TriviaQA ``_FACT_QA_WRAPPER`` with a necessary HotpotQA
# extension for observed answer/query meta-framing.  The indexed payload must
# state the underlying fact, not describe its role in a QA pair.
_FACT_QA_WRAPPER = re.compile(
    r"(?:\b(?:question|answer|prompt|response)\s*:|"
    r"\b(?:dataset\s+source\s+prompt|paired\s+response|"
    r"corresponding\s+answer|dataset\s+answer|hotpotqa\s+dataset)\b|"
    r"\bthe\s+answer\s+is\b|"
    r"\b(?:the\s+)?(?:answer|subject)\s+(?:of|to)\s+"
    r"(?:the\s+)?(?:question|query|inquiry)\b|"
    r"\b(?:in\s+(?:the\s+)?question|"
    r"(?:referenced|mentioned|described|identified)\s+(?:in|by)\s+"
    r"(?:the\s+)?(?:(?:original|specific)\s+)?(?:question|query|inquiry)|"
    r"(?:target|subject)\s+of\s+(?:the\s+)?"
    r"(?:(?:original|specific)\s+|this\s+specific\s+(?:\w+\s+)?|a\s+)?"
    r"(?:question|query|inquiry)|"
    r"(?:the\s+)?question\s+(?:asks|is\s+asking)|"
    r"(?:context|description)\s+(?:provided\s+)?(?:in|by)\s+"
    r"(?:the\s+)?(?:question|query|inquiry)|"
    r"in\s+(?:the\s+)?context\s+of\s+(?:the\s+)?"
    r"(?:question|query|inquiry)|"
    r"(?:have|has|associated\s+with)\s+(?:the\s+)?answer)\b)",
    re.IGNORECASE,
)
# DIRECT_REUSE: ``prepare_agentgraph_datasets`` annotates this official
# source anomaly by question/answer shape, never by task identity.  A binary
# ``Are/Were A and B both P?`` record with a non-binary official answer is an
# authoritative answer clause/prefix, not a yes/no answer-slot label.
_HOTPOTQA_BINARY_BOTH_QUESTION = re.compile(
    r"^(?:are|were)\s+.+\s+and\s+.+\s+both\s+.+?\?\s*$",
    re.IGNORECASE,
)
_HOTPOTQA_BINARY_ANSWERS = frozenset({"yes", "no"})
_NAMED_WORK_QUESTION = re.compile(
    r"\b(?:what|which)\b[^?]{0,100}\b(?:movie|film|book|novel|song|album|"
    r"series|show|play|game|work|title)\b",
    re.IGNORECASE,
)
_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being",
        "both", "by", "did", "do", "does", "for", "from", "had", "has",
        "have", "he", "her", "hers", "him", "his", "how", "i", "in",
        "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
        "that", "the", "their", "them", "they", "this", "those", "to",
        "was", "we", "were", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "with", "you", "your",
    }
)
_SENTENCE_INITIAL_NON_ENTITY = frozenset(
    {
        "after", "although", "among", "are", "before", "between",
        "because", "did", "does", "do", "during", "either", "following",
        "has", "have", "how", "identify", "is", "name", "neither",
        "since", "state", "was", "were", "what", "when", "where",
        "which", "while", "who", "whom", "whose", "why",
    }
)


def _lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _LEXICAL_TOKEN.finditer(text))


def _content_tokens(text: str) -> Counter[str]:
    return Counter(
        token.casefold()
        for token in _lexical_tokens(text)
        if token.casefold() not in _FUNCTION_WORDS
    )


def _identity_tokens(text: str) -> frozenset[str]:
    tokens = _lexical_tokens(text)
    identities: set[str] = set()
    for index, token in enumerate(tokens):
        folded = token.casefold()
        if folded in _FUNCTION_WORDS:
            continue
        if index == 0 and folded in _SENTENCE_INITIAL_NON_ENTITY:
            continue
        if token[:1].isupper() or token.isupper():
            if folded.endswith(("'s", "’s")):
                folded = folded[:-2]
            identities.add(folded)
    return frozenset(identities)


def _identity_token_preserved(
    required: str,
    observed_tokens: frozenset[str],
) -> bool:
    """DIRECT_REUSE of TriviaQA's case-insensitive identity check."""

    normalized_required = required.replace("’", "'").replace("‘", "'")
    normalized_observed = frozenset(
        token.replace("’", "'").replace("‘", "'")
        for token in observed_tokens
    )
    if normalized_required in normalized_observed:
        return True
    # Hyphenated class/type suffixes and optional spacing around the Hawaiian
    # okina are equivalent proper-name typography, not entity deletion.
    if any(
        token.startswith(f"{normalized_required}-")
        or token.startswith(f"{normalized_required}'")
        or (
            normalized_required.endswith("ʻ")
            and token.startswith(normalized_required)
        )
        for token in normalized_observed
    ):
        return True
    if normalized_required.endswith("'s"):
        return False
    return f"{normalized_required}'s" in normalized_observed


def _observed_content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in _lexical_tokens(text)
        if token.casefold() not in _FUNCTION_WORDS
    )


def _missing_identity_tokens(required_text: str, observed_text: str) -> frozenset[str]:
    observed = _observed_content_tokens(observed_text)
    return frozenset(
        token
        for token in _identity_tokens(required_text)
        if not _identity_token_preserved(token, observed)
    )


def _number_or_date_counts(text: str) -> Counter[str]:
    # NECESSARY_HOTPOT_ADAPTATION: TriviaQA's immutable-number boundary
    # compares numeric atoms rather than typography.  A comma between a
    # one/two-digit day and a four-digit year is a date delimiter, not a
    # thousands separator (``April 13,1979`` == ``April 13, 1979``).
    normalized_text = _DATE_COMMA_SEPARATOR.sub(
        r"\g<day> \g<year>",
        text,
    )
    normalized_text = _YEAR_COMMA_SEPARATOR.sub(
        r"\g<year> ",
        normalized_text,
    )
    values = [
        match.group(0).replace(",", "")
        for match in _ARABIC_NUMBER_ATOM.finditer(normalized_text)
    ]
    world_war_spans: list[tuple[int, int]] = []
    for match in _WORLD_WAR_ROMAN_TOKEN.finditer(text):
        roman = match.group("roman")
        if _VALID_ROMAN_NUMERAL.fullmatch(roman.upper()):
            values.append(roman.casefold())
            world_war_spans.append(match.span("roman"))
    for match in _ROMAN_NUMERAL_TOKEN.finditer(text):
        if any(
            start <= match.start() and match.end() <= end
            for start, end in world_war_spans
        ):
            continue
        roman = match.group(0)
        if _VALID_ROMAN_NUMERAL.fullmatch(roman):
            values.append(roman.casefold())
    values.extend(
        match.group("roman").casefold()
        for match in _WORLD_WAR_ABBREVIATION.finditer(text)
    )
    return Counter(values)


def _number_or_date_keys(text: str) -> frozenset[str]:
    return frozenset(_number_or_date_counts(text))


def _quoted_spans(text: str) -> frozenset[str]:
    return frozenset(
        next(group for group in match.groups() if group is not None).strip()
        for match in _QUOTED_SPAN.finditer(text)
    )


def _contains_ordered_tokens(text: str, required: Sequence[str]) -> bool:
    """Return whether normalized tokens occur in order, allowing insertions."""

    if not required:
        return True
    observed = iter(
        token.casefold().replace("’", "'")
        .replace("‘", "'")
        for token in _lexical_tokens(text)
    )
    for required_token in required:
        folded = (
            required_token.casefold().replace("’", "'").replace("‘", "'")
        )
        if not any(
            token == folded
            or token.startswith(f"{folded}-")
            or token.removesuffix("'s") == folded
            or token.startswith(f"{folded}'")
            or (folded.endswith("ʻ") and token.startswith(folded))
            for token in observed
        ):
            return False
    return True


def _source_confirmed_named_token_sequences(
    source_question: str,
) -> tuple[tuple[str, ...], ...]:
    """Return title-like source spans that may contain interrogative words."""

    matches = tuple(_LEXICAL_TOKEN.finditer(source_question))
    sequences: list[tuple[str, ...]] = []
    run: list[re.Match[str]] = []

    def publish() -> None:
        if len(run) < 3:
            return
        values = tuple(item.group(0) for item in run)
        uppercase_count = sum(token[:1].isupper() for token in values)
        if uppercase_count >= 2:
            sequences.append(tuple(token.casefold() for token in values))

    for match in matches:
        token = match.group(0)
        folded = token.casefold()
        if token[:1].isupper() or folded in _FUNCTION_WORDS:
            run.append(match)
        else:
            publish()
            run = []
    publish()
    return tuple(dict.fromkeys(sequences))


def _unbound_slot_is_inside_source_named_surface(
    source_question: str,
    fact: str,
    slot_match: re.Match[str],
) -> bool:
    """Allow answer-slot words only when nested in a source-confirmed title."""

    fact_matches = tuple(_LEXICAL_TOKEN.finditer(fact))
    fact_tokens = tuple(match.group(0).casefold() for match in fact_matches)
    for sequence in _source_confirmed_named_token_sequences(source_question):
        if len(sequence) > len(fact_tokens):
            continue
        for index in range(len(fact_tokens) - len(sequence) + 1):
            if fact_tokens[index : index + len(sequence)] != sequence:
                continue
            surface_start = fact_matches[index].start()
            surface_end = fact_matches[index + len(sequence) - 1].end()
            if surface_start <= slot_match.start() and slot_match.end() <= surface_end:
                return True
    return False


def _contains_contiguous_lexical_surface(text: str, required_text: str) -> bool:
    """Reject a complete raw source surface even when punctuation changes.

    DIRECT_REUSE + NECESSARY_HOTPOT_ADAPTATION: TriviaQA's exact-question
    shortcut gate rejects a complete source-question substring.  HotpotQA
    generations can preserve that same surface while replacing ``?`` with a
    colon/period and appending the answer, so compare contiguous lexical tokens.
    """

    observed = tuple(token.casefold() for token in _lexical_tokens(text))
    required = tuple(token.casefold() for token in _lexical_tokens(required_text))
    return bool(required) and any(
        observed[index : index + len(required)] == required
        for index in range(len(observed) - len(required) + 1)
    )


def _leading_will_is_source_entity(source_question: str, fact: str) -> bool:
    """Distinguish a source-mentioned ``Will <Surname>`` from auxiliary will."""

    match = re.match(r"^(?P<name>Will\s+[A-Z][^\W_]+)\b", fact)
    if match is None:
        return False
    source_position = source_question.casefold().find(
        match.group("name").casefold()
    )
    if source_position <= 0:
        return False
    remainder = fact[match.end() :].lstrip()
    return bool(
        re.match(
            r"^(?:,|and\b|or\b|is\b|are\b|was\b|were\b|has\b|have\b|"
            r"had\b|[A-Za-z][^\W_]+(?:ed|ing)\b)",
            remainder,
            re.IGNORECASE,
        )
    )


def _leading_source_named_entity(
    source_question: str,
    fact: str,
) -> bool:
    """Recognize source-mentioned titles such as ``Those Calloways``.

    This is deliberately narrower than a generic proper-noun heuristic: the
    leading two-or-more-token title surface must occur verbatim in the source
    question, start with the otherwise-anaphoric token, and be followed in the
    fact by a predicate boundary.  Bare ``It/Those ...`` references remain
    invalid.
    """

    tokens = _lexical_tokens(fact)
    if len(tokens) < 3:
        return False
    for length in range(min(6, len(tokens) - 1), 1, -1):
        candidate = " ".join(tokens[:length])
        if not _contains_contiguous_lexical_surface(source_question, candidate):
            continue
        if not any(token[:1].isupper() for token in tokens[1:length]):
            continue
        remainder_tokens = tokens[length:]
        remainder = " ".join(remainder_tokens)
        if _FINITE_CLAUSE_VERB.search(remainder) or _COMMON_FINITE_CLAUSE_PREDICATE.search(
            remainder
        ):
            return True
    return False


def _source_question_is_bare_canonical_surface(
    source_question: str,
    canonical_answer: str,
) -> bool:
    """Identify upstream rows whose 'question' is only the answer entity."""

    question_tokens = tuple(
        token.casefold() for token in _lexical_tokens(source_question)
    )
    answer_tokens = tuple(
        token.casefold() for token in _lexical_tokens(canonical_answer)
    )
    return bool(question_tokens) and question_tokens == answer_tokens


def canonical_answer_is_declarative_clause(
    answer: str,
    *,
    question: str | None = None,
) -> bool:
    """Use TriviaQA's narrow, source-aware clausal answer-slot boundary.

    DIRECT_REUSE: TriviaQA ``_clausal_canonical_relation_statement`` only
    treats a pronoun-led canonical label as a clause when the source question
    has the exact ``What <slot> did <subject> have?`` relation.  This avoids
    misclassifying title spans such as ``I Knew You Were Trouble`` merely
    because they contain an auxiliary verb.
    """

    normalized = " ".join(_required_text(answer, "canonical_answer").split())
    answer_tokens = _lexical_tokens(normalized)
    normalized_label = normalized.rstrip(" .!?").casefold()
    if normalized_label in _HOTPOTQA_BINARY_ANSWERS:
        return False
    if answer_tokens and answer_tokens[0].casefold() in {
        "am", "are", "be", "been", "being", "can", "could", "did",
        "do", "does", "had", "has", "have", "is", "may", "might",
        "must", "shall", "should", "was", "were", "will", "would",
    }:
        return False
    if (
        question is not None
        and len(answer_tokens) <= 10
        and _NAMED_WORK_QUESTION.search(question) is not None
    ):
        return False
    if question is not None:
        normalized_question = " ".join(
            _required_text(question, "question").split()
        )
        if (
            _HOTPOTQA_BINARY_BOTH_QUESTION.fullmatch(normalized_question)
            is not None
            and normalized.casefold() not in _HOTPOTQA_BINARY_ANSWERS
        ):
            return True
    # NECESSARY_HOTPOT_ADAPTATION: unlike TriviaQA, some HotpotQA canonical
    # answers are complete punctuated sentences.  Terminal-period + finite
    # predicate is a fail-closed sentence signal that does not classify
    # unpunctuated song/film titles containing auxiliaries as propositions.
    has_finite_predicate = (
        _FINITE_CLAUSE_VERB.search(normalized) is not None
        or _COMMON_FINITE_CLAUSE_PREDICATE.search(normalized) is not None
    )
    if (
        len(answer_tokens) >= 3
        and normalized.endswith(".")
        and has_finite_predicate
    ) or (
        len(answer_tokens) >= 12
        and has_finite_predicate
    ):
        return True
    finite_match = _FINITE_CLAUSE_VERB.search(normalized)
    if finite_match is None:
        finite_match = _COMMON_FINITE_CLAUSE_PREDICATE.search(normalized)
    if finite_match is not None and len(answer_tokens) >= 5:
        predicate_token_index = len(_lexical_tokens(normalized[: finite_match.start()]))
        if 0 < predicate_token_index < len(answer_tokens) - 1:
            return True
    if question is None or (
        len(answer_tokens) < 3
        or answer_tokens[0].casefold()
        not in {"he", "i", "it", "she", "they", "we", "you"}
    ):
        return False
    normalized_question = " ".join(_required_text(question, "question").split())
    return re.fullmatch(
        r"(?i:what)\s+(?P<slot>.+?)\s+(?i:did)\s+"
        r"(?P<subject>.+?)\s+(?i:have)\?",
        normalized_question,
    ) is not None


def validate_hotpotqa_question_rewrite(
    source: HotpotQATrainQASource,
    paraphrase_question: object,
) -> str:
    """Apply TriviaQA-style immutable-field and lexical-change admission."""

    question = _required_text(paraphrase_question, "paraphrase_question")
    if _normalized_text(question) == _normalized_text(source.question):
        raise ValueError("paraphrase_question is identical to the source question")
    if _content_tokens(question) == _content_tokens(source.question):
        raise ValueError(
            "paraphrase_question changed only syntax or word order"
        )
    # Entity equivalence is checked by the dedicated semantic verifier.  A
    # capitalization-only heuristic is not an entity recognizer and wrongly
    # rejects valid rewrites such as ``American`` -> ``U.S.``.  Deterministic
    # admission remains strict for numbers/dates and answer leakage.
    if _number_or_date_keys(source.question) != _number_or_date_keys(question):
        raise ValueError(
            "paraphrase_question changed an immutable number or date"
        )
    canonical = " ".join(source.canonical_answer.split())
    if (
        canonical.casefold() not in source.question.casefold()
        and canonical.casefold() in question.casefold()
    ):
        raise ValueError("paraphrase_question introduced the canonical answer")
    return question


def validate_hotpotqa_fact_statement(
    source: HotpotQATrainQASource,
    fact_statement: object,
) -> str:
    """Admit one fact-only payload with no Q-A wire or unsupported entities."""

    fact = " ".join(_required_text(fact_statement, "fact_statement").split())
    if _FACT_QA_WRAPPER.search(fact):
        raise ValueError("fact_statement contains a Question/Answer wire")
    canonical = " ".join(source.canonical_answer.split())
    canonical_is_clause = canonical_answer_is_declarative_clause(
        canonical,
        question=source.question,
    )
    canonical_surface = canonical.rstrip(" .!?")
    structural_fact = fact
    if canonical_surface and not canonical_is_clause:
        structural_fact = re.sub(
            rf"(?<!\w){re.escape(canonical_surface)}(?:[.!?])?(?!\w)",
            "CanonicalEntity",
            structural_fact,
            flags=re.IGNORECASE,
        )
    terminal_surface = fact.rstrip('"\'’”)]} ')
    title_terminal_question = (
        not canonical_is_clause
        and canonical.rstrip().endswith("?")
        and fact.rstrip().casefold().endswith(canonical.rstrip().casefold())
    )
    leading_interrogative = _FACT_LEADING_INTERROGATIVE.match(
        structural_fact
    )
    if (
        (terminal_surface.endswith("?") and not title_terminal_question)
        or (
            leading_interrogative is not None
            and not _leading_will_is_source_entity(
                source.question,
                structural_fact,
            )
            and not _leading_source_named_entity(
                source.question,
                structural_fact,
            )
        )
    ):
        raise ValueError("fact_statement must be declarative")
    if (
        _FACT_ANAPHORIC_SUBJECT.match(structural_fact)
        and not _leading_source_named_entity(
            source.question,
            structural_fact,
        )
    ):
        raise ValueError("fact_statement begins with an unbound anaphoric subject")
    if any(
        not _unbound_slot_is_inside_source_named_surface(
            source.question,
            structural_fact,
            match,
        )
        for match in _FACT_UNBOUND_ANSWER_SLOT.finditer(structural_fact)
    ):
        raise ValueError("fact_statement contains an unbound interrogative answer slot")
    if (
        not fact.rstrip('"\'’”)]} ').endswith((".", "!"))
        and not title_terminal_question
    ):
        raise ValueError("fact_statement must be a complete declarative sentence")
    # Raw source questions are provenance-only.  A yes/no question can look
    # declarative after changing only its terminal punctuation, so use the
    # same ordered lexical boundary as the canonical-answer shortcut gate.
    if (
        _contains_contiguous_lexical_surface(fact, source.question)
        and not _source_question_is_bare_canonical_surface(
            source.question,
            canonical,
        )
    ):
        raise ValueError(
            "fact_statement contains the complete source question lexical surface"
        )

    # DIRECT_REUSE: TriviaQA does not infer a new entity from capitalization
    # in a generated fact.  Its deterministic gate preserves source identity
    # material, while the separate fact semantic verifier rejects unsupported
    # entities.  The removed novelty heuristic falsely rejected source tokens
    # whose case changed at sentence boundaries (``korea`` -> ``Korea``).
    allowed_numbers = _number_or_date_keys(
        f"{source.question} {source.canonical_answer}"
    )
    observed_numbers = _number_or_date_keys(fact)
    if not observed_numbers.issubset(allowed_numbers):
        raise ValueError("fact_statement introduced a number or date")
    # The raw canonical answer is provenance-only.  Even when it already
    # looks like a complete sentence, it cannot be the indexed fact payload;
    # require a distinct semantic realization for every answer type.
    # DIRECT_REUSE + NECESSARY_HOTPOT_ADAPTATION: TriviaQA's
    # ``relation_bearing_answer_statement`` rejects a canonical span that
    # differs only by surrounding punctuation.  HotpotQA generation may also
    # change quote/comma punctuation while retaining the same bare lexical
    # payload, so compare the complete ordered lexical surface as well.
    if (
        _normalized_text(fact) == _normalized_text(canonical)
        or tuple(token.casefold() for token in _lexical_tokens(fact))
        == tuple(token.casefold() for token in _lexical_tokens(canonical))
    ):
        raise ValueError(
            "fact_statement is identical to the canonical answer"
        )
    if canonical_is_clause:
        required_answer_identities = _identity_tokens(canonical)
        if _missing_identity_tokens(canonical, fact):
            raise ValueError(
                "fact_statement removed an immutable entity from the answer clause"
            )
        if not _number_or_date_keys(canonical).issubset(
            _number_or_date_keys(fact)
        ):
            raise ValueError(
                "fact_statement changed a number or date in the answer clause"
            )
    else:
        # Preserve immutable answer material while allowing an ordinary phrase
        # or a binary label to be realized as an equivalent declarative fact.
        # This follows TriviaQA's fact-memory admission: a literal full-span
        # requirement would reject valid transformations such as ``yes`` -> an
        # affirmative proposition and ``16-year-old`` -> ``16 years old``.
        required_answer_identities = _identity_tokens(canonical)
        required_answer_numbers = _number_or_date_counts(canonical)
        required_identity_sequence = tuple(
            dict.fromkeys(
                token.casefold().replace("’", "'")
                for token in _lexical_tokens(canonical)
                if token.casefold() in required_answer_identities
            )
        )
        if (
            len(required_answer_identities) >= 2
            and _missing_identity_tokens(canonical, fact)
        ):
            raise ValueError(
                "fact_statement removed immutable answer tokens"
            )
        if (
            len(required_answer_identities) >= 2
            and required_identity_sequence
            and not _contains_ordered_tokens(fact, required_identity_sequence)
        ):
            raise ValueError(
                "fact_statement changed immutable answer token order"
            )
        observed_fact_numbers = _number_or_date_keys(fact)
        if not frozenset(required_answer_numbers).issubset(observed_fact_numbers):
            raise ValueError(
                "fact_statement removed a number or date from the answer"
            )
    return fact


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _hotpot_config(dataset_catalog_path: Path) -> Mapping[str, object]:
    with dataset_catalog_path.expanduser().resolve().open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    sources = catalog.get("sources") if isinstance(catalog, Mapping) else None
    config = sources.get("hotpotqa") if isinstance(sources, Mapping) else None
    if not isinstance(config, Mapping):
        raise ValueError("dataset catalog has no HotpotQA source")
    if tuple(config.get("candidate_sequence", ())) != ("train", "validation"):
        raise ValueError("HotpotQA native split order differs from train, validation")
    return config


@dataclass(frozen=True, slots=True)
class HotpotQAFullDatasetQASources:
    """Index-external Q-A provenance projected from native HotpotQA splits."""

    train: tuple[HotpotQATrainQASource, ...]
    validation: tuple[HotpotQATrainQASource, ...]
    # Public distractor passages are generation-only recovery evidence for
    # malformed upstream Q/A rows.  They are never projected into the sidecar,
    # embedding index, or Agent Tool wire.
    public_context_by_source_id: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        train_ids = {source.source_train_task_id for source in self.train}
        validation_ids = {
            source.source_train_task_id for source in self.validation
        }
        if len(train_ids) != len(self.train):
            raise ValueError("native HotpotQA train source IDs are not unique")
        if len(validation_ids) != len(self.validation):
            raise ValueError("native HotpotQA validation source IDs are not unique")
        if train_ids & validation_ids:
            raise ValueError("native HotpotQA train/validation source IDs overlap")
        if any(source.cycled for source in self.combined):
            raise ValueError("native full-dataset HotpotQA sources cannot be cycled")
        context_ids = set(self.public_context_by_source_id)
        if context_ids and context_ids != train_ids | validation_ids:
            raise ValueError(
                "public context IDs differ from native HotpotQA source IDs"
            )
        if any(
            not passages or any(not str(passage).strip() for passage in passages)
            for passages in self.public_context_by_source_id.values()
        ):
            raise ValueError("public HotpotQA context contains an empty passage")

    @property
    def combined(self) -> tuple[HotpotQATrainQASource, ...]:
        return self.train + self.validation


def load_hotpotqa_full_dataset_qa_sources(
    *,
    dataset_catalog_path: Path,
    expected_train_count: int = 90_447,
    expected_validation_count: int = 7_405,
) -> HotpotQAFullDatasetQASources:
    """Load raw Q-A only as index-external generation/evaluation provenance."""

    if expected_train_count < 1 or expected_validation_count < 1:
        raise ValueError("full-dataset source counts must be positive")
    config = _hotpot_config(dataset_catalog_path)
    projected: dict[str, list[HotpotQATrainQASource]] = {
        "train": [],
        "validation": [],
    }
    public_context_by_source_id: dict[str, tuple[str, ...]] = {}
    for record in _hotpot_records(config):
        split = record.get("split")
        if split not in projected:
            raise ValueError(f"unexpected native HotpotQA split: {split}")
        task_id = _required_text(record.get("task_id"), "task_id")
        if not task_id.startswith("hotpotqa:"):
            raise ValueError("native HotpotQA task ID is incompatible")
        projected[str(split)].append(
            HotpotQATrainQASource(
                source_train_task_id=task_id,
                base_task_id=task_id,
                cycled=False,
                question=qa_question_scope(
                    _required_text(record.get("question"), "question")
                ),
                canonical_answer=_required_text(
                    record.get("ground_truth"), "ground_truth"
                ),
            )
        )
        public_context = record.get("context")
        if not isinstance(public_context, Sequence) or isinstance(
            public_context, (str, bytes)
        ):
            raise TypeError("native HotpotQA public context must be a sequence")
        passages = tuple(
            _required_text(item, "public_context_passage")
            for item in public_context
        )
        if not passages:
            raise ValueError("native HotpotQA public context cannot be empty")
        public_context_by_source_id[task_id] = passages
    if len(projected["train"]) != expected_train_count:
        raise ValueError(
            f"expected {expected_train_count} native train records, got "
            f"{len(projected['train'])}"
        )
    if len(projected["validation"]) != expected_validation_count:
        raise ValueError(
            f"expected {expected_validation_count} native validation records, got "
            f"{len(projected['validation'])}"
        )
    return HotpotQAFullDatasetQASources(
        train=tuple(projected["train"]),
        validation=tuple(projected["validation"]),
        public_context_by_source_id=MappingProxyType(
            public_context_by_source_id
        ),
    )


@dataclass(frozen=True, slots=True)
class HotpotQADeclarativeFact:
    """SkillFlow-style public passage containing only one declarative fact."""

    memory_id: str
    fact_text: str

    def __post_init__(self) -> None:
        _required_text(self.memory_id, "memory_id")
        fact = _required_text(self.fact_text, "fact_text")
        lowered = fact.casefold()
        if lowered.startswith("question:") or lowered.startswith("answer:"):
            raise ValueError("fact_text cannot use a Question/Answer label")
        if "\nquestion:" in lowered or "\nanswer:" in lowered:
            raise ValueError("fact_text cannot contain a Question/Answer wire")

    def to_value(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: object) -> "HotpotQADeclarativeFact":
        mapping = _mapping(value, "declarative fact")
        if set(mapping) != _FACT_FIELDS:
            raise ValueError("declarative fact fields differ from the fact-only wire")
        return cls(
            memory_id=_required_text(mapping["memory_id"], "memory_id"),
            fact_text=_required_text(mapping["fact_text"], "fact_text"),
        )


@dataclass(frozen=True, slots=True)
class HotpotQADeclarativeFactSearchHit:
    memory_id: str
    fact_snippet: str
    similarity: float
    rank: int


def materialize_hotpotqa_declarative_facts(
    sources: Sequence[HotpotQATrainQASource],
    paraphrases: Sequence[Mapping[str, object]],
) -> tuple[HotpotQADeclarativeFact, ...]:
    """Admit verified sidecar records while projecting only fact text."""

    if len(sources) != len(paraphrases):
        raise ValueError("every source must have exactly one fact materialization")
    facts: list[HotpotQADeclarativeFact] = []
    for index, (source, raw_value) in enumerate(zip(sources, paraphrases)):
        value = _mapping(raw_value, "fact materialization")
        if set(value) != _MATERIALIZATION_FIELDS:
            raise ValueError("fact materialization fields differ")
        if value["source_train_task_id"] != source.source_train_task_id:
            raise ValueError("fact materialization source order or identity differs")
        if value["semantic_preservation_attested"] is not True:
            raise ValueError("fact materialization lacks semantic verification")
        validate_hotpotqa_question_rewrite(
            source, value["paraphrase_question"]
        )
        fact_text = validate_hotpotqa_fact_statement(
            source, value["fact_statement"]
        )
        _required_text(value["paraphrase_provenance"], "paraphrase_provenance")
        _required_text(value["paraphrase_version"], "paraphrase_version")
        facts.append(
            HotpotQADeclarativeFact(
                memory_id=f"hotpotqa-fact-{index:06d}",
                fact_text=fact_text,
            )
        )
    return tuple(facts)


@dataclass(frozen=True, slots=True)
class HotpotQAFullDatasetFactMemoryIndexManifest:
    schema_version: str
    index_id: str
    corpus_version: str
    source: str
    source_splits: tuple[str, ...]
    embedding_model: str
    embedding_model_path: str
    embedding_dimension: int
    normalized: bool
    similarity: str
    frozen_top_k: int
    source_record_count: int
    source_train_count: int
    source_validation_count: int
    unique_source_count: int
    cycled_record_count: int
    question_rewrite_count: int
    fact_count: int
    semantic_rewrite_coverage: float
    frozen_evaluation_count: int
    evaluation_overlap_count: int
    contains_evaluation_source_facts: bool
    contains_raw_questions: bool
    contains_raw_answers: bool
    evaluation_scope: str
    official_heldout_eligible: bool
    paraphrase_versions: tuple[str, ...]
    paraphrase_provenances: tuple[str, ...]
    document_template: str
    document_format: str
    indexed_text_field: str
    source_dataset_catalog_path: str
    source_train_path: str
    source_validation_path: str
    facts_path: str
    embeddings_path: str

    def __post_init__(self) -> None:
        if self.schema_version != FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported full-dataset fact-memory index schema")
        if self.corpus_version != FULL_DATASET_FACT_MEMORY_CORPUS_VERSION:
            raise ValueError("unsupported full-dataset fact-memory corpus schema")
        if self.source_splits != ("train", "validation"):
            raise ValueError("full-dataset fact-memory source splits differ")
        if self.embedding_dimension < 1 or self.frozen_top_k < 1:
            raise ValueError("embedding dimension and top-k must be positive")
        if not self.normalized or self.similarity != "cosine":
            raise ValueError("full-dataset fact-memory requires normalized cosine")
        if self.source_record_count != (
            self.source_train_count + self.source_validation_count
        ):
            raise ValueError("full-dataset source counts are inconsistent")
        if self.unique_source_count != self.source_record_count:
            raise ValueError("every native HotpotQA source ID must be unique")
        if self.cycled_record_count != 0:
            raise ValueError("native full-dataset sources cannot be cycled")
        if self.question_rewrite_count != self.source_record_count:
            raise ValueError("every source question must be rewritten")
        if self.fact_count != self.source_record_count:
            raise ValueError("every source must have one declarative fact")
        if self.semantic_rewrite_coverage != 1.0:
            raise ValueError("semantic rewrite coverage must be exactly 100%")
        if self.frozen_evaluation_count < 1:
            raise ValueError("frozen evaluation identity count must be positive")
        if self.evaluation_overlap_count != self.frozen_evaluation_count:
            raise ValueError("every frozen evaluation source fact must be present")
        if self.contains_evaluation_source_facts is not True:
            raise ValueError("transductive source facts must be declared")
        if self.contains_raw_questions or self.contains_raw_answers:
            raise ValueError("raw Q-A cannot enter the fact index")
        if self.evaluation_scope != FULL_DATASET_EVALUATION_SCOPE:
            raise ValueError("full-dataset evaluation scope differs")
        if self.official_heldout_eligible is not False:
            raise ValueError("in-database fact retrieval is not held-out eligible")
        if not self.paraphrase_versions or not self.paraphrase_provenances:
            raise ValueError("paraphrase version and provenance are required")
        if self.document_template != FULL_DATASET_FACT_DOCUMENT_TEMPLATE:
            raise ValueError("fact index document template differs")
        if self.document_format != FULL_DATASET_FACT_DOCUMENT_FORMAT:
            raise ValueError("fact index document format differs")
        if self.indexed_text_field != FULL_DATASET_FACT_INDEXED_TEXT_FIELD:
            raise ValueError("fact index text field differs")

    @property
    def train_record_count(self) -> int:
        """Compatibility count consumed by the shared retrieval adapter."""

        return self.fact_count

    def to_value(self) -> dict[str, object]:
        value = asdict(self)
        value["source_splits"] = list(self.source_splits)
        value["paraphrase_versions"] = list(self.paraphrase_versions)
        value["paraphrase_provenances"] = list(self.paraphrase_provenances)
        return value

    @classmethod
    def from_value(
        cls, value: object
    ) -> "HotpotQAFullDatasetFactMemoryIndexManifest":
        mapping = _mapping(value, "full-dataset fact-memory manifest")
        expected = frozenset(cls.__dataclass_fields__)
        if set(mapping) != expected:
            raise ValueError("full-dataset fact-memory manifest fields differ")
        fields = {name: mapping[name] for name in expected}
        fields["source_splits"] = tuple(str(item) for item in fields["source_splits"])
        fields["paraphrase_versions"] = tuple(
            str(item) for item in fields["paraphrase_versions"]
        )
        fields["paraphrase_provenances"] = tuple(
            str(item) for item in fields["paraphrase_provenances"]
        )
        return cls(**fields)  # type: ignore[arg-type]


def _native_source_paths(dataset_catalog_path: Path) -> tuple[str, str]:
    config = _hotpot_config(dataset_catalog_path)
    files = _mapping(config.get("files"), "HotpotQA files")
    base = _path(str(config["path"]))
    return str(base / str(files["train"])), str(base / str(files["validation"]))


def build_hotpotqa_full_dataset_fact_memory_index(
    *,
    index_dir: Path,
    dataset_catalog_path: Path,
    frozen_evaluation_task_ids: Sequence[str],
    paraphrases: Sequence[Mapping[str, object]],
    embedding_model_path: str,
    embedding_model_id: str,
    embedding_device: str,
    frozen_top_k: int,
    expected_train_count: int = 90_447,
    expected_validation_count: int = 7_405,
) -> HotpotQAFullDatasetFactMemoryIndexManifest:
    """Build one global normalized-cosine index over declarative facts only."""

    if frozen_top_k < 1:
        raise ValueError("frozen_top_k must be positive")
    sources = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=dataset_catalog_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    source_ids = {source.source_train_task_id for source in sources.combined}
    evaluation_ids = {
        _required_text(item, "evaluation task ID")
        for item in frozen_evaluation_task_ids
    }
    if len(evaluation_ids) != len(frozen_evaluation_task_ids):
        raise ValueError("frozen evaluation task IDs are not unique")
    overlap_count = len(source_ids & evaluation_ids)
    if overlap_count != len(evaluation_ids):
        raise ValueError("some frozen evaluation sources are absent")

    facts = materialize_hotpotqa_declarative_facts(sources.combined, paraphrases)
    model = _load_sentence_transformer(embedding_model_path, embedding_device)
    # Required boundary: only the self-contained fact text is vectorized.
    vectors = _encode(model, [fact.fact_text for fact in facts])

    index_dir = index_dir.expanduser().resolve()
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError("full-dataset fact-memory index directory must be empty")
    index_dir.mkdir(parents=True, exist_ok=True)
    facts_path = index_dir / "facts.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"
    facts_path.write_text(
        "".join(
            json.dumps(fact.to_value(), ensure_ascii=False, sort_keys=True) + "\n"
            for fact in facts
        ),
        encoding="utf-8",
    )
    np.save(embeddings_path, vectors, allow_pickle=False)
    source_train_path, source_validation_path = _native_source_paths(
        dataset_catalog_path
    )
    paraphrase_versions = tuple(
        sorted({_required_text(item["paraphrase_version"], "paraphrase_version") for item in paraphrases})
    )
    paraphrase_provenances = tuple(
        sorted({_required_text(item["paraphrase_provenance"], "paraphrase_provenance") for item in paraphrases})
    )
    manifest = HotpotQAFullDatasetFactMemoryIndexManifest(
        schema_version=FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION,
        index_id=(
            "hotpotqa-full-dataset-fact-memory-"
            f"d{vectors.shape[1]}-n{len(facts)}-topk{frozen_top_k}-v1"
        ),
        corpus_version=FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
        source="HotpotQA generated self-contained declarative facts",
        source_splits=("train", "validation"),
        embedding_model=embedding_model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimension=int(vectors.shape[1]),
        normalized=True,
        similarity="cosine",
        frozen_top_k=frozen_top_k,
        source_record_count=len(facts),
        source_train_count=len(sources.train),
        source_validation_count=len(sources.validation),
        unique_source_count=len(sources.combined),
        cycled_record_count=0,
        question_rewrite_count=len(facts),
        fact_count=len(facts),
        semantic_rewrite_coverage=1.0,
        frozen_evaluation_count=len(evaluation_ids),
        evaluation_overlap_count=overlap_count,
        contains_evaluation_source_facts=True,
        contains_raw_questions=False,
        contains_raw_answers=False,
        evaluation_scope=FULL_DATASET_EVALUATION_SCOPE,
        official_heldout_eligible=False,
        paraphrase_versions=paraphrase_versions,
        paraphrase_provenances=paraphrase_provenances,
        document_template=FULL_DATASET_FACT_DOCUMENT_TEMPLATE,
        document_format=FULL_DATASET_FACT_DOCUMENT_FORMAT,
        indexed_text_field=FULL_DATASET_FACT_INDEXED_TEXT_FIELD,
        source_dataset_catalog_path=str(dataset_catalog_path.expanduser().resolve()),
        source_train_path=source_train_path,
        source_validation_path=source_validation_path,
        facts_path=facts_path.name,
        embeddings_path=embeddings_path.name,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_value(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class HotpotQAFullDatasetFactMemoryIndex:
    """Read-only global embedding index with SkillFlow-style search/read."""

    def __init__(
        self,
        *,
        manifest: HotpotQAFullDatasetFactMemoryIndexManifest,
        facts: Sequence[HotpotQADeclarativeFact],
        embeddings: np.ndarray,
        model: object,
    ) -> None:
        if embeddings.shape != (len(facts), manifest.embedding_dimension):
            raise ValueError("embedding matrix does not match fact-memory manifest")
        if len(facts) != manifest.fact_count:
            raise ValueError("fact count does not match fact-memory manifest")
        memory_index = {fact.memory_id: index for index, fact in enumerate(facts)}
        if len(memory_index) != len(facts):
            raise ValueError("fact memory IDs are not unique")
        self.manifest = manifest
        self._facts = tuple(facts)
        self._memory_index = MappingProxyType(memory_index)
        self._embeddings = embeddings
        self._model = model
        self._encode_lock = Lock()

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        embedding_model_path: str | None = None,
        embedding_device: str = "cpu",
    ) -> "HotpotQAFullDatasetFactMemoryIndex":
        index_dir = index_dir.expanduser().resolve()
        manifest = HotpotQAFullDatasetFactMemoryIndexManifest.from_value(
            json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        )
        facts = tuple(
            HotpotQADeclarativeFact.from_value(json.loads(line))
            for line in (index_dir / manifest.facts_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        embeddings = np.load(index_dir / manifest.embeddings_path, allow_pickle=False)
        model = _load_sentence_transformer(
            embedding_model_path or manifest.embedding_model_path,
            embedding_device,
        )
        return cls(
            manifest=manifest,
            facts=facts,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model=model,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        query = _required_text(query, "retrieval query")
        with self._encode_lock:
            return _encode(self._model, [query], batch_size=1)[0]

    async def search(
        self, query: str, k: int
    ) -> tuple[HotpotQADeclarativeFactSearchHit, ...]:
        if k != self.manifest.frozen_top_k:
            raise ValueError("search k differs from the frozen fact-memory top-k")
        # The shared runtime invokes Tool calls under an async boundary, but
        # sentence-transformer query encoding itself is synchronous.  Keep it
        # on the calling thread so the per-task Tool lifecycle owns no orphaned
        # default-executor thread at shutdown.
        query_vector = self._encode_query(query)
        # DIRECT_REUSE: SkillFlow/TriviaQA fact-memory computes one dense
        # matrix-vector product and then applies the same deterministic
        # score/memory_id ordering.  Calling np.dot once per record adds a
        # Python loop over the full 97,852-record corpus without changing the
        # retrieval semantics.
        scores = np.asarray(self._embeddings @ query_vector, dtype=np.float32)
        order = sorted(
            range(len(self._facts)),
            key=lambda index: (
                -float(scores[index]),
                self._facts[index].memory_id,
            ),
        )[:k]
        hits: list[HotpotQADeclarativeFactSearchHit] = []
        for rank, index in enumerate(order, start=1):
            fact = self._facts[index]
            snippet = fact.fact_text
            if len(snippet) > 320:
                snippet = f"{snippet[:320]}…"
            hits.append(
                HotpotQADeclarativeFactSearchHit(
                    memory_id=fact.memory_id,
                    fact_snippet=snippet,
                    similarity=float(scores[index]),
                    rank=rank,
                )
            )
        return tuple(hits)

    def read(self, memory_id: str) -> HotpotQADeclarativeFact:
        try:
            return self._facts[self._memory_index[memory_id]]
        except KeyError as exc:
            raise KeyError("memory_id is absent from the fact index") from exc


__all__ = [
    "FULL_DATASET_EVALUATION_SCOPE",
    "FULL_DATASET_FACT_DOCUMENT_FORMAT",
    "FULL_DATASET_FACT_DOCUMENT_TEMPLATE",
    "FULL_DATASET_FACT_INDEXED_TEXT_FIELD",
    "FULL_DATASET_FACT_MEMORY_CORPUS_VERSION",
    "FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION",
    "HotpotQADeclarativeFact",
    "HotpotQADeclarativeFactSearchHit",
    "HotpotQAFullDatasetFactMemoryIndex",
    "HotpotQAFullDatasetFactMemoryIndexManifest",
    "HotpotQAFullDatasetQASources",
    "build_hotpotqa_full_dataset_fact_memory_index",
    "load_hotpotqa_full_dataset_qa_sources",
    "materialize_hotpotqa_declarative_facts",
    "canonical_answer_is_declarative_clause",
    "validate_hotpotqa_fact_statement",
    "validate_hotpotqa_question_rewrite",
]
