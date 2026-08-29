#!/usr/bin/env python3
"""Materialize verified HotpotQA declarative facts with local Qwen3.5-9B.

Every native question is semantically reworded and paired with one
self-contained declarative fact.  The JSONL is an index-external provenance
sidecar; the index builder projects only ``fact_statement`` into the runtime
corpus.  There is deliberately no verbatim or dataset-pair fallback.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_hotpotqa_qa_memory import (  # noqa: E402
    _generate_json,
    _json_schema,
    _read_jsonl,
    _write_jsonl,
)
from src.interactive.config_loader import load_model_registry  # noqa: E402
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (  # noqa: E402
    _ARABIC_NUMBER_ATOM,
    _FINITE_CLAUSE_VERB,
    _HOTPOTQA_BINARY_ANSWERS,
    _ROMAN_NUMERAL_TOKEN,
    _WORLD_WAR_ABBREVIATION,
    _content_tokens,
    _identity_tokens,
    _lexical_tokens,
    _quoted_spans,
    FULL_DATASET_EVALUATION_SCOPE,
    canonical_answer_is_declarative_clause,
    load_hotpotqa_full_dataset_qa_sources,
    materialize_hotpotqa_declarative_facts,
    validate_hotpotqa_fact_statement,
    validate_hotpotqa_question_rewrite,
)
from src.interactive.hotpotqa_qa_memory_index import (  # noqa: E402
    HotpotQATrainQASource,
)


PROMPT_VERSION = "hotpotqa.full_dataset_fact.qwen35.field_repair.v12"
PARAPHRASE_VERSION = "hotpotqa-full-dataset-declarative-fact-v12"
PARAPHRASE_PROVENANCE = (
    "local-qwen3.5-9b-semantic-rewrite-and-field-verification-v12"
)
GENERATION_ROUND_SEED_STRIDE = 100_000_000

FACT_GENERATION_SCHEMA = _json_schema(
    {
        "paraphrase_question": {"type": "string", "minLength": 1},
        "fact_statement": {"type": "string", "minLength": 1},
    }
)
QUESTION_REPAIR_SCHEMA = _json_schema(
    {
        "paraphrase_question": {"type": "string", "minLength": 1},
        "replaced_source_token": {"type": "string", "minLength": 1},
        "replacement_phrase": {"type": "string", "minLength": 1},
    }
)
SYNONYM_REPAIR_SCHEMA = _json_schema(
    {
        "source_token": {"type": "string", "minLength": 1},
        "replacement_phrase": {"type": "string", "minLength": 1},
    }
)
FACT_REPAIR_SCHEMA = _json_schema(
    {"fact_statement": {"type": "string", "minLength": 1}}
)
QUESTION_VERIFICATION_SCHEMA = _json_schema(
    {
        "semantic_preserved": {"type": "boolean"},
        "question_changed": {"type": "boolean"},
        "constraints_preserved": {"type": "boolean"},
        "answer_slot_preserved": {"type": "boolean"},
        "answer_not_revealed": {"type": "boolean"},
    }
)
FACT_VERIFICATION_SCHEMA = _json_schema(
    {
        "fact_declarative": {"type": "boolean"},
        "fact_self_contained": {"type": "boolean"},
        "fact_supported_by_qa": {"type": "boolean"},
        "canonical_span_preserved_when_required": {"type": "boolean"},
        "no_qa_wire_format": {"type": "boolean"},
        "answer_slot_bound": {"type": "boolean"},
        "relation_direction_preserved": {"type": "boolean"},
        "no_new_fact_or_relation": {"type": "boolean"},
    }
)
CLAUSE_FACT_VERIFICATION_SCHEMA = _json_schema(
    {
        "semantically_equivalent_to_answer_clause": {"type": "boolean"},
        "fact_declarative": {"type": "boolean"},
        "fact_self_contained": {"type": "boolean"},
        "canonical_identities_preserved": {"type": "boolean"},
        "canonical_numbers_preserved": {"type": "boolean"},
        "no_qa_wire_format": {"type": "boolean"},
        "no_new_fact": {"type": "boolean"},
    }
)
BINARY_FACT_VERIFICATION_SCHEMA = _json_schema(
    {
        "fact_declarative": {"type": "boolean"},
        "fact_self_contained": {"type": "boolean"},
        "source_proposition_preserved": {"type": "boolean"},
        "binary_polarity_preserved": {"type": "boolean"},
        "quantifier_scope_preserved": {"type": "boolean"},
        "no_qa_wire_format": {"type": "boolean"},
        "no_new_fact_or_relation": {"type": "boolean"},
    }
)
_REQUIRED_QUESTION_VERIFICATION_FIELDS = tuple(
    QUESTION_VERIFICATION_SCHEMA["required"]
)
_REQUIRED_FACT_VERIFICATION_FIELDS = tuple(
    FACT_VERIFICATION_SCHEMA["required"]
)
_REQUIRED_CLAUSE_FACT_VERIFICATION_FIELDS = tuple(
    CLAUSE_FACT_VERIFICATION_SCHEMA["required"]
)
_REQUIRED_BINARY_FACT_VERIFICATION_FIELDS = tuple(
    BINARY_FACT_VERIFICATION_SCHEMA["required"]
)
_RELATION_REBUILD_REJECTION_MARKERS = (
    "fact_supported_by_qa",
    "answer_slot_bound",
    "relation_direction_preserved",
    "no_new_fact_or_relation",
    "fact_declarative",
    "fact_self_contained",
    "no_qa_wire_format",
    "canonical_span_preserved_when_required",
    "contains the complete source question lexical surface",
    "is identical to the canonical answer",
    "contains a question/answer wire",
    "must be declarative",
    "must be a complete declarative sentence",
    "begins with an unbound anaphoric subject",
)
_ANSWER_RECONSTRUCTION_PATTERNS = (
    "literal_answer_slot_substitution",
    "relative_clause_binding",
    "fronted_canonical_span",
    "clausal_statement_paraphrase",
)
_QUESTION_DIRECT_SYNONYM_REJECTION_MARKERS = (
    "changed an immutable number or date",
    "introduced the canonical answer",
)
_CLAUSE_SYNONYM_REJECTION_MARKERS = (
    "is identical to the canonical answer",
    "changed a number or date in the answer clause",
    "removed a number or date from the answer",
    "introduced a number or date",
    "must be a complete declarative sentence",
    "removed an immutable entity from the answer clause",
)
_COMMON_FINITE_PREDICATES = frozenset(
    {
        "became",
        "began",
        "built",
        "created",
        "died",
        "fell",
        "founded",
        "gave",
        "made",
        "played",
        "produced",
        "released",
        "served",
        "took",
        "won",
        "wrote",
    }
)


class FactMaterializationRejected(RuntimeError):
    """Bounded strict-generation rejection with complete attempt receipts."""

    def __init__(
        self,
        source_id: str,
        attempt_receipts: Sequence[Mapping[str, object]],
    ) -> None:
        self.source_id = source_id
        self.attempt_receipts = tuple(dict(item) for item in attempt_receipts)
        last = self.attempt_receipts[-1] if self.attempt_receipts else {}
        detail = str(last.get("error", "strict field verification exhausted"))
        super().__init__(
            f"full-dataset fact materialization failed for {source_id}: {detail}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _generated_text(generated: Mapping[str, object], field: str) -> str:
    value = generated.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return " ".join(value.split())


def _immutable_number_or_date_surfaces(text: str) -> tuple[str, ...]:
    """Project the exact surfaces protected by deterministic admission."""

    values = [
        match.group(0).rstrip(",")
        for match in _ARABIC_NUMBER_ATOM.finditer(text)
    ]
    values.extend(
        match.group(0) for match in _ROMAN_NUMERAL_TOKEN.finditer(text)
    )
    values.extend(
        match.group(0) for match in _WORLD_WAR_ABBREVIATION.finditer(text)
    )
    return tuple(dict.fromkeys(values))


def _immutable_identity_surfaces(text: str) -> tuple[str, ...]:
    """Render HotpotQA's normalized identity boundary as source surfaces."""

    identities = _identity_tokens(text)
    values: list[str] = []
    for token in _lexical_tokens(text):
        normalized = token.casefold()
        if normalized.endswith(("'s", "’s")):
            normalized = normalized[:-2]
        if normalized in identities:
            values.append(token)
    return tuple(dict.fromkeys(values))


def _question_immutable_payload(
    source: HotpotQATrainQASource,
) -> dict[str, object]:
    """Thin adaptation of TriviaQA's immutable-field request payload."""

    canonical_tokens = set(_content_tokens(source.canonical_answer))
    question_tokens = set(_content_tokens(source.question))
    return {
        "original_question": source.question,
        "immutable_original_entity_tokens": list(
            _immutable_identity_surfaces(source.question)
        ),
        "immutable_number_or_date_tokens": list(
            _immutable_number_or_date_surfaces(source.question)
        ),
        "immutable_quoted_spans": sorted(_quoted_spans(source.question)),
        "forbidden_question_canonical_tokens": sorted(
            canonical_tokens - question_tokens
        ),
    }


def _fact_binding_payload(
    source: HotpotQATrainQASource,
    *,
    binding_mode: str,
) -> dict[str, object]:
    """Build TriviaQA-calibrated answer-slot binding metadata."""

    return {
        "original_question": source.question,
        "canonical_training_answer": source.canonical_answer,
        "fact_binding_mode": binding_mode,
        "immutable_answer_entity_tokens": list(
            _immutable_identity_surfaces(source.canonical_answer)
        ),
        "immutable_answer_number_or_date_tokens": list(
            _immutable_number_or_date_surfaces(source.canonical_answer)
        ),
        "allowed_fact_number_or_date_tokens": list(
            _immutable_number_or_date_surfaces(
                f"{source.question} {source.canonical_answer}"
            )
        ),
        "immutable_answer_quoted_spans": sorted(
            _quoted_spans(source.canonical_answer)
        ),
    }


def _problem_payload(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _lexical_replacement_source_tokens(
    source: HotpotQATrainQASource,
) -> tuple[str, ...]:
    """Thin adaptation of TriviaQA's required-source-token repair boundary."""

    content_tokens = set(_content_tokens(source.question))
    immutable_identities = {
        token.casefold().removesuffix("'s").removesuffix("’s")
        for token in _immutable_identity_surfaces(source.question)
    }
    quoted_tokens = {
        token.casefold()
        for span in _quoted_spans(source.question)
        for token in _lexical_tokens(span)
    }
    candidates: list[str] = []
    for token in _lexical_tokens(source.question):
        normalized = token.casefold()
        identity_base = normalized.removesuffix("'s").removesuffix("’s")
        if (
            normalized in content_tokens
            and identity_base not in immutable_identities
            and normalized not in quoted_tokens
            and not any(character.isdigit() for character in normalized)
        ):
            candidates.append(token)
    if not candidates:
        relation_auxiliaries = {
            "are",
            "did",
            "do",
            "does",
            "had",
            "has",
            "have",
            "is",
            "was",
            "were",
        }
        candidates.extend(
            token
            for token in _lexical_tokens(source.question)
            if token.casefold() in relation_auxiliaries
        )
    return tuple(dict.fromkeys(candidates))


def _clause_replacement_source_tokens(
    source: HotpotQATrainQASource,
) -> tuple[str, ...]:
    """Select one non-identity, non-number clause token for synonym repair."""

    canonical = source.canonical_answer
    content_tokens = set(_content_tokens(canonical))
    immutable_identities = {
        token.casefold().removesuffix("'s").removesuffix("’s")
        for token in _immutable_identity_surfaces(canonical)
    }
    candidates: list[str] = []
    for token in _lexical_tokens(canonical):
        normalized = token.casefold()
        identity_base = normalized.removesuffix("'s").removesuffix("’s")
        if (
            normalized in content_tokens
            and identity_base not in immutable_identities
            and not any(character.isdigit() for character in normalized)
        ):
            candidates.append(token)
    return tuple(dict.fromkeys(candidates))


def _question_requires_direct_synonym(prior_rejection: str) -> bool:
    normalized = prior_rejection.casefold()
    return any(
        marker in normalized
        for marker in _QUESTION_DIRECT_SYNONYM_REJECTION_MARKERS
    )


def _clause_requires_synonym_repair(prior_rejection: str) -> bool:
    normalized = prior_rejection.casefold()
    return any(
        marker in normalized
        for marker in _CLAUSE_SYNONYM_REJECTION_MARKERS
    )


def _complete_declarative_punctuation(text: str) -> str:
    """DIRECT_REUSE of TriviaQA's terminal punctuation surface repair."""

    normalized = " ".join(text.split()).strip()
    match = re.fullmatch(
        r"(?P<body>.*?)(?P<closers>[\"'’”)\]}]*)",
        normalized,
    )
    assert match is not None
    body = match.group("body").rstrip()
    closers = match.group("closers")
    if body.endswith((".", "!", "?")):
        return normalized
    if body.endswith((",", ";", ":")):
        body = body[:-1].rstrip()
    return f"{body}{closers}."


def _repair_missing_terminal_punctuation(fact: str) -> str | None:
    """Append punctuation only when the existing surface is clause-complete."""

    normalized = " ".join(fact.split()).strip()
    if not normalized or normalized.rstrip('"\'’”)]} ').endswith(
        (".", "!", "?")
    ):
        return None
    tokens = _lexical_tokens(normalized)
    if len(tokens) < 3:
        return None
    identities = _identity_tokens(normalized)
    predicate_tokens = [
        token.casefold()
        for token in tokens
        if token.casefold().removesuffix("'s").removesuffix("’s")
        not in identities
    ]
    has_predicate = (
        _FINITE_CLAUSE_VERB.search(normalized) is not None
        or any(token in _COMMON_FINITE_PREDICATES for token in predicate_tokens)
        or any(
            len(token) > 3 and token.endswith(("ed", "ing"))
            for token in predicate_tokens
        )
    )
    if not has_predicate:
        return None
    return _complete_declarative_punctuation(normalized)


def _parse_question_repair(
    generated: Mapping[str, object],
    *,
    eligible_source_tokens: Sequence[str],
    required_source_token: str,
) -> str:
    """Validate the structured lexical-replacement receipt before admission."""

    question = _generated_text(generated, "paraphrase_question")
    replaced = _generated_text(generated, "replaced_source_token")
    replacement = _generated_text(generated, "replacement_phrase")
    eligible = {token.casefold() for token in eligible_source_tokens}
    if replaced.casefold() not in eligible:
        raise ValueError("question repair replaced_source_token is ineligible")
    if replaced.casefold() != required_source_token.casefold():
        raise ValueError("question repair replaced the wrong source token")
    if replacement.casefold() not in question.casefold():
        raise ValueError("question repair replacement_phrase is absent")
    if replaced.casefold() == replacement.casefold():
        raise ValueError("question repair did not change lexical wording")
    return question


def _parse_synonym_repair(
    generated: Mapping[str, object],
    *,
    required_source_token: str,
) -> str:
    """DIRECT_REUSE of TriviaQA's synonym-only repair receipt boundary."""

    source_token = _generated_text(generated, "source_token")
    replacement = _generated_text(generated, "replacement_phrase")
    if source_token.casefold() != required_source_token.casefold():
        raise ValueError("synonym repair source token is incompatible")
    if replacement.casefold() == required_source_token.casefold():
        raise ValueError("synonym repair did not change lexical wording")
    return replacement


def _replace_source_token_once(
    source_question: str,
    *,
    source_token: str,
    replacement_phrase: str,
) -> str:
    """Apply one boundary-safe replacement to the authoritative question."""

    pattern = re.compile(
        rf"(?<!\w){re.escape(source_token)}(?!\w)",
        re.IGNORECASE,
    )
    candidate, count = pattern.subn(
        replacement_phrase,
        source_question,
        count=1,
    )
    if count != 1 or candidate.casefold() == source_question.casefold():
        raise ValueError("synonym repair did not rewrite the source question")
    return " ".join(candidate.split())


def _answer_reconstruction_patterns(binding_mode: str) -> tuple[str, ...]:
    if binding_mode == "declarative_clause_paraphrase":
        return (
            "clausal_statement_paraphrase",
            "literal_answer_slot_substitution",
            "relative_clause_binding",
            "fronted_canonical_span",
        )
    return _ANSWER_RECONSTRUCTION_PATTERNS


def _answer_reconstruction_instruction(pattern: str) -> str:
    instructions = {
        "literal_answer_slot_substitution": (
            "Copy the source relation and replace only the complete wh-constituent "
            "with the canonical answer semantics, then convert interrogative "
            "punctuation and word order to one declarative sentence."
        ),
        "relative_clause_binding": (
            "Use a relation-bearing relative clause: 'The <answer type> that "
            "<source predicate> is/was <canonical answer semantics>.' Preserve the "
            "source predicate, arguments, tense, scope, and constraints."
        ),
        "fronted_canonical_span": (
            "When the canonical span is prepositional or possessive, keep that span "
            "intact at the front and render the remaining source subject and "
            "predicate in declarative order; otherwise keep it intact as the "
            "answer-slot complement."
        ),
        "clausal_statement_paraphrase": (
            "When the canonical answer is already a clause, paraphrase that clause "
            "as one self-contained proposition; resolve only a leading pronoun to "
            "the explicit source subject and add no other relation."
        ),
    }
    try:
        return instructions[pattern]
    except KeyError as exc:
        raise ValueError("unknown answer reconstruction pattern") from exc


def _fact_repair_strategy(prior_rejection: str) -> str:
    normalized = prior_rejection.casefold()
    if any(
        marker in normalized
        for marker in _RELATION_REBUILD_REJECTION_MARKERS
    ):
        return "authoritative_answer_slot_reconstruction"
    return "preserve_and_repair_immutable_fields"


def _binary_answer_label(answer: str) -> str | None:
    normalized = " ".join(answer.split()).rstrip(" .!?").casefold()
    return normalized if normalized in _HOTPOTQA_BINARY_ANSWERS else None


def _candidate(
    source: HotpotQATrainQASource,
    generated: Mapping[str, object],
) -> dict[str, object]:
    question = validate_hotpotqa_question_rewrite(
        source, generated.get("paraphrase_question")
    )
    normalized_fact = validate_hotpotqa_fact_statement(
        source, generated.get("fact_statement")
    )
    return {
        "source_train_task_id": source.source_train_task_id,
        "paraphrase_question": question.strip(),
        "fact_statement": normalized_fact,
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_version": PARAPHRASE_VERSION,
        "semantic_preservation_attested": True,
    }


async def _materialize_one(
    source: HotpotQATrainQASource,
    *,
    index: int,
    model: object,
    provider: object,
    seed: int,
    max_attempts: int,
    generation_rounds: int = 2,
) -> tuple[dict[str, object], dict[str, object]]:
    if generation_rounds < 1:
        raise ValueError("generation_rounds must be positive")
    binary_answer = _binary_answer_label(source.canonical_answer)
    binding_mode = (
        "binary_polarity_binding"
        if binary_answer is not None
        else (
            "declarative_clause_paraphrase"
            if canonical_answer_is_declarative_clause(
                source.canonical_answer,
                question=source.question,
            )
            else "answer_slot_binding"
        )
    )
    question_source_payload = _question_immutable_payload(source)
    fact_source_payload = _fact_binding_payload(
        source,
        binding_mode=binding_mode,
    )
    question: str | None = None
    fact: str | None = None
    question_admitted = False
    fact_admitted = False
    question_rejection = "not yet generated"
    fact_rejection = "not yet generated"
    question_repair_count = 0
    fact_reconstruction_count = 0
    clause_repair_count = 0
    fact_candidate_count = 0
    fact_candidate_keys: set[str] = set()
    force_fact_reconstruction = False
    attempt_receipts: list[dict[str, object]] = []

    for generation_round in range(generation_rounds):
        for attempt in range(max_attempts):
            request_seed = (
                seed
                + index * max_attempts * 8
                + generation_round * GENERATION_ROUND_SEED_STRIDE
                + attempt * 8
            )
            trace: dict[str, object] = {
                "generation_round": generation_round + 1,
                "attempt": attempt + 1,
                "fact_binding_mode": binding_mode,
                "question": {"preserved_from_prior_attempt": question_admitted},
                "fact": {"preserved_from_prior_attempt": fact_admitted},
            }
            question_trace = trace["question"]
            fact_trace = trace["fact"]
            assert isinstance(question_trace, dict)
            assert isinstance(fact_trace, dict)

            if question is None and fact is None:
                try:
                    generated, generation_receipt = await _generate_json(
                        model=model,
                        provider=provider,
                        schema=FACT_GENERATION_SCHEMA,
                        contract=(
                            "Semantically reword the HotpotQA question while "
                            "preserving every entity, relation, scope, constraint, "
                            "multi-hop path, answer slot, name, number, date, and "
                            "quoted span. Replace real wording, not only word order. "
                            + (
                                "The dataset answer is already a declarative clause; "
                                "write a semantically equivalent self-contained "
                                "declarative fact. If the clause begins with an "
                                "unbound pronoun, replace only that pronoun with the "
                                "explicit subject already named in the source question. "
                                "Use the question for no other inference and do not "
                                "invent a relation between a possibly mismatched "
                                "question and answer. "
                                if binding_mode == "declarative_clause_paraphrase"
                                else
                                (
                                    "The canonical answer is a binary label. Convert "
                                    "the complete source proposition into a "
                                    "self-contained declarative fact with exactly the "
                                    "same polarity and scope: yes affirms it; no gives "
                                    "its scope-preserving negation. In particular, "
                                    "'not both P' must not become 'neither is P'. "
                                    if binding_mode == "binary_polarity_binding"
                                    else
                                    "Bind the dataset answer semantics to the original "
                                    "question's answer slot in a self-contained "
                                    "declarative fact. Preserve proper names, numbers, "
                                    "dates, and quoted titles; ordinary phrases may use "
                                    "equivalent wording. Preserve relation direction, "
                                    "polarity, and scope. "
                                )
                            )
                            + "Every pronoun or demonstrative must have an explicit "
                            "antecedent inside the same fact. State the fact itself; "
                            "do not describe an answer, question, query, or inquiry. "
                            "Copy every supplied immutable field exactly and keep "
                            "every forbidden canonical token out of the paraphrased "
                            "question. For answer-slot binding, retain the source "
                            "predicate, arguments, direction, polarity, and scope; "
                            "substitute the answer semantics only into the requested "
                            "slot. State the supported relation once rather than "
                            "restating the question or appending a bare answer. Do "
                            "not add entities, aliases, facts, or Q-A labels. "
                            "Return only the requested JSON fields."
                        ),
                        problem=_problem_payload(
                            {
                                **question_source_payload,
                                **fact_source_payload,
                            }
                        ),
                        request_id=(
                            f"hotpotqa-full-dataset-fact:{index:06d}:"
                            f"round:{generation_round:02d}:generate:{attempt:02d}"
                        ),
                        seed=request_seed,
                        temperature=0.1,
                    )
                    question = _generated_text(
                        generated, "paraphrase_question"
                    )
                    fact = _generated_text(generated, "fact_statement")
                    fact_candidate_key = fact.casefold()
                    fact_candidate_repeated = (
                        fact_candidate_key in fact_candidate_keys
                    )
                    fact_candidate_count += 1
                    fact_candidate_keys.add(fact_candidate_key)
                    force_fact_reconstruction = fact_candidate_repeated
                    fact_trace["candidate_repeated"] = fact_candidate_repeated
                    fact_trace["candidate_number"] = fact_candidate_count
                    trace["joint_generation_response"] = dict(
                        generation_receipt
                    )
                except Exception as exc:
                    trace["error"] = (
                        f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                    )
                    attempt_receipts.append(trace)
                    continue
            else:
                if not question_admitted:
                    try:
                        eligible_source_tokens = (
                            _lexical_replacement_source_tokens(source)
                        )
                        if not eligible_source_tokens:
                            raise ValueError(
                                "question repair has no eligible lexical source token"
                            )
                        required_source_token = eligible_source_tokens[
                            question_repair_count % len(eligible_source_tokens)
                        ]
                        question_repair_count += 1
                        direct_synonym_repair = (
                            _question_requires_direct_synonym(
                                question_rejection
                            )
                        )
                        try:
                            if direct_synonym_repair:
                                raise ValueError(
                                    "whole-question repair skipped for deterministic "
                                    "immutable-field rejection"
                                )
                            repaired, response = await _generate_json(
                                model=model,
                                provider=provider,
                                schema=QUESTION_REPAIR_SCHEMA,
                                contract=(
                                    "Repair only the HotpotQA question paraphrase. "
                                    "Preserve every entity, relation, scope, constraint, "
                                    "multi-hop path, answer slot, name, number, date, "
                                    "and quoted span. Replace at least one non-entity "
                                    "word or phrase; changing only word order is invalid. "
                                    "Treat the immutable fields as exact dataset strings; "
                                    "never correct, replace, delete, or complete them from "
                                    "world knowledge. Do not use any forbidden canonical "
                                    "token or reveal the answer. Replace the supplied "
                                    "required_source_token_to_replace, or a phrase that "
                                    "contains it, with a genuine equivalent expression. "
                                    "Report that exact source token and the replacement "
                                    "phrase in the structured response. Return only JSON."
                                ),
                                problem=_problem_payload(
                                    {
                                        **question_source_payload,
                                        "rejected_question": question or "",
                                        "prior_admission_result": question_rejection,
                                        "lexical_replacement_source_tokens": list(
                                            eligible_source_tokens
                                        ),
                                        "required_source_token_to_replace": (
                                            required_source_token
                                        ),
                                    }
                                ),
                                request_id=(
                                    f"hotpotqa-full-dataset-fact:{index:06d}:"
                                    f"round:{generation_round:02d}:"
                                    f"repair-question:{attempt:02d}"
                                ),
                                seed=request_seed,
                                temperature=0.0,
                            )
                            question = _parse_question_repair(
                                repaired,
                                eligible_source_tokens=eligible_source_tokens,
                                required_source_token=required_source_token,
                            )
                            question_trace["repair_mode"] = "structured_rewrite"
                        except Exception as structured_exc:
                            question_trace["structured_repair_skipped"] = (
                                direct_synonym_repair
                            )
                            question_trace["structured_repair_error"] = (
                                f"{type(structured_exc).__name__}: "
                                f"{' '.join(str(structured_exc).split())}"
                            )
                            synonym, response = await _generate_json(
                                model=model,
                                provider=provider,
                                schema=SYNONYM_REPAIR_SCHEMA,
                                contract=(
                                    "Produce only one context-appropriate synonym or "
                                    "equivalent phrase for required_source_token as it "
                                    "is used in original_question. Preserve part of "
                                    "speech and meaning. Do not rewrite the question, "
                                    "name the answer, replace an immutable field, or "
                                    "return the same token. Return only JSON."
                                ),
                                problem=_problem_payload(
                                    {
                                        **question_source_payload,
                                        "required_source_token": (
                                            required_source_token
                                        ),
                                    }
                                ),
                                request_id=(
                                    f"hotpotqa-full-dataset-fact:{index:06d}:"
                                    f"round:{generation_round:02d}:"
                                    f"repair-question-synonym:{attempt:02d}"
                                ),
                                seed=request_seed + 4,
                                temperature=0.0,
                            )
                            replacement = _parse_synonym_repair(
                                synonym,
                                required_source_token=required_source_token,
                            )
                            question = _replace_source_token_once(
                                source.question,
                                source_token=required_source_token,
                                replacement_phrase=replacement,
                            )
                            question_trace["repair_mode"] = "synonym_only"
                        question_trace["generation_response"] = dict(response)
                        question_trace["required_source_token_to_replace"] = (
                            required_source_token
                        )
                    except Exception as exc:
                        question_trace["generation_error"] = (
                            f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                        )
                        question_rejection = str(
                            question_trace["generation_error"]
                        )
                if not fact_admitted:
                    try:
                        terminal_punctuation_repair = (
                            _repair_missing_terminal_punctuation(fact or "")
                            if "must be a complete declarative sentence"
                            in fact_rejection.casefold()
                            else None
                        )
                        clause_synonym_repair = (
                            binding_mode == "declarative_clause_paraphrase"
                            and _clause_requires_synonym_repair(
                                fact_rejection
                            )
                            and terminal_punctuation_repair is None
                        )
                        fact_repair_strategy = (
                            "terminal_punctuation_only"
                            if terminal_punctuation_repair is not None
                            else (
                                "binary_polarity_reconstruction"
                                if binding_mode == "binary_polarity_binding"
                                else (
                                "clause_synonym_only"
                                if clause_synonym_repair
                                else (
                                    "authoritative_answer_slot_reconstruction"
                                    if force_fact_reconstruction
                                    else _fact_repair_strategy(fact_rejection)
                                )
                                )
                            )
                        )
                        reconstruct_from_source = (
                            fact_repair_strategy in {
                                "authoritative_answer_slot_reconstruction",
                                "binary_polarity_reconstruction",
                            }
                        )
                        reconstruction_pattern: str | None = None
                        if (
                            reconstruct_from_source
                            and binding_mode != "binary_polarity_binding"
                        ):
                            reconstruction_patterns = (
                                _answer_reconstruction_patterns(binding_mode)
                            )
                            reconstruction_pattern = reconstruction_patterns[
                                fact_reconstruction_count
                                % len(reconstruction_patterns)
                            ]
                            fact_reconstruction_count += 1
                        if terminal_punctuation_repair is not None:
                            fact = terminal_punctuation_repair
                            response = {
                                "local_surface_repair": (
                                    "terminal_punctuation_only"
                                )
                            }
                            fact_repair_temperature = 0.0
                        elif clause_synonym_repair:
                            clause_source_tokens = (
                                _clause_replacement_source_tokens(source)
                            )
                            if not clause_source_tokens:
                                raise ValueError(
                                    "clause repair has no eligible synonym token"
                                )
                            required_clause_token = clause_source_tokens[
                                clause_repair_count % len(clause_source_tokens)
                            ]
                            clause_repair_count += 1
                            synonym, response = await _generate_json(
                                model=model,
                                provider=provider,
                                schema=SYNONYM_REPAIR_SCHEMA,
                                contract=(
                                    "Produce only one context-appropriate synonym or "
                                    "equivalent phrase for required_source_token in the "
                                    "authoritative canonical answer clause. Preserve "
                                    "part of speech and semantics. Do not rewrite the "
                                    "clause, alter an entity or number/date, add a fact, "
                                    "or return the same token. Return only JSON."
                                ),
                                problem=_problem_payload(
                                    {
                                        **fact_source_payload,
                                        "required_source_token": (
                                            required_clause_token
                                        ),
                                        "eligible_clause_source_tokens": list(
                                            clause_source_tokens
                                        ),
                                        "prior_admission_result": fact_rejection,
                                    }
                                ),
                                request_id=(
                                    f"hotpotqa-full-dataset-fact:{index:06d}:"
                                    f"round:{generation_round:02d}:"
                                    f"repair-fact-clause-synonym:{attempt:02d}"
                                ),
                                seed=request_seed + 1,
                                temperature=0.0,
                            )
                            replacement = _parse_synonym_repair(
                                synonym,
                                required_source_token=required_clause_token,
                            )
                            fact = _replace_source_token_once(
                                source.canonical_answer,
                                source_token=required_clause_token,
                                replacement_phrase=replacement,
                            )
                            fact = _complete_declarative_punctuation(fact)
                            fact_repair_temperature = 0.0
                            fact_trace["required_clause_source_token"] = (
                                required_clause_token
                            )
                        else:
                            fact_repair_temperature = (
                                0.0
                                if binding_mode in {
                                    "declarative_clause_paraphrase",
                                    "binary_polarity_binding",
                                }
                                else 0.1
                            )
                            repaired, response = await _generate_json(
                                model=model,
                                provider=provider,
                                schema=FACT_REPAIR_SCHEMA,
                                contract=(
                                    "Repair only one self-contained declarative fact. "
                                    + (
                                        "The dataset answer is a complete declarative "
                                        "clause. Semantically paraphrase that clause; "
                                        "resolve only a leading pronoun to the explicit "
                                        "source subject and add no other relation. "
                                        if binding_mode
                                        == "declarative_clause_paraphrase"
                                        else
                                        (
                                            "The canonical answer is a binary label. "
                                            "Reconstruct the complete source "
                                            "proposition with the exact binary "
                                            "polarity: yes affirms it and no gives its "
                                            "scope-preserving negation. Do not require "
                                            "the literal word yes or no. Preserve the "
                                            "difference between 'not both' and "
                                            "'neither'. "
                                            if binding_mode
                                            == "binary_polarity_binding"
                                            else
                                            "Bind the dataset answer semantics to the "
                                            "question's original answer slot. Preserve "
                                            "proper names, numbers, dates, quoted titles, "
                                            "relation direction, polarity, and scope. "
                                        )
                                    )
                                    + "Every pronoun or demonstrative must have an "
                                    "explicit antecedent. State one supported fact, "
                                    "not a Q-A label or repeated relation. "
                                    + (
                                        "Construct it only from original_question and "
                                        "canonical_training_answer. Do not imitate the "
                                        "rejected fact. "
                                        + (
                                            "Render only the proposition and its "
                                            "polarity; do not append a binary label. "
                                            if binding_mode
                                            == "binary_polarity_binding"
                                            else _answer_reconstruction_instruction(
                                                reconstruction_pattern
                                            )
                                            + " "
                                        )
                                        if reconstruct_from_source
                                        else
                                        "Preserve correct material while repairing only "
                                        "the reported immutable-field failure. "
                                    )
                                    + "Return only JSON."
                                ),
                                problem=_problem_payload(
                                    {
                                        **fact_source_payload,
                                        **(
                                            {}
                                            if reconstruct_from_source
                                            else {"rejected_fact": fact or ""}
                                        ),
                                        "fact_repair_strategy": (
                                            fact_repair_strategy
                                        ),
                                        "answer_reconstruction_pattern": (
                                            reconstruction_pattern
                                        ),
                                        "prior_admission_result": fact_rejection,
                                    }
                                ),
                                request_id=(
                                    f"hotpotqa-full-dataset-fact:{index:06d}:"
                                    f"round:{generation_round:02d}:"
                                    f"repair-fact:{attempt:02d}"
                                ),
                                seed=request_seed + 1,
                                temperature=fact_repair_temperature,
                            )
                            fact = _generated_text(repaired, "fact_statement")
                        fact_candidate_key = fact.casefold()
                        fact_candidate_repeated = (
                            fact_candidate_key in fact_candidate_keys
                        )
                        fact_candidate_count += 1
                        fact_candidate_keys.add(fact_candidate_key)
                        force_fact_reconstruction = fact_candidate_repeated
                        fact_trace["candidate_repeated"] = (
                            fact_candidate_repeated
                        )
                        fact_trace["candidate_number"] = fact_candidate_count
                        fact_trace["generation_response"] = dict(response)
                        fact_trace["repair_strategy"] = fact_repair_strategy
                        fact_trace["repair_temperature"] = (
                            fact_repair_temperature
                        )
                        fact_trace["answer_reconstruction_pattern"] = (
                            reconstruction_pattern
                        )
                    except Exception as exc:
                        fact_trace["generation_error"] = (
                            f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                        )
                        fact_rejection = str(fact_trace["generation_error"])

            if not question_admitted and question is not None:
                try:
                    question = validate_hotpotqa_question_rewrite(
                        source, question
                    )
                    question_trace["deterministic_admission"] = True
                    verified, response = await _generate_json(
                        model=model,
                        provider=provider,
                        schema=QUESTION_VERIFICATION_SCHEMA,
                        contract=(
                            "Verify only semantic equivalence of the question "
                            "paraphrase. Preserve entity identity, relation, scope, "
                            "constraints, multi-hop path, answer slot, and answer "
                            "cardinality. Surface wording must genuinely change. "
                            "Treat the source as an authoritative dataset string; do "
                            "not correct it from world knowledge. Canonical-answer "
                            "leakage and immutable numbers/dates are also checked "
                            "deterministically. Do not solve the question. Evaluate each boolean "
                            "independently and return only JSON."
                        ),
                        problem=_problem_payload(
                            {
                                "original_question": source.question,
                                "paraphrased_question": question,
                                "canonical_answer_leakage_checked_deterministically": True,
                                "immutable_number_or_date_tokens": question_source_payload[
                                    "immutable_number_or_date_tokens"
                                ],
                            }
                        ),
                        request_id=(
                            f"hotpotqa-full-dataset-fact:{index:06d}:"
                            f"round:{generation_round:02d}:"
                            f"verify-question:{attempt:02d}"
                        ),
                        seed=request_seed + 2,
                        temperature=0.0,
                    )
                    failed = [
                        name
                        for name in _REQUIRED_QUESTION_VERIFICATION_FIELDS
                        if verified.get(name) is not True
                    ]
                    question_trace["verification"] = dict(verified)
                    question_trace["verification_response"] = dict(response)
                    question_trace["failed_fields"] = failed
                    question_admitted = not failed
                    question_rejection = (
                        "accepted"
                        if question_admitted
                        else "semantic verifier rejected: " + ",".join(failed)
                    )
                except Exception as exc:
                    question_admitted = False
                    question_trace["deterministic_or_verification_error"] = (
                        f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                    )
                    question_rejection = str(
                        question_trace["deterministic_or_verification_error"]
                    )

            if not fact_admitted and fact is not None:
                try:
                    fact = validate_hotpotqa_fact_statement(source, fact)
                    fact_trace["deterministic_admission"] = True
                    clause_verification = (
                        binding_mode == "declarative_clause_paraphrase"
                    )
                    binary_verification = (
                        binding_mode == "binary_polarity_binding"
                    )
                    fact_verification_schema = (
                        CLAUSE_FACT_VERIFICATION_SCHEMA
                        if clause_verification
                        else (
                            BINARY_FACT_VERIFICATION_SCHEMA
                            if binary_verification
                            else FACT_VERIFICATION_SCHEMA
                        )
                    )
                    required_fact_verification_fields = (
                        _REQUIRED_CLAUSE_FACT_VERIFICATION_FIELDS
                        if clause_verification
                        else (
                            _REQUIRED_BINARY_FACT_VERIFICATION_FIELDS
                            if binary_verification
                            else _REQUIRED_FACT_VERIFICATION_FIELDS
                        )
                    )
                    fact_verification_contract = (
                        "Verify only whether the proposed fact is a semantic "
                        "paraphrase of canonical_training_answer, which is already "
                        "a declarative answer clause. It must be declarative and "
                        "self-contained, preserve every canonical identity and "
                        "number/date, contain no Q-A wire, and add no fact. A leading "
                        "pronoun may be resolved only to the explicit subject in "
                        "original_question; otherwise do not infer from the question. "
                        "Do not require answer-slot binding, source-question relation "
                        "direction, or support beyond equivalence to the authoritative "
                        "answer clause. Do not fact-check source strings. Evaluate "
                        "every boolean independently and return only JSON."
                        if clause_verification
                        else (
                        "Verify a binary-polarity proposition. The canonical answer "
                        "is a label: yes means the complete source proposition is "
                        "affirmed; no means its scope-preserving negation. A correct "
                        "negative proposition need not contain the literal token "
                        "'no'. Treat original_question plus its authoritative binary "
                        "label as the complete source of support; do not fact-check "
                        "it from world knowledge. Set source_proposition_preserved "
                        "true only when all entities, predicates, comparison axes, "
                        "and constraints remain intact. Set binary_polarity_preserved "
                        "true only for the label's exact polarity. Preserve quantifier "
                        "scope: 'not both P' is not equivalent to 'neither is P'. "
                        "Reject any claim assigning which individual fails unless "
                        "the source proposition and label establish that assignment. "
                        "The fact must also be declarative, self-contained, free of "
                        "Q-A wire, and add no fact or relation. Evaluate every "
                        "boolean independently and return only JSON."
                        if binary_verification
                        else
                        "Verify only the proposed answer-slot fact. It must be "
                        "self-contained, declarative, free of Q-A labels, and "
                        "supported by binding the canonical answer semantics to the "
                        "source question's answer slot without changing relation "
                        "direction, polarity, scope, or constraints. Preserve proper "
                        "names, numbers, dates, and quoted titles; reject unresolved "
                        "anaphora, meta-framing, added facts, a bare answer, or a "
                        "question restatement with an appended answer. Treat source "
                        "strings as authoritative and return only JSON."
                        )
                    )
                    verified, response = await _generate_json(
                        model=model,
                        provider=provider,
                        schema=fact_verification_schema,
                        contract=fact_verification_contract,
                        problem=_problem_payload(
                            {
                                **fact_source_payload,
                                "declarative_fact": fact,
                            }
                        ),
                        request_id=(
                            f"hotpotqa-full-dataset-fact:{index:06d}:"
                            f"round:{generation_round:02d}:"
                            f"verify-fact:{attempt:02d}"
                        ),
                        seed=request_seed + 3,
                        temperature=0.0,
                    )
                    failed = [
                        name
                        for name in required_fact_verification_fields
                        if verified.get(name) is not True
                    ]
                    fact_trace["verification_mode"] = (
                        "answer_clause"
                        if clause_verification
                        else (
                            "binary_polarity"
                            if binary_verification
                            else "answer_slot"
                        )
                    )
                    fact_trace["verification"] = dict(verified)
                    fact_trace["verification_response"] = dict(response)
                    fact_trace["failed_fields"] = failed
                    fact_admitted = not failed
                    fact_rejection = (
                        "accepted"
                        if fact_admitted
                        else "fact verifier rejected: " + ",".join(failed)
                    )
                except Exception as exc:
                    fact_admitted = False
                    fact_trace["deterministic_or_verification_error"] = (
                        f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                    )
                    fact_rejection = str(
                        fact_trace["deterministic_or_verification_error"]
                    )

            trace["candidate"] = {
                "paraphrase_question": question,
                "fact_statement": fact,
            }
            trace["error"] = (
                "question="
                + ("accepted" if question_admitted else "rejected")
                + ",fact="
                + ("accepted" if fact_admitted else "rejected")
            )
            attempt_receipts.append(trace)
            if question_admitted and fact_admitted:
                candidate = _candidate(
                    source,
                    {
                        "paraphrase_question": question,
                        "fact_statement": fact,
                    },
                )
                materialize_hotpotqa_declarative_facts(
                    (source,), (candidate,)
                )
                return candidate, {
                    "source_train_task_id": source.source_train_task_id,
                    "status": "accepted",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "generation_round": generation_round + 1,
                    "attempt": attempt + 1,
                    "generation_seed": request_seed,
                    "fact_binding_mode": binding_mode,
                    "attempt_receipts": attempt_receipts,
                    "completed_at": _utc_now(),
                }

        if generation_round + 1 < generation_rounds:
            # A generation round is an independent bounded proposal, not an
            # alias for more repairs of the same rejected surface.  Preserve
            # the complete receipt history, but reset both fields so the next
            # round performs a fresh joint semantic rewrite under a new seed.
            question = None
            fact = None
            question_admitted = False
            fact_admitted = False
            question_rejection = "not-yet-validated"
            fact_rejection = "not-yet-validated"
            force_fact_reconstruction = False

    raise FactMaterializationRejected(
        source.source_train_task_id, attempt_receipts
    )


async def materialize(args: argparse.Namespace) -> dict[str, object]:
    source_bundle = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=Path(args.dataset_catalog),
        expected_train_count=args.train_count,
        expected_validation_count=args.validation_count,
    )
    sources = source_bundle.combined
    if args.limit is not None:
        sources = sources[: args.limit]
    source_by_id = {source.source_train_task_id: source for source in sources}
    source_index = {
        source.source_train_task_id: index for index, source in enumerate(sources)
    }
    ordered_ids = [source.source_train_task_id for source in sources]

    output_path = Path(args.output).expanduser().resolve()
    receipt_path = Path(args.receipts).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_rows = _read_jsonl(output_path)
    accepted = {
        str(value["source_train_task_id"]): value
        for value in output_rows
        if value.get("source_train_task_id") in source_by_id
    }
    if len(accepted) != len(output_rows):
        raise ValueError("resume sidecar has duplicate or foreign source IDs")
    receipt_rows = _read_jsonl(receipt_path)
    receipts = {
        str(value["source_train_task_id"]): value
        for value in receipt_rows
        if value.get("source_train_task_id") in source_by_id
    }
    if len(receipts) != len(receipt_rows):
        raise ValueError("resume receipts have duplicate or foreign source IDs")
    resume_rejected: list[str] = []
    for source_id, value in list(accepted.items()):
        try:
            materialize_hotpotqa_declarative_facts(
                (source_by_id[source_id],), (value,)
            )
        except (TypeError, ValueError) as exc:
            accepted.pop(source_id)
            previous_status = receipts.get(source_id, {}).get("status")
            receipts[source_id] = {
                "source_train_task_id": source_id,
                "status": "resume_rejected_by_current_admission",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "previous_status": previous_status,
                "admission_error_type": type(exc).__name__,
                "admission_error": " ".join(str(exc).split())[:512],
                "completed_at": _utc_now(),
            }
            resume_rejected.append(source_id)
            continue
        previous = dict(receipts.get(source_id, {}))
        previous.update(
            {
                "source_train_task_id": source_id,
                "status": "resume_revalidated",
                "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                "current_deterministic_admission": True,
                "completed_at": _utc_now(),
            }
        )
        receipts[source_id] = previous

    if resume_rejected:
        # Remove stale attestations before any model request.  The successful
        # checkpoint remains source-order stable and the rejected IDs become
        # ordinary pending rows on this same resume invocation.
        _write_jsonl(
            output_path,
            [accepted[item] for item in ordered_ids if item in accepted],
        )
        _write_jsonl(
            receipt_path,
            [receipts[item] for item in ordered_ids if item in receipts],
        )

    pending = [
        source
        for source in sources
        if source.source_train_task_id not in accepted
    ]
    model = None
    provider = None
    if pending:
        registry = load_model_registry(Path(args.model_catalog))
        model = registry.require_model(args.model_id)
        provider = registry.provider_for(args.model_id)
        if model.model_id != "qwen3.5-9b-local":
            raise ValueError("full-dataset generator must be local Qwen3.5-9B")
        # DIRECT_REUSE: TriviaQA's SkillFlow-derived materializer sizes its
        # HTTP worker pool from the requested generation concurrency.  The
        # shared Gateway uses ``asyncio.to_thread``; without this assignment,
        # Python silently caps real request concurrency at its small default
        # executor size even when the materializer semaphore is larger.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(
                max_workers=args.concurrency,
                thread_name_prefix="hotpotqa-fact-materializer",
            )
        )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def generate(source: HotpotQATrainQASource) -> object:
        assert model is not None and provider is not None
        async with semaphore:
            return await _materialize_one(
                source,
                index=source_index[source.source_train_task_id],
                model=model,
                provider=provider,
                seed=args.seed,
                max_attempts=args.max_attempts,
                generation_rounds=getattr(args, "generation_rounds", 2),
            )

    failed_source_ids: list[str] = []
    checkpoint_size = max(args.checkpoint_every, args.concurrency)
    processed_since_checkpoint = 0
    pending_iterator = iter(pending)
    active: dict[asyncio.Task[object], HotpotQATrainQASource] = {}
    for source in islice(pending_iterator, args.concurrency):
        active[asyncio.create_task(generate(source))] = source
    while active:
        # DIRECT_REUSE: TriviaQA's SkillFlow-derived materializer uses a
        # FIRST_COMPLETED rolling future pool.  Refill each completed slot
        # immediately so a small number of bounded-repair tail cases cannot
        # leave the GPU queue idle at a fixed chunk boundary.
        done, _still_pending = await asyncio.wait(
            tuple(active),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            source = active.pop(task)
            source_id = source.source_train_task_id
            try:
                result = task.result()
            except Exception as exc:
                failed_source_ids.append(source_id)
                rejection_receipt: dict[str, object] = {
                    "source_train_task_id": source_id,
                    "status": "rejected_after_bounded_attempts",
                    "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
                    "generation_error_type": type(exc).__name__,
                    "generation_error": " ".join(str(exc).split())[:512],
                    "completed_at": _utc_now(),
                }
                if isinstance(exc, FactMaterializationRejected):
                    rejection_receipt["attempt_receipts"] = list(
                        exc.attempt_receipts
                    )
                receipts[source_id] = rejection_receipt
            else:
                candidate, receipt = result
                accepted[source_id] = candidate
                receipts[source_id] = receipt
            processed_since_checkpoint += 1
            next_source = next(pending_iterator, None)
            if next_source is not None:
                active[asyncio.create_task(generate(next_source))] = next_source
        if processed_since_checkpoint >= checkpoint_size or not active:
            _write_jsonl(
                output_path,
                [accepted[item] for item in ordered_ids if item in accepted],
            )
            _write_jsonl(
                receipt_path,
                [receipts[item] for item in ordered_ids if item in receipts],
            )
            print(
                json.dumps(
                    {
                        "checkpoint_accepted_count": len(accepted),
                        "checkpoint_receipt_count": len(receipts),
                        "rejected_source_count": len(failed_source_ids),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            processed_since_checkpoint = 0

    if failed_source_ids:
        raise RuntimeError(
            f"{len(failed_source_ids)} fact materializations failed; accepted "
            f"progress was preserved ({len(accepted)}/{len(sources)}); first="
            f"{failed_source_ids[0]}; rerun with a different --seed to retry "
            "only rejected sources"
        )

    ordered = [accepted[source.source_train_task_id] for source in sources]
    materialize_hotpotqa_declarative_facts(sources, ordered)
    train_ids = {
        source.source_train_task_id for source in source_bundle.train
    }
    source_train_count = sum(
        source.source_train_task_id in train_ids for source in sources
    )
    manifest: dict[str, object] = {
        "schema_version": (
            "flowsteer.hotpotqa.full_dataset_fact_materialization.v1"
        ),
        "prompt_version": PROMPT_VERSION,
        "paraphrase_version": PARAPHRASE_VERSION,
        "paraphrase_provenance": PARAPHRASE_PROVENANCE,
        "paraphrase_versions": sorted(
            {str(value["paraphrase_version"]) for value in ordered}
        ),
        "paraphrase_provenances": sorted(
            {str(value["paraphrase_provenance"]) for value in ordered}
        ),
        "source_dataset": "HotpotQA",
        "source_configuration": "distractor",
        "source_splits": ["train", "validation"],
        "source_record_count": len(sources),
        "source_train_count": source_train_count,
        "source_validation_count": len(sources) - source_train_count,
        "unique_source_count": len({source.base_task_id for source in sources}),
        "cycled_record_count": sum(source.cycled for source in sources),
        "question_rewrite_count": len(ordered),
        "fact_count": len(ordered),
        "semantic_rewrite_coverage": 1.0,
        "fallback_count": 0,
        "index_external_metadata_fields": [
            "source_train_task_id",
            "paraphrase_question",
            "paraphrase_provenance",
            "paraphrase_version",
        ],
        "indexed_text_field": "fact_statement",
        "document_format": "declarative_fact_only",
        "contains_raw_questions_in_fact_records": False,
        "contains_raw_answers_in_fact_records": False,
        "evaluation_scope": FULL_DATASET_EVALUATION_SCOPE,
        "contains_evaluation_source_facts": True,
        "official_heldout_eligible": False,
        "generator_model_id": args.model_id,
        "seed": args.seed,
        "max_attempts": args.max_attempts,
        "generation_rounds": getattr(args, "generation_rounds", 2),
        "generation_concurrency": args.concurrency,
        "http_executor_workers": args.concurrency,
        "checkpoint_every": checkpoint_size,
        "accepted_count": len(ordered),
        "rejected_count": 0,
        "materialization_path": str(output_path),
        "generation_receipts_path": str(receipt_path),
        "completed_at": _utc_now(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-catalog", default="config/datasets_agentgraph.yaml")
    parser.add_argument(
        "--model-catalog",
        default="config/model_catalog_hotpotqa_qa_memory_v10.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument(
        "--output",
        default=(
            "data/hotpotqa_full_dataset_fact_memory_v1/"
            "fact_provenance_sidecar.jsonl"
        ),
    )
    parser.add_argument(
        "--receipts",
        default=(
            "data/hotpotqa_full_dataset_fact_memory_v1/"
            "generation_receipts.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "data/hotpotqa_full_dataset_fact_memory_v1/"
            "materialization_manifest.json"
        ),
    )
    parser.add_argument("--train-count", type=_positive_integer, default=90_447)
    parser.add_argument("--validation-count", type=_positive_integer, default=7_405)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--concurrency", type=_positive_integer, default=64)
    parser.add_argument("--checkpoint-every", type=_positive_integer, default=1_024)
    parser.add_argument("--max-attempts", type=_positive_integer, default=4)
    parser.add_argument(
        "--generation-rounds", type=_positive_integer, default=2
    )
    parser.add_argument("--limit", type=_positive_integer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(materialize(args))
    except Exception as exc:
        print(
            f"HotpotQA full-dataset fact materialization failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
