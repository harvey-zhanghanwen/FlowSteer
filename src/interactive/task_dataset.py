"""Strict JSONL task loading for the AgentGraph path.

The reader keeps FlowSteer's line-oriented loader boundary while returning the
``TaskRecord`` contract from the project design note.  Dataset-specific fields
remain in ``metadata`` and are never added to the Director question.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .records import TaskRecord, VALID_SPLITS


TASK_SCHEMA_VERSION = "flowsteer.agentgraph.task.v1"
REQUIRED_FIELDS = frozenset(
    {"schema_version", "task_id", "question", "ground_truth", "split", "metadata"}
)
_HOTPOTQA_QUESTION_MARKER = "\n\nQuestion:"


def build_hotpotqa_question(question: str, passages: List[str]) -> str:
    """Render SkillFlow-style multi-hop QA input with all supplied passages."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("HotpotQA question must be non-empty")
    if not passages or any(not isinstance(item, str) or not item.strip() for item in passages):
        raise ValueError("HotpotQA passages must contain non-empty text")
    parts = ["Based on the following passages, answer the question."]
    parts.extend(f"[{passage}]" for passage in passages[:10])
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(parts)


def qa_question_scope(rendered_question: str) -> str:
    """Return the exact QA question span without narrowing its semantics.

    HotpotQA may use :func:`build_hotpotqa_question`, while TriviaQA supplies
    the factual question directly.  Both protocols therefore share one
    question-only boundary; no context, answer alias, or evaluator field is
    consulted here.
    """

    if not isinstance(rendered_question, str) or not rendered_question.strip():
        raise ValueError("QA question must be non-empty")
    if _HOTPOTQA_QUESTION_MARKER not in rendered_question:
        return rendered_question.strip()
    scope = rendered_question.rsplit(_HOTPOTQA_QUESTION_MARKER, 1)[1].strip()
    if not scope:
        raise ValueError("rendered QA input has an empty Question field")
    return scope


def hotpotqa_question_scope(rendered_question: str) -> str:
    """Backward-compatible HotpotQA wrapper for :func:`qa_question_scope`."""

    return qa_question_scope(rendered_question)


def verified_year_to_decade_normalization(
    *,
    original_question: Optional[str],
    source_value: object,
    candidate_answer: object,
) -> bool:
    """Verify a deterministic year-to-decade answer normalization.

    The source year remains an exact proposition argument grounded by the
    evidence-provenance gate. No accepted answer or evaluator state enters the
    runtime use of this check; it applies only when the question explicitly
    requests a decade. The evaluator may reuse the same deterministic rule for
    diagnosis without changing official EM/F1.
    """

    if (
        not isinstance(original_question, str)
        or not isinstance(source_value, str)
        or not isinstance(candidate_answer, str)
    ):
        return False
    question = " ".join(qa_question_scope(original_question).casefold().split())
    if not re.search(r"\b(?:what|which) decade\b", question):
        return False
    years = tuple(
        dict.fromkeys(
            int(value)
            for value in re.findall(
                r"(?<!\d)(?:1\d{3}|20\d{2})(?!\d)",
                source_value,
            )
        )
    )
    if len(years) != 1:
        return False
    decade_start = years[0] // 10 * 10
    short_start = decade_start % 100
    normalized = unicodedata.normalize("NFKC", candidate_answer).casefold()
    normalized = " ".join(normalized.split()).strip(" .,:;[]")
    normalized = re.sub(r"\s*\(decade\)\s*$", "", normalized).strip()
    normalized = re.sub(r"\s+(?:ad|ce)$", "", normalized).strip()
    normalized = re.sub(r"^the\s+", "", normalized).strip()
    numeric_forms = {
        f"{decade_start}s",
        f"{decade_start}'s",
        f"{decade_start}’s",
        f"{decade_start}-{decade_start + 9}",
        f"{decade_start}–{decade_start + 9}",
        f"{short_start:02d}s",
        f"{short_start:02d}'s",
        f"{short_start:02d}’s",
        f"'{short_start:02d}s",
        f"’{short_start:02d}s",
        f"{short_start:02d}-{(short_start + 9) % 100:02d}",
        f"{short_start:02d}–{(short_start + 9) % 100:02d}",
    }
    decade_words = {
        20: "twenties",
        30: "thirties",
        40: "forties",
        50: "fifties",
        60: "sixties",
        70: "seventies",
        80: "eighties",
        90: "nineties",
    }
    return normalized in numeric_forms or normalized == decade_words.get(
        short_start
    )


def qa_answer_type_constraint(rendered_question: str) -> str:
    """Return the question's surface-syntax answer-type constraint.

    This is an answer-independent wh-word/auxiliary classification.  It does
    not inspect passages, candidate answers, Ground Truth, or evaluator state.
    """

    question = qa_question_scope(rendered_question)
    normalized = " ".join(question.casefold().split())
    if re.search(r"\bwhat nationality\b", normalized):
        return "nationality"
    if re.search(r"\bhow (?:many|much)\b", normalized):
        return "number"
    if normalized.startswith("when ") or re.search(
        r"\b(?:what|which) (?:year|date|decade|century)\b", normalized
    ):
        return "date"
    if normalized.startswith("where ") or re.search(
        r"\bwhere\s*\?\s*$", normalized
    ):
        return "location"
    if normalized.startswith("who ") or re.search(r"\bwho\s*\?\s*$", normalized):
        # "Who" can denote a person, team, band, company, government, or
        # another organization.  A question-only constraint must not narrow
        # that open entity class before evidence is available.
        return "entity"
    if re.search(
        r"\bwhat (?:country|city|place|location)\b", normalized
    ):
        return "location"
    # HotpotQA frequently places the interrogative constituent after a named
    # subject (for example, "X includes which ...?").  Classify that actual
    # answer slot rather than only the first token of the sentence.
    if normalized.startswith("which ") or re.search(
        r"\bwhich\s+[^?]+\?\s*$", normalized
    ):
        return "entity"
    if re.search(
        r"\bwhat (?:network|organization|company|band|team|school|university)\b",
        normalized,
    ):
        return "entity"
    if normalized.startswith(
        (
            "are ",
            "is ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "has ",
            "have ",
            "had ",
            "can ",
            "could ",
        )
    ):
        return "yes_no"
    return "short_answer"


def hotpotqa_answer_type_constraint(rendered_question: str) -> str:
    """Backward-compatible HotpotQA wrapper for the shared QA classifier."""

    return qa_answer_type_constraint(rendered_question)


def qa_answer_argument_constraint(rendered_question: str) -> Optional[str]:
    """Return a conservative question-only answer argument constraint.

    The constraint is derived only from an overt wh-dependency.  It deliberately
    returns ``None`` for syntactically ambiguous questions rather than guessing a
    semantic role.  In the high-confidence active-subject construction
    ``Which/Who ... <finite lexical predicate>``, the wh constituent fills the
    proposition subject.  Auxiliary-inverted object questions are left
    unconstrained so retrieved evidence, rather than a surface heuristic, owns
    the remaining binding decision.
    """

    question = qa_question_scope(rendered_question)
    if qa_answer_type_constraint(question) != "entity":
        return None
    normalized = " ".join(question.casefold().split())
    if not re.match(r"^(?:who|which|what)\b", normalized):
        return None
    # Do-support and modal inversion introduce an overt post-auxiliary subject,
    # so the leading wh constituent is not safely classifiable as the subject.
    if re.match(
        r"^(?:who|which|what)\b.*?\b"
        r"(?:am|are|is|was|were|has|have|had|do|does|did|can|could|"
        r"will|would|shall|should|may|might|must)\b",
        normalized,
    ):
        return None
    finite_lexical_predicate = re.compile(
        r"\b(?:became|began|came|held|made|went|won|wrote|"
        r"[a-z]+(?:ed|s))\b"
    )
    prefix = re.sub(r"^(?:who|which|what)\s+", "", normalized, count=1)
    if finite_lexical_predicate.search(prefix) is None:
        return None
    return "subject"


def qa_answer_cardinality_constraint(rendered_question: str) -> str:
    """Return a question-only single-value or multiple-value answer constraint."""

    question = qa_question_scope(rendered_question)
    normalized = " ".join(question.casefold().split())
    if re.search(
        r"^(?:what (?:are|were) the names?\b|who (?:are|were)\b|"
        r"name (?:all|the)\b)",
        normalized,
    ):
        return "multiple"
    return "single"


def hotpotqa_answer_cardinality_constraint(rendered_question: str) -> str:
    """Backward-compatible HotpotQA wrapper for shared QA cardinality."""

    return qa_answer_cardinality_constraint(rendered_question)


def task_record_from_mapping(
    item: Mapping[str, Any], *, expected_split: Optional[str] = None
) -> TaskRecord:
    """Validate one aligned mapping and return the runtime task record."""

    missing = sorted(REQUIRED_FIELDS.difference(item))
    if missing:
        raise ValueError(f"aligned task is missing required fields: {missing}")
    if item["schema_version"] != TASK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported task schema {item['schema_version']!r}; "
            f"expected {TASK_SCHEMA_VERSION!r}"
        )

    split = str(item["split"])
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid task split {split!r}")
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"split isolation violation: expected {expected_split!r}, got {split!r}"
        )

    raw_metadata = item["metadata"]
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("task metadata must be a mapping")
    metadata = dict(raw_metadata)
    # SkillFlow consumes these optional top-level fields.  Rehydrate them under
    # one metadata key so the strict TaskRecord does not discard environment or
    # code-task handles when a caller streams the canonical JSONL.
    skillflow_fields = {
        key: item[key]
        for key in (
            "answer",
            "task_type",
            "context",
            "extra",
            "env_type",
            "env_config",
            "code_files",
        )
        if key in item
    }
    if skillflow_fields:
        metadata["skillflow"] = skillflow_fields

    question = str(item["question"])
    if metadata.get("dataset_key") == "hotpotqa":
        raw_context = skillflow_fields.get("context")
        marker = _HOTPOTQA_QUESTION_MARKER
        if (
            isinstance(raw_context, list)
            and raw_context
            and all(isinstance(value, str) for value in raw_context)
            and marker in question
        ):
            # Older aligned files truncated each passage to 300 characters even
            # though their canonical top-level context retained the full text.
            # Rehydrate only evidence, never evaluator payload or ground truth.
            question = build_hotpotqa_question(
                question.rsplit(marker, 1)[1],
                list(raw_context),
            )
            metadata["hotpot_context_mode"] = "full_passages_v1"

    return TaskRecord(
        task_id=str(item["task_id"]),
        question=question,
        ground_truth=item["ground_truth"],
        split=split,
        metadata=metadata,
    )


def iter_task_records(
    path: str | Path,
    *,
    expected_split: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[TaskRecord]:
    """Stream records without loading a multi-benchmark split into memory."""

    source = Path(path)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return
    with source.open("r", encoding="utf-8") as handle:
        emitted = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item: Dict[str, Any] = json.loads(line)
                record = task_record_from_mapping(item, expected_split=expected_split)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def load_task_records(
    path: str | Path,
    *,
    expected_split: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[TaskRecord]:
    """Materialize a split for callers that retain FlowSteer's list API."""

    return list(iter_task_records(path, expected_split=expected_split, limit=limit))


__all__ = [
    "TASK_SCHEMA_VERSION",
    "build_hotpotqa_question",
    "hotpotqa_answer_type_constraint",
    "hotpotqa_answer_cardinality_constraint",
    "hotpotqa_question_scope",
    "iter_task_records",
    "load_task_records",
    "qa_answer_cardinality_constraint",
    "qa_answer_type_constraint",
    "qa_question_scope",
    "task_record_from_mapping",
    "verified_year_to_decade_normalization",
]
