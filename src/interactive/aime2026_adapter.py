"""AIME 2026 target-blind extraction and integer Accuracy boundary.

The trusted-target comparison remains the thin port of downstream SkillEval's
``PrivateStaticTarget.score`` branch for ``StaticScoringRule.INTEGER``.  A free
AgentGraph returns text instead of SkillEval's already-structured
``{"answer": str}`` action, so the project-specific boundary below performs a
small deterministic projection first.  Its admitted markers are the explicit
integer, ``\\boxed{...}``, and ``Final Answer: ...`` forms used by SkillFlow's
math parsing path.  It deliberately omits SkillFlow training reward's broad
"last number" fallback: extraction never solves, repairs, looks up, or compares
against the trusted target when choosing a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


AIME2026_DATASET_KEY = "aime_2026"
AIME2026_TASK_FAMILY = "aime-2026/integer-answer"
AIME2026_EVALUATOR_VERSION = "skillev.integer.target-blind-extraction.v2.1"
AIME2026_ANSWER_FORMAT = "integer-000-to-999"

_ANSWER_TAG = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINKING_END = "</think>"
_BOXED_INTEGER = re.compile(r"\\boxed\s*\{\s*([+]?\d+)\s*\}")
_FINAL_INTEGER = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer)\s*[:=]\s*"
    r"\$?\s*([+]?\d+)\s*\$?\s*[.!]?\s*$"
)
_ANSWER_IS_INTEGER = re.compile(
    r"(?im)^\s*(?:the\s+)?answer\s+is\s*[:=]?\s*"
    r"\$?\s*([+]?\d+)\s*\$?\s*[.!]?\s*$"
)
_BARE_INTEGER = re.compile(r"[+]?\d+")


@dataclass(frozen=True)
class AIME2026IntegerScore:
    """Result of the official integer-answer normalization path."""

    accuracy: float
    raw_prediction: str
    scored_prediction: str
    structured_answer_extracted: bool
    parsing_succeeded: bool
    parsing_failure_reason: str | None
    canonical_prediction: str | None


def extract_aime2026_submission(prediction: str) -> tuple[str, bool, str | None]:
    """Map one terminal output to a target-blind integer submission string.

    A single complete answer boundary is permitted and its contents are
    submitted.  With no boundary, the complete response is submitted exactly
    as SkillEval's private static task expects.  Multiple or malformed
    boundaries fail closed instead of selecting a convenient candidate.
    """

    if not isinstance(prediction, str):
        raise TypeError("AIME prediction must be text")
    tagged = _ANSWER_TAG.findall(prediction)
    if len(tagged) > 1:
        return "", False, "multiple_answer_boundaries"
    if len(tagged) == 1:
        remainder = _ANSWER_TAG.sub("", prediction, count=1).casefold()
        if (
            "<answer" in remainder
            or "</answer" in remainder
            or "<answer" in tagged[0].casefold()
            or "</answer" in tagged[0].casefold()
        ):
            return "", False, "malformed_answer_boundary"
        return tagged[0].strip(), True, None
    if "<answer" in prediction.casefold() or "</answer" in prediction.casefold():
        return "", False, "malformed_answer_boundary"
    return prediction.strip(), False, None


def canonical_aime_integer(value: object) -> str:
    """Validate and canonicalize one trusted AIME target to ``0``--``999``."""

    if isinstance(value, bool):
        raise ValueError("AIME answer must be an integer, not bool")
    if isinstance(value, int):
        integer = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(r"[+]?[0-9]+", stripped):
            raise ValueError("AIME answer must contain one decimal integer")
        integer = int(stripped)
    else:
        raise ValueError("AIME answer must be integer text")
    if not 0 <= integer <= 999:
        raise ValueError("AIME answer must lie in [0, 999]")
    return str(integer)


def extract_aime2026_candidate(
    prediction: str,
) -> tuple[str | None, bool, str | None]:
    """Extract one unambiguous public AIME candidate without using the target.

    The optional FlowSteer terminal envelope is removed first.  An explicit
    marker is admitted only when every marker present names the same integer;
    contradictory markers fail closed.  Without a marker, the entire visible
    response must be one integer.  This preserves SkillFlow's public math
    answer forms without importing its reward-only last-number heuristic.
    """

    submitted, structured, boundary_failure = extract_aime2026_submission(
        prediction
    )
    if boundary_failure is not None:
        return None, structured, boundary_failure
    visible = submitted
    if _THINKING_END in visible:
        visible = visible.rsplit(_THINKING_END, 1)[1]
    visible = visible.strip()
    if not visible:
        return None, structured, "empty_answer"

    marked = [
        *(_BOXED_INTEGER.findall(visible)),
        *(_FINAL_INTEGER.findall(visible)),
        *(_ANSWER_IS_INTEGER.findall(visible)),
    ]
    if marked:
        try:
            candidates = {canonical_aime_integer(value) for value in marked}
        except ValueError:
            return None, structured, "aime_integer_out_of_range"
        if len(candidates) != 1:
            return None, structured, "conflicting_explicit_candidates"
        return next(iter(candidates)), structured, None

    candidate = visible
    if _BARE_INTEGER.fullmatch(candidate) is None:
        # SkillFlow's real ``extract_math_answer`` falls back to a number in
        # the final three lines.  The formal adapter ports only the narrower,
        # target-blind case where the final non-empty line is exactly one
        # integer; arbitrary last-number selection remains inadmissible.
        candidate = next(
            (line.strip() for line in reversed(visible.splitlines()) if line.strip()),
            "",
        )
        if _BARE_INTEGER.fullmatch(candidate) is None:
            return None, structured, "aime_integer_not_found"
    try:
        return canonical_aime_integer(candidate), structured, None
    except ValueError:
        return None, structured, "aime_integer_out_of_range"


def score_aime2026_integer(
    prediction: str,
    accepted_answers: Sequence[str],
) -> AIME2026IntegerScore:
    """Score a submitted answer exactly as SkillEval's ``INTEGER`` rule.

    SkillEval applies ``str(int(prediction.strip()))`` and compares it with the
    equivalently canonicalized trusted answers.  FlowSteer's terminal protocol
    can wrap the submission in one ``<answer>`` boundary.  Without that
    boundary the whole output is scored, preserving the fail-closed behavior.
    """

    if not accepted_answers:
        raise ValueError("AIME evaluator requires at least one accepted answer")
    expected = {canonical_aime_integer(answer) for answer in accepted_answers}
    submitted, _, _ = extract_aime2026_submission(prediction)
    predicted, structured, parsing_failure_reason = extract_aime2026_candidate(
        prediction
    )
    accuracy = float(predicted in expected) if predicted is not None else 0.0
    return AIME2026IntegerScore(
        accuracy=accuracy,
        raw_prediction=prediction,
        scored_prediction=submitted,
        structured_answer_extracted=structured,
        parsing_succeeded=parsing_failure_reason is None,
        parsing_failure_reason=parsing_failure_reason,
        canonical_prediction=predicted,
    )


__all__ = [
    "AIME2026_ANSWER_FORMAT",
    "AIME2026_DATASET_KEY",
    "AIME2026_EVALUATOR_VERSION",
    "AIME2026_TASK_FAMILY",
    "AIME2026IntegerScore",
    "canonical_aime_integer",
    "extract_aime2026_candidate",
    "extract_aime2026_submission",
    "score_aime2026_integer",
]
