"""AIME 2026 integer-answer boundary used by the dataset adapter.

The integer canonicalization is a thin port of downstream SkillEval's
``PrivateStaticTarget.score`` branch for ``StaticScoringRule.INTEGER``.  The
public SkillFlow repository does not contain this AIME-2026-specific scorer.
The only project-specific layer here maps FlowSteer's existing single
``<answer>...</answer>`` terminal boundary to the private scorer's
``{"answer": str}`` submission.  It never solves, repairs, or looks up an
answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


AIME2026_DATASET_KEY = "aime_2026"
AIME2026_TASK_FAMILY = "aime-2026/integer-answer"
AIME2026_EVALUATOR_VERSION = "skillev.private-static.integer.v1"
AIME2026_ANSWER_FORMAT = "integer-000-to-999"

_ANSWER_TAG = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)


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
    submitted, structured, boundary_failure = extract_aime2026_submission(prediction)
    predicted: str | None = None
    parsing_failure_reason = boundary_failure
    if boundary_failure is not None:
        accuracy = 0.0
    else:
        try:
            predicted = str(int(submitted.strip()))
        except ValueError:
            accuracy = 0.0
            parsing_failure_reason = "integer_conversion_failed"
        else:
            accuracy = float(predicted in expected)
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
    "extract_aime2026_submission",
    "score_aime2026_integer",
]
