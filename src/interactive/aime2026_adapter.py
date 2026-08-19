"""AIME 2026 integer-answer boundary used by the dataset-specific adapter.

This is a thin local port of SkillFlow Protocol 10's
``PrivateStaticTarget.score`` branch for ``StaticScoringRule.INTEGER``.  The
only project adaptation is recognition of FlowSteer's existing
``<answer>...</answer>`` terminal boundary before the submitted value reaches
the scorer.  Free-form explanations, decimal values, percentages, and
symbolic-equivalence heuristics are intentionally not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


AIME2026_DATASET_KEY = "aime_2026"
AIME2026_TASK_FAMILY = "aime-2026/integer-answer"
AIME2026_EVALUATOR_VERSION = "skillflow.protocol-v10.static.integer.v1"
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
    """Score a submitted answer exactly as SkillFlow's ``INTEGER`` rule.

    SkillFlow applies ``str(int(prediction.strip()))`` and compares it with the
    equivalently canonicalized trusted answers.  FlowSteer's Format operator
    can wrap the submission in ``<answer>``; when present, the last complete
    boundary is the submitted value.  Without that boundary the whole output
    is scored, preserving the fail-closed official behavior.
    """

    if not isinstance(prediction, str):
        raise TypeError("AIME prediction must be text")
    if not accepted_answers:
        raise ValueError("AIME evaluator requires at least one accepted answer")
    expected = {canonical_aime_integer(answer) for answer in accepted_answers}
    tagged = _ANSWER_TAG.findall(prediction)
    submitted = tagged[-1].strip() if tagged else prediction
    try:
        predicted = str(int(submitted.strip()))
    except ValueError:
        accuracy = 0.0
    else:
        accuracy = float(predicted in expected)
    return AIME2026IntegerScore(
        accuracy=accuracy,
        raw_prediction=prediction,
        scored_prediction=submitted,
        structured_answer_extracted=bool(tagged),
    )


__all__ = [
    "AIME2026_ANSWER_FORMAT",
    "AIME2026_DATASET_KEY",
    "AIME2026_EVALUATOR_VERSION",
    "AIME2026_TASK_FAMILY",
    "AIME2026IntegerScore",
    "canonical_aime_integer",
    "score_aime2026_integer",
]
