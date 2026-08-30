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
import json
import re
from typing import Mapping, Sequence


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
_ARTIFACT_ASSESSMENTS = re.compile(
    r"<artifact_assessments>\s*(.*?)\s*</artifact_assessments>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ARTIFACT_ASSESSMENT_FENCED_FALLBACK = re.compile(
    r"(?ims)^\s*#{1,6}\s*Artifact Assessment\s*$\s*"
    r"```json\s*(\{.*?\})\s*```"
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


@dataclass(frozen=True)
class AIME2026ArtifactAssessment:
    """Target-blind assessment of one provenance-bound upstream artifact."""

    assessed_artifact_id: str
    candidate: str
    assessment: str
    basis: str
    counterexample: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "assessed_artifact_id": self.assessed_artifact_id,
            "candidate": self.candidate,
            "assessment": self.assessment,
            "basis": self.basis,
            "counterexample": self.counterexample,
        }


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

def extract_aime2026_artifact_assessments(
    prediction: str,
) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    """Parse target-blind, provenance-bound upstream assessments.

    This parser only validates the public protocol. It does not recompute the
    problem, consult a target, or decide whether the stated basis is true.
    """

    if not isinstance(prediction, str):
        raise TypeError("AIME artifact assessment must be text")
    matches = _ARTIFACT_ASSESSMENTS.findall(prediction)
    fallback_variant = False
    if matches:
        if len(matches) != 1:
            return (), "multiple_artifact_assessment_boundaries"
        try:
            raw_items = json.loads(matches[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return (), "artifact_assessment_json_invalid"
    else:
        fallback_matches = _ARTIFACT_ASSESSMENT_FENCED_FALLBACK.findall(
            prediction
        )
        if not fallback_matches:
            return (), "artifact_assessment_not_found"
        if len(fallback_matches) != 1:
            return (), "multiple_artifact_assessment_fallback_boundaries"
        try:
            raw_container = json.loads(fallback_matches[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return (), "artifact_assessment_fallback_json_invalid"
        if (
            not isinstance(raw_container, Mapping)
            or set(raw_container) != {"assessed_artifacts"}
        ):
            return (), "artifact_assessment_fallback_object_invalid"
        raw_items = raw_container["assessed_artifacts"]
        fallback_variant = True
    if (
        isinstance(raw_items, (str, bytes))
        or not isinstance(raw_items, Sequence)
        or not raw_items
    ):
        return (), "artifact_assessment_list_invalid"

    assessments: list[AIME2026ArtifactAssessment] = []
    seen_artifact_ids: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            return (), "artifact_assessment_item_invalid"
        if fallback_variant and set(raw_item) != {
            "assessed_artifact_id",
            "candidate",
            "assessment",
            "basis",
            "counterexample",
        }:
            return (), "artifact_assessment_fallback_item_fields_invalid"
        artifact_id = raw_item.get("assessed_artifact_id")
        basis = raw_item.get("basis")
        status = raw_item.get("assessment")
        if fallback_variant and status == "sufficient_evidence":
            status = "supported"
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            return (), "assessed_artifact_id_invalid"
        artifact_id = artifact_id.strip()
        if artifact_id in seen_artifact_ids:
            return (), "duplicate_assessed_artifact_id"
        if not isinstance(basis, str) or not basis.strip():
            return (), "artifact_assessment_basis_missing"
        if status not in {
            "supported",
            "insufficient_evidence",
            "refuted",
        }:
            return (), "artifact_assessment_status_invalid"
        try:
            candidate = canonical_aime_integer(raw_item.get("candidate"))
        except ValueError:
            return (), "artifact_assessment_candidate_invalid"
        raw_counterexample = raw_item.get("counterexample")
        if raw_counterexample is not None and not isinstance(
            raw_counterexample, str
        ):
            return (), "artifact_assessment_counterexample_invalid"
        counterexample = (
            raw_counterexample.strip()
            if isinstance(raw_counterexample, str)
            and raw_counterexample.strip()
            else None
        )
        if status == "refuted" and counterexample is None:
            return (), "refuted_assessment_requires_counterexample"
        if status == "supported" and counterexample is not None:
            return (), "supported_assessment_has_counterexample"
        assessments.append(
            AIME2026ArtifactAssessment(
                assessed_artifact_id=artifact_id,
                candidate=candidate,
                assessment=str(status),
                basis=basis.strip(),
                counterexample=counterexample,
            )
        )
        seen_artifact_ids.add(artifact_id)
    return tuple(item.to_dict() for item in assessments), None



def _supported_assessment_terminal_candidate(
    prediction: str,
) -> tuple[str | None, str | None, bool]:
    """Project one explicit assessment candidate without consulting a target.

    The assessment block is an execution artifact, not a new scoring rule.
    It can expose a terminal candidate only when one candidate is supported,
    no assessment is insufficient, and every different candidate is refuted
    with the counterexample already required by the public assessment parser.
    """

    has_boundary = bool(
        _ARTIFACT_ASSESSMENTS.search(prediction)
        or _ARTIFACT_ASSESSMENT_FENCED_FALLBACK.search(prediction)
    )
    if not has_boundary:
        return None, None, False
    assessments, failure = extract_aime2026_artifact_assessments(prediction)
    if failure is not None:
        return None, "assessment_terminal_" + failure, True

    statuses_by_candidate: dict[str, set[str]] = {}
    for item in assessments:
        candidate = str(item["candidate"])
        statuses_by_candidate.setdefault(candidate, set()).add(
            str(item["assessment"])
        )
    if any(
        "insufficient_evidence" in statuses
        for statuses in statuses_by_candidate.values()
    ):
        return None, "assessment_terminal_insufficient_evidence", True
    supported = {
        candidate
        for candidate, statuses in statuses_by_candidate.items()
        if "supported" in statuses
    }
    if len(supported) != 1:
        return None, "assessment_terminal_no_unique_supported_candidate", True
    selected = next(iter(supported))
    if "refuted" in statuses_by_candidate[selected]:
        return None, "assessment_terminal_conflicting_status", True
    if any(
        candidate != selected and statuses != {"refuted"}
        for candidate, statuses in statuses_by_candidate.items()
    ):
        return None, "assessment_terminal_unresolved_candidate", True
    return selected, None, True


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

    marker_visible = _ARTIFACT_ASSESSMENTS.sub("", visible)
    marker_visible = _ARTIFACT_ASSESSMENT_FENCED_FALLBACK.sub(
        "", marker_visible
    ).strip()
    marked = [
        *(_BOXED_INTEGER.findall(marker_visible)),
        *(_FINAL_INTEGER.findall(marker_visible)),
        *(_ANSWER_IS_INTEGER.findall(marker_visible)),
    ]
    if marked:
        try:
            candidates = {canonical_aime_integer(value) for value in marked}
        except ValueError:
            return None, structured, "aime_integer_out_of_range"
        if len(candidates) != 1:
            return None, structured, "conflicting_explicit_candidates"
        return next(iter(candidates)), structured, None

    assessment_candidate, assessment_failure, assessment_attempted = (
        _supported_assessment_terminal_candidate(visible)
    )
    if assessment_attempted:
        return (
            assessment_candidate,
            structured,
            assessment_failure,
        )

    candidate = marker_visible
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
    "AIME2026ArtifactAssessment",
    "AIME2026IntegerScore",
    "canonical_aime_integer",
    "extract_aime2026_artifact_assessments",
    "extract_aime2026_candidate",
    "extract_aime2026_submission",
    "score_aime2026_integer",
]
