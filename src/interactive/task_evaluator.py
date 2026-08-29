"""Terminal evaluators for the seven AgentGraph datasets.

The reward boundary is deliberately strict: an unavailable judge, benchmark
environment, or SWE-bench harness produces an invalid outcome rather than a
proxy reward.  Invalid outcomes must not be used by GRPO.

Source boundaries
-----------------
* HotpotQA/TriviaQA token F1 and AIME exact matching are retained from
  SkillFlow ``training/reward.py``.
* HealthBench grading follows OpenAI simple-evals ``healthbench_eval.py``
  (``GRADER_TEMPLATE`` and ``calculate_score``).
* WebShop/ALFWorld execution uses SkillFlow's deployed ``RAGENAdapter`` via a
  dynamic import.  This module only adds the AgentGraph callback boundary and
  exact task locking required by the aligned records.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import string
import sys
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from .aime2026_adapter import (
    AIME2026_EVALUATOR_VERSION,
    score_aime2026_integer,
)
from .mbppplus_adapter import MBPPPLUS_EVALUATOR_VERSION
from .records import TaskRecord
from .task_dataset import verified_year_to_decade_normalization


DEFAULT_RAGEN_ADAPTER_PATH = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py"
)

SKILLFLOW_REWARD_VERSION = "skillflow.training.reward.v1"
HOTPOTQA_ANSWER_EVALUATOR_VERSION = "hotpotqa.official.answer.v1"
TRIVIAQA_ANSWER_EVALUATOR_VERSION = "triviaqa.official.answer.v1"
HEALTHBENCH_EVALUATOR_VERSION = "openai.simple-evals.healthbench.v1"
RAGEN_EVALUATOR_VERSION = "skillflow.ragen_adapter.v2"
SWEBENCH_EVALUATOR_VERSION = "swebench.harness.v1"
UNAVAILABLE_EVALUATOR_VERSION = "agentgraph.evaluator.unavailable.v1"


JudgeCallback = Callable[[Sequence[Mapping[str, str]], str], Awaitable[Any]]
RunGraphCallback = Callable[[str], Awaitable[str]]
SWEHarnessCallback = Callable[[TaskRecord | Mapping[str, Any], str], Awaitable[Any]]
MBPPPlusHarnessCallback = Callable[
    [TaskRecord | Mapping[str, Any], str], Awaitable[Any]
]


@dataclass(frozen=True)
class EvaluationOutcome:
    """One terminal evaluation result suitable for an evaluator receipt."""

    valid: bool
    reward: Optional[float]
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    evaluator_version: str = UNAVAILABLE_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        if self.valid and self.reward is None:
            raise ValueError("a valid evaluation requires a reward")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("evaluation reward must be finite")


# Retained from SkillFlow training/reward.py.  The local copy avoids making
# static benchmark evaluation depend on another checkout at runtime.
def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _token_f1_multi(prediction: str, candidates: Sequence[str]) -> float:
    return max(_token_f1(prediction, candidate) for candidate in candidates)


# HotpotQA official answer scorer, kept separate from SkillFlow's TriviaQA
# reward because the two normalizers intentionally differ.
def _normalize_hotpotqa_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _hotpotqa_answer_f1(prediction: str, gold: str) -> float:
    normalized_prediction = _normalize_hotpotqa_answer(prediction)
    normalized_gold = _normalize_hotpotqa_answer(gold)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers
        or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# SkillFlow Formal Protocol 10 scorer from
# ``skillev_private.benchmarks.static``.  TriviaQA and HotpotQA share the
# standard lower/punctuation/article/whitespace normalization, while HotpotQA
# additionally carries the benchmark's yes/no/noanswer special case above.
def _normalize_triviaqa_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _triviaqa_answer_f1(prediction: str, gold: str) -> float:
    prediction_tokens = _normalize_triviaqa_answer(prediction).split()
    gold_tokens = _normalize_triviaqa_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _extract_math_answer(text: str) -> str:
    if "boxed" in text:
        index = text.rfind("\\boxed{")
        if index >= 0:
            start = index + len("\\boxed{")
            stack = 1
            answer = ""
            for character in text[start:]:
                if character == "{":
                    stack += 1
                    answer += character
                elif character == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    answer += character
                else:
                    answer += character
            if answer:
                return (
                    answer.replace("\\displaystyle", "")
                    .replace("\\textstyle", "")
                    .strip()
                )

    stripped = text.strip()
    if len(stripped) < 100 and "boxed" not in stripped and any(
        command in stripped
        for command in ("\\frac", "\\sqrt", "\\pi", "\\infty", "\\begin")
    ):
        return stripped

    final_marker = re.search(r"####\s*(.+?)$", stripped, re.MULTILINE)
    if final_marker:
        return final_marker.group(1).strip().replace(",", "")

    answer_pattern = re.search(
        r"(?:the\s+)?answer\s+is\s+[:\s]*(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if answer_pattern:
        return answer_pattern.group(1).replace(",", "")

    number_pattern = r"-?[\d,]+(?:\.\d+)?"
    if len(text) < 200:
        numbers = re.findall(number_pattern, text)
        if numbers:
            return numbers[-1].replace(",", "")
    last_lines = "\n".join(stripped.split("\n")[-3:])
    numbers = re.findall(number_pattern, last_lines)
    if numbers:
        return numbers[-1].replace(",", "")
    numbers = re.findall(number_pattern, text)
    if numbers:
        return numbers[-1].replace(",", "")
    return text


def _strip_math_string(value: str) -> str:
    result = str(value).strip()
    result = result.replace("\\!", "").replace("\\ ", "").replace("\\,", "")
    result = result.replace("\\left", "").replace("\\right", "")
    result = result.replace("\\displaystyle", "").replace("\\textstyle", "")
    result = result.replace("tfrac", "frac").replace("dfrac", "frac")
    result = result.replace("^{\\circ}", "").replace("\\circ", "")
    result = result.replace("$", "").replace("\\%", "%")

    if "\\boxed{" in result:
        index = result.find("\\boxed{")
        start = index + len("\\boxed{")
        stack = 1
        end = len(result)
        for position in range(start, len(result)):
            if result[position] == "{":
                stack += 1
            elif result[position] == "}":
                stack -= 1
                if stack == 0:
                    end = position
                    break
        result = result[:index] + result[start:end] + result[end + 1 :]

    result = re.sub(r"\\text\{[^}]*\}", "", result).strip()
    result = re.sub(r"\\mathrm\{[^}]*\}", "", result).strip()
    for _ in range(5):
        updated = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", result)
        if updated == result:
            break
        result = updated
    for _ in range(5):
        updated = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", result)
        if updated == result:
            break
        result = updated
    return result.replace("\\pi", "pi").replace(" ", "")


def _symbolic_equal(prediction: str, gold: str) -> bool:
    try:
        from sympy import N, simplify
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False

    def parse(value: str) -> Any:
        try:
            return parse_expr(value.strip())
        except Exception:
            return None

    left = parse(prediction)
    right = parse(gold)
    if left is None or right is None:
        return False
    try:
        if simplify(left - right) == 0:
            return True
    except Exception:
        pass
    try:
        return math.isclose(float(N(left)), float(N(right)), rel_tol=1e-4)
    except Exception:
        return False


def _exact_match(prediction: str, gold: str) -> float:
    if _normalize_answer(prediction) == _normalize_answer(gold):
        return 1.0

    pred_answer = _extract_math_answer(prediction)
    gold_answer = _extract_math_answer(gold)
    if pred_answer and gold_answer:
        try:
            pred_value = float(pred_answer.replace(",", ""))
            gold_value = float(gold_answer.replace(",", ""))
            if abs(pred_value - gold_value) < 1e-6:
                return 1.0
            for variant in (gold_value, gold_value / 100, gold_value * 100):
                if math.isclose(pred_value, variant, rel_tol=1e-4):
                    return 1.0
        except (ValueError, OverflowError):
            pass
        if _normalize_answer(pred_answer) == _normalize_answer(gold_answer):
            return 1.0

    pred_math = _strip_math_string(pred_answer or prediction)
    gold_math = _strip_math_string(gold_answer or gold)
    if pred_math and gold_math and pred_math == gold_math:
        return 1.0
    if pred_answer and gold_answer and _symbolic_equal(pred_answer, gold_answer):
        return 1.0
    if len(prediction) < 50 and len(gold) < 50:
        if _symbolic_equal(prediction.strip(), gold.strip()):
            return 1.0
    return 0.0


# Verbatim prompt contract from OpenAI simple-evals healthbench_eval.py.  The
# scoring below likewise divides achieved signed points by all positive points.
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.
# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>
# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.
For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.
# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```
# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:
```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.
# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def _record_field(record: TaskRecord | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _metadata(record: TaskRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    value = _record_field(record, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _dataset_key(record: TaskRecord | Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    values = (
        metadata.get("dataset_key"),
        metadata.get("source"),
        _record_field(record, "dataset"),
        _record_field(record, "source"),
    )
    joined = " ".join(str(value).strip().lower() for value in values if value)
    if "hotpot" in joined:
        return "hotpotqa"
    if "trivia" in joined:
        return "triviaqa"
    if "aime" in joined:
        return "aime"
    if "healthbench" in joined:
        return "healthbench"
    if "webshop" in joined:
        return "webshop"
    if "alfworld" in joined:
        return "alfworld"
    if "swe" in joined and "bench" in joined:
        return "swe_bench"
    if "mbpp" in joined:
        return "mbpp_plus"
    return ""


def _evaluator_payload(record: TaskRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    value = _metadata(record).get("evaluator_payload", {})
    return value if isinstance(value, Mapping) else {}


def _accepted_answers(record: TaskRecord | Mapping[str, Any]) -> list[str]:
    payload = _evaluator_payload(record)
    raw = payload.get("accepted_answers")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        answers = [str(value) for value in raw if str(value).strip()]
        if answers:
            return answers
    ground_truth = _record_field(record, "ground_truth", "")
    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (str, bytes)):
        return [str(value) for value in ground_truth if str(value).strip()]
    text = str(ground_truth)
    if _dataset_key(record) == "triviaqa" and "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text] if text.strip() else []


def _invalid(
    reason: str,
    *,
    evaluator_version: str = UNAVAILABLE_EVALUATOR_VERSION,
    details: Optional[Mapping[str, Any]] = None,
) -> EvaluationOutcome:
    return EvaluationOutcome(
        valid=False,
        reward=None,
        reason=reason,
        details=dict(details or {}),
        evaluator_version=evaluator_version,
    )


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _evaluate_static(
    record: TaskRecord | Mapping[str, Any], prediction: str, dataset: str
) -> EvaluationOutcome:
    answers = _accepted_answers(record)
    evaluator_version = (
        HOTPOTQA_ANSWER_EVALUATOR_VERSION
        if dataset == "hotpotqa"
        else (
            TRIVIAQA_ANSWER_EVALUATOR_VERSION
            if dataset == "triviaqa"
            else SKILLFLOW_REWARD_VERSION
        )
    )
    if not answers:
        return _invalid(
            "missing_ground_truth",
            evaluator_version=evaluator_version,
        )
    # Both upstream implementations make ``<answer>...</answer>`` an explicit
    # final-answer boundary (FlowSteer's Format operator and SkillFlow's base
    # task prompt).  Prefer the last complete boundary when it is present, but
    # retain the historical raw-response behavior when it is absent.  This is
    # intentionally not a containment or free-form "answer is" heuristic.
    tagged_answers = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", prediction, re.IGNORECASE | re.DOTALL
    )
    scored_prediction = tagged_answers[-1].strip() if tagged_answers else prediction
    if dataset == "hotpotqa":
        token_f1 = max(
            _hotpotqa_answer_f1(scored_prediction, answer)
            for answer in answers
        )
        exact_match = max(
            float(
                _normalize_hotpotqa_answer(scored_prediction)
                == _normalize_hotpotqa_answer(answer)
            )
            for answer in answers
        )
        score = token_f1
        metrics = {"exact_match": exact_match, "token_f1": token_f1}
    elif dataset == "triviaqa":
        # SkillFlow Formal Protocol 10 uses maximum token F1 across accepted
        # answers and reports normalized exact match as the companion metric.
        token_f1 = max(
            _triviaqa_answer_f1(scored_prediction, answer) for answer in answers
        )
        exact_match = max(
            float(
                _normalize_triviaqa_answer(scored_prediction)
                == _normalize_triviaqa_answer(answer)
            )
            for answer in answers
        )
        score = token_f1
        metrics = {"exact_match": exact_match, "token_f1": token_f1}
    else:
        score = max(_exact_match(scored_prediction, answer) for answer in answers)
        metrics = {"exact_match": score}
    answer_mismatch_type = "none"
    if dataset == "triviaqa" and metrics.get("exact_match") != 1.0:
        normalized_prediction = _normalize_triviaqa_answer(scored_prediction)
        prediction_tokens = set(normalized_prediction.split())
        normalized_answers = tuple(
            _normalize_triviaqa_answer(answer) for answer in answers
        )
        accepted_token_sets = tuple(
            set(answer.split()) for answer in normalized_answers if answer
        )
        question = _record_field(record, "question")
        decade_alias_mismatch = any(
            verified_year_to_decade_normalization(
                original_question=question,
                source_value=scored_prediction,
                candidate_answer=answer,
            )
            or verified_year_to_decade_normalization(
                original_question=question,
                source_value=answer,
                candidate_answer=scored_prediction,
            )
            for answer in answers
        )
        if decade_alias_mismatch or (
            normalized_prediction and any(
            prediction_tokens < accepted_tokens
            or accepted_tokens < prediction_tokens
            for accepted_tokens in accepted_token_sets
            )
        ):
            # Evaluation-only diagnosis: accepted answers never enter the
            # Director, Agent, retrieval query, or Tool observation.  Official
            # EM/F1 above remain unchanged.
            answer_mismatch_type = "accepted_answer_canonicalization_mismatch"
        elif token_f1 > 0.0:
            answer_mismatch_type = "partial_answer_overlap"
        else:
            answer_mismatch_type = "no_accepted_answer_overlap"
    return EvaluationOutcome(
        valid=True,
        reward=score,
        metrics=metrics,
        reason="evaluated",
        details={
            "accepted_answer_count": len(answers),
            "raw_prediction": prediction,
            "scored_prediction": scored_prediction,
            "structured_answer_extracted": bool(tagged_answers),
            "metric_scope": (
                "answer_only"
                if dataset in {"hotpotqa", "triviaqa"}
                else "task_answer"
            ),
            **(
                {
                    "answer_mismatch_type": answer_mismatch_type,
                    "answer_mismatch_diagnostic_scope": "evaluator_only",
                }
                if dataset == "triviaqa"
                else {}
            ),
        },
        evaluator_version=evaluator_version,
    )


def _evaluate_aime2026(
    record: TaskRecord | Mapping[str, Any], prediction: str
) -> EvaluationOutcome:
    """Apply downstream SkillEval's strict INTEGER Accuracy scorer."""

    answers = _accepted_answers(record)
    if not answers:
        return _invalid(
            "missing_ground_truth",
            evaluator_version=AIME2026_EVALUATOR_VERSION,
        )
    try:
        result = score_aime2026_integer(prediction, answers)
    except (TypeError, ValueError) as exc:
        return _invalid(
            "aime_trusted_answer_invalid",
            evaluator_version=AIME2026_EVALUATOR_VERSION,
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
    return EvaluationOutcome(
        valid=True,
        reward=result.accuracy,
        metrics={"accuracy": result.accuracy},
        reason="evaluated",
        details={
            "accepted_answer_count": len(answers),
            "raw_prediction": result.raw_prediction,
            "scored_prediction": result.scored_prediction,
            "structured_answer_extracted": result.structured_answer_extracted,
            "canonical_prediction": result.canonical_prediction,
            "parsing_succeeded": result.parsing_succeeded,
            "parsing_failure_reason": result.parsing_failure_reason,
            "metric_scope": "canonicalized_integer_exact_accuracy",
        },
        evaluator_version=AIME2026_EVALUATOR_VERSION,
    )


def _health_conversation(
    record: TaskRecord | Mapping[str, Any], prediction: str
) -> str:
    question = str(_record_field(record, "question", "")).strip()
    if question.startswith("Conversation:"):
        question = question[len("Conversation:") :].strip()
    marker = re.compile(r"(?:^|\n\n)\[([^\]]+)\]\s*", re.MULTILINE)
    matches = list(marker.finditer(question))
    messages: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        content = question[match.end() : end].strip()
        if content:
            messages.append((match.group(1).strip().lower(), content))
    if not messages and question:
        messages.append(("user", question))
    messages.append(("assistant", prediction))
    return "\n\n".join(f"{role}: {content}" for role, content in messages)


def _rubric_text(item: Mapping[str, Any]) -> str:
    criterion = item.get("criterion", item.get("criterion_text", ""))
    return f"[{float(item['points']):g}] {str(criterion).strip()}"


def _json_from_text(text: str) -> Mapping[str, Any]:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def _judge_result(value: Any) -> tuple[Mapping[str, Any], Any]:
    if isinstance(value, Mapping) and "criteria_met" in value:
        return value, value
    text: Optional[str] = value if isinstance(value, str) else None
    if isinstance(value, Mapping):
        for key in ("response_text", "content", "text"):
            if isinstance(value.get(key), str):
                text = str(value[key])
                break
        if text is None:
            choices = value.get("choices")
            if isinstance(choices, Sequence) and choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    message = choice.get("message")
                    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                        text = str(message["content"])
    return (_json_from_text(text), text) if text is not None else ({}, value)


def _detail_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _detail_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_detail_value(item) for item in value]
    return repr(value)


async def _evaluate_healthbench(
    record: TaskRecord | Mapping[str, Any],
    prediction: str,
    *,
    judge: Optional[JudgeCallback],
    judge_model: str,
) -> EvaluationOutcome:
    if judge is None:
        return _invalid(
            "healthbench_judge_unavailable",
            evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
        )
    raw_items = _evaluator_payload(record).get("rubric_items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return _invalid(
            "healthbench_rubrics_missing",
            evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
        )

    rubric_items: list[Mapping[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            return _invalid(
                "healthbench_rubric_invalid",
                evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
            )
        criterion = str(item.get("criterion", item.get("criterion_text", ""))).strip()
        try:
            points = float(item["points"])
        except (KeyError, TypeError, ValueError):
            return _invalid(
                "healthbench_rubric_invalid",
                evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
            )
        if not criterion or not math.isfinite(points):
            return _invalid(
                "healthbench_rubric_invalid",
                evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
            )
        rubric_items.append({"criterion": criterion, "points": points})

    positive_total = sum(float(item["points"]) for item in rubric_items if item["points"] > 0)
    if positive_total <= 0:
        return _invalid(
            "healthbench_no_positive_rubric_points",
            evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
        )

    conversation = _health_conversation(record, prediction)

    async def grade(item: Mapping[str, Any]) -> Any:
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace(
            "<<rubric_item>>", _rubric_text(item)
        )
        result = judge([{"role": "user", "content": prompt}], judge_model)
        return await result if inspect.isawaitable(result) else result

    results = await asyncio.gather(
        *(grade(item) for item in rubric_items), return_exceptions=True
    )
    grades: list[Mapping[str, Any]] = []
    grade_details: list[dict[str, Any]] = []
    for item, result in zip(rubric_items, results, strict=True):
        if isinstance(result, BaseException):
            return _invalid(
                "healthbench_judge_error",
                evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
                details={"error_type": type(result).__name__, "error": str(result)},
            )
        parsed, raw = _judge_result(result)
        label = parsed.get("criteria_met")
        grade_details.append(
            {
                "criterion": item["criterion"],
                "points": item["points"],
                "criteria_met": label,
                "explanation": parsed.get("explanation", ""),
                "raw_judge_response": _detail_value(raw),
            }
        )
        if label is not True and label is not False:
            return _invalid(
                "healthbench_judge_response_invalid",
                evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
                details={"rubric_grades": grade_details, "judge_model": judge_model},
            )
        grades.append(parsed)

    achieved = sum(
        float(item["points"])
        for item, grade in zip(rubric_items, grades, strict=True)
        if grade["criteria_met"]
    )
    raw_score = achieved / positive_total
    reward = _clip_unit(raw_score)
    return EvaluationOutcome(
        valid=True,
        reward=reward,
        metrics={
            "raw_score": raw_score,
            "grpo_reward": reward,
            "rubric_count": float(len(rubric_items)),
        },
        reason="evaluated",
        details={
            "judge_model": judge_model,
            "achieved_points": achieved,
            "positive_possible_points": positive_total,
            "rubric_grades": grade_details,
        },
        evaluator_version=HEALTHBENCH_EVALUATOR_VERSION,
    )


def _load_ragen_module(path: Path) -> Any:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"RAGEN adapter not found: {source}")
    module_name = "_flowsteer_deployed_ragen_adapter"
    loaded = sys.modules.get(module_name)
    loaded_source = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_source and Path(str(loaded_source)).expanduser().resolve() == source:
        # Keep SkillFlow's process-local ALFWorld inventory and WebShop server
        # caches alive across evaluator calls.  Re-executing the deployed
        # module here would discard those upstream caches and reload the full
        # environment once per task.
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RAGEN adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _environment_config(
    record: TaskRecord | Mapping[str, Any], dataset: str
) -> tuple[str, dict[str, Any]]:
    metadata = _metadata(record)
    environment = metadata.get("environment", {})
    if isinstance(environment, Mapping):
        env_type = str(environment.get("env_type", dataset)).strip().lower()
        env_config = environment.get("env_config", {})
        if isinstance(env_config, Mapping):
            return env_type, dict(env_config)
    skillflow = metadata.get("skillflow", {})
    if isinstance(skillflow, Mapping):
        env_type = str(skillflow.get("env_type", dataset)).strip().lower()
        env_config = skillflow.get("env_config", {})
        if isinstance(env_config, Mapping):
            return env_type, dict(env_config)
    env_type = str(_record_field(record, "env_type", dataset)).strip().lower()
    env_config = _record_field(record, "env_config", {})
    return env_type, dict(env_config) if isinstance(env_config, Mapping) else {}


def _path_identity(value: Any) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))


def _webshop_instruction_matches(aligned_goal: str, runtime_instruction: str) -> bool:
    """Match the public human goal to WebShop's generated price constraint.

    The aligned source is WebShop's ``human_goals.json``.  Upstream
    ``web_agent_site.engine.goal.get_human_goals`` appends exactly one optional
    ``, and price lower than {price_upper:.2f} dollars`` suffix when it builds
    the live goal.  The goal index and full environment protocol are checked
    separately below, so accepting only that upstream suffix preserves task
    identity without requiring the data converter to reimplement goal
    generation.
    """

    aligned = " ".join(str(aligned_goal).split())
    runtime = " ".join(str(runtime_instruction).split())
    if not aligned or not runtime:
        return aligned == runtime
    if runtime == aligned:
        return True
    suffix = runtime[len(aligned) :] if runtime.startswith(aligned) else ""
    return bool(
        re.fullmatch(
            r", and price lower than \d+(?:\.\d{2}) dollars",
            suffix,
        )
    )


def _lock_alfworld_task(module: Any, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = config.get("game_file")
    if not requested:
        raise ValueError("ALFWorld env_config.game_file is required")
    if not hasattr(module, "AlfredEnvConfig") or not hasattr(module, "ALFWorldEnv"):
        raise RuntimeError("RAGEN adapter does not expose ALFWorld task inventory")

    alf_config = module.AlfredEnvConfig()
    if config.get("config_file"):
        alf_config.config_file = str(config["config_file"])
    inventory = module.ALFWorldEnv(config=alf_config, mode=str(config.get("mode", "train")))
    target = _path_identity(requested)
    game_files = list(getattr(inventory, "game_files", ()) or ())
    matching = [
        index for index, game_file in enumerate(game_files) if _path_identity(game_file) == target
    ]
    if len(matching) != 1:
        raise ValueError(
            f"ALFWorld requested game_file matched {len(matching)} inventory entries"
        )
    locked = dict(config)
    locked["seed"] = matching[0]
    return locked, {
        "requested_game_file": target,
        "locked_game_index": matching[0],
        "inventory_size": len(game_files),
    }


def _environment_task_description(
    record: TaskRecord | Mapping[str, Any],
    dataset: str,
    observation: str,
    adapter: Any,
) -> str:
    """Resolve the fixed task description using SkillFlow's ReAct boundary."""

    if dataset == "webshop" and " [SEP] " in observation:
        parts = observation.split(" [SEP] ")
        if len(parts) >= 3 and parts[2].strip():
            return parts[2].strip()
    if dataset == "alfworld" and "Your task is to:" in observation:
        return observation.split("Your task is to:", 1)[1].strip()

    env = getattr(adapter, "_env", None)
    authoritative_goal = getattr(env, "current_goal_instruction", "")
    if str(authoritative_goal).strip():
        return str(authoritative_goal).strip()

    metadata = _metadata(record)
    for container_name in ("skillflow", "extra"):
        container = metadata.get(container_name, {})
        if not isinstance(container, Mapping):
            continue
        extra = container.get("extra", container)
        if isinstance(extra, Mapping):
            for field_name in ("goal", "task", "task_description"):
                value = str(extra.get(field_name, "")).strip()
                if value:
                    return value
    return str(_record_field(record, "question", "")).strip()


def _environment_actions(
    dataset: str, available_actions: Any
) -> tuple[list[str], bool]:
    """Expand RAGEN actions exactly as SkillFlow's WebShop renderer does."""

    if dataset == "webshop" and isinstance(available_actions, Mapping):
        has_search_bar = bool(available_actions.get("has_search_bar"))
        actions = ["search[<your query>]"] if has_search_bar else []
        clickables = available_actions.get("clickables", ())
        if isinstance(clickables, Sequence) and not isinstance(clickables, (str, bytes)):
            actions.extend(f"click[{value}]" for value in clickables)
        return actions, has_search_bar
    if isinstance(available_actions, Sequence) and not isinstance(
        available_actions, (str, bytes)
    ):
        return [str(action) for action in available_actions], False
    return [], False


def _recent_environment_history(trace: Sequence[Mapping[str, Any]]) -> str:
    if not trace:
        return "(none)"
    lines = []
    for entry in trace[-4:]:
        lines.append(
            "[Step {step}: Observation: {observation!r}, Action: {action!r}, "
            "Result: {result!r}]".format(
                step=int(entry["step"]) + 1,
                observation=str(entry["observation"]),
                action=str(entry["action"]),
                result=str(
                    entry.get("feedback", entry.get("next_observation", ""))
                ),
            )
        )
    return "\n".join(lines)


# Copied verbatim from SkillFlow ``training/react_prompts.py``.  Keeping the
# upstream action-format block avoids adding a project-specific ALFWorld role
# or strategy while giving the Executor the same syntax examples it sees in
# SkillFlow's deployed ReAct path.
_ALFWORLD_ACTION_EXAMPLES = """Action format examples:
> go to cabinet 1
> take apple 1 from countertop 1
> open fridge 1
> move apple 1 to fridge 1
> heat apple 1 with microwave 1
> clean mug 1 with sinkbasin 1
> cool potato 1 with fridge 1
> move plate 1 to countertop 1
> examine shelf 1
"""


def _environment_prompt(
    *,
    dataset: str,
    task_description: str,
    observation: str,
    legal_actions: Sequence[str],
    trace: Sequence[Mapping[str, Any]],
    step_index: int,
) -> str:
    """Render the stateful subset of SkillFlow's WebShop/ALFWorld templates."""

    actions = "\n".join(legal_actions)
    history = _recent_environment_history(trace)
    if dataset == "webshop":
        return (
            "You are an expert autonomous agent operating in the WebShop "
            "e-commerce environment.\n"
            f"Your task is to: {task_description}.\n"
            f"Prior to this step, you have already taken {step_index} step(s). "
            "Below are the most recent observations and corresponding actions:\n"
            f"{history}\n"
            f"You are now at step {step_index + 1} and your current observation is: "
            f"{observation}.\n"
            "Your admissible actions of the current situation are:\n[\n"
            f"{actions}\n].\n\n"
            "Return exactly one executable action string in the form "
            "search[keywords] or click[value].\n"
            "For click actions, copy one value from the admissible action list "
            "exactly. You may instead enclose that one action in <action> tags."
        )
    return (
        "You are an expert agent operating in the ALFRED Embodied Environment.\n"
        f"{_ALFWORLD_ACTION_EXAMPLES}"
        f"Your task is to: {task_description}\n"
        f"Prior to this step, you have already taken {step_index} step(s). "
        "Below are the most recent observations and corresponding actions:\n"
        f"{history}\n"
        f"You are now at step {step_index + 1} and your current observation is: "
        f"{observation}\n"
        "Your admissible actions of the current situation are: [\n"
        f"{actions}\n].\n\n"
        "Pick exactly one action from the admissible actions list. Output only "
        "that action, or enclose that one action in <action> tags."
    )


def _parse_environment_action(
    output: Any,
    *,
    dataset: str,
    legal_actions: Sequence[str],
    webshop_has_search_bar: bool,
) -> Optional[str]:
    """Parse only an explicit tag or a complete legal raw response."""

    if not isinstance(output, str) or not output.strip():
        return None
    raw = output.strip()
    tagged = re.findall(r"<action>\s*(.*?)\s*</action>", raw, re.IGNORECASE | re.DOTALL)
    if len(tagged) > 1:
        return None
    candidate = tagged[0].strip() if tagged else raw
    if not candidate or (not tagged and candidate != raw):
        return None
    if dataset == "webshop" and candidate == "search[<your query>]":
        return None
    if candidate in legal_actions:
        return candidate
    if dataset == "webshop" and webshop_has_search_bar:
        match = re.fullmatch(r"search\[([^\[\]\n]+)\]", candidate)
        if match and match.group(1).strip() and match.group(1).strip() != "<your query>":
            return candidate
    return None


async def _evaluate_environment(
    record: TaskRecord | Mapping[str, Any],
    *,
    dataset: str,
    run_graph: Optional[RunGraphCallback],
    max_environment_steps: int,
    ragen_adapter_path: Path,
    replay_trace: Sequence[Mapping[str, Any]] = (),
) -> EvaluationOutcome:
    if isinstance(replay_trace, (str, bytes)) or not isinstance(
        replay_trace, Sequence
    ):
        return _invalid(
            "environment_replay_trace_invalid",
            evaluator_version=RAGEN_EVALUATOR_VERSION,
        )
    if run_graph is None and not replay_trace:
        return _invalid(
            "environment_graph_callback_unavailable",
            evaluator_version=RAGEN_EVALUATOR_VERSION,
        )
    try:
        module = _load_ragen_module(ragen_adapter_path)
        env_type, config = _environment_config(record, dataset)
        lock_details: dict[str, Any] = {}
        if dataset == "alfworld":
            config, lock_details = _lock_alfworld_task(module, config)
        elif dataset == "webshop":
            # SkillFlow exposes ``env_seed`` on WebShopEnv, but the deployed
            # adapter passes it to the constructor without seeding Python's
            # RNG before upstream ``get_human_goals`` samples price bounds.
            # Seed immediately before the synchronous reset so the live goal
            # order exactly matches the authoritative aligned catalog.
            check_webshop = getattr(module, "_check_webshop", None)
            if callable(check_webshop) and not bool(check_webshop()):
                raise RuntimeError("formal WebShop dependencies are unavailable")
            # Importing WebShop loads modules that mutate Python's global RNG.
            # Therefore seed only after the deployed dependency check has
            # completed, immediately before the first SimServer construction.
            random.seed(int(config.get("env_seed", 1000)))
        adapter = module.RAGENAdapter()
        observation = str(
            adapter.reset(
                env_type,
                config,
                question=str(_record_field(record, "question", "")),
                extra=dict(_metadata(record)),
            )
        )
    except Exception as exc:
        return _invalid(
            "environment_reset_failed",
            evaluator_version=RAGEN_EVALUATOR_VERSION,
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )

    if observation.startswith("[ENV_UNAVAILABLE]") or getattr(adapter, "_env", None) is None:
        return _invalid(
            "environment_unavailable",
            evaluator_version=RAGEN_EVALUATOR_VERSION,
            details={"observation": observation, **lock_details},
        )

    if dataset == "alfworld":
        actual = _path_identity(getattr(adapter._env, "current_game_file", ""))
        requested = str(lock_details["requested_game_file"])
        lock_details["actual_game_file"] = actual
        if actual != requested:
            return _invalid(
                "alfworld_task_lock_mismatch",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details=lock_details,
            )
        # SkillFlow's official ALFWorld bridge rejects a reset when the exact
        # public query differs from the canonical instruction returned by the
        # pinned game.  The RAGENAdapter is retained here, so reproduce only
        # that identity check at the AgentGraph record boundary.
        requested_instruction = str(_record_field(record, "question", "")).strip()
        actual_instruction = _environment_task_description(
            record, dataset, observation, adapter
        ).strip()
        lock_details.update(
            requested_instruction=requested_instruction,
            actual_instruction=actual_instruction,
        )
        if not requested_instruction or actual_instruction != requested_instruction:
            return _invalid(
                "alfworld_instruction_mismatch",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details=lock_details,
            )
        if "max_steps" in config:
            requested_max_steps = config["max_steps"]
            lock_details.update(
                requested_max_steps=requested_max_steps,
                evaluator_max_steps=max_environment_steps,
            )
            if (
                isinstance(requested_max_steps, bool)
                or not isinstance(requested_max_steps, int)
                or requested_max_steps != max_environment_steps
            ):
                return _invalid(
                    "alfworld_step_limit_mismatch",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                    details=lock_details,
                )
    elif dataset == "webshop":
        webshop_env = adapter._env
        if "goal_index" in config:
            actual_goal = getattr(webshop_env, "current_goal_index", None)
            lock_details.update(
                requested_goal_index=config["goal_index"],
                actual_goal_index=actual_goal,
            )
            try:
                goal_matches = (
                    actual_goal is not None
                    and int(actual_goal) == int(config["goal_index"])
                )
            except (TypeError, ValueError):
                goal_matches = False
            if not goal_matches:
                return _invalid(
                    "webshop_goal_lock_mismatch",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                    details=lock_details,
                )

        # SkillFlow's RAGENAdapter._reset_webshop passes these fields directly
        # into WebShopEnv, which retains them on the live environment.  Check
        # the requested protocol before accepting any terminal score so the
        # same goal index from a reduced or different catalog is not mistaken
        # for the aligned task.  This is an identity check only; it does not
        # inspect file contents.
        protocol_mismatches: dict[str, dict[str, Any]] = {}
        for field_name in (
            "human_goals",
            "use_small",
            "num_products",
            "goal_split",
            "file_path",
            "attr_path",
            "env_seed",
        ):
            if field_name not in config or not hasattr(webshop_env, field_name):
                continue
            requested_value = config[field_name]
            actual_value = getattr(webshop_env, field_name)
            if field_name in {"file_path", "attr_path"}:
                requested_value = _path_identity(requested_value)
                actual_value = _path_identity(actual_value)
            if actual_value != requested_value:
                protocol_mismatches[field_name] = {
                    "requested": requested_value,
                    "actual": actual_value,
                }

        skillflow = _metadata(record).get("skillflow", {})
        aligned_extra = (
            skillflow.get("extra", {}) if isinstance(skillflow, Mapping) else {}
        )
        requested_instruction = (
            str(aligned_extra.get("goal", "")).strip()
            if isinstance(aligned_extra, Mapping)
            else ""
        )
        actual_instruction = str(
            getattr(webshop_env, "current_goal_instruction", "")
        ).strip()
        if (
            requested_instruction
            and (
                not actual_instruction
                or not _webshop_instruction_matches(
                    requested_instruction,
                    actual_instruction,
                )
            )
        ):
            protocol_mismatches["goal_instruction"] = {
                "requested": requested_instruction,
                "actual": actual_instruction,
            }

        lock_details["webshop_protocol"] = {
            field_name: getattr(webshop_env, field_name)
            for field_name in (
                "human_goals",
                "use_small",
                "num_products",
                "goal_split",
                "file_path",
                "attr_path",
                "env_seed",
            )
            if hasattr(webshop_env, field_name)
        }
        if protocol_mismatches:
            lock_details["protocol_mismatches"] = protocol_mismatches
            return _invalid(
                "webshop_protocol_mismatch",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details=lock_details,
            )

    task_description = _environment_task_description(
        record, dataset, observation, adapter
    )
    trace: list[dict[str, Any]] = []
    terminal = False
    terminal_reward = 0.0
    terminal_info: Mapping[str, Any] = {}

    # An infrastructure failure can happen after several successful paid
    # environment-policy calls.  SkillFlow classifies that as an attempt
    # failure, not a task failure.  On evaluator retry, deterministically
    # replay the already accepted local environment actions and continue from
    # the first missing policy call instead of paying for the prefix again.
    for step_index, raw_entry in enumerate(replay_trace):
        if step_index >= max_environment_steps or not isinstance(raw_entry, Mapping):
            return _invalid(
                "environment_replay_trace_invalid",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"replay_step": step_index, **lock_details},
            )
        entry = dict(raw_entry)
        if (
            isinstance(entry.get("step"), bool)
            or not isinstance(entry.get("step"), int)
            or entry.get("step") != step_index
            or not isinstance(entry.get("observation"), str)
            or entry.get("observation") != observation
        ):
            return _invalid(
                "environment_replay_state_mismatch",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"replay_step": step_index, "trace": trace, **lock_details},
            )
        available_actions = getattr(adapter, "available_actions", ()) or ()
        legal_actions, webshop_has_search_bar = _environment_actions(
            dataset, available_actions
        )
        if (
            not legal_actions
            or entry.get("legal_actions") != _detail_value(legal_actions)
        ):
            return _invalid(
                "environment_replay_actions_mismatch",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"replay_step": step_index, "trace": trace, **lock_details},
            )

        action = entry.get("action")
        state_advanced = entry.get("state_advanced")
        if action == "<INVALID>" and state_advanced is False:
            stored_reward = entry.get("reward")
            if (
                entry.get("parse_error") is not True
                or entry.get("feedback")
                != "[INVALID] No valid <action> tag found."
                or (
                    "raw_graph_output" in entry
                    and _parse_environment_action(
                        entry.get("raw_graph_output"),
                        dataset=dataset,
                        legal_actions=legal_actions,
                        webshop_has_search_bar=webshop_has_search_bar,
                    )
                    is not None
                )
                or entry.get("next_observation") != observation
                or type(entry.get("done")) is not bool
                or entry.get("done") is not False
                or isinstance(stored_reward, bool)
                or not isinstance(stored_reward, (int, float))
                or not math.isfinite(float(stored_reward))
                or float(stored_reward) != 0.0
                or entry.get("info") != {"parse_error": True}
            ):
                return _invalid(
                    "environment_replay_parse_mismatch",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                    details={"replay_step": step_index, "trace": trace, **lock_details},
                )
            trace.append(entry)
            continue

        if (
            entry.get("parse_error") is True
            or state_advanced is False
            or (state_advanced is not None and state_advanced is not True)
            or not isinstance(action, str)
            or (
                "raw_graph_output" in entry
                and _parse_environment_action(
                    entry.get("raw_graph_output"),
                    dataset=dataset,
                    legal_actions=legal_actions,
                    webshop_has_search_bar=webshop_has_search_bar,
                )
                != action
            )
            or _parse_environment_action(
                action,
                dataset=dataset,
                legal_actions=legal_actions,
                webshop_has_search_bar=webshop_has_search_bar,
            )
            != action
        ):
            return _invalid(
                "environment_replay_action_invalid",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"replay_step": step_index, "trace": trace, **lock_details},
            )
        try:
            next_observation, raw_reward, done, info = adapter.step(action)
            reward_value = float(raw_reward)
        except Exception as exc:
            return _invalid(
                "environment_replay_step_failed",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={
                    "replay_step": step_index,
                    "trace": trace,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **lock_details,
                },
            )
        info_is_mapping = isinstance(info, Mapping)
        info = info if info_is_mapping else {}
        next_observation_text = str(next_observation)
        stored_reward_raw = entry.get("reward")
        stored_reward = (
            float(stored_reward_raw)
            if not isinstance(stored_reward_raw, bool)
            and isinstance(stored_reward_raw, (int, float))
            else math.nan
        )
        stored_done = entry.get("done")
        stored_info = entry.get("info")
        if (
            isinstance(raw_reward, bool)
            or not math.isfinite(reward_value)
            or not math.isfinite(stored_reward)
            or reward_value != stored_reward
            or type(done) is not bool
            or type(stored_done) is not bool
            or bool(done) != stored_done
            or not isinstance(next_observation, str)
            or not isinstance(entry.get("next_observation"), str)
            or next_observation_text != entry.get("next_observation")
            or not info_is_mapping
            or not isinstance(stored_info, Mapping)
            or _detail_value(info) != _detail_value(stored_info)
        ):
            return _invalid(
                "environment_replay_transition_mismatch",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"replay_step": step_index, "trace": trace, **lock_details},
            )
        if info.get("error") or next_observation_text.startswith("[ENV_UNAVAILABLE]"):
            return _invalid(
                "environment_replay_transition_invalid",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"replay_step": step_index, "trace": trace, **lock_details},
            )
        trace.append(entry)
        observation = next_observation_text
        terminal_reward = reward_value
        terminal_info = info
        if bool(done):
            terminal = True
            if step_index + 1 != len(replay_trace):
                return _invalid(
                    "environment_replay_trace_after_terminal",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                    details={"replay_step": step_index, "trace": trace, **lock_details},
                )
            break

    if trace:
        lock_details["replayed_environment_steps"] = len(trace)

    if not terminal and len(trace) < max_environment_steps and run_graph is None:
        return _invalid(
            "environment_graph_callback_unavailable",
            evaluator_version=RAGEN_EVALUATOR_VERSION,
            details={"trace": trace, **lock_details},
        )

    for step_index in range(len(trace), max_environment_steps):
        if terminal:
            break
        available_actions = getattr(adapter, "available_actions", ()) or ()
        legal_actions, webshop_has_search_bar = _environment_actions(
            dataset, available_actions
        )
        if not legal_actions:
            return _invalid(
                "environment_has_no_legal_actions",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"trace": trace, "observation": observation, **lock_details},
            )
        prompt = _environment_prompt(
            dataset=dataset,
            task_description=task_description,
            observation=observation,
            legal_actions=legal_actions,
            trace=trace,
            step_index=step_index,
        )
        try:
            callback_result = run_graph(prompt)
            raw_action = (
                await callback_result
                if inspect.isawaitable(callback_result)
                else callback_result
            )
        except Exception as exc:
            return _invalid(
                "environment_graph_callback_failed",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={
                    "trace": trace,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **lock_details,
                },
            )
        action = _parse_environment_action(
            raw_action,
            dataset=dataset,
            legal_actions=legal_actions,
            webshop_has_search_bar=webshop_has_search_bar,
        )
        if action is None:
            # SkillFlow ``GenericTaskEnvironment._react_step`` treats a parse
            # miss as a zero-reward, non-terminal turn: it records the failed
            # turn, leaves the RAGEN state untouched, and lets the policy try
            # again until the episode budget is exhausted.  ``run_graph`` is a
            # stateless callback here, so retain the same parse feedback in the
            # local trace and render it into the next callback prompt.
            feedback = "[INVALID] No valid <action> tag found."
            trace.append(
                {
                    "step": step_index,
                    "observation": observation,
                    "legal_actions": _detail_value(legal_actions),
                    "action": "<INVALID>",
                    "raw_graph_output": _detail_value(raw_action),
                    "next_observation": observation,
                    "feedback": feedback,
                    "reward": 0.0,
                    "done": False,
                    "state_advanced": False,
                    "parse_error": True,
                    "info": {"parse_error": True},
                }
            )
            terminal_reward = 0.0
            terminal_info = {}
            continue
        try:
            next_observation, raw_reward, done, info = adapter.step(action)
            reward_value = float(raw_reward)
        except Exception as exc:
            return _invalid(
                "environment_step_failed",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={
                    "trace": trace,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **lock_details,
                },
            )
        if not math.isfinite(reward_value):
            return _invalid(
                "environment_reward_invalid",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"trace": trace, **lock_details},
            )
        info = info if isinstance(info, Mapping) else {}
        next_observation_text = str(next_observation)
        trace.append(
            {
                "step": step_index,
                "observation": observation,
                "legal_actions": _detail_value(legal_actions),
                "action": action,
                "raw_graph_output": _detail_value(raw_action),
                "next_observation": next_observation_text,
                "reward": reward_value,
                "done": bool(done),
                "info": _detail_value(info),
            }
        )
        if info.get("error"):
            return _invalid(
                "environment_step_failed",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"trace": trace, **lock_details},
            )
        # RAGENAdapter.step uses this exact sentinel when its live environment
        # is absent.  Upstream returns an empty info mapping on that path, so it
        # must be checked explicitly rather than accepted as a task failure.
        if next_observation_text.startswith("[ENV_UNAVAILABLE]"):
            return _invalid(
                "environment_unavailable",
                evaluator_version=RAGEN_EVALUATOR_VERSION,
                details={"trace": trace, **lock_details},
            )
        observation = next_observation_text
        terminal_reward = reward_value
        terminal_info = info
        if bool(done):
            terminal = True
            break

    if dataset == "alfworld":
        # SkillFlow's private ALFWorld terminal evaluator reads the boolean
        # official goal predicate only after a simulator terminal.  A missing
        # or non-boolean predicate is infrastructure/evaluator failure; reward
        # magnitude is never a substitute.  Exhausting the public step budget
        # without a simulator terminal remains the official no-submission
        # failure (valid zero), matching SkillFlow's terminal-input boundary.
        if terminal:
            won = terminal_info.get("won")
            if type(won) is not bool:
                return _invalid(
                    "alfworld_terminal_success_unavailable",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                    details={
                        "env_type": env_type,
                        "terminal_observation": observation,
                        "terminal_info": _detail_value(terminal_info),
                        "trace": trace,
                        **lock_details,
                    },
                )
            success = won
        else:
            success = False
        reward = 1.0 if success else 0.0
    else:
        reward = _clip_unit(terminal_reward)
        success = reward >= 1.0
    return EvaluationOutcome(
        valid=True,
        reward=reward,
        metrics={
            "success": float(success),
            "environment_return": terminal_reward,
            "steps": float(len(trace)),
            "terminal": float(terminal),
        },
        reason="evaluated" if terminal else "environment_step_limit",
        details={
            "env_type": env_type,
            "terminal_observation": observation,
            "terminal_info": _detail_value(terminal_info),
            "trace": trace,
            **lock_details,
        },
        evaluator_version=RAGEN_EVALUATOR_VERSION,
    )


async def _evaluate_swebench(
    record: TaskRecord | Mapping[str, Any],
    prediction: str,
    *,
    swe_harness: Optional[SWEHarnessCallback],
) -> EvaluationOutcome:
    if swe_harness is None:
        return _invalid(
            "swebench_harness_unavailable",
            evaluator_version=SWEBENCH_EVALUATOR_VERSION,
            details={"proxy_similarity_used": False},
        )
    try:
        callback_result = swe_harness(record, prediction)
        result = await callback_result if inspect.isawaitable(callback_result) else callback_result
    except Exception as exc:
        details: dict[str, Any] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        diagnostic = getattr(exc, "diagnostic", None)
        if diagnostic is not None:
            if isinstance(diagnostic, Mapping):
                diagnostic_value = diagnostic.get
            else:
                diagnostic_value = lambda name, default=None: getattr(  # noqa: E731
                    diagnostic, name, default
                )

            for field_name in ("classification", "phase"):
                value = diagnostic_value(field_name)
                if isinstance(value, str) and value:
                    details[field_name] = value
            for field_name in (
                "retryable",
                "test_output_present",
                "report_present",
            ):
                value = diagnostic_value(field_name)
                if isinstance(value, bool):
                    details[field_name] = value
            container_exit_code = diagnostic_value("container_exit_code")
            if isinstance(container_exit_code, int) and not isinstance(
                container_exit_code, bool
            ):
                details["container_exit_code"] = container_exit_code
            oom_killed = diagnostic_value(
                "oom_killed", diagnostic_value("container_oom_killed")
            )
            if isinstance(oom_killed, bool):
                details["oom_killed"] = oom_killed
        return _invalid(
            "swebench_harness_failed",
            evaluator_version=SWEBENCH_EVALUATOR_VERSION,
            details=details,
        )

    details: dict[str, Any]
    if isinstance(result, bool):
        resolved = result
        details = {"resolved": resolved}
    elif isinstance(result, Mapping) and isinstance(result.get("resolved"), bool):
        resolved = bool(result["resolved"])
        details = _detail_value(result)
    else:
        return _invalid(
            "swebench_harness_result_invalid",
            evaluator_version=SWEBENCH_EVALUATOR_VERSION,
            details={"harness_result": _detail_value(result)},
        )
    reward = 1.0 if resolved else 0.0
    return EvaluationOutcome(
        valid=True,
        reward=reward,
        metrics={"resolved": reward},
        reason="evaluated",
        details=details,
        evaluator_version=SWEBENCH_EVALUATOR_VERSION,
    )


async def _evaluate_mbpp_plus(
    record: TaskRecord | Mapping[str, Any],
    prediction: str,
    *,
    mbppplus_harness: Optional[MBPPPlusHarnessCallback],
) -> EvaluationOutcome:
    if mbppplus_harness is None:
        return _invalid(
            "mbppplus_harness_unavailable",
            evaluator_version=MBPPPLUS_EVALUATOR_VERSION,
            details={"proxy_evaluator_used": False},
        )
    try:
        callback_result = mbppplus_harness(record, prediction)
        result = (
            await callback_result
            if inspect.isawaitable(callback_result)
            else callback_result
        )
    except Exception as exc:
        return _invalid(
            "mbppplus_harness_failed",
            evaluator_version=MBPPPLUS_EVALUATOR_VERSION,
            details={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    if not isinstance(result, Mapping):
        return _invalid(
            "mbppplus_harness_result_invalid",
            evaluator_version=MBPPPLUS_EVALUATOR_VERSION,
        )
    base_status = result.get("base_status")
    plus_status = result.get("plus_status")
    base_passed = result.get("base_passed")
    plus_passed = result.get("plus_passed")
    pass_at_1 = result.get("pass_at_1")
    official_statuses = {"pass", "fail", "timeout"}
    if (
        base_status not in official_statuses
        or plus_status not in official_statuses
        or type(base_passed) is not bool
        or type(plus_passed) is not bool
        or isinstance(pass_at_1, bool)
        or not isinstance(pass_at_1, (int, float))
        or not math.isfinite(float(pass_at_1))
    ):
        return _invalid(
            "mbppplus_harness_result_invalid",
            evaluator_version=MBPPPLUS_EVALUATOR_VERSION,
        )
    expected_base_passed = base_status == "pass"
    expected_plus_passed = plus_status == "pass"
    expected_pass_at_1 = float(expected_plus_passed)
    if (
        base_passed != expected_base_passed
        or plus_passed != expected_plus_passed
        or float(pass_at_1) != expected_pass_at_1
    ):
        return _invalid(
            "mbppplus_harness_result_inconsistent",
            evaluator_version=MBPPPLUS_EVALUATOR_VERSION,
        )

    raw_diagnostics = result.get("format_diagnostics", {})
    format_diagnostics: dict[str, Any] = {}
    if isinstance(raw_diagnostics, Mapping):
        for key in (
            "raw_prediction_empty",
            "submission_character_count",
            "submission_transformation",
            "entry_point",
        ):
            if key in raw_diagnostics:
                format_diagnostics[key] = _detail_value(raw_diagnostics[key])
    details = {
        "task_id": str(result.get("task_id", "")),
        "base_status": base_status,
        "plus_status": plus_status,
        "base_passed": base_passed,
        "plus_passed": plus_passed,
        "format_diagnostics": format_diagnostics,
        "evaluator_protocol": str(result.get("evaluator_protocol", "")),
        "runtime_version": str(result.get("runtime_version", "")),
        "dataset_version": str(result.get("dataset_version", "")),
    }
    return EvaluationOutcome(
        valid=True,
        reward=expected_pass_at_1,
        metrics={
            "base_pass_at_1": float(base_passed),
            "pass_at_1": expected_pass_at_1,
        },
        reason="evaluated",
        details=details,
        evaluator_version=MBPPPLUS_EVALUATOR_VERSION,
    )


async def evaluate_task(
    record: TaskRecord | Mapping[str, Any],
    prediction: str,
    *,
    judge: Optional[JudgeCallback] = None,
    judge_model: str = "",
    run_graph: Optional[RunGraphCallback] = None,
    swe_harness: Optional[SWEHarnessCallback] = None,
    mbppplus_harness: Optional[MBPPPlusHarnessCallback] = None,
    max_environment_steps: int = 50,
    ragen_adapter_path: str | Path = DEFAULT_RAGEN_ADAPTER_PATH,
    environment_replay_trace: Sequence[Mapping[str, Any]] = (),
) -> EvaluationOutcome:
    """Evaluate one final answer without manufacturing unavailable rewards.

    ``run_graph`` receives one short environment prompt per step and must return
    exactly one action string.  ``judge`` receives the official HealthBench
    grader message list and the selected judge model name.  SWE-bench remains
    invalid unless a real harness callback explicitly reports ``resolved``;
    MBPP+ remains invalid unless the official EvalPlus callback reports its
    base and plus statuses.
    ``environment_replay_trace`` may contain a prior WebShop/ALFWorld evaluator
    prefix: the environment is reset and each recorded transition is checked
    exactly before ``run_graph`` is called for the first missing step.
    """

    if max_environment_steps <= 0:
        raise ValueError("max_environment_steps must be positive")
    dataset = _dataset_key(record)
    if dataset in {"hotpotqa", "triviaqa"}:
        return _evaluate_static(record, str(prediction), dataset)
    if dataset == "aime":
        return _evaluate_aime2026(record, str(prediction))
    if dataset == "healthbench":
        return await _evaluate_healthbench(
            record,
            str(prediction),
            judge=judge,
            judge_model=judge_model,
        )
    if dataset in {"webshop", "alfworld"}:
        return await _evaluate_environment(
            record,
            dataset=dataset,
            run_graph=run_graph,
            max_environment_steps=max_environment_steps,
            ragen_adapter_path=Path(ragen_adapter_path),
            replay_trace=environment_replay_trace,
        )
    if dataset == "swe_bench":
        return await _evaluate_swebench(record, str(prediction), swe_harness=swe_harness)
    if dataset == "mbpp_plus":
        return await _evaluate_mbpp_plus(
            record,
            str(prediction),
            mbppplus_harness=mbppplus_harness,
        )
    return _invalid("unsupported_dataset", details={"dataset_key": dataset})


__all__ = [
    "DEFAULT_RAGEN_ADAPTER_PATH",
    "EvaluationOutcome",
    "GRADER_TEMPLATE",
    "AIME2026_EVALUATOR_VERSION",
    "HOTPOTQA_ANSWER_EVALUATOR_VERSION",
    "MBPPPLUS_EVALUATOR_VERSION",
    "MBPPPlusHarnessCallback",
    "SWEHarnessCallback",
    "TRIVIAQA_ANSWER_EVALUATOR_VERSION",
    "evaluate_task",
]
