"""HealthBench Professional public-input and private-evaluator boundary.

The public dataset release stores model-visible ``conversation.messages`` and
evaluator-only rubrics in one JSONL row.  FlowSteer's ``TaskRecord`` transport
is a text boundary, so this adapter supplies a reversible JSON rendering for
the conversation and a task-id join to a separate private evaluator file.

This module does not define an Agent role, topology, Tool, medical workflow,
or scoring policy.  It only validates and separates the official public
HealthBench Professional row schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


HEALTHBENCH_PROFESSIONAL_DATASET_KEY = "healthbench_professional"
HEALTHBENCH_PROFESSIONAL_TASK_FAMILY = (
    "healthbench-professional/conversation-response"
)
HEALTHBENCH_PROFESSIONAL_SOURCE_ID = "openai/healthbench-professional"
HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT = "test"
HEALTHBENCH_PROFESSIONAL_PUBLIC_COUNT = 525
HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE = (
    "healthbench-professional/rubric-grader"
)
HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION = (
    "openai-simple-evals-healthbench-professional-652c89d@1"
)
HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION = (
    "flowsteer.healthbench-professional.evaluator-case.v1"
)

_CONVERSATION_HEADER = (
    "Conversation messages (respond to the final user message):\n"
)
_OFFICIAL_ROW_FIELDS = {
    "canary_string",
    "conversation",
    "difficulty",
    "id",
    "physician_response",
    "rubric_items",
    "specialty",
    "type",
    "use_case",
}
_SLICE_FIELDS = ("use_case", "type", "difficulty", "specialty")


def _validated_messages(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("HealthBench conversation.messages must be a list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(
                "HealthBench message "
                f"{index} must contain exactly role and content"
            )
        role = message["role"]
        content = message["content"]
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"HealthBench message {index} has unsupported role {role!r}"
            )
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"HealthBench message {index} content must be non-empty text"
            )
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("HealthBench conversation must contain at least one message")
    if messages[-1]["role"] != "user":
        raise ValueError("HealthBench Professional conversation must end with user")
    return tuple(messages)


def render_model_visible_conversation(
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Serialize model-visible messages without adding evaluator information.

    JSON is used because message content may itself contain newlines, quotes,
    brackets, or tag-like text.  The fixed ordinary-language header remains
    readable by the Director, while ``json.loads`` makes the boundary exactly
    reversible for native chat-message execution.
    """

    validated = _validated_messages(messages)
    payload = {"messages": list(validated)}
    return _CONVERSATION_HEADER + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_model_visible_conversation(question: str) -> tuple[dict[str, str], ...]:
    """Recover native role/content messages from one rendered TaskRecord."""

    if not isinstance(question, str) or not question.startswith(_CONVERSATION_HEADER):
        raise ValueError("not a HealthBench Professional conversation rendering")
    serialized = question[len(_CONVERSATION_HEADER) :]
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid HealthBench conversation JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"messages"}:
        raise ValueError("HealthBench conversation payload must contain only messages")
    return _validated_messages(payload["messages"])


def validate_official_healthbench_professional_row(
    row: Mapping[str, Any],
    *,
    line_number: int | None = None,
) -> dict[str, Any]:
    """Validate one row of the official 525-example public test release."""

    location = f" at line {line_number}" if line_number is not None else ""
    if set(row) != _OFFICIAL_ROW_FIELDS:
        raise ValueError(
            "HealthBench Professional row fields differ from the official schema"
            f"{location}"
        )
    source_id = row["id"]
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError(f"HealthBench id must be non-empty text{location}")
    conversation = row["conversation"]
    if not isinstance(conversation, Mapping) or set(conversation) != {"messages"}:
        raise ValueError(
            "HealthBench conversation must contain exactly messages" + location
        )
    messages = _validated_messages(conversation["messages"])

    rubrics = row["rubric_items"]
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError(f"HealthBench rubric_items must be a non-empty list{location}")
    normalized_rubrics: list[dict[str, object]] = []
    for rubric_index, rubric in enumerate(rubrics):
        if not isinstance(rubric, Mapping) or set(rubric) != {
            "criterion_text",
            "points",
        }:
            raise ValueError(
                "HealthBench rubric item must contain exactly criterion_text and "
                f"points{location}, rubric {rubric_index}"
            )
        criterion = rubric["criterion_text"]
        points = rubric["points"]
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(
                f"HealthBench rubric criterion must be non-empty{location}"
            )
        if isinstance(points, bool) or not isinstance(points, int) or points == 0:
            raise ValueError(
                f"HealthBench rubric points must be a non-zero integer{location}"
            )
        normalized_rubrics.append(
            {"criterion_text": criterion, "points": points}
        )

    physician_response = row["physician_response"]
    canary_string = row["canary_string"]
    if not isinstance(physician_response, str) or not physician_response:
        raise ValueError(
            f"HealthBench physician_response must be non-empty text{location}"
        )
    if not isinstance(canary_string, str) or not canary_string:
        raise ValueError(f"HealthBench canary_string must be non-empty text{location}")
    slices: dict[str, str] = {}
    for field_name in _SLICE_FIELDS:
        value = row[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"HealthBench {field_name} must be non-empty text{location}"
            )
        slices[field_name] = value

    return {
        "id": source_id,
        "conversation": {"messages": list(messages)},
        "rubric_items": normalized_rubrics,
        "physician_response": physician_response,
        "canary_string": canary_string,
        **slices,
    }


def public_task_record_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that may cross the model-visible TaskRecord boundary."""

    validated = validate_official_healthbench_professional_row(row)
    source_id = validated["id"]
    task_id = f"healthbench-professional:{source_id}"
    messages = validated["conversation"]["messages"]
    return {
        "task_id": task_id,
        "source_id": source_id,
        "question": render_model_visible_conversation(messages),
        "conversation": {"messages": messages},
        "evaluator_route": HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE,
    }


def evaluator_case_from_official_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the evaluator-only case joined to the public record by task_id."""

    validated = validate_official_healthbench_professional_row(row)
    source_id = validated["id"]
    return {
        "schema_version": HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION,
        "task_id": f"healthbench-professional:{source_id}",
        "source_id": source_id,
        "evaluator_route": HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE,
        # The reference grader needs the same public conversation to construct
        # its rubric prompt. Keeping a copy in the private join lets the public
        # caller cross the evaluator boundary with task_id + candidate only.
        "prompt": validated["conversation"]["messages"],
        "rubric_items": validated["rubric_items"],
        "physician_response": validated["physician_response"],
        "evaluator_metadata": {
            "use_case": validated["use_case"],
            "type": validated["type"],
            "difficulty": validated["difficulty"],
            "specialty": validated["specialty"],
            "canary_string": validated["canary_string"],
        },
    }


def load_healthbench_professional_evaluator_cases(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load evaluator-only cases and enforce a unique task-id join."""

    source = Path(path)
    result: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected an object")
            if value.get("schema_version") != (
                HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{source}:{line_number}: unsupported evaluator case schema"
                )
            task_id = value.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{source}:{line_number}: invalid task_id")
            if task_id in result:
                raise ValueError(f"{source}:{line_number}: duplicate task_id {task_id}")
            result[task_id] = value
    return result


__all__ = [
    "HEALTHBENCH_PROFESSIONAL_DATASET_KEY",
    "HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE",
    "HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION",
    "HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION",
    "HEALTHBENCH_PROFESSIONAL_PUBLIC_COUNT",
    "HEALTHBENCH_PROFESSIONAL_SOURCE_ID",
    "HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT",
    "HEALTHBENCH_PROFESSIONAL_TASK_FAMILY",
    "evaluator_case_from_official_row",
    "load_healthbench_professional_evaluator_cases",
    "parse_model_visible_conversation",
    "public_task_record_fields",
    "render_model_visible_conversation",
    "validate_official_healthbench_professional_row",
]
