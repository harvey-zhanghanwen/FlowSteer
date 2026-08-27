#!/usr/bin/env python3
"""Materialize semantic-preserving TriviaQA train-question paraphrases.

Split consistency is checked before the local Qwen3.5 OpenAI-compatible
endpoint is contacted.  The model generates only ``paraphrase_question``.
The answer statement is rendered deterministically from the frozen train
``accepted_answers[0]`` canonical span by :mod:`triviaqa_qa_memory`.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    TriviaQAQAMemoryRecord,
    TriviaQATrainSource,
    load_materialized_qa_memory,
    load_triviaqa_qa_memory_sources,
    validate_qa_memory_against_sources,
    write_materialized_qa_memory,
)


PROMPT_TEMPLATE_VERSION = "triviaqa.qa_memory.question_paraphrase.v3"
PARAPHRASE_METHOD = "semantic-preserving-question-paraphrase"
GENERATOR_PROVIDER = "local-openai-compatible"

SYSTEM_PROMPT = """Paraphrase one TriviaQA training question.
Preserve the exact entity identity, requested relation, answer type, temporal or geographic scope, and every constraint. Do not answer the question, add facts, broaden or narrow its meaning. Change the wording or syntax; do not copy the original question unchanged.
Return exactly one JSON object with this schema and no other text:
{"paraphrase_question":"..."}"""


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

    # The canonical span is intentionally absent from the model request.  It
    # is used only by the local validator and by deterministic answer-statement
    # rendering after the question paraphrase has been accepted.
    user_payload = {"original_question": source.original_question}
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


def _canonical_is_original_spelling_variant(
    original_question: str,
    canonical_answer: str,
) -> bool:
    canonical_tokens = canonical_answer.casefold().split()
    original_tokens = original_question.casefold().split()
    width = len(canonical_tokens)
    if width < 1 or len(original_tokens) < width:
        return False
    canonical = " ".join(canonical_tokens)
    return any(
        SequenceMatcher(
            None,
            canonical,
            " ".join(original_tokens[start : start + width]),
        ).ratio()
        >= 0.9
        for start in range(len(original_tokens) - width + 1)
    )


def parse_paraphrase_response(text: str, source: TriviaQATrainSource) -> str:
    """Accept only the declared one-field JSON response."""

    try:
        value = json.loads(str(text).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("paraphrase response is not strict JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"paraphrase_question"}:
        raise ValueError("paraphrase response fields are incompatible")
    question = value["paraphrase_question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("paraphrase_question must be non-empty text")
    question = " ".join(question.split())
    if question.casefold() == " ".join(source.original_question.split()).casefold():
        raise ValueError("model returned the original question unchanged")
    canonical = source.canonical_answer.casefold()
    if (
        canonical not in source.original_question.casefold()
        and canonical in question.casefold()
        and not _canonical_is_original_spelling_variant(
            source.original_question,
            source.canonical_answer,
        )
    ):
        raise ValueError("paraphrase_question introduced the canonical answer")
    return question


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

    def generate(self, source: TriviaQATrainSource, *, seed: int) -> tuple[str, int]:
        payload = {
            "model": self.model_id,
            "messages": build_paraphrase_messages(source),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 128,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                attempt_payload = dict(payload)
                attempt_payload["seed"] = seed + attempt
                if attempt:
                    attempt_payload["temperature"] = 0.1
                    retry_instruction = (
                        "Use a different grammatical construction, such as an "
                        "indirect request beginning with Identify, Name, or State, "
                        "while preserving the question exactly."
                        if attempt == 1
                        else "Reorder the clauses and replace at least one non-entity "
                        "verb or phrase with an equivalent expression."
                    )
                    attempt_payload["messages"] = payload["messages"] + [
                        {
                            "role": "user",
                            "content": (
                                "The prior response did not satisfy the declared "
                                "JSON or semantic-preservation contract. Return a "
                                "different surface form now, with the same entity, "
                                "relation, scope, constraints, and answer type. "
                                + retry_instruction
                            ),
                        }
                    ]
                request = Request(
                    self.endpoint,
                    data=json.dumps(attempt_payload, ensure_ascii=False).encode(
                        "utf-8"
                    ),
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
                return parse_paraphrase_response(content, source), seed + attempt
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
            else type(last_error).__name__
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--model-id", default="supervisor_theta")
    parser.add_argument("--api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--paraphrase-version", required=True)
    parser.add_argument("--base-seed", type=_nonnegative_integer, default=20260827)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=_nonnegative_integer, default=2)
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
        sources, _validation_ids = load_triviaqa_qa_memory_sources(
            args.train_tasks,
            args.validation_tasks,
            expected_train_count=args.expected_train_count,
            expected_validation_count=args.expected_validation_count,
        )
        output_path = Path(args.output)
        existing = (
            load_materialized_qa_memory(output_path)
            if output_path.is_file()
            else ()
        )
        validate_qa_memory_against_sources(
            existing,
            sources,
            require_complete=False,
        )
        for record in existing:
            admitted_seeds = range(
                args.base_seed + record.selection_index,
                args.base_seed + record.selection_index + args.max_retries + 1,
            )
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
        for source in sources:
            if source.source_train_task_id in records:
                continue
            seed = args.base_seed + source.selection_index
            paraphrase, accepted_seed = client.generate(source, seed=seed)
            records[source.source_train_task_id] = TriviaQAQAMemoryRecord.create(
                source=source,
                paraphrase_question=paraphrase,
                paraphrase_version=args.paraphrase_version,
                paraphrase_method=PARAPHRASE_METHOD,
                generator_provider=GENERATOR_PROVIDER,
                model_id=args.model_id,
                model_revision=args.model_revision,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                generation_seed=accepted_seed,
            )
            # Atomic incremental checkpointing prevents repeated local model
            # calls when materialization is resumed after an interruption.
            write_materialized_qa_memory(output_path, tuple(records.values()))
        completed = tuple(records.values())
        validate_qa_memory_against_sources(
            completed,
            sources,
            require_complete=True,
        )
        write_materialized_qa_memory(output_path, completed)
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"TriviaQA QA-memory paraphrase generation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": "flowsteer.triviaqa.qa_memory.materialization.v1",
                "output": str(output_path.resolve()),
                "record_count": len(completed),
                "unique_source_count": len(
                    {record.base_task_id for record in completed}
                ),
                "cycled_count": sum(
                    record.cycled_training_sample for record in completed
                ),
                "paraphrase_version": args.paraphrase_version,
                "validation_content_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
