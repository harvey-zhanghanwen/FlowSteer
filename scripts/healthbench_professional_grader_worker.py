"""Isolated pinned simple-evals worker for HealthBench Professional.

The process receives evaluator-only prompt/rubrics over stdin, grades each
rubric through OpenAI's Responses API, and emits one bounded JSON receipt.
No benchmark evaluator payload is placed on the model-generation interface.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import importlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping


GRADER_MODEL = "gpt-5.4-2026-03-05"
REASONING_EFFORT = "low"
LENGTH_CENTER = 2000.0
LENGTH_PENALTY_PER_500_CHARS = 0.0147
EVALUATOR_VERSION = "openai-simple-evals-healthbench-professional-652c89d@1"


def _absolute_source_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not (path / "healthbench_eval.py").is_file():
        raise ValueError("pinned simple-evals HealthBench source is absent")
    return path.resolve()


def _request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "candidate_answer",
        "operation",
        "private_case",
        "task_id",
    }:
        raise ValueError("HealthBench worker request fields differ")
    if value["operation"] not in {"grade", "preflight"}:
        raise ValueError("HealthBench worker operation is unsupported")
    if type(value["task_id"]) is not str or not value["task_id"].strip():
        raise ValueError("HealthBench worker task_id is invalid")
    if (
        type(value["candidate_answer"]) is not str
        or not value["candidate_answer"].strip()
    ):
        raise ValueError("HealthBench candidate response is invalid")
    if not isinstance(value["private_case"], dict):
        raise ValueError("HealthBench private case must be an object")
    return value


def _messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("HealthBench private prompt must be non-empty")
    messages: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"content", "role"}:
            raise ValueError("HealthBench private prompt message fields differ")
        if type(raw["role"]) is not str or type(raw["content"]) is not str:
            raise ValueError("HealthBench private prompt message is invalid")
        messages.append({"role": raw["role"], "content": raw["content"]})
    return messages


def _load_official(source_root: Path) -> tuple[Any, Any, type[Any]]:
    # The official repository is an implicit namespace package. importlib can
    # load its checkout directory name (including a hyphen) while retaining
    # the relative imports used by healthbench_eval.py.
    sys.path.insert(0, str(source_root.parent))
    package = source_root.name
    module = importlib.import_module(f"{package}.healthbench_eval")
    types_module = importlib.import_module(f"{package}.types")
    return module.HealthBenchEval, module.RubricItem, types_module.SamplerResponse


def _rubrics(value: object, rubric_type: type[Any]) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError("HealthBench private rubrics must be non-empty")
    rubrics: list[Any] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"criterion", "points", "tags"}:
            raise ValueError("HealthBench private rubric fields differ")
        rubrics.append(rubric_type.from_dict(item))
    return rubrics


def _model_dump(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    method = getattr(value, "model_dump", None)
    if callable(method):
        dumped = method()
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _usage(value: object) -> dict[str, int | None]:
    raw = _model_dump(value)
    input_details = _model_dump(raw.get("input_tokens_details"))
    output_details = _model_dump(raw.get("output_tokens_details"))
    # Chat Completions aliases are retained only for provider-compatible
    # usage receipts; the request itself is always the Responses API.
    prompt_details = _model_dump(raw.get("prompt_tokens_details"))
    completion_details = _model_dump(raw.get("completion_tokens_details"))
    return {
        "input_tokens": _integer_or_none(
            raw.get("input_tokens", raw.get("prompt_tokens"))
        ),
        "input_cached_tokens": _integer_or_none(
            input_details.get("cached_tokens", prompt_details.get("cached_tokens"))
        ),
        "output_tokens": _integer_or_none(
            raw.get("output_tokens", raw.get("completion_tokens"))
        ),
        "output_reasoning_tokens": _integer_or_none(
            output_details.get(
                "reasoning_tokens", completion_details.get("reasoning_tokens")
            )
        ),
        "total_tokens": _integer_or_none(raw.get("total_tokens")),
    }


def _sum_usage(receipts: list[dict[str, Any]]) -> dict[str, int | None]:
    names = (
        "input_tokens",
        "input_cached_tokens",
        "output_tokens",
        "output_reasoning_tokens",
        "total_tokens",
    )
    aggregate: dict[str, int | None] = {}
    for name in names:
        values = [
            receipt["token_usage"].get(name)
            for receipt in receipts
            if receipt.get("status") == "success"
        ]
        numeric = [value for value in values if type(value) is int]
        aggregate[name] = sum(numeric) if numeric else None
    return aggregate


def _provider_error(error: BaseException, *, call_index: int) -> dict[str, Any]:
    status_code = getattr(error, "status_code", None)
    request_id = getattr(error, "request_id", None)
    return {
        "api_call_index": call_index,
        "error_type": type(error).__name__,
        "message": str(error)[:500],
        "request_id": request_id if isinstance(request_id, str) else None,
        "status_code": status_code if type(status_code) is int else None,
    }


@dataclass(slots=True)
class _BoundedResponsesSampler:
    """Official SamplerBase-compatible GPT-5.4 low Responses API adapter."""

    client: Any
    sampler_response_type: type[Any]
    max_parse_attempts: int
    max_calls: int
    max_provider_attempts: int = 3
    calls: int = 0
    attempts_by_prompt: dict[str, int] = field(default_factory=dict)
    last_output_by_prompt: dict[str, str] = field(default_factory=dict)
    api_call_receipts: list[dict[str, Any]] = field(default_factory=list)
    provider_errors: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, messages: list[dict[str, str]]) -> Any:
        prompt_key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        with self.lock:
            attempt = self.attempts_by_prompt.get(prompt_key, 0) + 1
            if attempt > self.max_parse_attempts:
                raise RuntimeError(
                    "HealthBench grader exhausted its bounded JSON parse attempts"
                )
            self.attempts_by_prompt[prompt_key] = attempt
            prior_output = self.last_output_by_prompt.get(prompt_key)

        request_messages = [dict(message) for message in messages]
        if attempt > 1:
            if prior_output:
                request_messages.append({"role": "assistant", "content": prior_output})
            request_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Repair the prior response. Return only one JSON object with "
                        "a string explanation and a boolean criteria_met field."
                    ),
                }
            )

        # The official simple-evals ChatCompletionSampler retries provider
        # exceptions with exponential backoff.  Preserve that source behavior
        # but bound it and retain every physical attempt in the receipt.
        for provider_attempt in range(1, self.max_provider_attempts + 1):
            with self.lock:
                self.calls += 1
                if self.calls > self.max_calls:
                    raise RuntimeError(
                        "HealthBench grader exceeded its API call bound"
                    )
                call_index = self.calls
            started = time.perf_counter()
            try:
                response = self.client.responses.create(
                    model=GRADER_MODEL,
                    input=request_messages,
                    reasoning={"effort": REASONING_EFFORT},
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                response_text = getattr(response, "output_text", None)
                if type(response_text) is not str:
                    response_text = ""
                token_usage = _usage(getattr(response, "usage", None))
                receipt = {
                    "api_call_index": call_index,
                    "attempt": attempt,
                    "provider_attempt": provider_attempt,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "token_usage": token_usage,
                }
                with self.lock:
                    self.last_output_by_prompt[prompt_key] = response_text
                    self.api_call_receipts.append(receipt)
                return self.sampler_response_type(
                    response_text=response_text,
                    response_metadata={
                        "usage": getattr(response, "usage", None)
                    },
                    actual_queried_message_list=request_messages,
                )
            except Exception as error:
                latency_ms = (time.perf_counter() - started) * 1000.0
                provider_error = _provider_error(error, call_index=call_index)
                with self.lock:
                    self.provider_errors.append(provider_error)
                    self.api_call_receipts.append(
                        {
                            "api_call_index": call_index,
                            "attempt": attempt,
                            "provider_attempt": provider_attempt,
                            "latency_ms": latency_ms,
                            "status": "provider_error",
                            "token_usage": _usage(None),
                        }
                    )
                if provider_attempt >= self.max_provider_attempts:
                    raise
                time.sleep(min(2 ** (provider_attempt - 1), 4))
        raise RuntimeError("HealthBench grader provider retry loop exhausted")


def _base_receipt(
    *,
    task_id: str,
    candidate_answer: str,
    latency_ms: float,
    sampler: _BoundedResponsesSampler | None,
) -> dict[str, Any]:
    api_receipts = list(sampler.api_call_receipts) if sampler is not None else []
    return {
        "api_call_receipts": api_receipts,
        "evaluator_version": EVALUATOR_VERSION,
        "grader_api_calls": sampler.calls if sampler is not None else 0,
        "grader_error": None,
        "grader_latency_ms": latency_ms,
        "grader_model": GRADER_MODEL,
        "grader_reasoning_effort": REASONING_EFFORT,
        "grader_token_usage": _sum_usage(api_receipts),
        "length_adjustment_center": LENGTH_CENTER,
        "length_adjustment_penalty_per_500_chars": LENGTH_PENALTY_PER_500_CHARS,
        "overall_score": None,
        "overall_score_length_adjusted": None,
        "provider_errors": list(sampler.provider_errors) if sampler is not None else [],
        "response_characters": len(candidate_answer),
        "rubric_level_receipts": [],
        "task_id": task_id,
        "termination": "grader_error",
        "triggered_negative_rubric_count": 0,
    }


def _run(arguments: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    task_id = request["task_id"]
    candidate_answer = request["candidate_answer"]
    sampler: _BoundedResponsesSampler | None = None
    try:
        source_root = _absolute_source_root(arguments.official_source_root)
        case = request["private_case"]
        if set(case) != {"prompt", "rubrics"}:
            raise ValueError("HealthBench private case fields differ")
        prompt = _messages(case["prompt"])
        health_eval_type, rubric_type, sampler_response_type = _load_official(source_root)
        rubric_items = _rubrics(case["rubrics"], rubric_type)

        api_key = os.environ.get(arguments.api_key_environment)
        if not api_key:
            raise RuntimeError("HealthBench grader API credential is unavailable")
        from openai import OpenAI

        client_options: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
            "timeout": arguments.request_timeout_seconds,
        }
        if arguments.api_base_url is not None:
            client_options["base_url"] = arguments.api_base_url
        sampler = _BoundedResponsesSampler(
            client=OpenAI(**client_options),
            sampler_response_type=sampler_response_type,
            max_parse_attempts=arguments.max_parse_attempts,
            max_calls=(
                len(rubric_items)
                * arguments.max_parse_attempts
                * arguments.max_provider_attempts
            ),
            max_provider_attempts=arguments.max_provider_attempts,
        )
        evaluator = object.__new__(health_eval_type)
        evaluator.grader_model = sampler
        evaluator.length_adjustment_center = LENGTH_CENTER
        evaluator.length_adjustment_penalty_per_500_chars = (
            LENGTH_PENALTY_PER_500_CHARS
        )
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, _, rubric_grades = evaluator.grade_sample(
                prompt=prompt,
                response_text=candidate_answer,
                example_tags=[],
                rubric_items=rubric_items,
            )
        raw_score = metrics.get("overall_score")
        adjusted_score = metrics.get("overall_score_length_adjusted")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, int | float)
            or isinstance(adjusted_score, bool)
            or not isinstance(adjusted_score, int | float)
        ):
            raise RuntimeError("HealthBench official scores are unavailable")
        latency_ms = (time.perf_counter() - started) * 1000.0
        receipt = _base_receipt(
            task_id=task_id,
            candidate_answer=candidate_answer,
            latency_ms=latency_ms,
            sampler=sampler,
        )
        receipt.update(
            {
                "grader_error": None,
                "overall_score": float(raw_score),
                "overall_score_length_adjusted": float(adjusted_score),
                "rubric_level_receipts": rubric_grades,
                "termination": "graded",
                "triggered_negative_rubric_count": sum(
                    1
                    for item in rubric_grades
                    if item.get("points", 0) < 0
                    and item.get("criteria_met") is True
                ),
            }
        )
        return receipt
    except Exception as error:
        latency_ms = (time.perf_counter() - started) * 1000.0
        receipt = _base_receipt(
            task_id=task_id,
            candidate_answer=candidate_answer,
            latency_ms=latency_ms,
            sampler=sampler,
        )
        receipt["grader_error"] = {
            "error_type": type(error).__name__,
            "message": str(error)[:500],
        }
        return receipt


def main(arguments: argparse.Namespace) -> None:
    request = _request(json.loads(sys.stdin.buffer.readline()))
    result = _run(arguments, request)
    json.dump(result, sys.stdout, separators=(",", ":"), ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source-root", required=True)
    parser.add_argument("--api-key-environment", default="OPENAI_API_KEY")
    parser.add_argument("--api-base-url")
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-parse-attempts", type=int, default=2)
    parser.add_argument("--max-provider-attempts", type=int, default=3)
    main(parser.parse_args())
