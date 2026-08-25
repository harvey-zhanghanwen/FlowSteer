"""Private, reference-compatible HealthBench Professional grader binding.

This is a thin project adapter over two upstream boundaries:

* SkillEval ``OfficialHealthBenchProcessGrader`` / ``PrivateJSONWorker`` keep
  evaluator-only data in a task-id keyed private store and invoke an isolated
  worker process.
* OpenAI simple-evals ``HealthBenchEval.grade_sample`` owns rubric grading and
  score calculation inside that worker.

Only ``task_id`` and the candidate response cross the public caller boundary.
Rubrics and the original conversation are joined from ``private_cases.jsonl``
after generation and are never returned to the Director or AgentGraph runtime.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Protocol


HEALTHBENCH_PROFESSIONAL_GRADER_MODEL = "gpt-5.4-2026-03-05"
HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT = "low"
HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER = 2000.0
HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS = 0.0147
HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION = (
    "openai-simple-evals-healthbench-professional-652c89d@1"
)

_WORKER = Path(__file__).resolve().parents[2] / "scripts" / (
    "healthbench_professional_grader_worker.py"
)
_RECEIPT_FIELDS = frozenset(
    {
        "api_call_receipts",
        "evaluator_version",
        "grader_api_calls",
        "grader_error",
        "grader_latency_ms",
        "grader_model",
        "grader_reasoning_effort",
        "grader_token_usage",
        "length_adjustment_center",
        "length_adjustment_penalty_per_500_chars",
        "overall_score",
        "overall_score_length_adjusted",
        "provider_errors",
        "response_characters",
        "rubric_level_receipts",
        "task_id",
        "termination",
        "triggered_negative_rubric_count",
    }
)


class HealthBenchProfessionalGraderError(RuntimeError):
    """The private route or isolated reference grader failed structurally."""


class JSONWorkerTransport(Protocol):
    """Bounded asynchronous transport for one worker request."""

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes: ...


async def _stop_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int] | None,
) -> None:
    if process.returncode is not None or (wait_task is not None and wait_task.done()):
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
        await asyncio.wait_for(asyncio.shield(waiter), timeout=1.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
        await asyncio.shield(waiter)


@dataclass(frozen=True, slots=True)
class AsyncSubprocessJSONTransport:
    """SkillEval-style cancellation-safe one-request subprocess transport."""

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes:
        process: asyncio.subprocess.Process | None = None
        wait_task: asyncio.Task[int] | None = None
        async def execute() -> bytes:
            nonlocal process, wait_task
            try:
                environment = os.environ.copy()
                environment.pop("PYTHONPATH", None)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=working_directory,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                if process.stdin is None or process.stdout is None:  # pragma: no cover
                    raise HealthBenchProfessionalGraderError(
                        "HealthBench grader pipes are unavailable"
                    )
                wait_task = asyncio.create_task(process.wait())
                try:
                    process.stdin.write(encoded)
                    await process.stdin.drain()
                    process.stdin.close()
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    response = await process.stdout.readexactly(max_response_bytes + 1)
                except asyncio.IncompleteReadError as error:
                    response = error.partial
                if len(response) > max_response_bytes:
                    raise HealthBenchProfessionalGraderError(
                        "HealthBench grader response exceeded its size bound"
                    )
                return_code = await asyncio.shield(wait_task)
                if return_code != 0:
                    raise HealthBenchProfessionalGraderError(
                        "HealthBench grader process returned failure"
                    )
                return response
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                raise HealthBenchProfessionalGraderError(
                    "HealthBench grader pipe closed unexpectedly"
                )

        try:
            return await asyncio.wait_for(execute(), timeout=timeout_seconds)
        except asyncio.CancelledError:
            if process is not None:
                await asyncio.shield(_stop_process(process, wait_task))
            raise
        except HealthBenchProfessionalGraderError:
            if process is not None:
                await _stop_process(process, wait_task)
            raise
        except (OSError, asyncio.TimeoutError) as error:
            if process is not None:
                await _stop_process(process, wait_task)
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader process did not complete"
            ) from error


InMemoryResponder = Callable[[bytes], bytes | Awaitable[bytes]]


@dataclass(slots=True)
class InMemoryJSONTransport:
    """Injectable fake used to test private routing without an API call."""

    responder: InMemoryResponder
    requests: list[bytes] = field(default_factory=list)

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes:
        del command, working_directory
        self.requests.append(encoded)
        try:
            result = self.responder(encoded)
            if inspect.isawaitable(result):
                response = await asyncio.wait_for(result, timeout=timeout_seconds)
            else:
                response = result
        except asyncio.TimeoutError as error:
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader process did not complete"
            ) from error
        if not isinstance(response, bytes):
            raise TypeError("in-memory HealthBench worker response must be bytes")
        if len(response) > max_response_bytes:
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader response exceeded its size bound"
            )
        return response


@dataclass(frozen=True, slots=True)
class PrivateJSONWorker:
    """One-request/one-response JSON worker with bounded private I/O."""

    command: tuple[str, ...]
    working_directory: Path
    timeout_seconds: float = 900.0
    max_response_bytes: int = 4 * 1024 * 1024
    transport: JSONWorkerTransport | None = None

    def __post_init__(self) -> None:
        if not self.command or any(not part for part in self.command):
            raise ValueError("HealthBench worker command must be non-empty")
        if not self.working_directory.is_absolute():
            raise ValueError("HealthBench worker directory must be absolute")
        if self.timeout_seconds <= 0 or self.max_response_bytes <= 0:
            raise ValueError("HealthBench worker limits must be positive")

    async def request(self, value: Mapping[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(
                _json_compatible(value),
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            raise HealthBenchProfessionalGraderError(
                "HealthBench worker request is not JSON-compatible"
            ) from error
        transport = self.transport or AsyncSubprocessJSONTransport()
        response = await transport.request(
            command=self.command,
            working_directory=self.working_directory,
            encoded=encoded,
            timeout_seconds=self.timeout_seconds,
            max_response_bytes=self.max_response_bytes,
        )
        if not response:
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader returned an empty response"
            )
        try:
            result = json.loads(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader returned malformed JSON"
            ) from error
        if not isinstance(result, dict):
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader response must be an object"
            )
        return result


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _messages(value: object) -> list[dict[str, str]]:
    if isinstance(value, Mapping) and set(value) == {"messages"}:
        value = value["messages"]
    if not isinstance(value, list) or not value:
        raise ValueError("HealthBench private conversation must be non-empty")
    messages: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
            raise ValueError("HealthBench private conversation message fields differ")
        role = raw["role"]
        content = raw["content"]
        if type(role) is not str or type(content) is not str or not content.strip():
            raise ValueError("HealthBench private conversation message is invalid")
        messages.append({"role": role, "content": content})
    return messages


def _rubrics(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("HealthBench private rubrics must be non-empty")
    rubrics: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("HealthBench private rubric must be an object")
        criterion = raw.get("criterion", raw.get("criterion_text"))
        points = raw.get("points")
        tags = raw.get("tags", [])
        if type(criterion) is not str or not criterion.strip():
            raise ValueError("HealthBench private rubric criterion is invalid")
        if isinstance(points, bool) or not isinstance(points, int | float):
            raise ValueError("HealthBench private rubric points are invalid")
        if not isinstance(tags, list) or any(type(tag) is not str for tag in tags):
            raise ValueError("HealthBench private rubric tags are invalid")
        rubrics.append(
            {"criterion": criterion, "points": float(points), "tags": list(tags)}
        )
    if not any(item["points"] > 0 for item in rubrics):
        raise ValueError("HealthBench private rubrics need positive possible points")
    return rubrics


def _private_case(value: Mapping[str, Any]) -> dict[str, Any]:
    prompt = value.get("prompt", value.get("conversation"))
    rubrics = value.get("rubrics", value.get("rubric_items"))
    # Project only fields consumed by the official evaluator. In particular,
    # physician_response and benchmark metadata never enter the worker request.
    return {"prompt": _messages(prompt), "rubrics": _rubrics(rubrics)}


def load_private_cases(path: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load and strictly project a task-id keyed evaluator-only JSONL store."""

    if not path.is_absolute() or not path.is_file():
        raise ValueError("HealthBench private_cases.jsonl must be an absolute file")
    cases: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"HealthBench private case line {line_number} is malformed"
                ) from error
            if not isinstance(raw, Mapping):
                raise ValueError("HealthBench private case must be an object")
            task_id = raw.get("task_id", raw.get("id"))
            if type(task_id) is not str or not task_id.strip():
                raise ValueError("HealthBench private case task_id is invalid")
            if task_id in cases:
                raise ValueError("HealthBench private case task_id is duplicated")
            cases[task_id] = MappingProxyType(_private_case(raw))
    if not cases:
        raise ValueError("HealthBench private case store is empty")
    return MappingProxyType(cases)


def _numeric(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _validate_receipt(
    result: Mapping[str, Any],
    *,
    task_id: str,
    candidate_answer: str,
) -> dict[str, Any]:
    if set(result) != _RECEIPT_FIELDS:
        raise HealthBenchProfessionalGraderError(
            "HealthBench grader response fields differ"
        )
    expected_constants = {
        "task_id": task_id,
        "evaluator_version": HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION,
        "grader_model": HEALTHBENCH_PROFESSIONAL_GRADER_MODEL,
        "grader_reasoning_effort": HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT,
        "length_adjustment_center": HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER,
        "length_adjustment_penalty_per_500_chars": (
            HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS
        ),
        "response_characters": len(candidate_answer),
    }
    for key, expected in expected_constants.items():
        if result[key] != expected:
            raise HealthBenchProfessionalGraderError(
                f"HealthBench grader response {key} differs"
            )
    if result["termination"] not in {"graded", "grader_error"}:
        raise HealthBenchProfessionalGraderError(
            "HealthBench grader termination is invalid"
        )
    if type(result["grader_api_calls"]) is not int or result["grader_api_calls"] < 0:
        raise HealthBenchProfessionalGraderError(
            "HealthBench grader API call count is invalid"
        )
    if not _numeric(result["grader_latency_ms"]) or result["grader_latency_ms"] < 0:
        raise HealthBenchProfessionalGraderError(
            "HealthBench grader latency is invalid"
        )
    if type(result["triggered_negative_rubric_count"]) is not int:
        raise HealthBenchProfessionalGraderError(
            "HealthBench negative rubric count is invalid"
        )
    for key in (
        "api_call_receipts",
        "provider_errors",
        "rubric_level_receipts",
    ):
        if not isinstance(result[key], list):
            raise HealthBenchProfessionalGraderError(
                f"HealthBench grader {key} is invalid"
            )
    if not isinstance(result["grader_token_usage"], Mapping):
        raise HealthBenchProfessionalGraderError(
            "HealthBench grader token usage is invalid"
        )
    if result["termination"] == "graded":
        if not _numeric(result["overall_score"]) or not _numeric(
            result["overall_score_length_adjusted"]
        ):
            raise HealthBenchProfessionalGraderError(
                "HealthBench successful scores are invalid"
            )
        if result["grader_error"] is not None or not result["rubric_level_receipts"]:
            raise HealthBenchProfessionalGraderError(
                "HealthBench successful grader receipt is incomplete"
            )
    else:
        if result["overall_score"] is not None or result[
            "overall_score_length_adjusted"
        ] is not None:
            raise HealthBenchProfessionalGraderError(
                "HealthBench failed grader cannot emit scores"
            )
        if not isinstance(result["grader_error"], Mapping):
            raise HealthBenchProfessionalGraderError(
                "HealthBench grader error receipt is absent"
            )
    return dict(result)


_PREFLIGHT_TASK_ID = "healthbench-professional:synthetic-preflight"
_PREFLIGHT_CASE: Mapping[str, Any] = MappingProxyType(
    {
        "prompt": [
            {
                "role": "user",
                "content": "Please answer this synthetic evaluator preflight request clearly.",
            }
        ],
        "rubrics": [
            {
                "criterion": "The response addresses the synthetic request.",
                "points": 1.0,
                "tags": ["synthetic-preflight"],
            }
        ],
    }
)
_PREFLIGHT_ANSWER = "This response addresses the synthetic evaluator preflight request."


@dataclass(frozen=True, slots=True)
class HealthBenchProfessionalGrader:
    """Task-id-only public facade over the evaluator-only private case store."""

    worker: PrivateJSONWorker
    private_cases: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_private_cases_jsonl(
        cls,
        *,
        private_cases_path: Path,
        official_source_root: Path,
        interpreter_path: Path | None = None,
        api_key_environment: str = "OPENAI_API_KEY",
        api_base_url: str | None = None,
        request_timeout_seconds: float = 90.0,
        worker_timeout_seconds: float = 900.0,
        max_parse_attempts: int = 2,
        max_provider_attempts: int = 3,
        transport: JSONWorkerTransport | None = None,
    ) -> "HealthBenchProfessionalGrader":
        """Bind a pinned simple-evals checkout and private case JSONL."""

        interpreter = interpreter_path or Path(sys.executable)
        if not interpreter.is_absolute() or not interpreter.is_file():
            raise ValueError("HealthBench grader interpreter must be an absolute file")
        if not official_source_root.is_absolute() or not (
            official_source_root / "healthbench_eval.py"
        ).is_file():
            raise ValueError("pinned simple-evals HealthBench source is absent")
        if not api_key_environment.strip():
            raise ValueError("HealthBench grader API key environment is empty")
        if api_base_url is not None and not api_base_url.strip():
            raise ValueError("HealthBench grader API base URL is empty")
        if request_timeout_seconds <= 0 or worker_timeout_seconds <= request_timeout_seconds:
            raise ValueError("HealthBench grader timeouts are invalid")
        if max_parse_attempts < 1:
            raise ValueError("HealthBench grader parse attempts must be positive")
        if max_provider_attempts < 1:
            raise ValueError("HealthBench grader provider attempts must be positive")
        command = [
            str(interpreter),
            str(_WORKER),
            "--official-source-root",
            str(official_source_root),
            "--api-key-environment",
            api_key_environment,
            "--request-timeout-seconds",
            str(request_timeout_seconds),
            "--max-parse-attempts",
            str(max_parse_attempts),
            "--max-provider-attempts",
            str(max_provider_attempts),
        ]
        if api_base_url is not None:
            command.extend(("--api-base-url", api_base_url))
        return cls(
            worker=PrivateJSONWorker(
                command=tuple(command),
                working_directory=official_source_root.parent,
                timeout_seconds=worker_timeout_seconds,
                transport=transport,
            ),
            private_cases=load_private_cases(private_cases_path),
        )

    async def grade(self, task_id: str, candidate_answer: str) -> Mapping[str, Any]:
        """Grade one candidate after joining its evaluator-only private case."""

        if type(task_id) is not str or not task_id.strip():
            raise ValueError("HealthBench grader task_id must be non-empty")
        if type(candidate_answer) is not str or not candidate_answer.strip():
            raise ValueError("HealthBench candidate response must be non-empty")
        try:
            private_case = self.private_cases[task_id]
        except KeyError as error:
            raise HealthBenchProfessionalGraderError(
                "HealthBench task is not privately routed"
            ) from error
        result = await self.worker.request(
            {
                "candidate_answer": candidate_answer,
                "operation": "grade",
                "private_case": private_case,
                "task_id": task_id,
            }
        )
        return MappingProxyType(
            _validate_receipt(
                result, task_id=task_id, candidate_answer=candidate_answer
            )
        )

    async def preflight(self) -> Mapping[str, Any]:
        """Exercise the exact grader protocol using only a synthetic case."""

        result = await self.worker.request(
            {
                "candidate_answer": _PREFLIGHT_ANSWER,
                "operation": "preflight",
                "private_case": _PREFLIGHT_CASE,
                "task_id": _PREFLIGHT_TASK_ID,
            }
        )
        return MappingProxyType(
            _validate_receipt(
                result,
                task_id=_PREFLIGHT_TASK_ID,
                candidate_answer=_PREFLIGHT_ANSWER,
            )
        )


__all__ = [
    "AsyncSubprocessJSONTransport",
    "HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION",
    "HEALTHBENCH_PROFESSIONAL_GRADER_MODEL",
    "HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER",
    "HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500_CHARS",
    "HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT",
    "HealthBenchProfessionalGrader",
    "HealthBenchProfessionalGraderError",
    "InMemoryJSONTransport",
    "PrivateJSONWorker",
    "load_private_cases",
]
