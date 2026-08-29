"""Official EvalPlus adapter for MBPP+ terminal evaluation.

This module is deliberately a thin boundary around EvalPlus.  EvalPlus owns
solution sanitization, dataset deserialization, expected-output generation,
subprocess execution, and base/plus status assignment.  The adapter only
binds those APIs to one configured runtime and one frozen task selection.

Neither hidden inputs nor failed test cases are copied into AgentGraph
receipts.  They remain inside the evaluator process and EvalPlus cache.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from .records import TaskRecord


MBPPPLUS_OFFICIAL_PROTOCOL = "evalplus.mbpp-plus.pass-at-1.v0.3.1"
MBPPPLUS_EVALUATOR_VERSION = MBPPPLUS_OFFICIAL_PROTOCOL
_OFFICIAL_RUNTIME_LOCK = threading.RLock()
_TASK_ID_PATTERN = re.compile(r"(?:^|:)(Mbpp/\d+)$", re.IGNORECASE)


class MBPPPlusEvaluatorUnavailable(RuntimeError):
    """The configured official EvalPlus runtime cannot evaluate MBPP+."""


@dataclass(frozen=True)
class _OfficialEvalPlusRuntime:
    sanitize: Callable[..., str]
    get_groundtruth: Callable[..., Mapping[str, Any]]
    check_correctness: Callable[..., Mapping[str, Any]]
    pass_status: str
    output_not_none_tasks: Sequence[str]
    runtime_version: str
    dataset_version: str


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _record_metadata(record: TaskRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        metadata = record.get("metadata", {})
    else:
        metadata = record.metadata
    return metadata if isinstance(metadata, Mapping) else {}


def _record_task_id(record: TaskRecord | Mapping[str, Any]) -> str:
    metadata = _record_metadata(record)
    evaluator_payload = metadata.get("evaluator_payload", {})
    candidates: list[Any] = []
    if isinstance(evaluator_payload, Mapping):
        candidates.extend(
            (
                evaluator_payload.get("task_id"),
                evaluator_payload.get("benchmark_task_id"),
            )
        )
    candidates.extend(
        (
            metadata.get("benchmark_task_id"),
            metadata.get("official_task_id"),
            record.get("task_id") if isinstance(record, Mapping) else record.task_id,
        )
    )
    for value in candidates:
        if not isinstance(value, str):
            continue
        match = _TASK_ID_PATTERN.search(value.strip())
        if match is not None:
            _prefix, number = match.group(1).split("/", 1)
            return f"Mbpp/{number}"
    raise MBPPPlusEvaluatorUnavailable(
        "record does not identify one official MBPP+ task"
    )


class MBPPPlusOfficialEvaluator:
    """Evaluate a frozen MBPP+ selection with official EvalPlus 0.3.1 APIs."""

    def __init__(
        self,
        *,
        runtime_path: Path,
        dataset_path: Path,
        cache_root: Path,
        selected_task_ids: Sequence[str],
    ) -> None:
        if isinstance(selected_task_ids, (str, bytes)) or not isinstance(
            selected_task_ids, Sequence
        ):
            raise ValueError("selected_task_ids must be a sequence")
        normalized_ids = tuple(str(task_id).strip() for task_id in selected_task_ids)
        if not normalized_ids or any(not task_id for task_id in normalized_ids):
            raise ValueError("selected_task_ids must contain non-empty task IDs")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("selected_task_ids must be unique")
        if any(re.fullmatch(r"Mbpp/\d+", task_id) is None for task_id in normalized_ids):
            raise ValueError("selected_task_ids must use official Mbpp/<number> IDs")

        self.runtime_path = Path(runtime_path).expanduser().resolve()
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.selected_task_ids = normalized_ids
        self._selected_task_id_set = frozenset(normalized_ids)
        self._runtime: _OfficialEvalPlusRuntime | None = None
        self._problems: Mapping[str, Mapping[str, Any]] | None = None
        self._expected_outputs: dict[str, Mapping[str, Any]] = {}
        self._load_lock = threading.RLock()

    def _load_runtime(
        self,
    ) -> tuple[_OfficialEvalPlusRuntime, Mapping[str, Mapping[str, Any]]]:
        with self._load_lock:
            if self._runtime is not None and self._problems is not None:
                return self._runtime, self._problems
            if not (self.runtime_path / "evalplus" / "__init__.py").is_file():
                raise MBPPPlusEvaluatorUnavailable(
                    "configured EvalPlus runtime_path is unavailable"
                )
            if not self.dataset_path.is_file():
                raise MBPPPlusEvaluatorUnavailable(
                    "configured MBPP+ dataset_path is unavailable"
                )
            self.cache_root.mkdir(parents=True, exist_ok=True)

            with _OFFICIAL_RUNTIME_LOCK:
                # EvalPlus reads MBPP_OVERRIDE_PATH while importing
                # evalplus.data.mbpp, so configure it before every import.
                os.environ["MBPP_OVERRIDE_PATH"] = str(self.dataset_path)
                runtime_string = str(self.runtime_path)
                if runtime_string not in sys.path:
                    sys.path.insert(0, runtime_string)

                package = importlib.import_module("evalplus")
                package_file = Path(str(package.__file__)).resolve()
                if not _is_within(package_file, self.runtime_path):
                    raise MBPPPlusEvaluatorUnavailable(
                        "another EvalPlus runtime is already imported in this process"
                    )

                sanitize_module = importlib.import_module("evalplus.sanitize")
                data_module = importlib.import_module("evalplus.data")
                mbpp_module = importlib.import_module("evalplus.data.mbpp")
                data_utils_module = importlib.import_module("evalplus.data.utils")
                evaluate_module = importlib.import_module("evalplus.evaluate")
                eval_module = importlib.import_module("evalplus.eval")
                special_oracle_module = importlib.import_module(
                    "evalplus.eval._special_oracle"
                )

                # These are process-global constants in EvalPlus.  Setting
                # them here only redirects the official loader and its
                # expected-output cache to the configured paths.
                mbpp_module.MBPP_OVERRIDE_PATH = str(self.dataset_path)
                mbpp_module.CACHE_DIR = str(self.cache_root)
                data_utils_module.CACHE_DIR = str(self.cache_root)
                evaluate_module.CACHE_DIR = str(self.cache_root)

                runtime_version = str(
                    getattr(package, "__version__", "unknown")
                )
                if runtime_version != "0.3.1":
                    raise MBPPPlusEvaluatorUnavailable(
                        "MBPP+ evaluation requires EvalPlus 0.3.1"
                    )
                problems = data_module.get_mbpp_plus()

            if not isinstance(problems, Mapping) or not problems:
                raise MBPPPlusEvaluatorUnavailable(
                    "official EvalPlus returned no MBPP+ tasks"
                )
            missing = sorted(self._selected_task_id_set.difference(problems))
            if missing:
                raise MBPPPlusEvaluatorUnavailable(
                    "selected MBPP+ task IDs are absent from the official dataset: "
                    + ", ".join(missing)
                )

            runtime = _OfficialEvalPlusRuntime(
                sanitize=sanitize_module.sanitize,
                get_groundtruth=evaluate_module.get_groundtruth,
                check_correctness=evaluate_module.check_correctness,
                pass_status=str(eval_module.PASS),
                output_not_none_tasks=tuple(
                    special_oracle_module.MBPP_OUTPUT_NOT_NONE_TASKS
                ),
                runtime_version=runtime_version,
                dataset_version=str(
                    getattr(mbpp_module, "MBPP_PLUS_VERSION", "unknown")
                ),
            )
            self._runtime = runtime
            self._problems = problems
            return runtime, problems

    def _expected_output(
        self,
        task_id: str,
        *,
        runtime: _OfficialEvalPlusRuntime,
        problem: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._load_lock:
            cached = self._expected_outputs.get(task_id)
            if cached is not None:
                return cached
            # EvalPlus normally stores one dataset-wide expected-output cache.
            # A task-scoped cache keeps preflight and selected evaluation lazy
            # without changing get_groundtruth or its trusted execution.
            cache_key = (
                f"mbpp-plus-{runtime.dataset_version}-"
                f"{task_id.replace('/', '_')}"
            )
            outputs = runtime.get_groundtruth(
                {task_id: problem},
                cache_key,
                runtime.output_not_none_tasks,
            )
            if not isinstance(outputs, Mapping) or task_id not in outputs:
                raise MBPPPlusEvaluatorUnavailable(
                    "official EvalPlus did not return expected outputs"
                )
            expected = outputs[task_id]
            if not isinstance(expected, Mapping):
                raise MBPPPlusEvaluatorUnavailable(
                    "official EvalPlus expected-output record is invalid"
                )
            self._expected_outputs[task_id] = expected
            return expected

    def _evaluate_task_id(
        self,
        task_id: str,
        prediction: str,
        *,
        allow_unselected: bool,
    ) -> Mapping[str, Any]:
        runtime, problems = self._load_runtime()
        if not allow_unselected and task_id not in self._selected_task_id_set:
            raise MBPPPlusEvaluatorUnavailable(
                "MBPP+ task is outside the frozen evaluation selection"
            )
        problem = problems.get(task_id)
        if not isinstance(problem, Mapping):
            raise MBPPPlusEvaluatorUnavailable(
                "MBPP+ task is absent from the official dataset"
            )
        entry_point = problem.get("entry_point")
        if not isinstance(entry_point, str) or not entry_point:
            raise MBPPPlusEvaluatorUnavailable(
                "official MBPP+ task has no entry_point"
            )

        raw_prediction = str(prediction)
        try:
            sanitized_solution = runtime.sanitize(
                code=raw_prediction,
                entrypoint=entry_point,
            )
        except Exception as exc:
            raise MBPPPlusEvaluatorUnavailable(
                f"official EvalPlus sanitizer failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(sanitized_solution, str):
            raise MBPPPlusEvaluatorUnavailable(
                "official EvalPlus sanitizer returned a non-string result"
            )

        expected_output = self._expected_output(
            task_id,
            runtime=runtime,
            problem=problem,
        )
        result = runtime.check_correctness(
            "mbpp",
            0,
            dict(problem),
            sanitized_solution,
            dict(expected_output),
            base_only=False,
            fast_check=True,
            identifier=task_id,
        )
        if not isinstance(result, Mapping):
            raise MBPPPlusEvaluatorUnavailable(
                "official EvalPlus correctness result is invalid"
            )
        base = result.get("base")
        plus = result.get("plus")
        if (
            not isinstance(base, Sequence)
            or isinstance(base, (str, bytes))
            or not base
            or not isinstance(plus, Sequence)
            or isinstance(plus, (str, bytes))
            or not plus
        ):
            raise MBPPPlusEvaluatorUnavailable(
                "official EvalPlus base/plus status is missing"
            )
        base_status = str(base[0])
        plus_status = str(plus[0])
        base_passed = base_status == runtime.pass_status
        plus_passed = plus_status == runtime.pass_status
        plus_pass_at_1 = plus_passed
        return {
            "task_id": task_id,
            "base_status": base_status,
            "plus_status": plus_status,
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "pass_at_1": float(plus_pass_at_1),
            "format_diagnostics": {
                "raw_prediction_empty": not bool(raw_prediction.strip()),
                "sanitized_prediction_empty": not bool(sanitized_solution.strip()),
                "sanitization_changed": sanitized_solution != raw_prediction,
                "sanitized_character_count": len(sanitized_solution),
                "entry_point": entry_point,
            },
            "evaluator_protocol": MBPPPLUS_OFFICIAL_PROTOCOL,
            "runtime_version": runtime.runtime_version,
            "dataset_version": runtime.dataset_version,
        }

    async def evaluate(
        self,
        record: TaskRecord | Mapping[str, Any],
        prediction: str,
    ) -> Mapping[str, Any]:
        """Evaluate one selected task and return only non-hidden statuses."""

        task_id = _record_task_id(record)
        return await asyncio.to_thread(
            self._evaluate_task_id,
            task_id,
            str(prediction),
            allow_unselected=False,
        )

    async def preflight(self) -> Mapping[str, Any]:
        """Run one canonical solution outside the frozen selected tasks."""

        runtime, problems = await asyncio.to_thread(self._load_runtime)
        candidates = sorted(
            (task_id for task_id in problems if task_id not in self._selected_task_id_set),
            key=lambda task_id: int(task_id.split("/", 1)[1]),
        )
        if not candidates:
            raise MBPPPlusEvaluatorUnavailable(
                "evaluator preflight requires an MBPP+ task outside the selection"
            )
        task_id = candidates[0]
        canonical_solution = problems[task_id].get("canonical_solution")
        if not isinstance(canonical_solution, str) or not canonical_solution.strip():
            raise MBPPPlusEvaluatorUnavailable(
                "official MBPP+ preflight task has no canonical solution"
            )
        result = await asyncio.to_thread(
            self._evaluate_task_id,
            task_id,
            canonical_solution,
            allow_unselected=True,
        )
        # Deliberately omit the canonical solution, test inputs, expected
        # outputs, and per-test details from this evaluator-only receipt.
        return {
            "ready": result["pass_at_1"] == 1.0,
            "task_id": task_id,
            "base_status": result["base_status"],
            "plus_status": result["plus_status"],
            "base_passed": result["base_passed"],
            "plus_passed": result["plus_passed"],
            "pass_at_1": result["pass_at_1"],
            "selection_disjoint": True,
            "selected_task_count": len(self.selected_task_ids),
            "evaluator_protocol": MBPPPLUS_OFFICIAL_PROTOCOL,
            "runtime_version": runtime.runtime_version,
            "dataset_version": runtime.dataset_version,
        }


__all__ = [
    "MBPPPLUS_EVALUATOR_VERSION",
    "MBPPPLUS_OFFICIAL_PROTOCOL",
    "MBPPPlusEvaluatorUnavailable",
    "MBPPPlusOfficialEvaluator",
]
