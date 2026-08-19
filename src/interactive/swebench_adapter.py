"""Thin dataset-source adapter for SkillFlow's official SWE-bench evaluator.

The patch evaluator remains the deployed SkillFlow implementation in
``training/swe_bench_eval.py``.  This module only supplies the project record
boundary, formal-runtime configuration, and a fail-closed Docker/harness
preflight; it never computes a similarity or other proxy score.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .records import TaskRecord


DEFAULT_SKILLFLOW_SWE_EVALUATOR = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/swe_bench_eval.py"
)
DEFAULT_SWEBENCH_HARNESS = Path(
    "/ssd1/iclr/.private/skillflow-resources/SWE-bench-harness-d83"
)
DEFAULT_SWEBENCH_VERIFIED = Path(
    "/ssd1/iclr/.private/skillflow-resources/swebench-verified"
)
SWEBENCH_DATASET_SOURCE_REGULAR_DEV = "regular_dev"
SWEBENCH_DATASET_SOURCE_VERIFIED = "verified"
SWEBENCH_EVALUATION_SOURCES = frozenset(
    {
        SWEBENCH_DATASET_SOURCE_REGULAR_DEV,
        SWEBENCH_DATASET_SOURCE_VERIFIED,
    }
)
_EXPECTED_SPLIT_BY_SOURCE = {
    SWEBENCH_DATASET_SOURCE_REGULAR_DEV: "validation",
    SWEBENCH_DATASET_SOURCE_VERIFIED: "test",
}


class SWEbenchHarnessUnavailable(RuntimeError):
    """The official SWE-bench harness cannot produce a resolved-rate result."""


def _load_skillflow_evaluator(path: Path) -> Any:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise SWEbenchHarnessUnavailable(
            f"SkillFlow SWE-bench evaluator is unavailable: {source}"
        )
    module_name = "_flowsteer_skillflow_swe_bench_eval"
    loaded = sys.modules.get(module_name)
    loaded_source = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_source and Path(str(loaded_source)).expanduser().resolve() == source:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise SWEbenchHarnessUnavailable(
            f"cannot load SkillFlow SWE-bench evaluator: {source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _instance_id(record: TaskRecord | Mapping[str, Any]) -> str:
    metadata = (
        record.metadata
        if isinstance(record, TaskRecord)
        else record.get("metadata", {})
    )
    if not isinstance(metadata, Mapping):
        metadata = {}
    payload = metadata.get("evaluator_payload", {})
    extra = metadata.get("extra", {})
    candidates = (
        payload.get("instance_id") if isinstance(payload, Mapping) else None,
        extra.get("instance_id") if isinstance(extra, Mapping) else None,
        metadata.get("instance_id"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("SWE-bench record has no instance_id")


def _record_metadata(record: TaskRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = (
        record.metadata
        if isinstance(record, TaskRecord)
        else record.get("metadata", {})
    )
    if not isinstance(metadata, Mapping):
        raise ValueError("SWE-bench record metadata must be a mapping")
    return metadata


def _record_dataset_source(record: TaskRecord | Mapping[str, Any]) -> str:
    metadata = _record_metadata(record)
    payload = metadata.get("evaluator_payload", {})
    candidates = (
        metadata.get("dataset_source"),
        metadata.get("benchmark_slice"),
        payload.get("dataset_source") if isinstance(payload, Mapping) else None,
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("SWE-bench record has no dataset_source")


def _record_split(record: TaskRecord | Mapping[str, Any]) -> str:
    value = record.split if isinstance(record, TaskRecord) else record.get("split")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SWE-bench record has no split")
    return value.strip()


def _require_source_split(
    record: TaskRecord | Mapping[str, Any], dataset_source: str
) -> None:
    if _record_dataset_source(record) != dataset_source:
        raise SWEbenchHarnessUnavailable(
            "SWE-bench record crossed the configured dataset-source boundary"
        )
    expected_split = _EXPECTED_SPLIT_BY_SOURCE[dataset_source]
    if _record_split(record) != expected_split:
        raise SWEbenchHarnessUnavailable(
            f"SWE-bench {dataset_source} record must use project split {expected_split}"
        )


def _test_ids(value: object, *, field_name: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SWE-bench {field_name} is not valid JSON") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"SWE-bench {field_name} must be a sequence")
    result = [str(item) for item in value]
    if any(not item.strip() for item in result):
        raise ValueError(f"SWE-bench {field_name} contains an empty test ID")
    return result


def _regular_evaluator_row(
    record: TaskRecord | Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    metadata = _record_metadata(record)
    payload = metadata.get("evaluator_payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("SWE-bench evaluator_payload must be a mapping")
    instance_id = _instance_id(record)
    required = ("repo", "base_commit", "test_patch", "version")
    missing = [
        field_name
        for field_name in required
        if not isinstance(payload.get(field_name), str)
        or (field_name != "version" and not str(payload[field_name]).strip())
    ]
    if missing:
        raise ValueError(
            f"SWE-bench {instance_id} evaluator payload is missing: "
            + ", ".join(missing)
        )
    patch = (
        record.ground_truth
        if isinstance(record, TaskRecord)
        else record.get("ground_truth")
    )
    if not isinstance(patch, str):
        raise ValueError(f"SWE-bench {instance_id} has no evaluator gold patch")
    return instance_id, {
        "instance_id": instance_id,
        "repo": str(payload["repo"]),
        "version": str(payload.get("version", "")),
        "base_commit": str(payload["base_commit"]),
        "environment_setup_commit": str(payload.get("environment_setup_commit", "")),
        "patch": patch,
        "test_patch": str(payload["test_patch"]),
        "FAIL_TO_PASS": _test_ids(
            payload.get("FAIL_TO_PASS", []), field_name="FAIL_TO_PASS"
        ),
        "PASS_TO_PASS": _test_ids(
            payload.get("PASS_TO_PASS", []), field_name="PASS_TO_PASS"
        ),
    }


@dataclass(frozen=True)
class OfficialSWEbenchHarness:
    """Callable wrapper around SkillFlow's official Docker evaluator."""

    evaluator_path: Path = DEFAULT_SKILLFLOW_SWE_EVALUATOR
    harness_path: Path = DEFAULT_SWEBENCH_HARNESS
    dataset_source: str = SWEBENCH_DATASET_SOURCE_VERIFIED
    dataset_path: Path = DEFAULT_SWEBENCH_VERIFIED
    evaluation_root: Path = Path("artifacts/swebench_official_evaluation")
    docker_namespace: str = "swebench"
    timeout_seconds: int = 900

    @property
    def verified_path(self) -> Path:
        """Compatibility alias for callers that only inspect final-eval provenance."""

        return self.dataset_path

    def _configured_module(
        self,
        records: Sequence[TaskRecord | Mapping[str, Any]] = (),
    ) -> Any:
        evaluator = self.evaluator_path.expanduser().resolve()
        harness = self.harness_path.expanduser().resolve()
        dataset_path = self.dataset_path.expanduser().resolve()
        evaluation_root = self.evaluation_root.expanduser().resolve()
        if self.dataset_source not in SWEBENCH_EVALUATION_SOURCES:
            raise SWEbenchHarnessUnavailable(
                f"unsupported SWE-bench dataset source: {self.dataset_source}"
            )
        if not harness.is_dir():
            raise SWEbenchHarnessUnavailable(
                f"official SWE-bench harness checkout is unavailable: {harness}"
            )
        if self.dataset_source == SWEBENCH_DATASET_SOURCE_VERIFIED:
            if not dataset_path.is_dir():
                raise SWEbenchHarnessUnavailable(
                    f"SWE-bench Verified dataset is unavailable: {dataset_path}"
                )
        elif not dataset_path.is_file():
            raise SWEbenchHarnessUnavailable(
                f"SWE-bench regular-dev dataset is unavailable: {dataset_path}"
            )
        if not self.docker_namespace.strip():
            raise SWEbenchHarnessUnavailable("SWE-bench Docker namespace is empty")
        if self.timeout_seconds <= 0:
            raise SWEbenchHarnessUnavailable("SWE-bench timeout must be positive")
        evaluation_root.mkdir(parents=True, exist_ok=True)
        os.environ.update(
            {
                "SKILLEV_FORMAL_RUNTIME": "1",
                # SkillFlow names this variable after Verified.  The thin
                # regular-dev adapter primes its evaluator cache directly and
                # therefore never asks the upstream loader to open this JSONL.
                "SWE_BENCH_VERIFIED_PATH": str(dataset_path),
                "SWEBENCH_HARNESS_PATH": str(harness),
                "SWE_BENCH_EVALUATION_ROOT": str(evaluation_root),
                "SWE_BENCH_DOCKER_NAMESPACE": self.docker_namespace.strip(),
            }
        )
        module = _load_skillflow_evaluator(evaluator)
        module.VERIFIED_DS_PATH = str(dataset_path)
        source_receipt = (self.dataset_source, str(dataset_path))
        if getattr(module, "_flowsteer_dataset_source_receipt", None) != source_receipt:
            module._verified_cache = {}
            module._flowsteer_dataset_source_receipt = source_receipt
        if self.dataset_source == SWEBENCH_DATASET_SOURCE_VERIFIED:
            try:
                module._load_verified_dataset()
            except Exception as exc:
                raise SWEbenchHarnessUnavailable(
                    "SkillFlow could not load SWE-bench Verified"
                ) from exc
        else:
            if not records:
                raise SWEbenchHarnessUnavailable(
                    "regular-dev evaluation requires selected evaluator records"
                )
            rows: dict[str, dict[str, Any]] = {}
            for record in records:
                _require_source_split(record, SWEBENCH_DATASET_SOURCE_REGULAR_DEV)
                instance_id, row = _regular_evaluator_row(record)
                if instance_id in rows:
                    raise SWEbenchHarnessUnavailable(
                        "regular-dev harness received duplicate instance_id values"
                    )
                rows[instance_id] = row
            module._verified_cache = rows
        return module

    def preflight(
        self,
        records: Sequence[TaskRecord | Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        """Require the pinned source rows, harness, and a live Docker daemon."""

        module = self._configured_module(records)
        instance_ids = [_instance_id(record) for record in records]
        if len(set(instance_ids)) != len(instance_ids):
            raise SWEbenchHarnessUnavailable(
                "selected SWE-bench records contain duplicate instance_id values"
            )
        expected_source = self.dataset_source
        for record in records:
            _require_source_split(record, expected_source)
        missing = [
            value for value in instance_ids if value not in module._verified_cache
        ]
        if missing:
            raise SWEbenchHarnessUnavailable(
                f"{len(missing)} selected SWE-bench instance(s) are absent from "
                f"{self.dataset_source}"
            )
        harness = self.harness_path.expanduser().resolve()
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        try:
            import docker
            from swebench.harness import run_evaluation  # noqa: F401

            client = docker.from_env(timeout=60)
            try:
                client.ping()
            finally:
                client.close()
        except Exception as exc:
            raise SWEbenchHarnessUnavailable(
                "official SWE-bench Docker harness is unavailable"
            ) from exc
        return {
            "evaluator": "SkillFlow training.swe_bench_eval.evaluate_patch",
            "harness_path": str(harness),
            "dataset_source": self.dataset_source,
            "dataset_path": str(self.dataset_path.expanduser().resolve()),
            "selected_instances": len(instance_ids),
            "docker_namespace": self.docker_namespace.strip(),
            "proxy_metric_used": False,
        }

    async def __call__(
        self,
        record: TaskRecord | Mapping[str, Any],
        prediction: str,
    ) -> Mapping[str, Any]:
        _require_source_split(record, self.dataset_source)
        module = self._configured_module((record,))
        instance_id = _instance_id(record)
        resolved, score, details = await asyncio.to_thread(
            module.evaluate_patch,
            instance_id,
            str(prediction),
            timeout=self.timeout_seconds,
        )
        return {
            "resolved": bool(resolved),
            "official_score": float(score),
            "harness_details": str(details),
            "instance_id": instance_id,
            "proxy_metric_used": False,
        }


__all__ = [
    "DEFAULT_SKILLFLOW_SWE_EVALUATOR",
    "DEFAULT_SWEBENCH_HARNESS",
    "DEFAULT_SWEBENCH_VERIFIED",
    "OfficialSWEbenchHarness",
    "SWEBENCH_DATASET_SOURCE_REGULAR_DEV",
    "SWEBENCH_DATASET_SOURCE_VERIFIED",
    "SWEBENCH_EVALUATION_SOURCES",
    "SWEbenchHarnessUnavailable",
]
