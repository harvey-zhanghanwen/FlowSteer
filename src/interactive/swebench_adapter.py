"""Thin adapter for SkillFlow's official SWE-bench Verified evaluator.

The patch evaluator remains the deployed SkillFlow implementation in
``training/swe_bench_eval.py``.  This module only supplies the project record
boundary, formal-runtime configuration, and a fail-closed Docker/harness
preflight; it never computes a similarity or other proxy score.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
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
    metadata = record.metadata if isinstance(record, TaskRecord) else record.get("metadata", {})
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


@dataclass(frozen=True)
class OfficialSWEbenchHarness:
    """Callable wrapper around SkillFlow's official Docker evaluator."""

    evaluator_path: Path = DEFAULT_SKILLFLOW_SWE_EVALUATOR
    harness_path: Path = DEFAULT_SWEBENCH_HARNESS
    verified_path: Path = DEFAULT_SWEBENCH_VERIFIED
    evaluation_root: Path = Path("artifacts/swebench_official_evaluation")
    docker_namespace: str = "swebench"
    timeout_seconds: int = 900

    def _configured_module(self) -> Any:
        evaluator = self.evaluator_path.expanduser().resolve()
        harness = self.harness_path.expanduser().resolve()
        verified = self.verified_path.expanduser().resolve()
        evaluation_root = self.evaluation_root.expanduser().resolve()
        if not harness.is_dir():
            raise SWEbenchHarnessUnavailable(
                f"official SWE-bench harness checkout is unavailable: {harness}"
            )
        if not verified.is_dir():
            raise SWEbenchHarnessUnavailable(
                f"SWE-bench Verified dataset is unavailable: {verified}"
            )
        if not self.docker_namespace.strip():
            raise SWEbenchHarnessUnavailable("SWE-bench Docker namespace is empty")
        if self.timeout_seconds <= 0:
            raise SWEbenchHarnessUnavailable("SWE-bench timeout must be positive")
        evaluation_root.mkdir(parents=True, exist_ok=True)
        os.environ.update(
            {
                "SKILLEV_FORMAL_RUNTIME": "1",
                "SWE_BENCH_VERIFIED_PATH": str(verified),
                "SWEBENCH_HARNESS_PATH": str(harness),
                "SWE_BENCH_EVALUATION_ROOT": str(evaluation_root),
                "SWE_BENCH_DOCKER_NAMESPACE": self.docker_namespace.strip(),
            }
        )
        module = _load_skillflow_evaluator(evaluator)
        # The deployed module reads this path at import time.  Assigning the
        # configured value keeps a reused process on the same frozen dataset.
        module.VERIFIED_DS_PATH = str(verified)
        return module

    def preflight(self, instance_ids: Sequence[str] = ()) -> Mapping[str, Any]:
        """Require the pinned harness, verified rows, and a live Docker daemon."""

        module = self._configured_module()
        try:
            module._load_verified_dataset()
        except Exception as exc:
            raise SWEbenchHarnessUnavailable(
                "SkillFlow could not load SWE-bench Verified"
            ) from exc
        missing = [value for value in instance_ids if value not in module._verified_cache]
        if missing:
            raise SWEbenchHarnessUnavailable(
                f"{len(missing)} selected SWE-bench instance(s) are absent from Verified"
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
            "verified_path": str(self.verified_path.expanduser().resolve()),
            "selected_instances": len(instance_ids),
            "docker_namespace": self.docker_namespace.strip(),
            "proxy_metric_used": False,
        }

    async def __call__(
        self,
        record: TaskRecord | Mapping[str, Any],
        prediction: str,
    ) -> Mapping[str, Any]:
        module = self._configured_module()
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
    "SWEbenchHarnessUnavailable",
]
