"""Materialized held-out and read-only GPU evidence for W&B step logging.

The smoke runner deliberately does not own a validation loop.  This module
implements its ``WandbStepEvidenceProvider`` protocol without introducing a
second evaluator: it consumes the report and trajectory receipts produced by
the existing AgentGraph evaluation path, revalidates their fixed-denominator
aggregate, and reads CUDA allocator telemetry.  It never initializes W&B,
loads credentials, calls a model, or synthesizes validation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .records import EvaluationReceipt, TaskRecord
from .task_evaluator import TRIVIAQA_ANSWER_EVALUATOR_VERSION
from .wandb_training import WandbTrainingError


TRIVIAQA_HELD_OUT_PROTOCOL = (
    "TriviaQA_official_normalization_exact_match_and_token_F1"
)
_SUPPORTED_REPORT_SCHEMAS = frozenset(
    {
        "flowsteer.completion_benchmark.round_report.v1",
        "flowsteer.triviaqa.round_report.v1",
    }
)
_METRIC_NAMES = ("exact_match", "token_f1")


class WandbEvidenceError(WandbTrainingError):
    """Materialized validation or GPU telemetry violated its contract."""


class GpuTelemetryReader(Protocol):
    """Read-only telemetry dependency used by the evidence provider."""

    def preflight(self) -> None: ...

    def read(self) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class HeldOutValidationSpec:
    """Frozen comparison coordinates for one formal held-out evaluation."""

    expected_task_ids: tuple[str, ...]
    dataset_key: str = "triviaqa"
    split: str = "validation"
    evaluator_version: str = TRIVIAQA_ANSWER_EVALUATOR_VERSION
    protocol: str = TRIVIAQA_HELD_OUT_PROTOCOL
    evaluation_scope: str = "held_out"
    primary_metric: str = "exact_match"

    def __post_init__(self) -> None:
        if not self.expected_task_ids:
            raise ValueError("held-out task IDs cannot be empty")
        if any(
            not isinstance(task_id, str) or not task_id.strip()
            for task_id in self.expected_task_ids
        ):
            raise ValueError("held-out task IDs must be non-empty strings")
        if len(set(self.expected_task_ids)) != len(self.expected_task_ids):
            raise ValueError("held-out task IDs must be unique")
        for name in (
            "dataset_key",
            "split",
            "evaluator_version",
            "protocol",
            "evaluation_scope",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.primary_metric not in _METRIC_NAMES:
            raise ValueError(
                "TriviaQA primary_metric must be exact_match or token_f1"
            )

    @property
    def denominator(self) -> int:
        return len(self.expected_task_ids)


class TorchGpuTelemetryReader:
    """Read CUDA allocator counters without resetting peaks or selecting a device."""

    def __init__(
        self,
        device_indices: Sequence[int],
        *,
        torch_module: Any = None,
    ) -> None:
        indices = tuple(device_indices)
        if not indices:
            raise ValueError("at least one CUDA device index is required")
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("CUDA device indices must be non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("CUDA device indices must be unique")
        self._device_indices = indices
        self._torch_module = torch_module

    def _torch(self) -> Any:
        if self._torch_module is None:  # pragma: no cover - host dependent
            try:
                import torch
            except ImportError as exc:
                raise WandbEvidenceError(
                    "PyTorch is unavailable for CUDA telemetry"
                ) from exc
            self._torch_module = torch
        return self._torch_module

    def preflight(self) -> None:
        torch = self._torch()
        cuda = getattr(torch, "cuda", None)
        if cuda is None or cuda.is_available() is not True:
            raise WandbEvidenceError("CUDA telemetry is unavailable")
        count = cuda.device_count()
        if type(count) is not int or count <= max(self._device_indices):
            raise WandbEvidenceError(
                "a configured CUDA telemetry device is not visible"
            )
        for index in self._device_indices:
            properties = cuda.get_device_properties(index)
            total_memory = getattr(properties, "total_memory", None)
            if (
                isinstance(total_memory, bool)
                or not isinstance(total_memory, (int, float))
                or not math.isfinite(float(total_memory))
                or float(total_memory) <= 0.0
            ):
                raise WandbEvidenceError(
                    f"CUDA device {index} has no finite total-memory telemetry"
                )

    def read(self) -> Mapping[str, float]:
        self.preflight()
        cuda = self._torch().cuda
        metrics: dict[str, float] = {
            "visible_device_count": float(cuda.device_count()),
            "tracked_device_count": float(len(self._device_indices)),
        }
        readers = {
            "memory_allocated_bytes": cuda.memory_allocated,
            "memory_reserved_bytes": cuda.memory_reserved,
            "max_memory_allocated_bytes": cuda.max_memory_allocated,
            "max_memory_reserved_bytes": cuda.max_memory_reserved,
        }
        for index in self._device_indices:
            prefix = f"cuda_{index}"
            properties = cuda.get_device_properties(index)
            metrics[f"{prefix}/total_memory_bytes"] = float(
                properties.total_memory
            )
            for metric_name, reader in readers.items():
                value = float(reader(index))
                if not math.isfinite(value) or value < 0.0:
                    raise WandbEvidenceError(
                        f"CUDA {metric_name} is not a finite non-negative value"
                    )
                metrics[f"{prefix}/{metric_name}"] = value
        return metrics


@dataclass(frozen=True)
class _ValidationMaterialization:
    policy_version: str
    metrics: Mapping[str, float]
    report_schema_version: str


class MaterializedWandbStepEvidenceProvider:
    """Adapt existing evaluator receipts to the runner's W&B evidence protocol.

    ``validation_report_path`` and ``validation_trajectories_path`` are output
    paths owned by the existing held-out evaluation driver.  By default those
    outputs must be created or replaced after ``preflight``; this prevents a
    previous optimizer step's receipt from being reused silently.
    """

    def __init__(
        self,
        *,
        validation_report_path: str | Path,
        validation_trajectories_path: str | Path,
        validation_spec: HeldOutValidationSpec,
        gpu_telemetry: GpuTelemetryReader,
        previous_best_report_path: str | Path | None = None,
        previous_best_trajectories_path: str | Path | None = None,
        require_fresh_after_preflight: bool = True,
    ) -> None:
        if (previous_best_report_path is None) != (
            previous_best_trajectories_path is None
        ):
            raise ValueError(
                "previous-best report and trajectories must be supplied together"
            )
        self._report_path = Path(validation_report_path)
        self._trajectories_path = Path(validation_trajectories_path)
        self._spec = validation_spec
        self._gpu_telemetry = gpu_telemetry
        self._best_report_path = (
            Path(previous_best_report_path)
            if previous_best_report_path is not None
            else None
        )
        self._best_trajectories_path = (
            Path(previous_best_trajectories_path)
            if previous_best_trajectories_path is not None
            else None
        )
        self._require_fresh = bool(require_fresh_after_preflight)
        self._preflight_signatures: dict[Path, tuple[int, int, int]] = {}
        self._preflight_complete = False

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        if not path.is_file():
            raise WandbEvidenceError(f"evidence path is not a file: {path}")
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def preflight(
        self,
        *,
        config: Mapping[str, Any],
        selected_tasks: Sequence[TaskRecord],
    ) -> None:
        if not isinstance(config, Mapping):
            raise WandbEvidenceError("training config must be a mapping")
        if not selected_tasks:
            raise WandbEvidenceError("training task selection cannot be empty")
        held_out = set(self._spec.expected_task_ids)
        observed_ids: set[str] = set()
        for task in selected_tasks:
            if not isinstance(task, TaskRecord):
                raise WandbEvidenceError(
                    "selected training tasks must use TaskRecord"
                )
            if task.split != "train":
                raise WandbEvidenceError(
                    "W&B training preflight accepts only train-split tasks"
                )
            if task.metadata.get("dataset_key") != self._spec.dataset_key:
                raise WandbEvidenceError(
                    "training task dataset differs from validation contract"
                )
            if task.task_id in held_out:
                raise WandbEvidenceError(
                    "a selected training task overlaps the held-out task set"
                )
            if task.task_id in observed_ids:
                raise WandbEvidenceError("selected training task IDs are duplicated")
            observed_ids.add(task.task_id)

        self._gpu_telemetry.preflight()
        if self._best_report_path is not None:
            best_report = self._read_json(self._best_report_path)
            best_policy = self._required_string(best_report, "policy_version")
            self._load_validation(
                report_path=self._best_report_path,
                trajectories_path=self._best_trajectories_path,
                expected_policy_version=best_policy,
            )

        self._preflight_signatures = {}
        for path in (self._report_path, self._trajectories_path):
            signature = self._signature(path)
            if signature is not None:
                self._preflight_signatures[path] = signature
        self._preflight_complete = True

    def optimizer_step_evidence(
        self,
        *,
        config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self._preflight_complete:
            raise WandbEvidenceError("evidence preflight was not completed")
        if not all(
            isinstance(value, Mapping) for value in (config, manifest, summary)
        ):
            raise WandbEvidenceError(
                "optimizer-step config, manifest, and summary must be mappings"
            )
        updated_policy = self._required_string(
            summary, "updated_policy_version"
        )
        checkpoint_version = self._required_string(summary, "checkpoint_version")

        if self._require_fresh:
            for path in (self._report_path, self._trajectories_path):
                current = self._signature(path)
                if current is None:
                    raise WandbEvidenceError(
                        f"held-out evidence is not materialized: {path}"
                    )
                if self._preflight_signatures.get(path) == current:
                    raise WandbEvidenceError(
                        "held-out evidence was not refreshed after preflight: "
                        f"{path}"
                    )

        current = self._load_validation(
            report_path=self._report_path,
            trajectories_path=self._trajectories_path,
            expected_policy_version=updated_policy,
        )
        is_best = False
        comparison: dict[str, bool] | None = None
        if self._best_report_path is not None:
            best_report = self._read_json(self._best_report_path)
            best_policy = self._required_string(best_report, "policy_version")
            previous = self._load_validation(
                report_path=self._best_report_path,
                trajectories_path=self._best_trajectories_path,
                expected_policy_version=best_policy,
            )
            is_best = (
                current.metrics[self._spec.primary_metric]
                > previous.metrics[self._spec.primary_metric] + 1e-12
            )
            comparison = {
                "same_protocol": True,
                "same_split": True,
                "same_task_denominator": True,
                "improved": is_best,
            }

        validation: dict[str, Any] = {
            "complete": True,
            "status": "materialized_evaluator_receipts_verified",
            "scope": "held_out",
            "split": self._spec.split,
            "dataset": self._spec.dataset_key,
            "evaluator_version": self._spec.evaluator_version,
            "protocol": self._spec.protocol,
            "task_denominator": self._spec.denominator,
            "policy_version": current.policy_version,
            "checkpoint_version": checkpoint_version,
            "report_schema_version": current.report_schema_version,
            "metrics": dict(current.metrics),
            "is_best": is_best,
            "receipt_paths": {
                "report": str(self._report_path),
                "trajectories": str(self._trajectories_path),
            },
        }
        if comparison is not None:
            validation["best_comparison"] = comparison
        return {
            "validation": validation,
            "gpu_metrics": dict(self._gpu_telemetry.read()),
        }

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WandbEvidenceError(
                f"held-out report is not materialized: {path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise WandbEvidenceError(
                f"held-out report is unreadable: {path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise WandbEvidenceError("held-out report must be a mapping")
        return value

    @staticmethod
    def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise WandbEvidenceError(
                f"held-out trajectories are not materialized: {path}"
            ) from exc
        except OSError as exc:
            raise WandbEvidenceError(
                f"held-out trajectories are unreadable: {path}"
            ) from exc
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WandbEvidenceError(
                    f"held-out trajectory line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise WandbEvidenceError(
                    f"held-out trajectory line {line_number} is not a mapping"
                )
            rows.append(value)
        return tuple(rows)

    @staticmethod
    def _required_string(value: Mapping[str, Any], name: str) -> str:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise WandbEvidenceError(f"{name} must be a non-empty string")
        return item.strip()

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        if type(value) is not int or value < 0:
            raise WandbEvidenceError(f"{name} must be a non-negative integer")
        return value

    @staticmethod
    def _unit_metric(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WandbEvidenceError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise WandbEvidenceError(f"{name} must be finite and in [0, 1]")
        return result

    def _load_validation(
        self,
        *,
        report_path: Path,
        trajectories_path: Path | None,
        expected_policy_version: str,
    ) -> _ValidationMaterialization:
        if trajectories_path is None:
            raise WandbEvidenceError("held-out trajectories path is unavailable")
        report = self._read_json(report_path)
        schema = self._required_string(report, "schema_version")
        if schema not in _SUPPORTED_REPORT_SCHEMAS:
            raise WandbEvidenceError("held-out report schema is unsupported")
        dataset_key = report.get("dataset_key")
        if dataset_key is None and report.get("dataset") == "TriviaQA":
            dataset_key = "triviaqa"
        if dataset_key != self._spec.dataset_key:
            raise WandbEvidenceError(
                "held-out report dataset differs from validation contract"
            )
        if report.get("project_split") != self._spec.split:
            raise WandbEvidenceError(
                "held-out report split differs from validation contract"
            )
        if report.get("evaluation_scope") != self._spec.evaluation_scope:
            raise WandbEvidenceError(
                "held-out report evaluation scope differs from validation contract"
            )
        if report.get("metric_scope") != self._spec.protocol:
            raise WandbEvidenceError(
                "held-out report protocol differs from validation contract"
            )
        if report.get("policy_version") != expected_policy_version:
            raise WandbEvidenceError(
                "held-out report policy differs from the updated policy"
            )
        if not isinstance(report.get("completed_at"), str) or not str(
            report["completed_at"]
        ).strip():
            raise WandbEvidenceError("held-out report is not marked complete")
        denominator = self._spec.denominator
        if self._integer(report.get("sample_count"), "sample_count") != denominator:
            raise WandbEvidenceError(
                "held-out report sample count differs from the frozen denominator"
            )

        aggregate = report.get("agentgraph")
        if not isinstance(aggregate, Mapping):
            raise WandbEvidenceError("held-out report has no AgentGraph aggregate")
        for name in ("denominator", "completed", "evaluator_valid"):
            if self._integer(aggregate.get(name), f"agentgraph.{name}") != denominator:
                raise WandbEvidenceError(
                    f"agentgraph.{name} differs from the frozen denominator"
                )

        rows = self._read_jsonl(trajectories_path)
        if len(rows) != denominator:
            raise WandbEvidenceError(
                "held-out trajectory count differs from the frozen denominator"
            )
        expected_ids = set(self._spec.expected_task_ids)
        observed_ids: set[str] = set()
        metric_sums = {name: 0.0 for name in _METRIC_NAMES}
        for row in rows:
            try:
                task = TaskRecord.from_dict(row["task"])
                evaluation = EvaluationReceipt.from_dict(row["evaluation"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WandbEvidenceError(
                    "held-out trajectory task/evaluator receipt is invalid"
                ) from exc
            if task.task_id in observed_ids:
                raise WandbEvidenceError("held-out trajectory task IDs are duplicated")
            observed_ids.add(task.task_id)
            if task.task_id not in expected_ids:
                raise WandbEvidenceError(
                    "held-out trajectory contains a task outside the frozen set"
                )
            if task.split != self._spec.split:
                raise WandbEvidenceError(
                    "held-out trajectory task has the wrong split"
                )
            if task.metadata.get("dataset_key") != self._spec.dataset_key:
                raise WandbEvidenceError(
                    "held-out trajectory task has the wrong dataset"
                )
            versions = row.get("versions")
            if not isinstance(versions, Mapping):
                raise WandbEvidenceError(
                    "held-out trajectory has no version receipt"
                )
            if versions.get("policy") != expected_policy_version:
                raise WandbEvidenceError(
                    "held-out trajectory policy differs from the updated policy"
                )
            if versions.get("evaluator") != self._spec.evaluator_version:
                raise WandbEvidenceError(
                    "held-out trajectory evaluator version differs"
                )
            if (
                not evaluation.valid
                or evaluation.evaluator_version != self._spec.evaluator_version
                or evaluation.reason != "evaluated"
            ):
                raise WandbEvidenceError(
                    "held-out trajectory lacks a valid terminal evaluator receipt"
                )
            if evaluation.details.get("metric_scope") != "answer_only":
                raise WandbEvidenceError(
                    "held-out trajectory evaluator metric scope differs"
                )
            if any(
                row.get(flag) is True
                for flag in (
                    "forced_probe",
                    "api_fallback_used",
                    "manual_repair_used",
                )
            ):
                raise WandbEvidenceError(
                    "formal held-out validation contains an intervention or fallback"
                )
            for name in _METRIC_NAMES:
                metric_sums[name] += self._unit_metric(
                    evaluation.metrics.get(name),
                    f"evaluation.metrics.{name}",
                )
            reward = self._unit_metric(evaluation.reward, "evaluation.reward")
            token_f1 = self._unit_metric(
                evaluation.metrics.get("token_f1"),
                "evaluation.metrics.token_f1",
            )
            if not math.isclose(reward, token_f1, abs_tol=1e-12):
                raise WandbEvidenceError(
                    "TriviaQA terminal reward differs from token F1"
                )

        if observed_ids != expected_ids:
            raise WandbEvidenceError(
                "held-out trajectories do not cover the frozen task set"
            )
        metrics = {
            name: metric_sums[name] / denominator for name in _METRIC_NAMES
        }
        metrics.update(
            {
                "em_percent": metrics["exact_match"] * 100.0,
                "f1_percent": metrics["token_f1"] * 100.0,
                "task_denominator": float(denominator),
                "evaluator_valid": float(denominator),
            }
        )
        for name in _METRIC_NAMES:
            reported = self._unit_metric(
                aggregate.get(f"strict_{name}"),
                f"agentgraph.strict_{name}",
            )
            if not math.isclose(reported, metrics[name], abs_tol=1e-12):
                raise WandbEvidenceError(
                    f"reported {name} differs from terminal evaluator receipts"
                )
        return _ValidationMaterialization(
            policy_version=expected_policy_version,
            metrics=metrics,
            report_schema_version=schema,
        )


__all__ = [
    "GpuTelemetryReader",
    "HeldOutValidationSpec",
    "MaterializedWandbStepEvidenceProvider",
    "TRIVIAQA_HELD_OUT_PROTOCOL",
    "TorchGpuTelemetryReader",
    "WandbEvidenceError",
]
