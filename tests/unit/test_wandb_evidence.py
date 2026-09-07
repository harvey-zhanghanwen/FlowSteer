from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.interactive.records import TaskRecord
from src.interactive.task_evaluator import TRIVIAQA_ANSWER_EVALUATOR_VERSION
from src.interactive.wandb_evidence import (
    HeldOutValidationSpec,
    MaterializedWandbStepEvidenceProvider,
    TRIVIAQA_HELD_OUT_PROTOCOL,
    TorchGpuTelemetryReader,
    WandbEvidenceError,
)


class _GpuTelemetry:
    def __init__(self) -> None:
        self.preflight_count = 0
        self.read_count = 0

    def preflight(self) -> None:
        self.preflight_count += 1

    def read(self):
        self.read_count += 1
        return {
            "tracked_device_count": 2.0,
            "cuda_3/max_memory_allocated_bytes": 2048.0,
        }


def _task(task_id: str, *, split: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        question=f"Question for {task_id}?",
        ground_truth="answer",
        split=split,
        metadata={"dataset_key": "triviaqa"},
    )


def _trajectory(
    task_id: str,
    *,
    policy: str,
    exact_match: float,
    token_f1: float,
    evaluator: str = TRIVIAQA_ANSWER_EVALUATOR_VERSION,
) -> dict:
    return {
        "task": _task(task_id, split="validation").to_dict(),
        "versions": {"policy": policy, "evaluator": evaluator},
        "evaluation": {
            "evaluator_version": evaluator,
            "valid": True,
            "reward": token_f1,
            "metrics": {
                "exact_match": exact_match,
                "token_f1": token_f1,
            },
            "reason": "evaluated",
            "details": {"metric_scope": "answer_only"},
        },
        "forced_probe": False,
        "api_fallback_used": False,
        "manual_repair_used": False,
    }


def _materialize(
    report_path: Path,
    trajectories_path: Path,
    *,
    task_ids: tuple[str, ...],
    policy: str,
    values: tuple[tuple[float, float], ...],
) -> None:
    exact_match = sum(value[0] for value in values) / len(values)
    token_f1 = sum(value[1] for value in values) / len(values)
    report = {
        "schema_version": "flowsteer.completion_benchmark.round_report.v1",
        "dataset_key": "triviaqa",
        "dataset": "TriviaQA",
        "project_split": "validation",
        "evaluation_scope": "held_out",
        "metric_scope": TRIVIAQA_HELD_OUT_PROTOCOL,
        "sample_count": len(task_ids),
        "policy_version": policy,
        "completed_at": "2026-09-07T00:00:00+00:00",
        "agentgraph": {
            "denominator": len(task_ids),
            "completed": len(task_ids),
            "evaluator_valid": len(task_ids),
            "strict_exact_match": exact_match,
            "strict_token_f1": token_f1,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    trajectories_path.write_text(
        "".join(
            json.dumps(
                _trajectory(
                    task_id,
                    policy=policy,
                    exact_match=metrics[0],
                    token_f1=metrics[1],
                )
            )
            + "\n"
            for task_id, metrics in zip(task_ids, values)
        ),
        encoding="utf-8",
    )


def _provider(tmp_path: Path, *, fresh: bool = True):
    task_ids = ("triviaqa:heldout:1", "triviaqa:heldout:2")
    gpu = _GpuTelemetry()
    provider = MaterializedWandbStepEvidenceProvider(
        validation_report_path=tmp_path / "current" / "report.json",
        validation_trajectories_path=tmp_path / "current" / "trajectories.jsonl",
        validation_spec=HeldOutValidationSpec(expected_task_ids=task_ids),
        gpu_telemetry=gpu,
        require_fresh_after_preflight=fresh,
    )
    return provider, task_ids, gpu


def test_materialized_provider_recomputes_official_heldout_metrics(tmp_path) -> None:
    provider, task_ids, gpu = _provider(tmp_path)
    provider.preflight(
        config={}, selected_tasks=(_task("triviaqa:train:1", split="train"),)
    )
    _materialize(
        tmp_path / "current" / "report.json",
        tmp_path / "current" / "trajectories.jsonl",
        task_ids=task_ids,
        policy="policy-step-0001",
        values=((1.0, 1.0), (0.0, 0.5)),
    )

    evidence = provider.optimizer_step_evidence(
        config={},
        manifest={},
        summary={
            "updated_policy_version": "policy-step-0001",
            "checkpoint_version": "checkpoint-step-0001",
        },
    )

    validation = evidence["validation"]
    assert validation["complete"] is True
    assert validation["scope"] == "held_out"
    assert validation["task_denominator"] == 2
    assert validation["metrics"]["exact_match"] == 0.5
    assert validation["metrics"]["token_f1"] == 0.75
    assert validation["metrics"]["em_percent"] == 50.0
    assert validation["metrics"]["f1_percent"] == 75.0
    assert validation["is_best"] is False
    assert evidence["gpu_metrics"]["tracked_device_count"] == 2.0
    assert gpu.preflight_count == 1
    assert gpu.read_count == 1


def test_preflight_rejects_train_heldout_overlap(tmp_path) -> None:
    provider, task_ids, _ = _provider(tmp_path)
    with pytest.raises(WandbEvidenceError, match="overlaps"):
        provider.preflight(
            config={}, selected_tasks=(_task(task_ids[0], split="train"),)
        )


def test_provider_rejects_unchanged_preflight_receipt(tmp_path) -> None:
    provider, task_ids, _ = _provider(tmp_path)
    _materialize(
        tmp_path / "current" / "report.json",
        tmp_path / "current" / "trajectories.jsonl",
        task_ids=task_ids,
        policy="policy-step-0001",
        values=((1.0, 1.0), (0.0, 0.0)),
    )
    provider.preflight(
        config={}, selected_tasks=(_task("triviaqa:train:1", split="train"),)
    )
    with pytest.raises(WandbEvidenceError, match="not refreshed"):
        provider.optimizer_step_evidence(
            config={},
            manifest={},
            summary={
                "updated_policy_version": "policy-step-0001",
                "checkpoint_version": "checkpoint-step-0001",
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("project_split", "train", "split differs"),
        ("evaluation_scope", "transductive", "evaluation scope differs"),
        ("metric_scope", "different-protocol", "protocol differs"),
        ("policy_version", "old-policy", "policy differs"),
        ("sample_count", 1, "sample count differs"),
    ),
)
def test_provider_rejects_mismatched_report_coordinates(
    tmp_path, field, value, message
) -> None:
    provider, task_ids, _ = _provider(tmp_path, fresh=False)
    report_path = tmp_path / "current" / "report.json"
    trajectories_path = tmp_path / "current" / "trajectories.jsonl"
    _materialize(
        report_path,
        trajectories_path,
        task_ids=task_ids,
        policy="policy-step-0001",
        values=((1.0, 1.0), (0.0, 0.0)),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    provider.preflight(
        config={}, selected_tasks=(_task("triviaqa:train:1", split="train"),)
    )
    with pytest.raises(WandbEvidenceError, match=message):
        provider.optimizer_step_evidence(
            config={},
            manifest={},
            summary={
                "updated_policy_version": "policy-step-0001",
                "checkpoint_version": "checkpoint-step-0001",
            },
        )


def test_provider_rejects_wrong_terminal_evaluator_receipt(tmp_path) -> None:
    provider, task_ids, _ = _provider(tmp_path, fresh=False)
    report_path = tmp_path / "current" / "report.json"
    trajectories_path = tmp_path / "current" / "trajectories.jsonl"
    _materialize(
        report_path,
        trajectories_path,
        task_ids=task_ids,
        policy="policy-step-0001",
        values=((1.0, 1.0), (0.0, 0.0)),
    )
    rows = trajectories_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["evaluation"]["evaluator_version"] = "wrong-evaluator"
    rows[0] = json.dumps(first)
    trajectories_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    provider.preflight(
        config={}, selected_tasks=(_task("triviaqa:train:1", split="train"),)
    )
    with pytest.raises(WandbEvidenceError, match="valid terminal evaluator"):
        provider.optimizer_step_evidence(
            config={},
            manifest={},
            summary={
                "updated_policy_version": "policy-step-0001",
                "checkpoint_version": "checkpoint-step-0001",
            },
        )


def test_best_alias_requires_same_contract_materialized_comparison(tmp_path) -> None:
    task_ids = ("triviaqa:heldout:1", "triviaqa:heldout:2")
    best_report = tmp_path / "best" / "report.json"
    best_trajectories = tmp_path / "best" / "trajectories.jsonl"
    _materialize(
        best_report,
        best_trajectories,
        task_ids=task_ids,
        policy="policy-step-0000",
        values=((1.0, 1.0), (0.0, 0.0)),
    )
    provider = MaterializedWandbStepEvidenceProvider(
        validation_report_path=tmp_path / "current" / "report.json",
        validation_trajectories_path=tmp_path / "current" / "trajectories.jsonl",
        validation_spec=HeldOutValidationSpec(expected_task_ids=task_ids),
        gpu_telemetry=_GpuTelemetry(),
        previous_best_report_path=best_report,
        previous_best_trajectories_path=best_trajectories,
    )
    provider.preflight(
        config={}, selected_tasks=(_task("triviaqa:train:1", split="train"),)
    )
    _materialize(
        tmp_path / "current" / "report.json",
        tmp_path / "current" / "trajectories.jsonl",
        task_ids=task_ids,
        policy="policy-step-0001",
        values=((1.0, 1.0), (1.0, 1.0)),
    )
    validation = provider.optimizer_step_evidence(
        config={},
        manifest={},
        summary={
            "updated_policy_version": "policy-step-0001",
            "checkpoint_version": "checkpoint-step-0001",
        },
    )["validation"]
    assert validation["is_best"] is True
    assert validation["best_comparison"] == {
        "same_protocol": True,
        "same_split": True,
        "same_task_denominator": True,
        "improved": True,
    }


class _Cuda:
    def __init__(self) -> None:
        self.mutating_calls: list[str] = []

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 6

    @staticmethod
    def get_device_properties(index):
        return SimpleNamespace(total_memory=(index + 1) * 1_000_000)

    @staticmethod
    def memory_allocated(index):
        return index * 100.0

    @staticmethod
    def memory_reserved(index):
        return index * 200.0

    @staticmethod
    def max_memory_allocated(index):
        return index * 300.0

    @staticmethod
    def max_memory_reserved(index):
        return index * 400.0


def test_torch_gpu_reader_only_reads_allocator_telemetry() -> None:
    cuda = _Cuda()
    reader = TorchGpuTelemetryReader(
        (3, 5), torch_module=SimpleNamespace(cuda=cuda)
    )
    reader.preflight()
    metrics = reader.read()
    assert metrics["tracked_device_count"] == 2.0
    assert metrics["cuda_3/memory_allocated_bytes"] == 300.0
    assert metrics["cuda_5/max_memory_reserved_bytes"] == 2000.0
    assert cuda.mutating_calls == []


def test_torch_gpu_reader_fails_closed_without_visible_cuda() -> None:
    cuda = SimpleNamespace(is_available=lambda: False)
    reader = TorchGpuTelemetryReader(
        (0,), torch_module=SimpleNamespace(cuda=cuda)
    )
    with pytest.raises(WandbEvidenceError, match="unavailable"):
        reader.preflight()
