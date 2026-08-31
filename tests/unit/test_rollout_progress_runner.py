from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from src.interactive.records import TaskRecord
from tests.unit.test_probe_runtime_wiring import (
    ProbeRuntimeWiringTests as _ProbeRuntimeWiringTests,
    _MODULE as _SMOKE_MODULE,
    _task as _smoke_task,
    _versions as _smoke_versions,
)


def _probe_backend(factory=_ProbeRuntimeWiringTests):
    return factory()._backend()


del _ProbeRuntimeWiringTests


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_hotpotqa_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_hotpotqa_round_progress_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _event(index: int, stage: str = "rollout_step_committed") -> dict[str, object]:
    return {
        "task_id": f"task-{index}",
        "rollout_id": f"rollout-{index}",
        "round_index": index,
        "stage": stage,
        "action": "add_subgraph",
        "accepted": True,
        "done": False,
        "graph_revision": index + 1,
        "timestamp": "2026-08-31T00:00:00Z",
        "error_type": None,
    }


def test_rollout_progress_sink_is_thread_safe_and_append_only(tmp_path):
    path = tmp_path / "progress.jsonl"
    sink = _MODULE._RolloutProgressJsonlSink(
        path,
        run_attempt_id="run-attempt-1",
        condition_id="condition-1",
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(sink, (_event(index) for index in range(32))))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 32
    assert {row["event"]["task_id"] for row in rows} == {
        f"task-{index}" for index in range(32)
    }
    assert all(
        row["schema_version"] == _MODULE.ROLLOUT_PROGRESS_SCHEMA_VERSION
        and row["run_attempt_id"] == "run-attempt-1"
        and row["condition_id"] == "condition-1"
        for row in rows
    )


def test_progress_sink_is_not_created_without_explicit_storage_path(tmp_path):
    callback = _MODULE._rollout_progress_callback(
        {"storage": {}},
        tmp_path,
        run_attempt_id="run-attempt-1",
        condition_id="condition-1",
    )

    assert callback is None
    assert tuple(tmp_path.iterdir()) == ()


def test_live_backend_passes_progress_callback_to_collector():
    captured = []
    sentinel = object()

    class CapturingCollector:
        def __init__(self, _orchestrator, _environment, _versions, _store, **kwargs):
            captured.append(kwargs)

        async def collect(self, *_args, **_kwargs):
            return sentinel

    callback = lambda _event: None
    backend = _probe_backend()
    with patch.object(
        _SMOKE_MODULE,
        "AgentGraphRolloutCollector",
        CapturingCollector,
    ):
        result = asyncio.run(
            backend.collect(
                _smoke_task(),
                0,
                _smoke_versions(),
                expected_task_split="validation",
                progress_callback=callback,
            )
        )

    assert result is sentinel
    assert captured[0]["progress_callback"] is callback


def test_outer_timeout_preserves_progress_without_scored_trajectory(tmp_path):
    task = TaskRecord(
        task_id="hotpotqa:progress-timeout",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa", "source": "HotpotQA"},
    )

    class EmptyTrajectoryStore:
        def payloads(self):
            return ()

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = type(
            "Evidence",
            (),
            {"trajectories": EmptyTrajectoryStore()},
        )()

        async def collect(
            self,
            task,
            rollout_index,
            versions,
            *,
            expected_task_split="train",
            progress_callback=None,
        ):
            del rollout_index, versions
            assert expected_task_split == "validation"
            assert progress_callback is not None
            committed = _event(0)
            committed["task_id"] = task.task_id
            committed["rollout_id"] = "rollout-timeout"
            progress_callback(committed)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled = dict(committed)
                cancelled.update(
                    stage="rollout_cancelled",
                    action=None,
                    accepted=None,
                    error_type="CancelledError",
                )
                progress_callback(cancelled)
                raise

    progress_path = tmp_path / "progress.jsonl"
    trajectory_path = tmp_path / "trajectories.jsonl"
    config = {
        "experiment": {
            "condition_id": "condition-timeout",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {
            "concurrency": 1,
            "split": "validation",
            "task_timeout_seconds": 0.01,
        },
        "storage": {"rollout_progress_path": str(progress_path)},
    }
    failures = []

    result = asyncio.run(
        _MODULE._collect_graph(
            Backend(),
            (task,),
            config,
            trajectory_path,
            failures,
            {},
            tmp_path / "manifest.json",
            project_root=tmp_path,
            run_attempt_id="run-attempt-timeout",
        )
    )

    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert result == {}
    assert [row["event"]["stage"] for row in rows] == [
        "rollout_step_committed",
        "rollout_cancelled",
    ]
    assert all(row["run_attempt_id"] == "run-attempt-timeout" for row in rows)
    assert all(row["condition_id"] == "condition-timeout" for row in rows)
    assert _MODULE._read_jsonl(trajectory_path) == []
    assert len(failures) == 1
    assert "TimeoutError" in failures[0]["error"]
