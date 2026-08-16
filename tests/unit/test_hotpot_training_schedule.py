from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.hotpot_training_schedule import (
    FrozenHotpotTrainingSchedule,
    HotpotTrainingCursorState,
    HotpotTrainingProgress,
    freeze_hotpot_training_schedule,
)


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "freeze_hotpot_training_schedule.py"
)
_SPEC = importlib.util.spec_from_file_location("freeze_hotpot_training_schedule", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT_MODULE)


def _record(task_id: str, split: str, *, dataset_key: str = "hotpotqa") -> dict:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": f"question for {task_id}",
        "ground_truth": "answer",
        "split": split,
        "metadata": {"dataset_key": dataset_key},
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def aligned_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write(
        train,
        [
            _record("hotpotqa:train-a", "train"),
            _record("triviaqa:ignored", "train", dataset_key="triviaqa"),
            _record("hotpotqa:train-b", "train"),
            _record("hotpotqa:train-c", "train"),
        ],
    )
    _write(validation, [_record("hotpotqa:heldout-v", "validation")])
    _write(test, [_record("hotpotqa:heldout-t", "test")])
    return train, validation, test


def test_freeze_binds_existing_train_positions_and_rollout_ordinals(
    aligned_paths: tuple[Path, Path, Path],
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_hotpot_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions=(2, 0),
        rollouts_per_task=3,
    )

    assert schedule.source_task_count == 3
    assert [step.task_position for step in schedule.steps] == [2, 0]
    assert [step.task_id for step in schedule.steps] == [
        "hotpotqa:train-c",
        "hotpotqa:train-a",
    ]
    assert [step.step_ordinal for step in schedule.steps] == [1, 2]
    assert all(step.rollout_ordinals == (0, 1, 2) for step in schedule.steps)
    assert schedule.rollout_count == 6
    assert [record.task_id for record in schedule.resolve(
        train_path=train,
        validation_path=validation,
        test_path=test,
    )] == ["hotpotqa:train-c", "hotpotqa:train-a"]
    assert FrozenHotpotTrainingSchedule.from_value(schedule.to_value()) == schedule


def test_freeze_rejects_validation_or_test_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write(train, [_record("hotpotqa:overlap", "train")])
    _write(validation, [_record("hotpotqa:overlap", "validation")])
    _write(test, [])

    with pytest.raises(ValueError, match="overlap validation/test"):
        freeze_hotpot_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions=(0,),
            rollouts_per_task=2,
        )


def test_schedule_rejects_repeated_or_out_of_range_positions(
    aligned_paths: tuple[Path, Path, Path],
) -> None:
    train, validation, test = aligned_paths
    with pytest.raises(ValueError, match="positions must be unique"):
        freeze_hotpot_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions=(0, 0),
            rollouts_per_task=2,
        )
    with pytest.raises(ValueError, match="outside the HotpotQA train split"):
        freeze_hotpot_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions=(3,),
            rollouts_per_task=2,
        )


def test_write_once_and_exact_resumable_cursor(
    aligned_paths: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_hotpot_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions=(0, 1),
        rollouts_per_task=2,
    )
    schedule_path = tmp_path / "schedule.json"
    cursor_path = tmp_path / "cursor-0.json"
    schedule.write_once(schedule_path)
    with pytest.raises(FileExistsError):
        schedule.write_once(schedule_path)
    restored_schedule = FrozenHotpotTrainingSchedule.read(schedule_path)

    progress = HotpotTrainingProgress.fresh(restored_schedule)
    assert progress.current_step.task_id == "hotpotqa:train-a"
    with pytest.raises(ValueError, match="next exact step"):
        progress.commit_step(step_ordinal=2)
    first_state = progress.commit_step(step_ordinal=1)
    first_state.write_once(cursor_path)
    with pytest.raises(FileExistsError):
        first_state.write_once(cursor_path)

    resumed = HotpotTrainingProgress.from_state(
        restored_schedule,
        HotpotTrainingCursorState.read(cursor_path),
    )
    assert resumed.current_step.task_id == "hotpotqa:train-b"
    skipped = HotpotTrainingCursorState(
        curriculum_id=restored_schedule.content_hash,
        cursor=2,
    )
    fresh = HotpotTrainingProgress.fresh(restored_schedule)
    with pytest.raises(ValueError, match="advance exactly one step"):
        fresh.commit_step_state(skipped)
    resumed.commit_step(step_ordinal=2)
    with pytest.raises(RuntimeError, match="exhausted"):
        _ = resumed.current_step


def test_freeze_script_writes_schedule_and_initial_cursor_once(
    aligned_paths: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    train, validation, test = aligned_paths
    schedule_path = tmp_path / "formal" / "schedule.json"
    cursor_path = tmp_path / "formal" / "cursor-step-0.json"
    summary = _SCRIPT_MODULE.freeze_schedule_artifacts(
        train_path=train,
        validation_path=validation,
        test_path=test,
        schedule_path=schedule_path,
        cursor_path=cursor_path,
        step_count=2,
        rollouts_per_task=2,
        task_positions=None,
    )

    assert summary["training_started"] is False
    assert summary["step_count"] == 2
    assert FrozenHotpotTrainingSchedule.read(schedule_path).rollout_count == 4
    cursor = HotpotTrainingCursorState.read(cursor_path)
    assert cursor.cursor == 0
    with pytest.raises(FileExistsError):
        _SCRIPT_MODULE.freeze_schedule_artifacts(
            train_path=train,
            validation_path=validation,
            test_path=test,
            schedule_path=schedule_path,
            cursor_path=cursor_path,
            step_count=2,
            rollouts_per_task=2,
            task_positions=None,
        )
