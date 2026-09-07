from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.triviaqa_training_schedule import (
    FrozenTriviaQATrainingSchedule,
    TriviaQATrainingCursorState,
    TriviaQATrainingProgress,
    freeze_triviaqa_training_schedule,
)


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "freeze_triviaqa_training_schedule.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "freeze_triviaqa_training_schedule",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT_MODULE)


def _record(
    task_id: str,
    split: str,
    *,
    dataset_key: str = "triviaqa",
    base_task_id: str | None = None,
) -> dict:
    base_id = task_id if base_task_id is None else base_task_id
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": f"question for {task_id}",
        "ground_truth": "answer",
        "split": split,
        "metadata": {
            "dataset_key": dataset_key,
            "sampling": {
                "base_task_id": base_id,
                "cycled_training_sample": task_id != base_id,
            },
        },
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _aligned_records() -> tuple[list[dict], list[dict], list[dict]]:
    train = [_record(f"triviaqa:train-{index}", "train") for index in range(512)]
    train.insert(7, _record("hotpotqa:ignored", "train", dataset_key="hotpotqa"))
    validation = [
        _record(f"triviaqa:heldout-{index}", "validation") for index in range(128)
    ]
    return train, validation, []


@pytest.fixture
def aligned_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    train_records, validation_records, test_records = _aligned_records()
    _write(train, train_records)
    _write(validation, validation_records)
    _write(test, test_records)
    return train, validation, test


def test_freeze_binds_512_train_128_heldout_and_four_rollouts(
    aligned_paths: tuple[Path, Path, Path],
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_triviaqa_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions=(2, 0),
        rollouts_per_task=4,
    )

    assert schedule.source_task_count == 512
    assert [step.task_position for step in schedule.steps] == [2, 0]
    assert [step.task_id for step in schedule.steps] == [
        "triviaqa:train-2",
        "triviaqa:train-0",
    ]
    assert all(step.rollout_ordinals == (0, 1, 2, 3) for step in schedule.steps)
    assert schedule.rollout_count == 8
    assert [
        record.task_id
        for record in schedule.resolve(
            train_path=train,
            validation_path=validation,
            test_path=test,
        )
    ] == ["triviaqa:train-2", "triviaqa:train-0"]
    assert FrozenTriviaQATrainingSchedule.from_value(schedule.to_value()) == schedule


def test_freeze_rejects_base_task_id_heldout_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    train, validation, test = _aligned_records()
    train[0] = _record(
        "triviaqa:cycled-train-0",
        "train",
        base_task_id="triviaqa:heldout-0",
    )
    _write(train_path, train)
    _write(validation_path, validation)
    _write(test_path, test)

    with pytest.raises(ValueError, match="directly or through base_task_id"):
        freeze_triviaqa_training_schedule(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            task_positions=(0,),
        )


def test_freeze_rejects_wrong_counts_and_group_size(
    aligned_paths: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    train, validation, test = aligned_paths
    with pytest.raises(ValueError, match="exactly four rollouts"):
        freeze_triviaqa_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions=(0,),
            rollouts_per_task=3,
        )

    short_train = tmp_path / "short-train.jsonl"
    records, _, _ = _aligned_records()
    trivia_records = [
        record for record in records if record["metadata"]["dataset_key"] == "triviaqa"
    ]
    _write(short_train, trivia_records[:-1])
    with pytest.raises(ValueError, match="exactly 512 records"):
        freeze_triviaqa_training_schedule(
            train_path=short_train,
            validation_path=validation,
            test_path=test,
            task_positions=(0,),
        )


def test_write_once_and_cursor_cannot_skip(
    aligned_paths: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_triviaqa_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions=(0, 1),
    )
    schedule_path = tmp_path / "schedule.json"
    cursor_path = tmp_path / "cursor-1.json"
    schedule.write_once(schedule_path)
    with pytest.raises(FileExistsError):
        schedule.write_once(schedule_path)

    restored = FrozenTriviaQATrainingSchedule.read(schedule_path)
    progress = TriviaQATrainingProgress.fresh(restored)
    with pytest.raises(ValueError, match="next exact step"):
        progress.commit_step(step_ordinal=2)
    first_state = progress.commit_step(step_ordinal=1)
    first_state.write_once(cursor_path)
    with pytest.raises(FileExistsError):
        first_state.write_once(cursor_path)

    fresh = TriviaQATrainingProgress.fresh(restored)
    skipped = TriviaQATrainingCursorState(
        curriculum_id=restored.content_hash,
        cursor=2,
    )
    with pytest.raises(ValueError, match="advance exactly one step"):
        fresh.commit_step_state(skipped)


def test_freeze_script_writes_one_task_four_rollouts_once(
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
        step_count=1,
    )

    assert summary["training_started"] is False
    assert summary["source_task_count"] == 512
    assert summary["validation_task_count"] == 128
    assert summary["rollout_count"] == 4
    schedule = FrozenTriviaQATrainingSchedule.read(schedule_path)
    assert len(schedule.steps) == 1
    assert schedule.steps[0].rollout_ordinals == (0, 1, 2, 3)
    assert TriviaQATrainingCursorState.read(cursor_path).cursor == 0
    with pytest.raises(FileExistsError):
        _SCRIPT_MODULE.freeze_schedule_artifacts(
            train_path=train,
            validation_path=validation,
            test_path=test,
            schedule_path=schedule_path,
            cursor_path=cursor_path,
            step_count=1,
        )
