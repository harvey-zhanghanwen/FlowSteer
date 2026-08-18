from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.joint_qa_training_schedule import (
    FrozenJointQATrainingSchedule,
    JointQATrainingCursorState,
    JointQATrainingProgress,
    freeze_joint_qa_training_schedule,
)


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "freeze_joint_qa_training_schedule.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "freeze_joint_qa_training_schedule", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT_MODULE)


def _record(task_id: str, split: str, *, dataset_key: str) -> dict:
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
            _record("hotpotqa:train-a", "train", dataset_key="hotpotqa"),
            _record("triviaqa:train-a", "train", dataset_key="triviaqa"),
            _record("hotpotqa:train-b", "train", dataset_key="hotpotqa"),
            _record("triviaqa:train-b", "train", dataset_key="triviaqa"),
            _record("hotpotqa:train-c", "train", dataset_key="hotpotqa"),
            _record("triviaqa:train-c", "train", dataset_key="triviaqa"),
        ],
    )
    _write(
        validation,
        [
            _record("hotpotqa:heldout-v", "validation", dataset_key="hotpotqa"),
            _record("triviaqa:heldout-v", "validation", dataset_key="triviaqa"),
        ],
    )
    _write(
        test,
        [
            _record("hotpotqa:heldout-t", "test", dataset_key="hotpotqa"),
            _record("triviaqa:heldout-t", "test", dataset_key="triviaqa"),
        ],
    )
    return train, validation, test


def test_freeze_binds_one_task_per_dataset_for_each_optimizer_step(
    aligned_paths: tuple[Path, Path, Path],
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_joint_qa_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions_by_dataset={
            "hotpotqa": (2, 0),
            "triviaqa": (1, 2),
        },
        rollouts_per_task=3,
    )

    assert [source.dataset_key for source in schedule.sources] == [
        "hotpotqa",
        "triviaqa",
    ]
    assert [source.source_task_count for source in schedule.sources] == [3, 3]
    assert [step.step_ordinal for step in schedule.steps] == [1, 2]
    assert [task.task_id for task in schedule.steps[0].tasks] == [
        "hotpotqa:train-c",
        "triviaqa:train-b",
    ]
    assert [task.task_id for task in schedule.steps[1].tasks] == [
        "hotpotqa:train-a",
        "triviaqa:train-c",
    ]
    assert all(step.rollout_ordinals == (0, 1, 2) for step in schedule.steps)
    assert schedule.rollout_count == 12

    resolved = schedule.resolve(
        train_path=train,
        validation_path=validation,
        test_path=test,
    )
    assert [[record.task_id for record in step] for step in resolved] == [
        ["hotpotqa:train-c", "triviaqa:train-b"],
        ["hotpotqa:train-a", "triviaqa:train-c"],
    ]
    assert FrozenJointQATrainingSchedule.from_value(schedule.to_value()) == schedule


def test_freeze_requires_equal_unique_positions_for_both_datasets(
    aligned_paths: tuple[Path, Path, Path],
) -> None:
    train, validation, test = aligned_paths
    with pytest.raises(ValueError, match="same optimizer-step count"):
        freeze_joint_qa_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions_by_dataset={
                "hotpotqa": (0, 1),
                "triviaqa": (0,),
            },
            rollouts_per_task=2,
        )
    with pytest.raises(ValueError, match="triviaqa task positions must be unique"):
        freeze_joint_qa_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions_by_dataset={
                "hotpotqa": (0, 1),
                "triviaqa": (0, 0),
            },
            rollouts_per_task=2,
        )


def test_freeze_rejects_heldout_overlap_for_either_dataset(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write(
        train,
        [
            _record("hotpotqa:train-a", "train", dataset_key="hotpotqa"),
            _record("triviaqa:overlap", "train", dataset_key="triviaqa"),
        ],
    )
    _write(
        validation,
        [
            _record("triviaqa:overlap", "validation", dataset_key="triviaqa"),
        ],
    )
    _write(test, [])

    with pytest.raises(ValueError, match="triviaqa train IDs overlap"):
        freeze_joint_qa_training_schedule(
            train_path=train,
            validation_path=validation,
            test_path=test,
            task_positions_by_dataset={"hotpotqa": (0,), "triviaqa": (0,)},
            rollouts_per_task=2,
        )


def test_freeze_rejects_skill_confirmation_base_task_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    confirmation = tmp_path / "skill-confirmation.jsonl"
    test = tmp_path / "test.jsonl"
    hotpot_cycle = _record(
        "hotpotqa:heldout-c:cycle-0001",
        "train",
        dataset_key="hotpotqa",
    )
    hotpot_cycle["metadata"]["sampling"] = {
        "base_task_id": "hotpotqa:heldout-c",
        "cycled_training_sample": True,
    }
    _write(
        train,
        [
            hotpot_cycle,
            _record("triviaqa:train-a", "train", dataset_key="triviaqa"),
        ],
    )
    _write(validation, [])
    _write(
        confirmation,
        [
            _record(
                "hotpotqa:heldout-c",
                "validation",
                dataset_key="hotpotqa",
            )
        ],
    )
    _write(test, [])

    with pytest.raises(ValueError, match="base_task_id"):
        freeze_joint_qa_training_schedule(
            train_path=train,
            validation_path=validation,
            skill_confirmation_path=confirmation,
            test_path=test,
            task_positions_by_dataset={"hotpotqa": (0,), "triviaqa": (0,)},
            rollouts_per_task=2,
        )


def test_resolve_rejects_changed_source_order(
    aligned_paths: tuple[Path, Path, Path],
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_joint_qa_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions_by_dataset={"hotpotqa": (0,), "triviaqa": (0,)},
        rollouts_per_task=2,
    )
    records = [
        json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()
    ]
    records[0], records[2] = records[2], records[0]
    _write(train, records)

    with pytest.raises(ValueError, match="hotpotqa train order differs"):
        schedule.resolve(
            train_path=train,
            validation_path=validation,
            test_path=test,
        )


def test_write_once_and_exact_joint_cursor(
    aligned_paths: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    train, validation, test = aligned_paths
    schedule = freeze_joint_qa_training_schedule(
        train_path=train,
        validation_path=validation,
        test_path=test,
        task_positions_by_dataset={
            "hotpotqa": (0, 1),
            "triviaqa": (1, 2),
        },
        rollouts_per_task=2,
    )
    schedule_path = tmp_path / "schedule.json"
    cursor_path = tmp_path / "cursor-step-0.json"
    schedule.write_once(schedule_path)
    with pytest.raises(FileExistsError):
        schedule.write_once(schedule_path)
    restored = FrozenJointQATrainingSchedule.read(schedule_path)

    progress = JointQATrainingProgress.fresh(restored)
    resolved = restored.resolve(
        train_path=train,
        validation_path=validation,
        test_path=test,
    )
    assert [record.task_id for record in resolved[progress.state.cursor]] == [
        "hotpotqa:train-a",
        "triviaqa:train-b",
    ]
    with pytest.raises(ValueError, match="next exact step"):
        progress.commit_step(step_ordinal=2)
    first_state = progress.commit_step(step_ordinal=1)
    first_state.write_once(cursor_path)
    with pytest.raises(FileExistsError):
        first_state.write_once(cursor_path)

    resumed = JointQATrainingProgress.from_state(
        restored,
        JointQATrainingCursorState.read(cursor_path),
    )
    assert resumed.current_step.step_ordinal == 2
    skipped = JointQATrainingCursorState(
        curriculum_id=restored.content_hash,
        cursor=2,
    )
    fresh = JointQATrainingProgress.fresh(restored)
    with pytest.raises(ValueError, match="advance exactly one step"):
        fresh.commit_step_state(skipped)
    resumed.commit_step(step_ordinal=2)
    with pytest.raises(RuntimeError, match="exhausted"):
        _ = resumed.current_step


def test_freeze_script_writes_joint_schedule_and_initial_cursor_once(
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
        hotpotqa_task_positions=(2, 0),
        triviaqa_task_positions=(1, 2),
    )

    assert summary["training_started"] is False
    assert summary["dataset_keys"] == ["hotpotqa", "triviaqa"]
    assert summary["step_count"] == 2
    assert summary["rollout_count"] == 8
    schedule = FrozenJointQATrainingSchedule.read(schedule_path)
    assert [task.task_id for task in schedule.steps[0].tasks] == [
        "hotpotqa:train-c",
        "triviaqa:train-b",
    ]
    cursor = JointQATrainingCursorState.read(cursor_path)
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
            hotpotqa_task_positions=(2, 0),
            triviaqa_task_positions=(1, 2),
        )
