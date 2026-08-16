"""Frozen HotpotQA-only optimizer-step schedule and resumable cursor.

This is a dependency-light adaptation of SkillFlow's
``PrivateFrozenTaskSequence`` / ``frozen_sequence_from_task_ids``,
``OrderedBenchmarkTaskProvider``, ``OrderedTaskCursorState``, and
``AttemptRunProgress``.  The adaptation binds each optimizer step to an
existing position in this project's already-aligned HotpotQA train split and
also freezes the within-task rollout ordinals required by grouped rollout.

The module only handles schedule identity and progress.  It does not collect
rollouts, evaluate tasks, or update model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .persistence.ids import canonical_json
from .records import TaskRecord
from .scientific_sampling import stable_hash
from .task_dataset import iter_task_records


HOTPOT_DATASET_KEY = "hotpotqa"
HOTPOT_TRAINING_SCHEDULE_ALGORITHM = "flowsteer-hotpot-ordered-training@1"
HOTPOT_TRAINING_SCHEDULE_FORMAT = "flowsteer-hotpot-frozen-training-schedule@1"
HOTPOT_TRAINING_STEP_FORMAT = "flowsteer-hotpot-training-step@1"
HOTPOT_TRAINING_CURSOR_FORMAT = "flowsteer-hotpot-training-cursor@1"


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    """Use SkillFlow's exclusive-create artifact boundary."""

    encoded = canonical_json(value).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _hotpot_records(path: Path, *, expected_split: str) -> tuple[TaskRecord, ...]:
    records = tuple(
        record
        for record in iter_task_records(path, expected_split=expected_split)
        if record.metadata.get("dataset_key") == HOTPOT_DATASET_KEY
    )
    task_ids = tuple(record.task_id for record in records)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate HotpotQA task ID in {expected_split} split")
    return records


def _source_and_heldout(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
) -> tuple[tuple[TaskRecord, ...], frozenset[str]]:
    train = _hotpot_records(train_path, expected_split="train")
    if not train:
        raise ValueError("HotpotQA train split is empty")
    validation = _hotpot_records(validation_path, expected_split="validation")
    test = _hotpot_records(test_path, expected_split="test")
    heldout_ids = frozenset(
        record.task_id for record in (*validation, *test)
    )
    overlap = sorted(record.task_id for record in train if record.task_id in heldout_ids)
    if overlap:
        raise ValueError("HotpotQA train IDs overlap validation/test IDs")
    return train, heldout_ids


@dataclass(frozen=True, slots=True)
class HotpotTrainingStep:
    """One predeclared optimizer step over one train task and rollout group."""

    step_ordinal: int
    task_position: int
    task_id: str
    rollout_ordinals: tuple[int, ...]
    format: str = HOTPOT_TRAINING_STEP_FORMAT

    def __post_init__(self) -> None:
        if type(self.step_ordinal) is not int or self.step_ordinal < 1:
            raise ValueError("step_ordinal must be positive")
        if type(self.task_position) is not int or self.task_position < 0:
            raise ValueError("task_position must be non-negative")
        if type(self.task_id) is not str or not self.task_id.startswith("hotpotqa:"):
            raise ValueError("task_id must identify a HotpotQA task")
        if not isinstance(self.rollout_ordinals, tuple) or not self.rollout_ordinals:
            raise ValueError("rollout_ordinals must be a non-empty tuple")
        if any(type(value) is not int or value < 0 for value in self.rollout_ordinals):
            raise ValueError("rollout ordinals must be non-negative integers")
        expected = tuple(range(len(self.rollout_ordinals)))
        if self.rollout_ordinals != expected:
            raise ValueError("rollout ordinals must be contiguous from zero")
        if self.format != HOTPOT_TRAINING_STEP_FORMAT:
            raise ValueError("unsupported HotpotQA training-step format")

    def to_value(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "rollout_ordinals": list(self.rollout_ordinals),
            "step_ordinal": self.step_ordinal,
            "task_id": self.task_id,
            "task_position": self.task_position,
        }

    @classmethod
    def from_value(cls, value: object) -> "HotpotTrainingStep":
        fields = {
            "format",
            "rollout_ordinals",
            "step_ordinal",
            "task_id",
            "task_position",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("HotpotTrainingStep has incompatible fields")
        raw_ordinals = value["rollout_ordinals"]
        if not isinstance(raw_ordinals, list):
            raise TypeError("rollout_ordinals must be an array")
        return cls(
            step_ordinal=value["step_ordinal"],
            task_position=value["task_position"],
            task_id=value["task_id"],
            rollout_ordinals=tuple(raw_ordinals),
            format=value["format"],
        )


@dataclass(frozen=True, slots=True)
class FrozenHotpotTrainingSchedule:
    """Write-once positions over the existing aligned HotpotQA train order."""

    ordered_train_task_ids_hash: str
    source_task_count: int
    steps: tuple[HotpotTrainingStep, ...]
    dataset_key: str = HOTPOT_DATASET_KEY
    source_split: str = "train"
    schedule_algorithm: str = HOTPOT_TRAINING_SCHEDULE_ALGORITHM
    format: str = HOTPOT_TRAINING_SCHEDULE_FORMAT

    def __post_init__(self) -> None:
        if type(self.ordered_train_task_ids_hash) is not str or not self.ordered_train_task_ids_hash.startswith(
            "sha256:"
        ):
            raise ValueError("ordered_train_task_ids_hash must be a content hash")
        if type(self.source_task_count) is not int or self.source_task_count < 1:
            raise ValueError("source_task_count must be positive")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("training schedule requires at least one step")
        if any(not isinstance(step, HotpotTrainingStep) for step in self.steps):
            raise TypeError("steps must contain HotpotTrainingStep values")
        expected_ordinals = tuple(range(1, len(self.steps) + 1))
        if tuple(step.step_ordinal for step in self.steps) != expected_ordinals:
            raise ValueError("training step ordinals must be contiguous from one")
        positions = tuple(step.task_position for step in self.steps)
        task_ids = tuple(step.task_id for step in self.steps)
        if len(set(positions)) != len(positions) or len(set(task_ids)) != len(task_ids):
            raise ValueError("one frozen HotpotQA schedule cannot repeat a train task")
        if any(position >= self.source_task_count for position in positions):
            raise ValueError("training task position is outside the frozen source split")
        if self.dataset_key != HOTPOT_DATASET_KEY or self.source_split != "train":
            raise ValueError("schedule must remain HotpotQA train-only")
        if self.schedule_algorithm != HOTPOT_TRAINING_SCHEDULE_ALGORITHM:
            raise ValueError("unsupported HotpotQA training schedule algorithm")
        if self.format != HOTPOT_TRAINING_SCHEDULE_FORMAT:
            raise ValueError("unsupported HotpotQA training schedule format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @property
    def rollout_count(self) -> int:
        return sum(len(step.rollout_ordinals) for step in self.steps)

    def to_value(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "format": self.format,
            "ordered_train_task_ids_hash": self.ordered_train_task_ids_hash,
            "schedule_algorithm": self.schedule_algorithm,
            "source_split": self.source_split,
            "source_task_count": self.source_task_count,
            "steps": [step.to_value() for step in self.steps],
        }

    @classmethod
    def from_value(cls, value: object) -> "FrozenHotpotTrainingSchedule":
        fields = {
            "dataset_key",
            "format",
            "ordered_train_task_ids_hash",
            "schedule_algorithm",
            "source_split",
            "source_task_count",
            "steps",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("FrozenHotpotTrainingSchedule has incompatible fields")
        raw_steps = value["steps"]
        if not isinstance(raw_steps, list):
            raise TypeError("steps must be an array")
        return cls(
            ordered_train_task_ids_hash=value["ordered_train_task_ids_hash"],
            source_task_count=value["source_task_count"],
            steps=tuple(HotpotTrainingStep.from_value(item) for item in raw_steps),
            dataset_key=value["dataset_key"],
            source_split=value["source_split"],
            schedule_algorithm=value["schedule_algorithm"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> "FrozenHotpotTrainingSchedule":
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        _write_once(path, self.to_value())

    def resolve(
        self,
        *,
        train_path: Path,
        validation_path: Path,
        test_path: Path,
    ) -> tuple[TaskRecord, ...]:
        """Resolve frozen positions without changing the pre-existing split."""

        train, heldout_ids = _source_and_heldout(
            train_path,
            validation_path,
            test_path,
        )
        train_ids = tuple(record.task_id for record in train)
        if len(train_ids) != self.source_task_count:
            raise ValueError("HotpotQA train task count differs from frozen schedule")
        if stable_hash(list(train_ids)) != self.ordered_train_task_ids_hash:
            raise ValueError("HotpotQA train order differs from frozen schedule")
        resolved = tuple(train[step.task_position] for step in self.steps)
        if any(record.task_id != step.task_id for record, step in zip(resolved, self.steps)):
            raise ValueError("frozen task position resolves to another task ID")
        if any(record.task_id in heldout_ids for record in resolved):
            raise ValueError("frozen schedule references validation/test task IDs")
        return resolved


def freeze_hotpot_training_schedule(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    task_positions: Sequence[int],
    rollouts_per_task: int,
) -> FrozenHotpotTrainingSchedule:
    """Freeze selected train positions before result-dependent execution."""

    if type(rollouts_per_task) is not int or rollouts_per_task < 1:
        raise ValueError("rollouts_per_task must be positive")
    positions = tuple(task_positions)
    if not positions:
        raise ValueError("task_positions must not be empty")
    if any(type(position) is not int or position < 0 for position in positions):
        raise ValueError("task positions must be non-negative integers")
    if len(set(positions)) != len(positions):
        raise ValueError("task positions must be unique")

    train, heldout_ids = _source_and_heldout(
        train_path,
        validation_path,
        test_path,
    )
    if any(position >= len(train) for position in positions):
        raise ValueError("task position is outside the HotpotQA train split")
    selected = tuple(train[position] for position in positions)
    if any(record.task_id in heldout_ids for record in selected):
        raise ValueError("training schedule cannot contain validation/test task IDs")
    rollout_ordinals = tuple(range(rollouts_per_task))
    steps = tuple(
        HotpotTrainingStep(
            step_ordinal=index,
            task_position=position,
            task_id=record.task_id,
            rollout_ordinals=rollout_ordinals,
        )
        for index, (position, record) in enumerate(zip(positions, selected), start=1)
    )
    return FrozenHotpotTrainingSchedule(
        ordered_train_task_ids_hash=stable_hash(
            [record.task_id for record in train]
        ),
        source_task_count=len(train),
        steps=steps,
    )


@dataclass(frozen=True, slots=True)
class HotpotTrainingCursorState:
    """Exact resumable cursor over committed schedule steps."""

    curriculum_id: str
    cursor: int
    format: str = HOTPOT_TRAINING_CURSOR_FORMAT

    def __post_init__(self) -> None:
        if type(self.curriculum_id) is not str or not self.curriculum_id.startswith("sha256:"):
            raise ValueError("curriculum_id must be the frozen schedule content hash")
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        if self.format != HOTPOT_TRAINING_CURSOR_FORMAT:
            raise ValueError("unsupported HotpotQA training cursor format")

    @classmethod
    def fresh(cls, schedule: FrozenHotpotTrainingSchedule) -> "HotpotTrainingCursorState":
        return cls(curriculum_id=schedule.content_hash, cursor=0)

    def require_schedule(self, schedule: FrozenHotpotTrainingSchedule) -> None:
        if self.curriculum_id != schedule.content_hash:
            raise ValueError("training cursor belongs to another frozen schedule")
        if self.cursor > len(schedule.steps):
            raise ValueError("training cursor is beyond the frozen schedule")

    def after_step(
        self,
        schedule: FrozenHotpotTrainingSchedule,
        *,
        step_ordinal: int,
    ) -> "HotpotTrainingCursorState":
        self.require_schedule(schedule)
        if self.cursor >= len(schedule.steps):
            raise RuntimeError("HotpotQA training schedule is exhausted")
        expected = schedule.steps[self.cursor].step_ordinal
        if step_ordinal != expected:
            raise ValueError("training cursor must commit the next exact step")
        return HotpotTrainingCursorState(
            curriculum_id=self.curriculum_id,
            cursor=self.cursor + 1,
        )

    def to_value(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "cursor": self.cursor,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> "HotpotTrainingCursorState":
        fields = {"curriculum_id", "cursor", "format"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("HotpotTrainingCursorState has incompatible fields")
        return cls(
            curriculum_id=value["curriculum_id"],
            cursor=value["cursor"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> "HotpotTrainingCursorState":
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        _write_once(path, self.to_value())


@dataclass(slots=True)
class HotpotTrainingProgress:
    """Single mutable owner of an immutable exact cursor, as in SkillFlow."""

    schedule: FrozenHotpotTrainingSchedule
    _state: HotpotTrainingCursorState

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FrozenHotpotTrainingSchedule):
            raise TypeError("schedule must be FrozenHotpotTrainingSchedule")
        if not isinstance(self._state, HotpotTrainingCursorState):
            raise TypeError("state must be HotpotTrainingCursorState")
        self._state.require_schedule(self.schedule)

    @classmethod
    def fresh(cls, schedule: FrozenHotpotTrainingSchedule) -> "HotpotTrainingProgress":
        return cls(schedule=schedule, _state=HotpotTrainingCursorState.fresh(schedule))

    @classmethod
    def from_state(
        cls,
        schedule: FrozenHotpotTrainingSchedule,
        state: HotpotTrainingCursorState,
    ) -> "HotpotTrainingProgress":
        return cls(schedule=schedule, _state=state)

    @property
    def state(self) -> HotpotTrainingCursorState:
        return self._state

    @property
    def current_step(self) -> HotpotTrainingStep:
        if self._state.cursor >= len(self.schedule.steps):
            raise RuntimeError("HotpotQA training schedule is exhausted")
        return self.schedule.steps[self._state.cursor]

    def preview_step(self, *, step_ordinal: int) -> HotpotTrainingCursorState:
        return self._state.after_step(
            self.schedule,
            step_ordinal=step_ordinal,
        )

    def commit_step(self, *, step_ordinal: int) -> HotpotTrainingCursorState:
        self._state = self.preview_step(step_ordinal=step_ordinal)
        return self._state

    def commit_step_state(self, state: HotpotTrainingCursorState) -> None:
        state.require_schedule(self.schedule)
        if state.cursor != self._state.cursor + 1:
            raise ValueError("training cursor must advance exactly one step")
        self._state = state


__all__ = [
    "FrozenHotpotTrainingSchedule",
    "HOTPOT_DATASET_KEY",
    "HOTPOT_TRAINING_CURSOR_FORMAT",
    "HOTPOT_TRAINING_SCHEDULE_ALGORITHM",
    "HOTPOT_TRAINING_SCHEDULE_FORMAT",
    "HOTPOT_TRAINING_STEP_FORMAT",
    "HotpotTrainingCursorState",
    "HotpotTrainingProgress",
    "HotpotTrainingStep",
    "freeze_hotpot_training_schedule",
]
