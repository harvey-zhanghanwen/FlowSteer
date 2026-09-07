"""Frozen TriviaQA-only optimizer-step schedule and resumable cursor.

This is a dataset-thin adaptation of :mod:`hotpot_training_schedule` and
SkillFlow's ``PrivateFrozenTaskSequence`` / ``frozen_sequence_from_task_ids``,
``OrderedBenchmarkTaskProvider``, ``OrderedTaskCursorState``, and
``AttemptRunProgress``.  It binds optimizer-step positions to the project's
already-aligned 512-record TriviaQA train split, verifies isolation from the
128-record held-out validation split through ``base_task_id``, and freezes the
four rollout ordinals required by one same-question GRPO group.

The module only handles schedule identity and progress.  It does not collect
rollouts, evaluate tasks, or update model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hotpot_training_schedule import _write_once
from .records import TaskRecord
from .scientific_sampling import stable_hash
from .task_dataset import iter_task_records


TRIVIAQA_DATASET_KEY = "triviaqa"
TRIVIAQA_EXPECTED_TRAIN_TASK_COUNT = 512
TRIVIAQA_EXPECTED_VALIDATION_TASK_COUNT = 128
TRIVIAQA_ROLLOUTS_PER_TASK = 4
TRIVIAQA_TRAINING_SCHEDULE_ALGORITHM = "flowsteer-triviaqa-ordered-training@1"
TRIVIAQA_TRAINING_SCHEDULE_FORMAT = (
    "flowsteer-triviaqa-frozen-training-schedule@1"
)
TRIVIAQA_TRAINING_STEP_FORMAT = "flowsteer-triviaqa-training-step@1"
TRIVIAQA_TRAINING_CURSOR_FORMAT = "flowsteer-triviaqa-training-cursor@1"


def _base_task_id(record: TaskRecord) -> str:
    sampling = record.metadata.get("sampling")
    if isinstance(sampling, Mapping):
        value = sampling.get("base_task_id")
        if isinstance(value, str) and value:
            if not value.startswith("triviaqa:"):
                raise ValueError("TriviaQA base_task_id has an incompatible prefix")
            return value
    return record.task_id


def _triviaqa_records(
    path: Path,
    *,
    expected_split: str,
) -> tuple[TaskRecord, ...]:
    records = tuple(
        record
        for record in iter_task_records(path, expected_split=expected_split)
        if record.metadata.get("dataset_key") == TRIVIAQA_DATASET_KEY
    )
    task_ids = tuple(record.task_id for record in records)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate TriviaQA task ID in {expected_split} split")
    if any(not task_id.startswith("triviaqa:") for task_id in task_ids):
        raise ValueError(f"TriviaQA {expected_split} split has an incompatible task ID")
    return records


def _source_and_heldout(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
) -> tuple[tuple[TaskRecord, ...], frozenset[str]]:
    train = _triviaqa_records(train_path, expected_split="train")
    validation = _triviaqa_records(validation_path, expected_split="validation")
    test = _triviaqa_records(test_path, expected_split="test")
    if len(train) != TRIVIAQA_EXPECTED_TRAIN_TASK_COUNT:
        raise ValueError(
            "TriviaQA train split must contain exactly "
            f"{TRIVIAQA_EXPECTED_TRAIN_TASK_COUNT} records"
        )
    if len(validation) != TRIVIAQA_EXPECTED_VALIDATION_TASK_COUNT:
        raise ValueError(
            "TriviaQA validation split must contain exactly "
            f"{TRIVIAQA_EXPECTED_VALIDATION_TASK_COUNT} held-out records"
        )

    heldout_ids = frozenset(
        identity
        for record in (*validation, *test)
        for identity in (record.task_id, _base_task_id(record))
    )
    overlap = sorted(
        record.task_id
        for record in train
        if record.task_id in heldout_ids or _base_task_id(record) in heldout_ids
    )
    if overlap:
        raise ValueError(
            "TriviaQA train IDs overlap held-out IDs directly or through "
            "base_task_id"
        )
    return train, heldout_ids


@dataclass(frozen=True, slots=True)
class TriviaQATrainingStep:
    """One predeclared optimizer step over one task and four rollouts."""

    step_ordinal: int
    task_position: int
    task_id: str
    rollout_ordinals: tuple[int, ...]
    format: str = TRIVIAQA_TRAINING_STEP_FORMAT

    def __post_init__(self) -> None:
        if type(self.step_ordinal) is not int or self.step_ordinal < 1:
            raise ValueError("step_ordinal must be positive")
        if type(self.task_position) is not int or self.task_position < 0:
            raise ValueError("task_position must be non-negative")
        if type(self.task_id) is not str or not self.task_id.startswith("triviaqa:"):
            raise ValueError("task_id must identify a TriviaQA task")
        if not isinstance(self.rollout_ordinals, tuple):
            raise TypeError("rollout_ordinals must be a tuple")
        expected = tuple(range(TRIVIAQA_ROLLOUTS_PER_TASK))
        if self.rollout_ordinals != expected:
            raise ValueError("TriviaQA training requires rollout ordinals (0, 1, 2, 3)")
        if self.format != TRIVIAQA_TRAINING_STEP_FORMAT:
            raise ValueError("unsupported TriviaQA training-step format")

    def to_value(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "rollout_ordinals": list(self.rollout_ordinals),
            "step_ordinal": self.step_ordinal,
            "task_id": self.task_id,
            "task_position": self.task_position,
        }

    @classmethod
    def from_value(cls, value: object) -> "TriviaQATrainingStep":
        fields = {
            "format",
            "rollout_ordinals",
            "step_ordinal",
            "task_id",
            "task_position",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("TriviaQATrainingStep has incompatible fields")
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
class FrozenTriviaQATrainingSchedule:
    """Write-once positions over the aligned TriviaQA train order."""

    ordered_train_task_ids_hash: str
    source_task_count: int
    steps: tuple[TriviaQATrainingStep, ...]
    dataset_key: str = TRIVIAQA_DATASET_KEY
    source_split: str = "train"
    schedule_algorithm: str = TRIVIAQA_TRAINING_SCHEDULE_ALGORITHM
    format: str = TRIVIAQA_TRAINING_SCHEDULE_FORMAT

    def __post_init__(self) -> None:
        if type(
            self.ordered_train_task_ids_hash
        ) is not str or not self.ordered_train_task_ids_hash.startswith("sha256:"):
            raise ValueError("ordered_train_task_ids_hash must be a content hash")
        if self.source_task_count != TRIVIAQA_EXPECTED_TRAIN_TASK_COUNT:
            raise ValueError(
                "source_task_count must match the frozen 512-record TriviaQA train split"
            )
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("training schedule requires at least one step")
        if any(not isinstance(step, TriviaQATrainingStep) for step in self.steps):
            raise TypeError("steps must contain TriviaQATrainingStep values")
        expected_ordinals = tuple(range(1, len(self.steps) + 1))
        if tuple(step.step_ordinal for step in self.steps) != expected_ordinals:
            raise ValueError("training step ordinals must be contiguous from one")
        positions = tuple(step.task_position for step in self.steps)
        task_ids = tuple(step.task_id for step in self.steps)
        if len(set(positions)) != len(positions) or len(set(task_ids)) != len(task_ids):
            raise ValueError("one frozen TriviaQA schedule cannot repeat a train task")
        if any(position >= self.source_task_count for position in positions):
            raise ValueError("training task position is outside the frozen source split")
        if self.dataset_key != TRIVIAQA_DATASET_KEY or self.source_split != "train":
            raise ValueError("schedule must remain TriviaQA train-only")
        if self.schedule_algorithm != TRIVIAQA_TRAINING_SCHEDULE_ALGORITHM:
            raise ValueError("unsupported TriviaQA training schedule algorithm")
        if self.format != TRIVIAQA_TRAINING_SCHEDULE_FORMAT:
            raise ValueError("unsupported TriviaQA training schedule format")

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
    def from_value(cls, value: object) -> "FrozenTriviaQATrainingSchedule":
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
            raise ValueError("FrozenTriviaQATrainingSchedule has incompatible fields")
        raw_steps = value["steps"]
        if not isinstance(raw_steps, list):
            raise TypeError("steps must be an array")
        return cls(
            ordered_train_task_ids_hash=value["ordered_train_task_ids_hash"],
            source_task_count=value["source_task_count"],
            steps=tuple(TriviaQATrainingStep.from_value(item) for item in raw_steps),
            dataset_key=value["dataset_key"],
            source_split=value["source_split"],
            schedule_algorithm=value["schedule_algorithm"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> "FrozenTriviaQATrainingSchedule":
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
        """Resolve frozen positions without changing the aligned split."""

        train, heldout_ids = _source_and_heldout(
            train_path,
            validation_path,
            test_path,
        )
        train_ids = tuple(record.task_id for record in train)
        if len(train_ids) != self.source_task_count:
            raise ValueError("TriviaQA train task count differs from frozen schedule")
        if stable_hash(list(train_ids)) != self.ordered_train_task_ids_hash:
            raise ValueError("TriviaQA train order differs from frozen schedule")
        resolved = tuple(train[step.task_position] for step in self.steps)
        if any(record.task_id != step.task_id for record, step in zip(resolved, self.steps)):
            raise ValueError("frozen task position resolves to another task ID")
        if any(
            record.task_id in heldout_ids or _base_task_id(record) in heldout_ids
            for record in resolved
        ):
            raise ValueError("frozen schedule references held-out TriviaQA task IDs")
        return resolved


def freeze_triviaqa_training_schedule(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    task_positions: Sequence[int],
    rollouts_per_task: int = TRIVIAQA_ROLLOUTS_PER_TASK,
) -> FrozenTriviaQATrainingSchedule:
    """Freeze selected train positions before result-dependent execution."""

    if rollouts_per_task != TRIVIAQA_ROLLOUTS_PER_TASK:
        raise ValueError("TriviaQA training requires exactly four rollouts per task")
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
        raise ValueError("task position is outside the TriviaQA train split")
    selected = tuple(train[position] for position in positions)
    if any(
        record.task_id in heldout_ids or _base_task_id(record) in heldout_ids
        for record in selected
    ):
        raise ValueError("training schedule cannot contain held-out TriviaQA task IDs")
    rollout_ordinals = tuple(range(TRIVIAQA_ROLLOUTS_PER_TASK))
    steps = tuple(
        TriviaQATrainingStep(
            step_ordinal=index,
            task_position=position,
            task_id=record.task_id,
            rollout_ordinals=rollout_ordinals,
        )
        for index, (position, record) in enumerate(zip(positions, selected), start=1)
    )
    return FrozenTriviaQATrainingSchedule(
        ordered_train_task_ids_hash=stable_hash(
            [record.task_id for record in train]
        ),
        source_task_count=len(train),
        steps=steps,
    )


@dataclass(frozen=True, slots=True)
class TriviaQATrainingCursorState:
    """Exact resumable cursor over committed TriviaQA schedule steps."""

    curriculum_id: str
    cursor: int
    format: str = TRIVIAQA_TRAINING_CURSOR_FORMAT

    def __post_init__(self) -> None:
        if type(self.curriculum_id) is not str or not self.curriculum_id.startswith(
            "sha256:"
        ):
            raise ValueError("curriculum_id must be the frozen schedule content hash")
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        if self.format != TRIVIAQA_TRAINING_CURSOR_FORMAT:
            raise ValueError("unsupported TriviaQA training cursor format")

    @classmethod
    def fresh(
        cls,
        schedule: FrozenTriviaQATrainingSchedule,
    ) -> "TriviaQATrainingCursorState":
        return cls(curriculum_id=schedule.content_hash, cursor=0)

    def require_schedule(self, schedule: FrozenTriviaQATrainingSchedule) -> None:
        if self.curriculum_id != schedule.content_hash:
            raise ValueError("training cursor belongs to another frozen schedule")
        if self.cursor > len(schedule.steps):
            raise ValueError("training cursor is beyond the frozen schedule")

    def after_step(
        self,
        schedule: FrozenTriviaQATrainingSchedule,
        *,
        step_ordinal: int,
    ) -> "TriviaQATrainingCursorState":
        self.require_schedule(schedule)
        if self.cursor >= len(schedule.steps):
            raise RuntimeError("TriviaQA training schedule is exhausted")
        expected = schedule.steps[self.cursor].step_ordinal
        if step_ordinal != expected:
            raise ValueError("training cursor must commit the next exact step")
        return TriviaQATrainingCursorState(
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
    def from_value(cls, value: object) -> "TriviaQATrainingCursorState":
        fields = {"curriculum_id", "cursor", "format"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("TriviaQATrainingCursorState has incompatible fields")
        return cls(
            curriculum_id=value["curriculum_id"],
            cursor=value["cursor"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> "TriviaQATrainingCursorState":
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        _write_once(path, self.to_value())


@dataclass(slots=True)
class TriviaQATrainingProgress:
    """Single mutable owner of an immutable exact cursor, as in SkillFlow."""

    schedule: FrozenTriviaQATrainingSchedule
    _state: TriviaQATrainingCursorState

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FrozenTriviaQATrainingSchedule):
            raise TypeError("schedule must be FrozenTriviaQATrainingSchedule")
        if not isinstance(self._state, TriviaQATrainingCursorState):
            raise TypeError("state must be TriviaQATrainingCursorState")
        self._state.require_schedule(self.schedule)

    @classmethod
    def fresh(
        cls,
        schedule: FrozenTriviaQATrainingSchedule,
    ) -> "TriviaQATrainingProgress":
        return cls(
            schedule=schedule,
            _state=TriviaQATrainingCursorState.fresh(schedule),
        )

    @classmethod
    def from_state(
        cls,
        schedule: FrozenTriviaQATrainingSchedule,
        state: TriviaQATrainingCursorState,
    ) -> "TriviaQATrainingProgress":
        return cls(schedule=schedule, _state=state)

    @property
    def state(self) -> TriviaQATrainingCursorState:
        return self._state

    @property
    def current_step(self) -> TriviaQATrainingStep:
        if self._state.cursor >= len(self.schedule.steps):
            raise RuntimeError("TriviaQA training schedule is exhausted")
        return self.schedule.steps[self._state.cursor]

    def preview_step(self, *, step_ordinal: int) -> TriviaQATrainingCursorState:
        return self._state.after_step(self.schedule, step_ordinal=step_ordinal)

    def commit_step(self, *, step_ordinal: int) -> TriviaQATrainingCursorState:
        self._state = self.preview_step(step_ordinal=step_ordinal)
        return self._state

    def commit_step_state(self, state: TriviaQATrainingCursorState) -> None:
        state.require_schedule(self.schedule)
        if state.cursor != self._state.cursor + 1:
            raise ValueError("training cursor must advance exactly one step")
        self._state = state


__all__ = [
    "FrozenTriviaQATrainingSchedule",
    "TRIVIAQA_DATASET_KEY",
    "TRIVIAQA_EXPECTED_TRAIN_TASK_COUNT",
    "TRIVIAQA_EXPECTED_VALIDATION_TASK_COUNT",
    "TRIVIAQA_ROLLOUTS_PER_TASK",
    "TRIVIAQA_TRAINING_CURSOR_FORMAT",
    "TRIVIAQA_TRAINING_SCHEDULE_ALGORITHM",
    "TRIVIAQA_TRAINING_SCHEDULE_FORMAT",
    "TRIVIAQA_TRAINING_STEP_FORMAT",
    "TriviaQATrainingCursorState",
    "TriviaQATrainingProgress",
    "TriviaQATrainingStep",
    "freeze_triviaqa_training_schedule",
]
