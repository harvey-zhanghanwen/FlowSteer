"""Frozen HotpotQA+TriviaQA optimizer-step schedule and exact cursor.

This is the two-dataset counterpart of :mod:`hotpot_training_schedule`.  It
keeps the same dependency-light adaptation of SkillFlow's
``PrivateFrozenTaskSequence``, ``OrderedTaskCursorState``, and
``AttemptRunProgress`` while binding each optimizer step to one existing
HotpotQA train task and one existing TriviaQA train task.  Rollout collection,
evaluation, and model updates remain outside this module.
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


JOINT_QA_DATASET_KEYS = ("hotpotqa", "triviaqa")
JOINT_QA_TRAINING_SCHEDULE_ALGORITHM = "flowsteer-joint-qa-ordered-training@1"
JOINT_QA_TRAINING_SCHEDULE_FORMAT = "flowsteer-joint-qa-frozen-training-schedule@1"
JOINT_QA_TRAINING_SOURCE_FORMAT = "flowsteer-joint-qa-training-source@1"
JOINT_QA_TRAINING_TASK_FORMAT = "flowsteer-joint-qa-training-task@1"
JOINT_QA_TRAINING_STEP_FORMAT = "flowsteer-joint-qa-training-step@1"
JOINT_QA_TRAINING_CURSOR_FORMAT = "flowsteer-joint-qa-training-cursor@1"


def _dataset_records(
    path: Path,
    *,
    expected_split: str,
    dataset_key: str,
) -> tuple[TaskRecord, ...]:
    records = tuple(
        record
        for record in iter_task_records(path, expected_split=expected_split)
        if record.metadata.get("dataset_key") == dataset_key
    )
    task_ids = tuple(record.task_id for record in records)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate {dataset_key} task ID in {expected_split} split")
    if any(not task_id.startswith(f"{dataset_key}:") for task_id in task_ids):
        raise ValueError(f"{dataset_key} split contains an incompatible task ID")
    return records


def _source_and_heldout(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    *,
    dataset_key: str,
) -> tuple[tuple[TaskRecord, ...], frozenset[str]]:
    train = _dataset_records(
        train_path,
        expected_split="train",
        dataset_key=dataset_key,
    )
    if not train:
        raise ValueError(f"{dataset_key} train split is empty")
    validation = _dataset_records(
        validation_path,
        expected_split="validation",
        dataset_key=dataset_key,
    )
    test = _dataset_records(
        test_path,
        expected_split="test",
        dataset_key=dataset_key,
    )
    heldout_ids = frozenset(record.task_id for record in (*validation, *test))
    overlap = sorted(
        record.task_id for record in train if record.task_id in heldout_ids
    )
    if overlap:
        raise ValueError(f"{dataset_key} train IDs overlap validation/test IDs")
    return train, heldout_ids


@dataclass(frozen=True, slots=True)
class JointQATrainingSource:
    """Frozen identity of one aligned train source."""

    dataset_key: str
    ordered_train_task_ids_hash: str
    source_task_count: int
    format: str = JOINT_QA_TRAINING_SOURCE_FORMAT

    def __post_init__(self) -> None:
        if self.dataset_key not in JOINT_QA_DATASET_KEYS:
            raise ValueError("unsupported joint-QA dataset key")
        if type(
            self.ordered_train_task_ids_hash
        ) is not str or not self.ordered_train_task_ids_hash.startswith("sha256:"):
            raise ValueError("ordered_train_task_ids_hash must be a content hash")
        if type(self.source_task_count) is not int or self.source_task_count < 1:
            raise ValueError("source_task_count must be positive")
        if self.format != JOINT_QA_TRAINING_SOURCE_FORMAT:
            raise ValueError("unsupported joint-QA training-source format")

    def to_value(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "format": self.format,
            "ordered_train_task_ids_hash": self.ordered_train_task_ids_hash,
            "source_task_count": self.source_task_count,
        }

    @classmethod
    def from_value(cls, value: object) -> "JointQATrainingSource":
        fields = {
            "dataset_key",
            "format",
            "ordered_train_task_ids_hash",
            "source_task_count",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("JointQATrainingSource has incompatible fields")
        return cls(
            dataset_key=value["dataset_key"],
            ordered_train_task_ids_hash=value["ordered_train_task_ids_hash"],
            source_task_count=value["source_task_count"],
            format=value["format"],
        )


@dataclass(frozen=True, slots=True)
class JointQATrainingTask:
    """One dataset-qualified train position inside an optimizer step."""

    dataset_key: str
    task_position: int
    task_id: str
    format: str = JOINT_QA_TRAINING_TASK_FORMAT

    def __post_init__(self) -> None:
        if self.dataset_key not in JOINT_QA_DATASET_KEYS:
            raise ValueError("unsupported joint-QA dataset key")
        if type(self.task_position) is not int or self.task_position < 0:
            raise ValueError("task_position must be non-negative")
        if type(self.task_id) is not str or not self.task_id.startswith(
            f"{self.dataset_key}:"
        ):
            raise ValueError("task_id must match its joint-QA dataset key")
        if self.format != JOINT_QA_TRAINING_TASK_FORMAT:
            raise ValueError("unsupported joint-QA training-task format")

    def to_value(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "format": self.format,
            "task_id": self.task_id,
            "task_position": self.task_position,
        }

    @classmethod
    def from_value(cls, value: object) -> "JointQATrainingTask":
        fields = {"dataset_key", "format", "task_id", "task_position"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("JointQATrainingTask has incompatible fields")
        return cls(
            dataset_key=value["dataset_key"],
            task_position=value["task_position"],
            task_id=value["task_id"],
            format=value["format"],
        )


@dataclass(frozen=True, slots=True)
class JointQATrainingStep:
    """One optimizer step with one frozen task from each QA dataset."""

    step_ordinal: int
    tasks: tuple[JointQATrainingTask, ...]
    rollout_ordinals: tuple[int, ...]
    format: str = JOINT_QA_TRAINING_STEP_FORMAT

    def __post_init__(self) -> None:
        if type(self.step_ordinal) is not int or self.step_ordinal < 1:
            raise ValueError("step_ordinal must be positive")
        if not isinstance(self.tasks, tuple):
            raise TypeError("tasks must be a tuple")
        if any(not isinstance(task, JointQATrainingTask) for task in self.tasks):
            raise TypeError("tasks must contain JointQATrainingTask values")
        if tuple(task.dataset_key for task in self.tasks) != JOINT_QA_DATASET_KEYS:
            raise ValueError(
                "joint-QA step must contain one HotpotQA task then one TriviaQA task"
            )
        if not isinstance(self.rollout_ordinals, tuple) or not self.rollout_ordinals:
            raise ValueError("rollout_ordinals must be a non-empty tuple")
        if any(type(value) is not int or value < 0 for value in self.rollout_ordinals):
            raise ValueError("rollout ordinals must be non-negative integers")
        expected = tuple(range(len(self.rollout_ordinals)))
        if self.rollout_ordinals != expected:
            raise ValueError("rollout ordinals must be contiguous from zero")
        if self.format != JOINT_QA_TRAINING_STEP_FORMAT:
            raise ValueError("unsupported joint-QA training-step format")

    def to_value(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "rollout_ordinals": list(self.rollout_ordinals),
            "step_ordinal": self.step_ordinal,
            "tasks": [task.to_value() for task in self.tasks],
        }

    @classmethod
    def from_value(cls, value: object) -> "JointQATrainingStep":
        fields = {"format", "rollout_ordinals", "step_ordinal", "tasks"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("JointQATrainingStep has incompatible fields")
        raw_tasks = value["tasks"]
        raw_ordinals = value["rollout_ordinals"]
        if not isinstance(raw_tasks, list):
            raise TypeError("tasks must be an array")
        if not isinstance(raw_ordinals, list):
            raise TypeError("rollout_ordinals must be an array")
        return cls(
            step_ordinal=value["step_ordinal"],
            tasks=tuple(JointQATrainingTask.from_value(item) for item in raw_tasks),
            rollout_ordinals=tuple(raw_ordinals),
            format=value["format"],
        )


@dataclass(frozen=True, slots=True)
class FrozenJointQATrainingSchedule:
    """Write-once paired positions over aligned HotpotQA and TriviaQA train order."""

    sources: tuple[JointQATrainingSource, ...]
    steps: tuple[JointQATrainingStep, ...]
    source_split: str = "train"
    schedule_algorithm: str = JOINT_QA_TRAINING_SCHEDULE_ALGORITHM
    format: str = JOINT_QA_TRAINING_SCHEDULE_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple):
            raise TypeError("sources must be a tuple")
        if any(
            not isinstance(source, JointQATrainingSource) for source in self.sources
        ):
            raise TypeError("sources must contain JointQATrainingSource values")
        if (
            tuple(source.dataset_key for source in self.sources)
            != JOINT_QA_DATASET_KEYS
        ):
            raise ValueError(
                "joint-QA schedule must freeze HotpotQA then TriviaQA sources"
            )
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("joint-QA training schedule requires at least one step")
        if any(not isinstance(step, JointQATrainingStep) for step in self.steps):
            raise TypeError("steps must contain JointQATrainingStep values")
        expected_ordinals = tuple(range(1, len(self.steps) + 1))
        if tuple(step.step_ordinal for step in self.steps) != expected_ordinals:
            raise ValueError("training step ordinals must be contiguous from one")
        for dataset_index, source in enumerate(self.sources):
            tasks = tuple(step.tasks[dataset_index] for step in self.steps)
            positions = tuple(task.task_position for task in tasks)
            task_ids = tuple(task.task_id for task in tasks)
            if len(set(positions)) != len(positions) or len(set(task_ids)) != len(
                task_ids
            ):
                raise ValueError(
                    f"one frozen joint-QA schedule cannot repeat a {source.dataset_key} train task"
                )
            if any(position >= source.source_task_count for position in positions):
                raise ValueError(
                    f"{source.dataset_key} task position is outside its frozen source split"
                )
        if self.source_split != "train":
            raise ValueError("joint-QA schedule must remain train-only")
        if self.schedule_algorithm != JOINT_QA_TRAINING_SCHEDULE_ALGORITHM:
            raise ValueError("unsupported joint-QA training schedule algorithm")
        if self.format != JOINT_QA_TRAINING_SCHEDULE_FORMAT:
            raise ValueError("unsupported joint-QA training schedule format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @property
    def rollout_count(self) -> int:
        return sum(len(step.tasks) * len(step.rollout_ordinals) for step in self.steps)

    def to_value(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schedule_algorithm": self.schedule_algorithm,
            "source_split": self.source_split,
            "sources": [source.to_value() for source in self.sources],
            "steps": [step.to_value() for step in self.steps],
        }

    @classmethod
    def from_value(cls, value: object) -> "FrozenJointQATrainingSchedule":
        fields = {
            "format",
            "schedule_algorithm",
            "source_split",
            "sources",
            "steps",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("FrozenJointQATrainingSchedule has incompatible fields")
        raw_sources = value["sources"]
        raw_steps = value["steps"]
        if not isinstance(raw_sources, list):
            raise TypeError("sources must be an array")
        if not isinstance(raw_steps, list):
            raise TypeError("steps must be an array")
        return cls(
            sources=tuple(
                JointQATrainingSource.from_value(item) for item in raw_sources
            ),
            steps=tuple(JointQATrainingStep.from_value(item) for item in raw_steps),
            source_split=value["source_split"],
            schedule_algorithm=value["schedule_algorithm"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> "FrozenJointQATrainingSchedule":
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        _write_once(path, self.to_value())

    def resolve(
        self,
        *,
        train_path: Path,
        validation_path: Path,
        test_path: Path,
    ) -> tuple[tuple[TaskRecord, ...], ...]:
        """Resolve each paired step without changing either aligned split order."""

        records_by_dataset: dict[str, tuple[TaskRecord, ...]] = {}
        heldout_by_dataset: dict[str, frozenset[str]] = {}
        for source in self.sources:
            train, heldout_ids = _source_and_heldout(
                train_path,
                validation_path,
                test_path,
                dataset_key=source.dataset_key,
            )
            train_ids = tuple(record.task_id for record in train)
            if len(train_ids) != source.source_task_count:
                raise ValueError(
                    f"{source.dataset_key} train task count differs from frozen schedule"
                )
            if stable_hash(list(train_ids)) != source.ordered_train_task_ids_hash:
                raise ValueError(
                    f"{source.dataset_key} train order differs from frozen schedule"
                )
            records_by_dataset[source.dataset_key] = train
            heldout_by_dataset[source.dataset_key] = heldout_ids

        resolved_steps: list[tuple[TaskRecord, ...]] = []
        for step in self.steps:
            resolved_tasks: list[TaskRecord] = []
            for task in step.tasks:
                record = records_by_dataset[task.dataset_key][task.task_position]
                if record.task_id != task.task_id:
                    raise ValueError("frozen task position resolves to another task ID")
                if record.task_id in heldout_by_dataset[task.dataset_key]:
                    raise ValueError(
                        "frozen joint-QA schedule references validation/test task IDs"
                    )
                resolved_tasks.append(record)
            resolved_steps.append(tuple(resolved_tasks))
        return tuple(resolved_steps)


def freeze_joint_qa_training_schedule(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    task_positions_by_dataset: Mapping[str, Sequence[int]],
    rollouts_per_task: int,
) -> FrozenJointQATrainingSchedule:
    """Freeze paired train positions before result-dependent execution."""

    if type(rollouts_per_task) is not int or rollouts_per_task < 1:
        raise ValueError("rollouts_per_task must be positive")
    if not isinstance(task_positions_by_dataset, Mapping) or set(
        task_positions_by_dataset
    ) != set(JOINT_QA_DATASET_KEYS):
        raise ValueError(
            "task_positions_by_dataset must contain exactly hotpotqa and triviaqa"
        )

    positions_by_dataset: dict[str, tuple[int, ...]] = {}
    source_records: dict[str, tuple[TaskRecord, ...]] = {}
    sources: list[JointQATrainingSource] = []
    for dataset_key in JOINT_QA_DATASET_KEYS:
        positions = tuple(task_positions_by_dataset[dataset_key])
        if not positions:
            raise ValueError(f"{dataset_key} task positions must not be empty")
        if any(type(position) is not int or position < 0 for position in positions):
            raise ValueError(
                f"{dataset_key} task positions must be non-negative integers"
            )
        if len(set(positions)) != len(positions):
            raise ValueError(f"{dataset_key} task positions must be unique")
        train, heldout_ids = _source_and_heldout(
            train_path,
            validation_path,
            test_path,
            dataset_key=dataset_key,
        )
        if any(position >= len(train) for position in positions):
            raise ValueError(f"task position is outside the {dataset_key} train split")
        selected = tuple(train[position] for position in positions)
        if any(record.task_id in heldout_ids for record in selected):
            raise ValueError(
                "joint-QA training schedule cannot contain validation/test task IDs"
            )
        positions_by_dataset[dataset_key] = positions
        source_records[dataset_key] = train
        sources.append(
            JointQATrainingSource(
                dataset_key=dataset_key,
                ordered_train_task_ids_hash=stable_hash(
                    [record.task_id for record in train]
                ),
                source_task_count=len(train),
            )
        )

    step_counts = {
        len(positions_by_dataset[dataset_key]) for dataset_key in JOINT_QA_DATASET_KEYS
    }
    if len(step_counts) != 1:
        raise ValueError(
            "HotpotQA and TriviaQA must declare the same optimizer-step count"
        )

    rollout_ordinals = tuple(range(rollouts_per_task))
    step_count = step_counts.pop()
    steps = tuple(
        JointQATrainingStep(
            step_ordinal=step_ordinal + 1,
            tasks=tuple(
                JointQATrainingTask(
                    dataset_key=dataset_key,
                    task_position=positions_by_dataset[dataset_key][step_ordinal],
                    task_id=source_records[dataset_key][
                        positions_by_dataset[dataset_key][step_ordinal]
                    ].task_id,
                )
                for dataset_key in JOINT_QA_DATASET_KEYS
            ),
            rollout_ordinals=rollout_ordinals,
        )
        for step_ordinal in range(step_count)
    )
    return FrozenJointQATrainingSchedule(sources=tuple(sources), steps=steps)


@dataclass(frozen=True, slots=True)
class JointQATrainingCursorState:
    """Exact resumable cursor over committed joint-QA optimizer steps."""

    curriculum_id: str
    cursor: int
    format: str = JOINT_QA_TRAINING_CURSOR_FORMAT

    def __post_init__(self) -> None:
        if type(self.curriculum_id) is not str or not self.curriculum_id.startswith(
            "sha256:"
        ):
            raise ValueError("curriculum_id must be the frozen schedule content hash")
        if type(self.cursor) is not int or self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        if self.format != JOINT_QA_TRAINING_CURSOR_FORMAT:
            raise ValueError("unsupported joint-QA training cursor format")

    @classmethod
    def fresh(
        cls,
        schedule: FrozenJointQATrainingSchedule,
    ) -> "JointQATrainingCursorState":
        return cls(curriculum_id=schedule.content_hash, cursor=0)

    def require_schedule(self, schedule: FrozenJointQATrainingSchedule) -> None:
        if self.curriculum_id != schedule.content_hash:
            raise ValueError("training cursor belongs to another frozen schedule")
        if self.cursor > len(schedule.steps):
            raise ValueError("training cursor is beyond the frozen schedule")

    def after_step(
        self,
        schedule: FrozenJointQATrainingSchedule,
        *,
        step_ordinal: int,
    ) -> "JointQATrainingCursorState":
        self.require_schedule(schedule)
        if self.cursor >= len(schedule.steps):
            raise RuntimeError("joint-QA training schedule is exhausted")
        expected = schedule.steps[self.cursor].step_ordinal
        if step_ordinal != expected:
            raise ValueError("training cursor must commit the next exact step")
        return JointQATrainingCursorState(
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
    def from_value(cls, value: object) -> "JointQATrainingCursorState":
        fields = {"curriculum_id", "cursor", "format"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("JointQATrainingCursorState has incompatible fields")
        return cls(
            curriculum_id=value["curriculum_id"],
            cursor=value["cursor"],
            format=value["format"],
        )

    @classmethod
    def read(cls, path: Path) -> "JointQATrainingCursorState":
        return cls.from_value(json.loads(path.read_text(encoding="utf-8")))

    def write_once(self, path: Path) -> None:
        _write_once(path, self.to_value())


@dataclass(slots=True)
class JointQATrainingProgress:
    """Single mutable owner of one immutable exact cursor, as in SkillFlow."""

    schedule: FrozenJointQATrainingSchedule
    _state: JointQATrainingCursorState

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, FrozenJointQATrainingSchedule):
            raise TypeError("schedule must be FrozenJointQATrainingSchedule")
        if not isinstance(self._state, JointQATrainingCursorState):
            raise TypeError("state must be JointQATrainingCursorState")
        self._state.require_schedule(self.schedule)

    @classmethod
    def fresh(
        cls,
        schedule: FrozenJointQATrainingSchedule,
    ) -> "JointQATrainingProgress":
        return cls(
            schedule=schedule,
            _state=JointQATrainingCursorState.fresh(schedule),
        )

    @classmethod
    def from_state(
        cls,
        schedule: FrozenJointQATrainingSchedule,
        state: JointQATrainingCursorState,
    ) -> "JointQATrainingProgress":
        return cls(schedule=schedule, _state=state)

    @property
    def state(self) -> JointQATrainingCursorState:
        return self._state

    @property
    def current_step(self) -> JointQATrainingStep:
        if self._state.cursor >= len(self.schedule.steps):
            raise RuntimeError("joint-QA training schedule is exhausted")
        return self.schedule.steps[self._state.cursor]

    def preview_step(self, *, step_ordinal: int) -> JointQATrainingCursorState:
        return self._state.after_step(self.schedule, step_ordinal=step_ordinal)

    def commit_step(self, *, step_ordinal: int) -> JointQATrainingCursorState:
        self._state = self.preview_step(step_ordinal=step_ordinal)
        return self._state

    def commit_step_state(self, state: JointQATrainingCursorState) -> None:
        state.require_schedule(self.schedule)
        if state.cursor != self._state.cursor + 1:
            raise ValueError("training cursor must advance exactly one step")
        self._state = state


__all__ = [
    "FrozenJointQATrainingSchedule",
    "JOINT_QA_DATASET_KEYS",
    "JOINT_QA_TRAINING_CURSOR_FORMAT",
    "JOINT_QA_TRAINING_SCHEDULE_ALGORITHM",
    "JOINT_QA_TRAINING_SCHEDULE_FORMAT",
    "JOINT_QA_TRAINING_SOURCE_FORMAT",
    "JOINT_QA_TRAINING_STEP_FORMAT",
    "JOINT_QA_TRAINING_TASK_FORMAT",
    "JointQATrainingCursorState",
    "JointQATrainingProgress",
    "JointQATrainingSource",
    "JointQATrainingStep",
    "JointQATrainingTask",
    "freeze_joint_qa_training_schedule",
]
