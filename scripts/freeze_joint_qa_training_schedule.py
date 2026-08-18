#!/usr/bin/env python3
"""Freeze a HotpotQA+TriviaQA training schedule and its initial cursor.

The inputs are the existing aligned splits.  This command never re-splits
data and never starts a model, rollout, evaluator, or trainer.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.joint_qa_training_schedule import (
    JointQATrainingCursorState,
    freeze_joint_qa_training_schedule,
)


def _task_positions(value: str) -> tuple[int, ...]:
    try:
        positions = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "task positions must be comma-separated integers"
        ) from exc
    if not positions or any(position < 0 for position in positions):
        raise argparse.ArgumentTypeError(
            "task positions must be non-empty and non-negative"
        )
    return positions


def freeze_schedule_artifacts(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    skill_confirmation_path: Path | None = None,
    schedule_path: Path,
    cursor_path: Path,
    step_count: int,
    rollouts_per_task: int,
    hotpotqa_task_positions: tuple[int, ...] | None = None,
    triviaqa_task_positions: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if type(step_count) is not int or step_count < 1:
        raise ValueError("step_count must be positive")
    hotpotqa_positions = (
        tuple(range(step_count))
        if hotpotqa_task_positions is None
        else hotpotqa_task_positions
    )
    triviaqa_positions = (
        tuple(range(step_count))
        if triviaqa_task_positions is None
        else triviaqa_task_positions
    )
    if len(hotpotqa_positions) != step_count:
        raise ValueError("HotpotQA task position count must equal step_count")
    if len(triviaqa_positions) != step_count:
        raise ValueError("TriviaQA task position count must equal step_count")

    schedule = freeze_joint_qa_training_schedule(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        skill_confirmation_path=skill_confirmation_path,
        task_positions_by_dataset={
            "hotpotqa": hotpotqa_positions,
            "triviaqa": triviaqa_positions,
        },
        rollouts_per_task=rollouts_per_task,
    )
    cursor = JointQATrainingCursorState.fresh(schedule)
    if schedule_path == cursor_path:
        raise ValueError("schedule and cursor outputs must be different paths")
    existing = [path for path in (schedule_path, cursor_path) if path.exists()]
    if existing:
        raise FileExistsError(f"write-once output already exists: {existing[0]}")
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_once(schedule_path)
    cursor.write_once(cursor_path)
    return {
        "cursor": str(cursor_path),
        "dataset_keys": [source.dataset_key for source in schedule.sources],
        "rollout_count": schedule.rollout_count,
        "schedule": str(schedule_path),
        "schedule_id": schedule.content_hash,
        "source_task_counts": {
            source.dataset_key: source.source_task_count for source in schedule.sources
        },
        "step_count": len(schedule.steps),
        "training_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default="data/agentgraph_v1/train.jsonl",
        help="existing aligned train JSONL",
    )
    parser.add_argument(
        "--validation",
        default="data/agentgraph_v1/validation.jsonl",
        help="existing aligned validation JSONL",
    )
    parser.add_argument(
        "--test",
        default="data/agentgraph_v1/test.jsonl",
        help="existing aligned test JSONL",
    )
    parser.add_argument(
        "--skill-confirmation",
        help="optional independent validation JSONL excluded from training",
    )
    parser.add_argument(
        "--schedule-output",
        default="artifacts/joint_qa_training/formal_schedule.json",
    )
    parser.add_argument(
        "--cursor-output",
        default="artifacts/joint_qa_training/cursor_step_000000.json",
    )
    parser.add_argument(
        "--step-count",
        type=int,
        required=True,
        help="number of exact optimizer-step slots to freeze",
    )
    parser.add_argument(
        "--rollouts-per-task",
        type=int,
        required=True,
        help="frozen grouped-rollout count for each task in both datasets",
    )
    parser.add_argument(
        "--hotpotqa-task-positions",
        type=_task_positions,
        help="optional comma-separated positions in the existing HotpotQA train order",
    )
    parser.add_argument(
        "--triviaqa-task-positions",
        type=_task_positions,
        help="optional comma-separated positions in the existing TriviaQA train order",
    )
    return parser


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = freeze_schedule_artifacts(
        train_path=_resolve(args.train),
        validation_path=_resolve(args.validation),
        test_path=_resolve(args.test),
        skill_confirmation_path=(
            _resolve(args.skill_confirmation) if args.skill_confirmation else None
        ),
        schedule_path=_resolve(args.schedule_output),
        cursor_path=_resolve(args.cursor_output),
        step_count=args.step_count,
        rollouts_per_task=args.rollouts_per_task,
        hotpotqa_task_positions=args.hotpotqa_task_positions,
        triviaqa_task_positions=args.triviaqa_task_positions,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
