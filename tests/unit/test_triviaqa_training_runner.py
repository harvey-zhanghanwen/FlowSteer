from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from scripts.train_agentgraph_smoke import run_smoke, validate_smoke_bounds
from src.interactive.config_loader import ConfigurationError
from src.interactive.triviaqa_training_schedule import (
    TriviaQATrainingCursorState,
    freeze_triviaqa_training_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/training_triviaqa_v16_onepass_grpo_1step_prepare.yaml"


def _task(index: int, *, split: str) -> dict[str, object]:
    task_id = f"triviaqa:{split}-{index}"
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": f"Question {index}?",
        "ground_truth": "Answer",
        "split": split,
        "metadata": {
            "dataset_key": "triviaqa",
            "sampling": {
                "base_task_id": task_id,
                "cycled_training_sample": False,
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    train_path = root / "data/train.jsonl"
    validation_path = root / "data/validation.jsonl"
    test_path = root / "data/test.jsonl"
    _write_jsonl(train_path, [_task(index, split="train") for index in range(512)])
    _write_jsonl(
        validation_path,
        [_task(index, split="validation") for index in range(128)],
    )
    _write_jsonl(test_path, [])

    schedule = freeze_triviaqa_training_schedule(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        task_positions=(0,),
    )
    schedule_path = root / "artifacts/schedule.json"
    cursor_path = root / "artifacts/cursor0.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_once(schedule_path)
    TriviaQATrainingCursorState.fresh(schedule).write_once(cursor_path)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data"].update(
        train_path="data/train.jsonl",
        validation_path="data/validation.jsonl",
        test_path="data/test.jsonl",
    )
    config["data"]["triviaqa_micro"].update(
        schedule_path="artifacts/schedule.json",
        cursor_path="artifacts/cursor0.json",
        next_cursor_path="artifacts/cursor1.json",
    )
    config["gpu"].update(
        learner_physical=0,
        rollout_physical=1,
        gradient_replica_physical=2,
        learner_device="cuda:0",
        gradient_replica_device="cuda:2",
        supervisor_gpu_id=1,
        allocation_receipt_path="artifacts/gpu-allocation.json",
    )
    config["director"]["api_base"] = "http://127.0.0.1:18015/v1"
    config["policy_sync"]["api_base"] = "http://127.0.0.1:18015"
    config["gpu"]["supervisor_api_base"] = "http://127.0.0.1:18015/v1"
    for name in (
        "root",
        "selected_tasks_path",
        "trajectories_path",
        "grpo_groups_path",
        "retrieval_receipts_path",
        "manifest_path",
        "sync_receipt_path",
        "post_update_trajectories_path",
        "behavior_policy_preflight_path",
    ):
        config["storage"][name] = f"artifacts/step1/{Path(config['storage'][name]).name}"
    config["experiment"]["output_dir"] = "artifacts/step1/checkpoint"
    config_path = root / "config/trivia-step1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, config_path, root / "artifacts/cursor1.json"


def test_trivia_prepare_only_selects_one_group_without_prefetch_or_cursor_commit(
    tmp_path: Path,
) -> None:
    root, config_path, next_cursor_path = _project(tmp_path)

    manifest = asyncio.run(
        run_smoke(config_path, prepare_only=True, project_root=root)
    )

    assert manifest["status"] == "prepared"
    assert manifest["bounds"]["selected_tasks"] == 1
    assert manifest["bounds"]["expected_initial_rollouts"] == 4
    assert manifest["selection_receipt"]["static_retrieval_prefetch"] is False
    assert manifest["selected_by_source"] == {"triviaqa": 1}
    selected = json.loads(
        (root / "artifacts/step1/selected_tasks.jsonl").read_text(encoding="utf-8")
    )
    assert selected["task_id"] == "triviaqa:train-0"
    assert "retrieval" not in selected["question"].lower()
    assert not next_cursor_path.exists()
    assert not (root / "artifacts/step1/trajectories.jsonl").exists()


def test_trivia_bounds_reject_non_four_trajectory_group(tmp_path: Path) -> None:
    _, config_path, _ = _project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["grpo"]["samples_per_problem"] = 3
    config["grpo"]["expected_rollout_count"] = 3

    with pytest.raises(ConfigurationError, match="samples_per_problem"):
        validate_smoke_bounds(config)
