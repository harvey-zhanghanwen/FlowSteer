"""CPU-only coverage of the runner's proven zero-update resampling boundary.

Reuse the existing sequential-runner fixtures.  These fakes test control flow
and receipts only; they provide no evidence of real model training.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.unit.test_hotpotqa_grpo_runner import (
    FakeBackend,
    FakeSummary,
    FakeTracker,
    HotpotTrainingError,
    _MODULE,
    _create_project,
    _trajectory,
    run_hotpotqa_training,
)


class RecordingBackend(FakeBackend):
    def __init__(self, config, events, *, outcome="success"):
        super().__init__(config, events)
        self.outcome = outcome
        self.collections = []

    async def collect(
        self, task, rollout_index, versions, *, expected_task_split="train"
    ):
        assert expected_task_split == task.split
        # Retry IDs deliberately exceed the fixture's old canary threshold.
        # Remove only the namespace for classification, never from the receipt.
        local_index = rollout_index % 100_000_000
        kind = (
            "validation"
            if expected_task_split == "validation"
            else "canary" if local_index >= 1_000_000 else "rollout"
        )
        self.events.append(
            f"collect:{self.training_step}:{kind}:{versions.policy}:{self.active_adapter}"
        )
        self.collections.append(
            (kind, task.task_id, rollout_index, versions.policy, self.active_adapter)
        )
        return _trajectory(task, rollout_index, versions, adapter=self.active_adapter)

    def train(self, trajectories, output_dir):
        if self.outcome == "success":
            return super().train(trajectories, output_dir)
        if self.outcome == "unknown_failure":
            self.events.append(f"train:{self.training_step}")
            raise RuntimeError("unknown learner failure before a summary was saved")
        if self.outcome == "post_update_failure":
            summary = super().train(trajectories, output_dir)
            Path(output_dir, "training_summary.json").write_text(
                json.dumps(summary.to_dict()), encoding="utf-8"
            )
            raise RuntimeError("publication preparation failed after optimizer update")

        self.events.append(f"train:{self.training_step}")
        group_ids = sorted({item.group_id for item in trajectories})
        exclusions = {
            group_id: (
                "zero_information_group"
                if index < 3
                else "behavior_logprob_tolerance_exceeded"
            )
            for index, group_id in enumerate(group_ids)
        }
        if self.outcome == "unknown_exclusion":
            exclusions[group_ids[0]] = "unknown_failure"
        values = {
            "optimizer_updates": 0,
            "input_trajectories": len(trajectories),
            "exact_groups": len(group_ids),
            "excluded_groups": len(group_ids),
            "trained_groups": 0,
            "trained_trajectories": 0,
            "exclusions": exclusions,
            "loss": 0.0,
            "grad_norm": 0.0,
            "trainable_update_l2": 0.0,
            "behavior_policy_version": self.active_policy,
            "updated_policy_version": "",
            "checkpoint_dir": "",
            "optimizer_state_checkpoint": "",
            "training_state_checkpoint": "",
            "checkpoint_recoverable": False,
            "optimizer_state_saved": False,
            "training_state_saved": False,
            "scheduler_state_saved": False,
            "rng_state_saved": False,
            "update_step": self.absolute_step,
            "committed_step": self.absolute_step - 1,
            "continuation_adapter_checkpoint": self.config["director"][
                "behavior_adapter_checkpoint"
            ],
        }
        Path(output_dir).mkdir(parents=True)
        Path(output_dir, "training_summary.json").write_text(
            json.dumps(values), encoding="utf-8"
        )
        return FakeSummary(values)


@pytest.fixture
def runner(tmp_path):
    config_path = _create_project(tmp_path)
    events = []
    backends = []

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    def run(*, resume=False, boundary=1, outcome="success", tracker=None):
        def factory(config, project_root):
            assert project_root == tmp_path
            backend = RecordingBackend(config, events, outcome=outcome)
            backends.append(backend)
            return backend

        return asyncio.run(
            run_hotpotqa_training(
                config_path,
                project_root=tmp_path,
                allow_md_grpo=True,
                resume=resume,
                stop_after_optimizer_steps=boundary,
                backend_factory=factory,
                tracker=tracker or FakeTracker(),
            )
        )

    with (
        patch.object(_MODULE.os, "fsync"),
        patch.object(_MODULE.asyncio, "to_thread", new=inline_to_thread),
        patch.object(_MODULE, "_STOP_REQUESTED", False),
    ):
        yield run, tmp_path / "artifacts/training", events, backends


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_zero_update_archives_and_resamples_without_advancing_policy(runner):
    run, root, events, backends = runner
    first_tracker = FakeTracker()
    first = run(tracker=first_tracker)
    assert first["optimizer_updates_completed"] == 1
    state_before = read_json(root / "run_state.json")
    state_bytes = (root / "run_state.json").read_bytes()
    optimizer_checkpoint = Path(state_before["optimizer_state_checkpoint"])
    checkpoint_bytes = optimizer_checkpoint.read_bytes()

    rejected_tracker = FakeTracker()
    rejected = run(
        resume=True, boundary=2, outcome="zero_update", tracker=rejected_tracker
    )
    assert rejected["status"] == "retryable_zero_update"
    assert rejected["optimizer_updates_completed"] == 1
    assert (root / "run_state.json").read_bytes() == state_bytes
    assert optimizer_checkpoint.read_bytes() == checkpoint_bytes
    assert rejected_tracker.logs == []
    assert rejected_tracker.artifacts == []
    assert rejected_tracker.finished == [0]
    assert "publish:2" not in events
    assert not (root / "steps/step_000002").exists()
    assert not (root / "recovery/current_step.json").exists()

    recovery = read_json(root / "recovery/zero_update_retry_state.json")
    archive = Path(recovery["archived_step_directory"])
    assert recovery["attempt_index"] == 1
    assert recovery["rollout_index_offset"] == 100_000_000
    assert recovery["consumed_by_optimizer"] is False
    assert recovery["behavior_policy_version"] == state_before["behavior_policy_version"]
    assert recovery["behavior_adapter_checkpoint"] == state_before["behavior_adapter_checkpoint"]
    assert read_json(archive / "learner/training_summary.json")["optimizer_updates"] == 0
    assert len((archive / "trajectories.jsonl").read_text().splitlines()) == 28
    assert not (archive / "learner/checkpoint_final").exists()
    assert not (archive / "sync_receipt.json").exists()

    resumed_tracker = FakeTracker()
    resumed = run(resume=True, boundary=2, tracker=resumed_tracker)
    assert resumed["status"] == "paused_at_requested_boundary"
    assert resumed["optimizer_updates_completed"] == 2
    assert [step for step, _ in resumed_tracker.logs] == [2]
    assert len(resumed_tracker.artifacts) == 1
    assert resumed_tracker.logs[0][1]["policy/sync_success"] is True
    assert resumed_tracker.logs[0][1]["policy/canary_success"] is True
    assert events.count("publish:2") == 1
    assert optimizer_checkpoint.read_bytes() == checkpoint_bytes

    rejected_backend, resumed_backend = backends[1:]
    for backend in (rejected_backend, resumed_backend):
        assert backend.config["director"]["behavior_policy_version"] == state_before["behavior_policy_version"]
        assert backend.config["director"]["behavior_adapter_checkpoint"] == state_before["behavior_adapter_checkpoint"]
        assert backend.config["director"]["optimizer_state_checkpoint"] == state_before["optimizer_state_checkpoint"]
    previous = [item for item in rejected_backend.collections if item[0] == "rollout"]
    fresh = [item for item in resumed_backend.collections if item[0] == "rollout"]
    assert len(previous) == len(fresh) == 28
    assert [item[1] for item in previous] == [item[1] for item in fresh]
    assert all(new[2] - old[2] == 100_000_000 for old, new in zip(previous, fresh))
    assert all(item[3] == state_before["behavior_policy_version"] for item in fresh)
    assert all(item[4] == state_before["behavior_adapter_name"] for item in fresh)
    old_ids = {read_json_line["trajectory_id"] for read_json_line in map(json.loads, (archive / "trajectories.jsonl").read_text().splitlines())}
    new_ids = {read_json_line["trajectory_id"] for read_json_line in map(json.loads, (root / "steps/step_000002/trajectories.jsonl").read_text().splitlines())}
    assert old_ids.isdisjoint(new_ids)
    state_after = read_json(root / "run_state.json")
    assert state_after["optimizer_updates_completed"] == 2
    assert state_after["behavior_policy_version"] != state_before["behavior_policy_version"]


@pytest.mark.parametrize(
    "outcome", ["unknown_failure", "unknown_exclusion", "post_update_failure"]
)
def test_unknown_or_post_update_failure_is_not_automatically_retried(runner, outcome):
    run, root, events, backends = runner
    run()
    state_bytes = (root / "run_state.json").read_bytes()
    tracker = FakeTracker()
    with pytest.raises(HotpotTrainingError):
        run(resume=True, boundary=2, outcome=outcome, tracker=tracker)
    assert read_json(root / "run_manifest.json")["status"] == "failed"
    assert (root / "run_state.json").read_bytes() == state_bytes
    assert (root / "recovery/current_step.json").is_file()
    assert (root / "steps/step_000002").is_dir()
    assert not (root / "recovery/zero_update_retry_state.json").exists()
    assert not (root / "recovery/zero_update_attempts").exists()
    assert tracker.artifacts == []
    assert tracker.logs == []
    assert "publish:2" not in events
    backend_count = len(backends)
    with pytest.raises((ValueError, OSError, HotpotTrainingError)):
        run(resume=True, boundary=2)
    assert len(backends) == backend_count
    assert (root / "run_state.json").read_bytes() == state_bytes
    assert (root / "recovery/current_step.json").is_file()
