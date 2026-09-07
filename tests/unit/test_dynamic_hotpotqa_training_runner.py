"""Offline orchestration test for the first dynamic HotpotQA GRPO step."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from tests.unit.test_hotpotqa_grpo_runner import (
    FakeReceipt,
    FakeSummary,
    FakeTracker,
    _task,
    _trajectory,
    _write_tasks,
    _write_validation_tasks,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_hotpotqa_grpo.py"
CONFIG = ROOT / "config" / "training_hotpotqa_dynamic_ledger_grpo.yaml"
SPEC = importlib.util.spec_from_file_location(
    "test_dynamic_hotpotqa_training_runner_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class _JsonRecord:
    def __init__(self, record_id: str, **values: object) -> None:
        self.record_id = record_id
        self.__dict__.update(values)
        self._values = {"record_id": record_id, **values}

    def to_dict(self) -> dict[str, object]:
        return dict(self._values)


class _Hook:
    def __init__(self, condition_id: str) -> None:
        self.condition_id = condition_id

    def sidecar(self, trajectory_id: str) -> _JsonRecord:
        return _JsonRecord(
            f"sidecar:{trajectory_id}",
            trajectory_id=trajectory_id,
            condition_id=self.condition_id,
        )


class _FakeCoordinator:
    def __init__(self, epoch) -> None:
        self.epoch = epoch
        self.natural_batch = None
        self.intervention_ids: tuple[str, ...] = ()

    def make_hook(self, seed: int) -> _Hook:
        del seed
        return _Hook(self.epoch.condition.condition_id)

    async def collect_natural(
        self,
        backend,
        tasks,
        *,
        rollouts_per_task: int,
        start_rollout_index: int,
    ):
        trajectories = []
        rollout_index = start_rollout_index
        for task in tasks:
            for _ in range(rollouts_per_task):
                trajectories.append(
                    await backend.collect(
                        task,
                        rollout_index,
                        self.epoch.condition.versions,
                        expected_task_split="train",
                        observation_hook=self.make_hook(rollout_index),
                        condition_id=self.epoch.condition.condition_id,
                    )
                )
                rollout_index += 1
        sidecars = tuple(
            _JsonRecord(
                f"natural-sidecar:{item.trajectory_id}",
                trajectory_id=item.trajectory_id,
                condition_id=self.epoch.condition.condition_id,
            )
            for item in trajectories
        )
        ledgers = tuple(
            _JsonRecord(
                f"natural-ledger:{item.trajectory_id}",
                trajectory_id=item.trajectory_id,
                is_natural=True,
                grpo_eligible=True,
            )
            for item in trajectories
        )
        self.natural_batch = SimpleNamespace(
            epoch=self.epoch,
            natural_trajectories=tuple(trajectories),
            natural_sidecars=sidecars,
            natural_ledger_records=ledgers,
            intervention_trajectories=(),
            intervention_ledger_records=(),
            probe_records=(),
        )
        return self.natural_batch

    async def collect_selected_probes(
        self,
        backend,
        natural,
        *,
        selected_sites,
        start_rollout_index: int,
    ):
        del backend, start_rollout_index
        assert natural is self.natural_batch
        interventions = tuple(
            _JsonRecord(
                f"probe-branch:{index}",
                trajectory_id=f"probe-branch:{index}",
                forced_probe=True,
                grpo_eligible=False,
                evidence_plane="paired_intervention",
            )
            for index in range(6)
        )
        self.intervention_ids = tuple(item.trajectory_id for item in interventions)
        return SimpleNamespace(
            epoch=self.epoch,
            natural_trajectories=natural.natural_trajectories,
            natural_sidecars=natural.natural_sidecars,
            natural_ledger_records=natural.natural_ledger_records,
            selected_sites=tuple(selected_sites),
            intervention_trajectories=interventions,
            intervention_ledger_records=tuple(
                _JsonRecord(
                    f"probe-ledger:{item.trajectory_id}",
                    trajectory_id=item.trajectory_id,
                    is_natural=False,
                    grpo_eligible=False,
                )
                for item in interventions
            ),
            probe_records=(
                _JsonRecord(
                    "probe:0",
                    probe_id="probe:0",
                    enters_grpo=False,
                    enters_standard_metrics=False,
                ),
            ),
        )


class _CloseResult:
    def __init__(self, next_epoch, natural_ids, intervention_ids) -> None:
        summary = _JsonRecord(
            "transition-summary",
            grpo_trajectory_ids=list(natural_ids),
            excluded_intervention_ids=list(intervention_ids),
            natural_trajectories=len(natural_ids),
            intervention_trajectories=len(intervention_ids),
            invalid_trajectories=0,
            baseline_updates=len(natural_ids) + len(intervention_ids),
            sensor_updates=0,
            contrast_probe_updates=1,
        )
        self.transition = SimpleNamespace(next_epoch=next_epoch, summary=summary)
        self.receipt = _JsonRecord(
            "transition-receipt",
            source_condition_id="initial-condition",
            next_condition_id=next_epoch.condition.condition_id,
        )

    @property
    def next_epoch(self):
        return self.transition.next_epoch


class _FakeBackend:
    model_catalog_version = "catalog-dynamic-runner-test-v1"

    def __init__(self, config: dict, events: list[dict[str, object]]) -> None:
        self.config = config
        self.events = events
        self.registry = SimpleNamespace(model_ids=("worker-model",))
        self.active_policy = str(config["director"]["behavior_policy_version"])
        self.active_adapter = str(config["director"]["behavior_adapter_name"])
        self.trainer_inputs = ()
        self.checkpoint: Path | None = None
        self.owner_loop = None

    async def ensure_behavior_ready(self):
        return {"success": True, "status": "offline-ready"}

    async def collect(
        self,
        task,
        rollout_index,
        versions,
        *,
        expected_task_split="train",
        observation_hook=None,
        skills=(),
        condition_id=None,
    ):
        del skills
        current_loop = asyncio.get_running_loop()
        if self.owner_loop is None:
            self.owner_loop = current_loop
        assert current_loop is self.owner_loop
        assert expected_task_split == task.split
        kind = (
            "validation"
            if task.split == "validation"
            else "canary"
            if rollout_index >= 1_000_000
            else "natural"
        )
        self.events.append(
            {
                "kind": kind,
                "policy": versions.policy,
                "condition_id": condition_id,
                "hook_condition_id": getattr(observation_hook, "condition_id", None),
                "versions_fingerprint": versions.fingerprint,
            }
        )
        value = _trajectory(
            task,
            rollout_index,
            versions,
            adapter=self.active_adapter,
        )
        return replace(
            value,
            condition_id=str(condition_id),
            group_id=f"{task.task_id}:{condition_id}:{versions.fingerprint}",
        )

    def train(self, trajectories, output_dir):
        self.trainer_inputs = tuple(trajectories)
        checkpoint = Path(output_dir) / "checkpoint_final" / "theta"
        checkpoint.mkdir(parents=True)
        training_state = checkpoint / "training_state.pt"
        training_state.write_bytes(b"offline-state")
        self.checkpoint = checkpoint
        return FakeSummary(
            {
                "optimizer_updates": 1,
                "input_trajectories": len(trajectories),
                "informative_groups": 7,
                "trained_trajectories": len(trajectories),
                "loss": 0.25,
                "grad_norm": 1.0,
                "trainable_update_l2": 0.1,
                "behavior_policy_version": self.active_policy,
                "updated_policy_version": self.config["director"][
                    "updated_policy_version"
                ],
                "micro_batch_size_used": 4,
                "oom_backoff_count": 0,
                "checkpoint_dir": str(checkpoint),
                "optimizer_state_checkpoint": str(training_state),
                "optimizer_state_saved": True,
                "training_state_checkpoint": str(training_state),
                "training_state_saved": True,
                "scheduler_state_saved": True,
                "rng_state_saved": True,
                "checkpoint_recoverable": True,
                "scheduler_resume_status": "restored_scheduler_and_rng",
                "learning_rate": 1.0e-4,
                "gpu_memory_allocated_mib": {"cuda:5": 1.0},
                "committed_step": int(self.config["experiment"]["update_step"]),
            }
        )

    async def publish(self, summary):
        assert asyncio.get_running_loop() is self.owner_loop
        candidate = summary.to_dict()["updated_policy_version"]
        self.active_policy = candidate
        self.active_adapter = "theta_dynamic_step_000002"
        return FakeReceipt(
            {
                "success": True,
                "status": "published",
                "behavior_policy_version": self.config["director"][
                    "behavior_policy_version"
                ],
                "candidate_policy_version": candidate,
                "new_policy_version": candidate,
                "adapter_name": self.active_adapter,
                "checkpoint_version": f"checkpoint:{candidate}",
                "route_switch_success": True,
            }
        )

    async def publish_without_gate(self, summary):
        return await self.publish(summary)


class _FakeSequentialRuntime:
    """Run lifecycle callbacks in the worker thread used by the real runner."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start_rollout_service(self) -> int:
        self.started = True
        self.stopped = False
        return 43210

    def stop_rollout_service(self) -> None:
        self.stopped = True

    def run_update_cycle(
        self,
        *,
        training_callback,
        publication_callback,
        canary_callback,
    ):
        assert self.started and not self.stopped
        training_callback()
        publication_callback()
        canary_callback()
        return _JsonRecord(
            "single-gpu-cycle",
            success=True,
            status="rollout_resumed_after_verified_policy_update",
        )


def _project(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["data"].update(
        train_path="data/train.jsonl",
        validation_path="data/validation.jsonl",
    )
    config["tracking"]["validation_protocol"]["monitor_task_ids"] = [
        f"hotpotqa:validation:{index}" for index in range(7)
    ]
    config["experiment"]["output_dir"] = "artifacts/dynamic-training"
    config["storage"].update(
        root="artifacts/dynamic-training/evidence",
        manifest_path="artifacts/dynamic-training/run_manifest.json",
        state_path="artifacts/dynamic-training/run_state.json",
        training_log_path="artifacts/dynamic-training/training_log.jsonl",
    )
    config_path = tmp_path / "config" / "training.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_tasks(tmp_path / "data" / "train.jsonl")
    _write_validation_tasks(tmp_path / "data" / "validation.jsonl")

    versions = RUNNER.version_bundle_for(
        _task(0),
        policy_version=config["director"]["behavior_policy_version"],
        model_catalog_version=_FakeBackend.model_catalog_version,
        prompt_version=config["versions"]["prompt"],
        tool_version=config["versions"]["tool"],
    )
    initial_epoch = RUNNER.freeze_initial_epoch(
        versions=versions,
        model_ids=("worker-model",),
    )
    acceptance_root = tmp_path / "phase-e"
    epoch_receipt = initial_epoch.save(acceptance_root / "ledger_epoch_000000")
    acceptance_path = acceptance_root / "acceptance_receipt.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "phase_e": {
                    "status": "passed",
                    "epoch_snapshot": epoch_receipt.to_dict(),
                },
                "all_required_gates_passed": True,
                "optimizer_step_authorized": True,
                "optimizer_updates": 0,
                "ttb_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    return config, config_path, acceptance_path


def test_first_dynamic_step_is_natural_only_and_commits_next_condition(
    tmp_path: Path,
) -> None:
    config, config_path, acceptance_path = _project(tmp_path)
    tracker = FakeTracker()
    events: list[dict[str, object]] = []
    backends: list[_FakeBackend] = []
    coordinators: list[_FakeCoordinator] = []
    close_results: list[_CloseResult] = []
    sequential_runtime = _FakeSequentialRuntime()

    def backend_factory(step_config, project_root):
        assert project_root == tmp_path
        backend = _FakeBackend(step_config, events)
        backends.append(backend)
        return backend

    def coordinator_factory(step_config, backend, epoch, **kwargs):
        del step_config, backend, kwargs
        coordinator = _FakeCoordinator(epoch)
        coordinators.append(coordinator)
        return coordinator

    def close_epoch(batch, *, next_policy_version, **kwargs):
        natural_ids = tuple(item.trajectory_id for item in batch.natural_trajectories)
        intervention_ids = tuple(
            item.trajectory_id for item in batch.intervention_trajectories
        )
        assert tuple(kwargs["natural_grpo_trajectory_ids"]) == natural_ids
        assert tuple(kwargs["intervention_exclusion_trajectory_ids"]) == intervention_ids
        next_versions = replace(
            batch.epoch.condition.versions,
            policy=next_policy_version,
        )
        next_epoch = RUNNER.freeze_initial_epoch(
            versions=next_versions,
            model_ids=("worker-model",),
            epoch=batch.epoch.condition.epoch + 1,
        )
        result = _CloseResult(next_epoch, natural_ids, intervention_ids)
        close_results.append(result)
        return result

    selected_site = SimpleNamespace(
        audit=False,
        sampling_probability=0.1,
        site=_JsonRecord("site:0", trajectory_id="natural:0"),
    )
    dynamic_metrics = {
        "probe_count": 1,
        "audit_count": 0,
        "warning_precision": 0.5,
        "warning_recall": 0.25,
        "decision_key_explained_variance": 0.125,
        "routing_assertions": {
            "natural_only_grpo": True,
            "probe_and_audit_excluded_from_grpo": True,
            "probe_and_audit_excluded_from_standard_metrics": True,
        },
    }

    with (
        patch.object(
            RUNNER,
            "_single_gpu_sequential_runtime",
            return_value=sequential_runtime,
        ),
        patch.object(RUNNER, "_dynamic_epoch_coordinator", side_effect=coordinator_factory),
        patch.object(RUNNER, "select_probe_sites", return_value=(selected_site,)),
        patch.object(RUNNER, "dynamic_epoch_metrics", return_value=dynamic_metrics),
        patch.object(RUNNER, "close_dynamic_epoch", side_effect=close_epoch),
        patch.object(
            RUNNER,
            "WandbTracker",
            MagicMock(side_effect=AssertionError("W&B must remain injected/offline")),
        ),
        patch.object(RUNNER.os, "fsync"),
    ):
        manifest = asyncio.run(
            RUNNER.run_hotpotqa_training(
                config_path,
                project_root=tmp_path,
                allow_md_grpo=True,
                stop_after_optimizer_steps=1,
                backend_factory=backend_factory,
                tracker=tracker,
                dynamic_acceptance_receipt=acceptance_path,
            )
        )

    assert manifest["status"] == "paused_at_requested_boundary"
    assert manifest["optimizer_updates_completed"] == 1
    assert len(backends) == 1
    assert len(coordinators) == 2
    assert len(close_results) == 1
    assert sequential_runtime.started
    assert sequential_runtime.stopped

    backend = backends[0]
    natural_ids = {
        event_index
        for event_index, event in enumerate(events)
        if event["kind"] == "natural"
    }
    assert len(natural_ids) == 28
    assert len(backend.trainer_inputs) == 28
    assert all(item.grpo_eligible and not item.forced_probe for item in backend.trainer_inputs)
    trained_ids = {item.trajectory_id for item in backend.trainer_inputs}
    assert trained_ids == {
        item.trajectory_id
        for item in coordinators[0].natural_batch.natural_trajectories
    }
    assert trained_ids.isdisjoint(coordinators[0].intervention_ids)

    next_epoch = close_results[0].next_epoch
    next_condition = next_epoch.condition
    updated_calls = [
        event for event in events if event["kind"] in {"canary", "validation"}
    ]
    assert [event["kind"] for event in updated_calls].count("canary") == 1
    assert [event["kind"] for event in updated_calls].count("validation") == 7
    assert all(event["policy"] == next_condition.versions.policy for event in updated_calls)
    assert all(event["condition_id"] == next_condition.condition_id for event in updated_calls)
    assert all(
        event["hook_condition_id"] == next_condition.condition_id
        for event in updated_calls
    )
    assert all(
        event["versions_fingerprint"] == next_condition.versions.fingerprint
        for event in updated_calls
    )

    assert backend.checkpoint is not None
    ledger_receipt_path = backend.checkpoint / "ledger_epoch" / "receipt.json"
    recovery_manifest_path = backend.checkpoint / "recovery_manifest.json"
    checkpoint_acceptance_path = (
        backend.checkpoint / "phase_e_acceptance_receipt.json"
    )
    assert ledger_receipt_path.is_file()
    assert recovery_manifest_path.is_file()
    assert checkpoint_acceptance_path.is_file()
    saved_receipt = json.loads(ledger_receipt_path.read_text(encoding="utf-8"))
    assert saved_receipt["condition"]["condition_id"] == next_condition.condition_id
    recovery_manifest = json.loads(
        recovery_manifest_path.read_text(encoding="utf-8")
    )
    assert recovery_manifest["status"] == "ready"
    assert recovery_manifest["objective"] == "action_masked_one_pass_grpo"
    assert recovery_manifest["ttb_enabled"] is False
    recovered_state = RUNNER._state_from_dynamic_checkpoint(backend.checkpoint)
    assert recovered_state["optimizer_updates_completed"] == 1
    assert recovered_state["behavior_policy_version"] == next_condition.versions.policy
    assert recovered_state["ledger_epoch_receipt"] == str(ledger_receipt_path)
    assert recovered_state["dynamic_phase_e_acceptance"][
        "all_required_gates_passed"
    ] is True

    transition = json.loads(
        (
            tmp_path
            / "artifacts/dynamic-training/steps/step_000001/dynamic_epoch_transition.json"
        ).read_text(encoding="utf-8")
    )
    assert transition["receipt"]["next_epoch_snapshot"]["condition"][
        "condition_id"
    ] == next_condition.condition_id
    assert transition["receipt"]["next_epoch_receipt_path"] == str(
        ledger_receipt_path
    )

    state = json.loads(
        (tmp_path / "artifacts/dynamic-training/run_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert Path(state["ledger_epoch_receipt"]) == ledger_receipt_path
    assert state["ledger_condition_id"] == next_condition.condition_id
    assert state["ledger_snapshot_id"] == next_condition.ledger_snapshot_id
    assert state["skill_snapshot_id"] == next_condition.skill_snapshot_id

    closure = json.loads(
        (tmp_path / "artifacts/dynamic-training/one_step_closure_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert closure["status"] == "passed"
    assert closure["long_training_authorized"] is False
    assert closure["ledger_epoch_receipt"] == str(ledger_receipt_path)

    assert len(tracker.logs) == 1
    logged = tracker.logs[0][1]
    assert set(config["tracking"]["required_step_fields"]) <= set(logged)
    assert logged["ledger/heldout_nll"] is None
    assert logged["ledger/heldout_brier"] is None
    assert logged["policy/canary_success"] is True
    assert tracker.finished == [0]


def test_bounded_phase_e_rejects_training_only_flags(
    tmp_path: Path,
    capsys,
) -> None:
    _, config_path, _ = _project(tmp_path)
    assert (
        RUNNER.main(
            [
                "--config",
                str(config_path),
                "--dynamic-phase-e-only",
                "--allow-md-grpo",
            ]
        )
        == 2
    )
    assert "bounded modes cannot be combined" in capsys.readouterr().err
