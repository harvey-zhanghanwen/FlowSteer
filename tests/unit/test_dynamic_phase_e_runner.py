"""Injected Phase-E runner tests with no GPU, API, optimizer, or W&B use."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.interactive.persistence import stable_id
from src.interactive.records import (
    EvaluationReceipt,
    TaskRecord,
    TrajectoryRecord,
    TurnRecord,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_hotpotqa_grpo.py"
DYNAMIC_CONFIG = ROOT / "config" / "training_hotpotqa_dynamic_ledger_grpo.yaml"
SPEC = importlib.util.spec_from_file_location("test_dynamic_phase_e_runner_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class JsonRecord:
    """Small artifact object accepted by the production JSONL writer."""

    def __init__(self, record_id: str, **values: object) -> None:
        self.record_id = record_id
        self.values = {"record_id": record_id, **values}

    def to_dict(self) -> dict[str, object]:
        return dict(self.values)


class FakeSyntheticAcceptance:
    synthetic_gates_passed = True

    def assert_ready_for_phase_e(self) -> None:
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "latent-loss-phase-acceptance-v1",
            "synthetic_gates_passed": True,
            "phases": {name: {"status": "passed"} for name in "ABCD"},
        }


class FakeOwnedRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.start_count = 0
        self.stop_count = 0

    def start_rollout_service(self) -> None:
        self.start_count += 1
        self.events.append("runtime:start")

    def stop_rollout_service(self) -> None:
        self.stop_count += 1
        self.events.append("runtime:stop")


class FakePhaseEBackend:
    """Injected collector; training/publication entry points are fail-fast."""

    model_catalog_version = "catalog-phase-e-test-v1"

    def __init__(self, config: dict, events: list[str], *, fail_natural: bool) -> None:
        self.config = config
        self.events = events
        self.fail_natural = fail_natural
        self.registry = SimpleNamespace(model_ids=("worker-model-a", "worker-model-b"))
        self.adapter = str(config["director"]["behavior_adapter_name"])
        self.natural_calls: list[tuple[str, int, str]] = []
        self.probe_calls: list[tuple[str, str]] = []

    async def ensure_behavior_ready(self) -> dict[str, object]:
        self.events.append("backend:ready")
        return {"success": True, "status": "fake-route-ready"}

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
    ) -> TrajectoryRecord:
        del observation_hook, skills
        if self.fail_natural:
            raise RuntimeError("injected natural collection failure")
        assert expected_task_split == "train"
        assert task.split == "train"
        assert condition_id
        self.natural_calls.append((task.task_id, rollout_index, condition_id))
        return _natural_trajectory(
            task,
            rollout_index,
            versions,
            adapter=self.adapter,
            condition_id=condition_id,
        )

    async def collect_probe_branch(
        self,
        task,
        rollout_index,
        versions,
        *,
        initial_snapshot,
        intervention,
        branch_id,
        expected_task_split="train",
        observation_hook=None,
        skills=(),
        condition_id=None,
    ) -> JsonRecord:
        del initial_snapshot, intervention, observation_hook, skills
        assert expected_task_split == "train"
        assert condition_id
        assert versions.policy == self.config["director"]["behavior_policy_version"]
        self.probe_calls.append((task.task_id, branch_id))
        return JsonRecord(
            branch_id,
            trajectory_id=branch_id,
            task_id=task.task_id,
            rollout_index=rollout_index,
            forced_probe=True,
            grpo_eligible=False,
            evidence_plane="paired_intervention",
        )

    def train(self, *args, **kwargs):
        raise AssertionError("Phase E must not call optimizer training")

    async def publish(self, *args, **kwargs):
        raise AssertionError("Phase E must not publish policy weights")


class FakeCoordinator:
    """Preserves the runner boundary while avoiding live hook/model calls."""

    instances: list["FakeCoordinator"] = []

    def __init__(self, *, epoch, **kwargs) -> None:
        del kwargs
        self.epoch = epoch
        self.natural_batch = None
        self.final_batch = None
        type(self).instances.append(self)

    async def collect_natural(
        self,
        backend: FakePhaseEBackend,
        tasks,
        *,
        rollouts_per_task,
        start_rollout_index,
    ):
        assert len(tasks) == 50
        assert len({task.task_id for task in tasks}) == 50
        assert rollouts_per_task == 4
        trajectories = []
        next_index = start_rollout_index
        for task in tasks:
            for _ in range(rollouts_per_task):
                trajectories.append(
                    await backend.collect(
                        task,
                        next_index,
                        self.epoch.condition.versions,
                        expected_task_split="train",
                        observation_hook=object(),
                        condition_id=self.epoch.condition.condition_id,
                    )
                )
                next_index += 1
        sidecars = tuple(
            JsonRecord(
                f"sidecar:{item.trajectory_id}",
                trajectory_id=item.trajectory_id,
                evidence_plane="natural",
            )
            for item in trajectories
        )
        ledgers = tuple(
            JsonRecord(
                f"ledger:{item.trajectory_id}",
                trajectory_id=item.trajectory_id,
                is_natural=True,
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
        backend: FakePhaseEBackend,
        natural,
        *,
        selected_sites,
        start_rollout_index,
    ):
        assert natural is self.natural_batch
        assert len(natural.natural_trajectories) == 200
        assert not natural.intervention_trajectories
        branches = []
        probe_records = []
        for probe_ordinal, selection in enumerate(selected_sites):
            natural_record = natural.natural_trajectories[probe_ordinal]
            probe_id = f"probe:{natural_record.trajectory_id}"
            probe_records.append(
                JsonRecord(
                    probe_id,
                    probe_id=probe_id,
                    evidence_plane="paired_intervention",
                    enters_grpo=False,
                    enters_standard_metrics=False,
                    audit=selection.audit,
                )
            )
            for arm in range(6):
                branch_id = f"{probe_id}:branch:{arm}"
                branches.append(
                    await backend.collect_probe_branch(
                        natural_record.task,
                        start_rollout_index + probe_ordinal * 10 + arm,
                        self.epoch.condition.versions,
                        initial_snapshot=object(),
                        intervention=object(),
                        branch_id=branch_id,
                        condition_id=self.epoch.condition.condition_id,
                    )
                )
        branch_ledgers = tuple(
            JsonRecord(
                f"ledger:{item.record_id}",
                trajectory_id=item.record_id,
                is_natural=False,
                enters_grpo=False,
            )
            for item in branches
        )
        self.final_batch = SimpleNamespace(
            epoch=self.epoch,
            natural_trajectories=natural.natural_trajectories,
            natural_sidecars=natural.natural_sidecars,
            natural_ledger_records=natural.natural_ledger_records,
            selected_sites=tuple(selected_sites),
            intervention_trajectories=tuple(branches),
            intervention_ledger_records=branch_ledgers,
            probe_records=tuple(probe_records),
        )
        return self.final_batch


def _task(index: int, *, split: str) -> TaskRecord:
    return TaskRecord(
        task_id=f"hotpotqa:{split}:{index}",
        question=f"HotpotQA {split} question {index}?",
        ground_truth="answer",
        split=split,
        metadata={
            "dataset_key": "hotpotqa",
            "source": "HotpotQA",
            "sampling": {"base_task_id": f"hotpotqa:{split}:base:{index}"},
        },
    )


def _write_tasks(path: Path, *, split: str, count: int) -> None:
    rows = []
    for index in range(count):
        task = _task(index, split=split)
        rows.append(
            json.dumps(
                {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _phase_e_project(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    config = yaml.safe_load(DYNAMIC_CONFIG.read_text(encoding="utf-8"))
    config["data"].update(
        train_path="data/train.jsonl",
        validation_path="data/validation.jsonl",
    )
    config["experiment"]["output_dir"] = "artifacts/phase_e"
    config["storage"].update(
        root="artifacts/phase_e/evidence",
        manifest_path="artifacts/phase_e/run_manifest.json",
        state_path="artifacts/phase_e/run_state.json",
        training_log_path="artifacts/phase_e/training_log.jsonl",
    )
    config_path = tmp_path / "config" / "dynamic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_tasks(tmp_path / "data" / "train.jsonl", split="train", count=512)
    _write_tasks(
        tmp_path / "data" / "validation.jsonl", split="validation", count=128
    )
    return config_path


def _natural_trajectory(
    task,
    rollout_index,
    versions,
    *,
    adapter: str,
    condition_id: str,
) -> TrajectoryRecord:
    graph = {}
    snapshot_id = stable_id(
        "snapshot", {"revision": 0, "graph": graph, "previous_snapshot_id": None}
    )
    turn = TurnRecord(
        turn_id=f"turn:{task.task_id}:{rollout_index}",
        round_index=0,
        prompt="ordinary prompt",
        policy_response='{"action":"finish"}',
        prompt_token_ids=(1,),
        output_token_ids=(2,),
        behavior_log_probs=(-0.1,),
        executed_prefix_tokens=1,
        action={"action": "finish"},
        canvas_feedback="workflow finished",
        graph_revision=0,
        graph_snapshot=graph,
        graph_snapshot_id=snapshot_id,
        previous_graph_snapshot_id=None,
        policy_version=versions.policy,
        policy_adapter=adapter,
        server_weight_version="default",
        receipt_verified=True,
    )
    reward = float(rollout_index % 2)
    return TrajectoryRecord(
        trajectory_id=f"trajectory:{task.task_id}:{rollout_index}",
        task=task,
        group_id=f"{task.task_id}:{condition_id}:{versions.policy}",
        condition_id=condition_id,
        rollout_id=f"rollout:{rollout_index}",
        versions=versions,
        turns=(turn,),
        final_answer="answer",
        evaluation=EvaluationReceipt(
            evaluator_version=versions.evaluator,
            valid=True,
            reward=reward,
            metrics={"exact_match": reward, "token_f1": reward},
        ),
        termination_reason="finish",
        explicit_finish=True,
    )


def _selected_sites(count: int = 2):
    return tuple(
        SimpleNamespace(
            audit=index == count - 1,
            sampling_probability=0.05 if index == count - 1 else 0.10,
            site=JsonRecord(f"site:{index}", trajectory_id=f"trajectory:{index}"),
        )
        for index in range(count)
    )


def _phase_e_metrics(batch) -> dict[str, object]:
    natural_ids = {item.trajectory_id for item in batch.natural_trajectories}
    intervention_ids = {item.record_id for item in batch.intervention_trajectories}
    assert len(natural_ids) == 200
    assert len(intervention_ids) == 6 * len(batch.probe_records)
    assert natural_ids.isdisjoint(intervention_ids)
    assert all(item.grpo_eligible for item in batch.natural_trajectories)
    assert all(item.values["grpo_eligible"] is False for item in batch.intervention_trajectories)
    return {
        "natural_trajectory_count": 200,
        "valid_natural_trajectory_count": 200,
        "grpo_trajectory_count": 200,
        "intervention_trajectory_count": len(batch.intervention_trajectories),
        "standard_metric_intervention_count": 0,
        "probe_count": sum(not item.audit for item in batch.selected_sites),
        "audit_count": sum(item.audit for item in batch.selected_sites),
        "exact_match": 0.5,
        "token_f1": 0.5,
        "terminal_failure_count": 0,
        "warning_precision": 0.5,
        "warning_recall": 0.5,
        "decision_key_explained_variance": 0.25,
        "routing_assertions": {
            "natural_only_grpo": True,
            "probe_and_audit_excluded_from_grpo": True,
            "probe_and_audit_excluded_from_standard_metrics": True,
            "comparison_fit_source": "same_snapshot_paired_probes_only",
            "latent_risk_reward_contribution": 0.0,
            "skill_reward_contribution": 0.0,
        },
    }


async def _phase0_receipt(trajectories, **kwargs) -> dict[str, object]:
    assert len(trajectories) == 200
    assert len({item.task.task_id for item in trajectories}) == 50
    assert kwargs["expected_count"] == 200
    return {
        "schema_version": "flowsteer.hotpotqa.phase0_lineage.v1",
        "status": "passed",
        "trajectory_count": 200,
        "task_count": 50,
        "optimizer_updates": 0,
        "ttb_enabled": False,
    }


def _patches(runtime: FakeOwnedRuntime):
    return (
        patch.object(RUNNER, "run_phase_acceptance", return_value=FakeSyntheticAcceptance()),
        patch.object(RUNNER, "_dynamic_role_boundaries", return_value=(object(), object())),
        patch.object(RUNNER, "DynamicLedgerEpochCoordinator", FakeCoordinator),
        patch.object(RUNNER, "_single_gpu_sequential_runtime", return_value=runtime),
        patch.object(RUNNER, "select_probe_sites", return_value=_selected_sites()),
        patch.object(RUNNER, "dynamic_epoch_metrics", side_effect=_phase_e_metrics),
        patch.object(RUNNER, "validate_phase0_rollout_batch", side_effect=_phase0_receipt),
        patch.object(
            RUNNER,
            "WandbTracker",
            MagicMock(side_effect=AssertionError("Phase E must not start W&B")),
        ),
        patch.object(RUNNER.os, "fsync"),
    )


def test_phase_e_routes_50x4_natural_and_paired_probes_without_training(
    tmp_path: Path,
) -> None:
    FakeCoordinator.instances.clear()
    config_path = _phase_e_project(tmp_path)
    events: list[str] = []
    runtime = FakeOwnedRuntime(events)
    backends: list[FakePhaseEBackend] = []

    def backend_factory(config, project_root):
        assert project_root == tmp_path
        backend = FakePhaseEBackend(config, events, fail_natural=False)
        backends.append(backend)
        return backend

    contexts = _patches(runtime)
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7], contexts[8]:
        manifest = asyncio.run(
            RUNNER.run_hotpotqa_dynamic_phase_e(
                config_path,
                project_root=tmp_path,
                backend_factory=backend_factory,
            )
        )

    assert manifest["status"] == "passed"
    assert manifest["completed_natural_trajectories"] == 200
    assert manifest["completed_intervention_trajectories"] == 12
    assert manifest["completed_probe_records"] == 2
    assert manifest["optimizer_updates"] == 0
    assert manifest["wandb_started"] is False
    assert events == ["runtime:start", "backend:ready", "runtime:stop"]
    assert runtime.start_count == runtime.stop_count == 1

    backend = backends[0]
    assert len(backend.natural_calls) == 200
    per_task: dict[str, int] = {}
    for task_id, _, _ in backend.natural_calls:
        per_task[task_id] = per_task.get(task_id, 0) + 1
    assert len(per_task) == 50
    assert set(per_task.values()) == {4}
    assert len(backend.probe_calls) == 12

    receipt = json.loads(Path(manifest["acceptance_receipt"]).read_text(encoding="utf-8"))
    assert receipt["all_required_gates_passed"] is True
    assert receipt["optimizer_step_authorized"] is True
    assert receipt["long_training_authorized"] is False
    assert receipt["optimizer_updates"] == 0
    assert receipt["wandb_started"] is False
    assert receipt["ttb_enabled"] is False
    checks = receipt["phase_e"]["checks"]
    assert checks["natural_trajectory_count"] is True
    assert checks["unique_question_count"] is True
    assert checks["same_condition_groups"] is True
    assert checks["natural_only_grpo"] is True
    assert checks["probe_and_audit_excluded_from_grpo"] is True
    assert checks["probe_and_audit_excluded_from_standard_metrics"] is True
    assert checks["six_branches_per_probe"] is True

    natural_rows = Path(manifest["artifacts"]["natural_trajectories"]).read_text(
        encoding="utf-8"
    ).splitlines()
    intervention_rows = Path(
        manifest["artifacts"]["intervention_trajectories"]
    ).read_text(encoding="utf-8").splitlines()
    assert len(natural_rows) == 200
    assert len(intervention_rows) == 12
    assert all(json.loads(row)["forced_probe"] is True for row in intervention_rows)
    assert all(json.loads(row)["grpo_eligible"] is False for row in intervention_rows)


def test_phase_e_failure_still_stops_only_the_injected_owned_runtime(
    tmp_path: Path,
) -> None:
    FakeCoordinator.instances.clear()
    config_path = _phase_e_project(tmp_path)
    events: list[str] = []
    runtime = FakeOwnedRuntime(events)
    backends: list[FakePhaseEBackend] = []

    def backend_factory(config, project_root):
        assert project_root == tmp_path
        backend = FakePhaseEBackend(config, events, fail_natural=True)
        backends.append(backend)
        return backend

    contexts = _patches(runtime)
    with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7], contexts[8]:
        with pytest.raises(RUNNER.HotpotTrainingError, match="dynamic Phase E failed"):
            asyncio.run(
                RUNNER.run_hotpotqa_dynamic_phase_e(
                    config_path,
                    project_root=tmp_path,
                    backend_factory=backend_factory,
                )
            )

    assert events == ["runtime:start", "backend:ready", "runtime:stop"]
    assert runtime.start_count == 1
    assert runtime.stop_count == 1
    assert not backends[0].probe_calls
    assert not list(tmp_path.rglob("acceptance_receipt.json"))
    manifests = list(tmp_path.rglob("latent_loss_acceptance/phase_e/*/manifest.json"))
    assert len(manifests) == 1
    failed = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "injected natural collection failure" in failed["error"]
    assert failed["optimizer_updates"] == 0
    assert failed["wandb_started"] is False
