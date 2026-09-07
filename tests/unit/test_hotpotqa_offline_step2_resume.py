"""Offline committed-step-1 to step-2 resume integration test.

The production runner is exercised twice, but every external boundary is
injected: no GPU process, model/API request, or W&B SDK run is started.  The
first call materializes a committed step-1 artifact tree with the evidence
fields used by the real supplemental admission gate.  The second call must
derive a complete no-candidate Skill decision, authorize only
evidence-producing continuation, and commit exactly one step-2 update.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.interactive.exploration.ledger_probe import make_probe_branch_ids
from src.interactive.records import ProbeRecord
from tests.unit import test_dynamic_hotpotqa_training_runner as fixture
from tests.unit.test_hotpotqa_grpo_runner import FakeReceipt, FakeSummary, FakeTracker


RUNNER = fixture.RUNNER


class _CommittedShapeBackend(fixture._FakeBackend):
    """Extend the existing fake with fields present in the real step receipt."""

    def __init__(self, config, events) -> None:
        super().__init__(config, events)
        self.training_step = int(config["experiment"]["training_step"])
        self.train_calls = 0

    def train(self, trajectories, output_dir):
        self.train_calls += 1
        summary = super().train(trajectories, output_dir).to_dict()
        summary.update(
            exact_groups=7,
            record_eligible_trajectories=len(trajectories),
        )
        return FakeSummary(summary)

    async def publish(self, summary):
        receipt = (await super().publish(summary)).to_dict()
        candidate = str(receipt["candidate_policy_version"])
        receipt.update(
            route_policy_version=candidate,
            checkpoint_path=str(summary.to_dict()["checkpoint_dir"]),
            canary_succeeded=True,
        )
        return FakeReceipt(receipt)

    async def publish_without_gate(self, summary):
        return await self.publish(summary)


class _RecoverableProbeCoordinator(fixture._FakeCoordinator):
    """Persist a complete ProbeRecord that the resume-time Skill gate can read."""

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
        source = natural.natural_trajectories[0]
        probe_id = (
            f"probe:{self.epoch.condition.epoch}:"
            f"{source.task.task_id}:{source.rollout_id}"
        )
        branches = make_probe_branch_ids(
            probe_id,
            seed=100 + self.epoch.condition.epoch,
        )
        catalog = self.epoch.condition.versions.model_catalog
        keep = {
            "task_family": "hotpotqa",
            "role_cluster": "retrieve",
            "model_id": "worker-model",
            "edge_type": "independent",
            "same_model_as_upstream": False,
            "stage": "other",
        }
        switch = {**keep, "model_id": "worker-model-alternative"}
        probe = ProbeRecord(
            probe_id=probe_id,
            problem_id=source.task.task_id,
            task_split="train",
            snapshot_id=f"snapshot:{probe_id}",
            policy_version=self.epoch.condition.versions.policy,
            state_features={"is_audit": False},
            incumbent_action=keep,
            candidate_action=switch,
            sampling_probability=0.1,
            incumbent_returns=(1.0, 1.0, 1.0),
            candidate_returns=(1.0, 1.0, 1.0),
            executor_versions={
                "worker-model": catalog,
                "worker-model-alternative": catalog,
            },
            evaluator_version=self.epoch.condition.versions.evaluator,
            feature_schema_version=self.epoch.condition.versions.feature_schema,
            branch_order=branches.execution_order,
        )
        interventions = tuple(
            fixture._JsonRecord(
                branch_id,
                trajectory_id=branch_id,
                forced_probe=True,
                grpo_eligible=False,
                evidence_plane="paired_intervention",
            )
            for branch_id in branches.execution_order
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
                fixture._JsonRecord(
                    f"probe-ledger:{item.trajectory_id}",
                    trajectory_id=item.trajectory_id,
                    is_natural=False,
                    grpo_eligible=False,
                )
                for item in interventions
            ),
            probe_records=(probe,),
        )


def _add_real_admission_fields(acceptance_path: Path) -> None:
    payload = json.loads(acceptance_path.read_text(encoding="utf-8"))
    payload["phase_e"]["checks"] = {
        "natural_only_grpo": True,
        "same_condition_groups": True,
        "unique_question_count": True,
        "probe_and_audit_excluded_from_grpo": True,
    }
    payload["phase_e"]["metrics"] = {
        "routing_assertions": {
            "comparison_fit_source": "same_snapshot_paired_probes_only",
            "natural_only_grpo": True,
            "probe_and_audit_excluded_from_grpo": True,
            "probe_and_audit_excluded_from_standard_metrics": True,
            "latent_risk_reward_contribution": 0,
            "skill_reward_contribution": 0,
        }
    }
    acceptance_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def test_resume_committed_step1_through_supplemental_gate_to_one_step2(
    tmp_path: Path,
) -> None:
    _, config_path, acceptance_path = fixture._project(tmp_path)
    _add_real_admission_fields(acceptance_path)

    events: list[dict[str, object]] = []
    backends: list[_CommittedShapeBackend] = []
    runtimes: list[fixture._FakeSequentialRuntime] = []

    def backend_factory(step_config, project_root):
        assert project_root == tmp_path
        backend = _CommittedShapeBackend(step_config, events)
        backends.append(backend)
        return backend

    def runtime_factory(*_args, **_kwargs):
        runtime = fixture._FakeSequentialRuntime()
        runtimes.append(runtime)
        return runtime

    def coordinator_factory(step_config, backend, epoch, **kwargs):
        del step_config, backend, kwargs
        return _RecoverableProbeCoordinator(epoch)

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
        return fixture._CloseResult(next_epoch, natural_ids, intervention_ids)

    selected_site = SimpleNamespace(
        audit=False,
        sampling_probability=0.1,
        site=fixture._JsonRecord("site:0", trajectory_id="natural:0"),
    )
    dynamic_metrics = {
        "probe_count": 1,
        "audit_count": 0,
        "warning_precision": 0.5,
        "warning_recall": 0.25,
        "decision_key_explained_variance": 0.125,
        "routing_assertions": {
            "comparison_fit_source": "same_snapshot_paired_probes_only",
            "natural_only_grpo": True,
            "probe_and_audit_excluded_from_grpo": True,
            "probe_and_audit_excluded_from_standard_metrics": True,
            "latent_risk_reward_contribution": 0,
            "skill_reward_contribution": 0,
        },
    }
    first_tracker = FakeTracker()
    second_tracker = FakeTracker()

    with (
        patch.object(
            RUNNER,
            "_single_gpu_sequential_runtime",
            side_effect=runtime_factory,
        ),
        patch.object(
            RUNNER,
            "_dynamic_epoch_coordinator",
            side_effect=coordinator_factory,
        ),
        patch.object(RUNNER, "select_probe_sites", return_value=(selected_site,)),
        patch.object(RUNNER, "dynamic_epoch_metrics", return_value=dynamic_metrics),
        patch.object(RUNNER, "close_dynamic_epoch", side_effect=close_epoch),
        patch.object(
            RUNNER,
            "WandbTracker",
            MagicMock(side_effect=AssertionError("W&B SDK must not be started")),
        ),
        patch.object(
            RUNNER.LiveSmokeBackend,
            "from_config",
            MagicMock(side_effect=AssertionError("live backend must not be built")),
        ),
        patch.object(RUNNER.os, "fsync"),
    ):
        first = asyncio.run(
            RUNNER.run_hotpotqa_training(
                config_path,
                project_root=tmp_path,
                allow_md_grpo=True,
                stop_after_optimizer_steps=1,
                backend_factory=backend_factory,
                tracker=first_tracker,
                dynamic_acceptance_receipt=acceptance_path,
            )
        )
        assert first["optimizer_updates_completed"] == 1
        run_root = tmp_path / "artifacts" / "dynamic-training"
        closure_path = run_root / "one_step_closure_receipt.json"
        closure_before = closure_path.read_bytes()
        closure = json.loads(closure_before)
        assert closure["long_training_authorized"] is False

        second = asyncio.run(
            RUNNER.run_hotpotqa_training(
                config_path,
                project_root=tmp_path,
                allow_md_grpo=True,
                resume=True,
                stop_after_optimizer_steps=2,
                backend_factory=backend_factory,
                tracker=second_tracker,
            )
        )

    assert second["status"] == "paused_at_requested_boundary"
    assert second["optimizer_updates_completed"] == 2
    assert closure_path.read_bytes() == closure_before
    assert json.loads(closure_path.read_text(encoding="utf-8"))[
        "long_training_authorized"
    ] is False

    admission_path = (
        run_root
        / "long_training_admission"
        / "step_000001"
        / "admission_receipt.json"
    )
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    assert admission["status"] == "authorized"
    assert admission["long_training_authorized"] is True
    assert admission["scope"] == "evidence-producing-training-only"
    assert admission["skill_runtime_status"] == "complete_no_qualified_candidate"
    assert admission["heldout_calibration_status"] == (
        "not_applicable_no_qualified_candidate"
    )
    assert admission["ttb_enabled"] is False

    skill_update = json.loads(
        (run_root / "skills" / "epoch_000001" / "runtime_update.json").read_text(
            encoding="utf-8"
        )
    )
    assert skill_update["receipt"]["status"] == "complete_no_qualified_candidate"
    assert skill_update["receipt"]["producer_complete"] is True
    assert skill_update["receipt"]["discovery_probe_count"] == 1
    assert skill_update["receipt"]["discovery_qualified_rule_count"] == 0
    assert skill_update["grpo_reward_contribution"] == 0.0
    assert skill_update["ttb_enabled"] is False

    assert [backend.training_step for backend in backends] == [1, 2]
    assert [backend.train_calls for backend in backends] == [1, 1]
    assert [step for step, _ in second_tracker.logs] == [2]
    step2 = json.loads(
        (run_root / "steps" / "step_000002" / "step_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert step2["status"] == "committed"
    assert step2["optimizer_step"] == 2
    assert step2["training"]["optimizer_updates"] == 1
    assert step2["dynamic_epoch"]["skill_runtime_update"]["status"] == (
        "complete_no_qualified_candidate"
    )
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["optimizer_updates_completed"] == 2
    assert not (run_root / "steps" / "step_000003").exists()
    assert len(runtimes) == 2
    assert all(runtime.started and runtime.stopped for runtime in runtimes)

