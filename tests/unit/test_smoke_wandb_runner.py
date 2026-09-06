from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import scripts.train_agentgraph_smoke as runner
from tests.unit.test_smoke_runner import create_project, make_task, trajectory


class _Artifact:
    def __init__(self, name, *, type, metadata):
        self.name = name
        self.type = type
        self.metadata = dict(metadata)
        self.directories = []

    def add_dir(self, path):
        self.directories.append(path)


class _LoggedArtifact:
    def wait(self):
        return None


class _Run:
    def __init__(self, events, *, url="https://wandb.example/run/trivia-step-1"):
        self.events = events
        self.url = url
        self.id = "trivia-step-1"
        self.logs = []
        self.artifacts = []
        self.finished = []

    def log(self, data, *, step, commit):
        self.events.append("wandb:log")
        self.logs.append((dict(data), step, commit))

    def log_artifact(self, artifact, *, aliases):
        self.artifacts.append((artifact, list(aliases)))
        return _LoggedArtifact()

    def finish(self, exit_code=0):
        self.events.append(f"wandb:finish:{exit_code}")
        self.finished.append(exit_code)


class _SDK:
    def __init__(self, events):
        self.events = events
        self.run = _Run(events)
        self.init_kwargs = None

    def init(self, **kwargs):
        self.events.append("wandb:init")
        self.init_kwargs = dict(kwargs)
        return self.run

    def Artifact(self, name, *, type, metadata):
        return _Artifact(name, type=type, metadata=metadata)


class _EvidenceProvider:
    def __init__(self, events, *, validation_complete=True):
        self.events = events
        self.validation_complete = validation_complete

    def preflight(self, *, config, selected_tasks):
        del config
        assert {task.metadata["dataset_key"] for task in selected_tasks} == {
            "triviaqa"
        }
        self.events.append("telemetry:preflight")

    def optimizer_step_evidence(self, *, config, manifest, summary):
        del config, manifest, summary
        if not self.validation_complete:
            return {
                "validation": {
                    "complete": False,
                    "status": "not_run_publish_failed",
                },
                "gpu_metrics": {"peak_memory_allocated_bytes": 1024.0},
            }
        return {
            "validation": {
                "complete": True,
                "scope": "held_out",
                "split": "validation",
                "evaluator_version": "triviaqa.official.answer.v1",
                "metrics": {"em": 50.0, "f1": 60.0},
                "is_best": False,
            },
            "gpu_metrics": {"peak_memory_allocated_bytes": 1024.0},
        }


async def _in_process_to_thread(function, /, *args, **kwargs):
    """Keep the fake backend in-process; no production executor semantics change."""

    return function(*args, **kwargs)


class _Summary:
    def __init__(self, checkpoint):
        self.value = {
            "optimizer_updates": 1,
            "behavior_policy_version": "qwen35-9b-base-step-0000",
            "updated_policy_version": "qwen35-9b-smoke-step-0001",
            "checkpoint_dir": str(checkpoint),
            "checkpoint_ready": True,
            "checkpoint_version": "checkpoint-step-0001",
            "optimizer_state_saved": True,
            "optimizer_state_checkpoint": str(checkpoint / "training_state.pt"),
            "training_state_checkpoint": str(checkpoint / "training_state.pt"),
            "training_state_format": "flowsteer-one-pass-grpo-training-state-v2",
            "scheduler_state_status": "disabled_constant_learning_rate",
            "rng_state_saved": True,
            "committed_step": 1,
            "loss": 0.25,
            "grad_norm": 1.5,
            "trainable_update_l2": 0.01,
        }
        for key, value in self.value.items():
            setattr(self, key, value)

    def to_dict(self):
        return dict(self.value)


class _Backend:
    model_catalog_version = "catalog-test-v1"

    def __init__(self, events, *, fail_publish=False):
        self.events = events
        self.fail_publish = fail_publish

    async def collect(self, task, rollout_index, versions):
        self.events.append(f"collect:{rollout_index}")
        return trajectory(task, rollout_index, versions)

    def train(self, trajectories, output_dir):
        consumed_trajectory_ids = tuple(
            record.trajectory_id for record in trajectories
        )
        consumed_group_keys = tuple(
            sorted({record.group_key for record in trajectories})
        )
        self.events.append("optimizer:step")
        checkpoint = output_dir / "checkpoint_final" / "theta"
        checkpoint.mkdir(parents=True)
        (checkpoint / "training_state.pt").write_bytes(b"complete-state")
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "policy_version.json").write_text(
            json.dumps(
                {
                    "committed_step": 1,
                    "behavior_policy_version": "qwen35-9b-base-step-0000",
                    "updated_policy_version": "qwen35-9b-smoke-step-0001",
                    "checkpoint_version": "checkpoint-step-0001",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary = _Summary(checkpoint)
        summary.value["consumed_trajectory_ids"] = consumed_trajectory_ids
        summary.value["consumed_group_keys"] = consumed_group_keys
        summary.consumed_trajectory_ids = consumed_trajectory_ids
        summary.consumed_group_keys = consumed_group_keys
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary.to_dict()) + "\n", encoding="utf-8"
        )
        return summary

    async def publish(self, summary):
        del summary
        self.events.append("publish")
        if self.fail_publish:
            raise RuntimeError("publisher unavailable")
        return SimpleNamespace(
            to_dict=lambda: {
                "success": True,
                "status": "published",
                "adapter_name": "theta_smoke_step_000001",
                "behavior_policy_version": "qwen35-9b-base-step-0000",
                "candidate_policy_version": "qwen35-9b-smoke-step-0001",
                "new_policy_version": "qwen35-9b-smoke-step-0001",
            }
        )


def _project(tmp_path, *, phase0_accepted=True):
    root, config_path = create_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["name"] = "triviaqa-one-pass-grpo"
    config["data"]["smoke"]["expected_total_tasks"] = 1
    config["grpo"]["expected_rollout_count"] = 2
    config["wandb_binding"] = {
        "required": True,
        "entity": runner.WANDB_ENTITY,
        "project": runner.WANDB_PROJECT,
        "mode": "online",
        "credential_source": "wandb_sdk_standard",
        "require_run_url_before_training_started_status": True,
        "api_key_in_config_allowed": False,
    }
    config["acceptance_gates"] = {
        "strict_current_phase": "phase_0_data_trust",
        "phase_0_accepted": phase0_accepted,
        "phase_1_accepted": True,
        "phase_2_accepted": True,
        "phase_3_accepted": True,
        "phase_4_accepted": True,
        "phase_5_accepted": True,
        "selected_loss_and_token_mask_verified": True,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    task = make_task("triviaqa", 0)
    scope = ((task,), (0, 1), None, None, {"selection": "unit_test"})
    return root, config_path, scope


def test_phase0_false_blocks_wandb_init_and_rollout(tmp_path):
    events = []
    root, config_path, scope = _project(tmp_path, phase0_accepted=False)
    sdk = _SDK(events)
    backend = _Backend(events)
    with patch.object(runner, "validate_smoke_bounds", return_value=None), patch.object(
        runner, "_select_run_scope", return_value=scope
    ), patch.object(runner.asyncio, "to_thread", side_effect=_in_process_to_thread):
        with pytest.raises(
            runner.ConfigurationError,
            match="blocked at Phase 0",
        ):
            asyncio.run(
                runner.run_smoke(
                    config_path,
                    backend=backend,
                    project_root=root,
                    wandb_sdk=sdk,
                    wandb_step_evidence_provider=_EvidenceProvider(events),
                )
            )
    assert "wandb:init" not in events
    assert not any(event.startswith("collect:") for event in events)


def test_online_run_url_and_complete_step_are_recorded(tmp_path):
    events = []
    root, config_path, scope = _project(tmp_path)
    sdk = _SDK(events)
    with patch.object(runner, "validate_smoke_bounds", return_value=None), patch.object(
        runner, "_select_run_scope", return_value=scope
    ), patch.object(runner.asyncio, "to_thread", side_effect=_in_process_to_thread):
        manifest = asyncio.run(
            asyncio.wait_for(
                runner.run_smoke(
                    config_path,
                    backend=_Backend(events),
                    project_root=root,
                    wandb_sdk=sdk,
                    wandb_step_evidence_provider=_EvidenceProvider(events),
                ),
                timeout=10,
            )
        )
    assert events.index("wandb:init") < events.index("collect:0")
    assert events.index("optimizer:step") < events.index("wandb:log")
    assert sdk.init_kwargs["entity"] == runner.WANDB_ENTITY
    assert sdk.init_kwargs["project"] == runner.WANDB_PROJECT
    assert sdk.init_kwargs["mode"] == "online"
    assert manifest["wandb"]["run_url"] == sdk.run.url
    assert manifest["wandb"]["run_id"] == sdk.run.id
    assert sdk.run.logs[0][0]["dataset"] == "triviaqa"
    assert sdk.run.artifacts[0][1] == ["latest"]
    assert sdk.run.finished == [0]


def test_optimizer_step_is_logged_when_publish_fails(tmp_path):
    events = []
    root, config_path, scope = _project(tmp_path)
    sdk = _SDK(events)
    with patch.object(runner, "validate_smoke_bounds", return_value=None), patch.object(
        runner, "_select_run_scope", return_value=scope
    ), patch.object(runner.asyncio, "to_thread", side_effect=_in_process_to_thread):
        with pytest.raises(runner.SmokeRunError, match="publication failed"):
            asyncio.run(
                asyncio.wait_for(
                    runner.run_smoke(
                        config_path,
                        backend=_Backend(events, fail_publish=True),
                        project_root=root,
                        wandb_sdk=sdk,
                        wandb_step_evidence_provider=_EvidenceProvider(
                            events,
                            validation_complete=False,
                        ),
                    ),
                    timeout=10,
                )
            )
    assert events.index("optimizer:step") < events.index("publish")
    assert events.index("publish") < events.index("wandb:log")
    assert sdk.run.logs[0][0]["publish_status"] == "failed"
    assert sdk.run.logs[0][0]["canary_status"] == "not_completed"
    assert sdk.run.logs[0][0]["validation/evaluated"] == 0.0
    assert sdk.run.finished == [1]
