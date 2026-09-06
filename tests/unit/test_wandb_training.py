from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from src.interactive.wandb_training import (
    OptimizerStepTelemetry,
    WandbBinding,
    WandbTrainingError,
    WandbTrainingRun,
)


class _LoggedArtifact:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _Artifact:
    def __init__(self, name: str, *, type: str, metadata: dict) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.directories: list[str] = []

    def add_dir(self, path: str) -> None:
        self.directories.append(path)


class _Run:
    def __init__(
        self,
        url: str = "https://wandb.example/run/one",
        run_id: str = "run-one",
    ) -> None:
        self.url = url
        self.id = run_id
        self.logs: list[tuple[dict, int, bool]] = []
        self.artifacts: list[tuple[_Artifact, list[str], _LoggedArtifact]] = []
        self.finished: list[int] = []

    def log(self, data: dict, *, step: int, commit: bool) -> None:
        self.logs.append((dict(data), step, commit))

    def log_artifact(self, artifact: _Artifact, *, aliases: list[str]):
        logged = _LoggedArtifact()
        self.artifacts.append((artifact, list(aliases), logged))
        return logged

    def finish(self, exit_code: int = 0) -> None:
        self.finished.append(exit_code)


class _SDK:
    def __init__(self, run: _Run | None = None) -> None:
        self.run = run or _Run()
        self.init_kwargs: dict = {}
        self.artifact_values: list[_Artifact] = []

    def init(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        return self.run

    def Artifact(self, name: str, *, type: str, metadata: dict):
        artifact = _Artifact(name, type=type, metadata=dict(metadata))
        self.artifact_values.append(artifact)
        return artifact


def _telemetry() -> OptimizerStepTelemetry:
    return OptimizerStepTelemetry(
        global_step=1,
        dataset="triviaqa",
        behavior_policy_version="policy-step-0000",
        updated_policy_version="policy-step-0001",
        checkpoint_version="checkpoint-step-0001",
        terminal_reward=0.5,
        group_reward_mean=0.5,
        group_reward_std=0.5,
        valid_rollouts=4,
        grpo_loss=0.25,
        grad_norm=1.2,
        lora_update_l2=0.01,
        train_metrics={"em": 50.0},
        validation_metrics={"em": 48.0},
        gpu_metrics={"memory_allocated_bytes": 1024.0},
        step_elapsed_seconds=120.0,
        checkpoint_status="saved",
        publish_status="published",
        route_switch_status="switched",
        canary_status="passed",
    )


def test_online_binding_requires_url_and_never_accepts_offline_mode() -> None:
    with pytest.raises(ValueError, match="online"):
        WandbBinding("entity", "project", "run", mode="offline")

    sdk = _SDK(_Run(url=""))
    tracker = WandbTrainingRun(
        sdk=sdk,
        binding=WandbBinding("entity", "project", "run"),
    )
    with pytest.raises(WandbTrainingError, match="no URL"):
        tracker.start(run_config={"objective": "action_masked_one_pass_grpo"})
    assert sdk.run.finished == [1]


def test_step_logs_required_metrics_and_checkpoint_aliases() -> None:
    sdk = _SDK()
    tracker = WandbTrainingRun(
        sdk=sdk,
        binding=WandbBinding(
            entity="zhanghanwen6660909-dut",
            project="flowsteer-triviaqa",
            run_name="phase-test",
            tags=("triviaqa", "one-pass-grpo"),
        ),
    )
    url = tracker.start(run_config={"objective": "action_masked_one_pass_grpo"})
    assert url == sdk.run.url
    assert sdk.init_kwargs["mode"] == "online"
    assert sdk.init_kwargs["entity"] == "zhanghanwen6660909-dut"
    assert sdk.init_kwargs["project"] == "flowsteer-triviaqa"

    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "training_state.pt").write_bytes(b"state")
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "policy_version.json").write_text(
            "{\n"
            '  "committed_step": 1,\n'
            '  "behavior_policy_version": "policy-step-0000",\n'
            '  "updated_policy_version": "policy-step-0001",\n'
            '  "checkpoint_version": "checkpoint-step-0001"\n'
            "}\n"
        )
        tracker.log_optimizer_step(
            _telemetry(),
            checkpoint_dir=checkpoint,
            recovery_metadata={"training_state": "training_state.pt"},
            is_best=True,
        )
        recovery = checkpoint / "wandb_recovery.json"
        assert recovery.is_file()

    assert sdk.run.logs[0][1:] == (1, True)
    payload = sdk.run.logs[0][0]
    assert payload["dataset"] == "triviaqa"
    assert payload["behavior_policy_version"] == "policy-step-0000"
    assert payload["updated_policy_version"] == "policy-step-0001"
    assert payload["grpo_loss"] == 0.25
    assert payload["train/em"] == 50.0
    artifact, aliases, logged = sdk.run.artifacts[0]
    assert artifact.type == "model"
    assert artifact.name == "triviaqa-director-lora"
    assert aliases == ["latest", "best"]
    assert logged.waited is True


def test_step_logging_fails_closed_without_materialized_checkpoint() -> None:
    sdk = _SDK()
    tracker = WandbTrainingRun(
        sdk=sdk,
        binding=WandbBinding("entity", "project", "run"),
    )
    tracker.start(run_config={})
    with pytest.raises(WandbTrainingError, match="checkpoint directory"):
        tracker.log_optimizer_step(
            _telemetry(),
            checkpoint_dir="/path/that/does/not/exist",
            recovery_metadata={"training_state": "training_state.pt"},
            is_best=False,
        )
