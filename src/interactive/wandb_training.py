"""Fail-closed W&B telemetry for the existing one-pass GRPO transaction.

SkillFlow's released trainer initializes W&B and logs step statistics in
``training/gflownet_trainer.py``.  This module keeps that SDK boundary but
strengthens it for the project's MD acceptance contract: online-only binding,
a required run URL, complete optimizer-step fields, and a recoverable
checkpoint artifact.  It does not implement a loss, rollout policy, trainer,
or credential loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence


WANDB_SCHEMA_VERSION = "flowsteer.wandb.optimizer-step.v1"


class WandbTrainingError(RuntimeError):
    """The mandatory online W&B transaction could not be completed."""


class _WandbRun(Protocol):
    id: str
    url: str

    def log(self, data: Mapping[str, Any], *, step: int, commit: bool) -> Any: ...

    def log_artifact(
        self, artifact: Any, *, aliases: Sequence[str]
    ) -> Any: ...

    def finish(self, exit_code: int = 0) -> Any: ...


class _WandbSDK(Protocol):
    def init(self, **kwargs: Any) -> _WandbRun: ...

    def Artifact(
        self, name: str, *, type: str, metadata: Mapping[str, Any]
    ) -> Any: ...


@dataclass(frozen=True)
class WandbBinding:
    entity: str
    project: str
    run_name: str
    mode: str = "online"
    group: str | None = None
    tags: tuple[str, ...] = ()
    resume: str = "never"
    run_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("entity", "project", "run_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"W&B {name} must be non-empty")
        if self.mode != "online":
            raise ValueError("formal training requires W&B mode='online'")
        if self.resume not in {"never", "must"}:
            raise ValueError("W&B resume must be 'never' or 'must'")
        if self.resume == "must" and (
            not isinstance(self.run_id, str) or not self.run_id.strip()
        ):
            raise ValueError("W&B resume='must' requires a run_id")
        if self.resume == "never" and self.run_id is not None:
            raise ValueError("a new W&B run cannot carry a resume run_id")
        if self.group is not None and not self.group.strip():
            raise ValueError("W&B group must be non-empty when supplied")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("W&B tags must be non-empty strings")


@dataclass(frozen=True)
class OptimizerStepTelemetry:
    global_step: int
    dataset: str
    behavior_policy_version: str
    updated_policy_version: str
    checkpoint_version: str
    terminal_reward: float
    group_reward_mean: float
    group_reward_std: float
    valid_rollouts: int
    grpo_loss: float
    grad_norm: float
    lora_update_l2: float
    train_metrics: Mapping[str, float] = field(default_factory=dict)
    validation_metrics: Mapping[str, float] = field(default_factory=dict)
    gpu_metrics: Mapping[str, float] = field(default_factory=dict)
    step_elapsed_seconds: float = 0.0
    checkpoint_status: str = ""
    publish_status: str = ""
    route_switch_status: str = ""
    canary_status: str = ""

    def __post_init__(self) -> None:
        if type(self.global_step) is not int or self.global_step <= 0:
            raise ValueError("global_step must be a positive integer")
        if type(self.valid_rollouts) is not int or self.valid_rollouts < 0:
            raise ValueError("valid_rollouts must be a non-negative integer")
        for name in (
            "dataset",
            "behavior_policy_version",
            "updated_policy_version",
            "checkpoint_version",
            "checkpoint_status",
            "publish_status",
            "route_switch_status",
            "canary_status",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        scalars = (
            self.terminal_reward,
            self.group_reward_mean,
            self.group_reward_std,
            self.grpo_loss,
            self.grad_norm,
            self.lora_update_l2,
            self.step_elapsed_seconds,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("optimizer-step scalar metrics must be finite")
        if self.group_reward_std < 0:
            raise ValueError("group_reward_std cannot be negative")
        if self.grad_norm < 0 or self.lora_update_l2 < 0:
            raise ValueError("gradient and LoRA update norms cannot be negative")
        if self.step_elapsed_seconds < 0:
            raise ValueError("step_elapsed_seconds cannot be negative")
        for name in ("train_metrics", "validation_metrics", "gpu_metrics"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(
                not isinstance(key, str)
                or not key.strip()
                or not math.isfinite(float(value))
                for key, value in values.items()
            ):
                raise ValueError(f"{name} must contain finite named metrics")

    def to_wandb(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": WANDB_SCHEMA_VERSION,
            "global_step": self.global_step,
            "dataset": self.dataset,
            "behavior_policy_version": self.behavior_policy_version,
            "updated_policy_version": self.updated_policy_version,
            # Keep the conventional singular key as an explicit alias for the
            # policy used by the next rollout; the behavior version remains
            # separately bound above.
            "policy_version": self.updated_policy_version,
            "checkpoint_version": self.checkpoint_version,
            "terminal_reward": float(self.terminal_reward),
            "group_reward_mean": float(self.group_reward_mean),
            "group_reward_std": float(self.group_reward_std),
            "valid_rollouts": self.valid_rollouts,
            "grpo_loss": float(self.grpo_loss),
            "grad_norm": float(self.grad_norm),
            "lora_update_l2": float(self.lora_update_l2),
            "step_elapsed_seconds": float(self.step_elapsed_seconds),
            "checkpoint_status": self.checkpoint_status,
            "publish_status": self.publish_status,
            "route_switch_status": self.route_switch_status,
            "canary_status": self.canary_status,
        }
        for namespace, values in (
            ("train", self.train_metrics),
            ("validation", self.validation_metrics),
            ("gpu", self.gpu_metrics),
        ):
            for key, value in values.items():
                result[f"{namespace}/{key}"] = float(value)
        return result


class WandbTrainingRun:
    """One mandatory online W&B run; credentials remain SDK-owned."""

    def __init__(self, *, sdk: _WandbSDK, binding: WandbBinding) -> None:
        self._sdk = sdk
        self._binding = binding
        self._run: _WandbRun | None = None
        self._url = ""
        self._run_id = ""

    @property
    def url(self) -> str:
        return self._url

    @property
    def run_id(self) -> str:
        return self._run_id

    def start(self, *, run_config: Mapping[str, Any]) -> str:
        if self._run is not None:
            raise WandbTrainingError("W&B run is already initialized")
        if not isinstance(run_config, Mapping):
            raise TypeError("run_config must be a mapping")
        kwargs: MutableMapping[str, Any] = {
            "entity": self._binding.entity,
            "project": self._binding.project,
            "name": self._binding.run_name,
            "mode": self._binding.mode,
            "config": dict(run_config),
            "tags": list(self._binding.tags),
        }
        if self._binding.group is not None:
            kwargs["group"] = self._binding.group
        if self._binding.resume == "must":
            kwargs["id"] = self._binding.run_id
            kwargs["resume"] = "must"
        else:
            kwargs["resume"] = "never"
        try:
            run = self._sdk.init(**kwargs)
        except Exception as exc:
            raise WandbTrainingError("mandatory online W&B initialization failed") from exc
        url = getattr(run, "url", None)
        run_id = getattr(run, "id", None)
        if (
            not isinstance(url, str)
            or not url.strip()
            or not isinstance(run_id, str)
            or not run_id.strip()
        ):
            try:
                run.finish(exit_code=1)
            finally:
                raise WandbTrainingError("W&B run has no URL or persistent run ID")
        self._run = run
        self._url = url.strip()
        self._run_id = run_id.strip()
        return self._url

    def log_optimizer_step(
        self,
        telemetry: OptimizerStepTelemetry,
        *,
        checkpoint_dir: str | Path,
        recovery_metadata: Mapping[str, Any],
        is_best: bool,
    ) -> None:
        if self._run is None:
            raise WandbTrainingError("W&B run is not initialized")
        root = Path(checkpoint_dir)
        if not root.is_dir():
            raise WandbTrainingError("checkpoint directory is not materialized")
        if not isinstance(recovery_metadata, Mapping) or not recovery_metadata:
            raise WandbTrainingError("checkpoint recovery metadata is required")
        required_files = {
            "adapter_config": root / "adapter_config.json",
            "adapter_model": root / "adapter_model.safetensors",
            "policy_receipt": root / "policy_version.json",
            "training_state": root / "training_state.pt",
        }
        missing_files = [
            name for name, path in required_files.items() if not path.is_file()
        ]
        if missing_files:
            raise WandbTrainingError(
                "checkpoint artifact is incomplete: " + ", ".join(missing_files)
            )
        try:
            policy_receipt = json.loads(
                required_files["policy_receipt"].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise WandbTrainingError("checkpoint policy receipt is unreadable") from exc
        if not isinstance(policy_receipt, Mapping):
            raise WandbTrainingError("checkpoint policy receipt must be a mapping")
        expected_receipt = {
            "committed_step": telemetry.global_step,
            "behavior_policy_version": telemetry.behavior_policy_version,
            "updated_policy_version": telemetry.updated_policy_version,
            "checkpoint_version": telemetry.checkpoint_version,
        }
        if any(
            policy_receipt.get(name) != expected
            for name, expected in expected_receipt.items()
        ):
            raise WandbTrainingError(
                "checkpoint policy receipt differs from optimizer-step telemetry"
            )

        reserved = {
            "schema_version",
            "global_step",
            "dataset",
            "behavior_policy_version",
            "updated_policy_version",
            "checkpoint_version",
            "wandb_run_id",
            "wandb_run_url",
            "artifact_name",
            "artifact_reference_latest",
            "artifact_reference_best",
            "artifact_aliases",
        }
        overlap = sorted(reserved.intersection(recovery_metadata))
        if overlap:
            raise WandbTrainingError(
                "recovery metadata cannot override core fields: " + ", ".join(overlap)
            )
        artifact_name = f"{telemetry.dataset}-director-lora"
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
        latest_reference = (
            f"{self._binding.entity}/{self._binding.project}/"
            f"{artifact_name}:latest"
        )
        best_reference = (
            f"{self._binding.entity}/{self._binding.project}/"
            f"{artifact_name}:best"
            if is_best
            else None
        )
        metadata = {
            **dict(recovery_metadata),
            "schema_version": WANDB_SCHEMA_VERSION,
            "global_step": telemetry.global_step,
            "dataset": telemetry.dataset,
            "behavior_policy_version": telemetry.behavior_policy_version,
            "updated_policy_version": telemetry.updated_policy_version,
            "checkpoint_version": telemetry.checkpoint_version,
            "wandb_run_id": self._run_id,
            "wandb_run_url": self._url,
            "artifact_name": artifact_name,
            "artifact_reference_latest": latest_reference,
            "artifact_reference_best": best_reference,
            "artifact_aliases": aliases,
        }
        recovery_path = root / "wandb_recovery.json"
        try:
            recovery_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise WandbTrainingError(
                "W&B recovery metadata could not be materialized"
            ) from exc
        try:
            artifact = self._sdk.Artifact(
                artifact_name,
                type="model",
                metadata=metadata,
            )
            artifact.add_dir(str(root))
            logged = self._run.log_artifact(artifact, aliases=aliases)
            wait = getattr(logged, "wait", None)
            if callable(wait):
                wait()
            step_payload = telemetry.to_wandb()
            step_payload.update(
                {
                    "checkpoint/artifact_name": artifact_name,
                    "checkpoint/artifact_reference_latest": latest_reference,
                    "checkpoint/artifact_reference_best": best_reference,
                    "checkpoint/artifact_aliases": ",".join(aliases),
                    "wandb/run_id": self._run_id,
                    "wandb/run_url": self._url,
                }
            )
            self._run.log(
                step_payload,
                step=telemetry.global_step,
                commit=True,
            )
        except Exception as exc:
            raise WandbTrainingError(
                "mandatory W&B optimizer-step logging failed"
            ) from exc

    def finish(self, *, exit_code: int) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as exc:
            raise WandbTrainingError("W&B finish failed") from exc
        finally:
            self._run = None


__all__ = [
    "OptimizerStepTelemetry",
    "WANDB_SCHEMA_VERSION",
    "WandbBinding",
    "WandbTrainingError",
    "WandbTrainingRun",
]
