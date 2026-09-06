"""Fail-closed Weights & Biases reporting for committed TTB steps.

This is a project monitoring adapter, not part of the SkillFlow algorithm.
SkillFlow supplies the Tempered Trajectory Balance objective and its training
signals; this module only validates an already-produced step receipt and sends
the resulting operational metrics to an online W&B run.  It does not perform
rollout, optimization, checkpointing, adapter publication, or Skill evolution.

W&B is imported lazily so importing this module or running prepare-only checks
cannot initialize the SDK.  A live caller that requests monitoring must obtain
an online run and successfully log every committed step: initialization and
logging failures are raised instead of being downgraded to unmonitored
training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Optional, Sequence


TTB_MONITOR_SCHEMA_VERSION = "flowsteer.agentgraph.ttb-monitor.step.v1"
TTB_MONITOR_PROVENANCE = "project-monitoring-adapter"
_PARAMETER_NAMES = ("theta", "phi", "z")


class TTBMonitoringError(RuntimeError):
    """A required TTB monitoring receipt or W&B operation is invalid."""


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TTBMonitoringError(f"{name} must be a mapping")
    return value


def _required_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TTBMonitoringError(f"{name} must be a boolean")
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TTBMonitoringError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TTBMonitoringError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise TTBMonitoringError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise TTBMonitoringError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise TTBMonitoringError(f"{name} must be at least {minimum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise TTBMonitoringError(f"{name} must be an integer {qualifier}")
    return value


def _non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TTBMonitoringError(f"{name} must be non-empty text")
    return value.strip()


def _optional_path(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_text(value, name)


def _json_events(value: Any, name: str) -> tuple[str, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TTBMonitoringError(f"{name} must be a sequence")
    events = list(value)
    for index, event in enumerate(events):
        if isinstance(event, str):
            _non_empty_text(event, f"{name}[{index}]")
        elif not isinstance(event, Mapping):
            raise TTBMonitoringError(
                f"{name}[{index}] must be non-empty text or a mapping"
            )
        elif not event:
            raise TTBMonitoringError(f"{name}[{index}] must not be empty")
    try:
        encoded = json.dumps(
            events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TTBMonitoringError(f"{name} must be JSON serializable") from exc
    return encoded, len(events)


@dataclass(frozen=True)
class WandbTTBMonitorConfig:
    """Online W&B settings for the project TTB monitoring adapter."""

    project: str
    run_name: str
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "agentgraph-ttb"
    tags: tuple[str, ...] = ("mbppplus", "agentgraph", "ttb", "lora")
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        _non_empty_text(self.project, "wandb project")
        _non_empty_text(self.run_name, "wandb run_name")
        _non_empty_text(self.job_type, "wandb job_type")
        for name, value in (("entity", self.entity), ("group", self.group)):
            if value is not None:
                _non_empty_text(value, f"wandb {name}")
        if self.run_id is not None:
            _non_empty_text(self.run_id, "wandb run_id")
        if not isinstance(self.tags, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.tags
        ):
            raise TTBMonitoringError(
                "wandb tags must be a tuple of non-empty strings"
            )


@dataclass(frozen=True)
class TTBStepMetrics:
    """Validated telemetry for one completed TTB optimizer transaction."""

    step: int
    behavior_theta_version: str
    behavior_phi_version: str
    behavior_z_version: str
    updated_theta_version: str
    updated_phi_version: str
    updated_z_version: str
    ttb_loss: float
    delta_squared_mean: float
    reward_mean: float
    reward_sum: float
    trajectory_length_mean: float
    trajectory_length_max: int
    valid_rollout_count: int
    filtered_rollout_count: int
    gpu_device_count: int
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int
    step_seconds: float
    rollouts_per_second: float
    action_tokens_per_second: float
    error_count: int
    errors_json: str
    theta_grad_norm: float
    theta_update_l2: float
    theta_update_nonzero: bool
    phi_grad_norm: float
    phi_update_l2: float
    phi_update_nonzero: bool
    z_grad_norm: float
    z_update_l2: float
    z_update_nonzero: bool
    publication_sync_success: bool
    publication_canary_success: bool
    published_adapter_name: str
    published_server_weight_version: str
    publication_checkpoint_version: str
    publication_sync_seconds: float
    official_checkpoint_due: bool
    official_checkpoint_saved: bool
    official_checkpoint_path: str | None
    project_recovery_checkpoint_saved: bool
    project_recovery_checkpoint_path: str
    skill_enabled: bool
    skill_phase: str
    skill_event_count: int
    skill_events_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_wandb(self) -> dict[str, Any]:
        """Return stable flat W&B keys for one committed optimizer step."""

        return {
            "monitor/schema_version": TTB_MONITOR_SCHEMA_VERSION,
            "monitor/provenance": TTB_MONITOR_PROVENANCE,
            "step": self.step,
            "policy/theta/behavior_version": self.behavior_theta_version,
            "policy/phi/behavior_version": self.behavior_phi_version,
            "policy/z/behavior_version": self.behavior_z_version,
            "policy/theta/updated_version": self.updated_theta_version,
            "policy/phi/updated_version": self.updated_phi_version,
            "policy/z/updated_version": self.updated_z_version,
            "ttb/loss": self.ttb_loss,
            "ttb/delta_squared_mean": self.delta_squared_mean,
            "rollout/reward_mean": self.reward_mean,
            "rollout/reward_sum": self.reward_sum,
            "rollout/trajectory_length_mean": self.trajectory_length_mean,
            "rollout/trajectory_length_max": self.trajectory_length_max,
            "rollout/valid_count": self.valid_rollout_count,
            "rollout/filtered_count": self.filtered_rollout_count,
            "rollout/total_count": (
                self.valid_rollout_count + self.filtered_rollout_count
            ),
            "gpu/device_count": self.gpu_device_count,
            "gpu/peak_allocated_bytes": self.gpu_peak_allocated_bytes,
            "gpu/peak_reserved_bytes": self.gpu_peak_reserved_bytes,
            "throughput/step_seconds": self.step_seconds,
            "throughput/rollouts_per_second": self.rollouts_per_second,
            "throughput/action_tokens_per_second": self.action_tokens_per_second,
            "errors/count": self.error_count,
            "errors/events_json": self.errors_json,
            "gradient/theta/norm": self.theta_grad_norm,
            "gradient/phi/norm": self.phi_grad_norm,
            "gradient/z/norm": self.z_grad_norm,
            "update/theta/l2": self.theta_update_l2,
            "update/phi/l2": self.phi_update_l2,
            "update/z/l2": self.z_update_l2,
            "update/theta/nonzero": self.theta_update_nonzero,
            "update/phi/nonzero": self.phi_update_nonzero,
            "update/z/nonzero": self.z_update_nonzero,
            "publication/sync_success": self.publication_sync_success,
            "publication/canary_success": self.publication_canary_success,
            "publication/adapter_name": self.published_adapter_name,
            "publication/server_weight_version": (
                self.published_server_weight_version
            ),
            "publication/checkpoint_version": self.publication_checkpoint_version,
            "publication/sync_seconds": self.publication_sync_seconds,
            "checkpoint/official_due": self.official_checkpoint_due,
            "checkpoint/official_saved": self.official_checkpoint_saved,
            "checkpoint/official_path": self.official_checkpoint_path or "",
            "checkpoint/project_recovery_saved": (
                self.project_recovery_checkpoint_saved
            ),
            "checkpoint/project_recovery_path": (
                self.project_recovery_checkpoint_path
            ),
            "skill/enabled": self.skill_enabled,
            "skill/phase": self.skill_phase,
            "skill/event_count": self.skill_event_count,
            "skill/events_json": self.skill_events_json,
        }


def _versions(
    receipt: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    versions = _required_mapping(receipt.get("versions"), "receipt.versions")
    behavior_raw = _required_mapping(
        versions.get("behavior"), "receipt.versions.behavior"
    )
    updated_raw = _required_mapping(
        versions.get("updated"), "receipt.versions.updated"
    )
    behavior = {
        name: _non_empty_text(
            behavior_raw.get(name), f"receipt.versions.behavior.{name}"
        )
        for name in _PARAMETER_NAMES
    }
    updated = {
        name: _non_empty_text(
            updated_raw.get(name), f"receipt.versions.updated.{name}"
        )
        for name in _PARAMETER_NAMES
    }
    for name in _PARAMETER_NAMES:
        if behavior[name] == updated[name]:
            raise TTBMonitoringError(
                f"{name} updated version must differ from its behavior version"
            )
    return behavior, updated


def _update_metrics(
    receipt: Mapping[str, Any],
) -> dict[str, tuple[float, float, bool]]:
    updates = _required_mapping(receipt.get("updates"), "receipt.updates")
    result: dict[str, tuple[float, float, bool]] = {}
    for name in _PARAMETER_NAMES:
        update = _required_mapping(
            updates.get(name), f"receipt.updates.{name}"
        )
        grad_norm = _finite_float(
            update.get("grad_norm"),
            f"receipt.updates.{name}.grad_norm",
            minimum=0.0,
        )
        update_l2 = _finite_float(
            update.get("update_l2"),
            f"receipt.updates.{name}.update_l2",
            minimum=0.0,
        )
        nonzero = _required_bool(
            update.get("nonzero"), f"receipt.updates.{name}.nonzero"
        )
        observed_nonzero = grad_norm > 0.0 and update_l2 > 0.0
        if nonzero != observed_nonzero:
            raise TTBMonitoringError(
                f"receipt.updates.{name}.nonzero disagrees with grad_norm/update_l2"
            )
        result[name] = (grad_norm, update_l2, nonzero)
    return result


def metrics_from_ttb_step_receipt(receipt: Mapping[str, Any]) -> TTBStepMetrics:
    """Validate and flatten one completed TTB step receipt.

    The receipt is expected to be emitted only after optimizer commit, theta
    publication/canary, and the project recovery checkpoint.  This function
    validates reporting evidence; it does not infer missing values from model
    output or filenames.
    """

    if not isinstance(receipt, Mapping):
        raise TTBMonitoringError("TTB step receipt must be a mapping")
    if receipt.get("schema_version") != TTB_MONITOR_SCHEMA_VERSION:
        raise TTBMonitoringError("unsupported TTB monitoring schema_version")
    if receipt.get("status") != "completed":
        raise TTBMonitoringError("TTB step receipt status is not completed")
    step = _integer(receipt.get("step"), "receipt.step", minimum=1)
    behavior, updated = _versions(receipt)

    ttb = _required_mapping(receipt.get("ttb"), "receipt.ttb")
    ttb_loss = _finite_float(
        ttb.get("loss"), "receipt.ttb.loss", minimum=0.0
    )
    delta_squared_mean = _finite_float(
        ttb.get("delta_squared_mean"),
        "receipt.ttb.delta_squared_mean",
        minimum=0.0,
    )

    rollout = _required_mapping(receipt.get("rollout"), "receipt.rollout")
    valid_count = _integer(
        rollout.get("valid_count"), "receipt.rollout.valid_count", minimum=1
    )
    filtered_count = _integer(
        rollout.get("filtered_count"),
        "receipt.rollout.filtered_count",
        minimum=0,
    )
    reward_mean = _finite_float(
        rollout.get("reward_mean"), "receipt.rollout.reward_mean", minimum=0.0
    )
    reward_sum = _finite_float(
        rollout.get("reward_sum"), "receipt.rollout.reward_sum", minimum=0.0
    )
    if reward_mean > 1.0 or reward_sum > valid_count:
        raise TTBMonitoringError("rollout rewards must be in the [0, 1] range")
    if not math.isclose(
        reward_sum,
        reward_mean * valid_count,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise TTBMonitoringError(
            "rollout reward_sum is inconsistent with reward_mean and valid_count"
        )
    trajectory_length_mean = _finite_float(
        rollout.get("trajectory_length_mean"),
        "receipt.rollout.trajectory_length_mean",
        strictly_positive=True,
    )
    trajectory_length_max = _integer(
        rollout.get("trajectory_length_max"),
        "receipt.rollout.trajectory_length_max",
        minimum=1,
    )
    if trajectory_length_mean > trajectory_length_max:
        raise TTBMonitoringError(
            "trajectory_length_mean cannot exceed trajectory_length_max"
        )

    runtime = _required_mapping(receipt.get("runtime"), "receipt.runtime")
    gpu_device_count = _integer(
        runtime.get("gpu_device_count"),
        "receipt.runtime.gpu_device_count",
        minimum=1,
    )
    gpu_peak_allocated_bytes = _integer(
        runtime.get("gpu_peak_allocated_bytes"),
        "receipt.runtime.gpu_peak_allocated_bytes",
        minimum=0,
    )
    gpu_peak_reserved_bytes = _integer(
        runtime.get("gpu_peak_reserved_bytes"),
        "receipt.runtime.gpu_peak_reserved_bytes",
        minimum=0,
    )
    if gpu_peak_reserved_bytes < gpu_peak_allocated_bytes:
        raise TTBMonitoringError(
            "gpu_peak_reserved_bytes cannot be smaller than allocated bytes"
        )
    step_seconds = _finite_float(
        runtime.get("step_seconds"),
        "receipt.runtime.step_seconds",
        strictly_positive=True,
    )
    rollouts_per_second = _finite_float(
        runtime.get("rollouts_per_second"),
        "receipt.runtime.rollouts_per_second",
        strictly_positive=True,
    )
    action_tokens_per_second = _finite_float(
        runtime.get("action_tokens_per_second"),
        "receipt.runtime.action_tokens_per_second",
        strictly_positive=True,
    )
    error_count = _integer(
        runtime.get("error_count"), "receipt.runtime.error_count", minimum=0
    )
    errors_json, observed_error_count = _json_events(
        runtime.get("errors"), "receipt.runtime.errors"
    )
    if error_count != observed_error_count:
        raise TTBMonitoringError(
            "receipt.runtime.error_count does not match receipt.runtime.errors"
        )

    updates = _update_metrics(receipt)

    publication = _required_mapping(
        receipt.get("publication"), "receipt.publication"
    )
    publication_sync_success = _required_bool(
        publication.get("success"), "receipt.publication.success"
    )
    publication_canary_success = _required_bool(
        publication.get("canary_success"),
        "receipt.publication.canary_success",
    )
    if not publication_sync_success or not publication_canary_success:
        raise TTBMonitoringError(
            "completed TTB step requires successful theta publication and canary"
        )
    if publication.get("theta_version") != updated["theta"]:
        raise TTBMonitoringError(
            "published theta version differs from the updated theta version"
        )
    published_adapter_name = _non_empty_text(
        publication.get("adapter_name"), "receipt.publication.adapter_name"
    )
    published_server_weight_version = _non_empty_text(
        publication.get("server_weight_version"),
        "receipt.publication.server_weight_version",
    )
    publication_checkpoint_version = _non_empty_text(
        publication.get("checkpoint_version"),
        "receipt.publication.checkpoint_version",
    )
    publication_sync_seconds = _finite_float(
        publication.get("sync_seconds"),
        "receipt.publication.sync_seconds",
        minimum=0.0,
    )

    checkpoints = _required_mapping(
        receipt.get("checkpoints"), "receipt.checkpoints"
    )
    official = _required_mapping(
        checkpoints.get("official"), "receipt.checkpoints.official"
    )
    official_due = _required_bool(
        official.get("due"), "receipt.checkpoints.official.due"
    )
    official_saved = _required_bool(
        official.get("saved"), "receipt.checkpoints.official.saved"
    )
    official_path = _optional_path(
        official.get("path"), "receipt.checkpoints.official.path"
    )
    if official_due and not official_saved:
        raise TTBMonitoringError("due official checkpoint was not saved")
    if official_saved != (official_path is not None):
        raise TTBMonitoringError(
            "official checkpoint saved/path fields are inconsistent"
        )
    recovery = _required_mapping(
        checkpoints.get("project_recovery"),
        "receipt.checkpoints.project_recovery",
    )
    recovery_saved = _required_bool(
        recovery.get("saved"),
        "receipt.checkpoints.project_recovery.saved",
    )
    if not recovery_saved:
        raise TTBMonitoringError(
            "completed TTB step requires a project recovery checkpoint"
        )
    recovery_path = _non_empty_text(
        recovery.get("path"), "receipt.checkpoints.project_recovery.path"
    )

    skill = _required_mapping(receipt.get("skill"), "receipt.skill")
    skill_enabled = _required_bool(
        skill.get("enabled"), "receipt.skill.enabled"
    )
    skill_phase = _non_empty_text(skill.get("phase"), "receipt.skill.phase")
    skill_events_json, skill_event_count = _json_events(
        skill.get("events"), "receipt.skill.events"
    )

    return TTBStepMetrics(
        step=step,
        behavior_theta_version=behavior["theta"],
        behavior_phi_version=behavior["phi"],
        behavior_z_version=behavior["z"],
        updated_theta_version=updated["theta"],
        updated_phi_version=updated["phi"],
        updated_z_version=updated["z"],
        ttb_loss=ttb_loss,
        delta_squared_mean=delta_squared_mean,
        reward_mean=reward_mean,
        reward_sum=reward_sum,
        trajectory_length_mean=trajectory_length_mean,
        trajectory_length_max=trajectory_length_max,
        valid_rollout_count=valid_count,
        filtered_rollout_count=filtered_count,
        gpu_device_count=gpu_device_count,
        gpu_peak_allocated_bytes=gpu_peak_allocated_bytes,
        gpu_peak_reserved_bytes=gpu_peak_reserved_bytes,
        step_seconds=step_seconds,
        rollouts_per_second=rollouts_per_second,
        action_tokens_per_second=action_tokens_per_second,
        error_count=error_count,
        errors_json=errors_json,
        theta_grad_norm=updates["theta"][0],
        theta_update_l2=updates["theta"][1],
        theta_update_nonzero=updates["theta"][2],
        phi_grad_norm=updates["phi"][0],
        phi_update_l2=updates["phi"][1],
        phi_update_nonzero=updates["phi"][2],
        z_grad_norm=updates["z"][0],
        z_update_l2=updates["z"][1],
        z_update_nonzero=updates["z"][2],
        publication_sync_success=publication_sync_success,
        publication_canary_success=publication_canary_success,
        published_adapter_name=published_adapter_name,
        published_server_weight_version=published_server_weight_version,
        publication_checkpoint_version=publication_checkpoint_version,
        publication_sync_seconds=publication_sync_seconds,
        official_checkpoint_due=official_due,
        official_checkpoint_saved=official_saved,
        official_checkpoint_path=official_path,
        project_recovery_checkpoint_saved=recovery_saved,
        project_recovery_checkpoint_path=recovery_path,
        skill_enabled=skill_enabled,
        skill_phase=skill_phase,
        skill_event_count=skill_event_count,
        skill_events_json=skill_events_json,
    )


class WandbTTBMonitor:
    """Fail-closed online W&B adapter for committed TTB steps."""

    def __init__(self, run: Any) -> None:
        if run is None or not callable(getattr(run, "log", None)):
            raise TTBMonitoringError("wandb.init did not return a usable run")
        _non_empty_text(getattr(run, "id", None), "wandb run id")
        self._run = run

    @property
    def run_id(self) -> str:
        return _non_empty_text(getattr(self._run, "id", None), "wandb run id")

    @classmethod
    def start(
        cls,
        settings: WandbTTBMonitorConfig,
        *,
        run_config: Mapping[str, Any],
        wandb_module: Any = None,
    ) -> "WandbTTBMonitor":
        """Start an online run; missing SDK/authentication has no fallback."""

        if wandb_module is None:
            try:
                import wandb as wandb_module  # type: ignore[no-redef]
            except Exception as exc:  # pragma: no cover - environment dependent
                raise TTBMonitoringError(
                    "Weights & Biases is required for live TTB training"
                ) from exc
        init = getattr(wandb_module, "init", None)
        if not callable(init):
            raise TTBMonitoringError("wandb.init is unavailable")
        kwargs: dict[str, Any] = {
            "project": settings.project,
            "name": settings.run_name,
            "job_type": settings.job_type,
            "tags": list(settings.tags),
            "config": {
                **dict(run_config),
                "monitor_schema_version": TTB_MONITOR_SCHEMA_VERSION,
                "monitor_provenance": TTB_MONITOR_PROVENANCE,
            },
            "mode": "online",
        }
        if settings.entity is not None:
            kwargs["entity"] = settings.entity
        if settings.group is not None:
            kwargs["group"] = settings.group
        if settings.run_id is not None:
            kwargs["id"] = settings.run_id
            kwargs["resume"] = "allow"
        try:
            run = init(**kwargs)
        except Exception as exc:
            raise TTBMonitoringError(
                "wandb.init failed; live TTB training must not continue unmonitored"
            ) from exc
        return cls(run)

    def log_completed_step(
        self, receipt: Mapping[str, Any]
    ) -> TTBStepMetrics:
        metrics = metrics_from_ttb_step_receipt(receipt)
        try:
            self._run.log(metrics.to_wandb(), step=metrics.step, commit=True)
        except Exception as exc:
            raise TTBMonitoringError(
                f"wandb.log failed at TTB optimizer step {metrics.step}"
            ) from exc
        return metrics

    def log_failure(self, *, step: int, error: BaseException) -> None:
        step_value = _integer(step, "step", minimum=0)
        try:
            self._run.log(
                {
                    "monitor/schema_version": TTB_MONITOR_SCHEMA_VERSION,
                    "monitor/provenance": TTB_MONITOR_PROVENANCE,
                    "step": step_value,
                    "errors/count": 1,
                    "errors/fatal": True,
                    "errors/type": type(error).__name__,
                    "errors/message": str(error),
                },
                step=step_value,
                commit=True,
            )
        except Exception as exc:
            raise TTBMonitoringError(
                f"wandb failure logging failed at TTB optimizer step {step_value}"
            ) from exc

    def finish(self, *, exit_code: int) -> None:
        finish = getattr(self._run, "finish", None)
        if not callable(finish):
            raise TTBMonitoringError("wandb run has no finish method")
        try:
            finish(exit_code=int(exit_code))
        except Exception as exc:
            raise TTBMonitoringError("wandb.finish failed") from exc


def ttb_monitor_run_config(
    *,
    dataset: str,
    total_steps: int,
    start_step: int,
    condition_id: str,
) -> dict[str, Any]:
    """Return non-secret metadata identifying the monitored TTB condition."""

    return {
        "dataset": _non_empty_text(dataset, "dataset"),
        "training_objective": "tempered_trajectory_balance",
        "total_optimizer_steps": _integer(
            total_steps, "total_steps", minimum=1
        ),
        "start_optimizer_step": _integer(
            start_step, "start_step", minimum=0
        ),
        "rollout_policy_sync_interval": 1,
        "condition_id": _non_empty_text(condition_id, "condition_id"),
        "monitor_schema_version": TTB_MONITOR_SCHEMA_VERSION,
        "monitor_provenance": TTB_MONITOR_PROVENANCE,
    }


__all__ = [
    "TTB_MONITOR_PROVENANCE",
    "TTB_MONITOR_SCHEMA_VERSION",
    "TTBMonitoringError",
    "TTBStepMetrics",
    "WandbTTBMonitor",
    "WandbTTBMonitorConfig",
    "metrics_from_ttb_step_receipt",
    "ttb_monitor_run_config",
]
