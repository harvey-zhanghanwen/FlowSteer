#!/usr/bin/env python3
"""Run the HotpotQA MD Action-Masked One-Pass GRPO transaction loop.

The learning objective and exact receipt gate come from the existing
``train_agentgraph_smoke.py``/``smoke_trainer.py`` path.  The outer loop reuses
SkillFlow's same-step rollout concurrency, Qwen3.5/PEFT, SGLang, and LoRA
transport.  The strict transaction order—finish the batch, commit one update,
save recoverable state, publish under pause/drain, canary, then admit the next
batch—is MD-required project engineering because SkillFlow prefetches the next
batch before the current optimizer update.  SkillFlow's TTB objective,
backward policy, and partition function are disabled.

MACE, Bayesian posterior updates, and Skill evolution are not part of this
runner.  Their config flags remain disabled until their respective MD phases
are implemented and accepted; their evidence is never added to the GRPO
reward.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlsplit
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the executable smoke runner directly.  The legacy ``scripts`` package
# initializer eagerly imports optional OpenAI dependencies, whereas these
# dependency-light helpers do not require them during prepare-only validation.
_SMOKE_PATH = PROJECT_ROOT / "scripts" / "train_agentgraph_smoke.py"
_SMOKE_SPEC = importlib.util.spec_from_file_location(
    "flowsteer_train_agentgraph_smoke", _SMOKE_PATH
)
assert _SMOKE_SPEC is not None and _SMOKE_SPEC.loader is not None
_SMOKE = importlib.util.module_from_spec(_SMOKE_SPEC)
_SMOKE_SPEC.loader.exec_module(_SMOKE)
LiveSmokeBackend = _SMOKE.LiveSmokeBackend
SmokeBackend = _SMOKE.SmokeBackend
_mapping = _SMOKE._mapping
_resolve = _SMOKE._resolve
_safe_error = _SMOKE._safe_error
_summary_dict = _SMOKE._summary_dict
_write_grpo_groups = _SMOKE._write_grpo_groups
_write_jsonl = _SMOKE._write_jsonl
version_bundle_for = _SMOKE.version_bundle_for
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.agent_action_parser import (
    AgentAction,
    AgentActionParseError,
    AgentActionParser,
    AgentActionType,
)
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.persistence import GraphSnapshotEvent
from src.interactive.records import ProbeRecord, TaskRecord, TrajectoryRecord
from src.interactive.exploration.dynamic_training_runtime import (
    DynamicEpochCloseResult,
    DynamicLedgerBatch,
    DynamicLedgerEpochCoordinator,
    close_dynamic_epoch,
    dynamic_epoch_metrics,
    freeze_initial_epoch,
    select_probe_sites,
)
from src.interactive.exploration.latent_loss import LatentLossConfig
from src.interactive.exploration.ledger_epoch import LedgerEpoch
from src.interactive.exploration.long_training_admission import (
    build_supplemental_long_training_admission,
    load_supplemental_long_training_admission,
    save_supplemental_long_training_admission,
    validate_supplemental_long_training_admission,
)
from src.interactive.exploration.phase_acceptance import run_phase_acceptance
from src.interactive.exploration.role_classifier import (
    ContractRoleRewriter,
    OpenAICompatibleRoleCompletionClient,
    RoleClassifier,
)
from src.interactive.exploration.training_bridge import (
    LEDGER_FEATURE_SCHEMA_VERSION,
)
from src.interactive.skills.runtime_producer import (
    SkillRuntimeProducerConfig,
    SkillRuntimeUpdate,
    produce_skill_runtime_update,
)
from src.interactive.sequential_gpu_runtime import (
    SequentialGpuCycleReceipt,
    SequentialGpuRuntime,
    SequentialGpuRuntimeError,
)
from src.interactive.sglang_manager import SGLangSupervisorManager
from src.interactive.step_transaction import StepPhase, StepTransaction
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import evaluate_task


class HotpotTrainingError(RuntimeError):
    """The sequential training transaction failed closed."""


class TrainingTracker(Protocol):
    run_id: str
    run_url: str

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        ...

    def update_summary(self, values: Mapping[str, Any]) -> None:
        ...

    def log_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        metadata: Mapping[str, Any],
        aliases: Sequence[str],
    ) -> Mapping[str, Any]:
        ...

    def finish(self, *, exit_code: int) -> None:
        ...


class WandbTracker:
    """Fail-closed W&B adapter following SkillFlow's per-step logging boundary."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        run_id: Optional[str] = None,
    ) -> None:
        tracking = _mapping(config["tracking"], "tracking")
        if tracking.get("enabled") is not True or tracking.get("mode") != "online":
            raise ConfigurationError("tracking must be enabled in online mode")
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("wandb is required for HotpotQA training") from exc

        public_config = {
            "objective": "action_masked_one_pass_grpo",
            "dataset": "hotpotqa",
            "target_optimizer_steps": int(
                _mapping(config["experiment"], "experiment")[
                    "target_optimizer_steps"
                ]
            ),
            "tasks_per_step": int(
                _mapping(_mapping(config["data"], "data")["batch"], "data.batch")
                ["tasks_per_step"]
            ),
            "rollouts_per_task": int(
                _mapping(_mapping(config["data"], "data")["batch"], "data.batch")
                ["rollouts_per_task"]
            ),
            "learning_rate": float(
                _mapping(config["grpo"], "grpo")["learning_rate"]
            ),
            "lora_rank": int(
                _mapping(_mapping(config["director"], "director")["lora"], "director.lora")
                ["rank"]
            ),
            "lora_alpha": int(
                _mapping(_mapping(config["director"], "director")["lora"], "director.lora")
                ["alpha"]
            ),
            "source_backup_branch": str(
                _mapping(config["source"], "source")["backup_branch"]
            ),
            "source_backup_commit": str(
                _mapping(config["source"], "source")["backup_commit"]
            ),
        }
        init_kwargs: dict[str, Any] = {
            "project": str(tracking["project"]),
            "name": str(tracking["run_name"]),
            "config": public_config,
            "mode": "online",
            "resume": "allow",
            "tags": list(tracking.get("tags", ())),
        }
        entity = tracking.get("entity")
        if isinstance(entity, str) and entity.strip():
            init_kwargs["entity"] = entity.strip()
        requested_id = run_id or tracking.get("run_id")
        if isinstance(requested_id, str) and requested_id.strip():
            init_kwargs["id"] = requested_id.strip()
        try:
            run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases initialization failed") from exc
        if run is None or not str(getattr(run, "id", "")).strip():
            raise RuntimeError("Weights & Biases returned no active run")
        self._run = run
        self.run_id = str(run.id)
        self.run_url = str(getattr(run, "url", "") or "")
        if tracking.get("require_run_url") is True and not self.run_url.strip():
            try:
                run.finish(exit_code=1)
            finally:
                raise RuntimeError("Weights & Biases returned no run URL")
        self._required_step_fields = frozenset(
            str(name) for name in tracking.get("required_step_fields", ())
        )
        self._wandb = wandb
        artifact_config = _mapping(
            tracking.get("checkpoint_artifact"), "tracking.checkpoint_artifact"
        )
        self._artifact_type = str(artifact_config["type"])

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        missing = self._required_step_fields - set(values)
        if missing:
            raise RuntimeError(
                "Weights & Biases step metrics are incomplete: "
                + ", ".join(sorted(missing))
            )
        try:
            self._run.log(dict(values), step=step, commit=True)
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases step logging failed") from exc

    def update_summary(self, values: Mapping[str, Any]) -> None:
        try:
            self._run.summary.update(dict(values))
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases summary update failed") from exc

    def log_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        metadata: Mapping[str, Any],
        aliases: Sequence[str],
    ) -> Mapping[str, Any]:
        if not checkpoint_dir.is_dir():
            raise RuntimeError("checkpoint artifact directory is absent")
        normalized_aliases = [str(value) for value in aliases]
        if "latest" not in normalized_aliases:
            raise RuntimeError("checkpoint artifact must publish the latest alias")
        try:
            artifact = self._wandb.Artifact(
                name="hotpotqa-director-checkpoint",
                type=self._artifact_type,
                description=(
                    "Recoverable Qwen3.5-9B Director LoRA checkpoint for "
                    "Action-Masked One-Pass GRPO"
                ),
                metadata=dict(metadata),
            )
            artifact.add_dir(str(checkpoint_dir), name="checkpoint")
            logged = self._run.log_artifact(
                artifact,
                aliases=normalized_aliases,
            )
            logged.wait()
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases checkpoint artifact failed") from exc
        return {
            "name": str(getattr(logged, "name", "")),
            "version": str(getattr(logged, "version", "")),
            "aliases": normalized_aliases,
            "checkpoint_dir": str(checkpoint_dir),
        }

    def finish(self, *, exit_code: int) -> None:
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases finalization failed") from exc


BackendFactory = Callable[[Mapping[str, Any], Path], SmokeBackend]
TrackerFactory = Callable[[Mapping[str, Any], Optional[str]], TrainingTracker]


_STOP_REQUESTED = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_stop(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one durable state file using the MD-required project boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _dynamic_ledger_selected(config: Mapping[str, Any]) -> bool:
    boundary = config.get("method_boundary", {})
    return isinstance(boundary, Mapping) and (
        boundary.get("latent_loss_marker") == "LATENT_LOSS_DYNAMIC_LEDGER_V1"
    )


def validate_hotpotqa_training_config(config: Mapping[str, Any]) -> None:
    """Reject objective drift, disabled tracking, or a non-formal step budget."""

    validate_agent_graph_config(config)
    source = _mapping(config.get("source"), "source")
    method_boundary = _mapping(config.get("method_boundary"), "method_boundary")
    compliance = _mapping(config.get("md_compliance"), "md_compliance")
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    batch = _mapping(data.get("batch"), "data.batch")
    director = _mapping(config.get("director"), "director")
    grpo = _mapping(config.get("grpo"), "grpo")
    sync = _mapping(config.get("policy_sync"), "policy_sync")
    tracking = _mapping(config.get("tracking"), "tracking")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")
    dynamic_mode = _dynamic_ledger_selected(config)

    checks = {
        "source.backup_branch": source.get("backup_branch")
        == "backup/hotpotqa-compliant-best-round01-20260906",
        "source.backup_commit": source.get("backup_commit")
        == "740e53ec6ccac635ecbe7f1f379b002bfb2574d1",
        "method_boundary.compliance_marker": method_boundary.get(
            "compliance_marker"
        )
        == "MD_FULL_COMPLIANCE_20260906_V2",
        "method_boundary.decision_status": method_boundary.get("decision_status")
        == "selected",
        "method_boundary.role": method_boundary.get("role")
        == "primary_task_learning_objective",
        "method_boundary.primary_objective": method_boundary.get(
            "primary_objective"
        )
        == "action_masked_one_pass_grpo",
        "method_boundary.ttb_enabled": method_boundary.get("ttb_enabled") is False,
        "method_boundary.mixing_losses_allowed": method_boundary.get(
            "mixing_losses_allowed"
        )
        is False,
        "md_compliance.marker": compliance.get("marker")
        == "MD_FULL_COMPLIANCE_20260906_V2",
        "experiment.phase": experiment.get("phase")
        == (
            "hotpotqa_dynamic_ledger_grpo_training"
            if dynamic_mode
            else "hotpotqa_grpo_training"
        ),
        "experiment.training_enabled": type(experiment.get("training_enabled"))
        is bool,
        "data.enforce_split_isolation": data.get("enforce_split_isolation") is True,
        "data.expected_unique_train_tasks": data.get(
            "expected_unique_train_tasks"
        )
        == 512,
        "data.expected_validation_tasks": data.get("expected_validation_tasks")
        == 128,
        "data.batch.dataset_key": batch.get("dataset_key") == "hotpotqa",
        "data.batch.tasks_per_step": batch.get("tasks_per_step") == 7,
        "data.batch.rollouts_per_task": batch.get("rollouts_per_task") == 4,
        "data.batch.expected_rollouts_per_step": batch.get(
            "expected_rollouts_per_step"
        )
        == 28,
        "data.batch.selection": batch.get("selection")
        == "skillflow_seeded_sample",
        "director.behavior_policy_version": bool(
            str(director.get("behavior_policy_version", "")).strip()
        ),
        "director.behavior_adapter_name": bool(
            str(director.get("behavior_adapter_name", "")).strip()
        ),
        "director.behavior_adapter_checkpoint": bool(
            str(director.get("behavior_adapter_checkpoint", "")).strip()
        ),
        "director.temperature": float(director.get("temperature", -1)) == 1.0,
        "director.top_p": float(director.get("top_p", -1)) == 1.0,
        "director.top_k": director.get("top_k") == -1,
        "grpo.enabled": grpo.get("enabled") is True,
        "grpo.objective": grpo.get("objective") == "action_masked_one_pass",
        "grpo.samples_per_problem": grpo.get("samples_per_problem") == 4,
        "grpo.optimization_passes_per_rollout_batch": grpo.get(
            "optimization_passes_per_rollout_batch"
        )
        == 1,
        "grpo.terminal_task_reward_only": grpo.get("terminal_task_reward_only")
        is True,
        "policy_sync.enabled": sync.get("enabled") is True,
        "policy_sync.post_update_canary_count": sync.get(
            "post_update_canary_count"
        )
        == 1,
        "tracking.enabled": tracking.get("enabled") is True,
        "tracking.mode": tracking.get("mode") == "online",
        "tracking.binding_marker": tracking.get("binding_marker")
        == "WANDB_BINDING_20260906_V1",
        "tracking.entity": tracking.get("entity") == "zhanghanwen6660909-dut",
        "tracking.project": tracking.get("project") == "flowsteer-hotpotqa",
        "tracking.credential_source": tracking.get("credential_source")
        == "wandb_sdk_default",
        "tracking.require_run_url": tracking.get("require_run_url") is True,
        "exploration.enabled": exploration.get("enabled") is dynamic_mode,
        "skills.enabled": skills.get("enabled") is dynamic_mode,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "HotpotQA training config violates fixed method boundaries: "
            + ", ".join(failed)
        )

    target = experiment.get("target_optimizer_steps")
    if type(target) is not int or not 250 <= target <= 300:
        raise ConfigurationError("target_optimizer_steps must be between 250 and 300")
    initial_update_step = experiment.get("initial_update_step")
    if type(initial_update_step) is not int or initial_update_step < 2:
        raise ConfigurationError(
            "initial_update_step must be at least two when continuing Round-01"
        )
    if experiment.get("checkpoint_every_steps") != 10:
        raise ConfigurationError("checkpoint_every_steps must preserve SkillFlow value 10")
    if experiment.get("prefetch_next_step") is not False:
        raise ConfigurationError("formal on-policy training forbids next-step prefetch")
    if grpo.get("max_optimizer_updates_per_step") != 1:
        raise ConfigurationError("each sealed batch must perform exactly one update")
    if grpo.get("scheduler") != "cosine_with_warmup":
        raise ConfigurationError("scheduler must preserve FlowSteer's cosine schedule")
    if grpo.get("warmup_steps") != 0:
        raise ConfigurationError(
            "warmup_steps must be zero so every accepted optimizer step has non-zero LR"
        )
    if float(grpo.get("weight_decay", -1.0)) != 0.01:
        raise ConfigurationError("weight_decay must be 0.01")
    if grpo.get("ttb_enabled") is not False:
        raise ConfigurationError("SkillFlow TTB is outside the MD's GRPO task flow")
    if config.get("mace", {}).get("enabled") is not False:
        raise ConfigurationError("MACE must remain disabled in this task-learning run")
    if config.get("bayesian_posterior", {}).get("enabled") is not False:
        raise ConfigurationError(
            "Bayesian posterior updates must remain disabled in this task-learning run"
        )
    if dynamic_mode:
        dynamic = _mapping(config.get("dynamic_ledger"), "dynamic_ledger")
        posterior = _mapping(dynamic.get("posterior"), "dynamic_ledger.posterior")
        latent = _mapping(dynamic.get("latent_risk"), "dynamic_ledger.latent_risk")
        probe = _mapping(exploration.get("probe"), "exploration.probe")
        audit = _mapping(exploration.get("audit"), "exploration.audit")
        dynamic_checks = {
            "versions.feature_schema": _mapping(
                config.get("versions"), "versions"
            ).get("feature_schema")
            == LEDGER_FEATURE_SCHEMA_VERSION,
            "method_boundary.latent_loss_is_differentiable_objective": (
                method_boundary.get("latent_loss_is_differentiable_objective")
                is False
            ),
            "method_boundary.probe_and_skill_reward_contribution": float(
                method_boundary.get("probe_and_skill_reward_contribution", -1.0)
            )
            == 0.0,
            "dynamic_ledger.enabled": dynamic.get("enabled") is True,
            "dynamic_ledger.differentiable": dynamic.get("differentiable") is False,
            "dynamic_ledger.contributes_to_grpo_reward": (
                dynamic.get("contributes_to_grpo_reward") is False
            ),
            "dynamic_ledger.terminal_binary_outcome": (
                dynamic.get("terminal_binary_outcome") == "exact_match"
            ),
            "dynamic_ledger.posterior.family": (
                posterior.get("family") == "gaussian_conjugate"
            ),
            "dynamic_ledger.posterior.interaction_activation_probes": (
                posterior.get("interaction_activation_probes") == 5
            ),
            "dynamic_ledger.posterior.noise_variance_floor": float(
                posterior.get("noise_variance_floor", -1.0)
            )
            == 0.01,
            "dynamic_ledger.latent_risk.tau": float(latent.get("tau", -1.0))
            == 0.8,
            "dynamic_ledger.latent_risk.delta_min": float(
                latent.get("delta_min", -1.0)
            )
            == 0.05,
            "dynamic_ledger.latent_risk.candidate_top_k": (
                latent.get("candidate_top_k") == 5
            ),
            "dynamic_ledger.latent_risk.posterior_samples": (
                latent.get("posterior_samples") == 200
            ),
            "exploration.method": (
                exploration.get("method") == "dynamic_combination_posterior"
            ),
            "exploration.legacy_mace_enabled": (
                exploration.get("legacy_mace_enabled") is False
            ),
            "exploration.legacy_joint_bayesian_enabled": (
                exploration.get("legacy_joint_bayesian_enabled") is False
            ),
            "exploration.legacy_particle_evsi_enabled": (
                exploration.get("legacy_particle_evsi_enabled") is False
            ),
            "exploration.probe.budget": float(
                probe.get("budget_fraction_of_natural_trajectories", -1.0)
            )
            == 0.10,
            "exploration.probe.repeats": probe.get("repeats_per_arm") == 3,
            "exploration.probe.enters_grpo": probe.get("enters_grpo") is False,
            "exploration.audit.probability": float(audit.get("probability", -1.0))
            == 0.05,
            "exploration.audit.enters_grpo": audit.get("enters_grpo") is False,
            "skills.minimum_discovery_pairs": (
                skills.get("minimum_discovery_pairs") == 10
            ),
            "skills.minimum_confirmation_problems": (
                skills.get("minimum_confirmation_problems") == 20
            ),
            "skills.benjamini_hochberg_fdr": float(
                skills.get("benjamini_hochberg_fdr", -1.0)
            )
            == 0.10,
            "skills.reward_contribution": float(
                skills.get("reward_contribution", -1.0)
            )
            == 0.0,
        }
        failed_dynamic = [
            name for name, valid in dynamic_checks.items() if not valid
        ]
        if failed_dynamic:
            raise ConfigurationError(
                "HotpotQA dynamic-ledger config violates the selected spec: "
                + ", ".join(failed_dynamic)
            )
    required_wandb_fields = {
        "global_step",
        "dataset",
        "policy/behavior_version",
        "policy/updated_version",
        "checkpoint/version",
        "terminal_reward/mean",
        "group_reward/mean",
        "group_reward/std",
        "rollout/valid_count",
        "train/grpo_loss",
        "train/grad_norm",
        "train/lora_update_l2",
        "train/exact_match",
        "train/token_f1",
        "validation/exact_match",
        "validation/token_f1",
        "gpu/memory",
        "timing/step_seconds",
        "checkpoint/saved",
        "policy/publish_success",
        "policy/route_switch_success",
        "policy/canary_success",
    }
    configured_wandb_fields = set(tracking.get("required_step_fields", ()))
    if not required_wandb_fields.issubset(configured_wandb_fields):
        raise ConfigurationError("tracking.required_step_fields is incomplete")
    validation_protocol = _mapping(
        tracking.get("validation_protocol"), "tracking.validation_protocol"
    )
    artifact = _mapping(
        tracking.get("checkpoint_artifact"), "tracking.checkpoint_artifact"
    )
    validation_task_ids = validation_protocol.get("monitor_task_ids")
    if (
        validation_protocol.get("status") != "frozen"
        or validation_protocol.get("split") != "validation"
        or validation_protocol.get("expected_validation_pool_size") != 128
        or validation_protocol.get("monitor_size") != 7
        or validation_protocol.get("rollouts_per_task") != 1
        or validation_protocol.get("used_for_optimization") is not False
        or validation_protocol.get("used_for_posterior") is not False
        or validation_protocol.get("used_for_skill") is not False
        or validation_protocol.get("best_metric") != "validation/token_f1"
        or not isinstance(validation_task_ids, list)
        or len(validation_task_ids) != 7
        or len(set(validation_task_ids)) != 7
        or any(
            not isinstance(task_id, str) or not task_id.startswith("hotpotqa:")
            for task_id in validation_task_ids
        )
    ):
        raise ConfigurationError("per-step validation protocol is not frozen exactly")
    if (
        artifact.get("enabled") is not True
        or artifact.get("type") != "model"
        or artifact.get("aliases") != ["latest", "best"]
        or artifact.get("recovery_metadata_required") is not True
    ):
        raise ConfigurationError("recoverable W&B checkpoint artifact contract differs")
    if experiment.get("training_enabled") is True:
        if validation_protocol.get("status") != "frozen":
            raise ConfigurationError(
                "per-step validation protocol must be frozen before training"
            )
        if artifact.get("status") != "ready":
            raise ConfigurationError(
                "recoverable W&B checkpoint artifact contract is not ready"
            )

    oom = _mapping(_mapping(config["gpu"], "gpu")["oom_policy"], "gpu.oom_policy")
    if tuple(oom.get("micro_batch_schedule", ())) != (4, 2, 1):
        raise ConfigurationError("gpu.oom_policy.micro_batch_schedule must be [4, 2, 1]")


def _base_task_id(task: TaskRecord) -> str:
    sampling = task.metadata.get("sampling", {})
    if isinstance(sampling, Mapping):
        value = sampling.get("base_task_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return task.task_id


def load_hotpotqa_training_pool(path: str | Path) -> tuple[TaskRecord, ...]:
    """Load only train-split HotpotQA tasks and remove cycled base-ID copies."""

    unique: dict[str, TaskRecord] = {}
    for task in iter_task_records(path, expected_split="train"):
        if task.metadata.get("dataset_key") != "hotpotqa":
            continue
        unique.setdefault(_base_task_id(task), task)
    if len(unique) < 7:
        raise ValueError("HotpotQA training pool has fewer than seven unique base tasks")
    return tuple(unique.values())


def load_hotpotqa_validation_monitor(
    path: str | Path,
    *,
    task_ids: Sequence[str],
    expected_pool_size: int,
) -> tuple[TaskRecord, ...]:
    """Load the frozen held-out monitor without exposing it to optimization."""

    pool: dict[str, TaskRecord] = {}
    for task in iter_task_records(path, expected_split="validation"):
        if task.metadata.get("dataset_key") != "hotpotqa":
            continue
        if task.task_id in pool:
            raise ValueError("HotpotQA validation task IDs must be unique")
        pool[task.task_id] = task
    if len(pool) != expected_pool_size:
        raise ValueError(
            "HotpotQA validation pool size differs from the frozen protocol"
        )
    requested = list(task_ids)
    missing = [task_id for task_id in requested if task_id not in pool]
    if missing:
        raise ValueError(
            "frozen validation monitor task is absent: " + ", ".join(missing)
        )
    selected = tuple(pool[task_id] for task_id in requested)
    if any(task.split != "validation" for task in selected):
        raise ValueError("validation monitor tasks must remain held out")
    return selected


def validate_hotpotqa_split_isolation(
    train_pool: Sequence[TaskRecord],
    validation_path: str | Path,
    *,
    expected_train_tasks: int,
    expected_validation_tasks: int,
) -> Mapping[str, Any]:
    """Validate the MD train/validation boundary before constructing runtime."""

    train_task_ids = {task.task_id for task in train_pool}
    train_base_ids = {_base_task_id(task) for task in train_pool}
    validation_tasks = tuple(
        task
        for task in iter_task_records(validation_path, expected_split="validation")
        if task.metadata.get("dataset_key") == "hotpotqa"
    )
    validation_task_ids = {task.task_id for task in validation_tasks}
    validation_base_ids = {_base_task_id(task) for task in validation_tasks}
    if len(train_base_ids) != expected_train_tasks:
        raise ValueError("HotpotQA unique train pool size differs")
    if (
        len(validation_tasks) != expected_validation_tasks
        or len(validation_task_ids) != expected_validation_tasks
        or len(validation_base_ids) != expected_validation_tasks
    ):
        raise ValueError("HotpotQA validation pool is not exactly unique")
    base_overlap = sorted(train_base_ids & validation_base_ids)
    task_overlap = sorted(train_task_ids & validation_task_ids)
    if base_overlap or task_overlap:
        raise ValueError("HotpotQA train/validation task identity overlap")
    return {
        "status": "passed",
        "train_split": "train",
        "validation_split": "validation",
        "unique_train_base_task_ids": len(train_base_ids),
        "unique_validation_base_task_ids": len(validation_base_ids),
        "overlap_count": 0,
        "task_id_overlap_count": 0,
    }


def sample_hotpotqa_tasks(
    pool: Sequence[TaskRecord],
    *,
    optimizer_step: int,
    tasks_per_step: int,
    seed: int,
    seed_offset: int,
) -> tuple[TaskRecord, ...]:
    """Use SkillFlow's per-step seeded sampling rule for unique questions."""

    if type(optimizer_step) is not int or optimizer_step < 1:
        raise ValueError("optimizer_step must be positive")
    if type(tasks_per_step) is not int or tasks_per_step < 1:
        raise ValueError("tasks_per_step must be positive")
    by_base_id = {_base_task_id(task): task for task in pool}
    if len(by_base_id) < tasks_per_step:
        raise ValueError("training pool cannot provide one exact unique-question batch")
    rng = random.Random(seed + optimizer_step + seed_offset * 131)
    return tuple(rng.sample(list(by_base_id.values()), tasks_per_step))


def _training_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config["storage"], "storage")
    experiment = _mapping(config["experiment"], "experiment")
    run_root = _resolve(root, str(experiment["output_dir"]))
    return {
        "root": run_root,
        "manifest": _resolve(root, str(storage["manifest_path"])),
        "state": _resolve(root, str(storage["state_path"])),
        "training_log": _resolve(root, str(storage["training_log_path"])),
    }


def _probe_record_from_mapping(value: Mapping[str, Any]) -> ProbeRecord:
    """Restore one persisted paired probe without accepting derived fields."""

    payload = dict(value)
    payload.pop("paired_effect", None)
    try:
        return ProbeRecord(**payload)
    except (TypeError, ValueError) as error:
        raise HotpotTrainingError("persisted ProbeRecord is malformed") from error


def _read_probe_records(path: Path) -> tuple[ProbeRecord, ...]:
    if not path.is_file():
        return ()
    records: list[ProbeRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HotpotTrainingError(
                    f"invalid ProbeRecord JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise HotpotTrainingError(
                    f"ProbeRecord JSON is not an object at {path}:{line_number}"
                )
            records.append(_probe_record_from_mapping(value))
    identifiers = [record.probe_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise HotpotTrainingError(f"duplicate ProbeRecord IDs in {path}")
    return tuple(records)


def _committed_discovery_probes(
    run_root: Path,
    *,
    completed_steps: int,
) -> tuple[ProbeRecord, ...]:
    """Load only train probes belonging to already committed optimizer steps."""

    probes: list[ProbeRecord] = []
    for step in range(1, completed_steps + 1):
        step_root = run_root / "steps" / f"step_{step:06d}"
        manifest_path = step_root / "step_manifest.json"
        if not manifest_path.is_file():
            raise HotpotTrainingError(
                f"committed step {step} has no step manifest for Skill discovery"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or manifest.get("status") != "committed":
            raise HotpotTrainingError(
                f"step {step} is not committed Skill discovery evidence"
            )
        probe_path = step_root / "probe_records.jsonl"
        step_probes = _read_probe_records(probe_path)
        if any(record.task_split != "train" for record in step_probes):
            raise HotpotTrainingError("Skill discovery evidence is not train-only")
        probes.extend(step_probes)
    identifiers = [record.probe_id for record in probes]
    if len(identifiers) != len(set(identifiers)):
        raise HotpotTrainingError("cumulative Skill discovery has duplicate probe IDs")
    return tuple(probes)


def _skill_runtime_config(config: Mapping[str, Any]) -> SkillRuntimeProducerConfig:
    skills = _mapping(config["skills"], "skills")
    return SkillRuntimeProducerConfig(
        delta_min=float(skills["delta_min"]),
        minimum_discovery_pairs=int(skills["minimum_discovery_pairs"]),
        minimum_confirmation_problems=int(
            skills["minimum_confirmation_problems"]
        ),
        calibration_alpha=float(skills["calibration_alpha"]),
        benjamini_hochberg_fdr=float(skills["benjamini_hochberg_fdr"]),
        maximum_harm_probability=float(skills.get("maximum_harm_probability", 0.05)),
        retire_after_suspended_epochs=int(
            skills.get("retire_after_suspended_epochs", 3)
        ),
    )


def _materialize_skill_runtime_update(
    config: Mapping[str, Any],
    *,
    run_root: Path,
    ledger_epoch: LedgerEpoch,
    completed_steps: int,
) -> tuple[SkillRuntimeUpdate, Path]:
    """Run the MD Skill evidence gate before the next frozen rollout epoch.

    Discovery is restricted to committed train probes.  A policy-bound held-out
    probe file is consumed only when it has been produced independently; its
    absence remains an empty confirmation set and therefore fail-closes any
    discovery-qualified candidate rather than synthesizing validation evidence.
    """

    epoch = ledger_epoch.condition.epoch
    if epoch < 1:
        raise HotpotTrainingError(
            "Skill runtime production requires at least one completed evidence epoch"
        )
    epoch_root = run_root / "skills" / f"epoch_{epoch:06d}"
    confirmation_path = epoch_root / "heldout_probe_records.jsonl"
    discovery = _committed_discovery_probes(
        run_root,
        completed_steps=completed_steps,
    )
    confirmation = _read_probe_records(confirmation_path)
    update = produce_skill_runtime_update(
        ledger_epoch.ledger,
        discovery_probes=discovery,
        confirmation_probes=confirmation,
        publication_versions=ledger_epoch.condition.versions,
        discovery_epoch=epoch - 1,
        current_skills=ledger_epoch.skills,
        config=_skill_runtime_config(config),
    )
    output_path = epoch_root / "runtime_update.json"
    _atomic_write_json(
        output_path,
        {
            "schema_version": "flowsteer.skill-runtime-epoch.v1",
            "condition": ledger_epoch.condition.to_dict(),
            "discovery_probe_ids": [record.probe_id for record in discovery],
            "confirmation_probe_ids": [record.probe_id for record in confirmation],
            "confirmation_probe_path": (
                str(confirmation_path) if confirmation_path.is_file() else None
            ),
            "skills": [skill.to_dict() for skill in update.skills],
            "evidence_records": {
                key: dict(value) for key, value in update.evidence_records.items()
            },
            "receipt": update.receipt.to_dict(),
            "grpo_reward_contribution": 0.0,
            "ttb_enabled": False,
        },
    )
    return update, output_path


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise HotpotTrainingError(f"{name} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HotpotTrainingError(f"{name} is malformed")
    return dict(value)


def _prepare_or_validate_long_training_admission(
    config: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    state: Mapping[str, Any],
) -> Path:
    """Create once, then revalidate, the post-one-step admission receipt.

    The original one-step closure remains immutable.  A frozen copy of its
    step-1 run state is retained because ``run_state.json`` advances after
    every later optimizer transaction.
    """

    run_root = paths["root"]
    admission_root = run_root / "long_training_admission" / "step_000001"
    admission_path = admission_root / "admission_receipt.json"
    frozen_state_path = admission_root / "run_state.json"
    closure_path = run_root / "one_step_closure_receipt.json"
    step_manifest_path = run_root / "steps" / "step_000001" / "step_manifest.json"

    if admission_path.is_file():
        receipt = load_supplemental_long_training_admission(admission_path)
        references = receipt.evidence_references
        closure = _read_json_mapping(
            Path(references["one_step_closure"]), name="one-step closure"
        )
        frozen_state = _read_json_mapping(
            Path(references["run_state"]), name="frozen step-1 run state"
        )
        step_manifest = _read_json_mapping(
            Path(references["step_manifest"]), name="step-1 manifest"
        )
        ledger_receipt = _read_json_mapping(
            Path(references["ledger_epoch_receipt"]), name="ledger epoch receipt"
        )
        skill_payload = _read_json_mapping(
            Path(references["skill_runtime_update"]), name="Skill runtime update"
        )
        skill_receipt = _mapping(
            skill_payload.get("receipt"), "Skill runtime update receipt"
        )
        try:
            validate_supplemental_long_training_admission(
                receipt,
                one_step_closure=closure,
                run_state=frozen_state,
                step_manifest=step_manifest,
                ledger_epoch_receipt=ledger_receipt,
                skill_runtime_receipt=skill_receipt,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HotpotTrainingError(
                "supplemental long-training admission no longer validates"
            ) from error
    else:
        if state.get("optimizer_updates_completed") != 1:
            raise HotpotTrainingError(
                "first supplemental admission must be built at the committed step-1 boundary"
            )
        closure = _read_json_mapping(closure_path, name="one-step closure")
        step_manifest = _read_json_mapping(
            step_manifest_path, name="step-1 manifest"
        )
        ledger_receipt_path = Path(
            str(state.get("ledger_epoch_receipt", ""))
        ).expanduser().resolve()
        ledger_epoch = LedgerEpoch.load(ledger_receipt_path)
        skill_update, skill_update_path = _materialize_skill_runtime_update(
            config,
            run_root=run_root,
            ledger_epoch=ledger_epoch,
            completed_steps=1,
        )
        admission_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(frozen_state_path, dict(state))
        ledger_receipt = _read_json_mapping(
            ledger_receipt_path, name="ledger epoch receipt"
        )
        receipt = build_supplemental_long_training_admission(
            one_step_closure=closure,
            run_state=state,
            step_manifest=step_manifest,
            ledger_epoch_receipt=ledger_receipt,
            skill_runtime_receipt=skill_update.receipt,
            evidence_references={
                "one_step_closure": str(closure_path.resolve()),
                "run_state": str(frozen_state_path.resolve()),
                "step_manifest": str(step_manifest_path.resolve()),
                "ledger_epoch_receipt": str(ledger_receipt_path),
                "skill_runtime_update": str(skill_update_path.resolve()),
            },
        )
        save_supplemental_long_training_admission(receipt, admission_path)

    if receipt.long_training_authorized is not True:
        raise HotpotTrainingError(
            "supplemental long-training admission is blocked: "
            + ", ".join(receipt.blockers)
        )
    if receipt.scope != "evidence-producing-training-only":
        raise HotpotTrainingError("supplemental admission has an invalid scope")
    return admission_path


def _validate_dynamic_acceptance_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the immutable Phase-E admission receipt."""

    acceptance = dict(value)
    phase_e = acceptance.get("phase_e", {})
    if (
        not isinstance(phase_e, Mapping)
        or phase_e.get("status") != "passed"
        or acceptance.get("all_required_gates_passed") is not True
        or acceptance.get("optimizer_step_authorized") is not True
        or acceptance.get("optimizer_updates") != 0
        or acceptance.get("ttb_enabled") is not False
    ):
        raise HotpotTrainingError(
            "dynamic Phase-E receipt does not authorize an optimizer step"
        )
    epoch_snapshot = phase_e.get("epoch_snapshot")
    if not isinstance(epoch_snapshot, Mapping):
        raise HotpotTrainingError("dynamic Phase-E receipt has no epoch snapshot")
    return acceptance


def _read_dynamic_acceptance_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HotpotTrainingError("dynamic Phase-E receipt does not exist")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HotpotTrainingError("dynamic Phase-E receipt is malformed")
    return _validate_dynamic_acceptance_receipt(value)


def _dynamic_acceptance_from_state(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], Optional[Path]]:
    """Recover Phase-E admission from committed state without another CLI flag."""

    embedded = state.get("dynamic_phase_e_acceptance")
    receipt_value = state.get("dynamic_phase_e_acceptance_receipt")
    if not isinstance(receipt_value, str) or not receipt_value.strip():
        # Accept the short-lived pre-portability field if one was produced by a
        # partially upgraded run, while always writing the canonical name.
        receipt_value = state.get("dynamic_acceptance_receipt")
    receipt_path = (
        Path(receipt_value).expanduser().resolve()
        if isinstance(receipt_value, str) and receipt_value.strip()
        else None
    )
    if isinstance(embedded, Mapping):
        return _validate_dynamic_acceptance_receipt(embedded), receipt_path
    if receipt_path is not None:
        return _read_dynamic_acceptance_receipt(receipt_path), receipt_path
    raise HotpotTrainingError(
        "dynamic committed state has no recoverable Phase-E receipt"
    )


def _checkpoint_recovery_manifest(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path).expanduser().resolve()
    candidates: tuple[tuple[Path, Path], ...]
    if requested.is_file():
        candidates = ((requested, requested.parent),)
    else:
        candidates = (
            (requested / "recovery_manifest.json", requested),
            (
                requested / "checkpoint" / "recovery_manifest.json",
                requested / "checkpoint",
            ),
        )
    for manifest_path, checkpoint_dir in candidates:
        if manifest_path.is_file():
            return manifest_path, checkpoint_dir
    raise HotpotTrainingError(
        "explicit checkpoint has no dynamic recovery_manifest.json"
    )


def _checkpoint_relative_path(
    checkpoint_dir: Path,
    value: object,
    *,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HotpotTrainingError(f"dynamic recovery manifest has no {field}")
    relative = Path(value)
    if relative.is_absolute():
        raise HotpotTrainingError(
            f"dynamic recovery manifest {field} must be checkpoint-relative"
        )
    resolved = (checkpoint_dir / relative).resolve()
    try:
        resolved.relative_to(checkpoint_dir.resolve())
    except ValueError as error:
        raise HotpotTrainingError(
            f"dynamic recovery manifest {field} escapes the checkpoint"
        ) from error
    return resolved


def _state_from_dynamic_checkpoint(path: str | Path) -> dict[str, Any]:
    """Construct committed runner state from one portable dynamic checkpoint."""

    manifest_path, checkpoint_dir = _checkpoint_recovery_manifest(path)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, Mapping):
        raise HotpotTrainingError("dynamic recovery manifest is malformed")
    manifest = dict(raw_manifest)
    if (
        manifest.get("schema_version") != "flowsteer.dynamic-grpo-recovery.v1"
        or manifest.get("status") != "ready"
        or manifest.get("checkpoint_recoverable") is not True
        or manifest.get("ttb_enabled") is not False
        or manifest.get("objective") != "action_masked_one_pass_grpo"
    ):
        raise HotpotTrainingError("dynamic recovery manifest is not resumable")
    optimizer_step = manifest.get("optimizer_step")
    absolute_update_step = manifest.get("absolute_update_step")
    if type(optimizer_step) is not int or optimizer_step < 1:
        raise HotpotTrainingError("dynamic recovery optimizer_step is invalid")
    if type(absolute_update_step) is not int or absolute_update_step < 1:
        raise HotpotTrainingError("dynamic recovery absolute_update_step is invalid")
    policy_version = str(manifest.get("policy_version", "")).strip()
    adapter_name = str(manifest.get("behavior_adapter_name", "")).strip()
    if not policy_version or not adapter_name:
        raise HotpotTrainingError(
            "dynamic recovery manifest has no policy or adapter route"
        )
    training_state = _checkpoint_relative_path(
        checkpoint_dir,
        manifest.get("training_state"),
        field="training_state",
    )
    ledger_receipt = _checkpoint_relative_path(
        checkpoint_dir,
        manifest.get("ledger_epoch_receipt"),
        field="ledger_epoch_receipt",
    )
    phase_e_receipt = _checkpoint_relative_path(
        checkpoint_dir,
        manifest.get("phase_e_acceptance_receipt"),
        field="phase_e_acceptance_receipt",
    )
    if not training_state.is_file():
        raise HotpotTrainingError("dynamic recovery training state is missing")
    acceptance = _read_dynamic_acceptance_receipt(phase_e_receipt)
    try:
        ledger_epoch = LedgerEpoch.load(ledger_receipt)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise HotpotTrainingError("dynamic recovery ledger epoch is invalid") from error
    condition = manifest.get("condition")
    if not isinstance(condition, Mapping) or dict(condition) != (
        ledger_epoch.condition.to_dict()
    ):
        raise HotpotTrainingError(
            "dynamic recovery ledger condition differs from the manifest"
        )
    if ledger_epoch.condition.versions.policy != policy_version:
        raise HotpotTrainingError(
            "dynamic recovery ledger policy differs from the checkpoint policy"
        )
    best_validation = manifest.get("best_validation_token_f1", -1.0)
    try:
        best_validation_value = float(best_validation)
    except (TypeError, ValueError) as error:
        raise HotpotTrainingError(
            "dynamic recovery best validation metric is invalid"
        ) from error
    return {
        "schema_version": "flowsteer.hotpotqa.grpo_run_state.v1",
        "optimizer_updates_completed": optimizer_step,
        "absolute_update_step": absolute_update_step,
        "behavior_policy_version": policy_version,
        "behavior_adapter_name": adapter_name,
        "behavior_adapter_checkpoint": str(checkpoint_dir),
        "optimizer_state_checkpoint": str(training_state),
        "training_state_checkpoint": str(training_state),
        "best_validation_token_f1": best_validation_value,
        "wandb_run_id": str(manifest.get("wandb_run_id", "")),
        "ledger_epoch_receipt": str(ledger_receipt),
        "ledger_condition_id": ledger_epoch.condition.condition_id,
        "ledger_snapshot_id": ledger_epoch.condition.ledger_snapshot_id,
        "skill_snapshot_id": ledger_epoch.condition.skill_snapshot_id,
        "dynamic_phase_e_acceptance_receipt": str(phase_e_receipt),
        "dynamic_phase_e_acceptance": acceptance,
        "recovery_manifest": str(manifest_path),
        "recovered_from_checkpoint": True,
    }


def _persist_dynamic_epoch_snapshot(
    transition: DynamicEpochCloseResult,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Save an epoch and return a receipt whose nested recovery is persisted."""

    receipt_path = checkpoint_dir / "ledger_epoch" / "receipt.json"
    snapshot = transition.next_epoch.save(receipt_path.parent)
    persisted = dict(transition.receipt.to_dict())
    persisted["next_epoch_snapshot"] = snapshot.to_dict()
    persisted["next_epoch_receipt_path"] = str(receipt_path)
    return snapshot.to_dict(), persisted, receipt_path


def _candidate_policy(config: Mapping[str, Any], training_step: int) -> str:
    prefix = str(
        _mapping(config["director"], "director")["updated_policy_version_prefix"]
    )
    return f"{prefix}{training_step:06d}"


def _single_gpu_sequential_runtime(
    config: Mapping[str, Any],
    backend: SmokeBackend,
) -> Optional[SequentialGpuRuntime]:
    """Build the project single-GPU lifecycle only for its explicit layout.

    SkillFlow does not prescribe this mapping.  It is a resource-constrained
    adapter which preserves its stop/train/restart/LoRA-publication boundary
    while never adopting a service that this run did not start.
    """

    gpu = _mapping(config["gpu"], "gpu")
    if gpu.get("execution_layout", "three_gpu_concurrent") != "single_gpu_sequential":
        return None
    if not isinstance(backend, LiveSmokeBackend):
        factory = getattr(backend, "make_sequential_gpu_runtime", None)
        if not callable(factory):
            raise HotpotTrainingError(
                "single_gpu_sequential backend must expose a managed runtime factory"
            )
        runtime = factory(config)
        if not isinstance(runtime, SequentialGpuRuntime):
            raise HotpotTrainingError("backend returned an invalid sequential GPU runtime")
        return runtime

    director = _mapping(config["director"], "director")
    parsed = urlsplit(str(director["api_base"]))
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        raise HotpotTrainingError("single-GPU Supervisor endpoint must be explicit localhost")
    lora = _mapping(director["lora"], "director.lora")
    supervisor = _mapping(gpu.get("supervisor", {}), "gpu.supervisor")
    manager = SGLangSupervisorManager(
        model_path=str(director["base_model"]),
        tokenizer_path=str(director["tokenizer_path"]),
        host="127.0.0.1",
        port=int(parsed.port),
        api_key=os.environ.get("SGLANG_API_KEY", "EMPTY"),
        gpu_id=int(gpu["supervisor_gpu_id"]),
        max_lora_rank=int(lora["rank"]),
        lora_target_modules=[str(value) for value in lora["target_modules"]],
        max_loras_per_batch=int(supervisor.get("max_loras_per_batch", 1)),
        max_loaded_loras=int(supervisor.get("max_loaded_loras", 2)),
        mem_fraction_static=float(supervisor.get("mem_fraction_static", 0.82)),
        context_length=int(supervisor.get("context_length", director["max_context_tokens"])),
        served_model_name=str(director["served_model_name"]),
        reasoning_parser=str(supervisor.get("reasoning_parser", "qwen3")),
        tool_call_parser=str(supervisor.get("tool_call_parser", "qwen3_coder")),
        schedule_policy=str(supervisor.get("schedule_policy", "lpm")),
        sampling_backend=str(supervisor.get("sampling_backend", "pytorch")),
        enable_multimodal=bool(supervisor.get("enable_multimodal", True)),
        ready_timeout_seconds=float(supervisor.get("ready_timeout_seconds", 600.0)),
    )
    return SequentialGpuRuntime(
        manager=manager,
        rollout_gate=backend.rollout_gate,
        drain_timeout_seconds=float(supervisor.get("drain_timeout_seconds", 300.0)),
        stop_timeout_seconds=float(supervisor.get("stop_timeout_seconds", 60.0)),
    )


def _step_config(
    config: Mapping[str, Any],
    *,
    training_step: int,
    absolute_update_step: int,
    behavior_policy_version: str,
    behavior_adapter_name: str,
    behavior_adapter_checkpoint: str,
    optimizer_state_checkpoint: Optional[str],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value["experiment"]["update_step"] = absolute_update_step
    value["experiment"]["training_step"] = training_step
    value["director"]["behavior_policy_version"] = behavior_policy_version
    value["director"]["updated_policy_version"] = _candidate_policy(
        config, training_step
    )
    value["director"]["behavior_adapter_name"] = behavior_adapter_name
    value["director"]["behavior_adapter_checkpoint"] = behavior_adapter_checkpoint
    value["director"]["optimizer_state_checkpoint"] = optimizer_state_checkpoint
    return value


async def _ensure_behavior_ready(
    backend: SmokeBackend,
    step_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the exact behavior adapter before admitting this step's rollouts."""

    custom = getattr(backend, "ensure_behavior_ready", None)
    if callable(custom):
        value = await custom()
        if not isinstance(value, Mapping) or value.get("success") is not True:
            raise HotpotTrainingError("backend rejected the behavior-policy route")
        return value
    if not isinstance(backend, LiveSmokeBackend):
        # Test/injected backends own their route verification contract.
        return {"success": True, "status": "injected_backend_managed"}

    director = _mapping(step_config["director"], "director")
    receipt = await asyncio.to_thread(
        backend.publisher.ensure_loaded_adapter,
        checkpoint_path=str(director["behavior_adapter_checkpoint"]),
        adapter_name=str(director["behavior_adapter_name"]),
    )
    if receipt.get("success") is not True:
        raise HotpotTrainingError("behavior adapter readiness canary failed")
    return receipt


def _verify_rollout_batch(
    trajectories: Sequence[TrajectoryRecord],
    *,
    expected_count: int,
    behavior_policy: str,
    behavior_adapter: str,
) -> None:
    if len(trajectories) != expected_count:
        raise HotpotTrainingError("rollout count differs from the sealed step plan")
    if {item.versions.policy for item in trajectories} != {behavior_policy}:
        raise HotpotTrainingError("sealed batch mixes logical behavior policies")
    turn_policies = {
        turn.policy_version for item in trajectories for turn in item.turns
    }
    turn_adapters = {turn.policy_adapter for item in trajectories for turn in item.turns}
    if turn_policies != {behavior_policy} or turn_adapters != {behavior_adapter}:
        raise HotpotTrainingError("sealed batch mixes behavior policy routes")
    counts = Counter(item.task.task_id for item in trajectories)
    if set(counts.values()) != {4}:
        raise HotpotTrainingError("each HotpotQA question must have four rollouts")
    group_counts = Counter(item.group_key for item in trajectories)
    if len(group_counts) != len(counts) or set(group_counts.values()) != {4}:
        raise HotpotTrainingError(
            "each HotpotQA question must form one exact same-condition group"
        )
    if len({item.trajectory_id for item in trajectories}) != expected_count:
        raise HotpotTrainingError("sealed batch contains duplicate trajectory IDs")
    if len({item.rollout_id for item in trajectories}) != expected_count:
        raise HotpotTrainingError("sealed batch contains duplicate rollout IDs")
    ineligible = [item.trajectory_id for item in trajectories if not item.grpo_eligible]
    if ineligible:
        raise HotpotTrainingError(
            "sealed batch contains non-GRPO-eligible trajectories: "
            + ", ".join(ineligible[:3])
        )


def _apply_replayed_action(graph: AgentGraph, action: AgentAction) -> None:
    """Replay one parsed Canvas mutation through the existing AgentGraph API."""

    if action.action_type is AgentActionType.ADD_AGENT:
        graph.add_agent(
            AgentNode(
                id=str(action.agent_id),
                model_id=str(action.model_id),
                contract=str(action.contract),
            )
        )
    elif action.action_type is AgentActionType.MODIFY_AGENT:
        graph.modify_agent(
            str(action.agent_id),
            model_id=action.model_id,
            contract=action.contract,
        )
    elif action.action_type is AgentActionType.DELETE_AGENT:
        graph.delete_agent(str(action.agent_id))
    elif action.action_type is AgentActionType.SET_RELATION:
        graph.set_relation(
            str(action.source_id),
            str(action.target_id),
            bool(action.source_to_target),
            bool(action.target_to_source),
        )
    elif action.action_type is AgentActionType.SET_OUTPUT:
        graph.set_output(str(action.agent_id))
    elif action.action_type is not AgentActionType.FINISH:
        raise HotpotTrainingError("unsupported replay action")


def _validate_agent_retry_lineage(
    receipts: object,
    *,
    attempt_count: int,
    request_id: str,
    provider_id: str,
    model_id: str,
    provider_request_id: str,
    trajectory_id: str,
) -> None:
    """Validate FlowSteer's per-attempt provider receipts for one Agent call."""

    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise HotpotTrainingError(
            f"Phase 0 Agent retry lacks per-attempt lineage: {trajectory_id}"
        )
    if len(receipts) != attempt_count:
        raise HotpotTrainingError(
            f"Phase 0 Agent retry count differs from lineage: {trajectory_id}"
        )
    for index, raw_receipt in enumerate(receipts, start=1):
        if not isinstance(raw_receipt, Mapping):
            raise HotpotTrainingError(
                f"Phase 0 Agent retry lineage is malformed: {trajectory_id}"
            )
        latency_ms = raw_receipt.get("latency_ms")
        backoff_seconds = raw_receipt.get("backoff_seconds")
        if (
            raw_receipt.get("attempt") != index
            or raw_receipt.get("request_id") != request_id
            or raw_receipt.get("provider_id") != provider_id
            or raw_receipt.get("model_id") != model_id
            or isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not math.isfinite(float(latency_ms))
            or float(latency_ms) < 0.0
            or isinstance(backoff_seconds, bool)
            or not isinstance(backoff_seconds, (int, float))
            or not math.isfinite(float(backoff_seconds))
            or float(backoff_seconds) < 0.0
        ):
            raise HotpotTrainingError(
                f"Phase 0 Agent retry lineage differs: {trajectory_id}"
            )
        if index < attempt_count:
            if (
                raw_receipt.get("status") != "retryable_failure"
                or raw_receipt.get("retryable") is not True
                or not str(raw_receipt.get("error_type", "")).strip()
                or float(backoff_seconds) <= 0.0
            ):
                raise HotpotTrainingError(
                    f"Phase 0 Agent retry predecessor is invalid: {trajectory_id}"
                )
        elif (
            raw_receipt.get("status") != "completed"
            or raw_receipt.get("retryable") is not False
            or raw_receipt.get("http_status") != 200
            or raw_receipt.get("provider_request_id") != provider_request_id
            or float(backoff_seconds) != 0.0
        ):
            raise HotpotTrainingError(
                f"Phase 0 Agent retry terminal receipt is invalid: {trajectory_id}"
            )


async def validate_phase0_rollout_batch(
    trajectories: Sequence[TrajectoryRecord],
    *,
    expected_task_ids: Sequence[str],
    expected_count: int,
    behavior_policy: str,
    behavior_adapter: str,
    rollouts_per_task: int = 4,
) -> Mapping[str, Any]:
    """Fail closed unless one fresh 7x4 train batch has complete lineage.

    This is the MD Phase-0 admission gate over the existing FlowSteer Canvas,
    AgentGraph, trajectory, execution, and terminal evaluator records.  It is
    validation only: it neither changes rewards nor supplies an optimization
    signal.
    """

    _verify_rollout_batch(
        trajectories,
        expected_count=expected_count,
        behavior_policy=behavior_policy,
        behavior_adapter=behavior_adapter,
    )
    expected_tasks = set(expected_task_ids)
    actual_tasks = {record.task.task_id for record in trajectories}
    if rollouts_per_task < 1:
        raise HotpotTrainingError("rollouts_per_task must be positive")
    if actual_tasks != expected_tasks or not expected_tasks:
        raise HotpotTrainingError("Phase 0 task set differs from the frozen plan")
    if expected_count != len(expected_tasks) * rollouts_per_task:
        raise HotpotTrainingError("Phase 0 expected count differs from the frozen plan")
    if len({record.trajectory_id for record in trajectories}) != expected_count:
        raise HotpotTrainingError("Phase 0 trajectory IDs are not unique")
    if len({record.rollout_id for record in trajectories}) != expected_count:
        raise HotpotTrainingError("Phase 0 rollout IDs are not unique")
    group_counts = Counter(record.group_key for record in trajectories)
    if len(group_counts) != len(expected_tasks) or set(group_counts.values()) != {
        rollouts_per_task
    }:
        raise HotpotTrainingError(
            "Phase 0 requires one exact same-condition group per frozen task"
        )
    if len({record.condition_id for record in trajectories}) != 1:
        raise HotpotTrainingError("Phase 0 batch mixes visible conditions")
    if len({record.versions.fingerprint for record in trajectories}) != 1:
        raise HotpotTrainingError("Phase 0 batch mixes VersionBundle receipts")
    if len(
        {
            turn.server_weight_version
            for record in trajectories
            for turn in record.turns
        }
    ) != 1:
        raise HotpotTrainingError("Phase 0 batch mixes SGLang weight versions")

    parser = AgentActionParser()
    director_request_ids: set[str] = set()
    agent_request_ids: set[str] = set()
    provider_request_ids: set[str] = set()
    execution_ids: set[str] = set()
    turn_ids: set[str] = set()
    evaluator_replays = 0
    execution_count = 0
    communication_count = 0
    action_token_count = 0
    agent_provider_retry_count = 0

    for record in trajectories:
        label = record.trajectory_id
        if not record.grpo_eligible:
            raise HotpotTrainingError(f"Phase 0 trajectory is not GRPO eligible: {label}")
        if record.task.split != "train" or not record.condition_satisfied:
            raise HotpotTrainingError(f"Phase 0 trajectory has invalid condition: {label}")
        if record.forced_probe or record.api_fallback_used or record.manual_repair_used:
            raise HotpotTrainingError(f"Phase 0 trajectory mixes a separate evidence flow: {label}")
        if [turn.round_index for turn in record.turns] != list(range(len(record.turns))):
            raise HotpotTrainingError(f"Phase 0 turn order is not contiguous: {label}")

        graph = AgentGraph()
        previous_snapshot_id: Optional[str] = None
        previous_graph = graph.to_dict()
        last_runtime: Mapping[str, Any] = {}
        last_action: Optional[AgentAction] = None
        for turn in record.turns:
            if turn.turn_id in turn_ids:
                raise HotpotTrainingError("Phase 0 turn IDs are not unique")
            turn_ids.add(turn.turn_id)
            if (
                not turn.receipt_verified
                or turn.reconstructed_context
                or not turn.prompt_token_ids
                or not turn.output_token_ids
                or len(turn.behavior_log_probs) != len(turn.output_token_ids)
                or not all(math.isfinite(float(value)) for value in turn.behavior_log_probs)
                or not 0 <= turn.executed_prefix_tokens <= len(turn.output_token_ids)
            ):
                raise HotpotTrainingError(f"Phase 0 Director receipt is incomplete: {label}")
            if turn.policy_version != behavior_policy or turn.policy_adapter != behavior_adapter:
                raise HotpotTrainingError(f"Phase 0 Director route drifted: {label}")
            if turn.director_attempt_count != 1:
                raise HotpotTrainingError(f"Phase 0 Director retry lacks per-attempt lineage: {label}")
            request_id = (turn.director_request_id or "").strip()
            if not request_id or request_id in director_request_ids:
                raise HotpotTrainingError(f"Phase 0 Director request ID is absent or reused: {label}")
            director_request_ids.add(request_id)

            parsed: Optional[AgentAction]
            try:
                parsed = parser.parse(turn.policy_response)
            except AgentActionParseError:
                parsed = None
                if (
                    turn.action
                    or turn.executed_prefix_tokens != 0
                    or not turn.canvas_feedback.startswith("invalid action:")
                ):
                    raise HotpotTrainingError(
                        f"Phase 0 invalid Director action receipt differs: {label}"
                    )
            else:
                if parsed.to_dict() != dict(turn.action):
                    raise HotpotTrainingError(
                        f"Phase 0 parsed action differs from receipt: {label}"
                    )
                if turn.executed_prefix_tokens <= 0:
                    raise HotpotTrainingError(
                        f"Phase 0 executed Director action has no token mask: {label}"
                    )
                last_action = parsed

            if turn.previous_graph_snapshot_id != previous_snapshot_id:
                raise HotpotTrainingError(f"Phase 0 snapshot predecessor differs: {label}")
            snapshot = GraphSnapshotEvent(
                revision=turn.graph_revision,
                graph=turn.graph_snapshot,
                snapshot_id=turn.graph_snapshot_id,
                previous_snapshot_id=turn.previous_graph_snapshot_id,
            )
            snapshot.verify()
            saved_graph = dict(turn.graph_snapshot)
            if saved_graph.get("revision") != turn.graph_revision:
                raise HotpotTrainingError(f"Phase 0 graph revision receipt differs: {label}")

            accepted = turn.canvas_feedback.startswith("accepted ") or (
                turn.canvas_feedback == "workflow finished"
            )
            candidate = graph.fork()
            mutation_error: Optional[Exception] = None
            if parsed is not None:
                try:
                    _apply_replayed_action(candidate, parsed)
                except (TypeError, ValueError) as exc:
                    mutation_error = exc
            elif accepted:
                mutation_error = AgentActionParseError(
                    "an unparsed action cannot be accepted by the Canvas"
                )
            if accepted:
                if mutation_error is not None:
                    raise HotpotTrainingError(f"Phase 0 accepted action cannot replay: {label}")
                if candidate.to_dict() != saved_graph:
                    raise HotpotTrainingError(f"Phase 0 Canvas replay differs: {label}")
                graph = candidate
            elif saved_graph != previous_graph:
                raise HotpotTrainingError(f"Phase 0 rejected edit changed the Canvas: {label}")

            revision_delta = turn.graph_revision - int(previous_graph["revision"])
            if revision_delta not in {0, 1}:
                raise HotpotTrainingError(f"Phase 0 graph revision is non-local: {label}")
            if revision_delta == 0 and saved_graph != previous_graph:
                raise HotpotTrainingError(f"Phase 0 unchanged revision changed graph: {label}")
            if "execution_error=" in turn.canvas_feedback:
                raise HotpotTrainingError(f"Phase 0 execution failure lacks call receipt: {label}")
            if turn.execution_reused and turn.executions:
                raise HotpotTrainingError(f"Phase 0 reused execution was duplicated: {label}")
            if turn.execution_reused and not turn.runtime_summary:
                raise HotpotTrainingError(f"Phase 0 reused execution has no runtime receipt: {label}")
            if not turn.execution_reused and turn.runtime_summary and not turn.executions:
                raise HotpotTrainingError(f"Phase 0 runtime has no execution receipts: {label}")

            runtime = turn.runtime_summary
            if runtime:
                if (
                    runtime.get("graph_revision") != turn.graph_revision
                    or not isinstance(runtime.get("outputs"), Mapping)
                    or runtime.get("output_agent_id") not in runtime.get("outputs", {})
                    or runtime.get("final_answer")
                    != runtime.get("outputs", {}).get(runtime.get("output_agent_id"))
                ):
                    raise HotpotTrainingError(f"Phase 0 runtime receipt differs: {label}")
                last_runtime = runtime

            node_by_id = {node.id: node for node in graph.nodes}
            edges = {
                edge
                for relation in graph.relations
                for edge in relation.directed_edges()
            }
            for execution in turn.executions:
                execution_count += 1
                if execution.execution_id in execution_ids:
                    raise HotpotTrainingError("Phase 0 execution IDs are not unique")
                execution_ids.add(execution.execution_id)
                if execution.error_type is not None or execution.graph_revision != turn.graph_revision:
                    raise HotpotTrainingError(f"Phase 0 execution receipt is invalid: {label}")
                request = execution.metadata.get("request", {})
                response = execution.metadata.get("response", {})
                if not isinstance(request, Mapping) or not isinstance(response, Mapping):
                    raise HotpotTrainingError(f"Phase 0 Agent receipt is malformed: {label}")
                agent_request_id = str(request.get("request_id", "")).strip()
                provider_request_id = str(response.get("provider_request_id", "")).strip()
                if (
                    not agent_request_id
                    or agent_request_id in agent_request_ids
                    or not provider_request_id
                    or provider_request_id in provider_request_ids
                ):
                    raise HotpotTrainingError(f"Phase 0 Agent request ID is absent or reused: {label}")
                agent_request_ids.add(agent_request_id)
                provider_request_ids.add(provider_request_id)
                node = node_by_id.get(execution.agent_id)
                if (
                    request.get("run_id") != execution.experiment_id
                    or request.get("graph_revision") != turn.graph_revision
                    or request.get("problem") != record.task.question
                    or request.get("provider_id") != execution.provider
                    or not isinstance(request.get("agent"), Mapping)
                    or request.get("agent", {}).get("id") != execution.agent_id
                    or node is None
                    or node.model_id != execution.model_id
                ):
                    raise HotpotTrainingError(f"Phase 0 Agent execution lineage differs: {label}")
                attempt_count = response.get("attempt_count")
                if (
                    isinstance(attempt_count, bool)
                    or not isinstance(attempt_count, int)
                    or attempt_count < 1
                ):
                    raise HotpotTrainingError(
                        f"Phase 0 Agent attempt count is invalid: {label}"
                    )
                retry_receipts = response.get("retry_receipts", ())
                if attempt_count > 1 or retry_receipts:
                    _validate_agent_retry_lineage(
                        retry_receipts,
                        attempt_count=attempt_count,
                        request_id=agent_request_id,
                        provider_id=execution.provider,
                        model_id=execution.model_id,
                        provider_request_id=provider_request_id,
                        trajectory_id=label,
                    )
                agent_provider_retry_count += attempt_count - 1
                upstream = request.get("upstream", ())
                if not isinstance(upstream, Sequence):
                    raise HotpotTrainingError(f"Phase 0 Agent communication is malformed: {label}")
                for message in upstream:
                    if (
                        not isinstance(message, Mapping)
                        or message.get("target_agent_id") != execution.agent_id
                        or (
                            str(message.get("source_agent_id", "")),
                            str(message.get("target_agent_id", "")),
                        )
                        not in edges
                        or not str(message.get("content", ""))
                    ):
                        raise HotpotTrainingError(f"Phase 0 Agent communication lineage differs: {label}")
                    communication_count += 1

            previous_snapshot_id = turn.graph_snapshot_id
            previous_graph = saved_graph
            action_token_count += turn.executed_prefix_tokens

        if record.explicit_finish:
            if last_action is None or last_action.action_type is not AgentActionType.FINISH:
                raise HotpotTrainingError(f"Phase 0 explicit finish has no finish action: {label}")
            if not graph.validate(require_complete=True).valid:
                raise HotpotTrainingError(f"Phase 0 finished graph is incomplete: {label}")
            if not last_runtime or last_runtime.get("final_answer") != record.final_answer:
                raise HotpotTrainingError(f"Phase 0 final answer lacks runtime lineage: {label}")
        elif not record.terminal_failure:
            raise HotpotTrainingError(f"Phase 0 trajectory has invalid terminal semantics: {label}")

        replay = await evaluate_task(record.task, record.final_answer or "")
        if (
            not replay.valid
            or replay.evaluator_version != record.evaluation.evaluator_version
            or replay.evaluator_version != record.versions.evaluator
            or float(replay.reward) != float(record.evaluation.reward)
            or dict(replay.metrics) != dict(record.evaluation.metrics)
        ):
            raise HotpotTrainingError(f"Phase 0 terminal evaluator replay differs: {label}")
        evaluator_replays += 1

    return {
        "schema_version": "flowsteer.hotpotqa.phase0_lineage.v1",
        "status": "passed",
        "task_count": len(actual_tasks),
        "trajectory_count": len(trajectories),
        "exact_group_count": len(group_counts),
        "turn_count": len(turn_ids),
        "action_token_count": action_token_count,
        "agent_execution_count": execution_count,
        "agent_communication_count": communication_count,
        "director_request_count": len(director_request_ids),
        "agent_request_count": len(agent_request_ids),
        "agent_provider_retry_count": agent_provider_retry_count,
        "evaluator_replay_count": evaluator_replays,
        "behavior_policy_version": behavior_policy,
        "behavior_adapter_name": behavior_adapter,
        "condition_id": next(iter({record.condition_id for record in trajectories})),
        "version_fingerprint": next(
            iter({record.versions.fingerprint for record in trajectories})
        ),
        "ttb_enabled": False,
        "optimizer_updates": 0,
        "validated_at": _utc_now(),
    }


def _verify_training_summary(
    summary: Mapping[str, Any],
    *,
    behavior_policy: str,
    candidate_policy: str,
    absolute_update_step: int,
    gradient_worker_count: int = 2,
) -> None:
    required_positive = ("grad_norm", "trainable_update_l2")
    if int(summary.get("optimizer_updates", 0)) != 1:
        raise HotpotTrainingError("sealed GRPO batch produced no optimizer update")
    if summary.get("behavior_policy_version") != behavior_policy:
        raise HotpotTrainingError("trainer behavior-policy version differs")
    if summary.get("updated_policy_version") != candidate_policy:
        raise HotpotTrainingError("trainer candidate-policy version differs")
    if int(summary.get("committed_step", 0)) != absolute_update_step:
        raise HotpotTrainingError("trainer committed-step receipt differs")
    for name in required_positive:
        value = float(summary.get(name, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise HotpotTrainingError(f"trainer reported non-positive {name}")
    checkpoint = Path(str(summary.get("checkpoint_dir", "")))
    if not checkpoint.is_dir():
        raise HotpotTrainingError("trainer checkpoint directory is absent")
    optimizer_state = Path(str(summary.get("optimizer_state_checkpoint", "")))
    if summary.get("optimizer_state_saved") is not True or not optimizer_state.is_file():
        raise HotpotTrainingError("optimizer continuation state was not saved")
    training_state = Path(str(summary.get("training_state_checkpoint", "")))
    required_recovery_flags = (
        "training_state_saved",
        "scheduler_state_saved",
        "rng_state_saved",
        "checkpoint_recoverable",
    )
    if (
        any(summary.get(name) is not True for name in required_recovery_flags)
        or not training_state.is_file()
        or training_state != optimizer_state
    ):
        raise HotpotTrainingError("complete recoverable training state was not saved")
    learning_rate = float(summary.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise HotpotTrainingError("accepted optimizer step used a non-positive LR")
    memory = summary.get("gpu_memory_allocated_mib")
    if (
        not isinstance(memory, Mapping)
        or len(memory) != gradient_worker_count
        or any(float(value) <= 0.0 for value in memory.values())
    ):
        raise HotpotTrainingError(
            "training summary lacks the configured gradient-worker memory evidence"
        )


def _verify_canaries(
    canaries: Sequence[TrajectoryRecord],
    *,
    expected_count: int,
    policy_version: str,
    adapter_name: str,
) -> None:
    valid = len(canaries) == expected_count and all(
        record.versions.policy == policy_version
        and bool(record.turns)
        and all(turn.policy_version == policy_version for turn in record.turns)
        and all(turn.policy_adapter == adapter_name for turn in record.turns)
        for record in canaries
    )
    if not valid:
        raise HotpotTrainingError("post-update canary did not use the new policy route")


def _verify_validation_batch(
    trajectories: Sequence[TrajectoryRecord],
    *,
    expected_task_ids: Sequence[str],
    policy_version: str,
    adapter_name: str,
) -> None:
    if [record.task.task_id for record in trajectories] != list(expected_task_ids):
        raise HotpotTrainingError("validation monitor task order differs")
    if any(record.task.split != "validation" for record in trajectories):
        raise HotpotTrainingError("validation monitor contains a non-validation task")
    if any(record.grpo_eligible for record in trajectories):
        raise HotpotTrainingError("validation monitor entered the GRPO eligibility set")
    valid_route = all(
        record.versions.policy == policy_version
        and bool(record.turns)
        and all(turn.policy_version == policy_version for turn in record.turns)
        and all(turn.policy_adapter == adapter_name for turn in record.turns)
        for record in trajectories
    )
    if not valid_route:
        raise HotpotTrainingError("validation monitor did not use the updated policy route")


def _step_metrics(
    trajectories: Sequence[TrajectoryRecord],
    validation_trajectories: Sequence[TrajectoryRecord],
    summary: Mapping[str, Any],
    *,
    training_step: int,
    absolute_update_step: int,
    sync: Mapping[str, Any],
    canaries: Sequence[TrajectoryRecord],
    step_seconds: float,
    dynamic_metrics: Optional[Mapping[str, Any]] = None,
    next_ledger_epoch: Optional[LedgerEpoch] = None,
) -> dict[str, Any]:
    rewards = [
        float(record.evaluation.reward)
        for record in trajectories
        if record.evaluation.valid and record.evaluation.reward is not None
    ]
    exact = [
        float(record.evaluation.metrics["exact_match"])
        for record in trajectories
        if "exact_match" in record.evaluation.metrics
    ]
    f1 = [
        float(record.evaluation.metrics["token_f1"])
        for record in trajectories
        if "token_f1" in record.evaluation.metrics
    ]
    validation_exact = [
        float(record.evaluation.metrics["exact_match"])
        for record in validation_trajectories
        if record.evaluation.valid
        and "exact_match" in record.evaluation.metrics
    ]
    validation_f1 = [
        float(record.evaluation.metrics["token_f1"])
        for record in validation_trajectories
        if record.evaluation.valid
        and "token_f1" in record.evaluation.metrics
    ]
    grouped_rewards: dict[str, list[float]] = {}
    for record in trajectories:
        if record.evaluation.valid and record.evaluation.reward is not None:
            grouped_rewards.setdefault(record.task.task_id, []).append(
                float(record.evaluation.reward)
            )
    group_means = [
        sum(values) / len(values) for values in grouped_rewards.values() if values
    ]
    group_mean = sum(group_means) / len(group_means) if group_means else 0.0
    group_variance = (
        sum((value - group_mean) ** 2 for value in group_means) / len(group_means)
        if group_means
        else 0.0
    )
    memory_by_device = summary.get("gpu_memory_allocated_mib", {})
    if not isinstance(memory_by_device, Mapping):
        memory_by_device = {}
    gpu_memory = sum(float(value) for value in memory_by_device.values())
    checkpoint_version = str(
        sync.get("checkpoint_version")
        or f"checkpoint:{summary['updated_policy_version']}"
    )
    values: dict[str, Any] = {
        "global_step": training_step,
        "dataset": "hotpotqa",
        "optimizer_step": training_step,
        "absolute_update_step": absolute_update_step,
        "checkpoint/version": checkpoint_version,
        "terminal_reward/mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "group_reward/mean": group_mean,
        "group_reward/std": math.sqrt(group_variance),
        "rollout/valid_count": len(rewards),
        "reward/mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward/count": len(rewards),
        "hotpotqa/exact_match": sum(exact) / len(exact) if exact else 0.0,
        "hotpotqa/token_f1": sum(f1) / len(f1) if f1 else 0.0,
        "hotpotqa/terminal_failures": sum(
            int(record.terminal_failure) for record in trajectories
        ),
        "train/grpo_loss": float(summary["loss"]),
        "train/exact_match": sum(exact) / len(exact) if exact else 0.0,
        "train/token_f1": sum(f1) / len(f1) if f1 else 0.0,
        "validation/exact_match": (
            sum(validation_exact) / len(validation_exact)
            if validation_exact
            else 0.0
        ),
        "validation/token_f1": (
            sum(validation_f1) / len(validation_f1)
            if validation_f1
            else 0.0
        ),
        "validation/valid_count": len(validation_exact),
        "validation/terminal_failures": sum(
            int(record.terminal_failure) for record in validation_trajectories
        ),
        "train/grad_norm": float(summary["grad_norm"]),
        "train/lora_update_l2": float(summary["trainable_update_l2"]),
        "train/learning_rate": float(summary["learning_rate"]),
        "train/informative_groups": int(summary["informative_groups"]),
        "train/trained_trajectories": int(summary["trained_trajectories"]),
        "train/micro_batch_size": int(summary["micro_batch_size_used"]),
        "train/oom_backoff_count": int(summary["oom_backoff_count"]),
        "policy/behavior_version": str(summary["behavior_policy_version"]),
        "policy/updated_version": str(summary["updated_policy_version"]),
        "policy/adapter_name": str(sync["adapter_name"]),
        "policy/sync_success": bool(sync["success"]),
        "policy/publish_success": bool(sync["success"]),
        "policy/route_switch_success": bool(sync["route_switch_success"]),
        "policy/canary_success": bool(canaries),
        "gpu/memory": gpu_memory,
        "gpu/memory_by_device_mib": dict(memory_by_device),
        "timing/step_seconds": float(step_seconds),
        "checkpoint/path": str(summary["checkpoint_dir"]),
        "checkpoint/optimizer_state": str(summary["optimizer_state_checkpoint"]),
        "checkpoint/training_state": str(summary["training_state_checkpoint"]),
        "checkpoint/saved": bool(summary["checkpoint_recoverable"]),
    }
    if dynamic_metrics is not None:
        if next_ledger_epoch is None:
            raise HotpotTrainingError(
                "dynamic step metrics require the frozen next ledger epoch"
            )
        skill_counts = {
            "candidate": 0,
            "active": 0,
            "suspended": 0,
            "retired": 0,
        }
        for skill in next_ledger_epoch.skills:
            skill_counts[skill.status.value] += 1
        values.update(
            {
                "ledger/probe_count": int(dynamic_metrics["probe_count"]),
                "ledger/audit_count": int(dynamic_metrics["audit_count"]),
                # The ordinary seven-question validation monitor is not a
                # held-out paired calibration set.  Preserve the required W&B
                # fields as explicit missing values until that independent
                # evidence exists; never substitute or fabricate zeros.
                "ledger/heldout_nll": None,
                "ledger/heldout_brier": None,
                "ledger/interval_coverage_90": None,
                "ledger/interval_width_mean": None,
                "ledger/heldout_status": "pending_independent_paired_evidence",
                "latent_risk/warning_precision": float(
                    dynamic_metrics["warning_precision"]
                ),
                "latent_risk/warning_recall": float(
                    dynamic_metrics["warning_recall"]
                ),
                "ledger/decision_key_explained_variance": float(
                    dynamic_metrics["decision_key_explained_variance"]
                ),
                "skill/candidate_count": skill_counts["candidate"],
                "skill/active_count": skill_counts["active"],
                "skill/suspended_count": skill_counts["suspended"],
                "skill/retired_count": skill_counts["retired"],
            }
        )
    return values


async def run_hotpotqa_phase0(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    backend_factory: Optional[BackendFactory] = None,
) -> Mapping[str, Any]:
    """Collect and validate one fresh Phase-0 batch without optimization/W&B."""

    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_hotpotqa_training_config(config)
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    batch_config = _mapping(data["batch"], "data.batch")
    director = _mapping(config["director"], "director")
    paths = _training_paths(config, root)
    phase0_parent = paths["root"] / "phase_0"
    phase0_parent.mkdir(parents=True, exist_ok=True)
    epoch = 1
    while (phase0_parent / f"evidence_epoch_{epoch:06d}").exists():
        epoch += 1
    phase0_root = phase0_parent / f"evidence_epoch_{epoch:06d}"
    phase0_root.mkdir()
    manifest_path = phase0_root / "manifest.json"
    selected_path = phase0_root / "selected_tasks.jsonl"
    trajectories_path = phase0_root / "trajectories.jsonl"
    groups_path = phase0_root / "grpo_groups.jsonl"
    receipt_path = phase0_root / "phase0_lineage_receipt.json"

    train_path = _resolve(root, str(data["train_path"]))
    validation_path = _resolve(root, str(data["validation_path"]))
    pool = load_hotpotqa_training_pool(train_path)
    split_receipt = validate_hotpotqa_split_isolation(
        pool,
        validation_path,
        expected_train_tasks=int(data["expected_unique_train_tasks"]),
        expected_validation_tasks=int(data["expected_validation_tasks"]),
    )
    tasks = sample_hotpotqa_tasks(
        pool,
        optimizer_step=1,
        tasks_per_step=int(batch_config["tasks_per_step"]),
        seed=int(experiment["seed"]),
        seed_offset=int(batch_config["seed_offset"]),
    )
    _write_jsonl(
        selected_path,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            for task in tasks
        ],
    )
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.phase0_manifest.v1",
        "status": "collecting",
        "compliance_marker": "MD_FULL_COMPLIANCE_20260906_V2",
        "phase": "phase_0_data_trustworthiness",
        "evidence_epoch": epoch,
        "objective": "action_masked_one_pass",
        "ttb_enabled": False,
        "optimizer_updates": 0,
        "wandb_started": False,
        "behavior_policy_version": str(director["behavior_policy_version"]),
        "behavior_adapter_name": str(director["behavior_adapter_name"]),
        "task_ids": [task.task_id for task in tasks],
        "expected_trajectories": int(batch_config["expected_rollouts_per_step"]),
        "split_isolation": dict(split_receipt),
        "artifacts": {
            "manifest": str(manifest_path),
            "selected_tasks": str(selected_path),
            "trajectories": str(trajectories_path),
            "grpo_groups": str(groups_path),
            "lineage_receipt": str(receipt_path),
        },
        "started_at": _utc_now(),
    }
    _atomic_write_json(manifest_path, manifest)

    step_value = _step_config(
        config,
        training_step=1,
        absolute_update_step=int(experiment["initial_update_step"]),
        behavior_policy_version=str(director["behavior_policy_version"]),
        behavior_adapter_name=str(director["behavior_adapter_name"]),
        behavior_adapter_checkpoint=str(director["behavior_adapter_checkpoint"]),
        optimizer_state_checkpoint=(
            str(director["optimizer_state_checkpoint"])
            if director.get("optimizer_state_checkpoint")
            else None
        ),
    )
    backend = (
        backend_factory(step_value, root)
        if backend_factory is not None
        else LiveSmokeBackend.from_config(
            step_value,
            root,
            evaluation_only=True,
            # Phase-0 evidence epochs are independently recoverable and may
            # intentionally replay the same frozen rollout IDs.
            evidence_root=phase0_root / "evidence",
        )
    )
    try:
        readiness = await _ensure_behavior_ready(backend, step_value)
        jobs = []
        rollout_index = 0
        for task in tasks:
            versions = version_bundle_for(
                task,
                policy_version=str(director["behavior_policy_version"]),
                model_catalog_version=backend.model_catalog_version,
                prompt_version=str(_mapping(config["versions"], "versions")["prompt"]),
                tool_version=str(_mapping(config["versions"], "versions")["tool"]),
            )
            for _ in range(int(batch_config["rollouts_per_task"])):
                jobs.append(
                    backend.collect(
                        task,
                        rollout_index,
                        versions,
                        expected_task_split="train",
                    )
                )
                rollout_index += 1
        results = await asyncio.gather(*jobs, return_exceptions=True)
        trajectories = tuple(
            item for item in results if isinstance(item, TrajectoryRecord)
        )
        _write_jsonl(trajectories_path, trajectories)
        failures = [item for item in results if isinstance(item, BaseException)]
        if failures:
            manifest.update(
                status="incomplete",
                completed_trajectories=len(trajectories),
                failed_trajectories=len(failures),
                failure_types=sorted({type(item).__name__ for item in failures}),
                completed_at=_utc_now(),
            )
            _atomic_write_json(manifest_path, manifest)
            raise HotpotTrainingError(
                "Phase 0 rollout collection is incomplete; successful receipts were preserved"
            )
        _write_grpo_groups(groups_path, trajectories)
        receipt = await validate_phase0_rollout_batch(
            trajectories,
            expected_task_ids=[task.task_id for task in tasks],
            expected_count=int(batch_config["expected_rollouts_per_step"]),
            behavior_policy=str(director["behavior_policy_version"]),
            behavior_adapter=str(director["behavior_adapter_name"]),
        )
        _atomic_write_json(receipt_path, receipt)
        manifest.update(
            status="passed",
            completed_trajectories=len(trajectories),
            failed_trajectories=0,
            behavior_readiness=dict(readiness),
            lineage_receipt=dict(receipt),
            completed_at=_utc_now(),
        )
        _atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        if manifest.get("status") != "incomplete":
            manifest.update(
                status="failed",
                error_type=type(exc).__name__,
                completed_at=_utc_now(),
            )
            _atomic_write_json(manifest_path, manifest)
        if isinstance(exc, HotpotTrainingError):
            raise
        raise HotpotTrainingError("HotpotQA Phase 0 failed") from exc


def _dynamic_role_boundaries(
    config: Mapping[str, Any],
    *,
    adapter_name: Optional[str] = None,
) -> tuple[RoleClassifier, ContractRoleRewriter]:
    """Bind role classification/rewrite to the frozen behavior adapter."""

    director = _mapping(config["director"], "director")
    route_adapter = (
        str(adapter_name).strip()
        if adapter_name is not None
        else str(director["behavior_adapter_name"]).strip()
    )
    if not route_adapter:
        raise HotpotTrainingError("dynamic role boundary requires a LoRA adapter route")
    api_key_env = "SGLANG_API_KEY" if os.environ.get("SGLANG_API_KEY") else None
    completion = OpenAICompatibleRoleCompletionClient(
        base_url=str(director["api_base"]),
        model=(
            f"{director['served_model_name']}:"
            f"{route_adapter}"
        ),
        api_key_env=api_key_env,
        timeout_seconds=60.0,
        max_retries=2,
        max_tokens=96,
        disable_thinking=True,
        request_json_object=True,
    )
    return RoleClassifier(completion), ContractRoleRewriter(completion)


def _backend_model_ids(backend: SmokeBackend) -> tuple[str, ...]:
    registry = getattr(backend, "registry", None)
    raw = getattr(registry, "model_ids", None)
    if raw is None:
        raw = getattr(backend, "model_ids", None)
    if not isinstance(raw, (tuple, list)):
        raise HotpotTrainingError("dynamic ledger backend has no frozen model catalog")
    values = tuple(dict.fromkeys(str(value).strip() for value in raw))
    if not values or any(not value for value in values):
        raise HotpotTrainingError("dynamic ledger backend model catalog is empty")
    return values


def _dynamic_epoch_coordinator(
    config: Mapping[str, Any],
    backend: SmokeBackend,
    epoch: LedgerEpoch,
    *,
    adapter_name: str,
    seed: int,
) -> DynamicLedgerEpochCoordinator:
    dynamic = _mapping(config["dynamic_ledger"], "dynamic_ledger")
    latent = _mapping(dynamic["latent_risk"], "dynamic_ledger.latent_risk")
    director = _mapping(config["director"], "director")
    model_ids = _backend_model_ids(backend)
    if epoch.condition.versions.model_catalog != backend.model_catalog_version:
        raise HotpotTrainingError(
            "dynamic epoch model catalog differs from the runtime catalog"
        )
    role_classifier, contract_rewriter = _dynamic_role_boundaries(
        config,
        adapter_name=adapter_name,
    )
    return DynamicLedgerEpochCoordinator(
        epoch=epoch,
        role_classifier=role_classifier,
        contract_rewriter=contract_rewriter,
        model_catalog=model_ids,
        model_catalog_version=backend.model_catalog_version,
        max_rounds=int(director["max_rounds"]),
        seed=seed,
        rollout_concurrency=28,
        probe_concurrency=4,
        latent_config=LatentLossConfig(
            tau=float(latent["tau"]),
            delta_min=float(latent["delta_min"]),
            top_k_candidates=int(latent["candidate_top_k"]),
            posterior_samples=int(latent["posterior_samples"]),
            rollback_cost=float(latent["post_revision_cost"]),
        ),
    )


def _next_phase_e_root(run_root: Path) -> Path:
    parent = run_root / "latent_loss_acceptance" / "phase_e"
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (parent / f"attempt_{attempt:06d}").exists():
        attempt += 1
    destination = parent / f"attempt_{attempt:06d}"
    destination.mkdir()
    return destination


async def run_hotpotqa_dynamic_phase_e(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    backend_factory: Optional[BackendFactory] = None,
) -> Mapping[str, Any]:
    """Run the real 50-question x 4-trajectory Phase-E acceptance epoch.

    This gate performs no optimizer update and starts no W&B run.  It uses the
    exact frozen policy/ledger/Skill condition that will be consumed by the
    first natural GRPO batch, then executes the specified paired probes in a
    separate evidence plane.
    """

    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_hotpotqa_training_config(config)
    if not _dynamic_ledger_selected(config):
        raise HotpotTrainingError("Phase E requires the selected dynamic-ledger profile")

    synthetic = run_phase_acceptance(seed=20260907)
    synthetic.assert_ready_for_phase_e()
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    director = _mapping(config["director"], "director")
    dynamic = _mapping(config["dynamic_ledger"], "dynamic_ledger")
    latent = _mapping(dynamic["latent_risk"], "dynamic_ledger.latent_risk")
    exploration = _mapping(config["exploration"], "exploration")
    probe_config = _mapping(exploration["probe"], "exploration.probe")
    audit_config = _mapping(exploration["audit"], "exploration.audit")
    paths = _training_paths(config, root)
    phase_root = _next_phase_e_root(paths["root"])
    manifest_path = phase_root / "manifest.json"
    receipt_path = phase_root / "acceptance_receipt.json"
    natural_path = phase_root / "natural_trajectories.jsonl"
    sidecar_path = phase_root / "natural_ledger_sidecars.jsonl"
    natural_ledger_path = phase_root / "natural_ledger_records.jsonl"
    intervention_path = phase_root / "intervention_trajectories.jsonl"
    intervention_ledger_path = phase_root / "intervention_ledger_records.jsonl"
    probe_path = phase_root / "probe_records.jsonl"
    selected_sites_path = phase_root / "selected_probe_sites.jsonl"
    metrics_path = phase_root / "metrics.json"
    epoch_root = phase_root / "ledger_epoch_000000"

    train_path = _resolve(root, str(data["train_path"]))
    validation_path = _resolve(root, str(data["validation_path"]))
    pool = load_hotpotqa_training_pool(train_path)
    split_receipt = validate_hotpotqa_split_isolation(
        pool,
        validation_path,
        expected_train_tasks=int(data["expected_unique_train_tasks"]),
        expected_validation_tasks=int(data["expected_validation_tasks"]),
    )
    tasks = sample_hotpotqa_tasks(
        pool,
        optimizer_step=1,
        tasks_per_step=50,
        seed=int(experiment["seed"]),
        seed_offset=9173,
    )
    step_value = _step_config(
        config,
        training_step=1,
        absolute_update_step=int(experiment["initial_update_step"]),
        behavior_policy_version=str(director["behavior_policy_version"]),
        behavior_adapter_name=str(director["behavior_adapter_name"]),
        behavior_adapter_checkpoint=str(director["behavior_adapter_checkpoint"]),
        optimizer_state_checkpoint=(
            str(director["optimizer_state_checkpoint"])
            if director.get("optimizer_state_checkpoint")
            else None
        ),
    )
    backend = (
        backend_factory(step_value, root)
        if backend_factory is not None
        else LiveSmokeBackend.from_config(
            step_value,
            root,
            evaluation_only=True,
            # Each Phase-E attempt is an immutable evidence epoch.  Reusing
            # the global EvidenceStore would collide with a preserved failed
            # attempt when the same frozen rollout IDs are replayed.
            evidence_root=phase_root / "evidence",
        )
    )
    model_ids = _backend_model_ids(backend)
    base_versions = version_bundle_for(
        tasks[0],
        policy_version=str(director["behavior_policy_version"]),
        model_catalog_version=backend.model_catalog_version,
        prompt_version=str(_mapping(config["versions"], "versions")["prompt"]),
        tool_version=str(_mapping(config["versions"], "versions")["tool"]),
    )
    base_versions = replace(
        base_versions,
        feature_schema=LEDGER_FEATURE_SCHEMA_VERSION,
    )
    ledger_epoch = freeze_initial_epoch(
        versions=base_versions,
        model_ids=model_ids,
    )
    epoch_receipt = ledger_epoch.save(epoch_root)
    coordinator = _dynamic_epoch_coordinator(
        step_value,
        backend,
        ledger_epoch,
        adapter_name=str(director["behavior_adapter_name"]),
        seed=int(experiment["seed"]) + 40_000,
    )
    runtime = _single_gpu_sequential_runtime(step_value, backend)
    if runtime is None:
        raise HotpotTrainingError("Phase E requires the configured managed GPU runtime")
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.dynamic_phase_e_manifest.v1",
        "status": "initializing",
        "phase": "E",
        "synthetic_acceptance": synthetic.to_dict(),
        "expected_unique_questions": 50,
        "rollouts_per_question": 4,
        "expected_natural_trajectories": 200,
        "optimizer_updates": 0,
        "wandb_started": False,
        "ttb_enabled": False,
        "grpo_objective": "action_masked_one_pass",
        "split_isolation": dict(split_receipt),
        "task_ids": [task.task_id for task in tasks],
        "epoch_snapshot": epoch_receipt.to_dict(),
        "started_at": _utc_now(),
        "artifacts": {
            "manifest": str(manifest_path),
            "acceptance_receipt": str(receipt_path),
            "natural_trajectories": str(natural_path),
            "natural_ledger_sidecars": str(sidecar_path),
            "natural_ledger_records": str(natural_ledger_path),
            "intervention_trajectories": str(intervention_path),
            "intervention_ledger_records": str(intervention_ledger_path),
            "probe_records": str(probe_path),
            "selected_probe_sites": str(selected_sites_path),
            "metrics": str(metrics_path),
            "ledger_epoch": str(epoch_root / "receipt.json"),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    try:
        await asyncio.to_thread(runtime.start_rollout_service)
        readiness = await _ensure_behavior_ready(backend, step_value)
        manifest.update(status="collecting_natural", behavior_readiness=dict(readiness))
        _atomic_write_json(manifest_path, manifest)
        natural = await coordinator.collect_natural(
            backend,
            tasks,
            rollouts_per_task=4,
            start_rollout_index=4_000_000,
        )
        _verify_rollout_batch(
            natural.natural_trajectories,
            expected_count=200,
            behavior_policy=str(director["behavior_policy_version"]),
            behavior_adapter=str(director["behavior_adapter_name"]),
        )
        lineage = await validate_phase0_rollout_batch(
            natural.natural_trajectories,
            expected_task_ids=[task.task_id for task in tasks],
            expected_count=200,
            behavior_policy=str(director["behavior_policy_version"]),
            behavior_adapter=str(director["behavior_adapter_name"]),
        )
        _write_jsonl(natural_path, natural.natural_trajectories)
        _write_jsonl(sidecar_path, natural.natural_sidecars)
        _write_jsonl(natural_ledger_path, natural.natural_ledger_records)
        selected = select_probe_sites(
            natural.natural_sidecars,
            natural_trajectory_count=len(natural.natural_trajectories),
            seed=int(experiment["seed"]) + 50_000,
            probe_fraction=float(
                probe_config["budget_fraction_of_natural_trajectories"]
            ),
            audit_probability=float(audit_config["probability"]),
            tau=float(latent["tau"]),
        )
        if not selected:
            raise HotpotTrainingError("Phase E produced no eligible paired probe site")
        _write_jsonl(
            selected_sites_path,
            [
                {
                    "schema_version": "flowsteer.selected-probe-site.v1",
                    "audit": value.audit,
                    "sampling_probability": value.sampling_probability,
                    **value.site.to_dict(),
                }
                for value in selected
            ],
        )
        manifest.update(
            status="collecting_interventions",
            completed_natural_trajectories=len(natural.natural_trajectories),
            selected_probe_sites=len(selected),
        )
        _atomic_write_json(manifest_path, manifest)
        batch = await coordinator.collect_selected_probes(
            backend,
            natural,
            selected_sites=selected,
            start_rollout_index=5_000_000,
        )
        _write_jsonl(intervention_path, batch.intervention_trajectories)
        _write_jsonl(intervention_ledger_path, batch.intervention_ledger_records)
        _write_jsonl(probe_path, batch.probe_records)
        metrics = dict(dynamic_epoch_metrics(batch))
        _atomic_write_json(metrics_path, metrics)
        routing = _mapping(metrics["routing_assertions"], "routing_assertions")
        checks = {
            "natural_trajectory_count": metrics["natural_trajectory_count"] == 200,
            "unique_question_count": len(
                {item.task.task_id for item in batch.natural_trajectories}
            )
            == 50,
            "same_condition_groups": len(
                {item.group_key for item in batch.natural_trajectories}
            )
            == 50,
            "natural_only_grpo": routing.get("natural_only_grpo") is True,
            "probe_and_audit_excluded_from_grpo": (
                routing.get("probe_and_audit_excluded_from_grpo") is True
            ),
            "probe_and_audit_excluded_from_standard_metrics": (
                routing.get("probe_and_audit_excluded_from_standard_metrics") is True
            ),
            "paired_probe_present": len(batch.probe_records) > 0,
            "six_branches_per_probe": len(batch.intervention_trajectories)
            == 6 * len(batch.probe_records),
            "diagnostics_finite": all(
                math.isfinite(float(metrics[name]))
                for name in (
                    "decision_key_explained_variance",
                    "warning_precision",
                    "warning_recall",
                )
            ),
        }
        passed = all(checks.values())
        phase_e = {
            "phase": "E",
            "execution_boundary": "real_runner",
            "status": "passed" if passed else "failed",
            "checks": checks,
            "metrics": metrics,
            "lineage_receipt": dict(lineage),
            "epoch_snapshot": epoch_receipt.to_dict(),
        }
        acceptance = {
            "schema_version": "latent-loss-phase-acceptance-runtime-v1",
            "synthetic": synthetic.to_dict(),
            "phase_e": phase_e,
            "all_required_gates_passed": synthetic.synthetic_gates_passed and passed,
            "optimizer_step_authorized": synthetic.synthetic_gates_passed and passed,
            "long_training_authorized": False,
            "optimizer_updates": 0,
            "wandb_started": False,
            "ttb_enabled": False,
            "created_at": _utc_now(),
        }
        _atomic_write_json(receipt_path, acceptance)
        if not passed:
            raise HotpotTrainingError("Phase E acceptance assertions failed")
        manifest.update(
            status="passed",
            completed_natural_trajectories=len(batch.natural_trajectories),
            completed_intervention_trajectories=len(batch.intervention_trajectories),
            completed_probe_records=len(batch.probe_records),
            metrics=metrics,
            acceptance_receipt=str(receipt_path),
            completed_at=_utc_now(),
        )
        _atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            status="failed",
            error=_safe_error(exc),
            failed_at=_utc_now(),
        )
        _atomic_write_json(manifest_path, manifest)
        if isinstance(exc, HotpotTrainingError):
            raise
        raise HotpotTrainingError("HotpotQA dynamic Phase E failed") from exc
    finally:
        try:
            await asyncio.to_thread(runtime.stop_rollout_service)
        except Exception:
            # Preserve the primary failure and keep the runtime's own
            # fail-closed receipt semantics.  It never adopts external jobs.
            pass


async def run_hotpotqa_training(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prepare_only: bool = False,
    allow_md_grpo: bool = False,
    resume: bool = False,
    resume_checkpoint: Optional[str | Path] = None,
    stop_after_optimizer_steps: Optional[int] = None,
    backend_factory: Optional[BackendFactory] = None,
    tracker: Optional[TrainingTracker] = None,
    tracker_factory: Optional[TrackerFactory] = None,
    dynamic_acceptance_receipt: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """Run formal, strictly sequential HotpotQA optimizer transactions."""

    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_hotpotqa_training_config(config)
    dynamic_mode = _dynamic_ledger_selected(config)
    experiment = _mapping(config["experiment"], "experiment")
    compliance = _mapping(config["md_compliance"], "md_compliance")
    gpu = _mapping(config["gpu"], "gpu")
    paths = _training_paths(config, root)
    state: Optional[dict[str, Any]] = None
    if not prepare_only:
        if resume and resume_checkpoint is not None:
            raise ConfigurationError(
                "--resume and --resume-checkpoint are mutually exclusive"
            )
        if resume_checkpoint is not None:
            if not dynamic_mode:
                raise HotpotTrainingError(
                    "--resume-checkpoint currently requires the dynamic-ledger profile"
                )
            state = _state_from_dynamic_checkpoint(resume_checkpoint)
        elif paths["state"].is_file():
            if not resume:
                raise HotpotTrainingError(
                    "a committed run state already exists; pass --resume to continue"
                )
            raw_state = json.loads(paths["state"].read_text(encoding="utf-8"))
            if not isinstance(raw_state, dict):
                raise HotpotTrainingError("committed run state is malformed")
            state = raw_state
        elif resume:
            raise HotpotTrainingError(
                "--resume was requested but no committed state exists"
            )
    acceptance_path: Optional[Path] = None
    dynamic_acceptance: Optional[Mapping[str, Any]] = None
    supplemental_admission_path: Optional[Path] = None
    if not prepare_only:
        if not allow_md_grpo:
            raise HotpotTrainingError(
                "the selected MD GRPO requires explicit --allow-md-grpo authorization"
            )
        if dynamic_mode:
            if dynamic_acceptance_receipt is not None:
                acceptance_path = (
                    Path(dynamic_acceptance_receipt).expanduser().resolve()
                )
                dynamic_acceptance = _read_dynamic_acceptance_receipt(
                    acceptance_path
                )
            elif state is not None:
                dynamic_acceptance, acceptance_path = (
                    _dynamic_acceptance_from_state(state)
                )
            else:
                raise HotpotTrainingError(
                    "initial dynamic training requires an explicit passed Phase-E receipt"
                )
            # The checked-in profile remains fail-closed.  Runtime admission is
            # granted only by this explicit immutable receipt; it is not written
            # back to source configuration.
            config = copy.deepcopy(dict(config))
            config["experiment"]["training_enabled"] = True
            config["gpu"]["training_enabled"] = True
            config["md_compliance"]["real_step_authorized"] = True
            config["latent_loss_compliance"]["all_required_gates_passed"] = True
            config["latent_loss_compliance"]["optimizer_step_authorized"] = True
            config["latent_loss_compliance"]["runtime_acceptance_receipt"] = str(
                acceptance_path or "embedded-in-committed-state"
            )
            experiment = _mapping(config["experiment"], "experiment")
            compliance = _mapping(config["md_compliance"], "md_compliance")
            gpu = _mapping(config["gpu"], "gpu")
        elif experiment.get("training_enabled") is not True or gpu.get(
            "training_enabled"
        ) is not True:
            raise HotpotTrainingError(
                "training is disabled by the current MD compliance gate"
            )
        if compliance.get("phase_0_status") != "passed" or compliance.get(
            "real_step_authorized"
        ) is not True:
            raise HotpotTrainingError(
                "Phase 0 and real-step admission must pass before optimizer execution"
            )
        requested_steps = stop_after_optimizer_steps
        if requested_steps is None or requested_steps > 1:
            if dynamic_mode:
                if state is None:
                    raise HotpotTrainingError(
                        "long training requires a committed step-1 state"
                    )
                supplemental_admission_path = (
                    _prepare_or_validate_long_training_admission(
                        config,
                        paths=paths,
                        state=state,
                    )
                )
            elif (
                compliance.get("one_step_closure_status") != "passed"
                or compliance.get("long_training_authorized") is not True
            ):
                raise HotpotTrainingError(
                    "long training requires a passed real one-step closure"
                )
    paths["root"].mkdir(parents=True, exist_ok=True)
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    batch_config = _mapping(data["batch"], "data.batch")
    director_config = _mapping(config["director"], "director")
    sync_config = _mapping(config["policy_sync"], "policy_sync")
    tracking_config = _mapping(config["tracking"], "tracking")
    validation_protocol = _mapping(
        tracking_config["validation_protocol"], "tracking.validation_protocol"
    )
    target_steps = int(experiment["target_optimizer_steps"])
    if stop_after_optimizer_steps is not None and (
        type(stop_after_optimizer_steps) is not int
        or not 1 <= stop_after_optimizer_steps <= target_steps
    ):
        raise ConfigurationError(
            "stop_after_optimizer_steps must be within the configured target"
        )

    train_path = _resolve(root, str(data["train_path"]))
    validation_path = _resolve(root, str(data["validation_path"]))
    pool = load_hotpotqa_training_pool(train_path)
    split_isolation_receipt = validate_hotpotqa_split_isolation(
        pool,
        validation_path,
        expected_train_tasks=int(data["expected_unique_train_tasks"]),
        expected_validation_tasks=int(data["expected_validation_tasks"]),
    )
    validation_tasks = load_hotpotqa_validation_monitor(
        validation_path,
        task_ids=tuple(validation_protocol["monitor_task_ids"]),
        expected_pool_size=int(
            validation_protocol["expected_validation_pool_size"]
        ),
    )
    first_tasks = sample_hotpotqa_tasks(
        pool,
        optimizer_step=1,
        tasks_per_step=int(batch_config["tasks_per_step"]),
        seed=int(experiment["seed"]),
        seed_offset=int(batch_config["seed_offset"]),
    )
    source = _mapping(config["source"], "source")
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.grpo_training_manifest.v1",
        "status": "prepared" if prepare_only else "initializing",
        "config_path": str(resolved_config),
        "source_backup_branch": str(source["backup_branch"]),
        "source_backup_commit": str(source["backup_commit"]),
        "objective": "action_masked_one_pass",
        "compliance_marker": "MD_FULL_COMPLIANCE_20260906_V2",
        "method_role": "primary_task_learning_objective",
        "ttb_enabled": False,
        "mace_enabled": False,
        "bayesian_posterior_enabled": dynamic_mode,
        "skills_enabled": dynamic_mode,
        "dynamic_combination_posterior_enabled": dynamic_mode,
        "latent_risk_is_differentiable_objective": False,
        "supplemental_long_training_admission": (
            str(supplemental_admission_path)
            if supplemental_admission_path is not None
            else None
        ),
        "target_optimizer_steps": target_steps,
        "optimizer_updates_completed": 0,
        "tasks_per_step": int(batch_config["tasks_per_step"]),
        "rollouts_per_task": int(batch_config["rollouts_per_task"]),
        "rollouts_per_step": int(batch_config["expected_rollouts_per_step"]),
        "prefetch_next_step": False,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "unique_hotpotqa_train_tasks": len(pool),
        "validation_monitor_task_ids": [
            task.task_id for task in validation_tasks
        ],
        "validation_monitor_used_for_optimization": False,
        "split_isolation": dict(split_isolation_receipt),
        "started_at": _utc_now(),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    if prepare_only:
        selected_path = paths["root"] / "prepare" / "selected_tasks_step_000001.jsonl"
        validation_selected_path = (
            paths["root"] / "prepare" / "validation_monitor_tasks.jsonl"
        )
        split_receipt_path = paths["root"] / "prepare" / "split_isolation.json"
        _write_jsonl(
            selected_path,
            [
                {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                for task in first_tasks
            ],
        )
        _atomic_write_json(split_receipt_path, split_isolation_receipt)
        _write_jsonl(
            validation_selected_path,
            [
                {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                for task in validation_tasks
            ],
        )
        manifest["prepared_first_step_task_ids"] = [task.task_id for task in first_tasks]
        manifest["prepared_validation_monitor_path"] = str(
            validation_selected_path
        )
        manifest["split_isolation_receipt_path"] = str(split_receipt_path)
        manifest["completed_at"] = _utc_now()
        _atomic_write_json(paths["manifest"], manifest)
        return manifest

    completed = int(state.get("optimizer_updates_completed", 0)) if state else 0
    if completed > target_steps:
        raise HotpotTrainingError("committed optimizer count exceeds configured target")
    if stop_after_optimizer_steps is not None and completed >= stop_after_optimizer_steps:
        raise HotpotTrainingError("requested boundary was already committed")

    manifest["optimizer_updates_completed"] = completed
    manifest["status"] = "running"
    _atomic_write_json(paths["manifest"], manifest)

    active_tracker = tracker
    if active_tracker is None:
        factory = tracker_factory or (
            lambda value, run_id: WandbTracker(value, run_id=run_id)
        )
        active_tracker = factory(config, state.get("wandb_run_id") if state else None)
    if not active_tracker.run_id.strip():
        raise HotpotTrainingError("tracking backend returned no run ID")
    manifest["wandb"] = {
        "run_id": active_tracker.run_id,
        "run_url": active_tracker.run_url,
        "mode": "online",
    }
    _atomic_write_json(paths["manifest"], manifest)

    transaction = StepTransaction(paths["root"])
    unfinished = transaction.load()
    if unfinished is not None:
        raise HotpotTrainingError(
            "an uncommitted step recovery record exists; automatic replay is "
            "intentionally blocked to avoid a duplicate optimizer update"
        )

    if state:
        behavior_policy = str(state["behavior_policy_version"])
        behavior_adapter = str(state["behavior_adapter_name"])
        behavior_checkpoint = str(state["behavior_adapter_checkpoint"])
        optimizer_checkpoint: Optional[str] = str(
            state["optimizer_state_checkpoint"]
        )
    else:
        behavior_policy = str(director_config["behavior_policy_version"])
        behavior_adapter = str(director_config["behavior_adapter_name"])
        behavior_checkpoint = str(director_config["behavior_adapter_checkpoint"])
        configured_optimizer = director_config.get("optimizer_state_checkpoint")
        optimizer_checkpoint = (
            str(configured_optimizer) if configured_optimizer else None
        )
    best_validation_token_f1 = (
        float(state.get("best_validation_token_f1", -1.0)) if state else -1.0
    )
    ledger_epoch: Optional[LedgerEpoch] = None
    if dynamic_mode:
        if state:
            ledger_receipt_value = state.get("ledger_epoch_receipt")
            if not isinstance(ledger_receipt_value, str) or not ledger_receipt_value.strip():
                raise HotpotTrainingError(
                    "dynamic resume state has no recoverable ledger epoch receipt"
                )
            ledger_receipt_path = Path(ledger_receipt_value).expanduser().resolve()
        else:
            if acceptance_path is None or dynamic_acceptance is None:
                raise HotpotTrainingError("dynamic Phase-E admission was not retained")
            ledger_receipt_path = acceptance_path.parent / "ledger_epoch_000000" / "receipt.json"
            expected_epoch_receipt = _mapping(
                _mapping(dynamic_acceptance["phase_e"], "phase_e")["epoch_snapshot"],
                "phase_e.epoch_snapshot",
            )
            if not ledger_receipt_path.is_file():
                raise HotpotTrainingError(
                    "dynamic Phase-E initial ledger epoch receipt is missing"
                )
        ledger_epoch = LedgerEpoch.load(ledger_receipt_path)
        if ledger_epoch.condition.versions.policy != behavior_policy:
            raise HotpotTrainingError(
                "dynamic ledger epoch policy differs from the behavior policy"
            )
        if ledger_epoch.condition.versions.feature_schema != LEDGER_FEATURE_SCHEMA_VERSION:
            raise HotpotTrainingError("dynamic ledger epoch feature schema drifted")
        if state:
            expected_state_lineage = {
                "ledger_condition_id": ledger_epoch.condition.condition_id,
                "ledger_snapshot_id": ledger_epoch.condition.ledger_snapshot_id,
                "skill_snapshot_id": ledger_epoch.condition.skill_snapshot_id,
            }
            differing = [
                name
                for name, expected in expected_state_lineage.items()
                if state.get(name) != expected
            ]
            if differing:
                raise HotpotTrainingError(
                    "dynamic resume ledger lineage differs: "
                    + ", ".join(differing)
                )
        else:
            saved_epoch_receipt = json.loads(
                ledger_receipt_path.read_text(encoding="utf-8")
            )
            if expected_epoch_receipt.get("receipt_id") != saved_epoch_receipt.get(
                "receipt_id"
            ):
                raise HotpotTrainingError(
                    "dynamic Phase-E acceptance points to another ledger epoch"
                )

    backend_builder = backend_factory or (
        lambda value, run_root: LiveSmokeBackend.from_config(value, run_root)
    )
    exit_code = 1
    active_sequential_runtime: Optional[SequentialGpuRuntime] = None
    try:
        for training_step in range(completed + 1, target_steps + 1):
            if _STOP_REQUESTED or (paths["root"] / "STOP_REQUESTED").is_file():
                manifest["status"] = "paused_at_safe_boundary"
                break
            skill_runtime_update: Optional[SkillRuntimeUpdate] = None
            skill_runtime_update_path: Optional[Path] = None
            if dynamic_mode and ledger_epoch is not None and completed > 0:
                skill_runtime_update, skill_runtime_update_path = (
                    _materialize_skill_runtime_update(
                        config,
                        run_root=paths["root"],
                        ledger_epoch=ledger_epoch,
                        completed_steps=completed,
                    )
                )
                skill_receipt = skill_runtime_update.receipt
                if not skill_receipt.producer_complete:
                    manifest.update(
                        status="paused_pending_heldout_confirmation",
                        optimizer_updates_completed=completed,
                        pending_skill_runtime_update=str(skill_runtime_update_path),
                        pending_skill_rule_ids=list(
                            skill_receipt.pending_confirmation_rule_ids
                        ),
                    )
                    _atomic_write_json(paths["manifest"], manifest)
                    break
                # Publication is delayed by one evidence epoch.  Re-freezing
                # changes only the version-bound Skill snapshot; policy and
                # posterior state remain those of the committed checkpoint.
                ledger_epoch = LedgerEpoch.freeze(
                    ledger_epoch.ledger,
                    skill_runtime_update.skills,
                    ledger_epoch.condition.versions,
                )
                ledger_epoch.save(
                    skill_runtime_update_path.parent / "frozen_epoch"
                )
            step_started = time.monotonic()
            absolute_update_step = int(experiment["initial_update_step"]) + training_step - 1
            candidate_policy = _candidate_policy(config, training_step)
            step_dir = paths["root"] / "steps" / f"step_{training_step:06d}"
            if step_dir.exists():
                raise HotpotTrainingError(
                    f"step artifact directory already exists: {step_dir}"
                )
            step_dir.mkdir(parents=True)
            selected_path = step_dir / "selected_tasks.jsonl"
            trajectories_path = step_dir / "trajectories.jsonl"
            groups_path = step_dir / "grpo_groups.jsonl"
            summary_path = step_dir / "training_summary.json"
            sync_path = step_dir / "sync_receipt.json"
            canary_path = step_dir / "post_update_canary.jsonl"
            canary_ledger_path = step_dir / "post_update_canary_ledger_sidecars.jsonl"
            validation_path_for_step = step_dir / "validation_monitor.jsonl"
            validation_ledger_path = step_dir / "validation_ledger_sidecars.jsonl"
            natural_sidecar_path = step_dir / "natural_ledger_sidecars.jsonl"
            natural_ledger_path = step_dir / "natural_ledger_records.jsonl"
            selected_probe_path = step_dir / "selected_probe_sites.jsonl"
            intervention_path = step_dir / "intervention_trajectories.jsonl"
            intervention_ledger_path = step_dir / "intervention_ledger_records.jsonl"
            probe_records_path = step_dir / "probe_records.jsonl"
            dynamic_metrics_path = step_dir / "dynamic_epoch_metrics.json"
            dynamic_transition_path = step_dir / "dynamic_epoch_transition.json"
            artifact_receipt_path = step_dir / "wandb_checkpoint_artifact.json"
            step_manifest_path = step_dir / "step_manifest.json"

            tasks = sample_hotpotqa_tasks(
                pool,
                optimizer_step=training_step,
                tasks_per_step=int(batch_config["tasks_per_step"]),
                seed=int(experiment["seed"]),
                seed_offset=int(batch_config["seed_offset"]),
            )
            _write_jsonl(
                selected_path,
                [
                    {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                    for task in tasks
                ],
            )
            step_value = _step_config(
                config,
                training_step=training_step,
                absolute_update_step=absolute_update_step,
                behavior_policy_version=behavior_policy,
                behavior_adapter_name=behavior_adapter,
                behavior_adapter_checkpoint=behavior_checkpoint,
                optimizer_state_checkpoint=optimizer_checkpoint,
            )
            transaction.transition(
                step=training_step,
                phase=StepPhase.PREPARED,
                behavior_policy_version=behavior_policy,
                behavior_adapter_name=behavior_adapter,
                candidate_policy_version=candidate_policy,
            )
            transaction.seal(
                step=training_step,
                state={
                    "phase": StepPhase.PREPARED.value,
                    "behavior_policy_version": behavior_policy,
                    "behavior_adapter_name": behavior_adapter,
                    "candidate_policy_version": candidate_policy,
                    "selected_tasks_path": str(selected_path),
                },
            )
            backend = backend_builder(step_value, root)
            active_sequential_runtime = _single_gpu_sequential_runtime(
                step_value,
                backend,
            )
            if active_sequential_runtime is not None:
                await asyncio.to_thread(active_sequential_runtime.start_rollout_service)
            readiness = await _ensure_behavior_ready(backend, step_value)

            dynamic_batch: Optional[DynamicLedgerBatch] = None
            dynamic_metrics: Optional[dict[str, Any]] = None
            if dynamic_mode:
                if ledger_epoch is None:
                    raise HotpotTrainingError("dynamic ledger epoch is not initialized")
                coordinator = _dynamic_epoch_coordinator(
                    step_value,
                    backend,
                    ledger_epoch,
                    adapter_name=behavior_adapter,
                    seed=int(experiment["seed"]) + training_step * 100_000,
                )
                natural_batch = await coordinator.collect_natural(
                    backend,
                    tasks,
                    rollouts_per_task=int(batch_config["rollouts_per_task"]),
                    start_rollout_index=(training_step - 1) * 1000,
                )
                # Preserve completed on-policy sampling before optional paired
                # exploration can fail. These files are evidence, not an
                # optimizer commit or authorization to reuse the batch.
                _write_jsonl(trajectories_path, natural_batch.natural_trajectories)
                _write_jsonl(natural_sidecar_path, natural_batch.natural_sidecars)
                _write_jsonl(natural_ledger_path, natural_batch.natural_ledger_records)
                dynamic_config = _mapping(
                    step_value["dynamic_ledger"], "dynamic_ledger"
                )
                latent_config = _mapping(
                    dynamic_config["latent_risk"], "dynamic_ledger.latent_risk"
                )
                exploration_config = _mapping(
                    step_value["exploration"], "exploration"
                )
                probe_config = _mapping(
                    exploration_config["probe"], "exploration.probe"
                )
                audit_config = _mapping(
                    exploration_config["audit"], "exploration.audit"
                )
                selected_sites = select_probe_sites(
                    natural_batch.natural_sidecars,
                    natural_trajectory_count=len(natural_batch.natural_trajectories),
                    seed=int(experiment["seed"]) + training_step * 100_000 + 50_000,
                    probe_fraction=float(
                        probe_config["budget_fraction_of_natural_trajectories"]
                    ),
                    audit_probability=float(audit_config["probability"]),
                    tau=float(latent_config["tau"]),
                )
                # LatentLoss specification §§6.1/6.3/6.4: probe/audit budgets
                # are conditional, not minimum counts. The existing collector
                # handles an empty selection without inventing interventions;
                # natural trajectories still train GRPO independently. Phase E
                # retains its separate, mandatory paired-probe acceptance gate.
                dynamic_batch = await coordinator.collect_selected_probes(
                    backend,
                    natural_batch,
                    selected_sites=selected_sites,
                    start_rollout_index=5_000_000 + training_step * 100_000,
                )
                trajectories = dynamic_batch.natural_trajectories
                dynamic_metrics = dict(dynamic_epoch_metrics(dynamic_batch))
                routing = _mapping(
                    dynamic_metrics["routing_assertions"],
                    "dynamic_epoch_metrics.routing_assertions",
                )
                if not all(
                    routing.get(name) is True
                    for name in (
                        "natural_only_grpo",
                        "probe_and_audit_excluded_from_grpo",
                        "probe_and_audit_excluded_from_standard_metrics",
                    )
                ):
                    raise HotpotTrainingError(
                        "dynamic evidence routing assertions failed before GRPO"
                    )
                _write_jsonl(
                    selected_probe_path,
                    [
                        {
                            "schema_version": "flowsteer.selected-probe-site.v1",
                            "audit": value.audit,
                            "sampling_probability": value.sampling_probability,
                            **value.site.to_dict(),
                        }
                        for value in dynamic_batch.selected_sites
                    ],
                )
                _write_jsonl(intervention_path, dynamic_batch.intervention_trajectories)
                _write_jsonl(
                    intervention_ledger_path,
                    dynamic_batch.intervention_ledger_records,
                )
                _write_jsonl(probe_records_path, dynamic_batch.probe_records)
                _atomic_write_json(dynamic_metrics_path, dynamic_metrics)
            else:
                rollout_jobs = []
                rollout_index = 0
                for task in tasks:
                    versions = version_bundle_for(
                        task,
                        policy_version=behavior_policy,
                        model_catalog_version=backend.model_catalog_version,
                        prompt_version=str(
                            _mapping(config["versions"], "versions")["prompt"]
                        ),
                        tool_version=str(
                            _mapping(config["versions"], "versions")["tool"]
                        ),
                    )
                    for _ in range(int(batch_config["rollouts_per_task"])):
                        rollout_jobs.append(
                            backend.collect(
                                task,
                                (training_step - 1) * 1000 + rollout_index,
                                versions,
                                expected_task_split="train",
                            )
                        )
                        rollout_index += 1
                trajectories = tuple(await asyncio.gather(*rollout_jobs))
            _verify_rollout_batch(
                trajectories,
                expected_count=int(batch_config["expected_rollouts_per_step"]),
                behavior_policy=behavior_policy,
                behavior_adapter=behavior_adapter,
            )
            _write_jsonl(trajectories_path, trajectories)
            _write_grpo_groups(groups_path, trajectories)
            dynamic_transition: Optional[DynamicEpochCloseResult] = None
            if dynamic_batch is not None:
                if ledger_epoch is None:
                    raise HotpotTrainingError("dynamic ledger epoch disappeared")
                dynamic_transition = close_dynamic_epoch(
                    dynamic_batch,
                    next_policy_version=candidate_policy,
                    # Keep the frozen lifecycle snapshot until the separate
                    # discovery/held-out confirmation producer supplies a vetted
                    # next_skills set.  This cannot activate a Skill by itself.
                    next_skills=ledger_epoch.skills,
                    natural_grpo_trajectory_ids=tuple(
                        item.trajectory_id for item in trajectories
                    ),
                    intervention_exclusion_trajectory_ids=tuple(
                        item.trajectory_id
                        for item in dynamic_batch.intervention_trajectories
                    ),
                    heldout_coverage=None,
                )
                transition_summary = dynamic_transition.transition.summary.to_dict()
                natural_ids = {item.trajectory_id for item in trajectories}
                intervention_ids = {
                    item.trajectory_id
                    for item in dynamic_batch.intervention_trajectories
                }
                if set(transition_summary["grpo_trajectory_ids"]) != natural_ids:
                    raise HotpotTrainingError(
                        "ledger transition GRPO IDs differ from the natural batch"
                    )
                if set(transition_summary["excluded_intervention_ids"]) != intervention_ids:
                    raise HotpotTrainingError(
                        "ledger transition did not exclude every intervention branch"
                    )
                _atomic_write_json(
                    dynamic_transition_path,
                    {
                        "schema_version": "flowsteer.dynamic-epoch-transition.v1",
                        "current_condition": ledger_epoch.condition.to_dict(),
                        "next_condition": dynamic_transition.next_epoch.condition.to_dict(),
                        "summary": transition_summary,
                        "heldout_calibration_status": (
                            "not_applicable_no_qualified_candidate"
                            if skill_runtime_update is not None
                            and skill_runtime_update.receipt.status
                            == "complete_no_qualified_candidate"
                            else "complete"
                            if skill_runtime_update is not None
                            and skill_runtime_update.receipt.confirmation_probe_count > 0
                            else "pending"
                        ),
                        "skill_runtime_update": (
                            skill_runtime_update.receipt.to_dict()
                            if skill_runtime_update is not None
                            else None
                        ),
                        "skill_runtime_update_path": (
                            str(skill_runtime_update_path)
                            if skill_runtime_update_path is not None
                            else None
                        ),
                        "receipt": dynamic_transition.receipt.to_dict(),
                    },
                )
            transaction.seal(
                step=training_step,
                state={
                    "phase": StepPhase.ROLLOUT_COMPLETE.value,
                    "behavior_policy_version": behavior_policy,
                    "behavior_adapter_name": behavior_adapter,
                    "selected_tasks_path": str(selected_path),
                    "trajectories_path": str(trajectories_path),
                    "grpo_groups_path": str(groups_path),
                    "dynamic_epoch_metrics_path": (
                        str(dynamic_metrics_path) if dynamic_mode else None
                    ),
                    "dynamic_epoch_transition_path": (
                        str(dynamic_transition_path) if dynamic_mode else None
                    ),
                },
            )
            transaction.transition(
                step=training_step,
                phase=StepPhase.ROLLOUT_COMPLETE,
                trajectory_count=len(trajectories),
            )

            summary_object: Any
            summary: dict[str, Any]
            sync: dict[str, Any]
            adapter_name: str
            canaries: tuple[TrajectoryRecord, ...]
            next_epoch_receipt: Optional[Mapping[str, Any]] = None
            checkpoint_phase_e_receipt_path: Optional[Path] = None
            recovery_manifest_path: Optional[Path] = None
            updated_coordinator: Optional[DynamicLedgerEpochCoordinator] = None

            def train_and_commit() -> Any:
                nonlocal summary_object, summary, next_epoch_receipt
                nonlocal checkpoint_phase_e_receipt_path, recovery_manifest_path
                transaction.transition(
                    step=training_step,
                    phase=StepPhase.GRADIENT_IN_PROGRESS,
                )
                summary_object = backend.train(trajectories, step_dir / "learner")
                summary = _summary_dict(summary_object)
                _verify_training_summary(
                    summary,
                    behavior_policy=behavior_policy,
                    candidate_policy=candidate_policy,
                    absolute_update_step=absolute_update_step,
                    gradient_worker_count=int(gpu.get("gradient_worker_count", 2)),
                )
                if int(summary.get("input_trajectories", -1)) != len(trajectories):
                    raise HotpotTrainingError(
                        "trainer input count differs from the sealed natural batch"
                    )
                if dynamic_transition is not None:
                    checkpoint_dir = Path(str(summary["checkpoint_dir"]))
                    (
                        next_epoch_receipt,
                        persisted_transition_receipt,
                        ledger_receipt_path,
                    ) = _persist_dynamic_epoch_snapshot(
                        dynamic_transition,
                        checkpoint_dir,
                    )
                    if dynamic_acceptance is None:
                        raise HotpotTrainingError(
                            "dynamic checkpoint has no retained Phase-E receipt"
                        )
                    checkpoint_phase_e_receipt_path = (
                        checkpoint_dir / "phase_e_acceptance_receipt.json"
                    )
                    _atomic_write_json(
                        checkpoint_phase_e_receipt_path,
                        dynamic_acceptance,
                    )
                    training_state = Path(str(summary["training_state_checkpoint"]))
                    try:
                        training_state_relative = str(
                            training_state.resolve().relative_to(checkpoint_dir.resolve())
                        )
                    except ValueError as error:
                        raise HotpotTrainingError(
                            "training state is outside the recoverable checkpoint"
                        ) from error
                    recovery_manifest_path = checkpoint_dir / "recovery_manifest.json"
                    _atomic_write_json(
                        recovery_manifest_path,
                        {
                            "schema_version": "flowsteer.dynamic-grpo-recovery.v1",
                            "status": "checkpointed_pending_publication",
                            "checkpoint_recoverable": bool(
                                summary["checkpoint_recoverable"]
                            ),
                            "optimizer_step": training_step,
                            "absolute_update_step": absolute_update_step,
                            "policy_version": candidate_policy,
                            "behavior_policy_version": behavior_policy,
                            "training_state": training_state_relative,
                            "ledger_epoch_receipt": "ledger_epoch/receipt.json",
                            "phase_e_acceptance_receipt": (
                                "phase_e_acceptance_receipt.json"
                            ),
                            "condition": dynamic_transition.next_epoch.condition.to_dict(),
                            "ttb_enabled": False,
                            "objective": "action_masked_one_pass_grpo",
                        },
                    )
                    transition_payload = json.loads(
                        dynamic_transition_path.read_text(encoding="utf-8")
                    )
                    transition_payload["next_epoch_receipt"] = next_epoch_receipt
                    transition_payload["next_epoch_receipt_path"] = str(
                        ledger_receipt_path
                    )
                    transition_payload["receipt"] = persisted_transition_receipt
                    _atomic_write_json(dynamic_transition_path, transition_payload)
                _atomic_write_json(summary_path, summary)
                transaction.transition(
                    step=training_step,
                    phase=StepPhase.GRADIENT_COMPLETE,
                    loss=float(summary["loss"]),
                    grad_norm=float(summary["grad_norm"]),
                )
                transaction.transition(
                    step=training_step,
                    phase=StepPhase.OPTIMIZER_COMMITTED,
                    trainable_update_l2=float(summary["trainable_update_l2"]),
                    checkpoint_dir=str(summary["checkpoint_dir"]),
                )
                transaction.seal(
                    step=training_step,
                    state={
                        "phase": StepPhase.OPTIMIZER_COMMITTED.value,
                        "behavior_policy_version": behavior_policy,
                        "behavior_adapter_name": behavior_adapter,
                        "candidate_policy_version": candidate_policy,
                        "trajectories_path": str(trajectories_path),
                        "grpo_groups_path": str(groups_path),
                        "training_summary_path": str(summary_path),
                        "checkpoint_dir": str(summary["checkpoint_dir"]),
                        "training_state_checkpoint": str(
                            summary["training_state_checkpoint"]
                        ),
                    },
                )
                return summary_object

            async def publish_and_commit(*, manage_gate: bool) -> Mapping[str, Any]:
                nonlocal sync, adapter_name
                if manage_gate:
                    sync_object = await backend.publish(summary_object)
                else:
                    if not isinstance(backend, LiveSmokeBackend):
                        publish_without_gate = getattr(backend, "publish_without_gate", None)
                        if not callable(publish_without_gate):
                            raise HotpotTrainingError(
                                "single-GPU backend cannot publish under the external gate"
                            )
                        sync_object = await publish_without_gate(summary_object)
                    else:
                        sync_object = await backend.publish(
                            summary_object,
                            manage_gate=False,
                        )
                sync = _summary_dict(sync_object)
                if sync.get("success") is not True:
                    raise HotpotTrainingError("SGLang publication receipt is unsuccessful")
                if sync.get("new_policy_version") != candidate_policy:
                    raise HotpotTrainingError("SGLang published the wrong policy version")
                if sync.get("route_switch_success") is not True:
                    raise HotpotTrainingError("Director policy route switch was not verified")
                adapter_name = str(sync.get("adapter_name", ""))
                if not adapter_name:
                    raise HotpotTrainingError("SGLang publication receipt has no adapter")
                _atomic_write_json(sync_path, sync)
                transaction.transition(
                    step=training_step,
                    phase=StepPhase.SGLANG_SYNCED,
                    adapter_name=adapter_name,
                    policy_version=candidate_policy,
                )
                transaction.seal(
                    step=training_step,
                    state={
                        "phase": StepPhase.SGLANG_SYNCED.value,
                        "behavior_policy_version": behavior_policy,
                        "candidate_policy_version": candidate_policy,
                        "training_summary_path": str(summary_path),
                        "checkpoint_dir": str(summary["checkpoint_dir"]),
                        "training_state_checkpoint": str(
                            summary["training_state_checkpoint"]
                        ),
                        "sync_receipt_path": str(sync_path),
                        "adapter_name": adapter_name,
                    },
                )
                return sync

            async def collect_update_canaries() -> tuple[TrajectoryRecord, ...]:
                nonlocal updated_coordinator
                canary_count = int(sync_config["post_update_canary_count"])
                jobs = []
                canary_hooks: list[Any] = []
                if dynamic_transition is not None:
                    updated_coordinator = _dynamic_epoch_coordinator(
                        step_value,
                        backend,
                        dynamic_transition.next_epoch,
                        adapter_name=adapter_name,
                        seed=int(experiment["seed"])
                        + training_step * 100_000
                        + 80_000,
                    )
                for index in range(canary_count):
                    task = tasks[index % len(tasks)]
                    if updated_coordinator is not None:
                        versions = dynamic_transition.next_epoch.condition.versions
                        hook = updated_coordinator.make_hook(
                            1_000_000 + training_step * 10 + index
                        )
                        canary_hooks.append(hook)
                        condition_id = dynamic_transition.next_epoch.condition.condition_id
                    else:
                        versions = version_bundle_for(
                            task,
                            policy_version=candidate_policy,
                            model_catalog_version=backend.model_catalog_version,
                            prompt_version=str(
                                _mapping(config["versions"], "versions")["prompt"]
                            ),
                            tool_version=str(
                                _mapping(config["versions"], "versions")["tool"]
                            ),
                        )
                        hook = None
                        condition_id = None
                    if hook is None:
                        jobs.append(
                            backend.collect(
                                task,
                                1_000_000 + training_step * 10 + index,
                                versions,
                                expected_task_split="train",
                            )
                        )
                    else:
                        jobs.append(
                            backend.collect(
                                task,
                                1_000_000 + training_step * 10 + index,
                                versions,
                                expected_task_split="train",
                                observation_hook=hook,
                                condition_id=condition_id,
                            )
                        )
                values = tuple([await job for job in jobs])
                _verify_canaries(
                    values,
                    expected_count=canary_count,
                    policy_version=candidate_policy,
                    adapter_name=adapter_name,
                )
                _write_jsonl(canary_path, values)
                if canary_hooks:
                    _write_jsonl(
                        canary_ledger_path,
                        [
                            hook.sidecar(record.trajectory_id)
                            for hook, record in zip(canary_hooks, values)
                        ],
                    )
                return values

            lifecycle_receipt: Optional[SequentialGpuCycleReceipt] = None
            if active_sequential_runtime is None:
                summary_object = await asyncio.to_thread(train_and_commit)
                await publish_and_commit(manage_gate=True)
                canaries = await collect_update_canaries()
            else:
                holder: dict[str, Any] = {}
                runner_loop = asyncio.get_running_loop()

                def run_on_runner_loop(coroutine: Any) -> Any:
                    """Execute backend async work on its original event loop.

                    ``run_update_cycle`` is intentionally executed in a worker
                    thread while the main runner loop stays alive.  Reusing
                    ``asyncio.run`` in that worker creates a second loop and
                    can break backend semaphores/clients already bound to the
                    runner loop.  Submit the coroutine to the owning loop and
                    synchronously wait for its result from the lifecycle
                    thread instead.
                    """

                    future = asyncio.run_coroutine_threadsafe(
                        coroutine,
                        runner_loop,
                    )
                    return future.result()

                def publication_callback() -> object:
                    holder["sync"] = run_on_runner_loop(
                        publish_and_commit(manage_gate=False)
                    )
                    return holder["sync"]

                def canary_callback() -> object:
                    holder["canaries"] = run_on_runner_loop(
                        collect_update_canaries()
                    )
                    return {"canary_succeeded": True}

                lifecycle_receipt = await asyncio.to_thread(
                    active_sequential_runtime.run_update_cycle,
                    training_callback=train_and_commit,
                    publication_callback=publication_callback,
                    canary_callback=canary_callback,
                )
                if not lifecycle_receipt.success:
                    raise HotpotTrainingError("single-GPU update lifecycle failed")
                canaries = tuple(holder["canaries"])
                _atomic_write_json(
                    step_dir / "single_gpu_lifecycle_receipt.json",
                    lifecycle_receipt.to_dict(),
                )

            validation_jobs = []
            validation_hooks: list[Any] = []
            for index, task in enumerate(validation_tasks):
                if dynamic_transition is not None:
                    if updated_coordinator is None:
                        raise HotpotTrainingError(
                            "updated dynamic epoch was not used by the canary"
                        )
                    versions = dynamic_transition.next_epoch.condition.versions
                    hook = updated_coordinator.make_hook(
                        2_000_000 + training_step * 100 + index
                    )
                    validation_hooks.append(hook)
                    condition_id = dynamic_transition.next_epoch.condition.condition_id
                else:
                    versions = version_bundle_for(
                        task,
                        policy_version=candidate_policy,
                        model_catalog_version=backend.model_catalog_version,
                        prompt_version=str(
                            _mapping(config["versions"], "versions")["prompt"]
                        ),
                        tool_version=str(
                            _mapping(config["versions"], "versions")["tool"]
                        ),
                    )
                    hook = None
                    condition_id = None
                if hook is None:
                    validation_jobs.append(
                        backend.collect(
                            task,
                            2_000_000 + training_step * 100 + index,
                            versions,
                            expected_task_split="validation",
                        )
                    )
                else:
                    validation_jobs.append(
                        backend.collect(
                            task,
                            2_000_000 + training_step * 100 + index,
                            versions,
                            expected_task_split="validation",
                            observation_hook=hook,
                            condition_id=condition_id,
                        )
                    )
            validation_trajectories = tuple(await asyncio.gather(*validation_jobs))
            _verify_validation_batch(
                validation_trajectories,
                expected_task_ids=[task.task_id for task in validation_tasks],
                policy_version=candidate_policy,
                adapter_name=adapter_name,
            )
            _write_jsonl(validation_path_for_step, validation_trajectories)
            if validation_hooks:
                _write_jsonl(
                    validation_ledger_path,
                    [
                        hook.sidecar(record.trajectory_id)
                        for hook, record in zip(
                            validation_hooks,
                            validation_trajectories,
                        )
                    ],
                )
            transaction.transition(
                step=training_step,
                phase=StepPhase.VALIDATION_COMPLETE,
                validation_trajectory_count=len(validation_trajectories),
            )
            transaction.seal(
                step=training_step,
                state={
                    "phase": StepPhase.VALIDATION_COMPLETE.value,
                    "behavior_policy_version": behavior_policy,
                    "candidate_policy_version": candidate_policy,
                    "training_summary_path": str(summary_path),
                    "sync_receipt_path": str(sync_path),
                    "post_update_canary_path": str(canary_path),
                    "validation_monitor_path": str(validation_path_for_step),
                    "adapter_name": adapter_name,
                },
            )

            metrics = _step_metrics(
                trajectories,
                validation_trajectories,
                summary,
                training_step=training_step,
                absolute_update_step=absolute_update_step,
                sync=sync,
                canaries=canaries,
                step_seconds=time.monotonic() - step_started,
                dynamic_metrics=dynamic_metrics,
                next_ledger_epoch=(
                    dynamic_transition.next_epoch
                    if dynamic_transition is not None
                    else None
                ),
            )
            if skill_runtime_update is not None:
                skill_receipt = skill_runtime_update.receipt
                metrics.update(
                    {
                        "skill/runtime_status": skill_receipt.status,
                        "skill/discovery_probe_count": (
                            skill_receipt.discovery_probe_count
                        ),
                        "skill/confirmation_probe_count": (
                            skill_receipt.confirmation_probe_count
                        ),
                        "skill/qualified_rule_count": (
                            skill_receipt.discovery_qualified_rule_count
                        ),
                        "skill/runtime_grpo_reward_contribution": (
                            skill_receipt.grpo_reward_contribution
                        ),
                    }
                )
            previous_best = best_validation_token_f1
            current_validation_f1 = float(metrics["validation/token_f1"])
            is_best = current_validation_f1 > previous_best
            artifact_aliases = ["latest"] + (["best"] if is_best else [])
            if dynamic_transition is not None:
                if (
                    recovery_manifest_path is None
                    or checkpoint_phase_e_receipt_path is None
                ):
                    raise HotpotTrainingError(
                        "dynamic checkpoint recovery files were not materialized"
                    )
                recovery_payload = json.loads(
                    recovery_manifest_path.read_text(encoding="utf-8")
                )
                recovery_payload.update(
                    status="ready",
                    checkpoint_recoverable=True,
                    behavior_adapter_name=adapter_name,
                    best_validation_token_f1=(
                        current_validation_f1 if is_best else previous_best
                    ),
                    wandb_run_id=active_tracker.run_id,
                    finalized_at=_utc_now(),
                )
                _atomic_write_json(recovery_manifest_path, recovery_payload)
            artifact_receipt = dict(
                active_tracker.log_checkpoint(
                    Path(str(summary["checkpoint_dir"])),
                    metadata={
                        "global_step": training_step,
                        "absolute_update_step": absolute_update_step,
                        "dataset": "hotpotqa",
                        "behavior_policy_version": behavior_policy,
                        "updated_policy_version": candidate_policy,
                        "checkpoint_recoverable": bool(
                            summary["checkpoint_recoverable"]
                        ),
                        "training_state_checkpoint": str(
                            summary["training_state_checkpoint"]
                        ),
                        "validation_exact_match": float(
                            metrics["validation/exact_match"]
                        ),
                        "validation_token_f1": current_validation_f1,
                    },
                    aliases=artifact_aliases,
                )
            )
            _atomic_write_json(artifact_receipt_path, artifact_receipt)
            transaction.transition(
                step=training_step,
                phase=StepPhase.WANDB_ARTIFACT_LOGGED,
                artifact_name=str(artifact_receipt.get("name", "")),
                artifact_version=str(artifact_receipt.get("version", "")),
            )
            transaction.seal(
                step=training_step,
                state={
                    "phase": StepPhase.WANDB_ARTIFACT_LOGGED.value,
                    "behavior_policy_version": behavior_policy,
                    "candidate_policy_version": candidate_policy,
                    "training_summary_path": str(summary_path),
                    "sync_receipt_path": str(sync_path),
                    "validation_monitor_path": str(validation_path_for_step),
                    "wandb_checkpoint_artifact_path": str(
                        artifact_receipt_path
                    ),
                    "adapter_name": adapter_name,
                },
            )
            metrics["timing/step_seconds"] = time.monotonic() - step_started
            metrics["checkpoint/wandb_artifact_name"] = str(
                artifact_receipt.get("name", "")
            )
            metrics["checkpoint/wandb_artifact_version"] = str(
                artifact_receipt.get("version", "")
            )
            next_state = {
                "schema_version": "flowsteer.hotpotqa.grpo_run_state.v1",
                "optimizer_updates_completed": training_step,
                "absolute_update_step": absolute_update_step,
                "behavior_policy_version": candidate_policy,
                "behavior_adapter_name": adapter_name,
                "behavior_adapter_checkpoint": str(summary["checkpoint_dir"]),
                "optimizer_state_checkpoint": str(
                    summary["optimizer_state_checkpoint"]
                ),
                "training_state_checkpoint": str(
                    summary["training_state_checkpoint"]
                ),
                "best_validation_token_f1": (
                    current_validation_f1 if is_best else previous_best
                ),
                "latest_wandb_checkpoint_artifact": artifact_receipt,
                "wandb_run_id": active_tracker.run_id,
                "last_step_manifest": str(step_manifest_path),
                "updated_at": _utc_now(),
            }
            if dynamic_transition is not None:
                if next_epoch_receipt is None:
                    raise HotpotTrainingError(
                        "dynamic checkpoint has no ledger epoch receipt"
                    )
                next_state.update(
                    {
                        "ledger_epoch_receipt": str(
                            Path(str(summary["checkpoint_dir"]))
                            / "ledger_epoch"
                            / "receipt.json"
                        ),
                        "ledger_condition_id": (
                            dynamic_transition.next_epoch.condition.condition_id
                        ),
                        "ledger_snapshot_id": (
                            dynamic_transition.next_epoch.condition.ledger_snapshot_id
                        ),
                        "skill_snapshot_id": (
                            dynamic_transition.next_epoch.condition.skill_snapshot_id
                        ),
                        "dynamic_phase_e_acceptance_receipt": str(
                            checkpoint_phase_e_receipt_path
                        ),
                        "dynamic_phase_e_acceptance": dict(dynamic_acceptance or {}),
                        "recovery_manifest": str(recovery_manifest_path),
                        "skill_runtime_update_receipt": (
                            str(skill_runtime_update_path)
                            if skill_runtime_update_path is not None
                            else None
                        ),
                    }
                )
            step_manifest = {
                "schema_version": "flowsteer.hotpotqa.grpo_step_manifest.v1",
                "status": "committed",
                "optimizer_step": training_step,
                "absolute_update_step": absolute_update_step,
                "behavior_policy_version": behavior_policy,
                "candidate_policy_version": candidate_policy,
                "behavior_readiness": dict(readiness),
                "selected_task_ids": [task.task_id for task in tasks],
                "trajectory_count": len(trajectories),
                "training": summary,
                "policy_sync": sync,
                "canary_trajectory_ids": [item.trajectory_id for item in canaries],
                "validation_trajectory_ids": [
                    item.trajectory_id for item in validation_trajectories
                ],
                "wandb_checkpoint_artifact": artifact_receipt,
                "metrics": metrics,
                "dynamic_epoch": (
                    {
                        "current_condition": (
                            ledger_epoch.condition.to_dict()
                            if ledger_epoch is not None
                            else None
                        ),
                        "next_condition": (
                            dynamic_transition.next_epoch.condition.to_dict()
                            if dynamic_transition is not None
                            else None
                        ),
                        "transition_summary": (
                            dynamic_transition.transition.summary.to_dict()
                            if dynamic_transition is not None
                            else None
                        ),
                        "next_epoch_receipt": next_epoch_receipt,
                        "heldout_calibration_status": (
                            "not_applicable_no_qualified_candidate"
                            if skill_runtime_update is not None
                            and skill_runtime_update.receipt.status
                            == "complete_no_qualified_candidate"
                            else "complete"
                            if skill_runtime_update is not None
                            and skill_runtime_update.receipt.confirmation_probe_count > 0
                            else "pending_independent_paired_evidence"
                        ),
                        "skill_runtime_update": (
                            skill_runtime_update.receipt.to_dict()
                            if skill_runtime_update is not None
                            else None
                        ),
                    }
                    if dynamic_mode
                    else None
                ),
                "artifacts": {
                    "selected_tasks": str(selected_path),
                    "trajectories": str(trajectories_path),
                    "grpo_groups": str(groups_path),
                    "training_summary": str(summary_path),
                    "sync_receipt": str(sync_path),
                    "post_update_canary": str(canary_path),
                    "validation_monitor": str(validation_path_for_step),
                    "wandb_checkpoint_artifact": str(artifact_receipt_path),
                    "natural_ledger_sidecars": (
                        str(natural_sidecar_path) if dynamic_mode else None
                    ),
                    "natural_ledger_records": (
                        str(natural_ledger_path) if dynamic_mode else None
                    ),
                    "selected_probe_sites": (
                        str(selected_probe_path) if dynamic_mode else None
                    ),
                    "intervention_trajectories": (
                        str(intervention_path) if dynamic_mode else None
                    ),
                    "intervention_ledger_records": (
                        str(intervention_ledger_path) if dynamic_mode else None
                    ),
                    "probe_records": (
                        str(probe_records_path) if dynamic_mode else None
                    ),
                    "skill_runtime_update": (
                        str(skill_runtime_update_path)
                        if skill_runtime_update_path is not None
                        else None
                    ),
                    "dynamic_epoch_metrics": (
                        str(dynamic_metrics_path) if dynamic_mode else None
                    ),
                    "dynamic_epoch_transition": (
                        str(dynamic_transition_path) if dynamic_mode else None
                    ),
                    "post_update_canary_ledger_sidecars": (
                        str(canary_ledger_path) if dynamic_mode else None
                    ),
                    "validation_ledger_sidecars": (
                        str(validation_ledger_path) if dynamic_mode else None
                    ),
                },
                "committed_at": _utc_now(),
                "wandb_logged": False,
            }
            _atomic_write_json(step_manifest_path, step_manifest)
            _append_jsonl(paths["training_log"], metrics)
            # W&B is part of the formal step boundary for this run.  Do not
            # publish COMMITTED state (and therefore do not admit the next
            # rollout batch) until the online record succeeds.
            active_tracker.log(metrics, step=training_step)
            transaction.transition(
                step=training_step,
                phase=StepPhase.WANDB_LOGGED,
                wandb_run_id=active_tracker.run_id,
            )
            transaction.seal(
                step=training_step,
                state={
                    "phase": StepPhase.WANDB_LOGGED.value,
                    "behavior_policy_version": behavior_policy,
                    "candidate_policy_version": candidate_policy,
                    "step_manifest_path": str(step_manifest_path),
                    "wandb_run_id": active_tracker.run_id,
                    "adapter_name": adapter_name,
                },
            )
            step_manifest["wandb_logged"] = True
            _atomic_write_json(step_manifest_path, step_manifest)
            _atomic_write_json(paths["state"], next_state)
            transaction.transition(
                step=training_step,
                phase=StepPhase.COMMITTED,
                policy_version=candidate_policy,
                adapter_name=adapter_name,
            )
            if dynamic_mode and training_step == 1:
                _atomic_write_json(
                    paths["root"] / "one_step_closure_receipt.json",
                    {
                        "schema_version": "flowsteer.dynamic-grpo-one-step-closure.v1",
                        "status": "passed",
                        "optimizer_updates": 1,
                        "objective": "action_masked_one_pass_grpo",
                        "ttb_enabled": False,
                        "behavior_policy_version": behavior_policy,
                        "updated_policy_version": candidate_policy,
                        "nonzero_gradient": float(summary["grad_norm"]) > 0.0,
                        "nonzero_lora_update": (
                            float(summary["trainable_update_l2"]) > 0.0
                        ),
                        "checkpoint_recoverable": bool(
                            summary["checkpoint_recoverable"]
                        ),
                        "publish_success": bool(sync["success"]),
                        "route_switch_success": bool(
                            sync["route_switch_success"]
                        ),
                        "next_policy_rollout_verified": bool(canaries),
                        "wandb_logged": True,
                        "wandb_run_id": active_tracker.run_id,
                        "ledger_epoch_receipt": next_state.get(
                            "ledger_epoch_receipt"
                        ),
                        "heldout_calibration_status": (
                            "pending_independent_paired_evidence"
                        ),
                        "skill_runtime_update_status": (
                            "pending_discovery_and_heldout_confirmation_producer"
                        ),
                        "long_training_authorized": False,
                        "created_at": _utc_now(),
                    },
                )
            transaction.clear()

            completed = training_step
            behavior_policy = candidate_policy
            behavior_adapter = adapter_name
            behavior_checkpoint = str(summary["checkpoint_dir"])
            optimizer_checkpoint = str(summary["optimizer_state_checkpoint"])
            if dynamic_transition is not None:
                ledger_epoch = dynamic_transition.next_epoch
            best_validation_token_f1 = float(
                next_state["best_validation_token_f1"]
            )
            manifest.update(
                optimizer_updates_completed=completed,
                current_policy_version=behavior_policy,
                current_adapter_name=behavior_adapter,
                current_adapter_checkpoint=behavior_checkpoint,
                optimizer_state_checkpoint=optimizer_checkpoint,
                latest_step_manifest=str(step_manifest_path),
                last_step_metrics=metrics,
            )
            _atomic_write_json(paths["manifest"], manifest)

            if active_sequential_runtime is not None:
                await asyncio.to_thread(
                    active_sequential_runtime.stop_rollout_service
                )
                active_sequential_runtime = None

            if stop_after_optimizer_steps == completed:
                manifest["status"] = "paused_at_requested_boundary"
                break
            if _STOP_REQUESTED or (paths["root"] / "STOP_REQUESTED").is_file():
                manifest["status"] = "paused_at_safe_boundary"
                break
        else:
            manifest["status"] = "completed"

        manifest["optimizer_updates_completed"] = completed
        manifest["completed_at"] = _utc_now()
        _atomic_write_json(paths["manifest"], manifest)
        active_tracker.update_summary(
            {
                "status": manifest["status"],
                "optimizer_updates_completed": completed,
                "target_optimizer_steps": target_steps,
                "current_policy_version": manifest.get("current_policy_version", ""),
                "current_adapter_name": manifest.get("current_adapter_name", ""),
            }
        )
        active_tracker.finish(exit_code=0)
        exit_code = 0
        return manifest
    except Exception as exc:
        if active_sequential_runtime is not None:
            try:
                await asyncio.to_thread(
                    active_sequential_runtime.stop_rollout_service
                )
            except Exception:
                # Preserve the original stage failure; the lifecycle receipt
                # already records fail-closed cleanup state when available.
                pass
            active_sequential_runtime = None
        manifest.update(
            status="failed",
            optimizer_updates_completed=completed,
            error=_safe_error(exc),
            failed_at=_utc_now(),
        )
        _atomic_write_json(paths["manifest"], manifest)
        try:
            active_tracker.update_summary(
                {
                    "status": "failed",
                    "optimizer_updates_completed": completed,
                    "error_type": type(exc).__name__,
                }
            )
            active_tracker.finish(exit_code=exit_code)
        except Exception:
            pass
        if isinstance(exc, HotpotTrainingError):
            raise
        raise HotpotTrainingError("HotpotQA training failed") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/training_hotpotqa_grpo.yaml",
        help="HotpotQA sequential GRPO YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and materialize the first task plan without API/GPU/W&B",
    )
    parser.add_argument(
        "--phase0-rollout-only",
        action="store_true",
        help="collect and validate one fresh 7x4 train batch without optimization/W&B",
    )
    parser.add_argument(
        "--dynamic-phase-e-only",
        action="store_true",
        help=(
            "run the LatentLoss 50x4 real integration gate with paired probes; "
            "performs no optimizer update and starts no W&B run"
        ),
    )
    parser.add_argument(
        "--allow-md-grpo",
        action="store_true",
        help="explicitly authorize the selected MD Action-Masked One-Pass GRPO",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the last committed optimizer step",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help=(
            "continue a dynamic-ledger run from an explicit checkpoint directory "
            "containing recovery_manifest.json"
        ),
    )
    parser.add_argument(
        "--stop-after-optimizer-steps",
        type=int,
        help="pause at this committed run step (use 1 for the required closure proof)",
    )
    parser.add_argument(
        "--dynamic-acceptance-receipt",
        help=(
            "explicit Phase-E acceptance_receipt.json required before the first "
            "dynamic-ledger optimizer step"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bounded_modes = sum(
        bool(value)
        for value in (
            args.prepare_only,
            args.phase0_rollout_only,
            args.dynamic_phase_e_only,
        )
    )
    if bounded_modes > 1:
        print(
            "--prepare-only, --phase0-rollout-only and --dynamic-phase-e-only "
            "are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    training_only = {
        "--allow-md-grpo": bool(args.allow_md_grpo),
        "--resume": bool(args.resume),
        "--resume-checkpoint": bool(args.resume_checkpoint),
        "--stop-after-optimizer-steps": args.stop_after_optimizer_steps is not None,
        "--dynamic-acceptance-receipt": bool(args.dynamic_acceptance_receipt),
    }
    invalid_training_flags = [
        name for name, selected in training_only.items() if selected
    ]
    if bounded_modes and invalid_training_flags:
        print(
            "bounded modes cannot be combined with training-only flags: "
            + ", ".join(invalid_training_flags),
            file=sys.stderr,
        )
        return 2
    if args.resume and args.resume_checkpoint:
        print(
            "--resume and --resume-checkpoint are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        if args.dynamic_phase_e_only:
            manifest = asyncio.run(
                run_hotpotqa_dynamic_phase_e(
                    _resolve(PROJECT_ROOT, args.config),
                    project_root=PROJECT_ROOT,
                )
            )
        elif args.phase0_rollout_only:
            manifest = asyncio.run(
                run_hotpotqa_phase0(
                    _resolve(PROJECT_ROOT, args.config),
                    project_root=PROJECT_ROOT,
                )
            )
        else:
            manifest = asyncio.run(
                run_hotpotqa_training(
                    _resolve(PROJECT_ROOT, args.config),
                    project_root=PROJECT_ROOT,
                    prepare_only=bool(args.prepare_only),
                    allow_md_grpo=bool(args.allow_md_grpo),
                    resume=bool(args.resume),
                    resume_checkpoint=args.resume_checkpoint,
                    stop_after_optimizer_steps=args.stop_after_optimizer_steps,
                    dynamic_acceptance_receipt=args.dynamic_acceptance_receipt,
                )
            )
    except (ConfigurationError, HotpotTrainingError, ValueError, RuntimeError) as exc:
        print(f"HotpotQA training failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "optimizer_updates_completed": manifest.get(
                    "optimizer_updates_completed", manifest.get("optimizer_updates", 0)
                ),
                "target_optimizer_steps": manifest.get("target_optimizer_steps"),
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
