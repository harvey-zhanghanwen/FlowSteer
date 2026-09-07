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
from src.interactive.records import TaskRecord, TrajectoryRecord
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
        "experiment.phase": experiment.get("phase") == "hotpotqa_grpo_training",
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
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.enabled": skills.get("enabled") is False,
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


def _candidate_policy(config: Mapping[str, Any], training_step: int) -> str:
    prefix = str(
        _mapping(config["director"], "director")["updated_policy_version_prefix"]
    )
    return f"{prefix}{training_step:06d}"


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


async def validate_phase0_rollout_batch(
    trajectories: Sequence[TrajectoryRecord],
    *,
    expected_task_ids: Sequence[str],
    expected_count: int,
    behavior_policy: str,
    behavior_adapter: str,
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
    if actual_tasks != expected_tasks or len(expected_tasks) != 7:
        raise HotpotTrainingError("Phase 0 task set differs from the frozen 7x4 plan")
    if len({record.trajectory_id for record in trajectories}) != expected_count:
        raise HotpotTrainingError("Phase 0 trajectory IDs are not unique")
    if len({record.rollout_id for record in trajectories}) != expected_count:
        raise HotpotTrainingError("Phase 0 rollout IDs are not unique")
    group_counts = Counter(record.group_key for record in trajectories)
    if len(group_counts) != 7 or set(group_counts.values()) != {4}:
        raise HotpotTrainingError("Phase 0 requires seven exact same-condition groups")
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
                if response.get("attempt_count") != 1:
                    raise HotpotTrainingError(f"Phase 0 Agent retry lacks per-attempt lineage: {label}")
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
        or len(memory) != 2
        or any(float(value) <= 0.0 for value in memory.values())
    ):
        raise HotpotTrainingError("training summary lacks two-GPU memory evidence")


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
    return {
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
        else LiveSmokeBackend.from_config(step_value, root, evaluation_only=True)
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


async def run_hotpotqa_training(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prepare_only: bool = False,
    allow_md_grpo: bool = False,
    resume: bool = False,
    stop_after_optimizer_steps: Optional[int] = None,
    backend_factory: Optional[BackendFactory] = None,
    tracker: Optional[TrainingTracker] = None,
    tracker_factory: Optional[TrackerFactory] = None,
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
    experiment = _mapping(config["experiment"], "experiment")
    compliance = _mapping(config["md_compliance"], "md_compliance")
    gpu = _mapping(config["gpu"], "gpu")
    if not prepare_only:
        if not allow_md_grpo:
            raise HotpotTrainingError(
                "the selected MD GRPO requires explicit --allow-md-grpo authorization"
            )
        if experiment.get("training_enabled") is not True or gpu.get(
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
            if compliance.get("one_step_closure_status") != "passed" or compliance.get(
                "long_training_authorized"
            ) is not True:
                raise HotpotTrainingError(
                    "long training requires a passed real one-step closure"
                )
    paths = _training_paths(config, root)
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
        "bayesian_posterior_enabled": False,
        "skills_enabled": False,
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

    state: Optional[dict[str, Any]] = None
    if paths["state"].is_file():
        if not resume:
            raise HotpotTrainingError(
                "a committed run state already exists; pass --resume to continue"
            )
        raw_state = json.loads(paths["state"].read_text(encoding="utf-8"))
        if not isinstance(raw_state, dict):
            raise HotpotTrainingError("committed run state is malformed")
        state = raw_state
    elif resume:
        raise HotpotTrainingError("--resume was requested but no committed state exists")

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

    backend_builder = backend_factory or (
        lambda value, run_root: LiveSmokeBackend.from_config(value, run_root)
    )
    exit_code = 1
    try:
        for training_step in range(completed + 1, target_steps + 1):
            if _STOP_REQUESTED or (paths["root"] / "STOP_REQUESTED").is_file():
                manifest["status"] = "paused_at_safe_boundary"
                break
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
            validation_path_for_step = step_dir / "validation_monitor.jsonl"
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
            readiness = await _ensure_behavior_ready(backend, step_value)

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
            transaction.seal(
                step=training_step,
                state={
                    "phase": StepPhase.ROLLOUT_COMPLETE.value,
                    "behavior_policy_version": behavior_policy,
                    "behavior_adapter_name": behavior_adapter,
                    "selected_tasks_path": str(selected_path),
                    "trajectories_path": str(trajectories_path),
                    "grpo_groups_path": str(groups_path),
                },
            )
            transaction.transition(
                step=training_step,
                phase=StepPhase.ROLLOUT_COMPLETE,
                trajectory_count=len(trajectories),
            )

            transaction.transition(
                step=training_step,
                phase=StepPhase.GRADIENT_IN_PROGRESS,
            )
            summary_object = await asyncio.to_thread(
                backend.train, trajectories, step_dir / "learner"
            )
            summary = _summary_dict(summary_object)
            _verify_training_summary(
                summary,
                behavior_policy=behavior_policy,
                candidate_policy=candidate_policy,
                absolute_update_step=absolute_update_step,
            )
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

            sync_object = await backend.publish(summary_object)
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

            canary_count = int(sync_config["post_update_canary_count"])
            canary_jobs = []
            for index in range(canary_count):
                task = tasks[index % len(tasks)]
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
                canary_jobs.append(
                    backend.collect(
                        task,
                        1_000_000 + training_step * 10 + index,
                        versions,
                        expected_task_split="train",
                    )
                )
            # The formal path uses one canary and waits for it directly.  This
            # leaves no admission window for a next-step rollout while route
            # verification is pending.
            canaries = tuple([await job for job in canary_jobs])
            _verify_canaries(
                canaries,
                expected_count=canary_count,
                policy_version=candidate_policy,
                adapter_name=adapter_name,
            )
            _write_jsonl(canary_path, canaries)

            validation_jobs = []
            for index, task in enumerate(validation_tasks):
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
                validation_jobs.append(
                    backend.collect(
                        task,
                        2_000_000 + training_step * 100 + index,
                        versions,
                        expected_task_split="validation",
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
            )
            previous_best = best_validation_token_f1
            current_validation_f1 = float(metrics["validation/token_f1"])
            is_best = current_validation_f1 > previous_best
            artifact_aliases = ["latest"] + (["best"] if is_best else [])
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
                "artifacts": {
                    "selected_tasks": str(selected_path),
                    "trajectories": str(trajectories_path),
                    "grpo_groups": str(groups_path),
                    "training_summary": str(summary_path),
                    "sync_receipt": str(sync_path),
                    "post_update_canary": str(canary_path),
                    "validation_monitor": str(validation_path_for_step),
                    "wandb_checkpoint_artifact": str(artifact_receipt_path),
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
            transaction.clear()

            completed = training_step
            behavior_policy = candidate_policy
            behavior_adapter = adapter_name
            behavior_checkpoint = str(summary["checkpoint_dir"])
            optimizer_checkpoint = str(summary["optimizer_state_checkpoint"])
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
        "--stop-after-optimizer-steps",
        type=int,
        help="pause at this committed run step (use 1 for the required closure proof)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare_only and args.phase0_rollout_only:
        print("--prepare-only and --phase0-rollout-only are mutually exclusive", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        if args.phase0_rollout_only:
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
                    stop_after_optimizer_steps=args.stop_after_optimizer_steps,
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
