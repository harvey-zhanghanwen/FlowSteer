#!/usr/bin/env python3
"""Train the HotpotQA Flow-Director with sequential one-pass GRPO updates.

The learning objective and exact receipt gate come from the existing
``train_agentgraph_smoke.py``/``smoke_trainer.py`` path.  The outer loop reuses
SkillFlow's formal execution boundary: sample a complete batch, finish all
rollouts, compute and commit one optimizer update, publish the new theta LoRA
under pause/drain, verify it with a canary, and only then admit the next batch.
SkillFlow's TTB backward policy is deliberately not used because the project
design specifies action-masked one-pass GRPO for the Flow-Director.

MACE, Bayesian posterior updates, and Skill evolution are not part of this
runner.  Their config flags are required to remain disabled.
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
from src.interactive.records import TaskRecord, TrajectoryRecord
from src.interactive.step_transaction import StepPhase, StepTransaction
from src.interactive.task_dataset import iter_task_records


class HotpotTrainingError(RuntimeError):
    """The sequential training transaction failed closed."""


class TrainingTracker(Protocol):
    run_id: str
    run_url: str

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        ...

    def update_summary(self, values: Mapping[str, Any]) -> None:
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
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise ConfigurationError("WANDB_API_KEY is required for online training")
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

    def log(self, values: Mapping[str, Any], *, step: int) -> None:
        try:
            self._run.log(dict(values), step=step, commit=True)
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases step logging failed") from exc

    def update_summary(self, values: Mapping[str, Any]) -> None:
        try:
            self._run.summary.update(dict(values))
        except Exception as exc:  # pragma: no cover - network/account runtime
            raise RuntimeError("Weights & Biases summary update failed") from exc

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
    """Publish one state file with SkillFlow's staging-and-replace boundary."""

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
        "experiment.phase": experiment.get("phase") == "hotpotqa_grpo_training",
        "experiment.training_enabled": experiment.get("training_enabled") is True,
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
    if grpo.get("ttb_enabled") is not False:
        raise ConfigurationError("SkillFlow TTB is outside the MD's GRPO task flow")
    if config.get("mace", {}).get("enabled") is not False:
        raise ConfigurationError("MACE must remain disabled in this task-learning run")
    if config.get("bayesian_posterior", {}).get("enabled") is not False:
        raise ConfigurationError(
            "Bayesian posterior updates must remain disabled in this task-learning run"
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


def _step_metrics(
    trajectories: Sequence[TrajectoryRecord],
    summary: Mapping[str, Any],
    *,
    training_step: int,
    absolute_update_step: int,
    sync: Mapping[str, Any],
    canaries: Sequence[TrajectoryRecord],
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
    return {
        "optimizer_step": training_step,
        "absolute_update_step": absolute_update_step,
        "reward/mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "reward/count": len(rewards),
        "hotpotqa/exact_match": sum(exact) / len(exact) if exact else 0.0,
        "hotpotqa/token_f1": sum(f1) / len(f1) if f1 else 0.0,
        "hotpotqa/terminal_failures": sum(
            int(record.terminal_failure) for record in trajectories
        ),
        "train/loss": float(summary["loss"]),
        "train/grad_norm": float(summary["grad_norm"]),
        "train/update_l2": float(summary["trainable_update_l2"]),
        "train/informative_groups": int(summary["informative_groups"]),
        "train/trained_trajectories": int(summary["trained_trajectories"]),
        "train/micro_batch_size": int(summary["micro_batch_size_used"]),
        "train/oom_backoff_count": int(summary["oom_backoff_count"]),
        "policy/behavior_version": str(summary["behavior_policy_version"]),
        "policy/updated_version": str(summary["updated_policy_version"]),
        "policy/adapter_name": str(sync["adapter_name"]),
        "policy/sync_success": bool(sync["success"]),
        "policy/canary_success": bool(canaries),
        "checkpoint/path": str(summary["checkpoint_dir"]),
        "checkpoint/optimizer_state": str(summary["optimizer_state_checkpoint"]),
    }


async def run_hotpotqa_training(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prepare_only: bool = False,
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
    paths = _training_paths(config, root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    experiment = _mapping(config["experiment"], "experiment")
    data = _mapping(config["data"], "data")
    batch_config = _mapping(data["batch"], "data.batch")
    director_config = _mapping(config["director"], "director")
    sync_config = _mapping(config["policy_sync"], "policy_sync")
    target_steps = int(experiment["target_optimizer_steps"])
    if stop_after_optimizer_steps is not None and (
        type(stop_after_optimizer_steps) is not int
        or not 1 <= stop_after_optimizer_steps <= target_steps
    ):
        raise ConfigurationError(
            "stop_after_optimizer_steps must be within the configured target"
        )

    train_path = _resolve(root, str(data["train_path"]))
    pool = load_hotpotqa_training_pool(train_path)
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
        "unique_hotpotqa_train_tasks": len(pool),
        "started_at": _utc_now(),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    if prepare_only:
        selected_path = paths["root"] / "prepare" / "selected_tasks_step_000001.jsonl"
        _write_jsonl(
            selected_path,
            [
                {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
                for task in first_tasks
            ],
        )
        manifest["prepared_first_step_task_ids"] = [task.task_id for task in first_tasks]
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

    backend_builder = backend_factory or (
        lambda value, run_root: LiveSmokeBackend.from_config(value, run_root)
    )
    exit_code = 1
    try:
        for training_step in range(completed + 1, target_steps + 1):
            if _STOP_REQUESTED or (paths["root"] / "STOP_REQUESTED").is_file():
                manifest["status"] = "paused_at_safe_boundary"
                break
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

            sync_object = await backend.publish(summary_object)
            sync = _summary_dict(sync_object)
            if sync.get("success") is not True:
                raise HotpotTrainingError("SGLang publication receipt is unsuccessful")
            if sync.get("new_policy_version") != candidate_policy:
                raise HotpotTrainingError("SGLang published the wrong policy version")
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

            metrics = _step_metrics(
                trajectories,
                summary,
                training_step=training_step,
                absolute_update_step=absolute_update_step,
                sync=sync,
                canaries=canaries,
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
                "metrics": metrics,
                "artifacts": {
                    "selected_tasks": str(selected_path),
                    "trajectories": str(trajectories_path),
                    "grpo_groups": str(groups_path),
                    "training_summary": str(summary_path),
                    "sync_receipt": str(sync_path),
                    "post_update_canary": str(canary_path),
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
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        manifest = asyncio.run(
            run_hotpotqa_training(
                _resolve(PROJECT_ROOT, args.config),
                project_root=PROJECT_ROOT,
                prepare_only=bool(args.prepare_only),
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
                "optimizer_updates_completed": manifest[
                    "optimizer_updates_completed"
                ],
                "target_optimizer_steps": manifest["target_optimizer_steps"],
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
