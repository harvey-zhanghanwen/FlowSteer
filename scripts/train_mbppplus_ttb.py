#!/usr/bin/env python3
"""Run the MBPP+ project adaptation of SkillFlow's primary TTB training.

This entry point deliberately does not call ``train_agentgraph_smoke.run_smoke``:
that entry point owns the Action-Masked One-Pass GRPO baseline.  It reuses only
the existing FlowSteer Canvas/runtime/trajectory wiring, SkillFlow's MBPP public
test reward adapter, the TTB trainer, and the SGLang LoRA publisher.

Live execution is fail-closed.  No model is loaded until all selected GPUs are
exclusive and have the configured free-memory reserve, and an online W&B run
has started successfully.  ``--prepare-only`` performs only configuration,
dataset, W&B-presence, and read-only GPU inventory checks.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_agentgraph_smoke import LiveSmokeBackend
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.records import TaskRecord, TrajectoryRecord
from src.interactive.sglang_manager import SGLangSupervisorManager
from src.interactive.task_dataset import load_task_records
from src.interactive.ttb_config import (
    GPUAssignment,
    ProjectRuntimeProfile,
    TTBTrainingConfig,
    validate_ttb_config,
)
from src.interactive.ttb_monitor import (
    TTB_MONITOR_SCHEMA_VERSION,
    WandbTTBMonitor,
    WandbTTBMonitorConfig,
    ttb_monitor_run_config,
)
from src.interactive.ttb_trainer import (
    Qwen35TTBTrainer,
    TTBStepCoordinate,
    TTBTrainerConfig,
)
from src.interactive.versioning import VersionBundle


RUN_MANIFEST_SCHEMA = "flowsteer.agentgraph.mbppplus-ttb-run.v1"
RESOURCE_GATE_SCHEMA = "flowsteer.agentgraph.gpu-resource-gate.v1"
MBPP_TRAINING_EVALUATOR_VERSION = (
    "skillflow.training.reward.code_test_pass_rate"
)
DEFAULT_CONFIG = "config/training_mbppplus_ttb_v1.yaml"
THETA_PREFIX = "qwen35-9b-mbppplus-ttb-theta-step-"
PHI_PREFIX = "qwen35-9b-mbppplus-ttb-phi-step-"
Z_PREFIX = "qwen35-9b-mbppplus-ttb-z-step-"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class MBPPPlusTTBRunError(RuntimeError):
    """The fail-closed MBPP+ TTB transaction could not continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MBPPPlusTTBRunError(f"{name} must be a mapping")
    return value


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MBPPPlusTTBRunError(f"{name} must be non-empty text")
    return value.strip()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _safe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:4000]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = row.to_dict() if callable(getattr(row, "to_dict", None)) else row
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _version(prefix: str, step: int) -> str:
    if type(step) is not int or step < 0:
        raise ValueError("version step must be a non-negative integer")
    return f"{prefix}{step:06d}"


def _theta_version(step: int) -> str:
    return _version(THETA_PREFIX, step)


def _phi_version(step: int) -> str:
    return _version(PHI_PREFIX, step)


def _z_version(step: int) -> str:
    return _version(Z_PREFIX, step)


def _run_id(value: Optional[str]) -> str:
    resolved = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if _RUN_ID.fullmatch(resolved) is None:
        raise MBPPPlusTTBRunError(
            "run_id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return resolved


def _continuation_server_weight_version(
    configured_initial_version: str,
    continuation_checkpoint: Optional[Path],
) -> str:
    """Resolve the exact committed SGLang weight receipt for the behavior route."""

    if continuation_checkpoint is None:
        return _non_empty(
            configured_initial_version,
            "director.expected_server_weight_version",
        )
    receipt_path = continuation_checkpoint.expanduser().resolve() / "publication_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MBPPPlusTTBRunError(
            f"cannot read continuation publication receipt: {receipt_path}"
        ) from exc
    receipt = _mapping(receipt, "continuation publication receipt")
    return _non_empty(
        receipt.get("server_weight_version"),
        "continuation publication receipt.server_weight_version",
    )


def _bind_exact_serving_route(
    backend: LiveSmokeBackend,
    *,
    policy_version: str,
    adapter_name: str,
    server_weight_version: str,
) -> None:
    """Bind the discovered post-switch SGLang receipt before new rollouts."""

    policy_version = _non_empty(policy_version, "policy_version")
    adapter_name = _non_empty(adapter_name, "adapter_name")
    server_weight_version = _non_empty(
        server_weight_version,
        "server_weight_version",
    )
    backend.rollout_gate.pause()
    try:
        backend.rollout_gate.drain()
        backend.director_client.update_policy_route(
            policy_version=policy_version,
            adapter_name=adapter_name,
            expected_server_weight_version=server_weight_version,
        )
    finally:
        backend.rollout_gate.resume()


class MBPPPlusTTBBackend(LiveSmokeBackend):
    """Use the existing runtime with SkillFlow's public-test training reward."""

    async def evaluate_final_graph(
        self,
        task: TaskRecord,
        final_answer: Optional[str],
        final_graph: Mapping[str, Any],
        *,
        rollout_index: int,
        environment_replay_trace: Sequence[Mapping[str, Any]] = (),
        repository_patch: Optional[str] = None,
    ) -> Any:
        if task.metadata.get("training_population") in {
            "flowsteer_mbpp_train",
            "flowsteer_mbpp_validation",
        }:
            try:
                from src.interactive.mbpp_training_adapter import (
                    evaluate_mbpp_public_tests,
                )
            except ImportError as exc:
                raise MBPPPlusTTBRunError(
                    "the SkillFlow MBPP public-test reward adapter is unavailable"
                ) from exc
            runtime = _mapping(
                self.config.get("mbppplus_tool_runtime"),
                "mbppplus_tool_runtime",
            )
            return await asyncio.to_thread(
                evaluate_mbpp_public_tests,
                task,
                final_answer or "",
                timeout_seconds=float(runtime["python_timeout_seconds"]),
            )
        return await super().evaluate_final_graph(
            task,
            final_answer,
            final_graph,
            rollout_index=rollout_index,
            environment_replay_trace=environment_replay_trace,
            repository_patch=repository_patch,
        )


def validate_mbppplus_ttb_config(config: Mapping[str, Any]) -> TTBTrainingConfig:
    """Validate the unmixed formal TTB profile and project runtime choices."""

    validate_agent_graph_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    ttb = _mapping(config.get("ttb"), "ttb")
    director = _mapping(config.get("director"), "director")
    graph = _mapping(config.get("agent_graph"), "agent_graph")
    grpo = _mapping(config.get("grpo"), "grpo")
    skills = _mapping(config.get("skills"), "skills")
    gpu = _mapping(config.get("gpu"), "gpu")
    runtime = _mapping(ttb.get("project_runtime"), "ttb.project_runtime")
    wandb = _mapping(config.get("wandb"), "wandb")

    checks = {
        "experiment.training_enabled": experiment.get("training_enabled") is True,
        "experiment.phase": experiment.get("phase") == "mbppplus_ttb_training",
        "ttb.enabled": ttb.get("enabled") is True,
        "ttb.objective": ttb.get("objective") == "tempered_trajectory_balance",
        "ttb.losses_mixed": ttb.get("losses_mixed") is False,
        "ttb.on_policy_rollouts": ttb.get("on_policy_rollouts") is True,
        "ttb.executor_frozen": ttb.get("executor_frozen") is True,
        "ttb.optimizer_steps": ttb.get("optimizer_steps") == 250,
        "ttb.questions_per_batch": ttb.get("questions_per_batch") == 7,
        "ttb.trajectories_per_question": ttb.get("trajectories_per_question") == 4,
        "ttb.effective_batch_size": ttb.get("effective_batch_size") == 28,
        "ttb.max_trajectory_steps": ttb.get("max_trajectory_steps") == 12,
        "ttb.beta": float(ttb.get("beta", math.nan)) == 1.0,
        "ttb.epsilon_min": float(ttb.get("epsilon_min", math.nan)) == 0.1,
        "ttb.edge_logprob_normalization": ttb.get("edge_logprob_normalization")
        == "per_action_token",
        "ttb.reasoning_token_treatment": ttb.get("reasoning_token_treatment")
        == "context_only",
        "ttb.learning_rate": float(ttb.get("learning_rate", math.nan)) == 1.0e-4,
        "ttb.weight_decay": float(ttb.get("weight_decay", math.nan)) == 0.01,
        "ttb.weight_decay_source": ttb.get("weight_decay_source")
        == "SkillFlow released trainer; absent from paper main table",
        "ttb.max_grad_norm": float(ttb.get("max_grad_norm", math.nan)) == 3.0,
        "ttb.kl_coefficient": float(ttb.get("kl_coefficient", math.nan)) == 0.01,
        "ttb.checkpoint_interval_steps": ttb.get("checkpoint_interval_steps") == 10,
        "director.action_decoding": director.get("action_decoding")
        == "unconstrained",
        "director.sampling_action_profile": director.get(
            "sampling_action_profile"
        )
        is None,
        "director.max_rounds": director.get("max_rounds") == 12,
        "grpo.enabled": grpo.get("enabled") is False,
        "skills.enabled": skills.get("enabled") is False,
        "gpu.training_enabled": gpu.get("training_enabled") is True,
        "wandb.enabled": wandb.get("enabled") is True,
        "wandb.mode": wandb.get("mode") == "online",
        "agent_graph.contract_type": graph.get("contract_type") == "free_text",
        "project_runtime.rollout_workers": runtime.get("rollout_workers") == 28,
        "project_runtime.micro_batch_size": runtime.get("micro_batch_size") == 1,
        "project_runtime.policy_sync_interval_steps": runtime.get(
            "policy_sync_interval_steps"
        )
        == 1,
        "project_runtime.require_updated_theta_before_next_rollout": runtime.get(
            "require_updated_theta_before_next_rollout"
        )
        is True,
    }
    theta = _mapping(ttb.get("theta_lora"), "ttb.theta_lora")
    phi = _mapping(ttb.get("phi_lora"), "ttb.phi_lora")
    z_profile = _mapping(ttb.get("z"), "ttb.z")
    checks.update(
        {
            "ttb.theta_lora.rank": theta.get("rank") == 64,
            "ttb.theta_lora.alpha": theta.get("alpha") == 128,
            "ttb.theta_lora.dropout": float(theta.get("dropout", math.nan))
            == 0.05,
            "ttb.theta_lora.dropout_source": theta.get("dropout_source")
            == "SkillFlow released trainer; absent from paper main table",
            "ttb.theta_lora.target_modules": theta.get("target_modules")
            == ["q_proj", "k_proj", "v_proj", "o_proj"],
            "ttb.phi_lora.rank": phi.get("rank") == 16,
            "ttb.phi_lora.alpha": phi.get("alpha") == 32,
            "ttb.phi_lora.dropout": float(phi.get("dropout", math.nan)) == 0.05,
            "ttb.phi_lora.dropout_source": phi.get("dropout_source")
            == "SkillFlow released trainer; absent from paper main table",
            "ttb.phi_lora.target_modules": phi.get("target_modules")
            == ["q_proj", "v_proj"],
            "ttb.phi_lora.appendix_rank_ambiguity": phi.get(
                "appendix_rank_ambiguity"
            )
            == 32,
            "ttb.z.separate_parameter": z_profile.get("separate_parameter") is True,
            "ttb.z.optimized": z_profile.get("optimized") is True,
        }
    )
    for section_name in ("mace", "bayesian_posterior", "skill_evolution"):
        section = _mapping(config.get(section_name), section_name)
        checks[f"{section_name}.enabled"] = section.get("enabled") is False
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise MBPPPlusTTBRunError(
            "MBPP+ TTB config violates the selected formal condition: "
            + ", ".join(failed)
        )

    assignments = (
        GPUAssignment("learner", int(gpu["learner_physical"])),
        GPUAssignment("rollout_supervisor", int(gpu["rollout_physical"])),
        GPUAssignment("gradient_replica", int(gpu["gradient_replica_physical"])),
    )
    formal = TTBTrainingConfig(
        runtime=ProjectRuntimeProfile(
            gpu_assignments=assignments,
            rollout_workers=int(runtime["rollout_workers"]),
            micro_batch_size=int(runtime["micro_batch_size"]),
        )
    )
    return validate_ttb_config(formal, require_runtime_resources=True)


def _gpu_inventory() -> tuple[dict[int, dict[str, Any]], list[str]]:
    blockers: list[str] = []
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {}, [f"nvidia-smi GPU inventory failed: {exc}"]
    inventory: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    try:
        for row in csv.reader(io.StringIO(output)):
            index = int(row[0].strip())
            uuid = row[1].strip()
            inventory[index] = {
                "index": index,
                "uuid": uuid,
                "name": row[2].strip(),
                "memory_total_mib": int(row[3].strip()),
                "memory_used_mib": int(row[4].strip()),
                "memory_free_mib": int(row[5].strip()),
                "compute_processes": [],
            }
            uuid_to_index[uuid] = index
    except (IndexError, ValueError) as exc:
        return {}, [f"nvidia-smi GPU inventory was malformed: {exc}"]

    process_query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        process_output = subprocess.check_output(
            process_query,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        blockers.append(f"nvidia-smi process inventory failed: {exc}")
        return inventory, blockers
    for row in csv.reader(io.StringIO(process_output)):
        if not row or not row[0].strip():
            continue
        index = uuid_to_index.get(row[0].strip())
        if index is None:
            continue
        try:
            inventory[index]["compute_processes"].append(
                {
                    "pid": int(row[1].strip()),
                    "used_memory_mib": int(row[2].strip()),
                    "process_name": row[3].strip(),
                }
            )
        except (IndexError, ValueError) as exc:
            blockers.append(f"nvidia-smi process row was malformed: {exc}")
    return inventory, blockers


def gpu_resource_gate(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require exclusive selected GPUs and the configured memory reserve."""

    gpu = _mapping(config.get("gpu"), "gpu")
    gate = _mapping(gpu.get("resource_gate"), "gpu.resource_gate")
    minimum = int(gate["minimum_free_memory_mib_per_gpu"])
    require_exclusive = gate.get("require_no_compute_processes") is True
    roles = {
        "learner": int(gpu["learner_physical"]),
        "rollout_supervisor": int(gpu["rollout_physical"]),
        "gradient_replica": int(gpu["gradient_replica_physical"]),
    }
    inventory, blockers = _gpu_inventory()
    selected: dict[str, Any] = {}
    for role, index in roles.items():
        observed = inventory.get(index)
        if observed is None:
            blockers.append(f"{role}: physical GPU {index} is unavailable")
            continue
        selected[role] = observed
        if int(observed["memory_free_mib"]) < minimum:
            blockers.append(
                f"{role}: GPU {index} has {observed['memory_free_mib']} MiB free; "
                f"requires at least {minimum} MiB"
            )
        processes = observed["compute_processes"]
        if require_exclusive and processes:
            blockers.append(
                f"{role}: GPU {index} has {len(processes)} existing compute process(es)"
            )
    return {
        "schema_version": RESOURCE_GATE_SCHEMA,
        "checked_at": _utc_now(),
        "ready": not blockers,
        "require_no_compute_processes": require_exclusive,
        "minimum_free_memory_mib_per_gpu": minimum,
        "selected": selected,
        "blockers": blockers,
        "source": "nvidia-smi read-only launch inventory",
    }


def _select_step_tasks(
    tasks: Sequence[TaskRecord],
    *,
    step: int,
    questions_per_batch: int,
) -> tuple[TaskRecord, ...]:
    if len(tasks) < questions_per_batch:
        raise MBPPPlusTTBRunError(
            "the disjoint MBPP training population has fewer than seven tasks"
        )
    start = ((step - 1) * questions_per_batch) % len(tasks)
    selected = tuple(
        tasks[(start + offset) % len(tasks)]
        for offset in range(questions_per_batch)
    )
    if len({task.task_id for task in selected}) != questions_per_batch:
        raise MBPPPlusTTBRunError("one TTB batch must contain seven distinct tasks")
    return selected


def _static_preflight(
    config: Mapping[str, Any],
    *,
    root: Path,
    start_step: int,
    continuation_checkpoint: Optional[Path],
) -> tuple[Mapping[str, Any], tuple[TaskRecord, ...]]:
    data = _mapping(config.get("data"), "data")
    director = _mapping(config.get("director"), "director")
    graph = _mapping(config.get("agent_graph"), "agent_graph")
    wandb = _mapping(config.get("wandb"), "wandb")
    ttb = _mapping(config.get("ttb"), "ttb")
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    required_paths = {
        "model": _resolve(root, str(director["base_model"])),
        "tokenizer": _resolve(root, str(director["tokenizer_path"])),
        "training_dataset": _resolve(root, str(data["train_path"])),
        "model_catalog": _resolve(root, str(graph["model_catalog_path"])),
        "mbpp_training_reward_adapter": root
        / "src"
        / "interactive"
        / "mbpp_training_adapter.py",
    }
    for name, path in required_paths.items():
        ready = path.is_dir() if name in {"model", "tokenizer"} else path.is_file()
        checks[name] = {"path": str(path), "ready": ready}
        if not ready:
            blockers.append(f"required {name} path is unavailable: {path}")

    if start_step > 1:
        if continuation_checkpoint is None:
            blockers.append("step 2+ requires a continuation checkpoint")
        else:
            checkpoint = continuation_checkpoint.expanduser().resolve()
            checkpoint_files = {
                "training_state": checkpoint / "training_state.json",
                "optimizer_states": checkpoint / "optimizer_states.pt",
                "z_head": checkpoint / "z_head.pt",
                "theta_config": checkpoint / "theta" / "adapter_config.json",
                "theta_weights": checkpoint / "theta" / "adapter_model.safetensors",
                "phi_config": checkpoint / "phi" / "adapter_config.json",
                "phi_weights": checkpoint / "phi" / "adapter_model.safetensors",
                "publication_receipt": checkpoint / "publication_receipt.json",
            }
            checks["continuation_checkpoint"] = {
                "path": str(checkpoint),
                "required_files": {
                    name: path.is_file() for name, path in checkpoint_files.items()
                },
            }
            missing = [
                name for name, path in checkpoint_files.items() if not path.is_file()
            ]
            if missing:
                blockers.append(
                    "continuation checkpoint is incomplete: " + ", ".join(missing)
                )
            else:
                try:
                    state = json.loads(
                        checkpoint_files["training_state"].read_text(encoding="utf-8")
                    )
                    expected = {
                        "committed_step": start_step - 1,
                        "theta_version": _theta_version(start_step - 1),
                        "phi_version": _phi_version(start_step - 1),
                        "z_version": _z_version(start_step - 1),
                        "publication_committed": True,
                    }
                    if not isinstance(state, Mapping) or any(
                        state.get(name) != value for name, value in expected.items()
                    ):
                        raise MBPPPlusTTBRunError(
                            "continuation training_state does not match the preceding "
                            "committed TTB step"
                        )
                    checks["continuation_checkpoint"]["state_matches"] = True
                except (OSError, ValueError, MBPPPlusTTBRunError) as exc:
                    blockers.append(f"continuation checkpoint validation failed: {exc}")
    else:
        checks["continuation_checkpoint"] = {
            "path": None,
            "required": False,
            "base_policy_start": True,
        }

    tasks: tuple[TaskRecord, ...] = ()
    if checks["training_dataset"]["ready"]:
        try:
            loaded = tuple(
                load_task_records(required_paths["training_dataset"], expected_split="train")
            )
            expected_count = int(data["expected_train_task_count"])
            if len(loaded) != expected_count:
                raise MBPPPlusTTBRunError(
                    f"training task count {len(loaded)} differs from {expected_count}"
                )
            if any(
                task.metadata.get("training_population") != "flowsteer_mbpp_train"
                for task in loaded
            ):
                raise MBPPPlusTTBRunError(
                    "training dataset contains a task outside flowsteer_mbpp_train"
                )
            _select_step_tasks(
                loaded,
                step=start_step,
                questions_per_batch=int(ttb["questions_per_batch"]),
            )
            tasks = loaded
            checks["training_dataset"]["task_count"] = len(tasks)
        except (OSError, TypeError, ValueError, MBPPPlusTTBRunError) as exc:
            blockers.append(f"training dataset validation failed: {exc}")

    auth_env = _non_empty(wandb.get("auth_env"), "wandb.auth_env")
    wandb_sdk_ready = importlib.util.find_spec("wandb") is not None
    wandb_auth_present = bool(os.environ.get(auth_env, ""))
    checks["wandb"] = {
        "sdk_available": wandb_sdk_ready,
        "online_auth_environment_present": wandb_auth_present,
        "auth_environment_name": auth_env,
    }
    if not wandb_sdk_ready:
        blockers.append("the W&B SDK is unavailable in the active Python environment")
    if not wandb_auth_present:
        blockers.append(f"required W&B online auth environment is unset: {auth_env}")

    resource = gpu_resource_gate(config)
    blockers.extend(str(value) for value in resource["blockers"])
    return (
        {
            "checked_at": _utc_now(),
            "ready": not blockers,
            "checks": checks,
            "resource_gate": resource,
            "blockers": blockers,
        },
        tasks,
    )


def _manifest(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    run_id: str,
    run_root: Path,
    start_step: int,
    target_step: int,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    ttb = _mapping(config["ttb"], "ttb")
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "ready_to_launch" if preflight["ready"] else "blocked_preflight",
        "run_id": run_id,
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "run_root": str(run_root),
        "training_base": {
            "branch": "backup/mbppplus-best-v6-20260906",
            "commit": "66e66f02c908d16d097ceb75d65745d90a155d29",
            "immutable": True,
        },
        "scope": {
            "benchmark": "mbpp_plus_project_adaptation",
            "skillflow_joint_iid_reproduction": False,
            "selected_objective": "tempered_trajectory_balance",
            "design_document_objective": "action_masked_one_pass_grpo",
            "objective_conflict_recorded": True,
            "losses_mixed": False,
            "grpo_enabled": False,
            "mace_enabled": False,
            "bayesian_posterior_enabled": False,
            "skill_evolution_enabled": False,
            "executor_frozen": True,
        },
        "formal_ttb": {
            "optimizer_steps": int(ttb["optimizer_steps"]),
            "start_step": start_step,
            "target_step_this_invocation": target_step,
            "questions_per_batch": 7,
            "trajectories_per_question": 4,
            "effective_batch_size": 28,
            "max_trajectory_steps": 12,
            "beta": 1.0,
            "epsilon_min": 0.1,
            "edge_logprob_normalization": "per_action_token",
            "reasoning_token_treatment": "context_only",
            "theta_lora": {
                "rank": 64,
                "alpha": 128,
                "dropout": 0.05,
                "dropout_source": "SkillFlow released trainer; paper table omits it",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "phi_lora": {
                "main_table_rank": 16,
                "appendix_rank_ambiguity": 32,
                "selected_rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "dropout_source": "SkillFlow released trainer; paper table omits it",
                "target_modules": ["q_proj", "v_proj"],
            },
            "z_separately_optimized": True,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "weight_decay_source": "SkillFlow released trainer; paper table omits it",
            "max_grad_norm": 3.0,
            "kl_coefficient": 0.01,
            "checkpoint_interval_steps": 10,
        },
        "project_implementation": {
            "gpu_role_mapping": True,
            "intra_batch_rollout_concurrency": True,
            "two_replica_gradient_partition": True,
            "single_phase_unconstrained_action_sampling": True,
            "policy_publication_barrier": True,
            "recovery_checkpoint_each_step": True,
            "mbpp_transfer_training_population": True,
        },
        "transaction_order": [
            "on_policy_rollout",
            "terminal_public_test_reward",
            "ttb_loss",
            "backward",
            "theta_phi_z_optimizer_step",
            "checkpoint",
            "theta_lora_publish_and_canary",
            "wandb_commit",
            "next_step_rollout",
        ],
        "cross_step_rollout_prefetch": False,
        "optimizer_updates_completed": 0,
        "last_committed_step": start_step - 1,
        "preflight": dict(preflight),
        "steps": [],
    }


def _versions(config: Mapping[str, Any], backend: LiveSmokeBackend, step: int) -> VersionBundle:
    experiment = _mapping(config["experiment"], "experiment")
    return VersionBundle(
        policy=_theta_version(step - 1),
        model_catalog=backend.model_catalog_version,
        evaluator=MBPP_TRAINING_EVALUATOR_VERSION,
        prompt=str(experiment["prompt_version"]),
        tool=str(experiment["tool_version"]),
        posterior="none",
        skill_library="none",
    )


async def _collect_ttb_batch(
    backend: LiveSmokeBackend,
    tasks: Sequence[TaskRecord],
    versions: VersionBundle,
    *,
    condition_id: str,
    schedule_purpose: str,
    step: int,
    trajectories_per_question: int,
    max_attempts_per_question: int,
) -> tuple[tuple[TrajectoryRecord, ...], tuple[TrajectoryRecord, ...]]:
    admitted: dict[str, list[TrajectoryRecord]] = {task.task_id: [] for task in tasks}
    filtered: list[TrajectoryRecord] = []
    attempts = {task.task_id: 0 for task in tasks}
    while any(len(admitted[task.task_id]) < trajectories_per_question for task in tasks):
        wave: list[tuple[TaskRecord, int]] = []
        for task in tasks:
            missing = trajectories_per_question - len(admitted[task.task_id])
            available = max_attempts_per_question - attempts[task.task_id]
            for _ in range(min(missing, available)):
                rollout_index = attempts[task.task_id]
                attempts[task.task_id] += 1
                wave.append((task, rollout_index))
        if not wave:
            shortages = {
                task.task_id: trajectories_per_question - len(admitted[task.task_id])
                for task in tasks
                if len(admitted[task.task_id]) < trajectories_per_question
            }
            raise MBPPPlusTTBRunError(
                "TTB rollout admission exhausted its bounded attempts: "
                + json.dumps(shortages, sort_keys=True)
            )
        records = await asyncio.gather(
            *(
                backend.collect(
                    task,
                    rollout_index,
                    versions,
                    expected_task_split="train",
                    condition_id=condition_id,
                    sampling_schedule_purpose=schedule_purpose,
                    sampling_anchor_ordinal=step,
                )
                for task, rollout_index in wave
            )
        )
        for record in records:
            bucket = admitted[record.task.task_id]
            if record.ttb_eligible and len(bucket) < trajectories_per_question:
                bucket.append(record)
            else:
                filtered.append(record)
    sealed = tuple(record for task in tasks for record in admitted[task.task_id])
    return sealed, tuple(filtered)


async def _post_update_canary(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    config: Mapping[str, Any],
    *,
    step: int,
    adapter_name: str,
) -> TrajectoryRecord:
    versions = _versions(config, backend, step + 1)
    record = await backend.collect(
        task,
        100_000 + step,
        versions,
        expected_task_split="train",
        condition_id=str(config["experiment"]["condition_id"]),
        sampling_schedule_purpose=str(
            config["experiment"]["sampling_schedule_purpose"]
        ),
        sampling_anchor_ordinal=step + 1,
    )
    if not record.turns or any(
        turn.policy_version != _theta_version(step)
        or turn.policy_adapter != adapter_name
        for turn in record.turns
    ):
        raise MBPPPlusTTBRunError(
            "post-update AgentGraph canary did not use the published theta route"
        )
    return record


def _step_receipt(
    summary: Any,
    publication: Any,
    *,
    valid_count: int,
    filtered_count: int,
    step_seconds: float,
    gpu_metrics: Mapping[str, int],
    server_weight_version: str,
) -> dict[str, Any]:
    reward_sum = float(summary.reward_mean) * valid_count
    return {
        "schema_version": TTB_MONITOR_SCHEMA_VERSION,
        "status": "completed",
        "step": int(summary.update_step),
        "versions": {
            "behavior": {
                "theta": summary.behavior_policy_version,
                "phi": summary.phi_behavior_version,
                "z": summary.z_behavior_version,
            },
            "updated": {
                "theta": summary.updated_policy_version,
                "phi": summary.phi_updated_version,
                "z": summary.z_updated_version,
            },
        },
        "ttb": {
            "loss": float(summary.ttb_loss),
            "delta_squared_mean": float(summary.delta_squared_mean),
        },
        "rollout": {
            "valid_count": valid_count,
            "filtered_count": filtered_count,
            "reward_mean": float(summary.reward_mean),
            "reward_sum": reward_sum,
            "trajectory_length_mean": float(summary.trajectory_edges_mean),
            "trajectory_length_max": int(summary.trajectory_edges_max),
        },
        "runtime": {
            "gpu_device_count": int(gpu_metrics["device_count"]),
            "gpu_peak_allocated_bytes": int(gpu_metrics["peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(gpu_metrics["peak_reserved_bytes"]),
            "step_seconds": step_seconds,
            "rollouts_per_second": valid_count / step_seconds,
            "action_tokens_per_second": int(summary.action_token_count) / step_seconds,
            "error_count": 0,
            "errors": [],
        },
        "updates": {
            "theta": {
                "grad_norm": float(summary.theta_grad_norm),
                "update_l2": float(summary.theta_update_l2),
                "nonzero": True,
            },
            "phi": {
                "grad_norm": float(summary.phi_grad_norm),
                "update_l2": float(summary.phi_update_l2),
                "nonzero": True,
            },
            "z": {
                "grad_norm": float(summary.z_grad_norm),
                "update_l2": float(summary.z_update_l2),
                "nonzero": True,
            },
        },
        "publication": {
            "success": bool(publication.success),
            "canary_success": bool(publication.canary_succeeded),
            "theta_version": publication.new_policy_version,
            "adapter_name": publication.adapter_name,
            "server_weight_version": _non_empty(
                server_weight_version,
                "publication server_weight_version",
            ),
            "checkpoint_version": publication.checkpoint_version,
            "sync_seconds": float(publication.duration_seconds),
        },
        "checkpoints": {
            "official": {
                "due": int(summary.update_step) % 10 == 0,
                "saved": bool(summary.official_checkpoint_saved),
                "path": summary.official_checkpoint_dir or None,
            },
            "project_recovery": {
                "saved": True,
                "path": summary.recovery_checkpoint_dir,
            },
        },
        "skill": {"enabled": False, "phase": "disabled", "events": []},
    }


def _gpu_peak_metrics(trainer: Qwen35TTBTrainer) -> Mapping[str, int]:
    torch = getattr(trainer, "_torch", None)
    if torch is None:
        raise MBPPPlusTTBRunError("TTB trainer did not expose its CUDA runtime")
    devices = (
        trainer.config.learner_device,
        trainer.config.gradient_replica_device,
    )
    allocated = sum(int(torch.cuda.max_memory_allocated(device)) for device in devices)
    reserved = sum(int(torch.cuda.max_memory_reserved(device)) for device in devices)
    return {
        # torch sees the two in-process training replicas. The SGLang rollout
        # Supervisor is a separate process and is intentionally not folded
        # into these allocator counters.
        "device_count": 2,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": max(reserved, allocated),
    }


async def run_ttb(
    config_path: Path,
    *,
    prepare_only: bool,
    stop_after_step: Optional[int],
    start_step: int,
    continuation_checkpoint: Optional[Path],
    run_id: str,
    project_root: Path,
) -> Mapping[str, Any]:
    config = load_yaml(config_path)
    formal = validate_mbppplus_ttb_config(config)
    total_steps = formal.optimizer_steps
    if not 1 <= start_step <= total_steps:
        raise MBPPPlusTTBRunError("start_step must be in [1, 250]")
    if start_step > 1 and continuation_checkpoint is None:
        raise MBPPPlusTTBRunError("step 2+ requires --continuation-checkpoint")
    if start_step == 1 and continuation_checkpoint is not None:
        raise MBPPPlusTTBRunError("step 1 must start from the base policy")
    target_step = start_step if stop_after_step == 1 else total_steps
    output = _resolve(project_root, str(config["storage"]["output_root"]))
    run_root = output / "runs" / run_id
    if run_root.exists():
        raise MBPPPlusTTBRunError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path = run_root / "run_manifest.json"

    preflight, tasks = _static_preflight(
        config,
        root=project_root,
        start_step=start_step,
        continuation_checkpoint=continuation_checkpoint,
    )
    current_server_weight_version = _continuation_server_weight_version(
        str(config["director"]["expected_server_weight_version"]),
        continuation_checkpoint,
    )
    manifest = _manifest(
        config_path,
        config,
        run_id=run_id,
        run_root=run_root,
        start_step=start_step,
        target_step=target_step,
        preflight=preflight,
    )
    schedule = [
        {
            "step": step,
            "task_ids": [
                task.task_id
                for task in _select_step_tasks(
                    tasks,
                    step=step,
                    questions_per_batch=formal.questions_per_batch,
                )
            ],
        }
        for step in range(start_step, target_step + 1)
    ] if tasks else []
    _write_json(run_root / "schedule.json", {"steps": schedule})
    _write_json(manifest_path, manifest)
    if prepare_only:
        return manifest
    if not preflight["ready"]:
        raise MBPPPlusTTBRunError(
            "live TTB launch is blocked by preflight: "
            + "; ".join(str(value) for value in preflight["blockers"])
        )

    wandb_section = _mapping(config["wandb"], "wandb")
    monitor: Optional[WandbTTBMonitor] = None
    supervisor: Optional[SGLangSupervisorManager] = None
    trainer: Optional[Qwen35TTBTrainer] = None
    active_step = start_step - 1
    exit_code = 1
    try:
        monitor = WandbTTBMonitor.start(
            WandbTTBMonitorConfig(
                project=str(wandb_section["project"]),
                run_name=f"{wandb_section['run_name']}-{run_id}",
                entity=wandb_section.get("entity"),
                group=wandb_section.get("group"),
                job_type=str(wandb_section["job_type"]),
                tags=tuple(str(value) for value in wandb_section["tags"]),
            ),
            run_config={
                **ttb_monitor_run_config(
                    dataset="mbpp_plus_project_adaptation",
                    total_steps=total_steps,
                    start_step=start_step - 1,
                    condition_id=str(config["experiment"]["condition_id"]),
                ),
                "formal_ttb": formal.to_manifest_dict(),
                "run_manifest_schema": RUN_MANIFEST_SCHEMA,
            },
        )
        manifest["wandb"] = {"mode": "online", "run_id": monitor.run_id}
        manifest["status"] = "starting_supervisor"
        _write_json(manifest_path, manifest)

        gpu = _mapping(config["gpu"], "gpu")
        supervisor_config = _mapping(config["supervisor"], "supervisor")
        os.environ.setdefault("SGLANG_API_KEY", str(supervisor_config["api_key"]))
        supervisor = SGLangSupervisorManager(
            model_path=str(config["director"]["base_model"]),
            port=int(supervisor_config["port"]),
            api_key=str(supervisor_config["api_key"]),
            gpu_id=int(gpu["rollout_physical"]),
            max_lora_rank=64,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            max_loras_per_batch=1,
            max_loaded_loras=2,
            mem_fraction_static=float(supervisor_config["mem_fraction_static"]),
            context_length=int(config["director"]["max_context_tokens"]),
            served_model_name=str(config["director"]["served_model_name"]),
            ready_timeout_seconds=float(supervisor_config["ready_timeout_seconds"]),
        )
        await asyncio.to_thread(supervisor.start)
        runtime_config = deepcopy(config)
        runtime_config["storage"]["root"] = str(run_root / "evidence")
        backend = MBPPPlusTTBBackend.from_config(
            runtime_config,
            project_root,
            evaluation_only=True,
        )
        manifest["supervisor_runtime"] = dict(
            await asyncio.to_thread(backend.publisher.server_runtime_receipt)
        )

        previous_adapter: Optional[str] = None
        if start_step > 1:
            assert continuation_checkpoint is not None
            previous_adapter = backend.publisher.adapter_name(start_step - 1)
            await asyncio.to_thread(
                backend.publisher.ensure_loaded_adapter,
                checkpoint_path=continuation_checkpoint / "theta",
                adapter_name=previous_adapter,
            )
            backend.rollout_gate.pause()
            try:
                backend.rollout_gate.drain()
                backend.director_client.update_policy_route(
                    policy_version=_theta_version(start_step - 1),
                    adapter_name=previous_adapter,
                    expected_server_weight_version=current_server_weight_version,
                )
            finally:
                backend.rollout_gate.resume()

        trainer = Qwen35TTBTrainer(
            TTBTrainerConfig(
                model_path=str(config["director"]["base_model"]),
                tokenizer_path=str(config["director"]["tokenizer_path"]),
                behavior_policy_version=_theta_version(start_step - 1),
                updated_policy_version=_theta_version(start_step),
                behavior_phi_version=_phi_version(start_step - 1),
                updated_phi_version=_phi_version(start_step),
                behavior_z_version=_z_version(start_step - 1),
                updated_z_version=_z_version(start_step),
                behavior_server_weight_version=current_server_weight_version,
                update_step=start_step,
                learner_device=str(gpu["learner_device"]),
                gradient_replica_device=str(gpu["gradient_replica_device"]),
                behavior_policy_adapter=previous_adapter,
                continuation_checkpoint=(
                    str(continuation_checkpoint)
                    if continuation_checkpoint is not None
                    else None
                ),
                learning_rate=1.0e-4,
                weight_decay=float(config["ttb"]["weight_decay"]),
                beta=1.0,
                epsilon_min=0.1,
                kl_coefficient=0.01,
                max_grad_norm=3.0,
                expected_batch_size=28,
                questions_per_batch=7,
                trajectories_per_question=4,
                max_trajectory_edges=12,
                max_sequence_tokens=int(config["director"]["max_context_tokens"]),
                checkpoint_interval=10,
                gradient_checkpointing=True,
                require_unconstrained_action_sampling=True,
            )
        )
        trainer.setup(run_root / "trainer")
        manifest["status"] = "running"
        _write_json(manifest_path, manifest)

        for step in range(start_step, target_step + 1):
            active_step = step
            expected_adapter = None if step == 1 else backend.publisher.adapter_name(step - 1)
            if (
                backend.director_client.policy_version != _theta_version(step - 1)
                or backend.director_client.adapter_name != expected_adapter
                or backend.director_client.expected_server_weight_version
                != current_server_weight_version
            ):
                raise MBPPPlusTTBRunError(
                    "behavior route does not equal the preceding published theta version"
                )
            selected = _select_step_tasks(
                tasks,
                step=step,
                questions_per_batch=7,
            )
            step_root = run_root / "steps" / f"step_{step:06d}"
            _write_jsonl(step_root / "selected_tasks.jsonl", selected)
            started = time.monotonic()
            versions = _versions(config, backend, step)
            admitted, filtered = await _collect_ttb_batch(
                backend,
                selected,
                versions,
                condition_id=str(config["experiment"]["condition_id"]),
                schedule_purpose=str(
                    config["experiment"]["sampling_schedule_purpose"]
                ),
                step=step,
                trajectories_per_question=4,
                max_attempts_per_question=int(
                    config["ttb"]["max_rollout_attempts_per_question"]
                ),
            )
            _write_jsonl(step_root / "admitted_trajectories.jsonl", admitted)
            _write_jsonl(step_root / "filtered_trajectories.jsonl", filtered)
            coordinate = TTBStepCoordinate(
                step=step,
                behavior_theta_version=_theta_version(step - 1),
                updated_theta_version=_theta_version(step),
                behavior_phi_version=_phi_version(step - 1),
                updated_phi_version=_phi_version(step),
                behavior_z_version=_z_version(step - 1),
                updated_z_version=_z_version(step),
                behavior_policy_adapter=expected_adapter,
                behavior_server_weight_version=current_server_weight_version,
            )
            summary = await asyncio.to_thread(
                trainer.train_step,
                admitted,
                coordinate=coordinate,
            )
            old_policy = backend.director_client.policy_version
            old_adapter = backend.director_client.adapter_name
            old_weight = backend.director_client.expected_server_weight_version

            def switch_route(policy_version: str, adapter_name: str) -> None:
                backend.director_client.update_policy_route(
                    policy_version=policy_version,
                    adapter_name=adapter_name,
                    # Discover and bind the exact post-switch value from the
                    # first AgentGraph receipt instead of assuming it remains
                    # equal to the previous SGLang route.
                    expected_server_weight_version=None,
                )

            def restore_route() -> None:
                backend.director_client.update_policy_route(
                    policy_version=old_policy,
                    adapter_name=old_adapter,
                    expected_server_weight_version=old_weight,
                )

            publication = await asyncio.to_thread(
                backend.publisher.publish,
                checkpoint_path=summary.checkpoint_dir,
                checkpoint_version=f"checkpoint:{summary.updated_policy_version}",
                behavior_policy_version=summary.behavior_policy_version,
                candidate_policy_version=summary.updated_policy_version,
                step=step,
                previous_adapter=expected_adapter,
                gate=backend.rollout_gate,
                route_switch=switch_route,
                route_rollback=restore_route,
            )
            _write_json(step_root / "policy_sync_receipt.json", publication.to_dict())
            # The publisher's chat canary proves that SGLang can load the
            # adapter.  TTB's on-policy barrier additionally needs an exact
            # AgentGraph receipt from the post-switch Director route before
            # the trainer is allowed to commit this step.
            explicit_canary = await _post_update_canary(
                backend,
                selected[0],
                config,
                step=step,
                adapter_name=publication.adapter_name,
            )
            _write_jsonl(
                step_root / "post_update_agentgraph_canary.jsonl",
                (explicit_canary,),
            )
            server_weight_versions = {
                turn.server_weight_version for turn in explicit_canary.turns
            }
            if len(server_weight_versions) != 1:
                raise MBPPPlusTTBRunError(
                    "post-update canary has no unique server weight version"
                )
            exact_server_weight_version = next(iter(server_weight_versions))
            if not isinstance(exact_server_weight_version, str) or not (
                exact_server_weight_version.strip()
            ):
                raise MBPPPlusTTBRunError(
                    "post-update canary lacks an exact server weight version"
                )
            _bind_exact_serving_route(
                backend,
                policy_version=summary.updated_policy_version,
                adapter_name=publication.adapter_name,
                server_weight_version=exact_server_weight_version,
            )
            acknowledgement = {
                **publication.to_dict(),
                "server_weight_version": exact_server_weight_version,
            }
            _write_json(
                step_root / "publication_acknowledgement_receipt.json",
                acknowledgement,
            )
            summary = trainer.acknowledge_publication(acknowledgement)
            current_server_weight_version = exact_server_weight_version
            duration = max(time.monotonic() - started, 1.0e-9)
            receipt = _step_receipt(
                summary,
                publication,
                valid_count=len(admitted),
                filtered_count=len(filtered),
                step_seconds=duration,
                gpu_metrics=_gpu_peak_metrics(trainer),
                server_weight_version=exact_server_weight_version,
            )
            _write_json(step_root / "completed_step_receipt.json", receipt)
            metrics = monitor.log_completed_step(receipt)
            manifest["steps"].append(
                {
                    "step": step,
                    "status": "completed",
                    "behavior_theta_version": _theta_version(step - 1),
                    "updated_theta_version": _theta_version(step),
                    "published_adapter": publication.adapter_name,
                    "valid_rollouts": len(admitted),
                    "filtered_rollouts": len(filtered),
                    "reward_mean": metrics.reward_mean,
                    "ttb_loss": metrics.ttb_loss,
                    "delta_squared_mean": metrics.delta_squared_mean,
                    "theta_update_l2": metrics.theta_update_l2,
                    "phi_update_l2": metrics.phi_update_l2,
                    "z_update_l2": metrics.z_update_l2,
                    "recovery_checkpoint": summary.recovery_checkpoint_dir,
                    "official_checkpoint": summary.official_checkpoint_dir or None,
                    "publisher_canary": publication.canary_succeeded,
                    "post_update_agentgraph_canary": True,
                    "publication_acknowledged_by_trainer": True,
                    "server_weight_version": exact_server_weight_version,
                }
            )
            manifest["optimizer_updates_completed"] = (
                int(manifest["optimizer_updates_completed"]) + 1
            )
            manifest["last_committed_step"] = step
            _write_json(manifest_path, manifest)

        manifest["status"] = (
            "completed_smoke_step" if stop_after_step == 1 else "completed"
        )
        manifest["completed_at"] = _utc_now()
        exit_code = 0
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_step"] = active_step
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        if monitor is not None:
            try:
                monitor.log_failure(step=active_step, error=exc)
            except BaseException as monitor_error:
                manifest["wandb_failure_logging_error"] = _safe_error(monitor_error)
                _write_json(manifest_path, manifest)
        raise
    finally:
        if trainer is not None:
            trainer.close()
        if supervisor is not None:
            await asyncio.to_thread(supervisor.stop)
        if monitor is not None:
            monitor.finish(exit_code=exit_code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write a plan and read-only preflight; never start W&B, GPU models, or APIs",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        choices=(1,),
        default=None,
        help="run exactly one real optimizer step and the post-update route canary",
    )
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--continuation-checkpoint", default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = _resolve(PROJECT_ROOT, args.config)
        continuation = (
            _resolve(PROJECT_ROOT, args.continuation_checkpoint)
            if args.continuation_checkpoint
            else None
        )
        result = asyncio.run(
            run_ttb(
                config_path,
                prepare_only=bool(args.prepare_only),
                stop_after_step=args.stop_after_step,
                start_step=int(args.start_step),
                continuation_checkpoint=continuation,
                run_id=_run_id(args.run_id),
                project_root=PROJECT_ROOT,
            )
        )
    except (ConfigurationError, MBPPPlusTTBRunError, ValueError, RuntimeError) as exc:
        print(f"MBPP+ TTB run failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "optimizer_updates_completed": result["optimizer_updates_completed"],
                "last_committed_step": result["last_committed_step"],
                "run_root": result["run_root"],
                "preflight_ready": result["preflight"]["ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
