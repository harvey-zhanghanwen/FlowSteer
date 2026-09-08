"""One-update-per-sealed-batch Qwen3.5 LoRA trainer for AgentGraph rollouts.

Source boundary: Qwen3.5/PEFT loading, trainable adapter continuation, gradient
checkpointing, and the adapter-directory convention are adapted from
SkillFlow.  The one-or-two-physical-GPU gradient layout, group-preserving
token-cost split, optimizer-state persistence, exact behavior receipt checks,
and recoverable step metadata are project engineering required by the MD;
SkillFlow's released checkpoint does not provide those contracts.  The
single-worker option preserves the same objective and is used only when one
physical GPU is available and rollout/training are time-multiplexed.
Terminal-only, action-masked one-pass GRPO is the MD algorithm implemented
from FlowSteer's policy-gradient boundary.  SkillFlow's TTB backward
policy/partition head is disabled.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, Mapping, Sequence, Tuple
import uuid

import numpy as np

from .grpo_objective import (
    GRPOTrajectory,
    same_condition_advantages,
    torch_action_masked_one_pass_loss,
)
from .records import TrajectoryRecord


@dataclass(frozen=True)
class SmokeTrainerConfig:
    model_path: str
    tokenizer_path: str
    behavior_policy_version: str = "qwen35-9b-base-step-0000"
    updated_policy_version: str = "qwen35-9b-smoke-step-0001"
    behavior_policy_adapter: str | None = None
    behavior_adapter_checkpoint: str | None = None
    behavior_server_weight_version: str = "default"
    learner_device: str = "cuda:3"
    gradient_replica_device: str = "cuda:5"
    gradient_worker_count: int = 2
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    seed: int = 42
    training_step: int = 1
    total_training_steps: int = 300
    warmup_steps: int = 0
    scheduler_name: str = "cosine_with_warmup"
    update_step: int = 1
    optimizer_state_checkpoint: str | None = None
    max_grad_norm: float = 1.0
    advantage_epsilon: float = 1.0e-8
    trajectories_per_group: int = 4
    behavior_logprob_tolerance: float = 0.25
    max_sequence_tokens: int = 8192
    gradient_checkpointing: bool = True
    micro_batch_backoff: Tuple[int, ...] = (4, 2, 1)

    def __post_init__(self) -> None:
        required_strings = {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "behavior_policy_version": self.behavior_policy_version,
            "updated_policy_version": self.updated_policy_version,
            "behavior_server_weight_version": self.behavior_server_weight_version,
            "learner_device": self.learner_device,
            "gradient_replica_device": self.gradient_replica_device,
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in required_strings.values()
        ):
            raise ValueError(
                "model/tokenizer paths, policy/server versions, and devices "
                "must be non-empty strings"
            )
        if self.gradient_worker_count not in {1, 2}:
            raise ValueError("gradient_worker_count must be one or two")
        if (
            self.gradient_worker_count == 2
            and self.learner_device == self.gradient_replica_device
        ):
            raise ValueError("learner and gradient replica devices must differ")
        if not self.lora_target_modules or any(
            not isinstance(value, str) or not value.strip()
            for value in self.lora_target_modules
        ):
            raise ValueError("LoRA target modules must be non-empty strings")
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("LoRA target modules must be unique")
        if not self.behavior_policy_version.strip() or not self.updated_policy_version.strip():
            raise ValueError("policy versions must be non-empty")
        if self.behavior_policy_version == self.updated_policy_version:
            raise ValueError("updated policy version must differ from behavior policy")
        if not self.behavior_server_weight_version.strip():
            raise ValueError("behavior server weight version must be non-empty")
        if self.behavior_policy_adapter is not None and not self.behavior_policy_adapter.strip():
            raise ValueError("behavior policy adapter must be non-empty when supplied")
        optional_paths = {
            "behavior_adapter_checkpoint": self.behavior_adapter_checkpoint,
            "optimizer_state_checkpoint": self.optimizer_state_checkpoint,
        }
        for name, value in optional_paths.items():
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be non-empty when supplied")
        if type(self.update_step) is not int or self.update_step <= 0:
            raise ValueError("update_step must be a positive integer")
        if type(self.training_step) is not int or self.training_step <= 0:
            raise ValueError("training_step must be a positive integer")
        if (
            type(self.total_training_steps) is not int
            or self.total_training_steps <= 0
            or self.training_step > self.total_training_steps
        ):
            raise ValueError(
                "total_training_steps must cover the current training_step"
            )
        if (
            type(self.warmup_steps) is not int
            or self.warmup_steps < 0
            or self.warmup_steps >= self.total_training_steps
        ):
            raise ValueError(
                "warmup_steps must be non-negative and below total_training_steps"
            )
        if self.scheduler_name != "cosine_with_warmup":
            raise ValueError("scheduler_name must be cosine_with_warmup")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.update_step > 1 and self.behavior_adapter_checkpoint is None:
            raise ValueError("step 2+ requires a behavior adapter checkpoint")
        if (
            self.optimizer_state_checkpoint is not None
            and self.behavior_adapter_checkpoint is None
        ):
            raise ValueError(
                "optimizer continuation requires a behavior adapter checkpoint"
            )
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        numeric_settings = (
            self.learning_rate,
            self.weight_decay,
            self.max_grad_norm,
            self.advantage_epsilon,
            self.behavior_logprob_tolerance,
            self.lora_dropout,
        )
        if not all(math.isfinite(float(value)) for value in numeric_settings):
            raise ValueError("optimizer, GRPO, and LoRA settings must be finite")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.max_grad_norm <= 0
        ):
            raise ValueError("optimizer settings must be positive")
        if self.advantage_epsilon <= 0 or self.behavior_logprob_tolerance <= 0:
            raise ValueError("GRPO tolerances must be positive")
        if type(self.trajectories_per_group) is not int or self.trajectories_per_group < 2:
            raise ValueError("trajectories_per_group must be an integer of at least two")
        if self.max_sequence_tokens <= 0:
            raise ValueError("max_sequence_tokens must be positive")
        if (
            not isinstance(self.micro_batch_backoff, tuple)
            or not self.micro_batch_backoff
            or any(type(value) is not int for value in self.micro_batch_backoff)
            or self.micro_batch_backoff[-1] != 1
            or any(value <= 0 for value in self.micro_batch_backoff)
            or any(
                left <= right
                for left, right in zip(
                    self.micro_batch_backoff,
                    self.micro_batch_backoff[1:],
                )
            )
        ):
            raise ValueError(
                "micro-batch backoff must be an integer tuple that is positive, "
                "strictly decreasing, and ends at one"
            )
        # OOM backoff restarts both partitions even when gradient checkpointing
        # is disabled.  Non-zero LoRA dropout would therefore require saving
        # and restoring the CUDA RNG state on both physical training GPUs.
        if self.lora_dropout != 0.0:
            raise ValueError(
                "OOM backoff requires zero LoRA dropout for deterministic replay"
            )


@dataclass(frozen=True)
class SmokeTrainingSummary:
    optimizer_updates: int
    input_trajectories: int
    record_eligible_trajectories: int
    exact_groups: int
    informative_groups: int
    trained_groups: int
    trained_trajectories: int
    zero_information_groups: int
    excluded_groups: int
    loss: float
    grad_norm: float
    max_behavior_logprob_delta: float
    behavior_policy_version: str
    updated_policy_version: str
    micro_batch_size_used: int
    oom_backoff_count: int
    trainable_update_l2: float
    checkpoint_dir: str
    exclusions: Mapping[str, str]
    continuation_adapter_checkpoint: str = ""
    continuation_loaded: bool = False
    update_step: int = 0
    committed_step: int = 0
    optimizer_resume_status: str = "not_started"
    optimizer_state_checkpoint: str = ""
    optimizer_state_saved: bool = False
    training_state_checkpoint: str = ""
    training_state_saved: bool = False
    scheduler_state_saved: bool = False
    rng_state_saved: bool = False
    checkpoint_recoverable: bool = False
    scheduler_resume_status: str = "not_started"
    learning_rate: float = 0.0
    next_learning_rate: float = 0.0
    gpu_memory_allocated_mib: Mapping[str, float] = field(default_factory=dict)
    cuda_rng_restore_device_map: Mapping[str, str] = field(default_factory=dict)
    gradient_partition_token_costs: Tuple[int, ...] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _flatten_receipt(record: TrajectoryRecord) -> tuple[list[float], list[int]]:
    log_probs: list[float] = []
    action_mask: list[int] = []
    for turn in record.turns:
        log_probs.extend(float(value) for value in turn.behavior_log_probs)
        action_mask.extend(
            1 if index < turn.executed_prefix_tokens else 0
            for index in range(len(turn.output_token_ids))
        )
    return log_probs, action_mask


def trajectory_to_grpo(record: TrajectoryRecord) -> GRPOTrajectory:
    """Project an immutable rollout record onto the existing GRPO contract."""

    log_probs, action_mask = _flatten_receipt(record)
    return GRPOTrajectory(
        trajectory_id=record.trajectory_id,
        task_id=record.task.task_id,
        condition_id=record.condition_id,
        # The complete version bundle, not just a display policy label, is the
        # behavior-policy identity used by TrajectoryRecord.group_key.
        policy_version=record.versions.fingerprint,
        terminal_reward=float(record.evaluation.reward or 0.0),
        token_log_probs=tuple(log_probs),
        action_mask=tuple(action_mask),
        evaluator_valid=record.evaluation.valid,
        explicit_finish=record.explicit_finish,
        forced_probe=record.forced_probe,
        fallback_or_manual_repair=(
            record.api_fallback_used or record.manual_repair_used
        ),
        reconstructed_context=any(turn.reconstructed_context for turn in record.turns),
        exact_receipt_verified=all(turn.receipt_verified for turn in record.turns),
    )


def _partition_groups_by_token_cost(
    groups: Sequence[tuple[tuple[str, str, str], list[GRPOTrajectory]]],
    records_by_id: Mapping[str, TrajectoryRecord],
    worker_count: int = 2,
) -> tuple[
    list[list[tuple[tuple[str, str, str], list[GRPOTrajectory]]]],
    tuple[int, ...],
]:
    """Greedily balance complete exact groups by sampled token cost.

    SkillFlow's released ``_batched_logprob_backward`` uses a midpoint split;
    it does not publish a token-cost partitioner.  This group-preserving greedy
    split is project engineering required to keep each exact GRPO group atomic
    while balancing the two training GPUs.  Empty bins are allowed when fewer
    exact groups than physical replicas are available.
    """

    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker_count must be positive")
    indexed: list[
        tuple[
            int,
            tuple[tuple[str, str, str], list[GRPOTrajectory]],
            int,
        ]
    ] = []
    for original_index, group_entry in enumerate(groups):
        _, group = group_entry
        token_cost = sum(
            len(turn.prompt_token_ids) + len(turn.output_token_ids)
            for item in group
            for turn in records_by_id[item.trajectory_id].turns
        )
        indexed.append((original_index, group_entry, max(token_cost, 1)))
    indexed.sort(key=lambda entry: (-entry[2], entry[0]))
    bins: list[
        list[
            tuple[
                int,
                tuple[tuple[str, str, str], list[GRPOTrajectory]],
            ]
        ]
    ] = [[] for _ in range(worker_count)]
    costs = [0] * worker_count
    for original_index, group_entry, token_cost in indexed:
        target = min(range(worker_count), key=lambda index: (costs[index], index))
        bins[target].append((original_index, group_entry))
        costs[target] += token_cost
    partitions = [
        [entry for _, entry in sorted(worker_items, key=lambda pair: pair[0])]
        for worker_items in bins
    ]
    return partitions, tuple(costs)


class Qwen35OnePassSmokeTrainer:
    """Run exactly one optimizer update over frozen-policy smoke trajectories."""

    def __init__(self, config: SmokeTrainerConfig) -> None:
        self.config = config
        self.cuda_rng_restore_device_map: dict[str, str] = {}

    def train(
        self,
        trajectories: Sequence[TrajectoryRecord],
        output_dir: str | Path,
    ) -> SmokeTrainingSummary:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records = list(trajectories)
        observed_policy_versions = {item.versions.policy for item in records}
        if observed_policy_versions != {self.config.behavior_policy_version}:
            raise ValueError(
                "smoke batch must contain exactly the configured frozen behavior "
                f"policy {self.config.behavior_policy_version!r}; observed "
                f"{sorted(observed_policy_versions)!r}"
            )
        observed_turn_policy_versions = {
            turn.policy_version for record in records for turn in record.turns
        }
        if observed_turn_policy_versions != {self.config.behavior_policy_version}:
            raise ValueError(
                "smoke batch turn receipts must contain exactly the configured "
                f"logical behavior policy {self.config.behavior_policy_version!r}; "
                f"observed {sorted(observed_turn_policy_versions)!r}"
            )
        expected_route = (
            self.config.behavior_policy_adapter,
            self.config.behavior_server_weight_version,
        )
        observed_routes = {
            (turn.policy_adapter, turn.server_weight_version)
            for record in records
            for turn in record.turns
        }
        if observed_routes != {expected_route}:
            raise ValueError(
                "smoke batch must contain one exact behavior route receipt "
                f"{expected_route!r}; observed {sorted(observed_routes, key=repr)!r}"
            )
        projected = {item.trajectory_id: trajectory_to_grpo(item) for item in records}
        records_by_id = {item.trajectory_id: item for item in records}

        exact_groups: dict[tuple[str, str, str], list[GRPOTrajectory]] = {}
        for record in records:
            item = projected[record.trajectory_id]
            # TrajectoryRecord additionally checks split isolation, immutable
            # graph-snapshot continuity, and exact evaluator/policy versions.
            if record.grpo_eligible and item.eligible:
                exact_groups.setdefault(item.group_key, []).append(item)

        advantages: dict[str, float] = {}
        informative: list[tuple[tuple[str, str, str], list[GRPOTrajectory]]] = []
        zero_groups = 0
        exclusions: dict[str, str] = {}
        for key, group in sorted(exact_groups.items()):
            if len(group) != self.config.trajectories_per_group:
                exclusions["|".join(key)] = "incomplete_exact_group"
                continue
            values = same_condition_advantages(group, self.config.advantage_epsilon)
            for item, value in zip(group, values):
                advantages[item.trajectory_id] = float(value)
            if len(group) < 2 or not any(float(value) != 0.0 for value in values):
                zero_groups += 1
                exclusions["|".join(key)] = "zero_information_group"
                continue
            too_long = [
                item.trajectory_id
                for item in group
                if any(
                    len(turn.prompt_token_ids) + len(turn.output_token_ids)
                    > self.config.max_sequence_tokens
                    for turn in records_by_id[item.trajectory_id].turns
                )
            ]
            if too_long:
                exclusions["|".join(key)] = "sequence_exceeds_training_limit"
                continue
            informative.append((key, group))

        self._write_batch(
            output_path / "grpo_batch.jsonl",
            records,
            projected,
            advantages,
        )

        if not informative:
            summary = SmokeTrainingSummary(
                optimizer_updates=0,
                input_trajectories=len(records),
                record_eligible_trajectories=sum(item.grpo_eligible for item in records),
                exact_groups=len(exact_groups),
                informative_groups=0,
                trained_groups=0,
                trained_trajectories=0,
                zero_information_groups=zero_groups,
                excluded_groups=len(exclusions),
                loss=0.0,
                grad_norm=0.0,
                max_behavior_logprob_delta=0.0,
                behavior_policy_version=self.config.behavior_policy_version,
                updated_policy_version="",
                micro_batch_size_used=0,
                oom_backoff_count=0,
                trainable_update_l2=0.0,
                checkpoint_dir="",
                exclusions=exclusions,
                continuation_adapter_checkpoint=(
                    self.config.behavior_adapter_checkpoint or ""
                ),
                update_step=self.config.update_step,
                committed_step=(
                    self.config.update_step - 1
                    if self.config.behavior_adapter_checkpoint is not None
                    else 0
                ),
            )
            self._write_summary(output_path, summary)
            return summary

        torch, learner, replica = self._load_models()
        models = [learner] if replica is None else [learner, replica]
        devices = [self.config.learner_device]
        if replica is not None:
            devices.append(self.config.gradient_replica_device)
        group_partitions, _ = _partition_groups_by_token_cost(
            informative,
            records_by_id,
            worker_count=len(models),
        )
        if torch.cuda.is_available():
            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
        named_trainable = [
            (name, parameter)
            for name, parameter in learner.named_parameters()
            if parameter.requires_grad
        ]
        unexpected_trainable = [
            name for name, _ in named_trainable if "lora_" not in name
        ]
        if not named_trainable or unexpected_trainable:
            raise RuntimeError(
                "the Director optimizer must contain theta LoRA parameters only; "
                f"unexpected={unexpected_trainable!r}"
            )
        trainable = [parameter for _, parameter in named_trainable]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        try:
            from transformers import get_cosine_schedule_with_warmup
        except ImportError as exc:  # pragma: no cover - heavy runtime only
            raise RuntimeError(
                "transformers is required for the FlowSteer cosine scheduler"
            ) from exc
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.total_training_steps,
        )
        optimizer_resume_status, scheduler_resume_status = (
            self._restore_training_state(torch, optimizer, scheduler)
        )

        try:
            with ThreadPoolExecutor(max_workers=len(models)) as pool:
                futures = [
                    pool.submit(
                        self._preflight_partition,
                        model,
                        device,
                        partition,
                        records_by_id,
                    )
                    for model, device, partition in zip(models, devices, group_partitions)
                ]
                preflight = [future.result() for future in futures]

            accepted_keys: set[tuple[str, str, str]] = set()
            max_delta = 0.0
            for partition_result in preflight:
                for key, result in partition_result.items():
                    max_delta = max(max_delta, float(result[0]))
                    if result[1]:
                        accepted_keys.add(key)
                    else:
                        exclusions["|".join(key)] = result[2]

            accepted_groups = [
                (key, group)
                for key, group in informative
                if key in accepted_keys
            ]
            # Receipt rejection seals the final gradient batch.  Re-run the
            # project token-cost partitioner so the actual backward work,
            # rather than the larger preflight candidate set, is balanced.
            accepted_partitions, partition_token_costs = (
                _partition_groups_by_token_cost(
                    accepted_groups,
                    records_by_id,
                    worker_count=len(models),
                )
            )
            trained_group_count = sum(len(partition) for partition in accepted_partitions)
            if trained_group_count == 0:
                summary = SmokeTrainingSummary(
                    optimizer_updates=0,
                    input_trajectories=len(records),
                    record_eligible_trajectories=sum(
                        item.grpo_eligible for item in records
                    ),
                    exact_groups=len(exact_groups),
                    informative_groups=len(informative),
                    trained_groups=0,
                    trained_trajectories=0,
                    zero_information_groups=zero_groups,
                    excluded_groups=len(exclusions),
                    loss=0.0,
                    grad_norm=0.0,
                    max_behavior_logprob_delta=max_delta,
                    behavior_policy_version=self.config.behavior_policy_version,
                    updated_policy_version="",
                    micro_batch_size_used=0,
                    oom_backoff_count=0,
                    trainable_update_l2=0.0,
                    checkpoint_dir="",
                    exclusions=exclusions,
                    continuation_adapter_checkpoint=(
                        self.config.behavior_adapter_checkpoint or ""
                    ),
                    continuation_loaded=(
                        self.config.behavior_adapter_checkpoint is not None
                    ),
                    update_step=self.config.update_step,
                    committed_step=(
                        self.config.update_step - 1
                        if self.config.behavior_adapter_checkpoint is not None
                        else 0
                    ),
                    gradient_partition_token_costs=partition_token_costs,
                )
                self._write_summary(output_path, summary)
                return summary

            learner.train()
            if replica is not None:
                replica.train()
            partition_stats = None
            micro_batch_size_used = 0
            oom_backoff_count = 0
            for micro_batch_size in self.config.micro_batch_backoff:
                for model in models:
                    model.zero_grad(set_to_none=True)
                with ThreadPoolExecutor(max_workers=len(models)) as pool:
                    futures = [
                        pool.submit(
                            self._backward_partition,
                            model,
                            device,
                            partition,
                            records_by_id,
                            advantages,
                            trained_group_count,
                            micro_batch_size,
                        )
                        for model, device, partition in zip(
                            models, devices, accepted_partitions
                        )
                    ]
                    results: list[tuple[float, int] | None] = []
                    failures: list[BaseException] = []
                    for future in futures:
                        try:
                            results.append(future.result())
                        except BaseException as exc:
                            results.append(None)
                            failures.append(exc)
                if not failures:
                    partition_stats = [result for result in results if result is not None]
                    micro_batch_size_used = micro_batch_size
                    break
                non_oom = [
                    exc
                    for exc in failures
                    if not isinstance(exc, RuntimeError)
                    or not self._is_cuda_oom(torch, exc)
                ]
                if non_oom:
                    raise non_oom[0]
                oom_backoff_count += 1
                for model in models:
                    model.zero_grad(set_to_none=True)
                self._empty_device_caches(torch, devices)
            if partition_stats is None:
                raise RuntimeError(
                    "gradient computation exhausted the configured 4->2->1 "
                    "micro-batch schedule"
                )

            if replica is not None:
                self._merge_replica_grads(learner, replica)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                trainable, self.config.max_grad_norm
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
            if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise RuntimeError("informative GRPO batch produced no finite parameter gradient")
            before_step = [parameter.detach().clone() for parameter in trainable]
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(learning_rate_used) or learning_rate_used <= 0.0:
                raise RuntimeError("optimizer step requires a finite positive learning rate")
            optimizer.step()
            scheduler.step()
            update_l2_sq = sum(
                float(
                    (parameter.detach().float() - before.detach().float())
                    .square()
                    .sum()
                    .cpu()
                )
                for parameter, before in zip(trainable, before_step)
            )
            trainable_update_l2 = math.sqrt(update_l2_sq)
            del before_step
            if not math.isfinite(trainable_update_l2) or trainable_update_l2 <= 0.0:
                raise RuntimeError("optimizer.step() did not change the Director LoRA weights")
            optimizer.zero_grad(set_to_none=True)

            attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            checkpoint_root = (
                output_path
                / "checkpoint_final"
                / "supervisor_lora"
                / f"step_{self.config.update_step:06d}_{attempt_id}"
            )
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            learner.set_adapter("theta")
            learner.save_pretrained(
                checkpoint_root,
                selected_adapters=["theta"],
                safe_serialization=True,
            )
            # PEFT stores a non-default named adapter in a child directory.
            # SGLang's path loader must receive the directory containing both
            # adapter_config.json and adapter_model.safetensors.
            checkpoint = checkpoint_root / "theta"
            required_checkpoint_files = (
                checkpoint / "adapter_config.json",
                checkpoint / "adapter_model.safetensors",
            )
            if not all(path.is_file() for path in required_checkpoint_files):
                raise RuntimeError(
                    "PEFT did not materialize a complete theta adapter checkpoint"
                )
            training_state_checkpoint, training_state_saved = (
                self._save_training_state(
                    torch,
                    optimizer,
                    scheduler,
                    checkpoint,
                    records,
                    learning_rate_used=learning_rate_used,
                    next_learning_rate=float(scheduler.get_last_lr()[0]),
                )
            )
            # Retain the historical field names as aliases so existing runner
            # state can resolve the same complete FlowSteer training state.
            optimizer_state_checkpoint = training_state_checkpoint
            optimizer_state_saved = training_state_saved
            next_learning_rate = float(scheduler.get_last_lr()[0])
            (checkpoint / "policy_version.json").write_text(
                json.dumps(
                    {
                        "behavior_policy_version": self.config.behavior_policy_version,
                        "updated_policy_version": self.config.updated_policy_version,
                        "optimizer_updates": 1,
                        "optimizer_updates_this_run": 1,
                        "committed_step": self.config.update_step,
                        "continuation_adapter_checkpoint": (
                            self.config.behavior_adapter_checkpoint
                        ),
                        "optimizer_resume_status": optimizer_resume_status,
                        "optimizer_state_checkpoint": optimizer_state_checkpoint,
                        "optimizer_state_saved": optimizer_state_saved,
                        "training_state_checkpoint": training_state_checkpoint,
                        "training_state_saved": training_state_saved,
                        "scheduler_state_saved": training_state_saved,
                        "rng_state_saved": training_state_saved,
                        "cuda_rng_restore_device_map": dict(
                            self.cuda_rng_restore_device_map
                        ),
                        "checkpoint_recoverable": training_state_saved,
                        "scheduler_resume_status": scheduler_resume_status,
                        "learning_rate": learning_rate_used,
                        "next_learning_rate": next_learning_rate,
                        "trainable_update_l2": trainable_update_l2,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            total_loss = sum(item[0] for item in partition_stats)
            trained_trajectories = sum(item[1] for item in partition_stats)
            gpu_memory_allocated_mib = self._gpu_memory_allocated_mib(
                torch,
                devices,
            )
            summary = SmokeTrainingSummary(
                optimizer_updates=1,
                input_trajectories=len(records),
                record_eligible_trajectories=sum(item.grpo_eligible for item in records),
                exact_groups=len(exact_groups),
                informative_groups=len(informative),
                trained_groups=trained_group_count,
                trained_trajectories=trained_trajectories,
                zero_information_groups=zero_groups,
                excluded_groups=len(exclusions),
                loss=float(total_loss),
                grad_norm=grad_norm,
                max_behavior_logprob_delta=max_delta,
                behavior_policy_version=self.config.behavior_policy_version,
                updated_policy_version=self.config.updated_policy_version,
                micro_batch_size_used=micro_batch_size_used,
                oom_backoff_count=oom_backoff_count,
                trainable_update_l2=trainable_update_l2,
                checkpoint_dir=str(checkpoint),
                exclusions=exclusions,
                continuation_adapter_checkpoint=(
                    self.config.behavior_adapter_checkpoint or ""
                ),
                continuation_loaded=(
                    self.config.behavior_adapter_checkpoint is not None
                ),
                update_step=self.config.update_step,
                committed_step=self.config.update_step,
                optimizer_resume_status=optimizer_resume_status,
                optimizer_state_checkpoint=optimizer_state_checkpoint,
                optimizer_state_saved=optimizer_state_saved,
                training_state_checkpoint=training_state_checkpoint,
                training_state_saved=training_state_saved,
                scheduler_state_saved=training_state_saved,
                rng_state_saved=training_state_saved,
                checkpoint_recoverable=training_state_saved,
                scheduler_resume_status=scheduler_resume_status,
                learning_rate=learning_rate_used,
                next_learning_rate=next_learning_rate,
                gpu_memory_allocated_mib=gpu_memory_allocated_mib,
                cuda_rng_restore_device_map=dict(self.cuda_rng_restore_device_map),
                gradient_partition_token_costs=partition_token_costs,
            )
            self._write_summary(output_path, summary)
            return summary
        finally:
            del learner
            if replica is not None:
                del replica
            self._empty_device_caches(torch, tuple(dict.fromkeys(devices)))

    def _load_models(self):
        try:
            import torch
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:  # pragma: no cover - heavy runtime only
            raise RuntimeError(
                "Qwen smoke training requires torch, transformers, and peft"
            ) from exc

        def load(device: str):
            # Qwen3.5 requires its conditional-generation loader in this
            # environment.  This is a model-compatibility adaptation: the
            # released SkillFlow trainer uses AutoModelForCausalLM.
            base = AutoModelForMultimodalLM.from_pretrained(
                self.config.model_path,
                dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self.config.behavior_adapter_checkpoint is not None:
                # The MD requires both physical gradient replicas to attach
                # the same frozen behavior theta adapter before receipt
                # preflight.  SkillFlow reloads theta on its shared model but
                # does not publish this exact two-replica continuation gate.
                model = PeftModel.from_pretrained(
                    base,
                    self.config.behavior_adapter_checkpoint,
                    adapter_name="theta",
                    is_trainable=True,
                )
            else:
                # PEFT mutates parts of its config while attaching an adapter,
                # so each physical replica receives its own independent copy.
                lora_config = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=list(self.config.lora_target_modules),
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(base, lora_config, adapter_name="theta")
            model.set_adapter("theta")
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                model.enable_input_require_grads()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            return model

        learner = load(self.config.learner_device)
        replica = None
        if self.config.gradient_worker_count == 2:
            replica = load(self.config.gradient_replica_device)
            self._sync_lora_weights(learner, replica)
        return torch, learner, replica

    def _restore_training_state(self, torch, optimizer, scheduler) -> tuple[str, str]:
        """Restore the complete FlowSteer optimizer/scheduler/RNG boundary."""

        self.cuda_rng_restore_device_map = {}
        state_checkpoint = self.config.optimizer_state_checkpoint
        if state_checkpoint is None:
            self._seed_training_rng(torch)
            if (
                self.config.behavior_adapter_checkpoint is not None
                and self.config.update_step > 1
            ):
                # The Round-01 adapter predates complete training-state
                # persistence.  It is the explicit starting policy for this
                # run, never reported as an exact optimizer continuation.
                return "warm_start_fresh_optimizer", "fresh_scheduler_seeded_rng"
            return "fresh_optimizer", "fresh_scheduler_seeded_rng"

        payload = torch.load(
            state_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("training checkpoint payload must be a mapping")
        if payload.get("format") != "flowsteer-training-state-v3":
            raise ValueError("training checkpoint format differs")
        committed_step = payload.get("committed_step")
        if committed_step != self.config.update_step - 1:
            raise ValueError(
                "training checkpoint committed step must immediately precede "
                "update_step"
            )
        committed_training_step = payload.get("committed_training_step")
        if committed_training_step != self.config.training_step - 1:
            raise ValueError(
                "training checkpoint training step must immediately precede "
                "training_step"
            )
        if payload.get("updated_policy_version") != self.config.behavior_policy_version:
            raise ValueError(
                "training checkpoint policy does not match the current behavior policy"
            )
        saved_contract = payload.get("training_contract")
        if not isinstance(saved_contract, Mapping):
            raise ValueError("training checkpoint is missing training_contract")
        current_contract = self._training_contract()
        if dict(saved_contract) != current_contract:
            differing = sorted(
                key
                for key in set(saved_contract) | set(current_contract)
                if saved_contract.get(key) != current_contract.get(key)
            )
            raise ValueError(
                "training checkpoint contract differs: " + ", ".join(differing)
            )
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("training checkpoint is missing optimizer_state_dict")
        scheduler_state = payload.get("scheduler_state_dict")
        if not isinstance(scheduler_state, Mapping):
            raise ValueError("training checkpoint is missing scheduler_state_dict")
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        self._restore_rng_state(torch, payload)
        return "restored_optimizer", "restored_scheduler_and_rng"

    def _seed_training_rng(self, torch) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            for device in self._training_devices():
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(self.config.seed)

    def _restore_rng_state(self, torch, payload: Mapping[str, Any]) -> None:
        required = (
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state_by_device",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "training checkpoint is missing RNG state: " + ", ".join(missing)
            )
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        cpu_rng_state = payload["torch_cpu_rng_state"]
        if hasattr(cpu_rng_state, "cpu"):
            cpu_rng_state = cpu_rng_state.cpu()
        torch.set_rng_state(cpu_rng_state)
        cuda_states = payload["torch_cuda_rng_state_by_device"]
        if not isinstance(cuda_states, Mapping):
            raise ValueError("training checkpoint CUDA RNG state must be a mapping")
        if torch.cuda.is_available():
            expected_devices = set(self._training_devices())
            saved_devices = set(cuda_states)
            if any(
                not isinstance(device, str)
                or re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is None
                for device in saved_devices | expected_devices
            ):
                raise ValueError("training checkpoint CUDA RNG device key is invalid")
            if saved_devices == expected_devices:
                device_map = {device: device for device in saved_devices}
            elif (
                self.config.gradient_worker_count == 1
                and len(saved_devices) == len(expected_devices) == 1
            ):
                # Project resource-migration adaptation of FlowSteer's RNG
                # restore boundary: a single learner has an unambiguous role.
                # Keep its exact saved stream when changing physical GPU;
                # never reseed or infer a multi-replica device permutation.
                device_map = {
                    next(iter(saved_devices)): self.config.learner_device
                }
            else:
                raise ValueError("training checkpoint CUDA RNG devices differ")
            for device, state in cuda_states.items():
                if hasattr(state, "cpu"):
                    state = state.cpu()
                torch.cuda.set_rng_state(state, device=device_map[device])
            self.cuda_rng_restore_device_map = device_map

    def _save_training_state(
        self,
        torch,
        optimizer,
        scheduler,
        checkpoint: Path,
        records: Sequence[TrajectoryRecord],
        *,
        learning_rate_used: float,
        next_learning_rate: float,
    ) -> tuple[str, bool]:
        """Atomically save the exact state required to resume the next step.

        FlowSteer's ``train_interactive.py`` supplies the optimizer, cosine
        scheduler, and Python/NumPy/PyTorch RNG checkpoint boundary.  This
        AgentGraph adaptation limits CUDA RNG capture to the two GPUs owned by
        this trainer and adds the frozen rollout version metadata required by
        the project MD.
        """

        cuda_states: dict[str, Any] = {}
        if torch.cuda.is_available():
            for device in self._training_devices():
                cuda_states[device] = torch.cuda.get_rng_state(device)
        state_path = checkpoint / "training_state.pt"
        temporary = checkpoint / f".training_state.pt.tmp-{uuid.uuid4().hex}"
        payload = {
            "format": "flowsteer-training-state-v3",
            "committed_step": self.config.update_step,
            "committed_training_step": self.config.training_step,
            "behavior_policy_version": self.config.behavior_policy_version,
            "updated_policy_version": self.config.updated_policy_version,
            "behavior_policy_adapter": self.config.behavior_policy_adapter,
            "behavior_server_weight_version": (
                self.config.behavior_server_weight_version
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_name": self.config.scheduler_name,
            "training_contract": self._training_contract(),
            "scheduler_state_dict": scheduler.state_dict(),
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": next_learning_rate,
            "seed": self.config.seed,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_by_device": cuda_states,
            "cuda_rng_restore_device_map": dict(self.cuda_rng_restore_device_map),
            "frozen_version_fingerprints": sorted(
                {record.versions.fingerprint for record in records}
            ),
            "condition_ids": sorted({record.condition_id for record in records}),
            "task_ids": sorted({record.task.task_id for record in records}),
            "trajectory_ids": [record.trajectory_id for record in records],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, state_path)
        return str(state_path), True

    def _training_contract(self) -> dict[str, Any]:
        """Return immutable optimizer, schedule, and theta-LoRA settings.

        FlowSteer restores optimizer/scheduler state together with the model.
        This MD-required adaptation also rejects continuation under a changed
        optimization or adapter contract so that a checkpoint cannot silently
        change the next policy update.
        """

        return {
            "scheduler_name": self.config.scheduler_name,
            "total_training_steps": self.config.total_training_steps,
            "warmup_steps": self.config.warmup_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target_modules": list(self.config.lora_target_modules),
            "gradient_worker_count": self.config.gradient_worker_count,
        }

    def _training_devices(self) -> tuple[str, ...]:
        if self.config.gradient_worker_count == 1:
            return (self.config.learner_device,)
        return (
            self.config.learner_device,
            self.config.gradient_replica_device,
        )

    @staticmethod
    def _is_cuda_oom(torch, error: RuntimeError) -> bool:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        return isinstance(error, oom_type) or "out of memory" in str(error).lower()

    @staticmethod
    def _gpu_memory_allocated_mib(
        torch,
        devices: Sequence[str],
    ) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            device: float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            for device in devices
        }

    @staticmethod
    def _empty_device_caches(torch, devices: Sequence[str]) -> None:
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    @staticmethod
    def _sync_lora_weights(learner, replica) -> None:
        source = {
            name: parameter.detach()
            for name, parameter in learner.named_parameters()
            if "lora_" in name
        }
        for name, parameter in replica.named_parameters():
            if name in source:
                parameter.data.copy_(source[name].to(parameter.device))

    @staticmethod
    def _merge_replica_grads(learner, replica) -> None:
        replica_grads = {
            name: parameter.grad.detach()
            for name, parameter in replica.named_parameters()
            if parameter.grad is not None and "lora_" in name
        }
        for name, parameter in learner.named_parameters():
            gradient = replica_grads.get(name)
            if gradient is None:
                continue
            incoming = gradient.to(parameter.device)
            if parameter.grad is None:
                parameter.grad = incoming.clone()
            else:
                parameter.grad.add_(incoming)

    def _preflight_partition(
        self,
        model,
        device: str,
        partition,
        records_by_id: Mapping[str, TrajectoryRecord],
    ):
        import torch

        model.eval()
        results = {}
        with torch.no_grad():
            for key, group in partition:
                group_max = 0.0
                accepted = True
                reason = ""
                for item in group:
                    record = records_by_id[item.trajectory_id]
                    for turn in record.turns:
                        action_tokens = turn.executed_prefix_tokens
                        if (
                            type(action_tokens) is not int
                            or action_tokens <= 0
                            or action_tokens > len(turn.output_token_ids)
                        ):
                            accepted = False
                            reason = "invalid_executed_action_span"
                            break
                        computed = self._turn_log_probs(model, device, turn)
                        behavior = torch.tensor(
                            list(turn.behavior_log_probs),
                            dtype=torch.float32,
                            device=computed.device,
                        )
                        if computed.shape != behavior.shape:
                            accepted = False
                            reason = "behavior_receipt_shape_mismatch"
                            break
                        # FlowSteer's action mask and SkillFlow's action
                        # teacher-forcing both restrict optimization to the
                        # executed action.  Preserve the complete receipt shape
                        # check above, but do not reject a group for unused
                        # sampled suffix tokens outside the backward mask.
                        computed = computed[:action_tokens]
                        behavior = behavior[:action_tokens]
                        if computed.numel():
                            delta = float(
                                (computed.detach().float() - behavior).abs().max().cpu()
                            )
                            group_max = max(group_max, delta)
                            if not math.isfinite(delta) or (
                                delta > self.config.behavior_logprob_tolerance
                            ):
                                accepted = False
                                reason = "behavior_logprob_tolerance_exceeded"
                                break
                    if not accepted:
                        break
                results[key] = (group_max, accepted, reason)
        return results

    def _backward_partition(
        self,
        model,
        device: str,
        partition,
        records_by_id: Mapping[str, TrajectoryRecord],
        advantages: Mapping[str, float],
        total_groups: int,
        micro_batch_size: int,
    ) -> tuple[float, int]:
        import torch

        total_loss = 0.0
        trained = 0
        items = [
            (item, len(group))
            for _, group in partition
            for item in group
        ]
        for start in range(0, len(items), micro_batch_size):
            batch_terms = []
            batch_items = items[start : start + micro_batch_size]
            for item, group_size in batch_items:
                record = records_by_id[item.trajectory_id]
                turn_log_probs = []
                turn_masks = []
                for turn in record.turns:
                    values = self._turn_log_probs(model, device, turn)
                    mask = torch.tensor(
                        [
                            1 if index < turn.executed_prefix_tokens else 0
                            for index in range(len(turn.output_token_ids))
                        ],
                        dtype=torch.long,
                        device=values.device,
                    )
                    turn_log_probs.append(values)
                    turn_masks.append(mask)
                if not turn_log_probs:
                    continue
                flat_log_probs = torch.cat(turn_log_probs)
                flat_mask = torch.cat(turn_masks)
                term = torch_action_masked_one_pass_loss(
                    [flat_log_probs],
                    [flat_mask],
                    [advantages[item.trajectory_id]],
                )
                scaled = term / float(group_size * total_groups)
                total_loss += float(scaled.detach().float().cpu())
                trained += 1
                batch_terms.append(scaled)
                del flat_log_probs, flat_mask, term, scaled
            if batch_terms:
                torch.stack(batch_terms).sum().backward()
            del batch_terms
        return total_loss, trained

    @staticmethod
    def _turn_log_probs(model, device: str, turn):
        import torch

        prompt_ids = list(turn.prompt_token_ids)
        output_ids = list(turn.output_token_ids)
        if not prompt_ids or not output_ids:
            return torch.empty(0, dtype=torch.float32, device=device)
        full_ids = prompt_ids + output_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        # Qwen3.5 can project only the final action positions.  Keeping one
        # additional position supplies the predictor for the first output token.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=len(output_ids) + 1,
            )
            logits = outputs.logits[0]
            action_logits = logits[-(len(output_ids) + 1) : -1]
            targets = input_ids[0, -len(output_ids) :]
            log_probs = torch.log_softmax(action_logits.float(), dim=-1)
            selected = log_probs[
                torch.arange(len(output_ids), device=targets.device),
                targets,
            ]
        del outputs, logits, action_logits, log_probs, input_ids, attention_mask
        return selected

    @staticmethod
    def _write_batch(
        path: Path,
        records: Sequence[TrajectoryRecord],
        projected: Mapping[str, GRPOTrajectory],
        advantages: Mapping[str, float],
    ) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                item = projected[record.trajectory_id]
                _, mask = _flatten_receipt(record)
                row = {
                    "trajectory_id": record.trajectory_id,
                    "task_id": record.task.task_id,
                    "source": record.task.metadata.get("source", "unknown"),
                    "condition_id": record.condition_id,
                    "policy_group_version": record.versions.fingerprint,
                    "policy_adapter": (
                        record.turns[0].policy_adapter if record.turns else None
                    ),
                    "server_weight_version": (
                        record.turns[0].server_weight_version if record.turns else None
                    ),
                    "reward": record.evaluation.reward,
                    "evaluator_valid": record.evaluation.valid,
                    "record_grpo_eligible": record.grpo_eligible,
                    "objective_eligible": item.eligible,
                    "advantage": advantages.get(record.trajectory_id),
                    "sampled_output_tokens": len(mask),
                    "action_tokens": sum(mask),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_summary(output_path: Path, summary: SmokeTrainingSummary) -> None:
        (output_path / "training_summary.json").write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


__all__ = [
    "Qwen35OnePassSmokeTrainer",
    "SmokeTrainerConfig",
    "SmokeTrainingSummary",
    "trajectory_to_grpo",
]
