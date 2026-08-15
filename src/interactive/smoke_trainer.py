"""One-update-per-sealed-batch Qwen3.5 LoRA trainer for AgentGraph rollouts.

Source boundary: model/PEFT loading, trainable adapter continuation, optimizer
checkpointing, gradient checkpointing, token-cost-balanced replica splitting,
and checkpoint layout are direct SkillFlow reuse.  Qwen3.5 multimodal loading
and the two-physical-GPU split are necessary adaptations.  Terminal-only,
action-masked one-pass GRPO is the project algorithm addition built on
FlowSteer's objective contract.  SkillFlow's TTB backward policy/partition head
and the project's MACE, Bayesian, and Skill loops are not implemented here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

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
    update_step: int = 1
    optimizer_state_checkpoint: str | None = None
    max_grad_norm: float = 1.0
    advantage_epsilon: float = 1.0e-8
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
        if self.learner_device == self.gradient_replica_device:
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
            self.max_grad_norm,
            self.advantage_epsilon,
            self.behavior_logprob_tolerance,
            self.lora_dropout,
        )
        if not all(math.isfinite(float(value)) for value in numeric_settings):
            raise ValueError("optimizer, GRPO, and LoRA settings must be finite")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("optimizer settings must be positive")
        if self.advantage_epsilon <= 0 or self.behavior_logprob_tolerance <= 0:
            raise ValueError("GRPO tolerances must be positive")
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
    gradient_partition_token_costs: Tuple[int, int] = (0, 0)

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

    This is the local two-GPU equivalent of SkillFlow's
    ``partition_items_by_token_cost``: sort sealed work items by descending
    cost, place each in the currently lightest bin, then restore source order
    inside each bin.  An exact GRPO group is the indivisible item here because
    its behavior-policy acceptance gate is evaluated as one unit.  Empty bins
    are allowed when fewer exact groups than physical replicas are available.
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
        group_partitions, _ = _partition_groups_by_token_cost(
            informative,
            records_by_id,
            worker_count=2,
        )
        models = [learner, replica]
        devices = [self.config.learner_device, self.config.gradient_replica_device]

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
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
            # same SkillFlow token-cost partitioner so the actual backward
            # work, rather than the larger preflight candidate set, is balanced.
            accepted_partitions, partition_token_costs = (
                _partition_groups_by_token_cost(
                    accepted_groups,
                    records_by_id,
                    worker_count=2,
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
            replica.train()
            partition_stats = None
            micro_batch_size_used = 0
            oom_backoff_count = 0
            for micro_batch_size in self.config.micro_batch_backoff:
                for model in models:
                    model.zero_grad(set_to_none=True)
                with ThreadPoolExecutor(max_workers=2) as pool:
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

            self._merge_replica_grads(learner, replica)
            trainable = [parameter for parameter in learner.parameters() if parameter.requires_grad]
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                trainable, self.config.max_grad_norm
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
            if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise RuntimeError("informative GRPO batch produced no finite parameter gradient")
            optimizer = torch.optim.AdamW(
                trainable,
                lr=self.config.learning_rate,
                weight_decay=0.01,
            )
            optimizer_resume_status = self._restore_optimizer_state(
                torch,
                optimizer,
            )
            before_step = [parameter.detach().clone() for parameter in trainable]
            optimizer.step()
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
            optimizer_state_checkpoint, optimizer_state_saved = (
                self._save_optimizer_state(torch, optimizer, checkpoint)
            )
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
                gradient_partition_token_costs=partition_token_costs,
            )
            self._write_summary(output_path, summary)
            return summary
        finally:
            del learner, replica
            self._empty_device_caches(
                torch,
                (self.config.learner_device, self.config.gradient_replica_device),
            )

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
            # The verified local 9B artifact is the Qwen3.5 conditional-
            # generation checkpoint used by SkillFlow's formal deployment,
            # not the older text-only Qwen3-8B checkpoint used by FlowSteer.
            base = AutoModelForMultimodalLM.from_pretrained(
                self.config.model_path,
                dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self.config.behavior_adapter_checkpoint is not None:
                # SkillFlow continuation contract: both physical gradient
                # replicas attach the exact frozen behavior adapter as a
                # trainable theta adapter before any receipt preflight.
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
        replica = load(self.config.gradient_replica_device)
        self._sync_lora_weights(learner, replica)
        return torch, learner, replica

    def _restore_optimizer_state(self, torch, optimizer) -> str:
        state_checkpoint = self.config.optimizer_state_checkpoint
        if state_checkpoint is None:
            if (
                self.config.behavior_adapter_checkpoint is not None
                and self.config.update_step > 1
            ):
                # The original step-1 smoke checkpoint predates optimizer-state
                # persistence.  This is an explicit warm start, never reported
                # as a fully resumed optimizer trajectory.
                return "warm_start_fresh_optimizer"
            return "fresh_optimizer"

        payload = torch.load(
            state_checkpoint,
            map_location=self.config.learner_device,
            weights_only=False,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("optimizer checkpoint payload must be a mapping")
        if payload.get("format") != "flowsteer-smoke-optimizer-v1":
            raise ValueError("optimizer checkpoint format differs")
        committed_step = payload.get("committed_step")
        if committed_step != self.config.update_step - 1:
            raise ValueError(
                "optimizer checkpoint committed step must immediately precede "
                "update_step"
            )
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("optimizer checkpoint is missing optimizer_state_dict")
        optimizer.load_state_dict(optimizer_state)
        return "restored_optimizer"

    def _save_optimizer_state(self, torch, optimizer, checkpoint: Path) -> tuple[str, bool]:
        # Step 1 remains compatible with the already-materialized smoke
        # checkpoint.  Starting at step 2, every committed adapter carries the
        # AdamW state needed for exact continuation at the next update.
        if self.config.update_step < 2:
            return "", False
        state_path = checkpoint / "optimizer_state.pt"
        torch.save(
            {
                "format": "flowsteer-smoke-optimizer-v1",
                "committed_step": self.config.update_step,
                "behavior_policy_version": self.config.behavior_policy_version,
                "updated_policy_version": self.config.updated_policy_version,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            state_path,
        )
        return str(state_path), True

    @staticmethod
    def _is_cuda_oom(torch, error: RuntimeError) -> bool:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        return isinstance(error, oom_type) or "out of memory" in str(error).lower()

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
