"""SkillFlow Tempered Trajectory Balance trainer for AgentGraph receipts.

This is a local AgentGraph implementation anchored to the referenced SkillFlow
training boundary, not a new training method and not a direct import of the
upstream trainer. Its source anchors are:

* ``training/gflownet_trainer.py``: named theta/phi LoRA adapters,
  ``PartitionFunctionHead``, TTB regression, AdamW updates, and the released
  optimizer-step order;
* ``training/backward_policy.py``: the backward policy scores the same
  structured action after adding the execution observation to its context;
* ``training/flow_metrics.py``: each edge is normalized by its structured
  action-token count and each trajectory by its action-edge horizon.

The necessary project adaptation is the input contract: FlowSteer's
``TrajectoryRecord`` stores exact sampled token IDs, the parser-admitted
structured-action span, Canvas feedback, and policy/version receipts.  No
GRPO advantage, MACE, Bayesian posterior, or Skill-evolution signal is used.

The two-device gradient sharding, complete per-step recovery checkpoint, and
publication adapter are project engineering needed by this runtime; SkillFlow's
paper does not prescribe a TP/DP/ZeRO layout, an atomic route switch, or a
recovery transaction. The released trainer also prefetches the next batch
before the current optimizer update. A caller must therefore disable that
cross-step ordering, resource-gate the concrete devices, and publish ``theta``
before collecting the next on-policy batch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

from .records import TrajectoryRecord
from .ttb_objective import (
    torch_tempered_trajectory_balance_loss,
    torch_tempered_trajectory_balance_residual,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TTBTrainerConfig:
    model_path: str
    tokenizer_path: str
    behavior_policy_version: str
    updated_policy_version: str
    behavior_phi_version: str
    updated_phi_version: str
    behavior_z_version: str
    updated_z_version: str
    behavior_server_weight_version: str
    update_step: int
    learner_device: str
    gradient_replica_device: str
    behavior_policy_adapter: str | None = None
    continuation_checkpoint: str | None = None
    theta_rank: int = 64
    theta_alpha: int = 128
    theta_dropout: float = 0.05
    theta_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    phi_rank: int = 16
    phi_alpha: int = 32
    phi_dropout: float = 0.05
    phi_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    beta: float = 1.0
    epsilon_min: float = 0.1
    kl_coefficient: float = 0.01
    max_grad_norm: float = 3.0
    expected_batch_size: int = 28
    questions_per_batch: int = 7
    trajectories_per_question: int = 4
    max_trajectory_edges: int = 12
    max_sequence_tokens: int = 32768
    checkpoint_interval: int = 10
    gradient_checkpointing: bool = True
    require_unconstrained_action_sampling: bool = True

    def __post_init__(self) -> None:
        required = (
            self.model_path,
            self.tokenizer_path,
            self.behavior_policy_version,
            self.updated_policy_version,
            self.behavior_phi_version,
            self.updated_phi_version,
            self.behavior_z_version,
            self.updated_z_version,
            self.behavior_server_weight_version,
            self.learner_device,
            self.gradient_replica_device,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("model, version, and device fields must be non-empty")
        if self.behavior_policy_version == self.updated_policy_version:
            raise ValueError("updated policy version must differ from behavior policy")
        if self.behavior_phi_version == self.updated_phi_version:
            raise ValueError("updated phi version must differ from behavior phi")
        if self.behavior_z_version == self.updated_z_version:
            raise ValueError("updated Z version must differ from behavior Z")
        if self.learner_device == self.gradient_replica_device:
            raise ValueError("learner and gradient replica devices must differ")
        if type(self.update_step) is not int or self.update_step < 1:
            raise ValueError("update_step must be a positive integer")
        if self.update_step > 1 and not self.continuation_checkpoint:
            raise ValueError("step 2+ requires the preceding TTB recovery checkpoint")
        if self.continuation_checkpoint is not None and (
            not isinstance(self.continuation_checkpoint, str)
            or not self.continuation_checkpoint.strip()
        ):
            raise ValueError("continuation_checkpoint must be non-empty when supplied")
        if self.behavior_policy_adapter is not None and (
            not isinstance(self.behavior_policy_adapter, str)
            or not self.behavior_policy_adapter.strip()
        ):
            raise ValueError("behavior_policy_adapter must be non-empty when supplied")
        if (
            self.theta_rank != 64
            or self.theta_alpha != 128
            or self.theta_dropout != 0.05
            or self.theta_target_modules != ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            raise ValueError(
                "referenced theta-LoRA profile must be "
                "r64/alpha128/dropout0.05 q,k,v,o_proj"
            )
        if (
            self.phi_rank != 16
            or self.phi_alpha != 32
            or self.phi_dropout != 0.05
            or self.phi_target_modules != ("q_proj", "v_proj")
        ):
            raise ValueError(
                "referenced phi-LoRA profile must be "
                "r16/alpha32/dropout0.05 q,v_proj"
            )
        if self.expected_batch_size != 28:
            raise ValueError("formal TTB effective batch must be 28")
        if self.questions_per_batch != 7 or self.trajectories_per_question != 4:
            raise ValueError("formal TTB batch shape must be 7 questions x 4 trajectories")
        if self.max_trajectory_edges != 12:
            raise ValueError("formal TTB maximum trajectory horizon must be 12")
        if self.checkpoint_interval != 10:
            raise ValueError("formal checkpoint interval must be 10 steps")
        fixed = {
            "learning_rate": (self.learning_rate, 1.0e-4),
            "beta": (self.beta, 1.0),
            "epsilon_min": (self.epsilon_min, 0.1),
            "kl_coefficient": (self.kl_coefficient, 0.01),
            "max_grad_norm": (self.max_grad_norm, 3.0),
        }
        for name, (observed, expected) in fixed.items():
            if not math.isfinite(float(observed)) or float(observed) != expected:
                raise ValueError(f"formal TTB {name} must equal {expected}")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.max_sequence_tokens < 1:
            raise ValueError("max_sequence_tokens must be positive")
        if self.require_unconstrained_action_sampling is not True:
            raise ValueError(
                "formal TTB currently requires single-phase unconstrained action "
                "sampling; grammar-conditioned log-probability is not implemented"
            )


@dataclass(frozen=True)
class TTBStepCoordinate:
    """Version and serving coordinate for one strictly on-policy TTB step."""

    step: int
    behavior_theta_version: str
    updated_theta_version: str
    behavior_phi_version: str
    updated_phi_version: str
    behavior_z_version: str
    updated_z_version: str
    behavior_policy_adapter: str | None
    behavior_server_weight_version: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 1:
            raise ValueError("TTB step must be a positive integer")
        required = (
            self.behavior_theta_version,
            self.updated_theta_version,
            self.behavior_phi_version,
            self.updated_phi_version,
            self.behavior_z_version,
            self.updated_z_version,
            self.behavior_server_weight_version,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("TTB step versions and server weight version must be non-empty")
        for behavior, updated, name in (
            (self.behavior_theta_version, self.updated_theta_version, "theta"),
            (self.behavior_phi_version, self.updated_phi_version, "phi"),
            (self.behavior_z_version, self.updated_z_version, "Z"),
        ):
            if behavior == updated:
                raise ValueError(f"TTB {name} version must advance at every step")
        if self.behavior_policy_adapter is not None and (
            not isinstance(self.behavior_policy_adapter, str)
            or not self.behavior_policy_adapter.strip()
        ):
            raise ValueError("behavior_policy_adapter must be non-empty when supplied")


@dataclass(frozen=True)
class TTBTrainingSummary:
    optimizer_updates: int
    update_step: int
    input_trajectories: int
    eligible_trajectories: int
    filtered_trajectories: int
    ttb_loss: float
    total_loss: float
    delta_squared_mean: float
    delta_squared_sum: float
    delta_min: float
    delta_max: float
    log_z_mean: float
    kl_estimate: float
    reward_mean: float
    reward_min: float
    reward_max: float
    trajectory_edges_mean: float
    trajectory_edges_min: int
    trajectory_edges_max: int
    action_token_count: int
    theta_grad_norm: float
    phi_grad_norm: float
    z_grad_norm: float
    theta_update_l2: float
    phi_update_l2: float
    z_update_l2: float
    behavior_policy_version: str
    updated_policy_version: str
    phi_behavior_version: str
    phi_updated_version: str
    z_behavior_version: str
    z_updated_version: str
    checkpoint_dir: str
    phi_checkpoint_dir: str
    recovery_checkpoint_dir: str
    official_checkpoint_dir: str
    official_checkpoint_saved: bool
    exclusions: Mapping[str, str]
    step_seconds: float
    started_at: str
    completed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PartitionFunctionHead:
    """Factory for SkillFlow's task-conditioned ``log Z_theta(q)`` head."""

    @staticmethod
    def build(torch, hidden_size: int, *, device: str):
        nn = torch.nn

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.task_embed = nn.Embedding(1, 64)
                self.head = nn.Linear(hidden_size + 64, 1, bias=True)
                nn.init.zeros_(self.head.weight)
                nn.init.constant_(self.head.bias, -2.3)

            def forward(self, query_hidden):
                task_ids = torch.zeros(
                    query_hidden.shape[0],
                    dtype=torch.long,
                    device=query_hidden.device,
                )
                combined = torch.cat((query_hidden, self.task_embed(task_ids)), dim=-1)
                return torch.clamp(self.head(combined).squeeze(-1), -10.0, 10.0)

        return _Head().to(device)


def _adapter_parameters(model, adapter_name: str) -> list[Any]:
    marker_a = f".lora_A.{adapter_name}"
    marker_b = f".lora_B.{adapter_name}"
    result = [
        parameter
        for name, parameter in model.named_parameters()
        if marker_a in name or marker_b in name
    ]
    if not result:
        raise RuntimeError(f"no trainable {adapter_name} LoRA parameters were found")
    return result


def _set_trainable_adapters(model, theta_params, phi_params) -> None:
    theta_ids = {id(parameter) for parameter in theta_params}
    phi_ids = {id(parameter) for parameter in phi_params}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in theta_ids or id(parameter) in phi_ids)


def _parameter_update_l2(torch, parameters, before) -> float:
    total = 0.0
    for parameter, old in zip(parameters, before):
        total += float(
            (parameter.detach().float() - old).square().sum().detach().cpu()
        )
    return math.sqrt(total)


class Qwen35TTBTrainer:
    """Persistent SkillFlow-style trainer for sealed on-policy TTB batches."""

    def __init__(self, config: TTBTrainerConfig) -> None:
        self.config = config
        self._output_root: Path | None = None
        self._torch = None
        self._tokenizer = None
        self._learner_model = None
        self._replica_model = None
        self._learner_z = None
        self._replica_z = None
        self._learner_theta_params: list[Any] = []
        self._learner_phi_params: list[Any] = []
        self._replica_theta_params: list[Any] = []
        self._replica_phi_params: list[Any] = []
        self._theta_optimizer = None
        self._phi_optimizer = None
        self._z_optimizer = None
        self._committed_step = self.config.update_step - 1
        self._current_theta_version = self.config.behavior_policy_version
        self._current_phi_version = self.config.behavior_phi_version
        self._current_z_version = self.config.behavior_z_version
        self._current_policy_adapter = self.config.behavior_policy_adapter
        self._current_server_weight_version = (
            self.config.behavior_server_weight_version
        )
        self._pending_publication: dict[str, Any] | None = None
        self._active_transaction_root: Path | None = None
        self._active_transaction: dict[str, Any] | None = None
        self._faulted = False
        self._closed = False

    @property
    def is_setup(self) -> bool:
        return self._learner_model is not None

    @property
    def committed_step(self) -> int:
        return self._committed_step

    def initial_step_coordinate(self) -> TTBStepCoordinate:
        return TTBStepCoordinate(
            step=self.config.update_step,
            behavior_theta_version=self.config.behavior_policy_version,
            updated_theta_version=self.config.updated_policy_version,
            behavior_phi_version=self.config.behavior_phi_version,
            updated_phi_version=self.config.updated_phi_version,
            behavior_z_version=self.config.behavior_z_version,
            updated_z_version=self.config.updated_z_version,
            behavior_policy_adapter=self.config.behavior_policy_adapter,
            behavior_server_weight_version=(
                self.config.behavior_server_weight_version
            ),
        )

    @staticmethod
    def _load_runtime_dependencies():
        try:
            import torch
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import AutoModelForMultimodalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - heavy runtime only
            raise RuntimeError(
                "Qwen3.5 TTB training requires torch, transformers, and peft"
            ) from exc
        return (
            torch,
            LoraConfig,
            PeftModel,
            get_peft_model,
            AutoModelForMultimodalLM,
            AutoTokenizer,
        )

    def _continuation_metadata(self) -> Mapping[str, Any] | None:
        if self.config.continuation_checkpoint is None:
            if self.config.behavior_policy_adapter is not None:
                raise ValueError(
                    "a non-base behavior adapter requires a TTB continuation checkpoint"
                )
            return None
        checkpoint = Path(self.config.continuation_checkpoint).expanduser().resolve()
        metadata_path = checkpoint / "training_state.json"
        if not checkpoint.is_dir() or not metadata_path.is_file():
            raise ValueError(
                "TTB continuation checkpoint must contain training_state.json"
            )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("TTB continuation metadata must be a mapping")
        expected = {
            "committed_step": self.config.update_step - 1,
            "theta_version": self.config.behavior_policy_version,
            "phi_version": self.config.behavior_phi_version,
            "z_version": self.config.behavior_z_version,
            "model_path": self.config.model_path,
            "tokenizer_path": self.config.tokenizer_path,
            "published_policy_adapter": self.config.behavior_policy_adapter,
            "published_server_weight_version": (
                self.config.behavior_server_weight_version
            ),
            "publication_committed": True,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"TTB continuation {name} differs from the configured behavior state"
                )
        expected_profile = self._continuation_profile()
        if payload.get("training_profile") != expected_profile:
            raise ValueError(
                "TTB continuation training profile differs from the configured run"
            )
        publication_path = checkpoint / "publication_receipt.json"
        if not publication_path.is_file():
            raise ValueError(
                "TTB continuation checkpoint has no committed publication receipt"
            )
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        if not isinstance(publication, Mapping):
            raise ValueError("TTB continuation publication receipt must be a mapping")
        if (
            publication.get("success") is not True
            or publication.get("canary_succeeded") is not True
            or publication.get("new_policy_version")
            != self.config.behavior_policy_version
            or publication.get("adapter_name")
            != self.config.behavior_policy_adapter
            or publication.get("server_weight_version")
            != self.config.behavior_server_weight_version
        ):
            raise ValueError(
                "TTB continuation publication receipt differs from the behavior route"
            )
        return payload

    def _continuation_profile(self) -> dict[str, Any]:
        """Return non-secret fields that must not drift across TTB resume."""

        return {
            "objective": "tempered_trajectory_balance",
            "theta_lora": {
                "rank": self.config.theta_rank,
                "alpha": self.config.theta_alpha,
                "dropout": self.config.theta_dropout,
                "target_modules": list(self.config.theta_target_modules),
            },
            "phi_lora": {
                "rank": self.config.phi_rank,
                "alpha": self.config.phi_alpha,
                "dropout": self.config.phi_dropout,
                "target_modules": list(self.config.phi_target_modules),
            },
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "beta": self.config.beta,
            "epsilon_min": self.config.epsilon_min,
            "kl_coefficient": self.config.kl_coefficient,
            "max_grad_norm": self.config.max_grad_norm,
            "expected_batch_size": self.config.expected_batch_size,
            "questions_per_batch": self.config.questions_per_batch,
            "trajectories_per_question": self.config.trajectories_per_question,
            "max_trajectory_edges": self.config.max_trajectory_edges,
            "max_sequence_tokens": self.config.max_sequence_tokens,
            "checkpoint_interval": self.config.checkpoint_interval,
            "require_unconstrained_action_sampling": (
                self.config.require_unconstrained_action_sampling
            ),
        }

    def setup(self, output_root: str | Path) -> None:
        """Load θ/φ/Z once and restore the three AdamW states when resuming.

        This persistent lifecycle follows SkillFlow's ``GFlowNetTrainer``.  The
        two physical gradient replicas are a project implementation detail and
        must already have passed the external resource gate.
        """

        if self._closed:
            raise RuntimeError("a closed TTB trainer cannot be set up again")
        if self.is_setup:
            raise RuntimeError("TTB trainer is already set up")
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        metadata = self._continuation_metadata()
        (
            torch,
            LoraConfig,
            PeftModel,
            get_peft_model,
            AutoModelForMultimodalLM,
            AutoTokenizer,
        ) = self._load_runtime_dependencies()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path,
            trust_remote_code=True,
        )
        learner = self._load_model_replica(
            device=self.config.learner_device,
            torch=torch,
            LoraConfig=LoraConfig,
            PeftModel=PeftModel,
            get_peft_model=get_peft_model,
            AutoModelForMultimodalLM=AutoModelForMultimodalLM,
        )
        replica = self._load_model_replica(
            device=self.config.gradient_replica_device,
            torch=torch,
            LoraConfig=LoraConfig,
            PeftModel=PeftModel,
            get_peft_model=get_peft_model,
            AutoModelForMultimodalLM=AutoModelForMultimodalLM,
        )
        (
            learner_model,
            learner_z,
            learner_theta,
            learner_phi,
        ) = learner
        (
            replica_model,
            replica_z,
            replica_theta,
            replica_phi,
        ) = replica
        self._validate_replica_layout(
            learner_model,
            replica_model,
            learner_z,
            replica_z,
        )
        self._sync_replicas(learner_model, replica_model, learner_z, replica_z)
        theta_optimizer = torch.optim.AdamW(
            learner_theta,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        phi_optimizer = torch.optim.AdamW(
            learner_phi,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        z_optimizer = torch.optim.AdamW(
            learner_z.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if metadata is not None:
            state_path = (
                Path(self.config.continuation_checkpoint).expanduser().resolve()
                / "optimizer_states.pt"
            )
            if not state_path.is_file():
                raise ValueError(
                    "TTB continuation checkpoint is missing optimizer_states.pt"
                )
            optimizer_state = torch.load(
                state_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("TTB optimizer checkpoint must be a mapping")
            for name, optimizer in (
                ("theta", theta_optimizer),
                ("phi", phi_optimizer),
                ("z", z_optimizer),
            ):
                state = optimizer_state.get(name)
                if not isinstance(state, Mapping):
                    raise ValueError(
                        f"TTB optimizer checkpoint is missing {name} state"
                    )
                optimizer.load_state_dict(state)
            rng_state = optimizer_state.get("rng_state")
            if not isinstance(rng_state, Mapping):
                raise ValueError("TTB continuation checkpoint is missing RNG state")
            python_state = rng_state.get("python")
            torch_cpu_state = rng_state.get("torch_cpu")
            cuda_states = rng_state.get("torch_cuda")
            if python_state is None or torch_cpu_state is None or not isinstance(
                cuda_states, Mapping
            ):
                raise ValueError("TTB continuation RNG state is incomplete")
            random.setstate(python_state)
            torch.set_rng_state(torch_cpu_state.cpu())
            for device in (
                self.config.learner_device,
                self.config.gradient_replica_device,
            ):
                state = cuda_states.get(device)
                if state is None:
                    raise ValueError(
                        f"TTB continuation RNG state is missing device {device}"
                    )
                torch.cuda.set_rng_state(state.cpu(), device=device)

        self._output_root = root
        self._torch = torch
        self._tokenizer = tokenizer
        self._learner_model = learner_model
        self._replica_model = replica_model
        self._learner_z = learner_z
        self._replica_z = replica_z
        self._learner_theta_params = learner_theta
        self._learner_phi_params = learner_phi
        self._replica_theta_params = replica_theta
        self._replica_phi_params = replica_phi
        self._theta_optimizer = theta_optimizer
        self._phi_optimizer = phi_optimizer
        self._z_optimizer = z_optimizer

    def _load_model_replica(
        self,
        *,
        device: str,
        torch,
        LoraConfig,
        PeftModel,
        get_peft_model,
        AutoModelForMultimodalLM,
    ):
        base = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        continuation = (
            Path(self.config.continuation_checkpoint)
            if self.config.continuation_checkpoint
            else None
        )
        if continuation is None:
            theta_config = LoraConfig(
                r=self.config.theta_rank,
                lora_alpha=self.config.theta_alpha,
                target_modules=list(self.config.theta_target_modules),
                # The paper's main-run table does not report dropout; 0.05 is
                # retained from the referenced SkillFlow implementation.
                lora_dropout=self.config.theta_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, theta_config, adapter_name="theta")
            phi_config = LoraConfig(
                r=self.config.phi_rank,
                lora_alpha=self.config.phi_alpha,
                target_modules=list(self.config.phi_target_modules),
                lora_dropout=self.config.phi_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.add_adapter("phi", phi_config)
        else:
            theta_path = continuation / "theta"
            phi_path = continuation / "phi"
            if not theta_path.is_dir() or not phi_path.is_dir():
                raise ValueError(
                    "TTB continuation checkpoint must contain theta/ and phi/ adapters"
                )
            model = PeftModel.from_pretrained(
                base,
                theta_path,
                adapter_name="theta",
                is_trainable=True,
            )
            model.load_adapter(phi_path, adapter_name="phi", is_trainable=True)
        theta_params = _adapter_parameters(model, "theta")
        phi_params = _adapter_parameters(model, "phi")
        _set_trainable_adapters(model, theta_params, phi_params)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.train()
        hidden_size = getattr(base.config, "hidden_size", None)
        if hidden_size is None and hasattr(base.config, "text_config"):
            hidden_size = getattr(base.config.text_config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise RuntimeError("Qwen3.5 model config has no usable hidden_size")
        z_head = PartitionFunctionHead.build(torch, hidden_size, device=device)
        if continuation is not None:
            state_path = continuation / "z_head.pt"
            if not state_path.is_file():
                raise ValueError("TTB continuation checkpoint is missing z_head.pt")
            z_head.load_state_dict(
                torch.load(state_path, map_location=device, weights_only=True)
            )
        z_head.train()
        return model, z_head, theta_params, phi_params

    @staticmethod
    def _activate_adapter(model, adapter_name: str, theta_params, phi_params) -> None:
        model.set_adapter(adapter_name)
        _set_trainable_adapters(model, theta_params, phi_params)

    @staticmethod
    def _selected_log_probs(torch, model, device: str, prefix_ids, action_ids):
        if not prefix_ids or not action_ids:
            raise ValueError("TTB edge requires non-empty prefix and action token IDs")
        full_ids = tuple(prefix_ids) + tuple(action_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            request = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                "logits_to_keep": len(action_ids) + 1,
                "output_hidden_states": False,
            }
            try:
                outputs = model(**request)
            except TypeError as exc:
                # Qwen3.5's current Transformers model supports
                # ``logits_to_keep``.  The fallback preserves compatibility
                # with an older local Transformers build without changing the
                # probability distribution.
                if "logits_to_keep" not in str(exc):
                    raise
                request.pop("logits_to_keep")
                outputs = model(**request)
            logits = outputs.logits[0]
            action_logits = logits[-(len(action_ids) + 1) : -1]
            targets = input_ids[0, -len(action_ids) :]
            log_probs = torch.log_softmax(action_logits.float(), dim=-1)
            selected = log_probs[
                torch.arange(len(action_ids), device=targets.device),
                targets,
            ]
        del outputs, logits, action_logits, log_probs, input_ids, attention_mask
        return selected

    @staticmethod
    def _question_hidden(torch, model, tokenizer, device: str, question: str, theta_params, phi_params):
        Qwen35TTBTrainer._activate_adapter(
            model,
            "theta",
            theta_params,
            phi_params,
        )
        ids = tokenizer.encode(
            question,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)
        if ids.numel() == 0:
            raise ValueError("TTB question encoded to no token IDs")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(ids, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1][0, -1, :].detach().float()
        del ids, outputs
        return hidden

    @staticmethod
    def _forward_edge_tokens(turn) -> tuple[tuple[int, ...], tuple[int, ...]]:
        start = turn.structured_action_token_start
        end = start + turn.structured_action_token_count
        prefix = tuple(turn.prompt_token_ids) + tuple(turn.output_token_ids[:start])
        action = tuple(turn.output_token_ids[start:end])
        return prefix, action

    @staticmethod
    def _backward_edge_tokens(tokenizer, turn) -> tuple[tuple[int, ...], tuple[int, ...]]:
        start = turn.structured_action_token_start
        end = start + turn.structured_action_token_count
        hindsight_text = turn.prompt
        if turn.canvas_feedback:
            hindsight_text += "\nObservation: " + turn.canvas_feedback
        # SkillFlow tokenizes ``hindsight state + reasoning`` as one context.
        # Re-tokenizing the concatenated text matters because a tokenizer can
        # merge bytes across the hindsight/reasoning boundary; concatenating
        # independently encoded ID lists would score a different context.
        reasoning_text = tokenizer.decode(
            list(turn.output_token_ids[:start]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        context_ids = tuple(
            tokenizer.encode(
                hindsight_text + reasoning_text,
                add_special_tokens=False,
            )
        )
        return context_ids, tuple(turn.output_token_ids[start:end])

    def _reference_edge_mean(
        self,
        torch,
        model,
        device: str,
        prefix_ids,
        action_ids,
        theta_params,
        phi_params,
    ):
        model.disable_adapter_layers()
        try:
            with torch.no_grad():
                return self._selected_log_probs(
                    torch,
                    model,
                    device,
                    prefix_ids,
                    action_ids,
                ).mean().detach()
        finally:
            model.enable_adapter_layers()
            self._activate_adapter(model, "theta", theta_params, phi_params)

    def _validate_batch(
        self,
        records: Sequence[TrajectoryRecord],
        coordinate: TTBStepCoordinate,
    ) -> tuple[TrajectoryRecord, ...]:
        batch = tuple(records)
        if len(batch) != self.config.expected_batch_size:
            raise ValueError(
                f"TTB batch must contain exactly {self.config.expected_batch_size} trajectories"
            )
        if len({record.trajectory_id for record in batch}) != len(batch):
            raise ValueError("TTB batch contains duplicate trajectory IDs")
        if len({record.rollout_id for record in batch}) != len(batch):
            raise ValueError("TTB batch contains duplicate rollout IDs")
        observed_regimes = {record.versions.fingerprint for record in batch}
        if len(observed_regimes) != 1:
            raise ValueError("TTB batch contains mixed VersionBundle regimes")
        sampling_coordinates = {
            json.dumps(
                record.director_sampling.get("coordinate"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in batch
        }
        if len(sampling_coordinates) != len(batch):
            raise ValueError("TTB batch contains duplicate scientific sampling coordinates")
        per_question: dict[str, int] = {}
        question_groups: dict[str, set[str]] = {}
        for record in batch:
            per_question[record.task.task_id] = per_question.get(record.task.task_id, 0) + 1
            question_groups.setdefault(record.task.task_id, set()).add(record.group_id)
        if len(per_question) != self.config.questions_per_batch or set(
            per_question.values()
        ) != {self.config.trajectories_per_question}:
            raise ValueError("TTB batch must contain seven questions with four trajectories each")
        if any(len(groups) != 1 for groups in question_groups.values()):
            raise ValueError("each TTB question must use one frozen trajectory group regime")
        observed_policy_versions = {record.versions.policy for record in batch}
        if observed_policy_versions != {coordinate.behavior_theta_version}:
            raise ValueError("TTB batch differs from the configured behavior policy version")
        observed_turn_policy_versions = {
            turn.policy_version for record in batch for turn in record.turns
        }
        if observed_turn_policy_versions != {coordinate.behavior_theta_version}:
            raise ValueError(
                "TTB turn receipts differ from the configured behavior policy version"
            )
        observed_conditions = {record.condition_id for record in batch}
        if len(observed_conditions) != 1:
            raise ValueError("TTB batch must use one frozen rollout condition")
        expected_route = (
            coordinate.behavior_policy_adapter,
            coordinate.behavior_server_weight_version,
        )
        observed_routes = {
            (turn.policy_adapter, turn.server_weight_version)
            for record in batch
            for turn in record.turns
        }
        if observed_routes != {expected_route}:
            raise ValueError("TTB batch contains mixed behavior adapter/server routes")
        for record in batch:
            if not record.ttb_eligible:
                raise ValueError(
                    f"trajectory {record.trajectory_id!r} is not TTB eligible"
                )
            if not record.condition_satisfied:
                raise ValueError("TTB batch contains an unsatisfied rollout condition")
            if len(record.turns) > self.config.max_trajectory_edges:
                raise ValueError(
                    f"trajectory {record.trajectory_id!r} exceeds max T=12"
                )
            if record.active_skill_ids or record.retrieved_skill_ids or record.invoked_skill_ids:
                raise ValueError("initial MBPP+ TTB training requires Skill evolution disabled")
            reward = float(record.evaluation.reward)
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError("TTB outcome reward must be finite and in [0, 1]")
            for turn in record.turns:
                if any(
                    key in turn.runtime_summary
                    for key in (
                        "director_action_schema_version",
                        "director_action_schema_branch",
                        "director_action_target_domain_version",
                        "director_action_decoding",
                    )
                ):
                    raise ValueError(
                        "TTB raw-logit objective cannot consume grammar-constrained "
                        "or hierarchical action receipts"
                    )
                forward_prefix, action = self._forward_edge_tokens(turn)
                if len(forward_prefix) + len(action) > self.config.max_sequence_tokens:
                    raise ValueError("TTB forward edge exceeds the configured context limit")
        return batch

    @staticmethod
    def _partition_by_token_cost(records: Sequence[TrajectoryRecord]):
        indexed = []
        for index, record in enumerate(records):
            cost = sum(
                len(turn.prompt_token_ids) + len(turn.output_token_ids)
                for turn in record.turns
            )
            indexed.append((index, record, cost))
        indexed.sort(key=lambda item: (-item[2], item[0]))
        partitions: list[list[tuple[int, TrajectoryRecord]]] = [[], []]
        costs = [0, 0]
        for index, record, cost in indexed:
            target = min(range(2), key=lambda worker: (costs[worker], worker))
            partitions[target].append((index, record))
            costs[target] += cost
        return tuple(
            tuple(sorted(partition, key=lambda item: item[0]))
            for partition in partitions
        ), tuple(costs)

    @staticmethod
    def _validate_replica_layout(learner_model, replica_model, learner_z, replica_z) -> None:
        learner_lora = {
            name: tuple(parameter.shape)
            for name, parameter in learner_model.named_parameters()
            if "lora_" in name
        }
        replica_lora = {
            name: tuple(parameter.shape)
            for name, parameter in replica_model.named_parameters()
            if "lora_" in name
        }
        if learner_lora != replica_lora or not learner_lora:
            raise RuntimeError("TTB gradient replicas have different LoRA layouts")
        learner_z_layout = {
            name: tuple(parameter.shape)
            for name, parameter in learner_z.named_parameters()
        }
        replica_z_layout = {
            name: tuple(parameter.shape)
            for name, parameter in replica_z.named_parameters()
        }
        if learner_z_layout != replica_z_layout or not learner_z_layout:
            raise RuntimeError("TTB gradient replicas have different Z-head layouts")

    @staticmethod
    def _sync_replicas(learner_model, replica_model, learner_z, replica_z) -> None:
        source = {
            name: parameter.detach()
            for name, parameter in learner_model.named_parameters()
            if "lora_" in name
        }
        for name, parameter in replica_model.named_parameters():
            if name in source:
                parameter.data.copy_(source[name].to(parameter.device))
        replica_z.load_state_dict(learner_z.state_dict())

    @staticmethod
    def _require_complete_finite_gradients(parameters, *, label: str) -> None:
        missing = []
        non_finite = []
        for name, parameter in parameters:
            if parameter.grad is None:
                missing.append(name)
            elif not parameter.grad.detach().isfinite().all().item():
                non_finite.append(name)
        if missing or non_finite:
            raise RuntimeError(
                f"TTB {label} gradients are incomplete or non-finite: "
                f"missing={missing}, non_finite={non_finite}"
            )

    @staticmethod
    def _merge_replica_gradients(learner_model, replica_model, learner_z, replica_z) -> None:
        learner_lora = {
            name: parameter
            for name, parameter in learner_model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        }
        replica_lora = {
            name: parameter
            for name, parameter in replica_model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        }
        if set(learner_lora) != set(replica_lora):
            raise RuntimeError("TTB gradient replicas have different trainable LoRA names")
        Qwen35TTBTrainer._require_complete_finite_gradients(
            learner_lora.items(),
            label="learner-shard LoRA",
        )
        Qwen35TTBTrainer._require_complete_finite_gradients(
            replica_lora.items(),
            label="replica-shard LoRA",
        )
        incoming = {
            name: parameter.grad.detach()
            for name, parameter in replica_lora.items()
        }
        for name, parameter in learner_lora.items():
            gradient = incoming[name]
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise RuntimeError("TTB replica LoRA gradient shape mismatch")
            value = gradient.to(parameter.device)
            parameter.grad.add_(value)
        learner_z_params = dict(learner_z.named_parameters())
        replica_z_params = dict(replica_z.named_parameters())
        if set(learner_z_params) != set(replica_z_params):
            raise RuntimeError("TTB gradient replicas have different Z parameter names")
        Qwen35TTBTrainer._require_complete_finite_gradients(
            learner_z_params.items(),
            label="learner-shard Z",
        )
        Qwen35TTBTrainer._require_complete_finite_gradients(
            replica_z_params.items(),
            label="replica-shard Z",
        )
        replica_z_grads = {
            name: parameter.grad.detach() for name, parameter in replica_z_params.items()
        }
        for name, parameter in learner_z_params.items():
            gradient = replica_z_grads[name]
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise RuntimeError("TTB replica Z gradient shape mismatch")
            value = gradient.to(parameter.device)
            parameter.grad.add_(value)

    def _backward_partition(
        self,
        *,
        torch,
        model,
        z_head,
        theta_params,
        phi_params,
        tokenizer,
        device: str,
        partition,
        global_batch_size: int,
    ) -> list[dict[str, Any]]:
        model.zero_grad(set_to_none=True)
        z_head.zero_grad(set_to_none=True)
        artifacts: list[dict[str, Any]] = []
        for position, record in partition:
            question_hidden = self._question_hidden(
                torch,
                model,
                tokenizer,
                device,
                record.task.question,
                theta_params,
                phi_params,
            )
            log_z = z_head(question_hidden.unsqueeze(0)).squeeze()
            forward_sums = []
            backward_sums = []
            token_counts = []
            reference_means = []
            self._activate_adapter(model, "theta", theta_params, phi_params)
            for turn in record.turns:
                prefix_ids, action_ids = self._forward_edge_tokens(turn)
                selected = self._selected_log_probs(
                    torch,
                    model,
                    device,
                    prefix_ids,
                    action_ids,
                )
                forward_sums.append(selected.sum())
                token_counts.append(len(action_ids))
                reference_means.append(
                    self._reference_edge_mean(
                        torch,
                        model,
                        device,
                        prefix_ids,
                        action_ids,
                        theta_params,
                        phi_params,
                    )
                )
            self._activate_adapter(model, "phi", theta_params, phi_params)
            for turn in record.turns:
                prefix_ids, action_ids = self._backward_edge_tokens(tokenizer, turn)
                if len(prefix_ids) + len(action_ids) > self.config.max_sequence_tokens:
                    raise ValueError("TTB hindsight edge exceeds the configured context limit")
                selected = self._selected_log_probs(
                    torch,
                    model,
                    device,
                    prefix_ids,
                    action_ids,
                )
                backward_sums.append(selected.sum())
            forward_tensor = torch.stack(forward_sums)
            backward_tensor = torch.stack(backward_sums)
            count_tensor = torch.tensor(token_counts, dtype=torch.long, device=device)
            reward = float(record.evaluation.reward)
            r_tilde = max(reward + self.config.epsilon_min, self.config.epsilon_min)
            ttb_loss = torch_tempered_trajectory_balance_loss(
                log_z=log_z,
                forward_action_logprob_sums=forward_tensor,
                backward_action_logprob_sums=backward_tensor,
                action_token_counts=count_tensor,
                r_tilde=r_tilde,
                beta=self.config.beta,
            )
            residual_tensor = torch_tempered_trajectory_balance_residual(
                log_z=log_z,
                forward_action_logprob_sums=forward_tensor,
                backward_action_logprob_sums=backward_tensor,
                action_token_counts=count_tensor,
                r_tilde=r_tilde,
                beta=self.config.beta,
            )
            forward_means = forward_tensor / count_tensor.to(forward_tensor.dtype)
            reference_tensor = torch.stack(reference_means).to(forward_tensor.device)
            kl_estimate = (forward_means - reference_tensor).sum()
            combined = (
                ttb_loss + self.config.kl_coefficient * kl_estimate
            ) / global_batch_size
            combined.backward()
            residual = float(residual_tensor.detach().cpu())
            artifacts.append(
                {
                    "position": position,
                    "trajectory_id": record.trajectory_id,
                    "task_id": record.task.task_id,
                    "reward": reward,
                    "r_tilde": r_tilde,
                    "horizon": len(record.turns),
                    "action_token_count": sum(token_counts),
                    "delta": residual,
                    "delta_squared": residual * residual,
                    "ttb_loss": float(ttb_loss.detach().cpu()),
                    "kl_estimate": float(kl_estimate.detach().cpu()),
                    "log_z": float(log_z.detach().cpu()),
                    "forward_edge_means": [
                        float(value) for value in forward_means.detach().cpu().tolist()
                    ],
                    "backward_edge_means": [
                        float(value)
                        for value in (
                            backward_tensor
                            / count_tensor.to(backward_tensor.dtype)
                        ).detach().cpu().tolist()
                    ],
                    "action_token_counts": token_counts,
                }
            )
            del (
                combined,
                ttb_loss,
                residual_tensor,
                forward_tensor,
                backward_tensor,
                count_tensor,
            )
        return artifacts

    def _require_setup(self) -> None:
        if self._closed:
            raise RuntimeError("TTB trainer is closed")
        if not self.is_setup or self._output_root is None:
            raise RuntimeError("TTB trainer setup() must complete before train_step()")

    def _validate_step_coordinate(self, coordinate: TTBStepCoordinate) -> None:
        if self._pending_publication is not None:
            raise RuntimeError(
                "the previous TTB optimizer update is awaiting verified publication"
            )
        if self._faulted:
            raise RuntimeError(
                "the TTB trainer is faulted and must restart from its recorded recovery boundary"
            )
        if coordinate.step != self._committed_step + 1:
            raise ValueError("TTB optimizer steps must be contiguous")
        expected = {
            "theta": (coordinate.behavior_theta_version, self._current_theta_version),
            "phi": (coordinate.behavior_phi_version, self._current_phi_version),
            "Z": (coordinate.behavior_z_version, self._current_z_version),
        }
        for name, (observed, current) in expected.items():
            if observed != current:
                raise ValueError(
                    f"TTB {name} behavior version does not match trainer state"
                )
        if coordinate.behavior_policy_adapter != self._current_policy_adapter:
            raise ValueError(
                "TTB behavior adapter does not match the committed serving route"
            )
        if (
            coordinate.behavior_server_weight_version
            != self._current_server_weight_version
        ):
            raise ValueError(
                "TTB server weight version does not match the committed serving route"
            )

    @staticmethod
    def _finite_positive(name: str, value: Any) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise RuntimeError(f"{name} must be finite and non-zero")
        return result

    @staticmethod
    def _adapter_checkpoint_files(root: Path, adapter_name: str) -> tuple[Path, Path]:
        adapter = root / adapter_name
        return adapter / "adapter_config.json", adapter / "adapter_model.safetensors"

    def _save_adapters(self, model, root: Path) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=False)
        model.set_adapter("theta")
        model.save_pretrained(
            root,
            selected_adapters=["theta", "phi"],
            safe_serialization=True,
        )
        for adapter_name in ("theta", "phi"):
            required = self._adapter_checkpoint_files(root, adapter_name)
            if not all(path.is_file() for path in required):
                raise RuntimeError(
                    f"PEFT did not materialize a complete {adapter_name} adapter"
                )
        return root / "theta", root / "phi"

    def _checkpoint_metadata(
        self,
        coordinate: TTBStepCoordinate,
        *,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "flowsteer.agentgraph.ttb-training-state.v1",
            "objective": "tempered_trajectory_balance",
            "checkpoint_kind": checkpoint_kind,
            # Optimizer state is materialized before external SGLang
            # publication.  ``committed_step`` becomes non-null only after a
            # verified publisher/canary acknowledgement.
            "optimizer_committed_step": coordinate.step,
            "committed_step": None,
            "theta_version": coordinate.updated_theta_version,
            "phi_version": coordinate.updated_phi_version,
            "z_version": coordinate.updated_z_version,
            "behavior_theta_version": coordinate.behavior_theta_version,
            "behavior_phi_version": coordinate.behavior_phi_version,
            "behavior_z_version": coordinate.behavior_z_version,
            "behavior_policy_adapter": coordinate.behavior_policy_adapter,
            "behavior_server_weight_version": (
                coordinate.behavior_server_weight_version
            ),
            "published_policy_adapter": None,
            "published_server_weight_version": None,
            "publication_committed": False,
            "model_path": self.config.model_path,
            "tokenizer_path": self.config.tokenizer_path,
            "training_profile": self._continuation_profile(),
            "dataset_scope": "mbpp_plus_project_adaptation",
            "skillflow_joint_iid_reproduction": False,
            "executor_frozen": True,
            "grpo_enabled": False,
            "mace_enabled": False,
            "bayesian_posterior_enabled": False,
            "skill_evolution_enabled": False,
            "theta_lora": {
                "rank": self.config.theta_rank,
                "alpha": self.config.theta_alpha,
                "dropout": self.config.theta_dropout,
                "target_modules": list(self.config.theta_target_modules),
            },
            "phi_lora": {
                "rank": self.config.phi_rank,
                "alpha": self.config.phi_alpha,
                "dropout": self.config.phi_dropout,
                "target_modules": list(self.config.phi_target_modules),
                "appendix_rank_ambiguity": 32,
            },
            "z_separately_optimized": True,
            "project_implementation": {
                "gradient_devices": [
                    self.config.learner_device,
                    self.config.gradient_replica_device,
                ],
                "gradient_partition": "two_replica_token_cost_partition",
                "recovery_checkpoint_each_step": True,
                "external_theta_publication_required": True,
            },
            "supplementary_source_details_not_reported_in_main_run_table": {
                "lora_dropout": 0.05,
                "adamw_weight_decay": self.config.weight_decay,
            },
            "created_at": _utc_now(),
        }

    def _save_training_checkpoint(
        self,
        root: Path,
        coordinate: TTBStepCoordinate,
        *,
        checkpoint_kind: str,
    ) -> tuple[Path, Path]:
        assert self._torch is not None
        assert self._learner_model is not None
        assert self._learner_z is not None
        assert self._theta_optimizer is not None
        assert self._phi_optimizer is not None
        assert self._z_optimizer is not None
        theta_path, phi_path = self._save_adapters(self._learner_model, root)
        self._torch.save(self._learner_z.state_dict(), root / "z_head.pt")
        self._torch.save(
            {
                "theta": self._theta_optimizer.state_dict(),
                "phi": self._phi_optimizer.state_dict(),
                "z": self._z_optimizer.state_dict(),
                "rng_state": {
                    "python": random.getstate(),
                    "torch_cpu": self._torch.get_rng_state(),
                    "torch_cuda": {
                        device: self._torch.cuda.get_rng_state(device).cpu()
                        for device in (
                            self.config.learner_device,
                            self.config.gradient_replica_device,
                        )
                    },
                },
            },
            root / "optimizer_states.pt",
        )
        (root / "training_state.json").write_text(
            json.dumps(
                self._checkpoint_metadata(
                    coordinate,
                    checkpoint_kind=checkpoint_kind,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return theta_path, phi_path

    def _save_publication_adapter(
        self,
        root: Path,
        coordinate: TTBStepCoordinate,
    ) -> Path:
        assert self._learner_model is not None
        root.mkdir(parents=True, exist_ok=False)
        self._learner_model.set_adapter("theta")
        self._learner_model.save_pretrained(
            root,
            selected_adapters=["theta"],
            safe_serialization=True,
        )
        theta = root / "theta"
        if not all(
            path.is_file()
            for path in self._adapter_checkpoint_files(root, "theta")
        ):
            raise RuntimeError("PEFT did not materialize the publication theta adapter")
        (theta / "policy_version.json").write_text(
            json.dumps(
                {
                    "schema_version": "flowsteer.agentgraph.ttb-theta-publication.v1",
                    "objective": "tempered_trajectory_balance",
                    "committed_step": coordinate.step,
                    "behavior_policy_version": coordinate.behavior_theta_version,
                    "updated_policy_version": coordinate.updated_theta_version,
                    "training_performed": True,
                    "external_publication_pending": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return theta

    @staticmethod
    def _write_ttb_batch(path: Path, records, artifacts) -> None:
        artifact_by_id = {item["trajectory_id"]: item for item in artifacts}
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                diagnostic = artifact_by_id[record.trajectory_id]
                row = {
                    "schema_version": "flowsteer.agentgraph.ttb-batch-row.v1",
                    "trajectory_id": record.trajectory_id,
                    "task_id": record.task.task_id,
                    "condition_id": record.condition_id,
                    "policy_version": record.versions.policy,
                    "policy_adapter": record.turns[0].policy_adapter,
                    "server_weight_version": record.turns[0].server_weight_version,
                    "reward": record.evaluation.reward,
                    "evaluator_version": record.evaluation.evaluator_version,
                    "ttb_eligible": record.ttb_eligible,
                    "horizon": diagnostic["horizon"],
                    "action_token_count": diagnostic["action_token_count"],
                    "delta": diagnostic["delta"],
                    "delta_squared": diagnostic["delta_squared"],
                    "ttb_loss": diagnostic["ttb_loss"],
                    "kl_estimate": diagnostic["kl_estimate"],
                    "log_z": diagnostic["log_z"],
                    "forward_edge_means": diagnostic["forward_edge_means"],
                    "backward_edge_means": diagnostic["backward_edge_means"],
                    "action_token_counts": diagnostic["action_token_counts"],
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        temporary.write_text(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _begin_transaction(
        self,
        *,
        step_root: Path,
        coordinate: TTBStepCoordinate,
    ) -> None:
        self._active_transaction_root = step_root
        self._active_transaction = {
            "schema_version": "flowsteer.agentgraph.ttb-step-transaction.v1",
            "objective": "tempered_trajectory_balance",
            "step": coordinate.step,
            "phase": "PREPARED",
            "optimizer_committed": False,
            "publication_committed": False,
            "behavior_theta_version": coordinate.behavior_theta_version,
            "updated_theta_version": coordinate.updated_theta_version,
            "restart_boundary": (
                self.config.continuation_checkpoint
                or "base_model_with_fresh_named_adapters"
            ),
            "started_at": _utc_now(),
        }
        self._persist_transaction()

    def _persist_transaction(self) -> None:
        if self._active_transaction_root is None or self._active_transaction is None:
            raise RuntimeError("there is no active TTB step transaction")
        self._write_json_atomic(
            self._active_transaction_root / "step_transaction.json",
            self._active_transaction,
        )

    def _advance_transaction(self, phase: str, **fields: Any) -> None:
        if self._active_transaction is None:
            raise RuntimeError("there is no active TTB step transaction")
        self._active_transaction.update(fields)
        self._active_transaction["phase"] = phase
        self._active_transaction["updated_at"] = _utc_now()
        self._persist_transaction()

    def train_step(
        self,
        trajectories: Sequence[TrajectoryRecord],
        *,
        coordinate: TTBStepCoordinate | None = None,
    ) -> TTBTrainingSummary:
        """Execute one update and stop at the external-publication barrier.

        A successful return proves θ/φ/Z ``optimizer.step`` and checkpoint
        materialization, but it deliberately does not advance the behavior
        policy.  The caller must publish the returned θ adapter and call
        :meth:`acknowledge_publication` with the verified SGLang receipt before
        either collecting or training the next step.
        """

        if self._faulted:
            raise RuntimeError(
                "the TTB trainer is faulted and must restart from its recorded recovery boundary"
            )
        if self._pending_publication is not None:
            raise RuntimeError(
                "the previous TTB optimizer update is awaiting verified publication"
            )
        try:
            return self._train_step_attempt(
                trajectories,
                coordinate=coordinate,
            )
        except Exception as error:
            self._faulted = True
            if self._active_transaction is not None:
                failed_after = self._active_transaction.get("phase")
                self._advance_transaction(
                    "FAILED",
                    failed_after_phase=failed_after,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    failed_at=_utc_now(),
                )
            self._active_transaction = None
            self._active_transaction_root = None
            raise

    def _train_step_attempt(
        self,
        trajectories: Sequence[TrajectoryRecord],
        *,
        coordinate: TTBStepCoordinate | None = None,
    ) -> TTBTrainingSummary:
        """Run one θ/φ/Z AdamW update over a sealed 7×4 on-policy batch."""

        self._require_setup()
        coordinate = coordinate or self.initial_step_coordinate()
        self._validate_step_coordinate(coordinate)
        records = self._validate_batch(trajectories, coordinate)
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._learner_model is not None
        assert self._replica_model is not None
        assert self._learner_z is not None
        assert self._replica_z is not None
        assert self._theta_optimizer is not None
        assert self._phi_optimizer is not None
        assert self._z_optimizer is not None
        assert self._output_root is not None

        torch = self._torch
        started_at = _utc_now()
        started_monotonic = time.monotonic()
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        step_root = (
            self._output_root
            / "steps"
            / f"step_{coordinate.step:06d}_{attempt_id}"
        )
        step_root.mkdir(parents=True, exist_ok=False)
        self._begin_transaction(step_root=step_root, coordinate=coordinate)

        for device in (
            self.config.learner_device,
            self.config.gradient_replica_device,
        ):
            with torch.cuda.device(device):
                torch.cuda.reset_peak_memory_stats()

        self._theta_optimizer.zero_grad(set_to_none=True)
        self._phi_optimizer.zero_grad(set_to_none=True)
        self._z_optimizer.zero_grad(set_to_none=True)
        self._sync_replicas(
            self._learner_model,
            self._replica_model,
            self._learner_z,
            self._replica_z,
        )
        partitions, partition_costs = self._partition_by_token_cost(records)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (
                pool.submit(
                    self._backward_partition,
                    torch=torch,
                    model=model,
                    z_head=z_head,
                    theta_params=theta_params,
                    phi_params=phi_params,
                    tokenizer=self._tokenizer,
                    device=device,
                    partition=partition,
                    global_batch_size=len(records),
                )
                for model, z_head, theta_params, phi_params, device, partition in (
                    (
                        self._learner_model,
                        self._learner_z,
                        self._learner_theta_params,
                        self._learner_phi_params,
                        self.config.learner_device,
                        partitions[0],
                    ),
                    (
                        self._replica_model,
                        self._replica_z,
                        self._replica_theta_params,
                        self._replica_phi_params,
                        self.config.gradient_replica_device,
                        partitions[1],
                    ),
                )
            )
            shard_artifacts = [future.result() for future in futures]
        artifacts = sorted(
            (item for shard in shard_artifacts for item in shard),
            key=lambda item: item["position"],
        )
        if len(artifacts) != len(records):
            raise RuntimeError("TTB gradient shards did not cover the sealed batch")
        self._merge_replica_gradients(
            self._learner_model,
            self._replica_model,
            self._learner_z,
            self._replica_z,
        )

        theta_grad_norm = self._finite_positive(
            "theta gradient norm",
            torch.nn.utils.clip_grad_norm_(
                self._learner_theta_params,
                self.config.max_grad_norm,
            ),
        )
        phi_grad_norm = self._finite_positive(
            "phi gradient norm",
            torch.nn.utils.clip_grad_norm_(
                self._learner_phi_params,
                self.config.max_grad_norm,
            ),
        )
        z_parameters = list(self._learner_z.parameters())
        z_grad_norm = self._finite_positive(
            "Z gradient norm",
            torch.nn.utils.clip_grad_norm_(
                z_parameters,
                self.config.max_grad_norm,
            ),
        )
        theta_before = [
            parameter.detach().float().clone()
            for parameter in self._learner_theta_params
        ]
        phi_before = [
            parameter.detach().float().clone()
            for parameter in self._learner_phi_params
        ]
        z_before = [parameter.detach().float().clone() for parameter in z_parameters]

        self._theta_optimizer.step()
        self._phi_optimizer.step()
        self._z_optimizer.step()
        theta_update_l2 = self._finite_positive(
            "theta update L2",
            _parameter_update_l2(torch, self._learner_theta_params, theta_before),
        )
        phi_update_l2 = self._finite_positive(
            "phi update L2",
            _parameter_update_l2(torch, self._learner_phi_params, phi_before),
        )
        z_update_l2 = self._finite_positive(
            "Z update L2",
            _parameter_update_l2(torch, z_parameters, z_before),
        )
        self._theta_optimizer.zero_grad(set_to_none=True)
        self._phi_optimizer.zero_grad(set_to_none=True)
        self._z_optimizer.zero_grad(set_to_none=True)
        self._advance_transaction(
            "OPTIMIZER_COMMITTED",
            optimizer_committed=True,
            theta_grad_norm=theta_grad_norm,
            phi_grad_norm=phi_grad_norm,
            z_grad_norm=z_grad_norm,
            theta_update_l2=theta_update_l2,
            phi_update_l2=phi_update_l2,
            z_update_l2=z_update_l2,
        )

        recovery_root = step_root / "recovery"
        _, recovery_phi = self._save_training_checkpoint(
            recovery_root,
            coordinate,
            checkpoint_kind="project_recovery_each_step",
        )
        official_saved = coordinate.step % self.config.checkpoint_interval == 0
        official_root: Path | None = None
        if official_saved:
            official_root = (
                self._output_root
                / "checkpoints"
                / f"checkpoint_step_{coordinate.step:06d}"
            )
            self._save_training_checkpoint(
                official_root,
                coordinate,
                checkpoint_kind="skillflow_interval_10_steps",
            )
        publication_theta = self._save_publication_adapter(
            step_root / "publication",
            coordinate,
        )
        self._write_ttb_batch(step_root / "ttb_batch.jsonl", records, artifacts)

        rewards = [float(record.evaluation.reward) for record in records]
        horizons = [len(record.turns) for record in records]
        delta_squared = [float(item["delta_squared"]) for item in artifacts]
        ttb_losses = [float(item["ttb_loss"]) for item in artifacts]
        kl_values = [float(item["kl_estimate"]) for item in artifacts]
        log_z_values = [float(item["log_z"]) for item in artifacts]
        ttb_loss_mean = sum(ttb_losses) / len(ttb_losses)
        kl_mean = sum(kl_values) / len(kl_values)
        summary = TTBTrainingSummary(
            optimizer_updates=1,
            update_step=coordinate.step,
            input_trajectories=len(records),
            eligible_trajectories=len(records),
            filtered_trajectories=0,
            ttb_loss=ttb_loss_mean,
            total_loss=ttb_loss_mean + self.config.kl_coefficient * kl_mean,
            delta_squared_mean=sum(delta_squared) / len(delta_squared),
            delta_squared_sum=sum(delta_squared),
            delta_min=min(float(item["delta"]) for item in artifacts),
            delta_max=max(float(item["delta"]) for item in artifacts),
            log_z_mean=sum(log_z_values) / len(log_z_values),
            kl_estimate=kl_mean,
            reward_mean=sum(rewards) / len(rewards),
            reward_min=min(rewards),
            reward_max=max(rewards),
            trajectory_edges_mean=sum(horizons) / len(horizons),
            trajectory_edges_min=min(horizons),
            trajectory_edges_max=max(horizons),
            action_token_count=sum(
                int(item["action_token_count"]) for item in artifacts
            ),
            theta_grad_norm=theta_grad_norm,
            phi_grad_norm=phi_grad_norm,
            z_grad_norm=z_grad_norm,
            theta_update_l2=theta_update_l2,
            phi_update_l2=phi_update_l2,
            z_update_l2=z_update_l2,
            behavior_policy_version=coordinate.behavior_theta_version,
            updated_policy_version=coordinate.updated_theta_version,
            phi_behavior_version=coordinate.behavior_phi_version,
            phi_updated_version=coordinate.updated_phi_version,
            z_behavior_version=coordinate.behavior_z_version,
            z_updated_version=coordinate.updated_z_version,
            checkpoint_dir=str(publication_theta),
            phi_checkpoint_dir=str(recovery_phi),
            recovery_checkpoint_dir=str(recovery_root),
            official_checkpoint_dir=(
                str(official_root) if official_root is not None else ""
            ),
            official_checkpoint_saved=official_saved,
            exclusions={
                "training_method": "grpo_mace_bayesian_skill_evolution_disabled",
                "dataset_protocol": "mbpp_plus_project_adaptation_not_skillflow_joint_iid",
                "gradient_partition": (
                    "project_implementation:" + ",".join(map(str, partition_costs))
                ),
            },
            step_seconds=max(time.monotonic() - started_monotonic, 0.0),
            started_at=started_at,
            completed_at=_utc_now(),
        )
        (step_root / "ttb_training_summary.json").write_text(
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._advance_transaction(
            "AWAITING_PUBLICATION",
            recovery_checkpoint_dir=str(recovery_root),
            publication_checkpoint_dir=str(publication_theta),
            official_checkpoint_dir=(
                str(official_root) if official_root is not None else None
            ),
        )
        assert self._active_transaction is not None
        self._pending_publication = {
            "coordinate": coordinate,
            "summary": summary,
            "step_root": step_root,
            "recovery_root": recovery_root,
            "official_root": official_root,
            "publication_theta": publication_theta,
            "transaction": dict(self._active_transaction),
        }
        del theta_before, phi_before, z_before
        return summary

    def acknowledge_publication(
        self,
        receipt: Mapping[str, Any],
    ) -> TTBTrainingSummary:
        """Commit a step only after the new θ route and canary are verified.

        ``SGLangPolicyPublisher`` supplies the publication fields.  The runner
        must additionally bind ``server_weight_version`` from an exact
        post-switch Director receipt, because the current publisher interface
        does not expose that SGLang metadata itself.  This acknowledgement
        barrier is project engineering required to enforce on-policy batches.
        """

        self._require_setup()
        if self._faulted:
            raise RuntimeError(
                "the TTB trainer is faulted and cannot commit a publication"
            )
        pending = self._pending_publication
        if pending is None:
            raise RuntimeError("there is no pending TTB publication")
        if not isinstance(receipt, Mapping):
            raise ValueError("TTB publication receipt must be a mapping")
        coordinate = pending["coordinate"]
        assert isinstance(coordinate, TTBStepCoordinate)
        expected = {
            "success": True,
            "status": "published",
            "canary_succeeded": True,
            "training_performed": True,
            "policy_published": True,
            "gate_used": True,
            "gate_drained": True,
            "route_switch_requested": True,
            "route_switch_succeeded": True,
            "behavior_policy_version": coordinate.behavior_theta_version,
            "candidate_policy_version": coordinate.updated_theta_version,
            "new_policy_version": coordinate.updated_theta_version,
            "previous_adapter": coordinate.behavior_policy_adapter,
        }
        for name, value in expected.items():
            if receipt.get(name) != value:
                raise ValueError(
                    f"TTB publication receipt {name} does not match the pending step"
                )
        adapter_name = receipt.get("adapter_name")
        server_weight_version = receipt.get("server_weight_version")
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise ValueError("TTB publication receipt has no adapter name")
        if (
            not isinstance(server_weight_version, str)
            or not server_weight_version.strip()
        ):
            raise ValueError(
                "TTB publication receipt has no exact post-switch server weight version"
            )
        publication_theta = Path(pending["publication_theta"]).resolve()
        checkpoint_path = receipt.get("checkpoint_path")
        if (
            not isinstance(checkpoint_path, str)
            or Path(checkpoint_path).expanduser().resolve() != publication_theta
        ):
            raise ValueError(
                "TTB publication receipt checkpoint differs from the pending theta adapter"
            )

        receipt_payload = dict(receipt)
        receipt_payload["ttb_step"] = coordinate.step
        receipt_payload["acknowledged_at"] = _utc_now()
        step_root = Path(pending["step_root"])
        recovery_root = Path(pending["recovery_root"])
        official_root = pending["official_root"]
        checkpoint_roots = [recovery_root]
        if official_root is not None:
            checkpoint_roots.append(Path(official_root))
        for checkpoint_root in checkpoint_roots:
            state_path = checkpoint_root / "training_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                not isinstance(state, Mapping)
                or state.get("optimizer_committed_step") != coordinate.step
                or state.get("theta_version") != coordinate.updated_theta_version
            ):
                raise RuntimeError(
                    "TTB checkpoint state differs from the pending publication"
                )
            committed_state = dict(state)
            committed_state.update(
                {
                    "committed_step": coordinate.step,
                    "published_policy_adapter": adapter_name,
                    "published_server_weight_version": server_weight_version,
                    "publication_committed": True,
                    "publication_committed_at": _utc_now(),
                }
            )
            self._write_json_atomic(state_path, committed_state)
            self._write_json_atomic(
                checkpoint_root / "publication_receipt.json",
                receipt_payload,
            )

        policy_metadata_path = publication_theta / "policy_version.json"
        policy_metadata = json.loads(
            policy_metadata_path.read_text(encoding="utf-8")
        )
        if not isinstance(policy_metadata, Mapping):
            raise RuntimeError("TTB publication policy metadata is invalid")
        committed_policy_metadata = dict(policy_metadata)
        committed_policy_metadata.update(
            {
                "external_publication_pending": False,
                "published_adapter_name": adapter_name,
                "published_server_weight_version": server_weight_version,
                "publication_committed_at": _utc_now(),
            }
        )
        self._write_json_atomic(policy_metadata_path, committed_policy_metadata)
        self._write_json_atomic(
            step_root / "publication_receipt.json",
            receipt_payload,
        )
        self._advance_transaction(
            "COMMITTED",
            publication_committed=True,
            published_policy_adapter=adapter_name,
            published_server_weight_version=server_weight_version,
            completed_at=_utc_now(),
        )

        self._committed_step = coordinate.step
        self._current_theta_version = coordinate.updated_theta_version
        self._current_phi_version = coordinate.updated_phi_version
        self._current_z_version = coordinate.updated_z_version
        self._current_policy_adapter = adapter_name
        self._current_server_weight_version = server_weight_version
        summary = pending["summary"]
        assert isinstance(summary, TTBTrainingSummary)
        self._pending_publication = None
        self._active_transaction = None
        self._active_transaction_root = None
        return summary

    def close(self) -> None:
        """Release only model objects owned by this trainer."""

        if self._closed:
            return
        torch = self._torch
        self._learner_model = None
        self._replica_model = None
        self._learner_z = None
        self._replica_z = None
        self._learner_theta_params = []
        self._learner_phi_params = []
        self._replica_theta_params = []
        self._replica_phi_params = []
        self._theta_optimizer = None
        self._phi_optimizer = None
        self._z_optimizer = None
        self._tokenizer = None
        if torch is not None and torch.cuda.is_available():
            for device in (
                self.config.learner_device,
                self.config.gradient_replica_device,
            ):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        self._closed = True


__all__ = [
    "PartitionFunctionHead",
    "Qwen35TTBTrainer",
    "TTBStepCoordinate",
    "TTBTrainerConfig",
    "TTBTrainingSummary",
]
