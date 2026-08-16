"""Materialize the untrained HotpotQA Director LoRA at policy step zero.

The initialization transaction follows SkillFlow's
``scripts/build_gate_4c_initial_policy.py``: fix the process seed, bind the
fresh trainable state before it can be changed, and publish to a new path.
The checkpoint shape is a necessary project adaptation.  FlowSteer's
Qwen3.5 Director has one SGLang-facing ``theta`` action adapter, whereas the
referenced SkillFlow backbone saves forward/backward adapters plus a Z head.
This module therefore reuses the Qwen3.5 multimodal/PEFT loader already used
by ``smoke_trainer.py`` and saves only the untrained ``theta`` adapter.

This module contains no optimizer, backward, rollout, API, or service path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


INITIALIZATION_SEED = 20_260_811
POLICY_STEP = 0
POLICY_VERSION = "qwen35-9b-hotpot-step-000000"
ADAPTER_NAME = "theta_hotpot_step_000000"
PEFT_ADAPTER_NAME = "theta"
PROVENANCE_FORMAT = "flowsteer-formal-initial-policy-v1"


@dataclass(frozen=True)
class HotpotStep0Config:
    """Fixed inputs for the never-updated HotpotQA Director adapter."""

    model_path: str
    output_dir: str
    base_model_id: str = "Qwen/Qwen3.5-9B"
    revision: str = "74be52bb6bd9f0e9e68dacb72636b75649197983"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    def __post_init__(self) -> None:
        for field, value in (
            ("model_path", self.model_path),
            ("output_dir", self.output_dir),
            ("base_model_id", self.base_model_id),
            ("revision", self.revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty text")
        if type(self.lora_rank) is not int or self.lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer")
        if type(self.lora_alpha) is not int or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be a positive integer")
        if self.lora_dropout != 0.0:
            raise ValueError("formal Step 0 requires zero LoRA dropout")
        if (
            not isinstance(self.lora_target_modules, tuple)
            or not self.lora_target_modules
            or any(
                not isinstance(module, str) or not module.strip()
                for module in self.lora_target_modules
            )
            or len(set(self.lora_target_modules)) != len(self.lora_target_modules)
        ):
            raise ValueError("lora_target_modules must be unique non-empty strings")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def model_local_path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    def public_provenance(self, *, trainable_tensor_count: int) -> dict[str, Any]:
        """Return path-free, non-secret metadata for the initial adapter."""

        if type(trainable_tensor_count) is not int or trainable_tensor_count <= 0:
            raise ValueError("trainable_tensor_count must be a positive integer")
        return {
            "adapter_name": ADAPTER_NAME,
            "base_model_id": self.base_model_id,
            "format": PROVENANCE_FORMAT,
            "initialization_seed": INITIALIZATION_SEED,
            "lora": {
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "rank": self.lora_rank,
                "target_modules": list(self.lora_target_modules),
            },
            "optimizer_updates": 0,
            "committed_step": POLICY_STEP,
            "peft_adapter_name": PEFT_ADAPTER_NAME,
            "policy_step": POLICY_STEP,
            "policy_version": POLICY_VERSION,
            "revision": self.revision,
            "source": {
                "initialization_transaction": (
                    "SkillFlow/scripts/build_gate_4c_initial_policy.py::main"
                ),
                "initial_state_binding": (
                    "SkillFlow/src/skillev/policy/hf_backbone.py::"
                    "bind_initial_trainable_state"
                ),
                "model_loader": (
                    "FlowSteer/src/interactive/smoke_trainer.py::"
                    "Qwen35OnePassSmokeTrainer._load_models"
                ),
                "save_once_semantics": (
                    "SkillFlow/src/skillev/policy/hf_backbone.py::save_checkpoint"
                ),
            },
            "trainable_tensor_count": trainable_tensor_count,
            "training_performed": False,
        }


@dataclass(frozen=True)
class BoundInitialTrainableState:
    """In-memory binding used to reject mutation before Step-0 publication."""

    names: tuple[str, ...]
    tensors: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.tensors):
            raise ValueError("initial trainable state must contain named tensors")


def preflight_hotpot_step0(config: HotpotStep0Config) -> Mapping[str, Any]:
    """Validate Step-0 settings without importing or loading model libraries."""

    if not isinstance(config, HotpotStep0Config):
        raise TypeError("config must be HotpotStep0Config")
    if config.output_path.exists():
        raise FileExistsError(config.output_path)
    return {
        "adapter_name": ADAPTER_NAME,
        "initialization_seed": INITIALIZATION_SEED,
        "model_load_performed": False,
        "optimizer_or_backward_performed": False,
        "output": str(config.output_path),
        "policy_step": POLICY_STEP,
        "policy_version": POLICY_VERSION,
        "status": "ready_to_materialize",
        "will_overwrite": False,
        "will_write": False,
    }


def bind_initial_trainable_state(model: Any) -> BoundInitialTrainableState:
    """Bind fresh theta tensors, adapting SkillFlow's initial-state gate."""

    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if not selected:
        raise RuntimeError("fresh theta adapter exposes no trainable LoRA tensors")
    return BoundInitialTrainableState(
        names=tuple(name for name, _ in selected),
        tensors=tuple(
            parameter.detach().to(device="cpu").contiguous().clone()
            for _, parameter in selected
        ),
    )


def _require_bound_state_unchanged(
    torch: Any,
    model: Any,
    bound: BoundInitialTrainableState,
) -> None:
    current = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    }
    if tuple(current) != bound.names or any(
        not torch.equal(current[name], initial)
        for name, initial in zip(bound.names, bound.tensors)
    ):
        raise RuntimeError("theta tensors changed after initial-state binding")


def save_initial_checkpoint(
    *,
    torch: Any,
    model: Any,
    config: HotpotStep0Config,
    bound: BoundInitialTrainableState,
) -> Path:
    """Save one SGLang-compatible named adapter to a fresh directory."""

    output = config.output_path
    if output.exists():
        raise FileExistsError(output)
    _require_bound_state_unchanged(torch, model, bound)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output,
        selected_adapters=[PEFT_ADAPTER_NAME],
        safe_serialization=True,
    )
    adapter = output / PEFT_ADAPTER_NAME
    required = (
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("PEFT did not materialize a complete theta adapter")
    metadata_path = adapter / "policy_version.json"
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(
            config.public_provenance(trainable_tensor_count=len(bound.names)),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return adapter


def materialize_hotpot_step0(config: HotpotStep0Config) -> Mapping[str, Any]:
    """Load Qwen3.5 on CPU and publish its untouched theta initialization."""

    preflight_hotpot_step0(config)
    if not config.model_local_path.is_dir():
        raise NotADirectoryError(config.model_local_path)
    if not (config.model_local_path / "config.json").is_file():
        raise FileNotFoundError(config.model_local_path / "config.json")

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:  # pragma: no cover - real runtime only
        raise RuntimeError(
            "Step-0 materialization requires torch, transformers, and peft"
        ) from exc

    # Direct SkillFlow initialization semantics.  CPU placement makes this an
    # offline artifact build and cannot start or alter a rollout service.
    torch.manual_seed(INITIALIZATION_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    base = AutoModelForMultimodalLM.from_pretrained(
        str(config.model_local_path),
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config, adapter_name=PEFT_ADAPTER_NAME)
    model.set_adapter(PEFT_ADAPTER_NAME)
    bound = bind_initial_trainable_state(model)
    adapter = save_initial_checkpoint(
        torch=torch,
        model=model,
        config=config,
        bound=bound,
    )
    return {
        "adapter_checkpoint": str(adapter),
        "initialization_seed": INITIALIZATION_SEED,
        "optimizer_or_backward_performed": False,
        "policy_step": POLICY_STEP,
        "policy_version": POLICY_VERSION,
        "status": "materialized",
        "trainable_tensor_count": len(bound.names),
    }


__all__: Sequence[str] = (
    "ADAPTER_NAME",
    "BoundInitialTrainableState",
    "HotpotStep0Config",
    "INITIALIZATION_SEED",
    "POLICY_STEP",
    "POLICY_VERSION",
    "PEFT_ADAPTER_NAME",
    "bind_initial_trainable_state",
    "materialize_hotpot_step0",
    "preflight_hotpot_step0",
    "save_initial_checkpoint",
)
