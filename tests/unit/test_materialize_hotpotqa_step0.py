from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.interactive.hotpot_step0 import (
    ADAPTER_NAME,
    HotpotStep0Config,
    INITIALIZATION_SEED,
    PEFT_ADAPTER_NAME,
    POLICY_VERSION,
    bind_initial_trainable_state,
    materialize_hotpot_step0,
    preflight_hotpot_step0,
)


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/materialize_hotpotqa_step0.py"
_SPEC = importlib.util.spec_from_file_location("materialize_hotpotqa_step0", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


class _FakeParameter:
    def __init__(self, value: str, *, requires_grad: bool = True) -> None:
        self.value = value
        self.requires_grad = requires_grad

    def detach(self) -> _FakeParameter:
        return self

    def to(self, *, device: str) -> _FakeParameter:
        if device != "cpu":
            raise AssertionError("initial-state binding must use CPU")
        return self

    def contiguous(self) -> _FakeParameter:
        return self

    def clone(self) -> _FakeParameter:
        return _FakeParameter(self.value, requires_grad=self.requires_grad)


class _FakePeftModel:
    def __init__(self) -> None:
        self.parameter = _FakeParameter("initial")
        self.selected_adapter = None

    def named_parameters(self):
        return iter((("layer.lora_A.theta.weight", self.parameter),))

    def set_adapter(self, name: str) -> None:
        self.selected_adapter = name

    def save_pretrained(
        self,
        output: Path,
        *,
        selected_adapters: list[str],
        safe_serialization: bool,
    ) -> None:
        if selected_adapters != [PEFT_ADAPTER_NAME] or not safe_serialization:
            raise AssertionError("theta must use safe PEFT serialization")
        adapter = Path(output) / PEFT_ADAPTER_NAME
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"initial-lora")


def _runtime_modules(model: _FakePeftModel):
    torch = ModuleType("torch")
    torch.bfloat16 = object()
    torch.manual_seed = MagicMock()
    torch.use_deterministic_algorithms = MagicMock()
    torch.backends = SimpleNamespace(
        cudnn=SimpleNamespace(deterministic=False, benchmark=True)
    )
    torch.equal = lambda left, right: left.value == right.value

    lora_config = MagicMock(name="LoraConfig")
    peft = ModuleType("peft")
    peft.LoraConfig = lora_config
    peft.get_peft_model = MagicMock(return_value=model)

    loader = MagicMock(name="AutoModelForMultimodalLM")
    loader.from_pretrained.return_value = object()
    transformers = ModuleType("transformers")
    transformers.AutoModelForMultimodalLM = loader
    return torch, peft, transformers, lora_config, loader


class HotpotStep0PreflightTests(unittest.TestCase):
    def test_preflight_does_not_import_or_load_model_runtime(self) -> None:
        with TemporaryDirectory() as directory:
            config = HotpotStep0Config(
                model_path="/model/need-not-exist-for-preflight",
                output_dir=str(Path(directory) / "step0"),
            )
            with patch.dict(sys.modules, {"torch": None, "peft": None}):
                receipt = preflight_hotpot_step0(config)

        self.assertEqual(receipt["status"], "ready_to_materialize")
        self.assertFalse(receipt["model_load_performed"])
        self.assertFalse(receipt["will_write"])
        self.assertEqual(receipt["initialization_seed"], INITIALIZATION_SEED)

    def test_preflight_and_materialization_reject_existing_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "step0"
            output.mkdir()
            config = HotpotStep0Config("/model", str(output))
            with self.assertRaises(FileExistsError):
                preflight_hotpot_step0(config)
            with self.assertRaises(FileExistsError):
                materialize_hotpot_step0(config)

    def test_nonzero_dropout_is_not_a_formal_step_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero LoRA dropout"):
            HotpotStep0Config("/model", "/output", lora_dropout=0.1)

    def test_cli_defaults_to_preflight(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "step0"
            with patch("builtins.print") as printer:
                status = main(
                    [
                        "--model-path",
                        "/model/need-not-exist-for-preflight",
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(status, 0)
        payload = json.loads(printer.call_args.args[0])
        self.assertEqual(payload["status"], "ready_to_materialize")


class HotpotStep0MaterializationTests(unittest.TestCase):
    def test_materialize_uses_fixed_seed_and_writes_sglang_theta(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            output = root / "step0"
            config = HotpotStep0Config(str(model_path), str(output))
            model = _FakePeftModel()
            torch, peft, transformers, lora_config, loader = _runtime_modules(model)

            with patch.dict(
                sys.modules,
                {"torch": torch, "peft": peft, "transformers": transformers},
            ):
                receipt = materialize_hotpot_step0(config)

            torch.manual_seed.assert_called_once_with(INITIALIZATION_SEED)
            torch.use_deterministic_algorithms.assert_called_once_with(True)
            loader.from_pretrained.assert_called_once_with(
                str(model_path.resolve()),
                dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            lora_config.assert_called_once_with(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.assertEqual(model.selected_adapter, PEFT_ADAPTER_NAME)
            adapter = output / PEFT_ADAPTER_NAME
            self.assertTrue((adapter / "adapter_config.json").is_file())
            self.assertTrue((adapter / "adapter_model.safetensors").is_file())
            metadata = json.loads(
                (adapter / "policy_version.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["policy_version"], POLICY_VERSION)
            self.assertEqual(metadata["format"], "flowsteer-formal-initial-policy-v1")
            self.assertEqual(metadata["committed_step"], 0)
            self.assertEqual(metadata["adapter_name"], ADAPTER_NAME)
            self.assertEqual(metadata["optimizer_updates"], 0)
            self.assertFalse(metadata["training_performed"])
            self.assertNotIn("model_path", metadata)
            self.assertEqual(receipt["status"], "materialized")

    def test_bound_initial_state_rejects_mutation_before_save(self) -> None:
        model = _FakePeftModel()
        bound = bind_initial_trainable_state(model)
        model.parameter.value = "changed"
        torch, _, _, _, _ = _runtime_modules(model)
        with TemporaryDirectory() as directory:
            config = HotpotStep0Config("/model", str(Path(directory) / "step0"))
            from src.interactive.hotpot_step0 import save_initial_checkpoint

            with self.assertRaisesRegex(RuntimeError, "changed after"):
                save_initial_checkpoint(
                    torch=torch,
                    model=model,
                    config=config,
                    bound=bound,
                )
            self.assertFalse(config.output_path.exists())


if __name__ == "__main__":
    unittest.main()
