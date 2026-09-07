from __future__ import annotations

import unittest
from pathlib import Path
import os

from src.interactive.sglang_manager import (
    SGLangSupervisorManager,
    _configure_worker_environment,
)


class SGLangSupervisorManagerTests(unittest.TestCase):
    def test_skillflow_qwen35_defaults_are_side_effect_free(self) -> None:
        manager = SGLangSupervisorManager()
        args = manager.server_args()
        self.assertEqual(args["model_path"], "Qwen/Qwen3.5-9B")
        self.assertEqual(args["served_model_name"], "supervisor_theta")
        self.assertEqual(args["reasoning_parser"], "qwen3")
        self.assertEqual(args["tool_call_parser"], "qwen3_coder")
        self.assertEqual(args["host"], "127.0.0.1")
        self.assertEqual(args["schedule_policy"], "lpm")
        self.assertEqual(args["sampling_backend"], "pytorch")
        self.assertTrue(args["enable_multimodal"])
        self.assertEqual(args["lora_target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(manager.api_base, "http://127.0.0.1:8015/v1")
        self.assertFalse(manager.is_alive())

    def test_explicit_tokenizer_path_is_forwarded(self) -> None:
        manager = SGLangSupervisorManager(tokenizer_path="/models/qwen-tokenizer")
        self.assertEqual(manager.server_args()["tokenizer_path"], "/models/qwen-tokenizer")

    def test_runtime_limits_are_validated_without_importing_sglang(self) -> None:
        with self.assertRaises(ValueError):
            SGLangSupervisorManager(mem_fraction_static=1.0)
        with self.assertRaises(ValueError):
            SGLangSupervisorManager(context_length=0)

    def test_worker_environment_exposes_venv_tools_and_matching_cuda(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "venv" / "bin" / "python"
            executable.parent.mkdir(parents=True)
            cuda_root = root / "toolkits"
            nvcc = cuda_root / "cuda-12.9" / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.touch()
            previous_path = os.environ.get("PATH")
            previous_cuda_home = os.environ.pop("CUDA_HOME", None)
            previous_deepgemm = os.environ.pop("SGLANG_ENABLE_JIT_DEEPGEMM", None)
            os.environ["PATH"] = "/usr/bin"
            try:
                _configure_worker_environment(
                    cuda_version="12.9",
                    executable=str(executable),
                    cuda_root=cuda_root,
                )
                self.assertEqual(
                    os.environ["PATH"].split(os.pathsep)[0],
                    str(executable.parent),
                )
                self.assertEqual(os.environ["CUDA_HOME"], str(cuda_root / "cuda-12.9"))
                self.assertEqual(os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"], "0")
            finally:
                if previous_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = previous_path
                if previous_cuda_home is None:
                    os.environ.pop("CUDA_HOME", None)
                else:
                    os.environ["CUDA_HOME"] = previous_cuda_home
                if previous_deepgemm is None:
                    os.environ.pop("SGLANG_ENABLE_JIT_DEEPGEMM", None)
                else:
                    os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = previous_deepgemm


if __name__ == "__main__":
    unittest.main()
