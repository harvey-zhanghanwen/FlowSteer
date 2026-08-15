from __future__ import annotations

import unittest

from src.interactive.sglang_manager import SGLangSupervisorManager


class SGLangSupervisorManagerTests(unittest.TestCase):
    def test_skillflow_qwen35_defaults_are_side_effect_free(self) -> None:
        manager = SGLangSupervisorManager()
        args = manager.server_args()
        self.assertEqual(args["model_path"], "Qwen/Qwen3.5-9B")
        self.assertEqual(args["served_model_name"], "supervisor_theta")
        self.assertEqual(args["reasoning_parser"], "qwen3")
        self.assertEqual(args["tool_call_parser"], "qwen3_coder")
        self.assertEqual(args["lora_target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(manager.api_base, "http://127.0.0.1:8015/v1")
        self.assertFalse(manager.is_alive())

    def test_runtime_limits_are_validated_without_importing_sglang(self) -> None:
        with self.assertRaises(ValueError):
            SGLangSupervisorManager(mem_fraction_static=1.0)
        with self.assertRaises(ValueError):
            SGLangSupervisorManager(context_length=0)


if __name__ == "__main__":
    unittest.main()
