from pathlib import Path
import json
import unittest

import yaml

from src.interactive.director import DIRECTOR_PROMPT_VERSION_V14


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_5.yaml"
)
CATALOG_PATH = (
    ROOT
    / "config/model_catalog_healthbench_professional_mixed_all_thinking_v1.yaml"
)
SOURCE_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_best_thinking_subgraph_v2.yaml"
)


class HealthBenchMixedAllThinkingV25ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
        self.source_config = yaml.safe_load(
            SOURCE_CONFIG_PATH.read_text(encoding="utf-8")
        )

    def test_direct_condition_is_exactly_reused_and_graph_namespace_is_new(self) -> None:
        bounded = self.config["healthbench_professional_evaluation"]
        source = self.source_config["healthbench_professional_evaluation"]
        for key in (
            "dataset_key",
            "stage",
            "split",
            "benchmark_slice",
            "selection",
            "sample_count",
            "direct_model_id",
            "direct_protocol",
            "direct_contract",
            "direct_generation_seed",
        ):
            self.assertEqual(source[key], bounded[key], key)
        self.assertEqual(
            "artifacts/healthbench_professional_best_thinking_subgraph_v2_4/"
            "evaluation/direct_predictions.jsonl",
            bounded["direct_reused_from"],
        )
        self.assertFalse(bounded["protocol_equivalent_to_direct"])
        self.assertEqual(
            "healthbench_professional_mixed_all_thinking_v2_5",
            self.config["experiment"]["condition_id"],
        )
        self.assertIn(
            "healthbench_professional_mixed_all_thinking_v2_5",
            self.config["storage"]["trajectories_path"],
        )

    def test_director_is_fixed_local_but_each_executor_may_choose_any_arm(self) -> None:
        self.assertEqual(
            DIRECTOR_PROMPT_VERSION_V14,
            self.config["experiment"]["prompt_version"],
        )
        self.assertEqual("sglang", self.config["director"]["backend"])
        self.assertEqual(
            "supervisor_theta", self.config["director"]["served_model_name"]
        )
        self.assertTrue(self.config["director"]["chat_template_enable_thinking"])
        self.assertEqual(
            "director_catalog_choice",
            self.config["agent_graph"]["executor_selection"],
        )
        self.assertEqual("free_text", self.config["agent_graph"]["contract_type"])
        self.assertFalse(self.config["agent_graph"]["require_format_agent"])

    def test_every_catalog_arm_is_equal_weight_and_thinking_enabled(self) -> None:
        models = self.catalog["models"]
        self.assertEqual(
            {
                "qwen3.5-9b-local",
                "qwen3.5-flash",
                "deepseek-v4-flash",
                "MiniMax-M3",
            },
            {model["model_id"] for model in models},
        )
        for model in models:
            self.assertEqual(1.0, model["selection_weight"])
            self.assertEqual(1.0, model["cheap_weight"])
            self.assertEqual(1.0, model["fast_weight"])
            metadata = model["metadata"]
            self.assertEqual("true", metadata["chat_template_enable_thinking"])
            self.assertGreaterEqual(int(metadata["max_tokens"]), 4096)
            self.assertNotIn("doctor", metadata["profile"].casefold())
            self.assertNotIn("reviewer", metadata["profile"].casefold())
        local = next(
            model for model in models if model["model_id"] == "qwen3.5-9b-local"
        )
        self.assertEqual("4096", local["metadata"]["thinking_budget"])
        for remote in (
            model for model in models if model["model_id"] != "qwen3.5-9b-local"
        ):
            self.assertNotIn("supports_top_k", remote["metadata"])
            self.assertNotIn("supports_repetition_penalty", remote["metadata"])

    def test_capability_receipt_covers_every_admitted_arm(self) -> None:
        evidence_path = (
            ROOT
            / "artifacts/model_capability_canary/"
            "healthbench_mixed_all_thinking_v1.json"
        )
        receipt = json.loads(evidence_path.read_text(encoding="utf-8"))
        admitted = receipt["admitted_models"]
        self.assertEqual(
            {model["model_id"] for model in self.catalog["models"]},
            {model["model_id"] for model in admitted},
        )
        self.assertTrue(receipt["director_remains_fixed"])
        self.assertEqual(
            "equal_weight_no_role_binding", receipt["selection_policy"]
        )
        self.assertTrue(
            all(model["requested_thinking"] is True for model in admitted)
        )
        for remote in admitted[1:]:
            self.assertTrue(remote["reasoning_content_present"])

    def test_training_and_skills_remain_disabled(self) -> None:
        self.assertFalse(self.config["experiment"]["training_enabled"])
        self.assertFalse(self.config["grpo"]["enabled"])
        self.assertEqual(0, self.config["grpo"]["max_optimizer_updates"])
        self.assertFalse(self.config["policy_sync"]["enabled"])
        self.assertFalse(self.config["skills"]["enabled"])


if __name__ == "__main__":
    unittest.main()
