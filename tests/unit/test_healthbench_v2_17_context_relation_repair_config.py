from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
V216_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_16_heldout20_authoritative_profile_choice.yaml"
)
V217_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_17_heldout20_context_relation_repair.yaml"
)


class HealthBenchV217ContextRelationRepairConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.v216 = yaml.safe_load(V216_CONFIG_PATH.read_text(encoding="utf-8"))
        self.v217 = yaml.safe_load(V217_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_fixed_population_and_scientific_condition_are_preserved(self) -> None:
        old_eval = self.v216["healthbench_professional_evaluation"]
        new_eval = self.v217["healthbench_professional_evaluation"]
        self.assertEqual(old_eval, new_eval)
        self.assertEqual(20, len(new_eval["task_ids"]))
        self.assertEqual(20, len(set(new_eval["task_ids"])))
        self.assertEqual(self.v216["data"], self.v217["data"])
        self.assertEqual(self.v216["director"], self.v217["director"])
        self.assertEqual(self.v216["agent_graph"], self.v217["agent_graph"])
        self.assertEqual(self.v216["evaluation"], self.v217["evaluation"])

    def test_database_web_search_and_model_pool_are_unchanged(self) -> None:
        old_tool = dict(self.v216["healthbench_tool_runtime"])
        new_tool = dict(self.v217["healthbench_tool_runtime"])
        old_tool.pop("condition_id")
        new_tool.pop("condition_id")
        self.assertEqual(old_tool, new_tool)
        self.assertTrue(new_tool["authoritative_web_search"]["enabled"])
        self.assertEqual("ncbi_pubmed_eutils", new_tool["authoritative_web_search"]["provider"])
        self.assertEqual(
            "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v2.yaml",
            self.v217["agent_graph"]["model_catalog_path"],
        )

    def test_new_namespace_does_not_resume_or_overwrite_v216(self) -> None:
        old_name = self.v216["experiment"]["condition_id"]
        new_name = self.v217["experiment"]["condition_id"]
        self.assertNotEqual(old_name, new_name)
        self.assertIn("v2_17", new_name)
        for path_value in self.v217["storage"].values():
            if isinstance(path_value, str) and ("/" in path_value):
                self.assertIn("v2_17", path_value)

    def test_training_and_skill_evolution_remain_disabled(self) -> None:
        self.assertFalse(self.v217["experiment"]["training_enabled"])
        self.assertFalse(self.v217["grpo"]["enabled"])
        self.assertEqual(0, self.v217["grpo"]["max_optimizer_updates"])
        self.assertFalse(self.v217["policy_sync"]["enabled"])
        self.assertFalse(self.v217["skills"]["enabled"])
        self.assertFalse(self.v217["gpu"]["training_enabled"])


if __name__ == "__main__":
    unittest.main()
