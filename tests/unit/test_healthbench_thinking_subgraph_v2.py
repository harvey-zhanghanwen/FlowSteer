from pathlib import Path
import unittest

import yaml

from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION_V14,
    DIRECTOR_SYSTEM_PROMPT_V14,
    director_system_prompt_for_version,
)


ROOT = Path(__file__).resolve().parents[2]


class HealthBenchThinkingSubgraphV2Tests(unittest.TestCase):
    def test_neutral_v14_prompt_keeps_open_search_space(self) -> None:
        self.assertEqual(
            DIRECTOR_SYSTEM_PROMPT_V14,
            director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V14),
        )
        prompt = DIRECTOR_SYSTEM_PROMPT_V14.casefold()
        self.assertIn("distinct free-text contract", prompt)
        self.assertIn("complete user-facing response", prompt)
        self.assertIn("producer artifact", prompt)
        self.assertIn("already on the canvas", prompt)
        self.assertIn("otherwise use reasoning", prompt)
        self.assertIn("without adding assumptions", prompt)
        self.assertIn("set output_agent_id in that same action", prompt)
        self.assertIn("without executing the agent again", prompt)
        self.assertIn("preserve critical quantities", prompt)
        self.assertIn("unordered agent endpoint pair at most once", prompt)
        self.assertNotIn("doctor", prompt)
        self.assertNotIn("researcher", prompt)
        self.assertNotIn("rubric", prompt)

    def test_evaluation_condition_is_inference_only_and_reference_graded(self) -> None:
        path = ROOT / "config/evaluation_healthbench_professional_best_thinking_subgraph_v2.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            DIRECTOR_PROMPT_VERSION_V14,
            config["experiment"]["prompt_version"],
        )
        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertEqual(525, config["healthbench_professional_evaluation"]["sample_count"])
        self.assertEqual(
            "openai_simple_evals_healthbench_professional_reference",
            config["evaluation"]["healthbench_grader_mode"],
        )
        self.assertEqual(
            [
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "finish",
            ],
            config["agent_graph"]["actions"],
        )
        self.assertEqual(3, config["agent_graph"]["max_agents_per_subgraph"])
        self.assertTrue(config["agent_graph"]["require_informative_contracts"])
        self.assertTrue(config["agent_graph"]["reuse_unchanged_agent_inputs"])
        self.assertFalse(config["skills"]["enabled"])
        self.assertFalse(config["grpo"]["enabled"])
        self.assertFalse(config["policy_sync"]["enabled"])

        catalog_path = ROOT / config["agent_graph"]["model_catalog_path"]
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        metadata = catalog["models"][0]["metadata"]
        self.assertEqual("true", metadata["chat_template_enable_thinking"])
        self.assertEqual("4096", metadata["max_tokens"])
        self.assertEqual("4096", metadata["thinking_budget"])
        self.assertEqual("1.10", metadata["repetition_penalty"])


if __name__ == "__main__":
    unittest.main()
