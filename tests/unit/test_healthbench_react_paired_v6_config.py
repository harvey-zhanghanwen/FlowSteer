from __future__ import annotations

from pathlib import Path
import unittest

from src.interactive.config_loader import load_yaml, validate_agent_graph_config
from src.interactive.director import (
    SCALAR_DIRECTOR_PROMPT_VERSION_V5,
    SCALAR_DIRECTOR_SYSTEM_PROMPT_V5,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / (
    "evaluation_healthbench_professional_react_paired_two_phase_artifact_v6_gpt54_rubric.yaml"
)


class HealthBenchReactPairedV6ConfigTest(unittest.TestCase):
    def test_v6_is_evaluation_only_react_paired_and_topology_neutral(self) -> None:
        config = load_yaml(CONFIG)
        validate_agent_graph_config(config)

        director = config["director"]
        graph = config["agent_graph"]
        bounded = config["healthbench_professional_evaluation"]
        tool_runtime = config["healthbench_tool_runtime"]
        self.assertEqual(SCALAR_DIRECTOR_PROMPT_VERSION_V5, config["experiment"]["prompt_version"])
        self.assertTrue(director["two_phase_generation"])
        self.assertTrue(director["chat_template_enable_thinking"])
        self.assertEqual(512, director["max_reasoning_tokens"])
        self.assertEqual(4096, director["max_action_tokens"])
        self.assertEqual(1.05, director["repetition_penalty"])
        self.assertTrue(director["execute_on_edit"])
        self.assertEqual("producer_context_exact_dedup_v1", graph["artifact_communication_profile"])
        self.assertTrue(graph["reuse_unchanged_agent_inputs"])
        self.assertFalse(graph["finish_only_when_admissible"])
        self.assertFalse(graph["require_format_agent"])
        self.assertEqual("none", graph["semantic_protocol_by_source"]["healthbench_professional"])
        self.assertEqual(525, bounded["sample_count"])
        self.assertEqual("test", bounded["split"])
        self.assertFalse(config["grpo"]["enabled"])
        self.assertEqual(0, config["grpo"]["max_optimizer_updates"])
        self.assertFalse(config["skills"]["enabled"])
        self.assertFalse(config["exploration"]["enabled"])
        self.assertEqual(
            [
                {
                    "execution_mode": "react",
                    "allowed_tools": ["healthbench-authoritative.search"],
                }
            ],
            tool_runtime["execution_profile_allowlist"],
        )

    def test_v6_prompt_adds_no_fixed_medical_role_or_topology(self) -> None:
        prompt = SCALAR_DIRECTOR_SYSTEM_PROMPT_V5.casefold()
        for prohibited in (
            "doctor",
            "researcher",
            "reviewer",
            "verifier",
            "doctor ->",
            "reasoner ->",
        ):
            self.assertNotIn(prohibited, prompt)
        self.assertIn("exactly one valid json action", prompt)
        self.assertIn("additional agents or relations only when", prompt)
        self.assertIn("avoid redundant execution", prompt)

    def test_v6_model_visible_config_contains_no_evaluator_truth(self) -> None:
        config = load_yaml(CONFIG, expand_env=False)
        model_visible = {
            "director": config["director"],
            "agent_graph": config["agent_graph"],
            "direct_contract": config["healthbench_professional_evaluation"]["direct_contract"],
            "direct_completion_condition": config["healthbench_professional_evaluation"]["direct_completion_condition"],
        }
        rendered = repr(model_visible).casefold()
        for evaluator_only in (
            "rubric_items",
            "physician_response",
            "canary_string",
            "ground_truth",
            "reference_response",
        ):
            self.assertNotIn(evaluator_only, rendered)


if __name__ == "__main__":
    unittest.main()
