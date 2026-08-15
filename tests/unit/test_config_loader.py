from __future__ import annotations

import unittest

from src.interactive.config_loader import (
    ConfigurationError,
    expand_environment,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)


class ConfigLoaderTests(unittest.TestCase):
    def test_environment_defaults_and_required_variables(self) -> None:
        value = {"a": "${SET}", "b": "x-${MISSING:-fallback}", "c": ["${SET}"]}
        self.assertEqual(
            expand_environment(value, {"SET": "value"}),
            {"a": "value", "b": "x-fallback", "c": ["value"]},
        )
        with self.assertRaises(ConfigurationError):
            expand_environment("${REQUIRED}", {})

    def test_checked_in_agent_graph_config_obeys_strict_invariants(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        validate_agent_graph_config(config)
        self.assertEqual(config["director"]["base_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(config["director"]["backend"], "sglang")
        self.assertEqual(config["gpu"]["learner_physical"], 3)
        self.assertEqual(config["gpu"]["rollout_physical"], 4)
        self.assertEqual(config["gpu"]["gradient_replica_physical"], 5)
        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["grpo"]["enabled"])

    def test_nonzero_exploration_reward_is_rejected(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["grpo"]["exploration_reward"] = 0.1
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_architecture_phase_cannot_enable_training(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["gpu"]["training_enabled"] = True
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_smoke_config_is_strictly_bounded(self) -> None:
        config = load_yaml("config/training_agentgraph_smoke.yaml")
        validate_agent_graph_config(config)

        self.assertEqual(config["experiment"]["phase"], "smoke_training")
        self.assertTrue(config["experiment"]["training_enabled"])
        self.assertEqual(config["data"]["smoke"]["tasks_per_dataset"], 2)
        self.assertEqual(config["data"]["smoke"]["expected_total_tasks"], 14)
        self.assertEqual(len(config["data"]["smoke"]["source_order"]), 7)
        self.assertEqual(config["grpo"]["samples_per_problem"], 2)
        self.assertEqual(config["grpo"]["expected_rollout_count"], 28)
        self.assertEqual(config["grpo"]["max_optimizer_updates"], 1)
        self.assertEqual(config["gpu"]["oom_policy"]["micro_batch_schedule"], [4, 2, 1])
        self.assertTrue(config["policy_sync"]["enabled"])
        self.assertEqual(config["policy_sync"]["adapter_name_prefix"], "theta_smoke_step_")
        self.assertEqual(config["policy_sync"]["post_update_canary_count"], 1)
        self.assertEqual(
            config["policy_sync"]["canary"]["adapter_selector_field"],
            "lora_path",
        )
        self.assertFalse(config["exploration"]["enabled"])
        self.assertFalse(config["skills"]["enabled"])

    def test_example_catalog_contains_only_verified_smoke_ids(self) -> None:
        registry = load_model_registry("config/model_catalog.yaml.example")
        expected_names = {
            "supervisor_theta",
            "qwen3.5-flash",
            "deepseek-v4-flash",
            "gpt-4o-mini",
            "grok-4-1-fast-non-reasoning",
            "MiniMax-M2.5",
        }
        actual_names = {
            registry.require_model(model_id).model_name for model_id in registry.model_ids
        }
        self.assertEqual(actual_names, expected_names)
        self.assertFalse(any("gemini" in model_id.lower() for model_id in registry.model_ids))


if __name__ == "__main__":
    unittest.main()
