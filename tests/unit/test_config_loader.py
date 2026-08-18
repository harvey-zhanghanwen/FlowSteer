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
        self.assertTrue(config["director"]["execute_on_edit"])
        self.assertEqual(config["director"]["history_window"], 4)
        self.assertEqual(config["agent_graph"]["max_agents"], 10)
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

    def test_declared_canvas_search_space_is_enforced(self) -> None:
        invalid_values = (
            ("max_agents", 0),
            ("executor_selection", "unseeded"),
            ("max_bidirectional_block_size", 3),
            ("require_unique_output", False),
            ("require_all_agents_reach_output", False),
        )
        for field, value in invalid_values:
            with self.subTest(field=field):
                config = load_yaml("config/training_agent_graph.yaml")
                config["agent_graph"][field] = value
                with self.assertRaises(ConfigurationError):
                    validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["execute_on_edit"] = False
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["history_window"] = 0
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["base_model"] = "Qwen/Qwen3-8B"
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["api_base"] = "https://provider.example/v1"
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_flowsteer_add_subgraph_action_profile_is_strict(self) -> None:
        subgraph_actions = [
            "add_subgraph",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "finish",
        ]
        config = load_yaml("config/training_agent_graph.yaml")
        config["agent_graph"]["actions"] = subgraph_actions
        config["agent_graph"]["max_agents_per_subgraph"] = 3
        validate_agent_graph_config(config)

        for invalid_limit in (None, 0, 2, 4, True):
            with self.subTest(invalid_limit=invalid_limit):
                invalid = load_yaml("config/training_agent_graph.yaml")
                invalid["agent_graph"]["actions"] = subgraph_actions
                if invalid_limit is None:
                    invalid["agent_graph"].pop("max_agents_per_subgraph", None)
                else:
                    invalid["agent_graph"]["max_agents_per_subgraph"] = invalid_limit
                with self.assertRaises(ConfigurationError):
                    validate_agent_graph_config(invalid)

        mixed = load_yaml("config/training_agent_graph.yaml")
        mixed["agent_graph"]["actions"] = [
            "add_subgraph",
            "add_agent",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "finish",
        ]
        mixed["agent_graph"]["max_agents_per_subgraph"] = 3
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(mixed)

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

    def test_hotpot_multiagent_catalog_uses_only_canary_backed_exact_ids(self) -> None:
        registry = load_model_registry(
            "config/model_catalog_hotpotqa_multiagent_v1.yaml"
        )
        self.assertEqual(
            {
                "qwen3.5-9b-local",
                "qwen3.5-flash",
                "qwen3.5-plus",
                "deepseek-v4-flash",
                "deepseek-v4-pro",
                "gpt-4o-mini",
                "minimax-m2.5",
                "minimax-m3",
            },
            set(registry.model_ids),
        )
        self.assertFalse(any("gemini" in model_id.lower() for model_id in registry.model_ids))
        self.assertFalse(any("grok" in model_id.lower() for model_id in registry.model_ids))

    def test_terminal_protocol_map_is_fail_closed(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["agent_graph"]["terminal_protocol_by_source"] = {
            "hotpotqa": "exact_single_answer_tag"
        }
        validate_agent_graph_config(config)
        config["agent_graph"]["terminal_protocol_by_source"] = {
            "hotpotqa": "prompt_only"
        }
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)


if __name__ == "__main__":
    unittest.main()
