from __future__ import annotations

import unittest

from src.interactive.config_loader import (
    ConfigurationError,
    expand_environment,
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
        self.assertEqual(config["gpu"]["director_training_physical"], 3)
        self.assertEqual(config["gpu"]["director_inference_physical"], 4)
        self.assertEqual(config["gpu"]["probe_skill_training_physical"], 5)

    def test_nonzero_exploration_reward_is_rejected(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["grpo"]["exploration_reward"] = 0.1
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)


if __name__ == "__main__":
    unittest.main()
