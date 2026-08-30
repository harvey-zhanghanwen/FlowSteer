from __future__ import annotations

from pathlib import Path
import unittest

from src.interactive.config_loader import (
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.director import (
    STEPWISE_SUBGRAPH_DIRECTOR_PROMPT_VERSION_V2,
    director_system_prompt_for_version,
    encode_director_transcript,
)

from tests.unit.test_alfworld_multiagent_v7 import _environment


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ALFWorldMultiAgentCurrentTests(unittest.TestCase):
    def test_v10_is_evaluation_only_on_gpu0_with_new_receipt_paths(self) -> None:
        config = load_yaml(
            str(
                _REPOSITORY_ROOT
                / "config"
                / "evaluation_alfworld_stepwise_multiagent_v10.yaml"
            )
        )
        validate_agent_graph_config(config)

        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["grpo"]["enabled"])
        self.assertFalse(config["skills"]["enabled"])
        self.assertEqual(
            STEPWISE_SUBGRAPH_DIRECTOR_PROMPT_VERSION_V2,
            config["experiment"]["prompt_version"],
        )
        self.assertEqual(0, config["gpu"]["rollout_physical"])
        self.assertEqual(0, config["gpu"]["supervisor_gpu_id"])
        self.assertIn(
            "alfworld_stepwise_multiagent_v10",
            config["storage"]["trajectories_path"],
        )

    def test_director_defines_react_as_execution_mode_and_keeps_roles_free(
        self,
    ) -> None:
        prompt = director_system_prompt_for_version(
            STEPWISE_SUBGRAPH_DIRECTOR_PROMPT_VERSION_V2
        )

        self.assertIn("ReAct is an execution mode, not an Agent role", prompt)
        self.assertIn("functional decomposition", prompt)
        self.assertIn("distinct complementary contracts", prompt)
        for fixed_role in ("Navigator", "Manipulator", "Planner", "Verifier"):
            self.assertNotIn(fixed_role, prompt)
        transcript = encode_director_transcript(
            (
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Task: test"},
            )
        )
        self.assertIn("flowsteer.director.transcript.v1", transcript)

    def test_initial_collaborator_receives_public_text_simulator_protocol(
        self,
    ) -> None:
        env = _environment("put two toiletpaper in toilet.")
        problem = env._runtime_problem()

        self.assertIn("task-scoped ALFWorld text-simulator episode", problem)
        self.assertIn("task-relevant artifact", problem)
        self.assertNotIn("reward", problem.casefold())
        self.assertNotIn("reference plan", problem.casefold())


if __name__ == "__main__":
    unittest.main()
