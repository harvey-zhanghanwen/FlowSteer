from __future__ import annotations

import unittest

from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DirectorError,
    DirectorResponse,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


class ScriptedDirector:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []

    async def propose(self, prompt):
        self.prompts.append(prompt)
        return DirectorResponse(self.actions.pop(0), {"policy_version": "test"})


class FakeGateway:
    async def generate(self, request):
        return AgentResponse(f"answer from {request.agent.id}")


def registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("provider", endpoint="http://local/v1")],
        [
            ModelSpec("qwen", "provider", cheap_weight=10, fast_weight=10),
            ModelSpec("other", "provider", cheap_weight=1, fast_weight=1),
        ],
    )


class DirectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end_scripted_canvas_and_execution(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                '{"action":"add_agent","agent_id":"solver","model_id":"qwen","contract":"solve"}',
                '{"action":"set_output","agent_id":"solver"}',
                '{"action":"finish"}',
            ]
        )
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())
        result = await AgentGraphOrchestrator(model_registry, client, seed=1).run(env, "2+2?")
        self.assertEqual(result.final_answer, "answer from solver")
        self.assertEqual(len(result.turns), 3)
        self.assertIn("weighted_preferred_model", client.prompts[0])
        self.assertEqual(result.final_graph["output_agent_id"], "solver")

    async def test_round_limit_is_explicit_failure(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            ['{"action":"add_agent","agent_id":"solver","model_id":"qwen","contract":"solve"}']
        )
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())
        with self.assertRaises(DirectorError):
            await AgentGraphOrchestrator(model_registry, client, max_rounds=1).run(env, "task")


if __name__ == "__main__":
    unittest.main()
