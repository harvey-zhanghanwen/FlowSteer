from __future__ import annotations

import json
import unittest

from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DirectorResponse,
    OpenAIDirectorClient,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec


class ScriptedDirector:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []
        self.seeds = []

    async def propose(self, prompt, *, seed=None):
        self.prompts.append(prompt)
        self.seeds.append(seed)
        return DirectorResponse(self.actions.pop(0), {"policy_version": "test"})


class FakeGateway:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
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
    async def test_openai_director_boundary_is_local_qwen_supervisor(self) -> None:
        OpenAIDirectorClient(
            base_url="http://127.0.0.1:8015/v1",
            model="supervisor_theta",
        )
        with self.assertRaises(ValueError):
            OpenAIDirectorClient(
                base_url="https://provider.example/v1",
                model="supervisor_theta",
            )
        with self.assertRaises(ValueError):
            OpenAIDirectorClient(
                base_url="http://127.0.0.1:8015/v1",
                model="gpt-4o-mini",
            )

    async def test_end_to_end_scripted_canvas_and_execution(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            [
                '{"action":"add_agent","agent_id":"solver","model_id":"qwen","contract":"solve"}',
                '{"action":"set_output","agent_id":"solver"}',
                '{"action":"finish"}',
            ]
        )
        gateway = FakeGateway()
        env = AgentWorkflowEnv(
            model_registry,
            gateway=gateway,
            execute_on_edit=True,
            max_agents=10,
        )
        result = await AgentGraphOrchestrator(model_registry, client, seed=1).run(env, "2+2?")
        self.assertEqual(result.final_answer, "answer from solver")
        self.assertEqual(len(result.turns), 3)
        self.assertIn("weighted_preferred_model", client.prompts[0])
        self.assertNotIn("available_skills", client.prompts[0])
        self.assertEqual(result.final_graph["output_agent_id"], "solver")
        self.assertEqual(client.seeds, [1, 2, 3])

        # An incomplete graph is not executed.  Once set_output makes the graph
        # complete, FlowSteer's progressive result is returned in Canvas
        # feedback and therefore appears in the next neutral Director state.
        self.assertIsNone(result.turns[0].canvas_result.execution)
        self.assertIsNotNone(result.turns[1].canvas_result.execution)
        self.assertIn("execution_result=", result.turns[1].canvas_result.feedback)
        self.assertIn("answer from solver", result.turns[1].canvas_result.feedback)
        # Finish reuses the successful result for the unchanged graph revision.
        self.assertEqual(len(gateway.requests), 1)

        initial_state = json.loads(client.prompts[0].split("\n\n", 1)[1])
        complete_state = json.loads(client.prompts[2].split("\n\n", 1)[1])
        self.assertEqual(initial_state["max_rounds"], 20)
        self.assertEqual(initial_state["remaining_rounds"], 20)
        self.assertEqual(initial_state["max_agents"], 10)
        self.assertEqual(initial_state["recent_canvas_history"], [])
        self.assertFalse(initial_state["complete_validation"]["valid"])
        self.assertEqual(complete_state["remaining_rounds"], 18)
        self.assertTrue(complete_state["complete_validation"]["valid"])
        self.assertIn("execution_result=", complete_state["canvas_feedback"])
        self.assertEqual(2, len(complete_state["recent_canvas_history"]))
        self.assertEqual(
            "set_output",
            complete_state["recent_canvas_history"][-1]["action"]["action"],
        )

    async def test_round_limit_is_explicit_failure(self) -> None:
        model_registry = registry()
        client = ScriptedDirector(
            ['{"action":"add_agent","agent_id":"solver","model_id":"qwen","contract":"solve"}']
        )
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway())
        result = await AgentGraphOrchestrator(
            model_registry,
            client,
            max_rounds=1,
        ).run(env, "task")

        self.assertIsNone(result.final_answer)
        self.assertFalse(result.explicit_finish)
        self.assertEqual("max_rounds", result.termination_reason)
        self.assertEqual(1, len(result.turns))


if __name__ == "__main__":
    unittest.main()
