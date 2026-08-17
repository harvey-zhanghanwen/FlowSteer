from __future__ import annotations

import json
import unittest

from src.interactive.agent_runtime import AgentResponse
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    AgentGraphOrchestrator,
    DIRECTOR_SYSTEM_PROMPT,
    DirectorResponse,
    OpenAIDirectorClient,
)
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)


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
            ModelSpec(
                "qwen",
                "provider",
                cheap_weight=10,
                fast_weight=10,
                metadata={
                    "family": "qwen",
                    "profile": "text_qa",
                    "text_qa_canary": "passed",
                    "max_tokens": "512",
                },
            ),
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
        self.assertNotIn("weighted_preferred_model", client.prompts[0])
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
        self.assertIn("output_format", result.turns[1].canvas_result.feedback)
        self.assertNotIn('"final_answer"', result.turns[1].canvas_result.feedback)
        self.assertIn("output_inbox", result.turns[1].canvas_result.feedback)
        # Finish reuses the successful result for the unchanged graph revision.
        self.assertEqual(len(gateway.requests), 1)

        initial_state = json.loads(client.prompts[0].split("\n\n", 1)[1])
        complete_state = json.loads(client.prompts[2].split("\n\n", 1)[1])
        self.assertEqual(initial_state["max_rounds"], 20)
        self.assertEqual(initial_state["remaining_rounds"], 20)
        self.assertEqual(initial_state["max_agents"], 10)
        self.assertEqual(initial_state["recent_canvas_history"], [])
        self.assertFalse(initial_state["graph_validation"]["structurally_complete"])
        self.assertNotIn("complete_validation", initial_state)
        self.assertEqual(0, initial_state["topology_statistics"]["agent_count"])
        self.assertNotIn("construction_progress", initial_state)
        qwen_catalog = next(
            item for item in initial_state["model_catalog"] if item["model_id"] == "qwen"
        )
        self.assertEqual(
            [item["model_id"] for item in initial_state["model_catalog"]],
            [item["model_id"] for item in complete_state["model_catalog"]],
        )
        self.assertEqual(
            {
                "family": "qwen",
                "profile": "text_qa",
                "text_qa_canary": "passed",
            },
            qwen_catalog["routing_metadata"],
        )
        self.assertEqual(complete_state["remaining_rounds"], 18)
        self.assertTrue(complete_state["graph_validation"]["structurally_complete"])
        self.assertNotIn("complete_validation", complete_state)
        self.assertNotIn("construction_progress", complete_state)
        self.assertNotIn("weighted_preferred_model", complete_state)
        self.assertIn("execution_result=", complete_state["canvas_feedback"])
        self.assertEqual(2, len(complete_state["recent_canvas_history"]))
        self.assertEqual(
            "set_output",
            complete_state["recent_canvas_history"][-1]["action"]["action"],
        )
        self.assertNotIn("feedback", complete_state["recent_canvas_history"][-1])
        self.assertEqual(
            "accepted add_agent at revision 1",
            complete_state["recent_canvas_history"][-1][
                "observation_before_action"
            ],
        )
        # The latest progressive result is the current observation exactly
        # once, not duplicated inside the history tail.
        self.assertEqual(1, client.prompts[2].count("execution_result="))

    async def test_director_terminal_policy_is_issue_driven_without_role_template(self) -> None:
        self.assertIn("Only the graph's Output Agent", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("specific missing evidence hop", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("expected input or dependency", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("two message directions", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("independent artifacts that later converge", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("one artifact sent to multiple consumers", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("optional shapes", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("distinct evidence dependencies", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("never JSON or explanation", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("unused rounds", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("graph size alone is neither a benefit nor a cost", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("only a terminal protocol check", DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("complete singleton may be sufficient", DIRECTOR_SYSTEM_PROMPT)
        self.assertNotIn("prefer finish", DIRECTOR_SYSTEM_PROMPT.lower())
        self.assertNotIn("not to make the graph larger", DIRECTOR_SYSTEM_PROMPT)
        self.assertNotIn("Researcher", DIRECTOR_SYSTEM_PROMPT)
        self.assertNotIn("Critic", DIRECTOR_SYSTEM_PROMPT)
        self.assertNotIn("must use three", DIRECTOR_SYSTEM_PROMPT.lower())

    async def test_catalog_order_is_decoupled_from_rollout_sampling_seed(self) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway(), problem="same task")
        first = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            seed=101,
            catalog_order_seed="condition:same-task",
        )
        second = AgentGraphOrchestrator(
            model_registry,
            ScriptedDirector([]),
            seed=202,
            catalog_order_seed="condition:same-task",
        )

        first_state = json.loads(first.build_prompt(env, 0, ()).split("\n\n", 1)[1])
        second_state = json.loads(second.build_prompt(env, 0, ()).split("\n\n", 1)[1])
        self.assertEqual(first_state["model_catalog"], second_state["model_catalog"])
        self.assertNotEqual(first.seed, second.seed)

    async def test_scientific_rollout_ordinal_changes_sampling_not_catalog(self) -> None:
        model_registry = registry()
        env = AgentWorkflowEnv(model_registry, gateway=FakeGateway(), problem="same task")

        def orchestrator(rollout_ordinal: int) -> AgentGraphOrchestrator:
            coordinate = ScientificSamplingCoordinate(
                sampling_schedule_hash=scientific_sampling_schedule_hash(
                    base_seed=17
                ),
                schedule_purpose="architecture-dev",
                ordered_sequence_hash=stable_hash(["hotpotqa:one"]),
                sequence_position=rollout_ordinal,
                task_id="hotpotqa:one",
                optimizer_step_or_anchor_ordinal=0,
            )
            return AgentGraphOrchestrator(
                model_registry,
                ScriptedDirector([]),
                seed=17,
                catalog_order_seed="architecture-dev:hotpotqa:one",
                sampling_base_seed=17,
                sampling_coordinate=coordinate,
            )

        first = orchestrator(0)
        second = orchestrator(1)
        first_state = json.loads(first.build_prompt(env, 0, ()).split("\n\n", 1)[1])
        second_state = json.loads(second.build_prompt(env, 0, ()).split("\n\n", 1)[1])

        self.assertEqual(first_state["model_catalog"], second_state["model_catalog"])
        self.assertNotEqual(first.generation_seed(0), second.generation_seed(0))
        self.assertEqual(
            "skillev-scientific-sampling@1",
            first.sampling_receipt["algorithm"],
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
