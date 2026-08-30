from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from test_environment_execution import (
    FakeSession,
    SequenceGateway,
    resources,
)


class _WebShopSession(FakeSession):
    environment_id = "fake:webshop:multi-agent-subgraph"
    task_family = "webshop"

    def __init__(self) -> None:
        super().__init__()
        self._available = {
            "has_search_bar": True,
            "clickables": [],
        }

    def reset(self) -> str:
        self.reset_count += 1
        return "Search page"

    def step(self, action: str):  # type: ignore[no-untyped-def]
        self.actions.append(action)
        if action.startswith("search["):
            self._available = {
                "has_search_bar": True,
                "clickables": ["B000ITEM01", "Back to Search"],
            }
            return (
                "Results page [SEP] B000ITEM01 [SEP] Blue steel table",
                0.25,
                False,
                {"graded_score": 0.25, "evaluator_private": "hidden-search"},
            )
        self._available = {
            "has_search_bar": False,
            "clickables": ["Buy Now", "Back to Search"],
        }
        return (
            "Product page [SEP] Blue steel table [SEP] Buy Now",
            0.75,
            False,
            {"graded_score": 0.75, "evaluator_private": "hidden-click"},
        )


class WebShopMultiAgentSubgraphTests(unittest.IsolatedAsyncioTestCase):
    def test_no_progress_opens_subgraph_augmentation_not_another_scalar_owner(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway([])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=4,
            stepwise_director=True,
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        graph = AgentGraph(
            [
                AgentNode(
                    "environment_owner",
                    "m",
                    "Take one action from the current public state.",
                    execution_mode="react",
                    allowed_tools=(environment.tool_id,),
                )
            ],
            output_agent_id="environment_owner",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Buy a blue steel table.",
            required_tool_id=environment.tool_id,
            recovery_policy="preserve_diagnose_repair_augment",
            allowed_actions=(
                "add_agent",
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
            max_agents=3,
            max_agents_per_subgraph=2,
        )
        no_progress = {
            "environment_episode_id": "episode-1",
            "environment_id": session.environment_id,
            "task_family": "webshop",
            "environment_revision": 2,
            "last_action": "search[blue steel table]",
            "state_advanced": False,
            "observation_status": "success",
            "current_observation": "Results page",
            "admissible_action_count": 2,
            "public_progress": {
                "latest_transition": {
                    "action": "search[blue steel table]",
                    "state_advanced": False,
                },
                "no_progress": {
                    "detected": True,
                    "reasons": ["repeated_state_action"],
                    "repeated_state_action_count": 2,
                    "action_cycle": False,
                },
            },
            "turns_used": 2,
            "remaining_action_budget": 2,
            "environment_terminal": False,
            "environment_truncated": False,
        }
        canvas._progressive_output_metadata["environment_owner"] = {
            "environment_current_state": no_progress
        }

        self.assertEqual(
            ("modify_agent",),
            canvas.model_admissible_action_types(),
        )

        no_progress["public_progress"]["no_progress"][
            "repeated_state_action_count"
        ] = 3
        admitted = set(canvas.model_admissible_action_types())

        self.assertEqual({"modify_agent", "add_subgraph"}, admitted)
        for forbidden in (
            "add_agent",
            "continue",
            "delete_agent",
            "finish",
        ):
            self.assertNotIn(forbidden, admitted)

    async def test_subgraph_routes_artifact_to_single_stateful_owner_and_steps_once(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway(
            [
                "search[blue steel table]",
                (
                    "The visible result B000ITEM01 matches the requested item; "
                    "inspect that product next."
                ),
                "click[B000ITEM01]",
            ]
        )
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=4,
            stepwise_director=True,
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Buy a blue steel table.",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=(
                "add_agent",
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
            max_agents=3,
            max_agents_per_subgraph=2,
        )

        owner_added = await canvas.step(
            json.dumps(
                {
                    "action": "add_agent",
                    "agent_id": "environment_owner",
                    "model_id": "m",
                    "contract": (
                        "Use the current public WebShop state and take one "
                        "admissible environment action."
                    ),
                    "execution_mode": "react",
                    "allowed_tools": [environment.tool_id],
                    "artifact_type": "environment_observation",
                }
            )
        )

        self.assertTrue(owner_added.accepted, owner_added.feedback)
        self.assertEqual(["search[blue steel table]"], session.actions)
        self.assertEqual(
            ["environment_owner"],
            [
                node.id
                for node in canvas.graph.nodes
                if environment.tool_id in node.allowed_tools
            ],
        )

        action_count_before_subgraph = len(session.actions)
        subgraph_added = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "advisor",
                            "model_id": "m",
                            "contract": (
                                "Inspect the visible task state and provide one "
                                "grounded next-step artifact."
                            ),
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                            "artifact_type": "text",
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "advisor",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "environment_owner",
                }
            )
        )

        self.assertTrue(subgraph_added.accepted, subgraph_added.feedback)
        self.assertIsNotNone(subgraph_added.execution)
        self.assertEqual(
            action_count_before_subgraph + 1,
            len(session.actions),
            "one accepted subgraph edit must advance exactly one Action-Observation",
        )
        self.assertEqual(
            ["search[blue steel table]", "click[B000ITEM01]"],
            session.actions,
        )

        stateful_owners = [
            node.id
            for node in canvas.graph.nodes
            if environment.tool_id in node.allowed_tools
        ]
        self.assertEqual(["environment_owner"], stateful_owners)
        relation = canvas.graph.relation_bits("advisor", "environment_owner")
        self.assertTrue(relation.source_to_target)
        self.assertFalse(relation.target_to_source)

        owner_requests: list[AgentRequest] = [
            request
            for request in gateway.requests
            if request.agent.id == "environment_owner"
        ]
        self.assertEqual(2, len(owner_requests))
        routed = owner_requests[-1].upstream
        self.assertEqual(["advisor"], [item.source_agent_id for item in routed])
        self.assertIn("B000ITEM01", routed[0].content)
        self.assertIn("inspect that product next", routed[0].content)

        public_state = canvas.public_environment_state()
        self.assertIsNotNone(public_state)
        assert public_state is not None
        self.assertEqual("click[B000ITEM01]", public_state["last_action"])
        self.assertEqual(2, public_state["turns_used"])
        self.assertIn("Product page", public_state["current_observation"])

        public_feedback = subgraph_added.feedback
        self.assertIn('"source_agent_id":"advisor"', public_feedback)
        self.assertIn("B000ITEM01", public_feedback)
        for private_field in (
            "reward",
            "graded_score",
            "evaluator_private",
            "hidden-search",
            "hidden-click",
        ):
            self.assertNotIn(private_field, public_feedback)
            self.assertNotIn(private_field, str(public_state))

    async def test_candidate_subgraph_cannot_add_owner_or_reciprocal_owner_block(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway(["search[blue steel table]"])
        environment = resources(
            session=session,
            gateway=gateway,
            max_turns=4,
            stepwise_director=True,
        )
        registry = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [ModelSpec("m", "fake")],
        )
        runtime = AgentRuntime(
            registry,
            gateway,
            execution_adapters={"react": environment.execution_adapter},
            tool_registry=environment.tool_registry,
            dataset_id="webshop",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Buy a blue steel table.",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=(
                "add_agent",
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ),
            max_agents=3,
            max_agents_per_subgraph=2,
        )
        owner_added = await canvas.step(
            json.dumps(
                {
                    "action": "add_agent",
                    "agent_id": "environment_owner",
                    "model_id": "m",
                    "contract": "Take one admissible WebShop action.",
                    "execution_mode": "react",
                    "allowed_tools": [environment.tool_id],
                }
            )
        )
        self.assertTrue(owner_added.accepted, owner_added.feedback)
        baseline_revision = canvas.graph.revision
        baseline_actions = list(session.actions)

        duplicate_owner = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "second_owner",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        }
                    ],
                    "relations": [],
                }
            )
        )
        self.assertFalse(duplicate_owner.accepted)
        self.assertIn("requires exactly one Agent owner", duplicate_owner.feedback)
        self.assertEqual(baseline_revision, canvas.graph.revision)
        self.assertFalse(canvas.graph.has_node("second_owner"))
        self.assertEqual(baseline_actions, session.actions)

        reciprocal_owner = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "advisor",
                            "model_id": "m",
                            "contract": "Publish one public-state artifact.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "advisor",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": True,
                        }
                    ],
                }
            )
        )
        self.assertFalse(reciprocal_owner.accepted)
        self.assertIn(
            "cannot execute inside a reciprocal Agent block",
            reciprocal_owner.feedback,
        )
        self.assertEqual(baseline_revision, canvas.graph.revision)
        self.assertFalse(canvas.graph.has_node("advisor"))
        self.assertEqual(baseline_actions, session.actions)


if __name__ == "__main__":
    unittest.main()
