from __future__ import annotations

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentRequest, AgentResponse, AgentRuntime
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowStateError,
)
from src.interactive.director import (
    director_live_action_parameter_json_schema_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_live_stateful_add_subgraph_relation_candidates,
)
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
    def test_stateful_execution_projects_incoming_artifact_and_relation_effects(
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
                    "Take one public environment action.",
                    execution_mode="react",
                    allowed_tools=(environment.tool_id,),
                ),
                AgentNode(
                    "auxiliary",
                    "m",
                    "Produce an evidence artifact from public state.",
                ),
            ],
            output_agent_id="environment_owner",
        )
        graph.set_relation("auxiliary", "environment_owner", True, False)
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Buy a blue steel table.",
            required_tool_id=environment.tool_id,
            allowed_actions=("set_relation", "continue", "finish"),
            max_agents=3,
            max_agents_per_subgraph=2,
        )
        canvas._progressive_output_metadata["environment_owner"] = {
            "input_artifact_provenance": [
                {
                    "source_agent_id": "auxiliary",
                    "target_agent_id": "environment_owner",
                    "artifact_id": "artifact-1",
                    "artifact_type": "text",
                }
            ],
            "environment_receipts": [
                {
                    "turn": 1,
                    "action": "search[blue steel table]",
                    "observation_status": "success",
                    "state_advanced": True,
                    "terminal": False,
                    "reward": 0.75,
                    "info": {"hidden": True},
                }
            ],
            "environment_current_state": {
                "task_family": "webshop",
                "environment_revision": 1,
                "remaining_action_budget": 3,
                "environment_terminal": False,
                "environment_truncated": False,
                "observation_status": "success",
            },
        }

        projected = canvas.stateful_execution_state()
        assert projected is not None
        self.assertEqual(
            ["auxiliary"], projected["declared_incoming_source_ids"]
        )
        self.assertEqual(
            ["auxiliary"], projected["consumed_input_source_ids"]
        )
        self.assertEqual(
            ["auxiliary"],
            projected["environment_actor_directed_ancestor_ids"],
        )
        self.assertEqual(1, projected["environment_actor_in_degree"])
        self.assertTrue(
            projected["environment_actor_must_be_execution_sink"]
        )
        self.assertEqual(
            [],
            projected[
                "tool_free_agent_ids_without_directed_path_to_environment_actor"
            ],
        )
        self.assertEqual(
            ["artifact-1"], projected["consumed_input_artifact_ids"]
        )
        self.assertNotIn(
            "reward", projected["recent_action_observations"][0]
        )
        self.assertNotIn("info", projected["recent_action_observations"][0])

        relation_domain = canvas.model_admissible_action_targets()[
            "set_relation"
        ]
        self.assertEqual(
            "environment_owner",
            relation_domain["stateful_environment_actor_id"],
        )
        self.assertEqual(
            "incoming_directed_artifacts_only",
            relation_domain["stateful_action_input_semantics"],
        )
        self.assertNotIn("relation_execution_effects", relation_domain)

    async def test_stateful_owner_outgoing_relation_is_not_in_live_domain(
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
                    "Take one public environment action.",
                    execution_mode="react",
                    allowed_tools=(environment.tool_id,),
                ),
                AgentNode(
                    "advisor",
                    "m",
                    "Produce an artifact from the current public state.",
                ),
            ],
            output_agent_id="environment_owner",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=graph,
            problem="Buy a blue steel table.",
            execute_on_edit=False,
            required_tool_id=environment.tool_id,
            allowed_actions=("set_relation", "set_output", "finish"),
            max_agents=3,
            max_agents_per_subgraph=2,
        )

        candidates = canvas.model_admissible_action_targets()[
            "set_relation"
        ]["candidates"]
        self.assertIn(
            {
                "source_id": "advisor",
                "target_id": "environment_owner",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )
        self.assertNotIn(
            {
                "source_id": "environment_owner",
                "target_id": "advisor",
                "source_to_target": True,
                "target_to_source": False,
            },
            candidates,
        )

        rejected = await canvas.step(
            json.dumps(
                {
                    "action": "set_relation",
                    "source_id": "environment_owner",
                    "target_id": "advisor",
                    "source_to_target": True,
                    "target_to_source": False,
                }
            )
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("must be an execution sink", rejected.feedback)
        self.assertEqual(0, canvas.graph.revision)

        rejected_output = await canvas.step(
            json.dumps(
                {
                    "action": "set_output",
                    "agent_id": "advisor",
                }
            )
        )
        self.assertFalse(rejected_output.accepted)
        self.assertIn("must be the Output Agent", rejected_output.feedback)
        self.assertEqual("environment_owner", canvas.graph.output_agent_id)

    def test_stateful_auxiliary_profiles_are_strictly_tool_free(self) -> None:
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
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Buy a blue steel table.",
            required_tool_id=environment.tool_id,
            allowed_actions=("add_agent", "add_subgraph", "finish"),
            max_agents=3,
            max_agents_per_subgraph=3,
        )
        registered = (
            ("reasoning", ()),
            ("react", ()),
            ("react", (environment.tool_id,)),
            ("react", ("webshop.catalog.read",)),
        )

        with patch.object(
            runtime,
            "registered_execution_profiles",
            return_value=registered,
        ):
            domain = canvas.model_admissible_action_targets()["add_subgraph"]

        self.assertEqual(
            [
                {"execution_mode": "reasoning", "allowed_tools": []},
            ],
            sorted(
                domain["stateful_tool_owner"][
                    "auxiliary_execution_profiles"
                ],
                key=lambda item: item["execution_mode"],
                reverse=True,
            ),
        )
        self.assertNotIn(
            ["webshop.catalog.read"],
            [
                item["allowed_tools"]
                for item in domain["registered_execution_profiles"]
            ],
        )

    async def test_tool_free_react_auxiliary_is_rejected_before_execution(
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
                    "Take one WebShop action.",
                    execution_mode="react",
                    allowed_tools=(environment.tool_id,),
                ),
                AgentNode(
                    "auxiliary",
                    "m",
                    "Analyze the public observation.",
                    execution_mode="react",
                ),
            ],
            output_agent_id="environment_owner",
        )

        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "tool-free reasoning profile",
        ):
            AgentWorkflowEnv(
                registry,
                runtime=runtime,
                graph=graph,
                problem="Buy a blue steel table.",
                required_tool_id=environment.tool_id,
            )

    def test_constructor_and_restore_reject_invalid_stateful_profile(
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
        invalid_graph = AgentGraph(
            [
                AgentNode(
                    "invalid_owner",
                    "m",
                    "Take one WebShop action.",
                    execution_mode="reasoning",
                    allowed_tools=(environment.tool_id,),
                )
            ],
            output_agent_id="invalid_owner",
        )
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "exactly one Agent with execution_mode='react'",
        ):
            AgentWorkflowEnv(
                registry,
                runtime=runtime,
                graph=invalid_graph,
                problem="Buy a blue steel table.",
                required_tool_id=environment.tool_id,
            )

        valid_graph = AgentGraph(
            [
                AgentNode(
                    "environment_owner",
                    "m",
                    "Take one WebShop action.",
                    execution_mode="react",
                    allowed_tools=(environment.tool_id,),
                )
            ],
            output_agent_id="environment_owner",
        )
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            graph=valid_graph,
            problem="Buy a blue steel table.",
            required_tool_id=environment.tool_id,
        )
        invalid_snapshot = replace(
            canvas.snapshot(),
            graph=invalid_graph.snapshot(),
        )
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "exactly one Agent with execution_mode='react'",
        ):
            canvas.restore(invalid_snapshot)

    async def test_raw_owner_profile_mutation_is_rejected_before_execution(
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
                    "Take one WebShop action.",
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
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=("modify_agent", "finish"),
        )
        baseline_revision = canvas.graph.revision

        result = await canvas.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "environment_owner",
                    "execution_mode": "reasoning",
                }
            )
        )

        self.assertFalse(result.accepted)
        self.assertIn("exactly one Agent with execution_mode='react'", result.feedback)
        self.assertEqual(baseline_revision, canvas.graph.revision)
        self.assertEqual([], session.actions)

    async def test_out_of_band_invalid_graph_is_rejected_before_step(
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
        graph = AgentGraph(
            [
                AgentNode(
                    "environment_owner",
                    "m",
                    "Take one WebShop action.",
                    execution_mode="react",
                    allowed_tools=(environment.tool_id,),
                )
            ],
            output_agent_id="environment_owner",
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
            graph=graph,
            problem="Buy a blue steel table.",
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=("finish",),
        )
        canvas.graph.add_agent(
            AgentNode(
                "second_owner",
                "m",
                "Take another WebShop action.",
                execution_mode="react",
                allowed_tools=(environment.tool_id,),
            )
        )

        result = await canvas.step(json.dumps({"action": "finish"}))

        self.assertFalse(result.accepted)
        self.assertIn("stateful AgentGraph invariant violated", result.feedback)
        self.assertEqual([], session.actions)

    def test_empty_canvas_exposes_exact_atomic_stateful_component_domain(
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
        canvas = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Buy a blue steel table.",
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

        self.assertEqual(
            ("add_agent", "add_subgraph"),
            canvas.model_admissible_action_types(),
        )
        domains = canvas.model_admissible_action_targets()
        subgraph = domains["add_subgraph"]
        self.assertEqual(2, subgraph["min_new_agents"])
        self.assertEqual(2, subgraph["max_new_agents"])
        self.assertEqual(1, subgraph["stateful_tool_owner"]["required_count"])
        self.assertEqual(
            environment.tool_id,
            subgraph["stateful_tool_owner"]["tool_id"],
        )
        self.assertEqual(
            {"execution_mode", "allowed_tools"},
            {"execution_mode", "allowed_tools"}.intersection(
                subgraph["required_agent_fields"]
            ),
        )

        declaration_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains
            )
        )
        # Two mutually exclusive branches correspond to the two possible
        # owner positions; Agent contracts and models remain sampled.
        self.assertEqual(
            2,
            len(declaration_schema["properties"]["agents"]["oneOf"]),
        )
        valid_agents = [
            {
                "agent_id": "node_1",
                "model_id": "m",
                "contract": "Analyze the visible constraints.",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "agent_id": "node_2",
                "model_id": "m",
                "contract": "Take one admissible WebShop action.",
                "execution_mode": "react",
                "allowed_tools": [environment.tool_id],
            },
        ]
        relation_candidates = (
            director_live_stateful_add_subgraph_relation_candidates(
                domains,
                valid_agents,
            )
        )
        self.assertEqual(
            (
                {
                    "source_id": "node_1",
                    "target_id": "node_2",
                    "source_to_target": True,
                    "target_to_source": False,
                },
            ),
            relation_candidates,
        )
        parameter_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=valid_agents,
            )
        )
        self.assertEqual(
            1,
            parameter_schema["properties"]["relations"]["maxItems"],
        )
        self.assertEqual(
            {"anyOf": [{"const": "node_2"}, {"type": "null"}]},
            parameter_schema["properties"]["output_agent_id"],
        )
        with self.assertRaisesRegex(ValueError, "exactly one stateful Tool owner"):
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=[
                    {
                        **valid_agents[0],
                        "agent_id": "node_1",
                    },
                    {
                        **valid_agents[0],
                        "agent_id": "node_2",
                    },
                ],
            )

    def test_existing_owner_two_new_advisors_expose_all_relation_pairs(
        self,
    ) -> None:
        domains = {
            "add_subgraph": {
                "min_new_agents": 1,
                "max_new_agents": 2,
                "existing_agent_ids": ["environment_owner"],
                "model_ids": ["m"],
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "execution_mode",
                    "allowed_tools",
                ],
                "optional_agent_fields": [
                    "role_family",
                    "artifact_type",
                    "completion_condition",
                ],
                "registered_execution_profiles": [
                    {
                        "execution_mode": "reasoning",
                        "allowed_tools": [],
                    }
                ],
                "endpoint_scope": {
                    "relation_endpoint_sources": [
                        "existing_agent_ids",
                        "same_action_agent_ids",
                    ],
                    "output_agent_id_sources": [
                        "existing_agent_ids",
                        "same_action_agent_ids",
                    ],
                },
                "stateful_tool_owner_agent_id": "environment_owner",
                "semantic_protocol": "none",
            }
        }
        advisors = [
            {
                "agent_id": "node_1",
                "model_id": "m",
                "contract": "Inspect the current public state for constraints.",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "agent_id": "node_2",
                "model_id": "m",
                "contract": "Check the proposed environment action.",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
        ]

        relation_candidates = (
            director_live_stateful_add_subgraph_relation_candidates(
                domains,
                advisors,
            )
        )
        endpoint_pairs = {
            frozenset((candidate["source_id"], candidate["target_id"]))
            for candidate in relation_candidates
        }
        self.assertEqual(
            {
                frozenset(("environment_owner", "node_1")),
                frozenset(("environment_owner", "node_2")),
                frozenset(("node_1", "node_2")),
            },
            endpoint_pairs,
        )
        self.assertEqual(
            {
                ("node_1", "environment_owner", True, False),
                ("node_2", "environment_owner", True, False),
            },
            {
                (
                    candidate["source_id"],
                    candidate["target_id"],
                    candidate["source_to_target"],
                    candidate["target_to_source"],
                )
                for candidate in relation_candidates
                if "environment_owner"
                in {candidate["source_id"], candidate["target_id"]}
            },
        )

        parameter_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=advisors,
            )
        )
        self.assertEqual(
            3,
            parameter_schema["properties"]["relations"]["maxItems"],
        )
        self.assertEqual(
            {
                "anyOf": [
                    {"const": "environment_owner"},
                    {"type": "null"},
                ]
            },
            parameter_schema["properties"]["output_agent_id"],
        )

    async def test_empty_canvas_atomic_subgraph_routes_once_to_stateful_owner(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway(
            [
                "The visible constraints require a blue steel table.",
                "search[blue steel table]",
                "Preserve the currently observed product constraints.",
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

        result = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "node_1",
                            "model_id": "m",
                            "contract": "Analyze the visible constraints.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "node_2",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "node_1",
                            "target_id": "node_2",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "node_2",
                }
            )
        )

        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual(["search[blue steel table]"], session.actions)
        self.assertEqual(
            ["node_1", "node_2"],
            list(result.execution.executed_agent_ids),
        )
        owner_request = [
            request for request in gateway.requests if request.agent.id == "node_2"
        ][0]
        self.assertEqual(
            ["node_1"],
            [message.source_agent_id for message in owner_request.upstream],
        )
        self.assertIn("blue steel table", owner_request.upstream[0].content)

        independent = await canvas.step(
            json.dumps(
                {
                    "action": "add_agent",
                    "agent_id": "node_3",
                    "model_id": "m",
                    "contract": "Record the public constraints independently.",
                    "execution_mode": "reasoning",
                    "allowed_tools": [],
                }
            )
        )
        self.assertTrue(independent.accepted, independent.feedback)
        feedback_payload = json.loads(
            independent.feedback.split("execution_result=", 1)[1]
        )
        self.assertEqual("node_2", feedback_payload["environment_actor_id"])
        self.assertEqual(
            ["node_1"],
            [
                item["source_agent_id"]
                for item in feedback_payload["environment_actor_inbox"]
            ],
        )
        self.assertIn(
            "blue steel table",
            feedback_payload["environment_actor_inbox"][0]["raw_output"],
        )

    async def test_continue_refreshes_predecessor_from_latest_public_state(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway(
            [
                "Use the original goal to search for a blue steel table.",
                "search[blue steel table]",
                (
                    "The latest result exposes B000ITEM01; inspect that exact "
                    "product next."
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
        added = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "advisor",
                            "model_id": "m",
                            "contract": (
                                "Use the public task and environment state to "
                                "produce a grounded next-step artifact."
                            ),
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "environment_owner",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
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
        self.assertTrue(added.accepted, added.feedback)
        self.assertEqual(["search[blue steel table]"], session.actions)

        continued = await canvas.step(json.dumps({"action": "continue"}))

        self.assertTrue(continued.accepted, continued.feedback)
        self.assertEqual(
            ["advisor", "environment_owner"],
            list(continued.execution.executed_agent_ids),
        )
        self.assertEqual(
            ["search[blue steel table]", "click[B000ITEM01]"],
            session.actions,
        )
        advisor_requests = [
            request
            for request in gateway.requests
            if request.agent.id == "advisor"
        ]
        self.assertEqual(2, len(advisor_requests))
        refreshed_problem = advisor_requests[-1].problem
        self.assertIn("Buy a blue steel table.", refreshed_problem)
        self.assertIn("B000ITEM01", refreshed_problem)
        self.assertIn("model_visible_admissible_actions", refreshed_problem)
        owner_requests = [
            request
            for request in gateway.requests
            if request.agent.id == "environment_owner"
        ]
        self.assertEqual(2, len(owner_requests))
        self.assertIn("B000ITEM01", owner_requests[-1].upstream[0].content)
        self.assertIn(
            '"model_visible_admissible_actions"',
            continued.feedback,
        )

    async def test_reasoning_auxiliary_does_not_receive_owner_action_interface(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway(
            [
                "The goal requires a blue steel table.",
                "search[blue steel table]",
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
            problem=(
                "Buy a blue steel table.\n\n"
                "Execution interface: a react node may use the WebShop "
                "environment. Return exactly one currently admissible native "
                "WebShop action. Do not return a prose answer."
            ),
            execute_on_edit=True,
            required_tool_id=environment.tool_id,
            allowed_actions=("add_subgraph", "finish"),
            max_agents=2,
            max_agents_per_subgraph=2,
        )

        result = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "auxiliary",
                            "model_id": "m",
                            "contract": "Analyze the public state.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "environment_owner",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "auxiliary",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "environment_owner",
                }
            )
        )

        self.assertTrue(result.accepted, result.feedback)
        self.assertEqual(
            "Buy a blue steel table.",
            canvas.original_task_instruction,
        )
        public_state = canvas.public_environment_state()
        self.assertIsNotNone(public_state)
        self.assertEqual(
            "Buy a blue steel table.",
            public_state["task_instruction"],
        )
        self.assertNotIn(
            "Return exactly one currently admissible native WebShop action",
            json.dumps(public_state),
        )
        auxiliary_request = next(
            request
            for request in gateway.requests
            if request.agent.id == "auxiliary"
        )
        self.assertIn("Buy a blue steel table.", auxiliary_request.problem)
        self.assertNotIn(
            "Return exactly one currently admissible native WebShop action",
            auxiliary_request.problem,
        )
        self.assertNotIn("Do not return a prose answer", auxiliary_request.problem)
        owner_request = next(
            request
            for request in gateway.requests
            if request.agent.id == "environment_owner"
        )
        self.assertIn("Return exactly one native WebShop action", owner_request.problem)

    async def test_predecessor_failure_preserves_latest_public_environment_state(
        self,
    ) -> None:
        class FailingGateway(SequenceGateway):
            async def generate(self, request: AgentRequest) -> AgentResponse:
                if len(self.requests) == 2:
                    self.requests.append(request)
                    raise RuntimeError("reasoning predecessor failed")
                return await super().generate(request)

        session = _WebShopSession()
        gateway = FailingGateway(
            [
                "Search for the requested blue steel table.",
                "search[blue steel table]",
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
            allowed_actions=("add_subgraph", "continue", "finish"),
            max_agents=2,
            max_agents_per_subgraph=2,
        )
        added = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "auxiliary",
                            "model_id": "m",
                            "contract": "Analyze the latest public state.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "environment_owner",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "auxiliary",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "environment_owner",
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)
        before = canvas.public_environment_state()
        self.assertIsNotNone(before)

        continued = await canvas.step(json.dumps({"action": "continue"}))

        self.assertTrue(continued.accepted, continued.feedback)
        self.assertIsNotNone(continued.partial_execution)
        self.assertEqual(["search[blue steel table]"], session.actions)
        after = canvas.public_environment_state()
        self.assertIsNotNone(after)
        for key in (
            "environment_episode_id",
            "environment_revision",
            "current_observation",
            "remaining_action_budget",
        ):
            self.assertEqual(before[key], after[key])
        self.assertNotIn("environment_owner", canvas._progressive_outputs)

    async def test_edit_failure_preserves_state_and_repair_reruns_owner(
        self,
    ) -> None:
        class FailOnceGateway(SequenceGateway):
            def __init__(self) -> None:
                super().__init__(
                    [
                        "Search for the requested blue steel table.",
                        "search[blue steel table]",
                        "Inspect B000ITEM01 from the latest result.",
                        "click[B000ITEM01]",
                    ]
                )
                self.failed = False

            async def generate(self, request: AgentRequest) -> AgentResponse:
                if len(self.requests) == 2 and not self.failed:
                    self.requests.append(request)
                    self.failed = True
                    raise RuntimeError("revised predecessor failed")
                return await super().generate(request)

        session = _WebShopSession()
        gateway = FailOnceGateway()
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
            allowed_actions=("add_subgraph", "modify_agent", "finish"),
            max_agents=2,
            max_agents_per_subgraph=2,
        )
        added = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "auxiliary",
                            "model_id": "m",
                            "contract": "Analyze the public state.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "environment_owner",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "auxiliary",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "environment_owner",
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)
        before = canvas.public_environment_state()
        self.assertIsNotNone(before)

        failed_edit = await canvas.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "auxiliary",
                    "contract": "Re-evaluate the latest public result.",
                }
            )
        )

        self.assertTrue(failed_edit.accepted, failed_edit.feedback)
        self.assertIsNotNone(failed_edit.partial_execution)
        self.assertEqual(["search[blue steel table]"], session.actions)
        after_failure = canvas.public_environment_state()
        self.assertIsNotNone(after_failure)
        self.assertEqual(
            before["environment_revision"],
            after_failure["environment_revision"],
        )
        self.assertNotIn("environment_owner", canvas._progressive_outputs)
        failed_request = gateway.requests[-1]
        self.assertEqual("auxiliary", failed_request.agent.id)
        self.assertIn("B000ITEM01", failed_request.problem)

        repaired = await canvas.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "auxiliary",
                    "contract": "Use the preserved result to select the next item.",
                }
            )
        )

        self.assertTrue(repaired.accepted, repaired.feedback)
        self.assertEqual(
            ["search[blue steel table]", "click[B000ITEM01]"],
            session.actions,
        )
        self.assertIn("environment_owner", canvas._progressive_outputs)
        after_repair = canvas.public_environment_state()
        self.assertIsNotNone(after_repair)
        self.assertEqual(2, after_repair["environment_revision"])

    async def test_continue_refreshes_transitive_tool_free_predecessors(
        self,
    ) -> None:
        session = _WebShopSession()
        gateway = SequenceGateway(
            [
                "Extract the requested blue steel table constraints.",
                "Use those constraints for a product search.",
                "search[blue steel table]",
                "The new public result contains B000ITEM01.",
                "Inspect B000ITEM01 next.",
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
            allowed_actions=("add_subgraph", "continue", "finish"),
            max_agents=3,
            max_agents_per_subgraph=3,
        )
        added = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "node_1",
                            "model_id": "m",
                            "contract": "Interpret the public task constraints.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "node_2",
                            "model_id": "m",
                            "contract": "Turn upstream evidence into a next-step artifact.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "node_3",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "node_1",
                            "target_id": "node_2",
                            "source_to_target": True,
                            "target_to_source": False,
                        },
                        {
                            "source_id": "node_2",
                            "target_id": "node_3",
                            "source_to_target": True,
                            "target_to_source": False,
                        },
                    ],
                    "output_agent_id": "node_3",
                }
            )
        )
        self.assertTrue(added.accepted, added.feedback)

        continued = await canvas.step('{"action":"continue"}')

        self.assertTrue(continued.accepted, continued.feedback)
        self.assertEqual(
            ["node_1", "node_2", "node_3"],
            list(continued.execution.executed_agent_ids),
        )
        self.assertEqual(
            ["search[blue steel table]", "click[B000ITEM01]"],
            session.actions,
        )
        second_node_1_request = [
            request for request in gateway.requests if request.agent.id == "node_1"
        ][-1]
        self.assertIn("B000ITEM01", second_node_1_request.problem)
        second_owner_request = [
            request for request in gateway.requests if request.agent.id == "node_3"
        ][-1]
        self.assertEqual(
            ["node_2"],
            [message.source_agent_id for message in second_owner_request.upstream],
        )
        self.assertIn("B000ITEM01", second_owner_request.upstream[0].content)

    async def test_fan_in_edit_refreshes_existing_owner_ancestors(self) -> None:
        class AgentAwareGateway(SequenceGateway):
            def __init__(self) -> None:
                super().__init__([])
                self.calls_by_agent: dict[str, int] = {}

            async def generate(self, request: AgentRequest) -> AgentResponse:
                self.requests.append(request)
                call = self.calls_by_agent.get(request.agent.id, 0)
                self.calls_by_agent[request.agent.id] = call + 1
                if request.agent.id == "environment_owner":
                    output = (
                        "search[blue steel table]"
                        if call == 0
                        else "click[B000ITEM01]"
                    )
                elif request.agent.id == "existing_advisor":
                    output = (
                        "Search for a blue steel table."
                        if call == 0
                        else "The latest result exposes B000ITEM01; inspect it."
                    )
                else:
                    output = "Independently verify B000ITEM01 against the goal."
                return AgentResponse(
                    output,
                    {"provider_request_id": f"provider-{len(self.requests)}"},
                )

        session = _WebShopSession()
        gateway = AgentAwareGateway()
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
            allowed_actions=("add_subgraph", "finish"),
            max_agents=3,
            max_agents_per_subgraph=2,
        )
        initial = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "existing_advisor",
                            "model_id": "m",
                            "contract": "Propose an action from public state.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                        {
                            "agent_id": "environment_owner",
                            "model_id": "m",
                            "contract": "Take one admissible WebShop action.",
                            "execution_mode": "react",
                            "allowed_tools": [environment.tool_id],
                        },
                    ],
                    "relations": [
                        {
                            "source_id": "existing_advisor",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                    "output_agent_id": "environment_owner",
                }
            )
        )
        self.assertTrue(initial.accepted, initial.feedback)
        self.assertEqual(["search[blue steel table]"], session.actions)

        augmented = await canvas.step(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": "new_advisor",
                            "model_id": "m",
                            "contract": "Check the next action independently.",
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        }
                    ],
                    "relations": [
                        {
                            "source_id": "new_advisor",
                            "target_id": "environment_owner",
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    ],
                }
            )
        )

        self.assertTrue(augmented.accepted, augmented.feedback)
        assert augmented.execution is not None
        self.assertEqual(
            {"existing_advisor", "environment_owner", "new_advisor"},
            set(augmented.execution.executed_agent_ids),
        )
        self.assertNotIn(
            "existing_advisor", augmented.execution.reused_agent_ids
        )
        self.assertEqual(2, gateway.calls_by_agent["existing_advisor"])
        self.assertEqual(
            ["search[blue steel table]", "click[B000ITEM01]"],
            session.actions,
        )
        owner_requests = [
            request
            for request in gateway.requests
            if request.agent.id == "environment_owner"
        ]
        self.assertEqual(2, len(owner_requests))
        routed = {
            message.source_agent_id: message.content
            for message in owner_requests[-1].upstream
        }
        self.assertIn("B000ITEM01", routed["existing_advisor"])
        self.assertIn("B000ITEM01", routed["new_advisor"])

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
            {"modify_agent", "add_subgraph"},
            set(canvas.model_admissible_action_types()),
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
        self.assertIn(
            "requires exactly one Agent with execution_mode='react'",
            duplicate_owner.feedback,
        )
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
