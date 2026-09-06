from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import (
    AgentRequest,
    AgentResponse,
    AgentRuntime,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.director import director_validate_live_action_target_domains
from src.interactive.model_registry import (
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)


_ALLOWED_ACTIONS = (
    "add_subgraph",
    "modify_agent",
    "delete_agent",
    "set_relation",
    "set_output",
    "continue",
    "finish",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _config_path(name: str) -> str:
    return str(_REPOSITORY_ROOT / "config" / name)


def _registry() -> ModelRegistry:
    provider = ProviderSpec("fake", kind="test", api_key_env="FAKE_API_KEY")
    return ModelRegistry(
        [provider],
        [
            ModelSpec("balanced", "fake"),
            ModelSpec("cheap", "fake"),
        ],
    )


class _RecordingGateway:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def generate(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return "public-state analysis for the next admissible action"


class _OneActionAdapter:
    """Stepwise execution fixture: one call represents one Tool action."""

    stepwise_director = True

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            "go to cabinet 2",
            metadata={
                "environment_current_state": {
                    "environment_episode_id": "episode-v7",
                    "environment_id": "alfworld",
                    "task_family": "alfworld",
                    "environment_revision": 5,
                    "last_action": "go to cabinet 2",
                    "current_observation": "You arrive at cabinet 2.",
                    "admissible_actions": ["open cabinet 2"],
                    "task_facts": {
                        "target_class": "toiletpaper",
                        "destination_class": "toilet",
                        "required_transform": None,
                        "count": 2,
                        "examine_with_desklamp": False,
                    },
                    "remaining_action_budget": 15,
                    "total_action_budget": 20,
                    "environment_terminal": False,
                    "environment_truncated": False,
                    "action_observation_history": [
                        {
                            "turn": 5,
                            "action": "go to cabinet 2",
                            "observation_result": "You arrive at cabinet 2.",
                            "state_advanced": True,
                            "environment_terminal": False,
                        }
                    ],
                }
            },
        )


class _BudgetExhaustedAdapter(_OneActionAdapter):
    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            "go to cabinet 2",
            metadata={
                "environment_terminal": False,
                "environment_truncated": True,
                "evaluator_environment_trace": [
                    {
                        "action": "go to cabinet 2",
                        "done": False,
                        "state_advanced": True,
                    }
                ],
                "environment_current_state": {
                    "environment_episode_id": "episode-v7-budget",
                    "environment_id": "alfworld",
                    "task_family": "alfworld",
                    "environment_revision": 20,
                    "last_action": "go to cabinet 2",
                    "current_observation": "You arrive at cabinet 2.",
                    "admissible_actions": [],
                    "task_facts": {
                        "target_class": "toiletpaper",
                        "destination_class": "toilet",
                        "required_transform": None,
                        "count": 2,
                        "examine_with_desklamp": False,
                    },
                    "remaining_action_budget": 0,
                    "total_action_budget": 20,
                    "environment_terminal": False,
                    "environment_truncated": True,
                    "action_observation_history": [
                        {
                            "turn": 20,
                            "action": "go to cabinet 2",
                            "observation_result": "You arrive at cabinet 2.",
                            "state_advanced": True,
                            "environment_terminal": False,
                        }
                    ],
                },
            },
        )


class _TerminalSuccessAdapter(_OneActionAdapter):
    async def execute(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            "move toiletpaper 2 to toilet 1",
            metadata={
                "environment_terminal": True,
                "environment_truncated": False,
                "evaluator_environment_trace": [
                    {
                        "action": "move toiletpaper 2 to toilet 1",
                        "done": True,
                        "state_advanced": True,
                    }
                ],
                "environment_current_state": {
                    "environment_episode_id": "episode-v7-terminal",
                    "environment_id": "alfworld",
                    "task_family": "alfworld",
                    "environment_revision": 20,
                    "last_action": "move toiletpaper 2 to toilet 1",
                    "current_observation": (
                        "You move the toiletpaper 2 to the toilet 1."
                    ),
                    "admissible_actions": [],
                    "task_facts": {
                        "target_class": "toiletpaper",
                        "destination_class": "toilet",
                        "required_transform": None,
                        "count": 2,
                        "examine_with_desklamp": False,
                    },
                    "remaining_action_budget": 0,
                    "total_action_budget": 20,
                    "environment_terminal": True,
                    "environment_truncated": False,
                    "action_observation_history": [
                        {
                            "turn": 20,
                            "action": "move toiletpaper 2 to toilet 1",
                            "observation_result": (
                                "You move the toiletpaper 2 to the toilet 1."
                            ),
                            "state_advanced": True,
                            "environment_terminal": True,
                        }
                    ],
                },
            },
        )


class _ALFWorldRuntime(AgentRuntime):
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        gateway: _RecordingGateway | None = None,
        adapter: _OneActionAdapter | None = None,
    ) -> None:
        self.recording_gateway = gateway or _RecordingGateway()
        self.action_adapter = adapter or _OneActionAdapter()
        super().__init__(
            registry,
            self.recording_gateway,
            execution_adapters={"react": self.action_adapter},
            dataset_id="alfworld",
        )

    def registered_execution_profiles(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("reasoning", ()),
            ("react", ()),
            ("react", ("alfworld.environment",)),
        )

    def validate_execution_contracts(self, nodes: tuple[AgentNode, ...]) -> None:
        # These tests target AgentWorkflowEnv's authoritative ALFWorld profile
        # and communication gates.  The real Tool registry is covered by the
        # existing environment-execution integration tests.
        return None


def _environment(
    problem: str,
    *,
    graph: AgentGraph | None = None,
    execute_on_edit: bool = False,
    runtime: _ALFWorldRuntime | None = None,
) -> AgentWorkflowEnv:
    registry = _registry() if runtime is None else runtime.model_registry
    resolved_runtime = runtime or _ALFWorldRuntime(registry)
    return AgentWorkflowEnv(
        registry,
        runtime=resolved_runtime,
        problem=problem,
        graph=graph,
        execute_on_edit=execute_on_edit,
        max_agents=8,
        max_agents_per_subgraph=3,
        required_tool_id="alfworld.environment",
        allowed_actions=_ALLOWED_ACTIONS,
        recovery_policy="preserve_diagnose_repair_augment",
        require_multi_agent_for_complex_tasks=True,
        minimum_agents_for_complex_tasks=2,
    )


def _collaborating_graph() -> AgentGraph:
    return AgentGraph(
        [
            AgentNode(
                "analysis",
                "cheap",
                "Analyze the current public observation and propose one next action.",
                execution_mode="reasoning",
            ),
            AgentNode(
                "actor",
                "balanced",
                "Use the routed analysis to execute one admissible environment action.",
                allowed_tools=("alfworld.environment",),
                execution_mode="react",
            ),
        ],
        [AgentRelation("analysis", "actor", True, False)],
        output_agent_id="actor",
    )


def _collaborating_graph_without_output() -> AgentGraph:
    graph = _collaborating_graph()
    return AgentGraph(graph.nodes, graph.relations)


def _terminal_downstream_graph(*, with_middle: bool = False) -> AgentGraph:
    """Return a free AgentGraph whose Tool owner feeds one unique sink."""

    graph = _collaborating_graph_without_output()
    upstream_id = "actor"
    if with_middle:
        graph.add_agent(
            AgentNode(
                "middle",
                "cheap",
                "Carry the current public environment artifact downstream.",
                execution_mode="reasoning",
            )
        )
        graph.set_relation(upstream_id, "middle", True, False)
        upstream_id = "middle"
    graph.add_agent(
        AgentNode(
            "downstream",
            "cheap",
            "Preserve the current terminal public artifact.",
            execution_mode="reasoning",
        )
    )
    graph.set_relation(upstream_id, "downstream", True, False)
    return graph


async def _prime_terminal_execution(
    env: AgentWorkflowEnv,
    runtime: _ALFWorldRuntime,
):
    execution = await runtime.execute(
        env.graph,
        env._runtime_problem(),
        require_complete=False,
    )
    env._progressive_outputs = dict(execution.outputs)
    env._progressive_output_metadata = {
        agent_id: dict(metadata)
        for agent_id, metadata in execution.output_metadata.items()
    }
    env._progressive_execution = execution
    env._progressive_execution_revision = env.revision
    return execution


class ALFWorldMultiAgentV7ConfigTests(unittest.TestCase):
    def test_v7_config_keeps_v6_evaluation_condition_and_disables_training(
        self,
    ) -> None:
        v6 = load_yaml(_config_path("evaluation_alfworld_stepwise_recovery_v6.yaml"))
        v7 = load_yaml(_config_path("evaluation_alfworld_stepwise_multiagent_v7.yaml"))
        validate_agent_graph_config(v7)

        self.assertEqual(
            [
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "continue",
                "finish",
            ],
            v7["agent_graph"]["actions"],
        )
        self.assertEqual(3, v7["agent_graph"]["max_agents_per_subgraph"])
        self.assertEqual(
            {
                "enabled": True,
                "dataset_scope": ["alfworld"],
                "minimum_agents": 2,
                "require_artifact_delivery": True,
            },
            v7["agent_graph"]["complex_task_collaboration"],
        )

        for path in (
            ("experiment", "seed"),
            ("data", "test_path"),
            ("alfworld_evaluation", "official_split"),
            ("alfworld_evaluation", "sample_count"),
            ("alfworld_evaluation", "stable_zero_sample_count"),
            ("alfworld_evaluation", "direct_model_id"),
            ("alfworld_evaluation", "direct_protocol"),
            ("environment_runtime", "max_environment_steps_by_source"),
            ("director", "base_model"),
            ("director", "max_rounds"),
            ("agent_graph", "model_catalog_path"),
            ("evaluation", "max_environment_steps_by_source"),
        ):
            section, field = path
            with self.subTest(path=path):
                self.assertEqual(v6[section][field], v7[section][field])

        self.assertFalse(v7["experiment"]["training_enabled"])
        self.assertFalse(v7["grpo"]["enabled"])
        self.assertEqual(0, v7["grpo"]["max_optimizer_updates"])
        self.assertFalse(v7["director"]["lora"]["enabled"])
        self.assertFalse(v7["policy_sync"]["enabled"])
        self.assertFalse(v7["skills"]["enabled"])
        self.assertFalse(v7["exploration"]["enabled"])
        self.assertFalse(v7["gpu"]["training_enabled"])
        self.assertIn("multiagent_v7", v7["experiment"]["output_dir"])
        self.assertNotIn("recovery_v6", v7["experiment"]["output_dir"])

    def test_collaboration_config_is_fail_closed(self) -> None:
        config = load_yaml(
            _config_path("evaluation_alfworld_stepwise_multiagent_v7.yaml")
        )
        invalid = copy.deepcopy(config)
        invalid["agent_graph"]["complex_task_collaboration"][
            "require_artifact_delivery"
        ] = False
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(invalid)


class ALFWorldMultiAgentV7CanvasTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_grounding_failure_targets_measured_agent_repair(
        self,
    ) -> None:
        env = _environment(
            "put a clean lettuce in countertop.",
            graph=_collaborating_graph(),
        )
        env._progressive_output_metadata["actor"] = {
            "environment_current_state": {
                "environment_episode_id": "episode-1",
                "environment_id": "alfworld",
                "environment_revision": 12,
                "current_observation": (
                    "You arrive at cabinet 4. The cabinet 4 is closed."
                ),
                "admissible_actions": ["open cabinet 4"],
                "remaining_action_budget": 8,
                "total_action_budget": 20,
                "environment_terminal": False,
                "environment_truncated": False,
                "collaborator_action_proposals": [
                    {
                        "source_agent_id": "analysis",
                        "artifact_id": "analysis-current",
                        "proposed_action": "go to cabinet 4",
                        "admissible": False,
                    }
                ],
                "collaborator_action_alignment": {
                    "schema_version": (
                        "alfworld.collaborator-action-grounding.v1"
                    ),
                    "status": "proposal_not_admissible",
                    "proposed_actions": ["go to cabinet 4"],
                    "admissible_proposed_actions": [],
                    "executed_action": "open cabinet 4",
                },
                "stall_diagnostic": {
                    "schema_version": "alfworld.public-stall.v1",
                    "stalled": True,
                    "signals": [
                        "no_goal_predicate_progress",
                        "collaborator_action_grounding_failure",
                    ],
                },
            },
            "input_artifact_provenance": [
                {
                    "source_agent_id": "analysis",
                    "target_agent_id": "actor",
                    "artifact_id": "analysis-current",
                    "artifact_body": '{"command":"go to cabinet 4"}',
                    "graph_revision": env.graph.revision,
                }
            ],
        }

        self.assertEqual(("modify_agent",), env.model_admissible_action_types())
        targets = env.model_admissible_action_targets()["modify_agent"]
        self.assertEqual(["analysis"], targets["agent_ids"])

    async def test_complex_public_task_requires_two_free_contract_agents(
        self,
    ) -> None:
        for problem in (
            "put two toiletpaper in toilet.",
            "heat some tomato and put it in fridge.",
        ):
            with self.subTest(problem=problem):
                env = _environment(problem)
                actions = env.model_admissible_action_types()
                targets = env.model_admissible_action_targets()

                self.assertEqual(("add_subgraph",), actions)
                director_validate_live_action_target_domains(actions, targets)
                domain = targets["add_subgraph"]
                self.assertEqual(2, domain["min_new_agents"])
                self.assertEqual(3, domain["max_new_agents"])
                self.assertEqual(
                    "free_contract_execution_profile",
                    domain["declaration_mode"],
                )
                self.assertEqual("free_text", domain["contract_type"])
                self.assertNotIn("role_family", domain["required_agent_fields"])
                self.assertNotIn("role_constraints", domain)
                self.assertEqual(
                    [
                        {
                            "execution_mode": "react",
                            "allowed_tools": ["alfworld.environment"],
                        },
                        {
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                        },
                    ],
                    domain["execution_profiles"],
                )
                self.assertEqual(
                    [
                        {
                            "execution_mode": "react",
                            "allowed_tools": ["alfworld.environment"],
                            "min_count": 1,
                            "max_count": 1,
                        },
                        {
                            "execution_mode": "reasoning",
                            "allowed_tools": [],
                            "min_count": 1,
                            "max_count": 2,
                        },
                    ],
                    domain["required_execution_profile_counts"],
                )
                self.assertTrue(
                    domain["require_environment_owner_inbound_relation"]
                )
                self.assertEqual(1, domain["min_relations"])

    async def test_simple_pick_and_place_keeps_one_to_three_agent_domain(
        self,
    ) -> None:
        env = _environment("put some apple in fridge.")
        actions = env.model_admissible_action_types()
        targets = env.model_admissible_action_targets()

        self.assertEqual(("add_subgraph",), actions)
        director_validate_live_action_target_domains(actions, targets)
        domain = targets["add_subgraph"]
        self.assertEqual(1, domain["min_new_agents"])
        self.assertEqual(3, domain["max_new_agents"])
        auxiliary_count = domain["required_execution_profile_counts"][1]
        self.assertEqual(0, auxiliary_count["min_count"])
        self.assertFalse(domain["require_environment_owner_inbound_relation"])
        self.assertEqual(0, domain["min_relations"])

    async def test_complex_add_subgraph_requires_owner_and_inbound_artifact_path(
        self,
    ) -> None:
        env = _environment("put two toiletpaper in toilet.")

        one_agent = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"actor","model_id":"balanced",'
            '"contract":"Execute one admissible action.",'
            '"execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
            '],"relations":[],"output_agent_id":"actor"}'
        )
        self.assertFalse(one_agent.accepted)
        self.assertIn("at least 2 Agents", one_agent.feedback)
        self.assertEqual((), env.graph.nodes)

        wrong_direction = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"actor","model_id":"balanced",'
            '"contract":"Execute one admissible action.",'
            '"execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]},'
            '{"agent_id":"analysis","model_id":"cheap",'
            '"contract":"Analyze the public observation.",'
            '"execution_mode":"reasoning","allowed_tools":[]}'
            '],"relations":['
            '{"source_id":"actor","target_id":"analysis",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"analysis"}'
        )
        self.assertFalse(wrong_direction.accepted)
        self.assertIn("route at least one stateless collaborator artifact", wrong_direction.feedback)
        self.assertEqual((), env.graph.nodes)

        accepted = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"actor","model_id":"balanced",'
            '"contract":"Execute one admissible action from routed analysis.",'
            '"execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]},'
            '{"agent_id":"analysis","model_id":"cheap",'
            '"contract":"Analyze the public observation.",'
            '"execution_mode":"reasoning","allowed_tools":[]}'
            '],"relations":['
            '{"source_id":"analysis","target_id":"actor",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"actor"}'
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(2, len(env.graph.nodes))
        self.assertEqual(("actor",), env._required_tool_actor_ids())
        self.assertTrue(all(node.role_family is None for node in env.graph.nodes))

    async def test_react_role_and_owner_reciprocal_relation_are_rejected(
        self,
    ) -> None:
        env = _environment("heat some tomato and put it in fridge.")
        react_role = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"analysis","model_id":"cheap",'
            '"contract":"Analyze public state.",'
            '"role_family":"react","execution_mode":"reasoning",'
            '"allowed_tools":[]},'
            '{"agent_id":"actor","model_id":"balanced",'
            '"contract":"Execute one native action.",'
            '"execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
            '],"relations":['
            '{"source_id":"analysis","target_id":"actor",'
            '"source_to_target":true,"target_to_source":false}'
            '],"output_agent_id":"actor"}'
        )
        self.assertFalse(react_role.accepted)
        self.assertIn("ReAct is an execution mode, not an Agent role", react_role.feedback)

        reciprocal = await env.step(
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"analysis","model_id":"cheap",'
            '"contract":"Analyze public state.",'
            '"execution_mode":"reasoning","allowed_tools":[]},'
            '{"agent_id":"actor","model_id":"balanced",'
            '"contract":"Execute one native action.",'
            '"execution_mode":"react",'
            '"allowed_tools":["alfworld.environment"]}'
            '],"relations":['
            '{"source_id":"analysis","target_id":"actor",'
            '"source_to_target":true,"target_to_source":true}'
            '],"output_agent_id":"actor"}'
        )
        self.assertFalse(reciprocal.accepted)
        self.assertIn("cannot participate in a reciprocal Agent block", reciprocal.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_public_state_excludes_private_evaluator_fields(self) -> None:
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_collaborating_graph(),
        )
        env._progressive_output_metadata["actor"] = {
            "environment_current_state": {
                "environment_revision": 4,
                "current_observation": "You are in the middle of a room.",
                "admissible_actions": ["go to cabinet 1"],
                "policy_action_domain": ["go to cabinet 1"],
                "task_facts": {
                    "target_class": "toiletpaper",
                    "destination_class": "toilet",
                    "required_transform": None,
                    "count": 2,
                    "examine_with_desklamp": False,
                },
                "goal_progress": {
                    "acquired_target_instances": [],
                    "placed_target_instances": [],
                    "turns_since_goal_progress": 4,
                },
                "held_objects": [],
                "policy_action_domain_receipt": {
                    "schema_version": (
                        "skillflow.alfworld.policy-action-domain.v2"
                    ),
                    "profile": "skillflow_public_invariants_v2",
                    "environment_admissible_actions": ["go to cabinet 1"],
                    "policy_action_domain": ["go to cabinet 1"],
                    "blocked_actions": [],
                    "fail_open": False,
                    "goal_progress": {
                        "acquired_target_instances": [],
                    },
                    "held_objects": [],
                    "reward": 1.0,
                    "won": True,
                    "info": {"expert_plan": ["hidden"]},
                },
                "remaining_action_budget": 16,
                "total_action_budget": 20,
                "environment_terminal": False,
                "environment_truncated": False,
                "reward": 1.0,
                "won": True,
                "info": {"expert_plan": ["hidden"]},
                "action_observation_history": [
                    {
                        "turn": 4,
                        "action": "go to cabinet 1",
                        "observation_result": "Cabinet 1 is empty.",
                        "state_advanced": True,
                        "reward": 1.0,
                        "info": {"won": True},
                    }
                ],
            },
            "input_artifact_provenance": [],
        }

        state = env.public_environment_state()

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(2, state["task_facts"]["count"])
        self.assertEqual(
            ["go to cabinet 1"],
            state["policy_action_domain"],
        )
        self.assertEqual(
            4,
            state["goal_progress"]["turns_since_goal_progress"],
        )
        self.assertEqual(
            "skillflow_public_invariants_v2",
            state["policy_action_domain_receipt"]["profile"],
        )
        rendered = str(state)
        self.assertNotIn("reward", rendered)
        self.assertNotIn("won", rendered)
        self.assertNotIn("info", rendered)
        self.assertNotIn("expert_plan", rendered)

    async def test_continue_reexecutes_all_agents_but_one_tool_owner_once(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _OneActionAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_collaborating_graph(),
            runtime=runtime,
        )
        graph_revision = env.revision
        env._progressive_outputs.update(
            {
                "analysis": "stale analysis",
                "actor": "go to cabinet 1",
            }
        )
        env._progressive_output_metadata["analysis"] = {
            "artifact_id": "analysis:old"
        }
        env._progressive_output_metadata["actor"] = {
            "artifact_id": "actor:old",
            "environment_current_state": {
                "environment_episode_id": "episode-v7",
                "environment_id": "alfworld",
                "task_family": "alfworld",
                "environment_revision": 4,
                "current_observation": "You are in the middle of a room.",
                "admissible_actions": ["go to cabinet 2"],
                "task_facts": {
                    "target_class": "toiletpaper",
                    "destination_class": "toilet",
                    "required_transform": None,
                    "count": 2,
                    "examine_with_desklamp": False,
                },
                "remaining_action_budget": 16,
                "total_action_budget": 20,
                "environment_terminal": False,
                "environment_truncated": False,
                "action_observation_history": [],
            },
            "input_artifact_provenance": [
                {
                    "source_agent_id": "analysis",
                    "target_agent_id": "actor",
                    "graph_revision": graph_revision,
                    "artifact_id": "analysis:old",
                    "artifact_body": "stale analysis",
                }
            ],
        }

        self.assertIn("continue", env.model_admissible_action_types())
        result = await env.step('{"action":"continue"}')

        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.execution)
        assert result.execution is not None
        self.assertEqual(
            {"analysis", "actor"},
            set(result.execution.executed_agent_ids),
        )
        self.assertEqual(1, len(gateway.requests))
        self.assertEqual("analysis", gateway.requests[0].agent.id)
        self.assertEqual(1, len(adapter.requests))
        owner_request = adapter.requests[0]
        self.assertEqual("actor", owner_request.agent.id)
        self.assertEqual(
            ["analysis"],
            [message.source_agent_id for message in owner_request.upstream],
        )
        self.assertIn("[PUBLIC ENVIRONMENT STATE]", gateway.requests[0].problem)
        self.assertIn("You are in the middle of a room.", owner_request.problem)
        self.assertEqual(5, env.public_environment_state()["environment_revision"])
        execution_feedback = json.loads(
            result.feedback.partition("execution_result=")[2]
        )
        artifacts_by_agent = {
            item["agent_id"]: item
            for item in execution_feedback["agent_artifacts"]
        }
        self.assertEqual(
            "public-state analysis for the next admissible action",
            artifacts_by_agent["analysis"]["artifact_body"],
        )
        actor_inputs = artifacts_by_agent["actor"][
            "input_artifact_provenance"
        ]
        self.assertEqual(1, len(actor_inputs))
        self.assertEqual("analysis", actor_inputs[0]["source_agent_id"])
        self.assertEqual("actor", actor_inputs[0]["target_agent_id"])
        self.assertEqual(
            "public-state analysis for the next admissible action",
            actor_inputs[0]["artifact_body"],
        )

    async def test_budget_exhaustion_closes_to_finish_without_output_ping_pong(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _BudgetExhaustedAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_collaborating_graph(),
            runtime=runtime,
        )
        graph_revision = env.revision
        env._progressive_outputs.update(
            {
                "analysis": "analyze the final available transition",
                "actor": "go to cabinet 1",
            }
        )
        env._progressive_output_metadata["analysis"] = {
            "artifact_id": "analysis:19"
        }
        env._progressive_output_metadata["actor"] = {
            "artifact_id": "actor:19",
            "environment_current_state": {
                "environment_episode_id": "episode-v7-budget",
                "environment_id": "alfworld",
                "task_family": "alfworld",
                "environment_revision": 19,
                "current_observation": "You are in the middle of a room.",
                "admissible_actions": ["go to cabinet 2"],
                "task_facts": {
                    "target_class": "toiletpaper",
                    "destination_class": "toilet",
                    "required_transform": None,
                    "count": 2,
                    "examine_with_desklamp": False,
                },
                "remaining_action_budget": 1,
                "total_action_budget": 20,
                "environment_terminal": False,
                "environment_truncated": False,
                "action_observation_history": [],
            },
            "input_artifact_provenance": [
                {
                    "source_agent_id": "analysis",
                    "target_agent_id": "actor",
                    "graph_revision": graph_revision,
                    "artifact_id": "analysis:19",
                    "artifact_body": "analyze the final available transition",
                }
            ],
        }

        continued = await env.step('{"action":"continue"}')
        self.assertTrue(continued.accepted)
        state = env.public_environment_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state["environment_truncated"])
        self.assertEqual(0, state["remaining_action_budget"])

        self.assertEqual(("finish",), env.model_admissible_action_types())
        targets = env.model_admissible_action_targets()
        self.assertEqual({"finish"}, set(targets))
        self.assertTrue(targets["finish"]["admissible"])

        output_ping_pong = await env.step(
            '{"action":"set_output","agent_id":"analysis"}'
        )
        self.assertFalse(output_ping_pong.accepted)
        self.assertIn("not currently admissible", output_ping_pong.feedback)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(1, len(adapter.requests))

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertTrue(env.finished)
        self.assertEqual(1, len(adapter.requests))

    async def test_budget_exhaustion_projects_unique_owner_without_output_edit(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _BudgetExhaustedAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_collaborating_graph_without_output(),
            execute_on_edit=True,
            runtime=runtime,
        )
        graph_revision = env.revision
        env._progressive_outputs.update(
            {
                "analysis": "analyze the final available transition",
                "actor": "go to cabinet 1",
            }
        )
        env._progressive_output_metadata["analysis"] = {
            "artifact_id": "analysis:19"
        }
        env._progressive_output_metadata["actor"] = {
            "artifact_id": "actor:19",
            "environment_current_state": {
                "environment_episode_id": "episode-v7-budget",
                "environment_id": "alfworld",
                "task_family": "alfworld",
                "environment_revision": 19,
                "current_observation": "You are in the middle of a room.",
                "admissible_actions": ["go to cabinet 2"],
                "task_facts": {
                    "target_class": "toiletpaper",
                    "destination_class": "toilet",
                    "required_transform": None,
                    "count": 2,
                    "examine_with_desklamp": False,
                },
                "remaining_action_budget": 1,
                "total_action_budget": 20,
                "environment_terminal": False,
                "environment_truncated": False,
                "action_observation_history": [],
            },
            "input_artifact_provenance": [
                {
                    "source_agent_id": "analysis",
                    "target_agent_id": "actor",
                    "graph_revision": graph_revision,
                    "artifact_id": "analysis:19",
                    "artifact_body": "analyze the final available transition",
                }
            ],
        }

        continued = await env.step('{"action":"continue"}')
        self.assertTrue(continued.accepted)
        self.assertEqual(1, len(adapter.requests))
        self.assertIsNone(env.graph.output_agent_id)
        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("actor", admission["prospective_output_agent_id"])

        ping_pong = await env.step(
            '{"action":"set_output","agent_id":"analysis"}'
        )
        self.assertFalse(ping_pong.accepted)
        self.assertIsNone(env.graph.output_agent_id)
        self.assertEqual(1, len(adapter.requests))

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(1, len(adapter.requests))
        self.assertNotIn(
            "set_output",
            [
                entry.action.action_type.value
                for entry in env.history
                if entry.accepted and entry.action is not None
            ],
        )

    async def test_terminal_success_projects_unique_tool_owner_on_finish(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_collaborating_graph_without_output(),
            execute_on_edit=True,
            runtime=runtime,
        )
        graph_revision = env.revision
        env._progressive_outputs.update(
            {
                "analysis": "analyze the final available transition",
                "actor": "move toiletpaper 2 to toilet 1",
            }
        )
        env._progressive_output_metadata["analysis"] = {
            "artifact_id": "analysis:19"
        }
        env._progressive_output_metadata["actor"] = {
            "artifact_id": "actor:19",
            "environment_current_state": {
                "environment_episode_id": "episode-v7-terminal",
                "environment_id": "alfworld",
                "task_family": "alfworld",
                "environment_revision": 19,
                "current_observation": "You arrive at toilet 1.",
                "admissible_actions": [
                    "move toiletpaper 2 to toilet 1"
                ],
                "task_facts": {
                    "target_class": "toiletpaper",
                    "destination_class": "toilet",
                    "required_transform": None,
                    "count": 2,
                    "examine_with_desklamp": False,
                },
                "remaining_action_budget": 1,
                "total_action_budget": 20,
                "environment_terminal": False,
                "environment_truncated": False,
                "action_observation_history": [],
            },
            "input_artifact_provenance": [
                {
                    "source_agent_id": "analysis",
                    "target_agent_id": "actor",
                    "graph_revision": graph_revision,
                    "artifact_id": "analysis:19",
                    "artifact_body": "analyze the final available transition",
                }
            ],
        }

        continued = await env.step('{"action":"continue"}')
        self.assertTrue(continued.accepted)
        self.assertIsNone(env.graph.output_agent_id)
        self.assertEqual(1, len(adapter.requests))
        calls_before_finish = len(gateway.requests)
        revision_before_finish = env.revision

        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("actor", admission["prospective_output_agent_id"])
        self.assertEqual(
            "unique_environment_tool_owner_receipt",
            admission["terminal_output_source"],
        )

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(revision_before_finish + 1, env.revision)
        self.assertEqual(
            "move toiletpaper 2 to toilet 1",
            finished.final_answer,
        )
        self.assertEqual(1, len(adapter.requests))
        self.assertEqual(calls_before_finish, len(gateway.requests))
        self.assertNotIn(
            "set_output",
            [
                entry.action.action_type.value
                for entry in env.history
                if entry.action is not None
            ],
        )

    async def test_terminal_projection_rebinds_invalid_existing_output_to_owner(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        graph = _collaborating_graph_without_output()
        graph.set_output("analysis")
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=graph,
            execute_on_edit=True,
            runtime=runtime,
        )
        execution = await _prime_terminal_execution(env, runtime)
        issue_codes = {
            issue.code
            for issue in env.graph.validate(
                registry,
                require_complete=True,
            ).issues
        }
        self.assertIn("output_not_sink", issue_codes)
        self.assertIn("cannot_reach_output", issue_codes)
        relations_before = env.graph.relations
        revision_before = env.revision
        gateway_calls = len(gateway.requests)
        tool_calls = len(adapter.requests)

        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("actor", admission["prospective_output_agent_id"])
        self.assertEqual(
            "unique_environment_tool_owner_receipt",
            admission["terminal_output_source"],
        )
        self.assertEqual("analysis", env.graph.output_agent_id)
        self.assertEqual(revision_before, env.revision)

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(revision_before + 1, env.revision)
        self.assertEqual(execution.outputs["actor"], finished.final_answer)
        self.assertEqual(relations_before, env.graph.relations)
        self.assertEqual(gateway_calls, len(gateway.requests))
        self.assertEqual(tool_calls, len(adapter.requests))
        self.assertNotIn(
            "set_output",
            [
                entry.action.action_type.value
                for entry in env.history
                if entry.action is not None
            ],
        )

    async def test_budget_closure_rebinds_invalid_existing_output_to_owner(
        self,
    ) -> None:
        """A SkillFlow fixed-budget closure still receives explicit FINISH."""

        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _BudgetExhaustedAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        graph = _collaborating_graph_without_output()
        graph.set_output("analysis")
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=graph,
            execute_on_edit=True,
            runtime=runtime,
        )
        execution = await _prime_terminal_execution(env, runtime)
        issue_codes = {
            issue.code
            for issue in env.graph.validate(
                registry,
                require_complete=True,
            ).issues
        }
        self.assertIn("output_not_sink", issue_codes)
        self.assertIn("cannot_reach_output", issue_codes)
        relations_before = env.graph.relations
        revision_before = env.revision
        gateway_calls = len(gateway.requests)
        tool_calls = len(adapter.requests)

        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("actor", admission["prospective_output_agent_id"])
        self.assertEqual(
            "unique_environment_tool_owner_receipt",
            admission["terminal_output_source"],
        )

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(revision_before + 1, env.revision)
        self.assertEqual(execution.outputs["actor"], finished.final_answer)
        self.assertEqual(relations_before, env.graph.relations)
        self.assertEqual(gateway_calls, len(gateway.requests))
        self.assertEqual(tool_calls, len(adapter.requests))

    async def test_terminal_projection_preserves_existing_valid_output(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        graph = _terminal_downstream_graph()
        graph.set_output("downstream")
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=graph,
            execute_on_edit=True,
            runtime=runtime,
        )
        execution = await _prime_terminal_execution(env, runtime)
        revision_before = env.revision
        gateway_calls = len(gateway.requests)
        tool_calls = len(adapter.requests)

        self.assertIsNone(env._terminal_environment_owner_projection())
        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertNotIn("prospective_output_agent_id", admission)

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("downstream", env.graph.output_agent_id)
        self.assertEqual(revision_before, env.revision)
        self.assertEqual(execution.outputs["downstream"], finished.final_answer)
        self.assertEqual(gateway_calls, len(gateway.requests))
        self.assertEqual(tool_calls, len(adapter.requests))

    async def test_terminal_multi_sink_canvas_finishes_without_structural_edits(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        nodes = (
            AgentNode(
                "analysis",
                "cheap",
                "Analyze the public environment receipt.",
                execution_mode="reasoning",
            ),
            AgentNode(
                "actor",
                "balanced",
                "Execute one admissible environment action.",
                allowed_tools=("alfworld.environment",),
                execution_mode="react",
            ),
            AgentNode(
                "aux1",
                "cheap",
                "Preserve one public diagnostic artifact.",
                execution_mode="reasoning",
            ),
            AgentNode(
                "aux2",
                "cheap",
                "Preserve another public diagnostic artifact.",
                execution_mode="reasoning",
            ),
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=AgentGraph(
                nodes,
                [AgentRelation("analysis", "actor", True, False)],
            ),
            execute_on_edit=True,
            runtime=runtime,
        )
        relations_before = env.graph.relations
        execution = await runtime.execute(
            env.graph,
            env._runtime_problem(),
            require_complete=False,
        )
        env._progressive_outputs = dict(execution.outputs)
        env._progressive_output_metadata = {
            agent_id: dict(metadata)
            for agent_id, metadata in execution.output_metadata.items()
        }
        env._progressive_execution = execution
        env._progressive_execution_revision = env.revision
        self.assertEqual(1, len(adapter.requests))
        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("actor", admission["prospective_output_agent_id"])
        state = env.public_environment_state()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state["environment_terminal"])
        calls_before_finish = len(gateway.requests)

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(relations_before, env.graph.relations)
        self.assertEqual("move toiletpaper 2 to toilet 1", finished.final_answer)
        self.assertEqual(1, len(adapter.requests))
        self.assertEqual(calls_before_finish, len(gateway.requests))
        self.assertNotIn(
            "set_output",
            [
                entry.action.action_type.value
                for entry in env.history
                if entry.action is not None
            ],
        )

    async def test_terminal_projection_finishes_with_current_downstream_sink(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_terminal_downstream_graph(),
            execute_on_edit=True,
            runtime=runtime,
        )
        relations_before = env.graph.relations

        execution = await _prime_terminal_execution(env, runtime)
        self.assertEqual(1, len(adapter.requests))
        candidate = env.graph.fork()
        candidate.set_output("actor")
        self.assertIn(
            "output_not_sink",
            {
                issue.code
                for issue in candidate.validate(
                    registry,
                    require_complete=True,
                ).issues
            },
        )
        owner_version = execution.output_metadata["actor"]["artifact_version"]
        downstream_metadata = execution.output_metadata["downstream"]
        self.assertEqual(
            owner_version,
            downstream_metadata["input_artifact_versions"]["actor"],
        )
        self.assertTrue(downstream_metadata["input_artifact_provenance"])

        calls_before_finish = len(gateway.requests)
        tool_calls_before_finish = len(adapter.requests)
        revision_before_finish = env.revision
        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("downstream", admission["prospective_output_agent_id"])
        self.assertEqual(
            "unique_complete_valid_sink_consumer",
            admission["terminal_output_source"],
        )

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("downstream", env.graph.output_agent_id)
        self.assertEqual(revision_before_finish + 1, env.revision)
        self.assertEqual(execution.outputs["downstream"], finished.final_answer)
        self.assertEqual(relations_before, env.graph.relations)
        self.assertEqual(calls_before_finish, len(gateway.requests))
        self.assertEqual(tool_calls_before_finish, len(adapter.requests))
        self.assertNotIn(
            "set_output",
            [
                entry.action.action_type.value
                for entry in env.history
                if entry.action is not None
            ],
        )

    async def test_terminal_projection_rebinds_invalid_output_to_current_sink(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        graph = _terminal_downstream_graph()
        graph.set_output("analysis")
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=graph,
            execute_on_edit=True,
            runtime=runtime,
        )
        execution = await _prime_terminal_execution(env, runtime)
        relations_before = env.graph.relations
        revision_before = env.revision
        gateway_calls = len(gateway.requests)
        tool_calls = len(adapter.requests)

        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual(
            "downstream",
            admission["prospective_output_agent_id"],
        )
        self.assertEqual(
            "unique_complete_valid_sink_consumer",
            admission["terminal_output_source"],
        )
        self.assertEqual("analysis", env.graph.output_agent_id)

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("downstream", env.graph.output_agent_id)
        self.assertEqual(revision_before + 1, env.revision)
        self.assertEqual(execution.outputs["downstream"], finished.final_answer)
        self.assertEqual(relations_before, env.graph.relations)
        self.assertEqual(gateway_calls, len(gateway.requests))
        self.assertEqual(tool_calls, len(adapter.requests))

    async def test_terminal_projection_finishes_over_two_current_artifact_hops(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_terminal_downstream_graph(with_middle=True),
            execute_on_edit=True,
            runtime=runtime,
        )
        relations_before = env.graph.relations

        execution = await _prime_terminal_execution(env, runtime)
        actor_version = execution.output_metadata["actor"]["artifact_version"]
        middle_version = execution.output_metadata["middle"]["artifact_version"]
        self.assertEqual(
            actor_version,
            execution.output_metadata["middle"]["input_artifact_versions"][
                "actor"
            ],
        )
        self.assertEqual(
            middle_version,
            execution.output_metadata["downstream"][
                "input_artifact_versions"
            ]["middle"],
        )

        calls_before_finish = len(gateway.requests)
        tool_calls_before_finish = len(adapter.requests)
        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        self.assertEqual("downstream", admission["prospective_output_agent_id"])
        self.assertEqual(
            "unique_complete_valid_sink_consumer",
            admission["terminal_output_source"],
        )

        finished = await env.step('{"action":"finish"}')

        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("downstream", env.graph.output_agent_id)
        self.assertEqual(execution.outputs["downstream"], finished.final_answer)
        self.assertEqual(relations_before, env.graph.relations)
        self.assertEqual(calls_before_finish, len(gateway.requests))
        self.assertEqual(tool_calls_before_finish, len(adapter.requests))
        self.assertNotIn(
            "set_output",
            [
                entry.action.action_type.value
                for entry in env.history
                if entry.action is not None
            ],
        )

    async def test_terminal_projection_uses_only_public_transition_receipt(
        self,
    ) -> None:
        registry = _registry()
        gateway = _RecordingGateway()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(
            registry,
            gateway=gateway,
            adapter=adapter,
        )
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=_terminal_downstream_graph(),
            execute_on_edit=True,
            runtime=runtime,
        )
        execution = await _prime_terminal_execution(env, runtime)
        metadata = {
            agent_id: dict(agent_metadata)
            for agent_id, agent_metadata in execution.output_metadata.items()
        }
        actor_metadata = dict(metadata["actor"])
        self.assertIn("evaluator_environment_trace", actor_metadata)
        actor_metadata.pop("evaluator_environment_trace")
        metadata["actor"] = actor_metadata
        public_only_execution = replace(
            execution,
            output_metadata=metadata,
        )
        env._progressive_output_metadata = {
            agent_id: dict(agent_metadata)
            for agent_id, agent_metadata in metadata.items()
        }
        env._progressive_execution = public_only_execution
        env._progressive_execution_revision = env.revision

        self.assertEqual(("finish",), env.model_admissible_action_types())
        admission = env.finish_admissibility()
        self.assertTrue(admission["admissible"])
        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual("downstream", env.graph.output_agent_id)

    async def test_terminal_projection_rejects_evaluator_only_or_untyped_state(
        self,
    ) -> None:
        for corruption in (
            "missing_public_transition",
            "missing_truncated_boolean",
            "null_truncated_boolean",
        ):
            with self.subTest(corruption=corruption):
                registry = _registry()
                adapter = _TerminalSuccessAdapter()
                runtime = _ALFWorldRuntime(registry, adapter=adapter)
                env = _environment(
                    "put two toiletpaper in toilet.",
                    graph=_terminal_downstream_graph(),
                    execute_on_edit=True,
                    runtime=runtime,
                )
                execution = await _prime_terminal_execution(env, runtime)
                metadata = {
                    agent_id: dict(agent_metadata)
                    for agent_id, agent_metadata in (
                        execution.output_metadata.items()
                    )
                }
                actor_metadata = dict(metadata["actor"])
                self.assertTrue(actor_metadata["evaluator_environment_trace"])
                current_state = dict(
                    actor_metadata["environment_current_state"]
                )
                if corruption == "missing_public_transition":
                    current_state.pop("latest_action_observation", None)
                    current_state.pop("action_observation_history", None)
                elif corruption == "missing_truncated_boolean":
                    current_state.pop("environment_truncated", None)
                else:
                    current_state["environment_truncated"] = None
                actor_metadata["environment_current_state"] = current_state
                metadata["actor"] = actor_metadata
                corrupted_execution = replace(
                    execution,
                    output_metadata=metadata,
                )
                env._progressive_output_metadata = {
                    agent_id: dict(agent_metadata)
                    for agent_id, agent_metadata in metadata.items()
                }
                env._progressive_execution = corrupted_execution
                env._progressive_execution_revision = env.revision

                self.assertIsNone(
                    env._terminal_environment_owner_projection()
                )
                admission = env.finish_admissibility()
                self.assertFalse(admission["admissible"])
                self.assertNotIn(
                    "finish",
                    env.model_admissible_action_types(),
                )

    async def test_terminal_projection_rejects_ambiguous_verified_sinks(
        self,
    ) -> None:
        registry = _registry()
        adapter = _TerminalSuccessAdapter()
        runtime = _ALFWorldRuntime(registry, adapter=adapter)
        graph = _collaborating_graph_without_output()
        for agent_id in ("sink_a", "sink_b"):
            graph.add_agent(
                AgentNode(
                    agent_id,
                    "cheap",
                    "Preserve one current public terminal artifact.",
                    execution_mode="reasoning",
                )
            )
            graph.set_relation("actor", agent_id, True, False)
        graph.set_relation("sink_a", "sink_b", True, True)
        env = _environment(
            "put two toiletpaper in toilet.",
            graph=graph,
            execute_on_edit=True,
            runtime=runtime,
        )
        execution = await _prime_terminal_execution(env, runtime)
        self.assertTrue(
            env._has_current_artifact_path(
                execution,
                source_id="actor",
                target_id="sink_a",
            )
        )
        self.assertTrue(
            env._has_current_artifact_path(
                execution,
                source_id="actor",
                target_id="sink_b",
            )
        )
        self.assertIsNone(env._terminal_environment_owner_projection())
        self.assertNotIn("finish", env.model_admissible_action_types())

    async def test_terminal_projection_rejects_dirty_or_failed_sink(self) -> None:
        for failure_state in ("dirty", "failed"):
            with self.subTest(failure_state=failure_state):
                registry = _registry()
                adapter = _TerminalSuccessAdapter()
                runtime = _ALFWorldRuntime(registry, adapter=adapter)
                env = _environment(
                    "put two toiletpaper in toilet.",
                    graph=_terminal_downstream_graph(),
                    execute_on_edit=True,
                    runtime=runtime,
                )
                await _prime_terminal_execution(env, runtime)
                if failure_state == "dirty":
                    env._unresolved_dirty_agents.add("downstream")
                else:
                    env._failed_agent_ids.add("downstream")

                self.assertIsNone(
                    env._terminal_environment_owner_projection()
                )
                self.assertNotIn(
                    "finish",
                    env.model_admissible_action_types(),
                )

    async def test_terminal_sink_projection_rejects_invalid_provenance(
        self,
    ) -> None:
        for corruption in (
            "missing_provenance",
            "version_mismatch",
            "body_mismatch",
            "revision_mismatch",
        ):
            with self.subTest(corruption=corruption):
                registry = _registry()
                gateway = _RecordingGateway()
                adapter = _TerminalSuccessAdapter()
                runtime = _ALFWorldRuntime(
                    registry,
                    gateway=gateway,
                    adapter=adapter,
                )
                env = _environment(
                    "put two toiletpaper in toilet.",
                    graph=_terminal_downstream_graph(),
                    execute_on_edit=True,
                    runtime=runtime,
                )
                execution = await _prime_terminal_execution(env, runtime)
                metadata = {
                    agent_id: dict(agent_metadata)
                    for agent_id, agent_metadata in (
                        execution.output_metadata.items()
                    )
                }
                sink_metadata = dict(metadata["downstream"])
                input_versions = dict(
                    sink_metadata["input_artifact_versions"]
                )
                provenance = [
                    dict(item)
                    for item in sink_metadata["input_artifact_provenance"]
                ]
                self.assertEqual(1, len(provenance))

                if corruption == "missing_provenance":
                    provenance = []
                elif corruption == "version_mismatch":
                    input_versions["actor"] = "stale-owner-artifact"
                    provenance[0]["artifact_version"] = (
                        "stale-owner-artifact"
                    )
                    provenance[0]["artifact_id"] = "stale-owner-artifact"
                elif corruption == "body_mismatch":
                    provenance[0]["artifact_body"] = "stale artifact body"
                else:
                    provenance[0]["graph_revision"] = env.revision + 1

                sink_metadata["input_artifact_versions"] = input_versions
                sink_metadata["input_artifact_provenance"] = provenance
                metadata["downstream"] = sink_metadata
                corrupted_execution = replace(
                    execution,
                    output_metadata=metadata,
                )
                env._progressive_output_metadata = {
                    agent_id: dict(agent_metadata)
                    for agent_id, agent_metadata in metadata.items()
                }
                env._progressive_execution = corrupted_execution
                env._progressive_execution_revision = env.revision

                gateway_calls = len(gateway.requests)
                tool_calls = len(adapter.requests)
                self.assertIsNone(
                    env._terminal_environment_owner_projection()
                )
                admission = env.finish_admissibility()
                self.assertFalse(admission["admissible"])
                self.assertEqual("graph_validation", admission["stage"])
                self.assertNotIn("finish", env.model_admissible_action_types())
                self.assertEqual(gateway_calls, len(gateway.requests))
                self.assertEqual(tool_calls, len(adapter.requests))

    async def test_preterminal_output_domain_contains_only_complete_graph_sinks(
        self,
    ) -> None:
        env = _environment(
            "put a hot mug in cabinet.",
            graph=_collaborating_graph_without_output(),
        )

        self.assertEqual(("actor",), env._model_admissible_output_agent_ids())
        rejected = await env.step(
            '{"action":"set_output","agent_id":"analysis"}'
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(
            "environment_output_target_unavailable",
            rejected.feedback_code,
        )
        self.assertIsNone(env.graph.output_agent_id)

    def test_relation_domain_never_removes_required_owner_ingress(self) -> None:
        env = _environment(
            "put a hot mug in cabinet.",
            graph=_collaborating_graph(),
        )

        self.assertEqual([], env._all_model_admissible_relation_candidates())


if __name__ == "__main__":
    unittest.main()
