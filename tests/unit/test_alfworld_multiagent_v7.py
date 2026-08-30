from __future__ import annotations

import copy
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

    async def test_budget_exhaustion_allows_one_output_selection_then_finish(
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
        self.assertEqual(("set_output",), env.model_admissible_action_types())
        output_targets = env.model_admissible_action_targets()["set_output"]
        # The Director remains free to choose any prospectively terminal-valid
        # Output Agent. In this directed topology only the sink is valid;
        # advertising the upstream analysis node would lead to a dead mask.
        self.assertEqual(["actor"], output_targets["agent_ids"])

        selected = await env.step(
            '{"action":"set_output","agent_id":"actor"}'
        )
        self.assertTrue(selected.accepted)
        self.assertEqual("actor", env.graph.output_agent_id)
        # SET_OUTPUT changes only the terminal artifact pointer; it must not
        # consume another native ALFWorld action after episode closure.
        self.assertEqual(1, len(adapter.requests))
        self.assertEqual(("finish",), env.model_admissible_action_types())

        ping_pong = await env.step(
            '{"action":"set_output","agent_id":"analysis"}'
        )
        self.assertFalse(ping_pong.accepted)
        self.assertEqual("actor", env.graph.output_agent_id)
        self.assertEqual(1, len(adapter.requests))

        finished = await env.step('{"action":"finish"}')
        self.assertTrue(finished.accepted)
        self.assertTrue(finished.done)
        self.assertEqual(1, len(adapter.requests))

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
