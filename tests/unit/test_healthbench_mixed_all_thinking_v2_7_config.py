from pathlib import Path
import json
import unittest

from jsonschema import Draft202012Validator
import yaml

from src.interactive.agent_graph import (
    AgentGraph,
    AgentNode,
    AgentRelation,
    GraphMutationError,
)
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION_V15,
    director_live_action_parameter_json_schema_text,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_system_prompt_for_version,
    encode_director_transcript,
)
from src.interactive.model_registry import (
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_7_heldout20_architecture.yaml"
)
V28_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_8_heldout20_architecture.yaml"
)
V210_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_10_heldout20_architecture.yaml"
)
V212_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_12_heldout20_output_closure.yaml"
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("model", "fake")],
    )


class HealthBenchMixedAllThinkingV27ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_heldout_population_and_reference_evaluator_are_fixed(self) -> None:
        condition = self.config["healthbench_professional_evaluation"]
        self.assertEqual("development", condition["stage"])
        self.assertEqual("test", condition["split"])
        self.assertEqual("task_ids", condition["selection"])
        self.assertEqual(20, condition["sample_count"])
        self.assertEqual(20, len(condition["task_ids"]))
        self.assertEqual(20, len(set(condition["task_ids"])))
        self.assertTrue(
            condition["direct_reused_from"].endswith(
                "healthbench_professional_mixed_all_thinking_v2_6/evaluation/direct_predictions.jsonl"
            )
        )
        self.assertEqual(
            "openai_simple_evals_healthbench_professional_reference",
            self.config["evaluation"]["healthbench_grader_mode"],
        )

    def test_v27_enables_only_inference_architecture_repairs(self) -> None:
        graph = self.config["agent_graph"]
        self.assertEqual(2, graph["max_relations_per_subgraph"])
        self.assertTrue(graph["require_output_protocol_artifact_for_set_output"])
        self.assertTrue(graph["require_reciprocal_terminal_artifact_lineage"])
        self.assertEqual(
            DIRECTOR_PROMPT_VERSION_V15,
            self.config["experiment"]["prompt_version"],
        )
        self.assertFalse(self.config["healthbench_tool_runtime"]["enabled"])
        self.assertFalse(self.config["grpo"]["enabled"])
        self.assertEqual(0, self.config["grpo"]["max_optimizer_updates"])
        self.assertFalse(self.config["policy_sync"]["enabled"])
        self.assertFalse(self.config["skills"]["enabled"])

    def test_v28_finishes_as_soon_as_the_repaired_output_is_admissible(self) -> None:
        v28 = yaml.safe_load(V28_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertTrue(v28["agent_graph"]["finish_only_when_admissible"])
        self.assertEqual(
            self.config["healthbench_professional_evaluation"]["task_ids"],
            v28["healthbench_professional_evaluation"]["task_ids"],
        )

    def test_v210_leaves_task_budget_for_canvas_failure_recovery(self) -> None:
        v210 = yaml.safe_load(V210_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertLess(
            v210["execution_timeout"],
            v210["healthbench_professional_evaluation"]["task_timeout_seconds"],
        )
        self.assertEqual(180.0, v210["execution_timeout"])
        self.assertEqual(
            self.config["healthbench_professional_evaluation"]["task_ids"],
            v210["healthbench_professional_evaluation"]["task_ids"],
        )
        self.assertFalse(v210["grpo"]["enabled"])
        self.assertFalse(v210["skills"]["enabled"])

    def test_v212_keeps_the_fixed_population_and_records_live_progress(self) -> None:
        v212 = yaml.safe_load(V212_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            self.config["healthbench_professional_evaluation"]["task_ids"],
            v212["healthbench_professional_evaluation"]["task_ids"],
        )
        self.assertEqual(20, v212["healthbench_professional_evaluation"]["sample_count"])
        self.assertEqual(2, v212["healthbench_professional_evaluation"]["stable_zero_sample_count"])
        self.assertLess(
            v212["execution_timeout"],
            v212["healthbench_professional_evaluation"]["task_timeout_seconds"],
        )
        self.assertTrue(
            v212["storage"]["rollout_progress_path"].endswith(
                "/rollout_progress.jsonl"
            )
        )
        self.assertFalse(v212["grpo"]["enabled"])
        self.assertFalse(v212["skills"]["enabled"])

    def test_director_policy_stays_role_and_topology_neutral(self) -> None:
        prompt = director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V15)
        self.assertIn("must not assert an unobserved domain fact", prompt)
        self.assertIn("any correction must be routed", prompt)
        for fixed_role in ("Doctor", "Researcher", "Verifier", "Formatter"):
            self.assertNotIn(fixed_role, prompt)
        self.assertNotIn("HealthBench", prompt)
        encoded = encode_director_transcript(
            (
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Canvas observation.\n\n{}"},
            )
        )
        self.assertIn("Canvas observation", encoded)


class HealthBenchV27CanvasAdmissionTests(unittest.TestCase):
    def test_add_domain_excludes_models_without_any_executable_profile(self) -> None:
        catalog = ModelRegistry(
            [ProviderSpec("fake", kind="test")],
            [
                ModelSpec(
                    "enabled-text-model",
                    "fake",
                    metadata={"text_capable": "true"},
                ),
                ModelSpec(
                    "disabled-model",
                    "fake",
                    metadata={"text_capable": "false"},
                ),
            ],
        )
        env = AgentWorkflowEnv(
            catalog,
            gateway=object(),
            max_agents=8,
            max_agents_per_subgraph=3,
            allowed_actions=("add_subgraph", "finish"),
        )
        env.reset("A healthcare conversation requiring a response")

        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(["enabled-text-model"], domain["model_ids"])
        self.assertEqual(
            [
                {
                    "model_id": "enabled-text-model",
                    "execution_mode": "reasoning",
                    "allowed_tools": [],
                }
            ],
            domain["model_execution_profiles"],
        )

    def test_free_subgraph_exposes_two_relations_without_fixing_topology(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            max_agents=8,
            max_agents_per_subgraph=3,
            max_relations_per_subgraph=2,
            allowed_actions=(
                "add_subgraph",
                "modify_agent",
                "delete_agent",
                "set_relation",
                "set_output",
                "finish",
            ),
        )
        env.reset("A healthcare conversation requiring a response")
        domain = env.model_admissible_action_targets()["add_subgraph"]
        self.assertEqual(0, domain["min_relations"])
        self.assertEqual(2, domain["max_relations"])
        self.assertEqual("free_contract_execution_profile", domain["declaration_mode"])

    def test_pointer_only_output_requires_output_protocol_artifact(self) -> None:
        graph = AgentGraph([AgentNode("candidate", "model", "Draft the response")])
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            require_output_protocol_artifact_for_set_output=True,
        )
        env.reset("A healthcare conversation requiring a response", graph)
        env._progressive_output_metadata["candidate"] = {
            "generated_as_output_agent": False,
        }
        self.assertEqual((), env._model_admissible_output_agent_ids())
        env._progressive_output_metadata["candidate"] = {
            "generated_as_output_agent": True,
        }
        self.assertEqual(("candidate",), env._model_admissible_output_agent_ids())

    def test_three_agent_transaction_schema_admits_two_distinct_relations(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            max_agents=8,
            max_agents_per_subgraph=3,
            max_relations_per_subgraph=2,
            allowed_actions=("add_subgraph", "finish"),
        )
        env.reset("A healthcare conversation requiring a response")
        domains = env.model_admissible_action_targets()
        selected_profiles = tuple(
            {
                "agent_id": f"node_{index}",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
            for index in range(1, 4)
        )
        declarations_text = json.dumps(
            {
                "action": "add_subgraph",
                "agents": [
                    {
                        "agent_id": item["agent_id"],
                        "model_id": "model",
                        "contract": f"Produce distinct artifact number {index} for downstream use.",
                        "execution_mode": "reasoning",
                        "allowed_tools": [],
                    }
                    for index, item in enumerate(selected_profiles, start=1)
                ],
            }
        )
        declarations = director_live_add_subgraph_agent_declarations_from_text(
            declarations_text,
            domains,
            selected_agent_profiles=selected_profiles,
        )
        schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph",
                domains,
                add_agents=declarations,
            )
        )
        relation_schema = schema["properties"]["relations"]
        Draft202012Validator.check_schema(relation_schema)
        validator = Draft202012Validator(relation_schema)
        forward = {
            "source_id": "node_1",
            "target_id": "node_2",
            "source_to_target": True,
            "target_to_source": False,
        }
        reverse_same_pair = {
            "source_id": "node_2",
            "target_id": "node_1",
            "source_to_target": True,
            "target_to_source": False,
        }
        second_pair = {
            "source_id": "node_2",
            "target_id": "node_3",
            "source_to_target": True,
            "target_to_source": False,
        }
        reciprocal = {
            "source_id": "node_1",
            "target_id": "node_2",
            "source_to_target": True,
            "target_to_source": True,
        }
        self.assertTrue(validator.is_valid([]))
        self.assertTrue(validator.is_valid([reciprocal]))
        self.assertTrue(validator.is_valid([forward, second_pair]))
        self.assertFalse(validator.is_valid([forward, reverse_same_pair]))

    def test_model_profile_joint_domain_keeps_remote_reasoning_but_local_react(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            max_agents=8,
            max_agents_per_subgraph=3,
            max_relations_per_subgraph=2,
            allowed_actions=("add_subgraph", "finish"),
        )
        env.reset("A healthcare conversation requiring a response")
        domains = env.model_admissible_action_targets()
        domain = domains["add_subgraph"]
        domain["min_new_agents"] = 1
        domain["max_new_agents"] = 1
        domain["model_ids"] = ["local-tool", "remote-text"]
        domain["execution_profiles"] = [
            {"execution_mode": "reasoning", "allowed_tools": []},
            {
                "execution_mode": "react",
                "allowed_tools": ["healthbench-authoritative.search"],
            },
        ]
        domain["model_execution_profiles"] = [
            {
                "model_id": "local-tool",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "model_id": "local-tool",
                "execution_mode": "react",
                "allowed_tools": ["healthbench-authoritative.search"],
            },
            {
                "model_id": "remote-text",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
        ]

        react_selection = [
            {
                "agent_id": "node_1",
                "execution_mode": "react",
                "allowed_tools": ["healthbench-authoritative.search"],
            }
        ]
        react_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains,
                selected_agent_profiles=react_selection,
            )
        )
        react_agent_schema = react_schema["properties"]["agents"]["oneOf"][0][
            "prefixItems"
        ][0]
        self.assertEqual(
            {"enum": ["local-tool"]},
            react_agent_schema["properties"]["model_id"],
        )

        reasoning_selection = [
            {
                "agent_id": "node_1",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            }
        ]
        reasoning_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                domains,
                selected_agent_profiles=reasoning_selection,
            )
        )
        reasoning_agent_schema = reasoning_schema["properties"]["agents"][
            "oneOf"
        ][0]["prefixItems"][0]
        self.assertEqual(
            {"enum": ["local-tool", "remote-text"]},
            reasoning_agent_schema["properties"]["model_id"],
        )

        invalid_remote_react = {
            "agent_id": "node_1",
            "model_id": "remote-text",
            "contract": "Search authoritative evidence and return one useful artifact.",
            "execution_mode": "react",
            "allowed_tools": ["healthbench-authoritative.search"],
        }
        with self.assertRaisesRegex(ValueError, "model/execution profile"):
            director_live_add_subgraph_agent_declarations_from_text(
                json.dumps(
                    {"action": "add_subgraph", "agents": [invalid_remote_react]}
                ),
                domains,
                selected_agent_profiles=react_selection,
            )

    def test_canvas_authoritative_gate_rejects_relation_count_above_domain(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            max_agents=8,
            max_agents_per_subgraph=3,
            max_relations_per_subgraph=2,
            allowed_actions=("add_subgraph", "finish"),
        )
        env.reset("A healthcare conversation requiring a response")
        action = env.parser.parse(
            json.dumps(
                {
                    "action": "add_subgraph",
                    "agents": [
                        {
                            "agent_id": f"node_{index}",
                            "model_id": "model",
                            "contract": f"Produce distinct artifact number {index} for downstream use.",
                        }
                        for index in range(1, 4)
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
                        {
                            "source_id": "node_1",
                            "target_id": "node_3",
                            "source_to_target": True,
                            "target_to_source": False,
                        },
                    ],
                }
            )
        )
        with self.assertRaisesRegex(GraphMutationError, "relation limit reached"):
            env._apply_mutation(env.graph.fork(), action)

    def test_reciprocal_final_revision_must_reach_output_lineage(self) -> None:
        graph = AgentGraph(
            [
                AgentNode("left", "model", "Produce one analysis artifact"),
                AgentNode("right", "model", "Review the peer artifact"),
                AgentNode("output", "model", "Produce the user-facing response"),
            ],
            [
                AgentRelation("left", "right", True, True),
                AgentRelation("left", "output", True, False),
            ],
            output_agent_id="output",
        )
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            require_reciprocal_terminal_artifact_lineage=True,
        )
        env.reset("A healthcare conversation requiring a response", graph)
        env._progressive_output_metadata.update(
            {
                "left": {
                    "artifact_version": "left:revision",
                    "input_artifact_versions": {"right": "right:draft"},
                },
                "right": {
                    "artifact_version": "right:revision",
                    "input_artifact_versions": {"left": "left:draft"},
                },
                "output": {
                    "artifact_version": "output:single",
                    "input_artifact_versions": {"left": "left:revision"},
                },
            }
        )
        self.assertEqual(
            ("right",),
            env._unconsumed_reciprocal_terminal_artifact_ids(),
        )
        env._progressive_output_metadata["left"] = {
            "artifact_version": "left:revision",
            "input_artifact_versions": {"right": "right:revision"},
        }
        self.assertEqual((), env._unconsumed_reciprocal_terminal_artifact_ids())

    def test_reciprocal_output_block_requires_peer_final_revision(self) -> None:
        graph = AgentGraph(
            [
                AgentNode("peer", "model", "Produce a peer artifact"),
                AgentNode("output", "model", "Produce the user-facing response"),
            ],
            [AgentRelation("peer", "output", True, True)],
            output_agent_id="output",
        )
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            require_reciprocal_terminal_artifact_lineage=True,
        )
        env.reset("A healthcare conversation requiring a response", graph)
        env._progressive_output_metadata.update(
            {
                "peer": {
                    "artifact_version": "peer:revision",
                    "input_artifact_versions": {"output": "output:draft"},
                },
                "output": {
                    "artifact_version": "output:revision",
                    "input_artifact_versions": {"peer": "peer:draft"},
                },
            }
        )
        self.assertEqual(
            ("peer",),
            env._unconsumed_reciprocal_terminal_artifact_ids(),
        )


if __name__ == "__main__":
    unittest.main()
