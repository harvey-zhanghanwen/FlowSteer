from pathlib import Path
import json
import unittest

from jsonschema import Draft202012Validator
import yaml

from src.interactive.agent_action_parser import AgentActionParser
from src.interactive.director import (
    DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION,
    DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
    DIRECTOR_PROMPT_VERSION_V14,
    director_model_admissible_schema_branch_v3,
    director_live_action_parameter_json_schema_text,
    director_live_action_target_domains_json,
    director_live_add_subgraph_agent_declarations_from_text,
    director_live_add_subgraph_agent_declarations_json_schema_text,
    director_live_add_subgraph_execution_profile_selection_from_text,
    director_live_add_subgraph_execution_profile_selection_json_schema_text,
)
from src.interactive.rollout_collector import (
    EXECUTION_PROFILE_FIRST_ADD_DECODING_STRATEGY,
    _ADD_ACTION_CONTINUATION,
    _ADD_EXECUTION_PROFILE_DECLARATION_CONTINUATION,
    _hierarchical_continuation_prompt,
    _validate_v3_hierarchical_action_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_6.yaml"
)
CATALOG_PATH = (
    ROOT
    / "config/model_catalog_healthbench_professional_mixed_all_thinking_v1.yaml"
)
V25_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_5.yaml"
)


class HealthBenchMixedAllThinkingV26ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
        self.v25 = yaml.safe_load(V25_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_v26_changes_only_graph_condition_and_live_action_mask(self) -> None:
        condition = self.config["healthbench_professional_evaluation"]
        previous = self.v25["healthbench_professional_evaluation"]
        for key in (
            "dataset_key",
            "stage",
            "split",
            "benchmark_slice",
            "selection",
            "sample_count",
            "stable_zero_sample_count",
            "rollouts_per_task",
            "concurrency",
            "direct_model_id",
            "direct_protocol",
            "direct_contract",
            "direct_generation_seed",
            "direct_reused_from",
        ):
            self.assertEqual(previous[key], condition[key], key)
        self.assertEqual(
            "healthbench_professional_mixed_all_thinking_v2_6",
            self.config["experiment"]["condition_id"],
        )
        self.assertIn(
            "healthbench_professional_mixed_all_thinking_v2_6",
            self.config["storage"]["trajectories_path"],
        )
        self.assertEqual(
            "agentgraph.model-admissible-action-mask.v3",
            self.config["director"]["sampling_schema_version"],
        )
        self.assertEqual(
            DIRECTOR_PROMPT_VERSION_V14,
            self.config["experiment"]["prompt_version"],
        )

    def test_director_is_local_and_every_executor_arm_has_thinking(self) -> None:
        self.assertEqual("sglang", self.config["director"]["backend"])
        self.assertEqual(
            "supervisor_theta", self.config["director"]["served_model_name"]
        )
        self.assertTrue(self.config["director"]["chat_template_enable_thinking"])
        self.assertTrue(self.config["director"]["two_phase_generation"])
        self.assertEqual(
            "director_catalog_choice",
            self.config["agent_graph"]["executor_selection"],
        )
        models = self.catalog["models"]
        self.assertEqual(
            {
                "qwen3.5-9b-local",
                "qwen3.5-flash",
                "deepseek-v4-flash",
                "MiniMax-M3",
            },
            {model["model_id"] for model in models},
        )
        for model in models:
            self.assertEqual(1.0, model["selection_weight"])
            self.assertEqual(
                "true", model["metadata"]["chat_template_enable_thinking"]
            )
            self.assertGreaterEqual(int(model["metadata"]["max_tokens"]), 4096)

    def test_no_training_tools_or_skills_are_enabled(self) -> None:
        self.assertFalse(self.config["experiment"]["training_enabled"])
        self.assertFalse(self.config["healthbench_tool_runtime"]["enabled"])
        self.assertFalse(self.config["grpo"]["enabled"])
        self.assertEqual(0, self.config["grpo"]["max_optimizer_updates"])
        self.assertFalse(self.config["policy_sync"]["enabled"])
        self.assertFalse(self.config["skills"]["enabled"])


class HealthBenchFreeContractSubgraphDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domains = {
            "add_subgraph": {
                "declaration_mode": "free_contract_execution_profile",
                "contract_type": "free_text",
                "semantic_protocol": "none",
                "min_new_agents": 1,
                "max_new_agents": 3,
                "existing_agent_ids": [],
                "required_agent_fields": [
                    "agent_id",
                    "model_id",
                    "contract",
                    "execution_mode",
                    "allowed_tools",
                ],
                "model_ids": ["qwen3.5-9b-local", "deepseek-v4-flash"],
                "execution_profiles": [
                    {"execution_mode": "reasoning", "allowed_tools": []}
                ],
                "existing_agents": [],
                "required_tool_id": None,
                "min_relations": 0,
                "max_relations": 1,
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
            }
        }

    def test_profile_declaration_keeps_free_contract_and_model_choice(self) -> None:
        director_live_action_target_domains_json(
            ("add_subgraph",), self.domains
        )
        selection_schema = json.loads(
            director_live_add_subgraph_execution_profile_selection_json_schema_text(
                self.domains
            )
        )
        self.assertEqual("add_subgraph", selection_schema["properties"]["action"]["const"])
        profile_text = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","execution_mode":"reasoning","allowed_tools":[]},'
            '{"agent_id":"node_2","execution_mode":"reasoning","allowed_tools":[]}]}'
        )
        profiles = director_live_add_subgraph_execution_profile_selection_from_text(
            profile_text, self.domains
        )
        declaration_schema = json.loads(
            director_live_add_subgraph_agent_declarations_json_schema_text(
                self.domains, selected_agent_profiles=profiles
            )
        )
        schema_text = json.dumps(declaration_schema, ensure_ascii=False)
        self.assertIn("qwen3.5-9b-local", schema_text)
        self.assertIn("deepseek-v4-flash", schema_text)
        self.assertIn('"contract"', schema_text)
        self.assertNotIn("role_family", schema_text)

    def test_final_schema_excludes_pseudo_nodes_and_caps_one_relation(self) -> None:
        profiles = (
            {
                "agent_id": "node_1",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "agent_id": "node_2",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
        )
        declarations_text = json.dumps(
            {
                "action": "add_subgraph",
                "agents": [
                    {
                        "agent_id": "node_1",
                        "model_id": "deepseek-v4-flash",
                        "contract": "Analyze the supplied conversation and produce a concise draft.",
                        "execution_mode": "reasoning",
                        "allowed_tools": [],
                    },
                    {
                        "agent_id": "node_2",
                        "model_id": "qwen3.5-9b-local",
                        "contract": "Use the upstream draft to produce the requested final response.",
                        "execution_mode": "reasoning",
                        "allowed_tools": [],
                    },
                ],
            },
            separators=(",", ":"),
        )
        declarations = director_live_add_subgraph_agent_declarations_from_text(
            declarations_text,
            self.domains,
            selected_agent_profiles=profiles,
        )
        final_schema = json.loads(
            director_live_action_parameter_json_schema_text(
                "add_subgraph", self.domains, add_agents=declarations
            )
        )
        relations = final_schema["properties"]["relations"]
        relation_validator = Draft202012Validator(relations)
        forward = {
            "source_id": "node_1",
            "target_id": "node_2",
            "source_to_target": True,
            "target_to_source": False,
        }
        reverse = {
            "source_id": "node_2",
            "target_id": "node_1",
            "source_to_target": True,
            "target_to_source": False,
        }
        self.assertTrue(relation_validator.is_valid([]))
        self.assertTrue(relation_validator.is_valid([forward]))
        self.assertFalse(relation_validator.is_valid([forward, reverse]))
        output_domain = final_schema["properties"]["output_agent_id"]["anyOf"][0][
            "enum"
        ]
        self.assertEqual(["node_1", "node_2"], output_domain)
        relation_schema_text = json.dumps(relations, sort_keys=True)
        for pseudo_node in (
            "input_data",
            "task",
            "user_message",
            "system_prompt",
            "output_agent",
        ):
            self.assertNotIn(pseudo_node, relation_schema_text)

    def test_profile_first_receipt_is_bound_to_selection_and_declaration(self) -> None:
        actions = ("add_subgraph",)
        domains_json = director_live_action_target_domains_json(
            actions, self.domains
        )
        base_prompt = "current Canvas"
        profile_text = (
            '{"action":"add_subgraph","agents":['
            '{"agent_id":"node_1","execution_mode":"reasoning","allowed_tools":[]},'
            '{"agent_id":"node_2","execution_mode":"reasoning","allowed_tools":[]}]}'
        )
        profiles = director_live_add_subgraph_execution_profile_selection_from_text(
            profile_text, self.domains
        )
        profile_json = json.dumps(
            {"action": "add_subgraph", "agents": list(profiles)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        declaration_prompt = _hierarchical_continuation_prompt(
            base_prompt,
            committed_json=profile_json,
            instruction=_ADD_EXECUTION_PROFILE_DECLARATION_CONTINUATION,
        )
        declarations = [
            {
                "agent_id": "node_1",
                "model_id": "deepseek-v4-flash",
                "contract": "Analyze the supplied conversation and produce a concise draft.",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
            {
                "agent_id": "node_2",
                "model_id": "qwen3.5-9b-local",
                "contract": "Use the upstream draft to produce the requested final response.",
                "execution_mode": "reasoning",
                "allowed_tools": [],
            },
        ]
        declaration_text = json.dumps(
            {"action": "add_subgraph", "agents": declarations},
            separators=(",", ":"),
        )
        declaration_json = json.dumps(
            {"action": "add_subgraph", "agents": declarations},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        parameter_prompt = _hierarchical_continuation_prompt(
            declaration_prompt,
            committed_json=declaration_json,
            instruction=_ADD_ACTION_CONTINUATION,
        )
        final_text = json.dumps(
            {
                "action": "add_subgraph",
                "agents": declarations,
                "relations": [
                    {
                        "source_id": "node_1",
                        "target_id": "node_2",
                        "source_to_target": True,
                        "target_to_source": False,
                    }
                ],
                "output_agent_id": "node_2",
            },
            separators=(",", ":"),
        )
        metadata = {
            "selected_action": "add_subgraph",
            "action_decoding_strategy": (
                EXECUTION_PROFILE_FIRST_ADD_DECODING_STRATEGY
            ),
            "parse_failure_phase": None,
            "hierarchical_phase_receipts": {
                "add_agent_execution_profile_selection": {
                    "text": profile_text,
                    "prompt_text": base_prompt,
                    "generation_seed": 17,
                },
                "add_agent_declarations": {
                    "text": declaration_text,
                    "prompt_text": declaration_prompt,
                    "generation_seed": 17,
                },
            },
            "selected_add_agent_ids": ["node_1", "node_2"],
            "selected_add_agent_roles": None,
            "selected_add_agent_profiles": list(profiles),
            "selected_modify_agent_id": None,
            "parameter_schema_branch": "add_subgraph",
            "prompt_text": parameter_prompt,
            "request_count": 3,
        }
        schema_request = {
            "action_json_schema_version": (
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            ),
            "action_schema_branch": director_model_admissible_schema_branch_v3(
                actions
            ),
            "action_target_domains_json": domains_json,
            "action_target_domain_version": (
                DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
            ),
        }
        self.assertEqual(
            {
                "add_agent_execution_profile_selection",
                "add_agent_declarations",
            },
            _validate_v3_hierarchical_action_receipt(
                AgentActionParser().parse(final_text),
                metadata,
                schema_request,
            ),
        )


if __name__ == "__main__":
    unittest.main()
