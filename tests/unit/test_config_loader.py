from __future__ import annotations

import unittest

from src.interactive.config_loader import (
    ConfigurationError,
    expand_environment,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)


class ConfigLoaderTests(unittest.TestCase):
    def test_environment_defaults_and_required_variables(self) -> None:
        value = {"a": "${SET}", "b": "x-${MISSING:-fallback}", "c": ["${SET}"]}
        self.assertEqual(
            expand_environment(value, {"SET": "value"}),
            {"a": "value", "b": "x-fallback", "c": ["value"]},
        )
        with self.assertRaises(ConfigurationError):
            expand_environment("${REQUIRED}", {})

    def test_checked_in_agent_graph_config_obeys_strict_invariants(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        validate_agent_graph_config(config)
        self.assertEqual(config["director"]["base_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(config["director"]["backend"], "sglang")
        self.assertEqual(config["gpu"]["learner_physical"], 3)
        self.assertEqual(config["gpu"]["rollout_physical"], 0)
        self.assertEqual(config["gpu"]["gradient_replica_physical"], 5)
        self.assertTrue(config["director"]["execute_on_edit"])
        self.assertEqual(config["director"]["history_window"], 4)
        self.assertEqual(config["agent_graph"]["max_agents"], 10)
        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["grpo"]["enabled"])

    def test_nonzero_exploration_reward_is_rejected(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["grpo"]["exploration_reward"] = 0.1
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_architecture_phase_cannot_enable_training(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["gpu"]["training_enabled"] = True
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_declared_canvas_search_space_is_enforced(self) -> None:
        invalid_values = (
            ("max_agents", 0),
            ("executor_selection", "unseeded"),
            ("max_bidirectional_block_size", 3),
            ("require_unique_output", False),
            ("require_all_agents_reach_output", False),
            ("require_format_agent", "false"),
        )
        for field, value in invalid_values:
            with self.subTest(field=field):
                config = load_yaml("config/training_agent_graph.yaml")
                config["agent_graph"][field] = value
                with self.assertRaises(ConfigurationError):
                    validate_agent_graph_config(config)

        for require_format_agent in (False, True):
            with self.subTest(require_format_agent=require_format_agent):
                config = load_yaml("config/training_agent_graph.yaml")
                config["agent_graph"]["require_format_agent"] = require_format_agent
                validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["execute_on_edit"] = False
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["history_window"] = 0
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["base_model"] = "Qwen/Qwen3-8B"
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

        config = load_yaml("config/training_agent_graph.yaml")
        config["director"]["api_base"] = "https://provider.example/v1"
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_flowsteer_add_subgraph_action_profile_is_strict(self) -> None:
        subgraph_actions = [
            "add_subgraph",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "finish",
        ]
        config = load_yaml("config/training_agent_graph.yaml")
        config["agent_graph"]["actions"] = subgraph_actions
        config["agent_graph"]["max_agents_per_subgraph"] = 3
        validate_agent_graph_config(config)

        for invalid_limit in (None, 0, 2, 4, True):
            with self.subTest(invalid_limit=invalid_limit):
                invalid = load_yaml("config/training_agent_graph.yaml")
                invalid["agent_graph"]["actions"] = subgraph_actions
                if invalid_limit is None:
                    invalid["agent_graph"].pop("max_agents_per_subgraph", None)
                else:
                    invalid["agent_graph"]["max_agents_per_subgraph"] = invalid_limit
                with self.assertRaises(ConfigurationError):
                    validate_agent_graph_config(invalid)

        mixed = load_yaml("config/training_agent_graph.yaml")
        mixed["agent_graph"]["actions"] = [
            "add_subgraph",
            "add_agent",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "finish",
        ]
        mixed["agent_graph"]["max_agents_per_subgraph"] = 3
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(mixed)

    def test_stepwise_continue_requires_explicit_environment_runtime(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["agent_graph"]["actions"] = [
            "add_agent",
            "modify_agent",
            "delete_agent",
            "set_relation",
            "set_output",
            "continue",
            "finish",
        ]
        with self.assertRaisesRegex(
            ConfigurationError,
            "continue action requires an enabled stepwise environment runtime",
        ):
            validate_agent_graph_config(config)

        config["environment_runtime"] = {
            "enabled": True,
            "stepwise_director": True,
        }
        validate_agent_graph_config(config)

        config["environment_runtime"]["stepwise_director"] = False
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_smoke_config_is_strictly_bounded(self) -> None:
        config = load_yaml("config/training_agentgraph_smoke.yaml")
        validate_agent_graph_config(config)

        self.assertEqual(config["experiment"]["phase"], "smoke_training")
        self.assertTrue(config["experiment"]["training_enabled"])
        self.assertEqual(config["data"]["smoke"]["tasks_per_dataset"], 2)
        self.assertEqual(config["data"]["smoke"]["expected_total_tasks"], 14)
        self.assertEqual(len(config["data"]["smoke"]["source_order"]), 7)
        self.assertEqual(config["grpo"]["samples_per_problem"], 2)
        self.assertEqual(config["grpo"]["expected_rollout_count"], 28)
        self.assertEqual(config["grpo"]["max_optimizer_updates"], 1)
        self.assertEqual(config["gpu"]["oom_policy"]["micro_batch_schedule"], [4, 2, 1])
        self.assertTrue(config["policy_sync"]["enabled"])
        self.assertEqual(config["policy_sync"]["adapter_name_prefix"], "theta_smoke_step_")
        self.assertEqual(config["policy_sync"]["post_update_canary_count"], 1)
        self.assertEqual(
            config["policy_sync"]["canary"]["adapter_selector_field"],
            "lora_path",
        )
        self.assertFalse(config["exploration"]["enabled"])
        self.assertFalse(config["skills"]["enabled"])

    def test_example_catalog_contains_only_verified_smoke_ids(self) -> None:
        registry = load_model_registry("config/model_catalog.yaml.example")
        expected_names = {
            "supervisor_theta",
            "qwen3.5-flash",
            "deepseek-v4-flash",
            "gpt-4o-mini",
            "grok-4-1-fast-non-reasoning",
            "MiniMax-M2.5",
        }
        actual_names = {
            registry.require_model(model_id).model_name for model_id in registry.model_ids
        }
        self.assertEqual(actual_names, expected_names)
        self.assertFalse(any("gemini" in model_id.lower() for model_id in registry.model_ids))

    def test_hotpot_multiagent_catalog_uses_only_canary_backed_exact_ids(self) -> None:
        registry = load_model_registry(
            "config/model_catalog_hotpotqa_multiagent_v1.yaml"
        )
        self.assertEqual(
            {
                "qwen3.5-9b-local",
                "qwen3.5-flash",
                "qwen3.5-plus",
                "deepseek-v4-flash",
                "deepseek-v4-pro",
                "gpt-4o-mini",
                "minimax-m2.5",
                "minimax-m3",
            },
            set(registry.model_ids),
        )
        self.assertFalse(any("gemini" in model_id.lower() for model_id in registry.model_ids))
        self.assertFalse(any("grok" in model_id.lower() for model_id in registry.model_ids))

    def test_terminal_protocol_map_is_fail_closed(self) -> None:
        config = load_yaml("config/training_agent_graph.yaml")
        config["agent_graph"]["terminal_protocol_by_source"] = {
            "hotpotqa": "exact_single_answer_tag"
        }
        validate_agent_graph_config(config)
        config["agent_graph"]["terminal_protocol_by_source"] = {
            "hotpotqa": "prompt_only"
        }
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)

    def test_hotpot_unified_semantic_protocol_is_explicit_and_inference_only(self) -> None:
        config = load_yaml("config/evaluation_hotpotqa_unified_architecture_v1.yaml")
        validate_agent_graph_config(config)

        self.assertEqual(
            config["experiment"]["prompt_version"],
            "agentgraph.director.hotpotqa-semantic-recovery.v22",
        )
        self.assertEqual(
            config["experiment"]["tool_version"],
            "skillflow.qa-retrieval.multihop-semantic-repair.v16",
        )
        self.assertEqual(config["director"]["action_decoding"], "json_schema")
        self.assertEqual(
            config["director"]["sampling_action_profile"],
            "model_admissible_canvas_actions",
        )
        self.assertEqual(
            config["director"]["sampling_schema_version"],
            "agentgraph.model-admissible-action-mask.v3",
        )
        self.assertEqual(config["qa_tool_runtime"]["max_turns_per_agent_call"], 9)
        self.assertEqual(config["qa_tool_runtime"]["max_tool_calls_per_agent_call"], 6)
        self.assertEqual(
            config["agent_graph"]["semantic_protocol_by_source"],
            {"hotpotqa": "hotpotqa_verified_answer_slot_v1"},
        )
        self.assertEqual(
            config["agent_graph"]["recovery_policy"],
            "preserve_diagnose_repair_augment",
        )
        self.assertEqual(
            config["agent_graph"]["required_evidence_tool_id"],
            "qa-retrieval",
        )
        self.assertEqual(config["hotpotqa_evaluation"]["sample_count"], 128)
        self.assertEqual(
            config["hotpotqa_evaluation"]["required_partition"],
            "development",
        )
        self.assertEqual(config["gpu"]["rollout_physical"], 0)
        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["grpo"]["enabled"])

    def test_hotpot_semantic_protocol_combination_is_fail_closed(self) -> None:
        mutations = (
            ("prompt_version", "agentgraph.director.minimal-neutral.v10"),
            ("recovery_policy", "default"),
            ("required_evidence_tool_id", "other-tool"),
            ("terminal_protocol", "none"),
            ("qa_completion_policy", "optional"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                config = load_yaml(
                    "config/evaluation_hotpotqa_unified_architecture_v1.yaml"
                )
                if field == "prompt_version":
                    config["experiment"][field] = value
                elif field == "terminal_protocol":
                    config["agent_graph"]["terminal_protocol_by_source"][
                        "hotpotqa"
                    ] = value
                elif field == "qa_completion_policy":
                    config["qa_tool_runtime"]["completion_policy"] = value
                else:
                    config["agent_graph"][field] = value
                with self.assertRaises(ConfigurationError):
                    validate_agent_graph_config(config)

        wrong_dataset = load_yaml(
            "config/evaluation_hotpotqa_unified_architecture_v1.yaml"
        )
        wrong_dataset["agent_graph"]["semantic_protocol_by_source"] = {
            "triviaqa": "hotpotqa_verified_answer_slot_v1"
        }
        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(wrong_dataset)

    def test_trivia_unified_v2_uses_shared_semantic_recovery_contract(self) -> None:
        config = load_yaml("config/evaluation_triviaqa_unified_architecture_v2.yaml")
        validate_agent_graph_config(config)

        self.assertEqual(
            config["experiment"]["prompt_version"],
            "agentgraph.director.qa-semantic-recovery.v1",
        )
        self.assertEqual(
            config["agent_graph"]["semantic_protocol_by_source"],
            {"triviaqa": "qa_verified_answer_lineage_v2"},
        )
        self.assertEqual(
            config["agent_graph"]["recovery_policy"],
            "preserve_diagnose_repair_augment",
        )
        self.assertEqual(
            config["director"]["sampling_schema_version"],
            "agentgraph.model-admissible-action-mask.v3",
        )
        self.assertEqual(config["triviaqa_evaluation"]["sample_count"], 128)
        self.assertEqual(config["gpu"]["rollout_physical"], 0)
        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["grpo"]["enabled"])
        self.assertFalse(config["skills"]["enabled"])

    def test_shared_qa_semantic_protocol_combination_is_fail_closed(self) -> None:
        mutations = (
            ("prompt_version", "agentgraph.director.minimal-neutral.v10"),
            ("recovery_policy", "default"),
            ("required_evidence_tool_id", "other-tool"),
            ("terminal_protocol", "none"),
            ("qa_completion_policy", "optional"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                config = load_yaml(
                    "config/evaluation_triviaqa_unified_architecture_v2.yaml"
                )
                if field == "prompt_version":
                    config["experiment"][field] = value
                elif field == "terminal_protocol":
                    config["agent_graph"]["terminal_protocol_by_source"][
                        "triviaqa"
                    ] = value
                elif field == "qa_completion_policy":
                    config["qa_tool_runtime"]["completion_policy"] = value
                else:
                    config["agent_graph"][field] = value
                with self.assertRaises(ConfigurationError):
                    validate_agent_graph_config(config)

    def test_alfworld_stepwise_feedback_v4_uses_live_action_domains(self) -> None:
        config = load_yaml(
            "config/evaluation_alfworld_stepwise_feedback_v4.yaml"
        )
        validate_agent_graph_config(config)

        self.assertEqual(
            "agentgraph.model-admissible-action-mask.v3",
            config["director"]["sampling_schema_version"],
        )
        self.assertTrue(config["environment_runtime"]["stepwise_director"])
        self.assertEqual(
            20,
            config["environment_runtime"]["max_environment_steps_by_source"][
                "alfworld"
            ],
        )
        self.assertEqual(32768, config["director"]["max_context_tokens"])
        self.assertEqual(0, config["gpu"]["rollout_physical"])
        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["grpo"]["enabled"])
        self.assertFalse(config["skills"]["enabled"])

    def test_alfworld_stepwise_recovery_v5_preserves_paired_protocol(self) -> None:
        baseline = load_yaml(
            "config/evaluation_alfworld_stepwise_feedback_v4.yaml"
        )
        config = load_yaml(
            "config/evaluation_alfworld_stepwise_recovery_v5.yaml"
        )
        validate_agent_graph_config(config)

        self.assertEqual(
            "alfworld_stepwise_recovery_v5",
            config["experiment"]["condition_id"],
        )
        self.assertEqual(
            "skillflow.alfworld.native-stepwise-recovery.v5",
            config["environment_runtime"]["tool_version"],
        )
        self.assertEqual(
            "flowsteer.alfworld.stepwise-recovery.v5",
            config["storage"]["schema_version"],
        )

        # The v5 comparison changes only the adapter/recovery condition. The
        # formal split, pairing, budget, seed, and orchestration bounds remain
        # identical to the completed v4 condition.
        for field in (
            "dataset_key",
            "stage",
            "split",
            "official_split",
            "selection",
            "sample_count",
            "stable_zero_sample_count",
            "rollouts_per_task",
            "concurrency",
            "task_timeout_seconds",
            "direct_model_id",
            "direct_protocol",
            "direct_contract",
            "direct_generation_seed",
        ):
            with self.subTest(section="alfworld_evaluation", field=field):
                self.assertEqual(
                    baseline["alfworld_evaluation"][field],
                    config["alfworld_evaluation"][field],
                )
        self.assertEqual(
            baseline["experiment"]["seed"],
            config["experiment"]["seed"],
        )
        self.assertEqual(140, config["alfworld_evaluation"]["sample_count"])
        self.assertEqual(
            "valid_seen",
            config["alfworld_evaluation"]["official_split"],
        )
        self.assertEqual("sequential", config["alfworld_evaluation"]["selection"])
        self.assertFalse(
            config["alfworld_evaluation"]["protocol_equivalent_to_direct"]
        )
        self.assertEqual(1, config["alfworld_evaluation"]["concurrency"])
        self.assertEqual(20, config["evaluation"]["max_environment_steps"])
        self.assertEqual(
            20,
            config["environment_runtime"]["max_environment_steps_by_source"][
                "alfworld"
            ],
        )
        self.assertEqual(32, config["director"]["max_rounds"])
        self.assertEqual(3, config["gpu"]["rollout_physical"])
        self.assertEqual(3, config["gpu"]["supervisor_gpu_id"])
        self.assertEqual(
            "http://127.0.0.1:8023/v1",
            config["gpu"]["supervisor_api_base"],
        )
        self.assertEqual(
            baseline["agent_graph"]["model_catalog_path"],
            config["agent_graph"]["model_catalog_path"],
        )

        expected_artifact_prefix = (
            "artifacts/alfworld_stepwise_recovery_v5/valid_seen"
        )
        self.assertEqual(
            expected_artifact_prefix,
            config["experiment"]["output_dir"],
        )
        for field, value in config["storage"].items():
            if not field.endswith("_path") and field != "root":
                continue
            with self.subTest(section="storage", field=field):
                expected_prefix = (
                    "reports/alfworld_stepwise_recovery_v5"
                    if field.startswith("report_")
                    else expected_artifact_prefix
                )
                self.assertTrue(value.startswith(expected_prefix), value)
                self.assertNotIn("alfworld_stepwise_feedback_v4", value)

        self.assertFalse(config["experiment"]["training_enabled"])
        self.assertFalse(config["director"]["lora"]["enabled"])
        self.assertFalse(config["grpo"]["enabled"])
        self.assertEqual(
            0,
            config["grpo"]["optimization_passes_per_rollout_batch"],
        )
        self.assertEqual(0, config["grpo"]["max_optimizer_updates"])
        self.assertEqual(0.0, config["grpo"]["learning_rate"])
        self.assertFalse(config["policy_sync"]["enabled"])
        self.assertFalse(config["exploration"]["enabled"])
        self.assertFalse(config["skills"]["enabled"])
        self.assertEqual([], config["skills"]["initial_library"])
        self.assertEqual(0, config["skills"]["retrieval_top_k"])
        self.assertFalse(config["gpu"]["training_enabled"])


if __name__ == "__main__":
    unittest.main()
