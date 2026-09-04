from __future__ import annotations

from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_BASE_CONFIG = (
    _ROOT
    / "config"
    / "evaluation_healthbench_professional_authoritative_multiagent_feedback_v3_gpt54_rubric.yaml"
)
_THINKING_CONFIG = (
    _ROOT
    / "config"
    / "evaluation_healthbench_professional_authoritative_multiagent_feedback_v4_thinking_gpt54_rubric.yaml"
)


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_healthbench_v4_enables_thinking_for_direct_agent_and_director() -> None:
    config = _yaml(_THINKING_CONFIG)
    catalog_path = _ROOT / config["agent_graph"]["model_catalog_path"]
    catalog = _yaml(catalog_path)
    direct_model_id = config["healthbench_professional_evaluation"][
        "direct_model_id"
    ]
    direct_and_executor_model = next(
        model
        for model in catalog["models"]
        if model["model_id"] == direct_model_id
    )

    assert direct_and_executor_model["model_id"] == "qwen3.5-9b-local"
    assert direct_and_executor_model["provider_id"] == (
        "local-qwen35-healthbench-authoritative-thinking-v2"
    )
    assert direct_and_executor_model["metadata"][
        "chat_template_enable_thinking"
    ] == "true"
    assert "thinking=enabled" in direct_and_executor_model["metadata"]["profile"]
    assert "version=v2" in direct_and_executor_model["metadata"]["profile"]
    assert all(
        model["metadata"]["chat_template_enable_thinking"] == "true"
        for model in catalog["models"]
    )
    assert config["director"]["chat_template_enable_thinking"] is True
    assert config["director"]["max_action_tokens"] == 4096
    assert config["director"]["action_decoding"] == "json_schema"


def test_healthbench_v4_keeps_react_as_execution_mode_and_grader_unchanged() -> None:
    base = _yaml(_BASE_CONFIG)
    config = _yaml(_THINKING_CONFIG)
    bounded = config["healthbench_professional_evaluation"]

    assert bounded["direct_execution_mode"] == "react"
    assert config["agent_graph"]["contract_type"] == "free_text"
    assert "role" not in config["agent_graph"]
    assert "roles" not in config["agent_graph"]
    assert config["evaluation"] == base["evaluation"]
    assert config["evaluation"]["healthbench_judge_catalog_path"] == (
        "config/model_catalog_healthbench_professional_grader_gpt54_alias_v1.yaml"
    )


def test_healthbench_v4_preserves_the_frozen_evaluation_condition() -> None:
    base = _yaml(_BASE_CONFIG)
    config = _yaml(_THINKING_CONFIG)
    base_bounded = base["healthbench_professional_evaluation"]
    bounded = config["healthbench_professional_evaluation"]

    assert config["experiment"]["seed"] == base["experiment"]["seed"]
    assert config["data"] == base["data"]
    for key in (
        "split",
        "benchmark_slice",
        "selection",
        "sample_count",
        "stable_zero_sample_count",
        "rollouts_per_task",
        "concurrency",
        "direct_generation_seed",
    ):
        assert bounded[key] == base_bounded[key]
    assert bounded["sample_count"] == 525
    assert bounded["stable_zero_sample_count"] == 2
    assert bounded["concurrency"] == 4
    assert config["experiment"]["tool_version"] == base["experiment"][
        "tool_version"
    ]
    base_tool = dict(base["healthbench_tool_runtime"])
    thinking_tool = dict(config["healthbench_tool_runtime"])
    base_tool.pop("condition_id")
    thinking_tool.pop("condition_id")
    assert thinking_tool == base_tool


def test_healthbench_v4_uses_independent_condition_artifact_and_policy_namespaces() -> None:
    base = _yaml(_BASE_CONFIG)
    config = _yaml(_THINKING_CONFIG)
    condition = config["experiment"]["condition_id"]

    assert condition == (
        "healthbench_professional_authoritative_multiagent_feedback_"
        "v4_thinking_gpt54_rubric"
    )
    assert condition != base["experiment"]["condition_id"]
    assert config["experiment"]["name"] == condition
    assert config["experiment"]["catalog_order_namespace"] == condition
    assert config["healthbench_tool_runtime"]["condition_id"] == condition
    assert config["experiment"]["output_dir"].startswith(
        f"artifacts/{condition}/"
    )
    assert config["experiment"]["output_dir"] != base["experiment"][
        "output_dir"
    ]
    for key, value in config["storage"].items():
        if key == "schema_version":
            continue
        assert condition in value
        assert value != base["storage"][key]
    assert condition in config["director"]["behavior_policy_version"]
    assert condition in config["policy_sync"]["adapter_name_prefix"]
    assert config["agent_graph"]["model_catalog_path"] != base[
        "agent_graph"
    ]["model_catalog_path"]


def test_healthbench_v4_remains_evaluation_only_without_skills() -> None:
    config = _yaml(_THINKING_CONFIG)

    assert config["experiment"]["training_enabled"] is False
    assert config["gpu"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["optimization_passes_per_rollout_batch"] == 0
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["skills"]["initial_library"] == []
