from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.evaluate_completion_benchmark_round import (
    validate_completion_benchmark_config,
)
from src.interactive.config_loader import ConfigurationError, load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_initial_v1.yaml"
)


def _config() -> dict[str, object]:
    return dict(load_yaml(_CONFIG_PATH))


def test_initial_condition_is_evaluation_only_free_agentgraph() -> None:
    config = _config()

    validate_completion_benchmark_config(config)

    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    graph = config["agent_graph"]
    assert bounded["split"] == "test"
    assert bounded["sample_count"] == 128
    assert config["execution_timeout"] == 600.0
    assert "task_timeout_seconds" not in bounded
    assert bounded["paired_budget_protocol"] == "task_global_equal_v1"
    assert bounded["protocol_equivalent_to_direct"] is False
    assert bounded["repository_episode_budget_equivalent"] is True
    assert runtime["tool_profile"] == "skillflow_training_v1"
    assert runtime["completion_policy"] == "workspace_diff"
    assert runtime["require_task_environment"] is True
    assert runtime["task_max_turns"] == runtime["max_turns_per_agent_call"]
    assert (
        runtime["task_max_tool_calls"]
        == runtime["max_tool_calls_per_agent_call"]
    )
    assert graph["contract_type"] == "free_text"
    assert graph["actions"] == [
        "add_subgraph",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    assert graph["require_format_agent"] is False
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["director"]["lora"]["enabled"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("task_max_turns", 13), ("task_max_tool_calls", 11)),
)
def test_paired_condition_rejects_resettable_or_unequal_budget(
    field: str,
    value: int,
) -> None:
    config = deepcopy(_config())
    config["swe_coding_runtime"][field] = value

    with pytest.raises(ConfigurationError, match=field):
        validate_completion_benchmark_config(config)


def test_strict_profile_rejects_proxy_completion_policy() -> None:
    config = deepcopy(_config())
    config["swe_coding_runtime"]["completion_policy"] = "tested_diff_receipt"

    with pytest.raises(ConfigurationError, match="workspace_diff_completion"):
        validate_completion_benchmark_config(config)


def test_strict_profile_requires_task_specific_environment() -> None:
    config = deepcopy(_config())
    config["swe_coding_runtime"]["require_task_environment"] = False

    with pytest.raises(ConfigurationError, match="require_task_environment"):
        validate_completion_benchmark_config(config)
