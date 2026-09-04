from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from src.interactive.director import DIRECTOR_PROMPT_VERSION_V15


ROOT = Path(__file__).resolve().parents[2]
V225_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_25_heldout20_query_budget_slot_binding.yaml"
)
V226_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_26_heldout20_bounded_react_repair_horizon.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _without_version_namespace_and_react_horizon(config: dict) -> dict:
    normalized = deepcopy(config)
    experiment = normalized["experiment"]
    for key in ("name", "condition_id", "tool_version", "output_dir"):
        experiment.pop(key)
    normalized["healthbench_tool_runtime"].pop("condition_id")
    normalized["healthbench_tool_runtime"].pop("max_turns_per_agent_call")
    normalized.pop("storage")
    normalized["policy_sync"].pop("adapter_name_prefix")
    return normalized


def test_v226_only_changes_version_namespace_and_bounded_react_horizon() -> None:
    previous = _load(V225_CONFIG_PATH)
    current = _load(V226_CONFIG_PATH)

    assert previous["healthbench_tool_runtime"]["max_turns_per_agent_call"] == 6
    assert current["healthbench_tool_runtime"]["max_turns_per_agent_call"] == 8
    assert _without_version_namespace_and_react_horizon(current) == (
        _without_version_namespace_and_react_horizon(previous)
    )
    assert current["experiment"]["tool_version"] == (
        "skillflow.medrag+ncbi-pubmed-eutils-react.query-budget6+retry-bounded+"
        "completion-artifact-guard+repair-horizon8.v7"
    )


def test_v226_preserves_fixed20_model_protocol_tools_and_grader() -> None:
    previous = _load(V225_CONFIG_PATH)
    current = _load(V226_CONFIG_PATH)

    assert current["data"] == previous["data"]
    assert (
        current["healthbench_professional_evaluation"]
        == previous["healthbench_professional_evaluation"]
    )
    assert current["evaluation"] == previous["evaluation"]
    assert current["director"] == previous["director"]
    assert current["agent_graph"] == previous["agent_graph"]
    assert current["experiment"]["seed"] == previous["experiment"]["seed"]
    assert current["experiment"]["prompt_version"] == DIRECTOR_PROMPT_VERSION_V15
    assert (
        current["experiment"]["sampling_schedule_purpose"]
        == previous["experiment"]["sampling_schedule_purpose"]
    )
    fixed_panel = current["healthbench_professional_evaluation"]
    assert fixed_panel["selection"] == "task_ids"
    assert fixed_panel["sample_count"] == 20
    assert len(fixed_panel["task_ids"]) == 20
    assert current["agent_graph"]["model_catalog_path"].endswith(
        "model_catalog_healthbench_professional_mixed_authoritative_thinking_v6.yaml"
    )
    runtime = current["healthbench_tool_runtime"]
    assert runtime["max_query_content_tokens"] == 6
    assert runtime["max_completion_artifact_characters"] == 12000
    assert runtime["authoritative_web_search"]["max_retries"] == 2
    assert runtime["authoritative_web_search"]["retry_backoff_seconds"] == 1.0


def test_v226_has_independent_paths_and_keeps_training_disabled() -> None:
    config = _load(V226_CONFIG_PATH)
    condition = config["experiment"]["condition_id"]
    assert "v2_26" in condition
    assert "v2_25" not in condition
    assert "v2_26" in config["experiment"]["output_dir"]
    assert "v2_25" not in config["experiment"]["output_dir"]
    assert "v2_26" in config["healthbench_tool_runtime"]["condition_id"]
    assert "v2_26" in config["policy_sync"]["adapter_name_prefix"]
    for path_value in config["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_26" in path_value
            assert "v2_25" not in path_value

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
