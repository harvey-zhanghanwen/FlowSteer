from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from src.interactive.config_loader import load_model_registry
from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION_V15,
    director_system_prompt_for_version,
)


ROOT = Path(__file__).resolve().parents[2]
V222_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_22_heldout20_scope_preserving_all_model_react.yaml"
)
V223_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_23_heldout20_reliable_retrieval_scope_guard.yaml"
)
V4_CATALOG_PATH = (
    ROOT
    / "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v4.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v223_preserves_v222_fixed20_panel_direct_condition_and_evaluator() -> None:
    previous = _load(V222_CONFIG_PATH)
    current = _load(V223_CONFIG_PATH)

    assert current["data"] == previous["data"]
    assert (
        current["healthbench_professional_evaluation"]
        == previous["healthbench_professional_evaluation"]
    )
    assert current["evaluation"] == previous["evaluation"]
    assert current["director"] == previous["director"]
    assert (
        current["experiment"]["sampling_schedule_purpose"]
        == previous["experiment"]["sampling_schedule_purpose"]
    )
    assert current["healthbench_professional_evaluation"]["sample_count"] == 20
    assert len(current["healthbench_professional_evaluation"]["task_ids"]) == 20


def test_v223_changes_only_declared_tool_runtime_fields() -> None:
    previous = _load(V222_CONFIG_PATH)
    current = _load(V223_CONFIG_PATH)

    old_tool = deepcopy(previous["healthbench_tool_runtime"])
    new_tool = deepcopy(current["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")
    web = new_tool["authoritative_web_search"]
    assert web.pop("max_retries") == 2
    assert web.pop("retry_backoff_seconds") == 1.0
    assert new_tool == old_tool
    assert current["experiment"]["tool_version"] == (
        "skillflow.medrag+ncbi-pubmed-eutils-react.retry-bounded.v4"
    )


def test_v223_uses_short_neutral_v15_prompt_and_scope_guard() -> None:
    config = _load(V223_CONFIG_PATH)
    assert config["experiment"]["prompt_version"] == DIRECTOR_PROMPT_VERSION_V15
    prompt = director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V15)
    assert "HealthBench" not in prompt
    for fixed_role in ("Doctor", "Researcher", "Verifier", "Formatter"):
        assert fixed_role not in prompt

    graph = config["agent_graph"]
    assert graph["contract_type"] == "free_text"
    assert graph["require_scope_neutral_contracts"] is True
    assert graph["require_format_agent"] is False
    assert graph["semantic_protocol_by_source"] == {
        "healthbench_professional": "none"
    }
    assert graph["model_catalog_path"] == (
        "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v4.yaml"
    )


def test_v4_catalog_keeps_four_thinking_models_with_local_only_react() -> None:
    registry = load_model_registry(V4_CATALOG_PATH)
    assert set(registry.model_ids) == {
        "qwen3.5-9b-local",
        "qwen3.5-flash",
        "deepseek-v4-flash",
        "MiniMax-M3",
    }

    for model_id in registry.model_ids:
        metadata = registry.require_model(model_id).metadata
        assert metadata["chat_template_enable_thinking"] == "true"
        assert metadata["healthbench_execution_protocol"] == (
            "contract-is-not-evidence.v2"
        )

    local = registry.require_model("qwen3.5-9b-local").metadata
    assert local["tool_capable"] == "true"
    assert local["tool_capability_scope"] == "healthbench-authoritative.search"
    for model_id in ("qwen3.5-flash", "deepseek-v4-flash", "MiniMax-M3"):
        assert registry.require_model(model_id).metadata["tool_capable"] == "false"


def test_v223_has_independent_namespace_and_no_training_or_skills() -> None:
    config = _load(V223_CONFIG_PATH)
    assert "v2_23" in config["experiment"]["condition_id"]
    for path_value in config["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_23" in path_value
    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
