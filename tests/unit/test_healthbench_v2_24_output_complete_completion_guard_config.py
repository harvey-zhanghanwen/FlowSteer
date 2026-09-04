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
V223_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_23_heldout20_reliable_retrieval_scope_guard.yaml"
)
V224_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_24_heldout20_output_complete_completion_guard.yaml"
)
V4_CATALOG_PATH = (
    ROOT
    / "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v4.yaml"
)
V5_CATALOG_PATH = (
    ROOT
    / "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v5.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v224_preserves_v223_fixed20_panel_direct_evaluator_and_sampling() -> None:
    previous = _load(V223_CONFIG_PATH)
    current = _load(V224_CONFIG_PATH)

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
    assert current["experiment"]["prompt_version"] == DIRECTOR_PROMPT_VERSION_V15
    assert current["healthbench_professional_evaluation"]["sample_count"] == 20
    assert len(current["healthbench_professional_evaluation"]["task_ids"]) == 20


def test_v224_changes_only_declared_runtime_completion_guard() -> None:
    previous = _load(V223_CONFIG_PATH)
    current = _load(V224_CONFIG_PATH)

    old_tool = deepcopy(previous["healthbench_tool_runtime"])
    new_tool = deepcopy(current["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")
    assert new_tool.pop("max_completion_artifact_characters") == 12000
    assert new_tool == old_tool
    assert current["experiment"]["tool_version"] == (
        "skillflow.medrag+ncbi-pubmed-eutils-react.retry-bounded+completion-artifact-guard.v5"
    )


def test_v224_keeps_neutral_free_graph_and_only_updates_catalog_pointer() -> None:
    previous = _load(V223_CONFIG_PATH)
    current = _load(V224_CONFIG_PATH)

    old_graph = deepcopy(previous["agent_graph"])
    new_graph = deepcopy(current["agent_graph"])
    old_graph.pop("model_catalog_path")
    new_graph.pop("model_catalog_path")
    assert new_graph == old_graph
    assert current["agent_graph"]["model_catalog_path"] == (
        "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v5.yaml"
    )
    assert current["agent_graph"]["contract_type"] == "free_text"
    assert current["agent_graph"]["require_scope_neutral_contracts"] is True
    assert current["agent_graph"]["require_format_agent"] is False
    prompt = director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V15)
    assert "HealthBench" not in prompt
    for fixed_role in ("Doctor", "Researcher", "Verifier", "Formatter"):
        assert fixed_role not in prompt


def test_v5_catalog_only_updates_output_complete_execution_protocol() -> None:
    old_raw = _load(V4_CATALOG_PATH)
    new_raw = _load(V5_CATALOG_PATH)
    for catalog in (old_raw, new_raw):
        for model in catalog["models"]:
            model["metadata"].pop("healthbench_execution_protocol")
    assert new_raw == old_raw

    registry = load_model_registry(V5_CATALOG_PATH)
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
            "contract-is-not-evidence.output-complete.v3"
        )
    local = registry.require_model("qwen3.5-9b-local").metadata
    assert local["tool_capable"] == "true"
    assert local["tool_capability_scope"] == "healthbench-authoritative.search"
    for model_id in ("qwen3.5-flash", "deepseek-v4-flash", "MiniMax-M3"):
        assert registry.require_model(model_id).metadata["tool_capable"] == "false"


def test_v224_has_independent_namespace_and_no_training_or_skills() -> None:
    config = _load(V224_CONFIG_PATH)
    assert "v2_24" in config["experiment"]["condition_id"]
    assert "v2_23" not in config["experiment"]["condition_id"]
    assert "v2_24" in config["experiment"]["output_dir"]
    assert "v2_23" not in config["experiment"]["output_dir"]
    for path_value in config["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_24" in path_value
            assert "v2_23" not in path_value

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
