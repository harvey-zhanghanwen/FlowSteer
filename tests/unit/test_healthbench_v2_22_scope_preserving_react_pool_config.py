from copy import deepcopy
from pathlib import Path

import yaml

from src.interactive.config_loader import load_model_registry
from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION_V17,
    director_system_prompt_for_version,
)


ROOT = Path(__file__).resolve().parents[2]
V221_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_21_heldout20_material_fidelity_relation_semantics.yaml"
)
V222_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_22_heldout20_scope_preserving_all_model_react.yaml"
)
V3_CATALOG_PATH = (
    ROOT
    / "config/model_catalog_healthbench_professional_mixed_authoritative_thinking_v3.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v222_preserves_fixed_panel_tool_evaluator_and_sampling_schedule() -> None:
    previous = _load(V221_CONFIG_PATH)
    current = _load(V222_CONFIG_PATH)

    assert current["data"] == previous["data"]
    assert (
        current["healthbench_professional_evaluation"]
        == previous["healthbench_professional_evaluation"]
    )
    assert current["evaluation"] == previous["evaluation"]
    old_tool = deepcopy(previous["healthbench_tool_runtime"])
    new_tool = deepcopy(current["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")
    assert new_tool == old_tool
    assert (
        current["experiment"]["sampling_schedule_purpose"]
        == previous["experiment"]["sampling_schedule_purpose"]
    )


def test_v222_prompt_is_short_role_neutral_and_scope_preserving() -> None:
    config = _load(V222_CONFIG_PATH)
    assert config["experiment"]["prompt_version"] == DIRECTOR_PROMPT_VERSION_V17
    prompt = director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V17)
    assert "unresolved names, abbreviations, quantities" in prompt
    assert "requested answer slot" in prompt
    assert "observed artifact supplies the meaning" in prompt
    assert "directed producer-to-consumer relation" not in prompt
    for fixed_role in ("Doctor", "Researcher", "Verifier", "Formatter"):
        assert fixed_role not in prompt
    assert "HealthBench" not in prompt


def test_v3_catalog_admits_every_measured_model_to_authoritative_react() -> None:
    registry = load_model_registry(V3_CATALOG_PATH)
    assert set(registry.model_ids) == {
        "qwen3.5-9b-local",
        "qwen3.5-flash",
        "deepseek-v4-flash",
        "MiniMax-M3",
    }
    for model_id in registry.model_ids:
        metadata = registry.require_model(model_id).metadata
        assert metadata["chat_template_enable_thinking"] == "true"
        assert metadata["tool_capable"] == "true"
        assert metadata["tool_capability_scope"] == (
            "healthbench-authoritative.search"
        )
        if model_id != "qwen3.5-9b-local":
            assert metadata["react_capability_canary"] == (
                "passed_structured_action_2026-09-01"
            )
            assert metadata["react_canary_source"].endswith(
                "healthbench_mixed_react_thinking_v2_20260901.json"
            )


def test_v222_has_independent_namespace_and_no_training_or_skills() -> None:
    config = _load(V222_CONFIG_PATH)
    assert "v2_22" in config["experiment"]["condition_id"]
    for path_value in config["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_22" in path_value
    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
