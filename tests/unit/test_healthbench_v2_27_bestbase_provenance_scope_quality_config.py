from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION_V17,
    director_system_prompt_for_version,
)


ROOT = Path(__file__).resolve().parents[2]
V216 = ROOT / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_16_heldout20_authoritative_profile_choice.yaml"
V227 = ROOT / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_27_heldout20_bestbase_provenance_scope_quality.yaml"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v227_preserves_v216_panel_sampling_direct_evaluator_and_model_pool() -> None:
    best_base = _load(V216)
    current = _load(V227)

    assert current["data"] == best_base["data"]
    assert (
        current["healthbench_professional_evaluation"]
        == best_base["healthbench_professional_evaluation"]
    )
    assert current["evaluation"] == best_base["evaluation"]
    assert current["director"] == best_base["director"]
    assert current["experiment"]["seed"] == best_base["experiment"]["seed"]
    assert (
        current["experiment"]["sampling_schedule_purpose"]
        == best_base["experiment"]["sampling_schedule_purpose"]
    )
    assert (
        current["agent_graph"]["model_catalog_path"]
        == best_base["agent_graph"]["model_catalog_path"]
    )


def test_v227_changes_only_declared_public_state_admissions() -> None:
    best_base = _load(V216)
    current = _load(V227)

    old_tool = deepcopy(best_base["healthbench_tool_runtime"])
    new_tool = deepcopy(current["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")
    assert new_tool.pop("max_completion_artifact_characters") == 12_000
    assert new_tool.pop("require_relevant_evidence") is True
    assert new_tool == old_tool

    old_graph = deepcopy(best_base["agent_graph"])
    new_graph = deepcopy(current["agent_graph"])
    assert new_graph.pop("require_scope_neutral_contracts") is True
    assert new_graph.pop("artifact_quality_profile") == "public_text_quality_v1"
    assert new_graph == old_graph

    assert current["experiment"]["prompt_version"] == DIRECTOR_PROMPT_VERSION_V17
    prompt = director_system_prompt_for_version(DIRECTOR_PROMPT_VERSION_V17)
    assert "Keep unresolved names, abbreviations" in prompt
    for fixed_role in ("Doctor", "Researcher", "Verifier", "Formatter"):
        assert fixed_role not in prompt


def test_v227_keeps_free_agentgraph_and_all_training_paths_disabled() -> None:
    config = _load(V227)
    graph = config["agent_graph"]

    assert graph["contract_type"] == "free_text"
    assert graph["require_format_agent"] is False
    assert graph["semantic_protocol_by_source"] == {
        "healthbench_professional": "none"
    }
    assert graph["terminal_protocol_by_source"] == {
        "healthbench_professional": "none"
    }
    assert graph["recovery_policy"] == "preserve_diagnose_repair_augment"
    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
