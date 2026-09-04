from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V215_NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_15_heldout20_lineage_admission"
)
V216_NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_16_heldout20_authoritative_profile_choice"
)
V215_CONFIG_PATH = ROOT / "config" / f"evaluation_{V215_NAMESPACE}.yaml"
V216_CONFIG_PATH = ROOT / "config" / f"evaluation_{V216_NAMESPACE}.yaml"
CATALOG_PATH = (
    ROOT
    / "config"
    / "model_catalog_healthbench_professional_mixed_authoritative_thinking_v2.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v216_freezes_v215_fixed20_director_graph_and_reference_evaluator() -> None:
    v215 = _load(V215_CONFIG_PATH)
    v216 = _load(V216_CONFIG_PATH)

    assert v216["healthbench_professional_evaluation"]["task_ids"] == (
        v215["healthbench_professional_evaluation"]["task_ids"]
    )
    assert v216["healthbench_professional_evaluation"]["sample_count"] == 20
    assert v216["experiment"]["seed"] == v215["experiment"]["seed"]
    assert v216["experiment"]["prompt_version"] == (
        v215["experiment"]["prompt_version"]
    )
    assert v216["director"] == v215["director"]
    assert v216["evaluation"] == v215["evaluation"]

    graph = deepcopy(v216["agent_graph"])
    graph["model_catalog_path"] = v215["agent_graph"]["model_catalog_path"]
    assert graph == v215["agent_graph"]
    assert v216["agent_graph"]["contract_type"] == "free_text"
    assert v216["agent_graph"]["semantic_protocol_by_source"] == {
        "healthbench_professional": "none"
    }


def test_v216_exposes_only_reasoning_or_authoritative_react_profiles() -> None:
    config = _load(V216_CONFIG_PATH)
    bounded = config["healthbench_professional_evaluation"]
    tool = config["healthbench_tool_runtime"]

    assert bounded["direct_execution_mode"] == "react"
    assert bounded["direct_allowed_tools"] == [
        "healthbench-authoritative.search"
    ]
    assert bounded["protocol_equivalent_to_direct"] is False
    assert "direct_reused_from" not in bounded
    assert tool["mode"] == "model_driven_authoritative_search"
    assert tool["execution_profile_allowlist"] == [
        {"execution_mode": "reasoning", "allowed_tools": []},
        {
            "execution_mode": "react",
            "allowed_tools": ["healthbench-authoritative.search"],
        },
    ]
    assert tool["source_identity"] == "MedRAG/textbooks"
    assert tool["expected_rows"] == 125847
    assert tool["authoritative_web_search"]["provider"] == (
        "ncbi_pubmed_eutils"
    )


def test_v216_catalog_keeps_four_models_and_advertises_only_validated_local_tool() -> None:
    catalog = _load(CATALOG_PATH)
    models = {model["model_id"]: model for model in catalog["models"]}

    assert set(models) == {
        "qwen3.5-9b-local",
        "qwen3.5-flash",
        "deepseek-v4-flash",
        "MiniMax-M3",
    }
    local_metadata = models["qwen3.5-9b-local"]["metadata"]
    assert local_metadata["tool_capable"] == "true"
    assert local_metadata["tool_capability_scope"] == (
        "healthbench-authoritative.search"
    )
    for model_id in ("qwen3.5-flash", "deepseek-v4-flash", "MiniMax-M3"):
        assert models[model_id]["metadata"]["tool_capable"] == "false"


def test_v216_keeps_training_and_skill_updates_disabled() -> None:
    config = _load(V216_CONFIG_PATH)

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
