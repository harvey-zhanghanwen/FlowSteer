from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V218_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_18_heldout20_capability_repair_query.yaml"
)
V219_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_19_heldout20_executable_domain_query_guard.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v219_preserves_fixed20_models_generation_and_reference_evaluator() -> None:
    v218 = _load(V218_CONFIG_PATH)
    v219 = _load(V219_CONFIG_PATH)

    assert v219["data"] == v218["data"]
    assert v219["healthbench_professional_evaluation"] == (
        v218["healthbench_professional_evaluation"]
    )
    assert v219["director"] == v218["director"]
    assert v219["agent_graph"] == v218["agent_graph"]
    assert v219["evaluation"] == v218["evaluation"]
    assert len(v219["healthbench_professional_evaluation"]["task_ids"]) == 20
    assert len(set(v219["healthbench_professional_evaluation"]["task_ids"])) == 20


def test_v219_preserves_retrieval_sources_and_bumps_behavior_version() -> None:
    v218 = _load(V218_CONFIG_PATH)
    v219 = _load(V219_CONFIG_PATH)
    old_tool = dict(v218["healthbench_tool_runtime"])
    new_tool = dict(v219["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")

    assert new_tool == old_tool
    assert new_tool["source_identity"] == "MedRAG/textbooks"
    assert new_tool["authoritative_web_search"]["provider"] == (
        "ncbi_pubmed_eutils"
    )
    assert v219["experiment"]["tool_version"] == (
        "skillflow.medrag+ncbi-pubmed-eutils-react.query-guarded.v3"
    )


def test_v219_has_an_independent_receipt_namespace() -> None:
    v218 = _load(V218_CONFIG_PATH)
    v219 = _load(V219_CONFIG_PATH)

    assert v218["experiment"]["condition_id"] != (
        v219["experiment"]["condition_id"]
    )
    assert "v2_19" in v219["experiment"]["condition_id"]
    for path_value in v219["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_19" in path_value


def test_v219_keeps_training_and_skill_updates_disabled() -> None:
    config = _load(V219_CONFIG_PATH)

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
