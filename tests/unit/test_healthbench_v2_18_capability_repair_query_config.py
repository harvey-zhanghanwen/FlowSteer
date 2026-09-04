from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V217_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_17_heldout20_context_relation_repair.yaml"
)
V218_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_18_heldout20_capability_repair_query.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v218_preserves_fixed20_models_generation_and_reference_evaluator() -> None:
    v217 = _load(V217_CONFIG_PATH)
    v218 = _load(V218_CONFIG_PATH)

    assert v218["data"] == v217["data"]
    assert v218["healthbench_professional_evaluation"] == (
        v217["healthbench_professional_evaluation"]
    )
    assert v218["director"] == v217["director"]
    assert v218["agent_graph"] == v217["agent_graph"]
    assert v218["evaluation"] == v217["evaluation"]
    assert len(v218["healthbench_professional_evaluation"]["task_ids"]) == 20
    assert len(set(v218["healthbench_professional_evaluation"]["task_ids"])) == 20


def test_v218_preserves_database_and_web_search_condition() -> None:
    v217 = _load(V217_CONFIG_PATH)
    v218 = _load(V218_CONFIG_PATH)
    old_tool = dict(v217["healthbench_tool_runtime"])
    new_tool = dict(v218["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")

    assert new_tool == old_tool
    assert new_tool["mode"] == "model_driven_authoritative_search"
    assert new_tool["source_identity"] == "MedRAG/textbooks"
    assert new_tool["authoritative_web_search"]["provider"] == (
        "ncbi_pubmed_eutils"
    )
    assert v218["experiment"]["tool_version"] == (
        "skillflow.medrag+ncbi-pubmed-eutils-react.query-bounded.v2"
    )


def test_v218_has_an_independent_receipt_namespace() -> None:
    v217 = _load(V217_CONFIG_PATH)
    v218 = _load(V218_CONFIG_PATH)
    old_name = v217["experiment"]["condition_id"]
    new_name = v218["experiment"]["condition_id"]

    assert old_name != new_name
    assert "v2_18" in new_name
    for path_value in v218["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_18" in path_value


def test_v218_keeps_all_training_and_skill_updates_disabled() -> None:
    config = _load(V218_CONFIG_PATH)

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
