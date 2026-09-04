from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V220_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_20_heldout20_material_fidelity_relation_semantics.yaml"
)
V221_CONFIG_PATH = (
    ROOT
    / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_21_heldout20_material_fidelity_relation_semantics.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v221_changes_only_receipt_namespace_from_v220() -> None:
    v220 = _load(V220_CONFIG_PATH)
    v221 = _load(V221_CONFIG_PATH)

    for key in (
        "data",
        "healthbench_professional_evaluation",
        "director",
        "agent_graph",
        "evaluation",
        "grpo",
        "exploration",
        "skills",
        "gpu",
    ):
        assert v221[key] == v220[key]
    old_tool = dict(v220["healthbench_tool_runtime"])
    new_tool = dict(v221["healthbench_tool_runtime"])
    old_tool.pop("condition_id")
    new_tool.pop("condition_id")
    assert new_tool == old_tool
    assert "v2_21" in v221["experiment"]["condition_id"]
    for path_value in v221["storage"].values():
        if isinstance(path_value, str) and "/" in path_value:
            assert "v2_21" in path_value


def test_v221_stays_evaluation_only() -> None:
    config = _load(V221_CONFIG_PATH)
    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False
