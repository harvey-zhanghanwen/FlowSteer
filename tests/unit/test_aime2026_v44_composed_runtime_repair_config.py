from pathlib import Path

from src.interactive.config_loader import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "config/evaluation_aime2026_runtime_v44_composed_runtime_repair.yaml"
)
V45_CONFIG_PATH = (
    ROOT
    / "config/evaluation_aime2026_runtime_v45_provider_protocol_admission.yaml"
)
FIXED30_TASK_IDS = {
    f"aime-2026/{index:02d}" for index in range(1, 31)
}


def test_v44_preserves_same30_and_targets_only_two_canary_blockers():
    config = load_yaml(CONFIG_PATH)
    evaluation = config["aime2026_evaluation"]

    assert evaluation["selection"] == "task_ids"
    assert evaluation["sample_count"] == 30
    assert evaluation["stable_zero_sample_count"] == 2
    assert evaluation["task_ids"][:2] == [
        "aime-2026/05",
        "aime-2026/10",
    ]
    assert set(evaluation["task_ids"]) == FIXED30_TASK_IDS
    assert len(evaluation["task_ids"]) == len(FIXED30_TASK_IDS)


def test_v44_keeps_role_neutral_untrained_agentgraph_protocol():
    config = load_yaml(CONFIG_PATH)

    assert config["experiment"]["condition_id"] == (
        "aime2026_runtime_v44_composed_runtime_repair"
    )
    assert config["experiment"]["prompt_version"] == (
        "agentgraph.director.minimal-neutral.v15"
    )
    assert config["director"]["sampling_schema_version"] == (
        "agentgraph.model-admissible-action-mask.v3"
    )
    assert config["agent_graph"]["contract_type"] == "free_text"
    assert config["agent_graph"]["require_format_agent"] is False
    assert config["agent_graph"]["model_catalog_path"] == (
        "config/model_catalog_aime2026_heterogeneous_thinking_v18.yaml"
    )
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["exploration"]["enabled"] is False


def test_v45_preserves_same30_and_only_retries_the_remaining_canary_blocker():
    config = load_yaml(V45_CONFIG_PATH)
    evaluation = config["aime2026_evaluation"]

    assert evaluation["sample_count"] == 30
    assert evaluation["stable_zero_sample_count"] == 1
    assert evaluation["task_ids"][0] == "aime-2026/10"
    assert set(evaluation["task_ids"]) == FIXED30_TASK_IDS
    assert config["experiment"]["condition_id"] == (
        "aime2026_runtime_v45_provider_protocol_admission"
    )
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False
