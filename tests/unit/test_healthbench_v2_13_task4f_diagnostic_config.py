from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = (
    "healthbench_professional_mixed_all_thinking_"
    "v2_13_task4f_empty_artifact_repair_diagnostic"
)
TASK_ID = "healthbench-professional:4f118b7f2841f4816f4b8d4b989e4500"
CONFIG_PATH = ROOT / "config" / f"evaluation_{NAMESPACE}.yaml"
V212_CONFIG_PATH = (
    ROOT
    / "config"
    / "evaluation_healthbench_professional_mixed_all_thinking_"
    "v2_12_heldout20_output_closure.yaml"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v213_selects_only_the_task4f_diagnostic_case() -> None:
    config = _load(CONFIG_PATH)
    condition = config["healthbench_professional_evaluation"]

    assert condition["selection"] == "task_ids"
    assert condition["sample_count"] == 1
    assert condition["stable_zero_sample_count"] == 1
    assert condition["task_ids"] == [TASK_ID]


def test_v213_uses_an_isolated_condition_output_storage_and_report_namespace() -> None:
    config = _load(CONFIG_PATH)

    assert config["experiment"]["name"] == NAMESPACE
    assert config["experiment"]["condition_id"] == NAMESPACE
    assert config["experiment"]["output_dir"] == f"artifacts/{NAMESPACE}/evaluation"
    for key, path in config["storage"].items():
        if key == "schema_version":
            continue
        expected_root = "reports" if key.startswith("report_") else "artifacts"
        assert path.startswith(f"{expected_root}/{NAMESPACE}/"), key
        assert "v2_12_heldout20_output_closure" not in path, key
    assert config["policy_sync"]["adapter_name_prefix"] == f"unused_{NAMESPACE}_"


def test_v213_keeps_v212_models_director_tool_generation_seed_and_evaluator() -> None:
    config = _load(CONFIG_PATH)
    v212 = _load(V212_CONFIG_PATH)

    assert config["experiment"]["seed"] == v212["experiment"]["seed"]
    assert config["experiment"]["prompt_version"] == v212["experiment"]["prompt_version"]
    assert config["experiment"]["tool_version"] == v212["experiment"]["tool_version"]
    assert config["director"] == v212["director"]
    assert config["agent_graph"] == v212["agent_graph"]
    assert config["healthbench_tool_runtime"] == v212["healthbench_tool_runtime"]
    assert config["evaluation"] == v212["evaluation"]

    condition = config["healthbench_professional_evaluation"]
    previous = v212["healthbench_professional_evaluation"]
    for key in (
        "rollouts_per_task",
        "concurrency",
        "task_timeout_seconds",
        "direct_model_id",
        "direct_protocol",
        "protocol_equivalent_to_direct",
        "direct_contract",
        "direct_generation_seed",
        "direct_reused_from",
    ):
        assert condition[key] == previous[key], key


def test_v213_has_no_training_exploration_or_skills_enabled() -> None:
    config = _load(CONFIG_PATH)

    assert config["experiment"]["training_enabled"] is False
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["optimization_passes_per_rollout_batch"] == 0
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["exploration"]["forced_probe_rollouts"] == 0
    assert config["skills"] == {
        "enabled": False,
        "initial_library": [],
        "retrieval_top_k": 0,
        "library_version": "none",
    }
    assert config["gpu"]["training_enabled"] is False
    assert config["deployment"]["active_skills_only"] is False
