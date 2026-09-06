from pathlib import Path

from src.interactive.config_loader import load_yaml
from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config


def test_v237_is_only_five_development_cases_with_free_graph_and_no_training():
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(root / "config/evaluation_healthbench_professional_grounded_retrieval_v2_37_dev5.yaml")
    old = load_yaml(root / "config/evaluation_healthbench_professional_source_separated_retrieval_v2_36.yaml")
    validate_completion_benchmark_config(config)
    evaluation = config["healthbench_professional_evaluation"]
    assert evaluation["stage"] == "development"
    assert evaluation["sample_count"] == len(set(evaluation["task_ids"])) == 5
    assert evaluation["selection"] == "task_ids"
    assert config["agent_graph"] == old["agent_graph"]
    assert config["evaluation"] == old["evaluation"]
    assert config["director"]["base_model"] == old["director"]["base_model"]
    assert config["director"]["chat_template_enable_thinking"] is True
    assert config["experiment"]["prompt_version"] == old["experiment"]["prompt_version"]
    assert config["director"]["lora"]["enabled"] is False
    assert all(config[key]["enabled"] is False for key in ("grpo", "policy_sync", "exploration", "skills"))
    assert config["gpu"]["rollout_physical"] == 6
