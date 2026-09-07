"""Offline checks for the failure-derived candidate-only full population."""

from pathlib import Path

from scripts.healthbench_candidate_skill_profile import (
    build_candidate_prompt_priors,
    load_candidate_skill_profile,
)
from src.interactive.config_loader import load_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/evaluation_healthbench_failure_skills_full525.yaml"
BASE = ROOT / "config/evaluation_healthbench_deadline_recovery_single.yaml"


def test_full525_preserves_selected_architecture_and_inference_only_boundary():
    config = load_yaml(CONFIG, expand_env=False)
    base = load_yaml(BASE, expand_env=False)
    for section in ("director", "agent_graph", "evaluation", "grpo", "skills",
                    "policy_sync", "exploration", "deployment", "gpu", "data"):
        assert config[section] == base[section], section
    assert config["execution_timeout"] == base["execution_timeout"]
    for key, value in base["healthbench_tool_runtime"].items():
        if key not in {"condition_id", "knowledge_root"}:
            assert config["healthbench_tool_runtime"][key] == value, key
    population = config["healthbench_professional_evaluation"]
    assert population["sample_count"] == 525
    assert population["selection"] == "sequential"
    assert "task_ids" not in population
    assert population["stage"] == "development"
    assert population["split"] == "test"
    for key, value in base["healthbench_professional_evaluation"].items():
        if key not in {"sample_count", "selection", "task_ids"}:
            assert population[key] == value, key
    assert config["experiment"]["training_enabled"] is False
    for key in ("seed", "prompt_version", "sampling_schedule_purpose",
                "catalog_order_namespace", "tool_version"):
        assert config["experiment"][key] == base["experiment"][key]


def test_three_new_rejectable_priors_replace_previous_profile_without_publication():
    config = load_yaml(CONFIG, expand_env=False)
    assert config["candidate_skill_evaluation"]["enabled"] is True
    profile = load_candidate_skill_profile(
        ROOT / config["candidate_skill_evaluation"]["profile_path"], run_config=config,
    )
    priors = build_candidate_prompt_priors(profile)
    old = load_yaml(ROOT / "config/healthbench_candidate_skills_semantic_v1.yaml", expand_env=False)
    assert len(priors) == 3
    assert not ({p["condition_id"] for p in priors} &
                {p["condition_id"] for p in old["candidates"]})
    assert all(p["rejectable"] and "skill_id" not in p for p in priors)
    assert all(p["condition"]["required_tools"] == [] for p in priors)
    assert all(p["condition"]["graph_stage"] == "*" for p in priors)
    assert sum(len(p["content"]) for p in priors) <= 4000
    assert profile["status"] == "candidate"
    assert profile["training_enabled"] is False
    assert profile["skill_publication_enabled"] is False
