"""Freeze the population and inference/evaluator condition across this repair."""

from pathlib import Path

import pytest

from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
from scripts.healthbench_candidate_skill_profile import build_candidate_prompt_priors, load_candidate_skill_profile
from src.interactive.config_loader import load_yaml, validate_agent_graph_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("suffix", ["dev5", "full525"])
def test_v248_preserves_v247_condition_except_candidate_and_output_paths(suffix):
    def config(version):
        return load_yaml(ROOT / f"config/evaluation_healthbench_professional_candidate_skill_v2_{version}_{suffix}.yaml", expand_env=False)

    old, new = config(47), config(48)
    validate_agent_graph_config(new)
    validate_completion_benchmark_config(new)
    for key in ("director", "agent_graph", "evaluation", "gpu", "grpo"):
        assert new[key] == old[key]
    for key, value in old["healthbench_professional_evaluation"].items():
        if key != "task_ids":
            assert new["healthbench_professional_evaluation"][key] == value
    if suffix == "dev5":
        selected = new["healthbench_professional_evaluation"]["task_ids"]
        assert len(selected) == len(set(selected)) == 5
        assert set(selected).isdisjoint(old["healthbench_professional_evaluation"]["task_ids"])
    for key, value in old["healthbench_tool_runtime"].items():
        if key not in {"condition_id", "knowledge_root"}:
            assert new["healthbench_tool_runtime"][key] == value
    profile = load_candidate_skill_profile(ROOT / new["candidate_skill_evaluation"]["profile_path"], run_config=new)
    priors = build_candidate_prompt_priors(profile)
    assert len(priors) == 3 and all(item["rejectable"] for item in priors)
    assert new["experiment"]["training_enabled"] is False
    assert new["grpo"]["max_optimizer_updates"] == 0
    assert profile["skill_publication_enabled"] is False
