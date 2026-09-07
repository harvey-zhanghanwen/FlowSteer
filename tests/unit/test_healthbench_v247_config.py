"""Reuse the real runner and candidate validators for this fixed iteration."""

from pathlib import Path

import pytest

from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
from scripts.healthbench_candidate_skill_profile import (
    build_candidate_prompt_priors, load_candidate_skill_profile,
)
from src.interactive.config_loader import load_yaml, validate_agent_graph_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("suffix", ["dev5", "full525"])
def test_v247_keeps_v246_sample_models_tools_generation_and_evaluator(suffix):
    def config(version):
        return load_yaml(
            ROOT / f"config/evaluation_healthbench_professional_candidate_skill_v2_{version}_{suffix}.yaml",
            expand_env=False,
        )

    old, new = config(46), config(47)
    validate_agent_graph_config(new)
    validate_completion_benchmark_config(new)
    for key in ("director", "agent_graph", "evaluation", "healthbench_professional_evaluation"):
        assert old[key] == new[key]
    assert old["healthbench_tool_runtime"]["execution_profile_allowlist"] == new["healthbench_tool_runtime"]["execution_profile_allowlist"]
    for key in ("task_timeout_seconds", "concurrency"):
        assert new["healthbench_professional_evaluation"][key] == old["healthbench_professional_evaluation"][key]
    profile = load_candidate_skill_profile(
        ROOT / new["candidate_skill_evaluation"]["profile_path"], run_config=new,
    )
    priors = build_candidate_prompt_priors(profile)
    assert len(priors) == 3 and all(item["rejectable"] for item in priors)
    assert new["experiment"]["training_enabled"] is False
    assert new["grpo"]["max_optimizer_updates"] == 0
