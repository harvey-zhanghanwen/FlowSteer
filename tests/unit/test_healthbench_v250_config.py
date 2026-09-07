"""Only output identities change after the tokenizer compatibility repair."""

import json

import pytest

from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
from src.interactive.config_loader import validate_agent_graph_config
from tests.unit.test_healthbench_v249_finish_choice import config


@pytest.mark.parametrize("suffix", ["dev5", "full525"])
def test_v250_preserves_five_tasks_models_tools_and_official_evaluator(suffix):
    old, new = config(49, suffix), config(50, suffix)
    validate_agent_graph_config(new)
    validate_completion_benchmark_config(new)
    expected = json.loads(json.dumps(old).replace(
        f"healthbench_professional_candidate_skill_v2_49_{suffix}",
        f"healthbench_professional_candidate_skill_v2_50_{suffix}",
    ))
    assert new == expected
    assert new["agent_graph"]["local_agent_context_budget"] is True
    assert new["agent_graph"]["finish_only_when_admissible"] is False
    assert new["candidate_skill_evaluation"]["profile_path"] == "config/healthbench_candidate_skills_v248.yaml"
