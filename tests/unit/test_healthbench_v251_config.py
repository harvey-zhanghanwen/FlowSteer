"""The HTTP400 verification uses the same frozen five and conditions."""

import json

from scripts.evaluate_completion_benchmark_round import validate_completion_benchmark_config
from src.interactive.config_loader import validate_agent_graph_config
from tests.unit.test_healthbench_v249_finish_choice import config


def test_v251_changes_run_identity_only():
    old, new = config(50), config(51)
    validate_agent_graph_config(new)
    validate_completion_benchmark_config(new)
    expected = json.loads(json.dumps(old).replace(
        "healthbench_professional_candidate_skill_v2_50_dev5",
        "healthbench_professional_candidate_skill_v2_51_dev5",
    ))
    assert new == expected
