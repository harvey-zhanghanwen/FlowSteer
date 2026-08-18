from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.evaluate_hotpotqa_round import validate_hotpot_config
from scripts.evaluate_triviaqa_round import validate_trivia_config
from src.interactive.config_loader import ConfigurationError, load_yaml


ROOT = Path(__file__).resolve().parents[2]


def _memory_on(config):
    value = deepcopy(config)
    value["skills"] = {
        "enabled": True,
        "store_path": "artifacts/joint_qa_skill/skills.json",
        "retrieval_top_k": 1,
        "current_epoch": 2,
    }
    value["deployment"] = {
        "exploration_beta": 0.0,
        "allow_forced_probes": False,
        "active_skills_only": True,
        "require_version_compatible_skills": True,
    }
    return value


@pytest.mark.parametrize(
    ("config_name", "validator"),
    (
        ("evaluation_joint_qa_step2_hotpotqa.yaml", validate_hotpot_config),
        ("evaluation_joint_qa_step2_triviaqa.yaml", validate_trivia_config),
    ),
)
def test_fixed_evaluator_accepts_active_only_memory_on(config_name, validator):
    config = _memory_on(load_yaml(ROOT / "config" / config_name))
    validator(config)


@pytest.mark.parametrize(
    ("config_name", "validator"),
    (
        ("evaluation_joint_qa_step2_hotpotqa.yaml", validate_hotpot_config),
        ("evaluation_joint_qa_step2_triviaqa.yaml", validate_trivia_config),
    ),
)
def test_fixed_evaluator_rejects_memory_on_without_active_only(config_name, validator):
    config = _memory_on(load_yaml(ROOT / "config" / config_name))
    config["deployment"]["active_skills_only"] = False
    with pytest.raises(ConfigurationError, match="skills.evaluation_mode"):
        validator(config)
