"""Synthetic, offline tests for candidate-only prompt conditions."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.healthbench_candidate_skill_profile import (
    build_candidate_prompt_priors,
    load_candidate_skill_profile,
    validate_candidate_skill_profile,
    validate_candidate_skill_run_config,
)
from src.interactive.config_loader import ConfigurationError, load_yaml


PROFILE = Path(__file__).resolve().parents[2] / "config/healthbench_candidate_skills_v248.yaml"


def _run_config():
    return {
        "experiment": {"training_enabled": False},
        "grpo": {"enabled": False, "max_optimizer_updates": 0},
        "exploration": {"enabled": False},
        "skills": {"enabled": False},
        "policy_sync": {"enabled": False},
        "director": {"lora": {"enabled": False}},
    }


def _profile():
    return load_yaml(PROFILE, expand_env=False)


def test_existing_condition_wire_is_rejectable_without_active_identity():
    profile = load_candidate_skill_profile(PROFILE, run_config=_run_config())
    original = deepcopy(profile)
    priors = build_candidate_prompt_priors(profile)
    assert len(priors) == 3
    assert len({p["condition_id"] for p in priors}) == 3
    for prior in priors:
        assert prior["application_mode"] == "forced_probe_condition"
        assert prior["rejectable"] is True
        assert "Only when:" in prior["content"]
        assert "unvalidated" in prior["content"]
        assert "Otherwise ignore" in prior["content"]
        assert "skill_id" not in prior
        assert "effect_mean" not in prior
        assert set(prior["action"]) == {"instruction"}
    assert profile == original
    priors[0]["condition"]["required_tools"].append("unavailable")
    assert profile == original


@pytest.mark.parametrize("where", ("profile", "item"))
def test_rejects_active_status(where):
    profile = _profile()
    (profile if where == "profile" else profile["candidates"][0])["status"] = "active"
    with pytest.raises(ConfigurationError, match="candidate"):
        validate_candidate_skill_profile(profile)


@pytest.mark.parametrize("flag", ("training_enabled", "skill_publication_enabled"))
def test_rejects_profile_training_or_publication(flag):
    profile = _profile()
    profile[flag] = True
    with pytest.raises(ConfigurationError, match="training or publication"):
        validate_candidate_skill_profile(profile)


@pytest.mark.parametrize("section", ("grpo", "exploration", "skills", "policy_sync", "mace", "bayesian", "skill_evolution"))
def test_rejects_enabled_training_evolution_or_active_store(section):
    config = _run_config()
    config[section] = {"enabled": True}
    with pytest.raises(ConfigurationError):
        load_candidate_skill_profile(PROFILE, run_config=config)


def test_rejects_training_lora_and_optimizer_update():
    for update in ("training", "lora", "optimizer"):
        config = _run_config()
        if update == "training":
            config["experiment"]["training_enabled"] = True
        elif update == "lora":
            config["director"]["lora"]["enabled"] = True
        else:
            config["grpo"]["max_optimizer_updates"] = 1
        with pytest.raises(ConfigurationError):
            validate_candidate_skill_run_config(config)


@pytest.mark.parametrize("identity", ("", " ", "duplicate"))
def test_rejects_empty_or_duplicate_candidate_identity(identity):
    profile = _profile()
    profile["candidates"][0]["condition_id"] = (
        profile["candidates"][1]["condition_id"] if identity == "duplicate" else identity
    )
    with pytest.raises(ConfigurationError):
        validate_candidate_skill_profile(profile)


def test_rejects_excessive_prompt_length():
    profile = _profile()
    profile["candidates"][0]["action"]["instruction"] = "x" * 801
    with pytest.raises(ConfigurationError, match="800"):
        build_candidate_prompt_priors(profile)


@pytest.mark.parametrize("field", ("gate_receipt", "paired_effect_mean", "reference_answer"))
def test_no_fake_evidence_or_answer_payload_fields(field):
    profile = _profile()
    profile["candidates"][0][field] = "not allowed"
    with pytest.raises(ConfigurationError, match="field set"):
        validate_candidate_skill_profile(profile)


def test_rejects_unavailable_dataset_and_mismatched_family():
    profile = _profile()
    with pytest.raises(ConfigurationError, match="only healthbench"):
        validate_candidate_skill_profile(profile, dataset_key="hotpotqa")
    profile["candidates"][0]["condition"]["task_family"] = "other"
    with pytest.raises(ConfigurationError, match="task_family"):
        validate_candidate_skill_profile(profile)


def test_no_fixed_role_model_relation_or_tool_in_profile():
    for candidate in _profile()["candidates"]:
        assert set(candidate["action"]) == {"instruction"}
        assert candidate["condition"]["required_tools"] == []
    profile = _profile()
    profile["candidates"][0]["action"]["model_id"] = "fixed_model"
    with pytest.raises(ConfigurationError, match="field set"):
        validate_candidate_skill_profile(profile)
