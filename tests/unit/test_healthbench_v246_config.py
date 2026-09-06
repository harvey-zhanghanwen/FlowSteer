"""Offline checks for the fixed ReAct-only HealthBench iteration condition.

Reuse the existing candidate/config validators and the v6 ReAct-only profile
test pattern; this does not load a model, Tool backend or private evaluator.
"""

from pathlib import Path

import pytest

from scripts.healthbench_candidate_skill_profile import (
    build_candidate_prompt_priors,
    load_candidate_skill_profile,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(suffix="dev5"):
    return load_yaml(
        ROOT / f"config/evaluation_healthbench_professional_candidate_skill_v2_46_{suffix}.yaml",
        expand_env=False,
    )


@pytest.mark.parametrize("suffix", ("dev5", "full525"))
def test_react_only_condition_keeps_existing_free_graph_and_no_training(suffix):
    config = _config(suffix)
    validate_agent_graph_config(config)
    profile = load_candidate_skill_profile(
        ROOT / config["candidate_skill_evaluation"]["profile_path"],
        run_config=config,
    )
    priors = build_candidate_prompt_priors(profile)
    assert len(priors) == 3
    assert all(prior["rejectable"] for prior in priors)
    assert all("skill_id" not in prior for prior in priors)
    graph = config["agent_graph"]
    assert graph["contract_type"] == "free_text"
    assert graph["semantic_protocol_by_source"]["healthbench_professional"] == "none"
    assert graph["require_format_agent"] is False
    assert graph["allow_untried_react_model_repair"] is True
    profiles = config["healthbench_tool_runtime"]["execution_profile_allowlist"]
    assert len(profiles) == 1 and profiles[0]["execution_mode"] == "react"
    assert "healthbench-source.read" in profiles[0]["allowed_tools"]
    assert config["healthbench_tool_runtime"]["require_initial_search"] is False
    assert config["director"]["execute_on_edit"] is True
    assert config["director"]["chat_template_enable_thinking"] is True
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0


def test_new_five_remain_fixed_and_full_run_keeps_generation_conditions():
    dev = _config()
    full = _config("full525")
    bounded = dev["healthbench_professional_evaluation"]
    assert bounded["selection"] == "task_ids"
    assert bounded["task_ids"] == [
        "healthbench-professional:9566084de89c416408691006a6f06f9c",
        "healthbench-professional:c19c2113ba68bb3c4a3e63836e31b558",
        "healthbench-professional:a5778c7ecdb4eeccf9d252631e18a274",
        "healthbench-professional:e339f34a3a35f3f067422b5768287f7c",
        "healthbench-professional:c42bd4fc760487ac7b5e70fbb41a8edc",
    ]
    assert bounded["sample_count"] == 5
    assert full["healthbench_professional_evaluation"]["sample_count"] == 525
    assert full["healthbench_professional_evaluation"]["selection"] == "sequential"
    for field in ("director", "agent_graph", "evaluation", "candidate_skill_evaluation"):
        assert dev[field] == full[field]
    for field in ("concurrency", "task_timeout_seconds", "rollouts_per_task"):
        assert bounded[field] == full["healthbench_professional_evaluation"][field]
    for field in ("max_turns_per_agent_call", "max_tool_calls_per_agent_call", "execution_profile_allowlist"):
        assert dev["healthbench_tool_runtime"][field] == full["healthbench_tool_runtime"][field]


def test_untried_model_repair_flag_rejects_non_boolean_configuration():
    config = _config()
    config["agent_graph"]["allow_untried_react_model_repair"] = "true"
    with pytest.raises(ConfigurationError, match="allow_untried_react_model_repair"):
        validate_agent_graph_config(config)


@pytest.mark.parametrize("enabled", (False, True))
def test_inference_factory_passes_the_explicit_react_repair_flag(tmp_path, enabled):
    # Directly reuse the earlier mocked inference factory; stop at Env creation.
    import asyncio
    from unittest.mock import Mock, patch

    from tests.unit.test_healthbench_v241_factory_wiring import (
        SMOKE, VersionBundle, _task, backend_config, build_fake_backend,
    )

    config = backend_config(tmp_path)
    config["agent_graph"]["allow_untried_react_model_repair"] = enabled
    backend, _ = build_fake_backend(config)
    captured = {}

    class ReachedEnvironment(RuntimeError):
        pass

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise ReachedEnvironment

    close = Mock()
    versions = VersionBundle(
        policy="policy", model_catalog="catalog", evaluator="evaluator",
        prompt="prompt", tool="tool",
    )
    with patch.object(backend, "_runtime_for_task", return_value=(backend.runtime, None, close)), \
         patch.object(SMOKE, "AgentWorkflowEnv", side_effect=capture):
        with pytest.raises(ReachedEnvironment):
            asyncio.run(backend.collect(_task(), 0, versions, expected_task_split="test"))
    assert captured["allow_untried_react_model_repair"] is enabled
    close.assert_called_once()
