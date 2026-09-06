"""Offline checks of v2.42 inference-only configuration/factory adaptation."""
from pathlib import Path
from unittest.mock import patch

import pytest

from src.interactive.config_loader import ConfigurationError, load_yaml
from tests.unit.test_healthbench_v241_factory_wiring import _config, _settings


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("field", ["constrain_evidence_metadata", "respect_model_action_token_budget"])
def test_v242_flags_are_opt_in_and_boolean(field):
    config = _config()
    assert _settings(config)[field] is False
    config["healthbench_tool_runtime"][field] = True
    assert _settings(config)[field] is True
    for invalid in ("true", "false", 1, 0, None):
        config["healthbench_tool_runtime"][field] = invalid
        with pytest.raises(ConfigurationError, match=field):
            _settings(config)


@pytest.mark.parametrize("field", ["constrain_evidence_metadata", "respect_model_action_token_budget"])
def test_v242_flags_require_authoritative_mode(field):
    config = _config()
    config["healthbench_tool_runtime"].update(mode="model_driven_medrag_search", **{field: True})
    with pytest.raises(ConfigurationError, match=field):
        _settings(config)


@pytest.mark.parametrize("panel,count", [("dev5", 5), ("full525", 525)])
def test_frozen_inference_configs_use_existing_limits_and_no_training(panel, count):
    config = load_yaml(ROOT / f"config/evaluation_healthbench_professional_candidate_skill_v2_42_{panel}.yaml")
    from scripts.healthbench_candidate_skill_profile import validate_candidate_skill_run_config
    validate_candidate_skill_run_config(config)
    assert config["healthbench_professional_evaluation"]["sample_count"] == count
    assert config["healthbench_professional_evaluation"]["task_timeout_seconds"] == 900
    assert config["healthbench_professional_evaluation"]["concurrency"] == 4
    assert config["execution_timeout"] == 360
    assert config["healthbench_tool_runtime"]["max_turns_per_agent_call"] == 6
    assert config["healthbench_tool_runtime"]["max_tool_calls_per_agent_call"] == 3
    assert config["healthbench_tool_runtime"]["constrain_evidence_metadata"] is True
    assert config["healthbench_tool_runtime"]["respect_model_action_token_budget"] is True
    assert config["agent_graph"]["allow_same_provider_transient_repair"] is True
    assert config["candidate_skill_evaluation"]["profile_path"] == "config/healthbench_candidate_skills_v242.yaml"


def test_clinical_adapter_uses_state_aware_completion_hook():
    from src.interactive.healthbench_clinical_react import HealthBenchClinicalReactExecutionAdapter
    adapter = object.__new__(HealthBenchClinicalReactExecutionAdapter)
    observed = [{"observation_status": "success"}]
    schema = {"type": "object", "properties": {"value": {"const": "public"}}}
    with patch.object(adapter, "_state_conditioned_action_domain", return_value=(frozenset(), True)), \
         patch.object(adapter, "_completion_arguments_schema_for_state", return_value=schema) as hook:
        result = adapter._state_conditioned_response_schema(None, observed)
    hook.assert_called_once_with(None, observed)
    assert result["properties"]["arguments"] == schema
