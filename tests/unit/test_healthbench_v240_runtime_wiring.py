"""Reuse the clinical factory fixtures; no inference, network or evaluator calls."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.interactive.config_loader import ConfigurationError, load_yaml
from tests.unit.test_healthbench_optional_tools_wiring import (
    SMOKE, _settings, _task, optional_config, registry,
)


def test_initial_query_fidelity_is_opt_in_and_typed():
    config = optional_config()
    assert _settings(config)["require_initial_query_fidelity"] is False
    config["healthbench_tool_runtime"]["require_initial_query_fidelity"] = True
    assert _settings(config)["require_initial_query_fidelity"] is True
    config["healthbench_tool_runtime"]["require_initial_query_fidelity"] = "true"
    with pytest.raises(ConfigurationError, match="require_initial_query_fidelity"):
        _settings(config)


def test_runtime_factory_passes_fidelity_without_enabling_mandatory_search(tmp_path):
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = optional_config()
    backend.config["healthbench_tool_runtime"]["require_initial_query_fidelity"] = True
    backend.registry = registry()
    backend.runtime = SimpleNamespace(gateway=object(), timeout_seconds=30,
        artifact_communication_profile="producer_context_structured_evidence_v4")
    backend.project_root = tmp_path
    opened = SimpleNamespace(registry=object(), close=Mock())
    adapter = SimpleNamespace(execute=Mock())
    with patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()), \
         patch.object(SMOKE, "open_healthbench_clinical_tool_registry", return_value=opened), \
         patch.object(SMOKE, "HealthBenchClinicalReactExecutionAdapter", return_value=adapter) as factory:
        backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert factory.call_args.kwargs["require_initial_query_fidelity"] is True
    assert factory.call_args.kwargs["require_initial_search"] is False


def test_v240_retains_five_cases_director_models_generation_budgets_and_evaluator():
    root = Path(__file__).resolve().parents[2]
    old = load_yaml(root / "config/evaluation_healthbench_professional_clinical_reference_sources_v2_39_dev5.yaml")
    new = load_yaml(root / "config/evaluation_healthbench_professional_query_recovery_v2_40_dev5.yaml")
    for key in ("director", "agent_graph", "evaluation", "grpo", "policy_sync", "skills", "gpu"):
        assert new[key] == old[key]
    assert new["healthbench_professional_evaluation"] == old["healthbench_professional_evaluation"]
    assert new["experiment"]["prompt_version"] == old["experiment"]["prompt_version"]
    runtime = dict(new["healthbench_tool_runtime"])
    assert runtime.pop("require_initial_query_fidelity") is True
    runtime["condition_id"] = old["healthbench_tool_runtime"]["condition_id"]
    runtime["knowledge_root"] = old["healthbench_tool_runtime"]["knowledge_root"]
    assert runtime == old["healthbench_tool_runtime"]
    assert "/evaluator_private/partial_trajectories.jsonl" in new["storage"]["partial_trajectories_path"]
    assert new["experiment"]["condition_id"] != old["experiment"]["condition_id"]
