"""No-network regressions for explicitly selected multi-Tool HealthBench nodes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.interactive.agent_runtime import AgentRuntime
from src.interactive.computation_tools import create_aime_computation_registry
from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.model_registry import ModelRegistry, ModelSpec, ProviderSpec
from src.interactive.tool_runtime import ToolRegistration, ToolRegistry
from tests.unit.test_healthbench_query_budget_runtime_wiring import SMOKE, _config, _settings, _task


def registry():
    return ModelRegistry(
        providers=(ProviderSpec("p"),),
        models=(ModelSpec("m", "p", "m", metadata={"tool_capable": "true"}),),
    )


def test_explicit_bundle_uses_registered_tool_capabilities_without_changing_default():
    tools = create_aime_computation_registry()
    adapter = SimpleNamespace(execute=Mock())
    runtime = AgentRuntime(registry(), object(), execution_adapters={"react": adapter},
                           tool_registry=tools, dataset_id="aime_2026")
    original = runtime.registered_execution_profiles()
    bundle = ("react", tools.resource_ids)
    assert bundle not in original
    selected = AgentRuntime(registry(), object(), execution_adapters={"react": adapter},
                            tool_registry=tools, dataset_id="aime_2026",
                            execution_profile_allowlist=(("reasoning", ()), bundle))
    assert selected.registered_execution_profiles() == (("reasoning", ()), bundle)
    assert bundle in selected.registered_execution_profiles_for_model("m")
    assert runtime.registered_execution_profiles() == original


@pytest.mark.parametrize("case", ["missing", "unavailable", "wrong_dataset", "reasoning"])
def test_bundle_does_not_bypass_existing_capability_boundary(case):
    tools = create_aime_computation_registry()
    ids = tools.resource_ids
    profiles = (("reasoning" if case == "reasoning" else "react", ids),)
    if case == "missing":
        profiles = (("react", (ids[0], "not-registered")),)
    if case == "unavailable":
        tools = ToolRegistry(tuple(
            ToolRegistration(tool_id, tools._backend(tool_id),
                             replace(tools.require_capability(tool_id), availability=False))
            for tool_id in ids
        ))
    with pytest.raises(ValueError, match="unregistered"):
        AgentRuntime(registry(), object(), execution_adapters={"react": SimpleNamespace(execute=Mock())},
                     tool_registry=tools,
                     dataset_id="healthbench_professional" if case == "wrong_dataset" else "aime_2026",
                     execution_profile_allowlist=profiles)


def optional_config():
    config = _config()
    config["healthbench_tool_runtime"].update(toolset="optional_clinical_v1", require_initial_search=False)
    return config


def test_clinical_toolset_is_explicit_and_cannot_force_search():
    assert _settings(_config())["toolset"] == "default"
    assert _settings(optional_config())["toolset"] == "optional_clinical_v1"
    config = optional_config()
    config["healthbench_tool_runtime"]["require_initial_search"] = True
    with pytest.raises(ConfigurationError, match="without mandatory"):
        _settings(config)


def test_clinical_toolset_opens_correct_resources_and_keeps_cleanup(tmp_path):
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = optional_config()
    backend.registry = registry()
    backend.runtime = SimpleNamespace(gateway=object(), timeout_seconds=30,
                                      artifact_communication_profile="producer_context_structured_evidence_v4")
    backend.project_root = tmp_path
    opened = SimpleNamespace(registry=object(), close=Mock())
    adapter = SimpleNamespace(execute=Mock())
    with (
        patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
        patch.object(SMOKE, "open_healthbench_clinical_tool_registry", return_value=opened) as clinical,
        patch.object(SMOKE, "open_healthbench_authoritative_tool_registry") as historical,
        patch.object(SMOKE, "HealthBenchClinicalReactExecutionAdapter", return_value=adapter) as execution,
    ):
        runtime, tools, close = backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert clinical.call_count == 1
    historical.assert_not_called()
    assert execution.call_args.kwargs["require_initial_search"] is False
    assert execution.call_args.kwargs["require_structured_evidence_artifact"] is True
    assert runtime.execution_adapters["react"] is adapter
    assert tools is opened.registry and close is opened.close


def test_v235_keeps_free_graph_and_requires_new_matched_direct():
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(root / "config/evaluation_healthbench_professional_optional_clinical_tools_v2_35.yaml")
    reference = load_yaml(root / "config/evaluation_healthbench_professional_mixed_all_thinking_v2_34_full525_director_evidence_feedback.yaml")
    graph = dict(config["agent_graph"])
    graph.pop("model_catalog_path")
    old_graph = dict(reference["agent_graph"])
    old_graph.pop("model_catalog_path")
    assert graph == old_graph
    assert config["director"]["served_model_name"] == "supervisor_theta"
    assert config["director"]["lora"]["enabled"] is False
    assert config["grpo"]["enabled"] is False
    runtime = config["healthbench_tool_runtime"]
    profiles = runtime["execution_profile_allowlist"]
    assert len(profiles[-1]["allowed_tools"]) == 5
    direct = config["healthbench_professional_evaluation"]
    assert direct["direct_allowed_tools"] == profiles[-1]["allowed_tools"]
    assert not any(key in direct for key in ("direct_reference_config", "direct_reused_from", "direct_reference_manifest"))
    assert config["evaluation"] == reference["evaluation"]
    from tests.unit.test_completion_benchmark_round import _MODULE
    _MODULE.validate_completion_benchmark_config(config)
    config["healthbench_professional_evaluation"]["direct_allowed_tools"] = ["healthbench-authoritative.search"]
    with pytest.raises(ConfigurationError):
        _MODULE.validate_completion_benchmark_config(config)
