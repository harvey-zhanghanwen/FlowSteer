"""Inference-only factory wiring with fake tokenizer, gateway and registries."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.versioning import VersionBundle
from tests.unit.test_healthbench_optional_tools_wiring import (
    SMOKE, _settings, _task, optional_config, registry,
)
from tests.unit.test_healthbench_query_budget_runtime_wiring import _config


ROOT = Path(__file__).resolve().parents[2]


def test_evidence_repair_defaults_false_and_accepts_true_only_in_authoritative_mode():
    config = _config()
    assert _settings(config)["enable_evidence_repair_feedback"] is False
    config["healthbench_tool_runtime"]["enable_evidence_repair_feedback"] = True
    assert _settings(config)["enable_evidence_repair_feedback"] is True
    config["healthbench_tool_runtime"]["mode"] = "model_driven_medrag_search"
    with pytest.raises(ConfigurationError, match="enable_evidence_repair_feedback"):
        _settings(config)
    config["healthbench_tool_runtime"]["enable_evidence_repair_feedback"] = False
    assert _settings(config)["enable_evidence_repair_feedback"] is False


@pytest.mark.parametrize("invalid", ["true", "false", 0, 1, None])
def test_evidence_repair_flag_rejects_coercion(invalid):
    config = _config()
    config["healthbench_tool_runtime"]["enable_evidence_repair_feedback"] = invalid
    with pytest.raises(ConfigurationError, match="enable_evidence_repair_feedback"):
        _settings(config)


@pytest.mark.parametrize("toolset", ["default", "optional_clinical_v1", "source_separated_clinical_v1"])
def test_evidence_repair_reaches_all_authoritative_factory_variants(tmp_path, toolset):
    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = optional_config() if toolset != "default" else _config()
    backend.config["healthbench_tool_runtime"].update(
        enable_evidence_repair_feedback=True, toolset=toolset,
    )
    if toolset == "source_separated_clinical_v1":
        backend.config["healthbench_tool_runtime"].update(
            knowledge_root=str(tmp_path / "knowledge"), skillflow_source=str(tmp_path / "source"),
        )
    backend.registry = registry()
    backend.runtime = SimpleNamespace(gateway=object(), timeout_seconds=30,
        artifact_communication_profile="producer_context_structured_evidence_v4")
    backend.project_root = tmp_path
    opened = SimpleNamespace(registry=object(), close=Mock())
    adapter = SimpleNamespace(execute=Mock())
    kind = {"default": "authoritative", "optional_clinical_v1": "clinical",
            "source_separated_clinical_v1": "knowledge"}[toolset]
    class_name = {"default": "Authoritative", "optional_clinical_v1": "Clinical",
                  "source_separated_clinical_v1": "Knowledge"}[toolset]
    with patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()), \
         patch.object(SMOKE, f"open_healthbench_{kind}_tool_registry", return_value=opened), \
         patch.object(SMOKE, f"HealthBench{class_name}ReactExecutionAdapter", return_value=adapter) as factory:
        backend._runtime_for_task(_task(), condition_id="healthbench-query-budget-wiring")
    assert factory.call_args.kwargs["enable_evidence_repair_feedback"] is True


def backend_config(tmp_path):
    config = load_yaml(ROOT / "config/evaluation_healthbench_professional_query_recovery_v2_40_dev5.yaml", expand_env=False)
    config["storage"]["root"] = str(tmp_path / "evidence")
    config["healthbench_tool_runtime"]["enabled"] = False
    return config


def build_fake_backend(config):
    fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()))
    fake_client = SimpleNamespace(prompt_token_ids=lambda prompt: ())
    with patch.dict(sys.modules, {"transformers": fake_transformers}), \
         patch.object(SMOKE, "load_model_registry", return_value=registry()), \
         patch.object(SMOKE, "SGLangReceiptDirectorClient", return_value=fake_client) as factory:
        backend = SMOKE.LiveSmokeBackend.from_config(config, ROOT, evaluation_only=True)
    return backend, factory.call_args.kwargs


@pytest.mark.parametrize("enabled", [False, True])
def test_context_and_phase_reserve_are_wired_without_model_calls(tmp_path, enabled):
    config = backend_config(tmp_path)
    if enabled:
        config["director"].update(context_projection=True, max_prompt_tokens=24000,
            reasoning_context_reserve_tokens=4608)
    backend, client_kwargs = build_fake_backend(config)
    assert backend.trainer is None
    assert client_kwargs["reasoning_context_reserve_tokens"] == (4608 if enabled else 0)
    assert client_kwargs["max_context_tokens"] == config["director"]["max_context_tokens"]
    captured = {}

    class ReachedOrchestrator(RuntimeError):
        pass

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise ReachedOrchestrator

    close = Mock()
    versions = VersionBundle(policy="policy", model_catalog="catalog", evaluator="evaluator", prompt="prompt", tool="tool")
    with patch.object(backend, "_runtime_for_task", return_value=(backend.runtime, None, close)), \
         patch.object(SMOKE, "AgentGraphOrchestrator", side_effect=capture):
        with pytest.raises(ReachedOrchestrator):
            asyncio.run(backend.collect(_task(), 0, versions, expected_task_split="test"))
    assert captured["context_projection"] is enabled
    assert captured["max_prompt_tokens"] == (24000 if enabled else None)
    close.assert_called_once()


@pytest.mark.parametrize("field, value", [
    ("context_projection", "true"), ("context_projection", 1),
    ("max_prompt_tokens", "24000"), ("max_prompt_tokens", True), ("max_prompt_tokens", 0),
    ("reasoning_context_reserve_tokens", "4608"), ("reasoning_context_reserve_tokens", True),
    ("reasoning_context_reserve_tokens", -1),
])
def test_director_budget_settings_reject_implicit_type_coercion(tmp_path, field, value):
    config = backend_config(tmp_path)
    config["director"][field] = value
    with pytest.raises(ConfigurationError, match=field):
        build_fake_backend(config)


def test_context_projection_requires_an_explicit_prompt_budget(tmp_path):
    config = backend_config(tmp_path)
    config["director"]["context_projection"] = True
    with pytest.raises(ConfigurationError, match="requires max_prompt_tokens"):
        build_fake_backend(config)
