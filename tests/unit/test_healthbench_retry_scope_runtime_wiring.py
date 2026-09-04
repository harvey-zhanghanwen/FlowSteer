from __future__ import annotations

import asyncio
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.interactive.agent_runtime import AgentRuntime
from src.interactive.config_loader import (
    ConfigurationError,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.records import TaskRecord
from src.interactive.versioning import VersionBundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "train_agentgraph_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "train_agentgraph_smoke_healthbench_retry_scope_wiring",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="healthbench-professional:wiring-test",
        question="Conversation:\n\n[user] What should I discuss with my clinician?",
        ground_truth="EVALUATOR_ONLY",
        split="test",
        metadata={
            "dataset_key": "healthbench_professional",
            "source": "HealthBench Professional",
        },
    )


def _authoritative_config() -> dict:
    condition_id = "healthbench-retry-scope-wiring"
    return {
        "experiment": {"condition_id": condition_id},
        "director": {"max_action_tokens": 512},
        "healthbench_tool_runtime": {
            "enabled": True,
            "condition_id": condition_id,
            "mode": "model_driven_authoritative_search",
            "dataset_scope": ["healthbench_professional"],
            "resource_dir": "resources/medrag-textbooks-runtime",
            "source_identity": "MedRAG/textbooks",
            "source_revision": "fixture-revision",
            "expected_rows": 2,
            "max_turns_per_agent_call": 4,
            "max_tool_calls_per_agent_call": 3,
            "max_successful_queries": 2,
            "require_initial_search": True,
            "tool_timeout_seconds": 20.0,
            "authoritative_web_search": {
                "enabled": True,
                "provider": "ncbi_pubmed_eutils",
                "base_url": (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
                ),
                "tool_name": "FlowSteer-HealthBench",
                "retmax": 3,
                "request_timeout_seconds": 8.0,
                "minimum_interval_seconds": 0.4,
            },
        },
    }


def test_authoritative_retry_settings_default_to_historical_behavior() -> None:
    settings = SMOKE._healthbench_tool_runtime_settings(
        _authoritative_config(),
        _task(),
        condition_id="healthbench-retry-scope-wiring",
    )

    assert settings is not None
    web = settings["authoritative_web_search"]
    assert web["max_retries"] == 0
    assert web["retry_backoff_seconds"] == 1.0


def test_authoritative_retry_settings_validate_and_normalize() -> None:
    config = _authoritative_config()
    web = config["healthbench_tool_runtime"]["authoritative_web_search"]
    web["max_retries"] = 2
    web["retry_backoff_seconds"] = 0.25

    settings = SMOKE._healthbench_tool_runtime_settings(
        config,
        _task(),
        condition_id="healthbench-retry-scope-wiring",
    )

    assert settings is not None
    assert settings["authoritative_web_search"]["max_retries"] == 2
    assert settings["authoritative_web_search"]["retry_backoff_seconds"] == 0.25

    invalid_values = (
        ("max_retries", True),
        ("max_retries", -1),
        ("max_retries", 1.5),
        ("retry_backoff_seconds", True),
        ("retry_backoff_seconds", -0.1),
        ("retry_backoff_seconds", math.inf),
    )
    for field_name, value in invalid_values:
        invalid = copy.deepcopy(config)
        invalid["healthbench_tool_runtime"]["authoritative_web_search"][
            field_name
        ] = value
        with pytest.raises(ConfigurationError, match=field_name):
            SMOKE._healthbench_tool_runtime_settings(
                invalid,
                _task(),
                condition_id="healthbench-retry-scope-wiring",
            )


def test_authoritative_retry_settings_reach_pubmed_client(tmp_path: Path) -> None:
    config = _authoritative_config()
    web = config["healthbench_tool_runtime"]["authoritative_web_search"]
    web["max_retries"] = 3
    web["retry_backoff_seconds"] = 0.125

    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = config
    backend.registry = object()
    backend.runtime = SimpleNamespace(
        gateway=object(),
        timeout_seconds=30.0,
        artifact_communication_profile="legacy",
    )
    backend.project_root = tmp_path
    opened = SimpleNamespace(registry=object(), close=Mock())
    created_runtime = object()

    with (
        patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()) as client,
        patch.object(
            SMOKE,
            "open_healthbench_authoritative_tool_registry",
            return_value=opened,
        ),
        patch.object(
            SMOKE,
            "HealthBenchAuthoritativeReactExecutionAdapter",
            return_value=object(),
        ),
        patch.object(SMOKE, "AgentRuntime", return_value=created_runtime),
    ):
        runtime, registry, close = backend._runtime_for_task(
            _task(),
            condition_id="healthbench-retry-scope-wiring",
        )

    assert runtime is created_runtime
    assert registry is opened.registry
    assert close is opened.close
    assert client.call_args.kwargs["max_retries"] == 3
    assert client.call_args.kwargs["retry_backoff_seconds"] == 0.125


def test_scope_neutral_contract_flag_is_validated_and_reaches_environment() -> None:
    config_path = (
        PROJECT_ROOT
        / "config"
        / (
            "evaluation_healthbench_professional_mixed_all_thinking_v2_22_"
            "heldout20_scope_preserving_all_model_react.yaml"
        )
    )
    config = load_yaml(config_path, expand_env=False)
    config["healthbench_tool_runtime"]["enabled"] = False
    config["agent_graph"]["require_scope_neutral_contracts"] = True
    validate_agent_graph_config(config)

    invalid = copy.deepcopy(config)
    invalid["agent_graph"]["require_scope_neutral_contracts"] = "true"
    with pytest.raises(ConfigurationError, match="require_scope_neutral_contracts"):
        validate_agent_graph_config(invalid)

    registry = load_model_registry(
        PROJECT_ROOT / "config" / "model_catalog_triviaqa_v1.yaml"
    )

    class NoCallGateway:
        async def generate(self, request):  # pragma: no cover - constructor guard
            raise AssertionError(f"unexpected model call: {request.request_id}")

    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = config
    backend.registry = registry
    backend.runtime = AgentRuntime(registry, NoCallGateway())
    backend.director_client = object()
    backend.skill_pipeline = None
    backend.project_root = PROJECT_ROOT
    captured: dict = {}

    class EnvironmentReached(RuntimeError):
        pass

    def capture_environment(*args, **kwargs):
        captured.update(kwargs)
        raise EnvironmentReached

    versions = VersionBundle(
        policy="policy",
        model_catalog="catalog",
        evaluator="evaluator",
        prompt="prompt",
        tool="tool",
    )
    with patch.object(SMOKE, "AgentWorkflowEnv", side_effect=capture_environment):
        with pytest.raises(EnvironmentReached):
            asyncio.run(
                backend.collect(
                    _task(),
                    0,
                    versions,
                    expected_task_split="test",
                )
            )

    assert captured["require_scope_neutral_contracts"] is True
