from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.interactive.config_loader import ConfigurationError
from src.interactive.records import TaskRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "train_agentgraph_smoke.py"
SPEC = importlib.util.spec_from_file_location(
    "train_agentgraph_smoke_healthbench_completion_guard_wiring",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="healthbench-professional:completion-guard-wiring",
        question="Conversation:\n\n[user] What should I discuss with my clinician?",
        ground_truth="EVALUATOR_ONLY",
        split="test",
        metadata={
            "dataset_key": "healthbench_professional",
            "source": "HealthBench Professional",
        },
    )


def _config() -> dict:
    condition_id = "healthbench-completion-guard-wiring"
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


def _settings(config: dict) -> dict:
    settings = SMOKE._healthbench_tool_runtime_settings(
        config,
        _task(),
        condition_id="healthbench-completion-guard-wiring",
    )
    assert settings is not None
    return settings


def test_completion_guard_defaults_to_historical_unbounded_behavior() -> None:
    settings = _settings(_config())
    assert settings["max_completion_artifact_characters"] is None
    assert settings["require_relevant_evidence"] is False
    assert settings["require_complete_natural_language_artifact"] is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_relevant_evidence_gate_requires_boolean(invalid: object) -> None:
    config = copy.deepcopy(_config())
    config["healthbench_tool_runtime"]["require_relevant_evidence"] = invalid

    with pytest.raises(ConfigurationError, match="require_relevant_evidence"):
        _settings(config)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_complete_natural_language_guard_requires_boolean(invalid: object) -> None:
    config = copy.deepcopy(_config())
    config["healthbench_tool_runtime"][
        "require_complete_natural_language_artifact"
    ] = invalid

    with pytest.raises(
        ConfigurationError,
        match="require_complete_natural_language_artifact",
    ):
        _settings(config)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "12000"])
def test_completion_guard_requires_a_positive_non_boolean_integer(
    invalid: object,
) -> None:
    config = copy.deepcopy(_config())
    config["healthbench_tool_runtime"][
        "max_completion_artifact_characters"
    ] = invalid

    with pytest.raises(
        ConfigurationError,
        match="max_completion_artifact_characters",
    ):
        _settings(config)


def test_completion_guard_is_preserved_and_reaches_authoritative_adapter(
    tmp_path: Path,
) -> None:
    config = _config()
    config["healthbench_tool_runtime"][
        "max_completion_artifact_characters"
    ] = 12_000
    config["healthbench_tool_runtime"]["require_relevant_evidence"] = True
    config["healthbench_tool_runtime"][
        "require_complete_natural_language_artifact"
    ] = True
    assert _settings(config)["max_completion_artifact_characters"] == 12_000
    assert (
        _settings(config)["require_complete_natural_language_artifact"] is True
    )

    backend = object.__new__(SMOKE.LiveSmokeBackend)
    backend.config = config
    backend.registry = object()
    backend.runtime = SimpleNamespace(
        gateway=object(),
        timeout_seconds=30.0,
        artifact_communication_profile="legacy",
        artifact_quality_profile="public_text_quality_v1",
    )
    backend.project_root = tmp_path
    opened = SimpleNamespace(registry=object(), close=Mock())
    created_runtime = object()

    with (
        patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
        patch.object(
            SMOKE,
            "open_healthbench_authoritative_tool_registry",
            return_value=opened,
        ),
        patch.object(
            SMOKE,
            "HealthBenchAuthoritativeReactExecutionAdapter",
            return_value=object(),
        ) as adapter,
        patch.object(
            SMOKE,
            "AgentRuntime",
            return_value=created_runtime,
        ) as runtime_factory,
    ):
        runtime, registry, close = backend._runtime_for_task(
            _task(),
            condition_id="healthbench-completion-guard-wiring",
        )

    assert runtime is created_runtime
    assert registry is opened.registry
    assert close is opened.close
    assert (
        adapter.call_args.kwargs["max_completion_artifact_characters"]
        == 12_000
    )
    assert adapter.call_args.kwargs["require_relevant_evidence"] is True
    assert (
        adapter.call_args.kwargs[
            "require_complete_natural_language_artifact"
        ]
        is True
    )
    assert (
        runtime_factory.call_args.kwargs["artifact_quality_profile"]
        == "public_text_quality_v1"
    )
