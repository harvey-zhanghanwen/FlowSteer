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
    "train_agentgraph_smoke_healthbench_query_budget_wiring",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def _task() -> TaskRecord:
    return TaskRecord(
        task_id="healthbench-professional:query-budget-wiring",
        question="Conversation:\n\n[user] What should I discuss with my clinician?",
        ground_truth="EVALUATOR_ONLY",
        split="test",
        metadata={
            "dataset_key": "healthbench_professional",
            "source": "HealthBench Professional",
        },
    )


def _config() -> dict:
    condition_id = "healthbench-query-budget-wiring"
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
        condition_id="healthbench-query-budget-wiring",
    )
    assert settings is not None
    return settings


def test_query_budget_defaults_to_historical_twelve_and_is_receipted() -> None:
    assert _settings(_config())["max_query_content_tokens"] == 12


@pytest.mark.parametrize("valid", [1, 6, 12])
def test_query_budget_accepts_the_bounded_positive_integer_range(valid: int) -> None:
    config = _config()
    config["healthbench_tool_runtime"]["max_query_content_tokens"] = valid

    assert _settings(config)["max_query_content_tokens"] == valid


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 13, 1.5, "6"])
def test_query_budget_rejects_boolean_or_out_of_range_values(
    invalid: object,
) -> None:
    config = copy.deepcopy(_config())
    config["healthbench_tool_runtime"]["max_query_content_tokens"] = invalid

    with pytest.raises(ConfigurationError, match="max_query_content_tokens"):
        _settings(config)


def test_query_budget_reaches_registry_and_authoritative_adapter(
    tmp_path: Path,
) -> None:
    config = _config()
    config["healthbench_tool_runtime"]["max_query_content_tokens"] = 6

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
        patch.object(SMOKE, "PubMedEUtilitiesClient", return_value=object()),
        patch.object(
            SMOKE,
            "open_healthbench_authoritative_tool_registry",
            return_value=opened,
        ) as open_registry,
        patch.object(
            SMOKE,
            "HealthBenchAuthoritativeReactExecutionAdapter",
            return_value=object(),
        ) as adapter,
        patch.object(SMOKE, "AgentRuntime", return_value=created_runtime),
    ):
        runtime, registry, close = backend._runtime_for_task(
            _task(),
            condition_id="healthbench-query-budget-wiring",
        )

    assert runtime is created_runtime
    assert registry is opened.registry
    assert close is opened.close
    assert open_registry.call_args.kwargs["max_query_content_tokens"] == 6
    assert adapter.call_args.kwargs["max_query_content_tokens"] == 6
