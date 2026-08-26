from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.evaluate_completion_benchmark_round import (
    validate_completion_benchmark_config,
)
from src.interactive.config_loader import ConfigurationError, load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_initial_v1.yaml"
)
_STABLE_ZERO_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v8.yaml"
)
_STABLE_ZERO_CWD_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v9.yaml"
)
_STABLE_ZERO_MEXEC_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v11.yaml"
)
_STABLE_ZERO_OUTPUT_ADMISSION_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v12.yaml"
)
_STABLE_ZERO_FREE_AGENT_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v13.yaml"
)
_STABLE_ZERO_DIRECT_REUSE_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v14.yaml"
)
_STABLE_ZERO_SHARED_WORKSPACE_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v15.yaml"
)
_STABLE_ZERO_DOCKER_FALLBACK_CONFIG_PATH = (
    _ROOT / "config" / "evaluation_swebench_skillflow_v3_stable_zero_v16.yaml"
)


def _config() -> dict[str, object]:
    return dict(load_yaml(_CONFIG_PATH))


def test_initial_condition_is_evaluation_only_free_agentgraph() -> None:
    config = _config()

    validate_completion_benchmark_config(config)

    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    graph = config["agent_graph"]
    assert bounded["split"] == "test"
    assert bounded["sample_count"] == 128
    assert config["execution_timeout"] == 600.0
    assert "task_timeout_seconds" not in bounded
    assert bounded["paired_budget_protocol"] == "task_global_equal_v1"
    assert bounded["protocol_equivalent_to_direct"] is False
    assert bounded["repository_episode_budget_equivalent"] is True
    assert runtime["tool_profile"] == "skillflow_training_v1"
    assert runtime["completion_policy"] == "workspace_diff"
    assert runtime["require_task_environment"] is True
    assert runtime["task_max_turns"] == runtime["max_turns_per_agent_call"]
    assert (
        runtime["task_max_tool_calls"]
        == runtime["max_tool_calls_per_agent_call"]
    )
    assert graph["contract_type"] == "free_text"
    assert graph["actions"] == [
        "add_subgraph",
        "modify_agent",
        "delete_agent",
        "set_relation",
        "set_output",
        "finish",
    ]
    assert graph["require_format_agent"] is False
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["director"]["lora"]["enabled"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("task_max_turns", 13), ("task_max_tool_calls", 11)),
)
def test_paired_condition_rejects_resettable_or_unequal_budget(
    field: str,
    value: int,
) -> None:
    config = deepcopy(_config())
    config["swe_coding_runtime"][field] = value

    with pytest.raises(ConfigurationError, match=field):
        validate_completion_benchmark_config(config)


def test_strict_profile_rejects_proxy_completion_policy() -> None:
    config = deepcopy(_config())
    config["swe_coding_runtime"]["completion_policy"] = "tested_diff_receipt"

    with pytest.raises(ConfigurationError, match="workspace_diff_completion"):
        validate_completion_benchmark_config(config)


def test_strict_profile_requires_task_specific_environment() -> None:
    config = deepcopy(_config())
    config["swe_coding_runtime"]["require_task_environment"] = False

    with pytest.raises(ConfigurationError, match="require_task_environment"):
        validate_completion_benchmark_config(config)


@pytest.mark.parametrize(
    "config_path", (_STABLE_ZERO_CONFIG_PATH, _STABLE_ZERO_CWD_CONFIG_PATH)
)
def test_v8_v9_reuse_skillflow_code_generation_episode_without_training(
    config_path: Path,
) -> None:
    config = dict(load_yaml(config_path))

    validate_completion_benchmark_config(config)

    runtime = config["swe_coding_runtime"]
    assert runtime["max_turns_per_agent_call"] == 28
    assert runtime["max_tool_calls_per_agent_call"] == 28
    assert runtime["task_max_turns"] == 28
    assert runtime["task_max_tool_calls"] == 28
    assert config["swebench_evaluation"]["concurrency"] == 2
    assert config["director"]["sampling_schema_version"] == (
        "agentgraph.model-admissible-action-mask.v2"
    )
    assert config["agent_graph"]["contract_type"] == "free_text"
    assert config["agent_graph"]["require_format_agent"] is False
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_v11_freezes_mexec_force_termination_and_swe_memory_condition() -> None:
    config = dict(load_yaml(_STABLE_ZERO_MEXEC_CONFIG_PATH))

    validate_completion_benchmark_config(config)

    experiment = config["experiment"]
    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    tool_version = experiment["tool_version"]
    assert experiment["condition_id"] == (
        "swebench_skillflow_v3_stable_zero_v11"
    )
    assert experiment["seed"] == 20260825
    assert bounded["split"] == "test"
    assert bounded["sample_count"] == 128
    assert bounded["stable_zero_sample_count"] == 2
    assert bounded["concurrency"] == 2
    assert bounded["direct_generation_seed"] == 20260825
    assert bounded["official_metric"] == "resolved_rate"
    assert runtime["max_turns_per_agent_call"] == 28
    assert runtime["max_tool_calls_per_agent_call"] == 28
    assert runtime["task_max_turns"] == 28
    assert runtime["task_max_tool_calls"] == 28
    assert config["agent_graph"]["model_catalog_path"] == (
        "config/model_catalog_multidataset_tool_v2.yaml"
    )
    assert config["evaluation"]["swebench_harness_enabled"] is True
    assert "skillflow.mexec-edit-file.v1" in tool_version
    assert (
        "skillflow.force-terminate-empty-patch-official-evaluation.v1"
        in tool_version
    )
    assert "skillflow.swe-memory-context-compaction.v1" in tool_version
    assert "including an empty patch" in bounded["direct_completion_condition"]
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_v12_changes_only_runtime_output_admission_condition() -> None:
    previous = dict(load_yaml(_STABLE_ZERO_MEXEC_CONFIG_PATH))
    config = dict(load_yaml(_STABLE_ZERO_OUTPUT_ADMISSION_CONFIG_PATH))

    validate_completion_benchmark_config(config)

    experiment = config["experiment"]
    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    assert experiment["condition_id"] == (
        "swebench_skillflow_v3_stable_zero_v12"
    )
    assert experiment["prompt_version"] == previous["experiment"][
        "prompt_version"
    ]
    assert experiment["tool_version"] == (
        previous["experiment"]["tool_version"]
        + "+flowsteer.runtime-output-admission.v1"
    )
    assert experiment["seed"] == previous["experiment"]["seed"] == 20260825
    assert bounded == previous["swebench_evaluation"]
    assert bounded["sample_count"] == 128
    assert bounded["concurrency"] == 2
    assert runtime["max_turns_per_agent_call"] == 28
    assert runtime["max_tool_calls_per_agent_call"] == 28
    assert runtime["task_max_turns"] == 28
    assert runtime["task_max_tool_calls"] == 28
    assert config["agent_graph"]["model_catalog_path"] == previous[
        "agent_graph"
    ]["model_catalog_path"]
    assert config["evaluation"]["swebench_evaluator_path"] == previous[
        "evaluation"
    ]["swebench_evaluator_path"]
    assert config["evaluation"]["swebench_harness_path"] == previous[
        "evaluation"
    ]["swebench_harness_path"]
    assert config["evaluation"]["swebench_dataset_path"] == previous[
        "evaluation"
    ]["swebench_dataset_path"]
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_v13_freezes_free_agent_runtime_and_checkpoint_resume_condition() -> None:
    previous = dict(load_yaml(_STABLE_ZERO_OUTPUT_ADMISSION_CONFIG_PATH))
    config = dict(load_yaml(_STABLE_ZERO_FREE_AGENT_CONFIG_PATH))

    validate_completion_benchmark_config(config)

    experiment = config["experiment"]
    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    director = config["director"]
    assert experiment["condition_id"] == (
        "swebench_skillflow_v3_stable_zero_v13"
    )
    assert experiment["prompt_version"] == (
        "agentgraph.director.minimal-neutral.v10"
    )
    assert experiment["tool_version"] == (
        previous["experiment"]["tool_version"]
        + "+flowsteer.free-agent-runtime-profile-declaration.v1"
        + "+flowsteer.direct-evaluator-checkpoint-resume.v1"
    )
    assert experiment["seed"] == previous["experiment"]["seed"] == 20260825
    assert bounded == previous["swebench_evaluation"]
    assert bounded["sample_count"] == 128
    assert bounded["concurrency"] == 2
    assert runtime["max_turns_per_agent_call"] == 28
    assert runtime["max_tool_calls_per_agent_call"] == 28
    assert runtime["task_max_turns"] == 28
    assert runtime["task_max_tool_calls"] == 28
    assert director["sampling_schema_version"] == (
        "agentgraph.model-admissible-action-mask.v3"
    )
    assert config["agent_graph"]["contract_type"] == "free_text"
    assert config["agent_graph"]["require_format_agent"] is False
    assert config["agent_graph"]["model_catalog_path"] == previous[
        "agent_graph"
    ]["model_catalog_path"]
    assert config["evaluation"]["swebench_evaluator_path"] == previous[
        "evaluation"
    ]["swebench_evaluator_path"]
    assert config["evaluation"]["swebench_harness_path"] == previous[
        "evaluation"
    ]["swebench_harness_path"]
    assert config["evaluation"]["swebench_dataset_path"] == previous[
        "evaluation"
    ]["swebench_dataset_path"]
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_v14_reuses_v13_direct_checkpoint_without_changing_protocol() -> None:
    previous = dict(load_yaml(_STABLE_ZERO_FREE_AGENT_CONFIG_PATH))
    config = dict(load_yaml(_STABLE_ZERO_DIRECT_REUSE_CONFIG_PATH))

    validate_completion_benchmark_config(config)

    experiment = config["experiment"]
    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    director = config["director"]
    assert experiment["condition_id"] == (
        "swebench_skillflow_v3_stable_zero_v14"
    )
    assert experiment["prompt_version"] == previous["experiment"][
        "prompt_version"
    ] == "agentgraph.director.minimal-neutral.v10"
    assert experiment["tool_version"] == (
        previous["experiment"]["tool_version"]
        + "+flowsteer.generic-add-receipt-root-binding.v1"
    )
    assert experiment["seed"] == previous["experiment"]["seed"] == 20260825
    previous_bounded = dict(previous["swebench_evaluation"])
    assert bounded["direct_reused_from"] == (
        "artifacts/swebench_skillflow_v3_stable_zero_v13/"
        "direct_predictions.jsonl"
    )
    assert {
        key: value
        for key, value in bounded.items()
        if key != "direct_reused_from"
    } == previous_bounded
    assert bounded["sample_count"] == 128
    assert bounded["concurrency"] == 2
    assert runtime["max_turns_per_agent_call"] == 28
    assert runtime["max_tool_calls_per_agent_call"] == 28
    assert runtime["task_max_turns"] == 28
    assert runtime["task_max_tool_calls"] == 28
    assert director["sampling_schema_version"] == (
        "agentgraph.model-admissible-action-mask.v3"
    )
    assert config["agent_graph"]["model_catalog_path"] == previous[
        "agent_graph"
    ]["model_catalog_path"]
    assert config["evaluation"]["swebench_evaluator_path"] == previous[
        "evaluation"
    ]["swebench_evaluator_path"]
    assert config["evaluation"]["swebench_harness_path"] == previous[
        "evaluation"
    ]["swebench_harness_path"]
    assert config["evaluation"]["swebench_dataset_path"] == previous[
        "evaluation"
    ]["swebench_dataset_path"]
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_v15_freezes_shared_workspace_empty_patch_and_finish_feedback() -> None:
    previous = dict(load_yaml(_STABLE_ZERO_DIRECT_REUSE_CONFIG_PATH))
    config = dict(load_yaml(_STABLE_ZERO_SHARED_WORKSPACE_CONFIG_PATH))

    validate_completion_benchmark_config(config)

    experiment = config["experiment"]
    bounded = config["swebench_evaluation"]
    runtime = config["swe_coding_runtime"]
    director = config["director"]
    assert experiment["condition_id"] == (
        "swebench_skillflow_v3_stable_zero_v15"
    )
    assert experiment["prompt_version"] == previous["experiment"][
        "prompt_version"
    ] == "agentgraph.director.minimal-neutral.v10"
    assert experiment["tool_version"] == (
        previous["experiment"]["tool_version"]
        + "+flowsteer.serialized-shared-repository-workspace.v1"
        + "+flowsteer.empty-patch-communication-envelope.v1"
        + "+flowsteer.swebench-finish-admissibility-feedback.v1"
    )
    assert experiment["seed"] == previous["experiment"]["seed"] == 20260825
    assert bounded == previous["swebench_evaluation"]
    assert bounded["direct_reused_from"] == (
        "artifacts/swebench_skillflow_v3_stable_zero_v13/"
        "direct_predictions.jsonl"
    )
    assert bounded["sample_count"] == 128
    assert bounded["concurrency"] == 2
    assert runtime["max_turns_per_agent_call"] == 28
    assert runtime["max_tool_calls_per_agent_call"] == 28
    assert runtime["task_max_turns"] == 28
    assert runtime["task_max_tool_calls"] == 28
    assert director["sampling_schema_version"] == (
        "agentgraph.model-admissible-action-mask.v3"
    )
    assert config["agent_graph"]["contract_type"] == "free_text"
    assert config["agent_graph"]["require_format_agent"] is False
    assert config["agent_graph"]["model_catalog_path"] == previous[
        "agent_graph"
    ]["model_catalog_path"]
    assert config["evaluation"]["swebench_evaluator_path"] == previous[
        "evaluation"
    ]["swebench_evaluator_path"]
    assert config["evaluation"]["swebench_harness_path"] == previous[
        "evaluation"
    ]["swebench_harness_path"]
    assert config["evaluation"]["swebench_dataset_path"] == previous[
        "evaluation"
    ]["swebench_dataset_path"]
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False


def test_v16_limits_official_docker_fallback_to_two_xarray_instances() -> None:
    previous = dict(load_yaml(_STABLE_ZERO_SHARED_WORKSPACE_CONFIG_PATH))
    config = dict(load_yaml(_STABLE_ZERO_DOCKER_FALLBACK_CONFIG_PATH))

    validate_completion_benchmark_config(config)

    assert config["experiment"]["condition_id"] == (
        "swebench_skillflow_v3_stable_zero_v16"
    )
    assert config["experiment"]["tool_version"] == (
        previous["experiment"]["tool_version"]
        + "+flowsteer.official-docker-testbed-fallback.v1"
    )
    assert config["experiment"]["catalog_order_namespace"] == previous[
        "experiment"
    ]["catalog_order_namespace"]
    runtime = config["swe_coding_runtime"]
    assert runtime["official_docker_fallback_instance_ids"] == [
        "pydata__xarray-7229",
        "pydata__xarray-7393",
    ]
    assert runtime["official_docker_workspace_root"].endswith(
        "/docker_workspaces"
    )
    assert config["swebench_evaluation"]["sample_count"] == 128
    assert config["grpo"]["enabled"] is False
    assert config["skills"]["enabled"] is False
