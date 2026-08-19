from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from src.interactive.config_loader import load_model_registry, load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = (
    _ROOT / "config" / "evaluation_swebench_verified_coding_agent_stable_zero.yaml"
)
_RUNNER_PATH = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_swe_coding_config_test",
    _RUNNER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def test_swe_coding_stable_zero_config_is_fixed_and_evaluation_only() -> None:
    config = load_yaml(_CONFIG)

    _RUNNER.validate_completion_benchmark_config(config)
    bounded = config["swebench_evaluation"]
    assert bounded["dataset_key"] == "swe_bench"
    assert bounded["stage"] == "development"
    assert bounded["split"] == "validation"
    assert bounded["selection"] == "sequential"
    assert bounded["sample_count"] == 128
    assert bounded["stable_zero_sample_count"] == 2
    assert bounded["rollouts_per_task"] == 1
    assert bounded["concurrency"] == 1
    assert bounded["official_metric"] == "resolved_rate"
    assert bounded["proxy_metrics_allowed"] is False
    assert config["data"] == {
        "catalog_path": "config/datasets_swebench.yaml",
        "train_path": "data/swebench_v2/train.jsonl",
        "validation_path": "data/swebench_v2/validation.jsonl",
        "test_path": "data/swebench_v2/test.jsonl",
        "enforce_split_isolation": True,
        "task_schema_version": "flowsteer.agentgraph.task.v1",
    }
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["optimization_passes_per_rollout_batch"] == 0
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["exploration"]["enabled"] is False
    assert config["gpu"]["training_enabled"] is False


def test_swe_coding_agentgraph_arm_uses_the_shared_runtime_contract() -> None:
    config = load_yaml(_CONFIG)
    runtime = config["swe_coding_runtime"]

    assert runtime == {
        "enabled": True,
        "condition_id": "swebench_regular_dev_coding_agent_stable_zero",
        "mode": "iterative_repository_coding",
        "dataset_scope": ["swe_bench"],
        "repository_store": (
            "/ssd1/iclr/.private/skillflow-resources/swe-repositories"
        ),
        "worktree_root": ("/ssd1/iclr/.private/skillflow-resources/swe-worktrees"),
        "max_turns_per_agent_call": 6,
        "max_tool_calls_per_agent_call": 6,
        "max_test_timeout_seconds": 60.0,
        "setup_timeout_seconds": 30.0,
        "cleanup_timeout_seconds": 10.0,
    }
    assert config["agent_graph"]["model_catalog_path"] == (
        "config/model_catalog_multidataset_tool_v1.yaml"
    )
    registry = load_model_registry(_ROOT / config["agent_graph"]["model_catalog_path"])
    assert registry.require_model("qwen3.5-9b-local").model_id == ("qwen3.5-9b-local")


def test_direct_arm_is_the_supported_single_coding_agent_baseline() -> None:
    config = load_yaml(_CONFIG)
    bounded = config["swebench_evaluation"]
    baseline = bounded["direct_baseline"]

    assert baseline["topology"] == "single_agent"
    assert baseline["execution_mode"] == "coding"
    assert baseline["execution_adapter"] == "CodingExecutionAdapter"
    assert baseline["tool_registry"] == "swebench_repository"
    assert baseline["require_same_repository_tools_as_agentgraph"] is True
    assert baseline["one_shot_patch_is_equivalent"] is False
    assert baseline["status"] == "supported"
    assert bounded["direct_execution_mode"] == "coding"
    assert bounded["direct_completion_condition"] == (
        "submit the tested unified workspace diff"
    )

    # Keep the protocol honest: the Direct path must construct the task-scoped
    # repository runtime and select the CodingExecutionAdapter path.
    source = inspect.getsource(_RUNNER._direct_one)
    assert "_runtime_for_task" in source
    assert 'execution_mode="coding"' in source
    assert 'simple_baseline_topology": "single_coding_agent"' in source


def test_swe_coding_stable_zero_uses_only_the_official_resolved_metric() -> None:
    config = load_yaml(_CONFIG)
    evaluation = config["evaluation"]

    assert evaluation["swebench_harness_enabled"] is True
    assert evaluation["swebench_evaluator_path"] == (
        "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/"
        "training/swe_bench_eval.py"
    )
    assert evaluation["swebench_harness_path"] == (
        "/ssd1/iclr/.private/skillflow-resources/SWE-bench-harness-d83"
    )
    assert evaluation["swebench_dataset_source"] == "regular_dev"
    assert evaluation["swebench_dataset_path"] == "data/swebench_v2/validation.jsonl"
    assert evaluation["swebench_docker_namespace"] == "swebench"
    assert evaluation["swebench_timeout_seconds"] == 900
    assert _RUNNER._BENCHMARKS["swe_bench"]["metric_names"] == ("resolved",)
