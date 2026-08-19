from __future__ import annotations

import importlib.util
from pathlib import Path

from src.interactive.config_loader import load_model_registry, load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = (
    _ROOT / "config" / "development_hotpotqa_tool_react_stable_zero_v4.yaml"
)
_RUNNER_PATH = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_hotpot_v4_config_test",
    _RUNNER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def test_hotpot_v4_is_partition_guarded_development_only() -> None:
    config = load_yaml(_CONFIG)

    _RUNNER.validate_completion_benchmark_config(config)
    bounded = config["hotpotqa_evaluation"]
    assert bounded["stage"] == "development"
    assert bounded["required_partition"] == "development"
    assert bounded["split"] == "validation"
    assert bounded["sample_count"] == 128
    assert bounded["stable_zero_sample_count"] == 2
    assert config["data"]["train_path"] == "data/joint_qa_v2/train.jsonl"
    assert config["data"]["validation_path"] == (
        "data/joint_qa_v2/development.jsonl"
    )
    assert config["data"]["test_path"] == "data/joint_qa_v2/test.jsonl"
    assert config["agent_graph"]["model_catalog_path"] == (
        "config/model_catalog_multidataset_tool_v2.yaml"
    )
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["skills"]["enabled"] is False


def test_hotpot_v4_selection_contains_only_declared_development_partition(
    tmp_path: Path,
) -> None:
    config = load_yaml(_CONFIG)

    selected = _RUNNER._select_tasks(
        config,
        _ROOT,
        tmp_path / "selected_tasks.jsonl",
    )

    assert len(selected) == 128
    assert all(task.split == "validation" for task in selected)
    assert all(
        task.metadata.get("joint_qa_partition") == "development"
        for task in selected
    )


def test_future_catalog_records_exact_canary_and_capability_boundaries() -> None:
    config = load_yaml(_CONFIG)
    catalog_path = _ROOT / config["agent_graph"]["model_catalog_path"]
    registry = load_model_registry(catalog_path)

    assert len(registry) == 7
    for model_id in registry.model_ids:
        model = registry.require_model(model_id)
        assert model.metadata["text_capable"] == "true"
        assert model.metadata["tool_capable"] == "true"
        assert model.metadata["coding_capable"] == "true"

    local = registry.require_model("qwen3.5-9b-local")
    assert local.metadata["capability_canary"] == "passed_2026-08-20"
    assert local.metadata["canary_source"] == (
        "artifacts/model_capability_canary/"
        "local_qwen35_9b_nonthinking_20260820.json"
    )
    assert local.metadata["chat_template_enable_thinking"] == "false"
