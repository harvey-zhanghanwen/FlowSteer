from __future__ import annotations

import importlib.util
from pathlib import Path

from src.interactive.config_loader import load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_aime_config", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)

_DEVELOPMENT_CONFIG = (
    _ROOT / "config" / "development_aime2026_computation_tool_stable_zero.yaml"
)
_FINAL_CONFIG = (
    _ROOT / "config" / "evaluation_aime2026_computation_tool_stable_zero.yaml"
)


def test_aime_computation_development_config_is_evaluation_only() -> None:
    config = load_yaml(_DEVELOPMENT_CONFIG)

    _RUNNER.validate_completion_benchmark_config(config)

    bounded = config["aime2026_evaluation"]
    assert bounded["stage"] == "development"
    assert bounded["split"] == "validation"
    assert bounded["benchmark_slice"] == "development_aime_2025"
    assert "official_2026_only" not in bounded
    assert bounded["sample_count"] == 30
    assert bounded["stable_zero_sample_count"] == 2
    assert config["experiment"]["training_enabled"] is False
    assert config["grpo"]["enabled"] is False
    assert config["grpo"]["max_optimizer_updates"] == 0
    assert config["policy_sync"]["enabled"] is False
    assert config["skills"]["enabled"] is False
    assert config["aime_tool_runtime"]["condition_id"] == (
        config["experiment"]["condition_id"]
    )


def test_aime_development_and_final_protocols_remain_disjoint() -> None:
    development = load_yaml(_DEVELOPMENT_CONFIG)
    final = load_yaml(_FINAL_CONFIG)

    _RUNNER.validate_completion_benchmark_config(development)
    _RUNNER.validate_completion_benchmark_config(final)

    development_bounded = development["aime2026_evaluation"]
    final_bounded = final["aime2026_evaluation"]
    assert development_bounded["split"] == "validation"
    assert development_bounded["benchmark_slice"] == "development_aime_2025"
    assert final_bounded["stage"] == "final_evaluation"
    assert final_bounded["split"] == "test"
    assert final_bounded["benchmark_slice"] == "official_aime_2026"
    assert final_bounded["official_2026_only"] is True
    assert development["experiment"]["condition_id"] != (
        final["experiment"]["condition_id"]
    )

    development_paths = {
        str(development["experiment"]["output_dir"]),
        *(
            str(value)
            for key, value in development["storage"].items()
            if key != "schema_version"
        ),
    }
    final_paths = {
        str(final["experiment"]["output_dir"]),
        *(
            str(value)
            for key, value in final["storage"].items()
            if key != "schema_version"
        ),
    }
    assert development_paths.isdisjoint(final_paths)
    assert all("/development" in path for path in development_paths)
    assert all("/evaluation" in path for path in final_paths)


def test_aime_development_selection_uses_only_aime_2025_validation(
    tmp_path: Path,
) -> None:
    config = load_yaml(_DEVELOPMENT_CONFIG)

    selected = _RUNNER._select_tasks(
        config,
        _ROOT,
        tmp_path / "selected_tasks.jsonl",
    )

    assert len(selected) == 30
    assert all(task.split == "validation" for task in selected)
    assert all(task.task_id.startswith("aime-2025:") for task in selected)
    assert all(
        task.metadata["benchmark_slice"] == "development_aime_2025"
        for task in selected
    )
    assert not any(task.task_id.startswith("aime-2026:") for task in selected)
