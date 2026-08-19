from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from src.interactive.config_loader import ConfigurationError, load_yaml
from src.interactive.records import TaskRecord
from src.interactive.swebench_adapter import SWEbenchHarnessUnavailable


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_swebench_config_test",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)

_CONFIG_CASES = (
    (
        "development_swebench_verified_round_01.yaml",
        "development",
        "validation",
        32,
        "regular_dev",
    ),
    (
        "evaluation_swebench_verified_round_01.yaml",
        "evaluation",
        "test",
        128,
        "verified",
    ),
)
_SKILLFLOW_EVALUATOR = (
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/swe_bench_eval.py"
)
_OFFICIAL_HARNESS = "/ssd1/iclr/.private/skillflow-resources/SWE-bench-harness-d83"
_OFFICIAL_VERIFIED = "/ssd1/iclr/.private/skillflow-resources/swebench-verified"


def _config(name: str) -> dict:
    return load_yaml(_ROOT / "config" / name)


@pytest.mark.parametrize(
    ("name", "stage", "split", "sample_count", "dataset_source"),
    _CONFIG_CASES,
)
def test_swebench_configs_freeze_official_harness_and_resolved_rate(
    name: str,
    stage: str,
    split: str,
    sample_count: int,
    dataset_source: str,
) -> None:
    config = _config(name)

    _RUNNER.validate_completion_benchmark_config(config)

    bounded = config["swebench_evaluation"]
    assert config["experiment"]["phase"] == "swebench_evaluation"
    assert config["experiment"]["training_enabled"] is False
    assert bounded["dataset_key"] == "swe_bench"
    assert bounded["stage"] == stage
    assert bounded["split"] == split
    assert bounded["selection"] == "sequential"
    assert bounded["sample_count"] == sample_count
    assert bounded["rollouts_per_task"] == 1
    assert bounded["concurrency"] == 1
    assert bounded["official_metric"] == "resolved_rate"
    assert bounded["proxy_metrics_allowed"] is False
    assert config["data"]["catalog_path"] == "config/datasets_swebench.yaml"

    evaluation = config["evaluation"]
    assert evaluation["swebench_harness_enabled"] is True
    assert evaluation["swebench_evaluator_path"] == _SKILLFLOW_EVALUATOR
    assert evaluation["swebench_harness_path"] == _OFFICIAL_HARNESS
    assert evaluation["swebench_dataset_source"] == dataset_source
    assert evaluation["swebench_dataset_path"] == (
        "data/swebench_v2/validation.jsonl"
        if dataset_source == "regular_dev"
        else _OFFICIAL_VERIFIED
    )
    assert evaluation["swebench_docker_namespace"] == "swebench"
    assert evaluation["swebench_timeout_seconds"] == 900
    assert _RUNNER._BENCHMARKS["swe_bench"]["primary_metric"] == "resolved"


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("swebench_harness_enabled", False),
        ("swebench_evaluator_path", ""),
        ("swebench_harness_path", ""),
        ("swebench_dataset_path", ""),
        ("swebench_dataset_source", "unknown"),
        ("swebench_evaluation_root", ""),
        ("swebench_docker_namespace", ""),
        ("swebench_timeout_seconds", 0),
    ),
)
def test_swebench_config_fails_closed_without_official_harness(
    field: str,
    invalid: object,
) -> None:
    config = deepcopy(_config("evaluation_swebench_verified_round_01.yaml"))
    config["evaluation"][field] = invalid

    with pytest.raises(ConfigurationError, match=field):
        _RUNNER.validate_completion_benchmark_config(config)


def test_swebench_report_exposes_only_official_resolved_rate() -> None:
    config = _config("evaluation_swebench_verified_round_01.yaml")
    rows = [
        {
            "task_id": "swe-bench:owner__repo-1",
            "direct": {
                "available": True,
                "valid": True,
                "resolved": 0.0,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "resolved": 1.0,
                "explicit_finish": True,
            },
            "failure_type": "agentgraph_higher_resolved",
        }
    ]

    report = _RUNNER._report(rows, config)

    assert report["primary_metric"] == "resolved"
    assert report["metric_scope"] == (
        "SWE_bench_Verified_official_Docker_harness_resolved_rate"
    )
    assert report["direct_local_baseline"]["strict_resolved"] == 0.0
    assert report["agentgraph"]["strict_resolved"] == 1.0
    assert report["agentgraph_minus_direct"] == {"resolved": 1.0}
    assert not {
        "strict_accuracy",
        "strict_exact_match",
        "strict_raw_score",
        "strict_success",
    }.intersection(report["agentgraph"])


def test_regular_dev_report_keeps_its_dataset_source_distinct() -> None:
    config = _config("development_swebench_verified_round_01.yaml")
    report = _RUNNER._report([], config)

    assert report["benchmark_slice"] == "regular_dev"
    assert report["metric_scope"] == (
        "SWE_bench_regular_dev_official_Docker_harness_resolved_rate"
    )


def test_dataset_source_and_project_split_must_match() -> None:
    config = deepcopy(_config("evaluation_swebench_verified_round_01.yaml"))
    config["swebench_evaluation"]["split"] = "validation"

    with pytest.raises(ConfigurationError, match="dataset_source_split_isolation"):
        _RUNNER.validate_completion_benchmark_config(config)


def test_swebench_prepare_only_needs_neither_backend_nor_docker(
    tmp_path: Path,
) -> None:
    config = deepcopy(_config("evaluation_swebench_verified_round_01.yaml"))
    config["swebench_evaluation"]["sample_count"] = 1

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    selected_source = data_dir / "test.jsonl"
    task = TaskRecord(
        task_id="swe-bench:owner__repo-1",
        question="Fix the issue",
        ground_truth="",
        split="test",
        metadata={
            "dataset_key": "swe_bench",
            "dataset_source": "verified",
            "benchmark_slice": "verified",
            "evaluator_payload": {
                "instance_id": "owner__repo-1",
                "dataset_source": "verified",
            },
        },
    )
    selected_source.write_text(
        json.dumps(
            {
                "schema_version": "flowsteer.agentgraph.task.v1",
                **task.to_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config["data"].update(
        {
            "train_path": str(data_dir / "train.jsonl"),
            "validation_path": str(data_dir / "validation.jsonl"),
            "test_path": str(selected_source),
        }
    )

    output_dir = tmp_path / "output"
    storage_names = {
        "root": "evidence",
        "selected_tasks_path": "selected_tasks.jsonl",
        "direct_predictions_path": "direct_predictions.jsonl",
        "trajectories_path": "agentgraph_trajectories.jsonl",
        "failures_path": "collection_failures.jsonl",
        "paired_results_path": "paired_results.jsonl",
        "wrong_demos_path": "wrong_demos.jsonl",
        "manifest_path": "run_manifest.json",
        "preflight_receipt_path": "preflight_receipt.json",
        "report_json_path": "evaluation_report.json",
        "report_markdown_path": "evaluation_report.md",
    }
    for field, name in storage_names.items():
        config["storage"][field] = str(output_dir / name)

    config_path = tmp_path / "evaluation_swebench_verified_round_01.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch.object(
            _RUNNER.LiveSmokeBackend,
            "from_config",
            side_effect=AssertionError("prepare-only started the backend"),
        ) as backend_factory,
        patch.object(
            _RUNNER,
            "_attach_swebench_official_harness",
            side_effect=AssertionError("prepare-only touched Docker"),
        ) as attach_harness,
    ):
        manifest = asyncio.run(
            _RUNNER.run_completion_benchmark_round(
                config_path,
                project_root=_ROOT,
                prepare_only=True,
            )
        )

    assert manifest["status"] == "prepared"
    assert manifest["dataset_key"] == "swe_bench"
    assert manifest["selected_task_ids"] == ["swe-bench:owner__repo-1"]
    assert Path(config["storage"]["manifest_path"]).is_file()
    assert not Path(config["storage"]["preflight_receipt_path"]).exists()
    backend_factory.assert_not_called()
    attach_harness.assert_not_called()


def test_docker_preflight_failure_does_not_attach_swebench_harness() -> None:
    config = _config("evaluation_swebench_verified_round_01.yaml")
    backend = SimpleNamespace(swe_harness=None)
    task = TaskRecord(
        task_id="swe-bench:owner__repo-1",
        question="Fix the issue",
        ground_truth="",
        split="test",
        metadata={
            "dataset_key": "swe_bench",
            "dataset_source": "verified",
            "benchmark_slice": "verified",
            "evaluator_payload": {
                "instance_id": "owner__repo-1",
                "dataset_source": "verified",
            },
        },
    )
    harness = Mock()
    harness.preflight.side_effect = SWEbenchHarnessUnavailable(
        "official SWE-bench Docker harness is unavailable"
    )

    with patch.object(
        _RUNNER,
        "OfficialSWEbenchHarness",
        return_value=harness,
    ):
        with pytest.raises(SWEbenchHarnessUnavailable, match="Docker harness"):
            _RUNNER._attach_swebench_official_harness(
                backend,
                config,
                _ROOT,
                (task,),
            )

    harness.preflight.assert_called_once_with((task,))
    assert backend.swe_harness is None
