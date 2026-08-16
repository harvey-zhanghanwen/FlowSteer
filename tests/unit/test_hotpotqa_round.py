from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from src.interactive.config_loader import load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_hotpotqa_round.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_hotpotqa_round", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_round_config_is_fixed_heldout_and_training_disabled():
    config = load_yaml(_ROOT / "config" / "evaluation_hotpotqa_round_01.yaml")
    _MODULE.validate_hotpot_config(config)


def test_task_id_diagnostic_selection_is_explicit_and_bounded():
    config = load_yaml(
        _ROOT / "config" / "evaluation_hotpotqa_multiagent_v1_diagnostic.yaml"
    )
    _MODULE.validate_hotpot_config(config)

    invalid = deepcopy(config)
    invalid["hotpotqa_evaluation"]["task_ids"] = [
        invalid["hotpotqa_evaluation"]["task_ids"][0]
    ] * 14
    try:
        _MODULE.validate_hotpot_config(invalid)
    except Exception as exc:
        assert "task_ids selection" in str(exc)
    else:  # pragma: no cover - fail-closed guard
        raise AssertionError("duplicate task IDs were accepted")


def test_declared_direct_reuse_is_copied_without_gateway_call(tmp_path):
    task = _MODULE.TaskRecord(
        task_id="hotpotqa:one",
        question="question",
        ground_truth="answer",
        split="validation",
        metadata={"dataset_key": "hotpotqa"},
    )
    record = {
        "task_id": task.task_id,
        "model_id": "qwen3.5-9b-local",
        "protocol": "direct-v1",
        "generation_seed": 17,
        "evaluation": {"valid": True},
        "execution": {"execution_id": "existing"},
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    destination = tmp_path / "destination.jsonl"
    manifest_path = tmp_path / "manifest.json"
    config = {
        "experiment": {"name": "test", "seed": 17},
        "hotpotqa_evaluation": {
            "direct_model_id": "qwen3.5-9b-local",
            "direct_protocol": "direct-v1",
            "direct_reused_from": source.name,
            "concurrency": 1,
        },
    }
    manifest = {}

    result = asyncio.run(
        _MODULE._collect_direct(
            None,
            (task,),
            config,
            tmp_path,
            destination,
            [],
            manifest,
            manifest_path,
        )
    )

    assert result[task.task_id]["execution"]["execution_id"] == "existing"
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 1
    assert manifest["direct_progress"]["completed"] == 1
    assert manifest["direct_progress"]["reused_from"] == str(source)


def test_strict_aggregate_keeps_failed_task_in_denominator():
    rows = [
        {
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 0.8,
            }
        },
        {
            "agentgraph": {
                "available": False,
                "valid": False,
                "exact_match": 0.0,
                "token_f1": 0.0,
            }
        },
    ]

    result = _MODULE._aggregate(rows, "agentgraph")

    assert result["denominator"] == 2
    assert result["completed"] == 1
    assert result["evaluator_valid"] == 1
    assert result["strict_exact_match"] == 0.5
    assert result["strict_token_f1"] == 0.4
    assert result["completed_only_exact_match"] == 1.0
