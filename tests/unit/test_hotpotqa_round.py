from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from src.interactive.config_loader import load_yaml
from src.interactive.records import EvaluationReceipt


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
    stale = dict(record)
    stale["generation_seed"] = 99
    stale["execution"] = {"execution_id": "stale-canary"}
    destination.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    config = {
        "experiment": {"name": "test", "seed": 99},
        "hotpotqa_evaluation": {
            "direct_model_id": "qwen3.5-9b-local",
            "direct_protocol": "direct-v1",
            "direct_generation_seed": 17,
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
    assert result[task.task_id]["reuse_receipt"] == {
        "reused": True,
        "source": str(source),
    }
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 1
    assert manifest["direct_progress"]["completed"] == 1
    assert manifest["direct_progress"]["reused_from"] == str(source)
    assert manifest["direct_progress"]["reused_records"] == 1
    assert manifest["direct_progress"]["newly_collected_records"] == 0


def test_graph_evaluation_uses_task_local_rollout_zero(tmp_path):
    tasks = tuple(
        _MODULE.TaskRecord(
            task_id=f"hotpotqa:{index}",
            question=f"question {index}",
            ground_truth="answer",
            split="validation",
            metadata={"dataset_key": "hotpotqa"},
        )
        for index in range(3)
    )

    class EmptyTrajectoryStore:
        def payloads(self):
            return ()

    class Backend:
        model_catalog_version = "catalog-v1"
        evidence_store = type("Evidence", (), {"trajectories": EmptyTrajectoryStore()})()

        def __init__(self):
            self.rollout_indices = []

        async def collect(
            self,
            task,
            rollout_index,
            versions,
            *,
            expected_task_split="train",
        ):
            assert expected_task_split == "validation"
            self.rollout_indices.append(rollout_index)
            return _MODULE.TrajectoryRecord(
                trajectory_id=f"trajectory:{task.task_id}",
                task=task,
                group_id=f"{task.task_id}:condition:{versions.policy}",
                condition_id="condition",
                rollout_id=f"{task.task_id}:rollout:0000",
                versions=versions,
                turns=(),
                final_answer="answer",
                evaluation=EvaluationReceipt(
                    versions.evaluator,
                    True,
                    1.0,
                    metrics={"exact_match": 1.0, "token_f1": 1.0},
                ),
                termination_reason="finish",
                explicit_finish=True,
            )

    backend = Backend()
    config = {
        "experiment": {
            "condition_id": "condition",
            "prompt_version": "prompt-v1",
            "tool_version": "tool-v1",
        },
        "director": {"behavior_policy_version": "policy-v1"},
        "hotpotqa_evaluation": {"concurrency": 2},
    }
    manifest_path = tmp_path / "manifest.json"
    result = asyncio.run(
        _MODULE._collect_graph(
            backend,
            tasks,
            config,
            tmp_path / "trajectories.jsonl",
            [],
            {},
            manifest_path,
        )
    )

    assert set(result) == {task.task_id for task in tasks}
    assert backend.rollout_indices == [0, 0, 0]


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


def test_report_counts_terminal_failure_without_dropping_evaluator_result():
    rows = [
        {
            "task_id": "hotpotqa:one",
            "direct": {
                "available": True,
                "valid": True,
                "exact_match": 1.0,
                "token_f1": 1.0,
            },
            "agentgraph": {
                "available": True,
                "valid": True,
                "exact_match": 0.0,
                "token_f1": 0.0,
                "explicit_finish": False,
                "termination_reason": "max_rounds",
            },
            "failure_type": "architecture_regression_candidate",
        }
    ]
    config = {
        "experiment": {"name": "terminal-failure"},
        "director": {
            "behavior_policy_version": "policy",
            "behavior_adapter_name": "adapter",
        },
        "agent_graph": {"model_catalog_path": "catalog.yaml"},
    }

    report = _MODULE._report(rows, config)

    assert report["terminal_failure_count"] == 1
    assert report["explicit_finished_count"] == 0
    assert report["operational_failure_count"] == 0
    assert report["agentgraph"]["completed"] == 1
    assert report["agentgraph"]["evaluator_valid"] == 1
    assert report["agentgraph"]["strict_exact_match"] == 0.0
