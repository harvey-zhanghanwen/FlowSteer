"""Offline arm selection at the existing completion collector boundary.

These synthetic receipts exercise no model, Tool, grader or GPU service. All
collection remains delegated to the pre-existing Direct/AgentGraph functions.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.unit.test_completion_benchmark_round import _evaluation_config


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "healthbench_parallel_collection_test_runner",
    ROOT / "scripts" / "evaluate_completion_benchmark_round.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@contextmanager
def collection_fixture(directory):
    config = _evaluation_config("healthbench_professional")
    bounded = config["healthbench_professional_evaluation"]
    bounded.update(sample_count=2, concurrency=2, task_timeout_seconds=123.0)
    selected = tuple(
        RUNNER.TaskRecord(
            task_id=f"healthbench-professional:synthetic-{index}",
            question=json.dumps([{"role": "user", "content": f"Synthetic conversation {index}."}]),
            ground_truth="", split="validation",
            metadata={"dataset_key": "healthbench_professional"},
        )
        for index in range(2)
    )
    paths = {name: directory / f"{name}.json" for name in (
        "selected", "direct", "trajectories", "failures", "paired", "wrong",
        "manifest", "preflight", "report_json", "report_markdown",
    )}
    direct = {
        task.task_id: {
            "task_id": task.task_id, "final_answer": "Synthetic complete response.",
            "evaluation": {"valid": True, "metrics": {
                "overall_score": 0.6, "overall_score_length_adjusted": 0.55,
            }},
        }
        for task in selected
    }
    trajectories = {
        task.task_id: {
            **deepcopy(direct[task.task_id]), "task": task.to_dict(),
            "trajectory_id": f"trajectory-{index}", "explicit_finish": True,
            "termination_reason": "finish", "steps": [],
        }
        for index, task in enumerate(selected)
    }
    backend = SimpleNamespace(
        config=config, judge_model="synthetic-grader",
        publisher=SimpleNamespace(
            server_runtime_receipt=Mock(return_value={}),
            ensure_loaded_adapter=Mock(return_value={"success": True}),
        ),
    )
    direct_collector = AsyncMock(return_value=direct)
    graph_collector = AsyncMock(return_value=trajectories)
    paired = Mock(return_value=[{
        "task_id": task.task_id,
        "agentgraph": {"overall_score_length_adjusted": 0.55},
    } for task in selected])
    report = Mock(return_value={
        "operational_failure_count": 0, "terminal_failure_count": 0,
        "direct_local_baseline": {"strict_overall_score_length_adjusted": 0.55},
        "agentgraph": {"strict_overall_score_length_adjusted": 0.55},
        "agentgraph_minus_direct": {"overall_score_length_adjusted": 0.0},
        "failure_types": {},
    })
    stable_zero = Mock(return_value={"passed": True})
    existing_graph = Mock(return_value=trajectories)
    with ExitStack() as stack:
        mocks = {
            "load_yaml": Mock(return_value=config),
            "_validate_runtime_dataset_registry": Mock(return_value={"enabled": False}),
            "_paths": Mock(return_value=paths),
            "_select_tasks": Mock(return_value=selected),
            "_healthbench_direct_reference": Mock(return_value=None),
            "_git_state": Mock(return_value={"branch": "synthetic"}),
            "_attach_healthbench_reference_judge": Mock(return_value={}),
            "_run_evaluator_preflight": AsyncMock(return_value={"passed": True}),
            "_collect_direct": direct_collector,
            "_collect_graph": graph_collector,
            "_paired_rows": paired,
            "_report": report,
            "_report_markdown": Mock(return_value="Synthetic paired report."),
            "_completion_stable_zero_check": stable_zero,
            "_existing_trajectory_checkpoint": existing_graph,
        }
        for name, mock in mocks.items():
            stack.enter_context(patch.object(RUNNER, name, mock))
        stack.enter_context(patch.object(RUNNER.LiveSmokeBackend, "from_config", return_value=backend))
        yield SimpleNamespace(
            config=config, bounded=bounded, selected=selected, paths=paths,
            backend=backend, direct=direct_collector, graph=graph_collector,
            paired=paired, report=report, stable_zero=stable_zero,
            existing_graph=existing_graph,
        )


def run(directory, **kwargs):
    return asyncio.run(RUNNER.run_completion_benchmark_round(
        directory / "synthetic.yaml", project_root=directory, **kwargs,
    ))


@pytest.mark.parametrize("arm", ["direct", "agentgraph"])
def test_single_arm_calls_only_its_existing_collector_without_paired_results(tmp_path, arm):
    with collection_fixture(tmp_path) as fixture:
        manifest = run(tmp_path, collection_arm=arm)
        called = fixture.direct if arm == "direct" else fixture.graph
        uncalled = fixture.graph if arm == "direct" else fixture.direct
        called.assert_awaited_once()
        uncalled.assert_not_awaited()
        fixture.paired.assert_not_called()
        fixture.report.assert_not_called()
        fixture.stable_zero.assert_not_called()
        fixture.existing_graph.assert_not_called()
        assert not fixture.paths["paired"].exists()
        assert not fixture.paths["wrong"].exists()
        assert manifest["collection_arm"] == arm
        assert set(manifest["metrics"]) == {arm}
        assert manifest["sample_count"] == 2
        assert manifest["selected_task_ids"] == [task.task_id for task in fixture.selected]
        assert manifest["training_enabled"] is False
        assert manifest["optimizer_updates"] == 0
        # The other arm was not measured: do not turn it into an invented zero.
        other = "agentgraph" if arm == "direct" else "direct"
        assert other not in manifest["metrics"]
        assert "delta" not in manifest["metrics"]
        assert manifest["collection_arm_completed"] is True
        assert manifest["metrics"][arm]["strict_overall_score_length_adjusted"] == 0.55
        saved = json.loads(fixture.paths["report_json"].read_text())
        assert saved["paired_comparison_available"] is False
        assert set(saved["metrics"]) == {arm}


@pytest.mark.parametrize("arm", ["direct", "agentgraph"])
def test_missing_collection_result_does_not_become_a_measured_zero(tmp_path, arm):
    with collection_fixture(tmp_path) as fixture:
        collector = fixture.direct if arm == "direct" else fixture.graph
        del collector.return_value[fixture.selected[1].task_id]
        manifest = run(tmp_path, collection_arm=arm)
        assert manifest["collection_arm_completed"] is False
        assert manifest["collection_admitted_count"] == 1
        score = manifest["metrics"][arm]
        assert score["denominator"] == 2
        assert score["evaluator_valid"] == 1
        assert score["strict_overall_score"] is None
        assert score["strict_overall_score_length_adjusted"] is None
        assert score["completed_only_overall_score_length_adjusted"] == 0.55
        fixture.paired.assert_not_called()


@pytest.mark.parametrize("arm", ["direct", "agentgraph"])
def test_single_arm_preserves_selected_tasks_config_and_existing_budgets(tmp_path, arm):
    with collection_fixture(tmp_path) as fixture:
        run(tmp_path, collection_arm=arm)
        call = fixture.direct.await_args if arm == "direct" else fixture.graph.await_args
        assert call.args[0] is fixture.backend
        assert call.args[1] is fixture.selected
        passed_config = call.args[2]
        assert passed_config["healthbench_professional_evaluation"] == fixture.bounded
        assert passed_config["healthbench_professional_evaluation"]["concurrency"] == 2
        assert passed_config["healthbench_professional_evaluation"]["task_timeout_seconds"] == 123.0
        assert passed_config["director"] == fixture.config["director"]
        assert passed_config["experiment"] == fixture.config["experiment"]
        if arm == "direct":
            assert passed_config is fixture.config
            assert call.args[4] == fixture.paths["direct"]
        else:
            assert passed_config["hotpotqa_evaluation"] is fixture.bounded
            assert call.args[3] == fixture.paths["trajectories"]
            assert call.kwargs["failure_path"] == fixture.paths["failures"]


@pytest.mark.parametrize("explicit", [False, True])
def test_default_and_explicit_both_keep_original_paired_workflow(tmp_path, explicit):
    with collection_fixture(tmp_path) as fixture:
        manifest = run(tmp_path, **({"collection_arm": "both"} if explicit else {}))
        fixture.direct.assert_awaited_once()
        fixture.graph.assert_awaited_once()
        fixture.paired.assert_called_once()
        fixture.report.assert_called_once()
        fixture.stable_zero.assert_called_once()
        assert fixture.paths["paired"].exists()
        assert set(manifest["metrics"]) == {"direct", "agentgraph", "delta"}


def test_existing_direct_only_semantics_still_reuse_graph_checkpoint(tmp_path):
    with collection_fixture(tmp_path) as fixture:
        run(tmp_path, direct_only=True)
        fixture.direct.assert_awaited_once()
        fixture.graph.assert_not_awaited()
        fixture.existing_graph.assert_called_once()
        fixture.paired.assert_called_once()
        fixture.report.assert_called_once()


@pytest.mark.parametrize("arm", ["direct", "agentgraph"])
def test_single_arm_cannot_be_mixed_with_legacy_direct_only_before_any_preflight(tmp_path, arm):
    with patch.object(RUNNER, "load_yaml") as load:
        with pytest.raises(RUNNER.CompletionBenchmarkRoundError):
            run(tmp_path, collection_arm=arm, direct_only=True)
        load.assert_not_called()


@pytest.mark.parametrize("arm", ["direct", "agentgraph"])
def test_single_arm_rejects_paired_canary_before_any_preflight(tmp_path, arm):
    with patch.object(RUNNER, "load_yaml") as load:
        with pytest.raises(RUNNER.CompletionBenchmarkRoundError):
            run(tmp_path, collection_arm=arm, canary_only=True)
        load.assert_not_called()


@pytest.mark.parametrize("arm", ["direct", "agentgraph"])
def test_prepare_only_can_freeze_each_arm_without_collection(tmp_path, arm):
    with collection_fixture(tmp_path) as fixture:
        manifest = run(tmp_path, collection_arm=arm, prepare_only=True)
        fixture.direct.assert_not_awaited()
        fixture.graph.assert_not_awaited()
        fixture.paired.assert_not_called()
        fixture.report.assert_not_called()
        assert manifest["status"] == "prepared"
        assert manifest["collection_arm"] == arm
        assert manifest["selected_task_ids"] == [task.task_id for task in fixture.selected]
        assert "metrics" not in manifest


def test_invalid_collection_arm_is_rejected_before_any_preflight(tmp_path):
    with patch.object(RUNNER, "load_yaml") as load:
        with pytest.raises(RUNNER.CompletionBenchmarkRoundError):
            run(tmp_path, collection_arm="not-an-arm")
        load.assert_not_called()


def test_cli_defaults_to_both_and_only_accepts_declared_collection_arms():
    parser = RUNNER.build_parser()
    assert parser.parse_args(["--config", "synthetic.yaml"]).collection_arm == "both"
    for arm in ("direct", "agentgraph", "both"):
        assert parser.parse_args(["--config", "synthetic.yaml", "--collection-arm", arm]).collection_arm == arm
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "synthetic.yaml", "--collection-arm", "invalid"])


def test_cli_forwards_selected_arm_without_starting_another_collector(capsys):
    completed = {
        "status": "completed", "dataset_key": "healthbench_professional",
        "sample_count": 2, "artifacts": {"manifest": "synthetic-manifest.json"},
    }
    runner = AsyncMock(return_value=completed)
    with patch.object(RUNNER, "run_completion_benchmark_round", runner):
        assert RUNNER.main(["--config", "synthetic.yaml", "--collection-arm", "agentgraph"]) == 0
    runner.assert_awaited_once()
    assert runner.await_args.kwargs["collection_arm"] == "agentgraph"
    assert runner.await_args.kwargs["direct_only"] is False
    assert json.loads(capsys.readouterr().out)["sample_count"] == 2
