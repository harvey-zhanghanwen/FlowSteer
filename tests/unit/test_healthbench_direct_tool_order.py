"""Regression for sorted registry IDs versus frozen Direct tool-list order.

Reuse the real ToolRegistry and Direct generation/resume identity functions.
Only the Agent runtime, external evaluator and execution-record fixture are
mocked: no corpus, model, network request or GPU service is loaded.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.interactive.config_loader import load_model_registry
from src.interactive.task_evaluator import EvaluationOutcome
from src.interactive.tool_runtime import ToolRegistration, ToolRegistry
from tests.unit.test_completion_benchmark_round import _healthbench_authoritative_config


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "healthbench_direct_tool_order_test_runner",
    ROOT / "scripts" / "evaluate_completion_benchmark_round.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
TOOLS = (
    "healthbench-authoritative.search",
    "healthbench-medrag.search",
    "healthbench-source.read",
    "healthbench-drug.lookup",
    "healthbench-computation.calculator",
)


@contextmanager
def direct_fixture(registered=TOOLS, configured=TOOLS):
    config = _healthbench_authoritative_config()
    config["experiment"]["seed"] = 29
    bounded = config["healthbench_professional_evaluation"]
    bounded.update(direct_allowed_tools=list(configured), direct_generation_seed=29)
    registry = load_model_registry(ROOT / "config" / "model_catalog_hotpotqa_deep_v6.yaml")
    tools = ToolRegistry(tuple(ToolRegistration(tool_id, Mock()) for tool_id in registered))
    task = RUNNER.TaskRecord(
        task_id="healthbench-professional:synthetic-tool-order",
        question="Conversation:\n\n[user] Synthetic conversation; no medical recommendation.\n\n[assistant]",
        ground_truth="", split="validation",
        metadata={"dataset_key": "healthbench_professional"},
    )
    observed = {}
    close = Mock()

    async def execute(graph, problem, *, run_id):
        node = graph.nodes[0]
        observed.update(graph=graph, problem=problem, node=node)
        coordinate = observed["sampling_coordinate"]
        seed = RUNNER.derive_generation_seed(
            base_seed=29, coordinate=coordinate, step_index=1,
            phase=RUNNER.GenerationPhase.ACTION,
        )
        # Retain the real frozen sampling receipt checks, as in the existing
        # test_healthbench_direct_react_is_one_agent_with_same_medrag_registry.
        identity = RUNNER._healthbench_direct_generation_identity(
            backend, task, model_id=bounded["direct_model_id"],
            protocol=bounded["direct_protocol"], contract=bounded["direct_contract"],
            seed=29, coordinate=coordinate,
        )
        requested = {**identity["scientific_sampling"]["requested_sampling"], "seed": seed}
        scientific_requested = {key: value for key, value in requested.items()
                                if key != "repetition_penalty"}
        metadata = {
            "model_calls": [{
                "turn": 1, "request_id": "synthetic-tool-order:1",
                "request_status": "completed", "algorithm": RUNNER.SCIENTIFIC_SAMPLING_ALGORITHM,
                "scientific_sampling": {
                    "algorithm": RUNNER.SCIENTIFIC_SAMPLING_ALGORITHM,
                    "base_seed": 29, "coordinate": coordinate.to_value(),
                    "phase": "action", "step_index": 1, "generation_seed": seed,
                    "requested_sampling": scientific_requested,
                },
                "requested_sampling": requested,
                "metadata": {"generation_seed": seed, "requested_sampling": requested},
            }],
            "tool_receipts": [],
        }
        call = SimpleNamespace(response=SimpleNamespace(metadata=metadata))
        return SimpleNamespace(
            final_answer="Synthetic complete response.", calls=(call,),
            run_id=run_id, output_agent_id=node.id,
            block_completion_order=((node.id,),), executed_agent_ids=(node.id,),
        )

    runtime = SimpleNamespace(execute=AsyncMock(side_effect=execute))

    def runtime_for_task(runtime_task, *, condition_id, sampling_base_seed, sampling_coordinate):
        observed.update(runtime_task=runtime_task, condition_id=condition_id,
                        sampling_base_seed=sampling_base_seed, sampling_coordinate=sampling_coordinate)
        return runtime, tools, close

    backend = SimpleNamespace(
        registry=registry, config=config, judge_model="synthetic-grader",
        _runtime_for_task=Mock(side_effect=runtime_for_task),
    )
    evaluate = AsyncMock(return_value=EvaluationOutcome(
        valid=True, reward=0.65,
        metrics={"overall_score": 0.7, "overall_score_length_adjusted": 0.65},
        reason="synthetic offline fixture",
        evaluator_version=RUNNER.evaluator_version_for(task),
        details={"judge_model": backend.judge_model},
    ))

    def execution_from_call(call):
        return SimpleNamespace(to_dict=lambda: {
            "output": "Synthetic complete response.",
            "metadata": {"response": dict(call.response.metadata)},
        })

    with ExitStack() as stack:
        stack.enter_context(patch.object(RUNNER, "_evaluate_prediction", evaluate))
        stack.enter_context(patch.object(RUNNER, "execution_record_from_call", side_effect=execution_from_call))
        yield SimpleNamespace(
            config=config, bounded=bounded, backend=backend, task=task,
            tools=tools, runtime=runtime, close=close, observed=observed,
            evaluate=evaluate,
        )


def generate(fixture):
    return asyncio.run(RUNNER._direct_one(
        fixture.backend, fixture.task, 0,
        model_id=fixture.bounded["direct_model_id"],
        protocol=fixture.bounded["direct_protocol"],
        contract=fixture.bounded["direct_contract"],
        seed=29, run_label=fixture.config["experiment"]["name"],
    ))


def test_same_five_tools_in_different_order_execute_and_preserve_both_receipts():
    with direct_fixture() as fixture:
        assert fixture.tools.resource_ids == tuple(sorted(TOOLS))
        assert fixture.tools.resource_ids != TOOLS
        result = generate(fixture)
        fixture.runtime.execute.assert_awaited_once()
        fixture.evaluate.assert_awaited_once()
        fixture.close.assert_called_once()
        assert result["final_answer"] == "Synthetic complete response."
        assert result["tool_resource_ids"] == list(TOOLS)
        assert result["registered_tool_resource_ids"] == list(fixture.tools.resource_ids)
        assert result["direct_generation_identity"]["tool"]["resource_ids"] == list(TOOLS)
        assert result["scientific_sampling_receipt"]["verified"] is True
        assert RUNNER._persisted_healthbench_direct_identity_matches(
            result, result["direct_generation_identity"],
        )
        assert set(fixture.observed["node"].allowed_tools) == set(TOOLS)
        assert len(fixture.observed["graph"].nodes) == 1


@pytest.mark.parametrize("registered", [TOOLS[:-1], TOOLS + ("unexpected-tool.search",)])
def test_missing_or_extra_tool_is_rejected_before_runtime_execution(registered):
    with direct_fixture(registered=registered) as fixture:
        with pytest.raises(RUNNER.CompletionBenchmarkRoundError, match="Tool resources differ"):
            generate(fixture)
        fixture.runtime.execute.assert_not_awaited()
        fixture.evaluate.assert_not_awaited()
        fixture.close.assert_called_once()


def test_duplicate_tool_in_config_is_not_hidden_by_set_comparison():
    with direct_fixture(configured=TOOLS + (TOOLS[0],)) as fixture:
        with pytest.raises(RUNNER.CompletionBenchmarkRoundError, match="Tool resources differ"):
            generate(fixture)
        fixture.runtime.execute.assert_not_awaited()
        fixture.close.assert_called_once()


def test_already_sorted_frozen_config_remains_compatible():
    configured = tuple(sorted(TOOLS))
    with direct_fixture(configured=configured) as fixture:
        result = generate(fixture)
        assert result["tool_resource_ids"] == list(configured)
        assert result["registered_tool_resource_ids"] == list(configured)
        fixture.runtime.execute.assert_awaited_once()


def test_collect_direct_reuses_new_receipt_without_generation_or_regrading(tmp_path):
    with direct_fixture() as fixture:
        generated = generate(fixture)
        predictions = tmp_path / "direct.jsonl"
        manifest_path = tmp_path / "manifest.json"
        RUNNER._atomic_jsonl(predictions, [generated])
        fixture.evaluate.reset_mock()
        fixture.runtime.execute.reset_mock()
        fixture.backend._runtime_for_task.reset_mock()
        failures = []
        manifest = {}
        with patch.object(RUNNER, "_direct_one", new_callable=AsyncMock) as direct_one:
            reused = asyncio.run(RUNNER._collect_direct(
                fixture.backend, (fixture.task,), fixture.config, tmp_path,
                predictions, failures, manifest, manifest_path,
            ))
            direct_one.assert_not_awaited()
        fixture.evaluate.assert_not_awaited()
        fixture.runtime.execute.assert_not_awaited()
        fixture.backend._runtime_for_task.assert_not_called()
        assert set(reused) == {fixture.task.task_id}
        assert reused[fixture.task.task_id]["tool_resource_ids"] == list(TOOLS)
        assert reused[fixture.task.task_id]["registered_tool_resource_ids"] == list(sorted(TOOLS))
        assert manifest["direct_progress"]["completed"] == 1
        assert manifest["direct_progress"]["pending_evaluator_retries"] == 0
        assert failures == []
