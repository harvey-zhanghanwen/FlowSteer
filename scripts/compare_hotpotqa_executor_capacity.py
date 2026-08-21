#!/usr/bin/env python3
"""Run a preregistered HotpotQA Executor-capacity diagnostic.

This evaluation-only forced probe isolates one Executor model behind a frozen
two-node AgentGraph::

    semantic_reasoning -> format -> FINISH

The graph is constructed through ``AgentWorkflowEnv.step`` with
``execute_on_edit=True``.  Each task therefore plans exactly one candidate
remote reasoning call and one fixed local Format call.  Flow-Director, Skill
injection, GRPO admission, training, and topology selection are all disabled.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# DIRECT_REUSE: the same checkpoint/usage helpers used by the existing
# progressive topology diagnostics.  Importing either module has no model/API
# side effects.
from compare_triviaqa_progressive_topologies import (  # noqa: E402
    _atomic_write_json,
    _response_usage,
)
from compare_hotpotqa_topologies import (  # noqa: E402
    FORMAT_CONTRACT,
    _agent,
    _component,
    _evaluation_dict,
    _execution_dict,
    _plan_summary,
    _relation,
)
from src.interactive.agent_action_parser import (  # noqa: E402
    AgentAction,
    AgentActionType,
)
from src.interactive.agent_runtime import AgentRuntime  # noqa: E402
from src.interactive.agent_workflow_env import AgentWorkflowEnv  # noqa: E402
from src.interactive.config_loader import load_model_registry  # noqa: E402
from src.interactive.openai_gateway import OpenAICompatibleGateway  # noqa: E402
from src.interactive.records import TaskRecord  # noqa: E402
from src.interactive.task_dataset import iter_task_records  # noqa: E402
from src.interactive.task_evaluator import evaluate_task  # noqa: E402


PREREGISTERED_TASK_IDS = (
    "hotpotqa:5ac4bfac5542997ea680caaa",
    "hotpotqa:5a78b59d554299148911f958",
    "hotpotqa:5ac0a765554299012d1db612",
    "hotpotqa:5abd15b955429933744ab718",
    "hotpotqa:5a8ccf10554299653c1aa129",
    "hotpotqa:5ac42df25542997ea680ca1d",
)
FIXED_SEED = 20260821
FIXED_TEMPERATURE = 0.0
FIXED_TOP_P = 1.0
PLANNED_CALLS_PER_TASK = 2


# THIN_ADAPTATION of FlowSteer's VERIFY prompt and the project's existing
# verified-candidate/singleton-Format Skill hypothesis.  The capacity probe has
# no upstream candidate, so the reasoning Executor derives and verifies one
# candidate directly from the model-visible question and supplied passages.
SEMANTIC_REASONING_CONTRACT = (
    "Using only the supplied question and passages, derive and verify exactly one "
    "semantic answer candidate. Check the requested subject, relation, answer "
    "type, qualifiers, aliases, and decisive evidence span. Preserve necessary "
    "names, modifiers, units, and dates. Return the candidate and decisive "
    "evidence without answer tags."
)


def capacity_actions(
    reasoning_model_id: str,
    format_model_id: str,
) -> tuple[AgentAction, ...]:
    """Return the frozen two-component progressive AgentGraph plan."""

    semantic = _component(
        _agent(
            "semantic_reasoning",
            reasoning_model_id,
            SEMANTIC_REASONING_CONTRACT,
            "reasoning",
            "verified_semantic_answer",
            "Exactly one evidence-grounded semantic answer candidate is returned.",
        )
    )
    singleton_format = _component(
        _agent(
            "format",
            format_model_id,
            FORMAT_CONTRACT,
            "format",
            "terminal_answer",
            "Exactly one non-empty <answer>...</answer> artifact is returned.",
        ),
        relations=(_relation("semantic_reasoning", "format"),),
        output_agent_id="format",
    )
    return semantic, singleton_format, AgentAction(AgentActionType.FINISH)


def _select_preregistered_tasks(
    path: Path,
    *,
    expected_split: str,
    task_ids: Sequence[str],
) -> tuple[TaskRecord, ...]:
    if tuple(task_ids) != PREREGISTERED_TASK_IDS:
        raise ValueError(
            "--task-ids must exactly match the preregistered capacity panel"
        )
    records = tuple(iter_task_records(path, expected_split=expected_split))
    by_id = {task.task_id: task for task in records}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError("task IDs absent from source: " + ", ".join(missing))
    selected = tuple(by_id[task_id] for task_id in task_ids)
    for task in selected:
        dataset = str(task.metadata.get("dataset_key") or "").casefold()
        native_split = str(task.metadata.get("native_split") or "").casefold()
        if dataset != "hotpotqa" or native_split != "train":
            raise ValueError(
                f"{task.task_id} is not a preregistered native-train HotpotQA task"
            )
    return selected


def _checkpoint_metadata(
    *,
    tasks_path: Path,
    tasks: Sequence[TaskRecord],
    expected_split: str,
    catalog_path: Path,
    registry: Any,
    reasoning_model_id: str,
    format_model_id: str,
    run_id: str,
) -> Mapping[str, Any]:
    actions = capacity_actions(reasoning_model_id, format_model_id)
    plan = _plan_summary(actions)
    if plan["planned_model_calls"] != PLANNED_CALLS_PER_TASK:
        raise ValueError(
            "frozen capacity plan must contain exactly two planned model calls"
        )
    return {
        "dataset": "hotpotqa",
        "partition": expected_split,
        "tasks_path": str(tasks_path),
        "task_ids": [task.task_id for task in tasks],
        "task_selection": "exact_preregistered_native_train_panel_v1",
        "catalog_path": str(catalog_path),
        # Freeze the complete non-secret catalog and exact selected arms.  The
        # ModelRegistry never resolves credentials into this representation.
        "catalog": registry.to_dict(),
        "reasoning_model": registry.require_model(reasoning_model_id).to_dict(),
        "format_model": registry.require_model(format_model_id).to_dict(),
        "reasoning_model_id": reasoning_model_id,
        "format_model_id": format_model_id,
        "seed": FIXED_SEED,
        "temperature": FIXED_TEMPERATURE,
        "top_p": FIXED_TOP_P,
        "run_id": run_id,
        "planned_model_calls_per_task": PLANNED_CALLS_PER_TASK,
        "plan": plan,
    }


async def _run_task(
    *,
    runtime: AgentRuntime,
    registry: Any,
    task: TaskRecord,
    reasoning_model_id: str,
    format_model_id: str,
    run_id: str,
) -> Mapping[str, Any]:
    """Execute one task; evaluator-only fields are read after generation."""

    actions = capacity_actions(reasoning_model_id, format_model_id)
    environment = AgentWorkflowEnv(
        registry,
        runtime=runtime,
        # Leakage boundary: only the aligned model-visible question and its
        # supplied passages enter AgentRuntime.
        problem=task.question,
        execute_on_edit=True,
        max_agents=2,
        max_agents_per_subgraph=3,
        require_exact_answer_tag=True,
        require_format_agent=True,
    )
    steps: list[dict[str, Any]] = []
    terminal_execution = None
    started = time.monotonic()
    try:
        for action in actions:
            result = await environment.step(action)
            execution = _execution_dict(result.execution)
            steps.append(
                {
                    "step_index": len(steps) + 1,
                    "action": action.to_dict(),
                    "accepted": result.accepted,
                    "done": result.done,
                    "graph_revision": result.revision,
                    "feedback": result.feedback,
                    "execution_reused": result.execution_reused,
                    "graph_snapshot": result.snapshot.graph.to_dict(),
                    "execution": execution,
                }
            )
            if not result.accepted:
                raise RuntimeError(
                    f"Canvas action {len(steps)} rejected: {result.feedback}"
                )
            if (
                action.action_type is AgentActionType.ADD_SUBGRAPH
                and result.execution is None
            ):
                raise RuntimeError(
                    "accepted Canvas edit did not produce its required "
                    f"execute_on_edit receipt: {result.feedback}"
                )
            if result.execution is not None:
                terminal_execution = result.execution
        if not environment.finished or terminal_execution is None:
            raise RuntimeError("capacity probe did not reach explicit FINISH")
        final_answer = terminal_execution.final_answer or ""

        # Ground truth and evaluator payload are consumed only after both model
        # calls and explicit FINISH have completed.
        outcome = await evaluate_task(task, final_answer)
        evaluation: Optional[Mapping[str, Any]] = _evaluation_dict(outcome)
        status = "completed"
        failure = None
    except Exception as exc:  # fixed-denominator operational failure
        final_answer = ""
        evaluation = None
        status = "operational_failure"
        failure = {"type": type(exc).__name__, "message": str(exc)}

    calls = [
        call
        for step in steps
        if not step["execution_reused"]
        for call in ((step.get("execution") or {}).get("calls") or [])
    ]
    prompt_tokens = completion_tokens = api_attempts = 0
    for call in calls:
        prompt, completion, attempts = _response_usage(call["response_metadata"])
        prompt_tokens += prompt
        completion_tokens += completion
        api_attempts += attempts
    observed_models = [call["model_id"] for call in calls]
    expected_models = [reasoning_model_id, format_model_id]
    if status == "completed" and observed_models != expected_models:
        status = "operational_failure"
        evaluation = None
        failure = {
            "type": "CallBudgetMismatch",
            "message": (
                f"observed model calls {observed_models!r}; expected "
                f"{expected_models!r}"
            ),
        }
    return {
        "status": status,
        "operational_failure": status != "completed",
        "failure": failure,
        "explicit_finish": environment.finished,
        "final_answer": final_answer,
        "evaluation": evaluation,
        "graph": environment.graph.snapshot().to_dict(),
        "topology_statistics": environment.graph.topology_statistics(),
        "plan": _plan_summary(actions),
        "wall_latency_ms": (time.monotonic() - started) * 1000.0,
        "api_calls": len(calls),
        "api_attempts": api_attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "observed_model_call_order": observed_models,
        "steps": steps,
        "run_id": run_id,
    }


def _metric(row: Mapping[str, Any], name: str) -> float:
    evaluation = row.get("evaluation")
    if not isinstance(evaluation, Mapping) or not evaluation.get("valid"):
        return 0.0
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    return float(metrics.get(name) or 0.0)


def _aggregate(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    denominator = len(results)
    return {
        "denominator": denominator,
        "completed": sum(item.get("status") == "completed" for item in results),
        "operational_failures": sum(
            bool(item.get("operational_failure")) for item in results
        ),
        "evaluator_valid": sum(
            isinstance(item.get("evaluation"), Mapping)
            and bool(item["evaluation"].get("valid"))
            for item in results
        ),
        # Strict fixed denominator: operational/evaluator failures contribute 0.
        "strict_exact_match": (
            sum(_metric(item, "exact_match") for item in results) / denominator
            if denominator
            else None
        ),
        "strict_token_f1": (
            sum(_metric(item, "token_f1") for item in results) / denominator
            if denominator
            else None
        ),
        "api_calls": sum(int(item.get("api_calls") or 0) for item in results),
        "api_attempts": sum(
            int(item.get("api_attempts") or 0) for item in results
        ),
        "prompt_tokens": sum(
            int(item.get("prompt_tokens") or 0) for item in results
        ),
        "completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in results
        ),
        "wall_latency_ms": sum(
            float(item.get("wall_latency_ms") or 0.0) for item in results
        ),
    }


async def _run(args: argparse.Namespace) -> int:
    tasks_path = Path(args.tasks).expanduser()
    if not tasks_path.is_absolute():
        tasks_path = PROJECT_ROOT / tasks_path
    catalog_path = Path(args.catalog).expanduser()
    if not catalog_path.is_absolute():
        catalog_path = PROJECT_ROOT / catalog_path
    output = (
        Path(args.output).expanduser()
        if args.output
        else Path(
            "artifacts/hotpotqa_executor_capacity/"
            f"{args.reasoning_model_id}__{args.format_model_id}.json"
        )
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.json")

    registry = load_model_registry(catalog_path)
    reasoning_model = registry.require_model(args.reasoning_model_id)
    format_model = registry.require_model(args.format_model_id)
    reasoning_provider = registry.require_provider(reasoning_model.provider_id)
    format_provider = registry.require_provider(format_model.provider_id)
    if reasoning_provider.provider_id == "local-director":
        raise ValueError("--reasoning-model-id must select a remote Executor")
    if format_provider.provider_id != "local-director":
        raise ValueError("--format-model-id must select the fixed local Executor")

    tasks = _select_preregistered_tasks(
        tasks_path,
        expected_split=args.expected_split,
        task_ids=args.task_ids,
    )
    metadata = _checkpoint_metadata(
        tasks_path=tasks_path,
        tasks=tasks,
        expected_split=args.expected_split,
        catalog_path=catalog_path,
        registry=registry,
        reasoning_model_id=args.reasoning_model_id,
        format_model_id=args.format_model_id,
        run_id=args.run_id,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "api_calls_planned": len(tasks) * PLANNED_CALLS_PER_TASK,
                    "remote_api_calls_planned": len(tasks),
                    "local_model_calls_planned": len(tasks),
                    "api_calls_executed": 0,
                    "metadata": metadata,
                    "controls": _controls(),
                    "output": str(output),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    gateway = OpenAICompatibleGateway(
        timeout_seconds=args.timeout,
        max_retries=0,
        default_temperature=FIXED_TEMPERATURE,
        default_top_p=FIXED_TOP_P,
        default_seed=FIXED_SEED,
    )
    runtime = AgentRuntime(
        registry,
        gateway,
        max_concurrency=1,
        timeout_seconds=args.timeout,
        dataset_id="hotpotqa",
    )

    checkpoint_results: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise ValueError("existing checkpoint belongs to a different probe")
        raw_results = saved.get("results")
        if isinstance(raw_results, Mapping):
            checkpoint_results = {
                str(task_id): dict(value)
                for task_id, value in raw_results.items()
                if isinstance(value, Mapping)
            }

    results: list[dict[str, Any]] = []
    for task in tasks:
        result = checkpoint_results.get(task.task_id)
        if result is None:
            result = dict(
                await _run_task(
                    runtime=runtime,
                    registry=registry,
                    task=task,
                    reasoning_model_id=args.reasoning_model_id,
                    format_model_id=args.format_model_id,
                    run_id=f"{args.run_id}:{task.task_id}",
                )
            )
            checkpoint_results[task.task_id] = result
            _atomic_write_json(
                checkpoint,
                {
                    "schema_version": (
                        "flowsteer.hotpotqa.executor_capacity_checkpoint.v1"
                    ),
                    "status": "in_progress",
                    "metadata": metadata,
                    "results": checkpoint_results,
                },
            )

        # Labels are persisted only after generation and evaluation.  Task
        # metadata (including supporting facts/native type) is intentionally
        # excluded from this diagnostic result.
        results.append(
            {
                "task": {
                    "task_id": task.task_id,
                    "split": task.split,
                    "question": task.question,
                    "ground_truth": task.ground_truth,
                },
                **result,
            }
        )

    payload = {
        "schema_version": "flowsteer.hotpotqa.executor_capacity_diagnostic.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "controls": _controls(),
        "source_mapping": {
            "canvas_execution": (
                "src/interactive/agent_workflow_env.py::AgentWorkflowEnv.step"
            ),
            "runtime": "src/interactive/agent_runtime.py::AgentRuntime",
            "format_boundary": "scripts/prompts/prompt.py::FORMAT_PROMPT",
            "checkpoint_receipts": (
                "scripts/compare_triviaqa_progressive_topologies.py"
            ),
            "hotpot_evaluator": "src/interactive/task_evaluator.py::evaluate_task",
        },
        "aggregate": _aggregate(results),
        "results": results,
    }
    _atomic_write_json(output, payload)
    _atomic_write_json(
        checkpoint,
        {
            "schema_version": (
                "flowsteer.hotpotqa.executor_capacity_checkpoint.v1"
            ),
            "status": "complete",
            "metadata": metadata,
            "results": checkpoint_results,
            "final_output": str(output),
        },
    )
    print(
        json.dumps(
            {"output": str(output), "aggregate": payload["aggregate"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _controls() -> Mapping[str, Any]:
    return {
        "diagnostic_only": True,
        "forced_probe": True,
        "director_invoked": False,
        "director_off": True,
        "training_enabled": False,
        "grpo_eligible": False,
        "skill_evidence_eligible": False,
        "skill_injection_performed": False,
        "progressive_canvas_execute_on_edit": True,
        "functional_component_action": "add_subgraph",
        "fixed_topology": "semantic_reasoning_to_singleton_format",
        "reasoning_calls_per_task": 1,
        "format_calls_per_task": 1,
        "planned_model_calls_per_task": PLANNED_CALLS_PER_TASK,
        "temperature": FIXED_TEMPERATURE,
        "top_p": FIXED_TOP_P,
        "seed": FIXED_SEED,
        "retrieval_tools_enabled": False,
        "ground_truth_visibility": "evaluator_after_generation_only",
        "supporting_facts_visible_during_generation": False,
        "native_question_type_used_for_generation": False,
        "failure_denominator_policy": "retain_as_zero_in_strict_metrics",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="data/joint_qa_v2/skill_confirmation_round7.jsonl",
    )
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=list(PREREGISTERED_TASK_IDS),
        help="Exact preregistered native-train HotpotQA capacity panel.",
    )
    parser.add_argument(
        "--catalog",
        default="config/model_catalog_hotpotqa_deep_v6.yaml",
    )
    parser.add_argument("--reasoning-model-id", required=True)
    parser.add_argument("--format-model-id", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--run-id",
        default="hotpotqa-executor-capacity-diagnostic-v1",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze tasks, catalog, and actions without starting a model/API.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if len(args.task_ids) != len(set(args.task_ids)):
        raise SystemExit("--task-ids must be unique")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
