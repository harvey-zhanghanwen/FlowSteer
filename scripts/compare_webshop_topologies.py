#!/usr/bin/env python3
"""Compare fixed AgentGraph topologies with the formal WebShop environment.

This is an evaluation-only diagnostic forced probe.  It reuses the existing
AgentGraph/AgentRuntime execution path and delegates every environment reset,
transition, and terminal score to
``LiveSmokeBackend.evaluate_final_graph`` and the deployed SkillFlow
``RAGENAdapter``.  It never invokes the Director, trains a model, or admits a
result to GRPO or Skill evidence.

The three controlled conditions use the same task, model, seed, environment
step limit, and sequential reset boundary.  Serial and fan-in each contain
three singleton Agents and therefore use three model calls per environment
step.  Reciprocal is one bounded two-Agent exchange and therefore uses two
parallel draft calls followed by two parallel revision calls per environment
step.  The unequal call budget is recorded explicitly and reciprocal must not
be interpreted as a call-count-matched comparison.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# Reuse the fixed-topology receipt/checkpoint helpers from the existing
# progressive TriviaQA diagnostic rather than introducing another format.
from compare_triviaqa_progressive_topologies import (  # noqa: E402
    _atomic_write_json,
    _execution_dict,
    _response_usage,
)
from train_agentgraph_smoke import LiveSmokeBackend  # noqa: E402
from src.interactive.agent_graph import (  # noqa: E402
    AgentGraph,
    AgentNode,
    AgentRelation,
)
from src.interactive.agent_runtime import AgentRuntime  # noqa: E402
from src.interactive.config_loader import load_model_registry  # noqa: E402
from src.interactive.openai_gateway import OpenAICompatibleGateway  # noqa: E402
from src.interactive.records import TaskRecord  # noqa: E402
from src.interactive.task_dataset import iter_task_records  # noqa: E402


TOPOLOGIES = ("serial", "fan_in", "reciprocal")
CALLS_PER_ENVIRONMENT_STEP = {
    "serial": 3,
    "fan_in": 3,
    "reciprocal": 4,
}


class _ReceiptRecordingRuntime:
    """Record results while delegating execution to the canonical runtime."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self.executions: list[Any] = []

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._runtime.execute(*args, **kwargs)
        self.executions.append(result)
        return result


def _node(
    agent_id: str,
    model_id: str,
    contract: str,
    role_family: str,
) -> AgentNode:
    return AgentNode(
        agent_id,
        model_id,
        contract,
        role_family=role_family,
    )


def _fixed_graph(topology: str, model_id: str) -> AgentGraph:
    """Return one task-generic WebShop step-policy graph."""

    if topology == "serial":
        nodes = (
            _node(
                "state_analysis",
                model_id,
                "Read the shopping goal, current observation, recent interaction "
                "history, and admissible actions. Extract the unresolved product "
                "constraints and the current environment state. Return only a compact "
                "state artifact for the next Agent; do not issue the final action.",
                "state_analysis",
            ),
            _node(
                "action_planning",
                model_id,
                "Consume the routed state artifact and select the best next action from "
                "the current admissible-action list. Check product attributes, options, "
                "price, and navigation state. Return one action candidate plus a brief "
                "justification for the Output Agent; do not add alternatives.",
                "planning",
            ),
            _node(
                "action_output",
                model_id,
                "Consume the routed action candidate, verify that it is currently "
                "admissible, and return exactly one executable WebShop action with no "
                "explanation.",
                "operator",
            ),
        )
        relations = (
            AgentRelation("state_analysis", "action_planning", True, False),
            AgentRelation("action_planning", "action_output", True, False),
        )
        output_agent_id = "action_output"
    elif topology == "fan_in":
        nodes = (
            _node(
                "goal_constraints",
                model_id,
                "Independently track the requested product attributes, required options, "
                "and price bound. Compare them with the current observation and return "
                "a compact constraint-satisfaction artifact; do not issue the final action.",
                "constraint_tracking",
            ),
            _node(
                "environment_state",
                model_id,
                "Independently inspect the current observation, recent interaction "
                "history, and admissible actions. Identify the safest useful navigation, "
                "option-selection, or purchase action and return it as a candidate artifact.",
                "state_analysis",
            ),
            _node(
                "action_output",
                model_id,
                "Consume both routed artifacts. Resolve any conflict against the shopping "
                "goal and current admissible-action list, then return exactly one executable "
                "WebShop action with no explanation.",
                "operator",
            ),
        )
        relations = (
            AgentRelation("goal_constraints", "action_output", True, False),
            AgentRelation("environment_state", "action_output", True, False),
        )
        output_agent_id = "action_output"
    elif topology == "reciprocal":
        nodes = (
            _node(
                "navigator",
                model_id,
                "Propose the best next WebShop action from the current observation, recent "
                "history, shopping constraints, and admissible actions. In revision, inspect "
                "the peer draft and correct navigation, option, attribute, or price errors.",
                "planning",
            ),
            _node(
                "action_output",
                model_id,
                "Independently verify the best next WebShop action. In revision, inspect "
                "the navigator draft, resolve conflicts against the current admissible-action "
                "list and shopping goal, and return exactly one executable action with no "
                "explanation.",
                "operator",
            ),
        )
        relations = (
            AgentRelation("navigator", "action_output", True, True),
        )
        output_agent_id = "action_output"
    else:  # pragma: no cover - argparse and caller constrain the value
        raise ValueError(f"unsupported topology: {topology}")
    return AgentGraph(nodes, relations, output_agent_id=output_agent_id)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _select_tasks(path: Path, task_ids: Sequence[str]) -> tuple[TaskRecord, ...]:
    requested = tuple(task_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("--task-ids must be unique")
    found: dict[str, TaskRecord] = {}
    requested_set = set(requested)
    for task in iter_task_records(path):
        if task.task_id in requested_set:
            if task.metadata.get("dataset_key") != "webshop":
                raise ValueError(f"task is not WebShop: {task.task_id}")
            found[task.task_id] = task
    missing = [task_id for task_id in requested if task_id not in found]
    if missing:
        raise ValueError("task IDs not found: " + ", ".join(missing))
    return tuple(found[task_id] for task_id in requested)


def _backend(
    registry: Any,
    runtime: _ReceiptRecordingRuntime,
    max_environment_steps: int,
) -> LiveSmokeBackend:
    """Bind the canonical WebShop evaluator to a receipt-recording runtime."""

    return LiveSmokeBackend(
        config={
            "evaluation": {
                "max_environment_steps": max_environment_steps,
                "max_environment_steps_by_source": {
                    "webshop": max_environment_steps,
                },
            }
        },
        registry=registry,
        runtime=runtime,  # type: ignore[arg-type]
        director_client=SimpleNamespace(),
        rollout_gate=SimpleNamespace(),
        evidence_store=None,  # type: ignore[arg-type]
        trainer=None,
        publisher=SimpleNamespace(),
        judge=None,
        judge_model="",
    )


def _condition_identity(
    *,
    tasks_path: Path,
    catalog_path: Path,
    task_ids: Sequence[str],
    topologies: Sequence[str],
    model_id: str,
    max_environment_steps: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "tasks_path": str(tasks_path.resolve()),
        "catalog_path": str(catalog_path.resolve()),
        "task_ids": list(task_ids),
        "topologies": list(topologies),
        "model_id": model_id,
        "max_environment_steps": max_environment_steps,
        "seed": seed,
    }


def _checkpoint_payload(
    identity: Mapping[str, Any],
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    status: str,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "flowsteer.webshop.topology_checkpoint.v1",
        "status": status,
        "identity": dict(identity),
        "conditions": {
            task_id: dict(value) for task_id, value in conditions.items()
        },
    }
    if output is not None:
        payload["final_output"] = str(output)
    return payload


async def _run_condition(
    *,
    registry: Any,
    gateway: OpenAICompatibleGateway,
    task: TaskRecord,
    topology: str,
    model_id: str,
    max_environment_steps: int,
    rollout_index: int,
) -> dict[str, Any]:
    graph = _fixed_graph(topology, model_id)
    validation = graph.validate(registry, require_complete=True)
    validation.raise_if_invalid()
    canonical_runtime = AgentRuntime(
        registry,
        gateway,
        max_concurrency=4,
        timeout_seconds=gateway.timeout_seconds,
    )
    recording_runtime = _ReceiptRecordingRuntime(canonical_runtime)
    backend = _backend(registry, recording_runtime, max_environment_steps)
    started = time.monotonic()
    outcome = await backend.evaluate_final_graph(
        task,
        "",
        graph.snapshot().to_dict(),
        rollout_index=rollout_index,
    )
    wall_latency_ms = (time.monotonic() - started) * 1000.0
    executions = [
        value
        for result in recording_runtime.executions
        for value in (_execution_dict(result),)
        if value is not None
    ]
    calls = [call for execution in executions for call in execution["calls"]]
    prompt_tokens = completion_tokens = api_attempts = 0
    for call in calls:
        prompt, completion, attempts = _response_usage(call["response_metadata"])
        prompt_tokens += prompt
        completion_tokens += completion
        api_attempts += attempts
    environment_steps = int(float(outcome.metrics.get("steps", 0.0)))
    calls_per_step = CALLS_PER_ENVIRONMENT_STEP[topology]
    return {
        "status": "completed",
        "topology": topology,
        "graph": graph.snapshot().to_dict(),
        "topology_statistics": graph.topology_statistics(),
        "evaluation": asdict(outcome),
        "environment_steps": environment_steps,
        "model_calls_per_environment_step": calls_per_step,
        "expected_model_calls_for_observed_steps": environment_steps
        * calls_per_step,
        "maximum_model_call_budget": max_environment_steps * calls_per_step,
        "api_calls": len(calls),
        "api_attempts": api_attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_latency_ms": wall_latency_ms,
        "executions": executions,
    }


def _aggregate(
    tasks: Sequence[TaskRecord],
    topologies: Sequence[str],
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for topology in topologies:
        values = [conditions.get(task.task_id, {}).get(topology) for task in tasks]
        completed = [
            value
            for value in values
            if isinstance(value, Mapping) and value.get("status") == "completed"
        ]
        evaluator_valid = [
            value
            for value in completed
            if value.get("evaluation", {}).get("valid") is True
        ]
        success_count = sum(
            float(value.get("evaluation", {}).get("metrics", {}).get("success", 0.0))
            for value in evaluator_valid
        )
        aggregate[topology] = {
            "tasks": len(tasks),
            "completed": len(completed),
            "evaluator_valid": len(evaluator_valid),
            "strict_success_rate": success_count / len(tasks),
            "model_calls_per_environment_step": CALLS_PER_ENVIRONMENT_STEP[topology],
            "api_calls": sum(int(value.get("api_calls", 0)) for value in completed),
            "api_attempts": sum(
                int(value.get("api_attempts", 0)) for value in completed
            ),
            "prompt_tokens": sum(
                int(value.get("prompt_tokens", 0)) for value in completed
            ),
            "completion_tokens": sum(
                int(value.get("completion_tokens", 0)) for value in completed
            ),
            "environment_steps": sum(
                int(value.get("environment_steps", 0)) for value in completed
            ),
            "failed_or_missing": len(tasks) - len(completed),
        }
    return aggregate


async def _run(args: argparse.Namespace) -> int:
    tasks_path = _resolve(args.tasks)
    catalog_path = _resolve(args.catalog)
    output = _resolve(args.output)
    checkpoint = output.with_suffix(".checkpoint.json")
    tasks = _select_tasks(tasks_path, args.task_ids)
    registry = load_model_registry(catalog_path)
    registry.require_model(args.model_id)
    identity = _condition_identity(
        tasks_path=tasks_path,
        catalog_path=catalog_path,
        task_ids=args.task_ids,
        topologies=args.topologies,
        model_id=args.model_id,
        max_environment_steps=args.max_environment_steps,
        seed=args.seed,
    )
    conditions: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise ValueError(
                "existing topology checkpoint belongs to a different fixed condition"
            )
        raw_conditions = saved.get("conditions", {})
        if isinstance(raw_conditions, Mapping):
            conditions = {
                str(task_id): dict(value)
                for task_id, value in raw_conditions.items()
                if isinstance(value, Mapping)
            }

    gateway = OpenAICompatibleGateway(
        timeout_seconds=args.timeout,
        max_retries=0,
        default_temperature=0.0,
        default_top_p=1.0,
        default_seed=args.seed,
    )
    for task_index, task in enumerate(tasks):
        task_conditions = conditions.setdefault(task.task_id, {})
        for topology_index, topology in enumerate(args.topologies):
            if topology in task_conditions:
                print(
                    f"{task.task_id} {topology}: resumed from checkpoint",
                    flush=True,
                )
                continue
            try:
                task_conditions[topology] = await _run_condition(
                    registry=registry,
                    gateway=gateway,
                    task=task,
                    topology=topology,
                    model_id=args.model_id,
                    max_environment_steps=args.max_environment_steps,
                    rollout_index=task_index * len(TOPOLOGIES) + topology_index,
                )
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                task_conditions[topology] = {
                    "status": "failed",
                    "topology": topology,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "model_calls_per_environment_step": (
                        CALLS_PER_ENVIRONMENT_STEP[topology]
                    ),
                    "maximum_model_call_budget": (
                        args.max_environment_steps
                        * CALLS_PER_ENVIRONMENT_STEP[topology]
                    ),
                }
            _atomic_write_json(
                checkpoint,
                _checkpoint_payload(identity, conditions, status="in_progress"),
            )
            value = task_conditions[topology]
            evaluation = value.get("evaluation", {})
            metrics = evaluation.get("metrics", {})
            print(
                f"{task.task_id} {topology}: status={value['status']} "
                f"valid={evaluation.get('valid')} success={metrics.get('success')} "
                f"calls={value.get('api_calls')}",
                flush=True,
            )

    aggregate = _aggregate(tasks, args.topologies, conditions)
    failed = sum(
        int(value.get("status") != "completed")
        for task in tasks
        for topology in args.topologies
        for value in (conditions.get(task.task_id, {}).get(topology, {}),)
    )
    payload = {
        "schema_version": "flowsteer.webshop.fixed_topology_diagnostic.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if failed == 0 else "complete_with_failures",
        "selection": {
            "split": sorted({task.split for task in tasks}),
            "method": "explicit_fixed_task_ids",
            "task_ids": [task.task_id for task in tasks],
        },
        "controls": {
            "diagnostic_only": True,
            "forced_probe": True,
            "grpo_eligible": False,
            "skill_evidence_eligible": False,
            "director_invoked": False,
            "training_enabled": False,
            "sequential_condition_execution": True,
            "fresh_environment_reset_per_condition": True,
            "same_task": True,
            "same_model_for_all_agents": args.model_id,
            "same_temperature": 0.0,
            "same_top_p": 1.0,
            "same_seed": args.seed,
            "same_environment_step_limit": args.max_environment_steps,
            "formal_evaluator": "skillflow.ragen_adapter.v2",
            "call_budget_control": {
                "serial": "three singleton Agent calls per environment step",
                "fan_in": "three singleton Agent calls per environment step",
                "reciprocal": (
                    "two draft plus two revision calls per environment step; "
                    "not call-count-matched to serial/fan-in"
                ),
            },
        },
        "identity": identity,
        "tasks": [task.to_dict() for task in tasks],
        "aggregate": aggregate,
        "conditions": conditions,
    }
    _atomic_write_json(output, payload)
    _atomic_write_json(
        checkpoint,
        _checkpoint_payload(
            identity,
            conditions,
            status=payload["status"],
            output=output,
        ),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "checkpoint": str(checkpoint),
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="data/webshop_v2/train.jsonl",
        help="aligned WebShop JSONL path",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        required=True,
        help="one or more explicit fixed WebShop task IDs",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=TOPOLOGIES,
        default=list(TOPOLOGIES),
        help="fixed topology conditions, executed sequentially in the given order",
    )
    parser.add_argument(
        "--catalog",
        default="config/model_catalog_hotpotqa_deep_v6.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument("--max-environment-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/webshop_round_04/topology_probe/"
            "fixed_topology_comparison.json"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_environment_steps < 1:
        parser.error("--max-environment-steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if len(set(args.topologies)) != len(args.topologies):
        parser.error("--topologies must not contain duplicates")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
