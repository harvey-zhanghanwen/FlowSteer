#!/usr/bin/env python3
"""Run fixed progressive topology diagnostics for AIME 2026 or HealthBench.

This is an evaluation-only forced probe.  It reuses the progressive
``AgentWorkflowEnv.step``/``execute_on_edit`` execution boundary used by
``compare_triviaqa_progressive_topologies.py``, the fixed-graph call receipt
serialization used by ``compare_hotpotqa_topologies.py``, and the existing
completion-benchmark evaluator routing.  It never invokes Flow-Director and
never produces GRPO- or Skill-eligible evidence.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from compare_hotpotqa_topologies import _call_dict as _serialize_call
from compare_triviaqa_progressive_topologies import (
    _atomic_write_json,
    _response_usage,
)
from evaluate_completion_benchmark_round import (
    _attach_healthbench_reference_judge,
    _evaluate_prediction,
    _evaluation_section,
    validate_completion_benchmark_config,
)
from train_agentgraph_smoke import _dataset_key
from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_runtime import AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import (
    load_model_registry,
    load_yaml,
)
from src.interactive.openai_gateway import OpenAICompatibleGateway
from src.interactive.records import TaskRecord
from src.interactive.task_dataset import iter_task_records


SUPPORTED_DATASETS = ("aime_2026", "healthbench_professional")
SUPPORTED_TOPOLOGIES = ("serial", "fan_in", "complex_mixed")
PRIMARY_METRIC = {
    "aime_2026": "exact_match",
    "healthbench_professional": "raw_score",
}
SEMANTIC_OUTPUT_AGENT = {
    "serial": "verification",
    "fan_in": "synthesis",
    "complex_mixed": "synthesis",
}


def _add(
    agent_id: str,
    model_id: str,
    contract: str,
    role_family: str,
) -> AgentAction:
    return AgentAction(
        AgentActionType.ADD_AGENT,
        agent_id=agent_id,
        model_id=model_id,
        contract=contract,
        role_family=role_family,
    )


def _edge(source: str, target: str, *, reciprocal: bool = False) -> AgentAction:
    return AgentAction(
        AgentActionType.SET_RELATION,
        source_id=source,
        target_id=target,
        source_to_target=True,
        target_to_source=reciprocal,
    )


def _output_contract(dataset: str) -> tuple[str, str]:
    if dataset == "aime_2026":
        return (
            "Extract the routed verified integer and serialize exactly one non-empty "
            "<answer>...</answer> response without additional text.",
            "format",
        )
    return (
        "Present the routed verified clinical response clearly and completely. Preserve "
        "clinically important qualifications and do not introduce unsupported facts.",
        "presenter",
    )


def _dataset_terms(dataset: str) -> Mapping[str, str]:
    if dataset == "aime_2026":
        return {
            "analysis": (
                "Analyze the mathematical problem, identify the governing definitions, "
                "constraints, and a valid solution strategy. Return a derivation plan, not "
                "a task-level final answer."
            ),
            "solution": (
                "Use the routed analysis to derive the requested integer rigorously. Check "
                "algebra, counting, boundary cases, and arithmetic. Return the candidate "
                "integer with its derivation, without answer tags."
            ),
            "independent": (
                "Independently solve the mathematical problem using a second derivation or "
                "explicit check. Return one integer candidate and the decisive reasoning, "
                "without answer tags."
            ),
            "verification": (
                "Verify the routed mathematical candidate against the original constraints. "
                "Correct proof gaps or computational errors and return one verified integer "
                "without answer tags."
            ),
            "synthesis": (
                "Consume all routed mathematical artifacts, reconcile disagreements by "
                "checking the derivations, and return one verified integer without answer tags."
            ),
        }
    return {
        "analysis": (
            "Analyze the healthcare conversation and identify the requested clinical task, "
            "relevant patient facts, safety constraints, and missing information. Return a "
            "response plan, not a task-level final answer."
        ),
        "solution": (
            "Use the routed clinical analysis to draft a clear, evidence-informed response. "
            "Address the user's request and preserve appropriate uncertainty and safety "
            "qualifications."
        ),
        "independent": (
            "Independently formulate a clinically appropriate response to the conversation. "
            "Check factual completeness, safety, and whether the requested format is satisfied."
        ),
        "verification": (
            "Verify the routed clinical response for factual accuracy, safety, completeness, "
            "and fidelity to the conversation. Correct unsupported or omitted claims and "
            "return the verified response."
        ),
        "synthesis": (
            "Consume all routed clinical artifacts, resolve disagreements using the supplied "
            "patient context and established clinical guidance, and return one complete, "
            "well-qualified response."
        ),
    }


def topology_actions(
    dataset: str,
    topology: str,
    model_id: str,
) -> tuple[AgentAction, ...]:
    """Return the fixed Canvas action sequence for one diagnostic condition."""

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    terms = _dataset_terms(dataset)
    output_contract, output_role = _output_contract(dataset)

    if topology == "serial":
        return (
            _add("analysis", model_id, terms["analysis"], "analysis"),
            _add("reasoning", model_id, terms["solution"], "reasoning"),
            _add("verification", model_id, terms["verification"], "verification"),
            _add("output", model_id, output_contract, output_role),
            _edge("analysis", "reasoning"),
            _edge("reasoning", "verification"),
            _edge("verification", "output"),
            AgentAction(AgentActionType.SET_OUTPUT, agent_id="output"),
            AgentAction(AgentActionType.FINISH),
        )
    if topology == "fan_in":
        return (
            _add("primary_reasoning", model_id, terms["solution"], "reasoning"),
            _add("independent_check", model_id, terms["independent"], "verification"),
            _add("synthesis", model_id, terms["synthesis"], "synthesis"),
            _add("output", model_id, output_contract, output_role),
            _edge("primary_reasoning", "synthesis"),
            _edge("independent_check", "synthesis"),
            _edge("synthesis", "output"),
            AgentAction(AgentActionType.SET_OUTPUT, agent_id="output"),
            AgentAction(AgentActionType.FINISH),
        )
    if topology == "complex_mixed":
        return (
            _add("task_analysis", model_id, terms["analysis"], "analysis"),
            _add("primary_branch", model_id, terms["solution"], "reasoning"),
            _add("independent_branch", model_id, terms["independent"], "verification"),
            _add(
                "candidate_reasoning",
                model_id,
                terms["solution"]
                + " During reciprocal revision, inspect the peer draft and correct "
                "unsupported reasoning before returning the candidate.",
                "reasoning",
            ),
            _add(
                "critical_verification",
                model_id,
                terms["verification"]
                + " During reciprocal revision, inspect the peer draft and return a "
                "corrected verification artifact.",
                "verification",
            ),
            _add("synthesis", model_id, terms["synthesis"], "synthesis"),
            _add("output", model_id, output_contract, output_role),
            _edge("task_analysis", "primary_branch"),
            _edge("task_analysis", "independent_branch"),
            _edge("primary_branch", "candidate_reasoning"),
            _edge("independent_branch", "critical_verification"),
            _edge("candidate_reasoning", "critical_verification", reciprocal=True),
            _edge("candidate_reasoning", "synthesis"),
            _edge("critical_verification", "synthesis"),
            _edge("synthesis", "output"),
            AgentAction(AgentActionType.SET_OUTPUT, agent_id="output"),
            AgentAction(AgentActionType.FINISH),
        )
    raise ValueError(f"unsupported topology: {topology}")


def _plan_summary(actions: Sequence[AgentAction]) -> Mapping[str, Any]:
    agents = [
        str(action.agent_id)
        for action in actions
        if action.action_type is AgentActionType.ADD_AGENT
    ]
    relations = [
        action
        for action in actions
        if action.action_type is AgentActionType.SET_RELATION
    ]
    reciprocal_pairs = sum(bool(action.target_to_source) for action in relations)
    outputs = [
        action.agent_id
        for action in actions
        if action.action_type is AgentActionType.SET_OUTPUT
    ]
    if len(agents) != len(set(agents)):
        raise ValueError("fixed topology contains duplicate Agent IDs")
    known = set(agents)
    for relation in relations:
        if relation.source_id not in known or relation.target_id not in known:
            raise ValueError("fixed topology relation references an unknown Agent")
    if len(outputs) != 1 or outputs[0] not in known:
        raise ValueError("fixed topology must set exactly one known Output Agent")
    if not actions or actions[-1].action_type is not AgentActionType.FINISH:
        raise ValueError("fixed topology must end with FINISH")
    return {
        "agent_count": len(agents),
        "relation_count": len(relations),
        "reciprocal_pair_count": reciprocal_pairs,
        "canvas_edit_count": len(actions) - 1,
        "output_agent_id": outputs[0],
    }


def _execution_dict(execution: Any) -> Optional[dict[str, Any]]:
    if execution is None:
        return None
    calls = [_serialize_call(call) for call in execution.calls]
    return {
        "run_id": execution.run_id,
        "graph_revision": execution.graph_revision,
        "output_agent_id": execution.output_agent_id,
        "final_answer": execution.final_answer,
        "outputs": dict(execution.outputs),
        "block_completion_order": [
            list(block) for block in execution.block_completion_order
        ],
        "executed_agent_ids": list(execution.executed_agent_ids),
        "reused_agent_ids": list(execution.reused_agent_ids),
        "calls": calls,
    }


def _terminal_feedback_edit(
    environment: AgentWorkflowEnv,
    topology: str,
) -> AgentAction:
    agent_id = SEMANTIC_OUTPUT_AGENT[topology]
    node = environment.graph.get_node(agent_id)
    return AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id=agent_id,
        contract=(
            node.contract.rstrip()
            + " Terminal validation reported an empty routed answer. Re-run the assigned "
            "reasoning and return exactly one non-empty best-supported semantic answer; "
            "do not use answer tags."
        ),
    )


async def _run_condition(
    *,
    runtime: AgentRuntime,
    registry: Any,
    evaluator_backend: Any,
    task: TaskRecord,
    dataset: str,
    topology: str,
    model_id: str,
    run_id: str,
) -> Mapping[str, Any]:
    environment = AgentWorkflowEnv(
        registry,
        runtime=runtime,
        problem=task.question,
        execute_on_edit=True,
        max_agents=8,
        require_exact_answer_tag=dataset == "aime_2026",
        require_format_agent=dataset == "aime_2026",
    )
    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    terminal_execution = None
    actions = list(topology_actions(dataset, topology, model_id))
    action_index = 0
    terminal_feedback_edits = 0
    while action_index < len(actions):
        action = actions[action_index]
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
            if (
                dataset == "aime_2026"
                and action.action_type is AgentActionType.FINISH
                and terminal_feedback_edits == 0
                and "non_empty=False" in result.feedback
            ):
                recovery = _terminal_feedback_edit(environment, topology)
                recovery_result = await environment.step(recovery)
                steps.append(
                    {
                        "step_index": len(steps) + 1,
                        "action": recovery.to_dict(),
                        "accepted": recovery_result.accepted,
                        "done": recovery_result.done,
                        "graph_revision": recovery_result.revision,
                        "feedback": recovery_result.feedback,
                        "execution_reused": recovery_result.execution_reused,
                        "graph_snapshot": recovery_result.snapshot.graph.to_dict(),
                        "execution": _execution_dict(recovery_result.execution),
                    }
                )
                if not recovery_result.accepted:
                    raise RuntimeError(
                        f"{topology} terminal-feedback edit rejected: "
                        f"{recovery_result.feedback}"
                    )
                if recovery_result.execution is not None:
                    terminal_execution = recovery_result.execution
                terminal_feedback_edits += 1
                continue
            raise RuntimeError(
                f"{topology} Canvas edit {len(steps)} rejected: {result.feedback}"
            )
        if result.execution is not None:
            terminal_execution = result.execution
        action_index += 1

    if not environment.finished or terminal_execution is None:
        raise RuntimeError(f"{topology} did not reach explicit FINISH")
    final_answer = terminal_execution.final_answer or ""
    outcome = await _evaluate_prediction(evaluator_backend, task, final_answer)
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
    return {
        "topology": topology,
        "graph": environment.graph.snapshot().to_dict(),
        "topology_statistics": environment.graph.topology_statistics(),
        "plan": _plan_summary(actions),
        "explicit_finish": environment.finished,
        "final_answer": final_answer,
        "evaluation": asdict(outcome),
        "wall_latency_ms": (time.monotonic() - started) * 1000.0,
        "api_calls": len(calls),
        "api_attempts": api_attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "terminal_feedback_edit_count": terminal_feedback_edits,
        "steps": steps,
        "run_id": run_id,
    }


def _select_tasks(path: Path, dataset: str, task_ids: Sequence[str]) -> tuple[TaskRecord, ...]:
    by_id = {
        task.task_id: task
        for task in iter_task_records(path)
        if _dataset_key(task) == dataset
    }
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(
            f"task IDs are absent from {path}: " + ", ".join(missing)
        )
    return tuple(by_id[task_id] for task_id in task_ids)


def _checkpoint_metadata(
    *,
    dataset: str,
    task_ids: Sequence[str],
    topologies: Sequence[str],
    model_id: str,
    run_id: str,
) -> Mapping[str, Any]:
    return {
        "dataset": dataset,
        "task_ids": list(task_ids),
        "topologies": list(topologies),
        "model_id": model_id,
        "run_id": run_id,
    }


async def _run(args: argparse.Namespace) -> int:
    tasks_path = Path(args.tasks).expanduser()
    if not tasks_path.is_absolute():
        tasks_path = PROJECT_ROOT / tasks_path
    catalog_path = Path(args.catalog).expanduser()
    if not catalog_path.is_absolute():
        catalog_path = PROJECT_ROOT / catalog_path
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    config = load_yaml(config_path)
    validate_completion_benchmark_config(config)
    _, bounded = _evaluation_section(config)
    configured_dataset = str(bounded["dataset_key"])
    if configured_dataset != args.dataset:
        raise ValueError(
            f"config dataset {configured_dataset} does not match --dataset {args.dataset}"
        )
    registry = load_model_registry(catalog_path)
    registry.require_model(args.model_id)
    tasks = _select_tasks(tasks_path, args.dataset, args.task_ids)
    run_id = args.run_id or (
        f"{args.dataset}-completion-topology-{args.seed}"
    )
    metadata = _checkpoint_metadata(
        dataset=args.dataset,
        task_ids=args.task_ids,
        topologies=args.topologies,
        model_id=args.model_id,
        run_id=run_id,
    )
    plans = {
        topology: _plan_summary(
            topology_actions(args.dataset, topology, args.model_id)
        )
        for topology in args.topologies
    }
    if plans.get("serial", {}).get("agent_count") != plans.get(
        "fan_in", {}
    ).get("agent_count") and {"serial", "fan_in"}.issubset(args.topologies):
        raise ValueError("serial and fan_in must use the same Agent count")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    **metadata,
                    "tasks_path": str(tasks_path),
                    "catalog_path": str(catalog_path),
                    "config_path": str(config_path),
                    "output": str(output),
                    "plans": plans,
                    "api_calls": 0,
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
        default_temperature=0.0,
        default_top_p=1.0,
        default_seed=args.seed,
    )
    runtime = AgentRuntime(
        registry,
        gateway,
        max_concurrency=4,
        timeout_seconds=args.timeout,
    )
    evaluator_backend = SimpleNamespace(judge=None, judge_model="")
    judge_receipt = None
    if args.dataset == "healthbench_professional":
        judge_receipt = _attach_healthbench_reference_judge(
            evaluator_backend,
            config,
            PROJECT_ROOT,
        )

    checkpoint = output.with_suffix(output.suffix + ".checkpoint.json")
    checkpoint_conditions: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise ValueError("existing checkpoint belongs to a different diagnostic")
        raw_conditions = saved.get("conditions", {})
        if isinstance(raw_conditions, Mapping):
            checkpoint_conditions = {
                str(task_id): dict(value)
                for task_id, value in raw_conditions.items()
                if isinstance(value, Mapping)
            }

    results: list[dict[str, Any]] = []
    for task in tasks:
        conditions = checkpoint_conditions.setdefault(task.task_id, {})
        for topology in args.topologies:
            if topology in conditions:
                print(
                    f"{task.task_id} {topology}: resumed from checkpoint",
                    flush=True,
                )
                continue
            condition_run_id = f"{run_id}:{task.task_id}:{topology}"
            conditions[topology] = dict(
                await _run_condition(
                    runtime=runtime,
                    registry=registry,
                    evaluator_backend=evaluator_backend,
                    task=task,
                    dataset=args.dataset,
                    topology=topology,
                    model_id=args.model_id,
                    run_id=condition_run_id,
                )
            )
            _atomic_write_json(
                checkpoint,
                {
                    "schema_version": (
                        "flowsteer.completion_topology_checkpoint.v1"
                    ),
                    "status": "in_progress",
                    "metadata": metadata,
                    "conditions": checkpoint_conditions,
                },
            )
            metric = PRIMARY_METRIC[args.dataset]
            value = conditions[topology]["evaluation"]["metrics"].get(metric)
            print(
                f"{task.task_id} {topology}: {metric}={value}",
                flush=True,
            )
        results.append({"task": task.to_dict(), "conditions": conditions})

    primary_metric = PRIMARY_METRIC[args.dataset]
    aggregate: dict[str, Any] = {}
    for topology in args.topologies:
        conditions = [item["conditions"][topology] for item in results]
        valid = [item for item in conditions if item["evaluation"]["valid"]]
        aggregate[topology] = {
            "tasks": len(conditions),
            "evaluator_valid": len(valid),
            primary_metric: (
                sum(
                    float(item["evaluation"]["metrics"][primary_metric])
                    for item in valid
                )
                / len(valid)
                if valid
                else None
            ),
            "api_calls": sum(int(item["api_calls"]) for item in conditions),
            "api_attempts": sum(int(item["api_attempts"]) for item in conditions),
            "prompt_tokens": sum(int(item["prompt_tokens"]) for item in conditions),
            "completion_tokens": sum(
                int(item["completion_tokens"]) for item in conditions
            ),
            "wall_latency_ms": sum(
                float(item["wall_latency_ms"]) for item in conditions
            ),
        }

    payload = {
        "schema_version": "flowsteer.completion_topology_diagnostic.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "controls": {
            "diagnostic_only": True,
            "forced_probe": True,
            "grpo_eligible": False,
            "skill_evidence_eligible": False,
            "director_invoked": False,
            "training_enabled": False,
            "same_task_across_conditions": True,
            "same_model_for_all_agents": args.model_id,
            "same_temperature": 0.0,
            "same_top_p": 1.0,
            "same_seed": args.seed,
            "progressive_canvas_execute_on_edit": True,
            "serial_vs_fan_in_control": (
                "same Agent count and equal planned Canvas edit count"
            ),
            "complex_mixed_interpretation": (
                "capacity probe with additional Agents, relations, and reciprocal calls"
            ),
        },
        "evaluator": {
            "primary_metric": primary_metric,
            "healthbench_judge_receipt": judge_receipt,
        },
        "plans": plans,
        "aggregate": aggregate,
        "results": results,
    }
    _atomic_write_json(output, payload)
    _atomic_write_json(
        checkpoint,
        {
            "schema_version": "flowsteer.completion_topology_checkpoint.v1",
            "status": "complete",
            "metadata": metadata,
            "conditions": checkpoint_conditions,
            "final_output": str(output),
        },
    )
    print(
        json.dumps(
            {"output": str(output), "aggregate": aggregate},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=SUPPORTED_TOPOLOGIES,
        default=list(SUPPORTED_TOPOLOGIES),
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and fixed action plans without starting a model or API",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.task_ids) != len(set(args.task_ids)):
        raise SystemExit("--task-ids must be unique")
    if len(args.topologies) != len(set(args.topologies)):
        raise SystemExit("--topologies must be unique")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
