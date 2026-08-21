#!/usr/bin/env python3
"""Run a fixed-budget progressive HotpotQA topology diagnostic.

This is an evaluation-only forced probe.  Every condition is built from
functional ``ADD_SUBGRAPH`` transactions through FlowSteer's existing
``AgentWorkflowEnv.step`` / ``execute_on_edit`` boundary.  It never invokes
Flow-Director, trains a policy, or emits GRPO- or Skill-eligible evidence.

The three conditions have the same planned six model calls:

* ``serial``: Decompose -> serial evidence -> Aggregate -> Verify -> Format;
* ``fan_in``: independent evidence branches for every question;
* ``finite_reciprocal``: Decompose -> bounded candidate/verifier draft and
  peer revision -> Format.

The component contracts are thin AgentGraph adaptations of FlowSteer's
Decompose, Verify, Aggregate, and Format operator prompt semantics.  They are
task-agnostic and never contain a reference answer or supporting-fact label.
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

# DIRECT_REUSE: progressive checkpoint/usage helpers from the existing
# TriviaQA topology diagnostic.  Importing the module has no model/API side
# effects.
from compare_triviaqa_progressive_topologies import (  # noqa: E402
    _atomic_write_json,
    _response_usage,
)
from src.interactive.agent_action_parser import (  # noqa: E402
    AgentAction,
    AgentActionType,
    AgentSpec,
    RelationSpec,
)
from src.interactive.agent_runtime import AgentRuntime  # noqa: E402
from src.interactive.agent_workflow_env import AgentWorkflowEnv  # noqa: E402
from src.interactive.config_loader import load_model_registry  # noqa: E402
from src.interactive.openai_gateway import OpenAICompatibleGateway  # noqa: E402
from src.interactive.records import TaskRecord  # noqa: E402
from src.interactive.task_dataset import iter_task_records  # noqa: E402
from src.interactive.task_evaluator import evaluate_task  # noqa: E402


CONDITIONS = ("serial", "fan_in", "finite_reciprocal")
LATIN_SQUARE = (
    CONDITIONS,
    (CONDITIONS[1], CONDITIONS[2], CONDITIONS[0]),
    (CONDITIONS[2], CONDITIONS[0], CONDITIONS[1]),
)
PLANNED_MODEL_CALLS = 6


# THIN_ADAPTATION of scripts/prompts/prompt.py::DECOMPOSE_PROMPT.  The source
# asks for manageable sub-problems, independent where possible, that lead to
# the final solution.  HotpotQA adds only its evidence-artifact contract.
DECOMPOSE_CONTRACT = (
    "Break the question into the smallest evidence sub-problems that lead to the "
    "answer. State the requested answer type, entities, relation, qualifiers, and "
    "whether the evidence sub-problems are independent or sequential. Return an "
    "evidence plan only; do not return a task-level final answer."
)

# THIN_ADAPTATION of FlowSteer's evidence stages plus Aggregate prompt: consume
# routed candidates and select/compose the answer supported by their reasoning.
SERIAL_FIRST_EVIDENCE_CONTRACT = (
    "Resolve the first evidence objective from the routed decomposition and supplied "
    "passages. Return its subject, relation, answer, qualifiers, source title, and "
    "supporting span; do not return a task-level final answer."
)
SERIAL_SECOND_EVIDENCE_CONTRACT = (
    "Use the routed first evidence artifact to resolve the next sequential bridge or "
    "comparison objective from the supplied passages. Preserve entity identity, "
    "relation, qualifiers, and the exact supporting span; do not use answer tags."
)
LEFT_EVIDENCE_CONTRACT = (
    "Independently resolve the first evidence objective from the routed decomposition "
    "and supplied passages. Return a subject-relation-answer-qualifier proposition, "
    "source title, and supporting span; do not use answer tags."
)
RIGHT_EVIDENCE_CONTRACT = (
    "Independently resolve the second evidence objective from the routed decomposition "
    "and supplied passages. Return a subject-relation-answer-qualifier proposition, "
    "source title, and supporting span; do not use answer tags."
)
AGGREGATE_CONTRACT = (
    "Consume every routed evidence artifact and aggregate them into exactly one "
    "supported semantic answer candidate. Resolve entity and relation mismatches from "
    "the supplied passages, preserve necessary names, modifiers, units, and dates, "
    "and return the candidate with its decisive evidence span without answer tags."
)

# THIN_ADAPTATION of scripts/prompts/prompt.py::VERIFY_PROMPT: independently
# check before comparing with the proposed answer, then return a corrected one.
VERIFY_CONTRACT = (
    "Independently verify the routed candidate against the original question and "
    "supplied passages before comparing conclusions. Check subject, relation, answer "
    "type, qualifiers, aliases, and exact evidence span. Return one corrected verified "
    "semantic answer candidate and decisive evidence without answer tags."
)
RECIPROCAL_CANDIDATE_CONTRACT = (
    "Use the routed decomposition and supplied passages to derive one source-grounded "
    "semantic answer candidate. During reciprocal revision, inspect the verifier's "
    "independent draft, correct entity or relation drift, and preserve the supported "
    "answer span. Return the candidate and evidence without answer tags."
)
RECIPROCAL_VERIFY_CONTRACT = (
    "Independently solve and verify the evidence objectives from the routed "
    "decomposition. During reciprocal revision, compare the candidate peer draft with "
    "your independent result, aggregate consistent evidence, and correct subject, "
    "relation, answer-type, qualifier, or span errors. Return exactly one verified "
    "semantic answer candidate and evidence without answer tags."
)

# THIN_ADAPTATION of scripts/prompts/prompt.py::FORMAT_PROMPT.  AgentRuntime's
# canonical Format path supplies the full upstream extraction prompt; this node
# contract retains only the same no-reasoning boundary.
FORMAT_CONTRACT = (
    "The solution has already been computed in one routed upstream artifact. Extract "
    "that semantic answer without re-solving or verifying it, preserve its exact "
    "non-math surface form, and serialize only it in one <answer>...</answer> tag."
)


def _agent(
    agent_id: str,
    model_id: str,
    contract: str,
    role_family: str,
    artifact_type: str,
    completion_condition: str,
) -> AgentSpec:
    return AgentSpec(
        agent_id=agent_id,
        model_id=model_id,
        contract=contract,
        role_family=role_family,
        allowed_tools=(),
        execution_mode="reasoning",
        artifact_type=artifact_type,
        completion_condition=completion_condition,
    )


def _relation(
    source_id: str,
    target_id: str,
    *,
    reciprocal: bool = False,
) -> RelationSpec:
    return RelationSpec(
        source_id=source_id,
        target_id=target_id,
        source_to_target=True,
        target_to_source=reciprocal,
    )


def _component(
    *agents: AgentSpec,
    relations: Sequence[RelationSpec] = (),
    output_agent_id: Optional[str] = None,
) -> AgentAction:
    return AgentAction(
        AgentActionType.ADD_SUBGRAPH,
        agents=tuple(agents),
        relations=tuple(relations),
        output_agent_id=output_agent_id,
    )


def _decompose_component(model_id: str) -> AgentAction:
    return _component(
        _agent(
            "decompose",
            model_id,
            DECOMPOSE_CONTRACT,
            "decompose",
            "evidence_plan",
            "A task-faithful evidence plan identifies answer type and dependencies.",
        )
    )


def _serial_actions(model_id: str) -> tuple[AgentAction, ...]:
    return (
        _decompose_component(model_id),
        _component(
            _agent(
                "evidence_1",
                model_id,
                SERIAL_FIRST_EVIDENCE_CONTRACT,
                "evidence",
                "evidence_proposition",
                "The first evidence objective has a cited proposition and span.",
            ),
            _agent(
                "evidence_2",
                model_id,
                SERIAL_SECOND_EVIDENCE_CONTRACT,
                "evidence",
                "evidence_chain",
                "The sequential bridge or comparison objective is resolved.",
            ),
            _agent(
                "aggregate",
                model_id,
                AGGREGATE_CONTRACT,
                "aggregate",
                "semantic_answer_candidate",
                "One evidence-grounded semantic answer candidate is produced.",
            ),
            relations=(
                _relation("decompose", "evidence_1"),
                _relation("evidence_1", "evidence_2"),
                _relation("evidence_2", "aggregate"),
            ),
        ),
        _component(
            _agent(
                "verify",
                model_id,
                VERIFY_CONTRACT,
                "verify",
                "verified_semantic_answer",
                "Exactly one independently verified semantic answer is returned.",
            ),
            relations=(_relation("aggregate", "verify"),),
        ),
        _component(
            _agent(
                "format",
                model_id,
                FORMAT_CONTRACT,
                "format",
                "terminal_answer",
                "Exactly one non-empty <answer>...</answer> artifact is returned.",
            ),
            relations=(_relation("verify", "format"),),
            output_agent_id="format",
        ),
        AgentAction(AgentActionType.FINISH),
    )


def _fan_in_actions(model_id: str) -> tuple[AgentAction, ...]:
    return (
        _decompose_component(model_id),
        _component(
            _agent(
                "evidence_left",
                model_id,
                LEFT_EVIDENCE_CONTRACT,
                "evidence",
                "evidence_proposition",
                "The first independent evidence objective has a cited proposition.",
            ),
            _agent(
                "evidence_right",
                model_id,
                RIGHT_EVIDENCE_CONTRACT,
                "evidence",
                "evidence_proposition",
                "The second independent evidence objective has a cited proposition.",
            ),
            _agent(
                "aggregate",
                model_id,
                AGGREGATE_CONTRACT,
                "aggregate",
                "semantic_answer_candidate",
                "Both evidence branches are reconciled into one semantic answer.",
            ),
            relations=(
                _relation("decompose", "evidence_left"),
                _relation("decompose", "evidence_right"),
                _relation("evidence_left", "aggregate"),
                _relation("evidence_right", "aggregate"),
            ),
        ),
        _component(
            _agent(
                "verify",
                model_id,
                VERIFY_CONTRACT,
                "verify",
                "verified_semantic_answer",
                "Exactly one independently verified semantic answer is returned.",
            ),
            relations=(_relation("aggregate", "verify"),),
        ),
        _component(
            _agent(
                "format",
                model_id,
                FORMAT_CONTRACT,
                "format",
                "terminal_answer",
                "Exactly one non-empty <answer>...</answer> artifact is returned.",
            ),
            relations=(_relation("verify", "format"),),
            output_agent_id="format",
        ),
        AgentAction(AgentActionType.FINISH),
    )


def _reciprocal_actions(model_id: str) -> tuple[AgentAction, ...]:
    return (
        _decompose_component(model_id),
        _component(
            _agent(
                "candidate_reasoning",
                model_id,
                RECIPROCAL_CANDIDATE_CONTRACT,
                "reasoning",
                "answer_candidate",
                "A source-grounded candidate survives bounded peer revision.",
            ),
            _agent(
                "critical_verification",
                model_id,
                RECIPROCAL_VERIFY_CONTRACT,
                "verify",
                "verified_semantic_answer",
                "One independently checked answer survives bounded peer revision.",
            ),
            relations=(
                _relation("decompose", "candidate_reasoning"),
                _relation("decompose", "critical_verification"),
                _relation(
                    "candidate_reasoning",
                    "critical_verification",
                    reciprocal=True,
                ),
            ),
        ),
        _component(
            _agent(
                "format",
                model_id,
                FORMAT_CONTRACT,
                "format",
                "terminal_answer",
                "Exactly one non-empty <answer>...</answer> artifact is returned.",
            ),
            relations=(_relation("critical_verification", "format"),),
            output_agent_id="format",
        ),
        AgentAction(AgentActionType.FINISH),
    )


def _hotpot_question_type(task: TaskRecord) -> str:
    skillflow = task.metadata.get("skillflow")
    if isinstance(skillflow, Mapping):
        extra = skillflow.get("extra")
        if isinstance(extra, Mapping):
            value = str(extra.get("type") or "").strip().casefold()
            if value in {"comparison", "bridge"}:
                return value
    value = str(task.metadata.get("type") or "").strip().casefold()
    return value if value in {"comparison", "bridge"} else "bridge"


def _execution_plan(condition: str, question_type: str) -> str:
    # ``question_type`` is retained only for stratified reporting.  It must not
    # route generation because HotpotQA's native comparison/bridge annotation
    # is evaluator-side metadata rather than a model-visible observation.
    del question_type
    if condition == "serial":
        return "serial"
    if condition == "fan_in":
        return "fan_in"
    if condition == "finite_reciprocal":
        return "finite_reciprocal"
    raise ValueError(f"unsupported condition: {condition}")


def topology_actions(
    condition: str,
    question_type: str,
    model_id: str,
) -> tuple[AgentAction, ...]:
    plan = _execution_plan(condition, question_type)
    if plan == "serial":
        return _serial_actions(model_id)
    if plan == "fan_in":
        return _fan_in_actions(model_id)
    return _reciprocal_actions(model_id)


def _plan_summary(actions: Sequence[AgentAction]) -> Mapping[str, Any]:
    agents = [
        spec
        for action in actions
        if action.action_type is AgentActionType.ADD_SUBGRAPH
        for spec in action.agents
    ]
    relations = [
        relation
        for action in actions
        if action.action_type is AgentActionType.ADD_SUBGRAPH
        for relation in action.relations
    ]
    reciprocal_pairs = sum(relation.target_to_source for relation in relations)
    planned_calls = len(agents) + (2 * reciprocal_pairs)
    return {
        "functional_component_transactions": sum(
            action.action_type is AgentActionType.ADD_SUBGRAPH for action in actions
        ),
        "agent_count": len(agents),
        "relation_count": len(relations),
        "reciprocal_pair_count": reciprocal_pairs,
        "planned_model_calls": planned_calls,
        "output_agent_id": next(
            (
                action.output_agent_id
                for action in actions
                if action.output_agent_id is not None
            ),
            None,
        ),
        "actions": [action.to_dict() for action in actions],
    }


def _evaluation_dict(outcome: Any) -> dict[str, Any]:
    return {
        "valid": outcome.valid,
        "reward": outcome.reward,
        "metrics": dict(outcome.metrics),
        "reason": outcome.reason,
        "details": dict(outcome.details),
        "evaluator_version": outcome.evaluator_version,
    }


def _call_dict(call: Any) -> dict[str, Any]:
    request = call.request
    return {
        "agent_id": request.agent.id,
        "role_family": request.agent.role_family,
        "contract": request.agent.contract,
        "model_id": request.model.model_id,
        "provider_model": request.model.model_name,
        "phase": request.phase.value,
        "is_output_agent": request.is_output_agent,
        "is_format_agent": request.is_format_agent,
        "upstream": [message.to_dict() for message in request.upstream],
        "output": call.response.text,
        "response_metadata": dict(call.response.metadata),
    }


def _execution_dict(execution: Any) -> Optional[dict[str, Any]]:
    if execution is None:
        return None
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
        "calls": [_call_dict(call) for call in execution.calls],
    }


async def _run_condition(
    *,
    runtime: AgentRuntime,
    registry: Any,
    task: TaskRecord,
    condition: str,
    model_id: str,
    run_id: str,
) -> Mapping[str, Any]:
    """Run one frozen condition; reference answers are read only after generation."""

    question_type = _hotpot_question_type(task)
    plan = _execution_plan(condition, question_type)
    actions = topology_actions(condition, question_type, model_id)
    environment = AgentWorkflowEnv(
        registry,
        runtime=runtime,
        problem=task.question,
        execute_on_edit=True,
        max_agents=8,
        max_agents_per_subgraph=3,
        require_exact_answer_tag=True,
        require_format_agent=True,
    )
    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    terminal_execution = None
    try:
        for action in actions:
            result = await environment.step(action)
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
                    "execution": _execution_dict(result.execution),
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
            raise RuntimeError("condition did not reach explicit FINISH")
        final_answer = terminal_execution.final_answer or ""

        # Leakage boundary: the Agent Runtime receives task.question only.  The
        # TaskRecord (and therefore ground_truth/evaluator payload) is consumed
        # by the evaluator only after every generation step has completed.
        outcome = await evaluate_task(task, final_answer)
        evaluation: Optional[Mapping[str, Any]] = _evaluation_dict(outcome)
        status = "completed"
        failure = None
    except Exception as exc:  # retained in the fixed denominator
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
    if status == "completed" and len(calls) != PLANNED_MODEL_CALLS:
        status = "operational_failure"
        evaluation = None
        failure = {
            "type": "CallBudgetMismatch",
            "message": (
                f"observed {len(calls)} model calls; expected "
                f"{PLANNED_MODEL_CALLS}"
            ),
        }
    return {
        "condition": condition,
        "execution_plan": plan,
        "question_type": question_type,
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
        "api_calls_executed": len(calls),
        "api_attempts": api_attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "steps": steps,
        "run_id": run_id,
        "execution_reused_from_condition": None,
    }


def _is_hotpotqa(task: TaskRecord) -> bool:
    dataset = str(task.metadata.get("dataset_key") or "").strip().casefold()
    source = str(task.metadata.get("source") or "").strip().casefold()
    return dataset == "hotpotqa" or source == "hotpotqa"


def _select_tasks(
    path: Path,
    *,
    expected_split: str,
    task_ids: Sequence[str],
    per_type: int,
) -> tuple[TaskRecord, ...]:
    records = tuple(
        task
        for task in iter_task_records(path, expected_split=expected_split)
        if _is_hotpotqa(task)
    )
    by_id = {task.task_id: task for task in records}
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError("task IDs absent from source: " + ", ".join(missing))
        return tuple(by_id[task_id] for task_id in task_ids)

    strata = {
        question_type: [
            task
            for task in records
            if _hotpot_question_type(task) == question_type
        ][:per_type]
        for question_type in ("comparison", "bridge")
    }
    for question_type, tasks in strata.items():
        if len(tasks) != per_type:
            raise ValueError(
                f"source contains only {len(tasks)} {question_type} HotpotQA tasks"
            )
    selected: list[TaskRecord] = []
    for index in range(per_type):
        selected.extend((strata["comparison"][index], strata["bridge"][index]))
    return tuple(selected)


def _checkpoint_metadata(
    *,
    tasks_path: Path,
    tasks: Sequence[TaskRecord],
    expected_split: str,
    conditions: Sequence[str],
    model_id: str,
    seed: int,
    run_id: str,
    selection_method: str,
) -> Mapping[str, Any]:
    return {
        "dataset": "hotpotqa",
        "partition": expected_split,
        "tasks_path": str(tasks_path),
        "task_ids": [task.task_id for task in tasks],
        "question_types": {
            task.task_id: _hotpot_question_type(task) for task in tasks
        },
        "selection_method": selection_method,
        "conditions": list(conditions),
        "model_id": model_id,
        "seed": seed,
        "run_id": run_id,
        "planned_model_calls_per_condition": PLANNED_MODEL_CALLS,
        "topology_routing_policy": "fixed_condition_no_task_router_v4",
    }


def _metric(condition: Mapping[str, Any], name: str) -> float:
    evaluation = condition.get("evaluation")
    if not isinstance(evaluation, Mapping) or not evaluation.get("valid"):
        return 0.0
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    return float(metrics.get(name) or 0.0)


def _aggregate(
    results: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
) -> Mapping[str, Mapping[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    denominator = len(results)
    for condition in conditions:
        rows = [item["conditions"][condition] for item in results]
        aggregate[condition] = {
            "denominator": denominator,
            "completed": sum(row.get("status") == "completed" for row in rows),
            "operational_failures": sum(
                bool(row.get("operational_failure")) for row in rows
            ),
            "evaluator_valid": sum(
                isinstance(row.get("evaluation"), Mapping)
                and bool(row["evaluation"].get("valid"))
                for row in rows
            ),
            # Strict fixed denominator: operational/evaluator failures contribute 0.
            "strict_exact_match": (
                sum(_metric(row, "exact_match") for row in rows) / denominator
                if denominator
                else None
            ),
            "strict_token_f1": (
                sum(_metric(row, "token_f1") for row in rows) / denominator
                if denominator
                else None
            ),
            "api_calls": sum(int(row.get("api_calls") or 0) for row in rows),
            "api_calls_executed": sum(
                int(row.get("api_calls_executed") or 0) for row in rows
            ),
            "api_attempts": sum(
                int(row.get("api_attempts") or 0) for row in rows
            ),
            "prompt_tokens": sum(
                int(row.get("prompt_tokens") or 0) for row in rows
            ),
            "completion_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in rows
            ),
            "wall_latency_ms": sum(
                float(row.get("wall_latency_ms") or 0.0) for row in rows
            ),
        }
    return aggregate


def _paired_differences(
    results: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
) -> Mapping[str, Any]:
    comparisons: dict[str, Any] = {}
    if "serial" not in conditions:
        return comparisons
    for condition in ("fan_in", "finite_reciprocal"):
        if condition not in conditions:
            continue
        rows = []
        for item in results:
            baseline = item["conditions"]["serial"]
            candidate = item["conditions"][condition]
            rows.append(
                {
                    "task_id": item["task"]["task_id"],
                    "question_type": item["task"]["question_type"],
                    "delta_exact_match": (
                        _metric(candidate, "exact_match")
                        - _metric(baseline, "exact_match")
                    ),
                    "delta_token_f1": (
                        _metric(candidate, "token_f1")
                        - _metric(baseline, "token_f1")
                    ),
                }
            )
        comparisons[f"{condition}_minus_serial"] = {
            "tasks": len(rows),
            "mean_delta_exact_match": (
                sum(row["delta_exact_match"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "mean_delta_token_f1": (
                sum(row["delta_token_f1"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "win_tie_loss_exact_match": {
                "win": sum(row["delta_exact_match"] > 0 for row in rows),
                "tie": sum(row["delta_exact_match"] == 0 for row in rows),
                "loss": sum(row["delta_exact_match"] < 0 for row in rows),
            },
            "per_task": rows,
        }
    return comparisons


async def _run(args: argparse.Namespace) -> int:
    conditions = tuple(args.conditions)
    tasks_path = Path(args.tasks).expanduser()
    if not tasks_path.is_absolute():
        tasks_path = PROJECT_ROOT / tasks_path
    catalog_path = Path(args.catalog).expanduser()
    if not catalog_path.is_absolute():
        catalog_path = PROJECT_ROOT / catalog_path
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.json")

    registry = load_model_registry(catalog_path)
    registry.require_model(args.model_id)
    tasks = _select_tasks(
        tasks_path,
        expected_split=args.expected_split,
        task_ids=args.task_ids,
        per_type=args.per_type,
    )
    selection_method = (
        "explicit_preregistered_task_ids"
        if args.task_ids
        else "sequential_first_n_by_public_hotpot_question_type"
    )
    metadata = _checkpoint_metadata(
        tasks_path=tasks_path,
        tasks=tasks,
        expected_split=args.expected_split,
        conditions=conditions,
        model_id=args.model_id,
        seed=args.seed,
        run_id=args.run_id,
        selection_method=selection_method,
    )

    plans: dict[str, Any] = {}
    for question_type in ("comparison", "bridge"):
        plans[question_type] = {}
        for condition in conditions:
            actions = topology_actions(condition, question_type, args.model_id)
            summary = _plan_summary(actions)
            if summary["planned_model_calls"] != PLANNED_MODEL_CALLS:
                raise ValueError(
                    f"{question_type}/{condition} planned call budget is "
                    f"{summary['planned_model_calls']}, expected {PLANNED_MODEL_CALLS}"
                )
            plans[question_type][condition] = {
                "execution_plan": _execution_plan(condition, question_type),
                **summary,
            }
    metadata = {
        **metadata,
        # Freeze the exact AgentGraph actions/contracts in the checkpoint so a
        # changed diagnostic cannot silently resume older condition receipts.
        "plans": plans,
    }

    orders = {
        task.task_id: [
            condition
            for condition in LATIN_SQUARE[index % len(LATIN_SQUARE)]
            if condition in conditions
        ]
        for index, task in enumerate(tasks)
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "api_calls": 0,
                    "metadata": metadata,
                    "catalog_path": str(catalog_path),
                    "output": str(output),
                    "condition_orders": orders,
                    "plans": plans,
                    "controls": {
                        "same_model_for_all_agents": args.model_id,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": args.seed,
                        "same_full_ten_passage_context": True,
                        "retrieval_tools_enabled": False,
                        "ground_truth_visible_during_generation": False,
                    },
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
        dataset_id="hotpotqa",
    )

    checkpoint_conditions: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise ValueError("existing checkpoint belongs to a different diagnostic")
        raw_conditions = saved.get("conditions")
        if isinstance(raw_conditions, Mapping):
            checkpoint_conditions = {
                str(task_id): dict(value)
                for task_id, value in raw_conditions.items()
                if isinstance(value, Mapping)
            }

    results: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        question_type = _hotpot_question_type(task)
        conditions = checkpoint_conditions.setdefault(task.task_id, {})
        order = tuple(orders[task.task_id])
        for condition in order:
            if condition in conditions:
                continue
            conditions[condition] = dict(
                await _run_condition(
                    runtime=runtime,
                    registry=registry,
                    task=task,
                    condition=condition,
                    model_id=args.model_id,
                    run_id=f"{args.run_id}:{task.task_id}:{condition}",
                )
            )
            _atomic_write_json(
                checkpoint,
                {
                    "schema_version": (
                        "flowsteer.hotpotqa.progressive_topology_checkpoint.v4"
                    ),
                    "status": "in_progress",
                    "metadata": metadata,
                    "condition_orders": orders,
                    "conditions": checkpoint_conditions,
                },
            )

        # Ground truth is serialized only after all condition generations for
        # this task have completed; it was never passed to AgentRuntime.
        results.append(
            {
                "task": {
                    "task_id": task.task_id,
                    "split": task.split,
                    "question_type": question_type,
                    "question": task.question,
                    "ground_truth": task.ground_truth,
                },
                "condition_order": list(order),
                "conditions": conditions,
            }
        )

    payload = {
        "schema_version": "flowsteer.hotpotqa.progressive_topology_diagnostic.v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "controls": {
            "diagnostic_only": True,
            "forced_probe": True,
            "director_invoked": False,
            "training_enabled": False,
            "grpo_eligible": False,
            "skill_evidence_eligible": False,
            "skill_injection_performed": False,
            "same_task_across_conditions": True,
            "same_model_for_all_agents": args.model_id,
            "same_temperature": 0.0,
            "same_top_p": 1.0,
            "same_seed": args.seed,
            "same_full_ten_passage_context": True,
            "retrieval_tools_enabled": False,
            "planned_model_calls_per_condition": PLANNED_MODEL_CALLS,
            "latin_square_condition_order": True,
            "progressive_canvas_execute_on_edit": True,
            "functional_component_action": "add_subgraph",
            "ground_truth_visibility": "evaluator_after_generation_only",
            "failure_denominator_policy": "retain_as_zero_in_strict_metrics",
            "topology_routing_input": "fixed_condition_no_task_router",
            "native_question_type_used_for_generation": False,
        },
        "source_mapping": {
            "canvas_execution": (
                "src/interactive/agent_workflow_env.py::AgentWorkflowEnv.step"
            ),
            "finite_reciprocal": (
                "src/interactive/agent_runtime.py::AgentRuntime._execute_block"
            ),
            "decompose_verify_aggregate_format": (
                "scripts/operators.py and scripts/prompts/prompt.py"
            ),
            "checkpoint_receipts": (
                "scripts/compare_triviaqa_progressive_topologies.py"
            ),
            "hotpot_evaluator": "src/interactive/task_evaluator.py::evaluate_task",
        },
        "plans": plans,
        "aggregate": _aggregate(results, conditions),
        "aggregate_by_question_type": {
            question_type: _aggregate(
                [
                    item
                    for item in results
                    if item["task"]["question_type"] == question_type
                ],
                conditions,
            )
            for question_type in ("comparison", "bridge")
        },
        "paired_differences": _paired_differences(results, conditions),
        "results": results,
    }
    _atomic_write_json(output, payload)
    _atomic_write_json(
        checkpoint,
        {
            "schema_version": (
                "flowsteer.hotpotqa.progressive_topology_checkpoint.v4"
            ),
            "status": "complete",
            "metadata": metadata,
            "condition_orders": orders,
            "conditions": checkpoint_conditions,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="data/joint_qa_v2/development.jsonl",
    )
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=(),
        help=(
            "Explicit preregistered Task IDs. If omitted, select the first "
            "--per-type comparison and bridge records sequentially."
        ),
    )
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument(
        "--catalog",
        default="config/model_catalog_hotpotqa_deep_v6.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--run-id",
        default="hotpotqa-progressive-topology-diagnostic-v4",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/hotpotqa_topology_diagnostic_v4/"
            "progressive_topology_comparison.json"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen tasks and action plans without starting a model/API.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.per_type < 1:
        raise SystemExit("--per-type must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if len(args.task_ids) != len(set(args.task_ids)):
        raise SystemExit("--task-ids must be unique")
    if len(args.conditions) != len(set(args.conditions)):
        raise SystemExit("--conditions must be unique")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
