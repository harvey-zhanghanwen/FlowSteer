#!/usr/bin/env python3
"""Compare three progressive TriviaQA AgentGraph topologies on three tasks.

Every graph is constructed through ``AgentWorkflowEnv.step`` with one atomic
FlowSteer Canvas edit at a time and ``execute_on_edit=True``.  The fixed graphs
are diagnostic forced probes; they are never admitted to GRPO or Skill
evidence and do not replace the natural Flow-Director validation round.
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
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.agent_action_parser import AgentAction, AgentActionType
from src.interactive.agent_runtime import AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import load_model_registry
from src.interactive.openai_gateway import OpenAICompatibleGateway
from src.interactive.qa_retrieval import SkillFlowQARetriever, build_keyword_query
from src.interactive.records import TaskRecord
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import evaluate_task


MODEL_ID = "qwen3.5-9b-local"


def _add(
    agent_id: str,
    contract: str,
    role_family: str,
    *,
    model_id: str = MODEL_ID,
) -> AgentAction:
    return AgentAction(
        AgentActionType.ADD_AGENT,
        agent_id=agent_id,
        model_id=model_id,
        contract=contract,
        role_family=role_family,
    )


def _edge(
    source: str,
    target: str,
    *,
    reciprocal: bool = False,
) -> AgentAction:
    return AgentAction(
        AgentActionType.SET_RELATION,
        source_id=source,
        target_id=target,
        source_to_target=True,
        target_to_source=reciprocal,
    )


def _format_action() -> AgentAction:
    return _add(
        "format",
        "Extract the routed semantic answer and serialize the shortest answer span.",
        "format",
    )


def _serial_actions() -> tuple[AgentAction, ...]:
    return (
        _add(
            "evidence",
            "Read the question and public retrieval passages. Identify the most relevant "
            "passage and return the factual evidence needed to answer; do not emit a "
            "task-level final answer.",
            "evidence",
        ),
        _add(
            "answer_reasoning",
            "Use the routed evidence to derive one concise TriviaQA answer candidate. "
            "Return the candidate with its supporting fact, without answer tags.",
            "reasoning",
        ),
        _add(
            "verification",
            "Verify the routed candidate against the question and public passages. "
            "Correct entity, date, title, or location mismatches and return one supported "
            "semantic answer.",
            "verification",
        ),
        _format_action(),
        _edge("evidence", "answer_reasoning"),
        _edge("answer_reasoning", "verification"),
        _edge("verification", "format"),
        AgentAction(AgentActionType.SET_OUTPUT, agent_id="format"),
        AgentAction(AgentActionType.FINISH),
    )


def _fan_in_actions() -> tuple[AgentAction, ...]:
    return (
        _add(
            "retrieval_evidence",
            "Independently identify the passage that directly supports the requested "
            "entity, date, title, or location. Return the quoted factual relation and "
            "source title; do not emit a task-level final answer.",
            "evidence",
        ),
        _add(
            "independent_candidate",
            "Independently solve the TriviaQA question from the public passages and prior "
            "knowledge. Return one candidate plus a short confidence rationale; do not "
            "use answer tags.",
            "reasoning",
        ),
        _add(
            "synthesis",
            "Consume both routed artifacts. Resolve disagreement by matching the candidate "
            "to the retrieved factual relation, then return one supported semantic answer.",
            "synthesis",
        ),
        _format_action(),
        _edge("retrieval_evidence", "synthesis"),
        _edge("independent_candidate", "synthesis"),
        _edge("synthesis", "format"),
        AgentAction(AgentActionType.SET_OUTPUT, agent_id="format"),
        AgentAction(AgentActionType.FINISH),
    )


def _complex_actions() -> tuple[AgentAction, ...]:
    return (
        _add(
            "question_analysis",
            "Classify the requested answer type and identify the decisive entities, "
            "time constraints, and relations. Return only a factual resolution plan.",
            "reasoning",
        ),
        _add(
            "source_evidence",
            "Use the routed plan and public passages to extract source-grounded evidence, "
            "including title and decisive factual relation. Do not emit a final answer.",
            "evidence",
        ),
        _add(
            "independent_candidate",
            "Use the routed plan to derive an independent answer candidate from the public "
            "passages and prior knowledge. State the candidate and the relation supporting it.",
            "reasoning",
        ),
        _add(
            "candidate_reasoning",
            "Use source evidence to derive a candidate. During reciprocal revision, compare "
            "your draft with the verifier's previous draft and correct unsupported claims.",
            "reasoning",
        ),
        _add(
            "critical_verification",
            "Use the independent candidate to test entity identity, answer type, spelling, "
            "and temporal constraints. During reciprocal revision, inspect the peer draft "
            "and return a corrected verified candidate.",
            "verification",
        ),
        _add(
            "synthesis",
            "Consume both revised artifacts from the reciprocal block. Resolve conflicts "
            "against the public passages and return one supported semantic answer.",
            "synthesis",
        ),
        _format_action(),
        _edge("question_analysis", "source_evidence"),
        _edge("question_analysis", "independent_candidate"),
        _edge("source_evidence", "candidate_reasoning"),
        _edge("independent_candidate", "critical_verification"),
        _edge("candidate_reasoning", "critical_verification", reciprocal=True),
        _edge("candidate_reasoning", "synthesis"),
        _edge("critical_verification", "synthesis"),
        _edge("synthesis", "format"),
        AgentAction(AgentActionType.SET_OUTPUT, agent_id="format"),
        AgentAction(AgentActionType.FINISH),
    )


TOPOLOGY_ACTIONS = {
    "serial": _serial_actions,
    "fan_in": _fan_in_actions,
    "complex_mixed": _complex_actions,
}

SEMANTIC_OUTPUT_AGENT = {
    "serial": "verification",
    "fan_in": "synthesis",
    "complex_mixed": "synthesis",
}


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _terminal_feedback_edit(environment: AgentWorkflowEnv, topology: str) -> AgentAction:
    """Return one FlowSteer-style edit after an empty terminal artifact.

    The terminal validator already tells the Director to modify the graph before
    retrying FINISH.  A fixed-topology diagnostic has no Director, so it applies
    the same bounded continuation to the semantic Output predecessor.  The edit
    never supplies a reference answer and is retained in the trajectory.
    """

    agent_id = SEMANTIC_OUTPUT_AGENT[topology]
    node = environment.graph.get_node(agent_id)
    contract = (
        node.contract.rstrip()
        + " Terminal validation reported an empty routed answer. Re-run the assigned "
        "reasoning and return exactly one non-empty best-supported semantic answer "
        "candidate. If retrieval is inconclusive, use the question and established "
        "factual knowledge to select the best candidate; do not use answer tags."
    )
    return AgentAction(
        AgentActionType.MODIFY_AGENT,
        agent_id=agent_id,
        contract=contract,
    )


def _response_usage(metadata: Mapping[str, object]) -> tuple[int, int, int]:
    return (
        int(metadata.get("prompt_tokens") or 0),
        int(metadata.get("completion_tokens") or 0),
        int(metadata.get("attempt_count") or 1),
    )


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


def _execution_dict(execution: Any) -> dict[str, Any] | None:
    if execution is None:
        return None
    calls = [_call_dict(call) for call in execution.calls]
    return {
        "run_id": execution.run_id,
        "graph_revision": execution.graph_revision,
        "output_agent_id": execution.output_agent_id,
        "final_answer": execution.final_answer,
        "outputs": dict(execution.outputs),
        "block_completion_order": [list(block) for block in execution.block_completion_order],
        "executed_agent_ids": list(execution.executed_agent_ids),
        "reused_agent_ids": list(execution.reused_agent_ids),
        "calls": calls,
    }


async def _run_condition(
    *,
    runtime: AgentRuntime,
    registry: Any,
    task: TaskRecord,
    problem: str,
    topology: str,
) -> dict[str, Any]:
    environment = AgentWorkflowEnv(
        registry,
        runtime=runtime,
        problem=problem,
        execute_on_edit=True,
        max_agents=8,
        require_exact_answer_tag=True,
        require_format_agent=True,
    )
    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    terminal_execution = None
    actions = list(TOPOLOGY_ACTIONS[topology]())
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
                action.action_type is AgentActionType.FINISH
                and terminal_feedback_edits == 0
                and "non_empty=False" in result.feedback
            ):
                # FlowSteer keeps a rejected terminal action as feedback for the
                # next atomic Canvas edit.  Apply exactly one answer-free edit,
                # execute its dirty closure immediately, then retry FINISH.
                recovery = _terminal_feedback_edit(environment, topology)
                recovery_result = await environment.step(recovery)
                recovery_execution = _execution_dict(recovery_result.execution)
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
                        "execution": recovery_execution,
                    }
                )
                if not recovery_result.accepted:
                    raise RuntimeError(
                        f"{topology} terminal-feedback edit was rejected: "
                        f"{recovery_result.feedback}"
                    )
                if recovery_result.execution is not None:
                    terminal_execution = recovery_result.execution
                terminal_feedback_edits += 1
                continue
            raise RuntimeError(
                f"{topology} edit {len(steps)} was rejected: {result.feedback}"
            )
        if result.execution is not None:
            terminal_execution = result.execution
        action_index += 1
    if not environment.finished or terminal_execution is None:
        raise RuntimeError(f"{topology} did not reach explicit FINISH")
    final_answer = terminal_execution.final_answer or ""
    outcome = await evaluate_task(task, final_answer)
    all_calls = [
        call
        for step in steps
        if not step["execution_reused"]
        for call in ((step.get("execution") or {}).get("calls") or [])
    ]
    prompt_tokens = completion_tokens = api_attempts = 0
    for call in all_calls:
        p, c, a = _response_usage(call["response_metadata"])
        prompt_tokens += p
        completion_tokens += c
        api_attempts += a
    return {
        "topology": topology,
        "graph": environment.graph.snapshot().to_dict(),
        "topology_statistics": environment.graph.topology_statistics(),
        "atomic_edit_count": sum(
            int(step["accepted"] and step["action"]["action"] != "finish")
            for step in steps
        ),
        "terminal_attempt_count": sum(
            int(step["action"]["action"] == "finish") for step in steps
        ),
        "terminal_feedback_edit_count": terminal_feedback_edits,
        "each_edit_executed": all(
            step["execution"] is not None
            for step in steps
            if step["accepted"] and step["action"]["action"] != "finish"
        ),
        "explicit_finish": environment.finished,
        "final_answer": final_answer,
        "evaluation": asdict(outcome),
        "wall_latency_ms": (time.monotonic() - started) * 1000.0,
        "api_calls": len(all_calls),
        "api_attempts": api_attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "steps": steps,
    }


def _select_train_tasks(path: Path, count: int) -> tuple[TaskRecord, ...]:
    selected: list[TaskRecord] = []
    for task in iter_task_records(path, expected_split="train"):
        if task.metadata.get("dataset_key") == "triviaqa":
            selected.append(task)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"train source contains only {len(selected)} TriviaQA tasks")
    return tuple(selected)


async def _run(args: argparse.Namespace) -> int:
    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = PROJECT_ROOT / tasks_path
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = PROJECT_ROOT / catalog_path
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    checkpoint = output.with_suffix(".checkpoint.json")

    tasks = _select_train_tasks(tasks_path, 3)
    registry = load_model_registry(catalog_path)
    registry.require_model(MODEL_ID)
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

    retrieval: dict[str, Any] = {}
    with SkillFlowQARetriever(
        index_path=args.retrieval_index,
        skillflow_source=args.skillflow_source,
        search_limit=args.search_limit,
    ) as retriever:
        for task in tasks:
            receipt = retriever.retrieve(build_keyword_query(task.question))
            retrieval[task.task_id] = receipt

    checkpoint_conditions: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        expected_ids = [task.task_id for task in tasks]
        if saved.get("run_id") != args.run_id or saved.get("task_ids") != expected_ids:
            raise ValueError(
                "existing topology checkpoint belongs to a different run or task selection"
            )
        raw_conditions = saved.get("conditions", {})
        if isinstance(raw_conditions, dict):
            checkpoint_conditions = {
                str(task_id): dict(value)
                for task_id, value in raw_conditions.items()
                if isinstance(value, dict)
            }

    results: list[dict[str, Any]] = []
    for task in tasks:
        receipt = retrieval[task.task_id]
        problem = receipt.render_problem(task.question)
        conditions = checkpoint_conditions.setdefault(task.task_id, {})
        for topology in ("serial", "fan_in", "complex_mixed"):
            if topology in conditions:
                print(f"{task.task_id} {topology}: resumed from checkpoint", flush=True)
                continue
            conditions[topology] = await _run_condition(
                runtime=runtime,
                registry=registry,
                task=task,
                problem=problem,
                topology=topology,
            )
            _atomic_write_json(
                checkpoint,
                {
                    "schema_version": (
                        "flowsteer.triviaqa.progressive_topology_checkpoint.v1"
                    ),
                    "run_id": args.run_id,
                    "task_ids": [item.task_id for item in tasks],
                    "status": "in_progress",
                    "conditions": checkpoint_conditions,
                },
            )
            print(
                f"{task.task_id} {topology}: "
                f"EM={conditions[topology]['evaluation']['metrics']['exact_match']:.3f} "
                f"F1={conditions[topology]['evaluation']['metrics']['token_f1']:.3f}",
                flush=True,
            )
        results.append(
            {
                "task": task.to_dict(),
                "retrieval": receipt.to_dict(),
                "model_input": problem,
                "conditions": conditions,
            }
        )

    aggregate: dict[str, Any] = {}
    for topology in ("serial", "fan_in", "complex_mixed"):
        conditions = [item["conditions"][topology] for item in results]
        aggregate[topology] = {
            "tasks": len(conditions),
            "exact_match": sum(
                item["evaluation"]["metrics"]["exact_match"] for item in conditions
            )
            / len(conditions),
            "token_f1": sum(
                item["evaluation"]["metrics"]["token_f1"] for item in conditions
            )
            / len(conditions),
            "api_calls": sum(item["api_calls"] for item in conditions),
            "prompt_tokens": sum(item["prompt_tokens"] for item in conditions),
            "completion_tokens": sum(item["completion_tokens"] for item in conditions),
            "wall_latency_ms": sum(item["wall_latency_ms"] for item in conditions),
        }

    payload = {
        "schema_version": "flowsteer.triviaqa.progressive_topology_probe.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "selection": {
            "split": "train",
            "method": "first_three_sequential_triviaqa_tasks",
            "task_ids": [task.task_id for task in tasks],
        },
        "controls": {
            "diagnostic_only": True,
            "forced_probe": True,
            "grpo_eligible": False,
            "skill_evidence_eligible": False,
            "director_invoked": False,
            "training_enabled": False,
            "same_model_for_all_agents": MODEL_ID,
            "same_public_retrieval_boundary": True,
            "temperature": 0.0,
            "seed": args.seed,
            "progressive_canvas_execute_on_edit": True,
            "comparison_interpretation": (
                "serial_vs_fan_in_controls_agent_count; complex_mixed_is_an_"
                "architecture_capacity_comparison_with_more_agents_and_calls"
            ),
        },
        "aggregate": aggregate,
        "results": results,
    }
    _atomic_write_json(output, payload)
    _atomic_write_json(
        checkpoint,
        {
            "schema_version": "flowsteer.triviaqa.progressive_topology_checkpoint.v1",
            "run_id": args.run_id,
            "task_ids": [item.task_id for item in tasks],
            "status": "complete",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/agentgraph_v1/train.jsonl")
    parser.add_argument("--catalog", default="config/model_catalog_triviaqa_v1.yaml")
    parser.add_argument(
        "--retrieval-index",
        default=(
            "/ssd1/iclr/SKILLEV/skillev-new-b2-temp/data/datasets/"
            "dpr-wikipedia/atlas-retrieval.sqlite3"
        ),
    )
    parser.add_argument(
        "--skillflow-source",
        default="/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src",
    )
    parser.add_argument("--search-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--run-id", default="triviaqa-progressive-topology-demo-001")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/triviaqa_round_01/topology_probe3/"
            "progressive_topology_comparison.json"
        ),
    )
    args = parser.parse_args()
    if args.search_limit < 1:
        parser.error("--search-limit must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
