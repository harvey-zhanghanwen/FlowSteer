#!/usr/bin/env python3
"""Run one fixed complex non-serial AgentGraph on sequential HotpotQA tasks.

The script is evaluation-only and reuses the same AgentGraph, AgentRuntime,
gateway, Format Agent, and HotpotQA evaluator as the main inference path.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any



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


def _complex_graph(model_id: str):
    from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation

    nodes = [
        AgentNode(
            "task_decomposition",
            model_id,
            "Decompose the HotpotQA question into two distinct evidence objectives. "
            "Identify the entities, constraints, and relation that must be connected. "
            "Return only an evidence plan; do not answer the question.",
            role_family="reasoning",
        ),
        AgentNode(
            "entity_evidence",
            model_id,
            "Use the routed evidence plan and supplied passages to resolve the first "
            "entity or comparison objective. Return grounded evidence with passage "
            "titles and the relevant entity; do not produce a task-level final answer.",
            role_family="evidence",
        ),
        AgentNode(
            "relation_evidence",
            model_id,
            "Use the routed evidence plan and supplied passages to resolve the second "
            "relation, attribute, or bridge objective. Return grounded evidence with "
            "passage titles; do not produce a task-level final answer.",
            role_family="evidence",
        ),
        AgentNode(
            "candidate_reasoning",
            model_id,
            "Use the routed entity evidence to derive a candidate answer. During peer "
            "revision, check the candidate against the peer's independent relation "
            "evidence and correct unsupported reasoning. Return a supported candidate "
            "and its evidence chain.",
            role_family="reasoning",
        ),
        AgentNode(
            "critical_verification",
            model_id,
            "Use the routed relation evidence to test candidate identities and question "
            "constraints. During peer revision, examine the peer's candidate, identify "
            "any contradiction or missing bridge, and return a corrected verified "
            "candidate with evidence.",
            role_family="verification",
        ),
        AgentNode(
            "synthesis",
            model_id,
            "Consume both revised artifacts from the reciprocal reasoning block. Resolve "
            "disagreement using the supplied passages and return one supported semantic "
            "answer with a concise evidence chain.",
            role_family="synthesis",
        ),
        AgentNode(
            "format",
            model_id,
            "Extract and serialize the routed semantic answer.",
            role_family="format",
        ),
    ]
    relations = [
        AgentRelation("task_decomposition", "entity_evidence", True, False),
        AgentRelation("task_decomposition", "relation_evidence", True, False),
        AgentRelation("entity_evidence", "candidate_reasoning", True, False),
        AgentRelation("relation_evidence", "critical_verification", True, False),
        AgentRelation("candidate_reasoning", "critical_verification", True, True),
        AgentRelation("candidate_reasoning", "synthesis", True, False),
        AgentRelation("critical_verification", "synthesis", True, False),
        AgentRelation("synthesis", "format", True, False),
    ]
    return AgentGraph(nodes, relations, output_agent_id="format")


async def _run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.interactive.agent_runtime import AgentRuntime
    from src.interactive.config_loader import load_model_registry
    from src.interactive.openai_gateway import OpenAICompatibleGateway
    from src.interactive.task_dataset import iter_task_records
    from src.interactive.task_evaluator import evaluate_task

    source = Path(args.tasks)
    if not source.is_absolute():
        source = root / source
    records = list(iter_task_records(source))
    selected = records[args.start_index : args.start_index + args.task_count]
    if len(selected) != args.task_count:
        raise ValueError("task source does not contain the requested sequential range")

    catalog = Path(args.catalog)
    if not catalog.is_absolute():
        catalog = root / catalog
    registry = load_model_registry(catalog)
    registry.require_model(args.model_id)
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
    graph = _complex_graph(args.model_id)
    validation = graph.validate(registry)
    validation.raise_if_invalid()

    trajectories: list[dict[str, Any]] = []
    for index, task in enumerate(selected):
        started = time.monotonic()
        result = await runtime.execute(
            graph,
            task.question,
            run_id=f"{args.run_id}-{index:02d}",
            format_output_agent=True,
        )
        wall_latency_ms = (time.monotonic() - started) * 1000.0
        evaluation = await evaluate_task(task, result.final_answer or "")
        calls = [_call_dict(call) for call in result.calls]
        trajectories.append(
            {
                "task": {
                    "task_id": task.task_id,
                    "split": task.split,
                    "question": task.question,
                    "ground_truth": task.ground_truth,
                },
                "graph": graph.snapshot().to_dict(),
                "topology_statistics": graph.topology_statistics(),
                "final_answer": result.final_answer,
                "evaluation": _evaluation_dict(evaluation),
                "outputs": dict(result.outputs),
                "block_completion_order": [
                    list(block) for block in result.block_completion_order
                ],
                "executed_agent_ids": list(result.executed_agent_ids),
                "wall_latency_ms": wall_latency_ms,
                "api_calls": len(calls),
                "prompt_tokens": sum(
                    int(call["response_metadata"].get("prompt_tokens") or 0)
                    for call in calls
                ),
                "completion_tokens": sum(
                    int(call["response_metadata"].get("completion_tokens") or 0)
                    for call in calls
                ),
                "calls": calls,
            }
        )

    valid = [item for item in trajectories if item["evaluation"]["valid"]]
    payload: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.complex_topology_demo.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "selection": {
            "method": "sequential",
            "source": str(source),
            "start_index": args.start_index,
            "task_count": args.task_count,
        },
        "controls": {
            "model_id_for_all_agents": args.model_id,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": args.seed,
            "director_invoked": False,
            "training_enabled": False,
            "skill_update_enabled": False,
        },
        "graph": graph.snapshot().to_dict(),
        "topology_statistics": graph.topology_statistics(),
        "aggregate": {
            "tasks": len(trajectories),
            "valid": len(valid),
            "correct": sum(
                int(item["evaluation"]["metrics"].get("exact_match", 0.0) == 1.0)
                for item in valid
            ),
            "exact_match": (
                sum(item["evaluation"]["metrics"].get("exact_match", 0.0) for item in valid)
                / len(valid)
                if valid
                else None
            ),
            "token_f1": (
                sum(item["evaluation"]["metrics"].get("token_f1", 0.0) for item in valid)
                / len(valid)
                if valid
                else None
            ),
            "api_calls": sum(item["api_calls"] for item in trajectories),
            "prompt_tokens": sum(item["prompt_tokens"] for item in trajectories),
            "completion_tokens": sum(
                item["completion_tokens"] for item in trajectories
            ),
            "wall_latency_ms": sum(item["wall_latency_ms"] for item in trajectories),
        },
        "trajectories": trajectories,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)

    summary = {
        "output": str(output),
        "topology_statistics": payload["topology_statistics"],
        "aggregate": payload["aggregate"],
        "tasks": [
            {
                "task_id": item["task"]["task_id"],
                "ground_truth": item["task"]["ground_truth"],
                "final_answer": item["final_answer"],
                "metrics": item["evaluation"]["metrics"],
                "api_calls": item["api_calls"],
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": item["completion_tokens"],
                "wall_latency_ms": item["wall_latency_ms"],
            }
            for item in trajectories
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default=(
            "artifacts/hotpotqa_multiagent_skill/incremental_graph_v9_5_confirm32/"
            "selected_tasks.jsonl"
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument(
        "--catalog",
        default="config/model_catalog_hotpotqa_deep_v6.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--run-id", default="hotpotqa-complex-topology-demo-001")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/hotpotqa_multiagent_skill/complex_topology_demo3/"
            "complex_topology_trajectories.json"
        ),
    )
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.task_count < 1:
        parser.error("--task-count must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
