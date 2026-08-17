#!/usr/bin/env python3
"""Compare fixed serial and fan-in AgentGraphs on one HotpotQA record.

This is an evaluation-only wiring script.  It reuses the project's AgentGraph,
AgentRuntime, OpenAI-compatible gateway, and official-compatible HotpotQA
answer evaluator; it does not add a new orchestration or execution path.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


def _graph(condition: str, model_id: str):
    from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation

    format_node = AgentNode(
        "format",
        model_id,
        "Extract and serialize the routed semantic answer.",
        role_family="format",
    )
    if condition == "serial":
        nodes = [
            AgentNode(
                "documentary_evidence",
                model_id,
                "Identify the website or company featured in the documentary named "
                "24 Hours on Craigslist. Return that entity and the supporting passage "
                "evidence; do not identify its founder.",
                role_family="evidence",
            ),
            AgentNode(
                "founder_evidence",
                model_id,
                "Use the routed entity and the supplied passages to identify its founder. "
                "Return the founder's name and evidence that the person is the American "
                "internet entrepreneur requested by the question.",
                role_family="bridge",
            ),
            AgentNode(
                "verification",
                model_id,
                "Verify the routed candidate against the question and passages. Return the "
                "supported semantic answer and concise evidence; reject an unsupported name.",
                role_family="verification",
            ),
            format_node,
        ]
        relations = [
            AgentRelation("documentary_evidence", "founder_evidence", True, False),
            AgentRelation("founder_evidence", "verification", True, False),
            AgentRelation("verification", "format", True, False),
        ]
    elif condition == "fan_in":
        nodes = [
            AgentNode(
                "documentary_evidence",
                model_id,
                "Independently identify the website or company featured in the documentary "
                "named 24 Hours on Craigslist. Return that entity and supporting passage "
                "evidence; do not identify its founder.",
                role_family="evidence",
            ),
            AgentNode(
                "founder_candidates",
                model_id,
                "Independently inspect the passages for American internet entrepreneurs and "
                "the websites or companies they founded. Return candidate founder-entity "
                "pairs with supporting passage evidence; do not use the documentary title "
                "alone to select the final answer.",
                role_family="evidence",
            ),
            AgentNode(
                "synthesis",
                model_id,
                "Combine both routed evidence artifacts. Match the entity identified from "
                "the documentary to the corresponding founder, verify the nationality and "
                "occupation constraints, and return the supported semantic answer.",
                role_family="synthesis",
            ),
            format_node,
        ]
        relations = [
            AgentRelation("documentary_evidence", "synthesis", True, False),
            AgentRelation("founder_candidates", "synthesis", True, False),
            AgentRelation("synthesis", "format", True, False),
        ]
    else:  # pragma: no cover - argparse and caller constrain this value
        raise ValueError(f"unsupported condition: {condition}")
    return AgentGraph(nodes, relations, output_agent_id="format")


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
    task = next(
        (record for record in iter_task_records(source) if record.task_id == args.task_id),
        None,
    )
    if task is None:
        raise ValueError(f"task not found: {args.task_id}")

    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path
    registry = load_model_registry(catalog_path)
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

    results: dict[str, Any] = {}
    for condition in ("serial", "fan_in"):
        graph = _graph(condition, args.model_id)
        started = time.monotonic()
        runtime_result = await runtime.execute(
            graph,
            task.question,
            run_id=f"{args.run_id}-{condition}",
            format_output_agent=True,
        )
        wall_latency_ms = (time.monotonic() - started) * 1000.0
        evaluation = await evaluate_task(task, runtime_result.final_answer or "")
        calls = [_call_dict(call) for call in runtime_result.calls]
        results[condition] = {
            "graph": graph.snapshot().to_dict(),
            "topology_statistics": graph.topology_statistics(),
            "final_answer": runtime_result.final_answer,
            "evaluation": _evaluation_dict(evaluation),
            "outputs": dict(runtime_result.outputs),
            "block_completion_order": [list(block) for block in runtime_result.block_completion_order],
            "executed_agent_ids": list(runtime_result.executed_agent_ids),
            "wall_latency_ms": wall_latency_ms,
            "api_calls": len(calls),
            "prompt_tokens": sum(
                int(call["response_metadata"].get("prompt_tokens") or 0) for call in calls
            ),
            "completion_tokens": sum(
                int(call["response_metadata"].get("completion_tokens") or 0) for call in calls
            ),
            "calls": calls,
        }

    payload: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.topology_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "comparison_type": "fixed_graph_controlled_topology_comparison",
        "training_enabled": False,
        "director_invoked": False,
        "controls": {
            "same_task": True,
            "same_model_for_all_agents": args.model_id,
            "same_temperature": 0.0,
            "same_top_p": 1.0,
            "same_seed": args.seed,
            "same_agent_count": 4,
            "same_api_call_count_target": 4,
            "only_intended_difference": "serial versus fan-in dependency structure and role contracts",
        },
        "task": {
            "task_id": task.task_id,
            "split": task.split,
            "question": task.question,
            "ground_truth": task.ground_truth,
        },
        "conditions": results,
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
        "task_id": task.task_id,
        "ground_truth": task.ground_truth,
        "conditions": {
            name: {
                "final_answer": result["final_answer"],
                "metrics": result["evaluation"]["metrics"],
                "topology": result["topology_statistics"],
                "api_calls": result["api_calls"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "wall_latency_ms": result["wall_latency_ms"],
            }
            for name, result in results.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default=(
            "artifacts/hotpotqa_multiagent_skill/incremental_graph_v9_5_train16/"
            "selected_tasks.jsonl"
        ),
    )
    parser.add_argument(
        "--task-id",
        default="hotpotqa:5a7e567b55429949594199a0",
    )
    parser.add_argument(
        "--catalog",
        default="config/model_catalog_hotpotqa_deep_v6.yaml",
    )
    parser.add_argument("--model-id", default="qwen3.5-9b-local")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--run-id", default="hotpotqa-topology-demo-001")
    parser.add_argument(
        "--output",
        default=(
            "artifacts/hotpotqa_multiagent_skill/topology_comparison_craig_newmark/"
            "paired_topology_demo.json"
        ),
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
