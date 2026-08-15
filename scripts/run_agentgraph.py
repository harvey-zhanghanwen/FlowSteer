#!/usr/bin/env python3
"""Run one AgentGraph workflow with Qwen3.5-9B as the Flow-Director."""

from __future__ import annotations

import argparse
import asyncio
from itertools import islice
import json
from pathlib import Path
import sys


async def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.interactive.agent_runtime import AgentRuntime
    from src.interactive.agent_workflow_env import AgentWorkflowEnv
    from src.interactive.config_loader import load_model_registry
    from src.interactive.director import AgentGraphOrchestrator, OpenAIDirectorClient
    from src.interactive.openai_gateway import OpenAICompatibleGateway
    from src.interactive.task_dataset import iter_task_records

    task = None
    if args.dataset:
        try:
            dataset_path = (
                Path(args.dataset)
                if Path(args.dataset).is_absolute()
                else root / args.dataset
            )
            task = next(
                islice(
                    iter_task_records(
                        dataset_path,
                        expected_split=args.expected_split,
                    ),
                    args.task_index,
                    None,
                )
            )
        except (IndexError, StopIteration) as exc:
            raise ValueError(f"dataset has no task at index {args.task_index}") from exc
        question = task.question
        if args.show_task_id:
            print(task.task_id)
    else:
        if not args.question:
            raise ValueError("provide a question or --dataset")
        question = args.question

    if args.dry_load:
        if task is None:
            raise ValueError("--dry-load requires --dataset")
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "split": task.split,
                    "source": task.metadata.get("source", "unknown"),
                    "task_type": task.metadata.get("task_type", "unknown"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    registry = load_model_registry(root / args.catalog)
    gateway = OpenAICompatibleGateway(timeout_seconds=args.timeout)
    runtime = AgentRuntime(
        registry,
        gateway,
        max_concurrency=args.max_concurrency,
        timeout_seconds=args.timeout,
    )
    env = AgentWorkflowEnv(registry, runtime=runtime)
    director = OpenAIDirectorClient(
        base_url=args.director_url,
        model=args.director_model,
        policy_version=args.policy_version,
        timeout_seconds=args.timeout,
    )
    result = await AgentGraphOrchestrator(
        registry,
        director,
        max_rounds=args.max_rounds,
        seed=args.seed,
    ).run(env, question)
    print(result.final_answer)
    if args.show_graph:
        print(
            json.dumps(result.final_graph, ensure_ascii=False, indent=2, sort_keys=True)
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--dataset", help="aligned JSONL path relative to the repository"
    )
    parser.add_argument("--expected-split", choices=["train", "validation", "test"])
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--show-task-id", action="store_true")
    parser.add_argument(
        "--dry-load",
        action="store_true",
        help="validate/load one dataset record without starting model calls",
    )
    parser.add_argument("--catalog", default="config/model_catalog.yaml")
    parser.add_argument("--director-url", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--director-model", default="supervisor_theta")
    parser.add_argument("--policy-version", default="qwen3.5-9b-sglang-local-v1")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-graph", action="store_true")
    args = parser.parse_args()
    if args.task_index < 0:
        parser.error("--task-index must be non-negative")
    if bool(args.question) == bool(args.dataset):
        parser.error("provide exactly one of question or --dataset")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
