#!/usr/bin/env python3
"""Run one AgentGraph workflow with Qwen3.5-9B as the Flow-Director."""

from __future__ import annotations

import argparse
import asyncio
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
    ).run(env, args.question)
    print(result.final_answer)
    if args.show_graph:
        import json

        print(json.dumps(result.final_graph, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--catalog", default="config/model_catalog.yaml")
    parser.add_argument("--director-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--director-model", default="supervisor_theta")
    parser.add_argument("--policy-version", default="qwen3.5-9b-sglang-local-v1")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-graph", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
