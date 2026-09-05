#!/usr/bin/env python3
"""Replay frozen v50 canary actions after separating public option metadata.

No model generation, action repair, reward modification or evaluator relaxation.
Reuse evaluate_hotpotqa_round's existing evaluator-only append protocol. Original
events remain intact. This migration is unnecessary for newly collected records.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path

from evaluate_hotpotqa_round import _retry_terminal_evaluator
from train_agentgraph_smoke import LiveSmokeBackend
from src.interactive.config_loader import load_yaml
from src.interactive.records import TaskRecord


def native_replay_source(source: dict) -> dict:
    if source["evaluation"].get("reason") != "environment_replay_transition_mismatch":
        raise ValueError("Only the diagnosed native transition mismatch is supported")
    metadata = source["turns"][-1]["runtime_summary"]["output_metadata"]
    candidates = [value for value in metadata.values()
                  if value.get("evaluator_environment_trace")]
    if not candidates or len({value["environment_episode_id"] for value in candidates}) != 1:
        raise ValueError("One frozen environment episode is required")
    final = max(candidates, key=lambda value: len(value["evaluator_environment_trace"]))
    trace = deepcopy(final["evaluator_environment_trace"])
    receipts = final["environment_receipts"]
    if len(trace) != final["environment_steps"] or len(trace) != len(receipts):
        raise ValueError("Incomplete frozen action receipts")
    moved = 0
    for index, (entry, receipt) in enumerate(zip(trace, receipts)):
        if entry["step"] != index or entry["action"] != receipt["action"]:
            raise ValueError("Frozen action order mismatch")
        assignment = entry["info"].get("public_option_assignment")
        if assignment is not None:
            if assignment != receipt.get("public_option_assignment"):
                raise ValueError("Public projection does not match its saved receipt")
            del entry["info"]["public_option_assignment"]
            moved += 1
    if not moved:
        raise ValueError("No diagnosed public projection serialization to separate")
    repaired = deepcopy(source)
    repaired["evaluation"]["details"]["trace"] = trace
    return repaired


async def main(config_path: Path) -> None:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    backend = LiveSmokeBackend.from_config(config, config_path.parent.parent, evaluation_only=True)
    sources = list(backend.evidence_store.trajectories.payloads())
    valid = {item["task"]["task_id"] for item in sources
             if item["condition_id"] == config["experiment"]["condition_id"]
             and item["evaluation"].get("valid") is True}
    for source in sources:
        task_id = source["task"]["task_id"]
        if (task_id in valid
                or source["condition_id"] != config["experiment"]["condition_id"]
                or source["evaluation"].get("reason") != "environment_replay_transition_mismatch"):
            continue
        repaired = native_replay_source(source)
        result = await _retry_terminal_evaluator(
            backend, TaskRecord.from_dict(source["task"]), repaired,
            versions=source["versions"], attempt=1,
        )
        print(json.dumps({"task_id": task_id, "mode": "native_evaluator_only",
                          "model_calls": 0, "evaluation": {
                              key: result["evaluation"].get(key)
                              for key in ("valid", "reward", "reason")}}), flush=True)
        if result["evaluation"].get("valid") is not True:
            raise RuntimeError("Strict native replay failed; original evidence retained")
        valid.add(task_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    asyncio.run(main(parser.parse_args().config))
