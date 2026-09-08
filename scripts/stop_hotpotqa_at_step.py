#!/usr/bin/env python3
"""Request the existing trainer's safe stop; never train, kill, or restart it.

Operational adapter for train_hotpotqa_grpo.py's existing STOP_REQUESTED checks
before the next step and after checkpoint/publication/validation/W&B commit.
This does not change the configured scheduler horizon or any learning objective.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.step_transaction import StepPhase


def should_request_stop(
    committed: int, current_step: dict | None, target: int
) -> bool:
    if type(committed) is not int or committed < 0:
        raise ValueError("committed must be a non-negative integer")
    if type(target) is not int or target < 1:
        raise ValueError("target must be a positive integer")
    if current_step is not None:
        if not isinstance(current_step, dict):
            raise ValueError("current_step must be a mapping or None")
        step = current_step.get("step")
        if type(step) is not int or step < 1:
            raise ValueError("current_step.step must be a positive integer")
        phase = current_step.get("phase")
        if not isinstance(phase, str) or phase not in {p.value for p in StepPhase}:
            raise ValueError("current_step.phase must use the existing StepPhase")
        # PREPARED is persisted only after the runner's pre-step stop check.
        # It will therefore finish this whole step before observing the marker.
        if step >= target:
            return True
    return committed >= target


def watch(run_root: Path, unit: str, target: int, poll_seconds: float) -> int:
    run_root = run_root.resolve(strict=True)
    if not run_root.is_dir() or not (run_root / "run_state.json").is_file():
        raise ValueError("run_root must contain an existing committed run_state")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    should_request_stop(0, None, target)
    marker = run_root / "STOP_REQUESTED"
    previous = None
    while True:
        state = json.loads((run_root / "run_state.json").read_text())
        try:
            current = json.loads((run_root / "recovery/current_step.json").read_text())
        except FileNotFoundError:
            current = None
        committed = state["optimizer_updates_completed"]
        request_stop = should_request_stop(committed, current, target)
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        active_state = result.stdout.strip()
        if active_state not in {"active", "activating", "deactivating", "inactive", "failed"}:
            raise RuntimeError("training service returned an unknown state")
        snapshot = {
            "committed_optimizer_steps": committed,
            "current_step": current.get("step") if current else None,
            "phase": current.get("phase") if current else None,
            "requested_stop_step": target,
            "training_service_state": active_state,
        }
        if snapshot != previous:
            print(json.dumps(snapshot, sort_keys=True), flush=True)
            previous = snapshot
        if request_stop:
            try:
                with marker.open("x", encoding="utf-8") as stream:
                    stream.write(f"User requested a safe stop after approximately {target} committed optimizer steps.\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print("STOP_REQUESTED created; the trainer owns completion and shutdown.", flush=True)
            except FileExistsError:
                pass  # Preserve an existing user stop request verbatim.
        if active_state in {"inactive", "failed"}:
            if committed >= target and active_state == "inactive":
                print(f"Training stopped at {committed} committed optimizer steps.", flush=True)
                return 0
            print("Training stopped before confirmed target completion; no automatic restart.", flush=True)
            return 1
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    return watch(args.run_root, args.unit, args.target, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
