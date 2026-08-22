#!/usr/bin/env python3
"""Validate FlowSteer's three-GPU assignment against current NVIDIA state."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys


ROLES = (
    ("learner model/optimizer", "FLOWSTEER_LEARNER_GPU", "3"),
    ("SGLang Supervisor rollout", "FLOWSTEER_ROLLOUT_GPU", "0"),
    ("gradient replica/backward", "FLOWSTEER_GRADIENT_GPU", "5"),
)


def main() -> int:
    selected: dict[str, int] = {}
    for role, env_name, default in ROLES:
        raw = os.getenv(env_name, default)
        if not raw.isdigit():
            print(f"{env_name} must be a non-negative GPU index", file=sys.stderr)
            return 2
        selected[role] = int(raw)
    if len(set(selected.values())) != len(selected):
        print("the three FlowSteer roles must use distinct physical GPUs", file=sys.stderr)
        return 2

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"cannot query NVIDIA GPUs: {exc}", file=sys.stderr)
        return 3
    rows = {int(row[0].strip()): [cell.strip() for cell in row] for row in csv.reader(io.StringIO(output))}

    failed = False
    for role, index in selected.items():
        row = rows.get(index)
        if row is None:
            print(f"{role}: GPU {index} does not exist", file=sys.stderr)
            failed = True
            continue
        _, name, total, used, free, utilization = row
        free_ratio = float(free) / float(total)
        state = "ready" if free_ratio >= 0.80 else "busy"
        print(
            f"{role}: GPU {index} {name}; used {float(used)/1024:.1f} GiB, "
            f"free {float(free)/1024:.1f} GiB, util {utilization}%; {state}"
        )
        if free_ratio < 0.50:
            failed = True
    return 4 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
