#!/usr/bin/env python3
"""Aggregate saved HotpotQA AgentGraph trajectories without model execution."""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.graph_diagnostics import aggregate_trajectory_diagnostics


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _labelled_input(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input must use LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read saved AgentGraph JSONL and report atomic construction cost, "
            "structural/effective depth, topology, failures, and execution frequency."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=_labelled_input,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit per-trajectory rows from stdout",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "flowsteer.graph-search-diagnostic.v1",
        "dependency_evidence_policy": {
            "runtime_delivery": "weak",
            "structural_edge_without_delivery": "unverified",
            "verified": "requires an explicit independently validated paired-intervention receipt",
            "mask_result_auto_promoted": False,
        },
        "versions": {},
    }
    for label, path in args.input:
        if label in report["versions"]:
            raise ValueError(f"duplicate input label: {label}")
        summary = aggregate_trajectory_diagnostics(_load_jsonl(path))
        if args.summary_only:
            summary.pop("trajectories", None)
        report["versions"][label] = summary
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
