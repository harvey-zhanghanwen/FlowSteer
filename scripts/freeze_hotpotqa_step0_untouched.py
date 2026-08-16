#!/usr/bin/env python3
"""Freeze the untouched HotpotQA confirmation slice for Training-ready Step 0.

The converter and retagging logic are reused from
``prepare_agentgraph_datasets.py``.  This script only selects raw candidate
indices after the existing 128 held-out plus 512 training candidates; it does
not start a model, evaluator, or trainer.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_agentgraph_datasets import _hotpot_records, _retag_record
from src.interactive.config_loader import load_yaml
from src.interactive.task_dataset import iter_task_records


def freeze_untouched_slice(
    catalog_path: Path,
    output_path: Path,
    *,
    candidate_offset: int = 640,
    count: int = 32,
    used_paths: Sequence[Path] = (),
) -> tuple[Mapping[str, Any], ...]:
    if candidate_offset < 0 or count < 1:
        raise ValueError("candidate_offset must be non-negative and count positive")
    catalog = load_yaml(catalog_path)
    source = catalog.get("sources", {}).get("hotpotqa")
    if not isinstance(source, Mapping):
        raise ValueError("catalog has no HotpotQA source")

    candidates = tuple(
        islice(_hotpot_records(source), candidate_offset, candidate_offset + count)
    )
    if len(candidates) != count:
        raise ValueError(
            f"HotpotQA candidate stream ended after {len(candidates)} selected items"
        )
    frozen = tuple(
        _retag_record(
            item,
            split="validation",
            selection_index=candidate_offset + index,
        )
        for index, item in enumerate(candidates)
    )
    used_ids = {
        task.task_id
        for path in used_paths
        if path.exists()
        for task in iter_task_records(path)
        if task.metadata.get("dataset_key") == "hotpotqa"
    }
    overlap = sorted(str(item["task_id"]) for item in frozen if item["task_id"] in used_ids)
    if overlap:
        raise ValueError("untouched HotpotQA slice overlaps existing aligned tasks")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in frozen:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return frozen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="config/datasets_agentgraph.yaml")
    parser.add_argument(
        "--output",
        default="data/hotpotqa_training_ready_step0/untouched_validation_32.jsonl",
    )
    parser.add_argument("--candidate-offset", type=int, default=640)
    parser.add_argument("--count", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog_path = (PROJECT_ROOT / args.catalog).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    frozen = freeze_untouched_slice(
        catalog_path,
        output_path,
        candidate_offset=args.candidate_offset,
        count=args.count,
        used_paths=(
            PROJECT_ROOT / "data/agentgraph_v1/validation.jsonl",
            PROJECT_ROOT / "data/agentgraph_v1/train.jsonl",
        ),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "count": len(frozen),
                "candidate_offset": args.candidate_offset,
                "training_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
