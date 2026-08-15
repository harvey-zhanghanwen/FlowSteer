#!/usr/bin/env python3
"""Validate the aligned 128-held-out/512-train dataset view."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interactive.task_dataset import iter_task_records  # noqa: E402


def validate(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    recipe = manifest["alignment_recipe"]
    heldout_split = str(recipe["heldout_split"])
    heldout_target = int(recipe["heldout_count_per_dataset"])
    train_target = int(recipe["train_count_per_dataset"])

    counts: dict[str, Counter[str]] = {}
    base_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    official_aime = Counter()
    cycled = Counter()
    for split in ("train", "validation", "test"):
        split_counts: Counter[str] = Counter()
        for record in iter_task_records(
            data_dir / f"{split}.jsonl", expected_split=split
        ):
            source = str(record.metadata["source"])
            split_counts[source] += 1
            sampling = record.metadata["sampling"]
            base_ids[split][source].add(str(sampling["base_task_id"]))
            if sampling["cycled_training_sample"]:
                if split != "train":
                    raise ValueError("cycled sample found outside the training split")
                cycled[source] += 1
            extra = record.metadata.get("skillflow", {}).get("extra", {})
            if extra.get("benchmark_slice") == "official_aime_2026":
                official_aime[split] += 1
        counts[split] = split_counts

    sources = sorted(counts["train"])
    for source in sources:
        if counts["train"][source] != train_target:
            raise ValueError(f"{source}: expected {train_target} train records")
        if counts[heldout_split][source] != heldout_target:
            raise ValueError(f"{source}: expected {heldout_target} held-out records")
        overlap = base_ids["train"][source] & base_ids[heldout_split][source]
        if overlap:
            raise ValueError(f"{source}: held-out base tasks leaked into training")
    if official_aime != Counter({heldout_split: 30}):
        raise ValueError("all 30 official AIME 2026 tasks must be held out")

    return {
        "valid": True,
        "heldout_split": heldout_split,
        "counts": {
            split: dict(sorted(split_counts.items()))
            for split, split_counts in counts.items()
        },
        "unique_train_base_tasks": {
            source: len(base_ids["train"][source]) for source in sources
        },
        "cycled_train_records": dict(sorted(cycled.items())),
        "official_aime_2026": dict(official_aime),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/agentgraph_v1")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    print(json.dumps(validate(data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
