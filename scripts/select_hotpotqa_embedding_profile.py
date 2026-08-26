#!/usr/bin/env python3
"""Select and freeze HotpotQA retrieval bounds on development tasks only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_embedding_index import (
    _encode,
    _load_sentence_transformer,
    _task_native_id,
)
from src.interactive.task_dataset import iter_task_records


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _development_rows(
    parquet_paths: Sequence[Path],
    task_ids: Sequence[str],
) -> Mapping[str, Mapping[str, object]]:
    import pandas as pd

    by_native = {_task_native_id(task_id): task_id for task_id in task_ids}
    rows: dict[str, Mapping[str, object]] = {}
    for path in sorted(parquet_paths):
        frame = pd.read_parquet(
            path,
            columns=["id", "context", "supporting_facts"],
        )
        for row in frame.itertuples(index=False):
            task_id = by_native.get(str(row.id))
            if task_id is not None and task_id not in rows:
                rows[task_id] = {
                    "context": row.context,
                    "supporting_facts": row.supporting_facts,
                }
    if set(rows) != set(task_ids):
        raise ValueError("development profile source is missing requested tasks")
    return rows


def select_profile(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    section = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    count = int(section["development_sample_count"])
    tasks = tuple(
        task
        for task in iter_task_records(
            _resolve(root, section["development_tasks"]),
            expected_split="train",
        )
        if task.task_id.startswith("hotpotqa:")
    )[:count]
    if len(tasks) != count:
        raise ValueError("insufficient HotpotQA development tasks")
    validation_ids = {
        task.task_id
        for task in iter_task_records(
            _resolve(root, section["frozen_validation_tasks"]),
            expected_split="validation",
        )
    }
    if validation_ids & {task.task_id for task in tasks}:
        raise ValueError("development profile selection overlaps validation")

    parquet_paths = tuple(
        Path(value).expanduser().resolve()
        for value in sorted(glob.glob(str(section["source_parquet_glob"])))
    )
    rows = _development_rows(parquet_paths, [task.task_id for task in tasks])
    model = _load_sentence_transformer(
        str(section["embedding_model"]),
        str(section["embedding_device"]),
    )
    candidates = (2, 3, 4, 5)
    full_support: dict[int, int] = {candidate: 0 for candidate in candidates}
    mean_recall: dict[int, float] = {candidate: 0.0 for candidate in candidates}
    supporting_counts: list[int] = []
    for task in tasks:
        row = rows[task.task_id]
        context = _mapping(row["context"], "context")
        titles = [str(value) for value in context["title"]]
        sentence_groups = context["sentences"]
        texts = [
            f"{title}\n{''.join(str(item) for item in sentences)}"
            for title, sentences in zip(titles, sentence_groups, strict=True)
        ]
        passage_vectors = _encode(model, texts)
        query_vector = _encode(model, [task.question], batch_size=1)[0]
        ranking = sorted(
            range(len(titles)),
            key=lambda index: (-float(np.dot(passage_vectors[index], query_vector)), index),
        )
        supporting = {
            str(value)
            for value in _mapping(
                row["supporting_facts"], "supporting_facts"
            )["title"]
        }
        supporting_counts.append(len(supporting))
        for candidate in candidates:
            retrieved = {titles[index] for index in ranking[:candidate]}
            overlap = len(supporting & retrieved)
            recall = overlap / len(supporting) if supporting else 1.0
            mean_recall[candidate] += recall
            full_support[candidate] += overlap == len(supporting)
    for candidate in candidates:
        mean_recall[candidate] /= len(tasks)
    chosen_top_k = min(
        candidates,
        key=lambda candidate: (
            -full_support[candidate] / len(tasks),
            -mean_recall[candidate],
            candidate,
        ),
    )
    # Development tasks are two-hop and expose two answer-free public evidence
    # documents. Freeze two search/read pairs; no validation answer is consulted.
    max_support_documents = max(supporting_counts, default=2)
    chosen_tool_budget = max(2, 2 * max_support_documents)
    chosen_max_turns = chosen_tool_budget + 2
    return {
        "schema_version": "flowsteer.hotpotqa.embedding_profile_selection.v1",
        "selection_split": "train/architecture-development",
        "validation_answers_consulted": False,
        "development_sample_count": len(tasks),
        "development_task_ids": [task.task_id for task in tasks],
        "embedding_model": str(section["embedding_model_id"]),
        "normalize_embeddings": True,
        "similarity": "cosine",
        "candidate_top_k": list(candidates),
        "full_support_recall_rate_by_top_k": {
            str(candidate): full_support[candidate] / len(tasks)
            for candidate in candidates
        },
        "mean_support_title_recall_by_top_k": {
            str(candidate): mean_recall[candidate] for candidate in candidates
        },
        "selected_top_k": chosen_top_k,
        "selected_max_tool_calls_per_agent_call": chosen_tool_budget,
        "selected_max_turns_per_agent_call": chosen_max_turns,
        "selection_rule": (
            "maximize full supporting-title recall, then mean supporting-title "
            "recall, then choose the smallest top-k; Tool budget is two "
            "search/read operations per maximum development supporting document"
        ),
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_embedding_retrieval_v2.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hotpotqa_embedding_retrieval_v1/profile_selection.json",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    value = select_profile(config_path)
    output = _resolve(config_path.parent.parent, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
