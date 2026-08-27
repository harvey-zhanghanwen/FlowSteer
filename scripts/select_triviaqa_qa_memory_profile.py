#!/usr/bin/env python3
"""Select TriviaQA QA-memory top-k by frozen-train self-retrieval only.

The temporary index must be built with the largest candidate as its frozen
``top_k``.  Selection uses source-task retrieval coverage on the 512 train
rows and never reads validation questions, answers, aliases, or evaluator
payloads.  The resulting selected value is then supplied to the final index
build, which freezes it in the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_embedding_index import (  # noqa: E402
    _canonical_json,
    _write_atomic_bytes,
)
from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    TriviaQAQAMemoryIndex,
    load_triviaqa_qa_memory_sources,
)


def _candidate(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("top-k candidates must be positive")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--candidate-top-k", nargs="+", type=_candidate, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-train-count", type=_positive_integer, default=512)
    parser.add_argument(
        "--expected-validation-count",
        type=_positive_integer,
        default=128,
    )
    args = parser.parse_args(argv)

    candidates = tuple(sorted(set(args.candidate_top_k)))
    sources, _validation_ids = load_triviaqa_qa_memory_sources(
        args.train_tasks,
        args.validation_tasks,
        expected_train_count=args.expected_train_count,
        expected_validation_count=args.expected_validation_count,
    )
    row_hits = {candidate: 0 for candidate in candidates}
    unique_base_hits = {candidate: set() for candidate in candidates}
    unique_bases = {source.base_task_id for source in sources}
    with TriviaQAQAMemoryIndex.open(args.index_dir) as index:
        maximum = index.manifest.frozen_top_k
        if candidates[-1] != maximum or any(candidate > maximum for candidate in candidates):
            raise ValueError(
                "largest candidate must equal the temporary index frozen_top_k"
            )
        if index.manifest.train_count != len(sources):
            raise ValueError("QA-memory index train_count differs from frozen source")
        for source in sources:
            hits = index.search(source.original_question, limit=maximum)
            hit_base_ids = [index.read(hit.passage_id).base_task_id for hit in hits]
            for candidate in candidates:
                if source.base_task_id in hit_base_ids[:candidate]:
                    row_hits[candidate] += 1
                    unique_base_hits[candidate].add(source.base_task_id)
        best_count = max(row_hits.values())
        selected_top_k = min(
            candidate
            for candidate in candidates
            if row_hits[candidate] == best_count
        )
        report = {
            "schema_version": "flowsteer.triviaqa.qa_memory.profile_selection.v1",
            "selection_split": "train",
            "validation_used_for_selection": False,
            "validation_content_read": False,
            "metric": "base_task_id self-retrieval coverage",
            "selection_rule": "smallest top-k with maximal frozen-train row coverage",
            "train_task_count": len(sources),
            "unique_train_source_count": len(unique_bases),
            "cycled_train_count": sum(source.cycled_training_sample for source in sources),
            "candidate_results": [
                {
                    "top_k": candidate,
                    "train_row_hit_count": row_hits[candidate],
                    "train_row_hit_coverage": row_hits[candidate] / len(sources),
                    "unique_source_hit_count": len(unique_base_hits[candidate]),
                    "unique_source_hit_coverage": len(unique_base_hits[candidate])
                    / len(unique_bases),
                }
                for candidate in candidates
            ],
            "selected_top_k": selected_top_k,
            "temporary_index_top_k": maximum,
            "index_id": index.manifest.index_id,
            "tool_id": index.manifest.tool_id,
            "embedding_model": index.manifest.embedding_model,
            "embedding_model_revision": index.manifest.embedding_model_revision,
            "paraphrase_version": index.manifest.paraphrase_version,
            "tool_budget": dict(index.manifest.tool_budget),
            "freeze_next_step": (
                "rebuild the final QA-memory index with selected_top_k and keep "
                "embedding/paraphrase/Tool-budget conditions unchanged"
            ),
        }
    output_path = Path(args.output)
    _write_atomic_bytes(
        output_path,
        lambda handle: handle.write(_canonical_json(report) + b"\n"),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
