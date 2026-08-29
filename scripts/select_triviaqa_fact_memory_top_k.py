#!/usr/bin/env python3
"""Freeze TriviaQA fact-memory Top-K from architecture-development retrieval.

Selection is an offline control-plane operation.  It queries the fact index
with the fixed architecture-development questions, joins opaque ``memory_id``
values to ``base_task_id`` only through the external provenance sidecar, and
computes Recall@1/3/5 plus MRR.  Validation contributes identity fields only
for split-isolation checks.  Provenance is never written into the index or a
Tool payload.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_embedding_index import (  # noqa: E402
    EmbeddingEncoder,
    _canonical_json,
    _write_atomic_bytes,
)
from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    FACT_MEMORY_RECORD_KIND,
    TriviaQAFactMemoryIndex,
    load_materialized_fact_memory,
)


CANDIDATE_TOP_K = (1, 3, 5)
RECEIPT_SCHEMA_VERSION = (
    "flowsteer.triviaqa.fact_memory.top_k_selection.v1"
)


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sampling_base_task_id(
    row: Mapping[str, object],
    *,
    line_number: int,
) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("dataset_key") != "triviaqa":
        raise ValueError(
            f"TriviaQA task row {line_number} has incompatible metadata"
        )
    sampling = metadata.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(
            f"TriviaQA task row {line_number} has no sampling metadata"
        )
    base_task_id = _required_text(
        sampling.get("base_task_id"),
        field_name="base_task_id",
    )
    if not base_task_id.startswith("triviaqa:"):
        raise ValueError("TriviaQA base_task_id is incompatible")
    return base_task_id


@dataclass(frozen=True, slots=True)
class _DevelopmentTask:
    task_id: str
    base_task_id: str
    selection_index: int
    question: str


def _load_development_tasks(
    path: str | Path,
    *,
    expected_count: int,
) -> tuple[_DevelopmentTask, ...]:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            "fixed TriviaQA architecture-development JSONL is unavailable"
        )
    tasks: list[_DevelopmentTask] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"development JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"development row {line_number} is not an object"
                )
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get(
                "dataset_key"
            ) != "triviaqa":
                continue
            if row.get("split") != "train":
                raise ValueError(
                    "TriviaQA architecture-development row must use train split"
                )
            sampling = metadata.get("sampling")
            if not isinstance(sampling, Mapping):
                raise ValueError(
                    f"TriviaQA task row {line_number} has no sampling metadata"
                )
            selection_index = sampling.get("selection_index")
            if type(selection_index) is not int or selection_index < 0:
                raise ValueError("selection_index must be a non-negative integer")
            tasks.append(
                _DevelopmentTask(
                    task_id=_required_text(
                        row.get("task_id"),
                        field_name="task_id",
                    ),
                    base_task_id=_sampling_base_task_id(
                        row,
                        line_number=line_number,
                    ),
                    selection_index=selection_index,
                    question=_required_text(
                        row.get("question"),
                        field_name="question",
                    ),
                )
            )
    if len(tasks) != expected_count:
        raise ValueError(
            f"expected {expected_count} TriviaQA development rows, "
            f"found {len(tasks)}"
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("TriviaQA development task_id values are not unique")
    if sorted(task.selection_index for task in tasks) != list(range(expected_count)):
        raise ValueError(
            "TriviaQA development selection_index differs from frozen order"
        )
    return tuple(sorted(tasks, key=lambda item: item.selection_index))


def _load_validation_base_task_ids(
    path: str | Path,
    *,
    expected_count: int,
) -> frozenset[str]:
    """Read validation identity/split only; never consume its question/labels."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError("fixed TriviaQA validation JSONL is unavailable")
    base_task_ids: list[str] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"validation JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"validation row {line_number} is not an object")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get(
                "dataset_key"
            ) != "triviaqa":
                continue
            if row.get("split") != "validation":
                raise ValueError("TriviaQA validation row has an incompatible split")
            base_task_ids.append(
                _sampling_base_task_id(row, line_number=line_number)
            )
    if len(base_task_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} TriviaQA validation rows, "
            f"found {len(base_task_ids)}"
        )
    if len(set(base_task_ids)) != len(base_task_ids):
        raise ValueError("TriviaQA validation base_task_id values are not unique")
    return frozenset(base_task_ids)


def _load_provenance_join(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load only opaque memory/source/base identities from the external sidecar."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError("TriviaQA fact-memory provenance is unavailable")
    memory_to_base: dict[str, str] = {}
    memory_to_source: dict[str, str] = {}
    source_ids: set[str] = set()
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"provenance JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"provenance row {line_number} is not an object")
            memory_id = _required_text(
                row.get("memory_id"),
                field_name="provenance.memory_id",
            )
            source_task_id = _required_text(
                row.get("source_train_task_id"),
                field_name="provenance.source_train_task_id",
            )
            base_task_id = _required_text(
                row.get("base_task_id"),
                field_name="provenance.base_task_id",
            )
            if not base_task_id.startswith("triviaqa:"):
                raise ValueError("provenance base_task_id is incompatible")
            if memory_id in memory_to_base:
                raise ValueError("provenance memory_id values are not unique")
            if source_task_id in source_ids:
                raise ValueError(
                    "provenance source_train_task_id values are not unique"
                )
            memory_to_base[memory_id] = base_task_id
            memory_to_source[memory_id] = source_task_id
            source_ids.add(source_task_id)
    if not memory_to_base:
        raise ValueError("fact-memory provenance sidecar is empty")
    return memory_to_base, memory_to_source


def select_triviaqa_fact_memory_top_k(
    *,
    index_dir: str | Path,
    development_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    provenance_path: str | Path,
    output_path: str | Path,
    expected_development_count: int = 512,
    expected_validation_count: int = 128,
    encoder: EmbeddingEncoder | None = None,
) -> dict[str, object]:
    """Compute and atomically persist the frozen development-only receipt."""

    if type(expected_development_count) is not int or expected_development_count < 1:
        raise ValueError("expected_development_count must be positive")
    if type(expected_validation_count) is not int or expected_validation_count < 1:
        raise ValueError("expected_validation_count must be positive")
    development_tasks = _load_development_tasks(
        development_tasks_path,
        expected_count=expected_development_count,
    )
    validation_base_ids = _load_validation_base_task_ids(
        validation_tasks_path,
        expected_count=expected_validation_count,
    )
    development_base_ids = {
        task.base_task_id for task in development_tasks
    }
    overlap = sorted(development_base_ids.intersection(validation_base_ids))
    if overlap:
        raise ValueError(
            "TriviaQA development and validation base_task_id values overlap: "
            + ", ".join(overlap[:8])
        )
    memory_to_base, memory_to_source = _load_provenance_join(provenance_path)

    reciprocal_rank_sums = {candidate: 0.0 for candidate in CANDIDATE_TOP_K}
    hit_counts = {candidate: 0 for candidate in CANDIDATE_TOP_K}
    with TriviaQAFactMemoryIndex.open(index_dir, encoder=encoder) as index:
        if index.manifest.record_kind != FACT_MEMORY_RECORD_KIND:
            raise ValueError("Top-K selection requires a fact-memory index")
        maximum = CANDIDATE_TOP_K[-1]
        if index.manifest.frozen_top_k != maximum:
            raise ValueError(
                "temporary fact-memory index frozen_top_k must equal 5"
            )
        facts_path = index.root / index.manifest.files["facts"]["name"]
        index_memory_ids = {
            record.memory_id
            for record in load_materialized_fact_memory(
                facts_path,
                expected_count=index.manifest.memory_count,
            )
        }
        if index_memory_ids != set(memory_to_base):
            raise ValueError(
                "external provenance memory_id set differs from fact index"
            )
        if set(memory_to_source) != index_memory_ids:
            raise ValueError("external provenance source join is incomplete")
        missing_development_bases = sorted(
            development_base_ids.difference(memory_to_base.values())
        )
        if missing_development_bases:
            raise ValueError(
                "fact-memory provenance has no development source for: "
                + ", ".join(missing_development_bases[:8])
            )

        for task in development_tasks:
            hits = index.search(task.question, limit=maximum)
            first_relevant_rank: int | None = None
            for hit in hits:
                if memory_to_base[hit.memory_id] == task.base_task_id:
                    first_relevant_rank = hit.rank
                    break
            for candidate in CANDIDATE_TOP_K:
                if (
                    first_relevant_rank is not None
                    and first_relevant_rank <= candidate
                ):
                    hit_counts[candidate] += 1
                    reciprocal_rank_sums[candidate] += 1.0 / first_relevant_rank

        maximum_hit_count = max(hit_counts.values())
        selected_top_k = min(
            candidate
            for candidate in CANDIDATE_TOP_K
            if hit_counts[candidate] == maximum_hit_count
        )
        candidate_results = [
            {
                "top_k": candidate,
                "hit_count": hit_counts[candidate],
                "recall": hit_counts[candidate] / len(development_tasks),
                "mean_reciprocal_rank": (
                    reciprocal_rank_sums[candidate] / len(development_tasks)
                ),
            }
            for candidate in CANDIDATE_TOP_K
        ]
        receipt: dict[str, object] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "selection_split": "architecture_development",
            "selection_rule": "smallest top-k with maximal development recall",
            "candidate_top_k": list(CANDIDATE_TOP_K),
            "candidate_results": candidate_results,
            "selected_top_k": selected_top_k,
            "development_task_count": len(development_tasks),
            "unique_development_base_task_count": len(development_base_ids),
            "validation_identity_count": len(validation_base_ids),
            "development_validation_base_task_id_overlap_count": 0,
            "validation_used_for_selection": False,
            "validation_content_read": False,
            "provenance_usage": "offline_memory_id_to_source_and_base_join_only",
            "provenance_record_count": len(memory_to_base),
            "provenance_written_to_index": False,
            "provenance_exposed_to_tool": False,
            "temporary_index_top_k": index.manifest.frozen_top_k,
            "index_id": index.manifest.index_id,
            "record_kind": index.manifest.record_kind,
            "tool_id": index.manifest.tool_id,
            "embedding_model": index.manifest.embedding_model,
            "embedding_model_revision": index.manifest.embedding_model_revision,
            "query_encoding": "BGE query_prefix + development question",
            "fact_embedding_input_field": index.manifest.embedding_input_field,
            "freeze_next_step": (
                "rebuild the final fact-memory index with selected_top_k and "
                "keep the fact corpus, embedding model, and Tool budget unchanged"
            ),
        }

    target = Path(output_path)
    _write_atomic_bytes(
        target,
        lambda handle: handle.write(_canonical_json(receipt) + b"\n"),
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--development-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-development-count",
        type=_positive_integer,
        default=512,
    )
    parser.add_argument(
        "--expected-validation-count",
        type=_positive_integer,
        default=128,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = select_triviaqa_fact_memory_top_k(
            index_dir=args.index_dir,
            development_tasks_path=args.development_tasks,
            validation_tasks_path=args.validation_tasks,
            provenance_path=args.provenance,
            output_path=args.output,
            expected_development_count=args.expected_development_count,
            expected_validation_count=args.expected_validation_count,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"TriviaQA fact-memory Top-K selection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
