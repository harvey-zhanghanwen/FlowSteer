#!/usr/bin/env python3
"""Freeze HotpotQA fact-memory Top-K on 64 development sources only.

This is the HotpotQA adaptation of SkillFlow/TriviaQA's fact-memory profile
selection boundary.  It queries a temporary ``top_k=5`` fact index with the
fixed, sequential architecture-development questions and joins opaque
``memory_id`` values to source identities only through the index-external
provenance sidecar.  The 128 validation rows contribute identity fields only
for the train/validation overlap assertion; their questions, answers,
supporting facts, and evaluator payloads never enter profile selection.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.hotpotqa_full_dataset_fact_memory_index import (  # noqa: E402
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_FACT_INDEXED_TEXT_FIELD,
    HotpotQAFullDatasetFactMemoryIndex,
)
from src.interactive.task_dataset import qa_question_scope  # noqa: E402


CANDIDATE_TOP_K = (1, 2, 3, 5)
DEVELOPMENT_COUNT = 64
VALIDATION_IDENTITY_COUNT = 128
FULL_DATASET_SOURCE_COUNT = 97_852
RECEIPT_SCHEMA_VERSION = (
    "flowsteer.hotpotqa.full_dataset_fact_memory.top_k_selection.v1"
)


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _sampling_base_task_id(
    row: Mapping[str, object],
    *,
    line_number: int,
) -> tuple[str, int, bool]:
    metadata = _mapping(
        row.get("metadata"), field_name=f"row {line_number} metadata"
    )
    if metadata.get("dataset_key") != "hotpotqa":
        raise ValueError(f"row {line_number} is not a HotpotQA task")
    sampling = _mapping(
        metadata.get("sampling"), field_name=f"row {line_number} sampling"
    )
    base_task_id = _required_text(
        sampling.get("base_task_id"), field_name="base_task_id"
    )
    if not base_task_id.startswith("hotpotqa:"):
        raise ValueError("HotpotQA base_task_id is incompatible")
    selection_index = sampling.get("selection_index")
    if type(selection_index) is not int or selection_index < 0:
        raise ValueError("selection_index must be a non-negative integer")
    cycled = sampling.get("cycled_training_sample")
    if not isinstance(cycled, bool):
        raise ValueError("cycled_training_sample must be boolean")
    return base_task_id, selection_index, cycled


@dataclass(frozen=True, slots=True)
class _DevelopmentTask:
    task_id: str
    base_task_id: str
    selection_index: int
    question: str


def _load_development_tasks(path: str | Path) -> tuple[_DevelopmentTask, ...]:
    """Project exactly the first 64 unique sequential train questions."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            "fixed HotpotQA architecture-development JSONL is unavailable"
        )
    tasks: list[_DevelopmentTask] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = _mapping(
                    json.loads(line), field_name=f"development row {line_number}"
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"development JSON is invalid at line {line_number}"
                ) from exc
            if row.get("split") != "train":
                raise ValueError("HotpotQA development row must use train split")
            base_task_id, selection_index, cycled = _sampling_base_task_id(
                row, line_number=line_number
            )
            task_id = _required_text(row.get("task_id"), field_name="task_id")
            if task_id != base_task_id:
                raise ValueError(
                    "architecture-development task_id must equal base_task_id"
                )
            if selection_index != len(tasks):
                raise ValueError(
                    "HotpotQA development selection_index is not sequential"
                )
            if cycled:
                raise ValueError(
                    "architecture-development sources cannot be cycle copies"
                )
            tasks.append(
                _DevelopmentTask(
                    task_id=task_id,
                    base_task_id=base_task_id,
                    selection_index=selection_index,
                    question=qa_question_scope(
                        _required_text(row.get("question"), field_name="question")
                    ),
                )
            )
            if len(tasks) == DEVELOPMENT_COUNT:
                break
    if len(tasks) != DEVELOPMENT_COUNT:
        raise ValueError(
            f"expected {DEVELOPMENT_COUNT} HotpotQA development rows, "
            f"found {len(tasks)}"
        )
    if len({task.task_id for task in tasks}) != DEVELOPMENT_COUNT:
        raise ValueError("HotpotQA development task_id values are not unique")
    return tuple(tasks)


def _load_validation_base_task_ids(path: str | Path) -> frozenset[str]:
    """Read validation identity/split only; do not consume question or labels."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError("fixed HotpotQA validation JSONL is unavailable")
    base_task_ids: list[str] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = _mapping(
                    json.loads(line), field_name=f"validation row {line_number}"
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"validation JSON is invalid at line {line_number}"
                ) from exc
            if row.get("split") != "validation":
                raise ValueError("HotpotQA validation row has an incompatible split")
            base_task_id, selection_index, cycled = _sampling_base_task_id(
                row, line_number=line_number
            )
            if selection_index != len(base_task_ids):
                raise ValueError(
                    "HotpotQA validation selection_index is not sequential"
                )
            if cycled:
                raise ValueError("HotpotQA validation identities cannot be cycled")
            task_id = _required_text(row.get("task_id"), field_name="task_id")
            if task_id != base_task_id:
                raise ValueError("validation task_id must equal base_task_id")
            base_task_ids.append(base_task_id)
    if len(base_task_ids) != VALIDATION_IDENTITY_COUNT:
        raise ValueError(
            f"expected {VALIDATION_IDENTITY_COUNT} HotpotQA validation rows, "
            f"found {len(base_task_ids)}"
        )
    if len(set(base_task_ids)) != VALIDATION_IDENTITY_COUNT:
        raise ValueError("HotpotQA validation base_task_id values are not unique")
    return frozenset(base_task_ids)


def _load_provenance_join(path: str | Path) -> dict[str, str]:
    """Join deterministic memory IDs to source IDs without reading Q-A fields."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError("HotpotQA fact-memory provenance is unavailable")
    memory_to_source: dict[str, str] = {}
    source_ids: set[str] = set()
    with source_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = _mapping(
                    json.loads(line), field_name=f"provenance row {index + 1}"
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"provenance JSON is invalid at line {index + 1}"
                ) from exc
            source_task_id = _required_text(
                row.get("source_train_task_id"),
                field_name="provenance.source_train_task_id",
            )
            if not source_task_id.startswith("hotpotqa:"):
                raise ValueError("provenance source_train_task_id is incompatible")
            if source_task_id in source_ids:
                raise ValueError(
                    "provenance source_train_task_id values are not unique"
                )
            memory_id = f"hotpotqa-fact-{len(memory_to_source):06d}"
            memory_to_source[memory_id] = source_task_id
            source_ids.add(source_task_id)
    if len(memory_to_source) != FULL_DATASET_SOURCE_COUNT:
        raise ValueError(
            f"expected {FULL_DATASET_SOURCE_COUNT} provenance rows, "
            f"found {len(memory_to_source)}"
        )
    return memory_to_source


def _load_index_memory_ids(path: Path, *, expected_count: int) -> frozenset[str]:
    memory_ids: list[str] = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = _mapping(
                    json.loads(line), field_name=f"fact row {line_number}"
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"fact index JSON is invalid at line {line_number}"
                ) from exc
            memory_ids.append(
                _required_text(row.get("memory_id"), field_name="fact.memory_id")
            )
    if len(memory_ids) != expected_count:
        raise ValueError("fact index memory count differs from its manifest")
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("fact index memory_id values are not unique")
    return frozenset(memory_ids)


def _write_atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def select_hotpotqa_full_dataset_fact_memory_top_k(
    *,
    index_dir: str | Path,
    development_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    provenance_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Compute and atomically persist the development-only Top-K receipt."""

    development_tasks = _load_development_tasks(development_tasks_path)
    validation_base_ids = _load_validation_base_task_ids(validation_tasks_path)
    development_base_ids = {task.base_task_id for task in development_tasks}
    overlap = sorted(development_base_ids.intersection(validation_base_ids))
    if overlap:
        raise ValueError(
            "HotpotQA development and validation base_task_id values overlap: "
            + ", ".join(overlap[:8])
        )
    memory_to_source = _load_provenance_join(provenance_path)

    index_root = Path(index_dir).expanduser().resolve()
    index = HotpotQAFullDatasetFactMemoryIndex.open(index_root)
    manifest = index.manifest
    maximum = CANDIDATE_TOP_K[-1]
    if manifest.frozen_top_k != maximum:
        raise ValueError("temporary fact-memory index frozen_top_k must equal 5")
    if manifest.indexed_text_field != FULL_DATASET_FACT_INDEXED_TEXT_FIELD:
        raise ValueError("Top-K selection requires a fact_text-only index")
    if manifest.contains_raw_questions or manifest.contains_raw_answers:
        raise ValueError("Top-K selection rejects an index containing raw Q-A")
    if manifest.evaluation_scope != FULL_DATASET_EVALUATION_SCOPE:
        raise ValueError("Top-K selection evaluation scope differs")
    if manifest.official_heldout_eligible is not False:
        raise ValueError("full-dataset fact-memory cannot be held-out eligible")
    index_memory_ids = _load_index_memory_ids(
        index_root / manifest.facts_path,
        expected_count=manifest.fact_count,
    )
    if index_memory_ids != set(memory_to_source):
        raise ValueError("external provenance memory_id set differs from fact index")
    missing_development = sorted(
        development_base_ids.difference(memory_to_source.values())
    )
    if missing_development:
        raise ValueError(
            "fact-memory provenance has no development source for: "
            + ", ".join(missing_development[:8])
        )

    reciprocal_rank_sums = {candidate: 0.0 for candidate in CANDIDATE_TOP_K}
    hit_counts = {candidate: 0 for candidate in CANDIDATE_TOP_K}
    for task in development_tasks:
        hits = await index.search(task.question, maximum)
        first_relevant_rank: int | None = None
        for hit in hits:
            if memory_to_source[hit.memory_id] == task.base_task_id:
                first_relevant_rank = hit.rank
                break
        for candidate in CANDIDATE_TOP_K:
            if first_relevant_rank is not None and first_relevant_rank <= candidate:
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
            "recall": hit_counts[candidate] / DEVELOPMENT_COUNT,
            "mean_reciprocal_rank": (
                reciprocal_rank_sums[candidate] / DEVELOPMENT_COUNT
            ),
        }
        for candidate in CANDIDATE_TOP_K
    ]
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "selection_split": "train/architecture-development",
        "selection_source": "fixed_sequential_first_64_unique_train_sources",
        "selection_rule": "smallest top-k with maximal development recall",
        "candidate_top_k": list(CANDIDATE_TOP_K),
        "candidate_results": candidate_results,
        "selected_top_k": selected_top_k,
        "development_task_count": DEVELOPMENT_COUNT,
        "unique_development_base_task_count": len(development_base_ids),
        "development_task_ids": [task.task_id for task in development_tasks],
        "development_base_task_ids": [
            task.base_task_id for task in development_tasks
        ],
        "validation_identity_count": len(validation_base_ids),
        "development_validation_base_task_id_overlap_count": 0,
        "validation_used_for_selection": False,
        "validation_content_read": False,
        "validation_question_consulted": False,
        "validation_answer_or_alias_consulted": False,
        "validation_supporting_facts_consulted": False,
        "validation_evaluator_payload_consulted": False,
        "provenance_usage": "offline_memory_id_to_source_id_join_only",
        "provenance_record_count": len(memory_to_source),
        "provenance_written_to_index": False,
        "provenance_exposed_to_tool": False,
        "temporary_index_top_k": manifest.frozen_top_k,
        "index_id": manifest.index_id,
        "fact_count": manifest.fact_count,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "normalize_embeddings": manifest.normalized,
        "similarity": manifest.similarity,
        "query_encoding": "question_only",
        "fact_embedding_input_field": manifest.indexed_text_field,
        "document_format": manifest.document_format,
        "contains_raw_questions": manifest.contains_raw_questions,
        "contains_raw_answers": manifest.contains_raw_answers,
        "evaluation_scope": manifest.evaluation_scope,
        "official_heldout_eligible": manifest.official_heldout_eligible,
        "freeze_next_step": (
            "rebuild the final fact-memory index with selected_top_k and keep "
            "the fact corpus, embedding model, and Tool budget unchanged"
        ),
    }
    _write_atomic_json(Path(output_path), receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--development-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = asyncio.run(
            select_hotpotqa_full_dataset_fact_memory_top_k(
                index_dir=args.index_dir,
                development_tasks_path=args.development_tasks,
                validation_tasks_path=args.validation_tasks,
                provenance_path=args.provenance,
                output_path=args.output,
            )
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"HotpotQA fact-memory Top-K selection failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
