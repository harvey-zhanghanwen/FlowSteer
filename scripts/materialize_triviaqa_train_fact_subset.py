#!/usr/bin/env python3
"""Materialize the frozen TriviaQA train-only fact-memory subset.

This is a selection adapter over the existing full-native fact corpus.  It
does not generate, paraphrase, embed, or retrieve anything.  The frozen
aligned train/validation files define the split; opaque fact rows are selected
through the existing external full-native provenance sidecar.  Original
questions and answers remain in the output provenance sidecar and never enter
the Agent-facing fact JSONL.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.triviaqa_embedding_index import (  # noqa: E402
    _canonical_json,
    _write_atomic_bytes,
)
from src.interactive.triviaqa_qa_memory import (  # noqa: E402
    FACT_MEMORY_RECORD_SCHEMA_VERSION,
    FACT_MEMORY_TOOL_ID,
    TriviaQAFactMemoryRecord,
    load_materialized_fact_memory,
    load_triviaqa_qa_memory_sources,
    write_materialized_fact_memory,
)
from scripts.generate_triviaqa_qa_memory_paraphrases import (  # noqa: E402
    FACT_PROVENANCE_SCHEMA_VERSION,
    write_jsonl_projection,
)


MATERIALIZATION_SCHEMA_VERSION = (
    "flowsteer.triviaqa.fact_memory.train_subset.materialization.v1"
)
_PROVENANCE_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "memory_id",
        "source_train_task_id",
        "base_task_id",
        "selection_index",
        "original_question",
        "accepted_answers",
        "canonical_answer",
        "fact_text",
    }
)
_AGENT_FACING_FIELDS = frozenset(
    {"schema_version", "memory_id", "tool_id", "fact_text"}
)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _load_full_native_provenance(
    path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load the established offline provenance without projecting it to Tool data."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(
            "full-native TriviaQA fact provenance JSONL is unavailable"
        )
    by_base_task_id: dict[str, dict[str, Any]] = {}
    by_memory_id: dict[str, dict[str, Any]] = {}
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"fact provenance JSON is invalid at line {line_number}"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"fact provenance row {line_number} is not an object"
                )
            if not _PROVENANCE_REQUIRED_FIELDS.issubset(value):
                missing = sorted(_PROVENANCE_REQUIRED_FIELDS.difference(value))
                raise ValueError(
                    f"fact provenance row {line_number} is missing: {missing}"
                )
            if value.get("schema_version") != FACT_PROVENANCE_SCHEMA_VERSION:
                raise ValueError("fact provenance schema version is unsupported")
            memory_id = _required_text(
                value.get("memory_id"), field_name="memory_id"
            )
            base_task_id = _required_text(
                value.get("base_task_id"), field_name="base_task_id"
            )
            original_question = _required_text(
                value.get("original_question"), field_name="original_question"
            )
            canonical_answer = _required_text(
                value.get("canonical_answer"), field_name="canonical_answer"
            )
            fact_text = _required_text(
                value.get("fact_text"), field_name="fact_text"
            )
            accepted_answers = value.get("accepted_answers")
            if (
                not isinstance(accepted_answers, list)
                or not accepted_answers
                or any(
                    not isinstance(answer, str) or not answer.strip()
                    for answer in accepted_answers
                )
            ):
                raise ValueError("fact provenance accepted_answers is invalid")
            if accepted_answers[0] != canonical_answer:
                raise ValueError(
                    "fact provenance canonical_answer must equal accepted_answers[0]"
                )
            normalized = dict(value)
            normalized["memory_id"] = memory_id
            normalized["base_task_id"] = base_task_id
            normalized["original_question"] = original_question
            normalized["canonical_answer"] = canonical_answer
            normalized["fact_text"] = fact_text
            if base_task_id in by_base_task_id:
                raise ValueError(
                    "full-native fact provenance base_task_id values are not unique"
                )
            if memory_id in by_memory_id:
                raise ValueError(
                    "full-native fact provenance memory_id values are not unique"
                )
            by_base_task_id[base_task_id] = normalized
            by_memory_id[memory_id] = normalized
    if not by_base_task_id:
        raise ValueError("full-native fact provenance must contain at least one row")
    return by_base_task_id, by_memory_id


def materialize_train_only_fact_subset(
    *,
    train_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    full_fact_memory_path: str | Path,
    full_provenance_path: str | Path,
    output_dir: str | Path,
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
) -> Mapping[str, Any]:
    """Select aligned training facts and prove held-out identity isolation."""

    sources, validation_base_ids = load_triviaqa_qa_memory_sources(
        train_tasks_path,
        validation_tasks_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    train_base_ids = tuple(source.base_task_id for source in sources)
    if len(set(train_base_ids)) != len(train_base_ids):
        raise ValueError(
            "TriviaQA train-only fact release requires one unique base_task_id "
            "per aligned training task"
        )

    full_facts = load_materialized_fact_memory(full_fact_memory_path)
    fact_by_memory_id = {record.memory_id: record for record in full_facts}
    provenance_by_base, provenance_by_memory = _load_full_native_provenance(
        full_provenance_path
    )
    if set(fact_by_memory_id) != set(provenance_by_memory):
        raise ValueError(
            "full-native fact corpus and provenance memory identities differ"
        )
    for memory_id, record in fact_by_memory_id.items():
        if provenance_by_memory[memory_id]["fact_text"] != record.fact_text:
            raise ValueError(
                "full-native fact corpus and provenance fact_text values differ"
            )

    missing_train = sorted(set(train_base_ids).difference(provenance_by_base))
    if missing_train:
        preview = ", ".join(missing_train[:8])
        raise ValueError(
            "aligned TriviaQA train IDs are absent from full-native provenance: "
            + preview
        )
    missing_validation = sorted(validation_base_ids.difference(provenance_by_base))
    if missing_validation:
        preview = ", ".join(missing_validation[:8])
        raise ValueError(
            "held-out TriviaQA IDs are absent from full-native provenance: "
            + preview
        )

    selected_facts: list[TriviaQAFactMemoryRecord] = []
    selected_provenance: list[dict[str, Any]] = []
    for source in sources:
        upstream = provenance_by_base[source.base_task_id]
        if upstream["original_question"] != source.original_question:
            raise ValueError(
                "aligned train question differs from full-native provenance"
            )
        if upstream["canonical_answer"] != source.canonical_answer:
            raise ValueError(
                "aligned train canonical answer differs from full-native provenance"
            )
        if tuple(upstream["accepted_answers"]) != (
            source.accepted_answers_for_admission
        ):
            raise ValueError(
                "aligned train accepted answers differ from full-native provenance"
            )
        memory_id = str(upstream["memory_id"])
        selected_facts.append(fact_by_memory_id[memory_id])
        # Keep Q-A and generation metadata only in this offline sidecar.  The
        # aligned selection fields replace the full-dataset selection offset.
        provenance_row = dict(upstream)
        provenance_row["source_train_task_id"] = source.source_train_task_id
        provenance_row["base_task_id"] = source.base_task_id
        provenance_row["selection_index"] = source.selection_index
        selected_provenance.append(provenance_row)

    selected_memory_ids = {record.memory_id for record in selected_facts}
    validation_memory_ids = {
        str(provenance_by_base[base_task_id]["memory_id"])
        for base_task_id in validation_base_ids
    }
    validation_memory_overlap = selected_memory_ids.intersection(
        validation_memory_ids
    )
    if validation_memory_overlap:
        raise ValueError("held-out validation memory entered train-only fact subset")
    if len(selected_memory_ids) != expected_train_count:
        raise ValueError(
            "train-only fact subset does not contain one unique fact per train task"
        )
    if any(set(record.to_value()) != _AGENT_FACING_FIELDS for record in selected_facts):
        raise ValueError("Agent-facing fact projection contains unsupported fields")

    root = Path(output_dir)
    facts_path = root / "facts" / "fact_memory.jsonl"
    provenance_path = root / "provenance" / "fact_provenance.jsonl"
    manifest_path = root / "materialization_manifest.json"
    write_materialized_fact_memory(facts_path, selected_facts)
    write_jsonl_projection(provenance_path, selected_provenance)

    manifest: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "source_dataset": "TriviaQA",
        "source_project_split": "train",
        "fact_memory_schema_version": FACT_MEMORY_RECORD_SCHEMA_VERSION,
        "tool_id": FACT_MEMORY_TOOL_ID,
        "train_task_count": len(sources),
        "unique_train_base_task_id_count": len(set(train_base_ids)),
        "selected_fact_count": len(selected_facts),
        "selected_provenance_count": len(selected_provenance),
        "validation_task_count": len(validation_base_ids),
        "full_native_validation_provenance_count": len(validation_base_ids),
        "train_validation_base_task_id_overlap_count": len(
            set(train_base_ids).intersection(validation_base_ids)
        ),
        "validation_memory_id_overlap_count": len(validation_memory_overlap),
        "validation_content_indexed": False,
        "agent_facing_record_fields": sorted(_AGENT_FACING_FIELDS),
        "embedding_input_field": "fact_text",
        "original_question_indexed": False,
        "accepted_answers_indexed": False,
        "canonical_answer_indexed": False,
        "provenance_loaded_by_index": False,
        "files": {
            "agent_facing_facts": str(facts_path.relative_to(root)),
            "external_provenance": str(provenance_path.relative_to(root)),
        },
    }
    _write_atomic_bytes(
        manifest_path,
        lambda handle: handle.write(_canonical_json(manifest) + b"\n"),
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-tasks",
        default="data/agentgraph_v1/train.jsonl",
    )
    parser.add_argument(
        "--validation-tasks",
        default="data/agentgraph_v1/validation.jsonl",
    )
    parser.add_argument("--full-fact-memory", required=True)
    parser.add_argument("--full-provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-train-count", type=_positive_integer, default=512
    )
    parser.add_argument(
        "--expected-validation-count", type=_positive_integer, default=128
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = materialize_train_only_fact_subset(
            train_tasks_path=args.train_tasks,
            validation_tasks_path=args.validation_tasks,
            full_fact_memory_path=args.full_fact_memory,
            full_provenance_path=args.full_provenance,
            output_dir=args.output_dir,
            expected_train_count=args.expected_train_count,
            expected_validation_count=args.expected_validation_count,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"TriviaQA train-only fact materialization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
