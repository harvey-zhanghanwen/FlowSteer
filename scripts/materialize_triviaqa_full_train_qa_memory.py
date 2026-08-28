#!/usr/bin/env python3
"""Materialize every unique TriviaQA native-train Q-A source record.

This is a scale adapter over the existing TriviaQA converter and QA-memory
record/index contracts.  The native ``rc.nocontext`` train parquet repeats a
question once per evidence row, so this command keeps the first source-order
Q-A for each ``question_id`` and globally removes every fixed held-out ID.

This command only freezes the paired Q-A source projection.  Actual semantic
paraphrases are produced by the existing strict local-Qwen generator and its
semantic-admission boundary before the embedding index is built.  This source
projection must never be used as an Agent-facing retrieval index by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_agentgraph_datasets import (  # noqa: E402
    _retag_record,
    _trivia_records,
)
from src.interactive.triviaqa_embedding_index import (  # noqa: E402
    _canonical_json,
    _write_atomic_bytes,
)


MATERIALIZATION_SCHEMA_VERSION = "flowsteer.triviaqa.full_train_source_projection.v2"


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _heldout_base_ids(
    validation_tasks_path: str | Path,
    *,
    expected_count: int,
) -> tuple[str, ...]:
    """Read only the public fixed-split IDs, never evaluator answers."""

    base_ids: list[str] = []
    with Path(validation_tasks_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row, Mapping) else None
            if not isinstance(metadata, Mapping):
                continue
            if metadata.get("dataset_key") != "triviaqa":
                continue
            sampling = metadata.get("sampling")
            base_task_id = (
                sampling.get("base_task_id")
                if isinstance(sampling, Mapping)
                else None
            )
            if not isinstance(base_task_id, str) or not base_task_id.strip():
                raise ValueError(
                    f"validation row {line_number} has no base_task_id"
                )
            base_ids.append(base_task_id.strip())
    if len(base_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} TriviaQA held-out IDs, found "
            f"{len(base_ids)}"
        )
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("TriviaQA held-out base_task_id values are not unique")
    return tuple(base_ids)


def _qa_payload(record: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = record.get("question")
    metadata = record.get("metadata")
    evaluator_payload = (
        metadata.get("evaluator_payload")
        if isinstance(metadata, Mapping)
        else None
    )
    accepted_answers = (
        evaluator_payload.get("accepted_answers")
        if isinstance(evaluator_payload, Mapping)
        else None
    )
    if not isinstance(question, str) or not question.strip():
        raise ValueError("TriviaQA native record has no question")
    if not isinstance(accepted_answers, list) or not accepted_answers:
        raise ValueError("TriviaQA native record has no accepted answers")
    answers = tuple(str(answer).strip() for answer in accepted_answers)
    if any(not answer for answer in answers):
        raise ValueError("TriviaQA native record has an empty accepted answer")
    return question.strip(), answers


def project_unique_nonheldout_train(
    records: Iterable[Mapping[str, Any]],
    *,
    heldout_base_ids: Sequence[str],
    include_heldout: bool = False,
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    """Deduplicate native train and apply the explicit evaluation scope."""

    heldout = frozenset(heldout_base_ids)
    if type(include_heldout) is not bool:
        raise TypeError("include_heldout must be boolean")
    first_payload_by_id: dict[str, tuple[str, tuple[str, ...]]] = {}
    projected: list[dict[str, Any]] = []
    raw_train_rows = 0
    duplicate_raw_rows = 0
    heldout_raw_rows = 0
    heldout_post_first_rows = 0

    for record in records:
        if record.get("split") != "train":
            continue
        raw_train_rows += 1
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.startswith("triviaqa:"):
            raise ValueError("TriviaQA native train task_id is incompatible")
        payload = _qa_payload(record)
        previous = first_payload_by_id.get(task_id)
        if previous is not None:
            duplicate_raw_rows += 1
            if previous != payload:
                raise ValueError(
                    f"duplicate TriviaQA question_id has conflicting Q-A: {task_id}"
                )
            if task_id in heldout:
                heldout_raw_rows += 1
                heldout_post_first_rows += 1
            continue
        first_payload_by_id[task_id] = payload
        if task_id in heldout and not include_heldout:
            heldout_raw_rows += 1
            continue
        projected.append(
            _retag_record(
                record,
                split="train",
                selection_index=len(projected),
            )
        )

    missing_heldout = heldout.difference(first_payload_by_id)
    if missing_heldout:
        preview = ", ".join(sorted(missing_heldout)[:8])
        raise ValueError("held-out TriviaQA IDs are absent from native train: " + preview)
    stats = {
        "raw_train_row_count": raw_train_rows,
        "raw_unique_question_id_count": len(first_payload_by_id),
        "duplicate_raw_row_count": duplicate_raw_rows,
        "heldout_unique_id_count": len(heldout),
        "heldout_raw_rows_excluded": heldout_raw_rows,
        "heldout_post_first_duplicates_excluded": heldout_post_first_rows,
        "heldout_unique_ids_indexed": len(heldout) if include_heldout else 0,
        "admitted_unique_qa_count": len(projected),
    }
    return tuple(projected), stats


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    def writer(handle: Any) -> None:
        for record in records:
            handle.write(_canonical_json(record) + b"\n")

    _write_atomic_bytes(path, writer)


def materialize_full_train_sources(
    *,
    dataset_catalog_path: str | Path,
    validation_tasks_path: str | Path,
    output_dir: str | Path,
    expected_validation_count: int,
    expected_unique_train_count: int | None,
    include_heldout: bool = False,
) -> Mapping[str, Any]:
    catalog_path = Path(dataset_catalog_path).resolve()
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    sources = catalog.get("sources") if isinstance(catalog, Mapping) else None
    trivia_config = sources.get("triviaqa") if isinstance(sources, Mapping) else None
    if not isinstance(trivia_config, Mapping):
        raise ValueError("dataset catalog has no TriviaQA source")

    heldout_ids = _heldout_base_ids(
        validation_tasks_path,
        expected_count=expected_validation_count,
    )
    projected, projection_stats = project_unique_nonheldout_train(
        _trivia_records(trivia_config),
        heldout_base_ids=heldout_ids,
        include_heldout=include_heldout,
    )
    train_count = len(projected)
    if (
        expected_unique_train_count is not None
        and train_count != expected_unique_train_count
    ):
        raise ValueError(
            f"expected {expected_unique_train_count} unique non-held-out Q-A, "
            f"found {train_count}"
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_tasks_path = root / "train_tasks.jsonl"
    manifest_path = root / "source_projection_manifest.json"
    _write_jsonl(train_tasks_path, projected)

    manifest: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "source_dataset": "TriviaQA",
        "source_configuration": "rc.nocontext",
        "source_native_split": "train",
        "source_path": str(trivia_config.get("path", "")),
        "deduplication_key": "question_id",
        "deduplication_policy": "source_order_first_identical_qa",
        **projection_stats,
        "evaluation_scope": (
            "in_database_transductive"
            if include_heldout
            else "held_out_generalization"
        ),
        "validation_isolation_count": 0 if include_heldout else len(heldout_ids),
        "validation_content_indexed": include_heldout,
        "heldout_overlap_count": len(heldout_ids) if include_heldout else 0,
        "source_record_count": len(projected),
        "paraphrase_count": 0,
        "agent_facing_index_ready": False,
        "next_stage": "strict_local_qwen_semantic_paraphrase_admission",
        "files": {
            "train_tasks": train_tasks_path.name,
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
        "--dataset-catalog",
        default="config/datasets_agentgraph.yaml",
    )
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-validation-count",
        type=_positive_integer,
        default=128,
    )
    parser.add_argument(
        "--expected-unique-train-count",
        type=_positive_integer,
        default=None,
    )
    parser.add_argument(
        "--include-heldout",
        action="store_true",
        help=(
            "Include all declared evaluation Q-A source records for an explicit "
            "in-database/transductive retrieval condition."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = materialize_full_train_sources(
            dataset_catalog_path=args.dataset_catalog,
            validation_tasks_path=args.validation_tasks,
            output_dir=args.output_dir,
            expected_validation_count=args.expected_validation_count,
            expected_unique_train_count=args.expected_unique_train_count,
            include_heldout=args.include_heldout,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"TriviaQA full-train source projection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
