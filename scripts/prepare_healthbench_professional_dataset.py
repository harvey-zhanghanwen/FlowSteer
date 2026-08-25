#!/usr/bin/env python3
"""Materialize the official HealthBench Professional public test boundary.

The shared ``TaskRecord`` shape and ``SplitWriters`` are reused from
``prepare_agentgraph_datasets.py``.  The task-specific adaptation separates
the 525 public conversations from rubric/reference/slice fields before any
model runtime can load the records.  Evaluator-only cases are joined by
``task_id`` from ``private_cases.jsonl``.

This command performs data alignment only.  It starts no model, Tool, grader,
training job, backward pass, optimizer update, or Skill process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_agentgraph_datasets import (  # noqa: E402
    TASK_SCHEMA_VERSION,
    SplitWriters,
    _compat_record,
    _path,
)
from src.interactive.healthbench_professional_adapter import (  # noqa: E402
    HEALTHBENCH_PROFESSIONAL_DATASET_KEY,
    HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE,
    HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION,
    HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION,
    HEALTHBENCH_PROFESSIONAL_PUBLIC_COUNT,
    HEALTHBENCH_PROFESSIONAL_SOURCE_ID,
    HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT,
    HEALTHBENCH_PROFESSIONAL_TASK_FAMILY,
    evaluator_case_from_official_row,
    public_task_record_fields,
    validate_official_healthbench_professional_row,
)


HEALTHBENCH_PROFESSIONAL_CATALOG_SCHEMA_VERSION = (
    "flowsteer.agentgraph.healthbench-professional.dataset.v1"
)


def _read_official_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(
                validate_official_healthbench_professional_row(
                    raw,
                    line_number=line_number,
                )
            )
    return tuple(rows)


def _public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = public_task_record_fields(row)
    task_id = fields["task_id"]
    source_id = fields["source_id"]
    record = _compat_record(
        dataset_key=HEALTHBENCH_PROFESSIONAL_DATASET_KEY,
        source=HEALTHBENCH_PROFESSIONAL_SOURCE_ID,
        task_id=task_id,
        question=fields["question"],
        ground_truth=None,
        split="test",
        task_type="conversation_response",
        metric="healthbench_professional_score",
        extra={
            "task_family": HEALTHBENCH_PROFESSIONAL_TASK_FAMILY,
            "source_split": HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT,
            "benchmark_slice": "public_test",
            "evaluator_route": HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE,
            "evaluator_source_id": source_id,
            "evaluator_version": HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION,
        },
        preserve_question_text=True,
    )
    metadata = dict(record["metadata"])
    metadata.update(
        {
            "task_family": HEALTHBENCH_PROFESSIONAL_TASK_FAMILY,
            "source_split": HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT,
            "benchmark_slice": "public_test",
            "evaluator_route": HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE,
            "evaluator_source_id": source_id,
            "evaluator_version": HEALTHBENCH_PROFESSIONAL_EVALUATOR_VERSION,
        }
    )
    record["metadata"] = metadata
    record["conversation"] = fields["conversation"]
    record["evaluator_route"] = HEALTHBENCH_PROFESSIONAL_EVALUATOR_ROUTE
    record["evaluator_source_id"] = source_id
    return record


def _write_private_cases(
    output_dir: Path,
    cases: tuple[Mapping[str, Any], ...],
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="x",
        encoding="utf-8",
        prefix="healthbench-evaluator-cases-",
        suffix=".jsonl",
        dir=str(output_dir.parent),
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        for case in cases:
            handle.write(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary_path.replace(output_dir / "private_cases.jsonl")


def prepare(catalog_path: Path) -> Path:
    repo_root = catalog_path.resolve().parent.parent
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("HealthBench catalog must be an object")
    if catalog.get("schema_version") != (
        HEALTHBENCH_PROFESSIONAL_CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("unsupported HealthBench Professional dataset catalog schema")
    if catalog.get("task_schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("unsupported TaskRecord schema")

    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("HealthBench catalog sources must be an object")
    if sources.get("dataset_id") != HEALTHBENCH_PROFESSIONAL_SOURCE_ID:
        raise ValueError("HealthBench catalog dataset_id differs from official source")
    if sources.get("source_split") != HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT:
        raise ValueError("HealthBench catalog source_split must be test")
    expected_count = sources.get("expected_count")
    if expected_count != HEALTHBENCH_PROFESSIONAL_PUBLIC_COUNT:
        raise ValueError(
            "HealthBench catalog expected_count must match the official 525 rows"
        )

    source_path = _path(str(sources["source_path"]), base=repo_root)
    rows = _read_official_rows(source_path)
    if len(rows) != expected_count:
        raise ValueError(
            f"official HealthBench source has {len(rows)} rows; expected {expected_count}"
        )
    source_ids = [str(row["id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("official HealthBench source ids must be unique")

    public_records = tuple(_public_record(row) for row in rows)
    private_cases = tuple(evaluator_case_from_official_row(row) for row in rows)
    if [record["task_id"] for record in public_records] != [
        case["task_id"] for case in private_cases
    ]:
        raise ValueError("HealthBench public/private task-id order differs")

    output_dir = _path(str(catalog["aligned_dir"]), base=repo_root)
    writers = SplitWriters(output_dir)
    for record in public_records:
        writers.write(record)
    manifest = {
        "schema_version": HEALTHBENCH_PROFESSIONAL_CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "private_case_schema_version": (
            HEALTHBENCH_PROFESSIONAL_PRIVATE_CASE_SCHEMA_VERSION
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path.resolve()),
        "dataset_id": HEALTHBENCH_PROFESSIONAL_SOURCE_ID,
        "source_split": HEALTHBENCH_PROFESSIONAL_SOURCE_SPLIT,
        "source_count": len(rows),
        "training_enabled": False,
        "evaluation_only": True,
        "model_visible_fields": [
            "task_id",
            "question",
            "conversation",
            "evaluator_route",
            "evaluator_source_id",
        ],
        "evaluator_only_fields": [
            "rubric_items",
            "physician_response",
            "evaluator_metadata",
        ],
        "private_grader_public_input_copy": "prompt",
        "evaluator_join": {
            "public_file": "test.jsonl",
            "private_file": "private_cases.jsonl",
            "key": "task_id",
        },
        "counts_by_split": {
            "train": 0,
            "validation": 0,
            "test": len(public_records),
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "evaluator_private_cases": "private_cases.jsonl",
        },
    }
    writers.publish(manifest)
    _write_private_cases(output_dir, private_cases)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="config/datasets_healthbench_professional_official_v1.yaml",
        help="HealthBench Professional dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(Path(args.catalog))


if __name__ == "__main__":
    main()
