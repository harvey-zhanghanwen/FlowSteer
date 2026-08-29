#!/usr/bin/env python3
"""Materialize SkillFlow's fixed-100 MBPP+ evaluation population.

The source is the official EvalPlus ``MbppPlus-v0.2.0.jsonl`` release.  As in
SkillFlow's ``mbpp-plus-fixed-100@1`` protocol, tasks are ordered by the
numeric component of their canonical EvalPlus task ID and the first 100 are
selected.  The public AgentGraph records contain the prompt and public entry
point only.  EvalPlus solutions and tests are written to a separate,
evaluator-private JSONL file and are never copied into public task metadata.

This command only aligns data.  It does not run a model, execute candidate
code, evaluate a sample, or start training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_agentgraph_datasets import (
    TASK_SCHEMA_VERSION,
    SplitWriters,
    _compat_record,
    _path,
    _plain,
)


CATALOG_SCHEMA_VERSION = "flowsteer.agentgraph.mbppplus.dataset.v1"
PRIVATE_SCHEMA_VERSION = "flowsteer.agentgraph.mbppplus.evaluator-private.v1"
DATASET_KEY = "mbpp_plus"
DISPLAY_NAME = "MBPP+"
DATASET_VERSION = "v0.2.0"
PROTOCOL = "mbpp-plus-fixed-100@1"
METRIC = "pass@1"
TASK_TYPE = "code_generation"
SOURCE_COUNT = 378
EVALUATION_COUNT = 100
_TASK_ID_PATTERN = re.compile(r"^Mbpp/([1-9][0-9]*)$")
_PRIVATE_FIELDS = (
    "canonical_solution",
    "base_input",
    "plus_input",
    "contract",
    "assertion",
    "atol",
)

RowProvider = Callable[[Path], Iterable[Mapping[str, Any]]]


def _jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} has invalid {field}")
    return value


def _numeric_task_id(row: Mapping[str, Any]) -> int:
    task_id = _required_text(row, "task_id", label="EvalPlus MBPP+ row")
    match = _TASK_ID_PATTERN.fullmatch(task_id)
    if match is None:
        raise ValueError(f"invalid canonical EvalPlus MBPP+ task_id: {task_id!r}")
    return int(match.group(1))


def _validated_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_source_count: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    numeric_ids: set[int] = set()
    for source_row in rows:
        row = _plain(dict(source_row))
        if not isinstance(row, dict):
            raise TypeError("EvalPlus MBPP+ row must remain a mapping")
        task_id = _required_text(row, "task_id", label="EvalPlus MBPP+ row")
        numeric_id = _numeric_task_id(row)
        if task_id in task_ids or numeric_id in numeric_ids:
            raise ValueError(f"duplicate EvalPlus MBPP+ task identity: {task_id}")
        task_ids.add(task_id)
        numeric_ids.add(numeric_id)
        _required_text(row, "prompt", label=task_id)
        _required_text(row, "entry_point", label=task_id)
        for field in ("canonical_solution", "contract", "assertion"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"{task_id} has invalid evaluator field {field}")
        if not isinstance(row.get("base_input"), list):
            raise ValueError(f"{task_id} has invalid evaluator field base_input")
        plus_input = row.get("plus_input")
        # EvalPlus MBPP+ v0.2.0 contains one upstream row whose plus_input is
        # the exact empty mapping. Preserve that release schema while still
        # rejecting non-empty mappings; this is source-format compatibility,
        # not task- or answer-specific behavior.
        if not isinstance(plus_input, list) and plus_input != {}:
            raise ValueError(f"{task_id} has invalid evaluator field plus_input")
        if isinstance(row.get("atol"), bool) or not isinstance(
            row.get("atol"), (int, float)
        ):
            raise ValueError(f"{task_id} has invalid evaluator field atol")
        result.append(row)

    if len(result) != expected_source_count:
        raise ValueError(
            "official EvalPlus MBPP+ source count drift: "
            f"{len(result)} != {expected_source_count}"
        )
    return sorted(result, key=_numeric_task_id)


def _aligned_task_id(source_task_id: str) -> str:
    return f"mbpp-plus:{source_task_id}"


def _public_record(row: Mapping[str, Any], *, selection_index: int) -> dict[str, Any]:
    source_task_id = str(row["task_id"])
    numeric_id = _numeric_task_id(row)
    entry_point = str(row["entry_point"])
    record = _compat_record(
        dataset_key=DATASET_KEY,
        source=DISPLAY_NAME,
        task_id=_aligned_task_id(source_task_id),
        question=str(row["prompt"]),
        ground_truth=None,
        split="test",
        task_type=TASK_TYPE,
        metric=METRIC,
        context=(),
        extra={
            "benchmark": DISPLAY_NAME,
            "dataset_version": DATASET_VERSION,
            "protocol": PROTOCOL,
            "source_task_id": source_task_id,
            "canonical_numeric_task_id": numeric_id,
            "entry_point": entry_point,
            "selection_index": selection_index,
            "ground_truth_role": "evaluator_only_redacted",
        },
        preserve_question_text=True,
    )
    metadata = dict(record["metadata"])
    metadata.update(
        {
            "benchmark": DISPLAY_NAME,
            "dataset_version": DATASET_VERSION,
            "protocol": PROTOCOL,
            "source_task_id": source_task_id,
            "canonical_numeric_task_id": numeric_id,
            "entry_point": entry_point,
            "selection_index": selection_index,
            "evaluation_role": "fixed-held-out-evaluation",
            "ground_truth_role": "evaluator_only_redacted",
        }
    )
    record["metadata"] = metadata
    return record


def _private_record(row: Mapping[str, Any]) -> dict[str, Any]:
    source_task_id = str(row["task_id"])
    return {
        "schema_version": PRIVATE_SCHEMA_VERSION,
        "task_id": _aligned_task_id(source_task_id),
        "source_task_id": source_task_id,
        "evaluator_payload": _plain(dict(row)),
    }


def _assert_public_redaction(records: Iterable[Mapping[str, Any]]) -> None:
    for record in records:
        if record.get("ground_truth") is not None or record.get("answer") is not None:
            raise ValueError("public MBPP+ records must redact evaluator ground truth")
        public_payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        for field in _PRIVATE_FIELDS:
            if f'"{field}"' in public_payload:
                raise ValueError(
                    f"public MBPP+ record exposes evaluator-private field: {field}"
                )


def _write_private_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def prepare(
    catalog_path: Path,
    *,
    row_provider: RowProvider | None = None,
) -> Path:
    catalog_path = catalog_path.expanduser().resolve()
    repo_root = catalog_path.parent.parent
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("MBPP+ catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported MBPP+ catalog schema")
    if catalog.get("task_schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("unsupported AgentGraph task schema")

    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("MBPP+ sources must be a mapping")
    official = sources.get("official_evalplus")
    if not isinstance(official, Mapping):
        raise ValueError("official EvalPlus MBPP+ source is required")
    if official.get("dataset_version") != DATASET_VERSION:
        raise ValueError("unsupported EvalPlus MBPP+ dataset version")

    policy = catalog.get("split_policy")
    expected_policy = {
        "mode": "official_evaluation_only",
        "protocol": PROTOCOL,
        "ordering": "canonical_numeric_task_id_ascending",
        "source_count": SOURCE_COUNT,
        "test_count": EVALUATION_COUNT,
        "training_enabled": False,
        "metric": METRIC,
    }
    if not isinstance(policy, Mapping):
        raise ValueError("MBPP+ split_policy must be a mapping")
    drift = {
        key: {"expected": value, "actual": policy.get(key)}
        for key, value in expected_policy.items()
        if policy.get(key) != value
    }
    if drift:
        raise ValueError(
            "MBPP+ fixed-100 split policy drift: "
            + json.dumps(drift, sort_keys=True)
        )

    source_path = _path(str(official["path"]), base=repo_root)
    read_rows = row_provider or _jsonl_rows
    ordered = _validated_rows(
        read_rows(source_path), expected_source_count=SOURCE_COUNT
    )
    selected = ordered[:EVALUATION_COUNT]
    if len(selected) != EVALUATION_COUNT:
        raise ValueError("MBPP+ fixed-100 selection is incomplete")
    public = [
        _public_record(row, selection_index=index)
        for index, row in enumerate(selected)
    ]
    private = [_private_record(row) for row in selected]
    _assert_public_redaction(public)

    output_dir = _path(str(catalog["aligned_dir"]), base=repo_root)
    writers = SplitWriters(output_dir)
    for record in public:
        writers.write(record)

    private_temp = writers.temp_dir.parent / (
        f".{writers.temp_dir.name}-evaluator-private.jsonl"
    )
    _write_private_rows(private_temp, private)
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "evaluator_private_schema_version": PRIVATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path),
        "source": {
            "name": "EvalPlus MBPP+",
            "dataset_version": DATASET_VERSION,
            "path": str(source_path),
            "row_count": len(ordered),
        },
        "protocol": PROTOCOL,
        "metric": METRIC,
        "training_started": False,
        "evaluation_started": False,
        "split_policy": dict(expected_policy),
        "counts_by_split": {
            "train": 0,
            "validation": 0,
            "test": len(public),
        },
        "selection": {
            "ordering": "canonical_numeric_task_id_ascending",
            "first_source_task_id": selected[0]["task_id"],
            "last_source_task_id": selected[-1]["task_id"],
            "selected_count": len(selected),
        },
        "visibility": {
            "public_ground_truth": None,
            "public_evaluator_payload": False,
            "evaluator_private_file": "evaluator_private.jsonl",
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "evaluator_private": "evaluator_private.jsonl",
        },
    }
    try:
        writers.publish(manifest)
        private_temp.replace(output_dir / "evaluator_private.jsonl")
    finally:
        if private_temp.exists():
            private_temp.unlink()
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="config/datasets_mbppplus_v1.yaml",
        help="MBPP+ dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(Path(args.catalog))


if __name__ == "__main__":
    main()
