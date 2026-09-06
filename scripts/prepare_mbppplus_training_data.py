#!/usr/bin/env python3
"""Materialize disjoint MBPP training and validation populations.

The candidates come from FlowSteer's public ``train_12k.jsonl`` and
``eval/mbpp.jsonl`` releases.  Every candidate that overlaps any task in the
complete EvalPlus MBPP+ v0.2.0 population by task ID, normalized problem text,
or entry point is removed.  This is deliberately stronger than excluding only
the fixed-100 evaluation subset.

Public AgentGraph records contain the problem, public test program, and entry
point but never the reference solution.  The reference solution and the same
test program are kept in ``evaluator_private.jsonl`` for the SkillFlow
``code_test_pass_rate`` training reward.  The existing fixed-100 MBPP+
evaluation files are referenced by the catalog and are never opened for
writing by this command.

This command only aligns data.  It does not execute code, call a model, or
start training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping
import unicodedata

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


CATALOG_SCHEMA_VERSION = "flowsteer.agentgraph.mbppplus.training-dataset.v1"
PRIVATE_SCHEMA_VERSION = (
    "flowsteer.agentgraph.mbppplus.training-evaluator-private.v1"
)
PROTOCOL = "flowsteer-mbpp-no-evalplus-overlap@1"
DATASET_KEY = "mbpp_plus"
DISPLAY_NAME = "FlowSteer MBPP"
TASK_TYPE = "code_generation"
METRIC = "test_pass_rate"
TRAINING_EVALUATOR = "skillflow.training.reward.code_test_pass_rate"

FLOWSTEER_TOTAL_ROWS = 12_000
FLOWSTEER_MBPP_TRAIN_ROWS = 2_000
FLOWSTEER_MBPP_TRAIN_UNIQUE = 374
FLOWSTEER_MBPP_VALIDATION_ROWS = 128
EVALPLUS_ROWS = 378
EXPECTED_TRAIN_EXCLUDED = 133
EXPECTED_VALIDATION_EXCLUDED = 66
EXPECTED_TRAIN_OUTPUT = 241
EXPECTED_VALIDATION_OUTPUT = 62

_EVALPLUS_TASK_ID = re.compile(r"^Mbpp/([1-9][0-9]*)$")
_FLOWSTEER_TASK_ID = re.compile(r"^[1-9][0-9]*$")

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


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _normalized_problem(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _flowsteer_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    meta = row.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("FlowSteer MBPP row has invalid meta")
    task_id = str(meta.get("task_id", "")).strip()
    if _FLOWSTEER_TASK_ID.fullmatch(task_id) is None:
        raise ValueError(f"invalid FlowSteer MBPP task_id: {task_id!r}")
    problem = _required_text(row.get("problem"), label=f"MBPP/{task_id} problem")
    entry_point = _required_text(
        meta.get("entry_point"), label=f"MBPP/{task_id} entry_point"
    ).strip()
    return task_id, _normalized_problem(problem), entry_point


def _validated_flowsteer_train(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    total = 0
    mbpp_rows = 0
    unique: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for source_row in rows:
        total += 1
        row = _plain(dict(source_row))
        if not isinstance(row, dict):
            raise TypeError("FlowSteer row must remain a mapping")
        if row.get("source") != "mbpp":
            continue
        mbpp_rows += 1
        if row.get("problem_type") != "code":
            raise ValueError("FlowSteer MBPP training row is not code")
        task_id, _, _ = _flowsteer_identity(row)
        meta = row["meta"]
        _required_text(meta.get("test"), label=f"MBPP/{task_id} test")
        _required_text(
            row.get("ground_truth"), label=f"MBPP/{task_id} reference solution"
        )
        previous = seen.get(task_id)
        if previous is not None:
            if previous != row:
                raise ValueError(
                    f"conflicting duplicate FlowSteer MBPP task_id: {task_id}"
                )
            continue
        seen[task_id] = row
        unique.append(row)

    if total != FLOWSTEER_TOTAL_ROWS:
        raise ValueError(
            f"FlowSteer train_12k row-count drift: {total} != {FLOWSTEER_TOTAL_ROWS}"
        )
    if mbpp_rows != FLOWSTEER_MBPP_TRAIN_ROWS:
        raise ValueError(
            "FlowSteer train_12k MBPP row-count drift: "
            f"{mbpp_rows} != {FLOWSTEER_MBPP_TRAIN_ROWS}"
        )
    if len(unique) != FLOWSTEER_MBPP_TRAIN_UNIQUE:
        raise ValueError(
            "FlowSteer train_12k unique MBPP task-count drift: "
            f"{len(unique)} != {FLOWSTEER_MBPP_TRAIN_UNIQUE}"
        )
    return unique


def _validated_flowsteer_validation(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row in rows:
        row = _plain(dict(source_row))
        if not isinstance(row, dict):
            raise TypeError("FlowSteer validation row must remain a mapping")
        if row.get("source") != "mbpp" or row.get("problem_type") != "code":
            raise ValueError("FlowSteer eval/mbpp contains a non-MBPP code row")
        task_id, _, _ = _flowsteer_identity(row)
        if task_id in seen:
            raise ValueError(f"duplicate FlowSteer validation task_id: {task_id}")
        seen.add(task_id)
        meta = row["meta"]
        _required_text(meta.get("test"), label=f"MBPP/{task_id} test")
        _required_text(
            row.get("ground_truth"), label=f"MBPP/{task_id} reference solution"
        )
        result.append(row)
    if len(result) != FLOWSTEER_MBPP_VALIDATION_ROWS:
        raise ValueError(
            "FlowSteer eval/mbpp row-count drift: "
            f"{len(result)} != {FLOWSTEER_MBPP_VALIDATION_ROWS}"
        )
    return result


def _validated_evalplus_identities(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    task_ids: set[str] = set()
    problems: set[str] = set()
    entry_points: set[str] = set()
    row_count = 0
    for row in rows:
        row_count += 1
        source_task_id = _required_text(
            row.get("task_id"), label="EvalPlus task_id"
        ).strip()
        match = _EVALPLUS_TASK_ID.fullmatch(source_task_id)
        if match is None:
            raise ValueError(
                f"invalid canonical EvalPlus MBPP+ task_id: {source_task_id!r}"
            )
        numeric_task_id = match.group(1)
        if numeric_task_id in task_ids:
            raise ValueError(f"duplicate EvalPlus MBPP+ task_id: {source_task_id}")
        task_ids.add(numeric_task_id)
        problems.add(
            _normalized_problem(
                _required_text(row.get("prompt"), label=f"{source_task_id} prompt")
            )
        )
        entry_points.add(
            _required_text(
                row.get("entry_point"), label=f"{source_task_id} entry_point"
            ).strip()
        )
    if row_count != EVALPLUS_ROWS:
        raise ValueError(
            f"EvalPlus MBPP+ row-count drift: {row_count} != {EVALPLUS_ROWS}"
        )
    return task_ids, problems, entry_points


def _overlap_reasons(
    row: Mapping[str, Any],
    evalplus: tuple[set[str], set[str], set[str]],
) -> tuple[str, ...]:
    task_id, problem, entry_point = _flowsteer_identity(row)
    task_ids, problems, entry_points = evalplus
    reasons: list[str] = []
    if task_id in task_ids:
        reasons.append("task_id")
    if problem in problems:
        reasons.append("problem")
    if entry_point in entry_points:
        reasons.append("entry_point")
    return tuple(reasons)


def _filter_disjoint(
    rows: Iterable[dict[str, Any]],
    evalplus: tuple[set[str], set[str], set[str]],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    selected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter(
        {"task_id": 0, "problem": 0, "entry_point": 0}
    )
    excluded = 0
    for row in rows:
        reasons = _overlap_reasons(row, evalplus)
        if reasons:
            excluded += 1
            reason_counts.update(reasons)
        else:
            selected.append(row)
    return selected, reason_counts, excluded


def _public_question(problem: str, public_test_program: str) -> str:
    return (
        problem.rstrip()
        + "\n\nPublic tests:\n```python\n"
        + public_test_program.strip()
        + "\n```\n"
    )


def _training_population(split: str) -> str:
    if split == "train":
        return "flowsteer_mbpp_train"
    if split == "validation":
        return "flowsteer_mbpp_validation"
    raise ValueError(f"unsupported MBPP training split: {split}")


def _aligned_task_id(source_task_id: str, *, split: str) -> str:
    prefix = "mbpp-training" if split == "train" else "mbpp-validation"
    _training_population(split)
    return f"{prefix}:{source_task_id}"


def _public_record(
    row: Mapping[str, Any],
    *,
    split: str,
    selection_index: int,
) -> dict[str, Any]:
    source_task_id, _, entry_point = _flowsteer_identity(row)
    meta = row["meta"]
    public_test_program = str(meta["test"])
    training_population = _training_population(split)
    task_id = _aligned_task_id(source_task_id, split=split)
    question = _public_question(str(row["problem"]), public_test_program)
    if str(row["ground_truth"]).strip() in question:
        raise ValueError(
            f"public MBPP/{source_task_id} problem exposes the reference solution"
        )
    record = _compat_record(
        dataset_key=DATASET_KEY,
        source=DISPLAY_NAME,
        task_id=task_id,
        question=question,
        ground_truth=None,
        split=split,
        task_type=TASK_TYPE,
        metric=METRIC,
        context=(),
        extra={
            "source_task_id": source_task_id,
            "entry_point": entry_point,
            "training_population": training_population,
            "training_evaluator": TRAINING_EVALUATOR,
        },
        preserve_question_text=True,
    )
    metadata = dict(record["metadata"])
    metadata.update(
        {
            "protocol": PROTOCOL,
            "source_task_id": source_task_id,
            "entry_point": entry_point,
            "selection_index": selection_index,
            "training_population": training_population,
            "training_evaluator": TRAINING_EVALUATOR,
            "public_test_program": public_test_program,
            "ground_truth_role": "evaluator_only_redacted",
        }
    )
    record["metadata"] = metadata
    return record


def _private_record(
    row: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    source_task_id, _, entry_point = _flowsteer_identity(row)
    test_program = str(row["meta"]["test"])
    test_cases = [line.strip() for line in test_program.splitlines() if line.strip()]
    training_population = _training_population(split)
    task_id = _aligned_task_id(source_task_id, split=split)
    return {
        "schema_version": PRIVATE_SCHEMA_VERSION,
        "task_id": task_id,
        "source_task_id": source_task_id,
        "split": split,
        "training_population": training_population,
        "training_evaluator": TRAINING_EVALUATOR,
        "evaluator_payload": {
            "entry_point": entry_point,
            # SkillFlow training/reward.py converts ``extra.test`` to one
            # code_test_pass_rate case per non-empty source line.
            "test_cases": test_cases,
            "test": test_program,
            "reference_solution": str(row["ground_truth"]),
        },
    }


def _assert_public_redaction(records: Iterable[Mapping[str, Any]]) -> None:
    for record in records:
        if record.get("ground_truth") is not None or record.get("answer") is not None:
            raise ValueError("public MBPP training record exposes a reference answer")
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("public MBPP training record has invalid metadata")
        if "evaluator_payload" in metadata:
            raise ValueError("public MBPP training record exposes evaluator payload")


def _write_private_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _expected_policy() -> dict[str, Any]:
    return {
        "mode": "flowsteer_public_mbpp_disjoint_from_evalplus",
        "protocol": PROTOCOL,
        "selection": "preserve_source_order_after_deduplication_and_exclusion",
        "overlap_identity_fields": ["task_id", "problem", "entry_point"],
        "overlap_match": "any",
        "flowsteer_total_rows": FLOWSTEER_TOTAL_ROWS,
        "flowsteer_train_mbpp_rows": FLOWSTEER_MBPP_TRAIN_ROWS,
        "flowsteer_train_unique_tasks": FLOWSTEER_MBPP_TRAIN_UNIQUE,
        "flowsteer_validation_rows": FLOWSTEER_MBPP_VALIDATION_ROWS,
        "evalplus_full_rows": EVALPLUS_ROWS,
        "expected_train_excluded": EXPECTED_TRAIN_EXCLUDED,
        "expected_validation_excluded": EXPECTED_VALIDATION_EXCLUDED,
        "expected_train_count": EXPECTED_TRAIN_OUTPUT,
        "expected_validation_count": EXPECTED_VALIDATION_OUTPUT,
        "training_evaluator": TRAINING_EVALUATOR,
        "test_population": "external_mbpp-plus-fixed-100@1",
    }


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
        raise ValueError("MBPP training catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported MBPP training catalog schema")
    if catalog.get("task_schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("unsupported AgentGraph task schema")

    policy = catalog.get("split_policy")
    expected_policy = _expected_policy()
    if not isinstance(policy, Mapping):
        raise ValueError("MBPP training split_policy must be a mapping")
    drift = {
        key: {"expected": value, "actual": policy.get(key)}
        for key, value in expected_policy.items()
        if policy.get(key) != value
    }
    if drift:
        raise ValueError(
            "MBPP training split policy drift: "
            + json.dumps(drift, sort_keys=True)
        )

    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("MBPP training sources must be a mapping")
    flowsteer = sources.get("flowsteer_public")
    evalplus = sources.get("evalplus_full")
    fixed_test = sources.get("fixed_test_reference")
    if not isinstance(flowsteer, Mapping) or not isinstance(evalplus, Mapping):
        raise ValueError("FlowSteer and EvalPlus sources are required")
    if not isinstance(fixed_test, Mapping):
        raise ValueError("fixed-100 test reference is required")
    if fixed_test.get("protocol") != "mbpp-plus-fixed-100@1":
        raise ValueError("fixed-100 test reference protocol drift")

    flowsteer_root = _path(str(flowsteer["root"]), base=repo_root)
    train_path = _path(str(flowsteer["train_file"]), base=flowsteer_root)
    validation_path = _path(
        str(flowsteer["validation_file"]), base=flowsteer_root
    )
    evalplus_path = _path(str(evalplus["path"]), base=repo_root)
    read_rows = row_provider or _jsonl_rows

    train_candidates = _validated_flowsteer_train(read_rows(train_path))
    validation_candidates = _validated_flowsteer_validation(
        read_rows(validation_path)
    )
    evalplus_identities = _validated_evalplus_identities(
        read_rows(evalplus_path)
    )
    train, train_reasons, train_excluded = _filter_disjoint(
        train_candidates, evalplus_identities
    )
    validation, validation_reasons, validation_excluded = _filter_disjoint(
        validation_candidates, evalplus_identities
    )
    if train_excluded != EXPECTED_TRAIN_EXCLUDED or len(train) != EXPECTED_TRAIN_OUTPUT:
        raise ValueError(
            "MBPP training overlap result drift: "
            f"excluded={train_excluded}, selected={len(train)}"
        )
    if (
        validation_excluded != EXPECTED_VALIDATION_EXCLUDED
        or len(validation) != EXPECTED_VALIDATION_OUTPUT
    ):
        raise ValueError(
            "MBPP validation overlap result drift: "
            f"excluded={validation_excluded}, selected={len(validation)}"
        )

    public_train = [
        _public_record(row, split="train", selection_index=index)
        for index, row in enumerate(train)
    ]
    public_validation = [
        _public_record(row, split="validation", selection_index=index)
        for index, row in enumerate(validation)
    ]
    public = [*public_train, *public_validation]
    private = [
        *(_private_record(row, split="train") for row in train),
        *(_private_record(row, split="validation") for row in validation),
    ]
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
        "protocol": PROTOCOL,
        "training_evaluator": TRAINING_EVALUATOR,
        "training_started": False,
        "evaluation_started": False,
        "split_policy": expected_policy,
        "counts_by_split": {
            "train": len(public_train),
            "validation": len(public_validation),
            "test": 0,
        },
        "source_counts": {
            "flowsteer_total_rows": FLOWSTEER_TOTAL_ROWS,
            "flowsteer_train_mbpp_rows": FLOWSTEER_MBPP_TRAIN_ROWS,
            "flowsteer_train_unique_tasks": len(train_candidates),
            "flowsteer_validation_rows": len(validation_candidates),
            "evalplus_full_rows": EVALPLUS_ROWS,
        },
        "exclusion": {
            "match": "any",
            "identity_fields": ["task_id", "problem", "entry_point"],
            "train_excluded": train_excluded,
            "validation_excluded": validation_excluded,
            "train_reason_counts": dict(sorted(train_reasons.items())),
            "validation_reason_counts": dict(
                sorted(validation_reasons.items())
            ),
        },
        "visibility": {
            "public_reference_solution": False,
            "public_test_program": True,
            "evaluator_private_file": "evaluator_private.jsonl",
        },
        "fixed_test_reference": _plain(dict(fixed_test)),
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
        default="config/datasets_mbppplus_training_v1.yaml",
        help="MBPP training dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(Path(args.catalog))


if __name__ == "__main__":
    main()
