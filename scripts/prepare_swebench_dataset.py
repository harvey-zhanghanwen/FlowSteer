#!/usr/bin/env python3
"""Prepare split-isolated SWE-bench records for AgentGraph.

The shared ``TaskRecord`` shape and atomic ``SplitWriters`` are reused from
``prepare_agentgraph_datasets.py``.  This source-specific adapter only assigns
the benchmark populations that the shared held-out-first recipe cannot express:

* first 512 regular SWE-bench train rows -> project train;
* first 128 regular SWE-bench dev rows -> project validation; and
* the complete SWE-bench Verified release -> project test.

The gold patch and test payload remain evaluator-only fields.  They are never
rendered into the task question or public ``extra`` contract.  This command
does not start Docker, a model/API service, or training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHARED_PREPARER_PATH = (
    Path(__file__).resolve().parent / "prepare_agentgraph_datasets.py"
)


def _load_shared_preparer() -> Any:
    module_name = "_flowsteer_prepare_agentgraph_datasets_for_swebench"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, _SHARED_PREPARER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load shared dataset preparer: {_SHARED_PREPARER_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SHARED = _load_shared_preparer()
TASK_SCHEMA_VERSION = _SHARED.TASK_SCHEMA_VERSION
SplitWriters = _SHARED.SplitWriters
_compat_record = _SHARED._compat_record
_iter_parquet_rows = _SHARED._iter_parquet_rows
_path = _SHARED._path
_plain = _SHARED._plain


CATALOG_SCHEMA_VERSION = "flowsteer.agentgraph.swebench.dataset.v1"
TRAIN_COUNT = 512
VALIDATION_COUNT = 128
DATASET_KEY = "swe_bench"
DISPLAY_NAME = "SWE-bench"

RowProvider = Callable[[Path], Iterable[Mapping[str, Any]]]


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SWE-bench row has invalid {field}")
    return value.strip()


def _record(
    row: Mapping[str, Any],
    *,
    split: str,
    dataset_source: str,
    selection_index: int,
) -> dict[str, Any]:
    instance_id = _required_text(row, "instance_id")
    repo = _required_text(row, "repo")
    base_commit = _required_text(row, "base_commit")
    problem_statement = _required_text(row, "problem_statement")
    patch = row.get("patch")
    if not isinstance(patch, str):
        raise ValueError(f"SWE-bench {instance_id} has no evaluator gold patch")

    evaluator_payload = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "test_patch": row.get("test_patch", ""),
        "FAIL_TO_PASS": _plain(row.get("FAIL_TO_PASS", [])),
        "PASS_TO_PASS": _plain(row.get("PASS_TO_PASS", [])),
        "version": row.get("version", ""),
        "environment_setup_commit": row.get("environment_setup_commit", ""),
        "dataset_source": dataset_source,
    }
    extra = {
        "repo": repo,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "dataset_source": dataset_source,
        "selection_index": selection_index,
    }
    result = _compat_record(
        dataset_key=DATASET_KEY,
        source=DISPLAY_NAME,
        task_id=f"swe-bench:{instance_id}",
        question=f"Fix the following software issue:\n\n{problem_statement}",
        ground_truth=patch,
        split=split,
        task_type="code_generation",
        metric="resolved_rate",
        extra=extra,
        evaluator_payload=evaluator_payload,
        code_files={},
    )
    metadata = dict(result["metadata"])
    metadata.update(
        {
            "benchmark_slice": dataset_source,
            "dataset_source": dataset_source,
            "native_split": {
                "regular_train": "train",
                "regular_dev": "dev",
                "verified": "verified",
            }[dataset_source],
            "evaluation_role": {
                "regular_train": "training",
                "regular_dev": "development",
                "verified": "final-evaluation",
            }[dataset_source],
        }
    )
    result["metadata"] = metadata
    return result


def _take_exact(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    split: str,
    dataset_source: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index >= count:
            break
        records.append(
            _record(
                row,
                split=split,
                dataset_source=dataset_source,
                selection_index=index,
            )
        )
    if len(records) != count:
        raise ValueError(
            f"{dataset_source} provides {len(records)} rows; expected {count}"
        )
    return records


def _all_verified(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        _record(
            row,
            split="test",
            dataset_source="verified",
            selection_index=index,
        )
        for index, row in enumerate(rows)
    ]
    if not records:
        raise ValueError("SWE-bench Verified population is empty")
    return records


def _instance_ids(records: Sequence[Mapping[str, Any]], group: str) -> set[str]:
    values = {
        str(
            record.get("metadata", {})
            .get("evaluator_payload", {})
            .get("instance_id", "")
        )
        for record in records
    }
    if "" in values or len(values) != len(records):
        raise ValueError(f"SWE-bench {group} contains missing or duplicate instance_id")
    return values


def _assert_split_isolation(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> None:
    groups = {
        "regular_train": _instance_ids(train, "regular_train"),
        "regular_dev": _instance_ids(validation, "regular_dev"),
        "verified": _instance_ids(test, "verified"),
    }
    for left, right in (
        ("regular_train", "regular_dev"),
        ("regular_train", "verified"),
        ("regular_dev", "verified"),
    ):
        overlap = groups[left] & groups[right]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                f"SWE-bench instance_id overlap between {left} and {right}: {example}"
            )


def _verified_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to load SWE-bench Verified") from exc
    return load_from_disk(str(path))


def prepare(
    catalog_path: Path,
    *,
    train_provider: RowProvider | None = None,
    development_provider: RowProvider | None = None,
    verified_provider: RowProvider | None = None,
) -> Path:
    catalog_path = catalog_path.expanduser().resolve()
    repo_root = catalog_path.parent.parent
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, Mapping):
        raise ValueError("SWE-bench catalog must be a mapping")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported SWE-bench dataset catalog schema")
    if catalog.get("task_schema_version") != TASK_SCHEMA_VERSION:
        raise ValueError("unsupported AgentGraph task schema")

    policy = catalog.get("split_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("SWE-bench split_policy must be a mapping")
    expected_policy = {
        "selection": "sequential",
        "train_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT,
        "test_population": "complete_verified",
        "train_source": "regular_train",
        "validation_source": "regular_dev",
        "test_source": "verified",
        "require_pairwise_disjoint_instance_ids": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": policy.get(key)}
        for key, expected in expected_policy.items()
        if policy.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "SWE-bench split policy drift: " + json.dumps(mismatches, sort_keys=True)
        )

    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("SWE-bench sources must be a mapping")
    regular = sources.get("regular")
    verified = sources.get("verified")
    if not isinstance(regular, Mapping) or not isinstance(verified, Mapping):
        raise ValueError("SWE-bench regular and verified sources are required")
    regular_root = _path(str(regular["path"]), base=repo_root)
    train_path = regular_root / str(regular["train_file"])
    development_path = regular_root / str(regular["development_file"])
    verified_path = _path(str(verified["path"]), base=repo_root)

    read_train = train_provider or _iter_parquet_rows
    read_development = development_provider or _iter_parquet_rows
    read_verified = verified_provider or _verified_rows
    train = _take_exact(
        read_train(train_path),
        count=TRAIN_COUNT,
        split="train",
        dataset_source="regular_train",
    )
    validation = _take_exact(
        read_development(development_path),
        count=VALIDATION_COUNT,
        split="validation",
        dataset_source="regular_dev",
    )
    test = _all_verified(read_verified(verified_path))
    _assert_split_isolation(train, validation, test)

    output_dir = _path(str(catalog["aligned_dir"]), base=repo_root)
    writers = SplitWriters(output_dir)
    for record in (*train, *validation, *test):
        writers.write(record)
    manifest = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path),
        "training_started": False,
        "split_policy": dict(expected_policy),
        "counts_by_split": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "instance_id_isolation": {
            "status": "pairwise_disjoint",
            "groups": {
                "regular_train": len(train),
                "regular_dev": len(validation),
                "verified": len(test),
            },
        },
        "files": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
    }
    writers.publish(manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="config/datasets_swebench.yaml",
        help="SWE-bench dataset catalog path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(Path(args.catalog))


if __name__ == "__main__":
    main()
