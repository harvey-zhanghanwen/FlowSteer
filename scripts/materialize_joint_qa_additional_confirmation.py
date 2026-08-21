#!/usr/bin/env python3
"""Freeze a fresh QA confirmation block with the existing data converters.

This is a thin data-partition adapter over ``prepare_agentgraph_datasets`` and
``prepare_joint_qa_partitions``.  It does not download data, run a model, or
change the existing train/development/test files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml

try:
    from scripts.prepare_agentgraph_datasets import CONVERTERS, TASK_SCHEMA_VERSION
    from scripts.prepare_joint_qa_partitions import (
        _partition_record,
        _take_candidate_prefix,
    )
except ModuleNotFoundError:  # Direct script execution.
    from prepare_agentgraph_datasets import (  # type: ignore[no-redef]
        CONVERTERS,
        TASK_SCHEMA_VERSION,
    )
    from prepare_joint_qa_partitions import (  # type: ignore[no-redef]
        _partition_record,
        _take_candidate_prefix,
    )


SCHEMA_VERSION = "flowsteer.joint-qa.additional-confirmation.v1"
SUPPORTED_DATASETS = frozenset({"hotpotqa", "triviaqa"})


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _existing_ids(manifest: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    datasets = manifest.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("base manifest has no datasets mapping")
    for dataset in datasets.values():
        if not isinstance(dataset, Mapping):
            raise ValueError("base manifest dataset must be a mapping")
        partitions = dataset.get("ordered_task_ids")
        if not isinstance(partitions, Mapping):
            raise ValueError("base manifest dataset has no ordered task IDs")
        for ids in partitions.values():
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
                raise ValueError("base manifest task IDs must be string arrays")
            result.update(ids)
    return result


def materialize(config_path: Path) -> Path:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported additional-confirmation schema")
    raw_datasets = config.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("additional confirmation datasets must be a non-empty list")
    if not all(isinstance(dataset, str) for dataset in raw_datasets):
        raise ValueError("additional confirmation dataset keys must be strings")
    if len(raw_datasets) != len(set(raw_datasets)):
        raise ValueError("additional confirmation datasets must be unique")
    unsupported = set(raw_datasets) - SUPPORTED_DATASETS
    if unsupported:
        raise ValueError(
            "unsupported additional-confirmation datasets: "
            + ", ".join(sorted(unsupported))
        )
    dataset_keys = tuple(raw_datasets)
    if config.get("task_split") != "validation":
        raise ValueError("additional confirmation must remain validation-only")
    partition = str(config.get("partition", ""))
    if not partition:
        raise ValueError("additional confirmation partition is empty")

    raw_range = config.get("candidate_range")
    if not isinstance(raw_range, Mapping):
        raise ValueError("candidate_range must be a mapping")
    start, stop = raw_range.get("start"), raw_range.get("stop")
    if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
        raise ValueError("candidate_range must be a positive half-open interval")

    catalog_path = _resolve(root, config["dataset_catalog_path"])
    catalog = _load_yaml(catalog_path)
    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("dataset catalog has no sources mapping")
    base_manifest_path = _resolve(root, config["base_manifest_path"])
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(base_manifest, Mapping):
        raise ValueError("base manifest must be an object")
    reserved_ids = _existing_ids(base_manifest)

    output_path = _resolve(root, config["output_path"])
    combined_path = _resolve(root, config["combined_output_path"])
    manifest_path = _resolve(root, config["manifest_path"])
    for target in (output_path, combined_path, manifest_path):
        if target.exists():
            raise FileExistsError(f"write-once confirmation output exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    ordered_ids: dict[str, list[str]] = {}
    for dataset_key in dataset_keys:
        source = sources.get(dataset_key)
        converter = CONVERTERS.get(dataset_key)
        if not isinstance(source, Mapping) or converter is None:
            raise ValueError(f"missing converter for {dataset_key}")
        candidates = _take_candidate_prefix(converter(source), stop)
        block = candidates[start:stop]
        ids = [str(record["task_id"]) for record in block]
        if len(ids) != len(set(ids)) or reserved_ids.intersection(ids):
            raise ValueError(f"{dataset_key} additional confirmation overlaps existing data")
        ordered_ids[dataset_key] = ids
        reserved_ids.update(ids)
        records.extend(
            _partition_record(
                record,
                partition=partition,
                task_split="validation",
                partition_index=index,
                native_candidate_position=native_position,
            )
            for index, (native_position, record) in enumerate(
                zip(range(start, stop), block)
            )
        )

    base_confirmation_path = _resolve(root, config["base_confirmation_path"])
    base_lines = [
        line for line in base_confirmation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base_ids = {str(json.loads(line)["task_id"]) for line in base_lines}
    new_ids = {str(record["task_id"]) for record in records}
    if base_ids.intersection(new_ids):
        raise ValueError("additional confirmation overlaps base confirmation")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "source_catalog": str(catalog_path),
        "base_manifest": str(base_manifest_path),
        "partition": partition,
        "task_split": "validation",
        "candidate_range": {"start": start, "stop": stop},
        "count_per_dataset": stop - start,
        "ordered_task_ids": ordered_ids,
        "excluded_from_grpo_and_reported_metrics": True,
        "final_test_untouched": True,
    }
    temp_dir = Path(tempfile.mkdtemp(prefix="joint-qa-confirmation-", dir=output_path.parent))
    try:
        new_temp = temp_dir / output_path.name
        combined_temp = temp_dir / combined_path.name
        manifest_temp = temp_dir / manifest_path.name
        with new_temp.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        with combined_temp.open("x", encoding="utf-8") as handle:
            for line in base_lines:
                handle.write(line + "\n")
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        new_temp.replace(output_path)
        combined_temp.replace(combined_path)
        manifest_temp.replace(manifest_path)
        temp_dir.rmdir()
    except BaseException:
        raise
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/joint_qa_round4_confirmation.yaml",
    )
    args = parser.parse_args()
    print(materialize(Path(args.config)), flush=True)


if __name__ == "__main__":
    main()
