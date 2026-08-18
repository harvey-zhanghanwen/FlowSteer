#!/usr/bin/env python3
"""Freeze non-overlapping HotpotQA/TriviaQA experiment partitions.

The source converters and record schema are reused from
``prepare_agentgraph_datasets.py``.  This script only assigns a second,
experiment-specific partition over each converter's canonical candidate
sequence; it does not download data or start model services.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import yaml

try:
    from scripts.prepare_agentgraph_datasets import (
        CONVERTERS,
        TASK_SCHEMA_VERSION,
        _retag_record,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from prepare_agentgraph_datasets import (  # type: ignore[no-redef]
        CONVERTERS,
        TASK_SCHEMA_VERSION,
        _retag_record,
    )


PARTITION_SCHEMA_VERSION = "flowsteer.joint-qa.partitions.v2"
WRITTEN_PARTITIONS = (
    "development",
    "train",
    "skill_confirmation",
    "test",
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _resolve(repo_root: Path, value: object) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _partition_record(
    record: Mapping[str, Any],
    *,
    partition: str,
    task_split: str,
    partition_index: int,
    native_candidate_position: int,
) -> dict[str, Any]:
    result = _retag_record(
        record,
        split=task_split,
        selection_index=partition_index,
    )
    metadata = dict(result.get("metadata", {}))
    metadata["joint_qa_partition"] = partition
    metadata["native_candidate_position"] = native_candidate_position
    result["metadata"] = metadata
    extra = dict(result.get("extra", {}))
    extra["joint_qa_partition"] = partition
    extra["native_candidate_position"] = native_candidate_position
    result["extra"] = extra
    return result


def _take_candidate_prefix(
    records: Iterable[Mapping[str, Any]], stop: int
) -> tuple[Mapping[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    iterator = iter(records)
    while len(selected) < stop:
        try:
            selected.append(next(iterator))
        except StopIteration as exc:
            raise ValueError(
                f"canonical candidate stream ended at {len(selected)} before {stop}"
            ) from exc
    ids = [str(item["task_id"]) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical candidate prefix contains duplicate task IDs")
    return tuple(selected)


def prepare(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config = _load_yaml(config_path)
    if config.get("schema_version") != PARTITION_SCHEMA_VERSION:
        raise ValueError("unsupported joint-QA partition schema")

    catalog_path = _resolve(repo_root, config["dataset_catalog_path"])
    catalog = _load_yaml(catalog_path)
    sources = catalog.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("dataset catalog has no sources mapping")

    raw_partitions = config.get("partitions")
    if not isinstance(raw_partitions, Mapping):
        raise ValueError("joint-QA config has no partitions mapping")
    expected_names = (*WRITTEN_PARTITIONS[:2], "quarantine", *WRITTEN_PARTITIONS[2:])
    if tuple(raw_partitions) != expected_names:
        raise ValueError("joint-QA partitions must preserve the declared canonical order")

    ranges: dict[str, tuple[int, int, str | None]] = {}
    previous_stop = 0
    for name, raw in raw_partitions.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"partition {name} must be a mapping")
        start, stop = raw.get("start"), raw.get("stop")
        task_split = raw.get("task_split")
        if type(start) is not int or type(stop) is not int or start != previous_stop or stop <= start:
            raise ValueError("joint-QA partition ranges must be positive and contiguous")
        if name == "quarantine":
            if task_split is not None:
                raise ValueError("quarantine cannot be exposed as a TaskRecord split")
        elif task_split not in {"train", "validation", "test"}:
            raise ValueError(f"partition {name} has an invalid task_split")
        ranges[str(name)] = (start, stop, task_split)
        previous_stop = stop

    output_dir = _resolve(repo_root, config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"joint-QA output already exists and is non-empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix="joint-qa-partitions-", dir=str(output_dir.parent))
    )
    handles = {
        name: (temporary_dir / f"{name}.jsonl").open("x", encoding="utf-8")
        for name in WRITTEN_PARTITIONS
    }

    manifest_datasets: dict[str, Any] = {}
    all_ids_by_partition: dict[str, set[str]] = {
        name: set() for name in raw_partitions
    }
    try:
        dataset_keys = config.get("datasets")
        if dataset_keys != ["hotpotqa", "triviaqa"]:
            raise ValueError("joint-QA partitions require HotpotQA then TriviaQA")
        for dataset_key in dataset_keys:
            source_config = sources.get(dataset_key)
            converter = CONVERTERS.get(dataset_key)
            if not isinstance(source_config, Mapping) or converter is None:
                raise ValueError(f"missing converter configuration for {dataset_key}")
            candidates = _take_candidate_prefix(
                converter(source_config),
                stop=previous_stop,
            )
            ordered_ids: dict[str, list[str]] = {}
            for partition, (start, stop, task_split) in ranges.items():
                block = candidates[start:stop]
                ids = [str(record["task_id"]) for record in block]
                ordered_ids[partition] = ids
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate {dataset_key} IDs in {partition}")
                if all_ids_by_partition[partition] & set(ids):
                    raise ValueError(f"cross-dataset task ID collision in {partition}")
                all_ids_by_partition[partition].update(ids)
                if task_split is None:
                    continue
                handle = handles[partition]
                for partition_index, (native_position, record) in enumerate(
                    zip(range(start, stop), block)
                ):
                    value = _partition_record(
                        record,
                        partition=partition,
                        task_split=task_split,
                        partition_index=partition_index,
                        native_candidate_position=native_position,
                    )
                    handle.write(
                        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
            manifest_datasets[dataset_key] = {
                "counts": {name: len(ids) for name, ids in ordered_ids.items()},
                "ordered_task_ids": ordered_ids,
            }

        names = tuple(all_ids_by_partition)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = all_ids_by_partition[left] & all_ids_by_partition[right]
                if overlap:
                    raise ValueError(f"joint-QA partitions overlap: {left}/{right}")
        for handle in handles.values():
            handle.close()
        manifest = {
            "schema_version": PARTITION_SCHEMA_VERSION,
            "task_schema_version": TASK_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_catalog": str(catalog_path),
            "config": str(config_path),
            "partitions": {
                name: {
                    "start": start,
                    "stop": stop,
                    "count_per_dataset": stop - start,
                    "task_split": task_split,
                    "file": f"{name}.jsonl" if task_split is not None else None,
                }
                for name, (start, stop, task_split) in ranges.items()
            },
            "datasets": manifest_datasets,
        }
        with (temporary_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(temporary_dir.iterdir()):
            path.replace(output_dir / path.name)
        temporary_dir.rmdir()
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/joint_qa_partitions_v2.yaml",
    )
    return parser.parse_args()


def main() -> None:
    output_dir = prepare(Path(parse_args().config))
    print(f"published joint-QA partitions to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
