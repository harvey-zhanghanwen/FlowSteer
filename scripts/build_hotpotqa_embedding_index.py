#!/usr/bin/env python3
"""Build the frozen answer-free HotpotQA public-context embedding index."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_embedding_index import build_hotpotqa_embedding_index
from src.interactive.task_dataset import iter_task_records


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def build_from_config(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    section = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    data = _mapping(config.get("data"), "data")
    validation_path = _resolve(root, section["frozen_validation_tasks"])
    development_path = _resolve(root, section["development_tasks"])
    validation = tuple(iter_task_records(validation_path, expected_split="validation"))
    development_count = int(section["development_sample_count"])
    development = tuple(
        task
        for task in iter_task_records(development_path, expected_split="train")
        if str(task.metadata.get("dataset", "")).casefold() == "hotpotqa"
        or task.task_id.startswith("hotpotqa:")
    )[:development_count]
    if len(validation) != 128 or len(development) != development_count:
        raise ValueError("frozen validation/development task counts differ from config")
    validation_ids = {task.task_id for task in validation}
    if validation_ids & {task.task_id for task in development}:
        raise ValueError("development and validation task IDs overlap")
    task_splits = {
        **{task.task_id: "validation" for task in validation},
        **{task.task_id: "train" for task in development},
    }
    source_files = tuple(
        Path(value).expanduser().resolve()
        for value in sorted(glob.glob(str(section["source_parquet_glob"])))
    )
    if not source_files:
        raise FileNotFoundError("HotpotQA source parquet glob matched no files")
    manifest = build_hotpotqa_embedding_index(
        index_dir=_resolve(root, section["index_dir"]),
        parquet_paths=source_files,
        task_splits=task_splits,
        embedding_model_path=str(section["embedding_model"]),
        embedding_model_id=str(section["embedding_model_id"]),
        embedding_device=str(section["embedding_device"]),
        frozen_top_k=int(section["search_top_k"]),
    )
    return manifest.to_value()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_embedding_retrieval_v2.yaml",
    )
    args = parser.parse_args(argv)
    value = build_from_config(Path(args.config))
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
