#!/usr/bin/env python3
"""Build the frozen train-only HotpotQA QA-memory embedding index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_qa_memory_index import (
    build_hotpotqa_qa_memory_index,
    load_paraphrase_materialization,
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _validation_ids(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                task_id = value["task_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid task identity") from exc
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path}:{line_number}: empty task identity")
            result.append(task_id.strip())
    return tuple(result)


def build_from_config(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    retrieval = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    if retrieval.get("corpus_kind") != "train_qa_memory":
        raise ValueError("config does not select train_qa_memory")
    validation_ids = _validation_ids(
        _resolve(root, retrieval["frozen_validation_tasks"])
    )
    paraphrases = load_paraphrase_materialization(
        _resolve(root, retrieval["paraphrase_materialization_path"])
    )
    manifest = build_hotpotqa_qa_memory_index(
        index_dir=_resolve(root, retrieval["index_dir"]),
        train_jsonl=_resolve(root, retrieval["train_tasks"]),
        validation_task_ids=validation_ids,
        paraphrases=paraphrases,
        embedding_model_path=str(retrieval["embedding_model"]),
        embedding_model_id=str(retrieval["embedding_model_id"]),
        embedding_device=str(retrieval["embedding_device"]),
        frozen_top_k=int(retrieval["search_top_k"]),
        index_version=int(retrieval.get("index_version", 1)),
        expected_train_count=int(retrieval["train_sample_count"]),
        expected_validation_count=int(retrieval["validation_sample_count"]),
    )
    return manifest.to_value()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_qa_memory_v1.yaml",
    )
    args = parser.parse_args(argv)
    try:
        value = build_from_config(Path(args.config))
    except Exception as exc:
        print(f"HotpotQA QA-memory index build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
