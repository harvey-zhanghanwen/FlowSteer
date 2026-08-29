#!/usr/bin/env python3
"""Build the all-native HotpotQA declarative-fact embedding index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml  # noqa: E402
from src.interactive.hotpotqa_full_dataset_fact_memory_index import (  # noqa: E402
    FULL_DATASET_EVALUATION_SCOPE,
    build_hotpotqa_full_dataset_fact_memory_index,
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _task_ids(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
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
    if len(set(result)) != len(result):
        raise ValueError("frozen evaluation task IDs are not unique")
    return tuple(result)


def _fact_sidecar(path: Path) -> tuple[Mapping[str, object], ...]:
    values: list[Mapping[str, object]] = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            values.append(value)
    return tuple(values)


def build_from_config(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    retrieval = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    if retrieval.get("corpus_kind") != "full_dataset_fact_memory":
        raise ValueError("config does not select full_dataset_fact_memory")
    if retrieval.get("contains_evaluation_source_facts") is not True:
        raise ValueError("full-dataset config must declare evaluation source facts")
    if retrieval.get("contains_raw_questions") is not False:
        raise ValueError("fact index must exclude raw questions")
    if retrieval.get("contains_raw_answers") is not False:
        raise ValueError("fact index must exclude raw answers")
    if retrieval.get("indexed_text_field") != "fact_text":
        raise ValueError("fact index must vectorize only fact_text")
    if retrieval.get("evaluation_scope") != FULL_DATASET_EVALUATION_SCOPE:
        raise ValueError("full-dataset evaluation scope differs")
    if retrieval.get("official_heldout_eligible") is not False:
        raise ValueError("full-dataset retrieval cannot be held-out eligible")

    paraphrases = _fact_sidecar(
        _resolve(root, retrieval["paraphrase_materialization_path"])
    )
    manifest = build_hotpotqa_full_dataset_fact_memory_index(
        index_dir=_resolve(root, retrieval["index_dir"]),
        dataset_catalog_path=_resolve(root, retrieval["dataset_catalog"]),
        frozen_evaluation_task_ids=_task_ids(
            _resolve(root, retrieval["frozen_validation_tasks"])
        ),
        paraphrases=paraphrases,
        embedding_model_path=str(retrieval["embedding_model"]),
        embedding_model_id=str(retrieval["embedding_model_id"]),
        embedding_device=str(retrieval["embedding_device"]),
        frozen_top_k=int(retrieval["search_top_k"]),
        expected_train_count=int(retrieval["native_train_count"]),
        expected_validation_count=int(retrieval["native_validation_count"]),
    )
    return manifest.to_value()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        value = build_from_config(Path(args.config))
    except Exception as exc:
        print(
            f"HotpotQA full-dataset fact-memory index build failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
