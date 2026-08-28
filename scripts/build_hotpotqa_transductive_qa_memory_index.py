#!/usr/bin/env python3
"""Build the isolated HotpotQA 512+128 transductive QA-memory index."""

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
from src.interactive.hotpotqa_qa_memory_index import load_paraphrase_materialization
from src.interactive.hotpotqa_transductive_qa_memory_index import (
    TRANSDUCTIVE_EVALUATION_REGIME,
    build_hotpotqa_transductive_qa_memory_index,
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def build_from_config(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    retrieval = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    if retrieval.get("corpus_kind") != "transductive_qa_memory":
        raise ValueError("config does not select transductive_qa_memory")
    if retrieval.get("contains_evaluation_answers") is not True:
        raise ValueError("transductive config must declare evaluation answers")
    if retrieval.get("evaluation_regime") != TRANSDUCTIVE_EVALUATION_REGIME:
        raise ValueError("transductive evaluation regime differs")
    if retrieval.get("official_heldout_eligible") is not False:
        raise ValueError("transductive config cannot be held-out eligible")

    paraphrases = load_paraphrase_materialization(
        _resolve(root, retrieval["paraphrase_materialization_path"])
    )
    manifest = build_hotpotqa_transductive_qa_memory_index(
        index_dir=_resolve(root, retrieval["index_dir"]),
        train_jsonl=_resolve(root, retrieval["train_tasks"]),
        evaluation_jsonl=_resolve(root, retrieval["frozen_validation_tasks"]),
        paraphrases=paraphrases,
        embedding_model_path=str(retrieval["embedding_model"]),
        embedding_model_id=str(retrieval["embedding_model_id"]),
        embedding_device=str(retrieval["embedding_device"]),
        frozen_top_k=int(retrieval["search_top_k"]),
        expected_train_count=int(retrieval["train_sample_count"]),
        expected_evaluation_count=int(retrieval["validation_sample_count"]),
    )
    return manifest.to_value()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/hotpotqa_transductive_qa_memory_v1.yaml",
    )
    args = parser.parse_args(argv)
    try:
        value = build_from_config(Path(args.config))
    except Exception as exc:
        print(f"HotpotQA transductive QA-memory build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
