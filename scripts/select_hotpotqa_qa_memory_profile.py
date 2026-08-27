#!/usr/bin/env python3
"""Freeze HotpotQA QA-memory top-k on train development records only.

The selector evaluates whether an original *training* question retrieves a
semantics-preserving paraphrase with the same ``base_task_id``.  Held-out
validation contributes task identities only to the split-isolation check; its
questions, answers, aliases, supporting facts, and evaluator data are never
loaded by this module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.config_loader import load_yaml
from src.interactive.hotpotqa_embedding_index import (
    _encode,
    _load_sentence_transformer,
)
from src.interactive.hotpotqa_qa_memory_index import (
    load_hotpotqa_train_qa_sources,
    load_paraphrase_materialization,
    materialize_hotpotqa_qa_memories,
)


CANDIDATE_TOP_K = (2, 3, 4, 5)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _task_ids(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _mapping(json.loads(line), "task identity")
                task_id = value["task_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid task identity") from exc
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path}:{line_number}: empty task identity")
            result.append(task_id.strip())
    return tuple(result)


def _rank_base_task_ids(
    query_vectors: np.ndarray,
    memory_vectors: np.ndarray,
    *,
    memory_ids: Sequence[str],
    memory_base_task_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    if query_vectors.ndim != 2 or memory_vectors.ndim != 2:
        raise ValueError("profile embeddings must be matrices")
    if query_vectors.shape[1] != memory_vectors.shape[1]:
        raise ValueError("query and memory embedding dimensions differ")
    if len(memory_ids) != len(memory_base_task_ids) or len(memory_ids) != len(
        memory_vectors
    ):
        raise ValueError("QA-memory ranking metadata differs from embedding count")
    rankings: list[tuple[str, ...]] = []
    for query_vector in query_vectors:
        scored = [
            (
                float(np.dot(memory_vectors[index], query_vector)),
                str(memory_ids[index]),
                str(memory_base_task_ids[index]),
            )
            for index in range(len(memory_vectors))
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        rankings.append(tuple(item[2] for item in scored))
    return tuple(rankings)


def select_profile(config_path: Path) -> Mapping[str, object]:
    config_path = config_path.expanduser().resolve()
    root = config_path.parent.parent
    config = load_yaml(config_path)
    section = _mapping(config.get("qa_embedding_retrieval"), "qa_embedding_retrieval")
    if section.get("corpus_kind") != "train_qa_memory":
        raise ValueError("config does not select the train_qa_memory corpus")

    expected_train_count = int(section.get("train_sample_count", 512))
    expected_validation_count = int(section.get("validation_sample_count", 128))
    if expected_train_count != 512 or expected_validation_count != 128:
        raise ValueError("HotpotQA QA-memory profile requires the frozen 512/128 split")
    validation_ids = _task_ids(
        _resolve(root, section["frozen_validation_tasks"])
    )
    sources = load_hotpotqa_train_qa_sources(
        _resolve(root, section["train_tasks"]),
        validation_task_ids=validation_ids,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    memories = materialize_hotpotqa_qa_memories(
        sources,
        load_paraphrase_materialization(
            _resolve(root, section["paraphrase_materialization_path"])
        ),
    )

    development_count = int(section.get("development_sample_count", 32))
    if development_count < 1:
        raise ValueError("development_sample_count must be positive")
    # Architecture development is a frozen sequential subset of unique train
    # records.  Cycle copies remain in the retrieval corpus but cannot inflate
    # the number of development queries.
    unique_train = tuple(source for source in sources if not source.cycled)
    development = unique_train[:development_count]
    if len(development) != development_count:
        raise ValueError("insufficient unique train records for architecture development")

    model = _load_sentence_transformer(
        str(section["embedding_model"]),
        str(section.get("embedding_device", "cpu")),
    )
    memory_vectors = _encode(model, [memory.document_text for memory in memories])
    query_vectors = _encode(
        model,
        [source.question for source in development],
        batch_size=min(32, development_count),
    )
    rankings = _rank_base_task_ids(
        query_vectors,
        memory_vectors,
        memory_ids=[memory.memory_id for memory in memories],
        memory_base_task_ids=[memory.base_task_id for memory in memories],
    )

    first_relevant_ranks: list[int] = []
    for source, ranking in zip(development, rankings, strict=True):
        try:
            first_relevant_ranks.append(ranking.index(source.base_task_id) + 1)
        except ValueError as exc:
            raise ValueError(
                f"QA-memory has no record for development base_task_id {source.base_task_id}"
            ) from exc

    hit_rate: dict[int, float] = {}
    reciprocal_rank: dict[int, float] = {}
    for candidate in CANDIDATE_TOP_K:
        hit_rate[candidate] = sum(
            rank <= candidate for rank in first_relevant_ranks
        ) / len(first_relevant_ranks)
        reciprocal_rank[candidate] = sum(
            (1.0 / rank) if rank <= candidate else 0.0
            for rank in first_relevant_ranks
        ) / len(first_relevant_ranks)
    selected = min(
        CANDIDATE_TOP_K,
        key=lambda candidate: (
            -hit_rate[candidate],
            -reciprocal_rank[candidate],
            candidate,
        ),
    )
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for source, query_vector in zip(development, query_vectors, strict=True):
        scored = [
            (
                float(np.dot(memory_vectors[index], query_vector)),
                memories[index].base_task_id,
            )
            for index in range(len(memories))
        ]
        positive_scores.append(
            max(score for score, base_task_id in scored if base_task_id == source.base_task_id)
        )
        negative_scores.append(
            max(score for score, base_task_id in scored if base_task_id != source.base_task_id)
        )
    # DIRECT_REUSE + NECESSARY ADAPTATION: the QA-memory read gate follows
    # unified QA's evidence-grounding rule that embedding rank alone is not
    # evidence.  Freeze the smallest threshold that rejects every non-source
    # neighbour on the train-only architecture-development subset; validation
    # content is never consulted.
    selected_min_similarity = float(
        np.nextafter(max(negative_scores), np.float64(np.inf))
    )
    positive_recall = sum(
        score >= selected_min_similarity for score in positive_scores
    ) / len(positive_scores)
    return {
        "schema_version": "flowsteer.hotpotqa.qa_memory_profile_selection.v1",
        "selection_split": "train/architecture-development",
        "selection_source": "frozen_sequential_unique_train_records",
        "validation_identity_used_for_split_check_only": True,
        "validation_question_consulted": False,
        "validation_answer_or_alias_consulted": False,
        "validation_supporting_facts_consulted": False,
        "development_sample_count": len(development),
        "development_task_ids": [source.source_train_task_id for source in development],
        "development_base_task_ids": [source.base_task_id for source in development],
        "train_record_count": len(memories),
        "unique_source_count": len({memory.base_task_id for memory in memories}),
        "cycled_record_count": sum(memory.cycled for memory in memories),
        "paraphrase_count": len(memories),
        "embedding_model": str(section["embedding_model_id"]),
        "normalize_embeddings": True,
        "similarity": "cosine",
        "candidate_top_k": list(CANDIDATE_TOP_K),
        "base_task_hit_rate_by_top_k": {
            str(candidate): hit_rate[candidate] for candidate in CANDIDATE_TOP_K
        },
        "mean_reciprocal_rank_by_top_k": {
            str(candidate): reciprocal_rank[candidate]
            for candidate in CANDIDATE_TOP_K
        },
        "selected_top_k": selected,
        "selected_min_similarity": selected_min_similarity,
        "similarity_threshold_selection": {
            "selection_rule": (
                "smallest threshold strictly above the maximum non-source "
                "similarity on the frozen train architecture-development subset"
            ),
            "negative_rejection": 1.0,
            "positive_recall": positive_recall,
            "positive_score_min": min(positive_scores),
            "positive_score_mean": sum(positive_scores) / len(positive_scores),
            "negative_score_max": max(negative_scores),
            "negative_score_mean": sum(negative_scores) / len(negative_scores),
        },
        "selection_rule": (
            "maximize train-development base_task_hit@k, then truncated MRR@k, "
            "then choose the smallest top-k"
        ),
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/evaluation_hotpotqa_qa_memory_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hotpotqa_qa_memory_v1/profile_selection.json",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    value = select_profile(config_path)
    output = _resolve(config_path.parent.parent, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
