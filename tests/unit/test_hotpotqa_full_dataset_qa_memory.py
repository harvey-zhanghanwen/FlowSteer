from __future__ import annotations

import asyncio
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import materialize_hotpotqa_full_dataset_qa_memory as materializer
from scripts.materialize_hotpotqa_full_dataset_qa_memory import (
    DATASET_PAIR_FALLBACK_PROVENANCE,
    _dataset_pair_fallback,
)
from src.interactive import hotpotqa_full_dataset_qa_memory_index as full_index
from src.interactive.hotpotqa_full_dataset_qa_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    FULL_DATASET_QA_MEMORY_CORPUS_VERSION,
    FULL_DATASET_QA_MEMORY_SCHEMA_VERSION,
    HotpotQAFullDatasetQAMemoryIndex,
    HotpotQAFullDatasetQAMemoryIndexManifest,
    HotpotQAFullDatasetQASources,
    load_hotpotqa_full_dataset_qa_sources,
)
from src.interactive.hotpotqa_qa_memory_index import (
    QA_MEMORY_DOCUMENT_TEMPLATE,
    HotpotQATrainQASource,
    materialize_hotpotqa_qa_memories,
)


def _record(
    task_id: str,
    *,
    split: str,
    question: str,
    answer: str,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "split": split,
        "question": (
            "Based on the following passages, answer the question.\n\n"
            "[[private context that must not survive]]\n\n"
            f"Question: {question}"
        ),
        "ground_truth": answer,
        "context": ["private context that must not survive"],
        "metadata": {
            "dataset_key": "hotpotqa",
            "evaluator_payload": {"supporting_facts": {"title": ["private"]}},
        },
    }


def _catalog(path: Path) -> None:
    path.write_text(
        """
sources:
  hotpotqa:
    path: /unused
    files:
      train: train-*.parquet
      validation: validation-*.parquet
    candidate_sequence: [train, validation]
""".lstrip(),
        encoding="utf-8",
    )


def test_full_dataset_loader_projects_only_qa_and_native_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "datasets.yaml"
    _catalog(catalog)
    records = (
        _record(
            "hotpotqa:train-1",
            split="train",
            question="Who wrote it?",
            answer="Ada",
        ),
        _record(
            "hotpotqa:validation-1",
            split="validation",
            question="Where was it built?",
            answer="Rome",
        ),
    )
    monkeypatch.setattr(full_index, "_hotpot_records", lambda _config: iter(records))

    sources = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=catalog,
        expected_train_count=1,
        expected_validation_count=1,
    )

    assert sources.train[0].question == "Who wrote it?"
    assert sources.validation[0].question == "Where was it built?"
    assert sources.train[0].canonical_answer == "Ada"
    assert set(sources.train[0].__dataclass_fields__) == {
        "source_train_task_id",
        "base_task_id",
        "cycled",
        "question",
        "canonical_answer",
    }
    assert "supporting" not in repr(sources.combined).casefold()
    assert "private context" not in repr(sources.combined).casefold()


def test_full_dataset_loader_rejects_native_split_id_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "datasets.yaml"
    _catalog(catalog)
    records = (
        _record(
            "hotpotqa:same",
            split="train",
            question="Who?",
            answer="Ada",
        ),
        _record(
            "hotpotqa:same",
            split="validation",
            question="Who?",
            answer="Ada",
        ),
    )
    monkeypatch.setattr(full_index, "_hotpot_records", lambda _config: iter(records))

    with pytest.raises(ValueError, match="overlap"):
        load_hotpotqa_full_dataset_qa_sources(
            dataset_catalog_path=catalog,
            expected_train_count=1,
            expected_validation_count=1,
        )


def test_dataset_pair_fallback_has_independent_provenance_and_valid_contract() -> None:
    source = HotpotQATrainQASource(
        source_train_task_id="hotpotqa:train-1",
        base_task_id="hotpotqa:train-1",
        cycled=False,
        question="Who wrote the book?",
        canonical_answer="Ada Lovelace",
    )

    value = _dataset_pair_fallback(source)
    memory = materialize_hotpotqa_qa_memories((source,), (value,))[0]

    assert value["paraphrase_provenance"] == DATASET_PAIR_FALLBACK_PROVENANCE
    assert set(value) == {
        "source_train_task_id",
        "paraphrase_question",
        "paraphrase_answer_statement",
        "paraphrase_provenance",
        "paraphrase_version",
        "semantic_preservation_attested",
    }
    assert memory.canonical_answer == "Ada Lovelace"
    assert "Ada Lovelace" in memory.paraphrase_answer_statement


def test_materializer_admits_bootstrap_and_does_not_regenerate_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = HotpotQATrainQASource(
        "hotpotqa:train-1", "hotpotqa:train-1", False, "Who?", "Ada"
    )
    validation = HotpotQATrainQASource(
        "hotpotqa:validation-1",
        "hotpotqa:validation-1",
        False,
        "Where?",
        "Rome",
    )
    monkeypatch.setattr(
        materializer,
        "load_hotpotqa_full_dataset_qa_sources",
        lambda **_kwargs: HotpotQAFullDatasetQASources((train,), (validation,)),
    )
    bootstrap_path = tmp_path / "bootstrap.jsonl"
    bootstrap_path.write_text(
        json.dumps(
            {
                "source_train_task_id": "hotpotqa:train-1",
                "paraphrase_question": "Which person?",
                "paraphrase_answer_statement": "The person was Ada.",
                "paraphrase_provenance": "existing-local-qwen",
                "paraphrase_version": "existing-v1",
                "semantic_preservation_attested": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = Namespace(
        dataset_catalog=str(tmp_path / "unused.yaml"),
        train_count=1,
        validation_count=1,
        limit=None,
        output=str(tmp_path / "paraphrases.jsonl"),
        receipts=str(tmp_path / "receipts.jsonl"),
        manifest=str(tmp_path / "manifest.json"),
        bootstrap_materialization=[str(bootstrap_path)],
        materialize_pending_as_dataset_pair_fallback=True,
        model_catalog=str(tmp_path / "unused-models.yaml"),
        model_id="qwen3.5-9b-local",
        concurrency=2,
        checkpoint_every=2,
        seed=7,
        max_attempts=1,
        allow_dataset_pair_fallback=False,
    )

    manifest = asyncio.run(materializer.materialize(args))

    assert manifest["source_record_count"] == 2
    assert manifest["bootstrap_admitted_count"] == 1
    assert manifest["bootstrap_reused_count"] == 1
    assert manifest["dataset_pair_fallback_count"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "paraphrases.jsonl").read_text().splitlines()
    ]
    assert rows[0]["paraphrase_provenance"] == "existing-local-qwen"


def _manifest() -> HotpotQAFullDatasetQAMemoryIndexManifest:
    return HotpotQAFullDatasetQAMemoryIndexManifest(
        schema_version=FULL_DATASET_QA_MEMORY_SCHEMA_VERSION,
        index_id="hotpotqa-full-dataset-test",
        corpus_version=FULL_DATASET_QA_MEMORY_CORPUS_VERSION,
        source="HotpotQA native train + validation Q-A",
        source_splits=("train", "validation"),
        embedding_model="local-bge",
        embedding_model_path="/models/bge",
        embedding_dimension=2,
        normalized=True,
        similarity="cosine",
        frozen_top_k=1,
        source_record_count=2,
        source_train_count=1,
        source_validation_count=1,
        unique_source_count=2,
        cycled_record_count=0,
        paraphrase_count=2,
        frozen_evaluation_count=1,
        evaluation_overlap_count=1,
        contains_evaluation_answers=True,
        evaluation_scope=FULL_DATASET_EVALUATION_SCOPE,
        official_heldout_eligible=False,
        paraphrase_versions=("test-v1",),
        paraphrase_provenances=("unit-test",),
        document_template=QA_MEMORY_DOCUMENT_TEMPLATE,
        source_dataset_catalog_path="/datasets/catalog.yaml",
        source_train_path="/datasets/train-*.parquet",
        source_validation_path="/datasets/validation-*.parquet",
        memories_path="memories.jsonl",
        embeddings_path="embeddings.npy",
    )


def test_manifest_round_trip_and_reused_search_read_runtime() -> None:
    manifest = _manifest()
    assert (
        HotpotQAFullDatasetQAMemoryIndexManifest.from_value(manifest.to_value())
        == manifest
    )
    sources = (
        HotpotQATrainQASource(
            "hotpotqa:train-1", "hotpotqa:train-1", False, "Who?", "Ada"
        ),
        HotpotQATrainQASource(
            "hotpotqa:validation-1",
            "hotpotqa:validation-1",
            False,
            "Where?",
            "Rome",
        ),
    )
    memories = materialize_hotpotqa_qa_memories(
        sources,
        (
            {
                "source_train_task_id": "hotpotqa:train-1",
                "paraphrase_question": "Which person?",
                "paraphrase_answer_statement": "The person was Ada.",
                "paraphrase_provenance": "unit-test",
                "paraphrase_version": "test-v1",
                "semantic_preservation_attested": True,
            },
            {
                "source_train_task_id": "hotpotqa:validation-1",
                "paraphrase_question": "In which place?",
                "paraphrase_answer_statement": "The place was Rome.",
                "paraphrase_provenance": "unit-test",
                "paraphrase_version": "test-v1",
                "semantic_preservation_attested": True,
            },
        ),
    )

    class Model:
        def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
            assert texts
            return np.asarray([[1.0, 0.0] for _item in texts], dtype=np.float32)

    index = HotpotQAFullDatasetQAMemoryIndex(
        manifest=manifest,  # type: ignore[arg-type]
        memories=memories,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        model=Model(),
    )
    hits = asyncio.run(index.search("Who?", 1))
    assert hits[0].source_train_task_id == "hotpotqa:train-1"
    assert index.read(hits[0].memory_id).canonical_answer == "Ada"


def test_manifest_rejects_non_transductive_claim() -> None:
    value = _manifest().to_value()
    value["official_heldout_eligible"] = True
    with pytest.raises(ValueError, match="not held-out eligible"):
        HotpotQAFullDatasetQAMemoryIndexManifest.from_value(
            json.loads(json.dumps(value))
        )
