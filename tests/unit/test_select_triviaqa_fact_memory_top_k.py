from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from scripts.select_triviaqa_fact_memory_top_k import (
    RECEIPT_SCHEMA_VERSION,
    select_triviaqa_fact_memory_top_k,
)
from src.interactive.triviaqa_embedding_index import BGE_QUERY_PREFIX
from src.interactive.triviaqa_qa_memory import (
    TriviaQAFactMemoryRecord,
    build_triviaqa_fact_memory_index,
    write_materialized_fact_memory,
)


class _UnitVectorEncoder:
    def __init__(self, fact_texts: Sequence[str]) -> None:
        self._fact_vectors = {
            text: np.eye(len(fact_texts), dtype=np.float32)[index]
            for index, text in enumerate(fact_texts)
        }
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        sentences: Sequence[str],
        **_: object,
    ) -> np.ndarray:
        values = tuple(sentences)
        self.calls.append(values)
        query_vector = np.asarray(
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            dtype=np.float32,
        )
        return np.stack(
            [
                self._fact_vectors[text]
                if text in self._fact_vectors
                else query_vector
                for text in values
            ]
        )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _task_row(
    *,
    task_id: str,
    base_task_id: str,
    selection_index: int,
    question: str,
    split: str,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "question": question,
        "ground_truth": "EVALUATOR-PRIVATE",
        "split": split,
        "metadata": {
            "dataset_key": "triviaqa",
            "sampling": {
                "base_task_id": base_task_id,
                "selection_index": selection_index,
                "cycled_training_sample": False,
            },
            "evaluator_payload": {
                "accepted_answers": ["EVALUATOR-PRIVATE"]
            },
        },
    }


def _fixture(
    tmp_path: Path,
    *,
    development_fact_indices: Sequence[int] = (0, 1, 2, 4),
) -> dict[str, object]:
    fact_texts = tuple(
        f"Entity {index} has relation value {index}." for index in range(6)
    )
    records = tuple(
        TriviaQAFactMemoryRecord.create(fact_text=text) for text in fact_texts
    )
    facts_path = tmp_path / "facts.jsonl"
    index_path = tmp_path / "index"
    development_path = tmp_path / "development.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    provenance_path = tmp_path / "provenance.jsonl"
    receipt_path = tmp_path / "top-k-receipt.json"
    encoder = _UnitVectorEncoder(fact_texts)
    write_materialized_fact_memory(facts_path, records)
    build_triviaqa_fact_memory_index(
        facts_path=facts_path,
        output_dir=index_path,
        embedding_model="local-bge",
        embedding_model_revision="local-revision",
        frozen_top_k=5,
        max_tool_calls_per_agent_call=6,
        max_turns_per_agent_call=7,
        encoder=encoder,
        expected_count=6,
    )

    development_rows = [
        _task_row(
            task_id=f"triviaqa:dev_{position}",
            base_task_id=f"triviaqa:base_{fact_index}",
            selection_index=position,
            question=f"Development query {position}",
            split="train",
        )
        for position, fact_index in enumerate(development_fact_indices)
    ]
    validation_indices = tuple(
        index for index in range(6) if index not in development_fact_indices
    )[:2]
    validation_rows = [
        _task_row(
            task_id=f"triviaqa:validation_{position}",
            base_task_id=f"triviaqa:base_{fact_index}",
            selection_index=position,
            question="VALIDATION QUESTION MUST NOT BE CONSUMED",
            split="validation",
        )
        for position, fact_index in enumerate(validation_indices)
    ]
    provenance_rows = [
        {
            "schema_version": "flowsteer.triviaqa.fact_memory.provenance.v1",
            "memory_id": record.memory_id,
            "source_train_task_id": f"triviaqa:base_{index}",
            "base_task_id": f"triviaqa:base_{index}",
            "canonical_answer": f"SECRET ANSWER {index}",
            "accepted_answers": [f"SECRET ALIAS {index}"],
            "original_question": f"SECRET ORIGINAL QUESTION {index}",
        }
        for index, record in enumerate(records)
    ]
    _write_jsonl(development_path, development_rows)
    _write_jsonl(validation_path, validation_rows)
    _write_jsonl(provenance_path, provenance_rows)
    return {
        "encoder": encoder,
        "index_path": index_path,
        "development_path": development_path,
        "validation_path": validation_path,
        "provenance_path": provenance_path,
        "receipt_path": receipt_path,
        "records": records,
        "development_count": len(development_rows),
        "validation_count": len(validation_rows),
    }


def _select(paths: dict[str, object]) -> dict[str, object]:
    return select_triviaqa_fact_memory_top_k(
        index_dir=paths["index_path"],
        development_tasks_path=paths["development_path"],
        validation_tasks_path=paths["validation_path"],
        provenance_path=paths["provenance_path"],
        output_path=paths["receipt_path"],
        expected_development_count=paths["development_count"],
        expected_validation_count=paths["validation_count"],
        encoder=paths["encoder"],
    )


def test_selector_reports_recall_mrr_and_freezes_smallest_maximum_k(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    index_path = paths["index_path"]
    assert isinstance(index_path, Path)
    index_before = {
        path.name: path.read_bytes() for path in index_path.iterdir() if path.is_file()
    }

    receipt = _select(paths)

    results = {
        row["top_k"]: row for row in receipt["candidate_results"]
    }
    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert results[1]["hit_count"] == 1
    assert results[1]["recall"] == pytest.approx(0.25)
    assert results[1]["mean_reciprocal_rank"] == pytest.approx(0.25)
    assert results[3]["hit_count"] == 3
    assert results[3]["recall"] == pytest.approx(0.75)
    assert results[3]["mean_reciprocal_rank"] == pytest.approx(
        (1.0 + 0.5 + (1.0 / 3.0)) / 4.0
    )
    assert results[5]["hit_count"] == 4
    assert results[5]["recall"] == pytest.approx(1.0)
    assert results[5]["mean_reciprocal_rank"] == pytest.approx(
        (1.0 + 0.5 + (1.0 / 3.0) + 0.2) / 4.0
    )
    assert receipt["selected_top_k"] == 5
    assert receipt["validation_used_for_selection"] is False
    assert receipt["validation_content_read"] is False
    assert receipt["provenance_written_to_index"] is False
    assert receipt["provenance_exposed_to_tool"] is False
    assert receipt["development_validation_base_task_id_overlap_count"] == 0
    assert index_before == {
        path.name: path.read_bytes() for path in index_path.iterdir() if path.is_file()
    }

    receipt_path = paths["receipt_path"]
    assert isinstance(receipt_path, Path)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    assert "SECRET ANSWER" not in serialized_receipt
    assert "SECRET ALIAS" not in serialized_receipt
    assert "SECRET ORIGINAL QUESTION" not in serialized_receipt
    encoder = paths["encoder"]
    assert isinstance(encoder, _UnitVectorEncoder)
    query_calls = encoder.calls[1:]
    assert query_calls == [
        (BGE_QUERY_PREFIX + f"Development query {index}",)
        for index in range(4)
    ]
    assert all("VALIDATION QUESTION" not in call[0] for call in query_calls)


def test_selector_chooses_three_when_three_and_five_have_equal_recall(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, development_fact_indices=(0, 1))

    receipt = _select(paths)

    results = {
        row["top_k"]: row for row in receipt["candidate_results"]
    }
    assert results[1]["recall"] == pytest.approx(0.5)
    assert results[3]["recall"] == pytest.approx(1.0)
    assert results[5]["recall"] == pytest.approx(1.0)
    assert receipt["selected_top_k"] == 3


def test_selector_rejects_development_validation_base_id_overlap(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    validation_path = paths["validation_path"]
    assert isinstance(validation_path, Path)
    overlapping = [
        _task_row(
            task_id="triviaqa:validation_overlap",
            base_task_id="triviaqa:base_0",
            selection_index=0,
            question="PRIVATE VALIDATION QUESTION",
            split="validation",
        )
    ]
    _write_jsonl(validation_path, overlapping)
    paths["validation_count"] = 1

    with pytest.raises(ValueError, match="overlap"):
        _select(paths)


def test_selector_rejects_stale_or_incomplete_external_provenance(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    provenance_path = paths["provenance_path"]
    assert isinstance(provenance_path, Path)
    rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(provenance_path, rows[:-1])

    with pytest.raises(ValueError, match="memory_id set differs"):
        _select(paths)
