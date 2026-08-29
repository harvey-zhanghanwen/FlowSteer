from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from src.interactive.triviaqa_embedding_index import BGE_QUERY_PREFIX
from src.interactive.triviaqa_qa_memory import (
    FACT_MEMORIES_FILENAME,
    FACT_MEMORY_EMBEDDING_INPUT_FIELD,
    FACT_MEMORY_RECORD_KIND,
    FACT_MEMORY_TOOL_ID,
    TriviaQAFactMemoryIndex,
    TriviaQAFactMemoryRecord,
    build_triviaqa_fact_memory_index,
    load_materialized_fact_memory,
    open_triviaqa_memory_index,
    write_materialized_fact_memory,
)


class _RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        sentences: Sequence[str],
        **_: object,
    ) -> np.ndarray:
        values = tuple(sentences)
        self.calls.append(values)
        vectors: list[list[float]] = []
        for text in values:
            lowered = text.casefold()
            if "capital" in lowered or "paris" in lowered:
                vectors.append([1.0, 0.0])
            elif "marengo" in lowered or "battle" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _records() -> tuple[TriviaQAFactMemoryRecord, ...]:
    return (
        TriviaQAFactMemoryRecord.create(
            fact_text="Paris is the capital of France.",
        ),
        TriviaQAFactMemoryRecord.create(
            fact_text="The Battle of Marengo occurred on June 14, 1800.",
        ),
    )


def test_fact_record_is_exactly_four_agent_facing_fields() -> None:
    record = _records()[0]

    assert set(record.to_value()) == {
        "schema_version",
        "memory_id",
        "tool_id",
        "fact_text",
    }
    assert record.tool_id == FACT_MEMORY_TOOL_ID == "triviaqa.qa_memory"
    assert record.embedding_text() == record.fact_text
    assert record.document_id == record.memory_id
    assert record.title == "TriviaQA fact"

    for forbidden_field in (
        "source_train_task_id",
        "question",
        "answer",
        "canonical_answer",
        "accepted_answers",
    ):
        value = record.to_value()
        value[forbidden_field] = "not Agent-facing"
        with pytest.raises(ValueError, match="only schema_version"):
            TriviaQAFactMemoryRecord.from_value(value)

    legacy_opaque = record.to_value()
    legacy_opaque["memory_id"] = "triviaqa-qa-memory-" + ("a" * 64)
    assert TriviaQAFactMemoryRecord.from_value(legacy_opaque).memory_id == (
        legacy_opaque["memory_id"]
    )


@pytest.mark.parametrize(
    "fact_text",
    (
        "Question: Who wrote Dune?\nAnswer: Frank Herbert",
        "Q: Who wrote Dune?\nA: Frank Herbert",
        "The answer is Frank Herbert.",
        "Who wrote Dune?",
        "Who wrote Dune? Frank Herbert.",
        'Frank Herbert wrote the novel "Dune?.',
    ),
)
def test_fact_record_rejects_qa_pair_and_non_declarative_wrappers(
    fact_text: str,
) -> None:
    with pytest.raises(ValueError, match="fact_text"):
        TriviaQAFactMemoryRecord.create(fact_text=fact_text)


def test_fact_record_accepts_question_mark_inside_balanced_quoted_title() -> None:
    fact = 'Frank Herbert wrote the novel "Dune?".'

    record = TriviaQAFactMemoryRecord.create(fact_text=fact)

    assert record.fact_text == fact
    assert record.embedding_text() == fact


def test_builder_fails_closed_before_embedding_mixed_question_answer_text(
    tmp_path: Path,
) -> None:
    record = _records()[0].to_value()
    record["fact_text"] = "Who wrote Dune? Frank Herbert."
    facts_path = tmp_path / "mixed-question-answer.jsonl"
    facts_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output_dir = tmp_path / "index"
    encoder = _RecordingEncoder()

    with pytest.raises(ValueError, match="declarative rather than interrogative"):
        build_triviaqa_fact_memory_index(
            facts_path=facts_path,
            output_dir=output_dir,
            embedding_model="local-bge",
            embedding_model_revision="local-revision",
            frozen_top_k=1,
            max_tool_calls_per_agent_call=2,
            max_turns_per_agent_call=3,
            encoder=encoder,
            expected_count=1,
        )

    assert encoder.calls == []
    assert not output_dir.exists()


def test_builder_embeds_only_fact_text_and_manifest_is_fact_only(
    tmp_path: Path,
) -> None:
    records = _records()
    facts_path = tmp_path / "materialized-facts.jsonl"
    index_path = tmp_path / "index"
    provenance_path = tmp_path / "external-provenance.jsonl"
    write_materialized_fact_memory(facts_path, records)
    provenance_path.write_text(
        json.dumps(
            {
                "source_question": "Who wrote Dune?",
                "canonical_answer": "Frank Herbert",
                "accepted_answers": ["F. Herbert"],
            }
        ),
        encoding="utf-8",
    )
    encoder = _RecordingEncoder()

    manifest = build_triviaqa_fact_memory_index(
        facts_path=facts_path,
        output_dir=index_path,
        embedding_model="local-bge",
        embedding_model_revision="local-revision",
        frozen_top_k=1,
        max_tool_calls_per_agent_call=2,
        max_turns_per_agent_call=3,
        encoder=encoder,
        batch_size=2,
        snippet_characters=256,
        expected_count=2,
    )

    ordered = tuple(
        record.fact_text for record in sorted(records, key=lambda item: item.memory_id)
    )
    assert encoder.calls == [ordered]
    assert manifest.record_kind == FACT_MEMORY_RECORD_KIND
    assert manifest.fact_only is True
    assert manifest.embedding_input_field == FACT_MEMORY_EMBEDDING_INPUT_FIELD
    assert manifest.provenance_loaded_by_index is False
    assert manifest.agent_facing_record_fields == (
        "schema_version",
        "memory_id",
        "tool_id",
        "fact_text",
    )
    manifest_value = manifest.to_value()
    assert "provenance" not in manifest_value
    assert "provenance_path" not in manifest_value
    assert "source" not in manifest_value

    persisted_rows = [
        json.loads(line)
        for line in (index_path / FACT_MEMORIES_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(set(row) == set(records[0].to_value()) for row in persisted_rows)
    serialized = json.dumps(persisted_rows, sort_keys=True)
    for forbidden in (
        "source_train_task_id",
        "canonical_answer",
        "accepted_answers",
        "Question:",
        "Answer:",
    ):
        assert forbidden not in serialized

    # The external provenance projection is deliberately unavailable when the
    # runtime index opens; open/search/read therefore cannot depend on it.
    provenance_path.unlink()
    with TriviaQAFactMemoryIndex.open(index_path, encoder=encoder) as index:
        hits = index.search("Where is France's capital?", limit=1)
        assert len(hits) == 1
        assert hits[0].fact_text == "Paris is the capital of France."
        assert index.read(hits[0].memory_id).fact_text == hits[0].fact_text
    assert encoder.calls[-1] == (
        BGE_QUERY_PREFIX + "Where is France's capital?",
    )
    assert "Question:" not in encoder.calls[-1][0]


def test_versioned_opener_dispatches_fact_memory_without_provenance(
    tmp_path: Path,
) -> None:
    records = _records()
    facts_path = tmp_path / "facts.jsonl"
    index_path = tmp_path / "index"
    write_materialized_fact_memory(facts_path, records)
    encoder = _RecordingEncoder()
    build_triviaqa_fact_memory_index(
        facts_path=facts_path,
        output_dir=index_path,
        embedding_model="local-bge",
        embedding_model_revision="local-revision",
        frozen_top_k=1,
        max_tool_calls_per_agent_call=2,
        max_turns_per_agent_call=3,
        encoder=encoder,
        expected_count=2,
    )

    index = open_triviaqa_memory_index(index_path, encoder=encoder)
    try:
        assert isinstance(index, TriviaQAFactMemoryIndex)
        assert index.manifest.record_kind == "fact_memory"
    finally:
        index.close()


def test_fact_loader_fails_closed_on_extra_provenance_fields(
    tmp_path: Path,
) -> None:
    record = _records()[0].to_value()
    record["source_task_id"] = "triviaqa:private"
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only schema_version"):
        load_materialized_fact_memory(path, expected_count=1)
