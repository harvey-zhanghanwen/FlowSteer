from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_triviaqa_full_train_qa_memory import (
    project_unique_nonheldout_train,
)
from scripts.generate_triviaqa_qa_memory_paraphrases import (
    SemanticPreservationError,
    _semantic_relation_and_scope_preserved,
    reject_exact_question_identity_shortcut,
)
from src.interactive.triviaqa_qa_memory import (
    TriviaQAQAMemoryManifest,
    TriviaQATrainSource,
    load_triviaqa_qa_memory_sources,
)


def _native_record(task_id: str, question: str, answer: str) -> dict[str, object]:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": task_id,
        "question": question,
        "ground_truth": answer,
        "split": "train",
        "metadata": {
            "dataset_key": "triviaqa",
            "evaluator_payload": {"accepted_answers": [answer]},
        },
        "extra": {},
    }


def _validation_record(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "question": "held-out question",
        "ground_truth": "held-out answer",
        "split": "validation",
        "metadata": {
            "dataset_key": "triviaqa",
            "sampling": {
                "base_task_id": task_id,
                "selection_index": 0,
                "cycled_training_sample": False,
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_full_projection_deduplicates_identical_rows_and_can_include_heldout() -> None:
    records = [
        _native_record("triviaqa:tc_1", "Who won?", "Ada"),
        _native_record("triviaqa:tc_1", "Who won?", "Ada"),
        _native_record("triviaqa:tc_2", "Where?", "Rome"),
    ]

    heldout_projection, heldout_stats = project_unique_nonheldout_train(
        records,
        heldout_base_ids=["triviaqa:tc_1"],
    )
    transductive_projection, transductive_stats = (
        project_unique_nonheldout_train(
            records,
            heldout_base_ids=["triviaqa:tc_1"],
            include_heldout=True,
        )
    )

    assert [row["task_id"] for row in heldout_projection] == ["triviaqa:tc_2"]
    assert heldout_stats["admitted_unique_qa_count"] == 1
    assert [row["task_id"] for row in transductive_projection] == [
        "triviaqa:tc_1",
        "triviaqa:tc_2",
    ]
    assert [
        row["metadata"]["sampling"]["selection_index"]
        for row in transductive_projection
    ] == [0, 1]
    assert transductive_stats["raw_train_row_count"] == 3
    assert transductive_stats["raw_unique_question_id_count"] == 2
    assert transductive_stats["duplicate_raw_row_count"] == 1
    assert transductive_stats["heldout_unique_ids_indexed"] == 1


def test_full_projection_rejects_conflicting_duplicate_qa() -> None:
    with pytest.raises(ValueError, match="conflicting Q-A"):
        project_unique_nonheldout_train(
            [
                _native_record("triviaqa:tc_1", "Who won?", "Ada"),
                _native_record("triviaqa:tc_1", "Who won?", "Grace"),
            ],
            heldout_base_ids=["triviaqa:tc_1"],
            include_heldout=True,
        )


def test_exact_query_identity_shortcut_is_rejected() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_1",
        base_task_id="triviaqa:tc_1",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Who authored the novel Dune?",
        canonical_answer="Frank Herbert",
        native_split="train",
        accepted_answers_for_admission=("Frank Herbert",),
    )

    with pytest.raises(ValueError, match="complete original question"):
        reject_exact_question_identity_shortcut(
            source,
            paraphrase_question=(
                "Determine the answer to this: Who authored the novel Dune?"
            ),
            paraphrase_answer_statement=(
                "For Who authored the novel Dune?, the answer is Frank Herbert."
            ),
        )
    reject_exact_question_identity_shortcut(
        source,
        paraphrase_question="Which person wrote the novel Dune?",
        paraphrase_answer_statement="Frank Herbert wrote the novel Dune.",
    )


def test_answer_type_and_zodiac_scope_narrowing_are_rejected() -> None:
    with pytest.raises(SemanticPreservationError, match="type-of-what"):
        _semantic_relation_and_scope_preserved(
            original_question="The VS-300 was a type of what?",
            paraphrase_question="The VS-300 was what kind of aircraft?",
            paraphrase_answer_statement="The VS-300 was a helicopter.",
        )
    with pytest.raises(SemanticPreservationError, match="star-sign"):
        _semantic_relation_and_scope_preserved(
            original_question="What is the star sign for a June birthday?",
            paraphrase_question="Which celestial sign applies to a June birthday?",
            paraphrase_answer_statement="The star sign is Gemini.",
        )


def test_transductive_loader_and_manifest_require_explicit_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train = _native_record("triviaqa:tc_1", "Who won?", "Ada")
    train["metadata"]["native_split"] = "train"
    train["metadata"]["sampling"] = {
        "base_task_id": "triviaqa:tc_1",
        "selection_index": 0,
        "cycled_training_sample": False,
    }
    _write_jsonl(train_path, [train])
    _write_jsonl(validation_path, [_validation_record("triviaqa:tc_1")])

    with pytest.raises(ValueError, match="overlap"):
        load_triviaqa_qa_memory_sources(
            train_path,
            validation_path,
            expected_train_count=1,
            expected_validation_count=1,
        )
    sources, validation_ids = load_triviaqa_qa_memory_sources(
        train_path,
        validation_path,
        expected_train_count=1,
        expected_validation_count=1,
        allow_validation_overlap=True,
    )
    assert len(sources) == 1
    assert validation_ids == {"triviaqa:tc_1"}

    manifest = TriviaQAQAMemoryManifest.create(
        train_count=1,
        validation_isolation_count=0,
        validation_content_indexed=True,
        unique_source_count=1,
        cycled_count=0,
        paraphrase_version="full-train-v1",
        embedding_model="local-bge",
        embedding_model_revision="local-revision",
        embedding_dimension=2,
        frozen_top_k=1,
        snippet_characters=128,
        max_tool_calls_per_agent_call=2,
        max_turns_per_agent_call=3,
        memories_sha256="a" * 64,
        embeddings_sha256="b" * 64,
    )
    assert manifest.validation_content_indexed is True
    assert manifest.validation_isolation_count == 0
