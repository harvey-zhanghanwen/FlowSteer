from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_triviaqa_qa_memory_paraphrases import (
    FACT_PROVENANCE_SCHEMA_VERSION,
    write_jsonl_projection,
)
from scripts.materialize_triviaqa_train_fact_subset import (
    MATERIALIZATION_SCHEMA_VERSION,
    materialize_train_only_fact_subset,
)
from src.interactive.triviaqa_qa_memory import (
    FACT_MEMORY_RECORD_SCHEMA_VERSION,
    FACT_MEMORY_TOOL_ID,
    TriviaQAFactMemoryRecord,
    load_materialized_fact_memory,
    write_materialized_fact_memory,
)


def _task(
    base_task_id: str,
    *,
    question: str,
    answer: str,
    split: str,
    selection_index: int,
) -> dict[str, object]:
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": base_task_id,
        "question": question,
        "ground_truth": answer,
        "split": split,
        "metadata": {
            "dataset_key": "triviaqa",
            "native_split": "train",
            "evaluator_payload": {"accepted_answers": [answer]},
            "sampling": {
                "selection": "sequential",
                "selection_index": selection_index,
                "base_task_id": base_task_id,
                "cycled_training_sample": False,
            },
        },
    }


def _provenance(
    task: dict[str, object],
    fact: TriviaQAFactMemoryRecord,
    *,
    selection_index: int,
) -> dict[str, object]:
    metadata = task["metadata"]
    assert isinstance(metadata, dict)
    evaluator_payload = metadata["evaluator_payload"]
    assert isinstance(evaluator_payload, dict)
    answer = evaluator_payload["accepted_answers"][0]
    return {
        "schema_version": FACT_PROVENANCE_SCHEMA_VERSION,
        "memory_id": fact.memory_id,
        "source_train_task_id": task["task_id"],
        "base_task_id": task["task_id"],
        "selection_index": selection_index,
        "original_question": task["question"],
        "accepted_answers": [answer],
        "canonical_answer": answer,
        "fact_text": fact.fact_text,
        "paraphrase_question": f"Reworded {task['question']}",
        "paraphrase_version": "triviaqa.qa_memory.paraphrase.v12",
        "paraphrase_method": "semantic-preserving-question-and-answer-paraphrase",
        "generator_provider": "local-openai-compatible",
        "model_id": "supervisor_theta",
        "model_revision": "fixture",
        "prompt_template_version": "triviaqa.qa_memory.qa_paraphrase.v12",
        "generation_seed": selection_index,
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_materializes_train_facts_and_keeps_qa_only_in_provenance(
    tmp_path: Path,
) -> None:
    train_tasks = (
        _task(
            "triviaqa:train_a",
            question="Who founded Example A?",
            answer="Founder A",
            split="train",
            selection_index=0,
        ),
        _task(
            "triviaqa:train_b",
            question="Where is Example B located?",
            answer="Place B",
            split="train",
            selection_index=1,
        ),
    )
    validation_tasks = (
        _task(
            "triviaqa:heldout_c",
            question="When did Example C begin?",
            answer="1901",
            split="validation",
            selection_index=0,
        ),
    )
    facts = (
        TriviaQAFactMemoryRecord.create(
            fact_text="Founder A founded Example A."
        ),
        TriviaQAFactMemoryRecord.create(
            fact_text="Example B is located in Place B."
        ),
        TriviaQAFactMemoryRecord.create(
            fact_text="Example C began in 1901."
        ),
    )

    train_path = tmp_path / "aligned" / "train.jsonl"
    validation_path = tmp_path / "aligned" / "validation.jsonl"
    full_facts_path = tmp_path / "full" / "facts" / "fact_memory.jsonl"
    full_provenance_path = (
        tmp_path / "full" / "provenance" / "fact_provenance.jsonl"
    )
    write_jsonl_projection(train_path, train_tasks)
    write_jsonl_projection(validation_path, validation_tasks)
    write_materialized_fact_memory(full_facts_path, facts)
    write_jsonl_projection(
        full_provenance_path,
        (
            _provenance(validation_tasks[0], facts[2], selection_index=0),
            _provenance(train_tasks[0], facts[0], selection_index=1),
            _provenance(train_tasks[1], facts[1], selection_index=2),
        ),
    )

    output_dir = tmp_path / "train_only"
    manifest = materialize_train_only_fact_subset(
        train_tasks_path=train_path,
        validation_tasks_path=validation_path,
        full_fact_memory_path=full_facts_path,
        full_provenance_path=full_provenance_path,
        output_dir=output_dir,
        expected_train_count=2,
        expected_validation_count=1,
    )

    selected_facts = load_materialized_fact_memory(
        output_dir / "facts" / "fact_memory.jsonl",
        expected_count=2,
    )
    selected_rows = _jsonl(output_dir / "facts" / "fact_memory.jsonl")
    selected_provenance = _jsonl(
        output_dir / "provenance" / "fact_provenance.jsonl"
    )
    heldout_memory_id = facts[2].memory_id
    forbidden_agent_fields = {
        "source_train_task_id",
        "base_task_id",
        "original_question",
        "accepted_answers",
        "canonical_answer",
        "paraphrase_question",
    }

    assert {record.memory_id for record in selected_facts} == {
        facts[0].memory_id,
        facts[1].memory_id,
    }
    assert heldout_memory_id not in {record.memory_id for record in selected_facts}
    assert all(
        set(row)
        == {"schema_version", "memory_id", "tool_id", "fact_text"}
        for row in selected_rows
    )
    assert all(not forbidden_agent_fields.intersection(row) for row in selected_rows)
    assert [row["base_task_id"] for row in selected_provenance] == [
        "triviaqa:train_a",
        "triviaqa:train_b",
    ]
    assert [row["selection_index"] for row in selected_provenance] == [0, 1]
    assert all("original_question" in row for row in selected_provenance)
    assert all("canonical_answer" in row for row in selected_provenance)
    assert manifest == json.loads(
        (output_dir / "materialization_manifest.json").read_text()
    )
    assert manifest["schema_version"] == MATERIALIZATION_SCHEMA_VERSION
    assert manifest["fact_memory_schema_version"] == (
        FACT_MEMORY_RECORD_SCHEMA_VERSION
    )
    assert manifest["tool_id"] == FACT_MEMORY_TOOL_ID
    assert manifest["train_task_count"] == 2
    assert manifest["selected_fact_count"] == 2
    assert manifest["validation_task_count"] == 1
    assert manifest["train_validation_base_task_id_overlap_count"] == 0
    assert manifest["validation_memory_id_overlap_count"] == 0
    assert manifest["validation_content_indexed"] is False
    assert manifest["original_question_indexed"] is False
    assert manifest["accepted_answers_indexed"] is False
    assert manifest["canonical_answer_indexed"] is False
    assert not tuple(output_dir.rglob("*.partial"))


def test_rejects_train_validation_identity_overlap_before_release(
    tmp_path: Path,
) -> None:
    overlapping_train = _task(
        "triviaqa:shared",
        question="Who founded Shared?",
        answer="Founder",
        split="train",
        selection_index=0,
    )
    overlapping_validation = _task(
        "triviaqa:shared",
        question="Who founded Shared?",
        answer="Founder",
        split="validation",
        selection_index=0,
    )
    fact = TriviaQAFactMemoryRecord.create(
        fact_text="Founder founded Shared."
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    full_facts_path = tmp_path / "full" / "facts.jsonl"
    full_provenance_path = tmp_path / "full" / "provenance.jsonl"
    output_dir = tmp_path / "should_not_exist"
    write_jsonl_projection(train_path, (overlapping_train,))
    write_jsonl_projection(validation_path, (overlapping_validation,))
    write_materialized_fact_memory(full_facts_path, (fact,))
    write_jsonl_projection(
        full_provenance_path,
        (_provenance(overlapping_train, fact, selection_index=0),),
    )

    with pytest.raises(ValueError, match="train and validation.*overlap"):
        materialize_train_only_fact_subset(
            train_tasks_path=train_path,
            validation_tasks_path=validation_path,
            full_fact_memory_path=full_facts_path,
            full_provenance_path=full_provenance_path,
            output_dir=output_dir,
            expected_train_count=1,
            expected_validation_count=1,
        )

    assert not output_dir.exists()
