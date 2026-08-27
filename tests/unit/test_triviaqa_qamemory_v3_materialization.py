from __future__ import annotations

import json

import pytest

from src.interactive.triviaqa_qa_memory import (
    TriviaQAQAMemoryRecord,
    TriviaQATrainSource,
    relation_bearing_answer_statement,
    validate_qa_memory_against_sources,
)
from scripts.generate_triviaqa_qa_memory_paraphrases import (
    load_resume_records,
    parse_paraphrase_response,
)


def _source() -> TriviaQATrainSource:
    return TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_999",
        base_task_id="triviaqa:tc_999",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which river contains the Kariba Dam?",
        canonical_answer="Zambezi",
        native_split="train",
    )


def _record(
    *,
    question: str = "Name the river on which the Kariba Dam was built.",
    statement: str = "The Kariba Dam was built on the Zambezi river.",
) -> TriviaQAQAMemoryRecord:
    return TriviaQAQAMemoryRecord.create(
        source=_source(),
        paraphrase_question=question,
        paraphrase_answer_statement=statement,
        paraphrase_version="triviaqa.qa_memory.paraphrase.v2",
        paraphrase_method="semantic-preserving-question-and-answer-paraphrase",
        generator_provider="local-openai-compatible",
        model_id="supervisor_theta",
        model_revision="Qwen3.5-9B-local",
        prompt_template_version="triviaqa.qa_memory.qa_paraphrase.v4",
        generation_seed=20260827,
    )


def test_v4_materialization_keeps_relation_and_exact_canonical_span() -> None:
    record = _record()

    assert record.canonical_answer in record.paraphrase_answer_statement
    validate_qa_memory_against_sources((record,), (_source(),), require_complete=True)


@pytest.mark.parametrize(
    "statement",
    (
        "Zambezi",
        "Zambezi.",
        "The answer is Zambezi",
        "The canonical answer is Zambezi.",
        "Zambezi is the answer.",
    ),
)
def test_v4_materialization_rejects_answer_only_statement(statement: str) -> None:
    with pytest.raises(ValueError, match="express the question relation beyond"):
        _record(statement=statement)


def test_relation_bearing_statement_requires_context_beyond_canonical() -> None:
    assert relation_bearing_answer_statement(
        "The Kariba Dam was built on the Zambezi river.",
        "Zambezi",
    )
    assert not relation_bearing_answer_statement("Zambezi", "Zambezi")


def test_response_parser_rejects_bare_canonical_answer() -> None:
    response = json.dumps(
        {
            "paraphrase_question": "Name the river holding the Kariba Dam.",
            "paraphrase_answer_statement": "Zambezi",
        }
    )

    with pytest.raises(ValueError, match="express the question relation beyond"):
        parse_paraphrase_response(response, _source())


def test_resume_keeps_valid_rows_and_drops_only_answer_only_rows(tmp_path) -> None:
    valid = _record()
    raw_invalid = valid.to_value()
    raw_invalid["source_train_task_id"] = "triviaqa:tc_1000"
    raw_invalid["base_task_id"] = "triviaqa:tc_1000"
    raw_invalid["selection_index"] = 1
    raw_invalid["paraphrase_answer_statement"] = "Zambezi"
    # Resume filtering happens before strict record construction, so the stale
    # identity from the rejected row is intentionally not recomputed.
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (valid.to_value(), raw_invalid)
        )
        + "\n",
        encoding="utf-8",
    )

    records, rejected = load_resume_records(checkpoint)

    assert records == (valid,)
    assert rejected == ("triviaqa:tc_1000",)


def test_v4_materialization_rejects_new_answer_surface_in_question() -> None:
    record = _record(question="Is the answer Zambezi?")

    with pytest.raises(ValueError, match="introduced the canonical answer"):
        validate_qa_memory_against_sources(
            (record,),
            (_source(),),
            require_complete=True,
        )


def test_v4_materialization_allows_original_spelling_normalization() -> None:
    source = TriviaQATrainSource(
        source_train_task_id="triviaqa:tc_735",
        base_task_id="triviaqa:tc_735",
        selection_index=0,
        cycled_training_sample=False,
        cycle_index=None,
        original_question="Which state is misspelled as Rode island?",
        canonical_answer="Rhode Island",
        native_split="train",
    )
    record = TriviaQAQAMemoryRecord.create(
        source=source,
        paraphrase_question="Which state is intended by the spelling Rhode Island?",
        paraphrase_answer_statement=(
            "The state intended by the misspelling is Rhode Island."
        ),
        paraphrase_version="triviaqa.qa_memory.paraphrase.v4",
        paraphrase_method="semantic-preserving-question-and-answer-paraphrase",
        generator_provider="local-openai-compatible",
        model_id="supervisor_theta",
        model_revision="Qwen3.5-9B-local",
        prompt_template_version="triviaqa.qa_memory.qa_paraphrase.v4",
        generation_seed=20260827,
    )

    validate_qa_memory_against_sources((record,), (source,), require_complete=True)
