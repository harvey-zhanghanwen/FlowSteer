import json

import pytest

from src.interactive.task_dataset import (
    TASK_SCHEMA_VERSION,
    hotpotqa_answer_cardinality_constraint,
    hotpotqa_answer_type_constraint,
    hotpotqa_question_scope,
    qa_answer_type_constraint_accepts,
    iter_task_records,
    task_record_from_mapping,
)


def _item(split="train"):
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": "hotpotqa:1",
        "question": "Who wrote the book?",
        "ground_truth": "An Author",
        "split": split,
        "metadata": {
            "source": "HotpotQA",
            "dataset_key": "hotpotqa",
            "metric": "token_f1",
        },
        "source": "HotpotQA",
        "answer": "An Author",
        "task_type": "multi_hop_qa",
        "context": [],
        "extra": {"source": "HotpotQA", "metric": "token_f1"},
    }


def test_task_record_from_mapping_keeps_only_runtime_contract():
    record = task_record_from_mapping(_item(), expected_split="train")
    assert record.task_id == "hotpotqa:1"
    assert record.ground_truth == "An Author"
    assert record.metadata["source"] == "HotpotQA"


def test_expected_split_blocks_cross_split_loading():
    with pytest.raises(ValueError, match="split isolation violation"):
        task_record_from_mapping(_item("test"), expected_split="train")


def test_hotpot_loader_rehydrates_full_context_without_evaluator_payload():
    item = _item()
    tail = "decisive evidence after the former 300-character boundary"
    item["question"] = (
        "Based on the following passages, answer the question.\n\n"
        "[[Title] truncated]\n\nQuestion: Who wrote the book?"
    )
    item["context"] = ["[Title] " + ("x" * 350) + tail]
    item["metadata"]["evaluator_payload"] = {
        "supporting_facts": [["Title", 1]]
    }

    record = task_record_from_mapping(item)

    assert tail in record.question
    assert "supporting_facts" not in record.question
    assert record.metadata["hotpot_context_mode"] == "full_passages_v1"
    assert hotpotqa_question_scope(record.question) == "Who wrote the book?"


def test_hotpot_question_scope_preserves_plain_question():
    assert hotpotqa_question_scope("Who wrote the book?") == "Who wrote the book?"


@pytest.mark.parametrize(
    ("question", "answer_type"),
    [
        ("Which magazine was started first, A or B?", "entity"),
        ("The character was named after who?", "entity"),
        ("Who won Super Bowl XX?", "entity"),
        ("What nationality was the singer's wife?", "nationality"),
        ("How many albums were released?", "number"),
        ("Was the film released first?", "yes_no"),
        ('"Human Error" belongs to a show that aired on what network?', "entity"),
        ("The documentary was narrated by which actor?", "entity"),
    ],
)
def test_hotpot_answer_type_constraint_uses_question_surface_only(
    question: str,
    answer_type: str,
) -> None:
    assert hotpotqa_answer_type_constraint(question) == answer_type


@pytest.mark.parametrize(
    ("question", "observed_type", "accepted"),
    [
        ("The company has a head office in what city?", "location", True),
        ("The company has a head office in what city?", "city", True),
        ("The company has a head office in what city?", "date", False),
        ("Where is the company headquartered?", "location", True),
        ("Where is the company headquartered?", "city", False),
        ("Which magazine was started first, A or B?", "entity", True),
        ("Which magazine was started first, A or B?", "magazine", True),
        ("Which magazine was started first, A or B?", "date", False),
    ],
)
def test_qa_answer_type_constraint_accepts_explicit_lexical_subtypes(
    question: str,
    observed_type: str,
    accepted: bool,
) -> None:
    assert qa_answer_type_constraint_accepts(question, observed_type) is accepted


@pytest.mark.parametrize(
    ("question", "cardinality"),
    [
        ("Who are Metallica's current members?", "multiple"),
        ("What are the names of the two founders?", "multiple"),
        ("Name all members of the group.", "multiple"),
        ("Which magazine was started first, A or B?", "single"),
        ("Who narrated the documentary?", "single"),
    ],
)
def test_hotpot_answer_cardinality_constraint_uses_question_surface_only(
    question: str,
    cardinality: str,
) -> None:
    assert hotpotqa_answer_cardinality_constraint(question) == cardinality


def test_iter_task_records_reports_jsonl_line(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_item()) + "\nnot-json\n", encoding="utf-8")
    iterator = iter_task_records(source, expected_split="train")
    assert next(iterator).task_id == "hotpotqa:1"
    with pytest.raises(ValueError, match=r"train.jsonl:2"):
        next(iterator)


def test_zero_limit_does_not_read_a_record(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    assert list(iter_task_records(source, expected_split="train", limit=0)) == []
