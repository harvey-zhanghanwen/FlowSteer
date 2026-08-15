import json

import pytest

from src.interactive.task_dataset import (
    TASK_SCHEMA_VERSION,
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
        "metadata": {"source": "HotpotQA", "metric": "token_f1"},
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
