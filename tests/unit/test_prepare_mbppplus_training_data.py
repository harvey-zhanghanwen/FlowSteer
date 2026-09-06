from scripts.prepare_mbppplus_training_data import _private_record, _public_record


def _source_row() -> dict:
    return {
        "source": "mbpp",
        "problem_type": "code",
        "problem": "Write a function named f.",
        "ground_truth": "def f(x):\n    return x + 1\n",
        "meta": {
            "task_id": "901",
            "entry_point": "f",
            "test": "value = 1\nassert f(value) == 2\n",
        },
    }


def test_private_reward_cases_follow_skillflow_non_empty_line_split() -> None:
    record = _private_record(_source_row(), split="train")

    assert record["evaluator_payload"]["test_cases"] == [
        "value = 1",
        "assert f(value) == 2",
    ]
    assert record["evaluator_payload"]["test"] == (
        "value = 1\nassert f(value) == 2\n"
    )


def test_public_record_exposes_tests_but_not_reference_solution() -> None:
    record = _public_record(_source_row(), split="train", selection_index=0)

    assert record["ground_truth"] is None
    assert record["answer"] is None
    assert record["metadata"]["public_test_program"] == (
        "value = 1\nassert f(value) == 2\n"
    )
    assert "def f(x)" not in record["question"]
