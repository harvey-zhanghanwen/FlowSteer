from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.interactive.aime2026_adapter import (
    AIME2026_EVALUATOR_VERSION,
    AIME2026_TASK_FAMILY,
    canonical_aime_integer,
    score_aime2026_integer,
)
from src.interactive.task_dataset import iter_task_records


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_aime2026_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_aime2026_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_integer_scorer_matches_skillflow_protocol_10() -> None:
    assert score_aime2026_integer("42", ["42"]).accuracy == 1.0
    assert score_aime2026_integer("042", ["42"]).accuracy == 1.0
    assert score_aime2026_integer("+42", ["42"]).accuracy == 1.0
    assert score_aime2026_integer("42.0", ["42"]).accuracy == 0.0
    assert score_aime2026_integer("The answer is 42", ["42"]).accuracy == 0.0
    assert score_aime2026_integer("100%", ["1"]).accuracy == 0.0


def test_format_operator_boundary_submits_only_the_last_complete_answer() -> None:
    result = score_aime2026_integer(
        "scratch 999 <answer>41</answer> revised <answer>042</answer>",
        ["42"],
    )
    assert result.accuracy == 1.0
    assert result.scored_prediction == "042"
    assert result.structured_answer_extracted is True


@pytest.mark.parametrize("value", [True, -1, 1000, "42.0", "answer 42", ""])
def test_trusted_answer_validation_rejects_non_aime_targets(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_aime_integer(value)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_preparation_keeps_2025_development_and_2026_final_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.jsonl"
    development = tmp_path / "aime2025.jsonl"
    output = tmp_path / "aligned"
    catalog = tmp_path / "config" / "datasets_aime2026.yaml"
    catalog.parent.mkdir()
    _write_jsonl(
        history,
        [
            {
                "year": 2024,
                "index": index,
                "part": "AIME I" if index % 2 else "AIME II",
                "problem": f"Historical problem {index}",
                "answer": index,
            }
            for index in range(1, 5)
        ],
    )
    _write_jsonl(
        development,
        [
            {
                "problem": "Development problem I",
                "source": "AIME2025-I",
                "ground_truth": "7",
            },
            {
                "problem": "Development problem II",
                "source": "AIME2025-II",
                "ground_truth": "008",
            },
        ],
    )
    monkeypatch.setattr(
        PREPARE,
        "_iter_parquet_rows",
        lambda _: iter(
            [
                {"problem_idx": 2, "problem": "Final problem 2", "answer": 12},
                {"problem_idx": 1, "problem": "Final problem 1", "answer": 11},
            ]
        ),
    )
    catalog.write_text(
        "\n".join(
            [
                'schema_version: "flowsteer.agentgraph.aime2026.dataset.v1"',
                'task_schema_version: "flowsteer.agentgraph.task.v1"',
                f'aligned_dir: "{output}"',
                "sources:",
                f'  historical_path: "{history}"',
                f'  development_path: "{development}"',
                '  final_path: "unused-*.parquet"',
                "split_policy:",
                "  training_maximum_year: 2024",
                "  development_historical_count: 1",
                "  train_count: 3",
                "  development_count: 3",
                "  final_count: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    PREPARE.prepare(catalog)

    train = list(iter_task_records(output / "train.jsonl", expected_split="train"))
    validation = list(
        iter_task_records(output / "validation.jsonl", expected_split="validation")
    )
    final = list(iter_task_records(output / "test.jsonl", expected_split="test"))
    assert len(train) == 3
    assert "aime-historical:2024:i:01" not in {task.task_id for task in train}
    assert [task.task_id for task in validation] == [
        "aime-2025:i:01",
        "aime-2025:ii:01",
        "aime-historical:2024:i:01",
    ]
    assert [task.task_id for task in final] == ["aime-2026:01", "aime-2026:02"]
    assert all(task.metadata["task_family"] == AIME2026_TASK_FAMILY for task in final)
    assert all(
        task.metadata["evaluator_version"] == AIME2026_EVALUATOR_VERSION
        for task in (*validation, *final)
    )
    assert {task.metadata["evaluation_role"] for task in validation} == {
        "development"
    }
    assert validation[-1].metadata["benchmark_slice"] == (
        "heldout_historical_aime_2000_2024"
    )
    assert {task.metadata["evaluation_role"] for task in final} == {
        "final-evaluation"
    }


def test_preparation_rejects_statement_overlap_across_splits() -> None:
    common = {
        "question": "Same normalized problem!",
        "ground_truth": "1",
    }
    train = [{**common, "task_id": "train"}]
    development = [{**common, "task_id": "development"}]
    with pytest.raises(ValueError, match="overlap"):
        PREPARE._assert_split_isolation(train, development, [])


def test_flowsteer_aime2025_degree_unit_is_target_only_normalization() -> None:
    assert PREPARE._flowsteer_development_answer(r"336^\circ") == "336"
    assert score_aime2026_integer(r"336^\circ", ["336"]).accuracy == 0.0
