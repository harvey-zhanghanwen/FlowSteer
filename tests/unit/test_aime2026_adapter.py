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


def test_integer_scorer_matches_skillev_private_static_rule() -> None:
    assert score_aime2026_integer("42", ["42"]).accuracy == 1.0
    assert score_aime2026_integer("042", ["42"]).accuracy == 1.0
    assert score_aime2026_integer("+42", ["42"]).accuracy == 1.0
    assert score_aime2026_integer("42.0", ["42"]).accuracy == 0.0
    assert score_aime2026_integer("The answer is 42", ["42"]).accuracy == 0.0
    assert score_aime2026_integer("100%", ["1"]).accuracy == 0.0


def test_single_terminal_boundary_maps_to_private_integer_submission() -> None:
    result = score_aime2026_integer(
        "scratch 999 <answer>042</answer>",
        ["42"],
    )
    assert result.accuracy == 1.0
    assert result.scored_prediction == "042"
    assert result.structured_answer_extracted is True
    assert result.parsing_succeeded is True
    assert result.canonical_prediction == "42"


@pytest.mark.parametrize(
    ("prediction", "reason"),
    [
        (
            "<answer>41</answer><answer>42</answer>",
            "multiple_answer_boundaries",
        ),
        ("<answer>42</answer><answer>", "malformed_answer_boundary"),
        ("</answer><answer>42</answer>", "malformed_answer_boundary"),
        ("<answer>42", "malformed_answer_boundary"),
        ("Thus 42", "integer_conversion_failed"),
    ],
)
def test_prediction_parser_fails_closed_without_selecting_an_answer(
    prediction: str, reason: str
) -> None:
    result = score_aime2026_integer(prediction, ["42"])
    assert result.accuracy == 0.0
    assert result.parsing_succeeded is False
    assert result.parsing_failure_reason == reason
    assert result.canonical_prediction is None


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
        "_read_official_parquet_rows",
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
    assert [task.task_id for task in final] == ["aime-2026/02", "aime-2026/01"]
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
    assert [task.metadata["problem_index"] for task in final] == [2, 1]
    assert {task.metadata["source_split"] for task in final} == {"train"}
    assert {task.metadata["benchmark_id"] for task in final} == {"aime-2026"}


def test_official_converter_rejects_any_nonproduction_row_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PREPARE,
        "_read_official_parquet_rows",
        lambda _: iter(
            [
                {
                    "problem_idx": 1,
                    "problem": "Problem",
                    "answer": 42,
                    "solution": "not public",
                }
            ]
        ),
    )
    with pytest.raises(ValueError, match="fields differ"):
        PREPARE._final_records(Path("unused.parquet"))


def test_official_converter_preserves_exact_30_row_population_and_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"problem_idx": index, "problem": f"Problem {index}", "answer": index}
        for index in range(1, 31)
    ]
    monkeypatch.setattr(
        PREPARE, "_read_official_parquet_rows", lambda _: tuple(rows)
    )

    records = PREPARE._final_records(Path("unused.parquet"))

    assert len(records) == 30
    assert [record["task_id"] for record in records] == [
        f"aime-2026/{index:02d}" for index in range(1, 31)
    ]
    assert len({record["task_id"] for record in records}) == 30
    assert len({record["question"] for record in records}) == 30
    assert [record["metadata"]["problem_index"] for record in records] == list(
        range(1, 31)
    )


def test_official_converter_preserves_problem_text_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_problem = "  Preserve leading and trailing whitespace.\n"
    monkeypatch.setattr(
        PREPARE,
        "_read_official_parquet_rows",
        lambda _: (
            {"problem_idx": 1, "problem": source_problem, "answer": 1},
        ),
    )

    record = PREPARE._final_records(Path("unused.parquet"))[0]

    assert record["question"] == source_problem


def test_official_only_catalog_materializes_no_train_or_development_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "official"
    catalog = tmp_path / "config" / "datasets_aime2026_official_v1.yaml"
    catalog.parent.mkdir()
    rows = tuple(
        {"problem_idx": index, "problem": f"Problem {index}", "answer": index}
        for index in range(1, 31)
    )
    monkeypatch.setattr(PREPARE, "_read_official_parquet_rows", lambda _: rows)
    catalog.write_text(
        "\n".join(
            [
                'schema_version: "flowsteer.agentgraph.aime2026.dataset.v1"',
                'task_schema_version: "flowsteer.agentgraph.task.v1"',
                f'aligned_dir: "{output}"',
                "sources:",
                '  final_path: "/unused/train-00000-of-00001.parquet"',
                '  dataset_revision: "fixed-revision"',
                "split_policy:",
                '  mode: "official_evaluation_only"',
                "  final_count: 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    PREPARE.prepare(catalog)

    assert list(iter_task_records(output / "train.jsonl", expected_split="train")) == []
    assert list(
        iter_task_records(output / "validation.jsonl", expected_split="validation")
    ) == []
    final = list(iter_task_records(output / "test.jsonl", expected_split="test"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(final) == 30
    assert manifest["counts_by_split"] == {"train": 0, "validation": 0, "test": 30}
    assert manifest["dataset_revision"] == "fixed-revision"
    assert manifest["dataset_mode"] == "official_evaluation_only"


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"problem_idx": 1, "problem": "A", "answer": 1},
            {"problem_idx": 1, "problem": "B", "answer": 2},
        ],
        [{"problem_idx": 0, "problem": "A", "answer": 1}],
        [{"problem_idx": 31, "problem": "A", "answer": 1}],
        [{"problem_idx": "1", "problem": "A", "answer": 1}],
        [{"problem_idx": 1, "problem": "A", "answer": "1"}],
        [{"problem_idx": 1, "problem": "", "answer": 1}],
    ],
)
def test_official_converter_rejects_invalid_identity_or_scalar_types(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict],
) -> None:
    monkeypatch.setattr(
        PREPARE, "_read_official_parquet_rows", lambda _: tuple(rows)
    )
    with pytest.raises(ValueError):
        PREPARE._final_records(Path("unused.parquet"))


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
