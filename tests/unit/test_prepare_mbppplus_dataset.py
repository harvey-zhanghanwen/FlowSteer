from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "prepare_mbppplus_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("prepare_mbppplus_dataset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _catalog(tmp_path: Path, *, output_name: str = "aligned") -> Path:
    value = {
        "schema_version": "flowsteer.agentgraph.mbppplus.dataset.v1",
        "task_schema_version": "flowsteer.agentgraph.task.v1",
        "aligned_dir": str(tmp_path / output_name),
        "sources": {
            "official_evalplus": {
                "dataset_version": "v0.2.0",
                "path": str(tmp_path / "MbppPlus-v0.2.0.jsonl"),
            }
        },
        "split_policy": {
            "mode": "official_evaluation_only",
            "protocol": "mbpp-plus-fixed-100@1",
            "ordering": "canonical_numeric_task_id_ascending",
            "source_count": 378,
            "test_count": 100,
            "training_enabled": False,
            "metric": "pass@1",
        },
    }
    path = tmp_path / "config" / "datasets_mbppplus_v1.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _row(index: int) -> dict[str, Any]:
    return {
        "task_id": f"Mbpp/{index}",
        "prompt": f'"""Implement function_{index}.\nassert function_{index}() == {index}\n"""\n',
        "entry_point": f"function_{index}",
        "canonical_solution": f"def function_{index}():\n    return {index}\n",
        "base_input": [[]],
        "plus_input": [[index]],
        "contract": f"CONTRACT_SENTINEL_{index}",
        "assertion": f"ASSERTION_SENTINEL_{index}",
        "atol": 0,
    }


def _source_rows() -> list[dict[str, Any]]:
    return [_row(index) for index in range(1, 379)][::-1]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fixed_100_is_numeric_id_ordered_and_evaluation_only(tmp_path: Path) -> None:
    output = _MODULE.prepare(
        _catalog(tmp_path), row_provider=lambda _path: _source_rows()
    )

    assert _read(output / "train.jsonl") == []
    assert _read(output / "validation.jsonl") == []
    public = _read(output / "test.jsonl")
    private = _read(output / "evaluator_private.jsonl")
    assert len(public) == 100
    assert len(private) == 100
    assert [row["task_id"] for row in public] == [
        f"mbpp-plus:Mbpp/{index}" for index in range(1, 101)
    ]
    assert [row["task_id"] for row in private] == [
        row["task_id"] for row in public
    ]
    assert public[0]["question"] == _row(1)["prompt"]
    assert public[0]["extra"]["entry_point"] == "function_1"
    assert public[-1]["extra"]["canonical_numeric_task_id"] == 100

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "mbpp-plus-fixed-100@1"
    assert manifest["counts_by_split"] == {
        "train": 0,
        "validation": 0,
        "test": 100,
    }
    assert manifest["selection"] == {
        "ordering": "canonical_numeric_task_id_ascending",
        "first_source_task_id": "Mbpp/1",
        "last_source_task_id": "Mbpp/100",
        "selected_count": 100,
    }
    assert manifest["training_started"] is False
    assert manifest["evaluation_started"] is False


def test_public_records_redact_all_evalplus_answer_and_test_fields(
    tmp_path: Path,
) -> None:
    output = _MODULE.prepare(
        _catalog(tmp_path), row_provider=lambda _path: _source_rows()
    )
    public = _read(output / "test.jsonl")
    first = public[0]
    assert first["ground_truth"] is None
    assert first["answer"] is None
    assert "evaluator_payload" not in first["metadata"]
    rendered = json.dumps(public, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "canonical_solution",
        "base_input",
        "plus_input",
        "contract",
        "assertion",
        "atol",
        "CONTRACT_SENTINEL_1",
        "ASSERTION_SENTINEL_1",
        "def function_1",
    ):
        assert forbidden not in rendered

    private = _read(output / "evaluator_private.jsonl")
    payload = private[0]["evaluator_payload"]
    assert payload == _row(1)
    assert private[0]["source_task_id"] == "Mbpp/1"


def test_duplicate_canonical_task_identity_is_rejected(tmp_path: Path) -> None:
    rows = _source_rows()
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError, match=r"duplicate EvalPlus MBPP\+ task identity"):
        _MODULE.prepare(_catalog(tmp_path), row_provider=lambda _path: rows)


def test_source_count_drift_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source count drift"):
        _MODULE.prepare(
            _catalog(tmp_path), row_provider=lambda _path: _source_rows()[:-1]
        )


def test_empty_mapping_plus_input_is_preserved_for_official_schema(
    tmp_path: Path,
) -> None:
    rows = _source_rows()
    selected_index = next(
        index for index, row in enumerate(rows) if row["task_id"] == "Mbpp/42"
    )
    rows[selected_index] = {**rows[selected_index], "plus_input": {}}
    output = _MODULE.prepare(_catalog(tmp_path), row_provider=lambda _path: rows)

    private = _read(output / "evaluator_private.jsonl")
    by_source_id = {row["source_task_id"]: row for row in private}
    assert by_source_id["Mbpp/42"]["evaluator_payload"]["plus_input"] == {}


def test_nonempty_mapping_plus_input_is_rejected(tmp_path: Path) -> None:
    rows = _source_rows()
    rows[0] = {**rows[0], "plus_input": {"unexpected": []}}
    with pytest.raises(ValueError, match="invalid evaluator field plus_input"):
        _MODULE.prepare(_catalog(tmp_path), row_provider=lambda _path: rows)


@pytest.mark.parametrize("task_id", ["MBPP/1", "Mbpp/0", "Mbpp/-1", "1"])
def test_noncanonical_evalplus_task_id_is_rejected(
    tmp_path: Path, task_id: str
) -> None:
    rows = _source_rows()
    rows[0] = {**rows[0], "task_id": task_id}
    with pytest.raises(ValueError, match=r"invalid canonical EvalPlus MBPP\+ task_id"):
        _MODULE.prepare(_catalog(tmp_path), row_provider=lambda _path: rows)
