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
    / "prepare_swebench_skillflow_v3_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "prepare_swebench_skillflow_v3_dataset", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _catalog(tmp_path: Path, output_name: str = "aligned") -> Path:
    value = {
        "schema_version": (
            "flowsteer.agentgraph.swebench.skillflow-v3.dataset.v1"
        ),
        "task_schema_version": "flowsteer.agentgraph.task.v1",
        "aligned_dir": str(tmp_path / output_name),
        "sources": {
            "skillflow_v3": {
                "path": str(tmp_path / "skillflow"),
                "train_file": "train_v3.json",
                "iid_test_file": "test_iid_v3.json",
            },
            "verified": {"path": str(tmp_path / "verified")},
        },
        "split_policy": {
            "filter_task_type": "code_generation",
            "selection": "preserve_source_order",
            "train_source": "skillflow_train_v3",
            "train_count": 500,
            "train_unique_instance_ids": 372,
            "train_repeated_rows": 128,
            "validation_population": "none",
            "test_source": "skillflow_test_iid_v3",
            "test_count": 128,
            "test_unique_instance_ids": 128,
            "require_train_test_disjoint_instance_ids": True,
            "evaluator_join": "official_verified_by_instance_id",
        },
    }
    path = tmp_path / f"{output_name}.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _source_row(instance_id: str) -> dict[str, Any]:
    repo = instance_id.split("__", maxsplit=1)[0] + "/repo"
    patch = f"diff --git a/{instance_id}.py b/{instance_id}.py\n+fixed\n"
    return {
        "question": f"Fix the following software issue:\n\nIssue {instance_id}",
        "answer": patch,
        "task_type": "code_generation",
        "context": [],
        "code_files": {f"{instance_id}.py": "before\n"},
        "extra": {
            "repo": repo,
            "instance_id": instance_id,
            "base_commit": f"base-{instance_id}",
            "source": "SWE-bench",
            "metric": "resolved_rate",
            "subset": "SWE-bench Verified",
        },
    }


def _official_row(source: dict[str, Any]) -> dict[str, Any]:
    extra = source["extra"]
    instance_id = extra["instance_id"]
    return {
        "instance_id": instance_id,
        "repo": extra["repo"],
        "base_commit": extra["base_commit"],
        "patch": source["answer"],
        "test_patch": f"diff --git a/test_{instance_id}.py b/test_{instance_id}.py",
        "problem_statement": f"Issue {instance_id}",
        "version": "1.0",
        "FAIL_TO_PASS": json.dumps([f"test_{instance_id}"]),
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": f"environment-{instance_id}",
    }


def _protocol_rows() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    train_unique = [_source_row(f"train__repo-{index:04d}") for index in range(372)]
    train = train_unique + [dict(row) for row in train_unique[:128]]
    test = [_source_row(f"test__repo-{index:04d}") for index in range(128)]
    official = [_official_row(row) for row in (*train_unique, *test)]
    return train, test, official


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_skillflow_v3_counts_repeats_and_iid_isolation(tmp_path: Path) -> None:
    train_rows, test_rows, official_rows = _protocol_rows()
    output = _MODULE.prepare(
        _catalog(tmp_path),
        train_provider=lambda _path: train_rows,
        iid_test_provider=lambda _path: test_rows,
        verified_provider=lambda _path: official_rows,
    )

    train = _read(output / "train.jsonl")
    validation = _read(output / "validation.jsonl")
    test = _read(output / "test.jsonl")
    assert len(train) == 500
    assert validation == []
    assert len(test) == 128
    assert len({row["task_id"] for row in train}) == 500
    train_ids = [row["extra"]["instance_id"] for row in train]
    test_ids = [row["extra"]["instance_id"] for row in test]
    assert len(set(train_ids)) == 372
    assert sum(count - 1 for count in __import__("collections").Counter(train_ids).values()) == 128
    assert len(set(test_ids)) == 128
    assert set(train_ids).isdisjoint(test_ids)
    assert train[0]["question"] == train_rows[0]["question"]
    assert train[372]["extra"]["occurrence_index"] == 1

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts_by_split"] == {
        "train": 500,
        "validation": 0,
        "test": 128,
    }
    assert manifest["instance_id_protocol"] == {
        "status": "train_repeats_preserved_test_unique_train_test_disjoint",
        "train_rows": 500,
        "train_unique_instance_ids": 372,
        "train_repeated_rows": 128,
        "test_rows": 128,
        "test_unique_instance_ids": 128,
        "train_test_overlap": 0,
    }


def test_gold_and_test_truth_are_not_model_visible(tmp_path: Path) -> None:
    train_rows, test_rows, official_rows = _protocol_rows()
    output = _MODULE.prepare(
        _catalog(tmp_path),
        train_provider=lambda _path: train_rows,
        iid_test_provider=lambda _path: test_rows,
        verified_provider=lambda _path: official_rows,
    )
    record = _read(output / "test.jsonl")[0]
    official = official_rows[372]
    model_visible = json.dumps(
        {
            "question": record["question"],
            "context": record["context"],
            "extra": record["extra"],
            "code_files": record["code_files"],
        },
        sort_keys=True,
    )
    assert official["patch"] not in model_visible
    assert official["test_patch"] not in model_visible
    assert "FAIL_TO_PASS" not in model_visible
    assert "PASS_TO_PASS" not in model_visible
    assert record["code_files"] == {}
    assert record["ground_truth"] == official["patch"]
    assert record["answer"] == official["patch"]
    payload = record["metadata"]["evaluator_payload"]
    assert payload["test_patch"] == official["test_patch"]
    assert payload["FAIL_TO_PASS"] == official["FAIL_TO_PASS"]
    assert payload["PASS_TO_PASS"] == official["PASS_TO_PASS"]


def test_duplicate_instance_is_rejected_from_iid_test(tmp_path: Path) -> None:
    train_rows, test_rows, official_rows = _protocol_rows()
    test_rows[-1] = dict(test_rows[0])

    with pytest.raises(
        ValueError, match="skillflow_test_iid_v3 unique instance count drift"
    ):
        _MODULE.prepare(
            _catalog(tmp_path),
            train_provider=lambda _path: train_rows,
            iid_test_provider=lambda _path: test_rows,
            verified_provider=lambda _path: official_rows,
        )


def test_train_test_instance_overlap_is_rejected(tmp_path: Path) -> None:
    train_rows, test_rows, official_rows = _protocol_rows()
    replacement = train_rows[0]
    test_rows[0] = dict(replacement)

    with pytest.raises(ValueError, match="train/test instance_id overlap"):
        _MODULE.prepare(
            _catalog(tmp_path),
            train_provider=lambda _path: train_rows,
            iid_test_provider=lambda _path: test_rows,
            verified_provider=lambda _path: official_rows,
        )


def test_conflicting_train_repeat_is_rejected(tmp_path: Path) -> None:
    train_rows, test_rows, official_rows = _protocol_rows()
    conflicting = dict(train_rows[372])
    conflicting["question"] = conflicting["question"] + " changed"
    train_rows[372] = conflicting

    with pytest.raises(ValueError, match="repeated instance has conflicting rows"):
        _MODULE.prepare(
            _catalog(tmp_path),
            train_provider=lambda _path: train_rows,
            iid_test_provider=lambda _path: test_rows,
            verified_provider=lambda _path: official_rows,
        )
