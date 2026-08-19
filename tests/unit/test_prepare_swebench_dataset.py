from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml


_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "prepare_swebench_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("prepare_swebench_dataset", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _catalog(tmp_path: Path, output_name: str = "aligned") -> Path:
    value = {
        "schema_version": "flowsteer.agentgraph.swebench.dataset.v1",
        "task_schema_version": "flowsteer.agentgraph.task.v1",
        "aligned_dir": str(tmp_path / output_name),
        "sources": {
            "regular": {
                "path": str(tmp_path / "regular"),
                "train_file": "train-*.parquet",
                "development_file": "dev-*.parquet",
            },
            "verified": {"path": str(tmp_path / "verified")},
        },
        "split_policy": {
            "selection": "sequential",
            "train_count": 512,
            "validation_count": 128,
            "test_population": "complete_verified",
            "train_source": "regular_train",
            "validation_source": "regular_dev",
            "test_source": "verified",
            "require_pairwise_disjoint_instance_ids": True,
        },
    }
    path = tmp_path / f"{output_name}.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _rows(prefix: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": f"{prefix}__repo-{index:04d}",
            "repo": f"{prefix}/repo",
            "base_commit": f"base-{index:04d}",
            "patch": f"gold patch {prefix} {index}",
            "test_patch": f"test patch {prefix} {index}",
            "problem_statement": f"problem {prefix} {index}",
            "version": "1.0",
            "FAIL_TO_PASS": [f"test_{index}"],
            "PASS_TO_PASS": [],
            "environment_setup_commit": f"environment-{index:04d}",
        }
        for index in range(count)
    ]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_regular_train_dev_and_complete_verified_are_isolated(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    train_rows = _rows("train", 515)
    dev_rows = _rows("dev", 130)
    verified_rows = _rows("verified", 3)

    output = _MODULE.prepare(
        catalog,
        train_provider=lambda _path: train_rows,
        development_provider=lambda _path: dev_rows,
        verified_provider=lambda _path: verified_rows,
    )

    train = _read(output / "train.jsonl")
    validation = _read(output / "validation.jsonl")
    test = _read(output / "test.jsonl")
    assert len(train) == 512
    assert len(validation) == 128
    assert len(test) == 3
    assert train[0]["task_id"] == "swe-bench:train__repo-0000"
    assert validation[0]["task_id"] == "swe-bench:dev__repo-0000"
    assert test[0]["task_id"] == "swe-bench:verified__repo-0000"
    assert train[-1]["task_id"] == "swe-bench:train__repo-0511"
    assert validation[-1]["task_id"] == "swe-bench:dev__repo-0127"
    assert [
        row["metadata"]["dataset_source"] for row in (train[0], validation[0], test[0])
    ] == [
        "regular_train",
        "regular_dev",
        "verified",
    ]

    first = validation[0]
    gold_patch = dev_rows[0]["patch"]
    assert gold_patch not in first["question"]
    assert gold_patch not in json.dumps(first["extra"], sort_keys=True)
    assert gold_patch not in json.dumps(first["context"], sort_keys=True)
    assert "patch" not in first["metadata"]["evaluator_payload"]
    assert first["ground_truth"] == gold_patch
    assert (
        first["metadata"]["evaluator_payload"]["test_patch"]
        == dev_rows[0]["test_patch"]
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts_by_split"] == {
        "train": 512,
        "validation": 128,
        "test": 3,
    }
    assert manifest["instance_id_isolation"] == {
        "status": "pairwise_disjoint",
        "groups": {"regular_train": 512, "regular_dev": 128, "verified": 3},
    }


def test_instance_overlap_between_any_source_groups_fails_closed(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    train_rows = _rows("train", 512)
    dev_rows = _rows("dev", 128)
    verified_rows = _rows("verified", 2)
    verified_rows[0]["instance_id"] = dev_rows[0]["instance_id"]

    with pytest.raises(ValueError, match="overlap between regular_dev and verified"):
        _MODULE.prepare(
            catalog,
            train_provider=lambda _path: train_rows,
            development_provider=lambda _path: dev_rows,
            verified_provider=lambda _path: verified_rows,
        )


def test_short_regular_population_fails_instead_of_cycling(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    with pytest.raises(ValueError, match="regular_train provides 511 rows"):
        _MODULE.prepare(
            catalog,
            train_provider=lambda _path: _rows("train", 511),
            development_provider=lambda _path: _rows("dev", 128),
            verified_provider=lambda _path: _rows("verified", 2),
        )
