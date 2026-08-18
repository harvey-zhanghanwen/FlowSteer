from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import prepare_joint_qa_partitions as module


def _candidate_stream(dataset_key: str, count: int = 864):
    for index in range(count):
        yield {
            "schema_version": "flowsteer.agentgraph.task.v1",
            "task_id": f"{dataset_key}:candidate-{index:04d}",
            "question": f"question {index}",
            "ground_truth": "answer",
            "split": "train",
            "metadata": {"dataset_key": dataset_key},
            "extra": {},
        }


def _write_config(root: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    catalog_path = config_dir / "datasets.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "sources": {
                    "hotpotqa": {"source": "synthetic"},
                    "triviaqa": {"source": "synthetic"},
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = config_dir / "partitions.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": module.PARTITION_SCHEMA_VERSION,
                "dataset_catalog_path": "config/datasets.yaml",
                "output_dir": "data/joint_qa_v2",
                "datasets": ["hotpotqa", "triviaqa"],
                "partitions": {
                    "development": {"start": 0, "stop": 128, "task_split": "validation"},
                    "train": {"start": 128, "stop": 640, "task_split": "train"},
                    "quarantine": {"start": 640, "stop": 672, "task_split": None},
                    "skill_confirmation": {"start": 672, "stop": 736, "task_split": "validation"},
                    "test": {"start": 736, "stop": 864, "task_split": "test"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_prepare_freezes_ordered_non_overlapping_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setitem(
        module.CONVERTERS,
        "hotpotqa",
        lambda _config: _candidate_stream("hotpotqa"),
    )
    monkeypatch.setitem(
        module.CONVERTERS,
        "triviaqa",
        lambda _config: _candidate_stream("triviaqa"),
    )

    output = module.prepare(config_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert not (output / "quarantine.jsonl").exists()
    assert {
        name: len((output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines())
        for name in module.WRITTEN_PARTITIONS
    } == {
        "development": 256,
        "train": 1024,
        "skill_confirmation": 128,
        "test": 256,
    }
    hotpot = manifest["datasets"]["hotpotqa"]["ordered_task_ids"]
    assert hotpot["development"][0] == "hotpotqa:candidate-0000"
    assert hotpot["train"][0] == "hotpotqa:candidate-0128"
    assert hotpot["quarantine"] == [
        f"hotpotqa:candidate-{index:04d}" for index in range(640, 672)
    ]
    assert hotpot["skill_confirmation"][0] == "hotpotqa:candidate-0672"
    assert hotpot["test"][-1] == "hotpotqa:candidate-0863"

    records = [
        json.loads(line)
        for line in (output / "skill_confirmation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["split"] == "validation"
    assert records[0]["metadata"]["joint_qa_partition"] == "skill_confirmation"
    assert records[0]["metadata"]["native_candidate_position"] == 672

    partition_ids = [
        set(manifest["datasets"][dataset]["ordered_task_ids"][partition])
        for dataset in ("hotpotqa", "triviaqa")
        for partition in manifest["partitions"]
    ]
    for index, left in enumerate(partition_ids):
        assert all(left.isdisjoint(right) for right in partition_ids[index + 1 :])


def test_prepare_rejects_candidate_stream_shorter_than_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setitem(
        module.CONVERTERS,
        "hotpotqa",
        lambda _config: _candidate_stream("hotpotqa", 863),
    )
    monkeypatch.setitem(
        module.CONVERTERS,
        "triviaqa",
        lambda _config: _candidate_stream("triviaqa"),
    )
    with pytest.raises(ValueError, match="ended at 863 before 864"):
        module.prepare(config_path)
