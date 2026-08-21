from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import materialize_joint_qa_additional_confirmation as materializer


def _record(dataset: str, index: int) -> dict[str, object]:
    source = "HotpotQA" if dataset == "hotpotqa" else "TriviaQA"
    return {
        "schema_version": "flowsteer.agentgraph.task.v1",
        "task_id": f"{dataset}:candidate-{index:04d}",
        "question": f"question {index}",
        "ground_truth": f"answer {index}",
        "split": "train",
        "metadata": {"dataset_key": dataset, "source": source},
        "source": source,
        "dataset": dataset,
        "answer": f"answer {index}",
        "task_type": "qa",
        "context": [],
        "extra": {"source": source},
    }


def test_materializes_fresh_validation_and_combined_union(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    catalog = {
        "sources": {
            "hotpotqa": {"display_name": "HotpotQA"},
            "triviaqa": {"display_name": "TriviaQA"},
        }
    }
    (config_dir / "datasets.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
    base_ids = {
        "hotpotqa": ["hotpotqa:base"],
        "triviaqa": ["triviaqa:base"],
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    key: {"ordered_task_ids": {"train": ids}}
                    for key, ids in base_ids.items()
                }
            }
        ),
        encoding="utf-8",
    )
    base_rows = [
        {"task_id": "hotpotqa:old-confirmation"},
        {"task_id": "triviaqa:old-confirmation"},
    ]
    (data_dir / "base.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
    )
    config = {
        "schema_version": materializer.SCHEMA_VERSION,
        "dataset_catalog_path": "config/datasets.yaml",
        "base_manifest_path": "data/manifest.json",
        "base_confirmation_path": "data/base.jsonl",
        "output_path": "data/new.jsonl",
        "combined_output_path": "data/all.jsonl",
        "manifest_path": "data/new_manifest.json",
        "datasets": ["hotpotqa", "triviaqa"],
        "partition": "skill_confirmation_round4",
        "task_split": "validation",
        "candidate_range": {"start": 4, "stop": 7},
    }
    config_path = config_dir / "round4.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        materializer,
        "CONVERTERS",
        {
            dataset: (lambda _source, dataset=dataset: (_record(dataset, i) for i in range(10)))
            for dataset in ("hotpotqa", "triviaqa")
        },
    )

    manifest_path = materializer.materialize(config_path)
    records = [json.loads(line) for line in (data_dir / "new.jsonl").read_text().splitlines()]
    combined = [json.loads(line) for line in (data_dir / "all.jsonl").read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert len(records) == 6
    assert len(combined) == 8
    assert all(record["split"] == "validation" for record in records)
    assert all(
        record["metadata"]["joint_qa_partition"] == "skill_confirmation_round4"
        for record in records
    )
    assert manifest["candidate_range"] == {"start": 4, "stop": 7}
    assert manifest["excluded_from_grpo_and_reported_metrics"] is True
    assert manifest["final_test_untouched"] is True


def test_materializes_hotpotqa_only_confirmation(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    (config_dir / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": {
                    "hotpotqa": {"display_name": "HotpotQA"},
                    "triviaqa": {"display_name": "TriviaQA"},
                }
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "hotpotqa": {
                        "ordered_task_ids": {"test": ["hotpotqa:frozen-test"]}
                    },
                    "triviaqa": {
                        "ordered_task_ids": {"test": ["triviaqa:frozen-test"]}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    base_rows = [
        {"task_id": "hotpotqa:round7"},
        {"task_id": "triviaqa:round7"},
    ]
    (data_dir / "base.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in base_rows), encoding="utf-8"
    )
    config = {
        "schema_version": materializer.SCHEMA_VERSION,
        "dataset_catalog_path": "config/datasets.yaml",
        "base_manifest_path": "data/manifest.json",
        "base_confirmation_path": "data/base.jsonl",
        "output_path": "data/new.jsonl",
        "combined_output_path": "data/all.jsonl",
        "manifest_path": "data/new_manifest.json",
        "datasets": ["hotpotqa"],
        "partition": "skill_confirmation_round8",
        "task_split": "validation",
        "candidate_range": {"start": 4, "stop": 7},
    }
    config_path = config_dir / "round8.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        materializer,
        "CONVERTERS",
        {"hotpotqa": lambda _source: (_record("hotpotqa", i) for i in range(10))},
    )

    manifest_path = materializer.materialize(config_path)
    records = [json.loads(line) for line in (data_dir / "new.jsonl").read_text().splitlines()]
    combined = [json.loads(line) for line in (data_dir / "all.jsonl").read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())

    assert [record["task_id"] for record in records] == [
        "hotpotqa:candidate-0004",
        "hotpotqa:candidate-0005",
        "hotpotqa:candidate-0006",
    ]
    assert len(combined) == len(base_rows) + len(records)
    assert set(manifest["ordered_task_ids"]) == {"hotpotqa"}
    assert manifest["count_per_dataset"] == 3
    assert manifest["task_split"] == "validation"
    assert manifest["excluded_from_grpo_and_reported_metrics"] is True
    assert manifest["final_test_untouched"] is True


def test_rejects_empty_duplicate_and_unsupported_dataset_scopes(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    base_config = {
        "schema_version": materializer.SCHEMA_VERSION,
        "datasets": ["hotpotqa"],
        "task_split": "validation",
        "partition": "skill_confirmation_round8",
    }

    for datasets, expected_error in (
        ([], "non-empty list"),
        (["hotpotqa", "hotpotqa"], "must be unique"),
        (["aime_2026"], "unsupported additional-confirmation datasets"),
    ):
        config = {**base_config, "datasets": datasets}
        config_path = config_dir / f"invalid-{len(list(config_dir.iterdir()))}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        try:
            materializer.materialize(config_path)
        except ValueError as exc:
            assert expected_error in str(exc)
        else:
            raise AssertionError("invalid dataset scope was accepted")


def test_round8_config_is_hotpotqa_only_and_versioned() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "config/joint_qa_round8_confirmation.yaml").read_text(encoding="utf-8")
    )

    assert config["datasets"] == ["hotpotqa"]
    assert config["task_split"] == "validation"
    assert config["partition"] == "skill_confirmation_round8"
    assert config["candidate_range"] == {"start": 1024, "stop": 1064}
    assert config["base_confirmation_path"].endswith(
        "skill_confirmation_all_round7.jsonl"
    )
    assert config["output_path"].endswith("skill_confirmation_round8.jsonl")
    assert config["combined_output_path"].endswith(
        "skill_confirmation_all_round8.jsonl"
    )
    assert config["manifest_path"].endswith(
        "skill_confirmation_round8_manifest.json"
    )
