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

