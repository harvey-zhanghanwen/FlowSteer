from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.interactive.config_loader import load_yaml


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "evaluate_completion_benchmark_round.py"
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_completion_benchmark_round_runtime_registry", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_CONFIG_BY_DATASET = {
    "hotpotqa": "development_hotpotqa_tool_react_stable_zero_v4.yaml",
    "triviaqa": "evaluation_triviaqa_tool_react_stable_zero.yaml",
    "aime_2026": "development_aime2026_computation_tool_stable_zero.yaml",
    "healthbench_professional": (
        "evaluation_healthbench_professional_medrag_tool_stable_zero.yaml"
    ),
    "webshop": "evaluation_webshop_ragen_environment_stable_zero.yaml",
    "alfworld": "evaluation_alfworld_ragen_environment_stable_zero.yaml",
    "swe_bench": "evaluation_swebench_verified_coding_agent_stable_zero.yaml",
}


def _opted_in_config(dataset_key: str) -> dict:
    config = deepcopy(load_yaml(_ROOT / "config" / _CONFIG_BY_DATASET[dataset_key]))
    registry = load_yaml(_ROOT / "config" / "datasets_runtime_v2.yaml")
    entry = registry["datasets"][dataset_key]
    config["data"].update(
        {
            "registry_path": "config/datasets_runtime_v2.yaml",
            "registry_dataset_key": dataset_key,
            "train_path": entry["paths"]["train"],
            "validation_path": entry["paths"]["validation"],
            "test_path": entry["paths"]["test"],
            "task_schema_version": registry["task_schema_version"],
        }
    )
    if dataset_key == "swe_bench":
        config["evaluation"]["swebench_dataset_path"] = entry["paths"][
            "validation"
        ]
        config["evaluation"]["swebench_dataset_source"] = "regular_dev"
    return config


def test_checked_in_registry_validates_all_seven_runtime_bindings() -> None:
    for dataset_key in _CONFIG_BY_DATASET:
        receipt = _MODULE._validate_runtime_dataset_registry(
            _opted_in_config(dataset_key), _ROOT
        )

        assert receipt is not None
        assert receipt["registry_dataset_key"] == dataset_key
        assert receipt["checks"]["explicit_split_paths_match_registry"] is True
        assert receipt["checks"]["manifest_schema_matches_registry"] is True
        assert receipt["checks"][
            "manifest_preparation_provenance_matches_registry"
        ] is True
        if dataset_key == "swe_bench":
            assert receipt["checks"]["swebench_split_sources_match_manifest"] is True
        assert receipt["skipped_checks"] == []


@pytest.mark.parametrize(
    ("config_name", "dataset_key"),
    [
        ("development_aime_family_128_computation_tool_eval128.yaml", "aime_2026"),
        ("evaluation_swebench_regular_dev_coding_agent_v2.yaml", "swe_bench"),
    ],
)
def test_current_128_task_conditions_are_registry_bound(
    config_name: str,
    dataset_key: str,
) -> None:
    config = load_yaml(_ROOT / "config" / config_name)

    _MODULE.validate_completion_benchmark_config(config)
    receipt = _MODULE._validate_runtime_dataset_registry(config, _ROOT)

    assert receipt is not None
    assert receipt["registry_dataset_key"] == dataset_key
    section_name, bounded = _MODULE._evaluation_section(config)
    assert section_name
    assert bounded["sample_count"] == 128


def test_legacy_config_does_not_enable_runtime_registry_validation() -> None:
    config = load_yaml(_ROOT / "config" / "evaluation_hotpotqa_tool_react_stable_zero.yaml")

    assert _MODULE._validate_runtime_dataset_registry(config, _ROOT) is None


def test_partial_registry_opt_in_fails_closed() -> None:
    config = load_yaml(_ROOT / "config" / "evaluation_hotpotqa_tool_react_stable_zero.yaml")
    config["data"]["registry_path"] = "config/datasets_runtime_v2.yaml"

    with pytest.raises(
        _MODULE.ConfigurationError,
        match="requires non-empty data.registry_path and data.registry_dataset_key",
    ):
        _MODULE.validate_completion_benchmark_config(config)


def test_explicit_split_path_drift_fails_closed() -> None:
    config = _opted_in_config("hotpotqa")
    config["data"]["validation_path"] = "data/agentgraph_v1/validation.jsonl"

    with pytest.raises(
        _MODULE.ConfigurationError,
        match="data.validation_path differs from the runtime dataset registry",
    ):
        _MODULE._validate_runtime_dataset_registry(config, _ROOT)


def test_swebench_evaluator_path_drift_fails_closed() -> None:
    config = _opted_in_config("swe_bench")
    config["evaluation"]["swebench_dataset_path"] = "data/swebench_v2/test.jsonl"

    with pytest.raises(
        _MODULE.ConfigurationError,
        match="swebench_dataset_path differs from the selected runtime registry split",
    ):
        _MODULE._validate_runtime_dataset_registry(config, _ROOT)


def test_missing_manifest_provenance_is_recorded_without_inference(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("schema_version: test.catalog.v1\n", encoding="utf-8")
    paths = {
        split: tmp_path / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    for path in paths.values():
        path.write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "test.manifest.v1",
                "task_schema_version": "flowsteer.agentgraph.task.v1",
                "files": {split: path.name for split, path in paths.items()},
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": "flowsteer.agentgraph.runtime-datasets.v2",
                "task_schema_version": "flowsteer.agentgraph.task.v1",
                "datasets": {
                    "hotpotqa": {
                        "protocol_label": "test_protocol",
                        "preparation_catalog_path": str(catalog),
                        "preparation_catalog_schema_version": "test.catalog.v1",
                        "manifest": {
                            "path": str(manifest),
                            "schema_version": "test.manifest.v1",
                            "provenance_field": "catalog",
                        },
                        "paths": {
                            split: str(path) for split, path in paths.items()
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = _opted_in_config("hotpotqa")
    config["data"].update(
        {
            "registry_path": str(registry),
            "train_path": str(paths["train"]),
            "validation_path": str(paths["validation"]),
            "test_path": str(paths["test"]),
        }
    )

    receipt = _MODULE._validate_runtime_dataset_registry(config, _ROOT)

    assert receipt is not None
    assert receipt["checks"].get(
        "manifest_preparation_provenance_matches_registry"
    ) is None
    assert receipt["skipped_checks"] == [
        {
            "check": "manifest_preparation_provenance",
            "reason": "manifest has no non-empty 'catalog' field",
        }
    ]
