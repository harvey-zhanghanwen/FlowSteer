#!/usr/bin/env python3
"""Run one fixed completion or interactive benchmark evaluation round.

This runner is evaluation-only.  It keeps the frozen task, resume, Canvas,
trajectory-receipt, and Stable Zero boundaries from
``evaluate_hotpotqa_round.py`` and uses ``LiveSmokeBackend`` for both the
Director and AgentGraph execution.  It never trains, performs a backward pass,
updates an optimizer, or publishes model/Skill state.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import evaluate_hotpotqa_round as hotpot_round
from scripts.prompts.prompt import ANSWER_GENERATION_PROMPT
from scripts.formatter import XmlFormatter
from scripts.operator_analysis import AnswerGenerateOp
from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _workflow_problem,
    _write_json,
    evaluator_version_for,
)
from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import (
    AgentCallRecord,
    AgentRequest,
    ExecutionPhase,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.graph_diagnostics import (
    aggregate_trajectory_diagnostics,
    diagnose_trajectory,
)
from src.interactive.records import TaskRecord
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.swebench_adapter import OfficialSWEbenchHarness
from src.interactive.task_dataset import TASK_SCHEMA_VERSION, iter_task_records
from src.interactive.task_evaluator import evaluate_task


# These are aliases, rather than local reimplementations, so the completion
# benchmarks retain the HotpotQA runner's checkpoint and receipt boundaries.
_atomic_jsonl = hotpot_round._atomic_jsonl
_by_task = hotpot_round._by_task
_collect_graph = hotpot_round._collect_graph
_git_state = hotpot_round._git_state
_graph_telemetry = hotpot_round._graph_telemetry
_output_inbox = hotpot_round._output_inbox
_read_jsonl = hotpot_round._read_jsonl
_stable_zero_check = hotpot_round._stable_zero_check
_trajectory_resume_matches = hotpot_round._trajectory_resume_matches


class CompletionBenchmarkRoundError(RuntimeError):
    """The fixed benchmark evaluation protocol could not complete."""


_BENCHMARKS: Mapping[str, Mapping[str, Any]] = {
    "hotpotqa": {
        "label": "HotpotQA",
        "section_names": ("hotpotqa_evaluation",),
        "phase_names": ("hotpotqa_evaluation",),
        "primary_metric": "exact_match",
        "metric_names": ("exact_match", "token_f1"),
    },
    "triviaqa": {
        "label": "TriviaQA",
        "section_names": ("triviaqa_evaluation",),
        "phase_names": ("triviaqa_evaluation",),
        "primary_metric": "exact_match",
        "metric_names": ("exact_match", "token_f1"),
    },
    "aime_2026": {
        "label": "AIME 2026",
        "section_names": ("aime2026_evaluation", "aime_2026_evaluation"),
        "phase_names": ("aime2026_evaluation", "aime_2026_evaluation"),
        "primary_metric": "accuracy",
        "metric_names": ("accuracy",),
    },
    "healthbench_professional": {
        "label": "HealthBench Professional",
        "section_names": (
            "healthbench_evaluation",
            "healthbench_professional_evaluation",
        ),
        "phase_names": (
            "healthbench_evaluation",
            "healthbench_professional_evaluation",
        ),
        "primary_metric": "raw_score",
        "metric_names": ("raw_score",),
    },
    "webshop": {
        "label": "WebShop",
        "section_names": ("webshop_evaluation",),
        "phase_names": ("webshop_evaluation",),
        "primary_metric": "average_score",
        # ``success`` is retained as a receipt-compatible alias for older
        # consumers; Average Score and Success Rate are the two native
        # WebShop metrics reported by SkillFlow and the upstream benchmark.
        "metric_names": ("average_score", "success_rate", "success"),
    },
    "alfworld": {
        "label": "ALFWorld",
        "section_names": ("alfworld_evaluation",),
        "phase_names": ("alfworld_evaluation",),
        "primary_metric": "success",
        "metric_names": ("success",),
    },
    "swe_bench": {
        "label": "SWE-bench Verified",
        "section_names": ("swebench_evaluation",),
        "phase_names": ("swebench_evaluation",),
        "primary_metric": "resolved",
        "metric_names": ("resolved",),
    },
}

_INTERACTIVE_BENCHMARKS = frozenset({"webshop", "alfworld"})
_RUNTIME_DATASET_REGISTRY_SCHEMA = "flowsteer.agentgraph.runtime-datasets.v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _evaluation_section(
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    """Return the one declared dataset-specific evaluation section."""

    known_names = {
        name
        for specification in _BENCHMARKS.values()
        for name in specification["section_names"]
    }
    present = [name for name in sorted(known_names) if name in config]
    if len(present) != 1:
        raise ConfigurationError(
            "exactly one supported dataset-specific evaluation section "
            "must be configured"
        )
    section_name = present[0]
    bounded = _mapping(config[section_name], section_name)
    dataset_key = str(bounded.get("dataset_key", "")).strip()
    specification = _BENCHMARKS.get(dataset_key)
    if specification is None or section_name not in specification["section_names"]:
        raise ConfigurationError(
            f"{section_name}.dataset_key is not compatible with the section name"
        )
    return section_name, bounded


def _skill_evaluation_mode(config: Mapping[str, Any]) -> bool:
    skills = _mapping(config.get("skills"), "skills")
    deployment = _mapping(config.get("deployment"), "deployment")
    if skills.get("enabled") is False:
        return True
    return bool(
        skills.get("enabled") is True
        and isinstance(skills.get("store_path"), str)
        and str(skills.get("store_path", "")).strip()
        and type(skills.get("retrieval_top_k")) is int
        and int(skills["retrieval_top_k"]) > 0
        and type(skills.get("current_epoch")) is int
        and int(skills["current_epoch"]) >= 1
        and deployment.get("exploration_beta") == 0.0
        and deployment.get("allow_forced_probes") is False
        and deployment.get("active_skills_only") is True
        and deployment.get("require_version_compatible_skills") is True
    )


def validate_completion_benchmark_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless the config describes a bounded evaluation-only run."""

    validate_agent_graph_config(config)
    section_name, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    specification = _BENCHMARKS[dataset_key]
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    director = _mapping(config.get("director"), "director")
    grpo = _mapping(config.get("grpo"), "grpo")
    exploration = _mapping(config.get("exploration"), "exploration")
    gpu = _mapping(config.get("gpu"), "gpu")
    checks = {
        "experiment.phase": experiment.get("phase")
        in specification["phase_names"],
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "split": bounded.get("split") in {"train", "validation", "test"},
        "selection": bounded.get("selection") in {"sequential", "task_ids"},
        "rollouts_per_task": bounded.get("rollouts_per_task") == 1,
        "direct_model_id": bounded.get("direct_model_id") == "qwen3.5-9b-local",
        "direct_protocol": bool(str(bounded.get("direct_protocol", "")).strip()),
        "direct_contract": bool(str(bounded.get("direct_contract", "")).strip()),
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "director.execute_on_edit": director.get("execute_on_edit") is True,
        "grpo.enabled": grpo.get("enabled") is False,
        "optimizer_passes": grpo.get("optimization_passes_per_rollout_batch") == 0,
        "optimizer_updates": grpo.get("max_optimizer_updates") == 0,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.evaluation_mode": _skill_evaluation_mode(config),
        "gpu.training_enabled": gpu.get("training_enabled") is False,
    }
    stage = bounded.get("stage")
    if stage is not None:
        checks["evaluation.stage"] = stage in {
            "development",
            "evaluation",
            "final_evaluation",
        }
        if stage == "final_evaluation":
            checks["evaluation.final_split"] = bounded.get("split") == "test"
    required_partition = bounded.get("required_partition")
    if required_partition is not None:
        checks["evaluation.required_partition"] = bool(
            isinstance(required_partition, str) and required_partition.strip()
        )
    if dataset_key == "healthbench_professional":
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        checks["evaluation.healthbench_judge_model"] = bool(
            str(evaluation.get("healthbench_judge_model", "")).strip()
        )
        checks["evaluation.healthbench_judge_catalog_path"] = bool(
            str(evaluation.get("healthbench_judge_catalog_path", "")).strip()
        )
    if dataset_key in _INTERACTIVE_BENCHMARKS:
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        per_source = evaluation.get("max_environment_steps_by_source")
        checks["evaluation.max_environment_steps_by_source"] = bool(
            isinstance(per_source, Mapping)
            and type(per_source.get(dataset_key)) is int
            and int(per_source[dataset_key]) > 0
        )
    if dataset_key == "swe_bench":
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        checks["evaluation.swebench_harness_enabled"] = (
            evaluation.get("swebench_harness_enabled") is True
        )
        for field_name in (
            "swebench_evaluator_path",
            "swebench_harness_path",
            "swebench_dataset_path",
            "swebench_evaluation_root",
            "swebench_docker_namespace",
        ):
            checks[f"evaluation.{field_name}"] = bool(
                str(evaluation.get(field_name, "")).strip()
            )
        timeout = evaluation.get("swebench_timeout_seconds")
        checks["evaluation.swebench_timeout_seconds"] = bool(
            type(timeout) is int and timeout > 0
        )
        dataset_source = evaluation.get("swebench_dataset_source")
        checks["evaluation.swebench_dataset_source"] = dataset_source in {
            "regular_dev",
            "verified",
        }
        expected_split = {
            "regular_dev": "validation",
            "verified": "test",
        }.get(dataset_source)
        checks["swe_bench.dataset_source_split_isolation"] = (
            expected_split is not None and bounded.get("split") == expected_split
        )
        swe_runtime = config.get("swe_coding_runtime")
        if isinstance(swe_runtime, Mapping) and swe_runtime.get("enabled") is True:
            checks["swe_bench.direct_execution_mode"] = (
                bounded.get("direct_execution_mode") == "coding"
            )
            checks["swe_bench.direct_completion_condition"] = bool(
                str(bounded.get("direct_completion_condition", "")).strip()
            )
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            f"{section_name} violates fixed evaluation bounds: " + ", ".join(failed)
        )

    sample_count = bounded.get("sample_count")
    official_aime = dataset_key == "aime_2026" and (
        bounded.get("official_2026_only") is True
        or bounded.get("benchmark_slice") == "official_aime_2026"
    )
    maximum = 30 if official_aime else 128
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= maximum
    ):
        raise ConfigurationError(
            f"{section_name}.sample_count must be between 1 and {maximum}"
        )
    stable_zero_sample_count = bounded.get(
        "stable_zero_sample_count", min(2, sample_count)
    )
    if (
        isinstance(stable_zero_sample_count, bool)
        or not isinstance(stable_zero_sample_count, int)
        or not 1 <= stable_zero_sample_count <= sample_count
    ):
        raise ConfigurationError(
            f"{section_name}.stable_zero_sample_count must be between 1 and sample_count"
        )
    concurrency = bounded.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError(f"{section_name}.concurrency must be positive")
    task_timeout_seconds = bounded.get("task_timeout_seconds")
    if task_timeout_seconds is not None and (
        isinstance(task_timeout_seconds, bool)
        or not isinstance(task_timeout_seconds, (int, float))
        or task_timeout_seconds <= 0
    ):
        raise ConfigurationError(
            f"{section_name}.task_timeout_seconds must be positive when supplied"
        )
    if bounded.get("selection") == "task_ids":
        task_ids = bounded.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or len(task_ids) != sample_count
            or not all(isinstance(item, str) and item.strip() for item in task_ids)
            or len(set(task_ids)) != len(task_ids)
        ):
            raise ConfigurationError(
                "task_ids selection requires sample_count unique non-empty task IDs"
            )
    if bounded.get("official_2026_only") is True and dataset_key != "aime_2026":
        raise ConfigurationError("official_2026_only is valid only for AIME 2026")
    registry_coordinates = _runtime_dataset_registry_coordinates(data)
    if (
        registry_coordinates is not None
        and registry_coordinates[1] != dataset_key
    ):
        raise ConfigurationError(
            "data.registry_dataset_key must match the evaluation dataset_key"
        )
    for name in ("behavior_policy_version", "expected_server_weight_version"):
        if not str(director.get(name, "")).strip():
            raise ConfigurationError(f"director.{name} must be non-empty")
    behavior_adapter = director.get("behavior_adapter_name")
    behavior_checkpoint = director.get("behavior_adapter_checkpoint")
    adapter_configured = bool(
        isinstance(behavior_adapter, str) and behavior_adapter.strip()
    )
    checkpoint_configured = bool(
        isinstance(behavior_checkpoint, str) and behavior_checkpoint.strip()
    )
    if adapter_configured != checkpoint_configured:
        raise ConfigurationError(
            "director.behavior_adapter_name and "
            "director.behavior_adapter_checkpoint must be supplied together"
        )


def _runtime_dataset_registry_coordinates(
    data: Mapping[str, Any],
) -> Optional[tuple[str, str]]:
    """Return the future-only registry coordinates or reject partial opt-in."""

    registry_path_present = "registry_path" in data
    registry_key_present = "registry_dataset_key" in data
    if not registry_path_present and not registry_key_present:
        return None
    registry_path = data.get("registry_path")
    registry_key = data.get("registry_dataset_key")
    if (
        not registry_path_present
        or not registry_key_present
        or not isinstance(registry_path, str)
        or not registry_path.strip()
        or not isinstance(registry_key, str)
        or not registry_key.strip()
    ):
        raise ConfigurationError(
            "runtime dataset registry opt-in requires non-empty "
            "data.registry_path and data.registry_dataset_key"
        )
    return registry_path.strip(), registry_key.strip()


def _load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must contain a mapping: {path}")
    return value


def _same_resolved_path(root: Path, left: str, right: str) -> bool:
    return _resolve(root, left).resolve() == _resolve(root, right).resolve()


def _validate_runtime_dataset_registry(
    config: Mapping[str, Any], root: Path
) -> Optional[Mapping[str, Any]]:
    """Validate an explicitly opted-in runtime dataset registry entry.

    Legacy conditions have neither registry coordinate and return ``None``.
    The validation intentionally uses only fields that are present in the
    versioned registry and an existing preparation manifest.  A missing
    optional provenance field is recorded rather than inferred.
    """

    data = _mapping(config.get("data"), "data")
    coordinates = _runtime_dataset_registry_coordinates(data)
    if coordinates is None:
        return None
    registry_value, registry_dataset_key = coordinates
    section_name, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    if registry_dataset_key != dataset_key:
        raise ConfigurationError(
            "data.registry_dataset_key must match the evaluation dataset_key"
        )

    registry_path = _resolve(root, registry_value)
    registry = load_yaml(registry_path)
    if registry.get("schema_version") != _RUNTIME_DATASET_REGISTRY_SCHEMA:
        raise ConfigurationError(
            "unsupported runtime dataset registry schema: "
            f"{registry.get('schema_version')!r}"
        )
    registry_task_schema = registry.get("task_schema_version")
    if registry_task_schema != TASK_SCHEMA_VERSION:
        raise ConfigurationError(
            "runtime dataset registry task schema differs from the loader"
        )
    datasets = _mapping(registry.get("datasets"), "runtime registry.datasets")
    if registry_dataset_key not in datasets:
        raise ConfigurationError(
            f"runtime dataset registry has no entry for {registry_dataset_key!r}"
        )
    entry = _mapping(
        datasets[registry_dataset_key],
        f"runtime registry.datasets.{registry_dataset_key}",
    )
    protocol_label = entry.get("protocol_label")
    if not isinstance(protocol_label, str) or not protocol_label.strip():
        raise ConfigurationError("runtime dataset protocol_label must be non-empty")

    preparation_catalog_value = entry.get("preparation_catalog_path")
    if (
        not isinstance(preparation_catalog_value, str)
        or not preparation_catalog_value.strip()
    ):
        raise ConfigurationError(
            "runtime dataset preparation_catalog_path must be non-empty"
        )
    preparation_catalog_path = _resolve(root, preparation_catalog_value)
    if not preparation_catalog_path.is_file():
        raise ConfigurationError(
            "runtime dataset preparation catalog does not exist: "
            f"{preparation_catalog_path}"
        )
    expected_catalog_schema = entry.get("preparation_catalog_schema_version")
    if (
        not isinstance(expected_catalog_schema, str)
        or not expected_catalog_schema.strip()
    ):
        raise ConfigurationError(
            "runtime dataset preparation_catalog_schema_version must be non-empty"
        )
    preparation_catalog = load_yaml(preparation_catalog_path)
    if preparation_catalog.get("schema_version") != expected_catalog_schema:
        raise ConfigurationError(
            "runtime dataset preparation catalog schema differs from the registry"
        )

    registry_paths = _mapping(entry.get("paths"), "runtime dataset paths")
    validated_paths: dict[str, str] = {}
    for split, data_field in (
        ("train", "train_path"),
        ("validation", "validation_path"),
        ("test", "test_path"),
    ):
        registry_split_path = registry_paths.get(split)
        configured_split_path = data.get(data_field)
        if (
            not isinstance(registry_split_path, str)
            or not registry_split_path.strip()
            or not isinstance(configured_split_path, str)
            or not configured_split_path.strip()
        ):
            raise ConfigurationError(
                f"runtime dataset {split} path must be non-empty"
            )
        if not _same_resolved_path(
            root, registry_split_path, configured_split_path
        ):
            raise ConfigurationError(
                f"data.{data_field} differs from the runtime dataset registry"
            )
        resolved_split_path = _resolve(root, registry_split_path)
        if not resolved_split_path.is_file():
            raise ConfigurationError(
                f"runtime dataset {split} file does not exist: "
                f"{resolved_split_path}"
            )
        validated_paths[split] = str(resolved_split_path.resolve())

    configured_task_schema = data.get("task_schema_version")
    if configured_task_schema != registry_task_schema:
        raise ConfigurationError(
            "data.task_schema_version differs from the runtime dataset registry"
        )

    manifest_spec = _mapping(entry.get("manifest"), "runtime dataset manifest")
    manifest_value = manifest_spec.get("path")
    expected_manifest_schema = manifest_spec.get("schema_version")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ConfigurationError("runtime dataset manifest.path must be non-empty")
    if (
        not isinstance(expected_manifest_schema, str)
        or not expected_manifest_schema.strip()
    ):
        raise ConfigurationError(
            "runtime dataset manifest.schema_version must be non-empty"
        )
    manifest_path = _resolve(root, manifest_value)
    manifest = _load_json_mapping(manifest_path, "runtime dataset manifest")
    if manifest.get("schema_version") != expected_manifest_schema:
        raise ConfigurationError(
            "runtime dataset manifest schema differs from the registry"
        )

    checks: dict[str, Any] = {
        "registry_dataset_key_matches_evaluation": True,
        "preparation_catalog_schema_matches_registry": True,
        "explicit_split_paths_match_registry": True,
        "split_files_exist": True,
        "task_schema_matches_loader_and_config": True,
        "manifest_schema_matches_registry": True,
    }
    skipped_checks: list[Mapping[str, str]] = []
    manifest_task_schema = manifest.get("task_schema_version")
    if manifest_task_schema is None:
        skipped_checks.append(
            {
                "check": "manifest_task_schema",
                "reason": "manifest has no task_schema_version field",
            }
        )
    elif manifest_task_schema != registry_task_schema:
        raise ConfigurationError(
            "runtime dataset manifest task schema differs from the registry"
        )
    else:
        checks["manifest_task_schema_matches_registry"] = True

    provenance_field = manifest_spec.get("provenance_field")
    if not isinstance(provenance_field, str) or not provenance_field.strip():
        skipped_checks.append(
            {
                "check": "manifest_preparation_provenance",
                "reason": "registry manifest spec has no provenance_field",
            }
        )
    else:
        observed_provenance = manifest.get(provenance_field)
        if not isinstance(observed_provenance, str) or not observed_provenance.strip():
            skipped_checks.append(
                {
                    "check": "manifest_preparation_provenance",
                    "reason": (
                        "manifest has no non-empty "
                        f"{provenance_field!r} field"
                    ),
                }
            )
        elif not _same_resolved_path(
            root, observed_provenance, preparation_catalog_value
        ):
            raise ConfigurationError(
                "runtime dataset manifest provenance differs from the registry"
            )
        else:
            checks["manifest_preparation_provenance_matches_registry"] = True

    manifest_files = manifest.get("files")
    if isinstance(manifest_files, Mapping):
        for split, resolved_path in validated_paths.items():
            if manifest_files.get(split) != Path(resolved_path).name:
                raise ConfigurationError(
                    f"runtime dataset manifest {split} file differs from the registry"
                )
        checks["manifest_split_files_match_registry"] = True
    else:
        manifest_partition_keys = entry.get("manifest_partition_keys")
        manifest_partitions = manifest.get("partitions")
        if isinstance(manifest_partition_keys, Mapping) and isinstance(
            manifest_partitions, Mapping
        ):
            for split, resolved_path in validated_paths.items():
                partition_key = manifest_partition_keys.get(split)
                partition = manifest_partitions.get(partition_key)
                if (
                    not isinstance(partition_key, str)
                    or not isinstance(partition, Mapping)
                    or partition.get("file") != Path(resolved_path).name
                ):
                    raise ConfigurationError(
                        "runtime dataset manifest partition file differs from "
                        f"the registry for {split}"
                    )
            checks["manifest_partition_files_match_registry"] = True
        else:
            skipped_checks.append(
                {
                    "check": "manifest_split_files",
                    "reason": (
                        "manifest has no files mapping and the registry has no "
                        "reliable manifest_partition_keys mapping"
                    ),
                }
            )

    selected_split = str(bounded["split"])
    if dataset_key == "swe_bench":
        evaluation = _mapping(config.get("evaluation"), "evaluation")
        source_by_split = entry.get("swebench_dataset_source_by_split")
        split_policy = manifest.get("split_policy")
        source_field_by_split = {
            "train": "train_source",
            "validation": "validation_source",
            "test": "test_source",
        }
        if not isinstance(source_by_split, Mapping) or not isinstance(
            split_policy, Mapping
        ):
            raise ConfigurationError(
                "SWE-bench runtime registry requires manifest-bound split sources"
            )
        for split, source_field in source_field_by_split.items():
            expected_source = source_by_split.get(split)
            if (
                not isinstance(expected_source, str)
                or not expected_source.strip()
                or split_policy.get(source_field) != expected_source
            ):
                raise ConfigurationError(
                    "SWE-bench runtime registry split source differs from the manifest"
                )
        checks["swebench_split_sources_match_manifest"] = True
        evaluator_dataset_path = evaluation.get("swebench_dataset_path")
        if (
            not isinstance(evaluator_dataset_path, str)
            or not evaluator_dataset_path.strip()
            or not _same_resolved_path(
                root, evaluator_dataset_path, str(registry_paths[selected_split])
            )
        ):
            raise ConfigurationError(
                "evaluation.swebench_dataset_path differs from the selected "
                "runtime registry split"
            )
        checks["swebench_evaluator_dataset_path_matches_selected_split"] = True
        expected_source = (
            source_by_split.get(selected_split)
            if isinstance(source_by_split, Mapping)
            else None
        )
        if isinstance(expected_source, str) and expected_source.strip():
            if evaluation.get("swebench_dataset_source") != expected_source:
                raise ConfigurationError(
                    "evaluation.swebench_dataset_source differs from the runtime "
                    "dataset registry"
                )
            checks["swebench_dataset_source_matches_selected_split"] = True
        else:
            skipped_checks.append(
                {
                    "check": "swebench_dataset_source",
                    "reason": (
                        "registry has no reliable source binding for the selected "
                        "split"
                    ),
                }
            )

    return {
        "schema_version": "flowsteer.runtime-dataset-registry-validation.v1",
        "enabled": True,
        "registry_path": str(registry_path.resolve()),
        "registry_schema_version": _RUNTIME_DATASET_REGISTRY_SCHEMA,
        "registry_dataset_key": registry_dataset_key,
        "evaluation_section": section_name,
        "selected_split": selected_split,
        "protocol_label": protocol_label.strip(),
        "preparation_catalog_path": str(preparation_catalog_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "validated_paths": validated_paths,
        "checks": checks,
        "skipped_checks": skipped_checks,
    }


def _paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config["storage"], "storage")
    names = {
        "selected": "selected_tasks_path",
        "direct": "direct_predictions_path",
        "trajectories": "trajectories_path",
        "failures": "failures_path",
        "paired": "paired_results_path",
        "wrong": "wrong_demos_path",
        "manifest": "manifest_path",
        "preflight": "preflight_receipt_path",
        "report_json": "report_json_path",
        "report_markdown": "report_markdown_path",
    }
    return {name: _resolve(root, str(storage[field])) for name, field in names.items()}


def _benchmark_slice(task: TaskRecord) -> str:
    value = task.metadata.get("benchmark_slice")
    return str(value).strip() if value is not None else ""


def _select_tasks(
    config: Mapping[str, Any], root: Path, selected_path: Path
) -> tuple[TaskRecord, ...]:
    """Freeze exactly the aligned records selected by the benchmark section."""

    _, bounded = _evaluation_section(config)
    data = _mapping(config["data"], "data")
    dataset_key = str(bounded["dataset_key"])
    split = str(bounded["split"])
    source_field = {
        "train": "train_path",
        "validation": "validation_path",
        "test": "test_path",
    }[split]
    source = _resolve(root, str(data[source_field]))
    requested_slice = bounded.get("benchmark_slice")
    if bounded.get("official_2026_only") is True:
        requested_slice = "official_aime_2026"
    candidates = tuple(
        task
        for task in iter_task_records(source, expected_split=split)
        if _dataset_key(task) == dataset_key
        and (
            not requested_slice
            or _benchmark_slice(task) == str(requested_slice)
        )
    )
    count = int(bounded["sample_count"])
    if len(candidates) < count:
        suffix = f" in benchmark_slice={requested_slice}" if requested_slice else ""
        raise CompletionBenchmarkRoundError(
            f"aligned {split} contains only {len(candidates)} {dataset_key} tasks"
            f"{suffix}; expected {count}"
        )
    if bounded.get("selection") == "task_ids":
        by_id = {task.task_id: task for task in candidates}
        requested = [str(task_id) for task_id in bounded["task_ids"]]
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise CompletionBenchmarkRoundError(
                f"requested {dataset_key} task IDs are absent from {split}: "
                + ", ".join(missing)
            )
        expected = tuple(by_id[task_id] for task_id in requested)
    else:
        expected = candidates[:count]

    required_partition = bounded.get("required_partition")

    def require_partition(task: TaskRecord) -> None:
        if required_partition is None:
            return
        observed = task.metadata.get("joint_qa_partition")
        if observed != required_partition:
            raise CompletionBenchmarkRoundError(
                f"task {task.task_id!r} belongs to partition {observed!r}, "
                f"expected {required_partition!r}"
            )

    for task in expected:
        require_partition(task)

    if selected_path.exists():
        frozen = tuple(iter_task_records(selected_path, expected_split=split))
        if len(frozen) != count:
            raise CompletionBenchmarkRoundError(
                f"frozen {dataset_key} selection has the wrong size"
            )
        for expected_task, frozen_task in zip(expected, frozen, strict=True):
            require_partition(frozen_task)
            if (
                expected_task.task_id != frozen_task.task_id
                or expected_task.question != frozen_task.question
                or expected_task.ground_truth != frozen_task.ground_truth
            ):
                raise CompletionBenchmarkRoundError(
                    f"frozen {dataset_key} selection differs from aligned {split} data"
                )
        return frozen

    _atomic_jsonl(
        selected_path,
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            for task in expected
        ],
    )
    return expected


async def _evaluate_prediction(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    prediction: str,
    *,
    run_graph: Optional[Any] = None,
) -> Any:
    """Use the existing evaluator and its configured benchmark dependency."""

    dataset_key = _dataset_key(task)
    if dataset_key == "healthbench_professional":
        return await evaluate_task(
            task,
            prediction,
            judge=backend.judge,
            judge_model=backend.judge_model,
        )
    if dataset_key in _INTERACTIVE_BENCHMARKS:
        evaluation = _mapping(backend.config["evaluation"], "evaluation")
        configured_steps = evaluation.get("max_environment_steps_by_source", {})
        if not isinstance(configured_steps, Mapping):
            configured_steps = {}
        return await evaluate_task(
            task,
            prediction,
            run_graph=run_graph,
            max_environment_steps=int(
                configured_steps.get(
                    dataset_key,
                    evaluation["max_environment_steps"],
                )
            ),
        )
    swe_runtime = backend.config.get("swe_coding_runtime")
    if (
        dataset_key == "swe_bench"
        and isinstance(swe_runtime, Mapping)
        and swe_runtime.get("enabled") is True
    ):
        return await evaluate_task(
            task,
            prediction,
            swe_harness=backend.swe_harness,
        )
    return await evaluate_task(task, prediction)


def _synthetic_evaluator_preflight_fixture(
    dataset_key: str,
) -> tuple[TaskRecord, str]:
    """Return a fixed evaluator-only fixture that is outside benchmark data.

    Completion benchmark answers are evaluator-only data, but a final-test
    preflight must not read even one selected answer.  Static evaluators and
    callback-only evaluators can exercise their existing public boundary with
    these synthetic records.  Stateful environment evaluators use a fixed
    non-test aligned record instead (see
    :func:`_non_test_environment_preflight_task`).
    """

    common = {
        "task_id": f"evaluator-preflight:{dataset_key}:synthetic-v1",
        "split": "train",
    }
    if dataset_key == "hotpotqa":
        return (
            TaskRecord(
                **common,
                question="Synthetic evaluator preflight question.",
                ground_truth="Alpha River",
                metadata={
                    "dataset_key": dataset_key,
                    "evaluator_payload": {"accepted_answers": ["Alpha River"]},
                },
            ),
            "<answer>Alpha River</answer>",
        )
    if dataset_key == "triviaqa":
        return (
            TaskRecord(
                **common,
                question="Synthetic evaluator preflight question.",
                ground_truth="Beta City",
                metadata={
                    "dataset_key": dataset_key,
                    "evaluator_payload": {"accepted_answers": ["Beta City"]},
                },
            ),
            "<answer>Beta City</answer>",
        )
    if dataset_key == "aime_2026":
        return (
            TaskRecord(
                **common,
                question="Synthetic integer evaluator preflight question.",
                ground_truth="42",
                metadata={
                    "dataset_key": dataset_key,
                    "evaluator_payload": {"accepted_answers": ["42"]},
                },
            ),
            "<answer>42</answer>",
        )
    if dataset_key == "healthbench_professional":
        return (
            TaskRecord(
                **common,
                question="Conversation:\n\n[user] Return the word preflight.\n\n[assistant]",
                ground_truth=None,
                metadata={
                    "dataset_key": dataset_key,
                    "evaluator_payload": {
                        "rubric_items": [
                            {
                                "criterion_text": (
                                    "The assistant response contains the word preflight."
                                ),
                                "points": 1,
                            }
                        ]
                    },
                },
            ),
            "preflight",
        )
    if dataset_key == "swe_bench":
        return (
            TaskRecord(
                **common,
                question="Synthetic SWE-bench evaluator callback preflight.",
                ground_truth=None,
                metadata={"dataset_key": dataset_key},
            ),
            "",
        )
    raise CompletionBenchmarkRoundError(
        f"{dataset_key} requires a non-test aligned environment fixture"
    )


def _non_test_environment_preflight_task(
    config: Mapping[str, Any], root: Path, dataset_key: str
) -> TaskRecord:
    """Load the first fixed training record for an environment preflight."""

    if dataset_key not in _INTERACTIVE_BENCHMARKS:
        raise CompletionBenchmarkRoundError(
            "a non-test environment fixture is valid only for interactive benchmarks"
        )
    data = _mapping(config["data"], "data")
    train_path = _resolve(root, str(data["train_path"]))
    try:
        task = next(iter_task_records(train_path, expected_split="train"))
    except StopIteration as exc:
        raise CompletionBenchmarkRoundError(
            f"{dataset_key} has no fixed training evaluator fixture"
        ) from exc
    if _dataset_key(task) != dataset_key:
        raise CompletionBenchmarkRoundError(
            f"{dataset_key} training evaluator fixture has the wrong dataset"
        )
    return task


def _evaluator_preflight_receipt(
    outcome: Any, dataset_key: str
) -> Mapping[str, Any]:
    """Validate one fixture outcome and retain no answer-bearing fields."""

    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    metric_names = tuple(_BENCHMARKS[dataset_key]["metric_names"])
    values = {name: outcome.metrics.get(name) for name in metric_names}
    passed = outcome.valid and all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values.values()
    )
    if passed and dataset_key == "aime_2026":
        passed = passed and float(values[metric_name]) == 1.0
    elif passed and dataset_key in {"hotpotqa", "triviaqa"}:
        passed = passed and all(float(value) == 1.0 for value in values.values())
    if not passed:
        raise CompletionBenchmarkRoundError(
            f"fixed-fixture {dataset_key} evaluator preflight failed"
        )
    return {
        "passed": True,
        "evaluator_version": str(outcome.evaluator_version),
    }


async def _run_evaluator_preflight(
    backend: LiveSmokeBackend,
    config: Mapping[str, Any],
    root: Path,
    dataset_key: str,
) -> Mapping[str, Any]:
    """Exercise the evaluator without consulting the selected benchmark set."""

    if dataset_key in _INTERACTIVE_BENCHMARKS:
        task = _non_test_environment_preflight_task(config, root, dataset_key)

        async def invalid_environment_action(_prompt: str) -> str:
            return "<INVALID>"

        outcome = await _evaluate_prediction(
            backend,
            task,
            "",
            run_graph=invalid_environment_action,
        )
    elif dataset_key == "swe_bench":
        # The official harness performs its own runtime preflight before this
        # callback check.  Use a synthetic unresolved result here so neither a
        # selected instance nor its trusted result enters this generic
        # evaluator receipt.
        task, prediction = _synthetic_evaluator_preflight_fixture(dataset_key)

        async def synthetic_unresolved_harness(
            _task: TaskRecord | Mapping[str, Any], _prediction: str
        ) -> Mapping[str, bool]:
            return {"resolved": False}

        outcome = await evaluate_task(
            task,
            prediction,
            swe_harness=synthetic_unresolved_harness,
        )
    else:
        task, prediction = _synthetic_evaluator_preflight_fixture(dataset_key)
        outcome = await _evaluate_prediction(backend, task, prediction)
    return _evaluator_preflight_receipt(outcome, dataset_key)


async def _direct_one(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    index: int,
    *,
    model_id: str,
    protocol: str,
    contract: str,
    seed: int,
    run_label: str,
) -> Mapping[str, Any]:
    model = backend.registry.require_model(model_id)
    provider = backend.registry.provider_for(model_id)
    run_id = f"{run_label}-direct-{index:04d}"
    dataset_key = _dataset_key(task)
    started_at = _utc_now()

    if dataset_key == "swe_bench":
        bounded = _evaluation_section(backend.config)[1]
        condition_id = str(backend.config["experiment"]["condition_id"])
        task_runtime, tool_registry, close_runtime = backend._runtime_for_task(
            task,
            condition_id=condition_id,
        )
        try:
            if tool_registry is None:
                raise CompletionBenchmarkRoundError(
                    "SWE-bench Direct Coding Agent has no repository ToolRegistry"
                )
            node = AgentNode(
                "direct_coding_agent",
                model_id,
                contract,
                role_family="coding",
                allowed_tools=tool_registry.resource_ids,
                execution_mode="coding",
                artifact_type="unified_diff",
                completion_condition=str(
                    bounded["direct_completion_condition"]
                ),
            )
            graph = AgentGraph(
                nodes=(node,),
                output_agent_id=node.id,
            )
            runtime_result = await task_runtime.execute(
                graph,
                task.question,
                run_id=run_id,
            )
            if not runtime_result.final_answer:
                raise CompletionBenchmarkRoundError(
                    "SWE-bench Direct Coding Agent produced no workspace diff"
                )
            calls = tuple(runtime_result.calls)
            if len(calls) != 1:
                raise CompletionBenchmarkRoundError(
                    "SWE-bench single Coding Agent produced an invalid outer call count"
                )
            raw_model_calls = calls[0].response.metadata.get("model_calls", ())
            model_call_seeds = [
                item.get("metadata", {}).get("generation_seed")
                for item in raw_model_calls
                if isinstance(item, Mapping)
                and isinstance(item.get("metadata"), Mapping)
            ]
            if not model_call_seeds or any(
                actual_seed != seed for actual_seed in model_call_seeds
            ):
                raise CompletionBenchmarkRoundError(
                    "Direct Coding Agent generation seed receipts differ from config"
                )
            executions = [
                execution_record_from_call(call).to_dict() for call in calls
            ]
            evaluation = await _evaluate_prediction(
                backend,
                task,
                runtime_result.final_answer,
            )
            return {
                "schema_version": "flowsteer.completion_benchmark.direct_prediction.v1",
                "dataset_key": dataset_key,
                "task_id": task.task_id,
                "task": task.to_dict(),
                "condition": "direct_local_qwen35_9b",
                "protocol": protocol,
                "simple_baseline_topology": "single_coding_agent",
                "model_id": model_id,
                "provider_id": provider.provider_id,
                "provider_model": model.model_name,
                "generation_seed": seed,
                "final_answer": runtime_result.final_answer,
                "evaluation": asdict(evaluation),
                "execution": executions[-1],
                "executions": executions,
                "runtime": {
                    "run_id": runtime_result.run_id,
                    "output_agent_id": runtime_result.output_agent_id,
                    "block_completion_order": [
                        list(block)
                        for block in runtime_result.block_completion_order
                    ],
                    "executed_agent_ids": list(
                        runtime_result.executed_agent_ids
                    ),
                },
                "started_at": started_at,
                "completed_at": _utc_now(),
            }
        finally:
            close_runtime()

    if dataset_key in _INTERACTIVE_BENCHMARKS:
        executions: list[dict[str, Any]] = []
        runtime_section = _mapping(
            backend.config.get("environment_runtime", {}),
            "environment_runtime",
        )
        director_section = _mapping(backend.config.get("director", {}), "director")
        max_action_tokens = runtime_section.get(
            "max_action_tokens", director_section.get("max_action_tokens", 512)
        )
        if (
            isinstance(max_action_tokens, bool)
            or not isinstance(max_action_tokens, int)
            or max_action_tokens < 1
        ):
            raise CompletionBenchmarkRoundError(
                "environment max_action_tokens must be a positive integer"
            )
        action_model = replace(
            model,
            metadata={
                **dict(model.metadata),
                "max_tokens": str(max_action_tokens),
            },
        )

        async def direct_action(environment_prompt: str) -> str:
            step_index = len(executions)
            request = AgentRequest(
                request_id=f"{run_id}:direct:environment:{step_index:04d}",
                run_id=run_id,
                graph_revision=0,
                problem=environment_prompt,
                agent=AgentNode("direct", model_id, contract),
                model=action_model,
                provider=provider,
                phase=ExecutionPhase.SINGLE,
                is_output_agent=True,
            )
            response = await backend.runtime.gateway.generate(request)
            execution = execution_record_from_call(AgentCallRecord(request, response))
            actual_seed = execution.metadata.get("response", {}).get(
                "generation_seed"
            )
            if actual_seed != seed:
                raise CompletionBenchmarkRoundError(
                    "Direct environment generation seed receipt differs from config"
                )
            executions.append(execution.to_dict())
            return response.text

        evaluation = await _evaluate_prediction(
            backend,
            task,
            "",
            run_graph=direct_action,
        )
        if not executions:
            raise CompletionBenchmarkRoundError(
                "interactive evaluator completed without a Direct policy call"
            )
        final_execution = executions[-1]
        return {
            "schema_version": "flowsteer.completion_benchmark.direct_prediction.v1",
            "dataset_key": dataset_key,
            "task_id": task.task_id,
            "task": task.to_dict(),
            "condition": "direct_local_qwen35_9b",
            "protocol": protocol,
            "model_id": model_id,
            "provider_id": provider.provider_id,
            "provider_model": model.model_name,
            "generation_seed": seed,
            "final_answer": str(final_execution.get("output", "")),
            "evaluation": asdict(evaluation),
            "execution": final_execution,
            "executions": executions,
            "started_at": started_at,
            "completed_at": _utc_now(),
        }

    direct_problem = _workflow_problem(task, backend.config)
    if dataset_key == "aime_2026":
        # SkillFlow/SkillEval does not expose an independent single-model
        # Direct runner: its formal baseline is a bounded policy episode.  For
        # the requested Direct comparator, reuse FlowSteer's actual
        # AnswerGenerate input protocol rather than relabelling that bounded
        # rollout.  Only the public problem and answer-format metadata enter
        # the operator prompt; the hidden target remains evaluator-only.
        answer_format = str(
            task.metadata.get("answer_format", "integer-000-to-999")
        )
        public_input = (
            f"{task.question}\n\n"
            f"Public answer format: {answer_format}."
        )
        direct_problem = XmlFormatter.from_model(AnswerGenerateOp).prepare_prompt(
            ANSWER_GENERATION_PROMPT.format(input=public_input)
        )

    request = AgentRequest(
        request_id=f"{run_id}:direct:single",
        run_id=run_id,
        graph_revision=0,
        problem=direct_problem,
        agent=AgentNode("direct", model_id, contract),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
        is_output_agent=True,
    )
    response = await backend.runtime.gateway.generate(request)
    execution = execution_record_from_call(AgentCallRecord(request, response))
    actual_seed = execution.metadata.get("response", {}).get("generation_seed")
    if actual_seed != seed:
        raise CompletionBenchmarkRoundError(
            "Direct generation seed receipt differs from config"
        )
    evaluation = await _evaluate_prediction(backend, task, response.text)
    return {
        "schema_version": "flowsteer.completion_benchmark.direct_prediction.v1",
        "dataset_key": dataset_key,
        "task_id": task.task_id,
        "task": task.to_dict(),
        "condition": "direct_local_qwen35_9b",
        "protocol": protocol,
        "model_id": model_id,
        "provider_id": provider.provider_id,
        "provider_model": model.model_name,
        "generation_seed": actual_seed,
        "final_answer": response.text,
        "evaluation": asdict(evaluation),
        "execution": execution.to_dict(),
        "started_at": started_at,
        "completed_at": _utc_now(),
    }


async def _collect_direct(
    backend: LiveSmokeBackend,
    selected: Sequence[TaskRecord],
    config: Mapping[str, Any],
    root: Path,
    path: Path,
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Checkpoint and resume the paired Direct condition task-by-task."""

    _, bounded = _evaluation_section(config)
    experiment = _mapping(config["experiment"], "experiment")
    model_id = str(bounded["direct_model_id"])
    protocol = str(bounded["direct_protocol"])
    contract = str(bounded["direct_contract"])
    seed = int(bounded.get("direct_generation_seed", experiment["seed"]))
    run_label = str(experiment["name"])
    concurrency = int(bounded["concurrency"])
    direct_candidates = _read_jsonl(path)
    reuse_source = bounded.get("direct_reused_from")
    reuse_path: Optional[Path] = None
    if isinstance(reuse_source, str) and reuse_source.strip():
        reuse_path = _resolve(root, reuse_source)
        if not reuse_path.is_file():
            raise CompletionBenchmarkRoundError(
                f"declared Direct reuse source does not exist: {reuse_path}"
            )
        if reuse_path.resolve() != path.resolve():
            reused = []
            for value in _read_jsonl(reuse_path):
                copied = dict(value)
                copied["reuse_receipt"] = {"reused": True, "source": str(reuse_path)}
                reused.append(copied)
            direct_candidates = reused + direct_candidates

    selected_by_id = {task.task_id: task for task in selected}
    rescored: list[dict[str, Any]] = []
    for candidate in direct_candidates:
        task = selected_by_id.get(candidate.get("task_id"))
        evaluation = candidate.get("evaluation")
        if (
            task is not None
            and _dataset_key(task) not in _INTERACTIVE_BENCHMARKS
            and candidate.get("model_id") == model_id
            and candidate.get("protocol") == protocol
            and candidate.get("generation_seed") == seed
            and isinstance(candidate.get("final_answer"), str)
            and (
                not isinstance(evaluation, Mapping)
                or evaluation.get("evaluator_version") != evaluator_version_for(task)
            )
        ):
            updated = dict(candidate)
            updated["evaluation"] = asdict(
                await _evaluate_prediction(backend, task, str(candidate["final_answer"]))
            )
            updated["rescore_receipt"] = {
                "mode": "offline_existing_prediction",
                "source_evaluator_version": (
                    evaluation.get("evaluator_version")
                    if isinstance(evaluation, Mapping)
                    else None
                ),
                "target_evaluator_version": evaluator_version_for(task),
            }
            rescored.append(updated)
        else:
            rescored.append(dict(candidate))

    by_task = {
        task_id: value
        for task_id, value in _by_task(rescored).items()
        for task in selected
        if task.task_id == task_id
        and hotpot_round._direct_resume_matches(
            value,
            task=task,
            model_id=model_id,
            protocol=protocol,
            seed=seed,
        )
    }
    hotpot_round._persist_ordered(path, selected, by_task)

    def checkpoint() -> None:
        manifest["direct_progress"] = {
            "completed": len(by_task),
            "reused_from": None if reuse_path is None else str(reuse_path),
            "reused_records": sum(
                value.get("reuse_receipt", {}).get("reused") is True
                for value in by_task.values()
            ),
            "newly_collected_records": sum(
                value.get("reuse_receipt", {}).get("reused") is not True
                for value in by_task.values()
            ),
            "failed_attempts": sum(
                item.get("condition") == "direct_local_qwen35_9b"
                for item in failures
            ),
        }
        _write_json(manifest_path, manifest)

    checkpoint()
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int, task: TaskRecord) -> tuple[TaskRecord, Any]:
        async with semaphore:
            try:
                return task, await _direct_one(
                    backend,
                    task,
                    index,
                    model_id=model_id,
                    protocol=protocol,
                    contract=contract,
                    seed=seed,
                    run_label=run_label,
                )
            except BaseException as exc:
                return task, exc

    jobs = [
        asyncio.create_task(run(index, task))
        for index, task in enumerate(selected)
        if task.task_id not in by_task
    ]
    for completed in asyncio.as_completed(jobs):
        task, result = await completed
        if isinstance(result, BaseException):
            failures.append(
                {
                    "task_id": task.task_id,
                    "condition": "direct_local_qwen35_9b",
                    "stage": "generation_or_evaluator",
                    "error": _safe_error(result),
                    "recorded_at": _utc_now(),
                }
            )
        else:
            by_task[task.task_id] = dict(result)
            hotpot_round._persist_ordered(path, selected, by_task)
        checkpoint()
    return by_task


def _metric(
    value: Optional[Mapping[str, Any]], dataset_key: str
) -> tuple[bool, float]:
    valid, metrics = _metrics(value, dataset_key)
    primary = str(_BENCHMARKS[dataset_key]["primary_metric"])
    return valid, metrics.get(primary, 0.0)


def _metrics(
    value: Optional[Mapping[str, Any]], dataset_key: str
) -> tuple[bool, dict[str, float]]:
    """Return every native report metric, failing closed as one receipt."""

    metric_names = tuple(_BENCHMARKS[dataset_key]["metric_names"])
    empty = {str(name): 0.0 for name in metric_names}
    if value is None:
        return False, empty
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("valid") is not True:
        return False, empty
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, empty
    values: dict[str, float] = {}
    for metric_name in metric_names:
        raw = metrics.get(metric_name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False, empty
        metric_value = float(raw)
        if not math.isfinite(metric_value):
            return False, empty
        values[str(metric_name)] = metric_value
    return True, values


def _direct_telemetry(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    execution = value.get("execution") if value else None
    recorded = value.get("executions") if value else None
    if isinstance(recorded, Sequence) and not isinstance(recorded, (str, bytes)):
        executions = [item for item in recorded if isinstance(item, Mapping)]
    elif isinstance(execution, Mapping):
        executions = [execution]
    else:
        executions = []
    return {
        "api_attempts": sum(
            int(item.get("metadata", {}).get("response", {}).get("attempt_count") or 0)
            for item in executions
        ),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in executions),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in executions),
        "latency_ms": sum(float(item.get("latency_ms") or 0.0) for item in executions),
        "environment_policy_calls": len(executions),
    }


def _latest_failure_diagnostic_text(
    graph_value: Mapping[str, Any],
) -> str:
    """Return only the terminal/latest public Canvas failure receipt."""

    turns = graph_value.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return ""
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        runtime_summary = turn.get("runtime_summary")
        summary = runtime_summary if isinstance(runtime_summary, Mapping) else {}
        terminal_diagnosis = summary.get("terminal_canvas_diagnosis")
        failure_records = summary.get("failure_records")
        has_failure_records = bool(
            isinstance(failure_records, Sequence)
            and not isinstance(failure_records, (str, bytes))
            and any(isinstance(item, Mapping) for item in failure_records)
        )
        if not (
            isinstance(terminal_diagnosis, Mapping)
            or has_failure_records
            or summary.get("execution_status") == "failed"
        ):
            continue
        responsible_agent_ids: set[str] = set()
        if isinstance(terminal_diagnosis, Mapping):
            finish_admissibility = terminal_diagnosis.get("finish_admissibility")
            attribution = (
                finish_admissibility.get("failure_attribution")
                if isinstance(finish_admissibility, Mapping)
                else None
            )
            if isinstance(attribution, Mapping):
                responsible_agent_id = attribution.get("responsible_agent_id")
                if isinstance(responsible_agent_id, str):
                    responsible_agent_ids.add(responsible_agent_id)
                attributed_ids = attribution.get("responsible_agent_ids")
                if isinstance(attributed_ids, Sequence) and not isinstance(
                    attributed_ids, (str, bytes)
                ):
                    responsible_agent_ids.update(
                        item for item in attributed_ids if isinstance(item, str)
                    )
        mapped_failure_records = (
            tuple(item for item in failure_records if isinstance(item, Mapping))
            if has_failure_records
            else ()
        )
        latest_failure_record = next(
            (
                item
                for item in reversed(mapped_failure_records)
                if item.get("agent_id") in responsible_agent_ids
            ),
            None,
        )
        if latest_failure_record is None and mapped_failure_records:
            latest_failure_record = mapped_failure_records[-1]
        latest_public_observation = None
        if isinstance(latest_failure_record, Mapping):
            metadata = latest_failure_record.get("metadata")
            trace = metadata.get("react_trace") if isinstance(metadata, Mapping) else None
            if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
                latest_public_observation = next(
                    (
                        item
                        for item in reversed(trace)
                        if isinstance(item, Mapping)
                    ),
                    None,
                )
        payload = {
            "canvas_feedback": turn.get("canvas_feedback"),
            "terminal_canvas_diagnosis": terminal_diagnosis,
            "latest_failure_record": (
                {
                    field_name: latest_failure_record.get(field_name)
                    for field_name in (
                        "agent_id",
                        "error_type",
                        "message",
                        "phase",
                    )
                }
                if isinstance(latest_failure_record, Mapping)
                else None
            ),
            "latest_public_observation": latest_public_observation,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).casefold()
    return ""


def _failure_type(
    direct_value: Optional[Mapping[str, Any]],
    graph_value: Optional[Mapping[str, Any]],
    *,
    direct_valid: bool,
    graph_valid: bool,
    direct_score: float,
    graph_score: float,
    dataset_key: str,
) -> str:
    if direct_value is None or not direct_valid:
        return "direct_operational_or_evaluator_failure"
    if graph_value is None:
        return "agentgraph_operational_or_evaluator_failure"
    if (
        dataset_key in {"aime_2026", "webshop", "alfworld"}
        and graph_value.get("explicit_finish") is not True
    ):
        return "agentgraph_terminal_failure"
    if not graph_valid:
        return "agentgraph_operational_or_evaluator_failure"
    if dataset_key == "triviaqa" and graph_score < 1.0:
        evaluation = graph_value.get("evaluation")
        details = (
            evaluation.get("details", {})
            if isinstance(evaluation, Mapping)
            else {}
        )
        mismatch_type = (
            details.get("answer_mismatch_type")
            if isinstance(details, Mapping)
            else None
        )
        if mismatch_type == "accepted_answer_canonicalization_mismatch":
            return "accepted_answer_canonicalization_mismatch"
        if (
            mismatch_type == "partial_answer_overlap"
            and graph_value.get("explicit_finish") is True
        ):
            # Earlier rejected retrieval/completion attempts remain in the
            # lossless trajectory.  They must not relabel a later complete,
            # explicitly finished evidence lineage as retrieval failure when
            # the official evaluator's final-answer diagnosis is partial
            # accepted-answer overlap.
            return "partial_answer_overlap"
        if (
            mismatch_type == "no_accepted_answer_overlap"
            and graph_value.get("explicit_finish") is True
        ):
            # An accepted explicit FINISH is the final-state receipt that the
            # evidence/semantic/Verifier/Formatter lineage passed admission.
            # Earlier rejected retrieval attempts remain in the lossless
            # trajectory for diagnosis, but cannot relabel this final
            # evaluator mismatch as a retrieval failure.
            return "reasoning_failure"
        latest_diagnostic_text = _latest_failure_diagnostic_text(graph_value)
        if "knowledge_base_coverage_failure" in latest_diagnostic_text:
            return "knowledge_base_coverage_failure"
        if "retrieval_strategy_failure" in latest_diagnostic_text:
            return "retrieval_strategy_failure"
        if "retrieval_recall_failure" in latest_diagnostic_text:
            return "retrieval_recall_failure"
        if any(
            marker in latest_diagnostic_text
            for marker in (
                "answer_slot",
                "entity_attribute_binding",
                "alias_binding",
                "scope_preserved",
                "target_relation",
                "qa_location_containment_lineage_missing",
                "reasoner_semantic_artifact",
            )
        ):
            return "relation_or_answer_slot_binding_failure"
        if any(
            marker in latest_diagnostic_text
            for marker in (
                "parse_error",
                "completion_schema",
                "structured_action",
                "terminal_protocol",
            )
        ):
            return "structured_output_or_format_failure"

        diagnostic_payload = {
            "turns": [
                {
                    "canvas_feedback": turn.get("canvas_feedback"),
                    "runtime_summary": turn.get("runtime_summary"),
                }
                for turn in graph_value.get("turns", ())
                if isinstance(turn, Mapping)
            ],
            "fallback": graph_value.get("valid_lineage_fallback_receipt"),
        }
        diagnostic_text = json.dumps(
            diagnostic_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).casefold()
        if "knowledge_base_coverage_failure" in diagnostic_text:
            return "knowledge_base_coverage_failure"
        if "retrieval_strategy_failure" in diagnostic_text:
            return "retrieval_strategy_failure"
        if any(
            marker in diagnostic_text
            for marker in (
                "answer_slot",
                "entity_attribute_binding",
                "alias_binding",
                "scope_preserved",
                "target_relation",
                "qa_location_containment_lineage_missing",
            )
        ):
            return "relation_or_answer_slot_binding_failure"
        if any(
            marker in diagnostic_text
            for marker in (
                "semantic_evidence_provenance_invalid",
                "qa_completion_requires_successful_read_evidence",
                "qa_read_requires_successful_search",
                "retrieval_recall_failure",
                "tool_error",
            )
        ):
            return "retrieval_recall_failure"
        if any(
            marker in diagnostic_text
            for marker in (
                "parse_error",
                "completion_schema",
                "structured_action",
                "terminal_protocol",
            )
        ):
            return "structured_output_or_format_failure"
        if graph_value.get("explicit_finish") is not True:
            return "agentgraph_terminal_failure"
        return "reasoning_failure"
    if graph_value.get("explicit_finish") is not True:
        return "agentgraph_terminal_failure"
    if dataset_key == "aime_2026":
        evaluation = graph_value.get("evaluation")
        details = (
            evaluation.get("details", {})
            if isinstance(evaluation, Mapping)
            else {}
        )
        if (
            isinstance(details, Mapping)
            and details.get("parsing_succeeded") is False
        ):
            return "output_parsing_failure"
        if graph_score == 1.0 and direct_score == 0.0:
            return "agentgraph_accuracy_gain"
        if graph_score == 0.0 and direct_score == 1.0:
            return "agentgraph_accuracy_regression"
        return "both_correct" if graph_score == 1.0 else "both_incorrect"
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    if graph_score > direct_score:
        return f"agentgraph_higher_{metric_name}"
    if graph_score < direct_score:
        return f"direct_higher_{metric_name}"
    return f"equal_{metric_name}"


def _terminal_canvas_graph(
    graph_value: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Return the graph from the final Canvas turn, if it was recorded."""

    if graph_value is None:
        return None
    turns = graph_value.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return None
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        snapshot = turn.get("graph_snapshot")
        if isinstance(snapshot, Mapping):
            return dict(snapshot)
    return None


def _evaluated_graph(
    graph_value: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Return the exact graph whose answer/Runtime tuple was evaluated."""

    if graph_value is None:
        return None
    if graph_value.get("valid_lineage_fallback_used") is True:
        receipt = graph_value.get("valid_lineage_fallback_receipt")
        if isinstance(receipt, Mapping):
            snapshot = receipt.get("graph_snapshot")
            if isinstance(snapshot, Mapping):
                return dict(snapshot)
    return _terminal_canvas_graph(graph_value)


def _evaluated_graph_diagnostic_view(
    graph_value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project the evaluator graph into the existing read-only diagnostic."""

    evaluated = _evaluated_graph(graph_value)
    terminal = _terminal_canvas_graph(graph_value)
    if evaluated is None or evaluated == terminal:
        return graph_value
    turns = graph_value.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return graph_value
    projected_turns = [
        dict(turn) for turn in turns if isinstance(turn, Mapping)
    ]
    if not projected_turns:
        return graph_value
    projected_turns[-1]["graph_snapshot"] = dict(evaluated)
    projected = dict(graph_value)
    projected["turns"] = projected_turns
    return projected


def _first_action_agent_id(action: Mapping[str, Any]) -> Optional[str]:
    agent_id = action.get("agent_id")
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    agents = action.get("agents")
    if isinstance(agents, Sequence) and not isinstance(agents, (str, bytes)):
        for agent in agents:
            if isinstance(agent, Mapping):
                value = agent.get("agent_id")
                if isinstance(value, str) and value:
                    return value
    return None


def _aime_wrong_demo_diagnosis(
    graph_value: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Locate the first observable AIME failure in the saved trajectory.

    This is an offline receipt diagnosis.  It does not expose the target to a
    model and does not infer an unobserved reasoning error from the answer.
    """

    if graph_value is None:
        return {
            "diagnosis_scope": "first_observable_failure",
            "failure_layer": "runtime",
            "first_error_turn": None,
            "first_error_action": None,
            "first_error_agent_id": None,
            "error": "trajectory_missing",
            "subsequent_error_propagation": "no trajectory was persisted",
            "terminal_result": "collection_failed",
        }
    turns = graph_value.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        turns = ()
    first: Optional[dict[str, Any]] = None
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        action = turn.get("action")
        action = action if isinstance(action, Mapping) else {}
        action_name = action.get("action")
        action_name = action_name if isinstance(action_name, str) else None
        feedback = str(turn.get("canvas_feedback", ""))
        folded = feedback.casefold()
        round_index = turn.get("round_index")
        runtime_summary = turn.get("runtime_summary")
        runtime_summary = (
            runtime_summary if isinstance(runtime_summary, Mapping) else {}
        )
        failure_records = runtime_summary.get("failure_records", ())
        failure_records = (
            failure_records
            if isinstance(failure_records, Sequence)
            and not isinstance(failure_records, (str, bytes))
            else ()
        )
        first_failure_record = next(
            (value for value in failure_records if isinstance(value, Mapping)),
            None,
        )
        if not action_name:
            first = {
                "failure_layer": "director",
                "first_error_turn": round_index,
                "first_error_action": None,
                "first_error_agent_id": None,
                "error": "invalid_or_unparsed_director_action",
            }
        elif (
            "rejected" in folded
            or folded.startswith("[invalid]")
            or "invalid action:" in folded
            or "cannot finish:" in folded
            or "parse_error" in folded
            or "schema_invalid" in folded
        ):
            first = {
                "failure_layer": (
                    "director"
                    if "parse" in folded or "schema" in folded
                    else "graph"
                ),
                "first_error_turn": round_index,
                "first_error_action": action_name,
                "first_error_agent_id": _first_action_agent_id(action),
                "error": "canvas_action_rejected",
            }
        elif (
            runtime_summary.get("execution_status") == "failed"
            or first_failure_record is not None
        ):
            error_type = (
                first_failure_record.get("error_type")
                if isinstance(first_failure_record, Mapping)
                else None
            )
            folded_error_type = str(error_type or "").casefold()
            first = {
                "failure_layer": (
                    "tool" if "tool" in folded_error_type else "runtime"
                ),
                "first_error_turn": round_index,
                "first_error_action": action_name,
                "first_error_agent_id": (
                    first_failure_record.get("agent_id")
                    if isinstance(first_failure_record, Mapping)
                    else _first_action_agent_id(action)
                ),
                "error": error_type or "agent_or_provider_execution_failure",
            }
        elif any(
            marker in folded
            for marker in (
                "execution_error=",
                "agentruntimeerror",
                "gateway failed",
                "provider request failed",
                "tool execution failure",
            )
        ):
            first = {
                "failure_layer": "tool" if "tool" in folded else "runtime",
                "first_error_turn": round_index,
                "first_error_action": action_name,
                "first_error_agent_id": _first_action_agent_id(action),
                "error": "agent_or_provider_execution_failure",
            }
        if first is not None:
            break

    evaluation = graph_value.get("evaluation")
    details = evaluation.get("details", {}) if isinstance(evaluation, Mapping) else {}
    if first is None and graph_value.get("explicit_finish") is not True:
        first = {
            "failure_layer": "director",
            # This is a terminal-boundary failure after the final sampled
            # turn, not an invented Director turn.
            "first_error_turn": None,
            "first_error_action": "finish_absent",
            "first_error_agent_id": None,
            "error": "missing_explicit_finish",
            "failure_boundary": "terminal_boundary_after_last_turn",
            "last_observed_turn_index": (
                turns[-1].get("round_index")
                if turns and isinstance(turns[-1], Mapping)
                else None
            ),
        }
    if (
        first is None
        and isinstance(details, Mapping)
        and details.get("parsing_succeeded") is False
    ):
        final_graph = _evaluated_graph(graph_value) or {}
        first = {
            "failure_layer": "output_extraction",
            "first_error_turn": len(turns) - 1 if turns else None,
            "first_error_action": "finish",
            "first_error_agent_id": final_graph.get("output_agent_id"),
            "error": details.get("parsing_failure_reason") or "output_parsing_failure",
        }
    if first is None:
        final_graph = _evaluated_graph(graph_value) or {}
        first = {
            "failure_layer": "agent",
            "first_error_turn": None,
            "first_error_action": None,
            "first_error_agent_id": final_graph.get("output_agent_id"),
            "error": "incorrect_terminal_integer_without_observable_runtime_failure",
        }
    first_turn = first.get("first_error_turn")
    later_turns = (
        max(0, len(turns) - int(first_turn) - 1)
        if isinstance(first_turn, int)
        else None
    )
    return {
        "diagnosis_scope": "first_observable_failure",
        **first,
        "subsequent_error_propagation": {
            "interpretation": "subsequent_receipt_span_not_proven_causality",
            "later_turn_count": later_turns,
            "final_answer": graph_value.get("final_answer"),
            "evaluator_reason": (
                evaluation.get("reason") if isinstance(evaluation, Mapping) else None
            ),
        },
        "terminal_result": graph_value.get("termination_reason"),
    }


def _webshop_wrong_demo_diagnosis(
    graph_value: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Classify only the first failure visible in saved WebShop receipts.

    The native reward and hidden goal are never model-visible.  This offline
    report adapter may identify an invalid public action, missing environment
    termination, evaluator failure, or a below-full terminal score; it does
    not infer which earlier shopping decision was causally wrong.
    """

    generic = dict(_aime_wrong_demo_diagnosis(graph_value))
    if graph_value is None:
        return generic
    evaluation = graph_value.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    trace = _saved_environment_trace(graph_value)
    invalid = next(
        (
            entry
            for entry in trace
            if isinstance(entry, Mapping)
            and (
                entry.get("parse_error") is True
                or entry.get("action") == "<INVALID>"
                or entry.get("state_advanced") is False
            )
        ),
        None,
    )
    if invalid is not None:
        return {
            "diagnosis_scope": "first_observable_failure",
            "failure_layer": "environment_action",
            "first_error_turn": invalid.get("step"),
            "first_error_action": invalid.get("action"),
            "first_error_agent_id": None,
            "error": "invalid_native_environment_action",
            "subsequent_error_propagation": {
                "interpretation": "subsequent_receipt_span_not_proven_causality",
                "later_environment_turn_count": max(
                    0, len(trace) - int(invalid.get("step", 0)) - 1
                ),
            },
            "terminal_result": graph_value.get("termination_reason"),
        }
    # Reuse the existing graph/runtime/provider diagnosis whenever it found a
    # concrete observable fault instead of its AIME-only semantic fallback.
    if generic.get("error") != "incorrect_terminal_integer_without_observable_runtime_failure":
        return generic
    metrics = evaluation.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    reason = evaluation.get("reason")
    terminal = float(metrics.get("terminal", 0.0) or 0.0) == 1.0
    score = metrics.get("average_score", metrics.get("environment_return"))
    if evaluation.get("valid") is not True:
        layer = "environment_runtime"
        error = str(reason or "environment_evaluator_failure")
    elif not terminal or reason == "environment_step_limit":
        layer = "environment_termination"
        error = "environment_step_limit_without_terminal"
    else:
        layer = "environment_outcome"
        error = "native_terminal_score_below_full"
    return {
        "diagnosis_scope": "first_observable_failure",
        "failure_layer": layer,
        "first_error_turn": (len(trace) - 1) if trace else None,
        "first_error_action": (
            trace[-1].get("action") if trace and isinstance(trace[-1], Mapping) else None
        ),
        "first_error_agent_id": None,
        "error": error,
        "native_average_score": score,
        "causal_decision_identified": False,
        "subsequent_error_propagation": {
            "interpretation": (
                "the first causally wrong shopping decision is not observable "
                "from terminal reward alone"
            ),
            "environment_turn_count": len(trace),
        },
        "terminal_result": graph_value.get("termination_reason"),
    }


def _saved_environment_trace(
    value: Optional[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return the formal trace or the latest saved public execution prefix."""

    if not isinstance(value, Mapping):
        return ()
    evaluation = value.get("evaluation")
    details = evaluation.get("details") if isinstance(evaluation, Mapping) else None
    raw_trace = details.get("trace") if isinstance(details, Mapping) else None
    if isinstance(raw_trace, Sequence) and not isinstance(raw_trace, (str, bytes)):
        trace = tuple(dict(item) for item in raw_trace if isinstance(item, Mapping))
        if trace:
            return trace

    turns = value.get("turns", ())
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        return ()
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        runtime = turn.get("runtime_summary")
        output_metadata = (
            runtime.get("output_metadata") if isinstance(runtime, Mapping) else None
        )
        if not isinstance(output_metadata, Mapping):
            continue
        candidates: list[tuple[Mapping[str, Any], ...]] = []
        for metadata in output_metadata.values():
            if not isinstance(metadata, Mapping):
                continue
            raw_candidate = metadata.get("evaluator_environment_trace")
            if not isinstance(raw_candidate, Sequence) or isinstance(
                raw_candidate, (str, bytes)
            ):
                continue
            candidate = tuple(
                dict(item) for item in raw_candidate if isinstance(item, Mapping)
            )
            if candidate:
                candidates.append(candidate)
        if candidates:
            # AgentRuntime admits only one owner of a stateful environment
            # resource.  Longest is a deterministic fail-closed reporting
            # projection if malformed legacy metadata contains duplicates.
            return max(candidates, key=len)
    return ()


def _execution_environment_receipts(
    value: Optional[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    """Return every saved public environment episode in one rollout.

    The formal evaluator replay trace describes the single episode that
    determines reward.  FlowSteer's execute-on-edit loop may additionally run
    request-scoped WebShop episodes while the Director edits the Canvas.  Those
    public receipts are diagnostic execution cost and must not be conflated
    with the formal evaluator episode.
    """

    if not isinstance(value, Mapping):
        return ()
    episodes: list[tuple[Mapping[str, Any], ...]] = []
    seen_artifact_ids: set[str] = set()

    def append_from_metadata(metadata: object) -> bool:
        if not isinstance(metadata, Mapping):
            return False
        receipts = metadata.get("environment_receipts")
        if not isinstance(receipts, Sequence) or isinstance(
            receipts, (str, bytes)
        ):
            return False
        episode = tuple(dict(item) for item in receipts if isinstance(item, Mapping))
        if not episode:
            return False
        artifact_id = metadata.get("artifact_id")
        if isinstance(artifact_id, str) and artifact_id.strip():
            normalized_artifact_id = artifact_id.strip()
            if normalized_artifact_id in seen_artifact_ids:
                return True
            seen_artifact_ids.add(normalized_artifact_id)
        episodes.append(episode)
        return True

    def append_from_execution(execution: object) -> bool:
        if not isinstance(execution, Mapping):
            return False
        metadata = execution.get("metadata")
        response = metadata.get("response") if isinstance(metadata, Mapping) else None
        return append_from_metadata(response)

    direct_executions = value.get("executions")
    if isinstance(direct_executions, Sequence) and not isinstance(
        direct_executions, (str, bytes)
    ):
        for execution in direct_executions:
            append_from_execution(execution)

    turns = value.get("turns")
    if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes)):
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            found_execution_episode = False
            executions = turn.get("executions")
            if isinstance(executions, Sequence) and not isinstance(
                executions, (str, bytes)
            ):
                for execution in executions:
                    found_execution_episode = (
                        append_from_execution(execution) or found_execution_episode
                    )
            if found_execution_episode:
                continue
            # Cancellation/failure prefixes can survive only in Runtime output
            # metadata after the execution Task has been normalized.
            runtime = turn.get("runtime_summary")
            output_metadata = (
                runtime.get("output_metadata") if isinstance(runtime, Mapping) else None
            )
            if not isinstance(output_metadata, Mapping):
                continue
            for metadata in output_metadata.values():
                append_from_metadata(metadata)
    return tuple(episodes)


def _rollout_environment_diagnostics(
    value: Optional[Mapping[str, Any]],
) -> Mapping[str, int]:
    """Summarize all public environment attempts across execute-on-edit."""

    episodes = _execution_environment_receipts(value)
    # Direct evaluation owns one continuous evaluator episode, so its policy
    # calls do not carry EnvironmentExecutionAdapter receipts.  In that arm the
    # formal trace is also the complete rollout trace.
    if not episodes:
        formal = _saved_environment_trace(value)
        episodes = (formal,) if formal else ()
    attempt_count = 0
    state_advancing = 0
    invalid = 0
    terminal_episodes = 0
    for episode in episodes:
        attempt_count += len(episode)
        terminal_episodes += int(
            any(
                isinstance(item, Mapping)
                and (
                    item.get("terminal") is True
                    or item.get("done") is True
                )
                for item in episode
            )
        )
        for item in episode:
            if not isinstance(item, Mapping):
                continue
            item_invalid = (
                item.get("parse_error") is True
                or item.get("observation_status") == "parse_error"
                or item.get("action") in {None, "<INVALID>"}
                or item.get("state_advanced") is False
            )
            invalid += int(item_invalid)
            state_advancing += int(not item_invalid)
    return {
        "episode_count": len(episodes),
        "environment_attempt_count": attempt_count,
        "state_advancing_action_count": state_advancing,
        "invalid_action_count": invalid,
        "terminal_episode_count": terminal_episodes,
    }


def _direct_execution_receipt_projection(
    value: Optional[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Keep the provider receipt fields needed by the aggregate report."""

    if not isinstance(value, Mapping):
        return ()
    recorded = value.get("executions")
    fallback = value.get("execution")
    executions = (
        [item for item in recorded if isinstance(item, Mapping)]
        if isinstance(recorded, Sequence) and not isinstance(recorded, (str, bytes))
        else [fallback]
        if isinstance(fallback, Mapping)
        else []
    )
    projected: list[Mapping[str, Any]] = []
    for execution in executions:
        metadata = execution.get("metadata")
        response = metadata.get("response") if isinstance(metadata, Mapping) else None
        projected.append(
            {
                "model_id": execution.get("model_id"),
                "provider": execution.get("provider"),
                "error_type": execution.get("error_type"),
                "response": {
                    key: response.get(key)
                    for key in (
                        "provider_id",
                        "provider_model",
                        "finish_reason",
                        "attempt_count",
                    )
                    if isinstance(response, Mapping) and key in response
                },
            }
        )
    return tuple(projected)


def _paired_rows(
    selected: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    dataset_key: str,
) -> list[dict[str, Any]]:
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    metric_names = tuple(_BENCHMARKS[dataset_key]["metric_names"])
    rows: list[dict[str, Any]] = []
    for task in selected:
        direct_value = direct.get(task.task_id)
        graph_value = trajectories.get(task.task_id)
        direct_valid, direct_metrics = _metrics(direct_value, dataset_key)
        graph_valid, graph_metrics = _metrics(graph_value, dataset_key)
        direct_score = direct_metrics[metric_name]
        graph_score = graph_metrics[metric_name]
        direct_execution = direct_value.get("execution") if direct_value else None
        rows.append(
            {
                "schema_version": "flowsteer.completion_benchmark.paired_result.v1",
                "dataset_key": dataset_key,
                "task_id": task.task_id,
                "question": task.question,
                "ground_truth": task.ground_truth,
                "primary_metric": metric_name,
                "direct": {
                    "available": direct_value is not None,
                    "valid": direct_valid,
                    "final_answer": (
                        direct_value.get("final_answer") if direct_value else None
                    ),
                    **{name: direct_metrics[name] for name in metric_names},
                    "evaluation": (
                        direct_value.get("evaluation") if direct_value else None
                    ),
                    "telemetry": _direct_telemetry(direct_value),
                    "execution": direct_execution,
                    "execution_receipts": [
                        dict(item)
                        for item in _direct_execution_receipt_projection(direct_value)
                    ],
                    "environment_trace": [
                        dict(item) for item in _saved_environment_trace(direct_value)
                    ],
                    "rollout_environment_diagnostics": dict(
                        _rollout_environment_diagnostics(direct_value)
                    ),
                },
                "agentgraph": {
                    "available": graph_value is not None,
                    "valid": graph_valid,
                    "final_answer": (
                        graph_value.get("final_answer") if graph_value else None
                    ),
                    **{name: graph_metrics[name] for name in metric_names},
                    "evaluation": (
                        graph_value.get("evaluation") if graph_value else None
                    ),
                    "explicit_finish": (
                        graph_value.get("explicit_finish") if graph_value else False
                    ),
                    "termination_reason": (
                        graph_value.get("termination_reason")
                        if graph_value
                        else "collection_failed"
                    ),
                    "trajectory_id": (
                        graph_value.get("trajectory_id") if graph_value else None
                    ),
                    "final_graph": _evaluated_graph(graph_value),
                    "terminal_canvas_graph": _terminal_canvas_graph(graph_value),
                    "output_agent_inbox": _output_inbox(graph_value),
                    "telemetry": _graph_telemetry(graph_value),
                    "graph_diagnostic": (
                        diagnose_trajectory(
                            _evaluated_graph_diagnostic_view(graph_value)
                        ).to_dict()
                        if graph_value is not None
                        else None
                    ),
                    "environment_trace": [
                        dict(item) for item in _saved_environment_trace(graph_value)
                    ],
                    "rollout_environment_diagnostics": dict(
                        _rollout_environment_diagnostics(graph_value)
                    ),
                },
                **{
                    f"delta_{name}": graph_metrics[name] - direct_metrics[name]
                    for name in metric_names
                },
                "failure_type": _failure_type(
                    direct_value,
                    graph_value,
                    direct_valid=direct_valid,
                    graph_valid=graph_valid,
                    direct_score=direct_score,
                    graph_score=graph_score,
                    dataset_key=dataset_key,
                ),
                "wrong_demo_diagnosis": (
                    _aime_wrong_demo_diagnosis(graph_value)
                    if dataset_key == "aime_2026" and graph_score < 1.0
                    else _webshop_wrong_demo_diagnosis(graph_value)
                    if dataset_key == "webshop" and graph_score < 1.0
                    else None
                ),
            }
        )
    return rows


def _environment_arm_diagnostics(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> Mapping[str, int]:
    """Aggregate public environment transitions without exposing hidden state."""

    evaluator_valid = 0
    terminal_episodes = 0
    step_limit_episodes = 0
    environment_timeouts = 0
    environment_turns = 0
    state_advancing_actions = 0
    invalid_actions = 0
    formal_environment_turns = 0
    formal_state_advancing_actions = 0
    formal_invalid_actions = 0
    saved_prefix_turns = 0
    saved_prefix_state_advancing_actions = 0
    saved_prefix_invalid_actions = 0
    environment_or_evaluator_failures = 0
    formal_evaluator_skipped = 0
    rollout_episode_count = 0
    rollout_environment_attempts = 0
    rollout_state_advancing_actions = 0
    rollout_invalid_actions = 0
    rollout_terminal_episodes = 0
    for row in rows:
        arm = row.get(condition)
        arm = arm if isinstance(arm, Mapping) else {}
        evaluation = arm.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        skipped = _is_reportable_terminal_failure(arm)
        evaluation_valid = evaluation.get("valid") is True
        if evaluation_valid:
            evaluator_valid += 1
        elif skipped:
            formal_evaluator_skipped += 1
        else:
            environment_or_evaluator_failures += 1
        reason = str(evaluation.get("reason", ""))
        if reason == "environment_step_limit":
            step_limit_episodes += 1
        if "timeout" in reason.casefold():
            environment_timeouts += 1
        metrics = evaluation.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        if float(metrics.get("terminal", 0.0) or 0.0) == 1.0:
            terminal_episodes += 1
        raw_trace = arm.get("environment_trace")
        if not isinstance(raw_trace, Sequence) or isinstance(
            raw_trace, (str, bytes)
        ):
            raw_trace = _saved_environment_trace(arm)
        trace = (
            raw_trace
            if isinstance(raw_trace, Sequence)
            and not isinstance(raw_trace, (str, bytes))
            else ()
        )
        environment_turns += len(trace)
        if evaluation_valid:
            formal_environment_turns += len(trace)
        else:
            saved_prefix_turns += len(trace)
        for entry in trace:
            if not isinstance(entry, Mapping):
                continue
            invalid = (
                entry.get("parse_error") is True
                or entry.get("action") == "<INVALID>"
                or entry.get("state_advanced") is False
            )
            invalid_actions += int(invalid)
            state_advancing_actions += int(not invalid)
            if evaluation_valid:
                formal_invalid_actions += int(invalid)
                formal_state_advancing_actions += int(not invalid)
            else:
                saved_prefix_invalid_actions += int(invalid)
                saved_prefix_state_advancing_actions += int(not invalid)
        rollout = arm.get("rollout_environment_diagnostics")
        if not isinstance(rollout, Mapping):
            rollout = _rollout_environment_diagnostics(arm)
        rollout_episode_count += int(rollout.get("episode_count", 0) or 0)
        rollout_environment_attempts += int(
            rollout.get("environment_attempt_count", 0) or 0
        )
        rollout_state_advancing_actions += int(
            rollout.get("state_advancing_action_count", 0) or 0
        )
        rollout_invalid_actions += int(rollout.get("invalid_action_count", 0) or 0)
        rollout_terminal_episodes += int(
            rollout.get("terminal_episode_count", 0) or 0
        )
    return {
        "evaluator_valid_count": evaluator_valid,
        "terminal_episode_count": terminal_episodes,
        "environment_step_limit_count": step_limit_episodes,
        "environment_timeout_count": environment_timeouts,
        "environment_turn_count": environment_turns,
        "state_advancing_action_count": state_advancing_actions,
        "invalid_action_count": invalid_actions,
        "formal_environment_action_count": formal_environment_turns,
        "formal_state_advancing_action_count": formal_state_advancing_actions,
        "formal_invalid_action_count": formal_invalid_actions,
        "saved_prefix_action_count": saved_prefix_turns,
        "saved_prefix_state_advancing_action_count": (
            saved_prefix_state_advancing_actions
        ),
        "saved_prefix_invalid_action_count": saved_prefix_invalid_actions,
        "rollout_environment_episode_count": rollout_episode_count,
        "rollout_environment_attempt_count": rollout_environment_attempts,
        "rollout_state_advancing_action_count": rollout_state_advancing_actions,
        "rollout_invalid_action_count": rollout_invalid_actions,
        "rollout_terminal_episode_count": rollout_terminal_episodes,
        "formal_evaluator_skipped_count": formal_evaluator_skipped,
        "environment_or_evaluator_failure_count": environment_or_evaluator_failures,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]], condition: str, dataset_key: str
) -> Mapping[str, Any]:
    metric_names = tuple(_BENCHMARKS[dataset_key]["metric_names"])
    total = len(rows)
    values = [row[condition] for row in rows]
    valid = [value for value in values if value.get("valid") is True]
    result = {
        "denominator": total,
        "completed": sum(value.get("available") is True for value in values),
        "evaluator_valid": len(valid),
    }
    for metric_name in metric_names:
        strict = (
            sum(float(value.get(metric_name, 0.0)) for value in values) / total
            if total
            else 0.0
        )
        completed_only = (
            sum(float(value.get(metric_name, 0.0)) for value in valid) / len(valid)
            if valid
            else None
        )
        result[f"strict_{metric_name}"] = strict
        result[f"completed_only_{metric_name}"] = completed_only
    if dataset_key == "aime_2026":
        result["correct"] = sum(
            float(value.get("accuracy", 0.0)) == 1.0 for value in values
        )
    return result


def _telemetry_totals(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> Mapping[str, float | int]:
    fields = ("api_attempts", "input_tokens", "output_tokens", "latency_ms")
    return {
        field: sum(
            float(row[condition].get("telemetry", {}).get(field, 0.0))
            for row in rows
        )
        for field in fields
    }


def _direct_execution_receipts(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    model_ids: Counter[str] = Counter()
    provider_ids: Counter[str] = Counter()
    provider_models: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    attempt_count = 0
    call_count = 0
    terminal_output_parsing_failure_count = 0
    for row in rows:
        direct = row.get("direct")
        execution = direct.get("execution") if isinstance(direct, Mapping) else None
        receipt_projection = (
            direct.get("execution_receipts") if isinstance(direct, Mapping) else None
        )
        executions = (
            [item for item in receipt_projection if isinstance(item, Mapping)]
            if isinstance(receipt_projection, Sequence)
            and not isinstance(receipt_projection, (str, bytes))
            else [execution]
            if isinstance(execution, Mapping)
            else []
        )
        evaluation = direct.get("evaluation") if isinstance(direct, Mapping) else None
        details = (
            evaluation.get("details") if isinstance(evaluation, Mapping) else None
        )
        if isinstance(details, Mapping) and details.get("parsing_succeeded") is False:
            terminal_output_parsing_failure_count += 1
        for receipt in executions:
            call_count += 1
            model_id = receipt.get("model_id")
            provider_id = receipt.get("provider")
            if isinstance(model_id, str) and model_id:
                model_ids[model_id] += 1
            if isinstance(provider_id, str) and provider_id:
                provider_ids[provider_id] += 1
            error_type = receipt.get("error_type")
            if isinstance(error_type, str) and error_type:
                error_types[error_type] += 1
            metadata = receipt.get("metadata")
            response = (
                receipt.get("response")
                if isinstance(receipt.get("response"), Mapping)
                else metadata.get("response")
                if isinstance(metadata, Mapping)
                else None
            )
            if isinstance(response, Mapping):
                response_provider = response.get("provider_id")
                response_model = response.get("provider_model")
                finish_reason = response.get("finish_reason")
                if isinstance(response_provider, str) and response_provider:
                    provider_ids[response_provider] += int(
                        not (
                            isinstance(provider_id, str)
                            and provider_id == response_provider
                        )
                    )
                if isinstance(response_model, str) and response_model:
                    provider_models[response_model] += 1
                if isinstance(finish_reason, str) and finish_reason:
                    finish_reasons[finish_reason] += 1
                raw_attempts = response.get("attempt_count", 1)
                if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool):
                    attempt_count += raw_attempts
    return {
        "call_count": call_count,
        "provider_attempt_count": attempt_count,
        "model_id_distribution": dict(sorted(model_ids.items())),
        "provider_id_distribution": dict(sorted(provider_ids.items())),
        "provider_model_distribution": dict(sorted(provider_models.items())),
        "error_type_distribution": dict(sorted(error_types.items())),
        "finish_reason_distribution": dict(sorted(finish_reasons.items())),
        "terminal_output_parsing_failure_count": (
            terminal_output_parsing_failure_count
        ),
    }


def _agentgraph_execution_receipts(
    trajectories: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    node_models: Counter[str] = Counter()
    executor_models: Counter[str] = Counter()
    provider_ids: Counter[str] = Counter()
    provider_models: Counter[str] = Counter()
    execution_error_types: Counter[str] = Counter()
    runtime_failure_types: Counter[str] = Counter()
    execution_call_count = 0
    provider_attempt_count = 0
    director_attempt_count = 0
    director_latency_ms = 0.0
    runtime_failed_turn_count = 0
    runtime_failed_without_structured_failure_record_count = 0

    for trajectory in trajectories:
        graph = _evaluated_graph(trajectory)
        nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
        if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
            for node in nodes:
                model_id = node.get("model_id") if isinstance(node, Mapping) else None
                if isinstance(model_id, str) and model_id:
                    node_models[model_id] += 1

        turns = trajectory.get("turns")
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            raw_director_attempts = turn.get("director_attempt_count", 0)
            if isinstance(raw_director_attempts, int) and not isinstance(
                raw_director_attempts, bool
            ):
                director_attempt_count += raw_director_attempts
            raw_director_latency = turn.get("director_latency_ms", 0.0)
            if isinstance(raw_director_latency, (int, float)) and not isinstance(
                raw_director_latency, bool
            ):
                director_latency_ms += float(raw_director_latency)

            executions = turn.get("executions")
            if isinstance(executions, Sequence) and not isinstance(
                executions, (str, bytes)
            ):
                for execution in executions:
                    if not isinstance(execution, Mapping):
                        continue
                    execution_call_count += 1
                    model_id = execution.get("model_id")
                    provider_id = execution.get("provider")
                    if isinstance(model_id, str) and model_id:
                        executor_models[model_id] += 1
                    if isinstance(provider_id, str) and provider_id:
                        provider_ids[provider_id] += 1
                    error_type = execution.get("error_type")
                    if isinstance(error_type, str) and error_type:
                        execution_error_types[error_type] += 1
                    metadata = execution.get("metadata")
                    response = (
                        metadata.get("response")
                        if isinstance(metadata, Mapping)
                        else None
                    )
                    if isinstance(response, Mapping):
                        response_provider = response.get("provider_id")
                        response_model = response.get("provider_model")
                        if (
                            isinstance(response_provider, str)
                            and response_provider
                            and response_provider != provider_id
                        ):
                            provider_ids[response_provider] += 1
                        if isinstance(response_model, str) and response_model:
                            provider_models[response_model] += 1
                        raw_attempts = response.get("attempt_count", 1)
                        if isinstance(raw_attempts, int) and not isinstance(
                            raw_attempts, bool
                        ):
                            provider_attempt_count += raw_attempts

            runtime = turn.get("runtime_summary")
            if not isinstance(runtime, Mapping) or runtime.get("execution_status") != "failed":
                continue
            runtime_failed_turn_count += 1
            failure_records = runtime.get("failure_records")
            if not isinstance(failure_records, Sequence) or isinstance(
                failure_records, (str, bytes)
            ) or not failure_records:
                runtime_failed_without_structured_failure_record_count += 1
                continue
            for failure in failure_records:
                if not isinstance(failure, Mapping):
                    runtime_failure_types["untyped_runtime_failure"] += 1
                    continue
                failure_type = next(
                    (
                        failure.get(key)
                        for key in ("error_type", "failure_type", "public_error_code")
                        if isinstance(failure.get(key), str) and failure.get(key)
                    ),
                    "untyped_runtime_failure",
                )
                runtime_failure_types[str(failure_type)] += 1

    return {
        "reported_graph_node_model_distribution": dict(sorted(node_models.items())),
        "reported_graph_scope": (
            "evaluated_graph_for_explicit_finish_else_terminal_canvas_graph"
        ),
        "executor_call_count": execution_call_count,
        "executor_model_distribution": dict(sorted(executor_models.items())),
        "provider_attempt_count": provider_attempt_count,
        "provider_id_distribution": dict(sorted(provider_ids.items())),
        "provider_model_distribution": dict(sorted(provider_models.items())),
        "execution_error_type_distribution": dict(
            sorted(execution_error_types.items())
        ),
        "director_attempt_count": director_attempt_count,
        "director_latency_ms": director_latency_ms,
        "runtime_failed_turn_count": runtime_failed_turn_count,
        "runtime_failure_type_distribution": dict(
            sorted(runtime_failure_types.items())
        ),
        "runtime_failed_without_structured_failure_record_count": (
            runtime_failed_without_structured_failure_record_count
        ),
    }


def _is_reportable_terminal_failure(value: Mapping[str, Any]) -> bool:
    """Recognize a saved terminal Canvas that correctly skipped evaluation."""

    evaluation = value.get("evaluation")
    details = evaluation.get("details") if isinstance(evaluation, Mapping) else None
    return bool(
        value.get("available") is True
        and value.get("valid") is False
        and value.get("explicit_finish") is False
        and value.get("termination_reason")
        in {"max_rounds", "canvas_action_domain_exhausted"}
        and isinstance(evaluation, Mapping)
        and evaluation.get("valid") is False
        and evaluation.get("reward") is None
        and evaluation.get("reason")
        == "not_evaluated_without_explicit_finish"
        and isinstance(details, Mapping)
        and details.get("formal_evaluator_called") is False
    )


def _report(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]] = (),
    *,
    collection_failures: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    _, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    specification = _BENCHMARKS[dataset_key]
    metric_name = str(specification["primary_metric"])
    metric_names = tuple(specification["metric_names"])
    direct = _aggregate(rows, "direct", dataset_key)
    graph = _aggregate(rows, "agentgraph", dataset_key)
    strict_key = f"strict_{metric_name}"
    failure_counts = Counter(str(row["failure_type"]) for row in rows)
    below_full = [
        row
        for row in rows
        if float(row["agentgraph"].get(metric_name, 0.0)) < 1.0
    ]
    graph_diagnostics = [
        row["agentgraph"].get("graph_diagnostic")
        for row in rows
        if isinstance(row["agentgraph"].get("graph_diagnostic"), Mapping)
    ]
    agent_count_distribution = Counter(
        int(value.get("agent_count", 0)) for value in graph_diagnostics
    )
    topology_distribution = Counter(
        str(value.get("topology_family", "unknown")) for value in graph_diagnostics
    )
    relation_count_distribution = Counter(
        int(value.get("relation_count", 0)) for value in graph_diagnostics
    )
    parsing_failure_count = sum(
        isinstance(row["agentgraph"].get("evaluation"), Mapping)
        and isinstance(row["agentgraph"]["evaluation"].get("details"), Mapping)
        and row["agentgraph"]["evaluation"]["details"].get(
            "parsing_succeeded"
        )
        is False
        for row in rows
    )
    return {
        "schema_version": "flowsteer.completion_benchmark.round_report.v1",
        "dataset_key": dataset_key,
        "dataset": specification["label"],
        "project_split": str(bounded["split"]),
        "benchmark_slice": (
            _mapping(config["evaluation"], "evaluation").get(
                "swebench_dataset_source"
            )
            if dataset_key == "swe_bench"
            else bounded.get("benchmark_slice")
        ),
        "sample_count": len(rows),
        "primary_metric": metric_name,
        "protocol_equivalent_to_direct": bool(
            bounded.get("protocol_equivalent_to_direct", False)
        ),
        "comparison_interpretation": (
            "paired_architecture_comparison"
            if bounded.get("protocol_equivalent_to_direct") is True
            else "separate_protocol_descriptive_comparison"
        ),
        "metric_scope": (
            "HotpotQA_official_normalization_exact_match_and_token_F1"
            if dataset_key == "hotpotqa"
            else "TriviaQA_official_normalization_exact_match_and_token_F1"
            if dataset_key == "triviaqa"
            else "SkillEval_canonicalized_integer_exact_accuracy"
            if dataset_key == "aime_2026"
            else "OpenAI_simple_evals_HealthBench_rubric_raw_score"
            if dataset_key == "healthbench_professional"
            else (
                "SWE_bench_regular_dev_official_Docker_harness_resolved_rate"
                if _mapping(config["evaluation"], "evaluation").get(
                    "swebench_dataset_source"
                )
                == "regular_dev"
                else "SWE_bench_Verified_official_Docker_harness_resolved_rate"
            )
            if dataset_key == "swe_bench"
            else "WebShop_official_environment_Average_Score_and_Success_Rate"
            if dataset_key == "webshop"
            else "SkillFlow_RAGEN_official_environment_terminal_success"
        ),
        "direct_local_baseline": direct,
        "agentgraph": graph,
        "agentgraph_minus_direct": {
            name: float(graph[f"strict_{name}"])
            - float(direct[f"strict_{name}"])
            for name in metric_names
        },
        "graph_search_diagnostics": aggregate_trajectory_diagnostics(
            _evaluated_graph_diagnostic_view(value) for value in trajectories
        ),
        "agent_count_distribution": {
            str(key): value for key, value in sorted(agent_count_distribution.items())
        },
        "relation_count_distribution": {
            str(key): value
            for key, value in sorted(relation_count_distribution.items())
        },
        "topology_distribution": dict(sorted(topology_distribution.items())),
        "telemetry_totals": {
            "direct": _telemetry_totals(rows, "direct"),
            "agentgraph": _telemetry_totals(rows, "agentgraph"),
        },
        "environment_diagnostics": (
            {
                "direct": _environment_arm_diagnostics(rows, "direct"),
                "agentgraph": _environment_arm_diagnostics(rows, "agentgraph"),
            }
            if dataset_key in _INTERACTIVE_BENCHMARKS
            else None
        ),
        "execution_receipts": {
            "direct": _direct_execution_receipts(rows),
            "agentgraph": _agentgraph_execution_receipts(trajectories),
            "collection_failures": [
                {
                    key: failure.get(key)
                    for key in ("task_id", "condition", "stage", "error", "recorded_at")
                    if key in failure
                }
                for failure in collection_failures
            ],
        },
        "failure_types": dict(sorted(failure_counts.items())),
        "below_full_score_demo_count": len(below_full),
        "typical_below_full_score_task_ids": [
            row["task_id"] for row in below_full[:10]
        ],
        "explicit_finished_count": sum(
            row["agentgraph"].get("explicit_finish") is True for row in rows
        ),
        "terminal_failure_count": sum(
            row["agentgraph"].get("available") is True
            and row["agentgraph"].get("explicit_finish") is not True
            for row in rows
        ),
        "max_rounds_count": sum(
            row["agentgraph"].get("termination_reason") == "max_rounds"
            for row in rows
        ),
        "direct_parsing_failure_count": sum(
            isinstance(row["direct"].get("evaluation"), Mapping)
            and isinstance(row["direct"]["evaluation"].get("details"), Mapping)
            and row["direct"]["evaluation"]["details"].get(
                "parsing_succeeded"
            )
            is False
            for row in rows
        ),
        "parsing_failure_count": parsing_failure_count,
        "operational_failure_count": sum(
            row["direct"].get("available") is not True
            or row["direct"].get("valid") is not True
            or row["agentgraph"].get("available") is not True
            or (
                row["agentgraph"].get("valid") is not True
                and not _is_reportable_terminal_failure(row["agentgraph"])
            )
            for row in rows
        ),
        "policy_version": str(config["director"]["behavior_policy_version"]),
        "policy_adapter": config["director"].get("behavior_adapter_name"),
        "model_catalog_path": str(config["agent_graph"]["model_catalog_path"]),
        "training_performed": False,
        "skill_injection_performed": bool(config.get("skills", {}).get("enabled", False)),
        "skill_evaluation_mode": (
            "memory_on_active_only"
            if bool(config.get("skills", {}).get("enabled", False))
            else "memory_off"
        ),
        "skill_store_path": config.get("skills", {}).get("store_path"),
        "completed_at": _utc_now(),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    metric_name = str(report["primary_metric"])
    strict_key = f"strict_{metric_name}"
    direct = report["direct_local_baseline"]
    graph = report["agentgraph"]
    delta = report["agentgraph_minus_direct"][metric_name]
    failures = "\n".join(
        f"- `{name}`: {count}" for name, count in report["failure_types"].items()
    ) or "- None"
    skill_sentence = (
        "Only evidence-gated ACTIVE Skills were retrieved as rejectable prompt priors."
        if report.get("skill_injection_performed")
        else "No Skill was injected."
    )
    protocol_sentence = (
        "Direct and AgentGraph use protocol-equivalent task/evaluator conditions."
        if report.get("protocol_equivalent_to_direct") is True
        else "Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate."
    )
    if report["dataset_key"] == "webshop":
        environment = report.get("environment_diagnostics") or {}
        direct_environment = environment.get("direct", {})
        graph_environment = environment.get("agentgraph", {})
        execution_receipts = report.get("execution_receipts") or {}
        direct_receipts = execution_receipts.get("direct", {})
        graph_receipts = execution_receipts.get("agentgraph", {})
        collection_failures = execution_receipts.get("collection_failures", [])
        return f"""# WebShop Architecture Validation

Fixed {report['project_split']} samples: **{report['sample_count']}**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. {skill_sentence}

Native evaluator: **WebShop Average Score** and **Success Rate** (`{report['metric_scope']}`). AgentGraph explicit FINISH: **{report['explicit_finished_count']}/{report['sample_count']}**; terminal failures: **{report['terminal_failure_count']}**; operational/evaluator failures: **{report['operational_failure_count']}**.

| Condition | Completed | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | {direct['completed']} | {direct['evaluator_valid']} | {100 * float(direct['strict_average_score']):.2f} | {100 * float(direct['strict_success_rate']):.2f}% |
| AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * float(graph['strict_average_score']):.2f} | {100 * float(graph['strict_success_rate']):.2f}% |

AgentGraph - Direct: **{100 * float(report['agentgraph_minus_direct']['average_score']):+.2f} Average Score**, **{100 * float(report['agentgraph_minus_direct']['success_rate']):+.2f} percentage points Success Rate**.

{protocol_sentence}

## Formal evaluator episode

| Condition | Formal actions | State-advancing actions | Invalid actions | Saved non-formal prefix actions | Terminal episodes | Step-limit episodes | Evaluator skipped (no FINISH) | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | {direct_environment.get('formal_environment_action_count', 0)} | {direct_environment.get('formal_state_advancing_action_count', 0)} | {direct_environment.get('formal_invalid_action_count', 0)} | {direct_environment.get('saved_prefix_action_count', 0)} | {direct_environment.get('terminal_episode_count', 0)} | {direct_environment.get('environment_step_limit_count', 0)} | {direct_environment.get('formal_evaluator_skipped_count', 0)} | {direct_environment.get('environment_timeout_count', 0)} |
| AgentGraph | {graph_environment.get('formal_environment_action_count', 0)} | {graph_environment.get('formal_state_advancing_action_count', 0)} | {graph_environment.get('formal_invalid_action_count', 0)} | {graph_environment.get('saved_prefix_action_count', 0)} | {graph_environment.get('terminal_episode_count', 0)} | {graph_environment.get('environment_step_limit_count', 0)} | {graph_environment.get('formal_evaluator_skipped_count', 0)} | {graph_environment.get('environment_timeout_count', 0)} |

## Full rollout environment execution

| Condition | Request-scoped episodes | Action attempts | State-advancing actions | Invalid actions | Terminal episodes |
|---|---:|---:|---:|---:|---:|
| Direct | {direct_environment.get('rollout_environment_episode_count', 0)} | {direct_environment.get('rollout_environment_attempt_count', 0)} | {direct_environment.get('rollout_state_advancing_action_count', 0)} | {direct_environment.get('rollout_invalid_action_count', 0)} | {direct_environment.get('rollout_terminal_episode_count', 0)} |
| AgentGraph | {graph_environment.get('rollout_environment_episode_count', 0)} | {graph_environment.get('rollout_environment_attempt_count', 0)} | {graph_environment.get('rollout_state_advancing_action_count', 0)} | {graph_environment.get('rollout_invalid_action_count', 0)} | {graph_environment.get('rollout_terminal_episode_count', 0)} |

## Natural AgentGraph structure

- Agent count distribution: `{report['agent_count_distribution']}`
- Relation count distribution: `{report['relation_count_distribution']}`
- Topology distribution: `{report['topology_distribution']}`
- Director `max_rounds`: **{report['max_rounds_count']}**
- Runtime failed turns: **{graph_receipts.get('runtime_failed_turn_count', 0)}**
- Runtime failure types: `{graph_receipts.get('runtime_failure_type_distribution', {})}`
- Executor/provider error types: `{graph_receipts.get('execution_error_type_distribution', {})}`
- Direct provider error types: `{direct_receipts.get('error_type_distribution', {})}`
- Collection failures: **{len(collection_failures) if isinstance(collection_failures, list) else 0}**

## Failure types

{failures}
"""
    if tuple(report["agentgraph_minus_direct"]) == ("exact_match", "token_f1"):
        return f"""# {report['dataset']} Architecture Validation

Fixed {report['project_split']} samples: **{report['sample_count']}**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. {skill_sentence}

Native metrics: **exact_match** and **token_f1** (`{report['metric_scope']}`). AgentGraph explicit FINISH: **{report['explicit_finished_count']}/{report['sample_count']}**; terminal failures: **{report['terminal_failure_count']}**; operational/evaluator failures: **{report['operational_failure_count']}**.

Terminal-output parsing failures: **Direct {report['direct_parsing_failure_count']}**, **AgentGraph {report['parsing_failure_count']}**.

| Condition | Completed | Evaluator valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | {direct['completed']} | {direct['evaluator_valid']} | {100 * float(direct['strict_exact_match']):.2f}% | {100 * float(direct['strict_token_f1']):.2f}% |
| AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * float(graph['strict_exact_match']):.2f}% | {100 * float(graph['strict_token_f1']):.2f}% |

AgentGraph - Direct: **{100 * float(report['agentgraph_minus_direct']['exact_match']):+.2f} EM**, **{100 * float(report['agentgraph_minus_direct']['token_f1']):+.2f} F1**.

{protocol_sentence}

## Failure types

{failures}
"""
    return f"""# {report['dataset']} Architecture Validation

Fixed {report['project_split']} samples: **{report['sample_count']}**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. {skill_sentence}

Primary metric: **{metric_name}** (`{report['metric_scope']}`). AgentGraph explicit FINISH: **{report['explicit_finished_count']}/{report['sample_count']}**; terminal failures: **{report['terminal_failure_count']}**; operational/evaluator failures: **{report['operational_failure_count']}**.

Terminal-output parsing failures: **Direct {report['direct_parsing_failure_count']}**, **AgentGraph {report['parsing_failure_count']}**.

| Condition | Completed | Evaluator valid | Strict {metric_name} |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | {direct['completed']} | {direct['evaluator_valid']} | {100 * float(direct[strict_key]):.2f}% |
| AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * float(graph[strict_key]):.2f}% |

AgentGraph - Direct: **{100 * float(delta):+.2f} percentage points**.

{protocol_sentence}

## Failure types

{failures}
"""


def _compatibility_config(
    config: Mapping[str, Any], bounded: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Present the dataset section to the unchanged HotpotQA graph collector."""

    value = dict(config)
    value["hotpotqa_evaluation"] = bounded
    return value


def _attach_healthbench_reference_judge(
    backend: LiveSmokeBackend,
    config: Mapping[str, Any],
    root: Path,
) -> Mapping[str, str]:
    """Attach the existing HealthBench grader without exposing it to Director."""

    evaluation = _mapping(config["evaluation"], "evaluation")
    catalog_path = _resolve(root, str(evaluation["healthbench_judge_catalog_path"]))
    if not catalog_path.is_file():
        raise ConfigurationError(
            f"HealthBench judge catalog does not exist: {catalog_path}"
        )
    judge_registry = load_model_registry(catalog_path)
    judge, provider_model = LiveSmokeBackend._build_healthbench_judge(
        judge_registry,
        os.environ.get("VECTOR_ENGINE_API_KEY", ""),
        str(evaluation["healthbench_judge_model"]),
    )
    backend.judge = judge
    backend.judge_model = provider_model
    return {
        "mode": "openai_simple_evals_compatible_reference",
        "configured_model_id": str(evaluation["healthbench_judge_model"]),
        "provider_model": provider_model,
        "catalog_path": str(catalog_path),
        "official_private_evaluator": "unavailable",
    }


def _attach_swebench_official_harness(
    backend: LiveSmokeBackend,
    config: Mapping[str, Any],
    root: Path,
    selected: Sequence[TaskRecord],
) -> Mapping[str, Any]:
    """Attach SkillFlow's official SWE-bench Docker evaluator or fail closed."""

    evaluation = _mapping(config["evaluation"], "evaluation")
    harness = OfficialSWEbenchHarness(
        evaluator_path=_resolve(root, str(evaluation["swebench_evaluator_path"])),
        harness_path=_resolve(root, str(evaluation["swebench_harness_path"])),
        dataset_source=str(evaluation["swebench_dataset_source"]),
        dataset_path=_resolve(root, str(evaluation["swebench_dataset_path"])),
        evaluation_root=_resolve(root, str(evaluation["swebench_evaluation_root"])),
        docker_namespace=str(evaluation["swebench_docker_namespace"]),
        timeout_seconds=int(evaluation["swebench_timeout_seconds"]),
    )
    instance_ids: list[str] = []
    for task in selected:
        payload = task.metadata.get("evaluator_payload", {})
        instance_id = payload.get("instance_id") if isinstance(payload, Mapping) else None
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise CompletionBenchmarkRoundError(
                f"selected SWE-bench task {task.task_id} has no instance_id"
            )
        instance_ids.append(instance_id.strip())
    receipt = harness.preflight(selected)
    backend.swe_harness = harness
    return receipt


def _completion_stable_zero_check(
    tasks: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    dataset_key: str,
    judge_model: str = "",
) -> Mapping[str, Any]:
    """Extend the reused chain check with dataset evaluator receipts."""

    environment_dataset = dataset_key in {"webshop", "alfworld"}
    base = _stable_zero_check(
        tasks,
        direct,
        trajectories,
        require_non_empty_final_answer=not environment_dataset,
    )
    checks: list[dict[str, Any]] = []
    for task, base_check in zip(tasks, base["checks"], strict=True):
        direct_value = direct.get(task.task_id)
        graph_value = trajectories.get(task.task_id)
        direct_evaluation = (
            direct_value.get("evaluation", {}) if direct_value else {}
        )
        graph_evaluation = graph_value.get("evaluation", {}) if graph_value else {}
        evaluator_version = evaluator_version_for(task)
        direct_evaluator_valid = bool(
            direct_evaluation.get("valid") is True
            and direct_evaluation.get("evaluator_version") == evaluator_version
        )
        graph_evaluator_valid = bool(
            graph_evaluation.get("valid") is True
            and graph_evaluation.get("evaluator_version") == evaluator_version
        )
        judge_receipts_valid = True
        official_harness_receipts_valid = True
        environment_terminal_receipt_valid = True
        if dataset_key == "healthbench_professional":
            judge_receipts_valid = all(
                isinstance(value.get("details"), Mapping)
                and value["details"].get("judge_model") == judge_model
                and isinstance(value["details"].get("rubric_grades"), list)
                and bool(value["details"]["rubric_grades"])
                for value in (direct_evaluation, graph_evaluation)
            )
        elif dataset_key == "swe_bench":
            official_harness_receipts_valid = all(
                isinstance(value.get("details"), Mapping)
                and isinstance(value["details"].get("resolved"), bool)
                and value["details"].get("proxy_metric_used") is False
                for value in (direct_evaluation, graph_evaluation)
            )
        elif environment_dataset:
            environment_terminal_receipt_valid = _graph_environment_terminal_receipt(
                graph_value,
                dataset_key=dataset_key,
            )
        check = {
            **dict(base_check),
            "direct_evaluator_valid": direct_evaluator_valid,
            "agentgraph_evaluator_valid": graph_evaluator_valid,
            "judge_receipts_valid": judge_receipts_valid,
            "official_harness_receipts_valid": official_harness_receipts_valid,
            "environment_terminal_receipt_valid": environment_terminal_receipt_valid,
        }
        check["passed"] = bool(
            check.get("passed")
            and direct_evaluator_valid
            and graph_evaluator_valid
            and judge_receipts_valid
            and official_harness_receipts_valid
            and environment_terminal_receipt_valid
        )
        checks.append(check)
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "criterion": (
            "every_fixed_task_completed_the_full_chain_with_valid_evaluator_receipts"
        ),
        "checks": checks,
    }


def _graph_environment_terminal_receipt(
    trajectory: Optional[Mapping[str, Any]],
    *,
    dataset_key: str,
) -> bool:
    """Validate a persisted terminal or fixed-budget truncation receipt.

    SkillFlow treats both a simulator terminal transition and exhaustion of the
    configured episode-step budget as completed bounded WebShop/ALFWorld
    rollouts.  The latter remains a valid zero-reward evaluator input even
    though the underlying simulator did not emit ``done=True``.
    """

    if not isinstance(trajectory, Mapping):
        return False
    turns = trajectory.get("turns")
    if not isinstance(turns, list) or not turns:
        return False
    final_turn = turns[-1]
    if not isinstance(final_turn, Mapping):
        return False
    action = final_turn.get("action")
    if not isinstance(action, Mapping):
        return False
    action_name = action.get("action", action.get("action_type"))
    if not isinstance(action_name, str) or action_name.strip().lower() != "finish":
        return False
    runtime_summary = final_turn.get("runtime_summary")
    if not isinstance(runtime_summary, Mapping):
        return False
    output_metadata = runtime_summary.get("output_metadata")
    if not isinstance(output_metadata, Mapping):
        return False
    candidates = [
        metadata
        for metadata in output_metadata.values()
        if isinstance(metadata, Mapping)
        and "evaluator_environment_trace" in metadata
    ]
    if len(candidates) != 1:
        return False
    metadata = candidates[0]
    task_family = metadata.get("task_family")
    if (
        not isinstance(task_family, str)
        or task_family.strip().lower() != dataset_key.strip().lower()
    ):
        return False
    trace = metadata.get("evaluator_environment_trace")
    if (
        not isinstance(trace, list)
        or not trace
        or any(not isinstance(entry, Mapping) for entry in trace)
    ):
        return False
    if any(
        not isinstance(entry.get("step"), int)
        or isinstance(entry.get("step"), bool)
        or entry.get("step") != index
        for index, entry in enumerate(trace)
    ):
        return False
    turns_used = metadata.get("environment_turns_used")
    max_turns = metadata.get("environment_max_turns")
    if (
        not isinstance(turns_used, int)
        or isinstance(turns_used, bool)
        or not isinstance(max_turns, int)
        or isinstance(max_turns, bool)
        or max_turns <= 0
        or turns_used != len(trace)
        or turns_used > max_turns
    ):
        return False
    terminal = metadata.get("environment_terminal")
    truncated = metadata.get("environment_truncated")
    if terminal is True:
        return bool(
            truncated is False
            and all(entry.get("done") is False for entry in trace[:-1])
            and trace[-1].get("done") is True
            and trace[-1].get("state_advanced") is True
        )
    return bool(
        terminal is False
        and truncated is True
        and turns_used == max_turns
        and all(entry.get("done") is False for entry in trace)
    )


def _existing_trajectory_checkpoint(
    selected: Sequence[TaskRecord],
    path: Path,
    *,
    condition_id: str,
) -> dict[str, dict[str, Any]]:
    """Load only the frozen AgentGraph receipts already admitted by a run.

    This read-only path exists so a corrected Direct comparator can be
    collected without scheduling a second AgentGraph rollout.  The normal
    collector already validated evaluator or reportable-terminal-failure
    admission before writing this ordered checkpoint; this projection retains
    only exact task and condition identities from the current frozen panel.
    """

    selected_ids = {task.task_id for task in selected}
    existing: dict[str, dict[str, Any]] = {}
    for value in _read_jsonl(path):
        embedded = value.get("task")
        task_id = embedded.get("task_id") if isinstance(embedded, Mapping) else None
        if (
            isinstance(task_id, str)
            and task_id in selected_ids
            and value.get("condition_id") == condition_id
        ):
            existing[task_id] = dict(value)
    return existing


async def run_completion_benchmark_round(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prepare_only: bool = False,
    canary_only: bool = False,
    direct_only: bool = False,
) -> Mapping[str, Any]:
    if direct_only and (prepare_only or canary_only):
        raise CompletionBenchmarkRoundError(
            "direct_only cannot be combined with prepare_only or canary_only"
        )
    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_completion_benchmark_config(config)
    section_name, bounded = _evaluation_section(config)
    dataset_key = str(bounded["dataset_key"])
    dataset_registry_validation = _validate_runtime_dataset_registry(config, root)
    paths = _paths(config, root)
    selected = _select_tasks(config, root, paths["selected"])
    failures = _read_jsonl(paths["failures"])
    gpu = _mapping(config["gpu"], "gpu")
    director_config = _mapping(config["director"], "director")
    configured_rollout_gpu = int(gpu["rollout_physical"])
    effective_rollout_gpu = int(
        os.environ.get("FLOWSTEER_ROLLOUT_GPU", configured_rollout_gpu)
    )
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.completion_benchmark.round_manifest.v1",
        "status": "prepared" if prepare_only else "runtime_preflight",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "evaluation_section": section_name,
        "dataset_key": dataset_key,
        "git_start": _git_state(root),
        "selected_task_ids": [task.task_id for task in selected],
        "sample_count": len(selected),
        "fixed_split": str(bounded["split"]),
        "evaluation_stage": str(bounded.get("stage", "legacy_unspecified")),
        "required_partition": bounded.get("required_partition"),
        "dataset_registry_validation": dataset_registry_validation,
        "metric_interpretation": (
            "stable_zero_chain_only_not_a_benchmark_estimate"
            if canary_only
            else (
                "development_result_not_a_final_benchmark_estimate"
                if bounded.get("stage") == "development"
                else (
                    "formal_evaluation_result"
                    if bounded.get("stage") == "final_evaluation"
                    else "legacy_evaluation_scope_not_formal_heldout"
                )
            )
        ),
        "model_visible_task_boundary": {
            "prompt_source": "TaskRecord.question",
            "ground_truth_role": "evaluator_only",
            "evaluator_payload_role": "evaluator_only",
        },
        "training_enabled": False,
        "optimizer_updates": 0,
        "direct_only": direct_only,
        "runtime_resource": {
            "configured_rollout_physical": configured_rollout_gpu,
            "effective_rollout_physical": effective_rollout_gpu,
            "resource_adaptation": effective_rollout_gpu != configured_rollout_gpu,
            "evaluation_concurrency": int(bounded["concurrency"]),
            "task_timeout_seconds": bounded.get("task_timeout_seconds"),
            "supervisor_port": int(
                os.environ.get("FLOWSTEER_SUPERVISOR_PORT", "8015")
            ),
            "context_length": int(
                os.environ.get(
                    "FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH",
                    str(director_config.get("max_context_tokens", 8192)),
                )
            ),
            "mem_fraction_static": float(
                os.environ.get("FLOWSTEER_SUPERVISOR_MEM_FRACTION", "0.82")
            ),
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    if prepare_only:
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        return manifest

    try:
        backend = LiveSmokeBackend.from_config(config, root, evaluation_only=True)
        judge_receipt = None
        swebench_harness_receipt = None
        if dataset_key == "healthbench_professional":
            judge_receipt = _attach_healthbench_reference_judge(
                backend,
                config,
                root,
            )
        elif dataset_key == "swe_bench":
            swebench_harness_receipt = _attach_swebench_official_harness(
                backend,
                config,
                root,
                selected,
            )
        director = _mapping(config["director"], "director")
        behavior_adapter = director.get("behavior_adapter_name")
        behavior_checkpoint = director.get("behavior_adapter_checkpoint")
        if (
            isinstance(behavior_adapter, str)
            and behavior_adapter.strip()
            and isinstance(behavior_checkpoint, str)
            and behavior_checkpoint.strip()
        ):
            adapter_preflight = await asyncio.to_thread(
                backend.publisher.ensure_loaded_adapter,
                checkpoint_path=str(_resolve(root, behavior_checkpoint)),
                adapter_name=behavior_adapter,
            )
        else:
            adapter_preflight = {
                "status": "base_model_ready",
                "success": True,
                "mode": "base_model",
                "adapter_name": None,
                "checkpoint_path": None,
                "loaded_now": False,
                "training_performed": False,
                "policy_published": False,
            }
        sglang_server_runtime = await asyncio.to_thread(
            backend.publisher.server_runtime_receipt
        )
        evaluator_preflight = await _run_evaluator_preflight(
            backend,
            config,
            root,
            dataset_key,
        )
        preflight = {
            **dict(adapter_preflight),
            "sglang_server_runtime": sglang_server_runtime,
            "evaluator_preflight": evaluator_preflight,
            "healthbench_judge_model": (
                backend.judge_model
                if dataset_key == "healthbench_professional"
                else None
            ),
            "healthbench_judge_receipt": judge_receipt,
            "swebench_harness_receipt": swebench_harness_receipt,
        }
        manifest["runtime_resource"]["sglang_server_runtime"] = (
            sglang_server_runtime
        )
        _write_json(paths["manifest"], manifest)
        _write_json(paths["preflight"], preflight)
    except Exception as exc:
        manifest.update(
            status="failed_runtime_preflight",
            error=_safe_error(exc),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        raise CompletionBenchmarkRoundError(
            f"{dataset_key} runtime preflight failed"
        ) from exc

    stable_zero_sample_count = int(
        bounded.get("stable_zero_sample_count", min(2, len(selected)))
    )
    active = selected[:stable_zero_sample_count] if canary_only else selected
    manifest["status"] = "direct_baseline"
    _write_json(paths["manifest"], manifest)
    direct = await _collect_direct(
        backend,
        active,
        config,
        root,
        paths["direct"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    if direct_only:
        manifest["status"] = "paired_report_from_existing_agentgraph"
        trajectories = _existing_trajectory_checkpoint(
            active,
            paths["trajectories"],
            condition_id=str(config["experiment"]["condition_id"]),
        )
        manifest["agentgraph_progress"] = {
            "completed": len(trajectories),
            "reused_without_execution": True,
        }
        _write_json(paths["manifest"], manifest)
    else:
        manifest["status"] = "agentgraph"
        _write_json(paths["manifest"], manifest)
        trajectories = await _collect_graph(
            backend,
            active,
            _compatibility_config(config, bounded),
            paths["trajectories"],
            failures,
            manifest,
            paths["manifest"],
            failure_path=paths["failures"],
        )
        _atomic_jsonl(paths["failures"], failures)

    rows = _paired_rows(active, direct, trajectories, dataset_key)
    _atomic_jsonl(paths["paired"], rows)
    metric_name = str(_BENCHMARKS[dataset_key]["primary_metric"])
    wrong = [
        row
        for row in rows
        if float(row["agentgraph"].get(metric_name, 0.0)) < 1.0
    ]
    _atomic_jsonl(paths["wrong"], wrong)
    stable_zero = _completion_stable_zero_check(
        active,
        direct,
        trajectories,
        dataset_key=dataset_key,
        judge_model=backend.judge_model,
    )
    manifest["stable_zero"] = stable_zero
    if canary_only:
        manifest.update(
            status=(
                "stable_zero_confirmed"
                if stable_zero["passed"]
                else "failed_stable_zero"
            ),
            canary_task_count=len(active),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        if not stable_zero["passed"]:
            raise CompletionBenchmarkRoundError(
                f"{dataset_key} canary failed the Stable Zero chain"
            )
        return manifest

    report = _report(
        rows,
        config,
        tuple(trajectories.values()),
        collection_failures=failures,
    )
    _write_json(paths["report_json"], report)
    paths["report_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_markdown"].write_text(
        _report_markdown(report), encoding="utf-8"
    )
    if report["operational_failure_count"]:
        final_status = "completed_with_operational_failures"
    elif report["terminal_failure_count"]:
        final_status = "completed_with_terminal_failures"
    else:
        final_status = "completed"
    manifest.update(
        status=final_status,
        direct_progress={**dict(manifest.get("direct_progress", {})), "completed": len(direct)},
        agentgraph_progress={
            **dict(manifest.get("agentgraph_progress", {})),
            "completed": len(trajectories),
        },
        metrics={
            "direct": report["direct_local_baseline"],
            "agentgraph": report["agentgraph"],
            "delta": report["agentgraph_minus_direct"],
        },
        failure_type_counts=report["failure_types"],
        git_end=_git_state(root),
        completed_at=_utc_now(),
    )
    _write_json(paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="fixed benchmark evaluation YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze selected tasks without starting a model or API",
    )
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help=(
            "run stable_zero_sample_count frozen tasks through the Stable Zero chain"
        ),
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help=(
            "collect/resume Direct only and rebuild paired reports from the "
            "existing frozen AgentGraph checkpoint"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = asyncio.run(
            run_completion_benchmark_round(
                _resolve(PROJECT_ROOT, args.config),
                project_root=PROJECT_ROOT,
                prepare_only=bool(args.prepare_only),
                canary_only=bool(args.canary_only),
                direct_only=bool(args.direct_only),
            )
        )
    except (
        CompletionBenchmarkRoundError,
        ConfigurationError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Completion benchmark round failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dataset_key": manifest["dataset_key"],
                "sample_count": manifest["sample_count"],
                "metrics": manifest.get("metrics"),
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
