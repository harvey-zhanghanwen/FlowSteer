#!/usr/bin/env python3
"""Run a two-task HotpotQA Tool-availability intent-to-treat pair.

This is a bounded coordinator over the existing HotpotQA evaluator,
FlowSteer-style paired-probe utilities, and ``LiveSmokeBackend``.  It does not
define another planner or runtime.  ``--prepare-only`` performs task locking
and exact Direct reuse without constructing a model backend; live collection
requires the explicit ``--run`` switch.
"""

# ruff: noqa: E402 -- executable scripts add repository paths before imports.

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_hotpotqa_round import (
    _atomic_jsonl,
    _by_task,
    _direct_resume_matches,
    _mapping,
    _read_jsonl,
    _resolve,
    _select_tasks,
    _utc_now,
    validate_hotpot_config,
)
from run_joint_qa_mace_skill import _executor_versions, _valid_outcome
from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _qa_tool_runtime_settings,
    _write_json,
    evaluator_version_for,
    version_bundle_for,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_model_registry,
    load_yaml,
)
from src.interactive.director import decode_director_transcript
from src.interactive.exploration.paired_probe import randomize_probe_order
from src.interactive.persistence import stable_id
from src.interactive.qa_tool_adapter import (
    QA_RETRIEVAL_READ_TOOL_ID,
    QA_RETRIEVAL_SEARCH_TOOL_ID,
)
from src.interactive.records import ProbeRecord, TaskRecord, TrajectoryRecord
from src.interactive.scientific_sampling import ScientificSamplingCoordinate
from src.interactive.task_evaluator import evaluate_task


DEFAULT_CONFIG = (
    PROJECT_ROOT / "config" / "development_hotpotqa_tool_availability_pair_v1.yaml"
)
ESTIMAND = (
    "qa_tool_availability_assignment_intent_to_treat_effect_"
    "on_hotpotqa_official_answer_metrics"
)
EMPTY_CANVAS = {"nodes": [], "relations": [], "output_agent_id": None}


class ToolAvailabilityPairError(RuntimeError):
    """A preregistered Tool-availability comparison failed closed."""


def _required_pair_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("tool_availability_probe"), "tool_availability_probe")


def _condition_ids(config: Mapping[str, Any]) -> dict[str, str]:
    runtime = _mapping(config.get("qa_tool_runtime"), "qa_tool_runtime")
    raw = _mapping(
        runtime.get("paired_probe_condition_ids"),
        "qa_tool_runtime.paired_probe_condition_ids",
    )
    if set(raw) != {"tool_off", "tool_on"}:
        raise ConfigurationError(
            "paired_probe_condition_ids must contain exactly tool_off and tool_on"
        )
    result = {name: str(raw[name]).strip() for name in ("tool_off", "tool_on")}
    if any(not value for value in result.values()) or len(set(result.values())) != 2:
        raise ConfigurationError("Tool-availability arm condition IDs must be distinct")
    return result


def validate_pair_config(config: Mapping[str, Any]) -> None:
    """Validate the bounded pair in addition to the existing HotpotQA checks."""

    validate_hotpot_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    bounded = _mapping(config.get("hotpotqa_evaluation"), "hotpotqa_evaluation")
    runtime = _mapping(config.get("qa_tool_runtime"), "qa_tool_runtime")
    probe = _required_pair_section(config)
    deployment = _mapping(config.get("deployment"), "deployment")
    skills = _mapping(config.get("skills"), "skills")
    expected_task_ids = (
        "hotpotqa:5a7a06935542990198eaf050",
        "hotpotqa:5a879ab05542996e4f30887e",
    )
    checks = {
        "development stage": bounded.get("stage") == "development",
        "development partition": bounded.get("required_partition") == "development",
        "two fixed tasks": bounded.get("sample_count") == 2,
        "task-id selection": bounded.get("selection") == "task_ids",
        "v4 first development tasks": tuple(bounded.get("task_ids", ()))
        == expected_task_ids,
        "declared Direct reuse": bool(str(bounded.get("direct_reused_from", "")).strip()),
        "QA runtime enabled": runtime.get("enabled") is True,
        "forced probes admitted": deployment.get("allow_forced_probes") is True,
        "Skills disabled": skills.get("enabled") is False,
        "probe enabled": probe.get("enabled") is True,
        "intent-to-treat estimand": probe.get("estimand") == ESTIMAND,
        "empty Canvas": probe.get("shared_canvas_state") == "empty_canvas",
        "randomized arm order": probe.get("randomize_arm_order") is True,
        "one rollout per arm": probe.get("rollout_index") == 0,
        "half assignment probability": probe.get("sampling_probability") == 0.5,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "HotpotQA Tool-availability pair violates bounds: " + ", ".join(failed)
        )
    condition_ids = _condition_ids(config)
    if str(experiment.get("condition_id", "")).strip() in set(condition_ids.values()):
        raise ConfigurationError(
            "pair experiment condition ID must be distinct from both arm IDs"
        )
    for field in ("arm_order_seed", "sampling_anchor_ordinal"):
        value = probe.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"tool_availability_probe.{field} must be non-negative")


def _accepted_answers(task: TaskRecord) -> tuple[str, ...]:
    payload = task.metadata.get("evaluator_payload", {})
    raw = payload.get("accepted_answers") if isinstance(payload, Mapping) else None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(str(value) for value in raw if str(value).strip())
        if values:
            return values
    ground_truth = task.ground_truth
    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (str, bytes)):
        return tuple(str(value) for value in ground_truth if str(value).strip())
    text = str(ground_truth)
    return (text,) if text.strip() else ()


def _direct_semantics(task: TaskRecord) -> Mapping[str, Any]:
    metadata = task.metadata
    return {
        "task_id": task.task_id,
        "question": task.question,
        "ground_truth": task.ground_truth,
        "split": task.split,
        "dataset_key": metadata.get("dataset_key"),
        "source": metadata.get("source"),
        "task_type": metadata.get("task_type"),
        "metric": metadata.get("metric"),
        "native_split": metadata.get("native_split"),
        "sampling": metadata.get("sampling"),
        "evaluator_payload": metadata.get("evaluator_payload", {}),
        "accepted_answers": list(_accepted_answers(task)),
        "evaluator_version": evaluator_version_for(task),
    }


async def _validated_direct_reuse(
    config: Mapping[str, Any],
    selected: Sequence[TaskRecord],
    root: Path,
    destination: Path,
) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any]]:
    """Copy exactly two semantically identical v3 Direct records; never generate."""

    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    source = _resolve(root, str(bounded["direct_reused_from"]))
    if not source.is_file():
        raise ToolAvailabilityPairError(f"Direct reuse source is absent: {source}")
    candidates = _read_jsonl(source)
    selected_ids = {task.task_id for task in selected}
    candidate_ids = [value.get("task_id") for value in candidates]
    if len(candidates) != 2 or set(candidate_ids) != selected_ids or len(set(candidate_ids)) != 2:
        raise ToolAvailabilityPairError(
            "Direct reuse source must contain exactly the two frozen development tasks"
        )
    by_task = _by_task(candidates)
    reused: dict[str, dict[str, Any]] = {}
    for task in selected:
        value = by_task.get(task.task_id)
        if value is None or not _direct_resume_matches(
            value,
            task=task,
            model_id=str(bounded["direct_model_id"]),
            protocol=str(bounded["direct_protocol"]),
            seed=int(bounded["direct_generation_seed"]),
        ):
            raise ToolAvailabilityPairError(
                f"Direct reuse receipt mismatch for {task.task_id}"
            )
        embedded = value.get("task")
        if not isinstance(embedded, Mapping):
            raise ToolAvailabilityPairError(
                f"Direct reuse has no embedded task for {task.task_id}"
            )
        try:
            reused_task = TaskRecord.from_dict(embedded)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolAvailabilityPairError(
                f"Direct embedded task is invalid for {task.task_id}"
            ) from exc
        if _direct_semantics(reused_task) != _direct_semantics(task):
            raise ToolAvailabilityPairError(
                f"Direct model/scoring semantics differ for {task.task_id}"
            )
        rescored = await evaluate_task(task, str(value.get("final_answer", "")))
        persisted_evaluation = value.get("evaluation")
        if not isinstance(persisted_evaluation, Mapping) or dict(
            persisted_evaluation
        ) != asdict(rescored):
            raise ToolAvailabilityPairError(
                f"Direct official evaluator receipt differs for {task.task_id}"
            )
        copied = dict(value)
        copied["reuse_receipt"] = {
            "reused": True,
            "source": str(source),
            "task_semantics_exact": True,
            "official_evaluator_rescore_exact": True,
        }
        reused[task.task_id] = copied
    receipt = {
        "source": str(source),
        "selected_records": len(selected),
        "reused_records": len(reused),
        "newly_collected_records": 0,
        "gateway_calls": 0,
        "model_and_scoring_semantics_exact": True,
    }
    if receipt["reused_records"] != 2 or receipt["newly_collected_records"] != 0:
        raise ToolAvailabilityPairError("Direct receipt is not reused=2,new=0")
    _atomic_jsonl(destination, [reused[task.task_id] for task in selected])
    return reused, receipt


def _paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config.get("storage"), "storage")
    fields = {
        "selected": "selected_tasks_path",
        "direct": "direct_predictions_path",
        "trajectories": "trajectories_path",
        "pairs": "paired_results_path",
        "manifest": "manifest_path",
    }
    return {name: _resolve(root, str(storage[field])) for name, field in fields.items()}


async def prepare(
    config: Mapping[str, Any],
    root: Path = PROJECT_ROOT,
) -> tuple[tuple[TaskRecord, ...], Mapping[str, Any]]:
    """Lock two tasks and Direct reuse without constructing a backend."""

    validate_pair_config(config)
    paths = _paths(config, root)
    selected = _select_tasks(config, root, paths["selected"])
    expected_positions = (0, 1)
    for task, position in zip(selected, expected_positions, strict=True):
        if (
            task.split != "validation"
            or task.metadata.get("joint_qa_partition") != "development"
            or task.metadata.get("native_candidate_position") != position
        ):
            raise ToolAvailabilityPairError(
                f"{task.task_id} is not the declared v4 development position {position}"
            )
        condition_ids = _condition_ids(config)
        if _qa_tool_runtime_settings(
            config, task, condition_id=condition_ids["tool_off"]
        ) is not None:
            raise ToolAvailabilityPairError("tool_off unexpectedly received a Tool runtime")
        if _qa_tool_runtime_settings(
            config, task, condition_id=condition_ids["tool_on"]
        ) is None:
            raise ToolAvailabilityPairError("tool_on did not select QA retrieval")
    _, direct_receipt = await _validated_direct_reuse(
        config, selected, root, paths["direct"]
    )
    graph_config = _mapping(config["agent_graph"], "agent_graph")
    registry = load_model_registry(_resolve(root, str(graph_config["model_catalog_path"])))
    experiment = _mapping(config["experiment"], "experiment")
    director = _mapping(config["director"], "director")
    probe = _required_pair_section(config)
    manifest = {
        "schema_version": "flowsteer.hotpotqa.qa-tool-availability-pair-manifest.v1",
        "status": "prepared",
        "prepared_at": _utc_now(),
        "prepare_only_constructed_backend": False,
        "model_api_calls": 0,
        "task_ids": [task.task_id for task in selected],
        "task_split": "validation",
        "joint_qa_partition": "development",
        "direct_reuse": dict(direct_receipt),
        "pair_preregistration": {
            "estimand": ESTIMAND,
            "treatment": "QA Tool availability assignment",
            "not_an_estimand": ["Tool invocation", "Tool usefulness", "Skill usefulness"],
            "condition_ids": _condition_ids(config),
            "randomize_arm_order": True,
            "sampling_probability": 0.5,
            "sampling_schedule_purpose": str(
                experiment["sampling_schedule_purpose"]
            ),
            "sampling_anchor_ordinal": int(probe["sampling_anchor_ordinal"]),
            "shared_canvas_state": dict(EMPTY_CANVAS),
            "rollout_index": 0,
            "forced_probe": True,
            "grpo_eligible": False,
        },
        "versions": {
            "policy": str(director["behavior_policy_version"]),
            "model_catalog": registry.catalog_id,
            "evaluator": evaluator_version_for(selected[0]),
            "prompt": str(experiment["prompt_version"]),
            "tool": str(experiment["tool_version"]),
        },
        "pair_progress": {"completed": 0, "expected": 2},
    }
    _write_json(paths["manifest"], manifest)
    return tuple(selected), manifest


def _tool_receipts(record: TrajectoryRecord) -> tuple[dict[str, Any], ...]:
    """Read ToolReceipts from actual executor calls, never cumulative summaries."""

    receipts: list[dict[str, Any]] = []
    serialized: set[str] = set()
    for turn in record.turns:
        for execution in turn.executions:
            metadata = execution.metadata
            if not isinstance(metadata, Mapping):
                raise ToolAvailabilityPairError("ExecutionRecord metadata is not a mapping")
            response = metadata.get("response", {})
            if not isinstance(response, Mapping):
                raise ToolAvailabilityPairError("ExecutionRecord response is not a mapping")
            raw = response.get("tool_receipts", ())
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ToolAvailabilityPairError("tool_receipts is not an ordered sequence")
            for receipt in raw:
                if not isinstance(receipt, Mapping):
                    raise ToolAvailabilityPairError("ToolReceipt is not a mapping")
                tool_id = receipt.get("tool_id")
                if not isinstance(tool_id, str) or not tool_id.strip():
                    raise ToolAvailabilityPairError("ToolReceipt has no tool_id")
                tool_version = receipt.get("tool_version")
                if not isinstance(tool_version, str) or not tool_version.strip():
                    raise ToolAvailabilityPairError("ToolReceipt has no tool_version")
                request = receipt.get("request")
                if not isinstance(request, Mapping):
                    raise ToolAvailabilityPairError("ToolReceipt has no request receipt")
                if (
                    not isinstance(request.get("action"), str)
                    or not str(request["action"]).strip()
                    or not isinstance(request.get("arguments"), Mapping)
                ):
                    raise ToolAvailabilityPairError("ToolReceipt request is invalid")
                result = receipt.get("result")
                error_type = receipt.get("error_type")
                if (result is None) == (error_type is None):
                    raise ToolAvailabilityPairError(
                        "ToolReceipt requires exactly one result or error_type"
                    )
                if result is not None and not isinstance(result, Mapping):
                    raise ToolAvailabilityPairError("ToolReceipt result is invalid")
                if error_type is not None and (
                    not isinstance(error_type, str) or not error_type.strip()
                ):
                    raise ToolAvailabilityPairError("ToolReceipt error_type is invalid")
                for field in (
                    "started_at_monotonic",
                    "ended_at_monotonic",
                    "latency_ms",
                ):
                    value = receipt.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or float(value) < 0.0
                    ):
                        raise ToolAvailabilityPairError(
                            f"ToolReceipt {field} is invalid"
                        )
                if float(receipt["ended_at_monotonic"]) < float(
                    receipt["started_at_monotonic"]
                ):
                    raise ToolAvailabilityPairError("ToolReceipt time order is invalid")
                normalized = dict(receipt)
                identity = json.dumps(
                    normalized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if identity in serialized:
                    raise ToolAvailabilityPairError(
                        "duplicate ToolReceipt across executor calls"
                    )
                serialized.add(identity)
                receipts.append(normalized)
    return tuple(receipts)


def _initial_observation(record: TrajectoryRecord) -> Mapping[str, Any]:
    if not record.turns:
        raise ToolAvailabilityPairError("paired trajectory has no Director turn")
    messages = decode_director_transcript(record.turns[0].prompt)
    if not messages or messages[-1].get("role") != "user":
        raise ToolAvailabilityPairError("initial Director transcript is invalid")
    content = messages[-1].get("content")
    if not isinstance(content, str) or "\n\n" not in content:
        raise ToolAvailabilityPairError("initial Director observation is absent")
    try:
        payload = json.loads(content.split("\n\n", 1)[1])
    except json.JSONDecodeError as exc:
        raise ToolAvailabilityPairError(
            "initial Director observation is not JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ToolAvailabilityPairError("initial Director observation is not a mapping")
    return dict(payload)


def _validate_tool_catalog_exposure(
    off: TrajectoryRecord,
    on: TrajectoryRecord,
) -> Mapping[str, Any]:
    """Verify treatment exposure using the persisted initial Director prompt."""

    off_observation = dict(_initial_observation(off))
    on_observation = dict(_initial_observation(on))
    off_catalog = off_observation.pop("tool_catalog", None)
    on_catalog = on_observation.pop("tool_catalog", None)
    if off_catalog not in (None, []):
        raise ToolAvailabilityPairError("tool_off initial prompt exposes a Tool catalog")
    if not isinstance(on_catalog, list) or any(
        not isinstance(item, Mapping) for item in on_catalog
    ):
        raise ToolAvailabilityPairError("tool_on initial Tool catalog is invalid")
    tool_ids = [item.get("tool_id") for item in on_catalog]
    expected = {QA_RETRIEVAL_SEARCH_TOOL_ID, QA_RETRIEVAL_READ_TOOL_ID}
    if len(tool_ids) != 2 or set(tool_ids) != expected:
        raise ToolAvailabilityPairError(
            "tool_on must expose exactly QA retrieval search/read"
        )
    if off_observation != on_observation:
        raise ToolAvailabilityPairError(
            "initial Director observations differ beyond Tool catalog availability"
        )
    return {
        "tool_off_catalog_tool_ids": [],
        "tool_on_catalog_tool_ids": tool_ids,
        "non_treatment_observation_projection_equal": True,
    }


def _coordinate(record: TrajectoryRecord) -> ScientificSamplingCoordinate:
    if not record.sampling_receipt_verified:
        raise ToolAvailabilityPairError(
            f"invalid sampling receipt: {record.trajectory_id}"
        )
    return ScientificSamplingCoordinate.from_value(
        record.director_sampling["coordinate"]
    )


def _admit_pair(
    task: TaskRecord,
    observed: Mapping[str, TrajectoryRecord],
    *,
    condition_ids: Mapping[str, str],
    schedule_purpose: str,
    anchor: int,
    branch_order: Sequence[str],
    versions: Any,
) -> Mapping[str, Any]:
    """Fail closed unless the pair differs only by Tool availability."""

    if set(observed) != {"tool_off", "tool_on"}:
        raise ToolAvailabilityPairError("both Tool-availability arms are required")
    off, on = observed["tool_off"], observed["tool_on"]
    for arm_name, record in observed.items():
        _valid_outcome(record)
        if record.task.to_dict() != task.to_dict() or record.task.split != task.split:
            raise ToolAvailabilityPairError(f"{arm_name} task/split mismatch")
        if record.condition_id != condition_ids[arm_name]:
            raise ToolAvailabilityPairError(f"{arm_name} condition ID mismatch")
        if record.versions != versions:
            raise ToolAvailabilityPairError(f"{arm_name} version bundle mismatch")
        if not record.forced_probe or record.grpo_eligible:
            raise ToolAvailabilityPairError(
                f"{arm_name} is not an isolated forced probe"
            )
        if not record.condition_satisfied:
            raise ToolAvailabilityPairError(f"{arm_name} assignment was not satisfied")
        if record.active_skill_ids or record.retrieved_skill_ids or record.invoked_skill_ids:
            raise ToolAvailabilityPairError("Skill exposure is outside this estimand")
        coordinate = _coordinate(record)
        if (
            coordinate.task_id != task.task_id
            or coordinate.sequence_position != 0
            or coordinate.schedule_purpose != schedule_purpose
            or coordinate.optimizer_step_or_anchor_ordinal != anchor
        ):
            raise ToolAvailabilityPairError(f"{arm_name} sampling coordinate mismatch")
    if off.director_sampling != on.director_sampling:
        raise ToolAvailabilityPairError("paired arms do not share Director sampling")
    if tuple(branch_order) not in {
        ("tool_off", "tool_on"),
        ("tool_on", "tool_off"),
    }:
        raise ToolAvailabilityPairError("branch order is not a permutation of both arms")
    catalog_exposure = _validate_tool_catalog_exposure(off, on)
    receipts = {
        "tool_off": _tool_receipts(off),
        "tool_on": _tool_receipts(on),
    }
    if receipts["tool_off"]:
        raise ToolAvailabilityPairError("tool_off persisted an impossible ToolReceipt")
    invoked = {
        arm: sorted({str(receipt["tool_id"]) for receipt in arm_receipts})
        for arm, arm_receipts in receipts.items()
    }
    metrics: dict[str, dict[str, float]] = {}
    for arm, record in observed.items():
        exact_match = record.evaluation.metrics.get("exact_match")
        token_f1 = record.evaluation.metrics.get("token_f1")
        if not isinstance(exact_match, (int, float)) or not isinstance(
            token_f1, (int, float)
        ):
            raise ToolAvailabilityPairError("official HotpotQA EM/F1 metrics are missing")
        metrics[arm] = {
            "exact_match": float(exact_match),
            "token_f1": float(token_f1),
        }
    snapshot_id = stable_id(
        "canvas_snapshot",
        {
            "task_id": task.task_id,
            "task_split": task.split,
            "prefix_stage": "empty_canvas",
            "sampling_anchor_ordinal": anchor,
            "graph": EMPTY_CANVAS,
        },
    )
    probe_id = stable_id(
        "probe",
        {
            "task_id": task.task_id,
            "snapshot_id": snapshot_id,
            "policy_version": versions.policy,
            "condition_ids": dict(condition_ids),
            "estimand": ESTIMAND,
        },
    )
    return {
        "schema_version": "flowsteer.hotpotqa.qa-tool-availability-paired-observation.v1",
        "pair_id": probe_id,
        "task_id": task.task_id,
        "task_split": task.split,
        "shared_empty_canvas_snapshot_id": snapshot_id,
        "shared_empty_canvas": dict(EMPTY_CANVAS),
        "estimand": ESTIMAND,
        "treatment": "QA Tool availability assignment",
        "tool_availability_is_not_invocation": True,
        "tool_availability_is_not_useful_skill": True,
        "treatment_exposure_receipt": dict(catalog_exposure),
        "condition_ids": dict(condition_ids),
        "branch_order": list(branch_order),
        "sampling_probability": 0.5,
        "shared_director_sampling": dict(off.director_sampling),
        "versions": versions.to_dict(),
        "forced_probe": True,
        "grpo_eligible": False,
        "arms": {
            arm: {
                "tool_available": arm == "tool_on",
                "tool_invoked": bool(invoked[arm]),
                "trajectory_id": record.trajectory_id,
                "final_answer": record.final_answer,
                "evaluation_receipt": record.evaluation.to_dict(),
                "exact_match": metrics[arm]["exact_match"],
                "token_f1": metrics[arm]["token_f1"],
                "tool_receipts": list(receipts[arm]),
                "invoked_tool_ids": invoked[arm],
            }
            for arm, record in observed.items()
        },
        "effects": {
            "exact_match": metrics["tool_on"]["exact_match"]
            - metrics["tool_off"]["exact_match"],
            "token_f1": metrics["tool_on"]["token_f1"]
            - metrics["tool_off"]["token_f1"],
        },
    }


def _version_bundle(config: Mapping[str, Any], backend: LiveSmokeBackend, task: TaskRecord):
    experiment = _mapping(config["experiment"], "experiment")
    director = _mapping(config["director"], "director")
    exploration = _mapping(config["exploration"], "exploration")
    skills = _mapping(config["skills"], "skills")
    return version_bundle_for(
        task,
        policy_version=str(director["behavior_policy_version"]),
        model_catalog_version=backend.model_catalog_version,
        prompt_version=str(experiment["prompt_version"]),
        tool_version=str(experiment["tool_version"]),
        encoder_version=str(exploration.get("encoder_version", "none")),
        feature_schema_version=str(
            exploration.get("feature_schema_version", "qa-tool-availability-itt.v1")
        ),
        posterior_version=str(exploration.get("posterior_version", "none")),
        skill_library_version=str(skills.get("library_version", "none")),
    )


def _resume_cache(
    backend: LiveSmokeBackend,
    selected: Sequence[TaskRecord],
    trajectories_path: Path,
    condition_ids: Mapping[str, str],
    versions_by_task: Mapping[str, Any],
) -> dict[tuple[str, str], TrajectoryRecord]:
    selected_by_id = {task.task_id: task for task in selected}
    payloads = [
        *_read_jsonl(trajectories_path),
        *backend.evidence_store.trajectories.payloads(),
    ]
    by_trajectory_id: dict[str, TrajectoryRecord] = {}
    for payload in payloads:
        record = TrajectoryRecord.from_dict(payload)
        if record.condition_id not in set(condition_ids.values()):
            continue
        previous = by_trajectory_id.get(record.trajectory_id)
        if previous is not None and previous.to_dict() != record.to_dict():
            raise ToolAvailabilityPairError("conflicting resumed trajectory payload")
        by_trajectory_id[record.trajectory_id] = record
    cache: dict[tuple[str, str], TrajectoryRecord] = {}
    for record in by_trajectory_id.values():
        task = selected_by_id.get(record.task.task_id)
        if task is None:
            raise ToolAvailabilityPairError("paired evidence contains an unfrozen task")
        if record.task.to_dict() != task.to_dict():
            raise ToolAvailabilityPairError("resumed paired task semantics differ")
        if record.versions != versions_by_task[task.task_id]:
            raise ToolAvailabilityPairError("resumed paired versions differ")
        key = (task.task_id, record.condition_id)
        if key in cache and cache[key].trajectory_id != record.trajectory_id:
            raise ToolAvailabilityPairError(f"ambiguous resumed arm: {key}")
        cache[key] = record
    return cache


async def _collect_arm(
    backend: LiveSmokeBackend,
    cache: dict[tuple[str, str], TrajectoryRecord],
    task: TaskRecord,
    versions: Any,
    *,
    condition_id: str,
    schedule_purpose: str,
    anchor: int,
) -> TrajectoryRecord:
    key = (task.task_id, condition_id)
    record = cache.get(key)
    if record is None:
        record = await backend.collect(
            task,
            0,
            versions,
            expected_task_split=task.split,
            condition_id=condition_id,
            sampling_schedule_purpose=schedule_purpose,
            forced_probe=True,
            condition_satisfied=True,
            sampling_anchor_ordinal=anchor,
        )
        cache[key] = record
    return record


def _persist_trajectory_mirror(
    path: Path,
    selected: Sequence[TaskRecord],
    cache: Mapping[tuple[str, str], TrajectoryRecord],
    orders: Mapping[str, Sequence[str]],
    condition_ids: Mapping[str, str],
) -> None:
    values = []
    for task in selected:
        for arm in orders[task.task_id]:
            record = cache.get((task.task_id, condition_ids[arm]))
            if record is not None:
                values.append(record.to_dict())
    _atomic_jsonl(path, values)


async def run_live(config: Mapping[str, Any], root: Path = PROJECT_ROOT) -> Mapping[str, Any]:
    """Collect only the preregistered OFF/ON pairs after explicit authorization."""

    selected, manifest = await prepare(config, root)
    paths = _paths(config, root)
    backend = LiveSmokeBackend.from_config(config, root, evaluation_only=True)
    director = _mapping(config["director"], "director")
    behavior_preflight = await asyncio.to_thread(
        backend.publisher.ensure_loaded_adapter,
        checkpoint_path=str(
            _resolve(root, str(director["behavior_adapter_checkpoint"]))
        ),
        adapter_name=str(director["behavior_adapter_name"]),
    )
    manifest = dict(manifest)
    manifest["behavior_policy_preflight"] = dict(behavior_preflight)
    _write_json(paths["manifest"], manifest)
    versions_by_task = {
        task.task_id: _version_bundle(config, backend, task) for task in selected
    }
    if any(
        versions.model_catalog != manifest["versions"]["model_catalog"]
        for versions in versions_by_task.values()
    ):
        raise ToolAvailabilityPairError("prepared and live model catalogs differ")
    condition_ids = _condition_ids(config)
    probe = _required_pair_section(config)
    schedule = str(_mapping(config["experiment"], "experiment")["sampling_schedule_purpose"])
    anchor = int(probe["sampling_anchor_ordinal"])
    rng = np.random.default_rng(int(probe["arm_order_seed"]))
    orders = {
        task.task_id: (
            lambda order: (order.presented_first, order.presented_second)
        )(randomize_probe_order("tool_off", "tool_on", rng))
        for task in selected
    }
    cache = _resume_cache(
        backend, selected, paths["trajectories"], condition_ids, versions_by_task
    )
    pair_rows_by_task = {
        str(row.get("task_id")): dict(row) for row in _read_jsonl(paths["pairs"])
    }
    for task in selected:
        observed: dict[str, TrajectoryRecord] = {}
        for arm in orders[task.task_id]:
            observed[arm] = await _collect_arm(
                backend,
                cache,
                task,
                versions_by_task[task.task_id],
                condition_id=condition_ids[arm],
                schedule_purpose=schedule,
                anchor=anchor,
            )
            _persist_trajectory_mirror(
                paths["trajectories"], selected, cache, orders, condition_ids
            )
        row = _admit_pair(
            task,
            observed,
            condition_ids=condition_ids,
            schedule_purpose=schedule,
            anchor=anchor,
            branch_order=orders[task.task_id],
            versions=versions_by_task[task.task_id],
        )
        snapshot_payload = {
            "snapshot_id": row["shared_empty_canvas_snapshot_id"],
            "problem_id": task.task_id,
            "task_split": task.split,
            "prefix_stage": "empty_canvas",
            "graph": dict(EMPTY_CANVAS),
            "policy_version": versions_by_task[task.task_id].policy,
            "sampling_anchor_ordinal": anchor,
            "estimand": ESTIMAND,
        }
        backend.evidence_store.append_snapshot(snapshot_payload)
        expected_probe = ProbeRecord(
            probe_id=str(row["pair_id"]),
            problem_id=task.task_id,
            task_split=task.split,
            snapshot_id=str(row["shared_empty_canvas_snapshot_id"]),
            policy_version=versions_by_task[task.task_id].policy,
            state_features={
                "dataset_key": "hotpotqa",
                "prefix_stage": "empty_canvas",
                "estimand": ESTIMAND,
            },
            incumbent_action={"qa_tool_available": False},
            candidate_action={"qa_tool_available": True},
            sampling_probability=0.5,
            incumbent_returns=(float(observed["tool_off"].evaluation.reward),),
            candidate_returns=(float(observed["tool_on"].evaluation.reward),),
            executor_versions=_executor_versions(backend),
            evaluator_version=versions_by_task[task.task_id].evaluator,
            feature_schema_version=versions_by_task[task.task_id].feature_schema,
            branch_order=tuple(orders[task.task_id]),
        )
        persisted_probe = backend.evidence_store.resolve_probe(expected_probe.probe_id)
        if persisted_probe is None:
            backend.evidence_store.append_probe(expected_probe)
            probe_payload = expected_probe.to_dict()
        else:
            expected_semantics = expected_probe.to_dict()
            if {
                key: value for key, value in persisted_probe.items() if key != "created_at"
            } != {
                key: value for key, value in expected_semantics.items() if key != "created_at"
            }:
                raise ToolAvailabilityPairError("persisted pair probe semantics differ")
            probe_payload = dict(persisted_probe)
        completed_row = dict(row)
        completed_row["probe"] = probe_payload
        existing_row = pair_rows_by_task.get(task.task_id)
        if existing_row is not None and existing_row != completed_row:
            raise ToolAvailabilityPairError("persisted paired observation differs")
        pair_rows_by_task[task.task_id] = completed_row
        _atomic_jsonl(
            paths["pairs"],
            [pair_rows_by_task[item.task_id] for item in selected if item.task_id in pair_rows_by_task],
        )
        manifest = dict(manifest)
        manifest["status"] = "running"
        manifest["pair_progress"] = {
            "completed": len(pair_rows_by_task),
            "expected": len(selected),
        }
        _write_json(paths["manifest"], manifest)
    manifest = dict(manifest)
    manifest["status"] = "complete"
    manifest["completed_at"] = _utc_now()
    manifest["pair_progress"] = {"completed": 2, "expected": 2}
    manifest["trajectory_ids"] = {
        task.task_id: {
            arm: cache[(task.task_id, condition_ids[arm])].trajectory_id
            for arm in ("tool_off", "tool_on")
        }
        for task in selected
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="lock tasks and Direct reuse; never construct the model backend",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="explicitly run the four forced-probe rollouts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_yaml(args.config)
    if args.prepare_only:
        _, manifest = asyncio.run(prepare(config, PROJECT_ROOT))
    else:
        manifest = asyncio.run(run_live(config, PROJECT_ROOT))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
