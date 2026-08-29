#!/usr/bin/env python3
"""Run the fixed HotpotQA Direct-vs-AgentGraph architecture validation.

The runner is intentionally evaluation-only.  It reuses the SkillFlow-derived
SGLang/Supervisor boundary and the existing FlowSteer AgentGraph collector; it
does not call a trainer, optimizer, backward pass, policy update, or Skill loop.
"""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_agentgraph_smoke import (
    LiveSmokeBackend,
    _dataset_key,
    _safe_error,
    _write_json,
    _write_jsonl,
    qa_retrieval_scopes,
    version_bundle_for,
)
from src.interactive.agent_graph import AgentNode
from src.interactive.agent_runtime import (
    AgentCallRecord,
    AgentRequest,
    ExecutionPhase,
)
from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.records import TaskRecord, TrajectoryRecord
from src.interactive.rollout_collector import execution_record_from_call
from src.interactive.task_dataset import iter_task_records
from src.interactive.task_evaluator import evaluate_task


class HotpotRoundError(RuntimeError):
    """The fixed evaluation protocol could not complete as declared."""


DIRECT_CONTRACT = (
    "Chain evidence across passages to answer multi-hop questions. Answer briefly: "
    "name, date, number, or short phrase."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _git_state(root: Path) -> Mapping[str, Optional[str]]:
    git_args: list[str] = []
    git_file = root / ".git"
    if git_file.is_file():
        first_line = git_file.read_text(encoding="utf-8").splitlines()[0]
        if first_line.startswith("gitdir: "):
            git_args = [
                f"--git-dir={first_line[len('gitdir: '):].strip()}",
                f"--work-tree={root}",
            ]

    def read(*args: str) -> Optional[str]:
        result = subprocess.run(
            ["git", *git_args, *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None

    return {
        "branch": read("branch", "--show-current"),
        "commit": read("rev-parse", "HEAD"),
    }


def validate_hotpot_config(config: Mapping[str, Any]) -> None:
    """Reject any accidental expansion beyond the declared held-out round."""

    experiment = _mapping(config.get("experiment"), "experiment")
    if experiment.get("build_only") is True:
        raise ConfigurationError(
            "HotpotQA evaluation runner rejects retrieval-index build-only profiles"
        )
    validate_agent_graph_config(config)
    bounded = _mapping(config.get("hotpotqa_evaluation"), "hotpotqa_evaluation")
    director = _mapping(config.get("director"), "director")
    grpo = _mapping(config.get("grpo"), "grpo")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")
    gpu = _mapping(config.get("gpu"), "gpu")
    checks = {
        "experiment.phase": experiment.get("phase") == "hotpotqa_evaluation",
        "experiment.training_enabled": experiment.get("training_enabled") is False,
        "dataset_key": bounded.get("dataset_key") == "hotpotqa",
        "split": bounded.get("split") == "validation",
        "selection": bounded.get("selection") == "sequential",
        "sample_count": bounded.get("sample_count") == 128,
        "rollouts_per_task": bounded.get("rollouts_per_task") == 1,
        "direct_model_id": bounded.get("direct_model_id") == "qwen3.5-9b-local",
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "grpo.enabled": grpo.get("enabled") is False,
        "optimizer_passes": grpo.get("optimization_passes_per_rollout_batch") == 0,
        "optimizer_updates": grpo.get("max_optimizer_updates") == 0,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.enabled": skills.get("enabled") is False,
        "gpu.training_enabled": gpu.get("training_enabled") is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "HotpotQA round violates fixed evaluation bounds: " + ", ".join(failed)
        )
    concurrency = bounded.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError("hotpotqa_evaluation.concurrency must be positive")
    for name in (
        "behavior_policy_version",
        "behavior_adapter_name",
        "behavior_adapter_checkpoint",
        "expected_server_weight_version",
    ):
        if not str(director.get(name, "")).strip():
            raise ConfigurationError(f"director.{name} must be non-empty")

    raw_retrieval = config.get("qa_embedding_retrieval")
    if raw_retrieval is not None:
        retrieval = _mapping(raw_retrieval, "qa_embedding_retrieval")
        task_scope, retrieval_query_scope = qa_retrieval_scopes(retrieval)
        explicit_scope_split = (
            "task_scope" in retrieval or "retrieval_query_scope" in retrieval
        )
        retrieval_checks = {
            "enabled": retrieval.get("enabled") is True,
            "condition_id": retrieval.get("condition_id") == experiment.get("condition_id"),
            "mode": retrieval.get("mode") == "model_driven_search_read",
            "dataset_scope": retrieval.get("dataset_scope") == ["hotpotqa"],
            "task_scope": (
                task_scope == "public_task"
                if explicit_scope_split
                else task_scope == "question_only"
            ),
            "retrieval_query_scope": retrieval_query_scope == "question_only",
            "normalize_embeddings": retrieval.get("normalize_embeddings") is True,
            "similarity": retrieval.get("similarity") == "cosine",
            "web_search_enabled": retrieval.get("web_search_enabled") is False,
        }
        failed_retrieval = [
            name for name, valid in retrieval_checks.items() if not valid
        ]
        if failed_retrieval:
            raise ConfigurationError(
                "HotpotQA embedding retrieval condition is invalid: "
                + ", ".join(failed_retrieval)
            )
        corpus_kind = retrieval.get("corpus_kind", "public_context")
        if corpus_kind not in {
            "public_context",
            "train_qa_memory",
            "transductive_qa_memory",
            "full_dataset_fact_memory",
        }:
            raise ConfigurationError(
                "qa_embedding_retrieval.corpus_kind must be public_context "
                "or train_qa_memory or transductive_qa_memory "
                "or full_dataset_fact_memory"
            )
        if corpus_kind in {
            "train_qa_memory",
            "transductive_qa_memory",
            "full_dataset_fact_memory",
        }:
            expected_tool_id = (
                "hotpotqa.fact_memory"
                if corpus_kind == "full_dataset_fact_memory"
                else "hotpotqa.qa_memory"
            )
            qa_memory_checks = {
                "tool_id": retrieval.get("tool_id") == expected_tool_id,
                "required_evidence_tool_id": (
                    _mapping(config["agent_graph"], "agent_graph").get(
                        "required_evidence_tool_id"
                    )
                    == expected_tool_id
                ),
                "director_feedback_mode": (
                    _mapping(config["agent_graph"], "agent_graph").get(
                        "director_feedback_mode"
                    )
                    == "control_plane"
                ),
                "require_evidence_relation": (
                    _mapping(config["agent_graph"], "agent_graph").get(
                        "require_evidence_relation"
                    )
                    is True
                ),
                "fallback_policy": (
                    retrieval.get(
                        "fallback_policy",
                        "parametric_only_when_retrieval_unsupported",
                    )
                    == "parametric_only_when_retrieval_unsupported"
                ),
            }
            if corpus_kind == "train_qa_memory":
                qa_memory_checks.update(
                    {
                        "train_sample_count": (
                            retrieval.get("train_sample_count") == 512
                        ),
                        "validation_sample_count": (
                            retrieval.get("validation_sample_count") == 128
                        ),
                        "index_scope": (
                            retrieval.get("index_scope") == "global_train_only"
                        ),
                    }
                )
            elif corpus_kind == "transductive_qa_memory":
                qa_memory_checks.update(
                    {
                        "train_sample_count": (
                            retrieval.get("train_sample_count") == 512
                        ),
                        "validation_sample_count": (
                            retrieval.get("validation_sample_count") == 128
                        ),
                        "index_scope": (
                            retrieval.get("index_scope")
                            == "global_train_plus_frozen_validation"
                        ),
                        "source_record_count": (
                            retrieval.get("source_record_count") == 640
                        ),
                        "evaluation_overlap_count": (
                            retrieval.get("evaluation_overlap_count") == 128
                        ),
                        "contains_evaluation_answers": (
                            retrieval.get("contains_evaluation_answers") is True
                        ),
                        "evaluation_regime": (
                            retrieval.get("evaluation_regime")
                            == "transductive_retrieval"
                        ),
                        "official_heldout_eligible": (
                            retrieval.get("official_heldout_eligible") is False
                        ),
                    }
                )
            else:
                qa_memory_checks.update(
                    {
                        "index_scope": (
                            retrieval.get("index_scope")
                            == "global_native_train_plus_native_validation"
                        ),
                        "source_record_count": (
                            retrieval.get("source_record_count") == 97852
                        ),
                        "source_train_count": (
                            retrieval.get("source_train_count") == 90447
                        ),
                        "source_validation_count": (
                            retrieval.get("source_validation_count") == 7405
                        ),
                        "evaluation_overlap_count": (
                            retrieval.get("evaluation_overlap_count") == 128
                        ),
                        "contains_evaluation_source_facts": (
                            retrieval.get("contains_evaluation_source_facts") is True
                        ),
                        "contains_raw_questions": (
                            retrieval.get("contains_raw_questions") is False
                        ),
                        "contains_raw_answers": (
                            retrieval.get("contains_raw_answers") is False
                        ),
                        "document_format": (
                            retrieval.get("document_format")
                            == "declarative_fact_only"
                        ),
                        "indexed_text_field": (
                            retrieval.get("indexed_text_field") == "fact_text"
                        ),
                        "evaluation_scope": (
                            retrieval.get("evaluation_scope")
                            == "in_database_transductive"
                        ),
                        "official_heldout_eligible": (
                            retrieval.get("official_heldout_eligible") is False
                        ),
                    }
                )
            failed_qa_memory = [
                name for name, valid in qa_memory_checks.items() if not valid
            ]
            if failed_qa_memory:
                raise ConfigurationError(
                    "HotpotQA QA-memory condition is invalid: "
                    + ", ".join(failed_qa_memory)
                )
        for name in (
            "search_top_k",
            "max_turns_per_agent_call",
            "max_tool_calls_per_agent_call",
            "development_sample_count",
        ):
            value = retrieval.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(
                    f"qa_embedding_retrieval.{name} must be a positive integer"
                )


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
    paths = {name: _resolve(root, str(storage[field])) for name, field in names.items()}
    if storage.get("error_demos_path") is not None:
        paths["error_demos"] = _resolve(root, str(storage["error_demos_path"]))
    optional_names = {
        "retrieval_profile_selection": "retrieval_profile_selection_path",
        "retrieval_index_manifest": "retrieval_index_manifest_path",
        "retrieval_index_smoke": "retrieval_index_smoke_path",
        "retrieval_index_rebuild_smoke": "retrieval_index_rebuild_smoke_path",
        "paraphrase_manifest": "paraphrase_manifest_path",
    }
    for name, field in optional_names.items():
        if storage.get(field) is not None:
            paths[name] = _resolve(root, str(storage[field]))
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HotpotRoundError(f"{path}: expected one JSON object")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HotpotRoundError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise HotpotRoundError(f"{path}:{line_number}: expected an object")
            values.append(dict(value))
    return values


def _retrieval_evidence_names(
    retrieval_config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the existing retrieval evidence channels required by a corpus."""

    corpus_kind = retrieval_config.get("corpus_kind")
    if corpus_kind == "full_dataset_fact_memory":
        return (
            "retrieval_profile_selection",
            "retrieval_index_manifest",
            "retrieval_index_smoke",
            "paraphrase_manifest",
        )
    if corpus_kind == "transductive_qa_memory":
        return ("retrieval_index_manifest", "paraphrase_manifest")
    evidence_names = [
        "retrieval_profile_selection",
        "retrieval_index_manifest",
        "retrieval_index_smoke",
        "retrieval_index_rebuild_smoke",
    ]
    if corpus_kind == "train_qa_memory":
        evidence_names.append("paraphrase_manifest")
    return tuple(evidence_names)


def _validate_full_dataset_top_k_freeze(
    retrieval_config: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    top_k_selection: Mapping[str, Any],
) -> None:
    """Require the development selection, final index, and runner to agree."""

    selected_top_k = top_k_selection.get("selected_top_k")
    configured_top_k = retrieval_config.get("search_top_k")
    indexed_top_k = index_manifest.get("frozen_top_k")
    if (
        type(selected_top_k) is not int
        or selected_top_k != configured_top_k
        or indexed_top_k != configured_top_k
    ):
        raise HotpotRoundError(
            "fact-memory selected, configured, and indexed Top-K differ"
        )


def _validate_full_dataset_fact_memory_smoke(
    retrieval_config: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    smoke_receipt: Mapping[str, Any],
) -> None:
    """Validate only the fact Tool Adapter/receipt smoke boundary.

    Director isolation, worker Tool ownership in a real rollout, and relation
    routing are trajectory assertions and are deliberately not inferred from
    this scripted, model-free smoke.
    """

    smoke_manifest = smoke_receipt.get("index_manifest")
    if not isinstance(smoke_manifest, Mapping):
        raise HotpotRoundError("fact-memory smoke has no index manifest")
    required_true = (
        "passed",
        "same_query_top_k_deterministic",
        "worker_receipt_ownership_valid",
        "fact_only_search_read_projection_valid",
        "qa_fields_absent_from_retrieval_payload",
        "qa_wire_absent_from_retrieval_payload",
    )
    if any(smoke_receipt.get(name) is not True for name in required_true):
        raise HotpotRoundError("fact-memory Tool Adapter/receipt smoke failed")
    if (
        smoke_receipt.get("tool_id") != retrieval_config.get("tool_id")
        or smoke_receipt.get("web_search_used") is not False
        or smoke_receipt.get("model_api_calls") != 0
        or smoke_manifest.get("index_id") != index_manifest.get("index_id")
        or smoke_manifest.get("frozen_top_k")
        != retrieval_config.get("search_top_k")
    ):
        raise HotpotRoundError(
            "fact-memory smoke receipt differs from the frozen final index/config"
        )


def _read_and_validate_retrieval_evidence(
    retrieval_config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    """Read required retrieval evidence and fail closed on frozen-profile drift."""

    evidence_names = _retrieval_evidence_names(retrieval_config)
    missing_path_names = [name for name in evidence_names if name not in paths]
    if missing_path_names:
        raise HotpotRoundError(
            "required retrieval evidence path is not configured: "
            + ", ".join(missing_path_names)
        )
    missing = [str(paths[name]) for name in evidence_names if not paths[name].is_file()]
    if missing:
        raise HotpotRoundError(
            "required retrieval evidence artifact is missing: " + ", ".join(missing)
        )
    values = {name: _read_json(paths[name]) for name in evidence_names}
    if retrieval_config.get("corpus_kind") == "full_dataset_fact_memory":
        _validate_full_dataset_top_k_freeze(
            retrieval_config,
            values["retrieval_index_manifest"],
            values["retrieval_profile_selection"],
        )
        _validate_full_dataset_fact_memory_smoke(
            retrieval_config,
            values["retrieval_index_manifest"],
            values["retrieval_index_smoke"],
        )
    return values


def _validate_memory_retrieval_execution_boundary(
    boundary: Mapping[str, Any],
) -> None:
    """Require Director/worker/relation assertions from real trajectories."""

    if (
        boundary.get("director_tool_calls") != 0
        or not isinstance(boundary.get("retrieval_tool_calls_by_worker"), int)
        or int(boundary["retrieval_tool_calls_by_worker"]) < 1
        or boundary.get("retrieval_artifact_routed_via_relation") is not True
    ):
        raise HotpotRoundError(
            "memory Director/worker/relation execution assertions failed"
        )


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    _write_jsonl(temporary, values)
    temporary.replace(path)


def _select_tasks(config: Mapping[str, Any], root: Path, selected_path: Path) -> tuple[TaskRecord, ...]:
    data = _mapping(config["data"], "data")
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    count = int(bounded["sample_count"])
    source_path = _resolve(root, str(data["validation_path"]))
    candidates = tuple(
        task
        for task in iter_task_records(source_path, expected_split="validation")
        if _dataset_key(task) == "hotpotqa"
    )
    if len(candidates) < count:
        raise HotpotRoundError(
            f"validation contains only {len(candidates)} HotpotQA tasks; expected {count}"
        )
    expected = candidates[:count]
    if selected_path.exists():
        frozen = tuple(iter_task_records(selected_path, expected_split="validation"))
        if len(frozen) != count:
            raise HotpotRoundError("frozen HotpotQA selection has the wrong size")
        for expected_task, frozen_task in zip(expected, frozen, strict=True):
            if (
                expected_task.task_id != frozen_task.task_id
                or expected_task.question != frozen_task.question
                or expected_task.ground_truth != frozen_task.ground_truth
            ):
                raise HotpotRoundError(
                    "frozen HotpotQA task batch differs from the aligned validation data"
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


def _by_task(values: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        task_id = value.get("task_id")
        if not isinstance(task_id, str):
            task = value.get("task")
            task_id = task.get("task_id") if isinstance(task, Mapping) else None
        if isinstance(task_id, str) and task_id not in result:
            result[task_id] = dict(value)
    return result


def _persist_ordered(
    path: Path,
    selected: Sequence[TaskRecord],
    values: Mapping[str, Mapping[str, Any]],
) -> None:
    _atomic_jsonl(path, [values[task.task_id] for task in selected if task.task_id in values])


async def _direct_one(
    backend: LiveSmokeBackend,
    task: TaskRecord,
    index: int,
    *,
    model_id: str,
    protocol: str,
    seed: int,
) -> Mapping[str, Any]:
    model = backend.registry.require_model(model_id)
    provider = backend.registry.provider_for(model_id)
    run_id = f"hotpotqa-round-01-direct-{index:04d}"
    request = AgentRequest(
        request_id=f"{run_id}:direct:single",
        run_id=run_id,
        graph_revision=0,
        problem=task.question,
        agent=AgentNode("direct", model_id, DIRECT_CONTRACT),
        model=model,
        provider=provider,
        phase=ExecutionPhase.SINGLE,
    )
    started_at = _utc_now()
    response = await backend.runtime.gateway.generate(request)
    execution = execution_record_from_call(AgentCallRecord(request, response))
    actual_seed = execution.metadata.get("response", {}).get("generation_seed")
    if actual_seed != seed:
        raise HotpotRoundError("Direct generation seed receipt differs from config")
    evaluation = await evaluate_task(task, response.text)
    return {
        "schema_version": "flowsteer.hotpotqa.direct_prediction.v1",
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


def _direct_resume_matches(
    value: Mapping[str, Any],
    *,
    task: TaskRecord,
    model_id: str,
    protocol: str,
    seed: int,
) -> bool:
    evaluation = value.get("evaluation")
    execution = value.get("execution")
    return bool(
        value.get("task_id") == task.task_id
        and value.get("model_id") == model_id
        and value.get("protocol") == protocol
        and value.get("generation_seed") == seed
        and isinstance(evaluation, Mapping)
        and isinstance(execution, Mapping)
        and evaluation.get("valid") is True
    )


def _trajectory_resume_matches(
    value: Mapping[str, Any],
    *,
    task: TaskRecord,
    condition_id: str,
    versions: Mapping[str, str],
) -> bool:
    embedded = value.get("task")
    return bool(
        isinstance(embedded, Mapping)
        and embedded.get("task_id") == task.task_id
        and value.get("condition_id") == condition_id
        and value.get("versions") == versions
        and isinstance(value.get("turns"), list)
        and isinstance(value.get("evaluation"), Mapping)
        and isinstance(value.get("explicit_finish"), bool)
    )


async def _collect_direct(
    backend: LiveSmokeBackend,
    selected: Sequence[TaskRecord],
    config: Mapping[str, Any],
    path: Path,
    failures_path: Path,
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    experiment = _mapping(config["experiment"], "experiment")
    model_id = str(bounded["direct_model_id"])
    protocol = str(bounded["direct_protocol"])
    seed = int(experiment["seed"])
    concurrency = int(bounded["concurrency"])
    source_value = bounded.get("direct_source_path")
    if source_value is not None:
        source_path = _resolve(PROJECT_ROOT, str(source_value))
        source_by_task = _by_task(_read_jsonl(source_path))
        rescored: dict[str, dict[str, Any]] = {}
        for task in selected:
            source = source_by_task.get(task.task_id)
            if source is None:
                raise HotpotRoundError(
                    f"saved Direct source is missing frozen task {task.task_id}"
                )
            answer = source.get("final_answer")
            if not isinstance(answer, str):
                raise HotpotRoundError(
                    f"saved Direct source has no text answer for {task.task_id}"
                )
            evaluation = await evaluate_task(task, answer)
            if not evaluation.valid:
                raise HotpotRoundError(
                    f"saved Direct answer is evaluator-invalid for {task.task_id}"
                )
            rescored[task.task_id] = {
                "schema_version": "flowsteer.hotpotqa.direct_rescore.v1",
                "task_id": task.task_id,
                "task": task.to_dict(),
                "condition": "direct_local_qwen35_9b_official_rescore",
                "protocol": protocol,
                "model_id": source.get("model_id", model_id),
                "provider_id": source.get("provider_id"),
                "provider_model": source.get("provider_model"),
                "generation_seed": source.get("generation_seed", seed),
                "final_answer": answer,
                "evaluation": asdict(evaluation),
                "execution": source.get("execution"),
                "source_prediction_path": str(source_path),
                "model_call_reused": True,
                "completed_at": _utc_now(),
            }
        _persist_ordered(path, selected, rescored)
        manifest["direct_progress"] = {
            "completed": len(rescored),
            "model_calls": 0,
            "saved_predictions_rescored": len(rescored),
        }
        _write_json(manifest_path, manifest)
        return rescored
    by_task = {
        task_id: value
        for task_id, value in _by_task(_read_jsonl(path)).items()
        for task in selected
        if task.task_id == task_id
        and _direct_resume_matches(
            value,
            task=task,
            model_id=model_id,
            protocol=protocol,
            seed=seed,
        )
    }
    manifest["direct_progress"] = {
        "completed": len(by_task),
        "failed_attempts": sum(
            item.get("condition") == "direct_local_qwen35_9b" for item in failures
        ),
    }
    _write_json(manifest_path, manifest)
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
                    seed=seed,
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
            _atomic_jsonl(failures_path, failures)
        else:
            by_task[task.task_id] = dict(result)
            _persist_ordered(path, selected, by_task)
        manifest["direct_progress"] = {
            "completed": len(by_task),
            "failed_attempts": sum(
                item.get("condition") == "direct_local_qwen35_9b" for item in failures
            ),
        }
        _write_json(manifest_path, manifest)
    return by_task


async def _collect_graph(
    backend: LiveSmokeBackend,
    selected: Sequence[TaskRecord],
    config: Mapping[str, Any],
    path: Path,
    failures_path: Path,
    failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    bounded = _mapping(config["hotpotqa_evaluation"], "hotpotqa_evaluation")
    experiment = _mapping(config["experiment"], "experiment")
    director = _mapping(config["director"], "director")
    policy = str(director["behavior_policy_version"])
    condition_id = str(experiment["condition_id"])
    versions = {
        task.task_id: version_bundle_for(
            task,
            policy_version=policy,
            model_catalog_version=backend.model_catalog_version,
            prompt_version=str(experiment["prompt_version"]),
            tool_version=str(experiment["tool_version"]),
        )
        for task in selected
    }
    by_task = {
        task_id: value
        for task_id, value in _by_task(_read_jsonl(path)).items()
        for task in selected
        if task.task_id == task_id
        and _trajectory_resume_matches(
            value,
            task=task,
            condition_id=condition_id,
            versions=versions[task_id].to_dict(),
        )
    }
    # The collector commits to EvidenceStore before returning to this runner.
    # Recover that authoritative payload first if the process stopped between
    # the append and its ordered mirror checkpoint; never repay a successful
    # provider call merely because the mirror write was interrupted.
    selected_by_id = {task.task_id: task for task in selected}
    for value in backend.evidence_store.trajectories.payloads():
        embedded = value.get("task")
        task_id = embedded.get("task_id") if isinstance(embedded, Mapping) else None
        selected_task = selected_by_id.get(task_id)
        if (
            selected_task is not None
            and task_id not in by_task
            and _trajectory_resume_matches(
                value,
                task=selected_task,
                condition_id=condition_id,
                versions=versions[task_id].to_dict(),
            )
        ):
            by_task[task_id] = dict(value)
    _persist_ordered(path, selected, by_task)
    manifest["agentgraph_progress"] = {
        "completed": len(by_task),
        "failed_attempts": sum(
            item.get("condition") == "agentgraph" for item in failures
        ),
    }
    _write_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(int(bounded["concurrency"]))

    async def run(index: int, task: TaskRecord) -> tuple[TaskRecord, Any]:
        async with semaphore:
            try:
                trajectory = await backend.collect(
                    task,
                    index,
                    versions[task.task_id],
                    expected_task_split="validation",
                )
                return task, trajectory
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
                    "condition": "agentgraph",
                    "stage": "director_canvas_runtime_or_evaluator",
                    "error": _safe_error(result),
                    "recorded_at": _utc_now(),
                }
            )
            # Keep failure recovery as durable as successful trajectory
            # recovery. A later process restart must not repay a failed
            # provider/runtime attempt merely because the batch had not yet
            # reached its final mirror checkpoint.
            _atomic_jsonl(failures_path, failures)
        else:
            if not isinstance(result, TrajectoryRecord):
                raise HotpotRoundError("backend returned a non-trajectory result")
            by_task[task.task_id] = result.to_dict()
            _persist_ordered(path, selected, by_task)
        manifest["agentgraph_progress"] = {
            "completed": len(by_task),
            "failed_attempts": sum(
                item.get("condition") == "agentgraph" for item in failures
            ),
        }
        _write_json(manifest_path, manifest)
    return by_task


def _metrics(value: Optional[Mapping[str, Any]]) -> tuple[bool, float, float]:
    if value is None:
        return False, 0.0, 0.0
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("valid") is not True:
        return False, 0.0, 0.0
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, 0.0, 0.0
    return (
        True,
        float(metrics.get("exact_match", 0.0)),
        float(metrics.get("token_f1", 0.0)),
    )


def _output_inbox(trajectory: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if trajectory is None:
        return None
    turns = trajectory.get("turns")
    final_graph = turns[-1].get("graph_snapshot") if isinstance(turns, list) and turns else None
    if not isinstance(final_graph, Mapping):
        return None
    output_id = final_graph.get("output_agent_id")
    if not isinstance(output_id, str):
        return None
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        executions = turn.get("executions", ())
        if not isinstance(executions, list):
            continue
        for execution in reversed(executions):
            if not isinstance(execution, Mapping) or execution.get("agent_id") != output_id:
                continue
            metadata = execution.get("metadata")
            request = metadata.get("request") if isinstance(metadata, Mapping) else None
            if not isinstance(request, Mapping):
                continue
            return {
                "output_agent_id": output_id,
                "execution_id": execution.get("execution_id"),
                "agent": request.get("agent"),
                "model": request.get("model"),
                "phase": request.get("phase"),
                "upstream": request.get("upstream"),
                "own_draft": request.get("own_draft"),
                "peer_draft": request.get("peer_draft"),
                "rendered_messages": request.get("rendered_messages"),
            }
    return None


def _graph_telemetry(trajectory: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if trajectory is None:
        return {
            "director_turns": 0,
            "executor_calls": 0,
            "api_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        turns = []
    calls = [
        execution
        for turn in turns
        if isinstance(turn, Mapping)
        for execution in turn.get("executions", ())
        if isinstance(execution, Mapping)
    ]
    director_attempts = sum(
        int(turn.get("director_attempt_count") or 0)
        for turn in turns
        if isinstance(turn, Mapping)
    )
    executor_attempts = 0
    for call in calls:
        metadata = call.get("metadata")
        response = metadata.get("response") if isinstance(metadata, Mapping) else None
        if isinstance(response, Mapping):
            executor_attempts += int(response.get("attempt_count") or 0)
    return {
        "director_turns": len(turns),
        "executor_calls": len(calls),
        "api_attempts": director_attempts + executor_attempts,
        "input_tokens": sum(
            len(turn.get("prompt_token_ids", ()))
            for turn in turns
            if isinstance(turn, Mapping)
        )
        + sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(
            len(turn.get("output_token_ids", ()))
            for turn in turns
            if isinstance(turn, Mapping)
        )
        + sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_ms": sum(
            float(turn.get("director_latency_ms") or 0.0)
            for turn in turns
            if isinstance(turn, Mapping)
        )
        + sum(float(call.get("latency_ms") or 0.0) for call in calls),
    }


def _tool_receipts(trajectory: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if trajectory is None:
        return []
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return []
    receipts: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        executions = turn.get("executions", ())
        if not isinstance(executions, list):
            continue
        for execution in executions:
            if not isinstance(execution, Mapping):
                continue
            metadata = execution.get("metadata")
            response = metadata.get("response") if isinstance(metadata, Mapping) else None
            raw = response.get("tool_receipts", ()) if isinstance(response, Mapping) else ()
            if not isinstance(raw, list):
                continue
            receipts.extend(dict(item) for item in raw if isinstance(item, Mapping))
    return receipts


def _tool_statistics(
    trajectories: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_task = {task_id: _tool_receipts(value) for task_id, value in trajectories.items()}
    receipts = [receipt for values in by_task.values() for receipt in values]
    actions = [
        str(receipt.get("request", {}).get("action", ""))
        for receipt in receipts
        if isinstance(receipt.get("request"), Mapping)
    ]
    rewritten = 0
    for values in by_task.values():
        queries = []
        for receipt in values:
            request = receipt.get("request")
            if not isinstance(request, Mapping) or request.get("action") != "search":
                continue
            arguments = request.get("arguments")
            query = arguments.get("query") if isinstance(arguments, Mapping) else None
            if isinstance(query, str) and query not in queries:
                queries.append(query)
        rewritten += len(queries) > 1
    return {
        "tool_invoked_tasks": sum(bool(values) for values in by_task.values()),
        "tool_calls": len(receipts),
        "successful_calls": sum(receipt.get("error_type") is None for receipt in receipts),
        "failed_calls": sum(receipt.get("error_type") is not None for receipt in receipts),
        "search_calls": actions.count("search"),
        "read_calls": actions.count("read"),
        "query_rewrite_tasks": rewritten,
        "latency_ms": sum(float(receipt.get("latency_ms") or 0.0) for receipt in receipts),
    }


def _directed_path_exists(
    graph: Mapping[str, Any], source_id: str, target_id: str
) -> bool:
    if source_id == target_id:
        return True
    relations = graph.get("relations", ())
    if not isinstance(relations, list):
        return False
    successors: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if not isinstance(relation, Mapping):
            continue
        source = relation.get("source_id")
        target = relation.get("target_id")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if relation.get("source_to_target") is True:
            successors[source].add(target)
        if relation.get("target_to_source") is True:
            successors[target].add(source)
    pending = [source_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if target_id in successors.get(current, set()):
            return True
        pending.extend(successors.get(current, set()) - visited)
    return False


def _retrieval_boundary_statistics(
    trajectories: Mapping[str, Mapping[str, Any]], *, tool_id: str
) -> Mapping[str, Any]:
    worker_calls = 0
    invoked_tasks = 0
    finished_invoked_tasks = 0
    routed_tasks = 0
    worker_agent_ids: set[str] = set()
    memory_ids: set[str] = set()
    base_task_ids: set[str] = set()
    director_tool_calls = 0
    for trajectory in trajectories.values():
        turns = trajectory.get("turns", ())
        if not isinstance(turns, list):
            continue
        task_worker_ids: set[str] = set()
        final_graph: Mapping[str, Any] = {}
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            action = turn.get("action")
            if isinstance(action, Mapping) and (
                action.get("action") in {"search", "read"}
                or action.get("kind") in {"tool", "skill"}
            ):
                director_tool_calls += 1
            graph = turn.get("graph_snapshot")
            if isinstance(graph, Mapping):
                final_graph = graph
            executions = turn.get("executions", ())
            if not isinstance(executions, list):
                continue
            for execution in executions:
                if not isinstance(execution, Mapping):
                    continue
                agent_id = execution.get("agent_id")
                metadata = execution.get("metadata")
                response = metadata.get("response") if isinstance(metadata, Mapping) else None
                receipts = response.get("tool_receipts", ()) if isinstance(response, Mapping) else ()
                if not isinstance(agent_id, str) or not isinstance(receipts, list):
                    continue
                matching = [
                    receipt
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                    and receipt.get("tool_id") == tool_id
                ]
                if not matching:
                    continue
                task_worker_ids.add(agent_id)
                worker_agent_ids.add(agent_id)
                worker_calls += len(matching)
                for receipt in matching:
                    result = receipt.get("result")
                    value = result.get("value") if isinstance(result, Mapping) else None
                    if not isinstance(value, Mapping):
                        continue
                    for raw_id in value.get("memory_ids", value.get("doc_ids", ())):
                        if isinstance(raw_id, str):
                            memory_ids.add(raw_id)
                    hits = value.get("hits", ())
                    if isinstance(hits, list):
                        for hit in hits:
                            if not isinstance(hit, Mapping):
                                continue
                            base_id = hit.get(
                                "source_train_task_id",
                                hit.get("base_task_id"),
                            )
                            if isinstance(base_id, str):
                                base_task_ids.add(base_id)
        if task_worker_ids:
            invoked_tasks += 1
            output_agent_id = final_graph.get("output_agent_id")
            if trajectory.get("explicit_finish") is True:
                finished_invoked_tasks += 1
                if isinstance(output_agent_id, str) and all(
                    _directed_path_exists(final_graph, worker_id, output_agent_id)
                    for worker_id in task_worker_ids
                ):
                    routed_tasks += 1
    return {
        "director_tool_calls": director_tool_calls,
        "retrieval_tool_calls_by_worker": worker_calls,
        "retrieval_worker_agent_ids": sorted(worker_agent_ids),
        "retrieval_invoked_tasks": invoked_tasks,
        "finished_retrieval_invoked_tasks": finished_invoked_tasks,
        "retrieval_artifact_routed_tasks": routed_tasks,
        "retrieval_artifact_routed_via_relation": (
            finished_invoked_tasks > 0 and routed_tasks == finished_invoked_tasks
        ),
        "unique_memory_ids_retrieved": len(memory_ids),
        "unique_train_base_task_ids_retrieved": len(base_task_ids),
    }


def _failure_type(
    direct: Optional[Mapping[str, Any]],
    trajectory: Optional[Mapping[str, Any]],
    direct_em: float,
    graph_em: float,
    graph_f1: float,
) -> str:
    if trajectory is None:
        return "agentgraph_operational_failure"
    if trajectory.get("explicit_finish") is not True:
        return "director_max_rounds"
    turns = trajectory.get("turns", ())
    feedback = " ".join(
        str(turn.get("canvas_feedback", ""))
        for turn in turns
        if isinstance(turn, Mapping)
    )
    if "execution_error=" in feedback:
        return "executor_or_provider_failure"
    if direct is None:
        return "direct_operational_failure_comparison_unavailable"
    if direct_em == 1.0 and graph_em == 0.0:
        return "architecture_regression_candidate"
    if direct_em == 0.0 and graph_em == 1.0:
        return "architecture_gain"
    if graph_em == 0.0 and graph_f1 > 0.0:
        return "partial_or_overlong_answer"
    if direct_em == 0.0 and graph_em == 0.0:
        return "shared_reasoning_or_model_failure_candidate"
    return "correct"


def _paired_rows(
    selected: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for task in selected:
        direct_value = direct.get(task.task_id)
        graph_value = trajectories.get(task.task_id)
        direct_valid, direct_em, direct_f1 = _metrics(direct_value)
        graph_valid, graph_em, graph_f1 = _metrics(graph_value)
        direct_execution = direct_value.get("execution") if direct_value else None
        direct_response = (
            direct_execution.get("metadata", {}).get("response", {})
            if isinstance(direct_execution, Mapping)
            else {}
        )
        direct_telemetry = {
            "api_attempts": int(direct_response.get("attempt_count") or 0),
            "input_tokens": int(direct_execution.get("input_tokens") or 0)
            if isinstance(direct_execution, Mapping)
            else 0,
            "output_tokens": int(direct_execution.get("output_tokens") or 0)
            if isinstance(direct_execution, Mapping)
            else 0,
            "latency_ms": float(direct_execution.get("latency_ms") or 0.0)
            if isinstance(direct_execution, Mapping)
            else 0.0,
        }
        rows.append(
            {
                "task_id": task.task_id,
                "question": task.question,
                "ground_truth": task.ground_truth,
                "direct": {
                    "available": direct_value is not None,
                    "valid": direct_valid,
                    "final_answer": direct_value.get("final_answer") if direct_value else None,
                    "exact_match": direct_em,
                    "token_f1": direct_f1,
                    "telemetry": direct_telemetry,
                    "execution": direct_execution,
                },
                "agentgraph": {
                    "available": graph_value is not None,
                    "valid": graph_valid,
                    "final_answer": graph_value.get("final_answer") if graph_value else None,
                    "exact_match": graph_em,
                    "token_f1": graph_f1,
                    "explicit_finish": graph_value.get("explicit_finish")
                    if graph_value
                    else False,
                    "termination_reason": graph_value.get("termination_reason")
                    if graph_value
                    else "collection_failed",
                    "trajectory_id": graph_value.get("trajectory_id") if graph_value else None,
                    "final_graph": (
                        graph_value.get("turns", [{}])[-1].get("graph_snapshot")
                        if graph_value and graph_value.get("turns")
                        else None
                    ),
                    "output_agent_inbox": _output_inbox(graph_value),
                    "telemetry": _graph_telemetry(graph_value),
                },
                "delta_exact_match": graph_em - direct_em,
                "delta_token_f1": graph_f1 - direct_f1,
                "failure_type": _failure_type(
                    direct_value,
                    graph_value,
                    direct_em,
                    graph_em,
                    graph_f1,
                ),
            }
        )
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]], condition: str) -> Mapping[str, Any]:
    total = len(rows)
    values = [row[condition] for row in rows]
    valid = [value for value in values if value.get("valid") is True]
    return {
        "denominator": total,
        "completed": sum(value.get("available") is True for value in values),
        "evaluator_valid": len(valid),
        "strict_exact_match": (
            sum(float(value.get("exact_match", 0.0)) for value in values) / total
            if total
            else 0.0
        ),
        "strict_token_f1": (
            sum(float(value.get("token_f1", 0.0)) for value in values) / total
            if total
            else 0.0
        ),
        "completed_only_exact_match": (
            sum(float(value.get("exact_match", 0.0)) for value in valid) / len(valid)
            if valid
            else None
        ),
        "completed_only_token_f1": (
            sum(float(value.get("token_f1", 0.0)) for value in valid) / len(valid)
            if valid
            else None
        ),
    }


async def _official_rescore_saved_agentgraph(
    selected: Sequence[TaskRecord],
    source_path: Path,
) -> Mapping[str, Any]:
    """Rescore saved Round-01 text without making any model or Tool call."""

    source_by_task = _by_task(_read_jsonl(source_path))
    exact_match = 0.0
    token_f1 = 0.0
    valid = 0
    for task in selected:
        source = source_by_task.get(task.task_id)
        if source is None:
            continue
        answer = source.get("final_answer")
        if not isinstance(answer, str):
            answer = ""
        evaluation = await evaluate_task(task, answer)
        if not evaluation.valid:
            continue
        valid += 1
        exact_match += float(evaluation.metrics.get("exact_match", 0.0))
        token_f1 += float(evaluation.metrics.get("token_f1", 0.0))
    denominator = len(selected)
    return {
        "source_path": str(source_path),
        "denominator": denominator,
        "completed": sum(task.task_id in source_by_task for task in selected),
        "evaluator_valid": valid,
        "strict_exact_match": exact_match / denominator if denominator else 0.0,
        "strict_token_f1": token_f1 / denominator if denominator else 0.0,
        "evaluator_version": "hotpotqa.official.answer.v1",
        "model_calls": 0,
    }


def _input_context(config: Mapping[str, Any]) -> str:
    retrieval = config.get("qa_embedding_retrieval")
    if not isinstance(retrieval, Mapping):
        return "full_10_passages"
    task_scope, _ = qa_retrieval_scopes(retrieval)
    scope_prefix = (
        "public_task" if task_scope == "public_task" else "question_only"
    )
    if retrieval.get("corpus_kind") == "train_qa_memory":
        return f"{scope_prefix}_dynamic_train_qa_memory_search_read"
    if retrieval.get("corpus_kind") == "transductive_qa_memory":
        return f"{scope_prefix}_dynamic_transductive_qa_memory_search_read"
    if retrieval.get("corpus_kind") == "full_dataset_fact_memory":
        return f"{scope_prefix}_dynamic_full_dataset_fact_memory_search_read"
    return f"{scope_prefix}_dynamic_embedding_search_read"


def _report(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _aggregate(rows, "direct")
    graph = _aggregate(rows, "agentgraph")
    failure_counts = Counter(str(row["failure_type"]) for row in rows)
    wrong = [row for row in rows if float(row["agentgraph"]["exact_match"]) < 1.0]
    retrieval = config.get("qa_embedding_retrieval")
    retrieval_mapping = retrieval if isinstance(retrieval, Mapping) else {}
    return {
        "schema_version": "flowsteer.hotpotqa.round_report.v1",
        "dataset": "HotpotQA",
        "project_split": "validation",
        "native_source_split": "train",
        "evaluation_regime": retrieval_mapping.get(
            "evaluation_regime", "heldout_evaluation"
        ),
        "official_heldout_eligible": retrieval_mapping.get(
            "official_heldout_eligible", True
        ),
        "input_context": _input_context(config),
        "sample_count": len(rows),
        "direct_local_baseline": direct,
        "agentgraph": graph,
        "agentgraph_minus_direct": {
            "exact_match": graph["strict_exact_match"] - direct["strict_exact_match"],
            "token_f1": graph["strict_token_f1"] - direct["strict_token_f1"],
        },
        "paper_references_percent": {
            "qwen3.5_9b_direct": {"exact_match": 60.94, "token_f1": 75.70},
            "flowsteer": {"exact_match": 89.84, "token_f1": 91.20},
            "skillflow": {"exact_match": 92.19, "token_f1": 93.95},
            "comparability_note": (
                "Reference numbers are contextual only; the paired Local Direct "
                "baseline on these fixed project-held-out samples is primary."
            ),
        },
        "failure_types": dict(sorted(failure_counts.items())),
        "wrong_demo_count": len(wrong),
        "terminal_failures": sum(
            row["agentgraph"].get("explicit_finish") is not True for row in rows
        ),
        "typical_wrong_demo_task_ids": [row["task_id"] for row in wrong[:10]],
        "policy_version": config["director"]["behavior_policy_version"],
        "policy_adapter": config["director"]["behavior_adapter_name"],
        "model_catalog_path": config["agent_graph"]["model_catalog_path"],
        "training_performed": False,
        "method_level_changes_performed": False,
        "known_limitations": [
            "A Director/API exception before terminal collection may not preserve partial turns.",
            "Paper references use a different published evaluation setup and are not a paired baseline.",
        ],
        "completed_at": _utc_now(),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    direct = report["direct_local_baseline"]
    graph = report["agentgraph"]
    delta = report["agentgraph_minus_direct"]
    failures = report["failure_types"]
    failure_lines = "\n".join(f"- `{name}`: {count}" for name, count in failures.items())
    tool = report.get("tool_usage") if isinstance(report.get("tool_usage"), Mapping) else {}
    boundary = (
        report.get("retrieval_execution_boundary")
        if isinstance(report.get("retrieval_execution_boundary"), Mapping)
        else {}
    )
    retrieval_memory = (
        report.get("retrieval_memory")
        if isinstance(report.get("retrieval_memory"), Mapping)
        else {}
    )
    baseline = (
        report.get("round01_agentgraph_official_rescore")
        if isinstance(report.get("round01_agentgraph_official_rescore"), Mapping)
        else None
    )
    baseline_line = ""
    if baseline is not None:
        baseline_line = (
            "\nRound-01 saved AgentGraph outputs rescored with the same official "
            f"answer evaluator: **{100 * baseline['strict_exact_match']:.2f} EM**, "
            f"**{100 * baseline['strict_token_f1']:.2f} F1**.\n"
        )
    input_context = str(report.get("input_context", ""))
    if input_context.endswith("_dynamic_train_qa_memory_search_read"):
        input_description = (
            "The Director receives the original question and control-plane Canvas "
            "receipts only. Tool-capable worker Agents dynamically search/read the "
            "global train-only QA-memory and route evidence through graph relations."
        )
    elif input_context.endswith("_dynamic_transductive_qa_memory_search_read"):
        input_description = (
            "The Director receives the public task and control-plane receipts only. "
            "Tool-capable worker Agents dynamically search/read the transductive "
            "QA-memory and route evidence through explicit graph relations."
        )
    elif input_context.endswith("_dynamic_full_dataset_fact_memory_search_read"):
        input_description = (
            "The Director receives the public task and control-plane receipts only. "
            "Tool-capable worker Agents dynamically search/read the full-dataset "
            "declarative-fact memory and route evidence through explicit graph "
            "relations."
        )
    elif input_context.endswith("_dynamic_embedding_search_read"):
        input_description = (
            "The Director and Agent Runtime receive only the original question; "
            "public passages are obtained dynamically through the task-scoped "
            "embedding search/read Tool."
        )
    else:
        input_description = "The model input uses all ten supplied passages."
    report_title = (
        "HotpotQA Architecture Validation — Dynamic Embedding Retrieval"
        if "_dynamic_" in input_context
        else "HotpotQA Architecture Validation — Round 01"
    )
    evaluation_scope = str(
        report.get("evaluation_regime", "heldout_evaluation")
    )
    sample_scope = (
        "Fixed evaluation samples"
        if report.get("official_heldout_eligible") is False
        else "Fixed project-held-out samples"
    )
    return f"""# {report_title}

{sample_scope}: **{report['sample_count']}**. Evaluation scope: **{evaluation_scope}**. {input_description} No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | {direct['completed']} | {direct['evaluator_valid']} | {100 * direct['strict_exact_match']:.2f} | {100 * direct['strict_token_f1']:.2f} |
| AgentGraph | {graph['completed']} | {graph['evaluator_valid']} | {100 * graph['strict_exact_match']:.2f} | {100 * graph['strict_token_f1']:.2f} |

AgentGraph − Direct: **{100 * delta['exact_match']:+.2f} EM**, **{100 * delta['token_f1']:+.2f} F1**.
{baseline_line}
Terminal failures: **{report.get('terminal_failures', 0)}**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **{tool.get('tool_invoked_tasks', 0)}**
- Calls: **{tool.get('tool_calls', 0)}** (`search`={tool.get('search_calls', 0)}, `read`={tool.get('read_calls', 0)})
- Successful / failed calls: **{tool.get('successful_calls', 0)} / {tool.get('failed_calls', 0)}**
- Tasks with query rewriting: **{tool.get('query_rewrite_tasks', 0)}**
- Director Tool calls: **{boundary.get('director_tool_calls', 0)}**
- Worker retrieval Tool calls: **{boundary.get('retrieval_tool_calls_by_worker', 0)}**
- Retrieval artifact routed via AgentGraph relation: **{boundary.get('retrieval_artifact_routed_via_relation', False)}**
- Retrieval records / unique sources / cycled: **{retrieval_memory.get('source_record_count', retrieval_memory.get('train_record_count', 'N/A'))} / {retrieval_memory.get('unique_source_count', 'N/A')} / {retrieval_memory.get('cycled_record_count', 'N/A')}**

## Failure types

{failure_lines or '- None'}

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
"""


def _stable_zero_check(
    tasks: Sequence[TaskRecord],
    direct: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    require_dynamic_retrieval: bool = False,
    retrieval_tool_id: str = "qa-retrieval",
) -> Mapping[str, Any]:
    checks: list[dict[str, Any]] = []
    for task in tasks:
        direct_value = direct.get(task.task_id)
        trajectory = trajectories.get(task.task_id)
        turns = trajectory.get("turns", ()) if trajectory else ()
        full_turn_receipts = bool(turns) and all(
            isinstance(turn, Mapping)
            and turn.get("receipt_verified") is True
            and turn.get("director_attempt_count")
            and turn.get("director_generation_seed") is not None
            and turn.get("director_latency_ms") is not None
            for turn in turns
        )
        tool_receipts = _tool_receipts(trajectory)
        retrieval_receipt = any(
            receipt.get("tool_id") == retrieval_tool_id
            and receipt.get("error_type") is None
            for receipt in tool_receipts
        )
        passed = bool(
            direct_value
            and trajectory
            and trajectory.get("explicit_finish") is True
            and trajectory.get("final_answer") not in (None, "")
            and trajectory.get("evaluation", {}).get("valid") is True
            and _output_inbox(trajectory) is not None
            and full_turn_receipts
            and (not require_dynamic_retrieval or retrieval_receipt)
        )
        checks.append(
            {
                "task_id": task.task_id,
                "passed": passed,
                "direct_complete": direct_value is not None,
                "agentgraph_complete": trajectory is not None,
                "explicit_finish": trajectory.get("explicit_finish") if trajectory else False,
                "output_inbox_saved": _output_inbox(trajectory) is not None,
                "full_turn_receipts": full_turn_receipts,
                "dynamic_retrieval_receipt": retrieval_receipt,
            }
        )
    passed = any(check["passed"] for check in checks)
    return {
        "passed": passed,
        "criterion": (
            "at_least_one_real_fixed_task_completed_full_chain_with_dynamic_retrieval"
            if require_dynamic_retrieval
            else "at_least_one_real_fixed_task_completed_the_full_chain"
        ),
        "checks": checks,
    }


async def run_hotpot_round(
    config_path: str | Path,
    *,
    project_root: Optional[str | Path] = None,
    prepare_only: bool = False,
    canary_only: bool = False,
) -> Mapping[str, Any]:
    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_hotpot_config(config)
    paths = _paths(config, root)
    selected = _select_tasks(config, root, paths["selected"])
    failures = _read_jsonl(paths["failures"])
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.hotpotqa.round_manifest.v1",
        "status": "prepared" if prepare_only else "runtime_preflight",
        "started_at": _utc_now(),
        "config_path": str(resolved_config),
        "git_start": _git_state(root),
        "selected_task_ids": [task.task_id for task in selected],
        "sample_count": len(selected),
        "fixed_split": "validation",
        "input_context": _input_context(config),
        "training_enabled": False,
        "optimizer_updates": 0,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    retrieval_configuration = config.get("qa_embedding_retrieval")
    if isinstance(retrieval_configuration, Mapping):
        manifest["evaluation_regime"] = retrieval_configuration.get(
            "evaluation_regime", "heldout_evaluation"
        )
        manifest["contains_evaluation_answers"] = retrieval_configuration.get(
            "contains_evaluation_answers", False
        )
        manifest["contains_evaluation_source_facts"] = (
            retrieval_configuration.get("contains_evaluation_source_facts", False)
        )
        manifest["contains_raw_questions"] = retrieval_configuration.get(
            "contains_raw_questions", False
        )
        manifest["contains_raw_answers"] = retrieval_configuration.get(
            "contains_raw_answers", False
        )
        manifest["official_heldout_eligible"] = retrieval_configuration.get(
            "official_heldout_eligible", True
        )
    _write_json(paths["manifest"], manifest)
    if prepare_only:
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        return manifest

    if (
        isinstance(retrieval_configuration, Mapping)
        and retrieval_configuration.get("corpus_kind")
        == "full_dataset_fact_memory"
    ):
        try:
            evidence = _read_and_validate_retrieval_evidence(
                retrieval_configuration,
                paths,
            )
            manifest["retrieval_evidence_preflight"] = {
                "passed": True,
                "scope": "fact_tool_adapter_and_receipt_only",
                "artifacts": {
                    name: str(paths[name]) for name in evidence
                },
                "trajectory_assertions_deferred": [
                    "director_tool_calls=0",
                    "retrieval_tool_calls_by_worker>0",
                    "retrieval_artifact_routed_via_relation=true",
                ],
            }
            _write_json(paths["manifest"], manifest)
        except Exception as exc:
            manifest.update(
                status="failed_retrieval_evidence_preflight",
                error=_safe_error(exc),
                completed_at=_utc_now(),
            )
            _write_json(paths["manifest"], manifest)
            raise HotpotRoundError(
                "HotpotQA retrieval evidence preflight failed"
            ) from exc

    try:
        backend = LiveSmokeBackend.from_config(config, root, evaluation_only=True)
        director = _mapping(config["director"], "director")
        preflight = await asyncio.to_thread(
            backend.publisher.ensure_loaded_adapter,
            checkpoint_path=str(director["behavior_adapter_checkpoint"]),
            adapter_name=str(director["behavior_adapter_name"]),
        )
        known_answer = await evaluate_task(
            selected[0], f"<answer>{selected[0].ground_truth}</answer>"
        )
        if (
            not known_answer.valid
            or known_answer.metrics.get("exact_match") != 1.0
            or known_answer.metrics.get("token_f1") != 1.0
        ):
            raise HotpotRoundError("known-answer Hotpot evaluator preflight failed")
        preflight = {
            **dict(preflight),
            "evaluator_known_answer": asdict(known_answer),
        }
        director_payload = backend.director_client.request_payload(
            "Canvas control-plane preflight",
            seed=int(_mapping(config["experiment"], "experiment")["seed"]),
        )
        forbidden_director_fields = sorted(
            set(director_payload)
            & {"tools", "tool_choice", "allowed_tools", "retrieval", "documents"}
        )
        if forbidden_director_fields:
            raise HotpotRoundError(
                "Director request unexpectedly exposes Tool/retrieval fields"
            )
        preflight["director_request_boundary"] = {
            "director_tool_calls": 0,
            "allowed_tools": [],
            "retrieval_payload_present": False,
            "forbidden_fields_present": forbidden_director_fields,
        }
        _write_json(paths["preflight"], preflight)
    except Exception as exc:
        manifest.update(
            status="failed_runtime_preflight",
            error=_safe_error(exc),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        raise HotpotRoundError("HotpotQA runtime preflight failed") from exc

    active = selected[:2] if canary_only else selected
    manifest["status"] = "direct_baseline"
    _write_json(paths["manifest"], manifest)
    direct = await _collect_direct(
        backend,
        active,
        config,
        paths["direct"],
        paths["failures"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    manifest["status"] = "agentgraph"
    _write_json(paths["manifest"], manifest)
    trajectories = await _collect_graph(
        backend,
        active,
        config,
        paths["trajectories"],
        paths["failures"],
        failures,
        manifest,
        paths["manifest"],
    )
    _atomic_jsonl(paths["failures"], failures)

    rows = _paired_rows(active, direct, trajectories)
    _atomic_jsonl(paths["paired"], rows)
    wrong = [row for row in rows if float(row["agentgraph"]["exact_match"]) < 1.0]
    _atomic_jsonl(paths["wrong"], wrong)
    if "error_demos" in paths:
        demos = []
        for row in wrong[:3]:
            task_id = str(row["task_id"])
            demos.append(
                {
                    "schema_version": "flowsteer.hotpotqa.error_demo.v1",
                    "task_id": task_id,
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "final_answer": row["agentgraph"]["final_answer"],
                    "exact_match": row["agentgraph"]["exact_match"],
                    "token_f1": row["agentgraph"]["token_f1"],
                    "failure_type": row["failure_type"],
                    "director_canvas_agent_tool_evaluator_trajectory": trajectories.get(task_id),
                    "direct_comparison": row["direct"],
                }
            )
        _atomic_jsonl(paths["error_demos"], demos)
    stable_zero = _stable_zero_check(
        active,
        direct,
        trajectories,
        require_dynamic_retrieval=config.get("qa_embedding_retrieval") is not None,
        retrieval_tool_id=str(
            _mapping(
                config.get("qa_embedding_retrieval", {}),
                "qa_embedding_retrieval",
            ).get("tool_id", "qa-retrieval")
        ),
    )
    manifest["stable_zero"] = stable_zero
    if not stable_zero["passed"]:
        manifest.update(status="failed_stable_zero", completed_at=_utc_now())
        _write_json(paths["manifest"], manifest)
        raise HotpotRoundError("no canary task completed the full Stable Zero chain")

    retrieval_boundary: Mapping[str, Any] | None = None
    if isinstance(retrieval_configuration, Mapping):
        tool_id = str(retrieval_configuration.get("tool_id", "qa-retrieval"))
        retrieval_boundary = _retrieval_boundary_statistics(
            trajectories,
            tool_id=tool_id,
        )
        if retrieval_configuration.get("corpus_kind") in {
            "train_qa_memory",
            "transductive_qa_memory",
            "full_dataset_fact_memory",
        }:
            _validate_memory_retrieval_execution_boundary(retrieval_boundary)
        manifest["retrieval_execution_boundary"] = dict(retrieval_boundary)
        _write_json(paths["manifest"], manifest)

    if canary_only:
        manifest.update(
            status="stable_zero_confirmed",
            canary_task_count=len(active),
            completed_at=_utc_now(),
        )
        _write_json(paths["manifest"], manifest)
        return manifest

    report = {
        **dict(_report(rows, config)),
        "tool_usage": _tool_statistics(trajectories),
        "retrieval_profile": (
            dict(config["qa_embedding_retrieval"])
            if isinstance(config.get("qa_embedding_retrieval"), Mapping)
            else None
        ),
    }
    retrieval_configuration = config.get("qa_embedding_retrieval")
    if isinstance(retrieval_configuration, Mapping):
        report = {
            **dict(report),
            "retrieval_execution_boundary": dict(retrieval_boundary or {}),
        }
    if config.get("qa_embedding_retrieval") is not None:
        retrieval_config = _mapping(
            config["qa_embedding_retrieval"], "qa_embedding_retrieval"
        )
        # Re-read all evidence after collection so the final report cannot
        # silently accept artifacts changed after the pre-run freeze gate.
        retrieval_evidence = _read_and_validate_retrieval_evidence(
            retrieval_config,
            paths,
        )
        report = {
            **dict(report),
            "retrieval_evidence": {
                name: {
                    "path": str(paths[name]),
                    "value": value,
                }
                for name, value in retrieval_evidence.items()
            },
        }
        if retrieval_config.get("corpus_kind") in {
            "train_qa_memory",
            "transductive_qa_memory",
            "full_dataset_fact_memory",
        }:
            index_manifest = _read_json(paths["retrieval_index_manifest"])
            paraphrase_manifest = _read_json(paths["paraphrase_manifest"])
            boundary = _mapping(
                report["retrieval_execution_boundary"],
                "retrieval_execution_boundary",
            )
            _validate_memory_retrieval_execution_boundary(boundary)
            report = {
                **dict(report),
                "retrieval_memory": {
                    "train_record_count": index_manifest.get("train_record_count"),
                    "source_record_count": index_manifest.get("source_record_count"),
                    "source_train_count": index_manifest.get("source_train_count"),
                    "source_evaluation_count": index_manifest.get(
                        "source_evaluation_count"
                    ),
                    "source_validation_count": index_manifest.get(
                        "source_validation_count"
                    ),
                    "unique_source_count": index_manifest.get("unique_source_count"),
                    "cycled_record_count": index_manifest.get("cycled_record_count"),
                    "paraphrase_count": index_manifest.get("paraphrase_count"),
                    "question_rewrite_count": index_manifest.get(
                        "question_rewrite_count"
                    ),
                    "fact_count": index_manifest.get("fact_count"),
                    "semantic_rewrite_coverage": index_manifest.get(
                        "semantic_rewrite_coverage"
                    ),
                    "paraphrase_versions": index_manifest.get("paraphrase_versions"),
                    "paraphrase_provenances": index_manifest.get(
                        "paraphrase_provenances"
                    ),
                    "heldout_validation_count": index_manifest.get(
                        "heldout_validation_count"
                    ),
                    "validation_overlap_count": index_manifest.get(
                        "validation_overlap_count"
                    ),
                    "frozen_validation_count": index_manifest.get(
                        "frozen_validation_count"
                    ),
                    "evaluation_overlap_count": index_manifest.get(
                        "evaluation_overlap_count"
                    ),
                    "contains_evaluation_answers": index_manifest.get(
                        "contains_evaluation_answers"
                    ),
                    "contains_evaluation_source_facts": index_manifest.get(
                        "contains_evaluation_source_facts"
                    ),
                    "contains_raw_questions": index_manifest.get(
                        "contains_raw_questions"
                    ),
                    "contains_raw_answers": index_manifest.get(
                        "contains_raw_answers"
                    ),
                    "document_format": index_manifest.get("document_format"),
                    "indexed_text_field": index_manifest.get(
                        "indexed_text_field"
                    ),
                    "evaluation_regime": index_manifest.get("evaluation_regime"),
                    "evaluation_scope": index_manifest.get("evaluation_scope"),
                    "official_heldout_eligible": index_manifest.get(
                        "official_heldout_eligible"
                    ),
                    "embedding_model": index_manifest.get("embedding_model"),
                    "embedding_dimension": index_manifest.get(
                        "embedding_dimension"
                    ),
                    "normalized": index_manifest.get("normalized"),
                    "similarity": index_manifest.get("similarity"),
                    "frozen_top_k": index_manifest.get("frozen_top_k"),
                    "materialization": paraphrase_manifest,
                },
            }
    baseline_source = _mapping(
        config["hotpotqa_evaluation"], "hotpotqa_evaluation"
    ).get("agentgraph_baseline_source_path")
    if baseline_source is not None:
        official_baseline = await _official_rescore_saved_agentgraph(
            active,
            _resolve(root, str(baseline_source)),
        )
        report = {
            **dict(report),
            "round01_agentgraph_official_rescore": official_baseline,
            "agentgraph_minus_round01_official_rescore": {
                "exact_match": (
                    report["agentgraph"]["strict_exact_match"]
                    - official_baseline["strict_exact_match"]
                ),
                "token_f1": (
                    report["agentgraph"]["strict_token_f1"]
                    - official_baseline["strict_token_f1"]
                ),
            },
        }
    _write_json(paths["report_json"], report)
    paths["report_markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_markdown"].write_text(_report_markdown(report), encoding="utf-8")
    manifest.update(
        status="completed",
        direct_progress={"completed": len(direct)},
        agentgraph_progress={"completed": len(trajectories)},
        metrics={
            "direct": report["direct_local_baseline"],
            "agentgraph": report["agentgraph"],
            "delta": report["agentgraph_minus_direct"],
        },
        tool_usage=report["tool_usage"],
        terminal_failures=report["terminal_failures"],
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
        default="config/evaluation_hotpotqa_round_01.yaml",
        help="fixed HotpotQA Round-01 YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze the 128 validation tasks without starting a model or API",
    )
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="run the first two frozen tasks to prove the Stable Zero chain",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = asyncio.run(
            run_hotpot_round(
                _resolve(PROJECT_ROOT, args.config),
                project_root=PROJECT_ROOT,
                prepare_only=bool(args.prepare_only),
                canary_only=bool(args.canary_only),
            )
        )
    except (ConfigurationError, HotpotRoundError, ValueError, RuntimeError) as exc:
        print(f"HotpotQA round failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
