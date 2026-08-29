#!/usr/bin/env python3
"""Run the bounded 7x2 AgentGraph Qwen3.5 smoke-training transaction."""

# ruff: noqa: E402 -- executable scripts add the repository root before imports.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interactive.agent_graph import AgentGraph, AgentNode, AgentRelation
from src.interactive.agent_runtime import AgentRuntime
from src.interactive.agent_workflow_env import AgentWorkflowEnv
from src.interactive.config_loader import (
    ConfigurationError,
    load_model_registry,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.director import AgentGraphOrchestrator
from src.interactive.grpo_objective import same_condition_advantages
from src.interactive.hotpotqa_embedding_index import HotpotQAEmbeddingIndex
from src.interactive.hotpotqa_qa_memory_index import HotpotQAQAMemoryIndex
from src.interactive.hotpotqa_transductive_qa_memory_index import (
    HotpotQATransductiveQAMemoryIndex,
)
from src.interactive.hotpotqa_full_dataset_qa_memory_index import (
    FULL_DATASET_EVALUATION_SCOPE,
    HotpotQAFullDatasetQAMemoryIndex,
)
from src.interactive.hotpotqa_embedding_tool import (
    HotpotQAEmbeddingReactExecutionAdapter,
    build_hotpotqa_embedding_tool_registry,
)
from src.interactive.openai_gateway import OpenAICompatibleGateway
from src.interactive.persistence import EvidenceStore
from src.interactive.policy_sync import (
    PolicySyncConfig,
    PolicySyncError,
    SGLangPolicyPublisher,
)
from src.interactive.records import TaskRecord, TrajectoryRecord
from src.interactive.rollout_collector import (
    AgentGraphRolloutCollector,
    RolloutGate,
    SGLangReceiptDirectorClient,
)
from src.interactive.scientific_sampling import (
    ScientificSamplingCoordinate,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from src.interactive.smoke_trainer import (
    Qwen35OnePassSmokeTrainer,
    SmokeTrainerConfig,
    trajectory_to_grpo,
)
from src.interactive.task_dataset import hotpotqa_question_scope, iter_task_records
from src.interactive.task_evaluator import (
    EvaluationOutcome,
    HEALTHBENCH_EVALUATOR_VERSION,
    HOTPOTQA_ANSWER_EVALUATOR_VERSION,
    RAGEN_EVALUATOR_VERSION,
    SKILLFLOW_REWARD_VERSION,
    SWEBENCH_EVALUATOR_VERSION,
    evaluate_task,
)
from src.interactive.versioning import VersionBundle


PROMPT_VERSION = "agentgraph.director.minimal.v1"
TOOL_VERSION = "agentgraph.atomic-actions.v1"

EXPECTED_SOURCE_ORDER = (
    "hotpotqa",
    "triviaqa",
    "aime_2026",
    "healthbench_professional",
    "webshop",
    "alfworld",
    "swe_bench",
)


class SmokeRunError(RuntimeError):
    """The bounded run could not prove a complete training transaction."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def qa_retrieval_scopes(
    retrieval: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve the public task and worker retrieval-query scopes.

    Older frozen retrieval conditions used one ``question_scope`` field and
    consequently supplied the question-only rendering to the whole Canvas.
    New conditions must declare the two planes independently: the Director
    and every worker keep the public task, while only a Tool-capable worker
    receives the question-only query scope used to formulate embedding
    searches.
    """

    has_explicit_split = (
        "task_scope" in retrieval or "retrieval_query_scope" in retrieval
    )
    if has_explicit_split:
        task_scope = retrieval.get("task_scope")
        query_scope = retrieval.get("retrieval_query_scope")
        if task_scope != "public_task":
            raise ConfigurationError(
                "qa_embedding_retrieval.task_scope must be public_task"
            )
        if query_scope != "question_only":
            raise ConfigurationError(
                "qa_embedding_retrieval.retrieval_query_scope must be question_only"
            )
        return "public_task", "question_only"

    if retrieval.get("question_scope") != "question_only":
        raise ConfigurationError(
            "legacy qa_embedding_retrieval.question_scope must be question_only"
        )
    return "question_only", "question_only"


def qa_retrieval_runtime_task(
    task: TaskRecord,
    retrieval: Mapping[str, Any],
) -> TaskRecord:
    """Apply only the declared Canvas task scope to one QA task."""

    task_scope, _ = qa_retrieval_scopes(retrieval)
    if task_scope == "public_task":
        return task
    return TaskRecord(
        task_id=task.task_id,
        question=hotpotqa_question_scope(task.question),
        ground_truth=task.ground_truth,
        split=task.split,
        metadata=task.metadata,
    )


def validate_smoke_bounds(config: Mapping[str, Any]) -> None:
    """Reject any config that silently expands this one-update smoke run."""

    validate_agent_graph_config(config)
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    smoke = _mapping(data.get("smoke"), "data.smoke")
    grpo = _mapping(config.get("grpo"), "grpo")
    director = _mapping(config.get("director"), "director")
    policy_sync = _mapping(config.get("policy_sync"), "policy_sync")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    exploration = _mapping(config.get("exploration"), "exploration")
    skills = _mapping(config.get("skills"), "skills")

    checks = {
        "experiment.phase": experiment.get("phase") == "smoke_training",
        "experiment.training_enabled": experiment.get("training_enabled") is True,
        "data.smoke.split": smoke.get("split") == "train",
        "data.smoke.selection": smoke.get("selection") == "sequential_per_source",
        "data.smoke.tasks_per_dataset": smoke.get("tasks_per_dataset") == 2,
        "data.smoke.expected_total_tasks": smoke.get("expected_total_tasks") == 14,
        "grpo.enabled": grpo.get("enabled") is True,
        "grpo.samples_per_problem": grpo.get("samples_per_problem") == 2,
        "grpo.expected_rollout_count": grpo.get("expected_rollout_count") == 28,
        "grpo.optimization_passes_per_rollout_batch": (
            grpo.get("optimization_passes_per_rollout_batch") == 1
        ),
        "grpo.max_optimizer_updates": grpo.get("max_optimizer_updates") == 1,
        "grpo.terminal_task_reward_only": grpo.get("terminal_task_reward_only") is True,
        "policy_sync.enabled": policy_sync.get("enabled") is True,
        "policy_sync.post_update_canary_count": (
            policy_sync.get("post_update_canary_count") == 1
        ),
        "evaluation.healthbench_judge_model": bool(
            str(evaluation.get("healthbench_judge_model", "")).strip()
        ),
        "evaluation.max_environment_steps": (
            evaluation.get("max_environment_steps") == 12
        ),
        "director.prompt_profile": director.get("prompt_profile") == "minimal",
        "director.temperature": float(director.get("temperature", -1)) == 1.0,
        "director.top_p": float(director.get("top_p", -1)) == 1.0,
        "director.top_k": director.get("top_k") == -1,
        "exploration.enabled": exploration.get("enabled") is False,
        "skills.enabled": skills.get("enabled") is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ConfigurationError(
            "smoke config violates fixed bounds: " + ", ".join(failed)
        )

    source_order = tuple(str(value) for value in smoke.get("source_order", ()))
    if source_order != EXPECTED_SOURCE_ORDER:
        raise ConfigurationError(
            "data.smoke.source_order must contain the fixed seven-source order"
        )
    for field_name in (
        "behavior_policy_version",
        "updated_policy_version",
        "expected_server_weight_version",
    ):
        value = director.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"director.{field_name} must be non-empty")
    if director["behavior_policy_version"] == director["updated_policy_version"]:
        raise ConfigurationError("Director behavior and updated policy versions must differ")
    oom = _mapping(_mapping(config["gpu"], "gpu")["oom_policy"], "gpu.oom_policy")
    if tuple(oom.get("micro_batch_schedule", ())) != (4, 2, 1):
        raise ConfigurationError("gpu.oom_policy.micro_batch_schedule must be [4, 2, 1]")


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _dataset_key(task: TaskRecord) -> str:
    value = task.metadata.get("dataset_key")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task {task.task_id!r} has no metadata.dataset_key")
    return value.strip()


def _base_task_id(task: TaskRecord) -> str:
    sampling = task.metadata.get("sampling", {})
    if isinstance(sampling, Mapping):
        value = sampling.get("base_task_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return task.task_id


def select_smoke_tasks(
    tasks: Sequence[TaskRecord] | Any,
    *,
    source_order: Sequence[str],
    per_source: int,
    require_unique_base_tasks: bool,
    skip_per_source: int = 0,
    expected_split: str = "train",
) -> tuple[TaskRecord, ...]:
    """Select the first N records per dataset key in config order."""

    if type(per_source) is not int or per_source <= 0:
        raise ValueError("per_source must be a positive integer")
    if type(skip_per_source) is not int or skip_per_source < 0:
        raise ValueError("skip_per_source must be a non-negative integer")
    if not isinstance(expected_split, str) or not expected_split.strip():
        raise ValueError("expected_split must be a non-empty string")
    ordered_sources = tuple(str(value).strip() for value in source_order)
    if not ordered_sources or any(not value for value in ordered_sources):
        raise ValueError("source_order must contain non-empty dataset keys")
    if len(ordered_sources) != len(set(ordered_sources)):
        raise ValueError("source_order contains duplicate dataset keys")

    selected: dict[str, list[TaskRecord]] = {source: [] for source in ordered_sources}
    base_ids: dict[str, set[str]] = {source: set() for source in ordered_sources}
    skipped: dict[str, int] = {source: 0 for source in ordered_sources}
    for task in tasks:
        if task.split != expected_split:
            raise ValueError(
                f"bounded task {task.task_id!r} is not in the {expected_split} split"
            )
        source = _dataset_key(task)
        if source not in selected or len(selected[source]) >= per_source:
            continue
        if skipped[source] < skip_per_source:
            skipped[source] += 1
            continue
        base_id = _base_task_id(task)
        if require_unique_base_tasks and base_id in base_ids[source]:
            continue
        selected[source].append(task)
        base_ids[source].add(base_id)
        if all(len(items) == per_source for items in selected.values()):
            break

    missing = {
        source: per_source - len(items)
        for source, items in selected.items()
        if len(items) != per_source
    }
    if missing:
        detail = ", ".join(f"{source}: {count}" for source, count in missing.items())
        raise ValueError(f"insufficient smoke tasks by source ({detail})")
    return tuple(task for source in ordered_sources for task in selected[source])


def evaluator_version_for(task: TaskRecord) -> str:
    source = _dataset_key(task)
    if source == "hotpotqa":
        return HOTPOTQA_ANSWER_EVALUATOR_VERSION
    if source in {"triviaqa", "aime_2026"}:
        return SKILLFLOW_REWARD_VERSION
    if source == "healthbench_professional":
        return HEALTHBENCH_EVALUATOR_VERSION
    if source in {"webshop", "alfworld"}:
        return RAGEN_EVALUATOR_VERSION
    if source == "swe_bench":
        return SWEBENCH_EVALUATOR_VERSION
    raise ValueError(f"unsupported smoke dataset key: {source}")


def version_bundle_for(
    task: TaskRecord,
    *,
    policy_version: str,
    model_catalog_version: str,
    prompt_version: str = PROMPT_VERSION,
    tool_version: str = TOOL_VERSION,
) -> VersionBundle:
    return VersionBundle(
        policy=policy_version,
        model_catalog=model_catalog_version,
        evaluator=evaluator_version_for(task),
        prompt=prompt_version,
        tool=tool_version,
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("artifact values must be mappings, dataclasses, or expose to_dict()")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value) if not isinstance(value, Mapping) else dict(value),
                   ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _safe_error(error: BaseException) -> str:
    message = str(error)
    secret = os.environ.get("VECTOR_ENGINE_API_KEY", "")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return f"{type(error).__name__}: {message}"


def _graph_from_mapping(value: Mapping[str, Any]) -> AgentGraph:
    raw_nodes = value.get("nodes", ())
    raw_relations = value.get("relations", ())
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        raise ValueError("final graph nodes are malformed")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        raise ValueError("final graph relations are malformed")
    nodes = []
    for item in raw_nodes:
        if not isinstance(item, Mapping):
            raise ValueError("final graph node is malformed")
        nodes.append(
            AgentNode(
                str(item.get("id", "")),
                str(item.get("model_id", "")),
                str(item.get("contract", item.get("prompt", ""))),
            )
        )
    relations = []
    for item in raw_relations:
        if not isinstance(item, Mapping):
            raise ValueError("final graph relation is malformed")
        relations.append(
            AgentRelation(
                str(item.get("source_id", "")),
                str(item.get("target_id", "")),
                item.get("source_to_target"),
                item.get("target_to_source"),
            )
        )
    revision = value.get("revision", 0)
    if type(revision) is not int:
        raise ValueError("final graph revision is malformed")
    output = value.get("output_agent_id")
    if output is not None and not isinstance(output, str):
        raise ValueError("final graph output_agent_id is malformed")
    return AgentGraph(nodes, relations, output_agent_id=output, revision=revision)


class SmokeBackend(Protocol):
    model_catalog_version: str

    async def collect(
        self,
        task: TaskRecord,
        rollout_index: int,
        versions: VersionBundle,
        *,
        expected_task_split: str = "train",
    ) -> TrajectoryRecord:
        ...

    def train(
        self,
        trajectories: Sequence[TrajectoryRecord],
        output_dir: Path,
    ) -> Any:
        ...

    async def publish(self, summary: Any) -> Any:
        ...


JudgeCallback = Callable[[Sequence[Mapping[str, str]], str], Awaitable[Any]]


class LiveSmokeBackend:
    """Thin wiring layer over the existing collector, trainer, and publisher."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        registry: Any,
        runtime: AgentRuntime,
        director_client: SGLangReceiptDirectorClient,
        rollout_gate: RolloutGate,
        evidence_store: EvidenceStore,
        trainer: Optional[Qwen35OnePassSmokeTrainer],
        publisher: SGLangPolicyPublisher,
        judge: Optional[JudgeCallback],
        judge_model: str,
        project_root: Path,
        hotpotqa_embedding_index: Optional[
            HotpotQAEmbeddingIndex
            | HotpotQAQAMemoryIndex
            | HotpotQATransductiveQAMemoryIndex
            | HotpotQAFullDatasetQAMemoryIndex
        ] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.runtime = runtime
        self.director_client = director_client
        self.rollout_gate = rollout_gate
        self.evidence_store = evidence_store
        self.trainer = trainer
        self.publisher = publisher
        self.judge = judge
        self.judge_model = judge_model
        self.project_root = project_root
        self.hotpotqa_embedding_index = hotpotqa_embedding_index

    @property
    def model_catalog_version(self) -> str:
        return self.registry.catalog_id

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        root: Path,
        *,
        evaluation_only: bool = False,
    ) -> "LiveSmokeBackend":
        secret = os.environ.get("VECTOR_ENGINE_API_KEY", "")
        if not secret:
            raise ConfigurationError(
                "missing required environment variable: VECTOR_ENGINE_API_KEY"
            )

        director = _mapping(config["director"], "director")
        experiment = _mapping(config["experiment"], "experiment")
        graph_config = _mapping(config["agent_graph"], "agent_graph")
        grpo = _mapping(config["grpo"], "grpo")
        gpu = _mapping(config["gpu"], "gpu")
        oom = _mapping(gpu["oom_policy"], "gpu.oom_policy")
        storage = _mapping(config["storage"], "storage")
        sync = _mapping(config["policy_sync"], "policy_sync")
        evaluation = _mapping(config["evaluation"], "evaluation")

        catalog_path = _resolve(root, str(graph_config["model_catalog_path"]))
        if not catalog_path.is_file():
            raise ConfigurationError(
                f"model catalog does not exist: {catalog_path}; copy the example first"
            )
        registry = load_model_registry(catalog_path)

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - heavy runtime only
            raise RuntimeError("transformers is required for Qwen3.5 smoke rollout") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(director["tokenizer_path"]),
            trust_remote_code=True,
        )

        gate = RolloutGate()
        behavior_adapter = director.get("behavior_adapter_name")
        if behavior_adapter is not None:
            behavior_adapter = str(behavior_adapter).strip() or None
        director_client = SGLangReceiptDirectorClient(
            tokenizer,
            base_url=str(director["api_base"]),
            api_key=os.environ.get("SGLANG_API_KEY", "EMPTY"),
            policy_version=str(director["behavior_policy_version"]),
            adapter_name=behavior_adapter,
            expected_server_weight_version=str(
                director["expected_server_weight_version"]
            ),
            rollout_gate=gate,
            temperature=float(director["temperature"]),
            top_p=float(director["top_p"]),
            top_k=int(director["top_k"]),
            max_tokens=int(director["max_action_tokens"]),
        )

        gateway = OpenAICompatibleGateway(default_seed=int(experiment["seed"]))
        runtime = AgentRuntime(registry, gateway)
        hotpotqa_embedding_index: Optional[
            HotpotQAEmbeddingIndex
            | HotpotQAQAMemoryIndex
            | HotpotQATransductiveQAMemoryIndex
            | HotpotQAFullDatasetQAMemoryIndex
        ] = None
        raw_embedding_retrieval = config.get("qa_embedding_retrieval")
        if raw_embedding_retrieval is not None:
            retrieval = _mapping(
                raw_embedding_retrieval,
                "qa_embedding_retrieval",
            )
            qa_retrieval_scopes(retrieval)
            index_dir = _resolve(root, str(retrieval["index_dir"]))
            corpus_kind = str(retrieval.get("corpus_kind", "public_context"))
            if corpus_kind == "train_qa_memory":
                hotpotqa_embedding_index = HotpotQAQAMemoryIndex.open(
                    index_dir,
                    embedding_model_path=str(retrieval["embedding_model"]),
                    embedding_device=str(retrieval["embedding_device"]),
                )
            elif corpus_kind == "transductive_qa_memory":
                hotpotqa_embedding_index = HotpotQATransductiveQAMemoryIndex.open(
                    index_dir,
                    embedding_model_path=str(retrieval["embedding_model"]),
                    embedding_device=str(retrieval["embedding_device"]),
                )
            elif corpus_kind == "full_dataset_qa_memory":
                hotpotqa_embedding_index = HotpotQAFullDatasetQAMemoryIndex.open(
                    index_dir,
                    embedding_model_path=str(retrieval["embedding_model"]),
                    embedding_device=str(retrieval["embedding_device"]),
                )
            elif corpus_kind == "public_context":
                hotpotqa_embedding_index = HotpotQAEmbeddingIndex.open(
                    index_dir,
                    embedding_model_path=str(retrieval["embedding_model"]),
                    embedding_device=str(retrieval["embedding_device"]),
                )
            else:
                raise ConfigurationError(
                    "qa_embedding_retrieval.corpus_kind must be public_context "
                    "or train_qa_memory or transductive_qa_memory "
                    "or full_dataset_qa_memory"
                )
            manifest = hotpotqa_embedding_index.manifest
            if (
                manifest.embedding_model != str(retrieval["embedding_model_id"])
                or manifest.frozen_top_k != int(retrieval["search_top_k"])
                or manifest.normalized is not True
                or manifest.similarity != str(retrieval["similarity"])
            ):
                raise ConfigurationError(
                    "HotpotQA embedding index manifest differs from the frozen config"
                )
            if corpus_kind == "train_qa_memory" and (
                manifest.train_record_count != int(retrieval["train_sample_count"])
                or manifest.heldout_validation_count
                != int(retrieval["validation_sample_count"])
                or manifest.validation_overlap_count != 0
            ):
                raise ConfigurationError(
                    "HotpotQA QA-memory manifest violates frozen split isolation"
                )
            if corpus_kind == "transductive_qa_memory" and (
                manifest.source_record_count
                != int(retrieval["source_record_count"])
                or manifest.source_train_count
                != int(retrieval["train_sample_count"])
                or manifest.source_evaluation_count
                != int(retrieval["validation_sample_count"])
                or manifest.frozen_validation_count
                != int(retrieval["validation_sample_count"])
                or manifest.evaluation_overlap_count
                != int(retrieval["evaluation_overlap_count"])
                or manifest.contains_evaluation_answers is not True
                or manifest.evaluation_regime != "transductive_retrieval"
                or manifest.official_heldout_eligible is not False
            ):
                raise ConfigurationError(
                    "HotpotQA transductive QA-memory manifest differs from config"
                )
            if corpus_kind == "full_dataset_qa_memory" and (
                manifest.source_record_count
                != int(retrieval["source_record_count"])
                or manifest.source_train_count
                != int(retrieval["source_train_count"])
                or manifest.source_validation_count
                != int(retrieval["source_validation_count"])
                or manifest.evaluation_overlap_count
                != int(retrieval["evaluation_overlap_count"])
                or manifest.contains_evaluation_answers is not True
                or manifest.evaluation_scope != FULL_DATASET_EVALUATION_SCOPE
                or manifest.official_heldout_eligible is not False
            ):
                raise ConfigurationError(
                    "HotpotQA full-dataset QA-memory manifest differs from config"
                )
        evidence_store = EvidenceStore(_resolve(root, str(storage["root"])))

        trainer: Optional[Qwen35OnePassSmokeTrainer] = None
        if not evaluation_only:
            lora = _mapping(director["lora"], "director.lora")
            trainer = Qwen35OnePassSmokeTrainer(
                SmokeTrainerConfig(
                    model_path=str(director["base_model"]),
                    tokenizer_path=str(director["tokenizer_path"]),
                    behavior_policy_version=str(director["behavior_policy_version"]),
                    updated_policy_version=str(director["updated_policy_version"]),
                    behavior_policy_adapter=behavior_adapter,
                    behavior_server_weight_version=str(
                        director["expected_server_weight_version"]
                    ),
                    behavior_adapter_checkpoint=(
                        str(_resolve(root, str(director["behavior_adapter_checkpoint"])))
                        if director.get("behavior_adapter_checkpoint")
                        else None
                    ),
                    update_step=int(experiment.get("update_step", 1)),
                    optimizer_state_checkpoint=(
                        str(_resolve(root, str(director["optimizer_state_checkpoint"])))
                        if director.get("optimizer_state_checkpoint")
                        else None
                    ),
                    learner_device=str(gpu["learner_device"]),
                    gradient_replica_device=str(gpu["gradient_replica_device"]),
                    lora_rank=int(lora["rank"]),
                    lora_alpha=int(lora["alpha"]),
                    lora_dropout=float(lora["dropout"]),
                    lora_target_modules=tuple(
                        str(value) for value in lora["target_modules"]
                    ),
                    learning_rate=float(grpo["learning_rate"]),
                    max_grad_norm=float(grpo["max_grad_norm"]),
                    advantage_epsilon=float(grpo["advantage_epsilon"]),
                    gradient_checkpointing=bool(grpo["gradient_checkpointing"]),
                    micro_batch_backoff=tuple(
                        int(value) for value in oom["micro_batch_schedule"]
                    ),
                )
            )
        publisher = SGLangPolicyPublisher(
            PolicySyncConfig(
                api_base=str(sync["api_base"]),
                api_key=os.environ.get("SGLANG_API_KEY", "EMPTY"),
                adapter_name_prefix=str(sync["adapter_name_prefix"]),
                request_timeout_seconds=float(sync["request_timeout_seconds"]),
                max_retries=int(sync["max_retries"]),
                retry_backoff_seconds=float(sync["retry_backoff_seconds"]),
            )
        )
        judge: Optional[JudgeCallback] = None
        judge_model = ""
        if not evaluation_only:
            judge, judge_model = cls._build_healthbench_judge(
                registry,
                secret,
                str(evaluation["healthbench_judge_model"]),
            )
        return cls(
            config=config,
            registry=registry,
            runtime=runtime,
            director_client=director_client,
            rollout_gate=gate,
            evidence_store=evidence_store,
            trainer=trainer,
            publisher=publisher,
            judge=judge,
            judge_model=judge_model,
            project_root=root,
            hotpotqa_embedding_index=hotpotqa_embedding_index,
        )

    @staticmethod
    def _build_healthbench_judge(
        registry: Any,
        secret: str,
        configured_model_id: str,
    ) -> tuple[JudgeCallback, str]:
        if not configured_model_id.strip():
            raise ConfigurationError("evaluation.healthbench_judge_model is empty")
        model = registry.require_model(configured_model_id)
        provider = registry.provider_for(model.model_id)
        if provider.api_key_env != "VECTOR_ENGINE_API_KEY":
            raise ConfigurationError(
                "configured HealthBench judge must use the VectorEngine provider"
            )
        if not provider.endpoint:
            raise ConfigurationError("HealthBench judge provider has no endpoint")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("openai is required for the HealthBench judge") from exc
        client = AsyncOpenAI(api_key=secret, base_url=provider.endpoint)

        async def judge(messages: Sequence[Mapping[str, str]], model_name: str) -> Any:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[dict(message) for message in messages],
                max_tokens=2048,
                temperature=0,
            )
            if not response.choices or response.choices[0].message.content is None:
                return ""
            return response.choices[0].message.content

        return judge, model.model_name

    async def collect(
        self,
        task: TaskRecord,
        rollout_index: int,
        versions: VersionBundle,
        *,
        expected_task_split: str = "train",
    ) -> TrajectoryRecord:
        director = _mapping(self.config["director"], "director")
        graph_config = _mapping(self.config["agent_graph"], "agent_graph")
        experiment = _mapping(self.config["experiment"], "experiment")
        base_seed = int(experiment["seed"])
        condition_id = str(
            experiment.get("condition_id", "natural_smoke")
        ).strip()
        if not condition_id:
            raise ConfigurationError("experiment.condition_id must be non-empty")
        sampling_schedule_purpose = str(
            experiment.get("sampling_schedule_purpose", condition_id)
        ).strip()
        if not sampling_schedule_purpose:
            raise ConfigurationError(
                "experiment.sampling_schedule_purpose must be non-empty"
            )
        sampling_coordinate = ScientificSamplingCoordinate(
            sampling_schedule_hash=scientific_sampling_schedule_hash(
                base_seed=base_seed
            ),
            schedule_purpose=sampling_schedule_purpose,
            ordered_sequence_hash=stable_hash([task.task_id]),
            sequence_position=rollout_index,
            task_id=task.task_id,
            optimizer_step_or_anchor_ordinal=0,
        )
        runtime_task = task
        task_runtime = self.runtime
        task_tool_registry = None
        raw_embedding_retrieval = self.config.get("qa_embedding_retrieval")
        if raw_embedding_retrieval is not None:
            retrieval = _mapping(
                raw_embedding_retrieval,
                "qa_embedding_retrieval",
            )
            if _dataset_key(task) != "hotpotqa":
                raise ConfigurationError(
                    "HotpotQA embedding retrieval cannot serve another dataset"
                )
            if self.hotpotqa_embedding_index is None:
                raise ConfigurationError("HotpotQA embedding index was not opened")
            task_tool_registry = build_hotpotqa_embedding_tool_registry(
                self.hotpotqa_embedding_index,
                task_id=task.task_id,
                tool_id=str(retrieval.get("tool_id", "qa-retrieval")),
                frozen_top_k=int(retrieval["search_top_k"]),
                timeout_seconds=float(retrieval["tool_timeout_seconds"]),
            )
            react_adapter = HotpotQAEmbeddingReactExecutionAdapter(
                gateway=self.runtime.gateway,
                tool_registry=task_tool_registry,
                retrieval_query_scope=hotpotqa_question_scope(task.question),
                max_turns=int(retrieval["max_turns_per_agent_call"]),
                max_tool_calls=int(retrieval["max_tool_calls_per_agent_call"]),
                max_action_tokens=int(
                    retrieval.get(
                        "max_action_tokens",
                        director["max_action_tokens"],
                    )
                ),
                sampling_base_seed=base_seed,
                sampling_coordinate=sampling_coordinate,
            )
            task_runtime = AgentRuntime(
                self.registry,
                self.runtime.gateway,
                execution_adapters={"react": react_adapter},
                tool_registry=task_tool_registry,
                dataset_id="hotpotqa",
            )
            runtime_task = qa_retrieval_runtime_task(task, retrieval)
        orchestrator = AgentGraphOrchestrator(
            self.registry,
            self.director_client,
            max_rounds=int(director["max_rounds"]),
            seed=int(experiment["seed"]) + rollout_index,
            history_window=int(director["history_window"]),
            tool_registry=task_tool_registry,
            sampling_action_profile=(
                str(director["sampling_action_profile"]).strip()
                if director.get("sampling_action_profile") is not None
                else None
            ),
            sampling_action_schema_version=str(
                director.get(
                    "sampling_action_schema_version",
                    "agentgraph.model-admissible-action-mask.v2",
                )
            ).strip(),
        )
        environment = AgentWorkflowEnv(
            self.registry,
            runtime=task_runtime,
            execute_on_edit=bool(director["execute_on_edit"]),
            max_agents=int(graph_config["max_agents"]),
            max_agents_per_subgraph=int(
                graph_config.get("max_agents_per_subgraph", 3)
            ),
            require_exact_answer_tag=(
                str(graph_config.get("terminal_protocol", "none"))
                == "exact_single_answer_tag"
            ),
            require_format_agent=bool(
                graph_config.get("require_format_agent", False)
            ),
            allowed_actions=tuple(
                str(action) for action in graph_config["actions"]
            ),
            recovery_policy=str(
                graph_config.get("recovery_policy", "default")
            ),
            director_feedback_mode=str(
                graph_config.get("director_feedback_mode", "content")
            ),
            required_evidence_tool_id=(
                str(graph_config["required_evidence_tool_id"])
                if task_tool_registry is not None
                else None
            ),
            require_evidence_relation=(
                bool(graph_config.get("require_evidence_relation", False))
                if task_tool_registry is not None
                else False
            ),
        )
        collector = AgentGraphRolloutCollector(
            orchestrator,
            environment,
            versions,
            self.evidence_store,
            condition_id=str(experiment.get("condition_id", "natural_smoke")),
            skills=(),
            forced_probe=False,
            expected_task_split=expected_task_split,
        )

        async def evaluator_callback(
            evaluated_task: TaskRecord,
            final_answer: Optional[str],
            final_graph: Mapping[str, Any],
            final_runtime: Any,
        ) -> Any:
            del final_runtime
            source_key = _dataset_key(evaluated_task)
            if source_key in {"webshop", "alfworld"} and final_answer is None:
                # A natural Director budget exhaustion is already a real
                # terminal failure in the MD/SkillFlow boundary.  Do not start
                # a fresh interactive environment after the workflow itself
                # failed to finish.
                return EvaluationOutcome(
                    valid=True,
                    reward=0.0,
                    metrics={"success": 0.0},
                    reason="director_max_rounds_without_explicit_finish",
                    evaluator_version=RAGEN_EVALUATOR_VERSION,
                )
            environment_graph = _graph_from_mapping(final_graph)
            environment_step = 0

            async def run_graph(observation: str) -> str:
                nonlocal environment_step
                environment_step += 1
                result = await task_runtime.execute(
                    environment_graph,
                    observation,
                    run_id=(
                        f"environment:{evaluated_task.task_id}:"
                        f"{rollout_index:04d}:{environment_step:04d}"
                    ),
                )
                return result.final_answer

            configured_steps = _mapping(
                self.config["evaluation"], "evaluation"
            ).get("max_environment_steps_by_source", {})
            if not isinstance(configured_steps, Mapping):
                configured_steps = {}
            return await evaluate_task(
                task,
                final_answer or "",
                judge=self.judge,
                judge_model=self.judge_model,
                run_graph=run_graph,
                max_environment_steps=int(
                    configured_steps.get(
                        source_key,
                        _mapping(self.config["evaluation"], "evaluation")[
                            "max_environment_steps"
                        ],
                    )
                ),
            )

        return await collector.collect(runtime_task, rollout_index, evaluator_callback)

    def train(
        self,
        trajectories: Sequence[TrajectoryRecord],
        output_dir: Path,
    ) -> Any:
        if self.trainer is None:
            raise RuntimeError("training is disabled for this evaluation-only backend")
        return self.trainer.train(trajectories, output_dir)

    async def publish(self, summary: Any) -> Any:
        director = _mapping(self.config["director"], "director")
        experiment = _mapping(self.config["experiment"], "experiment")
        checkpoint_version = f"checkpoint:{summary.updated_policy_version}"
        try:
            receipt = await asyncio.to_thread(
                self.publisher.publish,
                checkpoint_path=summary.checkpoint_dir,
                checkpoint_version=checkpoint_version,
                behavior_policy_version=summary.behavior_policy_version,
                candidate_policy_version=summary.updated_policy_version,
                step=int(experiment.get("update_step", 1)),
                previous_adapter=(
                    str(director["behavior_adapter_name"])
                    if director.get("behavior_adapter_name")
                    else None
                ),
                gate=self.rollout_gate,
            )
        except PolicySyncError:
            raise

        # The publisher has proven and committed the adapter. Bind subsequent
        # native /generate calls to both its registered name and logical policy
        # version under a second pause/drain boundary before any canary starts.
        self.rollout_gate.pause()
        try:
            self.rollout_gate.drain()
            self.director_client.update_policy_route(
                policy_version=str(receipt.new_policy_version),
                adapter_name=receipt.adapter_name,
                expected_server_weight_version=str(
                    director["expected_server_weight_version"]
                ),
            )
        finally:
            self.rollout_gate.resume()
        return receipt


def _summary_dict(summary: Any) -> dict[str, Any]:
    value = _json_value(summary)
    return dict(value)


def _write_grpo_groups(path: Path, trajectories: Sequence[TrajectoryRecord]) -> None:
    grouped: dict[tuple[str, str, str], list[tuple[TrajectoryRecord, Any]]] = defaultdict(
        list
    )
    for record in trajectories:
        item = trajectory_to_grpo(record)
        grouped[item.group_key].append((record, item))
    rows = []
    for key, entries in sorted(grouped.items()):
        eligible = [
            item for record, item in entries if record.grpo_eligible and item.eligible
        ]
        eligible_advantages = same_condition_advantages(eligible)
        advantages_by_id = {
            item.trajectory_id: float(value)
            for item, value in zip(eligible, eligible_advantages, strict=True)
        }
        rows.append(
            {
                "group_key": list(key),
                "trajectory_ids": [item.trajectory_id for _, item in entries],
                "rewards": [item.terminal_reward for _, item in entries],
                "eligible": [
                    record.grpo_eligible and item.eligible for record, item in entries
                ],
                "advantages": [
                    advantages_by_id.get(item.trajectory_id) for _, item in entries
                ],
                "informative": bool(
                    len(eligible) >= 2
                    and any(float(value) != 0.0 for value in eligible_advantages)
                ),
            }
        )
    _write_jsonl(path, rows)


def _artifact_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    storage = _mapping(config["storage"], "storage")
    return {
        "selected": _resolve(root, str(storage["selected_tasks_path"])),
        "trajectories": _resolve(root, str(storage["trajectories_path"])),
        "groups": _resolve(root, str(storage["grpo_groups_path"])),
        "manifest": _resolve(root, str(storage["manifest_path"])),
        "sync": _resolve(root, str(storage["sync_receipt_path"])),
        "post_update": _resolve(root, str(storage["post_update_trajectories_path"])),
        "training_root": _resolve(
            root, str(_mapping(config["experiment"], "experiment")["output_dir"])
        ),
    }


async def run_smoke(
    config_path: str | Path,
    *,
    prepare_only: bool = False,
    backend: Optional[SmokeBackend] = None,
    project_root: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """Execute the exact bounded pipeline and return its persisted manifest."""

    resolved_config = Path(config_path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else resolved_config.parent.parent
    )
    config = load_yaml(resolved_config)
    validate_smoke_bounds(config)
    paths = _artifact_paths(config, root)

    smoke = _mapping(_mapping(config["data"], "data")["smoke"], "data.smoke")
    train_path = _resolve(root, str(_mapping(config["data"], "data")["train_path"]))
    selected = select_smoke_tasks(
        iter_task_records(train_path, expected_split="train"),
        source_order=tuple(str(value) for value in smoke["source_order"]),
        per_source=int(smoke["tasks_per_dataset"]),
        require_unique_base_tasks=bool(smoke["require_unique_base_tasks"]),
    )
    if len(selected) != int(smoke["expected_total_tasks"]):
        raise SmokeRunError("selected task count differs from the fixed smoke bound")
    _write_jsonl(
        paths["selected"],
        [
            {"schema_version": "flowsteer.agentgraph.task.v1", **task.to_dict()}
            for task in selected
        ],
    )

    source_counts = Counter(_dataset_key(task) for task in selected)
    manifest: dict[str, Any] = {
        "schema_version": "flowsteer.agentgraph.smoke_manifest.v1",
        "status": "prepared" if prepare_only else "collecting",
        "config_path": str(resolved_config),
        "train_path": str(train_path),
        "started_at": _utc_now(),
        "bounds": {
            "tasks_per_dataset": 2,
            "selected_tasks": 14,
            "rollouts_per_task": 2,
            "expected_initial_rollouts": 28,
            "max_optimizer_updates": 1,
            "post_update_canaries": 1,
        },
        "selected_by_source": dict(sorted(source_counts.items())),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "training_root"},
        "exploration_enabled": False,
        "skills_enabled": False,
    }
    if prepare_only:
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        return manifest

    _write_json(paths["manifest"], manifest)
    try:
        live_backend = backend or LiveSmokeBackend.from_config(config, root)
    except Exception as exc:
        manifest["status"] = "failed_runtime_setup"
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("smoke runtime setup failed") from exc
    director = _mapping(config["director"], "director")
    grpo = _mapping(config["grpo"], "grpo")
    behavior_policy = str(director["behavior_policy_version"])
    updated_policy = str(director["updated_policy_version"])
    rollout_count = int(grpo["samples_per_problem"])

    initial_jobs = []
    for task in selected:
        versions = version_bundle_for(
            task,
            policy_version=behavior_policy,
            model_catalog_version=live_backend.model_catalog_version,
        )
        for rollout_index in range(rollout_count):
            initial_jobs.append(live_backend.collect(task, rollout_index, versions))
    try:
        initial = tuple(await asyncio.gather(*initial_jobs))
    except Exception as exc:
        manifest["status"] = "failed_initial_rollout"
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("initial rollout collection failed") from exc
    if len(initial) != int(grpo["expected_rollout_count"]):
        raise SmokeRunError("initial rollout count differs from the fixed smoke bound")
    if any(record.versions.policy != behavior_policy for record in initial):
        raise SmokeRunError("initial trajectories contain a non-behavior policy version")
    _write_jsonl(paths["trajectories"], initial)
    _write_grpo_groups(paths["groups"], initial)

    try:
        summary = await asyncio.to_thread(
            live_backend.train, initial, paths["training_root"]
        )
    except Exception as exc:
        manifest["status"] = "failed_training"
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("one-pass smoke training failed") from exc
    summary_value = _summary_dict(summary)
    manifest["initial_rollouts"] = {
        "collected": len(initial),
        "valid_evaluators": sum(record.evaluation.valid for record in initial),
        "grpo_eligible": sum(record.grpo_eligible for record in initial),
        "evaluator_versions": dict(
            sorted(Counter(record.evaluation.evaluator_version for record in initial).items())
        ),
    }
    manifest["training"] = summary_value
    if int(summary_value.get("optimizer_updates", 0)) != 1:
        sync_value = {
            "status": "not_attempted_no_optimizer_update",
            "success": False,
            "behavior_policy_version": behavior_policy,
            "candidate_policy_version": updated_policy,
        }
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_no_optimizer_update"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("smoke trainer completed zero optimizer updates")

    try:
        sync_receipt = await live_backend.publish(summary)
    except PolicySyncError as exc:
        sync_value = exc.receipt.to_dict()
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_policy_sync"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("SGLang policy publication failed") from exc
    except Exception as exc:
        sync_value = {
            "status": "failed",
            "success": False,
            "behavior_policy_version": behavior_policy,
            "candidate_policy_version": updated_policy,
            "error": _safe_error(exc),
        }
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_policy_sync"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("SGLang policy publication failed") from exc
    sync_value = _summary_dict(sync_receipt)
    if sync_value.get("success") is not True:
        _write_json(paths["sync"], sync_value)
        manifest["status"] = "failed_policy_sync"
        manifest["policy_sync"] = sync_value
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("policy publisher returned an unsuccessful receipt")
    adapter_name = sync_value.get("adapter_name")
    new_policy = sync_value.get("new_policy_version")
    if not isinstance(adapter_name, str) or not adapter_name.strip():
        raise SmokeRunError("policy sync receipt has no adapter_name")
    if new_policy != updated_policy:
        raise SmokeRunError("policy sync receipt has the wrong updated policy version")
    _write_json(paths["sync"], sync_value)

    canary_count = int(
        _mapping(config["policy_sync"], "policy_sync")["post_update_canary_count"]
    )
    canary_jobs = []
    for index in range(canary_count):
        task = selected[index % len(selected)]
        versions = version_bundle_for(
            task,
            policy_version=updated_policy,
            model_catalog_version=live_backend.model_catalog_version,
        )
        canary_jobs.append(live_backend.collect(task, 10_000 + index, versions))
    try:
        canaries = tuple(await asyncio.gather(*canary_jobs))
    except Exception as exc:
        manifest["status"] = "failed_post_update_canary"
        manifest["policy_sync"] = sync_value
        manifest["error"] = _safe_error(exc)
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("post-update canary collection failed") from exc
    canary_valid = len(canaries) == canary_count and all(
        record.versions.policy == updated_policy
        and bool(record.turns)
        and all(turn.policy_version == updated_policy for turn in record.turns)
        and all(turn.policy_adapter == adapter_name for turn in record.turns)
        for record in canaries
    )
    if not canary_valid:
        manifest["status"] = "failed_post_update_canary"
        manifest["policy_sync"] = sync_value
        manifest["error"] = "canary policy or adapter receipt mismatch"
        manifest["completed_at"] = _utc_now()
        _write_json(paths["manifest"], manifest)
        raise SmokeRunError("post-update canary did not use the published adapter")
    _write_jsonl(paths["post_update"], canaries)

    manifest.update(
        status="completed",
        policy_sync=sync_value,
        post_update_canaries={
            "collected": len(canaries),
            "adapter_name": adapter_name,
            "policy_version": updated_policy,
            "trajectory_ids": [record.trajectory_id for record in canaries],
        },
        completed_at=_utc_now(),
    )
    _write_json(paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/training_agentgraph_smoke.yaml",
        help="bounded smoke-training YAML",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="select and persist the 14 tasks without model, API, or GPU work",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve(PROJECT_ROOT, args.config)
    try:
        manifest = asyncio.run(
            run_smoke(
                config_path,
                prepare_only=bool(args.prepare_only),
                project_root=PROJECT_ROOT,
            )
        )
    except (ConfigurationError, SmokeRunError, ValueError, RuntimeError) as exc:
        print(f"smoke run failed: {_safe_error(exc)}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_tasks": manifest["bounds"]["selected_tasks"],
                "expected_initial_rollouts": manifest["bounds"][
                    "expected_initial_rollouts"
                ],
                "max_optimizer_updates": manifest["bounds"]["max_optimizer_updates"],
                "manifest": manifest["artifacts"]["manifest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
