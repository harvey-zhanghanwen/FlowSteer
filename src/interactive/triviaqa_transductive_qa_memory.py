"""TriviaQA all-QA transductive memory over the existing QA-memory index.

This module is deliberately separate from :mod:`triviaqa_qa_memory`.  The
existing module remains the official train-only/held-out implementation.  The
adapter here materializes the current experiment corpus of 512 train QA pairs
plus the 128 frozen development-validation QA pairs, and labels the resulting
index as transductive retrieval rather than held-out evaluation.

The stored QA rows reuse :class:`TriviaQAQAMemoryRecord`; dense encoding reuses
the existing normalized BGE implementation; and the index subclasses
:class:`TriviaQAQAMemoryIndex`, so ``search`` and ``read`` retain their current
Tool wire format.  A source-membership sidecar records the true source
partition because the reused record schema has a legacy ``source_split=train``
field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .triviaqa_embedding_index import (
    BGE_QUERY_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingEncoder,
    _canonical_json,
    _file_sha256,
    _get_embedding_model,
    _normalized_embeddings,
    _positive_integer,
    _required_sha256,
    _required_text,
    _stable_sha256,
    _write_atomic_bytes,
)
from .triviaqa_qa_memory import (
    EMBEDDING_TEXT_TEMPLATE,
    EMBEDDINGS_FILENAME,
    MEMORIES_FILENAME,
    QA_MEMORY_TOOL_ID,
    RETRIEVAL_BACKEND,
    SOURCE_DATASET,
    TriviaQAQAMemoryIndex,
    TriviaQAQAMemoryRecord,
    TriviaQATrainSource,
    load_materialized_qa_memory,
    load_triviaqa_qa_memory_sources,
    validate_qa_memory_against_sources,
)


TRANSDUCTIVE_MANIFEST_SCHEMA_VERSION = (
    "flowsteer.triviaqa.qa_memory.transductive.manifest.v1"
)
TRANSDUCTIVE_INDEX_FORMAT = (
    "flowsteer.triviaqa.qa-memory-transductive-embedding-index.v1"
)
TRANSDUCTIVE_CORPUS_NAME = "triviaqa-all-qa-transductive-memory"
EVALUATION_REGIME = "transductive_retrieval"
TRAIN_PARTITION = "train"
EVALUATION_PARTITION = "frozen_development_validation"
SOURCE_MEMBERSHIP_FILENAME = "source_membership.jsonl"
MANIFEST_FILENAME = "manifest.json"
NORMALIZATION = "l2"
SIMILARITY = "dot_product"

_MEMBERSHIP_SCHEMA_VERSION = (
    "flowsteer.triviaqa.qa_memory.transductive.source_membership.v1"
)
_MEMBERSHIP_FIELDS = frozenset(
    {
        "schema_version",
        "memory_id",
        "source_task_id",
        "base_task_id",
        "source_partition",
        "contains_evaluation_answer",
    }
)
_FILE_NAMES = MappingProxyType(
    {
        "embeddings": EMBEDDINGS_FILENAME,
        "memories": MEMORIES_FILENAME,
        "source_membership": SOURCE_MEMBERSHIP_FILENAME,
    }
)


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _task_row(
    row: object,
    *,
    line_number: int,
    required_split: str,
) -> tuple[str, str, str, str]:
    if not isinstance(row, Mapping):
        raise ValueError(f"TriviaQA row {line_number} is not an object")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("dataset_key") != "triviaqa":
        raise ValueError(f"row {line_number} is not a TriviaQA task")
    if row.get("split") != required_split:
        raise ValueError(f"TriviaQA row {line_number} has an incompatible split")
    sampling = metadata.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(f"TriviaQA row {line_number} has no sampling metadata")
    payload = metadata.get("evaluator_payload")
    accepted = payload.get("accepted_answers") if isinstance(payload, Mapping) else None
    if (
        not isinstance(accepted, list)
        or not accepted
        or not isinstance(accepted[0], str)
        or not accepted[0].strip()
    ):
        raise ValueError(f"TriviaQA row {line_number} has no canonical answer")
    task_id = _required_text(row.get("task_id"), field_name="task_id")
    base_task_id = _required_text(
        sampling.get("base_task_id"), field_name="base_task_id"
    )
    question = _required_text(row.get("question"), field_name="question")
    return task_id, base_task_id, question, accepted[0]


@dataclass(frozen=True, slots=True)
class TriviaQATransductiveSourceSet:
    """The explicit 512+128 source population for one transductive corpus."""

    sources: tuple[TriviaQATrainSource, ...]
    partition_by_source_task_id: Mapping[str, str]
    train_count: int
    evaluation_count: int
    evaluation_base_task_ids: frozenset[str]

    def __post_init__(self) -> None:
        train_count = _positive_integer(self.train_count, field_name="train_count")
        evaluation_count = _positive_integer(
            self.evaluation_count, field_name="evaluation_count"
        )
        if len(self.sources) != train_count + evaluation_count:
            raise ValueError("transductive source accounting does not close")
        partitions = dict(self.partition_by_source_task_id)
        if set(partitions) != {
            source.source_train_task_id for source in self.sources
        }:
            raise ValueError("source partition map does not cover the source population")
        if set(partitions.values()) != {TRAIN_PARTITION, EVALUATION_PARTITION}:
            raise ValueError("source partition map is incomplete")
        if sum(value == TRAIN_PARTITION for value in partitions.values()) != train_count:
            raise ValueError("train source count differs from partition map")
        if (
            sum(value == EVALUATION_PARTITION for value in partitions.values())
            != evaluation_count
        ):
            raise ValueError("evaluation source count differs from partition map")
        if len(self.evaluation_base_task_ids) != evaluation_count:
            raise ValueError("evaluation base-task count is incompatible")
        object.__setattr__(
            self,
            "partition_by_source_task_id",
            MappingProxyType(partitions),
        )

    @property
    def total_count(self) -> int:
        return len(self.sources)


def load_triviaqa_transductive_qa_memory_sources(
    train_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    *,
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
) -> TriviaQATransductiveSourceSet:
    """Load all train and frozen validation QA pairs into one source set.

    The train half is loaded through the existing train-only implementation,
    which also proves that train and validation base IDs are disjoint.  The
    validation half then deliberately reads both question and canonical answer;
    this is precisely why downstream metrics are transductive rather than
    official held-out results.
    """

    train_sources, isolated_validation_ids = load_triviaqa_qa_memory_sources(
        train_tasks_path,
        validation_tasks_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    validation_sources: list[TriviaQATrainSource] = []
    seen_task_ids: set[str] = set()
    validation_path = Path(validation_tasks_path)
    with validation_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"validation JSON is invalid at line {line_number}"
                ) from exc
            metadata = value.get("metadata") if isinstance(value, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get(
                "dataset_key"
            ) != "triviaqa":
                continue
            task_id, base_task_id, question, canonical_answer = _task_row(
                value,
                line_number=line_number,
                required_split="validation",
            )
            if task_id in seen_task_ids:
                raise ValueError("validation source task IDs are not unique")
            seen_task_ids.add(task_id)
            validation_sources.append(
                TriviaQATrainSource(
                    source_train_task_id=task_id,
                    base_task_id=base_task_id,
                    selection_index=len(train_sources) + len(validation_sources),
                    cycled_training_sample=False,
                    cycle_index=None,
                    original_question=question,
                    canonical_answer=canonical_answer,
                    native_split=EVALUATION_PARTITION,
                )
            )
    if len(validation_sources) != expected_validation_count:
        raise ValueError(
            "expected "
            f"{expected_validation_count} TriviaQA validation rows, found "
            f"{len(validation_sources)}"
        )
    actual_validation_ids = frozenset(
        source.base_task_id for source in validation_sources
    )
    if actual_validation_ids != isolated_validation_ids:
        raise ValueError("validation QA projection differs from frozen validation IDs")
    combined = tuple(train_sources) + tuple(validation_sources)
    source_ids = [source.source_train_task_id for source in combined]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("train and validation source task IDs overlap")
    partitions = {
        source.source_train_task_id: TRAIN_PARTITION for source in train_sources
    }
    partitions.update(
        {
            source.source_train_task_id: EVALUATION_PARTITION
            for source in validation_sources
        }
    )
    return TriviaQATransductiveSourceSet(
        sources=combined,
        partition_by_source_task_id=partitions,
        train_count=len(train_sources),
        evaluation_count=len(validation_sources),
        evaluation_base_task_ids=actual_validation_ids,
    )


@dataclass(frozen=True, slots=True)
class TriviaQATransductiveQAMemoryManifest:
    """Manifest that makes evaluation-answer inclusion non-ambiguous."""

    schema_version: str
    format: str
    record_kind: str
    tool_id: str
    retrieval_backend: str
    index_id: str
    corpus_name: str
    corpus_version: str
    source_dataset: str
    source_split: str
    contains_evaluation_answers: bool
    evaluation_regime: str
    official_heldout_eligible: bool
    validation_content_indexed: bool
    source_counts: Mapping[str, int]
    evaluation_memory_overlap_count: int
    unique_source_count: int
    cycled_count: int
    paraphrase_count: int
    memory_count: int
    paraphrase_versions: tuple[str, ...]
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    normalization: str
    similarity: str
    query_prefix: str
    embedding_text_template: str
    frozen_top_k: int
    snippet_characters: int
    tool_budget: Mapping[str, int]
    files: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        constants = {
            "schema_version": TRANSDUCTIVE_MANIFEST_SCHEMA_VERSION,
            "format": TRANSDUCTIVE_INDEX_FORMAT,
            "record_kind": "qa_memory",
            "tool_id": QA_MEMORY_TOOL_ID,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "corpus_name": TRANSDUCTIVE_CORPUS_NAME,
            "source_dataset": SOURCE_DATASET,
            "source_split": "train+frozen_development_validation",
            "evaluation_regime": EVALUATION_REGIME,
            "normalization": NORMALIZATION,
            "similarity": SIMILARITY,
            "query_prefix": BGE_QUERY_PREFIX,
            "embedding_text_template": EMBEDDING_TEXT_TEMPLATE,
        }
        for field_name, expected in constants.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"transductive manifest {field_name} is unsupported")
        if self.contains_evaluation_answers is not True:
            raise ValueError("contains_evaluation_answers must be true")
        if self.official_heldout_eligible is not False:
            raise ValueError("official_heldout_eligible must be false")
        if self.validation_content_indexed is not True:
            raise ValueError("validation_content_indexed must be true")
        expected_source_keys = {
            TRAIN_PARTITION,
            EVALUATION_PARTITION,
            "total",
        }
        source_counts = dict(self.source_counts)
        if set(source_counts) != expected_source_keys:
            raise ValueError("transductive manifest source_counts are incompatible")
        for name, count in source_counts.items():
            source_counts[name] = _positive_integer(count, field_name=f"source_counts.{name}")
        if source_counts["total"] != (
            source_counts[TRAIN_PARTITION] + source_counts[EVALUATION_PARTITION]
        ):
            raise ValueError("transductive source counts do not close")
        memory_count = _positive_integer(self.memory_count, field_name="memory_count")
        if source_counts["total"] != memory_count:
            raise ValueError("transductive source total differs from memory_count")
        overlap = _positive_integer(
            self.evaluation_memory_overlap_count,
            field_name="evaluation_memory_overlap_count",
        )
        if overlap != source_counts[EVALUATION_PARTITION]:
            raise ValueError("evaluation-memory overlap must cover all evaluation QA")
        unique_count = _positive_integer(
            self.unique_source_count, field_name="unique_source_count"
        )
        cycled_count = _nonnegative_integer(self.cycled_count, field_name="cycled_count")
        if unique_count + cycled_count != memory_count:
            raise ValueError("unique/cycled transductive accounting does not close")
        if _positive_integer(self.paraphrase_count, field_name="paraphrase_count") != memory_count:
            raise ValueError("paraphrase_count differs from memory_count")
        versions = tuple(
            _required_text(value, field_name="paraphrase_versions")
            for value in self.paraphrase_versions
        )
        if not versions or versions != tuple(sorted(set(versions))):
            raise ValueError("paraphrase_versions must be non-empty, unique, and sorted")
        object.__setattr__(self, "paraphrase_versions", versions)
        _required_text(self.embedding_model, field_name="embedding_model")
        _required_text(
            self.embedding_model_revision,
            field_name="embedding_model_revision",
        )
        _positive_integer(self.embedding_dimension, field_name="embedding_dimension")
        top_k = _positive_integer(self.frozen_top_k, field_name="frozen_top_k")
        if top_k > memory_count:
            raise ValueError("frozen_top_k exceeds transductive memory_count")
        _positive_integer(self.snippet_characters, field_name="snippet_characters")
        budget = dict(self.tool_budget)
        if set(budget) != {
            "max_tool_calls_per_agent_call",
            "max_turns_per_agent_call",
        }:
            raise ValueError("transductive tool_budget is incompatible")
        for name, count in budget.items():
            budget[name] = _positive_integer(count, field_name=f"tool_budget.{name}")
        if budget["max_turns_per_agent_call"] <= budget["max_tool_calls_per_agent_call"]:
            raise ValueError("ReAct turn budget must leave one completion turn")
        files = dict(self.files)
        if set(files) != set(_FILE_NAMES):
            raise ValueError("transductive file manifest is incomplete")
        normalized_files: dict[str, Mapping[str, str]] = {}
        for key, expected_name in _FILE_NAMES.items():
            entry = files[key]
            if not isinstance(entry, Mapping) or set(entry) != {"name", "sha256"}:
                raise ValueError(f"transductive files.{key} is incompatible")
            name = _required_text(entry["name"], field_name=f"files.{key}.name")
            if name != expected_name:
                raise ValueError(f"transductive files.{key}.name is incompatible")
            normalized_files[key] = MappingProxyType(
                {
                    "name": name,
                    "sha256": _required_sha256(
                        entry["sha256"], field_name=f"files.{key}.sha256"
                    ),
                }
            )
        if self.corpus_version != f"sha256:{normalized_files['memories']['sha256']}":
            raise ValueError("corpus_version differs from memories file")
        object.__setattr__(self, "source_counts", MappingProxyType(source_counts))
        object.__setattr__(self, "tool_budget", MappingProxyType(budget))
        object.__setattr__(self, "files", MappingProxyType(normalized_files))
        _required_sha256(self.index_id, field_name="index_id")
        if self.index_id != _stable_sha256(self._identity_value()):
            raise ValueError("transductive index_id differs from manifest identity")

    @property
    def train_count(self) -> int:
        return self.source_counts[TRAIN_PARTITION]

    @property
    def evaluation_count(self) -> int:
        return self.source_counts[EVALUATION_PARTITION]

    @property
    def validation_isolation_count(self) -> int:
        return 0

    def _identity_value(self) -> dict[str, object]:
        value = self.to_value()
        value.pop("index_id")
        return value

    def to_value(self) -> dict[str, object]:
        return {
            field_name: (
                dict(getattr(self, field_name))
                if field_name in {"source_counts", "tool_budget"}
                else {
                    key: dict(value)
                    for key, value in getattr(self, field_name).items()
                }
                if field_name == "files"
                else getattr(self, field_name)
            )
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def create(
        cls,
        *,
        source_set: TriviaQATransductiveSourceSet,
        paraphrase_versions: Sequence[str],
        embedding_model: str,
        embedding_model_revision: str,
        embedding_dimension: int,
        frozen_top_k: int,
        snippet_characters: int,
        max_tool_calls_per_agent_call: int,
        max_turns_per_agent_call: int,
        file_digests: Mapping[str, str],
    ) -> "TriviaQATransductiveQAMemoryManifest":
        source_counts = {
            TRAIN_PARTITION: source_set.train_count,
            EVALUATION_PARTITION: source_set.evaluation_count,
            "total": source_set.total_count,
        }
        value: dict[str, object] = {
            "schema_version": TRANSDUCTIVE_MANIFEST_SCHEMA_VERSION,
            "format": TRANSDUCTIVE_INDEX_FORMAT,
            "record_kind": "qa_memory",
            "tool_id": QA_MEMORY_TOOL_ID,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "index_id": "0" * 64,
            "corpus_name": TRANSDUCTIVE_CORPUS_NAME,
            "corpus_version": f"sha256:{file_digests['memories']}",
            "source_dataset": SOURCE_DATASET,
            "source_split": "train+frozen_development_validation",
            "contains_evaluation_answers": True,
            "evaluation_regime": EVALUATION_REGIME,
            "official_heldout_eligible": False,
            "validation_content_indexed": True,
            "source_counts": source_counts,
            "evaluation_memory_overlap_count": source_set.evaluation_count,
            "unique_source_count": len(
                {source.base_task_id for source in source_set.sources}
            ),
            "cycled_count": sum(
                source.cycled_training_sample for source in source_set.sources
            ),
            "paraphrase_count": source_set.total_count,
            "memory_count": source_set.total_count,
            "paraphrase_versions": tuple(sorted(set(paraphrase_versions))),
            "embedding_model": embedding_model,
            "embedding_model_revision": embedding_model_revision,
            "embedding_dimension": embedding_dimension,
            "normalization": NORMALIZATION,
            "similarity": SIMILARITY,
            "query_prefix": BGE_QUERY_PREFIX,
            "embedding_text_template": EMBEDDING_TEXT_TEMPLATE,
            "frozen_top_k": frozen_top_k,
            "snippet_characters": snippet_characters,
            "tool_budget": {
                "max_tool_calls_per_agent_call": max_tool_calls_per_agent_call,
                "max_turns_per_agent_call": max_turns_per_agent_call,
            },
            "files": {
                key: {"name": _FILE_NAMES[key], "sha256": file_digests[key]}
                for key in _FILE_NAMES
            },
        }
        identity = dict(value)
        identity.pop("index_id")
        value["index_id"] = _stable_sha256(identity)
        return cls.from_value(value)

    @classmethod
    def from_value(cls, value: object) -> "TriviaQATransductiveQAMemoryManifest":
        fields = frozenset(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("transductive QA-memory manifest fields are incompatible")
        normalized = {name: value[name] for name in fields}
        normalized["paraphrase_versions"] = tuple(
            str(item) for item in normalized["paraphrase_versions"]
        )
        return cls(**normalized)


def _membership_value(
    record: TriviaQAQAMemoryRecord,
    *,
    source_partition: str,
) -> dict[str, object]:
    if source_partition not in {TRAIN_PARTITION, EVALUATION_PARTITION}:
        raise ValueError("source_partition is unsupported")
    return {
        "schema_version": _MEMBERSHIP_SCHEMA_VERSION,
        "memory_id": record.memory_id,
        "source_task_id": record.source_train_task_id,
        "base_task_id": record.base_task_id,
        "source_partition": source_partition,
        "contains_evaluation_answer": source_partition == EVALUATION_PARTITION,
    }


def build_triviaqa_transductive_qa_memory_index(
    *,
    paraphrases_path: str | Path,
    train_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    output_dir: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_model_revision: str,
    frozen_top_k: int,
    max_tool_calls_per_agent_call: int,
    max_turns_per_agent_call: int,
    encoder: EmbeddingEncoder | None = None,
    batch_size: int = 64,
    snippet_characters: int = 512,
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
) -> TriviaQATransductiveQAMemoryManifest:
    """Build the explicitly transductive 640-record normalized BGE index."""

    model_name = _required_text(embedding_model, field_name="embedding_model")
    model_revision = _required_text(
        embedding_model_revision, field_name="embedding_model_revision"
    )
    frozen_top_k = _positive_integer(frozen_top_k, field_name="frozen_top_k")
    batch_size = _positive_integer(batch_size, field_name="batch_size")
    snippet_characters = _positive_integer(
        snippet_characters, field_name="snippet_characters"
    )
    max_tool_calls_per_agent_call = _positive_integer(
        max_tool_calls_per_agent_call,
        field_name="max_tool_calls_per_agent_call",
    )
    max_turns_per_agent_call = _positive_integer(
        max_turns_per_agent_call,
        field_name="max_turns_per_agent_call",
    )
    if max_turns_per_agent_call <= max_tool_calls_per_agent_call:
        raise ValueError("ReAct turn budget must leave one completion turn")
    source_set = load_triviaqa_transductive_qa_memory_sources(
        train_tasks_path,
        validation_tasks_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    records = load_materialized_qa_memory(
        paraphrases_path, expected_count=source_set.total_count
    )
    validate_qa_memory_against_sources(
        records, source_set.sources, require_complete=True
    )
    evaluation_records = {
        record.base_task_id
        for record in records
        if source_set.partition_by_source_task_id[record.source_train_task_id]
        == EVALUATION_PARTITION
    }
    if evaluation_records != source_set.evaluation_base_task_ids:
        raise ValueError("all frozen evaluation QA pairs must enter the memory corpus")
    versions = tuple(sorted({record.paraphrase_version for record in records}))
    if frozen_top_k > len(records):
        raise ValueError("frozen_top_k exceeds transductive QA-memory count")

    ordered = tuple(sorted(records, key=lambda record: record.memory_id))
    resolved_encoder = encoder or _get_embedding_model(model_name, model_revision)
    embeddings = _normalized_embeddings(
        resolved_encoder,
        [record.embedding_text() for record in ordered],
        batch_size=batch_size,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    memories_path = root / MEMORIES_FILENAME
    embeddings_path = root / EMBEDDINGS_FILENAME
    membership_path = root / SOURCE_MEMBERSHIP_FILENAME
    manifest_path = root / MANIFEST_FILENAME

    def write_memories(handle: Any) -> None:
        for record in ordered:
            handle.write(_canonical_json(record.to_value()) + b"\n")

    def write_memberships(handle: Any) -> None:
        for record in ordered:
            handle.write(
                _canonical_json(
                    _membership_value(
                        record,
                        source_partition=source_set.partition_by_source_task_id[
                            record.source_train_task_id
                        ],
                    )
                )
                + b"\n"
            )

    _write_atomic_bytes(memories_path, write_memories)
    _write_atomic_bytes(
        embeddings_path,
        lambda handle: np.save(handle, embeddings, allow_pickle=False),
    )
    _write_atomic_bytes(membership_path, write_memberships)
    digests = {
        "memories": _file_sha256(memories_path),
        "embeddings": _file_sha256(embeddings_path),
        "source_membership": _file_sha256(membership_path),
    }
    manifest = TriviaQATransductiveQAMemoryManifest.create(
        source_set=source_set,
        paraphrase_versions=versions,
        embedding_model=model_name,
        embedding_model_revision=model_revision,
        embedding_dimension=int(embeddings.shape[1]),
        frozen_top_k=frozen_top_k,
        snippet_characters=snippet_characters,
        max_tool_calls_per_agent_call=max_tool_calls_per_agent_call,
        max_turns_per_agent_call=max_turns_per_agent_call,
        file_digests=digests,
    )
    _write_atomic_bytes(
        manifest_path,
        lambda handle: handle.write(_canonical_json(manifest.to_value()) + b"\n"),
    )
    return manifest


class TriviaQATransductiveQAMemoryIndex(TriviaQAQAMemoryIndex):
    """Existing deterministic search/read implementation with a new manifest."""

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        encoder: EmbeddingEncoder | None = None,
    ) -> "TriviaQATransductiveQAMemoryIndex":
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError("transductive TriviaQA index directory is unavailable")
        manifest_path = root_path / MANIFEST_FILENAME
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("transductive manifest JSON is invalid") from exc
        manifest = TriviaQATransductiveQAMemoryManifest.from_value(value)
        for key, expected_name in _FILE_NAMES.items():
            path = root_path / expected_name
            if not path.is_file():
                raise FileNotFoundError(f"transductive index file is unavailable: {expected_name}")
            if _file_sha256(path) != manifest.files[key]["sha256"]:
                raise ValueError(f"transductive index file differs from manifest: {key}")
        records = load_materialized_qa_memory(
            root_path / MEMORIES_FILENAME,
            expected_count=manifest.memory_count,
        )
        if [record.memory_id for record in records] != sorted(
            record.memory_id for record in records
        ):
            raise ValueError("transductive QA-memory rows are not canonically ordered")
        memberships: list[Mapping[str, object]] = []
        with (root_path / SOURCE_MEMBERSHIP_FILENAME).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    membership = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"source membership JSON is invalid at line {line_number}"
                    ) from exc
                if not isinstance(membership, Mapping) or set(membership) != _MEMBERSHIP_FIELDS:
                    raise ValueError("source membership fields are incompatible")
                memberships.append(membership)
        if [entry["memory_id"] for entry in memberships] != [
            record.memory_id for record in records
        ]:
            raise ValueError("source membership order differs from QA-memory records")
        if sum(
            entry["source_partition"] == EVALUATION_PARTITION
            and entry["contains_evaluation_answer"] is True
            for entry in memberships
        ) != manifest.evaluation_memory_overlap_count:
            raise ValueError("evaluation-memory overlap differs from manifest")
        embeddings = np.load(
            root_path / EMBEDDINGS_FILENAME,
            mmap_mode="r",
            allow_pickle=False,
        )
        if embeddings.dtype != np.dtype("float32"):
            raise ValueError("transductive embedding dtype must be float32")
        if embeddings.shape != (
            manifest.memory_count,
            manifest.embedding_dimension,
        ):
            raise ValueError("transductive embedding shape differs from manifest")
        if not np.isfinite(embeddings).all():
            raise ValueError("transductive embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("transductive embeddings are not l2-normalized")
        resolved_encoder = encoder or _get_embedding_model(
            manifest.embedding_model,
            manifest.embedding_model_revision,
        )
        return cls(
            root=root_path,
            manifest=manifest,
            records=records,
            embeddings=embeddings,
            encoder=resolved_encoder,
        )


__all__ = [
    "EVALUATION_PARTITION",
    "EVALUATION_REGIME",
    "MANIFEST_FILENAME",
    "SOURCE_MEMBERSHIP_FILENAME",
    "TRAIN_PARTITION",
    "TRANSDUCTIVE_CORPUS_NAME",
    "TRANSDUCTIVE_INDEX_FORMAT",
    "TRANSDUCTIVE_MANIFEST_SCHEMA_VERSION",
    "TriviaQATransductiveQAMemoryIndex",
    "TriviaQATransductiveQAMemoryManifest",
    "TriviaQATransductiveSourceSet",
    "build_triviaqa_transductive_qa_memory_index",
    "load_triviaqa_transductive_qa_memory_sources",
]
