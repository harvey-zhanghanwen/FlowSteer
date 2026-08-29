"""Full native HotpotQA Q-A memory index.

This is a scale-only, explicitly transductive specialization of the existing
HotpotQA QA-memory boundary.  Native ``train`` and ``validation`` Q-A pairs are
projected through the repository's existing HotpotQA converter; contexts,
supporting-fact labels, evaluator payloads, and receipts are never copied into
the source or memory records.  The existing semantically-preserving memory
record, normalized BGE encoder, and deterministic ``search``/``read`` runtime
are reused unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from scripts.prepare_agentgraph_datasets import _hotpot_records, _path

from .hotpotqa_embedding_index import _encode, _load_sentence_transformer
from .hotpotqa_qa_memory_index import (
    QA_MEMORY_DOCUMENT_TEMPLATE,
    HotpotQAQAMemory,
    HotpotQAQAMemoryIndex,
    HotpotQATrainQASource,
    materialize_hotpotqa_qa_memories,
)
from .task_dataset import qa_question_scope


FULL_DATASET_QA_MEMORY_SCHEMA_VERSION = (
    "flowsteer.hotpotqa.full_dataset_qa_memory_index.v1"
)
FULL_DATASET_QA_MEMORY_CORPUS_VERSION = (
    "flowsteer.hotpotqa.native_train_validation_qa_memory.v1"
)
FULL_DATASET_EVALUATION_SCOPE = "in_database_transductive"


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _hotpot_config(dataset_catalog_path: Path) -> Mapping[str, object]:
    with dataset_catalog_path.expanduser().resolve().open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    sources = catalog.get("sources") if isinstance(catalog, Mapping) else None
    config = sources.get("hotpotqa") if isinstance(sources, Mapping) else None
    if not isinstance(config, Mapping):
        raise ValueError("dataset catalog has no HotpotQA source")
    if tuple(config.get("candidate_sequence", ())) != ("train", "validation"):
        raise ValueError("HotpotQA native split order differs from train, validation")
    return config


@dataclass(frozen=True, slots=True)
class HotpotQAFullDatasetQASources:
    """Safe Q-A-only projections of both native HotpotQA splits."""

    train: tuple[HotpotQATrainQASource, ...]
    validation: tuple[HotpotQATrainQASource, ...]

    def __post_init__(self) -> None:
        train_ids = {source.source_train_task_id for source in self.train}
        validation_ids = {
            source.source_train_task_id for source in self.validation
        }
        if len(train_ids) != len(self.train):
            raise ValueError("native HotpotQA train source IDs are not unique")
        if len(validation_ids) != len(self.validation):
            raise ValueError("native HotpotQA validation source IDs are not unique")
        if train_ids & validation_ids:
            raise ValueError("native HotpotQA train/validation source IDs overlap")
        if any(source.cycled for source in self.combined):
            raise ValueError("native full-dataset HotpotQA sources cannot be cycled")

    @property
    def combined(self) -> tuple[HotpotQATrainQASource, ...]:
        return self.train + self.validation


def load_hotpotqa_full_dataset_qa_sources(
    *,
    dataset_catalog_path: Path,
    expected_train_count: int = 90_447,
    expected_validation_count: int = 7_405,
) -> HotpotQAFullDatasetQASources:
    """Project only public question, answer, ID, and split provenance."""

    if expected_train_count < 1 or expected_validation_count < 1:
        raise ValueError("full-dataset source counts must be positive")
    config = _hotpot_config(dataset_catalog_path)
    projected: dict[str, list[HotpotQATrainQASource]] = {
        "train": [],
        "validation": [],
    }
    for record in _hotpot_records(config):
        split = record.get("split")
        if split not in projected:
            raise ValueError(f"unexpected native HotpotQA split: {split}")
        task_id = _required_text(record.get("task_id"), "task_id")
        if not task_id.startswith("hotpotqa:"):
            raise ValueError("native HotpotQA task ID is incompatible")
        source = HotpotQATrainQASource(
            # This legacy wire name is shared by the existing memory record.
            source_train_task_id=task_id,
            base_task_id=task_id,
            cycled=False,
            question=qa_question_scope(
                _required_text(record.get("question"), "question")
            ),
            canonical_answer=_required_text(
                record.get("ground_truth"), "ground_truth"
            ),
        )
        projected[str(split)].append(source)
    if len(projected["train"]) != expected_train_count:
        raise ValueError(
            f"expected {expected_train_count} native train records, got "
            f"{len(projected['train'])}"
        )
    if len(projected["validation"]) != expected_validation_count:
        raise ValueError(
            f"expected {expected_validation_count} native validation records, got "
            f"{len(projected['validation'])}"
        )
    return HotpotQAFullDatasetQASources(
        train=tuple(projected["train"]),
        validation=tuple(projected["validation"]),
    )


@dataclass(frozen=True, slots=True)
class HotpotQAFullDatasetQAMemoryIndexManifest:
    schema_version: str
    index_id: str
    corpus_version: str
    source: str
    source_splits: tuple[str, ...]
    embedding_model: str
    embedding_model_path: str
    embedding_dimension: int
    normalized: bool
    similarity: str
    frozen_top_k: int
    source_record_count: int
    source_train_count: int
    source_validation_count: int
    unique_source_count: int
    cycled_record_count: int
    paraphrase_count: int
    frozen_evaluation_count: int
    evaluation_overlap_count: int
    contains_evaluation_answers: bool
    evaluation_scope: str
    official_heldout_eligible: bool
    paraphrase_versions: tuple[str, ...]
    paraphrase_provenances: tuple[str, ...]
    document_template: str
    source_dataset_catalog_path: str
    source_train_path: str
    source_validation_path: str
    memories_path: str
    embeddings_path: str

    def __post_init__(self) -> None:
        if self.schema_version != FULL_DATASET_QA_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported full-dataset QA-memory index schema")
        if self.corpus_version != FULL_DATASET_QA_MEMORY_CORPUS_VERSION:
            raise ValueError("unsupported full-dataset QA-memory corpus schema")
        if self.source_splits != ("train", "validation"):
            raise ValueError("full-dataset QA-memory source splits differ")
        if self.embedding_dimension < 1 or self.frozen_top_k < 1:
            raise ValueError("embedding dimension and top-k must be positive")
        if not self.normalized or self.similarity != "cosine":
            raise ValueError("full-dataset QA-memory requires normalized cosine")
        if self.source_record_count != (
            self.source_train_count + self.source_validation_count
        ):
            raise ValueError("full-dataset source counts are inconsistent")
        if self.unique_source_count != self.source_record_count:
            raise ValueError("every native HotpotQA source ID must be unique")
        if self.cycled_record_count != 0:
            raise ValueError("native full-dataset sources cannot be cycled")
        if self.paraphrase_count != self.source_record_count:
            raise ValueError("every full-dataset source must have one memory")
        if self.frozen_evaluation_count < 1:
            raise ValueError("frozen evaluation identity count must be positive")
        if self.evaluation_overlap_count != self.frozen_evaluation_count:
            raise ValueError("every frozen evaluation Q-A must be in the corpus")
        if self.contains_evaluation_answers is not True:
            raise ValueError("full-dataset corpus must declare evaluation answers")
        if self.evaluation_scope != FULL_DATASET_EVALUATION_SCOPE:
            raise ValueError("full-dataset evaluation scope differs")
        if self.official_heldout_eligible is not False:
            raise ValueError("in-database retrieval is not held-out eligible")
        if not self.paraphrase_versions or not self.paraphrase_provenances:
            raise ValueError("paraphrase version and provenance are required")
        if self.document_template != QA_MEMORY_DOCUMENT_TEMPLATE:
            raise ValueError("QA-memory document template differs from the reused v1")

    @property
    def train_record_count(self) -> int:
        """Compatibility alias used by the reused read-only runtime."""

        return self.source_record_count

    def to_value(self) -> dict[str, object]:
        value = asdict(self)
        value["source_splits"] = list(self.source_splits)
        value["paraphrase_versions"] = list(self.paraphrase_versions)
        value["paraphrase_provenances"] = list(self.paraphrase_provenances)
        return value

    @classmethod
    def from_value(
        cls, value: object
    ) -> "HotpotQAFullDatasetQAMemoryIndexManifest":
        mapping = _mapping(value, "full-dataset QA-memory manifest")
        expected = frozenset(cls.__dataclass_fields__)
        actual = {str(key) for key in mapping}
        if actual != expected:
            raise ValueError(
                "full-dataset QA-memory manifest fields differ: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        fields = {name: mapping[name] for name in expected}
        fields["source_splits"] = tuple(str(item) for item in fields["source_splits"])
        fields["paraphrase_versions"] = tuple(
            str(item) for item in fields["paraphrase_versions"]
        )
        fields["paraphrase_provenances"] = tuple(
            str(item) for item in fields["paraphrase_provenances"]
        )
        return cls(**fields)  # type: ignore[arg-type]


def _native_source_paths(
    dataset_catalog_path: Path,
) -> tuple[str, str]:
    config = _hotpot_config(dataset_catalog_path)
    files = _mapping(config.get("files"), "HotpotQA files")
    base = _path(str(config["path"]))
    return str(base / str(files["train"])), str(base / str(files["validation"]))


def build_hotpotqa_full_dataset_qa_memory_index(
    *,
    index_dir: Path,
    dataset_catalog_path: Path,
    frozen_evaluation_task_ids: Sequence[str],
    paraphrases: Sequence[Mapping[str, object]],
    embedding_model_path: str,
    embedding_model_id: str,
    embedding_device: str,
    frozen_top_k: int,
    expected_train_count: int = 90_447,
    expected_validation_count: int = 7_405,
) -> HotpotQAFullDatasetQAMemoryIndexManifest:
    """Build the all-native-QA index with the existing normalized encoder."""

    if frozen_top_k < 1:
        raise ValueError("frozen_top_k must be positive")
    sources = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=dataset_catalog_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    source_ids = {source.source_train_task_id for source in sources.combined}
    evaluation_ids = {_required_text(item, "evaluation task ID") for item in frozen_evaluation_task_ids}
    if len(evaluation_ids) != len(frozen_evaluation_task_ids):
        raise ValueError("frozen evaluation task IDs are not unique")
    overlap_count = len(source_ids & evaluation_ids)
    if overlap_count != len(evaluation_ids):
        raise ValueError("some frozen evaluation Q-A records are absent from full dataset")

    memories = materialize_hotpotqa_qa_memories(sources.combined, paraphrases)
    model = _load_sentence_transformer(embedding_model_path, embedding_device)
    vectors = _encode(model, [memory.document_text for memory in memories])

    index_dir = index_dir.expanduser().resolve()
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError("full-dataset QA-memory index directory must be empty")
    index_dir.mkdir(parents=True, exist_ok=True)
    memories_path = index_dir / "memories.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"
    memories_path.write_text(
        "".join(
            json.dumps(memory.to_value(), ensure_ascii=False, sort_keys=True) + "\n"
            for memory in memories
        ),
        encoding="utf-8",
    )
    np.save(embeddings_path, vectors, allow_pickle=False)
    source_train_path, source_validation_path = _native_source_paths(
        dataset_catalog_path
    )
    manifest = HotpotQAFullDatasetQAMemoryIndexManifest(
        schema_version=FULL_DATASET_QA_MEMORY_SCHEMA_VERSION,
        index_id=(
            "hotpotqa-full-dataset-qa-memory-"
            f"d{vectors.shape[1]}-n{len(memories)}-topk{frozen_top_k}-v1"
        ),
        corpus_version=FULL_DATASET_QA_MEMORY_CORPUS_VERSION,
        source="HotpotQA native train + validation Q-A",
        source_splits=("train", "validation"),
        embedding_model=embedding_model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimension=int(vectors.shape[1]),
        normalized=True,
        similarity="cosine",
        frozen_top_k=frozen_top_k,
        source_record_count=len(memories),
        source_train_count=len(sources.train),
        source_validation_count=len(sources.validation),
        unique_source_count=len({memory.base_task_id for memory in memories}),
        cycled_record_count=sum(memory.cycled for memory in memories),
        paraphrase_count=len(memories),
        frozen_evaluation_count=len(evaluation_ids),
        evaluation_overlap_count=overlap_count,
        contains_evaluation_answers=True,
        evaluation_scope=FULL_DATASET_EVALUATION_SCOPE,
        official_heldout_eligible=False,
        paraphrase_versions=tuple(
            sorted({memory.paraphrase_version for memory in memories})
        ),
        paraphrase_provenances=tuple(
            sorted({memory.paraphrase_provenance for memory in memories})
        ),
        document_template=QA_MEMORY_DOCUMENT_TEMPLATE,
        source_dataset_catalog_path=str(dataset_catalog_path.expanduser().resolve()),
        source_train_path=source_train_path,
        source_validation_path=source_validation_path,
        memories_path=memories_path.name,
        embeddings_path=embeddings_path.name,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_value(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class HotpotQAFullDatasetQAMemoryIndex(HotpotQAQAMemoryIndex):
    """Reuse the existing deterministic QA-memory ``search``/``read`` runtime."""

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        embedding_model_path: str | None = None,
        embedding_device: str = "cpu",
    ) -> "HotpotQAFullDatasetQAMemoryIndex":
        index_dir = index_dir.expanduser().resolve()
        manifest = HotpotQAFullDatasetQAMemoryIndexManifest.from_value(
            json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        )
        memories = tuple(
            HotpotQAQAMemory.from_value(json.loads(line))
            for line in (index_dir / manifest.memories_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        embeddings = np.load(index_dir / manifest.embeddings_path, allow_pickle=False)
        model = _load_sentence_transformer(
            embedding_model_path or manifest.embedding_model_path,
            embedding_device,
        )
        return cls(
            manifest=manifest,  # type: ignore[arg-type]
            memories=memories,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model=model,
        )


__all__ = [
    "FULL_DATASET_EVALUATION_SCOPE",
    "FULL_DATASET_QA_MEMORY_CORPUS_VERSION",
    "FULL_DATASET_QA_MEMORY_SCHEMA_VERSION",
    "HotpotQAFullDatasetQAMemoryIndex",
    "HotpotQAFullDatasetQAMemoryIndexManifest",
    "HotpotQAFullDatasetQASources",
    "build_hotpotqa_full_dataset_qa_memory_index",
    "load_hotpotqa_full_dataset_qa_sources",
]
