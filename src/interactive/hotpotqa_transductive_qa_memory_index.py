"""HotpotQA 512-train + 128-evaluation paired QA-memory adapter.

This module is an explicitly transductive specialization of the existing
HotpotQA QA-memory implementation.  It reuses the existing memory record,
semantic-preserving paraphrase boundary, normalized BGE encoder and
``search``/``read`` implementation.  The only added data-plane behavior is to
append the frozen 128 evaluation QA records after the frozen 512 training QA
records before materialization.

Because evaluation answers are deliberately present in this corpus, results
obtained with it are retrieval diagnostics and are never official held-out
metrics.  The train-only index remains a separate, unchanged artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .hotpotqa_embedding_index import _encode, _load_sentence_transformer
from .hotpotqa_qa_memory_index import (
    QA_MEMORY_DOCUMENT_TEMPLATE,
    HotpotQAQAMemory,
    HotpotQAQAMemoryIndex,
    HotpotQATrainQASource,
    materialize_hotpotqa_qa_memories,
)
from .task_dataset import qa_question_scope


TRANSDUCTIVE_QA_MEMORY_SCHEMA_VERSION = (
    "flowsteer.hotpotqa.transductive_qa_memory_index.v1"
)
TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION = (
    "flowsteer.hotpotqa.train512_eval128_qa_memory.v1"
)
TRANSDUCTIVE_EVALUATION_REGIME = "transductive_retrieval"


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class HotpotQATransductiveQASources:
    """Safe projections of the frozen 512 training and 128 evaluation QA."""

    train: tuple[HotpotQATrainQASource, ...]
    evaluation: tuple[HotpotQATrainQASource, ...]

    def __post_init__(self) -> None:
        train_ids = {source.source_train_task_id for source in self.train}
        evaluation_ids = {
            source.source_train_task_id for source in self.evaluation
        }
        if len(train_ids) != len(self.train):
            raise ValueError("transductive training source IDs are not unique")
        if len(evaluation_ids) != len(self.evaluation):
            raise ValueError("transductive evaluation source IDs are not unique")
        if train_ids & evaluation_ids:
            raise ValueError("train/evaluation source task IDs overlap")
        train_base_ids = {source.base_task_id for source in self.train}
        evaluation_base_ids = {source.base_task_id for source in self.evaluation}
        if train_base_ids & evaluation_base_ids:
            raise ValueError("train/evaluation source base_task_ids overlap")
        if any(source.cycled for source in self.evaluation):
            raise ValueError("frozen evaluation sources cannot be cycled")

    @property
    def combined(self) -> tuple[HotpotQATrainQASource, ...]:
        """Return the deterministic train-then-evaluation materialization order."""

        return self.train + self.evaluation


def _load_aligned_split(
    path: Path,
    *,
    expected_split: str,
    expected_count: int,
) -> tuple[HotpotQATrainQASource, ...]:
    """Project only QA and sampling provenance from one aligned JSONL split."""

    sources: list[HotpotQATrainQASource] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = _mapping(json.loads(line), "aligned task")
                metadata = _mapping(raw.get("metadata"), "aligned task metadata")
                if str(metadata.get("dataset_key", "")).casefold() != "hotpotqa":
                    raise ValueError("aligned task is not HotpotQA")
                if raw.get("split") != expected_split:
                    raise ValueError(
                        f"expected split {expected_split}, got {raw.get('split')}"
                    )
                sampling = _mapping(metadata.get("sampling"), "sampling metadata")
                if int(sampling.get("selection_index", -1)) != len(sources):
                    raise ValueError("selection_index is not sequential")
                task_id = _required_text(raw.get("task_id"), "task_id")
                base_task_id = _required_text(
                    sampling.get("base_task_id"), "sampling.base_task_id"
                )
                cycled = sampling.get("cycled_training_sample")
                if not isinstance(cycled, bool):
                    raise TypeError("sampling.cycled_training_sample must be boolean")
                question = qa_question_scope(
                    _required_text(raw.get("question"), "question")
                )
                answer = _required_text(raw.get("ground_truth"), "ground_truth")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            sources.append(
                HotpotQATrainQASource(
                    # ``source_train_task_id`` is the legacy v1 memory wire key.
                    # The transductive manifest records which suffix belongs to
                    # evaluation and never represents it as official training.
                    source_train_task_id=task_id,
                    base_task_id=base_task_id,
                    cycled=cycled,
                    question=question,
                    canonical_answer=answer,
                )
            )
    if len(sources) != expected_count:
        raise ValueError(
            f"expected {expected_count} {expected_split} records, got {len(sources)}"
        )
    return tuple(sources)


def load_hotpotqa_transductive_qa_sources(
    *,
    train_jsonl: Path,
    evaluation_jsonl: Path,
    expected_train_count: int = 512,
    expected_evaluation_count: int = 128,
) -> HotpotQATransductiveQASources:
    """Load the frozen 512+128 QA corpus without evaluator-private payloads."""

    if expected_train_count < 1 or expected_evaluation_count < 1:
        raise ValueError("transductive source counts must be positive")
    return HotpotQATransductiveQASources(
        train=_load_aligned_split(
            train_jsonl,
            expected_split="train",
            expected_count=expected_train_count,
        ),
        evaluation=_load_aligned_split(
            evaluation_jsonl,
            expected_split="validation",
            expected_count=expected_evaluation_count,
        ),
    )


@dataclass(frozen=True, slots=True)
class HotpotQATransductiveQAMemoryIndexManifest:
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
    source_evaluation_count: int
    unique_source_count: int
    cycled_record_count: int
    paraphrase_count: int
    frozen_validation_count: int
    evaluation_overlap_count: int
    source_split_id_overlap_count: int
    contains_evaluation_answers: bool
    evaluation_regime: str
    official_heldout_eligible: bool
    paraphrase_versions: tuple[str, ...]
    paraphrase_provenances: tuple[str, ...]
    document_template: str
    source_train_path: str
    source_evaluation_path: str
    memories_path: str
    embeddings_path: str

    def __post_init__(self) -> None:
        if self.schema_version != TRANSDUCTIVE_QA_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported transductive QA-memory index schema")
        if self.corpus_version != TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION:
            raise ValueError("unsupported transductive QA-memory corpus schema")
        if self.source_splits != ("train", "frozen_validation"):
            raise ValueError("transductive QA-memory source splits differ")
        if self.embedding_dimension < 1 or self.frozen_top_k < 1:
            raise ValueError("embedding dimension and top-k must be positive")
        if not self.normalized or self.similarity != "cosine":
            raise ValueError("transductive QA-memory requires normalized cosine")
        if self.source_record_count != (
            self.source_train_count + self.source_evaluation_count
        ):
            raise ValueError("transductive source counts are inconsistent")
        if self.source_evaluation_count != self.frozen_validation_count:
            raise ValueError("frozen validation count differs from evaluation source")
        if self.evaluation_overlap_count != self.frozen_validation_count:
            raise ValueError("all frozen validation QA must be present in the corpus")
        if self.source_split_id_overlap_count != 0:
            raise ValueError("train and validation source IDs must remain distinct")
        if self.contains_evaluation_answers is not True:
            raise ValueError("transductive corpus must declare evaluation answers")
        if self.evaluation_regime != TRANSDUCTIVE_EVALUATION_REGIME:
            raise ValueError("transductive evaluation regime differs")
        if self.official_heldout_eligible is not False:
            raise ValueError("transductive retrieval cannot be held-out eligible")
        if self.paraphrase_count != self.source_record_count:
            raise ValueError("every transductive source must have one paraphrase")
        if self.unique_source_count + self.cycled_record_count != self.source_record_count:
            raise ValueError("unique/cycled transductive source counts are inconsistent")
        if not self.paraphrase_versions or not self.paraphrase_provenances:
            raise ValueError("paraphrase version and provenance are required")
        if self.document_template != QA_MEMORY_DOCUMENT_TEMPLATE:
            raise ValueError("QA-memory document template differs from the reused v1")

    @property
    def train_record_count(self) -> int:
        """Compatibility property used by the reused read-only index runtime."""

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
    ) -> "HotpotQATransductiveQAMemoryIndexManifest":
        mapping = _mapping(value, "transductive QA-memory manifest")
        expected = frozenset(cls.__dataclass_fields__)
        actual = {str(key) for key in mapping}
        if actual != expected:
            raise ValueError(
                "transductive QA-memory manifest fields differ: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
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


def build_hotpotqa_transductive_qa_memory_index(
    *,
    index_dir: Path,
    train_jsonl: Path,
    evaluation_jsonl: Path,
    paraphrases: Sequence[Mapping[str, object]],
    embedding_model_path: str,
    embedding_model_id: str,
    embedding_device: str,
    frozen_top_k: int,
    expected_train_count: int = 512,
    expected_evaluation_count: int = 128,
) -> HotpotQATransductiveQAMemoryIndexManifest:
    """Build an isolated 512+128 diagnostic index with the existing encoder."""

    if frozen_top_k < 1:
        raise ValueError("frozen_top_k must be positive")
    sources = load_hotpotqa_transductive_qa_sources(
        train_jsonl=train_jsonl,
        evaluation_jsonl=evaluation_jsonl,
        expected_train_count=expected_train_count,
        expected_evaluation_count=expected_evaluation_count,
    )
    memories = materialize_hotpotqa_qa_memories(sources.combined, paraphrases)
    model = _load_sentence_transformer(embedding_model_path, embedding_device)
    vectors = _encode(model, [memory.document_text for memory in memories])

    index_dir = index_dir.expanduser().resolve()
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError("transductive QA-memory index directory must be empty")
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
    versions = tuple(sorted({memory.paraphrase_version for memory in memories}))
    provenances = tuple(
        sorted({memory.paraphrase_provenance for memory in memories})
    )
    cycled_count = sum(memory.cycled for memory in memories)
    unique_source_count = len({memory.base_task_id for memory in memories})
    manifest = HotpotQATransductiveQAMemoryIndexManifest(
        schema_version=TRANSDUCTIVE_QA_MEMORY_SCHEMA_VERSION,
        index_id=(
            "hotpotqa-transductive-qa-memory-"
            f"d{vectors.shape[1]}-n{len(memories)}-topk{frozen_top_k}-v1"
        ),
        corpus_version=TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION,
        source="HotpotQA aligned 512 train + 128 frozen validation",
        source_splits=("train", "frozen_validation"),
        embedding_model=embedding_model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimension=int(vectors.shape[1]),
        normalized=True,
        similarity="cosine",
        frozen_top_k=frozen_top_k,
        source_record_count=len(memories),
        source_train_count=len(sources.train),
        source_evaluation_count=len(sources.evaluation),
        unique_source_count=unique_source_count,
        cycled_record_count=cycled_count,
        paraphrase_count=len(memories),
        frozen_validation_count=len(sources.evaluation),
        evaluation_overlap_count=len(sources.evaluation),
        source_split_id_overlap_count=0,
        contains_evaluation_answers=True,
        evaluation_regime=TRANSDUCTIVE_EVALUATION_REGIME,
        official_heldout_eligible=False,
        paraphrase_versions=versions,
        paraphrase_provenances=provenances,
        document_template=QA_MEMORY_DOCUMENT_TEMPLATE,
        source_train_path=str(train_jsonl.expanduser().resolve()),
        source_evaluation_path=str(evaluation_jsonl.expanduser().resolve()),
        memories_path=memories_path.name,
        embeddings_path=embeddings_path.name,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_value(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class HotpotQATransductiveQAMemoryIndex(HotpotQAQAMemoryIndex):
    """Reuse the existing deterministic QA-memory ``search``/``read`` runtime."""

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        embedding_model_path: str | None = None,
        embedding_device: str = "cpu",
    ) -> "HotpotQATransductiveQAMemoryIndex":
        index_dir = index_dir.expanduser().resolve()
        manifest = HotpotQATransductiveQAMemoryIndexManifest.from_value(
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
        # The parent runtime is deliberately reused through its structural
        # manifest boundary.  ``train_record_count`` above aliases the total
        # transductive corpus size without changing the existing train-only
        # manifest or index implementation.
        return cls(
            manifest=manifest,  # type: ignore[arg-type]
            memories=memories,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model=model,
        )


__all__ = [
    "TRANSDUCTIVE_EVALUATION_REGIME",
    "TRANSDUCTIVE_QA_MEMORY_CORPUS_VERSION",
    "TRANSDUCTIVE_QA_MEMORY_SCHEMA_VERSION",
    "HotpotQATransductiveQAMemoryIndex",
    "HotpotQATransductiveQAMemoryIndexManifest",
    "HotpotQATransductiveQASources",
    "build_hotpotqa_transductive_qa_memory_index",
    "load_hotpotqa_transductive_qa_sources",
]
