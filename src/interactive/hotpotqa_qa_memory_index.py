"""Train-only HotpotQA QA-memory dense index.

This is a thin specialization of the existing HotpotQA dense-index boundary:
the sentence-transformer loader/normalizing encoder are reused, while the
indexed document is an offline, semantically-preserving paraphrase of one
*training* question and answer.  Paraphrase generation is deliberately not
implemented here.  A caller must inject a complete offline materialization;
the builder fails closed when coverage, provenance, semantic-preservation
attestation, or canonical-answer preservation is missing.

The aligned training adapter projects only the fields needed to construct the
memory.  In particular, ``supporting_facts`` and ``evaluator_payload`` may
exist in the aligned source file but are never copied into a source record,
memory record, corpus, embedding input, or manifest.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .hotpotqa_embedding_index import _encode, _load_sentence_transformer
from .task_dataset import qa_question_scope


QA_MEMORY_SCHEMA_VERSION = "flowsteer.hotpotqa.qa_memory_index.v1"
QA_MEMORY_CORPUS_VERSION = "flowsteer.hotpotqa.train_qa_memory.v1"
QA_MEMORY_SCHEMA_VERSIONS = frozenset(
    {
        QA_MEMORY_SCHEMA_VERSION,
        "flowsteer.hotpotqa.qa_memory_index.v2",
    }
)
QA_MEMORY_CORPUS_VERSIONS = frozenset(
    {
        QA_MEMORY_CORPUS_VERSION,
        "flowsteer.hotpotqa.train_qa_memory.v2",
    }
)
QA_MEMORY_DOCUMENT_TEMPLATE = "Question: {question}\nAnswer: {answer_statement}"

_PARAPHRASE_FIELDS = frozenset(
    {
        "source_train_task_id",
        "paraphrase_question",
        "paraphrase_answer_statement",
        "paraphrase_provenance",
        "paraphrase_version",
        "semantic_preservation_attested",
    }
)
_MEMORY_FIELDS = frozenset(
    {
        "memory_id",
        "source_train_task_id",
        "base_task_id",
        "cycled",
        "paraphrase_question",
        "paraphrase_answer_statement",
        "canonical_answer",
        "paraphrase_provenance",
        "paraphrase_version",
    }
)
_FORBIDDEN_PRIVATE_KEYS = frozenset(
    {
        "accepted_aliases",
        "evaluator_payload",
        "evaluator_receipt",
        "ground_truth",
        "supporting_fact",
        "supporting_facts",
        "validation_answer",
        "validation_ground_truth",
        "validation_question",
    }
)
_CYCLE_SUFFIX = re.compile(r":cycle-\d{4}$")


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _reject_private_keys(value: object, *, path: str) -> None:
    """Reject evaluator/held-out fields at the materialized-memory boundary."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().casefold().replace("-", "_")
            if normalized_key in _FORBIDDEN_PRIVATE_KEYS:
                raise ValueError(f"private field {path}.{key} is not allowed")
            _reject_private_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_private_keys(item, path=f"{path}[{index}]")


def _expect_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = {str(key) for key in value}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class HotpotQATrainQASource:
    """Safe projection of one aligned HotpotQA training record."""

    source_train_task_id: str
    base_task_id: str
    cycled: bool
    question: str
    canonical_answer: str

    def __post_init__(self) -> None:
        _required_text(self.source_train_task_id, "source_train_task_id")
        _required_text(self.base_task_id, "base_task_id")
        _required_text(self.question, "question")
        _required_text(self.canonical_answer, "canonical_answer")
        if _base_task_id(self.source_train_task_id) != self.base_task_id:
            raise ValueError("training source task/base_task_id provenance is inconsistent")
        if self.cycled != bool(_CYCLE_SUFFIX.search(self.source_train_task_id)):
            raise ValueError("training source cycle provenance is inconsistent")


@dataclass(frozen=True, slots=True)
class HotpotQAQAMemory:
    memory_id: str
    source_train_task_id: str
    base_task_id: str
    cycled: bool
    paraphrase_question: str
    paraphrase_answer_statement: str
    canonical_answer: str
    paraphrase_provenance: str
    paraphrase_version: str

    def __post_init__(self) -> None:
        for name in (
            "memory_id",
            "source_train_task_id",
            "base_task_id",
            "paraphrase_question",
            "paraphrase_answer_statement",
            "canonical_answer",
            "paraphrase_provenance",
            "paraphrase_version",
        ):
            _required_text(getattr(self, name), name)
        if _base_task_id(self.source_train_task_id) != self.base_task_id:
            raise ValueError("QA-memory source/base_task_id provenance is inconsistent")
        if self.cycled != bool(_CYCLE_SUFFIX.search(self.source_train_task_id)):
            raise ValueError("QA-memory cycle provenance is inconsistent")
        canonical = _normalized_text(self.canonical_answer)
        statement = _normalized_text(self.paraphrase_answer_statement)
        exact_span_required = self.paraphrase_version.endswith("-v2")
        if (
            self.canonical_answer not in self.paraphrase_answer_statement
            if exact_span_required
            else canonical not in statement
        ):
            raise ValueError("QA-memory answer statement lost the canonical answer span")
        if statement == canonical:
            raise ValueError("QA-memory answer must be a declarative answer statement")

    def to_value(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: object) -> "HotpotQAQAMemory":
        mapping = _mapping(value, "QA-memory record")
        _reject_private_keys(mapping, path="memory")
        _expect_exact_keys(mapping, _MEMORY_FIELDS, "QA-memory record")
        if not isinstance(mapping["cycled"], bool):
            raise TypeError("QA-memory cycled must be a boolean")
        fields = {
            key: mapping[key] if key == "cycled" else _required_text(mapping[key], key)
            for key in _MEMORY_FIELDS
        }
        return cls(**fields)  # type: ignore[arg-type]

    @property
    def document_text(self) -> str:
        return QA_MEMORY_DOCUMENT_TEMPLATE.format(
            question=self.paraphrase_question,
            answer_statement=self.paraphrase_answer_statement,
        )


@dataclass(frozen=True, slots=True)
class HotpotQAQAMemorySearchHit:
    memory_id: str
    source_train_task_id: str
    paraphrase_question: str
    paraphrase_answer_statement: str
    similarity: float
    rank: int


@dataclass(frozen=True, slots=True)
class HotpotQAQAMemoryIndexManifest:
    schema_version: str
    index_id: str
    corpus_version: str
    source: str
    source_split: str
    embedding_model: str
    embedding_model_path: str
    embedding_dimension: int
    normalized: bool
    similarity: str
    frozen_top_k: int
    train_record_count: int
    unique_source_count: int
    cycled_record_count: int
    paraphrase_count: int
    heldout_validation_count: int
    validation_overlap_count: int
    paraphrase_versions: tuple[str, ...]
    paraphrase_provenances: tuple[str, ...]
    document_template: str
    source_train_path: str
    memories_path: str
    embeddings_path: str

    def __post_init__(self) -> None:
        if self.schema_version not in QA_MEMORY_SCHEMA_VERSIONS:
            raise ValueError("unsupported HotpotQA QA-memory index schema")
        if self.corpus_version not in QA_MEMORY_CORPUS_VERSIONS:
            raise ValueError("unsupported HotpotQA QA-memory corpus schema")
        if not self.index_id or self.source_split != "train":
            raise ValueError("QA-memory identity/split is invalid")
        if self.embedding_dimension < 1 or self.frozen_top_k < 1:
            raise ValueError("embedding dimension and frozen top-k must be positive")
        if not self.normalized or self.similarity != "cosine":
            raise ValueError("QA-memory index requires normalized cosine")
        if self.validation_overlap_count != 0:
            raise ValueError("held-out validation overlaps QA-memory sources")
        if self.train_record_count != self.paraphrase_count:
            raise ValueError("every training record must have one paraphrase")
        if self.unique_source_count + self.cycled_record_count != self.train_record_count:
            raise ValueError("unique/cycled QA-memory counts are inconsistent")
        if not self.paraphrase_versions or not self.paraphrase_provenances:
            raise ValueError("paraphrase provenance/version must be recorded")
        if self.document_template != QA_MEMORY_DOCUMENT_TEMPLATE:
            raise ValueError("QA-memory document template differs from v1")

    def to_value(self) -> dict[str, object]:
        value = asdict(self)
        value["paraphrase_versions"] = list(self.paraphrase_versions)
        value["paraphrase_provenances"] = list(self.paraphrase_provenances)
        return value

    @classmethod
    def from_value(cls, value: object) -> "HotpotQAQAMemoryIndexManifest":
        mapping = _mapping(value, "QA-memory manifest")
        expected = frozenset(cls.__dataclass_fields__)
        _reject_private_keys(mapping, path="manifest")
        _expect_exact_keys(mapping, expected, "QA-memory manifest")
        fields = {name: mapping[name] for name in expected}
        fields["paraphrase_versions"] = tuple(
            str(item) for item in fields["paraphrase_versions"]
        )
        fields["paraphrase_provenances"] = tuple(
            str(item) for item in fields["paraphrase_provenances"]
        )
        return cls(**fields)  # type: ignore[arg-type]


def _base_task_id(task_id: str) -> str:
    return _CYCLE_SUFFIX.sub("", task_id)


def load_hotpotqa_train_qa_sources(
    train_jsonl: Path,
    *,
    validation_task_ids: Sequence[str],
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
) -> tuple[HotpotQATrainQASource, ...]:
    """Load the frozen train split through an explicit safe projection.

    The aligned JSONL contains evaluator-only metadata for other consumers.
    This adapter intentionally reads only dataset/split, sampling provenance,
    question, and the *training* ground truth.  The returned dataclass has no
    container in which evaluator metadata or supporting-fact labels can survive.
    """

    validation_base_ids = {_base_task_id(str(item)) for item in validation_task_ids}
    if len(validation_base_ids) != expected_validation_count:
        raise ValueError("held-out validation task count/identity differs from freeze")
    sources: list[HotpotQATrainQASource] = []
    with train_jsonl.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = _mapping(json.loads(line), "aligned task")
                metadata = _mapping(raw.get("metadata"), "aligned task metadata")
                if str(metadata.get("dataset_key", "")).casefold() != "hotpotqa":
                    continue
                if raw.get("split") != "train":
                    raise ValueError("HotpotQA QA-memory source is not in train split")
                sampling = _mapping(metadata.get("sampling"), "sampling metadata")
                source_task_id = _required_text(raw.get("task_id"), "task_id")
                base_task_id = _required_text(
                    sampling.get("base_task_id"), "sampling.base_task_id"
                )
                cycled = sampling.get("cycled_training_sample")
                if not isinstance(cycled, bool):
                    raise TypeError("sampling.cycled_training_sample must be boolean")
                if int(sampling.get("selection_index", -1)) != len(sources):
                    raise ValueError("HotpotQA train selection_index is not sequential")
                if _base_task_id(source_task_id) != base_task_id:
                    raise ValueError("training task/base_task_id provenance is inconsistent")
                question = qa_question_scope(
                    _required_text(raw.get("question"), "training question")
                )
                answer = _required_text(raw.get("ground_truth"), "training answer")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{train_jsonl}:{line_number}: {exc}") from exc
            sources.append(
                HotpotQATrainQASource(
                    source_train_task_id=source_task_id,
                    base_task_id=base_task_id,
                    cycled=cycled,
                    question=question,
                    canonical_answer=answer,
                )
            )

    if len(sources) != expected_train_count:
        raise ValueError(
            f"expected {expected_train_count} HotpotQA train records, got {len(sources)}"
        )
    source_ids = [source.source_train_task_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("HotpotQA train source task IDs are not unique")
    source_base_ids = {source.base_task_id for source in sources}
    overlap = sorted(source_base_ids & validation_base_ids)
    if overlap:
        raise ValueError("HotpotQA train/held-out validation base_task_id overlap")
    noncycled_base_ids = {source.base_task_id for source in sources if not source.cycled}
    noncycled_sources = [source for source in sources if not source.cycled]
    cycled_sources = [source for source in sources if source.cycled]
    if len(noncycled_base_ids) != len(noncycled_sources):
        raise ValueError("non-cycled training base_task_ids are not unique")
    if cycled_sources and sources[-len(cycled_sources) :] != cycled_sources:
        raise ValueError("cycled training records must follow the unique training pool")
    if any(
        source.cycled and source.base_task_id not in noncycled_base_ids
        for source in sources
    ):
        raise ValueError("cycled training record has no preceding base training source")
    expected_cycle = [
        noncycled_sources[index % len(noncycled_sources)].base_task_id
        for index in range(len(cycled_sources))
    ]
    if [source.base_task_id for source in cycled_sources] != expected_cycle:
        raise ValueError("cycled training order does not restart at the first train sample")
    return tuple(sources)


def materialize_hotpotqa_qa_memories(
    sources: Sequence[HotpotQATrainQASource],
    paraphrases: Sequence[Mapping[str, object]],
) -> tuple[HotpotQAQAMemory, ...]:
    """Join injected offline paraphrases to training sources, fail closed."""

    source_by_id = {source.source_train_task_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("training source IDs must be unique")
    paraphrase_by_id: dict[str, Mapping[str, object]] = {}
    for raw in paraphrases:
        mapping = _mapping(raw, "paraphrase materialization")
        _reject_private_keys(mapping, path="paraphrase")
        _expect_exact_keys(mapping, _PARAPHRASE_FIELDS, "paraphrase materialization")
        source_id = _required_text(
            mapping["source_train_task_id"], "source_train_task_id"
        )
        if source_id in paraphrase_by_id:
            raise ValueError(f"duplicate paraphrase for {source_id}")
        if mapping["semantic_preservation_attested"] is not True:
            raise ValueError("semantic-preservation attestation must be true")
        paraphrase_by_id[source_id] = mapping
    missing = sorted(set(source_by_id) - set(paraphrase_by_id))
    extra = sorted(set(paraphrase_by_id) - set(source_by_id))
    if missing or extra:
        raise ValueError(
            f"paraphrase/source coverage differs: missing={missing}, extra={extra}"
        )

    memories: list[HotpotQAQAMemory] = []
    for index, source in enumerate(sources):
        item = paraphrase_by_id[source.source_train_task_id]
        question = _required_text(item["paraphrase_question"], "paraphrase_question")
        answer_statement = _required_text(
            item["paraphrase_answer_statement"], "paraphrase_answer_statement"
        )
        provenance = _required_text(
            item["paraphrase_provenance"], "paraphrase_provenance"
        )
        version = _required_text(item["paraphrase_version"], "paraphrase_version")
        if _normalized_text(question) == _normalized_text(source.question):
            raise ValueError("paraphrase_question is identical to the training question")
        canonical = _normalized_text(source.canonical_answer)
        exact_span_required = version.endswith("-v2")
        if (
            source.canonical_answer not in answer_statement
            if exact_span_required
            else canonical not in _normalized_text(answer_statement)
        ):
            raise ValueError(
                "paraphrase answer statement does not preserve canonical answer span"
            )
        if (
            exact_span_required
            and source.canonical_answer.casefold() not in source.question.casefold()
            and source.canonical_answer.casefold() in question.casefold()
        ):
            raise ValueError(
                "paraphrase_question introduced the canonical answer"
            )
        if _normalized_text(answer_statement) == canonical:
            raise ValueError("paraphrase answer must be a declarative answer statement")
        memories.append(
            HotpotQAQAMemory(
                memory_id=f"hotpotqa-qa-memory-{index:06d}",
                source_train_task_id=source.source_train_task_id,
                base_task_id=source.base_task_id,
                cycled=source.cycled,
                paraphrase_question=question,
                paraphrase_answer_statement=answer_statement,
                canonical_answer=source.canonical_answer,
                paraphrase_provenance=provenance,
                paraphrase_version=version,
            )
        )
    return tuple(memories)


def load_paraphrase_materialization(path: Path) -> tuple[Mapping[str, object], ...]:
    """Read an offline JSONL materialization without invoking a model/API."""

    values: list[Mapping[str, object]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = _mapping(json.loads(line), "paraphrase materialization")
                _reject_private_keys(value, path="paraphrase")
                _expect_exact_keys(value, _PARAPHRASE_FIELDS, "paraphrase materialization")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            values.append(value)
    return tuple(values)


def build_hotpotqa_qa_memory_index(
    *,
    index_dir: Path,
    train_jsonl: Path,
    validation_task_ids: Sequence[str],
    paraphrases: Sequence[Mapping[str, object]],
    embedding_model_path: str,
    embedding_model_id: str,
    embedding_device: str,
    frozen_top_k: int,
    index_version: int = 1,
    expected_train_count: int = 512,
    expected_validation_count: int = 128,
) -> HotpotQAQAMemoryIndexManifest:
    """Build a versioned dense index from frozen train QA paraphrases only."""

    if frozen_top_k < 1:
        raise ValueError("frozen_top_k must be positive")
    if index_version not in {1, 2}:
        raise ValueError("index_version must be 1 or 2")
    sources = load_hotpotqa_train_qa_sources(
        train_jsonl,
        validation_task_ids=validation_task_ids,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    memories = materialize_hotpotqa_qa_memories(sources, paraphrases)
    model = _load_sentence_transformer(embedding_model_path, embedding_device)
    vectors = _encode(model, [memory.document_text for memory in memories])

    index_dir = index_dir.expanduser().resolve()
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError("QA-memory index directory must be empty")
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
    provenances = tuple(sorted({memory.paraphrase_provenance for memory in memories}))
    cycled_count = sum(memory.cycled for memory in memories)
    unique_source_count = len({memory.base_task_id for memory in memories})
    manifest = HotpotQAQAMemoryIndexManifest(
        schema_version=f"flowsteer.hotpotqa.qa_memory_index.v{index_version}",
        index_id=(
            "hotpotqa-train-qa-memory-"
            f"d{vectors.shape[1]}-n{len(memories)}-topk{frozen_top_k}-v{index_version}"
        ),
        corpus_version=f"flowsteer.hotpotqa.train_qa_memory.v{index_version}",
        source="HotpotQA aligned frozen train",
        source_split="train",
        embedding_model=embedding_model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimension=int(vectors.shape[1]),
        normalized=True,
        similarity="cosine",
        frozen_top_k=frozen_top_k,
        train_record_count=len(memories),
        unique_source_count=unique_source_count,
        cycled_record_count=cycled_count,
        paraphrase_count=len(memories),
        heldout_validation_count=len({_base_task_id(item) for item in validation_task_ids}),
        validation_overlap_count=0,
        paraphrase_versions=versions,
        paraphrase_provenances=provenances,
        document_template=QA_MEMORY_DOCUMENT_TEMPLATE,
        source_train_path=str(train_jsonl.expanduser().resolve()),
        memories_path=memories_path.name,
        embeddings_path=embeddings_path.name,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_value(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class HotpotQAQAMemoryIndex:
    """Read-only global train QA-memory ``search/read`` index."""

    def __init__(
        self,
        *,
        manifest: HotpotQAQAMemoryIndexManifest,
        memories: Sequence[HotpotQAQAMemory],
        embeddings: np.ndarray,
        model: object,
    ) -> None:
        if embeddings.shape != (len(memories), manifest.embedding_dimension):
            raise ValueError("embedding matrix does not match QA-memory manifest")
        if len(memories) != manifest.train_record_count:
            raise ValueError("memory count does not match QA-memory manifest")
        memory_index = {memory.memory_id: index for index, memory in enumerate(memories)}
        if len(memory_index) != len(memories):
            raise ValueError("QA-memory IDs are not unique")
        self.manifest = manifest
        self._memories = tuple(memories)
        self._memory_index = MappingProxyType(memory_index)
        self._embeddings = embeddings
        self._model = model
        self._encode_lock = Lock()

    @classmethod
    def open(
        cls,
        index_dir: Path,
        *,
        embedding_model_path: str | None = None,
        embedding_device: str = "cpu",
    ) -> "HotpotQAQAMemoryIndex":
        index_dir = index_dir.expanduser().resolve()
        manifest = HotpotQAQAMemoryIndexManifest.from_value(
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
            manifest=manifest,
            memories=memories,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model=model,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        query = _required_text(query, "retrieval query")
        with self._encode_lock:
            return _encode(self._model, [query], batch_size=1)[0]

    async def search(
        self, query: str, k: int
    ) -> tuple[HotpotQAQAMemorySearchHit, ...]:
        if k != self.manifest.frozen_top_k:
            raise ValueError("search k differs from the frozen QA-memory top-k")
        query_vector = await asyncio.to_thread(self._encode_query, query)
        scored = [
            (float(np.dot(self._embeddings[index], query_vector)), memory.memory_id)
            for index, memory in enumerate(self._memories)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[HotpotQAQAMemorySearchHit] = []
        for rank, (similarity, memory_id) in enumerate(scored[:k], start=1):
            memory = self._memories[self._memory_index[memory_id]]
            hits.append(
                HotpotQAQAMemorySearchHit(
                    memory_id=memory.memory_id,
                    source_train_task_id=memory.source_train_task_id,
                    paraphrase_question=memory.paraphrase_question,
                    paraphrase_answer_statement=memory.paraphrase_answer_statement,
                    similarity=similarity,
                    rank=rank,
                )
            )
        return tuple(hits)

    def read(self, memory_id: str) -> HotpotQAQAMemory:
        try:
            return self._memories[self._memory_index[memory_id]]
        except KeyError as exc:
            raise KeyError("memory_id is absent from the QA-memory index") from exc


__all__ = [
    "QA_MEMORY_CORPUS_VERSION",
    "QA_MEMORY_CORPUS_VERSIONS",
    "QA_MEMORY_DOCUMENT_TEMPLATE",
    "QA_MEMORY_SCHEMA_VERSION",
    "QA_MEMORY_SCHEMA_VERSIONS",
    "HotpotQAQAMemory",
    "HotpotQAQAMemoryIndex",
    "HotpotQAQAMemoryIndexManifest",
    "HotpotQAQAMemorySearchHit",
    "HotpotQATrainQASource",
    "build_hotpotqa_qa_memory_index",
    "load_hotpotqa_train_qa_sources",
    "load_paraphrase_materialization",
    "materialize_hotpotqa_qa_memories",
]
