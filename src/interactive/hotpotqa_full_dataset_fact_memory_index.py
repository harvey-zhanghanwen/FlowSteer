"""Global HotpotQA declarative-fact embedding index.

The record and ``search``/``read`` boundary follow SkillFlow's
``DocumentPassage`` retrieval contract.  FlowSteer's normalized BGE encoder
and deterministic cosine ranking are reused.  Raw questions, canonical
answers, paraphrased questions, evaluator metadata, and generation receipts
remain outside this runtime index; only a self-contained declarative fact is
embedded and exposed to worker Agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import yaml

from scripts.prepare_agentgraph_datasets import _hotpot_records, _path

from .hotpotqa_embedding_index import _encode, _load_sentence_transformer
from .hotpotqa_qa_memory_index import HotpotQATrainQASource
from .task_dataset import qa_question_scope


FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION = (
    "flowsteer.hotpotqa.full_dataset_fact_memory_index.v1"
)
FULL_DATASET_FACT_MEMORY_CORPUS_VERSION = (
    "flowsteer.hotpotqa.native_train_validation_declarative_facts.v1"
)
FULL_DATASET_FACT_DOCUMENT_TEMPLATE = "{fact_text}"
FULL_DATASET_FACT_DOCUMENT_FORMAT = "declarative_fact_only"
FULL_DATASET_FACT_INDEXED_TEXT_FIELD = "fact_text"
FULL_DATASET_EVALUATION_SCOPE = "in_database_transductive"

_MATERIALIZATION_FIELDS = frozenset(
    {
        "source_train_task_id",
        "paraphrase_question",
        "fact_statement",
        "paraphrase_provenance",
        "paraphrase_version",
        "semantic_preservation_attested",
    }
)
_FACT_FIELDS = frozenset({"memory_id", "fact_text"})


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


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
    """Index-external Q-A provenance projected from native HotpotQA splits."""

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
    """Load raw Q-A only as index-external generation/evaluation provenance."""

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
        projected[str(split)].append(
            HotpotQATrainQASource(
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
        )
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
class HotpotQADeclarativeFact:
    """SkillFlow-style public passage containing only one declarative fact."""

    memory_id: str
    fact_text: str

    def __post_init__(self) -> None:
        _required_text(self.memory_id, "memory_id")
        fact = _required_text(self.fact_text, "fact_text")
        lowered = fact.casefold()
        if lowered.startswith("question:") or lowered.startswith("answer:"):
            raise ValueError("fact_text cannot use a Question/Answer label")
        if "\nquestion:" in lowered or "\nanswer:" in lowered:
            raise ValueError("fact_text cannot contain a Question/Answer wire")

    def to_value(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: object) -> "HotpotQADeclarativeFact":
        mapping = _mapping(value, "declarative fact")
        if set(mapping) != _FACT_FIELDS:
            raise ValueError("declarative fact fields differ from the fact-only wire")
        return cls(
            memory_id=_required_text(mapping["memory_id"], "memory_id"),
            fact_text=_required_text(mapping["fact_text"], "fact_text"),
        )


@dataclass(frozen=True, slots=True)
class HotpotQADeclarativeFactSearchHit:
    memory_id: str
    fact_snippet: str
    similarity: float
    rank: int


def materialize_hotpotqa_declarative_facts(
    sources: Sequence[HotpotQATrainQASource],
    paraphrases: Sequence[Mapping[str, object]],
) -> tuple[HotpotQADeclarativeFact, ...]:
    """Admit verified sidecar records while projecting only fact text."""

    if len(sources) != len(paraphrases):
        raise ValueError("every source must have exactly one fact materialization")
    facts: list[HotpotQADeclarativeFact] = []
    for index, (source, raw_value) in enumerate(zip(sources, paraphrases)):
        value = _mapping(raw_value, "fact materialization")
        if set(value) != _MATERIALIZATION_FIELDS:
            raise ValueError("fact materialization fields differ")
        if value["source_train_task_id"] != source.source_train_task_id:
            raise ValueError("fact materialization source order or identity differs")
        if value["semantic_preservation_attested"] is not True:
            raise ValueError("fact materialization lacks semantic verification")
        paraphrase = _required_text(
            value["paraphrase_question"], "paraphrase_question"
        )
        if _normalized_text(paraphrase) == _normalized_text(source.question):
            raise ValueError("paraphrase_question is identical to the source question")
        fact_text = _required_text(value["fact_statement"], "fact_statement")
        _required_text(value["paraphrase_provenance"], "paraphrase_provenance")
        _required_text(value["paraphrase_version"], "paraphrase_version")
        facts.append(
            HotpotQADeclarativeFact(
                memory_id=f"hotpotqa-fact-{index:06d}",
                fact_text=fact_text,
            )
        )
    return tuple(facts)


@dataclass(frozen=True, slots=True)
class HotpotQAFullDatasetFactMemoryIndexManifest:
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
    question_rewrite_count: int
    fact_count: int
    semantic_rewrite_coverage: float
    frozen_evaluation_count: int
    evaluation_overlap_count: int
    contains_evaluation_source_facts: bool
    contains_raw_questions: bool
    contains_raw_answers: bool
    evaluation_scope: str
    official_heldout_eligible: bool
    paraphrase_versions: tuple[str, ...]
    paraphrase_provenances: tuple[str, ...]
    document_template: str
    document_format: str
    indexed_text_field: str
    source_dataset_catalog_path: str
    source_train_path: str
    source_validation_path: str
    facts_path: str
    embeddings_path: str

    def __post_init__(self) -> None:
        if self.schema_version != FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported full-dataset fact-memory index schema")
        if self.corpus_version != FULL_DATASET_FACT_MEMORY_CORPUS_VERSION:
            raise ValueError("unsupported full-dataset fact-memory corpus schema")
        if self.source_splits != ("train", "validation"):
            raise ValueError("full-dataset fact-memory source splits differ")
        if self.embedding_dimension < 1 or self.frozen_top_k < 1:
            raise ValueError("embedding dimension and top-k must be positive")
        if not self.normalized or self.similarity != "cosine":
            raise ValueError("full-dataset fact-memory requires normalized cosine")
        if self.source_record_count != (
            self.source_train_count + self.source_validation_count
        ):
            raise ValueError("full-dataset source counts are inconsistent")
        if self.unique_source_count != self.source_record_count:
            raise ValueError("every native HotpotQA source ID must be unique")
        if self.cycled_record_count != 0:
            raise ValueError("native full-dataset sources cannot be cycled")
        if self.question_rewrite_count != self.source_record_count:
            raise ValueError("every source question must be rewritten")
        if self.fact_count != self.source_record_count:
            raise ValueError("every source must have one declarative fact")
        if self.semantic_rewrite_coverage != 1.0:
            raise ValueError("semantic rewrite coverage must be exactly 100%")
        if self.frozen_evaluation_count < 1:
            raise ValueError("frozen evaluation identity count must be positive")
        if self.evaluation_overlap_count != self.frozen_evaluation_count:
            raise ValueError("every frozen evaluation source fact must be present")
        if self.contains_evaluation_source_facts is not True:
            raise ValueError("transductive source facts must be declared")
        if self.contains_raw_questions or self.contains_raw_answers:
            raise ValueError("raw Q-A cannot enter the fact index")
        if self.evaluation_scope != FULL_DATASET_EVALUATION_SCOPE:
            raise ValueError("full-dataset evaluation scope differs")
        if self.official_heldout_eligible is not False:
            raise ValueError("in-database fact retrieval is not held-out eligible")
        if not self.paraphrase_versions or not self.paraphrase_provenances:
            raise ValueError("paraphrase version and provenance are required")
        if self.document_template != FULL_DATASET_FACT_DOCUMENT_TEMPLATE:
            raise ValueError("fact index document template differs")
        if self.document_format != FULL_DATASET_FACT_DOCUMENT_FORMAT:
            raise ValueError("fact index document format differs")
        if self.indexed_text_field != FULL_DATASET_FACT_INDEXED_TEXT_FIELD:
            raise ValueError("fact index text field differs")

    @property
    def train_record_count(self) -> int:
        """Compatibility count consumed by the shared retrieval adapter."""

        return self.fact_count

    def to_value(self) -> dict[str, object]:
        value = asdict(self)
        value["source_splits"] = list(self.source_splits)
        value["paraphrase_versions"] = list(self.paraphrase_versions)
        value["paraphrase_provenances"] = list(self.paraphrase_provenances)
        return value

    @classmethod
    def from_value(
        cls, value: object
    ) -> "HotpotQAFullDatasetFactMemoryIndexManifest":
        mapping = _mapping(value, "full-dataset fact-memory manifest")
        expected = frozenset(cls.__dataclass_fields__)
        if set(mapping) != expected:
            raise ValueError("full-dataset fact-memory manifest fields differ")
        fields = {name: mapping[name] for name in expected}
        fields["source_splits"] = tuple(str(item) for item in fields["source_splits"])
        fields["paraphrase_versions"] = tuple(
            str(item) for item in fields["paraphrase_versions"]
        )
        fields["paraphrase_provenances"] = tuple(
            str(item) for item in fields["paraphrase_provenances"]
        )
        return cls(**fields)  # type: ignore[arg-type]


def _native_source_paths(dataset_catalog_path: Path) -> tuple[str, str]:
    config = _hotpot_config(dataset_catalog_path)
    files = _mapping(config.get("files"), "HotpotQA files")
    base = _path(str(config["path"]))
    return str(base / str(files["train"])), str(base / str(files["validation"]))


def build_hotpotqa_full_dataset_fact_memory_index(
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
) -> HotpotQAFullDatasetFactMemoryIndexManifest:
    """Build one global normalized-cosine index over declarative facts only."""

    if frozen_top_k < 1:
        raise ValueError("frozen_top_k must be positive")
    sources = load_hotpotqa_full_dataset_qa_sources(
        dataset_catalog_path=dataset_catalog_path,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
    )
    source_ids = {source.source_train_task_id for source in sources.combined}
    evaluation_ids = {
        _required_text(item, "evaluation task ID")
        for item in frozen_evaluation_task_ids
    }
    if len(evaluation_ids) != len(frozen_evaluation_task_ids):
        raise ValueError("frozen evaluation task IDs are not unique")
    overlap_count = len(source_ids & evaluation_ids)
    if overlap_count != len(evaluation_ids):
        raise ValueError("some frozen evaluation sources are absent")

    facts = materialize_hotpotqa_declarative_facts(sources.combined, paraphrases)
    model = _load_sentence_transformer(embedding_model_path, embedding_device)
    # Required boundary: only the self-contained fact text is vectorized.
    vectors = _encode(model, [fact.fact_text for fact in facts])

    index_dir = index_dir.expanduser().resolve()
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError("full-dataset fact-memory index directory must be empty")
    index_dir.mkdir(parents=True, exist_ok=True)
    facts_path = index_dir / "facts.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"
    facts_path.write_text(
        "".join(
            json.dumps(fact.to_value(), ensure_ascii=False, sort_keys=True) + "\n"
            for fact in facts
        ),
        encoding="utf-8",
    )
    np.save(embeddings_path, vectors, allow_pickle=False)
    source_train_path, source_validation_path = _native_source_paths(
        dataset_catalog_path
    )
    paraphrase_versions = tuple(
        sorted({_required_text(item["paraphrase_version"], "paraphrase_version") for item in paraphrases})
    )
    paraphrase_provenances = tuple(
        sorted({_required_text(item["paraphrase_provenance"], "paraphrase_provenance") for item in paraphrases})
    )
    manifest = HotpotQAFullDatasetFactMemoryIndexManifest(
        schema_version=FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION,
        index_id=(
            "hotpotqa-full-dataset-fact-memory-"
            f"d{vectors.shape[1]}-n{len(facts)}-topk{frozen_top_k}-v1"
        ),
        corpus_version=FULL_DATASET_FACT_MEMORY_CORPUS_VERSION,
        source="HotpotQA generated self-contained declarative facts",
        source_splits=("train", "validation"),
        embedding_model=embedding_model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimension=int(vectors.shape[1]),
        normalized=True,
        similarity="cosine",
        frozen_top_k=frozen_top_k,
        source_record_count=len(facts),
        source_train_count=len(sources.train),
        source_validation_count=len(sources.validation),
        unique_source_count=len(sources.combined),
        cycled_record_count=0,
        question_rewrite_count=len(facts),
        fact_count=len(facts),
        semantic_rewrite_coverage=1.0,
        frozen_evaluation_count=len(evaluation_ids),
        evaluation_overlap_count=overlap_count,
        contains_evaluation_source_facts=True,
        contains_raw_questions=False,
        contains_raw_answers=False,
        evaluation_scope=FULL_DATASET_EVALUATION_SCOPE,
        official_heldout_eligible=False,
        paraphrase_versions=paraphrase_versions,
        paraphrase_provenances=paraphrase_provenances,
        document_template=FULL_DATASET_FACT_DOCUMENT_TEMPLATE,
        document_format=FULL_DATASET_FACT_DOCUMENT_FORMAT,
        indexed_text_field=FULL_DATASET_FACT_INDEXED_TEXT_FIELD,
        source_dataset_catalog_path=str(dataset_catalog_path.expanduser().resolve()),
        source_train_path=source_train_path,
        source_validation_path=source_validation_path,
        facts_path=facts_path.name,
        embeddings_path=embeddings_path.name,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_value(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class HotpotQAFullDatasetFactMemoryIndex:
    """Read-only global embedding index with SkillFlow-style search/read."""

    def __init__(
        self,
        *,
        manifest: HotpotQAFullDatasetFactMemoryIndexManifest,
        facts: Sequence[HotpotQADeclarativeFact],
        embeddings: np.ndarray,
        model: object,
    ) -> None:
        if embeddings.shape != (len(facts), manifest.embedding_dimension):
            raise ValueError("embedding matrix does not match fact-memory manifest")
        if len(facts) != manifest.fact_count:
            raise ValueError("fact count does not match fact-memory manifest")
        memory_index = {fact.memory_id: index for index, fact in enumerate(facts)}
        if len(memory_index) != len(facts):
            raise ValueError("fact memory IDs are not unique")
        self.manifest = manifest
        self._facts = tuple(facts)
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
    ) -> "HotpotQAFullDatasetFactMemoryIndex":
        index_dir = index_dir.expanduser().resolve()
        manifest = HotpotQAFullDatasetFactMemoryIndexManifest.from_value(
            json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        )
        facts = tuple(
            HotpotQADeclarativeFact.from_value(json.loads(line))
            for line in (index_dir / manifest.facts_path)
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
            facts=facts,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model=model,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        query = _required_text(query, "retrieval query")
        with self._encode_lock:
            return _encode(self._model, [query], batch_size=1)[0]

    async def search(
        self, query: str, k: int
    ) -> tuple[HotpotQADeclarativeFactSearchHit, ...]:
        if k != self.manifest.frozen_top_k:
            raise ValueError("search k differs from the frozen fact-memory top-k")
        # The shared runtime invokes Tool calls under an async boundary, but
        # sentence-transformer query encoding itself is synchronous.  Keep it
        # on the calling thread so the per-task Tool lifecycle owns no orphaned
        # default-executor thread at shutdown.
        query_vector = self._encode_query(query)
        scored = [
            (float(np.dot(self._embeddings[index], query_vector)), fact.memory_id)
            for index, fact in enumerate(self._facts)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[HotpotQADeclarativeFactSearchHit] = []
        for rank, (similarity, memory_id) in enumerate(scored[:k], start=1):
            fact = self._facts[self._memory_index[memory_id]]
            snippet = fact.fact_text
            if len(snippet) > 320:
                snippet = f"{snippet[:320]}…"
            hits.append(
                HotpotQADeclarativeFactSearchHit(
                    memory_id=fact.memory_id,
                    fact_snippet=snippet,
                    similarity=similarity,
                    rank=rank,
                )
            )
        return tuple(hits)

    def read(self, memory_id: str) -> HotpotQADeclarativeFact:
        try:
            return self._facts[self._memory_index[memory_id]]
        except KeyError as exc:
            raise KeyError("memory_id is absent from the fact index") from exc


__all__ = [
    "FULL_DATASET_EVALUATION_SCOPE",
    "FULL_DATASET_FACT_DOCUMENT_FORMAT",
    "FULL_DATASET_FACT_DOCUMENT_TEMPLATE",
    "FULL_DATASET_FACT_INDEXED_TEXT_FIELD",
    "FULL_DATASET_FACT_MEMORY_CORPUS_VERSION",
    "FULL_DATASET_FACT_MEMORY_SCHEMA_VERSION",
    "HotpotQADeclarativeFact",
    "HotpotQADeclarativeFactSearchHit",
    "HotpotQAFullDatasetFactMemoryIndex",
    "HotpotQAFullDatasetFactMemoryIndexManifest",
    "HotpotQAFullDatasetQASources",
    "build_hotpotqa_full_dataset_fact_memory_index",
    "load_hotpotqa_full_dataset_qa_sources",
    "materialize_hotpotqa_declarative_facts",
]
