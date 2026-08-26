"""Versioned task-scoped embedding index for HotpotQA public contexts.

The data contract follows SkillFlow's immutable ``open/search/read`` retrieval
boundary, while replacing its lexical backend with normalized dense embeddings.
Only ``id`` and ``context`` are read from the source parquet during index
construction.  Answers, supporting-fact labels, and evaluator state are never
materialized in the corpus or index artifacts.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np


INDEX_SCHEMA_VERSION = "flowsteer.hotpotqa.embedding_index.v1"
CORPUS_SCHEMA_VERSION = "flowsteer.hotpotqa.public_context.v1"


@dataclass(frozen=True, slots=True)
class HotpotQAPassage:
    passage_id: str
    document_id: str
    title: str
    text: str

    def to_value(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HotpotQASearchHit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    similarity: float
    rank: int


@dataclass(frozen=True, slots=True)
class HotpotQAEmbeddingIndexManifest:
    schema_version: str
    index_id: str
    corpus_version: str
    source: str
    source_split: str
    project_splits: tuple[str, ...]
    embedding_model: str
    embedding_model_path: str
    embedding_dimension: int
    normalized: bool
    similarity: str
    frozen_top_k: int
    task_count: int
    document_count: int
    passage_count: int
    passage_occurrence_count: int
    duplicate_occurrence_count: int
    source_files: tuple[str, ...]
    passages_path: str
    scopes_path: str
    embeddings_path: str

    def __post_init__(self) -> None:
        if self.schema_version != INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported HotpotQA embedding index schema")
        if not self.index_id or not self.corpus_version:
            raise ValueError("index identity fields must be non-empty")
        if self.embedding_dimension < 1 or self.frozen_top_k < 1:
            raise ValueError("embedding dimension and frozen top-k must be positive")
        if not self.normalized or self.similarity != "cosine":
            raise ValueError("HotpotQA embedding index requires normalized cosine")
        if self.document_count != self.passage_count:
            raise ValueError("v1 stores one passage per public context document")
        if self.duplicate_occurrence_count != (
            self.passage_occurrence_count - self.passage_count
        ):
            raise ValueError("manifest duplicate counts are inconsistent")

    def to_value(self) -> dict[str, object]:
        value = asdict(self)
        value["project_splits"] = list(self.project_splits)
        value["source_files"] = list(self.source_files)
        return value

    @classmethod
    def from_value(cls, value: object) -> "HotpotQAEmbeddingIndexManifest":
        if not isinstance(value, Mapping):
            raise TypeError("index manifest must be an object")
        fields = {
            field_name: value[field_name]
            for field_name in cls.__dataclass_fields__
        }
        fields["project_splits"] = tuple(str(item) for item in fields["project_splits"])
        fields["source_files"] = tuple(str(item) for item in fields["source_files"])
        return cls(**fields)  # type: ignore[arg-type]


def _task_native_id(task_id: str) -> str:
    prefix = "hotpotqa:"
    return task_id[len(prefix) :] if task_id.startswith(prefix) else task_id


def load_public_contexts(
    parquet_paths: Sequence[Path],
    task_ids: Sequence[str],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Read only public ``id`` and ``context`` columns for requested tasks."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("pandas/pyarrow are required to read HotpotQA parquet") from exc

    by_native = {_task_native_id(task_id): task_id for task_id in task_ids}
    contexts: dict[str, tuple[tuple[str, str], ...]] = {}
    for parquet_path in sorted(parquet_paths):
        frame = pd.read_parquet(parquet_path, columns=["id", "context"])
        for row in frame.itertuples(index=False):
            task_id = by_native.get(str(row.id))
            if task_id is None or task_id in contexts:
                continue
            raw_context = row.context
            if not isinstance(raw_context, Mapping):
                raise ValueError(f"HotpotQA context is not a mapping for {task_id}")
            titles = raw_context.get("title")
            sentences = raw_context.get("sentences")
            if titles is None or sentences is None or len(titles) != len(sentences):
                raise ValueError(f"HotpotQA context fields are invalid for {task_id}")
            passages: list[tuple[str, str]] = []
            for raw_title, raw_sentences in zip(titles, sentences, strict=True):
                title = str(raw_title).strip()
                text = "".join(str(sentence) for sentence in raw_sentences).strip()
                if not title or not text:
                    raise ValueError(f"HotpotQA contains an empty public passage for {task_id}")
                passages.append((title, text))
            contexts[task_id] = tuple(passages)
        if len(contexts) == len(by_native):
            break
    missing = sorted(set(task_ids) - set(contexts))
    if missing:
        raise ValueError(f"source parquet is missing {len(missing)} requested tasks")
    return contexts


def _load_sentence_transformer(model_path: str, device: str) -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("sentence-transformers is required for dense retrieval") from exc
    return SentenceTransformer(model_path, device=device)


def _encode(
    model: object,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    vectors = model.encode(  # type: ignore[attr-defined]
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts) or array.shape[1] < 1:
        raise ValueError("embedding model returned an incompatible matrix")
    if not np.isfinite(array).all():
        raise ValueError("embedding matrix contains non-finite values")
    return array


def build_hotpotqa_embedding_index(
    *,
    index_dir: Path,
    parquet_paths: Sequence[Path],
    task_splits: Mapping[str, str],
    embedding_model_path: str,
    embedding_model_id: str,
    embedding_device: str,
    frozen_top_k: int,
) -> HotpotQAEmbeddingIndexManifest:
    """Build a deterministic, deduplicated public-context dense index."""

    if not task_splits or frozen_top_k < 1:
        raise ValueError("task_splits and frozen_top_k must be non-empty")
    contexts = load_public_contexts(parquet_paths, tuple(sorted(task_splits)))
    occurrences = sum(len(items) for items in contexts.values())
    unique_keys = sorted(
        {item for passages in contexts.values() for item in passages},
        key=lambda item: (item[0].casefold(), item[0], item[1]),
    )
    passages = tuple(
        HotpotQAPassage(
            passage_id=f"hotpotqa-passage-{index:06d}",
            document_id=f"hotpotqa-document-{index:06d}",
            title=title,
            text=text,
        )
        for index, (title, text) in enumerate(unique_keys)
    )
    passage_id_by_key = {
        (passage.title, passage.text): passage.passage_id for passage in passages
    }
    scopes = {
        task_id: [passage_id_by_key[item] for item in contexts[task_id]]
        for task_id in sorted(contexts)
    }
    model = _load_sentence_transformer(embedding_model_path, embedding_device)
    vectors = _encode(
        model,
        [f"{passage.title}\n{passage.text}" for passage in passages],
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    passages_path = index_dir / "passages.jsonl"
    scopes_path = index_dir / "task_scopes.json"
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"
    passages_path.write_text(
        "".join(
            json.dumps(passage.to_value(), ensure_ascii=False, sort_keys=True) + "\n"
            for passage in passages
        ),
        encoding="utf-8",
    )
    scopes_path.write_text(
        json.dumps(scopes, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(embeddings_path, vectors, allow_pickle=False)
    manifest = HotpotQAEmbeddingIndexManifest(
        schema_version=INDEX_SCHEMA_VERSION,
        index_id=(
            "hotpotqa-public-context-bge-base-en-v1.5-"
            f"d{vectors.shape[1]}-topk{frozen_top_k}-v1"
        ),
        corpus_version=CORPUS_SCHEMA_VERSION,
        source="HotpotQA_HF/distractor",
        source_split="train",
        project_splits=tuple(sorted(set(task_splits.values()))),
        embedding_model=embedding_model_id,
        embedding_model_path=embedding_model_path,
        embedding_dimension=int(vectors.shape[1]),
        normalized=True,
        similarity="cosine",
        frozen_top_k=frozen_top_k,
        task_count=len(scopes),
        document_count=len(passages),
        passage_count=len(passages),
        passage_occurrence_count=occurrences,
        duplicate_occurrence_count=occurrences - len(passages),
        source_files=tuple(str(path) for path in sorted(parquet_paths)),
        passages_path=passages_path.name,
        scopes_path=scopes_path.name,
        embeddings_path=embeddings_path.name,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_value(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class HotpotQAEmbeddingIndex:
    """Read-only public ``search/read`` index scoped by HotpotQA task ID."""

    def __init__(
        self,
        *,
        manifest: HotpotQAEmbeddingIndexManifest,
        passages: Sequence[HotpotQAPassage],
        scopes: Mapping[str, Sequence[str]],
        embeddings: np.ndarray,
        model: object,
    ) -> None:
        if embeddings.shape != (len(passages), manifest.embedding_dimension):
            raise ValueError("embedding matrix does not match the index manifest")
        self.manifest = manifest
        self._passages = tuple(passages)
        self._passage_index = MappingProxyType(
            {passage.passage_id: index for index, passage in enumerate(passages)}
        )
        self._scopes = MappingProxyType(
            {task_id: tuple(values) for task_id, values in scopes.items()}
        )
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
    ) -> "HotpotQAEmbeddingIndex":
        index_dir = index_dir.expanduser().resolve()
        manifest = HotpotQAEmbeddingIndexManifest.from_value(
            json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        )
        passages = tuple(
            HotpotQAPassage(**json.loads(line))
            for line in (index_dir / manifest.passages_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        raw_scopes = json.loads(
            (index_dir / manifest.scopes_path).read_text(encoding="utf-8")
        )
        if not isinstance(raw_scopes, Mapping):
            raise TypeError("task scopes must be an object")
        scopes = {
            str(task_id): tuple(str(item) for item in values)
            for task_id, values in raw_scopes.items()
        }
        known = {passage.passage_id for passage in passages}
        if any(passage_id not in known for values in scopes.values() for passage_id in values):
            raise ValueError("task scope references an unknown passage")
        embeddings = np.load(index_dir / manifest.embeddings_path, allow_pickle=False)
        model = _load_sentence_transformer(
            embedding_model_path or manifest.embedding_model_path,
            embedding_device,
        )
        return cls(
            manifest=manifest,
            passages=passages,
            scopes=scopes,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model=model,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("retrieval query must be non-empty text")
        with self._encode_lock:
            return _encode(self._model, [query.strip()], batch_size=1)[0]

    async def search(
        self,
        task_id: str,
        query: str,
        k: int,
    ) -> tuple[HotpotQASearchHit, ...]:
        if k != self.manifest.frozen_top_k:
            raise ValueError("search k differs from the frozen index top-k")
        try:
            passage_ids = self._scopes[task_id]
        except KeyError as exc:
            raise KeyError("task is absent from the embedding index") from exc
        query_vector = await asyncio.to_thread(self._encode_query, query)
        scored = [
            (
                float(np.dot(self._embeddings[self._passage_index[passage_id]], query_vector)),
                passage_id,
            )
            for passage_id in passage_ids
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[HotpotQASearchHit] = []
        for rank, (similarity, passage_id) in enumerate(scored[:k], start=1):
            passage = self._passages[self._passage_index[passage_id]]
            snippet = " ".join(passage.text.split())[:400]
            hits.append(
                HotpotQASearchHit(
                    passage_id=passage.passage_id,
                    document_id=passage.document_id,
                    title=passage.title,
                    snippet=snippet,
                    similarity=similarity,
                    rank=rank,
                )
            )
        return tuple(hits)

    def read(self, task_id: str, passage_id: str) -> HotpotQAPassage:
        try:
            scope = self._scopes[task_id]
        except KeyError as exc:
            raise KeyError("task is absent from the embedding index") from exc
        if passage_id not in scope:
            raise ValueError("passage is outside the current task's public context")
        return self._passages[self._passage_index[passage_id]]


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "INDEX_SCHEMA_VERSION",
    "HotpotQAEmbeddingIndex",
    "HotpotQAEmbeddingIndexManifest",
    "HotpotQAPassage",
    "HotpotQASearchHit",
    "build_hotpotqa_embedding_index",
    "load_public_contexts",
]
