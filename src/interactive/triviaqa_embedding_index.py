"""Versioned dense retrieval index over TriviaQA public search evidence.

The index deliberately mirrors SkillFlow's public ``DocumentPassage`` and
``SearchHit`` boundary while replacing only the lexical ranker.  Corpus rows
are built from the two admitted source fields, ``SearchResults.Title`` and
``SearchResults.Description``.  Questions, reference answers, aliases,
evaluator payloads, and aligned ground-truth fields are never serialized or
sent to the embedding model.

SentenceTransformer encoding follows ``src/skills/workspace.py``: BGE runs on
CPU, ``normalize_embeddings=True`` is requested for passages and queries, and
similarity is the dot product of normalized float32 vectors.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import tempfile
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, TextIO

import numpy as np


MANIFEST_SCHEMA_VERSION = "flowsteer.triviaqa.embedding_index.manifest.v1"
INDEX_FORMAT = "flowsteer.triviaqa.embedding-index.v1"
RETRIEVAL_BACKEND = "sentence-transformers-bge-normalized-dot-product"
CORPUS_NAME = "triviaqa-unfiltered-web-public-search-results"
SOURCE_DATASET = "TriviaQA"
SOURCE_CONFIGURATION = "unfiltered"
SOURCE_SPLIT = "unfiltered-web-train"
NORMALIZATION = "l2"
SIMILARITY = "dot_product"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MANIFEST_FILENAME = "manifest.json"
PASSAGES_FILENAME = "passages.jsonl"
EMBEDDINGS_FILENAME = "embeddings.npy"

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "SKILLEV_EMBEDDING_MODEL_PATH",
    "BAAI/bge-base-en-v1.5",
)

_TASK_ID_PREFIX = "triviaqa:"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WHITESPACE = re.compile(r"\s+")
_PUBLIC_PASSAGE_FIELDS = frozenset({"passage_id", "document_id", "title", "text"})
_PROJECT_SCOPE_FIELDS = frozenset({"development", "validation"})
_CHUNK_FIELDS = frozenset(
    {
        "unit",
        "source_fields",
        "embedding_text_template",
        "overlap",
        "snippet_characters",
    }
)
_TOOL_BUDGET_FIELDS = frozenset(
    {"max_tool_calls_per_agent_call", "max_turns_per_agent_call"}
)
_FILE_ENTRY_FIELDS = frozenset({"name", "sha256"})
_FILE_FIELDS = frozenset({"passages", "embeddings"})


class EmbeddingEncoder(Protocol):
    """SentenceTransformer-compatible encoder used by build and search."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _positive_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _required_sha256(value: object, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_public_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()


def _native_task_id(value: object) -> str:
    task_id = _required_text(value, field_name="task_id")
    if task_id.startswith(_TASK_ID_PREFIX):
        task_id = task_id[len(_TASK_ID_PREFIX) :]
    if not task_id.startswith("tc_") or not task_id[3:].isdigit():
        raise ValueError("TriviaQA task_id must use triviaqa:tc_<integer>")
    return task_id


def _passage_identity(title: str, text: str) -> str:
    return _stable_sha256({"text": text, "title": title})


@dataclass(frozen=True, slots=True)
class DocumentPassage:
    """SkillFlow-compatible public passage projection."""

    passage_id: str
    document_id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        for field_name in ("passage_id", "document_id", "title", "text"):
            value = _required_text(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, value)

    def to_value(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "passage_id": self.passage_id,
            "text": self.text,
            "title": self.title,
        }

    @classmethod
    def from_value(cls, value: object) -> "DocumentPassage":
        if not isinstance(value, Mapping) or set(value) != _PUBLIC_PASSAGE_FIELDS:
            raise ValueError("passage row must contain only public passage fields")
        passage = cls(
            passage_id=value["passage_id"],
            document_id=value["document_id"],
            title=value["title"],
            text=value["text"],
        )
        normalized_title = _normalize_public_text(passage.title)
        normalized_text = _normalize_public_text(passage.text)
        if passage.title != normalized_title or passage.text != normalized_text:
            raise ValueError("passage title/text are not canonically normalized")
        digest = _passage_identity(passage.title, passage.text)
        if passage.passage_id != f"triviaqa-passage-{digest}":
            raise ValueError("passage_id does not match public passage content")
        if passage.document_id != f"triviaqa-document-{digest}":
            raise ValueError("document_id does not match public passage content")
        return passage


@dataclass(frozen=True, slots=True)
class SearchHit:
    """SkillFlow-compatible ranked hit extended with embedding similarity."""

    passage_id: str
    document_id: str
    title: str
    snippet: str
    rank: int
    similarity: float

    def __post_init__(self) -> None:
        for field_name in ("passage_id", "document_id", "title", "snippet"):
            value = _required_text(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, value)
        _positive_integer(self.rank, field_name="rank")
        if not isinstance(self.similarity, (int, float)) or not math.isfinite(
            float(self.similarity)
        ):
            raise ValueError("similarity must be finite")
        object.__setattr__(self, "similarity", float(self.similarity))

    def to_value(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "passage_id": self.passage_id,
            "rank": self.rank,
            "similarity": self.similarity,
            "snippet": self.snippet,
            "title": self.title,
        }


def _validated_scope(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != _PROJECT_SCOPE_FIELDS:
        raise ValueError("project_scope must contain development and validation")
    scope = {
        key: _positive_integer(value[key], field_name=f"project_scope.{key}")
        for key in sorted(_PROJECT_SCOPE_FIELDS)
    }
    return MappingProxyType(scope)


def _validated_chunk(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CHUNK_FIELDS:
        raise ValueError("chunk manifest fields are incompatible")
    if value.get("unit") != "one-search-result-per-passage":
        raise ValueError("chunk.unit is incompatible")
    if value.get("source_fields") != [
        "SearchResults.Title",
        "SearchResults.Description",
    ]:
        raise ValueError("chunk.source_fields must use the public field whitelist")
    if value.get("embedding_text_template") != "{title}\n\n{text}":
        raise ValueError("chunk.embedding_text_template is incompatible")
    if value.get("overlap") != 0:
        raise ValueError("chunk.overlap must be zero")
    snippet_characters = _positive_integer(
        value.get("snippet_characters"),
        field_name="chunk.snippet_characters",
    )
    return MappingProxyType(
        {
            "unit": "one-search-result-per-passage",
            "source_fields": [
                "SearchResults.Title",
                "SearchResults.Description",
            ],
            "embedding_text_template": "{title}\n\n{text}",
            "overlap": 0,
            "snippet_characters": snippet_characters,
        }
    )


def _validated_tool_budget(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != _TOOL_BUDGET_FIELDS:
        raise ValueError("tool_budget manifest fields are incompatible")
    budget = {
        key: _positive_integer(value[key], field_name=f"tool_budget.{key}")
        for key in sorted(_TOOL_BUDGET_FIELDS)
    }
    if budget["max_turns_per_agent_call"] <= budget["max_tool_calls_per_agent_call"]:
        raise ValueError("ReAct turn budget must leave one explicit completion turn")
    return MappingProxyType(budget)


def _validated_files(value: object) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping) or set(value) != _FILE_FIELDS:
        raise ValueError("files manifest fields are incompatible")
    files: dict[str, Mapping[str, str]] = {}
    expected_names = {
        "passages": PASSAGES_FILENAME,
        "embeddings": EMBEDDINGS_FILENAME,
    }
    for key in sorted(_FILE_FIELDS):
        entry = value[key]
        if not isinstance(entry, Mapping) or set(entry) != _FILE_ENTRY_FIELDS:
            raise ValueError(f"files.{key} fields are incompatible")
        name = _required_text(entry["name"], field_name=f"files.{key}.name")
        if name != expected_names[key]:
            raise ValueError(f"files.{key}.name is incompatible")
        files[key] = MappingProxyType(
            {
                "name": name,
                "sha256": _required_sha256(
                    entry["sha256"], field_name=f"files.{key}.sha256"
                ),
            }
        )
    return MappingProxyType(files)


@dataclass(frozen=True, slots=True)
class TriviaQAEmbeddingIndexManifest:
    """Content-addressed identity and frozen retrieval protocol."""

    schema_version: str
    format: str
    retrieval_backend: str
    index_id: str
    corpus_name: str
    corpus_version: str
    source_dataset: str
    source_configuration: str
    source_split: str
    project_scope: Mapping[str, int]
    selected_task_count: int
    selected_task_ids_sha256: str
    source_search_result_count: int
    skipped_empty_result_count: int
    duplicate_result_count: int
    document_count: int
    passage_count: int
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    normalization: str
    similarity: str
    query_prefix: str
    frozen_top_k: int
    chunk: Mapping[str, object]
    tool_budget: Mapping[str, int]
    files: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("embedding manifest schema version is unsupported")
        if self.format != INDEX_FORMAT:
            raise ValueError("embedding index format is unsupported")
        if self.retrieval_backend != RETRIEVAL_BACKEND:
            raise ValueError("embedding retrieval backend is unsupported")
        if self.corpus_name != CORPUS_NAME:
            raise ValueError("embedding corpus name is unsupported")
        if self.source_dataset != SOURCE_DATASET:
            raise ValueError("embedding source dataset is unsupported")
        if self.source_configuration != SOURCE_CONFIGURATION:
            raise ValueError("embedding source configuration is unsupported")
        if self.source_split != SOURCE_SPLIT:
            raise ValueError("embedding source split is unsupported")

        scope = _validated_scope(self.project_scope)
        object.__setattr__(self, "project_scope", scope)
        selected_task_count = _positive_integer(
            self.selected_task_count,
            field_name="selected_task_count",
        )
        if selected_task_count != sum(scope.values()):
            raise ValueError("selected_task_count differs from project_scope")
        _required_sha256(
            self.selected_task_ids_sha256,
            field_name="selected_task_ids_sha256",
        )
        for field_name in (
            "source_search_result_count",
            "skipped_empty_result_count",
            "duplicate_result_count",
        ):
            _nonnegative_integer(getattr(self, field_name), field_name=field_name)
        document_count = _positive_integer(
            self.document_count,
            field_name="document_count",
        )
        passage_count = _positive_integer(
            self.passage_count,
            field_name="passage_count",
        )
        if document_count != passage_count:
            raise ValueError("one-result-per-passage requires equal document counts")
        if self.source_search_result_count != (
            passage_count
            + self.duplicate_result_count
            + self.skipped_empty_result_count
        ):
            raise ValueError("search-result accounting does not close")

        _required_text(self.embedding_model, field_name="embedding_model")
        _required_text(
            self.embedding_model_revision,
            field_name="embedding_model_revision",
        )
        _positive_integer(
            self.embedding_dimension,
            field_name="embedding_dimension",
        )
        if self.normalization != NORMALIZATION:
            raise ValueError("embedding normalization must be l2")
        if self.similarity != SIMILARITY:
            raise ValueError("embedding similarity must be dot_product")
        if self.query_prefix != BGE_QUERY_PREFIX:
            raise ValueError("embedding query_prefix differs from SkillFlow BGE")
        frozen_top_k = _positive_integer(
            self.frozen_top_k,
            field_name="frozen_top_k",
        )
        if frozen_top_k > passage_count:
            raise ValueError("frozen_top_k exceeds passage_count")

        object.__setattr__(self, "chunk", _validated_chunk(self.chunk))
        object.__setattr__(
            self,
            "tool_budget",
            _validated_tool_budget(self.tool_budget),
        )
        object.__setattr__(self, "files", _validated_files(self.files))
        passages_digest = self.files["passages"]["sha256"]
        if self.corpus_version != f"sha256:{passages_digest}":
            raise ValueError("corpus_version differs from passages content")
        _required_sha256(self.index_id, field_name="index_id")
        if self.index_id != _stable_sha256(self._identity_value()):
            raise ValueError("index_id differs from the frozen index identity")

    def _identity_value(self) -> dict[str, object]:
        value = self.to_value()
        value.pop("index_id")
        return value

    def to_value(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "retrieval_backend": self.retrieval_backend,
            "index_id": self.index_id,
            "corpus_name": self.corpus_name,
            "corpus_version": self.corpus_version,
            "source_dataset": self.source_dataset,
            "source_configuration": self.source_configuration,
            "source_split": self.source_split,
            "project_scope": dict(self.project_scope),
            "selected_task_count": self.selected_task_count,
            "selected_task_ids_sha256": self.selected_task_ids_sha256,
            "source_search_result_count": self.source_search_result_count,
            "skipped_empty_result_count": self.skipped_empty_result_count,
            "duplicate_result_count": self.duplicate_result_count,
            "document_count": self.document_count,
            "passage_count": self.passage_count,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalization,
            "similarity": self.similarity,
            "query_prefix": self.query_prefix,
            "frozen_top_k": self.frozen_top_k,
            "chunk": {
                "unit": self.chunk["unit"],
                "source_fields": list(self.chunk["source_fields"]),
                "embedding_text_template": self.chunk["embedding_text_template"],
                "overlap": self.chunk["overlap"],
                "snippet_characters": self.chunk["snippet_characters"],
            },
            "tool_budget": dict(self.tool_budget),
            "files": {key: dict(self.files[key]) for key in sorted(self.files)},
        }

    @classmethod
    def create(
        cls,
        *,
        project_scope: Mapping[str, int],
        selected_task_ids_sha256: str,
        source_search_result_count: int,
        skipped_empty_result_count: int,
        duplicate_result_count: int,
        document_count: int,
        embedding_model: str,
        embedding_model_revision: str,
        embedding_dimension: int,
        frozen_top_k: int,
        snippet_characters: int,
        max_tool_calls_per_agent_call: int,
        max_turns_per_agent_call: int,
        passages_sha256: str,
        embeddings_sha256: str,
    ) -> "TriviaQAEmbeddingIndexManifest":
        base: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "format": INDEX_FORMAT,
            "retrieval_backend": RETRIEVAL_BACKEND,
            "index_id": "0" * 64,
            "corpus_name": CORPUS_NAME,
            "corpus_version": f"sha256:{passages_sha256}",
            "source_dataset": SOURCE_DATASET,
            "source_configuration": SOURCE_CONFIGURATION,
            "source_split": SOURCE_SPLIT,
            "project_scope": dict(project_scope),
            "selected_task_count": sum(project_scope.values()),
            "selected_task_ids_sha256": selected_task_ids_sha256,
            "source_search_result_count": source_search_result_count,
            "skipped_empty_result_count": skipped_empty_result_count,
            "duplicate_result_count": duplicate_result_count,
            "document_count": document_count,
            "passage_count": document_count,
            "embedding_model": embedding_model,
            "embedding_model_revision": embedding_model_revision,
            "embedding_dimension": embedding_dimension,
            "normalization": NORMALIZATION,
            "similarity": SIMILARITY,
            "query_prefix": BGE_QUERY_PREFIX,
            "frozen_top_k": frozen_top_k,
            "chunk": {
                "unit": "one-search-result-per-passage",
                "source_fields": [
                    "SearchResults.Title",
                    "SearchResults.Description",
                ],
                "embedding_text_template": "{title}\n\n{text}",
                "overlap": 0,
                "snippet_characters": snippet_characters,
            },
            "tool_budget": {
                "max_tool_calls_per_agent_call": max_tool_calls_per_agent_call,
                "max_turns_per_agent_call": max_turns_per_agent_call,
            },
            "files": {
                "passages": {
                    "name": PASSAGES_FILENAME,
                    "sha256": passages_sha256,
                },
                "embeddings": {
                    "name": EMBEDDINGS_FILENAME,
                    "sha256": embeddings_sha256,
                },
            },
        }
        identity = dict(base)
        identity.pop("index_id")
        base["index_id"] = _stable_sha256(identity)
        return cls.from_value(base)

    @classmethod
    def from_value(cls, value: object) -> "TriviaQAEmbeddingIndexManifest":
        fields = frozenset(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("embedding manifest fields are incompatible")
        return cls(**{field_name: value[field_name] for field_name in fields})


_MODEL_CACHE: dict[tuple[str, str], EmbeddingEncoder] = {}
_MODEL_CACHE_LOCK = RLock()


def _get_embedding_model(model_name: str, model_revision: str) -> EmbeddingEncoder:
    """Load the configured SentenceTransformer without network fallback."""

    model_name = _required_text(model_name, field_name="embedding_model")
    model_revision = _required_text(
        model_revision,
        field_name="embedding_model_revision",
    )
    key = (model_name, model_revision)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers is required for the embedding index"
            ) from exc
        try:
            model = SentenceTransformer(
                model_name,
                device="cpu",
                revision=model_revision,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "configured embedding model failed to load locally"
            ) from exc
        _MODEL_CACHE[key] = model
        return model


def _normalized_embeddings(
    encoder: EmbeddingEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        raise ValueError("embedding input cannot be empty")
    encoded = encoder.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    embeddings = np.asarray(encoded, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError("embedding encoder returned an incompatible shape")
    if embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise RuntimeError("embedding encoder returned invalid vectors")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise RuntimeError("embedding encoder returned a zero vector")
    # SentenceTransformer already normalizes because the call above sets
    # normalize_embeddings=True.  Re-normalizing in float32 closes small model-
    # backend differences and makes the persisted dot-product contract exact.
    embeddings = np.asarray(embeddings / norms, dtype=np.float32)
    return np.ascontiguousarray(embeddings)


def _embedding_text(passage: DocumentPassage) -> str:
    return f"{passage.title}\n\n{passage.text}"


def load_frozen_triviaqa_task_ids(
    development_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    *,
    expected_development_count: int = 512,
    expected_validation_count: int = 128,
) -> tuple[frozenset[str], frozenset[str]]:
    """Read only task IDs from the project's frozen aligned JSONL splits."""

    expected_counts = {
        "development": _positive_integer(
            expected_development_count,
            field_name="expected_development_count",
        ),
        "validation": _positive_integer(
            expected_validation_count,
            field_name="expected_validation_count",
        ),
    }
    paths = {
        "development": Path(development_tasks_path),
        "validation": Path(validation_tasks_path),
    }
    expected_project_splits = {"development": "train", "validation": "validation"}
    result: dict[str, frozenset[str]] = {}
    for scope_name in ("development", "validation"):
        path = paths[scope_name]
        if not path.is_file():
            raise FileNotFoundError(f"frozen {scope_name} task file is unavailable")
        task_ids: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"frozen task JSONL is invalid at line {line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError("frozen task row must be an object")
                metadata = row.get("metadata")
                if not isinstance(metadata, Mapping):
                    continue
                if metadata.get("dataset_key") != "triviaqa":
                    continue
                if row.get("split") != expected_project_splits[scope_name]:
                    raise ValueError(
                        f"frozen {scope_name} task has an incompatible project split"
                    )
                task_id = _native_task_id(row.get("task_id"))
                if task_id in task_ids:
                    raise ValueError(f"duplicate frozen TriviaQA task_id: {task_id}")
                task_ids.add(task_id)
        if len(task_ids) != expected_counts[scope_name]:
            raise ValueError(
                f"frozen {scope_name} TriviaQA count is {len(task_ids)}, "
                f"expected {expected_counts[scope_name]}"
            )
        result[scope_name] = frozenset(task_ids)
    overlap = result["development"] & result["validation"]
    if overlap:
        raise ValueError("frozen development and validation task IDs overlap")
    return result["development"], result["validation"]


@contextmanager
def _open_original_triviaqa_source(
    source_path: Path,
) -> Iterator[tuple[TextIO, str]]:
    """Open the original JSON directly or stream it from the official tar."""

    if not source_path.is_file():
        raise FileNotFoundError("original TriviaQA source is unavailable")
    lower_name = source_path.name.casefold()
    if lower_name.endswith(".json"):
        with source_path.open(encoding="utf-8") as handle:
            yield handle, "triviaqa-unfiltered/unfiltered-web-train.json"
        return
    if not lower_name.endswith((".tar", ".tar.gz", ".tgz")):
        raise ValueError("TriviaQA source must be unfiltered-web-train JSON or tar")
    with tarfile.open(source_path, mode="r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/unfiltered-web-train.json")
        ]
        if len(members) != 1:
            raise ValueError(
                "TriviaQA archive must contain exactly one unfiltered-web-train.json"
            )
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise RuntimeError("TriviaQA archive member could not be opened")
        with extracted:
            with io.TextIOWrapper(extracted, encoding="utf-8") as handle:
                yield handle, members[0].name


def _iter_triviaqa_data_records(handle: TextIO) -> Iterator[Mapping[str, object]]:
    """Incrementally decode the top-level ``Data`` array with stdlib JSON."""

    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    prefix = re.compile(r"\A\s*\{\s*\"Data\"\s*:\s*\[")
    while True:
        match = prefix.match(buffer)
        if match is not None:
            buffer = buffer[match.end() :]
            break
        chunk = handle.read(1024 * 1024)
        if not chunk:
            raise ValueError("TriviaQA JSON does not start with a Data array")
        buffer += chunk
        if len(buffer) > 4 * 1024 * 1024:
            raise ValueError("TriviaQA JSON prefix is incompatible")

    while True:
        stripped = buffer.lstrip()
        if stripped.startswith(","):
            stripped = stripped[1:].lstrip()
        buffer = stripped
        if buffer.startswith("]"):
            return
        while True:
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(
                        "TriviaQA Data array is truncated or invalid"
                    ) from exc
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            break
        if not isinstance(value, Mapping):
            raise ValueError("TriviaQA Data item must be an object")
        yield value
        buffer = buffer[end:]


def _public_passage(title_value: object, text_value: object) -> DocumentPassage | None:
    title = _normalize_public_text(title_value)
    text = _normalize_public_text(text_value)
    if not title or not text:
        return None
    digest = _passage_identity(title, text)
    return DocumentPassage(
        passage_id=f"triviaqa-passage-{digest}",
        document_id=f"triviaqa-document-{digest}",
        title=title,
        text=text,
    )


def _collect_public_passages(
    source_path: Path,
    selected_task_ids: frozenset[str],
) -> tuple[tuple[DocumentPassage, ...], int, int, int, str]:
    remaining = set(selected_task_ids)
    matched: set[str] = set()
    passages_by_id: dict[str, DocumentPassage] = {}
    source_result_count = 0
    skipped_empty_count = 0
    duplicate_count = 0
    source_member = ""

    with _open_original_triviaqa_source(source_path) as (handle, member_name):
        source_member = member_name
        for record in _iter_triviaqa_data_records(handle):
            question_id_value = record.get("QuestionId")
            if not isinstance(question_id_value, str):
                continue
            question_id = question_id_value.strip()
            if question_id not in remaining:
                continue
            if question_id in matched:
                raise ValueError(
                    f"duplicate original TriviaQA QuestionId: {question_id}"
                )
            matched.add(question_id)
            remaining.remove(question_id)

            # FIELD WHITELIST: only these two public SearchResults fields are
            # inspected.  Record.Question, Answer, Aliases, evaluator metadata,
            # URLs, ranks, and every other source field remain outside the corpus.
            search_results = record.get("SearchResults")
            if not isinstance(search_results, list):
                raise ValueError(
                    f"TriviaQA record {question_id} has invalid SearchResults"
                )
            for result in search_results:
                source_result_count += 1
                if not isinstance(result, Mapping):
                    skipped_empty_count += 1
                    continue
                passage = _public_passage(
                    result.get("Title"),
                    result.get("Description"),
                )
                if passage is None:
                    skipped_empty_count += 1
                    continue
                if passage.passage_id in passages_by_id:
                    duplicate_count += 1
                    continue
                passages_by_id[passage.passage_id] = passage
            if not remaining:
                break
    if remaining:
        preview = ", ".join(sorted(remaining)[:8])
        raise ValueError(
            f"original TriviaQA source is missing {len(remaining)} frozen IDs: {preview}"
        )
    passages = tuple(passages_by_id[key] for key in sorted(passages_by_id))
    if not passages:
        raise ValueError("selected TriviaQA scope produced no public passages")
    return (
        passages,
        source_result_count,
        skipped_empty_count,
        duplicate_count,
        source_member,
    )


def _write_atomic_bytes(path: Path, content_writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            content_writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_triviaqa_embedding_index(
    *,
    source_path: str | Path,
    development_tasks_path: str | Path,
    validation_tasks_path: str | Path,
    output_dir: str | Path,
    embedding_model: str,
    embedding_model_revision: str,
    frozen_top_k: int,
    max_tool_calls_per_agent_call: int,
    max_turns_per_agent_call: int,
    encoder: EmbeddingEncoder | None = None,
    batch_size: int = 64,
    snippet_characters: int = 512,
    expected_development_count: int = 512,
    expected_validation_count: int = 128,
) -> TriviaQAEmbeddingIndexManifest:
    """Build one deterministic, answer-private local embedding index."""

    model_name = _required_text(embedding_model, field_name="embedding_model")
    model_revision = _required_text(
        embedding_model_revision,
        field_name="embedding_model_revision",
    )
    frozen_top_k = _positive_integer(frozen_top_k, field_name="frozen_top_k")
    batch_size = _positive_integer(batch_size, field_name="batch_size")
    snippet_characters = _positive_integer(
        snippet_characters,
        field_name="snippet_characters",
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
        raise ValueError("ReAct turn budget must leave one explicit completion turn")

    development_ids, validation_ids = load_frozen_triviaqa_task_ids(
        development_tasks_path,
        validation_tasks_path,
        expected_development_count=expected_development_count,
        expected_validation_count=expected_validation_count,
    )
    selected_task_ids = frozenset((*development_ids, *validation_ids))
    project_scope = {
        "development": len(development_ids),
        "validation": len(validation_ids),
    }
    selected_task_ids_sha256 = _stable_sha256(sorted(selected_task_ids))
    (
        passages,
        source_search_result_count,
        skipped_empty_result_count,
        duplicate_result_count,
        source_member,
    ) = _collect_public_passages(Path(source_path), selected_task_ids)
    if not source_member.endswith("/unfiltered-web-train.json"):
        raise ValueError("TriviaQA source member is not unfiltered-web-train")
    if frozen_top_k > len(passages):
        raise ValueError("frozen_top_k exceeds the public corpus size")

    resolved_encoder = encoder or _get_embedding_model(model_name, model_revision)
    embedding_inputs = [_embedding_text(passage) for passage in passages]
    embeddings = _normalized_embeddings(
        resolved_encoder,
        embedding_inputs,
        batch_size=batch_size,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    passages_path = root / PASSAGES_FILENAME
    embeddings_path = root / EMBEDDINGS_FILENAME
    manifest_path = root / MANIFEST_FILENAME

    def write_passages(handle: Any) -> None:
        for passage in passages:
            handle.write(_canonical_json(passage.to_value()) + b"\n")

    _write_atomic_bytes(passages_path, write_passages)
    _write_atomic_bytes(
        embeddings_path,
        lambda handle: np.save(handle, embeddings, allow_pickle=False),
    )
    passages_sha256 = _file_sha256(passages_path)
    embeddings_sha256 = _file_sha256(embeddings_path)
    manifest = TriviaQAEmbeddingIndexManifest.create(
        project_scope=project_scope,
        selected_task_ids_sha256=selected_task_ids_sha256,
        source_search_result_count=source_search_result_count,
        skipped_empty_result_count=skipped_empty_result_count,
        duplicate_result_count=duplicate_result_count,
        document_count=len(passages),
        embedding_model=model_name,
        embedding_model_revision=model_revision,
        embedding_dimension=int(embeddings.shape[1]),
        frozen_top_k=frozen_top_k,
        snippet_characters=snippet_characters,
        max_tool_calls_per_agent_call=max_tool_calls_per_agent_call,
        max_turns_per_agent_call=max_turns_per_agent_call,
        passages_sha256=passages_sha256,
        embeddings_sha256=embeddings_sha256,
    )
    _write_atomic_bytes(
        manifest_path,
        lambda handle: handle.write(_canonical_json(manifest.to_value()) + b"\n"),
    )
    return manifest


class TriviaQAEmbeddingIndex:
    """Immutable local dense index implementing SkillFlow search/read semantics."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: TriviaQAEmbeddingIndexManifest,
        passages: tuple[DocumentPassage, ...],
        embeddings: np.ndarray,
        encoder: EmbeddingEncoder,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self._passages = passages
        self._passage_by_id = {passage.passage_id: passage for passage in passages}
        self._embeddings: np.ndarray | None = embeddings
        self._encoder = encoder
        self._lock = RLock()
        self._closed = False

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        encoder: EmbeddingEncoder | None = None,
    ) -> "TriviaQAEmbeddingIndex":
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError("TriviaQA embedding index directory is unavailable")
        manifest_path = root_path / MANIFEST_FILENAME
        passages_path = root_path / PASSAGES_FILENAME
        embeddings_path = root_path / EMBEDDINGS_FILENAME
        for path in (manifest_path, passages_path, embeddings_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"embedding index file is unavailable: {path.name}"
                )

        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("embedding manifest JSON is invalid") from exc
        manifest = TriviaQAEmbeddingIndexManifest.from_value(manifest_value)
        if _file_sha256(passages_path) != manifest.files["passages"]["sha256"]:
            raise ValueError("passages file differs from the embedding manifest")
        if _file_sha256(embeddings_path) != manifest.files["embeddings"]["sha256"]:
            raise ValueError("embeddings file differs from the embedding manifest")

        passages: list[DocumentPassage] = []
        with passages_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"empty passage row at line {line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"passage JSON is invalid at line {line_number}"
                    ) from exc
                passages.append(DocumentPassage.from_value(value))
        if len(passages) != manifest.passage_count:
            raise ValueError("passage count differs from the embedding manifest")
        passage_ids = [passage.passage_id for passage in passages]
        if passage_ids != sorted(passage_ids) or len(set(passage_ids)) != len(
            passage_ids
        ):
            raise ValueError("passage rows must have unique canonical ordering")

        embeddings = np.load(
            embeddings_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if embeddings.dtype != np.dtype("float32"):
            raise ValueError("embedding dtype must be float32")
        if embeddings.shape != (
            manifest.passage_count,
            manifest.embedding_dimension,
        ):
            raise ValueError("embedding shape differs from the manifest")
        if not np.isfinite(embeddings).all():
            raise ValueError("embedding matrix contains non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("embedding matrix is not l2-normalized")

        resolved_encoder = encoder or _get_embedding_model(
            manifest.embedding_model,
            manifest.embedding_model_revision,
        )
        return cls(
            root=root_path,
            manifest=manifest,
            passages=tuple(passages),
            embeddings=embeddings,
            encoder=resolved_encoder,
        )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open_embeddings(self) -> np.ndarray:
        if self._closed or self._embeddings is None:
            raise RuntimeError("TriviaQA embedding index is closed")
        return self._embeddings

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        query_text = _required_text(query, field_name="query")
        if type(limit) is not int or limit != self.manifest.frozen_top_k:
            raise ValueError("search limit differs from the frozen top-k")
        with self._lock:
            embeddings = self._require_open_embeddings()
            # DIRECT_REUSE: SkillFlow SkillWorkspace._enhance_query adds this
            # BGE retrieval instruction and ordinary ``Question:`` label before
            # normalized query encoding.  Passage encoding remains unprefixed.
            embedding_query = f"{self.manifest.query_prefix}Question: {query_text}"
            query_embedding = _normalized_embeddings(
                self._encoder,
                [embedding_query],
                batch_size=1,
            )[0]
            scores = np.asarray(embeddings @ query_embedding, dtype=np.float32)
            order = sorted(
                range(len(self._passages)),
                key=lambda index: (
                    -float(scores[index]),
                    self._passages[index].passage_id,
                ),
            )[:limit]
            snippet_characters = int(self.manifest.chunk["snippet_characters"])
            return tuple(
                SearchHit(
                    passage_id=self._passages[index].passage_id,
                    document_id=self._passages[index].document_id,
                    title=self._passages[index].title,
                    snippet=self._passages[index].text[:snippet_characters],
                    rank=rank,
                    similarity=float(scores[index]),
                )
                for rank, index in enumerate(order, start=1)
            )

    def read(self, passage_id: str) -> DocumentPassage:
        resolved_id = _required_text(passage_id, field_name="passage_id")
        with self._lock:
            self._require_open_embeddings()
            try:
                return self._passage_by_id[resolved_id]
            except KeyError as exc:
                raise KeyError("unknown TriviaQA passage_id") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            embeddings = self._embeddings
            self._embeddings = None
            self._closed = True
            mmap = getattr(embeddings, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "TriviaQAEmbeddingIndex":
        if self.closed:
            raise RuntimeError("TriviaQA embedding index is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "BGE_QUERY_PREFIX",
    "CORPUS_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DocumentPassage",
    "EMBEDDINGS_FILENAME",
    "INDEX_FORMAT",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "PASSAGES_FILENAME",
    "RETRIEVAL_BACKEND",
    "SearchHit",
    "TriviaQAEmbeddingIndex",
    "TriviaQAEmbeddingIndexManifest",
    "build_triviaqa_embedding_index",
    "load_frozen_triviaqa_task_ids",
]
