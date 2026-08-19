"""HealthBench Professional Tool over a frozen MedRAG textbook BM25 corpus.

The resource loading and ranking path is a thin, dependency-light adaptation
of SkillFlow ``training/environment.py``::_load_external_corpus and
``_search_external_corpus`` (upstream lines 6757-6830).  The formal resource
contract follows ``skillev.runtime.formal_preflight`` (upstream lines
713-763): one explicitly configured directory contains ``bm25_index.pkl``,
``all_chunks.jsonl``, and ``.source_revision`` and is checked against the
configured revision and row count.

Only the public observation-producing search path is adapted.  This module
does not import SkillFlow's GenericTaskEnvironment, task gold fields,
verification, reward, or evaluator state.  The upstream fixed BM25 constants,
top-3 limit, score threshold, tokenization, and 500-character chunk projection
are preserved.  Global environment lookup and class-level caching are replaced
by an explicit, closeable resource lifetime so the same ToolRegistry contract
can be used by FlowSteer's existing ToolReactExecutionAdapter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
import re
from threading import RLock
from typing import Mapping

from .tool_runtime import (
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID = "healthbench-medrag.search"
HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE = ("healthbench_professional",)
MEDRAG_BM25_TOP_K = 3
MEDRAG_BM25_K1 = 1.5
MEDRAG_BM25_B = 0.75
MEDRAG_BM25_MINIMUM_SCORE = 1.0


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _required_positive_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"configured MedRAG resource is unavailable: {label}")


@dataclass(slots=True)
class FrozenMedRAGBM25Corpus:
    """One immutable-at-execution corpus loaded from SkillFlow's BM25 assets."""

    source_identity: str
    source_revision: str
    corpus_rows: int
    _corpus: tuple[str, ...] = field(repr=False)
    _index: Mapping[str, object] = field(repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def open(
        cls,
        corpus_root: str | Path,
        *,
        source_identity: str,
        expected_source_revision: str,
        expected_rows: int,
    ) -> "FrozenMedRAGBM25Corpus":
        """Load the three files required by SkillFlow formal preflight."""

        root = Path(corpus_root)
        source_identity = _required_text(
            source_identity, field_name="source_identity"
        )
        expected_source_revision = _required_text(
            expected_source_revision,
            field_name="expected_source_revision",
        )
        expected_rows = _required_positive_integer(
            expected_rows, field_name="expected_rows"
        )
        if not root.is_dir():
            raise RuntimeError("configured MedRAG corpus root is unavailable")

        index_path = root / "bm25_index.pkl"
        corpus_path = root / "all_chunks.jsonl"
        revision_path = root / ".source_revision"
        _require_file(index_path, label="bm25_index.pkl")
        _require_file(corpus_path, label="all_chunks.jsonl")
        _require_file(revision_path, label=".source_revision")

        source_revision = revision_path.read_text(encoding="utf-8").strip()
        if source_revision != expected_source_revision:
            raise RuntimeError("MedRAG textbook source revision differs")

        # SkillFlow formal_preflight counts physical JSONL rows before runtime.
        with corpus_path.open("rb") as corpus_handle:
            physical_rows = sum(1 for _ in corpus_handle)
        if physical_rows != expected_rows:
            raise RuntimeError("MedRAG textbook snippet count differs")

        corpus: list[str] = []
        with corpus_path.open(encoding="utf-8") as corpus_handle:
            for line in corpus_handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("MedRAG textbook corpus row is invalid") from exc
                if not isinstance(record, Mapping):
                    raise RuntimeError("MedRAG textbook corpus row is not an object")
                contents = record.get("contents", record.get("content", ""))
                if not isinstance(contents, str):
                    raise RuntimeError("MedRAG textbook corpus contents are invalid")
                corpus.append(contents)
        if len(corpus) != expected_rows:
            raise RuntimeError("MedRAG textbook parsed snippet count differs")

        with index_path.open("rb") as index_handle:
            index = pickle.load(index_handle)
        if not isinstance(index, Mapping):
            raise RuntimeError("MedRAG BM25 index is incompatible")
        required_index_fields = {"avg_dl", "doc_lens", "idf", "inverted_index"}
        if not required_index_fields.issubset(index):
            raise RuntimeError("MedRAG BM25 index is incompatible")

        return cls(
            source_identity=source_identity,
            source_revision=source_revision,
            corpus_rows=expected_rows,
            _corpus=tuple(corpus),
            _index=index,
        )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def identity(self) -> dict[str, object]:
        return {
            "source": self.source_identity,
            "source_revision": self.source_revision,
            "corpus_rows": self.corpus_rows,
            "retrieval_backend": "bm25",
        }

    def search(self, query: str) -> list[dict[str, object]]:
        """Port SkillFlow's fixed BM25 top-3 public observation projection."""

        query = _required_text(query, field_name="query")
        with self._lock:
            if self._closed:
                raise RuntimeError("MedRAG BM25 corpus is closed")

            index = self._index
            idf = index["idf"]
            inverted_index = index["inverted_index"]
            document_lengths = index["doc_lens"]
            average_document_length = index["avg_dl"]
            if not isinstance(idf, Mapping) or not isinstance(
                inverted_index, Mapping
            ):
                raise RuntimeError("MedRAG BM25 index is incompatible")
            if not isinstance(document_lengths, (list, tuple)) or not isinstance(
                average_document_length, (int, float)
            ):
                raise RuntimeError("MedRAG BM25 index is incompatible")

            # Exact tokenization and scoring constants from SkillFlow.
            query_terms = re.findall(r"\b\w+\b", query.lower())
            scores: defaultdict[int, float] = defaultdict(float)
            for query_term in query_terms:
                if query_term not in idf:
                    continue
                term_idf = float(idf[query_term])
                postings = inverted_index.get(query_term, [])
                for document_id, term_frequency in postings:
                    document_length = document_lengths[document_id]
                    normalized_frequency = (
                        term_frequency * (MEDRAG_BM25_K1 + 1)
                    ) / (
                        term_frequency
                        + MEDRAG_BM25_K1
                        * (
                            1
                            - MEDRAG_BM25_B
                            + MEDRAG_BM25_B
                            * document_length
                            / average_document_length
                        )
                    )
                    scores[document_id] += term_idf * normalized_frequency

            top = sorted(scores.items(), key=lambda item: -item[1])[
                :MEDRAG_BM25_TOP_K
            ]
            ranked_chunks: list[dict[str, object]] = []
            for rank, (document_id, score) in enumerate(top, 1):
                if score < MEDRAG_BM25_MINIMUM_SCORE:
                    continue
                text = (
                    self._corpus[document_id]
                    if document_id < len(self._corpus)
                    else ""
                )
                matched_terms = [
                    term for term in query_terms if term in text.lower()
                ]
                ranked_chunks.append(
                    {
                        "chunk_id": str(document_id),
                        "rank": rank,
                        "score": float(score),
                        "matched_terms": matched_terms,
                        "text": text[:500],
                    }
                )
            return ranked_chunks

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._corpus = ()
            self._index = {}
            self._closed = True


@dataclass(frozen=True, slots=True)
class HealthBenchMedRAGSearchToolBackend:
    """Expose only frozen textbook search, never HealthBench evaluator fields."""

    corpus: FrozenMedRAGBM25Corpus

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != "search":
            raise ValueError(
                "HealthBench MedRAG backend received an incompatible action"
            )
        if set(request.arguments) != {"query"}:
            raise ValueError("search arguments must contain exactly query")
        query = request.arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be non-empty text")
        return ToolResult(
            {
                "operation": "search",
                "frozen_corpus": self.corpus.identity,
                "query": query,
                "top_k": MEDRAG_BM25_TOP_K,
                "ranked_chunks": self.corpus.search(query),
            }
        )


def build_healthbench_medrag_tool_registry(
    corpus: FrozenMedRAGBM25Corpus,
    *,
    timeout_seconds: float = 10.0,
) -> ToolRegistry:
    """Register one HealthBench-only, read-only SkillFlow BM25 Tool."""

    if not isinstance(corpus, FrozenMedRAGBM25Corpus):
        raise TypeError("corpus must be a FrozenMedRAGBM25Corpus")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    corpus_identity_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source",
            "source_revision",
            "corpus_rows",
            "retrieval_backend",
        ],
        "properties": {
            "source": {"type": "string"},
            "source_revision": {"type": "string"},
            "corpus_rows": {"type": "integer"},
            "retrieval_backend": {"const": "bm25"},
        },
    }
    capability = ToolCapability(
        tool_id=HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
        dataset_scope=HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
            },
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": [
                "operation",
                "frozen_corpus",
                "query",
                "top_k",
                "ranked_chunks",
            ],
            "properties": {
                "operation": {"const": "search"},
                "frozen_corpus": corpus_identity_schema,
                "query": {"type": "string"},
                "top_k": {"const": MEDRAG_BM25_TOP_K},
                "ranked_chunks": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
        side_effect="none",
        timeout_seconds=timeout_seconds,
        version=corpus.source_revision,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID,
                HealthBenchMedRAGSearchToolBackend(corpus),
                capability,
            ),
        )
    )


@dataclass(slots=True)
class OpenHealthBenchMedRAGToolRegistry:
    """Owned frozen corpus plus its immutable ToolRegistry registration."""

    registry: ToolRegistry
    frozen_corpus_identity: Mapping[str, object]
    _corpus: FrozenMedRAGBM25Corpus = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._corpus.close()
            self._closed = True

    def __enter__(self) -> "OpenHealthBenchMedRAGToolRegistry":
        if self._closed:
            raise RuntimeError("HealthBench MedRAG ToolRegistry is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def open_healthbench_medrag_tool_registry(
    *,
    corpus_root: str | Path,
    source_identity: str,
    expected_source_revision: str,
    expected_rows: int,
    timeout_seconds: float = 10.0,
) -> OpenHealthBenchMedRAGToolRegistry:
    """Open an explicitly configured frozen corpus and own its lifecycle."""

    corpus = FrozenMedRAGBM25Corpus.open(
        corpus_root,
        source_identity=source_identity,
        expected_source_revision=expected_source_revision,
        expected_rows=expected_rows,
    )
    try:
        registry = build_healthbench_medrag_tool_registry(
            corpus,
            timeout_seconds=timeout_seconds,
        )
    except BaseException:
        corpus.close()
        raise
    return OpenHealthBenchMedRAGToolRegistry(
        registry=registry,
        frozen_corpus_identity=corpus.identity,
        _corpus=corpus,
    )


__all__ = [
    "FrozenMedRAGBM25Corpus",
    "HEALTHBENCH_MEDRAG_SEARCH_TOOL_ID",
    "HEALTHBENCH_PROFESSIONAL_DATASET_SCOPE",
    "HealthBenchMedRAGSearchToolBackend",
    "MEDRAG_BM25_TOP_K",
    "OpenHealthBenchMedRAGToolRegistry",
    "build_healthbench_medrag_tool_registry",
    "open_healthbench_medrag_tool_registry",
]
