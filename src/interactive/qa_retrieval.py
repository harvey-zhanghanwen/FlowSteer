"""Thin AgentGraph adapter over SkillFlow's public QA retrieval index.

SkillFlow owns the retrieval implementation and corpus contract.  This module
does not implement another ranker or database schema: it imports
``skillev.benchmarks.RetrievalIndex`` from the configured SkillFlow checkout,
calls its read-only ``search`` and ``read`` methods, and renders only their
public observations for the existing FlowSteer task/Canvas boundary.

The adapter is intentionally answer-free.  Accepted answers and evaluator
state are never accepted as inputs and cannot enter the rendered problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from .records import TaskRecord


DEFAULT_SKILLFLOW_SOURCE = Path(
    "/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src"
)
DEFAULT_QA_RETRIEVAL_INDEX = Path(
    "/ssd1/iclr/SKILLEV/skillev-new-b2-temp/data/datasets/"
    "dpr-wikipedia/atlas-retrieval.sqlite3"
)


class SkillFlowRetrievalError(RuntimeError):
    """SkillFlow's configured public retrieval boundary is unavailable."""


# Standard information-retrieval stop-word filtering keeps common question
# syntax out of FTS5's OR query.  SkillFlow deliberately leaves ``query`` to
# the policy; this deterministic evaluation condition freezes the equivalent
# keyword-query policy so Direct and AgentGraph receive identical observations.
_QUERY_TOKEN = re.compile(r"\w+", flags=re.UNICODE)
_QUESTION_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    }
)


def build_keyword_query(question: str, *, max_terms: int = 12) -> str:
    """Freeze a compact keyword query for SkillFlow's public search action."""

    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    text = str(question).strip()
    if not text:
        raise ValueError("question must be non-empty")
    selected: list[str] = []
    seen: set[str] = set()
    for token in _QUERY_TOKEN.findall(text):
        normalized = token.casefold()
        if normalized in _QUESTION_STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(token)
        if len(selected) == max_terms:
            break
    if not selected:
        selected = _QUERY_TOKEN.findall(text)[:max_terms]
    return " ".join(selected)


@dataclass(frozen=True, slots=True)
class PublicPassageObservation:
    """One model-visible ``search`` hit followed by its ``read`` result."""

    rank: int
    passage_id: str
    document_id: str
    title: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class QARetrievalReceipt:
    """Answer-free receipt for one deterministic question query."""

    query: str
    search_limit: int
    passages: tuple[PublicPassageObservation, ...]
    implementation: str = (
        "skillflow.skillev.benchmarks.retrieval.RetrievalIndex"
    )

    @property
    def tool_calls(self) -> int:
        return 1 + len(self.passages)

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "query": self.query,
            "search_limit": self.search_limit,
            "tool_calls": self.tool_calls,
            "operations": [
                {"name": "search", "arguments": {"query": self.query, "limit": self.search_limit}},
                *[
                    {
                        "name": "read",
                        "arguments": {"passage_id": passage.passage_id},
                    }
                    for passage in self.passages
                ],
            ],
            "passages": [passage.to_dict() for passage in self.passages],
        }

    def render_problem(self, question: str) -> str:
        """Render the public observations without any evaluator information."""

        blocks = [
            question.strip(),
            "Public retrieval observations (SkillFlow search/read):",
            f"Search query: {self.query}",
        ]
        if not self.passages:
            blocks.append("No passages were returned.")
        for passage in self.passages:
            blocks.append(
                "\n".join(
                    (
                        f"[Passage {passage.rank}]",
                        f"passage_id: {passage.passage_id}",
                        f"title: {passage.title}",
                        f"text: {passage.text}",
                    )
                )
            )
        return "\n\n".join(blocks)


def _load_retrieval_module(skillflow_source: Path) -> Any:
    source = skillflow_source.expanduser().resolve()
    if not source.is_dir():
        raise SkillFlowRetrievalError(
            f"SkillFlow source directory does not exist: {source}"
        )
    source_text = str(source)
    if source_text not in sys.path:
        # The upstream checkout is not installed into the project runtime.
        # Adding its package root is the only compatibility shim here.
        sys.path.insert(0, source_text)
    try:
        return importlib.import_module("skillev.benchmarks.retrieval")
    except ImportError as exc:
        raise SkillFlowRetrievalError(
            "SkillFlow retrieval module could not be imported"
        ) from exc


def _load_retrieval_index_class(skillflow_source: Path) -> Any:
    module = _load_retrieval_module(skillflow_source)
    try:
        return module.RetrievalIndex
    except AttributeError as exc:
        raise SkillFlowRetrievalError(
            "SkillFlow RetrievalIndex could not be imported"
        ) from exc


class SkillFlowQARetriever:
    """Read-only lifecycle wrapper around SkillFlow ``RetrievalIndex``."""

    def __init__(
        self,
        *,
        index_path: str | Path = DEFAULT_QA_RETRIEVAL_INDEX,
        skillflow_source: str | Path = DEFAULT_SKILLFLOW_SOURCE,
        search_limit: int = 5,
    ) -> None:
        if isinstance(search_limit, bool) or not isinstance(search_limit, int) or search_limit < 1:
            raise ValueError("search_limit must be a positive integer")
        self.index_path = Path(index_path).expanduser().resolve()
        self.skillflow_source = Path(skillflow_source).expanduser().resolve()
        self.search_limit = search_limit
        retrieval_index = _load_retrieval_index_class(self.skillflow_source)
        try:
            self._index = retrieval_index.open(self.index_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SkillFlowRetrievalError(
                f"SkillFlow retrieval index could not be opened: {self.index_path}"
            ) from exc
        self._closed = False
        self._cache: dict[str, QARetrievalReceipt] = {}

    def retrieve(self, question: str) -> QARetrievalReceipt:
        if self._closed:
            raise SkillFlowRetrievalError("retrieval index is closed")
        query = str(question).strip()
        if not query:
            raise ValueError("question must be non-empty")
        cached = self._cache.get(query)
        if cached is not None:
            return cached
        try:
            hits: Sequence[Any] = self._index.search(query, limit=self.search_limit)
            passages = tuple(
                self._passage_observation(hit)
                for hit in hits
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise SkillFlowRetrievalError("SkillFlow search/read failed") from exc
        receipt = QARetrievalReceipt(
            query=query,
            search_limit=self.search_limit,
            passages=passages,
        )
        self._cache[query] = receipt
        return receipt

    def _passage_observation(self, hit: Any) -> PublicPassageObservation:
        passage = self._index.read(hit.passage_id)
        return PublicPassageObservation(
            rank=int(hit.rank),
            passage_id=str(passage.passage_id),
            document_id=str(passage.document_id),
            title=str(passage.title),
            text=str(passage.text),
        )

    def close(self) -> None:
        if not self._closed:
            self._index.close()
            self._closed = True

    def __enter__(self) -> "SkillFlowQARetriever":
        if self._closed:
            raise SkillFlowRetrievalError("retrieval index is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


def augment_task_with_retrieval(
    task: TaskRecord,
    receipt: QARetrievalReceipt,
) -> TaskRecord:
    """Expose only SkillFlow's public search/read observations to the model.

    The task identity, split, and evaluator payload remain unchanged.  Only the
    model-visible question is augmented through ``receipt.render_problem``;
    the metadata addition records the public retrieval boundary without
    copying passages or accepted answers into evaluator state.
    """

    metadata = dict(task.metadata)
    metadata["public_retrieval"] = {
        "implementation": receipt.implementation,
        "query": receipt.query,
        "search_limit": receipt.search_limit,
        "tool_calls": receipt.tool_calls,
        "passage_ids": [passage.passage_id for passage in receipt.passages],
    }
    return TaskRecord(
        task_id=task.task_id,
        question=receipt.render_problem(task.question),
        ground_truth=task.ground_truth,
        split=task.split,
        metadata=metadata,
    )


def receipt_from_mapping(value: Mapping[str, Any]) -> QARetrievalReceipt:
    """Restore a cached public receipt without reopening the index."""

    raw_passages = value.get("passages", ())
    if not isinstance(raw_passages, Sequence) or isinstance(raw_passages, (str, bytes)):
        raise ValueError("retrieval receipt passages must be a sequence")
    passages = tuple(
        PublicPassageObservation(
            rank=int(item["rank"]),
            passage_id=str(item["passage_id"]),
            document_id=str(item["document_id"]),
            title=str(item["title"]),
            text=str(item["text"]),
        )
        for item in raw_passages
        if isinstance(item, Mapping)
    )
    return QARetrievalReceipt(
        query=str(value["query"]),
        search_limit=int(value["search_limit"]),
        passages=passages,
        implementation=str(
            value.get(
                "implementation",
                "skillflow.skillev.benchmarks.retrieval.RetrievalIndex",
            )
        ),
    )


__all__ = [
    "DEFAULT_QA_RETRIEVAL_INDEX",
    "DEFAULT_SKILLFLOW_SOURCE",
    "PublicPassageObservation",
    "QARetrievalReceipt",
    "SkillFlowQARetriever",
    "SkillFlowRetrievalError",
    "augment_task_with_retrieval",
    "build_keyword_query",
    "receipt_from_mapping",
]
