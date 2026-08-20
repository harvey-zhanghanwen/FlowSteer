"""SkillFlow public-retrieval backends for the unified ToolRegistry.

This module is a thin adapter over SkillFlow
``skillev.benchmarks.retrieval.RetrievalIndex``.  It preserves the upstream
read-only ``open -> search/read`` call chain and only projects public corpus
observations.  Benchmark task IDs, accepted answers, and evaluator state are
not accepted by either tool schema.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .react_execution import ToolReactExecutionAdapter
from .qa_retrieval import (
    DEFAULT_QA_RETRIEVAL_INDEX,
    DEFAULT_SKILLFLOW_SOURCE,
    SkillFlowRetrievalError,
    _load_retrieval_index_class,
)
from .tool_runtime import (
    ActionKind,
    StructuredAction,
    ToolCapability,
    ToolRegistration,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


# DIRECT_REUSE: SkillFlow's QARetrievalEnvironment publishes one resource ID
# with the two executable action names ``search`` and ``read``.  Keeping them
# under one capability prevents the Canvas from assigning ``read`` without the
# search action that produces its opaque passage_id.
QA_RETRIEVAL_TOOL_ID = "qa-retrieval"
DEFAULT_QA_DATASET_SCOPE = ("hotpotqa", "triviaqa")


class _RetrievalIndex(Protocol):
    @property
    def manifest(self) -> object:
        ...

    def search(self, query: str, *, limit: int) -> Sequence[object]:
        ...

    def read(self, passage_id: str) -> object:
        ...

    def close(self) -> None:
        ...


class _ThreadAffineRetrievalWorker:
    """Necessary adapter for SkillFlow SQLite's connection-thread affinity."""

    def __init__(self, retrieval_index_class: object, index_path: Path) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="flowsteer-qa-retrieval",
        )
        try:
            self._index = self._executor.submit(
                getattr(retrieval_index_class, "open"), index_path
            ).result()
            self.manifest = self._executor.submit(
                lambda: self._index.manifest
            ).result()
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._closed = False

    async def search(self, query: str, *, limit: int) -> Sequence[object]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._index.search(query, limit=limit),
        )

    async def read(self, passage_id: str) -> object:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._index.read(passage_id),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._index.close).result()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"retrieval manifest {field_name} must be non-empty text")
    return value.strip()


def _index_identity(index: _RetrievalIndex) -> dict[str, object]:
    """Project the frozen public-corpus identity without its content digest."""

    manifest = index.manifest
    return {
        "source": _required_text(
            getattr(manifest, "corpus_name", None), field_name="corpus_name"
        ),
        "corpus_version": _required_text(
            getattr(manifest, "corpus_version", None),
            field_name="corpus_version",
        ),
        "index_id": _required_text(
            getattr(manifest, "index_id", None), field_name="index_id"
        ),
        "index_format": _required_text(
            getattr(manifest, "format", None), field_name="format"
        ),
        "retrieval_backend": _required_text(
            getattr(manifest, "retrieval_backend", None),
            field_name="retrieval_backend",
        ),
    }


def _validate_action(request: ToolRequest, expected_action: str) -> None:
    if request.action != expected_action:
        raise ValueError(
            f"{expected_action} backend received incompatible action {request.action!r}"
        )


@dataclass(frozen=True, slots=True)
class QASearchToolBackend:
    """Execute SkillFlow ``RetrievalIndex.search`` without evaluator access."""

    index: _RetrievalIndex
    index_identity: Mapping[str, object]

    async def invoke(self, request: ToolRequest) -> ToolResult:
        # SkillFlow opens SQLite with its default thread affinity.  Keeping the
        # call on the event-loop thread matches QARetrievalEnvironment.execute
        # and avoids moving the connection through ``asyncio.to_thread``.
        _validate_action(request, "search")
        if set(request.arguments) != {"query", "limit"}:
            raise ValueError("search arguments must contain exactly query and limit")
        query = request.arguments["query"]
        limit = request.arguments["limit"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be non-empty text")
        if type(limit) is not int or limit < 1:
            raise ValueError("search limit must be a positive integer")

        raw_hits = self.index.search(query, limit=limit)
        hits = await raw_hits if inspect.isawaitable(raw_hits) else raw_hits
        public_hits = [
            {
                "passage_id": str(getattr(hit, "passage_id")),
                "document_id": str(getattr(hit, "document_id")),
                "title": str(getattr(hit, "title")),
                "snippet": str(getattr(hit, "snippet")),
                "rank": int(getattr(hit, "rank")),
            }
            for hit in hits
        ]
        return ToolResult(
            {
                "operation": "search",
                "retrieval_index": dict(self.index_identity),
                "query": query,
                "top_k": limit,
                "passage_ids": [hit["passage_id"] for hit in public_hits],
                "hits": public_hits,
            }
        )


@dataclass(frozen=True, slots=True)
class QAReadToolBackend:
    """Execute SkillFlow ``RetrievalIndex.read`` and return one public passage."""

    index: _RetrievalIndex
    index_identity: Mapping[str, object]

    async def invoke(self, request: ToolRequest) -> ToolResult:
        # See QASearchToolBackend.invoke for the upstream SQLite thread
        # affinity preserved by this async adapter boundary.
        _validate_action(request, "read")
        if set(request.arguments) != {"passage_id"}:
            raise ValueError("read arguments must contain exactly passage_id")
        passage_id = request.arguments["passage_id"]
        if not isinstance(passage_id, str) or not passage_id.strip():
            raise ValueError("read passage_id must be non-empty text")

        raw_passage = self.index.read(passage_id)
        passage = (
            await raw_passage if inspect.isawaitable(raw_passage) else raw_passage
        )
        public_passage = {
            "passage_id": str(getattr(passage, "passage_id")),
            "document_id": str(getattr(passage, "document_id")),
            "title": str(getattr(passage, "title")),
            "text": str(getattr(passage, "text")),
        }
        return ToolResult(
            {
                "operation": "read",
                "retrieval_index": dict(self.index_identity),
                "passage_id": public_passage["passage_id"],
                "passage": public_passage,
            }
        )


@dataclass(frozen=True, slots=True)
class QARetrievalToolBackend:
    """Dispatch SkillFlow's unified QA retrieval action domain."""

    index: _RetrievalIndex
    index_identity: Mapping[str, object]

    async def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action == "search":
            return await QASearchToolBackend(
                self.index,
                self.index_identity,
            ).invoke(request)
        if request.action == "read":
            return await QAReadToolBackend(
                self.index,
                self.index_identity,
            ).invoke(request)
        raise ValueError(
            f"retrieval backend received unsupported action {request.action!r}"
        )


class QARetrievalReactExecutionAdapter(ToolReactExecutionAdapter):
    """Bounded QA retrieval with canonical search-to-read admission."""

    def _tool_action_error(
        self,
        *,
        action: StructuredAction,
        observations: list[Mapping[str, object]],
    ) -> str | None:
        if (
            action.kind is not ActionKind.TOOL
            or action.resource_id != QA_RETRIEVAL_TOOL_ID
            or action.name != "read"
        ):
            return None
        arguments = action.arguments
        if not isinstance(arguments, dict):
            return None
        passage_id = arguments.get("passage_id")
        if not isinstance(passage_id, str) or not passage_id.strip():
            return None

        admitted_passage_ids: set[str] = set()
        for observation in observations:
            if observation.get("observation_status") != "success":
                continue
            result = observation.get("result")
            if not isinstance(result, Mapping) or result.get("operation") != "search":
                continue
            raw_ids = result.get("passage_ids")
            if isinstance(raw_ids, list):
                admitted_passage_ids.update(
                    value for value in raw_ids if isinstance(value, str) and value
                )
            raw_hits = result.get("hits")
            if isinstance(raw_hits, list):
                for hit in raw_hits:
                    if isinstance(hit, Mapping) and isinstance(
                        hit.get("passage_id"), str
                    ):
                        admitted_passage_ids.add(str(hit["passage_id"]))
        if not admitted_passage_ids:
            return "qa_read_requires_successful_search"
        if passage_id not in admitted_passage_ids:
            return "qa_read_passage_id_not_from_search"
        return None


def build_qa_tool_registry(
    index: _RetrievalIndex,
    *,
    dataset_scope: Sequence[str] = DEFAULT_QA_DATASET_SCOPE,
    timeout_seconds: float = 10.0,
) -> ToolRegistry:
    """Register explicit search/read capabilities over an open frozen index.

    The caller owns ``index``.  Use :func:`open_qa_tool_registry` when this
    adapter should own and close the SkillFlow index connection.
    """

    if isinstance(dataset_scope, (str, bytes)):
        raise TypeError("dataset_scope must be a sequence of dataset IDs")
    scope = tuple(dataset_scope)
    identity = _index_identity(index)
    version = str(identity["index_id"])
    identity_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source",
            "corpus_version",
            "index_id",
            "index_format",
            "retrieval_backend",
        ],
        "properties": {
            "source": {"type": "string"},
            "corpus_version": {"type": "string"},
            "index_id": {"type": "string"},
            "index_format": {"type": "string"},
            "retrieval_backend": {"type": "string"},
        },
    }
    search_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
    }
    search_input_schema["properties"]["query"]["description"] = (
        "A focused query for the missing public fact or relation."
    )
    search_input_schema["properties"]["limit"]["description"] = (
        "The positive number of ranked public passages to return."
    )
    read_input_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["passage_id"],
        "properties": {
            "passage_id": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "The exact canonical passage_id returned by a successful "
                    "search action in this execution; a title or document_id "
                    "is not a passage_id."
                ),
            },
        },
    }
    retrieval_capability = ToolCapability(
        tool_id=QA_RETRIEVAL_TOOL_ID,
        dataset_scope=scope,
        action_schemas={
            "search": search_input_schema,
            "read": read_input_schema,
        },
        input_schema={
            "oneOf": [search_input_schema, read_input_schema],
        },
        output_schema={
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "retrieval_index",
                        "query",
                        "top_k",
                        "passage_ids",
                        "hits",
                    ],
                    "properties": {
                        "operation": {"const": "search"},
                        "retrieval_index": identity_schema,
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "passage_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "hits": {"type": "array", "items": {"type": "object"}},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "operation",
                        "retrieval_index",
                        "passage_id",
                        "passage",
                    ],
                    "properties": {
                        "operation": {"const": "read"},
                        "retrieval_index": identity_schema,
                        "passage_id": {"type": "string"},
                        "passage": {"type": "object"},
                    },
                },
            ],
        },
        side_effect="none",
        timeout_seconds=timeout_seconds,
        version=version,
    )
    return ToolRegistry(
        (
            ToolRegistration(
                QA_RETRIEVAL_TOOL_ID,
                QARetrievalToolBackend(index, identity),
                retrieval_capability,
            ),
        )
    )


@dataclass(slots=True)
class OpenQAToolRegistry:
    """Owned SkillFlow index connection plus its registered Tool resources."""

    registry: ToolRegistry
    retrieval_index_identity: Mapping[str, object]
    _index: _RetrievalIndex = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if not self._closed:
            self._index.close()
            self._closed = True

    def __enter__(self) -> "OpenQAToolRegistry":
        if self._closed:
            raise RuntimeError("QA retrieval ToolRegistry is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def open_qa_tool_registry(
    *,
    index_path: str | Path = DEFAULT_QA_RETRIEVAL_INDEX,
    skillflow_source: str | Path = DEFAULT_SKILLFLOW_SOURCE,
    dataset_scope: Sequence[str] = DEFAULT_QA_DATASET_SCOPE,
    timeout_seconds: float = 10.0,
) -> OpenQAToolRegistry:
    """Open SkillFlow's immutable index and register search/read resources."""

    retrieval_index_class = _load_retrieval_index_class(Path(skillflow_source))
    try:
        index = _ThreadAffineRetrievalWorker(
            retrieval_index_class,
            Path(index_path),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SkillFlowRetrievalError(
            f"SkillFlow retrieval index could not be opened: {Path(index_path)}"
        ) from exc
    try:
        registry = build_qa_tool_registry(
            index,
            dataset_scope=dataset_scope,
            timeout_seconds=timeout_seconds,
        )
        identity = _index_identity(index)
    except BaseException:
        index.close()
        raise
    return OpenQAToolRegistry(registry, identity, index)


__all__ = [
    "DEFAULT_QA_DATASET_SCOPE",
    "OpenQAToolRegistry",
    "QARetrievalReactExecutionAdapter",
    "QARetrievalToolBackend",
    "QAReadToolBackend",
    "QA_RETRIEVAL_TOOL_ID",
    "QASearchToolBackend",
    "build_qa_tool_registry",
    "open_qa_tool_registry",
]
